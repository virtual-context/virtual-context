"""Core tool loop: VC paging tool definitions, execution, and synchronous tool loop.

Extracted from proxy/server.py so that the engine, benchmark, and CLI
can use VC paging tools without going through the HTTP proxy.

Provider adapters live in provider_adapters.py and are re-exported here
for existing importers.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

from ..types import ToolCallRecord, ToolLoopResult

# Re-exported for existing importers
from .provider_adapters import (  # noqa: F401
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAICodexAdapter,
    ProviderAdapter,
    get_adapter,
)

if TYPE_CHECKING:
    from ..engine import VirtualContextEngine

logger = logging.getLogger(__name__)
VC_FIND_QUOTE_MAX_RESULTS = 20
_INTENT_CONTEXT_MAX_CHARS = 2000
_REMINDER_CONTEXT_MAX_CHARS = 400
_CONTEXT_BLOCK_RE = re.compile(
    r"<(?:virtual-context|system-reminder)\b[^>]*>.*?</(?:virtual-context|system-reminder)>",
    re.IGNORECASE | re.DOTALL,
)
_DATE_LIKE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|[A-Z][a-z]+-\d{2}-\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4})\b",
)
_NUMERIC_LIKE_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")


@runtime_checkable
class VCToolRuntime(Protocol):
    """Mutable request-body runtime for VC tools that need in-place effects."""

    def has_restorable_stubs(self) -> bool:
        """Whether the current in-flight payload contains restoreable stubs."""

    def restore_tool_output(self, ref: str) -> dict:
        """Restore a stubbed tool output in the active payload and return a result."""

# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

VC_TOOL_NAMES: frozenset[str] = frozenset({
    "vc_expand_topic",
    "vc_find_quote",
    "vc_find_session",
    "vc_search_summaries",
    "vc_query_facts",
    "vc_recall_all",
    "vc_remember_when",
    "vc_restore_tool",
})


def vc_tool_definitions() -> list[dict]:
    """Return Anthropic-format tool definitions for VC context tools.

    This is the canonical format. Provider adapters convert as needed.
    """
    return [
        {
            "name": "vc_expand_topic",
            "description": (
                "Load the full original conversation text for a topic. "
                "Use when a topic summary covers the area you need — expanding "
                "reveals the complete conversation including details the summary "
                "may have compressed. Also use after vc_find_quote returns "
                "snippets — expand the matching tag to read surrounding context "
                "before answering. For specific facts when you don't know which "
                "topic holds them, use vc_find_quote first to locate them."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Topic tag from the context-topics list to expand.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["segments", "full"],
                        "description": (
                            "Target depth: 'segments' for individual summaries, "
                            "'full' for original conversation text."
                        ),
                    },
                    "collapse_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of topic tags to collapse back to "
                            "summary depth before expanding. Frees context "
                            "budget in the same round-trip instead of requiring "
                            "a separate tool call."
                        ),
                    },
                },
                "required": ["tag"],
            },
        },
        {
            "name": "vc_find_quote",
            "description": (
                "Find direct quote-like evidence from raw conversation turns. "
                "This tool is turn-first: use it when you need the literal "
                "place something was said, especially for names, numbers, "
                "dosages, versions, dates, and short factual details. It "
                "returns turn-backed excerpts first, then falls back to older "
                "segment-backed evidence only when the turn search is "
                "insufficient. Use mode='lookup' for normal pinpoint search. "
                "Use mode='exact_value' only when one excerpt should contain "
                "the answer verbatim as an explicit number, percentage, "
                "version, date, or count. Do not use vc_find_quote for "
                "aggregate totals or coverage across multiple components; use "
                "vc_search_summaries for that. If the question is anchored to "
                "a specific date or date range, use vc_remember_when first."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The word or phrase to search for. Use the most specific and "
                            "distinctive terms — e.g. 'magnesium glycinate' rather than "
                            "'supplement', or 'reservation 7pm' rather than 'dinner'."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["lookup", "exact_value"],
                        "description": (
                            "Retrieval mode. Use 'lookup' for a standard search. "
                            "Use 'exact_value' only when one excerpt should contain "
                            "the answer verbatim as an explicit number, percentage, "
                            "version, date, or count. Do not use this tool when "
                            "you must combine multiple components or search broad "
                            "summaries across older compacted memory."
                        ),
                    },
                },
                "required": ["query", "mode"],
            },
        },
        {
            "name": "vc_search_summaries",
            "description": (
                "Search compacted summaries, segment text, descriptions, and "
                "related stored context when you need broader topic recall "
                "rather than one literal quote. Use this for cross-topic "
                "aggregation, coverage across multiple named components, or "
                "when you know the answer may live in older compacted memory. "
                "Use mode='lookup' for a normal segment-first summary search. "
                "Use mode='aggregate_total' when the answer is one total "
                "computed from multiple named components and no single excerpt "
                "is expected to state the final total verbatim. Use mode="
                "'coverage' when the answer requires multiple complementary "
                "pieces of evidence that should be combined or compared but not "
                "arithmetically summed. If the question asks for one target or "
                "throughput across several efforts or systems, prefer "
                "'coverage' even if it says 'combined'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The summary/topic search query. Use the key entities, "
                            "technologies, or goals you want to recover."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["lookup", "aggregate_total", "coverage"],
                        "description": (
                            "Summary-search mode. Use 'lookup' for standard "
                            "summary/segment search. Use 'aggregate_total' for "
                            "questions like 'in total when combining X and Y' "
                            "where you truly need to sum different component "
                            "values. Use 'coverage' for questions like "
                            "'one target across sharding, load balancing, and "
                            "partitioning' where the answer is a shared target "
                            "or one combined goal rather than an arithmetic sum. "
                            "Use 'coverage' when multiple sides of the answer must "
                            "be covered without summing them."
                        ),
                    },
                },
                "required": ["query", "mode"],
            },
        },
        {
            "name": "vc_recall_all",
            "description": (
                "Load summaries of ALL stored conversation topics at once. "
                "Use when the user asks for a broad overview, wants to know "
                "everything discussed, needs a full summary, or asks a vague "
                "question that spans multiple topics. Returns all tag summaries "
                "within the token budget. After reviewing, use vc_expand_topic "
                "on specific tags if you need more detail."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "vc_query_facts",
            "description": (
                "Query extracted facts with structured filters. Essential for "
                "questions about events, experiences, trips, activities, or anything "
                "the user has done — each fact has a date, location, and status. "
                "Also use for counting, listing, or filtering questions like "
                "'how many X have I done', 'what projects am I leading'. "
                "Returns matching facts with count. For counting questions, omit status "
                "to get the total across all statuses in a single call. "
                "Verb is automatically expanded to include morphological variants (e.g. "
                "'led' also matches 'leads', 'visited' also matches 'traveled'). "
                "Object filter is auto-relaxed if too narrow."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Who the fact is about. Usually 'user'.",
                    },
                    "verb": {
                        "type": "string",
                        "description": "Action verb to search for (e.g. 'led', 'built', 'prefers'). Automatically expanded to include similar verbs.",
                    },
                    "object_contains": {
                        "type": "string",
                        "description": "Keyword to match in the object field.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "planned", "abandoned", "recurring"],
                        "description": "Temporal status filter. Omit for counting queries to get all statuses at once.",
                    },
                    "fact_type": {
                        "type": "string",
                        "enum": ["personal", "experience", "world"],
                        "description": "Filter by fact type. Omit to get all types.",
                    },
                },
            },
        },
        {
            "name": "vc_remember_when",
            "description": (
                "Best tool for time-based questions. Retrieves conversations "
                "and facts from a specific date range. Use FIRST when the "
                "question mentions a time period ('past three months', "
                "'last week', 'in March', 'between June and July'). Also use "
                "FIRST when the question asks what was true on a specific "
                "date or within a date window, even if the final answer is "
                "numeric. "
                "Returns both conversation excerpts and structured facts "
                "within the window. Use relative presets when they match, "
                "or between_dates with explicit YYYY-MM-DD dates for "
                "custom ranges. Use mode='lookup' for narrow fact lookups "
                "within the window, mode='state_at_time' for the state on a "
                "specific date or short date window, mode='change_over_time' "
                "for chronology/evidence retrieval across the window, and "
                "mode='summarize_over_time' for broad temporal synthesis "
                "questions: how something evolved, changed, progressed, "
                "developed, or unfolded across the window."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic/fact query to search for within a time window.",
                    },
                    "time_range": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["relative", "between_dates"]},
                            "preset": {"type": "string", "enum": [
                                "last_7_days", "last_30_days", "last_90_days", "last_180_days",
                                "last_week", "last_month",
                                "this_week", "this_month",
                            ]},
                            "start": {"type": "string", "description": "YYYY-MM-DD"},
                            "end": {"type": "string", "description": "YYYY-MM-DD"},
                        },
                        "required": ["kind"],
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 12).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "lookup", "change_over_time", "summarize_over_time", "state_at_time"],
                        "description": (
                            "Retrieval mode. Use 'lookup' for narrow fact retrieval, "
                            "'state_at_time' when the question asks what was true "
                            "on a specific date or short date window, "
                            "'change_over_time' for chronology/evidence retrieval "
                            "with multiple dated items across the range, and "
                            "'summarize_over_time' for broad temporal synthesis, "
                            "progression, and shifts across the date range. "
                            "Default is 'auto'."
                        ),
                    },
                },
                "required": ["query", "time_range"],
            },
        },
        {
            "name": "vc_restore_tool",
            "description": (
                "Restore compacted conversation history in place. Compacted turns "
                "marked with [Compacted turn N | ... | vc_restore_tool(ref=...)] "
                "contain the FULL original conversation including thinking blocks, "
                "tool calls, tool outputs, and all details that the summary omits. "
                "Call this when you need the exact original content — raw command "
                "output, file contents, detailed reasoning, per-test results, etc. "
                "The ref is in the stub text. Supports both chain_ refs (full turn "
                "chain restore) and tool_ refs (single tool output restore)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": (
                            "The ref from the compacted stub "
                            "(e.g. chain_5_abc123 or tool_abc123def)"
                        ),
                    },
                },
                "required": ["ref"],
            },
        },
    ]


def _runtime_supports_restore(tool_runtime: VCToolRuntime | None) -> bool:
    """Whether *tool_runtime* can currently advertise vc_restore_tool."""
    if tool_runtime is None:
        return False
    try:
        return bool(tool_runtime.has_restorable_stubs())
    except Exception:
        logger.debug("VC tool runtime restore capability check failed", exc_info=True)
        return False


def vc_tool_definitions_for_runtime(
    tool_runtime: VCToolRuntime | None = None,
    *,
    restore_available: bool | None = None,
) -> list[dict]:
    """Return the VC tool catalogue filtered for the active runtime."""
    defs = vc_tool_definitions()
    if restore_available is None:
        restore_available = _runtime_supports_restore(tool_runtime)
    if restore_available:
        return defs
    return [d for d in defs if d.get("name") != "vc_restore_tool"]


_SUPPRESSION_MARKER = "[Older session ("

# Tools that change the VC paging working set.  Only expand/collapse
# actually modify the loaded topic set; all other tools (find_quote,
# query_facts, recall_all, etc.) are read-only queries.  We only call
# reassemble_context() after working-set mutations to avoid re-injecting
# an unchanged system prompt on every continuation — saving significant tokens.
_WORKING_SET_TOOLS: frozenset[str] = frozenset({
    "vc_expand_topic",
})


def _vc_find_session_def() -> dict:
    """Return the Anthropic-format tool definition for vc_find_session.

    This tool is NOT included in the default tool set.  It is injected
    dynamically into the continuation request only after a vc_find_quote
    result contains suppressed older-session excerpts, giving the model
    an escape hatch to retrieve full text from a specific session.
    """
    return {
        "name": "vc_find_session",
        "description": (
            "Retrieve full conversation excerpts from a specific older session "
            "that was marked as superseded in a previous vc_find_quote result. "
            "Use this ONLY when you see '[Older session — superseded]' and "
            "need the original text to answer the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The word or phrase to search for within the session.",
                },
                "session": {
                    "type": "string",
                    "description": (
                        "The session date to search (e.g. '2023/05/25'). "
                        "Copy the date shown in the '[Older session (...)]' marker."
                    ),
                },
            },
            "required": ["query", "session"],
        },
    }


def is_vc_tool(name: str) -> bool:
    """Return True if *name* is a known VC paging tool."""
    return name in VC_TOOL_NAMES


def _flatten_text_content(content: object) -> str:
    """Best-effort flattening of provider-specific message content to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _flatten_text_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return str(content.get("text"))
        if "content" in content:
            return _flatten_text_content(content.get("content"))
        if "parts" in content:
            return _flatten_text_content(content.get("parts"))
    return ""


def _extract_last_user_text(original_request: dict) -> str:
    # Anthropic/OpenAI chat-completions style
    messages = original_request.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "user":
                continue
            text = _flatten_text_content(msg.get("content", ""))
            if text.strip():
                return text.strip()

    # OpenAI Codex responses style
    inputs = original_request.get("input")
    if isinstance(inputs, list):
        for item in reversed(inputs):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).lower() != "user":
                continue
            text = _flatten_text_content(item.get("content", ""))
            if text.strip():
                return text.strip()

    # Gemini generateContent style
    contents = original_request.get("contents")
    if isinstance(contents, list):
        for item in reversed(contents):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).lower() != "user":
                continue
            text = _flatten_text_content(item.get("parts", []))
            if text.strip():
                return text.strip()

    return ""


def _sanitize_intent_context(text: str, *, max_chars: int = _INTENT_CONTEXT_MAX_CHARS) -> str:
    """Return the user intent without injected VC context wrappers."""
    cleaned = _CONTEXT_BLOCK_RE.sub("", (text or "")).strip()
    if not cleaned:
        cleaned = (text or "").strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines()]
    for idx, line in enumerate(lines):
        if not re.match(r"(?i)^question\s*:", line):
            continue
        question_parts = [line.split(":", 1)[1].strip()]
        for extra in lines[idx + 1:]:
            if not extra:
                break
            if re.match(r"(?i)^answer\s*:", extra):
                break
            question_parts.append(extra)
        cleaned = " ".join(part for part in question_parts if part).strip()
        break

    cleaned = re.sub(r"(?i)\s*answer\s*:\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[: max_chars - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip() + "..."


def _extract_last_user_intent_text(original_request: dict) -> str:
    """Extract the last user message and strip injected VC prompt scaffolding."""
    return _sanitize_intent_context(_extract_last_user_text(original_request))


def _truncate_reminder_text(text: str, *, max_chars: int = _REMINDER_CONTEXT_MAX_CHARS) -> str:
    """Keep reminder text short even when the original question is verbose."""
    return _sanitize_intent_context(text, max_chars=max_chars)


def _tool_result_has_dates_or_numeric_values(raw_result: str) -> bool:
    """Whether a tool-result payload contains date/numeric evidence.

    This is intentionally evidence-based rather than question-based so we can
    nudge arithmetic verification without a separate classifier step.
    """

    try:
        payload = json.loads(raw_result)
    except Exception:
        payload = raw_result

    def _walk(value: object) -> bool:
        if isinstance(value, dict):
            for key, inner in value.items():
                key_lower = str(key).lower()
                if "date" in key_lower and inner:
                    return True
                if key_lower in {
                    "value",
                    "values",
                    "chosen_exact_value_candidate",
                    "exact_value_candidates",
                    "coverage_value_candidates",
                    "shared_value_candidates",
                    "aggregate_total_candidates",
                    "chosen_aggregate_total",
                    "competing_aggregate_totals",
                } and inner:
                    return True
                if _walk(inner):
                    return True
            return False
        if isinstance(value, list):
            return any(_walk(item) for item in value)
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            return bool(_DATE_LIKE_RE.search(value) or _NUMERIC_LIKE_RE.search(value))
        return False

    return _walk(payload)


def _attach_related_facts(
    engine: VirtualContextEngine, result: object, query: str,
    presented_fact_ids: set[str] | None = None,
) -> object:
    """Search facts by the find_quote query and attach matching ones.

    Uses FTS on the facts table to find only facts relevant to the
    specific query, not all facts for matched tags.

    TODO: session_date is currently resolved via a JOIN from
    fact.segment_ref → segments.metadata_json at query time.  This is
    fragile — compacted segments may lose the date, and it requires an
    extra join on every read.  Longer-term, facts should store
    session_date directly (either as a column on the facts table or via
    a dedicated sessions table that all entities FK into).  See roadmap.
    """
    if not isinstance(result, dict) or not result.get("found"):
        return result
    if not query.strip():
        return result

    try:
        _sf_limit = engine.config.search.search_facts_max_results
        facts = engine._store.search_facts(query=query, limit=_sf_limit, conversation_id=engine.config.conversation_id)
    except Exception:
        return result

    if not facts:
        return result

    # Deduplicate: skip facts already returned in a previous tool call.
    if presented_fact_ids is not None:
        facts = [f for f in facts if f.id not in presented_fact_ids]
        if not facts:
            return result

    # Build a lookup of older facts that each returned fact supersedes.
    # search_facts only returns non-superseded facts, so superseded_by
    # is always NULL.  The useful signal is the *reverse* direction:
    # "this current fact replaced these older ones."
    supersedes_map: dict[str, list[dict[str, str]]] = {}
    try:
        fact_ids = [f.id for f in facts if f.id]
        if fact_ids:
            rows = engine._store.get_superseded_facts(fact_ids)
            for row in rows:
                target_id = row["superseded_by"]
                old_entry = {
                    "subject": row["subject"],
                    "verb": row["verb"],
                    "object": row["object"],
                }
                supersedes_map.setdefault(target_id, []).append(old_entry)
    except Exception:
        pass  # non-critical enrichment

    fact_entries = []
    for f in facts:
        entry: dict[str, str] = {}
        if f.session_date:
            entry["session_date"] = f.session_date
        if f.subject:
            entry["subject"] = f.subject
        if f.verb:
            entry["verb"] = f.verb
        if f.object:
            entry["object"] = f.object
        if f.status:
            entry["status"] = f.status
        if f.when_date:
            entry["when"] = f.when_date
        if f.what:
            entry["what"] = f.what
        if f.superseded_by:
            entry["superseded_by"] = f.superseded_by
        # Attach what this fact replaced (reverse supersession)
        old_facts = supersedes_map.get(f.id, [])
        if old_facts:
            # Deduplicate by object (multiple duplicate old facts are common)
            seen_objects: set[str] = set()
            unique_old: list[dict[str, str]] = []
            for of in old_facts:
                key = of.get("object", "")
                if key not in seen_objects:
                    seen_objects.add(key)
                    unique_old.append(of)
            entry["replaces_older_facts"] = unique_old
        fact_entries.append(entry)

    result["related_facts"] = fact_entries
    result["related_facts_count"] = len(fact_entries)

    # Pull source excerpts for facts whose segments aren't already in
    # the find_quote results.  This ensures the model sees the raw
    # conversation backing each fact — even when the original query
    # didn't surface that segment.
    existing_results = result.get("results", [])
    existing_session_topics = {
        (r.get("session", ""), r.get("topic", ""))
        for r in existing_results if isinstance(r, dict)
    }
    try:
        seen_refs: set[str] = set()
        for f in facts:
            if not f.segment_ref or f.segment_ref in seen_refs:
                continue
            seen_refs.add(f.segment_ref)
            seg = engine._store.get_segment(f.segment_ref, conversation_id=engine.config.conversation_id)
            if not seg or not seg.full_text:
                continue
            # Derive session label from segment metadata
            meta = seg.metadata
            session_label = getattr(meta, "session_date", "") if meta else ""
            if (session_label, seg.primary_tag) in existing_session_topics:
                continue  # already have an excerpt from this session+topic
            # Truncate to a reasonable size
            text = seg.full_text
            if len(text) > 800:
                text = text[:800] + "..."
            existing_results.append({
                "excerpt": text,
                "topic": seg.primary_tag,
                "session": session_label,
                "source": "fact_segment",
            })
            existing_session_topics.add((session_label, seg.primary_tag))
    except Exception:
        pass  # non-critical enrichment

    # Track these facts so they won't be repeated in subsequent calls.
    if presented_fact_ids is not None:
        for f in facts:
            presented_fact_ids.add(f.id)

    return result


def _suppress_presented_segments(
    result: object, presented: set[str] | None,
) -> object:
    """Remove find_quote results from segments already shown to the reader.

    Also adds newly seen segment_refs to the presented set so subsequent
    tool calls won't repeat them either.
    """
    if presented is None or not isinstance(result, dict):
        return result
    results = result.get("results")
    if not isinstance(results, list):
        return result

    filtered = []
    deduped_count = 0
    for item in results:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        ref = item.get("segment_ref") or ""
        refs = item.get("segment_refs") or []
        # Check if this result's segment(s) were already presented.
        # Use any() for merged results: if any constituent segment was
        # already shown, the merged excerpt contains duplicate content.
        if ref and ref in presented:
            deduped_count += 1
            continue
        if refs and any(r in presented for r in refs):
            deduped_count += 1
            continue
        filtered.append(item)
        # Track newly presented segments
        if ref:
            presented.add(ref)
        for r in refs:
            presented.add(r)

    result["results"] = filtered
    if not filtered and deduped_count > 0:
        result["found"] = "already_provided"
        result["message"] = (
            f"Found {deduped_count} matching result(s) but they were already "
            "returned in a previous search. The answer is in your context — "
            "re-read the earlier search results."
        )
    elif not filtered:
        result["found"] = False
    return result


def execute_vc_tool(
    engine: VirtualContextEngine,
    name: str,
    tool_input: dict,
    *,
    intent_context: str = "",
    presented_segment_refs: set[str] | None = None,
    presented_fact_ids: set[str] | None = None,
    tool_runtime: VCToolRuntime | None = None,
) -> str:
    """Execute a VC paging tool and return a JSON result string.

    presented_segment_refs: segment refs already shown to the reader
    (from assembly or prior tool calls).  find_quote results from these
    segments are suppressed to avoid repeating the same content.
    """
    def _trim_find_quote_payload(raw: object) -> object:
        """Return only model-relevant find_quote fields for tool output."""
        if not isinstance(raw, dict):
            return raw
        if "error" in raw or raw.get("is_error") is True:
            return raw
        if "found" not in raw and "results" not in raw:
            return raw

        results = raw.get("results")
        if not isinstance(results, list):
            results = []

        found = raw.get("found")
        if not isinstance(found, bool):
            found = bool(results)

        is_current_state = raw.get("current_state_multi_session") is True
        sanitized_results: list[object] = []
        for item in results:
            if isinstance(item, dict):
                clean_item = dict(item)
                # Internal segment ids are provenance/debug data; keep them
                # out of model-facing tool outputs to reduce noise.
                clean_item.pop("segment_ref", None)
                clean_item.pop("segment_refs", None)
                # For current-state multi-session queries, truncate older
                # session excerpts so the model can't latch onto superseded
                # evidence that contradicts the newest session.
                if is_current_state:
                    rank = clean_item.get("session_recency_rank")
                    if isinstance(rank, int) and rank > 1:
                        session = clean_item.get("session", "older session")
                        clean_item["excerpt"] = (
                            f"[Older session ({session}) — superseded by newest session. "
                            f"To view full text, call vc_find_session with "
                            f"session=\"{session}\".]"
                        )
                sanitized_results.append(clean_item)
            else:
                sanitized_results.append(item)

        trimmed: dict[str, object] = {
            "found": found,
            "results": sanitized_results,
        }
        mode = raw.get("mode")
        if isinstance(mode, str) and mode.strip():
            trimmed["mode"] = mode
        if raw.get("current_state_multi_session") is True:
            trimmed["current_state_multi_session"] = True
        priority_label = raw.get("priority_label")
        if isinstance(priority_label, str) and priority_label.strip():
            trimmed["priority_label"] = priority_label
        reader_hint = raw.get("reader_hint")
        if isinstance(reader_hint, str) and reader_hint.strip():
            trimmed["reader_hint"] = reader_hint
        message = raw.get("message")
        if isinstance(message, str) and message.strip() and not found:
            trimmed["message"] = message
        value_candidates = raw.get("value_candidates")
        if isinstance(value_candidates, list) and value_candidates:
            trimmed["value_candidates"] = value_candidates
        exact_value_candidates = raw.get("exact_value_candidates")
        if isinstance(exact_value_candidates, list) and exact_value_candidates:
            trimmed["exact_value_candidates"] = exact_value_candidates
        chosen_exact_value_candidate = raw.get("chosen_exact_value_candidate")
        if isinstance(chosen_exact_value_candidate, dict) and chosen_exact_value_candidate:
            trimmed["chosen_exact_value_candidate"] = chosen_exact_value_candidate
        conflicting_exact_value_candidates = raw.get("conflicting_exact_value_candidates")
        if isinstance(conflicting_exact_value_candidates, list) and conflicting_exact_value_candidates:
            trimmed["conflicting_exact_value_candidates"] = conflicting_exact_value_candidates
        coverage_summary = raw.get("coverage_summary")
        if isinstance(coverage_summary, dict) and coverage_summary:
            trimmed["coverage_summary"] = coverage_summary
        coverage_value_candidates = raw.get("coverage_value_candidates")
        if isinstance(coverage_value_candidates, list) and coverage_value_candidates:
            trimmed["coverage_value_candidates"] = coverage_value_candidates
        chosen_preference_anchor = raw.get("chosen_preference_anchor")
        if isinstance(chosen_preference_anchor, dict) and chosen_preference_anchor:
            trimmed["chosen_preference_anchor"] = chosen_preference_anchor
        anchor_example_calculation = raw.get("anchor_example_calculation")
        if isinstance(anchor_example_calculation, dict) and anchor_example_calculation:
            trimmed["anchor_example_calculation"] = anchor_example_calculation
        shared_value_candidates = raw.get("shared_value_candidates")
        if isinstance(shared_value_candidates, list) and shared_value_candidates:
            trimmed["shared_value_candidates"] = shared_value_candidates
        aggregate_total_candidates = raw.get("aggregate_total_candidates")
        if isinstance(aggregate_total_candidates, list) and aggregate_total_candidates:
            trimmed["aggregate_total_candidates"] = aggregate_total_candidates
        if raw.get("ambiguity_detected") is True:
            trimmed["ambiguity_detected"] = True
        ambiguity_reason = raw.get("ambiguity_reason")
        if isinstance(ambiguity_reason, str) and ambiguity_reason.strip():
            trimmed["ambiguity_reason"] = ambiguity_reason
        competing_aggregate_totals = raw.get("competing_aggregate_totals")
        if isinstance(competing_aggregate_totals, list) and competing_aggregate_totals:
            trimmed["competing_aggregate_totals"] = competing_aggregate_totals
        chosen_aggregate_total = raw.get("chosen_aggregate_total")
        if isinstance(chosen_aggregate_total, dict) and chosen_aggregate_total:
            trimmed["chosen_aggregate_total"] = chosen_aggregate_total
        return trimmed

    def _trim_remember_when_payload(raw: object) -> object:
        """Return only model-relevant remember_when fields for tool output."""
        if not isinstance(raw, dict):
            return raw
        if "error" in raw or raw.get("is_error") is True:
            return raw

        mode = raw.get("mode")
        trimmed: dict[str, object] = {}
        if isinstance(mode, str) and mode.strip():
            trimmed["mode"] = mode

        found = raw.get("found")
        if isinstance(found, bool):
            trimmed["found"] = found

        range_payload = raw.get("range")
        if isinstance(range_payload, dict) and range_payload:
            trimmed["range"] = range_payload

        message = raw.get("message")
        if isinstance(message, str) and message.strip():
            trimmed["message"] = message

        if mode == "summarize_over_time":
            ordered_milestones = raw.get("ordered_milestones")
            if isinstance(ordered_milestones, list) and ordered_milestones:
                trimmed["ordered_milestones"] = ordered_milestones
                trimmed["reader_hint"] = (
                    "Use ordered_milestones as the primary chronological evidence. "
                    "Do not combine extra themes from separate milestones."
                )
                return trimmed
        if mode == "change_over_time":
            date_buckets = raw.get("date_buckets")
            if isinstance(date_buckets, list) and date_buckets:
                trimmed["date_buckets"] = date_buckets
                trimmed["reader_hint"] = (
                    "Use date_buckets as the primary chronological evidence. "
                    "Prefer directly matching results and facts from each date. "
                    "Do not infer missing themes that are not present."
                )
                return trimmed

        results = raw.get("results")
        if isinstance(results, list):
            trimmed["results"] = results
        facts_in_window = raw.get("facts_in_window")
        if isinstance(facts_in_window, list):
            trimmed["facts_in_window"] = facts_in_window

        ordered_milestones = raw.get("ordered_milestones")
        if isinstance(ordered_milestones, list) and ordered_milestones:
            trimmed["ordered_milestones"] = ordered_milestones

        target_date = raw.get("target_date")
        if isinstance(target_date, str) and target_date.strip():
            trimmed["target_date"] = target_date
        chosen_state = raw.get("chosen_state")
        if isinstance(chosen_state, dict) and chosen_state:
            trimmed["chosen_state"] = chosen_state
        conflicting_candidates = raw.get("conflicting_candidates")
        if isinstance(conflicting_candidates, list) and conflicting_candidates:
            trimmed["conflicting_candidates"] = conflicting_candidates
        state_anchor = raw.get("state_anchor")
        if isinstance(state_anchor, dict) and state_anchor:
            trimmed["state_anchor"] = state_anchor
        return trimmed

    try:
        if name == "vc_expand_topic":
            # Collapse specified tags first to free budget
            collapse_results = []
            for ctag in tool_input.get("collapse_tags") or []:
                cr = engine.collapse_topic(tag=ctag, depth="summary")
                if cr.get("tokens_freed", 0) > 0:
                    collapse_results.append(cr)

            result = engine.expand_topic(
                tag=tool_input.get("tag", ""),
                depth=tool_input.get("depth", "full"),
            )

            # Merge collapse info into expand result
            if collapse_results:
                result["collapsed"] = collapse_results
                result["total_tokens_freed"] = sum(
                    cr.get("tokens_freed", 0) for cr in collapse_results
                )

            # Surface linked tool outputs as a recovery hint (Phase 8).
            # Do NOT inline full raw content — just note their existence
            # so the model knows to use vc_find_quote for details.
            if "error" not in result:
                _expand_tag = tool_input.get("tag", "")
                _conv_id = engine.config.conversation_id
                try:
                    _segments = engine._store.get_segments_by_tags(
                        tags=[_expand_tag], min_overlap=1, limit=500,
                        conversation_id=_conv_id or None,
                    )
                    _linked_refs: list[str] = []
                    for _seg in _segments:
                        _refs = engine._store.get_tool_outputs_for_segment(
                            _conv_id, _seg.ref,
                        )
                        for _r in _refs:
                            if _r not in _linked_refs:
                                _linked_refs.append(_r)
                    if _linked_refs:
                        result["linked_tool_outputs"] = len(_linked_refs)
                        result["tool_output_hint"] = (
                            f"This topic has {len(_linked_refs)} linked tool "
                            f"output(s). Use vc_find_quote(query) to search "
                            f"their content."
                        )
                except Exception:
                    # Non-critical — don't break expand on linkage errors
                    logger.debug(
                        "Failed to query tool output links for tag %s",
                        _expand_tag, exc_info=True,
                    )

        elif name == "vc_find_quote":
            fq_query = tool_input.get("query", "")
            fq_mode = tool_input.get("mode", "lookup")
            _fq_max = engine.config.search.find_quote_max_results
            result = engine.find_quote(
                query=fq_query,
                max_results=_fq_max,
                intent_context=intent_context,
                mode=fq_mode,
            )
            result = _suppress_presented_segments(result, presented_segment_refs)
            result = _trim_find_quote_payload(result)
            result = _attach_related_facts(engine, result, fq_query, presented_fact_ids)
        elif name == "vc_search_summaries":
            fq_query = tool_input.get("query", "")
            fq_mode = tool_input.get("mode", "lookup")
            _fq_max = engine.config.search.find_quote_max_results
            result = engine.search_summaries(
                query=fq_query,
                max_results=_fq_max,
                intent_context=intent_context,
                mode=fq_mode,
            )
            result = _suppress_presented_segments(result, presented_segment_refs)
            result = _trim_find_quote_payload(result)
            result = _attach_related_facts(engine, result, fq_query, presented_fact_ids)
        elif name == "vc_find_session":
            _fq_max = engine.config.search.find_quote_max_results
            result = engine.search_summaries(
                query=tool_input.get("query", ""),
                max_results=_fq_max,
                intent_context=intent_context,
                session_filter=tool_input.get("session", ""),
                mode="lookup",
            )
            result = _suppress_presented_segments(result, presented_segment_refs)
            result = _trim_find_quote_payload(result)
        elif name == "vc_recall_all":
            result = engine.recall_all()
        elif name == "vc_query_facts":
            meta = engine.query_facts(
                subject=tool_input.get("subject"),
                verb=tool_input.get("verb"),
                object_contains=tool_input.get("object_contains"),
                status=tool_input.get("status"),
                fact_type=tool_input.get("fact_type"),
                _return_meta=True,
                _intent_context=intent_context,
            )
            facts = meta["facts"]
            result = {
                "count": len(facts),
                "facts": [
                    {
                        "subject": f.subject,
                        "verb": f.verb,
                        "object": f.object,
                        "status": f.status,
                        "fact_type": f.fact_type,
                        "what": f.what,
                        "who": f.who,
                        "when": f.when_date,
                        "where": f.where,
                        "why": f.why,
                        "conversation_id": f.conversation_id,
                        "tags": f.tags,
                    }
                    for f in facts
                ],
            }
            # Status breakdown from the filtered results
            status_counts: dict[str, int] = {}
            for f in facts:
                s = f.status or "unknown"
                status_counts[s] = status_counts.get(s, 0) + 1
            if status_counts:
                result["by_status"] = status_counts
            # When a status filter was used, show the total across ALL
            # statuses so the reader can see the grand total without
            # making separate per-status calls.
            if meta.get("total_all_statuses") is not None:
                result["total_all_statuses"] = meta["total_all_statuses"]
                result["all_statuses_breakdown"] = meta["all_statuses"]
            # Annotate so the reader knows what broadening happened
            notes = []
            if meta.get("expanded_verbs"):
                notes.append(f"verb expanded to match: {meta['expanded_verbs']}")
            if meta.get("semantic_note"):
                notes.append(meta["semantic_note"])
            if notes:
                result["search_notes"] = "; ".join(notes)
            # Include linked facts when graph_links is enabled
            if meta.get("linked_facts"):
                result["linked_facts"] = [
                    {
                        "subject": lf.fact.subject,
                        "verb": lf.fact.verb,
                        "object": lf.fact.object,
                        "status": lf.fact.status,
                        "what": lf.fact.what,
                        "_linked_via": f"{lf.relation_type} from '{lf.linked_from_fact_id[:8]}'",
                        "_confidence": lf.confidence,
                    }
                    for lf in meta["linked_facts"]
                ]
        elif name == "vc_remember_when":
            max_results = tool_input.get("max_results")
            result = engine.remember_when(
                query=tool_input.get("query", ""),
                time_range=tool_input.get("time_range", {}),
                max_results=max_results,
                mode=tool_input.get("mode", "auto"),
                intent_context=intent_context,
            )
            result = _trim_remember_when_payload(result)
        elif name == "vc_restore_tool":
            ref = tool_input.get("ref", "")
            if not _runtime_supports_restore(tool_runtime):
                result = {"error": "vc_restore_tool unavailable: no restorable payload runtime"}
            else:
                result = tool_runtime.restore_tool_output(ref)
        else:
            result = {"error": f"unknown VC tool: {name}"}
        return json.dumps(result)
    except Exception as e:
        logger.error("VC tool %s raised %s: %s", name, type(e).__name__, e, exc_info=True)
        return json.dumps({"is_error": True, "content": str(e)})


# ---------------------------------------------------------------------------
# Synchronous tool loop
# ---------------------------------------------------------------------------

_DEFAULT_MAX_LOOPS = 10
_REDUNDANT_SEARCH_THRESHOLD = 3  # force-stop after this many all-found find_quote rounds
_EMPTY_STREAK_HINT_THRESHOLD = 3  # suggest strategy change after this many empty results


def _is_empty_result(result_json: str) -> bool:
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return False
    if data.get("found") is False:
        return True
    if data.get("count") == 0 or data.get("facts") == []:
        return True
    results = data.get("results")
    if isinstance(results, list) and len(results) == 0:
        return True
    return False


def _tool_result_grounding_hint(tool_name: str, tool_input: dict) -> str:
    """Return an evidence-grounding reminder for tool results when helpful."""
    if tool_name == "vc_find_quote" and tool_input.get("mode", "lookup") == "lookup":
        return (
            "\n\n[REMINDER] Answer from the most directly matching quoted evidence "
            "in these tool results. Prefer exact quoted statements over broader "
            "paraphrase, and do not combine details from separate events or sessions."
        )
    return ""


_STRATEGY_HINTS: dict[str, str] = {
    "vc_query_facts": (
        "[HINT] vc_query_facts has returned no results {n} times in a row. "
        "Try a different approach: use vc_find_quote(query) to search raw "
        "conversation text, or vc_expand_topic(tag) to browse a relevant "
        "topic directly."
    ),
    "vc_find_quote": (
        "[HINT] vc_find_quote has returned no results {n} times in a row. "
        "Try a different approach: use vc_search_summaries for broader "
        "compacted-memory search, vc_query_facts to search structured facts, "
        "or try different search terms."
    ),
    "vc_search_summaries": (
        "[HINT] vc_search_summaries has returned no results {n} times in a "
        "row. Try a different approach: use vc_find_quote(query) for raw "
        "turn-level evidence, vc_query_facts for structured facts, or "
        "broaden the topic search."
    ),
    "vc_remember_when": (
        "[HINT] vc_remember_when has returned no results {n} times in a row. "
        "Try a different approach: use vc_find_quote(query) to search raw "
        "text, or broaden your time range."
    ),
}
_DEFAULT_STRATEGY_HINT = (
    "[HINT] This tool has returned no results {n} times in a row. "
    "Try a different tool or approach."
)


def _parse_sse_response(text: str) -> dict:
    last_response: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload_raw = line[6:]
        if not payload_raw or payload_raw == "[DONE]":
            continue
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type", "")
        if event_type in {"response.created", "response.in_progress"}:
            resp = payload.get("response")
            if isinstance(resp, dict):
                last_response = resp
        elif event_type in {"response.completed", "response.failed"}:
            resp = payload.get("response")
            if isinstance(resp, dict):
                return resp
        elif event_type == "error":
            return {"error": payload.get("error", payload)}
    return last_response or {}


def _parse_provider_http_response(resp: httpx.Response) -> dict:
    content_type = (resp.headers.get("content-type", "") or "").lower()
    if "text/event-stream" in content_type:
        return _parse_sse_response(resp.text)
    try:
        return resp.json()
    except json.JSONDecodeError:
        return _parse_sse_response(resp.text)


def _send_with_retry(
    client: httpx.Client,
    url: str,
    headers: dict,
    body: dict,
    *,
    label: str = "continuation",
) -> tuple[httpx.Response, float]:
    """Send an HTTP POST with retry on 429/503. Returns (response, duration_ms)."""
    t0 = time.monotonic()
    resp = client.post(url, headers=headers, json=body)
    for _retry in range(3):
        if resp.status_code < 300 or resp.status_code not in (429, 503):
            break
        wait = 2 ** (_retry + 1)
        logger.warning(
            "%s got HTTP %d, retrying in %ds (%d/3)",
            label, resp.status_code, wait, _retry + 1,
        )
        time.sleep(wait)
        resp = client.post(url, headers=headers, json=body)
    return resp, round((time.monotonic() - t0) * 1000, 1)


def _execute_pending_tools(
    engine,
    vc_tools: list[dict],
    adapter: ProviderAdapter,
    result: ToolLoopResult,
    intent_context: str,
    presented_refs: set[str],
    presented_facts: set[str],
    tool_runtime: VCToolRuntime | None = None,
) -> list[dict]:
    """Execute pending VC tool calls, record them, return provider-formatted results."""
    tool_results: list[dict] = []
    for tc in vc_tools:
        t0 = time.monotonic()
        r_str = execute_vc_tool(
            engine, tc["name"], tc["input"],
            intent_context=intent_context,
            presented_segment_refs=presented_refs,
            presented_fact_ids=presented_facts,
            tool_runtime=tool_runtime,
        )
        dur = round((time.monotonic() - t0) * 1000, 1)
        result.tool_calls.append(ToolCallRecord(
            tool_name=tc["name"],
            tool_input=tc["input"],
            result_json=r_str,
            duration_ms=dur,
        ))
        tool_results.append(
            adapter.build_tool_result(tc["id"], tc["name"], r_str)
        )
    return tool_results


def _extract_call_ids(tool_results: list[dict]) -> set[str]:
    """Extract tool call IDs from provider-formatted tool results."""
    ids: set[str] = set()
    for tr in tool_results:
        cid = tr.get("call_id") or tr.get("tool_call_id") or ""
        if cid:
            ids.add(cid)
        if isinstance(tr.get("content"), list):
            for block in tr["content"]:
                if isinstance(block, dict) and block.get("tool_use_id"):
                    ids.add(block["tool_use_id"])
    return ids


def _force_text_response(
    client: httpx.Client,
    url: str,
    headers: dict,
    cont_body: dict | None,
    original_request: dict,
    current_response: dict,
    tool_results: list[dict],
    adapter: ProviderAdapter,
    result: ToolLoopResult,
    engine,
    model: str,
    *,
    label: str = "forced_final",
) -> None:
    """Build a tool-stripped continuation with a nudge and send it. Updates *result* in-place."""
    cont_body = adapter.build_continuation(
        cont_body, original_request, current_response, tool_results,
    )
    adapter.compress_previous_results(cont_body, _extract_call_ids(tool_results))
    adapter.strip_tools(cont_body)

    # Minimize thinking budget on forced answer (Gemini thinking models)
    gen_cfg = cont_body.get("generationConfig")
    if isinstance(gen_cfg, dict) and "thinkingConfig" in gen_cfg:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 128}

    # Append user nudge for all providers
    nudge = (
        "You have gathered enough information. "
        "Answer the original question now in 1-2 sentences."
    )
    contents = cont_body.get("contents")
    if isinstance(contents, list):
        contents.append({"role": "user", "parts": [{"text": nudge}]})
    messages = cont_body.get("messages")
    if isinstance(messages, list):
        messages.append({"role": "user", "content": nudge})

    result.continuation_count += 1
    result.raw_requests.append(copy.deepcopy(cont_body))

    resp, dur = _send_with_retry(client, url, headers, cont_body, label=label)
    if resp.status_code < 300:
        data = _parse_provider_http_response(resp)
        result.raw_responses.append(data)
        fi, fo = adapter.extract_usage(data)
        result.input_tokens += fi
        result.output_tokens += fo
        if hasattr(engine, '_telemetry') and engine._telemetry:
            engine._telemetry.log(
                "tool_loop", model, fi, fo,
                duration_ms=dur, detail=label,
            )
        result.text += adapter.extract_text(data)
        result.stop_reason = adapter.get_stop_reason(data)
    else:
        logger.error("%s failed after retries: HTTP %d", label, resp.status_code)
        result.stop_reason = "error"


def run_tool_loop(
    engine: VirtualContextEngine,
    initial_response: dict,
    original_request: dict,
    adapter: ProviderAdapter,
    *,
    url: str = "",
    max_loops: int = _DEFAULT_MAX_LOOPS,
    extra_headers: dict | None = None,
    tool_runtime: VCToolRuntime | None = None,
) -> ToolLoopResult:
    """Run a synchronous non-streaming tool loop.

    Given an initial LLM response that contains VC tool calls,
    execute the tools, send continuation requests, and return the final
    text result.  Works with any provider via the adapter pattern.

    Parameters
    ----------
    engine : VirtualContextEngine
        Engine instance for tool execution and context reassembly.
    initial_response : dict
        The first LLM response body (parsed JSON).
    original_request : dict
        The original request body (for building continuations).
    adapter : ProviderAdapter
        Provider-specific adapter for format conversion.
    url : str
        API endpoint URL (if empty, uses ``adapter.get_url()``).
    max_loops : int
        Maximum continuation rounds (default 10).

    Returns
    -------
    ToolLoopResult
        Final text, tool call records, and usage metrics.
    """
    if not url:
        url = adapter.get_url()

    # Anti-repetition: track segments returned by tool calls so the same
    # segment isn't repeated across multiple find_quote rounds.
    # Starts empty — assembly summaries are a *different view* (compressed)
    # of the data, so tool results from those segments are NOT suppressed.
    presented_refs: set[str] = set()
    presented_facts: set[str] = set()

    result = ToolLoopResult()
    result.raw_responses.append(initial_response)

    # Accumulate usage from initial response
    input_toks, output_toks = adapter.extract_usage(initial_response)
    result.input_tokens += input_toks
    result.output_tokens += output_toks

    model = original_request.get("model", "unknown")
    if hasattr(engine, '_telemetry') and engine._telemetry:
        engine._telemetry.log(
            "tool_loop", model,
            input_toks, output_toks,
            duration_ms=0.0,
            detail="initial",
        )

    # Collect text from the initial response
    result.text += adapter.extract_text(initial_response)

    stop_reason = adapter.get_stop_reason(initial_response)

    # Identify tool calls
    all_tool_calls = adapter.extract_tool_calls(initial_response)
    vc_tools = [tc for tc in all_tool_calls if is_vc_tool(tc["name"])]
    non_vc_tools = [tc for tc in all_tool_calls if not is_vc_tool(tc["name"])]

    # BAIL: no VC tools — just return text
    if not vc_tools:
        result.stop_reason = stop_reason
        return result

    # BAIL: mixed VC + non-VC tools
    if non_vc_tools:
        logger.warning("Mixed VC + non-VC tools in response — returning as-is")
        result.stop_reason = stop_reason
        return result

    headers = adapter.get_headers()
    if extra_headers:
        headers.update(extra_headers)
    cont_body: dict | None = None
    current_response = initial_response
    intent_context = _extract_last_user_intent_text(original_request)
    _find_session_injected = False
    with httpx.Client(timeout=300.0) as client:
        for loop_i in range(max_loops):
            # Execute VC tools
            tool_results: list[dict] = []
            for tc in vc_tools:
                t0 = time.monotonic()
                result_str = execute_vc_tool(
                    engine,
                    tc["name"],
                    tc["input"],
                    intent_context=intent_context,
                    presented_segment_refs=presented_refs,
                    presented_fact_ids=presented_facts,
                    tool_runtime=tool_runtime,
                )
                duration_ms = round((time.monotonic() - t0) * 1000, 1)

                result.tool_calls.append(ToolCallRecord(
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    result_json=result_str,
                    duration_ms=duration_ms,
                ))

                # Check for consecutive empty results from the same tool
                strategy_hint = ""
                if _is_empty_result(result_str):
                    streak = 0
                    for prev in reversed(result.tool_calls):
                        if prev.tool_name == tc["name"] and _is_empty_result(prev.result_json):
                            streak += 1
                        else:
                            break
                    if streak >= _EMPTY_STREAK_HINT_THRESHOLD:
                        tpl = _STRATEGY_HINTS.get(tc["name"], _DEFAULT_STRATEGY_HINT)
                        strategy_hint = "\n\n" + tpl.format(n=streak)

                verification_hint = ""
                if _tool_result_has_dates_or_numeric_values(result_str):
                    verification_hint = (
                        "\n\n[REMINDER] If your answer depends on dates or numbers in "
                        "these tool results, verify the arithmetic and units directly "
                        "from the evidence before responding."
                    )
                grounding_hint = _tool_result_grounding_hint(tc["name"], tc["input"])

                tool_results.append(
                    adapter.build_tool_result(
                        tc["id"], tc["name"],
                        result_str + strategy_hint + verification_hint + grounding_hint,
                    )
                )

            # Re-assemble context only when working-set tools were used
            # (expand, collapse, find_quote, etc.).  Pure query_facts rounds
            # don't change the working set, so re-injecting just wastes tokens.
            used_names = {tc["name"] for tc in vc_tools}
            needs_reassemble = bool(used_names & _WORKING_SET_TOOLS)

            new_prepend = engine.reassemble_context() if needs_reassemble else ""

            # Build continuation request
            cont_body = adapter.build_continuation(
                cont_body, original_request, current_response, tool_results,
            )

            # Compress previous rounds' tool results to avoid repetition
            adapter.compress_previous_results(cont_body, _extract_call_ids(tool_results))

            # Inject updated context into system prompt
            if new_prepend:
                adapter.inject_context(cont_body, new_prepend)

            # Dynamically inject vc_find_session tool if suppression occurred
            if not _find_session_injected:
                recent_results = result.tool_calls[-len(vc_tools):]
                if any(_SUPPRESSION_MARKER in r.result_json for r in recent_results):
                    adapter.add_tool_defs(cont_body, [_vc_find_session_def()])
                    _find_session_injected = True

            # After the first round, downgrade tool_choice from "any" to
            # "auto" so the model can answer with text when it has enough
            # information instead of being forced to make another tool call.
            if loop_i == 0:
                adapter.relax_tool_choice(cont_body)

            result.continuation_count += 1

            # Capture the continuation request body (deep copy to freeze state)
            result.raw_requests.append(copy.deepcopy(cont_body))

            # Send continuation (retry on 429/503 with backoff)
            resp, _cont_dur = _send_with_retry(
                client, url, headers, cont_body,
                label=f"continuation_{result.continuation_count}",
            )
            if resp.status_code >= 300:
                logger.error(
                    "Continuation %d failed: HTTP %d",
                    result.continuation_count, resp.status_code,
                )
                result.stop_reason = "error"
                break

            cont_data = _parse_provider_http_response(resp)
            current_response = cont_data
            result.raw_responses.append(cont_data)

            # Accumulate usage
            input_toks, output_toks = adapter.extract_usage(cont_data)
            result.input_tokens += input_toks
            result.output_tokens += output_toks

            if hasattr(engine, '_telemetry') and engine._telemetry:
                engine._telemetry.log(
                    "tool_loop", model,
                    input_toks, output_toks,
                    duration_ms=_cont_dur,
                    detail=f"round_{loop_i + 1}",
                )

            # Extract text
            result.text += adapter.extract_text(cont_data)

            stop_reason = adapter.get_stop_reason(cont_data)

            # Check for more VC tool calls
            all_tool_calls = adapter.extract_tool_calls(cont_data)
            vc_tools = [tc for tc in all_tool_calls if is_vc_tool(tc["name"])]
            non_vc_in_cont = [
                tc for tc in all_tool_calls if not is_vc_tool(tc["name"])
            ]

            if (adapter.is_tool_use_stop(cont_data)
                    and vc_tools and not non_vc_in_cont):
                # Detect redundant search loops: if the last N rounds
                # were all vc_find_quote calls that returned found:true,
                # the model already has the answer — force a text response.
                if loop_i >= _REDUNDANT_SEARCH_THRESHOLD - 1:
                    recent = result.tool_calls[-(_REDUNDANT_SEARCH_THRESHOLD * max(len(vc_tools), 1)):]
                    all_find = all(r.tool_name == "vc_find_quote" for r in recent)
                    all_found = all('"found": true' in r.result_json for r in recent)
                    if all_find and all_found and len(recent) >= _REDUNDANT_SEARCH_THRESHOLD:
                        logger.info(
                            "Redundant search loop detected after %d rounds "
                            "— executing pending tools then forcing text",
                            loop_i + 1,
                        )
                        forced_results = _execute_pending_tools(
                            engine, vc_tools, adapter, result,
                            intent_context, presented_refs, presented_facts,
                            tool_runtime=tool_runtime,
                        )
                        _force_text_response(
                            client, url, headers, cont_body,
                            original_request, current_response, forced_results,
                            adapter, result, engine, model,
                            label="forced_redundant",
                        )
                        if result.stop_reason != "error":
                            result.stop_reason = "redundant_loop"
                        break
                continue  # loop again

            # Done
            result.stop_reason = stop_reason
            break
        else:
            # Exhausted max_loops — force a text answer if none produced
            if not result.text.strip():
                logger.warning(
                    "Tool loop exhausted %d iterations with no text — "
                    "sending forced text continuation",
                    max_loops,
                )
                forced_results = _execute_pending_tools(
                    engine, vc_tools, adapter, result,
                    intent_context, presented_refs, presented_facts,
                    tool_runtime=tool_runtime,
                )
                _force_text_response(
                    client, url, headers, cont_body,
                    original_request, current_response, forced_results,
                    adapter, result, engine, model,
                    label="forced_exhausted",
                )
            else:
                result.stop_reason = stop_reason

    return result
