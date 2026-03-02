"""Core tool loop: VC paging tool definitions, execution, and synchronous tool loop.

Extracted from proxy/server.py so that the engine, benchmark, and CLI
can use VC paging tools without going through the HTTP proxy.

Provider adapters live in provider_adapters.py and are re-exported here
for backward compatibility.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import TYPE_CHECKING

import httpx

from ..types import ToolCallRecord, ToolLoopResult

# Re-export adapter classes for backward compatibility
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

# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

VC_TOOL_NAMES: frozenset[str] = frozenset({
    "vc_expand_topic",
    "vc_collapse_topic",
    "vc_find_quote",
    "vc_find_session",
    "vc_query_facts",
    "vc_recall_all",
    "vc_remember_when",
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
                },
                "required": ["tag"],
            },
        },
        {
            "name": "vc_collapse_topic",
            "description": (
                "Collapse an expanded topic back to its summary to free context "
                "budget. Use after you've retrieved what you need from an expanded "
                "topic, or to make room before expanding a different one."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Topic tag to collapse.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["summary", "none"],
                        "description": (
                            "Target depth: 'summary' for brief overview, "
                            "'none' to remove from context entirely."
                        ),
                    },
                },
                "required": ["tag"],
            },
        },
        {
            "name": "vc_find_quote",
            "description": (
                "Search the full original conversation text and truncated tool "
                "outputs for a specific word, phrase, or detail. Use this when "
                "you see '... N bytes truncated — call vc_find_quote(query) ...' "
                "in a tool result, or when the user asks about a specific fact — "
                "a name, number, dosage, recommendation, date, or decision — "
                "especially when no topic summary mentions it or you don't know "
                "which topic it falls under. This bypasses tags entirely and "
                "searches raw text, so it finds content even when it's filed "
                "under an unexpected topic. Returns short excerpts — use "
                "vc_expand_topic on a matching tag if you need more context."
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
                },
                "required": ["query"],
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
                "Query extracted facts with structured filters. Use for counting, "
                "listing, or filtering questions like 'how many X have I done', "
                "'what projects am I leading', 'things I prefer', 'what tools do I use'. "
                "Returns matching facts with count. For counting questions, omit status "
                "to get the total across all statuses in a single call. "
                "Verb is automatically expanded to include morphological variants (e.g. "
                "'led' also matches 'leads'). Object filter is auto-relaxed if too narrow."
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
                "Find memory by topic within a time window. "
                "Use for requests like 'last week', 'last month', "
                "'3 days ago', or 'between June and July'. "
                "Do not compute calendar dates yourself; prefer relative "
                "presets and let the backend resolve exact dates."
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
                        "description": (
                            "Time window selector. One of: "
                            "{kind:'relative', preset:'last_24_hours|last_7_days|last_30_days|"
                            "this_week|last_week|this_month|last_month|this_year|last_year'} "
                            "or {kind:'between_dates', start:'YYYY-MM-DD or YYYY-MM', "
                            "end:'YYYY-MM-DD or YYYY-MM'}."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5).",
                    },
                },
                "required": ["query", "time_range"],
            },
        },
    ]


_SUPPRESSION_MARKER = "[Older session ("

# Tools that change the VC paging working set.  Only expand/collapse
# actually modify the loaded topic set; all other tools (find_quote,
# query_facts, recall_all, etc.) are read-only queries.  We only call
# reassemble_context() after working-set mutations to avoid re-injecting
# an unchanged system prompt on every continuation — saving significant tokens.
_WORKING_SET_TOOLS: frozenset[str] = frozenset({
    "vc_expand_topic",
    "vc_collapse_topic",
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
    """Extract the latest user question from provider request payloads."""
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
        facts = engine._store.search_facts(query=query, limit=10)
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
        conn = engine._store._get_conn()
        fact_ids = [f.id for f in facts if f.id]
        if fact_ids:
            placeholders = ",".join("?" * len(fact_ids))
            rows = conn.execute(
                f"SELECT superseded_by, subject, verb, object FROM facts "
                f"WHERE superseded_by IN ({placeholders})",
                fact_ids,
            ).fetchall()
            for row in rows:
                target_id = row[0]
                old_entry = {
                    "subject": row[1],
                    "verb": row[2],
                    "object": row[3],
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
            seg = engine._store.get_segment(f.segment_ref)
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
            continue
        if refs and any(r in presented for r in refs):
            continue
        filtered.append(item)
        # Track newly presented segments
        if ref:
            presented.add(ref)
        for r in refs:
            presented.add(r)

    result["results"] = filtered
    if not filtered:
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
        return trimmed

    try:
        if name == "vc_expand_topic":
            result = engine.expand_topic(
                tag=tool_input.get("tag", ""),
                depth=tool_input.get("depth", "full"),
            )
        elif name == "vc_collapse_topic":
            result = engine.collapse_topic(
                tag=tool_input.get("tag", ""),
                depth=tool_input.get("depth", "summary"),
            )
        elif name == "vc_find_quote":
            fq_query = tool_input.get("query", "")
            result = engine.find_quote(
                query=fq_query,
                max_results=VC_FIND_QUOTE_MAX_RESULTS,
                intent_context=intent_context,
            )
            result = _suppress_presented_segments(result, presented_segment_refs)
            result = _trim_find_quote_payload(result)
            result = _attach_related_facts(engine, result, fq_query, presented_fact_ids)
        elif name == "vc_find_session":
            result = engine.find_quote(
                query=tool_input.get("query", ""),
                max_results=VC_FIND_QUOTE_MAX_RESULTS,
                intent_context=intent_context,
                session_filter=tool_input.get("session", ""),
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
                        "session_id": f.session_id,
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
        elif name == "vc_remember_when":
            result = engine.remember_when(
                query=tool_input.get("query", ""),
                time_range=tool_input.get("time_range", {}),
                max_results=tool_input.get("max_results", 5),
            )
        else:
            result = {"error": f"unknown VC tool: {name}"}
        return json.dumps(result)
    except Exception as e:
        logger.error("VC tool %s raised %s: %s", name, type(e).__name__, e, exc_info=True)
        return json.dumps({"is_error": True, "content": str(e)})


# ---------------------------------------------------------------------------
# Synchronous tool loop
# ---------------------------------------------------------------------------

_MAX_LOOPS = 10
_REDUNDANT_SEARCH_THRESHOLD = 3  # force-stop after this many all-found find_quote rounds
_EMPTY_STREAK_HINT_THRESHOLD = 3  # suggest strategy change after this many empty results


def _is_empty_result(result_json: str) -> bool:
    """Return True if a tool result indicates nothing was found."""
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


_STRATEGY_HINTS: dict[str, str] = {
    "vc_query_facts": (
        "[HINT] vc_query_facts has returned no results {n} times in a row. "
        "Try a different approach: use vc_find_quote(query) to search raw "
        "conversation text, or vc_expand_topic(tag) to browse a relevant "
        "topic directly."
    ),
    "vc_find_quote": (
        "[HINT] vc_find_quote has returned no results {n} times in a row. "
        "Try a different approach: use vc_query_facts to search structured "
        "facts, vc_expand_topic(tag) to browse a topic, or try broader/"
        "different search terms."
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
    """Parse SSE text and return the final `response` object."""
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
    """Parse provider HTTP response, supporting JSON and SSE payloads."""
    content_type = (resp.headers.get("content-type", "") or "").lower()
    if "text/event-stream" in content_type:
        return _parse_sse_response(resp.text)
    try:
        return resp.json()
    except json.JSONDecodeError:
        return _parse_sse_response(resp.text)


def run_tool_loop(
    engine: VirtualContextEngine,
    initial_response: dict,
    original_request: dict,
    adapter: ProviderAdapter,
    *,
    url: str = "",
    max_loops: int = _MAX_LOOPS,
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
        Maximum continuation rounds (default 5).

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
    cont_body: dict | None = None
    current_response = initial_response
    intent_context = _extract_last_user_text(original_request)
    _find_session_injected = False

    # Short reminder appended to each tool result so the model
    # doesn't lose sight of the original question during long loops.
    _question_reminder = (
        f"\n\n[REMINDER] Question: {intent_context}\n"
        "If you have enough information, stop calling tools and respond."
        if intent_context else ""
    )

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
                )
                duration_ms = round((time.monotonic() - t0) * 1000, 1)

                result.tool_calls.append(ToolCallRecord(
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    result_json=result_str,
                    duration_ms=duration_ms,
                ))

                # Check for consecutive empty results from the same tool
                # and append a strategy hint if the streak is long enough.
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

                tool_results.append(
                    adapter.build_tool_result(
                        tc["id"], tc["name"],
                        result_str + strategy_hint + _question_reminder,
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

            # Compress previous rounds' tool results + assistant text
            # to avoid content repetition across continuations.
            # Extract call IDs from the tool_results we just appended.
            current_call_ids: set[str] = set()
            for tr in tool_results:
                cid = tr.get("call_id") or tr.get("tool_call_id") or ""
                if cid:
                    current_call_ids.add(cid)
                # Anthropic nests tool_use_id inside content blocks
                if isinstance(tr.get("content"), list):
                    for block in tr["content"]:
                        if isinstance(block, dict) and block.get("tool_use_id"):
                            current_call_ids.add(block["tool_use_id"])
            adapter.compress_previous_results(cont_body, current_call_ids)

            # Inject updated context into system prompt
            if new_prepend:
                adapter.inject_context(cont_body, new_prepend)

            # Dynamically inject vc_find_session tool if suppression occurred
            if not _find_session_injected:
                recent_results = result.tool_calls[-len(vc_tools):]
                if any(_SUPPRESSION_MARKER in r.result_json for r in recent_results):
                    adapter.add_tool_defs(cont_body, [_vc_find_session_def()])
                    _find_session_injected = True

            result.continuation_count += 1

            # Capture the continuation request body (deep copy to freeze state)
            result.raw_requests.append(copy.deepcopy(cont_body))

            # Send continuation (retry on 503)
            resp = client.post(url, headers=headers, json=cont_body)
            for _retry in range(2):
                if resp.status_code != 503:
                    break
                time.sleep(5)
                resp = client.post(url, headers=headers, json=cont_body)
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
                # the model already has the answer — strip tools and
                # force a text response.
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
                        # Execute the model's pending tool calls so the
                        # forced-text request has ALL collected data.
                        forced_results: list[dict] = []
                        for tc in vc_tools:
                            t0 = time.monotonic()
                            r_str = execute_vc_tool(
                                engine, tc["name"], tc["input"],
                                intent_context=intent_context,
                                presented_segment_refs=presented_refs,
                                presented_fact_ids=presented_facts,
                            )
                            dur = round((time.monotonic() - t0) * 1000, 1)
                            result.tool_calls.append(ToolCallRecord(
                                tool_name=tc["name"],
                                tool_input=tc["input"],
                                result_json=r_str, duration_ms=dur,
                            ))
                            forced_results.append(
                                adapter.build_tool_result(tc["id"], tc["name"], r_str)
                            )

                        # Build continuation with all tool data, then strip tools
                        cont_body = adapter.build_continuation(
                            cont_body, original_request, current_response,
                            forced_results,
                        )
                        forced_ids: set[str] = set()
                        for tr in forced_results:
                            cid = tr.get("call_id") or tr.get("tool_call_id") or ""
                            if cid:
                                forced_ids.add(cid)
                        adapter.compress_previous_results(cont_body, forced_ids)
                        adapter.strip_tools(cont_body)

                        result.continuation_count += 1
                        result.raw_requests.append(copy.deepcopy(cont_body))
                        resp = client.post(url, headers=headers, json=cont_body)
                        if resp.status_code < 300:
                            forced_data = _parse_provider_http_response(resp)
                            result.raw_responses.append(forced_data)
                            fi, fo = adapter.extract_usage(forced_data)
                            result.input_tokens += fi
                            result.output_tokens += fo
                            result.text += adapter.extract_text(forced_data)
                            result.stop_reason = "redundant_loop"
                        else:
                            logger.error(
                                "Forced text after redundant loop failed: HTTP %d",
                                resp.status_code,
                            )
                            result.stop_reason = "error"
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
                # Execute the last batch of tool calls so we can include
                # their results in the forced continuation
                tool_results_final: list[dict] = []
                for tc in vc_tools:
                    t0 = time.monotonic()
                    result_str = execute_vc_tool(
                        engine,
                        tc["name"],
                        tc["input"],
                        intent_context=intent_context,
                        presented_segment_refs=presented_refs,
                        presented_fact_ids=presented_facts,
                    )
                    duration_ms = round((time.monotonic() - t0) * 1000, 1)
                    result.tool_calls.append(ToolCallRecord(
                        tool_name=tc["name"],
                        tool_input=tc["input"],
                        result_json=result_str,
                        duration_ms=duration_ms,
                    ))
                    tool_results_final.append(
                        adapter.build_tool_result(
                            tc["id"], tc["name"], result_str,
                        )
                    )

                # Build continuation WITHOUT tools to force text output.
                # Do NOT reassemble context — the working set hasn't
                # changed, and reassembly can re-inject expanded summaries
                # that increase repetition.
                cont_body = adapter.build_continuation(
                    cont_body, original_request, current_response,
                    tool_results_final,
                )
                # Compress prior rounds before the forced send
                forced_call_ids: set[str] = set()
                for tr in tool_results_final:
                    cid = tr.get("call_id") or tr.get("tool_call_id") or ""
                    if cid:
                        forced_call_ids.add(cid)
                adapter.compress_previous_results(cont_body, forced_call_ids)
                adapter.strip_tools(cont_body)

                # For Gemini thinking models: minimize thinking budget on
                # the forced answer so tokens go to actual output text.
                gen_cfg = cont_body.get("generationConfig")
                if isinstance(gen_cfg, dict) and "thinkingConfig" in gen_cfg:
                    gen_cfg["thinkingConfig"] = {"thinkingBudget": 128}

                # Append a user nudge so the model responds with text
                # instead of burning tokens on internal reasoning.
                contents = cont_body.get("contents")
                if isinstance(contents, list):
                    contents.append({
                        "role": "user",
                        "parts": [{"text": (
                            "You have gathered enough information. "
                            "Answer the original question now in 1-2 sentences."
                        )}],
                    })

                result.continuation_count += 1
                result.raw_requests.append(copy.deepcopy(cont_body))
                resp = client.post(url, headers=headers, json=cont_body)
                if resp.status_code >= 300:
                    logger.warning(
                        "Forced text continuation failed (HTTP %d), retrying once",
                        resp.status_code,
                    )
                    time.sleep(1.0)
                    resp = client.post(url, headers=headers, json=cont_body)
                if resp.status_code < 300:
                    forced_data = _parse_provider_http_response(resp)
                    result.raw_responses.append(forced_data)
                    fi, fo = adapter.extract_usage(forced_data)
                    result.input_tokens += fi
                    result.output_tokens += fo
                    result.text += adapter.extract_text(forced_data)
                    result.stop_reason = adapter.get_stop_reason(forced_data)
                else:
                    logger.error(
                        "Forced text continuation failed after retry: HTTP %d",
                        resp.status_code,
                    )
                    result.stop_reason = "error"
            else:
                result.stop_reason = stop_reason

    return result
