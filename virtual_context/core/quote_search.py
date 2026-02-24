"""Quote search: full-text + semantic + description-aware search.

Orchestrates FTS, semantic embedding search, and description scanning
to find specific phrases or keywords across all stored conversation text.
Extracted from engine.py.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from datetime import date

from ..types import QuoteResult, SegmentMetadata
from .semantic_search import SemanticSearchManager
from .store import ContextStore

logger = logging.getLogger(__name__)


# Deterministic query intent detection for state-updated questions.
_CURRENT_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcurrently\b", re.IGNORECASE),
    re.compile(r"\bnow\b", re.IGNORECASE),
    re.compile(r"\blatest\b", re.IGNORECASE),
    re.compile(r"\bat the moment\b", re.IGNORECASE),
    re.compile(r"\bthese days\b", re.IGNORECASE),
)


def _detect_query_intent(query: str) -> str:
    """Return deterministic intent label for quote retrieval ordering."""
    for pattern in _CURRENT_STATE_PATTERNS:
        if pattern.search(query):
            return "current_state"
    return "default"


def _parse_session_date(raw: str) -> date | None:
    """Best-effort parse for session date strings into ``date`` values."""
    s = (raw or "").strip()
    if not s:
        return None

    m = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    # Month-granularity fallback if day is absent.
    m = re.search(r"\b(\d{4})[-/](\d{2})\b", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            return None

    return None


def _normalize_session_date(raw: str) -> str:
    """Normalize parseable session date to ISO-8601 date string."""
    parsed = _parse_session_date(raw)
    return parsed.isoformat() if parsed else ""


# ---------------------------------------------------------------------------
# Span-union helpers — merge overlapping excerpts from the same segment
# ---------------------------------------------------------------------------


def _union_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent (start, end) intervals into minimal set."""
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _locate_excerpt(full_text: str, excerpt: str) -> tuple[int, int] | None:
    """Find an excerpt's (start, end) char position in full_text.

    Strips ellipsis bookends and FTS5 highlight markers (>>> / <<<)
    before searching.  Returns None if the cleaned text can't be found.
    """
    clean = excerpt
    # Strip FTS5 highlight markers
    clean = clean.replace(">>>", "").replace("<<<", "")
    # Strip leading/trailing ellipsis added by extract_excerpt
    if clean.startswith("..."):
        clean = clean[3:]
    if clean.endswith("..."):
        clean = clean[:-3]
    clean = clean.strip()
    if not clean:
        return None

    idx = full_text.find(clean)
    if idx >= 0:
        return (idx, idx + len(clean))

    # Case-insensitive fallback
    idx = full_text.lower().find(clean.lower())
    if idx >= 0:
        return (idx, idx + len(clean))

    return None


def _merge_segment_excerpts(
    store: ContextStore,
    results: list[QuoteResult],
) -> list[QuoteResult]:
    """Merge overlapping excerpts from the same segment_ref via span-union.

    Groups results by segment_ref.  For segments with multiple hits,
    locates each excerpt in the segment's full_text, computes the
    interval union, and re-extracts the merged (non-overlapping) text.
    Single-hit segments pass through unchanged.
    """
    # Group by segment_ref, preserving insertion order
    by_ref: OrderedDict[str, list[QuoteResult]] = OrderedDict()
    for qr in results:
        by_ref.setdefault(qr.segment_ref, []).append(qr)

    merged: list[QuoteResult] = []
    for ref, group in by_ref.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Multiple excerpts from the same segment — try span-union
        seg = store.get_segment(ref)
        if not seg or not seg.full_text:
            # Can't locate spans — keep first result only
            merged.append(group[0])
            continue

        full_text = seg.full_text
        spans: list[tuple[int, int]] = []
        unlocatable: list[QuoteResult] = []

        for qr in group:
            span = _locate_excerpt(full_text, qr.text)
            if span is not None:
                spans.append(span)
            else:
                unlocatable.append(qr)

        if not spans:
            # None located — keep first result
            merged.append(group[0])
            continue

        # Compute span union
        united = _union_spans(spans)

        # Extract text for each merged span
        parts: list[str] = []
        for start, end in united:
            part = full_text[start:end]
            if start > 0:
                part = "..." + part
            if end < len(full_text):
                part = part + "..."
            parts.append(part)

        # Combine metadata from all results in this group
        all_tags: list[str] = []
        for qr in group:
            for t in (qr.tags or []):
                if t not in all_tags:
                    all_tags.append(t)

        best = group[0]
        best_similarity = max(qr.similarity for qr in group)

        merged.append(QuoteResult(
            text="\n---\n".join(parts),
            tag=best.tag,
            segment_ref=ref,
            tags=all_tags,
            match_type=best.match_type,
            similarity=best_similarity,
            session_date=best.session_date,
        ))

        # Append any excerpts we couldn't locate (don't lose data)
        merged.extend(unlocatable)

    return merged


def supplement_from_descriptions(
    store: ContextStore,
    query: str,
    results: list[QuoteResult],
    max_results: int,
) -> list[QuoteResult]:
    """Add results from tags whose descriptions match query terms.

    Scans all tag descriptions for query words.  For tags that match
    but aren't already represented in *results*, fetches their
    segments and searches full_text for the best excerpt.  This
    bridges the vocabulary gap: the compacted description may use
    words that don't appear in FTS or embedding results.
    """
    from ..storage.helpers import extract_excerpt

    # Tags already covered by existing results
    covered_tags: set[str] = set()
    for qr in results:
        covered_tags.add(qr.tag)

    # Tokenize query into meaningful words (3+ chars)
    query_words = [
        w.lower()
        for w in re.findall(r"[a-zA-Z']+", query)
        if len(w) >= 3
    ]
    if not query_words:
        return results

    # Precompile word-boundary patterns to avoid substring false positives
    # (e.g. "old" matching "gold").
    word_patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in query_words]

    # Score each tag description by how many query words it contains
    all_summaries = store.get_all_tag_summaries()
    candidates: list[tuple[str, int, str]] = []  # (tag, score, description)
    for ts in all_summaries:
        if ts.tag in covered_tags:
            continue
        desc_lower = ts.description.lower() if ts.description else ""
        if not desc_lower:
            continue
        score = sum(1 for p in word_patterns if p.search(desc_lower))
        if score > 0:
            candidates.append((ts.tag, score, ts.description))

    # Sort by score descending, take top candidates
    candidates.sort(key=lambda x: -x[1])
    slots = max_results - len(results)
    if slots <= 0:
        return results

    for tag, _score, _desc in candidates[:slots]:
        # Fetch segments for this tag and search their full_text
        segments = store.get_segments_by_tags([tag], limit=5)
        best: QuoteResult | None = None
        best_score = 0
        for seg in segments:
            if not seg.full_text:
                continue
            text_lower = seg.full_text.lower()
            seg_score = sum(1 for p in word_patterns if p.search(text_lower))
            if seg_score > best_score:
                best_score = seg_score
                excerpt = extract_excerpt(
                    seg.full_text, query, context_chars=200
                )
                meta = seg.metadata or SegmentMetadata(turn_count=0)
                best = QuoteResult(
                    text=excerpt,
                    tag=seg.primary_tag,
                    segment_ref=seg.ref,
                    tags=seg.tags,
                    match_type="description",
                    session_date=meta.session_date,
                )
        if best is not None:
            results.append(best)
            covered_tags.add(tag)

    return results


def find_quote(
    store: ContextStore,
    semantic: SemanticSearchManager,
    query: str,
    max_results: int = 5,
    intent_context: str = "",
    session_filter: str = "",
) -> dict:
    """Search stored conversation text for a specific phrase or keyword.

    Searches across all segments' full_text regardless of tags.
    Returns FTS matches supplemented by semantic (embedding) search
    to surface excerpts that use different vocabulary.  Works even when
    paging is disabled — this is a pure search tool with no working-set
    side effects.
    """
    if not query.strip():
        return {"error": "empty query"}

    query_intent = _detect_query_intent(query)
    if query_intent == "default" and intent_context.strip():
        # Fall back to the original user question/intent context when
        # the model issues a narrow tool query (e.g. "shoe rack").
        query_intent = _detect_query_intent(intent_context)

    results = store.search_full_text(query, limit=max_results)

    # Always run semantic search to supplement FTS — surfaces chunks
    # that match semantically but use different words, and may return
    # different excerpts from the same segment FTS already found.
    remaining = max_results - len(results)
    if remaining > 0:
        semantic_results = semantic.semantic_search(query, max_results=remaining)
        results.extend(semantic_results)

    # ---- Description-aware search ----
    results = supplement_from_descriptions(store, query, results, max_results)

    # ---- Span-union: merge overlapping excerpts from the same segment ----
    results = _merge_segment_excerpts(store, results)

    if not results:
        return {
            "query": query,
            "found": False,
            "results": [],
            "message": f"No matches for '{query}' in stored conversation history.",
        }

    # ---- Merge snippets from the same session ----
    session_groups: OrderedDict[str, list[QuoteResult]] = OrderedDict()
    no_session: list[QuoteResult] = []
    for qr in results:
        key = qr.session_date.strip() if qr.session_date else ""
        if key:
            session_groups.setdefault(key, []).append(qr)
        else:
            no_session.append(qr)

    # Parse + normalize session dates for every session-group hit.
    session_dates_parsed: dict[str, date | None] = {}
    session_dates_normalized: dict[str, str] = {}
    for session_date in session_groups:
        parsed = _parse_session_date(session_date)
        session_dates_parsed[session_date] = parsed
        session_dates_normalized[session_date] = parsed.isoformat() if parsed else ""

    # When session_filter is provided, restrict to matching sessions and
    # skip current-state suppression (the model is explicitly drilling in).
    if session_filter.strip():
        _sf = session_filter.strip()
        filtered = OrderedDict()
        for k, v in session_groups.items():
            norm = session_dates_normalized.get(k, "")
            if _sf in k or _sf in norm:
                filtered[k] = v
        if not filtered:
            return {
                "query": query,
                "found": False,
                "results": [],
                "message": f"No matches for '{query}' in session '{_sf}'.",
            }
        session_groups = filtered
        # Bypass current-state multi-session logic when drilling into a
        # specific session — return full excerpts, no suppression.
        session_items = list(session_groups.items())
        formatted = []
        for session_date, group in session_items:
            if len(group) == 1:
                qr = group[0]
                entry: dict = {
                    "excerpt": qr.text,
                    "topic": qr.tag,
                    "session": session_date,
                    "session_date_normalized": session_dates_normalized.get(session_date, ""),
                }
                if qr.match_type == "semantic":
                    entry["match_type"] = "semantic"
                    entry["similarity"] = qr.similarity
                formatted.append(entry)
            else:
                seen_texts: set[str] = set()
                merged_parts: list[str] = []
                topics: list[str] = []
                for qr in group:
                    norm = qr.text.strip()
                    if norm not in seen_texts:
                        seen_texts.add(norm)
                        merged_parts.append(qr.text.strip())
                    if qr.tag not in topics:
                        topics.append(qr.tag)
                entry = {
                    "excerpt": "\n---\n".join(merged_parts),
                    "topic": ", ".join(topics),
                    "session": session_date,
                    "session_date_normalized": session_dates_normalized.get(session_date, ""),
                    "merged_count": len(merged_parts),
                }
                formatted.append(entry)
        for qr in no_session:
            formatted.append({
                "excerpt": qr.text,
                "topic": qr.tag,
            })
        return {
            "query": query,
            "query_intent": query_intent,
            "session_filter": _sf,
            "found": True,
            "results": formatted,
        }

    session_items = list(session_groups.items())
    if query_intent == "current_state":
        # For "currently/now/latest" queries, prefer most recent sessions first.
        session_items.sort(
            key=lambda kv: (
                session_dates_parsed.get(kv[0]) is not None,
                session_dates_parsed.get(kv[0]) or date.min,
            ),
            reverse=True,
        )

    current_state_multi_session = (
        query_intent == "current_state" and len(session_items) > 1
    )
    formatted = []

    for session_rank, (session_date, group) in enumerate(session_items, start=1):
        if len(group) == 1:
            qr = group[0]
            entry: dict = {
                "excerpt": qr.text,
                "topic": qr.tag,
                "segment_ref": qr.segment_ref,
                "session": session_date,
                "session_date_normalized": session_dates_normalized.get(session_date, ""),
            }
            if current_state_multi_session:
                entry["session_recency_rank"] = session_rank
                if session_rank == 1:
                    entry["priority"] = "HIGHEST_PRIORITY_CURRENT_STATE"
            if qr.match_type == "semantic":
                entry["match_type"] = "semantic"
                entry["similarity"] = qr.similarity
            formatted.append(entry)
        else:
            # Multiple hits from the same session — merge excerpts
            seen_texts: set[str] = set()
            merged_parts: list[str] = []
            all_refs: list[str] = []
            topics: list[str] = []
            for qr in group:
                norm = qr.text.strip()
                if norm not in seen_texts:
                    seen_texts.add(norm)
                    merged_parts.append(qr.text.strip())
                if qr.tag not in topics:
                    topics.append(qr.tag)
                if qr.segment_ref not in all_refs:
                    all_refs.append(qr.segment_ref)

            entry = {
                "excerpt": "\n---\n".join(merged_parts),
                "topic": ", ".join(topics),
                "segment_refs": all_refs,
                "session": session_date,
                "session_date_normalized": session_dates_normalized.get(session_date, ""),
                "merged_count": len(merged_parts),
            }
            if current_state_multi_session:
                entry["session_recency_rank"] = session_rank
                if session_rank == 1:
                    entry["priority"] = "HIGHEST_PRIORITY_CURRENT_STATE"
            formatted.append(entry)

    # Append results with no session date individually
    for qr in no_session:
        entry = {
            "excerpt": qr.text,
            "topic": qr.tag,
            "segment_ref": qr.segment_ref,
        }
        if qr.match_type == "semantic":
            entry["match_type"] = "semantic"
            entry["similarity"] = qr.similarity
        formatted.append(entry)

    response = {
        "query": query,
        "query_intent": query_intent,
        "found": True,
        "results": formatted,
    }
    if current_state_multi_session:
        response["current_state_multi_session"] = True
        response["priority_label"] = "HIGHEST_PRIORITY_CURRENT_STATE"
        # Build a deterministic resolution from the newest session's date.
        newest_date = ""
        if formatted:
            newest_date = formatted[0].get("session_date_normalized", "") or formatted[0].get("session", "")
        response["reader_hint"] = (
            "CURRENT-STATE RESOLUTION RULE (mandatory): "
            "The most recent session"
            + (f" ({newest_date})" if newest_date else "")
            + " is the single authoritative source for the user's current state. "
            "Older sessions are superseded history — do NOT use them to override the newest session. "
            "Intent language ('going to', 'planning to', 'looking forward to') in the newest session "
            "counts as the current answer because the question date is after that session."
        )
    return response
