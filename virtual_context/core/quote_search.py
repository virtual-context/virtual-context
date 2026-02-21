"""Quote search: full-text + semantic + description-aware search.

Orchestrates FTS, semantic embedding search, and description scanning
to find specific phrases or keywords across all stored conversation text.
Extracted from engine.py.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Callable

from ..types import QuoteResult, SegmentMetadata
from .semantic_search import SemanticSearchManager
from .store import ContextStore

logger = logging.getLogger(__name__)


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

    # Score each tag description by how many query words it contains
    all_summaries = store.get_all_tag_summaries()
    candidates: list[tuple[str, int, str]] = []  # (tag, score, description)
    for ts in all_summaries:
        if ts.tag in covered_tags:
            continue
        desc_lower = ts.description.lower() if ts.description else ""
        if not desc_lower:
            continue
        score = sum(1 for w in query_words if w in desc_lower)
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
            seg_score = sum(1 for w in query_words if w in text_lower)
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

    formatted = []

    for session_date, group in session_groups.items():
        if len(group) == 1:
            qr = group[0]
            entry: dict = {
                "excerpt": qr.text,
                "topic": qr.tag,
                "segment_ref": qr.segment_ref,
                "session": session_date,
            }
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
                "merged_count": len(merged_parts),
            }
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

    return {
        "query": query,
        "found": True,
        "results": formatted,
    }
