"""TemporalResolver: time-bounded recall over stored conversation data.

Handles ``remember_when`` queries: resolves relative/absolute time ranges,
filters quote search results by session date, and queries experience facts
within the date window.  Extracted from engine.py.
"""

from __future__ import annotations

import logging
import re
from calendar import monthrange
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from math import ceil, floor
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .math_utils import cosine_similarity
from .quote_search import _parse_session_date as _parse_session_date_str

if TYPE_CHECKING:
    from .semantic_search import SemanticSearchManager
    from .search_engine import SearchEngine
    from .store import ContextStore
    from ..types import VirtualContextConfig

logger = logging.getLogger(__name__)

_TIMELINE_SUMMARY_WEIGHT = 1.25
_TIMELINE_SUMMARY_BONUS = 0.5

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUERY_STOPWORDS = {
    "about", "after", "all", "also", "and", "any", "are", "around", "back",
    "before", "between", "can", "could", "did", "does", "during", "each",
    "from", "have", "into", "just", "last", "like", "most", "much", "need",
    "over", "past", "should", "show", "some", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "under", "until", "very", "what", "when", "where", "which", "while",
    "with", "would", "your", "mine", "ours", "overall", "key",
}
_GENERIC_QUERY_TERMS = {
    "capability", "capabilities", "development", "developments",
    "change", "changed", "changes", "evolution", "evolve", "evolved",
    "history", "journey", "overview", "plan", "plans", "progress",
    "improvement", "improvements", "summaries", "summarization", "summarize",
    "summary", "system",
    "systems", "timeline", "topic", "topics", "update", "updates",
}
_TIMELINE_QUERY_RE = re.compile(
    r"\b(summarize|summary|evolv|changed|progress|timeline|history|roadmap|journey)\b",
    re.IGNORECASE,
)
_REMEMBER_WHEN_MODE_ALIASES = {
    "timeline": "change_over_time",
    "summary": "summarize_over_time",
    "timeline_summary": "summarize_over_time",
    "changes": "change_over_time",
    "change": "change_over_time",
    "state": "state_at_time",
    "overview": "window_overview",
    "window": "window_overview",
}
_REMEMBER_WHEN_VALID_MODES = {
    "auto",
    "lookup",
    "change_over_time",
    "summarize_over_time",
    "state_at_time",
    "window_overview",
}
_REMEMBER_WHEN_TIMELINE_MODES = {"change_over_time", "summarize_over_time", "window_overview"}
_REMEMBER_WHEN_BROAD_FACT_MODES = {"change_over_time", "summarize_over_time", "window_overview"}
_REMEMBER_WHEN_SUMMARY_MODES = {"summarize_over_time"}
_REMEMBER_WHEN_CHANGE_MODES = {"change_over_time"}
_REMEMBER_WHEN_WINDOW_OVERVIEW_MODES = {"window_overview"}
_REMEMBER_WHEN_DATE_BUCKET_MODES = {"change_over_time", "window_overview"}
_REMEMBER_WHEN_CHANGE_MIN_RESULTS = 22
_REMEMBER_WHEN_CHANGE_RESULTS_PER_DATE = 4
_REMEMBER_WHEN_CHANGE_FACTS_PER_DATE = 3
_REMEMBER_WHEN_STATE_MODES = {"state_at_time"}
_VALUE_TOKEN_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _query_priority_terms(
    query: str,
    *,
    drop_generic: bool = True,
) -> list[str]:
    """Return normalized query terms while preserving short acronyms like RAG/API."""
    raw_tokens = re.findall(r"[A-Za-z0-9]+", query)
    ordered_unique: list[str] = []
    seen: set[str] = set()

    for raw in raw_tokens:
        token = raw.lower()
        if token in seen:
            continue
        seen.add(token)

        is_short_acronym = len(raw) >= 3 and raw.isupper()
        has_digit = len(raw) >= 3 and any(ch.isdigit() for ch in raw)
        if len(token) >= 4 or is_short_acronym or has_digit:
            if token not in _QUERY_STOPWORDS:
                ordered_unique.append(token)

    if drop_generic and len(ordered_unique) > 2:
        filtered = [
            token
            for token in ordered_unique
            if token not in _GENERIC_QUERY_TERMS
        ]
        if filtered:
            return filtered
    return ordered_unique


class TemporalResolver:
    """Time-bounded recall: resolves date windows, filters quotes, queries facts.

    Constructor takes:
        store:          a ContextStore instance
        search_engine:  a SearchEngine instance (for ``find_quote``)
        config:         a VirtualContextConfig instance
    """

    # Bounded LRU cap on the summary-embedding cache. Sized to cover a large
    # segment set against a reasonable number of distinct query texts without
    # letting a long-lived conversation engine grow the cache without bound.
    # TODO: consider persisting summary embeddings at compaction time (like
    # full_text_chunks does for turn text) and dropping this in-memory cache
    # entirely. That would make retrieval hits warm across engine restarts
    # instead of re-embedding per process.
    _CACHE_MAX_SIZE = 2000

    def __init__(
        self,
        store: ContextStore,
        search_engine: SearchEngine,
        config: VirtualContextConfig,
        semantic: SemanticSearchManager | None = None,
    ) -> None:
        self._store = store
        self._search = search_engine
        self._semantic = semantic
        self._config = config
        self.reference_date: date | None = None
        self._summary_embedding_cache: "OrderedDict[tuple[str, str, str], list[float]]" = OrderedDict()

    def _cache_get(self, key: tuple[str, str, str]) -> list[float] | None:
        val = self._summary_embedding_cache.get(key)
        if val is not None:
            self._summary_embedding_cache.move_to_end(key)
        return val

    def _cache_set(self, key: tuple[str, str, str], val: list[float]) -> None:
        self._summary_embedding_cache[key] = val
        self._summary_embedding_cache.move_to_end(key)
        while len(self._summary_embedding_cache) > self._CACHE_MAX_SIZE:
            self._summary_embedding_cache.popitem(last=False)

    def _cache_contains(self, key: tuple[str, str, str]) -> bool:
        return key in self._summary_embedding_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember_when(
        self,
        query: str,
        time_range: dict,
        max_results: int | None = None,
        mode: str = "auto",
        intent_context: str = "",
    ) -> dict:
        """Find memory snippets for *query* constrained to a resolved date window."""
        resolved_mode = self._normalize_remember_when_mode(mode)
        if max_results is None:
            max_results = self._default_remember_when_max_results(resolved_mode)
        query = str(query or "").strip()
        if not query and resolved_mode not in _REMEMBER_WHEN_WINDOW_OVERVIEW_MODES:
            return {"error": "empty query"}

        try:
            start, end, resolved_kind = self._resolve_remember_when_range(time_range)
        except ValueError as exc:
            return {"error": str(exc)}

        state_target_date = self._state_target_date(start, end, resolved_mode)
        all_search_variants = self._build_search_variants(query) if query else []
        search_variants = all_search_variants
        if resolved_mode in _REMEMBER_WHEN_CHANGE_MODES:
            search_variants = self._prune_change_search_variants(all_search_variants)
        prefer_timeline = self._prefer_timeline_coverage(
            query,
            start,
            end,
            max_results,
            mode=resolved_mode,
        )

        filtered = self._search_segments_in_window(
            query=query,
            search_variants=search_variants,
            start=start,
            end=end,
            max_results=max_results,
            prefer_timeline=prefer_timeline,
            mode=resolved_mode,
            target_date=state_target_date,
            intent_context=intent_context,
        )
        if (
            resolved_mode in _REMEMBER_WHEN_CHANGE_MODES
            and not filtered
            and search_variants != all_search_variants
        ):
            filtered = self._search_segments_in_window(
                query=query,
                search_variants=all_search_variants,
                start=start,
                end=end,
                max_results=max_results,
                prefer_timeline=prefer_timeline,
                mode=resolved_mode,
                target_date=state_target_date,
                intent_context=intent_context,
            )

        # Structured facts should be topic-matched first, then filtered/diversified
        # inside the time window. Falling back to a raw date scan is a last resort.
        fact_results = self._search_facts_in_window(
            query=query,
            search_variants=search_variants,
            start=start,
            end=end,
            max_results=max_results,
            prefer_timeline=prefer_timeline,
            mode=resolved_mode,
            segment_results=filtered,
            target_date=state_target_date,
        )
        if (
            resolved_mode in _REMEMBER_WHEN_CHANGE_MODES
            and not fact_results
            and search_variants != all_search_variants
        ):
            fact_results = self._search_facts_in_window(
                query=query,
                search_variants=all_search_variants,
                start=start,
                end=end,
                max_results=max_results,
                prefer_timeline=prefer_timeline,
                mode=resolved_mode,
                segment_results=filtered,
                target_date=state_target_date,
            )
        if resolved_mode in _REMEMBER_WHEN_CHANGE_MODES:
            filtered = self._supplement_change_results_for_fact_backed_dates(
                query=query,
                intent_context=intent_context,
                results=filtered,
                facts=fact_results,
                start=start,
                end=end,
            )

        message = ""
        if not filtered and not fact_results and query:
            raw = self._search.find_quote(query=query, max_results=max(max_results * 4, 20))
            if raw.get("found"):
                fallback_results: list[dict] = []
                for item in raw.get("results", []):
                    session = str(item.get("session", "")).strip()
                    parsed = self._parse_session_date(session)
                    if parsed is None:
                        continue
                    if start <= parsed <= end:
                        fallback_results.append(item)
                    if len(fallback_results) >= max_results:
                        break
                filtered = fallback_results
            if not filtered:
                message = raw.get("message", f"No matches for '{query}' in the requested time window.")
        elif not filtered and not fact_results:
            message = "No remembered activity in the requested time window."

        result = {
            "query": query,
            "mode": resolved_mode,
            "found": bool(filtered) or bool(fact_results),
            "range": {
                "kind": resolved_kind,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "results": filtered,
            "facts_in_window": fact_results,
            "message": message,
        }
        if resolved_mode in (_REMEMBER_WHEN_SUMMARY_MODES | _REMEMBER_WHEN_CHANGE_MODES):
            ordered_milestones = self._build_ordered_milestones(
                query=query,
                results=filtered,
                facts=fact_results,
                max_results=max_results,
            )
            if ordered_milestones:
                result["ordered_milestones"] = ordered_milestones
                phase_milestones = self._build_phase_milestones(
                    ordered_milestones=ordered_milestones,
                )
                if phase_milestones:
                    result["phase_milestones"] = phase_milestones
        if resolved_mode in _REMEMBER_WHEN_DATE_BUCKET_MODES:
            date_buckets = self._build_change_date_buckets(
                results=filtered,
                facts=fact_results,
                max_results=max_results,
            )
            if date_buckets:
                result["date_buckets"] = date_buckets
        if state_target_date is not None:
            state_view = self._resolve_state_view(
                results=filtered,
                facts=fact_results,
                target_date=state_target_date,
                max_results=max_results,
            )
            result["results"] = state_view["results"]
            result["facts_in_window"] = state_view["facts_in_window"]
            if state_view.get("chosen_state") is not None:
                result["chosen_state"] = state_view["chosen_state"]
            if state_view.get("conflicting_candidates"):
                result["conflicting_candidates"] = state_view["conflicting_candidates"]
            result["target_date"] = state_target_date.isoformat()
            state_anchor = self._describe_state_anchor(
                results=result["results"],
                facts=result["facts_in_window"],
                target_date=state_target_date,
            )
            if state_anchor:
                result["state_anchor"] = state_anchor
        return result

    def _default_remember_when_max_results(self, resolved_mode: str) -> int:
        base = self._config.search.remember_when_max_results
        if resolved_mode in _REMEMBER_WHEN_CHANGE_MODES | _REMEMBER_WHEN_WINDOW_OVERVIEW_MODES:
            return max(base, _REMEMBER_WHEN_CHANGE_MIN_RESULTS)
        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_session_date(self, raw: str) -> date | None:
        """Best-effort parse for session date strings from stored metadata."""
        return _parse_session_date_str(raw)

    def _parse_fact_date(self, raw: str) -> date | None:
        raw = str(raw or "").strip()
        if not raw:
            return None
        if len(raw) >= 10 and _ISO_DATE_RE.match(raw[:10]):
            try:
                return date.fromisoformat(raw[:10])
            except ValueError:
                pass
        return self._parse_session_date(raw)

    def _normalize_remember_when_mode(self, mode: str | None) -> str:
        normalized = str(mode or "auto").strip().lower().replace(" ", "_")
        normalized = _REMEMBER_WHEN_MODE_ALIASES.get(normalized, normalized)
        if normalized not in _REMEMBER_WHEN_VALID_MODES:
            return "auto"
        return normalized

    def _build_search_variants(self, query: str) -> list[tuple[str, float, set[str]]]:
        primary_terms = _query_priority_terms(query)
        primary_terms = primary_terms[:6]

        variants: list[tuple[str, float, set[str]]] = []
        seen_variants: set[str] = set()

        def add_variant(text: str, weight: float, matched_terms: set[str]) -> None:
            cleaned = text.strip()
            if not cleaned or cleaned in seen_variants:
                return
            seen_variants.add(cleaned)
            variants.append((cleaned, weight, matched_terms))

        for idx in range(len(primary_terms) - 1):
            pair = f"{primary_terms[idx]} {primary_terms[idx + 1]}"
            add_variant(pair, 3.0, {primary_terms[idx], primary_terms[idx + 1]})

        for term in primary_terms:
            add_variant(term, 1.75, {term})

        if not variants:
            fallback_terms = _query_priority_terms(query, drop_generic=False)
            if fallback_terms:
                for term in fallback_terms[:3]:
                    add_variant(term, 1.5, {term})
            else:
                add_variant(query, 2.0, set())

        return variants

    def _window_query_terms(
        self,
        query: str,
        search_variants: list[tuple[str, float, set[str]]],
    ) -> set[str]:
        terms = set(_query_priority_terms(query, drop_generic=False))
        if terms:
            return terms
        return self._query_terms_from_variants(search_variants)

    def _segment_search_text(self, segment) -> str:
        return " ".join(
            part
            for part in [
                str(segment.primary_tag or "").strip(),
                " ".join(str(tag).strip() for tag in (segment.tags or []) if str(tag).strip()),
                str(segment.summary or "").strip(),
                str(segment.full_text or "").strip()[:2000],
            ]
            if part
        ).lower()

    def _seed_segment_candidates_from_window(
        self,
        *,
        query: str,
        search_variants: list[tuple[str, float, set[str]]],
        start: date,
        end: date,
        conversation_id: str | None,
    ) -> dict[str, dict]:
        try:
            segments = self._store.get_all_segments(conversation_id=conversation_id)
        except Exception as exc:
            logger.warning("remember_when window segment scan failed: %s", exc)
            return {}

        query_terms = self._window_query_terms(query, search_variants)
        candidates: dict[str, dict] = {}
        for seg in segments:
            parsed = self._parse_session_date(seg.metadata.session_date)
            if parsed is None or not (start <= parsed <= end):
                continue
            segment_ref = str(seg.ref or "").strip()
            if not segment_ref:
                continue
            summary_text = str(seg.summary or "").strip()
            excerpt = summary_text or str(seg.full_text or "").strip()
            if not excerpt:
                continue
            search_text = self._segment_search_text(seg)
            overlap = {term for term in query_terms if term in search_text}
            base_score = 0.15 if summary_text else 0.05
            base_score += min(0.25, 0.03 * len([tag for tag in (seg.tags or []) if str(tag).strip()]))
            base_score += 0.55 * len(overlap)
            candidates[segment_ref] = {
                "quote": SimpleNamespace(
                    segment_ref=segment_ref,
                    tag=str(seg.primary_tag or ""),
                    tags=list(seg.tags or []),
                    text=excerpt,
                    match_type="summary_window" if summary_text else "window_full_text",
                    session_date=str(seg.metadata.session_date or ""),
                    created_at=str(seg.created_at or ""),
                ),
                "date": parsed,
                "score": base_score,
                "matched_terms": set(),
                "soft_terms": set(overlap),
                "summary_text": summary_text or None,
                "sources": {"window_scan"},
            }
        return candidates

    def _segment_candidate_has_query_signal(self, item: dict) -> bool:
        if item.get("matched_terms"):
            return True
        return any(source != "window_scan" for source in item.get("sources", set()))

    def _fact_candidate_has_query_signal(self, item: dict) -> bool:
        if item.get("matched_terms"):
            return True
        return any(source != "window_scan" for source in item.get("sources", set()))

    def _should_prune_window_only_candidates(
        self,
        *,
        query_terms: set[str],
        mode: str,
        signaled_count: int,
    ) -> bool:
        if mode in _REMEMBER_WHEN_WINDOW_OVERVIEW_MODES:
            return False
        if not query_terms:
            return False
        if mode == "lookup":
            return signaled_count >= 1
        return signaled_count >= 4

    def _prefer_timeline_coverage(
        self,
        query: str,
        start: date,
        end: date,
        max_results: int,
        *,
        mode: str = "auto",
    ) -> bool:
        if mode in _REMEMBER_WHEN_TIMELINE_MODES:
            return True
        if mode in {"lookup", "state_at_time"}:
            return False
        span_days = (end - start).days + 1
        return bool(_TIMELINE_QUERY_RE.search(query)) or span_days > max(max_results * 2, 10)

    def _search_segments_in_window(
        self,
        *,
        query: str,
        search_variants: list[tuple[str, float, set[str]]],
        start: date,
        end: date,
        max_results: int,
        prefer_timeline: bool,
        mode: str = "auto",
        target_date: date | None = None,
        intent_context: str = "",
    ) -> list[dict]:
        conversation_id = self._config.conversation_id or None
        per_variant_limit = min(max(max_results * 10, 50), 200)
        candidates = self._seed_segment_candidates_from_window(
            query=query,
            search_variants=search_variants,
            start=start,
            end=end,
            conversation_id=conversation_id,
        )
        query_terms = self._window_query_terms(query, search_variants)
        prefer_summary_hits = mode in _REMEMBER_WHEN_TIMELINE_MODES
        summary_search = getattr(self._store, "search", None)

        for variant, base_weight, matched_terms in search_variants:
            if prefer_summary_hits and callable(summary_search):
                try:
                    summary_hits = summary_search(
                        variant,
                        limit=per_variant_limit,
                        conversation_id=conversation_id,
                    )
                except Exception as exc:
                    logger.warning("remember_when summary search failed for '%s': %s", variant, exc)
                    summary_hits = []

                for hit in summary_hits:
                    parsed = self._parse_session_date(
                        getattr(getattr(hit, "metadata", None), "session_date", ""),
                    )
                    if parsed is None or not (start <= parsed <= end):
                        continue
                    segment_ref = str(getattr(hit, "ref", "") or "").strip()
                    if not segment_ref:
                        continue
                    tag = str(getattr(hit, "primary_tag", "") or "")
                    summary_text = str(getattr(hit, "summary", "") or "")
                    bucket = candidates.setdefault(
                        segment_ref,
                        {
                            "quote": SimpleNamespace(
                                segment_ref=segment_ref,
                                tag=tag,
                                tags=list(getattr(hit, "tags", []) or []),
                                text=summary_text,
                                match_type="summary",
                                session_date=getattr(getattr(hit, "metadata", None), "session_date", ""),
                                created_at=str(getattr(hit, "created_at", "") or ""),
                            ),
                            "date": parsed,
                            "score": 0.0,
                            "matched_terms": set(),
                        },
                    )
                    bucket.setdefault("sources", set()).add("query_match")
                    bucket["matched_terms"].update(matched_terms)
                    summary_match_text = " ".join([tag, summary_text]).lower()
                    overlap_bonus = sum(1 for term in matched_terms if term in summary_match_text)
                    bucket["score"] += (
                        (base_weight + overlap_bonus * 0.5) * _TIMELINE_SUMMARY_WEIGHT
                        + _TIMELINE_SUMMARY_BONUS
                    )
                    if summary_text:
                        bucket["summary_text"] = summary_text
                    bucket["quote"] = SimpleNamespace(
                        segment_ref=segment_ref,
                        tag=tag,
                        tags=list(getattr(hit, "tags", []) or []),
                        text=summary_text,
                        match_type="summary",
                        session_date=getattr(getattr(hit, "metadata", None), "session_date", ""),
                        created_at=str(getattr(hit, "created_at", "") or ""),
                    )

            try:
                hits = self._store.search_full_text(
                    variant,
                    limit=per_variant_limit,
                    conversation_id=conversation_id,
                )
            except Exception as exc:
                logger.warning("remember_when segment search failed for '%s': %s", variant, exc)
                continue

            for hit in hits:
                parsed = self._parse_session_date(hit.session_date)
                if parsed is None or not (start <= parsed <= end):
                    continue
                bucket = candidates.setdefault(
                    hit.segment_ref,
                    {
                        "quote": hit,
                        "date": parsed,
                        "score": 0.0,
                        "matched_terms": set(),
                        "sources": set(),
                    },
                )
                bucket.setdefault("sources", set()).add("query_match")
                bucket["matched_terms"].update(matched_terms)
                text = " ".join([hit.tag or "", hit.text or ""]).lower()
                overlap_bonus = sum(1 for term in matched_terms if term in text)
                bucket["score"] += base_weight + overlap_bonus * 0.5

        if mode in _REMEMBER_WHEN_CHANGE_MODES:
            self._merge_semantic_summary_candidates(
                candidates=candidates,
                query=query,
                intent_context=intent_context,
                start=start,
                end=end,
                max_results=max_results,
                conversation_id=conversation_id,
            )

        if not candidates:
            return []

        candidate_items = list(candidates.values())
        signaled_candidates = [
            item for item in candidate_items
            if self._segment_candidate_has_query_signal(item)
        ]
        if self._should_prune_window_only_candidates(
            query_terms=query_terms,
            mode=mode,
            signaled_count=len(signaled_candidates),
        ):
            candidate_items = signaled_candidates

        for bucket in candidate_items:
            bucket["score"] += len(bucket["matched_terms"]) * 0.75
            if mode in _REMEMBER_WHEN_STATE_MODES:
                quote = bucket["quote"]
                try:
                    seg = self._store.get_segment(quote.segment_ref, conversation_id=conversation_id)
                except Exception:
                    seg = None
                if seg is not None and seg.summary:
                    bucket["summary_text"] = seg.summary

        if mode in _REMEMBER_WHEN_STATE_MODES and target_date is not None:
            selected = self._select_state_candidates(
                candidate_items,
                max_results=max_results,
                target_date=target_date,
            )
        elif mode in _REMEMBER_WHEN_CHANGE_MODES:
            selected = self._select_change_candidates(
                candidate_items,
                max_results=max_results,
            )
        elif mode in _REMEMBER_WHEN_WINDOW_OVERVIEW_MODES:
            selected = self._select_time_diverse_candidates(
                candidate_items,
                max_results=max_results,
                prefer_timeline=True,
                day_depth=2,
                prefer_concept_diversity=True,
            )
        else:
            selected = self._select_time_diverse_candidates(
                candidate_items,
                max_results=max_results,
                prefer_timeline=prefer_timeline,
            )

        results: list[dict] = []
        for idx, item in enumerate(selected):
            quote = item["quote"]
            excerpt = item.get("summary_text") or quote.text
            if "summary_text" not in item:
                try:
                    seg = self._store.get_segment(quote.segment_ref, conversation_id=conversation_id)
                except Exception:
                    seg = None
                if seg is not None and seg.summary:
                    excerpt = seg.summary
            if mode in _REMEMBER_WHEN_CHANGE_MODES:
                excerpt = self._change_result_excerpt(str(excerpt or ""))
            result = {
                "excerpt": excerpt,
                "topic": quote.tag,
                "segment_ref": quote.segment_ref,
                "session": quote.session_date,
                "session_date_normalized": item["date"].isoformat(),
                "match_type": quote.match_type,
                "matched_terms": sorted(item["matched_terms"]),
            }
            if target_date is not None and mode in _REMEMBER_WHEN_STATE_MODES:
                result.update({
                    "date_distance_days": (item["date"] - target_date).days,
                    "as_of_target": item["date"] <= target_date,
                    "state_anchor": idx == 0,
                })
            results.append(result)
        return results

    def _prune_change_search_variants(
        self,
        search_variants: list[tuple[str, float, set[str]]],
    ) -> list[tuple[str, float, set[str]]]:
        pruned = [
            item for item in search_variants
            if " " in item[0]
        ]
        return pruned or search_variants

    def _merge_semantic_summary_candidates(
        self,
        *,
        candidates: dict[str, dict],
        query: str,
        intent_context: str,
        start: date,
        end: date,
        max_results: int,
        conversation_id: str | None,
    ) -> None:
        embed_fn_getter = getattr(self._semantic, "get_embed_fn", None)
        if not callable(embed_fn_getter):
            return
        embed_fn = embed_fn_getter()
        if embed_fn is None:
            return

        semantic_queries = self._semantic_query_texts(query=query, intent_context=intent_context)
        if not semantic_queries:
            return

        try:
            segments = self._store.get_all_segments(conversation_id=conversation_id)
        except Exception as exc:
            logger.warning("remember_when semantic segment load failed: %s", exc)
            return

        segment_rows: list[tuple[StoredSegment, date, str, str, tuple[str, str, str]]] = []
        uncached_texts: list[str] = []
        uncached_keys: list[tuple[str, str, str]] = []
        for seg in segments:
            parsed = self._parse_session_date(seg.metadata.session_date)
            if parsed is None or not (start <= parsed <= end):
                continue
            summary_text = str(seg.summary or "").strip()
            if not summary_text:
                continue
            semantic_text = " ".join(
                part for part in [
                    str(seg.primary_tag or "").strip(),
                    " ".join(str(tag).strip() for tag in (seg.tags or []) if str(tag).strip()),
                    summary_text,
                ]
                if part
            ).strip()
            if not semantic_text:
                continue
            cache_key = (
                str(conversation_id or ""),
                str(seg.ref),
                semantic_text,
            )
            segment_rows.append((seg, parsed, summary_text, semantic_text, cache_key))
            if not self._cache_contains(cache_key):
                uncached_keys.append(cache_key)
                uncached_texts.append(semantic_text[:2000])

        if not segment_rows:
            return

        if uncached_texts:
            try:
                embeddings = embed_fn(uncached_texts)
            except Exception as exc:
                logger.warning("remember_when semantic summary embedding failed: %s", exc)
                return
            for key, embedding in zip(uncached_keys, embeddings):
                self._cache_set(key, list(embedding))

        try:
            query_embeddings = [list(vec) for vec in embed_fn([text[:2000] for text in semantic_queries])]
        except Exception as exc:
            logger.warning("remember_when semantic query embedding failed: %s", exc)
            return

        semantic_terms = {
            term
            for text in semantic_queries
            for term in _query_priority_terms(text)
        }
        ranked: list[tuple[float, StoredSegment, date, str]] = []
        for seg, parsed, summary_text, semantic_text, cache_key in segment_rows:
            segment_vec = self._cache_get(cache_key)
            if not segment_vec:
                continue
            similarity = max(
                cosine_similarity(query_vec, segment_vec)
                for query_vec in query_embeddings
            )
            ranked.append((similarity, seg, parsed, summary_text))

        if not ranked:
            return

        ranked.sort(key=lambda item: item[0], reverse=True)
        semantic_limit = max(max_results * 8, 60)
        for similarity, seg, parsed, summary_text in ranked[:semantic_limit]:
            if similarity < 0.22:
                continue
            segment_ref = str(seg.ref or "").strip()
            if not segment_ref:
                continue
            semantic_match_text = " ".join([
                str(seg.primary_tag or ""),
                " ".join(str(tag).strip() for tag in (seg.tags or []) if str(tag).strip()),
                summary_text,
            ]).lower()
            overlap = {
                term for term in semantic_terms
                if term in semantic_match_text
            }
            bucket = candidates.setdefault(
                segment_ref,
                {
                    "quote": SimpleNamespace(
                        segment_ref=segment_ref,
                        tag=str(seg.primary_tag or ""),
                        tags=list(seg.tags or []),
                        text=summary_text,
                        match_type="summary_semantic",
                        session_date=str(seg.metadata.session_date or ""),
                        created_at=str(seg.created_at or ""),
                    ),
                    "date": parsed,
                    "score": 0.0,
                    "matched_terms": set(),
                },
            )
            bucket["matched_terms"].update(overlap)
            bucket["score"] += similarity * 8.0 + len(overlap) * 0.5 + _TIMELINE_SUMMARY_BONUS
            if summary_text:
                bucket["summary_text"] = summary_text
            if bucket["quote"].match_type != "summary":
                bucket["quote"] = SimpleNamespace(
                    segment_ref=segment_ref,
                    tag=str(seg.primary_tag or ""),
                    tags=list(seg.tags or []),
                    text=summary_text,
                    match_type="summary_semantic",
                    session_date=str(seg.metadata.session_date or ""),
                    created_at=str(seg.created_at or ""),
                )

    def _semantic_query_texts(self, *, query: str, intent_context: str) -> list[str]:
        texts: list[str] = []
        seen: set[str] = set()
        for raw in [intent_context, query]:
            cleaned = str(raw or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            texts.append(cleaned)
        return texts

    def _search_facts_in_window(
        self,
        *,
        query: str,
        search_variants: list[tuple[str, float, set[str]]],
        start: date,
        end: date,
        max_results: int,
        prefer_timeline: bool,
        mode: str = "auto",
        segment_results: list[dict] | None = None,
        target_date: date | None = None,
    ) -> list[dict]:
        conversation_id = self._config.conversation_id or None
        fact_limit = max(max_results * 4, 25)
        per_variant_limit = min(max(max_results * 8, 25), 100)
        candidates: dict[str, dict] = {}
        query_terms = self._window_query_terms(query, search_variants)
        broad_fact_mode = mode in _REMEMBER_WHEN_BROAD_FACT_MODES
        state_mode = mode in _REMEMBER_WHEN_STATE_MODES

        try:
            raw_facts = self._store.query_experience_facts_by_date(
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                limit=self._window_fact_scan_limit(
                    max_results,
                    broad_mode=broad_fact_mode,
                    state_mode=state_mode,
                ),
                conversation_id=conversation_id,
            )
        except Exception as exc:
            logger.warning("remember_when fact fallback failed: %s", exc)
            raw_facts = []

        segment_refs = {
            str(item.get("segment_ref", "")).strip()
            for item in (segment_results or [])
            if str(item.get("segment_ref", "")).strip()
        }
        segment_dates = {
            str(item.get("session_date_normalized", "")).strip()
            for item in (segment_results or [])
            if str(item.get("session_date_normalized", "")).strip()
        }
        for fact in raw_facts:
            parsed = self._parse_fact_date(fact.when_date or fact.session_date)
            if parsed is None:
                continue
            text = self._fact_search_text(fact)
            overlap = {term for term in query_terms if term in text}
            segment_ref_bonus = 2.0 if fact.segment_ref and fact.segment_ref in segment_refs else 0.0
            segment_date_bonus = 0.75 if parsed.isoformat() in segment_dates else 0.0
            state_value_bonus = 0.0
            if state_mode and target_date is not None:
                state_value_bonus = self._value_signal_bonus(text) * 0.35
            self._add_fact_candidate(
                candidates,
                fact=fact,
                parsed=parsed,
                matched_terms=set(),
                soft_terms=overlap,
                score_delta=float(len(overlap)) + segment_ref_bonus + segment_date_bonus + state_value_bonus,
                source="window_scan",
            )

        for variant, base_weight, matched_terms in search_variants:
            try:
                hits = self._store.search_facts(
                    variant,
                    limit=per_variant_limit,
                    conversation_id=conversation_id,
                )
            except Exception as exc:
                logger.warning("remember_when fact search failed for '%s': %s", variant, exc)
                continue

            for fact in hits:
                parsed = self._parse_fact_date(fact.when_date or fact.session_date)
                if parsed is None or not (start <= parsed <= end):
                    continue
                text = self._fact_search_text(fact)
                overlap_bonus = sum(1 for term in matched_terms if term in text)
                self._add_fact_candidate(
                    candidates,
                    fact=fact,
                    parsed=parsed,
                    matched_terms=matched_terms,
                    soft_terms=set(),
                    score_delta=base_weight + overlap_bonus * 0.5,
                    source="query_match",
                )

        if broad_fact_mode or state_mode:
            self._expand_associative_fact_candidates(
                candidates=candidates,
                segment_results=segment_results or [],
                query_terms=query_terms,
                start=start,
                end=end,
                max_results=max_results,
                conversation_id=conversation_id,
                state_target_date=target_date if state_mode else None,
            )

        if not candidates:
            return []

        candidate_items = list(candidates.values())
        signaled_candidates = [
            item for item in candidate_items
            if self._fact_candidate_has_query_signal(item)
        ]
        if self._should_prune_window_only_candidates(
            query_terms=query_terms,
            mode=mode,
            signaled_count=len(signaled_candidates),
        ):
            candidate_items = signaled_candidates

        if state_mode:
            date_signal_counts: dict[date, int] = {}
            for item in candidate_items:
                soft_terms = set(item.get("soft_terms", set()) or set())
                if (
                    item.get("matched_terms")
                    or soft_terms
                    or self._value_signal_bonus(self._candidate_text(item)) > 0.0
                ):
                    date_signal_counts[item["date"]] = date_signal_counts.get(item["date"], 0) + 1
            for item in candidate_items:
                bundle_count = date_signal_counts.get(item["date"], 0)
                if bundle_count > 1:
                    item["score"] += 2.5 * (bundle_count - 1)

        if state_mode and segment_results:
            anchor_excerpt = str(segment_results[0].get("excerpt", "") or "")
            anchor_date = str(segment_results[0].get("session_date_normalized", "") or "")
            for bucket in candidate_items:
                fact_text = self._candidate_text(bucket)
                bucket["score"] += self._anchor_text_bonus(fact_text, anchor_excerpt)
                if anchor_date and bucket["date"].isoformat() == anchor_date:
                    bucket["score"] += 0.75

        for bucket in candidate_items:
            bucket["score"] += len(bucket["matched_terms"]) * 0.75

        if state_mode and target_date is not None:
            selected = self._select_state_candidates(
                candidate_items,
                max_results=fact_limit,
                target_date=target_date,
            )
        elif mode in _REMEMBER_WHEN_CHANGE_MODES:
            selected = self._select_change_candidates(
                candidate_items,
                max_results=max(max_results * 4, 24),
            )
        elif mode in _REMEMBER_WHEN_WINDOW_OVERVIEW_MODES:
            selected = self._select_time_diverse_candidates(
                candidate_items,
                max_results=fact_limit,
                prefer_timeline=True,
                day_depth=2,
                prefer_concept_diversity=True,
            )
        else:
            selected = self._select_time_diverse_candidates(
                candidate_items,
                max_results=fact_limit,
                prefer_timeline=prefer_timeline,
                day_depth=2 if broad_fact_mode else 1,
                prefer_concept_diversity=broad_fact_mode,
            )

        results: list[dict] = []
        for idx, item in enumerate(selected):
            result = {
                "type": "fact",
                "what": item["fact"].what or f"{item['fact'].subject} {item['fact'].verb} {item['fact'].object}",
                "when": item["date"].isoformat(),
                "where": item["fact"].where or "",
                "status": item["fact"].status or "",
                "tags": list(getattr(item["fact"], "tags", []) or []),
                "matched_terms": sorted(item["matched_terms"]),
                "segment_ref": getattr(item["fact"], "segment_ref", "") or "",
            }
            if target_date is not None and state_mode:
                result.update({
                    "date_distance_days": (item["date"] - target_date).days,
                    "as_of_target": item["date"] <= target_date,
                    "state_anchor": idx == 0 and not results,
                })
            results.append(result)
        return results

    def _supplement_change_results_for_fact_backed_dates(
        self,
        *,
        query: str,
        intent_context: str,
        results: list[dict],
        facts: list[dict],
        start: date,
        end: date,
    ) -> list[dict]:
        if not facts:
            return results

        results_by_date: dict[str, list[dict]] = {}
        existing_refs: set[str] = set()
        for item in results:
            date_key = str(item.get("session_date_normalized", "") or "").strip()
            if date_key:
                results_by_date.setdefault(date_key, []).append(item)
            segment_ref = str(item.get("segment_ref", "") or "").strip()
            if segment_ref:
                existing_refs.add(segment_ref)

        facts_by_date: dict[str, dict[str, object]] = {}
        for item in facts:
            date_key = str(item.get("when", "") or "").strip()
            if not date_key:
                continue
            bucket = facts_by_date.setdefault(
                date_key,
                {
                    "facts": [],
                    "tags": set(),
                    "segment_refs": set(),
                },
            )
            bucket["facts"].append(item)
            for tag in item.get("tags", []) or []:
                cleaned = str(tag).strip().lower()
                if cleaned:
                    bucket["tags"].add(cleaned)
            segment_ref = str(item.get("segment_ref", "") or "").strip()
            if segment_ref:
                bucket["segment_refs"].add(segment_ref)

        target_dates = [
            date_key
            for date_key, bucket in sorted(facts_by_date.items())
            if not results_by_date.get(date_key)
            and (
                len(bucket["facts"]) >= 2
                or len(bucket["tags"]) >= 2
                or len(bucket["segment_refs"]) >= 2
            )
        ]
        if not target_dates:
            return results

        try:
            segments = self._store.get_all_segments(
                conversation_id=self._config.conversation_id or None,
            )
        except Exception as exc:
            logger.warning("remember_when date expansion segment load failed: %s", exc)
            return results
        if not segments:
            return results

        query_terms = set(_query_priority_terms(query, drop_generic=False))
        query_terms.update(_query_priority_terms(intent_context, drop_generic=False))
        semantic_queries = self._semantic_query_texts(query=query, intent_context=intent_context)
        query_embeddings: list[list[float]] = []
        embed_fn = None
        embed_fn_getter = getattr(self._semantic, "get_embed_fn", None)
        if callable(embed_fn_getter):
            embed_fn = embed_fn_getter()
        if embed_fn and semantic_queries:
            try:
                query_embeddings = [
                    list(vec)
                    for vec in embed_fn([text[:2000] for text in semantic_queries])
                ]
            except Exception as exc:
                logger.warning("remember_when date expansion query embedding failed: %s", exc)
                query_embeddings = []

        segment_rows: list[tuple[object, str, str, str, tuple[str, str, str]]] = []
        uncached_texts: list[str] = []
        uncached_keys: list[tuple[str, str, str]] = []
        target_date_set = set(target_dates)
        for seg in segments:
            parsed = self._parse_session_date(seg.metadata.session_date)
            if parsed is None or not (start <= parsed <= end):
                continue
            date_key = parsed.isoformat()
            if date_key not in target_date_set:
                continue
            segment_ref = str(seg.ref or "").strip()
            if not segment_ref or segment_ref in existing_refs:
                continue
            summary_text = str(seg.summary or "").strip()
            if not summary_text:
                continue
            semantic_text = " ".join(
                part for part in [
                    str(seg.primary_tag or "").strip(),
                    " ".join(str(tag).strip() for tag in (seg.tags or []) if str(tag).strip()),
                    summary_text,
                ]
                if part
            ).strip()
            if not semantic_text:
                continue
            cache_key = (
                str(self._config.conversation_id or ""),
                str(seg.ref),
                semantic_text,
            )
            segment_rows.append((seg, date_key, summary_text, semantic_text, cache_key))
            if embed_fn and not self._cache_contains(cache_key):
                uncached_keys.append(cache_key)
                uncached_texts.append(semantic_text[:2000])

        if not segment_rows:
            return results

        if embed_fn and uncached_texts:
            try:
                embeddings = embed_fn(uncached_texts)
            except Exception as exc:
                logger.warning("remember_when date expansion segment embedding failed: %s", exc)
                embeddings = []
            for key, embedding in zip(uncached_keys, embeddings):
                self._cache_set(key, list(embedding))

        candidates_by_date: dict[str, list[tuple[float, dict]]] = {}
        for seg, date_key, summary_text, semantic_text, cache_key in segment_rows:
            fact_bucket = facts_by_date.get(date_key)
            if fact_bucket is None:
                continue
            segment_tags = {
                str(seg.primary_tag or "").strip().lower(),
                *(
                    str(tag).strip().lower()
                    for tag in (seg.tags or [])
                    if str(tag).strip()
                ),
            }
            segment_tags.discard("")
            overlap_terms = {
                term
                for term in query_terms
                if term in semantic_text.lower()
            }
            similarity = 0.0
            if query_embeddings:
                segment_vec = self._cache_get(cache_key)
                if segment_vec:
                    similarity = max(
                        cosine_similarity(query_vec, segment_vec)
                        for query_vec in query_embeddings
                    )
            segment_ref = str(seg.ref or "").strip()
            ref_bonus = 2.5 if segment_ref in fact_bucket["segment_refs"] else 0.0
            tag_bonus = 0.8 * len(segment_tags & fact_bucket["tags"])
            score = ref_bonus + tag_bonus + similarity * 6.0 + 0.25 * len(overlap_terms)
            if (
                ref_bonus <= 0.0
                and tag_bonus <= 0.0
                and similarity < 0.22
            ):
                continue
            matched_terms = sorted(overlap_terms)
            candidates_by_date.setdefault(date_key, []).append(
                (
                    score,
                    {
                        "excerpt": summary_text,
                        "topic": str(seg.primary_tag or ""),
                        "segment_ref": segment_ref,
                        "session": str(seg.metadata.session_date or ""),
                        "session_date_normalized": date_key,
                        "match_type": "summary_date_expand",
                        "matched_terms": matched_terms,
                    },
                )
            )

        if not candidates_by_date:
            return results

        supplemented = list(results)
        for date_key in target_dates:
            ranked = sorted(
                candidates_by_date.get(date_key, []),
                key=lambda item: item[0],
                reverse=True,
            )
            if not ranked:
                continue
            supplemented.append(ranked[0][1])
        supplemented.sort(
            key=lambda item: (
                str(item.get("session_date_normalized", "") or ""),
                str(item.get("topic", "") or ""),
                str(item.get("segment_ref", "") or ""),
            ),
        )
        return supplemented

    def _change_result_excerpt(self, text: str, *, limit: int = 220) -> str:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return ""

        match = re.search(r"(?<=[.!?])\s+", cleaned)
        if match:
            first_sentence = cleaned[:match.start()].strip()
            if first_sentence:
                return first_sentence

        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3].rstrip() + "..."

    def _select_time_diverse_candidates(
        self,
        candidates: list[dict],
        *,
        max_results: int,
        prefer_timeline: bool,
        day_depth: int = 1,
        prefer_concept_diversity: bool = False,
    ) -> list[dict]:
        if not candidates or max_results <= 0:
            return []

        by_day: dict[date, list[dict]] = {}
        for item in candidates:
            by_day.setdefault(item["date"], []).append(item)

        days = sorted(by_day)
        for day in days:
            if prefer_timeline:
                by_day[day].sort(
                    key=lambda item: (
                        item["score"],
                        len(item["matched_terms"]),
                    ),
                    reverse=True,
                )
            else:
                by_day[day].sort(
                    key=lambda item: (
                        item["score"],
                        len(item["matched_terms"]),
                        getattr(item.get("quote"), "created_at", "") or "",
                    ),
                    reverse=True,
                )

        chosen: list[dict] = []
        chosen_ids: set[str] = set()

        if prefer_timeline:
            if len(days) <= max_results:
                day_indices = list(range(len(days)))
            else:
                day_indices = self._evenly_spaced_indices(len(days), max_results)
            target_depth = max(1, day_depth)
            for depth_idx in range(target_depth):
                for idx in day_indices:
                    day_items = by_day[days[idx]]
                    if depth_idx >= len(day_items):
                        continue
                    item = day_items[depth_idx]
                    item_id = self._candidate_id(item)
                    if item_id in chosen_ids:
                        continue
                    chosen.append(item)
                    chosen_ids.add(item_id)
                    if len(chosen) >= max_results:
                        break
                if len(chosen) >= max_results:
                    break
        else:
            best_per_day = [by_day[day][0] for day in days]
            best_per_day.sort(
                key=lambda item: (
                    item["score"],
                    len(item["matched_terms"]),
                    item["date"],
                ),
                reverse=True,
            )
            for item in best_per_day[:max_results]:
                item_id = self._candidate_id(item)
                chosen.append(item)
                chosen_ids.add(item_id)

        if len(chosen) < max_results:
            remainder: list[dict] = []
            for day in days:
                remainder.extend(by_day[day])
            if prefer_concept_diversity:
                seen_tags = set().union(*(self._candidate_tags(item) for item in chosen))
                seen_segments = {
                    segment_ref
                    for segment_ref in (self._candidate_segment_ref(item) for item in chosen)
                    if segment_ref
                }
                pool = [
                    item for item in remainder
                    if self._candidate_id(item) not in chosen_ids
                ]
                while pool and len(chosen) < max_results:
                    best = max(
                        pool,
                        key=lambda item: (
                            item["score"] + self._novelty_bonus(item, seen_tags, seen_segments),
                            self._novelty_bonus(item, seen_tags, seen_segments),
                            len(item["matched_terms"]),
                            item["date"],
                        ),
                    )
                    item_id = self._candidate_id(best)
                    chosen.append(best)
                    chosen_ids.add(item_id)
                    seen_tags.update(self._candidate_tags(best))
                    segment_ref = self._candidate_segment_ref(best)
                    if segment_ref:
                        seen_segments.add(segment_ref)
                    pool = [item for item in pool if self._candidate_id(item) != item_id]
            else:
                remainder.sort(
                    key=lambda item: (
                        item["score"],
                        len(item["matched_terms"]),
                        item["date"],
                    ),
                    reverse=True,
                )
                for item in remainder:
                    item_id = self._candidate_id(item)
                    if item_id in chosen_ids:
                        continue
                    chosen.append(item)
                    chosen_ids.add(item_id)
                    if len(chosen) >= max_results:
                        break

        chosen.sort(key=lambda item: item["date"])
        return chosen[:max_results]

    def _add_fact_candidate(
        self,
        candidates: dict[str, dict],
        *,
        fact,
        parsed: date,
        matched_terms: set[str],
        soft_terms: set[str] | None = None,
        score_delta: float,
        source: str,
    ) -> None:
        bucket = candidates.setdefault(
            fact.id,
            {
                "fact": fact,
                "date": parsed,
                "score": 0.0,
                "matched_terms": set(),
                "soft_terms": set(),
                "sources": set(),
            },
        )
        bucket["matched_terms"].update(matched_terms)
        if soft_terms:
            bucket["soft_terms"].update(soft_terms)
        bucket["score"] += score_delta
        if source:
            bucket["sources"].add(source)

    def _expand_associative_fact_candidates(
        self,
        *,
        candidates: dict[str, dict],
        segment_results: list[dict],
        query_terms: set[str],
        start: date,
        end: date,
        max_results: int,
        conversation_id: str | None,
        state_target_date: date | None = None,
    ) -> None:
        anchor_segment_refs: list[str] = []
        seen_segment_refs: set[str] = set()
        for item in segment_results:
            segment_ref = str(item.get("segment_ref", "")).strip()
            if not segment_ref or segment_ref in seen_segment_refs:
                continue
            anchor_segment_refs.append(segment_ref)
            seen_segment_refs.add(segment_ref)

        ranked_seed_candidates = sorted(
            candidates.values(),
            key=lambda item: (
                item["score"],
                len(item["matched_terms"]),
                item["date"],
            ),
            reverse=True,
        )
        seed_limit = max(max_results * 2, 8)
        for item in ranked_seed_candidates[:seed_limit]:
            segment_ref = self._candidate_segment_ref(item)
            if not segment_ref or segment_ref in seen_segment_refs:
                continue
            anchor_segment_refs.append(segment_ref)
            seen_segment_refs.add(segment_ref)

        anchor_segment_refs = anchor_segment_refs[: max(max_results * 2, 12)]
        anchor_dates = {
            str(item.get("session_date_normalized", "")).strip()
            for item in segment_results
            if str(item.get("session_date_normalized", "")).strip()
        }
        anchor_dates.update(item["date"].isoformat() for item in ranked_seed_candidates[:seed_limit])
        anchor_tags = {
            tag.lower()
            for item in ranked_seed_candidates[:seed_limit]
            for tag in getattr(item["fact"], "tags", []) or []
            if str(tag).strip()
        }
        anchor_fact_ids = [
            item["fact"].id
            for item in ranked_seed_candidates[: max(max_results * 3, 12)]
            if getattr(item.get("fact"), "id", "")
        ]

        for segment_ref in anchor_segment_refs:
            try:
                sibling_facts = self._store.get_facts_by_segment(segment_ref)
            except Exception as exc:
                logger.warning("remember_when sibling fact expansion failed for %s: %s", segment_ref, exc)
                continue
            for fact in sibling_facts:
                if conversation_id and fact.conversation_id and fact.conversation_id != conversation_id:
                    continue
                parsed = self._parse_fact_date(fact.when_date or fact.session_date)
                if parsed is None or not (start <= parsed <= end):
                    continue
                text = self._fact_search_text(fact)
                overlap = {term for term in query_terms if term in text}
                tag_bonus = self._tag_overlap_bonus(fact.tags, anchor_tags)
                date_bonus = 0.5 if parsed.isoformat() in anchor_dates else 0.0
                if state_target_date is not None:
                    date_bonus += self._state_temporal_bonus(parsed, state_target_date) * 0.35
                self._add_fact_candidate(
                    candidates,
                    fact=fact,
                    parsed=parsed,
                    matched_terms=overlap,
                    soft_terms=set(),
                    score_delta=float(len(overlap)) + 2.25 + tag_bonus + date_bonus,
                    source="segment_neighbor",
                )

        if not anchor_fact_ids or not hasattr(self._store, "get_linked_facts"):
            return

        try:
            linked_facts = self._store.get_linked_facts(anchor_fact_ids, depth=1)
        except Exception as exc:
            logger.warning("remember_when linked fact expansion failed: %s", exc)
            return

        for linked in linked_facts:
            fact = linked.fact
            if conversation_id and fact.conversation_id and fact.conversation_id != conversation_id:
                continue
            parsed = self._parse_fact_date(fact.when_date or fact.session_date)
            if parsed is None or not (start <= parsed <= end):
                continue
            text = self._fact_search_text(fact)
            overlap = {term for term in query_terms if term in text}
            same_segment_bonus = 1.0 if fact.segment_ref and fact.segment_ref in seen_segment_refs else 0.0
            tag_bonus = self._tag_overlap_bonus(fact.tags, anchor_tags)
            date_bonus = 0.5 if parsed.isoformat() in anchor_dates else 0.0
            if state_target_date is not None:
                date_bonus += self._state_temporal_bonus(parsed, state_target_date) * 0.25
            if query_terms and not overlap and same_segment_bonus <= 0.0 and tag_bonus <= 0.0 and date_bonus <= 0.0:
                continue
            confidence = max(0.0, min(1.0, float(getattr(linked, "confidence", 0.0) or 0.0)))
            self._add_fact_candidate(
                candidates,
                fact=fact,
                parsed=parsed,
                matched_terms=overlap,
                soft_terms=set(),
                score_delta=float(len(overlap)) + 1.0 + confidence + same_segment_bonus + tag_bonus + date_bonus,
                source="linked_fact",
            )

    def _query_terms_from_variants(
        self,
        search_variants: list[tuple[str, float, set[str]]],
    ) -> set[str]:
        return {
            term
            for _variant, _weight, terms in search_variants
            for term in terms
        }

    def _fact_search_text(self, fact) -> str:
        return " ".join([
            fact.subject or "",
            fact.verb or "",
            fact.object or "",
            fact.what or "",
            " ".join(fact.tags or []),
        ]).lower()

    def _tag_overlap_bonus(self, fact_tags: list[str], anchor_tags: set[str]) -> float:
        if not fact_tags or not anchor_tags:
            return 0.0
        overlap = {
            str(tag).strip().lower()
            for tag in fact_tags
            if str(tag).strip().lower() in anchor_tags
        }
        if not overlap:
            return 0.0
        return min(1.5, 0.4 * len(overlap))

    def _window_fact_scan_limit(
        self,
        max_results: int,
        *,
        broad_mode: bool,
        state_mode: bool = False,
    ) -> int:
        if broad_mode:
            return max(max_results * 500, 10_000)
        if state_mode:
            return max(max_results * 40, 400)
        return max(max_results * 20, 200)

    def _build_summary_highlights(
        self,
        *,
        query: str,
        results: list[dict],
        facts: list[dict],
        max_results: int,
    ) -> list[dict]:
        if max_results <= 0:
            return []

        query_terms = set(_query_priority_terms(query))

        candidates: list[dict] = []
        for item in results:
            excerpt = str(item.get("excerpt", "") or "").strip()
            if not excerpt:
                continue
            topic = str(item.get("topic", "") or "").strip()
            matched_terms = {
                str(term).strip().lower()
                for term in item.get("matched_terms", []) or []
                if str(term).strip()
            }
            theme_tokens = self._highlight_theme_tokens(
                text=excerpt,
                tags=[topic] if topic else [],
            )
            candidates.append(
                {
                    "source": "segment",
                    "date": str(item.get("session_date_normalized", "") or ""),
                    "theme": self._highlight_theme_label(excerpt, [topic] if topic else []),
                    "point": self._trim_state_text(excerpt, limit=220),
                    "theme_tokens": theme_tokens,
                    "score": 4.0
                    + 0.9 * len(matched_terms & query_terms)
                    + 0.35 * min(4, len(theme_tokens))
                    + (0.8 if topic else 0.0),
                }
            )

        for item in facts:
            what = str(item.get("what", "") or "").strip()
            if not what:
                continue
            tags = [str(tag).strip() for tag in item.get("tags", []) or [] if str(tag).strip()]
            matched_terms = {
                str(term).strip().lower()
                for term in item.get("matched_terms", []) or []
                if str(term).strip()
            }
            theme_tokens = self._highlight_theme_tokens(text=what, tags=tags)
            candidates.append(
                {
                    "source": "fact",
                    "date": str(item.get("when", "") or ""),
                    "theme": self._highlight_theme_label(what, tags),
                    "point": self._trim_state_text(what, limit=220),
                    "theme_tokens": theme_tokens,
                    "score": 3.1
                    + 0.8 * len(matched_terms & query_terms)
                    + 0.45 * min(5, len(theme_tokens))
                    + 0.35 * len(tags),
                }
            )

        if not candidates:
            return []

        max_highlights = min(max(max_results, 5), 8)
        selected: list[dict] = []
        seen_tokens: set[str] = set()
        seen_dates: set[str] = set()
        remaining = candidates[:]
        while remaining and len(selected) < max_highlights:
            best = max(
                remaining,
                key=lambda item: (
                    item["score"]
                    + 0.85 * len(item["theme_tokens"] - seen_tokens)
                    + (0.35 if item["date"] and item["date"] not in seen_dates else 0.0),
                    len(item["theme_tokens"] - seen_tokens),
                    item["score"],
                    item["date"],
                ),
            )
            selected.append(best)
            seen_tokens.update(best["theme_tokens"])
            if best["date"]:
                seen_dates.add(best["date"])
            remaining.remove(best)

        selected.sort(key=lambda item: (item["date"], item["source"], item["theme"]))
        return [
            {
                "date": item["date"],
                "theme": item["theme"],
                "point": item["point"],
                "source": item["source"],
            }
            for item in selected
        ]

    def _build_fact_highlights(
        self,
        *,
        query: str,
        facts: list[dict],
        max_results: int,
    ) -> list[dict]:
        if max_results <= 0:
            return []

        query_terms = set(_query_priority_terms(query))

        candidates: list[dict] = []
        for item in facts:
            what = str(item.get("what", "") or "").strip()
            if not what:
                continue
            tags = [str(tag).strip() for tag in item.get("tags", []) or [] if str(tag).strip()]
            matched_terms = {
                str(term).strip().lower()
                for term in item.get("matched_terms", []) or []
                if str(term).strip()
            }
            theme_tokens = self._highlight_theme_tokens(text=what, tags=tags)
            candidates.append(
                {
                    "date": str(item.get("when", "") or ""),
                    "point": self._trim_state_text(what, limit=220),
                    "tags": tags,
                    "theme_tokens": theme_tokens,
                    "score": 3.1
                    + 0.8 * len(matched_terms & query_terms)
                    + 0.45 * min(5, len(theme_tokens))
                    + 0.35 * len(tags),
                }
            )

        if not candidates:
            return []

        selected = sorted(
            candidates,
            key=lambda item: (
                item["score"],
                len(item["tags"]),
                item["date"],
            ),
            reverse=True,
        )[: min(max(max_results, 5), 8)]
        selected.sort(key=lambda item: item["date"])
        return [
            {
                "date": item["date"],
                "point": item["point"],
                "tags": item["tags"],
                "source": "fact",
            }
            for item in selected
        ]

    def _build_ordered_milestones(
        self,
        *,
        query: str,
        results: list[dict],
        facts: list[dict],
        max_results: int,
    ) -> list[dict]:
        if max_results <= 0:
            return []

        query_terms = set(_query_priority_terms(query))

        segment_candidates: list[dict] = []
        for item in results:
            excerpt = str(item.get("excerpt", "") or "").strip()
            if not excerpt:
                continue
            topic = str(item.get("topic", "") or "").strip()
            matched_terms = {
                str(term).strip().lower()
                for term in item.get("matched_terms", []) or []
                if str(term).strip()
            }
            theme_tokens = self._highlight_theme_tokens(
                text=excerpt,
                tags=[topic] if topic else [],
            )
            segment_candidates.append(
                {
                    "date": str(item.get("session_date_normalized", "") or ""),
                    "theme": self._highlight_theme_label(excerpt, [topic] if topic else []),
                    "point": self._trim_state_text(excerpt, limit=220),
                    "source": "segment",
                    "theme_tokens": theme_tokens,
                    "score": 2.8
                    + 0.95 * len(matched_terms & query_terms)
                    + 0.35 * min(6, len(theme_tokens))
                    + (0.5 if topic else 0.0),
                }
            )

        fact_candidates: list[dict] = []
        for item in facts:
            what = str(item.get("what", "") or "").strip()
            if not what:
                continue
            tags = [str(tag).strip() for tag in item.get("tags", []) or [] if str(tag).strip()]
            matched_terms = {
                str(term).strip().lower()
                for term in item.get("matched_terms", []) or []
                if str(term).strip()
            }
            theme_tokens = self._highlight_theme_tokens(text=what, tags=tags)
            fact_candidates.append(
                {
                    "date": str(item.get("when", "") or ""),
                    "theme": self._highlight_theme_label(what, tags),
                    "point": self._trim_state_text(what, limit=220),
                    "source": "fact",
                    "theme_tokens": theme_tokens,
                    "score": 3.2
                    + 0.85 * len(matched_terms & query_terms)
                    + 0.4 * min(6, len(theme_tokens))
                    + 0.12 * min(10, len(tags)),
                }
            )

        if not segment_candidates and not fact_candidates:
            return []

        selected: list[dict] = []
        seen_tokens: set[str] = set()
        by_date_segments: dict[str, list[dict]] = {}
        by_date_facts: dict[str, list[dict]] = {}
        for item in segment_candidates:
            by_date_segments.setdefault(item["date"], []).append(item)
        for item in fact_candidates:
            by_date_facts.setdefault(item["date"], []).append(item)

        def choose_best(pool: list[dict], *, prefer_segment: bool = False) -> dict:
            return max(
                pool,
                key=lambda item: (
                    item["score"] + 0.8 * len(item["theme_tokens"] - seen_tokens),
                    len(item["theme_tokens"] - seen_tokens),
                    item["score"],
                    1 if prefer_segment and item["source"] == "segment" else 0,
                ),
            )

        all_dates = sorted(set(by_date_segments) | set(by_date_facts))
        for date_key in all_dates:
            segment_pool = by_date_segments.get(date_key, [])
            fact_pool = by_date_facts.get(date_key, [])
            if segment_pool:
                best_segment = choose_best(segment_pool, prefer_segment=True)
                milestone = {
                    "date": best_segment["date"],
                    "theme": best_segment["theme"],
                    "point": best_segment["point"],
                    "source": "segment",
                }
                if fact_pool:
                    best_fact = choose_best(fact_pool)
                    if best_fact["point"] != best_segment["point"]:
                        milestone["supporting_point"] = best_fact["point"]
                selected.append(milestone)
                seen_tokens.update(best_segment["theme_tokens"])
                continue
            if fact_pool:
                best_fact = choose_best(fact_pool)
                selected.append(
                    {
                        "date": best_fact["date"],
                        "theme": best_fact["theme"],
                        "point": best_fact["point"],
                        "source": "fact",
                    }
                )
                seen_tokens.update(best_fact["theme_tokens"])

        selected.sort(key=lambda item: (item["date"], item["source"], item["theme"]))
        if len(selected) > max_results:
            spaced_indices = self._evenly_spaced_indices(len(selected), max_results)
            selected = [selected[idx] for idx in spaced_indices]
        return selected[:max_results]

    def _build_phase_milestones(
        self,
        *,
        ordered_milestones: list[dict],
    ) -> list[dict]:
        if len(ordered_milestones) < 6:
            return []

        phase_count = min(6, max(3, ceil(len(ordered_milestones) / 4)))
        chunk_size = max(1, ceil(len(ordered_milestones) / phase_count))
        phases: list[dict] = []

        for start_idx in range(0, len(ordered_milestones), chunk_size):
            chunk = ordered_milestones[start_idx:start_idx + chunk_size]
            if not chunk:
                continue

            seen_themes: set[str] = set()
            focus_parts: list[str] = []
            points: list[str] = []
            for item in chunk:
                theme = str(item.get("theme", "") or "").strip()
                if theme and theme not in seen_themes and len(focus_parts) < 3:
                    seen_themes.add(theme)
                    focus_parts.append(theme)

                point = str(item.get("point", "") or "").strip()
                if point:
                    points.append(point)
                supporting = str(item.get("supporting_point", "") or "").strip()
                if supporting and supporting != point:
                    points.append(supporting)

            deduped_points: list[str] = []
            seen_points: set[str] = set()
            for point in points:
                normalized = point.lower()
                if normalized in seen_points:
                    continue
                seen_points.add(normalized)
                deduped_points.append(self._trim_state_text(point, limit=180))
                if len(deduped_points) >= 3:
                    break

            phases.append(
                {
                    "start_date": chunk[0]["date"],
                    "end_date": chunk[-1]["date"],
                    "focus": ", ".join(focus_parts) if focus_parts else "timeline evolution",
                    "points": deduped_points,
                }
            )

        return phases

    def _select_change_candidates(
        self,
        candidates: list[dict],
        *,
        max_results: int,
    ) -> list[dict]:
        if not candidates or max_results <= 0:
            return []
        limit = max(max_results * 3, 24)
        # Chronology questions need broad date coverage more than repeated
        # same-day near-duplicates, so reuse the timeline selector here.
        return self._select_time_diverse_candidates(
            candidates,
            max_results=limit,
            prefer_timeline=True,
            day_depth=2,
            prefer_concept_diversity=False,
        )

    def _build_change_date_buckets(
        self,
        *,
        results: list[dict],
        facts: list[dict],
        max_results: int,
    ) -> list[dict]:
        if max_results <= 0:
            return []

        buckets: dict[str, dict[str, object]] = {}
        for item in results:
            date_key = str(item.get("session_date_normalized", "") or "").strip()
            if not date_key:
                continue
            bucket = buckets.setdefault(date_key, {"date": date_key, "results": [], "facts": []})
            bucket["results"].append({
                "topic": item.get("topic", ""),
                "excerpt": item.get("excerpt", ""),
                "matched_terms": item.get("matched_terms", []),
                "segment_ref": item.get("segment_ref", ""),
                "match_type": item.get("match_type", ""),
            })

        for item in facts:
            date_key = str(item.get("when", "") or "").strip()
            if not date_key:
                continue
            bucket = buckets.setdefault(date_key, {"date": date_key, "results": [], "facts": []})
            bucket["facts"].append({
                "what": item.get("what", ""),
                "tags": item.get("tags", []),
                "matched_terms": item.get("matched_terms", []),
                "segment_ref": item.get("segment_ref", ""),
            })

        selected: list[dict] = []
        for date_key in sorted(buckets):
            bucket = buckets[date_key]
            results_for_day = list(bucket["results"])[:_REMEMBER_WHEN_CHANGE_RESULTS_PER_DATE]
            facts_for_day = list(bucket["facts"])[:_REMEMBER_WHEN_CHANGE_FACTS_PER_DATE]
            if not results_for_day and not facts_for_day:
                continue
            selected.append(
                {
                    "date": date_key,
                    "results": results_for_day,
                    "facts": facts_for_day,
                }
            )

        if len(selected) > max_results:
            selected = selected[:max_results]
        return selected

    def _highlight_theme_tokens(self, *, text: str, tags: list[str]) -> set[str]:
        tokens: set[str] = set()
        for tag in tags:
            for token in _QUERY_TOKEN_RE.findall(tag.lower().replace("-", " ")):
                if (
                    len(token) >= 4
                    and token not in _QUERY_STOPWORDS
                    and token not in _GENERIC_QUERY_TERMS
                ):
                    tokens.add(token)
        for token in _QUERY_TOKEN_RE.findall(text.lower()):
            if (
                len(token) >= 4
                and token not in _QUERY_STOPWORDS
                and token not in _GENERIC_QUERY_TERMS
            ):
                tokens.add(token)
        return tokens

    def _highlight_theme_label(
        self,
        text: str,
        tags: list[str],
    ) -> str:
        for tag in tags:
            cleaned = str(tag).strip().replace("-", " ")
            if cleaned:
                return cleaned
        tokens = [
            token
            for token in _QUERY_TOKEN_RE.findall(text.lower())
            if (
                len(token) >= 4
                and token not in _QUERY_STOPWORDS
                and token not in _GENERIC_QUERY_TERMS
            )
        ]
        if not tokens:
            return "timeline development"
        return " ".join(tokens[:3])

    def _candidate_id(self, item: dict) -> str:
        quote = item.get("quote")
        if quote is not None:
            return quote.segment_ref
        fact = item.get("fact")
        return getattr(fact, "id", "")

    def _candidate_tags(self, item: dict) -> set[str]:
        quote = item.get("quote")
        if quote is not None:
            raw_tags = getattr(quote, "tags", []) or []
        else:
            raw_tags = getattr(item.get("fact"), "tags", []) or []
        return {
            str(tag).strip().lower()
            for tag in raw_tags
            if str(tag).strip()
        }

    def _candidate_segment_ref(self, item: dict) -> str:
        quote = item.get("quote")
        if quote is not None:
            return str(getattr(quote, "segment_ref", "") or "").strip()
        fact = item.get("fact")
        return str(getattr(fact, "segment_ref", "") or "").strip()

    def _novelty_bonus(
        self,
        item: dict,
        seen_tags: set[str],
        seen_segments: set[str],
    ) -> float:
        tags = self._candidate_tags(item)
        tag_bonus = min(2.0, 0.45 * len(tags - seen_tags))
        segment_ref = self._candidate_segment_ref(item)
        segment_bonus = 0.5 if segment_ref and segment_ref not in seen_segments else 0.0
        return tag_bonus + segment_bonus

    def _select_state_candidates(
        self,
        candidates: list[dict],
        *,
        max_results: int,
        target_date: date,
    ) -> list[dict]:
        if not candidates or max_results <= 0:
            return []

        ranked = sorted(
            candidates,
            key=lambda item: (
                self._state_anchor_score(item, target_date),
                item["score"],
                len(item["matched_terms"]),
            ),
            reverse=True,
        )
        anchor = ranked[0]
        chosen = [anchor]
        chosen_ids = {self._candidate_id(anchor)}
        anchor_date = anchor["date"]
        anchor_segment_ref = self._candidate_segment_ref(anchor)
        anchor_tags = self._candidate_tags(anchor)
        anchor_terms = set(anchor.get("matched_terms", set()) or set())

        remainder = [item for item in ranked[1:] if self._candidate_id(item) not in chosen_ids]
        while remainder and len(chosen) < max_results:
            best = max(
                remainder,
                key=lambda item: (
                    self._state_support_score(
                        item,
                        target_date=target_date,
                        anchor_date=anchor_date,
                        anchor_segment_ref=anchor_segment_ref,
                        anchor_tags=anchor_tags,
                        anchor_terms=anchor_terms,
                    ),
                    self._state_anchor_score(item, target_date),
                    item["score"],
                    len(item["matched_terms"]),
                ),
            )
            chosen.append(best)
            chosen_ids.add(self._candidate_id(best))
            remainder = [
                item
                for item in remainder
                if self._candidate_id(item) not in chosen_ids
            ]
        return chosen[:max_results]

    def _state_anchor_score(self, item: dict, target_date: date) -> float:
        matched_terms = set(item.get("matched_terms", set()) or set())
        text = self._candidate_text(item)
        coverage_bonus = 1.1 * len(matched_terms) * len(matched_terms)
        value_bonus = self._value_signal_bonus(text)
        temporal_bonus = self._state_temporal_bonus(item["date"], target_date)
        return item["score"] + coverage_bonus + value_bonus + temporal_bonus

    def _state_support_score(
        self,
        item: dict,
        *,
        target_date: date,
        anchor_date: date,
        anchor_segment_ref: str,
        anchor_tags: set[str],
        anchor_terms: set[str],
    ) -> float:
        matched_terms = set(item.get("matched_terms", set()) or set())
        text = self._candidate_text(item)
        score = item["score"] + self._value_signal_bonus(text) + 0.9 * len(matched_terms)
        if item["date"] == anchor_date:
            score += 3.0
        if anchor_segment_ref and self._candidate_segment_ref(item) == anchor_segment_ref:
            score += 1.5
        score += 1.25 * len(matched_terms & anchor_terms)
        score += self._tag_overlap_bonus(list(self._candidate_tags(item)), anchor_tags)
        score += max(0.0, 1.25 - 0.3 * abs((item["date"] - anchor_date).days))
        if item["date"] > target_date:
            score -= 0.5
        return score

    def _candidate_text(self, item: dict) -> str:
        quote = item.get("quote")
        if quote is not None:
            summary_text = str(item.get("summary_text", "") or "").strip()
            if summary_text:
                return summary_text
            return str(getattr(quote, "text", "") or "")
        fact = item.get("fact")
        return " ".join([
            str(getattr(fact, "what", "") or ""),
            str(getattr(fact, "subject", "") or ""),
            str(getattr(fact, "verb", "") or ""),
            str(getattr(fact, "object", "") or ""),
        ])

    def _value_signal_bonus(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = _VALUE_TOKEN_RE.findall(text)
        if not tokens:
            return 0.0
        percent_count = sum(1 for token in tokens if token.endswith("%"))
        numeric_count = max(0, len(tokens) - percent_count)
        bonus = min(2.5, 0.4 * numeric_count) + min(2.0, 1.0 * percent_count)
        if percent_count and numeric_count:
            bonus += 1.0
        return bonus

    def _anchor_text_bonus(self, text: str, anchor_text: str) -> float:
        if not text or not anchor_text:
            return 0.0
        anchor_lower = anchor_text.lower()
        text_lower = text.lower()
        anchor_values = set(_VALUE_TOKEN_RE.findall(anchor_lower))
        text_values = set(_VALUE_TOKEN_RE.findall(text_lower))
        shared_values = anchor_values & text_values
        unmatched_values = text_values - anchor_values
        value_bonus = 5.0 * len(shared_values) - 2.5 * len(unmatched_values)

        anchor_terms = {
            token
            for token in _QUERY_TOKEN_RE.findall(anchor_lower)
            if len(token) >= 4 and token not in _QUERY_STOPWORDS
        }
        text_terms = set(_QUERY_TOKEN_RE.findall(text_lower))
        lexical_overlap = len(anchor_terms & text_terms)
        lexical_bonus = min(1.5, 0.2 * lexical_overlap)
        return value_bonus + lexical_bonus

    def _state_temporal_bonus(self, candidate_date: date, target_date: date) -> float:
        delta_days = (candidate_date - target_date).days
        proximity_bonus = max(0.0, 2.5 - 0.35 * abs(delta_days))
        exact_bonus = 3.0 if delta_days == 0 else 0.0
        as_of_bonus = 1.25 if delta_days <= 0 else 0.25
        return proximity_bonus + exact_bonus + as_of_bonus

    def _state_target_date(self, start: date, end: date, mode: str) -> date | None:
        if mode not in _REMEMBER_WHEN_STATE_MODES:
            return None
        midpoint_days = max(0, (end - start).days // 2)
        return start + timedelta(days=midpoint_days)

    def _describe_state_anchor(
        self,
        *,
        results: list[dict],
        facts: list[dict],
        target_date: date,
    ) -> dict | None:
        if results:
            anchor = results[0]
            anchor_date = anchor.get("session_date_normalized", "")
            return {
                "source": "segment",
                "date": anchor_date,
                "days_from_target": anchor.get("date_distance_days"),
                "as_of_target": anchor.get("as_of_target"),
                "topic": anchor.get("topic", ""),
            }
        if facts:
            anchor = facts[0]
            return {
                "source": "fact",
                "date": anchor.get("when", ""),
                "days_from_target": anchor.get("date_distance_days"),
                "as_of_target": anchor.get("as_of_target"),
                "topic": "",
            }
        return None

    def _resolve_state_view(
        self,
        *,
        results: list[dict],
        facts: list[dict],
        target_date: date,
        max_results: int,
    ) -> dict:
        if not results and not facts:
            return {
                "results": results,
                "facts_in_window": facts,
                "chosen_state": None,
                "conflicting_candidates": [],
            }

        anchor_kind, anchor = self._state_anchor_item(results, facts)
        if anchor is None:
            return {
                "results": results,
                "facts_in_window": facts,
                "chosen_state": None,
                "conflicting_candidates": [],
            }

        anchor_text = self._state_item_text(anchor_kind, anchor)
        anchor_date = self._state_item_date(anchor_kind, anchor)
        anchor_segment_ref = self._state_item_segment_ref(anchor_kind, anchor)
        anchor_topic = self._state_item_topic(anchor_kind, anchor)
        anchor_terms = {
            str(term).strip().lower()
            for term in anchor.get("matched_terms", []) or []
            if str(term).strip()
        }
        anchor_values = self._normalized_value_tokens(anchor_text)

        kept_results: list[tuple[float, dict]] = []
        kept_facts: list[tuple[float, dict]] = []
        conflicts: list[tuple[float, dict]] = []

        for idx, item in enumerate(results):
            if idx == 0 and anchor_kind == "segment":
                kept_results.append((float("inf"), item))
                continue
            analysis = self._analyze_state_item(
                kind="segment",
                item=item,
                anchor_text=anchor_text,
                anchor_date=anchor_date,
                anchor_segment_ref=anchor_segment_ref,
                anchor_topic=anchor_topic,
                anchor_terms=anchor_terms,
                anchor_values=anchor_values,
                target_date=target_date,
            )
            if analysis["conflict"]:
                conflicts.append((analysis["score"], self._summarize_state_conflict("segment", item, analysis)))
                continue
            if analysis["support"]:
                kept_results.append((analysis["score"], item))

        for idx, item in enumerate(facts):
            if idx == 0 and anchor_kind == "fact" and not results:
                kept_facts.append((float("inf"), item))
                continue
            analysis = self._analyze_state_item(
                kind="fact",
                item=item,
                anchor_text=anchor_text,
                anchor_date=anchor_date,
                anchor_segment_ref=anchor_segment_ref,
                anchor_topic=anchor_topic,
                anchor_terms=anchor_terms,
                anchor_values=anchor_values,
                target_date=target_date,
            )
            if analysis["conflict"]:
                conflicts.append((analysis["score"], self._summarize_state_conflict("fact", item, analysis)))
                continue
            if analysis["support"]:
                kept_facts.append((analysis["score"], item))

        chosen_results = [item for _score, item in kept_results[:max_results]]
        fact_cap = max(max_results + 2, 8)
        chosen_facts = [item for _score, item in kept_facts[:fact_cap]]

        chosen_state = self._build_chosen_state(
            anchor_kind=anchor_kind,
            anchor=anchor,
            results=chosen_results,
            facts=chosen_facts,
            target_date=target_date,
        )
        conflicts.sort(key=lambda item: item[0], reverse=True)
        return {
            "results": chosen_results,
            "facts_in_window": chosen_facts,
            "chosen_state": chosen_state,
            "conflicting_candidates": [item for _score, item in conflicts[:6]],
        }

    def _state_anchor_item(
        self,
        results: list[dict],
        facts: list[dict],
    ) -> tuple[str, dict | None]:
        if results:
            return "segment", results[0]
        if facts:
            return "fact", facts[0]
        return "", None

    def _state_item_text(self, kind: str, item: dict) -> str:
        if kind == "segment":
            return str(item.get("excerpt", "") or "")
        return str(item.get("what", "") or "")

    def _state_item_date(self, kind: str, item: dict) -> date:
        raw = item.get("session_date_normalized") if kind == "segment" else item.get("when")
        return date.fromisoformat(str(raw))

    def _state_item_segment_ref(self, kind: str, item: dict) -> str:
        return str(item.get("segment_ref", "") or "").strip()

    def _state_item_topic(self, kind: str, item: dict) -> str:
        return str(item.get("topic", "") or "").strip()

    def _normalized_value_tokens(self, text: str) -> set[str]:
        return {
            token.replace(",", "").strip()
            for token in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", text or "")
            if token.strip()
        }

    def _value_kinds(self, values: set[str]) -> set[str]:
        kinds: set[str] = set()
        for token in values:
            if token.endswith("%"):
                kinds.add("percentage")
            else:
                kinds.add("numeric")
        return kinds

    def _analyze_state_item(
        self,
        *,
        kind: str,
        item: dict,
        anchor_text: str,
        anchor_date: date,
        anchor_segment_ref: str,
        anchor_topic: str,
        anchor_terms: set[str],
        anchor_values: set[str],
        target_date: date,
    ) -> dict:
        text = self._state_item_text(kind, item)
        item_date = self._state_item_date(kind, item)
        item_segment_ref = self._state_item_segment_ref(kind, item)
        item_topic = self._state_item_topic(kind, item)
        item_terms = {
            str(term).strip().lower()
            for term in item.get("matched_terms", []) or []
            if str(term).strip()
        }
        item_terms.update(
            str(term).strip().lower()
            for term in item.get("soft_terms", []) or []
            if str(term).strip()
        )
        text_terms = {
            token
            for token in _QUERY_TOKEN_RE.findall(text.lower())
            if len(token) >= 4 and token not in _QUERY_STOPWORDS
        }
        item_terms.update(text_terms & anchor_terms)
        item_values = self._normalized_value_tokens(text)
        anchor_value_kinds = self._value_kinds(anchor_values)
        item_value_kinds = self._value_kinds(item_values)
        shared_value_kinds = anchor_value_kinds & item_value_kinds
        shared_values = anchor_values & item_values
        unmatched_values = item_values - anchor_values
        anchor_bonus = self._anchor_text_bonus(text, anchor_text)
        same_date = item_date == anchor_date
        same_segment = bool(anchor_segment_ref and item_segment_ref and item_segment_ref == anchor_segment_ref)
        same_topic = bool(anchor_topic and item_topic and item_topic == anchor_topic)
        shared_terms = anchor_terms & item_terms

        score = anchor_bonus
        if same_date:
            score += 1.25
        if same_segment:
            score += 1.25
        if same_topic:
            score += 0.75
        score += 0.5 * len(shared_terms)
        score += 0.9 * len(shared_values)
        score += self._state_temporal_bonus(item_date, target_date) * 0.2

        semantic_overlap = same_date or same_segment or same_topic or bool(shared_terms)
        conflict = bool(
            anchor_values
            and item_values
            and shared_value_kinds
            and not shared_values
            and semantic_overlap
            and anchor_bonus <= -1.5
        )

        support = bool(
            shared_values
            or same_segment
            or (
                item_values
                and not shared_value_kinds
                and semantic_overlap
            )
            or (
                not item_values
                and (
                    same_segment
                    or same_topic
                    or anchor_bonus > 0.75
                    or (same_date and anchor_bonus > 0.0)
                )
            )
            or (item_values and score >= 1.5)
        )
        return {
            "score": score,
            "conflict": conflict,
            "support": support,
            "shared_values": sorted(shared_values),
            "value_signals": sorted(item_values),
        }

    def _summarize_state_conflict(self, kind: str, item: dict, analysis: dict) -> dict:
        item_date = item.get("session_date_normalized") if kind == "segment" else item.get("when")
        summary = {
            "source": kind,
            "date": item_date,
            "days_from_target": item.get("date_distance_days"),
            "as_of_target": item.get("as_of_target"),
            "value_signals": analysis.get("value_signals", []),
            "matched_terms": item.get("matched_terms", []),
            "reason": "conflicting_value_bundle",
        }
        if kind == "segment":
            summary["topic"] = item.get("topic", "")
            summary["excerpt"] = self._trim_state_text(str(item.get("excerpt", "") or ""))
        else:
            summary["what"] = self._trim_state_text(str(item.get("what", "") or ""))
            summary["segment_ref"] = item.get("segment_ref", "")
        return summary

    def _build_chosen_state(
        self,
        *,
        anchor_kind: str,
        anchor: dict,
        results: list[dict],
        facts: list[dict],
        target_date: date,
    ) -> dict:
        anchor_text = self._state_item_text(anchor_kind, anchor)
        anchor_date = self._state_item_date(anchor_kind, anchor)
        value_signals = self._normalized_value_tokens(anchor_text)
        for fact in facts:
            value_signals.update(self._normalized_value_tokens(str(fact.get("what", "") or "")))
        return {
            "effective_date": anchor_date.isoformat(),
            "as_of_target": anchor_date <= target_date,
            "source": anchor_kind,
            "topic": anchor.get("topic", "") if anchor_kind == "segment" else "",
            "summary": self._trim_state_text(anchor_text, limit=360),
            "value_signals": sorted(value_signals),
            "supporting_facts": [
                self._trim_state_text(str(fact.get("what", "") or ""), limit=180)
                for fact in facts[:4]
            ],
            "supporting_segment_topics": [
                str(item.get("topic", "") or "")
                for item in results[:3]
                if str(item.get("topic", "") or "")
            ],
        }

    def _trim_state_text(self, text: str, *, limit: int = 220) -> str:
        text = " ".join((text or "").split())
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _evenly_spaced_indices(self, total: int, count: int) -> list[int]:
        if count >= total:
            return list(range(total))
        if count <= 1:
            return [0]

        step = (total - 1) / (count - 1)
        indices: list[int] = []
        seen: set[int] = set()
        for i in range(count):
            idx = floor(i * step + 0.5)
            idx = min(total - 1, max(0, idx))
            if idx in seen:
                continue
            indices.append(idx)
            seen.add(idx)
        if len(indices) < count:
            for idx in range(total):
                if idx in seen:
                    continue
                indices.append(idx)
                seen.add(idx)
                if len(indices) >= count:
                    break
        return sorted(indices[:count])

    def _resolve_remember_when_range(self, time_range: dict) -> tuple[date, date, str]:
        """Resolve tool time_range input to absolute [start_date, end_date]."""
        if not isinstance(time_range, dict):
            raise ValueError("time_range must be an object")

        kind = str(time_range.get("kind", "")).strip().lower()
        today = self.reference_date or datetime.now(timezone.utc).date()

        if kind == "relative":
            preset = str(time_range.get("preset", "")).strip().lower()
            if preset == "last_24_hours":
                return today - timedelta(days=2), today, preset
            if preset == "last_7_days":
                return today - timedelta(days=7), today, preset
            if preset == "last_30_days":
                return today - timedelta(days=30), today, preset
            if preset == "last_90_days":
                return today - timedelta(days=90), today, preset
            if preset == "last_180_days":
                return today - timedelta(days=180), today, preset
            if preset == "this_week":
                start = today - timedelta(days=today.weekday())
                return start, start + timedelta(days=6), preset
            if preset == "last_week":
                this_week_start = today - timedelta(days=today.weekday())
                start = this_week_start - timedelta(days=7)
                return start, start + timedelta(days=6), preset
            if preset == "this_month":
                start = date(today.year, today.month, 1)
                end = date(today.year, today.month, monthrange(today.year, today.month)[1])
                return start, end, preset
            if preset == "last_month":
                year = today.year
                month = today.month - 1
                if month == 0:
                    month = 12
                    year -= 1
                start = date(year, month, 1)
                end = date(year, month, monthrange(year, month)[1])
                return start, end, preset
            if preset == "this_year":
                return date(today.year, 1, 1), date(today.year, 12, 31), preset
            if preset == "last_year":
                y = today.year - 1
                return date(y, 1, 1), date(y, 12, 31), preset
            raise ValueError(f"unsupported relative preset: {preset}")

        if kind == "between_dates":
            start_raw = str(time_range.get("start", "")).strip()
            end_raw = str(time_range.get("end", "")).strip()
            if not start_raw or not end_raw:
                raise ValueError("between_dates requires start and end")

            def parse_boundary(raw: str, is_end: bool) -> date:
                if _ISO_DATE_RE.match(raw):
                    return date.fromisoformat(raw)
                if _ISO_MONTH_RE.match(raw):
                    y, mo = [int(x) for x in raw.split("-")]
                    if is_end:
                        return date(y, mo, monthrange(y, mo)[1])
                    return date(y, mo, 1)
                raise ValueError(f"invalid date format: {raw}")

            start = parse_boundary(start_raw, is_end=False)
            end = parse_boundary(end_raw, is_end=True)
            if end < start:
                raise ValueError("time_range end must be >= start")
            return start, end, "between_dates"

        raise ValueError("time_range.kind must be 'relative' or 'between_dates'")
