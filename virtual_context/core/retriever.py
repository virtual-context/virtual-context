"""ContextRetriever: tag inbound message and fetch relevant summaries by tag overlap."""

from __future__ import annotations

import logging
import math
import re
import time

from .store import ContextStore
from .tag_generator import TagGenerator, detect_temporal_heuristic
from .turn_tag_index import TurnTagIndex
from ..patterns import DEFAULT_TEMPORAL_PATTERNS
from ..types import (
    Message,
    RetrievalCostReport,
    RetrievalResult,
    RetrieverConfig,
    StoredSegment,
    StoredSummary,
)

logger = logging.getLogger(__name__)


class ContextRetriever:

    def __init__(
        self,
        tag_generator: TagGenerator,
        store: ContextStore,
        config: RetrieverConfig,
        turn_tag_index: TurnTagIndex | None = None,
        inbound_tagger: TagGenerator | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.tag_generator = tag_generator
        self.store = store
        self.config = config
        self._turn_tag_index = turn_tag_index
        self._inbound_tagger = inbound_tagger
        self._conversation_id = conversation_id
        # Pre-compile heuristic patterns for embedding-based inbound tagger
        self._temporal_patterns = [re.compile(p, re.IGNORECASE) for p in DEFAULT_TEMPORAL_PATTERNS]

    def _compute_idf_weights(self) -> dict[str, float]:
        """Compute IDF weights from tag usage counts in the store.

        Returns a dict mapping tag → log(1 + total_segments / usage_count).
        Rare tags get higher weights than common ones.
        """
        all_tags = self.store.get_all_tags(
            conversation_id=self._conversation_id,
        )
        if not all_tags:
            return {}
        total = sum(ts.usage_count for ts in all_tags)
        if total == 0:
            return {}
        return {
            ts.tag: math.log(1 + total / max(ts.usage_count, 1))
            for ts in all_tags
        }

    def _fetch_all_facts(self) -> list:
        try:
            return self.store.query_facts(limit=9999, conversation_id=self._conversation_id)
        except Exception:
            return []

    def _load_all_tag_summaries(self, token_budget: int) -> tuple[list[StoredSummary], int]:
        """Load all tag summaries within *token_budget*.

        Returns ``(selected_summaries, total_tokens)``.  Used by the
        post-compaction summary floor.
        """
        tag_summaries = self.store.get_all_tag_summaries(conversation_id=self._conversation_id)
        if not tag_summaries:
            return [], 0

        selected: list[StoredSummary] = []
        total_tokens = 0
        for ts in tag_summaries:
            if total_tokens + ts.summary_tokens > token_budget:
                break
            selected.append(StoredSummary(
                ref=f"tag-summary-{ts.tag}",
                primary_tag=ts.tag,
                tags=[ts.tag],
                summary=ts.summary,
                summary_tokens=ts.summary_tokens,
                created_at=ts.updated_at,
                start_timestamp=ts.created_at,
                end_timestamp=ts.updated_at,
            ))
            total_tokens += ts.summary_tokens
        return selected, total_tokens

    def retrieve(
        self,
        message: str,
        current_active_tags: list[str] | None = None,
        current_utilization: float = 0.0,
        post_compaction: bool = False,
        context_turns: list[str] | None = None,
    ) -> RetrievalResult:
        """Tag inbound message, fetch relevant summaries by tag overlap.

        Args:
            message: The inbound user message.
            current_active_tags: Tags from recent conversation turns to skip.
            current_utilization: Current context window usage ratio (0.0-1.0).
            context_turns: Recent user/assistant text for context-aware tagging.
        """
        start_time = time.monotonic()
        active_tags = set(current_active_tags or [])

        # Tag the inbound message — scope vocabulary to current conversation
        store_tags = [ts.tag for ts in self.store.get_all_tags(
            conversation_id=self._conversation_id,
        )]
        if self._inbound_tagger is not None:
            # Embedding-based: match against existing vocabulary (no hallucination)
            # Store tags are conversation-scoped via get_all_tags().
            # TurnTagIndex is inherently per-conversation (loaded from that
            # conversation's engine_state), so all its tags are in scope.
            vocab_tags = list(set(store_tags))
            if self._turn_tag_index:
                all_index_tags = {t for e in self._turn_tag_index.entries for t in e.tags}
                vocab_tags = list(set(vocab_tags) | all_index_tags)
            tag_result = self._inbound_tagger.generate_tags(
                message, vocab_tags, context_turns=context_turns,
            )
            # Apply temporal heuristic (embedding tagger doesn't detect these)
            if not tag_result.temporal:
                tag_result.temporal = detect_temporal_heuristic(message, self._temporal_patterns)
            logger.debug("Inbound embedding match: tags=%s temporal=%s",
                         tag_result.tags, tag_result.temporal)
        else:
            # LLM-based: open-ended tag generation (pass known tags for reuse)
            tag_result = self.tag_generator.generate_tags(
                message, store_tags, context_turns=context_turns,
            )

        retrieval_metadata: dict = {}

        # Temporal detection is now advisory; time-scoped recall is tool-driven
        # via vc_remember_when rather than an automatic retriever branch.
        if tag_result.temporal:
            retrieval_metadata["temporal_hint"] = True

        if tag_result.source == "fallback":
            # Tagging failed — fall back to working set tags from live index
            if self._turn_tag_index:
                working_set = list(self._turn_tag_index.get_active_tags(
                    lookback=self.config.anchorless_lookback
                ))
                query_tags = [t for t in working_set if t != "_general"]
                retrieval_metadata["fallback"] = "working_set"
            else:
                query_tags = [t for t in tag_result.tags if t != "_general"]
        else:
            # Normal flow — filter out active tags (already in recent context)
            query_tags = [
                t for t in tag_result.tags
                if not (self.config.skip_active_tags and t in active_tags)
                and t != "_general"
            ]

        skipped_tags = [t for t in tag_result.tags if t in active_tags]

        if not query_tags:
            # Summary floor: post-compaction, no query tags — inject all tag summaries
            if post_compaction:
                token_budget = self.config.tag_context_max_tokens
                floor_summaries, floor_tokens = self._load_all_tag_summaries(token_budget)
                if floor_summaries:
                    elapsed = time.monotonic() - start_time
                    return RetrievalResult(
                        tags_matched=[],
                        summaries=floor_summaries,
                        total_tokens=floor_tokens,
                        facts=self._fetch_all_facts(),
                        retrieval_metadata={
                            "elapsed_ms": round(elapsed * 1000, 1),
                            "tags_from_message": tag_result.tags,
                            "tags_skipped_active": skipped_tags,
                            "summary_floor": True,
                        },
                        cost_report=RetrievalCostReport(
                            tags_queried=[],
                            tags_skipped=skipped_tags,
                            strategy_active="summary_floor",
                        ),
                    )

            elapsed = time.monotonic() - start_time
            return RetrievalResult(
                tags_matched=[],
                summaries=[],
                total_tokens=0,
                facts=self._fetch_all_facts(),
                retrieval_metadata={
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "tags_from_message": tag_result.tags,
                    "tags_skipped_active": skipped_tags,
                },
                cost_report=RetrievalCostReport(
                    tags_queried=[],
                    tags_skipped=skipped_tags,
                    strategy_active="default",
                ),
            )

        # Determine strategy and budget
        strategy = self.config.strategy_configs.get("default")
        if strategy is None:
            from ..types import StrategyConfig
            strategy = StrategyConfig()

        # Scale budget by utilization — as context fills, retrieve less
        budget_fraction = strategy.max_budget_fraction
        if current_utilization > 0.5:
            scale = max(0.1, 1.0 - current_utilization)
            budget_fraction *= scale

        token_budget = int(self.config.tag_context_max_tokens * budget_fraction)

        # Expand query with related tags from tagger
        related_query_tags = [
            t for t in tag_result.related_tags
            if t not in set(query_tags) and t != "_general"
        ]
        expanded_tags = list(set(query_tags) | set(related_query_tags))

        # Overfetch 3x for IDF re-ranking
        overfetch_limit = strategy.max_results * 3
        summaries = self.store.get_summaries_by_tags(
            tags=expanded_tags,
            min_overlap=strategy.min_overlap,
            limit=overfetch_limit,
            conversation_id=self._conversation_id,
        )

        # IDF re-rank: score each result by weighted tag overlap
        idf_weights = self._compute_idf_weights()
        query_tag_set = set(query_tags)
        related_tag_set = set(related_query_tags)

        scored: list[tuple[float, StoredSummary]] = []
        for summary in summaries:
            summary_tag_set = set(summary.tags)
            # Primary matches: full IDF weight
            primary_score = sum(
                idf_weights.get(t, 1.0) for t in query_tag_set & summary_tag_set
            )
            # Related matches: 0.5x IDF weight
            related_score = 0.5 * sum(
                idf_weights.get(t, 1.0) for t in related_tag_set & summary_tag_set
            )
            scored.append((primary_score + related_score, summary))

        # Sort by score descending, take top max_results
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [s for _, s in scored[:strategy.max_results]]

        # Apply token budget
        selected: list[StoredSummary] = []
        selected_refs: set[str] = set()
        total_tokens = 0
        for summary in ranked:
            if total_tokens + summary.summary_tokens > token_budget:
                break
            selected.append(summary)
            selected_refs.add(summary.ref)
            total_tokens += summary.summary_tokens

        # Alias ride-along: if a query tag made it into the selected results,
        # find additional segments via its aliases that didn't make max_results.
        # These ride free — they don't count against max_results.
        if selected:
            alias_extras = _alias_ride_along(
                self.store, query_tag_set, selected, selected_refs,
                token_budget, total_tokens,
                conversation_id=self._conversation_id,
            )
            if alias_extras:
                selected.extend(alias_extras)
                selected_refs.update(s.ref for s in alias_extras)
                total_tokens += sum(s.summary_tokens for s in alias_extras)
                logger.info(
                    "Alias ride-along: +%d segments (%d tokens)",
                    len(alias_extras),
                    sum(s.summary_tokens for s in alias_extras),
                )

        # FTS fallback: tag overlap missed — try full-text search on stored segments
        # Convert tags to FTS query terms: "cook-mode" → "cook mode"
        if not selected and expanded_tags:
            fts_terms = " OR ".join(
                f'"{tag.replace("-", " ")}"' for tag in expanded_tags
            )
            fts_results = self.store.search(
                query=fts_terms,
                limit=strategy.max_results,
                conversation_id=self._conversation_id,
            )
            for summary in fts_results:
                if total_tokens + summary.summary_tokens > token_budget:
                    break
                selected.append(summary)
                total_tokens += summary.summary_tokens
            if fts_results:
                retrieval_metadata["fts_fallback"] = True
                logger.debug(
                    "FTS fallback: tag overlap empty, text search found %d results",
                    len(selected),
                )

        # Summary floor: post-compaction, no results — inject all tag summaries
        if not selected and post_compaction:
            selected, total_tokens = self._load_all_tag_summaries(token_budget)
            if selected:
                retrieval_metadata["summary_floor"] = True

        # Deep retrieval: fetch full segments when we have matches
        full_detail: list[StoredSegment] = []
        if selected:
            for summary in selected[:3]:  # limit to top 3
                segment = self.store.get_segment(summary.ref, conversation_id=self._conversation_id)
                if segment:
                    full_detail.append(segment)

        # Collect matched tags (include related tag matches too)
        expanded_set = set(expanded_tags)
        matched_tags = list({tag for s in selected for tag in s.tags if tag in expanded_set})

        elapsed = time.monotonic() - start_time

        retrieval_metadata.update({
            "elapsed_ms": round(elapsed * 1000, 1),
            "tags_from_message": tag_result.tags,
            "tags_queried": query_tags,
            "tags_skipped_active": skipped_tags,
            "summaries_returned": len(selected),
            "strategy": "default",
            "budget_fraction": round(budget_fraction, 3),
            "idf_reranked": bool(idf_weights),
            "related_tags_used": related_query_tags,
            "query_expanded": len(related_query_tags) > 0,
        })

        return RetrievalResult(
            tags_matched=matched_tags,
            summaries=selected,
            full_detail=full_detail,
            total_tokens=total_tokens,
            facts=self._fetch_all_facts(),
            retrieval_metadata=retrieval_metadata,
            cost_report=RetrievalCostReport(
                tokens_retrieved=total_tokens,
                budget_fraction_used=total_tokens / token_budget if token_budget > 0 else 0.0,
                strategy_active="default",
                tags_queried=query_tags,
                tags_skipped=skipped_tags,
            ),
        )


def _alias_ride_along(
    store: ContextStore,
    query_tag_set: set[str],
    selected: list[StoredSummary],
    selected_refs: set[str],
    token_budget: int,
    tokens_used: int,
    conversation_id: str | None = None,
) -> list[StoredSummary]:
    """Find extra segments via aliases for tags that already made the cut.

    If query tag X qualified (appears in selected results), find all tags
    in X's alias group and retrieve segments that match those aliases but
    weren't already selected. These ride free — no max_results cap.
    """
    aliases = store.get_tag_aliases()
    if not aliases:
        return []

    # Build reverse: canonical → {aliases}
    reverse: dict[str, set[str]] = {}
    for alias, canonical in aliases.items():
        reverse.setdefault(canonical, set()).add(alias)

    # Which query tags actually matched in the selected results?
    qualified_tags: set[str] = set()
    for s in selected:
        qualified_tags.update(set(s.tags) & query_tag_set)

    if not qualified_tags:
        return []

    # For each qualified tag, collect its full alias group
    alias_tags: set[str] = set()
    for qt in qualified_tags:
        # qt might be an alias → get its canonical + siblings
        if qt in aliases:
            canon = aliases[qt]
            alias_tags.add(canon)
            alias_tags.update(reverse.get(canon, set()))
        # qt might be a canonical → get all its aliases
        if qt in reverse:
            alias_tags.update(reverse[qt])

    # Remove tags we already queried with
    alias_tags -= query_tag_set
    if not alias_tags:
        return []

    # Fetch segments matching alias tags
    alias_summaries = store.get_summaries_by_tags(
        tags=list(alias_tags),
        min_overlap=1,
        limit=50,
        conversation_id=conversation_id,
    )

    # Add new segments within token budget
    extras: list[StoredSummary] = []
    remaining = token_budget - tokens_used
    for s in alias_summaries:
        if s.ref in selected_refs:
            continue
        if s.summary_tokens > remaining:
            continue
        extras.append(s)
        selected_refs.add(s.ref)
        remaining -= s.summary_tokens

    return extras
