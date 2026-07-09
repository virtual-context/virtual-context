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
    SegmentMetadata,
    StoredSegment,
    StoredSummary,
)

logger = logging.getLogger(__name__)

_RETRIEVAL_BREAKDOWN_LOG_THRESHOLD_MS = 500.0


class ContextRetriever:

    def __init__(
        self,
        tag_generator: TagGenerator,
        store: ContextStore,
        config: RetrieverConfig,
        turn_tag_index: TurnTagIndex | None = None,
        inbound_tagger: TagGenerator | None = None,
        conversation_id: str | None = None,
        session_state_provider=None,
        query_embed_fn=None,
    ) -> None:
        self.tag_generator = tag_generator
        self.store = store
        self.config = config
        self._turn_tag_index = turn_tag_index
        self._inbound_tagger = inbound_tagger
        self._conversation_id = conversation_id
        self._session_state_provider = session_state_provider
        # Shared local embed function for dense fact retrieval. Independent of
        # ``inbound_tagger`` so ``fact_dense_retrieval`` works under any
        # ``inbound_tagger_type``. Signature: ``list[str] -> list[list[float]]``.
        self._query_embed_fn = query_embed_fn
        # Pre-compile heuristic patterns for embedding-based inbound tagger
        self._temporal_patterns = [re.compile(p, re.IGNORECASE) for p in DEFAULT_TEMPORAL_PATTERNS]

    def _load_all_tags_snapshot(self) -> list:
        if self._session_state_provider is not None and self._conversation_id:
            cached = self._session_state_provider.load_tag_stats_snapshot(
                self._conversation_id,
            )
            if cached is not None:
                return cached
        all_tags = self.store.get_all_tags(
            conversation_id=self._conversation_id,
        )
        if self._session_state_provider is not None and self._conversation_id:
            self._session_state_provider.save_tag_stats_snapshot(
                self._conversation_id,
                all_tags,
            )
        return all_tags

    def _load_tag_summary_embedding_snapshot(self) -> dict[str, list[float]] | None:
        if self._session_state_provider is not None and self._conversation_id:
            cached = self._session_state_provider.load_tag_summary_embedding_snapshot(
                self._conversation_id,
            )
            if cached is not None:
                return cached
        stored = self.store.load_tag_summary_embeddings(
            conversation_id=self._conversation_id,
        )
        if self._session_state_provider is not None and self._conversation_id:
            self._session_state_provider.save_tag_summary_embedding_snapshot(
                self._conversation_id,
                stored,
            )
        return stored

    def _build_context_query_embedding(
        self, message: str, context_turns: list[str] | None,
    ) -> tuple[list[float] | None, str, float | None]:
        """Embed recent-turn context + the current message for the guard signal.

        Returns ``(embedding, mode, embed_ms)``. ``embedding`` is None (and mode
        ``"bare"``) whenever context blending is disabled, no context is
        available, or the inbound tagger cannot embed — in which case retrieval
        scores the bare query embedding exactly as before. ``mode`` is
        ``"guard"`` or ``"concat"`` per ``scoring.embedding_context_guard`` when
        a context vector is produced.
        """
        scoring = self.config.scoring
        n_ctx = getattr(scoring, "embedding_context_turns", 0)
        if n_ctx <= 0 or not context_turns:
            return None, "bare", None
        embed_text = getattr(self._inbound_tagger, "embed_text", None)
        if embed_text is None:
            return None, "bare", None
        recent = [t for t in context_turns[-n_ctx:] if t]
        if not recent:
            return None, "bare", None
        concat_text = (" ".join(recent) + " " + message).strip()
        _stage = time.monotonic()
        ctx_embedding = embed_text(concat_text)
        embed_ms = round((time.monotonic() - _stage) * 1000, 1)
        if not ctx_embedding:
            return None, "bare", embed_ms
        mode = "guard" if getattr(scoring, "embedding_context_guard", True) else "concat"
        return ctx_embedding, mode, embed_ms

    def _compute_idf_weights(self, all_tags: list | None = None) -> dict[str, float]:
        """Compute IDF weights from tag usage counts in the store.

        Returns a dict mapping tag → log(1 + total_segments / usage_count).
        Rare tags get higher weights than common ones.
        """
        if all_tags is None:
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

    def _fetch_facts_by_tags(self, tags: list[str], limit: int = 500) -> list:
        try:
            return self.store.query_facts(tags=tags, limit=limit, conversation_id=self._conversation_id)
        except Exception:
            return []

    def _fetch_facts_dense(
        self,
        message: str,
        context_turns: list[str] | None,
        expanded_tags: list[str],
        retrieval_metadata: dict,
    ) -> list:
        """Augment the legacy tag-gated fact fetch with a dense kNN ranking.

        Runs the exact gate-off legacy fetch first to preserve the
        non-evictable tag-gated floor (INV-6), then ranks the conversation's
        current-model fact vectors by cosine similarity to the (optionally
        recent-turn-blended) query embedding and unions the top
        ``fact_dense_top_n`` in front of the floor. A fact present in both
        sets appears once at its dense position. Attaches dense rank/score,
        the legacy floor ids, and per-fact source to *retrieval_metadata* so
        the assembler can preserve dense ordering without re-scoring.
        """
        from .math_utils import cosine_similarity

        _t0 = time.monotonic()
        # 1. Legacy floor — the exact gate-off fetch, order preserved.
        if self.config.prefetch_facts and expanded_tags:
            legacy_facts = self._fetch_facts_by_tags(expanded_tags)
        else:
            legacy_facts = self._fetch_all_facts()
        floor_ids = [f.id for f in legacy_facts]
        floor_set = set(floor_ids)

        embed_fn = self._query_embed_fn
        if embed_fn is None:
            logger.warning(
                "FACT_DENSE_BREAKDOWN dense retrieval enabled but no query "
                "embedder configured; falling back to legacy floor only",
            )
            return legacy_facts

        # 2. Query embeddings on the shared local provider.
        try:
            bare_vec = embed_fn([message])[0]
        except Exception:
            logger.warning("FACT_DENSE_BREAKDOWN query embed failed; legacy floor only")
            return legacy_facts

        scoring = self.config.scoring
        n_ctx = getattr(scoring, "embedding_context_turns", 0)
        ctx_vec = None
        if n_ctx > 0 and context_turns:
            recent = [t for t in context_turns[-n_ctx:] if t]
            if recent:
                concat_text = (" ".join(recent) + " " + message).strip()
                try:
                    ctx_vec = embed_fn([concat_text])[0]
                except Exception:
                    ctx_vec = None
        guard = getattr(scoring, "embedding_context_guard", True)

        # 3. Load current-model fact vectors and cosine-rank.
        model = getattr(self.config, "embedding_model", "")
        try:
            loaded = self.store.load_fact_embeddings(
                self._conversation_id, model, expected_dim=len(bare_vec),
            )
        except Exception:
            loaded = {}
        scored: list[tuple[float, str, object]] = []
        for fid, (fact, vec) in loaded.items():
            bare_cos = cosine_similarity(bare_vec, vec)
            if ctx_vec is not None:
                ctx_cos = cosine_similarity(ctx_vec, vec)
                score = max(bare_cos, ctx_cos) if guard else ctx_cos
            else:
                score = bare_cos
            scored.append((score, fid, fact))
        # Cosine DESC, tie-break fact.id ASC (deterministic union order).
        scored.sort(key=lambda x: (-x[0], x[1]))
        top_n = getattr(self.config, "fact_dense_top_n", 20)
        top = scored[:top_n]

        # 4. Deterministic union by fact.id: dense first, floor-only after.
        dense_rank_by_id: dict[str, int] = {}
        dense_score_by_id: dict[str, float] = {}
        source_by_id: dict[str, str] = {}
        union: list = []
        seen: set[str] = set()
        for rank, (score, fid, fact) in enumerate(top):
            dense_rank_by_id[fid] = rank
            dense_score_by_id[fid] = score
            source_by_id[fid] = "both" if fid in floor_set else "dense"
            union.append(fact)
            seen.add(fid)
        for fact in legacy_facts:
            if fact.id in seen:
                continue
            union.append(fact)
            seen.add(fact.id)
            source_by_id[fact.id] = "tag"

        # 5. Metadata for the assembler dense-priority branch.
        retrieval_metadata["fact_dense_rank_by_id"] = dense_rank_by_id
        retrieval_metadata["fact_dense_score_by_id"] = dense_score_by_id
        retrieval_metadata["fact_tag_floor_ids"] = floor_ids
        retrieval_metadata["fact_source_by_id"] = source_by_id

        overlap = len(set(dense_rank_by_id) & floor_set)
        top1 = top[0][0] if top else 0.0
        logger.info(
            "FACT_DENSE_BREAKDOWN conv=%s dense=%d floor=%d overlap=%d union=%d "
            "top1_cos=%.4f load_score_ms=%.1f",
            (self._conversation_id or "none")[:12],
            len(dense_rank_by_id), len(floor_ids), overlap, len(union),
            top1, round((time.monotonic() - _t0) * 1000, 1),
        )
        return union

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
                metadata=SegmentMetadata(code_refs=list(getattr(ts, "code_refs", []) or [])),
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
        *,
        entries_snapshot: list | None = None,
    ) -> RetrievalResult:
        """Tag inbound message, fetch relevant summaries by tag overlap.

        Args:
            message: The inbound user message.
            current_active_tags: Tags from recent conversation turns to skip.
            current_utilization: Current context window usage ratio (0.0-1.0).
            context_turns: Recent user/assistant text for context-aware tagging.
            entries_snapshot: optional bounded view of
                ``TurnTagIndex.entries`` captured by the caller at the
                inbound method entry. When provided, the retriever's
                fallback paths (working-set tags on tagger failure,
                inherit-from-previous on _general) read tags from the
                snapshot so a concurrent legacy-tagger append does not
                give one inbound call mixed views of the index. Default
                None preserves backward compat for non-proxy callers.
        """
        start_time = time.monotonic()
        _breakdown: dict[str, float] = {}

        def _note(stage: str, started_at: float) -> None:
            _breakdown[stage] = round((time.monotonic() - started_at) * 1000, 1)

        active_tags = set(current_active_tags or [])
        overflow: list[StoredSummary] = []

        # Tag the inbound message — scope vocabulary to current conversation
        _load_tags_stage = time.monotonic()
        all_tags = self._load_all_tags_snapshot()
        _note("load_all_tags", _load_tags_stage)
        store_tags = [ts.tag for ts in all_tags]
        _tag_stage = time.monotonic()
        if self._inbound_tagger is not None:
            # Embedding-based: match against existing vocabulary (no hallucination)
            # Store tags are conversation-scoped via get_all_tags().
            # TurnTagIndex is inherently per-conversation (loaded from that
            # conversation's engine_state), so all its tags are in scope.
            vocab_tags = list(set(store_tags))
            if self._turn_tag_index:
                vocab_tags = list(set(vocab_tags) | self._turn_tag_index.all_tags())
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
        _note("tag_generate", _tag_stage)

        query_embedding = getattr(tag_result, 'query_embedding', None)

        retrieval_metadata: dict = {}
        retrieval_scores: dict[str, float] = {}

        # Temporal detection is now advisory; time-scoped recall is tool-driven
        # via vc_remember_when rather than an automatic retriever branch.
        if tag_result.temporal:
            retrieval_metadata["temporal_hint"] = True

        if tag_result.source == "fallback":
            # Tagging failed — fall back to working set tags. When the
            # caller supplied a bounded entries_snapshot (proxy mode),
            # the working-set computation uses it to stay consistent
            # with the rest of the inbound call's view of the index.
            if self._turn_tag_index:
                working_set = list(self._turn_tag_index.get_active_tags(
                    lookback=self.config.anchorless_lookback,
                    entries_snapshot=entries_snapshot,
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
            # _general fallback: use previous turn's tags for focused
            # retrieval. Honor the entries_snapshot when supplied so
            # the inherited tags match the same bounded view as the
            # rest of this inbound call.
            if self._turn_tag_index:
                prev = self._turn_tag_index.latest_meaningful_tags(
                    entries_snapshot=entries_snapshot,
                )
                if prev:
                    query_tags = [t for t in prev.tags if t not in self._turn_tag_index._NON_INHERITABLE_TAGS]
                    if query_tags:
                        logger.info(
                            "Retriever: _general fallback → using previous turn tags %s (T%d)",
                            query_tags, prev.turn_number,
                        )
                        retrieval_metadata["general_fallback"] = "previous_turn"

        if not query_tags:
            logger.info(
                "Retriever: no query tags after filtering (message_tags=%s, active_tags=%s, skipped=%s)",
                tag_result.tags, list(active_tags), skipped_tags,
            )
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
                        overflow_summaries=overflow,
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
                overflow_summaries=overflow,
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

        # Expand query with related tags from tagger
        related_query_tags = [
            t for t in tag_result.related_tags
            if t not in set(query_tags) and t != "_general"
        ]
        expanded_tags = list(set(query_tags) | set(related_query_tags))
        query_tag_set = set(query_tags)

        from .retrieval_scoring import score_candidates

        _idf_stage = time.monotonic()
        idf_weights = self._compute_idf_weights(all_tags)
        tag_stats = {ts.tag: ts.usage_count for ts in all_tags}
        _note("idf_prepare", _idf_stage)
        _score_stage = time.monotonic()
        stored_embeddings = (
            self._load_tag_summary_embedding_snapshot()
            if query_embedding is not None else None
        )
        query_embedding_context = None
        if query_embedding is not None:
            query_embedding_context, embed_mode, ctx_embed_ms = (
                self._build_context_query_embedding(message, context_turns)
            )
            if query_embedding_context is not None:
                if ctx_embed_ms is not None:
                    _breakdown["context_embed"] = ctx_embed_ms
                logger.info(
                    "Retriever: embedding-context mode=%s turns=%d embed_ctx=%sms",
                    embed_mode, self.config.scoring.embedding_context_turns,
                    ctx_embed_ms,
                )
        # Resolve the strategy up front — its max_results is K for the
        # embedding reserved-seats pass inside score_candidates.
        strategy = self.config.strategy_configs.get("default")
        if strategy is None:
            from ..types import StrategyConfig
            strategy = StrategyConfig()

        scores, breakdowns = score_candidates(
            query_tags=query_tags,
            related_tags=related_query_tags,
            query_text=message,
            query_embedding=query_embedding,
            store=self.store,
            idf_weights=idf_weights,
            conversation_id=self._conversation_id,
            config=self.config.scoring,
            tag_stats=tag_stats,
            stored_embeddings=stored_embeddings,
            query_embedding_context=query_embedding_context,
            top_k=strategy.max_results,
        )
        _note("score_candidates", _score_stage)
        retrieval_scores = scores

        # Apply token budget

        budget_fraction = strategy.max_budget_fraction
        if current_utilization > 0.5:
            scale = max(0.1, 1.0 - current_utilization)
            budget_fraction *= scale
        token_budget = int(self.config.tag_context_max_tokens * budget_fraction)

        top_tags = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)[:strategy.max_results]

        # Fetch summaries for all top-scored tags at once, then rank by RRF score
        _summary_fetch_stage = time.monotonic()
        all_summaries = self.store.get_summaries_by_tags(
            tags=top_tags, min_overlap=1,
            limit=strategy.max_results * 3,
            conversation_id=self._conversation_id,
        )
        _note("fetch_ranked_summaries", _summary_fetch_stage)
        # Sort by (RRF fused score of best matching tag, IDF query-tag overlap) descending
        def _summary_sort_key(s: StoredSummary) -> tuple[float, float]:
            best_rrf = max((scores.get(t, 0.0) for t in s.tags), default=0.0)
            idf_overlap = sum(idf_weights.get(t, 1.0) for t in s.tags if t in query_tag_set)
            return (best_rrf, idf_overlap)
        all_summaries.sort(key=_summary_sort_key, reverse=True)

        selected: list[StoredSummary] = []
        selected_refs: set[str] = set()
        total_tokens = 0
        _overflow_cap = 50
        for summary in all_summaries:
            if summary.ref in selected_refs:
                continue
            if total_tokens + summary.summary_tokens > token_budget:
                if len(overflow) < _overflow_cap:
                    overflow.append(summary)
                    logger.debug(
                        "Retriever: '%s' OVERFLOW (%dt, %d candidates)",
                        summary.primary_tag, summary.summary_tokens, len(overflow),
                    )
                continue
            selected.append(summary)
            selected_refs.add(summary.ref)
            total_tokens += summary.summary_tokens
            logger.info(
                "Retriever: '%s' INCLUDE (%dt, budget %d/%dt used)",
                summary.primary_tag, summary.summary_tokens, total_tokens, token_budget,
            )

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
            _full_detail_stage = time.monotonic()
            for summary in selected[:3]:  # limit to top 3
                segment = self.store.get_segment(summary.ref, conversation_id=self._conversation_id)
                if segment:
                    full_detail.append(segment)
            _note("fetch_full_detail", _full_detail_stage)

        # Collect matched tags (include related tag matches too)
        expanded_set = set(expanded_tags)
        matched_tags = list({tag for s in selected for tag in s.tags if tag in expanded_set})

        elapsed = time.monotonic() - start_time

        retrieval_metadata.update({
            "elapsed_ms": round(elapsed * 1000, 1),
            "tags_from_message": tag_result.tags,
            "tags_queried": query_tags,
            "tags_skipped_active": skipped_tags,
            "candidates_found": len(all_summaries),
            "summaries_returned": len(selected),
            "strategy": "default",
            "budget_fraction": round(budget_fraction, 3),
            "idf_reranked": bool(idf_weights),
            "related_tags_used": related_query_tags,
            "query_expanded": len(related_query_tags) > 0,
        })

        # Fetch facts: dense-augmented (gated), tag prefetch, or fetch all
        _facts_stage = time.monotonic()
        if getattr(self.config, "fact_dense_retrieval", False):
            facts = self._fetch_facts_dense(
                message, context_turns, expanded_tags, retrieval_metadata,
            )
        elif self.config.prefetch_facts and expanded_tags:
            facts = self._fetch_facts_by_tags(expanded_tags)
            logger.info("Retriever: facts=%d (prefetch tags=%s)", len(facts), expanded_tags)
        else:
            facts = self._fetch_all_facts()
            logger.info("Retriever: facts=%d (all, prefetch=%s)", len(facts),
                        "off" if not self.config.prefetch_facts else "no-tags")
        _note("fetch_facts", _facts_stage)

        total_ms = round((time.monotonic() - start_time) * 1000, 1)
        if total_ms >= _RETRIEVAL_BREAKDOWN_LOG_THRESHOLD_MS:
            stages = " ".join(
                f"{stage}={ms:.1f}ms"
                for stage, ms in sorted(_breakdown.items(), key=lambda item: item[1], reverse=True)
                if ms > 0
            )
            logger.info(
                "RETRIEVE_BREAKDOWN conv=%s total=%sms query_tags=%d expanded=%d selected=%d facts=%d %s",
                (self._conversation_id or "")[:12] or "none",
                total_ms,
                len(query_tags),
                len(expanded_tags),
                len(selected),
                len(facts),
                stages or "no-stages",
            )

        return RetrievalResult(
            tags_matched=matched_tags,
            summaries=selected,
            full_detail=full_detail,
            total_tokens=total_tokens,
            facts=facts,
            overflow_summaries=overflow,
            retrieval_metadata=retrieval_metadata,
            retrieval_scores=retrieval_scores,
            query_embedding=query_embedding,
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
    try:
        aliases = store.get_tag_aliases(conversation_id=conversation_id)
    except TypeError:
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
