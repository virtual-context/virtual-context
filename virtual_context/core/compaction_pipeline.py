"""CompactionPipeline: segmentation, compaction, and storage.

Extracted from engine.py — handles Phase 2 of turn processing (compact_if_needed),
manual compaction (compact_manual), and the shared compaction core (_run_compaction).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

from .engine_utils import extract_turn_pairs
from .store import ContextStore
from .turn_tag_index import TurnTagIndex

if TYPE_CHECKING:
    from .compactor import DomainCompactor
    from .segmenter import TopicSegmenter
    from .semantic_search import SemanticSearchManager
    from .telemetry import TelemetryLedger
    from ..types import (
        CompactionReport,
        CompactionResult,
        CompactionSignal,
        EngineState,
        Message,
        SegmentMetadata,
        StoredSegment,
        VirtualContextConfig,
    )

logger = logging.getLogger(__name__)

# Lazy-import for _is_stub_content from engine to avoid circular imports.
_is_stub_content_fn: Callable[[str], bool] | None = None


def _ensure_engine_imports() -> None:
    """Lazy-import module-level symbols from engine to avoid circular imports."""
    global _is_stub_content_fn
    if _is_stub_content_fn is None:
        from ..engine import _is_stub_content as _stub
        _is_stub_content_fn = _stub


class CompactionPipeline:
    """Segmentation, compaction, storage, and tag summary building.

    Owns the ``compact_if_needed`` and ``compact_manual`` entry points as well
    as the shared ``_run_compaction`` core that both call.

    Constructor dependencies mirror what the engine previously wired internally.
    """

    def __init__(
        self,
        compactor: DomainCompactor | None,
        segmenter: TopicSegmenter,
        store: ContextStore,
        turn_tag_index: TurnTagIndex,
        engine_state: EngineState,
        config: VirtualContextConfig,
        supersession_checker,
        fact_curator,
        semantic: SemanticSearchManager,
        telemetry: TelemetryLedger,
        save_state_callback: Callable,
    ) -> None:
        self._compactor = compactor
        self._segmenter = segmenter
        self._store = store
        self._turn_tag_index = turn_tag_index
        self._engine_state = engine_state
        self._config = config
        self._supersession_checker = supersession_checker
        self._fact_curator = fact_curator
        self._semantic = semantic
        self._telemetry = telemetry
        self._save_state_callback = save_state_callback

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def compact_if_needed(
        self,
        conversation_history: list[Message],
        signal: CompactionSignal,
        progress_callback: Callable[..., None] | None = None,
    ) -> CompactionReport | None:
        """Phase 2 of turn processing: run compaction.

        Slow (~10s with LLM summarizer). Can run in background after
        tag_turn() completes — the next request only needs the tag index.

        *signal*: the CompactionSignal returned by tag_turn().
        """
        _t_compact = time.monotonic()

        if self._compactor is None:
            logger.warning(
                "Compaction triggered but no LLM provider configured. "
                "Configure a provider in the providers section."
            )
            return None

        logger.info(
            f"Compaction triggered ({signal.priority}): "
            f"{signal.current_tokens}/{signal.budget_tokens} tokens, "
            f"overflow={signal.overflow_tokens}"
        )

        # Select messages to compact (not in protected zone)
        protected_turns = self._config.monitor.protected_recent_turns
        protected_count = protected_turns * 2  # user + assistant per turn

        if len(conversation_history) <= protected_count:
            logger.info("Not enough messages outside protected zone to compact")
            return None

        # Messages to compact: everything between watermark and protected zone.
        # Compact all available messages (not just the minimum) so compaction
        # fires infrequently — one big batch instead of many small ones.
        offset = self._engine_state.history_offset(len(conversation_history))
        compact_messages = conversation_history[offset:-protected_count]

        if not compact_messages:
            logger.info(
                "Compaction skipped: no messages between offset=%d (watermark=%d) and protected zone "
                "(history=%d msgs, protected=%d turns)",
                offset, self._engine_state.compacted_through,
                len(conversation_history), protected_turns,
            )
            return None

        logger.info(
            "Compacting %d messages (offset=%d, watermark=%d, history=%d, protected=%d turns)",
            len(compact_messages), offset, self._engine_state.compacted_through,
            len(conversation_history), protected_turns,
        )
        report = self._run_compaction(conversation_history, compact_messages, progress_callback=progress_callback)

        self._engine_state.last_compact_ms = round((time.monotonic() - _t_compact) * 1000, 1)
        self._commit_compaction_state(conversation_history)
        return report

    def compact_manual(
        self,
        conversation_history: list[Message],
    ) -> CompactionReport | None:
        """Trigger manual compaction regardless of thresholds.

        Uses the same pipeline as on_turn_complete: respects the compaction
        watermark, protected recent turns, advances the watermark, stores
        segments, and rebuilds tag summaries for affected tags.
        """
        if self._compactor is None:
            logger.warning("No LLM provider configured for compaction")
            return None

        if not conversation_history:
            return None

        # Select messages to compact (same logic as on_turn_complete)
        protected_turns = self._config.monitor.protected_recent_turns
        protected_count = protected_turns * 2

        if len(conversation_history) <= protected_count:
            logger.info("Not enough messages outside protected zone to compact")
            return None

        offset = self._engine_state.history_offset(len(conversation_history))
        compact_messages = conversation_history[offset:-protected_count]

        if not compact_messages:
            return None

        report = self._run_compaction(conversation_history, compact_messages)

        self._commit_compaction_state(conversation_history)
        return report

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _propagate_tool_output_links(
        self, segment_ref: str, turn_start: int, turn_end: int,
    ) -> None:
        """Copy turn-level tool output links to the segment join table.

        Iterates turns in ``[turn_start, turn_end)`` and for each turn that
        has ``turn_tool_outputs`` entries, writes a corresponding
        ``segment_tool_outputs`` row.  Non-critical — failures are silenced.
        """
        try:
            for t in range(turn_start, turn_end):
                refs = self._store.get_tool_outputs_for_turn(
                    self._config.conversation_id, t,
                )
                for ref in refs:
                    self._store.link_segment_tool_output(
                        self._config.conversation_id, segment_ref, ref,
                    )
        except Exception:
            pass  # non-critical

    def _run_compaction(
        self,
        conversation_history: list[Message],
        compact_messages: list[Message],
        progress_callback: Callable[..., None] | None = None,
    ) -> CompactionReport:
        """Shared compaction core: segment, compact, store, build tag summaries.

        Called by both ``compact_if_needed`` (threshold-triggered) and
        ``compact_manual`` (explicit) after their respective guard checks
        have selected *compact_messages*.

        Returns a CompactionReport (never None — callers handle None guards).
        """
        from ..types import CompactionReport

        turn_offset = self._engine_state.compacted_through // 2

        def _emit_weighted_progress(
            done: int,
            total: int,
            result,
            *,
            phase: str,
            phase_name: str,
            base_percent: int,
            span_percent: int,
            **kwargs,
        ) -> None:
            if not progress_callback:
                return
            bounded_total = max(total, 1)
            bounded_done = max(0, min(done, bounded_total))
            overall_percent = base_percent + int(span_percent * bounded_done / bounded_total)
            progress_callback(
                done,
                total,
                result,
                phase=phase,
                overall_percent=overall_percent,
                phase_name=phase_name,
                **kwargs,
            )

        def _segmenter_progress(done: int, total: int, result, **kwargs) -> None:
            _emit_weighted_progress(
                done,
                total,
                result,
                phase="segmenter",
                phase_name=str(kwargs.pop("phase_name", "segmenter")),
                base_percent=0,
                span_percent=25,
                **kwargs,
            )

        # Phase 1: Segmenter (0-25%)
        segments = self._segmenter.segment(
            compact_messages,
            turn_offset=turn_offset,
            progress_callback=_segmenter_progress,
        )
        logger.info(
            "Segmented %d messages into %d segments (watermark=%d)",
            len(compact_messages), len(segments), self._engine_state.compacted_through,
        )

        # Phase 2+3: Compact + Store (25-75%)
        results = self._compact_and_store(segments, len(compact_messages), progress_callback=progress_callback)

        # Advance watermark past compacted messages
        self._engine_state.compacted_through += len(compact_messages)

        tokens_freed = sum(r.original_tokens - r.summary_tokens for r in results)
        tags = list({tag for r in results for tag in r.tags})

        # Build/update tag summaries — only for tags in newly compacted segments
        tag_summaries_built = 0
        cover_tags: list[str] = []
        if results and self._compactor:
            # Only rebuild tag summaries for tags that were just compacted
            compacted_tags = {tag for r in results for tag in r.tags}
            cover_tags = [
                t for t in self._turn_tag_index.compute_cover_set()
                if t in compacted_tags
            ]
            # Primary tag guarantee: ensure every segment's primary_tag gets
            # a tag summary, even if the greedy set cover dropped it.
            # Without this, ephemeral topics (2-3 turns) lose their most
            # specific tag to broader tags that cover more segments.
            cover_set = set(cover_tags)
            for r in results:
                if r.primary_tag and r.primary_tag not in cover_set:
                    cover_tags.append(r.primary_tag)
                    cover_set.add(r.primary_tag)
            if cover_tags:
                # Gather segment summaries per cover tag
                tag_to_summaries: dict[str, list] = {}
                for tag in cover_tags:
                    summaries = self._store.get_summaries_by_tags(
                        tags=[tag], min_overlap=1, limit=50,
                        conversation_id=self._config.conversation_id,
                    )
                    if summaries:
                        tag_to_summaries[tag] = summaries

                # Gather turn numbers per tag from index
                tag_to_turns: dict[str, list[int]] = {}
                for entry in self._turn_tag_index.entries:
                    for tag in entry.tags:
                        if tag in cover_tags:
                            tag_to_turns.setdefault(tag, []).append(entry.turn_number)

                # Load existing tag summaries for staleness check
                existing_tag_summaries = {}
                for tag in cover_tags:
                    ts = self._store.get_tag_summary(tag, conversation_id=self._config.conversation_id)
                    if ts:
                        existing_tag_summaries[tag] = ts

                if self._turn_tag_index.entries:
                    max_turn = max(e.turn_number for e in self._turn_tag_index.entries)

                    new_tag_summaries = self._compactor.compact_tag_summaries(
                        cover_tags=cover_tags,
                        tag_to_summaries=tag_to_summaries,
                        tag_to_turns=tag_to_turns,
                        existing_tag_summaries=existing_tag_summaries,
                        max_turn=max_turn,
                    )

                    for ts_i, ts in enumerate(new_tag_summaries):
                        self._store.save_tag_summary(ts, conversation_id=self._config.conversation_id)
                        # Compute and store tag summary embedding for RRF scoring
                        try:
                            embed_fn = self._semantic.get_embed_fn()
                            if embed_fn and ts.summary:
                                emb = embed_fn([ts.summary[:2000]])[0]
                                self._store.store_tag_summary_embedding(
                                    ts.tag, self._config.conversation_id, emb,
                                )
                        except Exception as e:
                            logger.debug("Failed to embed tag summary '%s': %s", ts.tag, e)
                        if progress_callback:
                            try:
                                _pct = 95 + int(5 * (ts_i + 1) / max(len(new_tag_summaries), 1))
                                progress_callback(
                                    ts_i + 1, len(new_tag_summaries), None,
                                    phase="tag_summary_built",
                                    overall_percent=_pct,
                                    phase_name="tag_summaries",
                                    tag=ts.tag,
                                )
                            except Exception:
                                pass
                    tag_summaries_built = len(new_tag_summaries)

        report = CompactionReport(
            segments_compacted=len(results),
            tokens_freed=tokens_freed,
            tags=tags,
            results=results,
            tag_summaries_built=tag_summaries_built,
            cover_tags=cover_tags,
        )

        # Enforce TTL from tag rules
        if self._config.tag_rules:
            min_ttl = min(
                (r.ttl_days for r in self._config.tag_rules if r.ttl_days is not None),
                default=None,
            )
            if min_ttl is not None:
                self._store.cleanup(max_age=timedelta(days=min_ttl))

        return report

    def _commit_compaction_state(self, conversation_history: list[Message]) -> None:
        """Persist the committed compaction checkpoint, then prune raw turns from that prefix."""
        expected_last_compacted_turn = int(self._engine_state.last_compacted_turn)
        saved = self._save_state_callback(conversation_history)
        if not saved:
            logger.warning(
                "Compaction checkpoint save failed for conversation %s; skipping turn_messages prune",
                self._config.conversation_id[:12],
            )
            return
        prune = getattr(self._store, "prune_turn_messages", None)
        if prune is None:
            return
        try:
            committed = self._store.load_engine_state(self._config.conversation_id)
        except Exception:
            logger.warning(
                "Failed to reload committed compaction checkpoint for conversation %s before pruning turn_messages",
                self._config.conversation_id[:12],
                exc_info=True,
            )
            return
        if not committed:
            return
        committed_turn = int(getattr(committed, "last_compacted_turn", -1) or -1)
        if committed_turn < expected_last_compacted_turn:
            logger.warning(
                "Committed compaction checkpoint regressed for conversation %s (%d < %d); skipping turn_messages prune",
                self._config.conversation_id[:12],
                committed_turn,
                expected_last_compacted_turn,
            )
            return
        keep_from_turn = committed_turn + 1
        try:
            removed = self._store.prune_turn_messages(
                self._config.conversation_id,
                keep_from_turn,
            )
        except Exception:
            logger.warning(
                "Committed compaction prune failed for conversation %s at keep_from_turn=%d",
                self._config.conversation_id[:12],
                keep_from_turn,
                exc_info=True,
            )
            return
        if removed:
            logger.info(
                "Pruned %d committed turn_messages before turn %d for conversation %s",
                removed,
                keep_from_turn,
                self._config.conversation_id[:12],
            )

    def _compact_and_store(
        self, segments: list, compact_messages_len: int,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[CompactionResult]:
        """Two-pass compact and store.

        Pass 1 (sequential, no LLM): handle stubs, check store for merge
        candidates, combine turns where matches are found.

        Pass 2 (batch, LLM): compact all prepared segments, then store results.
        """
        from datetime import datetime, timezone

        from ..types import CompactionResult, FactSignal, Message, SegmentMetadata, StoredSegment
        from .tag_scoring import compute_relatedness

        _ensure_engine_imports()

        all_results: list[CompactionResult] = []

        def _emit_progress(
            done: int,
            total: int,
            result,
            *,
            phase: str,
            phase_name: str,
            base_percent: int,
            span_percent: int,
            **kwargs,
        ) -> None:
            if not progress_callback:
                return
            bounded_total = max(total, 1)
            bounded_done = max(0, min(done, bounded_total))
            overall_percent = base_percent + int(span_percent * bounded_done / bounded_total)
            progress_callback(
                done,
                total,
                result,
                phase=phase,
                overall_percent=overall_percent,
                phase_name=phase_name,
                **kwargs,
            )

        # D1: Gather fact signals from TurnTagIndex scoped per segment.
        # Also record the contributing turn range per segment for tool-output linkage.
        turn_offset = self._engine_state.compacted_through // 2
        seg_cursor = turn_offset
        segment_signals: dict[str, list[FactSignal]] = {}
        segment_turn_ranges: dict[str, tuple[int, int]] = {}  # seg.id -> (start, end_exclusive)
        for seg in segments:
            seg_turn_count = getattr(seg, "turn_count", 0) or (len(seg.messages) // 2)
            segment_turn_ranges[seg.id] = (seg_cursor, seg_cursor + seg_turn_count)
            signals: list[FactSignal] = []
            for t in range(seg_cursor, seg_cursor + seg_turn_count):
                entry = self._turn_tag_index.get_tags_for_turn(t)
                if entry and entry.fact_signals:
                    signals.extend(entry.fact_signals)
            if signals:
                segment_signals[seg.id] = signals
            seg_cursor += seg_turn_count

        merge_lookback = self._config.compactor.merge_lookback
        max_seg_tokens = self._config.compactor.max_segment_tokens
        merge_threshold = self._config.compactor.merge_overlap_threshold

        # ==================================================================
        # Pass 1: Sequential pre-pass — stubs + merge check (no LLM calls)
        # ==================================================================
        compactable: list = []  # segments ready for LLM compaction
        now = datetime.now(timezone.utc)

        # P1: pre-load embeddings and embed_fn once (not per-segment)
        stored_embeddings = self._store.load_tag_summary_embeddings(
            conversation_id=self._config.conversation_id,
        )
        embed_fn = self._semantic.get_embed_fn() if self._semantic else None

        for seg in segments:
            # --- Stub passthrough (no LLM) ---
            text = " ".join(m.content for m in seg.messages)
            if _is_stub_content_fn(text):
                text = text.strip()
                logger.info(
                    "SEGMENT passthrough_stub ref=%s tokens=%d primary=%s",
                    seg.id[:8], seg.token_count, seg.primary_tag,
                )
                result = CompactionResult(
                    segment_id=seg.id,
                    primary_tag=seg.primary_tag,
                    tags=seg.tags,
                    summary=text or f"[empty turn: {seg.primary_tag}]",
                    summary_tokens=seg.token_count,
                    full_text=text,
                    original_tokens=seg.token_count,
                    messages=[{"role": m.role, "content": m.content} for m in seg.messages],
                    metadata=SegmentMetadata(
                        turn_count=seg.turn_count,
                        session_date=getattr(seg, "session_date", ""),
                    ),
                    compression_ratio=1.0,
                    timestamp=seg.start_timestamp,
                )
                stored = StoredSegment(
                    ref=result.segment_id,
                    conversation_id=self._config.conversation_id,
                    primary_tag=result.primary_tag,
                    tags=result.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model="passthrough",
                    compression_ratio=1.0,
                    start_timestamp=result.timestamp,
                    end_timestamp=result.timestamp,
                )
                self._store.store_segment(stored)
                # Propagate turn → segment tool output links
                turn_range = segment_turn_ranges.get(seg.id)
                if turn_range:
                    self._propagate_tool_output_links(stored.ref, *turn_range)
                all_results.append(result)
                continue

            # --- Merge check: find best existing segment to merge with ---
            if merge_lookback > 0:
                candidates = self._store.get_segments_by_tags(
                    tags=seg.tags, min_overlap=1, limit=merge_lookback,
                    conversation_id=self._config.conversation_id,
                )
                seg_tags = set(seg.tags)
                seg_text = " ".join(m.content for m in seg.messages)[:2000]
                # B4: Pre-compute segment embedding once (not per-candidate)
                seg_embedding = None
                if embed_fn and seg_text:
                    try:
                        seg_embedding = embed_fn([seg_text])[0]
                    except Exception:
                        pass
                best_score = 0.0
                best_candidate = None

                for candidate in candidates:
                    combined_tokens = candidate.full_tokens + seg.token_count
                    if combined_tokens > max_seg_tokens:
                        continue
                    # Multi-signal relatedness: tag overlap + embedding + keyword
                    cand_embedding = stored_embeddings.get(candidate.primary_tag)
                    relatedness = compute_relatedness(
                        tags_a=seg_tags,
                        tags_b=set(candidate.tags),
                        text_a=seg_text,
                        text_b=candidate.summary[:2000] if candidate.summary else "",
                        embedding_a=seg_embedding,
                        embedding_b=cand_embedding,
                    )
                    if relatedness < merge_threshold:
                        continue
                    try:
                        age_days = (now - candidate.created_at).days
                    except (TypeError, AttributeError):
                        age_days = 30
                    recency = max(0.5, 1.0 - age_days / 60)
                    combined_score = relatedness * recency
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate

                if best_candidate is not None:
                    # Combine turns: prepend existing segment's messages
                    candidate_messages = [
                        Message(role=m.get("role", "user"), content=m.get("content", ""))
                        for m in best_candidate.messages
                    ]
                    seg.messages = candidate_messages + list(seg.messages)
                    seg.merge_ref = best_candidate.ref
                    seg.token_count += best_candidate.full_tokens
                    seg.start_timestamp = best_candidate.start_timestamp
                    old_tc = best_candidate.metadata.turn_count if best_candidate.metadata else len(best_candidate.messages) // 2
                    seg.turn_count += old_tc
                    seg.tags = list(set(best_candidate.tags) | seg_tags)
                    logger.info(
                        "MERGE PREP: segment '%s' (%s) merging with stored %s "
                        "(%s, %d existing turns, relatedness=%.2f)",
                        seg.id[:8], seg.primary_tag,
                        best_candidate.ref[:8], best_candidate.primary_tag,
                        old_tc, best_score,
                    )

            compactable.append(seg)

        if not compactable:
            if all_results:
                _emit_progress(
                    len(all_results),
                    len(all_results),
                    all_results[-1],
                    phase="segment_stored",
                    phase_name="store",
                    base_percent=80,
                    span_percent=15,
                )
            return all_results

        logger.info("Pass 1 complete: %d stubs stored, %d segments ready for compaction (%d merges)",
                    len(all_results), len(compactable),
                    sum(1 for s in compactable if s.merge_ref))

        # ==================================================================
        # Pass 2: Batch LLM compaction + store
        # ==================================================================
        fact_signals_by_segment = {
            seg.id: segment_signals[seg.id]
            for seg in compactable if seg.id in segment_signals
        } or None

        def _compactor_progress(done: int, total: int, result, **kwargs) -> None:
            _emit_progress(
                done,
                total,
                result,
                phase="segment_compacting",
                phase_name=str(kwargs.pop("phase_name", "compactor")),
                base_percent=25,
                span_percent=55,
                **kwargs,
            )

        results = self._compactor.compact(
            compactable,
            fact_signals_by_segment=fact_signals_by_segment,
            progress_callback=_compactor_progress,
        )

        for seg_idx, result in enumerate(results):
            seg = compactable[seg_idx]

            # Store or update
            if seg.merge_ref:
                stored = StoredSegment(
                    ref=seg.merge_ref,
                    conversation_id=self._config.conversation_id,
                    primary_tag=result.primary_tag,
                    tags=seg.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model=self._compactor.model_name,
                    compression_ratio=result.compression_ratio,
                    start_timestamp=seg.start_timestamp,
                    end_timestamp=result.timestamp,
                )
                self._store.update_segment(stored)
                self._semantic.embed_and_store_chunks(stored)
                result.segment_id = seg.merge_ref
                session_date = getattr(result.metadata, 'session_date', '') if result.metadata else ''
                logger.info(
                    "  COMPACT MERGED %d/%d: %s (session_date=%s, %dt→%dt, %d turns)",
                    seg_idx + 1, len(results), result.primary_tag,
                    session_date or 'none',
                    result.original_tokens, result.summary_tokens, seg.turn_count,
                )
            else:
                stored = StoredSegment(
                    ref=result.segment_id,
                    conversation_id=self._config.conversation_id,
                    primary_tag=result.primary_tag,
                    tags=result.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model=self._compactor.model_name,
                    compression_ratio=result.compression_ratio,
                    start_timestamp=result.timestamp,
                    end_timestamp=result.timestamp,
                )
                self._store.store_segment(stored)
                self._semantic.embed_and_store_chunks(stored)
                session_date = getattr(result.metadata, 'session_date', '') if result.metadata else ''
                logger.info(
                    "  COMPACT NEW %d/%d: %s (session_date=%s, %dt→%dt, %d turns)",
                    seg_idx + 1, len(results), result.primary_tag,
                    session_date or 'none',
                    result.original_tokens, result.summary_tokens, seg.turn_count,
                )

            # Propagate turn → segment tool output links
            turn_range = segment_turn_ranges.get(seg.id)
            if turn_range:
                self._propagate_tool_output_links(stored.ref, *turn_range)

            all_results.append(result)
            stored_done = seg_idx + 1

            _emit_progress(
                stored_done,
                len(results),
                result,
                phase="segment_stored",
                phase_name="store",
                base_percent=80,
                span_percent=15,
            )

            _seg_ref = stored.ref
            if result.facts:
                for fact in result.facts:
                    fact.segment_ref = _seg_ref
                    fact.conversation_id = self._config.conversation_id
                self._store.store_facts(result.facts)
                logger.info("  Stored %d facts for segment %s", len(result.facts), result.primary_tag)
                _superseded_count = 0
                _links_count = 0
                if self._supersession_checker:
                    try:
                        if hasattr(self._supersession_checker, 'check_and_link'):
                            _links_count, _superseded_count = self._supersession_checker.check_and_link(result.facts)
                        else:
                            _superseded_count = self._supersession_checker.check_and_supersede(result.facts) or 0
                        if _superseded_count:
                            logger.info("  Superseded %d facts for segment %s", _superseded_count, result.primary_tag)
                        if _links_count:
                            logger.info("  Linked %d facts for segment %s", _links_count, result.primary_tag)
                    except Exception as e:
                        logger.warning("Supersession/linking failed: %s", e)
                _emit_progress(
                    stored_done,
                    len(results),
                    result,
                    phase="facts_extracted",
                    phase_name="store",
                    base_percent=80,
                    span_percent=15,
                    fact_count=len(result.facts),
                    superseded_count=_superseded_count,
                    links_count=_links_count,
                )

        return all_results
