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

    _COMPACT_BATCH_SIZE = 20  # segments per compaction batch -> DB after each batch

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
        compact_messages = conversation_history[self._engine_state.compacted_through:-protected_count]

        if not compact_messages:
            logger.info(
                "Compaction skipped: no messages between watermark=%d and protected zone "
                "(history=%d msgs, protected=%d turns)",
                self._engine_state.compacted_through, len(conversation_history), protected_turns,
            )
            return None

        # Hash guard: check if first pair in compact range was already processed
        # (defensive against watermark/index mismatch after restart)
        if len(compact_messages) >= 2:
            import hashlib
            _guard_text = f"{compact_messages[0].content} {compact_messages[1].content}"
            _guard_hash = hashlib.sha256(_guard_text.encode()).hexdigest()[:16]
            _guard_entry = self._turn_tag_index.get_entry_by_hash(_guard_hash)
            if _guard_entry is not None:
                _wm_turn = self._engine_state.compacted_through // 2
                if _guard_entry.turn_number < _wm_turn:
                    logger.warning(
                        "Compaction hash guard: first pair (hash=%s) matches turn %d "
                        "which is below watermark turn %d — skipping to prevent re-compaction",
                        _guard_hash, _guard_entry.turn_number, _wm_turn,
                    )
                    return None

        logger.info(
            "Compacting %d messages (watermark=%d, history=%d, protected=%d turns)",
            len(compact_messages), self._engine_state.compacted_through,
            len(conversation_history), protected_turns,
        )
        report = self._run_compaction(conversation_history, compact_messages, progress_callback=progress_callback)

        self._engine_state.last_compact_ms = round((time.monotonic() - _t_compact) * 1000, 1)
        self._save_state_callback(conversation_history)
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

        compact_messages = conversation_history[self._engine_state.compacted_through:-protected_count]

        if not compact_messages:
            return None

        report = self._run_compaction(conversation_history, compact_messages)

        self._save_state_callback(conversation_history)
        return report

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

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

        # Segment and compact in batches (results stored to DB incrementally)
        turn_offset = self._engine_state.compacted_through // 2
        segments = self._segmenter.segment(compact_messages, turn_offset=turn_offset)
        logger.info(
            "Segmented %d messages into %d segments (watermark=%d)",
            len(compact_messages), len(segments), self._engine_state.compacted_through,
        )
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
                        if progress_callback:
                            try:
                                progress_callback(
                                    ts_i + 1, len(new_tag_summaries), None,
                                    phase="tag_summary_built",
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

    def _compact_and_store(
        self, segments: list, compact_messages_len: int,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[CompactionResult]:
        """Compact segments in batches of ``_COMPACT_BATCH_SIZE`` and store each
        batch immediately so results are visible in the DB incrementally."""
        from ..types import CompactionResult, FactSignal, SegmentMetadata, StoredSegment

        _ensure_engine_imports()

        all_results: list[CompactionResult] = []
        batch_size = self._COMPACT_BATCH_SIZE

        # D1: Gather fact signals from TurnTagIndex scoped per segment.
        # Build a segment-to-turn mapping using turn_offset and each segment's
        # turn_count, then collect only the fact signals for each segment's
        # source turns.
        turn_offset = self._engine_state.compacted_through // 2
        seg_cursor = turn_offset
        segment_signals: dict[str, list[FactSignal]] = {}
        for seg in segments:
            seg_turn_count = getattr(seg, "turn_count", 0) or (len(seg.messages) // 2)
            signals: list[FactSignal] = []
            for t in range(seg_cursor, seg_cursor + seg_turn_count):
                entry = self._turn_tag_index.get_tags_for_turn(t)
                if entry and entry.fact_signals:
                    signals.extend(entry.fact_signals)
            if signals:
                segment_signals[seg.id] = signals
            seg_cursor += seg_turn_count

        for start in range(0, len(segments), batch_size):
            batch = segments[start:start + batch_size]
            batch_num = start // batch_size + 1
            total_batches = (len(segments) + batch_size - 1) // batch_size
            logger.info(
                "Compacting batch %d/%d (%d segments)...",
                batch_num, total_batches, len(batch),
            )
            # D1: Pass per-segment signals to the compactor for verification.
            fact_signals_by_segment = {
                seg.id: segment_signals[seg.id]
                for seg in batch if seg.id in segment_signals
            } or None
            # Filter out stub segments (media placeholders, image stubs) — not worth
            # LLM summarization. Real short messages ("im good") should be compacted
            # normally since they belong to their parent topic segment.
            def _is_passthrough(s):
                text = " ".join(m.content for m in s.messages)
                return _is_stub_content_fn(text)
            compactable = [s for s in batch if not _is_passthrough(s)]
            stubs = [s for s in batch if _is_passthrough(s)]
            for seg in stubs:
                text = " ".join(m.content for m in seg.messages).strip()
                logger.info(
                    "SEGMENT passthrough_stub ref=%s tokens=%d primary=%s — using raw text as summary",
                    seg.id[:8], seg.token_count, seg.primary_tag,
                )
                results_tiny = [CompactionResult(
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
                )]
                all_results.extend(results_tiny)
                for result in results_tiny:
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
            if not compactable:
                continue
            batch = compactable
            results = self._compactor.compact(batch, fact_signals_by_segment=fact_signals_by_segment)
            # Store each result to DB right away
            for i, result in enumerate(results):
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
                if progress_callback:
                    try:
                        progress_callback(
                            len(all_results) + i + 1, len(segments), result,
                            phase="segment_stored",
                        )
                    except Exception:
                        pass
                # D1: Store extracted facts with provenance
                if result.facts:
                    for fact in result.facts:
                        fact.segment_ref = stored.ref
                        fact.conversation_id = self._config.conversation_id
                    self._store.store_facts(result.facts)
                    logger.info(
                        "  Stored %d facts for segment %s",
                        len(result.facts), result.primary_tag,
                    )
                    # D1: Run supersession + fact linking
                    _superseded_count = 0
                    _links_count = 0
                    if self._supersession_checker:
                        try:
                            if hasattr(self._supersession_checker, 'check_and_link'):
                                _links_count, _superseded_count = self._supersession_checker.check_and_link(result.facts)
                            else:
                                _superseded_count = self._supersession_checker.check_and_supersede(result.facts) or 0
                            if _superseded_count:
                                logger.info(
                                    "  Superseded %d facts for segment %s",
                                    _superseded_count, result.primary_tag,
                                )
                            if _links_count:
                                logger.info(
                                    "  Linked %d facts for segment %s",
                                    _links_count, result.primary_tag,
                                )
                        except Exception as e:
                            logger.warning("Supersession/linking failed: %s", e)
                    if progress_callback:
                        try:
                            progress_callback(
                                len(all_results) + i + 1, len(segments), result,
                                phase="facts_extracted",
                                fact_count=len(result.facts),
                                superseded_count=_superseded_count,
                                links_count=_links_count,
                            )
                        except Exception:
                            pass
                session_date = getattr(result.metadata, 'session_date', '') if result.metadata else ''
                logger.info(
                    "  Stored segment %d/%d: %s (session_date=%s, %dt→%dt)",
                    start + i + 1, len(segments), result.primary_tag,
                    session_date or 'none',
                    result.original_tokens, result.summary_tokens,
                )
            all_results.extend(results)

        return all_results
