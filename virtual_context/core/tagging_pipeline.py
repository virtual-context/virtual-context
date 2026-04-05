"""TaggingPipeline: turn tagging and history ingestion.

Extracted from engine.py — handles Phase 1 of turn processing (tag_turn),
bulk historical ingestion (ingest_history), and tag splitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .engine_utils import extract_turn_pairs, get_recent_context
from .store import ContextStore
from .turn_tag_index import TurnTagIndex

if TYPE_CHECKING:
    from .compactor import DomainCompactor
    from .monitor import ContextMonitor
    from .semantic_search import SemanticSearchManager
    from .tag_canonicalizer import TagCanonicalizer
    from .tag_generator import TagGenerator
    from .tag_splitter import TagSplitter
    from .telemetry import TelemetryLedger
    from ..types import (
        CompactionSignal,
        EngineState,
        Message,
        SplitResult,
        StoredSummary,
        TagResult,
        TurnTagEntry,
        VirtualContextConfig,
    )

logger = logging.getLogger(__name__)

_TAG_BREAKDOWN_LOG_THRESHOLD_MS = 250.0
_TAG_BREAKDOWN_MAX_STAGES = 8

# Imported lazily at module level so tests that patch engine._is_stub_content
# still work.  The actual definitions live in engine.py.
_SESSION_HEADER_RE: re.Pattern | None = None
_is_stub_content_fn: Callable[[str], bool] | None = None


def _ensure_engine_imports() -> None:
    """Lazy-import module-level symbols from engine to avoid circular imports."""
    global _SESSION_HEADER_RE, _is_stub_content_fn
    if _SESSION_HEADER_RE is None:
        from ..engine import _SESSION_HEADER_RE as _hdr, _is_stub_content as _stub
        _SESSION_HEADER_RE = _hdr
        _is_stub_content_fn = _stub


class TaggingPipeline:
    """Turn tagging, tag splitting, and history ingestion.

    Constructor takes:
        tag_generator:       TagGenerator instance
        turn_tag_index:      TurnTagIndex (shared mutable reference)
        store:               ContextStore instance
        semantic:            SemanticSearchManager instance
        engine_state:        EngineState (shared mutable dataclass)
        config:              VirtualContextConfig instance
        tag_splitter:        TagSplitter | None
        canonicalizer:       TagCanonicalizer | None
        telemetry:           TelemetryLedger instance
        monitor:             ContextMonitor instance
        compactor:           DomainCompactor | None
        save_state_callback: Callable[[list[Message]], None]
    """

    def __init__(
        self,
        tag_generator: TagGenerator,
        turn_tag_index: TurnTagIndex,
        store: ContextStore,
        semantic: SemanticSearchManager,
        engine_state: EngineState,
        config: VirtualContextConfig,
        tag_splitter: TagSplitter | None,
        canonicalizer: TagCanonicalizer | None,
        telemetry: TelemetryLedger,
        monitor: ContextMonitor,
        compactor: DomainCompactor | None,
        save_state_callback: Callable,
        next_tool_tag_callback: Callable[[], int] | None = None,
    ) -> None:
        self._tag_generator = tag_generator
        self._turn_tag_index = turn_tag_index
        self._store = store
        self._semantic = semantic
        self._engine_state = engine_state
        self.config = config
        self._tag_splitter = tag_splitter
        self._canonicalizer = canonicalizer
        self._telemetry = telemetry
        self._monitor = monitor
        self._compactor = compactor
        self._save_state_callback = save_state_callback
        self._next_tool_tag = next_tool_tag_callback

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tool_turn(messages: list[Message]) -> bool:
        """Check if a turn is tool-only: has tool blocks in raw_content and empty text content."""
        combined_text = " ".join(m.content for m in messages)
        if combined_text.strip():
            return False  # has real text content — use LLM tagger
        has_tool_block = False
        for m in messages:
            if not m.raw_content:
                continue
            for block in m.raw_content:
                if block.get("type") in ("tool_use", "tool_result"):
                    has_tool_block = True
                    break
            if has_tool_block:
                break
        return has_tool_block

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _link_turn_tool_outputs(
        self,
        turn_number: int,
        refs: list[str] | None = None,
    ) -> None:
        """Link intercepted tool outputs to a canonical turn.

        ``refs`` is the preferred path: ingestion computes canonical turn
        ownership from parsed payload history and passes the matching tool
        output refs directly. When ``refs`` is omitted, fall back to the legacy
        request-time ``tool_outputs.turn`` lookup for compatibility with older
        single-turn flows.
        """
        try:
            if refs is None:
                refs = self._store.get_tool_output_refs_for_turn(
                    self.config.conversation_id, turn_number,
                )
            for ref in refs:
                self._store.link_turn_tool_output(
                    self.config.conversation_id, turn_number, ref,
                )
        except Exception:
            pass  # non-critical

    def _get_latest_turn_pair(self, history: list[Message]) -> list[Message] | None:
        """Extract the most recent user+assistant pair."""
        if len(history) < 2:
            return None
        for i in range(len(history) - 1, 0, -1):
            if history[i].role == "assistant" and history[i-1].role == "user":
                return [history[i-1], history[i]]
        return None

    def _get_recent_context(
        self, history: list[Message], n_pairs: int, exclude_last: int = 2,
        current_text: str | None = None,
    ) -> list[str] | None:
        """Collect up to *n_pairs* recent user+assistant text strings."""
        return get_recent_context(
            history,
            n_pairs,
            semantic=self._semantic,
            bleed_threshold=self.config.tag_generator.context_bleed_threshold,
            exclude_last=exclude_last,
            current_text=current_text,
        )

    def _collect_turn_text(
        self, tag: str, history: list[Message],
    ) -> list[tuple[int, str]]:
        """Collect truncated user text for turns tagged with the given tag."""
        pairs = extract_turn_pairs(history)
        result = []
        for entry in self._turn_tag_index.entries:
            if tag in entry.tags:
                if entry.turn_number < len(pairs):
                    text = pairs[entry.turn_number][0][:200]
                    result.append((entry.turn_number, text))
        return result

    def _build_broad_tag_summary(
        self, tag: str, history: list[Message],
    ) -> None:
        """Build a tag summary directly from raw turn text for unsplittable broad tags."""
        if not self._compactor:
            return

        from ..types import StoredSummary

        pairs = extract_turn_pairs(history)
        texts = []
        turn_numbers = []
        for entry in self._turn_tag_index.entries:
            if tag in entry.tags:
                if entry.turn_number < len(pairs):
                    user_text, assistant_text = pairs[entry.turn_number]
                    texts.append(
                        f"User: {user_text[:300]}\n"
                        f"Assistant: {assistant_text[:300]}"
                    )
                    turn_numbers.append(entry.turn_number)

        if not texts:
            return

        combined = "\n\n---\n\n".join(texts)
        max_turn = max(turn_numbers) if turn_numbers else 0

        synthetic = [StoredSummary(
            ref=f"broad-{tag}",
            tags=[tag],
            summary=combined[:4000],
            summary_tokens=len(combined[:4000]) // 4,
        )]
        summaries = self._compactor.compact_tag_summaries(
            cover_tags=[tag],
            tag_to_summaries={tag: synthetic},
            tag_to_turns={tag: turn_numbers},
            existing_tag_summaries={},
            max_turn=max_turn,
        )
        for ts in summaries:
            self._store.save_tag_summary(ts, conversation_id=self.config.conversation_id)

    @staticmethod
    def _record_timing(
        breakdown: dict[str, float],
        stage: str,
        started_at: float,
    ) -> float:
        elapsed = round((time.monotonic() - started_at) * 1000, 1)
        breakdown[stage] = round(breakdown.get(stage, 0.0) + elapsed, 1)
        return elapsed

    def _log_breakdown(
        self,
        label: str,
        *,
        turn_number: int,
        total_ms: float,
        breakdown: dict[str, float],
        extras: list[str],
    ) -> None:
        if total_ms < _TAG_BREAKDOWN_LOG_THRESHOLD_MS:
            return
        stages = sorted(
            ((stage, ms) for stage, ms in breakdown.items() if ms > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:_TAG_BREAKDOWN_MAX_STAGES]
        stage_bits = [f"{stage}={ms:.1f}ms" for stage, ms in stages]
        parts = [*extras, *stage_bits]
        logger.info(
            "%s conv=%s turn=%d total=%.1fms %s",
            label,
            self.config.conversation_id[:12],
            turn_number,
            total_ms,
            " ".join(parts) if parts else "no-stages",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_turn(
        self,
        conversation_history: list[Message],
        payload_tokens: int | None = None,
        *,
        run_broad_split: bool = True,
    ) -> CompactionSignal | None:
        """Phase 1 of turn processing: tag the latest turn and check thresholds.

        Fast (~2-3s with LLM tagger). Must complete before the next inbound
        request so the turn-tag index is up-to-date for retrieval.

        Returns a CompactionSignal if compaction is needed, None otherwise.

        *payload_tokens* (proxy mode): actual client payload token count.
        Overrides the stripped conversation_history token count in the
        compaction monitor so thresholds trigger at the right level.
        """
        from ..types import TagResult, TurnTagEntry, get_sender_name

        turn_number = len(self._turn_tag_index.entries)
        tag_started = time.monotonic()
        breakdown: dict[str, float] = {}

        # Tag the latest round trip
        latest_pair = self._get_latest_turn_pair(conversation_history)
        sender = get_sender_name(latest_pair[0].metadata) if latest_pair else ""
        if latest_pair:
            combined_text = " ".join(m.content for m in latest_pair)

            # Tool-only turns: skip LLM tagger, assign sequential tool_N tag
            if self._is_tool_turn(latest_pair):
                if self._next_tool_tag is not None:
                    tag_num = self._next_tool_tag()
                else:
                    self._engine_state.tool_tag_counter += 1
                    tag_num = self._engine_state.tool_tag_counter
                tag_name = f"tool_{tag_num}"
                self._turn_tag_index.append(TurnTagEntry(
                    turn_number=len(self._turn_tag_index.entries),
                    message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                    tags=[tag_name],
                    primary_tag=tag_name,
                    sender=sender or "",
                    session_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                ))
                latest_pair = None  # skip normal tagger flow below

            # BUG-013: Skip empty turns with no tool blocks
            elif not combined_text.strip():
                latest_pair = None

        if latest_pair:
            t_stage = time.monotonic()
            store_tags = [
                ts.tag for ts in self._store.get_all_tags(
                    conversation_id=self.config.conversation_id,
                )
            ]
            self._record_timing(breakdown, "load_store_tags", t_stage)
            n_context = self.config.tag_generator.context_lookback_pairs
            t_stage = time.monotonic()
            context = self._get_recent_context(
                conversation_history, n_context, current_text=combined_text,
            )
            self._record_timing(breakdown, "build_context", t_stage)
            t_stage = time.monotonic()
            tag_result = self._tag_generator.generate_tags(
                combined_text, store_tags, context_turns=context,
            )
            self._record_timing(breakdown, "generate_tags", t_stage)

            # Retry with expanded context if only _general was produced
            if tag_result.tags == ["_general"]:
                t_stage = time.monotonic()
                expanded = self._get_recent_context(
                    conversation_history, n_context * 2,
                    current_text=combined_text,
                )
                if expanded:
                    tag_result = self._tag_generator.generate_tags(
                        combined_text, store_tags, context_turns=expanded,
                    )
                self._record_timing(breakdown, "retry_general", t_stage)

            # Final fallback: inherit from most recent meaningful turn
            if tag_result.tags == ["_general"]:
                t_stage = time.monotonic()
                prev = self._turn_tag_index.latest_meaningful_tags()
                if prev:
                    tag_result = TagResult(
                        tags=list(prev.tags),
                        primary=prev.primary_tag,
                        source="inherited",
                    )
                self._record_timing(breakdown, "inherit_fallback", t_stage)

            t_stage = time.monotonic()
            self._turn_tag_index.append(TurnTagEntry(
                turn_number=len(self._turn_tag_index.entries),
                message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                tags=tag_result.tags,
                primary_tag=tag_result.primary,
                fact_signals=tag_result.fact_signals,
                sender=sender or "",
                session_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            ))
            self._record_timing(breakdown, "append_turn_index", t_stage)

        # Build snapshot (only count un-compacted messages)
        t_stage = time.monotonic()
        _total_turns = len(self._turn_tag_index.entries) if self._turn_tag_index else None
        _offset = self._engine_state.history_offset(
            len(conversation_history), total_turns_indexed=_total_turns,
        )
        snapshot = self._monitor.build_snapshot(
            conversation_history[_offset:],
            payload_tokens=payload_tokens,
        )
        self._record_timing(breakdown, "build_snapshot", t_stage)

        # Check thresholds
        t_stage = time.monotonic()
        signal = self._monitor.check(snapshot)
        self._record_timing(breakdown, "monitor_check", t_stage)

        if signal is None:
            self._engine_state.last_compact_ms = 0.0
            # Persist turn message text for post-restart recall
            if latest_pair:
                turn_num = len(self._turn_tag_index.entries) - 1
                t_stage = time.monotonic()
                try:
                    self._store.save_turn_message(
                        self.config.conversation_id,
                        turn_num,
                        latest_pair[0].content if len(latest_pair) > 0 else "",
                        latest_pair[1].content if len(latest_pair) > 1 else "",
                        user_raw_content=json.dumps(latest_pair[0].raw_content) if len(latest_pair) > 0 and latest_pair[0].raw_content else None,
                        assistant_raw_content=json.dumps(latest_pair[1].raw_content) if len(latest_pair) > 1 and latest_pair[1].raw_content else None,
                    )
                except Exception:
                    pass  # never block tagging for message persistence
                self._link_turn_tool_outputs(turn_num)
                self._record_timing(breakdown, "persist_turn_message", t_stage)
            t_stage = time.monotonic()
            self._save_state_callback(
                conversation_history,
                last_indexed_turn=len(self._turn_tag_index.entries) - 1,
            )
            self._record_timing(breakdown, "save_state", t_stage)

        if self._tag_splitter and run_broad_split:
            t_stage = time.monotonic()
            self.process_broad_tag_split(conversation_history, mode="inline")
            self._record_timing(breakdown, "broad_split", t_stage)

        total_ms = round((time.monotonic() - tag_started) * 1000, 1)
        self._engine_state.last_tag_ms = total_ms
        self._log_breakdown(
            "TAG_BREAKDOWN",
            turn_number=turn_number,
            total_ms=total_ms,
            breakdown=breakdown,
            extras=[
                f"history={len(conversation_history)}",
                f"payload={payload_tokens if payload_tokens is not None else 'na'}t",
                f"split_mode={'inline' if self._tag_splitter and run_broad_split else 'deferred' if self._tag_splitter else 'disabled'}",
                f"signal={signal.priority if signal is not None else 'none'}",
            ],
        )

        return signal

    def _check_and_split_broad_tags(
        self, conversation_history: list[Message],
    ) -> SplitResult | None:
        return self.process_broad_tag_split(conversation_history, mode="direct")

    def process_broad_tag_split(
        self,
        conversation_history: list[Message],
        *,
        mode: str = "deferred",
    ) -> SplitResult | None:
        """Check for overly-broad tags and split or summarize them."""
        if not self._tag_splitter:
            return None

        split_started = time.monotonic()
        breakdown: dict[str, float] = {}
        cfg = self.config.tag_generator.tag_splitting
        t_stage = time.monotonic()
        tag_counts = self._turn_tag_index.get_tag_counts()
        total_turns = len(self._turn_tag_index.entries)
        self._record_timing(breakdown, "find_candidates", t_stage)

        if total_turns == 0:
            return None

        # Find candidates: above both thresholds, not already processed
        candidates = [
            (tag, count) for tag, count in tag_counts.items()
            if tag != "_general"
            and tag not in self._engine_state.split_processed_tags
            and count >= cfg.frequency_threshold
            and count / total_turns >= cfg.frequency_pct_threshold
        ]

        if not candidates:
            return None

        # Pick highest-frequency first
        candidates.sort(key=lambda x: -x[1])
        tag, count = candidates[0]

        # Collect turn content
        t_stage = time.monotonic()
        turn_contents = self._collect_turn_text(tag, conversation_history)
        self._record_timing(breakdown, "collect_turn_text", t_stage)
        if not turn_contents:
            self._engine_state.split_processed_tags.add(tag)
            t_stage = time.monotonic()
            self._save_state_callback(
                conversation_history,
                last_indexed_turn=len(self._turn_tag_index.entries) - 1,
            )
            self._record_timing(breakdown, "save_state", t_stage)
            return None

        t_stage = time.monotonic()
        existing_tags = {t for e in self._turn_tag_index.entries for t in e.tags}
        self._record_timing(breakdown, "collect_existing_tags", t_stage)
        t_stage = time.monotonic()
        result = self._tag_splitter.split(tag, turn_contents, existing_tags, total_turns)
        self._record_timing(breakdown, "split_llm", t_stage)

        if result.splittable:
            # Apply split to TurnTagIndex
            t_stage = time.monotonic()
            turn_to_new: dict[int, list[str]] = {}
            for new_tag, turn_numbers in result.groups.items():
                for tn in turn_numbers:
                    turn_to_new.setdefault(tn, []).append(new_tag)
            self._turn_tag_index.replace_tag(tag, turn_to_new)

            # Register alias so old tag queries still resolve
            if self._canonicalizer:
                first_new = next(iter(result.groups))
                self._canonicalizer.register_alias(tag, first_new)

            # Update tagger vocabulary
            if hasattr(self._tag_generator, '_tag_vocabulary'):
                self._tag_generator._tag_vocabulary.pop(tag, None)
                for new_tag, turns in result.groups.items():
                    self._tag_generator._tag_vocabulary[new_tag] = len(turns)
            self._record_timing(breakdown, "apply_split", t_stage)

            logger.info(
                "Split '%s' (%d turns) → %s",
                tag, count, list(result.groups.keys()),
            )
        else:
            # Fallback: build tag summary from raw turn text
            t_stage = time.monotonic()
            self._build_broad_tag_summary(tag, conversation_history)
            self._record_timing(breakdown, "build_summary", t_stage)
            logger.info(
                "Tag '%s' unsplittable (%s), built summary", tag, result.reason,
            )

        self._engine_state.split_processed_tags.add(tag)
        self._engine_state.last_split_result = result
        t_stage = time.monotonic()
        self._save_state_callback(
            conversation_history,
            last_indexed_turn=len(self._turn_tag_index.entries) - 1,
        )
        self._record_timing(breakdown, "save_state", t_stage)
        total_ms = round((time.monotonic() - split_started) * 1000, 1)
        self._log_breakdown(
            "TAG_SPLIT_BREAKDOWN",
            turn_number=len(self._turn_tag_index.entries) - 1,
            total_ms=total_ms,
            breakdown=breakdown,
            extras=[
                f"mode={mode}",
                f"tag={tag}",
                f"candidate_turns={count}",
                f"total_turns={total_turns}",
                f"result={'split' if result.splittable else 'summary'}",
            ],
        )
        return result

    def ingest_history(
        self,
        history_pairs: list[Message],
        progress_callback: Callable[..., None] | None = None,
        turn_offset: int = 0,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> int:
        """Bootstrap TurnTagIndex from pre-existing conversation history.

        Tags each user+assistant pair and appends entries to the live index.
        Does NOT trigger compaction — the next on_turn_complete() handles that.

        Args:
            history_pairs: Flat list [user_0, asst_0, user_1, asst_1, ...].
            progress_callback: Optional ``(done, total, entry)`` called after
                each turn is ingested.  Used by the proxy for live progress.
            turn_offset: Global turn number of the first pair. Used by catch-up
                ingestion to prevent TurnTagIndex overwrites when multiple
                batches are ingested sequentially.
            tool_output_refs_by_turn: Mapping of batch-local turn index to
                intercepted tool-output refs discovered from the raw payload.

        Returns:
            Number of turns ingested.
        """
        import sys as _sys
        import time as _time

        from ..types import TagResult, TurnTagEntry, get_sender_name

        _ensure_engine_imports()

        _tag_start = _time.time()

        store_tags = [ts.tag for ts in self._store.get_all_tags(conversation_id=self.config.conversation_id)]
        ingested = 0
        _total_turns = len(history_pairs) // 2
        n_context = self.config.tag_generator.context_lookback_pairs
        running_session_date = ""

        for i in range(0, len(history_pairs) - 1, 2):
            user_msg = history_pairs[i]
            asst_msg = history_pairs[i + 1]
            batch_turn = i // 2
            turn_tool_refs = None
            if tool_output_refs_by_turn is not None:
                turn_tool_refs = tool_output_refs_by_turn.get(batch_turn, [])

            sender = get_sender_name(user_msg.metadata) if user_msg.metadata else ""

            # Tool-only turns: skip LLM tagger, assign sequential tool_N tag
            if self._is_tool_turn([user_msg, asst_msg]):
                if self._next_tool_tag is not None:
                    tag_num = self._next_tool_tag()
                else:
                    self._engine_state.tool_tag_counter += 1
                    tag_num = self._engine_state.tool_tag_counter
                tag_name = f"tool_{tag_num}"
                entry = TurnTagEntry(
                    turn_number=turn_offset + (i // 2),
                    message_hash=hashlib.sha256(
                        f"{user_msg.content} {asst_msg.content}".encode()
                    ).hexdigest()[:16],
                    tags=[tag_name],
                    primary_tag=tag_name,
                    sender=sender or "",
                    session_date=running_session_date,
                )
                self._turn_tag_index.append(entry)
                try:
                    self._store.save_turn_message(
                        self.config.conversation_id,
                        entry.turn_number,
                        user_msg.content,
                        asst_msg.content,
                        user_raw_content=json.dumps(user_msg.raw_content) if user_msg.raw_content else None,
                        assistant_raw_content=json.dumps(asst_msg.raw_content) if asst_msg.raw_content else None,
                    )
                except Exception:
                    pass
                self._link_turn_tool_outputs(entry.turn_number, turn_tool_refs)
                ingested += 1
                continue

            # BUG-013: Skip empty turns with no tool blocks
            if not user_msg.content.strip() and not asst_msg.content.strip():
                logger.debug("Skipping empty turn at pair index %d", i // 2)
                continue

            # Track running session date BEFORE stub/tagger — stubs need timestamps too
            m = _SESSION_HEADER_RE.search(user_msg.content)
            if m:
                running_session_date = m.group(1)
            elif user_msg.timestamp:
                running_session_date = user_msg.timestamp.strftime("%Y-%m-%dT%H:%M:%S")

            # Stub turns (media attachments, image placeholders, etc.):
            # skip tagger, assign _stub tag, preserve raw text for passthrough.
            combined_for_stub = f"{user_msg.content} {asst_msg.content}"
            if _is_stub_content_fn(combined_for_stub):
                entry = TurnTagEntry(
                    turn_number=turn_offset + (i // 2),
                    message_hash=hashlib.sha256(combined_for_stub.encode()).hexdigest()[:16],
                    tags=["_stub"],
                    primary_tag="_stub",
                    sender=sender or "",
                    session_date=running_session_date,
                )
                self._turn_tag_index.append(entry)
                try:
                    self._store.save_turn_message(
                        self.config.conversation_id,
                        entry.turn_number,
                        user_msg.content,
                        asst_msg.content,
                    )
                except Exception:
                    pass
                self._link_turn_tool_outputs(entry.turn_number, turn_tool_refs)
                ingested += 1
                logger.info(
                    "TAGGER turn=%d STUB content_len=%d preview=\"%s\"",
                    turn_offset + (i // 2), len(combined_for_stub),
                    combined_for_stub[:60].replace("\n", " "),
                )
                if progress_callback:
                    total = len(history_pairs) // 2
                    progress_callback(ingested, total, entry)
                continue

            combined_text = f"{user_msg.content} {asst_msg.content}"
            _turn_num = turn_offset + (i // 2)  # global turn number
            _content_preview = combined_text[:60].replace("\n", " ")

            # Flag short content that may be dominated by context
            if len(combined_text) < 80:
                logger.info(
                    "TAGGER turn=%d SHORT_CONTENT len=%d \"%s\"",
                    _turn_num, len(combined_text), _content_preview,
                )

            # Build context from preceding pairs in the flat history
            context: list[str] | None = None
            if i >= 2:
                ctx_pairs: list[str] = []
                start = max(0, i - n_context * 2)
                for j in range(start, i, 2):
                    if j + 1 < len(history_pairs):
                        ctx_pairs.append(history_pairs[j].content)
                        ctx_pairs.append(history_pairs[j + 1].content)
                context = ctx_pairs if ctx_pairs else None

            # Context bleed gate (BUG-010): skip stale context on topic shift
            _bleed_gate = "no_context"
            _bleed_sim = -1.0
            if (
                context
                and self.config.tag_generator.context_bleed_threshold > 0
            ):
                _relevant, _bleed_sim = self._semantic.context_is_relevant_with_score(combined_text, context)
                if not _relevant:
                    _bleed_gate = f"BLOCKED (similarity={_bleed_sim:.2f} threshold={self.config.tag_generator.context_bleed_threshold})"
                    context = None
                else:
                    _bleed_gate = f"passed (similarity={_bleed_sim:.2f})"
            elif context:
                _bleed_gate = "disabled"

            _ctx_preview = context[-2][:60].replace("\n", " ") if context and len(context) >= 2 else ""
            logger.info(
                "TAGGER turn=%d content_len=%d content_preview=\"%s\" "
                "context_pairs=%d context_preview=\"%s\" bleed_gate=%s",
                _turn_num, len(combined_text), _content_preview,
                len(context) // 2 if context else 0, _ctx_preview, _bleed_gate,
            )

            tag_result = self._tag_generator.generate_tags(
                combined_text, store_tags, context_turns=context,
            )

            logger.info(
                "TAGGER turn=%d result primary=%s tags=%s source=%s",
                _turn_num, tag_result.primary, sorted(tag_result.tags), tag_result.source,
            )

            # Retry with expanded context on _general
            if tag_result.tags == ["_general"] and i >= 2:
                expanded_start = max(0, i - n_context * 4)
                expanded_ctx: list[str] = []
                for j in range(expanded_start, i, 2):
                    if j + 1 < len(history_pairs):
                        expanded_ctx.append(history_pairs[j].content)
                        expanded_ctx.append(history_pairs[j + 1].content)
                # Gate expanded context too
                _expanded_gate = "no_context"
                if (
                    expanded_ctx
                    and self.config.tag_generator.context_bleed_threshold > 0
                ):
                    _rel, _sim = self._semantic.context_is_relevant_with_score(combined_text, expanded_ctx)
                    if not _rel:
                        _expanded_gate = f"BLOCKED (similarity={_sim:.2f})"
                        expanded_ctx = []
                    else:
                        _expanded_gate = f"passed (similarity={_sim:.2f})"
                if expanded_ctx:
                    logger.info(
                        "TAGGER turn=%d retry=expanded_context expanded_pairs=%d bleed_gate=%s",
                        _turn_num, len(expanded_ctx) // 2, _expanded_gate,
                    )
                    tag_result = self._tag_generator.generate_tags(
                        combined_text, store_tags, context_turns=expanded_ctx,
                    )
                    logger.info(
                        "TAGGER turn=%d retry_result primary=%s tags=%s source=%s",
                        _turn_num, tag_result.primary, sorted(tag_result.tags), tag_result.source,
                    )

            # Final fallback: inherit from most recent meaningful turn
            if tag_result.tags == ["_general"]:
                prev = self._turn_tag_index.latest_meaningful_tags()
                if prev:
                    tag_result = TagResult(
                        tags=list(prev.tags),
                        primary=prev.primary_tag,
                        source="inherited",
                    )
                    logger.info(
                        "TAGGER turn=%d fallback=inherited from_turn=%d tags=%s",
                        _turn_num, prev.turn_number, sorted(prev.tags),
                    )

            entry = TurnTagEntry(
                turn_number=turn_offset + (i // 2),
                message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                tags=tag_result.tags,
                primary_tag=tag_result.primary,
                sender=sender or "",
                session_date=running_session_date,
            )
            self._turn_tag_index.append(entry)
            try:
                self._store.save_turn_message(
                    self.config.conversation_id,
                    entry.turn_number,
                    user_msg.content,
                    asst_msg.content,
                    user_raw_content=json.dumps(user_msg.raw_content) if user_msg.raw_content else None,
                    assistant_raw_content=json.dumps(asst_msg.raw_content) if asst_msg.raw_content else None,
                )
            except Exception:
                pass
            self._link_turn_tool_outputs(entry.turn_number, turn_tool_refs)
            ingested += 1

            # Stderr progress for visibility
            if ingested % 5 == 0 or ingested == _total_turns:
                _elapsed = _time.time() - _tag_start
                _rate = ingested / _elapsed if _elapsed > 0 else 0
                _eta = int((_total_turns - ingested) / _rate) if _rate > 0 else 0
                _sys.stderr.write(
                    f"\r  TAGGING: {ingested}/{_total_turns} turns | "
                    f"{_rate:.1f} turn/s | ETA {_eta}s   "
                )
                _sys.stderr.flush()

            if progress_callback:
                total = len(history_pairs) // 2
                progress_callback(ingested, total, entry)

            # Refresh store tags every 10 turns so new tags influence later tagging
            if ingested % 10 == 0:
                store_tags = [ts.tag for ts in self._store.get_all_tags(conversation_id=self.config.conversation_id)]

            # Periodic state save so session_date + tags are queryable during ingestion
            if ingested % 20 == 0:
                checkpoint_turn = turn_offset + ingested - 1
                self._save_state_callback(
                    history_pairs,
                    last_completed_turn=checkpoint_turn,
                    last_indexed_turn=checkpoint_turn,
                )

        # Final save after all turns ingested
        final_turn = turn_offset + ingested - 1
        self._save_state_callback(
            history_pairs,
            last_completed_turn=final_turn,
            last_indexed_turn=final_turn,
        )
        _sys.stderr.write("\n")
        _sys.stderr.flush()
        logger.info("Ingested %d historical turns into TurnTagIndex", ingested)
        return ingested
