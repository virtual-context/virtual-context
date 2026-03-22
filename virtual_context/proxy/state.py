"""Proxy session state management.

Contains ProxyState, SessionState, and _IngestionCancelled — the core
state machine for non-blocking ingestion and turn-complete processing.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from ..engine import VirtualContextEngine
from ..core.turn_tag_index import TurnTagIndex
from ..types import Message, SplitResult

from .helpers import (
    _strip_envelope,
    _extract_history_pairs,
)
from .metrics import ProxyMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider derivation from upstream URL
# ---------------------------------------------------------------------------

_PROVIDER_HOSTS: dict[str, str] = {
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "generativelanguage.googleapis.com": "gemini",
    "api.groq.com": "groq",
    "openrouter.ai": "openrouter",
    "api.together.xyz": "together",
    "api.mistral.ai": "mistral",
    "api.cohere.com": "cohere",
    "api.deepseek.com": "deepseek",
}


def _derive_provider(upstream: str) -> str:
    if not upstream:
        return ""
    try:
        from urllib.parse import urlparse
        host = urlparse(upstream).hostname or ""
        for pattern, name in _PROVIDER_HOSTS.items():
            if host == pattern or host.endswith("." + pattern):
                return name
        return host.split(".")[0] if host else ""
    except Exception:
        logger.debug("provider name extraction failed", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# SessionState — state machine for non-blocking ingestion
# ---------------------------------------------------------------------------

class SessionState(enum.Enum):
    PASSTHROUGH = "passthrough"  # forwarding without enrichment (ingestion pending/running)
    INGESTING = "ingesting"     # background thread tagging historical turns
    ACTIVE = "active"           # normal enrichment mode


class _IngestionCancelled(Exception):
    """Raised inside progress callback to abort a running ingestion."""
    def __init__(self, done: int, total: int) -> None:
        self.done = done
        self.total = total
        super().__init__(f"Cancelled at {done}/{total}")


# ---------------------------------------------------------------------------
# ProxyState — mirrors HeadlessRunner threading pattern
# ---------------------------------------------------------------------------

class ProxyState:
    """Shared mutable state for the proxy lifetime.

    Thread-safety notes for ``conversation_history``:
        - Wholesale replacement (``conversation_history = list(...)``) only
          occurs during INGESTING state, which is single-threaded.
        - ``fire_turn_complete`` receives a snapshot via ``list()`` copy, so
          the background tagging thread operates on its own list instance.
        - Append-only mutations (``conversation_history.append(...)``) happen
          on the async request path, which is serialized by ``wait_for_tag()``.
        - This combination is safe: ingestion is single-threaded, live
          requests are serialized, and turn-complete always gets its own copy.
    """

    def __init__(
        self,
        engine: VirtualContextEngine,
        metrics: ProxyMetrics | None = None,
        upstream: str = "",
    ) -> None:
        self.engine = engine
        self.conversation_history: list[Message] = []
        self.metrics = metrics
        self.upstream = upstream
        self.provider = _derive_provider(upstream)
        self._pool = ThreadPoolExecutor(max_workers=1)       # tagging (serialized)
        self._compact_pool = ThreadPoolExecutor(max_workers=1)  # compaction (background)
        self._pending_tag: Future | None = None
        self._pending_compact: Future | None = None
        self._last_compact_priority: str = ""  # "soft" or "hard" from last tag_turn
        self._ingested_conversations: set[str] = set()
        self._ingested_first_hash: dict[str, str] = {}  # conversation_id → hash of first message
        self._ingested_turn_count: dict[str, int] = {}   # conversation_id → turn count at ingestion
        self._ingestion_lock = threading.Lock()
        self._compaction_lock = threading.Lock()
        # State machine for non-blocking ingestion
        self._state = SessionState.ACTIVE
        self._latest_body: dict | None = None
        self._ingestion_progress: tuple[int, int] = (0, 0)
        self._manual_passthrough = False
        self._ingestion_thread: threading.Thread | None = None
        self._ingestion_cancel = threading.Event()
        # Initial snapshot: captured at first ingestion start for growth tracking
        self._initial_turns: int | None = None
        self._initial_tag_count: int | None = None
        # Payload size tracking (KB + tokens)
        self._initial_payload_kb: float | None = None
        self._last_payload_kb: float = 0.0
        self._last_enriched_payload_kb: float = 0.0
        self._initial_payload_tokens: int | None = None
        self._last_payload_tokens: int = 0
        self._last_enriched_payload_tokens: int = 0
        self._last_non_virtualizable_floor: int = 0  # outbound - VC context tokens
        # Live request counter: incremented on each user turn processed through proxy
        self._total_requests: int = 0
        # Upstream context window enforcement
        self._instance_upstream_limit: int = 0  # set by create_app from ProxyInstanceConfig
        self._last_model: str = ""  # last model name seen (for dashboard)

        # Set provider on engine for persistence (only if not already restored)
        if self.provider and not engine._engine_state.provider:
            engine._engine_state.provider = self.provider

        # Wire up request captures persistence: engine pulls from metrics at save time
        if metrics:
            engine._request_captures_provider = metrics.get_captured_requests_summary
        # Restore persisted request captures into metrics
        if metrics and engine._restored_request_captures:
            metrics.restore_request_captures(engine._restored_request_captures)
            engine._restored_request_captures = []
        # Restore conversation history from persisted turn messages
        if engine._restored_conversation_history:
            for _turn, _user, _asst in engine._restored_conversation_history:
                self.conversation_history.append(
                    Message(role="user", content=_user)
                )
                self.conversation_history.append(
                    Message(role="assistant", content=_asst)
                )
            logger.info("Restored conversation_history: %d messages from %d turns",
                        len(self.conversation_history),
                        len(engine._restored_conversation_history))
            engine._restored_conversation_history = []

    @property
    def turn_offset(self) -> int:
        """Starting turn number from persisted engine state.

        When the proxy restarts, conversation_history is empty but the
        TurnTagIndex may have been restored from the store.  Use the
        highest indexed turn + 1 as the offset so turn numbering
        continues from the previous session.
        """
        try:
            entries = self.engine._turn_tag_index.entries
            if entries and len(entries) > 0:
                return max(e.turn_number for e in entries) + 1
        except (TypeError, AttributeError):
            pass
        return 0

    @property
    def session_state(self) -> SessionState:
        """Current session state, accounting for manual passthrough override."""
        if self._manual_passthrough:
            return SessionState.PASSTHROUGH
        return self._state

    def set_manual_passthrough(self, enabled: bool) -> None:
        self._manual_passthrough = enabled

    def _transition_to(self, new_state: SessionState) -> None:
        old = self._state
        self._state = new_state
        if self.metrics and old != new_state:
            self.metrics.record({
                "type": "session_state_change",
                "from": old.value,
                "to": new_state.value,
                "conversation_id": self.engine.config.conversation_id,
            })
        logger.info(
            "Conversation %s: %s → %s",
            self.engine.config.conversation_id[:12], old.value, new_state.value,
        )

    def live_snapshot(self) -> dict:
        engine = self.engine
        idx = engine._turn_tag_index

        # KB stats: tag summaries
        tag_summary_count = 0
        tag_summary_tokens = 0
        try:
            summaries = engine._store.get_all_tag_summaries()
            tag_summary_count = len(summaries)
            tag_summary_tokens = sum(ts.summary_tokens for ts in summaries)
        except Exception:
            logger.debug("tag summary stats collection failed", exc_info=True)

        # Estimate history size in tokens (chars / 4)
        history_tokens = 0
        for m in self.conversation_history:
            history_tokens += len(m.content) // 4

        context_window = engine.config.monitor.context_window
        utilization_pct = round(history_tokens / context_window * 100, 1) if context_window > 0 else 0

        # Distinct tag count from TurnTagIndex
        all_tags: set[str] = set()
        for entry in idx.entries:
            all_tags.update(entry.tags)
        all_tags.discard("_general")

        snap = {
            "conversation_id": engine.config.conversation_id,
            "turn_count": len(self.conversation_history) // 2,
            "total_requests": self._total_requests,
            "compacted_through": engine._engine_state.compacted_through,
            "tag_count": len(idx.entries),
            "distinct_tags": len(all_tags),
            "active_tags": list(idx.get_active_tags(lookback=6)),
            "session_state": self.session_state.value,
            "ingestion_progress": list(self._ingestion_progress),
            "manual_passthrough": self._manual_passthrough,
            "context_window": context_window,
            "history_tokens": history_tokens,
            "utilization_pct": utilization_pct,
            "tag_summary_count": tag_summary_count,
            "tag_summary_tokens": tag_summary_tokens,
            "initial_turns": self._initial_turns,
            "initial_tag_count": self._initial_tag_count,
            "initial_payload_kb": self._initial_payload_kb,
            "last_payload_kb": self._last_payload_kb,
            "last_enriched_payload_kb": self._last_enriched_payload_kb,
            "initial_payload_tokens": self._initial_payload_tokens,
            "last_payload_tokens": self._last_payload_tokens,
            "last_enriched_payload_tokens": self._last_enriched_payload_tokens,
            "non_virtualizable_floor": self._last_non_virtualizable_floor,
        }
        return snap

    def wait_for_tag(self) -> None:
        """Block until tagging finishes. Compaction may still be running."""
        if self._pending_tag is not None:
            self._pending_tag.result()
            self._pending_tag = None

    def wait_for_complete(self) -> None:
        """Block until tag + compact both finish."""
        self.wait_for_tag()
        if self._pending_compact is not None:
            self._pending_compact.result()
            self._pending_compact = None

    def fire_turn_complete(
        self,
        history_snapshot: list[Message],
        payload_tokens: int | None = None,
        turn_id: str = "",
    ) -> None:
        """Submit tagging to background thread.

        Compaction (if needed) fires automatically in a separate pool
        once tagging completes — the next request only waits for tagging.

        Skipped entirely during ingestion — the live tagger would write to
        turn numbers that conflict with historical turns being ingested.
        Ingestion handles all tagging; post-ingestion compaction handles the rest.
        """
        if self._state == SessionState.INGESTING:
            logger.info("fire_turn_complete skipped (ingestion in progress)")
            return
        self._pending_tag = self._pool.submit(
            self._run_tag_turn, history_snapshot, payload_tokens, turn_id,
        )

    def _run_tag_turn(
        self,
        history: list[Message],
        payload_tokens: int | None = None,
        turn_id: str = "",
    ) -> None:
        """Fast path: tag the turn, emit metrics, fire compaction if needed."""
        t0 = time.monotonic()
        turn = len(self.engine._turn_tag_index.entries)
        conversation_id = self.engine.config.conversation_id
        try:
            signal = self.engine.tag_turn(
                history, payload_tokens=payload_tokens,
            )
            self._last_compact_priority = signal.priority if signal else ""

            tag_ms = round((time.monotonic() - t0) * 1000, 1)
            _tti = self.engine._turn_tag_index.entries
            entry = _tti[-1] if len(_tti) > turn else None
            _tags = entry.tags if entry else []
            _primary = entry.primary_tag if entry else ""
            _needs_compact = signal is not None
            logger.info(
                "T%d TAG %dms tags=[%s] primary=%s%s",
                turn, int(tag_ms), ", ".join(_tags), _primary,
                " -> COMPACT queued" if _needs_compact else "",
            )
            logger.info(
                "T%d tagged (%dms) conversation=%s compacted_through=%d history=%d%s",
                turn, int(tag_ms), conversation_id[:12],
                self.engine._engine_state.compacted_through,
                len(history),
                " compact_queued" if _needs_compact else "",
            )

            # Emit turn_complete event (tag phase)
            if self.metrics:
                # Reuse entry from above (already the latest appended).
                active_tags = list(
                    self.engine._turn_tag_index.get_active_tags(lookback=6)
                )
                turn_pair_tokens = (
                    sum(len(m.content) for m in history[-2:]) // 4
                    if len(history) >= 2 else 0
                )
                # Write response tags to captured request
                response_tags = entry.tags if entry else []
                self.metrics.update_request_tags(
                    turn, response_tags=response_tags,
                )
                self.metrics.record({
                    "type": "turn_complete",
                    "turn": turn,
                    "turn_id": turn_id,
                    "tags": entry.tags if entry else [],
                    "primary_tag": entry.primary_tag if entry else "",
                    "complete_ms": tag_ms,
                    "tag_ms": tag_ms,
                    "active_tags": active_tags,
                    "store_tag_count": len(self.engine._store.get_all_tags()),
                    "turn_pair_tokens": turn_pair_tokens,
                    "conversation_id": conversation_id,
                })

                # Emit tag split event if splitting occurred
                split_result = self.engine._engine_state.last_split_result
                if isinstance(split_result, SplitResult):
                    if split_result.splittable:
                        new_tags = list(split_result.groups.keys())
                        logger.info(
                            "T%d SPLIT \"%s\" -> %s (%d turns)",
                            turn, split_result.tag, new_tags,
                            sum(len(v) for v in split_result.groups.values()),
                        )
                    else:
                        logger.info(
                            "T%d SUMMARIZED \"%s\" (unsplittable: %s)",
                            turn, split_result.tag, split_result.reason,
                        )
                    self.metrics.record({
                        "type": "tag_split",
                        "turn": turn,
                        "tag": split_result.tag,
                        "splittable": split_result.splittable,
                        "new_tags": list(split_result.groups.keys()) if split_result.splittable else [],
                        "conversation_id": conversation_id,
                    })
                    self.engine._engine_state.last_split_result = None  # consume

            # Fire compaction in background if needed — but NOT during ingestion.
            # During ingestion, only a fraction of turns are tagged. Compacting now
            # would process hundreds of untagged turns via expensive LLM fallback.
            # Post-ingestion compaction handles this once ingestion completes.
            if signal is not None:
                if self._state == SessionState.INGESTING:
                    logger.info("T%d compaction deferred (ingestion in progress)", turn)
                else:
                    self._pending_compact = self._compact_pool.submit(
                        self._run_compact, history, signal, turn,
                    )

        except Exception as e:
            logger.error("tag_turn error: %s", e, exc_info=True)

    def _run_compact(
        self,
        history: list[Message],
        signal: object,
        turn: int,
    ) -> None:
        """Background compaction — runs in _compact_pool, doesn't block next request."""
        conversation_id = self.engine.config.conversation_id

        def _compact_progress(done, total, result, *, phase="", **kwargs):
            if self.metrics:
                evt = {
                    "type": "compaction_progress",
                    "turn": turn,
                    "done": done,
                    "total": total,
                    "phase": phase,
                    "conversation_id": conversation_id,
                }
                if result is not None:
                    evt["primary_tag"] = result.primary_tag
                    evt["tags"] = result.tags
                    evt["original_tokens"] = result.original_tokens
                    evt["summary_tokens"] = result.summary_tokens
                for k, v in kwargs.items():
                    evt[k] = v
                self.metrics.record(evt)

        with self._compaction_lock:
            t0 = time.monotonic()
            try:
                report = self.engine.compact_if_needed(
                    history, signal, progress_callback=_compact_progress,
                )
                compact_ms = round((time.monotonic() - t0) * 1000, 1)

                if report is not None:
                    logger.info(
                        "T%d COMPACT %dms freed=%dt segments=%d",
                        turn, int(compact_ms), report.tokens_freed,
                        report.segments_compacted,
                    )
                    logger.info(
                        "T%d compaction (%dms): %d segments, freed %d tokens, tags=%s, "
                        "summaries_built=%d",
                        turn, int(compact_ms),
                        report.segments_compacted,
                        report.tokens_freed,
                        report.tags,
                        report.tag_summaries_built,
                    )

                    # Emit compaction event
                    if self.metrics:
                        original_tokens = sum(
                            r.original_tokens for r in report.results
                        )
                        summary_tokens = sum(
                            r.summary_tokens for r in report.results
                        )
                        self.metrics.record({
                            "type": "compaction",
                            "turn": turn,
                            "compact_ms": compact_ms,
                            "segments": report.segments_compacted,
                            "tokens_freed": report.tokens_freed,
                            "original_tokens": original_tokens,
                            "summary_tokens": summary_tokens,
                            "tags": report.tags,
                            "tag_summaries_built": report.tag_summaries_built,
                            "compacted_through": self.engine._engine_state.compacted_through,
                            "conversation_id": conversation_id,
                        })
                else:
                    logger.info("T%d compaction skipped (no messages to compact)", turn)

            except Exception as e:
                logger.error("compact_if_needed error: %s", e, exc_info=True)

    def _compact_after_ingestion(self, history: list[Message]) -> None:
        """Compact immediately after ingestion — no threshold check needed.

        After ingesting 300 turns, they all need segmenting/summarizing regardless
        of token count. The monitor threshold is for live requests where we decide
        IF compaction should run. Post-ingestion, we know it should.
        """
        try:
            from ..types import CompactionSignal
            # Use conversation_history (full proxy history) not just ingestion pairs
            compact_history = self.conversation_history if self.conversation_history else history
            protected = self.engine.config.monitor.protected_recent_turns * 2
            watermark = self.engine._engine_state.compacted_through
            compactable = len(compact_history) - watermark - protected
            if compactable <= 0:
                logger.info("POST-INGEST: no compactable messages (history=%d, watermark=%d, protected=%d)",
                            len(compact_history), watermark, protected)
                return
            turn = len(self.engine._turn_tag_index.entries)
            # Force compaction signal — bypass threshold check
            signal = CompactionSignal(
                priority="soft",
                current_tokens=compactable * 100,  # rough estimate, doesn't matter — compaction runs regardless
                budget_tokens=self.engine.config.monitor.context_window,
                overflow_tokens=compactable * 50,
            )
            logger.info(
                "POST-INGEST Compacting %d messages immediately (history=%d, watermark=%d, protected=%d)",
                compactable, len(compact_history), watermark, protected,
            )
            self._run_compact(compact_history, signal, turn)
        except Exception as e:
            logger.error("Post-ingestion compaction error: %s", e, exc_info=True)

    def _history_ingested(self) -> bool:
        return self.engine.config.conversation_id in self._ingested_conversations

    def _check_history_widening(self, history_pairs: list[Message], conversation_id: str) -> bool:
        """Detect if history prefix shifted (widening) and trigger full re-ingest.

        Returns True if widening was detected and state was cleared for re-ingestion.
        """
        import hashlib
        if not history_pairs or conversation_id not in self._ingested_first_hash:
            return False

        new_first_hash = hashlib.sha256(history_pairs[0].content.encode()).hexdigest()[:16]
        old_first_hash = self._ingested_first_hash.get(conversation_id, "")
        if new_first_hash == old_first_hash:
            return False

        new_turns = len(history_pairs) // 2
        old_turns = self._ingested_turn_count.get(conversation_id, 0)
        threshold = getattr(self.engine.config.proxy, "history_widening_threshold", 0.10)

        if new_turns <= old_turns * (1 + threshold):
            return False  # Not enough growth — likely aging, not widening

        logger.info(
            "HISTORY_WIDENED conversation=%s old_hash=%s new_hash=%s old_turns=%d new_turns=%d "
            "threshold=%.0f%% — clearing state and re-ingesting",
            conversation_id[:12], old_first_hash, new_first_hash,
            old_turns, new_turns, threshold * 100,
        )
        logger.info(
            "INGEST History widened: %d->%d turns, prefix changed -- full re-ingest (conversation=%s)",
            old_turns, new_turns, conversation_id[:12],
        )

        # Clear all conversation state
        try:
            self.engine._store.delete_conversation(conversation_id)
        except Exception as e:
            logger.warning("Failed to delete conversation during widening reset: %s", e)
        self.engine._turn_tag_index = TurnTagIndex()
        self.engine._engine_state.compacted_through = 0
        # Re-sync delegates that cached stale references to turn_tag_index
        for attr in ('_tagging', '_compaction', '_retrieval', '_search'):
            delegate = getattr(self.engine, attr, None)
            if delegate is not None:
                if hasattr(delegate, '_turn_tag_index'):
                    delegate._turn_tag_index = self.engine._turn_tag_index
                if hasattr(delegate, '_engine_state'):
                    delegate._engine_state = self.engine._engine_state
        self._ingested_conversations.discard(conversation_id)
        self._ingested_first_hash.pop(conversation_id, None)
        self._ingested_turn_count.pop(conversation_id, None)

        return True

    def _advance_compaction_watermark(self) -> None:
        """Advance compacted_through to cover all already-processed messages.

        Uses conversation_history length only — the TurnTagIndex may include
        restored entries from previous sessions whose messages are not in the
        current history array, and using its size would set the watermark beyond
        the actual message count, causing compaction drift.

        On a fresh volume (no stored segments), don't advance — everything needs
        first-time compaction.
        """
        try:
            # Don't advance if there are no stored segments — everything needs first-time compaction
            existing_tags = self.engine._store.get_all_tags(
                conversation_id=self.engine.config.conversation_id,
            )
            if not existing_tags:
                logger.info(
                    "Compaction watermark: no stored segments, keeping at %d for first-time compaction",
                    self.engine._engine_state.compacted_through,
                )
                return

            new_wm = len(self.conversation_history)
            old_wm = int(self.engine._engine_state.compacted_through)
            if new_wm > old_wm:
                self.engine._engine_state.compacted_through = new_wm
                logger.info("Compaction watermark advanced: %d → %d (post-ingestion)", old_wm, new_wm)
        except (TypeError, ValueError, AttributeError):
            pass

    def _record_ingestion_watermark(self, history_pairs: list[Message], conversation_id: str) -> None:
        import hashlib
        if history_pairs:
            self._ingested_first_hash[conversation_id] = (
                hashlib.sha256(history_pairs[0].content.encode()).hexdigest()[:16]
            )
        self._ingested_turn_count[conversation_id] = len(history_pairs) // 2

    def ingest_if_needed(self, history_pairs: list[Message]) -> None:
        """Bootstrap TurnTagIndex from pre-existing history (once per session).

        Double-checked locking: fast path skips the lock entirely.
        """
        conversation_id = self.engine.config.conversation_id
        if conversation_id in self._ingested_conversations:
            # Check for history widening even after ingestion is "done"
            self._check_history_widening(history_pairs, conversation_id)
            if conversation_id in self._ingested_conversations:
                return  # No widening detected
        with self._ingestion_lock:
            if conversation_id in self._ingested_conversations:
                return
            t0 = time.monotonic()
            turns = self.engine.ingest_history(history_pairs)
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            self._ingested_conversations.add(conversation_id)
            self._advance_compaction_watermark()
            self._record_ingestion_watermark(history_pairs, conversation_id)

            logger.info(
                "INGEST %d turns in %dms (conversation=%s)",
                turns, int(elapsed_ms), conversation_id[:12],
            )

            if self.metrics:
                # Emit per-turn events so the dashboard grid shows history
                baseline_history_tokens = 0
                for i in range(0, len(history_pairs) - 1, 2):
                    turn_num = i // 2
                    entry = self.engine._turn_tag_index.get_tags_for_turn(
                        turn_num,
                    )
                    raw_content = history_pairs[i].content
                    preview = _strip_envelope(raw_content)[:60]
                    # Estimate turn pair tokens for baseline calculation
                    pair_chars = len(history_pairs[i].content) + len(history_pairs[i + 1].content)
                    tpt = pair_chars // 4
                    baseline_history_tokens += tpt
                    self.metrics.record({
                        "type": "ingested_turn",
                        "turn": turn_num,
                        "tags": entry.tags if entry else [],
                        "primary_tag": entry.primary_tag if entry else "",
                        "message_preview": preview,
                        "turn_pair_tokens": tpt,
                        "conversation_id": conversation_id,
                    })
                self.metrics.record({
                    "type": "history_ingestion",
                    "turns_ingested": turns,
                    "pairs_received": len(history_pairs) // 2,
                    "elapsed_ms": elapsed_ms,
                    "conversation_id": conversation_id,
                    "baseline_history_tokens": baseline_history_tokens,
                })

    # ------------------------------------------------------------------
    # Non-blocking ingestion (background thread)
    # ------------------------------------------------------------------

    def start_ingestion_if_needed(self, history_pairs: list[Message]) -> None:
        """Start non-blocking history ingestion in a background thread.

        Returns immediately — the session stays in INGESTING while the
        background thread tags historical turns.  If called while ingestion
        is already running, cancels the old thread and resumes from the
        last tagged turn (PROXY-013).
        """
        conversation_id = self.engine.config.conversation_id
        if conversation_id in self._ingested_conversations:
            # Check for history widening even after ingestion is "done"
            self._check_history_widening(history_pairs, conversation_id)
            if conversation_id in self._ingested_conversations:
                return
        with self._ingestion_lock:
            if conversation_id in self._ingested_conversations:
                return

            logger.info(
                "INGEST_ENTRY conversation=%s pairs=%d index_size=%d thread_alive=%s",
                conversation_id[:12], len(history_pairs) // 2,
                len(self.engine._turn_tag_index.entries),
                self._ingestion_thread is not None and self._ingestion_thread.is_alive(),
            )

            if not history_pairs:
                self._ingested_conversations.add(conversation_id)
                return

            # Skip if persisted TurnTagIndex already covers history
            existing_turns = len(self.engine._turn_tag_index.entries)
            needed_turns = len(history_pairs) // 2
            if existing_turns >= needed_turns:
                self._ingested_conversations.add(conversation_id)
                self._advance_compaction_watermark()
                logger.info(
                    "Skipping ingestion: persisted index (%d) covers history (%d)",
                    existing_turns, needed_turns,
                )
                return

            # Slice past already-ingested turns to avoid re-tagging and index collision.
            # Only when NO ingestion is running (persisted state resume case).
            # If ingestion IS running, the cancel-and-resume path handles slicing.
            _thread_running = (
                self._ingestion_thread is not None
                and self._ingestion_thread.is_alive()
            )
            if existing_turns > 0 and not _thread_running:
                logger.info(
                    "Slicing history past %d existing turns (needed=%d)",
                    existing_turns, needed_turns,
                )
                history_pairs = list(history_pairs[existing_turns * 2:])
                if not history_pairs:
                    self._ingested_conversations.add(conversation_id)
                    self._advance_compaction_watermark()
                    return

            # ---- PROXY-013: cancel-and-resume if already running ----
            if (
                self._ingestion_thread is not None
                and self._ingestion_thread.is_alive()
            ):
                done, total = self._ingestion_progress
                logger.info(
                    "Cancelling running ingestion at turn %d/%d "
                    "(new request has %d pairs)",
                    done, total, needed_turns,
                )
                self._ingestion_cancel.set()
                self._ingestion_thread.join(timeout=5.0)
                if self._ingestion_thread.is_alive():
                    logger.warning("Old ingestion thread did not exit in 5s")
                # Reset cancel event for the new thread
                self._ingestion_cancel.clear()

                # Re-read existing_turns AFTER old thread stopped —
                # the thread may have appended one more entry between
                # the last callback and the cancel taking effect.
                existing_turns = len(self.engine._turn_tag_index.entries)

                logger.info(
                    "INGEST Cancel at T%d/%d, resuming from T%d (conversation=%s)",
                    done, total, existing_turns, conversation_id[:12],
                )

                # Verify hash at handoff point
                self._verify_handoff_hash(history_pairs, existing_turns)

                # Slice to remaining pairs only
                history_pairs = list(history_pairs[existing_turns * 2:])
                if not history_pairs:
                    self._ingested_conversations.add(conversation_id)
                    self._advance_compaction_watermark()
                    self._transition_to(SessionState.ACTIVE)
                    return
                needed_turns = len(history_pairs) // 2 + existing_turns

            total = needed_turns
            self._ingestion_progress = (existing_turns, total)

            # Capture initial snapshot once (first ingestion start only)
            if self._initial_turns is None:
                self._initial_turns = existing_turns
                self._initial_tag_count = len(self.engine._turn_tag_index.entries)

            self._transition_to(SessionState.INGESTING)

            # Separate daemon thread (not the _pool) so on_turn_complete
            # can use the pool once we transition to ACTIVE.
            self._ingestion_thread = threading.Thread(
                target=self._run_ingestion_with_catchup,
                args=(list(history_pairs), existing_turns, total),
                daemon=True,
                name="vc-ingest",
            )
            self._ingestion_thread.start()

    def _verify_handoff_hash(
        self, new_pairs: list[Message], handoff_turn: int,
    ) -> None:
        """Verify the last tagged turn matches the same content in new history.

        Logs a warning if the hash doesn't match — indicates potential data
        loss or history divergence between requests.
        """
        import hashlib as _hl
        if handoff_turn <= 0:
            return
        prev_turn = handoff_turn - 1
        entry = self.engine._turn_tag_index.get_tags_for_turn(prev_turn)
        if entry is None:
            return
        pair_idx = prev_turn * 2
        if pair_idx + 1 >= len(new_pairs):
            logger.warning(
                "Handoff verification: turn %d not in new history "
                "(new history has %d pairs) — potential data loss",
                prev_turn, len(new_pairs) // 2,
            )
            return
        combined = f"{new_pairs[pair_idx].content} {new_pairs[pair_idx + 1].content}"
        new_hash = _hl.sha256(combined.encode()).hexdigest()[:16]
        if new_hash != entry.message_hash:
            logger.warning(
                "Handoff hash MISMATCH at turn %d: "
                "indexed=%s new=%s — history may have diverged",
                prev_turn, entry.message_hash, new_hash,
            )
            logger.info(
                "INGEST WARNING: hash mismatch at T%d (indexed=%s vs new=%s)",
                prev_turn, entry.message_hash, new_hash,
            )
        else:
            logger.info(
                "Handoff hash verified at turn %d: %s",
                prev_turn, new_hash,
            )

    def _run_ingestion_with_catchup(
        self, initial_pairs: list[Message], baseline: int = 0, cumulative_total: int = 0,
    ) -> None:
        """Background thread: ingest initial pairs, then catch up any gap."""
        conversation_id = self.engine.config.conversation_id
        cancelled = False
        try:
            # Tag all initial history
            self._ingest_pairs_with_progress(
                initial_pairs, baseline=baseline, cumulative_total=cumulative_total or None,
            )

            # Catch-up loop — tag any turns that arrived during ingestion
            for _ in range(10):  # bounded to avoid infinite loops
                if self._ingestion_cancel.is_set():
                    break
                latest = self._latest_body
                if latest is None:
                    break
                latest_pairs = _extract_history_pairs(latest)
                needed = len(latest_pairs) // 2
                have = len(self.engine._turn_tag_index.entries)
                if needed <= have:
                    break
                # Tag the gap
                gap_pairs = latest_pairs[have * 2:]
                if not gap_pairs:
                    break
                logger.info(
                    "Ingestion catch-up: %d gap turns (have=%d, need=%d)",
                    len(gap_pairs) // 2, have, needed,
                )
                self._ingestion_progress = (have, needed)
                self._ingest_pairs_with_progress(gap_pairs, baseline=have, cumulative_total=needed)

        except _IngestionCancelled as e:
            # New request is taking over — exit cleanly without
            # transitioning to ACTIVE or marking as ingested.
            # NOTE: Python's finally block runs even after return,
            # so we use a flag to skip the finalization actions.
            cancelled = True
            logger.info("Ingestion cancelled at %d/%d", e.done, e.total)
        except Exception as e:
            logger.error("Ingestion error: %s", e, exc_info=True)
        finally:
            if not cancelled:
                self._ingested_conversations.add(conversation_id)
                self._advance_compaction_watermark()
                # Record watermark for history widening detection
                latest = self._latest_body
                if latest:
                    _pairs = _extract_history_pairs(latest)
                    self._record_ingestion_watermark(_pairs, conversation_id)
                # After ingestion, check if compaction is needed immediately.
                # Without this, the payload stays over-budget and upstream
                # rejects every request — a deadlock.
                self._compact_after_ingestion(initial_pairs)
                self._transition_to(SessionState.ACTIVE)

    def _ingest_pairs_with_progress(
        self, pairs: list[Message], baseline: int = 0, cumulative_total: int | None = None,
    ) -> None:
        """Call engine.ingest_history with a progress callback that emits events.

        Args:
            pairs: Message pairs to ingest.
            baseline: Already-ingested turns before this batch (for cumulative progress).
            cumulative_total: Total turns across all batches. Defaults to baseline + batch size.

        Raises ``_IngestionCancelled`` if ``_ingestion_cancel`` is set.
        """
        conversation_id = self.engine.config.conversation_id
        t0 = time.monotonic()
        baseline_history_tokens = 0
        _total = cumulative_total if cumulative_total is not None else baseline + len(pairs) // 2
        logger.info(
            "INGEST_BATCH baseline=%d cumulative_total=%s pairs=%d index_size=%d conversation=%s",
            baseline, _total, len(pairs) // 2,
            len(self.engine._turn_tag_index.entries),
            conversation_id[:12],
        )

        def on_progress(done: int, total: int, entry) -> None:
            nonlocal baseline_history_tokens
            cum_done = baseline + done
            # Check cancellation before updating progress
            if self._ingestion_cancel.is_set():
                raise _IngestionCancelled(cum_done, _total)
            self._ingestion_progress = (cum_done, _total)
            if self.metrics:
                # Find the pair for this turn to compute preview + tokens
                turn_num = entry.turn_number
                pair_idx = turn_num * 2
                preview = ""
                tpt = 0
                if pair_idx < len(pairs):
                    raw_content = pairs[pair_idx].content
                    preview = _strip_envelope(raw_content)[:60]
                    if pair_idx + 1 < len(pairs):
                        pair_chars = len(pairs[pair_idx].content) + len(pairs[pair_idx + 1].content)
                        tpt = pair_chars // 4
                        baseline_history_tokens += tpt
                self.metrics.record({
                    "type": "ingested_turn",
                    "turn": turn_num,
                    "tags": entry.tags if entry else [],
                    "primary_tag": entry.primary_tag if entry else "",
                    "message_preview": preview,
                    "turn_pair_tokens": tpt,
                    "conversation_id": conversation_id,
                    "done": cum_done,
                    "total": _total,
                })

        turns = self.engine.ingest_history(pairs, progress_callback=on_progress, turn_offset=baseline)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        logger.info(
            "INGEST %d turns in %dms (conversation=%s)",
            turns, int(elapsed_ms), conversation_id[:12],
        )

        if self.metrics:
            self.metrics.record({
                "type": "history_ingestion",
                "turns_ingested": turns,
                "pairs_received": len(pairs) // 2,
                "elapsed_ms": elapsed_ms,
                "conversation_id": conversation_id,
                "baseline_history_tokens": baseline_history_tokens,
            })

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        self._compact_pool.shutdown(wait=True)
