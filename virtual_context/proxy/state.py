"""Proxy session state management.

Contains ProxyState, SessionState, and _IngestionCancelled — the core
state machine for non-blocking ingestion and turn-complete processing.
"""

from __future__ import annotations

import enum
import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

from ..core.conversation_store import StaleConversationWriteError
from ..engine import VirtualContextEngine
from ..core.turn_tag_index import TurnTagIndex
from ..types import EngineState, Message, SplitResult

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
        self._pending_split: Future | None = None
        self._last_compact_priority: str = ""  # "soft" or "hard" from last tag_turn
        self._ingested_conversations: set[str] = set()
        self._ingested_first_hash: dict[str, str] = {}  # conversation_id → hash of first message
        self._ingested_turn_count: dict[str, int] = {}   # conversation_id → turn count at ingestion
        self._ingestion_lock = threading.Lock()
        self._compaction_lock = threading.Lock()
        self._compaction_cancelled = threading.Event()
        self._compaction_state_lock = threading.Lock()
        self._compaction_state: dict[str, object] = {}
        # State machine for non-blocking ingestion
        self._state = SessionState.ACTIVE
        self._latest_body: dict | None = None
        self._ingestion_progress: tuple[int, int] = (0, 0)
        self._manual_passthrough = False
        self._ingestion_thread: threading.Thread | None = None
        self._ingestion_cancel = threading.Event()
        self._deletion_requested = threading.Event()
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
        self._inbound_payload_token_cache = None
        self._chain_snapshot_cache: dict[str, object] = {
            "loaded": False,
            "refs_by_turn": {},
        }
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
            conv_id = engine.config.conversation_id
            engine._request_captures_provider = (
                lambda conv_id=conv_id: metrics.get_captured_requests_summary(
                    conversation_id=conv_id,
                )
            )
        # Restore persisted request captures into metrics
        if metrics and engine._restored_request_captures:
            metrics.restore_request_captures(engine._restored_request_captures)
            engine._restored_request_captures = []
        # Restore conversation history from persisted turn messages
        if engine._restored_conversation_history:
            for item in engine._restored_conversation_history:
                if isinstance(item, dict):
                    # Redis restore — full Message dicts with metadata, timestamps, raw_content
                    from datetime import datetime, timezone
                    ts = item.get("timestamp")
                    if isinstance(ts, str) and ts:
                        try:
                            ts = datetime.fromisoformat(ts)
                        except (ValueError, TypeError):
                            ts = None
                    else:
                        ts = None
                    self.conversation_history.append(Message(
                        role=item.get("role", "user"),
                        content=item.get("content", ""),
                        timestamp=ts,
                        metadata=item.get("metadata"),
                        raw_content=item.get("raw_content"),
                    ))
                else:
                    # Store restore — (turn, user, assistant) tuples
                    _turn, _user, _asst = item
                    self.conversation_history.append(Message(role="user", content=_user))
                    self.conversation_history.append(Message(role="assistant", content=_asst))
            _count = len(self.conversation_history)
            _source = "Redis" if engine._restored_conversation_history and isinstance(engine._restored_conversation_history[0], dict) else "store"
            logger.info("Restored conversation_history: %d messages from %s", _count, _source)
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
            return self._indexed_turn_count()
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

    def mark_conversation_deleted(self) -> None:
        self._deletion_requested.set()

    def is_conversation_deleted(self) -> bool:
        return self._deletion_requested.is_set()

    def _indexed_turn_count(self) -> int:
        raw = getattr(getattr(self.engine, "_engine_state", None), "last_indexed_turn", -1)
        marker = raw if isinstance(raw, int) else -1
        try:
            entries_len = len(self.engine._turn_tag_index.entries)
        except (TypeError, AttributeError):
            entries_len = 0
        return max(entries_len, marker + 1)

    def _completed_turn_count(self) -> int:
        raw = getattr(getattr(self.engine, "_engine_state", None), "last_completed_turn", -1)
        marker = raw if isinstance(raw, int) else -1
        history_turns = len(self.conversation_history) // 2 if self.conversation_history else 0
        return max(history_turns, marker + 1)

    def has_pending_indexing(self) -> bool:
        return self._completed_turn_count() > self._indexed_turn_count()

    def persist_completed_turn(self) -> None:
        if self.is_conversation_deleted():
            return
        if len(self.conversation_history) < 2 or len(self.conversation_history) % 2 != 0:
            return
        persist = getattr(self.engine, "persist_completed_turn", None)
        if callable(persist):
            persist(list(self.conversation_history))
            return
        turn_number = (len(self.conversation_history) // 2) - 1
        try:
            self.engine._store.save_turn_message(
                self.engine.config.conversation_id,
                turn_number,
                self.conversation_history[-2].content,
                self.conversation_history[-1].content,
                user_raw_content=json.dumps(self.conversation_history[-2].raw_content)
                if self.conversation_history[-2].raw_content else None,
                assistant_raw_content=json.dumps(self.conversation_history[-1].raw_content)
                if self.conversation_history[-1].raw_content else None,
            )
            self.engine._engine_state.last_completed_turn = max(
                getattr(self.engine._engine_state, "last_completed_turn", -1),
                turn_number,
            )
        except Exception:
            logger.warning("Failed to persist completed turn", exc_info=True)

    def resume_pending_ingestion_if_needed(self) -> bool:
        """Resume indexing from durable turn_messages when completed turns outpace indexed turns."""
        conversation_id = self.engine.config.conversation_id
        if not self.has_pending_indexing():
            return False
        with self._ingestion_lock:
            if self._ingestion_thread is not None and self._ingestion_thread.is_alive():
                return True

            baseline = self._indexed_turn_count()
            total = self._completed_turn_count()
            pending_rows = list(getattr(self.engine, "_restored_pending_turns", []) or [])
            if not pending_rows:
                turn_numbers = list(range(baseline, total))
                try:
                    rows = self.engine._store.get_turn_messages(conversation_id, turn_numbers)
                except Exception:
                    logger.warning(
                        "Failed to load pending turn_messages for durable resume (conv=%s)",
                        conversation_id[:12],
                        exc_info=True,
                    )
                    return False
                for turn_number in turn_numbers:
                    row = rows.get(turn_number)
                    if row is None:
                        continue
                    user_content, assistant_content, user_raw, assistant_raw = row
                    pending_rows.append(
                        (turn_number, user_content, assistant_content, user_raw, assistant_raw)
                    )

            if not pending_rows:
                return False

            pairs: list[Message] = []
            expected_turn = baseline
            for row in sorted(pending_rows, key=lambda item: item[0]):
                turn_number, user_content, assistant_content, *_rest = row
                if turn_number < baseline:
                    continue
                if turn_number != expected_turn:
                    logger.warning(
                        "Durable resume gap at turn %d for conversation %s; pending resume will stop at turn %d",
                        expected_turn,
                        conversation_id[:12],
                        turn_number - 1,
                    )
                    break
                pairs.append(Message(role="user", content=user_content))
                pairs.append(Message(role="assistant", content=assistant_content))
                expected_turn += 1

            if not pairs:
                return False

            if self.metrics:
                self.metrics.clear_ingestion_events(conversation_id)

            self.engine._restored_pending_turns = []
            self._ingestion_progress = (baseline, max(total, baseline + (len(pairs) // 2)))
            self._transition_to(SessionState.INGESTING)
            self._ingestion_thread = threading.Thread(
                target=self._run_ingestion_with_catchup,
                args=(pairs, baseline, total),
                daemon=True,
                name="vc-ingest-resume",
            )
            self._ingestion_thread.start()
            logger.info(
                "Resuming durable ingestion for conversation %s from turn %d to %d",
                conversation_id[:12],
                baseline,
                total - 1,
            )
            return True

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

    def _update_compaction_state(
        self,
        *,
        operation_id: str,
        status: str,
        phase: str | None = None,
        phase_name: str | None = None,
        done: int | None = None,
        total: int | None = None,
        overall_percent: int | float | None = None,
        elapsed_ms: float | None = None,
        eta_ms: int | None = None,
        heartbeat: bool = False,
        primary_tag: str = "",
        tag: str = "",
        error: str = "",
        **extra,
    ) -> None:
        now_epoch = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._compaction_state_lock:
            prev = (
                self._compaction_state
                if self._compaction_state.get("operation_id") == operation_id
                else {}
            )
            started_at = prev.get("started_at", now_iso)
            started_epoch = prev.get("_started_at_epoch", now_epoch)
            phase_value = phase if phase is not None else str(prev.get("phase", ""))
            phase_name_value = phase_name if phase_name is not None else str(prev.get("phase_name", phase_value))
            done_value = done if done is not None else int(prev.get("done", 0) or 0)
            total_value = total if total is not None else int(prev.get("total", 0) or 0)
            percent_value = overall_percent if overall_percent is not None else prev.get("overall_percent", 0)
            if elapsed_ms is None:
                elapsed_ms = round((now_epoch - float(started_epoch)) * 1000, 1)

            state = {
                "conversation_id": self.engine.config.conversation_id,
                "operation_id": operation_id,
                "status": status,
                "phase": phase_value,
                "phase_name": phase_name_value,
                "done": done_value,
                "total": total_value,
                "overall_percent": percent_value,
                "started_at": started_at,
                "updated_at": now_iso,
                "elapsed_ms": elapsed_ms,
                "eta_ms": eta_ms,
                "heartbeat": bool(heartbeat),
                "primary_tag": primary_tag or str(prev.get("primary_tag", "")),
                "tag": tag or str(prev.get("tag", "")),
                "error": error,
            }
            for key, value in extra.items():
                state[key] = value
            state["_started_at_epoch"] = started_epoch
            state["_updated_at_epoch"] = now_epoch
            self._compaction_state = state

    def compaction_snapshot(self) -> dict | None:
        with self._compaction_state_lock:
            if not self._compaction_state:
                return None
            snap = {
                key: value
                for key, value in self._compaction_state.items()
                if not key.startswith("_")
            }
            updated_epoch = self._compaction_state.get("_updated_at_epoch")
            if isinstance(updated_epoch, (int, float)):
                snap["heartbeat_age_ms"] = int(max(0.0, time.time() - updated_epoch) * 1000)
            return snap

    def live_snapshot(self) -> dict:
        engine = self.engine
        idx = engine._turn_tag_index

        # KB stats: tag summaries
        tag_summary_count = 0
        tag_summary_tokens = 0
        try:
            summaries = engine._store.get_all_tag_summaries(
                conversation_id=engine.config.conversation_id,
            )
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
        compaction = self.compaction_snapshot()

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
            "compacting": bool(compaction and compaction.get("status") in {"queued", "running"}),
            "compaction": compaction,
        }
        return snap

    def wait_for_tag(self) -> None:
        """Block until tagging finishes. Compaction may still be running."""
        if self._pending_tag is not None:
            self._pending_tag.result()
            self._pending_tag = None

    def wait_for_compact(self) -> None:
        """Block until compaction finishes. Tagging should already be complete."""
        if self._pending_compact is not None:
            self._pending_compact.result()
            self._pending_compact = None

    def wait_for_complete(self) -> None:
        """Block until tag + compact both finish."""
        self.wait_for_tag()
        self.wait_for_compact()

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
        if self.is_conversation_deleted():
            logger.info(
                "fire_turn_complete skipped for deleted session %s",
                self.engine.config.conversation_id[:12],
            )
            return
        try:
            self._pending_tag = self._pool.submit(
                self._run_tag_turn, history_snapshot, payload_tokens, turn_id,
            )
        except RuntimeError:
            logger.info(
                "fire_turn_complete suppressed for shut down session %s",
                self.engine.config.conversation_id[:12],
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
                history,
                payload_tokens=payload_tokens,
                run_broad_split=False,
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
                    conversation_id=conversation_id,
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
                    "store_tag_count": len(
                        self.engine._store.get_all_tags(
                            conversation_id=conversation_id,
                        )
                    ),
                    "turn_pair_tokens": turn_pair_tokens,
                    "conversation_id": conversation_id,
                })

            # Fire compaction in background if needed — but NOT during ingestion.
            # During ingestion, only a fraction of turns are tagged. Compacting now
            # would process hundreds of untagged turns via expensive LLM fallback.
            # Post-ingestion compaction handles this once ingestion completes.
            if signal is not None:
                if self._state == SessionState.INGESTING:
                    logger.info("T%d compaction deferred (ingestion in progress)", turn)
                else:
                    try:
                        self._pending_compact = self._compact_pool.submit(
                            self._run_compact, history, signal, turn,
                        )
                    except RuntimeError:
                        logger.info(
                            "T%d compaction suppressed for shut down session %s",
                            turn, conversation_id[:12],
                        )

            self._queue_deferred_tag_split(history, turn)

        except Exception as e:
            logger.error("tag_turn error: %s", e, exc_info=True)

    def _record_tag_split_event(
        self,
        turn: int,
        conversation_id: str,
        split_result: SplitResult,
    ) -> None:
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
        if self.metrics:
            self.metrics.record({
                "type": "tag_split",
                "turn": turn,
                "tag": split_result.tag,
                "splittable": split_result.splittable,
                "new_tags": list(split_result.groups.keys()) if split_result.splittable else [],
                "conversation_id": conversation_id,
            })
        self.engine._engine_state.last_split_result = None  # consume

    def _clear_pending_split(self, future: Future) -> None:
        if self._pending_split is future:
            self._pending_split = None

    def _queue_deferred_tag_split(
        self,
        history: list[Message],
        turn: int,
    ) -> None:
        conversation_id = self.engine.config.conversation_id
        if self._pending_split is not None and not self._pending_split.done():
            logger.info(
                "T%d TAG_SPLIT already queued for %s — skipping requeue",
                turn, conversation_id[:12],
            )
            return
        try:
            future = self._compact_pool.submit(
                self._run_deferred_tag_split, history, turn,
            )
            self._pending_split = future
            future.add_done_callback(self._clear_pending_split)
        except RuntimeError:
            logger.info(
                "T%d tag split suppressed for shut down session %s",
                turn, conversation_id[:12],
            )

    def _run_deferred_tag_split(
        self,
        history: list[Message],
        turn: int,
    ) -> None:
        conversation_id = self.engine.config.conversation_id
        try:
            split_result = self.engine.process_broad_tag_split(
                history,
                mode="deferred",
            )
            if isinstance(split_result, SplitResult):
                self._record_tag_split_event(
                    turn,
                    conversation_id,
                    split_result,
                )
        except Exception as e:
            logger.error("tag_split error: %s", e, exc_info=True)

    def _run_compact(
        self,
        history: list[Message],
        signal: object,
        turn: int,
    ) -> None:
        """Background compaction — runs in _compact_pool, doesn't block next request."""
        conversation_id = self.engine.config.conversation_id
        operation_id = uuid.uuid4().hex[:12]
        compaction_started = time.monotonic()
        self._update_compaction_state(
            operation_id=operation_id,
            status="queued",
            phase="queued",
            phase_name="queued",
            overall_percent=0,
            done=0,
            total=0,
            phase_detail="waiting for compaction slot",
        )

        def _compact_progress(done, total, result, *, phase="", **kwargs):
            if self._compaction_cancelled.is_set():
                raise InterruptedError("Compaction cancelled (conversation deleted)")
            evt = {
                "type": "compaction_progress",
                "turn": turn,
                "done": done,
                "total": total,
                "phase": phase,
                "conversation_id": conversation_id,
                "operation_id": operation_id,
                "status": "running",
            }
            if result is not None:
                evt["primary_tag"] = result.primary_tag
                evt["tags"] = result.tags
                evt["original_tokens"] = result.original_tokens
                evt["summary_tokens"] = result.summary_tokens
            evt["elapsed_ms"] = round((time.monotonic() - compaction_started) * 1000, 1)
            for k, v in kwargs.items():
                evt[k] = v
            self._update_compaction_state(
                operation_id=operation_id,
                status="running",
                phase=phase,
                phase_name=str(evt.get("phase_name", phase)),
                done=done,
                total=total,
                overall_percent=evt.get("overall_percent"),
                elapsed_ms=evt.get("elapsed_ms"),
                eta_ms=evt.get("eta_ms"),
                heartbeat=bool(evt.get("heartbeat", False)),
                primary_tag=str(evt.get("primary_tag", "")),
                tag=str(evt.get("tag", "")),
                **{
                    k: v for k, v in evt.items()
                    if k not in {
                        "type", "turn", "done", "total", "phase", "conversation_id",
                        "operation_id", "status", "primary_tag", "tag",
                        "overall_percent", "elapsed_ms", "eta_ms", "heartbeat",
                        "phase_name",
                    }
                },
            )
            if self.metrics:
                self.metrics.record(evt)

        if not self._compaction_lock.acquire(blocking=False):
            logger.info("Compaction already running for %s — skipping", conversation_id)
            self._update_compaction_state(
                operation_id=operation_id,
                status="skipped",
                phase="skipped",
                phase_name="skipped",
                done=0,
                total=0,
                overall_percent=100,
                phase_detail="compaction already running",
            )
            return

        try:
            t0 = time.monotonic()
            self._update_compaction_state(
                operation_id=operation_id,
                status="running",
                phase="starting",
                phase_name="starting",
                overall_percent=0,
                done=0,
                total=0,
                elapsed_ms=0.0,
                phase_detail="starting compaction",
            )
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
                            "operation_id": operation_id,
                        })
                    self._update_compaction_state(
                        operation_id=operation_id,
                        status="completed",
                        phase="completed",
                        phase_name="completed",
                        done=report.segments_compacted,
                        total=report.segments_compacted,
                        overall_percent=100,
                        elapsed_ms=compact_ms,
                        primary_tag="",
                        tag="",
                        phase_detail=f"{report.segments_compacted} segments compacted",
                        segments=report.segments_compacted,
                        tokens_freed=report.tokens_freed,
                        tag_summaries_built=report.tag_summaries_built,
                    )
                else:
                    logger.info("T%d compaction skipped (no messages to compact)", turn)
                    self._update_compaction_state(
                        operation_id=operation_id,
                        status="skipped",
                        phase="skipped",
                        phase_name="skipped",
                        done=0,
                        total=0,
                        overall_percent=100,
                        elapsed_ms=compact_ms,
                        phase_detail="no messages to compact",
                    )

            except Exception as e:
                elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                self._update_compaction_state(
                    operation_id=operation_id,
                    status="failed",
                    phase="failed",
                    phase_name="failed",
                    overall_percent=None,
                    elapsed_ms=elapsed_ms,
                    error=str(e),
                    phase_detail="compaction crashed",
                )
                if self.metrics:
                    self.metrics.record({
                        "type": "compaction_error",
                        "turn": turn,
                        "conversation_id": conversation_id,
                        "operation_id": operation_id,
                        "error": str(e),
                        "elapsed_ms": elapsed_ms,
                    })
                logger.error("compact_if_needed error: %s", e, exc_info=True)
        finally:
            self._compaction_lock.release()

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
        return (
            self.engine.config.conversation_id in self._ingested_conversations
            and not self.has_pending_indexing()
        )

    def reconcile_history_bootstrap(self, history_pairs: list[Message]) -> bool:
        """Finalize a restored session once the first post-restart history arrives."""
        conversation_id = self.engine.config.conversation_id
        if conversation_id in self._ingested_conversations:
            return True
        if self.has_pending_indexing():
            return False
        existing_turns = self._indexed_turn_count()
        needed_turns = len(history_pairs) // 2
        if existing_turns <= 0:
            return False
        if history_pairs and existing_turns < needed_turns:
            return False
        self._ingested_conversations.add(conversation_id)
        if history_pairs:
            self._record_ingestion_watermark(history_pairs, conversation_id)
        return True

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
        # Invalidate Redis cache — separate from store deletion so a store failure
        # doesn't leave a stale snapshot that resurrects the purged conversation
        try:
            if hasattr(self.engine, '_session_cache') and self.engine._session_cache:
                self.engine._session_cache.delete_conversation(conversation_id)
        except Exception:
            pass
        self._clear_runtime_state(conversation_id)

        return True

    def _rebind_engine_references(self) -> None:
        """Refresh delegate references after replacing engine runtime state."""
        new_tti = self.engine._turn_tag_index
        new_es = self.engine._engine_state
        for attr in ("_tagging", "_compaction", "_retrieval", "_search"):
            delegate = getattr(self.engine, attr, None)
            if delegate is None:
                continue
            if hasattr(delegate, "_turn_tag_index"):
                delegate._turn_tag_index = new_tti
            if hasattr(delegate, "_engine_state"):
                delegate._engine_state = new_es
        if hasattr(self.engine, "_retriever"):
            self.engine._retriever._turn_tag_index = new_tti
        retrieval = getattr(self.engine, "_retrieval", None)
        if retrieval and hasattr(retrieval, "_retriever"):
            retrieval._retriever._turn_tag_index = new_tti

    def _clear_runtime_state(self, conversation_id: str) -> None:
        """Clear in-memory state for a conversation without touching the store."""
        self.engine._turn_tag_index = TurnTagIndex()
        self.engine._engine_state = EngineState()
        if self.provider:
            self.engine._engine_state.provider = self.provider
        self.engine._restored_working_set = []
        self.engine._restored_request_captures = []
        self.engine._restored_conversation_history = []
        self.engine._restored_pending_turns = []
        self.engine._restored_from_checkpoint = False
        self.engine._restored_checkpoint_source = ""
        self._rebind_engine_references()

        retrieval = getattr(self.engine, "_retrieval", None)
        if retrieval is not None:
            retrieval._last_retrieval_result = None
            retrieval._last_conversation_history = None
            retrieval._presented_segment_refs.clear()
        paging = getattr(self.engine, "_paging", None)
        if paging is not None:
            paging.working_set.clear()
        telemetry = getattr(self.engine, "_telemetry", None)
        if telemetry is not None:
            try:
                telemetry.reset()
            except AttributeError:
                pass

        self.conversation_history.clear()
        self._ingested_conversations.discard(conversation_id)
        self._ingested_first_hash.pop(conversation_id, None)
        self._ingested_turn_count.pop(conversation_id, None)
        self._latest_body = None
        self._ingestion_progress = (0, 0)
        self._manual_passthrough = False
        self._compaction_cancelled.clear()
        with self._compaction_state_lock:
            self._compaction_state = {}
        self._state = SessionState.ACTIVE
        self._pending_tag = None
        self._pending_compact = None
        self._pending_split = None
        self._last_compact_priority = ""
        self._initial_turns = None
        self._initial_tag_count = None
        self._initial_payload_kb = None
        self._last_payload_kb = 0.0
        self._last_enriched_payload_kb = 0.0
        self._initial_payload_tokens = None
        self._last_payload_tokens = 0
        self._last_enriched_payload_tokens = 0
        self._last_non_virtualizable_floor = 0
        self._inbound_payload_token_cache = None
        self._chain_snapshot_cache = {
            "loaded": False,
            "refs_by_turn": {},
        }
        self._total_requests = 0
        self._last_model = ""

    def _drain_background_work(self) -> None:
        """Wait for queued tag/compaction work without propagating old failures."""
        for attr in ("_pending_tag", "_pending_compact", "_pending_split"):
            future = getattr(self, attr, None)
            if future is None:
                continue
            try:
                future.result()
            except Exception:
                logger.warning(
                    "Background task failed while draining %s for conv=%s",
                    attr,
                    self.engine.config.conversation_id[:12],
                    exc_info=True,
                )
            finally:
                setattr(self, attr, None)

    def _request_background_stop(self) -> None:
        """Signal queued/running background work to stop without dropping handles."""
        self._compaction_cancelled.set()
        for attr in ("_pending_tag", "_pending_compact", "_pending_split"):
            future = getattr(self, attr, None)
            if future is None:
                continue
            try:
                future.cancel()
            except Exception:
                logger.debug(
                    "Failed to request cancellation for %s on conv=%s",
                    attr,
                    self.engine.config.conversation_id[:12],
                    exc_info=True,
                )

    def _cancel_background_work(self) -> None:
        """Cancel queued tag/compaction futures without blocking on completion."""
        self._compaction_cancelled.set()
        for attr in ("_pending_tag", "_pending_compact", "_pending_split"):
            future = getattr(self, attr, None)
            if future is None:
                continue
            try:
                future.cancel()
            except Exception:
                logger.debug(
                    "Failed to cancel %s for conv=%s",
                    attr,
                    self.engine.config.conversation_id[:12],
                    exc_info=True,
                )
            finally:
                setattr(self, attr, None)

    def _stop_ingestion_thread(
        self,
        *,
        timeout_s: float = 5.0,
        raise_on_timeout: bool = True,
    ) -> None:
        thread = self._ingestion_thread
        if thread is None or not thread.is_alive():
            self._ingestion_thread = None
            self._ingestion_cancel.clear()
            return

        self._ingestion_cancel.set()
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            msg = (
                "Ingestion thread did not stop within "
                f"{timeout_s:.1f}s for conv={self.engine.config.conversation_id[:12]}"
            )
            if raise_on_timeout:
                raise RuntimeError(msg)
            logger.warning(msg)
            return

        self._ingestion_thread = None
        self._ingestion_cancel.clear()

    def reset_for_conversation_deletion(
        self,
        conversation_id: str | None = None,
        *,
        authoritative: bool = False,
    ) -> None:
        """Stop live work and clear runtime state before deleting a conversation."""
        conv_id = conversation_id or self.engine.config.conversation_id
        if authoritative:
            self.mark_conversation_deleted()
            # Keep the old runtime intact until workers actually stop so a
            # stale compaction/tagger cannot repopulate freshly-cleared state.
            self._request_background_stop()
        self._stop_ingestion_thread(
            timeout_s=5.0,
            raise_on_timeout=authoritative,
        )
        self._drain_background_work()
        acquired_compaction_lock = self._compaction_lock.acquire(
            timeout=5.0,
        )
        if not acquired_compaction_lock:
            msg = (
                "Compaction lock did not quiesce within 5.0s for "
                f"conv={self.engine.config.conversation_id[:12]}"
            )
            if authoritative:
                raise RuntimeError(msg)
            logger.warning(msg)
        try:
            self._clear_runtime_state(conv_id)
        finally:
            if acquired_compaction_lock:
                self._compaction_lock.release()

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
            if int(self.engine._engine_state.compacted_through) > 0:
                return
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

    def ingest_if_needed(
        self,
        history_pairs: list[Message],
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> None:
        """Bootstrap TurnTagIndex from pre-existing history (once per session).

        Double-checked locking: fast path skips the lock entirely.
        """
        conversation_id = self.engine.config.conversation_id
        if conversation_id in self._ingested_conversations and not self.has_pending_indexing():
            # Check for history widening even after ingestion is "done"
            self._check_history_widening(history_pairs, conversation_id)
            if conversation_id in self._ingested_conversations:
                return  # No widening detected
        with self._ingestion_lock:
            if conversation_id in self._ingested_conversations:
                return
            t0 = time.monotonic()
            if tool_output_refs_by_turn is None:
                turns = self.engine.ingest_history(history_pairs)
            else:
                turns = self.engine.ingest_history(
                    history_pairs,
                    tool_output_refs_by_turn=tool_output_refs_by_turn,
                )
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
                    preview = _strip_envelope(raw_content)[:200]
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

    def start_ingestion_if_needed(
        self,
        history_pairs: list[Message],
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> None:
        """Start non-blocking history ingestion in a background thread.

        Returns immediately — the session stays in INGESTING while the
        background thread tags historical turns.  If called while ingestion
        is already running, cancels the old thread and resumes from the
        last tagged turn (PROXY-013).
        """
        conversation_id = self.engine.config.conversation_id
        if conversation_id in self._ingested_conversations and not self.has_pending_indexing():
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
                self._indexed_turn_count(),
                self._ingestion_thread is not None and self._ingestion_thread.is_alive(),
            )

            # Clear stale ingestion events from any previous run so the
            # dashboard shows fresh progress for this ingestion.
            if self.metrics:
                self.metrics.clear_ingestion_events(conversation_id)

            if not history_pairs:
                self._ingested_conversations.add(conversation_id)
                return

            # Skip if persisted TurnTagIndex already covers history
            existing_turns = self._indexed_turn_count()
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
                if tool_output_refs_by_turn is not None:
                    tool_output_refs_by_turn = {
                        turn_idx - existing_turns: refs
                        for turn_idx, refs in tool_output_refs_by_turn.items()
                        if turn_idx >= existing_turns
                    }
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
                existing_turns = self._indexed_turn_count()

                logger.info(
                    "INGEST Cancel at T%d/%d, resuming from T%d (conversation=%s)",
                    done, total, existing_turns, conversation_id[:12],
                )

                # Verify hash at handoff point
                self._verify_handoff_hash(history_pairs, existing_turns)

                # Slice to remaining pairs only
                history_pairs = list(history_pairs[existing_turns * 2:])
                if tool_output_refs_by_turn is not None:
                    tool_output_refs_by_turn = {
                        turn_idx - existing_turns: refs
                        for turn_idx, refs in tool_output_refs_by_turn.items()
                        if turn_idx >= existing_turns
                    }
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
                args=(list(history_pairs), existing_turns, total, tool_output_refs_by_turn),
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
        self,
        initial_pairs: list[Message],
        baseline: int = 0,
        cumulative_total: int = 0,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> None:
        """Background thread: ingest initial pairs, then catch up any gap."""
        conversation_id = self.engine.config.conversation_id
        cancelled = False
        try:
            # Tag all initial history
            self._ingest_pairs_with_progress(
                initial_pairs,
                baseline=baseline,
                cumulative_total=cumulative_total or None,
                tool_output_refs_by_turn=tool_output_refs_by_turn,
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
        except StaleConversationWriteError as e:
            cancelled = True
            logger.info(
                "Ingestion abandoned for deleted/stale conversation %s: %s",
                conversation_id[:12],
                e,
            )
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
        self,
        pairs: list[Message],
        baseline: int = 0,
        cumulative_total: int | None = None,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
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
                    preview = _strip_envelope(raw_content)[:200]
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

        ingest_kwargs = {
            "progress_callback": on_progress,
            "turn_offset": baseline,
        }
        if tool_output_refs_by_turn is not None:
            ingest_kwargs["tool_output_refs_by_turn"] = tool_output_refs_by_turn
        turns = self.engine.ingest_history(pairs, **ingest_kwargs)
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

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        try:
            self._stop_ingestion_thread(
                timeout_s=5.0 if wait else 0.1,
                raise_on_timeout=False,
            )
        except Exception:
            logger.warning("Failed to stop ingestion thread during shutdown", exc_info=True)
        if wait:
            self._drain_background_work()
        else:
            self._cancel_background_work()
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._compact_pool.shutdown(wait=wait, cancel_futures=cancel_futures)
