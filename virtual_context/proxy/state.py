"""Proxy session state management.

Contains ProxyState, SessionState, and _IngestionCancelled — the core
state machine for non-blocking ingestion and turn-complete processing.
"""

from __future__ import annotations

import enum
import hashlib
import inspect
import logging
import os
import socket
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

from ..core.conversation_store import StaleConversationWriteError
from ..core.lifecycle_epoch import LifecycleEpochMismatch
from ..core.segmenter import pair_messages_into_turns
from ..engine import VirtualContextEngine
from ..core.turn_tag_index import TurnTagIndex
from ..types import CompactionLeaseLost, EngineState, Message, SplitResult, TurnTagEntry

from .helpers import (
    _strip_envelope,
    _extract_ingestible_messages,
)
from .metrics import ProxyMetrics

logger = logging.getLogger(__name__)

# Ingestion lease TTL (seconds). A claim older than this is considered stale
# and may be reclaimed by another worker. Used by step 6 of
# ``ProxyState.handle_prepare_payload`` via ``claim_ingestion_lease``.
INGESTION_LEASE_TTL_S: float = 30.0

# Ordered phase plan used by ``_run_compact`` to drive the DB-backed
# compaction_operation row. The initial ``"starting"`` phase is seeded
# by ``enter_compaction``; every other name corresponds to a
# ``phase_name=`` value emitted by the compactor pipeline's progress
# callback (see ``core/compaction_pipeline.py`` + ``core/compactor.py``).
# ``phase_count`` = len of this list so downstream consumers can render
# "step i of N" progress without re-deriving the plan.
_COMPACT_PHASE_PLAN: tuple[str, ...] = (
    "starting",
    "segment_tagging",
    "segment_grouping",
    "segment_postprocess",
    "compactor",
    "store",
    "tag_summaries",
)
_COMPACT_PHASE_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(_COMPACT_PHASE_PLAN)
}

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


@dataclass(frozen=True)
class PhaseDecision:
    """Result of ``ProxyState.handle_prepare_payload``.

    Describes the phase the conversation should settle into after the
    prepare-payload flow runs, plus whether this worker claimed the
    ingestion lease on this call.

    ``started_tagger``: historically named for the per-row tagger thread
    that ``handle_prepare_payload`` used to spawn directly. That spawn
    was removed when tagger dispatch was delegated to the legacy
    ``start_ingestion_if_needed`` path (which runs ``_ingestion_thread``
    with the richer ``tag_turn`` semantics). The field now means
    "this call successfully claimed the ingestion lease" — a downstream
    tagger thread will be launched by ``start_ingestion_if_needed``
    later in the request flow.
    """

    phase: str
    started_tagger: bool


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
        # Sidecar heartbeat thread: refreshes the ingestion lease every
        # ``INGESTION_LEASE_TTL_S / 2`` seconds while ``_ingestion_thread``
        # is alive. Separate from ``_ingestion_thread`` so a single
        # long-running tagging turn (>TTL) cannot let the lease expire.
        self._heartbeat_thread: threading.Thread | None = None
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
        self._raw_payload_entry_count: int = 0
        self._ingestible_entry_count: int = 0
        self._skipped_payload_entry_count: int = 0
        self._last_non_virtualizable_floor: int = 0  # outbound - VC context tokens
        self._shared_live_turn_count: int = 0
        self._shared_history_message_count: int = 0
        self._inbound_payload_token_cache = None
        self._outbound_payload_token_cache = None
        self._restore_readiness_pending: bool = False
        self._restore_readiness_signature: tuple[int, int, int, int] | None = None
        self._chain_snapshot_cache: dict[str, object] = {
            "loaded": False,
            "refs_by_turn": {},
            "recovery_loaded": False,
            "recovery_manifest": [],
        }
        self._background_state_lock = threading.Lock()
        self._queued_tag_turns: dict[int, str] = {}
        self._queued_compaction_request: dict[str, object] | None = None
        self._active_compaction_target_end = -1
        self._last_completed_compaction_target_end = -1
        # Live request counter: incremented on each user turn processed through proxy
        self._total_requests: int = 0
        # Upstream context window enforcement
        self._instance_upstream_limit: int = 0  # set by create_app from ProxyInstanceConfig
        self._last_model: str = ""  # last model name seen (for dashboard)

        # Ingestion ownership (step 6 of handle_prepare_payload). Unique per
        # process instance so claim_ingestion_lease can distinguish
        # concurrent workers.
        self._worker_id: str = (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )

        # The operation_id of the currently-running compaction, or None when no
        # compaction is in flight.  Set in _submit_compaction_request (before
        # submit) when a preexisting_operation_id is provided (takeover path),
        # and cleared in the _run_compact_wrapper finally block so the takeover
        # predicate `self._active_compaction_op != claim.prev_operation_id` is
        # always authoritative.
        self._active_compaction_op: str | None = None

        # Wire this worker's identity into the compaction pipeline so the
        # per-write ownership guard on store_segment can scope each INSERT
        # to the live compaction_operation row.  Done here (post-_worker_id
        # assignment) rather than at CompactionPipeline construction time
        # because the engine is constructed before ProxyState initialises
        # _worker_id.
        if hasattr(engine, "_compaction") and engine._compaction is not None:
            engine._compaction._worker_id = self._worker_id

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
        if engine._restored_from_checkpoint:
            self.note_engine_restore(force=True)

    def hydrate_from_session_state(self, state) -> None:
        """Hydrate both engine state and worker-visible runtime counters."""
        if hasattr(self.engine, "hydrate_from_session_state"):
            self.engine.hydrate_from_session_state(state)

        raw_state = str(getattr(state, "session_state", "") or "").strip().lower()
        if raw_state:
            try:
                self._state = SessionState(raw_state)
            except ValueError:
                logger.debug(
                    "Ignoring unknown shared session state '%s' for conv=%s",
                    raw_state,
                    self.engine.config.conversation_id[:12],
                )

        def _int_attr(name: str) -> int:
            try:
                return int(getattr(state, name, 0) or 0)
            except (TypeError, ValueError):
                return 0

        def _float_attr(name: str) -> float:
            try:
                return float(getattr(state, name, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        self._shared_live_turn_count = max(0, _int_attr("live_turn_count"))
        self._shared_history_message_count = max(0, _int_attr("history_message_count"))
        done = max(0, _int_attr("ingestion_done"))
        total = max(done, _int_attr("ingestion_total"))
        self._ingestion_progress = (done, total)
        self._raw_payload_entry_count = max(0, _int_attr("raw_payload_entry_count"))
        self._ingestible_entry_count = max(0, _int_attr("ingestible_entry_count"))
        self._skipped_payload_entry_count = max(0, _int_attr("skipped_payload_entry_count"))
        self._last_payload_kb = max(0.0, _float_attr("last_payload_kb"))
        self._last_payload_tokens = max(0, _int_attr("last_payload_tokens"))
        self.note_engine_restore(force=True)

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

    @staticmethod
    def _group_history_messages(messages: list[Message] | None) -> list:
        if not messages:
            return []
        return pair_messages_into_turns(list(messages))

    @staticmethod
    def _is_completed_turn(turn) -> bool:
        return any(msg.role == "assistant" for msg in getattr(turn, "messages", []))

    def _completed_history_groups(self, messages: list[Message] | None = None) -> list:
        grouped = self._group_history_messages(self.conversation_history if messages is None else messages)
        if grouped and not self._is_completed_turn(grouped[-1]):
            return grouped[:-1]
        return grouped

    def _completed_history_messages(self, messages: list[Message] | None = None) -> list[Message]:
        completed_messages: list[Message] = []
        for pair in self._completed_history_groups(messages):
            completed_messages.extend(pair.messages)
        return completed_messages

    def _history_turn_count(self, messages: list[Message] | None = None) -> int:
        return len(self._completed_history_groups(messages))

    def _slice_messages_from_turn(self, messages: list[Message], turn_offset: int) -> list[Message]:
        if turn_offset <= 0:
            return self._completed_history_messages(messages)
        grouped = self._completed_history_groups(messages)
        if turn_offset >= len(grouped):
            return []
        sliced: list[Message] = []
        for pair in grouped[turn_offset:]:
            sliced.extend(pair.messages)
        return sliced

    def _combined_turn_text(self, messages: list[Message], turn_index: int) -> str:
        grouped = self._completed_history_groups(messages)
        if turn_index < 0 or turn_index >= len(grouped):
            return ""
        return " ".join(msg.content for msg in grouped[turn_index].messages)

    def _completed_turn_count(self) -> int:
        raw = getattr(getattr(self.engine, "_engine_state", None), "last_completed_turn", -1)
        marker = raw if isinstance(raw, int) else -1
        history_turns = self._history_turn_count()
        return max(history_turns, marker + 1)

    def has_pending_indexing(self) -> bool:
        return self._completed_turn_count() > self._indexed_turn_count()

    def _restore_signature(self) -> tuple[int, int, int, int]:
        try:
            compacted = int(getattr(self.engine._engine_state, "compacted_prefix_messages", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            compacted = 0
        return (
            self._indexed_turn_count(),
            self._completed_turn_count(),
            compacted,
            self._history_turn_count(),
        )

    def _can_activate_from_persisted_state(
        self,
        history_messages: list[Message] | None = None,
    ) -> bool:
        existing_turns = self._indexed_turn_count()
        if existing_turns <= 0:
            return False
        if self.has_pending_indexing():
            return False
        needed_turns = self._history_turn_count(history_messages)
        if needed_turns > 0 and existing_turns < needed_turns:
            return False
        return True

    def note_engine_restore(self, *, force: bool = False) -> bool:
        """Mark that routing readiness must be revalidated after an engine hydrate.

        Returns True when persisted state looks ready enough to re-check against
        the next inbound history window.
        """
        signature = self._restore_signature()
        if not force and signature == self._restore_readiness_signature:
            return self._restore_readiness_pending
        self._restore_readiness_signature = signature
        conversation_id = self.engine.config.conversation_id
        ready_candidate = self._can_activate_from_persisted_state()
        self._restore_readiness_pending = ready_candidate
        if not ready_candidate:
            self._ingested_conversations.discard(conversation_id)
        return ready_candidate

    def resolve_prepare_state(
        self,
        history_messages: list[Message],
    ) -> tuple[SessionState, str | None]:
        """Return the routing state for the current request.

        This keeps legitimate passthrough intact while preventing restored,
        durably-ready conversations from falling back into passthrough solely
        because a worker-local ingested marker was lost.
        """
        conversation_id = self.engine.config.conversation_id
        if self._manual_passthrough:
            return SessionState.PASSTHROUGH, "manual_override"

        if self.has_pending_indexing():
            return SessionState.INGESTING, "pending_indexing"

        if not history_messages:
            self._ingested_conversations.add(conversation_id)
            self._restore_readiness_pending = False
            self._ingested_turn_count[conversation_id] = 0
            return SessionState.ACTIVE, None

        if conversation_id in self._ingested_conversations and not self._restore_readiness_pending:
            widened = self._check_history_widening(history_messages, conversation_id)
            if widened:
                return SessionState.PASSTHROUGH, "history_widening"
            return SessionState.ACTIVE, None

        if self._can_activate_from_persisted_state(history_messages):
            self._ingested_conversations.add(conversation_id)
            self._restore_readiness_pending = False
            if history_messages:
                self._record_ingestion_watermark(history_messages, conversation_id)
            return SessionState.ACTIVE, None

        self._restore_readiness_pending = False
        if conversation_id in self._ingested_conversations:
            widened = self._check_history_widening(history_messages, conversation_id)
            if widened:
                return SessionState.PASSTHROUGH, "history_widening"

        reason = "initial_ingest" if self._indexed_turn_count() <= 0 else "restore_not_ready"
        self._ingested_conversations.discard(conversation_id)
        return SessionState.PASSTHROUGH, reason

    # ------------------------------------------------------------------
    # Progress-event publishes (Task A31)
    # ------------------------------------------------------------------
    # Both helpers are called AFTER a successful ``set_phase`` (or epoch
    # increment) so a failed epoch-guard write never emits a spurious
    # event. The bus is per-Engine and thread-safe; subscriber exceptions
    # are swallowed inside ``ProgressEventBus.publish``.

    def _publish_phase_transition(self, old_phase: str, new_phase: str) -> None:
        from ..core.progress_events import PhaseTransitionEvent
        self.engine.progress_event_bus.publish(PhaseTransitionEvent(
            conversation_id=self.engine.config.conversation_id,
            lifecycle_epoch=int(self.engine._engine_state.lifecycle_epoch),
            kind="phase_transition",
            timestamp=time.time(),
            old_phase=old_phase,
            new_phase=new_phase,
        ))

    def _publish_lifecycle_reset(self, old_epoch: int, new_epoch: int) -> None:
        from ..core.progress_events import LifecycleResetEvent
        self.engine.progress_event_bus.publish(LifecycleResetEvent(
            conversation_id=self.engine.config.conversation_id,
            lifecycle_epoch=int(new_epoch),
            kind="lifecycle_reset",
            timestamp=time.time(),
            old_epoch=int(old_epoch),
            new_epoch=int(new_epoch),
        ))

    def _publish_compaction_progress(self) -> None:
        """Publish a ``CompactionProgressEvent`` from the current snapshot.

        Reads the live progress snapshot for this conversation and — when an
        active compaction row exists — emits an event carrying its
        operation_id / phase_name / phase_index / phase_count / status.
        Used by ``enter_compaction``, ``advance_compaction_phase``, and
        ``exit_compaction`` to surface phase progress to subscribers.
        """
        from ..core.progress_events import CompactionProgressEvent

        conv = self.engine.config.conversation_id
        snap = self.engine._store.read_progress_snapshot(conv)
        if snap.active_compaction is None:
            return
        c = snap.active_compaction
        self.engine.progress_event_bus.publish(CompactionProgressEvent(
            conversation_id=conv,
            lifecycle_epoch=int(self.engine._engine_state.lifecycle_epoch),
            kind="compaction",
            timestamp=time.time(),
            operation_id=c.operation_id,
            phase_name=c.phase_name,
            phase_index=c.phase_index,
            phase_count=c.phase_count,
            status=c.status,
        ))

    def _tagger_run(self) -> bool:
        """Background tagger loop. Processes untagged canonical rows until
        none remain, then completes the episode. Exits cleanly on epoch
        mismatch (lifecycle bumped by delete+resurrect).

        Returns ``True`` iff the untagged queue was drained to empty within
        the caller's lifecycle — i.e. the episode was completed (or was
        already completed by another worker) and the phase is ``active``.
        Returns ``False`` when the sweep exited early on an epoch boundary
        check or a rejected row-mark write. Callers use the return value to
        trigger post-ingestion side effects (watermark, compaction,
        SessionState transition) that ``_finalize_legacy_ingestion`` would
        otherwise own — ``_finalize_legacy_ingestion`` returns False after
        this sweep because the episode is already ``completed``.

        Four boundary ``verify_epoch`` checks ensure a stale tagger cannot
        affect a new lifecycle even if raced:

        1. Top of loop — before the row fetch.
        2. Before the completion write.
        3. Before each row-mark write.
        4. Before each heartbeat refresh.

        ``my_epoch`` is captured at spawn time. Every SQL call filters on
        ``my_epoch`` so stale-epoch writes are rejected at the DB level even
        if ``verify_epoch`` races past a lifecycle change.
        """
        conversation_id = self.engine.config.conversation_id
        # Capture epoch at spawn time. Every SQL call filters on my_epoch
        # so stale-epoch writes are rejected even if verify_epoch races past
        # a lifecycle change.
        my_epoch = int(self.engine._engine_state.lifecycle_epoch)

        def _verify_or_exit() -> bool:
            try:
                self.engine.verify_epoch()
                return True
            except LifecycleEpochMismatch:
                logger.info(
                    "Tagger %s exiting: lifecycle epoch changed for conv=%s",
                    self._worker_id, conversation_id[:12],
                )
                return False

        while True:
            # Boundary check #1 — before the row fetch.
            if not _verify_or_exit():
                return False

            batch = self.engine._store.iter_untagged_canonical_rows(
                conversation_id=conversation_id,
                expected_lifecycle_epoch=my_epoch,
                batch_size=32,
            )
            if not batch:
                # Boundary check #2 — before the completion write.
                if not _verify_or_exit():
                    return False
                completed = self.engine._store.complete_ingestion_episode(
                    conversation_id=conversation_id,
                    lifecycle_epoch=my_epoch,
                    worker_id=self._worker_id,
                )
                if completed:
                    ok = self.engine._store.set_phase(
                        conversation_id=conversation_id,
                        lifecycle_epoch=my_epoch,
                        phase="active",
                    )
                    if ok:
                        self._publish_phase_transition("ingesting", "active")
                        return True
                    # Episode is closed but the phase flip was rejected on
                    # stale epoch. Another lifecycle owns local state now;
                    # don't signal "drained for this caller's lifecycle".
                    return False
                # complete_ingestion_episode returned False. Either:
                #   (a) a new untagged row appeared between our scan and
                #       the completion guard (race) — loop and drain it; OR
                #   (b) the episode was already closed by another worker
                #       or the lifecycle moved — not ours to finalize.
                if not _verify_or_exit():
                    return False
                retry_batch = self.engine._store.iter_untagged_canonical_rows(
                    conversation_id=conversation_id,
                    expected_lifecycle_epoch=my_epoch,
                    batch_size=1,
                )
                if not retry_batch:
                    return False
                continue

            for row in batch:
                # Run the existing tagging pipeline on this row. On failure
                # we leave ``tagged_at`` NULL so a later run can retry the
                # row without dropping it on the floor.
                try:
                    self._run_tagging_pipeline(row)
                except Exception:
                    logger.exception(
                        "Tagger %s failed to tag row %s",
                        self._worker_id, row.canonical_turn_id[:12],
                    )
                    # On tagging failure, skip this row. A later run can
                    # retry if tagged_at is still NULL.
                    continue

                # Boundary check #3 — before the row-mark write.
                if not _verify_or_exit():
                    return False

                marked = self.engine._store.mark_canonical_row_tagged(
                    canonical_turn_id=row.canonical_turn_id,
                    conversation_id=conversation_id,
                    expected_lifecycle_epoch=my_epoch,
                )
                if not marked:
                    logger.info(
                        "Tagger %s exiting: mark_canonical_row_tagged"
                        " rejected (epoch mismatch); tagged_at untouched"
                        " for turn_id=%s",
                        self._worker_id, row.canonical_turn_id[:12],
                    )
                    return False

                # Boundary check #4 — before the heartbeat write.
                if not _verify_or_exit():
                    return False

                self.engine._store.refresh_ingestion_heartbeat(
                    conversation_id=conversation_id,
                    lifecycle_epoch=my_epoch,
                    worker_id=self._worker_id,
                )
                # Publish IngestionProgressEvent with fresh canonical counts.
                from ..core.progress_events import IngestionProgressEvent
                snap = self.engine._store.read_progress_snapshot(conversation_id)
                self.engine.progress_event_bus.publish(IngestionProgressEvent(
                    conversation_id=conversation_id,
                    lifecycle_epoch=my_epoch,
                    kind="ingestion",
                    timestamp=time.time(),
                    episode_id=snap.active_episode.episode_id if snap.active_episode else "",
                    done=snap.done_ingestible,
                    total=snap.total_ingestible,
                ))

    def _run_tagging_pipeline(self, row) -> None:
        """Run the real tagging pipeline on a single canonical row.

        Delegates to ``TaggingPipeline.tag_canonical_row`` which runs the
        configured tag generator against the row's combined user/assistant
        text and re-saves the row with ``primary_tag``, ``tags``,
        ``fact_signals``, and ``code_refs`` populated. ``tagged_at`` is left
        untouched — the outer ``_tagger_run`` loop flips it atomically via
        ``mark_canonical_row_tagged`` on the next step, which keeps the
        epoch guard as the single source of truth for "this row has been
        processed in this lifecycle".
        """
        self.engine._tagging.tag_canonical_row(row)

    # ------------------------------------------------------------------
    # Compaction lifecycle (Task A29)
    # ------------------------------------------------------------------
    # The conversation enters ``'compacting'`` via ``enter_compaction``; the
    # compactor records intermediate phase progress via
    # ``advance_compaction_phase``; and ``exit_compaction`` finalises the
    # operation and drains pending payload entries atomically, deciding
    # between re-entering ``'ingesting'`` (when untagged canonical rows
    # remain) and returning to ``'active'`` on a single
    # ``drain_compaction_exit`` transaction.
    #
    # All three methods are epoch-guarded. ``enter_compaction`` hard-checks
    # via ``verify_epoch`` (a stale caller at entry must not transition a
    # fresh lifecycle's phase). ``advance_compaction_phase`` yields on
    # mismatch instead of raising — it is called inside the compactor's
    # long-running loop where a lifecycle bump is a normal cancellation
    # path. ``exit_compaction`` relies on the SQL-level epoch filter on
    # each store call (complete/fail + drain) rather than raising, so a
    # stale compactor can always finish cleaning up its own bookkeeping
    # without stomping a new lifecycle.
    #
    # Event publishes: ``PhaseTransitionEvent`` (A31) fires on every
    # successful phase flip; ``CompactionProgressEvent`` (A33) fires on
    # enter (status ``queued``/``running``), on each phase advance, and
    # on exit (terminal ``completed``/``failed``). Publishes are gated on
    # the successful store write so a rejected epoch-guarded call never
    # emits a spurious event.

    def enter_compaction(
        self, *, phase_count: int, initial_phase_name: str = "init",
        operation_id: str | None = None,
    ) -> None:
        """Transition phase from ``'active'`` to ``'compacting'`` and start
        a fresh ``compaction_operation`` row.

        Epoch-guarded via ``verify_epoch`` at entry. The phase write is
        additionally epoch-filtered at the SQL layer — a stale caller whose
        epoch does not match the authoritative conversations row sees
        ``set_phase`` return False and this method exits without writing
        the operation row.

        *operation_id*: when provided, the DB row is inserted with this PK
        rather than a store-generated UUID. This ensures the row PK matches
        the id the caller already threaded into downstream per-write
        ownership-guard kwargs.
        """
        conv = self.engine.config.conversation_id
        self.engine.verify_epoch()
        epoch = int(self.engine._engine_state.lifecycle_epoch)
        ok = self.engine._store.set_phase(
            conversation_id=conv,
            lifecycle_epoch=epoch,
            phase="compacting",
        )
        if not ok:
            logger.info(
                "enter_compaction aborted: phase write rejected"
                " (epoch mismatch) for conv=%s",
                conv[:12],
            )
            return
        self.engine._store.start_compaction_operation(
            conversation_id=conv,
            lifecycle_epoch=epoch,
            worker_id=self._worker_id,
            phase_count=phase_count,
            phase_name=initial_phase_name,
            operation_id=operation_id,
        )
        self._publish_compaction_progress()
        self._publish_phase_transition("active", "compacting")

    def advance_compaction_phase(
        self, *, phase_index: int, phase_name: str,
    ) -> None:
        """Advance the active compaction's phase index/name.

        Called from inside the compactor pipeline between phases. On a
        lifecycle-epoch mismatch the method yields silently — the
        compactor is expected to cancel itself on the next boundary — so
        a stale advance write does not raise through the caller. The
        store-level ``advance_compaction_phase`` is also epoch-filtered
        at the SQL layer (double-scoped via the correlated subquery
        against ``conversations.lifecycle_epoch``).
        """
        from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch

        conv = self.engine.config.conversation_id
        try:
            self.engine.verify_epoch()
        except LifecycleEpochMismatch:
            logger.info(
                "Compactor %s yielding phase advance: epoch changed"
                " for conv=%s",
                self._worker_id, conv[:12],
            )
            return
        self.engine._store.advance_compaction_phase(
            conversation_id=conv,
            lifecycle_epoch=int(self.engine._engine_state.lifecycle_epoch),
            worker_id=self._worker_id,
            phase_index=phase_index,
            phase_name=phase_name,
        )
        self._publish_compaction_progress()

    def exit_compaction(
        self, *, success: bool, error_message: str | None = None,
    ) -> None:
        """Finalise the compaction operation and transition phase.

        Records the operation's terminal status (``'completed'`` vs
        ``'failed'``) first, then calls ``drain_compaction_exit`` which
        atomically decides between re-entering ``'ingesting'`` (with a
        seeded episode row) and returning to ``'active'``, draining
        ``pending_raw_payload_entries`` in the same transaction as the
        phase UPDATE.

        Every store call is epoch-filtered at the SQL layer. A stale
        caller whose epoch no longer matches simply sees ``False``/``None``
        returns — no exception is raised from this method.
        """
        from ..core.progress_events import CompactionProgressEvent

        conv = self.engine.config.conversation_id
        epoch = int(self.engine._engine_state.lifecycle_epoch)
        # Capture the active compaction row BEFORE the terminal write so we
        # can emit a terminal CompactionProgressEvent with the real
        # operation_id/phase metadata. ``read_progress_snapshot`` filters
        # ``active_compaction`` on ``status IN ('queued','running')`` — once
        # complete/fail lands the row drops out of the snapshot, so we must
        # grab it first.
        pre_snap = self.engine._store.read_progress_snapshot(conv)
        active = pre_snap.active_compaction
        if success:
            self.engine._store.complete_compaction_operation(
                conversation_id=conv,
                lifecycle_epoch=epoch,
                worker_id=self._worker_id,
            )
            terminal_status = "completed"
        else:
            self.engine._store.fail_compaction_operation(
                conversation_id=conv,
                lifecycle_epoch=epoch,
                worker_id=self._worker_id,
                error_message=error_message or "unknown",
            )
            terminal_status = "failed"
        # Publish terminal CompactionProgressEvent carrying the
        # now-terminal status alongside the operation_id/phase metadata
        # captured before the status flip. Only emit when we actually saw
        # an active row at entry — a stale caller whose row was already
        # cleaned up should not fabricate a spurious terminal event.
        if active is not None:
            self.engine.progress_event_bus.publish(CompactionProgressEvent(
                conversation_id=conv,
                lifecycle_epoch=epoch,
                kind="compaction",
                timestamp=time.time(),
                operation_id=active.operation_id,
                phase_name=active.phase_name,
                phase_index=active.phase_index,
                phase_count=active.phase_count,
                status=terminal_status,
            ))
        # Drain pending + transition phase atomically. The decision
        # between 'ingesting' and 'active' is made inside the same
        # transaction as the phase UPDATE via a direct EXISTS check on
        # canonical_turns.tagged_at, so a concurrent tagger cannot flip
        # the answer between read and write.
        new_phase = self.engine._store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=epoch,
            worker_id=self._worker_id,
        )
        if new_phase in ("ingesting", "active"):
            self._publish_phase_transition("compacting", new_phase)

    def handle_prepare_payload(
        self,
        *,
        body: dict,
        payload_accounting: dict,
    ) -> PhaseDecision:
        """Run the ingestion flow (spec §5, steps 1-8) for one inbound request.

        Manages the DB bookkeeping side of ingestion: canonical row
        persistence (IngestReconciler), per-request metadata, phase
        transitions, ingestion-episode widening, and lease claims.

        Tagger dispatch is delegated to ``start_ingestion_if_needed`` —
        this method only manages DB bookkeeping and phase transitions.
        The legacy ``_ingestion_thread`` (per-pair ``tag_turn`` with
        full history) is the authoritative tagger; this method never
        spawns a tagger thread itself, avoiding split-brain with two
        tagger threads racing on the same canonical rows.

        Every DB boundary op is epoch-safe:

        * ``IngestReconciler.ingest_batch`` takes
          ``expected_lifecycle_epoch`` and filters in SQL (A18).
        * ``update_request_metadata`` filters on ``lifecycle_epoch`` in
          its ``UPDATE`` (A20) and returns ``False`` when the caller is
          stale.
        * A defense-in-depth ``verify_epoch()`` call fires before each
          write so races between the entry check and the write are caught
          early.

        Raises
        ------
        LifecycleEpochMismatch
            When the in-memory epoch drifts from the authoritative DB
            epoch (external delete+resurrect). Callers rehydrate and
            retry.
        """
        conversation_id = self.engine.config.conversation_id
        new_raw = int(payload_accounting.get("raw_payload_entry_count", 0) or 0)
        new_ing = int(payload_accounting.get("ingestible_entry_count", 0) or 0)

        # Step 2: entry-verify. Re-verified before each subsequent write as
        # defense in depth — a delete+resurrect between verify and write
        # would otherwise stomp the new lifecycle's row.
        self.engine.verify_epoch()
        my_epoch = int(self.engine._engine_state.lifecycle_epoch)

        # Step 3: persist canonical rows via IngestReconciler. Skip when
        # the body carries no payload keys — empty bodies arrive on paths
        # like resurrect/init where there's nothing to merge yet.
        if body and any(k in body for k in ("messages", "input", "contents")):
            from ..proxy.formats import detect_format
            fmt = detect_format(body)
            try:
                self.engine._ingest_reconciler.ingest_batch(
                    conversation_id,
                    body=body,
                    fmt=fmt,
                    expected_lifecycle_epoch=my_epoch,
                )
            except AttributeError:
                # No reconciler configured on engine — acceptable for
                # test harnesses that inject a mock engine.
                logger.debug(
                    "handle_prepare_payload: engine._ingest_reconciler missing; "
                    "skipping canonical row persistence for conv=%s",
                    conversation_id[:12],
                )

        # Defense-in-depth: fresh epoch check before the next write.
        self.engine.verify_epoch()

        # Step 5 (always): update per-request metadata. Epoch-filtered in SQL.
        ok = self.engine._store.update_request_metadata(
            conversation_id=conversation_id,
            lifecycle_epoch=my_epoch,
            last_raw_payload_entries=new_raw,
            last_ingestible_payload_entries=new_ing,
        )
        if not ok:
            # Lifecycle bumped between the verify and the UPDATE. Stop.
            try:
                observed = int(self.engine._store.get_lifecycle_epoch(conversation_id))
            except (KeyError, NotImplementedError, AttributeError):
                observed = -1
            raise LifecycleEpochMismatch(
                conversation_id=conversation_id,
                expected=my_epoch,
                observed=observed,
            )

        # Step 4: phase gate.
        phase = self.engine._store.get_conversation_phase(conversation_id)

        if phase == "deleted":
            # Resurrect bumps the lifecycle epoch and resets phase to 'init'.
            # Keep the engine's in-memory epoch in lockstep so subsequent
            # epoch-guarded writes on this call use the new lifecycle.
            old_epoch = my_epoch
            new_epoch = self.engine._store.increment_lifecycle_epoch_on_resurrect(
                conversation_id,
            )
            self.engine._engine_state.lifecycle_epoch = int(new_epoch)
            my_epoch = int(new_epoch)
            self._publish_lifecycle_reset(old_epoch, int(new_epoch))
            # Continue past the gate with the new epoch.
        elif phase == "compacting":
            self.engine.verify_epoch()
            import uuid as _uuid

            claim = self.engine._store.claim_compaction_lease(
                conversation_id=conversation_id,
                lifecycle_epoch=my_epoch,
                worker_id=self._worker_id,
                lease_ttl_s=INGESTION_LEASE_TTL_S,
            )

            if not claim.claimed:
                # Live owner on another worker — widen and return.
                widened = self.engine._store.widen_pending_raw_payload_entries(
                    conversation_id=conversation_id,
                    lifecycle_epoch=my_epoch,
                    value=new_raw,
                )
                if not widened:
                    return PhaseDecision(phase=phase, started_tagger=False)
                return PhaseDecision(phase="compacting", started_tagger=False)

            # claim.claimed=True
            if self._active_compaction_op == claim.prev_operation_id:
                # We already own it — heartbeat refreshed. Widen and return.
                widened = self.engine._store.widen_pending_raw_payload_entries(
                    conversation_id=conversation_id,
                    lifecycle_epoch=my_epoch,
                    value=new_raw,
                )
                if not widened:
                    return PhaseDecision(phase=phase, started_tagger=False)
                return PhaseDecision(phase="compacting", started_tagger=False)

            # Takeover path.
            dead_op = claim.prev_operation_id
            new_op = _uuid.uuid4().hex
            phase_count = len(_COMPACT_PHASE_PLAN)  # 7
            fresh_takeover = self.engine._store.cleanup_abandoned_compaction(
                conversation_id=conversation_id,
                dead_operation_id=dead_op,
                new_operation_id=new_op,
                lifecycle_epoch=my_epoch,
                worker_id=self._worker_id,
                phase_count=phase_count,
            )

            if not fresh_takeover:
                # Cleanup's dead-op UPDATE matched zero rows → the dead_op was
                # already marked abandoned/completed by the time we ran. A peer
                # worker's earlier cleanup beat us to it, OR this call is an
                # idempotent retry after our own partial-progress attempt. In
                # either case, our ``new_op`` row was NEVER inserted (the
                # cleanup helper gates the INSERT on fresh_takeover=True to
                # preserve the one-active invariant), so submitting compaction
                # with preexisting_operation_id=new_op would reference a row
                # that doesn't exist — the heartbeat sidecar would fail every
                # refresh and every per-write guard would raise
                # CompactionLeaseLost on the first write.
                #
                # Correct behaviour: widen-and-return so the current POST
                # passes through; the actual current running op (either a
                # peer's new_op or none) will be seen on the next POST via the
                # normal predicate. Do NOT submit compaction.
                logger.info(
                    "COMPACTION_TAKEOVER_SKIP_NO_FRESH conv=%s dead_op=%s "
                    "worker=%s reason=dead_op_already_handled",
                    conversation_id[:12], dead_op, self._worker_id,
                )
                widened = self.engine._store.widen_pending_raw_payload_entries(
                    conversation_id=conversation_id,
                    lifecycle_epoch=my_epoch,
                    value=new_raw,
                )
                if not widened:
                    return PhaseDecision(phase=phase, started_tagger=False)
                return PhaseDecision(phase="compacting", started_tagger=False)

            logger.info(
                "COMPACTION_TAKEOVER conv=%s dead_op=%s new_op=%s worker=%s",
                conversation_id[:12], dead_op, new_op, self._worker_id,
            )

            # Spawn fresh compaction with preexisting_operation_id.
            #
            # CRITICAL: use the full restored ``self.conversation_history``
            # when available, not just the current POST body's messages. The
            # compaction pipeline persists whatever history it was given via
            # ``_commit_compaction_state(conversation_history)``
            # (compaction_pipeline.py:_run_compaction → _commit_compaction_state).
            # Seeding compaction from ``_extract_ingestible_messages(body)``
            # alone would persist a truncated snapshot whenever the takeover
            # POST's body is narrower than the full conversation — the exact
            # shape a "hey just ping" follow-up takes during an active
            # compaction. Mirror the pattern in ``_compact_after_ingestion``
            # (state.py:2228) which reads ``self.conversation_history`` first
            # and only falls back to the passed-in history when the engine's
            # in-memory history hasn't been hydrated yet.
            compact_history = (
                self.conversation_history
                if self.conversation_history
                else self._completed_history_messages(
                    _extract_ingestible_messages(body),
                )
            )
            from ..types import CompactionSignal
            signal = CompactionSignal(
                priority="takeover", current_tokens=0, budget_tokens=0,
                overflow_tokens=0,
            )
            target_end = len(compact_history)
            try:
                self._submit_compaction_request(
                    compact_history, signal, turn=0, target_end=target_end,
                    turn_id="",
                    preexisting_operation_id=new_op,
                )
            except Exception:
                # _submit_compaction_request's except already cleared
                # self._active_compaction_op; re-raise for observability.
                logger.exception(
                    "COMPACTION_TAKEOVER_SUBMIT_FAILED conv=%s new_op=%s",
                    conversation_id[:12], new_op,
                )
                # Do NOT re-raise — we still want to widen + return so the
                # current POST can passthrough. Next POST will reclaim the
                # empty new_op row as stale.

            widened = self.engine._store.widen_pending_raw_payload_entries(
                conversation_id=conversation_id,
                lifecycle_epoch=my_epoch,
                value=new_raw,
            )
            if not widened:
                return PhaseDecision(phase=phase, started_tagger=False)
            return PhaseDecision(phase="compacting", started_tagger=False)

        # Step 5.5: derive progress from canonical_turns SUM and transition
        # phase when the derived totals cross a boundary (total==done with
        # phase='init' → 'active'; total>done with phase in ('init','active')
        # → 'ingesting'). Steps 6-8 will be added by tasks A26-A29.
        snap = self.engine._store.read_progress_snapshot(conversation_id)
        total_ingestible = snap.total_ingestible
        done_ingestible = snap.done_ingestible

        # Post-resurrect the local ``phase`` variable still reflects the
        # pre-resurrect value ('deleted'); the DB phase is now 'init'. Reread
        # the effective phase from the snapshot before making decisions.
        if phase == "deleted":
            phase = snap.phase  # should now be 'init'

        if total_ingestible == done_ingestible:
            # Nothing untagged — no ingestion work to start.
            if phase == "init":
                # Empty conversation — transition init → active.
                self.engine.verify_epoch()  # fresh check before the write
                ok = self.engine._store.set_phase(
                    conversation_id=conversation_id,
                    lifecycle_epoch=my_epoch,
                    phase="active",
                )
                if not ok:
                    raise LifecycleEpochMismatch(
                        conversation_id=conversation_id,
                        expected=my_epoch,
                        observed=self.engine._store.get_lifecycle_epoch(
                            conversation_id,
                        ),
                    )
                self._publish_phase_transition("init", "active")
                phase = "active"
            return PhaseDecision(phase=phase, started_tagger=False)

        # total_ingestible > done_ingestible: there IS untagged work.
        if phase in ("init", "active"):
            old_phase = phase
            self.engine.verify_epoch()
            ok = self.engine._store.set_phase(
                conversation_id=conversation_id,
                lifecycle_epoch=my_epoch,
                phase="ingesting",
            )
            if not ok:
                raise LifecycleEpochMismatch(
                    conversation_id=conversation_id,
                    expected=my_epoch,
                    observed=self.engine._store.get_lifecycle_epoch(
                        conversation_id,
                    ),
                )
            self._publish_phase_transition(old_phase, "ingesting")
            phase = "ingesting"

        # Step 6 — reached only when phase == 'ingesting' AND total > done
        # (the total == done branch already returned from step 5.5).
        #
        # Step 6(a): ownership-free widening of the episode's raw payload
        # bound. Inserts a running episode with this worker as initial owner
        # if none exists; otherwise widens raw_payload_entries via MAX
        # without touching ownership.
        self.engine._store.upsert_ingestion_episode(
            conversation_id=conversation_id,
            lifecycle_epoch=my_epoch,
            worker_id=self._worker_id,
            raw_payload_entries=new_raw,
        )

        # Step 6(b): attempt to claim the ingestion lease. Succeeds iff the
        # caller already owns it or the current heartbeat is stale. Epoch
        # filter prevents a stale lifecycle from stealing a fresh lease.
        claimed = self.engine._store.claim_ingestion_lease(
            conversation_id=conversation_id,
            lifecycle_epoch=my_epoch,
            worker_id=self._worker_id,
            lease_ttl_s=INGESTION_LEASE_TTL_S,
        )

        if claimed:
            # Tagger dispatch is delegated to ``start_ingestion_if_needed``
            # (legacy ``_ingestion_thread`` running ``tag_turn`` with full
            # history). Do NOT spawn the per-row tagger here — two tagger
            # threads would race on the same canonical rows and the
            # ``TurnTagIndex``. ``started_tagger=True`` signals that this
            # call claimed the lease; the legacy path spawns the actual
            # worker thread downstream in the same request flow.
            return PhaseDecision(phase="ingesting", started_tagger=True)
        return PhaseDecision(phase="ingesting", started_tagger=False)

    def _another_worker_owns_lease(self, conversation_id: str) -> bool:
        """Defense-in-depth check: return True iff an active ingestion
        episode exists and is owned by a DIFFERENT worker than this one.

        This is belt-and-suspenders: the authoritative gate lives in
        ``server.py`` (which skips both legacy spawn paths when
        ``PhaseDecision.started_tagger`` is False). This helper handles
        callers that bypass the server-side gate — direct test harnesses,
        older entry points, and any future code that invokes the legacy
        spawners without routing through ``handle_prepare_payload``.

        A missing / failed snapshot read is treated as "don't block" — the
        server-side gate already covers the normal path, so spurious lock-out
        here would only break legacy fallback behaviour in edge cases. To
        keep the check robust in test harnesses that use ``MagicMock`` for
        the store, only a real ``ProgressSnapshot`` with a real
        ``ActiveEpisodeSnapshot`` triggers the block — a non-typed mock
        sentinel reads as "no active episode".
        """
        from ..core.progress_snapshot import (
            ActiveEpisodeSnapshot,
            ProgressSnapshot,
        )
        try:
            snap = self.engine._store.read_progress_snapshot(conversation_id)
        except Exception:
            return False
        if not isinstance(snap, ProgressSnapshot):
            return False
        active = snap.active_episode
        if not isinstance(active, ActiveEpisodeSnapshot):
            return False
        return active.owner_worker_id != self._worker_id

    def resume_pending_ingestion_if_needed(self) -> bool:
        """Resume indexing from durable canonical turns when completed turns outpace indexed turns.

        Multi-worker safety: this method can be called concurrently by
        different worker processes (each with its own ``ProxyState``) for
        the same conversation when all request paths point at the same
        worker pool. The server-side ``_owns_ingestion_lease`` gate is the
        primary guard, but callers outside that gate (tests, legacy
        entrypoints, edge cases where the server gate is bypassed) can
        still race here. We defend the spawn by ATOMICALLY claiming the
        ingestion lease via ``claim_ingestion_lease`` — if the claim
        fails (another worker owns a fresh lease) we back off without
        spawning. The claim also refreshes the heartbeat when the caller
        is already the owner, so repeated calls from the same worker stay
        safe.
        """
        conversation_id = self.engine.config.conversation_id
        if not self.has_pending_indexing():
            return False
        # Atomic lease claim. On success the caller owns the lease (either
        # took it over because the previous heartbeat was stale or was
        # already the owner). On failure another worker owns a live lease.
        lifecycle_epoch = int(
            getattr(self.engine._engine_state, "lifecycle_epoch", 1) or 1
        )
        try:
            claimed = self.engine._store.claim_ingestion_lease(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=self._worker_id,
                lease_ttl_s=INGESTION_LEASE_TTL_S,
            )
        except (AttributeError, NotImplementedError):
            # Store backend lacks the lease API (in-memory test stores,
            # legacy file backends). Fall through to the defense-in-depth
            # observer check so test harnesses with MagicMock stores keep
            # working.
            claimed = not self._another_worker_owns_lease(conversation_id)
        if not claimed:
            logger.info(
                "resume_pending_ingestion_if_needed: skipping spawn — "
                "could not claim ingestion lease "
                "(conv=%s, this=%s)",
                conversation_id[:12], self._worker_id,
            )
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
                    rows = self.engine._store.get_canonical_turn_rows(conversation_id, turn_numbers)
                except Exception:
                    logger.warning(
                        "Failed to load pending canonical turns for durable resume (conv=%s)",
                        conversation_id[:12],
                        exc_info=True,
                    )
                    return False
                for turn_number in turn_numbers:
                    row = rows.get(turn_number)
                    if row is None:
                        continue
                    pending_rows.append(
                        (
                            turn_number,
                            row.user_content,
                            row.assistant_content,
                            row.user_raw_content,
                            row.assistant_raw_content,
                        )
                    )

            if not pending_rows:
                return False

            messages: list[Message] = []
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
                messages.append(Message(role="user", content=user_content))
                messages.append(Message(role="assistant", content=assistant_content))
                expected_turn += 1

            if not messages:
                return False

            if self.metrics:
                self.metrics.clear_ingestion_events(conversation_id)

            self.engine._restored_pending_turns = []
            self._ingestion_progress = (
                baseline,
                max(total, baseline + self._history_turn_count(messages)),
            )
            self._transition_to(SessionState.INGESTING)
            self._spawn_ingestion_workers(
                history_messages=messages,
                existing_turns=baseline,
                total=total,
                tool_output_refs_by_turn=None,
                ingest_thread_name="vc-ingest-resume",
            )
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

    def extract_session_state(self):
        """Return the shared session snapshot, including UI-facing runtime fields.

        Ingestion progress is sourced from ``read_progress_snapshot`` (the
        DB-derived view over ``canonical_turns`` + ``ingestion_episode``) so
        the snapshot is always the single source of truth for ingestion
        phase and counts.  The legacy per-process ``_payload_ingestion_progress``
        tuple has been removed — no fabricated (0, total) counters.
        """
        snapshot = self.engine.extract_session_state()
        conversation_id = self.engine.config.conversation_id
        effective_state = self.session_state.value
        try:
            prog = self.engine._store.read_progress_snapshot(conversation_id)
        except Exception as exc:
            logger.warning(
                "extract_session_state: read_progress_snapshot failed for conv=%s: %s",
                conversation_id[:12], exc,
            )
            live_turn_count = 0
        else:
            live_turn_count = int(prog.total_ingestible or 0)
            if prog.phase == "ingesting" and effective_state == SessionState.ACTIVE.value:
                effective_state = SessionState.INGESTING.value
        history_message_count = max(
            self._raw_payload_entry_count,
            self._shared_history_message_count,
            live_turn_count,
        )
        snapshot.session_state = effective_state
        snapshot.live_turn_count = live_turn_count
        snapshot.history_message_count = history_message_count
        snapshot.last_payload_kb = float(self._last_payload_kb or 0.0)
        snapshot.last_payload_tokens = int(self._last_payload_tokens or 0)
        snapshot.raw_payload_entry_count = int(self._raw_payload_entry_count or 0)
        snapshot.ingestible_entry_count = int(self._ingestible_entry_count or 0)
        snapshot.skipped_payload_entry_count = int(self._skipped_payload_entry_count or 0)
        return snapshot

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

        # Ingestion phase + counts are DB-derived via read_progress_snapshot —
        # the process-local ``_payload_ingestion_progress`` tuple has been
        # removed so every consumer agrees with the canonical conversation row.
        _prog = engine._store.read_progress_snapshot(engine.config.conversation_id)
        _snap_phase = str(_prog.phase or "")
        _snap_done = int(_prog.done_ingestible or 0)
        _snap_total = int(_prog.total_ingestible or 0)

        snap = {
            "conversation_id": engine.config.conversation_id,
            "turn_count": _snap_total,
            "total_requests": self._total_requests,
            "compacted_prefix_messages": engine._engine_state.compacted_prefix_messages,
            "tag_count": len(idx.entries),
            "distinct_tags": len(all_tags),
            "active_tags": list(idx.get_active_tags(lookback=6)),
            "session_state": (
                SessionState.INGESTING.value
                if _snap_phase == "ingesting" and self.session_state == SessionState.ACTIVE
                else self.session_state.value
            ),
            "ingestion_progress": [_snap_done, _snap_total],
            "raw_payload_entry_count": self._raw_payload_entry_count,
            "ingestible_entry_count": self._ingestible_entry_count,
            "skipped_payload_entry_count": self._skipped_payload_entry_count,
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
        while True:
            future = self._pending_compact
            if future is None:
                return
            future.result()
            if self._pending_compact is future:
                self._pending_compact = None
                return

    def wait_for_complete(self) -> None:
        """Block until tag + compact both finish."""
        self.wait_for_tag()
        self.wait_for_compact()

    @staticmethod
    def _completed_turn_signature(
        history_snapshot: list[Message],
    ) -> tuple[int, str] | None:
        """Return the stable completed turn number and hash for a finished pair."""
        grouped = pair_messages_into_turns(list(history_snapshot))
        if not grouped:
            return None
        latest_turn = grouped[-1]
        if not any(msg.role == "assistant" for msg in latest_turn.messages):
            return None
        turn_number = len(grouped) - 1
        combined_text = " ".join(msg.content for msg in latest_turn.messages)
        message_hash = hashlib.sha256(combined_text.encode()).hexdigest()[:16]
        return turn_number, message_hash

    def _compaction_target_end(self, history: list[Message]) -> int:
        protected_recent_turns = 0
        try:
            protected_recent_turns = int(self.engine.config.monitor.protected_recent_turns)
        except (AttributeError, TypeError, ValueError):
            protected_recent_turns = 0
        protected_count = protected_recent_turns * 2
        return max(0, len(history) - protected_count)

    def _engine_state_int(self, field_name: str, default: int = 0) -> int:
        raw = getattr(getattr(self.engine, "_engine_state", None), field_name, default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _compaction_priority_rank(priority: str) -> int:
        return 2 if priority == "hard" else 1

    def _should_replace_compaction_request(
        self,
        existing: dict[str, object] | None,
        *,
        target_end: int,
        priority: str,
    ) -> bool:
        if existing is None:
            return True
        existing_target = int(existing.get("target_end", -1) or -1)
        if target_end > existing_target:
            return True
        if target_end < existing_target:
            return False
        existing_priority = str(existing.get("priority", "soft"))
        return self._compaction_priority_rank(priority) > self._compaction_priority_rank(existing_priority)

    def _submit_compaction_request(
        self,
        history: list[Message],
        signal: object,
        turn: int,
        target_end: int,
        turn_id: str = "",
        *,
        preexisting_operation_id: str | None = None,
    ) -> None:
        priority = str(getattr(signal, "priority", "soft") or "soft")
        with self._background_state_lock:
            self._active_compaction_target_end = target_end
        if preexisting_operation_id is not None:
            self._active_compaction_op = preexisting_operation_id
        try:
            self._pending_compact = self._compact_pool.submit(
                self._run_compact_wrapper,
                history,
                signal,
                turn,
                target_end,
                turn_id,
                preexisting_operation_id=preexisting_operation_id,
            )
        except Exception:
            if preexisting_operation_id is not None:
                self._active_compaction_op = None
            raise
        logger.info(
            "T%d compaction submitted target_end=%d priority=%s%s",
            turn,
            target_end,
            priority,
            f" preexisting_op={preexisting_operation_id}" if preexisting_operation_id else "",
        )

    def _queue_compaction(
        self,
        history: list[Message],
        signal: object,
        turn: int,
        turn_id: str = "",
        *,
        preexisting_operation_id: str | None = None,
    ) -> None:
        target_end = self._compaction_target_end(history)
        priority = str(getattr(signal, "priority", "soft") or "soft")

        with self._background_state_lock:
            pending_future = self._pending_compact
            if pending_future is not None and not pending_future.done():
                if target_end <= self._active_compaction_target_end:
                    logger.info(
                        "T%d compaction coalesced under active target_end=%d priority=%s",
                        turn,
                        self._active_compaction_target_end,
                        priority,
                    )
                    return
                if self._should_replace_compaction_request(
                    self._queued_compaction_request,
                    target_end=target_end,
                    priority=priority,
                ):
                    self._queued_compaction_request = {
                        "history": history,
                        "signal": signal,
                        "turn": turn,
                        "turn_id": turn_id,
                        "target_end": target_end,
                        "priority": priority,
                    }
                    logger.info(
                        "T%d compaction queued behind active run target_end=%d priority=%s",
                        turn,
                        target_end,
                        priority,
                    )
                else:
                    logger.info(
                        "T%d compaction request dropped as covered by newer queued target_end=%d priority=%s",
                        turn,
                        int(self._queued_compaction_request.get("target_end", -1)) if self._queued_compaction_request else -1,
                        str(self._queued_compaction_request.get("priority", "")) if self._queued_compaction_request else "",
                    )
                return

        self._submit_compaction_request(history, signal, turn, target_end, turn_id=turn_id,
                                        preexisting_operation_id=preexisting_operation_id)

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
        signature = self._completed_turn_signature(history_snapshot)
        if signature is None:
            logger.info("fire_turn_complete skipped (no completed pair)")
            return
        reserved_turn, message_hash = signature
        existing = self.engine._turn_tag_index.get_tags_for_logical_turn(reserved_turn)
        if existing is not None:
            if existing.message_hash != message_hash:
                logger.warning(
                    "fire_turn_complete skipped divergent duplicate for %s turn=%d indexed=%s new=%s",
                    self.engine.config.conversation_id[:12],
                    reserved_turn,
                    existing.message_hash,
                    message_hash,
                )
            else:
                logger.info(
                    "fire_turn_complete deduped for %s turn=%d (already indexed)",
                    self.engine.config.conversation_id[:12],
                    reserved_turn,
                )
            return
        with self._background_state_lock:
            queued_hash = self._queued_tag_turns.get(reserved_turn)
            if queued_hash == message_hash:
                logger.info(
                    "fire_turn_complete deduped for %s turn=%d (already queued)",
                    self.engine.config.conversation_id[:12],
                    reserved_turn,
                )
                return
            self._queued_tag_turns[reserved_turn] = message_hash
        try:
            self._pending_tag = self._pool.submit(
                self._run_tag_turn, history_snapshot, payload_tokens, turn_id, reserved_turn, message_hash,
            )
        except RuntimeError:
            with self._background_state_lock:
                if self._queued_tag_turns.get(reserved_turn) == message_hash:
                    self._queued_tag_turns.pop(reserved_turn, None)
            logger.info(
                "fire_turn_complete suppressed for shut down session %s",
                self.engine.config.conversation_id[:12],
            )

    def _run_tag_turn(
        self,
        history: list[Message],
        payload_tokens: int | None = None,
        turn_id: str = "",
        reserved_turn: int | None = None,
        message_hash: str = "",
    ) -> None:
        """Fast path: tag the turn, emit metrics, fire compaction if needed."""
        t0 = time.monotonic()
        signature = self._completed_turn_signature(history)
        turn = reserved_turn if reserved_turn is not None else (signature[0] if signature else len(self.engine._turn_tag_index.entries))
        turn_hash = message_hash or (signature[1] if signature else "")
        conversation_id = self.engine.config.conversation_id
        try:
            existing = self.engine._turn_tag_index.get_tags_for_logical_turn(turn)
            if not isinstance(existing, TurnTagEntry):
                existing = None
            if existing is not None:
                if turn_hash and existing.message_hash != turn_hash:
                    logger.warning(
                        "T%d TAG skipped divergent duplicate conversation=%s indexed=%s new=%s",
                        turn,
                        conversation_id[:12],
                        existing.message_hash,
                        turn_hash,
                    )
                else:
                    logger.info(
                        "T%d TAG skipped (already indexed) conversation=%s",
                        turn,
                        conversation_id[:12],
                    )
                return
            signal = self.engine.tag_turn(
                history,
                payload_tokens=payload_tokens,
                run_broad_split=False,
                turn_number=turn,
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
                "T%d tagged (%dms) conversation=%s compacted_prefix_messages=%d history=%d%s",
                turn, int(tag_ms), conversation_id[:12],
                self.engine._engine_state.compacted_prefix_messages,
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
                    turn_id=turn_id,
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
                        self._queue_compaction(history, signal, turn, turn_id=turn_id)
                    except RuntimeError:
                        logger.info(
                            "T%d compaction suppressed for shut down session %s",
                            turn, conversation_id[:12],
                        )

            self._queue_deferred_tag_split(history, turn)

        except Exception as e:
            logger.error("tag_turn error: %s", e, exc_info=True)
        finally:
            with self._background_state_lock:
                if self._queued_tag_turns.get(turn) == turn_hash:
                    self._queued_tag_turns.pop(turn, None)

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
        turn_id: str = "",
        *,
        preexisting_operation_id: str | None = None,
    ) -> None:
        """Background compaction — runs in _compact_pool, doesn't block next request.

        In addition to the legacy dict-based ``_update_compaction_state``
        mirror, this wires the DB-backed compaction lifecycle on
        ``ProxyState`` so downstream consumers (SSE, dashboards) see the
        ``conversations.phase`` flip to ``'compacting'``, a
        ``compaction_operation`` row get written, and
        ``PhaseTransitionEvent`` / ``CompactionProgressEvent`` fire on
        enter / each phase advance / exit. See ``enter_compaction``,
        ``advance_compaction_phase``, and ``exit_compaction`` on
        ``ProxyState`` for the underlying primitives.
        """
        conversation_id = self.engine.config.conversation_id
        if preexisting_operation_id is not None:
            # Takeover path: the caller pre-inserted the compaction_operation
            # row (e.g. via cleanup_abandoned_compaction).  Use the supplied
            # id; do NOT call enter_compaction() which would attempt a second
            # INSERT and could race or duplicate.
            operation_id = preexisting_operation_id
        else:
            operation_id = uuid.uuid4().hex[:12]

        # P1 fix: set BEFORE enter_compaction so same-worker concurrent POSTs
        # see this as "our running compaction" via the takeover predicate
        # ``self._active_compaction_op == claim.prev_operation_id``.
        # Without this, the normal-path (no preexisting_operation_id) left
        # _active_compaction_op=None and the takeover logic would classify our
        # own live op as abandoned, calling cleanup_abandoned_compaction against
        # it and raising CompactionLeaseLost. Covers:
        #   - takeover path (wrapper calls _run_compact with preexisting_id)
        #   - normal async path (wrapper calls _run_compact, generates locally)
        #   - synchronous direct path (_compact_after_ingestion → _run_compact)
        self._active_compaction_op = operation_id

        compaction_started = time.monotonic()
        # Mutable closure cell for the last phase index we advanced the
        # DB-backed compaction_operation row to. Prevents redundant
        # ``advance_compaction_phase`` writes when the pipeline emits
        # multiple progress callbacks inside a single phase.
        last_advanced_phase_index = -1
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
            nonlocal last_advanced_phase_index
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
            # Mirror the pipeline's phase_name onto the DB-backed
            # compaction_operation row. Only advance when the named
            # phase is one we planned for AND the plan index moved
            # forward, so a noisy callback stream collapses into a
            # single DB write + event per phase transition.
            phase_name_kwarg = str(evt.get("phase_name", phase) or "")
            plan_idx = _COMPACT_PHASE_INDEX.get(phase_name_kwarg, -1)
            if plan_idx > last_advanced_phase_index:
                self.advance_compaction_phase(
                    phase_index=plan_idx,
                    phase_name=phase_name_kwarg,
                )
                last_advanced_phase_index = plan_idx
            if self.metrics:
                self.metrics.record(evt)

        # Code-review P1 C1: acquire INSIDE the try so the finally-block's
        # release is always reachable, even if Thread() or .start() raises
        # (e.g. RuntimeError under thread exhaustion).  Without this, any
        # exception between acquire() and the outer try would leave
        # _compaction_lock permanently held, causing every subsequent
        # _run_compact to skip via the blocking=False fast-path.
        acquired = False
        try:
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
            acquired = True

            # Code-review P2.3: reset cancel state from any prior compaction on
            # this ProxyState so a fresh run (e.g. a post-takeover retry) does
            # not immediately raise InterruptedError against a stale cancel flag
            # set by a prior sidecar (e.g. from a heartbeat refresh failure).
            self._compaction_cancelled.clear()

            # Enter the DB-backed compaction lifecycle. ``enter_compaction``
            # is itself epoch-guarded and silently no-ops on mismatch
            # (leaving no ``compaction_operation`` row), which is why we
            # observe whether the row was created via the progress snapshot
            # before deciding to emit a terminal ``exit_compaction``.
            # Otherwise a concurrent delete+resurrect that flipped our
            # epoch mid-enter would leave us calling
            # ``drain_compaction_exit`` against a phase we never owned.
            #
            # Takeover path: when preexisting_operation_id is set the caller
            # has ALREADY inserted the compaction_operation row.  Skip
            # enter_compaction() entirely and treat phase_index 0 as the
            # baseline so advance_compaction_phase picks up from there.
            if preexisting_operation_id is not None:
                # Takeover path: the pre-inserted row is our responsibility to
                # close, so we are unconditionally in the lifecycle — no snapshot
                # probe needed.
                last_advanced_phase_index = 0
                entered_lifecycle = True
            else:
                try:
                    self.enter_compaction(
                        phase_count=len(_COMPACT_PHASE_PLAN),
                        initial_phase_name=_COMPACT_PHASE_PLAN[0],
                        operation_id=operation_id,
                    )
                    last_advanced_phase_index = 0
                except Exception:
                    logger.warning(
                        "enter_compaction failed for %s — continuing legacy path only",
                        conversation_id[:12],
                        exc_info=True,
                    )
                entered_lifecycle = False
                try:
                    snap_after_enter = self.engine._store.read_progress_snapshot(
                        conversation_id,
                    )
                    entered_lifecycle = snap_after_enter.active_compaction is not None
                except Exception:
                    entered_lifecycle = False

            # Spawn the heartbeat sidecar.  It refreshes the compaction_operation
            # row every INGESTION_LEASE_TTL_S / 2 seconds while the compactor
            # thread runs.  On a failed refresh (epoch mismatch, wrong owner, or
            # status != 'running') it sets _compaction_cancelled so the progress
            # callback raises InterruptedError and the compactor aborts cleanly.
            # The sidecar is stopped (via _compaction_heartbeat_stop) in the
            # finally block below — regardless of success, exception, or
            # CompactionLeaseLost.
            epoch_for_sidecar = int(self.engine._engine_state.lifecycle_epoch)
            _compaction_heartbeat_stop = threading.Event()
            _heartbeat_sidecar_thread = threading.Thread(
                target=self._run_compaction_heartbeat_sidecar,
                args=(conversation_id, epoch_for_sidecar, operation_id,
                      _compaction_heartbeat_stop),
                daemon=True,
                name="vc-compact-heartbeat",
            )
            _heartbeat_sidecar_thread.start()

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
                    compact_if_needed = self.engine.compact_if_needed
                    compact_target = getattr(compact_if_needed, "side_effect", None)
                    if not callable(compact_target):
                        compact_target = compact_if_needed
                    supports_turn_id = True
                    try:
                        signature = inspect.signature(compact_target)
                        supports_turn_id = (
                            "turn_id" in signature.parameters
                            or any(
                                param.kind == inspect.Parameter.VAR_KEYWORD
                                for param in signature.parameters.values()
                            )
                        )
                    except (TypeError, ValueError):
                        supports_turn_id = True

                    if supports_turn_id:
                        report = compact_if_needed(
                            history, signal, progress_callback=_compact_progress,
                            turn_id=turn_id, operation_id=operation_id,
                        )
                    else:
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
                                "compacted_prefix_messages": self.engine._engine_state.compacted_prefix_messages,
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
                    if entered_lifecycle:
                        try:
                            self.exit_compaction(success=True)
                        except Exception:
                            logger.warning(
                                "exit_compaction(success=True) failed for %s",
                                conversation_id[:12],
                                exc_info=True,
                            )

                except CompactionLeaseLost as e:
                    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                    logger.info(
                        "COMPACTION_WRITE_REJECTED op=%s site=%s elapsed_ms=%.1f",
                        e.operation_id, e.write_site, elapsed_ms,
                    )
                    self._update_compaction_state(
                        operation_id=operation_id,
                        status="cancelled",
                        phase="cancelled",
                        phase_name="cancelled",
                        overall_percent=None,
                        elapsed_ms=elapsed_ms,
                        error=f"lease_lost:{e.write_site}",
                        phase_detail="compaction aborted on lease loss",
                    )
                    if entered_lifecycle:
                        try:
                            self.exit_compaction(success=False, error_message=str(e))
                        except Exception:
                            logger.warning(
                                "exit_compaction(success=False) failed after lease-lost for conv=%s",
                                conversation_id[:12], exc_info=True,
                            )
                    return

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
                    if entered_lifecycle:
                        try:
                            self.exit_compaction(
                                success=False, error_message=str(e),
                            )
                        except Exception:
                            logger.warning(
                                "exit_compaction(success=False) failed for %s",
                                conversation_id[:12],
                                exc_info=True,
                            )
            finally:
                # Signal and join the heartbeat sidecar before releasing the
                # lock so it cannot fire a spurious cancel after the op ends.
                _compaction_heartbeat_stop.set()
                _heartbeat_sidecar_thread.join(timeout=5)
        finally:
            # Outer finally: always release the lock if we acquired it.
            # This fires even if Thread() or .start() raised before the inner
            # try was entered (code-review P1 C1 fix).
            if acquired:
                self._compaction_lock.release()
            # Clear _active_compaction_op regardless of success/failure so no
            # stale op_id leaks to the next compaction on this ProxyState.
            # Covers the synchronous direct path (_compact_after_ingestion →
            # _run_compact without _run_compact_wrapper). The wrapper's own
            # finally also clears it for the async path — double-clear is safe.
            self._active_compaction_op = None

    def _run_compact_wrapper(
        self,
        history: list[Message],
        signal: object,
        turn: int,
        target_end: int,
        turn_id: str = "",
        *,
        preexisting_operation_id: str | None = None,
    ) -> None:
        try:
            self._run_compact(history, signal, turn, turn_id=turn_id,
                              preexisting_operation_id=preexisting_operation_id)
        finally:
            # Always clear the active op so the takeover predicate is unblocked.
            self._active_compaction_op = None
            follow_up: dict[str, object] | None = None
            with self._background_state_lock:
                self._last_completed_compaction_target_end = max(
                    self._last_completed_compaction_target_end,
                    target_end,
                )
                self._active_compaction_target_end = -1
                queued = self._queued_compaction_request
                self._queued_compaction_request = None
                current_watermark = self._engine_state_int("compacted_prefix_messages", 0)
                if queued is not None and int(queued.get("target_end", -1) or -1) > current_watermark:
                    follow_up = queued
            if follow_up is not None:
                self._submit_compaction_request(
                    follow_up["history"],
                    follow_up["signal"],
                    int(follow_up["turn"]),
                    int(follow_up["target_end"]),
                    turn_id=str(follow_up.get("turn_id", "") or ""),
                )

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
            watermark = self.engine._engine_state.compacted_prefix_messages
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

    def reconcile_history_bootstrap(self, history_messages: list[Message]) -> bool:
        """Finalize a restored session once the first post-restart history arrives."""
        conversation_id = self.engine.config.conversation_id
        if conversation_id in self._ingested_conversations and not self._restore_readiness_pending:
            return True
        if self.has_pending_indexing():
            self._restore_readiness_pending = False
            return False
        if not self._can_activate_from_persisted_state(history_messages):
            self._restore_readiness_pending = False
            return False
        self._ingested_conversations.add(conversation_id)
        self._restore_readiness_pending = False
        if history_messages:
            self._record_ingestion_watermark(history_messages, conversation_id)
        return True

    def _check_history_widening(self, history_messages: list[Message], conversation_id: str) -> bool:
        """Detect if history prefix shifted (widening) and trigger full re-ingest.

        Returns True if widening was detected and state was cleared for re-ingestion.
        """
        import hashlib
        if not history_messages or conversation_id not in self._ingested_first_hash:
            return False

        first_turn_text = self._combined_turn_text(history_messages, 0)
        if not first_turn_text:
            return False
        new_first_hash = hashlib.sha256(first_turn_text.encode()).hexdigest()[:16]
        old_first_hash = self._ingested_first_hash.get(conversation_id, "")
        if new_first_hash == old_first_hash:
            return False

        new_turns = self._history_turn_count(history_messages)
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
        self._raw_payload_entry_count = 0
        self._ingestible_entry_count = 0
        self._skipped_payload_entry_count = 0
        self._last_non_virtualizable_floor = 0
        self._shared_live_turn_count = 0
        self._shared_history_message_count = 0
        self._inbound_payload_token_cache = None
        self._outbound_payload_token_cache = None
        self._restore_readiness_pending = False
        self._restore_readiness_signature = None
        self._chain_snapshot_cache = {
            "loaded": False,
            "refs_by_turn": {},
            "recovery_loaded": False,
            "recovery_manifest": [],
        }
        with self._background_state_lock:
            self._queued_tag_turns = {}
            self._queued_compaction_request = None
            self._active_compaction_target_end = -1
            self._last_completed_compaction_target_end = -1
        self._active_compaction_op = None
        self._total_requests = 0
        self._last_model = ""

    def _drain_background_work(self) -> None:
        """Wait for queued tag/compaction work without propagating old failures."""
        for attr in ("_pending_tag", "_pending_compact", "_pending_split"):
            while True:
                future = getattr(self, attr, None)
                if future is None:
                    break
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
                    if getattr(self, attr, None) is future:
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
        with self._background_state_lock:
            self._queued_tag_turns = {}
            self._queued_compaction_request = None
            self._active_compaction_target_end = -1
        self._active_compaction_op = None

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
            # If the worker is gone the sidecar will also exit on its
            # next loop iteration (is_alive False → return). Best-effort
            # join then clear so we don't leak a zombie handle.
            heartbeat = self._heartbeat_thread
            if heartbeat is not None and heartbeat.is_alive():
                heartbeat.join(timeout=timeout_s)
            self._heartbeat_thread = None
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
        # Drain the heartbeat sidecar too — setting cancel above already
        # woke it up, so the join should be quick.
        heartbeat = self._heartbeat_thread
        if heartbeat is not None and heartbeat.is_alive():
            heartbeat.join(timeout=timeout_s)
            if heartbeat.is_alive():
                logger.warning(
                    "Heartbeat sidecar did not stop within %.1fs for conv=%s",
                    timeout_s,
                    self.engine.config.conversation_id[:12],
                )
        self._heartbeat_thread = None
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
        """Refresh the derived compaction watermark from durable canonical turns.

        ``compacted_prefix_messages`` still drives status reporting and sliding-window
        history assembly, so it must reflect the actual compacted canonical
        prefix rather than whatever happens to be in memory right now.
        """
        try:
            rows = list(
                self.engine._store.get_all_canonical_turns(
                    self.engine.config.conversation_id,
                )
            )
            paired_rows = self.engine._group_canonical_rows_into_pairs(rows)
            new_wm, _ = self.engine._canonical_prefix_watermark(paired_rows)
            old_wm = int(self.engine._engine_state.compacted_prefix_messages)
            if new_wm != old_wm:
                self.engine._engine_state.compacted_prefix_messages = new_wm
                logger.info(
                    "Compaction watermark refreshed from canonical rows: %d -> %d "
                    "(paired_turns=%d)",
                    old_wm,
                    new_wm,
                    len(paired_rows),
                )
            self.engine._engine_state.flushed_prefix_messages = min(
                int(self.engine._engine_state.flushed_prefix_messages or 0),
                int(self.engine._engine_state.compacted_prefix_messages or 0),
            )
        except (TypeError, ValueError, AttributeError):
            pass

    def _record_ingestion_watermark(self, history_messages: list[Message], conversation_id: str) -> None:
        if history_messages:
            first_turn_text = self._combined_turn_text(history_messages, 0)
            if first_turn_text:
                self._ingested_first_hash[conversation_id] = (
                    hashlib.sha256(first_turn_text.encode()).hexdigest()[:16]
                )
        self._ingested_turn_count[conversation_id] = self._history_turn_count(history_messages)

    def ingest_if_needed(
        self,
        history_messages: list[Message],
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> None:
        """Bootstrap TurnTagIndex from pre-existing history (once per session).

        Double-checked locking: fast path skips the lock entirely.
        """
        conversation_id = self.engine.config.conversation_id
        if conversation_id in self._ingested_conversations and not self.has_pending_indexing():
            # Check for history widening even after ingestion is "done"
            self._check_history_widening(history_messages, conversation_id)
            if conversation_id in self._ingested_conversations:
                return  # No widening detected
        with self._ingestion_lock:
            if conversation_id in self._ingested_conversations:
                return
            t0 = time.monotonic()
            if tool_output_refs_by_turn is None:
                turns = self.engine.ingest_history(history_messages)
            else:
                turns = self.engine.ingest_history(
                    history_messages,
                    tool_output_refs_by_turn=tool_output_refs_by_turn,
                )
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            self._ingested_conversations.add(conversation_id)
            self._advance_compaction_watermark()
            self._record_ingestion_watermark(history_messages, conversation_id)

            logger.info(
                "INGEST %d turns in %dms (conversation=%s)",
                turns, int(elapsed_ms), conversation_id[:12],
            )

            if self.metrics:
                # Emit per-turn events so the dashboard grid shows history
                baseline_history_tokens = 0
                grouped = self._group_history_messages(history_messages)
                for turn_num, pair in enumerate(grouped):
                    entry = self.engine._turn_tag_index.get_tags_for_logical_turn(
                        turn_num,
                    )
                    if not pair.messages:
                        continue
                    raw_content = pair.messages[0].content
                    preview = _strip_envelope(raw_content)[:200]
                    turn_chars = sum(len(msg.content) for msg in pair.messages)
                    tpt = turn_chars // 4
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
                    "turns_received": len(grouped),
                    "elapsed_ms": elapsed_ms,
                    "conversation_id": conversation_id,
                    "baseline_history_tokens": baseline_history_tokens,
                })

    # ------------------------------------------------------------------
    # Non-blocking ingestion (background thread)
    # ------------------------------------------------------------------

    def start_ingestion_if_needed(
        self,
        history_messages: list[Message],
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
            self._check_history_widening(history_messages, conversation_id)
            if conversation_id in self._ingested_conversations:
                return
        # Defense-in-depth: refuse to spawn the legacy thread if another
        # worker holds the ingestion lease. The server-side gate in
        # ``prepare_payload`` already handles the primary path; this catches
        # direct callers that bypass that gate.
        if self._another_worker_owns_lease(conversation_id):
            logger.info(
                "start_ingestion_if_needed: skipping spawn — "
                "active ingestion episode is owned by another worker "
                "(conv=%s, this=%s)",
                conversation_id[:12], self._worker_id,
            )
            return
        with self._ingestion_lock:
            if conversation_id in self._ingested_conversations:
                return

            needed_turns = self._history_turn_count(history_messages)
            logger.info(
                "INGEST_ENTRY conversation=%s turns=%d index_size=%d thread_alive=%s",
                conversation_id[:12], needed_turns,
                self._indexed_turn_count(),
                self._ingestion_thread is not None and self._ingestion_thread.is_alive(),
            )

            # Clear stale ingestion events from any previous run so the
            # dashboard shows fresh progress for this ingestion.
            if self.metrics:
                self.metrics.clear_ingestion_events(conversation_id)

            if not history_messages:
                self._ingested_conversations.add(conversation_id)
                return

            # Skip if persisted TurnTagIndex already covers history
            existing_turns = self._indexed_turn_count()
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
                history_messages = self._slice_messages_from_turn(history_messages, existing_turns)
                if tool_output_refs_by_turn is not None:
                    tool_output_refs_by_turn = {
                        turn_idx - existing_turns: refs
                        for turn_idx, refs in tool_output_refs_by_turn.items()
                        if turn_idx >= existing_turns
                    }
                if not history_messages:
                    # Bug 1 fix (P1): do NOT locally mark the conversation
                    # as ingested without a DB-authoritative completion.
                    # The legacy-thread cancel policy forbids the old
                    # thread from calling ``complete_ingestion_episode``
                    # on cancel, so the DB episode may still be 'running'
                    # and the phase still 'ingesting'. Delegate to the
                    # shared helper; if the DB refuses (stale epoch,
                    # untagged rows, lease stolen), leave the worker in
                    # INGESTING and return without any local completion.
                    if self._finalize_legacy_ingestion(conversation_id):
                        self._transition_to(SessionState.ACTIVE)
                        self._compact_after_ingestion(history_messages)
                    return

            # ---- PROXY-013: cancel-and-resume if already running ----
            if (
                self._ingestion_thread is not None
                and self._ingestion_thread.is_alive()
            ):
                done, total = self._ingestion_progress
                logger.info(
                    "Cancelling running ingestion at turn %d/%d "
                    "(new request has %d turns)",
                    done, total, needed_turns,
                )
                self._ingestion_cancel.set()
                self._ingestion_thread.join(timeout=5.0)
                if self._ingestion_thread.is_alive():
                    logger.warning("Old ingestion thread did not exit in 5s")
                # Stop the old heartbeat sidecar too — it is bound to
                # the old ``_ingestion_cancel`` / thread pair and will
                # exit once it sees cancel set or the worker not alive.
                old_heartbeat = self._heartbeat_thread
                if old_heartbeat is not None and old_heartbeat.is_alive():
                    old_heartbeat.join(timeout=5.0)
                    if old_heartbeat.is_alive():
                        logger.warning("Old heartbeat sidecar did not exit in 5s")
                self._heartbeat_thread = None
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
                self._verify_handoff_hash(history_messages, existing_turns)

                # Slice to remaining turns only
                history_messages = self._slice_messages_from_turn(history_messages, existing_turns)
                if tool_output_refs_by_turn is not None:
                    tool_output_refs_by_turn = {
                        turn_idx - existing_turns: refs
                        for turn_idx, refs in tool_output_refs_by_turn.items()
                        if turn_idx >= existing_turns
                    }
                if not history_messages:
                    # Bug 1 fix (P1): the cancelled old thread did NOT
                    # call ``complete_ingestion_episode`` (per cancel
                    # policy — the finally block skips finalisation on
                    # cancel). If we short-circuit here with only local
                    # completion, the DB episode stays 'running' and
                    # phase stays 'ingesting' while the worker thinks
                    # it's done — classic split-brain. Delegate to the
                    # shared helper; if the DB refuses (stale epoch,
                    # untagged rows, lease stolen), leave the worker in
                    # INGESTING and return without local completion.
                    if self._finalize_legacy_ingestion(conversation_id):
                        self._transition_to(SessionState.ACTIVE)
                        self._compact_after_ingestion(history_messages)
                    return
                needed_turns = self._history_turn_count(history_messages) + existing_turns

            total = needed_turns
            self._ingestion_progress = (existing_turns, total)

            # Capture initial snapshot once (first ingestion start only)
            if self._initial_turns is None:
                self._initial_turns = existing_turns
                self._initial_tag_count = len(self.engine._turn_tag_index.entries)

            self._transition_to(SessionState.INGESTING)

            self._spawn_ingestion_workers(
                history_messages=list(history_messages),
                existing_turns=existing_turns,
                total=total,
                tool_output_refs_by_turn=tool_output_refs_by_turn,
                ingest_thread_name="vc-ingest",
            )

    def _spawn_ingestion_workers(
        self,
        *,
        history_messages: list[Message],
        existing_turns: int,
        total: int,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
        ingest_thread_name: str,
    ) -> None:
        # Heartbeat sidecar must start BEFORE the ingestion worker so its
        # first refresh is scheduled for ``INGESTION_LEASE_TTL_S / 2`` seconds
        # from now — independent of whether the worker is mid-turn or between
        # turns. A single long-running tagging turn (>TTL) would otherwise let
        # the lease expire before the next turn-completion callback fires.
        conversation_id = self.engine.config.conversation_id
        heartbeat_epoch = int(self.engine._engine_state.lifecycle_epoch)
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat_sidecar,
            args=(conversation_id, heartbeat_epoch),
            daemon=True,
            name="vc-ingest-heartbeat",
        )
        self._ingestion_thread = threading.Thread(
            target=self._run_ingestion_with_catchup,
            args=(history_messages, existing_turns, total, tool_output_refs_by_turn),
            daemon=True,
            name=ingest_thread_name,
        )
        self._heartbeat_thread.start()
        self._ingestion_thread.start()

    def _verify_handoff_hash(
        self, new_messages: list[Message], handoff_turn: int,
    ) -> None:
        """Verify the last tagged turn matches the same content in new history.

        Logs a warning if the hash doesn't match — indicates potential data
        loss or history divergence between requests.
        """
        import hashlib as _hl
        if handoff_turn <= 0:
            return
        prev_turn = handoff_turn - 1
        entry = self.engine._turn_tag_index.get_tags_for_logical_turn(prev_turn)
        if entry is None:
            return
        grouped = self._group_history_messages(new_messages)
        if prev_turn >= len(grouped):
            logger.warning(
                "Handoff verification: turn %d not in new history "
                "(new history has %d turns) — potential data loss",
                prev_turn, len(grouped),
            )
            return
        combined = " ".join(msg.content for msg in grouped[prev_turn].messages)
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

    def _finalize_legacy_ingestion(self, conversation_id: str) -> bool:
        """Shared DB-authoritative finalization for the legacy ingestion thread.

        Performs the canonical end-of-ingestion sequence:

          1. ``complete_ingestion_episode`` — atomic epoch-guarded write.
             Returns False if the lease was stolen, the epoch moved, or
             untagged rows remain.
          2. ``set_phase`` — epoch-guarded phase flip to ``'active'``.
             Returns False on stale-epoch race between step 1 and step 2.
          3. ``_publish_phase_transition`` — fires only on a successful
             phase flip.
          4. Local state advance — add to ``_ingested_conversations`` and
             call ``_advance_compaction_watermark``.

        Returns:
            True iff both DB ops succeeded and local state has been
            advanced. Callers should treat this as "it is safe to
            transition SessionState to ACTIVE, run post-ingestion
            compaction, and record the ingestion watermark". When False,
            the episode stays ``'running'`` so the next worker can pick
            it up; the caller MUST leave the local worker in INGESTING
            and skip all post-ingestion side effects.
        """
        try:
            epoch = int(self.engine._engine_state.lifecycle_epoch)
            episode_completed = self.engine._store.complete_ingestion_episode(
                conversation_id=conversation_id,
                lifecycle_epoch=epoch,
                worker_id=self._worker_id,
            )
        except Exception:
            logger.warning(
                "legacy ingestion: complete_ingestion_episode raised for "
                "conv=%s — leaving episode 'running' for next worker",
                conversation_id[:12],
                exc_info=True,
            )
            return False
        if not episode_completed:
            logger.info(
                "legacy ingestion: episode not completable for conv=%s — "
                "deferring to next worker",
                conversation_id[:12],
            )
            return False
        try:
            phase_ok = self.engine._store.set_phase(
                conversation_id=conversation_id,
                lifecycle_epoch=epoch,
                phase="active",
            )
        except Exception:
            logger.warning(
                "legacy ingestion: set_phase raised after successful "
                "episode completion for conv=%s — not publishing event",
                conversation_id[:12],
                exc_info=True,
            )
            return False
        if not phase_ok:
            logger.info(
                "legacy ingestion: set_phase('active') returned False for "
                "conv=%s — epoch stale; deferring",
                conversation_id[:12],
            )
            return False
        # Both DB ops succeeded — advance local state.
        self._ingested_conversations.add(conversation_id)
        self._advance_compaction_watermark()
        try:
            self._publish_phase_transition("ingesting", "active")
        except Exception:
            logger.warning(
                "_publish_phase_transition raised for conv=%s",
                conversation_id[:12],
                exc_info=True,
            )
        return True

    def _run_heartbeat_sidecar(self, conversation_id: str, epoch: int) -> None:
        """Refresh ingestion lease every ``INGESTION_LEASE_TTL_S / 2``
        seconds while ``_ingestion_thread`` is alive.

        Sidecar for the legacy ingestion thread. A single long-running
        tagging turn (>TTL) would otherwise let the lease expire before
        the turn-completion callback could refresh it. Running refresh
        from a dedicated thread decouples cadence from turn completion.

        Exits when:
          * ``_ingestion_cancel`` becomes set (cooperative stop),
          * ``_ingestion_thread`` is None or not alive (worker finished),
          * ``refresh_ingestion_heartbeat`` returns False (lease stolen
            or stale epoch) — in which case we set
            ``_ingestion_cancel`` to signal the worker to bail.

        Uses ``threading.Event.wait(timeout=interval)`` rather than
        ``time.sleep`` so a cancel signal interrupts the wait promptly.
        """
        interval = INGESTION_LEASE_TTL_S / 2
        while True:
            # Wait up to ``interval`` seconds, but wake early if cancel
            # is set. Event.wait returns True on set, False on timeout.
            if self._ingestion_cancel.wait(timeout=interval):
                return
            # Worker gone → lease no longer this sidecar's concern.
            worker = self._ingestion_thread
            if worker is None or not worker.is_alive():
                return
            try:
                ok = self.engine._store.refresh_ingestion_heartbeat(
                    conversation_id=conversation_id,
                    lifecycle_epoch=epoch,
                    worker_id=self._worker_id,
                )
            except Exception:
                logger.warning(
                    "heartbeat sidecar: refresh exception for conv=%s — "
                    "retrying next tick",
                    conversation_id[:12],
                    exc_info=True,
                )
                continue
            if not ok:
                logger.info(
                    "heartbeat sidecar: refresh rejected for conv=%s "
                    "(worker=%s) — lease lost; cancelling ingestion",
                    conversation_id[:12], self._worker_id,
                )
                self._ingestion_cancel.set()
                return

    def _run_compaction_heartbeat_sidecar(
        self,
        conversation_id: str,
        lifecycle_epoch: int,
        operation_id: str,
        stop_event: threading.Event,
    ) -> None:
        """Refresh compaction_operation.heartbeat_ts every TTL/2 while the
        compactor thread is alive. On failed refresh (operation_id /
        lifecycle_epoch / owner mismatch, status != 'running'), set the
        compaction cancel event so the compactor aborts cleanly.

        Exits when:
          * ``stop_event`` becomes set (cooperative stop, set in
            ``_run_compact``'s finally block when the compactor exits),
          * ``refresh_compaction_heartbeat`` returns False (lease stolen or
            stale epoch) — in which case we also set ``_compaction_cancelled``
            to signal the compactor to abort via the progress-callback path.

        Uses ``threading.Event.wait(timeout=interval)`` rather than
        ``time.sleep`` so the stop signal interrupts the wait promptly.
        """
        interval = INGESTION_LEASE_TTL_S / 2
        while not stop_event.wait(timeout=interval):
            try:
                ok = self.engine._store.refresh_compaction_heartbeat(
                    conversation_id=conversation_id,
                    lifecycle_epoch=lifecycle_epoch,
                    worker_id=self._worker_id,
                    operation_id=operation_id,
                )
            except Exception:
                logger.warning(
                    "compaction heartbeat sidecar: refresh raised for conv=%s "
                    "op=%s; retrying next tick",
                    conversation_id[:12], operation_id[:8],
                    exc_info=True,
                )
                continue
            if not ok:
                logger.info(
                    "compaction heartbeat sidecar: refresh rejected for conv=%s "
                    "op=%s (lease lost or epoch bumped) — signalling cancel",
                    conversation_id[:12], operation_id[:8],
                )
                self._compaction_cancelled.set()
                return

    def _run_ingestion_with_catchup(
        self,
        initial_messages: list[Message],
        baseline: int = 0,
        cumulative_total: int = 0,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> None:
        """Background thread: ingest initial message history, then catch up any gap.

        Architecture: two tagger paths run in sequence.

        1. Legacy pair-based tagger (``_ingest_messages_with_progress``) walks
           payload pairs and aligns them against existing canonical rows via
           strict role-shape matching. It can fail on a conversation whose
           canonical_turns table has orphan halves or half-tagged turn_groups
           left by prior crashes — strict alignment cannot map payload pairs
           to a messy DB.
        2. Row-based DB sweep (``_tagger_run``) is the safety-net. It tags
           untagged canonical rows directly from their stored
           ``user_content``/``assistant_content`` with no payload alignment
           required. It tolerates orphan halves and is the single
           authoritative "drain the untagged queue" step.

        The sweep runs unconditionally after the legacy path, regardless of
        whether the legacy path succeeded or errored, unless the lease was
        lost (``_IngestionCancelled``) or the conversation was deleted
        (``StaleConversationWriteError``). Without this, a legacy strict
        alignment failure would leave the episode wedged in INGESTING with
        no worker making progress until the next POST — and that next POST
        hits the same legacy failure.
        """
        conversation_id = self.engine.config.conversation_id
        cancelled = False
        completed_cleanly = False
        try:
            try:
                self._ingest_messages_with_progress(
                    initial_messages,
                    baseline=baseline,
                    cumulative_total=cumulative_total or None,
                    tool_output_refs_by_turn=tool_output_refs_by_turn,
                )

                for _ in range(10):
                    if self._ingestion_cancel.is_set():
                        break
                    latest = self._latest_body
                    if latest is None:
                        break
                    latest_messages = self._completed_history_messages(_extract_ingestible_messages(latest))
                    needed = self._history_turn_count(latest_messages)
                    have = len(self.engine._turn_tag_index.entries)
                    if needed <= have:
                        break
                    gap_messages = self._slice_messages_from_turn(latest_messages, have)
                    if not gap_messages:
                        break
                    logger.info(
                        "Ingestion catch-up: %d gap turns (have=%d, need=%d)",
                        self._history_turn_count(gap_messages), have, needed,
                    )
                    self._ingestion_progress = (have, needed)
                    self._ingest_messages_with_progress(gap_messages, baseline=have, cumulative_total=needed)

                if self._ingestion_cancel.is_set():
                    cancelled = True

            except _IngestionCancelled as e:
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
                logger.error(
                    "Legacy pair-based ingestion errored (falling through "
                    "to row-based DB sweep): %s",
                    e,
                    exc_info=True,
                )

            sweep_drained = False
            if not cancelled:
                try:
                    sweep_drained = bool(self._tagger_run())
                    if sweep_drained:
                        completed_cleanly = True
                except LifecycleEpochMismatch:
                    logger.info(
                        "Row-based DB sweep exited on lifecycle epoch "
                        "mismatch for conv=%s",
                        conversation_id[:12],
                    )
                    cancelled = True
                    completed_cleanly = False
                except StaleConversationWriteError as e:
                    cancelled = True
                    logger.info(
                        "Row-based DB sweep abandoned for deleted/stale "
                        "conversation %s: %s",
                        conversation_id[:12], e,
                    )
                except Exception:
                    logger.exception(
                        "Row-based DB sweep errored for conv=%s; "
                        "leaving episode INGESTING for next worker",
                        conversation_id[:12],
                    )
        finally:
            if not cancelled:
                # Finalization: when the row-based sweep drained the
                # queue it already wrote episode-completion + phase flip
                # + published the transition event inline. In that case
                # we skip ``_finalize_legacy_ingestion`` (which would
                # re-run those same DB writes — redundant and would
                # double-publish) but still need the caller-side
                # bookkeeping the helper normally owns:
                # ``_ingested_conversations`` membership and
                # ``_advance_compaction_watermark``. Post-ingestion side
                # effects (watermark recording, compaction check,
                # SessionState transition) fire from either path.
                #
                # ``sweep_drained=False`` + ``completed_cleanly=True``
                # shouldn't normally happen (sweep drained sets clean),
                # but the explicit ``and not sweep_drained`` guard makes
                # the single-finalization invariant clear.
                finalized = False
                if completed_cleanly and not sweep_drained:
                    finalized = self._finalize_legacy_ingestion(conversation_id)
                should_run_post_ingestion = finalized or sweep_drained
                if sweep_drained:
                    self._ingested_conversations.add(conversation_id)
                    try:
                        self._advance_compaction_watermark()
                    except Exception:
                        logger.warning(
                            "_advance_compaction_watermark raised after "
                            "row-based sweep for conv=%s",
                            conversation_id[:12],
                            exc_info=True,
                        )
                if should_run_post_ingestion:
                    latest = self._latest_body
                    if latest:
                        _messages = self._completed_history_messages(_extract_ingestible_messages(latest))
                        self._record_ingestion_watermark(_messages, conversation_id)
                    self._compact_after_ingestion(initial_messages)
                    self._transition_to(SessionState.ACTIVE)
                else:
                    logger.info(
                        "legacy ingestion finalisation deferred for conv=%s "
                        "— staying in INGESTING so the next worker can resume",
                        conversation_id[:12],
                    )

    def _ingest_messages_with_progress(
        self,
        messages: list[Message],
        baseline: int = 0,
        cumulative_total: int | None = None,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> None:
        """Call engine.ingest_history with a progress callback that emits events.

        Args:
            messages: Ingestible message stream to ingest.
            baseline: Already-ingested turns before this batch (for cumulative progress).
            cumulative_total: Total turns across all batches. Defaults to baseline + batch size.

        Raises ``_IngestionCancelled`` if ``_ingestion_cancel`` is set.
        """
        conversation_id = self.engine.config.conversation_id
        t0 = time.monotonic()
        baseline_history_tokens = 0
        grouped = self._group_history_messages(messages)
        _total = cumulative_total if cumulative_total is not None else baseline + len(grouped)
        # Bug D invariant: progress counters must never dip below the durable
        # canonical floor. ``done_ingestible`` in Postgres is a monotonic
        # SUM(covered_ingestible_entries WHERE tagged_at IS NOT NULL) that
        # cannot regress. The in-memory ``_ingestion_progress`` tuple is a
        # view over that floor — if we allow it to report ``(0, total)``
        # during a resume while the DB still has thousands of tagged rows,
        # the dashboard displays a false "ingestion restarted from zero".
        try:
            _snapshot_done_floor = int(
                self.engine._store.read_progress_snapshot(
                    conversation_id,
                ).done_ingestible or 0
            )
        except Exception:  # pragma: no cover — defensive
            _snapshot_done_floor = 0
        logger.info(
            "INGEST_BATCH baseline=%d cumulative_total=%s turns=%d index_size=%d "
            "durable_floor=%d conversation=%s",
            baseline, _total, len(grouped),
            len(self.engine._turn_tag_index.entries),
            _snapshot_done_floor,
            conversation_id[:12],
        )
        # Bug 2 fix (P2): heartbeat cadence is handled by the dedicated
        # ``_run_heartbeat_sidecar`` thread (started alongside the
        # ingestion worker in ``start_ingestion_if_needed``). We no longer
        # refresh the lease from the progress callback because a single
        # long-running tagging turn (>TTL) would let the lease expire
        # before the next callback fires. The sidecar uses
        # ``threading.Event.wait`` so cancel signals interrupt its sleep
        # promptly; when it detects a rejected refresh it sets
        # ``_ingestion_cancel`` — which we pick up at the top of each
        # progress tick via the cancellation check below.

        def on_progress(done: int, total: int, entry) -> None:
            nonlocal baseline_history_tokens
            cum_done = baseline + done
            # Check cancellation before updating progress. The sidecar
            # heartbeat thread flips ``_ingestion_cancel`` on lease loss,
            # and cancel-and-resume does the same for takeover.
            if self._ingestion_cancel.is_set():
                raise _IngestionCancelled(cum_done, _total)
            # Bug D: clamp displayed progress to the durable floor. Even
            # when this worker legitimately starts its in-memory counter
            # low (e.g. a fresh resume where ``baseline`` reflects only
            # this worker's restored ``TurnTagIndex`` size, not the DB
            # truth), the user-facing counter must reflect the canonical
            # row state. The floor itself is monotonic at the DB.
            display_done = max(cum_done, _snapshot_done_floor)
            self._ingestion_progress = (display_done, _total)
            if self.metrics:
                turn_num = entry.turn_number
                local_turn = turn_num - baseline
                preview = ""
                tpt = 0
                if 0 <= local_turn < len(grouped):
                    turn_messages = grouped[local_turn].messages
                    if turn_messages:
                        preview = _strip_envelope(turn_messages[0].content)[:200]
                    tpt = sum(len(msg.content) for msg in turn_messages) // 4
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

        ingest_kwargs = {
            "progress_callback": on_progress,
            "turn_offset": baseline,
            "require_existing_canonical": True,
            "expected_lifecycle_epoch": self.engine._engine_state.lifecycle_epoch,
        }
        if tool_output_refs_by_turn is not None:
            ingest_kwargs["tool_output_refs_by_turn"] = tool_output_refs_by_turn
        turns = self.engine.ingest_history(messages, **ingest_kwargs)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        logger.info(
            "INGEST %d turns in %dms (conversation=%s)",
            turns, int(elapsed_ms), conversation_id[:12],
        )

        if self.metrics:
            self.metrics.record({
                "type": "history_ingestion",
                "turns_ingested": turns,
                "turns_received": len(grouped),
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
