"""Legacy ingestion lease-ownership tests (P1 seam closure).

These tests close the final two P1 seams in the legacy ingestion authority
migration:

* Seam A — ``resume_pending_ingestion_if_needed`` must NOT spawn the
  legacy thread when another worker owns the ingestion lease. The gate
  lives in ``server.py`` (primary) and ``state.py`` (defense-in-depth).
* Seam B — the legacy ``_run_ingestion_with_catchup`` /
  ``_ingest_messages_with_progress`` pair must own the lease lifecycle
  end-to-end: heartbeat during the long run, complete the episode on
  clean exit, flip DB phase back to 'active', and publish a
  ``PhaseTransitionEvent``. Any failure path (cancel, stale-epoch,
  ownership loss) must NOT complete the episode.

Every test targets a single scenario and uses narrow mocks so it can be
driven with a single pytest node ID per the review constraints.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.core.progress_snapshot import (
    ActiveEpisodeSnapshot,
    ProgressSnapshot,
)
from virtual_context.proxy.formats import AnthropicFormat
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.server import (
    ProxyState,
    SessionState,
    prepare_payload,
)
from virtual_context.proxy.state import PhaseDecision
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import (
    AssembledContext,
    EngineState,
    Message,
    TurnTagEntry,
)


# ---------------------------------------------------------------------------
# Mock engine / state builders (deliberately narrow to keep each test
# focussed on a single seam). Mirrors ``tests/test_proxy.py::TestPrepareRouting``
# and ``tests/test_proxy_session.py::TestSessionStateMachine``.
# ---------------------------------------------------------------------------


def _make_state(*, conversation_id: str = "conv-legacy-ingest") -> tuple[ProxyState, ProxyMetrics]:
    engine = MagicMock()
    engine._turn_tag_index = TurnTagIndex()
    engine._engine_state = EngineState()
    engine._store = MagicMock()
    engine._store.get_all_tags.return_value = []
    # Row-based DB sweep runs after the legacy pair-based path in
    # ``_run_ingestion_with_catchup``; it fetches untagged rows via this
    # method. An unset MagicMock would return a truthy MagicMock and
    # infinite-loop the while True. Return an empty list so the sweep
    # no-ops (nothing to drain), then proceeds to complete the episode
    # via the already-mocked ``complete_ingestion_episode``.
    engine._store.iter_untagged_canonical_rows.return_value = []
    engine.config.conversation_id = conversation_id
    engine.config.context_window = 120_000
    engine.config.monitor.context_window = 120_000
    engine.config.monitor.protected_recent_turns = 0
    engine.config.monitor.store_recovery_threshold = 0.70
    engine.config.monitor.defer_payload_mutation = False
    engine.config.monitor.flush_ttl_seconds = 300
    engine.config.monitor.fill_pass_enabled = False
    engine.config.proxy.passthrough_trim_ratio = 0.40
    engine.config.proxy.upstream_context_limit = 120_000
    engine.config.proxy.enable_tool_output_compression = False
    engine.config.proxy.max_output_media_bytes = 0
    engine.config.proxy.history_widening_threshold = 0.10
    engine.config.tag_generator.context_lookback_pairs = 3
    engine.config.tag_generator.context_bleed_threshold = 0
    engine.config.tool_output.enabled = False
    engine.process_broad_tag_split.return_value = None
    engine.on_message_inbound.return_value = AssembledContext()
    metrics = ProxyMetrics()
    return ProxyState(engine, metrics=metrics), metrics


def _snapshot_with_owner(conv_id: str, owner_worker_id: str) -> ProgressSnapshot:
    """Build a ``ProgressSnapshot`` carrying an active episode owned by
    ``owner_worker_id`` — the defense-in-depth check reads
    ``snap.active_episode.owner_worker_id`` and blocks when it does not
    match ``self._worker_id``.
    """
    return ProgressSnapshot(
        conversation_id=conv_id,
        lifecycle_epoch=1,
        phase="ingesting",
        total_ingestible=5,
        done_ingestible=1,
        last_raw_payload_entries=0,
        last_ingestible_payload_entries=0,
        active_episode=ActiveEpisodeSnapshot(
            episode_id="ep-1",
            raw_payload_entries=0,
            owner_worker_id=owner_worker_id,
            heartbeat_ts="2026-04-17T00:00:00Z",
        ),
        active_compaction=None,
    )


# ---------------------------------------------------------------------------
# Seam A — ``resume_pending_ingestion_if_needed`` must be gated on lease
# ownership at both the server-side decision point AND inside the method
# itself (defense-in-depth).
# ---------------------------------------------------------------------------


def test_non_owner_pending_indexing_does_not_resume_ingestion() -> None:
    """When ``handle_prepare_payload`` returns ``started_tagger=False``
    and ``resolve_prepare_state`` returns ``"pending_indexing"``, the
    server-side dispatch must NOT call ``resume_pending_ingestion_if_needed``
    — another worker owns the lease.
    """
    state, metrics = _make_state()
    fmt = AnthropicFormat()
    body = {
        "model": "claude-opus-4-6",
        "stream": False,
        "messages": [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
            {"role": "user", "content": "now"},
        ],
    }

    # Non-owner PhaseDecision. Forces ``_owns_ingestion_lease`` False.
    def _hpp(*_a, **_k):
        return PhaseDecision(phase="ingesting", started_tagger=False)

    # Drive ACTIVE → pending_indexing via resolve_prepare_state. We stub
    # both the state and the reason the server.py dispatch reads.
    def _resolve(_history):
        return SessionState.INGESTING, "pending_indexing"

    resume_calls = {"count": 0}

    def _resume() -> bool:
        resume_calls["count"] += 1
        return False

    # Also stub start_ingestion_if_needed so the passthrough branch below
    # is a no-op we don't need to care about.
    def _start(_history):
        return None

    state.handle_prepare_payload = _hpp
    state.resolve_prepare_state = _resolve
    state.resume_pending_ingestion_if_needed = _resume
    state.start_ingestion_if_needed = _start

    asyncio.run(
        prepare_payload(
            body,
            state,
            fmt,
            metrics,
            body_bytes=json.dumps(body).encode("utf-8"),
        )
    )

    assert resume_calls["count"] == 0, (
        "Non-owner worker (started_tagger=False) must NOT call "
        "resume_pending_ingestion_if_needed — another worker holds the lease."
    )


def test_non_owner_resume_ingestion_defense_in_depth() -> None:
    """Direct call into ``resume_pending_ingestion_if_needed`` must NOT
    spawn a thread if ``read_progress_snapshot`` reports an active
    episode owned by another worker. Defense-in-depth for callers that
    bypass the server.py gate.
    """
    state, _metrics = _make_state()
    conv_id = state.engine.config.conversation_id

    # Force has_pending_indexing() True so we'd ordinarily spawn.
    state.engine._turn_tag_index.append(TurnTagEntry(
        turn_number=0, message_hash="h0", tags=["t0"], primary_tag="t0",
    ))
    state.engine._engine_state.last_indexed_turn = 0
    state.engine._engine_state.last_completed_turn = 2

    # Active episode owned by SOMEONE ELSE. ``resume_pending_ingestion_if_needed``
    # now atomically attempts to claim the lease — the store's
    # ``claim_ingestion_lease`` is authoritative: it returns False iff another
    # worker holds a live lease. The older ``read_progress_snapshot`` observer
    # check is only used as a fallback for stores lacking the lease API.
    state.engine._store.read_progress_snapshot.return_value = (
        _snapshot_with_owner(conv_id, owner_worker_id="other-worker:1:deadbeef")
    )
    state.engine._store.claim_ingestion_lease.return_value = False

    spawned = state.resume_pending_ingestion_if_needed()

    assert spawned is False, (
        "Defense-in-depth: resume_pending_ingestion_if_needed must return "
        "False when another worker owns the active episode."
    )
    assert state._ingestion_thread is None, (
        "No legacy ingestion thread should be created when another worker "
        "owns the lease — even for direct callers that bypass server.py."
    )


def test_non_owner_start_ingestion_defense_in_depth() -> None:
    """Direct call into ``start_ingestion_if_needed`` must NOT spawn a
    thread if ``read_progress_snapshot`` reports an active episode owned
    by another worker. Defense-in-depth for callers that bypass the
    server.py gate.
    """
    state, _metrics = _make_state()
    conv_id = state.engine.config.conversation_id
    state.engine.ingest_history.return_value = 2

    state.engine._store.read_progress_snapshot.return_value = (
        _snapshot_with_owner(conv_id, owner_worker_id="other-worker:2:cafe0001")
    )

    pairs = [
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
        Message(role="user", content="Q2"),
        Message(role="assistant", content="A2"),
    ]
    state.start_ingestion_if_needed(pairs)

    assert state._ingestion_thread is None, (
        "Defense-in-depth: start_ingestion_if_needed must not spawn the "
        "legacy thread when another worker owns the active episode."
    )


# ---------------------------------------------------------------------------
# Seam B — legacy thread owns the lease lifecycle
# (heartbeat while running, complete episode on success, flip phase,
# publish PhaseTransitionEvent; skip all three on any failure path).
# ---------------------------------------------------------------------------


def test_legacy_ingestion_heartbeats_during_long_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sidecar heartbeat thread must refresh the ingestion lease on
    wall-clock cadence (every ``INGESTION_LEASE_TTL_S / 2`` seconds) so a
    long-running legacy thread does not lose its lease — independent of
    whether the worker is mid-turn or between turns.
    """
    from virtual_context.proxy import state as state_module

    # Monkeypatch TTL to a very small value so the test runs quickly.
    monkeypatch.setattr(state_module, "INGESTION_LEASE_TTL_S", 0.2)

    state, _metrics = _make_state()

    heartbeat_counter = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        heartbeat_counter["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    # Run the sidecar directly with a fake worker alive for ~0.5s so we
    # get at least 2 refresh ticks at the 0.1s interval.
    stop = threading.Event()

    def _fake_worker() -> None:
        stop.wait(timeout=0.5)

    worker = threading.Thread(target=_fake_worker, daemon=True)
    state._ingestion_thread = worker
    worker.start()
    try:
        state._run_heartbeat_sidecar(
            state.engine.config.conversation_id,
            epoch=int(state.engine._engine_state.lifecycle_epoch),
        )
    finally:
        stop.set()
        worker.join(timeout=1.0)

    assert heartbeat_counter["count"] >= 2, (
        f"Expected at least 2 heartbeat refreshes during a 0.5s sidecar "
        f"run (wall-clock cadence INGESTION_LEASE_TTL_S/2 = 0.1s), "
        f"got {heartbeat_counter['count']}"
    )


def test_legacy_ingestion_completes_episode_and_flips_phase_on_success() -> None:
    """On clean exit of ``_run_ingestion_with_catchup``, the legacy
    thread must:
      1. call ``complete_ingestion_episode`` with the current epoch +
         worker id;
      2. call ``set_phase`` with ``phase="active"``;
      3. publish a ``PhaseTransitionEvent`` onto the engine's
         ``progress_event_bus``.
    """
    state, _metrics = _make_state()
    state.engine._engine_state.lifecycle_epoch = 7

    # Make complete+flip succeed so the publish fires.
    state.engine._store.complete_ingestion_episode.return_value = True
    state.engine._store.set_phase.return_value = True
    state.engine._store.refresh_ingestion_heartbeat.return_value = True

    # Replace progress_event_bus with a real-ish recorder so we can
    # introspect what was published.
    from virtual_context.core.event_bus import ProgressEventBus
    from virtual_context.core.progress_events import PhaseTransitionEvent
    bus = ProgressEventBus()
    published: list = []
    bus.subscribe(lambda ev: published.append(ev))
    state.engine.progress_event_bus = bus

    # Simple happy-path ingest_history that fires the progress callback
    # once per turn and returns the turn count. No catch-up work.
    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        n = len(messages) // 2
        for i in range(n):
            entry = TurnTagEntry(
                turn_number=turn_offset + i,
                message_hash=f"h{turn_offset + i}",
                tags=[f"t{turn_offset + i}"],
                primary_tag=f"t{turn_offset + i}",
            )
            if progress_callback:
                progress_callback(i + 1, n, entry)
        return n

    state.engine.ingest_history.side_effect = _fake_ingest

    initial_messages = [
        Message(role="user", content="Q0"),
        Message(role="assistant", content="A0"),
    ]
    state._run_ingestion_with_catchup(
        initial_messages, baseline=0, cumulative_total=1,
    )

    # 1. complete_ingestion_episode was called with our epoch + worker id.
    assert state.engine._store.complete_ingestion_episode.called, (
        "complete_ingestion_episode must be called on clean exit"
    )
    call_kwargs = state.engine._store.complete_ingestion_episode.call_args.kwargs
    assert call_kwargs["conversation_id"] == state.engine.config.conversation_id
    assert call_kwargs["lifecycle_epoch"] == 7
    assert call_kwargs["worker_id"] == state._worker_id

    # 2. set_phase was called with phase="active" at some point during
    #    finalisation. (``_compact_after_ingestion`` may enqueue a
    #    ``compacting`` flip afterwards; we only care that the
    #    ingesting→active flip fired at least once in the sequence.)
    assert state.engine._store.set_phase.called, (
        "set_phase must be called after complete_ingestion_episode succeeds"
    )
    active_flips = [
        call for call in state.engine._store.set_phase.call_args_list
        if call.kwargs.get("phase") == "active"
        and call.kwargs.get("lifecycle_epoch") == 7
    ]
    assert active_flips, (
        "Expected at least one set_phase(phase='active', lifecycle_epoch=7) "
        "call in the finalisation sequence, got "
        f"{state.engine._store.set_phase.call_args_list}"
    )

    # 3. A PhaseTransitionEvent ingesting→active was published.
    transitions = [
        ev for ev in published
        if isinstance(ev, PhaseTransitionEvent)
        and ev.old_phase == "ingesting"
        and ev.new_phase == "active"
    ]
    assert transitions, (
        "A PhaseTransitionEvent(ingesting→active) must be published on "
        "successful legacy ingestion completion."
    )


def test_legacy_ingestion_does_not_complete_episode_on_cancel() -> None:
    """If ``_ingestion_cancel`` is set mid-run, the legacy thread must
    NOT call ``complete_ingestion_episode`` — the episode stays
    'running' so the next worker can pick it up.
    """
    state, _metrics = _make_state()
    state.engine._engine_state.lifecycle_epoch = 3
    state.engine._store.refresh_ingestion_heartbeat.return_value = True

    # ingest_history that flips cancel mid-run. The progress callback
    # raises _IngestionCancelled inside our own on_progress, which
    # percolates up through ingest_history and out of
    # _ingest_messages_with_progress → caught by
    # _run_ingestion_with_catchup.
    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        n = len(messages) // 2
        for i in range(n):
            entry = TurnTagEntry(
                turn_number=turn_offset + i,
                message_hash=f"h{turn_offset + i}",
                tags=[f"t{turn_offset + i}"],
                primary_tag=f"t{turn_offset + i}",
            )
            # Simulate a new request taking over after one turn.
            if i == 0:
                state._ingestion_cancel.set()
            if progress_callback:
                progress_callback(i + 1, n, entry)
        return n

    state.engine.ingest_history.side_effect = _fake_ingest

    initial_messages = [
        Message(role="user", content="Q0"),
        Message(role="assistant", content="A0"),
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
    ]
    state._run_ingestion_with_catchup(
        initial_messages, baseline=0, cumulative_total=2,
    )

    assert not state.engine._store.complete_ingestion_episode.called, (
        "complete_ingestion_episode must NOT be called on cancel — the "
        "episode stays 'running' for the next worker to pick up."
    )


def test_legacy_ingestion_does_not_complete_episode_on_stale_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``refresh_ingestion_heartbeat`` returns False (stale epoch or
    ownership lost), the sidecar sets ``_ingestion_cancel`` and the legacy
    thread must bail out WITHOUT calling ``complete_ingestion_episode``.
    The episode stays 'running'.
    """
    from virtual_context.proxy import state as state_module

    # Shrink TTL so the sidecar fires quickly during the test.
    monkeypatch.setattr(state_module, "INGESTION_LEASE_TTL_S", 0.2)

    state, _metrics = _make_state()
    state.engine._engine_state.lifecycle_epoch = 5

    # Heartbeat refresh rejects — simulates stale epoch. Sidecar sees
    # this and sets ``_ingestion_cancel``.
    state.engine._store.refresh_ingestion_heartbeat.return_value = False

    # Slow ingest_history — simulate a single long turn by sleeping so
    # the sidecar runs at least one refresh cycle and flips cancel.
    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        # Wait long enough for the sidecar (TTL/2 = 0.1s cadence) to
        # issue a rejected refresh and set cancel.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if state._ingestion_cancel.is_set():
                # Mirror engine behaviour — stop ingesting once cancel
                # flips so the progress callback can raise.
                break
            time.sleep(0.02)
        # Fire one progress callback post-cancel so the cancel check
        # inside ``on_progress`` raises ``_IngestionCancelled``.
        entry = TurnTagEntry(
            turn_number=turn_offset,
            message_hash="h0", tags=["t0"], primary_tag="t0",
        )
        if progress_callback:
            progress_callback(1, 1, entry)
        return 1

    state.engine.ingest_history.side_effect = _fake_ingest

    initial_messages = [
        Message(role="user", content="Q0"),
        Message(role="assistant", content="A0"),
    ]

    # Spawn the sidecar ourselves — ``_run_ingestion_with_catchup`` is
    # invoked synchronously from this test, so the caller must wire up
    # the heartbeat thread the same way ``start_ingestion_if_needed``
    # would in production. Use the current thread as the "worker".
    worker = threading.current_thread()
    state._ingestion_thread = worker
    sidecar = threading.Thread(
        target=state._run_heartbeat_sidecar,
        args=(state.engine.config.conversation_id, 5),
        daemon=True,
    )
    state._heartbeat_thread = sidecar
    sidecar.start()
    try:
        state._run_ingestion_with_catchup(
            initial_messages, baseline=0, cumulative_total=1,
        )
    finally:
        state._ingestion_cancel.set()
        sidecar.join(timeout=1.0)

    # Heartbeat rejected → sidecar set cancel → progress callback raised
    # ``_IngestionCancelled`` → _run_ingestion_with_catchup saw the cancel
    # path and skipped completion.
    assert not state.engine._store.complete_ingestion_episode.called, (
        "complete_ingestion_episode must NOT be called when the heartbeat "
        "refresh rejects (stale epoch). The episode must remain 'running'."
    )


# ---------------------------------------------------------------------------
# Bug 1 (P1) — split-brain prevention. When the DB did NOT actually move to
# "active with nothing running", the local worker must NOT mark itself
# complete: no ``_ingested_conversations`` add, no watermark advance, no
# compaction, no SessionState ACTIVE transition, no
# ``PhaseTransitionEvent``.
# ---------------------------------------------------------------------------


def _install_happy_ingest(state: ProxyState) -> None:
    """Install a trivial ingest_history that fires the progress callback
    once and returns 1. Used by the Bug 1 tests that only care about
    the finalisation branch, not the ingestion body.
    """
    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        n = len(messages) // 2
        for i in range(n):
            entry = TurnTagEntry(
                turn_number=turn_offset + i,
                message_hash=f"h{turn_offset + i}",
                tags=[f"t{turn_offset + i}"],
                primary_tag=f"t{turn_offset + i}",
            )
            if progress_callback:
                progress_callback(i + 1, n, entry)
        return n

    state.engine.ingest_history.side_effect = _fake_ingest


def test_legacy_ingestion_defers_local_completion_when_episode_not_completable() -> None:
    """Bug 1 fix: if ``complete_ingestion_episode`` returns False
    (untagged rows remain or lease was stolen), the legacy thread must
    NOT flip local state into "ingested + active":

      * ``_ingested_conversations`` must NOT contain the conversation.
      * SessionState must NOT transition to ACTIVE.
      * ``_compact_after_ingestion`` must NOT be called.
      * No ``PhaseTransitionEvent(ingesting→active)`` is published.
    """
    state, _metrics = _make_state()
    state.engine._engine_state.lifecycle_epoch = 11
    # Start ingestion from INGESTING so we can detect "did NOT transition
    # to ACTIVE".
    state._state = SessionState.INGESTING

    # Episode completion refuses — untagged rows remain.
    state.engine._store.complete_ingestion_episode.return_value = False
    state.engine._store.set_phase.return_value = True
    state.engine._store.refresh_ingestion_heartbeat.return_value = True

    # Event bus recorder.
    from virtual_context.core.event_bus import ProgressEventBus
    from virtual_context.core.progress_events import PhaseTransitionEvent
    bus = ProgressEventBus()
    published: list = []
    bus.subscribe(lambda ev: published.append(ev))
    state.engine.progress_event_bus = bus

    # Spy on _compact_after_ingestion to verify it's NOT invoked.
    compact_calls = {"count": 0}

    def _spy_compact(history: list[Message]) -> None:
        compact_calls["count"] += 1

    state._compact_after_ingestion = _spy_compact

    _install_happy_ingest(state)

    conv_id = state.engine.config.conversation_id
    initial_messages = [
        Message(role="user", content="Q0"),
        Message(role="assistant", content="A0"),
    ]
    state._run_ingestion_with_catchup(
        initial_messages, baseline=0, cumulative_total=1,
    )

    # 1. complete_ingestion_episode was attempted but said no.
    assert state.engine._store.complete_ingestion_episode.called, (
        "complete_ingestion_episode must still be attempted on clean exit"
    )
    # 2. Local worker did NOT mark itself complete.
    assert conv_id not in state._ingested_conversations, (
        "Local worker must NOT add itself to _ingested_conversations when "
        "complete_ingestion_episode returned False"
    )
    # 3. No transition to ACTIVE.
    assert state.session_state != SessionState.ACTIVE, (
        f"SessionState must NOT transition to ACTIVE when the DB refused to "
        f"complete the episode; got {state.session_state}"
    )
    # 4. No compaction side effect.
    assert compact_calls["count"] == 0, (
        f"_compact_after_ingestion must NOT be called when the episode "
        f"refused to complete; got {compact_calls['count']} calls"
    )
    # 5. No PhaseTransitionEvent(ingesting→active) published.
    transitions = [
        ev for ev in published
        if isinstance(ev, PhaseTransitionEvent)
        and ev.old_phase == "ingesting"
        and ev.new_phase == "active"
    ]
    assert not transitions, (
        f"No PhaseTransitionEvent(ingesting→active) may be published when "
        f"complete_ingestion_episode returned False; got {transitions}"
    )


def test_legacy_ingestion_defers_local_completion_when_set_phase_failed() -> None:
    """Bug 1 fix: if ``complete_ingestion_episode`` returned True but
    ``set_phase`` subsequently returned False (stale epoch between the
    two calls), the legacy thread must NOT flip local state. The episode
    is closed but the DB phase is still 'ingesting'; a new lifecycle has
    taken over.
    """
    state, _metrics = _make_state()
    state.engine._engine_state.lifecycle_epoch = 13
    state._state = SessionState.INGESTING

    # Episode completion succeeds but phase flip rejects.
    state.engine._store.complete_ingestion_episode.return_value = True
    state.engine._store.set_phase.return_value = False
    state.engine._store.refresh_ingestion_heartbeat.return_value = True

    from virtual_context.core.event_bus import ProgressEventBus
    from virtual_context.core.progress_events import PhaseTransitionEvent
    bus = ProgressEventBus()
    published: list = []
    bus.subscribe(lambda ev: published.append(ev))
    state.engine.progress_event_bus = bus

    compact_calls = {"count": 0}

    def _spy_compact(history: list[Message]) -> None:
        compact_calls["count"] += 1

    state._compact_after_ingestion = _spy_compact

    _install_happy_ingest(state)

    conv_id = state.engine.config.conversation_id
    initial_messages = [
        Message(role="user", content="Q0"),
        Message(role="assistant", content="A0"),
    ]
    state._run_ingestion_with_catchup(
        initial_messages, baseline=0, cumulative_total=1,
    )

    assert state.engine._store.complete_ingestion_episode.called
    assert state.engine._store.set_phase.called, (
        "set_phase must be attempted after complete_ingestion_episode returned True"
    )
    assert conv_id not in state._ingested_conversations, (
        "Local worker must NOT add itself to _ingested_conversations when "
        "set_phase returned False (stale epoch)"
    )
    assert state.session_state != SessionState.ACTIVE, (
        f"SessionState must NOT transition to ACTIVE when set_phase failed; "
        f"got {state.session_state}"
    )
    assert compact_calls["count"] == 0, (
        f"_compact_after_ingestion must NOT be called when set_phase failed; "
        f"got {compact_calls['count']} calls"
    )
    transitions = [
        ev for ev in published
        if isinstance(ev, PhaseTransitionEvent)
        and ev.old_phase == "ingesting"
        and ev.new_phase == "active"
    ]
    assert not transitions, (
        f"No PhaseTransitionEvent(ingesting→active) may be published when "
        f"set_phase returned False; got {transitions}"
    )


# ---------------------------------------------------------------------------
# Bug 2 (P2) — wall-clock heartbeat cadence. The lease TTL is 30s; a single
# LLM-heavy tagging turn can exceed that, so we cannot rely on turn-count
# cadence. The new cadence is ``INGESTION_LEASE_TTL_S / 2`` wall-clock
# seconds between refreshes.
# ---------------------------------------------------------------------------


def test_heartbeat_refreshes_on_wall_clock_interval_not_turn_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sidecar heartbeat thread runs on wall-clock cadence —
    ``INGESTION_LEASE_TTL_S / 2`` — independent of turn completion, so a
    long-running single turn cannot let the lease expire before a
    refresh fires.
    """
    from virtual_context.proxy import state as state_module

    # Shrink TTL so the test completes in <1s. Interval becomes 0.1s.
    monkeypatch.setattr(state_module, "INGESTION_LEASE_TTL_S", 0.2)

    state, _metrics = _make_state()

    refresh_calls = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        refresh_calls["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    # Fake worker that stays alive for ~0.4s — long enough for ≥2
    # refresh ticks at the 0.1s interval. No turn callbacks fired.
    stop = threading.Event()

    def _fake_worker() -> None:
        stop.wait(timeout=0.4)

    worker = threading.Thread(target=_fake_worker, daemon=True)
    state._ingestion_thread = worker
    worker.start()
    try:
        state._run_heartbeat_sidecar(
            state.engine.config.conversation_id,
            epoch=int(state.engine._engine_state.lifecycle_epoch),
        )
    finally:
        stop.set()
        worker.join(timeout=1.0)

    assert refresh_calls["count"] >= 2, (
        f"Expected at least 2 heartbeat refreshes across a 0.4s sidecar "
        f"run with 0.1s cadence and ZERO turn callbacks, got "
        f"{refresh_calls['count']}. Cadence must NOT be gated on turn "
        f"completion."
    )


def test_heartbeat_does_not_refresh_when_wall_clock_has_not_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the sidecar is cancelled before the first interval elapses,
    no heartbeat refresh should fire. ``Event.wait(timeout)`` returns
    True on set (no timeout) → the loop returns without refreshing.
    """
    from virtual_context.proxy import state as state_module

    # 30s TTL → 15s interval. We cancel immediately so the first
    # refresh never fires.
    monkeypatch.setattr(state_module, "INGESTION_LEASE_TTL_S", 30.0)

    state, _metrics = _make_state()

    refresh_calls = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        refresh_calls["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    # Worker present, but cancel fires immediately so the sidecar's
    # first ``wait`` returns True (cancelled) and it exits without
    # calling refresh.
    stop_worker = threading.Event()

    def _fake_worker() -> None:
        stop_worker.wait(timeout=5.0)

    worker = threading.Thread(target=_fake_worker, daemon=True)
    state._ingestion_thread = worker
    worker.start()

    # Flip cancel BEFORE starting sidecar so its first wait returns
    # immediately.
    state._ingestion_cancel.set()
    try:
        state._run_heartbeat_sidecar(
            state.engine.config.conversation_id,
            epoch=int(state.engine._engine_state.lifecycle_epoch),
        )
    finally:
        stop_worker.set()
        worker.join(timeout=1.0)
        state._ingestion_cancel.clear()

    assert refresh_calls["count"] == 0, (
        f"No heartbeat refresh should fire when cancel is set before "
        f"the first interval elapses; got {refresh_calls['count']} "
        f"refreshes."
    )


# ---------------------------------------------------------------------------
# Bug 1 fix follow-up: cancel-and-resume empty-history shortcut must go
# through the DB-authoritative finalisation helper, not the 3-line
# local-only completion that caused the split-brain.
# ---------------------------------------------------------------------------


def test_cancel_and_resume_empty_history_requires_db_completion() -> None:
    """When a running ingestion is cancelled and the sliced remaining
    history is empty, the local worker must NOT self-complete without
    a DB-authoritative ``complete_ingestion_episode`` + ``set_phase``
    sequence. If ``complete_ingestion_episode`` returns False (lease
    stolen, untagged rows remain, or stale epoch), the worker must stay
    in INGESTING and skip all post-ingestion side effects.

    Covers the split-brain described in the Bug 1 review:
    ``start_ingestion_if_needed`` shortcut at state.py:~2745 (the
    cancel-and-resume empty-history path) formerly called
    ``_ingested_conversations.add`` + ``_transition_to(ACTIVE)`` without
    ever asking the DB whether the episode could actually close.

    Strategy: set up the state so the cancel-and-resume branch is taken
    AND the post-cancel slice leaves empty history. We use a fake
    ``ingest_history`` that blocks until signalled, so the first call
    leaves the thread alive. A threading.Event gates when the thread
    appends to the tag index. We then issue the second call, which
    cancels the first thread; after cancel, re-reading
    ``_indexed_turn_count()`` returns exactly ``needed_turns`` so the
    slice is empty — hitting the shortcut at state.py:~2745.
    """
    import time as _time

    state, _metrics = _make_state()
    conv_id = state.engine.config.conversation_id

    # Episode completion refuses — e.g. untagged rows remain.
    state.engine._store.complete_ingestion_episode.return_value = False
    state.engine._store.set_phase.return_value = True
    state.engine._store.refresh_ingestion_heartbeat.return_value = True

    # Spy on _compact_after_ingestion to verify it's NOT invoked.
    compact_calls = {"count": 0}

    def _spy_compact(history: list[Message]) -> None:
        compact_calls["count"] += 1

    state._compact_after_ingestion = _spy_compact

    # Thread coordination events. ``ready`` is set when the worker has
    # entered ``ingest_history`` (so we know the thread is alive);
    # ``release`` is set by the test to allow the thread to proceed.
    ready = threading.Event()
    release = threading.Event()

    def _blocking_ingest(pairs, progress_callback=None, turn_offset=0, **_kw):
        ready.set()
        # Block until the test releases us OR cancel fires. This keeps
        # the thread alive long enough for the second call to see
        # ``_ingestion_thread.is_alive()`` True at line 2722.
        while not release.is_set() and not state._ingestion_cancel.is_set():
            _time.sleep(0.01)
        n = len(pairs) // 2
        # Raise _IngestionCancelled via the progress callback if cancel
        # was the reason for wakeup (simulates real cancel path).
        if progress_callback and n > 0:
            entry = TurnTagEntry(
                turn_number=turn_offset, message_hash="h0",
                tags=["t"], primary_tag="t",
            )
            progress_callback(1, n, entry)
        return 0

    state.engine.ingest_history.side_effect = _blocking_ingest

    pairs = [
        Message(role=("user" if j % 2 == 0 else "assistant"), content=f"m{j}")
        for j in range(6)  # 3 pairs → needed_turns == 3
    ]

    # First call — spawns the background thread. existing_turns == 0,
    # needed_turns == 3 → bypasses the "already covered" and
    # "existing_turns > 0 and not _thread_running" gates and starts a
    # thread via the normal path.
    state.start_ingestion_if_needed(pairs)
    assert state._state == SessionState.INGESTING

    # Wait for the worker to confirm it's inside ingest_history.
    assert ready.wait(timeout=2.0), "worker did not enter ingest_history"
    assert state._ingestion_thread is not None
    assert state._ingestion_thread.is_alive()

    # Now seed the tag index to cover all 3 pairs. This simulates
    # another worker having tagged all three turns while our legacy
    # thread was blocked. When the second call arrives, the outer
    # ``existing_turns >= needed_turns`` gate at state.py:~2673 would
    # short-circuit — so we instead use exactly ``needed_turns - 1``
    # entries first, then top up right after cancel so the re-read at
    # state.py:~2742 returns >= needed_turns.
    # Strategy: set 2 entries now, then the cancel-and-resume block
    # does its cancel, re-reads indexed_turn_count, and at THAT point
    # we want 3 entries. We can't inject code into the cancel path, so
    # we use a different angle: inject a fake thread stop hook via a
    # thread that tops up the index once cancel fires.
    for i in range(2):
        state.engine._turn_tag_index.append(TurnTagEntry(
            turn_number=i, message_hash=f"h{i}",
            tags=["t"], primary_tag="t",
        ))
    state.engine._engine_state.last_indexed_turn = 1

    # Helper thread: watches for cancel, tops up the index to 3 before
    # the worker exits so the post-join re-read sees 3.
    def _topup_on_cancel() -> None:
        # Wait until the main test sets cancel (i.e., enters
        # cancel-and-resume).
        state._ingestion_cancel.wait(timeout=5.0)
        # Add the final entry so ``_indexed_turn_count() == needed_turns``.
        state.engine._turn_tag_index.append(TurnTagEntry(
            turn_number=2, message_hash="h2",
            tags=["t"], primary_tag="t",
        ))
        state.engine._engine_state.last_indexed_turn = 2
        # Also release the blocking worker so join() returns.
        release.set()

    topup = threading.Thread(target=_topup_on_cancel, daemon=True)
    topup.start()

    try:
        # Second call — enters cancel-and-resume at state.py:~2722.
        # After the worker exits, re-reads ``_indexed_turn_count()`` → 3,
        # slices history → empty, hits the shortcut that MUST route
        # through ``_finalize_legacy_ingestion`` (DB returns False,
        # so no local completion).
        state.start_ingestion_if_needed(pairs)
    finally:
        release.set()
        topup.join(timeout=2.0)

    # complete_ingestion_episode was attempted but said no.
    assert state.engine._store.complete_ingestion_episode.called, (
        "complete_ingestion_episode must be called on the cancel-and-resume "
        "empty-history shortcut — the shortcut can no longer self-complete."
    )
    # Local worker did NOT self-mark as ingested.
    assert conv_id not in state._ingested_conversations, (
        "Local worker must NOT add itself to _ingested_conversations when "
        "complete_ingestion_episode returned False on the cancel-and-resume "
        "empty-history shortcut."
    )
    # No transition to ACTIVE.
    assert state.session_state != SessionState.ACTIVE, (
        f"SessionState must NOT transition to ACTIVE when the DB refused to "
        f"complete the episode on the cancel-and-resume shortcut; got "
        f"{state.session_state}"
    )
    # No compaction side effect.
    assert compact_calls["count"] == 0, (
        f"_compact_after_ingestion must NOT be called when the episode "
        f"refused to complete on the cancel-and-resume shortcut; got "
        f"{compact_calls['count']} calls"
    )


def test_sidecar_heartbeats_during_single_long_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single long-running tagging turn (>TTL) must not lose the
    ingestion lease. The dedicated sidecar thread refreshes every
    ``INGESTION_LEASE_TTL_S / 2`` seconds regardless of turn completion.
    """
    from virtual_context.proxy import state as state_module

    # Shrink TTL so one "long turn" can be simulated in <1s.
    # Interval becomes 0.1s; we sleep 0.35s → ≥2 refresh ticks.
    monkeypatch.setattr(state_module, "INGESTION_LEASE_TTL_S", 0.2)

    state, _metrics = _make_state()

    refresh_calls = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        refresh_calls["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    # Fake "worker" = thread that simulates ONE very slow tagging turn
    # by sleeping for 3× the half-TTL interval. The sidecar must fire
    # multiple refreshes during this single turn.
    done = threading.Event()

    def _one_long_turn() -> None:
        # Single long turn — block for 3× the 0.1s interval.
        done.wait(timeout=0.35)

    worker = threading.Thread(target=_one_long_turn, daemon=True)
    state._ingestion_thread = worker
    worker.start()
    try:
        state._run_heartbeat_sidecar(
            state.engine.config.conversation_id,
            epoch=int(state.engine._engine_state.lifecycle_epoch),
        )
    finally:
        done.set()
        worker.join(timeout=1.0)

    assert refresh_calls["count"] >= 2, (
        f"Expected sidecar to issue at least 2 refreshes during a single "
        f"0.35s turn at 0.1s cadence, got {refresh_calls['count']}. "
        f"The callback-based heartbeat never fired mid-turn — the whole "
        f"point of the sidecar is to decouple refresh cadence from turn "
        f"completion."
    )


def test_resume_pending_ingestion_spawns_heartbeat_sidecar() -> None:
    """The resume path must spawn the heartbeat sidecar alongside the
    ingestion worker — a resumed tagging run longer than
    ``INGESTION_LEASE_TTL_S`` would otherwise lose its lease and let
    another worker race in.
    """
    state, _metrics = _make_state()

    # Force has_pending_indexing() True so the resume flow fires.
    state.engine._turn_tag_index.append(TurnTagEntry(
        turn_number=0, message_hash="h0", tags=["t0"], primary_tag="t0",
    ))
    state.engine._engine_state.last_indexed_turn = 0
    state.engine._engine_state.last_completed_turn = 2

    # No owning episode — ``_another_worker_owns_lease`` returns False
    # because the MagicMock return value is not a real ProgressSnapshot.
    state.engine._store.read_progress_snapshot.return_value = MagicMock()

    # Seed durable pending rows directly so the resume path skips the
    # ``get_canonical_turn_rows`` load.
    state.engine._restored_pending_turns = [
        (0, "Q1", "A1", None, None),
        (1, "Q2", "A2", None, None),
    ]

    # Stub the heavy worker loop so the test ends quickly; the test only
    # asserts that the sidecar was spawned alongside the ingestion thread.
    spawn_event = threading.Event()

    def _no_op_ingest(*_a, **_kw) -> None:
        spawn_event.wait(timeout=1.0)

    state._run_ingestion_with_catchup = _no_op_ingest  # type: ignore[assignment]

    # Stub the sidecar to block until the test releases it, so we can
    # observe ``is_alive()`` True under a deterministic condition.
    sidecar_release = threading.Event()

    def _blocking_sidecar(*_a, **_kw) -> None:
        sidecar_release.wait(timeout=1.0)

    state._run_heartbeat_sidecar = _blocking_sidecar  # type: ignore[assignment]

    try:
        spawned = state.resume_pending_ingestion_if_needed()
        assert spawned is True, "resume path must spawn when pending rows exist"
        assert state._heartbeat_thread is not None, (
            "resume_pending_ingestion_if_needed must spawn the heartbeat "
            "sidecar alongside the ingestion worker"
        )
        assert state._heartbeat_thread.is_alive(), (
            "heartbeat sidecar must be alive right after resume spawn"
        )
        assert state._ingestion_thread is not None
        assert state._ingestion_thread.is_alive()
    finally:
        state._ingestion_cancel.set()
        spawn_event.set()
        sidecar_release.set()
        if state._ingestion_thread is not None:
            state._ingestion_thread.join(timeout=2.0)
        if state._heartbeat_thread is not None:
            state._heartbeat_thread.join(timeout=2.0)
