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

    # Active episode owned by SOMEONE ELSE.
    state.engine._store.read_progress_snapshot.return_value = (
        _snapshot_with_owner(conv_id, owner_worker_id="other-worker:1:deadbeef")
    )

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


def test_legacy_ingestion_heartbeats_during_long_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The progress callback must refresh the ingestion heartbeat on
    wall-clock cadence (every ``INGESTION_LEASE_TTL_S / 2`` seconds of
    turn time) so a long-running legacy thread does not lose its lease.
    """
    from virtual_context.proxy import state as state_module

    state, _metrics = _make_state()

    heartbeat_counter = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        heartbeat_counter["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    # Manual clock that advances 20 seconds per progress callback,
    # simulating a slow LLM-heavy tagging turn. With
    # ``INGESTION_LEASE_TTL_S = 30`` the refresh threshold is 15 seconds,
    # so every turn crosses the threshold and a heartbeat should fire.
    clock = {"t": 1_000_000.0}

    def _monotonic() -> float:
        return clock["t"]

    monkeypatch.setattr(state_module.time, "monotonic", _monotonic)

    # Drive ingest_history synchronously so it fires the progress
    # callback ``turns`` times (done = 1 .. turns). Advance the clock
    # by 20s per turn so the wall-clock threshold (15s) is crossed each
    # call.
    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        n = len(messages) // 2
        for i in range(n):
            entry = TurnTagEntry(
                turn_number=turn_offset + i,
                message_hash=f"h{turn_offset + i}",
                tags=[f"t{turn_offset + i}"],
                primary_tag=f"t{turn_offset + i}",
            )
            clock["t"] += 20.0
            if progress_callback:
                progress_callback(i + 1, n, entry)
        return n

    state.engine.ingest_history.side_effect = _fake_ingest

    # 6 slow turns → each crosses the 15s threshold → 6 heartbeats.
    messages: list[Message] = []
    for i in range(6):
        messages.append(Message(role="user", content=f"Q{i}"))
        messages.append(Message(role="assistant", content=f"A{i}"))

    state._ingest_messages_with_progress(messages, baseline=0)

    assert heartbeat_counter["count"] >= 2, (
        f"Expected at least 2 heartbeat refreshes during a 6-slow-turn "
        f"ingestion (wall-clock cadence INGESTION_LEASE_TTL_S/2 = 15s, "
        f"each turn advances clock by 20s), got {heartbeat_counter['count']}"
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
    ownership lost), the legacy thread must bail out WITHOUT calling
    ``complete_ingestion_episode``. The episode stays 'running'.
    """
    from virtual_context.proxy import state as state_module

    state, _metrics = _make_state()
    state.engine._engine_state.lifecycle_epoch = 5

    # Heartbeat refresh rejects — simulates stale epoch.
    state.engine._store.refresh_ingestion_heartbeat.return_value = False

    # Manual clock that advances 20s per turn so the 15s wall-clock
    # heartbeat threshold is crossed and the refresh call fires.
    clock = {"t": 2_000_000.0}

    def _monotonic() -> float:
        return clock["t"]

    monkeypatch.setattr(state_module.time, "monotonic", _monotonic)

    # Drive ingest_history far enough to trigger at least one heartbeat
    # call. With 20s-per-turn advancement the first turn already crosses
    # the 15s threshold and the refresh fires.
    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        n = len(messages) // 2
        for i in range(n):
            entry = TurnTagEntry(
                turn_number=turn_offset + i,
                message_hash=f"h{turn_offset + i}",
                tags=[f"t{turn_offset + i}"],
                primary_tag=f"t{turn_offset + i}",
            )
            clock["t"] += 20.0
            if progress_callback:
                progress_callback(i + 1, n, entry)
        return n

    state.engine.ingest_history.side_effect = _fake_ingest

    initial_messages = [
        Message(role="user", content="Q0"),
        Message(role="assistant", content="A0"),
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
        Message(role="user", content="Q2"),
        Message(role="assistant", content="A2"),
    ]
    state._run_ingestion_with_catchup(
        initial_messages, baseline=0, cumulative_total=3,
    )

    # Heartbeat rejected → we raise _IngestionCancelled from the progress
    # callback → _run_ingestion_with_catchup sees the cancel path and
    # skips completion.
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
    """With a manual clock that advances 20 seconds per turn (simulating
    a slow LLM-heavy tagging turn), each turn should cross the 15-second
    wall-clock threshold and fire a heartbeat — regardless of turn count
    being low.
    """
    from virtual_context.proxy import state as state_module

    state, _metrics = _make_state()
    state.engine._store.refresh_ingestion_heartbeat.return_value = True

    # Manual clock. Advance 20s per progress callback; the wall-clock
    # threshold is INGESTION_LEASE_TTL_S / 2 = 15s.
    clock = {"t": 5_000_000.0}

    def _monotonic() -> float:
        return clock["t"]

    monkeypatch.setattr(state_module.time, "monotonic", _monotonic)

    refresh_calls = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        refresh_calls["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        n = len(messages) // 2
        for i in range(n):
            entry = TurnTagEntry(
                turn_number=turn_offset + i,
                message_hash=f"h{turn_offset + i}",
                tags=[f"t{turn_offset + i}"],
                primary_tag=f"t{turn_offset + i}",
            )
            # Simulate a slow LLM-heavy tagging turn (20s).
            clock["t"] += 20.0
            if progress_callback:
                progress_callback(i + 1, n, entry)
        return n

    state.engine.ingest_history.side_effect = _fake_ingest

    # 3 slow turns → 6 messages.
    messages: list[Message] = []
    for i in range(3):
        messages.append(Message(role="user", content=f"Q{i}"))
        messages.append(Message(role="assistant", content=f"A{i}"))

    state._ingest_messages_with_progress(messages, baseline=0)

    assert refresh_calls["count"] >= 2, (
        f"Expected at least 2 heartbeat refreshes across 3 slow turns "
        f"(wall-clock 20s-per-turn vs 15s threshold), got "
        f"{refresh_calls['count']}. Cadence must NOT be gated on turn "
        f"count being even."
    )


def test_heartbeat_does_not_refresh_when_wall_clock_has_not_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a manual clock that advances only 1 second per turn, no
    heartbeat should fire across 5 turns (total 5s elapsed, well under
    the 15s threshold).
    """
    from virtual_context.proxy import state as state_module

    state, _metrics = _make_state()
    state.engine._store.refresh_ingestion_heartbeat.return_value = True

    clock = {"t": 6_000_000.0}

    def _monotonic() -> float:
        return clock["t"]

    monkeypatch.setattr(state_module.time, "monotonic", _monotonic)

    refresh_calls = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        refresh_calls["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    def _fake_ingest(messages, progress_callback=None, turn_offset=0, **_kw):
        n = len(messages) // 2
        for i in range(n):
            entry = TurnTagEntry(
                turn_number=turn_offset + i,
                message_hash=f"h{turn_offset + i}",
                tags=[f"t{turn_offset + i}"],
                primary_tag=f"t{turn_offset + i}",
            )
            # Fast turn — 1 second.
            clock["t"] += 1.0
            if progress_callback:
                progress_callback(i + 1, n, entry)
        return n

    state.engine.ingest_history.side_effect = _fake_ingest

    # 5 fast turns → 10 messages. Cumulative wall-clock = 5s, below the
    # 15-second threshold.
    messages: list[Message] = []
    for i in range(5):
        messages.append(Message(role="user", content=f"Q{i}"))
        messages.append(Message(role="assistant", content=f"A{i}"))

    state._ingest_messages_with_progress(messages, baseline=0)

    assert refresh_calls["count"] == 0, (
        f"No heartbeat refresh should fire when cumulative wall-clock "
        f"(5s) is below INGESTION_LEASE_TTL_S/2 (15s); got "
        f"{refresh_calls['count']} refreshes."
    )
