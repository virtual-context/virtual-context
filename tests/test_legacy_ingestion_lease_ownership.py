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


def test_legacy_ingestion_heartbeats_during_long_run() -> None:
    """The progress callback must refresh the ingestion heartbeat every
    2 turns so a long-running legacy thread does not lose its lease.
    """
    state, _metrics = _make_state()

    heartbeat_counter = {"count": 0}

    def _refresh(**_kwargs) -> bool:
        heartbeat_counter["count"] += 1
        return True

    state.engine._store.refresh_ingestion_heartbeat.side_effect = _refresh

    # Drive ingest_history synchronously so it fires the progress
    # callback ``turns`` times (done = 1 .. turns).
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

    # 6 turns → 12 messages → progress callback fires 6 times with
    # done = 1,2,3,4,5,6. Heartbeat fires on done % 2 == 0 and done > 0,
    # i.e. at done=2, done=4, done=6 → 3 heartbeats.
    messages: list[Message] = []
    for i in range(6):
        messages.append(Message(role="user", content=f"Q{i}"))
        messages.append(Message(role="assistant", content=f"A{i}"))

    state._ingest_messages_with_progress(messages, baseline=0)

    assert heartbeat_counter["count"] >= 2, (
        f"Expected at least 2 heartbeat refreshes during a 6-turn ingestion "
        f"(cadence is every 2 turns), got {heartbeat_counter['count']}"
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


def test_legacy_ingestion_does_not_complete_episode_on_stale_epoch() -> None:
    """If ``refresh_ingestion_heartbeat`` returns False (stale epoch or
    ownership lost), the legacy thread must bail out WITHOUT calling
    ``complete_ingestion_episode``. The episode stays 'running'.
    """
    state, _metrics = _make_state()
    state.engine._engine_state.lifecycle_epoch = 5

    # Heartbeat refresh rejects — simulates stale epoch.
    state.engine._store.refresh_ingestion_heartbeat.return_value = False

    # Drive ingest_history far enough to trigger at least one heartbeat
    # call (done == 2 fires the heartbeat; we supply >= 2 turns).
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
