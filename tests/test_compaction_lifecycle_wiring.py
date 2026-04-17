"""Task F-E: compaction lifecycle triggers are wired to ``_run_compact``.

Verifies that ``_compact_after_ingestion`` → ``_run_compact`` now calls the
DB-backed compaction lifecycle (``enter_compaction`` /
``advance_compaction_phase`` / ``exit_compaction``) in addition to the
legacy dict-based ``_update_compaction_state`` mirror. Downstream
consumers (SSE, dashboards) rely on the ``conversations.phase`` flip,
the ``compaction_operation`` row, and the published progress events.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.progress_events import (
    CompactionProgressEvent,
    PhaseTransitionEvent,
)
from virtual_context.types import (
    CompactionReport,
    CompactionResult,
    Message,
)


def test_compact_after_ingestion_wires_db_backed_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_compact_after_ingestion`` must drive the new lifecycle end-to-end.

    * While the fake compactor is running, the progress snapshot exposes
      an ``active_compaction`` row and ``phase == 'compacting'``.
    * Once it returns successfully, the row drops out of the snapshot
      (status flips to ``'completed'`` → excluded from ``active_compaction``)
      and phase transitions back to ``'active'`` or ``'ingesting'``.
    * ``PhaseTransitionEvent`` and ``CompactionProgressEvent`` both fire
      via the engine's progress bus, including a terminal event with
      ``status='completed'`` and at least one phase-advance event
      from the fake compactor's progress callback.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Seed a phase outside 'compacting' so the ``active`` → ``'compacting'``
    # transition triggered by ``enter_compaction`` is observable.
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")

    # Capture published progress events.
    events: list = []
    state.engine.progress_event_bus.subscribe(events.append)

    # Monkey-patch the compactor: return a minimal report and invoke the
    # progress callback once per planned phase so we can assert the
    # ``advance_compaction_phase`` wiring fires for at least one
    # pipeline-advertised phase (``segment_tagging``).
    snapshot_during: dict = {}

    def _fake_compact_if_needed(history, signal, progress_callback=None, turn_id=""):
        # Snapshot the progress during compaction — the compaction_operation
        # row must be visible WHILE the compactor is still running.
        snapshot_during["phase"] = inner.read_progress_snapshot(conv_id).phase
        snapshot_during["active_compaction"] = (
            inner.read_progress_snapshot(conv_id).active_compaction
        )
        if progress_callback is not None:
            # Pipeline-style callback: ``phase`` + ``phase_name`` kwargs
            # matching the planned phase list so ``advance_compaction_phase``
            # records a forward move from the initial "starting".
            progress_callback(
                1, 2, None,
                phase="segment_tagging",
                phase_name="segment_tagging",
                overall_percent=10,
            )
            progress_callback(
                2, 2, None,
                phase="tag_summaries",
                phase_name="tag_summaries",
                overall_percent=100,
            )
        return CompactionReport(
            segments_compacted=1,
            tokens_freed=100,
            tags=["topic"],
            results=[CompactionResult(
                segment_id="seg_1",
                primary_tag="topic",
                original_tokens=200,
                summary_tokens=100,
            )],
            tag_summaries_built=0,
            cover_tags=[],
        )

    monkeypatch.setattr(state.engine, "compact_if_needed", _fake_compact_if_needed)

    # Seed enough proxy-side history so ``_compact_after_ingestion``'s
    # ``compactable > 0`` guard passes: history must exceed the
    # ``protected_recent_turns * 2`` floor (default 12 messages).
    state.conversation_history = [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(20)
    ]

    # Kick off the full post-ingestion compaction path.
    state._compact_after_ingestion(state.conversation_history)

    # --- Assertions on progress_snapshot (DB-backed state) ---
    # While the fake compactor ran, the snapshot saw the new row +
    # 'compacting' phase. This is the key proof that ``enter_compaction``
    # fired before ``compact_if_needed`` was invoked.
    assert snapshot_during["phase"] == "compacting"
    assert snapshot_during["active_compaction"] is not None
    assert snapshot_during["active_compaction"].phase_count == 7

    # After the fake compactor returned successfully, the operation row
    # is terminal ('completed') — ``read_progress_snapshot`` filters
    # ``active_compaction`` on ``status IN ('queued','running')``, so it
    # drops out of the snapshot.
    snap_after = inner.read_progress_snapshot(conv_id)
    assert snap_after.active_compaction is None
    # And the phase flipped back out of 'compacting' via
    # ``drain_compaction_exit`` — no pending canonical rows here so
    # drain selects 'active'.
    assert snap_after.phase == "active"

    # --- Assertions on event bus (subscribers) ---
    # ``enter_compaction`` publishes a PhaseTransitionEvent (active →
    # compacting) + the entry CompactionProgressEvent; the pipeline's
    # callback triggers advance_compaction_phase (more CompactionProgressEvents);
    # ``exit_compaction`` publishes the terminal CompactionProgressEvent
    # + the 'compacting' → 'active' PhaseTransitionEvent.
    phase_events = [e for e in events if isinstance(e, PhaseTransitionEvent)]
    assert any(
        e.old_phase == "active" and e.new_phase == "compacting"
        for e in phase_events
    )
    assert any(
        e.old_phase == "compacting" and e.new_phase in ("active", "ingesting")
        for e in phase_events
    )

    compaction_events = [
        e for e in events if isinstance(e, CompactionProgressEvent)
    ]
    # At least: 1 entry event + 1 advance event (from segment_tagging) +
    # 1 advance event (tag_summaries) + 1 terminal event.
    assert len(compaction_events) >= 3
    # Entry event carries the seed phase.
    assert any(
        e.status == "queued" and e.phase_name == "starting"
        for e in compaction_events
    )
    # A pipeline-driven advance for segment_tagging fired.
    assert any(
        e.status == "running" and e.phase_name == "segment_tagging"
        for e in compaction_events
    )
    # Terminal event marks completed.
    terminal = [e for e in compaction_events if e.status == "completed"]
    assert len(terminal) >= 1
