from dataclasses import FrozenInstanceError
import pytest
from virtual_context.core.progress_events import (
    ProgressEvent, IngestionProgressEvent, CompactionProgressEvent,
    PhaseTransitionEvent, LifecycleResetEvent,
)


def test_progress_events_are_frozen_dataclasses():
    ev = IngestionProgressEvent(
        conversation_id="c", lifecycle_epoch=1, kind="ingestion",
        timestamp=1.0, episode_id="e", done=5, total=10,
    )
    with pytest.raises(FrozenInstanceError):
        ev.done = 99


def test_ingestion_event_fields():
    ev = IngestionProgressEvent(
        conversation_id="c", lifecycle_epoch=1, kind="ingestion",
        timestamp=1.0, episode_id="e", done=5, total=10,
    )
    assert ev.conversation_id == "c"
    assert ev.lifecycle_epoch == 1
    assert ev.kind == "ingestion"
    assert ev.timestamp == 1.0
    assert ev.episode_id == "e"
    assert ev.done == 5
    assert ev.total == 10


def test_compaction_event_fields():
    ev = CompactionProgressEvent(
        conversation_id="c", lifecycle_epoch=1, kind="compaction",
        timestamp=1.0, operation_id="op1", phase_name="summarizing",
        phase_index=2, phase_count=5, status="running",
    )
    assert ev.operation_id == "op1"
    assert ev.phase_name == "summarizing"
    assert ev.phase_index == 2
    assert ev.phase_count == 5
    assert ev.status == "running"


def test_phase_transition_event_fields():
    ev = PhaseTransitionEvent(
        conversation_id="c", lifecycle_epoch=1, kind="phase_transition",
        timestamp=1.0, old_phase="init", new_phase="ingesting",
    )
    assert ev.old_phase == "init"
    assert ev.new_phase == "ingesting"


def test_lifecycle_reset_event_fields():
    ev = LifecycleResetEvent(
        conversation_id="c", lifecycle_epoch=2, kind="lifecycle_reset",
        timestamp=1.0, old_epoch=1, new_epoch=2,
    )
    assert ev.old_epoch == 1
    assert ev.new_epoch == 2


def test_all_subclasses_inherit_base_fields():
    # Ensure all 4 event subclasses have the base's conversation_id/lifecycle_epoch/kind/timestamp.
    for cls, extra_kwargs in [
        (IngestionProgressEvent, {"episode_id": "e", "done": 0, "total": 1}),
        (CompactionProgressEvent, {"operation_id": "o", "phase_name": "p", "phase_index": 0, "phase_count": 1, "status": "running"}),
        (PhaseTransitionEvent, {"old_phase": "init", "new_phase": "active"}),
        (LifecycleResetEvent, {"old_epoch": 1, "new_epoch": 2}),
    ]:
        ev = cls(conversation_id="c", lifecycle_epoch=1, kind="x", timestamp=1.0, **extra_kwargs)
        assert ev.conversation_id == "c"
        assert ev.lifecycle_epoch == 1
        assert ev.timestamp == 1.0
