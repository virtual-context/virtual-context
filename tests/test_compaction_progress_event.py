import pytest
from pathlib import Path
from virtual_context.core.progress_events import CompactionProgressEvent


def test_enter_compaction_publishes_compaction_event(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.enter_compaction(phase_count=3, initial_phase_name="init")
    compaction_events = [e for e in events if isinstance(e, CompactionProgressEvent)]
    assert len(compaction_events) == 1
    assert compaction_events[0].phase_name == "init"
    assert compaction_events[0].phase_count == 3
    assert compaction_events[0].phase_index == 0
    assert compaction_events[0].status == "queued"


def test_advance_compaction_phase_publishes_event(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")
    state.enter_compaction(phase_count=3)
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.advance_compaction_phase(phase_index=1, phase_name="summarizing")
    compaction_events = [e for e in events if isinstance(e, CompactionProgressEvent)]
    assert any(
        e.phase_index == 1 and e.phase_name == "summarizing"
        for e in compaction_events
    )


def test_exit_compaction_publishes_terminal_event(tmp_path):
    """exit_compaction(success=True) -> completed status event, then phase transition.
    The compaction operation row status is 'completed' or 'failed' before drain_compaction_exit
    changes phase."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="compacting")
    inner.start_compaction_operation(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, phase_count=3, phase_name="init",
    )
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.exit_compaction(success=True)
    compaction_events = [e for e in events if isinstance(e, CompactionProgressEvent)]
    # Note: exit_compaction first sets status='completed' on compaction_operation,
    # THEN drain_compaction_exit transitions phase. By the time we publish the
    # terminal event (before or after drain?), active_compaction is either
    # the completed one or None. Implementer should publish BEFORE drain so
    # the event reflects the completion state.
    # Either way, at least one CompactionProgressEvent should fire with
    # status='completed'.
    terminal = [e for e in compaction_events if e.status in ("completed", "failed")]
    assert len(terminal) >= 1
    assert terminal[-1].status == "completed"


def test_exit_compaction_failure_publishes_failed_event(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="compacting")
    inner.start_compaction_operation(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, phase_count=3, phase_name="init",
    )
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.exit_compaction(success=False, error_message="boom")
    compaction_events = [e for e in events if isinstance(e, CompactionProgressEvent)]
    terminal = [e for e in compaction_events if e.status in ("completed", "failed")]
    assert any(e.status == "failed" for e in terminal)
