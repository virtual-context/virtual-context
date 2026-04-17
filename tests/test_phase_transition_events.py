import pytest
from pathlib import Path
from virtual_context.core.progress_events import (
    PhaseTransitionEvent, LifecycleResetEvent,
)


def test_phase_transition_publishes_on_init_to_active_empty_case(tmp_path):
    """Empty init → active transition publishes PhaseTransitionEvent."""
    from tests.test_handle_prepare_payload import _make_proxy_state
    state = _make_proxy_state(tmp_path)
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
    )
    phase_events = [e for e in events if isinstance(e, PhaseTransitionEvent)]
    assert len(phase_events) == 1
    assert phase_events[0].old_phase == "init"
    assert phase_events[0].new_phase == "active"


def test_phase_transition_publishes_on_init_to_ingesting(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state
    state = _make_proxy_state(tmp_path)
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.handle_prepare_payload(
        body={"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]},
        payload_accounting={"raw_payload_entry_count": 2, "ingestible_entry_count": 2},
    )
    phase_events = [e for e in events if isinstance(e, PhaseTransitionEvent)]
    assert any(
        e.new_phase == "ingesting"
        for e in phase_events
    )


def test_lifecycle_reset_publishes_on_resurrect(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.mark_conversation_deleted(conv_id)
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
    )
    reset_events = [e for e in events if isinstance(e, LifecycleResetEvent)]
    assert len(reset_events) == 1
    assert reset_events[0].old_epoch == 1
    assert reset_events[0].new_epoch == 2


def test_enter_compaction_publishes_phase_transition(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state.enter_compaction(phase_count=3)
    phase_events = [e for e in events if isinstance(e, PhaseTransitionEvent)]
    assert len(phase_events) == 1
    assert phase_events[0].old_phase == "active"
    assert phase_events[0].new_phase == "compacting"


def test_tagger_completion_publishes_phase_transition(tmp_path):
    """Tagger completes episode → phase transitions ingesting → active."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    from virtual_context.core.canonical_turns import utcnow_iso
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Seed one untagged row + set up running episode.
    now = utcnow_iso()
    with inner._get_conn() as conn:
        conn.execute("""
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES ('t0', ?, 'h0', 1, 'u','a','u_raw','a_raw', 1000.0, 'b', ?, ?, 1, NULL, ?, ?)
        """, (conv_id, now, now, now, now))
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state._tagger_run()
    phase_events = [e for e in events if isinstance(e, PhaseTransitionEvent)]
    assert any(
        e.old_phase == "ingesting" and e.new_phase == "active"
        for e in phase_events
    )
