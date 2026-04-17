import pytest
from pathlib import Path
from virtual_context.core.progress_events import IngestionProgressEvent


def _seed_row(inner, conv_id, canonical_id, sort_key, tagged=False):
    from virtual_context.core.canonical_turns import utcnow_iso
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
            ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, ?, ?, ?)
        """, (canonical_id, conv_id, f"h_{canonical_id}", sort_key, now, now, now if tagged else None, now, now))


def test_tagger_publishes_ingestion_progress_per_row(tmp_path):
    """Tag 3 rows → 3 IngestionProgressEvents published; final event shows done==total."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    for i in range(3):
        _seed_row(inner, conv_id, f"t{i}", float((i + 1) * 1000))
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
    ingestion_events = [e for e in events if isinstance(e, IngestionProgressEvent)]
    assert len(ingestion_events) >= 3  # one per row
    last = ingestion_events[-1]
    assert last.done == 3
    assert last.total == 3
    assert last.kind == "ingestion"
    assert last.lifecycle_epoch == 1


def test_ingestion_event_carries_episode_id(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    _seed_row(inner, conv_id, "t0", 1000.0)
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    # Fetch episode_id for later comparison.
    snap_before = inner.read_progress_snapshot(conv_id)
    expected_episode_id = snap_before.active_episode.episode_id
    events = []
    state.engine.progress_event_bus.subscribe(events.append)
    state._tagger_run()
    ingestion_events = [e for e in events if isinstance(e, IngestionProgressEvent)]
    assert len(ingestion_events) >= 1
    # The event fires while the episode is still running — episode_id should match.
    assert ingestion_events[0].episode_id == expected_episode_id
