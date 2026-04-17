"""Task A35: Delete/resurrect + epoch-race integration tests.

Validates that every stale-thread protection mechanism works end-to-end.
Tests the invariants:

1. Stale ProxyState (cached lifecycle_epoch=1) cannot affect new lifecycle
   (epoch=2) after resurrect.
2. SQL-layer epoch guards reject stale writes even if verify_epoch races
   past.
3. Phase writes cannot stomp a new lifecycle.
4. Tagger exits cleanly on epoch change.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch


def test_stale_thread_cannot_stomp_phase_after_resurrect(tmp_path):
    """A thread that finished its epoch-scoped owner work cannot
    write phase onto a new lifecycle after a resurrect landed."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    # Resurrect bumps DB to epoch=2; engine's cached epoch is still 1.
    inner.mark_conversation_deleted(conv_id)
    inner.increment_lifecycle_epoch_on_resurrect(conv_id)

    # Stale thread (engine cache = epoch 1) tries to stomp phase.
    ok = inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")
    assert ok is False  # SQL-level guard rejected
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.lifecycle_epoch == 2
    assert snap.phase == "init"  # new lifecycle untouched

    # drain_compaction_exit on stale epoch also returns None.
    result = inner.drain_compaction_exit(
        conversation_id=conv_id, lifecycle_epoch=1, worker_id="stale_worker",
    )
    assert result is None


def test_stale_tagger_sql_guards_reject_fetch_and_mark(tmp_path):
    """SQL-layer defense: stale-epoch iter_untagged returns empty,
    mark_canonical_row_tagged returns False."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    from virtual_context.core.canonical_turns import utcnow_iso
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    now = utcnow_iso()
    # Seed an untagged row at epoch=1.
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
    # Resurrect bumps to epoch 2.
    inner.mark_conversation_deleted(conv_id)
    inner.increment_lifecycle_epoch_on_resurrect(conv_id)
    # Stale fetch returns empty.
    rows = inner.iter_untagged_canonical_rows(
        conversation_id=conv_id, expected_lifecycle_epoch=1, batch_size=10,
    )
    assert rows == []
    # Stale mark returns False.
    ok = inner.mark_canonical_row_tagged(
        canonical_turn_id="t0", conversation_id=conv_id, expected_lifecycle_epoch=1,
    )
    assert ok is False


def test_stale_tagger_loop_exits_cleanly_on_epoch_change(tmp_path):
    """After the tagger thread is spawned, an external resurrect
    bumps the epoch. The tagger's verify_epoch boundary check detects
    it and exits without writes."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    from virtual_context.core.canonical_turns import utcnow_iso
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    now = utcnow_iso()
    # Seed 3 untagged rows.
    with inner._get_conn() as conn:
        for i in range(3):
            conn.execute("""
                INSERT INTO canonical_turns (
                    canonical_turn_id, conversation_id, turn_hash, hash_version,
                    normalized_user_text, normalized_assistant_text,
                    user_content, assistant_content,
                    sort_key, source_batch_id, first_seen_at, last_seen_at,
                    covered_ingestible_entries, tagged_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, NULL, ?, ?)
            """, (f"t{i}", conv_id, f"h{i}", float((i + 1) * 1000), now, now, now, now))
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    # External resurrect BEFORE tagger runs.
    inner.mark_conversation_deleted(conv_id)
    inner.increment_lifecycle_epoch_on_resurrect(conv_id)
    # engine's cached lifecycle_epoch is still 1.

    # Run tagger; should detect epoch mismatch at first verify_epoch and exit.
    state._tagger_run()

    # Stale rows (epoch=1) are not in the new lifecycle's ledger anyway —
    # BUT the rows weren't purged by increment_lifecycle_epoch_on_resurrect
    # (that's a Phase B/deployment-level detail). Verify: the rows are still there
    # with tagged_at=NULL because the stale tagger couldn't touch them.
    with inner._get_conn() as conn:
        untagged = conn.execute(
            "SELECT COUNT(*) FROM canonical_turns WHERE conversation_id = ? AND tagged_at IS NULL",
            (conv_id,),
        ).fetchone()[0]
    assert untagged == 3  # nothing was tagged


def test_double_resurrect_is_idempotent(tmp_path):
    """Two concurrent threads calling increment_lifecycle_epoch_on_resurrect —
    only ONE bump happens (second returns current, already-bumped epoch)."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.mark_conversation_deleted(conv_id)
    e1 = inner.increment_lifecycle_epoch_on_resurrect(conv_id)
    e2 = inner.increment_lifecycle_epoch_on_resurrect(conv_id)
    assert e1 == 2
    assert e2 == 2  # NOT 3


def test_resurrect_clears_phase_to_init(tmp_path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Start in 'ingesting' to test that resurrect resets to 'init'.
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    inner.mark_conversation_deleted(conv_id)
    inner.increment_lifecycle_epoch_on_resurrect(conv_id)
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "init"
    assert snap.lifecycle_epoch == 2


def test_handle_prepare_payload_verifies_epoch_and_raises(tmp_path):
    """Top-level integration: a stale handle calling handle_prepare_payload
    after external resurrect raises LifecycleEpochMismatch on entry."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.mark_conversation_deleted(conv_id)
    inner.increment_lifecycle_epoch_on_resurrect(conv_id)
    # engine._engine_state.lifecycle_epoch still 1; DB at 2.
    with pytest.raises(LifecycleEpochMismatch):
        state.handle_prepare_payload(
            body={},
            payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
        )
