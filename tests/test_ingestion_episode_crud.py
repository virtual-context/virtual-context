import pytest
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from virtual_context.storage.sqlite import SQLiteStore


def _fresh(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    return s


def _seed_canonical(s: SQLiteStore, *, conv_id: str = "c", count: int = 0, tagged: int = 0):
    """Insert `count` canonical rows; first `tagged` of them have tagged_at set."""
    from virtual_context.core.canonical_turns import utcnow_iso
    now = utcnow_iso()
    with s._get_conn() as conn:
        for i in range(count):
            conn.execute("""
                INSERT INTO canonical_turns (
                    canonical_turn_id, conversation_id, turn_hash, hash_version,
                    normalized_user_text, normalized_assistant_text,
                    user_content, assistant_content,
                    sort_key, source_batch_id, first_seen_at, last_seen_at,
                    covered_ingestible_entries, tagged_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, ?, ?, ?)
            """, (
                f"t{i}", conv_id, f"h{i}", float((i + 1) * 1000),
                now, now, now if i < tagged else None, now, now,
            ))


def test_upsert_creates_row_with_initial_worker(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode is not None
    assert snap.active_episode.owner_worker_id == "w1"
    assert snap.active_episode.raw_payload_entries == 100


def test_upsert_on_conflict_widens_raw_via_max_and_preserves_owner(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w2", raw_payload_entries=50)  # smaller, different worker
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode.raw_payload_entries == 100  # MAX, not 50
    assert snap.active_episode.owner_worker_id == "w1"  # unchanged


def test_upsert_widens_to_larger(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w2", raw_payload_entries=500)
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode.raw_payload_entries == 500  # widened
    assert snap.active_episode.owner_worker_id == "w1"  # still w1


def test_claim_succeeds_when_caller_already_owns(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    assert s.claim_ingestion_lease(conversation_id="c", lifecycle_epoch=1,
                                   worker_id="w1", lease_ttl_s=30.0) is True


def test_claim_fails_when_other_worker_holds_fresh_lease(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    assert s.claim_ingestion_lease(conversation_id="c", lifecycle_epoch=1,
                                   worker_id="w2", lease_ttl_s=30.0) is False


def test_claim_succeeds_when_other_worker_lease_is_stale(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    # Force stale heartbeat.
    with s._get_conn() as conn:
        conn.execute(
            "UPDATE ingestion_episode SET heartbeat_ts = '2000-01-01T00:00:00+00:00'"
            " WHERE conversation_id = 'c'"
        )
    assert s.claim_ingestion_lease(conversation_id="c", lifecycle_epoch=1,
                                   worker_id="w2", lease_ttl_s=30.0) is True
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode.owner_worker_id == "w2"


def test_claim_fails_on_different_epoch(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    # Attempt to claim at epoch=2 — should fail (no running row at epoch 2).
    assert s.claim_ingestion_lease(conversation_id="c", lifecycle_epoch=2,
                                   worker_id="w1", lease_ttl_s=30.0) is False


def test_refresh_heartbeat_succeeds_for_owner(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    assert s.refresh_ingestion_heartbeat(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is True


def test_refresh_heartbeat_fails_for_non_owner(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    assert s.refresh_ingestion_heartbeat(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
    ) is False


def test_refresh_heartbeat_fails_on_stale_epoch(tmp_path):
    """SQL-level epoch guard: stale caller cannot refresh new lifecycle."""
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    # Simulate a resurrect → new episode at epoch 2 owned by same worker.
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=2,
                               worker_id="w1", raw_payload_entries=50)
    # Stale thread thinks epoch=1, tries to refresh. Must fail.
    assert s.refresh_ingestion_heartbeat(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is False


def test_complete_fails_when_untagged_rows_remain(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    _seed_canonical(s, conv_id="c", count=3, tagged=1)  # 2 untagged
    assert s.complete_ingestion_episode(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is False
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode is not None  # still running


def test_complete_succeeds_when_all_rows_tagged(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    _seed_canonical(s, conv_id="c", count=3, tagged=3)  # all tagged
    assert s.complete_ingestion_episode(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is True
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode is None  # completed, no longer active


def test_complete_succeeds_when_no_canonical_rows(tmp_path):
    """Empty conversation — NOT EXISTS (no untagged rows) is vacuously true."""
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=0)
    assert s.complete_ingestion_episode(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is True


def test_complete_fails_for_non_owner(tmp_path):
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    assert s.complete_ingestion_episode(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
    ) is False


def test_complete_fails_on_stale_epoch(tmp_path):
    """Stale caller carrying epoch=1 cannot complete an epoch=2 episode."""
    s = _fresh(tmp_path)
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=1,
                               worker_id="w1", raw_payload_entries=100)
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    s.upsert_ingestion_episode(conversation_id="c", lifecycle_epoch=2,
                               worker_id="w1", raw_payload_entries=50)
    assert s.complete_ingestion_episode(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is False
