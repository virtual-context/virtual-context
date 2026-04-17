import pytest
from datetime import datetime, timezone
from pathlib import Path
from virtual_context.storage.sqlite import SQLiteStore


def _now():
    return datetime.now(timezone.utc).isoformat()


def test_read_progress_snapshot_raises_keyerror_for_unknown(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    with pytest.raises(KeyError):
        s.read_progress_snapshot("nonexistent")


def test_read_progress_snapshot_empty_conversation(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    snap = s.read_progress_snapshot("c")
    assert snap.conversation_id == "c"
    assert snap.lifecycle_epoch == 1
    assert snap.phase == "init"
    assert snap.total_ingestible == 0
    assert snap.done_ingestible == 0
    assert snap.last_raw_payload_entries == 0
    assert snap.last_ingestible_payload_entries == 0
    assert snap.active_episode is None
    assert snap.active_compaction is None


def test_read_progress_snapshot_derives_total_and_done_from_canonical(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    # Insert 3 canonical rows: 2 tagged, 1 untagged. All have covered_ingestible_entries=1.
    now = _now()
    with s._get_conn() as conn:
        for i, tagged in enumerate([True, True, False]):
            conn.execute("""
                INSERT INTO canonical_turns (
                    canonical_turn_id, conversation_id, turn_hash, hash_version,
                    normalized_user_text, normalized_assistant_text,
                    user_content, assistant_content,
                    sort_key, source_batch_id, first_seen_at, last_seen_at,
                    covered_ingestible_entries, tagged_at,
                    created_at, updated_at
                ) VALUES (?, 'c', ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, ?, ?, ?)
            """, (f"t{i}", f"h{i}", float((i + 1) * 1000), now, now, now if tagged else None, now, now))
    snap = s.read_progress_snapshot("c")
    assert snap.total_ingestible == 3
    assert snap.done_ingestible == 2


def test_read_progress_snapshot_includes_active_episode(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    now = _now()
    with s._get_conn() as conn:
        conn.execute("""
            INSERT INTO ingestion_episode (
                episode_id, conversation_id, lifecycle_epoch,
                raw_payload_entries, started_at, status, owner_worker_id, heartbeat_ts
            ) VALUES ('ep1', 'c', 1, 500, ?, 'running', 'workerA', ?)
        """, (now, now))
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode is not None
    assert snap.active_episode.episode_id == "ep1"
    assert snap.active_episode.raw_payload_entries == 500
    assert snap.active_episode.owner_worker_id == "workerA"


def test_read_progress_snapshot_skips_completed_episode(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    now = _now()
    with s._get_conn() as conn:
        conn.execute("""
            INSERT INTO ingestion_episode (
                episode_id, conversation_id, lifecycle_epoch,
                raw_payload_entries, started_at, completed_at,
                status, owner_worker_id, heartbeat_ts
            ) VALUES ('ep_old', 'c', 1, 500, ?, ?, 'completed', 'w', ?)
        """, (now, now, now))
    snap = s.read_progress_snapshot("c")
    assert snap.active_episode is None


def test_read_progress_snapshot_includes_active_compaction(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    now = _now()
    with s._get_conn() as conn:
        conn.execute("""
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, owner_worker_id, heartbeat_ts
            ) VALUES ('op1', 'c', 1, 2, 5, 'summarizing', 'running', ?, 'w', ?)
        """, (now, now))
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None
    assert snap.active_compaction.operation_id == "op1"
    assert snap.active_compaction.phase_name == "summarizing"
    assert snap.active_compaction.phase_index == 2
    assert snap.active_compaction.phase_count == 5
    assert snap.active_compaction.status == "running"


def test_read_progress_snapshot_queued_compaction_is_also_active(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    now = _now()
    with s._get_conn() as conn:
        conn.execute("""
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, owner_worker_id, heartbeat_ts
            ) VALUES ('op1', 'c', 1, 0, 5, 'init', 'queued', ?, 'w', ?)
        """, (now, now))
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None
    assert snap.active_compaction.status == "queued"


def test_read_progress_snapshot_reads_request_metadata(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    with s._get_conn() as conn:
        conn.execute("""
            UPDATE conversations
               SET last_raw_payload_entries = 1000,
                   last_ingestible_payload_entries = 400
             WHERE conversation_id = 'c'
        """)
    snap = s.read_progress_snapshot("c")
    assert snap.last_raw_payload_entries == 1000
    assert snap.last_ingestible_payload_entries == 400


def test_read_progress_snapshot_returns_frozen_dataclass(tmp_path: Path):
    from dataclasses import FrozenInstanceError
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    snap = s.read_progress_snapshot("c")
    with pytest.raises(FrozenInstanceError):
        snap.phase = "ingesting"
