import sqlite3
import pytest
from pathlib import Path
from virtual_context.storage.sqlite import SQLiteStore


def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def test_ingestion_episode_table_exists(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    with store._get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ingestion_episode)")}
    expected = {
        "episode_id", "conversation_id", "lifecycle_epoch",
        "raw_payload_entries", "started_at", "completed_at",
        "status", "owner_worker_id", "heartbeat_ts",
    }
    missing = expected - cols
    assert not missing, f"Missing: {missing}"


def test_ingestion_episode_partial_unique_index_enforces_single_running(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    now = _utcnow()
    with store._get_conn() as conn:
        conn.execute("""
            INSERT INTO conversations (conversation_id, tenant_id, created_at, updated_at)
            VALUES ('c1', 't1', ?, ?)
        """, (now, now))
        conn.execute("""
            INSERT INTO ingestion_episode (
                episode_id, conversation_id, lifecycle_epoch,
                raw_payload_entries, started_at, status,
                owner_worker_id, heartbeat_ts
            ) VALUES ('e1', 'c1', 1, 100, ?, 'running', 'w1', ?)
        """, (now, now))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO ingestion_episode (
                    episode_id, conversation_id, lifecycle_epoch,
                    raw_payload_entries, started_at, status,
                    owner_worker_id, heartbeat_ts
                ) VALUES ('e2', 'c1', 1, 200, ?, 'running', 'w2', ?)
            """, (now, now))
        # A completed episode on same (conv, epoch) is fine.
        conn.execute("UPDATE ingestion_episode SET status = 'completed' WHERE episode_id = 'e1'")
        conn.execute("""
            INSERT INTO ingestion_episode (
                episode_id, conversation_id, lifecycle_epoch,
                raw_payload_entries, started_at, status,
                owner_worker_id, heartbeat_ts
            ) VALUES ('e3', 'c1', 1, 300, ?, 'running', 'w3', ?)
        """, (now, now))


def test_ingestion_episode_status_check_constraint_rejects_bogus_status(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    now = _utcnow()
    with store._get_conn() as conn:
        conn.execute("""
            INSERT INTO conversations (conversation_id, tenant_id, created_at, updated_at)
            VALUES ('c1', 't1', ?, ?)
        """, (now, now))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO ingestion_episode (
                    episode_id, conversation_id, lifecycle_epoch,
                    raw_payload_entries, started_at, status,
                    owner_worker_id, heartbeat_ts
                ) VALUES ('e1', 'c1', 1, 100, ?, 'bogus_status', 'w1', ?)
            """, (now, now))
