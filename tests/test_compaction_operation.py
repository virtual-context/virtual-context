import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone
from virtual_context.storage.sqlite import SQLiteStore


def _now():
    return datetime.now(timezone.utc).isoformat()


def _seed_conv(store: SQLiteStore, conv_id="c1", tenant="t1"):
    now = _now()
    with store._get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, tenant_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (conv_id, tenant, now, now),
        )


def test_compaction_operation_table_exists(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    with store._get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(compaction_operation)")}
    expected = {
        "operation_id", "conversation_id", "lifecycle_epoch",
        "phase_index", "phase_count", "phase_name", "status",
        "started_at", "completed_at", "owner_worker_id",
        "heartbeat_ts", "error_message",
    }
    missing = expected - cols
    assert not missing, f"Missing: {missing}"


def test_compaction_operation_partial_unique_index_rejects_double_active(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    _seed_conv(store)
    now = _now()
    with store._get_conn() as conn:
        conn.execute("""
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, owner_worker_id, heartbeat_ts
            ) VALUES ('op1', 'c1', 1, 0, 3, 'init', 'running', ?, 'w1', ?)
        """, (now, now))
        # Second 'running' on same (conv, epoch) rejected
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO compaction_operation (
                    operation_id, conversation_id, lifecycle_epoch,
                    phase_index, phase_count, phase_name, status,
                    started_at, owner_worker_id, heartbeat_ts
                ) VALUES ('op2', 'c1', 1, 0, 3, 'init', 'queued', ?, 'w2', ?)
            """, (now, now))
        # After completion, new active row is allowed
        conn.execute("UPDATE compaction_operation SET status = 'completed' WHERE operation_id = 'op1'")
        conn.execute("""
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, owner_worker_id, heartbeat_ts
            ) VALUES ('op3', 'c1', 1, 0, 3, 'init', 'running', ?, 'w3', ?)
        """, (now, now))


def test_compaction_operation_status_check_constraint_rejects_bogus_status(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    _seed_conv(store)
    now = _now()
    with store._get_conn() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO compaction_operation (
                    operation_id, conversation_id, lifecycle_epoch,
                    phase_index, phase_count, phase_name, status,
                    started_at, owner_worker_id, heartbeat_ts
                ) VALUES ('op1', 'c1', 1, 0, 3, 'init', 'bogus', ?, 'w1', ?)
            """, (now, now))
