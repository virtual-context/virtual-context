import threading

import pytest
from pathlib import Path
from virtual_context.storage.sqlite import SQLiteStore


def test_upsert_conversation_creates_row(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT tenant_id, lifecycle_epoch, phase FROM conversations WHERE conversation_id='c'"
        ).fetchone()
    assert row is not None
    assert row[0] == "t"
    assert row[1] == 1  # default epoch
    assert row[2] == "init"  # default phase


def test_upsert_conversation_is_idempotent(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.upsert_conversation(tenant_id="t", conversation_id="c")  # no error
    with s._get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE conversation_id='c'"
        ).fetchone()[0]
    assert count == 1


def test_get_lifecycle_epoch_returns_current(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    assert s.get_lifecycle_epoch("c") == 1


def test_get_lifecycle_epoch_raises_keyerror_for_unknown(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    with pytest.raises(KeyError):
        s.get_lifecycle_epoch("nonexistent")


def test_mark_conversation_deleted_sets_phase_and_deleted_at(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.mark_conversation_deleted("c")
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT phase, deleted_at FROM conversations WHERE conversation_id='c'"
        ).fetchone()
    assert row[0] == "deleted"
    assert row[1] is not None  # deleted_at is set


def test_mark_conversation_deleted_terminalizes_active_episode_and_compaction(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    with s._get_conn() as conn:
        conn.execute("""
            INSERT INTO ingestion_episode (
                episode_id, conversation_id, lifecycle_epoch,
                raw_payload_entries, started_at, status, owner_worker_id, heartbeat_ts
            ) VALUES ('ep1', 'c', 1, 96, '2026-04-17T00:00:00+00:00', 'running', 'w1', '2026-04-17T00:00:00+00:00')
        """)
        conn.execute("""
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, owner_worker_id, heartbeat_ts
            ) VALUES ('op1', 'c', 1, 0, 3, 'queued', 'queued', '2026-04-17T00:00:00+00:00', 'w1', '2026-04-17T00:00:00+00:00')
        """)
    s.mark_conversation_deleted("c")
    with s._get_conn() as conn:
        ep = conn.execute("SELECT status, completed_at FROM ingestion_episode WHERE episode_id='ep1'").fetchone()
        op = conn.execute("SELECT status, completed_at FROM compaction_operation WHERE operation_id='op1'").fetchone()
    assert ep[0] == "abandoned"
    assert ep[1] is not None
    assert op[0] == "cancelled"
    assert op[1] is not None


def test_increment_lifecycle_epoch_bumps_on_resurrect(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.mark_conversation_deleted("c")
    new_epoch = s.increment_lifecycle_epoch_on_resurrect("c")
    assert new_epoch == 2
    assert s.get_lifecycle_epoch("c") == 2
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT phase, deleted_at FROM conversations WHERE conversation_id='c'"
        ).fetchone()
    assert row[0] == "init"
    assert row[1] is None  # deleted_at cleared


def test_increment_lifecycle_epoch_resurrect_cleans_stale_active_rows(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    with s._get_conn() as conn:
        conn.execute("""
            INSERT INTO ingestion_episode (
                episode_id, conversation_id, lifecycle_epoch,
                raw_payload_entries, started_at, status, owner_worker_id, heartbeat_ts
            ) VALUES ('ep1', 'c', 1, 96, '2026-04-17T00:00:00+00:00', 'running', 'w1', '2026-04-17T00:00:00+00:00')
        """)
        conn.execute("""
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, owner_worker_id, heartbeat_ts
            ) VALUES ('op1', 'c', 1, 0, 3, 'queued', 'running', '2026-04-17T00:00:00+00:00', 'w1', '2026-04-17T00:00:00+00:00')
        """)
    s.mark_conversation_deleted("c")
    assert s.increment_lifecycle_epoch_on_resurrect("c") == 2
    with s._get_conn() as conn:
        ep = conn.execute("SELECT status, completed_at FROM ingestion_episode WHERE episode_id='ep1'").fetchone()
        op = conn.execute("SELECT status, completed_at FROM compaction_operation WHERE operation_id='op1'").fetchone()
    assert ep[0] == "abandoned"
    assert ep[1] is not None
    assert op[0] == "cancelled"
    assert op[1] is not None


def test_repeat_resurrect_does_not_double_bump(tmp_path: Path):
    """Second resurrect on an already-init conversation is a no-op; returns
    the current epoch without incrementing again (TOCTOU-safe guard)."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.mark_conversation_deleted("c")
    e1 = s.increment_lifecycle_epoch_on_resurrect("c")  # -> 2
    e2 = s.increment_lifecycle_epoch_on_resurrect("c")  # second call: phase is now 'init', not 'deleted'
    assert e1 == 2
    assert e2 == 2  # NOT 3


def test_concurrent_resurrect_does_not_double_bump(tmp_path: Path):
    """Two concurrent resurrects — the BEGIN IMMEDIATE serialization
    ensures only one bumps. Both return the same epoch."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.mark_conversation_deleted("c")

    barrier = threading.Barrier(2)
    results: list[int] = []
    lock = threading.Lock()

    def run() -> None:
        barrier.wait()
        e = s.increment_lifecycle_epoch_on_resurrect("c")
        with lock:
            results.append(e)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    assert results[0] == 2 and results[1] == 2  # both see the same bumped epoch
    assert s.get_lifecycle_epoch("c") == 2  # DB has epoch=2, not 3


def test_increment_on_never_deleted_conversation_is_noop(tmp_path: Path):
    """Calling resurrect on a conversation that was never deleted should not bump."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    e = s.increment_lifecycle_epoch_on_resurrect("c")
    assert e == 1  # unchanged


def test_increment_raises_keyerror_for_unknown(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    with pytest.raises(KeyError):
        s.increment_lifecycle_epoch_on_resurrect("nonexistent")


def test_mark_conversation_deleted_raises_keyerror_for_unknown(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    with pytest.raises(KeyError):
        s.mark_conversation_deleted("nonexistent")
