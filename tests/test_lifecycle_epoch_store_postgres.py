"""Tests for the lifecycle_epoch store API on the Postgres backend.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors
the SQLite tests in ``test_lifecycle_epoch_store.py`` so both backends stay
in lockstep on the upsert/get/mark-deleted/resurrect invariants and the
TOCTOU guard on ``phase='deleted'``.

Note: ``conversations.conversation_id`` is ``TEXT PRIMARY KEY`` in Postgres
(aligned with ``canonical_turns.conversation_id``). Test conversation IDs use
``str(uuid.uuid4())`` per-test to keep the suite idempotent across reruns
against a shared test database.
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def _store():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    return PostgresStore(PG_URL)


def _cid() -> str:
    return str(uuid.uuid4())


def test_upsert_conversation_creates_row_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    assert s.get_lifecycle_epoch(cid) == 1


def test_upsert_conversation_is_idempotent_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    assert s.get_lifecycle_epoch(cid) == 1
    conn = s._get_conn()
    count_row = conn.execute(
        "SELECT COUNT(*) AS c FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    assert int(count_row["c"]) == 1


def test_get_lifecycle_epoch_raises_keyerror_for_unknown_pg():
    s = _store()
    with pytest.raises(KeyError):
        s.get_lifecycle_epoch(_cid())


def test_mark_conversation_deleted_sets_phase_and_deleted_at_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    s.mark_conversation_deleted(cid)
    conn = s._get_conn()
    row = conn.execute(
        "SELECT phase, deleted_at FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    assert row["phase"] == "deleted"
    assert row["deleted_at"] is not None


def test_mark_conversation_deleted_terminalizes_active_episode_and_compaction_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    conn = s._get_conn()
    ep_id = str(uuid.uuid4())
    op_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO ingestion_episode (
            episode_id, conversation_id, lifecycle_epoch,
            raw_payload_entries, started_at, status, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 96, NOW(), 'running', 'w1', NOW())
        """,
        (ep_id, cid),
    )
    conn.execute(
        """
        INSERT INTO compaction_operation (
            operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 0, 3, 'queued', 'queued', NOW(), 'w1', NOW())
        """,
        (op_id, cid),
    )
    s.mark_conversation_deleted(cid)
    ep = conn.execute(
        "SELECT status, completed_at FROM ingestion_episode WHERE episode_id = %s",
        (ep_id,),
    ).fetchone()
    op = conn.execute(
        "SELECT status, completed_at FROM compaction_operation WHERE operation_id = %s",
        (op_id,),
    ).fetchone()
    assert ep["status"] == "abandoned"
    assert ep["completed_at"] is not None
    assert op["status"] == "cancelled"
    assert op["completed_at"] is not None


def test_mark_conversation_deleted_raises_keyerror_for_unknown_pg():
    s = _store()
    with pytest.raises(KeyError):
        s.mark_conversation_deleted(_cid())


def test_increment_lifecycle_epoch_bumps_on_resurrect_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    s.mark_conversation_deleted(cid)
    new_epoch = s.increment_lifecycle_epoch_on_resurrect(cid)
    assert new_epoch == 2
    assert s.get_lifecycle_epoch(cid) == 2
    conn = s._get_conn()
    row = conn.execute(
        "SELECT phase, deleted_at FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    assert row["phase"] == "init"
    assert row["deleted_at"] is None


def test_increment_lifecycle_epoch_resurrect_cleans_stale_active_rows_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    conn = s._get_conn()
    ep_id = str(uuid.uuid4())
    op_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO ingestion_episode (
            episode_id, conversation_id, lifecycle_epoch,
            raw_payload_entries, started_at, status, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 96, NOW(), 'running', 'w1', NOW())
        """,
        (ep_id, cid),
    )
    conn.execute(
        """
        INSERT INTO compaction_operation (
            operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 0, 3, 'queued', 'running', NOW(), 'w1', NOW())
        """,
        (op_id, cid),
    )
    s.mark_conversation_deleted(cid)
    assert s.increment_lifecycle_epoch_on_resurrect(cid) == 2
    ep = conn.execute(
        "SELECT status, completed_at FROM ingestion_episode WHERE episode_id = %s",
        (ep_id,),
    ).fetchone()
    op = conn.execute(
        "SELECT status, completed_at FROM compaction_operation WHERE operation_id = %s",
        (op_id,),
    ).fetchone()
    assert ep["status"] == "abandoned"
    assert ep["completed_at"] is not None
    assert op["status"] == "cancelled"
    assert op["completed_at"] is not None


def test_repeat_resurrect_does_not_double_bump_pg():
    """Second resurrect on an already-init conversation is a no-op; returns
    the current epoch without incrementing again (TOCTOU-safe guard)."""
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    s.mark_conversation_deleted(cid)
    e1 = s.increment_lifecycle_epoch_on_resurrect(cid)
    e2 = s.increment_lifecycle_epoch_on_resurrect(cid)
    assert e1 == 2
    assert e2 == 2  # NOT 3


def test_concurrent_resurrect_does_not_double_bump_pg():
    """Two concurrent resurrects via real threads — the atomic
    ``UPDATE ... WHERE phase='deleted' RETURNING`` guarantees only one bumps.
    Both return the same epoch."""
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    s.mark_conversation_deleted(cid)

    barrier = threading.Barrier(2)
    results: list[int] = []
    lock = threading.Lock()

    def run() -> None:
        barrier.wait()
        e = s.increment_lifecycle_epoch_on_resurrect(cid)
        with lock:
            results.append(e)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    assert results[0] == 2 and results[1] == 2  # both see the same bumped epoch
    assert s.get_lifecycle_epoch(cid) == 2  # DB has epoch=2, not 3


def test_increment_on_never_deleted_conversation_is_noop_pg():
    """Calling resurrect on a conversation that was never deleted should not bump."""
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    e = s.increment_lifecycle_epoch_on_resurrect(cid)
    assert e == 1


def test_increment_raises_keyerror_for_unknown_pg():
    s = _store()
    with pytest.raises(KeyError):
        s.increment_lifecycle_epoch_on_resurrect(_cid())


def test_delete_conversation_removes_lifecycle_and_progress_rows_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    s.save_canonical_turn(cid, 0, "u0", "a0")
    conn = s._get_conn()
    ep_id = str(uuid.uuid4())
    op_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO ingestion_episode (
            episode_id, conversation_id, lifecycle_epoch,
            raw_payload_entries, started_at, status, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 96, NOW(), 'running', 'w1', NOW())
        """,
        (ep_id, cid),
    )
    conn.execute(
        """
        INSERT INTO compaction_operation (
            operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 0, 3, 'queued', 'queued', NOW(), 'w1', NOW())
        """,
        (op_id, cid),
    )

    s.delete_conversation(cid)

    conv_count = conn.execute(
        "SELECT COUNT(*) AS c FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()["c"]
    episode_count = conn.execute(
        "SELECT COUNT(*) AS c FROM ingestion_episode WHERE conversation_id = %s",
        (cid,),
    ).fetchone()["c"]
    compaction_count = conn.execute(
        "SELECT COUNT(*) AS c FROM compaction_operation WHERE conversation_id = %s",
        (cid,),
    ).fetchone()["c"]
    canonical_count = conn.execute(
        "SELECT COUNT(*) AS c FROM canonical_turns WHERE conversation_id = %s",
        (cid,),
    ).fetchone()["c"]

    assert int(conv_count) == 0
    assert int(episode_count) == 0
    assert int(compaction_count) == 0
    assert int(canonical_count) == 0
