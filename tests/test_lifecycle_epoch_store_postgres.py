"""Tests for the lifecycle_epoch store API on the Postgres backend.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors
the SQLite tests in ``test_lifecycle_epoch_store.py`` so both backends stay
in lockstep on the upsert/get/mark-deleted/resurrect invariants and the
TOCTOU guard on ``phase='deleted'``.

Note: ``conversations.conversation_id`` is ``UUID PRIMARY KEY`` in Postgres
(see postgres.py:823), so test IDs use ``uuid.uuid4()`` rather than free-form
strings. UUIDs are generated per-test to keep the suite idempotent across
reruns against a shared test database.
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
