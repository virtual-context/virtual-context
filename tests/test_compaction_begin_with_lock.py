"""Compaction-fence Phase 1 tests for begin_compaction_with_lock.

Covers fencing plan §3.5 Phase 1 (lifecycle-locked begin primitive):

* T1.1 begin succeeds against an `active` conv with matching epoch.
* T1.2 loser race: two callers, only one wins.
* T1.3 phase mismatch: conv at `deleted` -> begin returns False.
* T1.4 epoch mismatch: caller's epoch != conv's epoch -> begin returns
  False.
* T1.5 required_phase kwarg lets the sweeper claim adapter require
  ``phase='active'`` specifically.
* T1.6 pre_begin_check callable runs under the held lock and aborts
  on False / raise.
* T1.7 cleanup_abandoned_compaction acquires the lifecycle lock before
  INSERTing new_op so a concurrent begin cannot slip in between the
  abandon UPDATE and the new-op INSERT.

PG smoke tests T1.8-T1.9 live in tests/test_compaction_begin_with_lock_postgres.py
gated on DATABASE_URL per the existing project convention.
"""

from __future__ import annotations

import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from virtual_context.storage.sqlite import SQLiteStore


def _make_store():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return SQLiteStore(handle.name)


def _seed_active_conv(store, conv_id: str, *, tenant_id: str = "t1"):
    # The conversations table requires created_at + updated_at; seed
    # with the same constant so deletes_at remains NULL and the row
    # passes the begin primitive's deleted/phase predicates.
    conn = store._get_conn()
    ts = "2026-05-29T00:00:00Z"
    conn.execute(
        "INSERT INTO conversation_lifecycle (conversation_id, generation, "
        "deleted, updated_at) VALUES (?, 0, 0, ?)",
        (conv_id, ts),
    )
    conn.execute(
        "INSERT INTO conversations (conversation_id, tenant_id, phase, "
        "lifecycle_epoch, created_at, updated_at) "
        "VALUES (?, ?, 'active', 1, ?, ?)",
        (conv_id, tenant_id, ts, ts),
    )
    conn.commit()


def _phase_of(store, conv_id: str) -> str | None:
    conn = store._get_conn()
    row = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = ?",
        (conv_id,),
    ).fetchone()
    return row[0] if row else None


def _active_op_count(store, conv_id: str) -> int:
    conn = store._get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM compaction_operation "
        " WHERE conversation_id = ? "
        "   AND status IN ('queued', 'running')",
        (conv_id,),
    ).fetchone()
    return int(row[0]) if row else 0


class TestT11_BeginSucceedsHappyPath:
    """T1.1: begin against an active conv with matching epoch."""

    def test_begin_inserts_running_op_and_transitions_phase(self):
        store = _make_store()
        conv = "conv-t1-1"
        _seed_active_conv(store, conv)

        op_id = uuid.uuid4().hex
        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            new_operation_id=op_id,
            phase_count=7,
        )
        assert result is True
        assert _phase_of(store, conv) == "compacting"
        assert _active_op_count(store, conv) == 1


class TestT12_LoserRace:
    """T1.2: two callers concurrent; only one wins."""

    def test_concurrent_second_caller_returns_false(self):
        store = _make_store()
        conv = "conv-t1-2"
        _seed_active_conv(store, conv)

        first_op = uuid.uuid4().hex
        second_op = uuid.uuid4().hex
        first_holds_lock = threading.Event()
        release_first = threading.Event()

        def _hold_under_lock(conn):
            first_holds_lock.set()
            assert release_first.wait(timeout=5)
            return True

        def _first_begin():
            return store.begin_compaction_with_lock(
                conversation_id=conv,
                lifecycle_epoch=1,
                worker_id="w-first",
                new_operation_id=first_op,
                phase_count=7,
                pre_begin_check=_hold_under_lock,
            )

        def _second_begin():
            assert first_holds_lock.wait(timeout=5)
            return store.begin_compaction_with_lock(
                conversation_id=conv,
                lifecycle_epoch=1,
                worker_id="w-second",
                new_operation_id=second_op,
                phase_count=7,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(_first_begin)
            assert first_holds_lock.wait(timeout=5)
            second_future = pool.submit(_second_begin)
            release_first.set()
            first = first_future.result(timeout=5)
            second = second_future.result(timeout=5)

        assert first is True
        assert second is False
        assert _active_op_count(store, conv) == 1


class TestT13_DeletedConvRejects:
    """T1.3: conv at phase='deleted' rejects begin."""

    def test_deleted_phase_returns_false(self):
        store = _make_store()
        conv = "conv-t1-3"
        _seed_active_conv(store, conv)
        conn = store._get_conn()
        conn.execute(
            "UPDATE conversations SET phase = 'deleted' "
            "WHERE conversation_id = ?",
            (conv,),
        )
        conn.commit()

        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            new_operation_id=uuid.uuid4().hex,
            phase_count=7,
        )
        assert result is False
        assert _active_op_count(store, conv) == 0


class TestT14_EpochMismatchRejects:
    """T1.4: caller's epoch != conv's epoch rejects begin."""

    def test_stale_epoch_returns_false(self):
        store = _make_store()
        conv = "conv-t1-4"
        _seed_active_conv(store, conv)

        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=999,
            worker_id="w-1",
            new_operation_id=uuid.uuid4().hex,
            phase_count=7,
        )
        assert result is False
        assert _phase_of(store, conv) == "active"
        assert _active_op_count(store, conv) == 0


class TestT15_RequiredPhaseGate:
    """T1.5: required_phase lets the sweeper require phase='active'."""

    def test_required_phase_active_rejects_ingesting(self):
        store = _make_store()
        conv = "conv-t1-5a"
        _seed_active_conv(store, conv)
        conn = store._get_conn()
        conn.execute(
            "UPDATE conversations SET phase = 'ingesting' "
            "WHERE conversation_id = ?",
            (conv,),
        )
        conn.commit()

        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            new_operation_id=uuid.uuid4().hex,
            phase_count=7,
            required_phase="active",
        )
        assert result is False
        assert _phase_of(store, conv) == "ingesting"
        assert _active_op_count(store, conv) == 0

    def test_required_phase_active_accepts_active(self):
        store = _make_store()
        conv = "conv-t1-5b"
        _seed_active_conv(store, conv)

        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            new_operation_id=uuid.uuid4().hex,
            phase_count=7,
            required_phase="active",
        )
        assert result is True
        assert _phase_of(store, conv) == "compacting"
        assert _active_op_count(store, conv) == 1


class TestT16_PreBeginCheckGate:
    """T1.6: pre_begin_check runs under the lock and aborts on False / raise."""

    def test_pre_begin_check_false_aborts_begin(self):
        store = _make_store()
        conv = "conv-t1-6a"
        _seed_active_conv(store, conv)

        def _reject(conn):
            return False

        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            new_operation_id=uuid.uuid4().hex,
            phase_count=7,
            pre_begin_check=_reject,
        )
        assert result is False
        assert _phase_of(store, conv) == "active"
        assert _active_op_count(store, conv) == 0

    def test_pre_begin_check_true_proceeds(self):
        store = _make_store()
        conv = "conv-t1-6b"
        _seed_active_conv(store, conv)

        def _accept(conn):
            return True

        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            new_operation_id=uuid.uuid4().hex,
            phase_count=7,
            pre_begin_check=_accept,
        )
        assert result is True
        assert _phase_of(store, conv) == "compacting"

    def test_pre_begin_check_raises_aborts_as_claim_lost(self):
        store = _make_store()
        conv = "conv-t1-6c"
        _seed_active_conv(store, conv)

        def _raise(conn):
            raise RuntimeError("synthetic")

        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            new_operation_id=uuid.uuid4().hex,
            phase_count=7,
            pre_begin_check=_raise,
        )
        assert result is False
        assert _phase_of(store, conv) == "active"


class TestT17_CleanupAcquiresLifecycleLock:
    """T1.7: cleanup_abandoned_compaction acquires the lifecycle lock
    before the abandon UPDATE + new-op INSERT.
    """

    def test_cleanup_inserts_new_op_after_abandon(self):
        store = _make_store()
        conv = "conv-t1-7"
        _seed_active_conv(store, conv)
        dead_op = uuid.uuid4().hex
        result = store.begin_compaction_with_lock(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-dead",
            new_operation_id=dead_op,
            phase_count=7,
        )
        assert result is True

        statements: list[str] = []
        conn = store._get_conn()
        conn.set_trace_callback(lambda stmt: statements.append(" ".join(stmt.split())))
        new_op = uuid.uuid4().hex
        try:
            cleanup_ok = store.cleanup_abandoned_compaction(
                conversation_id=conv,
                dead_operation_id=dead_op,
                new_operation_id=new_op,
                lifecycle_epoch=1,
                worker_id="w-new",
                phase_count=7,
            )
        finally:
            conn.set_trace_callback(None)
        assert cleanup_ok is True
        assert _active_op_count(store, conv) == 1
        upper = [stmt.upper() for stmt in statements]
        begin_idx = next(
            i for i, stmt in enumerate(upper)
            if stmt.startswith("BEGIN IMMEDIATE")
        )
        lock_idx = next(
            i for i, stmt in enumerate(upper)
            if "SELECT 1 FROM CONVERSATION_LIFECYCLE" in stmt
        )
        abandon_idx = next(
            i for i, stmt in enumerate(upper)
            if stmt.startswith("UPDATE COMPACTION_OPERATION")
        )
        insert_idx = next(
            i for i, stmt in enumerate(upper)
            if stmt.startswith("INSERT INTO COMPACTION_OPERATION")
        )
        assert begin_idx < lock_idx < abandon_idx < insert_idx
