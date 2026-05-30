"""Compaction-fence Phase 2 tests for drain_compaction_exit.

Covers fencing plan §4.4 Phase 2 (operation-id fenced drain):

* T2.1 owner drain succeeds: terminal op + no successor -> phase flips
  to active or ingesting.
* T2.2 loser drain skips: expected_operation_id mismatch -> return None,
  phase unchanged.
* T2.3 successor present: terminal op exists for caller but another
  worker has a running op -> return None.
* T2.4 same-worker successor blocks: caller terminalized op X, then
  inserted op Y as running -> return None.
* T2.5 epoch mismatch -> return None.
* T2.6 phase already not 'compacting' -> return None.
* Legacy callers (None expected_operation_id) still work for
  backward-compat with non-fenced exit paths.

PG smoke tests T2.7-T2.8 are deferred outside this patch.
"""

from __future__ import annotations

import tempfile
import uuid

import pytest

from virtual_context.storage.sqlite import SQLiteStore


def _make_store():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return SQLiteStore(handle.name)


def _seed_compacting_conv_with_terminal_op(
    store, conv_id: str, *, worker_id: str, operation_id: str,
    terminal_status: str = "completed",
):
    """Seed a conv in 'compacting' phase + a terminal op owned by worker_id.

    This mirrors the post-complete_compaction_operation state that
    exit_compaction reaches just before calling drain_compaction_exit.
    """
    ts = "2026-05-29T00:00:00+00:00"
    conn = store._get_conn()
    conn.execute(
        "INSERT INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv_id, ts),
    )
    conn.execute(
        "INSERT INTO conversations "
        "(conversation_id, tenant_id, phase, lifecycle_epoch, "
        " created_at, updated_at) "
        "VALUES (?, 't1', 'compacting', 1, ?, ?)",
        (conv_id, ts, ts),
    )
    conn.execute(
        """INSERT INTO compaction_operation
           (operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, heartbeat_ts, owner_worker_id, created_at,
            completed_at)
           VALUES (?, ?, 1, 6, 7, 'tag_summaries', ?,
                   ?, ?, ?, ?, ?)""",
        (operation_id, conv_id, terminal_status, ts, ts, worker_id, ts, ts),
    )
    conn.commit()


def _phase_of(store, conv_id: str) -> str | None:
    conn = store._get_conn()
    row = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = ?",
        (conv_id,),
    ).fetchone()
    return row[0] if row else None


class TestT21_OwnerDrainSucceeds:
    """T2.1: owner drain with matching expected_operation_id succeeds."""

    def test_drain_owner_flips_phase_to_active(self):
        store = _make_store()
        conv = "conv-t2-1"
        op = uuid.uuid4().hex
        _seed_compacting_conv_with_terminal_op(
            store, conv, worker_id="w-1", operation_id=op,
        )

        result = store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            expected_operation_id=op,
        )
        # No untagged canonical_turns and no live op -> phase 'active'.
        assert result == "active"
        assert _phase_of(store, conv) == "active"


class TestT22_LoserDrainSkips:
    """T2.2: caller passes expected_operation_id NOT owned by this worker.

    The drain finds no terminal row matching (operation_id,
    owner_worker_id) and returns None; the conversations.phase stays at
    'compacting' so the legitimate owner's later drain can succeed.
    """

    def test_drain_loser_mismatched_op_returns_none(self):
        store = _make_store()
        conv = "conv-t2-2"
        owner_op = uuid.uuid4().hex
        loser_op = uuid.uuid4().hex
        _seed_compacting_conv_with_terminal_op(
            store, conv, worker_id="w-owner", operation_id=owner_op,
        )

        result = store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-loser",
            expected_operation_id=loser_op,
        )
        assert result is None
        assert _phase_of(store, conv) == "compacting"


class TestT23_SuccessorBlocksDrain:
    """T2.3: terminal op exists for caller but a successor is at
    status='running'. The no-active-successor guard blocks the drain.
    """

    def test_drain_blocked_by_running_successor(self):
        store = _make_store()
        conv = "conv-t2-3"
        terminal_op = uuid.uuid4().hex
        successor_op = uuid.uuid4().hex
        _seed_compacting_conv_with_terminal_op(
            store, conv, worker_id="w-1", operation_id=terminal_op,
        )
        # Successor: a fresh running op from a takeover.
        ts = "2026-05-29T00:01:00+00:00"
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (successor_op, conv, ts, ts, "w-takeover", ts),
        )
        conn.commit()

        result = store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            expected_operation_id=terminal_op,
        )
        assert result is None
        assert _phase_of(store, conv) == "compacting"


class TestT24_SameWorkerSuccessorBlocksDrain:
    """T2.4: terminal op X plus same-worker running successor Y blocks.

    This catches regressions that accidentally exclude the caller's
    worker_id from the no-active-successor guard.
    """

    def test_drain_blocked_by_same_worker_running_successor(self):
        store = _make_store()
        conv = "conv-t2-4"
        terminal_op = uuid.uuid4().hex
        successor_op = uuid.uuid4().hex
        _seed_compacting_conv_with_terminal_op(
            store, conv, worker_id="w-1", operation_id=terminal_op,
        )
        ts = "2026-05-29T00:01:00+00:00"
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (successor_op, conv, ts, ts, "w-1", ts),
        )
        conn.commit()

        result = store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            expected_operation_id=terminal_op,
        )
        assert result is None
        assert _phase_of(store, conv) == "compacting"


class TestT25_EpochMismatchSkips:
    """T2.5: caller's lifecycle_epoch != conv's lifecycle_epoch."""

    def test_drain_stale_epoch_returns_none(self):
        store = _make_store()
        conv = "conv-t2-5"
        op = uuid.uuid4().hex
        _seed_compacting_conv_with_terminal_op(
            store, conv, worker_id="w-1", operation_id=op,
        )

        result = store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=999,
            worker_id="w-1",
            expected_operation_id=op,
        )
        assert result is None
        assert _phase_of(store, conv) == "compacting"


class TestT26_PhaseAlreadyNotCompacting:
    """T2.6: a peer already drained; phase is now 'active'. Owner-guarded
    drain must observe the mismatch and skip the phase advance.
    """

    def test_drain_skips_when_phase_not_compacting(self):
        store = _make_store()
        conv = "conv-t2-6"
        op = uuid.uuid4().hex
        _seed_compacting_conv_with_terminal_op(
            store, conv, worker_id="w-1", operation_id=op,
        )
        # Simulate a peer already drained: phase flipped to active.
        conn = store._get_conn()
        conn.execute(
            "UPDATE conversations SET phase = 'active' "
            "WHERE conversation_id = ?",
            (conv,),
        )
        conn.commit()

        result = store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
            expected_operation_id=op,
        )
        assert result is None
        assert _phase_of(store, conv) == "active"


class TestLegacyCallerNoneSucceeds:
    """When expected_operation_id is None, the drain runs without the
    terminal-op guard but still acquires the lifecycle lock and runs
    the no-active-successor guard. Preserves backward-compat with the
    pre-fence drain contract.
    """

    def test_drain_with_none_skips_owner_guard(self):
        store = _make_store()
        conv = "conv-t2-legacy"
        op = uuid.uuid4().hex
        _seed_compacting_conv_with_terminal_op(
            store, conv, worker_id="w-1", operation_id=op,
        )

        result = store.drain_compaction_exit(
            conversation_id=conv,
            lifecycle_epoch=1,
            worker_id="w-1",
        )
        assert result == "active"
        assert _phase_of(store, conv) == "active"
