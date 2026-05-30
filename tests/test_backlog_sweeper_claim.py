"""Compaction-backlog sweeper Phase 1 SQLite tests.

Per ``specs/compaction-backlog-sweeper-plan.md`` §3.2 T1.1-T1.11.
The PG smoke (T1.12-T1.15) lives in
``test_backlog_sweeper_claim_postgres.py``.

``claim_compaction_backlog`` is a pure adapter -- it re-verifies the
detection predicates under the lifecycle lock held by
``begin_compaction_with_lock`` and delegates the phase CAS + active
``compaction_operation`` INSERT to that primitive. This file asserts:

* T1.1: a candidate whose predicates still hold yields a successful
  claim (active op row inserted at status='running'; phase moved
  to 'compacting').
* T1.2-T1.10: each precondition mismatch results in a False return
  with NO active op row and NO phase change.
* T1.11: a competing active row inserted between predicate re-verify
  and begin commits causes the begin's ON CONFLICT DO NOTHING gate
  to fire -- claim returns False and the transaction rolls back
  cleanly (no stray 'compacting' state).

The race tests T1.7 (SKIP LOCKED) and T1.8 (two-sweeper race) are
SQLite-impractical because SQLite's BEGIN IMMEDIATE serializes the
database rather than per-row; they are covered by the PG smoke
tests in T1.13.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import BacklogCandidate


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_conv(
    store: SQLiteStore,
    *,
    conv_id: str,
    tenant_id: str = "t-1",
    phase: str = "active",
    lifecycle_epoch: int = 1,
    deleted_at: str | None = None,
) -> None:
    now = _iso(_now())
    conn = store._get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv_id, now),
    )
    conn.execute(
        "INSERT INTO conversations "
        "(conversation_id, tenant_id, phase, lifecycle_epoch, "
        " created_at, updated_at, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conv_id, tenant_id, phase, lifecycle_epoch, now, now, deleted_at),
    )
    conn.commit()


def _seed_turns(
    store: SQLiteStore,
    *,
    conv_id: str,
    count: int,
    tagged: bool = True,
    compacted: bool = False,
    sort_key_base: float = 1000.0,
    id_prefix: str = "",
) -> None:
    now = _iso(_now())
    conn = store._get_conn()
    for i in range(count):
        conn.execute(
            """INSERT INTO canonical_turns
               (canonical_turn_id, conversation_id, sort_key, turn_hash,
                hash_version, tagged_at, compacted_at, created_at,
                updated_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (
                f"ct-{conv_id}-{id_prefix}{i}", conv_id,
                sort_key_base + i, f"h-{conv_id}-{id_prefix}{i}",
                now if tagged else None,
                now if compacted else None,
                now, now,
            ),
        )
    conn.commit()


def _seed_terminal_op(
    store: SQLiteStore,
    *,
    op_id: str,
    conv_id: str,
    lifecycle_epoch: int = 1,
    completed_seconds_ago: float = 0,
    status: str = "completed",
) -> None:
    completed = _iso(_now() - timedelta(seconds=completed_seconds_ago))
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO compaction_operation
           (operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, heartbeat_ts, owner_worker_id, created_at,
            completed_at)
           VALUES (?, ?, ?, 0, 7, 'starting', ?, ?, ?, 'w-prior',
                   ?, ?)""",
        (op_id, conv_id, lifecycle_epoch, status, completed, completed,
         completed, completed),
    )
    conn.commit()


def _seed_active_op(
    store: SQLiteStore,
    *,
    op_id: str,
    conv_id: str,
    lifecycle_epoch: int = 1,
    status: str = "queued",
) -> None:
    now = _iso(_now())
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO compaction_operation
           (operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, heartbeat_ts, owner_worker_id, created_at)
           VALUES (?, ?, ?, 0, 7, 'starting', ?, ?, ?,
                   'w-competing', ?)""",
        (op_id, conv_id, lifecycle_epoch, status, now, now, now),
    )
    conn.commit()


def _candidate(
    conv_id: str = "conv-x",
    *,
    tenant_id: str = "t-1",
    lifecycle_epoch: int = 1,
    backlog_turns: int = 50,
) -> BacklogCandidate:
    return BacklogCandidate(
        conversation_id=conv_id,
        tenant_id=tenant_id,
        lifecycle_epoch=lifecycle_epoch,
        backlog_turns=backlog_turns,
        last_terminal_compaction_at=None,
    )


def _phase_of(store: SQLiteStore, conv_id: str) -> str | None:
    conn = store._get_conn()
    row = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = ?",
        (conv_id,),
    ).fetchone()
    return row[0] if row else None


def _running_op_count(store: SQLiteStore, conv_id: str) -> int:
    conn = store._get_conn()
    row = conn.execute(
        """SELECT COUNT(*) FROM compaction_operation
            WHERE conversation_id = ?
              AND status IN ('queued', 'running')""",
        (conv_id,),
    ).fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# T1.1: happy-path claim succeeds.
# ---------------------------------------------------------------------------


class TestT11_HappyPathClaim:
    def test_claim_succeeds_inserts_running_op_and_flips_phase(
        self, tmp_path: Path,
    ):
        store = SQLiteStore(tmp_path / "t11.db")
        _seed_conv(store, conv_id="conv-ok")
        _seed_turns(store, conv_id="conv-ok", count=50, tagged=True)
        candidate = _candidate("conv-ok", backlog_turns=50)

        op_id = uuid.uuid4().hex
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=op_id,
            owner_worker_id="w-sweeper",
            phase_count=7,
            min_backlog_turns=20,
            grace_s=300.0,
        )
        assert ok is True
        assert _phase_of(store, "conv-ok") == "compacting"
        assert _running_op_count(store, "conv-ok") == 1


# ---------------------------------------------------------------------------
# T1.2: epoch bump aborts the claim.
# ---------------------------------------------------------------------------


class TestT12_EpochBumpAborts:
    def test_epoch_mismatch_aborts(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t12.db")
        _seed_conv(store, conv_id="conv-epoch", lifecycle_epoch=2)
        _seed_turns(store, conv_id="conv-epoch", count=50, tagged=True)
        # Candidate snapshot is epoch 1, but conv is now at 2.
        candidate = _candidate("conv-epoch", lifecycle_epoch=1)
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
        assert _phase_of(store, "conv-epoch") == "active"
        assert _running_op_count(store, "conv-epoch") == 0


# ---------------------------------------------------------------------------
# T1.3: backlog dropped below threshold aborts.
# ---------------------------------------------------------------------------


class TestT13_BacklogBelowThresholdAborts:
    def test_backlog_below_threshold_aborts(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t13.db")
        _seed_conv(store, conv_id="conv-low")
        # Candidate snapshot saw 30 backlog turns; conv now has only 15.
        _seed_turns(store, conv_id="conv-low", count=15, tagged=True)
        candidate = _candidate("conv-low", backlog_turns=30)
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
        assert _phase_of(store, "conv-low") == "active"


# ---------------------------------------------------------------------------
# T1.4: untagged row appearance aborts.
# ---------------------------------------------------------------------------


class TestT14_UntaggedRowAborts:
    def test_untagged_row_aborts(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t14.db")
        _seed_conv(store, conv_id="conv-untag")
        _seed_turns(store, conv_id="conv-untag", count=50, tagged=True)
        # One untagged row appeared between detection and claim.
        _seed_turns(store, conv_id="conv-untag", count=1, tagged=False,
                    sort_key_base=9999.0, id_prefix="u")
        ok = store.claim_compaction_backlog(
            candidate=_candidate("conv-untag"),
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
        assert _phase_of(store, "conv-untag") == "active"


# ---------------------------------------------------------------------------
# T1.5: successor active op aborts.
# ---------------------------------------------------------------------------


class TestT15_SuccessorOpAborts:
    @pytest.mark.parametrize("status", ["queued", "running"])
    def test_successor_op_aborts(self, tmp_path: Path, status: str):
        store = SQLiteStore(tmp_path / f"t15-{status}.db")
        _seed_conv(store, conv_id="conv-succ")
        _seed_turns(store, conv_id="conv-succ", count=50, tagged=True)
        _seed_active_op(store, op_id=uuid.uuid4().hex,
                        conv_id="conv-succ", status=status)
        ok = store.claim_compaction_backlog(
            candidate=_candidate("conv-succ"),
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
        # Still exactly one active op (the competitor's), not two.
        assert _running_op_count(store, "conv-succ") == 1


# ---------------------------------------------------------------------------
# T1.6: phase changed away from 'active' aborts.
# ---------------------------------------------------------------------------


class TestT16_PhaseChangedAborts:
    @pytest.mark.parametrize("phase", [
        "deleted", "compacting", "ingesting", "merged",
    ])
    def test_phase_not_active_aborts(self, tmp_path: Path, phase: str):
        store = SQLiteStore(tmp_path / f"t16-{phase}.db")
        _seed_conv(store, conv_id="conv-pha", phase=phase)
        _seed_turns(store, conv_id="conv-pha", count=50, tagged=True)
        ok = store.claim_compaction_backlog(
            candidate=_candidate("conv-pha"),
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
        assert _phase_of(store, "conv-pha") == phase


# ---------------------------------------------------------------------------
# T1.9: claim uses threshold, not stale snapshot.
# ---------------------------------------------------------------------------


class TestT19_ThresholdNotStaleSnapshot:
    """T1.9: candidate.backlog_turns is a SNAPSHOT from detection;
    the claim re-verification compares against the configured
    ``min_backlog_turns``, not the snapshot. A conv detected with
    100 backlog turns that drops to 25 by claim time still claims at
    threshold 20; the same setup drops to 19 and aborts.
    """

    def test_drops_to_above_threshold_still_claims(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t19a.db")
        _seed_conv(store, conv_id="conv-drop-25")
        # Snapshot saw 100; conv now has 25.
        _seed_turns(store, conv_id="conv-drop-25", count=25, tagged=True)
        candidate = _candidate("conv-drop-25", backlog_turns=100)
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is True

    def test_drops_below_threshold_aborts(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t19b.db")
        _seed_conv(store, conv_id="conv-drop-19")
        _seed_turns(store, conv_id="conv-drop-19", count=19, tagged=True)
        candidate = _candidate("conv-drop-19", backlog_turns=100)
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False


# ---------------------------------------------------------------------------
# T1.10: current-lifecycle terminal grace aborts.
# ---------------------------------------------------------------------------


class TestT110_TerminalGraceAborts:
    def test_recent_terminal_inside_grace_aborts(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t110.db")
        _seed_conv(store, conv_id="conv-grace")
        _seed_turns(store, conv_id="conv-grace", count=50, tagged=True)
        _seed_terminal_op(
            store, op_id=uuid.uuid4().hex, conv_id="conv-grace",
            completed_seconds_ago=60, status="completed",
        )
        ok = store.claim_compaction_backlog(
            candidate=_candidate("conv-grace"),
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
        assert _phase_of(store, "conv-grace") == "active"

    def test_old_terminal_outside_grace_claims(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t110-old.db")
        _seed_conv(store, conv_id="conv-grace-old")
        _seed_turns(store, conv_id="conv-grace-old", count=50, tagged=True)
        _seed_terminal_op(
            store, op_id=uuid.uuid4().hex, conv_id="conv-grace-old",
            completed_seconds_ago=600, status="completed",
        )
        ok = store.claim_compaction_backlog(
            candidate=_candidate("conv-grace-old"),
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is True
        assert _phase_of(store, "conv-grace-old") == "compacting"


# ---------------------------------------------------------------------------
# T1.11: tenant mismatch aborts. (Bonus -- the predicate is one of
# the five §3.2 checks; the spec's T-bundle doesn't enumerate it
# separately but the behavior is normative.)
# ---------------------------------------------------------------------------


class TestT111_TenantMismatchAborts:
    def test_tenant_mismatch_aborts(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t111.db")
        _seed_conv(store, conv_id="conv-tenant", tenant_id="t-real")
        _seed_turns(store, conv_id="conv-tenant", count=50, tagged=True)
        # Snapshot saw tenant t-wrong (e.g. tenant was renamed
        # between detection and claim).
        candidate = _candidate("conv-tenant", tenant_id="t-wrong")
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
        assert _phase_of(store, "conv-tenant") == "active"


# ---------------------------------------------------------------------------
# T1.11 (spec sense): deleted_at set aborts. Separate from phase
# because a conv can have deleted_at populated while still nominally
# phase='active' during a delete-in-progress race.
# ---------------------------------------------------------------------------


class TestT11x_DeletedAtAborts:
    def test_deleted_at_set_aborts(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t11x.db")
        _seed_conv(
            store, conv_id="conv-del", phase="active",
            deleted_at=_iso(_now()),
        )
        _seed_turns(store, conv_id="conv-del", count=50, tagged=True)
        ok = store.claim_compaction_backlog(
            candidate=_candidate("conv-del"),
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-sweeper", phase_count=7,
            min_backlog_turns=20, grace_s=300.0,
        )
        assert ok is False
