"""Tests for the SQLite compaction_operation CRUD API.

Covers the 5 methods — start, claim, advance, complete, fail — with
epoch-scoped guards in SQL (correlated subquery against
conversations.lifecycle_epoch on advance/complete/fail). The partial unique
index on ``(conversation_id, lifecycle_epoch) WHERE status IN
('queued','running')`` enforces a single active operation per
(conversation, epoch) so a second ``start`` on the same active window
raises ``sqlite3.IntegrityError``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore


def _fresh(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    return s


# ----------------------------------------------------------------------
# start_compaction_operation
# ----------------------------------------------------------------------

def test_start_creates_queued_operation(tmp_path: Path):
    s = _fresh(tmp_path)
    op_id = s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=5, phase_name="init",
    )
    assert isinstance(op_id, str)
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None
    assert snap.active_compaction.operation_id == op_id
    assert snap.active_compaction.status == "queued"
    assert snap.active_compaction.phase_index == 0
    assert snap.active_compaction.phase_count == 5
    assert snap.active_compaction.phase_name == "init"


def test_start_returns_uuid_string(tmp_path: Path):
    import uuid
    s = _fresh(tmp_path)
    op_id = s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Must parse as a UUID.
    uuid.UUID(op_id)


def test_start_fails_when_another_queued_operation_exists(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    with pytest.raises(sqlite3.IntegrityError):
        s.start_compaction_operation(
            conversation_id="c", lifecycle_epoch=1, worker_id="w2",
            phase_count=3, phase_name="init",
        )


def test_start_fails_when_running_operation_exists(tmp_path: Path):
    s = _fresh(tmp_path)
    op_id = s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Transition the existing operation to 'running'.
    assert s.advance_compaction_phase(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is True
    with pytest.raises(sqlite3.IntegrityError):
        s.start_compaction_operation(
            conversation_id="c", lifecycle_epoch=1, worker_id="w2",
            phase_count=3, phase_name="init",
        )


def test_start_succeeds_after_completion(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is True
    # New start now allowed (completed row no longer occupies the partial
    # unique index).
    new_id = s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
        phase_count=3, phase_name="init",
    )
    assert isinstance(new_id, str)


# ----------------------------------------------------------------------
# claim_compaction_lease
# ----------------------------------------------------------------------

def test_claim_succeeds_when_caller_already_owns(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.claim_compaction_lease(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        lease_ttl_s=30.0,
    ) is True


def test_claim_fails_when_other_worker_holds_fresh_lease(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.claim_compaction_lease(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
        lease_ttl_s=30.0,
    ) is False


def test_claim_succeeds_when_other_worker_lease_is_stale(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Force stale heartbeat so w2 can take over.
    with s._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET heartbeat_ts = '2000-01-01T00:00:00+00:00'"
            " WHERE conversation_id = 'c'"
        )
    assert s.claim_compaction_lease(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
        lease_ttl_s=30.0,
    ) is True
    with s._get_conn() as conn:
        owner = conn.execute(
            "SELECT owner_worker_id FROM compaction_operation WHERE conversation_id = 'c'"
        ).fetchone()[0]
    assert owner == "w2"


def test_claim_fails_on_different_epoch(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Attempt to claim at epoch=2 — should fail (no active row at epoch 2).
    assert s.claim_compaction_lease(
        conversation_id="c", lifecycle_epoch=2, worker_id="w1",
        lease_ttl_s=30.0,
    ) is False


# ----------------------------------------------------------------------
# advance_compaction_phase
# ----------------------------------------------------------------------

def test_advance_transitions_queued_to_running(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.advance_compaction_phase(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is True
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None
    assert snap.active_compaction.status == "running"
    assert snap.active_compaction.phase_index == 1
    assert snap.active_compaction.phase_name == "summarize"


def test_advance_fails_on_different_worker(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.advance_compaction_phase(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
        phase_index=1, phase_name="summarize",
    ) is False
    snap = s.read_progress_snapshot("c")
    # Unchanged — still queued at phase 0.
    assert snap.active_compaction.status == "queued"
    assert snap.active_compaction.phase_index == 0


def test_advance_fails_on_stale_epoch_correlated_subquery(tmp_path: Path):
    """Stale caller carrying epoch=1 cannot advance after resurrect to epoch=2."""
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    )
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Stale thread thinks epoch=1, tries to advance. Correlated subquery
    # guard rejects even though worker id matches.
    assert s.advance_compaction_phase(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is False


# ----------------------------------------------------------------------
# complete_compaction_operation
# ----------------------------------------------------------------------

def test_complete_succeeds_for_owner(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is True
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is None


def test_complete_fails_for_non_owner(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
    ) is False
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None  # still active


def test_complete_fails_on_stale_epoch(tmp_path: Path):
    """Stale caller carrying epoch=1 cannot complete an epoch=2 operation."""
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    )
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Stale thread thinks epoch=1, tries to complete epoch=2. Must fail.
    assert s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is False
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None
    assert snap.lifecycle_epoch == 2


# ----------------------------------------------------------------------
# fail_compaction_operation
# ----------------------------------------------------------------------

def test_fail_records_error_message(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.fail_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        error_message="explode",
    ) is True
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT status, error_message FROM compaction_operation WHERE conversation_id = 'c'"
        ).fetchone()
    assert row[0] == "failed"
    assert row[1] == "explode"
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is None  # failed = no longer active


def test_fail_fails_for_non_owner(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.fail_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w2",
        error_message="nope",
    ) is False
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None  # still active, unchanged


def test_fail_fails_on_stale_epoch(tmp_path: Path):
    s = _fresh(tmp_path)
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    )
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.fail_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        error_message="stale",
    ) is False


# ----------------------------------------------------------------------
# Epoch isolation regression
# ----------------------------------------------------------------------

def test_stale_epoch_thread_cannot_affect_resurrected_epoch(tmp_path: Path):
    """A stale thread carrying epoch=1 must NOT mutate the resurrected epoch=2 row."""
    s = _fresh(tmp_path)
    # Run and complete epoch=1.
    s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is True
    # Resurrect to epoch=2.
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    op2 = s.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Stale thread tries every mutating op at epoch=1 — all must be no-ops.
    assert s.advance_compaction_phase(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is False
    assert s.complete_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
    ) is False
    assert s.fail_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="w1",
        error_message="should not record",
    ) is False
    # Epoch=2 row is still queued at phase 0 with no error message.
    snap = s.read_progress_snapshot("c")
    assert snap.active_compaction is not None
    assert snap.active_compaction.operation_id == op2
    assert snap.active_compaction.status == "queued"
    assert snap.active_compaction.phase_index == 0
    with s._get_conn() as conn:
        err = conn.execute(
            "SELECT error_message FROM compaction_operation WHERE operation_id = ?",
            (op2,),
        ).fetchone()[0]
    assert err is None
