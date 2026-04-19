"""Tests for the Postgres compaction_operation CRUD API.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set. Mirrors the SQLite tests in
``test_compaction_operation_crud.py`` so both backends stay in lockstep on
the epoch-scoped start/claim/advance/complete/fail semantics. The correlated
subquery in ``advance_compaction_phase`` / ``complete_compaction_operation``
/ ``fail_compaction_operation`` pins the epoch filter to the authoritative
``conversations.lifecycle_epoch`` so a stale thread whose conversation was
resurrected to a newer epoch is rejected at SQL level.

Note: ``conversations.conversation_id`` is ``TEXT PRIMARY KEY`` in Postgres
(aligned with ``canonical_turns.conversation_id``), and
``compaction_operation.operation_id`` is UUID. Test conversation IDs use
``str(uuid.uuid4())`` per-test to keep the suite idempotent across reruns
against a shared test database. psycopg3 adapts ``datetime.now(timezone.utc)``
directly to ``TIMESTAMPTZ`` so no ISO-string conversion is needed when
seeding rows.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def _store():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    return PostgresStore(PG_URL)


def _fresh(conv_id: str | None = None):
    """Create a fresh conversation with a unique UUID and return (store, conv_id)."""
    conv_id = conv_id or str(uuid.uuid4())
    s = _store()
    s.upsert_conversation(tenant_id="t", conversation_id=conv_id)
    return s, conv_id


# ----------------------------------------------------------------------
# start_compaction_operation
# ----------------------------------------------------------------------

def test_start_creates_queued_operation_pg():
    s, cid = _fresh()
    op_id = s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=5, phase_name="init",
    )
    assert isinstance(op_id, str)
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None
    assert snap.active_compaction.operation_id == op_id
    assert snap.active_compaction.status == "queued"
    assert snap.active_compaction.phase_index == 0
    assert snap.active_compaction.phase_count == 5
    assert snap.active_compaction.phase_name == "init"


def test_start_returns_uuid_string_pg():
    s, cid = _fresh()
    op_id = s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    uuid.UUID(op_id)


def test_start_fails_when_another_queued_operation_exists_pg():
    import psycopg
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        s.start_compaction_operation(
            conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
            phase_count=3, phase_name="init",
        )


def test_start_fails_when_running_operation_exists_pg():
    import psycopg
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.advance_compaction_phase(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is True
    with pytest.raises(psycopg.errors.UniqueViolation):
        s.start_compaction_operation(
            conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
            phase_count=3, phase_name="init",
        )


def test_start_succeeds_after_completion_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is True
    new_id = s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        phase_count=3, phase_name="init",
    )
    assert isinstance(new_id, str)


# ----------------------------------------------------------------------
# claim_compaction_lease
# ----------------------------------------------------------------------

def test_claim_succeeds_when_caller_already_owns_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.claim_compaction_lease(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        lease_ttl_s=30.0,
    ).claimed is True


def test_claim_fails_when_other_worker_holds_fresh_lease_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.claim_compaction_lease(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        lease_ttl_s=30.0,
    ).claimed is False


def test_claim_succeeds_when_other_worker_lease_is_stale_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    # Force stale heartbeat so w2 can take over.
    stale = datetime(2000, 1, 1, tzinfo=timezone.utc)
    conn = s._get_conn()
    conn.execute(
        "UPDATE compaction_operation SET heartbeat_ts = %s WHERE conversation_id = %s",
        (stale, cid),
    )
    assert s.claim_compaction_lease(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        lease_ttl_s=30.0,
    ).claimed is True
    row = conn.execute(
        "SELECT owner_worker_id FROM compaction_operation WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    owner = row["owner_worker_id"] if isinstance(row, dict) else row[0]
    assert owner == "w2"


def test_claim_fails_on_different_epoch_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.claim_compaction_lease(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        lease_ttl_s=30.0,
    ).claimed is False


# ----------------------------------------------------------------------
# advance_compaction_phase
# ----------------------------------------------------------------------

def test_advance_transitions_queued_to_running_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.advance_compaction_phase(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is True
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None
    assert snap.active_compaction.status == "running"
    assert snap.active_compaction.phase_index == 1
    assert snap.active_compaction.phase_name == "summarize"


def test_advance_fails_on_different_worker_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.advance_compaction_phase(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        phase_index=1, phase_name="summarize",
    ) is False
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction.status == "queued"
    assert snap.active_compaction.phase_index == 0


def test_advance_fails_on_stale_epoch_correlated_subquery_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    )
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.advance_compaction_phase(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is False


# ----------------------------------------------------------------------
# complete_compaction_operation
# ----------------------------------------------------------------------

def test_complete_succeeds_for_owner_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is True
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is None


def test_complete_fails_for_non_owner_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
    ) is False
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None


def test_complete_fails_on_stale_epoch_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    )
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is False
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None
    assert snap.lifecycle_epoch == 2


# ----------------------------------------------------------------------
# fail_compaction_operation
# ----------------------------------------------------------------------

def test_fail_records_error_message_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.fail_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        error_message="explode",
    ) is True
    conn = s._get_conn()
    row = conn.execute(
        "SELECT status, error_message FROM compaction_operation WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    status = row["status"] if isinstance(row, dict) else row[0]
    err = row["error_message"] if isinstance(row, dict) else row[1]
    assert status == "failed"
    assert err == "explode"
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is None


def test_fail_fails_for_non_owner_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.fail_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        error_message="nope",
    ) is False
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None


def test_fail_fails_on_stale_epoch_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    )
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.fail_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        error_message="stale",
    ) is False


# ----------------------------------------------------------------------
# Epoch isolation regression
# ----------------------------------------------------------------------

def test_stale_epoch_thread_cannot_affect_resurrected_epoch_pg():
    s, cid = _fresh()
    s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is True
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)
    op2 = s.start_compaction_operation(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        phase_count=3, phase_name="init",
    )
    assert s.advance_compaction_phase(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        phase_index=1, phase_name="summarize",
    ) is False
    assert s.complete_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is False
    assert s.fail_compaction_operation(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        error_message="should not record",
    ) is False
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None
    assert snap.active_compaction.operation_id == op2
    assert snap.active_compaction.status == "queued"
    assert snap.active_compaction.phase_index == 0
    conn = s._get_conn()
    row = conn.execute(
        "SELECT error_message FROM compaction_operation WHERE operation_id = %s",
        (op2,),
    ).fetchone()
    err = row["error_message"] if isinstance(row, dict) else row[0]
    assert err is None
