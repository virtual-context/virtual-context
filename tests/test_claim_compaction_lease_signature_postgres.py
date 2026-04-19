"""Postgres mirror of test_claim_compaction_lease_signature.py.

Pins that PostgresStore.claim_compaction_lease returns CompactionLeaseClaim
(not bool) and that the prev_operation_id / prev_owner_worker_id fields are
populated atomically from the row touched by the UPDATE.

Skipped unless VC_TEST_POSTGRES_URL is set.
"""
from __future__ import annotations

import os
import uuid

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def _store():
    from virtual_context.storage.postgres import PostgresStore
    return PostgresStore(PG_URL)


def _fresh():
    conv_id = str(uuid.uuid4())
    s = _store()
    s.upsert_conversation(tenant_id="t", conversation_id=conv_id)
    return s, conv_id


def test_claim_compaction_lease_returns_claim_object_on_success_pg():
    """Stale-heartbeat takeover → CompactionLeaseClaim with prev_* populated."""
    from virtual_context.types import CompactionLeaseClaim
    from datetime import datetime, timedelta, timezone

    s, cid = _fresh()
    conn = s._get_conn()

    stale = datetime.now(timezone.utc) - timedelta(seconds=600)
    now = datetime.now(timezone.utc)
    op_id = str(uuid.uuid4())

    conn.execute(
        """
        INSERT INTO compaction_operation (
            operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, heartbeat_ts, owner_worker_id
        ) VALUES (%s, %s, 1, 0, 7, 'segment_tagging', 'running',
                  %s, %s, %s)
        """,
        (op_id, cid, now, stale, "dead-worker"),
    )

    claim = s.claim_compaction_lease(
        conversation_id=cid, lifecycle_epoch=1,
        worker_id="new-worker", lease_ttl_s=30.0,
    )

    assert isinstance(claim, CompactionLeaseClaim)
    assert claim.claimed is True
    assert claim.prev_operation_id == op_id
    assert claim.prev_owner_worker_id == "dead-worker"


def test_claim_compaction_lease_returns_claim_none_on_failure_pg():
    """No running row at all → claimed=False, prev_* fields None."""
    from virtual_context.types import CompactionLeaseClaim

    s, cid = _fresh()
    claim = s.claim_compaction_lease(
        conversation_id=cid, lifecycle_epoch=1,
        worker_id="w", lease_ttl_s=30.0,
    )
    assert isinstance(claim, CompactionLeaseClaim)
    assert claim.claimed is False
    assert claim.prev_operation_id is None
    assert claim.prev_owner_worker_id is None


def test_claim_compaction_lease_rejects_fresh_heartbeat_pg():
    """Live lease — owner is 'other', heartbeat fresh → claimed=False,
    prev_* reflect the row we observed but did NOT claim."""
    from virtual_context.types import CompactionLeaseClaim
    from datetime import datetime, timezone

    s, cid = _fresh()
    conn = s._get_conn()

    now = datetime.now(timezone.utc)
    op_id = str(uuid.uuid4())

    conn.execute(
        """
        INSERT INTO compaction_operation (
            operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, heartbeat_ts, owner_worker_id
        ) VALUES (%s, %s, 1, 0, 7, 'segment_tagging', 'running',
                  %s, %s, %s)
        """,
        (op_id, cid, now, now, "other-worker"),
    )

    claim = s.claim_compaction_lease(
        conversation_id=cid, lifecycle_epoch=1,
        worker_id="me", lease_ttl_s=30.0,
    )

    assert isinstance(claim, CompactionLeaseClaim)
    assert claim.claimed is False
    assert claim.prev_operation_id == op_id
    assert claim.prev_owner_worker_id == "other-worker"
