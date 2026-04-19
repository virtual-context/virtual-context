"""Pins the ``CompactionLeaseClaim`` dataclass introduced by the
compaction-resume-parity design. The claim helper used to return
``bool``; the new contract returns the dataclass so the takeover path
can read ``prev_operation_id`` atomically alongside the claim decision.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest


def test_compaction_lease_claim_is_frozen_dataclass():
    from virtual_context.types import CompactionLeaseClaim
    import dataclasses

    assert dataclasses.is_dataclass(CompactionLeaseClaim), (
        "CompactionLeaseClaim must be a @dataclass"
    )
    params = dataclasses.fields(CompactionLeaseClaim)
    names = {f.name for f in params}
    assert names == {"claimed", "prev_operation_id", "prev_owner_worker_id"}, (
        f"fields mismatch: {names}"
    )
    # Frozen is required so the value is safe to pass across threads.
    claim = CompactionLeaseClaim(
        claimed=True,
        prev_operation_id="op-123",
        prev_owner_worker_id="worker-abc",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.claimed = False  # type: ignore[misc]


def test_compaction_lease_claim_accepts_none_for_prev_fields():
    from virtual_context.types import CompactionLeaseClaim
    claim = CompactionLeaseClaim(
        claimed=False,
        prev_operation_id=None,
        prev_owner_worker_id=None,
    )
    assert claim.claimed is False
    assert claim.prev_operation_id is None
    assert claim.prev_owner_worker_id is None


def test_compaction_lease_lost_is_a_distinct_exception():
    """Per-write guards raise this when the guarded INSERT/UPDATE
    matches zero rows (operation marked abandoned). The compactor
    pipeline catches it specifically to log COMPACTION_WRITE_REJECTED
    and exit cleanly without walking the remaining phases.
    """
    from virtual_context.types import CompactionLeaseLost
    exc = CompactionLeaseLost("op-xyz", write_site="store_segment")
    # It must carry the abandoned op_id and the write site so the
    # observability log line can reproduce the spec's format:
    # COMPACTION_WRITE_REJECTED op=... site=...
    assert exc.operation_id == "op-xyz"
    assert exc.write_site == "store_segment"
    assert isinstance(exc, Exception)
    # Distinct from generic Exception so callers can catch it narrowly.
    assert type(exc).__name__ == "CompactionLeaseLost"


def test_claim_compaction_lease_returns_claim_object_on_success(tmp_path):
    """On a stale-heartbeat takeover, the helper returns a
    CompactionLeaseClaim populated with prev_operation_id and
    prev_owner_worker_id captured atomically from the row we took over.
    """
    from virtual_context.core.canonical_turns import utcnow_iso
    from virtual_context.storage.sqlite import SQLiteStore
    from virtual_context.types import CompactionLeaseClaim
    import datetime as _dt

    store = SQLiteStore(tmp_path / "lease.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    # Insert a stale-heartbeat running row owned by a "dead" worker.
    stale = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=600)).isoformat()
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id)
               VALUES (?, ?, 1, 0, 7, 'segment_tagging', 'running',
                       ?, ?, ?)""",
            ("dead-op-uuid", "c", now, stale, "dead-worker"),
        )

    claim = store.claim_compaction_lease(
        conversation_id="c", lifecycle_epoch=1,
        worker_id="new-worker", lease_ttl_s=30.0,
    )

    assert isinstance(claim, CompactionLeaseClaim)
    assert claim.claimed is True
    assert claim.prev_operation_id == "dead-op-uuid"
    assert claim.prev_owner_worker_id == "dead-worker"


def test_claim_compaction_lease_returns_claim_none_on_failure(tmp_path):
    """No running row at all → claimed=False, prev_* fields None."""
    from virtual_context.storage.sqlite import SQLiteStore
    from virtual_context.types import CompactionLeaseClaim

    store = SQLiteStore(tmp_path / "nolease.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    claim = store.claim_compaction_lease(
        conversation_id="c", lifecycle_epoch=1,
        worker_id="w", lease_ttl_s=30.0,
    )
    assert isinstance(claim, CompactionLeaseClaim)
    assert claim.claimed is False
    assert claim.prev_operation_id is None
    assert claim.prev_owner_worker_id is None


def test_claim_compaction_lease_rejects_fresh_heartbeat(tmp_path):
    """Live lease — owner is 'other', heartbeat fresh → claimed=False."""
    from virtual_context.core.canonical_turns import utcnow_iso
    from virtual_context.storage.sqlite import SQLiteStore

    store = SQLiteStore(tmp_path / "fresh.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id)
               VALUES (?, ?, 1, 0, 7, 'segment_tagging', 'running',
                       ?, ?, ?)""",
            ("live-op", "c", now, now, "other-worker"),
        )
    claim = store.claim_compaction_lease(
        conversation_id="c", lifecycle_epoch=1,
        worker_id="me", lease_ttl_s=30.0,
    )
    assert claim.claimed is False
    # prev_* fields reflect the row we observed but did NOT claim —
    # they're informational either way.
    assert claim.prev_operation_id == "live-op"
    assert claim.prev_owner_worker_id == "other-worker"
