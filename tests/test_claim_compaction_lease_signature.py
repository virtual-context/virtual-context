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
