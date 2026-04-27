"""Storage method tests for VCMERGE (S1.1, S1.5, S1.6, S1.7).

Per VCMerge plan v1.11 sections 3.2 + 11.4 (idempotency) + 11.3 (tenant
isolation). Pins:

- try_reserve_merge_audit_in_progress correctly discriminates the 5
  states from plan section 3.1 T1.3 (reserved / in_progress /
  committed_match / committed_mismatch / race_retry).
- The unique partial index (M0.5) backs the reservation correctness:
  duplicate (tenant, source) at status IN ('in_progress', 'committed')
  collides; rolled_back rows do not collide.
- Tenant isolation invariant (per spec section 13 v3.8-2): two tenants
  with the same source_id can both reserve in_progress.
- lookup_committed_merge_audit_for_source returns the committed row
  when present, None otherwise.
- lookup_active_merge_audit_for_source returns either in_progress or
  committed.
- _mark_merge_rolled_back flips an in_progress row to rolled_back AND
  refuses to flip a row that's already committed (the predicate
  `WHERE status = 'in_progress'` enforces single-owner ownership).

All tests use the SQLite backend per project convention. PG path uses
the same SQL semantics; smoke covered by the schema test bundle.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
VCMerge plan v1.11 section 11 prologue.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore


pytestmark = pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26")


def _store(tmp_path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "store.db")


# ---------------------------------------------------------------------------
# S1.1 try_reserve_merge_audit_in_progress: 5-state discriminator
# ---------------------------------------------------------------------------

def test_try_reserve_returns_reserved_on_first_call(tmp_path):
    """Fresh (tenant, source) -> INSERT succeeds -> status='reserved'."""
    store = _store(tmp_path)
    merge_id = str(uuid.uuid4())
    result = store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id,
        tenant_id="tenant-A",
        source_conversation_id="src-1",
        target_conversation_id="tgt-1",
        source_label_at_merge="my-source",
    )
    assert result.status == "reserved"
    assert result.merge_id == merge_id
    assert result.existing is None


def test_try_reserve_returns_in_progress_on_concurrent_caller(tmp_path):
    """Second caller while first is still in_progress -> in_progress envelope."""
    store = _store(tmp_path)
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())
    r1 = store.try_reserve_merge_audit_in_progress(
        merge_id=first_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert r1.status == "reserved"
    r2 = store.try_reserve_merge_audit_in_progress(
        merge_id=second_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt-other",
        source_label_at_merge="lbl",
    )
    assert r2.status == "in_progress"
    assert r2.merge_id == first_id  # winner's id, not the loser's
    assert r2.existing is not None
    assert r2.existing.status == "in_progress"


def test_try_reserve_returns_committed_match_on_idempotent_retry(tmp_path):
    """Caller re-tries with same (tenant, source, target) after prior commit
    -> committed_match envelope (idempotent retry per spec section 12.7 +
    §6.1 idempotency contract). E-D3 fold: discriminator is
    target_conversation_id, NOT source_label_at_merge.
    """
    store = _store(tmp_path)
    first_id = str(uuid.uuid4())
    r1 = store.try_reserve_merge_audit_in_progress(
        merge_id=first_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="my-label",
    )
    assert r1.status == "reserved"
    store._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), first_id),
    )
    store._get_conn().commit()
    # Retry with the same target -> committed_match.
    r2 = store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="my-label",
    )
    assert r2.status == "committed_match"
    assert r2.existing is not None
    assert r2.existing.status == "committed"
    assert r2.merge_id == first_id


def test_try_reserve_committed_match_when_label_changes_but_target_same(tmp_path):
    """E-D3 fold (codex iter-1 P1): label can change between calls; if
    the target is unchanged, the merge identity is the same -> match.
    Pins the corrected discriminator behavior.
    """
    store = _store(tmp_path)
    first_id = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=first_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="old-label",
    )
    store._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), first_id),
    )
    store._get_conn().commit()
    r2 = store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="NEW-label",  # label changed
    )
    # Same target = same merge intent = match.
    assert r2.status == "committed_match"


def test_try_reserve_returns_committed_mismatch_on_target_change(tmp_path):
    """E-D3 fold (codex iter-1 P1): different target = different merge
    intent -> committed_mismatch. Source label is irrelevant.
    """
    store = _store(tmp_path)
    first_id = str(uuid.uuid4())
    r1 = store.try_reserve_merge_audit_in_progress(
        merge_id=first_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt-1",
        source_label_at_merge="lbl",
    )
    assert r1.status == "reserved"
    store._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), first_id),
    )
    store._get_conn().commit()
    r2 = store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tA",
        source_conversation_id="src",
        target_conversation_id="tgt-2",  # different target
        source_label_at_merge="lbl",  # same label
    )
    assert r2.status == "committed_mismatch"
    assert r2.existing is not None
    assert r2.existing.target_conversation_id == "tgt-1"


def test_try_reserve_after_rolled_back_succeeds_with_reserved(tmp_path):
    """The unique partial index excludes rolled_back rows (per D4 + M0.5).
    A retry after a rolled_back attempt should succeed cleanly.
    """
    store = _store(tmp_path)
    first_id = str(uuid.uuid4())
    r1 = store.try_reserve_merge_audit_in_progress(
        merge_id=first_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert r1.status == "reserved"
    # Simulate body failure: mark rolled_back.
    flipped = store._mark_merge_rolled_back("tA", first_id, "test-failure")
    assert flipped is True
    # Retry: should succeed with reserved (not collide with the
    # rolled_back row).
    r2 = store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert r2.status == "reserved"


def test_try_reserve_tenant_isolation_two_tenants_same_source(tmp_path):
    """Tenant isolation invariant (per spec section 13 v3.8-2): two
    tenants with the same source_id can both reserve in_progress.
    """
    store = _store(tmp_path)
    r1 = store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tenant-A",
        source_conversation_id="shared-src", target_conversation_id="tgt-A",
        source_label_at_merge="lbl",
    )
    r2 = store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tenant-B",
        source_conversation_id="shared-src", target_conversation_id="tgt-B",
        source_label_at_merge="lbl",
    )
    assert r1.status == "reserved"
    assert r2.status == "reserved"


# ---------------------------------------------------------------------------
# S1.5 / S1.6 lookup helpers
# ---------------------------------------------------------------------------

def test_lookup_committed_returns_none_when_no_row(tmp_path):
    store = _store(tmp_path)
    assert store.lookup_committed_merge_audit_for_source(
        "tenant-A", "no-such-src",
    ) is None


def test_lookup_committed_returns_view_after_commit(tmp_path):
    store = _store(tmp_path)
    mid = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    store._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), mid),
    )
    store._get_conn().commit()
    view = store.lookup_committed_merge_audit_for_source("tA", "src")
    assert view is not None
    assert view.status == "committed"
    assert view.merge_id == mid


def test_lookup_committed_ignores_in_progress(tmp_path):
    """S1.5 is committed-only; in_progress rows must NOT be returned."""
    store = _store(tmp_path)
    store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert store.lookup_committed_merge_audit_for_source("tA", "src") is None


def test_lookup_active_returns_in_progress_row(tmp_path):
    """S1.6: active includes both in_progress and committed."""
    store = _store(tmp_path)
    mid = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    view = store.lookup_active_merge_audit_for_source("tA", "src")
    assert view is not None
    assert view.status == "in_progress"


def test_lookup_active_ignores_rolled_back(tmp_path):
    """S1.6 explicitly excludes rolled_back; matches the partial index."""
    store = _store(tmp_path)
    mid = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    store._mark_merge_rolled_back("tA", mid, "fail")
    assert store.lookup_active_merge_audit_for_source("tA", "src") is None


def test_lookups_scoped_per_tenant(tmp_path):
    """Cross-tenant lookup returns None; same-tenant lookup returns the view.
    Pins the tenant-isolation invariant on the read path.
    """
    store = _store(tmp_path)
    mid = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tenant-A",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    store._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), mid),
    )
    store._get_conn().commit()
    assert store.lookup_committed_merge_audit_for_source(
        "tenant-A", "src",
    ) is not None
    assert store.lookup_committed_merge_audit_for_source(
        "tenant-B", "src",
    ) is None


# ---------------------------------------------------------------------------
# S1.7 _mark_merge_rolled_back
# ---------------------------------------------------------------------------

def test_mark_rolled_back_flips_in_progress(tmp_path):
    store = _store(tmp_path)
    mid = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    flipped = store._mark_merge_rolled_back("tA", mid, "the body raised")
    assert flipped is True
    row = store._get_conn().execute(
        "SELECT status, error_message FROM merge_audit WHERE merge_id=?",
        (mid,),
    ).fetchone()
    assert row["status"] == "rolled_back"
    assert row["error_message"] == "the body raised"


def test_mark_rolled_back_refuses_to_flip_committed(tmp_path):
    """Predicate `WHERE status = 'in_progress'` prevents stomping on a
    successful commit. Returns False; row stays committed.
    """
    store = _store(tmp_path)
    mid = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    store._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), mid),
    )
    store._get_conn().commit()
    flipped = store._mark_merge_rolled_back("tA", mid, "stale rollback attempt")
    assert flipped is False
    row = store._get_conn().execute(
        "SELECT status FROM merge_audit WHERE merge_id=?", (mid,),
    ).fetchone()
    assert row["status"] == "committed"


def test_mark_rolled_back_predicates_on_tenant_id(tmp_path):
    """D3: every user-routed write to merge_audit predicates on tenant_id.
    Cross-tenant call returns False (no match) and does NOT flip the row.
    """
    store = _store(tmp_path)
    mid = str(uuid.uuid4())
    store.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tenant-A",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    flipped = store._mark_merge_rolled_back("tenant-WRONG", mid, "mismatch")
    assert flipped is False
    row = store._get_conn().execute(
        "SELECT status FROM merge_audit WHERE merge_id=?", (mid,),
    ).fetchone()
    assert row["status"] == "in_progress"


def test_mark_rolled_back_returns_false_on_unknown_merge_id(tmp_path):
    store = _store(tmp_path)
    flipped = store._mark_merge_rolled_back(
        "tA", str(uuid.uuid4()), "no such merge",
    )
    assert flipped is False
