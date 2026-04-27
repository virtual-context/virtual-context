"""CompositeStore forwarder tests for VCMERGE storage methods (v1.13 / F-CR4).

Per cloud's review of engine 11013f4: cloud's handle_vc_merge_cloud calls
the Store protocol methods through whatever store the engine hands it,
which in production is a CompositeStore. Without explicit forwarders for
the new S1.x methods (try_reserve_merge_audit_in_progress, lookups,
_mark_merge_rolled_back), the underlying PostgresStore methods would not
be reachable on a graceful-fallback path. These tests pin:

- CompositeStore forwards each method to its `_segments` underlying store
  when that store implements the method.
- CompositeStore raises NotImplementedError for try_reserve when the
  underlying store predates Phase 0 schema (defense-in-depth signal so
  cloud can render a clean envelope rather than AttributeError).
- Lookup forwarders return None gracefully when the underlying store
  doesn't implement them (read-side, can't fail-loud safely).
- _mark_merge_rolled_back returns False when forwarded (single-owner
  semantics — matches the underlying PG/SQLite behavior).

All tests use the SQLite backend per project convention; SQLiteStore's
new S1.x methods (per commit 11013f4) are wrapped in a CompositeStore
to mirror production layering.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
VCMerge plan v1.11 section 11 prologue.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.core.composite_store import CompositeStore
from virtual_context.storage.sqlite import SQLiteStore


pytestmark = pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26")


def _composite(tmp_path) -> CompositeStore:
    """Build a CompositeStore wrapping a single SQLiteStore for all five
    sub-stores. Mirrors the simplest production layering — cloud's actual
    setup uses Postgres for all five but the forwarder semantics are
    identical because CompositeStore dispatches on `self._segments` for
    the merge methods.
    """
    inner = SQLiteStore(tmp_path / "store.db")
    return CompositeStore(
        segments=inner, facts=inner, fact_links=inner,
        state=inner, search=inner,
    )


def _composite_with_stub(stub) -> CompositeStore:
    """Build a CompositeStore where `_segments` is a stub object that may
    or may not implement merge methods, but the other four sub-stores can
    be None because the merge forwarders only consult `_segments`.
    Bypasses the type checker via cast since we only exercise the merge
    methods in these tests.
    """
    return CompositeStore(
        segments=stub,           # type: ignore[arg-type]
        facts=stub,              # type: ignore[arg-type]
        fact_links=stub,         # type: ignore[arg-type]
        state=stub,              # type: ignore[arg-type]
        search=stub,             # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# try_reserve_merge_audit_in_progress
# ---------------------------------------------------------------------------

def test_composite_forwards_try_reserve_returns_reserved(tmp_path):
    """CompositeStore forwards try_reserve to the underlying segments store
    AND returns the underlying ReservationResult unchanged.
    """
    composite = _composite(tmp_path)
    merge_id = str(uuid.uuid4())
    result = composite.try_reserve_merge_audit_in_progress(
        merge_id=merge_id,
        tenant_id="tA",
        source_conversation_id="src",
        target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert result.status == "reserved"
    assert result.merge_id == merge_id


def test_composite_forwards_try_reserve_5_state_discriminator(tmp_path):
    """Forwarder preserves the 5-state ReservationResult per plan T1.3."""
    composite = _composite(tmp_path)
    first = str(uuid.uuid4())
    r1 = composite.try_reserve_merge_audit_in_progress(
        merge_id=first, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert r1.status == "reserved"
    r2 = composite.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()), tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert r2.status == "in_progress"
    assert r2.merge_id == first


def test_composite_try_reserve_raises_when_underlying_lacks_method():
    """If the underlying segments store doesn't implement try_reserve
    (old backend predating Phase 0 schema), CompositeStore raises
    NotImplementedError so cloud renders a clean envelope rather than
    AttributeError on a missing-attribute lookup.
    """
    class _StubNoMerge:
        """Pre-Phase-0 stub — no merge methods."""

    composite = _composite_with_stub(_StubNoMerge())
    with pytest.raises(NotImplementedError) as exc_info:
        composite.try_reserve_merge_audit_in_progress(
            merge_id="m1", tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
        )
    assert "Phase 0 schema" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Lookup forwarders
# ---------------------------------------------------------------------------

def test_composite_forwards_lookup_committed(tmp_path):
    """S1.5 forwarder returns the underlying view post-commit."""
    composite = _composite(tmp_path)
    mid = str(uuid.uuid4())
    composite.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    composite._segments._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), mid),
    )
    composite._segments._get_conn().commit()
    view = composite.lookup_committed_merge_audit_for_source("tA", "src")
    assert view is not None
    assert view.status == "committed"
    assert view.merge_id == mid


def test_composite_lookup_committed_returns_none_for_missing_source(tmp_path):
    composite = _composite(tmp_path)
    assert composite.lookup_committed_merge_audit_for_source(
        "tA", "no-such-src",
    ) is None


def test_composite_lookup_committed_returns_none_when_underlying_lacks_method():
    """Read-side forwarders return None gracefully when underlying store
    doesn't implement them — matches the existing
    find_idle_deletable_conversations forwarder pattern.
    """
    class _StubNoMerge:
        pass

    composite = _composite_with_stub(_StubNoMerge())
    assert composite.lookup_committed_merge_audit_for_source(
        "tA", "src",
    ) is None


def test_composite_forwards_lookup_active_returns_in_progress(tmp_path):
    composite = _composite(tmp_path)
    mid = str(uuid.uuid4())
    composite.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    view = composite.lookup_active_merge_audit_for_source("tA", "src")
    assert view is not None
    assert view.status == "in_progress"


def test_composite_lookup_active_returns_none_when_underlying_lacks_method():
    class _StubNoMerge:
        pass

    composite = _composite_with_stub(_StubNoMerge())
    assert composite.lookup_active_merge_audit_for_source(
        "tA", "src",
    ) is None


# ---------------------------------------------------------------------------
# _mark_merge_rolled_back forwarder
# ---------------------------------------------------------------------------

def test_composite_forwards_mark_rolled_back_flips_in_progress(tmp_path):
    composite = _composite(tmp_path)
    mid = str(uuid.uuid4())
    composite.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    flipped = composite._mark_merge_rolled_back("tA", mid, "test-failure")
    assert flipped is True
    row = composite._segments._get_conn().execute(
        "SELECT status FROM merge_audit WHERE merge_id=?",
        (mid,),
    ).fetchone()
    assert row["status"] == "rolled_back"


def test_composite_mark_rolled_back_returns_false_when_underlying_lacks_method():
    """Single-owner semantics: a backend that doesn't implement
    _mark_merge_rolled_back returns False (no-op rather than raising)
    so cloud's rollback path falls through cleanly.
    """
    class _StubNoMerge:
        pass

    composite = _composite_with_stub(_StubNoMerge())
    flipped = composite._mark_merge_rolled_back(
        "tA", str(uuid.uuid4()), "no underlying merge surface",
    )
    assert flipped is False


def test_composite_mark_rolled_back_returns_false_for_already_committed(tmp_path):
    """Forwarder preserves the underlying single-owner predicate — refuses
    to flip a row that's already committed.
    """
    composite = _composite(tmp_path)
    mid = str(uuid.uuid4())
    composite.try_reserve_merge_audit_in_progress(
        merge_id=mid, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    composite._segments._get_conn().execute(
        "UPDATE merge_audit SET status='committed', completed_at=? WHERE merge_id=?",
        (datetime.now(timezone.utc).isoformat(), mid),
    )
    composite._segments._get_conn().commit()
    flipped = composite._mark_merge_rolled_back("tA", mid, "stale rollback")
    assert flipped is False
