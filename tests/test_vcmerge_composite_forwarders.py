"""CompositeStore forwarder tests for VCMERGE storage methods ( / ).

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
  semantics: matches the underlying PG/SQLite behavior).

All tests use the SQLite backend per project convention; SQLiteStore's
new S1.x methods (per commit 11013f4) are wrapped in a CompositeStore
to mirror production layering.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
.
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
    sub-stores. Mirrors the simplest production layering: cloud's actual
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
    """Forwarder preserves the 5-state ReservationResult per plan ."""
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
        """Pre-Phase-0 stub: no merge methods."""

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
    """ forwarder returns the underlying view post-commit."""
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
    doesn't implement them: matches the existing
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
    """Forwarder preserves the underlying single-owner predicate: refuses
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


# ---------------------------------------------------------------------------
# merge_conversation_data forwarder (/ — caught by cloud's )
# ---------------------------------------------------------------------------

def test_composite_forwards_merge_conversation_data(tmp_path):
    """Cloud: the body method forwarder must be
    present on CompositeStore. Without it,
    ``Engine.merge_conversation`` reaches ``store.merge_conversation_data(...)``
    via ``ConversationStoreView(CompositeStore(...))`` and AttributeError's
    before any work happens.

    This test seeds source + target, reserves, runs the body via
    composite.merge_conversation_data, and asserts the merge committed
    end-to-end. Catches the class of bug for any future S1.x
    additions.
    """
    composite = _composite(tmp_path)
    inner = composite._segments
    conn = inner._get_conn()
    now = datetime.now(timezone.utc).isoformat()
    # Seed conversations
    for conv_id in ("src", "tgt"):
        conn.execute(
            "INSERT INTO conversations (conversation_id, tenant_id, phase, "
            "lifecycle_epoch, created_at, updated_at) "
            "VALUES (?, ?, 'active', 1, ?, ?)",
            (conv_id, "tA", now, now),
        )
    conn.commit()
    # Reserve via composite forwarder
    merge_id = str(uuid.uuid4())
    result = composite.try_reserve_merge_audit_in_progress(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        source_label_at_merge="lbl",
    )
    assert result.status == "reserved"
    # Run body via composite forwarder (the fix path)
    stats = composite.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Assert MergeStats returned + audit committed + alias written
    assert stats.merge_id == merge_id
    audit_status = conn.execute(
        "SELECT status FROM merge_audit WHERE merge_id = ?", (merge_id,),
    ).fetchone()["status"]
    assert audit_status == "committed"
    alias = conn.execute(
        "SELECT target_id FROM conversation_aliases WHERE alias_id = 'src'",
    ).fetchone()
    assert alias is not None
    assert alias["target_id"] == "tgt"


def test_composite_merge_conversation_data_raises_when_underlying_lacks_method():
    """If the underlying segments store doesn't implement
    merge_conversation_data, CompositeStore raises NotImplementedError
    (mirrors the try_reserve forwarder's defense-in-depth contract).
    """
    class _StubNoMerge:
        pass

    composite = _composite_with_stub(_StubNoMerge())
    with pytest.raises(NotImplementedError) as exc_info:
        composite.merge_conversation_data(
            merge_id="m1", tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    assert "Phase 0 schema" in str(exc_info.value) or "Phase 1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# End-to-end Engine.merge_conversation through ConversationStoreView +
# CompositeStore (catches the class for any future S1.x methods)
# ---------------------------------------------------------------------------

def test_engine_merge_conversation_through_full_layer(tmp_path):
    """ regression test (cloud's request, ): construct a
    real ``VirtualContextEngine`` and exercise ``engine.merge_conversation``
    end-to-end through ``ConversationStoreView(CompositeStore(...))``.

    Body tests at test_vcmerge_body_method.py construct ``SQLiteStore``
    directly, bypassing the wrapper. That's how the missing
    ``merge_conversation_data`` forwarder slipped through. This test
    closes the gap by going through the full layer.
    """
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    db_path = str(tmp_path / "engine.db")
    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "tag_generator": {"type": "keyword"},
    })
    engine = VirtualContextEngine(config=config)
    try:
        # Reach into the engine's underlying SQLiteStore to seed conversations.
        # ConversationStoreView -> CompositeStore -> SQLiteStore at _segments.
        view = engine._store
        composite = getattr(view, "_store", view)
        inner = getattr(composite, "_segments", composite)
        conn = inner._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        for conv_id in ("src-engine", "tgt-engine"):
            conn.execute(
                "INSERT INTO conversations (conversation_id, tenant_id, phase, "
                "lifecycle_epoch, created_at, updated_at) "
                "VALUES (?, ?, 'active', 1, ?, ?)",
                (conv_id, "tA", now, now),
            )
        conn.commit()
        # Reserve via the engine's store (which goes through CompositeStore).
        merge_id = str(uuid.uuid4())
        reservation = engine._store.try_reserve_merge_audit_in_progress(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src-engine",
            target_conversation_id="tgt-engine",
            source_label_at_merge="engine-test",
        )
        assert reservation.status == "reserved"
        # Call engine.merge_conversation — this is the call path cloud's
        # handle_vc_merge_cloud uses. Without the forwarder it AttributeError's.
        stats = engine.merge_conversation(
            merge_id=merge_id,
            tenant_id="tA",
            source_conversation_id="src-engine",
            target_conversation_id="tgt-engine",
            source_lifecycle_epoch=1,
            target_lifecycle_epoch=1,
            source_label_at_merge="engine-test",
        )
        assert stats.merge_id == merge_id
        assert stats.tenant_id == "tA"
        # Verify merge committed end-to-end
        audit_status = conn.execute(
            "SELECT status FROM merge_audit WHERE merge_id = ?", (merge_id,),
        ).fetchone()["status"]
        assert audit_status == "committed"
        # Verify source phase flipped
        src_phase = conn.execute(
            "SELECT phase FROM conversations WHERE conversation_id = 'src-engine'",
        ).fetchone()["phase"]
        assert src_phase == "merged"
    finally:
        engine.close()


def test_engine_merge_rejects_same_source_as_target(tmp_path):
    """Engine.merge_conversation refuses same-source-as-target merges
    (per Q2 + plan E1.x). Defense-in-depth on top of
    cloud's same-tenant check.
    """
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "e.db")}},
        "tag_generator": {"type": "keyword"},
    })
    engine = VirtualContextEngine(config=config)
    try:
        with pytest.raises(ValueError, match="merge conversation into itself"):
            engine.merge_conversation(
                merge_id=str(uuid.uuid4()),
                tenant_id="tA",
                source_conversation_id="same-id",
                target_conversation_id="same-id",
                source_lifecycle_epoch=1,
                target_lifecycle_epoch=1,
                source_label_at_merge="x",
            )
    finally:
        engine.close()
