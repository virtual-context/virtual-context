"""Regression: once cleanup marks the operation 'abandoned', subsequent
write attempts by the stale worker must write zero rows (silently
rejected by the DB-layer guard).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timezone
from pathlib import Path

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import SegmentMetadata, StoredSegment

_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _seed_running(store, conv, op_id, worker="w"):
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (op_id, conv, now, now, worker, now),
        )


def _segment(conv, ref):
    return StoredSegment(
        ref=ref, conversation_id=conv, primary_tag="t", tags=["t"],
        summary="s", summary_tokens=10, full_text="f", full_tokens=20,
        messages=[], metadata=SegmentMetadata(), compaction_model="passthrough",
        compression_ratio=0.5, start_timestamp=_TS, end_timestamp=_TS,
    )


def test_store_segment_write_succeeds_while_operation_running(tmp_path: Path):
    store = SQLiteStore(tmp_path / "ok.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", worker="w")

    store.store_segment(
        _segment("c", "ref-1"),
        operation_id="op-1",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-1'"
        ).fetchone()[0]
    assert n == 1


def test_store_segment_raises_lease_lost_when_operation_abandoned(tmp_path: Path):
    """Rejection is NOT silent. The guard raises CompactionLeaseLost
    so the compactor pipeline catches it specifically, logs
    COMPACTION_WRITE_REJECTED, and exits cleanly instead of walking
    every remaining phase unaware.
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "abandon.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", worker="w")

    # Simulate takeover: cleanup flips dead op to 'abandoned'.
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-1'",
        )

    # Stale worker still trying to write — must raise.
    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.store_segment(
            _segment("c", "ref-ghost"),
            operation_id="op-1",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.operation_id == "op-1"
    assert exc_info.value.write_site == "store_segment"

    # And zero rows persisted.
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-ghost'"
        ).fetchone()[0]
    assert n == 0


def test_store_segment_raises_lease_lost_when_wrong_owner(tmp_path: Path):
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "wrongowner.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", worker="w-real")

    with pytest.raises(CompactionLeaseLost):
        store.store_segment(
            _segment("c", "ref-wrong"),
            operation_id="op-1",
            owner_worker_id="w-impostor",
            lifecycle_epoch=1,
        )
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-wrong'"
        ).fetchone()[0]
    assert n == 0


def test_store_segment_legacy_path_no_guard_kwargs_still_inserts(tmp_path: Path):
    """Callers that don't pass guard kwargs (pre-change tests,
    non-compaction write sites) keep the unconditional INSERT path.
    Otherwise we'd break all existing test harnesses that construct
    rows directly.
    """
    store = SQLiteStore(tmp_path / "legacy.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    ref = store.store_segment(_segment("c", "ref-legacy"))
    assert ref == "ref-legacy"
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-legacy'"
        ).fetchone()[0]
    assert n == 1
