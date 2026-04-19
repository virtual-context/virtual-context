"""Heartbeat helper must refresh only when (operation_id, epoch,
owner) all match, and must return False on any mismatch so the sidecar
knows to bail.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.storage.sqlite import SQLiteStore


def _seed_running(store, conv, op, owner, epoch=1):
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, ?, 0, 7, 'starting', 'running', ?, ?, ?, ?)""",
            (op, conv, epoch, now, "2026-01-01T00:00:00+00:00", owner, now),
        )


def test_refresh_updates_heartbeat_when_fully_matched(tmp_path: Path):
    store = SQLiteStore(tmp_path / "hb.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", "me")

    ok = store.refresh_compaction_heartbeat(
        conversation_id="c",
        lifecycle_epoch=1,
        worker_id="me",
        operation_id="op-1",
    )
    assert ok is True

    with store._get_conn() as conn:
        hb = conn.execute(
            "SELECT heartbeat_ts FROM compaction_operation WHERE operation_id='op-1'"
        ).fetchone()["heartbeat_ts"]
    assert hb > "2026-01-01T00:00:00+00:00"


def test_refresh_rejects_wrong_worker(tmp_path: Path):
    store = SQLiteStore(tmp_path / "hb2.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", "me")
    ok = store.refresh_compaction_heartbeat(
        conversation_id="c",
        lifecycle_epoch=1,
        worker_id="other",
        operation_id="op-1",
    )
    assert ok is False


def test_refresh_rejects_wrong_operation_id(tmp_path: Path):
    store = SQLiteStore(tmp_path / "hb3.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", "me")
    ok = store.refresh_compaction_heartbeat(
        conversation_id="c",
        lifecycle_epoch=1,
        worker_id="me",
        operation_id="op-2",
    )
    assert ok is False


def test_refresh_rejects_stale_epoch(tmp_path: Path):
    store = SQLiteStore(tmp_path / "hb4.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", "me", epoch=5)
    ok = store.refresh_compaction_heartbeat(
        conversation_id="c",
        lifecycle_epoch=1,
        worker_id="me",
        operation_id="op-1",
    )
    assert ok is False
