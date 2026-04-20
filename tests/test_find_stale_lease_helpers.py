"""Tests for the find_stale_ingestion_episodes / find_stale_compaction_operations
store helpers used by the background StaleLeaseSweeper. SQLite flavor —
the Postgres flavor mirrors the same SQL contract and is exercised in
integration tests + production.

Contract under test:
- Returns rows whose heartbeat_ts < NOW() - grace_s.
- Skips rows whose heartbeat is fresh (within grace).
- Skips rows whose conversation has been deleted.
- Skips rows whose lifecycle_epoch no longer matches the conversation
  (delete+resurrect race — the new epoch's owner will get a fresh
  episode on the next POST).
- Carries forward the fields the sweeper needs: conversation_id,
  lifecycle_epoch, tenant_id, owner_worker_id, hb_age_s.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid as _uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


def _make_store(tmp_path: Path):
    from virtual_context.storage.sqlite import SQLiteStore
    return SQLiteStore(db_path=str(tmp_path / "vc.db"))


def _seed_conversation(store, conv_id: str, *, tenant_id: str = "tnt-1",
                       lifecycle_epoch: int = 1, phase: str = "ingesting"):
    """Insert a conversations row + set phase via the store API."""
    now = datetime.now(timezone.utc).isoformat()
    with store._get_conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO conversations (
                 conversation_id, tenant_id, lifecycle_epoch, phase,
                 pending_raw_payload_entries, last_raw_payload_entries,
                 last_ingestible_payload_entries, created_at, updated_at
               ) VALUES (?, ?, ?, ?, 0, 0, 0, ?, ?)""",
            (conv_id, tenant_id, lifecycle_epoch, phase, now, now),
        )


def _seed_episode(store, conv_id: str, *, hb_age_s: float,
                  status: str = "running", lifecycle_epoch: int = 1,
                  owner: str = "worker-A"):
    now = datetime.now(timezone.utc)
    hb = (now - timedelta(seconds=hb_age_s)).isoformat()
    started = (now - timedelta(seconds=hb_age_s + 5)).isoformat()
    ep_id = str(_uuid.uuid4())
    with store._get_conn() as c:
        c.execute(
            """INSERT INTO ingestion_episode (
                 episode_id, conversation_id, lifecycle_epoch,
                 raw_payload_entries, started_at, status,
                 owner_worker_id, heartbeat_ts
               ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)""",
            (ep_id, conv_id, lifecycle_epoch, started, status, owner, hb),
        )
    return ep_id


def _seed_compaction(store, conv_id: str, *, hb_age_s: float,
                     status: str = "running", lifecycle_epoch: int = 1,
                     owner: str = "worker-A"):
    now = datetime.now(timezone.utc)
    hb = (now - timedelta(seconds=hb_age_s)).isoformat()
    started = (now - timedelta(seconds=hb_age_s + 5)).isoformat()
    op_id = str(_uuid.uuid4())
    with store._get_conn() as c:
        c.execute(
            """INSERT INTO compaction_operation (
                 operation_id, conversation_id, lifecycle_epoch,
                 phase_index, phase_count, phase_name, status,
                 started_at, heartbeat_ts, owner_worker_id, created_at
               ) VALUES (?, ?, ?, 0, 7, 'segment_grouping', ?, ?, ?, ?, ?)""",
            (op_id, conv_id, lifecycle_epoch, status, started, hb, owner, started),
        )
    return op_id


# ---------------------------------------------------------------------------
# Ingestion-episode sweeper queries
# ---------------------------------------------------------------------------

def test_finds_stale_episode_past_grace(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-stale")
    ep_id = _seed_episode(store, "conv-stale", hb_age_s=120.0)
    rows = store.find_stale_ingestion_episodes(grace_s=45.0)
    assert any(r["episode_id"] == ep_id for r in rows), (
        f"stale episode (hb_age=120s, grace=45s) should be returned; got {rows}"
    )
    row = next(r for r in rows if r["episode_id"] == ep_id)
    assert row["conversation_id"] == "conv-stale"
    assert row["tenant_id"] == "tnt-1"
    assert row["owner_worker_id"] == "worker-A"
    assert row["hb_age_s"] >= 119.0
    assert row["lifecycle_epoch"] == 1


def test_skips_fresh_episode_inside_grace(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-fresh")
    _seed_episode(store, "conv-fresh", hb_age_s=10.0)
    rows = store.find_stale_ingestion_episodes(grace_s=45.0)
    assert not [r for r in rows if r["conversation_id"] == "conv-fresh"], (
        f"fresh episode (hb_age=10s, grace=45s) must NOT be returned; got {rows}"
    )


def test_skips_completed_episode(tmp_path):
    """Completed episodes have status='completed' — sweeper must ignore."""
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-done")
    _seed_episode(store, "conv-done", hb_age_s=600.0, status="completed")
    rows = store.find_stale_ingestion_episodes(grace_s=45.0)
    assert not [r for r in rows if r["conversation_id"] == "conv-done"]


def test_skips_episode_for_deleted_conversation(tmp_path):
    """A deleted conversation shouldn't re-spawn ingestion via the sweeper."""
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-gone", phase="deleted")
    _seed_episode(store, "conv-gone", hb_age_s=120.0)
    rows = store.find_stale_ingestion_episodes(grace_s=45.0)
    assert not [r for r in rows if r["conversation_id"] == "conv-gone"]


def test_skips_episode_with_mismatched_lifecycle(tmp_path):
    """Episode at epoch=1, conversation now at epoch=2 → stale lifecycle.
    Sweeper must skip — the new lifecycle's first POST will create a
    fresh episode at epoch=2.
    """
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-resurrect", lifecycle_epoch=2)
    _seed_episode(store, "conv-resurrect", hb_age_s=120.0, lifecycle_epoch=1)
    rows = store.find_stale_ingestion_episodes(grace_s=45.0)
    assert not [r for r in rows if r["conversation_id"] == "conv-resurrect"]


def test_returns_multiple_stale_episodes_ordered_by_age(tmp_path):
    store = _make_store(tmp_path)
    for i, age in enumerate([300.0, 60.0, 600.0, 100.0]):
        cid = f"conv-{i}"
        _seed_conversation(store, cid)
        _seed_episode(store, cid, hb_age_s=age)
    rows = store.find_stale_ingestion_episodes(grace_s=45.0)
    assert len(rows) == 4
    ages = [r["hb_age_s"] for r in rows]
    assert ages == sorted(ages, reverse=True), (
        f"expected oldest-first ordering; got {ages}"
    )


# ---------------------------------------------------------------------------
# Compaction-operation sweeper queries
# ---------------------------------------------------------------------------

def test_finds_stale_compaction_past_grace(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-compact-stale", phase="compacting")
    op_id = _seed_compaction(store, "conv-compact-stale", hb_age_s=200.0)
    rows = store.find_stale_compaction_operations(grace_s=45.0)
    assert any(r["operation_id"] == op_id for r in rows)
    row = next(r for r in rows if r["operation_id"] == op_id)
    assert row["phase_count"] == 7
    assert row["phase_name"] == "segment_grouping"
    assert row["hb_age_s"] >= 199.0


def test_skips_fresh_compaction_inside_grace(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-compact-fresh", phase="compacting")
    _seed_compaction(store, "conv-compact-fresh", hb_age_s=10.0)
    rows = store.find_stale_compaction_operations(grace_s=45.0)
    assert not [r for r in rows if r["conversation_id"] == "conv-compact-fresh"]


def test_skips_queued_compaction(tmp_path):
    """status='queued' isn't actively being executed; the start_compaction_operation
    caller will pick it up. Sweeper only owns 'running' rows.
    """
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-compact-q", phase="compacting")
    _seed_compaction(store, "conv-compact-q", hb_age_s=200.0, status="queued")
    rows = store.find_stale_compaction_operations(grace_s=45.0)
    assert not [r for r in rows if r["conversation_id"] == "conv-compact-q"]
