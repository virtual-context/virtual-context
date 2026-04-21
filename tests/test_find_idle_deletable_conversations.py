"""Tests for find_idle_deletable_conversations — the store helper
that feeds the sweeper's idle-conv auto-delete pass.

Contract under test:
  - returns convs with canonical_turns count < max_msgs
  - returns convs whose last-activity timestamp is older than min_age_s
  - never returns deleted/compacting convs
  - never returns convs with a running ingestion_episode
  - never returns convs with a running or queued compaction_operation
  - returns oldest-activity-first, bounded by ``limit``
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


def _seed_conv(store, conv_id: str, *, tenant="tnt", phase="active",
               created_s_ago: float = 0.0, updated_s_ago: float = 0.0,
               deleted: bool = False):
    now = datetime.now(timezone.utc)
    created = (now - timedelta(seconds=created_s_ago)).isoformat()
    updated = (now - timedelta(seconds=updated_s_ago)).isoformat()
    del_at = now.isoformat() if deleted else None
    with store._get_conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO conversations (
                 conversation_id, tenant_id, lifecycle_epoch, phase,
                 pending_raw_payload_entries, last_raw_payload_entries,
                 last_ingestible_payload_entries, created_at, updated_at, deleted_at
               ) VALUES (?, ?, 1, ?, 0, 0, 0, ?, ?, ?)""",
            (conv_id, tenant, phase, created, updated, del_at),
        )


def _add_canonical_turns(store, conv_id: str, count: int, *, last_seen_s_ago: float = 0):
    now = datetime.now(timezone.utc)
    latest = now - timedelta(seconds=last_seen_s_ago)
    with store._get_conn() as c:
        for i in range(count):
            seen = (latest - timedelta(seconds=i)).isoformat()
            c.execute(
                """INSERT INTO canonical_turns (
                     canonical_turn_id, conversation_id, sort_key, turn_hash,
                     hash_version, user_content, assistant_content,
                     primary_tag, tags_json, session_date, sender,
                     first_seen_at, last_seen_at, created_at, updated_at,
                     turn_group_number, covered_ingestible_entries, tagged_at
                   ) VALUES (?, ?, ?, ?, 1, 'u', 'a', 'p', '["p"]',
                             '2026-04-21', 'user', ?, ?, ?, ?, 0, 1, ?)""",
                (f"{conv_id}-{i}", conv_id, float(i * 1000),
                 f"h{conv_id}-{i}", seen, seen, seen, seen, seen),
            )


def _seed_running_episode(store, conv_id: str):
    now = datetime.now(timezone.utc).isoformat()
    with store._get_conn() as c:
        c.execute(
            """INSERT INTO ingestion_episode (
                 episode_id, conversation_id, lifecycle_epoch, raw_payload_entries,
                 started_at, status, owner_worker_id, heartbeat_ts
               ) VALUES (?, ?, 1, 1, ?, 'running', 'w', ?)""",
            (str(_uuid.uuid4()), conv_id, now, now),
        )


def _seed_running_compaction(store, conv_id: str):
    now = datetime.now(timezone.utc).isoformat()
    with store._get_conn() as c:
        c.execute(
            """INSERT INTO compaction_operation (
                 operation_id, conversation_id, lifecycle_epoch,
                 phase_index, phase_count, phase_name, status,
                 started_at, heartbeat_ts, owner_worker_id, created_at
               ) VALUES (?, ?, 1, 0, 7, 'segment_grouping', 'running',
                         ?, ?, 'w', ?)""",
            (str(_uuid.uuid4()), conv_id, now, now, now),
        )


# ---------------------------------------------------------------------------
# Basic selection
# ---------------------------------------------------------------------------

def test_idle_conv_with_few_turns_returned(tmp_path):
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-idle", created_s_ago=7200, updated_s_ago=7200)
    _add_canonical_turns(store, "conv-idle", count=2, last_seen_s_ago=7200)

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    ids = [r["conversation_id"] for r in rows]
    assert "conv-idle" in ids
    row = next(r for r in rows if r["conversation_id"] == "conv-idle")
    assert row["msg_count"] == 2
    assert row["age_s"] >= 7199
    assert row["tenant_id"] == "tnt"


def test_fresh_conv_below_threshold_excluded(tmp_path):
    """Age < min_age_s means "still fresh, may grow" → don't delete."""
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-fresh", created_s_ago=30, updated_s_ago=30)
    _add_canonical_turns(store, "conv-fresh", count=2, last_seen_s_ago=30)

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    assert "conv-fresh" not in [r["conversation_id"] for r in rows]


def test_conv_at_or_above_threshold_excluded(tmp_path):
    """msg_count >= max_msgs → not trivial, keep."""
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-big", created_s_ago=7200, updated_s_ago=7200)
    _add_canonical_turns(store, "conv-big", count=5, last_seen_s_ago=7200)

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    assert "conv-big" not in [r["conversation_id"] for r in rows]


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def test_deleted_conv_excluded(tmp_path):
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-dead", phase="deleted", deleted=True,
               created_s_ago=7200, updated_s_ago=7200)
    _add_canonical_turns(store, "conv-dead", count=1, last_seen_s_ago=7200)

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    assert "conv-dead" not in [r["conversation_id"] for r in rows]


def test_compacting_conv_excluded(tmp_path):
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-comp", phase="compacting",
               created_s_ago=7200, updated_s_ago=7200)
    _add_canonical_turns(store, "conv-comp", count=2, last_seen_s_ago=7200)

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    assert "conv-comp" not in [r["conversation_id"] for r in rows]


def test_conv_with_running_episode_excluded(tmp_path):
    """Even if idle + tiny, if there's a running ingestion_episode row,
    the takeover pass owns it. Don't race."""
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-lease", created_s_ago=7200, updated_s_ago=7200)
    _add_canonical_turns(store, "conv-lease", count=2, last_seen_s_ago=7200)
    _seed_running_episode(store, "conv-lease")

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    assert "conv-lease" not in [r["conversation_id"] for r in rows]


def test_conv_with_running_compaction_excluded(tmp_path):
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-compop", created_s_ago=7200, updated_s_ago=7200)
    _add_canonical_turns(store, "conv-compop", count=2, last_seen_s_ago=7200)
    _seed_running_compaction(store, "conv-compop")

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    assert "conv-compop" not in [r["conversation_id"] for r in rows]


# ---------------------------------------------------------------------------
# Ordering + limit
# ---------------------------------------------------------------------------

def test_returns_oldest_first(tmp_path):
    store = _make_store(tmp_path)
    # Three idle convs with different ages
    for i, age in enumerate([3700, 10000, 5000]):
        cid = f"c{i}"
        _seed_conv(store, cid, created_s_ago=age, updated_s_ago=age)
        _add_canonical_turns(store, cid, count=2, last_seen_s_ago=age)

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    ages = [r["age_s"] for r in rows]
    assert ages == sorted(ages, reverse=True), (
        f"expected oldest-first; got {ages}"
    )


def test_limit_bounds_results(tmp_path):
    store = _make_store(tmp_path)
    for i in range(5):
        cid = f"c{i}"
        _seed_conv(store, cid, created_s_ago=7200, updated_s_ago=7200)
        _add_canonical_turns(store, cid, count=2, last_seen_s_ago=7200)

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600, limit=2)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Empty conv case
# ---------------------------------------------------------------------------

def test_empty_conv_with_old_created_at_returned(tmp_path):
    """A conv with zero canonical_turns but created > min_age_s ago —
    should fall out as deletable (msg_count=0, falls back to created_at
    for last_activity_at)."""
    store = _make_store(tmp_path)
    _seed_conv(store, "conv-empty", created_s_ago=7200, updated_s_ago=7200)
    # No canonical_turns added at all

    rows = store.find_idle_deletable_conversations(max_msgs=5, min_age_s=3600)
    ids = [r["conversation_id"] for r in rows]
    assert "conv-empty" in ids
    row = next(r for r in rows if r["conversation_id"] == "conv-empty")
    assert row["msg_count"] == 0
