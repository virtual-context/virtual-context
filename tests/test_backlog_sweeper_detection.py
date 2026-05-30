"""Compaction-backlog sweeper Phase 0 SQLite tests.

Per ``specs/compaction-backlog-sweeper-plan.md`` §2.5 T0.1-T0.13.
The PG smoke (T0.14-T0.15) lives in
``test_backlog_sweeper_detection_postgres.py``.

The detection method ``find_compaction_backlog_conversations`` must
surface conversations whose tagged-uncompacted ``canonical_turns``
backlog exceeds ``min_backlog_turns`` AND that meet five additional
liveness predicates:

* ``conversations.phase = 'active'`` (not ingesting, compacting,
  deleted, or merged).
* ``conversations.deleted_at IS NULL``.
* NO ``canonical_turns`` row at ``tagged_at IS NULL`` for the
  conversation (the compaction loader processes all uncompacted
  rows, not just tagged ones).
* NO ``compaction_operation`` row at status ``'queued'`` or
  ``'running'`` for the current ``lifecycle_epoch``.
* Most recent terminal compaction row in the current epoch (if any)
  completed more than ``grace_s`` seconds ago.

The tests construct an isolated SQLite store per case and seed the
exact shape required, then assert which candidates the detection
query returns.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import BacklogCandidate


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_conv(
    store: SQLiteStore,
    *,
    conv_id: str,
    tenant_id: str = "t-1",
    phase: str = "active",
    lifecycle_epoch: int = 1,
    deleted_at: str | None = None,
) -> None:
    now = _iso(_now())
    conn = store._get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv_id, now),
    )
    conn.execute(
        "INSERT INTO conversations "
        "(conversation_id, tenant_id, phase, lifecycle_epoch, "
        " created_at, updated_at, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conv_id, tenant_id, phase, lifecycle_epoch, now, now, deleted_at),
    )
    conn.commit()


def _seed_turns(
    store: SQLiteStore,
    *,
    conv_id: str,
    count: int,
    tagged: bool = True,
    compacted: bool = False,
    sort_key_base: float = 1000.0,
    id_prefix: str = "",
) -> None:
    """Seed ``count`` canonical_turns rows on ``conv_id``. When
    ``tagged`` is False, ``tagged_at`` is left NULL so the
    no-untagged-row predicate fires. When ``compacted`` is True,
    ``compacted_at`` is filled so the row drops out of the backlog
    count. ``id_prefix`` lets a single test seed two disjoint batches
    on the same conversation without colliding on the
    canonical_turn_id PK.
    """
    now = _iso(_now())
    conn = store._get_conn()
    for i in range(count):
        conn.execute(
            """INSERT INTO canonical_turns
               (canonical_turn_id, conversation_id, sort_key, turn_hash,
                hash_version, tagged_at, compacted_at, created_at,
                updated_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (
                f"ct-{conv_id}-{id_prefix}{i}", conv_id,
                sort_key_base + i, f"h-{conv_id}-{id_prefix}{i}",
                now if tagged else None,
                now if compacted else None,
                now, now,
            ),
        )
    conn.commit()


def _seed_compaction_op(
    store: SQLiteStore,
    *,
    op_id: str,
    conv_id: str,
    lifecycle_epoch: int = 1,
    status: str = "completed",
    completed_at: datetime | None = None,
    started_at: datetime | None = None,
    worker_id: str = "w-1",
) -> None:
    started = _iso(started_at or _now())
    completed = _iso(completed_at) if completed_at else None
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO compaction_operation
           (operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, heartbeat_ts, owner_worker_id, created_at,
            completed_at)
           VALUES (?, ?, ?, 0, 7, 'starting', ?, ?, ?, ?, ?, ?)""",
        (op_id, conv_id, lifecycle_epoch, status, started, started,
         worker_id, started, completed),
    )
    conn.commit()


def _detect(
    store: SQLiteStore, *, min_backlog_turns: int = 20, grace_s: float = 300.0,
    limit: int = 100,
) -> list[BacklogCandidate]:
    return store.find_compaction_backlog_conversations(
        min_backlog_turns=min_backlog_turns,
        grace_s=grace_s,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# T0.1: positive case -- conv above threshold with no untagged rows,
# no active op, phase=active surfaces.
# ---------------------------------------------------------------------------


class TestT01_PositiveCase:
    def test_conv_above_threshold_surfaces(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t01.db")
        _seed_conv(store, conv_id="conv-pos")
        _seed_turns(store, conv_id="conv-pos", count=21, tagged=True)
        candidates = _detect(store, min_backlog_turns=20)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.conversation_id == "conv-pos"
        assert c.backlog_turns == 21
        assert c.lifecycle_epoch == 1
        assert c.last_terminal_compaction_at is None


# ---------------------------------------------------------------------------
# T0.2: conv below threshold does not surface.
# ---------------------------------------------------------------------------


class TestT02_BelowThreshold:
    def test_below_threshold_skipped(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t02.db")
        _seed_conv(store, conv_id="conv-low")
        _seed_turns(store, conv_id="conv-low", count=19, tagged=True)
        assert _detect(store, min_backlog_turns=20) == []


# ---------------------------------------------------------------------------
# T0.3: phase filter -- only 'active' surfaces.
# ---------------------------------------------------------------------------


class TestT03_PhaseFilter:
    @pytest.mark.parametrize("phase", [
        "ingesting", "compacting", "deleted", "merged",
    ])
    def test_non_active_phase_skipped(self, tmp_path: Path, phase: str):
        store = SQLiteStore(tmp_path / f"t03-{phase}.db")
        _seed_conv(store, conv_id="conv-x", phase=phase)
        _seed_turns(store, conv_id="conv-x", count=25, tagged=True)
        assert _detect(store) == []

    def test_active_phase_surfaces(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t03-active.db")
        _seed_conv(store, conv_id="conv-act", phase="active")
        _seed_turns(store, conv_id="conv-act", count=25, tagged=True)
        assert len(_detect(store)) == 1


# ---------------------------------------------------------------------------
# T0.4: deleted_at filter.
# ---------------------------------------------------------------------------


class TestT04_DeletedAtFilter:
    def test_deleted_at_set_skipped(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t04.db")
        _seed_conv(store, conv_id="conv-del",
                   deleted_at=_iso(_now()))
        _seed_turns(store, conv_id="conv-del", count=30, tagged=True)
        assert _detect(store) == []


# ---------------------------------------------------------------------------
# T0.5: any untagged canonical_turns row blocks detection.
# ---------------------------------------------------------------------------


class TestT05_UntaggedRowBlocks:
    def test_one_untagged_row_blocks(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t05.db")
        _seed_conv(store, conv_id="conv-untag")
        # 25 tagged + 1 untagged. Backlog count would qualify but the
        # NOT EXISTS predicate must block.
        _seed_turns(store, conv_id="conv-untag", count=25, tagged=True)
        _seed_turns(store, conv_id="conv-untag", count=1, tagged=False,
                    sort_key_base=2000.0, id_prefix="u")
        assert _detect(store) == []


# ---------------------------------------------------------------------------
# T0.6 / T0.7: active compaction_operation row blocks detection.
# ---------------------------------------------------------------------------


class TestT06_QueuedBlocks:
    def test_queued_op_blocks(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t06.db")
        _seed_conv(store, conv_id="conv-q")
        _seed_turns(store, conv_id="conv-q", count=25, tagged=True)
        _seed_compaction_op(
            store, op_id="op-queued", conv_id="conv-q", status="queued",
        )
        assert _detect(store) == []


class TestT07_RunningBlocks:
    def test_running_op_blocks(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t07.db")
        _seed_conv(store, conv_id="conv-r")
        _seed_turns(store, conv_id="conv-r", count=25, tagged=True)
        _seed_compaction_op(
            store, op_id="op-running", conv_id="conv-r", status="running",
        )
        assert _detect(store) == []


# ---------------------------------------------------------------------------
# T0.8 / T0.9: grace_s window on recent terminal compactions.
# ---------------------------------------------------------------------------


class TestT08_RecentTerminalSuppresses:
    def test_recent_terminal_within_grace_skipped(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t08.db")
        _seed_conv(store, conv_id="conv-recent")
        _seed_turns(store, conv_id="conv-recent", count=25, tagged=True)
        # last_terminal_at = now - grace_s/2 -> still inside the
        # grace window so detection skips.
        recent = _now() - timedelta(seconds=150)
        _seed_compaction_op(
            store, op_id="op-recent", conv_id="conv-recent",
            status="completed", completed_at=recent, started_at=recent,
        )
        assert _detect(store, grace_s=300.0) == []


class TestT09_OldTerminalSurfaces:
    def test_old_terminal_outside_grace_surfaces(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t09.db")
        _seed_conv(store, conv_id="conv-old-terminal")
        _seed_turns(store, conv_id="conv-old-terminal", count=25, tagged=True)
        # last_terminal_at = now - grace_s * 2 -> well outside the
        # grace window so detection surfaces.
        long_ago = _now() - timedelta(seconds=600)
        _seed_compaction_op(
            store, op_id="op-old", conv_id="conv-old-terminal",
            status="completed", completed_at=long_ago, started_at=long_ago,
        )
        candidates = _detect(store, grace_s=300.0)
        assert len(candidates) == 1
        assert candidates[0].conversation_id == "conv-old-terminal"
        assert candidates[0].last_terminal_compaction_at is not None


# ---------------------------------------------------------------------------
# T0.10: grace_s does not apply when there are zero compaction_operation
# rows for the conversation (the historical-burst-backfill case).
# ---------------------------------------------------------------------------


class TestT10_NoTerminalIgnoresGrace:
    def test_zero_ops_picks_up_immediately(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t10.db")
        _seed_conv(store, conv_id="conv-burst")
        _seed_turns(store, conv_id="conv-burst", count=25, tagged=True)
        # No compaction_operation rows at all. last_terminal_at IS NULL
        # so the WHERE clause's first branch (last_terminal IS NULL)
        # admits the conv.
        candidates = _detect(store, grace_s=300.0)
        assert len(candidates) == 1
        assert candidates[0].conversation_id == "conv-burst"
        assert candidates[0].last_terminal_compaction_at is None


# ---------------------------------------------------------------------------
# T0.11: ordering -- highest backlog_turns first.
# ---------------------------------------------------------------------------


class TestT11_OrderingDescending:
    def test_orders_by_backlog_desc(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t11.db")
        for conv, n in (("conv-a", 30), ("conv-b", 50), ("conv-c", 25)):
            _seed_conv(store, conv_id=conv)
            _seed_turns(store, conv_id=conv, count=n, tagged=True)
        candidates = _detect(store)
        assert [c.conversation_id for c in candidates] == [
            "conv-b", "conv-a", "conv-c",
        ]


# ---------------------------------------------------------------------------
# T0.12: limit parameter caps the result count.
# ---------------------------------------------------------------------------


class TestT12_LimitRespected:
    def test_limit_caps_result(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t12.db")
        for i in range(5):
            conv = f"conv-l{i}"
            _seed_conv(store, conv_id=conv)
            _seed_turns(
                store, conv_id=conv, count=30 + i, tagged=True,
            )
        candidates = _detect(store, limit=3)
        assert len(candidates) == 3
        # Highest backlog first.
        assert [c.conversation_id for c in candidates] == [
            "conv-l4", "conv-l3", "conv-l2",
        ]


# ---------------------------------------------------------------------------
# T0.13: old-epoch terminal compactions do not suppress a fresh
# lifecycle.
# ---------------------------------------------------------------------------


class TestT13a_MixedFormatTerminalsTextMaxHazard:
    """Regression for the codex P2 finding: a conversation with two
    terminal compaction ops -- one in ISO ``T``-separated format and
    one in SQLite space-separated format -- must compare correctly so
    the grace check picks the actually-newest terminal, not the one
    that sorts highest as plain text.

    Without ``datetime()`` normalization in the ``MAX``, lexicographic
    ordering treats ``'T'`` (codepoint 84) as greater than ``' '``
    (codepoint 32) and so the ISO-T timestamp always wins regardless
    of which is actually more recent. That bug would let a
    conversation with a very recent space-formatted terminal still
    surface as a backlog candidate because the text MAX picked an
    older ISO-T timestamp that falls outside the grace window.
    """

    def test_mixed_format_terminals_picks_actually_newest(
        self, tmp_path: Path,
    ):
        store = SQLiteStore(tmp_path / "t13a.db")
        _seed_conv(store, conv_id="conv-mixed")
        _seed_turns(store, conv_id="conv-mixed", count=25, tagged=True)
        # Older terminal in ISO ``T``-separated form (would sort high
        # under naive text MAX because of the 'T' separator).
        old_iso = _iso(_now() - timedelta(seconds=900))
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at,
                completed_at)
               VALUES (?, 'conv-mixed', 1, 0, 7, 'starting',
                       'completed', ?, ?, 'w-1', ?, ?)""",
            ("op-old-iso", old_iso, old_iso, old_iso, old_iso),
        )
        # NEWER terminal in SQLite space-separated format (inside
        # grace -- should suppress the conv). Note: NO 'T' separator
        # and no timezone suffix.
        recent_space = (_now() - timedelta(seconds=60)).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at,
                completed_at)
               VALUES (?, 'conv-mixed', 1, 0, 7, 'starting',
                       'completed', ?, ?, 'w-1', ?, ?)""",
            ("op-recent-space", recent_space, recent_space, recent_space,
             recent_space),
        )
        conn.commit()
        candidates = _detect(store, grace_s=300.0)
        assert not [c for c in candidates if c.conversation_id == "conv-mixed"], (
            "mixed-format terminals: text MAX picked the older ISO-T "
            "timestamp instead of the newer space-separated one; "
            "the recent terminal inside the grace window should "
            "suppress the conversation"
        )


class TestT13_OldEpochIgnored:
    def test_old_epoch_terminal_does_not_suppress(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "t13.db")
        # Conversation now at epoch 2; a recent terminal op exists at
        # epoch 1. The CTE groups by (conversation_id, lifecycle_epoch)
        # and the outer LEFT JOIN's matching predicate is also on
        # lifecycle_epoch, so the recent epoch-1 terminal must NOT
        # leak into the epoch-2 row's last_terminal_at.
        _seed_conv(store, conv_id="conv-bump", lifecycle_epoch=2)
        _seed_turns(store, conv_id="conv-bump", count=25, tagged=True)
        recent = _now() - timedelta(seconds=10)
        _seed_compaction_op(
            store, op_id="op-old-epoch", conv_id="conv-bump",
            lifecycle_epoch=1, status="completed",
            completed_at=recent, started_at=recent,
        )
        candidates = _detect(store, grace_s=300.0)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.conversation_id == "conv-bump"
        assert c.lifecycle_epoch == 2
        # No epoch-2 terminal exists, so last_terminal IS NULL.
        assert c.last_terminal_compaction_at is None
