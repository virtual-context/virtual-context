"""Sort-key gap exhaustion in IngestReconciler (BUG-036).

``_allocate_sort_keys`` used to clamp its step to 0.001 when the gap
between the boundary rows was too tight for the requested count. The
clamped keys could land ON or PAST ``right_key``:

* a key equal to an existing row's key violates the
  ``UNIQUE (conversation_id, sort_key)`` constraint and aborts the whole
  prepare;
* a key past ``right_key`` that misses every existing key silently
  corrupts ordering — the inserted row sorts AFTER rows it should
  precede.

The failure is self-priming: the first clamped allocation writes rows
spaced exactly 0.001 apart, after which EVERY insertion between them
collides deterministically.

Fix: bounded allocation signals exhaustion instead of clamping, and the
reconciler opens the gap by shifting every row at or beyond the right
boundary upward (single UPDATE whose delta exceeds the shifted range's
spread, so it cannot transiently collide), then re-allocates.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from pathlib import Path

import pytest

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore


def _fmt():
    from virtual_context.proxy.formats import detect_format
    return detect_format({"messages": []})


def _reconciler(store: SQLiteStore) -> IngestReconciler:
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig, TagGeneratorConfig
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _pairs(*names) -> dict:
    msgs = []
    for n in names:
        msgs.append({"role": "user", "content": f"u{n}"})
        msgs.append({"role": "assistant", "content": f"a{n}"})
    return {"messages": msgs}


def _keys(db_path: Path) -> list[tuple[str, float]]:
    conn = sqlite3.connect(db_path)
    try:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT user_content || assistant_content, sort_key "
                "FROM canonical_turns WHERE conversation_id = 'c' "
                "ORDER BY sort_key"
            )
        ]
    finally:
        conn.close()


def _tighten_gap(db_path: Path, left_idx: int, spacing: float = 0.001) -> None:
    """Re-key every row after ``left_idx`` to ``spacing`` apart.

    Simulates the state a pre-fix clamped allocation left behind:
    adjacent rows whose gap can no longer host a midpoint.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT canonical_turn_id, sort_key FROM canonical_turns "
            "WHERE conversation_id = 'c' ORDER BY sort_key"
        ).fetchall()
        base = rows[left_idx][1]
        for offset, (ctid, _key) in enumerate(rows[left_idx + 1:], start=1):
            conn.execute(
                "UPDATE canonical_turns SET sort_key = ? WHERE canonical_turn_id = ?",
                (base + spacing * offset, ctid),
            )
        conn.commit()
    finally:
        conn.close()


class TestAllocatorBoundedGap:
    """Unit coverage: the allocator must never emit boundary-touching keys."""

    def _rec(self) -> IngestReconciler:
        return IngestReconciler.__new__(IngestReconciler)

    @pytest.mark.regression("BUG-036")
    def test_exhausted_gap_signals_none(self):
        rec = self._rec()
        assert rec._allocate_sort_keys(2000.0, 2000.001, 1) is None

    @pytest.mark.regression("BUG-036")
    def test_exhausted_gap_large_count_signals_none(self):
        rec = self._rec()
        assert rec._allocate_sort_keys(2000.0, 2001.0, 1500) is None

    def test_tight_but_sufficient_gap_still_allocates(self):
        # A gap whose step clears MIN_STEP allocates strictly inside —
        # no needless rebalance for merely-tight gaps.
        rec = self._rec()
        keys = rec._allocate_sort_keys(2000.0, 2000.004, 2)
        assert keys is not None
        assert all(2000.0 < k < 2000.004 for k in keys)
        assert keys == sorted(set(keys))

    def test_bounded_keys_strictly_inside(self):
        rec = self._rec()
        keys = rec._allocate_sort_keys(1000.0, 2000.0, 7)
        assert keys is not None
        assert all(1000.0 < k < 2000.0 for k in keys)
        assert keys == sorted(set(keys))

    def test_unbounded_branches_unchanged(self):
        rec = self._rec()
        assert rec._allocate_sort_keys(None, None, 2) == [1000.0, 2000.0]
        assert rec._allocate_sort_keys(4000.0, None, 2) == [5000.0, 6000.0]
        below = rec._allocate_sort_keys(None, 1000.0, 2)
        assert below is not None
        assert all(k < 1000.0 for k in below)

    def test_float_precision_degenerate_gap_signals_none(self):
        # A gap so tight relative to the magnitude that midpoints collapse
        # onto the boundaries must signal exhaustion, not return duplicates.
        rec = self._rec()
        left = 1e16
        right = left + 2.0  # ulp at 1e16 is 2.0 — no representable midpoint
        assert rec._allocate_sort_keys(left, right, 1) in (
            None,
            pytest.approx([left + 1.0]),
        )
        keys = rec._allocate_sort_keys(left, right, 1)
        if keys is not None:
            assert left < keys[0] < right


class TestMidInsertRebalance:
    """End-to-end: insertion into an exhausted gap must rebalance, not abort."""

    def _seeded(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        store.upsert_conversation(tenant_id="t", conversation_id="c")
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c", body=_pairs(0, 1, 2, 3), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        return store, rec

    @pytest.mark.regression("BUG-036")
    def test_mid_insert_into_exhausted_gap_succeeds(self, tmp_path: Path):
        store, rec = self._seeded(tmp_path)
        # Rows u3/a3 sit 0.001 apart from a2 — the post-clamp prod state.
        _tighten_gap(tmp_path / "vc.db", left_idx=5)

        # Overlap p0..p2 (6 rows ≥ anchor window), new pair X lands
        # between a2 and u3 — the exhausted gap.
        result = rec.ingest_batch(
            conversation_id="c", body=_pairs(0, 1, 2, "X"), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        assert result.turns_written == 2

        ordered = [content for content, _key in _keys(tmp_path / "vc.db")]
        assert ordered == [
            "u0", "a0", "u1", "a1", "u2", "a2", "uX", "aX", "u3", "a3",
        ]

    @pytest.mark.regression("BUG-036")
    def test_rebalance_keys_remain_unique(self, tmp_path: Path):
        store, rec = self._seeded(tmp_path)
        _tighten_gap(tmp_path / "vc.db", left_idx=5)
        rec.ingest_batch(
            conversation_id="c", body=_pairs(0, 1, 2, "X"), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        keys = [key for _content, key in _keys(tmp_path / "vc.db")]
        assert len(keys) == len(set(keys))
        assert keys == sorted(keys)

    @pytest.mark.regression("BUG-036")
    def test_rebalance_leaves_left_side_untouched(self, tmp_path: Path):
        store, rec = self._seeded(tmp_path)
        _tighten_gap(tmp_path / "vc.db", left_idx=5)
        before = dict(_keys(tmp_path / "vc.db"))
        rec.ingest_batch(
            conversation_id="c", body=_pairs(0, 1, 2, "X"), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        after = dict(_keys(tmp_path / "vc.db"))
        for content in ("u0", "a0", "u1", "a1", "u2", "a2"):
            assert after[content] == before[content], content

    @pytest.mark.regression("BUG-036")
    def test_rows_touched_sort_keys_match_db_after_rebalance(self, tmp_path: Path):
        store, rec = self._seeded(tmp_path)
        _tighten_gap(tmp_path / "vc.db", left_idx=5)
        result = rec.ingest_batch(
            conversation_id="c", body=_pairs(0, 1, 2, "X"), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        db_keys = {content: key for content, key in _keys(tmp_path / "vc.db")}
        for row in result.rows:
            content = (row.user_content or "") + (row.assistant_content or "")
            assert row.sort_key == db_keys[content], content

    @pytest.mark.regression("BUG-036")
    def test_repeated_insertions_never_collide(self, tmp_path: Path):
        """The original self-priming loop: repeated mid-insertions must
        keep succeeding indefinitely (each halving the gap pre-fix)."""
        store, rec = self._seeded(tmp_path)
        overlap = [0, 1, 2]
        for i in range(24):
            rec.ingest_batch(
                conversation_id="c",
                body=_pairs(*overlap, f"X{i}"),
                fmt=_fmt(),
                expected_lifecycle_epoch=1,
            )
        keys = [key for _content, key in _keys(tmp_path / "vc.db")]
        assert len(keys) == len(set(keys))

    @pytest.mark.regression("BUG-036")
    def test_per_row_fallback_when_store_lacks_bulk_shift(self, tmp_path: Path):
        store, rec = self._seeded(tmp_path)
        _tighten_gap(tmp_path / "vc.db", left_idx=5)

        class _NoShiftStore:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                if name == "shift_canonical_turn_sort_keys":
                    raise AttributeError(name)
                return getattr(self._inner, name)

        rec._store = _NoShiftStore(store)
        result = rec.ingest_batch(
            conversation_id="c", body=_pairs(0, 1, 2, "X"), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        assert result.turns_written == 2
        ordered = [content for content, _key in _keys(tmp_path / "vc.db")]
        assert ordered == [
            "u0", "a0", "u1", "a1", "u2", "a2", "uX", "aX", "u3", "a3",
        ]


class TestShiftHelperSQLite:
    def _store(self, tmp_path: Path) -> SQLiteStore:
        store = SQLiteStore(tmp_path / "vc.db")
        store.upsert_conversation(tenant_id="t", conversation_id="c")
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c", body=_pairs(0, 1, 2), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        return store

    def test_shifts_only_at_or_above_min(self, tmp_path: Path):
        store = self._store(tmp_path)
        moved = store.shift_canonical_turn_sort_keys(
            "c", min_sort_key=3000.0, delta=100000.0,
        )
        assert moved == 4  # rows at 3000..6000
        keys = [key for _content, key in _keys(tmp_path / "vc.db")]
        assert keys == [1000.0, 2000.0, 103000.0, 104000.0, 105000.0, 106000.0]

    def test_empty_range_returns_zero(self, tmp_path: Path):
        store = self._store(tmp_path)
        assert store.shift_canonical_turn_sort_keys(
            "c", min_sort_key=99999.0, delta=100000.0,
        ) == 0

    def test_insufficient_delta_rejected(self, tmp_path: Path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store.shift_canonical_turn_sort_keys(
                "c", min_sort_key=1000.0, delta=2000.0,
            )

    def test_non_positive_delta_rejected(self, tmp_path: Path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store.shift_canonical_turn_sort_keys(
                "c", min_sort_key=1000.0, delta=0.0,
            )
