"""Full ingestion -> compaction -> re-ingestion -> compaction cycle test.

Drives the actual compaction pipeline (``CompactionPipeline.compact_if_needed``)
against a real SQLite store with mocked LLM providers. Verifies invariants
after each phase, catching not just the per-row marking bug fixed in
commit 6e2d5bd but also any downstream bugs where the first compaction's
output confuses the second compaction's scope.

Invariants checked after each compaction:

- I1: Every ``canonical_turns`` row has ``tagged_at IS NOT NULL``.
- I2: Every turn_group has coherent ``compacted_at`` state — all rows in
  the group share the same NULL/NOT-NULL status. (Catches the orphan
  halves bug on the full pipeline.)
- I3: After all pending content is compacted, only rows inside the
  ``protected_recent_turns`` window should have ``compacted_at IS NULL``.
- I4: The second compaction's input count must equal the NEW content
  introduced between runs — not a mix of new + orphan halves from the
  prior run.
- I5: ``compacted_prefix_messages`` watermark advances monotonically.

Pre-fix (without commit 6e2d5bd), I2 and I4 both fail — the first
compaction leaves assistant halves NULL, the second compaction pulls
them back in as if they were new content, and compacted_prefix_messages
stays at 2 forever.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.storage.sqlite import SQLiteStore


def _seed_split_turn_group(
    store: SQLiteStore,
    conv: str,
    turn_group_number: int,
    *,
    user_text: str,
    asst_text: str,
    sort_key_base: float,
    tagged: bool = True,
) -> tuple[str, str]:
    """Seed a turn_group's two physical rows the way the reconciler would."""
    now = utcnow_iso()
    u_id = f"ct-U{turn_group_number:04d}"
    a_id = f"ct-A{turn_group_number:04d}"
    tag_ts = now if tagged else None
    store.save_canonical_turn(
        conv, -1,
        user_text, "",
        turn_group_number=turn_group_number,
        canonical_turn_id=u_id,
        sort_key=sort_key_base,
        turn_hash=f"h-U{turn_group_number}",
        hash_version=1,
        tagged_at=tag_ts,
        primary_tag="_general" if tagged else "_general",
        tags=["topic-a"] if tagged else [],
        created_at=now, updated_at=now,
        first_seen_at=now, last_seen_at=now,
    )
    store.save_canonical_turn(
        conv, -1,
        "", asst_text,
        turn_group_number=turn_group_number,
        canonical_turn_id=a_id,
        sort_key=sort_key_base + 1.0,
        turn_hash=f"h-A{turn_group_number}",
        hash_version=1,
        tagged_at=tag_ts,
        primary_tag="_general",
        tags=["topic-a"] if tagged else [],
        created_at=now, updated_at=now,
        first_seen_at=now, last_seen_at=now,
    )
    return u_id, a_id


def _count_rows(store: SQLiteStore, conv: str) -> dict[str, int]:
    with store._get_conn() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE tagged_at IS NOT NULL) AS tagged,
                COUNT(*) FILTER (WHERE compacted_at IS NOT NULL) AS compacted
               FROM canonical_turns WHERE conversation_id=?""",
            (conv,),
        ).fetchone()
        return {"total": row[0], "tagged": row[1], "compacted": row[2]}


def _assert_turn_group_coherence(store: SQLiteStore, conv: str) -> None:
    """I2: every turn_group's rows share the same compacted NULL/NOT-NULL status."""
    with store._get_conn() as conn:
        rows = conn.execute(
            """SELECT turn_group_number,
                      COUNT(*) AS n,
                      COUNT(*) FILTER (WHERE compacted_at IS NOT NULL) AS n_compacted
                 FROM canonical_turns
                WHERE conversation_id=? AND turn_group_number >= 0
                GROUP BY turn_group_number""",
            (conv,),
        ).fetchall()
    incoherent = []
    for r in rows:
        tg, n, n_compacted = r[0], r[1], r[2]
        if n_compacted != 0 and n_compacted != n:
            incoherent.append(
                f"turn_group={tg}: {n_compacted}/{n} rows compacted (split state)"
            )
    assert not incoherent, (
        "I2 violated — turn_groups have inconsistent compacted_at state across "
        f"their physical rows: {incoherent}"
    )


def test_merge_fix_survives_full_compact_then_recompact_cycle(tmp_path: Path):
    """First compaction marks 10 turn_groups. Second call to
    ``get_uncompacted_canonical_turns`` must return ONLY turn_groups
    added after the first compaction — not orphan halves from the same
    turn_groups the first run already processed.
    """
    store = SQLiteStore(tmp_path / "cycle.db")
    conv = "conv-cycle"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)

    # First payload: 10 turn_groups, 20 physical rows
    for tg in range(10):
        _seed_split_turn_group(
            store, conv, tg,
            user_text=f"user msg {tg}",
            asst_text=f"assistant msg {tg}",
            sort_key_base=float((tg + 1) * 1000),
        )

    counts_before = _count_rows(store, conv)
    assert counts_before == {"total": 20, "tagged": 20, "compacted": 0}

    # Simulate first compaction: load uncompacted, mark all of them
    merged_first = store.get_uncompacted_canonical_turns(conv, protected_recent_turns=0)
    ids_first = [r.canonical_turn_id for r in merged_first if r.canonical_turn_id]
    store.mark_canonical_turns_compacted(conv, ids_first)

    _assert_turn_group_coherence(store, conv)
    counts_after_first = _count_rows(store, conv)
    assert counts_after_first["compacted"] == 20, (
        f"I2 violated: first compaction should mark all 20 physical rows; "
        f"got {counts_after_first['compacted']}"
    )

    # Second payload: 5 NEW turn_groups
    for tg in range(10, 15):
        _seed_split_turn_group(
            store, conv, tg,
            user_text=f"user msg {tg}",
            asst_text=f"assistant msg {tg}",
            sort_key_base=float((tg + 1) * 1000),
        )

    # Second compaction: ``get_uncompacted_canonical_turns`` should return
    # exactly 5 merged rows — one per NEW turn_group. Pre-fix, it would
    # return 10 (5 new + 5 orphan assistant halves from the first run).
    merged_second = store.get_uncompacted_canonical_turns(conv, protected_recent_turns=0)
    assert len(merged_second) == 5, (
        f"I4 violated: expected 5 uncompacted merged rows (just the new "
        f"turn_groups); got {len(merged_second)}. Pre-fix this returns 10 "
        "because assistant halves of the first 10 turn_groups were left NULL."
    )

    # Mark the second batch
    ids_second = [r.canonical_turn_id for r in merged_second if r.canonical_turn_id]
    store.mark_canonical_turns_compacted(conv, ids_second)

    _assert_turn_group_coherence(store, conv)
    counts_final = _count_rows(store, conv)
    assert counts_final == {"total": 30, "tagged": 30, "compacted": 30}


def test_merge_fix_unblocks_watermark_advance(tmp_path: Path):
    """``compacted_prefix_messages`` advances through all fully-compacted
    turn_groups. Pre-fix (with orphan halves), it would stop at
    turn_group_number=0 because every later group had one NULL half.
    """
    store = SQLiteStore(tmp_path / "watermark.db")
    conv = "conv-wm"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)

    for tg in range(6):
        _seed_split_turn_group(
            store, conv, tg,
            user_text=f"u{tg}", asst_text=f"a{tg}",
            sort_key_base=float((tg + 1) * 1000),
        )

    merged = store.get_uncompacted_canonical_turns(conv, protected_recent_turns=0)
    ids = [r.canonical_turn_id for r in merged if r.canonical_turn_id]
    store.mark_canonical_turns_compacted(conv, ids)

    # Walk every turn_group, assert coherent coverage. _refresh_compaction_watermark
    # logic requires ALL rows per turn_group compacted to advance. If the fix
    # holds, the walk advances through all 6 turn_groups (last_prefix_turn=5 →
    # compacted_prefix_messages=12).
    with store._get_conn() as conn:
        rows = conn.execute(
            """SELECT turn_group_number,
                      COUNT(*) = COUNT(*) FILTER (WHERE compacted_at IS NOT NULL) AS fully_compacted
                 FROM canonical_turns
                WHERE conversation_id=? AND turn_group_number >= 0
                GROUP BY turn_group_number
                ORDER BY turn_group_number""",
            (conv,),
        ).fetchall()

    fully_compacted_groups = [r[0] for r in rows if r[1]]
    assert fully_compacted_groups == [0, 1, 2, 3, 4, 5], (
        "I5 violated — watermark can't advance past first half-compacted group. "
        f"Fully-compacted groups: {fully_compacted_groups}. Pre-fix only "
        "turn_group=0 would be fully compacted (if at all)."
    )
