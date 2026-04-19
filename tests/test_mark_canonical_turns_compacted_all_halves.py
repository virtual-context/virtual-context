"""Regression: ``mark_canonical_turns_compacted`` must mark both halves
of a turn_group.

Observed in production on conv ``77f110fc-0c00`` (2026-04-19): the first
compaction marked 1107 user halves as compacted_at NOT NULL but left 1082
assistant halves NULL. The next ``get_uncompacted_canonical_turns`` call
re-surfaced those 1082 orphan halves as "needs compaction", triggering a
second full LLM pipeline run on content that had already been compacted
once. Also stuck ``compacted_prefix_messages`` at 2 forever because
``_refresh_compaction_watermark`` requires ALL rows in a turn_group to be
compacted before advancing.

Root cause: ``_merge_canonical_turn_rows`` in ``storage/{sqlite,postgres}.py``
collapses a turn_group's 2 physical rows (user-only row + assistant-only
row, each with its own canonical_turn_id) into ONE merged CanonicalTurnRow
that carries a single id. The compaction pipeline at
``compaction_pipeline.py:383-391`` then calls
``mark_canonical_turns_compacted`` with the merged rows' ids — hitting
only one of the two physical rows per turn_group.

Fix: ``mark_canonical_turns_compacted`` expands to also match rows
sharing a ``turn_group_number`` (>=0) with any of the provided ids.
Legacy rows with ``turn_group_number = -1`` keep id-only semantics so
the sentinel -1 doesn't cause mass-marking.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.storage.sqlite import SQLiteStore


def _seed_split_turn_group(
    store: SQLiteStore,
    conv: str,
    turn_group_number: int,
    *,
    u_id: str,
    a_id: str,
    sort_key_u: float,
    sort_key_a: float,
) -> None:
    now = utcnow_iso()
    store.save_canonical_turn(
        conv, -1,
        f"user tg{turn_group_number}", "",
        turn_group_number=turn_group_number,
        canonical_turn_id=u_id,
        sort_key=sort_key_u,
        turn_hash=f"h-{u_id}",
        hash_version=1,
        tagged_at=now,
        created_at=now, updated_at=now,
        first_seen_at=now, last_seen_at=now,
    )
    store.save_canonical_turn(
        conv, -1,
        "", f"assistant tg{turn_group_number}",
        turn_group_number=turn_group_number,
        canonical_turn_id=a_id,
        sort_key=sort_key_a,
        turn_hash=f"h-{a_id}",
        hash_version=1,
        tagged_at=now,
        created_at=now, updated_at=now,
        first_seen_at=now, last_seen_at=now,
    )


def test_mark_by_id_propagates_to_all_rows_in_turn_group(tmp_path: Path):
    """Core reproducer. Seed 3 turn_groups with split U/A rows; get
    uncompacted (merged) rows; mark by the merged id; assert ALL 6
    physical rows end up compacted.
    """
    store = SQLiteStore(tmp_path / "repro.db")
    conv = "conv-merge"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)

    for tg in range(3):
        _seed_split_turn_group(
            store, conv, tg,
            u_id=f"ct-U{tg}", a_id=f"ct-A{tg}",
            sort_key_u=float((tg * 2 + 1) * 1000),
            sort_key_a=float((tg * 2 + 2) * 1000),
        )

    merged = store.get_uncompacted_canonical_turns(conv, protected_recent_turns=0)
    assert len(merged) == 3, "one merged row per turn_group"
    merged_ids = [r.canonical_turn_id for r in merged if r.canonical_turn_id]
    assert len(merged_ids) == 3

    marked = store.mark_canonical_turns_compacted(conv, merged_ids)
    assert marked == 6, (
        f"Expected all 6 physical rows marked (2 halves x 3 turn_groups); "
        f"got {marked}. Pre-fix this returns 3 — the merged rows' single "
        "canonical_turn_id only matches one half per turn_group, leaving "
        "the other half orphaned."
    )

    with store._get_conn() as conn:
        rows = conn.execute(
            "SELECT canonical_turn_id, turn_group_number, compacted_at "
            "FROM canonical_turns WHERE conversation_id=? ORDER BY sort_key",
            (conv,),
        ).fetchall()
    compacted = sum(1 for r in rows if r[2] is not None)
    assert compacted == 6, f"physical rows with compacted_at NOT NULL: {compacted}/6"


def test_legacy_rows_with_turn_group_negative_one_only_match_by_id(tmp_path: Path):
    """Guard: rows without an explicit turn_group (the -1 sentinel) must
    NOT be mass-marked when one such id is passed. Otherwise passing one
    legacy id would mark every other legacy row in the conversation.
    """
    store = SQLiteStore(tmp_path / "legacy.db")
    conv = "conv-legacy"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    now = utcnow_iso()

    for i in range(3):
        with store._get_conn() as conn:
            conn.execute(
                """INSERT INTO canonical_turns (
                    canonical_turn_id, conversation_id, sort_key, turn_hash, hash_version,
                    user_content, assistant_content, tagged_at,
                    first_seen_at, last_seen_at, created_at, updated_at,
                    covered_ingestible_entries, turn_group_number
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, 1, -1)""",
                (
                    f"ct-L{i}", conv, float((i + 1) * 1000), f"h{i}",
                    f"u{i}", f"a{i}", now, now, now, now, now,
                ),
            )

    marked = store.mark_canonical_turns_compacted(conv, ["ct-L1"])
    assert marked == 1, f"legacy id-only match expected 1, got {marked}"

    with store._get_conn() as conn:
        rows = conn.execute(
            "SELECT canonical_turn_id, compacted_at FROM canonical_turns "
            "WHERE conversation_id=? ORDER BY sort_key",
            (conv,),
        ).fetchall()
    assert rows[0][1] is None, "ct-L0 must stay NULL"
    assert rows[1][1] is not None, "ct-L1 must be marked"
    assert rows[2][1] is None, "ct-L2 must stay NULL"


def test_mixed_turn_groups_do_not_cross_contaminate(tmp_path: Path):
    """Mark only tg=1's ids. Rows in tg=0 and tg=2 must stay NULL."""
    store = SQLiteStore(tmp_path / "mixed.db")
    conv = "conv-mixed"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)

    for tg in range(3):
        _seed_split_turn_group(
            store, conv, tg,
            u_id=f"ct-U{tg}", a_id=f"ct-A{tg}",
            sort_key_u=float((tg * 2 + 1) * 1000),
            sort_key_a=float((tg * 2 + 2) * 1000),
        )

    # Mark only the user half of tg=1; via group-expand, the assistant
    # half of tg=1 should also get marked — but not tg=0 or tg=2.
    marked = store.mark_canonical_turns_compacted(conv, ["ct-U1"])
    assert marked == 2, f"expected 2 (both halves of tg=1), got {marked}"

    with store._get_conn() as conn:
        rows = conn.execute(
            "SELECT canonical_turn_id, turn_group_number, compacted_at "
            "FROM canonical_turns WHERE conversation_id=? ORDER BY sort_key",
            (conv,),
        ).fetchall()
    status = {r[0]: r[2] is not None for r in rows}
    assert status["ct-U0"] is False
    assert status["ct-A0"] is False
    assert status["ct-U1"] is True
    assert status["ct-A1"] is True
    assert status["ct-U2"] is False
    assert status["ct-A2"] is False


def test_empty_id_list_is_noop(tmp_path: Path):
    store = SQLiteStore(tmp_path / "empty.db")
    conv = "conv-empty"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    marked = store.mark_canonical_turns_compacted(conv, [])
    assert marked == 0
