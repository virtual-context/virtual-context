"""Property-based parity tests for _merge_canonical_turn_rows across backends.

Both storage backends ship byte-identical copies of _merge_canonical_turn_rows
(postgres.py and sqlite.py) and share the same content-heuristic fallback for
conversations that were ingested before turn_group_number was introduced. If
these implementations ever drift, legacy conversations would pair turns
differently than modern ones silently. These tests generate a broad set of
canonical row sequences and assert that:

    1. Both backends' merge helpers produce identical output.
    2. Merging is idempotent under repeated calls.
    3. Explicit turn_group_number assignments and content-heuristic fallback
       converge to the same grouping when both signals are consistent.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
from dataclasses import asdict, replace

import pytest

from virtual_context.storage.postgres import _merge_canonical_turn_rows as _merge_pg
from virtual_context.storage.sqlite import _merge_canonical_turn_rows as _merge_sqlite
from virtual_context.types import CanonicalTurnRow


_ROLE_STATES = ("user_only", "assistant_only", "both")
_GROUP_MODES = ("all_legacy", "all_explicit", "mixed")


def _make_row(
    *,
    idx: int,
    role: str,
    turn_group_number: int,
    conversation_id: str = "conv-a",
) -> CanonicalTurnRow:
    user_text = f"u-{idx}" if role in ("user_only", "both") else ""
    assistant_text = f"a-{idx}" if role in ("assistant_only", "both") else ""
    return CanonicalTurnRow(
        conversation_id=conversation_id,
        canonical_turn_id=f"ct-{idx:04d}",
        turn_number=idx,
        turn_group_number=turn_group_number,
        sort_key=float(idx),
        turn_hash=f"h{idx}",
        hash_version=1,
        normalized_user_text=user_text,
        normalized_assistant_text=assistant_text,
        user_content=user_text,
        assistant_content=assistant_text,
        primary_tag="_general",
        tags=[],
        session_date="",
        sender="",
        fact_signals=[],
        code_refs=[],
        created_at=f"2026-04-01T00:00:{idx:02d}",
        updated_at=f"2026-04-01T00:00:{idx:02d}",
    )


def _heuristic_groups(roles: list[str]) -> list[int]:
    # Mirror the content-heuristic grouping that _merge_canonical_turn_rows
    # falls back to when turn_group_number is missing. Used to derive explicit
    # group assignments that match the heuristic output for the convergence
    # check.
    groups: list[int] = []
    pending_user_group: int | None = None
    next_group = 0
    for role in roles:
        has_user = role in ("user_only", "both")
        has_assistant = role in ("assistant_only", "both")
        if has_user and has_assistant:
            if pending_user_group is not None:
                groups.append(next_group)
                next_group += 1
                pending_user_group = None
            else:
                groups.append(next_group)
                next_group += 1
            continue
        if has_user:
            if pending_user_group is not None:
                groups.append(next_group)
                next_group += 1
            pending_user_group = next_group
            groups.append(next_group)
            next_group += 1
            continue
        if has_assistant:
            if pending_user_group is not None:
                groups.append(pending_user_group)
                pending_user_group = None
            else:
                groups.append(next_group)
                next_group += 1
            continue
    return groups


def _generate_scenario(seed: int) -> tuple[list[CanonicalTurnRow], str]:
    rng = random.Random(seed)
    length = rng.randint(1, 20)
    roles = [rng.choice(_ROLE_STATES) for _ in range(length)]
    mode = rng.choice(_GROUP_MODES)

    if mode == "all_explicit":
        groups = _heuristic_groups(roles)
    elif mode == "all_legacy":
        groups = [-1] * length
    else:
        heuristic = _heuristic_groups(roles)
        groups = [
            heuristic[i] if rng.random() < 0.5 else -1
            for i in range(length)
        ]

    rows = [
        _make_row(idx=i, role=roles[i], turn_group_number=groups[i])
        for i in range(length)
    ]
    return rows, mode


def _normalize_for_compare(rows_map: dict) -> list[tuple]:
    ordered: list[tuple] = []
    for turn_number in sorted(rows_map.keys()):
        row = rows_map[turn_number]
        ordered.append((
            turn_number,
            row.user_content,
            row.assistant_content,
            row.canonical_turn_id,
            row.turn_group_number,
            sorted(row.tags),
        ))
    return ordered


@pytest.mark.parametrize("seed", list(range(50)))
def test_merge_parity_between_backends(seed: int) -> None:
    rows, _mode = _generate_scenario(seed)
    merged_pg = _merge_pg(list(rows))
    merged_sqlite = _merge_sqlite(list(rows))
    assert _normalize_for_compare(merged_pg) == _normalize_for_compare(merged_sqlite), (
        f"Backend drift on seed={seed}: inputs={[(r.user_content, r.assistant_content, r.turn_group_number) for r in rows]}"
    )


@pytest.mark.parametrize("seed", list(range(30)))
def test_merge_idempotent(seed: int) -> None:
    rows, _mode = _generate_scenario(seed)
    first = _merge_pg(list(rows))
    # Re-feed the merged rows back through the merger. The output should be
    # stable because each merged row has a unique turn_group_number matching
    # its turn_number, so the explicit-group branch activates on pass 2.
    second = _merge_pg(list(first.values()))
    assert _normalize_for_compare(first) == _normalize_for_compare(second)


def test_empty_input_returns_empty() -> None:
    assert _merge_pg([]) == {}
    assert _merge_sqlite([]) == {}


def test_explicit_groups_match_heuristic_when_consistent() -> None:
    # When every row has an explicit turn_group_number that matches what the
    # content heuristic would have produced, both code paths converge — this
    # is the invariant the lazy backfill (A3) relies on: recomputing legacy
    # conversations does not change their visible structure.
    roles = ["user_only", "assistant_only", "both", "user_only", "assistant_only"]
    heuristic_groups = _heuristic_groups(roles)
    explicit_rows = [
        _make_row(idx=i, role=roles[i], turn_group_number=heuristic_groups[i])
        for i in range(len(roles))
    ]
    legacy_rows = [replace(row, turn_group_number=-1) for row in explicit_rows]

    merged_explicit = _merge_pg(explicit_rows)
    merged_legacy = _merge_pg(legacy_rows)

    assert _normalize_for_compare(merged_explicit) == _normalize_for_compare(merged_legacy)


def test_mixed_legacy_falls_back_to_heuristic() -> None:
    # Any row with turn_group_number < 0 forces the whole list through the
    # heuristic path — this is deliberate, because we can't mix-and-match.
    roles = ["user_only", "assistant_only", "user_only", "assistant_only"]
    heuristic_groups = _heuristic_groups(roles)
    rows = [
        _make_row(idx=i, role=roles[i], turn_group_number=heuristic_groups[i])
        for i in range(len(roles))
    ]
    # Flip one to -1; result should be the same turn-count as pure heuristic.
    rows[1] = replace(rows[1], turn_group_number=-1)
    merged = _merge_pg(rows)

    pure_heuristic = _merge_pg([
        replace(row, turn_group_number=-1) for row in rows
    ])
    assert _normalize_for_compare(merged) == _normalize_for_compare(pure_heuristic)
