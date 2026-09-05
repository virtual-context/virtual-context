"""Bounded physical-source hydration for trusted maintenance boundaries.

These helpers never enumerate conversation history or infer authorship from
logical merged rows. Missing/foreign physical rows leave provenance incomplete.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import CanonicalTurnRow

SOURCE_BATCH_SIZE = 256


def physical_rows_by_id(store, keys: Iterable[tuple[str, str]]) -> dict:
    """Hydrate literal owner/id keys, in bounded storage calls."""
    getter = getattr(store, "get_canonical_turn_rows_by_id", None)
    if not callable(getter):
        return {}
    ordered = list(
        dict.fromkeys(
            (owner, source_id)
            for owner, source_id in keys
            if type(owner) is str
            and owner
            and owner == owner.strip()
            and type(source_id) is str
            and source_id
            and source_id == source_id.strip()
        )
    )
    found = {}
    for start in range(0, len(ordered), SOURCE_BATCH_SIZE):
        batch = ordered[start : start + SOURCE_BATCH_SIZE]
        requested = set(batch)
        rows = getter(batch, internal_validation=True)
        for key, row in rows.items():
            if (
                key in requested
                and (
                    row.conversation_id,
                    row.canonical_turn_id,
                )
                == key
            ):
                found[key] = row
    return found


def physical_rows_by_group(
    store,
    conversation_id: str,
    groups: Iterable[int],
) -> dict[int, list[CanonicalTurnRow]]:
    """Load every physical sibling of explicitly selected nonnegative groups."""
    getter = getattr(store, "get_canonical_turn_rows_by_group", None)
    if not callable(getter):
        return {}
    ordered = sorted({group for group in groups if type(group) is int and group >= 0})
    found: dict[int, list[CanonicalTurnRow]] = {}
    for start in range(0, len(ordered), SOURCE_BATCH_SIZE):
        batch = ordered[start : start + SOURCE_BATCH_SIZE]
        requested = set(batch)
        rows = getter(conversation_id, batch, internal_validation=True)
        for row in rows:
            group = row.turn_group_number
            if row.conversation_id == conversation_id and group in requested:
                found.setdefault(group, []).append(row)
    return found


def reply_target_rows(store, conversation_id: str, source_rows: Iterable) -> dict:
    """Load historical reply targets without losing duplicate-match ambiguity."""
    getter = getattr(store, "get_canonical_turn_rows_by_source_message_ids", None)
    if not callable(getter):
        return {}
    targets = sorted(
        {
            str(getattr(row, "reply_target_message_id", "") or "").strip()
            for row in source_rows
            if (getattr(row, "reply_target_body", "") or "").strip()
        }
        - {""}
    )
    found = {}
    for start in range(0, len(targets), SOURCE_BATCH_SIZE):
        batch = targets[start : start + SOURCE_BATCH_SIZE]
        requested = set(batch)
        rows = getter(conversation_id, batch, internal_validation=True)
        for row in rows:
            if (
                row.conversation_id == conversation_id
                and (row.source_message_id or "").strip() in requested
            ):
                found[row.canonical_turn_id] = row
    return found
