"""Tests for ``TurnTagIndex`` LRU bounding with dict coherence.

Covers the append-side eviction trigger, identity-guarded dict
removal (the load-bearing guard against silently deleting a
duplicate-rejection-survivor's dict slot), `_all_tags` recomputation
on eviction, and the eviction counter.
"""

from __future__ import annotations

from datetime import datetime, timezone

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import TurnTagEntry


def _entry(
    turn_number: int,
    *,
    canonical_turn_id: str = "",
    message_hash: str = "",
    tags: list[str] | None = None,
    primary_tag: str = "_general",
) -> TurnTagEntry:
    """Build a minimal TurnTagEntry for index tests."""
    return TurnTagEntry(
        turn_number=turn_number,
        message_hash=message_hash or f"hash-{turn_number}",
        canonical_turn_id=canonical_turn_id or f"cid-{turn_number}",
        tags=list(tags) if tags is not None else [f"topic-{turn_number}"],
        primary_tag=primary_tag,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# MAX_ENTRIES constant + post-condition
# ---------------------------------------------------------------------------


def test_max_entries_constant_value() -> None:
    """Defensive assertion: surface accidental cap regressions."""
    assert TurnTagIndex.MAX_ENTRIES == 5_000


def test_append_under_cap_grows_normally() -> None:
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES - 1):
        index.append(_entry(i))
    assert len(index.entries) == TurnTagIndex.MAX_ENTRIES - 1
    assert len(index._by_logical_turn) == TurnTagIndex.MAX_ENTRIES - 1
    assert len(index._by_canonical_turn) == TurnTagIndex.MAX_ENTRIES - 1
    assert len(index._by_hash) == TurnTagIndex.MAX_ENTRIES - 1
    assert index._evicted_count == 0


def test_append_at_cap_evicts_oldest() -> None:
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES):
        index.append(_entry(i))
    # Sanity at-cap.
    assert len(index.entries) == TurnTagIndex.MAX_ENTRIES
    # Trigger eviction.
    index.append(_entry(TurnTagIndex.MAX_ENTRIES))
    assert len(index.entries) == TurnTagIndex.MAX_ENTRIES
    assert index._evicted_count == 1
    # Oldest entry (turn 0) gone from all 4 structures.
    assert 0 not in index._by_logical_turn
    assert "cid-0" not in index._by_canonical_turn
    assert "hash-0" not in index._by_hash
    # Newest entry present in all structures.
    assert TurnTagIndex.MAX_ENTRIES in index._by_logical_turn
    assert f"cid-{TurnTagIndex.MAX_ENTRIES}" in index._by_canonical_turn
    assert f"hash-{TurnTagIndex.MAX_ENTRIES}" in index._by_hash


def test_eviction_count_increments_per_evicted_entry() -> None:
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES + 3):
        index.append(_entry(i))
    assert index._evicted_count == 3
    assert len(index.entries) == TurnTagIndex.MAX_ENTRIES


# ---------------------------------------------------------------------------
# _all_tags recomputation
# ---------------------------------------------------------------------------


def test_eviction_drops_orphan_tags_from_all_tags() -> None:
    """A tag carried ONLY by the eventually-evicted entry must drop
    out of ``_all_tags`` after that entry is evicted."""
    index = TurnTagIndex()
    # Entry 0 is the only carrier of "orphan-tag"; everyone else has shared tags.
    index.append(_entry(0, tags=["shared", "orphan-tag"]))
    for i in range(1, TurnTagIndex.MAX_ENTRIES):
        index.append(_entry(i, tags=["shared"]))
    assert "orphan-tag" in index._all_tags
    assert "shared" in index._all_tags
    # Trigger one eviction.
    index.append(_entry(TurnTagIndex.MAX_ENTRIES, tags=["shared"]))
    assert "orphan-tag" not in index._all_tags
    assert "shared" in index._all_tags


def test_eviction_preserves_shared_tags() -> None:
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES + 1):
        index.append(_entry(i, tags=["always-present"]))
    assert "always-present" in index._all_tags
    assert index._evicted_count == 1


# ---------------------------------------------------------------------------
# Identity guard against duplicate-detection-survivor corruption
# ---------------------------------------------------------------------------


def test_duplicate_turn_number_rejected_then_eviction_keeps_dict_pointing_at_survivor() -> None:
    """If the duplicate-detection guard rejects an incoming entry that
    would share a ``turn_number`` with the OLDEST entry, eviction must
    not silently delete the survivor's dict slot.

    Scenario: oldest entry has turn_number=0. A later append with
    turn_number=0 is REJECTED (kept original). When eviction fires on
    the cap, the oldest entry (turn 0) is evicted and dict slot
    ``_by_logical_turn[0]`` should be removed because it points AT
    that same evicted entry (identity match). The rejected duplicate
    never made it into a dict slot, so the identity guard correctly
    drops the slot.
    """
    index = TurnTagIndex()
    # Append the original turn-0.
    index.append(_entry(0, tags=["original"]))
    # Try to append a duplicate turn-0 — rejected.
    index.append(_entry(0, tags=["rejected-duplicate"]))
    assert len(index.entries) == 1  # duplicate rejected
    # Fill to cap.
    for i in range(1, TurnTagIndex.MAX_ENTRIES + 1):
        index.append(_entry(i))
    # The original turn-0 was the oldest and got evicted.
    assert 0 not in index._by_logical_turn


def test_eviction_uses_identity_guard_on_logical_turn() -> None:
    """Eviction must not remove a dict slot that does not point at the
    evicted entry.  Constructed scenario: pre-populate the index with
    an entry whose turn_number matches the to-be-evicted entry's id
    via direct dict manipulation, then evict.
    """
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES):
        index.append(_entry(i))
    # Direct manipulation: replace the dict slot for turn 0 with a
    # different (non-evicted) entry.  Eviction should NOT delete this
    # slot because the identity check fails.
    bogus_replacement = _entry(0, tags=["replacement"])
    index._by_logical_turn[0] = bogus_replacement
    # Trigger eviction.
    index.append(_entry(TurnTagIndex.MAX_ENTRIES))
    # Identity guard: slot 0 still points at the replacement, NOT removed.
    assert 0 in index._by_logical_turn
    assert index._by_logical_turn[0] is bogus_replacement


# ---------------------------------------------------------------------------
# Tail and lookup invariants
# ---------------------------------------------------------------------------


def test_get_active_tags_tail_invariant() -> None:
    """Eviction must not corrupt the tail of ``entries`` that
    ``get_active_tags`` slices."""
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES + 100):
        index.append(_entry(i, tags=[f"tag-{i}"]))
    # Tail of MAX_ENTRIES contains the most recent 4 entries.
    active = index.get_active_tags(lookback=4)
    # The 4 most recent turn numbers are MAX_ENTRIES+96 .. MAX_ENTRIES+99.
    expected = {f"tag-{TurnTagIndex.MAX_ENTRIES + 100 - 1 - k}" for k in range(4)}
    assert active == expected


def test_lookup_evicted_entry_returns_none() -> None:
    """Lookups for evicted entries return None from all three dict
    accessors."""
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES + 10):
        index.append(_entry(i))
    # Turns 0..9 are evicted.
    for evicted in range(10):
        assert index.get_tags_for_logical_turn(evicted) is None
        assert index.get_tags_for_canonical_turn(f"cid-{evicted}") is None
        assert index.get_entry_by_hash(f"hash-{evicted}") is None
    # A surviving turn is reachable.
    assert index.get_tags_for_logical_turn(TurnTagIndex.MAX_ENTRIES + 9) is not None


def test_latest_meaningful_tags_unaffected_by_eviction() -> None:
    """The tail-walker for ``latest_meaningful_tags`` reads from the
    tail; eviction at the head does not change tail semantics."""
    index = TurnTagIndex()
    for i in range(TurnTagIndex.MAX_ENTRIES + 50):
        index.append(_entry(i, tags=[f"topic-{i}"]))
    latest = index.latest_meaningful_tags()
    assert latest is not None
    assert latest.turn_number == TurnTagIndex.MAX_ENTRIES + 50 - 1
