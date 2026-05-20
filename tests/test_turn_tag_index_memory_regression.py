"""Memory regression tests for ``TurnTagIndex``.

Pins behavior against two reference workloads:

* Production main-conv shape (2702 turns) — under MAX_ENTRIES; the
  index should remain fully indexed with zero evictions.
* Hypothetical longer conversation (5500 turns) — exceeds MAX_ENTRIES;
  the index should clamp at MAX_ENTRIES with deterministic eviction
  counts and dict-size invariants.
"""

from __future__ import annotations

from datetime import datetime, timezone

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import TurnTagEntry


def _entry(turn_number: int, tag_count: int = 3) -> TurnTagEntry:
    """Realistic-shape entry for memory-workload tests."""
    return TurnTagEntry(
        turn_number=turn_number,
        message_hash=f"hash-{turn_number}",
        canonical_turn_id=f"cid-{turn_number}",
        tags=[f"topic-{turn_number}-{k}" for k in range(tag_count)],
        primary_tag=f"topic-{turn_number}-0",
        timestamp=datetime.now(timezone.utc),
    )


def test_2702_turn_workload_stays_under_cap_no_eviction() -> None:
    """The user's main Telegram-mirror conv currently has 2702 turns.
    With MAX_ENTRIES=5_000 the cap is not exercised; the index is
    fully indexed and aggregation methods see all turns."""
    index = TurnTagIndex()
    for i in range(2702):
        index.append(_entry(i))
    assert len(index.entries) == 2702
    assert len(index._by_logical_turn) == 2702
    assert len(index._by_canonical_turn) == 2702
    assert len(index._by_hash) == 2702
    assert index._evicted_count == 0


def test_5500_turn_workload_caps_at_max_entries() -> None:
    """A hypothetical longer conversation exceeds MAX_ENTRIES.  The
    index clamps to MAX_ENTRIES and eviction count equals the overflow.

    All four reference structures stay coherent (single source of
    truth: every entry in ``entries`` has a slot in each dict; no
    orphan dict slots).
    """
    target = 5_500
    index = TurnTagIndex()
    for i in range(target):
        index.append(_entry(i))
    assert len(index.entries) == TurnTagIndex.MAX_ENTRIES
    assert index._evicted_count == target - TurnTagIndex.MAX_ENTRIES
    # Dict coherence: each surviving entry has exactly one slot in each dict.
    assert len(index._by_logical_turn) == TurnTagIndex.MAX_ENTRIES
    assert len(index._by_canonical_turn) == TurnTagIndex.MAX_ENTRIES
    assert len(index._by_hash) == TurnTagIndex.MAX_ENTRIES
    # Surviving turn-numbers are the most recent block: [500 .. 5499].
    surviving_turns = {e.turn_number for e in index.entries}
    expected = set(range(target - TurnTagIndex.MAX_ENTRIES, target))
    assert surviving_turns == expected


def test_aggregation_methods_operate_on_bounded_window() -> None:
    """After eviction, aggregation methods operate on the most-recent
    MAX_ENTRIES turns rather than the full history — the documented
    §2.3 trade-off."""
    target = TurnTagIndex.MAX_ENTRIES + 500
    index = TurnTagIndex()
    for i in range(target):
        index.append(_entry(i, tag_count=1))
    # get_tag_counts: each entry contributes one unique tag.
    counts = index.get_tag_counts()
    assert len(counts) == TurnTagIndex.MAX_ENTRIES
    # compute_cover_set: bounded-window cover.
    cover = index.compute_cover_set()
    assert len(cover) == TurnTagIndex.MAX_ENTRIES
