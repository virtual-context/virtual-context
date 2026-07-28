"""Incremental anchor maintenance must equal a full anchor rebuild.

``_refresh_persisted_anchors`` used to rebuild the entire anchor table on
every ingest: a DELETE of every anchor row for the conversation followed by
a re-INSERT of one row per window of size 3, 4 and 5 across the whole
conversation. On a 9,200-row conversation that is 27,597 anchor rows
rewritten per ingest, of which an append-only turn changes exactly six
(two new rows times three window sizes) and removes none.

The incremental path derives the previously-persisted anchor set from the
pre-ingest row sequence, diffs it against the set the post-ingest sequence
requires, and writes only that difference. These tests pin the property the
optimisation rests on: **for any transition between two row sequences, the
persisted anchor set after the incremental write is byte-identical to the
set a full rebuild would have produced.**

Append is the easy case and is not sufficient on its own: it never has to
invalidate anything. The interior-insert, content-change, removal and
tail-shrink cases are the ones that exercise deletion, and a delta
implementation that only ever inserts passes the append case while
corrupting all four of the others.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import compute_anchor_hash
from virtual_context.core.ingest_reconciler import (
    _ALIGNMENT_WINDOW_SIZES,
    _ANCHOR_WINDOW_SIZES,
    IngestReconciler,
    _build_anchor_rows,
)
import virtual_context.core.ingest_reconciler as _ir
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CanonicalTurnRow


@pytest.fixture(autouse=True)
def _enable_incremental_anchor_writes(monkeypatch):
    """Exercise the delta path even though production runs the rebuild.

    Incremental writes are off in production because the read-modify-write
    races across workers. The machinery is kept and kept correct so it can
    be turned on once the write is idempotent, and that is only true if
    these tests keep running against it.
    """
    monkeypatch.setattr(_AnchorStore, "reconcile_excludes_concurrent_writers", True)


def _rows(tokens: list[str]) -> list[CanonicalTurnRow]:
    """Build a row sequence from short content tokens.

    ``canonical_turn_id`` is derived from the token so the same logical row
    keeps its identity across sequences, and ``turn_hash`` is derived from
    the token so a changed token models changed content.
    """
    return [
        CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id=f"id-{token}",
            turn_hash=f"hash-{token}",
            sort_key=float((idx + 1) * 1000.0),
        )
        for idx, token in enumerate(tokens)
    ]


def _canonical_anchor_set(rows: list[CanonicalTurnRow]) -> set[tuple[int, str, str]]:
    """The anchor set a full rebuild produces for ``rows``."""
    return {tuple(anchor) for anchor in _build_anchor_rows(rows)}


class _AnchorStore:
    """Store exposing only the surfaces the anchor refresh touches.

    Declines the exclusion capability by default, like a backend that has
    not established it. Tests exercising the delta opt in explicitly, so
    the opt-in is visible at the point it matters rather than ambient.
    """

    reconcile_excludes_concurrent_writers = False

    def __init__(self, rows: list[CanonicalTurnRow]) -> None:
        self.rows = list(rows)
        self.anchors: list[tuple[int, str, str]] = []
        self.full_rebuilds = 0
        self.delta_calls = 0
        self.rows_inserted = 0
        self.rows_deleted = 0

    def get_all_canonical_turns(self, conversation_id: str):
        return list(self.rows)

    def replace_canonical_turn_anchors(self, conversation_id: str, anchors):
        self.full_rebuilds += 1
        self.anchors = [tuple(anchor) for anchor in anchors]
        return len(self.anchors)

    def get_canonical_turn_anchors(self, conversation_id: str):
        return list(self.anchors)

    def apply_canonical_turn_anchor_delta(self, conversation_id: str, *, insert, delete):
        self.delta_calls += 1
        self.rows_inserted += len(insert)
        self.rows_deleted += len(delete)
        doomed = {tuple(anchor) for anchor in delete}
        self.anchors = [a for a in self.anchors if a not in doomed]
        self.anchors.extend(tuple(anchor) for anchor in insert)
        return len(insert)


def _reconciler(store) -> IngestReconciler:
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


# ---------------------------------------------------------------------------
# The identity property, across every transition shape.
# ---------------------------------------------------------------------------

_BASE = ["a", "b", "c", "d", "e", "f", "g", "h"]

TRANSITIONS = {
    # Append-only: the production steady state. Invalidates nothing.
    "append_one": (_BASE, _BASE + ["i"]),
    "append_pair": (_BASE, _BASE + ["i", "j"]),
    # Interior insert: every window spanning the insertion point changes.
    "interior_insert": (_BASE, ["a", "b", "c", "X", "d", "e", "f", "g", "h"]),
    # Prefix insert: shifts everything, invalidates only the leading windows.
    "prefix_insert": (_BASE, ["X", "a", "b", "c", "d", "e", "f", "g", "h"]),
    # Content change in place: same identity, different hash.
    "interior_modify": (_BASE, ["a", "b", "c", "D2", "e", "f", "g", "h"]),
    # Removal from the middle: windows spanning the hole must be rewritten
    # and the removed row's own anchors must disappear.
    "interior_removal": (_BASE, ["a", "b", "c", "e", "f", "g", "h"]),
    # Tail shrink: fewer rows means fewer valid window starts, so anchors
    # whose window no longer fits must be deleted even though every
    # surviving row is untouched.
    "tail_shrink": (_BASE, ["a", "b", "c", "d", "e"]),
    # Shrink below the largest window size entirely.
    "shrink_below_window": (_BASE, ["a", "b", "c"]),
    # Shrink below every window size: the anchor set becomes empty.
    "shrink_to_empty_set": (_BASE, ["a", "b"]),
    # Wholesale replacement: nothing survives.
    "full_replace": (_BASE, ["p", "q", "r", "s", "t"]),
}


@pytest.mark.regression("BUG-047")
@pytest.mark.parametrize("name", sorted(TRANSITIONS))
def test_incremental_delta_matches_full_rebuild(name):
    """The delta write must land on exactly the full-rebuild anchor set."""
    before, after = TRANSITIONS[name]
    before_rows, after_rows = _rows(before), _rows(after)

    store = _AnchorStore(before_rows)
    reconciler = _reconciler(store)

    # Seed the stored set, then forget how it got there.
    reconciler._refresh_persisted_anchors("c")
    assert set(store.anchors) == _canonical_anchor_set(before_rows)
    store.delta_calls = store.full_rebuilds = 0

    # Transition, then refresh.
    store.rows = after_rows
    reconciler._refresh_persisted_anchors("c")

    assert store.delta_calls == 1, "expected the incremental path, not a rebuild"
    assert store.full_rebuilds == 0
    assert set(store.anchors) == _canonical_anchor_set(after_rows)


@pytest.mark.regression("BUG-047")
@pytest.mark.parametrize("name", sorted(TRANSITIONS))
def test_incremental_result_is_order_independent(name):
    """Both temporal orderings converge on the same persisted set.

    Ordering A seeds with a full rebuild of the pre-state and then applies
    the delta. Ordering B reaches the same post-state by applying deltas
    from an empty conversation forward. Neither may diverge from the
    canonical rebuild of the post-state.
    """
    before, after = TRANSITIONS[name]
    before_rows, after_rows = _rows(before), _rows(after)
    expected = _canonical_anchor_set(after_rows)

    # Ordering A: rebuild the pre-state, then delta forward.
    store_a = _AnchorStore(before_rows)
    rec_a = _reconciler(store_a)
    rec_a._refresh_persisted_anchors("c")
    store_a.rows = after_rows
    rec_a._refresh_persisted_anchors("c")

    # Ordering B: build up from empty using only deltas.
    store_b = _AnchorStore([])
    rec_b = _reconciler(store_b)
    rec_b._refresh_persisted_anchors("c")
    store_b.rows = before_rows
    rec_b._refresh_persisted_anchors("c")
    store_b.rows = after_rows
    rec_b._refresh_persisted_anchors("c")

    assert set(store_a.anchors) == expected
    assert set(store_b.anchors) == expected
    assert set(store_a.anchors) == set(store_b.anchors)


# ---------------------------------------------------------------------------
# The cost property the change exists to buy.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-047")
def test_append_writes_bounded_anchor_rows_not_whole_conversation():
    """An append must touch a constant number of anchors, not O(N).

    This is the regression that matters operationally: the old code rewrote
    every anchor in the conversation on every ingest. A full-conversation
    rewrite fails this test at any conversation length.
    """
    before_rows = _rows([f"t{i}" for i in range(400)])
    after_rows = _rows([f"t{i}" for i in range(400)] + ["new"])

    store = _AnchorStore(before_rows)
    reconciler = _reconciler(store)
    reconciler._refresh_persisted_anchors("c")
    seeded = len(store.anchors)
    assert seeded > 1000, "fixture must be large enough for O(N) to be obvious"
    store.rows_inserted = store.rows_deleted = store.full_rebuilds = 0

    store.rows = after_rows
    reconciler._refresh_persisted_anchors("c")

    # One appended row creates exactly one new window per window size.
    # Derived from the constant rather than written as a literal, so the
    # assertion keeps measuring the invariant if the window set changes.
    assert store.rows_inserted == len(_ANCHOR_WINDOW_SIZES)
    assert store.rows_deleted == 0
    assert store.full_rebuilds == 0, "the append must not trigger a rebuild"


@pytest.mark.regression("BUG-047")
def test_unchanged_sequence_writes_nothing():
    """A resend that changes no row must issue no anchor write at all.

    ``ingest_single``'s exact-resend path mirrors stored identity and
    rewrites rows whose ``turn_hash`` matched, so the anchor set is
    provably unchanged. It used to pay a full rebuild anyway.
    """
    rows = _rows(_BASE)
    store = _AnchorStore(rows)
    reconciler = _reconciler(store)
    reconciler._refresh_persisted_anchors("c")
    baseline = set(store.anchors)
    store.delta_calls = store.full_rebuilds = 0

    reconciler._refresh_persisted_anchors("c")

    assert store.delta_calls == 0, "an unchanged sequence must write nothing"
    assert store.full_rebuilds == 0
    assert set(store.anchors) == baseline


# ---------------------------------------------------------------------------
# Divergence guard: the delta is only valid against a known prior state.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-047")
@pytest.mark.parametrize("corruption", ["missing", "stale", "duplicate", "swapped"])
def test_diverged_store_is_repaired_exactly(corruption):
    """Whatever state the table is in, one refresh restores the truth.

    The delta is computed against the anchors actually stored, not
    against what a prior row sequence implies they should be, so a
    diverged table converges instead of having the divergence written
    over and preserved. ``swapped`` is the case a cardinality check
    cannot see: one required anchor absent and one stale anchor present,
    so the row count is exactly right and the contents are wrong.
    """
    rows = _rows(_BASE)
    store = _AnchorStore(rows)
    reconciler = _reconciler(store)
    reconciler._refresh_persisted_anchors("c")
    truth = _canonical_anchor_set(rows)
    assert set(store.anchors) == truth

    bogus = (3, "hash-of-nothing", "id-ghost")
    if corruption == "missing":
        store.anchors.pop()
    elif corruption == "stale":
        store.anchors.append(bogus)
    elif corruption == "duplicate":
        store.anchors.append(store.anchors[0])
    elif corruption == "swapped":
        store.anchors.pop()
        store.anchors.append(bogus)
        assert len(store.anchors) == len(truth), "fixture must preserve cardinality"

    assert set(store.anchors) != truth or corruption == "duplicate"
    reconciler._refresh_persisted_anchors("c")

    assert set(store.anchors) == truth
    assert len(store.anchors) == len(truth), "duplicates must not survive repair"


@pytest.mark.regression("BUG-047")
def test_unreadable_anchor_store_rebuilds_rather_than_assuming_empty():
    """A backend that cannot report its anchors must not be read as empty.

    None and [] are different answers. Treating "cannot say" as "holds
    none" computes a delta that inserts the whole set on top of whatever
    is already there, duplicating every anchor in the conversation, and
    deletes nothing that has gone stale.
    """
    rows = _rows(_BASE)

    class _Unreadable(_AnchorStore):
        def get_canonical_turn_anchors(self, conversation_id: str):
            return None

    store = _Unreadable(rows)
    reconciler = _reconciler(store)
    reconciler._refresh_persisted_anchors("c")

    assert store.delta_calls == 0, "must not delta against an unknown set"
    assert store.full_rebuilds == 1
    assert set(store.anchors) == _canonical_anchor_set(rows)

    # And a second pass must not accumulate duplicates.
    reconciler._refresh_persisted_anchors("c")
    assert len(store.anchors) == len(_canonical_anchor_set(rows))


@pytest.mark.regression("BUG-047")
def test_store_without_delta_support_still_rebuilds():
    """A store lacking the delta surfaces keeps the old behavior."""

    class _LegacyStore(_AnchorStore):
        apply_canonical_turn_anchor_delta = None
        count_canonical_turn_anchors = None

    before_rows = _rows(_BASE)
    after_rows = _rows(_BASE + ["i"])
    store = _LegacyStore(before_rows)
    reconciler = _reconciler(store)
    reconciler._refresh_persisted_anchors("c")
    store.rows = after_rows
    reconciler._refresh_persisted_anchors("c")

    assert store.full_rebuilds == 2
    assert set(store.anchors) == _canonical_anchor_set(after_rows)


# ---------------------------------------------------------------------------
# Store-level: the SQL delta must equal the SQL rebuild.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-047")
@pytest.mark.parametrize("name", sorted(TRANSITIONS))
def test_sqlite_delta_write_matches_rebuild_write(tmp_path: Path, name):
    """Applying the delta leaves the same table state as replacing wholesale.

    The delta is derived the way the reconciler derives it: from the
    anchors the store actually reports, not from a remembered prior.
    """
    before, after = TRANSITIONS[name]
    desired = _build_anchor_rows(_rows(after))
    desired_set = set(desired)

    # Path 1: seed with the pre-state, then apply a delta computed against
    # what the store reports.
    delta_store = SQLiteStore(tmp_path / "delta.db")
    delta_store.replace_canonical_turn_anchors("c", _build_anchor_rows(_rows(before)))
    stored = delta_store.get_canonical_turn_anchors("c")
    counts: dict[tuple[int, str, str], int] = {}
    for anchor in stored:
        counts[anchor] = counts.get(anchor, 0) + 1
    to_delete = [a for a, n in counts.items() if a not in desired_set or n > 1]
    to_insert = [a for a in desired if counts.get(a, 0) != 1]
    delta_store.apply_canonical_turn_anchor_delta(
        "c", insert=to_insert, delete=to_delete,
    )

    # Path 2: replace wholesale with the desired set.
    rebuild_store = SQLiteStore(tmp_path / "rebuild.db")
    rebuild_store.replace_canonical_turn_anchors("c", desired)

    assert _stored_anchor_set(delta_store) == _stored_anchor_set(rebuild_store)
    assert _stored_anchor_set(delta_store) == desired_set
    assert len(delta_store.get_canonical_turn_anchors("c")) == len(desired_set)


@pytest.mark.regression("BUG-047")
def test_sqlite_reader_reports_duplicates(tmp_path: Path):
    """Duplicates must be visible, not collapsed.

    The table has no uniqueness constraint, so a repeated triple is real
    divergence. A reader that deduplicated would hide exactly the state
    the repair exists to fix.
    """
    store = SQLiteStore(tmp_path / "s.db")
    anchors = _build_anchor_rows(_rows(_BASE))
    store.replace_canonical_turn_anchors("c", anchors)
    store.apply_canonical_turn_anchor_delta("c", insert=[anchors[0]], delete=[])

    reported = store.get_canonical_turn_anchors("c")
    assert len(reported) == len(anchors) + 1
    assert reported.count(anchors[0]) == 2


@pytest.mark.regression("BUG-047")
def test_sqlite_delta_is_scoped_to_its_conversation(tmp_path: Path):
    """A delta must never touch a sibling conversation's anchors."""
    store = SQLiteStore(tmp_path / "s.db")
    mine = _build_anchor_rows(_rows(_BASE))
    theirs = _build_anchor_rows(_rows(["z1", "z2", "z3", "z4"]))
    store.replace_canonical_turn_anchors("c", mine)
    store.replace_canonical_turn_anchors("other", theirs)

    store.apply_canonical_turn_anchor_delta("c", insert=[], delete=list(mine))

    assert store.get_canonical_turn_anchors("c") == []
    assert len(store.get_canonical_turn_anchors("other")) == len(set(theirs))


def _stored_anchor_set(store: SQLiteStore) -> set[tuple[int, str, str]]:
    conn = store._get_conn()
    rows = conn.execute(
        "SELECT window_size, anchor_hash, start_turn_id "
        "FROM canonical_turn_anchors WHERE conversation_id = ?",
        ("c",),
    ).fetchall()
    return {(int(r[0]), str(r[1]), str(r[2])) for r in rows}


# ---------------------------------------------------------------------------
# The anchor set must remain usable for alignment after an incremental write.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-047")
def test_anchor_hashes_still_resolve_windows_after_delta():
    """Post-delta anchors must still map a window digest to its start row.

    ``_find_alignment`` trusts the persisted digest to position mapping
    without re-verifying the window, so a delta that leaves a stale digest
    behind would hand the aligner a bogus match.
    """
    before_rows = _rows(_BASE)
    after_rows = _rows(["a", "b", "c", "D2", "e", "f", "g", "h"])

    store = _AnchorStore(before_rows)
    reconciler = _reconciler(store)
    reconciler._refresh_persisted_anchors("c")
    store.rows = after_rows
    reconciler._refresh_persisted_anchors("c")

    for window_size in (3, 4, 5):
        for start in range(0, len(after_rows) - window_size + 1):
            digest = compute_anchor_hash(after_rows, start, window_size)
            expected = (window_size, digest, after_rows[start].canonical_turn_id)
            assert expected in store.anchors

    # And no digest from the superseded sequence may survive.
    stale = compute_anchor_hash(before_rows, 3, 3)
    assert not any(anchor[1] == stale for anchor in store.anchors)


# ---------------------------------------------------------------------------
# Production default: the rebuild, not the delta.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-047")
def test_a_store_that_declines_the_capability_rebuilds(monkeypatch):
    """With the default settings the refresh must rebuild, not diff.

    The delta races across workers and can leave an anchor set belonging
    to neither writer, so it stays off until the write is idempotent.
    This asserts the shipped default rather than the tested one: every
    other test in this file turns the delta on deliberately.
    """
    monkeypatch.setattr(_AnchorStore, "reconcile_excludes_concurrent_writers", False)
    rows = _rows(_BASE)
    store = _AnchorStore(rows)
    reconciler = _reconciler(store)

    reconciler._refresh_persisted_anchors("c")
    reconciler._refresh_persisted_anchors("c")

    assert store.delta_calls == 0, "the delta must not run by default"
    assert store.full_rebuilds == 2
    assert set(store.anchors) == _canonical_anchor_set(rows)
    # And the rebuild stays idempotent: no duplicate accumulation.
    assert len(store.anchors) == len(_canonical_anchor_set(rows))


@pytest.mark.regression("BUG-047")
def test_failed_delta_insert_does_not_leave_anchors_deleted(tmp_path: Path):
    """A delta that fails partway must not commit its deletions.

    Repairing a duplicate deletes every copy of a triple before adding one
    back. If the insert fails and the delete has already been committed,
    a required anchor is simply gone, and the aligner then gets a
    positional index missing an entry it expects.
    """
    store = SQLiteStore(tmp_path / "s.db")
    anchors = _build_anchor_rows(_rows(_BASE))
    store.replace_canonical_turn_anchors("c", anchors)
    before = set(store.get_canonical_turn_anchors("c"))

    doomed = anchors[:3]
    with pytest.raises(Exception):
        store.apply_canonical_turn_anchor_delta(
            "c",
            # A window_size that cannot be coerced fails at execute time,
            # after the deletions have run.
            insert=[(object(), "h", "id-x")],
            delete=doomed,
        )

    assert set(store.get_canonical_turn_anchors("c")) == before, (
        "a failed insert must not leave its deletions behind"
    )


@pytest.mark.regression("BUG-047")
def test_real_backends_declare_the_capability_honestly():
    """Each backend must state whether its reconcile actually excludes.

    Postgres holds a row lock on a pooled connection and other store
    calls take their own, so nothing it invokes can end the reconcile.
    SQLite holds BEGIN IMMEDIATE, but its prevailing read idiom commits
    on block exit, so a read from inside the reconcile ends it. The
    delta is only correct on the former, and the gate is a claim each
    backend makes rather than a switch someone flips globally.
    """
    from virtual_context.storage.postgres import PostgresStore
    from virtual_context.storage.sqlite import SQLiteStore

    assert PostgresStore.reconcile_excludes_concurrent_writers is True
    assert SQLiteStore.reconcile_excludes_concurrent_writers is False


@pytest.mark.regression("BUG-047")
def test_a_store_silent_on_the_capability_rebuilds():
    """Silence is not consent.

    A store that never mentions the capability, as a third-party backend
    would not, gets the rebuild. The delta has to be claimed by a backend
    that established the exclusion, never inherited by anything that
    happens to expose the delta surfaces.
    """
    rows = _rows(_BASE)
    inner = _AnchorStore(rows)

    class _Silent:
        """Delegates everything except the capability, which it lacks."""

        def __getattr__(self, name):
            if name == "reconcile_excludes_concurrent_writers":
                raise AttributeError(name)
            return getattr(inner, name)

    store = _Silent()
    assert not hasattr(store, "reconcile_excludes_concurrent_writers")

    reconciler = _reconciler(store)
    reconciler._refresh_persisted_anchors("c")

    assert inner.delta_calls == 0, "an unproven store must not get the delta"
    assert inner.full_rebuilds == 1
    assert set(inner.anchors) == _canonical_anchor_set(rows)
