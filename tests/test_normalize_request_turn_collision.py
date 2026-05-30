"""Regression: _normalize_request_turn_sequences must not violate the
unique index ``idx_request_context_conv_turn_unique`` while reassigning
rows whose stored ``request_turn`` values are out of order.

Root cause of the original bug:

* Storage bootstrap creates a UNIQUE INDEX on
  ``request_context(conversation_id, request_turn)``.
* Postgres (and SQLite) evaluate unique constraints at statement end
  (immediate, not deferred). So an UPDATE that assigns row A's
  ``request_turn`` to a value some other row B currently holds
  violates the constraint even when the FINAL post-normalization
  state would have no duplicates.
* ``_normalize_request_turn_sequences`` rewrites rows sequentially in
  ``(conversation_id, id)`` order, assigning ``seq = 1, 2, 3, ...``.
  When the kept rows for a conversation are NOT monotonically
  increasing in ``request_turn`` (typical after VCMERGE moves shift
  rows from the source into the target with an offset, after which
  trimming by ``id DESC`` keeps a mix that may include the target's
  pre-merge tail and the source's offset-aligned tail), the
  sequential UPDATEs can race against the unique index.

Concrete reproduction below: a conversation has two rows whose
``id`` ASC order is inverse to their stored ``request_turn``. The
old sequential implementation would assign row id=A (currently
``request_turn=2``) to ``request_turn=1``, which collides with the
other row id=B (currently ``request_turn=1``) on the unique index.
"""

from __future__ import annotations

import tempfile

import pytest

from virtual_context.storage.sqlite import SQLiteStore


_TS = "2026-05-29T00:00:00+00:00"


def _make_store() -> SQLiteStore:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return SQLiteStore(handle.name)


def _row_request_turn(store: SQLiteStore, row_id: int) -> int:
    conn = store._get_conn()
    row = conn.execute(
        "SELECT request_turn FROM request_context WHERE id = ?",
        (row_id,),
    ).fetchone()
    return int(row[0])


def _insert(
    store: SQLiteStore, *, conv_id: str, request_turn: int,
) -> int:
    """Insert a request_context row and return its assigned id.

    Uses INSERT OR IGNORE behavior is NOT needed; the unique index
    on (conversation_id, request_turn) is enforced and the caller
    designs the request_turn values not to collide on initial insert.
    """
    conn = store._get_conn()
    cur = conn.execute(
        """INSERT INTO request_context
           (conversation_id, request_turn, timestamp, user_message,
            inbound_tags, retrieval_method, candidates_found,
            candidates_selected, segments_injected, facts_injected,
            facts_count, facts_tags, pool_used, pool_budget,
            total_context_tokens, non_virtualizable_floor,
            tool_call_count)
           VALUES (?, ?, ?, '', '[]', '', 0, 0, '[]', '[]', 0,
                   '[]', 0, 0, 0, 0, 0)""",
        (conv_id, request_turn, _TS),
    )
    conn.commit()
    return int(cur.lastrowid)


class TestNormalizeAvoidsUniqueIndexCollision:
    """The bug: sequential UPDATE in (id ASC) order collides with the
    unique index when stored request_turn values are out of monotonic
    order. Re-running normalization must succeed and produce the
    expected 1..N sequence.
    """

    def test_inverse_order_does_not_violate_unique_index(self):
        store = _make_store()
        conv = "conv-collision-1"
        # Insert two rows where id ASC order is INVERSE of
        # request_turn order. The unique index allows this initial
        # insert because (conv, 2) and (conv, 1) are distinct keys.
        # Normalization expects: row(id=smaller) -> seq=1,
        # row(id=larger) -> seq=2. The sequential UPDATE on the
        # smaller id targets request_turn=1, which the larger id
        # already holds.
        row_a = _insert(store, conv_id=conv, request_turn=2)
        row_b = _insert(store, conv_id=conv, request_turn=1)
        assert row_a < row_b

        conn = store._get_conn()
        # The fix MUST allow this to complete without raising.
        store._normalize_request_turn_sequences(conn)
        conn.commit()

        # Final state: id ASC -> sequential 1, 2.
        assert _row_request_turn(store, row_a) == 1
        assert _row_request_turn(store, row_b) == 2

    def test_three_row_rotation_does_not_violate_unique_index(self):
        """Three rows with request_turn values rotated by one. Each
        sequential UPDATE produces an intermediate collision against
        the previous row's stored value.
        """
        store = _make_store()
        conv = "conv-collision-3"
        # Inserted order (by ascending id): request_turn = 3, 1, 2.
        # Target seq (by ascending id): 1, 2, 3.
        # UPDATE id=row_a SET request_turn=1 -> collides with row_b.
        row_a = _insert(store, conv_id=conv, request_turn=3)
        row_b = _insert(store, conv_id=conv, request_turn=1)
        row_c = _insert(store, conv_id=conv, request_turn=2)

        conn = store._get_conn()
        store._normalize_request_turn_sequences(conn)
        conn.commit()

        assert _row_request_turn(store, row_a) == 1
        assert _row_request_turn(store, row_b) == 2
        assert _row_request_turn(store, row_c) == 3

    def test_already_normalized_remains_unchanged(self):
        """Re-running normalization on a normalized table is a no-op."""
        store = _make_store()
        conv = "conv-already-normalized"
        row_a = _insert(store, conv_id=conv, request_turn=1)
        row_b = _insert(store, conv_id=conv, request_turn=2)
        row_c = _insert(store, conv_id=conv, request_turn=3)

        conn = store._get_conn()
        store._normalize_request_turn_sequences(conn)
        conn.commit()

        assert _row_request_turn(store, row_a) == 1
        assert _row_request_turn(store, row_b) == 2
        assert _row_request_turn(store, row_c) == 3

    def test_pass_one_rolls_back_when_pass_two_fails(self):
        """Atomicity contract: if pass 2 raises mid-flight, pass 1's
        negative sentinels must NOT remain in the table.

        Without the BEGIN IMMEDIATE wrapper, SQLite's autocommit mode
        would commit pass 1 immediately, leaving rows with negative
        ``request_turn`` values that a subsequent startup would then
        try to renormalize from a corrupted starting state. The two
        passes run inside a single explicit transaction so any
        exception between them rolls back both.

        Strategy: wrap the connection with a delegating proxy whose
        ``executemany`` raises on the second invocation. The real
        connection still owns the transaction (BEGIN IMMEDIATE +
        ROLLBACK go through the proxy back to the real connection).
        """
        store = _make_store()
        conv = "conv-rollback"
        row_a = _insert(store, conv_id=conv, request_turn=2)
        row_b = _insert(store, conv_id=conv, request_turn=1)

        real_conn = store._get_conn()

        class _FlakyConn:
            """Delegating proxy that intercepts the second
            ``executemany`` call to simulate a mid-pass failure.
            Every other attribute lookup goes straight through to the
            real sqlite3 connection.
            """

            def __init__(self, target):
                self._target = target
                self._em_calls = 0

            @property
            def in_transaction(self):
                return self._target.in_transaction

            def execute(self, *args, **kwargs):
                return self._target.execute(*args, **kwargs)

            def executemany(self, sql, params):
                self._em_calls += 1
                if self._em_calls == 2:
                    raise RuntimeError("synthetic pass-2 failure")
                return self._target.executemany(sql, params)

            def commit(self):
                return self._target.commit()

            def rollback(self):
                return self._target.rollback()

            def __getattr__(self, name):
                return getattr(self._target, name)

        flaky = _FlakyConn(real_conn)
        with pytest.raises(RuntimeError, match="synthetic"):
            store._normalize_request_turn_sequences(flaky)

        # Both rows must still have their ORIGINAL pre-normalization
        # request_turn values; no negative sentinel may remain.
        assert _row_request_turn(store, row_a) == 2
        assert _row_request_turn(store, row_b) == 1

    def test_post_vcmerge_offset_shift_does_not_collide(self):
        """Approximates a post-VCMERGE state: target conv has a
        pre-merge row at request_turn=1, then source rows arrive with
        ``request_turn += offset`` so their values are e.g. 2, 3, ...
        After several inserts and a trim, the kept set may not be
        monotonic in (id, request_turn). The normalization must rewrite
        it to 1..N without hitting the unique index mid-loop.
        """
        store = _make_store()
        conv = "conv-post-merge"
        # Simulate a kept-set shape after VCMERGE + trim: id ASC
        # order produces request_turn values 5, 1, 6, 2, 7, 3.
        # Target seq (by id ASC) is 1..6. The naive sequential UPDATE
        # of row(id=smallest, req=5) -> request_turn=1 collides with
        # row(id=2, req=1).
        rows = []
        for req in (5, 1, 6, 2, 7, 3):
            rows.append(_insert(store, conv_id=conv, request_turn=req))

        conn = store._get_conn()
        store._normalize_request_turn_sequences(conn)
        conn.commit()

        for expected_seq, row_id in enumerate(rows, start=1):
            assert _row_request_turn(store, row_id) == expected_seq, (
                f"row id={row_id} expected request_turn={expected_seq}, "
                f"got {_row_request_turn(store, row_id)}"
            )
