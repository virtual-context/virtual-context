"""Backend tests for ``has_any_alias`` + ``get_recent_canonical_turns`` on SQLite.

The Postgres backend has a matching test file
(``tests/test_postgres_mirror_store_methods.py``) that mirrors these
cases against a live Postgres instance when ``DATABASE_URL`` is set.
This SQLite suite is the always-on coverage for the protocol shape;
the Postgres variant guards index-name + SQL-dialect parity.
"""

from __future__ import annotations

import pytest

from virtual_context.storage.sqlite import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "store.db"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# has_any_alias
# ---------------------------------------------------------------------------


def test_has_any_alias_returns_false_when_no_rows(store: SQLiteStore) -> None:
    assert store.has_any_alias("conv-unattached") is False


def test_has_any_alias_returns_true_for_source(store: SQLiteStore) -> None:
    store.save_conversation_alias("source-1", "target-1")
    assert store.has_any_alias("source-1") is True


def test_has_any_alias_returns_true_for_target(store: SQLiteStore) -> None:
    store.save_conversation_alias("source-1", "target-1")
    assert store.has_any_alias("target-1") is True


def test_has_any_alias_returns_true_for_both(store: SQLiteStore) -> None:
    # A -> B and C -> A. ``A`` is both source (outgoing to B) and
    # target (incoming from C).
    store.save_conversation_alias("A", "B")
    store.save_conversation_alias("C", "A")
    assert store.has_any_alias("A") is True


def test_has_any_alias_unrelated_conv_returns_false(store: SQLiteStore) -> None:
    store.save_conversation_alias("A", "B")
    assert store.has_any_alias("D") is False


def test_has_any_alias_empty_conv_id_returns_false(store: SQLiteStore) -> None:
    assert store.has_any_alias("") is False


# ---------------------------------------------------------------------------
# get_recent_canonical_turns
# ---------------------------------------------------------------------------


def _seed_turn(
    store: SQLiteStore,
    conversation_id: str,
    turn_number: int,
    *,
    user_content: str = "u",
    assistant_content: str = "a",
    sort_key: float | None = None,
    tagged_at: str | None = None,
) -> None:
    store.save_canonical_turn(
        conversation_id,
        turn_number,
        user_content,
        assistant_content,
        primary_tag="topic",
        tags=["topic"],
        sort_key=sort_key,
        tagged_at=tagged_at,
    )


def test_get_recent_returns_empty_for_unknown_conv(store: SQLiteStore) -> None:
    rows = store.get_recent_canonical_turns("missing", limit=5)
    assert rows == []


def test_get_recent_limit_honored(store: SQLiteStore) -> None:
    for i in range(5):
        _seed_turn(store, "conv-1", i)
    rows = store.get_recent_canonical_turns("conv-1", limit=2)
    assert len(rows) == 2


def test_get_recent_ordered_desc_by_sort_key(store: SQLiteStore) -> None:
    for i in range(3):
        _seed_turn(store, "conv-1", i)
    rows = store.get_recent_canonical_turns("conv-1", limit=10)
    # Most recent first; sort_key is monotonically assigned by ingest.
    sort_keys = [r.sort_key for r in rows]
    assert sort_keys == sorted(sort_keys, reverse=True)


def test_get_recent_conversation_scoped(store: SQLiteStore) -> None:
    _seed_turn(store, "conv-1", 0)
    _seed_turn(store, "conv-2", 0)
    conv1_rows = store.get_recent_canonical_turns("conv-1", limit=10)
    conv2_rows = store.get_recent_canonical_turns("conv-2", limit=10)
    assert len(conv1_rows) == 1
    assert len(conv2_rows) == 1
    assert conv1_rows[0].conversation_id == "conv-1"
    assert conv2_rows[0].conversation_id == "conv-2"


def test_get_recent_includes_untagged_rows(store: SQLiteStore) -> None:
    """Spec §1.2: tagged_at filter is intentionally NOT applied. Fresh
    peer-channel rows whose tagger has not caught up MUST surface."""
    _seed_turn(store, "conv-1", 0, tagged_at="2025-01-01T00:00:00Z")
    _seed_turn(store, "conv-1", 1, tagged_at=None)  # untagged
    rows = store.get_recent_canonical_turns("conv-1", limit=10)
    assert len(rows) == 2
    untagged = [r for r in rows if not r.tagged_at]
    assert len(untagged) == 1


def test_get_recent_limit_zero_returns_empty(store: SQLiteStore) -> None:
    _seed_turn(store, "conv-1", 0)
    assert store.get_recent_canonical_turns("conv-1", limit=0) == []


def test_get_recent_negative_limit_returns_empty(store: SQLiteStore) -> None:
    _seed_turn(store, "conv-1", 0)
    assert store.get_recent_canonical_turns("conv-1", limit=-5) == []


def test_conversation_aliases_indexes_present(store: SQLiteStore) -> None:
    """Both legs of the Tier 1 OR clause must be indexed.

    Asserts the PK on ``alias_id`` (implicit ``sqlite_autoindex_...``)
    and the explicit ``idx_conversation_aliases_target_id``.
    """
    conn = store._get_conn()
    indexes = [
        row[1] for row in conn.execute(
            "PRAGMA index_list('conversation_aliases')",
        ).fetchall()
    ]
    # SQLite gives the PK index a sqlite_autoindex_* name.
    pk_indexes = [n for n in indexes if str(n).startswith("sqlite_autoindex_conversation_aliases")]
    assert pk_indexes, f"expected PK auto-index on conversation_aliases, got {indexes}"
    assert "idx_conversation_aliases_target_id" in indexes
