"""Backend tests for ``has_any_alias`` + ``get_recent_canonical_turns`` on Postgres.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set. Mirrors the
always-on SQLite suite at ``tests/test_sqlite_mirror_store_methods.py``
to keep both backends in lockstep on the Tier 1 / Tier 3 contract.
"""

from __future__ import annotations

import os
import uuid

import pytest
from tests.pg_helpers import pg_dsn, pg_test_conn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set")


@pytest.fixture
def store():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    s = PostgresStore(PG_URL)
    yield s
    s.close()


def _new_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# has_any_alias
# ---------------------------------------------------------------------------


def test_has_any_alias_returns_false_when_no_rows(store) -> None:
    assert store.has_any_alias(_new_id()) is False


def test_has_any_alias_returns_true_for_source(store) -> None:
    src, tgt = _new_id(), _new_id()
    store.save_conversation_alias(src, tgt)
    try:
        assert store.has_any_alias(src) is True
    finally:
        store.delete_conversation_alias(src)


def test_has_any_alias_returns_true_for_target(store) -> None:
    src, tgt = _new_id(), _new_id()
    store.save_conversation_alias(src, tgt)
    try:
        assert store.has_any_alias(tgt) is True
    finally:
        store.delete_conversation_alias(src)


def test_has_any_alias_empty_conv_id_returns_false(store) -> None:
    assert store.has_any_alias("") is False


# ---------------------------------------------------------------------------
# get_recent_canonical_turns
# ---------------------------------------------------------------------------


def _seed_turn(store, conversation_id: str, turn_number: int, **kwargs) -> None:
    store.save_canonical_turn(
        conversation_id,
        turn_number,
        kwargs.get("user_content", "u"),
        kwargs.get("assistant_content", "a"),
        primary_tag="topic",
        tags=["topic"],
        tagged_at=kwargs.get("tagged_at"),
    )


def test_get_recent_returns_empty_for_unknown_conv(store) -> None:
    rows = store.get_recent_canonical_turns(_new_id(), limit=5)
    assert rows == []


def test_get_recent_limit_honored(store) -> None:
    conv = _new_id()
    try:
        for i in range(5):
            _seed_turn(store, conv, i)
        rows = store.get_recent_canonical_turns(conv, limit=2)
        assert len(rows) == 2
    finally:
        with pg_test_conn() as conn:
            conn.execute("DELETE FROM canonical_turns WHERE conversation_id = %s", (conv,))


def test_get_recent_ordered_desc_by_sort_key(store) -> None:
    conv = _new_id()
    try:
        for i in range(3):
            _seed_turn(store, conv, i)
        rows = store.get_recent_canonical_turns(conv, limit=10)
        sort_keys = [r.sort_key for r in rows]
        assert sort_keys == sorted(sort_keys, reverse=True)
    finally:
        with pg_test_conn() as conn:
            conn.execute("DELETE FROM canonical_turns WHERE conversation_id = %s", (conv,))


def test_get_recent_includes_untagged_rows(store) -> None:
    conv = _new_id()
    try:
        _seed_turn(store, conv, 0, tagged_at="2025-01-01T00:00:00Z")
        _seed_turn(store, conv, 1, tagged_at=None)
        rows = store.get_recent_canonical_turns(conv, limit=10)
        assert len(rows) == 2
        untagged = [r for r in rows if not r.tagged_at]
        assert len(untagged) == 1
    finally:
        with pg_test_conn() as conn:
            conn.execute("DELETE FROM canonical_turns WHERE conversation_id = %s", (conv,))


# ---------------------------------------------------------------------------
# Catalog index coverage
# ---------------------------------------------------------------------------


def test_conversation_aliases_indexes_present(store) -> None:
    """Both legs of the Tier 1 OR predicate must be indexed.

    PK on ``alias_id`` -> ``conversation_aliases_pkey`` (Postgres
    convention). ``idx_conversation_aliases_target_id`` on ``target_id``.
    """
    with pg_test_conn() as conn:
        rows = conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s",
            ("conversation_aliases",),
        ).fetchall()
    names = {row["indexname"] for row in rows}
    assert "conversation_aliases_pkey" in names
    assert "idx_conversation_aliases_target_id" in names
