"""Tests for the canonical_turns progress-tracking schema on Postgres.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors the
SQLite schema test in ``test_canonical_turns_schema.py`` so both backends stay
in lockstep on ``covered_ingestible_entries``, ``tagged_at`` and the partial
indexes used by the DB-derived progress tracker.
"""

import os

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def test_canonical_turns_has_covered_and_tagged_columns_pg():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    with store._get_conn() as conn:
        rows = conn.execute("""
            SELECT column_name FROM information_schema.columns
             WHERE table_name = 'canonical_turns'
        """).fetchall()
    cols = {row["column_name"] for row in rows}
    assert "covered_ingestible_entries" in cols
    assert "tagged_at" in cols


def test_canonical_turns_partial_indexes_exist_pg():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    with store._get_conn() as conn:
        rows = conn.execute("""
            SELECT indexname, indexdef FROM pg_indexes
             WHERE tablename = 'canonical_turns'
        """).fetchall()
    names = {row["indexname"] for row in rows}
    assert "idx_canonical_turns_conv_untagged" in names, f"Got: {names}"
    assert "idx_canonical_turns_conv_tagged" in names, f"Got: {names}"
    # Verify they are partial indexes.
    for row in rows:
        if row["indexname"] == "idx_canonical_turns_conv_untagged":
            assert "tagged_at IS NULL" in row["indexdef"]
        if row["indexname"] == "idx_canonical_turns_conv_tagged":
            assert "tagged_at IS NOT NULL" in row["indexdef"]
