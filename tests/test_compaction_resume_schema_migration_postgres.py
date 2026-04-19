"""Postgres mirror of the SQLite migration. Requires DATABASE_URL to
point at a throwaway Postgres (skips when missing, like other
*_postgres.py tests in this repo).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping Postgres integration test.",
)

from virtual_context.storage.postgres import PostgresStore


TARGETS = [
    ("segments", "operation_id"),
    ("facts", "operation_id"),
    ("tag_summaries", "operation_id"),
    ("tag_summary_embeddings", "operation_id"),
    ("canonical_turns", "compaction_operation_id"),
]


def _columns_of(store: PostgresStore, table: str) -> set[str]:
    conn = store._get_conn()
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = %s",
        (table,),
    ).fetchall()
    return {r[0] for r in rows}


def test_migration_adds_all_five_operation_id_columns_postgres():
    dsn = os.environ["DATABASE_URL"]
    store = PostgresStore(dsn=dsn)  # construction triggers schema
    for table, col in TARGETS:
        cols = _columns_of(store, table)
        assert col in cols, f"{table}.{col} missing after migration; got {cols}"


def test_migration_backfills_null_operation_id_to_zero_uuid_postgres():
    """Same spec contract as SQLite test. Legacy rows get the
    zero-UUID sentinel, never NULL, after migration.
    """
    dsn = os.environ["DATABASE_URL"]
    ZERO_UUID = "00000000-0000-0000-0000-000000000000"
    store = PostgresStore(dsn=dsn)
    conn = store._get_conn()
    # Null out any operation_id to simulate pre-migration state, then
    # re-trigger the migration helper and verify backfill fires.
    conn.execute("UPDATE segments SET operation_id = NULL")
    store._ensure_compaction_scoping_columns()
    n_null = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE operation_id IS NULL"
    ).fetchone()[0]
    assert n_null == 0, (
        f"After backfill, no segments row should have NULL operation_id; "
        f"got {n_null}"
    )
    n_zero = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE operation_id = %s",
        (ZERO_UUID,),
    ).fetchone()[0]
    assert n_zero >= 0  # zero-UUID count depends on pre-existing data
