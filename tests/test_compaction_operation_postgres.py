"""Tests for the compaction_operation table schema on the Postgres backend.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors the
SQLite schema test in ``test_compaction_operation.py`` so both backends stay
in lockstep on the compaction operation columns and the partial unique index
that enforces at-most-one active (queued/running) operation per
(conversation_id, lifecycle_epoch).
"""

import os

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def test_compaction_operation_table_exists_pg():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    with store._get_conn() as conn:
        rows = conn.execute("""
            SELECT column_name FROM information_schema.columns
             WHERE table_name = 'compaction_operation'
        """).fetchall()
    cols = {row["column_name"] for row in rows}
    expected = {
        "operation_id", "conversation_id", "lifecycle_epoch",
        "phase_index", "phase_count", "phase_name", "status",
        "started_at", "completed_at", "owner_worker_id",
        "heartbeat_ts", "error_message",
    }
    missing = expected - cols
    assert not missing, f"Missing: {missing}"


def test_compaction_operation_partial_unique_index_predicate_pg():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    with store._get_conn() as conn:
        rows = conn.execute("""
            SELECT indexname, indexdef FROM pg_indexes
             WHERE tablename = 'compaction_operation'
        """).fetchall()
    by_name = {row["indexname"]: row["indexdef"] for row in rows}
    assert "idx_compaction_operation_active" in by_name, f"Got: {set(by_name)}"
    indexdef = by_name["idx_compaction_operation_active"]
    assert "UNIQUE" in indexdef.upper()
    # Status predicate includes both 'queued' and 'running'
    assert "queued" in indexdef.lower() and "running" in indexdef.lower()
