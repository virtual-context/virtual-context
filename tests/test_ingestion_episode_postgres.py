"""Tests for the ingestion_episode table schema on the Postgres backend.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors the
SQLite schema test in ``test_ingestion_episode.py`` so both backends stay in
lockstep on the ownership/lifecycle episode columns and the partial unique
index that enforces at-most-one running episode per
(conversation_id, lifecycle_epoch).
"""

import os

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def test_ingestion_episode_table_exists_pg():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    with store._get_conn() as conn:
        rows = conn.execute("""
            SELECT column_name FROM information_schema.columns
             WHERE table_name = 'ingestion_episode'
        """).fetchall()
    cols = {row["column_name"] for row in rows}
    expected = {
        "episode_id", "conversation_id", "lifecycle_epoch",
        "raw_payload_entries", "started_at", "completed_at",
        "status", "owner_worker_id", "heartbeat_ts",
    }
    missing = expected - cols
    assert not missing, f"Missing: {missing}"


def test_ingestion_episode_partial_unique_index_has_correct_predicate_pg():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    with store._get_conn() as conn:
        rows = conn.execute("""
            SELECT indexname, indexdef FROM pg_indexes
             WHERE tablename = 'ingestion_episode'
        """).fetchall()
    by_name = {row["indexname"]: row["indexdef"] for row in rows}
    assert "idx_ingestion_episode_active" in by_name, f"Got: {set(by_name)}"
    indexdef = by_name["idx_ingestion_episode_active"]
    assert "UNIQUE" in indexdef.upper()
    assert "status = 'running'" in indexdef.lower() or "status=running" in indexdef.lower() \
        or "(status = 'running'::" in indexdef
