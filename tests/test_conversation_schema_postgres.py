"""Tests for the conversations table schema on the Postgres backend.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors the
SQLite schema test in ``test_conversation_schema.py`` so both backends stay in
lockstep on the lifecycle/phase/counters columns.
"""

import os

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def test_conversations_table_has_phase_and_epoch_columns_pg():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    with store._get_conn() as conn:
        rows = conn.execute(
            """
            SELECT column_name FROM information_schema.columns
             WHERE table_name = 'conversations'
            """
        ).fetchall()
    cols = {row["column_name"] for row in rows}
    expected = {
        "conversation_id", "tenant_id", "lifecycle_epoch", "phase",
        "pending_raw_payload_entries", "last_raw_payload_entries",
        "last_ingestible_payload_entries",
        "created_at", "updated_at", "deleted_at",
    }
    missing = expected - cols
    assert not missing, f"Missing columns: {missing}"
