"""engine_state schema vs. save path on Postgres (BUG-039).

Skipped unless a Postgres DSN is configured. The bundled schema created
``engine_state`` with five columns while ``save_engine_state`` INSERTs
seven (``flushed_prefix_messages``, ``last_request_time``). On any
database bootstrapped from the bundled schema alone, EVERY engine-state
save failed with UndefinedColumn — swallowed upstream as a warning, so
session-restore state silently never persisted.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


def _snapshot(conv: str):
    from virtual_context.types import EngineStateSnapshot, TurnTagEntry
    return EngineStateSnapshot(
        conversation_id=conv,
        compacted_prefix_messages=4,
        turn_tag_entries=[
            TurnTagEntry(turn_number=0, message_hash="abc", tags=["t"], primary_tag="t"),
        ],
        turn_count=1,
        flushed_prefix_messages=2,
        last_request_time=1234.5,
        saved_at=datetime.now(timezone.utc),
    )


@pytest.mark.regression("BUG-039")
def test_engine_state_table_has_save_path_columns():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    try:
        conn = pg_test_conn()
        cols = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'engine_state'"
            ).fetchall()
        }
        assert "flushed_prefix_messages" in cols, cols
        assert "last_request_time" in cols, cols
    finally:
        store.close()


@pytest.mark.regression("BUG-039")
def test_legacy_column_names_converge_on_bootstrap():
    """Tables from the oldest vintage carry ``compacted_through`` /
    ``flushed_through`` instead of the names the save path writes. The
    bootstrap must rename them so the INSERT works. (Observed in a
    long-lived deployment: 7 columns, two with legacy names, zero rows —
    every save had failed since creation.)"""
    from virtual_context.storage.postgres import PostgresStore  # deferred
    conn = pg_test_conn()
    conn.execute("DROP TABLE IF EXISTS engine_state")
    conn.execute(
        """CREATE TABLE engine_state (
            conversation_id TEXT PRIMARY KEY,
            compacted_through INTEGER NOT NULL,
            turn_count INTEGER NOT NULL,
            turn_tag_entries TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            flushed_through INTEGER NOT NULL DEFAULT 0,
            last_request_time DOUBLE PRECISION NOT NULL DEFAULT 0
        )"""
    )
    store = PostgresStore(PG_URL)
    conv = f"es-legacy-{uuid.uuid4().hex[:12]}"
    try:
        cols = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'engine_state'"
            ).fetchall()
        }
        assert "compacted_prefix_messages" in cols, cols
        assert "flushed_prefix_messages" in cols, cols
        assert "compacted_through" not in cols, cols
        assert "flushed_through" not in cols, cols
        store.save_engine_state(_snapshot(conv))
        loaded = store.load_engine_state(conv)
        assert loaded is not None
        assert loaded.compacted_prefix_messages == 4
        assert loaded.flushed_prefix_messages == 2
    finally:
        store.close()


@pytest.mark.regression("BUG-039")
def test_save_engine_state_round_trips_on_fresh_schema():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    store = PostgresStore(PG_URL)
    conv = f"es-{uuid.uuid4().hex[:12]}"
    try:
        store.save_engine_state(_snapshot(conv))
        loaded = store.load_engine_state(conv)
        assert loaded is not None, "save must actually persist a row"
        assert loaded.conversation_id == conv
        assert loaded.compacted_prefix_messages == 4
        assert loaded.flushed_prefix_messages == 2
        assert loaded.last_request_time == 1234.5
    finally:
        store.close()
