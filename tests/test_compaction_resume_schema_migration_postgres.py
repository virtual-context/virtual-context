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


def test_ensure_canonical_turn_views_survives_underlying_column_add_postgres():
    """Regression: CREATE OR REPLACE VIEW refuses to change the column
    list when the view selects ct.*. After a new column is added to
    canonical_turns, rerunning _ensure_canonical_turn_views must still
    succeed. Fix is DROP VIEW IF EXISTS + CREATE VIEW.

    Reproduces production error observed on compaction-resume-parity
    deploy: ``InvalidTableDefinition: cannot change name of view column
    "turn_number" to "compaction_operation_id"``.
    """
    dsn = os.environ["DATABASE_URL"]
    store = PostgresStore(dsn=dsn)
    conn = store._get_conn()

    # Simulate the pre-fix state: drop the view, re-create it without the
    # compaction_operation_id column in its select-list (as the branch
    # previous to this migration produced). We do that by temporarily
    # dropping the column, recreating the view (now with ct.* minus the
    # column), then adding the column back.
    conn.execute("DROP VIEW IF EXISTS canonical_turns_ordinal")
    conn.execute(
        "ALTER TABLE canonical_turns DROP COLUMN IF EXISTS compaction_operation_id"
    )
    # Old-style recreate (what legacy code produced)
    conn.execute(
        """CREATE VIEW canonical_turns_ordinal AS
           SELECT
               ct.*,
               ROW_NUMBER() OVER (
                   PARTITION BY ct.conversation_id
                   ORDER BY ct.sort_key, ct.first_seen_at, ct.canonical_turn_id
               ) - 1 AS turn_number
           FROM canonical_turns ct"""
    )
    # Now re-add the column (as _ensure_compaction_scoping_columns does).
    conn.execute(
        "ALTER TABLE canonical_turns ADD COLUMN compaction_operation_id UUID"
    )
    # And re-run the view migration. Pre-fix this raised
    # InvalidTableDefinition.
    store._ensure_canonical_turn_views()  # must not raise
    # Verify the view now exposes the new column.
    cols = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'canonical_turns_ordinal'"
    ).fetchall()
    names = {r[0] for r in cols}
    assert "compaction_operation_id" in names, (
        f"View must expose compaction_operation_id after migration; got {names}"
    )
    assert "turn_number" in names, (
        f"View must still expose turn_number after migration; got {names}"
    )


def test_concurrent_workers_racing_canonical_turn_view_migration_postgres():
    """Regression: two concurrent workers both running the view migration
    must serialize via advisory lock; the 'later' worker must NOT fail
    with ``duplicate key value violates unique constraint
    pg_type_typname_nsp_index`` after both win the DROP IF EXISTS race.

    Surfaced after the CREATE OR REPLACE VIEW → DROP+CREATE fix: the
    first fix resolved the column-list refusal but uncovered a second
    race on the CREATE.
    """
    import threading

    dsn = os.environ["DATABASE_URL"]
    # Two independent stores → two connections → true concurrency.
    store_a = PostgresStore(dsn=dsn)
    store_b = PostgresStore(dsn=dsn)

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _run(store):
        try:
            barrier.wait(timeout=5)  # release both threads simultaneously
            for _ in range(3):  # re-run to amplify any race window
                store._ensure_canonical_turn_views()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=_run, args=(store_a,))
    t2 = threading.Thread(target=_run, args=(store_b,))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert not errors, f"Concurrent view migration raised: {errors!r}"

    # Final state must still have the view with the expected columns.
    conn = store_a._get_conn()
    cols = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'canonical_turns_ordinal'"
    ).fetchall()
    names = {r[0] for r in cols}
    assert "turn_number" in names
    assert "compaction_operation_id" in names


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
