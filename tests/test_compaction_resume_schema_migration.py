"""Regression: migration must add the five compaction-scoping columns
and stay idempotent under concurrent 8-worker startup (same invariant
as test_sqlite_schema_add_column_race.py).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from virtual_context.storage.sqlite import SQLiteStore


TARGETS = [
    ("segments", "operation_id"),
    ("facts", "operation_id"),
    ("tag_summaries", "operation_id"),
    ("tag_summary_embeddings", "operation_id"),
    ("canonical_turns", "compaction_operation_id"),
]


def _columns_of(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def test_migration_adds_all_five_operation_id_columns(tmp_path: Path):
    db = tmp_path / "migrate.db"
    SQLiteStore(db)  # construction triggers schema
    conn = sqlite3.connect(str(db))
    for table, col in TARGETS:
        cols = _columns_of(conn, table)
        assert col in cols, f"{table}.{col} missing after migration; got {cols}"


def test_migration_is_idempotent_across_repeat_opens(tmp_path: Path):
    db = tmp_path / "migrate.db"
    for _ in range(5):
        SQLiteStore(db)  # re-open should not raise
    conn = sqlite3.connect(str(db))
    for table, col in TARGETS:
        cols = _columns_of(conn, table)
        assert col in cols


def test_migration_backfills_existing_rows_with_zero_uuid(tmp_path: Path):
    """Spec line 61-63 + rollout line 397-401: existing rows backfill to
    the zero-UUID sentinel ``00000000-0000-0000-0000-000000000000``.
    Cleanup predicates scope on ``operation_id = :target`` so zero-UUID
    rows are invisible to cleanup (never match any real operation's
    UUID). The backfill provides defense-in-depth if a later code
    change accidentally treats ``operation_id IS NOT NULL`` as a
    stand-in for "operation-scoped" — the zero-UUID is still non-NULL,
    so behaviour stays predictable.

    Verify by seeding content before migration, running migration,
    asserting pre-existing rows carry the sentinel.
    """
    db = tmp_path / "backfill.db"
    ZERO_UUID = "00000000-0000-0000-0000-000000000000"

    # Create a store (runs migration), seed a segment directly (legacy
    # path, no operation_id passed) so a row exists with NULL
    # operation_id — then re-run migration to confirm the backfill sets
    # it to the sentinel.
    store = SQLiteStore(db)
    conv = "c-backfill"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)

    from virtual_context.types import StoredSegment
    _t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    _t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    store.store_segment(StoredSegment(
        ref="legacy-ref", conversation_id=conv, primary_tag="t", tags=["t"],
        summary="s", summary_tokens=1, full_text="f", full_tokens=2,
        messages=[], compaction_model="passthrough",
        compression_ratio=0.5, start_timestamp=_t0, end_timestamp=_t1,
    ))
    # Null out the operation_id to simulate a pre-migration row.
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "UPDATE segments SET operation_id = NULL WHERE ref = 'legacy-ref'"
        )

    # Re-open to trigger migration's backfill-if-null path.
    SQLiteStore(db)

    with sqlite3.connect(str(db)) as conn:
        op_id = conn.execute(
            "SELECT operation_id FROM segments WHERE ref = 'legacy-ref'"
        ).fetchone()[0]
    assert op_id == ZERO_UUID, (
        f"Pre-migration row's operation_id should be backfilled to "
        f"the zero-UUID sentinel; got {op_id!r}"
    )
