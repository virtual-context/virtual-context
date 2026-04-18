"""Regression test: race-safe ALTER TABLE ADD COLUMN in schema migration.

Multiple worker processes opening the same SQLite file during startup
all call ``_ensure_canonical_turn_schema``. The naive pattern reads
``PRAGMA table_info``, sees the column missing, then ``ALTER TABLE ADD
COLUMN``. Two concurrent workers both pass the TOCTOU check and both
issue ALTER; the second raises
``sqlite3.OperationalError: duplicate column name`` at startup even
though the end state is correct.

Fix: swallow the duplicate-column error narrowly via
``SQLiteStore._add_column_if_missing`` so the migration is truly
idempotent under concurrency. Other ``OperationalError`` types still
propagate.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from virtual_context.storage.sqlite import SQLiteStore


def test_add_column_if_missing_swallows_duplicate_column_only(tmp_path: Path):
    """When the column is already present, the helper must succeed
    silently. Any other ``OperationalError`` (e.g. bad definition) must
    propagate so real schema bugs remain visible.
    """
    db = tmp_path / "vc.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT)")

    # Duplicate add → silently succeeds.
    SQLiteStore._add_column_if_missing(conn, "t", "a", "TEXT")
    rows = conn.execute("PRAGMA table_info(t)").fetchall()
    cols = {r[1] for r in rows}
    assert cols == {"id", "a"}, (
        f"Duplicate ADD COLUMN must leave schema unchanged; got {cols}"
    )

    # Bad definition → propagates.
    with pytest.raises(sqlite3.OperationalError):
        SQLiteStore._add_column_if_missing(
            conn, "t", "b", "NOT A REAL TYPE WITH BAD SYNTAX )))",
        )


def test_ensure_canonical_turn_schema_is_idempotent_under_repeated_calls(tmp_path: Path):
    """The full ``_ensure_canonical_turn_schema`` flow must run multiple
    times on the same DB without raising. Simulates an 8-worker startup
    where each worker opens the same SQLite file and runs migrations.
    """
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    # Grab the private method via the store's connection. Running it
    # repeatedly simulates concurrent workers re-entering the migration.
    conn = store._get_conn()
    for _ in range(5):
        store._ensure_canonical_turn_schema(conn)

    rows = conn.execute("PRAGMA table_info(canonical_turns)").fetchall()
    cols = {r[1] for r in rows}
    for required in ("covered_ingestible_entries", "tagged_at", "turn_group_number"):
        assert required in cols, (
            f"Column {required!r} missing after 5 idempotent migration "
            f"runs: {cols}"
        )


def test_race_simulated_duplicate_add_does_not_raise(tmp_path: Path, monkeypatch):
    """Explicitly simulate the race: two callers each check pragma, both
    see the column missing, both call ``_add_column_if_missing``. The
    second must not raise.
    """
    db = tmp_path / "vc.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE canonical_turns (id INTEGER PRIMARY KEY)")

    # Worker A: column genuinely missing, adds it.
    SQLiteStore._add_column_if_missing(
        conn, "canonical_turns", "covered_ingestible_entries",
        "INTEGER NOT NULL DEFAULT 1",
    )
    # Worker B: stale pragma read, tries to add again. Must not raise.
    SQLiteStore._add_column_if_missing(
        conn, "canonical_turns", "covered_ingestible_entries",
        "INTEGER NOT NULL DEFAULT 1",
    )

    rows = conn.execute("PRAGMA table_info(canonical_turns)").fetchall()
    cols = {r[1] for r in rows}
    assert "covered_ingestible_entries" in cols
