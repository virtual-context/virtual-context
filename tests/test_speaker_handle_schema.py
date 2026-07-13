"""speaker_handles schema: fresh DDL, additive migration, startup assertions.

A half-migrated handle relation must be a startup failure, not a database
that silently cannot keep handles stable. Both contract unique keys are
verified as enforced constraints, not just as declared DDL.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import (
    SPEAKER_HANDLE_COLUMNS,
    SQLiteStore,
)


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "schema.db"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _insert(conn, tenant, audience, actor, handle):
    conn.execute(
        """INSERT INTO speaker_handles
               (tenant_id, audience_conversation_id, actor_id, handle,
                normalized_base, first_seen_sort_key, created_at,
                lifecycle_epoch)
           VALUES (?, ?, ?, ?, '', 0, ?, 1)""",
        (tenant, audience, actor, handle, _now()),
    )


def test_fresh_schema_has_all_columns(store):
    conn = store._get_conn()
    columns = {
        row["name"] for row in
        conn.execute("PRAGMA table_info(speaker_handles)").fetchall()
    }
    assert set(SPEAKER_HANDLE_COLUMNS) <= columns


def test_actor_unique_key_is_enforced(store):
    conn = store._get_conn()
    _insert(conn, "t1", "guild", "actor:a", "alex")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "t1", "guild", "actor:a", "other")
    # The same actor in a DIFFERENT audience is a separate namespace.
    _insert(conn, "t1", "dm", "actor:a", "alex")
    conn.commit()


def test_handle_unique_key_is_enforced(store):
    conn = store._get_conn()
    _insert(conn, "t1", "guild", "actor:a", "alex")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "t1", "guild", "actor:b", "alex")
    # The same handle in a DIFFERENT audience is a separate namespace.
    _insert(conn, "t1", "dm", "actor:b", "alex")
    conn.commit()


def test_migration_is_additive_on_an_existing_database(tmp_path):
    # A database from before this relation existed gains it on reopen
    # without disturbing other state.
    path = str(tmp_path / "migrate.db")
    first = SQLiteStore(db_path=path)
    conn = first.conn if hasattr(first, "conn") else first._get_conn()
    now = _now()
    conn.execute(
        """INSERT INTO conversations
               (conversation_id, tenant_id, lifecycle_epoch, phase,
                created_at, updated_at)
           VALUES ('guild', 't1', 1, 'active', ?, ?)""",
        (now, now),
    )
    conn.execute("DROP INDEX idx_speaker_handles_handle_unique")
    conn.execute("DROP TABLE speaker_handles")
    conn.commit()

    reopened = SQLiteStore(db_path=path)
    rconn = reopened._get_conn()
    assert rconn.execute(
        "SELECT COUNT(*) FROM speaker_handles"
    ).fetchone()[0] == 0
    row = rconn.execute(
        "SELECT tenant_id FROM conversations WHERE conversation_id = 'guild'"
    ).fetchone()
    assert row[0] == "t1"


def test_missing_table_fails_startup_when_bootstrap_is_swallowed(
    tmp_path, monkeypatch,
):
    # Simulate the real failure mode: the CREATE ran inside a broad catch
    # and silently did nothing. The assertion runs outside that catch and
    # must refuse to start.
    path = str(tmp_path / "partial.db")
    first = SQLiteStore(db_path=path)
    conn = first._get_conn()
    conn.execute("DROP INDEX idx_speaker_handles_handle_unique")
    conn.execute("DROP TABLE speaker_handles")
    conn.commit()

    monkeypatch.setattr(
        SQLiteStore, "_ensure_speaker_handle_schema",
        lambda self, conn: None,
    )
    with pytest.raises(RuntimeError, match="speaker_handles is missing"):
        SQLiteStore(db_path=path)


def test_missing_column_fails_startup(tmp_path):
    # A table that exists without one of the manifest columns is the
    # half-migrated state: CREATE TABLE IF NOT EXISTS cannot repair it, so
    # startup must fail loudly.
    path = str(tmp_path / "halfcol.db")
    first = SQLiteStore(db_path=path)
    conn = first._get_conn()
    conn.execute("ALTER TABLE speaker_handles DROP COLUMN created_at")
    conn.commit()
    with pytest.raises(RuntimeError, match="created_at"):
        SQLiteStore(db_path=path)


def test_missing_unique_key_fails_the_assertion(store):
    conn = store._get_conn()
    conn.execute("DROP INDEX idx_speaker_handles_handle_unique")
    conn.commit()
    with pytest.raises(RuntimeError, match="unique key"):
        store._assert_speaker_handle_schema(conn)
