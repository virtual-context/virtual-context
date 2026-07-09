"""Focused SQLiteStore schema/migration tests."""

from __future__ import annotations

from virtual_context.storage.sqlite import SQLiteStore


def test_fact_embeddings_schema_migration_is_idempotent_and_fk_on(tmp_path):
    db = str(tmp_path / "idem.db")
    # First construct creates the schema.
    store = SQLiteStore(db_path=db)
    conn = store._get_conn()
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_embeddings'"
    ).fetchone() is not None
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='idx_fact_embeddings_conv_model'"
    ).fetchone() is not None
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    # Re-opening the same DB re-runs the CREATE TABLE IF NOT EXISTS
    # migration idempotently without error, and foreign keys stay on.
    store2 = SQLiteStore(db_path=db)
    conn2 = store2._get_conn()
    assert conn2.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_embeddings'"
    ).fetchone() is not None
    fks = conn2.execute("PRAGMA foreign_key_list(fact_embeddings)").fetchall()
    assert any(fk["table"] == "facts" and fk["on_delete"].upper() == "CASCADE"
               for fk in fks)
    assert conn2.execute("PRAGMA foreign_keys").fetchone()[0] == 1
