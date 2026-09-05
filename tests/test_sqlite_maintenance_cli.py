"""FTS maintenance must not initialize, migrate, or repair a dry-run source."""
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3

import pytest

import virtual_context.cli.main as cli
from virtual_context.storage.maintenance import repair_fts_indexes, sqlite_maintenance_connection


def _seed(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript("""
        CREATE TABLE segments(ref TEXT PRIMARY KEY, summary TEXT NOT NULL);
        CREATE VIRTUAL TABLE segments_fts USING fts5(ref UNINDEXED, summary,
            content='segments', content_rowid='rowid');
        CREATE TRIGGER segments_ai AFTER INSERT ON segments BEGIN
            INSERT INTO segments_fts(rowid,ref,summary) VALUES(new.rowid,new.ref,new.summary);
        END;
        INSERT INTO segments VALUES ('source', 'aardvark');
    """)
    return conn


def _schema(conn):
    return list(conn.execute('SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY name'))


def _logical_rows(conn):
    return list(conn.execute('SELECT rowid,* FROM segments ORDER BY rowid'))


def _fts_matches(conn):
    return list(conn.execute("SELECT rowid FROM segments_fts WHERE segments_fts MATCH 'aardvark'"))


def test_repair_cli_dry_run_preserves_bytes_schema_and_broken_index_then_apply_repairs(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'legacy.db'
    conn = _seed(path)
    # A legacy minimal schema deliberately lacks every newer application table.
    # Constructing SQLiteStore would migrate it before the requested dry run.
    conn.execute("INSERT INTO segments_fts(segments_fts) VALUES('delete-all')")
    assert _fts_matches(conn) == []
    before_rows, before_schema = _logical_rows(conn), _schema(conn)
    conn.close()
    before_bytes = path.read_bytes()
    before_files = {p.name for p in tmp_path.iterdir()}
    args = SimpleNamespace(config=None, sqlite_path=str(path), index=['segments_fts'], apply=False)
    monkeypatch.setattr(cli, 'load_config', lambda _: SimpleNamespace(storage=SimpleNamespace(backend='sqlite', sqlite_path=str(path))))
    def constructor_forbidden(*args, **kwargs):
        raise AssertionError('maintenance constructed SQLiteStore')
    monkeypatch.setattr(cli, 'SQLiteStore', constructor_forbidden)

    cli.cmd_repair_search_indexes(args)
    assert json.loads(capsys.readouterr().out) == {'dry_run': True, 'indexes': {'segments_fts': 'needs_rebuild'}}
    assert path.read_bytes() == before_bytes
    assert {p.name for p in tmp_path.iterdir()} == before_files
    with sqlite3.connect(path) as check:
        assert _schema(check) == before_schema
        assert _logical_rows(check) == before_rows
        assert _fts_matches(check) == []

    args.apply = True
    cli.cmd_repair_search_indexes(args)
    assert json.loads(capsys.readouterr().out) == {'dry_run': False, 'indexes': {'segments_fts': 'rebuilt'}}
    with sqlite3.connect(path) as check:
        assert _schema(check) == before_schema
        assert _logical_rows(check) == before_rows
        assert _fts_matches(check) == [(before_rows[0][0],)]
        assert repair_fts_indexes(check, ['segments_fts']) == {'segments_fts': 'ok'}


def test_private_backup_includes_uncheckpointed_wal_and_removes_private_artifacts(tmp_path):
    path = tmp_path / 'wal.db'
    source = _seed(path)
    try:
        source.execute('PRAGMA journal_mode=WAL')
        source.execute('PRAGMA wal_autocheckpoint=0')
        source.execute("INSERT INTO segments VALUES ('wal-source', 'aardvark from WAL')")
        main_bytes = path.read_bytes()
        wal_path = Path(str(path) + '-wal')
        wal_bytes = wal_path.read_bytes()
        assert wal_bytes
        with sqlite_maintenance_connection(path) as copy:
            copy_path = Path(copy.execute('PRAGMA database_list').fetchone()[2])
            assert copy_path.parent.stat().st_mode & 0o777 == 0o700
            assert len(_logical_rows(copy)) == 2
            assert repair_fts_indexes(copy, ['segments_fts']) == {'segments_fts': 'ok'}
            assert not copy.in_transaction
            # Even an accidental write affects only the temporary copy.
            copy.execute("UPDATE segments SET summary='private copy only'")
        assert not copy_path.parent.exists()
        assert path.read_bytes() == main_bytes
        assert wal_path.read_bytes() == wal_bytes
        assert len(_fts_matches(source)) == 2
        assert source.execute("SELECT count(*) FROM segments WHERE summary='private copy only'").fetchone()[0] == 0
    finally:
        source.close()


@pytest.mark.parametrize('dry_run', [True, False])
def test_maintenance_connection_never_creates_missing_database(tmp_path, dry_run):
    path = tmp_path / 'missing.db'
    with pytest.raises(ValueError, match='does not exist'), sqlite_maintenance_connection(path, dry_run=dry_run):
        pytest.fail('missing database was opened')
    assert not path.exists()


def test_fts_check_refuses_caller_transaction_and_rolls_back_missing_table(tmp_path):
    path = tmp_path / 'check.db'
    conn = _seed(path)
    try:
        conn.execute('BEGIN')
        with pytest.raises(RuntimeError, match='own transaction'):
            repair_fts_indexes(conn, ['segments_fts'])
        assert conn.in_transaction
        conn.rollback()
        with pytest.raises(sqlite3.OperationalError, match='no such table'):
            repair_fts_indexes(conn, ['facts_fts'])
        assert not conn.in_transaction
    finally:
        conn.close()
