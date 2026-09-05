"""Large canonical indexes are explicit concurrent operations, never startup."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.test_storage_domain_contracts import store as store
from virtual_context.storage.postgres_maintenance import migrate_bounded_read_indexes


def test_group_read_index_bootstrap_and_explicit_migration(store):
    name = 'idx_canonical_turn_group_read'
    if store._relational_dialect == 'sqlite':
        assert store._get_conn().execute('SELECT 1 FROM sqlite_schema WHERE type=? AND name=?', ('index', name)).fetchone()
        return
    with store.pool.connection() as conn:
        assert conn.execute('SELECT to_regclass(%s) AS name', ('public.' + name,)).fetchone()['name'] is None
    assert store.migrate_bounded_read_indexes() == {'dry_run': True, 'indexes': {name: 'needs_build'}}
    with store.pool.connection() as conn:
        assert conn.execute('SELECT to_regclass(%s) AS name', ('public.' + name,)).fetchone()['name'] is None
    assert store.migrate_bounded_read_indexes(dry_run=False) == {'dry_run': False, 'indexes': {name: 'built'}}
    assert store.migrate_bounded_read_indexes() == {'dry_run': True, 'indexes': {name: 'ok'}}


class Connection:
    autocommit = True
    info = SimpleNamespace(transaction_status=0)
    def __init__(self):
        self.statements = []
        self.created = False
    def execute(self, sql, params=()):
        self.statements.append((sql, params))
        row = None
        if sql.startswith('CREATE INDEX'):
            self.created = True
        elif 'FROM pg_index' in sql and self.created:
            row = dict(indisvalid=True, indisready=True, first_column='conversation_id',
                       second_column='turn_group_number', third_column='sort_key',
                       indnkeyatts=3, unconditional=True, table_name='canonical_turns')
        elif 'pg_try_advisory_lock' in sql:
            row = {'acquired': True}
        elif sql == 'SHOW lock_timeout':
            row = {'lock_timeout': '0'}
        return SimpleNamespace(fetchone=lambda: row)


def test_concurrent_migration_uses_idle_connection_and_restores_session_settings():
    conn = Connection()
    assert migrate_bounded_read_indexes(conn)['indexes']['idx_canonical_turn_group_read'] == 'needs_build'
    assert all(sql.lstrip().startswith('SELECT') for sql, _ in conn.statements)
    conn.statements.clear()
    assert migrate_bounded_read_indexes(conn, dry_run=False)['indexes']['idx_canonical_turn_group_read'] == 'built'
    assert any(sql.startswith('CREATE INDEX CONCURRENTLY ') for sql, _ in conn.statements)
    assert not any(sql.startswith(('BEGIN', 'SET LOCAL')) for sql, _ in conn.statements)
    assert conn.statements[-2] == ("SELECT set_config('lock_timeout', %s, FALSE)", ('0',))
    assert 'pg_advisory_unlock' in conn.statements[-1][0]
    busy = Connection()
    busy.autocommit = False
    with pytest.raises(RuntimeError, match='idle autocommit'):
        migrate_bounded_read_indexes(busy, dry_run=False)
    assert not any(sql.startswith('CREATE') for sql, _ in busy.statements)


@pytest.mark.parametrize('apply', [False, True])
def test_read_index_cli_is_explicit_and_closes_store(apply, capsys):
    import virtual_context.cli.main as cli
    args = SimpleNamespace(config=None, postgres_dsn='postgresql://test.invalid/test', apply=apply)
    with patch('virtual_context.storage.postgres.PostgresStore') as factory:
        factory.return_value.migrate_bounded_read_indexes.return_value = {'dry_run': not apply}
        cli.cmd_migrate_read_indexes(args)
        factory.assert_called_once_with(args.postgres_dsn)
        factory.return_value.migrate_bounded_read_indexes.assert_called_once_with(dry_run=not apply)
        factory.return_value.close.assert_called_once_with()
    assert 'dry_run' in capsys.readouterr().out
