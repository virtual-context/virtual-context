"""Setup rejects unsupported storage before I/O and handles remote providers."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from virtual_context.config import load_config, validate_config
from virtual_context.engine import VirtualContextEngine
import virtual_context.cli.main as cli


@pytest.mark.parametrize('backend', ['neo4j', 'falkordb'])
def test_graph_engine_backend_is_rejected_before_connecting(backend):
    config = load_config(config_dict={'storage': {'backend': backend}}, validate=False)
    assert any('storage.backend' in error for error in validate_config(config))
    engine = object.__new__(VirtualContextEngine)
    engine.config = config
    with pytest.raises(ValueError, match='conversation-scoped atomic'):
        engine._build_raw_store()


def test_remote_provider_wizard_reads_environment(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    with (
        patch.object(cli, '_prompt_choice', side_effect=['anthropic', RuntimeError('past key check')]),
        patch.object(cli, '_prompt', return_value='https://example.invalid'),
        patch.object(cli, '_prompt_tagging_provider', return_value=('anthropic', 'test-model')),
        patch.object(cli, '_check_provider_reachable', return_value=True),
    ):
        with pytest.raises(RuntimeError, match='past key check'):
            cli._run_instance_wizard()


def test_search_repair_cli_refuses_missing_database(tmp_path):
    args = SimpleNamespace(config=None, sqlite_path=str(tmp_path / 'missing.db'),
                           index=None, apply=False)
    with pytest.raises(ValueError, match='does not exist'):
        cli.cmd_repair_search_indexes(args)
    assert not (tmp_path / 'missing.db').exists()


@pytest.mark.parametrize('backend,model,enabled,expected', [
    ('sqlite', 'all-MiniLM-L6-v2', False, None),
    ('postgres', 'all-MiniLM-L6-v2', True, None),
    ('sqlite', 'all-MiniLM-L6-v2', True, 'storage.backend=postgres'),
    ('postgres', 'another-model', True, 'all-MiniLM-L6-v2'),
    ('postgres', 'all-MiniLM-L6-v2', 'false', 'must be a boolean'),
])
def test_native_vector_configuration_fails_closed(backend, model, enabled, expected):
    config = load_config(config_dict={
        'storage': {'backend': backend},
        'retrieval': {'embedding_model': model, 'vector_search_enabled': enabled},
    }, validate=False)
    errors = [item for item in validate_config(config) if 'vector_search_enabled' in item]
    assert config.retriever.vector_search_enabled == enabled
    if expected:
        assert any(expected in item for item in errors)
    else:
        assert errors == []


@pytest.mark.parametrize('apply', [False, True])
def test_vector_migration_cli_is_explicit_and_closes_store(apply, capsys):
    args = SimpleNamespace(config=None, postgres_dsn='postgresql://test.invalid/test',
                           batch_size=123, apply=apply)
    with patch('virtual_context.storage.postgres.PostgresStore') as factory:
        store = factory.return_value
        store.migrate_semantic_vectors.return_value = {'ready': False, 'dry_run': not apply}
        cli.cmd_migrate_semantic_vectors(args)
        factory.assert_called_once_with(args.postgres_dsn)
        store.migrate_semantic_vectors.assert_called_once_with(
            dry_run=not apply, batch_size=123, model='all-MiniLM-L6-v2')
        store.close.assert_called_once_with()
        assert '"ready": false' in capsys.readouterr().out


def test_vector_migration_cli_rejects_sqlite_before_connecting():
    args = SimpleNamespace(config=None, postgres_dsn=None, batch_size=1000, apply=False)
    with patch('virtual_context.storage.postgres.PostgresStore') as factory:
        with pytest.raises(ValueError, match='storage.backend=postgres'):
            cli.cmd_migrate_semantic_vectors(args)
        factory.assert_not_called()


@pytest.mark.parametrize('batch_size', [0, 10001])
def test_vector_migration_cli_rejects_invalid_batch_before_connecting(batch_size):
    args = SimpleNamespace(config=None, postgres_dsn='postgresql://test.invalid/test',
                           batch_size=batch_size, apply=True)
    with patch('virtual_context.storage.postgres.PostgresStore') as factory:
        with pytest.raises(ValueError, match='between 1 and 10000'):
            cli.cmd_migrate_semantic_vectors(args)
        factory.assert_not_called()
