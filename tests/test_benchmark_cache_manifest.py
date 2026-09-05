"""Benchmark results must identify the memory pipeline actually executed."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.longmemeval.cache_manifest import build_manifest, prepare_cache, write_manifest
from benchmarks.longmemeval.dataset import LongMemEvalQuestion


@pytest.fixture
def question():
    return LongMemEvalQuestion('q1', 'single-session-user', 'Where?', 'Boston',
                              '2026-01-01', [[{'role': 'user', 'content': 'Boston'}]],
                              ['2026-01-01'], ['s1'])


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / 'source'
    (root / 'virtual_context').mkdir(parents=True)
    (root / 'virtual_context' / 'engine.py').write_text('version = 1\n')
    return root


def test_cache_fingerprint_tracks_data_code_and_models_not_credentials_or_location(question, source_root):
    cfg = {'storage_root': '/old', 'storage': {'backend': 'sqlite', 'sqlite': {'path': '/old/db'}},
           'summarization': {'model': 'model-a'},
           'providers': {'p': {'api_key': 'secret-a', 'model': 'model-a'}}}
    original = build_manifest(question, cfg, source_root)
    changed = deepcopy(cfg)
    changed['providers']['p']['api_key'] = 'secret-b'
    changed['storage_root'] = '/new'
    changed['storage']['sqlite']['path'] = '/new/db'
    assert build_manifest(question, changed, source_root) == original
    assert 'secret-a' not in str(original)
    changed['summarization']['model'] = 'model-b'
    assert build_manifest(question, changed, source_root)['fingerprint'] != original['fingerprint']
    assert build_manifest(replace(question, haystack_dates=['2025-01-01']), cfg, source_root)['fingerprint'] != original['fingerprint']
    (source_root / 'virtual_context' / 'engine.py').write_text('version = 2\n')
    assert build_manifest(question, cfg, source_root)['fingerprint'] != original['fingerprint']


def test_only_completed_matching_cache_is_reused(question, source_root, tmp_path):
    manifest = build_manifest(question, {}, source_root)
    directory, hit = prepare_cache(tmp_path / 'q', manifest)
    assert not hit
    with pytest.raises(ValueError, match='incomplete'):
        prepare_cache(tmp_path / 'q', manifest)
    write_manifest(directory, manifest, complete=True)
    assert prepare_cache(tmp_path / 'q', manifest) == (directory, True)
    assert prepare_cache(tmp_path / 'q', manifest, recompact=True) == (directory, False)
    with pytest.raises(ValueError, match='incomplete'):
        prepare_cache(tmp_path / 'q', manifest)
    assert prepare_cache(tmp_path / 'q', manifest, fresh=True) == (directory, False)


def test_legacy_cache_is_preserved_but_not_reused(question, source_root, tmp_path):
    directory = tmp_path / 'q'
    directory.mkdir()
    (directory / 'store.db').write_bytes(b'legacy cache')
    selected, hit = prepare_cache(directory, build_manifest(question, {}, source_root))
    assert selected != directory
    assert not hit
    assert (directory / 'store.db').read_bytes() == b'legacy cache'


def test_ingest_runner_changes_model_without_reusing_old_memory(question, source_root, tmp_path, monkeypatch):
    from benchmarks.longmemeval import vc_runner
    paths = []
    ingests = []

    class Engine:
        def __init__(self, config):
            self.config = config
            paths.append(config.storage.sqlite_path)
            self._turn_tag_index = SimpleNamespace(entries=[])
            self._engine_state = SimpleNamespace(compacted_prefix_messages=999)
            self._supersession_checker = None
        def ingest_history(self, messages, **kwargs):
            ingests.append(self.config.summarization.model)
            return 1
        def compact_manual(self, messages):
            return None
        def _save_state(self, messages):
            Path(self.config.storage.sqlite_path).touch()
            return True
        def get_telemetry(self):
            return SimpleNamespace(total=lambda: SimpleNamespace(call_count=0))
        def close(self):
            pass

    monkeypatch.setattr(vc_runner, 'VirtualContextEngine', Engine)
    monkeypatch.setattr(vc_runner, 'build_manifest', lambda q, c: build_manifest(q, c, source_root))
    first = vc_runner.run_vc_ingest_only(question, cache_dir=tmp_path, summarizer_model='model-a')
    repeated = vc_runner.run_vc_ingest_only(question, cache_dir=tmp_path, summarizer_model='model-a')
    changed = vc_runner.run_vc_ingest_only(question, cache_dir=tmp_path, summarizer_model='model-b')
    assert [first['cached'], repeated['cached'], changed['cached']] == [False, True, False]
    assert ingests == ['model-a', 'model-b']
    assert paths[0] == paths[1] != paths[2]


def _install_checkpoint_engine(monkeypatch, source_root, *, checkpoint=True):
    from benchmarks.longmemeval import vc_runner
    state = {'checkpoint': checkpoint, 'closed': 0, 'supersession_calls': 0}

    class Engine:
        def __init__(self, config):
            self.config = config
            self._turn_tag_index = SimpleNamespace(entries=[])
            self._engine_state = SimpleNamespace(compacted_prefix_messages=0)
            self._supersession_checker = SimpleNamespace(check_and_supersede=self.supersede)
            self._store = SimpleNamespace(query_facts=lambda **kwargs: [])

        def supersede(self, facts):
            state['supersession_calls'] += 1
            return 0

        def ingest_history(self, messages, **kwargs):
            return 1

        def compact_manual(self, messages):
            return None

        def _save_state(self, messages):
            return state['checkpoint']

        def get_telemetry(self):
            return SimpleNamespace(total=lambda: SimpleNamespace(call_count=0))

        def close(self):
            state['closed'] += 1

    monkeypatch.setattr(vc_runner, 'VirtualContextEngine', Engine)
    monkeypatch.setattr(vc_runner, 'build_manifest', lambda q, c: build_manifest(q, c, source_root))
    return vc_runner, state


@pytest.mark.parametrize('runner_name', ['run_vc', 'run_vc_ingest_only'])
def test_failed_checkpoint_never_publishes_complete_cache(question, source_root, tmp_path, monkeypatch, runner_name):
    import json
    runner, state = _install_checkpoint_engine(monkeypatch, source_root, checkpoint=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-no-network')
    kwargs = {'budget': None, 'api_key': 'test-no-network'} if runner_name == 'run_vc' else {}
    with pytest.raises(RuntimeError, match='checkpoint failed'):
        getattr(runner, runner_name)(question, cache_dir=tmp_path, **kwargs)
    manifests = list(tmp_path.rglob('pipeline_manifest.json'))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())['complete'] is False
    assert state['closed'] == 1
    with pytest.raises(ValueError, match='incomplete'):
        getattr(runner, runner_name)(question, cache_dir=tmp_path, **kwargs)


def test_completed_supersession_is_reused_without_mutating_cached_memory(question, source_root, tmp_path, monkeypatch):
    runner, state = _install_checkpoint_engine(monkeypatch, source_root)
    initial = runner.run_vc_ingest_only(question, cache_dir=tmp_path, supersession=True)
    repeated = runner.run_vc_ingest_only(question, cache_dir=tmp_path, supersession=True)
    assert state['supersession_calls'] == 1
    assert initial['cached'] is False and repeated['cached'] is True
    assert initial['cache_fingerprint'] == repeated['cache_fingerprint']


def test_failed_recompact_leaves_cache_incomplete(question, source_root, tmp_path, monkeypatch):
    import json
    runner, state = _install_checkpoint_engine(monkeypatch, source_root)
    first = runner.run_vc_ingest_only(question, cache_dir=tmp_path)
    monkeypatch.setattr(runner, '_clear_compaction_state', lambda *args: None)
    state['checkpoint'] = False
    with pytest.raises(RuntimeError, match='checkpoint failed'):
        runner.run_vc_ingest_only(question, cache_dir=tmp_path, recompact=True)
    manifest = tmp_path / question.question_id / first['cache_fingerprint'] / 'pipeline_manifest.json'
    assert json.loads(manifest.read_text())['complete'] is False
    with pytest.raises(ValueError, match='incomplete'):
        runner.run_vc_ingest_only(question, cache_dir=tmp_path)
