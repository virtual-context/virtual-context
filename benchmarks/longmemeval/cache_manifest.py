"""Content-addressed memory-pipeline caches with explicit completion state."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dataset import LongMemEvalQuestion

_MANIFEST = 'pipeline_manifest.json'
_SECRET_KEYS = {'api_key', 'access_token', 'refresh_token', 'password', 'authorization'}


def _public_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_config(item)
            for key, item in value.items()
            if key.lower() not in _SECRET_KEYS and not key.lower().endswith('_api_key')
        }
    if isinstance(value, list):
        return [_public_config(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def build_manifest(question: LongMemEvalQuestion, config: dict,
                   source_root: Path | None = None) -> dict:
    """Identify actual source bytes, input data, and non-secret memory settings."""
    root = source_root or Path(__file__).resolve().parents[2]
    paths = sorted((root / 'virtual_context').rglob('*.py'))
    paths += sorted((root / 'virtual_context' / 'data').glob('*.yaml'))
    paths += [root / 'benchmarks' / 'longmemeval' / name
              for name in ('vc_runner.py', 'cache_manifest.py')]
    source_hash = hashlib.sha256()
    for path in paths:
        if path.is_file():
            source_hash.update(str(path.relative_to(root)).encode() + b'\0')
            source_hash.update(path.read_bytes() + b'\0')
    normalized = _public_config(config)
    normalized.pop('storage_root', None)
    # Cache location cannot change the identity of the memory stored there.
    normalized['storage'] = {'backend': normalized.get('storage', {}).get('backend', 'sqlite')}
    manifest = {
        'version': 1,
        'source_sha256': source_hash.hexdigest(),
        'dataset_sha256': _digest(asdict(question)),
        'config': normalized,
    }
    return {**manifest, 'fingerprint': _digest(manifest)}


def prepare_cache(question_dir: Path, manifest: dict, *, fresh: bool = False,
                  recompact: bool = False) -> tuple[Path, bool]:
    """Select matching artifacts; never trust an interrupted or legacy cache."""
    import shutil
    cache_dir = question_dir / manifest['fingerprint']
    if fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    manifest_path = cache_dir / _MANIFEST
    complete = False
    if cache_dir.exists() and any(cache_dir.iterdir()):
        try:
            saved = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError('Benchmark cache has no valid manifest; rerun with --fresh') from exc
        if saved.get('fingerprint') != manifest['fingerprint'] or not saved.get('complete'):
            raise ValueError('Benchmark cache is incomplete or mismatched; rerun with --fresh')
        complete = True
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not complete or recompact:
        write_manifest(cache_dir, manifest, complete=False)
    return cache_dir, complete and not recompact


def write_manifest(cache_dir: Path, manifest: dict, *, complete: bool) -> None:
    """Publish completion only after memory and its persistent checkpoint succeed."""
    temporary = cache_dir / (_MANIFEST + '.tmp')
    temporary.write_text(json.dumps({**manifest, 'complete': complete}, indent=2, sort_keys=True))
    temporary.replace(cache_dir / _MANIFEST)
