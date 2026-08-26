"""Process-wide embedding model cache (staged; red until implemented).

Every EmbeddingProvider instance performs its own lazy SentenceTransformer
load, so a process holding many engines loads the same weights repeatedly.
The shared helper caches one loaded model per (process, model_name):
double-checked locking on first touch, successful loads only, injected and
disabled providers untouched.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import threading
from unittest.mock import patch

import pytest

from virtual_context.core import embedding_provider as _ep

# Self-arming: these tests stage the process-wide cache contract and skip
# until the implementation lands (it is measurement-gated). The skip lifts
# automatically the moment the cache exists, at which point every test
# below must pass.
pytestmark = pytest.mark.skipif(
    not hasattr(_ep, "_PROCESS_MODEL_CACHE"),
    reason="process-wide embedding model cache not implemented (measurement-gated)",
)


class _FakeModel:
    constructions = 0

    def __init__(self, model_name: str):
        type(self).constructions += 1
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        class _A:
            def __init__(self, n): self._n = n
            def tolist(self): return [[0.0] for _ in range(self._n)]
        return _A(len(texts))


@pytest.fixture(autouse=True)
def _fresh_cache():
    from virtual_context.core import embedding_provider as ep
    _FakeModel.constructions = 0
    saved = dict(getattr(ep, "_PROCESS_MODEL_CACHE", {}))
    getattr(ep, "_PROCESS_MODEL_CACHE", {}).clear()
    yield
    cache = getattr(ep, "_PROCESS_MODEL_CACHE", {})
    cache.clear()
    cache.update(saved)


def _provider(model_name="all-MiniLM-L6-v2", **kw):
    from virtual_context.core.embedding_provider import EmbeddingProvider
    return EmbeddingProvider(model_name=model_name, **kw)


def test_same_model_name_loads_once_across_providers():
    with patch(
        "sentence_transformers.SentenceTransformer", _FakeModel, create=True,
    ):
        fn_a = _provider().get_embed_fn()
        fn_b = _provider().get_embed_fn()
    assert fn_a is not None and fn_b is not None
    assert _FakeModel.constructions == 1
    assert fn_a([""]) == fn_b([""])


def test_distinct_model_names_load_separately():
    with patch(
        "sentence_transformers.SentenceTransformer", _FakeModel, create=True,
    ):
        assert _provider("model-x").get_embed_fn() is not None
        assert _provider("model-y").get_embed_fn() is not None
    assert _FakeModel.constructions == 2


def test_injected_and_disabled_providers_never_touch_the_cache():
    from virtual_context.core import embedding_provider as ep

    injected = _provider(embed_fn=lambda texts: [[1.0]] * len(texts))
    assert injected.get_embed_fn()([""]) == [[1.0]]
    assert _provider(disabled=True).get_embed_fn() is None
    assert getattr(ep, "_PROCESS_MODEL_CACHE") == {}


def test_failed_load_is_not_cached_and_retries():
    calls = {"n": 0}

    class _Flaky:
        def __init__(self, model_name):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient download failure")
            self.model_name = model_name

        def encode(self, texts, **kwargs):
            class _A:
                def tolist(self): return [[0.0]]
            return _A()

    with patch(
        "sentence_transformers.SentenceTransformer", _Flaky, create=True,
    ):
        assert _provider().get_embed_fn() is None       # first load fails
        assert _provider().get_embed_fn() is not None   # a fresh provider retries
    assert calls["n"] == 2


def test_concurrent_first_touch_loads_exactly_once():
    import time

    class _SlowModel(_FakeModel):
        constructions = 0

        def __init__(self, model_name):
            time.sleep(0.05)
            super().__init__(model_name)

    results = []
    with patch(
        "sentence_transformers.SentenceTransformer", _SlowModel, create=True,
    ):
        providers = [_provider() for _ in range(6)]
        threads = [
            threading.Thread(target=lambda p=p: results.append(p.get_embed_fn()))
            for p in providers
        ]
        for t in threads: t.start()
        for t in threads: t.join()
    assert all(fn is not None for fn in results)
    assert _SlowModel.constructions == 1
