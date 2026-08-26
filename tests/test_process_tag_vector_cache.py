"""The tag-vector runtime cache is process-wide, not per instance.

Materializing a conversation's tag vectors from the shared cache costs
CPU that scales with vocabulary (deserialize + normalize + copy), and
the cost was paid per provider instance: every cold engine re-imported
the same vectors its process already held. The runtime cache is now one
process-global store keyed by model name and tag — content-derived
values with no conversation or tenant dimension — so a cold engine in a
warm process starts with the vectors already resident.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import threading
from unittest.mock import MagicMock

from virtual_context.proxy.session_state import SessionStateProvider


def _provider(**kwargs):
    redis = MagicMock()
    return SessionStateProvider(redis_client=redis, **kwargs), redis


def test_default_bound_covers_measured_runtime_vocabularies():
    provider, _ = _provider()
    assert provider._tag_embedding_runtime_max_per_model == 10000


def test_vectors_survive_across_provider_instances():
    provider_a, _ = _provider()
    provider_a.save_tag_embeddings("model-x", {
        "database": [0.1, 0.2], "api": [0.3, 0.4],
    })

    provider_b, redis_b = _provider()
    loaded = provider_b.load_tag_embeddings("model-x", ["database", "api"])
    assert loaded == {"database": [0.1, 0.2], "api": [0.3, 0.4]}
    redis_b.mget.assert_not_called()
    redis_b.get.assert_not_called()


def test_models_stay_isolated():
    provider, _ = _provider()
    provider.save_tag_embeddings("model-x", {"database": [0.1]})
    assert provider._runtime_tag_cache("model-y").get("database") is None


def test_instance_alias_clears_the_shared_store():
    provider_a, _ = _provider()
    provider_a.save_tag_embeddings("model-x", {"database": [0.1]})

    provider_b, _ = _provider()
    provider_b._tag_embedding_runtime_cache.clear()
    assert provider_a._runtime_tag_cache("model-x").get("database") is None


def test_concurrent_inserts_stay_consistent():
    provider, _ = _provider(tag_embedding_runtime_max_per_model=10000)

    def insert(worker: int) -> None:
        for i in range(200):
            provider._remember_runtime_tag_embedding(
                "model-x", f"tag-{worker}-{i}", [float(worker), float(i)],
            )

    threads = [
        threading.Thread(target=insert, args=(w,)) for w in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cache = provider._runtime_tag_cache("model-x")
    assert len(cache) == 6 * 200
    assert cache["tag-3-77"] == [3.0, 77.0]
