"""Embeddings held in memory are float32 arrays shared across reads.

A 384-dimension vector kept as a Python list of floats costs about 32 bytes
per value; a float32 array costs 4. A conversation with thousands of tags
held several such copies per worker, and every snapshot read cloned one.
"""

from __future__ import annotations

import fakeredis
import numpy as np
import pytest

from virtual_context.core.embedding_tag_generator import EmbeddingTagGenerator
from virtual_context.core.retrieval_scoring import compute_embedding_candidates
from virtual_context.proxy import session_state
from virtual_context.proxy.session_state import SessionStateProvider
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import TagGeneratorConfig

CONV = "conv-1"
MODEL = "test-model"


@pytest.fixture(autouse=True)
def _fresh_process_cache():
    session_state._PROCESS_TAG_VECTOR_CACHE.clear()
    yield
    session_state._PROCESS_TAG_VECTOR_CACHE.clear()


def _provider():
    return SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)


def _is_f32(value) -> bool:
    return isinstance(value, np.ndarray) and value.dtype == np.float32


@pytest.mark.regression("PROXY-038")
def test_cached_tag_embeddings_are_shared_float32_arrays():
    provider = _provider()
    provider.save_tag_embeddings(MODEL, {"hcg": [3.0, 4.0]})
    first = provider.load_tag_embeddings(MODEL, ["hcg"])["hcg"]
    second = provider.load_tag_embeddings(MODEL, ["hcg"])["hcg"]
    assert _is_f32(first)
    assert second is first
    assert np.allclose(first, [3.0, 4.0])


@pytest.mark.regression("PROXY-038")
def test_tag_embeddings_read_back_from_redis_are_float32():
    writer, reader = _provider(), None
    writer.save_tag_embeddings(MODEL, {"hcg": [1.0, 2.0]})
    session_state._PROCESS_TAG_VECTOR_CACHE.clear()
    reader = SessionStateProvider(redis_client=writer._redis, store=None)
    assert _is_f32(reader.load_tag_embeddings(MODEL, ["hcg"])["hcg"])


@pytest.mark.regression("PROXY-038")
def test_snapshot_reads_share_float32_vectors_without_copying():
    provider = _provider()
    provider.save_tag_summary_embedding_snapshot(CONV, {"a": [3.0, 4.0], "b": [0.0, 1.0]})
    first = provider.load_tag_summary_embedding_snapshot(CONV)
    second = provider.load_tag_summary_embedding_snapshot(CONV)
    assert _is_f32(first["a"]) and np.allclose(first["a"], [0.6, 0.8])
    assert second["a"] is first["a"]
    second.pop("b")
    assert "b" in provider.load_tag_summary_embedding_snapshot(CONV)


@pytest.mark.regression("PROXY-038")
def test_inbound_tagger_holds_float32_vectors_and_still_matches():
    provider = _provider()
    provider.save_tag_embeddings(MODEL, {"hcg-dosing": [1.0, 0.0], "camera": [0.0, 1.0]})

    def embed(texts):
        return [[0.9, 0.1] for _ in texts]

    tagger = EmbeddingTagGenerator(
        config=TagGeneratorConfig(max_tags=2),
        model_name=MODEL,
        similarity_threshold=0.5,
        embed_fn=embed,
        load_cached_embeddings=provider.load_tag_embeddings,
        save_cached_embeddings=provider.save_tag_embeddings,
    )
    result = tagger.generate_tags("how long does the hcg kit last", ["hcg-dosing", "camera"])
    assert result.tags[0] == "hcg-dosing"
    assert all(_is_f32(v) for v in tagger._tag_embeddings.values())


def test_embedding_candidates_accept_array_vectors():
    stored = {"a": np.asarray([1.0, 0.0], dtype=np.float32), "empty": np.asarray([], dtype=np.float32)}
    scores = compute_embedding_candidates([1.0, 0.0], store=None, conversation_id=None, stored_embeddings=stored)
    assert list(scores) == ["a"]


def test_redis_values_are_packed_float32_and_legacy_float64_still_reads():
    provider = _provider()
    provider.save_tag_embeddings(MODEL, {"hcg": [3.0, 4.0]})
    raw = provider._redis.get(provider._tag_embedding_cache_key(MODEL, "hcg"))
    assert raw[:4] == b"VCF4" and len(raw) == 4 + 2 * 4

    from array import array
    legacy = b"VCF1" + array("d", [0.5, 0.25]).tobytes()
    provider._redis.set(provider._tag_embedding_cache_key(MODEL, "old"), legacy)
    session_state._PROCESS_TAG_VECTOR_CACHE.clear()
    assert np.allclose(provider.load_tag_embeddings(MODEL, ["old"])["old"], [0.5, 0.25])
    assert provider._redis.get(provider._tag_embedding_cache_key(MODEL, "old"))[:4] == b"VCF4"


def test_snapshot_matrix_is_the_decoded_rows_without_another_copy():
    writer, reader = _provider(), None
    writer.save_tag_summary_embedding_snapshot(CONV, {"a": [3.0, 4.0], "b": [0.0, 1.0]})
    reader = SessionStateProvider(redis_client=writer._redis, store=None)
    tags, matrix = reader.load_tag_summary_embedding_matrix(CONV)
    snapshot = reader.load_tag_summary_embedding_snapshot(CONV)
    assert matrix.dtype == np.float32
    assert all(np.shares_memory(snapshot[tag], matrix) for tag in tags)


class _ChunkStore:
    def __init__(self):
        self.reads = 0

    def get_segment_chunk_embedding_page(self, *, conversation_id=None, limit=200, after=None):
        self.reads += 1
        rows = [{"segment_ref": "s1", "chunk_index": 0, "embedding": [3.0, 4.0], "cursor": ("s1", 0)}]
        return [r for r in rows if after is None or r["cursor"] > after]


def test_segment_chunk_matrix_is_shared_through_redis():
    store = _ChunkStore()
    first = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=store)
    first.save_tag_summary_embedding_snapshot(CONV, {"t": [1.0, 0.0]})
    refs, matrix = first.load_segment_chunk_matrix(CONV)
    assert refs == ["s1"] and np.allclose(matrix, [[0.6, 0.8]]) and matrix.dtype == np.float32
    reads = store.reads

    second = SessionStateProvider(redis_client=first._redis, store=store)
    refs, matrix = second.load_segment_chunk_matrix(CONV)
    assert refs == ["s1"] and np.allclose(matrix, [[0.6, 0.8]])
    assert store.reads == reads


def test_inbound_tagger_shares_the_cached_vectors_and_its_matrix():
    provider = _provider()
    provider.save_tag_embeddings(MODEL, {"hcg-dosing": [1.0, 0.0], "camera": [0.0, 1.0]})
    tagger = EmbeddingTagGenerator(
        config=TagGeneratorConfig(max_tags=2), model_name=MODEL, similarity_threshold=0.5,
        embed_fn=lambda texts: [[0.9, 0.1] for _ in texts],
        load_cached_embeddings=provider.load_tag_embeddings,
        save_cached_embeddings=provider.save_tag_embeddings,
    )
    tagger.generate_tags("hcg kit", ["hcg-dosing", "camera"])
    cached = provider.load_tag_embeddings(MODEL, ["hcg-dosing"])["hcg-dosing"]
    assert tagger._tag_embeddings["hcg-dosing"] is cached
    matrix = tagger._similarity_matrix[2]
    tagger.generate_tags("camera feed", ["hcg-dosing", "camera"])
    assert tagger._similarity_matrix[2] is matrix


# Tag-summary embeddings were read from storage as Python float lists, once for
# the compaction merge check and again for the shared snapshot (BUG-128).

def _sqlite_with_embeddings(tmp_path):
    store = SQLiteStore(str(tmp_path / "e.db"))
    store.store_tag_summary_embedding("a", "c", [3.0, 4.0])
    store.store_tag_summary_embedding("b", "c", [0.0, 2.0])
    store.store_tag_summary_embedding("x", "other", [1.0, 0.0])
    return store


@pytest.mark.regression("BUG-128")
def test_embedding_vectors_are_float32_and_match_the_lists(tmp_path):
    store = _sqlite_with_embeddings(tmp_path)
    vectors = store.load_tag_summary_embedding_vectors(conversation_id="c")
    lists = store.load_tag_summary_embeddings(conversation_id="c")
    assert set(vectors) == set(lists) == {"a", "b"}
    for tag, values in lists.items():
        assert vectors[tag].dtype == np.float32
        assert np.allclose(vectors[tag], values)
    from virtual_context.core.store import tag_summary_embedding_vectors
    assert set(tag_summary_embedding_vectors(store, None)) == {"a", "b", "x"}


@pytest.mark.regression("BUG-128")
def test_snapshot_refresh_never_builds_float_lists(tmp_path, monkeypatch):
    store = _sqlite_with_embeddings(tmp_path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the refresh must not load embeddings as lists")

    monkeypatch.setattr(SQLiteStore, "load_tag_summary_embeddings", refuse)
    provider = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=store)
    snapshot = provider.refresh_tag_summary_embedding_snapshot("c")
    assert np.allclose(snapshot["a"], [0.6, 0.8]) and np.allclose(snapshot["b"], [0.0, 1.0])
    reader = SessionStateProvider(redis_client=provider._redis, store=None)
    tags, matrix = reader.load_tag_summary_embedding_matrix("c")
    assert sorted(tags) == ["a", "b"] and matrix.dtype == np.float32
    assert np.allclose(matrix[tags.index("a")], [0.6, 0.8])


@pytest.mark.regression("BUG-128")
def test_snapshot_from_lists_keeps_its_values(tmp_path):
    provider = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)
    provider.save_tag_summary_embedding_snapshot("c", {"a": [3.0, 4.0], "zero": [0.0, 0.0], "empty": []})
    snapshot = provider.load_tag_summary_embedding_snapshot("c")
    assert set(snapshot) == {"a", "zero"}
    assert np.allclose(snapshot["a"], [0.6, 0.8]) and np.allclose(snapshot["zero"], [0.0, 0.0])
    tags, _matrix = provider.load_tag_summary_embedding_matrix("c")
    assert tags == ["a"]
