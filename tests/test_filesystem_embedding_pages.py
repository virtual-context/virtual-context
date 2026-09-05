"""Filesystem fallback streams legacy arrays without reading archive vectors."""

import json
from pathlib import Path

import pytest

from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.filesystem import FilesystemStore, _iter_embedding_json_array, _segment_to_markdown
from virtual_context.types import ChunkEmbedding, SegmentMetadata, StoredSegment, VirtualContextConfig


def test_filesystem_pages_are_sorted_scoped_and_incrementally_decoded(tmp_path, monkeypatch):
    store = FilesystemStore(tmp_path)
    for ref, owner in [("a", "mine"), ("b", "mine"), ("z", "foreign")]:
        store.store_segment(StoredSegment(ref=ref, conversation_id=owner, primary_tag="topic"))
        store.store_chunk_embeddings(ref, [ChunkEmbedding(
            segment_ref=ref, chunk_index=n, text=f"{ref}-{n}", embedding=[1.0, 0.0],
        ) for n in reversed(range(7))])
    (tmp_path / "_embeddings" / "z.json").write_text("corrupt outside scope")
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.parent.name != "_embeddings", "embedding archive read_text"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    result, cursor = [], None
    while page := store.get_segment_chunk_embedding_page(conversation_id="mine", limit=3, after=cursor):
        assert len(page) <= 3
        result.extend(item["text"] for item in page)
        cursor = page[-1]["cursor"]
    assert result == [f"{ref}-{n}" for ref in ("a", "b") for n in range(7)]
    monkeypatch.setattr(store, "get_all_chunk_embeddings", lambda **kwargs: pytest.fail("archive scan"))
    monkeypatch.setattr(store, "get_segment", lambda *args, **kwargs: pytest.fail("per-candidate source read"))
    manager = SemanticSearchManager(store, VirtualContextConfig(conversation_id="mine"))
    manager._embed_fn = lambda values: [[1.0, 0.0] for _ in values]
    assert [hit.segment_ref for hit in manager.semantic_search("q", conversation_id="mine")] == ["a", "b"]
    assert store.get_canonical_turn_chunk_embedding_page() == []


def test_page_metadata_refreshes_legacy_and_edited_sources_and_rejects_orphans(tmp_path, monkeypatch):
    store = FilesystemStore(tmp_path)
    segment = StoredSegment(ref="a", conversation_id="mine", primary_tag="topic", tags=["topic", "old"],
                            metadata=SegmentMetadata(session_date="2026-09-01"))
    store.store_segment(segment)
    store.store_chunk_embeddings("a", [ChunkEmbedding(segment_ref="a", chunk_index=n, text=f"part-{n}", embedding=[1.0]) for n in range(3)])
    store._index["a"].pop("_embedding_metadata")  # Pre-upgrade index.
    original = store.get_segment
    calls = []
    def read(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "get_segment", read)
    first, = store.get_segment_chunk_embedding_page(conversation_id="mine", limit=1)
    store.get_segment_chunk_embedding_page(conversation_id="mine", limit=1, after=first["cursor"])
    assert len(calls) == 1
    assert first["tags"] == ["topic", "old"] and first["session_date"] == "2026-09-01"
    segment.tags = ["topic", "new"]
    segment.metadata.session_date = "2026-09-02"
    path = store._segment_path("topic", "a")
    path.write_text(_segment_to_markdown(segment))
    changed, = store.get_segment_chunk_embedding_page(conversation_id="mine", limit=1)
    assert changed["tags"] == ["topic", "new"] and changed["session_date"] == "2026-09-02"
    assert len(calls) == 2
    segment.conversation_id = "foreign"
    path.write_text(_segment_to_markdown(segment))
    assert store.get_segment_chunk_embedding_page(conversation_id="mine") == []
    path.unlink()
    assert store.get_segment_chunk_embedding_page(conversation_id="mine") == []


def test_parser_crosses_read_boundary_and_rejects_unbounded_single_items(tmp_path):
    path = tmp_path / "array.json"
    path.write_text(json.dumps([{"text": "你🧭" * 40000}, {"text": "last"}]))
    assert [row["text"] for row in _iter_embedding_json_array(path)] == ["你🧭" * 40000, "last"]
    path.write_text('[{"text":"' + "x" * (4 * 1024 * 1024) + '"}]')
    with pytest.raises(ValueError, match="exceeds 4 MiB"):
        next(_iter_embedding_json_array(path))


@pytest.mark.parametrize("text", ['{}', '[{}', '[{},]', '[{}]true'])
def test_streaming_parser_fails_visible_for_corrupt_arrays(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(_iter_embedding_json_array(path))
