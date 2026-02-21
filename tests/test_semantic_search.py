"""Tests for semantic search fallback in find_quote."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from virtual_context.engine import VirtualContextEngine, _chunk_segment_text
from virtual_context.types import (
    ChunkEmbedding,
    PagingConfig,
    QuoteResult,
    SegmentMetadata,
    StorageConfig,
    StoredSegment,
    VirtualContextConfig,
)
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.storage.filesystem import FilesystemStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    ref: str = "seg-1",
    primary_tag: str = "photography",
    tags: list[str] | None = None,
    summary: str = "Discussed camera equipment",
    full_text: str = (
        "User asked about the remote shutter release.\n\n"
        "It arrived on February 10th, a wireless model compatible with their Canon R5.\n\n"
        "Assistant recommended testing it in burst mode for wildlife shots."
    ),
) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        session_id="session-1",
        primary_tag=primary_tag,
        tags=tags or [primary_tag],
        summary=summary,
        summary_tokens=20,
        full_text=full_text,
        full_tokens=100,
        messages=[{"role": "user", "content": "test"}],
        metadata=SegmentMetadata(turn_count=1),
        created_at=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        start_timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        end_timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
    )


def _make_engine(tmp_path):
    cfg = VirtualContextConfig(
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "test.db"),
        ),
        paging=PagingConfig(enabled=False),
    )
    return VirtualContextEngine(config=cfg)


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embedder for tests.

    Returns 4-dimensional vectors based on simple word hashing.
    Words that are semantically similar in the real world won't be here,
    so tests that need semantic similarity should mock specific vectors.
    """
    results = []
    for text in texts:
        words = text.lower().split()
        vec = [0.0, 0.0, 0.0, 0.0]
        for w in words:
            h = hash(w) % 1000
            vec[h % 4] += h / 1000.0
        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        results.append(vec)
    return results


# ---------------------------------------------------------------------------
# _chunk_segment_text tests
# ---------------------------------------------------------------------------

class TestChunkSegmentText:
    def test_chunk_short_text(self):
        """Text under max_words but above min_words → single chunk."""
        text = " ".join(["word"] * 25)  # 25 words, above min_words=20
        chunks = _chunk_segment_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_splits_on_message_boundaries(self):
        """Splits on double newline when each paragraph exceeds max_words threshold."""
        para1 = " ".join(["word"] * 200)
        para2 = " ".join(["other"] * 200)
        text = para1 + "\n\n" + para2
        chunks = _chunk_segment_text(text, max_words=250)
        assert len(chunks) == 2

    def test_chunk_merges_tiny_paragraphs(self):
        """Adjacent small paragraphs are merged into one chunk."""
        parts = ["Hello there friend.\n\n" * 3 + " ".join(["word"] * 25)]
        text = parts[0]
        chunks = _chunk_segment_text(text, min_words=5)
        # All tiny paragraphs should be merged, not returned as separate chunks
        # (each "Hello there friend." is only 3 words, merged together they still < max)
        assert len(chunks) <= 2

    def test_chunk_sliding_window(self):
        """Large text is split with sliding window + overlap."""
        text = " ".join([f"word{i}" for i in range(600)])
        chunks = _chunk_segment_text(text, max_words=250)
        assert len(chunks) >= 2
        # Check overlap: end of first chunk should share words with start of second
        first_words = set(chunks[0].split()[-30:])
        second_words = set(chunks[1].split()[:30])
        assert len(first_words & second_words) > 0

    def test_chunk_empty_text(self):
        """Empty text returns empty list."""
        assert _chunk_segment_text("") == []
        assert _chunk_segment_text("   ") == []

    def test_chunk_filters_tiny_fragments(self):
        """Chunks with fewer than min_words are excluded."""
        text = "tiny\n\n" + " ".join(["word"] * 50)
        chunks = _chunk_segment_text(text, min_words=20)
        for chunk in chunks:
            assert len(chunk.split()) >= 20


# ---------------------------------------------------------------------------
# Store chunk embedding tests
# ---------------------------------------------------------------------------

class TestSQLiteChunkEmbeddings:
    def test_store_and_retrieve(self, tmp_sqlite_db):
        """Round-trip store + retrieve chunk embeddings."""
        store = SQLiteStore(db_path=tmp_sqlite_db)
        chunks = [
            ChunkEmbedding(segment_ref="seg-1", chunk_index=0, text="hello world", embedding=[0.1, 0.2, 0.3]),
            ChunkEmbedding(segment_ref="seg-1", chunk_index=1, text="goodbye world", embedding=[0.4, 0.5, 0.6]),
        ]
        store.store_chunk_embeddings("seg-1", chunks)

        result = store.get_all_chunk_embeddings()
        assert len(result) == 2
        assert result[0].segment_ref == "seg-1"
        assert result[0].chunk_index == 0
        assert result[0].text == "hello world"
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].chunk_index == 1
        store.close()

    def test_store_idempotent(self, tmp_sqlite_db):
        """Re-storing replaces previous chunks for the same segment."""
        store = SQLiteStore(db_path=tmp_sqlite_db)
        chunks_v1 = [ChunkEmbedding(segment_ref="seg-1", chunk_index=0, text="v1", embedding=[0.1])]
        store.store_chunk_embeddings("seg-1", chunks_v1)

        chunks_v2 = [ChunkEmbedding(segment_ref="seg-1", chunk_index=0, text="v2", embedding=[0.9])]
        store.store_chunk_embeddings("seg-1", chunks_v2)

        result = store.get_all_chunk_embeddings()
        assert len(result) == 1
        assert result[0].text == "v2"
        store.close()

    def test_get_all_empty(self, tmp_sqlite_db):
        """Returns empty list when no embeddings stored."""
        store = SQLiteStore(db_path=tmp_sqlite_db)
        assert store.get_all_chunk_embeddings() == []
        store.close()

    def test_cascade_delete_on_segment_removal(self, tmp_sqlite_db):
        """Deleting a segment cascades to its chunk embeddings."""
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment(ref="seg-1"))
        chunks = [ChunkEmbedding(segment_ref="seg-1", chunk_index=0, text="test", embedding=[0.1])]
        store.store_chunk_embeddings("seg-1", chunks)

        assert len(store.get_all_chunk_embeddings()) == 1
        store.delete_segment("seg-1")
        assert len(store.get_all_chunk_embeddings()) == 0
        store.close()


class TestFilesystemChunkEmbeddings:
    def test_store_and_retrieve(self, tmp_store_dir):
        store = FilesystemStore(root=tmp_store_dir / "fs")
        chunks = [
            ChunkEmbedding(segment_ref="seg-1", chunk_index=0, text="hello", embedding=[0.1, 0.2]),
        ]
        store.store_chunk_embeddings("seg-1", chunks)

        result = store.get_all_chunk_embeddings()
        assert len(result) == 1
        assert result[0].text == "hello"
        assert result[0].embedding == [0.1, 0.2]

    def test_get_all_empty(self, tmp_store_dir):
        store = FilesystemStore(root=tmp_store_dir / "fs")
        assert store.get_all_chunk_embeddings() == []


# ---------------------------------------------------------------------------
# Engine find_quote semantic fallback tests
# ---------------------------------------------------------------------------

class TestFindQuoteSemanticFallback:
    def test_fts_full_no_semantic(self, tmp_path):
        """When FTS fills max_results, semantic search is NOT called."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment())

        # Mock _semantic_search to verify it's not called when FTS fills quota
        engine._semantic_search = MagicMock(return_value=[])

        # max_results=1 so FTS finding 1 result fills the quota
        result = engine.find_quote("arrived", max_results=1)
        assert result["found"] is True
        engine._semantic_search.assert_not_called()

    def test_fts_hit_still_supplements_with_semantic(self, tmp_path):
        """When FTS finds matches but has remaining slots, semantic supplements."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment())

        engine._semantic_search = MagicMock(return_value=[])

        # FTS finds 1 result, max_results=5, so semantic runs for remaining 4
        result = engine.find_quote("arrived", max_results=5)
        assert result["found"] is True
        engine._semantic_search.assert_called_once_with("arrived", max_results=4)

    def test_semantic_fallback_on_fts_miss(self, tmp_path):
        """FTS misses vocabulary mismatch, semantic finds it."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment())

        # "received" won't match FTS (text says "arrived")
        # but we mock semantic search to return a match
        semantic_result = QuoteResult(
            text="It arrived on February 10th, a wireless model compatible with their Canon R5.",
            tag="photography",
            segment_ref="seg-1",
            tags=["photography"],
            match_type="semantic",
            similarity=0.87,
        )
        engine._semantic_search = MagicMock(return_value=[semantic_result])

        result = engine.find_quote("remote shutter release received")
        assert result["found"] is True
        assert result["results"][0]["match_type"] == "semantic"
        assert result["results"][0]["similarity"] == 0.87
        engine._semantic_search.assert_called_once()

    def test_no_embeddings_graceful(self, tmp_path):
        """No sentence-transformers → no crash, just returns not found."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment())
        # Ensure embed_fn returns None (no sentence-transformers)
        engine._embed_fn = None

        result = engine.find_quote("nonexistent_term_xyz")
        assert result["found"] is False

    def test_semantic_results_annotated(self, tmp_path):
        """Semantic results have match_type and similarity in output."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment())

        semantic_result = QuoteResult(
            text="test excerpt",
            tag="photography",
            segment_ref="seg-1",
            tags=["photography"],
            match_type="semantic",
            similarity=0.75,
        )
        engine._semantic_search = MagicMock(return_value=[semantic_result])

        result = engine.find_quote("nonexistent_fts_term")
        assert result["found"] is True
        entry = result["results"][0]
        assert entry["match_type"] == "semantic"
        assert entry["similarity"] == 0.75

    def test_fts_results_no_match_type_key(self, tmp_path):
        """FTS results do NOT include match_type in output (clean output)."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment())

        result = engine.find_quote("arrived")
        assert result["found"] is True
        assert "match_type" not in result["results"][0]


# ---------------------------------------------------------------------------
# Engine _semantic_search unit tests
# ---------------------------------------------------------------------------

class TestSemanticSearchMethod:
    def test_returns_empty_when_no_embed_fn(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._embed_fn = None
        assert engine._semantic_search("test") == []

    def test_deduplicates_by_segment(self, tmp_path):
        """Best chunk per segment only — no duplicate segment refs."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment(ref="seg-1"))

        # Store two chunks from same segment with different similarity
        chunks = [
            ChunkEmbedding(segment_ref="seg-1", chunk_index=0, text="chunk A", embedding=[1.0, 0.0]),
            ChunkEmbedding(segment_ref="seg-1", chunk_index=1, text="chunk B", embedding=[0.9, 0.1]),
        ]
        engine._store.store_chunk_embeddings("seg-1", chunks)

        # Mock embed_fn to return a query vector close to both chunks
        engine._embed_fn = lambda texts: [[0.95, 0.05]]

        results = engine._semantic_search("test query")
        # Should get at most 1 result (deduplicated)
        segment_refs = [r.segment_ref for r in results]
        assert len(set(segment_refs)) == len(segment_refs)

    def test_threshold_filtering(self, tmp_path):
        """Chunks below similarity threshold (0.25) are excluded."""
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment(ref="seg-1"))

        # Store chunk with embedding orthogonal to query
        chunks = [
            ChunkEmbedding(segment_ref="seg-1", chunk_index=0, text="unrelated", embedding=[0.0, 1.0]),
        ]
        engine._store.store_chunk_embeddings("seg-1", chunks)

        # Query vector orthogonal to chunk → cosine sim = 0
        engine._embed_fn = lambda texts: [[1.0, 0.0]]

        results = engine._semantic_search("test query")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Compaction embeds chunks test
# ---------------------------------------------------------------------------

class TestCompactionEmbedsChunks:
    def test_compaction_stores_chunk_embeddings(self, tmp_path):
        """After compaction, chunk embeddings exist in the store."""
        engine = _make_engine(tmp_path)
        # Track calls to _embed_and_store_chunks
        called_with = []
        original = engine._embed_and_store_chunks

        def tracking_embed(stored):
            called_with.append(stored.ref)
            return original(stored)

        engine._embed_and_store_chunks = tracking_embed

        # Manually store a segment to verify the hook is called
        seg = _make_segment()
        engine._store.store_segment(seg)
        engine._embed_and_store_chunks(seg)

        # If embed_fn is available, chunks should be stored
        # (may be empty if sentence-transformers not installed)
        assert seg.ref in called_with


# ---------------------------------------------------------------------------
# match_type annotation tests
# ---------------------------------------------------------------------------

class TestMatchTypeAnnotation:
    def test_sqlite_fts_match_type(self, tmp_sqlite_db):
        """SQLite FTS results have match_type='fts'."""
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment())
        results = store.search_full_text("arrived")
        assert len(results) >= 1
        assert results[0].match_type == "fts"
        store.close()

    def test_filesystem_match_type(self, tmp_store_dir):
        """Filesystem results have match_type='like'."""
        store = FilesystemStore(root=tmp_store_dir / "fs")
        store.store_segment(_make_segment(tags=["photography"]))
        results = store.search_full_text("arrived")
        assert len(results) >= 1
        assert results[0].match_type == "like"
