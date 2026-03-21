"""Tests for 3-signal RRF fusion retrieval scoring."""
from datetime import datetime, timezone
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import TagSummary


class TestFTSTagSummaries:
    def test_search_tag_summaries_fts(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        ts = TagSummary(
            tag="basketball", summary="Basketball tournament delivery date for remote shutter release",
            summary_tokens=20,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        store.save_tag_summary(ts, conversation_id="c1")
        results = store.search_tag_summaries_fts("delivery shutter", limit=10, conversation_id="c1")
        assert len(results) >= 1
        assert results[0][0] == "basketball"

    def test_search_returns_empty_when_no_match(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        results = store.search_tag_summaries_fts("nonexistent query", limit=10)
        assert results == []

    def test_fts_updates_on_tag_summary_update(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        ts = TagSummary(
            tag="cooking", summary="Italian pasta recipes",
            summary_tokens=10,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        store.save_tag_summary(ts, conversation_id="c1")
        # Update the summary
        ts2 = TagSummary(
            tag="cooking", summary="Japanese sushi preparation techniques",
            summary_tokens=10,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 6, 2, tzinfo=timezone.utc),
        )
        store.save_tag_summary(ts2, conversation_id="c1")
        # Old content should not match
        assert store.search_tag_summaries_fts("pasta", limit=10, conversation_id="c1") == []
        # New content should match
        results = store.search_tag_summaries_fts("sushi", limit=10, conversation_id="c1")
        assert len(results) >= 1
        assert results[0][0] == "cooking"


class TestTagSummaryEmbeddings:
    def test_store_and_load_embeddings(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        store.store_tag_summary_embedding("auth", "c1", [0.1, 0.2, 0.3])
        store.store_tag_summary_embedding("database", "c1", [0.4, 0.5, 0.6])
        loaded = store.load_tag_summary_embeddings("c1")
        assert "auth" in loaded
        assert "database" in loaded
        assert loaded["auth"] == [0.1, 0.2, 0.3]

    def test_load_empty_when_none_stored(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        assert store.load_tag_summary_embeddings("c1") == {}

    def test_upsert_replaces_embedding(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        store.store_tag_summary_embedding("auth", "c1", [0.1, 0.2])
        store.store_tag_summary_embedding("auth", "c1", [0.9, 0.8])
        loaded = store.load_tag_summary_embeddings("c1")
        assert loaded["auth"] == [0.9, 0.8]
