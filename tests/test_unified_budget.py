"""Tests for unified context injection budget."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from virtual_context.core.retriever import ContextRetriever
from virtual_context.types import (
    AssemblerConfig,
    Fact,
    RetrievalResult,
    RetrieverConfig,
    StoredSummary,
    TagStats,
)


class TestRetrievalScores:
    def test_retrieval_scores_populated(self):
        """Retriever populates retrieval_scores with IDF scores per primary_tag."""
        store = MagicMock()
        store.get_all_tags.return_value = [
            TagStats(tag="auth", usage_count=2),
            TagStats(tag="database", usage_count=5),
        ]
        store.get_summaries_by_tags.return_value = [
            StoredSummary(
                ref="seg-auth", primary_tag="auth", tags=["auth"],
                summary="Auth summary", summary_tokens=50,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            StoredSummary(
                ref="seg-db", primary_tag="database", tags=["database"],
                summary="DB summary", summary_tokens=50,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
        ]
        store.get_tag_aliases.return_value = {}
        store.query_facts.return_value = []

        tagger = MagicMock()
        tagger.generate_tags.return_value = MagicMock(
            tags=["auth", "database"], related_tags=[], temporal=False, source="llm",
        )

        retriever = ContextRetriever(
            tag_generator=tagger, store=store,
            config=RetrieverConfig(skip_active_tags=False, prefetch_facts=False),
        )
        result = retriever.retrieve("auth and database question")

        assert "auth" in result.retrieval_scores
        assert "database" in result.retrieval_scores
        assert result.retrieval_scores["auth"] > 0
        assert result.retrieval_scores["database"] > 0
