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


class TestAssemblerConfigBackwardCompat:
    def test_context_injection_defaults_to_sum(self):
        """When context_injection_max_tokens is not set, it equals tag + facts budgets."""
        cfg = AssemblerConfig()
        assert cfg.context_injection_max_tokens == 50_000  # 30k + 20k

    def test_context_injection_explicit(self):
        """Explicit context_injection_max_tokens overrides the sum."""
        cfg = AssemblerConfig(context_injection_max_tokens=40_000)
        assert cfg.context_injection_max_tokens == 40_000

    def test_context_injection_custom_tag_facts(self):
        """Custom tag + facts budgets change the default sum."""
        cfg = AssemblerConfig(tag_context_max_tokens=10_000, facts_max_tokens=5_000)
        assert cfg.context_injection_max_tokens == 15_000

    def test_context_injection_from_yaml(self):
        """Config parsing picks up context_injection_max_tokens from assembly section."""
        from virtual_context.config import load_config
        import tempfile, os, yaml
        cfg_data = {
            "assembly": {"context_injection_max_tokens": 60000},
            "providers": {"default": {"provider": "ollama", "model": "test"}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            path = f.name
        try:
            config = load_config(path, validate=False)
            assert config.assembler.context_injection_max_tokens == 60000
        finally:
            os.unlink(path)
