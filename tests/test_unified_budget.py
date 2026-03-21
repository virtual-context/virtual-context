"""Tests for unified context injection budget."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from virtual_context.core.assembler import ContextAssembler
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


def _make_summary(tag, text, tokens=None):
    return StoredSummary(
        ref=f"seg-{tag}", primary_tag=tag, tags=[tag],
        summary=text, summary_tokens=tokens or len(text) // 4,
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )


def _make_fact(subject, verb, obj, tags, mentioned_at=None):
    return Fact(
        subject=subject, verb=verb, object=obj, tags=tags,
        what=f"{subject} {verb} {obj}",
        mentioned_at=mentioned_at or datetime(2024, 6, 1, tzinfo=timezone.utc),
    )


class TestUnifiedPoolAllocation:
    def _make_assembler(self, pool=50_000, tag_cap=30_000, facts_cap=20_000):
        return ContextAssembler(
            config=AssemblerConfig(
                context_injection_max_tokens=pool,
                tag_context_max_tokens=tag_cap,
                facts_max_tokens=facts_cap,
            ),
            token_counter=lambda text: len(text) // 4,
        )

    def test_specific_query_facts_fill_pool_without_caps(self):
        """Without soft caps, facts can use the entire pool when tags use little."""
        asm = self._make_assembler(pool=1000, tag_cap=1000, facts_cap=1000)
        summaries = [_make_summary("oura", "Sleep data.")]
        facts = [
            _make_fact("user", "wears", f"oura ring model {i}", tags=["oura"])
            for i in range(50)
        ]
        result = asm.assemble(
            core_context="",
            retrieval_result=RetrievalResult(
                tags_matched=["oura"], summaries=summaries, facts=facts,
                retrieval_scores={"oura": 5.0},
                retrieval_metadata={"tags_queried": ["oura"], "related_tags_used": []},
            ),
            conversation_history=[],
            token_budget=100_000,
        )
        assert result.budget_breakdown["tags"] < 50
        assert result.budget_breakdown["facts"] > 100
        assert result.budget_breakdown["tags"] + result.budget_breakdown["facts"] <= 1000

    def test_broad_query_both_categories_fill(self):
        """Both summaries and facts compete for the pool."""
        asm = self._make_assembler(pool=500, tag_cap=500, facts_cap=500)
        summaries = [_make_summary(f"topic-{i}", f"Summary for topic {i}. " * 10) for i in range(5)]
        facts = [_make_fact("user", "likes", f"thing-{i}", tags=[f"topic-{i}"]) for i in range(5)]
        scores = {f"topic-{i}": 5.0 - i for i in range(5)}
        result = asm.assemble(
            core_context="",
            retrieval_result=RetrievalResult(
                tags_matched=[f"topic-{i}" for i in range(5)],
                summaries=summaries, facts=facts,
                retrieval_scores=scores,
                retrieval_metadata={"tags_queried": [f"topic-{i}" for i in range(5)], "related_tags_used": []},
            ),
            conversation_history=[],
            token_budget=100_000,
        )
        total = result.budget_breakdown["tags"] + result.budget_breakdown["facts"]
        assert total <= 500

    def test_soft_cap_prevents_one_category_dominating(self):
        """Tag soft cap prevents summaries from consuming the entire pool."""
        asm = self._make_assembler(pool=1000, tag_cap=200, facts_cap=1000)
        summaries = [_make_summary("big", "x" * 3200)]  # 800 tokens
        facts = [_make_fact("user", "does", "something", tags=["big"])]
        result = asm.assemble(
            core_context="",
            retrieval_result=RetrievalResult(
                tags_matched=["big"], summaries=summaries, facts=facts,
                retrieval_scores={"big": 5.0},
                retrieval_metadata={"tags_queried": ["big"], "related_tags_used": []},
            ),
            conversation_history=[],
            token_budget=100_000,
        )
        assert result.budget_breakdown["tags"] <= 200

    def test_backward_compat_no_retrieval_scores(self):
        """When retrieval_scores is empty, tag sections use priority order."""
        asm = self._make_assembler(pool=500)
        summaries = [_make_summary("alpha", "Alpha summary text.")]
        result = asm.assemble(
            core_context="",
            retrieval_result=RetrievalResult(
                tags_matched=["alpha"], summaries=summaries,
            ),
            conversation_history=[],
            token_budget=100_000,
        )
        assert "alpha" in result.tag_sections

    def test_presented_refs_preserved(self):
        """presented_segment_refs is populated correctly after pool allocation."""
        asm = self._make_assembler(pool=5000)
        summaries = [_make_summary("auth", "Auth summary.")]
        result = asm.assemble(
            core_context="",
            retrieval_result=RetrievalResult(
                tags_matched=["auth"], summaries=summaries,
                retrieval_scores={"auth": 3.0},
            ),
            conversation_history=[],
            token_budget=100_000,
        )
        assert "seg-auth" in result.presented_segment_refs
