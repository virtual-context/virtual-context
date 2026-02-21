"""Tests for CostTracker."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.cost_tracker import CostTracker
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.types import (
    CompactorConfig,
    CostTrackingConfig,
    Message,
    TagGeneratorConfig,
    TaggedSegment,
)


@pytest.fixture
def tracker():
    config = CostTrackingConfig(
        enabled=True,
        pricing={
            "ollama": {"input_per_1k": 0.0, "output_per_1k": 0.0},
            "anthropic": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
        },
    )
    return CostTracker(config)


class TestCostTracker:
    def test_initial_state(self, tracker):
        summary = tracker.get_summary()
        assert summary.total_retrievals == 0
        assert summary.total_compactions == 0
        assert summary.total_tag_generations == 0
        assert summary.estimated_cost_usd == 0.0

    def test_log_retrieval(self, tracker):
        tracker.log_retrieval(input_tokens=100, output_tokens=50, provider="ollama")
        summary = tracker.get_summary()
        assert summary.total_retrievals == 1
        assert summary.total_input_tokens == 100
        assert summary.total_output_tokens == 50
        assert summary.estimated_cost_usd == 0.0  # ollama is free

    def test_log_compaction(self, tracker):
        tracker.log_compaction(input_tokens=1000, output_tokens=200, provider="anthropic")
        summary = tracker.get_summary()
        assert summary.total_compactions == 1
        expected = (1000 / 1000) * 0.00025 + (200 / 1000) * 0.00125
        assert abs(summary.estimated_cost_usd - expected) < 1e-6

    def test_log_tag_generation(self, tracker):
        tracker.log_tag_generation(input_tokens=500, output_tokens=100, provider="ollama")
        summary = tracker.get_summary()
        assert summary.total_tag_generations == 1

    def test_accumulation(self, tracker):
        tracker.log_retrieval(input_tokens=100, provider="ollama")
        tracker.log_retrieval(input_tokens=200, provider="ollama")
        tracker.log_compaction(input_tokens=500, provider="anthropic")
        summary = tracker.get_summary()
        assert summary.total_retrievals == 2
        assert summary.total_compactions == 1
        assert summary.total_input_tokens == 800

    def test_unknown_provider_no_cost(self, tracker):
        tracker.log_retrieval(input_tokens=1000, output_tokens=500, provider="unknown")
        summary = tracker.get_summary()
        assert summary.estimated_cost_usd == 0.0


class TestCostTrackerIntegration:
    """Verify that compactor and tag generator actually populate the cost tracker."""

    def test_compactor_logs_usage_on_segment_summarization(self, tracker):
        """Compactor._compact_one should log to cost tracker after LLM call."""
        llm = MagicMock()
        llm.complete.return_value = '{"summary": "test", "entities": [], "key_decisions": [], "action_items": [], "date_references": [], "refined_tags": ["a"]}'
        llm.last_usage = {"input_tokens": 500, "output_tokens": 120}

        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
            model_name="claude-haiku-4-5",
            cost_tracker=tracker,
        )

        segment = TaggedSegment(
            id="seg-1",
            tags=["test"],
            primary_tag="test",
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
            turn_count=1,
        )

        compactor._compact_one(segment)

        summary = tracker.get_summary()
        assert summary.total_compactions == 1
        assert summary.total_input_tokens == 500
        assert summary.total_output_tokens == 120

    def test_compactor_logs_usage_on_tag_rollup(self, tracker):
        """compact_tag_summaries should also log usage via cost tracker."""
        from virtual_context.types import StoredSummary, TagSummary

        llm = MagicMock()
        llm.complete.return_value = '{"summary": "rolled up", "description": "test desc"}'
        llm.last_usage = {"input_tokens": 300, "output_tokens": 80}

        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
            model_name="claude-haiku-4-5",
            cost_tracker=tracker,
        )

        stored = [
            StoredSummary(ref="seg-1", summary="sum1", tags=["a"]),
            StoredSummary(ref="seg-2", summary="sum2", tags=["a"]),
        ]

        result = compactor.compact_tag_summaries(
            cover_tags=["a"],
            tag_to_summaries={"a": stored},
            tag_to_turns={"a": [1, 2]},
            existing_tag_summaries={},
            max_turn=2,
        )

        assert len(result) == 1
        summary = tracker.get_summary()
        assert summary.total_compactions == 1
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 80

    def test_tag_generator_logs_usage(self, tracker):
        """LLMTagGenerator.generate_tags should log to cost tracker."""
        llm = MagicMock()
        llm.complete.return_value = '{"tags": ["python", "coding"], "primary": "python", "broad": false, "temporal": false, "related_tags": ["programming"]}'
        llm.last_usage = {"input_tokens": 200, "output_tokens": 50}
        llm.model = "claude-haiku-4-5"

        tagger = LLMTagGenerator(
            llm_provider=llm,
            config=TagGeneratorConfig(),
            cost_tracker=tracker,
        )

        result = tagger.generate_tags("Let's discuss Python programming")

        assert "python" in result.tags
        summary = tracker.get_summary()
        assert summary.total_tag_generations == 1
        assert summary.total_input_tokens == 200
        assert summary.total_output_tokens == 50

    def test_compactor_no_tracker_no_crash(self):
        """Compactor works fine without cost_tracker (backward compat)."""
        llm = MagicMock()
        llm.complete.return_value = '{"summary": "ok", "entities": [], "key_decisions": [], "action_items": [], "date_references": [], "refined_tags": ["x"]}'
        llm.last_usage = {"input_tokens": 100, "output_tokens": 30}

        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
        )

        segment = TaggedSegment(
            id="seg-1", tags=["x"], primary_tag="x",
            messages=[Message(role="user", content="hi"), Message(role="assistant", content="hey")],
            turn_count=1,
        )
        result = compactor._compact_one(segment)
        assert result.summary == "ok"

    def test_openai_style_usage_keys(self, tracker):
        """Verify _log_usage handles OpenAI-style keys (prompt_tokens/completion_tokens)."""
        llm = MagicMock()
        llm.complete.return_value = '{"tags": ["test"], "primary": "test", "broad": false, "temporal": false, "related_tags": []}'
        llm.last_usage = {"prompt_tokens": 400, "completion_tokens": 100}
        llm.model = "qwen3:4b"

        tagger = LLMTagGenerator(
            llm_provider=llm,
            config=TagGeneratorConfig(),
            cost_tracker=tracker,
        )
        tagger.generate_tags("test message")

        summary = tracker.get_summary()
        assert summary.total_input_tokens == 400
        assert summary.total_output_tokens == 100
