"""Tests for CostTracker."""

import pytest

from virtual_context.core.cost_tracker import CostTracker
from virtual_context.types import CostTrackingConfig


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
