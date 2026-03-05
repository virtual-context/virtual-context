"""Tests for TelemetryLedger integration with compactor and tag generator."""

from unittest.mock import MagicMock

import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.model_catalog import ModelCatalog
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.core.telemetry import TelemetryLedger
from virtual_context.types import (
    CompactorConfig,
    Message,
    TagGeneratorConfig,
    TaggedSegment,
)


@pytest.fixture
def catalog():
    return ModelCatalog.default()


@pytest.fixture
def ledger(catalog):
    return TelemetryLedger(catalog)


class TestTelemetryLedger:
    def test_initial_state(self, ledger):
        total = ledger.total()
        assert total.call_count == 0
        assert total.input_tokens == 0
        assert total.output_tokens == 0
        assert total.cost_usd == 0.0
        assert total.duration_ms == 0.0

    def test_log_event(self, ledger):
        ledger.log(
            component="tagger", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=50, duration_ms=42.0,
        )
        total = ledger.total()
        assert total.call_count == 1
        assert total.input_tokens == 100
        assert total.output_tokens == 50
        assert total.duration_ms == 42.0
        # Cost should be > 0 for a known model
        assert total.cost_usd > 0.0

    def test_by_component(self, ledger):
        ledger.log(component="compactor", model="haiku", input_tokens=500, output_tokens=100)
        ledger.log(component="tagger", model="haiku", input_tokens=200, output_tokens=50)
        by_comp = ledger.by_component()
        assert "compactor" in by_comp
        assert "tagger" in by_comp
        assert by_comp["compactor"].call_count == 1
        assert by_comp["tagger"].call_count == 1

    def test_accumulation(self, ledger):
        ledger.log(component="tagger", model="haiku", input_tokens=100, output_tokens=0)
        ledger.log(component="tagger", model="haiku", input_tokens=200, output_tokens=0)
        ledger.log(component="compactor", model="haiku", input_tokens=500, output_tokens=0)
        total = ledger.total()
        assert total.call_count == 3
        assert total.input_tokens == 800

    def test_unknown_model_zero_cost(self, ledger):
        ledger.log(component="tagger", model="unknown-model-xyz", input_tokens=1000, output_tokens=500)
        total = ledger.total()
        assert total.cost_usd == 0.0


class TestTelemetryIntegration:
    """Verify that compactor and tag generator populate the telemetry ledger."""

    def test_compactor_logs_usage_on_segment_summarization(self, ledger):
        """Compactor._compact_one should log to telemetry ledger after LLM call."""
        llm = MagicMock()
        llm.complete.return_value = '{"summary": "test", "entities": [], "key_decisions": [], "action_items": [], "date_references": [], "refined_tags": ["a"]}'
        llm.last_usage = {"input_tokens": 500, "output_tokens": 120}

        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
            model_name="claude-haiku-4-5",
            telemetry_ledger=ledger,
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

        total = ledger.total()
        assert total.call_count == 1
        assert total.input_tokens == 500
        assert total.output_tokens == 120
        assert total.duration_ms > 0

        by_comp = ledger.by_component()
        assert "compactor" in by_comp
        assert by_comp["compactor"].call_count == 1

    def test_compactor_logs_usage_on_tag_rollup(self, ledger):
        """compact_tag_summaries should also log usage via telemetry ledger."""
        from virtual_context.types import StoredSummary, TagSummary

        llm = MagicMock()
        llm.complete.return_value = '{"summary": "rolled up", "description": "test desc"}'
        llm.last_usage = {"input_tokens": 300, "output_tokens": 80}

        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
            model_name="claude-haiku-4-5",
            telemetry_ledger=ledger,
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
        total = ledger.total()
        assert total.call_count == 1
        assert total.input_tokens == 300
        assert total.output_tokens == 80
        assert total.duration_ms > 0

    def test_tag_generator_logs_usage(self, ledger):
        """LLMTagGenerator.generate_tags should log to telemetry ledger."""
        llm = MagicMock()
        llm.complete.return_value = '{"tags": ["python", "coding"], "primary": "python", "broad": false, "temporal": false, "related_tags": ["programming"]}'
        llm.last_usage = {"input_tokens": 200, "output_tokens": 50}
        llm.model = "claude-haiku-4-5"

        tagger = LLMTagGenerator(
            llm_provider=llm,
            config=TagGeneratorConfig(),
            telemetry_ledger=ledger,
        )

        result = tagger.generate_tags("Let's discuss Python programming")

        assert "python" in result.tags
        total = ledger.total()
        assert total.call_count == 1
        assert total.input_tokens == 200
        assert total.output_tokens == 50
        assert total.duration_ms > 0

        by_comp = ledger.by_component()
        assert "tagger" in by_comp
        assert by_comp["tagger"].call_count == 1

    def test_compactor_no_ledger_no_crash(self):
        """Compactor works fine without telemetry_ledger (backward compat)."""
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

    def test_openai_style_usage_keys(self, ledger):
        """Verify _log_usage handles OpenAI-style keys (prompt_tokens/completion_tokens)."""
        llm = MagicMock()
        llm.complete.return_value = '{"tags": ["test"], "primary": "test", "broad": false, "temporal": false, "related_tags": []}'
        llm.last_usage = {"prompt_tokens": 400, "completion_tokens": 100}
        llm.model = "qwen3:4b"

        tagger = LLMTagGenerator(
            llm_provider=llm,
            config=TagGeneratorConfig(),
            telemetry_ledger=ledger,
        )
        tagger.generate_tags("test message")

        total = ledger.total()
        assert total.input_tokens == 400
        assert total.output_tokens == 100

    def test_backward_compat_cost_tracker_kwarg_ignored(self, ledger):
        """Passing deprecated cost_tracker kwarg should not crash."""
        llm = MagicMock()
        llm.complete.return_value = '{"summary": "ok", "entities": [], "key_decisions": [], "action_items": [], "date_references": [], "refined_tags": ["x"]}'
        llm.last_usage = {"input_tokens": 100, "output_tokens": 30}

        # Should not raise even though cost_tracker is passed
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
            cost_tracker="ignored_value",
        )
        assert compactor._telemetry is None  # telemetry_ledger was not passed

        tagger = LLMTagGenerator(
            llm_provider=llm,
            config=TagGeneratorConfig(),
            cost_tracker="ignored_value",
        )
        assert tagger._telemetry is None
