"""Tests for TelemetryLedger integration with compactor, tag generator, and tool loop."""

from unittest.mock import MagicMock, patch

import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.model_catalog import ModelCatalog
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.core.telemetry import TelemetryLedger
from virtual_context.core.tool_loop import run_tool_loop
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


class TestToolLoopTelemetry:
    """Verify that run_tool_loop logs telemetry events."""

    def _make_engine_with_telemetry(self, ledger):
        """Create a mock engine with a telemetry ledger attached."""
        engine = MagicMock()
        engine._telemetry = ledger
        return engine

    def _make_adapter(self, *, tool_calls=None, text="Hello", usage=(100, 50)):
        """Create a mock adapter that returns predictable values."""
        adapter = MagicMock()
        adapter.extract_usage.return_value = usage
        adapter.extract_text.return_value = text
        adapter.get_stop_reason.return_value = "end_turn"
        adapter.extract_tool_calls.return_value = tool_calls or []
        adapter.get_url.return_value = "https://api.example.com/v1/chat"
        adapter.get_headers.return_value = {"Authorization": "Bearer test"}
        return adapter

    def test_tool_loop_logs_initial_usage(self, ledger):
        """run_tool_loop should log initial response usage to telemetry."""
        engine = self._make_engine_with_telemetry(ledger)
        adapter = self._make_adapter(usage=(200, 80))

        initial_response = {"choices": [{"message": {"content": "hi"}}]}
        original_request = {"model": "gpt-4o-mini", "messages": []}

        run_tool_loop(engine, initial_response, original_request, adapter)

        total = ledger.total()
        assert total.call_count == 1
        assert total.input_tokens == 200
        assert total.output_tokens == 80

        events = ledger.events()
        assert len(events) == 1
        assert events[0].component == "tool_loop"
        assert events[0].model == "gpt-4o-mini"
        assert events[0].detail == "initial"
        assert events[0].duration_ms == 0.0

    def test_tool_loop_logs_continuation_usage(self, ledger):
        """run_tool_loop should log continuation round usage to telemetry."""
        engine = self._make_engine_with_telemetry(ledger)

        # First call returns tool calls; second call returns no tool calls
        adapter = MagicMock()
        adapter.get_url.return_value = "https://api.example.com/v1/chat"
        adapter.get_headers.return_value = {"Authorization": "Bearer test"}

        # Initial response: has VC tool calls
        adapter.extract_usage.side_effect = [(150, 60), (300, 120)]
        adapter.extract_text.side_effect = ["", "Final answer"]
        adapter.get_stop_reason.side_effect = ["tool_use", "end_turn"]

        vc_tool_call = {"id": "call_1", "name": "vc_expand_topic", "input": {"tag": "python"}}
        adapter.extract_tool_calls.side_effect = [
            [vc_tool_call],  # initial response has tool calls
            [],              # continuation has no tool calls
        ]
        adapter.is_tool_use_stop.return_value = False
        adapter.build_tool_result.return_value = {
            "tool_call_id": "call_1",
            "content": '{"result": "ok"}',
        }
        adapter.build_continuation.return_value = {"messages": [], "model": "gpt-4o-mini"}
        adapter.compress_previous_results.return_value = None
        adapter.inject_context.return_value = None
        adapter.add_tool_defs.return_value = None

        # Mock the execute_vc_tool to return a simple result
        with patch("virtual_context.core.tool_loop.execute_vc_tool", return_value='{"result": "ok"}'):
            # Mock httpx.Client to return a successful response
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"choices": [{"message": {"content": "done"}}]}
            mock_resp.text = '{"choices": [{"message": {"content": "done"}}]}'
            mock_resp.headers = {"content-type": "application/json"}

            with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                # Also mock _parse_provider_http_response to return predictable data
                with patch(
                    "virtual_context.core.tool_loop._parse_provider_http_response",
                    return_value={"choices": [{"message": {"content": "done"}}]},
                ):
                    # Mock reassemble_context to avoid engine internals
                    engine.reassemble_context.return_value = None

                    initial_response = {"choices": [{"message": {"content": ""}}]}
                    original_request = {"model": "gpt-4o-mini", "messages": []}

                    result = run_tool_loop(
                        engine, initial_response, original_request, adapter,
                    )

        total = ledger.total()
        assert total.call_count == 2  # initial + 1 continuation
        assert total.input_tokens == 150 + 300
        assert total.output_tokens == 60 + 120

        events = ledger.events()
        assert len(events) == 2
        assert events[0].detail == "initial"
        assert events[0].input_tokens == 150
        assert events[1].detail == "round_1"
        assert events[1].input_tokens == 300
        assert events[1].duration_ms >= 0  # timed HTTP call

        by_comp = ledger.by_component()
        assert "tool_loop" in by_comp
        assert by_comp["tool_loop"].call_count == 2

    def test_tool_loop_no_telemetry_no_crash(self):
        """run_tool_loop works when engine has no _telemetry attribute."""
        engine = MagicMock(spec=[])  # spec=[] means no attributes at all
        adapter = MagicMock()
        adapter.extract_usage.return_value = (100, 50)
        adapter.extract_text.return_value = "response"
        adapter.get_stop_reason.return_value = "end_turn"
        adapter.extract_tool_calls.return_value = []

        initial_response = {"choices": [{"message": {"content": "hi"}}]}
        original_request = {"model": "gpt-4o", "messages": []}

        # Should not raise even without _telemetry
        result = run_tool_loop(engine, initial_response, original_request, adapter)
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_tool_loop_model_defaults_to_unknown(self, ledger):
        """When model is missing from original_request, use 'unknown'."""
        engine = self._make_engine_with_telemetry(ledger)
        adapter = self._make_adapter(usage=(50, 20))

        initial_response = {}
        original_request = {"messages": []}  # no "model" key

        run_tool_loop(engine, initial_response, original_request, adapter)

        events = ledger.events()
        assert len(events) == 1
        assert events[0].model == "unknown"
