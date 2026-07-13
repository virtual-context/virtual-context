"""Tests for virtual_context.core.tool_guard."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from virtual_context.core.tool_guard import (
    GUARDED_TOOL_NAMES,
    ToolRepetitionGuard,
    guard_tool_execution,
    normalize_query_fingerprint,
    reset_default_guard,
)
from virtual_context.core.tool_loop import _is_empty_result, execute_vc_tool
from virtual_context.types import SearchConfig


@pytest.fixture(autouse=True)
def _clean_guard_state():
    reset_default_guard()
    yield
    reset_default_guard()


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _engine(conv_id: str, **search_overrides):
    """MagicMock engine with a real SearchConfig and clean find_quote results."""
    engine = MagicMock()
    engine.config.search = SearchConfig(**search_overrides)
    engine.config.conversation_id = conv_id
    engine.find_quote.return_value = {
        "found": True,
        "results": [{"excerpt": "the dosage was 5mg", "turn": 3}],
    }
    engine._store.search_facts.return_value = []
    return engine


# ---------------------------------------------------------------------------
# Fingerprint normalization
# ---------------------------------------------------------------------------

class TestFingerprintNormalization:

    def test_lowercase_and_whitespace_collapse(self):
        assert (
            normalize_query_fingerprint({"query": "  What DID\t Sania \n say  "})
            == normalize_query_fingerprint({"query": "what did sania say"})
            == "what did sania say"
        )

    def test_volatile_args_ignored(self):
        base = normalize_query_fingerprint({"query": "trip plans"})
        assert normalize_query_fingerprint({
            "query": "trip plans",
            "mode": "timeline",
            "channel": "discord",
            "max_results": 3,
            "time_range": {"after": "2026-01-01"},
        }) == base

    def test_fact_query_args_participate(self):
        fp = normalize_query_fingerprint({"subject": "Sania", "verb": "SAID"})
        assert fp == "sania | said"
        assert fp != normalize_query_fingerprint({"subject": "sania"})

    def test_empty_and_non_string_values(self):
        assert normalize_query_fingerprint({}) == ""
        assert normalize_query_fingerprint(None) == ""
        assert normalize_query_fingerprint({"query": 42}) == ""


# ---------------------------------------------------------------------------
# Sliding-window semantics
# ---------------------------------------------------------------------------

class TestToolRepetitionGuard:

    def _run(self, guard, conv, n, *, tool="vc_find_quote", threshold=3,
             window=120.0, query="q"):
        outcomes = []
        for i in range(n):
            outcomes.append(guard.check_and_record(
                conv, tool, {"query": f"{query} {i}"},
                window_seconds=window, threshold=threshold,
            ))
        return outcomes

    def test_trips_at_threshold(self):
        guard = ToolRepetitionGuard(time_fn=FakeClock())
        outcomes = self._run(guard, "conv-a", 4, threshold=3)
        assert outcomes[:3] == [None, None, None]
        trip = outcomes[3]
        assert trip is not None
        assert trip["count"] == 3
        assert len(trip["distinct_queries"]) == 3

    def test_no_trip_below_threshold(self):
        guard = ToolRepetitionGuard(time_fn=FakeClock())
        assert self._run(guard, "conv-b", 3, threshold=3) == [None, None, None]

    def test_window_expiry_resets(self):
        clock = FakeClock()
        guard = ToolRepetitionGuard(time_fn=clock)
        self._run(guard, "conv-c", 3, threshold=3)
        assert guard.check_and_record(
            "conv-c", "vc_find_quote", {"query": "again"},
            window_seconds=120.0, threshold=3,
        ) is not None
        clock.advance(121)
        assert guard.check_and_record(
            "conv-c", "vc_find_quote", {"query": "again"},
            window_seconds=120.0, threshold=3,
        ) is None

    def test_partial_window_expiry(self):
        clock = FakeClock()
        guard = ToolRepetitionGuard(time_fn=clock)
        for i in range(3):
            guard.check_and_record(
                "conv-d", "vc_find_quote", {"query": f"q{i}"},
                window_seconds=120.0, threshold=3,
            )
            clock.advance(50)
        # entries at 1000/1050/1100, now 1150: the first expired
        assert guard.check_and_record(
            "conv-d", "vc_find_quote", {"query": "q3"},
            window_seconds=120.0, threshold=3,
        ) is None

    def test_blocked_calls_are_not_recorded(self):
        clock = FakeClock()
        guard = ToolRepetitionGuard(time_fn=clock)
        self._run(guard, "conv-e", 2, threshold=2)
        for _ in range(5):
            clock.advance(20)
            assert guard.check_and_record(
                "conv-e", "vc_find_quote", {"query": "hammer"},
                window_seconds=120.0, threshold=2,
            ) is not None
        # 100s of blocked hammering later, the original two entries
        # still expire on schedule.
        clock.advance(25)
        assert guard.check_and_record(
            "conv-e", "vc_find_quote", {"query": "hammer"},
            window_seconds=120.0, threshold=2,
        ) is None

    def test_distinct_conversations_independent(self):
        guard = ToolRepetitionGuard(time_fn=FakeClock())
        self._run(guard, "conv-f", 2, threshold=2)
        assert guard.check_and_record(
            "conv-g", "vc_find_quote", {"query": "fresh"},
            window_seconds=120.0, threshold=2,
        ) is None
        assert guard.check_and_record(
            "conv-f", "vc_find_quote", {"query": "fresh"},
            window_seconds=120.0, threshold=2,
        ) is not None

    def test_distinct_tools_independent(self):
        guard = ToolRepetitionGuard(time_fn=FakeClock())
        self._run(guard, "conv-h", 2, threshold=2)
        assert guard.check_and_record(
            "conv-h", "vc_search_summaries", {"query": "fresh"},
            window_seconds=120.0, threshold=2,
        ) is None

    def test_non_search_tools_never_guarded(self):
        guard = ToolRepetitionGuard(time_fn=FakeClock())
        for tool in ("vc_expand_topic", "vc_recall_all", "vc_restore_tool"):
            assert tool not in GUARDED_TOOL_NAMES
            for _ in range(30):
                assert guard.check_and_record(
                    "conv-i", tool, {"tag": "x"},
                    window_seconds=120.0, threshold=1,
                ) is None

    def test_first_call_never_trips(self):
        for threshold in (0, 1, 10):
            guard = ToolRepetitionGuard(time_fn=FakeClock())
            assert guard.check_and_record(
                "conv-j", "vc_find_quote", {"query": "first"},
                window_seconds=120.0, threshold=threshold,
            ) is None

    def test_distinct_queries_deduplicated_by_fingerprint(self):
        guard = ToolRepetitionGuard(time_fn=FakeClock())
        for q in ("What DID  Sania  say", "what did sania say ", "dosage"):
            allowed = guard.check_and_record(
                "conv-k", "vc_find_quote", {"query": q},
                window_seconds=120.0, threshold=3,
            )
            assert allowed is None
        trip = guard.check_and_record(
            "conv-k", "vc_find_quote", {"query": "dosage"},
            window_seconds=120.0, threshold=3,
        )
        assert trip["count"] == 3
        assert trip["distinct_queries"] == ["what did sania say", "dosage"]


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:

    def test_search_config_guard_defaults(self):
        cfg = SearchConfig()
        assert cfg.tool_guard_enabled is True
        assert cfg.tool_guard_window_seconds == 120
        assert cfg.tool_guard_threshold == 10


# ---------------------------------------------------------------------------
# execute_vc_tool wiring
# ---------------------------------------------------------------------------

class TestExecuteVcToolWiring:

    def test_trip_returns_synthetic_result_instead_of_executing(self):
        engine = _engine("conv-wire-1", tool_guard_threshold=2)
        first = json.loads(execute_vc_tool(engine, "vc_find_quote", {"query": "dosage"}))
        second = json.loads(execute_vc_tool(engine, "vc_find_quote", {"query": "the dosage"}))
        assert first["found"] is True
        assert second["found"] is True
        third = json.loads(execute_vc_tool(engine, "vc_find_quote", {"query": "dosage again"}))
        assert third["repetition_guard"] is True
        assert third["searches_already_run"] == 2
        assert third["queries_tried"] == ["dosage", "the dosage"]
        assert "2 similar vc_find_quote searches already ran" in third["message"]
        assert "answer the question from the results already gathered" in third["message"]
        assert engine.find_quote.call_count == 2

    def test_no_trip_below_threshold(self):
        engine = _engine("conv-wire-2", tool_guard_threshold=3)
        for q in ("a", "b", "c"):
            result = json.loads(execute_vc_tool(engine, "vc_find_quote", {"query": q}))
            assert "repetition_guard" not in result
        assert engine.find_quote.call_count == 3

    def test_guard_disabled_never_trips(self):
        engine = _engine(
            "conv-wire-3", tool_guard_enabled=False, tool_guard_threshold=2,
        )
        for i in range(8):
            result = json.loads(execute_vc_tool(engine, "vc_find_quote", {"query": f"q{i}"}))
            assert "repetition_guard" not in result
        assert engine.find_quote.call_count == 8

    def test_non_search_tool_never_guarded(self):
        engine = _engine("conv-wire-4", tool_guard_threshold=1)
        engine.expand_topic.return_value = {"tag": "t", "content": "x"}
        engine.collapse_topic.return_value = {"tokens_freed": 0}
        engine._store.get_segments_by_tags.return_value = []
        for _ in range(5):
            result = json.loads(execute_vc_tool(engine, "vc_expand_topic", {"tag": "t"}))
            assert "repetition_guard" not in result
        assert engine.expand_topic.call_count == 5

    def test_empty_conversation_id_fails_open(self):
        engine = _engine("", tool_guard_threshold=1)
        for i in range(5):
            result = json.loads(execute_vc_tool(engine, "vc_find_quote", {"query": f"q{i}"}))
            assert "repetition_guard" not in result
        assert engine.find_quote.call_count == 5

    def test_mock_config_fails_open(self):
        engine = MagicMock()
        engine.find_quote.return_value = {"found": False}
        for _ in range(5):
            execute_vc_tool(engine, "vc_find_quote", {"query": "q"})
        assert engine.find_quote.call_count == 5

    def test_trip_logs_one_line(self, caplog):
        engine = _engine("conv-wire-log", tool_guard_threshold=2)
        with caplog.at_level(logging.WARNING, logger="virtual_context.core.tool_guard"):
            execute_vc_tool(engine, "vc_find_quote", {"query": "a"})
            execute_vc_tool(engine, "vc_find_quote", {"query": "b"})
            assert "TOOL_GUARD_TRIPPED" not in caplog.text
            execute_vc_tool(engine, "vc_find_quote", {"query": "c"})
        trip_lines = [
            r for r in caplog.records if "TOOL_GUARD_TRIPPED" in r.getMessage()
        ]
        assert len(trip_lines) == 1
        assert (
            "TOOL_GUARD_TRIPPED conv=conv-wire-log tool=vc_find_quote count=2"
            in trip_lines[0].getMessage()
        )

    def test_synthetic_result_is_not_classified_empty(self):
        engine = _engine("conv-wire-5", tool_guard_threshold=1)
        execute_vc_tool(engine, "vc_find_quote", {"query": "a"})
        guarded = execute_vc_tool(engine, "vc_find_quote", {"query": "b"})
        assert json.loads(guarded)["repetition_guard"] is True
        # The stop instruction must not read as an empty result, or the
        # loop would append "try another search tool" strategy hints
        # that contradict it.
        assert _is_empty_result(guarded) is False
