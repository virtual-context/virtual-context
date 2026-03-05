"""Tests for TelemetryEvent, TelemetryRollup, and TelemetryLedger."""

from __future__ import annotations

import threading

import pytest
import yaml

from virtual_context.core.model_catalog import ModelCatalog
from virtual_context.core.telemetry import TelemetryEvent, TelemetryLedger, TelemetryRollup


@pytest.fixture
def catalog_yaml(tmp_path):
    """Create a temporary models.yaml for testing."""
    data = {
        "models": {
            "claude-haiku-4-5-20251001": {
                "provider": "anthropic",
                "input_per_mtok": 1.00,
                "output_per_mtok": 5.00,
                "context_window": 200000,
                "aliases": ["haiku"],
            },
            "gpt-4.1-nano": {
                "provider": "openai",
                "input_per_mtok": 0.10,
                "output_per_mtok": 0.40,
                "context_window": 128000,
                "aliases": ["gpt4-nano"],
            },
        }
    }
    path = tmp_path / "models.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


@pytest.fixture
def catalog(catalog_yaml):
    return ModelCatalog(catalog_yaml)


@pytest.fixture
def ledger(catalog):
    return TelemetryLedger(catalog)


class TestEmptyLedger:
    def test_empty_total_returns_zeros(self, ledger):
        """Empty ledger total returns all-zero rollup."""
        t = ledger.total()
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.cost_usd == 0.0
        assert t.duration_ms == 0.0
        assert t.call_count == 0

    def test_empty_events_returns_empty_list(self, ledger):
        """Empty ledger events returns empty list."""
        assert ledger.events() == []

    def test_empty_by_component(self, ledger):
        """Empty ledger by_component returns empty dict."""
        assert ledger.by_component() == {}

    def test_empty_by_model(self, ledger):
        """Empty ledger by_model returns empty dict."""
        assert ledger.by_model() == {}


class TestSingleEvent:
    def test_log_and_total(self, ledger):
        """Single event log produces correct total."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500, duration_ms=150.0)
        t = ledger.total()
        assert t.input_tokens == 1000
        assert t.output_tokens == 500
        assert t.call_count == 1
        assert t.duration_ms == 150.0
        # cost = (1000 * 1.00 + 500 * 5.00) / 1_000_000 = 3500 / 1_000_000 = 0.0035
        assert t.cost_usd == pytest.approx(0.0035)

    def test_log_returns_event(self, ledger):
        """log() returns the created TelemetryEvent."""
        event = ledger.log("tagger", "haiku", input_tokens=100, output_tokens=50)
        assert isinstance(event, TelemetryEvent)
        assert event.component == "tagger"
        assert event.model == "haiku"
        assert event.input_tokens == 100
        assert event.output_tokens == 50


class TestByComponent:
    def test_multiple_components(self, ledger):
        """by_component groups events correctly."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500, duration_ms=100.0)
        ledger.log("compactor", "haiku", input_tokens=2000, output_tokens=300, duration_ms=200.0)
        ledger.log("tagger", "gpt4-nano", input_tokens=500, output_tokens=100, duration_ms=50.0)

        by_comp = ledger.by_component()
        assert set(by_comp.keys()) == {"compactor", "tagger"}

        comp = by_comp["compactor"]
        assert comp.input_tokens == 3000
        assert comp.output_tokens == 800
        assert comp.call_count == 2
        assert comp.duration_ms == 300.0

        tag = by_comp["tagger"]
        assert tag.input_tokens == 500
        assert tag.output_tokens == 100
        assert tag.call_count == 1
        assert tag.duration_ms == 50.0


class TestByModel:
    def test_multiple_models(self, ledger):
        """by_model groups events correctly."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500, duration_ms=100.0)
        ledger.log("tagger", "gpt4-nano", input_tokens=500, output_tokens=100, duration_ms=50.0)
        ledger.log("tool_loop", "haiku", input_tokens=2000, output_tokens=800, duration_ms=300.0)

        by_mod = ledger.by_model()
        assert set(by_mod.keys()) == {"haiku", "gpt4-nano"}

        haiku = by_mod["haiku"]
        assert haiku.input_tokens == 3000
        assert haiku.output_tokens == 1300
        assert haiku.call_count == 2
        # cost = (1000*1.00 + 500*5.00 + 2000*1.00 + 800*5.00) / 1_000_000
        # = (1000 + 2500 + 2000 + 4000) / 1_000_000 = 9500 / 1_000_000 = 0.0095
        assert haiku.cost_usd == pytest.approx(0.0095)

        nano = by_mod["gpt4-nano"]
        assert nano.input_tokens == 500
        assert nano.output_tokens == 100
        assert nano.call_count == 1


class TestEventsReturnsCopy:
    def test_events_copy(self, ledger):
        """events() returns a copy — mutating it does not affect ledger."""
        ledger.log("compactor", "haiku", input_tokens=100, output_tokens=50)
        events = ledger.events()
        assert len(events) == 1

        # Mutate the returned list
        events.clear()

        # Ledger's own list is untouched
        assert len(ledger.events()) == 1

    def test_events_are_telemetry_events(self, ledger):
        """events() returns TelemetryEvent instances."""
        ledger.log("compactor", "haiku", input_tokens=100, output_tokens=50)
        events = ledger.events()
        assert all(isinstance(e, TelemetryEvent) for e in events)


class TestTurnIdAndDetail:
    def test_turn_id_stored(self, ledger):
        """turn_id is recorded on the event."""
        ledger.log("compactor", "haiku", input_tokens=100, output_tokens=50, turn_id=7)
        event = ledger.events()[0]
        assert event.turn_id == 7

    def test_detail_stored(self, ledger):
        """detail string is recorded on the event."""
        ledger.log(
            "compactor", "haiku", input_tokens=100, output_tokens=50, detail="segment_summarize"
        )
        event = ledger.events()[0]
        assert event.detail == "segment_summarize"

    def test_defaults(self, ledger):
        """turn_id defaults to None and detail defaults to empty string."""
        ledger.log("compactor", "haiku", input_tokens=100, output_tokens=50)
        event = ledger.events()[0]
        assert event.turn_id is None
        assert event.detail == ""


class TestReset:
    def test_reset_clears_events(self, ledger):
        """reset() clears all events."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500)
        ledger.log("tagger", "gpt4-nano", input_tokens=500, output_tokens=100)
        assert len(ledger.events()) == 2

        ledger.reset()

        assert len(ledger.events()) == 0
        t = ledger.total()
        assert t.input_tokens == 0
        assert t.cost_usd == 0.0
        assert t.call_count == 0

    def test_reset_then_log(self, ledger):
        """Ledger works normally after reset."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500)
        ledger.reset()
        ledger.log("tagger", "gpt4-nano", input_tokens=200, output_tokens=50)
        assert len(ledger.events()) == 1
        assert ledger.total().input_tokens == 200


class TestToDict:
    def test_has_required_keys(self, ledger):
        """to_dict() has events, by_component, by_model, total keys."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500)
        d = ledger.to_dict()
        assert set(d.keys()) == {"events", "total", "by_component", "by_model"}

    def test_events_are_serializable(self, ledger):
        """to_dict() events are plain dicts (not dataclass instances)."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500, detail="test")
        d = ledger.to_dict()
        assert isinstance(d["events"], list)
        assert len(d["events"]) == 1
        event = d["events"][0]
        assert isinstance(event, dict)
        assert event["component"] == "compactor"
        assert event["model"] == "haiku"
        assert event["input_tokens"] == 1000
        assert event["detail"] == "test"

    def test_total_is_dict(self, ledger):
        """to_dict() total is a plain dict with rollup fields."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500)
        d = ledger.to_dict()
        total = d["total"]
        assert isinstance(total, dict)
        assert total["input_tokens"] == 1000
        assert total["output_tokens"] == 500
        assert total["call_count"] == 1

    def test_by_component_is_nested_dict(self, ledger):
        """to_dict() by_component maps component names to rollup dicts."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500)
        ledger.log("tagger", "gpt4-nano", input_tokens=200, output_tokens=50)
        d = ledger.to_dict()
        assert "compactor" in d["by_component"]
        assert "tagger" in d["by_component"]
        assert d["by_component"]["compactor"]["input_tokens"] == 1000

    def test_by_model_is_nested_dict(self, ledger):
        """to_dict() by_model maps model names to rollup dicts."""
        ledger.log("compactor", "haiku", input_tokens=1000, output_tokens=500)
        d = ledger.to_dict()
        assert "haiku" in d["by_model"]
        assert d["by_model"]["haiku"]["call_count"] == 1

    def test_empty_to_dict(self, ledger):
        """to_dict() on empty ledger returns zero-valued structures."""
        d = ledger.to_dict()
        assert d["events"] == []
        assert d["total"]["call_count"] == 0
        assert d["by_component"] == {}
        assert d["by_model"] == {}


class TestThreadSafety:
    def test_concurrent_logging(self, ledger):
        """4 threads x 50 events = 200 total events, no data loss."""
        barrier = threading.Barrier(4)

        def worker(component: str):
            barrier.wait()
            for _ in range(50):
                ledger.log(component, "haiku", input_tokens=100, output_tokens=50, duration_ms=1.0)

        threads = [
            threading.Thread(target=worker, args=(f"comp_{i}",))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ledger.events()) == 200
        total = ledger.total()
        assert total.call_count == 200
        assert total.input_tokens == 200 * 100
        assert total.output_tokens == 200 * 50
        assert total.duration_ms == pytest.approx(200 * 1.0)

        by_comp = ledger.by_component()
        assert len(by_comp) == 4
        for comp_rollup in by_comp.values():
            assert comp_rollup.call_count == 50


class TestTelemetryEventFrozen:
    def test_event_is_immutable(self):
        """TelemetryEvent is frozen — cannot reassign attributes."""
        event = TelemetryEvent(
            timestamp=0.0,
            component="compactor",
            model="haiku",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            duration_ms=10.0,
        )
        with pytest.raises(AttributeError):
            event.input_tokens = 999  # type: ignore[misc]


class TestTelemetryRollup:
    def test_add_accumulates(self):
        """TelemetryRollup._add accumulates values correctly."""
        rollup = TelemetryRollup()
        e1 = TelemetryEvent(
            timestamp=0.0, component="a", model="m",
            input_tokens=100, output_tokens=50, cost_usd=0.01, duration_ms=10.0,
        )
        e2 = TelemetryEvent(
            timestamp=0.0, component="b", model="m",
            input_tokens=200, output_tokens=75, cost_usd=0.02, duration_ms=20.0,
        )
        rollup._add(e1)
        rollup._add(e2)
        assert rollup.input_tokens == 300
        assert rollup.output_tokens == 125
        assert rollup.cost_usd == pytest.approx(0.03)
        assert rollup.duration_ms == pytest.approx(30.0)
        assert rollup.call_count == 2
