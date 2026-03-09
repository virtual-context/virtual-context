"""TelemetryEvent, TelemetryRollup, TelemetryLedger: structured LLM cost tracking."""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field

from .model_catalog import ModelCatalog


@dataclass(frozen=True)
class TelemetryEvent:
    """A single LLM call record."""

    timestamp: float  # time.time()
    component: str  # "compactor", "tagger", "tool_loop", "fact_curator", "proxy_upstream"
    model: str  # canonical model name
    input_tokens: int
    output_tokens: int
    cost_usd: float  # calculated via ModelCatalog
    duration_ms: float  # wall-clock time for this LLM call
    turn_id: int | None = None
    detail: str = ""  # e.g. "segment_summarize", "tag_rollup"


@dataclass
class TelemetryRollup:
    """Aggregate totals for a group of events."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    call_count: int = 0

    def _add(self, event: TelemetryEvent) -> None:
        self.input_tokens += event.input_tokens
        self.output_tokens += event.output_tokens
        self.cost_usd += event.cost_usd
        self.duration_ms += event.duration_ms
        self.call_count += 1


class TelemetryLedger:
    """Thread-safe ledger that records TelemetryEvents and produces rollups."""

    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog
        self._events: list[TelemetryEvent] = []
        self._lock = threading.Lock()

    def log(
        self,
        component: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        duration_ms: float = 0.0,
        turn_id: int | None = None,
        detail: str = "",
    ) -> TelemetryEvent:
        """Create a TelemetryEvent via ModelCatalog and append it."""
        cost = self._catalog.calculate_cost(model, input_tokens, output_tokens)
        event = TelemetryEvent(
            timestamp=time.time(),
            component=component,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            turn_id=turn_id,
            detail=detail,
        )
        with self._lock:
            self._events.append(event)
        return event

    def events(self) -> list[TelemetryEvent]:
        with self._lock:
            return list(self._events)

    def total(self) -> TelemetryRollup:
        rollup = TelemetryRollup()
        with self._lock:
            for event in self._events:
                rollup._add(event)
        return rollup

    def by_component(self) -> dict[str, TelemetryRollup]:
        result: dict[str, TelemetryRollup] = {}
        with self._lock:
            for event in self._events:
                if event.component not in result:
                    result[event.component] = TelemetryRollup()
                result[event.component]._add(event)
        return result

    def by_model(self) -> dict[str, TelemetryRollup]:
        result: dict[str, TelemetryRollup] = {}
        with self._lock:
            for event in self._events:
                if event.model not in result:
                    result[event.model] = TelemetryRollup()
                result[event.model]._add(event)
        return result

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def to_dict(self) -> dict:
        with self._lock:
            events_snapshot = list(self._events)

        # Build rollups from the snapshot (outside the lock)
        total = TelemetryRollup()
        by_comp: dict[str, TelemetryRollup] = {}
        by_mod: dict[str, TelemetryRollup] = {}
        for event in events_snapshot:
            total._add(event)
            if event.component not in by_comp:
                by_comp[event.component] = TelemetryRollup()
            by_comp[event.component]._add(event)
            if event.model not in by_mod:
                by_mod[event.model] = TelemetryRollup()
            by_mod[event.model]._add(event)

        return {
            "events": [dataclasses.asdict(e) for e in events_snapshot],
            "total": dataclasses.asdict(total),
            "by_component": {k: dataclasses.asdict(v) for k, v in by_comp.items()},
            "by_model": {k: dataclasses.asdict(v) for k, v in by_mod.items()},
        }
