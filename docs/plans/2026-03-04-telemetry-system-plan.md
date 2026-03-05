# Telemetry System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fragmented cost tracking with a unified telemetry system that tracks cost, token volume, and timing per LLM call, with views in the proxy dashboard, CLI, and JSON reports.

**Architecture:** A `ModelCatalog` loads pricing from `models.yaml`. A thread-safe `TelemetryLedger` collects `TelemetryEvent` records from every LLM call site. Rollup methods provide per-component and per-model views. The existing `CostTracker`, `SessionCostSummary`, and benchmark `BudgetTracker` are replaced.

**Tech Stack:** Python dataclasses, threading.Lock, YAML (PyYAML already a dependency), existing provider `last_usage` pattern.

---

### Task 1: ModelCatalog — pricing from YAML

**Files:**
- Create: `virtual_context/core/model_catalog.py`
- Create: `models.yaml` (project root, alongside `virtual-context.yaml`)
- Test: `tests/test_model_catalog.py`

**Step 1: Write the failing tests**

```python
"""Tests for ModelCatalog."""

import os
import tempfile

import pytest
import yaml

from virtual_context.core.model_catalog import ModelCatalog


class TestModelCatalog:
    def _write_catalog(self, tmp_path, models: dict) -> str:
        path = os.path.join(tmp_path, "models.yaml")
        with open(path, "w") as f:
            yaml.dump({"models": models}, f)
        return path

    def test_load_from_yaml(self, tmp_path):
        path = self._write_catalog(tmp_path, {
            "claude-haiku-4-5-20251001": {
                "provider": "anthropic",
                "input_per_mtok": 1.00,
                "output_per_mtok": 5.00,
                "context_window": 200000,
                "aliases": ["haiku", "claude-haiku"],
            },
        })
        catalog = ModelCatalog(path)
        inp, out = catalog.get_pricing("claude-haiku-4-5-20251001")
        assert inp == 1.00
        assert out == 5.00

    def test_alias_resolution(self, tmp_path):
        path = self._write_catalog(tmp_path, {
            "claude-haiku-4-5-20251001": {
                "provider": "anthropic",
                "input_per_mtok": 1.00,
                "output_per_mtok": 5.00,
                "context_window": 200000,
                "aliases": ["haiku", "claude-haiku"],
            },
        })
        catalog = ModelCatalog(path)
        inp, out = catalog.get_pricing("haiku")
        assert inp == 1.00
        assert out == 5.00

    def test_substring_fallback(self, tmp_path):
        path = self._write_catalog(tmp_path, {
            "claude-haiku-4-5-20251001": {
                "provider": "anthropic",
                "input_per_mtok": 1.00,
                "output_per_mtok": 5.00,
                "context_window": 200000,
                "aliases": ["haiku"],
            },
        })
        catalog = ModelCatalog(path)
        inp, out = catalog.get_pricing("claude-haiku-4-5")
        assert inp == 1.00

    def test_unknown_model_returns_zero(self, tmp_path):
        path = self._write_catalog(tmp_path, {})
        catalog = ModelCatalog(path)
        inp, out = catalog.get_pricing("unknown-model")
        assert inp == 0.0
        assert out == 0.0

    def test_calculate_cost(self, tmp_path):
        path = self._write_catalog(tmp_path, {
            "gpt-5-mini": {
                "provider": "openai",
                "input_per_mtok": 0.25,
                "output_per_mtok": 2.00,
                "context_window": 128000,
                "aliases": [],
            },
        })
        catalog = ModelCatalog(path)
        cost = catalog.calculate_cost("gpt-5-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - 2.25) < 1e-6

    def test_get_context_window(self, tmp_path):
        path = self._write_catalog(tmp_path, {
            "gpt-5-mini": {
                "provider": "openai",
                "input_per_mtok": 0.25,
                "output_per_mtok": 2.00,
                "context_window": 128000,
                "aliases": [],
            },
        })
        catalog = ModelCatalog(path)
        assert catalog.get_context_window("gpt-5-mini") == 128000

    def test_missing_file_creates_empty_catalog(self):
        catalog = ModelCatalog("/nonexistent/models.yaml")
        inp, out = catalog.get_pricing("anything")
        assert inp == 0.0
        assert out == 0.0

    def test_default_catalog_loads(self):
        """The bundled models.yaml at project root should load."""
        # Find models.yaml relative to the package
        catalog = ModelCatalog.default()
        # Should have at least one model
        assert len(catalog._models) > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_catalog.py -v`
Expected: FAIL — module not found

**Step 3: Create `models.yaml`**

```yaml
models:
  claude-haiku-4-5-20251001:
    provider: anthropic
    input_per_mtok: 1.00
    output_per_mtok: 5.00
    context_window: 200000
    aliases: [haiku, claude-haiku, claude-haiku-4-5]

  claude-sonnet-4-5-20250929:
    provider: anthropic
    input_per_mtok: 3.00
    output_per_mtok: 15.00
    context_window: 200000
    aliases: [sonnet, claude-sonnet, claude-sonnet-4-5]

  claude-opus-4-6:
    provider: anthropic
    input_per_mtok: 15.00
    output_per_mtok: 75.00
    context_window: 200000
    aliases: [opus, claude-opus]

  gpt-5-mini:
    provider: openai
    input_per_mtok: 0.25
    output_per_mtok: 2.00
    context_window: 128000
    aliases: [gpt5-mini]

  gpt-4.1-nano:
    provider: openai
    input_per_mtok: 0.10
    output_per_mtok: 0.40
    context_window: 128000
    aliases: [gpt4-nano, gpt-4.1-nano]

  gpt-4.1-mini:
    provider: openai
    input_per_mtok: 0.40
    output_per_mtok: 1.60
    context_window: 128000
    aliases: [gpt4-mini, gpt-4.1-mini]

  qwen3:4b-instruct-2507-fp16:
    provider: ollama
    input_per_mtok: 0.00
    output_per_mtok: 0.00
    context_window: 32768
    aliases: [qwen3-4b]
```

**Step 4: Write `model_catalog.py`**

```python
"""ModelCatalog: load model pricing from a YAML file."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    canonical_name: str
    provider: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int
    aliases: list[str] = field(default_factory=list)


class ModelCatalog:
    """Centralized model pricing loaded from a YAML file."""

    def __init__(self, path: str) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._alias_map: dict[str, str] = {}
        self._load(path)

    @classmethod
    def default(cls) -> ModelCatalog:
        """Load the default models.yaml shipped alongside the package."""
        # Walk up from this file to find project root models.yaml
        here = os.path.dirname(os.path.abspath(__file__))
        # core/ -> virtual_context/ -> project root
        root = os.path.dirname(os.path.dirname(here))
        path = os.path.join(root, "models.yaml")
        return cls(path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Model catalog not found at %s — costs will be zero", path)
            return
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        models_raw = raw.get("models", {})
        for name, info in models_raw.items():
            model = ModelInfo(
                canonical_name=name,
                provider=info.get("provider", ""),
                input_per_mtok=float(info.get("input_per_mtok", 0.0)),
                output_per_mtok=float(info.get("output_per_mtok", 0.0)),
                context_window=int(info.get("context_window", 0)),
                aliases=info.get("aliases", []),
            )
            self._models[name] = model
            for alias in model.aliases:
                self._alias_map[alias.lower()] = name

    def _resolve(self, model_name: str) -> ModelInfo | None:
        # 1. Exact match
        if model_name in self._models:
            return self._models[model_name]
        # 2. Alias match
        canonical = self._alias_map.get(model_name.lower())
        if canonical:
            return self._models[canonical]
        # 3. Substring match (e.g. "haiku" in "claude-haiku-4-5-20251001")
        name_lower = model_name.lower()
        for key, model in self._models.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                return model
        return None

    def get_pricing(self, model_name: str) -> tuple[float, float]:
        """Return (input_per_mtok, output_per_mtok) for a model."""
        model = self._resolve(model_name)
        if model is None:
            logger.debug("Unknown model '%s' — returning zero pricing", model_name)
            return (0.0, 0.0)
        return (model.input_per_mtok, model.output_per_mtok)

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a given token count."""
        inp_rate, out_rate = self.get_pricing(model_name)
        return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000

    def get_context_window(self, model_name: str) -> int:
        """Return context window size for a model, or 0 if unknown."""
        model = self._resolve(model_name)
        return model.context_window if model else 0
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_model_catalog.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add models.yaml virtual_context/core/model_catalog.py tests/test_model_catalog.py
git commit -m "feat: add ModelCatalog with YAML-based pricing"
```

---

### Task 2: TelemetryEvent and TelemetryLedger

**Files:**
- Create: `virtual_context/core/telemetry.py`
- Test: `tests/test_telemetry.py`

**Step 1: Write the failing tests**

```python
"""Tests for TelemetryLedger."""

import threading
import time

import pytest

from virtual_context.core.telemetry import TelemetryEvent, TelemetryLedger


@pytest.fixture
def ledger(tmp_path):
    """Ledger with a test ModelCatalog."""
    import yaml
    from virtual_context.core.model_catalog import ModelCatalog

    path = str(tmp_path / "models.yaml")
    with open(path, "w") as f:
        yaml.dump({"models": {
            "haiku": {
                "provider": "anthropic",
                "input_per_mtok": 1.00,
                "output_per_mtok": 5.00,
                "context_window": 200000,
                "aliases": [],
            },
            "gpt-5-mini": {
                "provider": "openai",
                "input_per_mtok": 0.25,
                "output_per_mtok": 2.00,
                "context_window": 128000,
                "aliases": [],
            },
        }}, f)
    catalog = ModelCatalog(path)
    return TelemetryLedger(catalog)


class TestTelemetryLedger:
    def test_empty_ledger(self, ledger):
        total = ledger.total()
        assert total.input_tokens == 0
        assert total.output_tokens == 0
        assert total.cost_usd == 0.0
        assert total.duration_ms == 0.0
        assert total.call_count == 0

    def test_log_event(self, ledger):
        ledger.log("compactor", "haiku", 1000, 200, duration_ms=150.0)
        total = ledger.total()
        assert total.input_tokens == 1000
        assert total.output_tokens == 200
        assert total.call_count == 1
        assert total.duration_ms == 150.0
        expected_cost = (1000 * 1.00 + 200 * 5.00) / 1_000_000
        assert abs(total.cost_usd - expected_cost) < 1e-9

    def test_by_component(self, ledger):
        ledger.log("compactor", "haiku", 1000, 200, duration_ms=100.0)
        ledger.log("tagger", "haiku", 500, 100, duration_ms=80.0)
        ledger.log("compactor", "haiku", 800, 150, duration_ms=120.0)
        breakdown = ledger.by_component()
        assert "compactor" in breakdown
        assert "tagger" in breakdown
        assert breakdown["compactor"].call_count == 2
        assert breakdown["compactor"].input_tokens == 1800
        assert breakdown["tagger"].call_count == 1

    def test_by_model(self, ledger):
        ledger.log("compactor", "haiku", 1000, 200, duration_ms=100.0)
        ledger.log("compactor", "gpt-5-mini", 500, 100, duration_ms=80.0)
        breakdown = ledger.by_model()
        assert "haiku" in breakdown
        assert "gpt-5-mini" in breakdown

    def test_events_returns_copy(self, ledger):
        ledger.log("compactor", "haiku", 100, 50, duration_ms=10.0)
        events = ledger.events()
        assert len(events) == 1
        assert isinstance(events[0], TelemetryEvent)

    def test_turn_id_and_detail(self, ledger):
        ledger.log("compactor", "haiku", 100, 50, duration_ms=10.0,
                   turn_id=3, detail="segment_summarize")
        event = ledger.events()[0]
        assert event.turn_id == 3
        assert event.detail == "segment_summarize"

    def test_reset(self, ledger):
        ledger.log("compactor", "haiku", 1000, 200, duration_ms=100.0)
        ledger.reset()
        assert ledger.total().call_count == 0
        assert ledger.events() == []

    def test_to_dict(self, ledger):
        ledger.log("compactor", "haiku", 1000, 200, duration_ms=100.0)
        d = ledger.to_dict()
        assert "events" in d
        assert "by_component" in d
        assert "by_model" in d
        assert "total" in d
        assert len(d["events"]) == 1

    def test_thread_safety(self, ledger):
        """Multiple threads logging concurrently should not lose events."""
        def log_n(n):
            for _ in range(n):
                ledger.log("compactor", "haiku", 100, 50, duration_ms=1.0)

        threads = [threading.Thread(target=log_n, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ledger.total().call_count == 200
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_telemetry.py -v`
Expected: FAIL — module not found

**Step 3: Write `telemetry.py`**

```python
"""Telemetry: per-call event log for cost, tokens, and timing."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass

from .model_catalog import ModelCatalog


@dataclass(frozen=True)
class TelemetryEvent:
    """A single LLM call record."""
    timestamp: float
    component: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: float
    turn_id: int | None = None
    detail: str = ""


@dataclass
class TelemetryRollup:
    """Aggregated telemetry for a group of events."""
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
    """Thread-safe per-call telemetry collector with rollup views."""

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
    ) -> None:
        """Record a single LLM call."""
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

    def events(self) -> list[TelemetryEvent]:
        with self._lock:
            return list(self._events)

    def total(self) -> TelemetryRollup:
        rollup = TelemetryRollup()
        with self._lock:
            for e in self._events:
                rollup._add(e)
        return rollup

    def by_component(self) -> dict[str, TelemetryRollup]:
        result: dict[str, TelemetryRollup] = {}
        with self._lock:
            for e in self._events:
                if e.component not in result:
                    result[e.component] = TelemetryRollup()
                result[e.component]._add(e)
        return result

    def by_model(self) -> dict[str, TelemetryRollup]:
        result: dict[str, TelemetryRollup] = {}
        with self._lock:
            for e in self._events:
                if e.model not in result:
                    result[e.model] = TelemetryRollup()
                result[e.model]._add(e)
        return result

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def to_dict(self) -> dict:
        with self._lock:
            events_copy = list(self._events)
        total = TelemetryRollup()
        by_comp: dict[str, TelemetryRollup] = {}
        by_mod: dict[str, TelemetryRollup] = {}
        for e in events_copy:
            total._add(e)
            by_comp.setdefault(e.component, TelemetryRollup())._add(e)
            by_mod.setdefault(e.model, TelemetryRollup())._add(e)
        return {
            "events": [asdict(e) for e in events_copy],
            "by_component": {k: asdict(v) for k, v in by_comp.items()},
            "by_model": {k: asdict(v) for k, v in by_mod.items()},
            "total": asdict(total),
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_telemetry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add virtual_context/core/telemetry.py tests/test_telemetry.py
git commit -m "feat: add TelemetryEvent and TelemetryLedger"
```

---

### Task 3: Wire TelemetryLedger into Engine, replace CostTracker

**Files:**
- Modify: `virtual_context/engine.py` — replace `CostTracker` with `TelemetryLedger`
- Modify: `virtual_context/config.py` — replace `cost_tracking` with `telemetry`
- Modify: `virtual_context/types.py` — replace `CostTrackingConfig` / `SessionCostSummary` with `TelemetryConfig`
- Modify: `tests/test_cost_tracker.py` — rewrite as `tests/test_telemetry_integration.py`

**Step 1: Update types.py**

Replace `CostTrackingConfig` and `SessionCostSummary` with:

```python
# In the Cost Tracking section (~line 398-410):
@dataclass
class TelemetryConfig:
    """Configuration for the telemetry system."""
    enabled: bool = False
    models_file: str = "models.yaml"
```

Remove `SessionCostSummary` (replaced by `TelemetryRollup` from telemetry.py).

Update `VirtualContextConfig.cost_tracking: CostTrackingConfig` → `telemetry: TelemetryConfig` (around line 667).

**Step 2: Update config.py**

Replace the cost tracking section (~lines 207-212):
```python
# Telemetry
telemetry_raw = raw.get("telemetry", raw.get("cost_tracking", {}))
telemetry_config = TelemetryConfig(
    enabled=telemetry_raw.get("enabled", False),
    models_file=telemetry_raw.get("models_file", "models.yaml"),
)
```

Update import: `CostTrackingConfig` → `TelemetryConfig`.
Update the config dataclass construction to pass `telemetry=telemetry_config`.

**Step 3: Update engine.py**

Replace imports:
```python
# Remove: from .core.cost_tracker import CostTracker
# Add:
from .core.model_catalog import ModelCatalog
from .core.telemetry import TelemetryLedger
```

Replace `_init_cost_tracker`:
```python
def _init_telemetry(self) -> None:
    models_path = self.config.telemetry.models_file
    if not os.path.isabs(models_path):
        config_dir = os.path.dirname(self._config_path) if self._config_path else "."
        models_path = os.path.join(config_dir, models_path)
    self._model_catalog = ModelCatalog(models_path)
    self._telemetry = TelemetryLedger(self._model_catalog)
```

Replace `get_cost_report() -> SessionCostSummary` with:
```python
def get_telemetry(self) -> TelemetryLedger:
    """Return the session telemetry ledger."""
    return self._telemetry
```

Pass `self._telemetry` instead of `self._cost_tracker` to compactor and tag generator constructors.

**Step 4: Rewrite test file**

Rename `tests/test_cost_tracker.py` → `tests/test_telemetry_integration.py`. Update all tests to use `TelemetryLedger` instead of `CostTracker`. Key changes:
- Fixture creates `ModelCatalog` + `TelemetryLedger` instead of `CostTracker`
- Assertions use `ledger.total()` and `ledger.by_component()` instead of `tracker.get_summary()`
- Compactor/tagger constructors receive `telemetry_ledger=` instead of `cost_tracker=`

**Step 5: Run full test suite**

Run: `pytest tests/ --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' -v`
Expected: PASS (existing tests should still work since we're preserving backward compat in constructors)

**Step 6: Remove old cost_tracker.py**

```bash
git rm virtual_context/core/cost_tracker.py
```

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor: replace CostTracker with TelemetryLedger in engine"
```

---

### Task 4: Instrument Compactor with timing + telemetry

**Files:**
- Modify: `virtual_context/core/compactor.py`
- Test: Update `tests/test_telemetry_integration.py`

**Step 1: Write the failing test**

Add to `tests/test_telemetry_integration.py`:

```python
def test_compactor_logs_telemetry_with_timing(self, ledger):
    llm = MagicMock()
    llm.complete.return_value = '{"summary": "test", "entities": [], "key_decisions": [], "action_items": [], "date_references": [], "refined_tags": ["a"]}'
    llm.last_usage = {"input_tokens": 500, "output_tokens": 120}

    compactor = DomainCompactor(
        llm_provider=llm,
        config=CompactorConfig(),
        model_name="haiku",
        telemetry_ledger=ledger,
    )

    segment = TaggedSegment(
        id="seg-1", tags=["test"], primary_tag="test",
        messages=[Message(role="user", content="hello"),
                  Message(role="assistant", content="hi")],
        turn_count=1,
    )
    compactor._compact_one(segment)

    events = ledger.events()
    assert len(events) == 1
    assert events[0].component == "compactor"
    assert events[0].detail == "segment_summarize"
    assert events[0].duration_ms > 0
    assert events[0].input_tokens == 500
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_telemetry_integration.py::TestTelemetryIntegration::test_compactor_logs_telemetry_with_timing -v`

**Step 3: Update compactor.py**

Replace `from .cost_tracker import CostTracker` with `from .telemetry import TelemetryLedger`.

Change constructor parameter: `cost_tracker: CostTracker | None = None` → `telemetry_ledger: TelemetryLedger | None = None` (keep `cost_tracker` as deprecated alias for backward compat if needed).

Replace `_log_usage`:
```python
def _log_usage(self, detail: str, duration_ms: float) -> None:
    if not self._telemetry_ledger:
        return
    usage = getattr(self.llm, "last_usage", {})
    if not usage:
        return
    input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    if input_tokens or output_tokens:
        self._telemetry_ledger.log(
            "compactor", self.model_name,
            input_tokens, output_tokens,
            duration_ms=duration_ms,
            detail=detail,
        )
```

At each `llm.complete()` call site, wrap with timing:
```python
t0 = time.time()
response = self.llm.complete(system=..., user=..., max_tokens=...)
duration_ms = (time.time() - t0) * 1000
self._log_usage("segment_summarize", duration_ms)
```

Do the same for `compact_tag_summaries` with `detail="tag_rollup"`.

**Step 4: Run tests**

Run: `pytest tests/test_telemetry_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add virtual_context/core/compactor.py tests/test_telemetry_integration.py
git commit -m "feat: instrument compactor with telemetry timing"
```

---

### Task 5: Instrument TagGenerator with timing + telemetry

**Files:**
- Modify: `virtual_context/core/tag_generator.py`
- Test: Update `tests/test_telemetry_integration.py`

**Step 1: Write the failing test**

```python
def test_tagger_logs_telemetry_with_timing(self, ledger):
    llm = MagicMock()
    llm.complete.return_value = '{"tags": ["python"], "primary": "python", "broad": false, "temporal": false, "related_tags": []}'
    llm.last_usage = {"input_tokens": 200, "output_tokens": 50}
    llm.model = "haiku"

    tagger = LLMTagGenerator(
        llm_provider=llm,
        config=TagGeneratorConfig(),
        telemetry_ledger=ledger,
    )
    tagger.generate_tags("test message")

    events = ledger.events()
    assert len(events) == 1
    assert events[0].component == "tagger"
    assert events[0].detail == "tag_generation"
    assert events[0].duration_ms > 0
```

**Step 2: Run test to verify it fails**

**Step 3: Update tag_generator.py**

Same pattern as compactor: replace `CostTracker` import/param with `TelemetryLedger`, wrap `llm.complete()` with timing, call `self._telemetry_ledger.log("tagger", ...)`.

**Step 4: Run tests**

Run: `pytest tests/test_telemetry_integration.py -v`

**Step 5: Commit**

```bash
git add virtual_context/core/tag_generator.py tests/test_telemetry_integration.py
git commit -m "feat: instrument tag generator with telemetry timing"
```

---

### Task 6: Instrument ToolLoop with telemetry

**Files:**
- Modify: `virtual_context/core/tool_loop.py`
- Test: Add test in `tests/test_telemetry_integration.py`

**Step 1: Write the failing test**

```python
def test_tool_loop_logs_telemetry(self, ledger):
    """query_with_tools should log telemetry for each LLM round."""
    # This test uses a mock adapter and engine to verify telemetry logging.
    # The tool loop calls extract_usage on each response — verify those
    # get forwarded to the telemetry ledger.
    events_before = len(ledger.events())
    # (Implementation will require passing ledger to query_with_tools
    #  and verifying events are logged after each adapter call)
    pass  # detailed test body depends on tool_loop signature
```

Note: The tool loop function `query_with_tools` currently takes an `engine` parameter. The ledger can be accessed via `engine._telemetry`. The test should verify that after a tool loop run, telemetry events with `component="tool_loop"` are present.

**Step 2: Update tool_loop.py**

At each `extract_usage` call site (lines 856, 1011, 1117, 1231), add telemetry logging:
```python
input_toks, output_toks = adapter.extract_usage(response)
result.input_tokens += input_toks
result.output_tokens += output_toks
# NEW: telemetry
if hasattr(engine, '_telemetry') and engine._telemetry:
    engine._telemetry.log(
        "tool_loop", model,
        input_toks, output_toks,
        duration_ms=duration_ms,
        detail=f"round_{loop_count}",
    )
```

Wrap each HTTP call with timing (the tool loop already uses `time.time()` for some metrics — extend to capture LLM call duration).

**Step 3: Run tests**

Run: `pytest tests/test_telemetry_integration.py -v`

**Step 4: Commit**

```bash
git add virtual_context/core/tool_loop.py tests/test_telemetry_integration.py
git commit -m "feat: instrument tool loop with telemetry"
```

---

### Task 7: Instrument FactCurator with telemetry

**Files:**
- Modify: `virtual_context/ingest/curator.py`
- Test: Add test in `tests/test_telemetry_integration.py`

**Step 1: Write the failing test**

```python
def test_fact_curator_logs_telemetry(self, ledger):
    from virtual_context.ingest.curator import FactCurator
    from virtual_context.types import CurationConfig, Fact

    llm = MagicMock()
    llm.complete.return_value = "0"
    llm.last_usage = {"input_tokens": 300, "output_tokens": 10}

    curator = FactCurator(
        llm_provider=llm,
        model="haiku",
        config=CurationConfig(),
        telemetry_ledger=ledger,
    )
    facts = [Fact(verb="likes", object="Python")]
    curator.curate(facts, "What language do they prefer?")

    events = ledger.events()
    assert len(events) == 1
    assert events[0].component == "fact_curator"
    assert events[0].detail == "fact_curation"
    assert events[0].duration_ms > 0
```

**Step 2: Update curator.py**

Add `telemetry_ledger` parameter to `__init__`. Wrap `llm.complete()` with timing and log to ledger.

**Step 3: Run tests and commit**

```bash
git add virtual_context/ingest/curator.py tests/test_telemetry_integration.py
git commit -m "feat: instrument fact curator with telemetry"
```

---

### Task 8: CLI `telemetry` command

**Files:**
- Modify: `virtual_context/cli/main.py`
- Test: `tests/test_cli.py` (add test for new command)

**Step 1: Replace `cmd_cost_report` with `cmd_telemetry`**

```python
def cmd_telemetry(args):
    """Show session telemetry report."""
    from ..engine import VirtualContextEngine

    engine = VirtualContextEngine(config_path=args.config)
    ledger = engine.get_telemetry()

    if args.json:
        import json
        print(json.dumps(ledger.to_dict(), indent=2))
        return

    total = ledger.total()
    by_comp = ledger.by_component()

    print("Session Telemetry Report")
    print("=" * 72)
    print(f"{'Component':<18} {'Calls':>5}   {'Input Tok':>11}   {'Output Tok':>11}   {'Cost (USD)':>10}   {'Time':>6}")
    print("-" * 72)
    for comp, rollup in sorted(by_comp.items()):
        t = f"{rollup.duration_ms / 1000:.1f}s" if rollup.duration_ms else "—"
        print(f"{comp:<18} {rollup.call_count:>5}   {rollup.input_tokens:>11,}   {rollup.output_tokens:>11,}   ${rollup.cost_usd:>9.4f}   {t:>6}")
    print("-" * 72)
    t_total = f"{total.duration_ms / 1000:.1f}s" if total.duration_ms else "—"
    print(f"{'TOTAL':<18} {total.call_count:>5}   {total.input_tokens:>11,}   {total.output_tokens:>11,}   ${total.cost_usd:>9.4f}   {t_total:>6}")

    if args.verbose:
        print("\nEvent Log:")
        print("-" * 72)
        for e in ledger.events():
            print(f"  [{e.component}] {e.model} | in={e.input_tokens} out={e.output_tokens} | ${e.cost_usd:.6f} | {e.duration_ms:.0f}ms | {e.detail}")
```

**Step 2: Update argument parser**

Replace the `cost-report` subcommand with `telemetry`:
```python
p_telem = subparsers.add_parser("telemetry", help="Show session telemetry report")
p_telem.add_argument("--verbose", "-v", action="store_true")
p_telem.add_argument("--json", action="store_true")
p_telem.set_defaults(func=cmd_telemetry)
```

Keep `cost-report` as a hidden alias pointing to `cmd_telemetry` for backward compat.

**Step 3: Run tests and commit**

```bash
git add virtual_context/cli/main.py
git commit -m "feat: add CLI telemetry command replacing cost-report"
```

---

### Task 9: Proxy dashboard telemetry integration

**Files:**
- Modify: `virtual_context/proxy/metrics.py`
- Modify: `virtual_context/proxy/server.py` (wire ledger to metrics)
- Modify: proxy dashboard HTML template (add telemetry panel)

**Step 1: Add telemetry to ProxyMetrics.snapshot()**

Add a `telemetry_ledger` parameter to `ProxyMetrics.__init__`:
```python
def __init__(self, context_window: int = 120_000, telemetry_ledger=None) -> None:
    ...
    self._telemetry_ledger = telemetry_ledger
```

In `snapshot()`, add telemetry data:
```python
# At end of snapshot dict:
telemetry = {}
if self._telemetry_ledger:
    telemetry = self._telemetry_ledger.to_dict()
    # Remove raw events for snapshot (too large)
    telemetry.pop("events", None)
return {
    ...existing keys...,
    "telemetry": telemetry,
}
```

**Step 2: Wire in proxy/server.py**

When creating `ProxyMetrics`, pass the engine's telemetry ledger:
```python
metrics = ProxyMetrics(
    context_window=...,
    telemetry_ledger=engine._telemetry,
)
```

**Step 3: Update dashboard HTML**

Add a "Telemetry" section to the dashboard that renders the component breakdown table. The dashboard already uses SSE events — add telemetry data to the snapshot payload so the JS can render it.

**Step 4: Commit**

```bash
git add virtual_context/proxy/metrics.py virtual_context/proxy/server.py virtual_context/proxy/dashboard.html
git commit -m "feat: add telemetry panel to proxy dashboard"
```

---

### Task 10: Update benchmark cost tracking

**Files:**
- Modify: `benchmarks/longmemeval/cost.py` — use `ModelCatalog` instead of hardcoded `PRICING`
- Modify: `benchmarks/longmemeval/vc_runner.py` — read from engine telemetry
- Modify: `benchmarks/longmemeval/run.py` — write telemetry JSON report

**Step 1: Update cost.py**

Replace `PRICING` dict with `ModelCatalog.default()` lookups. Keep `BudgetTracker` interface but back it with `ModelCatalog.calculate_cost()`.

**Step 2: Update vc_runner.py**

Where it currently calls `engine.get_cost_report()` (line ~733), switch to `engine.get_telemetry().total()`.

**Step 3: Write telemetry report**

In `run.py`, after each question, dump `engine.get_telemetry().to_dict()` to `telemetry_<run_label>.json`.

**Step 4: Run a quick benchmark smoke test**

Run: `python -m benchmarks.longmemeval --questions 1 --budget 5.0`
Expected: runs successfully, telemetry report written

**Step 5: Commit**

```bash
git add benchmarks/longmemeval/cost.py benchmarks/longmemeval/vc_runner.py benchmarks/longmemeval/run.py
git commit -m "feat: benchmark harness uses ModelCatalog and writes telemetry reports"
```

---

### Task 11: Cleanup and final verification

**Files:**
- Delete: `virtual_context/core/cost_tracker.py` (if not already removed in Task 3)
- Modify: Any remaining references to old types

**Step 1: Search for any remaining old references**

```bash
grep -r "CostTracker\|SessionCostSummary\|CostTrackingConfig\|cost_tracker\|log_compaction\|log_tag_generation\|log_retrieval\|get_cost_report" virtual_context/ tests/ --include="*.py"
```

Fix any remaining references.

**Step 2: Run full test suite**

Run: `pytest tests/ --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' -v`
Expected: all ~960 tests PASS

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove all CostTracker references, cleanup complete"
```
