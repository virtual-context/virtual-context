# Telemetry System Design

**Date:** 2026-03-04
**Status:** Approved
**Scope:** Model catalog, per-call telemetry (cost + tokens + timing), component instrumentation, reporting (dashboard, CLI, file)

## Problem

Cost tracking today is fragmented:
- Pricing hardcoded in two places (`config.yaml` per-1k rates, `benchmarks/longmemeval/cost.py` per-MTok rates) with different formats
- `CostTracker` only covers compactor and tagger — ToolLoop, FactCurator, Proxy don't log costs
- No timing instrumentation
- No per-call event log — only session-level totals
- Benchmark `BudgetTracker` is completely separate from engine `CostTracker`
- Proxy dashboard tracks tokens/latency but not cost

## Design

### 1. Model Catalog

**File:** `models.yaml` in the config directory (alongside `virtual-context.yaml`)

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
    aliases: [sonnet, claude-sonnet]

  gpt-5-mini:
    provider: openai
    input_per_mtok: 0.25
    output_per_mtok: 2.00
    context_window: 128000
    aliases: [gpt5-mini]

  qwen3:4b-instruct-2507-fp16:
    provider: ollama
    input_per_mtok: 0.00
    output_per_mtok: 0.00
    context_window: 32768
    aliases: [qwen3-4b]
```

**`ModelCatalog` class:**
- Loads `models.yaml`, builds alias→canonical lookup
- `get_pricing(model_name) -> (input_per_mtok, output_per_mtok)`
- `calculate_cost(model_name, input_tokens, output_tokens) -> float`
- `get_context_window(model_name) -> int`
- Alias resolution: exact match → alias match → substring match → zero cost with warning

**Replaces:** `cost_tracking.pricing` config section and benchmark `PRICING` dict.

### 2. TelemetryEvent & TelemetryLedger

**Source file:** `virtual_context/core/telemetry.py` (replaces `core/cost_tracker.py`)

**TelemetryEvent dataclass:**
```python
@dataclass(frozen=True)
class TelemetryEvent:
    timestamp: float          # time.time()
    component: str            # "compactor", "tagger", "tool_loop", "fact_curator", "proxy_upstream"
    model: str                # canonical model name
    input_tokens: int
    output_tokens: int
    cost_usd: float           # calculated via ModelCatalog
    duration_ms: float        # wall-clock time for this LLM call
    turn_id: int | None       # which user turn triggered this, if known
    detail: str = ""          # e.g. "segment_summarize", "tag_rollup"
```

**TelemetryLedger class:**
- Thread-safe (lock-based, matches sync-first + background-thread pattern)
- `log(component, model, input_tokens, output_tokens, duration_ms, turn_id=None, detail="")` — creates event, calculates cost via catalog, appends
- Rollup methods:
  - `by_component() -> dict[str, ComponentTelemetry]` — per-component totals (tokens, cost, timing, call count)
  - `by_model() -> dict[str, ModelTelemetry]` — per-model totals
  - `total() -> TotalTelemetry` — grand totals
  - `events() -> list[TelemetryEvent]` — raw event log
- `to_dict() -> dict` — serializable for JSON export
- `reset()` — clear for new session

**Replaces:** `CostTracker` and `SessionCostSummary`.

### 3. Instrumentation — LLM Call Sites

Each component gets a `TelemetryLedger` reference via constructor injection.

| Component | Call site | `component` | `detail` |
|-----------|----------|-------------|----------|
| Compactor | `_compact_one()` | `"compactor"` | `"segment_summarize"` |
| Compactor | `compact_tag_summaries()` | `"compactor"` | `"tag_rollup"` |
| TagGenerator | `generate_tags()` | `"tagger"` | `"tag_generation"` |
| ToolLoop | `query_with_tools()` iteration | `"tool_loop"` | `"tool_call_round_{n}"` |
| FactCurator | `curate()` | `"fact_curator"` | `"fact_curation"` |
| Proxy | upstream reader call | `"proxy_upstream"` | `"reader_call"` |

**Pattern at each site:**
```python
t0 = time.time()
result = llm.complete(system, user)
duration_ms = (time.time() - t0) * 1000
usage = llm.last_usage
self._ledger.log("compactor", self.model_name,
                 usage.get("input_tokens", 0),
                 usage.get("output_tokens", 0),
                 duration_ms=duration_ms,
                 detail="segment_summarize")
```

**Engine wiring:** `Engine.__init__` creates `ModelCatalog` + `TelemetryLedger`, passes ledger to all components. Benchmark `BudgetTracker` retired — `vc_runner.py` reads from engine ledger.

### 4. Reporting

#### Proxy Dashboard
- `ProxyMetrics` gets a `TelemetryLedger` reference
- **Session summary panel:** total cost, tokens, time — broken down by component
- **Per-request rows:** each request shows its telemetry (filtered by turn_id)
- Data flows through existing `get_dashboard_data()` — adds a `telemetry` key

#### CLI Command
- `virtual-context telemetry` — formatted table with per-component breakdown
- `--verbose` — includes per-call event log
- `--json` — raw JSON output

**Example output:**
```
Component        Calls   Input Tok   Output Tok   Cost (USD)   Time
──────────────────────────────────────────────────────────────────────
compactor            4      42,100        8,300       $0.084    3.2s
tagger               3      15,200        2,100       $0.026    1.8s
tool_loop            5      31,000        4,500       $0.108    6.1s
fact_curator         1       3,200          800       $0.007    0.9s
proxy_upstream       1      28,000        6,200       $0.177    2.4s
──────────────────────────────────────────────────────────────────────
TOTAL               14     119,500       21,900       $0.402   14.4s
```

#### File Reports
- `TelemetryLedger.to_dict()` → JSON
- Benchmark harness writes `telemetry_<run_label>.json` alongside autopsy files
- Proxy can optionally dump on session end

### 5. Config

```yaml
telemetry:
  enabled: true
  models_file: models.yaml   # relative to config dir, or absolute path
```

Replaces current `cost_tracking` section.

### 6. Budget Enforcement

Report only — no blocking. Budget enforcement can be layered on later if needed.

## What Gets Removed

- `virtual_context/core/cost_tracker.py` — replaced by `telemetry.py`
- `CostTrackingConfig.pricing` dict — replaced by `ModelCatalog`
- `SessionCostSummary` type — replaced by `TotalTelemetry`
- `benchmarks/longmemeval/cost.py` `PRICING` dict — uses `ModelCatalog`
- `log_compaction()`, `log_tag_generation()`, `log_retrieval()` methods — replaced by `ledger.log()`
