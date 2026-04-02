# Architecture

Virtual-context is a transparent context virtualization layer that sits between LLM clients and upstream providers. It compresses, indexes, and pages conversation history so that models operate on relevant summaries rather than raw token streams.

## Pipeline

Every request flows through a fixed sequence:

```
Client request
    |
    v
Format detection (Anthropic / OpenAI Chat / OpenAI Responses / Gemini)
    |
    v
Envelope stripping (OpenClaw channel metadata removed)
    |
    v
Wait for previous on_turn_complete
    |
    v
History ingestion (first request only: bootstrap TurnTagIndex)
    |
    v
on_message_inbound
    |-- Inbound tagging (embedding tagger + LLM tagger in parallel)
    |-- Retrieval (3-signal RRF: IDF tag overlap + BM25 keyword + embedding cosine)
    |-- Assembly (budget-aware context block construction)
    |
    v
Inject <virtual-context> block into system prompt
    |
    v
Forward to upstream provider
    |
    v
Stream/return response to client
    |
    v
on_turn_complete (background thread)
    |-- Response tagging (full turn: user + assistant)
    |-- TurnTagIndex update
    |-- Compaction check (soft/hard thresholds)
    |-- Tag summary (re)build
    |-- Fact extraction and supersession
```

## Component Map

The codebase is organized into three layers:

### Core (`virtual_context/core/`)

| Module | Responsibility |
|--------|---------------|
| `engine.py` | Top-level orchestrator. Owns the event lifecycle: `on_message_inbound`, `on_turn_complete`, `ingest_history` |
| `compactor.py` | Two-tier compaction (summary and deep). Selects segments, calls the summarization LLM, writes compressed segments back to storage |
| `segmenter.py` | Splits raw conversation turns into semantic segments with tag assignments |
| `tagging_pipeline.py` | Two-tagger architecture: parallel inbound embedding tagger + LLM response tagger. Context bleed gate on topic shifts |
| `assembler.py` | Budget-aware context block construction. Tag rules priority pass, then greedy fill pass for remaining budget |
| `monitor.py` | Tracks context window fill level, triggers compaction when soft/hard thresholds are crossed |
| `tool_loop.py` | Tool catalogue, execution dispatch for vc_* tools, SSE parsing, continuation rounds, anti-repetition tracking |
| `tool_query.py` | ToolQueryRunner: handles individual vc_* tool calls, manages presented segment deduplication |
| `retrieval.py` | 3-signal ranked retrieval with gravity/hub dampening. Configurable strategy per query type (broad, temporal, default) |
| `fact_extractor.py` | Structured fact extraction with supersession detection. Facts have subject/verb/object, status, dates, locations |
| `telemetry.py` | TelemetryEvent, TelemetryRollup, TelemetryLedger for operational metrics |
| `temporal_resolver.py` | Time-bounded recall: parses relative dates ("last week", "in March") to absolute date ranges |

### Proxy (`virtual_context/proxy/`)

| Module | Responsibility |
|--------|---------------|
| `handlers.py` | HTTP request handler. Format detection, request enrichment, streaming SSE forwarding, paging path with tool interception |
| `helpers.py` | Pure functions: format detection, envelope stripping, context injection, payload construction |
| `provider_adapters.py` | Adapter layer for Anthropic, OpenAI Chat, OpenAI Responses, and Gemini API formats |
| `metrics.py` | Thread-safe event collector with ring buffer, snapshot aggregation, cursor-based SSE streaming |
| `server.py` | ASGI application setup, route registration, dashboard serving |
| `dashboard_html.py` | Self-contained single-page dashboard (all CSS/JS inlined) |

### Infrastructure

| Module | Responsibility |
|--------|---------------|
| `token_counter.py` | Three-mode token counting: `anthropic` (exact), `tiktoken` (fast), `estimate` (len/4 fallback). Image-aware via dimension-based costing |
| `config.py` | YAML config loading with validation, preset system, multi-instance support |
| `types.py` | Dataclasses for the entire system: `VCConfig`, `TagEntry`, `TurnTagEntry`, `Segment`, `Fact`, retrieval/assembly configs |
| `storage/sqlite.py` | SQLite storage backend (default). Segments, facts, tag summaries, session state |
| `storage/postgres.py` | PostgreSQL backend for multi-worker deployments |
| `storage/neo4j_store.py` | Neo4j/FalkorDB backend for graph-based fact queries |
| `cli/` | Command-line interface: `proxy`, `daemon`, `onboard`, `init`, `config`, `presets`, `tui` |
| `tui/` | Terminal UI with live panels for context state, tags, compaction status |

## Storage Backends

All backends implement the same `Store` protocol:

- **SQLite** (default): Single-file, zero-config. Suitable for single-user and development.
- **PostgreSQL**: Multi-worker safe. Used when running multiple proxy instances against the same conversation store.
- **Neo4j / FalkorDB**: Graph-backed storage for fact relationships and traversal queries.

Storage is configured in `virtual-context.yaml`:

```yaml
storage:
  backend: "sqlite"   # "sqlite", "postgres", or "neo4j"
  sqlite:
    path: ".virtualcontext/store.db"
  postgres:
    dsn: "postgresql://user:pass@host:5432/vc"
  neo4j:
    uri: "bolt://localhost:7687"
```

## Provider Adapters

The proxy supports four API formats with auto-detection:

| Format | Detection Signal | Context Injection Point |
|--------|-----------------|------------------------|
| **Anthropic** | `"system"` field or model name starts with `"claude"` | `system` field (string or content blocks) |
| **OpenAI Chat** | `/v1/chat/completions` path | `messages[0]` with `role: "system"` |
| **OpenAI Responses** | `/v1/responses` path | `instructions` field |
| **Gemini** | `/v1beta/models` path pattern | `system_instruction` field |

Detection is automatic. No configuration needed.

## Session Management

Sessions track conversation continuity across requests. The proxy uses Redis-backed session state for multi-worker consistency (single-worker falls back to in-memory state).

Session identity is derived from:
1. Conversation ID embedded in HTML comments within the context block
2. API key + model combination as fallback
3. Explicit session headers when available

The `ProxyState` object holds per-session engine instances, ensuring each conversation gets its own tagging index, compaction watermark, and retrieval state.

## Threading Model

- The main request path is synchronous within the async ASGI handler
- `on_turn_complete` runs in a `ThreadPoolExecutor(max_workers=1)` background thread
- Each new request calls `wait_for_complete()` to block until the previous turn's background work finishes
- History ingestion uses double-checked locking for one-time bootstrap
- Compaction and tag summary rebuilds happen in the background thread, never blocking the response path
