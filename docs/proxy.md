# HTTP Proxy & Live Dashboard

The virtual-context proxy sits between any LLM client (OpenClaw, Cursor, custom apps) and an upstream provider (Anthropic, OpenAI, Ollama), transparently enriching every request with retrieved context and capturing responses to build long-term memory.

## Quick Start

```bash
# Install with bridge extras (adds uvicorn + fastapi)
pip install virtual-context[bridge]

# Start the proxy
ANTHROPIC_API_KEY=sk-... virtual-context -c virtual-context-proxy.yaml proxy \
  --upstream https://api.anthropic.com \
  --port 8100

# Point your LLM client at http://localhost:8100 instead of the real API
# Open the dashboard at http://localhost:8100/dashboard
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--upstream` | *(required)* | Upstream provider URL (e.g. `https://api.anthropic.com`, `http://localhost:11434/v1`) |
| `--port` | `8100` | Local port to listen on |
| `--host` | `0.0.0.0` | Bind address |
| `-c` / `--config` | auto-discover | Path to `virtual-context.yaml` config file |

---

## Architecture

```
LLM Client (OpenClaw, Cursor, etc.)
    │
    │  POST /v1/messages  (or /v1/chat/completions)
    ▼
┌─────────────────────────────────────────────┐
│  virtual-context proxy  (localhost:8100)     │
│                                             │
│  1. Detect API format (Anthropic / OpenAI)  │
│  2. Extract user message                    │
│  3. Strip OpenClaw envelope metadata        │
│  4. Wait for previous on_turn_complete      │
│  5. Ingest history (first request only)     │
│  6. Run on_message_inbound (tag + retrieve) │
│  7. Inject <virtual-context> block          │
│  8. Forward to upstream                     │
│  9. Stream response back to client          │
│ 10. Capture assistant text                  │
│ 11. Fire on_turn_complete (background)      │
└─────────────────────────────────────────────┘
    │
    │  Enriched request (original + VC context)
    ▼
Upstream Provider (Anthropic, OpenAI, Ollama)
```

### Two API Formats

The proxy auto-detects whether incoming requests use the Anthropic or OpenAI format:

- **Anthropic**: detected by `"system"` field or model name starting with `"claude"`. Context injected into the `system` field.
- **OpenAI**: default. Context injected as a system message at `messages[0]`.

Detection happens in `_detect_api_format()` — no configuration needed.

### Request Lifecycle

1. **Parse**: Read the POST body, detect API format, extract the last user message text.
2. **Strip envelope**: Remove OpenClaw channel metadata (Telegram headers, message IDs, `[vc:prompt]` markers, `[vc:user]` wrappers, `System:` event lines).
3. **Wait**: Block until the previous turn's `on_turn_complete` finishes (runs in a background thread).
4. **Ingest history** (first request only): Extract all prior user/assistant message pairs from the request body, tag them via `engine.ingest_history()`, and bootstrap the TurnTagIndex. This gives the engine full topic awareness from the client's conversation history without requiring a separate initialization step.
5. **Enrich**: Call `engine.on_message_inbound(message, history)` which tags the query, retrieves matching stored summaries, and assembles a context block.
6. **Inject**: Deep-copy the request body and prepend the `<virtual-context>` block to the system prompt (Anthropic) or system message (OpenAI).
7. **Forward**: Send the enriched request to the upstream provider. Streaming requests use raw byte forwarding to preserve exact SSE framing (critical for the Node.js Anthropic SDK).
8. **Capture**: Accumulate the assistant's text from streaming deltas or the non-streaming response body.
9. **Complete**: Append the assistant message to the conversation history and fire `on_turn_complete()` in a background thread. This tags the full turn, updates the TurnTagIndex, checks compaction thresholds, and builds tag summaries.

### Streaming Support

The proxy forwards raw SSE bytes from the upstream to preserve exact framing. A side-channel parser accumulates text deltas for `on_turn_complete`. Non-2xx responses (rate limits, overloads) are returned as JSON errors instead of broken SSE streams.

### OpenClaw Envelope Stripping

Messages from OpenClaw (Telegram, WhatsApp, etc.) contain channel metadata that pollutes tagging. The proxy strips five patterns in order:

1. `[vc:prompt]\n` — marker from the virtual-context-tagger OpenClaw plugin
2. `[vc:user]...[/vc:user]` — backward-compatible wrapper (inner content returned directly)
3. `System: [TIMESTAMP] event` lines — OpenClaw system events
4. `[ChannelName ... id:NNN ...] ` — channel header (Telegram, WhatsApp, etc.)
5. `[message_id: NNN]` — message footer

### Thread Safety

The proxy uses the same threading pattern as the headless CLI runner:

- `ProxyState` holds a `ThreadPoolExecutor(max_workers=1)` for background `on_turn_complete` work
- `wait_for_complete()` blocks until the pending future resolves — called at the start of each new request
- History ingestion uses double-checked locking (`_ingestion_lock`) to ensure one-time bootstrap

---

## Dashboard

The dashboard is a self-contained single-page HTML application served at `/dashboard`. No external dependencies — all CSS and JavaScript are inlined. It connects via Server-Sent Events (SSE) for real-time updates.

### Panels

#### Overview (top stat cards)

Five real-time counters:

| Counter | Description |
|---------|-------------|
| **Uptime** | Time since proxy started |
| **Requests** | Total LLM requests intercepted and enriched |
| **Turns** | Completed conversation turns (user + assistant pairs) |
| **Compactions** | Number of compaction events fired |
| **Freed** | Total tokens reclaimed by compaction |

#### Replay

Run a stress test by replaying a file of user prompts through the full pipeline. Each line in the file becomes one conversation turn processed through tagging, retrieval, assembly, LLM call, and `on_turn_complete`.

- **File**: Path to a text file with one prompt per line (relative to the proxy's working directory)
- **Start/Stop**: Begin or halt the replay. Progress bar and turn counter update live.
- The replay worker calls the engine's own configured LLM provider (Anthropic or OpenAI-compatible) with `"(Answer in 2 lines.)"` appended for brevity
- All metrics (request log, compaction events, cost savings) update live during replay
- Uses `filter_history()` for tag-based turn selection, matching the headless CLI behavior

**Endpoints**: `POST /dashboard/replay/start`, `POST /dashboard/replay/stop`, `GET /dashboard/replay/status`

#### Memory

A visual progress bar showing how much of the conversation history has been compacted.

- **Bar color**: Green (healthy) → Yellow (>70%) → Red (>85%)
- **Status line**: Shows compaction watermark, total message count, and context window size
- **Compact Now** button: Trigger manual compaction regardless of thresholds (`POST /dashboard/compact`)

#### Pipeline (avg)

Average latency breakdown across all requests:

| Metric | Description |
|--------|-------------|
| **wait** | Time waiting for previous `on_turn_complete` to finish |
| **inbound** | Time for `on_message_inbound`: tagging + retrieval + assembly |
| **context** | Average tokens of retrieved summaries injected per request |

#### Cost Savings

Compares actual input tokens against a simulated naive baseline:

| Metric | Description |
|--------|-------------|
| **Tokens Freed** | Total tokens reclaimed by compaction |
| **Summary Compression** | Ratio of summary tokens to original tokens |
| **Session Efficiency** | Percentage saved vs. the naive baseline |
| **Context Injected** | Total retrieved summary tokens across all requests |
| **Avg Context / Request** | Average injected context per turn |

**Estimated dollar savings** are shown for three pricing tiers:

| Tier | $/MTok | Description |
|------|--------|-------------|
| Haiku-class | $0.25 | Fast, cheap models |
| Sonnet-class | $3.00 | Mid-tier models |
| Opus-class | $15.00 | Frontier models |

**Baseline simulation**: The baseline models a naive system that sends full conversation history each turn. When history exceeds the context window, it simulates compaction at 30% ratio (3.3x compression). The baseline gets full credit for its own compaction — VC savings reflect only the incremental benefit of selective retrieval and tag-based filtering.

#### Sessions

Lists all sessions that have stored segments (from compaction). Each session card shows:

- Session ID (with `CURRENT` badge for the active session)
- Segment count, full tokens, compression ratio, tokens freed
- Tag cloud (up to 8 tags + overflow count)
- Time range (oldest → newest segment)
- Compaction model used
- **Delete** button for past sessions (removes all stored segments)

**Endpoints**: `GET /dashboard/sessions`, `DELETE /dashboard/sessions/{session_id}`

#### Active Tags

Shows the current conversation's "working set" of topics — the union of tags from the most recent N turns (controlled by `active_tag_lookback`). Active tags are **skipped** during retrieval because their content is already present in raw conversation history.

Also shows the total count of distinct tags in the store.

#### Request Log

Chronological table of every request processed (newest first, max 200 rows). Columns:

| Column | Description |
|--------|-------------|
| **T#** | Turn number (0-indexed) |
| **Inbound Tags** | Tags assigned to the user message during `on_message_inbound`. Shows `BROAD` (purple) or `TEMPORAL` (orange) badges for detected query types. |
| **Response Tags** | Tags assigned to the full turn (user + assistant) during `on_turn_complete`. Primary tag shown in bold. Populated asynchronously after the background thread completes. |
| **Message** | Preview of the user message (first 50 characters) |
| **Payload** | Turn count included in the LLM payload (`filtered/total` during replay, just `total` for proxy) |
| **Tokens** | Estimated total input tokens sent to the upstream (system + messages) |
| **Base** | Baseline tokens a naive system would have sent for this turn. Computed after `on_turn_complete` finishes. |
| **Injected** | Tokens of retrieved `<virtual-context>` summaries injected |
| **VC** | Virtual-context overhead (wait + inbound) in ms |
| **LLM** | Upstream API round-trip time |
| **Total** | End-to-end time (VC + LLM) |
| **Inspect** | Opens the Request Inspector modal |

**Ingested history rows** appear dimmed (50% opacity) with a `HISTORY` badge, showing tags from bootstrapped historical turns. These do not have timing or token data.

#### Compaction Events

History of compaction operations (newest first). Each entry shows:

- Turn that triggered compaction
- Number of segments compacted
- Tokens freed
- Tags covered by the compacted segments
- Tag summaries (re)built
- Compaction watermark (message index)

### Modals

#### Request Inspector

Click **inspect** on any request log row to open a full payload view. Shows:

- **Inbound Tags** and **Response Tags** side by side
- **System Prompt**: Full text with estimated token count (truncated at 5,000 chars in the UI)
- **Messages**: Every message in the raw request body, color-coded by role:
  - Green: user
  - Blue: assistant
  - Gold: system
- Content blocks within messages are labeled by type: `text`, `tool_use`, `tool_result`, `thinking`
- **Save JSON** button downloads the full captured request as `request-turn-{N}.json`

The inspector captures the **original** request body (before VC enrichment), stored in a ring buffer of the last 50 requests.

**Endpoints**: `GET /dashboard/requests` (summaries), `GET /dashboard/requests/{turn}` (full payload)

#### Settings

Click the gear icon to open runtime-adjustable engine settings. Changes apply to the current session only (not persisted to the YAML config file).

**Read-only** (set at startup):
- Context Window, Tagger Type/Model, Summarizer Model, Storage Backend

**Adjustable sections**:

| Section | Settings |
|---------|----------|
| **Compaction** | Soft/hard thresholds (sliders), protected turns, min/max summary tokens |
| **Tagging** | Broad heuristic toggle, temporal heuristic toggle |
| **Retrieval** | Active tag lookback, anchorless lookback, max results, budget fraction (slider), include related toggle |
| **Assembly** | Tag context max tokens, recent turns kept, context hint toggle, hint max tokens |
| **Summarization** | Temperature (slider) |

Cross-field validation prevents invalid combinations (soft >= hard, min > max). Every setting has a `?` help button with detailed explanations.

**Endpoints**: `GET /dashboard/settings`, `PUT /dashboard/settings`

### Export Session

Click **Export** in the header to download a full session snapshot as `vc-export-YYYY-MM-DDTHH-MM-SS.json`. The export includes:

- All metrics (requests, responses, compactions, turn completes)
- Ingested history turns with tags and message previews
- Full TurnTagIndex (per-turn tag data for every turn in the session)
- Store tags (all distinct tags with stored segments)
- Config summary (session ID, context window, thresholds, tagger/summarizer/storage)
- Conversation turn count and compaction watermark

This is designed for offline analysis and debugging — paste the JSON into a Claude conversation to get tag quality analysis, retrieval accuracy assessment, and compaction behavior review.

**Endpoint**: `GET /dashboard/export`

### Shutdown

Click **Shutdown** in the header to gracefully stop the proxy server. Sends `SIGINT` to the process, which triggers the same shutdown sequence as Ctrl+C: closes the HTTP client, drains the thread pool, and exits.

**Endpoint**: `POST /dashboard/shutdown`

---

## SSE Event Stream

The dashboard connects to `GET /dashboard/events` for real-time updates. The stream uses Server-Sent Events:

1. **Initial snapshot**: Full aggregate stats + all recent events (sent immediately on connection)
2. **Incremental events**: New events streamed every second via a cursor-based mechanism (`_seq` field)

### Event Types

| Type | When | Key Fields |
|------|------|------------|
| `request` | New request intercepted | `turn`, `tags`, `broad`, `temporal`, `context_tokens`, `input_tokens`, `system_tokens`, `wait_ms`, `inbound_ms` |
| `response` | Upstream response complete | `turn`, `upstream_ms`, `total_ms`, `streaming`, `error` |
| `turn_complete` | Background tagging done | `turn`, `tags`, `primary_tag`, `active_tags`, `store_tag_count`, `turn_pair_tokens`, `complete_ms` |
| `compaction` | Compaction event | `turn`, `segments`, `tokens_freed`, `original_tokens`, `summary_tokens`, `tags`, `compacted_through` |
| `history_ingestion` | History bootstrap done | `turns_ingested`, `pairs_received`, `elapsed_ms`, `session_id` |
| `ingested_turn` | Per-turn from ingestion | `turn`, `tags`, `primary_tag`, `message_preview` |
| `replay_progress` | Replay turn done | `turn`, `total`, `prompt_preview`, `elapsed_ms` |
| `replay_done` | Replay finished | `turns_completed`, `total`, `status`, `error` |

---

## Configuration Reference

See `virtual-context-proxy.yaml` for a complete annotated proxy config. Key sections:

```yaml
version: "0.2"

context_window: 120000
token_counter: "estimate"        # "estimate" (len/4) or "tiktoken"

tag_generator:
  type: "llm"                    # "llm", "keyword", or "embedding"
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  max_tags: 10
  min_tags: 5
  broad_patterns: []             # disable regex (Haiku handles natively)
  temporal_patterns: []

compaction:
  soft_threshold: 0.70           # start compacting at 70% fill
  hard_threshold: 0.85           # force compaction at 85% fill
  protected_recent_turns: 6

summarization:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  temperature: 0.3

storage:
  backend: "sqlite"

retrieval:
  active_tag_lookback: 4         # recent turns whose tags are skipped
  strategy_config:
    default:
      max_results: 10
      max_budget_fraction: 0.25  # 25% of window for retrieved context
      include_related: true

assembly:
  context_hint_enabled: true     # inject topic list after compaction
```

---

## Proxy Metrics

`ProxyMetrics` (`proxy/metrics.py`) is the thread-safe event collector that powers the dashboard. Key features:

- **Thread-safe**: `record()` uses a lock — safe to call from the background `on_turn_complete` thread
- **Sequenced events**: Every event gets a monotonic `_seq` number and ISO timestamp
- **Ring buffer**: Last 50 raw request bodies captured for the inspector (`capture_request()`)
- **Snapshot aggregation**: `snapshot()` computes all aggregate stats (totals, averages, baseline simulation) from the raw event list
- **Cursor-based streaming**: `events_since(seq)` returns only new events for the SSE stream

### Baseline Simulation

The baseline simulation models what a naive system would spend on input tokens. For each `turn_complete` event:

1. Add `turn_pair_tokens` to cumulative history
2. If history exceeds `context_window`, simulate compaction: protect the last 4 turns, compress the rest at 30% ratio
3. Per-turn baseline = `system_tokens + baseline_history_tokens`
4. Accumulate across all turns for the session total

This runs both server-side (in `snapshot()`) and client-side (in the dashboard JS) for consistency.
