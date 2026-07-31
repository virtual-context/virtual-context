# Configuration

Virtual-context is configured via a YAML file, typically `virtual-context.yaml` or `~/.virtualcontext/config.yaml`. The file is discovered automatically or specified with `-c`.

## Minimal Config

```yaml
version: "0.2"

context_window: 120000

tag_generator:
  type: "llm"
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"

summarization:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"

storage:
  backend: "sqlite"
```

## Full Reference

### Top-Level

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | string | `"0.2"` | Config schema version |
| `context_window` | int | `120000` | Total token budget for the context window |
| `token_counter` | string | `"estimate"` | Token counting mode: `"anthropic"`, `"tiktoken"`, or `"estimate"` |
| `storage_root` | string | `".virtualcontext"` | Root directory for data files |

### Tag Generator

Controls how conversation turns are tagged for indexing.

```yaml
tag_generator:
  type: "llm"                       # "llm" or "keyword"
  provider: "anthropic"             # "anthropic", "openai", "gemini", "local", or "openrouter"
  model: "claude-haiku-4-5-20251001"
  max_tags: 10                      # maximum tags per turn
  min_tags: 5                       # minimum tags to assign
```

**Type options**:

- `llm`: LLM-based tagging of completed turns. Best quality. Uses the configured provider and model.
- `keyword`: Keyword extraction with a configured vocabulary (`keyword_fallback`). Fastest, lowest quality. Useful for testing or very cost-sensitive setups.

Inbound tagging (the tags used for retrieval when a message arrives) is configured separately via `retrieval.inbound_tagger_type`: `"embedding"` (the default) assigns tags by vector similarity against the existing tag vocabulary using the local model at `retrieval.embedding_model` (default `all-MiniLM-L6-v2`), with no LLM call on the request path; `"llm"` routes inbound tagging through the LLM tagger instead. The LLM tagger always tags the completed turn on the background path, so every turn ends up LLM-tagged regardless of the inbound choice.

### Compaction

Controls when and how conversation history is compressed.

```yaml
compaction:
  soft_threshold: 0.70              # start compaction at this fill level
  hard_threshold: 0.85              # force compaction at this level
  protected_recent_turns: 6         # recent turns exempt from compaction
  min_summary_tokens: 200           # minimum tokens for a summary
  max_summary_tokens: 2000          # maximum tokens for a summary
```

**Thresholds** are fractions of the context window. At 70% fill with a 120K window, compaction starts when ~84K tokens are in use.

**Protected turns** are never compacted. This keeps the most recent conversation at full fidelity. Setting this too high wastes budget; too low loses important recent context.

### Summarization

Controls the LLM used for compaction summaries and fact extraction.

```yaml
summarization:
  provider: "anthropic"             # "anthropic", "openai", "gemini", "local", or "openrouter"
  model: "claude-haiku-4-5-20251001"
  temperature: 0.3                  # lower = more faithful summaries
```

The summarization LLM is separate from the upstream provider. You can use a cheap, fast model (Haiku, GPT-4o-mini) for summarization even if your upstream is a frontier model.

### Storage

```yaml
storage:
  backend: "sqlite"                 # "sqlite", "filesystem", "postgres", "neo4j", or "falkordb"
  sqlite:
    path: ".virtualcontext/store.db"
  postgres:
    dsn: "postgresql://user:pass@host:5432/vc"
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "password"
```

SQLite is the default and requires no setup. PostgreSQL (requires the `postgres` extra) is recommended for multi-worker proxy deployments. Neo4j/FalkorDB adds graph-based fact traversal. The `filesystem` backend stores segments as Markdown files with YAML frontmatter and does not host the full feature set of the database backends.

### Retrieval

```yaml
retrieval:
  inbound_tagger_type: "embedding"  # "embedding" or "llm" (inbound path)
  embedding_model: "all-MiniLM-L6-v2"  # local model for the embedding tagger
  active_tag_lookback: 4            # recent turns whose tags are skipped
  anchorless_lookback: 6            # turns used when no tags match
  strategy_config:
    default:
      max_results: 10               # max segments to retrieve
      max_budget_fraction: 0.25     # max fraction of window for context
      include_related: true         # include segments related to matches
```

Only the `default` strategy entry is read; per-strategy overrides beyond it have no effect.

**`active_tag_lookback`**: Tags from the last N turns are excluded from retrieval because their content is already in the raw conversation history. Higher values mean less redundancy but risk missing relevant older content under the same tags.

**`max_budget_fraction`**: The ceiling for injected context as a fraction of the total context window. At 0.25 with a 120K window, up to 30K tokens of retrieved summaries can be injected.

#### Scoring — context-augmented embedding signal

```yaml
retrieval:
  scoring:
    embedding_context_turns: 0      # 0 = bare query only (legacy default)
    embedding_context_guard: true   # per-tag max(bare, context) similarity
    embedding_reserved_seats: 0     # 0 = legacy; N reserves fused slots
```

**`embedding_context_turns`**: The embedding retrieval signal compares a query vector against stored tag-summary vectors. With `0` (default) the query vector is the bare inbound message. With `N > 0` the last `N` conversational turns are concatenated with the current message and embedded on the same encoder, so an under-specified query ("what should I get her") inherits the elided topic from recent context. Only takes effect with the embedding inbound tagger.

**`embedding_context_guard`**: When context turns are blended, `true` scores each tag by the maximum of its bare-query and context-augmented cosine similarity. This guarantees a tag can only rank at least as well as it would from the bare query alone, so irrelevant recent context cannot demote a relevant tag. `false` scores the context-augmented vector alone (plain concatenation, for experimentation).

**`embedding_reserved_seats`**: RRF fuses three ranked signals and penalizes candidates missing from a signal, which buries tags that only the embedding signal surfaces — common on analog queries with no keyword overlap. With `N > 0`, after fusion, dampening, and boosting, the top `N` embedding candidates that are not already in the fused top-K (K is the strategy's `max_results`) are forced into it by displacing its lowest-ranked entries. `0` (default) leaves fusion untouched. Composes with `embedding_context_turns`: the context blend surfaces the right embedding candidate, and reserved seats guarantee it survives the fused top-K cut.

### Assembly

```yaml
assembly:
  tag_context_max_tokens: 30000     # global budget for injected tag context
  recent_turns_always_included: 3   # recent turns always included in full
  context_hint_enabled: true        # inject topic list after compaction
  context_hint_max_tokens: 2000     # max tokens for the topic hint
```

**`context_hint_enabled`**: When true, after compaction the assembler injects a brief list of all available tags with segment counts. This gives the model topic awareness without spending full summary budget.

### Proxy

The single-instance listen address comes from the CLI (`--host`, default `127.0.0.1`; `--port`, default `5757`; `--upstream`), not from YAML. The `proxy:` block configures logging, limits, the session cache, and multi-instance mode:

```yaml
proxy:
  request_log_dir: null             # directory for per-request payload logs
  request_log_max_files: 200        # rotation cap for request logs
  llm_calls_log: null               # JSONL log of internal LLM calls
  upstream_context_limit: null      # cap on tokens forwarded upstream
  passthrough_trim_ratio: 0.40      # trim ratio applied on the passthrough path
  redis_url: null                   # Redis session cache (or REDIS_URL env)
  redis_history_cap: 200            # max history entries kept in Redis

  # Multi-instance mode
  instances:
    - port: 5757
      upstream: "https://api.anthropic.com"
      label: "anthropic"
      config: "./virtual-context-proxy-anthropic.yaml"
    - port: 5758
      upstream: "https://api.openai.com/v1"
      label: "openai"
      config: "./virtual-context-proxy-openai.yaml"
```

In multi-instance mode, each instance can have its own config file with isolated storage, tagger, and summarizer settings. Instances without a `config` field share the master engine.

### Tag Rules

Tag rules select a custom summarization prompt for segments whose tags match a pattern:

```yaml
tag_rules:
  - match: "legal-*"
    priority: 1
    summary_prompt: "Summarize precisely, preserving dates, names, and docket numbers."
```

`match` is a glob-style (fnmatch) pattern over tag names, `priority` breaks ties when several rules match (default 5, lower wins), and `summary_prompt` overrides the default summarization prompt for matching segments. Tag rules do not force segments into the assembled context.

## Presets

Virtual-context ships with presets for common use cases:

```bash
virtual-context presets list
virtual-context presets show coding
virtual-context presets show agentic
```

Use `virtual-context init <preset>` to bootstrap a config from a preset.

## Config Validation

```bash
virtual-context config validate
```

Reports missing required fields, invalid types, and cross-field constraint violations (e.g., soft threshold >= hard threshold).

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | API key for the Anthropic provider (read directly by the provider) |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | Provider API keys, honored through a `providers.<label>.api_key_env` entry in the config (the `onboard` and `init` commands generate these entries) |
| `REDIS_URL` | Redis session cache URL; overrides `proxy.redis_url` |
| `VC_DASHBOARD_TOKEN` | Require a token on dashboard endpoints (unauthenticated when unset) |
| `VIRTUAL_CONTEXT_CONFIG` | Config file path override, read by the MCP server only; the CLI uses `-c` and auto-discovery |

When a `providers:` block is present, `summarization.provider` must name one of its entries; config validation fails otherwise.
