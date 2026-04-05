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
  type: "llm"                       # "llm", "embedding", or "keyword"
  provider: "anthropic"             # "anthropic", "openai", "gemini", "local", or "openrouter"
  model: "claude-haiku-4-5-20251001"
  max_tags: 10                      # maximum tags per turn
  min_tags: 5                       # minimum tags to assign
  broad_patterns: []                # regex patterns for broad query detection
  temporal_patterns: []             # regex patterns for temporal query detection
  embedding_model: "text-embedding-3-small"  # model for embedding tagger
```

**Type options**:

- `llm`: Full LLM-based tagging. Best quality, ~200ms latency per turn. Uses the configured provider and model.
- `embedding`: Embedding-only tagging. Faster, deterministic, but limited vocabulary. Assigns tags by vector similarity to existing tag set.
- `keyword`: Regex-based keyword extraction. Fastest, lowest quality. Useful for testing or very cost-sensitive setups.

When `type` is `llm`, both the embedding tagger and LLM tagger run in parallel (two-tagger architecture). The embedding tagger provides the inbound tags for retrieval, while the LLM tagger provides richer response tags.

### Compaction

Controls when and how conversation history is compressed.

```yaml
compaction:
  soft_threshold: 0.70              # start compaction at this fill level
  hard_threshold: 0.85              # force deep compaction at this level
  protected_recent_turns: 6         # recent turns exempt from compaction
  min_summary_tokens: 100           # minimum tokens for a summary
  max_summary_tokens: 500           # maximum tokens for a summary
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
  backend: "sqlite"                 # "sqlite", "postgres", or "neo4j"
  sqlite:
    path: ".virtualcontext/store.db"
  postgres:
    dsn: "postgresql://user:pass@host:5432/vc"
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "password"
```

SQLite is the default and requires no setup. PostgreSQL is recommended for multi-worker proxy deployments. Neo4j/FalkorDB adds graph-based fact traversal.

### Retrieval

```yaml
retrieval:
  active_tag_lookback: 4            # recent turns whose tags are skipped
  anchorless_lookback: 2            # turns used when no tags match
  strategy_config:
    default:
      max_results: 10               # max segments to retrieve
      max_budget_fraction: 0.25     # max fraction of window for context
      include_related: true         # include segments related to matches
    broad:
      max_results: 15
      max_budget_fraction: 0.35
    temporal:
      max_results: 8
      max_budget_fraction: 0.20
```

**`active_tag_lookback`**: Tags from the last N turns are excluded from retrieval because their content is already in the raw conversation history. Higher values mean less redundancy but risk missing relevant older content under the same tags.

**`max_budget_fraction`**: The ceiling for injected context as a fraction of the total context window. At 0.25 with a 120K window, up to 30K tokens of retrieved summaries can be injected.

### Assembly

```yaml
assembly:
  tag_context_max_tokens: 2000      # max tokens per tag rule inclusion
  recent_turns_kept: 4              # recent turns always included in full
  context_hint_enabled: true        # inject topic list after compaction
  context_hint_max_tokens: 500      # max tokens for the topic hint
```

**`context_hint_enabled`**: When true, after compaction the assembler injects a brief list of all available tags with segment counts. This gives the model topic awareness without spending full summary budget.

### Proxy

```yaml
proxy:
  host: "0.0.0.0"
  port: 8100
  upstream: "https://api.anthropic.com"

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

Tag rules force specific segments to always be included when certain tags are active:

```yaml
tag_rules:
  - tags: ["project-setup", "architecture"]
    priority: 1
    max_tokens: 2000
```

When retrieval detects these tags in the query, the corresponding segments are included before the greedy fill pass, ensuring critical context is never dropped.

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
| `ANTHROPIC_API_KEY` | API key for Anthropic provider (tagger, summarizer, or upstream) |
| `OPENAI_API_KEY` | API key for OpenAI provider |
| `GEMINI_API_KEY` | API key for Google Gemini provider |
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `VIRTUAL_CONTEXT_CONFIG` | Override config file path (equivalent to `-c`) |
