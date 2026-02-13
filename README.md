# virtual-context

OS-style virtual memory for LLM session context management.

LLM agents have fixed context windows. As conversations grow, context is compacted by recency (drop oldest, keep newest), which is domain-blind. A legal discussion gets thrown away to keep a casual greeting. virtual-context brings OS-style virtual memory to LLM context: segment by domain, summarize each independently, store full detail externally, and retrieve relevant domain summaries on demand.

## Install

```bash
pip install virtual-context
```

## Quick Start

```python
from virtual_context import VirtualContextEngine, Message

engine = VirtualContextEngine(config_path="./virtual-context.yaml")

# Before sending to LLM: enrich context with domain summaries
assembled = await engine.on_message_inbound(
    message="What was the Henninger filing deadline?",
    conversation_history=my_messages,
)
# assembled.prepend_text = core files + domain summaries
# assembled.conversation_history = trimmed recent history

# After LLM responds: check thresholds, compact if needed
report = await engine.on_turn_complete(my_messages)
if report:
    print(f"Compacted {report.segments_compacted} segments, freed {report.tokens_freed} tokens")
```

## How It Works

```
User message arrives
    |
    v
ContextRetriever (classify message -> fetch domain summaries from store)
    |
    v
ContextAssembler (core files + domain summaries + recent conversation -> token budget)
    |
    v
LLM processes enriched context -> produces response
    |
    v
ContextMonitor (check token usage against thresholds)
    |
    v (if threshold hit)
TopicSegmenter (split conversation by domain) -> DomainCompactor (summarize each)
    |                                               |
    v                                               v
Remove compacted messages                    ContextStore (persist full + summary)
```

## Configuration

Create `virtual-context.yaml` in your project root:

```yaml
version: "1.0"
context_window: 120000

domains:
  legal:
    description: "Legal matters, court filings, case strategy"
    keywords: ["court", "filing", "motion", "attorney"]
    patterns: ["\\b\\d{2}-cv-\\d+"]
    priority: 9

  medical:
    description: "Health, medications, symptoms"
    keywords: ["insulin", "medication", "doctor", "glucose"]
    priority: 8

classifier:
  min_confidence: 0.3
  pipeline:
    - type: keyword

compaction:
  soft_threshold: 0.70    # proactive compaction
  hard_threshold: 0.85    # mandatory compaction
  protected_recent_turns: 6
  summary_ratio: 0.15

summarization:
  provider: "anthropic"
  model: "claude-haiku-4-5"

providers:
  anthropic:
    type: "anthropic"
    api_key_env: "ANTHROPIC_API_KEY"

storage:
  backend: "filesystem"
  filesystem:
    root: ".virtualcontext/store"
```

See `virtual-context.yaml.example` for the full annotated configuration.

## CLI

```bash
virtual-context status          # domain stats and token usage
virtual-context domains         # list configured domains
virtual-context recall legal    # retrieve stored context for a domain
virtual-context compact -i msgs.json  # trigger manual compaction
virtual-context config validate # validate config file
```

## Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| **ContextMonitor** | Two-tier threshold checking (soft 70%, hard 85%) |
| **TopicSegmenter** | Groups messages into turn pairs, classifies by domain |
| **DomainCompactor** | Summarizes each domain segment via LLM |
| **ContextStore** | Pluggable storage (filesystem, SQLite, vector) |
| **ContextRetriever** | Fetches relevant domain summaries for inbound messages |
| **ContextAssembler** | Builds final context within token budget |

### Classifier Pipeline

Ordered fallback chain. Cheap classifiers first, expensive only if needed:

1. **KeywordClassifier** (MVP): regex + keyword matching, zero deps
2. EmbeddingClassifier (v0.2): sentence-transformers cosine similarity
3. LLMClassifier (v0.3): fast model for structured classification

### Storage Backends

- **FilesystemStore** (MVP): Markdown files with YAML frontmatter + JSON index
- SQLiteStore (v0.2): FTS5 full-text search
- VectorStore (v0.3): ChromaDB semantic search

## Development

```bash
git clone https://github.com/virtual-context/virtual-context.git
cd virtual-context
pip install -e ".[dev]"
pytest
```

## License

AGPL-3.0
