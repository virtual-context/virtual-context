# virtual-context

**Your LLM never forgets. Even in a 500-turn conversation.**

LLMs have fixed context windows. When conversations grow long, most systems do one of two things: silently drop your oldest messages, or embed everything into a vector database and hope cosine similarity finds what matters. Both fail in predictable ways. The architecture decision from turn 12 vanishes when turn 80 arrives. The legal filing deadline gets evicted because the user asked about dinner recipes. A vague question like "what did we discuss earlier?" returns nothing because it doesn't embed close to anything specific.

virtual-context is a different approach entirely. It treats LLM context the way an operating system treats RAM — as a managed memory hierarchy where nothing is lost, everything is compressed intelligently, and the right context is paged in exactly when needed.

```
Layer 0: Raw conversation turns              (active memory — in the context window)
Layer 1: Segment summaries per tag           (compressed pages — per-topic summaries)
Layer 2: Tag summaries via greedy set cover   (working set descriptors — bird's-eye view)
```

The result: an LLM that recalls details from turn 12 at turn 200 with the same fidelity as if the conversation just started.

## Why Not Just Use RAG?

RAG retrieves by similarity. virtual-context manages by understanding.

| | Truncation | RAG | virtual-context |
|---|---|---|---|
| **What gets kept** | Most recent N messages | Whatever embeds close to the query | Everything, at varying compression |
| **Cross-topic recall** | Fails silently | Depends on embedding quality | Tag overlap guarantees retrieval |
| **"What did we discuss?"** | Only recent context | Poor — query is too vague to embed | Broad detection loads all summaries; temporal detection retrieves by time position |
| **Token efficiency** | Wastes budget on irrelevant turns | Retrieved chunks may not fit budget | Budget-aware assembly with priority ordering |
| **Interpretability** | None | Opaque similarity scores | Visible tags, matched segments, budget breakdown |
| **Latency** | Zero | Embedding computation per query | Subsecond with local models |

## How It Works

Two hooks into your LLM pipeline. That's it.

```python
from virtual_context import VirtualContextEngine, Message

engine = VirtualContextEngine(config_path="./virtual-context.yaml")

# BEFORE sending to LLM — retrieve relevant stored context
assembled = engine.on_message_inbound(
    message="What was the Henninger filing deadline?",
    conversation_history=messages,
)
# assembled.prepend_text → enriched system prompt with retrieved summaries
# assembled.matched_tags → ["legal", "filing"]
# assembled.broad → False (this is a specific query)
# assembled.temporal → False (no time-position reference)

# AFTER LLM responds — tag, index, compact if needed
report = engine.on_turn_complete(messages)
if report:
    print(f"Compacted {report.segments_compacted} segments, freed {report.tokens_freed:,} tokens")
```

Everything happens synchronously, in-process, in under a second with a local model. No external services, no background workers, no async complexity.

### The Full Pipeline

```
User message arrives
    │
    ▼
Tag the message (LLM / keyword / embedding)
    │  ├─ Vocabulary feedback: known tags passed to tagger to prevent synonym drift
    │  ├─ Tag canonicalization: "db" → "database", alias detection via edit distance
    │  └─ Related tags generated: alternate terms for future recall
    │
    ▼
Retrieve matching summaries from store
    │  ├─ Broad query? → load ALL tag summaries (bounded post-compaction)
    │  ├─ Temporal query? → load segment summaries sorted earliest-first (Layer 1)
    │  ├─ Query expansion: primary tags + related tags widen the search
    │  ├─ Overfetch 3x → IDF-weighted re-rank (rare tag matches score higher)
    │  ├─ FTS fallback: if tag overlap finds nothing, full-text search on stored segments
    │  └─ Deep retrieval: full stored segment fetch for top matches
    │
    ▼
Assemble context within token budget
    │  ├─ Context hint: lightweight <context-topics> block (~50-200t)
    │  ├─ Tag sections: retrieved summaries ordered by tag priority
    │  └─ Filtered history: unrelated older turns dropped, broad/temporal mode includes all
    │
    ▼
LLM processes enriched context → produces response
    │
    ▼
Tag the user+assistant pair → update TurnTagIndex
    │  └─ Compactor generates related_tags at write time (vocabulary bridging)
    │
    ▼
Check token thresholds (soft 70%, hard 85%)
    │
    ▼ (if threshold exceeded)
Segment by tag → summarize each segment (concurrent, ThreadPoolExecutor)
    │  ├─ Tags preserved: LLM can ADD refined/related tags but never REMOVE originals
    │  └─ Related tags written into stored segments for future cross-vocabulary retrieval
    │
    ▼
Compute greedy set cover → build/update per-tag summaries (Layer 2)
```

## Key Capabilities

### Tags Emerge From Conversation

There are no predefined domains to configure. An LLM tagger reads each turn and generates semantic tags — `database`, `auth`, `fitness`, `legal` — that naturally converge over the session. A vocabulary feedback loop passes known tags back into the tagger prompt, so it reuses `storage` instead of inventing `data-persistence` or `file-management`.

The same codebase handles legal briefs, medical notes, coding sessions, recipe planning, and marathon training — whatever the user talks about. Tag rules let you configure priority, TTL, and custom summary prompts per tag family using fnmatch patterns.

### Broad Query Detection

"What did we discuss earlier?" "Can you summarize everything?" "What did you say about image storage?"

These queries don't map cleanly to specific tags. The LLM tagger flags them as `broad: true`, and the system switches behavior:

- The retriever loads **all** tag summaries instead of filtering by tag overlap
- History filtering includes all remaining turns instead of dropping unrelated ones
- Post-compaction, broad queries are **bounded** — compacted messages are skipped (tag summaries replace them), preventing token blowout

This eliminates the failure mode where the LLM says "I don't recall discussing that" about something from 50 turns ago.

### Temporal Query Detection

"Going back to the very beginning — what were the key decisions?" "What did we set up with tokens at the start?" "Something you said early on about indexing."

These queries reference a *position in time*, not just a topic. The LLM tagger flags them as `temporal: true`, and the retriever switches to a different data path:

- Instead of merged **tag summaries** (Layer 2), it fetches granular **segment summaries** (Layer 1)
- Segments are sorted by creation time — **earliest first** — so foundational decisions surface before later refinements
- Deep retrieval pulls full stored segment content for the top matches

This solves a fundamental problem with summarization: when a tag like `project-structure` appears at turn 1, turn 57, and turn 71, a merged tag summary blends all three. A temporal query about "the very first thing we discussed" needs the turn-1 segment specifically — not the merged blob.

Detection uses two layers (same pattern as broad): the LLM detects temporal intent, and deterministic regex patterns catch phrases the LLM misses (`"at the beginning"`, `"early on"`, `"the very first thing"`).

### Context Awareness Hints

After compaction, the LLM loses visibility into what topics have been stored. virtual-context injects a lightweight `<context-topics>` block into the system prompt:

```xml
<context-topics>
Prior conversation topics available for recall:
- recipes (15 turns): recipe app development, schema design for ingredients...
- running (8 turns): half-marathon training plan, knee injury prevention...
- housing (10 turns): rent stabilization law, tenant rights under DHCR...
- auth (12 turns): JWT implementation, OAuth2 flow, session management...
</context-topics>
```

This costs ~50-200 tokens and enables a natural drill-down loop: the user asks a broad question, the LLM sees what's available, synthesizes or asks for clarification, and the next turn pulls full detail via narrow tag retrieval.

### Three-Layer Memory Hierarchy

**Layer 0 — Raw turns.** The live conversation in the context window. Protected recent turns are never compacted.

**Layer 1 — Segment summaries.** When token pressure hits thresholds, consecutive same-tag turns are grouped and summarized by an LLM. Each segment preserves key decisions, entities, specific names, and action items. Original tags are never lost — the LLM can add tags during summarization but never remove them.

**Layer 2 — Tag summaries.** A greedy set cover algorithm finds the minimum set of tags that covers every turn. For each cover tag, all segment summaries are rolled up into a single tag-level summary. A focused session might produce 3 cover tags; a sprawling multi-topic session produces 10+. Only stale summaries (where new segments exist since the last build) are recomputed.

### Two-Tier Compaction

Compaction mirrors OS page replacement:

- **Soft threshold (70%)** — proactive compaction. Summarize now while there's headroom.
- **Hard threshold (85%)** — mandatory compaction. Summarize immediately or the context window overflows.

Compaction is greedy-batch: everything between the watermark and the protected zone gets compacted in one pass, so it fires infrequently — one big batch instead of many small ones. Summarization runs concurrently via ThreadPoolExecutor, with order-preserving results and per-tag custom prompts.

### Tag Canonicalization

Tags naturally produce synonyms: `db`, `database`, `data-storage`. The TagCanonicalizer detects aliases via edit distance and normalizes them automatically. You can also register aliases manually:

```bash
virtual-context aliases suggest    # auto-detect potential aliases
virtual-context aliases add db database
```

### Cross-Vocabulary Retrieval

Users don't use the same words every time. A discussion about "materialized views for feed performance" at turn 46 might be recalled as "that caching trick for the feed" at turn 71. Pure tag overlap finds nothing — the vocabularies are completely disjoint.

virtual-context solves this with two complementary mechanisms:

**Related tag expansion.** Both the tagger (query-side) and compactor (write-side) generate `related_tags` — alternate terms someone might use to refer to the same concepts. A segment about "materialized views" gets stored with related tags like `caching`, `precomputed`, `feed-optimization`. A query about "caching trick" generates related tags that overlap with stored segments. The retriever expands its search to include both primary and related tags.

**IDF-weighted scoring.** When multiple segments match, common tags like `database` (appearing on 20+ segments) shouldn't score the same as rare tags like `postgres` (appearing on 3). The retriever computes inverse document frequency weights from tag usage counts, overfetches 3x the needed results, then re-ranks by `sum(IDF[tag])`. Related tag matches score at 0.5x weight. The correct segment surfaces even when all overlapping tags are high-frequency.

### Budget-Aware Assembly

The assembler builds context within a strict token budget, with priority ordering from tag rules:

```yaml
tag_rules:
  - match: "architecture*"
    priority: 10          # always included first
  - match: "debug*"
    priority: 7
    ttl_days: 7           # debugging context expires fast
  - match: "*"
    priority: 5
    ttl_days: 30
```

Higher-priority tags get assembled first. If the budget runs out, lower-priority summaries are dropped. The budget breakdown is fully transparent: core context, context hint, tag sections, and conversation history each have their own allocation.

## Install

```bash
pip install virtual-context
```

Optional extras:

```bash
pip install virtual-context[tui]         # interactive chat terminal
pip install virtual-context[embeddings]  # sentence-transformers tag generator
pip install virtual-context[tiktoken]    # exact token counting
pip install virtual-context[mcp]         # Model Context Protocol server
pip install virtual-context[all]         # everything
```

Minimal dependencies: `pyyaml` + `httpx`. Python 3.11+.

## Configuration

Create `virtual-context.yaml` in your project root:

```yaml
version: "0.2"
storage_root: ".virtualcontext"
context_window: 120000

# Tags emerge from conversation — the LLM generates them
tag_generator:
  type: "llm"                               # "llm", "keyword", or "embedding"
  provider: "ollama"
  model: "qwen3:4b-instruct-2507-fp16"
  min_tags: 5
  max_tags: 10
  keyword_fallback:                          # used if LLM unavailable
    tag_keywords:
      legal: [court, filing, motion, attorney]
      code: [function, bug, deploy, API, database]

# Per-tag behavior: priority, expiration, custom summary prompts
tag_rules:
  - match: "architecture*"
    priority: 10
    ttl_days: null                           # never expire
    summary_prompt: |
      Summarize architectural decisions and tradeoffs.
      Preserve component names and rationale.

  - match: "debug*"
    priority: 7
    ttl_days: 7                              # debugging context stales fast

  - match: "*"
    priority: 5
    ttl_days: 30

# Compaction triggers
compaction:
  soft_threshold: 0.70                       # proactive
  hard_threshold: 0.85                       # mandatory
  protected_recent_turns: 6
  max_concurrent_summaries: 4

# Context awareness hint (post-compaction topic list)
assembly:
  context_hint_enabled: true
  context_hint_max_tokens: 200

# Local LLM for tagging + summarization
providers:
  ollama:
    type: "generic_openai"
    base_url: "http://127.0.0.1:11434/v1"

storage:
  backend: "sqlite"
  sqlite:
    path: ".virtualcontext/store.db"
```

See `virtual-context.yaml.example` for the full annotated configuration.

Generate a starter config from a preset:

```bash
virtual-context init coding    # tuned for software development
```

## Three Tag Generators

**LLM tagger** (recommended) — Uses any local model via Ollama, LM Studio, or vLLM. Generates rich semantic tags with broad query detection. Vocabulary feedback ensures convergence. Falls back to keyword tagger if the LLM is unavailable.

**Keyword tagger** — Deterministic regex and keyword matching. Zero latency, zero cost, fully reproducible. Good for domains with well-defined vocabularies.

**Embedding tagger** — Uses sentence-transformers to compute cosine similarity against a tag vocabulary. Middle ground between LLM accuracy and keyword speed.

## CLI

```bash
virtual-context status                         # tag stats and token usage
virtual-context tags                           # list all tags with counts
virtual-context recall auth                    # retrieve stored summaries for a tag
virtual-context compact -i msgs.json           # manual compaction from message file
virtual-context retrieve -m "What about auth?" # tag + retrieve (JSON output)
virtual-context transform -m "What about auth?"# tag + retrieve + assemble
virtual-context aliases list                   # show all tag aliases
virtual-context aliases suggest                # auto-detect potential aliases
virtual-context aliases add db database        # register alias manually
virtual-context config validate                # check config syntax
virtual-context cost-report                    # show session LLM usage
```

## Interactive Chat (TUI)

```bash
pip install virtual-context[tui]
virtual-context chat --config virtual-context.yaml
```

A terminal chat interface with live context visualization:

- **Tag panel** — current tag working set with activity levels
- **Budget bar** — real-time token usage breakdown (core, tags, hint, conversation)
- **Turn list** — every turn with its tags, navigable with Ctrl+B/F
- **Turn inspector** (Ctrl+I) — full turn data: API payload, tags, assembled context, broad/temporal flags
- **Brief mode** (Ctrl+T) — silently appends "answer in 2 lines" for faster iteration
- **Manual compaction** — type `/compact` or press Ctrl+K
- **Session export** (Ctrl+S) — saves full session to `vc-session.json` with all metadata

### Headless Mode

Run prompts through the full pipeline without a terminal UI:

```bash
virtual-context chat --headless --replay prompts.txt
```

Session JSON captures every turn and can be replayed to test behavior changes against recorded conversations:

```bash
virtual-context chat --replay vc-session.json
```

## Integrations

### MCP Server (Model Context Protocol)

```bash
pip install virtual-context[mcp]
```

Exposes virtual-context as an MCP server for integration with Claude Desktop, Cursor, or any MCP-compatible client:

| Type | Name | Description |
|------|------|-------------|
| Tool | `recall_context` | Tag + retrieve + assemble context for a message |
| Tool | `compact_context` | Trigger compaction on a message history |
| Tool | `domain_status` | All tags with stats |
| Resource | `virtualcontext://domains` | List all tags |
| Resource | `virtualcontext://domains/{tag}` | Summaries for a specific tag |
| Prompt | `recall` | Suggest context retrieval for a topic |
| Prompt | `summarize_session` | Suggest compaction |

### OpenClaw Plugin

Plugin for OpenClaw agents using lifecycle hooks for sync retrieval (`message.pre`) and fire-and-forget compaction (`agent.post`) via CLI calls — no bridge server needed. Depends on the [plugin lifecycle hook architecture](https://github.com/openclaw/openclaw/pull/12082) currently in progress.

## Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Engine** | `engine.py` | Main orchestrator — `on_message_inbound()` and `on_turn_complete()` |
| **TurnTagIndex** | `core/turn_tag_index.py` | Live per-turn tag index, velocity tracking, greedy set cover |
| **TagGenerator** | `core/tag_generator.py` | LLM / keyword / embedding semantic tagging |
| **TagCanonicalizer** | `core/tag_canonicalizer.py` | Alias detection, plural folding, normalization |
| **Retriever** | `core/retriever.py` | IDF-weighted tag retrieval, related tag expansion, broad/temporal queries, FTS fallback |
| **Assembler** | `core/assembler.py` | Budget-aware context assembly with priority ordering |
| **Monitor** | `core/monitor.py` | Two-tier threshold detection (soft/hard) |
| **Segmenter** | `core/segmenter.py` | Turn pairing + contiguous tag grouping via TurnTagIndex |
| **Compactor** | `core/compactor.py` | LLM summarization + tag summary rollup, concurrent via ThreadPoolExecutor |
| **CostTracker** | `core/cost_tracker.py` | Per-session LLM usage and cost tracking |
| **ContextStore** | `core/store.py` | Storage interface (SQLite or filesystem) |

### Storage Backends

**SQLiteStore** — Primary backend. FTS5 full-text search, tag-overlap queries via junction table, tag aliases, tag summaries. Single file, no external dependencies.

**FilesystemStore** — Debug/inspection backend. Markdown files with YAML frontmatter, organized by tag directory. Human-readable, git-friendly.

Both implement the same abstract interface — swap backends without changing application code.

### LLM Providers

**GenericOpenAIProvider** — Works with Ollama, LM Studio, vLLM, or any OpenAI-compatible endpoint. Pure httpx, no SDK dependency.

**AnthropicProvider** — Direct Anthropic API via httpx. No SDK dependency.

Retry logic with exponential backoff on both.

## Design Decisions

**Sync-first.** Zero async/await. All I/O is synchronous httpx. Concurrent compaction uses `ThreadPoolExecutor`, not asyncio. Both engine entry points complete in under a second with a local Ollama model.

**Tag overlap with IDF scoring, not vector similarity.** Retrieval matches by IDF-weighted tag overlap, not cosine similarity. Related tag expansion handles vocabulary mismatch. Faster (no embedding computation at query time), fully interpretable, and composable with the tag hierarchy.

**Vocabulary feedback, not few-shot.** The LLM tagger gets a live vocabulary of tags already used in the session and store, and is instructed to reuse them when the topic matches. Convergence without manual curation.

**No SDK dependencies.** Both LLM providers use raw httpx. The only required dependencies are `pyyaml` and `httpx`.

**Tag preservation.** During compaction, the LLM can add refined tags but never remove original ones. A segment tagged `[ux, recipes, frontend]` stays tagged with all three even after summarization, ensuring cross-topic retrieval always works.

## Stress-Tested

virtual-context has been validated against 100-turn adversarial conversations with deliberately overlapping domains (Flask IoT API, music studio, ML pipeline, cross-domain integration), vocabulary mismatches, ambiguous callbacks, and cross-domain synthesis queries — using a 3,000-token context window with Claude Haiku. Results:

- **Cross-vocabulary recall**: "caching trick for the feed" correctly retrieves "materialized view" despite zero primary tag overlap — related tag expansion bridges the vocabulary gap
- **IDF-weighted precision**: "precomputed summary table" retrieves the correct segment over 20+ competing segments sharing common tags like `database` and `performance`
- **Ambiguous multi-match**: "what middleware pattern?" correctly identifies both auth and logging middleware across 4 overlapping domains; "plugins — Flask, audio, or ML?" correctly disambiguates
- **Temporal recall**: "going back to the very beginning — what were the key decisions?" retrieves original Flask blueprint architecture from turn 1 via segment-level retrieval, even after 4 compaction events
- **Broad query bounding**: "summary of everything we've discussed" loads 22 bundled tag summaries but stays bounded at ~2,900 tokens post-compaction
- **Adversarial pass rate**: 89% on 28 deliberately adversarial prompts (vocabulary mismatches, ambiguous references, cross-domain synthesis, late vague recalls)
- **Compaction**: 4 events across 100 turns, average 1,147 tokens per turn, peak 3,018 tokens
- **Tag convergence**: vocabulary stabilizes within 10-15 turns via feedback loop

## Development

```bash
git clone https://github.com/virtual-context/virtual-context.git
cd virtual-context
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v --ignore=tests/ollama    # 275 unit tests
python -m pytest tests/ollama/ -v -m ollama          # integration (requires Ollama)
```

## License

AGPL-3.0 — Copyright Y. Ahmed Kidwai

For commercial licensing inquiries, contact: ahmed@kidw.ai
