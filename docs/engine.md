# Engine

The engine is the core intelligence layer. It handles compression, tagging, fact extraction, retrieval, assembly, and paging. The proxy and CLI are delivery mechanisms; the engine does the work.

## Compactor

Two-tier compaction converts raw conversation turns into compressed segments:

**Summary compaction** fires when the context window fill level crosses the soft threshold (default 70%). It selects uncompacted turns outside the protected window, groups them by tag overlap, and calls the summarization LLM to produce condensed segment summaries. Each summary preserves the tag set, turn range, and token count of the original.

**Deep compaction** fires at the hard threshold (default 85%). It re-compresses existing summaries into even shorter forms, trading detail for budget.

Compaction is incremental. A watermark tracks which turns have been processed. Only turns above the watermark are candidates. Protected recent turns (default 6) are never compacted, keeping the most recent context at full fidelity.

The compactor runs in the background thread after `on_turn_complete`, never blocking the response path.

### Compaction Configuration

```yaml
compaction:
  soft_threshold: 0.70        # begin compaction at 70% fill
  hard_threshold: 0.85        # force deep compaction at 85% fill
  protected_recent_turns: 6   # recent turns exempt from compaction
  min_summary_tokens: 100     # floor for summary length
  max_summary_tokens: 500     # ceiling for summary length
```

## Tagging Pipeline

Tags are the primary indexing mechanism. Every turn gets tagged twice, by two independent systems running in parallel:

### Inbound Embedding Tagger

Runs on the user message before the LLM responds. Uses a local embedding model to compute vector similarity against existing tags, then assigns the closest matches above a threshold. This is fast, deterministic, and ensures retrieval safety: if a topic was discussed before, its tag will be found even if the user phrases it differently.

### LLM Response Tagger

Runs after `on_turn_complete` with the full turn (user + assistant). Calls the configured tagger LLM (typically Haiku-class) to assign semantic tags. This produces richer vocabulary and catches nuances the embedding tagger misses, but takes a few hundred milliseconds.

The two tag sets are merged. The union provides both retrieval reliability (embeddings) and vocabulary richness (LLM).

### Context Bleed Gate

When the conversation shifts topics abruptly, the inbound tagger may carry forward tags from the previous topic due to embedding similarity. The context bleed gate detects sharp topic shifts by measuring the overlap between the current inbound tags and the recent tag history. When overlap drops below a threshold, it suppresses carryover tags that would pollute retrieval.

### Tag Splitting and Aliases

When a tag grows too large (too many segments assigned), the engine splits it into subtags and registers aliases. This preserves retrieval continuity: queries against the original tag name still find segments under the split subtags. Alias resolution uses the durable store, not in-memory iteration.

### TurnTagIndex

The `TurnTagIndex` is the in-memory index of per-turn tag assignments. Each entry records:

- Turn number
- Inbound tags (from the embedding tagger)
- Response tags (from the LLM tagger)
- Primary tag (the strongest single tag for this turn)
- Token count of the turn pair

The index supports lookback queries (e.g., "what tags were active in the last 4 turns?") used by retrieval to determine the working set.

## Segmenter

The segmenter splits compacted output into discrete segments, each with:

- A tag set (inherited from the compacted turns)
- A token count
- A text body (the summary)
- A turn range (which original turns this covers)
- A segment type (summary or deep-summary)

Segments are the unit of storage and retrieval. When retrieval selects content to inject, it selects segments.

## Retrieval

Retrieval finds stored segments relevant to the current query. It uses a 3-signal Reciprocal Rank Fusion (RRF) approach:

### Signal 1: IDF Tag Overlap

Compares the inbound query tags against segment tag sets, weighted by inverse document frequency. Tags that appear on few segments score higher than ubiquitous tags. This is the primary recall signal.

### Signal 2: BM25 Keyword

Standard BM25 scoring of the query text against segment text. Catches keyword matches that the tag system might miss.

### Signal 3: Embedding Cosine Similarity

Vector similarity between the query embedding and segment embeddings. Catches semantic matches where neither tags nor keywords overlap.

The three scores are combined via RRF with configurable weights.

### Gravity and Hub Dampening

**Gravity dampening** penalizes segments from far in the past, preferring recent context when scores are close.

**Hub dampening** penalizes segments that match too many queries (hub nodes in the retrieval graph), preventing commonly-tagged segments from dominating every retrieval.

### Active Tag Skipping

Tags from the most recent N turns (configurable via `active_tag_lookback`) are skipped during retrieval. Their content is already present in the raw conversation history within the context window, so retrieving them would waste budget on duplicates.

### Strategy Configuration

Different query types use different retrieval strategies:

```yaml
retrieval:
  active_tag_lookback: 4
  strategy_config:
    default:
      max_results: 10
      max_budget_fraction: 0.25
      include_related: true
    broad:
      max_results: 15
      max_budget_fraction: 0.35
    temporal:
      max_results: 8
      max_budget_fraction: 0.20
```

Broad queries (detected by heuristic: questions about "everything", "all", summary requests) get more budget. Temporal queries (detected by time references: "last week", "in March") use the temporal resolver to constrain the date range.

## Assembly

The assembler constructs the `<virtual-context>` block that gets injected into the system prompt. It operates in two passes:

### Priority Pass (Tag Rules)

Tag rules define must-include content. If a tag rule matches the current query, segments under that tag are included first, consuming budget from the top.

### Fill Pass (Greedy Set Cover)

Remaining budget is filled by the retrieval results using greedy set cover: segments are added in score order until the budget is exhausted. Segments that would exceed the remaining budget are skipped in favor of smaller ones that fit.

### Budget Management

The assembly budget is a fraction of the total context window (default 25%). The assembler tracks token counts precisely, including overhead for XML tags, separators, and metadata lines. The total injected context never exceeds `context_window * max_budget_fraction`.

### Context Hints

After compaction, the assembler can inject a topic list hint: a brief enumeration of all available tags with segment counts. This gives the model awareness of what it could ask about without spending tokens on full summaries. Controlled by `assembly.context_hint_enabled`.

## Token Counter

Three counting modes, selected at startup:

| Mode | Method | Speed | Accuracy |
|------|--------|-------|----------|
| `anthropic` | Anthropic's tokenizer library | Slow | Exact for Claude models |
| `tiktoken` | OpenAI's tiktoken library | Fast | Exact for GPT models, close for others |
| `estimate` | `len(text) / 4` | Instant | ~10-20% variance |

The counter is image-aware: for base64-encoded images, it uses dimension-based token costing (matching provider pricing) rather than counting the base64 string characters. This prevents massive overestimates for image-heavy conversations.

The fallback chain is `anthropic` -> `tiktoken` -> `estimate`, based on what's installed.

## Fact Extraction

Facts are structured triples extracted from conversation content:

```
subject | verb | object
```

Each fact has metadata:
- **Status**: `active`, `completed`, `planned`, `abandoned`, `recurring`
- **Date**: When the event occurred (absolute, not relative)
- **Location**: Where applicable
- **Type**: `personal` (about the user), `experience` (things done), `world` (external facts)

### Supersession

When a new fact contradicts an existing one, the old fact is superseded. "User moved from NYC to LA" invalidates "User lives in NYC." Supersession runs during the compaction LLM pass, which has full context to judge contradictions.

### Fact Querying

The `vc_query_facts` tool allows structured queries against the fact store:

```
vc_query_facts(subject="user", verb="visited", status="completed")
```

Verb matching includes morphological expansion: querying "led" also matches "leads", "leading". Object matching auto-relaxes if too narrow.

## Chain Collapse

Tool-heavy conversations (common with Claude Code, Cursor, etc.) produce massive `tool_use`/`tool_result` message pairs that dominate the context window. Chain collapse compresses these:

1. Consecutive tool_use + tool_result pairs are identified
2. The full content is stored to the durable store
3. The original messages are replaced with compact stubs containing a restore reference
4. The `vc_restore_tool(ref)` tool allows the model to recover any collapsed chain at full fidelity

This is lossless compression: nothing is discarded, just moved to cheaper storage with a pointer left in the conversation.

### Orphan Stripping

Chain restore handles edge cases where the collapsed range starts or ends mid-exchange (trailing `tool_use` without a `tool_result`, or leading `tool_result` without a `tool_use`). These orphans are stripped to maintain valid message structure.

## Media Compression

When the engine encounters base64-encoded images in conversation messages:

1. The image is decoded and resized to reduce dimensions
2. The compressed version replaces the original in the message
3. The original is written to disk for recovery

A 391KB screenshot becomes ~40KB, cutting payload size by ~90%. Since providers use vision encoders with dimension-based token costs (not base64 string length), the token savings are modest, but the bandwidth and latency improvements are significant.

## Monitor

The monitor tracks context window fill level in real time:

- After each turn, it recalculates: `(raw_history_tokens + injected_context_tokens) / context_window`
- When the fill level crosses the soft threshold, it signals the compactor
- When it crosses the hard threshold, it forces immediate deep compaction
- The fill level is exposed via the dashboard and telemetry

## Tool Loop

The tool loop manages multi-round tool interactions where the model calls vc_* tools:

### Tool Catalogue

The engine exposes these tools to the model:

| Tool | Purpose |
|------|---------|
| `vc_expand_topic` | Load full text for a topic tag (with optional collapse of other tags to free budget) |
| `vc_find_quote` | Full-text search across all stored conversation text |
| `vc_query_facts` | Structured fact lookup with filters |
| `vc_remember_when` | Time-scoped recall (date ranges + query) |
| `vc_recall_all` | Load all topic summaries at once |
| `vc_restore_tool` | Recover a collapsed tool chain at full fidelity |

### Anti-Repetition

The tool loop tracks which segments have been presented to the model across rounds. Duplicate retrievals are suppressed. If the model enters a search loop (querying the same thing repeatedly), strategy hints are injected to suggest alternative approaches.

### Empty Streak Detection

If multiple consecutive tool calls return no results, the loop injects hints suggesting the model try a different query strategy or stop searching.
