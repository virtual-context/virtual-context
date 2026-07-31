# Engine

The engine is the core intelligence layer. It handles compression, tagging, fact extraction, retrieval, assembly, and paging. The proxy and CLI are delivery mechanisms; the engine does the work.

## Compactor

Compaction converts raw conversation turns into compressed segments. It starts when the context window fill level crosses the soft threshold (default 70%) and is forced at the hard threshold (default 85%). It selects uncompacted turns outside the protected window, groups them by tag overlap, and calls the summarization LLM to produce condensed segment summaries. Each summary preserves the tag set, turn range, and token count of the original, and every non-`_general` tag on a just-compacted segment gets its tag summary materialized at commit.

Compaction is incremental. A watermark tracks which turns have been processed; only turns above the watermark are candidates. Protected recent turns (default 6) are never compacted, keeping the most recent context at full fidelity. In multi-worker deployments each compaction runs as a leased, fenced operation, so a stalled worker cannot overwrite a takeover (see [architecture](architecture.md)).

There is no second summarize-the-summaries tier. The separately named `deep_compaction_ratio` is a payload-side filter in the proxy's message rewriting: turns far enough below the compacted boundary are dropped from the outgoing payload entirely instead of being stubbed, because the segment summaries already cover them. Stored data is never affected.

The compactor runs on a background pool after `on_turn_complete`, never blocking the response path.

### Compaction Configuration

```yaml
compaction:
  soft_threshold: 0.70        # begin compaction at 70% fill
  hard_threshold: 0.85        # force compaction at 85% fill
  protected_recent_turns: 6   # recent turns exempt from compaction
  min_summary_tokens: 200     # floor for summary length
  max_summary_tokens: 2000    # ceiling for summary length
```

## Tagging Pipeline

Tags are the primary indexing mechanism. Every turn is tagged on two paths, at two different moments:

### Inbound Embedding Tagger

Runs on the user message before the LLM responds (selected by `retrieval.inbound_tagger_type`, default `"embedding"`). Uses a local embedding model to compute vector similarity against existing tags, then assigns the closest matches above a threshold. This is fast, deterministic, and ensures retrieval safety: if a topic was discussed before, its tag will be found even if the user phrases it differently.

### LLM Turn Tagger

Runs on the background path with the full completed turn (user + assistant). Calls the configured tagger LLM (typically a small, fast model) to assign semantic tags. This produces richer vocabulary and catches nuances the embedding tagger misses, with no latency pressure because the response has already streamed. Every turn ends up LLM-tagged; the inbound embedding tags exist so retrieval has tags before the turn completes.

### Context Bleed Gate

The turn tagger can include preceding turns in its prompt for context. The context bleed gate is an embedding-similarity check (`tag_generator.context_bleed_threshold`, default 0.1) that decides whether that preceding context is related enough to include: when the current turn's similarity to the preceding context falls below the threshold, the context is left out of the tagger prompt so an abrupt topic shift is not tagged with the previous topic's vocabulary.

### Tag Splitting and Aliases

When a tag grows too large (too many segments assigned), the engine splits it into subtags and registers aliases. This preserves retrieval continuity: queries against the original tag name still find segments under the split subtags. Alias resolution uses the durable store, not in-memory iteration.

### TurnTagIndex

The `TurnTagIndex` is the in-memory index of per-turn tag assignments. Each entry records:

- Turn number
- The turn's tag list and primary tag (the strongest single tag)
- The backing canonical turn ID
- Session date, sender, and fact signals

The index supports lookback queries (e.g., "what tags were active in the last 4 turns?") used by retrieval to determine the working set. It is rebuilt from canonical turn rows on session restore.

## Segmenter

The segmenter splits compacted output into discrete segments, each with:

- A tag set (inherited from the compacted turns)
- A token count
- A text body (the summary)
- A turn range (which original turns this covers)

Segments are the unit of summary storage. Retrieval ranks *tags* (see below) and then fetches the segments stored under the selected tags.

## Retrieval

Retrieval decides which stored *topics* are relevant to the current query. The ranked unit is the tag: three signals each produce a ranked list of candidate tags, the lists are fused, and segments are then fetched for the winning tags.

### Signal 1: IDF Tag Overlap

Compares the inbound query tags against stored tags, weighted by inverse document frequency. Tags that appear on few segments score higher than ubiquitous tags. This is the primary recall signal.

### Signal 2: BM25 Keyword

BM25 scoring of the query text against stored summary text, aggregated per tag. Catches keyword matches that the tag system might miss.

### Signal 3: Embedding Cosine Similarity

Vector similarity between the query embedding and tag-summary embeddings. Catches semantic matches where neither tags nor keywords overlap. The query vector can optionally be blended with recent conversational context (`retrieval.scoring.embedding_context_turns`), with a guard that prevents the blend from demoting a tag below its bare-query score.

The three per-tag rankings are combined via Reciprocal Rank Fusion with configurable weights.

### Dampening and Boosts

After fusion, three adjustments run (all on by default, configurable under `retrieval.scoring.dampening`):

**Gravity dampening** halves the embedding score of tags that have no BM25 support at all, so a purely-semantic match cannot outrank tags with corroborating keyword evidence.

**Hub dampening** penalizes tags whose segment count exceeds the 90th percentile of the tag-count distribution (query tags exempt), preventing catch-all topics from dominating every retrieval.

**Resolution boost** promotes fact-bearing tags, so topics with extracted structured facts rank ahead of equally-scored topics without them.

**Reserved seats** (`retrieval.scoring.embedding_reserved_seats`, off by default) can force the top N embedding-only candidates into the fused top-K, for queries where the embedding signal is the only one that finds the right topic.

### Active Tag Skipping

Tags from the most recent N turns (configurable via `active_tag_lookback`) are skipped during retrieval. Their content is already present in the raw conversation history within the context window, so retrieving them would waste budget on duplicates.

### Strategy Configuration

```yaml
retrieval:
  active_tag_lookback: 4
  strategy_config:
    default:
      max_results: 10
      max_budget_fraction: 0.25
      include_related: true
```

Only the `default` strategy entry is read. Broad recall ("summarize everything we discussed") is served by the `vc_recall_all` tool, and time-scoped queries by `vc_remember_when`, which uses the temporal resolver to turn relative dates into absolute ranges.

## Assembly

The assembler constructs the `<virtual-context>` block that gets injected into the system prompt. It operates in two passes:

### Priority Pass (Tag Rules)

Tag rules define must-include content. If a tag rule matches the current query, segments under that tag are included first, consuming budget from the top.

### Fill Pass (Greedy Set Cover)

Remaining budget is filled by the retrieval results using greedy set cover: segments are added in score order until the budget is exhausted. Segments that would exceed the remaining budget are skipped in favor of smaller ones that fit.

### Budget Management

The assembly budget is a fraction of the total context window (default 25%). The assembler tracks token counts precisely, including overhead for XML tags, separators, and metadata lines. The total injected context never exceeds `context_window * max_budget_fraction`.

### Context Hints

After compaction, the assembler injects a structured `<context-topics>` block: a budgeted topic list with per-topic descriptors, a line naming how many topics exist in total, and guidance telling the model how to page more detail in via the vc_* tools. Controlled by `assembly.context_hint_enabled`, budgeted by `assembly.context_hint_max_tokens` (default 2000). The hint is cached and pre-warmed at compaction commit, so the first request after a compaction does not pay the rebuild cost.

## Token Counter

Three counting modes, selected at startup:

| Mode | Method | Speed | Accuracy |
|------|--------|-------|----------|
| `anthropic` | Bundled Claude-oriented tokenizer file | Slow | Approximate (~6% error vs. API-reported counts) |
| `tiktoken` | OpenAI's tiktoken library | Fast | Exact for GPT models, close for others |
| `estimate` | `len(text) / 4` | Instant | Rough |

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

When a new fact contradicts an existing one, the old fact is superseded. "User moved from NYC to LA" invalidates "User lives in NYC." Supersession is a dedicated LLM-backed checker with its own provider and model configuration (`supersession:` block), invoked from the compaction pipeline after fact extraction.

### Fact Querying

The `vc_query_facts` tool allows structured queries against the fact store:

```
vc_query_facts(subject="user", verb="visited", status="completed")
```

Verb matching expands through hand-curated synonym clusters plus embedding similarity against the verbs present in the store, so querying "led" can also match "managed" or "ran" where those verbs were extracted. Facts can additionally be ranked by dense similarity between the query and stored fact embeddings (`retrieval.fact_dense_retrieval`, model-versioned embeddings written at extraction time).

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

The engine exposes eight tools to the model on the proxy tool loop:

| Tool | Purpose |
|------|---------|
| `vc_expand_topic` | Load full text for a topic tag (with optional collapse of other tags to free budget) |
| `vc_find_quote` | Full-text and semantic search across all stored conversation text |
| `vc_search_summaries` | Search segment summaries instead of raw text |
| `vc_find_session` | Locate a session after a session-suppressed quote result |
| `vc_query_facts` | Structured fact lookup with filters |
| `vc_remember_when` | Time-scoped recall (date ranges + query) |
| `vc_recall_all` | Load all topic summaries at once |
| `vc_restore_tool` | Recover a collapsed tool chain at full fidelity |

This tool-loop surface is distinct from the MCP server's tool set (see the README's MCP section); the two lists overlap but are not the same.

### Anti-Repetition

The tool loop tracks which segments have been presented to the model across rounds. Duplicate retrievals are suppressed. If the model enters a search loop (querying the same thing repeatedly), strategy hints are injected to suggest alternative approaches.

### Empty Streak Detection

If multiple consecutive tool calls return no results, the loop injects hints suggesting the model try a different query strategy or stop searching.
