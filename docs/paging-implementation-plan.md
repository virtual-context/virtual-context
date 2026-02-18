# LLM-Driven Context Navigation (Virtual Memory Paging)

## Overview

Today virtual-context is transparent: it silently injects tag summaries into the LLM's context window, and the LLM has no idea VC exists. The compression hierarchy (raw turns, segment summaries, tag summaries) already stores data at every depth level, but the assembler always serves at a single depth.

This feature makes the hierarchy **bidirectional**. The LLM can expand topics to full detail or collapse them to summaries, and VC manages the budget. The analogy is a kernel page fault handler: pages in what's needed, pages out what's cold, keeps the working set within physical memory.

## The Problem

Every LLM application hits the context window ceiling. The standard answers are "use a bigger window" or "use RAG." Bigger windows are expensive and still finite. RAG retrieves and appends but never frees space from what's already in the window.

When a 100k document needs to enter a 120k window that already has 60k of conversation history, RAG has no answer. It can truncate (lossy), error out (useless), or chunk (every chunking approach either costs extra user turns, loses cross-chunk coherence, or both).

VC paging collapses 60k of stale conversation history to 8k of tag summaries, freeing 52k, then pages in the full document. The working set reshapes itself around whatever the user needs right now.

See `memory/vc-vs-rag-example.md` for the full design rationale with concrete scenarios, chunking failure mode analysis, and token overhead risk assessment.

## Depth Levels

Data already exists at each level in the store:

| Level | What's Injected | Typical Cost | Source |
|-------|----------------|--------------|--------|
| `none` | Nothing (listed in hint only) | 0t | N/A |
| `summary` | Tag summary | ~200t per tag | `TagSummary.summary` |
| `segments` | Individual segment summaries | ~2,000t per tag | `StoredSummary.summary` per segment |
| `full` | Complete original text | ~8,000t+ per tag | `StoredSegment.full_text` |

## Working Set

New per-session state: a map of `tag -> current depth level` that persists across turns.

Today VC rebuilds what to inject from scratch each turn (stateless retrieval). With paging, depth decisions are stateful: "recipes was expanded last turn, keep it expanded until someone collapses it or budget pressure evicts it."

```python
working_set = {
    "recipes":     WorkingSetEntry(tag="recipes",     depth=FULL,    tokens=8000, last_accessed_turn=28),
    "legal-brief": WorkingSetEntry(tag="legal-brief", depth=SUMMARY, tokens=500,  last_accessed_turn=5),
    "deployment":  WorkingSetEntry(tag="deployment",   depth=NONE,    tokens=0,    last_accessed_turn=15),
}
```

Persisted in `EngineStateSnapshot` alongside TurnTagIndex and compaction watermark.

## Model-Tiered Delegation

Not all LLMs are equally capable of managing their own context. A config-level or per-session mode adapts how much control the LLM gets:

### Supervised mode (weaker models: Haiku, small open-source)

VC gives the LLM a topic list, not a budget dashboard. The LLM can request expansions but doesn't make eviction decisions:

```xml
<context-topics>
Prior conversation topics (call expand_topic to see full detail):
- recipes (summary, 200t): Discussed sourdough bread timing...
- deployment (summary, 150t): Docker setup and CI pipeline...
- legal-brief (none): Large document, available on request
</context-topics>
```

### Autonomous mode (stronger models: Opus, Sonnet, GPT-4)

VC gives the LLM the full budget dashboard with token costs:

```xml
<context-topics budget="120000" used="62000" available="58000">
- recipes (depth: summary, current: 200t, full: 8000t, last_turn: 28)
- deployment (depth: summary, current: 150t, full: 3000t, last_turn: 15)
- legal-brief (depth: none, current: 0t, full: 100000t, last_turn: 5)

Tools: expand_topic(tag, depth?) | collapse_topic(tag, depth?)
</context-topics>
```

### Auto mode

Maps model name to delegation level. Haiku/small models get supervised. Opus/Sonnet/GPT-4 get autonomous. Unknown models default to supervised (safe fallback).

```yaml
paging:
  enabled: true
  mode: auto          # auto | supervised | autonomous
  auto_promote: true  # auto-expand on strong retrieval match
  auto_evict: true    # auto-collapse coldest when over budget
```

## Hybrid Safety Net

In both modes, VC enforces budget constraints and falls back to automatic management when the LLM doesn't manage. The LLM drives, VC enforces, like `madvise()` hints with kernel enforcement.

- LLM **requests**: "expand recipes, collapse deployment"
- VC **enforces**: won't exceed budget, auto-collapses coldest topic if LLM over-expands
- VC **provides**: budget numbers, token costs per topic, recency info
- VC **falls back**: if LLM never uses the tools, VC manages automatically

## Implementation Phases

### Phase 1: Types & Working Set State

**Files**: `types.py`, `engine.py`

New types:
- `DepthLevel` enum: NONE, SUMMARY, SEGMENTS, FULL
- `WorkingSetEntry` dataclass: tag, depth, tokens, last_accessed_turn
- `PagingConfig` dataclass: enabled, mode, auto_promote, auto_evict

Engine changes:
- `self._working_set: dict[str, WorkingSetEntry]` initialized from persisted state
- `EngineStateSnapshot.working_set: list[WorkingSetEntry]` for persistence
- Updated `_save_state()` and `_load_persisted_state()`
- Backward-compatible: old snapshots load with empty working set

**Tests**: ~6 tests (serialization round-trip, backward compat, entry creation on compaction)

### Phase 2: Multi-Depth Store Retrieval + Assembler

**Files**: `core/store.py`, `storage/sqlite.py`, `storage/filesystem.py`, `core/assembler.py`

Store:
- New `get_segments_by_tags(tags, min_overlap, limit) -> list[StoredSegment]`
- SQLite: same JOIN as `get_summaries_by_tags` but returns full `StoredSegment`
- Filesystem: same filter but returns full objects

Assembler:
- `assemble()` gains `working_set` and `full_segments` parameters
- Per-tag depth-aware formatting:
  - NONE: skip (hint only)
  - SUMMARY: current behavior (`_format_tag_section`)
  - SEGMENTS: new `_format_segments_section` (each segment summary separately)
  - FULL: new `_format_full_section` (`StoredSegment.full_text`)
- When `working_set` is None, current behavior preserved (backward compat)

**Tests**: ~10 tests (each depth level, budget enforcement, backward compat)

### Phase 3: Engine expand/collapse API + Budget Management

**Files**: `engine.py`

Public API:
```python
def expand_topic(self, tag: str, depth: str = "full") -> dict
def collapse_topic(self, tag: str, depth: str = "summary") -> dict
def get_working_set_summary(self) -> dict
```

Budget enforcement in `expand_topic()`:
1. Calculate token cost at requested depth
2. If over budget, auto-evict coldest tags (LRU by `last_accessed_turn`)
3. If still over after eviction, refuse and return error
4. Update working set, persist state

Integration with `on_message_inbound()`:
- Tags in working set at depth > SUMMARY: fetch data from store at that depth
- Pass working set + full segments to assembler
- Update `last_accessed_turn` for matched tags

Auto-promotion: when retriever finds strong match for a SUMMARY-depth tag, promote to SEGMENTS.

**Tests**: ~12 tests (expand/collapse, budget, auto-eviction, auto-promotion, persistence)

### Phase 4: Context Hint Modes (Supervised / Autonomous)

**Files**: `engine.py`, `config.py`, `types.py`

Config: `paging` section with `enabled`, `mode`, `auto_promote`, `auto_evict`.

`_build_context_hint()` updated for two modes:
- Supervised: topic list with "call expand_topic to see full detail"
- Autonomous: full budget dashboard with token costs and available tools

Auto mode: model name from upstream request → supervised or autonomous.

**Tests**: ~6 tests (hint format per mode, auto mode resolution)

### Phase 5: MCP Tools

**Files**: `mcp/server.py`

Two new tools via FastMCP:
- `expand_topic(tag, depth)`: calls `engine.expand_topic()`
- `collapse_topic(tag, depth)`: calls `engine.collapse_topic()`

**Tests**: ~4 tests (tool registration, expand/collapse via MCP, error handling)

### Phase 6: Proxy Tool Interception (Live MCP)

**Files**: `proxy/server.py`

The proxy intercepts `tool_use` blocks in the LLM's streaming response, fulfills them from the engine, and injects `tool_result` back into the conversation, all within a single client-visible request. The client sees one seamless response; the LLM gets MCP-equivalent tool access without the client needing MCP support.

Flow:
```
Client request → Proxy enriches (context hint lists available tools)
    → Forward to upstream LLM
    → LLM streams response with tool_use block for expand_topic("recipes")
    → Proxy intercepts tool_use, pauses streaming to client
    → Proxy calls engine.expand_topic("recipes")
    → Proxy sends tool_result back to upstream LLM (new API call)
    → LLM continues response with expanded context
    → Proxy resumes streaming to client
    → Client receives one continuous response
```

This is "Live MCP": every proxy-connected client gets tool access for free. No MCP configuration, no client-side tool handling, no extra user turns. The proxy acts as both MCP server and client in a single hop.

Implementation details:
- SSE stream parser detects `content_block_start` with `type: "tool_use"` and `name` matching VC tools
- Accumulate tool input JSON from `content_block_delta` events
- On `content_block_stop`, execute the tool synchronously
- Construct a `tool_result` message and make a continuation request to upstream
- Resume forwarding the continuation response to the client
- Handle nested tool calls (LLM might call expand then collapse in one turn)
- Error handling: if tool execution fails, inject an error tool_result so the LLM can recover gracefully

Dashboard integration:
- Tool calls shown in request grid with tool name and result
- Turn inspector shows intercepted tool calls with timing

**Depends on**: Phases 1-5 (expand/collapse API must exist)

## What Stays the Same

- Tagging pipeline (LLM tagger, embedding tagger, vocabulary feedback)
- Compaction triggers (soft/hard thresholds, monitor)
- Segmenter (turn pairing, tag grouping)
- Compactor (LLM summarization, concurrent execution)
- Storage schema (full text already stored in `StoredSegment.full_text`)
- Tag summary rollup (greedy set cover)
- Tag canonicalization and splitting

## What Changes

| Component | Today | With Paging |
|-----------|-------|-------------|
| `engine.py` | Stateless per-turn retrieval | + working set state, expand/collapse API |
| `core/retriever.py` | Returns tag summaries | + returns at requested depth level |
| `core/assembler.py` | Injects tag summaries | + injects at depth (summary/segments/full) |
| `types.py` | -- | + DepthLevel, WorkingSetEntry, PagingConfig |
| `core/store.py` | `get_summaries_by_tags()` | + `get_segments_by_tags()` (full text) |
| `proxy/server.py` | -- | + tool call interception (Phase 6) |
| `mcp/server.py` | 3 tools | + `expand_topic`, `collapse_topic` |
| Storage | Already stores full text | No change |
| Tagging | Unchanged | Unchanged |
| Compaction | Unchanged | + initializes working set entries |
| Monitor | Unchanged | + accounts for working set token usage |

## Verification

1. **Unit tests per phase**: each phase has its own test suite
2. **Integration**: proxy + dashboard, verify context hint shows depth info
3. **MCP**: `expand_topic` / `collapse_topic` tools work end-to-end
4. **Stress test**: headless replay with compact-test config, working set persists across compactions
5. **Backward compat**: 585+ existing tests pass (paging disabled by default)
6. **Phase 6**: send a message where the LLM would want more detail, verify proxy intercepts and fulfills tool call transparently
