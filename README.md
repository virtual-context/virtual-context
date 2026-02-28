<p align="center">
  <img src="assets/hero.png" alt="virtual-context" width="100%">
</p>

# virtual-context

**Your LLM never forgets. Even in a 500-turn conversation.**

virtual-context orchestrates a layered pipeline of LLM inference, embedding similarity, deterministic heuristics, and algorithmic rules (each compensating for the others' blind spots) to maintain a living, compressed memory of unbounded conversations.

LLMs have fixed context windows. When conversations grow long, most systems do one of two things: silently drop your oldest messages, or embed everything into a vector database and hope cosine similarity finds what matters. Both fail in predictable ways. The architecture decision from turn 12 vanishes when turn 80 arrives. The legal filing deadline gets evicted because the user asked about dinner recipes. A vague question like "what did we discuss earlier?" returns nothing because it doesn't embed close to anything specific.

virtual-context takes a fundamentally different approach. It treats LLM context the way an operating system treats RAM: tagging every exchange by topic, compressing intelligently, and paging in the right context exactly when needed. Overview queries can load everything via `vc_recall_all`/`recall_all`. Time-scoped queries use `vc_remember_when`/`remember_when` with backend-resolved date windows. Tag overlap and IDF scoring surface the right segment even when the user's vocabulary doesn't match the original discussion. A two-tagger architecture ensures the system can never hallucinate irrelevant topics into your context window.

```
Layer 0: Raw conversation turns              (active memory, in the context window)
Layer 1: Segment summaries per tag           (compressed pages, per-topic summaries)
Layer 2: Tag summaries via greedy set cover   (working set descriptors, bird's-eye view)
```

The result: an LLM that recalls details from turn 12 at turn 200 with the same fidelity as if the conversation just started.

### Configurable Context Ceiling

Most teams set `context_window` to whatever the model supports — 128K, 200K — and let it fill up. This is expensive and, counterintuitively, degrades quality. Research on "lost in the middle" shows that LLM attention degrades in long contexts: facts buried in 120K tokens of raw history are missed more often than the same facts concentrated in a managed 30K window.

virtual-context lets you set an artificial context ceiling well below the model's maximum:

```yaml
context_window: 30000  # Run a 200K model at 30K
```

The compression hierarchy (raw turns → segment summaries → tag summaries) keeps the window within this budget. When the ceiling is hit, compaction fires: stale turns are summarized, facts are extracted and indexed, and the working set reshapes around what's active. The result is a smaller, denser context where every token carries signal.

**Cost impact:** A 200K-capable model running at 30K uses ~85% fewer input tokens per request. Over thousands of requests, this is the difference between a viable product and a cost problem.

**Quality impact:** Concentrated context means the model's attention isn't spread across 120K tokens of mostly-stale history. Relevant facts surface through targeted retrieval and structured tools rather than hoping the model notices them buried in a long window. The managed context window becomes a feature, not a limitation.

## Virtual-Context vs RAG vs Compaction

These approaches are complementary, but optimize different failure modes.

| | RAG | Compaction-only | virtual-context |
|---|---|---|---|
| **Primary mechanism** | Query-time retrieval by embedding similarity | Summarize old history to fit window | Tagged memory + retrieval + compaction + paging tools |
| **What gets kept** | External documents + recent raw chat | Summaries of old turns + recent raw chat | Multi-layer memory (raw turns, segment summaries, tag summaries) |
| **Specific fact lookup** | Depends on embedding/query phrasing alignment | Lossy after summarization | `vc_find_quote` + `vc_query_facts` + summary/segment drill-down |
| **Broad overview ("what did we discuss?")** | Weak unless special orchestration | Can summarize, but often generic | `vc_recall_all` returns all topic summaries within budget |
| **Time-scoped recall ("last week", "between June and July")** | Custom logic outside core RAG | Requires date fidelity in summaries | `vc_remember_when` with backend-resolved time ranges |
| **Vocabulary mismatch tolerance** | Embedding-dependent | Low | Related-tag expansion + IDF-weighted ranking + quote search fallback |
| **Context budget control** | Append retrieved chunks | Compression with limited selective rehydration | Explicit paging: expand/collapse topics and bounded assembly |
| **Cost at scale** | Grows with corpus size (more chunks retrieved) | Grows with conversation length (summaries accumulate) | Configurable ceiling: run a 200K model at 30K, ~85% fewer input tokens |
| **Interpretability** | Medium (scores/chunks) | Low-medium (summary quality dependent) | High (tags, tool calls, budgets, sections, stored summaries) |
| **Failure mode** | Miss relevant chunk | Over-compress / lose detail | Requires tool-aware prompting + memory hygiene |
| **Best fit** | Knowledge/doc retrieval | Simple long-chat cost reduction | Long-running agent memory with mixed query types |

virtual-context combines retrieval and compaction, then adds explicit tools for overview/time/fact recall under strict token budgets.

## Install

```bash
pip install virtual-context
```

One-command installers (OpenClaw-style):

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/virtual-context/virtual-context/main/scripts/install.sh | bash
```

```powershell
# Windows PowerShell
iwr https://raw.githubusercontent.com/virtual-context/virtual-context/main/scripts/install.ps1 -useb | iex
```

Guided setup (interactive wizard → config + proxy instances + daemon):

```bash
virtual-context onboard --wizard
```

The wizard walks through: tagging provider/model selection → inbound tagger mode (embedding/LLM/keyword) → proxy instances (upstream provider, port, label per instance) → per-instance config generation with isolated storage → optional daemon install. Each instance gets a standalone YAML config pointing to its own SQLite DB.

Or non-interactive:

```bash
virtual-context onboard
virtual-context onboard --install-daemon --upstream https://api.anthropic.com
```

Daemon setup docs (macOS `launchd`, Linux `systemd --user`, Windows Task Scheduler): [`docs/install.md`](docs/install.md)

Optional extras:

```bash
pip install virtual-context[tui]         # interactive chat terminal
pip install virtual-context[bridge]      # HTTP proxy (FastAPI + uvicorn)
pip install virtual-context[embeddings]  # sentence-transformers tag generator
pip install virtual-context[tiktoken]    # exact token counting
pip install virtual-context[mcp]         # Model Context Protocol server
pip install virtual-context[all]         # everything
```

Minimal dependencies: `pyyaml` + `httpx`. Python 3.11+.

Two hooks into your LLM pipeline. Pick whichever integration fits:

**Option A: HTTP Proxy (zero code changes).** Point your existing LLM client at `localhost:5757` instead of the upstream API. The proxy handles everything transparently (inbound tagging, retrieval, history filtering, response tagging, compaction). Works with any client that speaks OpenAI, Anthropic, or Gemini API format. Includes a [live dashboard](#live-dashboard) for real-time monitoring and tuning.

```bash
virtual-context proxy --upstream https://api.anthropic.com
# Then change your client's base_url to http://127.0.0.1:5757
```

**Option B: Python SDK.** Two function calls wrap your existing LLM pipeline:

```python
from virtual_context import VirtualContextEngine, Message

engine = VirtualContextEngine(config_path="./virtual-context.yaml")

# BEFORE sending to LLM - retrieve relevant stored context
assembled = engine.on_message_inbound(
    message="What was the Henninger filing deadline?",
    conversation_history=messages,
)
# assembled.prepend_text → enriched system prompt with retrieved summaries
# assembled.matched_tags → ["legal", "filing"]
# For time-scoped recall, call vc_remember_when(query, time_range)
# For broad overviews, call the recall-all tool (vc_recall_all / recall_all)

# AFTER LLM responds - tag, index, compact if needed
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
Session routing (proxy mode)
    │  ├─ Extract session ID from <!-- vc:session=UUID --> markers in assistant messages
    │  ├─ Route to existing session or load persisted state from store
    │  ├─ No marker? → reuse default session (first request) or create new
    │  └─ Strip session markers before forwarding to upstream
    │
    ▼
Strip client envelope (OpenClaw metadata, channel headers, plugin markers)
    │
    ▼
History ingestion (first request only)
    │  └─ Extract and tag all prior user+assistant pairs → bootstrap TurnTagIndex
    │
    ▼
Inbound tagging - identify what this message is about
    │  ├─ Embedding tagger (recommended): cosine similarity against existing tag vocabulary
    │  │   (closed-set, deterministic, can't hallucinate novel tags)
    │  ├─ LLM / keyword tagger: alternative with vocabulary feedback
    │  ├─ Tag canonicalization: "db" → "database", alias detection via edit distance
    │  └─ Temporal detection: regex + LLM flags for time-referencing queries
    │
    ▼
Retrieve matching summaries from store
    │  ├─ Recall-all tool call? → load ALL tag summaries (bounded by token budget)
    │  ├─ Temporal query? → load segment summaries sorted earliest-first (Layer 1)
    │  ├─ Query expansion: primary tags + related tags widen the search
    │  ├─ Overfetch 3x → IDF-weighted re-rank (rare tag matches score higher)
    │  ├─ FTS fallback: if tag overlap finds nothing, full-text search on stored segments
    │  └─ Deep retrieval: full stored segment fetch for top matches
    │
    ▼
Assemble context within token budget
    │  ├─ Context hint: lightweight <context-topics> block (~50-200t)
    │  └─ Tag sections: retrieved summaries ordered by tag priority
    │
    ▼
Filter conversation history
    │  ├─ Drop turns whose tags don't overlap with inbound tags
    │  ├─ Preserve tool chains atomically (tool_use ↔ tool_result never separated)
    │  ├─ Protect recent turns (always kept regardless of tags)
    │  └─ Temporal queries skip filtering entirely
    │
    ▼
Inject <virtual-context> block → forward enriched request to LLM
    │
    ▼
LLM processes enriched context → produces response
    │
    ▼
Inject session marker into response (proxy mode)
    │  ├─ Streaming: emit final SSE delta with <!-- vc:session=UUID -->
    │  └─ Non-streaming: append marker to last text content block
    │
    ▼
Response tagging - LLM tags the full user+assistant pair (background thread)
    │  ├─ Context lookback: feed N recent pairs as tagger context for short/ambiguous messages
    │  ├─ Context bleed gate: embedding similarity blocks stale context on topic shifts
    │  ├─ Retry on _general: if tagger returns only _general, retry with expanded context
    │  ├─ Authoritative tags written to TurnTagIndex (vocabulary-building)
    │  ├─ Fact signal extraction: lightweight subject/verb/object triples per turn
    │  ├─ Related tags generated for cross-vocabulary retrieval
    │  └─ Compactor generates related_tags at write time (vocabulary bridging)
    │
    ▼
Check token thresholds (soft 70%, hard 85%)
    │
    ▼ (if threshold exceeded)
Segment by tag → summarize each segment (concurrent, ThreadPoolExecutor)
    │  ├─ Session dates: forced segment splits on session boundaries
    │  ├─ Tags preserved: LLM can ADD refined/related tags but never REMOVE originals
    │  ├─ Fact consolidation: per-turn fact signals → structured Fact records with provenance
    │  └─ Related tags written into stored segments for future cross-vocabulary retrieval
    │
    ▼
Compute greedy set cover → build/update per-tag summaries (Layer 2)
    │
    ▼
Persist engine state (TurnTagIndex + compaction watermark → store)
```

## Key Capabilities

### Tags Emerge From Conversation

There are no predefined domains to configure. An LLM tagger reads each turn and generates semantic tags (`database`, `auth`, `fitness`, `legal`) that naturally converge over the session. A vocabulary feedback loop passes known tags back into the tagger prompt, so it reuses `storage` instead of inventing `data-persistence` or `file-management`.

The same codebase handles legal briefs, medical notes, coding sessions, recipe planning, and marathon training, whatever the user talks about. Tag rules let you configure priority, TTL, and custom summary prompts per tag family using fnmatch patterns.

### Two-Tagger Architecture

virtual-context separates tagging from fact extraction, and splits tagging itself into two distinct operations with different objectives. Most memory systems go straight from raw text to fact/knowledge extraction in a single LLM call, processing each chunk independently with no surrounding context. virtual-context treats these as separate concerns, each with its own extraction strategy and context window.

**Inbound tagger** (embedding, runs before LLM responds): Uses sentence-transformers (`all-MiniLM-L6-v2`) to compute cosine similarity between the user's message and the existing tag vocabulary. Closed-set: it can only return tags that already exist in the TurnTagIndex. Deterministic, subsecond, zero LLM cost. Because it can only match existing tags, it is structurally incapable of hallucinating novel topics into your context window.

**Response tagger** (LLM, runs after LLM responds): Sees the full user+assistant turn pair plus N recent preceding turns (configurable via `context_lookback_pairs`, default 5) as surrounding context. A context bleed gate (embedding similarity threshold) prevents stale context from a previous topic from leaking in on topic shifts. This is the creative, vocabulary-building pass: inventing new tags when new topics emerge, generating related tags for cross-vocabulary retrieval, and extracting per-turn fact signals. Runs in a background thread so it never blocks the next request.

**Context-aware extraction.** The response tagger doesn't process turns in isolation. When the user says "yes, that one" or "can you expand on that?", a tagger seeing only that message has nothing to work with. By feeding surrounding turn pairs with bleed gating, the tagger correctly classifies ambiguous messages and extracts meaningful fact signals even from short, context-dependent replies. If the tagger still returns only `_general`, it retries with expanded context before falling back to tag inheritance from the TurnTagIndex.

The inbound tagger drives retrieval and filtering (what stored context to inject, which history turns to keep). The response tagger drives the permanent record (what tags describe this turn, and what facts it contains, for all future queries). Each tagger is optimized for its task: the inbound tagger prizes safety (never contaminate the context), the response tagger prizes richness (capture every nuance for future recall).

### Broad Overview Tool (`vc_recall_all`)

"What did we discuss earlier?" "Can you summarize everything?" "What did you say about image storage?"

These queries don't map cleanly to specific tags. virtual-context uses an MCP-style tool call:

- `vc_recall_all` (proxy tool loop) / `recall_all` (MCP server) loads **all** tag summaries
- Results are bounded by the configured tag-context token budget
- The reader can follow up with `vc_expand_topic` on specific tags for deeper detail

This eliminates the failure mode where the LLM says "I don't recall discussing that" about something from 50 turns ago.

### Time-Scoped Recall Tool (`vc_remember_when`)

"Going back to the very beginning, what were the key decisions?" "What did we set up with tokens at the start?" "Between June and July, what changed about indexing?"

These queries reference a *position in time*, not just a topic. virtual-context uses an explicit tool call:

- `vc_remember_when` (proxy tool loop) / `remember_when` (MCP server) combines semantic query + structured time range
- Time ranges use relative presets (e.g. `last_week`, `last_month`) or explicit date bounds (`between_dates`)
- Date math is backend-resolved, not LLM-resolved, so results are deterministic and testable

This solves a fundamental problem with summarization: when a tag like `project-structure` appears at turn 1, turn 57, and turn 71, a merged tag summary blends all three. A time-scoped query about "the very first thing we discussed" needs constrained retrieval against early sessions, not a generic merged blob.

### Session Date Propagation

Temporal reasoning requires knowing *when* each piece of information was recorded. virtual-context propagates session dates through the entire pipeline:

```
[Session from 2023/05/25] in user message
    → TurnTagEntry.session_date
    → forced segment split on session change
    → SegmentMetadata.session_date
    → SQLite metadata_json
    → find_quote results: {"session": "2023/05/25"}
    → assembled context: <virtual-context session="2023/05/25">
```

The segmenter forces a new segment boundary whenever the session date changes, even if the primary tag is the same. This guarantees no segment spans multiple sessions. When the reader sees two conflicting facts — "sneakers under my bed" (session 2023/05/25) and "moved sneakers to shoe rack" (session 2023/05/29) — it can determine temporal ordering and answer correctly.

For proxy/OpenClaw conversations, session dates come from `Message.timestamp` instead of text headers. Same pipeline, same temporal reasoning capability.

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

This costs ~50-200 tokens and enables a natural drill-down loop: the user asks for an overview, the LLM sees what's available, synthesizes or asks for clarification, and the next turn pulls full detail via narrow tag retrieval. When paging is enabled, the hint also includes tool usage rules and a budget indicator: use `vc_recall_all` for broad overviews, `vc_remember_when` for time-scoped recall, `vc_find_quote` for specific text (names, numbers, decisions), `vc_query_facts` for structured fact lookup (subject/verb/object filters with semantic expansion), `vc_expand_topic` for deeper understanding of a listed topic, and `vc_collapse_topic` to free budget.

### Structured Fact Extraction (`vc_query_facts`)

Summaries compress information but inevitably lose specific details. When the user says "I run 5K every morning" at turn 14, a summary might retain "runs regularly" but drop the exact distance and timing. Most memory systems extract facts in a single LLM pass and trust the output directly: raw text goes in, extracted facts come out, and those facts are stored as-is. virtual-context takes a fundamentally different approach with a two-phase pipeline where per-turn signals are treated as hints, not ground truth.

**Phase 1: Fact signals (per-turn).** The response tagger extracts lightweight subject/verb/object triples from each turn as it's processed, with full surrounding context (the same context lookback and bleed gating used for tagging). "I run 5K every morning" becomes `{subject: "user", verb: "runs", object: "5K every morning"}`. These are fast, cheap, and stored on the TurnTagIndex. Critically, they are not yet committed as permanent facts.

**Phase 2: Fact consolidation (at compaction).** When segments are compacted, per-turn fact signals are verified and consolidated into structured `Fact` records with the full multi-turn segment as context. The consolidation pass can see the complete conversation flow across multiple turns: what the user asked, how the assistant responded, what was clarified or corrected. This means a fact signal from turn 14 gets validated against turns 12-18 before becoming a permanent record. The result is a structured `Fact` with full provenance: subject, verb, what (the core assertion), temporal status (active/completed/planned/abandoned/recurring), associated tags, session ID, and source turn numbers. Facts are stored in dedicated SQLite tables with indexes for efficient querying.

**Why two phases matter.** A single-pass extractor processing "yes, let's go with PostgreSQL" in isolation has no idea what "yes" refers to. It might extract nothing, or hallucinate a fact. virtual-context's response tagger sees the surrounding turns ("Should we use PostgreSQL or MySQL for the user table?") and generates the correct signal. The consolidation pass then verifies it against the full segment before storing a permanent fact. Two chances to get it right, each with progressively more context.

**Querying.** The `vc_query_facts` tool (proxy tool loop) provides structured fact lookup with filters:

```
vc_query_facts(subject="user", verb="runs")
vc_query_facts(object_contains="5K")
vc_query_facts(status="active")
```

**Semantic verb expansion.** Queries like `verb="runs"` automatically expand to morphologically similar verbs in the database (e.g., "running", "run", "jogs") via sentence-transformer embedding similarity. This means the reader doesn't need to guess the exact verb form used during extraction.

**Semantic fact search.** When structured filters return sparse results, a fallback embedding search matches the query intent against all stored facts' `what` fields by cosine similarity, surfacing relevant facts even when the subject/verb/object decomposition doesn't align.

### Virtual Memory Paging

RAG retrieves content and appends it to the context window. It never frees space from what's already there. When a 100k document needs to enter a 120k window that already has 60k of conversation history, RAG has three options: truncate (lossy), error (useless), or chunk (every chunking approach either costs extra user turns, loses cross-chunk coherence, or both). Nobody touches the existing 60k. It sits there, potentially full of stale context from 30 turns ago that nobody needs anymore.

virtual-context treats the context window as managed memory. The three-layer compression hierarchy (raw turns, segment summaries, tag summaries) already stores data at every depth level. Paging makes this hierarchy bidirectional: topics can be expanded to full original detail or collapsed back to summaries, and the working set reshapes itself around whatever the user needs right now.

```
Tag summaries  <------->  Segment summaries  <------->  Full stored text
     ^                          ^                            ^
  collapse                   default                      expand
  (~200t)                  (~2,000t)                   (~8,000t+)
```

When the LLM needs more detail on a topic ("What was the exact sourdough timing?"), it expands that topic from summary to full text. When budget pressure hits, cold topics are automatically collapsed. A 100k document enters the window by collapsing 60k of stale conversation to 8k of summaries, freeing 52k. The working set (a per-session map of which topics are loaded at which depth) persists across turns, so expansion decisions are stateful: recipes stays expanded until explicitly collapsed or evicted by budget pressure.

**Model-tiered delegation.** Not all LLMs are equally capable of managing their own context. Weaker models (Haiku, small open-source) get a simplified topic list and can request expansions, but virtual-context handles all eviction decisions silently. Stronger models (Opus, Sonnet, GPT-4) see a full budget dashboard with token costs per topic, available budget, and depth levels, making explicit trade-off decisions. In both modes, virtual-context enforces budget constraints and falls back to automatic management when the LLM doesn't manage. The LLM drives, virtual-context enforces, like `madvise()` hints with kernel enforcement.

**Full-text search with semantic enrichment.** When tag-based retrieval misses (content filed under an unexpected topic, detail too specific for summaries), `find_quote` searches stored conversation text directly using two complementary strategies. FTS5 handles exact and partial keyword matches. Semantic search (segment text chunked into overlapping windows, embedded with sentence-transformers, matched by cosine similarity) surfaces paraphrased references that share no lexical overlap with the original text. Both run on every query — FTS results are supplemented with semantic matches to fill the result set. Each call returns a fixed top 20 results. Results include the matching excerpt, session date, match type, and all tags on the segment, so the LLM can chain into `expand_topic` for broader context.

**Working-set optimization.** Read-only tools (`find_quote`, `query_facts`, `recall_all`, `remember_when`) skip the expensive context reassembly step. Only `expand_topic` and `collapse_topic` (which change the working set) trigger a full context rebuild. This reduces per-round overhead in tool chains that make multiple read-only calls before expanding.

**Tool loop.** The reader model can chain multiple tool calls within a single turn. After `find_quote` returns a result, the reader can issue another `find_quote` with a refined query, `query_facts` for structured lookup, or follow up with `expand_topic`. Up to 10 continuation rounds run transparently within one client-visible request (configurable via `paging.max_tool_loops`). This is essential for multi-fact questions: "What is the total number of days I spent in Japan and Chicago?" requires two independent `find_quote` calls to locate each trip's details before computing the sum.

**Budget-aware reader prompting.** The context hint tells the reader exactly how many tool rounds it has, encouraging strategic tool use: "You have a maximum of N tool rounds. Plan your strategy upfront: use diverse queries, not repetitions. If a search already returned the answer, stop and respond." This prevents the reader from exhausting all rounds on redundant searches when the answer was found on the first call.

**Multi-provider tool loop.** The tool loop supports Anthropic, OpenAI, and Gemini as reader models via a `ProviderAdapter` pattern. Each adapter handles provider-specific request/response formats, tool call parsing, and context injection. The reader model can be different from the upstream provider — e.g., use GPT-5 Codex as the reader with an Anthropic upstream, or Gemini as the reader with an OpenAI upstream.

**Resilient continuation.** When the tool loop exhausts all rounds and forces a final text-only continuation, transient HTTP errors (server 500s) are retried once with a brief delay before falling back to error state. This prevents a single upstream hiccup from discarding an otherwise complete answer.

**Live MCP via proxy.** The proxy intercepts `tool_use` blocks in the LLM's streaming response, fulfills `vc_recall_all`, `vc_remember_when`, `vc_expand_topic`, `vc_collapse_topic`, `vc_find_quote`, and `vc_query_facts` calls from the engine, and injects `tool_result` back into the conversation, all within a single client-visible request. The LLM can chain tools within one turn (e.g. `vc_recall_all` → `vc_query_facts` → `vc_find_quote` → `vc_expand_topic`), using up to 10 continuation loops transparently. Every proxy-connected client gets MCP-equivalent tool access with zero configuration, zero client-side changes, and zero extra user turns.

### Three-Layer Memory Hierarchy

**Layer 0: Raw turns.** The live conversation in the context window. Protected recent turns are never compacted.

**Layer 1: Segment summaries.** When token pressure hits thresholds, consecutive same-tag turns are grouped and summarized by an LLM. Each segment preserves key decisions, entities, specific names, and action items. Original tags are never lost; the LLM can add tags during summarization but never remove them. Structured facts (subject/verb/object triples with temporal status) are extracted during compaction and stored separately for precise querying via `vc_query_facts`.

**Layer 2: Tag summaries.** A greedy set cover algorithm finds the minimum set of tags that covers every turn. For each cover tag, all segment summaries are rolled up into a single tag-level summary. A focused session might produce 3 cover tags; a sprawling multi-topic session produces 10+. Only stale summaries (where new segments exist since the last build) are recomputed.

### Two-Tier Compaction

Compaction mirrors OS page replacement:

- **Soft threshold (70%)**: proactive compaction. Summarize now while there's headroom.
- **Hard threshold (85%)**: mandatory compaction. Summarize immediately or the context window overflows.

Compaction is greedy-batch: everything between the watermark and the protected zone gets compacted in one pass, so it fires infrequently (one big batch instead of many small ones). Summarization runs concurrently via ThreadPoolExecutor, with order-preserving results, per-tag custom prompts, and per-segment progress logging. The summary prompt preserves exact numbers, proper nouns, and state assertions (e.g., "I now store sneakers on the shoe rack" is never softened to "plans to store").

### Tag Canonicalization

Tags naturally produce synonyms: `db`, `database`, `data-storage`. The TagCanonicalizer detects aliases via edit distance and normalizes them automatically. You can also register aliases manually:

```bash
virtual-context aliases suggest    # auto-detect potential aliases
virtual-context aliases add db database
```

### Automatic Tag Refinement

When a tag appears on too many turns (crossing configurable frequency thresholds), it loses discriminative power: proxy filtering keeps all matching turns, pulling unrelated history. virtual-context detects these overly-broad tags and automatically refines them.

An LLM pass examines all turns under the broad tag and determines whether they span distinct sub-topics. If they do, the tag is split into specific compound sub-tags. If the content is genuinely uniform (one topic that happens to be frequent), a tag summary is built instead. Each tag is only processed once; the result is persisted so split analysis doesn't re-trigger.

**Production example** (143-turn OpenClaw session):

`reservation-request` appeared on 43/143 turns (30.1%), spanning platform debugging, availability searches, browser session management, and general booking discussion. The splitter broke it into four sub-tags:

| Sub-tag | Turns | Content |
|---------|-------|---------|
| `reservation-platform-troubleshooting` | 11 | Debugging OpenTable/Resy platform issues |
| `reservation-availability-search` | 5 | Checking time slots and availability |
| `reservation-browser-access` | 4 | Getting logged-in browser sessions |
| `reservation-general` | 20 | General booking coordination |

`troubleshooting` appeared on 34/143 turns (23.8%), spanning browser connectivity issues, restaurant lookups, booking platform interaction, and credential access. Split into `browser-connection-troubleshooting` (11), `restaurant-lookup-troubleshooting` (7), `booking-platform-troubleshooting` (7), `credential-access-troubleshooting` (7).

This is the second emergent property of the system. Vocabulary convergence (the first) naturally collapses synonyms into canonical tags. Tag splitting pushes unrelated concepts apart. Together they create a two-sided pressure (convergence pulls related concepts together, splitting pushes unrelated concepts apart) and the vocabulary evolves toward maximum discriminative power without manual curation.

### Emergent Behaviors

- **Vocabulary convergence**: Tag reuse and canonicalization naturally collapse synonyms into stable tag vocabularies over long sessions.
- **Automatic tag refinement**: High-frequency broad tags split into narrower sub-tags, increasing retrieval precision without manual taxonomy work.
- **Tool-first recall loops**: Models tend to converge on `vc_find_quote`/`vc_query_facts`/`vc_recall_all`/`vc_remember_when` → `vc_expand_topic` sequences for multi-step recall.
- **Quote-then-context chaining**: Exact snippets from `find_quote` naturally route follow-up expansion to the right topic context.
- **Fact-then-quote verification**: Structured facts from `query_facts` provide quick answers; the reader chains into `find_quote` to verify or locate the original conversational context.
- **Session-date anchoring**: Time-scoped recall (`vc_remember_when`) biases responses toward chronology-correct evidence.
- **Vocabulary entropy reduction**: Canonicalization + tag feedback lowers random tag drift and improves cross-turn consistency.
- **Budget-shaped recall selection**: Budget-aware assembly consistently favors high-value context under tight token ceilings.
- **Compaction survivorship effects**: Frequently reinforced facts stay highly retrievable, while low-signal details trend toward summary-level recall.
- **Semantic verb bridging**: Verb expansion at query time lets the reader find facts regardless of morphological form — "runs" finds facts stored as "running", "jogging", "exercises".

Split tags are registered as aliases via TagCanonicalizer, so historical queries against the old tag still resolve. New sub-tags enter the vocabulary feedback loop immediately. The splitter never reuses existing tag names (which would cause cascading splits); it always creates new compound tags.

```yaml
tag_generator:
  tag_splitting:
    enabled: true
    frequency_threshold: 15       # min absolute turn count
    frequency_pct_threshold: 0.15  # min fraction of total turns
    max_splits_per_turn: 1        # max tags to split per on_turn_complete cycle
```

### Cross-Vocabulary Retrieval

Users don't use the same words every time. A discussion about "materialized views for feed performance" at turn 46 might be recalled as "that caching trick for the feed" at turn 71. Pure tag overlap finds nothing; the vocabularies are completely disjoint.

virtual-context solves this with two complementary mechanisms:

**Related tag expansion.** Both the tagger (query-side) and compactor (write-side) generate `related_tags` (alternate terms someone might use to refer to the same concepts). A segment about "materialized views" gets stored with related tags like `caching`, `precomputed`, `feed-optimization`. A query about "caching trick" generates related tags that overlap with stored segments. The retriever expands its search to include both primary and related tags.

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

This budget enforcement is what makes the configurable context ceiling work in practice. A 30K ceiling doesn't mean losing information — it means the assembler is forced to prioritize, and the compression hierarchy ensures everything is still available at some depth level. The model reasons over a dense, curated context instead of a sprawling raw history.

## Configuration

Create `virtual-context.yaml` in your project root:

```yaml
version: "0.2"
storage_root: ".virtualcontext"
context_window: 30000    # intentionally below model max — see "Configurable Context Ceiling"

# Tags emerge from conversation - the LLM generates them
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
  tag_splitting:                              # auto-refine overly-broad tags
    enabled: true
    frequency_threshold: 15                   # min absolute turn count
    frequency_pct_threshold: 0.15             # min fraction of total turns

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
virtual-context presets list   # see all available presets
virtual-context presets show coding  # dump a preset's config as YAML
```

## Three Tag Generators

**LLM tagger** (recommended for response tagging): Uses any local model via Ollama, LM Studio, or vLLM. Generates rich semantic tags with temporal query detection and related tag generation. Vocabulary feedback ensures convergence: the tagger sees all existing tags and reuses them instead of inventing synonyms. Falls back to keyword tagger if the LLM is unavailable. This is the creative, vocabulary-building tagger that runs after the LLM responds.

**Keyword tagger**: Deterministic regex and keyword matching. Zero latency, zero cost, fully reproducible. Good for domains with well-defined vocabularies where you don't want LLM variability.

**Embedding tagger** (recommended for inbound tagging): Uses sentence-transformers (`all-MiniLM-L6-v2`) to compute cosine similarity against the existing tag vocabulary. Closed-set by design: it can only return tags that already exist, making it impossible to hallucinate novel tags that contaminate retrieval. Understands semantic relationships ("font-weight" matches `css`, "deadlift form" matches `fitness`) without needing exact keyword overlap.

## CLI

```bash
virtual-context status                         # tag stats and token usage
virtual-context tags                           # list all tags with counts
virtual-context domains                        # all tags with turn counts and summaries
virtual-context recall auth                    # retrieve stored summaries for a tag
virtual-context compact -i msgs.json           # manual compaction from message file
virtual-context retrieve -m "What about auth?" # tag + retrieve (JSON output)
virtual-context transform -m "What about auth?"# tag + retrieve + assemble
virtual-context aliases list                   # show all tag aliases
virtual-context aliases suggest                # auto-detect potential aliases
virtual-context aliases add db database        # register alias manually
virtual-context proxy -u https://api.anthropic.com  # single-instance proxy
virtual-context proxy                               # multi-instance (from config)
virtual-context presets list                   # list available config presets
virtual-context presets show coding            # dump preset config as YAML
virtual-context daemon status                  # service status (platform-specific)
virtual-context daemon start                   # start/enable daemon
virtual-context daemon stop                    # stop daemon
virtual-context daemon restart                 # stop + start daemon
virtual-context daemon uninstall               # remove daemon definition
virtual-context config validate                # check config syntax
virtual-context cost-report                    # show session LLM usage
```

## Interactive Chat (TUI)

```bash
pip install virtual-context[tui]
virtual-context chat --config virtual-context.yaml
```

A terminal chat interface with live context visualization, useful for development, testing, and seeing exactly what virtual-context does at each turn:

- **Tag panel**: current tag working set with activity levels, updated live as `on_turn_complete` processes each turn
- **Budget bar**: real-time token usage breakdown (core, tags, hint, conversation)
- **Turn list**: every turn with its tags, navigable with Ctrl+B/F
- **Turn inspector** (Ctrl+I): full turn data: API payload, tags, assembled context, and tool activity
- **Brief mode** (Ctrl+T): silently appends "answer in 2 lines" for faster iteration during testing
- **Manual compaction**: type `/compact` or press Ctrl+K to trigger compaction on demand
- **Session export** (Ctrl+S): saves full session to `vc-session.json` with all metadata

### Headless Mode

Run prompts through the full pipeline without a terminal UI, ideal for automated stress testing and regression validation:

```bash
virtual-context chat --headless --replay prompts.txt
```

Session JSON captures every turn with tags, token counts, and timing. Replay a saved session to test behavior changes against recorded conversations:

```bash
virtual-context chat --replay vc-session.json
```

## Integrations

### HTTP Proxy

```bash
pip install virtual-context[bridge]
```

The fastest path to production. The proxy sits between any LLM client and an upstream provider, running the full virtual-context pipeline on every request. The client just changes its `base_url`. No SDK integration, no code changes, no plugin required.

Start the proxy:

```bash
# Anthropic upstream
virtual-context -c virtual-context.yaml proxy --upstream https://api.anthropic.com

# OpenAI upstream
virtual-context -c virtual-context.yaml proxy --upstream https://api.openai.com --port 8080

# Custom host/port
virtual-context -c virtual-context.yaml proxy -u https://api.anthropic.com --host 0.0.0.0 --port 9090
```

**Multi-instance mode.** Run multiple proxy listeners on different ports, each forwarding to a different upstream provider. Configure in YAML instead of CLI flags:

```yaml
proxy:
  instances:
    - port: 5757
      upstream: https://api.anthropic.com
      label: anthropic
    - port: 5758
      upstream: https://api.openai.com/v1
      label: openai
    - port: 5760
      upstream: https://generativelanguage.googleapis.com
      label: gemini
```

```bash
virtual-context -c virtual-context.yaml proxy
# No --upstream needed; instances are read from config
# Each port gets its own dashboard showing the instance label
```

By default, all instances share the same `VirtualContextEngine`, `ProxyMetrics`, and storage backend.

**Per-port config.** When different instances need different tagging providers, summarization models, or isolated storage, add a `config` field pointing to a standalone config file:

```yaml
proxy:
  instances:
    - port: 5757
      upstream: https://api.anthropic.com
      label: anthropic
      config: ./virtual-context-proxy-anthropic.yaml
    - port: 5758
      upstream: https://api.openai.com/v1
      label: openai
      config: ./virtual-context-proxy-openai.yaml
```

Each instance config is a full standalone config with its own storage path:

```yaml
# virtual-context-proxy-anthropic.yaml
version: '0.2'
storage_root: .virtualcontext/anthropic
tag_generator:
  type: llm
  provider: anthropic
  model: claude-haiku-4-5-20251001
storage:
  backend: sqlite
  sqlite:
    path: .virtualcontext/anthropic/store.db
```

Instances with a `config` field get their own `VirtualContextEngine` and isolated storage. Instances without `config` share the master engine. The `onboard --wizard` flow generates these per-instance config files automatically.

Point your client at the proxy:

```python
# Python (anthropic SDK)
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:5757")

# Python (openai SDK)
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5757/v1")

# curl
curl http://127.0.0.1:5757/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-haiku-4-5-20251001","max_tokens":256,"messages":[{"role":"user","content":"Hello"}]}'
```

**Session continuity.** The proxy injects an invisible `<!-- vc:session=UUID -->` marker into every assistant response. Client SDKs store this as part of the message. On subsequent requests, the proxy extracts the marker, routes to the correct session, and strips markers before forwarding upstream. If the proxy restarts, it loads persisted engine state (TurnTagIndex + compaction watermark) from the store, so no re-ingestion is needed. Multiple concurrent conversations are routed independently via a session registry.

**Session suppression.** When a session has no compacted data (no tag summaries, no stored segments), the full virtual-context pipeline is suppressed — no context injection, no tool definitions, no history filtering. The request passes through as-is with minimal overhead. Once the first compaction runs and summaries exist, the pipeline activates automatically.

**Dynamic session discovery.** A `vc_find_session` tool is injected into the reader's tool definitions only when multiple sessions exist in the store. This lets the reader discover and reference prior sessions when answering cross-session questions, without cluttering the tool set in single-session conversations.

**History ingestion.** On the first request, the proxy extracts user+assistant pairs from the client's existing conversation history and tags each to bootstrap the TurnTagIndex. No cold-start period; the tag vocabulary is immediately available for inbound matching.

**Format-agnostic.** Auto-detects Anthropic, OpenAI, and Gemini request formats via a `PayloadFormat` strategy pattern and injects context accordingly: into `system` for Anthropic, into `messages[0]` for OpenAI, into `system_instruction.parts` for Gemini. Paging tool interception works across Anthropic and Gemini formats (`tool_use`/`tool_result` and `functionCall`/`functionResponse` respectively).

**Streaming with zero added latency.** SSE streams are forwarded byte-for-byte as they arrive from upstream. Text deltas are accumulated in the background for response tagging. The user sees no delay.

**Error-resilient.** If the engine fails (config error, tagger timeout, etc.), the request is forwarded to upstream unmodified. The proxy never blocks your LLM calls.

**Envelope stripping.** Strips client metadata (channel headers, message footers, event lines, plugin markers) so the tagger sees clean conversational content. Handles consecutive user messages from Telegram-style batching gracefully.

**Terminal logging.** Every event is logged to stdout in real-time:

```
[INGEST] 43 turns in 45123ms (session=a1b2c3d4e5f6)
[T44] POST anthropic stream=True tags=[css, design] msgs=52 dropped=25 ctx=312t input=8421t vc=89ms | help me with some css styling
[T44] RESPONSE stream=True llm=3028ms total=3117ms chars=117
[T44] COMPLETE 1204ms tags=[css, web-development, html] primary=css
```

#### Live Dashboard

The proxy serves a real-time monitoring dashboard at `http://localhost:5757/dashboard`, a full operational view of what virtual-context is doing to every request.

**Request grid.** Every proxy request displayed with turn number, inbound tags, response tags (updated live when `on_turn_complete` finishes), token counts, latency breakdown (vc overhead vs upstream LLM), tool activity, and turns dropped by filtering. Newest requests appear on top. Each row is clickable for deep inspection.

**Turn inspector.** Click any request row to see the full picture: every message in the request with role labels, content block types (`text`, `tool_use`, `tool_result`, `thinking`), the raw text content, inbound tags vs response tags side by side, and the token budget breakdown showing how context was assembled.

**Ingested history.** When the proxy bootstraps from a client's existing conversation, every ingested turn appears in its own grid with per-turn tags and message previews, so you can verify the tag vocabulary was built correctly from history.

**Session stats.** Uptime, total requests processed, compaction events, total tokens freed, compression ratio, average VC overhead latency, and average upstream LLM latency.

**Request capture.** A ring buffer stores the last 50 raw request bodies (the actual `messages` array sent to the upstream LLM). Inspect any captured request through the dashboard or export as JSON for offline analysis. Essential for diagnosing tagger accuracy: you can see exactly what text the embedding matcher evaluated.

**Live updates.** SSE-powered: new events appear the instant they happen. No polling, no refresh.

**JSON export.** Download the full session state (all events, all stats) as a single JSON file for offline analysis or bug reporting.

### MCP Server (Model Context Protocol)

```bash
pip install virtual-context[mcp]
```

Exposes virtual-context as an MCP server for integration with Claude Desktop, Cursor, or any MCP-compatible client:

| Type | Name | Description |
|------|------|-------------|
| Tool | `recall_context` | Tag + retrieve + assemble context for a message |
| Tool | `recall_all` | Load summaries for all topics (broad overview path) |
| Tool | `remember_when` | Time-scoped recall with relative presets or explicit date bounds |
| Tool | `compact_context` | Trigger compaction on a message history |
| Tool | `domain_status` | All tags with stats |
| Tool | `expand_topic` | Expand a topic to segment or full detail depth |
| Tool | `collapse_topic` | Collapse a topic back to summary or none |
| Tool | `find_quote` | Full-text search across all stored conversation text (fixed top 20 results per call) |
| Tool | `query_facts` | Structured fact lookup with subject/verb/object/status filters and semantic expansion |
| Resource | `virtualcontext://domains` | List all tags |
| Resource | `virtualcontext://domains/{tag}` | Summaries for a specific tag |
| Prompt | `recall` | Suggest context retrieval for a topic |
| Prompt | `summarize_session` | Suggest compaction |

### OpenClaw Plugin

Plugin for OpenClaw agents using lifecycle hooks for sync retrieval (`message.pre`) and fire-and-forget compaction (`agent.post`) via CLI calls. No bridge server needed. Depends on the [plugin lifecycle hook architecture](https://github.com/openclaw/openclaw/pull/12082) currently in progress.

## Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Engine** | `engine.py` | Main orchestrator: `on_message_inbound()`, `on_turn_complete()`, `ingest_history()` |
| **TurnTagIndex** | `core/turn_tag_index.py` | Live per-turn tag index, velocity tracking, greedy set cover |
| **TagGenerator** | `core/tag_generator.py` | LLM and keyword semantic tagging with vocabulary feedback + per-turn fact signal extraction |
| **EmbeddingTagGenerator** | `core/embedding_tag_generator.py` | Sentence-transformers cosine similarity against tag vocabulary |
| **TagCanonicalizer** | `core/tag_canonicalizer.py` | Alias detection, plural folding, normalization |
| **Retriever** | `core/retriever.py` | IDF-weighted tag retrieval, related tag expansion, FTS fallback |
| **Assembler** | `core/assembler.py` | Budget-aware context assembly with priority ordering |
| **Monitor** | `core/monitor.py` | Two-tier threshold detection (soft/hard) |
| **Segmenter** | `core/segmenter.py` | Turn pairing + contiguous tag grouping via TurnTagIndex |
| **Compactor** | `core/compactor.py` | LLM summarization + fact extraction + tag summary rollup, concurrent via ThreadPoolExecutor |
| **CostTracker** | `core/cost_tracker.py` | Per-session LLM usage and cost tracking |
| **ToolLoop** | `core/tool_loop.py` | Multi-provider multi-round tool execution for reader model (Anthropic/OpenAI/Gemini) |
| **ContextStore** | `core/store.py` | Storage interface (SQLite or filesystem) |
| **PayloadFormat** | `proxy/formats.py` | Strategy pattern for Anthropic/OpenAI/Gemini request/response handling |
| **ProxyServer** | `proxy/server.py` | HTTP proxy: enrichment, filtering, history ingestion, session continuity |
| **MultiInstance** | `proxy/multi.py` | Multi-instance launcher: N uvicorn listeners, shared or per-port engine/store |
| **ProxyDashboard** | `proxy/dashboard.py` | Live SSE dashboard with request grid, turn inspector, session stats |
| **ProxyMetrics** | `proxy/metrics.py` | Thread-safe event collector + request capture ring buffer |

### Storage Backends

**SQLiteStore**: Primary backend. Two FTS5 indexes (summary search for retrieval, full-text search across raw stored conversation text for `find_quote`), tag-overlap queries via junction table, tag aliases, tag summaries, chunk embeddings for semantic search, structured fact tables with provenance tracking. Single file, no external dependencies.

**FilesystemStore**: Debug/inspection backend. Markdown files with YAML frontmatter, organized by tag directory. Human-readable, git-friendly.

Both implement the same abstract interface; swap backends without changing application code.

### LLM Providers

**GenericOpenAIProvider**: Works with Ollama, LM Studio, vLLM, or any OpenAI-compatible endpoint. Pure httpx, no SDK dependency.

**AnthropicProvider**: Direct Anthropic API via httpx. No SDK dependency.

Retry logic with exponential backoff on both.

## Design Decisions

**Sync-first.** Zero async/await in the engine. All I/O is synchronous httpx. Concurrent compaction uses `ThreadPoolExecutor`, not asyncio. Both engine entry points complete in under a second with a local Ollama model. The proxy uses FastAPI async for HTTP handling but calls the sync engine via `asyncio.to_thread`.

**Tagging and fact extraction are separate concerns.** Tagging drives retrieval (which stored context to inject). Fact extraction captures structured knowledge (what the user said, decided, or asked for). Both happen during response processing, but they serve different purposes and are optimized independently. Most memory systems conflate retrieval indexing with knowledge extraction into a single LLM call.

**Two-tagger architecture.** Inbound tagging (before the LLM responds) and response tagging (after) use different models optimized for different tasks. The recommended configuration uses embedding cosine similarity for inbound (closed-set, deterministic, can't hallucinate novel tags) and an LLM for response (creative, vocabulary-building, generates related tags). The response tagger sees surrounding conversation turns, not just the current message, so it can correctly handle ambiguous or context-dependent replies.

**Two-phase fact verification.** Per-turn fact signals are treated as hints, not ground truth. They are verified and consolidated at compaction time with the full multi-turn segment as context. This catches extraction errors that single-pass systems commit permanently.

**Compression improves reasoning, not just cost.** A 200K model running at a 30K ceiling doesn't just save tokens — it concentrates the model's attention on curated, high-signal context. Research on long-context attention degradation ("lost in the middle") shows that facts buried deep in long sequences are missed more often than the same facts presented in a shorter, structured window. The configurable ceiling turns context compression from a cost optimization into a quality improvement.

**Tag overlap with IDF scoring, not vector similarity.** Retrieval matches by IDF-weighted tag overlap, not cosine similarity. Related tag expansion handles vocabulary mismatch. Faster (no embedding computation at query time), fully interpretable, and composable with the tag hierarchy.

**Vocabulary feedback, not few-shot.** The LLM tagger gets a live vocabulary of tags already used in the session and store, and is instructed to reuse them when the topic matches. Convergence without manual curation.

**No SDK dependencies.** Both LLM providers use raw httpx. The only required dependencies are `pyyaml` and `httpx`.

**Tag preservation.** During compaction, the LLM can add refined tags but never remove original ones. A segment tagged `[ux, recipes, frontend]` stays tagged with all three even after summarization, ensuring cross-topic retrieval always works.

**Tool chain integrity.** The history filter preserves API-required message dependencies atomically. Every `tool_use` block in an assistant message is kept with its corresponding `tool_result`, and vice versa. Forward and backward scanning ensures multi-step tool chains are never broken, even when surrounding turns are filtered out.

## Stress-Tested

virtual-context has been validated across multiple dimensions: adversarial prompt suites, production traffic, and deliberate edge cases.

### Adversarial Prompt Suite

100-turn conversations with deliberately overlapping domains (Flask IoT API, music studio, ML pipeline, cross-domain integration), vocabulary mismatches, ambiguous callbacks, and cross-domain synthesis queries, using a 3,000-token context window with Claude Haiku:

- **Cross-vocabulary recall**: "caching trick for the feed" correctly retrieves "materialized view" despite zero primary tag overlap. Related tag expansion bridges the vocabulary gap
- **IDF-weighted precision**: "precomputed summary table" retrieves the correct segment over 20+ competing segments sharing common tags like `database` and `performance`
- **Ambiguous multi-match**: "what middleware pattern?" correctly identifies both auth and logging middleware across 4 overlapping domains; "plugins - Flask, audio, or ML?" correctly disambiguates
- **Temporal recall**: "going back to the very beginning, what were the key decisions?" retrieves original Flask blueprint architecture from turn 1 via segment-level retrieval, even after 4 compaction events
- **Overview query bounding**: `vc_recall_all` can load 22 bundled tag summaries while staying bounded at ~2,900 tokens post-compaction
- **Adversarial pass rate**: 89% on 28 deliberately adversarial prompts (vocabulary mismatches, ambiguous references, cross-domain synthesis, late vague recalls)
- **Compaction**: 4 events across 100 turns, average 1,147 tokens per turn, peak 3,018 tokens
- **Tag convergence**: vocabulary stabilizes within 10-15 turns via feedback loop

### Production Validation

The proxy has been validated in production with OpenClaw (Telegram bot) handling real multi-topic conversations:

- **Consecutive user message batching**: Telegram sends multiple user messages in rapid succession. The proxy handles misaligned message sequences without losing history pairs
- **Tool chain preservation**: 90-message conversations with interleaved `tool_use`/`tool_result` chains filtered from 52 messages down to 27 without breaking a single tool dependency
- **Embedding inbound matching**: Live tag vocabularies of 40+ tags correctly matched ("help me with css styling" → `[css, design]`, "what about font-weight" → `[css]` via semantic similarity)
- **History ingestion**: 43 pre-existing conversation turns tagged and indexed in a single pass, vocabulary immediately available for subsequent requests

## Development

```bash
git clone https://github.com/virtual-context/virtual-context.git
cd virtual-context
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v --ignore=tests/ollama    # ~1170 unit tests
python -m pytest tests/ollama/ -v -m ollama          # integration (requires Ollama)
```

## License

AGPL-3.0, Copyright Y. Ahmed Kidwai

For commercial licensing inquiries, contact: ahmed@kidw.ai
