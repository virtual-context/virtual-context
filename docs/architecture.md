# Architecture

Virtual-context is a transparent context virtualization layer that sits between LLM clients and upstream providers. It compresses, indexes, and pages conversation history so that models operate on relevant summaries rather than raw token streams.

## Memory Model

Storage is organized as three layers of decreasing resolution over one durable ground truth:

```
Canonical turns    (ground truth: every message, stored verbatim, per-message rows)
    |
Layer 0: Raw recent turns          (active memory, in the context window)
Layer 1: Segment summaries + facts (compressed pages, per-topic summaries)
Layer 2: Tag summaries             (working set descriptors, bird's-eye view)
```

**Canonical turns** are the durable record. Every user and assistant message is persisted as its own row with a content hash, a sort key that defines conversation order, sender and channel provenance, and the compaction lifecycle state. Everything else (segments, facts, tag summaries, embeddings) is derived from canonical turns and can be rebuilt from them.

**Segments** are per-topic layer-one summaries produced by compaction. Each stores a free-form retrieval synopsis plus structured claims bound to complete requester-authored canonical lanes. The synopsis helps rank memory but is never shown as factual evidence. **Tag summaries** are one-per-tag layer-two digests that copy and deduplicate validated segment claims without re-authoring them. A primary `_general` tag is materialized too, so unclassified content remains recallable.

Paging preserves the layers instead of flattening them: `SUMMARY` presents compact tag claims, `SEGMENTS` presents the richer individual segment claims and source evidence, and `FULL` reconstructs the exact canonical human and historical-assistant lanes. Legacy rows without structured claims fall back to admitted canonical requester text until they are regenerated.

## Request Pipeline (proxy path)

```
Client request
    |
    v
Format detection (Anthropic / OpenAI Chat / OpenAI Responses / Gemini)
    |
    v
Envelope extraction (sender, channel, reply metadata parsed and preserved;
                     transport wrappers removed from the model-visible text)
    |
    v
Conversation identity resolution (explicit id -> label -> chat id -> system-prompt hash,
                                  through the alias table)
    |
    v
Routing gate: does this conversation have retrievable content?
    |                       |
    | no (cold start)       | yes
    v                       v
Passthrough             Active path
(forward mostly           |-- Wait for previous turn's background work
 untouched)               |-- Canonical ingestion (reconcile payload against stored rows)
                          |-- Inbound tagging (embedding tagger by default)
                          |-- Retrieval (3-signal RRF: IDF tag overlap + BM25 + embedding cosine)
                          |-- Assembly (recent turns + summaries + facts + context hint, within budget)
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
                     Background completion path
                          |-- Persist the completed turn to canonical storage
                          |-- LLM tagging of the completed turn, TurnTagIndex update
                          |-- Compaction check (soft/hard thresholds)
                          |-- Tag summary materialization + context hint pre-warm
                          |-- Fact extraction and supersession
```

The routing gate is content-based: a conversation with no compacted content and no indexed turns is passed through with minimal modification, so brand-new conversations pay almost nothing. Once retrievable content exists, requests take the active path.

## The REST Path

Besides the proxy, the engine serves a prepare/ingest REST surface used by hosting services. Its shape differs from the proxy path in one critical way: **there is no response hook**. The proxy sees the model's response on the same connection and can run its completion work immediately. A REST client calls *prepare* (enrich this request) and later, separately, *ingest* (here is the assistant's reply); the ingest call may arrive late or never.

The engine is built around that asymmetry: prepare persists the user half of the turn, ingest reconciles the assistant half against the stored tail, and finalization logic tolerates ingests that never arrive. Every write is idempotent against redelivery.

## Canonical Turn Ingestion

Ingestion reconciles each incoming payload against the rows already stored, on every request, not just the first:

- **Alignment**: incoming messages are matched to stored rows by content hash. A tail-hash fast path recognizes the common case (payload extends the stored conversation by one turn) without rewriting anything.
- **Sort keys**: rows carry spaced numeric sort keys so mid-history inserts do not require renumbering. When repeated inserts exhaust a gap, the reconciler shifts subsequent keys to restore spacing.
- **Fragment guard**: payloads that look like a fragment of a different conversation (no overlap with the stored tail) are rejected rather than appended, which prevents cross-conversation contamination.
- **Turn groups**: canonical rows are per-message, so one logical turn (a user message and its assistant reply, or several user messages before one reply) spans multiple rows. A derived turn-group number ties those rows together, recomputed after ingestion, so compaction and windowing operate on whole logical turns and never split a reply from its prompt.
- **Provenance**: sender, channel, actor identity, and reply-target metadata extracted from transport envelopes are stored on each row.

## Multi-Worker Coordination

Multiple proxy or REST workers can serve the same conversation against a shared PostgreSQL store. Coordination is explicit:

- **Compaction operations are fenced.** A compaction runs under a leased operation row; a worker that loses its lease has every subsequent write rejected, so a stalled worker cannot clobber a takeover.
- **Lifecycle epochs** version a conversation's identity lifecycle. Writes carry the epoch they were started under and are rejected if the conversation was reset or merged in the meantime.
- **Schema bootstrap** is serialized under an advisory lock, so N workers starting simultaneously against a fresh database do not race the DDL.
- **The backlog sweeper** finds conversations whose tagged-but-uncompacted backlog has grown past a threshold (for example, because their traffic pattern never triggered an inline compaction) and queues them for compaction.

## Conversation Identity and Aliases

Identity resolution tries, in order: an explicitly supplied conversation ID, a conversation label, a transport chat ID, and finally a hash of the system prompt. IDs in the reserved `sk:` namespace are caller-asserted and passed through verbatim.

Every resolution step reads through the **alias table**. `VCATTACH` writes a durable alias redirecting one conversation identity to another (see [commands](commands.md)); `link_predecessor` records an idempotent predecessor link when a client's session identity rolls over but the conversation logically continues. Stale conversation markers embedded in old assistant responses keep resolving because they follow the alias chain.

## Attribution

Group conversations carry per-message sender identity. The envelope parser claims the sender, channel, and reply-target from transport metadata before it is stripped from the model-visible text; these land as columns on the canonical turn. On top of that sit **actor profiles** and **person cards** (durable, per-actor fact digests injected into assembly for the requester) and **speaker-conditioned retrieval**: search tools accept a speaker selection, so "what has this person said" resolves against rows attributed to that actor rather than the whole conversation. The full subsystem, its gates, and its operator surface are documented in [attribution](attribution.md).

## Component Map

### Package root (`virtual_context/`)

| Module | Responsibility |
|--------|---------------|
| `engine.py` | Top-level orchestrator. Owns `on_message_inbound`, `on_turn_complete`, `ingest_history`, admin operations |
| `conversation_identity.py` | Identity resolution ladder and the `sk:` caller-asserted namespace |
| `token_counter.py` | Token counting modes: `anthropic` (bundled tokenizer, approximate), `tiktoken`, `estimate` (len/4). Image-aware via dimension-based costing |
| `config.py` | YAML config loading with validation, preset system, multi-instance support |
| `types.py` | Dataclasses for the entire system; the dataclass defaults are the source of truth for configuration defaults |

### Core (`virtual_context/core/`)

| Module | Responsibility |
|--------|---------------|
| `compaction_pipeline.py` | The compaction operation: segment selection, summarization, tag summary materialization, watermark advance, context-hint pre-warm |
| `compactor.py` | Summarization LLM calls, tag-rule prompt selection, fact extraction |
| `compaction_fence.py` | Compaction operation leases and fencing modes |
| `sweeper_backlog.py` | Backlog detection queries for the compaction sweeper |
| `lifecycle_epoch.py` | Epoch guards for conversation lifecycle changes |
| `ingest_reconciler.py` | Canonical turn alignment, sort-key allocation and rebalance, fragment rejection |
| `segmenter.py` | Turn pairing and tag grouping for compaction |
| `tagging_pipeline.py` | Completed-turn LLM tagging, strict payload-context tagging, row-based sweep fallback |
| `turn_tag_index.py` | Live per-turn tag metadata and working-set computation |
| `tag_splitter.py` / `tag_canonicalizer.py` | Tag vocabulary lifecycle: splitting overloaded tags, normalizing aliases |
| `retriever.py` / `retrieval_scoring.py` | Candidate generation and 3-signal RRF fusion with dampening and boosts; the ranked unit is the tag |
| `retrieval_assembler.py` | Retrieval-to-assembly bridge, context hint cache |
| `semantic_search.py` / `quote_search.py` | Full-text and embedding search over stored conversation text, speaker-aware search |
| `fact_query.py` | Structured fact lookup with verb expansion and dense re-ranking |
| `assembler.py` | Budget-aware context block construction: recent turns, summaries, facts, person card, speaker roster |
| `hint_builder.py` | The `<context-topics>` hint block: topic list, budgets, paging guidance |
| `protected_window.py` | The recent-turn window exempt from compaction and payload rewriting |
| `alias_resolution.py` | Alias chain reads for identity resolution |
| `monitor.py` | Context window fill tracking against soft/hard thresholds |
| `tool_loop.py` / `tool_query.py` | vc_* tool catalogue, execution dispatch, anti-repetition, continuation rounds |
| `speaker_roster.py` / `speaker_labels.py` | Audience-scoped speaker rosters and labels for attribution |
| `embedding_provider.py` | Embedding model loading and caching |
| `provider_adapters.py` | Adapter layer for Anthropic, OpenAI Chat, OpenAI Responses, and Gemini formats |
| `state_recovery.py` | Store-backed recovery when a client truncates its own history |
| `telemetry.py` | TelemetryEvent, TelemetryRollup, TelemetryLedger |
| `temporal_resolver.py` | Time-bounded recall: relative dates to absolute ranges |

### Proxy (`virtual_context/proxy/`)

| Module | Responsibility |
|--------|---------------|
| `server.py` | ASGI application setup, route registration, VC command regex, multi-instance |
| `handlers.py` | Request handling: enrichment, streaming SSE forwarding, VC command dispatch, paging path |
| `state.py` | `ProxyState`: per-conversation state machine, routing gate, ingestion finalization, background pools |
| `message_filter.py` | Payload rewriting: chain collapse to stubs, deep-drop of old turns, history widening |
| `_envelope.py` | Envelope parsing: sender/actor/reply claims, marker and metadata stripping |
| `vcattach.py` | VCATTACH execution and `link_predecessor` |
| `registry.py` | Conversation registry, alias-aware existence checks |
| `session_state.py` | Redis-backed session cache for lossless restarts |
| `helpers.py` | Format detection, context injection, payload construction |
| `metrics.py` | Thread-safe event collector, snapshot aggregation, SSE streaming |
| `dashboard.py` | Dashboard routes (see [proxy](proxy.md) for endpoints and authentication) |

## Storage Backends

Four engine backends are accepted: `sqlite`, `postgres`, `neo4j`, `falkordb`.

- **SQLite** (default): single-file, zero-config. Suitable for single-user and development.
- **FilesystemStore utility**: segments as Markdown files with YAML frontmatter
  for direct archival/test use. It is not an engine backend because it does not
  host the canonical turns required to prove model-visible layered summaries.
- **PostgreSQL**: the multi-worker backend. Canonical turns, fencing, epochs, and the sweeper queries all run here in production deployments.
- **Neo4j / FalkorDB**: graph-backed fact relationships and traversal queries.

The engine backends share the `Store` protocol. Model-facing context never
degrades to an unproved filesystem projection.

## Provider Adapters

The proxy supports four API formats with auto-detection:

| Format | Detection Signal | Context Injection Point |
|--------|-----------------|------------------------|
| **Anthropic** | `"system"` field or model name starts with `"claude"` | `system` field (string or content blocks) |
| **OpenAI Chat** | `/v1/chat/completions` path | `messages[0]` with `role: "system"` |
| **OpenAI Responses** | `/v1/responses` path | `instructions` field |
| **Gemini** | `/v1beta/models` path pattern | `system_instruction` field |

Detection is automatic. No configuration needed.

## Threading Model

- The main request path is synchronous within the async ASGI handler.
- Each conversation's `ProxyState` owns two background pools: a single-worker pool that serializes tagging and turn persistence, and a compaction pool so a long compaction does not block the next turn's tagging.
- Each new request waits for the previous turn's background work on the same conversation before ingesting, so reconciliation always sees a consistent tail.
- Cross-process coordination (many workers, one conversation) is handled at the storage layer via the fencing, epoch, and advisory-lock mechanisms described above, not by in-process locks.
