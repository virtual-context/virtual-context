# Design Decisions

This document explains the rationale behind key architectural choices in virtual-context.

## Compression Improves Reasoning

The central thesis: compressed, structured context produces better model reasoning than raw conversation dumps. When a model receives 60K tokens of curated summaries organized by topic, it performs better than when it receives 60K tokens of raw chat history that includes noise, repetition, and irrelevant tangents.

This is counterintuitive. Compression is lossy. But conversation text has extremely low information density. Most turns contain phatic exchanges, restated context, debugging dead ends, and scaffolding that served a purpose in the moment but adds noise later. Compaction strips this while preserving the semantic core.

The benchmarks confirm it: on LongMemEval (100 questions), virtual-context answers 95/100 correctly vs. 33/100 for the same reader model given the full raw history, while sending 55% fewer tokens per question. Retrieval plus compression surfaces the right information; full history buries it. See [benchmarks](benchmarks.md) for details.

## Two-Tagger Architecture

Why two taggers instead of one?

**The embedding tagger is safe.** It runs locally, costs nothing per call, executes in milliseconds, and produces deterministic results. If the LLM tagger fails, times out, or hallucinates tags, the embedding tagger's results still anchor retrieval to the right topics.

**The LLM tagger is rich.** It understands context, catches implicit topics ("I'm worried about the deadline" -> `project-timeline`), and generates natural vocabulary that matches how users think about topics.

Running both costs one small-model LLM call per turn but provides the benefits of both approaches. The embedding tagger handles the inbound path (user message arrives, need tags now for retrieval), while the LLM tagger runs after the response (full turn context, no latency pressure).

This split is the default: `retrieval.inbound_tagger_type` is `"embedding"` out of the box, and can be set to `"llm"` to route inbound tagging through the LLM tagger instead. If you're optimizing for cost, set `tag_generator.type: "keyword"` to replace LLM tagging of completed turns with keyword extraction. The system degrades gracefully: retrieval still works, just with less vocabulary richness.

## Sync-First Processing

The engine processes requests synchronously on the inbound path and asynchronously on the completion path. This is deliberate:

**Inbound must be synchronous.** The model needs context before it can respond. Tagging, retrieval, and assembly must complete before the request is forwarded to the upstream. This adds latency to the request path, but the alternative (sending an un-enriched request) defeats the purpose. The inbound tagger is an embedding model by default precisely to keep this path off the LLM.

**Completion can be asynchronous.** After the response is streamed back to the client, the background path handles turn persistence, tagging, index updates, compaction checks, and fact extraction. The user is already reading the response; this work doesn't block them.

The tradeoff: each new request must wait for the previous turn's completion work on the same conversation. Users take seconds between turns, so the wait is rarely noticeable.

**The REST path has no completion signal of its own.** A hosting service calls *prepare* before its LLM call and *ingest* after it, as two separate requests; the ingest may arrive late or never. That asymmetry is why prepare persists the user half of the turn immediately, why ingest reconciles rather than appends, and why every write is idempotent against redelivery. The proxy gets completion for free from sitting on the response stream; the REST surface has to earn the same guarantees with reconciliation.

## Tag Preservation Through Compaction

When segments are compacted (summarized), their tag assignments are preserved. The summary inherits the tags of the original turns. This ensures that retrieval by tag still works after compaction; the tag space is stable even as the underlying text is compressed.

The tag *vocabulary* is not frozen, though. Overloaded tags are split into narrower subtags, near-duplicate tags are canonicalized through the alias table, and queries against an old name still resolve to its successors. Stability comes from the alias chain, not from immutability.

This is why tag quality matters so much at assignment time: tags are the primary index. A bad tag persists until the vocabulary lifecycle corrects it, and everything stored under it in the meantime inherits the damage.

## Chain Collapse Over Truncation

Many systems handle tool-heavy conversations by truncating old tool results. This is lossy and unpredictable: the model doesn't know what was lost, and truncation boundaries are arbitrary.

Chain collapse is different: it replaces tool exchanges with compact stubs that include a restore reference. The model can see that information exists (the stub is visible) and recover it on demand (via `vc_restore_tool`). Nothing is lost; it's just paged out.

This mirrors virtual memory: pages are swapped to disk and faulted back in on access. The model operates on the working set (recent turns + retrieved summaries) while the full history remains recoverable.

## No SDK Dependencies

Virtual-context operates as a proxy, not a library. It doesn't require changes to the LLM client, the model, or the application code. You point your API calls at `localhost:5757` instead of `api.anthropic.com`, and everything works.

This is a deliberate constraint. SDK integrations are tighter and can do more (e.g., client-side token counting, structured prompting), but they require adoption, maintenance per framework, and lock-in. A proxy is invisible and universal.

The SDK path exists (import `virtual_context` and drive `VirtualContextEngine` directly) for users who want direct engine access, but the primary distribution mechanism is the proxy.

## Format Detection Over Configuration

The proxy auto-detects whether a request uses the Anthropic, OpenAI Chat, OpenAI Responses, or Gemini API format. No configuration needed.

This eliminates a class of misconfiguration errors and makes the proxy genuinely transparent: swap the upstream URL and it works, regardless of which API format the client speaks.

Detection uses structural signals (field names, URL paths, model name prefixes) rather than content heuristics, so it's reliable.

## Greedy Set Cover for Assembly

The assembly pass uses greedy set cover to fill the context budget. Segments are sorted by retrieval score, then added in order until the budget is full. If a segment doesn't fit, smaller segments are tried.

This is within a constant factor of the theoretical best and fast (single pass, O(n) in the number of candidate segments). More sophisticated packing approaches (dynamic programming, ILP) would add complexity without changing what the model sees in practice: the budget boundary falls in the tail of the score distribution, where candidates are near-interchangeable.

## Background Compaction, Never Blocking

Compaction never blocks the request path. It runs in the background thread after `on_turn_complete`. If the user sends another message before compaction finishes, the new request waits for completion (which includes compaction) before proceeding.

This means compaction latency is hidden behind user think time. A compaction that takes 2 seconds is invisible if the user takes 5 seconds to type their next message. Only when the user types faster than compaction can complete does it add perceptible latency.

## Demand-Paged Context

The virtual-context block injected into the system prompt is a working set, not the full history. The model sees:

1. **Recent turns** at full fidelity (protected window)
2. **Retrieved summaries** for relevant topics (demand-paged based on the current query)
3. **Extracted facts** and, for the requester, a person card (in group conversations, a speaker roster)
4. **Topic hints** listing what else is available (table of contents, not full text)
5. **Tool definitions** for `vc_expand_topic`, `vc_find_quote`, etc. (the model can page in more)

This mirrors a demand-paged virtual memory system. The model operates on the working set. If it needs more, it calls a tool to fault the page in. The engine manages the page table (tag index), the page frames (segment budget), and eviction (compaction).

## Fact Supersession Over Versioning

Facts use supersession (new fact invalidates old) rather than versioning (keep all versions). This keeps the fact store clean: "User lives in LA" replaces "User lives in NYC" rather than accumulating a history.

The tradeoff is that supersession detection requires the compaction LLM to understand when two facts contradict. This is imperfect, but in practice the LLM is good at identifying direct contradictions ("moved from X to Y" supersedes "lives in X").

For cases where history matters ("User lived in NYC from 2020-2023, then moved to LA"), the fact's `when` field and the underlying conversation segments preserve the timeline. Supersession cleans up the active fact set, not the historical record.
