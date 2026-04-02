# Design Decisions

This document explains the rationale behind key architectural choices in virtual-context.

## Compression Improves Reasoning

The central thesis: compressed, structured context produces better model reasoning than raw conversation dumps. When a model receives 60K tokens of curated summaries organized by topic, it performs better than when it receives 60K tokens of raw chat history that includes noise, repetition, and irrelevant tangents.

This is counterintuitive. Compression is lossy. But conversation text has extremely low information density. Most turns contain phatic exchanges, restated context, debugging dead ends, and scaffolding that served a purpose in the moment but adds noise later. Compaction strips this while preserving the semantic core.

The benchmarks confirm it: virtual-context achieves 95% accuracy on LocOMo memory questions vs. 33% for full-history baselines, because retrieval + compression surfaces the right information while full history buries it.

## Two-Tagger Architecture

Why two taggers instead of one?

**The embedding tagger is safe.** It runs locally, costs nothing per call, executes in milliseconds, and produces deterministic results. If the LLM tagger fails, times out, or hallucinates tags, the embedding tagger's results still anchor retrieval to the right topics.

**The LLM tagger is rich.** It understands context, catches implicit topics ("I'm worried about the deadline" -> `project-timeline`), and generates natural vocabulary that matches how users think about topics.

Running both in parallel costs one Haiku call per turn (~0.01 cents) but provides the benefits of both approaches. The embedding tagger handles the inbound path (user message arrives, need tags now for retrieval), while the LLM tagger runs after the response (full turn context, no latency pressure).

If you're optimizing for cost, set `tag_generator.type: "embedding"` to disable the LLM tagger entirely. The system degrades gracefully: retrieval still works, just with less vocabulary richness.

## Sync-First Processing

The engine processes requests synchronously on the inbound path and asynchronously on the completion path. This is deliberate:

**Inbound must be synchronous.** The model needs context before it can respond. Tagging, retrieval, and assembly must complete before the request is forwarded to the upstream. This adds ~50-200ms to request latency, but the alternative (sending an un-enriched request) defeats the purpose.

**Completion can be asynchronous.** After the response is streamed back to the client, the background thread handles response tagging, index updates, compaction checks, and fact extraction. The user is already reading the response; this work doesn't block them.

The tradeoff: each new request must wait for the previous turn's completion to finish. The `wait_for_complete()` call at the start of each request ensures consistency. In practice, completion takes 200-500ms, and users take seconds between turns, so the wait is rarely noticeable.

## Tag Preservation Through Compaction

When segments are compacted (summarized), their tag assignments are preserved. The summary inherits the tags of the original turns. This ensures that retrieval by tag still works after compaction; the tag space is stable even as the underlying text is compressed.

When segments are deep-compacted (re-summarized), tags are preserved again. The tag set is monotonically stable across compaction tiers.

This is why tag quality matters so much at assignment time: tags are the permanent index. A bad tag persists through all compaction levels.

## Chain Collapse Over Truncation

Many systems handle tool-heavy conversations by truncating old tool results. This is lossy and unpredictable: the model doesn't know what was lost, and truncation boundaries are arbitrary.

Chain collapse is different: it replaces tool exchanges with compact stubs that include a restore reference. The model can see that information exists (the stub is visible) and recover it on demand (via `vc_restore_tool`). Nothing is lost; it's just paged out.

This mirrors virtual memory: pages are swapped to disk and faulted back in on access. The model operates on the working set (recent turns + retrieved summaries) while the full history remains recoverable.

## No SDK Dependencies

Virtual-context operates as a proxy, not a library. It doesn't require changes to the LLM client, the model, or the application code. You point your API calls at `localhost:8100` instead of `api.anthropic.com`, and everything works.

This is a deliberate constraint. SDK integrations are tighter and can do more (e.g., client-side token counting, structured prompting), but they require adoption, maintenance per framework, and lock-in. A proxy is invisible and universal.

The SDK path exists (`virtual-context[sdk]`) for users who want direct engine access, but the primary distribution mechanism is the proxy.

## Format Detection Over Configuration

The proxy auto-detects whether a request uses the Anthropic, OpenAI Chat, OpenAI Responses, or Gemini API format. No configuration needed.

This eliminates a class of misconfiguration errors and makes the proxy genuinely transparent: swap the upstream URL and it works, regardless of which API format the client speaks.

Detection uses structural signals (field names, URL paths, model name prefixes) rather than content heuristics, so it's reliable.

## Greedy Set Cover for Assembly

The assembly pass uses greedy set cover to fill the context budget. Segments are sorted by retrieval score, then added in order until the budget is full. If a segment doesn't fit, smaller segments are tried.

This is optimal in practice (within a constant factor of the theoretical best) and fast (single pass, O(n) in the number of candidate segments). More sophisticated approaches (dynamic programming, ILP) were tested but didn't improve results enough to justify the complexity.

## Background Compaction, Never Blocking

Compaction never blocks the request path. It runs in the background thread after `on_turn_complete`. If the user sends another message before compaction finishes, the new request waits for completion (which includes compaction) before proceeding.

This means compaction latency is hidden behind user think time. A compaction that takes 2 seconds is invisible if the user takes 5 seconds to type their next message. Only when the user types faster than compaction can complete does it add perceptible latency.

## Demand-Paged Context

The virtual-context block injected into the system prompt is a working set, not the full history. The model sees:

1. **Recent turns** at full fidelity (protected window)
2. **Retrieved summaries** for relevant topics (demand-paged based on the current query)
3. **Topic hints** listing what else is available (table of contents, not full text)
4. **Tool definitions** for `vc_expand_topic`, `vc_find_quote`, etc. (the model can page in more)

This mirrors a demand-paged virtual memory system. The model operates on the working set. If it needs more, it calls a tool to fault the page in. The engine manages the page table (tag index), the page frames (segment budget), and eviction (compaction).

## Fact Supersession Over Versioning

Facts use supersession (new fact invalidates old) rather than versioning (keep all versions). This keeps the fact store clean: "User lives in LA" replaces "User lives in NYC" rather than accumulating a history.

The tradeoff is that supersession detection requires the compaction LLM to understand when two facts contradict. This is imperfect, but in practice the LLM is good at identifying direct contradictions ("moved from X to Y" supersedes "lives in X").

For cases where history matters ("User lived in NYC from 2020-2023, then moved to LA"), the fact's `when` field and the underlying conversation segments preserve the timeline. Supersession cleans up the active fact set, not the historical record.
