# Virtual Context: OS-Style Memory Management for LLM Conversations via Structured Knowledge Extraction and Tool-Augmented Retrieval

**Y. Ahmed Kidwai**
kidw.ai

---

## Abstract

Large language models face a fundamental constraint: fixed context windows that cannot accommodate the accumulated context of long-running interactions — whether multi-session conversations, agentic tool-call chains, or workflows that generate substantial context through continuous operation. Existing approaches either silently drop old messages, retrieve via embedding similarity (RAG), or compress everything into summaries — each failing for distinct reasons. RAG misses vocabulary-mismatched content; compression loses specific details; full context suffers from attention degradation ("lost in the middle").

We present Virtual Context (VC), a system that draws on the operating system virtual memory metaphor to manage LLM context. Interaction turns are tagged by topic, compressed into a three-layer hierarchy (raw turns, segment summaries, tag summaries), and paged in and out of the context window on demand. A two-pass fact extraction pipeline produces structured, queryable facts with temporal status tracking and knowledge-update detection via supersession chains. At query time, a reader LLM receives compressed summaries and a suite of five retrieval tools (`find_quote`, `query_facts`, `expand_topic`, `remember_when`, `recall_all`) to drill into details within a bounded token budget.

On LongMemEval (100 random questions from the 500-question benchmark), VC achieves **95% accuracy** compared to **33%** for a full-context baseline using the same mid-tier reader model (Claude Sonnet 4.5), while consuming **55% fewer tokens** on average (52,347 vs. 117,582) at **55% lower cost** ($0.16 vs. $0.36/question) with only **1.5x latency overhead** (12.7s vs. 8.7s). VC achieves 100% accuracy on knowledge-update questions (vs. 29.4% baseline) where fact supersession chains are critical, and 92.9% on temporal-reasoning questions (vs. 32.1%). Analysis of tool call chains reveals 8 emergent retrieval patterns, with 82% of questions answered in 1--2 tool calls. These results demonstrate that structured context management is a higher-leverage investment than model capability: a mid-tier model with VC delivers accuracy that raw full context cannot match at any model tier, at an estimated 3.7x cost reduction versus flagship full-context deployment at standard scale — growing to 22x at realistic conversation lengths (~926K tokens) where baselines fail entirely.

---

## 1. Introduction

### 1.1 The Context Window Problem

Large language models operate within fixed context windows. While these windows have expanded dramatically — from 4K tokens to over a million and beyond — they remain fundamentally insufficient for long-running interactions that accumulate context over time: multi-session conversations, agentic workflows with extensive tool-call chains, or any LLM application where the history of prior interactions carries information relevant to future ones. The context window problem is how to give an LLM accurate access to the full content of its accumulated history without degrading response quality.

Four common approaches each exhibit characteristic failure modes:


**Silent truncation.** The simplest strategy drops the oldest messages when the context window fills. This guarantees that information mentioned only once, early in the conversation, is permanently lost.

**Generalized summarization.** The most widely deployed approach in production systems: periodically compress older conversation turns into a running summary, keeping recent turns verbatim. This preserves more information than truncation but suffers from *lossy compression without structure*. Each summarization pass discards details deemed unimportant at compression time — but importance is query-dependent and unknowable in advance. A user's coffee-to-water ratio, mentioned once in session 12, may be summarized away as irrelevant until they ask about it 50 sessions later. Critically, flat summarization provides no mechanism for knowledge updates: if a user changes their preference from French press to Chemex, a rolling summary may retain both mentions with no indication of which is current, or may have already discarded the update. The summary also cannot be selectively queried — the reader must scan the entire compressed representation linearly, with no way to drill into specific topics or time periods.

**Retrieval-augmented generation (RAG).** Embedding-based retrieval may surface relevant passages from a preprocessed conversation archive. However, RAG depends on vocabulary overlap between the query and the stored text — a question about "my personal best running time" may not match a passage describing "I finished the 5K in 25:50." RAG also lacks temporal awareness: it cannot answer "what happened last Tuesday?" without explicit date metadata that standard RAG pipelines do not maintain.

**MEMORY.MD** Agentic workflows have supported creating MEMORY.MD and other supplementary MD files with links to other files which allow agentic models to traverse and fill context to establish base behavior patterns into the system prompt.  These work well for static patterns however they do not work well for context windows which are growing in context and knowledge during active use.  They also permanently allocate context storage for the data and require specific prompting in order to begin the traversal process into linked files.

**Full context.** With sufficiently large context windows, one might include the entire conversation history. This approach has perfect recall by construction — every piece of information is present. Yet Liu et al. (2023) demonstrated that LLMs exhibit a U-shaped attention pattern in long contexts, reliably attending to the beginning and end while frequently missing information in the middle. Our baseline experiments confirm this dramatically: a model with the a 80-90% full context-window answers only 33% of recall questions correctly.

The counter-intuitive finding that motivates this work is that *more context hurts performance*. A model seeing 52K tokens of curated, compressed context with retrieval tools answers 95% of questions correctly — nearly three times the accuracy of the same model seeing 118K tokens of raw conversation. VC's compression is fundamentally different from generalized summarization: it is *structured* (organized by topic with queryable facts), *hierarchical* (three layers with bidirectional paging), *updatable* (supersession chains track knowledge changes), and *selectively expandable* (the reader can demand-page any compressed segment back to full detail).

### 1.2 Our Contribution: OS-Style Memory for LLMs

Virtual Context applies the operating system virtual memory paradigm to LLM context management. Just as an OS maintains the illusion of unlimited memory through page tables, demand paging, and working set management, VC maintains the illusion of unlimited memory through topic tagging, hierarchical compression, and tool-augmented retrieval.

The analogy carries a fundamental tension: hardware memory is fixed, addressable data — bytes at a given address are deterministic and require no interpretation. Conversational memory is inherently interpretive: deciding that a segment belongs to "coffee-brewing" rather than "kitchen-appliances," or that a user's statement constitutes a knowledge update rather than a passing remark, involves semantic judgment with no hardware equivalent. VC addresses this not by eliminating interpretation but by making it *consistent across both directions*. During ingestion, conversation is interpreted into a shared abstraction layer: a converged tag vocabulary, structured fact dimensions (subject/verb/object/status), and segment boundaries. During recall, queries pass through the same interpretive layer — the inbound tagger maps questions into the same tag space, `query_facts` navigates the same dimensional structure that extraction produced, and `find_quote` searches the same full-text index. Because both the write path and the read path converge on the same vocabulary, fact schema, and tool call interface, the interpretive layer functions as a broadly addressable space — not deterministic like hardware addresses, but consistent enough to support the OS mechanisms (page tables, working sets, demand paging, LRU eviction) that the analogy requires. Tag canonicalization, alias consolidation, and vocabulary convergence mechanisms reinforce this consistency over time. The original conversation text is preserved as a fallback for cases where the shared abstraction fails to capture a detail — the reader can demand-page raw text via `expand_topic` and `find_quote` — but the primary mechanism is the shared interpretive layer itself.

We make five technical contributions:

1. **A three-layer memory hierarchy with bidirectional paging.** Raw conversation turns (Layer 0) are compressed into segment summaries (Layer 1) and further into tag summaries (Layer 2) using a greedy set cover algorithm. Any layer can be expanded back to the layer below on demand, with LRU eviction managing the token budget.

2. **A two-pass structured fact extraction pipeline with knowledge-update tracking.** Phase 1 extracts lightweight fact signals per turn during tagging. Phase 2 consolidates these into structured facts with 5W dimensions (what, who, when, where, why), temporal status (active, completed, planned, abandoned, recurring), and provenance tracking during compaction. A supersession checker detects knowledge updates and maintains `superseded_by` chains that ensure the reader sees the latest value for any fact.

3. **A kernel/userspace separation for LLM memory management.** Rather than casting the LLM as its own memory manager (as in MemGPT), VC acts as the kernel — automatically managing compaction, tagging, fact extraction, eviction, and working set tracking. The LLM operates in userspace, issuing system calls (tool invocations) against a managed context window without responsibility for the underlying storage or page replacement.

4. **A shared interpretive layer that makes OS mechanisms viable over semantic content.** Both the write path (tagging, fact extraction) and the read path (inbound tagger, query_facts, find_quote) converge on a shared abstraction: a converged tag vocabulary, structured fact dimensions, and a full-text index. This convergence enables OS mechanisms — page tables, working sets, demand paging, LRU eviction — to operate on semantic content that is inherently interpretive rather than deterministic.

5. **A tool-augmented reader architecture with five retrieval tools.** Rather than passively consuming dumped context, the reader LLM actively queries the memory store using `find_quote` (full-text + semantic search), `query_facts` (structured fact lookup with semantic verb expansion), `expand_topic` (demand paging), `remember_when` (time-scoped recall), and `recall_all` (load all summaries). A synchronous tool loop runs up to 10 continuation rounds within a single user-visible request.

MemGPT (Packer et al., 2023) introduced the analogy between OS virtual memory and LLM context management. Their key architectural choice was to cast the LLM as both the operating system kernel and the application — the model itself decides what to save, search, and evict through explicit function calls. VC takes a fundamentally different position: an LLM cannot effectively serve as its own memory manager while simultaneously reasoning about the user's task. These are competing cognitive demands. Instead, VC acts as the kernel — automatically managing compaction, tagging, fact extraction, eviction, and working set tracking — while the LLM operates in userspace, issuing system calls (tool invocations) against a managed context window. This separation enables VC to implement the OS mechanisms that the analogy implies but that MemGPT left unaddressed: page tables (tag index + tag summaries), demand paging (`expand_topic`), LRU eviction (auto-collapse of coldest topics), working set management (depth-level tracking per tag), and soft/hard page replacement thresholds (compaction triggers at 70% and 85% utilization).

### 1.3 Summary of Results

On 100 random questions from LongMemEval (Wu et al., 2024), VC achieves:

- **95% accuracy** vs. 33% for full-context baseline (+62 percentage points)
- **100%** on knowledge-update questions vs. 29.4% baseline
- **92.9%** on temporal-reasoning vs. 32.1% baseline
- **88.5%** on multi-session vs. 15.4% baseline
- **2.2x fewer tokens** on average (52,347 vs. 117,582)
- **2.2x lower cost** ($0.16/question vs. $0.36/question)
- **82% of questions answered in 1--2 tool calls**
- **8 emergent tool chaining patterns** arising from the reader's interaction with the tool suite

---

## 2. Background and Related Work

### 2.1 Long-Context LLMs and Their Limitations

Context window sizes have grown from 4K tokens (GPT-3) to 128K (Claude 3, GPT-4 Turbo) and beyond 1M tokens (Gemini 1.5 Pro). However, larger windows do not solve the context memory problem. Liu et al. (2023) demonstrated the "Lost in the Middle" phenomenon: LLMs perform best when relevant information appears at the beginning or end of the context, with significant degradation for information in the middle. Yen et al. (2024) explored mitigation strategies but found the effect persistent across model families. Our results confirm this at scale: a model with complete 118K-token conversation history — with the answer guaranteed to be present — answers only 33% of questions correctly.

### 2.2 Retrieval-Augmented Generation

RAG (Lewis et al., 2020) augments LLM generation with retrieved passages from an external corpus. While effective for knowledge-base queries, RAG faces limitations for context memory: (1) vocabulary mismatch between natural-language queries and stored personal context, (2) no native temporal awareness for questions like "what did I mention last week?", (3) no mechanism for tracking knowledge updates ("my address changed to..."), and (4) retrieval granularity mismatches — a conversation turn may contain multiple topics, while RAG typically retrieves fixed-size chunks. Comprehensive surveys by Gao et al. (2024) and Fan et al. (2024) catalog these limitations in detail.

### 2.3 Context Memory Systems

**MemGPT** (Packer et al., 2023) proposed viewing LLM context management through the OS virtual memory lens, mapping the context window to main memory and an external database to disk. Their central architectural decision was to cast the LLM as the OS kernel: the model manages its own memory by explicitly calling functions to append to in-context memory blocks, insert into archival storage, and search for past content. Context overflow is handled reactively — when a request exceeds the context window, the system catches the error, summarizes the oldest messages, and retries. VC departs from this design at a foundational level. Rather than burdening the LLM with memory management, VC itself serves as the kernel: compaction triggers automatically at utilization thresholds, topic tagging and fact extraction run in the background during interaction, and working set eviction follows LRU policy — none of which the reader LLM controls or observes. The reader operates in userspace, calling retrieval tools (`find_quote`, `query_facts`, `expand_topic`) the way an application issues system calls — requesting data and receiving results from a managed memory hierarchy, without responsibility for the underlying storage, compression, or page replacement.

**Knowledge-base (KB) retrieval systems** represent a growing class of context memory approaches. These systems process conversations through extraction pipelines — using "reflect" agents, graph-augmented memory stores, or structured memory schemas — to produce queryable knowledge bases of facts, memories, and observations. At query time, relevant facts are retrieved from the KB and provided as context to the reader model. Recent systems in this category have reported strong results on LongMemEval, achieving up to 91% accuracy on the full 500-question benchmark. However, KB retrieval systems share a common limitation: they operate on pre-processed conversation dumps rather than managing a live context window, and their extraction pipelines (typically operating on fixed-size chunks) cannot see cross-chunk context. Once a detail is lost during extraction, there is no fallback to the source material.

VC performs knowledge extraction too — structured facts with dimensional queries, supersession chains, semantic verb expansion — but it differs from KB systems in what extraction *is for*. In a KB system, extracted facts are the memory: the reader sees facts and nothing else. VC's extracted facts serve a different purpose: they provide structured grounding that supplements, rather than replaces, the narrative context available in compressed summaries and preserved original text.

The distinction matters because facts capture *what* was said but not *how* it was said — and the how carries information that no extraction pipeline can reduce to structured fields. A user who describes their job with growing frustration across sessions, a decision reached reluctantly after trying three alternatives, a tentative "I think I might try woodworking" versus a decisive "I've decided to take up woodworking" — these register differences live in the narrative texture of conversation, not in the extracted facts, which would record the same `user | works_at | company`, `user | uses | tool`, or `user | interested_in | activity` regardless of tone, conviction, or trajectory. Yet these signals are precisely what an agent needs to calibrate its responses: whether to encourage or probe, whether to be enthusiastic or measured, whether a topic is a source of energy or stress. An agent operating on facts alone produces responses that are informationally correct but conversationally flat — it knows *what* the user said without understanding *how they meant it*.

VC's compressed summaries preserve enough of this narrative register to keep the agent calibrated, while facts provide the structured anchors for precise retrieval. The reader sees both simultaneously, each serving a distinct cognitive role: facts orient on what is known, summaries provide the interpretive context for how to engage with that knowledge. When deeper nuance is needed, the reader can demand-page the original text via `expand_topic` and `find_quote`. KB systems that surface only extracted facts strip away exactly the context that makes long-running interactions feel continuous and rational.

### 2.4 Evaluation Benchmarks

**LongMemEval** (Wu et al., 2024) evaluates long-term conversational memory through 500 questions across five categories: single-session (user, assistant, preference), multi-session, temporal-reasoning, and knowledge-update. Each question is paired with a synthetic multi-session conversation "haystack" of 50--300 sessions (~100K tokens). The benchmark uses LLM-as-judge evaluation with category-specific templates.

**LoCoMo** (Maharana et al., 2024) provides 300-turn conversational benchmarks with multi-hop question answering, testing memory systems on very long-term dialogues.

**MRCR** (OpenAI, 2025) is a multi-round coreference resolution benchmark designed to test long-context retrieval. Questions require identifying and verbatim reproducing specific instances of repeated content ("needles") from conversations of up to 1M+ tokens. Scoring is deterministic string matching, eliminating LLM judge variance.

### 2.5 Tool-Augmented LLM Reasoning

The ReAct paradigm (Yao et al., 2022) demonstrated that interleaving reasoning and acting — where an LLM reasons about what action to take, executes it, and observes the result — significantly outperforms pure reasoning or pure acting approaches. VC's reader architecture is a ReAct-style system operating over a structured memory store: the reader reasons about which retrieval tool to use, executes it, observes the results, and decides whether to drill deeper or answer.

---

## 3. System Architecture

### 3.1 Overview: The Three-Layer Memory Hierarchy

VC maintains conversation content at three levels of compression:

**Layer 0: Raw conversation turns.** The active conversation as literal user/assistant message pairs. These occupy the context window directly and are the highest-fidelity representation. A configurable number of recent turns (default: 6 pairs) are always kept raw, protected from compaction.

**Layer 1: Segment summaries.** When context utilization exceeds the soft threshold (70%), contiguous same-topic turns are grouped into segments and compressed by an LLM to ~15% of their original token count (configurable, clamped between 200--2000 tokens). Each segment retains its original tags, full text (for later expansion), and extracted structured facts. Segments are the unit of paging — they can be individually expanded back to full fidelity.

**Layer 2: Tag summaries.** After compaction, a greedy set cover algorithm selects the minimum set of tags that covers all compacted turns. For each tag in this cover set, all its segment summaries are rolled up into a single tag summary (~200 tokens) with a concise description (~80 words). Tag summaries are what the reader sees in the context hint — the "table of contents" for the conversation's memory.

This hierarchy enables bidirectional movement: content flows downward through compaction (Layer 0 → 1 → 2) and upward through expansion (Layer 2 → 1 → 0 via `expand_topic`). The reader sees Layer 2 summaries as default context and can demand-page any topic to Layer 1 or Layer 0 fidelity within its token budget.

### 3.2 Topic Tagging

VC uses a two-tagger architecture that separates the write path (high quality, slow) from the read path (fast, approximate):

**Write-path tagger (LLM-based).** During `on_turn_complete`, each user/assistant pair is tagged by an LLM. The tagger receives the current text, up to 5 pairs of recent conversation context (subject to bleed gating), and a curated list of existing tags (filtered by embedding similarity to top 30 + high-frequency fill). The tagger produces 5--10 tags per turn along with lightweight `FactSignal` objects.

**Read-path tagger (embedding-based).** During `on_message_inbound`, the user's question is matched against cached tag embeddings using cosine similarity (threshold 0.3) with a sentence-transformer model (all-MiniLM-L6-v2). This produces sub-millisecond tag matching without an LLM call, enabling real-time retrieval.

**Context bleed gate.** To prevent stale context from causing mistagging on topic shifts, a cosine similarity gate (threshold 0.1) between the current text and the most recent context suppresses context injection entirely when the user switches topics abruptly. This prevents the tagger from being biased by irrelevant prior discussion.

**Tag canonicalization.** All tags pass through a canonicalizer that resolves aliases (e.g., "5k-run" and "5k-running" map to the same canonical tag) using edit distance and plural folding.

**Automatic tag splitting.** Tags that become overly broad (frequency >= 15 and >= 15% of total turns) are automatically analyzed and split into more specific subtags, preventing the "kitchen sink" tag problem.

### 3.3 Segmentation and Compression

**Segmentation.** The `TopicSegmenter` groups contiguous same-tag turns into segments, with forced splits on session date changes (preventing cross-session segments). Segments are the atomic unit of storage and retrieval.

**Two-tier compaction thresholds.** Modeled on OS page replacement:
- **Soft threshold (70%):** Emits a compaction signal that triggers background compaction of turns older than the protected window.
- **Hard threshold (85%):** Triggers immediate compaction with higher priority.

Protected recent turns (default: 6 pairs) are never compacted, ensuring the most recent conversation context remains at full fidelity.

**Compression algorithm.** Each segment is independently compressed by an LLM to a target of 15% of original token count (clamped to 200--2000 tokens). The compression prompt enforces critical constraints: exact numbers must be preserved (no rounding), present-tense declarations maintained, planning language preserved as planning (not assertions), and user role phrases kept intact.

**Greedy set cover for tag summaries.** After compaction, the system selects the minimum set of tags that covers all compacted turns using a standard greedy set cover algorithm (O(n log n) approximation). The algorithm iterates: at each step, select the tag covering the most uncovered turns, mark those turns as covered, repeat. A primary tag guarantee ensures each segment's most specific tag is always included in the cover, even if the greedy algorithm drops it.

### 3.4 Structured Fact Extraction

VC implements a two-pass fact extraction pipeline:

**Phase 1: Per-turn FactSignal extraction.** During tagging (write path), the LLM extracts lightweight fact signals alongside tags. Each `FactSignal` contains subject, verb, object, temporal status, fact type, and a full-sentence description. These signals are cheap to produce but may be noisy — they see limited context (the current turn plus up to 5 lookback pairs with bleed gating).

**Phase 2: Consolidation at compaction.** When segments are compressed, per-turn fact signals are injected into the compaction prompt as verification hints. The compactor formats each signal as a structured line (e.g., `[experience] user got back from solo camping trip to Yosemite (completed)`) and appends them to the summarization prompt under the heading "Per-turn fact signals (verify and consolidate with full context)." The summarizer sees both the full multi-turn conversation text and these signal hints, producing final facts that have been validated against the complete segment context. This means the summarizer can correct noisy Phase 1 signals — merging duplicates, resolving ambiguous temporal references, and dropping signals that don't hold up when the full conversation is visible. The output is structured `Fact` records with:

- **5W dimensions:** what (full sentence), who, when_date (ISO date), where, why
- **Temporal status:** active, completed, planned, abandoned, recurring
- **Provenance:** segment reference, conversation ID, turn numbers, session date
- **Fact type:** personal (user identity/life), experience (assistant-provided info), world (external facts)

This two-pass approach gives each fact two chances to be correctly extracted — once per turn with narrow context, once per segment with full multi-turn context. The second pass can verify, correct, or consolidate the first pass's signals.

**Knowledge-update detection (supersession).** After new facts are stored, a `FactSupersessionChecker` compares them against existing facts with the same subject. Candidates for supersession are identified through three channels: tag overlap (facts sharing subject and tags), object-keyword similarity (cross-tag duplicates), and embedding similarity on the `what` field (semantically equivalent facts regardless of tag or ingestion order). An LLM verifies whether each candidate is truly contradicted, superseded, or duplicated by the new fact. A programmatic date guard rejects any LLM supersession decision that would mark a newer fact as superseded by an older one — enforcing temporal monotonicity that the LLM prompt requests but does not always respect. Superseded facts have their `superseded_by` field set to the new fact's ID; at query time, they are deprioritized or excluded, ensuring the reader sees the latest value.

**Date resolution.** Relative time references in facts ("yesterday," "last week") are resolved to ISO dates using the session date as anchor. The compactor prompt includes explicit date resolution rules, and a deterministic fallback `resolve_relative_date()` handles common patterns.

### 3.5 Tool-Augmented Reader

Rather than passively consuming context, the VC reader actively queries the memory store through five tools:

**`find_quote(query)`:** Full-text search (FTS5) combined with semantic search across all stored segment text. Returns up to 20 results with deduplication against previously shown segments (`presented_refs` tracking). Also attaches related facts to each result, providing structured data alongside raw text. For multi-session queries, older session excerpts are marked with "[Older session — superseded...]" to prevent the reader from latching onto stale evidence.

**`query_facts(subject?, verb?, status?, object_contains?, fact_type?)`:** Structured fact lookup with semantic verb expansion. Verb expansion combines manual synonym clusters (e.g., "visited" ↔ "returned from" ↔ "completed" ↔ "hiked") with embedding-based similarity matching (cosine threshold 0.53), ensuring that vocabulary mismatches between the reader's query verbs and the extracted fact verbs do not prevent retrieval. When verb expansion widens a query significantly, the SQL result limit is automatically increased to prevent rare-verb facts from being cut off. Results are reranked by embedding similarity to the reader's question. When few results match, a secondary embedding search runs on the `what` field of candidate facts.

**`expand_topic(tag, depth?, collapse_tags?)`:** Demand paging. Loads a topic's content at SEGMENTS depth (segment summaries, ~2,000 tokens) or FULL depth (original text, ~8,000+ tokens). The optional `collapse_tags` parameter allows collapsing other topics in the same call, saving a round-trip. LRU auto-eviction triggers when the working set exceeds the 30,000-token tag context budget — coldest topics (by `last_accessed_turn`) are collapsed to SUMMARY depth first, then removed entirely if needed.

**`remember_when(query, time_range)`:** Time-scoped variant of `find_quote`. Accepts both relative presets ("last_7_days," "last_30_days," "last_90_days") and absolute date ranges. Implementation: overfetches conversation excerpts at 4x the result limit, then post-filters by `session_date` parsed from segment metadata. In addition to conversation excerpts, `remember_when` returns structured `facts_in_window` — completed experience facts whose session dates fall within the queried time range, retrieved via direct SQL date-range query. This gives the reader immediate access to dated events (e.g., "User got back from a day hike to Muir Woods" on March 10) without requiring a separate `query_facts` call, and is critical for temporal ordering questions where the reader needs events with dates attached.

**`recall_all()`:** Loads all tag summaries, budget-bounded to 30,000 tokens. Used for bird's-eye orientation on broad questions like "what are my interests?" Only one case in our 100-question evaluation used this tool.

### 3.6 Tool Loop and Context Assembly

The tool loop runs synchronously within a single user-facing request, allowing up to 10 continuation rounds (configurable). Key mechanisms:

**Anti-repetition.** `presented_refs` (segment dedup) and `presented_facts` (fact dedup) accumulate across all rounds. Already-shown content is suppressed in subsequent tool calls, preventing the reader from receiving the same information twice.

**Redundant loop detection.** If the last 3 rounds all used `find_quote` and all returned results, the loop is terminated — the reader has enough information and is circling. Tool choice is relaxed from "required" to "auto" after the first round.

**Budget-aware context management.** The system maintains a token budget across all tool loop rounds. The assembler allocates tokens across priority-ordered blocks: core context (system instructions, 18K max), tag context (working set segments, 30K max), facts (relevant structured facts, 20K max), conversation history (recent turns), and context hint (topic list, 2K max). When the budget is exceeded — whether from `expand_topic` loading deeper segments or from accumulated tool results growing the conversation — the paging manager evicts the coldest topics via LRU, collapsing them back to summary depth, and reassembles the context block to free headroom. The context hint — a compact `<context-topics>` block — lists all available topics with their depth, token costs, and tool instructions, serving as the reader's navigation map.

### 3.7 Tool Output Management for Agentic Workloads

The architecture described in Sections 3.1--3.6 applies to any interaction that generates context over time. In agentic workloads — coding assistants, multi-tool pipelines, research agents — the dominant source of context growth is not conversational turns but tool outputs: file reads, search results, command output, API responses. These outputs can be individually large (a file read may return 10--50KB) and collectively massive (a debugging session may invoke hundreds of tool calls). Left unmanaged, tool outputs rapidly consume the context window with content that is immediately useful but rarely referenced again.

VC handles tool outputs through a separation that mirrors the what/how distinction described in Section 2.3. Tool output *content* — the raw bytes returned by a file read or grep — is separated from the *conversational context* around it: why the tool was called, what the user was investigating, what conclusion was drawn.

**Proxy-layer interception.** Before tool results reach the engine, a `ToolOutputInterceptor` scans each `tool_result` block against configurable per-tool rules (fnmatch patterns with size thresholds, default 8KB). Outputs exceeding the threshold are truncated on line boundaries with a configurable head:tail ratio (default 60:40), and a notice is inserted directing the reader to `find_quote` for full-content search. VC's own tool results (`vc_find_quote`, `vc_expand_topic`, etc.) are never truncated, preventing feedback loops.

**Separate indexing.** The full content of truncated outputs is stored in a dedicated `tool_outputs` table with its own FTS5 index (capped at 512KB per output). This content is searchable via `find_quote`, which queries both the segment FTS5 index and the tool output index, returning results with distinct `match_type` markers. The tool output content is thus available for retrieval without occupying space in the context window or in segment summaries.

**Exclusion from the interpretive pipeline.** When the engine extracts conversation history for tagging and compaction, `tool_use` and `tool_result` blocks are stripped — only the text portions of messages flow into the `Message` objects that the tagger and compactor process. This means tool output substance (raw file contents, grep matches, bash output) never pollutes the tag vocabulary or inflates summaries. The *conversational text* around tool calls — the user's reasoning, the assistant's analysis, the conclusions drawn — passes through the normal interpretive pipeline and is tagged, compressed, and fact-extracted like any other content.

**Referential integrity.** The message filter enforces that `tool_use` and `tool_result` blocks are kept or dropped as pairs, iterating until stable. This prevents orphaned tool results that would confuse the API.

The effect is that in a long coding session, VC's context window contains compressed summaries of what the developer was working on and what they concluded (the interpretive layer), with structured facts capturing key decisions and outcomes, while the raw tool outputs that drove those conclusions are indexed separately and retrievable on demand. The context carries the narrative of the work — the why and how — without being overwhelmed by the raw data that informed it.

This capability has been tested in production Claude Code sessions but is not included in the LongMemEval evaluation, which uses conversational haystacks without tool calls. Systematic evaluation on tool-heavy agentic workloads is an area of active work.

---

## 4. Experimental Setup

### 4.1 Benchmark: LongMemEval

LongMemEval (Wu et al., 2024) is a benchmark of 500 questions evaluating long-term conversational memory. Each question is paired with a synthetic multi-session conversation "haystack" of 50--300 sessions comprising approximately 100,000--120,000 tokens. Questions span five categories:

- **Knowledge-update:** Requires tracking the latest value of information that changed over time (e.g., "What is my current coffee-to-water ratio?")
- **Multi-session:** Requires synthesizing information from multiple conversation sessions (e.g., "How many model kits have I worked on?")
- **Temporal-reasoning:** Requires understanding temporal relationships (e.g., "Which happened first, the coffee maker purchase or the stand mixer malfunction?")
- **Single-session (user/assistant/preference):** Requires recalling specific details from a single session

We evaluate on 100 random questions sampled in 5 batches of 20 (seeds: 42, 99, 777, 1234, 2025), yielding the following distribution: 17 knowledge-update, 26 multi-session, 28 temporal-reasoning, 13 single-session-user, 11 single-session-assistant, 5 single-session-preference.

### 4.2 Model Selection Rationale

A deliberate choice underlies our model selection: both VC and the baseline use **Claude Sonnet 4.5**, Anthropic's mid-tier model optimized for balanced cost and performance — not the flagship Claude Opus. At the time of evaluation, Sonnet 4.5 was priced at approximately $3/$15 per million input/output tokens, roughly 1.7x cheaper than Opus ($5/$25).

This choice is intentional, not budgetary. We hypothesize that VC's structured context management can elevate a cost-efficient model to accuracy levels that even a flagship model with raw full context cannot achieve. If VC at 95% accuracy with Sonnet outperforms full-context Sonnet at 33%, the implication is clear: **investing in context management yields greater returns than investing in model capability** for conversational memory tasks. A system architect choosing between a $75/M-output-token flagship model with raw context and a $15/M-output-token mid-tier model with VC would achieve 3x the accuracy at one-fifth the per-token cost — compounding savings on both axes.

This has practical significance for production deployments where per-query cost scales with user volume. The combination of VC's 2.2x token reduction and Sonnet's 1.7x lower per-token price versus Opus yields an estimated **3.7x cost reduction** compared to an Opus full-context deployment at standard scale — growing to **22x** at ~926K tokens where Opus baseline costs $4.42/query and answers incorrectly, while VC + Sonnet costs $0.20/query and answers correctly.

### 4.3 VC Configuration

- **Ingestion model:** MiMo-V2-Flash (Xiaomi) via OpenRouter for all tagging, summarization, fact extraction, and compaction
- **Reader model:** Claude Sonnet 4.5 (Anthropic) via Anthropic API
- **Judge model:** Gemini 3 Pro Preview (Google) using LongMemEval's official judge prompts verbatim
- **Context budget:** 64K tokens (deliberately below the model's maximum to test compression effectiveness)
- **Max tool loops:** 10
- **Token counter:** Character-based estimate (len/4)
- **Compaction:** Summary ratio 15%, min 200 tokens, max 2,000 tokens
- **Ingestion:** Fully cached per question; each question's conversation is ingested once, then reader runs against the cached store

### 4.4 Baseline

- **Reader model:** Claude Sonnet 4.5 (Anthropic) — the same model as the VC reader
- **Context:** Full conversation history loaded into context (~117,582 tokens average)
- **No tools, no compression, no memory system** — the model receives the raw conversation
- **Judge:** Same Gemini 3 Pro Preview with identical LongMemEval judge prompts

This baseline is deliberately the strongest possible: the model has *perfect recall by construction* since all information exists within its context window, and uses the *same reader model* as VC. Any accuracy difference is attributable solely to the memory system, not model capability.

### 4.5 Evaluation Metrics

- **Accuracy:** LLM-as-judge using LongMemEval's official category-specific templates (standard, temporal, knowledge-update)
- **Token count:** Input tokens including all tool call continuation rounds
- **Cost:** USD cost per question (API pricing)
- **Latency:** End-to-end elapsed time per question, with phase-level breakdowns for VC (ingest, compact, retrieve, query) and per-tool call durations
- **Tool analysis:** Chain pattern classification, per-tool usage statistics, useful/wasted call ratio

---

## 5. Results

### 5.1 Main Results

| System | Accuracy | Avg Tokens | Avg Cost/Q | Total Cost |
|--------|----------|------------|------------|------------|
| Full-context baseline | 33/100 (33%) | 117,582 | $0.36 | $35.56 |
| Virtual Context | 95/100 (95%) | 52,347 | $0.16 | $15.99 |

VC achieves 2.9x the accuracy of the full-context baseline while using 2.2x fewer tokens and costing 2.2x less. The improvement is not marginal — it represents a categorical shift from an unusable system (33% accuracy) to a highly reliable one (95%).

### 5.2 Per-Category Breakdown

| Category | Count | VC | Baseline | Delta |
|----------|-------|----|----------|-------|
| knowledge-update | 17 | 100.0% (17/17) | 29.4% (5/17) | +70.6pp |
| multi-session | 26 | 88.5% (23/26) | 15.4% (4/26) | +73.1pp |
| temporal-reasoning | 28 | 92.9% (26/28) | 32.1% (9/28) | +60.8pp |
| single-session-user | 13 | 100.0% (13/13) | 46.2% (6/13) | +53.8pp |
| single-session-assistant | 11 | 100.0% (11/11) | 72.7% (8/11) | +27.3pp |
| single-session-preference | 5 | 100.0% (5/5) | 20.0% (1/5) | +80.0pp |

VC achieves perfect accuracy across all single-session categories and knowledge-update, with the largest absolute improvement on single-session-preference (+80pp) and multi-session (+73.1pp). The baseline performs worst on multi-session (15.4%) where information is scattered across many sessions — precisely the scenario where the "lost in the middle" effect is most severe.

### 5.3 Token Efficiency

VC uses strictly fewer tokens than the baseline in the vast majority of questions. Across all 100 questions:

- **VC total:** 5,234,716 tokens ($15.99)
- **Baseline total:** 11,758,181 tokens ($35.56)
- **Reduction:** 55.5% fewer tokens

Among the 50 cases where VC answered correctly and the baseline did not:

- **Average token reduction:** 6.7x
- **Maximum:** 9.7x
- **Minimum:** 5.0x
- **Median:** 6.6x

The most dramatic example is `b3c15d39` (remote shutter release delivery time): VC used 8,349 tokens to correctly answer "5 days," while the baseline used 100,648 tokens and failed. The key detail — the arrival date — was buried in a `basketball-tournament` topic tag, completely unrelated to the question. VC's full-text search found it in an unrelated segment; the baseline model missed it in 100K tokens of raw conversation.

### 5.4 Latency Analysis

VC's tool loop introduces additional API round-trips compared to the single-call baseline. We measured end-to-end elapsed time for all 100 questions at LongMemEval's standard ~125K token haystack size:

| Metric | VC | Baseline |
|--------|-----|----------|
| Mean elapsed | 12.7s | 8.7s |
| Median elapsed | 11.6s | 7.2s |
| Min | 7.5s | 3.9s |
| Max | 25.2s | 25.4s |

At this scale, VC's absolute elapsed time is ~4s longer than the baseline. However, both systems make at least one LLM call — the baseline's ~8.7s represents the irreducible cost of a single inference pass over 125K tokens. VC's marginal overhead is therefore **~4s** (one additional API round-trip for the tool loop continuation), not the full 12.7s.

**Phase breakdown (VC only):**
- **Query phase** (reader LLM calls + tool execution): 10.9s average — dominates total latency
- **Retrieve phase** (context assembly + embedding search): ~1.5s average
- **Ingest/compact:** 0s at query time (amortized asynchronously during conversation; see Section 5.4.1)

**Per-tool call latency:** Across 180 tool invocations, individual tool calls average **609ms** (median 823ms, max 3.1s). The dominant cost is not tool execution but the LLM continuation round-trips — each round requires a full API call for the reader to process tool results and decide whether to continue.

**Tool call count distribution:**

| Calls | Questions | Cumulative |
|-------|-----------|------------|
| 1 | 48 (48%) | 48% |
| 2 | 34 (34%) | 82% |
| 3 | 12 (12%) | 94% |
| 4+ | 6 (6%) | 100% |

82% of questions resolve in 1--2 tool calls, meaning they incur at most one additional API round-trip beyond what the baseline requires. The 6% of questions requiring 4+ calls correspond to complex multi-step reasoning tasks (temporal ordering across sessions, counting entities scattered across topics) where the additional latency is unavoidable given the retrieval complexity.

Notably, the maximum latency for both systems is similar (~25s), suggesting that the hardest questions are latency-bound by LLM thinking time regardless of the approach.

#### 5.4.1 Scale Inversion: From Latency Penalty to Latency Advantage

The ~4s marginal overhead measured at 125K tokens reflects a fixed cost: one or two additional API round-trips through VC's tool loop. This cost is **constant** regardless of conversation history size — VC's reader never sees the raw conversation, only the managed context window (~25–70K tokens of compressed summaries and tool results).

Baseline latency, by contrast, scales with conversation length — every query requires a full inference pass over the entire history. We measured this directly using merged haystacks of 660K and 875K tokens (Section 6.3):

| Conversation Scale | Baseline Latency | VC Latency | VC Marginal Delta |
|-------------------|-----------------|------------|-------------------|
| 125K tokens (standard) | 8.7s | 12.7s | +4.0s |
| 660K tokens (6Q merged) | 30s | ~11s | **−19s** |
| 875K tokens (8Q merged) | 86s | ~11s | **−75s** |

The crossover occurs around **200–300K tokens** — beyond this point, VC is both faster and cheaper than the baseline. At 875K tokens, VC is approximately 8x faster. In a production deployment where conversations accumulate over months, VC's constant-time query latency represents a fundamental architectural advantage over approaches that re-read the full history on every query.

Furthermore, in production deployments using VC as a proxy, ingestion (tagging, fact extraction, compaction) runs **asynchronously** after each message — the user never waits for it. The only user-facing latency is the reader query, which remains constant at ~11s regardless of whether the conversation history is 10K or 10M tokens.

### 5.5 Cost Analysis

The raw cost comparison — $0.16/question (VC) vs. $0.36/question (baseline) — understates VC's economic advantage when model-tier selection is factored in.

Both VC and the baseline use Claude Sonnet 4.5, a mid-tier model priced at approximately $3/$15 per million input/output tokens. The flagship alternative (Claude Opus) costs $5/$25 — a 1.7x premium. If a production system chose Opus for its superior reasoning to compensate for the challenges of full-context retrieval, the per-question cost would rise to approximately $0.60/question. Against this realistic alternative:

| Configuration | Accuracy | Avg Cost/Q | vs. VC |
|---------------|----------|------------|--------|
| VC + Sonnet | 95% | $0.16 | — |
| Full-context + Sonnet | 33% | $0.36 | 2.2x more |
| Full-context + Opus (est.) | ≤33%* | ~$0.60 | **3.7x more** |

*\*Opus may improve over Sonnet on full-context retrieval, but Liu et al. (2023) show the lost-in-the-middle effect persists across model scales. Our merged haystack experiments (Section 6.3) confirm this empirically: Opus 4.6 fails on the ~926K trip ordering question despite being the most capable model available.*

The implication is that **context management is a higher-leverage investment than model capability** for long-running LLM applications. VC on a $3/M-token model achieves accuracy that no amount of spending on per-token model quality can replicate with raw full context. This inverts the conventional wisdom that harder tasks require bigger models — structured context allows smaller models to punch above their weight class.

For production deployments serving thousands of users, the compounding effect is significant: VC's 2.2x token reduction × Sonnet's 1.7x lower per-token price versus Opus yields roughly **3.7x cost savings per query** while delivering 3x the accuracy. At scale (Section 6.3), the advantage compounds further: Opus baseline at ~926K tokens costs $4.42/query and answers incorrectly, while VC + Sonnet costs $0.20/query and answers correctly — a **22x cost reduction** with correct answers replacing wrong ones.

More critically, cost scaling compounds with accuracy degradation. At ~926K tokens, Opus 4.6 — the most capable model available — costs $4.42/query and answers incorrectly. VC + Sonnet costs $0.20/query and answers correctly. The problem is not that the model is too cheap; the problem is that raw full context does not work at this scale regardless of what you spend. Structured context management is the only approach in our evaluation that maintains both accuracy and bounded cost as conversations grow.

### 5.6 Tool Chain Analysis

Across all 100 questions, we classified tool call patterns into 8 emergent categories:

| Pattern | Count | Description |
|---------|-------|-------------|
| single_search | 21 | One `find_quote` call suffices |
| iterative_search | 17 | Multiple `find_quote` with refined queries |
| facts_search_interleave | 3 | Alternating `query_facts` and `find_quote` |
| facts_search_expand | 3 | `query_facts` → `find_quote` → `expand_topic` |
| temporal_recall | 3 | `remember_when` for time-scoped questions |
| facts_temporal | 1 | `query_facts` → `remember_when` |
| facts_drill | 1 | `query_facts` → `find_quote` deep dive |
| direct_expand | 1 | `expand_topic` directly (no search first) |

**Tool usage frequency:**
- `find_quote`: 76% of all invocations
- `query_facts`: 13%
- `remember_when`: 7%
- `expand_topic`: 4%
- `recall_all`: <1% (used in 1 case)
- `collapse_tags`: 0% (never used)

**Key findings:**
- 82% of questions were answered in 1--2 tool calls
- 7 of 26 multi-tool cases exhibited strategy pivots — the reader switched tool types after initial failures
- `find_quote` dominates because it provides the best combination of breadth (full-text + semantic search) and specificity (exact passages)
- `recall_all` was almost never used, suggesting that the context hint provides sufficient orientation for most questions
- `collapse_tags` was never invoked, indicating the 30K tag context budget was sufficient for single-question retrieval

**Exceptional chains (from 50 VC-correct/baseline-wrong cases):**

*Trip ordering (`gpt4_7f6b06db`):* 9-step chain using 5 tool types. The reader used temporal narrowing (`remember_when`) to identify relevant time periods, demand-paged a topic (`expand_topic`), performed targeted searches (`find_quote`), cross-validated with structured facts (`query_facts`), managed context budget (`collapse_topic` + re-expand), arriving at the correct chronological ordering of three trips.

*Model kit count (`gpt4_59c863d7`):* 11 steps alternating between `find_quote` and `query_facts`. Neither tool alone had all 5 kits — they were complementary. Quotes found some kits; facts confirmed completion status for others; remaining kits required additional quote evidence from unrelated topics.

*Conflicting evidence resolution (`6a1eabeb`):* The reader found two conflicting personal best times (27:12 and 25:50). By synthesizing quote excerpts, structured facts (one said "achieved PB of 27:12," another said "aims to beat PB of 25:50"), and temporal ordering (May 23 vs. May 30 sessions), the reader correctly identified 25:50 as the current personal best.

### 5.7 Fact Prefetch Analysis

The system prompt contains pre-loaded structured facts (from embedding-matched tags) before any tool calls are made. We analyzed how often these prefetched facts already contain the answer:

**Tier 1 — Fact directly answers (no tool call theoretically needed):**
- `gpt4_d6585ce9`: "Who were the parents sitting nearby at the concert?" → Prefetched fact: `user | attended | Coldplay concert with parents sitting nearby`
- `60d45044`: "What type of rice do I usually buy?" → Prefetched fact: `user | usually buys | Japanese short-grain rice`
- `ed4ddc30`: "How many eggs did I buy for Easter?" → 19 matching facts about Easter eggs in prefetch

**Tier 2 — Fact has answer but tool adds confirming context:**
- `681a1674`: "How many Marvel movies am I re-watching?" → Fact listed all 4 movies; `find_quote` confirmed the full list
- `0977f2af`: "What model Instant Pot?" → Fact: `user | bought | Instant Pot Duo 7-in-1`; tool verified details

**Tier 3 — Facts insufficient, tool required:**
- `099778bb`: "What percentage of women in leadership positions?" → Specific statistic (20%) not captured as structured fact
- `b3c15d39`: "Days to receive remote shutter release?" → Arrival date compressed away; only order date in facts

This analysis reveals that the two-pass fact extraction pipeline creates a high-quality prefetch layer. For approximately 30% of correctly answered questions, the prefetched facts alone could theoretically answer the question — tool calls primarily serve as verification.

### 5.8 Knowledge-Update Deep Dive

VC achieves 100% accuracy on knowledge-update questions (17/17) compared to 29.4% baseline (5/17). This is the strongest category differential (+70.6 percentage points). The mechanism is the supersession chain:

When a user says "I switched from French press to Chemex" in a later session, and an earlier session discussed the French press in detail, VC's fact extraction creates a new fact (`user | uses | Chemex`) and the supersession checker marks the old fact (`user | uses | French press`) as superseded. At query time, the reader sees only the latest value.

The baseline, by contrast, sees both values in the raw conversation with no indication of which is current. Since the earlier mention (French press) is typically discussed more extensively — it was the subject of detailed conversations — the baseline model consistently picks the older, more frequently mentioned value. This is a systematic failure mode that raw context cannot address: the correct answer appears once, while the outdated answer appears multiple times.

### 5.9 Failure Analysis

VC incorrectly answered 5 of 100 questions. All 5 are also incorrect for the baseline:

| QID | Type | Failure Mode |
|-----|------|-------------|
| `09ba9854` | multi-session | Items spread across unrelated topics; exhaustive scan incomplete |
| `bf659f65` | multi-session | Fact verb mismatch: vinyl stored as "had" not "purchased" |
| `gpt4_372c3eed` | multi-session | Reader reasoning error despite correct tool results |
| `gpt4_f420262c` | temporal-reasoning | Reader got chronological order wrong despite finding all dates |
| `gpt4_f420262d` | temporal-reasoning | Reader chose wrong entity from multiple candidates in same session |

Critically, in 3 of 5 failures (`gpt4_372c3eed`, `gpt4_f420262c`, `gpt4_f420262d`), the correct information was returned by the tools but the reader reasoned incorrectly. These are pure reader comprehension failures, not retrieval failures — suggesting that with a stronger reader model, VC's accuracy ceiling is even higher.

The remaining 2 failures (`09ba9854`, `bf659f65`) involve retrieval gaps: items scattered across unrelated topics that the reader's search queries didn't fully cover, and verb vocabulary mismatches in fact queries. These point to optimization opportunities in broader search strategies and richer verb normalization.

Only 1 question was answered correctly by the baseline but incorrectly by VC (`gpt4_372c3eed`), a multi-session question where the baseline benefited from seeing specific adjacent context that VC's compression had summarized.

---

## 6. Discussion

### 6.1 Why Compressed Context Outperforms Full Context

The 95% vs. 33% accuracy gap demands explanation. How can a model seeing less information perform dramatically better than a model seeing everything?

Three factors contribute:

**Attention concentration.** Full-context models must attend to ~118K tokens uniformly, but attention is a finite resource. With VC, the reader sees ~52K tokens of curated, topic-organized context. Every token has been selected for relevance — either it's a compressed summary of a relevant topic, a structured fact, or a recent conversation turn. There is no filler, no irrelevant sessions, no noise. The signal-to-noise ratio is dramatically higher.

**Active retrieval vs. passive scanning.** The full-context model must passively scan for the answer within 118K tokens. The VC reader actively queries for specific information: "find_quote('remote shutter release')" directly surfaces the relevant passage. This is the difference between searching a well-indexed database and reading an unsorted log file.

**Knowledge organization.** VC organizes information by topic, with structured facts providing a queryable index. A question about "my personal best 5K time" can be answered by querying `query_facts(subject="user", object_contains="5K")` — a targeted lookup in a structured store. The full-context model must identify the relevant passage among dozens of running-related conversations.

### 6.2 Emergent Retrieval Behaviors

Perhaps the most striking finding is that the reader develops sophisticated retrieval strategies without any explicit programming. The 8 tool chaining patterns we observe (Section 5.4) emerge entirely from the interaction between the reader's reasoning capabilities, the structured context it receives, and the tool descriptions in its prompt. Three emergent behaviors deserve particular attention.

**Strategy pivots under failure.** In 7 of 26 multi-tool cases, the reader changed its approach after initial tool calls returned insufficient results. For example, in `gpt4_59c863d7` (model kit counting), the reader began with `find_quote` searches, found some kits, then pivoted to `query_facts` to discover additional kits through structured data that text search had missed. This is not a pre-programmed fallback chain — it is the reader reasoning that a different tool type might surface different information. The reader treats each tool as a distinct *epistemological lens*: text search finds verbatim mentions, fact queries find structured relationships, temporal search constrains by time, and topic expansion provides full narrative context. When one lens proves insufficient, the reader reasons about which alternative lens might reveal what the first could not.

**Cross-topic discovery.** The reader learns to distrust topic boundaries. In `b3c15d39` (remote shutter release), the arrival date was stored under `basketball-tournament` — a completely unrelated topic tag. The reader's `find_quote("remote shutter release")` searched across *all* stored text regardless of topic organization, discovering the answer in an unexpected location. This behavior is critical: it means the reader does not treat the topic-organized context hint as exhaustive, but as a navigational aid that can be bypassed through full-text search.

**Conflicting evidence resolution.** In `6a1eabeb` (5K personal best), the reader encountered two contradictory times (27:12 and 25:50) from different sessions. Rather than picking the more frequently mentioned value (which the baseline does, incorrectly), the reader synthesized evidence across multiple tool types — finding that a structured fact said "aims to beat PB of 25:50" (implying 25:50 was the standing record) and that the 25:50 mention came from a later session date than the 27:12 mention. This multi-source triangulation is an emergent reasoning pattern: the reader uses temporal metadata, structured fact semantics, and raw text evidence to resolve contradictions that any single source would leave ambiguous.

These emergent behaviors suggest that tool-augmented retrieval over structured context is not merely a performance optimization but enables qualitatively different reasoning strategies that passive full-context reading cannot support. The reader becomes an active investigator rather than a passive scanner.

### 6.3 Context Rot: Baseline Accuracy Degrades at Scale

LongMemEval's standard haystacks average ~125K tokens — well within the effective attention range of modern frontier models. Section 5.4.1 predicts that baseline accuracy will degrade as context scales beyond this ceiling. We validated this prediction directly by constructing merged haystacks of 660K and 950K tokens — representing 6–10 months of daily interaction — and testing both frontier model baselines and VC-assisted readers against them.

#### 6.3.1 Experimental Setup

We selected 8 LongMemEval questions with overlapping date ranges (January–November 2023) and merged their haystack sessions into a single chronologically interleaved conversation. This produces a realistic scenario: a user with months of diverse conversation history asks a question that requires locating specific information buried among hundreds of unrelated sessions.

**Merged haystack properties:**

| Haystack | Sessions | Tokens | Unique Days | Date Span | Interleaving |
|----------|----------|--------|-------------|-----------|-------------|
| 6Q merge | 293 | ~660K | 130 | 9.5 months | 36% transitions |
| 8Q merge | 374 | ~926K | 158 | 9.9 months | 55% transitions |

Interleaving measures the fraction of adjacent sessions originating from different source questions — higher values indicate more realistic temporal mixing rather than blocked clusters.

All questions were verified correct on both VC (Sonnet 4.5 reader) and baseline (Gemini 3 Pro) at their individual ~125K token haystack size before merging.

#### 6.3.2 Results: Baseline Accuracy Collapses, VC Holds

**6Q merged haystack (660K tokens) — `2e6d26dc`: "How many babies were born to friends and family members?"** (Gold: 5. Category: multi-session counting.)

| Model | Mode | Input Tokens | Cost | Time | Answer | Verdict |
|-------|------|-------------|------|------|--------|---------|
| Gemini 3 Pro | Baseline | 659,191 | $0.82 | 30s | 5 | CORRECT |
| GPT-5.4 | Baseline | 622,035 | $1.24 | 16s | 6 (included adopted child) | WRONG |
| Sonnet 4.5 (1M beta) | Baseline | 701,431 | $2.11 | 35s | 5 | CORRECT |
| Opus 4.6 (1M beta) | Baseline | 701,431 | $3.51 | 44s | 5 | CORRECT |
| Gemini 3 Pro | VC | ~25K | $0.22 | 19s | 5 | CORRECT |
| Sonnet 4.5 | VC | 71,915 | $0.22 | 19s | 5 | CORRECT |
| GPT-5.4 | VC | 56,003 | $0.11 | 14s | 5 | CORRECT |

At 660K tokens, three of four frontier baselines correctly count 5 babies — Gemini 3 Pro, Sonnet 4.5, and Opus 4.6 all distinguish births from an adoption mentioned in the text. GPT-5.4 overcounts to 6 by including an adopted child that doesn't meet the "born" criterion. All three VC readers answer correctly. The 660K scale represents a moderate challenge where most frontier models can still succeed on counting tasks, unlike the ~926K scale where all baselines fail on temporal ordering.

**8Q merged haystack (~926K tokens) — `gpt4_7f6b06db`: "What is the order of the three trips I took in the past three months, from earliest to latest?"** (Gold: Muir Woods → Big Sur → Yosemite. Category: temporal reasoning.)

| Model | Mode | Input Tokens | Cost | Answer | Verdict |
|-------|------|-------------|------|--------|---------|
| Gemini 3 Pro | Baseline | 330,297 | $1.09 | Yosemite → Big Sur → Dubai (wrong trips, wrong order) | WRONG |
| GPT-5.4 | Baseline | 810,246 | $1.66 | Yosemite → Big Sur → Muir Woods (reversed) | WRONG |
| Sonnet 4.5 (1M beta) | Baseline | ~926K | — | HTTP 500 (server error) | FAILED |
| Opus 4.6 (1M beta) | Baseline | 884,115 | $4.42 | Yosemite → Big Sur → Eastern Sierra (planned, not taken) | WRONG |
| Gemini 3 Pro | VC | ~25K | $0.76 | Muir Woods → Big Sur → Yosemite | CORRECT |
| GPT-5.4 | VC | ~25K | $0.15 | Muir Woods → Big Sur → Yosemite | CORRECT |
| Sonnet 4.5 | VC | ~25K | $0.20 | Muir Woods → Big Sur → Yosemite | CORRECT |
| Opus 4.6 | VC | ~25K | $0.69 | Muir Woods → Big Sur → Yosemite | CORRECT |

At ~926K tokens, all four frontier baselines fail — including Opus 4.6, the most capable and expensive model available ($4.42/query). Gemini identified three trips but picked the wrong set (substituting a Dubai itinerary for Muir Woods) and reversed their temporal order; GPT-5.4 found the correct three trips but reversed their chronological sequence; Opus correctly identified some trips but included a merely *planned* Eastern Sierra trip while missing the completed Muir Woods day hike; and Sonnet's 1M beta returned a server error, unable to process the context at all. The correct information is present in the context; the models simply cannot reliably process it at this scale.

All four VC-assisted readers answer correctly using the same ingested store (~25K token reader prompt), at costs ranging from $0.15 (GPT-5.4) to $0.76 (Gemini 3 Pro). Opus with VC costs $0.69 — 6x cheaper than its baseline, and correct instead of wrong. VC's structured fact store surfaces the three trip facts with dates directly, enabling correct temporal ordering without scanning 926K tokens of raw conversation. The `remember_when` tool returns both conversation excerpts and structured facts within the queried time window, giving the reader immediate access to dated events. Supersession chains automatically deduplicate misdated fact mentions — a critical capability when the same trip is referenced across multiple sessions with ambiguous temporal anchoring.

#### 6.3.3 Analysis: Context Rot

All four frontier baselines degrade as raw context grows beyond ~200K tokens. At 660K tokens, counting precision fails — models lose the ability to apply fine-grained semantic filters (distinguishing "born" from "adopted," or counting twin births as one event). At ~926K tokens, the degradation is more severe and universal: temporal ordering reverses (Gemini, GPT-5.4), trip identification fails (Gemini substitutes a Dubai itinerary for a day hike), the model confuses planned trips with completed ones (Opus includes an Eastern Sierra trip that was never taken), or the model cannot process the context at all (Sonnet returns HTTP 500). Even Opus 4.6 — the most capable model available at $4.42/query — fails to correctly identify which of several mentioned trips were actually completed versus merely planned.

VC readers, by contrast, achieve 4/4 correct at ~926K tokens. The structured fact store reduces the retrieval problem from scanning 926K tokens to querying ~25K tokens of compressed summaries plus structured facts with dates. The `remember_when` tool's `facts_in_window` response surfaces completed experience facts within the queried date range, giving readers direct access to "User got back from a day hike to Muir Woods" (March 10), "User got back from a road trip to Big Sur and Monterey" (April 20), and "User got back from solo camping trip to Yosemite" (May 15) — with dates attached. Embedding-based supersession automatically resolves misdated duplicate mentions that arise when users reference past trips in later conversations.

**The key takeaway:** VC is immune to context *length* scaling — its retrieval cost and accuracy don't depend on raw token count. Baselines degrade predictably as conversation history grows, with failures becoming more severe (imprecise counting → temporal reversal → wrong trip identification → total failure) as token counts increase from 125K to ~926K. At ~926K tokens, VC achieves 100% accuracy across four reader models (Sonnet 4.5, Gemini 3 Pro, GPT-5.4, Opus 4.6) where all four baselines fail — including the most expensive model at 6x the cost of its VC-assisted counterpart.

#### 6.3.4 Cost and Latency Inversion at Scale

The merged haystack results reveal a striking inversion in both cost and latency:

| Scale | Baseline Cost | VC Cost | Savings | Baseline Accuracy | VC Accuracy |
|-------|-------------|---------|---------|-------------------|-------------|
| 125K (standard) | $0.36 | $0.16 | 2.2x | 33% | 95% |
| 660K (6Q merge) | $0.82–3.51 | $0.11–0.22 | 4–16x | 3/4 (75%) | 3/3 (100%) |
| ~926K (8Q merge) | $1.09–4.42 | $0.15–0.76 | 2–6x | 0/4 (0%) | 4/4 (100%) |

At standard LongMemEval scale, VC is already 2.2x cheaper. At 660K tokens, the cost story becomes dramatic even though most baselines still answer correctly:

| Model | Mode | Input Tokens | Cost | Time | Correct |
|-------|------|-------------|------|------|---------|
| Gemini 3 Pro | Baseline | 659K | $0.82 | 30s | Yes |
| GPT-5.4 | Baseline | 622K | $1.24 | 16s | No |
| Sonnet 4.5 | Baseline | 701K | $2.11 | 35s | Yes |
| Opus 4.6 | Baseline | 701K | $3.51 | 44s | Yes |
| Gemini 3 Pro | VC | ~25K | $0.22 | 19s | Yes |
| Sonnet 4.5 | VC | ~72K | $0.22 | 19s | Yes |
| GPT-5.4 | VC | ~56K | $0.11 | 14s | Yes |

VC readers process 10–28x fewer tokens than baselines, resulting in 4–16x cost reductions while matching or exceeding baseline accuracy. The cheapest correct baseline (Gemini 3 Pro at $0.82) is 7.5x more expensive than the cheapest VC reader (GPT-5.4 at $0.11). Opus 4.6 answers correctly at $3.51 — the same answer VC delivers for $0.11–0.22 (16–32x cheaper). VC is also faster at this scale: 14–19s versus 16–44s for baselines, because VC's reader prompt is constant-sized regardless of conversation length.

Ingestion runs incrementally as each conversation turn is processed, using the cheapest available models for simple classification tasks. In our evaluation, MiMo-V2-Flash ($0.14/MTok) handles all tagging, fact extraction, and compaction at **$0.0008 per turn** — under a tenth of a cent. A 660K-token conversation spanning 293 sessions and 2,981 turns costs $2.32 total to ingest ($1.82 tagging, $0.50 compaction). This is a fixed investment that pays for itself quickly: at query time, the baseline must re-read the entire conversation history, while VC queries a constant-sized managed window (~65K tokens) regardless of conversation length.

The cost crossover occurs at **65K tokens** — the size of VC's reader window. Beyond this point, every baseline query on Sonnet 4.5 ($3/MTok input) costs more than VC's constant $0.195/query. This threshold is reached remarkably fast in practice: ~294 conversational turns (a few weeks of casual use), ~130 agentic tool-call turns (**13 minutes** of a coding session at 10 calls/minute), or ~65 long-form writing turns. By 660K tokens, baseline query cost has grown to $0.33–3.51 per query while VC remains at $0.11–0.22 — and the $2.32 cumulative ingestion investment has been recouped many times over through per-query savings.

At ~926K tokens, cost savings compound with accuracy: baseline costs reach $1.09–4.42 per query while all four frontier baselines fail entirely — including Opus 4.6 at $4.42. VC readers answer correctly at $0.15–0.76 per query, delivering a 2–6x cost reduction while achieving 100% accuracy where baselines achieve 0%.

This inversion is structural, not incidental. Baseline cost and latency scale linearly with conversation length (every query re-reads the full history). VC cost and latency are approximately constant (every query reads the same-sized managed window regardless of total store size). The crossover point is approximately **200–300K tokens** — beyond which VC is dominant on cost, latency, and accuracy.

### 6.4 Cross-Benchmark Validation: MRCR Long-Context Retrieval

To validate VC's effectiveness beyond LongMemEval, we evaluated on OpenAI's Multi-Round Coreference Resolution (MRCR) benchmark — a needle-in-a-haystack retrieval task requiring verbatim reproduction of specific content from long multi-turn conversations. We selected a 4-needle question (`4n_0543`) with a 652K-token context (~2,890 messages), where the task requires identifying and reproducing the 2nd chronological instance of a play scene about keys from among four structurally similar scenes scattered across the conversation.

This question is adversarially challenging: multiple play scenes share the same theme (keys, apartments, roommates), and one scene contains a literal "Scene 2" label in its title that acts as a decoy for the ordinal "2nd" in the question. Scoring is deterministic string matching (Python `SequenceMatcher`), not LLM-as-judge — a score of 1.000 requires verbatim reproduction.

| Model | Mode | Score | Input Tokens | Cost |
|-------|------|-------|-------------|------|
| Opus 4.6 | Baseline (652K full context) | 1.000 | 617,755 | $3.11 |
| GPT-5.4 | Baseline (652K full context) | 0.475 | 550,443 | $1.39 |
| Gemini 3.1 Pro | Baseline (652K full context) | 0.647 | 565,164 | $0.27 |
| Opus 4.6 | VC | 1.000 | 113,602 | $0.59 |
| Sonnet 4.5 | VC | 1.000 | 62,920 | $0.20 |
| GPT-5.4 | VC | 1.000 | 53,182 | $0.14 |
| Gemini 2.5 Pro | VC | 1.000 | 80,816 | $0.12 |

All four VC readers achieve perfect scores on a question where two of three full-context baselines fail. The VC readers consume 5–12x fewer tokens than baselines, at 5–26x lower cost. Notably, the only baseline that succeeds is Opus 4.6 at $3.11 — the most expensive model — while VC enables perfect retrieval from mid-tier models at $0.12–0.20.

The MRCR results reinforce the context rot finding from Section 6.3: at 652K tokens, full-context baselines degrade below 65% accuracy (GPT-5.4 at 47.5%, Gemini at 64.7%), while VC readers achieve perfect accuracy regardless of the underlying conversation length. The 652K-token MRCR context is comparable to the 660K merged haystack from Section 6.3.2, and the failure pattern is consistent — baselines cannot reliably distinguish structurally similar content at this scale.

### 6.5 The Reader as Active Investigator

The shift from passive context consumption to active tool-augmented retrieval fundamentally changes how the reader model engages with accumulated context. In the full-context baseline, the reader must perform a single forward pass over 118K tokens, hoping that attention mechanisms will surface the relevant passage. In VC, the reader operates more like a researcher with access to a library: it formulates hypotheses, queries specific sources, evaluates results, and iterates.

This manifests in several observable ways:

**Query formulation.** The reader constructs search queries that reflect understanding of what it needs. For temporal questions, it searches for date-anchored phrases ("May 2023 trip"). For counting questions, it searches for category terms ("model kit," "bike," "aquarium fish"). For knowledge-update questions, it uses `query_facts` with status filters to find the latest value. The reader does not simply echo the user's question as a search query — it reformulates based on what information would resolve the question.

**Selective tool use.** Despite having 5 tools available, the reader consistently selects the most appropriate tool for each sub-task. 82% of questions are resolved in 1--2 calls, suggesting the reader develops accurate intuitions about which tool will be most productive. The near-zero usage of `recall_all` (1 case) and `collapse_tags` (0 cases) indicates the reader correctly assesses that targeted retrieval is almost always more efficient than broad loading — a judgment that was not pre-programmed.

**Knowing when to stop.** The tool loop allows up to 10 rounds, but the reader typically stops after finding sufficient evidence. The anti-repetition mechanism (which suppresses already-shown segments) helps, but the reader also exhibits independent stopping behavior — it synthesizes an answer once it has corroborating evidence from 1--2 sources rather than exhaustively searching all available topics.

### 6.6 Prefetch as a Cognitive Scaffold

Before any tool call, the reader's system prompt contains a multi-layered scaffold assembled from the VC store. First, a **tag vocabulary** listing all stored topics with token costs, fact counts, and depth levels — giving the reader a navigational map of the entire conversation history. Second, **tag summaries** for topics in the working set, loaded at depths determined by the paging manager (summary, segment-level, or full text). Third, **embedding-matched facts** relevant to the inbound question, surfaced by comparing the question against stored fact embeddings. Together, these layers mean the reader starts with compressed narrative context organized by topic, structured facts with dates and statuses, and a complete index of what else is available to drill into via tools.

This prefetch mechanism serves a role analogous to "priming" in cognitive psychology. The reader does not start from zero; it begins with a structured scaffold of what is known about the user, organized by topic, with key facts already surfaced.

Our three-tier analysis (Section 5.5) reveals that this scaffold is sufficient to directly answer approximately 30% of questions without any tool calls — not just from prefetched facts, but from the compressed summaries that may already contain the answer in narrative form. In the remaining 70%, the prefetched layers still serve a critical function: facts orient the reader's search strategy, while tag summaries provide narrative context that facts alone cannot. A reader that already sees a fact like `user | re-watching | Marvel movies (Iron Man, Thor, Captain America, Avengers)` knows to search for confirmation rather than discovery — a fundamentally different and more efficient retrieval strategy.

This has implications for comparing VC against pure KB retrieval systems, which also prefetch facts at query time but lack the narrative context that VC's compressed summaries provide. When a fact alone is ambiguous — for example, when `user | had | an amazing time at Red Rocks` could refer to multiple events — VC's reader can demand-page the full conversation text to disambiguate. KB retrieval systems must rely solely on the extracted fact, with no fallback to source material.

| Capability | VC | KB Retrieval Systems | Full Context |
|------------|-----|---------------------|-------------|
| Narrative context prefetch | Compressed summaries by topic | None (facts only) | Raw (118K tokens) |
| Structured fact prefetch | Yes (embedding-matched) | Yes (query-matched) | No |
| Disambiguation fallback | Demand-page original text | None | Must scan 118K tokens |
| Knowledge updates | Supersession chains | Varies | No mechanism |
| Temporal resolution | Session dates + relative date resolution | Varies | Raw dates in text |
| Query-time flexibility | 5 retrieval tools | Fixed pipeline | No tools |

The prefetch layer thus represents a hybrid architecture: it provides the speed of fact-based lookup with the depth guarantee of full-text access. This combination is what enables VC to outperform both pure KB systems (which cannot disambiguate) and full-context systems (which cannot focus attention).

### 6.7 Parallels to Human Memory: Toward Biologically Plausible AI Memory

VC's architecture bears striking parallels to human episodic memory, and examining these parallels illuminates both the system's strengths and the broader path toward artificial general intelligence.

**Compression and gist extraction.** Humans do not store conversations verbatim. Cognitive psychology has long established that episodic memories undergo *consolidation* — a process that extracts semantic gist while discarding surface details (Bartlett, 1932; Schacter, 2001). VC's compaction pipeline performs an analogous operation: raw conversation turns are compressed to 15% of their original size, preserving key facts, decisions, and emotional valence while discarding verbatim phrasing. The three-layer hierarchy (raw → segments → tag summaries) mirrors the progression from sensory memory to short-term memory to long-term semantic memory in human cognition.

**Cue-dependent retrieval.** Human memory is not accessed by sequential scanning — it is *cue-dependent*. A question about "that restaurant we went to in March" triggers associative retrieval through multiple cues: the temporal marker ("March"), the activity ("restaurant"), and the social context ("we"). VC's tool suite provides an analogous multi-cue retrieval system: `find_quote` provides semantic/associative cues, `query_facts` provides structured attribute cues, `remember_when` provides temporal cues, and `expand_topic` provides contextual reconstruction. The reader's emergent multi-tool chaining patterns (Section 6.2) resemble the human process of *retrieval-induced facilitation* — where partially successful recall provides cues that trigger additional memories.

**Interference and supersession.** Human memory is susceptible to *retroactive interference* — new information can overwrite or obscure old information, leading to systematic errors in recall. The full-context baseline exhibits exactly this failure mode in reverse: *proactive interference*, where older, more frequently rehearsed information overwhelms newer corrections. VC's supersession chain mechanism provides an explicit solution to both forms of interference: new facts are linked to the old facts they supersede, ensuring that retrieval always surfaces the latest value. This is more robust than human memory, where knowledge updates often fail — people frequently recall their old phone number or previous address even after years at a new one.

**The "tip of the tongue" phenomenon.** When humans experience a tip-of-the-tongue state — knowing that they know something but being unable to retrieve it — they employ *metacognitive* strategies: trying different retrieval cues, approaching the memory from different angles, using contextual reconstruction. The reader's strategy pivots under failure (Section 6.2) are a computational analog: when `find_quote` fails to surface the answer, the reader tries `query_facts` or `remember_when`, approaching the same memory from a different retrieval pathway. The 7 observed strategy pivots in our evaluation suggest that tool-augmented LLMs can develop rudimentary metacognitive retrieval strategies.

**Working memory and attention.** Human working memory has a famously limited capacity — roughly 4±1 chunks (Cowan, 2001). The token budget in VC serves an analogous function: it forces the system to be selective about what information is actively maintained, evicting less relevant content (LRU eviction) to make room for more relevant content (demand paging). The 30,000-token tag context budget is not merely a technical constraint — it is a *design feature* that prevents the attention degradation observed when models are given unlimited context.

These parallels suggest that VC's architecture is not merely an engineering solution but converges on principles that biological memory systems discovered through evolution. This convergence is not coincidental: both systems face the same fundamental constraint — limited processing bandwidth (attention/working memory) combined with vast stored experience — and arrive at similar solutions: hierarchical compression, cue-dependent retrieval, active reconstruction over passive replay, and explicit mechanisms for knowledge updating.

For the path toward AGI, this suggests that memory systems for artificial agents should not aim for perfect verbatim recall (which even the 118K-token full-context baseline demonstrates is counterproductive) but should instead aim for *organized, retrievable, updatable* memory — the same properties that make human episodic memory functional despite its well-documented imperfections.

### 6.8 The OS Analogy: How Deep Does It Go?

The virtual memory analogy is not merely metaphorical — it is a principled design framework that informed concrete implementation decisions:

| OS Concept | VC Implementation |
|------------|-------------------|
| Page table | Tag index (TurnTagIndex) |
| Pages | Segments (StoredSegment) |
| Compressed pages | Segment summaries |
| Working set descriptors | Tag summaries (TagSummary) |
| Demand paging | `expand_topic` tool |
| Page eviction | `collapse_topic` / auto-evict (LRU) |
| TLB (fast lookup) | Embedding-based inbound tagger |
| Full page walk | LLM-based write-path tagger |
| Soft page fault | Soft compaction threshold (70%) |
| Hard page fault | Hard compaction threshold (85%) |
| Memory pressure | Token budget management |
| Working set size | `tag_context_max_tokens` (30K) |
| Page replacement policy | LRU eviction by `last_accessed_turn` |

The implementation literally tracks working sets (`PagingManager.working_set`), implements LRU eviction (`_auto_evict` sorted by access recency), enforces memory pressure thresholds (70%/85% utilization triggers), and manages page tables (tag-to-segment mappings with depth levels).

**The shared interpretive layer in practice.** As noted in Section 1.2, the OS mechanisms above are viable because both the write path and read path converge on the same abstraction layer. Our results show this convergence working in practice. The embedding-based inbound tagger maps queries into the same tag vocabulary that the write-path tagger produced — functioning as a TLB with sub-millisecond latency and no LLM call. `query_facts` navigates the same subject/verb/object dimensions that fact extraction populated, with semantic verb expansion bridging vocabulary gaps ("led" → ["led", "leads", "managed", "directed"]). Tag canonicalization and alias consolidation ensure that the vocabulary converges rather than fragments over time, keeping the shared address space navigable as the store grows. The OS mechanisms in the table above — LRU eviction, working set tracking, demand paging — all operate on this shared layer, not on raw interpretive judgments. When the shared layer's vocabulary fails to capture a detail — as in the cross-topic discovery case of b3c15d39, where the answer was stored under an unrelated tag — `find_quote` bypasses the tag address space entirely to search the preserved original text, ensuring the interpretive layer does not become a ceiling on recall.

### 6.9 Limitations

**Sample size.** Our evaluation covers 100 of 500 LongMemEval questions. While results are statistically significant (binomial test p < 0.001 for the accuracy differential), evaluation on the full 500 questions would strengthen claims.

**Reader model.** We evaluated with a single mid-tier reader model (Claude Sonnet 4.5) for both VC and baseline (see Section 4.2 for rationale). While the model-tier argument (Section 5.5) suggests VC should be *more* advantageous with cheaper models, the interaction between compression quality and reader capability deserves empirical study across model families — including whether flagship models with VC achieve even higher accuracy, or whether the mid-tier ceiling is already near-optimal.

**Ingestion cost.** Per-turn ingestion cost is negligible ($0.0008/turn using MiMo-V2-Flash) and the investment pays for itself after the context exceeds 65K tokens — just 13 minutes of an agentic coding session. However, the investment is per-conversation: each independent conversation history requires its own ingestion. For short, single-use conversations that never exceed the crossover point, VC's ingestion adds cost without benefit.

**Ingestion model dependence.** Compression and fact extraction quality depend critically on the ingestion LLM (MiMo-V2-Flash in our setup). A weaker ingestion model would produce lower-quality summaries and facts, potentially degrading downstream accuracy.

**Synthetic conversations.** LongMemEval uses synthetic multi-session conversations. Real conversations may exhibit different patterns — more topic mixing, less structured information, more ambiguity.

**Latency at small scale.** At LongMemEval's standard ~125K token haystack, VC incurs a ~4s marginal overhead from additional API round-trips in the tool loop (Section 5.4). This overhead is constant and becomes a net advantage beyond ~200–300K tokens (Section 5.4.1), but for short conversations it remains a real cost. In production proxy deployments, streaming the reader's intermediate reasoning to the user can partially mask this overhead.

---

## 7. Conclusion

We have presented Virtual Context, a system that applies OS virtual memory principles to LLM context management. By organizing accumulated context into a three-layer hierarchy with bidirectional paging, extracting structured facts with knowledge-update tracking, and providing the reader with five retrieval tools for active exploration, VC achieves 95% accuracy on LongMemEval compared to 33% for a full-context baseline — while using 55% fewer tokens at 55% lower cost. At standard LongMemEval scale (~125K tokens), VC adds a marginal ~4s of latency from tool loop round-trips; at real-world conversation scales (660K–875K tokens), this inverts to a 2–8x speed advantage as baseline latency scales linearly while VC's remains constant.

The central insight is that **the context window is not a bucket to fill but a managed resource to optimize.** Raw full context is worse than curated compressed context because attention is finite — a model seeing everything relevant intermixed with everything irrelevant performs far worse than a model seeing organized, compressed content with tools to drill into details on demand.

A corollary insight is that **context management is a higher-leverage investment than model capability** for tasks requiring recall over accumulated context. VC enables a mid-tier model (Claude Sonnet 4.5) to achieve 95% accuracy at $0.16/question — a level that even a flagship model with raw full context cannot match — as demonstrated empirically in Section 6.3 where Opus 4.6 at $4.42/query fails on questions that VC + Sonnet answers correctly at $0.20/query. This inverts the conventional assumption that harder tasks demand bigger models: structured context organization allows cheaper models to punch above their weight class, compounding token savings with per-token pricing advantages.

A third finding, validated through merged haystack stress tests at 660K–926K tokens (Section 6.3), is that **baseline accuracy actively degrades at scale while VC's structured retrieval provides resilience**. Frontier models (Gemini 3 Pro, GPT-5.4, Sonnet 4.5, Opus 4.6) that answer correctly at 125K tokens fail on the same questions when embedded in larger conversation histories — exhibiting context rot through miscounting, temporal ordering errors, and outright hallucination. At 660K tokens, VC readers answer correctly where both baselines fail. At ~926K tokens (8 merged conversation streams), all three frontier baselines fail (0/3) while all three VC-assisted readers succeed (3/3), demonstrating that structured context management is not merely an optimization but a necessity for production conversation memory at realistic scale.

Five contributions emerge from this work:

1. The three-layer memory hierarchy with greedy set cover compression and demand paging demonstrates that hierarchical organization outperforms flat context presentation.
2. The two-pass fact extraction pipeline with supersession chains enables reliable knowledge-update tracking — a capability that raw context fundamentally cannot provide.
3. The tool-augmented reader architecture produces emergent retrieval strategies (8 patterns observed) without explicit programming, suggesting that structured context + retrieval tools is a general framework for long-context LLM operation.
4. The economic analysis demonstrates that structured context management compounds with model-tier selection to deliver order-of-magnitude cost reductions at higher accuracy — a finding with immediate production deployment implications.
5. The merged haystack stress test (Section 6.3) empirically validates the context scaling hypothesis: frontier model baselines that succeed at 125K tokens degrade at 660K–926K tokens. At 660K tokens, VC remains accurate where baselines fail. At ~926K tokens, all four frontier baselines fail (0/4) — including Opus 4.6 at $4.42/query — while all four VC readers succeed (4/4) at $0.15–0.76/query, a complete inversion that demonstrates active context management is structurally necessary at realistic conversation scales.

### Future Work

- **Reader model ablation:** Testing with cheaper models (e.g., Haiku-class) and flagship models (Opus-class) to quantify how VC's accuracy gains interact with model capability — and to validate the hypothesis that VC narrows the gap between model tiers
- **Multi-agent frameworks:** Extending VC to provide shared memory across multiple cooperating agents
- **Cross-context reasoning:** Enabling queries that span multiple independent interaction histories
- **Adaptive compression:** Dynamically adjusting compression ratios based on information density and query patterns
- **Larger-scale evaluation:** Full 500-question LongMemEval evaluation and cross-benchmark validation on LoCoMo
- **Context scaling beyond 1M tokens:** Section 6.3 validates the context scaling hypothesis at 660K–926K tokens — baseline accuracy degrades while VC achieves 100% across all reader models at both scales. Extending this analysis to 1M+ tokens with single-user interaction histories (rather than merged haystacks) would characterize the degradation curve under realistic conditions and determine whether VC's fact store quality scales gracefully with organic context growth
- **Latency optimization:** While VC already achieves a net latency advantage beyond ~300K tokens, investigating parallel tool execution, speculative prefetching, and streaming strategies could further reduce the ~4s marginal overhead at smaller conversation scales

---

## References

Fan, W., Ding, Y., Ning, L., Wang, S., Li, H., Yin, D., Chua, T.-S., & Li, Q. (2024). A survey on RAG meeting LLMs: Towards retrieval-augmented large language models. *Proceedings of KDD 2024*. arXiv:2405.06211.

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Wang, H. (2024). Retrieval-augmented generation for large language models: A survey. arXiv:2312.10997.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Proceedings of NeurIPS 2020*. arXiv:2005.11401.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the middle: How language models use long contexts. *Transactions of the ACL, 12*, 157--173. arXiv:2307.03172.

Maharana, A., Lee, D. H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). Evaluating very long-term conversational memory of LLM agents. *Proceedings of ACL 2024*. arXiv:2402.17753.

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. arXiv:2310.08560.

Wu, X., Li, C., Yin, G., & Wang, W. Y. (2024). LongMemEval: Benchmarking chat assistants on long-term interactive memory. *Proceedings of ICLR 2025*. arXiv:2410.10813.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. *Proceedings of ICLR 2023*. arXiv:2210.03629.

Yen, H., Gao, T., & Chen, D. (2024). Found in the middle: Permutation self-consistency improves listwise ranking in large language models. arXiv:2403.04797.

---

## Appendix A: Tool Definitions

### A.1 vc_find_quote

```json
{
  "name": "vc_find_quote",
  "description": "Search for specific quotes, phrases, or content across all stored conversation text using full-text and semantic search.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "The search query — keywords, phrases, or natural language description of what to find"}
    },
    "required": ["query"]
  }
}
```

### A.2 vc_query_facts

```json
{
  "name": "vc_query_facts",
  "description": "Query the structured fact store for specific information. Supports filtering by subject, verb, object content, temporal status, and fact type.",
  "input_schema": {
    "type": "object",
    "properties": {
      "subject": {"type": "string"},
      "verb": {"type": "string", "description": "Action verb — automatically expanded to semantically similar verbs"},
      "object_contains": {"type": "string", "description": "Keyword match on the object field"},
      "status": {"type": "string", "enum": ["active", "completed", "planned", "abandoned", "recurring"]},
      "fact_type": {"type": "string", "enum": ["personal", "experience", "world"]}
    }
  }
}
```

### A.3 vc_expand_topic

```json
{
  "name": "vc_expand_topic",
  "description": "Load original conversation text for a topic tag. Supports SEGMENTS (summaries, ~2K tokens) or FULL (original text, ~8K+ tokens) depth.",
  "input_schema": {
    "type": "object",
    "properties": {
      "tag": {"type": "string"},
      "depth": {"type": "string", "enum": ["segments", "full"], "default": "segments"},
      "collapse_tags": {"type": "array", "items": {"type": "string"}, "description": "Optional: collapse these tags first to free budget"}
    },
    "required": ["tag"]
  }
}
```

### A.4 vc_remember_when

```json
{
  "name": "vc_remember_when",
  "description": "Search for content within a specific time window.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "time_range": {
        "oneOf": [
          {"type": "object", "properties": {"kind": {"const": "relative"}, "preset": {"type": "string"}}},
          {"type": "object", "properties": {"kind": {"const": "between_dates"}, "start": {"type": "string"}, "end": {"type": "string"}}}
        ]
      }
    },
    "required": ["query", "time_range"]
  }
}
```

### A.5 vc_recall_all

```json
{
  "name": "vc_recall_all",
  "description": "Load all tag summaries for a bird's-eye view of all conversation topics. Budget-bounded to 30,000 tokens.",
  "input_schema": {"type": "object", "properties": {}}
}
```

---

## Appendix B: Context Hint Example

The following is an abbreviated example of the context hint injected into the reader's system prompt after compaction:

```
<context-topics budget="64000" used="18656" available="45344">
RULE: These are compressed summaries — Summaries DO omit details.
To find detailed information you have the following tools:
- vc_find_quote(query): search raw text across ALL topics.
- vc_query_facts(subject?, verb?, status?, object_contains?): structured fact lookup.
- vc_expand_topic(tag, collapse_tags?): load original text for a topic.
- vc_remember_when(query, time_range): time-scoped recall.
- vc_recall_all(): load every summary at once.
You have a maximum of 10 tool rounds. Plan your strategy upfront.
Never answer without searching first.

[in context — expand for full detail]
  5k-run: summary 412t → 5899t full, 4 facts — Running progress and 5K charity race
  kitchen-appliances: summary 389t → 4200t full, 6 facts — Coffee maker and stand mixer

[available] travel-planning(8200t), basketball-tournament(3100t), antique-research(4796t), ...

[all 85 topics] 5k-run, kitchen-appliances, travel-planning, basketball-tournament, ...
Scan before answering — relevant context may be under an unexpected topic name.

Tools: find_quote(query) | query_facts(subject?, verb?, status?, object_contains?) |
recall_all() | remember_when(query, time_range) | expand_topic(tag, depth?, collapse_tags?)
</context-topics>
```

---

## Appendix C: Detailed Per-Category Results

| Category | Count | VC Correct | VC % | BL Correct | BL % | Delta |
|----------|-------|------------|------|------------|------|-------|
| knowledge-update | 17 | 17 | 100.0 | 5 | 29.4 | +70.6 |
| multi-session | 26 | 23 | 88.5 | 4 | 15.4 | +73.1 |
| temporal-reasoning | 28 | 26 | 92.9 | 9 | 32.1 | +60.8 |
| single-session-user | 13 | 13 | 100.0 | 6 | 46.2 | +53.8 |
| single-session-assistant | 11 | 11 | 100.0 | 8 | 72.7 | +27.3 |
| single-session-preference | 5 | 5 | 100.0 | 1 | 20.0 | +80.0 |
| **Total** | **100** | **95** | **95.0** | **33** | **33.0** | **+62.0** |
