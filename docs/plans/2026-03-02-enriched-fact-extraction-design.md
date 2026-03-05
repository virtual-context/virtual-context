# Enriched Fact Extraction Design

**Date:** 2026-03-02
**Status:** Approved
**Goal:** Improve fact/memory extraction quality from ~75% LongMemEval to Hindsight-tier (~91%) by enriching extraction prompts, adding fact types, wiring supersession into the conversation pipeline, and adding post-compaction dedup.

## Problem

VC's fact extraction produces noisy, sparse facts. Analysis of 429 facts from the benchmark cache (6a1eabeb):

- **14% assistant action noise** — 62 facts like "assistant recommends Corsair K68" that aren't user memories
- **12% meta-conversational verbs** — "user asks about Cairo restaurants" instead of "user wants to try authentic Egyptian food in Cairo"
- **Sparse dimensions** — who: 33%, when: 22%, where: 22%, why: 39% populated
- **Zero supersessions** — supersession checker only exists in document ingestion path, never runs in conversation pipeline. Exact duplicates coexist (e.g. 4 facts about "5K PB of 27:12", 3 about "gym three times a week")
- **Lost specifics** — numbers, names, amounts get abstracted away ("personal best time" instead of "personal best time of 27:12")

## Validated Approach

POC tested across 5 models (Haiku 4.5, GPT-4o-mini, GPT-4.1-nano, Gemini 3 Flash, Qwen3-30B-A3B). Key findings:

1. **Enriched prompts** preserve specifics across all models (v1 POC)
2. **Supersession detection** works when prior facts are injected as context — passes on Haiku, Gemini Flash, Qwen3; fails on GPT-4o-mini and 4.1-nano with 40-fact haystack (v2 POC)
3. **Combined dedup+supersession in a single prompt is unreliable** — models trade one task for the other (v3 POC)
4. **Separated prompts solve this completely** — Qwen3-30B passes both tasks when they're separate prompts. Total cost: ~$0.0002 per compaction (v4 POC)

**Default model recommendation:** Qwen3-30B-A3B via OpenRouter ($0.06/M input, $0.22/M output). Cheapest model that passes all tests. Configurable per deployment.

## Architecture

```
Conversation turn
    │
    ├─ Tagger (existing LLM call) ──→ Tags + enriched FactSignals
    │                                  (fact_type, what, specifics preserved)
    │
    ├─ Compactor (existing LLM call) ──→ Summary + consolidated Facts
    │                                    (all 5 dimensions, within-segment dedup)
    │                                    Facts stored to SQLite
    │
    ├─ Post-compaction Pass 1 (NEW) ──→ Supersession check
    │   Input: new facts + pre-filtered existing facts (same subject)
    │   Output: superseded fact IDs
    │   Model: configurable (default Qwen3-30B)
    │
    └─ Post-compaction Pass 2 (NEW, periodic) ──→ Dedup scan
        Input: all facts for a subject
        Output: duplicate clusters (keep richest, remove others)
        Model: configurable (default Qwen3-30B)
        Frequency: after every N compactions or on-demand
```

## Changes

### 1. FactSignal enrichment (types.py)

Add `fact_type` and `what` to FactSignal:

```python
@dataclass
class FactSignal:
    subject: str = ""
    verb: str = ""
    object: str = ""
    status: str = ""
    fact_type: str = "personal"  # NEW: personal|experience|world
    what: str = ""               # NEW: full-sentence memory
```

### 2. Fact type field (types.py + sqlite.py)

Add `fact_type` to the Fact dataclass:

```python
fact_type: str = "personal"  # personal|experience|world
```

SQLite migration: `ALTER TABLE facts ADD COLUMN fact_type TEXT NOT NULL DEFAULT 'personal'`

Fact types:
- `personal` — about the user's life, identity, preferences, plans, experiences
- `experience` — assistant-provided info the user engaged with (lower priority for personal recall)
- `world` — facts about entities in the user's world (other people, places, things)

### 3. Enriched tagger prompt (tag_generator.py)

Replace fact extraction instructions (lines 89-92 of TAG_GENERATOR_PROMPT_DETAILED, lines 112-113 of TAG_GENERATOR_PROMPT_COMPACT) with:

- Extract the fact behind the question, not the conversational act
- Preserve ALL specifics (numbers, names, dates, amounts)
- Classify fact_type (personal/experience/world)
- Coreference resolution instruction
- DO NOT extract mere asks/mentions/discusses
- Include what field in the JSON schema

### 4. Enriched compactor prompt (compactor.py)

Update fact section of DEFAULT_SUMMARY_PROMPT (lines 110-121):

- Require all 5 dimensions (what/who/when/where/why) — populate when present, not "omit if n/a"
- Within-segment dedup instruction: "if two signals describe the same event, emit one fact with the richest details"
- fact_type classification
- Preserve ALL specifics emphasis

### 5. Wire supersession into engine.py

After `compact_manual()` stores facts, run the supersession checker. This is the critical missing piece — currently supersession only exists in `ingest/ingestor.py`.

Location: `engine.py`, inside the compaction result storage loop (around line ~300).

```python
# After storing compaction results with facts...
new_facts = [f for r in results for f in r.facts]
if new_facts and self._supersession_checker:
    self._supersession_checker.check_and_supersede(new_facts)
```

### 6. Expand supersession checker (ingest/supersession.py)

Update the LLM prompt from "which are CONTRADICTED?" to "which are CONTRADICTED or DUPLICATED by the new fact?". When duplicates are found, keep the richer version.

This reuses the existing infrastructure (query by subject, LLM comparison, mark superseded).

### 7. Add periodic dedup pass

New method on engine: `deduplicate_facts(subject="user")`.

- Queries all non-superseded facts for a subject
- Sends to LLM with dedup-only prompt (validated in v4 POC Pass 2)
- Marks duplicates as superseded, keeping the richest version
- Can be called manually via CLI, or triggered after every N compactions

### 8. Update vc_query_facts (tool_loop.py)

- Add `fact_type` to the response payload (already returns what/who/when/where/why)
- Add optional `fact_type` filter parameter to the tool schema
- Default query excludes `experience` type unless explicitly requested

### 9. Make supersession/dedup model configurable

Add to VirtualContextConfig or a new FactExtractionConfig:

```yaml
fact_extraction:
  supersession:
    enabled: true
    provider: "openrouter"
    model: "qwen/qwen3-30b-a3b"
    batch_size: 25
  dedup:
    enabled: true
    provider: "openrouter"
    model: "qwen/qwen3-30b-a3b"
    interval: 5  # run after every N compactions
```

### 10. Update tagger FactSignal parsing (tag_generator.py)

Update `_parse_response()` (line 396-407) to read fact_type and what from the LLM response:

```python
fact_signals.append(FactSignal(
    subject=f.get("subject", ""),
    verb=f.get("verb", f.get("role", "")),
    object=f.get("object", ""),
    status=f.get("status", "active"),
    fact_type=f.get("fact_type", "personal"),  # NEW
    what=f.get("what", ""),                     # NEW
))
```

## What Does NOT Change

- Tag system (tags are separate from facts and work well)
- Compaction/summarization pipeline structure
- Retrieval architecture (embedding-based tag matching)
- Tool loop / reader interaction pattern
- Existing subject/verb/object triple (augmented, not replaced)

## Validation

### Litmus test: question 6a1eabeb (5K PB 27:12 → 25:50)

After implementation, rerun this question. Success criteria:
1. Only 1 active PB fact (25:50), not 4 copies of 27:12
2. The 27:12 facts are marked as superseded
3. The 25:50 fact's `what` field contains the exact time
4. Tag summaries reference the current PB (25:50), not the old one

### Broader validation

- Rerun the 21-question LongMemEval suite
- Compare fact count, noise ratio, dimension coverage before/after
- Target: >85% LongMemEval accuracy (up from ~75%)

## POC Artifacts

- `scripts/poc_fact_extraction.py` — v1: enriched prompt quality across 5 models
- `scripts/poc_fact_extraction_v2.py` — v2: realistic supersession with 40-fact haystack
- `scripts/poc_fact_extraction_v3.py` — v3: combined dedup+supersession (showed limits)
- `scripts/poc_fact_extraction_v4.py` — v4: separated prompts (validated Qwen3 passes all)
- `scripts/poc_fact_results*.json` — raw results from all POC runs
