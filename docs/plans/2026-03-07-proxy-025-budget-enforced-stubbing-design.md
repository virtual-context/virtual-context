# PROXY-025: Budget-Enforced Message Stubbing

**Date:** 2026-03-07
**Bug:** PROXY-025 (logged in `memory/bugs.md`)
**Status:** Design approved

## Problem

`context_window` is not enforced as a hard budget on outbound proxy payloads. In tool-heavy conversations (Claude Code, OpenClaw), the filter's tool chain referential integrity logic force-keeps compacted messages, and a turn-counting mismatch between VC's internal history and client payloads means the compaction watermark maps to the wrong message indices.

**Evidence (request log `000005`, 2026-03-07):**
- `context_window: 5000` configured
- Client sent 117 messages (~40k message tokens)
- Client system prompt: 19k tokens, tools: 7.5k tokens (immovable)
- VC compacted 30 internal turns (`compacted_through=61`)
- Filter only dropped 18 pairs (expected 30) due to turn-count mismatch
- Outbound payload: ~28k message tokens — 5.6x over budget
- Zero tool chains cross the watermark — referential integrity was a red herring; the real issue is turn mapping

## Design

### 1. Budget auto-promotion

Before enrichment, measure immovable overhead per request:

```
overhead = system_prompt_tokens + tools_tokens
```

If `overhead >= context_window`:
- Set `effective_budget = overhead + 10_000`
- Emit dashboard alert: `{ type: "budget_auto_promoted", original: context_window, promoted: effective_budget, overhead }`
- Log CLI warning: `[BUDGET] Client overhead ({overhead}t) exceeds context_window ({cw}t). Auto-promoted to {effective_budget}t.`

Otherwise: `effective_budget = context_window`.

Runs once per request (overhead changes as tool count or system prompt changes).

### 2. Hash-based compacted turn identification

Replace the current pair-index watermark mapping with content hash matching.

**During `on_turn_complete` (already exists):** VC stores `message_hash` in TurnTagIndex entries. The hash is computed from the stripped user+assistant text.

**During `_filter_body_messages` (new):**
- For each user message in the client payload that contains text content:
  - Strip OpenClaw envelope (same `_strip_openclaw_envelope` used during ingestion)
  - Extract text content (same `_extract_message_text` logic)
  - Compute SHA-256 hash of the stripped text
  - Look up hash in TurnTagIndex entries
  - If the entry exists AND its turn number < `_compacted_through // 2`: mark this message (and its surrounding tool chain messages) as compacted

This decouples the filter from pair-index arithmetic. The hash is the source of truth.

**Edge case — hash miss:** If a client message doesn't match any hash (e.g., the client modified the message, or VC ingested a different version), treat it as uncompacted. Safe default — you keep the message.

### 3. Stub replacement

For each message group identified as a compacted turn, replace ALL content blocks with a single text stub:

```json
{"role": "assistant", "content": [
  {"type": "text", "text": "[Compacted turn 12: topics=pregnancy-test, clarification. Content stored in virtual-context.]"}
]}
```

For user messages in the same group:
```json
{"role": "user", "content": [
  {"type": "text", "text": "[Compacted turn 12]"}
]}
```

Properties:
- No `tool_use` blocks survive → no `tool_use_id` referential integrity to maintain
- Role alternation preserved (messages still exist, just with small content)
- ~30 tokens per stubbed message vs hundreds-to-thousands for originals
- Tags sourced from the TurnTagIndex entry for that turn
- Thinking blocks eliminated (replaced by stub)

**Message grouping:** A "turn" in the client payload may span multiple messages:
```
user(text) → assistant(thinking, text, tool_use) → user(tool_result) → assistant(text, tool_use) → user(tool_result) → assistant(text)
```
All messages between one user-text message and the next user-text message belong to the same turn and are stubbed together. The stub collapses the group to:
```
user(stub) → assistant(stub)
```
Intermediate messages (tool_use/tool_result chains) are removed entirely since both ends are stubbed. Role alternation is maintained by the outer user/assistant pair.

### 4. Over-budget alert (post-filter)

After stubbing, estimate total outbound tokens:
```
total = overhead + stubbed_tokens + remaining_message_tokens + vc_injection_tokens
```

If `total > effective_budget`:
- Emit dashboard event: `{ type: "budget_exceeded", total, budget, excess: total - budget, reason: "N uncompacted turns (Xk tokens)" }`
- Log: `[BUDGET] Payload {total}t exceeds budget {budget}t by {excess}t. {N} uncompacted turns pending compaction.`
- Do NOT drop or stub uncompacted messages — their content isn't in VC summaries yet
- Background `on_turn_complete` will compact them; next request they get stubbed
- Budget converges naturally over 1-3 turns

### 5. Unchanged components

- `on_turn_complete`: compaction logic, thresholds, segmenter — no changes
- Assembler: VC context injection budget — no changes
- Tool interception/continuation loop — no changes
- Tag-based filtering for uncompacted turns — continues as before
- Protected recent turns — still exempt from stubbing and filtering
- No new VC tools needed — `vc_expand_topic` and `vc_find_quote` already retrieve compacted content

## Expected data flow

```
Request arrives (117 msgs, ~40k msg tokens, 26.5k overhead)
  │
  ├─ Budget: 5k configured, overhead=26.5k → auto-promote to 36.5k (alert!)
  │
  ├─ Hash each user-text message against TurnTagIndex
  │   └─ 13 of 24 text turns match compacted entries
  │
  ├─ Stub 13 compacted turns
  │   ├─ ~40 messages collapsed to ~26 stub messages
  │   └─ ~15k tokens → ~800 tokens
  │
  ├─ Tag-based filter: drop irrelevant uncompacted turns (existing logic)
  │
  ├─ Estimate total: 26.5k + 0.8k + 12k + 5k ≈ 44.3k
  │   └─ Over 36.5k budget → alert, send anyway
  │
  └─ Background: on_turn_complete compacts more turns
      └─ Next request: newly compacted turns get stubbed → budget converges
```

## Files to modify

| File | Change |
|------|--------|
| `proxy/server.py` | Budget auto-promotion logic before enrichment; over-budget alert after filter |
| `proxy/message_filter.py` | Hash-based identification + stub replacement (replaces force-keep logic for compacted turns) |
| `proxy/metrics.py` | New event types: `budget_auto_promoted`, `budget_exceeded` |
| `proxy/dashboard.html` | Display budget alerts in dashboard |
| `core/turn_tag_index.py` | Expose hash→turn lookup method |
| `tests/test_message_filter.py` | New tests for stub replacement, hash matching, budget alerts |
