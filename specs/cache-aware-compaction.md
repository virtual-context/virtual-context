# Deferred Payload Compaction

**Status**: Under Review (v4)  
**Date**: 2026-04-06  
**Tenant**: b2e7014848c3cc47497bee62 (200k context window)

---

## 1. Problem

When the VC proxy compacts a conversation, it mutates the outbound request body: raw turns are replaced with summaries, tool outputs are stubbed/collapsed, and images are replaced with text descriptors. Every mutation changes the byte content of historical messages, breaking the prefix that Anthropic's prompt cache depends on. Cache hit rates drop from ~99% to ~30% immediately after compaction, wasting ~53k cached tokens per turn until the cache warms again.

Anthropic's prompt cache has a ~5-minute TTL. If no request hits the same prefix within that window, the cache expires naturally. The current system mutates the payload immediately after compaction, breaking the prefix mid-cache-lifetime unnecessarily.

## 2. Solution: Two-Phase Payload Compaction

Split every historical-payload mutation into two phases:

| Phase | When | Where | What it does |
|-------|------|-------|-------------|
| **Prepare** | Soft threshold (background) | Compaction pipeline | Summarizes text, archives tool outputs, archives media references, advances `compacted_through` |
| **Flush** | Hard threshold or cache-cold (request-time) | Server request path | Drops raw turns, stubs tool outputs, collapses chains, stubs media, switches to post-flush mode, advances `flushed_through` |

**Core invariant**: While the cache is warm and the payload fits within the hard threshold, the outbound request body is byte-identical to the previous request for all historical content. No operation may rewrite historical messages until the flush gate fires.

---

## 3. State Model

Two watermarks track compaction progress independently:

```
compacted_through   — "archives exist up to here" (set by background compactor)
flushed_through     — "raw content removed up to here" (set by flush gate)
```

**Invariant**: `0 <= flushed_through <= compacted_through` always.

### Lifecycle

```
 Turn 0      Turn 20        Turn 40         Turn 80
  |            |              |               |
  |<-- flushed_through=0     |               |  (no mutations applied)
  |            |              |               |
  |<------- compacted_through=40             |  (summaries + archives ready)
  |            |              |               |
  |            |              |               |
  [--- flush fires at hard threshold ---]
  |            |              |               |
  |<------- flushed_through=40               |  (raw content removed)
```

### State transitions

| Event | `compacted_through` | `flushed_through` | Payload |
|-------|--------------------|--------------------|---------|
| Initial | 0 | 0 | Raw history |
| Prepare runs at soft threshold | N | 0 | Unchanged (cache-stable) |
| More prepares accumulate | M > N | 0 | Unchanged (cache-stable) |
| Flush fires (hard threshold or cache-cold) | M | M | Shrunk: summaries replace raw content |
| Prepare runs again | P > M | M | Unchanged until next flush |

### What `compacted_through` means now

`compacted_through` represents the message index up to which ALL archival artifacts exist:
- Text summaries/segments are stored for these turns
- Tool outputs in this range have been archived to the store (or will be archived idempotently at flush time by the existing helpers)
- Media positions in this range are known
- It is safe to flush any content up to this boundary

`flushed_through` represents the message index up to which raw content has actually been removed from the outbound payload and replaced with compacted representations.

---

## 4. Configuration

### MonitorConfig (types.py)

```python
@dataclass
class MonitorConfig:
    # Existing fields — defaults UNCHANGED for backward compatibility
    soft_threshold: float = 0.70
    hard_threshold: float = 0.85
    # ... other existing fields ...

    # NEW
    defer_payload_mutation: bool = False   # when True, enable two-phase model
    flush_ttl_seconds: int = 300           # Anthropic cache TTL (seconds)
```

### Backward compatibility

When `defer_payload_mutation` is **False** (default):
- All behavior is identical to today
- `soft_threshold` / `hard_threshold` keep their existing defaults (0.70 / 0.85)
- `flushed_through` tracks `compacted_through` automatically (set to `compacted_through` on every request)
- All existing code paths run unchanged
- `flush_ttl_seconds` is ignored

When `defer_payload_mutation` is **True**:
- The two-phase model activates
- Recommended thresholds: `soft_threshold=0.30`, `hard_threshold=0.97` (set per-tenant via config override, NOT changed as global defaults)
- Payload mutations are deferred until the flush gate fires
- `flush_ttl_seconds` controls the cache-cold detection

This means enabling deferred mode requires setting THREE config values:
```json
{
  "defer_payload_mutation": true,
  "soft_threshold": 0.30,
  "hard_threshold": 0.97
}
```

Operators who enable deferred mode without adjusting thresholds get: compaction at 70%, flush at 85%. This works but provides a narrow deferral window. No silent behavior change for anyone.

### EngineState (types.py)

```python
@dataclass
class EngineState:
    # Existing fields unchanged
    compacted_through: int = 0        # semantics broadened: "all archives ready up to here"
    # ...

    # NEW
    flushed_through: int = 0          # "raw content removed from payload up to here"
    last_request_time: float = 0.0    # time.time() of last COMPLETED upstream call
```

**Invariant enforcement**: On every update to `flushed_through`, assert `flushed_through <= compacted_through`. If violated, log error and clamp.

---

## 5. Phase 1: Prepare (Background)

Runs at the soft threshold via the existing compaction pipeline. The compactor already produces text summaries; the broadened prepare phase additionally ensures tool outputs and media in the flushable range are archived.

### What the prepare phase produces

| Artifact | How | Storage |
|----------|-----|---------|
| Text summaries / segments | Existing compaction pipeline (`compaction_pipeline.py`) | Tag summaries in store |
| Tool-output archives | Existing helpers archive tool content to store as a side effect of stubbing. For prepare-only mode, the compaction pipeline should pre-archive tool outputs in the compacted range so flush-time archival is a no-op. **Fallback**: the flush-time helpers (`stub_tool_outputs_by_position`, `collapse_turn_chains`) archive idempotently before replacing, so flush still works correctly even if pre-archival was skipped. |
| Media position metadata | Lightweight: the compacted range boundary (`compacted_through`) is sufficient. `stub_media_by_position()` uses the protected window and turn groups to identify stale media at flush time. No separate media index needed. |
| Flushable range metadata | `compacted_through` itself — everything below this boundary has archives and can be safely flushed |

### What the prepare phase must NOT do

- Must not call `drop_compacted_turns()`
- Must not call `stub_tool_outputs_by_position()`
- Must not call `collapse_turn_chains()`
- Must not call `stub_media_by_position()`
- Must not switch the retrieval assembler to post-compaction mode
- Must not inject paging tools or context hints that reference compacted state
- Must not modify the outbound request body in any way

The prepare phase advances `compacted_through` only. All payload mutations are reserved for the flush gate.

---

## 6. Phase 2: Flush (Request-Time)

The flush gate runs on every inbound request and decides whether to apply payload mutations. It is the **single decision point** for all historical-content rewrites.

### Flush decision

```python
if not mon.defer_payload_mutation:
    # Legacy mode: flush immediately on every request (today's behavior)
    should_flush = (compacted_through > flushed_through)
    use_two_pass = False

elif cache_age >= mon.flush_ttl_seconds:
    # Cache is cold — flush in pass 1 directly (fast path)
    should_flush = (compacted_through > flushed_through)
    use_two_pass = False

else:
    # Cache is warm — defer, but check size after full assembly
    should_flush = False  # tentative; may be overridden by size check
    use_two_pass = True   # need to measure fully-assembled payload first
```

### Flush operations (atomic batch)

When `should_flush` is True, ALL of the following run as a single batch. The order matters:

```
1. Set flushed_through = compacted_through  (tentative — see §6.3)
2. drop_compacted_turns(body, drop_boundary=compacted_through)
3. collapse_turn_chains(body)               [stage 2 — tool chain collapse]
4. stub_tool_outputs_by_position(body)      [protected-zone intrusion]
5. stub_media_by_position(body)
6. Switch retrieval assembler to post-flush mode
7. Re-inject VC context with post-flush assembled content
8. Inject paging tools (now meaningful)
```

### Two-pass approach (warm-cache size check)

When the cache is warm and `use_two_pass` is True:

```
Pass 1: Build the full outbound payload WITHOUT any historical mutations.
         - flushed_through stays at its current value
         - All payload-affecting code paths use flushed_through (still 0 or previous flush)
         - Assembler runs in pre-flush mode (tag-based retrieval, no hints)
         - No tool/media stubbing, no turn dropping, no chain collapse
         - Context injection, paging, and assembly run normally
         - Measure outbound_tokens on the fully-assembled enriched_body

Size check: outbound_tokens >= context_window * hard_threshold?

Pass 2 (only if size check triggers):
         - Set tentative_flush = compacted_through
         - Re-run retrieval assembler with flushed_through = tentative_flush
           (switches to post-flush mode: all summaries, no tag filter)
         - Apply all flush operations (steps 1–8 above) on body
         - Re-inject context with post-flush assembled content
         - Re-inject paging tools
         - Re-measure outbound_tokens
         - Commit: flushed_through = tentative_flush (only after dispatch)
```

### Tentative flush and commit

Pass 2 uses a local `tentative_flush` variable. `flushed_through` on EngineState is only updated after the flushed payload is chosen and the upstream request is dispatched. If reassembly or dispatch fails, `flushed_through` stays at its old value and the next request retries. This prevents orphaned state where `flushed_through` advances but no flushed payload was ever served.

### Cold-cache fast path

When `cache_age >= flush_ttl_seconds`, there's no prefix to protect. All flush operations run in a single pass with no reassembly. The assembler is told `flushed_through = compacted_through` from the start.

---

## 7. Warm-Cache Prefix Stability

**Contract**: When `defer_payload_mutation` is True and the cache is warm (`cache_age < flush_ttl_seconds`) and the payload fits (`outbound_tokens < hard_threshold * context_window`), the outbound request body is byte-identical to the previous request's body for all historical content. Only new turns (appended at the end) differ.

This requires gating **every** operation that rewrites historical messages:

### Gated operations (ALL require `flushed_through > previous_flushed_through`)

| Operation | Location | What it rewrites |
|-----------|----------|-----------------|
| `drop_compacted_turns()` | `server.py:685` | Removes raw text turns |
| `collapse_turn_chains()` | `server.py:747` | Replaces tool-bearing turn chains with stub pairs |
| `stub_tool_outputs_by_position()` (stage 2, protected intrusion) | `server.py:766` | Replaces tool outputs with restore stubs |
| `stub_tool_outputs_by_position()` (stage 1, pre-compaction) | `server.py:780` | Replaces large tool outputs aging out of protected window |
| `stub_media_by_position()` | `server.py:796` | Replaces image blocks with text descriptors |
| Retrieval assembler `post_compaction` branches | `retrieval_assembler.py:112,141,199,336,419` | Changes context injection text, injects hints |
| Paging tool injection | `server.py:831` | Adds tool definitions to payload |
| VC tool injection | `tool_query.py:111` | Adds VC tools to tool schema |
| `_pre_compaction` flag | `server.py:723` | Selects which stubbing path runs |

**Stage-1 tool stubbing (server.py:780)**: In the current system, this runs on every request regardless of `compacted_through`. When a tool output ages out of the protected window between requests, it gets stubbed, changing a historical message. In deferred mode, this is explicitly gated: stage-1 tool stubbing is **skipped entirely** while the cache is warm. Tool outputs stay raw until flush. This trades payload size for prefix stability — the hard threshold catches the growth.

**Media stubbing (server.py:796)**: Same treatment. `stub_media_by_position()` is skipped while cache is warm. Images stay inline until flush. Conversations with many large images will hit the hard threshold sooner, triggering an earlier flush.

**`drop_topic_only_stubs()` (server.py:808)**: This removes dead-weight stubs (stubs without restore refs). It only operates on stubs, not raw content. In deferred mode with `flushed_through == 0`, there are no stubs to drop, so it's a no-op. No gating needed.

**`merge_consecutive_conversational()` (server.py:813)**: This merges adjacent same-role messages to fix alternation violations. It can change message boundaries. In deferred mode, this should only run on new (unflushed) content, or be skipped if the merge would touch historical messages. **Implementation note**: if the merge only ever touches the last few messages (which is the common case), this is safe. If it can reach into historical messages, gate it.

---

## 8. Timestamp Tracking

`last_request_time` records when the most recent upstream call **completed** (not when the current request started).

### Ordering (critical)

```
1. Read last_request_time from state (set by the PREVIOUS request)
2. Compute cache_age = time.time() - last_request_time
3. Run flush gate (decide whether to flush)
4. Build and send upstream request
5. AFTER upstream returns: state.last_request_time = time.time()
```

### Known approximations

- **Streaming**: The cache is warmed at request acceptance, not stream completion. Recording completion time makes `cache_age` smaller than reality by the stream duration. This biases toward deferral (the safe direction — preserves prefix). If precision matters, record at send time.
- **Model/provider changes**: `last_request_time` is a single scalar. A model change invalidates the cache independently of time. The TTL heuristic may defer a flush when the cache is already cold due to a model switch. Worst case: slightly larger payload until the hard threshold catches it. No correctness issue. Future refinement: scope warmth to `(model, provider, system_hash)` tuple.

---

## 9. Complete `compacted_through` Audit

Every code path that reads `compacted_through` is classified:

### Must use `flushed_through` (payload-affecting paths)

| Location | Current code | Why |
|----------|-------------|-----|
| `retrieval_assembler.py:112` | `if compacted_through > 0: active_tags = []` | Suppresses tag retrieval; only after raw turns are gone |
| `retrieval_assembler.py:141` | `post_compaction=(compacted_through > 0)` | Enables summary-floor injection in retriever |
| `retrieval_assembler.py:199` | `post_compaction=(compacted_through > 0)` | Same (retry path) |
| `retrieval_assembler.py:336` | `watermark = compacted_through` | Turn filtering in `_filter_irrelevant_turns()` |
| `retrieval_assembler.py:419` | `if compacted_through == 0: return ""` | Gates `_build_context_hint()` emission |
| `server.py:685` | `if compacted_through > 0: drop_compacted_turns(...)` | Gated by flush (§6) |
| `server.py:723` | `_pre_compaction = compacted_through == 0` | Selects stage-1 vs stage-2 stubbing path |
| `server.py:744` | `_ct = int(compacted_through)` | Post-compaction chain collapse |
| `server.py:831` | `compacted_count = int(compacted_through)` | Paging tool injection |
| `tool_query.py:111` | `compacted_through > 0` | VC tool injection |

### Must be parameterized (`history_offset`)

| Location | Current code | Fix |
|----------|-------------|-----|
| `types.py:317` | `history_offset()` returns `compacted_through` | Add `watermark` parameter (see §9.1) |
| `retrieval_assembler.py:119,274,359` | Callers of `history_offset()` | Pass `watermark=flushed_through` |
| `compaction_pipeline.py:128,179` | Callers of `history_offset()` | Pass no watermark (default = `compacted_through`) |
| `tagging_pipeline.py:337` | Caller of `history_offset()` | Pass no watermark (default = `compacted_through`) |

### Keep `compacted_through` (compaction/storage paths)

| Location | Why it stays |
|----------|-------------|
| `compaction_pipeline.py:137,144,233,297,304,508,531` | Pipeline tracks what's been summarized |
| `state.py:867,1162,1176,1178` | Compaction bookkeeping |
| `engine.py:610,622,624` | State restoration checks |
| `engine.py:892,979,981,1036,1089` | Snapshot save/diagnostic |
| `engine.py:1235,1236,1301,1304,1329,1334` | Checkpoint comparison |

### Informational (add `flushed_through` alongside)

| Location | Change |
|----------|--------|
| `dashboard.py:272,522,739,1098` | Display both watermarks |
| `helpers.py:510` | Log both watermarks |
| `state.py:473,574,796` | Log both watermarks |
| `dashboard.html:615,789,1142,1144,1178,1399,1826` | Show `flushed_through` indicator |

### 9.1 `history_offset()` Parameterization

`history_offset()` (`types.py:317`) is shared by compaction/tagging (need `compacted_through`) and payload-assembly (need `flushed_through`). A blanket swap breaks compaction.

**Fix**: Add an optional `watermark` parameter:

```python
def history_offset(self, history_len: int, *, total_turns_indexed: int | None = None,
                   watermark: int | None = None) -> int:
    wm = watermark if watermark is not None else self.compacted_through
    if wm < history_len:
        return wm
    # ... rest of sliding-window logic uses wm
```

- Compaction/tagging callers: no `watermark` arg (defaults to `compacted_through`)
- Retrieval assembler callers: `watermark=self._engine_state.flushed_through`

---

## 10. `drop_compacted_turns()` Boundary Fix

**Current behavior** (`message_filter.py:681–753`): Uses `compacted_through` as a gate only (`> 0` → enabled). Once enabled, drops ALL non-tool turns outside the protected window via `range(protected_start)`, regardless of whether those turns have summaries.

**Problem**: When flush fires after multiple deferred prepare rounds, `compacted_through` might be at turn 50 but 50 more turns have accumulated. Protected window covers the last 6. The function drops turns 0–94, but only 0–49 have summaries.

**Fix**: Add `drop_boundary` parameter. Only drop turns below the boundary:

```python
def drop_compacted_turns(
    body, turn_tag_index, compacted_through,
    *, fmt=None, protected_recent_turns=6,
    drop_boundary: int | None = None,   # NEW
) -> tuple[dict, int]:
    # ...
    boundary_turn = (drop_boundary // 2) if drop_boundary else protected_start
    limit = min(boundary_turn, protected_start)
    for tidx in range(limit):
        # ... existing drop logic
```

Flush gate calls: `drop_compacted_turns(body, ..., drop_boundary=compacted_through)`

---

## 11. Complete Persistence Surface

Both `flushed_through` and `last_request_time` must be persisted across ALL serialization paths.

### Write-side (SAVE)

| Location | What to add |
|----------|------------|
| `engine.py:891` (`extract_session_state()`) | `flushed_through=self._engine_state.flushed_through, last_request_time=self._engine_state.last_request_time` |
| `engine.py:979` (`EngineStateSnapshot(...)` constructor) | Same two fields |

### Dataclasses

| Location | What to add |
|----------|------------|
| `types.py` EngineState | `flushed_through: int = 0`, `last_request_time: float = 0.0` |
| `types.py` EngineStateSnapshot | `flushed_through: int = 0`, `last_request_time: float = 0.0` |
| `session_state.py` SessionState | `flushed_through: int = 0`, `last_request_time: float = 0.0` |

### Serialization (SessionState)

| Location | What to add |
|----------|------------|
| `session_state.py:39–58` (`to_json()`) | Include both fields in dict |
| `session_state.py:60–80` (`from_json()`) | Read with `.get("flushed_through", 0)` etc. |

### Conversion (SessionState ↔ EngineStateSnapshot)

| Location | What to add |
|----------|------------|
| `session_state.py:345–362` (`_state_to_snapshot()`) | Map both fields |
| `session_state.py:364–399` (`_snapshot_to_state()`) | Map both fields |

### Engine restore (LOAD)

| Location | What to add |
|----------|------------|
| `engine.py:554` | `self._engine_state.flushed_through = saved.flushed_through` |
| `engine.py:637` | `self._engine_state.flushed_through = es.get("flushed_through", 0)` |
| `engine.py:773` | `self._engine_state.flushed_through = state.flushed_through` |
| Same three lines | Parallel for `last_request_time` |

### Storage backends

| Location | Change |
|----------|--------|
| `filesystem.py:677` (save) | Add both fields to dict |
| `filesystem.py:737` (load) | Read with defaults |
| `sqlite.py:75` (schema) | `ALTER TABLE engine_state ADD COLUMN flushed_through INTEGER NOT NULL DEFAULT 0; ADD COLUMN last_request_time REAL NOT NULL DEFAULT 0.0;` |
| `sqlite.py:1582` (insert) | Include new columns |
| `sqlite.py:1608,1628,1664` (load) | Read new columns |
| `postgres.py:85` (schema) | Same DDL |
| `postgres.py:1454` (upsert) | Include new columns |
| `postgres.py:1503,1519,1543` (load) | Read new columns |

### Backward compatibility on deserialize

All new fields default to 0/0.0. Old state without these fields deserializes correctly. On first request after upgrade:
- `flushed_through = 0` + `compacted_through > 0` → pre-flush mode
- `last_request_time = 0.0` → `cache_age` huge → cold-cache fast path → immediate flush

First request flushes pending compaction, then subsequent requests benefit from deferral.

---

## 12. Logging

### Defer event

```
DEFER-PAYLOAD: cache_age=%.1fs outbound_tokens=%d threshold=%d compacted_through=%d flushed_through=%d — preserving prefix
```

### Flush event

```
FLUSH-PAYLOAD: reason=%s cache_age=%.1fs outbound_tokens=%d compacted_through=%d flushed_through=%d→%d turns_dropped=%d tools_stubbed=%d media_stubbed=%d
```

Where `reason` is one of: `cold_cache`, `size_threshold`, `legacy_immediate`.

### Two-pass event

```
FLUSH-TWO-PASS: pass1_tokens=%d pass2_tokens=%d delta=%d
```

---

## 13. Known Limitations (P2)

### 13.1 Cache warmth is a best-effort heuristic

`last_request_time` doesn't track model/provider changes. A model switch invalidates the cache but `cache_age` stays fresh. Worst case: defers a flush that could have happened sooner. The hard threshold catches it. Future: scope warmth to `(model, provider, system_hash)`.

### 13.2 Completion time overstates warmth for streaming

Recording at stream completion makes `cache_age` smaller than reality. Biases toward deferral (safe direction). Future: record at request send time.

### 13.3 `merge_consecutive_conversational()` may touch historical messages

This helper at `server.py:813` merges adjacent same-role messages. If a merge reaches into historical content, it changes the prefix. In practice, merge only affects the tail of the conversation. Implementation should verify this or gate the merge to unflushed content only.

---

## 14. Edge Cases

### Process restart / upgrade
`last_request_time = 0.0` → `cache_age` huge → cold-cache fast path. Correct: no warm cache after restart.

### Payload over hard threshold at enable time
Pass 1 measures full payload. Exceeds threshold. Pass 2 flushes and reassembles. Subsequent requests benefit from deferral.

### Multiple prepare rounds pending
Prepare runs several times advancing `compacted_through` from N to M to P. `flushed_through` stays at 0. Retrieval assembler stays in pre-flush mode (no duplicate content). On flush, `flushed_through` jumps to P. `drop_compacted_turns(drop_boundary=P)` drops only summarized turns.

### Slow conversations (>5 min between turns)
Every request finds `cache_age >= flush_ttl_seconds`. Cold-cache fast path. Behaves like today. No deferral benefit (correct: cache already expired).

### `flushed_through > compacted_through` guard
Assert invariant on every update. If violated, log error and clamp: `flushed_through = min(flushed_through, compacted_through)`.

### Cross-worker handoff
Both watermarks persisted in Redis checkpoint and store backends. New worker restores correct state. If checkpoint is lost, defaults to 0/0.0 → cold-cache flush on first request.

### Two-pass latency
Warm-cache-near-limit path pays for two assembly cycles. Fires once per flush cycle (when payload first hits hard threshold). For 200k window with soft=0.30 and hard=0.97, the payload grows from ~60k to ~194k over many turns before this triggers. Expected: rare, bounded cost.

### `defer_payload_mutation=False` with new fields
`flushed_through` is set to `compacted_through` on every request (auto-tracking). All existing code paths run unchanged. The new fields exist but are no-ops. Zero behavioral change.

---

## 15. Testing Strategy

### Unit tests

| Test | Verifies |
|------|----------|
| `test_defer_off_identical_to_today` | `defer_payload_mutation=False` → exact same behavior, `flushed_through` auto-tracks `compacted_through` |
| `test_prepare_does_not_mutate_body` | Prepare phase advances `compacted_through` but body is unchanged |
| `test_warm_cache_skips_all_mutations` | Cache warm + below threshold → no turn drops, no tool stubs, no media stubs, no chain collapse |
| `test_flush_at_hard_threshold` | Warm cache + outbound >= threshold → all mutations applied, `flushed_through` advanced |
| `test_flush_when_cache_cold` | `cache_age >= TTL` → cold-cache fast path, all mutations applied |
| `test_stage1_tool_stubbing_gated` | Warm cache → stage-1 `stub_tool_outputs_by_position()` is skipped (tool outputs stay raw) |
| `test_media_stubbing_gated` | Warm cache → `stub_media_by_position()` is skipped (images stay inline) |
| `test_chain_collapse_gated` | Warm cache → `collapse_turn_chains()` is skipped |
| `test_assembler_uses_flushed_through` | `compacted_through=10, flushed_through=0` → pre-flush mode (tag retrieval, no hints) |
| `test_assembler_post_flush` | `compacted_through=10, flushed_through=10` → post-flush mode (all summaries, hints) |
| `test_drop_respects_boundary` | `drop_boundary=50` with 100 turns → only turns below boundary dropped |
| `test_history_offset_parameterized` | Assembler callers pass `flushed_through`; compaction callers use default `compacted_through` |
| `test_timestamp_set_after_dispatch` | `last_request_time` updated after upstream call, not before gate |
| `test_tentative_flush_not_committed_on_failure` | If pass-2 dispatch fails, `flushed_through` stays at old value |
| `test_invariant_enforcement` | Setting `flushed_through > compacted_through` → error log + clamp |
| `test_persistence_roundtrip` | Save/restore round-trips both watermarks through all paths: SessionState, EngineStateSnapshot, filesystem, sqlite, postgres |
| `test_two_pass_reassembles_context` | Pass 2 re-runs assembler in post-flush mode with correct summaries |

### Integration test

1. Tenant with `defer_payload_mutation=true, soft_threshold=0.30, hard_threshold=0.97`
2. Send requests until prepare fires at soft threshold
3. Verify: `compacted_through` advanced, `flushed_through=0`, payload byte-identical to previous request (excluding new turns)
4. Verify: no tool stubs, no media stubs, no chain collapse, no context hints
5. Continue sending until fully-assembled payload hits 97%
6. Verify: pass 2 fires. All mutations applied atomically. `flushed_through = compacted_through`. Payload shrinks.
7. Verify: metrics report from pass 2 (not pass 1)

### Production validation

1. Deploy with `defer_payload_mutation=false` (no-op baseline)
2. Enable per-tenant: set `defer_payload_mutation=true, soft_threshold=0.30, hard_threshold=0.97`
3. Monitor `cache_read` — should stay >90% between prepare and flush events
4. Watch DEFER-PAYLOAD and FLUSH-PAYLOAD log lines
5. Compare token costs before/after

---

## 16. Files Changed

| File | Change |
|------|--------|
| `virtual_context/types.py` | Add `defer_payload_mutation`, `flush_ttl_seconds` to MonitorConfig. Add `flushed_through`, `last_request_time` to EngineState and EngineStateSnapshot. Add `watermark` param to `history_offset()`. |
| `virtual_context/proxy/session_state.py` | Add fields to SessionState, `to_json()`, `from_json()`, `_state_to_snapshot()`, `_snapshot_to_state()` |
| `virtual_context/engine.py` | Write-side: map new fields in `extract_session_state()` (line 891) and `EngineStateSnapshot()` constructor (line 979). Load-side: restore at lines 554, 637, 773. |
| `virtual_context/core/retrieval_assembler.py` | Lines 112, 141, 199, 336, 419: switch to `flushed_through`. Lines 119, 274, 359: pass `watermark=flushed_through` to `history_offset()`. |
| `virtual_context/core/tool_query.py` | Line 111: switch to `flushed_through` |
| `virtual_context/proxy/server.py` | Flush gate with two-pass logic. Gate ALL mutation operations (drop, stub, collapse, media, stage-1). Set `last_request_time` after dispatch. Tentative flush commit. Logging. Legacy auto-tracking when `defer_payload_mutation=False`. |
| `virtual_context/proxy/message_filter.py` | Add `drop_boundary` param to `drop_compacted_turns()` |
| `virtual_context/storage/filesystem.py` | Serialize/deserialize new fields |
| `virtual_context/storage/sqlite.py` | Schema DDL, insert, load paths |
| `virtual_context/storage/postgres.py` | Schema DDL, upsert, load paths |
| `virtual_context/proxy/dashboard.py` | Expose `flushed_through` alongside `compacted_through` |
| `virtual_context/proxy/helpers.py` | Log `flushed_through` |
