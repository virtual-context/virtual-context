# Regression Test Map

Maps bugs found during production deployment to their regression tests.
Use `pytest -m regression` to run all regression tests.

## By Bug ID

### BUG-001 — Headless runner shows `_general` as primary tag

- **Symptom**: Headless replay would show `_general` as the primary tag for every turn
- **Root cause**: Engine integration not wiring inbound tagging correctly
- **Fix**: Ensured `on_message_inbound()` is called and result flows into turn record
- **Tests**:
  - `test_headless.py::TestHeadlessWithEngine::test_engine_methods_called`

### BUG-002 — TUI tag panel not updating after `on_turn_complete`

- **Symptom**: Tag panel showed inbound-only tags, ignoring richer turn-complete tags
- **Root cause**: `Static.update()` didn't reliably repaint from `call_from_thread` callbacks
- **Fix**: Switched to `render()` override so compositor always reads current data
- **Tests**:
  - `test_tui.py::test_tag_panel_updates_after_turn_complete`

### BUG-003 — Tag summary rebuild fires every turn

- **Symptom**: Tag summaries rebuilt on every compaction, even when fresh
- **Root cause**: Missing `covers_through_turn` check to skip fresh summaries
- **Fix**: Added freshness check: skip if `existing.covers_through_turn >= max_turn`
- **Tests**:
  - `test_compactor.py::test_compact_tag_summaries_skips_fresh`
  - `test_broad_query.py::TestBroadFilterHistory::test_broad_true_skips_compacted_post_compaction`

### BUG-004 — Compaction loses granular tags (turn_offset)

- **Symptom**: After compaction, segments lost fine-grained tags — only primary survived
- **Root cause**: Segmenter used local pair indices instead of global turn numbers; compactor didn't merge `refined_tags`
- **Fix**: Added `turn_offset` parameter to Segmenter; compactor takes union of original + refined tags
- **Tests**:
  - `test_compactor.py::test_compact_refined_tags`

### BUG-005 — Vocabulary mismatch (caching → materialized)

- **Symptom**: Query "caching trick" at T71 missed T46's "materialized view" — zero tag overlap
- **Root cause**: Retriever didn't pass store tags to tagger, so tagger invented novel tags
- **Fix**: Pass existing store tags as vocabulary; added `related_tags` field for bridging
- **Tests**:
  - `test_idf_retrieval.py::TestRelatedTagExpansion::test_write_time_related_tags_enable_retrieval`

### BUG-006 — Common tags dominate retrieval

- **Symptom**: Target segment buried under many distractors sharing high-frequency tags
- **Root cause**: All tag matches scored equally — `database` matched 5 distractors equally
- **Fix**: IDF weighting: rare tags score higher than common tags
- **Tests**:
  - `test_idf_retrieval.py::TestIDFRanking::test_idf_ranks_rare_tags_higher`
  - `test_idf_retrieval.py::TestIDFRanking::test_idf_with_rare_query_tag_promotes_target`

### BUG-007 — Broad detection misses phrases

- **Symptom**: Queries like "what did you say earlier" not detected as broad
- **Root cause**: LLM unreliably sets `broad: true`; no deterministic fallback
- **Fix**: Added regex-based `detect_broad_heuristic()` with configurable patterns
- **Tests**:
  - `test_tag_generator.py::TestBroadHeuristic::test_llm_broad_miss_overridden`
  - `test_broad_query.py::TestBroadRetrieval::test_broad_retrieval_loads_tag_summaries`
  - `test_retriever.py::TestEmbeddingRetrieverWithInbound::test_broad_heuristic_applied`
  - `test_proxy.py::TestFilterByTagAndBroad::test_broad_keeps_everything`

### BUG-008 — Compaction dilutes early detail (temporal queries)

- **Symptom**: Temporal queries like "very first thing" couldn't recall early details
- **Root cause**: No temporal detection; no segment-level retrieval for earliest content
- **Fix**: Added `detect_temporal_heuristic()`; retriever sorts by time for temporal queries
- **Tests**:
  - `test_retriever.py::TestEmbeddingRetrieverWithInbound::test_temporal_heuristic_applied`
  - `test_proxy.py::TestFilterByTagAndBroad::test_temporal_keeps_everything`

### BUG-009 — Inbound embedding tagger only sees last 4 turns' tags

- **Symptom**: All inbound messages return `_general` after history ingestion of 47 turns — even specific queries like "tell me about cars"
- **Root cause**: Retriever built inbound vocabulary via `get_active_tags(lookback=4)`, hiding tags older than 4 turns
- **Fix**: Full scan of all TurnTagIndex entries instead of lookback-limited active tags
- **Tests**:
  - `test_retriever.py::TestInboundMatching::test_early_tag_visible_after_many_turns`

### BUG-010 — Context lookback bleeds tags across topic shifts

- **Symptom**: When conversation shifts topics (transit → identity), stale context from the previous topic poisons the tagger, producing irrelevant tags
- **Root cause**: `_get_recent_context()` blindly walked backward N pairs with no topic awareness
- **Fix**: Embedding similarity gate compares current turn against most recent context pair using cosine similarity; below threshold (0.1) → topic shift → context skipped
- **Tests**:
  - `test_context_bleed.py::TestContextBleedGate::test_topic_shift_strips_context`
  - `test_context_bleed.py::TestContextBleedGate::test_continuation_keeps_context`
  - `test_context_bleed.py::TestContextBleedGate::test_short_msg_after_shift_keeps_new_topic`
  - `test_context_bleed.py::TestContextBleedGate::test_low_overlap_continuation_keeps_context`
  - `test_context_bleed.py::TestContextBleedGate::test_single_word_continuation_keeps_context`

### PROXY-001 — Consecutive user messages produce empty pairs

- **Symptom**: OpenClaw batches multiple Telegram messages as consecutive user turns, breaking pair extraction
- **Root cause**: Pair walker assumed strict user/assistant alternation
- **Fix**: Skip mismatched pairs; advance one message at a time on role mismatch
- **Tests**:
  - `test_proxy.py::TestExtractHistoryPairs::test_consecutive_user_messages_at_end`
  - `test_proxy.py::TestExtractHistoryPairs::test_consecutive_user_messages_mid_conversation`

### PROXY-002 — `_general` causes destructive filtering

- **Symptom**: When TurnTagIndex had only `_general` entries, filtering dropped all history
- **Root cause**: `_general` selected as cover tag, matched nothing specific → everything dropped
- **Fix**: Exclude `_general` from cover set; skip filtering when index is empty
- **Tests**:
  - `test_proxy.py::TestFilterByTagAndBroad::test_no_index_entries_skips_filtering`
  - `test_turn_tag_index.py::TestTurnTagIndex::test_compute_cover_set_excludes_general`
  - `test_turn_tag_index.py::TestTurnTagIndex::test_compute_cover_set_only_general`

### PROXY-003 — OpenClaw envelope not stripped

- **Symptom**: Tags like `telegram`, `messaging`, `protocol` generated from channel metadata
- **Root cause**: OpenClaw wraps user messages with channel headers, footers, and `[vc:prompt]` markers
- **Fix**: `_strip_openclaw_envelope()` removes all structural metadata before tagging
- **Tests**:
  - `test_proxy.py::TestStripOpenClawEnvelope::test_strips_full_envelope`
  - `test_proxy.py::TestStripOpenClawEnvelope::test_history_pairs_strip_envelope`

### PROXY-004 — Filter drops unpaired messages (tool chains)

- **Symptom**: Filtered history broke tool_use/tool_result chains, causing API errors
- **Root cause**: Filter only kept tag-matched pairs; tool_result pair was unrelated to any tag
- **Fix**: If a kept pair has `tool_use`, force-keep the next pair; if a kept pair has `tool_result`, force-keep preceding pair
- **Tests**:
  - `test_proxy.py::TestFilterByTagAndBroad::test_tool_use_keeps_tool_result_pair`
  - `test_proxy.py::TestFilterByTagAndBroad::test_tool_result_keeps_preceding_tool_use_pair`

### PROXY-005 — Streaming breaks on tool_result turns (web search, all tool use)

- **Symptom**: "request ended without sending any chunks" when agent uses any tool (web search, file tools, etc.)
- **Root cause**: `tool_result` messages have no text content blocks, so `_extract_user_message()` returns `""`. Proxy fell through to `_passthrough_bytes()` which returns `JSONResponse` even when client expects SSE streaming.
- **Fix**: Route empty-user-message requests through `_handle_streaming()`/`_handle_non_streaming()` instead of `_passthrough_bytes()`. Preserves SSE framing while skipping VC enrichment.
- **Tests**:
  - `test_proxy.py::TestExtractUserMessage::test_tool_result_only_returns_empty`

### PROXY-014 — Third request during ingestion silently ignored (cancel-and-resume breaks)

- **Symptom**: Turns 1→2 cancel-and-resume works fine. Turn 3 (while ingestion still running) does nothing — progress bar stops, no new ingestion starts.
- **Root cause**: `_run_ingestion_with_catchup`'s `finally` block unconditionally runs `_ingested_sessions.add()` and `_transition_to(ACTIVE)`, even when `_IngestionCancelled` is caught. Python's `finally` always executes after `return` in `except`. Turn 3's fast path sees session as already ingested.
- **Fix**: Added `cancelled` flag; `finally` block only marks as ingested when `not cancelled`.
- **Tests**:
  - `test_proxy.py::TestSessionStateMachine::test_third_call_during_ingestion_cancels_second`

### PROXY-013 — Duplicate ingestion thread on second request during INGESTING

- **Symptom**: Second request during background ingestion spawns a duplicate thread that re-ingests from turn 0, causing progress bar to jump backwards and doubling Haiku API calls
- **Root cause**: `start_ingestion_if_needed` only checked `_ingested_sessions` (set after completion), not whether a thread was already running
- **Fix**: Track `_ingestion_thread` + `_ingestion_cancel` event. Cancel old thread, join, re-read turn count after join, verify hash at handoff, resume from last tagged turn.
- **Tests**:
  - `test_proxy.py::TestSessionStateMachine::test_second_call_during_ingestion_does_not_restart`

### PROXY-010 — Different conversations merged into same session

- **Symptom**: Two different Telegram chats (private + group) routed to the same proxy session, cross-contaminating TurnTagIndex and conversation history
- **Root cause**: `SessionRegistry.get_or_create()` returned the first existing session for any request without a `<!-- vc:session -->` marker
- **Fix**: Content fingerprint routing — hash first 5 user messages to distinguish conversations. Priority: marker > fingerprint > claim unclaimed > new session.
- **Tests**:
  - `test_proxy.py::TestContentFingerprintRouting::test_different_conversations_get_different_sessions`
  - `test_proxy.py::TestContentFingerprintRouting::test_same_conversation_reuses_session_via_fingerprint`
  - `test_proxy.py::TestContentFingerprintRouting::test_marker_takes_priority_over_fingerprint`

### BUG-013 — Empty turns produce phantom tag occurrences

- **Symptom**: Tool-use turns with no text content still got tagged via context lookback, inflating TurnTagIndex
- **Root cause**: No empty-content guard in `ingest_history()` or `on_turn_complete()`
- **Fix**: Skip tagging when both user and assistant content are empty; use pair index for turn_number
- **Tests**:
  - `test_empty_turn_skip.py::TestIngestHistorySkipsEmptyTurns::test_empty_pair_not_tagged`
  - `test_empty_turn_skip.py::TestIngestHistorySkipsEmptyTurns::test_whitespace_only_pair_not_tagged`
  - `test_empty_turn_skip.py::TestIngestHistorySkipsEmptyTurns::test_empty_user_nonempty_assistant_still_tagged`
  - `test_empty_turn_skip.py::TestIngestHistorySkipsEmptyTurns::test_nonempty_user_empty_assistant_still_tagged`
  - `test_empty_turn_skip.py::TestIngestHistorySkipsEmptyTurns::test_multiple_consecutive_empty_turns_skipped`
  - `test_empty_turn_skip.py::TestIngestHistorySkipsEmptyTurns::test_ingested_count_excludes_skipped`
  - `test_empty_turn_skip.py::TestOnTurnCompleteSkipsEmptyTurns::test_empty_latest_pair_not_tagged`
  - `test_empty_turn_skip.py::TestOnTurnCompleteSkipsEmptyTurns::test_nonempty_latest_pair_still_tagged`

### BUG-011 — Tag splitter parser rejects string turn numbers from LLM

- **Symptom**: Tag split always returns "Fewer than 2 valid groups" even when LLM returns a valid split — effectively disabling the entire tag splitting feature
- **Root cause**: Haiku returns turn numbers as `"T9"` strings (matching the `[T9]` format in the prompt) or plain string digits `"9"`. The parser only accepted `int`/`float` types via `isinstance(n, (int, float))`, silently dropping all string values
- **Fix**: (1) Parser now handles `str` values — strips `T`/`t` prefix and converts to int. (2) Prompt now explicitly instructs `IMPORTANT: Turn numbers must be plain integers (e.g., 9, 13, 20), NOT strings like "T9".`
- **Tests**:
  - `test_tag_splitter.py::TestTagSplitter::test_string_turn_numbers_with_t_prefix`
  - `test_tag_splitter.py::TestTagSplitter::test_string_turn_numbers_plain_digits`

### BUG-012 — Tag splitter collects empty/wrong text for turns in proxy history

- **Symptom**: 50% of turns sent to the split LLM prompt had empty text (`[T9] `) or contained MemOS preamble content (`# Role\nYou are an intelligent assistant...`) instead of actual user messages. LLM still managed reasonable splits from the turns that had content, but accuracy was degraded.
- **Root cause**: `_collect_turn_text()` used `turn_number * 2` to index into the conversation history, assuming strict user/assistant alternation. In proxy mode, OpenClaw injects MemOS preamble user messages before the real content, creating consecutive user messages that break the indexing.
- **Fix**: New `_extract_turn_pairs()` helper walks the history and pairs the last user message before each assistant response, handling consecutive user messages correctly. Both `_collect_turn_text()` and `_build_broad_tag_summary()` now use this pair-based approach instead of blind index math.
- **Tests**:
  - `test_tag_splitter.py::TestEngineTagSplitting::test_collect_turn_text_with_preamble_messages`

### BUG-014 — Broad query + FULL-depth working set = incomplete overview

- **Symptom**: Broad queries ("summarize everything") only show 1-2 expanded topics when those tags are at FULL depth in paging working set, missing 15+ other topic summaries
- **Root cause**: Paging depth override consumes the tag budget before broad overview summaries can render; assembler's budget check terminates early
- **Fix**: TBD — flatten working set to SUMMARY depth on broad queries
- **Tests**: None yet (open)

### BUG-015 — Temporal chronological ordering lost by paging depth override

- **Symptom**: Temporal queries ("what did we first discuss?") lose time ordering when tags are at FULL depth in paging working set
- **Root cause**: Assembler overrides temporal retriever's chronologically-sorted summaries with unordered full_segments from working set
- **Fix**: TBD — skip working set depth override for temporal result tags
- **Tests**: None yet (open)

### BUG-016 — Working set segment loading runs unconditionally on every inbound

- **Symptom**: Full segment DB reads for all FULL-depth working set tags happen on every inbound, even when retriever chose broad/temporal path where segments won't be used
- **Root cause**: Paging segment loading block doesn't check retrieval_result.broad/temporal
- **Fix**: TBD — gate segment loading on retrieval branch
- **Tests**: None yet (open)

---

## By Test File

| Test File | Bugs Covered |
|-----------|-------------|
| `test_headless.py` | BUG-001 |
| `test_tui.py` | BUG-002 |
| `test_compactor.py` | BUG-003, BUG-004 |
| `test_broad_query.py` | BUG-003, BUG-007 |
| `test_idf_retrieval.py` | BUG-005, BUG-006 |
| `test_tag_generator.py` | BUG-007 |
| `test_retriever.py` | BUG-007, BUG-008, BUG-009 |
| `test_context_bleed.py` | BUG-010 |
| `test_turn_tag_index.py` | PROXY-002 |
| `test_proxy.py` | BUG-007, BUG-008, PROXY-001, PROXY-002, PROXY-003, PROXY-004, PROXY-005, PROXY-010, PROXY-013, PROXY-014 |
| `test_empty_turn_skip.py` | BUG-013 |
| `test_tag_splitter.py` | BUG-011, BUG-012 |
