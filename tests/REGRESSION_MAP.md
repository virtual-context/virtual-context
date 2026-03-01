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

### BUG-007 — Broad detection misses phrases (SUPERSEDED)

- **Superseded by**: `vc_recall_all` tool — broad detection removed entirely.
  The LLM now decides when to load all summaries via a tool call instead of
  regex heuristics. See `test_recall_all.py` for replacement tests.
- **Original symptom**: Queries like "what did you say earlier" not detected as broad
- **Original fix**: regex-based `detect_broad_heuristic()` (removed)

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

### PROXY-004b — Consecutive assistant messages cause first-message-is-assistant

- **Symptom**: 400 error when pair 0 is dropped but an unpaired assistant message is force-kept via tool chain integrity, making the filtered output start with `role: assistant`
- **Root cause**: Claude Code extended thinking sends consecutive assistant messages: `user[0], assistant[1](thinking), assistant[2](tool_use)`. Pair 0 = (0,1). msg[2] is unpaired. If pair 0 is dropped but msg[2] is kept, filtered output starts with assistant — API requires first message to be user.
- **Fix**: User-first enforcement — backfill-keep all messages before first kept user message
- **Tests**:
  - `test_proxy.py::TestFilterBodyMessages::test_consecutive_assistant_dropped_pair_keeps_user_first`

### PROXY-004c — Thinking-strip creates dict copies, losing _vc_critical sentinel

- **Symptom**: 8/15 turns 400 in Claude Code A/B test: `unexpected tool_use_id found in tool_result blocks`. Same `toolu_01LX5izFrvfLu2Robj47pXwT` orphaned on every failed turn.
- **Root cause**: `_strip_thinking_blocks` creates shallow copies (`{**msg, "content": filtered}`) of assistant messages that have thinking blocks. `_vc_critical` was set on original `chat_msgs` dicts, not on the copies in `kept`. Alternation enforcement couldn't see the sentinel on the copy, dropped the critical assistant (which had the `tool_use`), orphaning its `tool_result`.
- **Evidence**: `request_log/000038` from A/B run 2026-03-01. Inbound: msg[49] assistant[thinking,text] (pair 24), msg[50] assistant[thinking,text,tool_use(X)] (UNPAIRED, consecutive), msg[51] user[tool_result(X)] (pair 25). After thinking-strip, msg[50] is a new dict. Sentinel on original invisible → dropped → orphan → 400.
- **Fix**: (1) Walk `kept` list in parallel with `keep_msg` to tag the actual objects in `kept` instead of originals in `chat_msgs`. (2) Final safety net: post-alternation orphan check falls back to unfiltered body.
- **Tests**:
  - `test_proxy.py::TestFilterBodyMessages::test_consecutive_assistant_thinking_strip_preserves_tool_chain`

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

### PROXY-021 — Compaction monitor uses stripped history tokens instead of client payload tokens

- **Symptom**: Compaction never triggers in proxy mode. Monitor sees 13.6% utilization (16k stripped tokens) while actual client payload is 68.4% (82k tokens) of 120k context window.
- **Root cause**: `monitor.build_snapshot()` counted tokens from envelope-stripped `conversation_history` instead of the real client payload
- **Fix**: Added `payload_tokens` override to `build_snapshot()`, `on_turn_complete()`, and `fire_turn_complete()`. Proxy passes `_last_payload_tokens` at all call sites.
- **Tests**:
  - `test_monitor.py::test_build_snapshot_payload_token_override`
  - `test_monitor.py::test_engine_on_turn_complete_payload_tokens`

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
- **Fix**: `_bypass_ws` gate in `engine.py on_message_inbound()` — when `retrieval_result.broad or .temporal`, pass `ws_param=None` and `full_segments_param=None` to assembler. Working set preserved for next normal query.
- **Tests**:
  - `test_paging.py::TestBroadBypassesWorkingSet::test_broad_query_gets_all_summaries_despite_expanded_tags`
  - `test_paging.py::TestBroadBypassesWorkingSet::test_broad_query_does_not_render_full_depth`
  - `test_paging.py::TestBroadBypassesWorkingSet::test_working_set_preserved_after_broad_query`

### BUG-015 — Temporal chronological ordering lost by paging depth override

- **Symptom**: Temporal queries ("what did we first discuss?") lose time ordering when tags are at FULL depth in paging working set
- **Root cause**: Assembler overrides temporal retriever's chronologically-sorted summaries with unordered full_segments from working set
- **Fix**: Same `_bypass_ws` gate as BUG-014
- **Tests**:
  - `test_paging.py::TestTemporalBypassesWorkingSet::test_temporal_query_ignores_working_set_depth`

### BUG-016 — Working set segment loading runs unconditionally on every inbound

- **Symptom**: Full segment DB reads for all FULL-depth working set tags happen on every inbound, even when retriever chose broad/temporal path where segments won't be used
- **Root cause**: Paging segment loading block doesn't check retrieval_result.broad/temporal
- **Fix**: Same `_bypass_ws` gate — segment loading skipped entirely for broad/temporal
- **Tests**:
  - `test_paging.py::TestSegmentLoadingGate::test_broad_query_skips_segment_loading`
  - `test_paging.py::TestSegmentLoadingGate::test_temporal_query_skips_segment_loading`
  - `test_paging.py::TestSegmentLoadingGate::test_normal_query_still_loads_segments`

### PROXY-022 — Consecutive same-role messages break alternation after filtering

- **Symptom**: Second message after ingestion crashes with Anthropic API error about role alternation
- **Root cause**: OpenClaw sends consecutive same-role messages (batched Telegram, tool_result + new user text). `_filter_body_messages` kept them as "unpaired" but when surrounding pairs were dropped, output had consecutive same-role entries
- **Fix**: Post-filter alternation enforcement pass — skip any message that repeats the previous role
- **Tests**:
  - `test_proxy.py::TestFilterBodyMessages::test_consecutive_user_messages_preserve_alternation`
  - `test_proxy.py::TestFilterBodyMessages::test_consecutive_user_after_tool_result_preserves_alternation`

### PROXY-023 — Filter keeps compacted messages when paging active, defeating tool interception

- **Symptom**: LLM never calls `vc_expand_topic` because `_filter_body_messages` keeps tag-relevant raw messages even when compacted. LLM reads detail from raw messages, paging tools are dead code.
- **Root cause**: Filter has no awareness of compaction watermark. Compacted turns with matching tags are retained alongside their VC summaries.
- **Fix**: Added `compacted_turn` param to `_filter_body_messages`. When > 0, pairs below watermark are unconditionally dropped. Call site passes `_compacted_through // 2` when `paging.enabled and _compacted_through > 0`.
- **Tests**:
  - `test_proxy.py::TestFilterBodyMessages::test_compacted_turns_dropped_when_paging_active`
  - `test_proxy.py::TestFilterBodyMessages::test_compacted_turn_zero_preserves_current_behavior`
  - `test_proxy.py::TestFilterBodyMessages::test_compacted_turns_rule_tag_still_dropped`

### PROXY-024 — Context-topics list truncates expanded tags, LLM can't discover paging tools

- **Symptom**: LLM sees fragrance summary but doesn't call `vc_expand_topic` — the tag isn't in the `<context-topics>` list due to truncation at 200 tokens. Only first 9 alphabetical tags survive.
- **Root cause**: `_build_context_hint()` builds flat alphabetical list of all ~80 tags, truncates from end. Verbose format (~30t/tag) means only 7-9 tags fit in 200t budget. Tags at summary depth scattered alphabetically, get truncated like depth:none tags.
- **Fix**: Two-tier compact format: expanded tags first (detailed), available tags as comma-separated list. Truncation drops available entries first. Default budget bumped 200→500.
- **Tests**:
  - `test_paging.py::TestContextHintModes::test_autonomous_hint_expanded_tags_listed_first`
  - `test_paging.py::TestContextHintModes::test_autonomous_hint_compact_format_fits_more_tags`
  - `test_paging.py::TestContextHintModes::test_autonomous_hint_truncation_drops_none_first`
  - `test_paging.py::TestContextHintModes::test_supervised_hint_compact_format`

### PROXY-007 — Manual compaction has no concurrency guard

- **Symptom**: Double-clicking "Compact Now" produces duplicate segments from the same messages
- **Root cause**: No lock on `compact_manual()`. Concurrent calls read the same `_compacted_through` watermark.
- **Fix**: `_compaction_lock = threading.Lock()` on ProxyState, non-blocking acquire in dashboard endpoint, 409 Conflict if busy, JS button disabled during flight
- **Tests**:
  - `test_proxy.py::TestCompactionConcurrencyGuard::test_compaction_lock_exists_on_proxy_state`
  - `test_proxy.py::TestCompactionConcurrencyGuard::test_compaction_lock_is_non_reentrant`
  - `test_proxy.py::TestCompactionConcurrencyGuard::test_dashboard_compact_endpoint_uses_lock`

### PROXY-015 — Continuation BAIL silently drops non-VC tools, leaving user with stub response

- **Symptom**: LLM says "Let me page that in instead of guessing." then calls `vc_expand_topic` (intercepted successfully), then tries `memory_search` (client tool). Proxy BAILs: emits `message_end` with `stop_reason=end_turn`, drops the non-VC tool. Client sees only the stub text, no answer.
- **Root cause**: Continuation loop's break path (line 2698) silently discards non-VC tool_use blocks and always emits `stop_reason=end_turn`, giving the client no indication that a tool call is pending.
- **Fix**: Forward non-VC tool_use blocks to client as SSE events and emit `stop_reason=tool_use` so the client can execute them and continue the conversation.
- **Tests**:
  - `test_proxy.py::TestContinuationBailForward::test_non_vc_tool_forwarded_after_vc_continuation`
  - `test_proxy.py::TestContinuationBailForward::test_multiple_vc_then_non_vc_all_forwarded`
  - `test_proxy.py::TestEmitToolUseAsSSE::test_emits_three_events`
  - `test_proxy.py::TestEmitToolUseAsSSE::test_content_block_start_has_tool_use_type`
  - `test_proxy.py::TestEmitToolUseAsSSE::test_delta_has_input_json`
  - `test_proxy.py::TestEmitToolUseAsSSE::test_content_block_stop`

### BUG-017 — Tool loop exhausts max_loops with empty text

- **Symptom**: LongMemEval Q1 returns empty hypothesis despite 364 output tokens. Model called `vc_find_quote` repeatedly without finding matches, exhausted max_loops without producing text.
- **Root cause**: `run_tool_loop()` `for/else` clause didn't force text generation on max_loops exhaustion. All output tokens were tool_use blocks.
- **Fix**: After exhausting max_loops with empty text, execute last pending tools, send one final continuation with `tools` stripped to force text output.
- **Tests**:
  - `test_tool_loop.py::TestRunToolLoop::test_forced_text_after_max_loops_exhausted`

### BUG-029 — Description search substring match promotes irrelevant session to rank 1

- **Symptom**: `find_quote("storing old sneakers")` returns Asian Games segment at rank 1 because `"old"` substring-matches `"gold"` in the tag description. Session-recency sorting then promotes this irrelevant newer session, suppressing the correct answer.
- **Root cause**: `supplement_from_descriptions()` used Python `in` operator (`w in desc_lower`) for word matching, which does substring match. `"old" in "gold"` → `True`.
- **Fix**: Precompile `\b`-bounded regex patterns for each query word; use `pattern.search()` instead of `in`.
- **Tests**:
  - `test_find_quote.py::TestSupplementFromDescriptionsWordBoundary::test_old_does_not_match_gold`
  - `test_find_quote.py::TestSupplementFromDescriptionsWordBoundary::test_old_matches_whole_word_old`

### BUG-018 — Context turns pollute inbound tagger post-compaction

- **Symptom**: LongMemEval benchmark — question about "antique items" generates `meal-prep` tags because last haystack turns were about food. 3/7 questions affected.
- **Root cause**: `on_message_inbound()` passes recent conversation turns as context to the tagger. Post-compaction, these are unrelated to the query and overwhelm the question text in the tagger prompt.
- **Fix**: Skip `context_turns` when `_compacted_through > 0`. Also skip the `_general` context-expansion retry post-compaction.
- **Tests**:
  - `test_broad_query.py::TestContextTurnsPostCompaction::test_no_context_turns_post_compaction`
  - `test_broad_query.py::TestContextTurnsPostCompaction::test_context_turns_passed_pre_compaction`

### BUG-031 — Current-state suppression triggers on topically irrelevant sessions

- **Symptom**: 07741c45 "Where do I currently keep my old sneakers?" — reader answered "under my bed" instead of "shoe rack in closet". The current-state suppression promoted an unrelated gaming-keyboard session (sim=0.26) to HIGHEST_PRIORITY and hid the sneaker sessions.
- **Root cause**: `quote_search.py` activated suppression whenever `current_state` intent + multiple sessions, without checking if the newest session was topically relevant.
- **Fix**: Added topical relevance gate — newest session must have FTS/like/description match or semantic similarity >= 0.4 before suppression activates.
- **Tests**:
  - `test_find_quote.py::TestFindQuoteIntentAndRecency::test_weak_semantic_newest_session_does_not_suppress`

### BUG-032 — Semantic fact search ignores reader's object_contains filter

- **Symptom**: 6d550036 "How many projects have I led?" — reader over-counts (answers 4 instead of 2). `query_facts(verb="led", object_contains="project", status="active")` returns "User leads a team of five engineers" — object is "a team of five engineers", not "project".
- **Root cause**: `_semantic_fact_search` fetches ALL facts for the subject ignoring `object_contains`, then matches by embedding similarity to the intent context. Facts semantically close to "projects led" but failing the explicit `object_contains="project"` filter were returned.
- **Fix**: Post-filter semantic results against `object_contains` when provided — fact must contain the substring in its `object` or `what` field.
- **Tests**:
  - `test_verb_expansion.py::TestQueryFactsSemanticIntegration::test_semantic_search_respects_object_contains_filter`

### BUG-034 — Greedy set cover drops ephemeral primary tags, killing retrieval

- **Symptom**: Ephemeral topics (2-3 turns, e.g., sourdough-starter, wedding-toast) get 0% precision in stress tests. They exist as segments with correct tags but are invisible to retrieval.
- **Root cause**: `compute_cover_set()` picks minimum tags to cover all turns. A broad tag like `baking` covers all 5 turns including the 2 sourdough turns, so `sourdough-starter` is dropped. No tag summary = invisible to embedding-based inbound retrieval.
- **Fix**: Primary tag guarantee — after `compute_cover_set()`, force-include every segment's `primary_tag` even if the greedy cover dropped it.
- **Tests**:
  - `test_engine_integration.py::test_primary_tag_guarantee_ephemeral_gets_tag_summary`

---

## By Test File

| Test File | Bugs Covered |
|-----------|-------------|
| `test_headless.py` | BUG-001 |
| `test_tui.py` | BUG-002 |
| `test_compactor.py` | BUG-003, BUG-004 |
| `test_recall_all.py` | (replaces BUG-007 broad detection) |
| `test_idf_retrieval.py` | BUG-005, BUG-006 |
| `test_retriever.py` | BUG-008, BUG-009 |
| `test_context_bleed.py` | BUG-010 |
| `test_turn_tag_index.py` | PROXY-002 |
| `test_proxy.py` | BUG-008, PROXY-001, PROXY-002, PROXY-003, PROXY-004, PROXY-004b, PROXY-004c, PROXY-005, PROXY-010, PROXY-013, PROXY-014, PROXY-015, PROXY-022, PROXY-023 |
| `test_paging.py` | PROXY-024 |
| `test_monitor.py` | PROXY-021 |
| `test_empty_turn_skip.py` | BUG-013 |
| `test_tag_splitter.py` | BUG-011, BUG-012 |
| `test_find_quote.py` | BUG-029, BUG-031 |
| `test_verb_expansion.py` | BUG-032 |
| `test_engine_integration.py` | BUG-034 |
