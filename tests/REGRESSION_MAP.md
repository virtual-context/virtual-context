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
| `test_proxy.py` | BUG-007, BUG-008, PROXY-001, PROXY-002, PROXY-003, PROXY-004, PROXY-005 |
