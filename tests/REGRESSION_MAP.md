# Regression Test Map

Maps bugs found during production deployment to their regression tests.
Use `pytest -m regression` to run all regression tests.

## By Bug ID

### BUG-131 — Raw bodies stored every resent image and the host's replayed context

- **Symptom**: one guild conversation's canonical rows held 727 MB of raw bodies, of which 25 MB were distinct images and under 2 MB was user text; each read of those rows moved megabytes per row into the process, and freed memory stayed resident.
- **Root cause**: the host resends recent images and its own rendering of the channel history with every request, and `save_canonical_turn` stored each turn's raw body as received: 4,051 image parts for 161 distinct images, plus about 1 MB of replayed context per turn that normalization discards.
- **Fix**: `core.raw_bodies.compact_raw_body` moves inline data into `canonical_raw_blobs` (one row per SHA-256, referenced as `vc-blob:sha256:<hex>`) and replaces a host prompt's replayed context section with a labeled hash marker, keeping the adapter's metadata blocks and the request; writes compact on save, `compact_stored_raw_bodies` rewrites existing rows in small batches, attested rows accept only their own compaction through `canonical_raw_compaction_grants`, and deletes sweep unreferenced blobs.
- **Tests**:
  - `test_raw_body_storage.py` — normalization, sender and channel survive compaction, images round-trip, rows share blobs, attested rows accept only their own compaction, backfill and blob cleanup
  - `test_raw_body_storage_postgres.py` — the same storage, backfill and cleanup on PostgreSQL

### BUG-130 — Per-request reads loaded canonical raw bodies

- **Symptom**: every request on a long conversation read megabytes of raw request bodies it never used: the summary audience check read up to hundreds of full source rows, the recent-turn readers read up to 200 rows (110 MB of raw bodies on one conversation), the context hint loaded every tag summary (146 MB peak for 8,035 topics), and a restore without session state read every row's bodies (818 MB peak).
- **Root cause**: those paths used full-row reads (`get_canonical_turn_rows_by_id`, `get_recent_canonical_turns`, `get_all_tag_summaries`, `get_all_canonical_turns`) while reading only audience fields, text, topic names, or the pending turns' bodies; an attachment-only turn also embedded its raw body, inline image data included.
- **Fix**: `get_canonical_turn_audiences` for the audience check, recent-turn reads without raw columns, `get_tag_summary_topics` for the hint, and a restore from `get_all_canonical_turn_texts` that reads raw bodies for the uncompacted turns only; embedding text never includes inline data.
- **Tests**:
  - `test_request_reads_without_bodies.py` — each read, its equivalence with the full read, and the embedding text
  - `test_request_reads_without_bodies_postgres.py` — PostgreSQL returns turn ids as UUIDs; the projections still match

### BUG-129 — Ingestion resume, compaction takeover and batch ingest read the whole conversation

- **Symptom**: every ingestion takeover of a long conversation loaded all of its canonical rows with their raw bodies (728 MB on one conversation, taken over 18 times in one day), so the work a resume or takeover cost grew with the conversation's age instead of with what was pending.
- **Root cause**: ingestion resume, compaction takeover, post-ingest compaction and strict history ingest each read every canonical row (or rebuilt the whole history) and then used only the pending tail, a single batch, or one session date.
- **Fix**: resume reads only the turn groups from its baseline or the first untagged complete group (`first_untagged_complete_group`, `get_canonical_turn_rows_in_groups`); takeover and post-ingest size their request from uncompacted turns only; strict ingest reads its batch's groups and seeds the session date with `latest_session_date_before`. Legacy ungrouped conversations still read whole and filter.
- **Tests**:
  - `test_canonical_read_projection.py` — group range and markers match the full read, strict ingest and takeover history make no full read
  - `test_proxy_session.py::TestSessionStateMachine::test_resume_pending_ingestion_from_store_gap`, `test_legacy_ingestion_lease_ownership.py::test_resume_pending_ingestion_spawns_heartbeat_sidecar` — resume through the bounded reads

### BUG-128 — Tag-summary embeddings were held as Python float lists

- **Symptom**: each compaction of a conversation with about 8,000 tag summaries allocated well over 100 MB of short-lived embedding data per pass (measured: 96 MB live, 162 MB peak per load, against 13 MB for the same values as float32).
- **Root cause**: `load_tag_summary_embeddings` parsed every stored JSON vector into a list of Python floats; the compaction merge check loaded the whole conversation's set, and the snapshot refresh loaded it again and copied each vector while normalizing.
- **Fix**: `load_tag_summary_embedding_vectors` streams rows (a server-side cursor on PostgreSQL) into read-only float32 arrays; the merge check, the snapshot refresh and the retrieval fallbacks use it, and the snapshot is normalized and encoded as one matrix.
- **Tests**:
  - `test_embedding_memory_shape.py` — float32 vectors match the list loader, the snapshot refresh never builds lists, list input keeps its values

### BUG-127 — Hot paths loaded every canonical turn with its raw bodies

- **Symptom**: on a long conversation each tagged turn spent 5–12 s persisting, and every turn that compacted grew the serving process by about 1.5 GB that it did not return; memory pressure restarted the service repeatedly.
- **Root cause**: `get_all_canonical_turns` returns full rows, including raw request and response bodies (728 MB for one conversation); the per-turn hash search, the proxy watermark refresh and the recovered-tag rollup each called it while reading only hashes, ids, group numbers, tags or the prefix count.
- **Fix**: the hash search uses the content-free reconcile projection, the watermark refresh reads `get_compaction_watermark`, and the rollup reads the new `get_canonical_turn_tags` projection (id, turn number, primary tag, tags).
- **Tests**:
  - `test_canonical_read_projection.py` — no full-row read from any of the three paths, results unchanged, projection matches the full rows

### BUG-126 — Tag summaries lost to an interrupted compaction were never rebuilt

- **Symptom**: a conversation's segment tags stayed without `tag_summaries` rows indefinitely (20 on one production conversation, some carried by 77 segments); the divergence check kept failing.
- **Root cause**: a compaction stores its segments before building their tag summaries; when the process restarted between the two, the new segments' tags never got a summary, because later compactions only roll up tags carried by the segments they compact themselves.
- **Fix**: every compaction also rolls up the conversation's segment tags that have no summary (`get_segment_tags_without_summaries`, one query on SQLite and PostgreSQL), taking those tags' turns from stored canonical rows since neither the compacted rows nor a freshly restored turn index hold them.
- **Tests**:
  - `test_tag_rollup_recovery.py` — store lookup, rebuild on the next compaction, turn anchors with an empty turn index, no double-counted turns

### BUG-125 — Facts were gathered and curated after the summaries

- **Symptom**: prepare ran topic selection, summary rerank and fact curation one after another, three judgment-model round trips in sequence.
- **Root cause**: `retrieve` fetched facts only after choosing summaries, and the caller curated them only after `retrieve` returned, although facts depend only on the expanded tags.
- **Fix**: `retrieve` gathers facts on a worker thread once the tags are known and passes them through the caller's `facts_transform` (request filtering and curation) while the summaries are chosen.
- **Tests**:
  - `test_retrieve_facts_in_parallel.py`

### BUG-124 — Regrouping rewrote every turn of the conversation

- **Symptom**: each ingest that wrote a row spent about 360 ms in `regroup` on a conversation of about 1,800 turns.
- **Root cause**: `recompute_canonical_turn_groups` issued one UPDATE per stored turn, although appending a turn leaves every earlier group unchanged.
- **Fix**: only turns whose computed group differs from the stored one are written; PostgreSQL sends them in one batch.
- **Tests**:
  - `test_regroup_writes_only_changes.py`

### BUG-123 — Each engine rebuilt its token counter

- **Symptom**: engine construction spent 150-600 ms in `create_token_counter` with the `tiktoken` counter.
- **Root cause**: every engine loaded its tokenizer again.
- **Fix**: counters are built once per mode and shared within the process.
- **Tests**:
  - `test_token_counter_shared.py`

### BUG-122 — Every Postgres store re-ran schema preparation

- **Symptom**: building an engine for a conversation spent about 450 ms constructing its Postgres store, on every request that landed on a worker without the conversation loaded.
- **Root cause**: `PostgresStore.__init__` called `_ensure_schema()`, which runs every bootstrap and migration statement, for each new store.
- **Fix**: schema preparation runs once per database per process; a failed preparation is retried by the next store.
- **Tests**:
  - `test_postgres_schema_once.py`

### BUG-121 — The context hint required a search before every answer

- **Symptom**: a greeting such as "can you hear me" made the model call `vc_find_quote` or `vc_query_facts` before replying, adding a second provider round to the turn.
- **Root cause**: the supervised hint's tool rules, and the compact rules used under a tight hint budget, ended with "Never answer without searching first", which applies to every message once the VC tools are offered.
- **Fix**: the rules ask for a search when the answer depends on earlier conversations and to reply directly to greetings, acknowledgements and small talk. The stored-hint cache key and the latest-hint fallback now carry a fingerprint of the hint builder, so hints cached before a wording change are rebuilt.
- **Tests**:
  - `test_hint_search_rule.py`
  - `test_context_hint_no_inline_rebuild.py::test_redis_latest_slot_ignores_hints_rendered_by_another_hint_builder`

### PROXY-042 — Offered VC tools changed with each turn's history

- **Symptom**: the first model call of a turn rarely reused the provider's prompt cache from the previous turn; the top-level tool list differed between turns.
- **Root cause**: `vc_restore_tool` was offered only when the turn's payload held stubs, `vc_find_quote` was injected by a separate branch, and the full catalogue depended on the paging mode, so the head of the prompt changed whenever history did.
- **Fix**: `offer_vc_tools` offers one fixed catalogue, restore tool included, on every request while VC's tools are configured on; autonomous paging only sets the tool-use requirement. The separate find-quote, restore-tool and post-valve additions are removed.
- **Tests**:
  - `test_vc_tools_fixed_catalogue.py`

### BUG-119 — Selected room history stayed inside the protected current turn

- **Symptom**: group-chat requests carried a current message of about 1 MB; after the replay block was expanded, roughly 318K tokens of earlier room messages remained in the current turn, which is never trimmed.
- **Root cause**: the host lists recent room messages in a separate "Conversation context (chronological, selected for current message)" section of `#session:` lines ahead of the replay block, and `expand_host_replay` only understood the block.
- **Fix**: `parse_selected_history` turns that section into user and assistant turns, which are placed before the replay block's turns; a message carrying only the section is expanded too.
- **Tests**:
  - `test_host_replay_expansion.py::test_selected_group_history_becomes_turns_before_the_replayed_block`
  - `test_host_replay_expansion.py::test_selected_group_history_without_a_replay_block_is_expanded`

### BUG-118 — Host history block in an older message was never expanded

- **Symptom**: on a client thread reused across turns, the request carried 4 turn groups, one of them about 300K tokens, and the outbound payload was larger than the inbound one (`in=336134t out=344027t`) with nothing trimmed.
- **Root cause**: `expand_host_replay` only looked at the newest user message. A host that projects history once into a thread's first message and then appends later turns leaves the `<conversation_context>` block in an older user message, where it stayed one protected turn.
- **Fix**: the newest user message that carries a block is expanded, wherever it sits; when the current message carries its own block, older blocks are still left alone.
- **Tests**:
  - `test_host_replay_expansion.py::test_a_reused_thread_expands_the_block_in_its_first_message`

### PROXY-041 — Provider stream failures were logged without the provider's reason

- **Symptom**: a request whose upstream stream ended in an error event returned `422 "The provider interrupted the response with an error."` and the logs carried no trace of what the provider reported.
- **Root cause**: `collect_response` raised on `error`, `response.failed` and `response.incomplete` events without recording the event.
- **Fix**: those events are logged as `PROVIDER_STREAM_ERROR` with the provider's error or incomplete details before the request fails.
- **Tests**:
  - `test_stream_error_logged.py`

### PROXY-040 — Provider cache reset whenever a client started a new thread

- **Symptom**: Responses requests for a long conversation reported zero cached input tokens on every turn, although the instructions, tool catalog and system items at the head of the payload were unchanged.
- **Root cause**: the client sets `prompt_cache_key` to its own thread id and starts a new thread each turn, so every turn was routed to an empty provider cache.
- **Fix**: `conversation_prompt_cache_key` replaces a client-supplied key with one derived from the VC conversation id before the request is sent upstream.
- **Tests**:
  - `test_prompt_cache_key.py`

### PROXY-039 — Per-response memory dump stalled every logged request

- **Symptom**: with a request log directory configured, each provider response returned about 3 seconds after the model finished; the delay matched the write time of a 12-13 MB `*.session.json` file beside the request log.
- **Root cause**: both response handlers called `_dump_session_state` before returning, which serialized the whole turn-tag index, every tag summary, tag stats and history on the event loop for every request.
- **Fix**: the session dump is removed; the request log keeps its payload stages (`1-inbound` through `4-to-client`).
- **Tests**:
  - `test_request_log_no_session_dump.py`

### BUG-117 — GPT-6 requests capped at the default context window

- **Symptom**: requests for `gpt-6-astra` were rejected with `413 Payload Too Large: Protected input requires 211704 tokens; only 195904 tokens remain after reserving output.` while the same conversation size on `gpt-5.6-sol` was admitted.
- **Root cause**: `MODEL_CONTEXT_LIMITS` had no `gpt-6` prefix, so `resolve_upstream_limit` returned `DEFAULT_UPSTREAM_LIMIT` (200,000) for every GPT-6 model.
- **Fix**: `gpt-6` resolves to the same limit as `gpt-5`; the two families share one context window.
- **Tests**:
  - `test_model_limits.py::TestResolveUpstreamLimit::test_gpt6_family_matches_gpt5`

### BUG-116 — Codex catalog tools were declared twice

- **Symptom**: a client whose Codex tool catalog already carried `vc_*` tools would also get VC's copies as top-level tools, two definitions of one name; the workaround of denying the client's own copies left scripted calls such as `tools.vc_find_quote(...)` failing with "is not a function".
- **Root cause**: the Responses `inject_tools` and `tool_names` read only the body's `tools` array, not the tools a client declares in `additional_tools` input items.
- **Fix**: `catalog_tool_names` reads the client catalog; injection skips names the client already declares and `tool_names` reports them. Direct calls to those tools are still intercepted by name.
- **Tests**:
  - `test_responses_catalog_tools.py`

### BUG-115 — Dashboard token left read routes open

- **Symptom**: with `VC_DASHBOARD_TOKEN` set, request captures, export, events, telemetry, conversations and settings were still readable without the token; only mutating routes checked it, and the bundled page never sent it.
- **Root cause**: `_check_dashboard_auth` was called per route, from the mutating handlers only.
- **Fix**: a middleware requires the token on every `/dashboard/` route except static assets once a token is set; the page reads `?token=` and sends it on every dashboard request and the event stream.
- **Tests**:
  - `proxy/test_dashboard_token_reads.py`

### BUG-114 — Native Gemini requests and current models got no paging tools

- **Symptom**: paging tools were never injected for native Gemini requests, nor for Claude 5, Fable, Haiku 4.5 or GPT-6 model ids.
- **Root cause**: the paging mode was resolved from the body's `model`, which a native Gemini request carries in its URL path; the default `autonomous_models` list predated those families.
- **Fix**: `model_from_path` supplies the model from the request path when the body names none, and the default list includes the current families.
- **Tests**:
  - `test_paging_model_resolution.py`

### BUG-113 — Component model settings were ignored

- **Symptom**: `tag_generator.model`, `supersession.model` and `curation.model` had no effect; those components ran on `summarization.model`.
- **Root cause**: `_build_provider` built every provider with the provider block's model or the summarization model, never the calling component's.
- **Fix**: `_build_provider` takes the component's model, which wins over the provider block's and the summarization model.
- **Tests**:
  - `test_component_models.py`

### BUG-112 — Facts reached requests from another audience

- **Symptom**: a fact extracted from a DM turn inside an owner conversation was returned by `vc_query_facts`, attached to `vc_find_quote` results, listed by `vc_remember_when` and injected before a guild request, although the summary of the same turn was withheld.
- **Root cause**: the audience check reads source turn ids off each item; summaries carry them and facts do not, so every fact passed.
- **Fix**: `facts_admitted_for_audience` admits a fact exactly when its segment would be, reaching the source turns through the segment. Applied to `vc_query_facts` (and its linked facts), the facts attached to `vc_find_quote`, the fact hits of `vc_remember_when`, and the retrieved facts before curation and injection.
- **Tests**:
  - `test_fact_audience.py`

### BUG-111 — Refused supersessions were proposed again

- **Symptom**: a supersession `fact_decisions` had refused was proposed and judged again on every later pass, with the same facts and sources, and refused again.
- **Root cause**: nothing outside the storage layer read `fact_decisions`; candidate selection in `ingest/supersession.py` never consulted it.
- **Fix**: `get_refused_supersessions` returns the refusals for a replacement fact, and `_admitted_snapshots` drops a candidate refused under the current policy for the same fact versions and source versions. A refusal for a stale snapshot does not count.
- **Tests**:
  - `test_refused_supersession.py`

### BUG-110 — MCP topic tool and resources served other conversations

- **Symptom**: `domain_status`, `virtualcontext://domains` and `virtualcontext://domains/{tag}` listed topics and returned summaries from every conversation in the store, not only the one the MCP server's engine is bound to.
- **Root cause**: the three handlers called `get_all_tags()` and `get_summaries_by_tags()` without `conversation_id`, which those store methods treat as "all conversations".
- **Fix**: pass the engine's `conversation_id` to both reads.
- **Tests**:
  - `test_mcp_conversation_scope.py`

### BUG-109 — Chat Completions clients could not page deeper

- **Symptom**: Chat Completions requests received retrieved context but none of the paging tools (`vc_expand_topic`, `vc_find_quote`, `vc_restore_tool`), so those clients could not open a topic's turns or restore a stubbed output.
- **Root cause**: `OpenAIFormat` kept the base `supports_tool_interception = False` and a no-op `inject_tools`, although the continuation and response codecs already handled Chat Completions tool calls.
- **Fix**: `OpenAIFormat` reports tool interception and injects VC tools as Chat Completions functions; `tool_names` reads tool names in every wire shape for the proxy's de-duplication and logging.
- **Tests**:
  - `test_chat_completions_paging_tools.py`

### BUG-108 — Facts shown to the model carried no pointer to their source

- **Symptom**: a fact in the model's context showed its content but nothing about where it came from, so the model could not page in the turns behind a fact without guessing the topic.
- **Root cause**: `Fact.format_for_prompt` rendered subject, verb, object and descriptive fields but no tag or segment reference, although every fact stores the tags of the segment it came from.
- **Fix**: the rendered line ends with `[topic: <first tag>]`, a topic `vc_expand_topic` can open.
- **Tests**:
  - `test_fact_topic_pointer.py`

### BUG-106 — Restoring a compacted turn produced a request the provider rejected

- **Symptom**: after the model called `vc_restore_tool` on a `[Compacted turn N]` stub in a Responses request, the next upstream call failed with `400 Unknown parameter: 'input[3].idempotencyKey'` and the turn ended with no reply.
- **Root cause**: chain snapshots are stored exactly as the host sent them, here chat-completions messages carrying host bookkeeping fields (`idempotencyKey`, `__openclaw`, `api`, `model`, `provider`), and `_restore_chain` spliced them unchanged into a request of a different wire format.
- **Fix**: `proxy/chain_restore.conform_chain` rewrites the chain into the request's format (chat, Anthropic and Responses items into Responses items; same-format chains reduced to that format's fields), drops opaque reasoning and unpaired tool calls, and a chain that cannot be expressed in the request's format is returned to the model as text.
- **Tests**:
  - `test_chain_restore_format.py`

### BUG-105 — The history filter kept an unanswered message and dropped a protected question

- **Symptom**: after `Turn filter: 3/194 kept`, the outbound payload opened the kept history with an old unanswered user message followed by the reply to a newer question whose user message was gone.
- **Root cause**: `filter_body_messages` keeps unpaired messages unconditionally; when one ended up directly before a kept turn's user message, role-alternation enforcement dropped the later message, which was the kept turn's question.
- **Fix**: alternation enforcement replaces the earlier same-role message when it belongs to no kept turn and the later one opens a kept turn.
- **Tests**:
  - `test_filter_keeps_paired_over_unanswered.py`

### BUG-104 — Recent turns reached the model twice when the host replayed its history

- **Symptom**: on proxied Discord turns every request logged `RECENT_CONVERSATION_RENDER requester_rows=3` and the outbound payload carried the three most recent turns twice: once expanded from the host's history block (with the host speaker tag) and once as injected stored rows.
- **Root cause**: the engine's history omits the host's history block, so recent-conversation continuity selected those turns as missing; the proxy expanded the block into real turns and then injected the stored copies beside them without checking the payload.
- **Fix**: `host_replay.without_host_replayed_groups` drops every stored group whose user message's source message id already appears in a host speaker tag in the outbound body, before injection; member-typed lookalike tags are escaped and do not parse.
- **Tests**:
  - `test_protected_window_host_speaker_dedup.py`

### BUG-103 — A tool loop's second call dropped one more history turn than its first

- **Symptom**: on every tool-loop turn the first call logged `DROP-COMPACTED: removed N` and every continuation `removed N+1`; the second call's cached input stopped at the start of the replayed history.
- **Root cause**: `drop_compacted_turns` excluded the trailing group from history only when it held nothing but user messages; a continuation's trailing group also holds the loop's tool items, so it counted as history and the protected window moved back one turn.
- **Fix**: the trailing group is treated as the turn in progress whenever the request does not end on an assistant message, so every call of a turn protects the same window.
- **Tests**:
  - `test_drop_compacted_in_tool_loop.py`

### BUG-102 — Attached images were sized by base64 length in the protected zone

- **Symptom**: on a tool loop with an attached screenshot the proxy logged `PROTECTED_INTRUSION` at 94-109% of a 200K budget and `PROTECTED_INTRUSION_DEEP` stubbed the loop's own outputs, while the provider billed about 90K input tokens for the same requests; `SANITY_BLOATED_PROTECTED` reported the same inflated size.
- **Root cause**: `stub_tool_outputs_by_position`, `stub_media_by_position` and the sanity log sized protected messages as serialized bytes / 4, so an image's base64 counted about 180K tokens; the Responses estimator also counted list-shaped tool outputs (such as an image returned by a tool) as raw JSON.
- **Fix**: all three use the format's media-aware `estimate_message_tokens`, and the Responses estimator sizes tool-output content parts like message content, images by the media estimate.
- **Tests**:
  - `test_protected_zone_media_size.py`

### BUG-101 — The second call of a tool loop undid the first call's reshaping

- **Symptom**: after a cold `FLUSH_GATE` call that logged `CHAIN-COLLAPSE` and offered `vc_restore_tool`, the next call of the same turn logged `WARM ... mutations HELD`, sent the chain uncollapsed with a different tool list, and reported zero cached input tokens.
- **Root cause**: the gate chose between reshaping (cold) and no reshaping (warm) per request; the turn's first call warmed the cache with a reshaped payload, so its continuations switched to the unreshaped shape the cache did not hold, and the advanced flushed boundary was not shared across workers.
- **Fix**: `proxy/flush_gate.py` keeps the flushed boundary a turn's first call used, keyed by the user message and turn count in shared session state; a warm continuation of that turn replays the same reshaping with that boundary.
- **Tests**:
  - `test_flush_gate_turn_memo.py`

### BUG-100 — VC's context block was re-billed on every round of a Responses tool loop

- **Symptom**: on a Codex tool loop each model call re-sent VC's context block (about 21K tokens, identical for every round of the turn) outside the cached prefix; rounds after the first reported cached input stopping at the previous round's tool traffic.
- **Root cause**: `responses_context.place_context_block` appended the block after every input item, so each round's new tool calls and outputs were inserted ahead of it and the block moved on every call.
- **Fix**: the block is inserted right after the latest user message, so a turn's tool traffic follows it and an unchanged block stays in the cached prefix; history before the latest user message is still untouched.
- **Tests**:
  - `test_responses_context_in_tool_loop.py`
  - `test_responses_context_placement.py::test_next_tool_round_keeps_the_previous_request_as_its_prefix`

### BUG-099 — A running tool loop was stubbed on nearly every call

- **Symptom**: once a tool loop's protected zone passed 60% of `context_window`, every following model call logged `PROTECTED_INTRUSION_DEEP` or `SAFETY-VALVE TOOL-STUB`, the loop's own outputs were replaced with stubs call after call, and the model re-read the same file in small slices; the loop took several times more calls than the same task without stubbing.
- **Root cause**: the pass that reaches into the newest turns used the same 60% trigger as turn-start trimming and stopped as soon as the zone dipped under it, so a growing loop crossed the trigger again on the next call.
- **Fix**: `stub_tool_outputs_by_position` takes `deep_intrusion_trigger`; the proxy passes `monitor.hard_threshold`, so the newest turns are only stubbed once the zone exceeds the hard ceiling, and then back down to the intrusion threshold.
- **Tests**:
  - `test_responses_custom_tool_outputs.py::test_a_running_loop_is_stubbed_only_past_the_hard_ceiling`

### BUG-098 — Stubbing inside a tool loop took the outputs the model was still using

- **Symptom**: on a proxied tool loop over the protected-zone threshold, `PROTECTED_INTRUSION_DEEP` stubbed the newest turn's outputs strictly oldest first, including the file the latest call was still working on, and each stub carried only the command; the model re-ran reads it had already made.
- **Root cause**: the deep intrusion pass ordered candidates by position only and had no notion of which outputs a later read had replaced or which the current call was building on.
- **Fix**: `_order_for_deep_stubbing` stubs superseded outputs first (a later call other than the latest touched the same file-like resource), then outputs unrelated to the latest call, then outputs about what the latest call touches, oldest first within each group; stubs made in that pass carry a short preview of the output.
- **Tests**:
  - `test_responses_custom_tool_outputs.py::test_in_loop_stubbing_takes_superseded_then_unrelated_outputs_first`
  - `test_responses_custom_tool_outputs.py::test_an_in_loop_stub_keeps_a_preview_of_the_output`

### BUG-097 — Outputs stubbed by the safety valve came without the restore tool

- **Symptom**: on proxied tool loops the model received stubbed tool outputs (`SAFETY-VALVE TOOL-STUB`, `PROTECTED_INTRUSION_DEEP`) while the injected catalogue listed no `vc_restore_tool`, and it re-ran the original commands instead.
- **Root cause**: the proxy decided whether to offer the restore tool when it injected the VC catalogue, from stubs made up to that point; the safety valve runs later in the same request and its stubs never updated the catalogue.
- **Fix**: `_add_restore_tool` offers `vc_restore_tool` once (never duplicating it and never modifying the caller's body), and the proxy calls it after the safety valve when the valve stubbed outputs the catalogue did not account for.
- **Tests**:
  - `test_vc_tools_fixed_catalogue.py` (the restore tool is now always offered; superseded by PROXY-042)

### BUG-096 — Fact curation re-ran on every model call of a tool loop

- **Symptom**: every continuation call of a proxied tool loop spent about 0.8 s in `fact_curate_primary` (two Jev calls) although the question and the retrieved facts were unchanged; retrieval memo entries also expired after 300 s while loops ran up to 600 s.
- **Root cause**: only retrieval was memoized per turn; curation was recomputed for identical inputs, and memo entries had a fixed expiry from their first write.
- **Fix**: `RetrievalAssembler._curate_facts` keeps the kept-fact indices in shared session state keyed by a hash of the question and the ordered candidate fact lines, so any change is curated afresh; `load_retrieval_memo` extends an entry's expiry on every read.
- **Tests**:
  - `test_curation_memo.py`

### BUG-095 — Assembly re-tokenized the whole facts block for every candidate fact

- **Symptom**: `pool_fill` took about 1.2 s of every assembly on a large conversation (`ASSEMBLE_BREAKDOWN … pool_fill=1227.5ms` with ~400 facts), repeated on every model call of a proxied tool loop.
- **Root cause**: each candidate fact was charged by tokenizing the whole prospective facts block, so a fill over n candidates tokenized a block of growing size n times.
- **Fix**: `_fill_pool_measured` fills using the block wrapper plus the sum of the lines' own counts, which equals the whole-block count for a tokenizer that never merges across a line break, and confirms it with one whole-block count of the final block; any mismatch (for example a character estimate) repeats the fill with whole-block measurement, so the selection always equals the exact fill. On 520 real facts the fill went from 773 ms to 5 ms with the same selection and token total.
- **Tests**:
  - `test_pool_fill_measurement.py`

### BUG-094 — A turn answered around the engine was never stored on a protected Discord route

- **Symptom**: a Discord turn answered while the engine was unreachable (the host fell back to a direct model call) had no canonical row afterwards, although the next request's replayed history contained it and the model could still quote it.
- **Root cause**: on source-attested Discord group routes only the current attested turn is admitted; a prepare without a claim returns `source_attestation_required_noop`, so replayed history is context only and a turn that never reached the engine could not be written by any later request. On the Codex harness the history also arrives as one `<conversation_context>` block that ingestion did not split.
- **Fix**: `_catch_up_host_history` selects replayed user turns whose host `message-speaker` tag names both a Discord actor and a message id, that come after the newest message the window shares with storage, and whose message id is not stored (`find_canonical_source_message_ids`); each is appended with its replies, its own speaker, the current message's channel and the proved audience, before the current turn is admitted. Untagged, lookalike or id-less turns are never admitted, and nothing is admitted without a stored anchor in the window.
- **Tests**:
  - `test_history_catchup.py`

### BUG-093 — LLM provider calls inherited the judgment client's 3 s timeout

- **Symptom**: actor card rebuilds failed with `LLM HTTP error ... qwen/qwen3-235b-a22b-2507 3024ms error=The read operation timed out` and `RuntimeError: actor card curation failed`; 65 such timeouts in one day, every one at 3,020 to 3,030 ms, and no card was rebuilt for over ten hours.
- **Root cause**: providers and the Jev judgment client share one process-wide httpx client, created with the timeout of whichever caller came first. Jev's `timeout_s` is 3.0 and it passes that per request, but providers relied on the client default, so in a worker where a judgment call came first every provider call was capped at 3 s.
- **Fix**: `BaseProvider.complete` passes `httpx.Timeout(self._timeout, pool=10.0)` on each request.
- **Tests**:
  - `test_provider_request_timeout.py`

### BUG-092 — Tombstone checks parsed every conversation's full session state

- **Symptom**: each tenant membership refresh read 39.9 MB from Redis and parsed 188 session states for one tenant; the two largest states took 140 ms and 105 ms to parse with 50 MB and 35 MB peak allocation, repeated per scoped store every 30 s.
- **Root cause**: the lifecycle tombstone check loaded the authoritative `SessionState` to read its `deleted` flag.
- **Fix**: `SessionStateProvider.is_deleted_authoritative` reads only the stored value's tail, since `to_json` (and marker repair) write `deleted` as the last top-level key; any other layout or a missing key falls back to the full authoritative load, and Redis errors raise. Of 3,978 stored states, 3,927 ended in the default-separator flag and 51 in the compact form; the tail matched the full parse for every default-separator state under 2 MB.
- **Tests**:
  - `test_session_tombstone_probe.py`

### BUG-091 — Speaker roster scans loaded every content column

- **Symptom**: building the speaker roster took 49 to 261 ms of prompt assembly per request, and listing tool definitions for a large conversation often exceeded 8 s. Through a tunnel, the roster scan for a 7,087-row conversation took 2.6 to 3.7 s.
- **Root cause**: `build_speaker_roster` and `resolve_speaker_labels` read the newest 400 logical turns through `get_recent_canonical_turns`, which returns raw and normalized content, tags and fact signals for up to 801 rows through the ordinal view, while the scans use only speaker, channel, audience and ordering fields.
- **Fix**: `get_recent_speaker_rows` returns the same rows in the same order with content reduced to presence markers; Postgres selects only those columns from the base table (45 to 53 ms for the same conversation through the tunnel), and other backends fall back to the full rows.
- **Tests**:
  - `test_speaker_rows_lean.py`

### BUG-090 — Loading a conversation restored everything twice

- **Symptom**: a worker's memory rose by roughly 300 to 700 MB per large conversation it loaded, and several such loads tripped the host memory watchdog. Measured off-box, five large conversations took a process to 839 MB while only 111 MB was live.
- **Root cause**: provider-mode engines start with an empty turn index, so conversation creation always ran the durable restore (parsing the saved engine-state snapshot and loading every canonical row) before hydration from the shared session state replaced all of it. The discarded load set each worker's memory high-water mark.
- **Fix**: when the shared session state exists, creation restores only the live and pending turns from uncompacted canonical rows (`restore_live_turns_from_canonical_rows`, gated on `SessionStateProvider.has_state`); the full restore remains the fallback. The same five conversations then take 389 MB.
- **Tests**:
  - `test_live_turn_restore.py`

### BUG-089 — Shared session state froze after a conversation's first save

- **Symptom**: `Save rejected for <conv> — stale version 0 < N` on every session save; the shared state stayed at its first checkpoint (for one conversation, 1,596 indexed turns while 3,532 were stored), so each request re-hydrated that checkpoint and replayed every turn stored after it, with a large memory spike per request.
- **Root cause**: `SessionStateProvider.save` is a compare-and-swap on `version`, but `extract_session_state` no longer carried the version the engine had loaded, so every snapshot saved as version 0 and was rejected once the stored version reached 1.
- **Fix**: `hydrate_from_session_state` records the loaded version, `extract_session_state` carries it, and `note_session_state_saved` records the version a successful save wrote.
- **Tests**:
  - `test_session_state_version_roundtrip.py`

### BUG-088 — A conversation without saved engine state was re-tagged from scratch

- **Symptom**: after a derived-data reset (which deletes engine state), compaction's segmenter found every turn missing from the turn-tag index (3,515 of 3,515) and called the tagging model again for each, ignoring the tags stored on the canonical rows.
- **Root cause**: engine start-up rebuilt the turn-tag index from canonical rows only while restoring a saved engine state; with none saved, the index stayed empty.
- **Fix**: outside provider mode, an engine that starts with an empty index rebuilds it from the conversation's tagged canonical rows, and after any start-up restore the segmenter and retriever (built before the restore replaces the index) are pointed at the restored index. Provider mode keeps restoring through the injected session state.
- **Tests**:
  - `test_tag_index_restore_without_state.py`

### BUG-087 — The proxy crashed on Apple silicon when threads embedded at once

- **Symptom**: the proxy process exited with SIGSEGV (`EXC_BAD_ACCESS`, faulting thread "metal gpu stream" in `at::native::arange_mps_out`) after its first requests on a Mac, with no Python traceback.
- **Root cause**: the process-wide sentence-transformers model was loaded on the default device, which is MPS on Apple silicon, and request, tagging and compaction threads encode concurrently without serialization.
- **Fix**: the shared model is loaded with `device="cpu"` on every platform.
- **Tests**:
  - `test_embedding_model_device.py`

### BUG-086 — tag_select raised IndexError when nothing was left to select

- **Symptom**: `IndexError: list index out of range` at `result.primary = tags[0]` in `_apply_tag_select` during a history re-tag.
- **Root cause**: when the judgment model kept no existing topic but reported an untagged subject, and every tag the tagging model proposed already existed, the selected list was empty.
- **Fix**: an empty selection leaves the tagging model's result unchanged.
- **Tests**:
  - `test_tag_select_seam.py::test_no_kept_topic_and_no_new_tag_keeps_the_tagging_model_result`

### BUG-085 — Every proxy-route Discord group turn logged a false admission error

- **Symptom**: `ERROR SOURCE_ATTESTATION_REQUIRED phase=prepare ... canonical admission skipped` on each proxied Discord group turn, although the turn was stored with its source message id when the reply completed.
- **Root cause**: the proxy prepare never carries the adapter's source claim, so the attested-route gate always skipped prepare admission and reported it as an error. Admission of these routes belongs to the attested completion, which keeps its own error for a genuinely unattested turn.
- **Fix**: the prepare-phase skip logs `SOURCE_ATTESTATION_DEFERRED` at INFO; behavior is unchanged.
- **Tests**:
  - `test_canonical_source_admission.py::test_an_unattested_prepare_defers_to_completion_without_an_error`

### BUG-084 — Dense facts removed by curation were logged as budget skips

- **Symptom**: `FACT_DENSE_BREAKDOWN assembler ... skipped_dense_budget=15` on a turn whose facts block used 5,139 tokens with 43,282 of the pool unused.
- **Root cause**: the counter counted every dense-ranked fact missing from the selection, including facts that fact curation removed before assembly ever saw them.
- **Fix**: `skipped_dense_budget` now counts only dense facts that reached assembly and were not selected; `dense_removed_before_assembly` counts the rest.
- **Tests**:
  - `test_fact_dense_retrieval.py::test_dense_facts_removed_before_assembly_are_not_counted_as_budget_skips`

### BUG-083 — Fact curation failed on large fact sets and on dropped connections

- **Symptom**: With about 520 retrieved facts the judgment service answered `400 {"error_type":"max_tokens_exceeded"}`; separately, some calls failed at once with `RemoteProtocolError: Server disconnected`. Either way curation made no judged choice and fell back.
- **Root cause**: `jev_fact_curation` sent every fact in one request regardless of size, and `JevClient._post` did not resend when a pooled keep-alive connection had been closed by the server.
- **Fix**: Facts are grouped into batches whose estimated request tokens fit `curation_batch_tokens` (40,000) and the batches are sent concurrently; a set that fits still goes in one call. `_post` resends once on `RemoteProtocolError`.
- **Tests**:
  - `test_jev_fact_curation_batches.py`

### BUG-082 — Dense fact retrieval loaded every fact and vector on each request

- **Symptom**: With `retrieval.fact_dense_retrieval` on, each retrieval read every live fact row and its JSON vector for the conversation (about 33K rows and 278 MB of JSON for a large conversation, 30.9 s through a remote connection) before ranking.
- **Root cause**: `_fetch_facts_dense` called `load_fact_embeddings`, which joins full fact rows to `embedding_json` and parses every vector, then scored them in a Python cosine loop.
- **Fix**: New store method `search_fact_embeddings` returns only the top facts and their scores. PostgreSQL ranks inside the database once `migrate-semantic-vectors` has added the pgvector cache to `fact_embeddings` (a trigger keeps it in sync from `embedding_json` and the row's own `model`), ranking vector rows first and checking liveness for a padded candidate pool; an incomplete cache raises instead of reloading every vector. SQLite ranks id plus vector rows with numpy and reads fact rows for the winners. Stores without it return `None` and the retriever keeps the old path.
- **Tests**:
  - `test_fact_dense_search.py`
  - `test_fact_dense_search_postgres.py`

### BUG-081 — Segments whose summary held the answer were missed by chunk matching

- **Symptom**: A question answered by one segment's summary ("average muscle 81.7% and fat 14.0% on
  Aug 12") never reached the judged candidates: its segment ranked 72nd by chunk similarity, outside
  the 30 candidates.
- **Root cause**: segment chunks were embedded from the raw turn text only, which buries figures in
  conversation, while the summary that states them compactly had no embedding.
- **Fix**: `embed_and_store_chunks` appends the segment summary as the segment's last chunk, and
  `backfill_segment_summary_chunks` rewrites the chunks of segments stored before that.
- **Tests**:
  - `test_segment_summary_chunk.py`

### BUG-080 — A compaction left running on an idle conversation was taken over forever

- **Symptom**: The stale-lease sweeper logged SWEEPER_TAKEOVER_SPAWN for the same compaction operation
  about thirty times a minute on every worker, for a conversation whose phase was already `active`.
- **Root cause**: the takeover drives the ordinary prepare path, which does nothing when the conversation
  is not compacting, so the operation stayed `running` with a stale heartbeat and was found again on
  every tick.
- **Fix**: `fail_orphaned_compaction_operation` marks exactly that operation failed, only while it is
  still running, still stale, on the current epoch and its conversation is not compacting, in one
  statement; the sweeper calls it when a takeover does not resume a compaction.
- **Tests**:
  - `test_orphaned_compaction_operation.py`
  - `test_orphaned_compaction_operation_postgres.py`

### BUG-079 — Unreplied messages never compacted and capped the compacted prefix

- **Symptom**: A conversation whose history contains a user message with no assistant reply
  (routine in multi-member channels) reports a compacted prefix that stops at that message
  forever: `get_compaction_watermark` returned `(52, 25)` for a conversation with 1,431 compacted
  turns, so `drop_compacted_turns` removed only 26 turns and the proxy shipped 2,817 of 2,889
  history items (223,984 tokens against a 200,000 window) after a full compaction.
- **Root cause**: `load_uncompacted_groups` admitted only groups with both a user and an
  assistant half, so a lone message was never compacted, and the watermark loop in
  `get_compaction_watermark` (and the engine fallback `_canonical_prefix_watermark`) required
  every leading group to be a compacted exact pair, so the first lone message ended the prefix.
- **Fix**: the reader admits any uncompacted group with at least one non-blank half; the
  watermark counts every leading fully-compacted group by its message halves and stops only at
  the first group with an uncompacted row or no content.
- **Tests**:
  - `test_unreplied_turns_compaction.py::test_unreplied_message_outside_protected_tail_is_compactable`
  - `test_unreplied_turns_compaction.py::test_store_watermark_counts_a_compacted_unreplied_turn_as_one_message`
  - `test_unreplied_turns_compaction.py::test_engine_prefix_counts_a_compacted_unreplied_turn_as_one_message`
  - `test_relational_contracts.py::test_compaction_watermark_stops_at_first_incomplete_pair`

### BUG-078 — Store pools pinned Postgres connections until the server ran out of slots

- **Symptom**: After a container recreate, `PostgresStore.__init__` failed with
  `psycopg_pool.PoolTimeout: couldn't get a connection after 30.00 sec` while the server
  logged `FATAL: remaining connection slots are reserved for roles with the SUPERUSER
  attribute`; prepare returned 500 and the sweeper looped on `ensure_loaded_none`.
  `pg_stat_activity` held 197 of 200 slots, 196 idle, all from the app user.
- **Root cause**: every store pool was built with `min_size=1, max_idle=300`, so each engine
  pinned at least one connection for its lifetime. A multi-worker deployment caches one
  engine per conversation per worker, so the backlog sweeper walking a dozen large
  conversations on eight workers right after a recreate opened more pools than the server
  had slots. A store whose bootstrap then failed left its pool alive and reconnecting in the
  background.
- **Fix**: pools open connections on demand (`min_size=0`) and release idle ones after 60
  seconds; a failed schema bootstrap closes the pool before the error propagates.
- **Tests**:
  - `test_postgres_store.py::test_postgres_store_uses_bounded_connection_pool`
  - `test_postgres_store.py::test_postgres_store_closes_pool_when_schema_bootstrap_fails`

### BUG-077 — Reply-aware participant continuity

- `test_reply_participant_continuity.py::test_third_party_reply_serves_history_without_borrowing_preferences` — current requester keeps their card while the referenced participant contributes attributed history only.
- `test_reply_participant_continuity.py::test_reply_query_reaches_primary_retry_and_curator_without_changing_requester` — bounded reply query survives retry and curation without changing canonical text or roles.
- `test_reply_participant_continuity.py` — direct-reply selection, unlinked and unproved scope exclusion, adapter-owned parent parsing, actual scoped-store expiry, structural escaping and token budget controls.

### BUG-076 — Attested source occurrence time

- `test_source_event_times.py` — optional UTC claim validation, exact pair admission and replay, legacy replay without weaker timestamp admission, no ingestion-date fallback, conflict/rollback, stale source and lifecycle rejection, schema guards, card rebuild invalidation and unchanged audience/channel receipt proof.
- `test_source_event_times_postgres.py` — the same admission/replay and later-enrichment boundaries on a named disposable PostgreSQL database, including exact guard-function validation.
- `test_actor_card_source_time_evidence.py` — occurrence versus ingestion distinction, missing-authority exclusion, bounded large reads, fingerprint invalidation and segment evidence timestamp labels.

### BUG-075 — Source-grounded finite agreements and resolved continuity

- **Cause:** Blanket transience and banter rejection discarded accepted ongoing agreements; cards lacked validity and scoped finite carryover.
- **Fix:** Preserve distinct acknowledged outcomes and bounded response preferences, independently verify dates, retain original intervals, and enforce validity during reads.
- **Tests:**
  - `test_actor_card_commitment_policy.py` — actual curator and admission prompt contracts, explicit date/attested occurrence requirements, playful acceptance, separate outcome history, and immutable bounds.
  - `test_actor_card_curation_shape.py` — grouped claims, label/shape refusal, complete fallback and retention of the previous card after invalid responses.
  - `test_actor_card_commitment_rebuild.py` — normalized proposals through admission/storage/read, cross-channel continuity with audience isolation, carryover, interval identity and malformed date rejection.
  - `test_actor_card_validity.py` — clock boundaries, stale/dirty cards, future carryover, invalid data, interval collision refusal, compatible normalization and additive schema migration.
  - `test_actor_card_validity_postgres.py` — corresponding PostgreSQL lifecycle, scope, collision and migration behavior in a disposable database.

### BUG-074 — Strict history indexing precedes tag completion

- **Cause**: Strict history marks exact existing rows tagged without the canonical embeddings deferred by admission, so the durable worker skips them.
- **Fix**: Embed verified physical rows before the tag compare-and-set; failed groups remain retryable without advancing in-memory entries, progress or checkpoints.
- **Tests**:
  - `test_strict_history_indexing.py` — real strict-history and durable-selector path, role-local retrieval, partial failures and retries, exact source/lifecycle controls, hydrated replay and legacy combined rows.
  - `test_history_tagging_index_postgres.py` — PostgreSQL strict-history indexing and retry parity through the real storage and query path.

### BUG-073 — Audited historical assistant-channel enrichment

- **Cause**: A channel fill on an existing audience-corrected assistant row changes the fingerprint in its immutable audience receipt.
- **Fix**: Explicit source-pinned manifests record the channel-only transition separately and preserve original audience/source evidence. Ordinary channel writers cannot bypass receipt authorization; stale cards retain their immutable claims for rebuilding.
- **Tests**:
  - `test_assistant_channel_enrichment.py` — exact source-pair proof, either audience-repair order, later audience lifecycle fencing, cross-channel assistant quote retrieval, strict manifests, rollback and immutable authorizations.
  - `test_assistant_channel_enrichment_integration.py` — backend guard capability, retained card claims and ordinary channel-writer refusal.
  - `test_assistant_channel_enrichment_cli.py` — explicit database selection, private manifests, default verification and planning without startup migrations.
  - `test_assistant_channel_enrichment_postgres.py` — real PostgreSQL proof, guard, rollback, deletion-cascade and card-evidence parity in a disposable database.
  - `test_channel_backfill_receipts.py` — receipt-aware dry-run/apply parity, skipped-row limits, retained label evidence, unrelated write-error propagation and tenant CLI totals.

### BUG-072 — Exact assistant completions retain their source channel

- **Cause**: An assistant completion without its own envelope was durably paired with its attested user source but lacked the channel required for retrieval.
- **Fix**: Exact completion admission derives the assistant channel only after validating the source claim, rejects explicit channel disagreement, and retains role-local human identity. Existing completed rows are not silently repaired.
- **Tests**:
  - `test_completed_assistant_channel.py::test_attested_completion_is_recalled_across_channels` — missing or matching explicit assistant channel yields source-backed recall of the original assistant concession and no human identity bleed; missing-channel case failed before the fix.
  - `test_completed_assistant_channel.py::test_attested_completion_rejects_conflicting_assistant_channel` — conflicting explicit assistant channel refuses the pair without writes; failed before the fix.
  - `test_completed_assistant_channel.py::test_unattested_completion_does_not_inherit_user_channel` — an unproved source cannot donate neighboring channel metadata.
  - `test_completed_assistant_channel.py::test_conflicting_user_source_is_rejected_before_assistant_admission` — a contradictory source claim cannot authorize assistant provenance.
  - `test_completed_assistant_channel.py::test_completed_turn_surface_preserves_channel_without_assistant_envelope` — the public completion API retains the exact channel when a provider reply contains only role and content.
  - `test_completed_assistant_channel.py::test_completion_of_existing_user_preserves_source_channel` — completing a pre-existing exact user row persists the new assistant's source channel.
  - `test_completed_assistant_channel.py::test_historical_resend_reports_stored_channel_without_repair` — a completed historical pair with a blank assistant channel is idempotent and reports the unchanged stored value rather than claiming a repair.

### BUG-071 — External actions are not lasting communication preferences

- **Cause**: A persistent external-resource change and the agent's compliance could be mistaken for a preference about future responses, then applied across contexts.
- **Fix**: Both semantic-model prompts distinguish external actions from lasting forms of address, tone, and response format, including stored reply settings. Admission cannot admit to satisfy coverage; one-shot service requests can curate as `no_durable_context`. Misclassified immutable fresh or existing entries reject with `wrong_kind` after subject checks. Policy 18 reconsiders previous admissions when a rebuild is explicitly scheduled; it does not enqueue clean cards.
- **Tests**:
  - `test_actor_card_resource_preferences.py::test_both_prompt_surfaces_distinguish_response_preferences_from_external_actions` — the shared contract reaches both models, including the synthetic action's honored reply; failed before the fix.
  - `test_actor_card_resource_preferences.py::test_rebuild_rejects_customization_but_keeps_genuine_address_and_response_preference` — fresh and carried-over rejection decisions remove the resource action while retaining a genuine address and explanation preference on a cross-context read. Mocked decisions validate the rebuild path rather than live semantic accuracy.
  - `test_actor_card_resource_preferences.py::test_action_only_substantive_actor_can_reject_every_candidate_without_failure` — primary, fallback, and coverage adjudicator allow zero admissions, yielding `no_durable_entries` with no failure count.
  - `test_actor_card_resource_preferences.py::test_curator_can_classify_action_only_evidence_as_no_durable_context` — no fabricated entry is required for an isolated completed request.
  - `test_actor_card_resource_preferences.py::test_reproposed_carryover_keeps_existing_origin_and_is_rejected` — an identical fresh proposal does not exempt an existing entry from re-admission.
  - `test_actor_card_resource_preferences.py::test_actor_subject_action_keeps_wrong_subject_priority` — source-role errors remain distinct from kind errors.
  - `test_actor_card_resource_preferences.py::test_repeated_resource_actions_do_not_exclude_stored_reply_preferences` — repeated actions remain separate from lasting stored reply-language preferences.

### BUG-070 — Audited audience correction preserves immutable source replay

- **Cause**: Ownership merges retain audience boundaries, while direct audience rewrites conflict with source attestation and can erase durable card evidence.
- **Fix**: Explicit opaque audience manifests record complete-pair authorizations, retain original receipts, and permit only exact original/current source replay. Stale cards remain hidden with their immutable evidence retained for reprojection and re-admission.
- **Tests**:
  - `test_audience_reassignment.py` — synthetic complete-pair, source integrity, tenant/lifecycle, active work, rollback, SQL guards, idempotency, card preservation and later owner merge contracts; unattested-row refusal, historical receipt adoption/replay rejection and valid-digest tenant/alias/pair fences.
  - `test_audience_reassignment_postgres.py` — real PostgreSQL parity for exact replay, stale and valid-digest manifests, guards, audit cascade cleanup, durable card preservation, unsupported-row refusal, legacy adoption rollback and current trigger-body requirements; requires a disposable database.
  - `test_audience_reassignment_cli.py` — read-only planning/default verification, non-WAL database byte/journal preservation, explicit database selection, exclusive private manifests, missing-schema/stale-trigger refusal and safe errors.

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

### PROXY-038 — Embeddings held as Python lists filled worker memory

- **Symptom**: A worker serving a conversation with about 7,500 tags held roughly 460 MB of embeddings
  (tag-name vectors for inbound tagging and the tag-summary snapshot), and every snapshot read cloned the
  whole snapshot, pushing the host under its memory floor.
- **Root cause**: vectors were stored as lists of Python floats (about 32 bytes per value) in the
  process-wide tag vector cache, the inbound and stored-turn taggers, and the snapshot cache, and the
  snapshot was copied on every read.
- **Fix**: vectors are held as shared read-only float32 arrays, packed snapshots decode straight into one
  float32 matrix whose rows are the per-tag vectors, and snapshot reads copy only the mapping.
- **Tests**:
  - `test_embedding_memory_shape.py`

### PROXY-038 — Embeddings held as Python lists filled worker memory

- **Symptom**: The first retrieval for a conversation with about 7,500 tags grew a worker by well over a
  gigabyte (tag-name vectors for inbound tagging, the tag-summary snapshot and segment chunk vectors),
  and every snapshot read cloned the snapshot, pushing the host under its memory floor.
- **Root cause**: vectors were stored in Redis as float64 or JSON and held in process as lists of Python
  floats (about 32 bytes per value) in several copies, and every read decoded or cloned them again.
- **Fix**: Redis holds float32 vectors (older float64 and JSON values are read and rewritten), vectors
  decode to read-only float32 views without intermediate copies, snapshot reads share the vectors, the
  segment chunk matrix is shared through Redis per snapshot version, and taggers keep one copy of each
  vector plus a similarity matrix reused until the candidate tags change.
- **Tests**:
  - `test_embedding_memory_shape.py`

### PROXY-037 — Workers kept serving the tag-summary embeddings they loaded first

- **Symptom**: After compaction saved new tag summaries from one worker, the other workers' embedding
  signal kept scoring the snapshot they had loaded earlier, so new topics were invisible to them until
  the process restarted.
- **Root cause**: the in-process copy of the snapshot was returned whenever present and never compared
  with the shared snapshot in Redis.
- **Fix**: every save and delete bumps a shared version key; each read compares it with the version
  held in process and reloads when they differ. The same entry holds a normalized float32 matrix built
  once per version.
- **Tests**:
  - `test_tag_summary_embedding_snapshot_freshness.py`

### PROXY-036 — A completed proxy turn was stored without its proved audience

- **Symptom**: A routed turn's rows were stored with attribution version 0 and an assistant row with no
  channel, so every summary later built from the turn would be withheld.
- **Root cause**: the turn-complete path appends the pair through `ingest_single` without a reply edge,
  so no audience was recorded, and the assistant half takes a channel only from its own metadata, which
  a provider response never carries.
- **Fix**: the path proves the message's raw route through `resolve_request_audience`, passes the
  derived edge, and gives a proved pair's assistant half the user half's channel.
- **Tests**:
  - `test_completion_audience_stamp.py`

### PROXY-035 — Summaries of a multi-channel guild were withheld under per-channel scope

- **Symptom**: Every tag and segment summary for a guild conversation rendered as "[summary withheld:
  speaker attribution is unresolved ...]"; for example 8 of the 16 source turns behind the hcg-dosing
  summary were admitted.
- **Root cause**: `speaker_audience_scope` defaulted to `channel`, which admits only source rows from the
  requesting channel. A guild whose channels share one conversation builds summaries from several
  channels, so no summary could prove all of its sources.
- **Fix**: the default is `conversation` (the proved audience boundary still applies, channel is
  provenance). A request with no channel (a DM) keeps exact channel matching, because conversation
  scope admits only group-channel rows.
- **Tests**:
  - `test_actor_attribution.py::test_conversation_scope_is_the_default_for_group_channel_requests`
  - `test_actor_attribution.py::test_conversation_scope_keeps_a_dm_request_on_its_exact_channel`
  - `test_config.py::TestSearchKnobPlumbing::test_defaults_keep_the_guard_on_and_annotations_dark`

### PROXY-034 — A conversation named out of band never proved its audience

- **Symptom**: On a signed route every summary was withheld and ingested rows were stored with a blank
  `audience_conversation_id` and attribution version 0.
- **Root cause**: preparation took the inbound route only from an in-band conversation marker. A route
  verified out of band carries none, so the request audience resolved to empty and the speaker context
  was ineligible.
- **Fix**: when the state resolver marks the conversation out of band and records
  `request.state.conversation_route_id`, that raw id is the inbound route for preparation and the
  request context.
- **Tests**:
  - `test_out_of_band_route_audience.py`

### PROXY-033 — A multi-round tool message was stored once per round

- **Symptom**: Each routed Vast message that used tools produced two stored turns: the question plus the
  first round's interim note ("Checking the bridge and camera paths live.") and the question again plus
  the final answer; the extra pair also changed retrieval's recent context between rounds.
- **Root cause**: every round's response was persisted as a finished turn, and ingest paired the user
  message with the interim note the host resent inside the tool loop.
- **Fix**: for fresh-thread requests (one real user message), responses that end in tool calls are not
  persisted, interim notes are not ingested as the reply, and the final response stores the message once
  with the interim notes folded into the answer. Requests that resend the whole conversation keep their
  existing behavior.
- **Tests**:
  - `test_one_turn_per_message.py`

### PROXY-032 — Every ingest recomputed groups and anchors over the whole conversation

- **Symptom**: Each call of a routed tool loop spent ~1.7s in ingest with zero rows written:
  `regroup_ms=1279`, `anchors_ms=381` on a 6,969-row conversation.
- **Root cause**: `_ingest_prepared_turns_locked` ran `recompute_canonical_turn_groups` and
  `_refresh_persisted_anchors` unconditionally, although both are functions of the stored rows and a
  resend of already-stored history changes neither.
- **Fix**: both passes run only when the call wrote rows; every writing call still runs them.
- **Tests**:
  - `test_noop_ingest_skips_whole_conversation_passes.py`

### PROXY-031 — Retrieval context depended on which worker served the call

- **Symptom**: Three calls of one routed message ran on workers 19, 15 and 20; each logged
  `RETRIEVAL_MEMO miss` with identical message and vocabulary fingerprints but a different `ctx`
  fingerprint, so every call re-ran the full 2.5s retrieval.
- **Root cause**: inbound tagging took its recent pairs from the worker's in-memory conversation
  history, which differs by worker depending on which requests each one served.
- **Fix**: the recent pairs come from the stored conversation through a dedicated store entry point
  (`get_recent_context_turns`), falling back to the passed-in history only when nothing is stored.
- **Tests**:
  - `test_retrieval_context_is_worker_independent.py`

### PROXY-030 — Requests rebuilt the context hint inline after every compaction

- **Symptom**: A routed request spent 2.6s rendering the context hint (`cache_layer=both_miss`,
  `get_all_tag_summaries_ms=1830`) although the post-compaction prewarm had already rendered it for the
  same compaction state an hour earlier.
- **Root cause**: the hint cache key included `flushed_prefix_messages`, a per-session payload
  watermark that lags while a warm prompt cache is held (the request had 3,188, the prewarm 6,951), so
  the request key never matched the prewarmed key; and on a miss the request always rendered inline.
- **Fix**: the key covers only what the hint depends on; on a miss the request serves the latest hint
  for the same conversation generation and paging mode, and renders inline only when none exists.
- **Tests**:
  - `test_context_hint_no_inline_rebuild.py`

### PROXY-029 — Last-resort budget enforcement truncated the biggest items, not the oldest

- **Symptom**: Over budget, `enforce_payload_budget` cut replies inside the protected recent turns
  while older turns stayed intact (10 long replies, 6 protected, 8,000-token window: turns 0-6 cut).
- **Root cause**: `_plan_budget_reductions` ranked candidates by estimated bytes saved, so the
  largest items went first wherever they sat.
- **Fix**: candidates are planned in payload order, oldest first, until the deficit is covered; the
  stage may still reach protected turns, but never cuts a newer turn before an older one.
- **Tests**:
  - `test_budget_enforcement_order.py`

### PROXY-028 — Attachments on the current message were dropped by the host-replay split

- **Symptom**: Routed Codex-harness calls carried an `input_image` on the current user message and
  the request sent upstream had none.
- **Root cause**: `expand_host_replay` rebuilt the current message from its text alone.
- **Fix**: only the text part holding `<conversation_context>` is rewritten; every other part
  (images, files, audio, other text) is kept in order.
- **Tests**:
  - `test_current_attachments_preserved.py`
  - `test_host_replay_expansion.py::test_attachments_in_the_current_message_survive_the_split`

### PROXY-027 — VC context ahead of the conversation defeated prompt caching in every format

- **Symptom**: Routed Codex-harness calls reported `cached_tokens: 0` on every call although the
  host's 67K of developer prompts and tool catalog were identical call to call.
- **Root cause**: a 2026-04-16 storage refactor moved VC's context block, which changes between
  calls, ahead of the conversation in every format: the head of `instructions` (Responses), the
  system message (Chat), `system_instruction` (Gemini), and, by routing Anthropic proxy injection
  through the tool-loop adapter, the Anthropic system prompt, which it then flattened to a string,
  dropping the client's own cache breakpoints. The per-format tests were rewritten to the new
  locations, so nothing failed.
- **Fix**: one rule per format, used by the proxy and by the tool-loop adapters: the block rides the
  latest user message (the last input item for Responses), earlier blocks are removed with only
  VC's own separator, the client's text is kept byte-for-byte, and Anthropic requests stay within
  four cache breakpoints.
- **Tests**:
  - `test_context_cache_prefix.py` states the property itself for all four formats and both
    injection paths; a placement ahead of the conversation fails it whatever the per-format tests
    say. Do not relax it to fit a new placement.
  - `test_responses_context_placement.py`, `test_chat_context_placement.py`

### PROXY-026 — Host-embedded history replay was never trimmed; compacted-turn drop removed instructions

- **Symptom**: A Codex-harness host payload of ~171K tokens against a 90,000-token window left VC
  at 155K-173K on every call (`BUDGET Payload 173118t exceeds budget 90000t`,
  `PROTECTED_INTRUSION: protected zone 208780t is 232% of budget`). 81.5K of it was the host's
  `<conversation_context>` replay inside the current user message.
- **Root cause**: the replay block lives inside the current user message, so every filter treated it
  as the protected current turn. Separately, `drop_compacted_turns` dropped any turn group outside
  the protected window, including groups holding only developer messages or the `additional_tools`
  catalog item; and because the Codex harness groups its catalog, instructions and scaffolding
  user items into the first turn, dropping that turn removed all of them (outbound 7,185 tokens,
  the model ran without its instructions or tools).
- **Fix**: `proxy/host_replay.expand_host_replay` splits the newest user message's replay block into
  ordinary user/assistant items before filtering (after ingestion and the completion snapshot, so
  nothing is stored twice). `drop_compacted_turns` removes only conversation items from a dropped turn; developer and system items, non-message items and host scaffolding user items stay. `filter_body_messages` treats the leading block of system/developer messages, non-message items and host scaffolding as instructions, so role-alternation enforcement can no longer collapse consecutive developer prompts.
- **Tests**:
  - `test_host_replay_expansion.py` (block parsing, expansion shape, no-op, input not mutated,
    expanded history shrunk by the drop, instructions and catalogs kept)

### PROXY-025 — Budget-enforced message stubbing (context_window not enforced)

- **Symptom**: With `context_window: 5000`, proxy sends 28k+ tokens upstream. Tool chain referential integrity keeps compacted messages.
- **Root cause**: Turn counting mismatch between VC internal history and client payloads with dense tool chains.
- **Fix**: Hash-based stub replacement via `stub_compacted_messages()`. Budget auto-promotion when overhead exceeds window. Over-budget alerting.
- **Tests**:
  - `test_proxy.py::TestStubCompactedMessages::test_stubs_compacted_turn_simple_pair`
  - `test_proxy.py::TestStubCompactedMessages::test_stubs_turn_with_tool_chain`
  - `test_proxy.py::TestStubCompactedMessages::test_stub_eliminates_tool_chain_integrity_issue`
  - `test_proxy.py::TestComputeEffectiveBudget::test_compute_effective_budget_auto_promotes`
  - `test_proxy.py::TestTurnTagIndexHashLookup::test_turn_tag_index_hash_lookup`

### BUG-035 — Cross-conversation context leakage in shared stores

- **Symptom**: New conversations get context injected from other conversations via shared store. A cron conversation about "oura data fetch" gets segments about crochet, Bridgerton, and makeup.
- **Root cause**: Retriever and store retrieval methods had no `conversation_id` filtering. All store methods searched the entire DB, not scoped to the current conversation.
- **Fix**: Added `conversation_id: str | None = None` to all retrieval methods in the ABC, all 3 store implementations (SQLite, Postgres, Filesystem), and threaded it through retriever, quote_search, tool_loop, and engine.
- **Tests**:
  - `test_conversation_scoping.py::TestEndToEndScoping::test_new_conversation_gets_no_context_from_other`
  - `test_conversation_scoping.py::TestEndToEndScoping::test_find_quote_scoped_no_cross_conversation_leaks`
  - `test_conversation_scoping.py::TestEndToEndScoping::test_alias_ride_along_scoped`

### BUG-036 — Sort-key gap exhaustion collides with boundary rows

- **Symptom**: Prepare-payload ingest aborts with `UNIQUE (conversation_id, sort_key)` violation (Postgres UniqueViolation / SQLite IntegrityError) on mid-insertion into a conversation whose insertion-point rows sit ≤0.002 apart; large mid-inserts also silently mis-order rows past the right boundary.
- **Root cause**: `IngestReconciler._allocate_sort_keys` clamped its step to 0.001 when the bounded gap was too tight, letting allocated keys land ON or PAST `right_key`. Self-priming: the first clamped allocation writes 0.001-spaced rows, after which every insertion between them collides deterministically.
- **Fix**: Bounded allocation signals exhaustion (returns `None`) instead of clamping, with a float-precision strict-interior guard. The reconciler then opens the gap via `shift_canonical_turn_sort_keys` (single UPDATE whose delta exceeds the shifted range's spread — transient-collision-safe in any row visit order) and re-allocates; per-row descending fallback for stores without the bulk helper.
- **Tests**:
  - `test_ingest_sort_key_rebalance.py::TestAllocatorBoundedGap` (exhaustion signaling)
  - `test_ingest_sort_key_rebalance.py::TestMidInsertRebalance` (end-to-end rebalance, self-priming loop, rows_touched consistency, per-row fallback)
  - `test_ingest_sort_key_rebalance.py::TestShiftHelperSQLite`
  - `test_sort_key_shift_postgres.py::TestShiftHelperPostgres` / `TestMidInsertRebalancePostgres`

### BUG-038 — Strict tagging aborts on concurrently-tagged rows

- **Symptom**: `RuntimeError: strict canonical tagging expected at least N covered ingestible entries, found M across K rows` on catch-up ingestion of a multi-worker conversation; the whole batch falls through to the row-based DB sweep, losing payload-context tagging for the still-untagged rows.
- **Root cause**: The strict precondition in `ingest_history` required every payload entry to map to an UNTAGGED canonical row. Rows legitimately tagged between the prepare and the follow-up pass — by another worker's row sweep or a prior pass over an overlapping payload — broke the count.
- **Fix**: The precondition counts coverage over the conversation's row tail INCLUDING tagged rows. The pair walker gains a hydrate fast-path: pairs whose backing rows are all tagged (matched by per-message `turn_hash`) get their TurnTagIndex entries from the stored row tags, consuming the strict cursor without invoking the tag generator or rewriting rows. Half-tagged pairs fall through to the normal tagger (idempotent). Supporting fix: the canonical-turn full-row loaders now SELECT `covered_ingestible_entries` so legacy combined rows (coverage 2) count correctly.
- **Tests**: `test_strict_tagging_tagged_rows.py` — prod-signature repro, hydration tag fidelity, zero-tagger-call full hydration, untagged-tail still tagged, half-tagged fall-through, missing-rows and stale-epoch invariants preserved.

### BUG-048 — Reconciliation loads every column of every stored turn, including the text it never reads

- **Symptom**: Ingest cost scales with stored conversation size and is independent of request size, and the per-row cost grows with row width rather than staying flat. A no-content projection of the same rows measures flat at 4.2-5.6 us/row across a 7x range in conversation size while the full row goes 22.9 to 95.2 us/row over the same range.
- **Root cause**: `IngestReconciler` loaded stored history through `get_all_canonical_turns`, which selects every column. Reconciliation keys on `turn_hash`, `sort_key` and identity/provenance columns, and never reads a stored row's text; the only thing it asks of the content is whether a row carries user text at all, as a role gate before taking speaker attribution. The content columns are the widest part of a row and are stored out of line, so on Postgres they dominate a load that never uses them.
- **Fix**: New `CanonicalTurnReconcileRow` and `get_canonical_turn_reconcile_rows` on both backends, selecting only the reconciliation columns plus `has_user_content` / `has_assistant_content` flags. All three reconciler loads go through `_load_reconcile_rows`, which falls back to the full load when a backend returns None; None is distinguished from an empty list because reading "cannot project" as "no rows" would reconcile every payload against no history. It is deliberately a separate type rather than a `CanonicalTurnRow` with empty text, so a projected row reaching a write path raises instead of blanking stored content, which is what the sort-key rebalance fallback surfaced. `exact_resend` now returns the prepared rows carrying the stored ordinal rather than the stored rows themselves, so `CanonicalIngestResult.rows` is one type on every path.
- **Tests**:
  - `test_ingest_projected_rows.py::test_presence_flag_agrees_with_python_strip` (the flag and `(value or "").strip()` must never disagree, including whitespace-only content; the default single-argument TRIM strips spaces only and reports a newline-only row as carrying content)
  - `test_ingest_projected_rows.py::test_attribution_still_persists_through_the_projection` (a late-derived sender still reaches storage through the role gate)
  - `test_ingest_projected_rows.py::test_projection_preserves_every_field_the_enrichment_merge_reads` (each merge-read column carries a distinguishing value, so a blanked projection is caught)
  - `test_ingest_projected_rows.py::test_result_rows_never_carry_a_projected_row` (across exact_resend, append and interior_overlap)
  - `test_ingest_projected_rows.py::test_exact_resend_rows_carry_the_stored_ordinal`
  - `test_ingest_projected_rows.py::test_projected_row_cannot_be_written_back`
  - `test_ingest_projected_rows.py::test_unsupported_backend_falls_back_to_the_full_load`

### BUG-050 — An actor card rebuild that loses its build marker to live traffic is reported as a failed commit

- **Symptom**: `RuntimeError: actor card replacement did not commit cleanly` aborts actor card consolidation on an actively-used conversation. Four occurrences in one production log, all on the same busy guild, with rebuild spans of 20.6s, 43.4s, 76.5s and 181.1s, and live rows written inside three of the four windows. The wording reads as data corruption, which is what put it on a blocker list; in fact nothing was written and the same actor rebuilt successfully ten minutes later with a matching input hash.
- **Root cause**: `_rebuild_actor_card` is optimistic-concurrency. It installs a unique `build_marker` on the profile, enumerates inputs, calls the curation and admission models, then commits under a compare-and-set on that marker; `replace_actor_card` declines the write when the marker is no longer installed. A database trigger clears the marker on every canonical turn inserted for a conversation the actor speaks in. So on a conversation that keeps talking, any rebuild slower than the gap between two messages loses the CAS by construction — the expected outcome, not the exceptional one. The caller could not tell that decline apart from a write that reported success but left the row wrong: both raised, and both recorded `stale_or_rejected_write`, which is a failed outcome at the storage layer and increments `failure_count` toward permanent suppression at three.
- **Fix**: the caller classifies the decline. When our input hash did not land AND the marker is no longer ours, the attempt is recorded as `superseded`, logged at info, and returns zero written without raising; the card is left untouched and still dirty, so the next consolidation rebuilds it against the newer evidence. `superseded` is not a failed outcome, so a busy actor is no longer counted toward suppression for being busy. Both halves of the test are required because `written == 0` alone cannot prove a decline — a clean-empty card also writes zero rows. Every genuine bad-write signal keeps its hard failure, and the replacement transaction, its CAS and its lock domain are unchanged.
- **Tests**:
  - `test_actor_cards.py::test_new_turn_during_model_call_cannot_be_lost_by_card_commit` (the protection is unchanged — stale card not written, profile still dirty, no card readable — only the report changes)
  - `test_actor_cards.py::test_superseded_rebuild_stays_rebuildable` (three consecutive race losses accumulate no failures, and the next quiet pass commits)
  - `test_actor_cards.py::test_missing_profile_after_replace_still_fails_hard`

### BUG-049 — Tools that cannot scope by speaker accepted `speaker` / `speaker_only` and silently ignored them

- **Symptom**: A `vc_remember_when` call carrying `speaker=<handle>` and `speaker_only=true` returned HTTP 200 with results covering every participant in the window. The response was byte-identical to the same call without the arguments (same md5), while a `vc_find_quote` control differed. A question asked about one participant was answered with another participant's material, and the reader attributed it to the participant named in the question.
- **Root cause**: `speaker` / `speaker_only` are read in exactly one place, `_resolve_speaker_conditioning`, reached from exactly two tools: `vc_find_quote` and `vc_query_facts`. The schema correctly withheld the properties from the other six tools, but withholding a property does not stop a caller sending it, and `execute_vc_tool` dispatched on tool name without ever inspecting unsupported arguments. The six non-scoping tools therefore accepted the selection, dropped it, and emitted no telemetry saying so. Silence is the defect: an argument that errors teaches the caller, an argument that is dropped produces a confident wrong attribution that looks like a successful call.
- **Fix**: `SPEAKER_SCOPING_TOOLS` derives the supported set from the two tools whose execution consumes the arguments, and `execute_vc_tool` refuses a speaker selection addressed to anything else before dispatch, naming the tool, the refused arguments and the tools that do enforce a selection, and logging `SPEAKER_SCOPING_REFUSED`. "Requested" uses the same predicates the scoping tools use, so `speaker: null` and `speaker_only: false` are not requests and are not refused. `vc_remember_when`'s description now states it cannot restrict to one participant and names the two tools that can, and the `speaker` property description says that on its own it only prefers a participant when ranking and that `speaker_only=true` is what restricts. Speaker filtering inside `remember_when` is deliberately not implemented here; refusing is the safe half.
- **Tests**:
  - `test_speaker_scoping_refusal.py::test_non_scoping_tool_refuses_speaker_arguments` (every non-scoping tool x each of the three request shapes; asserts no tool body ran)
  - `test_speaker_scoping_refusal.py::test_remember_when_refusal_names_the_supported_tools`
  - `test_speaker_scoping_refusal.py::test_non_scoping_tool_still_runs_without_speaker_arguments` (the refusal is scoped to the argument, not the tool)
  - `test_speaker_scoping_refusal.py::test_explicit_null_speaker_is_not_a_request_to_scope`
  - `test_speaker_scoping_refusal.py::test_scoping_tools_are_exactly_the_ones_that_read_the_arguments`
  - `test_speaker_scoping_refusal.py::test_remember_when_description_states_it_cannot_scope_by_speaker`

### BUG-047 — Anchor refresh rewrites the whole conversation's anchor table on every ingest

- **Symptom**: Ingest cost grows linearly with stored conversation size and is independent of what the caller sent. A 15-message request against a 9,173-row conversation costs the same as a 490 KB one. On the largest production conversation the prepare path exceeded the client's timeout often enough that a quarter of prepares were abandoned before a response was sent.
- **Root cause**: `_refresh_persisted_anchors` rebuilt the complete anchor set on every call and persisted it through `replace_canonical_turn_anchors`, which is a `DELETE` of every anchor row for the conversation followed by a re-INSERT of the entire rebuilt set. The set holds one row per window start per window size (3, 4, 5), so roughly 3N rows are deleted and 3N re-inserted per ingest. Appending a turn creates exactly one new window start per window size and invalidates none, so nearly the whole rewrite was writing back byte-identical rows. Measured on a 9,200-row conversation: 27,597 anchor rows rewritten, of which 6 were new and 0 removed. Both ingest fast paths (`exact_resend`, `tail_append`) paid it too, and on `exact_resend` the anchor set is provably unchanged, making the entire rewrite dead work.
- **Fix**: `_build_anchor_rows` extracted so the rebuild and the delta derive their sets with one ruler. `_refresh_persisted_anchors` accepts the row sequence persisted before the ingest, derives the required anchor set for both sequences in memory, and writes only the difference through the new `apply_canonical_turn_anchor_delta` store surface, which deletes and inserts exact `(window_size, anchor_hash, start_turn_id)` triples scoped to the conversation. Because the delta is derived from the caller's prior sequence rather than from a read of stored anchors, it is applied only when `count_canonical_turn_anchors` agrees with the derived prior set; any disagreement (anchors never built, a torn write, a concurrent writer) falls back to the full rebuild, preserving the self-healing property the unconditional rebuild provided. Callers that cannot name the prior sequence keep the rebuild.
- **Tests**:
  - `test_ingest_anchor_incremental.py::test_incremental_delta_matches_full_rebuild` (the identity property across append, interior insert, prefix insert, in-place modification, interior removal, tail shrink, shrink below window size, and wholesale replacement)
  - `test_ingest_anchor_incremental.py::test_incremental_result_is_order_independent` (both temporal orderings converge on the canonical rebuild)
  - `test_ingest_anchor_incremental.py::test_append_writes_bounded_anchor_rows_not_whole_conversation` (an append writes 3 anchors regardless of conversation length; a full-conversation rewrite fails this)
  - `test_ingest_anchor_incremental.py::test_unchanged_sequence_writes_nothing` (a resend that changes no row issues no anchor write)
  - `test_ingest_anchor_incremental.py::test_diverged_store_falls_back_to_full_rebuild`
  - `test_ingest_anchor_incremental.py::test_anchor_hashes_still_resolve_windows_after_delta` (post-delta digests still map to their start row, and superseded digests do not survive)
  - `test_ingest_anchor_incremental.py::test_sqlite_delta_write_matches_rebuild_write` (the SQL delta leaves the same table state as the wholesale replace)
### BUG-044 — Compaction offset splits a recovered cross-channel turn

- **Symptom**: A temporary reply preference authored in one Discord guild
  channel is acknowledged there but ignored in another channel. The outbound
  recent-conversation block contains only the assistant acknowledgement, not
  the authoritative user instruction.
- **Root cause**: Tier 3 inserts `source=db_recent` rows into channel-local
  payload history before `history_offset()` is applied. The payload-owned
  compaction offset is then used as a raw index into the merged list, so it can
  consume the user half of a recovered logical group while leaving the
  assistant half.
- **Fix**: Build the uncompacted view with a source-aware, active-tail-safe
  slice: consume the offset from payload-owned rows only, never from DB-recent
  rows or the trailing unstamped active-user block.
- **Tests**:
  - `test_protected_window_gate.py::test_tier3_db_pair_survives_payload_compaction_offset`

### BUG-045 — Payload dedup suppresses the canonical user before offset

- **Symptom**: A guild-wide temporary reply preference is acknowledged in its
  source Discord channel but ignored in another channel. Canonical storage
  contains the correctly attributed user instruction and assistant
  acknowledgement, yet the outbound `<recent-conversation>` contains only the
  assistant acknowledgement.
- **Root cause**: Tier 3 dedup ran before the channel-local payload watermark.
  The payload user suppressed its canonical duplicate by
  `source_message_id`, then `history_offset()` removed the payload copy. The
  unkeyed canonical assistant half survived alone. Separately, the Tier 3
  store query limited physical rows and could split a logical group at the
  oldest window boundary.
- **Fix**: Slice payload-owned history before canonical merge, force Tier 3
  when a nonzero payload offset makes a Tier-2 equality skip unsafe, and cache
  that exact merged view for paging reassembly. Dedup uses adjacency-scoped
  logical groups so reused historical group numbers cannot suppress unrelated
  turns; incomplete payload fragments are replaced by the complete canonical
  group, then recovered groups are interleaved ahead of newer stamped payload
  turns. Legacy adjacency is never accepted as channel/audience provenance
  proof. A failed/empty canonical read preserves the unsliced payload. SQLite
  and Postgres select the newest N logical groups and return every physical
  row in each.
- **Tests**:
  - `test_protected_window_gate.py::test_tier3_payload_duplicate_cannot_suppress_db_user_before_offset`
  - `test_protected_window_gate.py::test_tier2_equal_with_payload_offset_forces_canonical_replacement`
  - `test_protected_window_gate.py::test_tier3_read_failure_preserves_unsliced_payload`
  - `test_retrieval_assembler_protected_window_merge.py::test_merge_source_match_suppresses_entire_split_db_group`
  - `test_retrieval_assembler_protected_window_merge.py::test_merge_replaces_incomplete_payload_fragment_with_canonical_pair`
  - `test_retrieval_assembler_protected_window_merge.py::test_merge_replacement_keeps_canonical_group_before_newer_payload`
  - `test_retrieval_assembler_protected_window_merge.py::test_merge_complete_payload_pair_can_span_tool_scaffolding`
  - `test_retrieval_assembler_protected_window_merge.py::test_merge_dedup_does_not_conflate_reused_group_numbers`
  - `test_retrieval_assembler_protected_window_merge.py::test_merge_legacy_adjacency_does_not_inherit_channel_provenance`
  - `test_sqlite_mirror_store_methods.py::test_get_recent_limit_preserves_complete_split_groups`
  - `test_sqlite_mirror_store_methods.py::test_get_recent_limit_does_not_conflate_reused_group_number`
  - `test_postgres_mirror_store_methods.py::test_get_recent_limit_preserves_complete_split_groups`
  - `test_postgres_mirror_store_methods.py::test_get_recent_limit_does_not_conflate_reused_group_number`
  - `test_recent_conversation_assembly.py::test_budget_does_not_conflate_reused_raw_group_numbers`
  - `test_recent_conversation_assembly.py::test_budget_legacy_fallback_evicts_only_leading_contiguous_group`

### BUG-046 — Retained peer-channel history suppresses the canonical copy

- **Symptom**: The complete newest preference pair is correct in canonical
  storage but absent from the next channel's `<recent-conversation>`. Older
  groups render normally, and the preference continues to work only in the
  channel where it was authored.
- **Root cause**: A unified guild reuses one in-memory engine history across
  channels, while Discord sends channel-local model payloads. Tier 3 treated
  any matching retained engine row as proof that the current payload already
  carried the canonical turn, so the immediately preceding peer-channel group
  was suppressed even though it was not model-visible. An active-only ingest
  result could also suffix-stamp that current row's identity onto older
  completed history when the active-tail drop equaled the entire result.
- **Fix**: When a request has a proved origin channel, only retained rows from
  that exact channel may suppress a canonical copy; production nested channel
  metadata is read through `get_origin_channel`, with the DB-recent top-level
  shape as fallback. Active-tail ingest rows are removed with an empty-safe
  helper before completed-history stamping. Requests without a proved channel
  retain legacy dedup behavior.
- **Tests**:
  - `test_protected_window_gate.py::test_other_channel_engine_history_cannot_suppress_canonical_recent_pair`
  - `test_protected_window_gate.py::test_same_channel_payload_twin_still_suppresses_canonical_copy`
  - `test_protected_window_gate.py::test_same_channel_active_tail_race_still_suppresses_db_user`
  - `test_canonical_turn_id_stamping.py::test_drop_active_tail_equal_to_all_rows_returns_empty`
  - `test_canonical_turn_id_stamping.py::test_drop_active_tail_keeps_completed_prefix`
  - `test_canonical_turn_id_stamping.py::test_drop_active_tail_zero_is_identity_view`

### BUG-043 — RRF's missing-signal penalty buries embedding-only candidates below the fused top-K

- **Symptom**: On analog queries with no keyword overlap, the tag the embedding signal surfaces most strongly lands far outside the fused top-K (embedding candidates observed at fused rank 15-33 while the selection cut is 10), because RRF penalizes every candidate absent from the idf/bm25 signals. The context-augmented embedding signal (BUG-042) surfaces the right tag but fusion then discards it.
- **Root cause**: `score_candidates` fuses three ranked signals with a missing-signal penalty rank; a strong embedding-only candidate carries penalty ranks for idf and bm25 and cannot reach the top-K on the embedding weight alone.
- **Fix**: New `retrieval.scoring.embedding_reserved_seats` (default 0 = byte-identical legacy). After fusion/dampening/boost and before the top-K cut, `apply_embedding_reserved_seats` seats the top-N embedding-signal candidates (raw pre-gravity embedding order) that are outside the fused top-K (K = strategy `max_results`) by raising their fused scores into the gap between the last surviving entry and the first displaced one, so the retriever's score-descending cut yields surviving + reserved. Only the seated tags' scores change; a degenerate boundary tie nudges the displaced entries down. Composes with BUG-042.
- **Tests**:
  - `test_embedding_reserved_seats.py::TestReservedSeatMath` (seating parity vs the rule on a synthetic fixture and an anonymized real prod fused/embedding dump; lowest-of-top-K displaced)
  - `test_embedding_reserved_seats.py::TestLegacyByteIdentical` (N=0 no mutation, everything-already-fits, embedding-already-in-top-K, defaults)
  - `test_embedding_reserved_seats.py::TestNoGoldDisplacement` (buried gold seated without evicting gold already in top-K; seated score lands between survivor and displaced)
  - `test_embedding_reserved_seats.py::TestConfigPlumbing`

### BUG-042 — Under-specified analog queries miss the relevant tag on the bare embedding signal

- **Symptom**: The embedding retrieval signal scores only the bare inbound message against tag-summary vectors, so an under-specified query whose surface words name neither the gold tag nor the summary's distinctive nouns ("what should I get her for her birthday") fails to surface the relevant tag even when the elided topic sits in the immediately preceding turns.
- **Root cause**: `compute_embedding_candidates` had one query vector — the bare message embedding produced by the inbound embedding tagger, which discards the conversational context it receives. Recent-turn context that would disambiguate the query was never folded into signal 3.
- **Fix**: New `retrieval.scoring.embedding_context_turns` (default 0 = byte-identical legacy) blends the last N conversational turns + the current message into a second query embedding on the same encoder. `embedding_context_guard` (default true) takes the per-tag MAX of the bare and context similarities before threshold/ranking, so a tag can only ever rank at least as well as it would from the bare query alone — irrelevant recent context cannot demote a relevant tag below its bare rank. Guard false scores the context vector alone (plain concat, experimentation).
- **Tests**:
  - `test_embedding_context_guard.py::TestGuardMathParity` (per-tag max ranking matches the reference guard math; guard sim >= bare sim per tag)
  - `test_embedding_context_guard.py::TestLegacyByteIdentical` (N=0 / no-context path identical to legacy; defaults are 0/true)
  - `test_embedding_context_guard.py::TestCraterRegression` (irrelevant-context crater repaired: plain concat demotes the gold tag, guard recovers it near the bare rank)
  - `test_embedding_context_guard.py::TestConfigPlumbing` / `TestRetrieverContextEmbedding` (YAML parse + retriever concat construction)

### BUG-041 — Compaction materializes tag summaries for only the greedy cover

- **Symptom**: Every compaction leaves segment tags with no `tag_summaries` row (~69/day on a live multi-tag conversation); those tags are invisible to the context-hint topic list, absent from the broad/recall-all summary floor, and missing from tag-summary-embedding scoring. Rows materialized by an external repair sweep go permanently stale because later compactions keep skipping the tags. First observed May 30 (tags present in `segment_tags`, absent from `tag_summaries`).
- **Root cause**: By-design minimality that contradicted the system's own read and repair contracts: `_build_tag_summaries` intersected the greedy set-cover with the compacted segments' tags (plus a primary-tag guarantee), structurally omitting every non-primary secondary tag outside the cover. The backfill repair and the read paths both assume every non-`_general` segment tag has a summary.
- **Fix**: Compaction materializes a summary for every non-`_general` tag carried by the just-compacted segments; the primary-tag guarantee is unchanged. The pre-existing staleness check inside `compact_tag_summaries` bounds the LLM cost (fresh summaries skip), and secondary tags now also REFRESH on later compactions instead of going stale.
- **Tests**:
  - `test_tag_summary_materialization.py` — no gap after an organic compaction, `_general` exclusion, report coverage, secondary-tag refresh eligibility on later compactions
  - `test_tag_summary_backfill.py::test_compaction_tag_summaries_populated_when_in_memory_index_empty` (pin updated to the wide contract)

### BUG-040 — Assistant-half ingest duplicates the prepared user row

- **Symptom**: Three linked production symptoms on prepare-then-ingest REST conversations: (1) strict tagging fails with "could not map payload messages to existing rows for logical turn N" and turns fall through to the context-free row sweep (observed as a jump in `_general` primary tags on live traffic), (2) canonical rows contain duplicated user halves and mid-inserted copies of already-present content, (3) sort-key gaps at the insertion point halve with every prepare, priming the BUG-036 gap-exhaustion collisions.
- **Root cause**: `ingest_single` (the completed-pair persist path) prepares a `[user, assistant]` row pair, but the user half is normally already the conversation's LAST row, persisted by the preceding payload prepare. A 2-row incoming fragment has no ≥3-row anchor window and short-overlap matching is disallowed on this path, so alignment failed and `no_overlap_append` duplicated the user row; the duplicate scrambled every subsequent payload alignment.
- **Fix**: Before falling through to full alignment, `ingest_single` matches the pair's user hash against the tail row; on a match it mirrors the existing row's identity (no rewrite) and appends only the assistant row (`tail_append`). Resend dedup and genuinely-new-pair append behavior are unchanged.
- **Tests**: `test_ingest_single_tail_pair.py` — no duplication, multi-turn row cleanliness, strict tagging green across a full REST conversation, resend dedup preserved, unprepared-pair append preserved.

### BUG-039 — engine_state saves fail on schema-bootstrapped Postgres

- **Symptom**: `Failed to save engine state: column "flushed_prefix_messages" of relation "engine_state" does not exist` on every engine-state save; swallowed as a warning, so session-restore state silently never persists.
- **Root cause**: The bundled schema created `engine_state` with five columns while `save_engine_state` on Postgres INSERTs seven (`flushed_prefix_messages`, `last_request_time`).
- **Fix**: Both columns added to the bundled CREATE TABLE, plus idempotent `ADD COLUMN IF NOT EXISTS` migration in the schema bootstrap for tables created by earlier schemas.
- **Tests**:
  - `test_engine_state_schema_postgres.py::test_engine_state_table_has_save_path_columns`
  - `test_engine_state_schema_postgres.py::test_save_engine_state_round_trips_on_fresh_schema`

### BUG-037 — Schema bootstrap DDL races across workers

- **Symptom**: Multi-worker startup logs `tuple concurrently updated` from `CREATE OR REPLACE FUNCTION` (and historically DuplicateObject from trigger DROP+CREATE pairs); the losing worker logs "merge_post_commit_pending bootstrap failed" and skips the rest of its guarded block.
- **Root cause**: `PostgresStore._ensure_schema` ran all bootstrap DDL with no cross-worker serialization.
- **Fix**: Session-scoped `pg_advisory_lock` held on a dedicated connection for the whole bootstrap (`_SCHEMA_BOOTSTRAP_LOCK_KEY`); session-scoped rather than xact-scoped because the bootstrap relies on per-statement autocommit for its try/except-guarded statements.
- **Tests**:
  - `test_schema_bootstrap_postgres.py::TestSchemaBootstrapSerialization::test_concurrent_bootstrap_no_ddl_race_warnings`

---

## By Test File

| Test File | Bugs Covered |
|-----------|-------------|
| `test_headless.py` | BUG-001 |
| `test_live_turn_restore.py` | BUG-090 |
| `test_speaker_rows_lean.py` | BUG-091 |
| `test_session_tombstone_probe.py` | BUG-092 |
| `test_provider_request_timeout.py` | BUG-093 |
| `test_history_catchup.py` | BUG-094 |
| `test_pool_fill_measurement.py` | BUG-095 |
| `test_curation_memo.py` | BUG-096 |
| `test_vc_tools_fixed_catalogue.py` | PROXY-042 |
| `test_hint_search_rule.py` | BUG-121 |
| `test_postgres_schema_once.py` | BUG-122 |
| `test_token_counter_shared.py` | BUG-123 |
| `test_regroup_writes_only_changes.py` | BUG-124 |
| `test_retrieve_facts_in_parallel.py` | BUG-125 |
| `test_responses_custom_tool_outputs.py` | BUG-098, BUG-099 |
| `test_responses_context_in_tool_loop.py` | BUG-100 |
| `test_flush_gate_turn_memo.py` | BUG-101 |
| `test_protected_zone_media_size.py` | BUG-102 |
| `test_drop_compacted_in_tool_loop.py` | BUG-103 |
| `test_protected_window_host_speaker_dedup.py` | BUG-104 |
| `test_filter_keeps_paired_over_unanswered.py` | BUG-105 |
| `test_chain_restore_format.py` | BUG-106 |
| `test_fact_topic_pointer.py` | BUG-108 |
| `test_chat_completions_paging_tools.py` | BUG-109 |
| `test_mcp_conversation_scope.py` | BUG-110 |
| `test_refused_supersession.py` | BUG-111 |
| `test_fact_audience.py` | BUG-112 |
| `test_component_models.py` | BUG-113 |
| `test_paging_model_resolution.py` | BUG-114 |
| `proxy/test_dashboard_token_reads.py` | BUG-115 |
| `test_stream_error_logged.py` | PROXY-041 |
| `test_prompt_cache_key.py` | PROXY-040 |
| `test_request_log_no_session_dump.py` | PROXY-039 |
| `test_model_limits.py` | BUG-117 |
| `test_responses_catalog_tools.py` | BUG-116 |
| `test_session_state_version_roundtrip.py` | BUG-089 |
| `test_tag_index_restore_without_state.py` | BUG-088 |
| `test_embedding_model_device.py` | BUG-087 |
| `test_tag_select_seam.py` | BUG-086 |
| `test_jev_fact_curation_batches.py` | BUG-083 |
| `test_fact_dense_search.py` | BUG-082 |
| `test_fact_dense_search_postgres.py` | BUG-082 |
| `test_segment_summary_chunk.py` | BUG-081 |
| `test_orphaned_compaction_operation.py` | BUG-080 |
| `test_orphaned_compaction_operation_postgres.py` | BUG-080 |
| `test_embedding_memory_shape.py` | PROXY-038 |
| `test_embedding_memory_shape.py` | PROXY-038, BUG-128 |
| `test_tag_summary_embedding_snapshot_freshness.py` | PROXY-037 |
| `test_completion_audience_stamp.py` | PROXY-036 |
| `test_out_of_band_route_audience.py` | PROXY-034 |
| `test_one_turn_per_message.py` | PROXY-033 |
| `test_noop_ingest_skips_whole_conversation_passes.py` | PROXY-032 |
| `test_retrieval_context_is_worker_independent.py` | PROXY-031 |
| `test_context_hint_no_inline_rebuild.py` | PROXY-030 |
| `test_budget_enforcement_order.py` | PROXY-029 |
| `test_current_attachments_preserved.py` | PROXY-028 |
| `test_context_cache_prefix.py` | PROXY-027 |
| `test_responses_context_placement.py` | PROXY-027 |
| `test_chat_context_placement.py` | PROXY-027 |
| `test_host_replay_expansion.py` | PROXY-026, BUG-118, BUG-119 |
| `test_unreplied_turns_compaction.py` | BUG-079 |
| `test_postgres_store.py` | BUG-078 |
| `test_tui.py` | BUG-002 |
| `test_compactor.py` | BUG-003, BUG-004 |
| `test_recall_all.py` | (replaces BUG-007 broad detection) |
| `test_idf_retrieval.py` | BUG-005, BUG-006 |
| `test_retriever.py` | BUG-008, BUG-009 |
| `test_context_bleed.py` | BUG-010 |
| `test_turn_tag_index.py` | PROXY-002 |
| `test_proxy.py` | BUG-008, PROXY-001, PROXY-002, PROXY-003, PROXY-004, PROXY-004b, PROXY-004c, PROXY-005, PROXY-010, PROXY-013, PROXY-014, PROXY-015, PROXY-022, PROXY-023, PROXY-025 |
| `test_paging.py` | PROXY-024 |
| `test_monitor.py` | PROXY-021 |
| `test_empty_turn_skip.py` | BUG-013 |
| `test_tag_splitter.py` | BUG-011, BUG-012 |
| `test_find_quote.py` | BUG-029, BUG-031 |
| `test_verb_expansion.py` | BUG-032 |
| `test_engine_integration.py` | BUG-034 |
| `test_conversation_scoping.py` | BUG-035 |
| `test_ingest_sort_key_rebalance.py` | BUG-036 |
| `test_sort_key_shift_postgres.py` | BUG-036 |
| `test_schema_bootstrap_postgres.py` | BUG-037 |
| `test_strict_tagging_tagged_rows.py` | BUG-038 |
| `test_engine_state_schema_postgres.py` | BUG-039 |
| `test_ingest_single_tail_pair.py` | BUG-040, BUG-056 |
| `test_rest_phase_lifecycle.py` | BUG-057 |
| `test_assistant_audience_stamping.py` | BUG-058 |
| `test_benchmark_retrieval_authority.py` | BUG-059 |
| `test_prepare_user_message_strip.py` | BUG-060 |
| `test_history_widening_guard.py` | BUG-061 |
| `test_actor_card_admission_quality.py` | BUG-063 |
| `test_actor_card_style_evidence.py` | BUG-069 |
| `test_actor_card_resource_preferences.py` | BUG-071 |
| `test_card_availability_and_adjudication.py` | BUG-064, BUG-065 |
| `test_render_escape_host_attribution.py` | BUG-066, BUG-067 |
| `test_tag_rollup_recovery.py` | BUG-126 |
| `test_canonical_read_projection.py` | BUG-127, BUG-129 |
| `test_request_reads_without_bodies.py` | BUG-130 |
| `test_request_reads_without_bodies_postgres.py` | BUG-130 |
| `test_raw_body_storage.py` | BUG-131 |
| `test_raw_body_storage_postgres.py` | BUG-131 |
| `test_tag_summary_materialization.py` | BUG-041 |
| `test_embedding_context_guard.py` | BUG-042 |
| `test_embedding_reserved_seats.py` | BUG-043 |
| `test_ingest_anchor_incremental.py` | BUG-047 |
| `test_ingest_projected_rows.py` | BUG-048 |
| `test_speaker_scoping_refusal.py` | BUG-049 |
| `test_actor_cards.py` | BUG-050 |
| `test_protected_window_gate.py` | BUG-044, BUG-045, BUG-046 |
| `test_canonical_turn_id_stamping.py` | BUG-046 |
| `test_retrieval_assembler_protected_window_merge.py` | BUG-045 |
| `test_sqlite_mirror_store_methods.py` | BUG-045 |
| `test_postgres_mirror_store_methods.py` | BUG-045 |
| `test_rebuild_log_rejection_format.py` | BUG-051 |
| `test_ingestion_watermark_atomicity.py` | BUG-053 |
| `test_facts_block_budget.py` | BUG-052 |
| `test_agent_authored_quote_guard.py` | BUG-054 |
| `test_temporal_status_cessation.py` | BUG-055 |
| `test_bot_outbound_message_ledger_postgres.py` | BUG-054 |
| `test_lifecycle_epoch_start_postgres.py` | BUG-054 |

### BUG-051 — rejection counts break the JSON log envelope

- **Symptom**: `ACTOR_CARD_REBUILD` log lines carrying a non-empty rejection map are not valid JSON. Any JSON-based analysis of rebuild outcomes silently drops them, and those are exactly the lines that record rejections; lines with empty maps parse cleanly, so the reader looks correct while discarding every interesting row.
- **Root cause**: the rejection map was rendered with `json.dumps` and interpolated into a log message that downstream shipping wraps in JSON, placing unescaped double quotes inside that message.
- **Fix**: render the counts as sorted, comma-joined `reason:count` pairs in the same quote-free key=value idiom as the rest of the line, with `-` for an empty map so the field is never blank.
- **Tests**:
  - `test_rebuild_log_rejection_format.py`

### BUG-053 — the widening watermark's hash and count describe different observations

- **Symptom**: a conversation can reach a state where `_detect_and_reset_widened_history` has a recorded head hash but a turn count of zero. `new_turns > old_turns * (1 + threshold)` is then true for any count, so the growth threshold is inert and a single head-hash comparison is all that stands between an ordinary request and `delete_conversation`, which purges `canonical_turns`.
- **Root cause**: `_ingested_first_hash` and `_ingested_turn_count` were written non-atomically at two sites, and at both the count was written without the hash. In `_record_ingestion_watermark` the hash write was conditional on the buffer carrying a completed turn group while the count write was unconditional; the no-history early return in `resolve_prepare_state` wrote the count alone.
- **Fix**: the recorder gates both writes on the completed-group count and returns early when it is zero, so the two fields always describe one buffer. The gate is the count rather than the head text because those predicates diverge: a leading assistant message pairs into a completed group by itself, so a buffer can carry a positive count and empty head text, and gating on the head text would skip it and leave a smaller count from an older buffer in place. Both values are computed before either is stored, so a raising helper cannot leave a hash without a count, and the hash is stored first so a reader interleaving between the two statements sees a fresh hash that matches the current head and returns at the detector's equality check. The no-history path writes nothing. A buffer with no completed group preserves the previous baseline rather than clearing it, since clearing would retire widening detection for the single-message buffers that dominate traffic.
- **Tests**:
  - `test_ingestion_watermark_atomicity.py`

### BUG-052 — the facts block ships over the budget its selection enforced

- **Symptom**: the rendered `<facts>` block is larger than the token budget the greedy pool fill charged itself for it. `conversation_budget` subtracts the block's actual size, so the overrun is taken from the conversation window rather than from the facts allocation.
- **Root cause**: selection charged a sum of per-line `token_counter` results while the block delivered is a single joined string. Two costs were never charged: the newline separators the renderer joins lines on, and the XML wrapper. A third accumulated silently, since an estimator that divides characters truncates each line's remainder independently, so the shortfall grew with every line admitted. `_format_facts` repeated the same per-line arithmetic against a `+100` slack, so it could not correct the total either.
- **Fix**: the fill now charges what the assembled block costs with each candidate line in it, measured through the same `_facts_block` shape the renderer emits, so the running total is the block's real size at every step. This is affordable because the block never grows past its cap. `_format_facts` measures the assembled block too, finding the admitted prefix by bisection since adding a line to a newline-joined block cannot reduce its token count, and it reports how many lines it admitted so the caller can drop any it could not hold from the selection record. `retrieval_metadata["facts_block"]` carries the selected, rendered and trimmed counts, so a block that drops selected lines is visible rather than silent.
- **Tests**:
  - `test_facts_block_budget.py`

### BUG-054 — a reply quoting the agent is filed as a person's disclosure

- **Symptom**: quoted text that the agent itself wrote is recorded as a claim made by the person being replied to. The agent can then tell a member it holds a disclosure that member never made.
- **Root cause**: `_build_actor_roster` resolves a quote-reply against the transport messages it holds and builds `FactLane(role=AUTHOR_ROLE_SUBJECT)` when the quoted message is not among them. The agent's own outbound message ids were stored nowhere, so a reply quoting the agent could never resolve and always took that branch.
- **Fix**: `bot_outbound_messages` records the identities the agent authored, keyed on platform, account, channel and message so ids from unrelated platforms cannot collide. The guard consults it and suppresses the subject lane only on an exact match, built from the namespace `resolve_channel_namespace` reports for the channel so the reader and the writer use the same values. Absence, an ambiguous channel, a backend without the ledger and a failing lookup all mean unknown and keep the previous behaviour, because the recorded set is always partial and treating a gap in it as evidence would delete a real disclosure. Two lifecycle fences stop an identity speaking for a later incarnation of a conversation: a write is declined when the observation predates `lifecycle_epoch_started_at`, and a read requires the row's epoch to match the conversation's current one.
- **Known limit**: a reply delivered as several platform messages yields at most one recorded id, so the rest stay unknown and are unaffected.
- **Tests**:
  - `test_agent_authored_quote_guard.py`
  - `test_bot_outbound_message_ledger_postgres.py`
  - `test_lifecycle_epoch_start_postgres.py`

### BUG-055 — a stopped state is tokenised identically to a finished action

- **Symptom**: a fact recording that someone stopped an ongoing regimen carries the same `status` value as an inventory tally or a completed purchase, so the record that would correct a false "still doing X" reading is indistinguishable from a stock count. Separately, a status the parser did not recognise was coerced to `active`, turning an unclassifiable value into the strongest available assertion that the state continues.
- **Root cause**: `TemporalStatus` had no token for a state that ended, so `completed` absorbed both meanings, and the extraction prompts named the five tokens without defining any of them. Both parse sites then applied `if status not in valid_statuses: status = "active"`.
- **Fix**: adds `ceased` beside an unchanged `completed`, so no stored row changes meaning and only re-derivation reclassifies them. Both prompts now define every token, and state explicitly that a course run to its intended length is `completed` and not `ceased`, and that `ceased` asserts no reason for the stopping. `normalize_temporal_status` maps common synonyms to their token before falling back, since an unrecognised `ongoing` was previously coerced to `active` correctly and a bare fallback would have lost that; only a genuinely unmappable value becomes unset. It returns why it resolved as it did, and the parse sites log it, because the old fallback wrote a valid-looking `active` and the rate was therefore unmeasurable from stored rows.
- **Known limit**: `active` and unset render identically, so this corrects the stored record and any status-aware consumer, not what the model is shown. A hedged marker was rejected deliberately, since hedged provenance markers were measured to be ignored while definite ones are obeyed.
- **Tests**:
  - `test_temporal_status_cessation.py`

### BUG-056 — carrier-wrapped user text defeats the completed-pair tail anchor

- **Symptom**: on the prepare-then-ingest flow, a turn whose user entry is a host-assembled quoted-reference carrier persists two user rows: the prepared row holds the stripped current request while the completion write appends a second row carrying the full raw carrier (`merge_mode=no_overlap_append`, `CANONICAL_TURN_NO_ALIGNMENT` warning). The duplicated carrier row is orders of magnitude longer than real speech, matches nearly every retrieval, and scrambles later alignment.
- **Root cause**: `extract_ingestible_messages` strips the quoted-reference carrier at batch admission, but `ingest_single` hashed and persisted the caller's user text verbatim. The two admission surfaces therefore persisted different user bytes for the same logical turn; `compute_turn_hash_from_raw` produced unequal hashes, the tail-hash fast path could never match the prepared row, and alignment fell through to blind append. Any content transform applied on one hash-computing surface but not the other deterministically dual-persists every affected turn.
- **Fix**: `ingest_single` applies the same `strip_quoted_reference_carrier` transform to `user_content` before any row preparation or hash computation, restoring byte identity between the two admission surfaces. A carrier with no bundled request keeps the caller's bytes, because this surface receives exactly one user string and dropping it would orphan the assistant half.
- **Tests**:
  - `test_ingest_single_tail_pair.py::test_carrier_wrapped_ingest_tail_appends_after_stripped_prepare`
  - `test_ingest_single_tail_pair.py::test_both_lanes_persist_identical_user_bytes_for_carrier_turn`
  - `test_ingest_single_tail_pair.py::test_plain_ingest_unaffected_by_carrier_strip`

### BUG-057 — interactive prepare-then-ingest conversations never leave phase='init'

- **Symptom**: a conversation on the single-turn completion flow reports `phase='init'` forever no matter how many fully tagged turns it holds. Every phase-keyed consumer misclassifies it: idle cleanup treats it as never-established, backlog detection and reattribution skip it, and the dashboard shows a conversation that never activates.
- **Root cause**: the `total == done` branch that flips `'init'` to `'active'` in the prepare flow is unreachable on that lane, because every prepare persists a fresh untagged user half before the branch evaluates, so the phase decision always sees `total > done` with an incomplete physical group and returns without transitioning. Tagging then completes in the post-ingest path, whose self-heal only covers `phase == 'ingesting'` (there is an episode to finalize); the init lane never claims an episode, so no code path ever advanced the phase.
- **Fix**: phase is derived from retrievable content. `_activate_init_phase_if_tagged` flips `'init'` to `'active'` (epoch-guarded, idempotent, soft-failing) whenever the conversation holds at least one fully tagged group, invoked from the post-tag path beside the episode finalizer and as a self-heal in the prepare flow's `total > done` branch before the group claim. Conversations with no tagged content stay `'init'`; the empty-conversation flip and the `'ingesting'` self-heal are untouched.
- **Tests**:
  - `test_rest_phase_lifecycle.py::test_one_turn_conversation_activates_after_prepare_ingest_tag`
  - `test_rest_phase_lifecycle.py::test_next_prepare_activates_stuck_init_conversation`
  - `test_rest_phase_lifecycle.py::test_first_prepare_alone_keeps_init`

### BUG-058 — assistant rows carry no audience and are invisible to audience-scoped reads

- **Symptom**: every assistant canonical row stores an empty `audience_conversation_id` with `audience_attribution_version=0`, so audience-scoped candidate admission (which requires an exact audience and version match) excludes the assistant half of every conversation categorically. Quote search over a proved route returns only user speech; assistant-authored content is unreachable however relevant.
- **Root cause**: audience provenance rides the reply-edge struct, and the reply edge is speaker attribution, so both admission surfaces gave assistant rows the empty edge (`ingest_batch` role branch, `ingest_single` assistant row preparation). The audience, unlike the reply lanes, is a property of the request's proved route shared by both halves of the turn. No repair surface existed: the reply-roles backfill skips rows without user content.
- **Fix**: an audience-only edge (audience fields set from the pair's proved audience, every speaker-attribution field empty) is stamped on assistant rows at both admission surfaces; an unproved route keeps the empty edge. `backfill_assistant_audience` (engine method + `admin backfill-assistant-audience` CLI, dry-run by default with `--apply`) repairs existing assistant rows from the sibling user row's proved audience via the one-way-fill reply CAS: idempotent, epoch-guarded, and skipping groups whose stamped siblings disagree.
- **Tests**:
  - `test_assistant_audience_stamping.py` (both surfaces, proved and unproved, backfill dry-run/apply/idempotence, unstamped-sibling skip)
  - `test_ingest_audience_attribution.py::test_completed_turn_persist_stamps_proved_audience` (superseded role-local audience pin updated; reply lanes remain role-local)

### BUG-060 — request-derived inputs see the raw quoted-reference carrier

- **Symptom**: for a carrier-wrapped request, tagging and retrieval key on kilobytes of quoted scaffolding instead of the user's own words, the in-memory history tail stores different bytes than the canonical row, and the active-user roles guard logs "active user metadata mismatch" and silently disables actor-card selection and audience derivation for the request.
- **Root cause**: the request flow's `user_message` was the raw payload extraction, while the admission path strips the quoted-reference carrier from ingestible entries. Every consumer of `user_message` therefore diverged from admitted content, and the roles guard's byte comparison against the stripped active entry could never match on a carrier turn.
- **Fix**: `derived_user_message` returns the carrier's bundled current request (same recognizer and extraction as the admission strip) and the raw text otherwise; the request flow derives `user_message` through it. The outbound payload is untouched — the model still receives the full carrier. A carrier with no bundled request keeps the caller's bytes, matching the admission surfaces.
- **Tests**:
  - `test_prepare_user_message_strip.py::test_derived_user_message_strips_bundled_carrier`
  - `test_prepare_user_message_strip.py::test_derived_user_message_keeps_unbundled_and_plain_text`
  - `test_prepare_user_message_strip.py::test_roles_guard_matches_on_stripped_carrier_request`

### BUG-059 — benchmark harness sends retrieval tool calls with no request authority

- **Symptom**: every quote-search tool call issued during a benchmark run returns "No conversation search was performed because request retrieval authority is unproved" — the reader is told nothing is on record while the content sits in storage.
- **Root cause**: model-facing retrieval fails closed by design: the tool runtime coerces an absent `speaker_context` to the ineligible sentinel and quote search refuses it. The LongMemEval harness called `query_with_tools` with no context at all, so the fail-closed boundary refused every benchmark retrieval.
- **Fix**: `benchmark_speaker_context` constructs the owner-routed DM-shaped authority for the harness's single-conversation store (audience is the conversation itself, empty channel, exact-match channel scope) and the harness passes it on every reader call. Candidate admission additionally requires audience-stamped source rows; existing cached stores need the assistant-audience backfill before assistant-authored content admits.
- **Tests**:
  - `test_benchmark_retrieval_authority.py::test_benchmark_context_passes_quote_search_request_gate`
  - `test_benchmark_retrieval_authority.py::test_benchmark_context_is_scoped_to_its_own_conversation`
  - `test_benchmark_retrieval_authority.py::test_ineligible_sentinel_still_fails_the_gate`

### BUG-061 — the widening reset destroys history the payload cannot rebuild

- **Symptom**: on the single-turn completion flow, a prepare whose window starts at a different first turn and holds more pairs than the worker-local baseline trips the widening reset, which purges every table for the conversation (`HISTORY_WIDENED` then `delete_conversation`) — including turns the incoming window does not contain. A newest-turns-only client loses its whole durable history; the subsequent rebuild is refused by strict-tagging admission, leaving zero rows.
- **Root cause**: the reset's premise is that the incoming payload is the client's authoritative full history, so a full re-ingest can rebuild everything the purge destroys. On the single-turn completion lane the payload is a working window, not a history assertion, and the trigger compared against worker-local in-memory counters that undercount the durable store — the first-turn hash trivially differs and the growth ratio is trivially satisfied for any windowed client.
- **Fix**: before destroying anything, the reset compares the payload's ingestible entry count against the DURABLE canonical count from the progress snapshot — never the worker-local counters. A smaller payload suppresses the reset with a `HISTORY_WIDENING_SUPPRESSED` warning carrying both counts; a payload at least as large as the durable record keeps the existing reset behavior, so authoritative proxy-lane widening is unchanged.
- **Known deviation, out of scope here**: the reset still purges via a direct `delete_conversation` call rather than the sanctioned deletion path; recorded for the payload-shape class review.
- **Tests**:
  - `test_history_widening_guard.py::test_smaller_payload_never_resets_durable_history`
  - `test_history_widening_guard.py::test_genuine_widening_still_resets`

### BUG-062 — admission mints single-half groups the tagging CAS refuses

- **Symptom**: a payload holding a user message followed by two consecutive assistant messages persists three rows at prepare, then strict follow-up tagging raises "strict canonical tagging could not map payload messages to existing rows for logical turn 1" and the ingestion episode wedges with the rows untagged.
- **Root cause**: batch admission assigns a consecutive same-role payload entry its own continuation group holding one row with exactly one half, but `update_canonical_group_tagging_if_unchanged`'s shape gate accepted only a single combined row or a user+assistant pair. The system contradicted itself: admission created a durable group shape the tagging write path categorically refused, so the strict mapper's proven mapping failed closed at the store.
- **Fix**: the shape gate in both backends gains the single-half arm — one row carrying exactly one half — matching what admission itself mints. The strict payload mapper still proves per-message hash identity, group agreement, and role shape before the CAS runs; the gate remains the defense-in-depth check on group coherence.
- **Tests**:
  - `test_handle_prepare_payload.py::test_proxy_ingest_history_keeps_total_fixed_for_single_prepare_payload` (the pre-existing pin, assertions unchanged)

### BUG-063 — card curation admits banter, requests, and third-party asks as durable goals

- **Symptom**: live guild cards carried an `active_goal` "Plans to start using Tren." at confidence 1.00 sourced from a joking exchange, standing goals minted from single questions and one-shot service requests ("give me a leg workout for today" at 0.90), and a question about a third party stored as the asker's own goal at 0.95.
- **Root cause**: no prompt surface (curation, shared semantic contract, admission) carried any register/sincerity rule, so literal entailment admitted jokes; the explicit-durability/recurrence bar named only communication_pref and interaction_style, leaving active_goal and relevant_history with no recurrence, stated-intent, or question-is-not-intent requirement; active_goal's definition never required FIRST-PERSON intent, so asking on a third party's behalf read as pursuit; and confidence was the curator model's uncalibrated assertion, validated for range only and ratcheted upward on duplicate merge.
- **Fix**: a shared judgment-rules block appended to both the curation and admission prompts (register/sincerity test with a serious-corroboration escape hatch; a question or single imperative is never an active_goal, with transience markers decisive; active_goal restricted to the author's first-person intent, third-party subjects excluded absent their own cited utterance; the durability bar extended to every kind), a calibrated confidence scale added to the curation prompt, and a code-enforced cap: an entry citing exactly one source is stored at no more than 0.8 regardless of the asserted value. Duplicate-merge max() is unchanged.
- **Tests**:
  - `test_actor_card_admission_quality.py` (prompt-contract pins on both surfaces, confidence-scale pin, single-source cap on fresh AND carried-over entries, multi-source uncapped guard)

### BUG-064 — active actors are cardless: enrichment invalidation and a terminal coverage gate

- **Symptom**: the guild's most active members carry `card_invalid=1` almost continuously and serving fails closed to nothing, while rebuild attempts end `written=0` with `failure_count=3` on the first coverage failure — permanently cardless members with repeated model spend.
- **Root cause**: the canonical_turns UPDATE trigger's authorship arms set `card_invalid` on a change to ANY of the author's rows, so provenance enrichment one-way fills (actor CAS upgrades, audience stamping, alignment updates) on uncited rows invalidated continuously; `get_actor_card` returns None on invalid; the coverage gate hard-failed a substantive actor whose entries the hardened admission gate correctly rejects; the status recorder jumped coverage outcomes straight to the terminal failure count; and the invalid flag clears only on a successful write.
- **Fix**: invalidation is citation-scoped — the authorship arms of the UPDATE and DELETE triggers (both backends) set `card_dirty` only, the citation arms keep dirty+invalid, and read-time source re-verification remains; a substantive actor with zero durable entries now writes the (possibly empty) card as outcome `no_durable_entries`, clearing both flags; coverage outcomes increment failure counts normally in BOTH status recorders (the Postgres twin initially kept the instant `failure_count=3` jump — writer-scan miss, fixed) and are excluded from terminal suppression while keeping timed retry backoff; an unadjudicable coverage disagreement (the fallback already served the admission call, or no fallback exists) resolves to the admission verdict and writes the card instead of raising.
- **Tests**:
  - `test_card_availability_and_adjudication.py` (uncited enrichment/delete keep serving; cited update/delete still invalidate; empty-card success clears flags with zero failure count; coverage increments normally; unadjudicable disagreement resolves conservatively; three coverage failures do not terminally suppress the next attempt)
  - `test_actor_cards.py::test_semantic_admission_rejects_candidate_without_rewriting_card` (superseded instant-terminal pin updated to the empty-card success)
  - `test_actor_cards.py::test_malformed_primary_fallback_is_not_called_twice_on_disagreement` / `test_fallback_supplied_initial_judgment_is_not_counted_twice` (fallback-supplied judgment is never re-consulted; resolution is conservative, not a failure)
  - `test_actor_cards_postgres.py::test_pg_coverage_disagreement_increments_instead_of_terminal_jump` (Postgres recorder parity)

### BUG-065 — refused agent-directed requests admitted as card preferences

- **Symptom**: a member's card carried "not use a safety shield", "assume they know all the risks of Tren and don't mention them", and a harmful-advocacy persona as high-confidence communication preferences — requests the agent refused live, deferring to the authority holder.
- **Root cause**: the curation and admission inputs were actor-authored only, so the judge structurally could not see the agent's refusal, and the semantic contract's instruction-recasting rule routed agent-directed requests into communication_pref regardless of the live outcome.
- **Fix**: each prompt message now carries the paired agent reply (bounded, keyed by turn group) on both surfaces, fresh and carryover; the judgment rules make the agent's live adjudication the admission signal — honored requests may be preferences, refused or deferred requests reject as `agent_refused`, a behavior-change request with no visible honored signal rejects the same way, and safety-posture requests reject as `safety_posture_request` for any actor; both reasons join the validated enum, and the policy version bump re-judges existing carried-over entries on the next rebuild.
- **Tests**:
  - `test_card_availability_and_adjudication.py` (reply plumbing on both surfaces, reject-only token acceptance, judgment-rule pins)

### BUG-066 — host-attribution lookalikes render verbatim into model-facing context

- **Symptom**: a participant who types a lookalike of a trusted attribution wrapper (`<message-speaker ...>`, `<current-speaker ...>`, `<vc-prepared-context ...>`) gets it back verbatim inside rendered context — assembled prepend, tool results, MCP responses — where downstream consumers treat such wrappers as trusted host metadata, so one participant can forge another's attribution.
- **Root cause**: stored conversation content is exact-source by design and no model-facing render boundary escaped the host-attribution tag set; the message lane must stay byte-exact for turn-hash alignment, but rendered egress had no equivalent constraint and simply inherited exactness.
- **Fix**: `core/render_escape.py` escapes the leading `<` of the recognized tag set to the literal `\u003c` characters, idempotently, in a plain-text form and a serialized-JSON form (doubled backslash so a decode still carries the escape); applied at the assembler's prepend composition, at the `execute_vc_tool` result boundary, and via a decorator on every MCP tool/resource/prompt. Never applied at ingest, in storage, or in the message lane.
- **Tests**:
  - `test_render_escape_host_attribution.py` (prepend and tool-result egress escape parse-stably; message lane stays byte-exact; all five tags case-insensitively; idempotence and plain/serialized composability; MCP decorator behavior and a source lint pinning it on every registration)

### BUG-067 — fact prompt lines render raw markup; emitted wrapper set undeclared

- **Symptom**: `Fact.format_for_prompt` composed extracted fields (subject, verb, object, what, and the rest) with no angle-bracket escaping, so member text quoted into a fact field could open a forged wrapper lookalike wherever fact lines render into model-facing text; separately, the set of wrapper tags the render modules emit was undeclared, so a new emission could appear with unaudited insertion lanes.
- **Root cause**: fact fields are extracted from conversation content but the prompt rendering treated them as plain prose, unlike every other member-text lane, which escapes brackets at insertion; no test pinned the emitted-tag inventory, and manual inventories repeatedly undercounted it (five, then six, against nine found by scan).
- **Fix**: the composed fact line escapes angle brackets to literal `\u003c` / `\u003e` at the prompt boundary only — stored fields and `embed_text` stay byte-exact so existing embeddings do not drift; `ENGINE_EMITTED_TAGS` declares the nine-tag closed set in `core/render_escape.py`, with an AST-based lint test asserting the render modules emit exactly that set.
- **Tests**:
  - `test_render_escape_host_attribution.py::test_fact_prompt_line_escapes_engine_tag_lookalikes` (escape at render; embed text raw)
  - `test_render_escape_host_attribution.py::test_emitted_wrapper_tag_set_is_closed_and_declared` (closed-set lint, drift fails both directions)

### BUG-068 — actor card style observations become repeated speech habits

- **Symptom**: an interaction-style observation turns into the same address term on consecutive replies.
- **Cause**: the influence-only wrapper did not explain the distinct purposes of preferences, interaction style, goals and history.
- **Fix**: constant orientation before the entries lets style guide register naturally while forbidding quoted entries and per-message verbal tics. Preferences direct the response; goals/history guide relevance and depth. JSON scalar bodies and angle-bracket escaping remain unchanged.
- **Tests**:
  - `test_actor_card_assembly.py::test_actor_card_orientation_precedes_entries`
  - `test_actor_card_assembly.py::test_actor_card_render_golden`
  - `test_actor_card_assembly.py::test_card_scalars_cannot_escape_the_wrapper`

### BUG-069 — a single phrase becomes high-confidence recurring actor style

- **Symptom**: a card turns one playful utterance into a high-confidence claim of recurring language, and normalizes the actor's exact term. Regression fixtures use synthetic actors and quotations only.
- **Cause**: the semantic contract did not explicitly prohibit turning one phrase into a habitual-frequency claim or normalizing its terms. Confidence counted citation rows, allowing a fact and its own source message, or several facts from one message, to masquerade as repeated evidence.
- **Fix**: policy 17 adds the exact-term and distinct-message rules to both prompts. Preference/style confidence is capped at 0.7 for fewer than two proven distinct messages, including carryovers. Native-message identities deduplicate physical rows; version-two requester facts must resolve to one exact, already-loaded actor message within their segment's explicit source mapping. Unknown, ambiguous, legacy, or quoted-subject fact identities do not prove multiplicity. Only bounded existing source metadata is retained; identity fields enter the evidence fingerprint. Other kinds keep the existing 0.8 single-source cap.
- **Tests**:
  - `test_actor_card_style_evidence.py::test_single_quoted_phrase_cannot_inflate_existing_style_even_if_admission_allows` — synthetic regression exercises the same failure even with permissive independent admission; downgrade preserves the older exact-term entry.
  - `test_actor_card_style_evidence.py::test_style_confidence_counts_distinct_proven_messages` — both affected kinds; single fact/message, duplicate evidence, unknown provenance, segment-neighbor exclusions, and genuine repetition.
  - `test_actor_card_style_evidence.py::test_carryover_style_is_recalibrated_without_rewriting` — immutable carryovers receive the same evidence cap.
  - `test_actor_card_style_evidence.py::test_other_card_kinds_keep_existing_single_source_cap` — goal/history behavior is unchanged.
  - `test_actor_card_style_evidence.py::test_both_prompt_surfaces_forbid_single_utterance_habits_and_preserve_exact_terms` — actual curator/admission prompts carry the rule and retain below-cap confidence.
  - `test_actor_card_style_evidence.py::test_source_identity_change_invalidates_confidence_input_hash` — fact/native message ID corrections and swapped fact-to-segment references force recalibration even without relying on the dirty hint.
  - `test_actor_card_style_evidence.py::test_carryover_is_calibrated_before_confidence_sensitive_admission` — the judge receives the calibrated proposal, preserving a valid entry that its obsolete confidence would otherwise cause it to reject.
