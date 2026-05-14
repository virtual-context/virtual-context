# Overnight assumptions — cross-channel-mirror engine implementation

Branch: `feature/cross-channel-mirror` (off `fix/vcattach-redis-marker-write`).

Decisions made without an explicit spec/plan directive. Each entry names the call and the rationale so the reviewer can override quickly.

## 1. Helper-module placement

Per spec §2.6 the plan picked `virtual_context/core/protected_window.py`. I added the helpers there as specified (`_merge_protected_window`, `_stamp_canonical_turn_ids`, `_last_already_canonical_turn_number`).

Within the merge helper, the `mode == "off"` branch returns `list(payload_history)` (a shallow copy) rather than the input by reference. Mirrors the "side-effect-free" rule from spec §2 even on the no-op path.

## 2. `extract_ingestible_messages` mode parameter — wrapper kwarg

`virtual_context/proxy/helpers.py:_extract_ingestible_messages` gained a kw-only `mode: str = "ingest"` that threads through to the underlying `extract_ingestible_messages`. Existing call sites in `proxy/server.py` (5 sites) and `proxy/state.py` (3 sites) do not pass `mode` and therefore inherit the default. No call-site changes needed.

The `proxy/formats.py:514` internal self-call now explicitly passes `mode="ingest"` for clarity even though it's the default.

## 3. `_decision` helper in `handle_prepare_payload`

The 11 `PhaseDecision(phase=..., started_tagger=...)` return sites inside `handle_prepare_payload` were converted to a local helper `_decision(*, phase, started_tagger)` that closes over the local `canonical_ingest_rows` tuple. Closure uses Python's standard late-binding so the helper reads the current value at call time (after the `ingest_batch` block reassigns it). The original return statements at lines 1230-1445 became `return _decision(...)` calls.

The `_decision` body uses an intermediate variable (`_decision_result = PhaseDecision(...)`) so the `return PhaseDecision(...)` text shape only exists once in the helper itself, which let `Edit(replace_all=True)` migrate the 11 return sites by text match without touching the helper body.

## 4. Stamping placement in `proxy/server.py`

Per plan §2 Step 7, stamping fires in the ACTIVE-path block at `proxy/server.py:1125-1131`, immediately before the active-tail user message is appended to `state.conversation_history`. Gated on `protected_window_db_source == "merge"` and `_phase_decision.canonical_ingest_rows` being non-empty. Wrapped in a defensive try/except so a stamping failure degrades to Tier 3 fall-through rather than failing the request.

Alignment rule (suffix-tail drop on active-tail user) is implemented by calling `_extract_ingestible_messages(body)` + `state._completed_history_messages(...)` and dropping `len(extracted) - len(completed)` rows from the suffix.

## 5. RetrievalAssembler reads `_is_merge_participant` via `getattr(self, ...)`

The Tier 1 cached bool lives on the engine. Rather than add a constructor parameter to `RetrievalAssembler.__init__` (which would force every existing test fixture and `__new__` site to pass it), the engine assigns the bool onto the assembler instance after construction:

```python
self._retrieval._is_merge_participant = self._is_merge_participant
```

The assembler's gate code reads via `getattr(self, "_is_merge_participant", False)`, so tests that bypass engine construction via `RetrievalAssembler.__new__` (`engine.py:2366`) still work — they default to non-participant.

## 6. Tier 2 INT coercion shape

`int(redis_last_turn) == int(anchor_turn)` wrapped in `try/except (TypeError, ValueError)`. A coercion failure (e.g. the marker arrives as a dict instead of an int-coercible scalar) routes to `tier2_legacy_fallthrough` rather than raising. Tests cover both the string-coercible path (`"7"` → 7) and the non-coercible path (dict).

## 7. Engine `__init__` Tier 1 call does NOT catch broad Exception

Per plan §2 Step 8 explicit guidance: silently disabling the feature on a store-wrapper failure / missing allowlist / transient DB outage would hide mirror loss as stale-False. Only Tier 0 `off` short-circuits the call; merge mode propagates any exception from `has_any_alias`. This intentionally fails engine construction on misconfigured deploys.

## 8. Tier 1 propagation across re-init paths

Engine `__init__` sets `_is_merge_participant` once (line 220 area). The post-resurrect re-init path at engine.py:880+ keeps the same `RetrievalAssembler` instance (lines 941-946 mutate fields in place rather than re-constructing), so the assembler's `_is_merge_participant` attribute survives the re-init unchanged. No re-propagation is needed.

The `RetrievalAssembler.__new__` fallback at engine.py:2366 (used by `filter_history` for test stubs) does not exercise `on_message_inbound`, so the gate code is never reached on that path; no Tier 1 propagation is needed there either.

## 9. Observability: dedicated `PROTECTED_WINDOW_GATE` log line

Per plan §2 Step 10 + §5.8, the gate observability is rendered INLINE as key=value substrings in the message text (cloud's stdlib-logging formatter drops `extra=`). I added a dedicated `logger.info("PROTECTED_WINDOW_GATE conv=%s ...", ...)` log line right after the existing INBOUND_BREAKDOWN block in `on_message_inbound`. The new line is gated on `_gate_outcome is not None`, so Tier 0 `off` emits nothing.

The new line is NOT bound by the existing 500ms INBOUND_BREAKDOWN threshold — it fires on every `on_message_inbound` that runs in merge mode, which is the spec §8 observability contract.

`extra=` is included for in-process consumers (tests using `caplog.records[i].my_field` attribute access) but the message text carries the values for grep-based consumption.

## 10. `_gate_budget_evictions` reports 0 in current implementation

The merge helper does NOT evict; the downstream context-builder eviction happens later and isn't currently surfaced via a hook to the gate path. The field is emitted as `0` for now; a future change can wire a real counter from the downstream context-builder if metrics analysis surfaces a need.

## 11. Test files use the existing engine_factory pattern but DO NOT exercise filesystem backend for Tier 3

`tests/test_engine_init_tier1.py` constructs full engines but seeds aliases via the SQLite path (the only backend that hosts canonical_turns). The filesystem backend's `has_any_alias` is tested at the unit level via `test_storage_protocols_cross_channel_mirror.py` (via SimpleNamespace stub) and is not exercised end-to-end on the engine path.

## 12. Postgres tests skip when `VC_TEST_POSTGRES_URL` is not set

`tests/test_postgres_mirror_store_methods.py` mirrors the always-on SQLite suite and uses `pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")`. The CI environment is expected to set this when Postgres is wired; local dev skips cleanly.

## 13. NO code changes to compaction / tagging / fact-curator / vc_* tools

Plan §7 non-goals. Verified by grep — no edits to those modules.

## 14. NO new indexes on conversation_aliases or canonical_turns

Plan §1.2 + §7 non-goals. Both Tier 1 OR legs are already indexed (PK on `alias_id` + `idx_conversation_aliases_target_id`). Tier 3 read uses `idx_canonical_turns_conv_order` on `(conversation_id, sort_key)`.

## 15. NO deploy, NO release, NO merge to main

User explicitly revoked deploy authority tonight. Per spawn-prompt rule, release-class actions are user-direct only. Branch is pushed to `origin/feature/cross-channel-mirror` and stays there. Next step is for the user to review the diff and decide whether to merge `fix/vcattach-redis-marker-write` to main first, then this branch, OR fold both at once OR roll back.

## Open follow-ups (NOT done tonight — explicit non-action)

- Postgres EXPLAIN-plan smoke: I asserted catalog-level index presence in tests rather than `EXPLAIN ANALYZE` plan-node names (per plan §4 risk assessment — query planner choices vary on small test tables).
- End-to-end harness coverage: the `tests/fixtures/target_baseline.sql` already-shipped baseline can be used by the cloud-side harness to exercise the cross-channel-mirror end-to-end. The cloud-side test authoring is owned by cloud per spec §8.
- Spec drift on YAML key (`assembler:` vs actual `assembly:`): plan §3.1 documents this as a follow-up amendment to the engine spec + cloud spec. Implementation uses the correct `assembly:` key.
- `cross-channel-mirror-engine-spec.md` line 261 `cloud-spec §1.1` reference: that section in cloud-spec is currently named `1.4` and `5.5` per plan §3.1. Spec amendment pending.
