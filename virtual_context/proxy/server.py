"""HTTP proxy server for virtual-context LLM enrichment.

Sits between any LLM client and an upstream provider (OpenAI or Anthropic),
transparently injecting <virtual-context> blocks into requests and capturing
assistant responses for on_turn_complete.

Usage:
    virtual-context -c config.yaml proxy --upstream https://api.anthropic.com

This module is the ``create_app`` factory that wires together:
- ``state.py``    — ProxyState, SessionState, _IngestionCancelled
- ``registry.py`` — SessionRegistry
- ``handlers.py`` — _handle_streaming, _handle_non_streaming, _passthrough*
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request

from ..engine import VirtualContextEngine
from ..core.tool_loop import (
    VC_TOOL_NAMES,  # noqa: F401 — re-exported
    vc_tool_definitions,
    is_vc_tool,
    execute_vc_tool,
)
from ..types import Fact, Message, PreparedPayload, SplitResult, StoredSummary  # noqa: F401 — re-exported

from .dashboard import register_dashboard_routes
from .formats import (
    PayloadFormat,  # noqa: F401 — re-exported
    detect_format,
    get_format,
)
from .helpers import (  # noqa: F401 — re-exported for tests
    _VC_PROMPT_MARKER,
    _VC_CONVERSATION_RE,
    _HOP_BY_HOP,
    _last_text_block,
    _strip_vc_prompt,
    _strip_envelope,
    _forward_headers,
    _detect_api_format,
    _extract_conversation_id,
    _strip_conversation_markers,
    _extract_user_message,
    _extract_message_text,
    _extract_history_pairs,
    _inject_context,
    _inject_conversation_marker,
    _extract_delta_text,
    _extract_assistant_text,
    _inject_vc_tools,
    _parse_sse_events,
    _build_continuation_request,
    _emit_text_as_sse,
    _emit_tool_use_as_sse,
    _emit_message_end_sse,
    _emit_text_as_responses_sse,
    _emit_tool_use_as_responses_sse,
    _emit_response_done_sse,
    _dump_session_state,
)
from .message_filter import filter_body_messages as _filter_body_messages  # noqa: F401
from .metrics import ProxyMetrics

# --- Re-exported so existing imports from proxy.server still work ---
from .state import SessionState, _IngestionCancelled, ProxyState  # noqa: F401
from .registry import SessionRegistry  # noqa: F401
from .handlers import (  # noqa: F401
    _passthrough,
    _passthrough_bytes,
    _handle_streaming,
    _handle_non_streaming,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------


def _compute_effective_budget(
    context_window: int,
    system_tokens: int,
    tools_tokens: int,
) -> tuple[int, bool]:
    """Compute effective token budget, auto-promoting if client overhead exceeds window.

    Returns (effective_budget, was_promoted).
    """
    overhead = system_tokens + tools_tokens
    if overhead >= context_window:
        return overhead + 10_000, True
    return context_window, False


def _iso_or_none(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _selection_mode(retrieval_meta: dict) -> str:
    if retrieval_meta.get("summary_floor"):
        return "summary_floor"
    if retrieval_meta.get("fts_fallback"):
        return "fts_fallback"
    if retrieval_meta.get("general_fallback") == "previous_turn":
        return "previous_turn_fallback"
    if retrieval_meta.get("fallback") == "working_set":
        return "working_set_fallback"
    return retrieval_meta.get("strategy", "default")


def _segment_selection_reasons(
    summary: "StoredSummary",
    query_tags: set[str],
    related_tags: set[str],
    retrieval_meta: dict,
) -> tuple[list[str], list[str], list[str]]:
    matched_query_tags = sorted(t for t in summary.tags if t in query_tags)
    related_match_tags = sorted(t for t in summary.tags if t in related_tags and t not in query_tags)
    reasons: list[str] = []
    if matched_query_tags:
        reasons.append("Matched query tags: " + ", ".join(matched_query_tags))
    if related_match_tags:
        reasons.append("Included via related tags: " + ", ".join(related_match_tags))
    if retrieval_meta.get("fts_fallback"):
        reasons.append("Added by text-search fallback when tag overlap was insufficient.")
    if retrieval_meta.get("summary_floor"):
        reasons.append("Included by summary floor because the conversation was already compacted.")
    if retrieval_meta.get("general_fallback") == "previous_turn":
        reasons.append("Used previous-turn tags because the inbound request was too general.")
    if retrieval_meta.get("fallback") == "working_set":
        reasons.append("Used working-set tags because inbound tagging fell back.")
    if not reasons and summary.primary_tag:
        reasons.append(f"Selected from stored context under {summary.primary_tag}.")
    return reasons, matched_query_tags, related_match_tags


def _serialize_recall_segment(
    summary: "StoredSummary",
    *,
    rank: int,
    retrieval_meta: dict,
    retrieval_scores: dict[str, float],
) -> dict:
    query_tags = set(retrieval_meta.get("tags_queried", []) or [])
    related_tags = set(retrieval_meta.get("related_tags_used", []) or [])
    reasons, matched_query_tags, related_match_tags = _segment_selection_reasons(
        summary,
        query_tags,
        related_tags,
        retrieval_meta,
    )
    best_score = max((retrieval_scores.get(tag, 0.0) for tag in summary.tags), default=0.0)
    meta = summary.metadata
    return {
        "ref": summary.ref,
        "primary_tag": summary.primary_tag,
        "tags": list(summary.tags),
        "rank": rank,
        "score": round(best_score, 4),
        "selection_mode": _selection_mode(retrieval_meta),
        "selected_because": reasons,
        "matched_query_tags": matched_query_tags,
        "related_match_tags": related_match_tags,
        "summary": summary.summary,
        "summary_tokens": summary.summary_tokens,
        "full_tokens": summary.full_tokens,
        "turn_count": meta.turn_count if meta else 0,
        "session_date": meta.session_date if meta else "",
        "entities": list(meta.entities) if meta else [],
        "key_decisions": list(meta.key_decisions) if meta else [],
        "action_items": list(meta.action_items) if meta else [],
        "date_references": list(meta.date_references) if meta else [],
        "created_at": _iso_or_none(summary.created_at),
        "start_timestamp": _iso_or_none(summary.start_timestamp),
        "end_timestamp": _iso_or_none(summary.end_timestamp),
    }


def _serialize_recall_fact(
    fact: "Fact",
    *,
    retrieval_meta: dict,
) -> dict:
    query_tags = set(retrieval_meta.get("tags_queried", []) or [])
    related_tags = set(retrieval_meta.get("related_tags_used", []) or [])
    matched_query_tags = sorted(t for t in fact.tags if t in query_tags)
    related_match_tags = sorted(t for t in fact.tags if t in related_tags and t not in query_tags)
    reasons: list[str] = []
    if matched_query_tags:
        reasons.append("Matched query tags: " + ", ".join(matched_query_tags))
    if related_match_tags:
        reasons.append("Included via related tags: " + ", ".join(related_match_tags))
    if retrieval_meta.get("summary_floor"):
        reasons.append("Included by summary floor because the conversation was already compacted.")
    if retrieval_meta.get("fallback") == "working_set":
        reasons.append("Selected from working-set fallback.")
    return {
        "id": fact.id,
        "subject": fact.subject,
        "verb": fact.verb,
        "object": fact.object,
        "status": fact.status,
        "fact_type": fact.fact_type,
        "what": fact.what,
        "who": fact.who,
        "when_date": fact.when_date,
        "where": fact.where,
        "why": fact.why,
        "tags": list(fact.tags),
        "segment_ref": fact.segment_ref,
        "turn_numbers": list(fact.turn_numbers),
        "session_date": fact.session_date,
        "mentioned_at": _iso_or_none(fact.mentioned_at),
        "selection_mode": _selection_mode(retrieval_meta),
        "selected_because": reasons,
        "matched_query_tags": matched_query_tags,
        "related_match_tags": related_match_tags,
    }


# ---------------------------------------------------------------------------
# prepare_payload — extracted from catch_all() for reuse by REST API
# ---------------------------------------------------------------------------

async def prepare_payload(
    body: dict,
    state: "ProxyState | None",
    fmt: "PayloadFormat",
    metrics: "ProxyMetrics",
    *,
    body_bytes: bytes = b"",
    inbound_conversation_id: str = "",
    log_dir: Path | None = None,
    log_prefix: str = "",
) -> PreparedPayload:
    """Enrich a request body with virtual-context, returning a PreparedPayload.

    Encapsulates the passthrough and active enrichment paths previously inline
    in ``catch_all()``.  Pure extraction — identical behaviour to the original.
    """
    import asyncio
    import time

    api_format = fmt.name
    user_message = fmt.extract_user_message(body)
    is_streaming = body.get("stream", False)

    # Resolve upstream context window limit for this model
    from .helpers import (
        _extract_history_pairs,
        _inject_context,
        _inject_vc_tools,
    )
    from ..model_limits import resolve_upstream_limit

    _model_name = body.get("model", "")
    if state:
        state._last_model = _model_name
    try:
        _instance_limit = int(getattr(state, '_instance_upstream_limit', 0)) if state else 0
    except (TypeError, ValueError):
        _instance_limit = 0
    try:
        _global_limit = int(state.engine.config.proxy.upstream_context_limit) if state else 0
    except (TypeError, ValueError, AttributeError):
        _global_limit = 0
    _upstream_limit = resolve_upstream_limit(_model_name, _instance_limit, _global_limit)

    # Ground truth: actual byte-measured inbound token count
    _payload_kb = round(len(body_bytes) / 1024, 1) if body_bytes else 0
    _inbound_bytes = len(body_bytes)
    _inbound_tokens = fmt._count(body_bytes.decode("utf-8", errors="replace")) if body_bytes else 0
    if state:
        state._last_payload_kb = _payload_kb
        state._last_payload_tokens = _inbound_tokens
        if state._initial_payload_kb is None:
            state._initial_payload_kb = _payload_kb
            state._initial_payload_tokens = _inbound_tokens

    # ---------------------------------------------------------------
    # State-aware dispatch: PASSTHROUGH/INGESTING vs ACTIVE
    # ---------------------------------------------------------------
    if state:
        state._total_requests += 1
        current_state = state.session_state

        # Fresh session starts ACTIVE but may need ingestion — check and
        # redirect to passthrough path if there's history to ingest.
        if (
            current_state == SessionState.ACTIVE
            and state.engine.config.conversation_id not in state._ingested_conversations
        ):
            history_pairs = _extract_history_pairs(body)
            needed = len(history_pairs) // 2
            existing = state._indexed_turn_count()
            if state.reconcile_history_bootstrap(history_pairs):
                current_state = SessionState.ACTIVE
            elif state.has_pending_indexing():
                state.resume_pending_ingestion_if_needed()
                current_state = state.session_state
            elif needed > 0 and existing < needed:
                current_state = SessionState.PASSTHROUGH

        if current_state in (SessionState.PASSTHROUGH, SessionState.INGESTING):
            # Store latest body for catch-up loop
            state._latest_body = body

            # On first request: kick off non-blocking ingestion
            if not state._history_ingested():
                history_pairs = _extract_history_pairs(body)
                if history_pairs:
                    state.conversation_history = list(history_pairs)
                await asyncio.to_thread(
                    state.start_ingestion_if_needed, history_pairs,
                )

            state.conversation_history.append(
                Message(role="user", content=user_message,
                        timestamp=datetime.now(timezone.utc),
                        raw_content=fmt.extract_user_raw_content(body))
            )

            _conversation_id = state.engine.config.conversation_id
            turn = len(state.engine._turn_tag_index.entries)
            _turn_id = uuid.uuid4().hex[:12]

            # Passthrough payload trimming — trim to upstream_limit * ratio
            _pt_ratio = state.engine.config.proxy.passthrough_trim_ratio if state else 0.40
            _pt_limit = int(_upstream_limit * _pt_ratio) if _pt_ratio > 0 else _upstream_limit
            if _inbound_tokens > _pt_limit:
                from .message_filter import trim_to_upstream_limit
                body, _pt_trimmed = trim_to_upstream_limit(body, _pt_limit, fmt)
                if _pt_trimmed:
                    logger.info(
                        "PASSTHROUGH_TRIM: payload=%dt trimmed to %dt (ratio=%.0f%%, upstream=%dt)",
                        _inbound_tokens, _pt_limit, _pt_ratio * 100, _upstream_limit,
                    )
                    metrics.record({
                        "type": "upstream_trim",
                        "path": "passthrough",
                        "original_tokens": _inbound_tokens,
                        "passthrough_limit": _pt_limit,
                        "upstream_limit": _upstream_limit,
                        "pairs_trimmed": _pt_trimmed,
                    })

            # Compute outbound tokens (after trim + tool interception)
            _outbound_tokens = fmt._count(json.dumps(body, default=str))
            _outbound_bytes = len(json.dumps(body, default=str).encode("utf-8"))

            # Non-virtualizable floor: system prompt + tools + anything VC can't touch
            _pt_system_tokens = fmt._estimate_system_tokens(body)
            _pt_tools_tokens = fmt.estimate_tools_tokens(body)
            _pt_floor = _pt_system_tokens + _pt_tools_tokens
            if state:
                state._last_non_virtualizable_floor = _pt_floor

            # Record passthrough request event with accurate post-trim values
            metrics.record({
                "type": "request",
                "turn": turn,
                "turn_id": _turn_id,
                "message_preview": user_message[:60],
                "api_format": api_format,
                "streaming": is_streaming,
                "tags": [],
                "temporal": False,
                "context_tokens": 0,
                "budget": {},
                "history_len": len(state.conversation_history),
                "compacted_through": 0,
                "wait_ms": 0,
                "inbound_ms": 0,
                "overhead_ms": 0,
                "total_turns": turn,
                "filtered_turns": turn,
                "inbound_tokens": _inbound_tokens,
                "outbound_tokens": _outbound_tokens,
                "input_tokens": _outbound_tokens,
                "raw_input_tokens": _inbound_tokens,
                "system_tokens": _pt_system_tokens,
                "turns_dropped": 0,
                "non_virtualizable_floor": _pt_floor,
                "upstream_context_limit": _upstream_limit,
                "passthrough_trim_limit": _pt_limit,
                "conversation_id": _conversation_id,
                "passthrough": True,
            })

            metrics.capture_request(
                turn, body, api_format,
                conversation_id=_conversation_id,
                passthrough=True,
                inbound_tokens=_inbound_tokens,
                outbound_tokens=_outbound_tokens,
                inbound_bytes=_inbound_bytes,
                outbound_bytes=_outbound_bytes,
                message_preview=user_message[:60],
                non_virtualizable_floor=_pt_floor,
                upstream_context_limit=_upstream_limit,
                passthrough_trim_limit=_pt_limit,
                system_tokens=_pt_system_tokens,
            )

            logger.info(
                "T%d PASSTHROUGH %s stream=%s state=%s in=%dt out=%dt | %s",
                turn, api_format, is_streaming, current_state.value,
                _inbound_tokens, _outbound_tokens, user_message[:60],
            )

            return PreparedPayload(
                body=body,
                enriched_body=body,
                conversation_id=_conversation_id,
                is_passthrough=True,
                turn=turn,
                turn_id=_turn_id,
                api_format=api_format,
                user_message=user_message,
                is_streaming=is_streaming,
                inbound_tokens=_inbound_tokens,
                outbound_tokens=_outbound_tokens,
                context_tokens=0,
                non_virtualizable_floor=0,
                upstream_limit=_upstream_limit,
                tags_matched=[],
                budget_breakdown={},
                turns_dropped=0,
                turns_stubbed=0,
                wait_ms=0,
                inbound_ms=0,
                overhead_ms=0,
                assembled=None,
                pre_filter_body=None,
                paging_enabled=False,
                tool_output_find_quote=False,
                inbound_bytes=_inbound_bytes,
                outbound_bytes=_outbound_bytes,
            )

    # ---------------------------------------------------------------
    # ACTIVE path: full enrichment
    # ---------------------------------------------------------------

    # One-time history rebuild from client payload if persisted history is insufficient
    if state:
        _expected = len(state.engine._turn_tag_index.entries)
        _have = len(state.conversation_history) // 2
        if _have < _expected and _expected > 0:
            _client_pairs = _extract_history_pairs(body)
            if _client_pairs and len(_client_pairs) // 2 > _have:
                state.conversation_history = list(_client_pairs)
                logger.info(
                    "HISTORY_REBUILD: persisted=%d, expected=%d, rebuilt from client payload (%d pairs)",
                    _have, _expected, len(_client_pairs) // 2,
                )

    prepend_text = ""
    assembled = None
    wait_ms = 0.0
    inbound_ms = 0.0
    if state:
        try:
            t0 = time.monotonic()
            await asyncio.to_thread(state.wait_for_tag)
            # Backpressure: if last tag_turn hit the hard threshold,
            # wait for pending compaction to finish before proceeding.
            # Soft threshold → async (no wait), hard → block until caught up.
            if state._last_compact_priority == "hard":
                await asyncio.to_thread(state.wait_for_complete)
            wait_ms = round((time.monotonic() - t0) * 1000, 1)

            state.conversation_history.append(
                Message(role="user", content=user_message,
                        timestamp=datetime.now(timezone.utc),
                        raw_content=fmt.extract_user_raw_content(body))
            )

            t1 = time.monotonic()
            assembled = await asyncio.to_thread(
                state.engine.on_message_inbound,
                user_message,
                state.conversation_history,
                body.get("model", ""),
            )
            inbound_ms = round((time.monotonic() - t1) * 1000, 1)

            prepend_text = assembled.prepend_text
        except Exception as e:
            logger.error("Engine error (forwarding unmodified): %s", e)

    # PROXY-025: Budget auto-promotion
    _effective_budget = 0
    _budget_promoted = False
    try:
        if state:
            _cw = int(state.engine.config.context_window)
            _effective_budget = _cw
            _sys_tok = fmt._estimate_system_tokens(body)
            _tools_tok = fmt.estimate_tools_tokens(body)
            _effective_budget, _budget_promoted = _compute_effective_budget(
                _cw, _sys_tok, _tools_tok,
            )
            if _budget_promoted:
                logger.info(
                    "BUDGET Client overhead (%dt) exceeds context_window (%dt). Auto-promoted to %dt.",
                    _sys_tok + _tools_tok, _cw, _effective_budget,
                )
                metrics.record({
                    "type": "budget_auto_promoted",
                    "original": _cw,
                    "promoted": _effective_budget,
                    "overhead": _sys_tok + _tools_tok,
                    "system_tokens": _sys_tok,
                    "tools_tokens": _tools_tok,
                })
    except (TypeError, ValueError, AttributeError):
        _effective_budget = 0

    # Capture the raw client body BEFORE any VC modifications (stubbing/filtering)
    _pre_filter_body = body

    # PROXY-025: Stub compacted messages via hash matching
    turns_stubbed = 0
    try:
        if state and int(state.engine._engine_state.compacted_through) > 0:
            from .message_filter import stub_compacted_messages
            body, turns_stubbed = stub_compacted_messages(
                body,
                state.engine._turn_tag_index,
                state.engine._engine_state.compacted_through,
                fmt=fmt,
            )
            if turns_stubbed:
                logger.info("STUB Stubbed %d compacted turns", turns_stubbed)
    except (TypeError, ValueError, AttributeError):
        pass

    # Filter irrelevant history turns from the request body
    turns_dropped = 0
    _real_tags = [t for t in (assembled.matched_tags if assembled else []) if t != "_general"]
    if _real_tags and state:
        # Use protected_recent_turns (compaction protection) for the
        # drop filter — NOT recent_turns_always_included (assembly).
        # The drop filter removes raw history from the client payload;
        # it should respect the same protection window as compaction.
        recent = state.engine.config.monitor.protected_recent_turns
        # PROXY-023: when paging is active, drop compacted turns so the
        # LLM relies on VC summaries + vc_expand_topic for old content.
        # NOTE: compacted_through is a lifetime segment index, not a
        # body-local pair index.  The stub filter already handles
        # replacement of compacted turns via hash matching.  Passing
        # the raw watermark to filter_body_messages would over-drop
        # because pair_idx (0..N) != turn_number, especially in proxy
        # mode where the client may have done its own compaction.
        # Disabled: the stub filter is the correct mechanism for
        # handling compacted turns in proxy mode.
        _ct = 0
        _pcf_mode = getattr(
            state.engine.config.assembler,
            "pre_compaction_filtering", "aggressive",
        )
        _pre_compaction = state.engine._engine_state.compacted_through == 0
        body, turns_dropped = _filter_body_messages(
            body,
            state.engine._turn_tag_index,
            _real_tags,
            recent_turns=recent,
            compacted_turn=_ct,
            fmt=fmt,
            pre_compaction_mode=_pcf_mode,
        )
        if turns_dropped:
            _phase = f"mode={_pcf_mode}, pre-compaction" if _pre_compaction else "post-compaction"
            logger.info("FILTER Dropped %d turns (%s)", turns_dropped, _phase)
        elif _pre_compaction and _pcf_mode == "off":
            logger.info("FILTER Skipped filtering (mode=off, pre-compaction)")

    # Tool output handling — active path only
    # Stage gate: post-compaction uses full chain collapse (stage 2),
    # pre-compaction uses position-based tool result stubbing (stage 1).
    _tool_stubs_present = False
    if state and state.engine.config.tool_output.enabled:
        _ct = int(state.engine._engine_state.compacted_through)
        if _ct > 0:
            # Stage 2: full chain collapse (post-compaction)
            from .message_filter import collapse_turn_chains
            body, _collapse_count, _chain_refs = collapse_turn_chains(
                body, fmt,
                pre_filter_body=_pre_filter_body,
                protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                turn_tag_index=state.engine._turn_tag_index,
                store=state.engine._store,
                conversation_id=state.engine.config.conversation_id,
            )
            if _collapse_count:
                _tool_stubs_present = True
                logger.info("CHAIN-COLLAPSE: collapsed %d turn chains (%d chain refs)", _collapse_count, len(_chain_refs))
        else:
            # Stage 1: tool result stubbing only (pre-compaction)
            from .message_filter import stub_tool_outputs_by_position
            body, _stub_count, _stub_refs = stub_tool_outputs_by_position(
                body, fmt,
                protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                turn_tag_index=state.engine._turn_tag_index,
                store=state.engine._store,
                conversation_id=state.engine.config.conversation_id,
            )
            if _stub_count:
                _tool_stubs_present = True
                logger.info("TOOL-STUB: stubbed %d tool outputs outside protected window", _stub_count)

    enriched_body = _inject_context(body, prepend_text, api_format)

    # Inject VC paging tools for autonomous mode (formats that support it)
    paging_enabled = False
    if (
        state
        and fmt.supports_tool_interception
        and state.engine.config.paging.enabled
    ):
        _paging_mode = state.engine._retrieval._resolve_paging_mode(
            enriched_body.get("model", ""),
        )
        if _paging_mode == "autonomous":
            tool_turn_count = len(state.engine._turn_tag_index.entries)
            try:
                compacted_count = int(state.engine._engine_state.compacted_through)
            except (TypeError, ValueError):
                compacted_count = 0
            require_tools = compacted_count > 0
            enriched_body = _inject_vc_tools(
                enriched_body,
                state.engine,
                require_tool_use=require_tools,
                restore_available=_tool_stubs_present,
            )
            paging_enabled = True
            _vc_names = [t["name"] for t in enriched_body.get("tools", []) if t.get("name", "").startswith("vc_")]
            logger.info(
                "PAGING Tools injected: %s (total tools: %d, policy=%s, turns=%d, compacted_through=%d)",
                _vc_names, len(enriched_body.get("tools", [])),
                "required" if require_tools else "optional",
                tool_turn_count, compacted_count,
            )
        else:
            logger.info("PAGING Mode=%s for model=%s -- tools NOT injected", _paging_mode, enriched_body.get("model", "?"))

    # Inject vc_find_quote for tool output retrieval (when paging didn't already inject it)
    tool_output_find_quote = False
    if (
        not paging_enabled
        and state
        and fmt.supports_tool_interception
        and state.engine.config.tool_output.enabled
    ):
        from ..core.tool_loop import vc_tool_definitions
        _all_defs = vc_tool_definitions()
        _fq_def = [d for d in _all_defs if d["name"] == "vc_find_quote"]
        if _fq_def:
            enriched_body = fmt.inject_tools(enriched_body, _fq_def)
            tool_output_find_quote = True
            logger.info("TOOL-OUTPUT Injected vc_find_quote tool for truncated output retrieval")

    # Inject vc_restore_tool when stubs are present but paging didn't already inject it
    if _tool_stubs_present and not paging_enabled and fmt.supports_tool_interception:
        existing_names = {t.get("name") for t in enriched_body.get("tools", []) if isinstance(t, dict)}
        if "vc_restore_tool" not in existing_names:
            from ..core.tool_loop import vc_tool_definitions
            _restore_def = [d for d in vc_tool_definitions() if d["name"] == "vc_restore_tool"]
            if _restore_def:
                enriched_body = fmt.inject_tools(enriched_body, _restore_def)
                logger.info("TOOL-STUB Injected vc_restore_tool for stub restoration")

    # Track enriched payload size
    if state:
        state._last_enriched_payload_kb = round(len(json.dumps(enriched_body)) / 1024, 1)

    # 2-to-llm: enriched body sent to the LLM (after filtering + context + tools)
    if log_dir and log_prefix:
        try:
            _to_llm_log = log_dir / f"{log_prefix}.2-to-llm.json"
            _to_llm_log.write_text(json.dumps(enriched_body, default=str))
        except Exception:
            logger.debug("enriched body log write failed", exc_info=True)

    is_streaming = body.get("stream", False)

    # Component-level estimate (diagnostic breakdown, not source of truth)
    system_tokens = fmt._estimate_system_tokens(body)

    # Ground truth: actual byte-measured outbound token count
    _outbound_json = json.dumps(enriched_body, default=str)
    _outbound_bytes = len(_outbound_json.encode("utf-8"))
    outbound_tokens = fmt._count(_outbound_json)

    # Ground truth: inbound tokens (what the client sent us, measured above)
    inbound_tokens = _inbound_tokens

    # VC must never send more than the client sent. If enrichment bloated
    # the payload beyond inbound, revert to the original client body and
    # treat as passthrough — all downstream metrics/capture will record
    # the passthrough values.
    _bloat_fallback = False
    if outbound_tokens > inbound_tokens:
        logger.warning(
            "VC_BLOAT_FALLBACK: enriched %dt > inbound %dt — reverting to passthrough (delta +%dt)",
            outbound_tokens, inbound_tokens, outbound_tokens - inbound_tokens,
        )
        _bloat_fallback = True
        enriched_body = _pre_filter_body
        body = _pre_filter_body
        _outbound_json = json.dumps(enriched_body, default=str)
        _outbound_bytes = len(_outbound_json.encode("utf-8"))
        outbound_tokens = fmt._count(_outbound_json)
        prepend_text = ""
        context_tokens = 0
        turns_dropped = 0
        turns_stubbed = 0
        paging_enabled = False
        tool_output_find_quote = False
        _tool_stubs_present = False
        assembled = None

    # Legacy aliases for downstream consumers
    input_tokens = outbound_tokens
    raw_input_tokens = inbound_tokens

    # Track enriched payload tokens for dashboard — outbound_tokens is ground truth
    _non_virtualizable_floor = 0
    if state:
        state._last_enriched_payload_tokens = outbound_tokens
        # Non-virtualizable floor: system + tools (what VC can't touch)
        if _bloat_fallback:
            # After bloat fallback, compute floor from the original body's
            # system prompt + tools — not the entire passthrough payload
            _vc_tokens = 0
            _non_virtualizable_floor = fmt._estimate_system_tokens(_pre_filter_body) + fmt.estimate_tools_tokens(_pre_filter_body)
        else:
            _vc_tokens = fmt._count(prepend_text) if prepend_text else 0
            _non_virtualizable_floor = max(0, outbound_tokens - _vc_tokens)
        state._last_non_virtualizable_floor = _non_virtualizable_floor
        logger.info("Floor: %dt non-virtualizable, %dt VC context, %dt total",
                    _non_virtualizable_floor, _vc_tokens, outbound_tokens)
        # Warn if context_window is at or below the floor
        try:
            _cw = int(state.engine.config.monitor.context_window)
            if _cw <= state._last_non_virtualizable_floor * 1.1:
                logger.warning(
                    "NON_VIRTUALIZABLE floor=%dt exceeds context_window=%dt — "
                    "VC context cannot be effectively injected",
                    state._last_non_virtualizable_floor, _cw,
                )
                metrics.record({
                    "type": "non_virtualizable_warning",
                    "floor": state._last_non_virtualizable_floor,
                    "context_window": _cw,
                    "outbound_tokens": outbound_tokens,
                    "vc_tokens": _vc_tokens,
                })
        except (TypeError, ValueError):
            pass

    # Upstream context enforcement — trim if enriched payload exceeds upstream limit
    _upstream_trimmed = 0
    _pre_trim_tokens = outbound_tokens
    if outbound_tokens > _upstream_limit - body.get("max_tokens", 4096):
        from .message_filter import trim_to_upstream_limit
        enriched_body, _upstream_trimmed = trim_to_upstream_limit(enriched_body, _upstream_limit, fmt)
        if _upstream_trimmed:
            _outbound_json = json.dumps(enriched_body, default=str)
            _outbound_bytes = len(_outbound_json.encode("utf-8"))
            outbound_tokens = fmt._count(_outbound_json)
            logger.info(
                "ACTIVE_TRIM: payload=%dt exceeds upstream=%dt, trimmed %d pairs → %dt",
                _pre_trim_tokens, _upstream_limit, _upstream_trimmed, outbound_tokens,
            )
            metrics.record({
                "type": "upstream_trim",
                "path": "active",
                "original_tokens": _pre_trim_tokens,
                "upstream_limit": _upstream_limit,
                "pairs_trimmed": _upstream_trimmed,
                "final_tokens": outbound_tokens,
            })

    # PROXY-025: Over-budget alert
    if state and _effective_budget > 0 and outbound_tokens > _effective_budget:
        _excess = outbound_tokens - _effective_budget
        logger.info(
            "BUDGET Payload %dt exceeds budget %dt by %dt. Uncompacted turns pending compaction.",
            outbound_tokens, _effective_budget, _excess,
        )
        metrics.record({
            "type": "budget_exceeded",
            "total": outbound_tokens,
            "budget": _effective_budget,
            "excess": _excess,
        })

    # Record request event
    turn = len(state.engine._turn_tag_index.entries) if state else 0
    _turn_id = uuid.uuid4().hex[:12]
    context_tokens = fmt._count(prepend_text) if prepend_text else 0
    total_turns = turn
    overhead_ms = round(wait_ms + inbound_ms, 1)
    _conversation_id = state.engine.config.conversation_id if state else ""
    metrics.record({
        "type": "request",
        "turn": turn,
        "turn_id": _turn_id,
        "model": body.get("model", ""),
        "message_preview": user_message[:60],
        "api_format": api_format,
        "streaming": is_streaming,
        "tags": assembled.matched_tags if assembled else [],
        "context_tokens": context_tokens,
        "budget": assembled.budget_breakdown if assembled else {},
        "history_len": len(state.conversation_history) if state else 0,
        "compacted_through": state.engine._engine_state.compacted_through if state else 0,
        "wait_ms": wait_ms,
        "inbound_ms": inbound_ms,
        "overhead_ms": overhead_ms,
        "total_turns": total_turns,
        "filtered_turns": total_turns - turns_dropped,
        "inbound_tokens": inbound_tokens,
        "outbound_tokens": outbound_tokens,
        "input_tokens": input_tokens,       # legacy alias for outbound
        "raw_input_tokens": raw_input_tokens,  # legacy alias for inbound
        "system_tokens": system_tokens,      # component estimate
        "turns_dropped": turns_dropped,
        "turns_stubbed": turns_stubbed,
        "non_virtualizable_floor": _non_virtualizable_floor,
        "upstream_context_limit": _upstream_limit,
        "passthrough_trim_limit": int(
            _upstream_limit * (state.engine.config.proxy.passthrough_trim_ratio if state else 0.40)
        ),
        "conversation_id": _conversation_id,
    })

    # Persist request context for dashboard recall page
    if state and assembled:
        try:
            _retrieval_meta = assembled.retrieval_metadata or {}
            _budget = assembled.budget_breakdown or {}
            _retrieval_scores = assembled.retrieval_scores or {}
            _selected_refs = set(assembled.presented_segment_refs or set())
            _seg_injected: list[dict] = []

            if _selected_refs and assembled.retrieval_summaries:
                _rank = 0
                for summary in assembled.retrieval_summaries:
                    if summary.ref not in _selected_refs:
                        continue
                    _rank += 1
                    _seg_injected.append(
                        _serialize_recall_segment(
                            summary,
                            rank=_rank,
                            retrieval_meta=_retrieval_meta,
                            retrieval_scores=_retrieval_scores,
                        ),
                    )

            if not _seg_injected and assembled.tag_sections:
                _fallback_selection_mode = _selection_mode(_retrieval_meta)
                for rank, tag in enumerate(assembled.tag_sections, start=1):
                    _seg_injected.append({
                        "ref": "",
                        "primary_tag": tag,
                        "tags": [tag],
                        "rank": rank,
                        "score": round(float(_retrieval_scores.get(tag, 0.0)), 4),
                        "selection_mode": _fallback_selection_mode,
                        "selected_because": [f"Selected from assembled tag context for {tag}."],
                        "matched_query_tags": [tag] if tag in (_retrieval_meta.get("tags_queried", []) or []) else [],
                        "related_match_tags": [tag] if tag in (_retrieval_meta.get("related_tags_used", []) or []) else [],
                        "summary": assembled.tag_sections[tag],
                        "summary_tokens": len(assembled.tag_sections[tag]) // 4,
                        "full_tokens": 0,
                        "turn_count": 0,
                        "session_date": "",
                        "entities": [],
                        "key_decisions": [],
                        "action_items": [],
                        "date_references": [],
                        "created_at": None,
                        "start_timestamp": None,
                        "end_timestamp": None,
                    })

            _facts_injected = [
                _serialize_recall_fact(fact, retrieval_meta=_retrieval_meta)
                for fact in (assembled.selected_facts or [])
            ]
            state.engine._store.save_request_context({
                "conversation_id": _conversation_id,
                "request_turn": turn,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_message": user_message[:500],
                "inbound_tags": assembled.matched_tags or [],
                "retrieval_method": _retrieval_meta.get("strategy", "rrf"),
                "candidates_found": _retrieval_meta.get("candidates_found", len(assembled.retrieval_summaries or [])),
                "candidates_selected": _retrieval_meta.get("summaries_returned", len(_seg_injected)),
                "segments_injected": _seg_injected,
                "facts_injected": _facts_injected,
                "facts_count": len(_facts_injected),
                "facts_tags": (_retrieval_meta.get("tags_queried", []) or []) + (_retrieval_meta.get("related_tags_used", []) or []),
                "pool_used": _budget.get("tags", 0) + _budget.get("facts", 0),
                "pool_budget": state.engine.config.assembler.context_injection_max_tokens,
                "total_context_tokens": context_tokens,
                "non_virtualizable_floor": state._last_non_virtualizable_floor,
                "tool_call_count": 0,
            })
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).debug("Failed to save request context: %s", e)

    # Log request to terminal for debugging
    _tags_str = ", ".join(assembled.matched_tags) if assembled else "none"
    _flag_str = ""
    _out_str = f"out={outbound_tokens}t"
    if _upstream_trimmed:
        _out_str = f"out={outbound_tokens}t TRIMMED from {_pre_trim_tokens}t"
    logger.info(
        "T%d POST %s stream=%s tags=[%s]%s msgs=%d dropped=%d stubbed=%d "
        "ctx=%dt in=%dt %s upstream=%dt vc=%sms | %s",
        turn, api_format, is_streaming, _tags_str, _flag_str,
        len(body.get("messages", [])), turns_dropped, turns_stubbed,
        context_tokens, inbound_tokens, _out_str, _upstream_limit,
        overhead_ms, user_message[:60],
    )

    # Capture pre-filter request body for dashboard inspection
    metrics.capture_request(
        turn, _pre_filter_body, api_format,
        inbound_tags=assembled.matched_tags if assembled else [],
        conversation_id=_conversation_id,
        inbound_tokens=inbound_tokens,
        outbound_tokens=outbound_tokens,
        inbound_bytes=_inbound_bytes,
        outbound_bytes=_outbound_bytes,
        context_tokens=context_tokens,
        overhead_ms=overhead_ms,
        turns_dropped=turns_dropped,
        turns_stubbed=turns_stubbed,
        message_preview=user_message[:60],
        non_virtualizable_floor=_non_virtualizable_floor,
        upstream_context_limit=_upstream_limit,
        passthrough_trim_limit=int(
            _upstream_limit * (state.engine.config.proxy.passthrough_trim_ratio if state else 0.40)
        ),
        system_tokens=system_tokens,
    )
    # Capture enriched body (what we actually send to the LLM)
    metrics.capture_enriched(
        turn,
        enriched_body,
        conversation_id=_conversation_id,
    )

    return PreparedPayload(
        body=body,
        enriched_body=enriched_body,
        conversation_id=_conversation_id,
        is_passthrough=_bloat_fallback,
        turn=turn,
        turn_id=_turn_id,
        api_format=api_format,
        user_message=user_message,
        is_streaming=is_streaming,
        inbound_tokens=inbound_tokens,
        outbound_tokens=outbound_tokens,
        context_tokens=context_tokens,
        non_virtualizable_floor=_non_virtualizable_floor,
        upstream_limit=_upstream_limit,
        tags_matched=assembled.matched_tags if assembled else [],
        budget_breakdown=assembled.budget_breakdown if assembled else {},
        turns_dropped=turns_dropped,
        turns_stubbed=turns_stubbed,
        wait_ms=wait_ms,
        inbound_ms=inbound_ms,
        overhead_ms=overhead_ms,
        assembled=assembled,
        pre_filter_body=_pre_filter_body,
        paging_enabled=paging_enabled,
        tool_output_find_quote=tool_output_find_quote,
        inbound_bytes=_inbound_bytes,
        outbound_bytes=_outbound_bytes,
    )


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------

def create_app(
    upstream: str,
    config_path: str | None = None,
    *,
    shared_engine: VirtualContextEngine | None = None,
    shared_metrics: ProxyMetrics | None = None,
    instance_label: str = "",
    instance_upstream_limit: int = 0,
    embedding_provider=None,
) -> FastAPI:
    """Create the FastAPI proxy application.

    Args:
        upstream: Upstream provider base URL (e.g. https://api.anthropic.com).
        config_path: Path to virtual-context config file.
        shared_engine: Reuse an existing engine (multi-instance mode).
        shared_metrics: Reuse an existing metrics collector (multi-instance mode).
        instance_label: Human-readable label for this instance (e.g. "anthropic").
    """
    upstream = upstream.rstrip("/")

    # Initialize engine + session registry
    registry: SessionRegistry | None = None
    default_state: ProxyState | None = None
    try:
        # Redis session cache (optional) — must exist before engine __init__
        session_cache = None
        try:
            import os as _os
            _redis_url = _os.environ.get("REDIS_URL", "")
            if not _redis_url and config_path:
                from ..config import load_config as _load_cfg
                try:
                    _pre_cfg = _load_cfg(config_path)
                    _redis_url = _pre_cfg.proxy.redis_url
                except Exception:
                    pass
            # Also check shared_engine's config when no config_path
            if not _redis_url and shared_engine and hasattr(shared_engine.config, 'proxy'):
                try:
                    _redis_url = shared_engine.config.proxy.redis_url
                except (AttributeError, TypeError):
                    pass
            if _redis_url:
                from .session_cache import RedisSessionCache
                session_cache = RedisSessionCache(_redis_url)
        except Exception:
            pass

        if shared_engine is not None:
            engine = shared_engine
            if session_cache:
                engine._session_cache = session_cache
        else:
            engine = VirtualContextEngine(config_path=config_path, session_cache=session_cache, embedding_provider=embedding_provider)

            # Lossless restart: if engine has no persisted state for its
            # auto-generated conversation_id, try loading the most recent conversation.
            # This avoids re-ingestion on proxy restart.
            if (
                hasattr(engine, '_store')
                and hasattr(engine._store, 'load_latest_engine_state')
                and not engine._turn_tag_index.entries
            ):
                try:
                    latest = engine._store.load_latest_engine_state()
                    if (
                        latest
                        and isinstance(getattr(latest, 'turn_tag_entries', None), list)
                        and latest.turn_tag_entries
                    ):
                        # Try Redis first for the restored conversation ID
                        _redis_hit = False
                        if session_cache and session_cache.is_available():
                            _cached = session_cache.load_snapshot(latest.conversation_id)
                            if _cached and _cached.get("conversation_id") == latest.conversation_id:
                                engine.config.conversation_id = latest.conversation_id
                                engine._apply_cached_state(_cached)
                                engine._apply_persisted_state_to_delegates()
                                engine._bootstrap_vocabulary()
                                engine._init_retriever()
                                if hasattr(engine, '_retrieval'):
                                    engine._retrieval._retriever = engine._retriever
                                if hasattr(engine, '_paging'):
                                    engine._paging._conversation_id = engine.config.conversation_id
                                _redis_hit = True
                                logger.info(
                                    "Lossless restart: restored from Redis (conv=%s, version=%s)",
                                    latest.conversation_id[:12], _cached.get("version", "?"),
                                )

                        if not _redis_hit:
                            engine.config.conversation_id = latest.conversation_id
                            engine._load_persisted_state()
                            engine._apply_persisted_state_to_delegates()
                            engine._bootstrap_vocabulary()
                            engine._init_retriever()
                            if hasattr(engine, '_retrieval'):
                                engine._retrieval._retriever = engine._retriever
                            if hasattr(engine, '_paging'):
                                engine._paging._conversation_id = engine.config.conversation_id
                            logger.info(
                                "Lossless restart: restored from store (conv=%s, %d turns)",
                                latest.conversation_id[:12], len(latest.turn_tag_entries),
                            )
                except Exception as _e:
                    logger.info("Lossless restart failed: %s", _e)

        if shared_metrics is not None:
            metrics = shared_metrics
        else:
            metrics = ProxyMetrics(
                context_window=engine.config.monitor.context_window,
                telemetry_ledger=engine._telemetry,
            )
        # Create the default session (used by dashboard and first requests)
        default_state = ProxyState(engine, metrics=metrics, upstream=upstream)

        if instance_upstream_limit:
            default_state._instance_upstream_limit = instance_upstream_limit

        # Build registry and pre-register the default session
        registry = SessionRegistry(
            config_path=config_path,
            upstream=upstream,
            metrics=metrics,
            store=engine._store,
            session_cache=session_cache,
            embedding_provider=engine._embedding_provider,
        )
        registry._conversations[engine.config.conversation_id] = default_state

        logger.info(
            "Engine ready — conversation_id=%s, window=%d, storage=%s%s",
            engine.config.conversation_id,
            engine.config.monitor.context_window,
            engine.config.storage.backend,
            f", label={instance_label}" if instance_label else "",
        )
    except Exception as e:
        logger.info("Engine init failed: %s", e)
        metrics = shared_metrics or ProxyMetrics()

    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    shutdown_event = asyncio.Event()

    # --------------- Raw request log setup ---------------
    _request_log_dir: Path | None = None
    _request_log_max: int = 50

    if default_state:
        try:
            from ..types import ProxyConfig as _ProxyConfig
            proxy_cfg = default_state.engine.config.proxy
            if isinstance(proxy_cfg, _ProxyConfig):
                _request_log_dir = Path(proxy_cfg.request_log_dir)
                _request_log_max = proxy_cfg.request_log_max_files

                _request_log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

                # Prune old files on startup — keep only the newest N sets
                # (each request produces up to 6 files: 1-inbound, 2-to-llm,
                # 3-from-llm, 4-to-client, session, plus continuation files)
                existing = sorted(
                    list(_request_log_dir.glob("*.json"))
                    + list(_request_log_dir.glob("*.txt")),
                    key=lambda p: p.stat().st_mtime,
                )
                keep_files = _request_log_max * 6
                if len(existing) > keep_files:
                    for stale in existing[: len(existing) - keep_files]:
                        stale.unlink(missing_ok=True)
                    pruned = len(existing) - keep_files
                    logger.info("Request log: pruned %d old files, kept %d in %s", pruned, keep_files, _request_log_dir)
                else:
                    logger.info("Request log: %d existing files in %s", len(existing), _request_log_dir)
        except Exception:
            logger.debug("request log pruning failed", exc_info=True)  # engine may be a mock in tests

    import itertools as _itertools
    _log_seq = _itertools.count(1)  # atomic monotonic counter for filenames

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        yield
        shutdown_event.set()
        await client.aclose()
        if registry:
            registry.shutdown_all()

    _app_title = "virtual-context proxy"
    if instance_label:
        _app_title += f" [{instance_label}]"
    app = FastAPI(title=_app_title, lifespan=lifespan)
    app.state.instance_label = instance_label
    # Pluggable session resolver: if set, called instead of the built-in
    # SessionRegistry.  Signature:
    #   (request, body, conversation_id) -> (ProxyState, is_new)
    # Cloud wrappers (e.g. virtual-context-cloud) set this to route
    # requests to per-tenant engines.  Default None = use local registry.
    app.state.state_resolver = None

    # Register dashboard routes BEFORE the catch-all so /dashboard is not swallowed
    # Dashboard uses the default state for settings and config access
    register_dashboard_routes(
        app, metrics, default_state, shutdown_event,
        registry=registry, instance_label=instance_label,
    )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def catch_all(request: Request, path: str):
        # Preserve query string (e.g. ?beta=true for Claude Code extended thinking)
        _qs = request.url.query
        _suffix = f"?{_qs}" if _qs else ""
        url = f"{upstream}/{path}{_suffix}"
        # Per-request upstream override (e.g. subdomain routing in cloud)
        _req_upstream = getattr(request.state, "upstream", None)
        if _req_upstream:
            url = f"{_req_upstream}/{path}{_suffix}"
        raw_headers = dict(request.headers)
        fwd_headers = _forward_headers(raw_headers)

        # Non-POST or no body → passthrough
        if request.method != "POST":
            return await _passthrough(client, request, url, fwd_headers)

        body_bytes = await request.body()
        if not body_bytes:
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        # --- Raw request log: dump entire payload before any processing ---
        # Check app.state for runtime-configurable log dir (cloud layer sets this)
        # app.state takes priority over config default (cloud overrides engine config)
        _state_log_dir = getattr(app.state, "request_log_dir", None)
        _effective_log_dir = _state_log_dir or _request_log_dir
        if _effective_log_dir and isinstance(_effective_log_dir, (str, Path)):
            _effective_log_dir = Path(_effective_log_dir)
            _effective_log_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.warning(
                "DIAG_LOG_DIR state=%r config=%r effective=%r type=%s",
                _state_log_dir, _request_log_dir, _effective_log_dir,
                type(_effective_log_dir).__name__,
            )
            _effective_log_dir = None
        _response_log_path: Path | None = None
        _session_log_path: Path | None = None
        _log_prefix = ""
        if _effective_log_dir and body_bytes:
            _seq = next(_log_seq)
            import datetime as _dt_log
            ts = _dt_log.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            _log_prefix = f"{_seq:06d}_{ts}_{path.replace('/', '_')}"
            # 1-inbound: raw request from client
            req_log = _effective_log_dir / f"{_log_prefix}.1-inbound.json"
            _response_log_path = _effective_log_dir / f"{_log_prefix}.3-from-llm.json"
            _session_log_path = _effective_log_dir / f"{_log_prefix}.session.json"
            try:
                req_log.write_bytes(body_bytes)
            except Exception as _log_err:
                logger.error("DIAG_WRITE_FAIL 1-inbound: %s path=%s", _log_err, req_log)

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        # Only intercept if it has a messages/contents array (chat completion)
        fmt = detect_format(body)
        if not fmt.has_messages(body):
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        # Extract conversation ID from markers before stripping them
        inbound_conversation_id = _extract_conversation_id(body)

        # Strip conversation markers from assistant messages before any processing.
        # The LLM should never see stale markers.
        body = _strip_conversation_markers(body)

        # Route to the correct session
        state: ProxyState | None = None
        _resolver = getattr(app.state, "state_resolver", None)
        if _resolver is not None:
            state, is_new = _resolver(request, body, inbound_conversation_id)
        elif registry:
            state, is_new = registry.get_or_create(
                inbound_conversation_id, body=body,
            )
            # Update last-message hash so the next request can match.
            if state and body is not None:
                registry.update_last_message_hash(
                    body, state.engine.config.conversation_id,
                )

        # In multi-tenant mode, use the tenant's metrics so captures
        # (request, enriched, response) land on the correct instance.
        if state and state.metrics:
            metrics = state.metrics

        # Use format-specific token counter (anthropic tokenizer for Anthropic,
        # tiktoken for others, with fallback chain)
        from ..token_counter import get_counter_for_format
        fmt.set_token_counter(get_counter_for_format(fmt.name))

        api_format = fmt.name
        user_message = fmt.extract_user_message(body)
        is_streaming = body.get("stream", False)

        import datetime as _dt
        _payload_kb = round(len(body_bytes) / 1024, 1)
        _now = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _msg_count = len(fmt.get_messages(body))
        _sid = state.engine.config.conversation_id[:12] if state else "none"
        logger.info("%s POST /%s msgs=%d stream=%s conversation=%s payload=%sKB", _now, path, _msg_count, is_streaming, _sid, _payload_kb)

        if not user_message and _msg_count < 10:
            # Trivial tool-result or non-text turn with very few messages —
            # a one-off tool call, not a real conversation. Skip enrichment.
            # Conversations with 10+ messages still go through prepare_payload
            # so tool output stubbing runs even when the last message is a
            # tool result.
            _skip_sid = state.engine.config.conversation_id if state else ""
            _skip_turn = len(state.engine._turn_tag_index.entries) if state else 0
            _skip_turn_id = uuid.uuid4().hex[:12]
            if is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=_skip_turn, turn_id=_skip_turn_id,
                    conversation_id=_skip_sid, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=_skip_turn, turn_id=_skip_turn_id,
                    conversation_id=_skip_sid, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                )

        # ---------------------------------------------------------------
        # Enrichment: passthrough or active path via prepare_payload()
        # ---------------------------------------------------------------
        result = await prepare_payload(
            body, state, fmt, metrics,
            body_bytes=body_bytes,
            inbound_conversation_id=inbound_conversation_id,
            log_dir=_effective_log_dir,
            log_prefix=_log_prefix,
        )

        if result.is_passthrough:
            if result.is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, result.body, result.api_format, state,
                    metrics=metrics, turn=result.turn, turn_id=result.turn_id,
                    conversation_id=result.conversation_id,
                    passthrough=True, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, result.body, result.api_format, state,
                    metrics=metrics, turn=result.turn, turn_id=result.turn_id,
                    conversation_id=result.conversation_id,
                    passthrough=True, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                )
        else:
            _intercept_vc_tools = result.paging_enabled or result.tool_output_find_quote

            if result.is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, result.enriched_body, result.api_format, state,
                    metrics=metrics, turn=result.turn, turn_id=result.turn_id,
                    overhead_ms=result.overhead_ms,
                    conversation_id=result.conversation_id,
                    response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    paging_enabled=_intercept_vc_tools,
                    request_log_dir=_effective_log_dir,
                    log_prefix=_log_prefix if _effective_log_dir else "",
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, result.enriched_body, result.api_format, state,
                    metrics=metrics, turn=result.turn, turn_id=result.turn_id,
                    overhead_ms=result.overhead_ms,
                    conversation_id=result.conversation_id,
                    response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                )

    return app
