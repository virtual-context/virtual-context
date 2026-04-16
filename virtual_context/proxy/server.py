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
import copy
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
    TurnGroup,
    detect_format,
    get_format,
    summarize_payload_accounting,
    summarize_raw_payload_entries,
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

_PREP_BREAKDOWN_LOG_THRESHOLD_MS = 1_000.0
_PREP_BREAKDOWN_LOG_THRESHOLD_BYTES = 1_000_000
_PREP_BREAKDOWN_MAX_STAGES = 8
_PROTECTED_BREAKDOWN_LOG_THRESHOLD_TOKENS = 50_000


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


def _payload_messages(body: dict) -> list[dict]:
    messages = body.get(
        "messages",
        body.get("input", body.get("contents", [])),
    )
    return messages if isinstance(messages, list) else []


def _compute_protected_turn_stats(
    body: dict,
    fmt: PayloadFormat,
    protected_recent_turns: int,
    *,
    message_tokens: list[int] | None = None,
    turn_groups: list[TurnGroup] | None = None,
) -> dict[str, object]:
    """Return protected-turn token accounting for the final outbound body.

    This must be computed from the last body mutation we actually send upstream;
    earlier intermediate bodies can differ substantially after budget
    enforcement, fill, bloat fallback, or active trim.
    """
    if protected_recent_turns <= 0:
        return {"tokens": 0, "count": 0, "turn_tokens": []}

    messages = _payload_messages(body)
    if not messages:
        return {"tokens": 0, "count": 0, "turn_tokens": []}

    turn_groups = turn_groups or fmt.group_into_turns(body)
    if not turn_groups:
        return {"tokens": 0, "count": 0, "turn_tokens": []}

    protected_groups = turn_groups[max(0, len(turn_groups) - protected_recent_turns):]
    turn_tokens: list[int] = []
    total_tokens = 0
    seen_indices: set[int] = set()
    for group in protected_groups:
        group_total = 0
        for idx in group.indices:
            if idx in seen_indices or idx < 0 or idx >= len(messages):
                continue
            seen_indices.add(idx)
            if message_tokens is not None and idx < len(message_tokens):
                group_total += message_tokens[idx]
            else:
                group_total += fmt.estimate_message_tokens(messages[idx])
        turn_tokens.append(group_total)
        total_tokens += group_total

    return {
        "tokens": total_tokens,
        "count": len(protected_groups),
        "turn_tokens": turn_tokens,
    }


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

    _prepare_started = time.monotonic()
    _prep_breakdown: dict[str, float] = {}

    def _note_prep(stage: str, started_at: float) -> float:
        elapsed = round((time.monotonic() - started_at) * 1000, 1)
        _prep_breakdown[stage] = round(_prep_breakdown.get(stage, 0.0) + elapsed, 1)
        return elapsed

    def _prepare_metadata(total_ms: float | None = None) -> dict:
        total = total_ms if total_ms is not None else round(
            (time.monotonic() - _prepare_started) * 1000, 1,
        )
        breakdown = {
            stage: ms for stage, ms in _prep_breakdown.items()
            if ms > 0
        }
        accounted = round(sum(breakdown.values()), 1)
        unaccounted = round(max(total - accounted, 0.0), 1)
        if unaccounted > 0.1:
            breakdown["unaccounted"] = unaccounted
        return {
            "prepare_total_ms": total,
            "prepare_breakdown": breakdown,
        }

    def _log_prepare_breakdown(
        *,
        total_ms: float,
        conversation_id: str,
        turn: int,
        outbound_tokens: int,
    ) -> None:
        if (
            total_ms < _PREP_BREAKDOWN_LOG_THRESHOLD_MS
            and _inbound_bytes < _PREP_BREAKDOWN_LOG_THRESHOLD_BYTES
        ):
            return
        meta = _prepare_metadata(total_ms)
        breakdown = meta["prepare_breakdown"]
        stages = sorted(
            (
                (stage, ms) for stage, ms in breakdown.items()
                if stage != "unaccounted"
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:_PREP_BREAKDOWN_MAX_STAGES]
        stage_bits = [f"{stage}={ms:.1f}ms" for stage, ms in stages]
        if "unaccounted" in breakdown:
            stage_bits.append(f"unaccounted={breakdown['unaccounted']:.1f}ms")
        logger.info(
            "PREP_BREAKDOWN conv=%s turn=%s format=%s msgs=%d payload=%sKB in=%dt out=%dt total=%sms %s",
            conversation_id[:12] if conversation_id else "none",
            turn,
            api_format,
            _initial_message_count,
            _payload_kb,
            _inbound_tokens,
            outbound_tokens,
            total_ms,
            " ".join(stage_bits) if stage_bits else "no-stages",
        )

    def _serialize_payload_and_estimate(
        payload: dict,
        *,
        serialize_stage: str,
        count_stage: str,
        cache_scope: str | None = None,
    ) -> tuple[str, int, int]:
        _serialize_started = time.monotonic()
        payload_json = json.dumps(payload, default=str)
        payload_bytes = len(payload_json.encode("utf-8"))
        _note_prep(serialize_stage, _serialize_started)
        payload_tokens = _estimate_payload_tokens_cached(
            payload,
            cache_scope=cache_scope,
            count_stage=count_stage,
            serialized_json=payload_json,
        )
        return payload_json, payload_bytes, payload_tokens

    # Measure inbound BEFORE normalization using the media-aware segmented
    # estimator. It composes shell + per-message counts, which closely tracks
    # the legacy whole-body estimator while allowing stable prefix reuse.
    # Verified against the legacy estimate on a live 3.9MB Anthropic payload:
    #   - full-body estimator: 1,157,338t
    #   - segmented estimator: 1,158,117t (+0.07%)
    # This keeps media handling correct without re-tokenizing the entire raw
    # payload on every append-only turn.
    _payload_kb = round(len(body_bytes) / 1024, 1) if body_bytes else 0
    _inbound_bytes = len(body_bytes)
    _cache_conv_id = state.engine.config.conversation_id if state else ""
    _cache_provider = getattr(state.engine, "_session_state_provider", None) if state else None
    _outbound_cache = state._outbound_payload_token_cache if state else None
    _outbound_cache_loaded = _outbound_cache is not None
    _outbound_token_estimate = None
    _outbound_turn_groups: list[TurnGroup] | None = None
    _outbound_turn_groups_signature: tuple[str, ...] | None = None
    _outbound_floor_signature: tuple[str, str] | None = None
    _outbound_floor_components = (0, 0)
    _msg_key = "messages" if "messages" in body else "input" if "input" in body else "contents"
    _initial_message_count = len(body[_msg_key]) if _msg_key in body and isinstance(body[_msg_key], list) else 0
    _inbound_stage = time.monotonic()
    _inbound_cache_estimate = None
    _inbound_cache_source = "none"
    _inbound_cache = state._inbound_payload_token_cache if state else None
    if _inbound_cache is not None:
        _inbound_cache_source = "memory"
    elif _cache_provider and _cache_conv_id:
        _cache_load_stage = time.monotonic()
        _inbound_cache = _cache_provider.load_payload_token_cache(_cache_conv_id)
        _note_prep("inbound_token_cache_load", _cache_load_stage)
        if _inbound_cache is not None and state:
            state._inbound_payload_token_cache = _inbound_cache
            _inbound_cache_source = "redis"
    if body_bytes:
        _inbound_cache_estimate = fmt.estimate_payload_tokens_segmented(
            body,
            cache=_inbound_cache,
        )
        _inbound_tokens = _inbound_cache_estimate.total_tokens
        if state:
            state._inbound_payload_token_cache = _inbound_cache_estimate.cache
        if _cache_provider and _cache_conv_id:
            _cache_save_stage = time.monotonic()
            _cache_provider.save_payload_token_cache(
                _cache_conv_id,
                _inbound_cache_estimate.cache,
            )
            _note_prep("inbound_token_cache_save", _cache_save_stage)
    else:
        _inbound_tokens = 0
    _note_prep("inbound_token_count", _inbound_stage)
    if state:
        state._last_payload_kb = _payload_kb
        state._last_payload_tokens = _inbound_tokens
        if state._initial_payload_kb is None:
            state._initial_payload_kb = _payload_kb
            state._initial_payload_tokens = _inbound_tokens
        if state.is_conversation_deleted():
            state = None
    if _inbound_cache_estimate and _inbound_cache_source != "none":
        logger.info(
            "INBOUND_TOKEN_CACHE: conv=%s source=%s reused=%d/%d recounted=%d shell_cached=%s total=%dt",
            _cache_conv_id[:12] if _cache_conv_id else "none",
            _inbound_cache_source,
            _inbound_cache_estimate.reused_prefix_messages,
            _initial_message_count,
            _inbound_cache_estimate.recounted_messages,
            _inbound_cache_estimate.shell_cache_hit,
            _inbound_tokens,
        )

    def _estimate_payload_tokens_cached(
        payload: dict,
        *,
        cache_scope: str | None = None,
        count_stage: str | None = None,
        serialized_json: str | None = None,
    ) -> int:
        nonlocal _outbound_cache, _outbound_cache_loaded, _outbound_token_estimate

        _count_started = time.monotonic()
        if cache_scope == "outbound":
            if not _outbound_cache_loaded and _cache_provider and _cache_conv_id:
                _cache_load_stage = time.monotonic()
                _outbound_cache = _cache_provider.load_payload_token_cache(
                    _cache_conv_id,
                    scope="outbound",
                )
                _note_prep("outbound_token_cache_load", _cache_load_stage)
                if _outbound_cache is not None and state:
                    state._outbound_payload_token_cache = _outbound_cache
                _outbound_cache_loaded = True

            _estimate = fmt.estimate_payload_tokens_segmented(
                payload,
                cache=_outbound_cache,
            )
            _outbound_token_estimate = _estimate
            _outbound_cache = _estimate.cache
            if state:
                state._outbound_payload_token_cache = _outbound_cache
            if _cache_provider and _cache_conv_id:
                _cache_save_stage = time.monotonic()
                _cache_provider.save_payload_token_cache(
                    _cache_conv_id,
                    _outbound_cache,
                    scope="outbound",
                )
                _note_prep("outbound_token_cache_save", _cache_save_stage)
            payload_tokens = _estimate.total_tokens
        else:
            payload_tokens = fmt.estimate_payload_tokens_from_serialized(
                payload,
                serialized_json or json.dumps(payload, default=str),
            )

        if count_stage:
            _note_prep(count_stage, _count_started)
        return payload_tokens

    def _current_outbound_message_tokens(payload: dict) -> list[int]:
        messages = _payload_messages(payload)
        if (
            _outbound_token_estimate is not None
            and len(_outbound_token_estimate.cache.message_tokens) == len(messages)
        ):
            return list(_outbound_token_estimate.cache.message_tokens)
        return []

    def _current_outbound_turn_groups(payload: dict) -> list[TurnGroup]:
        nonlocal _outbound_turn_groups, _outbound_turn_groups_signature

        if _outbound_token_estimate is not None:
            messages = _payload_messages(payload)
            message_fingerprints = tuple(_outbound_token_estimate.cache.message_fingerprints)
            if len(message_fingerprints) == len(messages):
                if (
                    _outbound_turn_groups is not None
                    and _outbound_turn_groups_signature == message_fingerprints
                ):
                    return _outbound_turn_groups
                _turn_stage = time.monotonic()
                _outbound_turn_groups = fmt.group_into_turns(payload)
                _outbound_turn_groups_signature = message_fingerprints
                _note_prep("outbound_turn_grouping", _turn_stage)
                return _outbound_turn_groups

        _turn_stage = time.monotonic()
        turn_groups = fmt.group_into_turns(payload)
        _note_prep("outbound_turn_grouping", _turn_stage)
        return turn_groups

    def _current_outbound_floor_components(payload: dict) -> tuple[int, int]:
        nonlocal _outbound_floor_signature, _outbound_floor_components

        system_sig = json.dumps(
            payload.get("system", payload.get("instructions", "")),
            default=str,
            sort_keys=True,
        )
        tools_sig = json.dumps(
            payload.get("tools", []),
            default=str,
            sort_keys=True,
        )
        signature = (system_sig, tools_sig)
        if _outbound_floor_signature == signature:
            return _outbound_floor_components

        _floor_component_stage = time.monotonic()
        system_tokens = fmt._estimate_system_tokens(payload)
        tools_tokens = fmt.estimate_tools_tokens(payload)
        _note_prep("estimate_floor_components", _floor_component_stage)
        _outbound_floor_signature = signature
        _outbound_floor_components = (system_tokens, tools_tokens)
        return _outbound_floor_components

    # Normalize non-standard message formats (e.g. OpenClaw toolResult/toolCall)
    # before any pipeline processing. Runs for both proxy and REST paths.
    from .formats import normalize_messages
    _raw_payload_accounting: dict | None = None
    _normalize_stage = time.monotonic()
    if _msg_key in body and isinstance(body[_msg_key], list):
        _raw_payload_accounting = summarize_raw_payload_entries(body[_msg_key])
        normalize_messages(body[_msg_key])
    _note_prep("normalize_messages", _normalize_stage)
    _payload_accounting = summarize_payload_accounting(
        body,
        fmt,
        raw_summary=_raw_payload_accounting,
    )

    api_format = fmt.name
    _extract_user_stage = time.monotonic()
    user_message = fmt.extract_user_message(body)
    _note_prep("extract_user_message", _extract_user_stage)
    is_streaming = body.get("stream", False)

    # --- VC command detection (VCATTACH, VCLABEL, VCSTATUS, VCRECALL, VCCOMPACT, VCLIST, VCFORGET) ---
    # OpenClaw wraps user messages in metadata envelopes (```json``` fenced blocks)
    # with the actual user text after the last fence. Strip the envelope first.
    import re as _re
    _vc_parse_stage = time.monotonic()
    _vc_user_text = user_message.strip()
    _last_fence = _vc_user_text.rfind("```")
    if _last_fence >= 0:
        _after_fence = _vc_user_text[_last_fence + 3:].strip()
        if _after_fence:
            _vc_user_text = _after_fence
    _vc_cmd_match = _re.match(
        r"^VC(ATTACH|LABEL|STATUS|RECALL|COMPACT|LIST|FORGET)(?:\s+(.+))?$",
        _vc_user_text, _re.IGNORECASE,
    )
    _note_prep("parse_vc_command", _vc_parse_stage)
    if _vc_cmd_match:
        _vc_cmd = _vc_cmd_match.group(1).upper()
        _vc_arg = (_vc_cmd_match.group(2) or "").strip()
        # Early return — no pipeline processing for VC commands.
        _vc_meta = _prepare_metadata()
        return PreparedPayload(
            body=body,
            enriched_body=body,
            conversation_id=state.engine.config.conversation_id if state else (inbound_conversation_id or ""),
            is_passthrough=False,
            turn=0,
            request_turn=0,
            turn_id="",
            api_format=api_format,
            user_message=user_message,
            is_streaming=is_streaming,
            inbound_tokens=0,
            outbound_tokens=0,
            context_tokens=0,
            non_virtualizable_floor=0,
            upstream_limit=0,
            tags_matched=[],
            budget_breakdown={},
            turns_dropped=0,
            turns_stubbed=0,
            wait_ms=0,
            inbound_ms=0,
            overhead_ms=_vc_meta["prepare_total_ms"],
            assembled=None,
            pre_filter_body=None,
            paging_enabled=False,
            tool_output_find_quote=False,
            restore_tool_injected=False,
            inbound_bytes=0,
            outbound_bytes=0,
            metadata=_vc_meta,
            is_vcattach=(_vc_cmd == "ATTACH"),
            vcattach_target_id="",
            vcattach_label=_vc_arg if _vc_cmd == "ATTACH" else "",
            vc_command=_vc_cmd.lower(),
            vc_command_arg=_vc_arg,
        )

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
        _limit_stage = time.monotonic()
        _instance_limit = int(getattr(state, '_instance_upstream_limit', 0)) if state else 0
    except (TypeError, ValueError):
        _instance_limit = 0
    try:
        _global_limit = int(state.engine.config.proxy.upstream_context_limit) if state else 0
    except (TypeError, ValueError, AttributeError):
        _global_limit = 0
    _upstream_limit = resolve_upstream_limit(_model_name, _instance_limit, _global_limit)
    _note_prep("resolve_upstream_limit", _limit_stage)

    # Media compression — compress images on first sight, store on disk.
    # Runs BEFORE passthrough/active split so both paths benefit.
    # A 391KB screenshot → ~40KB compressed saves ~88k tokens in passthrough trim.
    if state and state.engine.config.tool_output.enabled:
        from .media import compress_media_in_payload
        _data_dir = getattr(state.engine._store, '_data_dir', '')
        if not _data_dir:
            _db_path = getattr(state.engine._store, 'db_path', None)
            if _db_path:
                _data_dir = str(getattr(_db_path, 'parent', ''))
            if not _data_dir:
                _data_dir = os.environ.get('VC_DATA_DIR', '/data/tenants')
        _media_dir = os.path.join(_data_dir, 'media')
        _media_stage = time.monotonic()
        body, _media_compressed = compress_media_in_payload(
            body, fmt,
            store=state.engine._store,
            conversation_id=state.engine.config.conversation_id,
            media_dir=_media_dir,
        )
        _note_prep("compress_media", _media_stage)
        if _media_compressed:
            logger.info("MEDIA-COMPRESS: compressed %d images", _media_compressed)

    # ---------------------------------------------------------------
    # State-aware dispatch: PASSTHROUGH/INGESTING vs ACTIVE
    # ---------------------------------------------------------------
    _dispatch_history_pairs: list[Message] | None = None
    _passthrough_reason: str | None = None
    _dispatch_existing_turns = 0
    _dispatch_needed_turns = 0
    _dispatch_completed_turns = -1
    _dispatch_indexed_turns = -1
    _dispatch_compacted_through = 0
    _dispatch_pending_indexing = False
    _dispatch_manual_passthrough = False
    if state:
        _dispatch_stage = time.monotonic()
        state._total_requests += 1
        current_state = state.session_state
        _dispatch_completed_turns = state._completed_turn_count()
        _dispatch_indexed_turns = state._indexed_turn_count()
        _dispatch_existing_turns = _dispatch_indexed_turns
        _dispatch_pending_indexing = state.has_pending_indexing()
        _dispatch_manual_passthrough = bool(getattr(state, "_manual_passthrough", False))
        _dispatch_compacted_through = int(
            getattr(getattr(getattr(state, "engine", None), "_engine_state", None), "compacted_through", 0) or 0
        )
        if _dispatch_manual_passthrough:
            _passthrough_reason = "manual_override"
        elif current_state == SessionState.INGESTING:
            _passthrough_reason = "pending_indexing"

        # Fresh session starts ACTIVE but may need ingestion — check and
        # redirect to passthrough path if there's history to ingest.
        if current_state == SessionState.ACTIVE:
            _dispatch_history_pairs = _extract_history_pairs(body)
            _dispatch_needed_turns = len(_dispatch_history_pairs) // 2
            current_state, _passthrough_reason = state.resolve_prepare_state(_dispatch_history_pairs)
            _dispatch_completed_turns = state._completed_turn_count()
            _dispatch_indexed_turns = state._indexed_turn_count()
            _dispatch_existing_turns = _dispatch_indexed_turns
            _dispatch_pending_indexing = state.has_pending_indexing()
            _dispatch_compacted_through = int(
                getattr(getattr(getattr(state, "engine", None), "_engine_state", None), "compacted_through", 0) or 0
            )
            if _passthrough_reason == "pending_indexing":
                state.resume_pending_ingestion_if_needed()
                current_state = state.session_state
        _note_prep("session_state_dispatch", _dispatch_stage)

        if current_state in (SessionState.PASSTHROUGH, SessionState.INGESTING):
            if not _passthrough_reason:
                _passthrough_reason = "pending_indexing" if current_state == SessionState.INGESTING else "restore_not_ready"
            logger.info(
                "PASSTHROUGH_DECISION conversation=%s reason=%s existing_turns=%d needed_turns=%d "
                "last_completed_turn=%d last_indexed_turn=%d compacted_through=%d "
                "pending_indexing=%s manual_passthrough=%s",
                state.engine.config.conversation_id[:12],
                _passthrough_reason,
                _dispatch_existing_turns,
                _dispatch_needed_turns,
                _dispatch_completed_turns - 1,
                _dispatch_indexed_turns - 1,
                _dispatch_compacted_through,
                _dispatch_pending_indexing,
                _dispatch_manual_passthrough,
            )
            # Store latest body for catch-up loop
            state._latest_body = body

            # On first request: kick off non-blocking ingestion
            if not state._history_ingested():
                _dispatch_history_pairs = _dispatch_history_pairs or _extract_history_pairs(body)
                if _dispatch_history_pairs and not state.is_conversation_deleted():
                    state.conversation_history = list(_dispatch_history_pairs)
                await asyncio.to_thread(
                    state.start_ingestion_if_needed, _dispatch_history_pairs,
                )

            if not state.is_conversation_deleted():
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
            _pt_outbound_json = ""
            _outbound_bytes = 0
            _outbound_tokens = _inbound_tokens
            if _inbound_tokens > _pt_limit:
                from .message_filter import trim_to_upstream_limit
                _pre_trim_msgs = len(body.get(fmt.get_message_key(body) if hasattr(fmt, 'get_message_key') else 'messages', []))
                body, _pt_trimmed = trim_to_upstream_limit(body, _pt_limit, fmt)
                _pt_outbound_json, _outbound_bytes, _post_trim_tokens = _serialize_payload_and_estimate(
                    body,
                    serialize_stage="passthrough_serialize_outbound",
                    count_stage="passthrough_count_outbound",
                )
                _outbound_tokens = _post_trim_tokens
                _post_trim_msgs = len(body.get('messages', body.get('input', body.get('contents', []))))
                if _pt_trimmed:
                    logger.info(
                        "PASSTHROUGH_TRIM: inbound=%dt target=%dt actual=%dt "
                        "msgs=%d->%d pairs_dropped=%d (ratio=%.0f%%, upstream=%dt)",
                        _inbound_tokens, _pt_limit, _post_trim_tokens,
                        _pre_trim_msgs, _post_trim_msgs, _pt_trimmed,
                        _pt_ratio * 100, _upstream_limit,
                    )
                    if _post_trim_tokens < _pt_limit * 0.5:
                        logger.warning(
                            "PASSTHROUGH_TRIM_UNDERSHOOT: actual=%dt is %.0f%% of target=%dt — "
                            "large atomic tool chains likely caused over-trimming",
                            _post_trim_tokens, (_post_trim_tokens / _pt_limit) * 100, _pt_limit,
                        )
                    metrics.record({
                        "type": "upstream_trim",
                        "path": "passthrough",
                        "original_tokens": _inbound_tokens,
                        "passthrough_limit": _pt_limit,
                        "actual_tokens": _post_trim_tokens,
                        "upstream_limit": _upstream_limit,
                        "pairs_trimmed": _pt_trimmed,
                    })

            # Compute outbound tokens (after trim + tool interception)
            if not _pt_outbound_json:
                _pt_outbound_json, _outbound_bytes, _outbound_tokens = _serialize_payload_and_estimate(
                    body,
                    serialize_stage="passthrough_serialize_outbound",
                    count_stage="passthrough_count_outbound",
                )

            # Non-virtualizable floor: system prompt + tools + anything VC can't touch
            _pt_system_tokens = fmt._estimate_system_tokens(body)
            _pt_tools_tokens = fmt.estimate_tools_tokens(body)
            _pt_floor = _pt_system_tokens + _pt_tools_tokens
            if state:
                state._last_non_virtualizable_floor = _pt_floor

            _prepare_meta = _prepare_metadata()
            _prepare_meta["payload_accounting"] = dict(_payload_accounting)
            _prepare_total_ms = _prepare_meta["prepare_total_ms"]
            _conversation_id = state.engine.config.conversation_id
            _log_prepare_breakdown(
                total_ms=_prepare_total_ms,
                conversation_id=_conversation_id,
                turn=turn,
                outbound_tokens=_outbound_tokens,
            )

            # Record passthrough request event with accurate post-trim values
            metrics.record({
                "type": "request",
                "turn": turn,
                "turn_id": _turn_id,
                "message_preview": user_message[:200],
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
                "overhead_ms": _prepare_total_ms,
                "prepare_total_ms": _prepare_total_ms,
                "prepare_breakdown": _prepare_meta["prepare_breakdown"],
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
                "passthrough_reason": _passthrough_reason,
            })

            metrics.capture_request(
                turn, body, api_format,
                turn_id=_turn_id,
                conversation_id=_conversation_id,
                passthrough=True,
                passthrough_reason=_passthrough_reason or "",
                inbound_tokens=_inbound_tokens,
                outbound_tokens=_outbound_tokens,
                inbound_bytes=_inbound_bytes,
                outbound_bytes=_outbound_bytes,
                message_preview=user_message[:200],
                prepare_total_ms=_prepare_total_ms,
                prepare_breakdown=_prepare_meta["prepare_breakdown"],
                non_virtualizable_floor=_pt_floor,
                upstream_context_limit=_upstream_limit,
                passthrough_trim_limit=_pt_limit,
                system_tokens=_pt_system_tokens,
                protected_turn_tokens=0,
                protected_turn_count=0,
                payload_accounting=_payload_accounting,
            )

            # 2-to-llm: log passthrough body sent to the LLM (after trim)
            if log_dir and log_prefix:
                try:
                    _to_llm_log = log_dir / f"{log_prefix}.2-to-llm.json"
                    _to_llm_log.write_text(_pt_outbound_json)
                except Exception:
                    logger.debug("passthrough to-llm log write failed", exc_info=True)

            logger.info(
                "T%d PASSTHROUGH %s stream=%s state=%s reason=%s in=%dt out=%dt | %s",
                turn, api_format, is_streaming, current_state.value,
                _passthrough_reason,
                _inbound_tokens, _outbound_tokens, user_message[:60],
            )

            return PreparedPayload(
                body=body,
                enriched_body=body,
                conversation_id=_conversation_id,
                is_passthrough=True,
                turn=turn,
                request_turn=0,
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
                overhead_ms=_prepare_total_ms,
                assembled=None,
                pre_filter_body=None,
                paging_enabled=False,
                tool_output_find_quote=False,
                restore_tool_injected=False,
                inbound_bytes=_inbound_bytes,
                outbound_bytes=_outbound_bytes,
                metadata=_prepare_meta,
            )

    # ---------------------------------------------------------------
    # ACTIVE path: full enrichment
    # ---------------------------------------------------------------

    # One-time history rebuild from client payload if persisted history is insufficient
    if state:
        _history_rebuild_stage = time.monotonic()
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
        _note_prep("history_rebuild", _history_rebuild_stage)

    prepend_text = ""
    assembled = None
    wait_ms = 0.0
    inbound_ms = 0.0
    if state:
        try:
            t0 = time.monotonic()
            await asyncio.to_thread(state.wait_for_tag)
            wait_ms += _note_prep("wait_for_prior_tag", t0)
            # Backpressure: if last tag_turn hit the hard threshold,
            # wait for pending compaction to finish before proceeding.
            # Soft threshold → async (no wait), hard → block until caught up.
            if state._last_compact_priority == "hard":
                t_compact = time.monotonic()
                await asyncio.to_thread(state.wait_for_compact)
                wait_ms += _note_prep("wait_for_prior_compact", t_compact)

            if not state.is_conversation_deleted():
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
                inbound_ms = _note_prep("on_message_inbound", t1)

                prepend_text = assembled.prepend_text
        except Exception as e:
            logger.error("Engine error (forwarding unmodified): %s", e)

    # PROXY-025: Budget auto-promotion
    _effective_budget = 0
    _budget_promoted = False
    _budget_stage = time.monotonic()
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
    _note_prep("budget_auto_promotion", _budget_stage)

    # Capture the raw client body BEFORE any VC modifications (stubbing/filtering)
    _copy_stage = time.monotonic()
    _pre_filter_body = copy.deepcopy(body)
    _note_prep("copy_pre_filter_body", _copy_stage)

    # Detect client truncation — compare what the CLIENT sent (pre-filter)
    # against the store. Must run BEFORE drop_compacted_turns / filter
    # mutate the body.
    _client_truncated = False
    _truncation_stage = time.monotonic()
    if state:
        _payload_turns = len(fmt.group_into_turns(_pre_filter_body))
        _store_turns = len(state.engine._turn_tag_index.entries)
        _recovery_threshold = getattr(
            state.engine.config.monitor, "store_recovery_threshold", 0.70,
        )
        if _store_turns > 10 and _payload_turns < _store_turns * _recovery_threshold:
            _client_truncated = True
            logger.info(
                "STORE-RECOVERY: payload_turns=%d store_turns=%d threshold=%.0f%% — recovering from store",
                _payload_turns, _store_turns, _recovery_threshold * 100,
            )
    _note_prep("client_truncation_check", _truncation_stage)

    # ── Flush gate: decide whether to apply payload mutations this request ──
    _warm_defer = False
    try:
        _has_engine = state and getattr(state, 'engine', None) is not None
        _es = state.engine._engine_state if _has_engine else None
        _ct = int(getattr(_es, 'compacted_through', 0) or 0)
        _ft = int(getattr(_es, 'flushed_through', 0) or 0)
        _defer = bool(getattr(state.engine.config.monitor if _has_engine else None, "defer_payload_mutation", False))
        _flush_ttl = int(getattr(state.engine.config.monitor if _has_engine else None, "flush_ttl_seconds", 300) or 300)
        _last_req = float(getattr(_es, 'last_request_time', 0.0) or 0.0)

        if not _defer:
            # 5a. Legacy auto-track: no deferral — flushed tracks compacted
            logger.info("FLUSH_GATE: defer=False (legacy) ct=%d ft=%d", _ct, _ft)
            if _has_engine and _ct > _ft:
                state.engine._engine_state.flushed_through = _ct
                _ft = _ct
        else:
            # 5b. Compute cache age — treat unknown (0) as warm, not cold.
            # After restart last_request_time is 0; assume cache is still
            # warm to avoid flushing mutations on the very first request.
            _cache_age = (time.time() - _last_req) if _last_req > 0 else 0.0
            _should_flush_cold = _cache_age >= _flush_ttl

            if _should_flush_cold:
                # 5c. Cold-cache fast path — safe to mutate
                logger.info(
                    "FLUSH_GATE: defer=True COLD cache_age=%.1fs ttl=%ds ct=%d ft=%d last_req=%.1f — mutations ALLOWED",
                    _cache_age, _flush_ttl, _ct, _ft, _last_req,
                )
                if _has_engine and _ct > _ft:
                    state.engine._engine_state.flushed_through = _ct
                    _ft = _ct
            else:
                # 5d. Warm-cache — only defer mutations when there's
                # pending compaction work (ct > ft).  When ct == ft the
                # payload already reflects all compaction; mutations must
                # still run to keep the payload within budget.
                _flush_pending = _ct > _ft
                if _flush_pending:
                    _warm_defer = True
                    logger.info(
                        "FLUSH_GATE: defer=True WARM cache_age=%.1fs ttl=%ds ct=%d ft=%d — mutations DEFERRED",
                        _cache_age, _flush_ttl, _ct, _ft,
                    )
                else:
                    logger.info(
                        "FLUSH_GATE: defer=True WARM cache_age=%.1fs ttl=%ds ct=%d ft=%d — no pending work, mutations RUN",
                        _cache_age, _flush_ttl, _ct, _ft,
                    )
    except (TypeError, ValueError, AttributeError) as _gate_exc:
        logger.warning("FLUSH_GATE: exception — %s", _gate_exc)
        pass  # Mocked or missing engine state — fall through with _warm_defer=False

    # Drop compacted non-tool turns — their content is already in VC segments
    turns_stubbed = 0  # kept for downstream metrics compatibility
    _fill_summaries = 0
    _fill_turns = 0
    _recovery_chains = 0
    _recovery_turns = 0
    _drop_compacted_stage = time.monotonic()
    try:
        if not _warm_defer and state and int(state.engine._engine_state.flushed_through) > 0:
            from .message_filter import drop_compacted_turns
            body, turns_stubbed = drop_compacted_turns(
                body,
                state.engine._turn_tag_index,
                state.engine._engine_state.flushed_through,
                fmt=fmt,
                protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                drop_boundary=state.engine._engine_state.flushed_through if _defer else None,
            )
            if turns_stubbed:
                logger.info("DROP-COMPACTED: removed %d non-tool compacted turns", turns_stubbed)
    except (TypeError, ValueError, AttributeError):
        pass
    _note_prep("drop_compacted_turns", _drop_compacted_stage)

    # Filter irrelevant history turns from the request body
    turns_dropped = 0
    _real_tags = [t for t in (assembled.matched_tags if assembled else []) if t != "_general"]
    _filter_stage = time.monotonic()
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
    _note_prep("filter_body_messages", _filter_stage)

    # Tool output handling — active path only
    # Stage gate: post-compaction uses full chain collapse (stage 2),
    # pre-compaction uses position-based tool result stubbing (stage 1).
    _tool_stubs_present = False
    try:
        if state and state.engine.config.tool_output.enabled and not _warm_defer:
            _ct = int(state.engine._engine_state.compacted_through)
            if _ct > 0:
                # Stage 2: full chain collapse (post-compaction)
                from .message_filter import collapse_turn_chains
                _deep_ratio = getattr(
                    state.engine.config.compactor, "deep_compaction_ratio", 0.5,
                )
                _collapse_stage = time.monotonic()
                body, _collapse_count, _chain_refs, _recovery_chains = collapse_turn_chains(
                    body, fmt,
                    pre_filter_body=_pre_filter_body,
                    protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                    turn_tag_index=state.engine._turn_tag_index,
                    store=state.engine._store,
                    conversation_id=state.engine.config.conversation_id,
                    deep_compaction_ratio=_deep_ratio,
                    client_truncated=_client_truncated,
                    collapse_runtime_cache=getattr(state, "_chain_snapshot_cache", None),
                )
                _note_prep("collapse_turn_chains", _collapse_stage)
                if _collapse_count:
                    _tool_stubs_present = True
                    logger.info("CHAIN-COLLAPSE: collapsed %d turn chains (%d chain refs)", _collapse_count, len(_chain_refs))
                # After collapse, stub tool outputs in the protected zone if it's bloated
                from .message_filter import stub_tool_outputs_by_position
                _protected_stub_stage = time.monotonic()
                body, _prot_stub_count, _prot_stub_refs = stub_tool_outputs_by_position(
                    body, fmt,
                    protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                    turn_tag_index=state.engine._turn_tag_index,
                    store=state.engine._store,
                    conversation_id=state.engine.config.conversation_id,
                    protected_intrusion_threshold=0.6,
                    context_budget=state.engine.config.monitor.context_window,
                )
                _note_prep("protected_tool_stub", _protected_stub_stage)
                if _prot_stub_count:
                    _tool_stubs_present = True
                    logger.info("PROTECTED-STUB: stubbed %d tool outputs in protected zone", _prot_stub_count)
            else:
                # Stage 1: tool result stubbing only (pre-compaction)
                from .message_filter import stub_tool_outputs_by_position
                _tool_stub_stage = time.monotonic()
                body, _stub_count, _stub_refs = stub_tool_outputs_by_position(
                    body, fmt,
                    protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                    turn_tag_index=state.engine._turn_tag_index,
                    store=state.engine._store,
                    conversation_id=state.engine.config.conversation_id,
                    protected_intrusion_threshold=0.6,
                    context_budget=state.engine.config.monitor.context_window,
                )
                _note_prep("tool_stub_outputs", _tool_stub_stage)
                if _stub_count:
                    _tool_stubs_present = True
                    logger.info("TOOL-STUB: stubbed %d tool outputs outside protected window", _stub_count)
    except (TypeError, ValueError, AttributeError):
        pass  # Mocked or missing engine state — skip tool stubbing

    # Media stubbing — stub images outside protected window
    try:
        if state and state.engine.config.tool_output.enabled and not _warm_defer:
            from .media import stub_media_by_position
            _media_stub_stage = time.monotonic()
            body, _media_stubbed = stub_media_by_position(
                body, fmt,
                protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                protected_intrusion_threshold=0.6,
                context_budget=state.engine.config.monitor.context_window,
            )
            _note_prep("media_stub", _media_stub_stage)
            if _media_stubbed:
                _tool_stubs_present = True
                logger.info("MEDIA-STUB: stubbed %d images outside protected window", _media_stubbed)
    except (TypeError, ValueError, AttributeError):
        pass  # Mocked or missing engine state — skip media stubbing

    # Drop topic-only stubs — stubs without restore refs are dead weight
    from .message_filter import drop_topic_only_stubs
    _drop_stub_stage = time.monotonic()
    body, _stubs_dropped = drop_topic_only_stubs(body, fmt)
    _note_prep("drop_topic_only_stubs", _drop_stub_stage)
    if _stubs_dropped:
        logger.info("STUB-DROP: removed %d topic-only stubs", _stubs_dropped)

    # Merge consecutive conversational messages — fixes alternation violations
    _merge_stage = time.monotonic()
    fmt.merge_consecutive_conversational(body)
    _note_prep("merge_consecutive_messages", _merge_stage)

    _inject_stage = time.monotonic()
    if api_format == "anthropic" and prepend_text:
        from ..core.provider_adapters import AnthropicAdapter

        enriched_body = copy.deepcopy(body)
        AnthropicAdapter(api_key="").inject_context(enriched_body, prepend_text)
        if isinstance(enriched_body.get("system"), list):
            enriched_body["system"] = "\n\n".join(
                block.get("text", "")
                for block in enriched_body["system"]
                if isinstance(block, dict) and block.get("type") == "text"
            )
    else:
        enriched_body = _inject_context(body, prepend_text, api_format)
    _note_prep("inject_context", _inject_stage)

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
            _paging_stage = time.monotonic()
            enriched_body = _inject_vc_tools(
                enriched_body,
                state.engine,
                require_tool_use=require_tools,
                restore_available=_tool_stubs_present,
            )
            _note_prep("inject_paging_tools", _paging_stage)
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
            _find_quote_stage = time.monotonic()
            enriched_body = fmt.inject_tools(enriched_body, _fq_def)
            _note_prep("inject_find_quote_tool", _find_quote_stage)
            tool_output_find_quote = True
            logger.info("TOOL-OUTPUT Injected vc_find_quote tool for truncated output retrieval")

    # Inject vc_restore_tool when stubs are present but paging didn't already inject it
    _restore_tool_injected = False
    if _tool_stubs_present and not paging_enabled and fmt.supports_tool_interception:
        existing_names = {t.get("name") for t in enriched_body.get("tools", []) if isinstance(t, dict)}
        if "vc_restore_tool" not in existing_names:
            from ..core.tool_loop import vc_tool_definitions
            _restore_def = [d for d in vc_tool_definitions() if d["name"] == "vc_restore_tool"]
            if _restore_def:
                _restore_stage = time.monotonic()
                enriched_body = fmt.inject_tools(enriched_body, _restore_def)
                _note_prep("inject_restore_tool", _restore_stage)
                _restore_tool_injected = True
                logger.info("TOOL-STUB Injected vc_restore_tool for stub restoration")

    # Sanitize stale vc_restore_tool errors from history so the model isn't
    # poisoned by previous client-side "No such tool" rejections.
    if _restore_tool_injected or paging_enabled:
        from .message_filter import sanitize_vc_tool_errors
        _sanitize_stage = time.monotonic()
        enriched_body = sanitize_vc_tool_errors(enriched_body, fmt)
        _note_prep("sanitize_vc_tool_errors", _sanitize_stage)

    is_streaming = body.get("stream", False)

    # Component-level estimate (diagnostic breakdown, not source of truth)
    _system_tokens_stage = time.monotonic()
    system_tokens = fmt._estimate_system_tokens(body)
    _note_prep("estimate_system_tokens", _system_tokens_stage)

    # Strip VC internal markers before token counting and upstream send
    _strip_stage = time.monotonic()
    fmt.strip_vc_markers(enriched_body)
    _note_prep("strip_vc_markers", _strip_stage)

    # Ground truth: actual byte-measured outbound token count
    _outbound_json, _outbound_bytes, outbound_tokens = _serialize_payload_and_estimate(
        enriched_body,
        serialize_stage="serialize_outbound",
        count_stage="count_outbound_tokens",
        cache_scope="outbound",
    )
    if state:
        state._last_enriched_payload_kb = round(_outbound_bytes / 1024, 1)

    # ------------------------------------------------------------------
    # 5e. Two-pass safety valve: if warm-defer produced an oversized payload,
    # force-flush and re-run all skipped mutations.
    # ------------------------------------------------------------------
    if _warm_defer and state and _has_engine:
        _budget = int(state.engine.config.monitor.context_window)
        _hard = float(state.engine.config.monitor.hard_threshold)
        _size_limit = int(_budget * _hard)
        if outbound_tokens > _size_limit:
            logger.info(
                "FLUSH_GATE: SAFETY VALVE — payload %dt exceeds %dt (hard=%.2f * budget=%dt). Forcing flush.",
                outbound_tokens, _size_limit, _hard, _budget,
            )
            # Force flush: advance flushed_through and re-run mutations
            state.engine._engine_state.flushed_through = state.engine._engine_state.compacted_through
            _warm_defer = False

            # Re-run drop_compacted_turns
            try:
                from .message_filter import drop_compacted_turns
                enriched_body, _sv_dropped = drop_compacted_turns(
                    enriched_body,
                    state.engine._turn_tag_index,
                    state.engine._engine_state.flushed_through,
                    fmt=fmt,
                    drop_boundary=state.engine._engine_state.flushed_through,
                )
                if _sv_dropped:
                    logger.info("SAFETY-VALVE DROP-COMPACTED: removed %d turns", _sv_dropped)
            except (TypeError, ValueError, AttributeError):
                pass

            # Re-run tool stubbing (chain collapse + position stubbing)
            try:
                if state.engine.config.tool_output.enabled:
                    _ct = int(state.engine._engine_state.compacted_through)
                    if _ct > 0:
                        from .message_filter import collapse_turn_chains, stub_tool_outputs_by_position
                        _deep_ratio = getattr(state.engine.config.compactor, "deep_compaction_ratio", 0.5)
                        enriched_body, _sv_collapse, _sv_refs, _sv_chains = collapse_turn_chains(
                            enriched_body, fmt,
                            pre_filter_body=_pre_filter_body,
                            protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                            turn_tag_index=state.engine._turn_tag_index,
                            store=state.engine._store,
                            conversation_id=state.engine.config.conversation_id,
                            deep_compaction_ratio=_deep_ratio,
                            client_truncated=_client_truncated,
                        )
                        if _sv_collapse:
                            _tool_stubs_present = True
                            logger.info("SAFETY-VALVE CHAIN-COLLAPSE: collapsed %d chains", _sv_collapse)
                        enriched_body, _sv_stub, _sv_stub_refs = stub_tool_outputs_by_position(
                            enriched_body, fmt,
                            protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                            turn_tag_index=state.engine._turn_tag_index,
                            store=state.engine._store,
                            conversation_id=state.engine.config.conversation_id,
                            protected_intrusion_threshold=0.6,
                            context_budget=state.engine.config.monitor.context_window,
                        )
                        if _sv_stub:
                            _tool_stubs_present = True
                            logger.info("SAFETY-VALVE TOOL-STUB: stubbed %d outputs", _sv_stub)
                    else:
                        from .message_filter import stub_tool_outputs_by_position
                        enriched_body, _sv_stub, _sv_stub_refs = stub_tool_outputs_by_position(
                            enriched_body, fmt,
                            protected_recent_turns=state.engine.config.monitor.protected_recent_turns,
                            turn_tag_index=state.engine._turn_tag_index,
                            store=state.engine._store,
                            conversation_id=state.engine.config.conversation_id,
                            protected_intrusion_threshold=0.6,
                            context_budget=state.engine.config.monitor.context_window,
                        )
                        if _sv_stub:
                            _tool_stubs_present = True
                            logger.info("SAFETY-VALVE TOOL-STUB: stubbed %d outputs", _sv_stub)
            except (TypeError, ValueError, AttributeError):
                pass

            # Re-serialize to get new outbound size
            fmt.strip_vc_markers(enriched_body)
            _outbound_json, _outbound_bytes, outbound_tokens = _serialize_payload_and_estimate(
                enriched_body,
                serialize_stage="serialize_outbound_pass2",
                count_stage="count_outbound_tokens_pass2",
                cache_scope="outbound_pass2",
            )
            state._last_enriched_payload_kb = round(_outbound_bytes / 1024, 1)
            logger.info(
                "SAFETY-VALVE: payload after mutations: %dt (was %dt before)",
                outbound_tokens, _size_limit,
            )

    # 2-to-llm: exact payload sent to the LLM (after strip — byte-for-byte what goes upstream)
    if log_dir and log_prefix:
        try:
            _llm_log_stage = time.monotonic()
            _to_llm_log = log_dir / f"{log_prefix}.2-to-llm.json"
            _to_llm_log.write_text(_outbound_json)
            _note_prep("write_to_llm_log", _llm_log_stage)
        except Exception:
            logger.debug("enriched body log write failed", exc_info=True)

    # Ground truth: inbound tokens (what the client sent us, measured above)
    inbound_tokens = _inbound_tokens

    # ------------------------------------------------------------------
    # PAYLOAD SANITY CHECK — fire alarms for anything abnormal
    # ------------------------------------------------------------------
    if state and outbound_tokens > 0:
        try:
            _budget = int(state.engine.config.monitor.context_window)
        except (TypeError, ValueError):
            _budget = 0
        _sanity_msgs = enriched_body.get("messages", enriched_body.get("input", enriched_body.get("contents", [])))
        if isinstance(_sanity_msgs, list):
            _sys_t, _tools_t = _current_outbound_floor_components(enriched_body)
            _nv_floor = _sys_t + _tools_t
            _nv_pct = (_nv_floor / _budget * 100) if _budget else 0

            # 1. Total payload exceeds budget
            if _budget and outbound_tokens > _budget:
                logger.warning(
                    "SANITY_OVER_BUDGET: outbound=%dt exceeds budget=%dt (%.0f%%)",
                    outbound_tokens, _budget, outbound_tokens / _budget * 100,
                )

            # 2. Non-virtualizable floor too high
            if _nv_pct > 40:
                logger.warning(
                    "SANITY_HIGH_FLOOR: system=%dt + tools=%dt = %dt (%.0f%% of budget %dt)",
                    _sys_t, _tools_t, _nv_floor, _nv_pct, _budget,
                )

            # 3. Protected zone size
            _prot_msgs = min(12, len(_sanity_msgs))
            _prot_bytes = sum(
                len(m.get("content", "")) if isinstance(m.get("content", ""), str)
                else len(json.dumps(m.get("content", [])))
                for m in _sanity_msgs[-_prot_msgs:]
            )
            _prot_t = _prot_bytes // 4
            _prot_pct = (_prot_t / _budget * 100) if _budget else 0
            if _prot_pct > 50:
                logger.warning(
                    "SANITY_BLOATED_PROTECTED: last %d msgs = %dt (%.0f%% of budget %dt)",
                    _prot_msgs, _prot_t, _prot_pct, _budget,
                )

            # 4. Individual large messages + unstubbed tool_results
            _large_threshold = _budget // 20  # 5% of budget
            for _si, _sm in enumerate(_sanity_msgs):
                _sc = _sm.get("content", "")
                _s_bytes = len(_sc) if isinstance(_sc, str) else len(json.dumps(_sc))
                _s_tokens = _s_bytes // 4
                if _s_tokens > _large_threshold:
                    _s_role = _sm.get("role", "?")
                    logger.warning(
                        "SANITY_LARGE_MSG: msg %d (%s) = %dt (%.0f%% of budget) — threshold %dt",
                        _si, _s_role, _s_tokens, _s_tokens / _budget * 100, _large_threshold,
                    )
                # Check for unstubbed tool_results
                if isinstance(_sc, list):
                    for _sb in _sc:
                        if isinstance(_sb, dict) and _sb.get("type") == "tool_result":
                            _tr_content = _sb.get("content", "")
                            _tr_bytes = len(_tr_content) if isinstance(_tr_content, str) else len(json.dumps(_tr_content))
                            if _tr_bytes > 4096:
                                logger.warning(
                                    "SANITY_UNSTUBBED_TOOL: msg %d tool_result %d bytes — should be stubbed",
                                    _si, _tr_bytes,
                                )

            # 5. Thinking signature bloat
            _think_bytes = 0
            for _sm in _sanity_msgs:
                _sc = _sm.get("content", [])
                if isinstance(_sc, list):
                    for _sb in _sc:
                        if isinstance(_sb, dict) and _sb.get("type") == "thinking":
                            _think_bytes += len(_sb.get("signature", ""))
            _think_t = _think_bytes // 4
            if _think_t > _budget // 10:
                logger.warning(
                    "SANITY_THINKING_BLOAT: %dt in thinking signatures (%.0f%% of budget %dt)",
                    _think_t, _think_t / _budget * 100, _budget,
                )

    _protected_turn_tokens = 0
    _protected_turn_count = 0
    _protected_turn_sums: list[int] = []

    _bloat_fallback = False

    # ------------------------------------------------------------------
    # PAYLOAD BUDGET ENFORCEMENT — hard cap on VC context window
    # ------------------------------------------------------------------
    _budget_enforce_stage = time.monotonic()
    try:
        _context_window_limit = int(state.engine.config.monitor.context_window) if state else 0
    except (TypeError, ValueError):
        _context_window_limit = 0
    if state and _context_window_limit and outbound_tokens > _context_window_limit:
        from .message_filter import enforce_payload_budget
        _budget_window = _context_window_limit
        def _budget_token_estimator(payload: dict) -> int:
            return _estimate_payload_tokens_cached(payload, cache_scope="outbound")
        enriched_body, _budget_reductions, _budget_freed = enforce_payload_budget(
            enriched_body, fmt, _budget_window,
            store=state.engine._store,
            conversation_id=state.engine.config.conversation_id,
            initial_tokens=outbound_tokens,
            token_estimator=_budget_token_estimator,
        )
        if _budget_reductions > 0:
            _outbound_json, _outbound_bytes, outbound_tokens = _serialize_payload_and_estimate(
                enriched_body,
                serialize_stage="serialize_outbound",
                count_stage="count_outbound_tokens",
                cache_scope="outbound",
            )
            logger.info(
                "BUDGET_ENFORCE: %d reductions, %d bytes freed, now %dt/%dt",
                _budget_reductions, _budget_freed, outbound_tokens, _budget_window,
            )
    _note_prep("budget_enforce", _budget_enforce_stage)

    # ------------------------------------------------------------------
    # FILL PASS — replenish from VC floor to soft threshold
    # ------------------------------------------------------------------
    _fill_stage = time.monotonic()
    if state and not _bloat_fallback:
        _fill_enabled = getattr(
            state.engine.config.monitor, "fill_pass_enabled", False,
        )
        _defer_enabled = getattr(
            state.engine.config.monitor, "defer_payload_mutation", False,
        )
        if _fill_enabled and _defer_enabled:
            logger.warning(
                "CONFIG CONFLICT: fill_pass_enabled=True AND defer_payload_mutation=True — "
                "fill pass backfills released turns into the payload, breaking the cache prefix "
                "that defer is trying to preserve. Disable fill_pass_enabled or defer_payload_mutation."
            )
        if _fill_enabled and outbound_tokens < inbound_tokens:
            from .message_filter import fill_pass

            _soft = state.engine.config.monitor.soft_threshold
            _fill_target_str = getattr(
                state.engine.config.monitor, "fill_pass_target", "soft",
            )
            if _fill_target_str == "soft":
                _fill_target = int(state.engine.config.monitor.context_window * _soft)
            elif _fill_target_str == "hard":
                _fill_target = int(state.engine.config.monitor.context_window * state.engine.config.monitor.hard_threshold)
            else:
                try:
                    _fill_target = int(state.engine.config.monitor.context_window * float(_fill_target_str))
                except (ValueError, TypeError):
                    _fill_target = int(state.engine.config.monitor.context_window * _soft)

            # Clamp: never exceed inbound, never exceed upstream - max_tokens
            _max_tokens = body.get("max_tokens", 0) or 0
            _fill_target = min(_fill_target, inbound_tokens, _upstream_limit - _max_tokens)

            _summary_ratio = getattr(
                state.engine.config.monitor, "fill_pass_summary_ratio", 0.60,
            )

            if _fill_target > outbound_tokens:
                enriched_body, _fill_summaries, _fill_turns = fill_pass(
                    body=enriched_body,
                    fmt=fmt,
                    outbound_tokens=outbound_tokens,
                    target_tokens=_fill_target,
                    assembled=assembled,
                    pre_filter_body=_pre_filter_body,
                    store=state.engine._store,
                    conversation_id=state.engine.config.conversation_id,
                    summary_ratio=_summary_ratio,
                    client_truncated=_client_truncated,
                    turn_tag_index=state.engine._turn_tag_index,
                )
                if _client_truncated:
                    _recovery_turns = _fill_turns
                if _fill_summaries or _fill_turns:
                    _outbound_json, _outbound_bytes, outbound_tokens = _serialize_payload_and_estimate(
                        enriched_body,
                        serialize_stage="serialize_outbound",
                        count_stage="count_outbound_tokens",
                        cache_scope="outbound",
                    )
                    # Update prepend_text to reflect fill additions so
                    # context_tokens and persisted metrics are accurate.
                    _sys = enriched_body.get("system", enriched_body.get("instructions", ""))
                    if isinstance(_sys, str):
                        prepend_text = _sys
                    elif isinstance(_sys, list):
                        prepend_text = "\n".join(
                            b.get("text", "") for b in _sys
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
    _note_prep("fill_pass", _fill_stage)

    # VC must never send more than the client sent. If enrichment bloated
    # the payload beyond inbound, revert to the original client body and
    # treat as passthrough — all downstream metrics/capture will record
    # the passthrough values.
    _bloat_stage = time.monotonic()
    _proxy_cfg = getattr(getattr(getattr(state, "engine", None), "config", None), "proxy", None)
    _enforce_client_payload_ceiling = (
        getattr(_proxy_cfg, "enforce_client_payload_ceiling", False) is True
    )
    if _enforce_client_payload_ceiling and outbound_tokens > inbound_tokens:
        logger.warning(
            "VC_BLOAT_FALLBACK: enriched %dt > inbound %dt — reverting to passthrough (delta +%dt)",
            outbound_tokens, inbound_tokens, outbound_tokens - inbound_tokens,
        )
        _bloat_fallback = True
        enriched_body = _pre_filter_body
        body = _pre_filter_body
        _outbound_json, _outbound_bytes, outbound_tokens = _serialize_payload_and_estimate(
            enriched_body,
            serialize_stage="serialize_outbound",
            count_stage="count_outbound_tokens",
            cache_scope="outbound",
        )
        prepend_text = ""
        context_tokens = 0
        turns_dropped = 0
        turns_stubbed = 0
        _fill_summaries = 0
        _fill_turns = 0
        paging_enabled = False
        tool_output_find_quote = False
        _tool_stubs_present = False
        assembled = None
    _note_prep("bloat_fallback", _bloat_stage)

    # Legacy aliases for downstream consumers
    input_tokens = outbound_tokens
    raw_input_tokens = inbound_tokens

    # Track enriched payload tokens for dashboard — outbound_tokens is ground truth
    _non_virtualizable_floor = 0
    _vc_tokens = 0
    _floor_stage = time.monotonic()
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
            _sys_t, _tools_t = _current_outbound_floor_components(enriched_body)
            _non_virtualizable_floor = _sys_t + _tools_t
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
    _note_prep("floor_accounting", _floor_stage)

    # Upstream context enforcement — trim if enriched payload exceeds upstream limit
    _upstream_trimmed = 0
    _pre_trim_tokens = outbound_tokens
    _prompt_limit = _upstream_limit - body.get("max_tokens", 4096)
    _upstream_trim_stage = time.monotonic()
    if outbound_tokens > _prompt_limit:
        from .message_filter import trim_to_upstream_limit
        enriched_body, _upstream_trimmed = trim_to_upstream_limit(enriched_body, _upstream_limit, fmt)
        if _upstream_trimmed:
            _outbound_json, _outbound_bytes, outbound_tokens = _serialize_payload_and_estimate(
                enriched_body,
                serialize_stage="serialize_outbound",
                count_stage="count_outbound_tokens",
                cache_scope="outbound",
            )
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
    _note_prep("upstream_trim", _upstream_trim_stage)
    if outbound_tokens > _prompt_limit:
        _overflow = outbound_tokens - _prompt_limit
        logger.error(
            "UPSTREAM_LIMIT_EXCEEDED: payload=%dt still exceeds prompt_limit=%dt by %dt after trim "
            "(trimmed_pairs=%d, context_window=%dt). Continuing with oversized payload.",
            outbound_tokens,
            _prompt_limit,
            _overflow,
            _upstream_trimmed,
            _upstream_limit,
        )
        metrics.record({
            "type": "upstream_limit_exceeded",
            "final_tokens": outbound_tokens,
            "prompt_limit": _prompt_limit,
            "upstream_limit": _upstream_limit,
            "overflow": _overflow,
            "pairs_trimmed": _upstream_trimmed,
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
    _conversation_id = state.engine.config.conversation_id if state else ""
    _context_tokens_stage = time.monotonic()
    context_tokens = _vc_tokens if state else (fmt._count(prepend_text) if prepend_text else 0)
    _note_prep("context_token_count", _context_tokens_stage)
    _protected_stage = time.monotonic()
    if state and outbound_tokens > 0:
        try:
            _protected_stats = _compute_protected_turn_stats(
                enriched_body,
                fmt,
                int(state.engine.config.monitor.protected_recent_turns),
                message_tokens=_current_outbound_message_tokens(enriched_body),
                turn_groups=_current_outbound_turn_groups(enriched_body),
            )
            _protected_turn_tokens = int(_protected_stats["tokens"])
            _protected_turn_count = int(_protected_stats["count"])
            _protected_turn_sums = list(_protected_stats["turn_tokens"])
        except Exception:
            logger.debug("protected turn accounting failed", exc_info=True)
    _note_prep("protected_turn_accounting", _protected_stage)
    if _protected_turn_tokens >= _PROTECTED_BREAKDOWN_LOG_THRESHOLD_TOKENS:
        logger.info(
            "PROTECTED_BREAKDOWN conv=%s turn=%s count=%d total=%dt turn_tokens=%s outbound=%dt",
            _conversation_id[:12] if _conversation_id else "none",
            turn,
            _protected_turn_count,
            _protected_turn_tokens,
            _protected_turn_sums,
            outbound_tokens,
        )
    total_turns = turn
    _prepare_meta = _prepare_metadata()
    _prepare_meta["payload_accounting"] = dict(_payload_accounting)
    overhead_ms = _prepare_meta["prepare_total_ms"]
    _log_prepare_breakdown(
        total_ms=overhead_ms,
        conversation_id=_conversation_id,
        turn=turn,
        outbound_tokens=outbound_tokens,
    )
    metrics.record({
        "type": "request",
        "turn": turn,
        "turn_id": _turn_id,
        "model": body.get("model", ""),
        "message_preview": user_message[:200],
        "api_format": api_format,
        "streaming": is_streaming,
        "tags": assembled.matched_tags if assembled else [],
        "context_tokens": context_tokens,
        "budget": assembled.budget_breakdown if assembled else {},
        "history_len": len(state.conversation_history) if state else 0,
        "compacted_through": state.engine._engine_state.compacted_through if state else 0,
        "flushed_through": state.engine._engine_state.flushed_through if state else 0,
        "wait_ms": wait_ms,
        "inbound_ms": inbound_ms,
        "overhead_ms": overhead_ms,
        "prepare_total_ms": overhead_ms,
        "prepare_breakdown": _prepare_meta["prepare_breakdown"],
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
        "protected_turn_tokens": _protected_turn_tokens,
        "protected_turn_count": _protected_turn_count,
        "upstream_context_limit": _upstream_limit,
        "passthrough_trim_limit": int(
            _upstream_limit * (state.engine.config.proxy.passthrough_trim_ratio if state else 0.40)
        ),
        "conversation_id": _conversation_id,
    })

    # Persist request context for dashboard recall page
    _request_turn = 0
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
            _request_turn = int(state.engine._store.save_request_context({
                "conversation_id": _conversation_id,
                "request_turn": 0,
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
            }))
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).debug("Failed to save request context: %s", e)

    # Log request to terminal for debugging
    _tags_str = ", ".join(assembled.matched_tags) if assembled else "none"
    _flag_str = ""
    _out_str = f"out={outbound_tokens}t"
    if _upstream_trimmed:
        _out_str = f"out={outbound_tokens}t TRIMMED from {_pre_trim_tokens}t"
    _recovery_str = f" recovery={_recovery_chains}+{_recovery_turns}" if _client_truncated else ""
    logger.info(
        "T%d POST %s stream=%s tags=[%s]%s msgs=%d dropped=%d stubbed=%d fill=%d+%d%s "
        "ctx=%dt in=%dt %s upstream=%dt vc=%sms | %s",
        turn, api_format, is_streaming, _tags_str, _flag_str,
        len(body.get("messages", [])), turns_dropped, turns_stubbed, _fill_summaries, _fill_turns,
        _recovery_str,
        context_tokens, inbound_tokens, _out_str, _upstream_limit,
        overhead_ms, user_message[:60],
    )

    # Capture pre-filter request body for dashboard inspection
    metrics.capture_request(
        turn, _pre_filter_body, api_format,
        turn_id=_turn_id,
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
        message_preview=user_message[:200],
        prepare_total_ms=overhead_ms,
        prepare_breakdown=_prepare_meta["prepare_breakdown"],
        non_virtualizable_floor=_non_virtualizable_floor,
        upstream_context_limit=_upstream_limit,
        passthrough_trim_limit=int(
            _upstream_limit * (state.engine.config.proxy.passthrough_trim_ratio if state else 0.40)
        ),
        system_tokens=system_tokens,
        protected_turn_tokens=_protected_turn_tokens,
        protected_turn_count=_protected_turn_count,
        payload_accounting=_payload_accounting,
    )
    # Capture enriched body (what we actually send to the LLM)
    metrics.capture_enriched(
        turn,
        enriched_body,
        conversation_id=_conversation_id,
        turn_id=_turn_id,
    )

    return PreparedPayload(
        body=body,
        enriched_body=enriched_body,
        conversation_id=_conversation_id,
        is_passthrough=_bloat_fallback,
        turn=turn,
        request_turn=_request_turn,
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
        restore_tool_injected=_restore_tool_injected,
        inbound_bytes=_inbound_bytes,
        outbound_bytes=_outbound_bytes,
        metadata=_prepare_meta,
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

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=0,  # disable keepalive — stale connections cause streaming stalls
            keepalive_expiry=0,
        ),
    )
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
        _base_log_dir = _state_log_dir or _request_log_dir
        if _base_log_dir and isinstance(_base_log_dir, (str, Path)):
            _base_log_dir = Path(_base_log_dir)
            _base_log_dir.mkdir(parents=True, exist_ok=True)
        else:
            if _base_log_dir is not None:
                logger.warning(
                    "DIAG_LOG_DIR state=%r config=%r base=%r type=%s",
                    _state_log_dir, _request_log_dir, _base_log_dir,
                    type(_base_log_dir).__name__,
                )
            _base_log_dir = None
        _effective_log_dir: Path | None = None
        _response_log_path: Path | None = None
        _session_log_path: Path | None = None
        _log_prefix = ""

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

        # Organize logs by conversation_id when available
        if _base_log_dir and body_bytes:
            if inbound_conversation_id:
                _effective_log_dir = _base_log_dir / inbound_conversation_id
            else:
                _effective_log_dir = _base_log_dir
            _effective_log_dir.mkdir(parents=True, exist_ok=True)
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
        nonlocal metrics
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
                    metrics=metrics, turn=_skip_turn, request_turn=0, turn_id=_skip_turn_id,
                    conversation_id=_skip_sid, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=_skip_turn, request_turn=0, turn_id=_skip_turn_id,
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

        if result.vc_command:
            from .handlers import _handle_vc_command
            _tenant_reg = getattr(app.state, "tenant_registry", None)
            _tid = getattr(request.state, "tenant_id", None)
            return await _handle_vc_command(
                result, fmt, state, registry,
                tenant_registry=_tenant_reg, tenant_id=_tid,
            )

        if result.is_passthrough:
            if result.is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, result.body, result.api_format, state,
                    metrics=metrics, turn=result.turn, request_turn=result.request_turn, turn_id=result.turn_id,
                    conversation_id=result.conversation_id,
                    passthrough=True, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                    skip_marker_injection=bool(inbound_conversation_id),
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, result.body, result.api_format, state,
                    metrics=metrics, turn=result.turn, request_turn=result.request_turn, turn_id=result.turn_id,
                    conversation_id=result.conversation_id,
                    passthrough=True, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                    skip_marker_injection=bool(inbound_conversation_id),
                )
        else:
            _intercept_vc_tools = result.paging_enabled or result.tool_output_find_quote or result.restore_tool_injected

            if result.is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, result.enriched_body, result.api_format, state,
                    metrics=metrics, turn=result.turn, request_turn=result.request_turn, turn_id=result.turn_id,
                    overhead_ms=result.overhead_ms,
                    conversation_id=result.conversation_id,
                    response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    paging_enabled=_intercept_vc_tools,
                    request_log_dir=_effective_log_dir,
                    log_prefix=_log_prefix if _effective_log_dir else "",
                    skip_marker_injection=bool(inbound_conversation_id),
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, result.enriched_body, result.api_format, state,
                    metrics=metrics, turn=result.turn, request_turn=result.request_turn, turn_id=result.turn_id,
                    overhead_ms=result.overhead_ms,
                    conversation_id=result.conversation_id,
                    response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                    skip_marker_injection=bool(inbound_conversation_id),
                )

    return app
