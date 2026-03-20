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
from ..types import Message, SplitResult  # noqa: F401 — re-exported

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

# --- Re-export state/registry/handlers for backward compatibility ---
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
        if shared_engine is not None:
            engine = shared_engine
        else:
            engine = VirtualContextEngine(config_path=config_path)

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
                        engine.config.conversation_id = latest.conversation_id
                        engine._load_persisted_state()
                        logger.info(
                            "Lossless restart: restored conversation %s (%d turns, compacted=%d)",
                            latest.conversation_id[:12], len(latest.turn_tag_entries),
                            latest.compacted_through,
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

        # If lossless restart recovered state, mark session as already ingested
        # so the proxy doesn't re-ingest history on first request.
        if engine._turn_tag_index.entries:
            default_state._ingested_conversations.add(engine.config.conversation_id)

        # Build registry and pre-register the default session
        registry = SessionRegistry(
            config_path=config_path,
            upstream=upstream,
            metrics=metrics,
            store=engine._store,
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
            pass  # engine may be a mock in tests

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
    # DIAGNOSTIC: set to True to bypass all VC processing (pure passthrough)
    app.state._force_passthrough = bool(os.environ.get("VC_FORCE_PASSTHROUGH"))
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
            if _req_upstream != upstream:
                logger.warning("Rejected upstream override: %s (allowed: %s)", _req_upstream, upstream)
                _req_upstream = None
            else:
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

        # --- DIAGNOSTIC: full passthrough (bypass all VC processing) ---
        # Remove this block after testing.
        if getattr(app.state, "_force_passthrough", False):
            try:
                _pt_body = json.loads(body_bytes)
                _pt_stream = _pt_body.get("stream", "NOT SET")
                _pt_msgs = len(_pt_body.get("messages", []))
                _pt_cm = "YES" if _pt_body.get("context_management") else "NO"
                _pt_kb = round(len(body_bytes) / 1024, 1)
                logger.info("PASSTHROUGH stream=%s msgs=%d cm=%s payload=%sKB", _pt_stream, _pt_msgs, _pt_cm, _pt_kb)
            except Exception:
                pass
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)
        # --- END DIAGNOSTIC ---

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

        # Use accurate token counter from engine when available (e.g. tiktoken)
        if state and hasattr(state.engine, "_token_counter"):
            _tc = state.engine._token_counter
            if callable(_tc) and not hasattr(_tc, '_mock_name'):
                fmt.set_token_counter(_tc)

        api_format = fmt.name
        user_message = fmt.extract_user_message(body)
        is_streaming = body.get("stream", False)

        # Ground truth: actual byte-measured inbound token count
        _payload_kb = round(len(body_bytes) / 1024, 1)
        _inbound_bytes = len(body_bytes)
        _inbound_tokens = fmt._count(body_bytes.decode("utf-8", errors="replace"))
        if state:
            state._last_payload_kb = _payload_kb
            state._last_payload_tokens = _inbound_tokens
            if state._initial_payload_kb is None:
                state._initial_payload_kb = _payload_kb
                state._initial_payload_tokens = _inbound_tokens

        import datetime as _dt
        _now = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _msg_count = len(fmt.get_messages(body))
        _sid = state.engine.config.conversation_id[:12] if state else "none"
        logger.info("%s POST /%s msgs=%d stream=%s conversation=%s payload=%sKB", _now, path, _msg_count, is_streaming, _sid, _payload_kb)

        if not user_message:
            # Tool-result or non-text turn — skip VC enrichment but
            # preserve streaming so the client SDK doesn't break.
            _skip_sid = state.engine.config.conversation_id if state else ""
            _skip_turn = len(state.engine._turn_tag_index.entries) if state else 0
            _skip_turn_id = uuid.uuid4().hex[:12]
            if is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=_skip_turn, turn_id=_skip_turn_id,
                    conversation_id=_skip_sid, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
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
                existing = len(state.engine._turn_tag_index.entries)
                if needed > 0 and existing < needed:
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

                # Record passthrough request event
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
                    "input_tokens": 0,
                    "raw_input_tokens": 0,
                    "system_tokens": 0,
                    "turns_dropped": 0,
                    "conversation_id": _conversation_id,
                    "passthrough": True,
                })

                metrics.capture_request(
                    turn, body, api_format,
                    conversation_id=_conversation_id,
                    passthrough=True,
                    inbound_tokens=_inbound_tokens,
                    outbound_tokens=_inbound_tokens,  # passthrough: same as inbound
                    inbound_bytes=_inbound_bytes,
                    outbound_bytes=_inbound_bytes,  # passthrough: same as inbound
                    message_preview=user_message[:60],
                )

                # Tool output interception applies even in passthrough —
                # truncating large tool_result blocks reduces upstream tokens
                # regardless of whether VC context is being injected.
                if state.engine.config.tool_output.enabled:
                    from .tool_output_interceptor import ToolOutputInterceptor

                    _pt_interceptor = ToolOutputInterceptor(
                        config=state.engine.config.tool_output,
                        store=state.engine._store,
                        conversation_id=state.engine.config.conversation_id,
                    )
                    _pt_interceptor._turn_counter = state._total_requests
                    _pre_stats = _pt_interceptor.stats.total_intercepted
                    body = _pt_interceptor.process(body, fmt)
                    _post_stats = _pt_interceptor.stats.total_intercepted
                    if _post_stats > _pre_stats:
                        logger.info("TOOL-INTERCEPT Passthrough: truncated %d tool_result(s), saved %dB",
                                    _post_stats - _pre_stats,
                                    _pt_interceptor.stats.total_bytes_original - _pt_interceptor.stats.total_bytes_returned)

                logger.info(
                    "T%d PASSTHROUGH %s stream=%s state=%s | %s",
                    turn, api_format, is_streaming, current_state.value,
                    user_message[:60],
                )

                if is_streaming:
                    return await _handle_streaming(
                        client, url, fwd_headers, body, api_format, state,
                        metrics=metrics, turn=turn, turn_id=_turn_id,
                        conversation_id=_conversation_id,
                        passthrough=True, response_log_path=_response_log_path,
                        session_log_path=_session_log_path,
                    )
                else:
                    return await _handle_non_streaming(
                        client, url, fwd_headers, body, api_format, state,
                        metrics=metrics, turn=turn, turn_id=_turn_id,
                        conversation_id=_conversation_id,
                        passthrough=True, response_log_path=_response_log_path,
                        session_log_path=_session_log_path,
                        request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
                    )

        # ---------------------------------------------------------------
        # ACTIVE path: full enrichment
        # ---------------------------------------------------------------
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

                # Compute available headroom for VC context injection
                _available_for_vc: int | None = None
                try:
                    _upstream_limit = int(state.engine.config.proxy.upstream_context_limit)
                    _output_budget = body.get("max_tokens", 4096)
                    _overhead = 5000  # tools, XML wrappers, safety margin
                    _available_for_vc = max(0, _upstream_limit - _inbound_tokens - _output_budget - _overhead)
                except (TypeError, ValueError, AttributeError):
                    pass  # headroom unknown — assembler uses default budget

                t1 = time.monotonic()
                assembled = await asyncio.to_thread(
                    state.engine.on_message_inbound,
                    user_message,
                    state.conversation_history,
                    body.get("model", ""),
                    max_context_tokens=_available_for_vc,
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
            if state and int(state.engine._compacted_through) > 0:
                from .message_filter import stub_compacted_messages
                body, turns_stubbed = stub_compacted_messages(
                    body,
                    state.engine._turn_tag_index,
                    state.engine._compacted_through,
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
            _pre_compaction = getattr(state.engine, "_compacted_through", 0) == 0
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

        # Tool output interception: truncate large tool_result blocks.
        if state and state.engine.config.tool_output.enabled:
            from .tool_output_interceptor import ToolOutputInterceptor

            interceptor = ToolOutputInterceptor(
                config=state.engine.config.tool_output,
                store=state.engine._store,
                conversation_id=state.engine.config.conversation_id,
            )
            interceptor._turn_counter = state._total_requests
            _pre = interceptor.stats.total_intercepted
            body = interceptor.process(body, fmt)
            _post = interceptor.stats.total_intercepted
            if _post > _pre:
                logger.info("TOOL-INTERCEPT Active: truncated %d tool_result(s), saved %dB",
                            _post - _pre,
                            interceptor.stats.total_bytes_original - interceptor.stats.total_bytes_returned)

        enriched_body = _inject_context(body, prepend_text, api_format)

        # Inject VC paging tools for autonomous mode (formats that support it)
        paging_enabled = False
        if (
            state
            and fmt.supports_tool_interception
            and state.engine.config.paging.enabled
        ):
            _paging_mode = state.engine._resolve_paging_mode(
                enriched_body.get("model", ""),
            )
            if _paging_mode == "autonomous":
                tool_turn_count = len(state.engine._turn_tag_index.entries)
                try:
                    compacted_count = int(getattr(state.engine, "_compacted_through", 0))
                except (TypeError, ValueError):
                    compacted_count = 0
                require_tools = compacted_count > 0
                enriched_body = _inject_vc_tools(
                    enriched_body,
                    state.engine,
                    require_tool_use=require_tools,
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

        # Track enriched payload size
        if state:
            state._last_enriched_payload_kb = round(len(json.dumps(enriched_body)) / 1024, 1)

        # 2-to-llm: enriched body sent to the LLM (after filtering + context + tools)
        if _effective_log_dir and _log_prefix:
            try:
                _to_llm_log = _effective_log_dir / f"{_log_prefix}.2-to-llm.json"
                _to_llm_log.write_text(json.dumps(enriched_body, default=str))
            except Exception:
                pass

        is_streaming = body.get("stream", False)

        # Component-level estimate (diagnostic breakdown, not source of truth)
        system_tokens = fmt._estimate_system_tokens(body)

        # Ground truth: actual byte-measured outbound token count
        _outbound_json = json.dumps(enriched_body, default=str)
        _outbound_bytes = len(_outbound_json.encode("utf-8"))
        outbound_tokens = fmt._count(_outbound_json)

        # Ground truth: inbound tokens (what the client sent us, measured above)
        inbound_tokens = _inbound_tokens

        # Legacy aliases for downstream consumers
        input_tokens = outbound_tokens
        raw_input_tokens = inbound_tokens

        # Track enriched payload tokens for dashboard — outbound_tokens is ground truth
        if state:
            state._last_enriched_payload_tokens = outbound_tokens

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
        context_tokens = len(prepend_text) // 4 if prepend_text else 0
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
            "compacted_through": getattr(
                state.engine, "_compacted_through", 0
            ) if state else 0,
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
            "conversation_id": _conversation_id,
        })

        # Log request to terminal for debugging
        _tags_str = ", ".join(assembled.matched_tags) if assembled else "none"
        _flag_str = ""
        logger.info(
            "T%d POST %s stream=%s tags=[%s]%s msgs=%d dropped=%d stubbed=%d "
            "ctx=%dt in=%dt out=%dt vc=%sms | %s",
            turn, api_format, is_streaming, _tags_str, _flag_str,
            len(body.get("messages", [])), turns_dropped, turns_stubbed,
            context_tokens, inbound_tokens, outbound_tokens,
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
        )
        # Capture enriched body (what we actually send to the LLM)
        metrics.capture_enriched(turn, enriched_body)

        _intercept_vc_tools = paging_enabled or tool_output_find_quote

        if is_streaming:
            return await _handle_streaming(
                client, url, fwd_headers, enriched_body, api_format, state,
                metrics=metrics, turn=turn, turn_id=_turn_id, overhead_ms=overhead_ms,
                conversation_id=_conversation_id, response_log_path=_response_log_path,
                session_log_path=_session_log_path,
                paging_enabled=_intercept_vc_tools,
                request_log_dir=_effective_log_dir,
                log_prefix=_log_prefix if _effective_log_dir else "",
            )
        else:
            return await _handle_non_streaming(
                client, url, fwd_headers, enriched_body, api_format, state,
                metrics=metrics, turn=turn, turn_id=_turn_id, overhead_ms=overhead_ms,
                conversation_id=_conversation_id, response_log_path=_response_log_path,
                session_log_path=_session_log_path,
                request_log_dir=_effective_log_dir, log_prefix=_log_prefix,
            )

    return app
