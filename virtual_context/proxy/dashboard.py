"""Live web dashboard for the virtual-context proxy.

Serves a self-contained single-page HTML dashboard at ``/dashboard``
and an SSE event stream at ``/dashboard/events``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from ..tui.state import load_replay_prompts
from ..types import Message, StrategyConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .metrics import ProxyMetrics
    from .server import ProxyState

logger = logging.getLogger(__name__)

# Module-level replay state (one replay at a time)
_replay_state: dict = {}


def _build_settings_response(cfg) -> dict:
    """Build the settings JSON from a VirtualContextConfig."""
    strategy = cfg.retriever.strategy_configs.get("default") or StrategyConfig()
    return {
        "readonly": {
            "context_window": cfg.monitor.context_window,
            "tagger_type": cfg.tag_generator.type,
            "tagger_model": cfg.tag_generator.model,
            "summarizer_model": cfg.summarization.model,
            "storage_backend": cfg.storage.backend,
        },
        "compaction": {
            "soft_threshold": cfg.monitor.soft_threshold,
            "hard_threshold": cfg.monitor.hard_threshold,
            "protected_recent_turns": cfg.monitor.protected_recent_turns,
            "min_summary_tokens": cfg.compactor.min_summary_tokens,
            "max_summary_tokens": cfg.compactor.max_summary_tokens,
        },
        "tagging": {
            "context_lookback_pairs": cfg.tag_generator.context_lookback_pairs,
            "context_bleed_threshold": cfg.tag_generator.context_bleed_threshold,
            "broad_heuristic_enabled": cfg.tag_generator.broad_heuristic_enabled,
            "temporal_heuristic_enabled": cfg.tag_generator.temporal_heuristic_enabled,
        },
        "retrieval": {
            "active_tag_lookback": cfg.retriever.active_tag_lookback,
            "anchorless_lookback": cfg.retriever.anchorless_lookback,
            "max_results": strategy.max_results,
            "max_budget_fraction": strategy.max_budget_fraction,
            "include_related": strategy.include_related,
        },
        "assembly": {
            "tag_context_max_tokens": cfg.assembler.tag_context_max_tokens,
            "recent_turns_always_included": cfg.assembler.recent_turns_always_included,
            "context_hint_enabled": cfg.assembler.context_hint_enabled,
            "context_hint_max_tokens": cfg.assembler.context_hint_max_tokens,
        },
        "summarization": {
            "temperature": cfg.summarization.temperature,
        },
    }


def get_dashboard_html() -> str:
    """Return the full self-contained HTML page for the dashboard."""
    return _DASHBOARD_HTML


def register_dashboard_routes(
    app: "FastAPI",
    metrics: "ProxyMetrics",
    state: "ProxyState | None",
    shutdown_event: asyncio.Event | None = None,
    *,
    registry: "object | None" = None,
) -> None:
    """Register ``/dashboard`` and ``/dashboard/events`` routes."""

    _static_dir = Path(__file__).parent / "static"

    @app.get("/dashboard")
    async def dashboard_page():
        return HTMLResponse(get_dashboard_html())

    @app.get("/dashboard/static/{filename}")
    async def dashboard_static(filename: str):
        filepath = _static_dir / filename
        if not filepath.is_file() or ".." in filename:
            return Response(status_code=404)
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return Response(content=filepath.read_bytes(), media_type=media_type)

    @app.get("/favicon.ico")
    async def favicon_ico():
        filepath = _static_dir / "favicon.ico"
        if not filepath.is_file():
            return Response(status_code=404)
        return Response(content=filepath.read_bytes(), media_type="image/x-icon")

    @app.get("/dashboard/events")
    async def dashboard_events(request: Request):
        async def event_stream():
            try:
                # Send snapshot immediately
                snap = metrics.snapshot()
                # Augment with live engine state
                if state:
                    try:
                        engine = state.engine
                        snap["active_tags"] = list(
                            engine._turn_tag_index.get_active_tags(lookback=6)
                        )
                        snap["store_tag_count"] = len(engine._store.get_all_tags())
                        snap["compacted_through"] = getattr(
                            engine, "_compacted_through", 0
                        )
                        snap["history_len"] = len(state.conversation_history)
                        snap["context_window"] = engine.config.monitor.context_window
                        snap["current_session_id"] = engine.config.session_id
                    except Exception:
                        pass

                # Add live sessions from registry
                live_sessions = []
                if registry and hasattr(registry, "_sessions"):
                    for sid, s in registry._sessions.items():
                        try:
                            live_sessions.append(s.live_snapshot())
                        except Exception:
                            pass
                snap["live_sessions"] = live_sessions

                yield f"data: {json.dumps(snap)}\n\n"

                cursor = snap.get("_seq", -1)
                # Use the highest _seq from snapshot events as cursor
                for evt_list in (
                    snap.get("recent_requests", []),
                    snap.get("responses", []),
                    snap.get("compactions", []),
                    snap.get("turn_completes", []),
                    snap.get("ingested_turns", []),
                ):
                    for evt in evt_list:
                        s = evt.get("_seq", -1)
                        if s > cursor:
                            cursor = s

                while True:
                    if await request.is_disconnected():
                        break
                    if shutdown_event and shutdown_event.is_set():
                        break
                    new_events = metrics.events_since(cursor)
                    for evt in new_events:
                        yield f"data: {json.dumps(evt)}\n\n"
                        s = evt.get("_seq", -1)
                        if s > cursor:
                            cursor = s
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/dashboard/sessions")
    async def dashboard_sessions():
        """Return per-session stats from the store as JSON."""
        if not state:
            return JSONResponse({"sessions": [], "current_session_id": ""})

        sessions_raw = await asyncio.to_thread(
            state.engine._store.get_session_stats
        )
        current_id = state.engine.config.session_id

        sessions = []
        for s in sessions_raw:
            sessions.append({
                "session_id": s.session_id,
                "is_current": s.session_id == current_id,
                "segment_count": s.segment_count,
                "total_full_tokens": s.total_full_tokens,
                "total_summary_tokens": s.total_summary_tokens,
                "compression_ratio": s.compression_ratio,
                "distinct_tags": s.distinct_tags,
                "oldest_segment": (
                    s.oldest_segment.isoformat() if s.oldest_segment else None
                ),
                "newest_segment": (
                    s.newest_segment.isoformat() if s.newest_segment else None
                ),
                "compaction_model": s.compaction_model,
            })

        return JSONResponse({
            "sessions": sessions,
            "current_session_id": current_id,
        })

    @app.delete("/dashboard/sessions/{session_id}")
    async def dashboard_delete_session(session_id: str):
        """Delete all stored segments for a session."""
        if not state:
            return JSONResponse(
                {"error": "Engine not initialized"}, status_code=503,
            )
        store = state.engine._store
        if not hasattr(store, "delete_session"):
            return JSONResponse(
                {"error": "Store does not support session deletion"},
                status_code=501,
            )
        deleted = await asyncio.to_thread(store.delete_session, session_id)
        logger.info("Deleted session %s: %d segments removed", session_id, deleted)
        return JSONResponse({"deleted": deleted})

    @app.get("/dashboard/sessions/live")
    async def dashboard_sessions_live():
        """Return real-time session data from the registry (not the store)."""
        live_sessions = []
        if registry and hasattr(registry, "_sessions"):
            for sid, s in registry._sessions.items():
                try:
                    live_sessions.append(s.live_snapshot())
                except Exception:
                    pass
        return JSONResponse(live_sessions)

    @app.post("/dashboard/sessions/{session_id}/passthrough")
    async def dashboard_toggle_passthrough(session_id: str, request: Request):
        """Toggle manual passthrough mode for a session."""
        if not registry or not hasattr(registry, "_sessions"):
            return JSONResponse(
                {"error": "No registry"}, status_code=503,
            )
        s = registry._sessions.get(session_id)
        if not s:
            return JSONResponse(
                {"error": "Session not found"}, status_code=404,
            )
        body = await request.json()
        enabled = body.get("enabled", False)
        s.set_manual_passthrough(enabled)
        return JSONResponse({"ok": True})

    # -----------------------------------------------------------------------
    # Session export
    # -----------------------------------------------------------------------

    @app.get("/dashboard/export")
    async def dashboard_export():
        """Return the full metrics snapshot augmented with engine state."""
        from .. import __version__

        snap = metrics.snapshot()
        snap.pop("type", None)
        snap["version"] = __version__

        if state:
            try:
                engine = state.engine
                # TurnTagIndex — full per-turn tag data
                entries = []
                for entry in engine._turn_tag_index.entries:
                    entries.append({
                        "turn": entry.turn_number,
                        "tags": entry.tags,
                        "primary_tag": entry.primary_tag,
                    })
                snap["turn_tag_index"] = entries

                # Store tags
                snap["store_tags"] = [
                    ts.tag for ts in engine._store.get_all_tags()
                ]

                # Config summary
                cfg = engine.config
                snap["config"] = {
                    "session_id": cfg.session_id,
                    "context_window": cfg.monitor.context_window,
                    "soft_threshold": cfg.monitor.soft_threshold,
                    "hard_threshold": cfg.monitor.hard_threshold,
                    "tagger_type": cfg.tag_generator.type,
                    "tagger_model": cfg.tag_generator.model,
                    "summarizer_model": cfg.summarization.model,
                    "storage_backend": cfg.storage.backend,
                }

                snap["conversation_turns"] = len(
                    state.conversation_history
                ) // 2
                snap["compacted_through"] = getattr(
                    engine, "_compacted_through", 0
                )
            except Exception as e:
                snap["_export_error"] = str(e)

        return JSONResponse(snap)

    # -----------------------------------------------------------------------
    # Request inspection endpoints
    # -----------------------------------------------------------------------

    @app.get("/dashboard/requests")
    async def list_captured_requests():
        """Return summaries of captured request bodies."""
        return JSONResponse(metrics.get_captured_requests_summary())

    @app.get("/dashboard/requests/{turn}")
    async def get_captured_request(turn: int):
        """Return the full raw request body for a specific turn."""
        req = metrics.get_captured_request(turn)
        if req is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(req)

    # -----------------------------------------------------------------------
    # Replay endpoints
    # -----------------------------------------------------------------------

    @app.post("/dashboard/replay/start")
    async def replay_start(request: Request):
        """Start a replay run from a prompts file."""
        if not state:
            return JSONResponse(
                {"error": "Engine not initialized"}, status_code=503,
            )
        if _replay_state.get("running"):
            return JSONResponse(
                {"error": "Replay already running"}, status_code=409,
            )

        if not getattr(state.engine, "_llm_provider", None):
            return JSONResponse(
                {"error": "No LLM provider configured in engine"},
                status_code=503,
            )

        body = await request.json()
        file_path = body.get("file", "")

        if not file_path:
            return JSONResponse(
                {"error": "Missing 'file' field"}, status_code=400,
            )

        p = Path(file_path)
        if not p.exists():
            return JSONResponse(
                {"error": f"File not found: {file_path}"}, status_code=400,
            )

        try:
            prompts = load_replay_prompts(p)
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to load prompts: {e}"}, status_code=400,
            )

        if not prompts:
            return JSONResponse(
                {"error": "No prompts found in file"}, status_code=400,
            )

        cancel = asyncio.Event()
        task = asyncio.create_task(
            _replay_worker(prompts, state, metrics, cancel)
        )

        def _on_replay_done(t: asyncio.Task) -> None:
            exc = t.exception() if not t.cancelled() else None
            if exc:
                logger.error("Replay task failed: %s", exc, exc_info=exc)
                metrics.record({
                    "type": "replay_done",
                    "turns_completed": 0,
                    "total": len(prompts),
                    "status": "error",
                    "error": str(exc),
                })
                _replay_state.clear()

        task.add_done_callback(_on_replay_done)
        _replay_state.update({
            "running": True,
            "task": task,
            "cancel": cancel,
            "turn": 0,
            "total": len(prompts),
        })

        return JSONResponse({
            "status": "started",
            "total_prompts": len(prompts),
        })

    @app.post("/dashboard/replay/stop")
    async def replay_stop():
        """Stop the running replay."""
        cancel = _replay_state.get("cancel")
        if cancel:
            cancel.set()
        return JSONResponse({"status": "stopping"})

    @app.get("/dashboard/replay/status")
    async def replay_status():
        """Return current replay status."""
        if _replay_state.get("running"):
            return JSONResponse({
                "running": True,
                "turn": _replay_state.get("turn", 0),
                "total": _replay_state.get("total", 0),
            })
        return JSONResponse({"running": False})

    @app.post("/dashboard/shutdown")
    async def dashboard_shutdown():
        """Shut down the proxy server."""
        import os
        import signal

        logger.info("Shutdown requested via dashboard")
        # Send SIGINT to ourselves — triggers the same graceful shutdown as Ctrl+C
        os.kill(os.getpid(), signal.SIGINT)
        return JSONResponse({"status": "shutting_down"})

    # -------------------------------------------------------------------
    # Manual compaction endpoint
    # -------------------------------------------------------------------

    @app.post("/dashboard/compact")
    async def dashboard_compact():
        """Trigger manual compaction regardless of thresholds."""
        if not state:
            return JSONResponse(
                {"error": "Engine not initialized"}, status_code=503,
            )

        # Reject if compaction is already running
        if not state._compaction_lock.acquire(blocking=False):
            return JSONResponse(
                {"status": "busy", "message": "Compaction already in progress"},
                status_code=409,
            )

        try:
            # Wait for any pending on_turn_complete to finish
            await asyncio.to_thread(state.wait_for_complete)

            # Run compaction in a thread (accesses engine internals)
            report = await asyncio.to_thread(
                state.engine.compact_manual, state.conversation_history
            )

            if report is None:
                return JSONResponse({
                    "status": "no_action",
                    "message": "Nothing to compact (not enough messages outside protected zone)",
                })

            # Emit compaction event so the dashboard updates live
            if metrics:
                turn = len(state.conversation_history) // 2 - 1
                original_tokens = sum(r.original_tokens for r in report.results)
                summary_tokens = sum(r.summary_tokens for r in report.results)
                metrics.record({
                    "type": "compaction",
                    "turn": turn,
                    "segments": report.segments_compacted,
                    "tokens_freed": report.tokens_freed,
                    "original_tokens": original_tokens,
                    "summary_tokens": summary_tokens,
                    "tags": report.tags,
                    "tag_summaries_built": report.tag_summaries_built,
                    "compacted_through": getattr(
                        state.engine, "_compacted_through", 0
                    ),
                })

            return JSONResponse({
                "status": "compacted",
                "segments": report.segments_compacted,
                "tokens_freed": report.tokens_freed,
                "tags": report.tags,
                "tag_summaries_built": report.tag_summaries_built,
            })
        finally:
            state._compaction_lock.release()

    # -------------------------------------------------------------------
    # Settings endpoints
    # -------------------------------------------------------------------

    @app.get("/dashboard/settings")
    async def dashboard_settings_get():
        """Return current engine config as JSON."""
        if not state:
            return JSONResponse(
                {"error": "Engine not initialized"}, status_code=503,
            )
        return JSONResponse(_build_settings_response(state.engine.config))

    @app.put("/dashboard/settings")
    async def dashboard_settings_put(request: Request):
        """Apply partial config updates to the running engine."""
        if not state:
            return JSONResponse(
                {"error": "Engine not initialized"}, status_code=503,
            )
        body = await request.json()
        cfg = state.engine.config
        strategy = cfg.retriever.strategy_configs.get("default")
        if strategy is None:
            strategy = StrategyConfig()
            cfg.retriever.strategy_configs["default"] = strategy

        # Map of (section, key) -> (target_obj, attr_name, type_cast)
        field_map = {
            ("compaction", "soft_threshold"): (cfg.monitor, "soft_threshold", float),
            ("compaction", "hard_threshold"): (cfg.monitor, "hard_threshold", float),
            ("compaction", "protected_recent_turns"): (cfg.monitor, "protected_recent_turns", int),
            ("compaction", "min_summary_tokens"): (cfg.compactor, "min_summary_tokens", int),
            ("compaction", "max_summary_tokens"): (cfg.compactor, "max_summary_tokens", int),
            ("tagging", "context_lookback_pairs"): (cfg.tag_generator, "context_lookback_pairs", int),
            ("tagging", "context_bleed_threshold"): (cfg.tag_generator, "context_bleed_threshold", float),
            ("tagging", "broad_heuristic_enabled"): (cfg.tag_generator, "broad_heuristic_enabled", bool),
            ("tagging", "temporal_heuristic_enabled"): (cfg.tag_generator, "temporal_heuristic_enabled", bool),
            ("retrieval", "active_tag_lookback"): (cfg.retriever, "active_tag_lookback", int),
            ("retrieval", "anchorless_lookback"): (cfg.retriever, "anchorless_lookback", int),
            ("retrieval", "max_results"): (strategy, "max_results", int),
            ("retrieval", "max_budget_fraction"): (strategy, "max_budget_fraction", float),
            ("retrieval", "include_related"): (strategy, "include_related", bool),
            ("assembly", "tag_context_max_tokens"): (cfg.assembler, "tag_context_max_tokens", int),
            ("assembly", "recent_turns_always_included"): (cfg.assembler, "recent_turns_always_included", int),
            ("assembly", "context_hint_enabled"): (cfg.assembler, "context_hint_enabled", bool),
            ("assembly", "context_hint_max_tokens"): (cfg.assembler, "context_hint_max_tokens", int),
            ("summarization", "temperature"): (cfg.summarization, "temperature", float),
        }

        # Collect updates
        updates = []
        for section, keys in body.items():
            if section == "readonly":
                continue
            if not isinstance(keys, dict):
                return JSONResponse(
                    {"error": f"Section '{section}' must be an object"},
                    status_code=400,
                )
            for key, value in keys.items():
                mapping = field_map.get((section, key))
                if mapping is None:
                    return JSONResponse(
                        {"error": f"Unknown setting: {section}.{key}"},
                        status_code=400,
                    )
                obj, attr, cast = mapping
                try:
                    value = cast(value)
                except (ValueError, TypeError):
                    return JSONResponse(
                        {"error": f"Invalid type for {section}.{key}"},
                        status_code=400,
                    )
                updates.append((obj, attr, value, section, key))

        # Preview-validate cross-field constraints
        preview = {
            "soft_threshold": cfg.monitor.soft_threshold,
            "hard_threshold": cfg.monitor.hard_threshold,
            "min_summary_tokens": cfg.compactor.min_summary_tokens,
            "max_summary_tokens": cfg.compactor.max_summary_tokens,
        }
        for obj, attr, value, section, key in updates:
            if key in preview:
                preview[key] = value

        if preview["soft_threshold"] >= preview["hard_threshold"]:
            return JSONResponse(
                {"error": "soft_threshold must be less than hard_threshold"},
                status_code=400,
            )
        if preview["min_summary_tokens"] > preview["max_summary_tokens"]:
            return JSONResponse(
                {"error": "min_summary_tokens must not exceed max_summary_tokens"},
                status_code=400,
            )

        # Apply updates
        for obj, attr, value, section, key in updates:
            setattr(obj, attr, value)

        # Sync tag_context_max_tokens to retriever when assembler changes
        if any(key == "tag_context_max_tokens" for _, _, _, _, key in updates):
            cfg.retriever.tag_context_max_tokens = cfg.assembler.tag_context_max_tokens

        return JSONResponse(_build_settings_response(cfg))


async def _call_llm(
    client: httpx.AsyncClient,
    prov: dict,
    system_text: str,
    messages: list[dict],
) -> str:
    """Call the LLM in the correct format (OpenAI or Anthropic)."""
    if prov["format"] == "anthropic":
        payload = {
            "model": prov["model"],
            "max_tokens": 4096,
            "messages": messages,
        }
        if system_text:
            context_block = (
                f"<virtual-context>\n{system_text}\n</virtual-context>"
            )
            payload["system"] = context_block
        resp = await client.post(prov["url"], headers=prov["headers"], json=payload)
        data = resp.json()
        content = data.get("content", [])
        parts = [b["text"] for b in content if b.get("type") == "text"]
        return "".join(parts)
    else:
        # OpenAI-compatible
        msgs = []
        if system_text:
            context_block = (
                f"<virtual-context>\n{system_text}\n</virtual-context>"
            )
            msgs.append({"role": "system", "content": context_block})
        msgs.extend(messages)
        payload = {
            "model": prov["model"],
            "messages": msgs,
            "stream": False,
        }
        resp = await client.post(prov["url"], headers=prov["headers"], json=payload)
        data = resp.json()
        choices = data.get("choices", [])
        return choices[0].get("message", {}).get("content", "") if choices else ""


def _get_provider_config(engine) -> dict:
    """Extract LLM call config from the engine's provider."""
    from ..providers.generic_openai import GenericOpenAIProvider

    provider = engine._llm_provider
    try:
        from ..providers.anthropic import AnthropicProvider
        if isinstance(provider, AnthropicProvider):
            return {
                "format": "anthropic",
                "url": "https://api.anthropic.com/v1/messages",
                "model": provider.model,
                "headers": {
                    "x-api-key": provider.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            }
    except ImportError:
        pass

    # GenericOpenAI (Ollama, OpenRouter, vLLM, etc.)
    return {
        "format": "openai",
        "url": f"{provider.base_url}/chat/completions",
        "model": provider.model,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.api_key}",
        },
    }


async def _replay_worker(
    prompts: list[str],
    state: "ProxyState",
    metrics: "ProxyMetrics",
    cancel: asyncio.Event,
) -> None:
    """Execute prompts through the proxy engine, calling the configured LLM."""
    total = len(prompts)
    completed = 0

    try:
        prov = _get_provider_config(state.engine)
        logger.info(
            "Replay starting: %d prompts, provider=%s, model=%s, url=%s",
            total, prov["format"], prov["model"], prov["url"],
        )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) as client:
            for i, prompt in enumerate(prompts):
                if cancel.is_set():
                    break

                t_start = time.monotonic()

                # 1. Wait for previous on_turn_complete
                t0 = time.monotonic()
                await asyncio.to_thread(state.wait_for_complete)
                wait_ms = round((time.monotonic() - t0) * 1000, 1)

                # 2. Append user message to history
                state.conversation_history.append(
                    Message(role="user", content=prompt)
                )

                # 3. Engine: tag + retrieve + assemble
                t1 = time.monotonic()
                assembled = await asyncio.to_thread(
                    state.engine.on_message_inbound,
                    prompt,
                    state.conversation_history,
                )
                inbound_ms = round((time.monotonic() - t1) * 1000, 1)

                system_text = assembled.prepend_text

                # 4. Filter history by tag relevance
                filtered = state.engine.filter_history(
                    state.conversation_history,
                    current_tags=assembled.matched_tags,
                    broad=assembled.broad,
                    temporal=assembled.temporal,
                )

                # 5. Build messages for LLM
                api_messages = []
                for m in filtered:
                    api_messages.append({"role": m.role, "content": m.content})

                # Brief mode: short answers for stress testing
                if api_messages and api_messages[-1]["role"] == "user":
                    api_messages[-1] = dict(api_messages[-1])
                    api_messages[-1]["content"] += "\n\n(Answer in 2 lines.)"

                # 6. Call LLM (format depends on provider)
                try:
                    assistant_text = await _call_llm(
                        client, prov, system_text, api_messages,
                    )
                except Exception as e:
                    logger.error("Replay LLM error at turn %d: %s", i, e)
                    assistant_text = f"[replay error: {e}]"

                # 7. Append assistant response
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text)
                )

                # 8. Emit request event (same shape as catch_all)
                turn = len(state.conversation_history) // 2 - 1
                context_tokens = len(system_text) // 4 if system_text else 0
                total_turns = len(state.conversation_history) // 2
                filtered_turns = len(filtered) // 2
                # Estimate total payload tokens (system + filtered messages)
                payload_chars = len(system_text)
                for m in filtered:
                    payload_chars += len(m.content)
                input_tokens = payload_chars // 4
                metrics.record({
                    "type": "request",
                    "turn": turn,
                    "message_preview": prompt[:60],
                    "api_format": prov["format"],
                    "streaming": False,
                    "tags": assembled.matched_tags,
                    "broad": assembled.broad,
                    "temporal": assembled.temporal,
                    "context_tokens": context_tokens,
                    "budget": assembled.budget_breakdown,
                    "history_len": len(state.conversation_history),
                    "compacted_through": getattr(
                        state.engine, "_compacted_through", 0
                    ),
                    "wait_ms": wait_ms,
                    "inbound_ms": inbound_ms,
                    "total_turns": total_turns,
                    "filtered_turns": filtered_turns,
                    "input_tokens": input_tokens,
                })

                # 9. Fire on_turn_complete in background
                state.fire_turn_complete(list(state.conversation_history))

                completed = i + 1
                _replay_state["turn"] = completed

                # 10. Emit progress event
                elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
                metrics.record({
                    "type": "replay_progress",
                    "turn": completed,
                    "total": total,
                    "prompt_preview": prompt[:80],
                    "elapsed_ms": elapsed_ms,
                })

        # Wait for final on_turn_complete
        await asyncio.to_thread(state.wait_for_complete)

    except Exception as e:
        logger.error("Replay worker error: %s", e, exc_info=True)
        metrics.record({
            "type": "replay_done",
            "turns_completed": completed,
            "total": total,
            "status": "error",
            "error": str(e),
        })
        return
    finally:
        _replay_state.clear()

    status = "stopped" if cancel.is_set() else "complete"
    logger.info("Replay %s: %d/%d turns", status, completed, total)
    metrics.record({
        "type": "replay_done",
        "turns_completed": completed,
        "total": total,
        "status": status,
    })


# ---------------------------------------------------------------------------
# Self-contained HTML dashboard (loaded from dashboard.html)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = (Path(__file__).with_name("dashboard.html")).read_text(encoding="utf-8")
