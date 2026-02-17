"""Live web dashboard for the virtual-context proxy.

Serves a self-contained single-page HTML dashboard at ``/dashboard``
and an SSE event stream at ``/dashboard/events``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

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
) -> None:
    """Register ``/dashboard`` and ``/dashboard/events`` routes."""

    @app.get("/dashboard")
    async def dashboard_page():
        return HTMLResponse(get_dashboard_html())

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
# Self-contained HTML dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>virtual-context proxy</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --purple: #bc8cff; --orange: #d18616;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: var(--bg); color: var(--text); font-size: 13px;
    line-height: 1.5;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 16px; }

  /* Header */
  header {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 12px; border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
  }
  header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
  header .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  header .status { color: var(--text-dim); font-size: 12px; margin-left: auto; }
  .shutdown-btn {
    background: transparent; border: 1px solid var(--red); color: var(--red);
    border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: 600;
    cursor: pointer; font-family: inherit;
  }
  .shutdown-btn:hover { background: var(--red); color: #fff; }

  /* Stat cards */
  .stats-wrapper { margin-bottom: 16px; }
  .stats {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
  }
  .stat {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; text-align: center;
  }
  .stat .label { font-size: 10px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.05em; }
  .stat .value { font-size: 22px; font-weight: 700; color: var(--text); margin-top: 2px; }

  /* Panels */
  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 14px; margin-bottom: 14px;
  }
  .panel h2 {
    font-size: 12px; text-transform: uppercase; color: var(--text-dim);
    letter-spacing: 0.05em; margin-bottom: 10px;
  }

  /* Memory bar */
  .memory-bar-track {
    height: 14px; background: var(--bg); border-radius: 4px;
    overflow: hidden; margin-bottom: 6px;
  }
  .memory-bar-fill {
    height: 100%; background: var(--green); border-radius: 4px;
    transition: width 0.4s ease, background 0.4s ease;
  }
  .memory-bar-fill.warn { background: var(--yellow); }
  .memory-bar-fill.crit { background: var(--red); }
  .memory-note { font-size: 11px; color: var(--text-dim); }

  /* Pipeline */
  .pipeline-row {
    display: flex; gap: 24px; font-size: 13px; margin-bottom: 4px;
  }
  .pipeline-row span { color: var(--text-dim); }
  .pipeline-row strong { color: var(--text); }

  /* Tags */
  .tag-cloud { display: flex; flex-wrap: wrap; gap: 6px; }
  .tag {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 2px 8px; font-size: 11px; color: var(--accent);
  }
  .store-info { font-size: 11px; color: var(--text-dim); margin-top: 8px; }

  /* Request log table */
  .log-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12px; }
  .log-table th {
    text-align: left; padding: 6px 8px; color: var(--text-dim);
    border-bottom: 1px solid var(--border); font-weight: 500;
    position: sticky; top: 0; background: var(--surface); z-index: 10;
  }
  .log-table td { padding: 5px 8px; border-bottom: 1px solid var(--bg); }
  .log-table tr.flash { animation: row-flash 1.5s ease-out; }
  @keyframes row-flash {
    0% { background: rgba(88,166,255,0.15); } 100% { background: transparent; }
  }
  .log-scroll { max-height: 320px; overflow-y: auto; }
  .broad-badge {
    background: var(--purple); color: #fff; border-radius: 3px;
    padding: 1px 5px; font-size: 10px; font-weight: 600;
  }
  .temporal-badge {
    background: var(--orange); color: #fff; border-radius: 3px;
    padding: 1px 5px; font-size: 10px; font-weight: 600;
  }
  .tags-cell { max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .msg-cell { max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-dim); }
  .timing-pending { opacity: 0.3; }
  .timing-vc, .timing-llm, .timing-total { white-space: nowrap; font-variant-numeric: tabular-nums; }
  .ingested-row { opacity: 0.5; }
  .ingested-row:hover { opacity: 0.8; }
  .ingested-badge {
    background: var(--border); color: var(--text-dim); border-radius: 3px;
    padding: 1px 5px; font-size: 10px; font-weight: 600;
  }

  /* Compaction events */
  .compaction-entry { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--bg); }
  .compaction-entry:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
  .compaction-header { font-weight: 600; color: var(--text); }
  .compaction-detail { font-size: 11px; color: var(--text-dim); margin-top: 2px; }

  .empty-state { color: var(--text-dim); font-style: italic; font-size: 12px; }

  /* Session list */
  .session-list { max-height: 280px; overflow-y: auto; }
  .session-card {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 10px 12px; margin-bottom: 8px;
  }
  .session-card:last-child { margin-bottom: 0; }
  .session-card.current { border-color: var(--green); }
  .session-card .session-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
  }
  .session-card .session-id { font-size: 11px; color: var(--accent); }
  .session-card .session-badge {
    background: var(--green); color: #fff; border-radius: 3px;
    padding: 1px 6px; font-size: 10px; font-weight: 600;
  }
  .session-card .session-stats {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;
    font-size: 11px;
  }
  .session-card .session-stats .label { color: var(--text-dim); }
  .session-card .session-stats .val { color: var(--text); font-weight: 600; }
  .session-card .session-tags { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
  .session-card .session-tags .tag { font-size: 10px; padding: 1px 5px; }
  .session-time { font-size: 10px; color: var(--text-dim); margin-top: 4px; }
  .session-delete {
    background: transparent; border: 1px solid var(--red); color: var(--red);
    border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: 600;
    cursor: pointer; font-family: inherit;
  }
  .session-delete:hover { background: var(--red); color: #fff; }

  /* Replay panel */
  .replay-controls { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
  .replay-controls input[type="text"],
  .replay-controls input[type="password"] {
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); padding: 5px 8px; font-size: 12px;
    font-family: inherit; flex: 1;
  }
  .replay-controls input::placeholder { color: var(--text-dim); }
  .replay-controls label { font-size: 11px; color: var(--text-dim); min-width: 50px; }
  .replay-btn {
    background: var(--accent); color: #fff; border: none; border-radius: 4px;
    padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer;
    font-family: inherit;
  }
  .replay-btn:hover { opacity: 0.9; }
  .replay-btn:disabled { opacity: 0.4; cursor: default; }
  .replay-btn.stop { background: var(--red); }
  .replay-progress { margin-top: 8px; }
  .replay-bar-track {
    height: 10px; background: var(--bg); border-radius: 4px;
    overflow: hidden; margin-bottom: 4px;
  }
  .replay-bar-fill {
    height: 100%; background: var(--accent); border-radius: 4px;
    transition: width 0.3s ease; width: 0%;
  }
  .replay-bar-fill.done { background: var(--green); }
  .replay-bar-fill.error { background: var(--red); }
  .replay-status { font-size: 11px; color: var(--text-dim); }

  /* Two-column layout for middle panels */
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }

  /* Cost savings panel */
  .savings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .savings-metric { }
  .savings-metric .label { font-size: 10px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.05em; }
  .savings-metric .value { font-size: 18px; font-weight: 700; margin-top: 1px; }
  .savings-metric .value.green { color: var(--green); }
  .savings-detail { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
  .cost-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; }
  .cost-table th { text-align: left; padding: 4px 8px; color: var(--text-dim); border-bottom: 1px solid var(--border); font-weight: 500; }
  .cost-table td { padding: 4px 8px; border-bottom: 1px solid var(--bg); }
  .cost-table td.saved { color: var(--green); font-weight: 600; }

  /* Compact button */
  .compact-btn {
    background: transparent; border: 1px solid var(--yellow); color: var(--yellow);
    border-radius: 4px; padding: 3px 10px; font-size: 11px; font-weight: 600;
    cursor: pointer; font-family: inherit; white-space: nowrap;
  }
  .compact-btn:hover { background: var(--yellow); color: #000; }
  .compact-btn:disabled { opacity: 0.4; cursor: default; }

  /* Settings button */
  .settings-btn {
    background: transparent; border: 1px solid var(--border); color: var(--text-dim);
    border-radius: 4px; padding: 4px 8px; cursor: pointer; display: flex; align-items: center;
  }
  .settings-btn:hover { color: var(--accent); border-color: var(--accent); }

  /* Settings modal */
  .modal-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.6); z-index: 100;
    display: flex; align-items: center; justify-content: center;
  }
  .modal {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    width: 580px; max-height: 85vh; display: flex; flex-direction: column;
  }
  .modal-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 16px 20px; border-bottom: 1px solid var(--border);
  }
  .modal-header h2 { font-size: 14px; font-weight: 600; color: var(--text); margin: 0; }
  .modal-subtitle { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
  .modal-close {
    background: none; border: none; color: var(--text-dim); font-size: 20px;
    cursor: pointer; padding: 0 4px; line-height: 1;
  }
  .modal-close:hover { color: var(--text); }
  .modal-body { padding: 16px 20px; overflow-y: auto; flex: 1; }
  .modal-footer {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 20px; border-top: 1px solid var(--border);
  }
  .settings-status { font-size: 11px; color: var(--text-dim); }
  .settings-status.error { color: var(--red); }
  .settings-status.ok { color: var(--green); }

  .settings-section { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--bg); }
  .settings-section:last-child { border-bottom: none; margin-bottom: 0; }
  .settings-section h3 {
    font-size: 11px; text-transform: uppercase; color: var(--accent);
    letter-spacing: 0.05em; margin-bottom: 8px; font-weight: 600;
  }
  .settings-row {
    display: flex; align-items: flex-start; margin-bottom: 8px; gap: 12px;
  }
  .settings-row .settings-label {
    flex: 0 0 50%; font-size: 12px; color: var(--text-dim); line-height: 1.3;
    padding-top: 3px; text-align: right;
  }
  .settings-row .settings-value { font-size: 12px; color: var(--text); font-weight: 500; padding-top: 3px; }
  .settings-desc { display: block; font-size: 10px; opacity: 0.65; margin-top: 2px; font-weight: 400; }
  .settings-input {
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); padding: 3px 8px; font-size: 12px; font-family: inherit; width: 100px;
  }
  .settings-input:focus { border-color: var(--accent); outline: none; }
  .settings-slider {
    -webkit-appearance: none; appearance: none; width: 120px; height: 4px;
    background: var(--bg); border-radius: 2px; outline: none;
  }
  .settings-slider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: var(--accent); cursor: pointer;
  }
  .settings-slider-val { font-size: 12px; color: var(--text); min-width: 40px; font-weight: 500; }
  .settings-toggle {
    width: 32px; height: 18px; border-radius: 9px; border: none;
    background: var(--border); cursor: pointer; position: relative;
    transition: background 0.2s;
  }
  .settings-toggle::after {
    content: ''; position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--text-dim); transition: transform 0.2s, background 0.2s;
  }
  .settings-toggle.on { background: var(--green); }
  .settings-toggle.on::after { transform: translateX(14px); background: #fff; }

  .section-header { display: flex; align-items: center; margin-bottom: 8px; }
  .section-header h2, .section-header h3 { margin-bottom: 0; flex: 1; }
  .help-btn {
    width: 18px; height: 18px; border-radius: 50%; border: 1px solid var(--border);
    background: transparent; color: var(--text-dim); font-size: 11px; font-weight: 600;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    font-family: inherit; line-height: 1; flex-shrink: 0;
  }
  .help-btn:hover { color: var(--accent); border-color: var(--accent); }
  .help-btn.active { color: var(--accent); border-color: var(--accent); }
  .help-content {
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    padding: 10px 12px; margin-bottom: 10px; font-size: 11px;
    line-height: 1.6; color: var(--text-dim);
  }
  .help-content p { margin: 0 0 8px 0; }
  .help-content p:last-child { margin-bottom: 0; }
  .help-content strong { color: var(--text); font-weight: 600; }
  .help-content dl { margin: 0; }
  .help-content dt {
    color: var(--text); font-weight: 600; margin-top: 6px;
  }
  .help-content dt:first-child { margin-top: 0; }
  .help-content dd { margin: 1px 0 0 12px; }

  @media (max-width: 700px) {
    .stats { grid-template-columns: repeat(3, 1fr); }
    .two-col { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="container">

<header>
  <div class="dot"></div>
  <h1>virtual-context proxy</h1>
  <span class="status" id="conn-status">connecting...</span>
  <span class="status" id="ingestion-status" style="margin-left:8px;color:var(--accent)"></span>
  <button class="shutdown-btn" style="border-color:var(--accent);color:var(--accent)" onclick="exportSession()">Export</button>
  <button class="settings-btn" onclick="openSettings()" title="Settings">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M9.405 1.05c-.413-1.4-2.397-1.4-2.81 0l-.1.34a1.464 1.464 0 0 1-2.105.872l-.31-.17c-1.283-.698-2.686.705-1.987 1.987l.169.311c.446.82.023 1.841-.872 2.105l-.34.1c-1.4.413-1.4 2.397 0 2.81l.34.1a1.464 1.464 0 0 1 .872 2.105l-.17.31c-.698 1.283.705 2.686 1.987 1.987l.311-.169a1.464 1.464 0 0 1 2.105.872l.1.34c.413 1.4 2.397 1.4 2.81 0l.1-.34a1.464 1.464 0 0 1 2.105-.872l.31.17c1.283.698 2.686-.705 1.987-1.987l-.169-.311a1.464 1.464 0 0 1 .872-2.105l.34-.1c1.4-.413 1.4-2.397 0-2.81l-.34-.1a1.464 1.464 0 0 1-.872-2.105l.17-.31c.698-1.283-.705-2.686-1.987-1.987l-.311.169a1.464 1.464 0 0 1-2.105-.872l-.1-.34zM8 10.93a2.929 2.929 0 1 1 0-5.86 2.929 2.929 0 0 1 0 5.858z"/></svg>
  </button>
  <button class="shutdown-btn" onclick="shutdownProxy()">Shutdown</button>
</header>

<div class="stats-wrapper">
  <div class="section-header"><h2 style="font-size:12px;text-transform:uppercase;color:var(--text-dim);letter-spacing:0.05em">Overview</h2><button class="help-btn" data-help="overview" onclick="toggleHelp(this)">?</button></div>
  <div class="help-content" data-help-for="overview" style="display:none">
    <p>High-level counters for the current proxy session. These update in real time as requests flow through the proxy.</p>
    <dl>
      <dt>Uptime</dt><dd>How long the proxy server has been running since it was started.</dd>
      <dt>Requests</dt><dd>Total number of LLM requests intercepted and enriched by the proxy. Each request has virtual-context summaries injected before forwarding to the upstream provider.</dd>
      <dt>Turns</dt><dd>Number of completed conversation turns (user message + assistant response pairs). Turns drive the tagging and compaction pipeline.</dd>
      <dt>Compactions</dt><dd>Number of compaction events that have fired. Each compaction summarizes older turns, frees token budget, and stores segments for later retrieval.</dd>
      <dt>Freed</dt><dd>Total tokens reclaimed by compaction. These tokens were occupied by raw conversation history and have been replaced by compressed summaries stored externally.</dd>
    </dl>
  </div>
  <div class="stats">
    <div class="stat"><div class="label">Uptime</div><div class="value" id="s-uptime">--</div></div>
    <div class="stat"><div class="label">Requests</div><div class="value" id="s-requests">0</div></div>
    <div class="stat"><div class="label">Turns</div><div class="value" id="s-turns">0</div></div>
    <div class="stat"><div class="label">Compactions</div><div class="value" id="s-compactions">0</div></div>
    <div class="stat"><div class="label">Freed</div><div class="value" id="s-freed">0</div></div>
  </div>
</div>

<div class="panel">
  <div class="section-header"><h2>Replay</h2><button class="help-btn" data-help="replay" onclick="toggleHelp(this)">?</button></div>
  <div class="help-content" data-help-for="replay" style="display:none">
    <p>Run a stress test by replaying a file of user prompts through the engine. Each line in the file becomes one conversation turn, processed through the full pipeline: tagging, retrieval, assembly, LLM call, and on_turn_complete. Use this to validate compaction behavior, retrieval accuracy, and context budget management over many turns.</p>
    <dl>
      <dt>File</dt><dd>Path to a text file with one user prompt per line. The proxy will send each line as a user message, call the configured LLM, and record the full turn.</dd>
      <dt>Start / Stop</dt><dd>Begin or halt the replay. Progress is shown in the bar below. Metrics (request log, compaction events, cost savings) update live as turns complete.</dd>
    </dl>
  </div>
  <div class="replay-controls">
    <label>File</label>
    <input type="text" id="replay-file" placeholder="path/to/prompts.txt" value="prompts100.txt">
    <button class="replay-btn" id="replay-start" onclick="startReplay()">Start</button>
    <button class="replay-btn stop" id="replay-stop" onclick="stopReplay()" disabled>Stop</button>
  </div>
  <div class="replay-progress" id="replay-progress" style="display:none">
    <div class="replay-bar-track"><div class="replay-bar-fill" id="replay-fill"></div></div>
    <div class="replay-status" id="replay-status"></div>
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <div class="section-header"><h2>Memory</h2><button class="help-btn" data-help="memory" onclick="toggleHelp(this)">?</button></div>
    <div class="help-content" data-help-for="memory" style="display:none">
      <p>Visualizes how much of the conversation history has been compacted. The bar fills as the compaction watermark advances through the message history. Green means healthy headroom; yellow and red indicate the engine is compacting aggressively to stay within the context window.</p>
      <dl>
        <dt>Bar</dt><dd>Percentage of total messages that have been compacted. The raw messages are replaced by stored summaries that can be retrieved on demand.</dd>
        <dt>Status line</dt><dd>Shows the compaction watermark (how far compaction has reached), total message count, and the configured context window size.</dd>
      </dl>
    </div>
    <div class="memory-bar-track"><div class="memory-bar-fill" id="mem-fill" style="width:0%"></div></div>
    <div style="display:flex;align-items:center;gap:8px">
      <div class="memory-note" id="mem-note" style="flex:1">--</div>
      <button class="compact-btn" id="compact-btn" onclick="compactNow()">Compact Now</button>
    </div>
  </div>
  <div class="panel">
    <div class="section-header"><h2>Pipeline (avg)</h2><button class="help-btn" data-help="pipeline" onclick="toggleHelp(this)">?</button></div>
    <div class="help-content" data-help-for="pipeline" style="display:none">
      <p>Average latency breakdown for each stage of the request processing pipeline, computed across all requests in this session.</p>
      <dl>
        <dt>wait</dt><dd>Time spent waiting for the previous on_turn_complete to finish (runs in a background thread). High values mean compaction or tagging from the prior turn is slow.</dd>
        <dt>inbound</dt><dd>Time for on_message_inbound: tagging the user message, retrieving matching summaries from the store, and assembling the context block. This is the core virtual-context overhead added to each request.</dd>
        <dt>context</dt><dd>Average number of tokens injected into the LLM request as retrieved context. Higher values mean more stored knowledge is being surfaced; zero means no relevant summaries were found.</dd>
      </dl>
    </div>
    <div class="pipeline-row"><span>wait</span> <strong id="p-wait">--</strong></div>
    <div class="pipeline-row"><span>inbound</span> <strong id="p-inbound">--</strong></div>
    <div class="pipeline-row"><span>context</span> <strong id="p-context">--</strong></div>
  </div>
</div>

<div class="panel">
  <div class="section-header"><h2>Cost Savings</h2><button class="help-btn" data-help="cost" onclick="toggleHelp(this)">?</button></div>
  <div class="help-content" data-help-for="cost" style="display:none">
    <p>Compares total input tokens sent by virtual-context against a naive baseline that sends full history each turn, compacting at 30% ratio when it hits the context window. The baseline gets full credit for its own compaction — savings reflect the combined benefit of VC's filtering, selective retrieval, and enrichment over what a standard system would do.</p>
    <dl>
      <dt>Tokens Freed</dt><dd>Total tokens reclaimed by compaction across all compaction events. These tokens were raw conversation history that has been replaced by shorter summaries stored externally.</dd>
      <dt>Summary Compression</dt><dd>Summary tokens divided by original tokens, as a percentage. Lower is better: 20% means summaries are 5x shorter than the originals. Typical values range from 15% to 30%.</dd>
      <dt>Context Injected</dt><dd>Total tokens of retrieved summaries injected across all requests. This is the knowledge the LLM received from stored context that would otherwise have been lost to compaction.</dd>
      <dt>Avg Context / Request</dt><dd>Average tokens injected per request. Shows how much stored knowledge is being surfaced per turn on average.</dd>
      <dt>Session Efficiency</dt><dd>Percentage of input tokens saved compared to a naive baseline system. The baseline simulates a standard chat that sends full history each turn, compacting at 30% ratio when it hits the context window. Higher is better.</dd>
      <dt>Estimated Saved</dt><dd>Dollar savings estimates at common LLM pricing tiers (per million input tokens). Calculated as the difference between baseline and actual cumulative input tokens. Actual savings depend on your provider and model.</dd>
    </dl>
  </div>
  <div class="savings-grid">
    <div class="savings-metric">
      <div class="label">Tokens Freed</div>
      <div class="value green" id="sv-freed">0</div>
      <div class="savings-detail" id="sv-freed-detail">no compactions yet</div>
    </div>
    <div class="savings-metric">
      <div class="label">Summary Compression</div>
      <div class="value green" id="sv-ratio">--</div>
      <div class="savings-detail" id="sv-ratio-detail">original vs summary</div>
    </div>
    <div class="savings-metric">
      <div class="label">Session Efficiency</div>
      <div class="value green" id="sv-efficiency">--</div>
      <div class="savings-detail" id="sv-efficiency-detail">vs naive baseline</div>
    </div>
    <div class="savings-metric">
      <div class="label">Context Injected</div>
      <div class="value" id="sv-injected">0</div>
      <div class="savings-detail" id="sv-injected-detail">total across all requests</div>
    </div>
    <div class="savings-metric">
      <div class="label">Avg Context / Request</div>
      <div class="value" id="sv-avg-ctx">0</div>
      <div class="savings-detail">enrichment per turn</div>
    </div>
  </div>
  <table class="cost-table" id="cost-table">
    <thead><tr><th>Model Tier</th><th>$/MTok</th><th>Baseline</th><th>VC</th><th>Saved</th><th>%</th></tr></thead>
    <tbody>
      <tr><td>Haiku-class</td><td>$0.25</td><td id="cost-haiku-base">--</td><td id="cost-haiku-vc">--</td><td class="saved" id="cost-haiku">--</td><td class="saved" id="cost-haiku-pct">--</td></tr>
      <tr><td>Sonnet-class</td><td>$3.00</td><td id="cost-sonnet-base">--</td><td id="cost-sonnet-vc">--</td><td class="saved" id="cost-sonnet">--</td><td class="saved" id="cost-sonnet-pct">--</td></tr>
      <tr><td>Opus-class</td><td>$15.00</td><td id="cost-opus-base">--</td><td id="cost-opus-vc">--</td><td class="saved" id="cost-opus">--</td><td class="saved" id="cost-opus-pct">--</td></tr>
    </tbody>
  </table>
</div>

<div class="panel">
  <div class="section-header"><h2>Sessions</h2><button class="help-btn" data-help="sessions" onclick="toggleHelp(this)">?</button></div>
  <div class="help-content" data-help-for="sessions" style="display:none">
    <p>Each proxy run creates a session. When compaction fires, it stores segments tagged with the session ID. This panel shows all sessions that have stored segments, along with their compression stats and tag coverage. The current session is highlighted in green.</p>
    <dl>
      <dt>Segments</dt><dd>Number of compacted segments stored for this session. Each segment represents a group of conversation turns summarized together.</dd>
      <dt>Full Tokens</dt><dd>Total original token count across all segments before compaction.</dd>
      <dt>Compression</dt><dd>Summary-to-original ratio. Lower means more aggressive compression.</dd>
      <dt>Freed</dt><dd>Tokens saved (full tokens minus summary tokens) for this session.</dd>
      <dt>Delete</dt><dd>Remove all stored segments for a past session. Cannot delete the current session.</dd>
    </dl>
  </div>
  <div class="session-list" id="session-list">
    <span class="empty-state">loading sessions...</span>
  </div>
</div>

<div class="panel">
  <div class="section-header"><h2>Active Tags</h2><button class="help-btn" data-help="tags" onclick="toggleHelp(this)">?</button></div>
  <div class="help-content" data-help-for="tags" style="display:none">
    <p>Tags currently in the conversation's working set, based on the most recent turns (controlled by active_tag_lookback). Active tags represent the topics being discussed right now. During retrieval, these tags are <strong>skipped</strong> because their content is already present in the raw conversation history. The store tag count shows how many distinct tags exist across all stored segments.</p>
  </div>
  <div class="tag-cloud" id="tag-cloud"><span class="empty-state">no tags yet</span></div>
  <div class="store-info" id="store-info"></div>
</div>

<div class="panel">
  <div class="section-header"><h2>Request Log</h2><button class="help-btn" data-help="reqlog" onclick="toggleHelp(this)">?</button></div>
  <div class="help-content" data-help-for="reqlog" style="display:none">
    <p>Chronological log of every LLM request processed by the proxy (newest first, max 200 rows). Each row shows one conversation turn from the moment the user message arrives through context injection.</p>
    <dl>
      <dt>T#</dt><dd>Turn number (0-indexed). Corresponds to the conversation turn in the session.</dd>
      <dt>Tags</dt><dd>Semantic tags assigned to this turn by the tagger, or a <strong style="color:var(--purple)">BROAD</strong> / <strong style="color:var(--orange)">TEMPORAL</strong> badge if the query was detected as broad or temporal. Tags update after on_turn_complete finishes.</dd>
      <dt>Message</dt><dd>Preview of the user message (first 50 characters).</dd>
      <dt>Payload</dt><dd>How many turns were included in the LLM payload after tag-based filtering (filtered/total).</dd>
      <dt>Tokens</dt><dd>Estimated total input tokens sent to the upstream provider for this request (system prompt + enriched messages).</dd>
      <dt>Base</dt><dd>Estimated baseline input tokens a naive system would have sent for this turn (system prompt + full raw history, compacting at 30% when hitting the context window). Appears after on_turn_complete finishes. Compare with Tokens to see per-turn savings.</dd>
      <dt>Injected</dt><dd>Number of tokens of retrieved virtual-context summaries injected into this request.</dd>
      <dt>Timing</dt><dd>Three-part breakdown: <strong>VC</strong> = virtual-context overhead (wait + inbound), <strong>LLM</strong> = upstream API round-trip, <strong>Total</strong> = end-to-end. LLM and Total appear once the upstream response completes.</dd>
    </dl>
  </div>
  <div class="log-scroll" id="log-scroll">
    <table class="log-table">
      <thead><tr><th>T#</th><th>Inbound Tags</th><th>Response Tags</th><th>Message</th><th>Payload</th><th>Tokens</th><th>Base</th><th>Injected</th><th>VC</th><th>LLM</th><th>Total</th><th></th></tr></thead>
      <tbody id="log-body"></tbody>
    </table>
  </div>
</div>

<div class="panel">
  <div class="section-header"><h2>Compaction Events</h2><button class="help-btn" data-help="compactions" onclick="toggleHelp(this)">?</button></div>
  <div class="help-content" data-help-for="compactions" style="display:none">
    <p>History of compaction events (newest first). Each entry represents one compaction operation where the engine summarized older conversation turns to free context window space.</p>
    <dl>
      <dt>Turn</dt><dd>The conversation turn that triggered compaction (when token usage crossed the threshold).</dd>
      <dt>Segments</dt><dd>Number of tag-grouped segments that were summarized in this compaction batch.</dd>
      <dt>Freed</dt><dd>Tokens reclaimed by replacing raw conversation turns with compressed summaries.</dd>
      <dt>Tags</dt><dd>The semantic tags covered by the compacted segments. These tags now have stored summaries available for future retrieval.</dd>
      <dt>Tag summaries built</dt><dd>Number of tag-level summaries (re)built. Tag summaries roll up all segment summaries for a given tag into one cohesive summary for efficient retrieval.</dd>
      <dt>Watermark</dt><dd>The compacted_through message index after this compaction. All messages up to this index have been compacted and are no longer in raw history.</dd>
    </dl>
  </div>
  <div id="compaction-list"><span class="empty-state">no compactions yet</span></div>
</div>

</div>

<div class="modal-overlay" id="settings-overlay" style="display:none" onclick="if(event.target===this)closeSettings()">
  <div class="modal">
    <div class="modal-header">
      <div>
        <h2>Settings</h2>
        <div class="modal-subtitle">Changes apply to current session only</div>
      </div>
      <button class="modal-close" onclick="closeSettings()">&times;</button>
    </div>
    <div class="modal-body" id="settings-body">
      <span class="empty-state">loading...</span>
    </div>
    <div class="modal-footer">
      <span class="settings-status" id="settings-status"></span>
      <button class="replay-btn" onclick="saveSettings()">Save</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="inspect-overlay" style="display:none" onclick="if(event.target===this)closeInspect()">
  <div class="modal" style="max-width:900px;width:90vw">
    <div class="modal-header">
      <div>
        <h2>Request Inspector</h2>
        <div class="modal-subtitle" id="inspect-subtitle">Turn —</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="replay-btn" onclick="saveInspectedRequest()" title="Download raw JSON">Save JSON</button>
        <button class="modal-close" onclick="closeInspect()">&times;</button>
      </div>
    </div>
    <div class="modal-body" id="inspect-body" style="max-height:70vh;overflow:auto;font-size:12px">
      <span class="empty-state">loading...</span>
    </div>
  </div>
</div>

<script>
(function() {
  const $ = id => document.getElementById(id);

  // State
  let totalRequests = 0, totalCompactions = 0, totalFreed = 0;
  let totalTurns = 0, contextWindow = 120000;
  let waitSum = 0, inboundSum = 0, contextSum = 0;
  let compactedThrough = 0, historyLen = 0;
  let uptimeBase = 0, uptimeStart = Date.now();
  let autoScroll = true;
  let totalOriginalTokens = 0, totalSummaryTokens = 0, totalContextInjected = 0;
  let cumActualInput = 0, cumBaselineInput = 0, baselineHistoryTokens = 0;
  let latestSystemTokens = 0;
  const BASELINE_RATIO = 0.30;

  const logScroll = $('log-scroll');
  logScroll.addEventListener('scroll', () => {
    const el = logScroll;
    autoScroll = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  });

  function fmtUptime(s) {
    s = Math.floor(s);
    if (s < 60) return s + 's';
    if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
    const h = Math.floor(s/3600);
    const m = Math.floor((s%3600)/60);
    return h + 'h ' + m + 'm';
  }
  function fmtNum(n) { return n >= 1000 ? (n/1000).toFixed(1) + 'k' : String(n); }
  function fmtTime(ms) {
    if (ms === undefined || ms === null) return '\\u2014';
    if (ms < 1000) return Math.round(ms) + 'ms';
    return (ms / 1000).toFixed(1) + 's';
  }

  function updateStats() {
    const elapsed = uptimeBase + (Date.now() - uptimeStart) / 1000;
    $('s-uptime').textContent = fmtUptime(elapsed);
    $('s-requests').textContent = totalRequests;
    $('s-turns').textContent = totalTurns;
    $('s-compactions').textContent = totalCompactions;
    $('s-freed').textContent = fmtNum(totalFreed);
  }

  function updatePipeline() {
    if (totalRequests === 0) return;
    $('p-wait').textContent = Math.round(waitSum / totalRequests) + 'ms';
    $('p-inbound').textContent = Math.round(inboundSum / totalRequests) + 'ms';
    $('p-context').textContent = fmtNum(Math.round(contextSum / totalRequests)) + 't';
  }

  function updateMemory() {
    // Rough estimate: compacted_through / history_len as progress
    const pct = historyLen > 0 ? Math.round((compactedThrough / historyLen) * 100) : 0;
    const fill = $('mem-fill');
    fill.style.width = pct + '%';
    fill.className = 'memory-bar-fill' + (pct > 85 ? ' crit' : pct > 70 ? ' warn' : '');
    $('mem-note').textContent = 'compacted through turn ' + Math.floor(compactedThrough / 2) +
      ' \\u00b7 ' + historyLen + ' messages \\u00b7 window ' + fmtNum(contextWindow) + 't';
  }

  function updateSavings() {
    $('sv-freed').textContent = fmtNum(totalFreed) + 't';
    if (totalOriginalTokens > 0) {
      const ratio = totalSummaryTokens / totalOriginalTokens;
      $('sv-ratio').textContent = Math.round(ratio * 100) + '%';
      $('sv-ratio-detail').textContent = fmtNum(totalOriginalTokens) + 't -> ' + fmtNum(totalSummaryTokens) + 't';
      $('sv-freed-detail').textContent = totalCompactions + ' compaction' + (totalCompactions !== 1 ? 's' : '');
    } else {
      $('sv-ratio').textContent = '--';
    }
    // Session efficiency
    if (cumBaselineInput > 0 && cumActualInput > 0) {
      const savings = Math.max(0, 1 - (cumActualInput / cumBaselineInput));
      $('sv-efficiency').textContent = Math.round(savings * 100) + '%';
      $('sv-efficiency-detail').textContent = fmtNum(cumActualInput) + 't vc vs ' + fmtNum(cumBaselineInput) + 't baseline';
    } else {
      $('sv-efficiency').textContent = '--';
    }

    $('sv-injected').textContent = fmtNum(totalContextInjected) + 't';
    $('sv-avg-ctx').textContent = totalRequests > 0 ? fmtNum(Math.round(totalContextInjected / totalRequests)) + 't' : '0';

    // Estimated dollar costs: baseline vs actual vs saved
    const baseM = cumBaselineInput / 1_000_000;
    const actualM = cumActualInput / 1_000_000;
    const savedTokens = Math.max(0, cumBaselineInput - cumActualInput);
    const savedM = savedTokens / 1_000_000;
    const pctSaved = cumBaselineInput > 0 ? Math.round((savedTokens / cumBaselineInput) * 100) : 0;
    const tiers = [
      {key: 'haiku', rate: 0.25},
      {key: 'sonnet', rate: 3.00},
      {key: 'opus', rate: 15.00},
    ];
    for (const t of tiers) {
      const hasData = cumBaselineInput > 0 && cumActualInput > 0;
      $('cost-' + t.key + '-base').textContent = hasData ? '$' + (baseM * t.rate).toFixed(4) : '--';
      $('cost-' + t.key + '-vc').textContent = hasData ? '$' + (actualM * t.rate).toFixed(4) : '--';
      $('cost-' + t.key).textContent = hasData ? '$' + (savedM * t.rate).toFixed(4) : '--';
      $('cost-' + t.key + '-pct').textContent = hasData ? pctSaved + '%' : '--';
    }
  }

  function setTags(tags) {
    const cloud = $('tag-cloud');
    if (!tags || tags.length === 0) {
      cloud.innerHTML = '<span class="empty-state">no tags yet</span>';
      return;
    }
    cloud.innerHTML = tags.map(t => '<span class="tag">' + t + '</span>').join('');
  }

  function buildTagCell(tags, broad, temporal) {
    const badges = [];
    if (broad) badges.push('<span class="broad-badge">BROAD</span>');
    if (temporal) badges.push('<span class="temporal-badge">TEMPORAL</span>');
    if (badges.length) return badges.join(' ');
    return (tags || []).join(', ');
  }

  function addRequestRow(evt) {
    const body = $('log-body');
    const tr = document.createElement('tr');
    tr.className = 'flash';
    tr.id = 'req-' + evt.turn;
    if (evt.broad) tr.dataset.broad = '1';
    if (evt.temporal) tr.dataset.temporal = '1';
    const tagStr = buildTagCell(evt.tags, evt.broad, evt.temporal);
    const preview = (evt.message_preview || '').substring(0, 50);
    const vcMs = evt.overhead_ms !== undefined ? evt.overhead_ms : Math.round((evt.wait_ms || 0) + (evt.inbound_ms || 0));
    // Payload: filtered/total turns
    let payload = '';
    if (evt.filtered_turns !== undefined) {
      payload = evt.filtered_turns + '/' + evt.total_turns;
    } else if (evt.total_turns !== undefined) {
      payload = String(evt.total_turns);
    }
    const tokens = evt.input_tokens ? fmtNum(evt.input_tokens) + 't' : '\\u2014';
    const injected = fmtNum(evt.context_tokens || 0) + 't';
    tr.innerHTML =
      '<td>' + (evt.turn ?? '') + '</td>' +
      '<td class="tags-cell">' + tagStr + '</td>' +
      '<td class="tags-cell resp-tags"></td>' +
      '<td class="msg-cell" title="' + preview.replace(/"/g,'&quot;') + '">' + preview + '</td>' +
      '<td>' + payload + '</td>' +
      '<td>' + tokens + '</td>' +
      '<td class="baseline-cell timing-pending">\\u2014</td>' +
      '<td>' + injected + '</td>' +
      '<td class="timing-vc">' + fmtTime(vcMs) + '</td>' +
      '<td class="timing-llm timing-pending">\\u2014</td>' +
      '<td class="timing-total timing-pending">\\u2014</td>' +
      '<td><a href="#" onclick="inspectRequest(' + evt.turn + ');return false" style="color:var(--accent);font-size:11px">inspect</a></td>';
    body.insertBefore(tr, body.firstChild);
    // Keep max 200 rows
    while (body.children.length > 200) body.removeChild(body.lastChild);
    if (autoScroll) logScroll.scrollTop = 0;
  }

  function addIngestedRow(evt) {
    const body = $('log-body');
    const tr = document.createElement('tr');
    tr.className = 'ingested-row';
    tr.id = 'req-' + evt.turn;
    const preview = (evt.message_preview || '').substring(0, 50);
    // Tags go in response column (generated from user+assistant pair)
    var tagHtml = '';
    if (evt.primary_tag) {
      tagHtml = '<strong>' + evt.primary_tag + '</strong>';
      var others = (evt.tags || []).filter(function(t) { return t !== evt.primary_tag; });
      if (others.length) tagHtml += ', ' + others.join(', ');
    } else {
      tagHtml = (evt.tags || []).join(', ');
    }
    tr.innerHTML =
      '<td>' + (evt.turn ?? '') + '</td>' +
      '<td class="tags-cell"><span class="ingested-badge">HISTORY</span></td>' +
      '<td class="tags-cell">' + tagHtml + '</td>' +
      '<td class="msg-cell" title="' + preview.replace(/"/g,'&quot;') + '">' + preview + '</td>' +
      '<td></td><td></td><td class="baseline-cell"></td><td></td>' +
      '<td></td><td></td><td></td><td></td>';
    body.insertBefore(tr, body.firstChild);
  }

  let lastSessionFetchTurn = -1;

  function fillResponseTags(row, tags, primaryTag) {
    const respCell = row.children[2];
    respCell.innerHTML = (tags || []).join(', ');
    if (primaryTag) {
      respCell.innerHTML = '<strong>' + primaryTag + '</strong>' +
        (tags && tags.length > 1 ? ', ' + tags.filter(t => t !== primaryTag).join(', ') : '');
    }
  }

  function handleTurnComplete(evt) {
    const row = $('req-' + evt.turn);
    if (row) {
      fillResponseTags(row, evt.tags, evt.primary_tag);
      row.className = 'flash';
    }
    totalTurns = Math.max(totalTurns, evt.turn + 1);
    // Baseline simulation: system prompt + growing history per turn
    const tpt = evt.turn_pair_tokens || 0;
    baselineHistoryTokens += tpt;
    if (baselineHistoryTokens > contextWindow) {
      const prot = tpt * 4;
      const compactable = Math.max(0, baselineHistoryTokens - prot);
      baselineHistoryTokens = Math.round(compactable * BASELINE_RATIO) + prot;
    }
    cumBaselineInput += latestSystemTokens + baselineHistoryTokens;
    // Update baseline cell in grid row
    if (row) {
      const baseCell = row.querySelector('.baseline-cell');
      if (baseCell) {
        baseCell.textContent = fmtNum(latestSystemTokens + baselineHistoryTokens) + 't';
        baseCell.classList.remove('timing-pending');
      }
    }
    updateStats();
    updateSavings();
    // Refresh active tags from turn_complete data
    if (evt.active_tags) setTags(evt.active_tags);
    if (evt.store_tag_count !== undefined) {
      $('store-info').textContent = evt.store_tag_count + ' tags in store';
    }
    // Refresh sessions every 5 turns (compaction may have stored new segments)
    if (evt.turn - lastSessionFetchTurn >= 5) {
      lastSessionFetchTurn = evt.turn;
      fetchSessions();
    }
  }

  function handleResponse(evt) {
    const row = $('req-' + evt.turn);
    if (!row) return;
    const llmCell = row.querySelector('.timing-llm');
    const totalCell = row.querySelector('.timing-total');
    if (llmCell) {
      llmCell.textContent = fmtTime(evt.upstream_ms);
      llmCell.classList.remove('timing-pending');
    }
    if (totalCell) {
      totalCell.textContent = fmtTime(evt.total_ms);
      totalCell.classList.remove('timing-pending');
    }
    if (evt.error) row.style.opacity = '0.6';
  }

  function addCompaction(evt) {
    const list = $('compaction-list');
    if (list.querySelector('.empty-state')) list.innerHTML = '';
    const div = document.createElement('div');
    div.className = 'compaction-entry';
    div.innerHTML =
      '<div class="compaction-header">T' + evt.turn + ' \\u00b7 ' +
      evt.segments + ' segments \\u00b7 freed ' + fmtNum(evt.tokens_freed || 0) + 't' +
      ' \\u00b7 [' + (evt.tags || []).join(', ') + ']</div>' +
      '<div class="compaction-detail">Built ' + (evt.tag_summaries_built || 0) +
      ' tag summaries \\u00b7 watermark ' + (evt.compacted_through || 0) + '</div>';
    list.insertBefore(div, list.firstChild);
    compactedThrough = evt.compacted_through || compactedThrough;
    updateMemory();
  }

  function handleSnapshot(data) {
    uptimeBase = data.uptime_s || 0;
    uptimeStart = Date.now();
    totalRequests = data.total_requests || 0;
    totalCompactions = data.total_compactions || 0;
    totalFreed = data.total_tokens_freed || 0;
    totalOriginalTokens = data.total_original_tokens || 0;
    totalSummaryTokens = data.total_summary_tokens || 0;
    totalContextInjected = data.total_context_injected || 0;
    cumActualInput = data.total_actual_input || 0;
    cumBaselineInput = data.total_baseline_input || 0;
    waitSum = (data.avg_wait_ms || 0) * totalRequests;
    inboundSum = (data.avg_inbound_ms || 0) * totalRequests;
    contextSum = (data.avg_context_tokens || 0) * totalRequests;
    compactedThrough = data.compacted_through || 0;
    historyLen = data.history_len || 0;
    contextWindow = data.context_window || 120000;
    totalTurns = data.history_len ? Math.floor(data.history_len / 2) : totalRequests;

    // Set latestSystemTokens from snapshot requests
    if (data.recent_requests && data.recent_requests.length) {
      latestSystemTokens = data.recent_requests[data.recent_requests.length - 1].system_tokens || 0;
    }

    updateStats();
    updatePipeline();
    updateMemory();
    updateSavings();
    fetchSessions();
    setTags(data.active_tags || []);
    if (data.store_tag_count !== undefined) {
      $('store-info').textContent = data.store_tag_count + ' tags in store';
    }

    // Fill ingested history turns (iterate oldest-first so prepend produces newest-at-top)
    var ingested = data.ingested_turns || [];
    for (var ii = 0; ii < ingested.length; ii++) addIngestedRow(ingested[ii]);
    // Fill request log (prepended = newest at top)
    (data.recent_requests || []).forEach(r => addRequestRow(r));
    // Fill response timing
    (data.responses || []).forEach(r => handleResponse(r));
    // Bootstrap baseline from ingested history BEFORE processing live turn_completes
    var snapBaselineHist = 0;
    (data.history_ingestions || []).forEach(function(h) {
      if (h.baseline_history_tokens) snapBaselineHist = h.baseline_history_tokens;
      handleHistoryIngestion(h);
    });
    // Process turn_completes: fill response tags + accumulate baseline on top of ingested
    (data.turn_completes || []).forEach(function(tc) {
      var row = $('req-' + tc.turn);
      if (row && tc.tags && tc.tags.length) {
        fillResponseTags(row, tc.tags, tc.primary_tag);
      }
      var tpt = tc.turn_pair_tokens || 0;
      snapBaselineHist += tpt;
      if (snapBaselineHist > contextWindow) {
        var prot = tpt * 4;
        var compactable = Math.max(0, snapBaselineHist - prot);
        snapBaselineHist = Math.round(compactable * BASELINE_RATIO) + prot;
      }
      if (row) {
        var baseCell = row.querySelector('.baseline-cell');
        if (baseCell) {
          baseCell.textContent = fmtNum(latestSystemTokens + snapBaselineHist) + 't';
          baseCell.classList.remove('timing-pending');
        }
      }
    });
    baselineHistoryTokens = snapBaselineHist;
    // Fill compactions
    (data.compactions || []).forEach(c => addCompaction(c));
  }

  // Sessions
  function fetchSessions() {
    fetch('/dashboard/sessions')
      .then(r => r.json())
      .then(data => renderSessions(data.sessions))
      .catch(() => {});
  }

  function renderSessions(sessions) {
    const list = $('session-list');
    if (!sessions || sessions.length === 0) {
      list.innerHTML = '<span class="empty-state">no sessions with stored segments yet</span>';
      return;
    }
    list.innerHTML = sessions.map(s => {
      const idShort = s.session_id;
      const ratio = s.compression_ratio > 0 ? Math.round(s.compression_ratio * 100) + '%' : '--';
      const freed = s.total_full_tokens - s.total_summary_tokens;
      const tagHtml = (s.distinct_tags || []).slice(0, 8).map(
        t => '<span class="tag">' + t + '</span>'
      ).join('');
      const more = (s.distinct_tags || []).length > 8
        ? '<span class="tag">+' + ((s.distinct_tags || []).length - 8) + '</span>' : '';
      const oldest = s.oldest_segment ? new Date(s.oldest_segment).toLocaleString() : '--';
      const newest = s.newest_segment ? new Date(s.newest_segment).toLocaleString() : '--';
      const delBtn = s.is_current ? '' :
        '<button class="session-delete" onclick="deleteSession(\\'' + s.session_id + '\\')">Delete</button>';
      return '<div class="session-card' + (s.is_current ? ' current' : '') + '">' +
        '<div class="session-header">' +
          '<span class="session-id">' + idShort + '</span>' +
          (s.is_current ? '<span class="session-badge">CURRENT</span>' : '') +
          delBtn +
        '</div>' +
        '<div class="session-stats">' +
          '<div><span class="label">Segments</span><br><span class="val">' + s.segment_count + '</span></div>' +
          '<div><span class="label">Full Tokens</span><br><span class="val">' + fmtNum(s.total_full_tokens) + '</span></div>' +
          '<div><span class="label">Compression</span><br><span class="val">' + ratio + '</span></div>' +
          '<div><span class="label">Freed</span><br><span class="val">' + fmtNum(freed) + 't</span></div>' +
        '</div>' +
        '<div class="session-tags">' + tagHtml + more + '</div>' +
        '<div class="session-time">' + oldest + ' \\u2192 ' + newest + '</div>' +
        (s.compaction_model ? '<div class="session-time">model: ' + s.compaction_model + '</div>' : '') +
      '</div>';
    }).join('');
  }

  // Replay controls
  let replayRunning = false;

  window.startReplay = function() {
    const file = $('replay-file').value.trim();
    if (!file) { alert('Enter a prompts file path'); return; }

    $('replay-start').disabled = true;
    fetch('/dashboard/replay/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({file: file}),
    })
    .then(r => r.json().then(d => ({status: r.status, data: d})))
    .then(({status, data}) => {
      if (status >= 400) {
        alert(data.error || 'Failed to start replay');
        $('replay-start').disabled = false;
        return;
      }
      replayRunning = true;
      $('replay-stop').disabled = false;
      $('replay-progress').style.display = 'block';
      $('replay-fill').style.width = '0%';
      $('replay-fill').className = 'replay-bar-fill';
      $('replay-status').textContent = 'Starting... 0/' + data.total_prompts;
    })
    .catch(() => { $('replay-start').disabled = false; });
  };

  window.stopReplay = function() {
    fetch('/dashboard/replay/stop', {method: 'POST'});
    $('replay-status').textContent = 'Stopping...';
  };

  function handleHistoryIngestion(data) {
    const turns = data.turns_ingested || 0;
    const ms = data.elapsed_ms || 0;
    const el = $('ingestion-status');
    if (el) el.textContent = 'Ingested ' + turns + ' historical turns (' + Math.round(ms) + 'ms)';
    // Bootstrap baseline from ingested history token count
    if (data.baseline_history_tokens) {
      baselineHistoryTokens = data.baseline_history_tokens;
    }
  }

  function handleReplayProgress(data) {
    const pct = data.total > 0 ? Math.round((data.turn / data.total) * 100) : 0;
    $('replay-fill').style.width = pct + '%';
    const preview = data.prompt_preview ? ' \\u2014 "' + data.prompt_preview.substring(0, 50) + '"' : '';
    $('replay-status').textContent = 'Turn ' + data.turn + '/' + data.total + ' (' + pct + '%)' + preview;
  }

  function handleReplayDone(data) {
    replayRunning = false;
    $('replay-start').disabled = false;
    $('replay-stop').disabled = true;
    const fill = $('replay-fill');
    if (data.status === 'complete') {
      fill.style.width = '100%';
      fill.className = 'replay-bar-fill done';
      $('replay-status').textContent = 'Complete \\u2014 ' + data.turns_completed + '/' + data.total + ' turns';
    } else if (data.status === 'stopped') {
      fill.className = 'replay-bar-fill';
      $('replay-status').textContent = 'Stopped at turn ' + data.turns_completed + '/' + data.total;
    } else {
      fill.className = 'replay-bar-fill error';
      $('replay-status').textContent = 'Error: ' + (data.error || 'unknown');
    }
    // Final refresh after replay ends
    fetchSessions();
    updateSavings();
  }

  // SSE connection
  function connect() {
    $('conn-status').textContent = 'connecting...';
    const es = new EventSource('/dashboard/events');

    es.onopen = () => {
      $('conn-status').textContent = 'connected';
      document.querySelector('.dot').style.background = 'var(--green)';
    };
    es.onerror = () => {
      $('conn-status').textContent = 'reconnecting...';
      document.querySelector('.dot').style.background = 'var(--red)';
    };
    es.onmessage = (e) => {
      let data;
      try { data = JSON.parse(e.data); } catch { return; }

      if (data.type === 'snapshot') {
        handleSnapshot(data);
      } else if (data.type === 'request') {
        totalRequests++;
        waitSum += data.wait_ms || 0;
        inboundSum += data.inbound_ms || 0;
        contextSum += data.context_tokens || 0;
        totalContextInjected += data.context_tokens || 0;
        cumActualInput += data.input_tokens || 0;
        latestSystemTokens = data.system_tokens || latestSystemTokens;
        historyLen = data.history_len || historyLen;
        compactedThrough = data.compacted_through || compactedThrough;
        addRequestRow(data);
        updateStats();
        updatePipeline();
        updateMemory();
        updateSavings();
      } else if (data.type === 'response') {
        handleResponse(data);
      } else if (data.type === 'turn_complete') {
        handleTurnComplete(data);
      } else if (data.type === 'compaction') {
        totalCompactions++;
        totalFreed += data.tokens_freed || 0;
        totalOriginalTokens += data.original_tokens || 0;
        totalSummaryTokens += data.summary_tokens || 0;
        addCompaction(data);
        updateStats();
        updateSavings();
        fetchSessions();
      } else if (data.type === 'ingested_turn') {
        addIngestedRow(data);
      } else if (data.type === 'history_ingestion') {
        handleHistoryIngestion(data);
      } else if (data.type === 'replay_progress') {
        handleReplayProgress(data);
      } else if (data.type === 'replay_done') {
        handleReplayDone(data);
      }
    };
  }

  window.deleteSession = function(sid) {
    if (!confirm('Delete all stored segments for session ' + sid.substring(0, 12) + '...?')) return;
    fetch('/dashboard/sessions/' + sid, {method: 'DELETE'})
      .then(r => r.json())
      .then(data => { fetchSessions(); })
      .catch(() => {});
  };

  window.shutdownProxy = function() {
    if (!confirm('Shut down the proxy server?')) return;
    fetch('/dashboard/shutdown', {method: 'POST'});
    $('conn-status').textContent = 'shutting down...';
  };

  window.exportSession = function() {
    fetch('/dashboard/export')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var ts = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
        var name = 'vc-export-' + ts + '.json';
        var blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = name;
        a.click();
        URL.revokeObjectURL(a.href);
      })
      .catch(function(err) { alert('Export failed: ' + err.message); });
  };

  // Manual compaction
  window.compactNow = function() {
    var btn = $('compact-btn');
    btn.disabled = true;
    btn.textContent = 'Compacting...';
    fetch('/dashboard/compact', {method: 'POST'})
      .then(function(r) { return r.json().then(function(d) { return {status: r.status, data: d}; }); })
      .then(function(res) {
        if (res.status === 409) {
          btn.textContent = 'Already compacting...';
        } else if (res.status >= 400) {
          btn.textContent = res.data.error || 'Error';
        } else if (res.data.status === 'no_action') {
          btn.textContent = 'Nothing to compact';
        } else {
          btn.textContent = 'Compacted ' + res.data.segments + ' segments';
        }
        setTimeout(function() {
          btn.textContent = 'Compact Now';
          btn.disabled = false;
        }, 3000);
      })
      .catch(function() {
        btn.textContent = 'Error';
        setTimeout(function() {
          btn.textContent = 'Compact Now';
          btn.disabled = false;
        }, 3000);
      });
  };

  // Request inspector
  var _inspectedData = null;

  window.inspectRequest = function(turn) {
    $('inspect-overlay').style.display = 'flex';
    $('inspect-subtitle').textContent = 'Turn ' + turn;
    $('inspect-body').innerHTML = '<span class="empty-state">loading...</span>';
    _inspectedData = null;

    fetch('/dashboard/requests/' + turn)
      .then(function(r) {
        if (!r.ok) throw new Error('Not found');
        return r.json();
      })
      .then(function(data) {
        _inspectedData = data;
        renderInspect(data);
      })
      .catch(function() {
        $('inspect-body').innerHTML = '<span class="empty-state">Request not in capture buffer</span>';
      });
  };

  function renderInspect(data) {
    var html = '';
    // Tags section
    var hasInbound = data.inbound_tags && data.inbound_tags.length;
    var hasResponse = data.response_tags && data.response_tags.length;
    if (hasInbound || hasResponse) {
      html += '<div style="display:flex;gap:24px;margin-bottom:12px">';
      html += '<div style="flex:1"><strong style="color:var(--accent)">Inbound Tags</strong>';
      html += '<div class="tag-cloud" style="margin-top:4px">';
      if (hasInbound) {
        data.inbound_tags.forEach(function(t) { html += '<span class="tag">' + escHtml(t) + '</span>'; });
      } else {
        html += '<span class="empty-state">none</span>';
      }
      html += '</div></div>';
      html += '<div style="flex:1"><strong style="color:var(--accent)">Response Tags</strong>';
      html += '<div class="tag-cloud" style="margin-top:4px">';
      if (hasResponse) {
        data.response_tags.forEach(function(t) { html += '<span class="tag">' + escHtml(t) + '</span>'; });
      } else {
        html += '<span class="empty-state">pending</span>';
      }
      html += '</div></div>';
      html += '</div>';
    }
    // System prompt
    if (data.system) {
      html += '<div style="margin-bottom:12px"><strong style="color:var(--accent)">System Prompt</strong>';
      html += ' <span style="color:var(--text-dim);font-size:11px">(' + estimateTokens(data.system) + ' tokens est.)</span>';
      html += '<pre style="background:var(--bg);padding:8px;border-radius:4px;white-space:pre-wrap;max-height:200px;overflow:auto;margin-top:4px;border:1px solid var(--border)">';
      var sysText = typeof data.system === 'string' ? data.system : JSON.stringify(data.system, null, 2);
      html += escHtml(sysText.substring(0, 5000));
      if (sysText.length > 5000) html += '\\n... (' + sysText.length + ' chars total)';
      html += '</pre></div>';
    }
    // Messages
    html += '<div><strong style="color:var(--accent)">Messages</strong>';
    var allMsgs = data.messages || [];
    html += ' <span style="color:var(--text-dim);font-size:11px">(' + allMsgs.length + ' messages, newest first)</span></div>';
    for (var _ri = allMsgs.length - 1; _ri >= 0; _ri--) { var msg = allMsgs[_ri]; var i = _ri;
      var role = msg.role || '?';
      var roleColor = role === 'user' ? '#4a9' : role === 'assistant' ? '#49a' : role === 'system' ? '#a94' : '#999';
      html += '<div style="margin:8px 0;border-left:3px solid ' + roleColor + ';padding-left:8px">';
      html += '<div style="font-weight:bold;color:' + roleColor + '">[' + i + '] ' + role + '</div>';
      if (typeof msg.content === 'string') {
        html += '<pre style="background:var(--bg);padding:6px;border-radius:4px;white-space:pre-wrap;max-height:150px;overflow:auto;margin:4px 0;font-size:11px">' + escHtml(msg.content.substring(0, 3000));
        if (msg.content.length > 3000) html += '\\n... (' + msg.content.length + ' chars)';
        html += '</pre>';
      } else if (Array.isArray(msg.content)) {
        msg.content.forEach(function(block) {
          var btype = block.type || 'unknown';
          var badge = '<span style="background:var(--border);padding:1px 6px;border-radius:3px;font-size:10px;margin-right:4px">' + btype + '</span>';
          html += '<div style="margin:4px 0">' + badge;
          if (btype === 'text') {
            var t = block.text || '';
            html += '<pre style="background:var(--bg);padding:6px;border-radius:4px;white-space:pre-wrap;max-height:150px;overflow:auto;display:inline-block;width:calc(100% - 16px);font-size:11px">' + escHtml(t.substring(0, 3000));
            if (t.length > 3000) html += '\\n... (' + t.length + ' chars)';
            html += '</pre>';
          } else if (btype === 'tool_use') {
            html += '<span style="color:var(--text-dim);font-size:11px"> ' + (block.name || '') + '()</span>';
          } else if (btype === 'tool_result') {
            var rc = typeof block.content === 'string' ? block.content : JSON.stringify(block.content);
            html += '<pre style="background:var(--bg);padding:4px;border-radius:4px;white-space:pre-wrap;max-height:80px;overflow:auto;font-size:10px;color:var(--text-dim)">' + escHtml((rc || '').substring(0, 1000)) + '</pre>';
          } else if (btype === 'thinking') {
            html += '<span style="color:var(--text-dim);font-size:11px"> (' + ((block.thinking || '').length) + ' chars)</span>';
          } else {
            html += '<span style="color:var(--text-dim);font-size:11px"> ' + JSON.stringify(block).substring(0, 200) + '</span>';
          }
          html += '</div>';
        });
      }
      html += '</div>';
    }
    $('inspect-body').innerHTML = html;
  }

  function estimateTokens(s) {
    if (typeof s === 'string') return Math.round(s.length / 4);
    return Math.round(JSON.stringify(s).length / 4);
  }

  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  window.saveInspectedRequest = function() {
    if (!_inspectedData) return;
    var blob = new Blob([JSON.stringify(_inspectedData, null, 2)], {type: 'application/json'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'request-turn-' + (_inspectedData.turn || 0) + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
  };

  window.closeInspect = function() {
    $('inspect-overlay').style.display = 'none';
    _inspectedData = null;
  };

  // Settings modal
  window.openSettings = function() {
    $('settings-overlay').style.display = 'flex';
    $('settings-body').innerHTML = '<span class="empty-state">loading...</span>';
    $('settings-status').textContent = '';
    $('settings-status').className = 'settings-status';
    fetch('/dashboard/settings')
      .then(function(r) {
        if (!r.ok) throw new Error('Failed to load settings');
        return r.json();
      })
      .then(function(data) { renderSettings(data); })
      .catch(function(err) {
        $('settings-body').innerHTML = '<span class="empty-state">' + err.message + '</span>';
      });
  };

  window.closeSettings = function() {
    $('settings-overlay').style.display = 'none';
  };

  var sectionHelp = {
    system:
      '<p>Read-only information about the engine configuration. These values are set at startup from your YAML config and cannot be changed at runtime.</p>' +
      '<dl>' +
      '<dt>Context Window</dt><dd>The maximum number of tokens your target LLM can process in a single request. All compaction thresholds, assembly budgets, and retrieval fractions are calculated relative to this value. A 120k window means the engine has 120,000 tokens to split between raw conversation history, retrieved summaries, and system context.</dd>' +
      '<dt>Tagger</dt><dd>The method used to generate semantic tags for each conversation turn. <strong>keyword</strong> uses deterministic regex pattern matching (fast, no API calls). <strong>llm</strong> sends conversation text to a language model for semantic understanding (slower, more accurate, understands nuance). Tags are the foundation of retrieval \u2014 they determine which stored summaries are relevant to the current query.</dd>' +
      '<dt>Summarizer</dt><dd>The LLM model used to create compaction summaries. When conversation history is compacted, this model reads the raw turns and produces a concise summary preserving key decisions, entities, and action items. Runs in a background thread so it does not block the next turn.</dd>' +
      '<dt>Storage</dt><dd>Where compacted segments and tag summaries are persisted. <strong>sqlite</strong> uses a local database with FTS5 full-text search. <strong>filesystem</strong> stores markdown files with YAML frontmatter and a JSON index. Both support the same operations; sqlite is recommended for performance.</dd>' +
      '</dl>',
    compaction:
      '<p>Compaction frees context window space by summarizing older conversation turns. When token usage crosses the soft threshold, the engine begins summarizing; at the hard threshold, it compacts immediately. Turns are grouped by tag, summarized by the LLM, and stored externally. The summaries can then be retrieved on-demand when their topics come up again.</p>' +
      '<dl>' +
      '<dt>Soft Threshold</dt><dd>Fraction of the context window that triggers voluntary compaction. When estimated token usage exceeds this ratio (e.g., 0.70 = 70% of window), the engine begins summarizing the oldest unprotected turns. Lower values compact more aggressively \u2014 more summaries and less raw history, but earlier recall via retrieval.</dd>' +
      '<dt>Hard Threshold</dt><dd>Fraction of the context window that forces immediate compaction. If token usage reaches this level (e.g., 0.85 = 85%), compaction runs before the next turn regardless. Must be strictly greater than soft threshold. The gap between soft and hard gives a buffer zone for graceful compaction.</dd>' +
      '<dt>Protected Turns</dt><dd>Number of most-recent conversation turns that are never compacted. These stay as raw messages in the history. Higher values preserve more recent context but leave less room for retrieved summaries. With a small context window, keep this low (3\u20134); with a large window, 6\u20138 is typical.</dd>' +
      '<dt>Min Summary Tokens</dt><dd>Minimum length for a compaction summary. Prevents the summarizer from generating a summary too terse to be useful. Even if the original segment is very short, the summary will be at least this many tokens. Too low and summaries lose critical detail.</dd>' +
      '<dt>Max Summary Tokens</dt><dd>Maximum length for a compaction summary. Caps summary size to prevent a single compacted segment from consuming too much of the retrieval budget. Must be greater than or equal to min. Typical range: 500\u20132000 depending on context window size.</dd>' +
      '</dl>',
    tagging:
      '<p>Tags are semantic labels assigned to each conversation turn (e.g., "authentication", "database", "deployment"). They drive retrieval: when you ask about a topic, the engine matches query tags against tags on stored segments to find relevant summaries. The LLM tagger generates tags from conversation content; the heuristic overrides are regex safety nets that catch specific query patterns the LLM might miss.</p>' +
      '<dl>' +
      '<dt>Broad Heuristic</dt><dd>A set of regex patterns that detect broad, retrospective queries like "what did we discuss?", "summarize everything", or "recap our conversation". When a broad query is detected, retrieval switches to a special branch that fetches <strong>all</strong> stored tag summaries (not just tag-overlap matches), giving the LLM a complete picture. This heuristic fires as a post-LLM override \u2014 if the LLM already flagged the query as broad, the heuristic is skipped. Disable if your tagger model reliably detects broad queries on its own.</dd>' +
      '<dt>Temporal Heuristic</dt><dd>A set of regex patterns that detect time-referencing queries like "what was the first thing we discussed?", "early on", or "going way back". When temporal intent is detected, retrieval fetches segment-level summaries sorted chronologically (oldest first), providing detailed recall of early conversation content that may have been compacted long ago. Disable if your tagger model reliably detects temporal intent.</dd>' +
      '</dl>',
    retrieval:
      '<p>Retrieval fetches stored summaries relevant to the current user message. The engine tags the incoming query, then matches those tags against tags on stored segments. Matching summaries are ranked, trimmed to budget, and assembled into a context block prepended to the LLM request as a <code>&lt;virtual-context&gt;</code> block.</p>' +
      '<dl>' +
      '<dt>Active Tag Lookback</dt><dd>How many recent turns to scan when building the "active tag" set (topics currently under discussion). Tags that appear in recent turns are considered active and <strong>skipped</strong> during retrieval, since that content is already present in the raw conversation history. Higher values skip more tags; lower values retrieve more aggressively.</dd>' +
      '<dt>Anchorless Lookback</dt><dd>When the tagger produces a fallback result (no strong tag match for the query), this controls how many recent turns define the "working set" for fallback retrieval. The engine collects tags from these recent turns and fetches their summaries. Larger values cast a wider net but may pull in less-relevant content.</dd>' +
      '<dt>Max Results</dt><dd>Maximum number of stored summaries returned per retrieval query. Each result is a compacted segment summary. Higher values provide more context for the LLM but consume more of the token budget. With a small context window, keep this low (3\u20135); large windows can support 10\u201320.</dd>' +
      '<dt>Budget Fraction</dt><dd>The fraction of the total context window allocated to retrieved summaries. For example, 0.25 on a 120k window = 30k tokens max for retrieved content. This limit prevents retrieval from crowding out the raw conversation history. The remaining budget goes to conversation messages and system context.</dd>' +
      '<dt>Include Related</dt><dd>When enabled, retrieval expands its search to include tags semantically related to the matched tags (via the <code>related_tags</code> field produced by the tagger). This bridges vocabulary mismatches \u2014 for example, finding content tagged "database" when the user says "storage", or "auth" when they say "login". Recommended to keep enabled.</dd>' +
      '</dl>',
    assembly:
      '<p>Assembly takes retrieved summaries and builds the final context block injected into every LLM request. It enforces token budgets, orders tags by priority (from tag_rules), and optionally adds a context hint \u2014 a brief topic list that tells the LLM what has been discussed previously so it does not claim ignorance about compacted topics.</p>' +
      '<dl>' +
      '<dt>Tag Context Max</dt><dd>Total token budget for injected tag summaries. Retrieved summaries are included highest-priority first and trimmed when this limit is reached. This value is synced to the retriever budget, so changing it here also updates the retrieval cap. Higher values provide richer context but leave less room for conversation history.</dd>' +
      '<dt>Recent Turns Kept</dt><dd>Number of recent conversation turns always included in the filtered history sent to the LLM, regardless of whether they match the current tags. Ensures the LLM always sees the most recent exchanges for continuity. Set to 0 to rely entirely on tag-based filtering.</dd>' +
      '<dt>Context Hint</dt><dd>When enabled, injects a brief topic list into the context after the first compaction occurs. This tells the LLM "you have previously discussed: [topic1, topic2, ...]" so it knows stored knowledge exists and can reference it naturally rather than claiming ignorance about compacted topics. Strongly recommended.</dd>' +
      '<dt>Hint Max Tokens</dt><dd>Token budget for the context hint block. Controls how much space the topic list can consume. Typically 100\u2013300 tokens is sufficient. Only relevant when Context Hint is enabled.</dd>' +
      '</dl>',
    summarization:
      '<p>Controls LLM behavior when generating compaction summaries. When older conversation turns are compacted, the summarizer model reads the raw messages and produces a concise summary. These summaries are stored and later retrieved when their topics become relevant again.</p>' +
      '<dl>' +
      '<dt>Temperature</dt><dd>Controls randomness in the summarizer output. <strong>0</strong> = fully deterministic (same input always produces the same summary). Higher values introduce variation and creativity. For compaction summaries, low values (0.1\u20130.4) are recommended to preserve factual accuracy and consistency. Values above 1.0 may produce unreliable summaries.</dd>' +
      '</dl>'
  };

  function sectionHeader(title, helpKey) {
    return '<div class="section-header"><h3>' + title + '</h3>' +
      '<button class="help-btn" data-help="' + helpKey + '" onclick="toggleHelp(this)">?</button></div>' +
      '<div class="help-content" data-help-for="' + helpKey + '" style="display:none">' +
      (sectionHelp[helpKey] || '') + '</div>';
  }

  window.toggleHelp = function(btn) {
    var key = btn.dataset.help;
    var content = document.querySelector('[data-help-for="' + key + '"]');
    if (!content) return;
    var visible = content.style.display !== 'none';
    // Close all others
    document.querySelectorAll('.help-content').forEach(function(el) { el.style.display = 'none'; });
    document.querySelectorAll('.help-btn').forEach(function(el) { el.classList.remove('active'); });
    if (!visible) {
      content.style.display = 'block';
      btn.classList.add('active');
    }
  };

  function renderSettings(data) {
    var html = '';
    // Readonly section
    html += '<div class="settings-section">' + sectionHeader('System', 'system');
    html += readonlyRow('Context Window', fmtNum(data.readonly.context_window) + ' tokens', 'Max tokens the target LLM accepts');
    html += readonlyRow('Tagger', data.readonly.tagger_type + (data.readonly.tagger_model ? ' / ' + data.readonly.tagger_model : ''), 'Tag generation method and model');
    html += readonlyRow('Summarizer', data.readonly.summarizer_model, 'Model used for compaction summaries');
    html += readonlyRow('Storage', data.readonly.storage_backend, 'Where segments and summaries are persisted');
    html += '</div>';
    // Compaction
    html += '<div class="settings-section">' + sectionHeader('Compaction', 'compaction');
    html += sliderRow('compaction', 'soft_threshold', 'Soft Threshold', 'Start compacting when context fills to this fraction', data.compaction.soft_threshold, 0.1, 0.95, 0.05);
    html += sliderRow('compaction', 'hard_threshold', 'Hard Threshold', 'Force immediate compaction at this fill level', data.compaction.hard_threshold, 0.2, 0.99, 0.01);
    html += numberRow('compaction', 'protected_recent_turns', 'Protected Turns', 'Recent turns shielded from compaction', data.compaction.protected_recent_turns, 1, 20);
    html += numberRow('compaction', 'min_summary_tokens', 'Min Summary Tokens', 'Floor for compaction summary length', data.compaction.min_summary_tokens, 50, 5000);
    html += numberRow('compaction', 'max_summary_tokens', 'Max Summary Tokens', 'Ceiling for compaction summary length', data.compaction.max_summary_tokens, 100, 10000);
    html += '</div>';
    // Tagging
    html += '<div class="settings-section">' + sectionHeader('Tagging', 'tagging');
    html += toggleRow('tagging', 'broad_heuristic_enabled', 'Broad Heuristic', 'Regex fallback to catch broad queries the LLM missed', data.tagging.broad_heuristic_enabled);
    html += toggleRow('tagging', 'temporal_heuristic_enabled', 'Temporal Heuristic', 'Regex fallback to catch temporal queries the LLM missed', data.tagging.temporal_heuristic_enabled);
    html += '</div>';
    // Retrieval
    html += '<div class="settings-section">' + sectionHeader('Retrieval', 'retrieval');
    html += numberRow('retrieval', 'active_tag_lookback', 'Active Tag Lookback', 'Recent turns scanned for active topic tags', data.retrieval.active_tag_lookback, 1, 20);
    html += numberRow('retrieval', 'anchorless_lookback', 'Anchorless Lookback', 'Working-set turns used for fallback retrieval', data.retrieval.anchorless_lookback, 1, 20);
    html += numberRow('retrieval', 'max_results', 'Max Results', 'Max stored summaries returned per query', data.retrieval.max_results, 1, 50);
    html += sliderRow('retrieval', 'max_budget_fraction', 'Budget Fraction', 'Context window share for retrieved summaries', data.retrieval.max_budget_fraction, 0.05, 0.5, 0.05);
    html += toggleRow('retrieval', 'include_related', 'Include Related', 'Expand retrieval to semantically related tags', data.retrieval.include_related);
    html += '</div>';
    // Assembly
    html += '<div class="settings-section">' + sectionHeader('Assembly', 'assembly');
    html += numberRow('assembly', 'tag_context_max_tokens', 'Tag Context Max', 'Token budget for injected tag summaries', data.assembly.tag_context_max_tokens, 1000, 100000);
    html += numberRow('assembly', 'recent_turns_always_included', 'Recent Turns Kept', 'Turns always included regardless of tag match', data.assembly.recent_turns_always_included, 0, 10);
    html += toggleRow('assembly', 'context_hint_enabled', 'Context Hint', 'Inject a topic list into context after compaction', data.assembly.context_hint_enabled);
    html += numberRow('assembly', 'context_hint_max_tokens', 'Hint Max Tokens', 'Token budget for the context hint block', data.assembly.context_hint_max_tokens, 50, 1000);
    html += '</div>';
    // Summarization
    html += '<div class="settings-section">' + sectionHeader('Summarization', 'summarization');
    html += sliderRow('summarization', 'temperature', 'Temperature', 'LLM randomness for compaction summaries (0 = deterministic)', data.summarization.temperature, 0, 2, 0.1);
    html += '</div>';

    $('settings-body').innerHTML = html;
    // Bind slider events
    document.querySelectorAll('.settings-slider').forEach(function(sl) {
      sl.addEventListener('input', function() {
        this.nextElementSibling.textContent = this.value;
      });
    });
    // Bind toggle events
    document.querySelectorAll('.settings-toggle').forEach(function(btn) {
      btn.addEventListener('click', function() {
        this.classList.toggle('on');
      });
    });
  }

  function labelHtml(label, desc) {
    var d = desc ? '<span class="settings-desc">' + desc + '</span>' : '';
    return '<div class="settings-label">' + label + d + '</div>';
  }
  function readonlyRow(label, value, desc) {
    return '<div class="settings-row">' + labelHtml(label, desc) +
      '<span class="settings-value">' + value + '</span></div>';
  }
  function numberRow(section, key, label, desc, value, min, max) {
    return '<div class="settings-row">' + labelHtml(label, desc) +
      '<input class="settings-input" type="number" data-section="' + section +
      '" data-key="' + key + '" value="' + value + '" min="' + min + '" max="' + max + '"></div>';
  }
  function sliderRow(section, key, label, desc, value, min, max, step) {
    return '<div class="settings-row">' + labelHtml(label, desc) +
      '<input class="settings-slider" type="range" data-section="' + section +
      '" data-key="' + key + '" value="' + value + '" min="' + min + '" max="' + max +
      '" step="' + step + '">' +
      '<span class="settings-slider-val">' + value + '</span></div>';
  }
  function toggleRow(section, key, label, desc, value) {
    return '<div class="settings-row">' + labelHtml(label, desc) +
      '<button class="settings-toggle' + (value ? ' on' : '') +
      '" data-section="' + section + '" data-key="' + key + '"></button></div>';
  }

  window.saveSettings = function() {
    var body = {};
    // Collect number inputs
    document.querySelectorAll('.settings-input').forEach(function(inp) {
      var section = inp.dataset.section, key = inp.dataset.key;
      if (!body[section]) body[section] = {};
      body[section][key] = Number(inp.value);
    });
    // Collect sliders
    document.querySelectorAll('.settings-slider').forEach(function(sl) {
      var section = sl.dataset.section, key = sl.dataset.key;
      if (!body[section]) body[section] = {};
      body[section][key] = Number(sl.value);
    });
    // Collect toggles
    document.querySelectorAll('.settings-toggle').forEach(function(btn) {
      var section = btn.dataset.section, key = btn.dataset.key;
      if (!body[section]) body[section] = {};
      body[section][key] = btn.classList.contains('on');
    });

    var statusEl = $('settings-status');
    statusEl.textContent = 'Saving...';
    statusEl.className = 'settings-status';
    fetch('/dashboard/settings', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    })
    .then(function(r) { return r.json().then(function(d) { return {status: r.status, data: d}; }); })
    .then(function(res) {
      if (res.status >= 400) {
        statusEl.textContent = res.data.error || 'Save failed';
        statusEl.className = 'settings-status error';
      } else {
        statusEl.textContent = 'Saved';
        statusEl.className = 'settings-status ok';
        renderSettings(res.data);
        setTimeout(function() { statusEl.textContent = ''; }, 2000);
      }
    })
    .catch(function(err) {
      statusEl.textContent = err.message;
      statusEl.className = 'settings-status error';
    });
  };

  // Tick uptime every second
  setInterval(updateStats, 1000);
  // Refresh sessions periodically (catches compaction results even if events missed)
  setInterval(fetchSessions, 15000);
  connect();
})();
</script>
</body>
</html>
"""
