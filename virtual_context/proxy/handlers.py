"""HTTP request handlers for the virtual-context proxy.

Contains _passthrough, _passthrough_bytes, _handle_streaming, and
_handle_non_streaming — the core request forwarding logic.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..core.tool_loop import (
    is_vc_tool,
    execute_vc_tool,
)
from ..types import Message

from .formats import get_format
from .helpers import (
    _forward_headers,
    _extract_delta_text,
    _extract_assistant_text,
    _extract_assistant_raw_content,
    _inject_context,
    _inject_conversation_marker,
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
from .metrics import ProxyMetrics
from .state import ProxyState

logger = logging.getLogger(__name__)


def _extract_usage(msg_usage: dict, target: dict) -> None:
    """Extract token usage from an Anthropic message_start usage dict.

    Anthropic splits input tokens into:
    - input_tokens: new (uncached) tokens
    - cache_creation_input_tokens: tokens written to cache
    - cache_read_input_tokens: tokens read from cache

    Total input = input_tokens + cache_creation + cache_read.
    """
    if "input_tokens" in msg_usage:
        uncached = msg_usage["input_tokens"]
        cache_create = msg_usage.get("cache_creation_input_tokens", 0) or 0
        cache_read = msg_usage.get("cache_read_input_tokens", 0) or 0
        target["input_tokens"] = uncached + cache_create + cache_read
        target["input_tokens_uncached"] = uncached
        target["cache_creation_input_tokens"] = cache_create
        target["cache_read_input_tokens"] = cache_read


async def _passthrough(
    client: httpx.AsyncClient,
    request: Request,
    url: str,
    headers: dict[str, str],
) -> StreamingResponse:
    body = await request.body()
    return await _passthrough_bytes(client, request.method, url, headers, body)


async def _passthrough_bytes(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> JSONResponse | Response:
    resp = await client.request(method, url, headers=headers, content=body)
    fwd = _forward_headers(dict(resp.headers))
    if resp.headers.get("content-type", "").startswith("application/json"):
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
            headers=fwd,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
        headers=fwd,
    )


async def _handle_streaming(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict,
    api_format: str,
    state: ProxyState | None,
    *,
    metrics: ProxyMetrics | None = None,
    turn: int = 0,
    turn_id: str = "",
    overhead_ms: float = 0.0,
    conversation_id: str = "",
    passthrough: bool = False,
    response_log_path: object | None = None,
    session_log_path: object | None = None,
    paging_enabled: bool = False,
    request_log_dir: object | None = None,
    log_prefix: str = "",
) -> StreamingResponse | JSONResponse:
    """Forward SSE stream, accumulating assistant text for on_turn_complete.

    Forwards raw bytes from the upstream to preserve exact SSE framing.
    The Node.js Anthropic SDK is strict about SSE formatting — decoding
    and re-encoding via ``aiter_lines()`` can break its parser.

    When *paging_enabled* is True, uses event-level forwarding: parses SSE
    events individually, forwards text events immediately, suppresses VC
    tool_use events, executes them locally, sends non-streaming continuations,
    and emits the continuation text as SSE back to the client.

    Non-2xx upstream responses (rate limits, overloads) are returned as
    JSON errors instead of broken SSE streams.
    """
    _MAX_CONTINUATION_LOOPS = 5

    headers = dict(headers)
    headers.pop("accept-encoding", None)

    # Open upstream connection — resolves after response headers arrive,
    # body streams lazily via aiter_bytes().
    t_upstream = time.monotonic()
    req = client.build_request("POST", url, headers=headers, json=body)
    upstream = await client.send(req, stream=True)

    # Non-2xx: drain body and return as JSON error (not broken SSE)
    if upstream.status_code >= 300:
        error_bytes = await upstream.aread()
        await upstream.aclose()
        upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)
        if metrics:
            metrics.record({
                "type": "response",
                "turn": turn,
                "turn_id": turn_id,
                "upstream_ms": upstream_ms,
                "total_ms": round(overhead_ms + upstream_ms, 1),
                "streaming": True,
                "error": True,
                "conversation_id": conversation_id,
            })
        logger.info(
            "T%d ERROR %d llm=%dms | %s",
            turn, upstream.status_code, int(upstream_ms),
            error_bytes[:200].decode("utf-8", errors="replace"),
        )
        try:
            error_body = json.loads(error_bytes)
        except (ValueError, json.JSONDecodeError):
            error_body = {"error": error_bytes.decode("utf-8", errors="replace")}
        return JSONResponse(
            content=error_body,
            status_code=upstream.status_code,
            headers=_forward_headers(dict(upstream.headers)),
        )

    # Forward upstream response headers + SSE-critical headers
    resp_headers = _forward_headers(dict(upstream.headers))
    resp_headers.setdefault("cache-control", "no-cache")
    resp_headers.setdefault("x-accel-buffering", "no")

    # ----- shared post-stream processing -----
    def _post_stream(text_chunks, raw_events, usage=None):
        """Return (assistant_text, upstream_ms) and handle side-effects."""
        nonlocal t_upstream
        _usage = usage or {}
        upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)
        if metrics:
            metrics.record({
                "type": "response",
                "turn": turn,
                "turn_id": turn_id,
                "upstream_ms": upstream_ms,
                "total_ms": round(overhead_ms + upstream_ms, 1),
                "streaming": True,
                "conversation_id": conversation_id,
            })
        assistant_text = "".join(text_chunks)
        # Capture response for dashboard inspector
        if metrics:
            metrics.capture_response(
                turn,
                {"streaming": True, "assistant_text": assistant_text},
                upstream_input_tokens=_usage.get("input_tokens", 0),
                upstream_output_tokens=_usage.get("output_tokens", 0),
                cache_creation_input_tokens=_usage.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=_usage.get("cache_read_input_tokens", 0),
            )
        # Log upstream LLM call to telemetry ledger
        if state and hasattr(state.engine, '_telemetry'):
            _model = body.get("model", "unknown")
            _out_tok = len(assistant_text) // 4 if assistant_text else 0
            _in_tok = state._last_enriched_payload_tokens or 0
            state.engine._telemetry.log(
                component="proxy_upstream",
                model=_model,
                input_tokens=_in_tok,
                output_tokens=_out_tok,
                duration_ms=upstream_ms,
                detail="streaming",
            )
        logger.info(
            "T%d RESPONSE stream=True llm=%dms total=%dms chars=%d",
            turn, int(upstream_ms), int(round(overhead_ms + upstream_ms)),
            len(assistant_text),
        )
        if response_log_path:
            try:
                response_log_path.write_text(
                    json.dumps({
                        "streaming": True,
                        "assistant_text": assistant_text,
                        "upstream_ms": upstream_ms,
                        "raw_events": "".join(raw_events),
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                logger.debug("streaming response log write failed", exc_info=True)
        return assistant_text, upstream_ms

    async def _inner_stream():
        text_chunks: list[str] = []
        raw_events: list[str] = []

        # ---------------------------------------------------------------
        # Paging path: event-level forwarding with VC tool interception
        # ---------------------------------------------------------------
        if paging_enabled:
            buf = b""
            vc_tools: list[dict] = []          # [{id, name, input}]
            non_vc_tools: list[dict] = []      # [{id, name}]
            all_content_blocks: list[dict] = []  # for continuation
            forwarded_block_count = 0
            suppressing = False
            current_vc_tool: dict | None = None
            current_text_parts: list[str] = []
            suppressed_raw: list[bytes] = []
            need_continuation = False
            _stream_usage: dict = {}  # accumulate input/output tokens

            try:
                async for raw_chunk in upstream.aiter_bytes():
                    raw_events.append(
                        raw_chunk.decode("utf-8", errors="replace"),
                    )
                    buf += raw_chunk
                    events, buf = _parse_sse_events(buf)

                    for _evt_type, data_str, raw_bytes in events:
                        if not data_str or data_str.strip() == "[DONE]":
                            yield raw_bytes
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            yield raw_bytes
                            continue

                        dtype = data.get("type", "")

                        # -- message_start: extract input_tokens --
                        if dtype == "message_start":
                            msg_usage = data.get("message", {}).get("usage", {})
                            _extract_usage(msg_usage, _stream_usage)

                        # -- content_block_start --
                        if dtype == "content_block_start":
                            block = data.get("content_block", {})
                            btype = block.get("type", "")
                            if (
                                btype == "tool_use"
                                and is_vc_tool(block.get("name", ""))
                            ):
                                suppressing = True
                                current_vc_tool = {
                                    "id": block["id"],
                                    "name": block["name"],
                                    "input_parts": [],
                                }
                                suppressed_raw.append(raw_bytes)
                                continue
                            elif btype == "tool_use":
                                non_vc_tools.append({
                                    "id": block["id"],
                                    "name": block["name"],
                                })
                            elif btype == "text":
                                current_text_parts = []

                        # -- content_block_delta --
                        elif dtype == "content_block_delta":
                            if suppressing:
                                delta = data.get("delta", {})
                                if delta.get("type") == "input_json_delta":
                                    current_vc_tool["input_parts"].append(
                                        delta.get("partial_json", ""),
                                    )
                                suppressed_raw.append(raw_bytes)
                                continue
                            else:
                                dt = _extract_delta_text(data, "anthropic")
                                if dt:
                                    text_chunks.append(dt)
                                    current_text_parts.append(dt)

                        # -- content_block_stop --
                        elif dtype == "content_block_stop":
                            if suppressing:
                                if current_vc_tool:
                                    input_str = "".join(
                                        current_vc_tool["input_parts"],
                                    )
                                    try:
                                        parsed_input = json.loads(input_str)
                                    except json.JSONDecodeError:
                                        parsed_input = {}
                                    vc_tools.append({
                                        "id": current_vc_tool["id"],
                                        "name": current_vc_tool["name"],
                                        "input": parsed_input,
                                    })
                                    all_content_blocks.append({
                                        "type": "tool_use",
                                        "id": current_vc_tool["id"],
                                        "name": current_vc_tool["name"],
                                        "input": parsed_input,
                                    })
                                    current_vc_tool = None
                                suppressing = False
                                suppressed_raw.append(raw_bytes)
                                continue
                            else:
                                # Finalize text block
                                if current_text_parts:
                                    all_content_blocks.append({
                                        "type": "text",
                                        "text": "".join(current_text_parts),
                                    })
                                    current_text_parts = []
                                    forwarded_block_count += 1

                        # -- message_delta: extract output_tokens --
                        elif dtype == "message_delta":
                            delta_usage = data.get("usage", {})
                            if "output_tokens" in delta_usage:
                                _stream_usage["output_tokens"] = delta_usage["output_tokens"]
                            sr = data.get("delta", {}).get("stop_reason")
                            if sr == "tool_use" and vc_tools:
                                if non_vc_tools:
                                    # BAIL: mixed VC + non-VC tools
                                    logger.warning(
                                        "Mixed VC + non-VC tools in "
                                        "response — passing all through",
                                    )
                                    for s in suppressed_raw:
                                        yield s
                                    suppressed_raw.clear()
                                    vc_tools.clear()
                                    need_continuation = False
                                else:
                                    # All VC — suppress, handle after
                                    # message_stop
                                    need_continuation = True
                                    suppressed_raw.append(raw_bytes)
                                    continue

                        # -- message_stop --
                        elif dtype == "message_stop":
                            if need_continuation:
                                suppressed_raw.append(raw_bytes)
                                continue

                        # ===== Responses API events =====

                        # -- response.output_item.added --
                        elif dtype == "response.output_item.added":
                            item = data.get("item", {})
                            itype = item.get("type", "")
                            if (
                                itype == "function_call"
                                and is_vc_tool(item.get("name", ""))
                            ):
                                suppressing = True
                                current_vc_tool = {
                                    "id": item.get(
                                        "call_id", item.get("id", ""),
                                    ),
                                    "name": item["name"],
                                    "input_parts": [],
                                }
                                suppressed_raw.append(raw_bytes)
                                continue
                            elif itype == "function_call":
                                non_vc_tools.append({
                                    "id": item.get(
                                        "call_id", item.get("id", ""),
                                    ),
                                    "name": item.get("name", ""),
                                })

                        # -- response.function_call_arguments.delta --
                        elif dtype == (
                            "response.function_call_arguments.delta"
                        ):
                            if suppressing and current_vc_tool:
                                current_vc_tool["input_parts"].append(
                                    data.get("delta", ""),
                                )
                                suppressed_raw.append(raw_bytes)
                                continue

                        # -- response.function_call_arguments.done --
                        elif dtype == (
                            "response.function_call_arguments.done"
                        ):
                            if suppressing and current_vc_tool:
                                args_str = data.get(
                                    "arguments",
                                    "".join(
                                        current_vc_tool["input_parts"],
                                    ),
                                )
                                try:
                                    parsed_input = json.loads(args_str)
                                except json.JSONDecodeError:
                                    parsed_input = {}
                                vc_tools.append({
                                    "id": current_vc_tool["id"],
                                    "name": current_vc_tool["name"],
                                    "input": parsed_input,
                                })
                                all_content_blocks.append({
                                    "type": "function_call",
                                    "call_id": current_vc_tool["id"],
                                    "name": current_vc_tool["name"],
                                    "arguments": args_str,
                                })
                                current_vc_tool = None
                                suppressing = False
                                suppressed_raw.append(raw_bytes)
                                continue

                        # -- response.output_text.delta --
                        elif dtype == "response.output_text.delta":
                            dt = data.get("delta", "")
                            if dt:
                                text_chunks.append(dt)
                                current_text_parts.append(dt)

                        # -- response.output_item.done --
                        elif dtype == "response.output_item.done":
                            if suppressing:
                                suppressed_raw.append(raw_bytes)
                                continue
                            item = data.get("item", {})
                            if item.get("type") == "message":
                                if current_text_parts:
                                    all_content_blocks.append({
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [{
                                            "type": "output_text",
                                            "text": "".join(
                                                current_text_parts,
                                            ),
                                        }],
                                    })
                                    current_text_parts = []
                                    forwarded_block_count += 1

                        # -- response.completed --
                        elif dtype == "response.completed":
                            resp = data.get("response", {})
                            output = resp.get("output", [])
                            has_vc = any(
                                it.get("type") == "function_call"
                                and is_vc_tool(it.get("name", ""))
                                for it in output
                            )
                            if has_vc and vc_tools:
                                has_non_vc = any(
                                    it.get("type") == "function_call"
                                    and not is_vc_tool(it.get("name", ""))
                                    for it in output
                                )
                                if has_non_vc:
                                    logger.warning(
                                        "Mixed VC + non-VC tools in "
                                        "Responses API — passing through",
                                    )
                                    for s in suppressed_raw:
                                        yield s
                                    suppressed_raw.clear()
                                    vc_tools.clear()
                                    need_continuation = False
                                else:
                                    need_continuation = True
                                    suppressed_raw.append(raw_bytes)
                                    continue

                        # Default: forward event to client
                        yield raw_bytes
            finally:
                await upstream.aclose()

            # --- Continuation phase ---
            if need_continuation and vc_tools and state:
                cont_body: dict | None = None
                cont_data: dict | None = None
                loop_content_blocks: list[dict] = []

                for loop_i in range(_MAX_CONTINUATION_LOOPS):
                    # Execute VC tools
                    tool_results: list[dict] = []
                    for tool in vc_tools:
                        t_tool = time.monotonic()
                        result_str = execute_vc_tool(
                            state.engine,
                            tool["name"],
                            tool["input"],
                        )
                        tool_ms = round(
                            (time.monotonic() - t_tool) * 1000, 1,
                        )
                        _input_preview = json.dumps(tool["input"])[:120]
                        _result_preview = result_str[:200].replace("\n", " ")
                        logger.info(
                            "TOOL_CALL %s %dms input=%s result_len=%d preview=%s",
                            tool["name"], tool_ms, _input_preview,
                            len(result_str), _result_preview,
                        )
                        if metrics:
                            metrics.record({
                                "type": "tool_intercept",
                                "turn": turn,
                                "tool_name": tool["name"],
                                "tool_input": tool["input"],
                                "result": result_str[:200],
                                "duration_ms": tool_ms,
                                "continuation_count": loop_i + 1,
                                "conversation_id": conversation_id,
                            })
                        if api_format == "openai_responses":
                            tool_results.append({
                                "type": "function_call_output",
                                "call_id": tool["id"],
                                "output": result_str,
                            })
                        else:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool["id"],
                                "content": result_str,
                            })

                    # Re-assemble context with updated working set
                    # so the LLM sees the expanded content in this
                    # turn (not deferred to next turn).
                    new_prepend = state.engine.reassemble_context()
                    reassembled_body = _inject_context(
                        body, new_prepend, api_format,
                    ) if new_prepend else body

                    # Build or extend continuation request
                    if cont_body is None:
                        cont_body = _build_continuation_request(
                            reassembled_body,
                            all_content_blocks,
                            tool_results,
                        )
                    else:
                        if api_format == "openai_responses":
                            # Update instructions with re-assembled context
                            if (
                                new_prepend
                                and "instructions" in reassembled_body
                            ):
                                cont_body["instructions"] = (
                                    reassembled_body["instructions"]
                                )
                            # Append function_call items + results to input
                            for block in loop_content_blocks:
                                if block.get("type") == "function_call":
                                    cont_body["input"].append(block)
                            cont_body["input"].extend(tool_results)
                        else:
                            # Anthropic / default
                            if (
                                new_prepend
                                and "system" in reassembled_body
                            ):
                                cont_body["system"] = (
                                    reassembled_body["system"]
                                )
                            cont_body["messages"].append({
                                "role": "assistant",
                                "content": loop_content_blocks,
                            })
                            cont_body["messages"].append({
                                "role": "user",
                                "content": tool_results,
                            })

                    # Send continuation — streaming for Responses API
                    # (Codex requires stream=true), non-streaming otherwise.
                    _cont_is_streaming = cont_body.get("stream", False)
                    if _cont_is_streaming:
                        cont_resp = await client.send(
                            client.build_request(
                                "POST", url, headers=headers, json=cont_body,
                            ),
                            stream=True,
                        )
                    else:
                        cont_resp = await client.post(
                            url, headers=headers, json=cont_body,
                        )

                    # Log continuation request/response to disk
                    _cont_resp_text: str | None = None
                    if _cont_is_streaming and cont_resp.status_code == 200:
                        # Collect SSE stream → extract response.completed
                        _cont_chunks: list[str] = []
                        cont_data = {}
                        async for line in cont_resp.aiter_lines():
                            _cont_chunks.append(line)
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if payload == "[DONE]":
                                break
                            try:
                                evt = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            if evt.get("type") == "response.completed":
                                cont_data = evt.get("response", {})
                        await cont_resp.aclose()
                        _cont_resp_text = json.dumps(cont_data, ensure_ascii=False)
                    else:
                        if _cont_is_streaming:
                            await cont_resp.aread()
                        _cont_resp_text = cont_resp.text

                    if request_log_dir and log_prefix:
                        _cn = loop_i + 1
                        try:
                            _cdir = Path(request_log_dir)
                            _cdir.joinpath(
                                f"{log_prefix}.continuation-{_cn}.request.json"
                            ).write_text(
                                json.dumps(cont_body, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            _cdir.joinpath(
                                f"{log_prefix}.continuation-{_cn}.response.json"
                            ).write_text(
                                _cont_resp_text or "", encoding="utf-8",
                            )
                        except Exception:
                            logger.debug("continuation log write failed", exc_info=True)  # never let logging break the request

                    logger.info(
                        "CONTINUATION round=%d status=%d tools=%s",
                        loop_i + 1,
                        cont_resp.status_code,
                        [t["name"] for t in vc_tools],
                    )

                    if cont_resp.status_code >= 300:
                        logger.error(
                            "Continuation failed: %d",
                            cont_resp.status_code,
                        )
                        if _cont_is_streaming:
                            try:
                                await cont_resp.aclose()
                            except Exception:
                                logger.debug("continuation response close failed", exc_info=True)
                        break

                    if not _cont_is_streaming:
                        cont_data = cont_resp.json()

                    if api_format == "openai_responses":
                        # Responses API: output array, no stop_reason
                        output = cont_data.get("output", [])
                        loop_content_blocks = output
                        tool_blocks = [
                            b for b in output
                            if b.get("type") == "function_call"
                        ]
                        vc_next = [
                            b for b in tool_blocks
                            if is_vc_tool(b.get("name", ""))
                        ]
                        has_tools = bool(tool_blocks)

                        # Emit text from message items
                        for item in output:
                            if item.get("type") != "message":
                                continue
                            for part in item.get("content", []):
                                if part.get("type") == "output_text":
                                    t = part.get("text", "")
                                    if t:
                                        text_chunks.append(t)
                                        for sse_evt in (
                                            _emit_text_as_responses_sse(
                                                t,
                                                forwarded_block_count,
                                            )
                                        ):
                                            yield sse_evt
                                        forwarded_block_count += 1

                        # More VC-only tool calls → loop
                        if (
                            has_tools
                            and vc_next
                            and all(
                                is_vc_tool(b.get("name", ""))
                                for b in tool_blocks
                            )
                        ):
                            fmt_obj = get_format("openai_responses")
                            vc_tools = [
                                {
                                    "id": c["id"],
                                    "name": c["name"],
                                    "input": c["input"],
                                }
                                for c in fmt_obj.extract_tool_calls(
                                    output,
                                )
                                if is_vc_tool(c["name"])
                            ]
                            continue

                        # Done — forward non-VC tools
                        non_vc_in_cont = [
                            b for b in tool_blocks
                            if not is_vc_tool(b.get("name", ""))
                        ]
                        if non_vc_in_cont:
                            fmt_obj = get_format("openai_responses")
                            for nvc in fmt_obj.extract_tool_calls(
                                non_vc_in_cont,
                            ):
                                for sse_evt in (
                                    _emit_tool_use_as_responses_sse(
                                        nvc,
                                        forwarded_block_count,
                                    )
                                ):
                                    yield sse_evt
                                forwarded_block_count += 1
                        break

                    else:
                        # Anthropic / default
                        stop_reason = cont_data.get(
                            "stop_reason", "end_turn",
                        )
                        content = cont_data.get("content", [])
                        loop_content_blocks = content

                        text_blocks = [
                            b for b in content
                            if b.get("type") == "text"
                        ]
                        tool_blocks = [
                            b for b in content
                            if b.get("type") == "tool_use"
                        ]
                        vc_next = [
                            b for b in tool_blocks
                            if is_vc_tool(b.get("name", ""))
                        ]

                        for tb in text_blocks:
                            t = tb.get("text", "")
                            if t:
                                text_chunks.append(t)
                                for sse_evt in _emit_text_as_sse(
                                    t, forwarded_block_count,
                                ):
                                    yield sse_evt
                                forwarded_block_count += 1

                        # More VC-only tool calls → loop
                        if (
                            stop_reason == "tool_use"
                            and vc_next
                            and all(
                                is_vc_tool(b.get("name", ""))
                                for b in tool_blocks
                            )
                        ):
                            vc_tools = [
                                {
                                    "id": b["id"],
                                    "name": b["name"],
                                    "input": b.get("input", {}),
                                }
                                for b in vc_next
                            ]
                            continue

                        # Done — forward non-VC tools
                        non_vc_in_cont = [
                            b for b in tool_blocks
                            if not is_vc_tool(b.get("name", ""))
                        ]
                        if non_vc_in_cont:
                            for nvc in non_vc_in_cont:
                                for sse_evt in _emit_tool_use_as_sse(
                                    nvc, forwarded_block_count,
                                ):
                                    yield sse_evt
                                forwarded_block_count += 1
                        break

                # Emit end events (format-aware)
                cont_usage = (
                    cont_data.get("usage") if cont_data else None
                )
                if api_format == "openai_responses":
                    # Collect all forwarded output items
                    _final_output: list[dict] = []
                    if cont_data:
                        for item in cont_data.get("output", []):
                            if item.get("type") == "function_call":
                                if not is_vc_tool(
                                    item.get("name", ""),
                                ):
                                    _final_output.append(item)
                            else:
                                _final_output.append(item)
                    for sse_evt in _emit_response_done_sse(
                        _final_output, usage=cont_usage,
                    ):
                        yield sse_evt
                else:
                    cont_stop = "end_turn"
                    if cont_data:
                        raw_stop = cont_data.get(
                            "stop_reason", "end_turn",
                        )
                        non_vc_forwarded = any(
                            b.get("type") == "tool_use"
                            and not is_vc_tool(b.get("name", ""))
                            for b in cont_data.get("content", [])
                        )
                        if raw_stop == "tool_use" and non_vc_forwarded:
                            cont_stop = "tool_use"
                    for sse_evt in _emit_message_end_sse(
                        cont_stop, usage=cont_usage,
                    ):
                        yield sse_evt

            # Post-stream processing
            assistant_text, _ = _post_stream(text_chunks, raw_events, usage=_stream_usage)
            if state and assistant_text:
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text,
                            timestamp=datetime.now(timezone.utc),
                            raw_content=all_content_blocks if all_content_blocks else None),
                )
                if not passthrough:
                    state.fire_turn_complete(
                        list(state.conversation_history),
                        payload_tokens=state._last_payload_tokens or None,
                        turn_id=turn_id,
                    )
                _marker_sid = state.engine.config.conversation_id
                _fmt = get_format(api_format)
                yield _fmt.emit_conversation_marker_sse(_marker_sid)

            if session_log_path and state:
                _dump_session_state(state, session_log_path)

            return  # exit — don't fall through to raw-byte path

        # ---------------------------------------------------------------
        # Non-paging path: raw-byte forwarding (unchanged)
        # ---------------------------------------------------------------
        line_buf = ""
        _raw_usage: dict = {}
        np_content_blocks: list[dict] = []
        np_current_text_parts: list[str] = []
        np_current_tool_input_parts: list[str] = []
        np_current_tool: dict | None = None
        try:
            async for raw_chunk in upstream.aiter_bytes():
                yield raw_chunk  # forward raw bytes unchanged

                # Side-channel: parse for text accumulation + log capture
                decoded = raw_chunk.decode("utf-8", errors="replace")
                raw_events.append(decoded)
                line_buf += decoded
                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                            delta = _extract_delta_text(data, api_format)
                            if delta:
                                text_chunks.append(delta)
                            # Extract usage from message_start / message_delta
                            _dtype = data.get("type", "")
                            if _dtype == "message_start":
                                _mu = data.get("message", {}).get("usage", {})
                                _extract_usage(_mu, _raw_usage)
                            elif _dtype == "message_delta":
                                _du = data.get("usage", {})
                                if "output_tokens" in _du:
                                    _raw_usage["output_tokens"] = _du["output_tokens"]

                            # Content block tracking for raw_content
                            if _dtype == "content_block_start":
                                block = data.get("content_block", {})
                                btype = block.get("type", "")
                                if btype == "tool_use":
                                    np_current_tool = {
                                        "type": "tool_use",
                                        "id": block.get("id", ""),
                                        "name": block.get("name", ""),
                                    }
                                    np_current_tool_input_parts = []
                                elif btype == "text":
                                    np_current_text_parts = []
                            elif _dtype == "content_block_delta":
                                cb_delta = data.get("delta", {})
                                if cb_delta.get("type") == "input_json_delta" and np_current_tool is not None:
                                    np_current_tool_input_parts.append(cb_delta.get("partial_json", ""))
                                elif cb_delta.get("type") == "text_delta":
                                    np_current_text_parts.append(cb_delta.get("text", ""))
                            elif _dtype == "content_block_stop":
                                if np_current_tool is not None:
                                    input_str = "".join(np_current_tool_input_parts)
                                    try:
                                        parsed_input = json.loads(input_str) if input_str else {}
                                    except json.JSONDecodeError:
                                        parsed_input = {}
                                    np_current_tool["input"] = parsed_input
                                    np_content_blocks.append(np_current_tool)
                                    np_current_tool = None
                                    np_current_tool_input_parts = []
                                elif np_current_text_parts:
                                    np_content_blocks.append({
                                        "type": "text",
                                        "text": "".join(np_current_text_parts),
                                    })
                                    np_current_text_parts = []
                        except json.JSONDecodeError:
                            pass
        finally:
            await upstream.aclose()
            assistant_text, _ = _post_stream(text_chunks, raw_events, usage=_raw_usage)
            if state and assistant_text:
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text,
                            timestamp=datetime.now(timezone.utc),
                            raw_content=np_content_blocks if np_content_blocks else None)
                )
                if not passthrough:
                    state.fire_turn_complete(
                        list(state.conversation_history),
                        payload_tokens=state._last_payload_tokens or None,
                        turn_id=turn_id,
                    )

                # Inject session marker as a final SSE delta so the client SDK
                # accumulates it into the stored assistant message.
                _marker_sid = state.engine.config.conversation_id
                _fmt = get_format(api_format)
                yield _fmt.emit_conversation_marker_sse(_marker_sid)

            # Session state dump (after response + history update)
            if session_log_path and state:
                _dump_session_state(state, session_log_path)

    async def stream_generator():
        client_chunks: list[bytes] = [] if request_log_dir else None
        async for chunk in _inner_stream():
            if client_chunks is not None:
                client_chunks.append(chunk)
            yield chunk
        # 4-to-client: log everything sent back to the client
        if client_chunks is not None and log_prefix:
            try:
                Path(request_log_dir).joinpath(
                    f"{log_prefix}.4-to-client.txt",
                ).write_bytes(b"".join(client_chunks))
            except Exception:
                logger.debug("client response log write failed", exc_info=True)

    return StreamingResponse(
        stream_generator(),
        status_code=upstream.status_code,
        headers=resp_headers,
    )


async def _handle_non_streaming(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict,
    api_format: str,
    state: ProxyState | None,
    *,
    metrics: ProxyMetrics | None = None,
    turn: int = 0,
    turn_id: str = "",
    overhead_ms: float = 0.0,
    conversation_id: str = "",
    passthrough: bool = False,
    response_log_path: object | None = None,
    session_log_path: object | None = None,
    request_log_dir: object | None = None,
    log_prefix: str = "",
) -> JSONResponse:
    """Forward JSON response, parse assistant text, fire on_turn_complete."""
    t_upstream = time.monotonic()
    resp = await client.request("POST", url, headers=headers, json=body)
    upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)

    try:
        response_body = resp.json()
    except Exception:
        return JSONResponse(content=resp.text, status_code=resp.status_code)

    # 3-from-llm: raw upstream response before any modification
    if request_log_dir and log_prefix:
        try:
            Path(request_log_dir).joinpath(
                f"{log_prefix}.3-from-llm.json",
            ).write_text(
                json.dumps(response_body, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("upstream response log write failed", exc_info=True)

    # Capture response for dashboard inspector
    _ns_raw = response_body.get("usage", {})
    _ns_usage: dict = {}
    _extract_usage(_ns_raw, _ns_usage)
    if metrics:
        metrics.capture_response(
            turn, response_body,
            upstream_input_tokens=_ns_usage.get("input_tokens", 0),
            upstream_output_tokens=_ns_raw.get("output_tokens", 0),
            cache_creation_input_tokens=_ns_usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=_ns_usage.get("cache_read_input_tokens", 0),
        )

    # Extract and record assistant text
    assistant_text = _extract_assistant_text(response_body, api_format)
    if state and assistant_text:
        state.conversation_history.append(
            Message(role="assistant", content=assistant_text,
                    timestamp=datetime.now(timezone.utc),
                    raw_content=_extract_assistant_raw_content(response_body, api_format))
        )
        if not passthrough:
            state.fire_turn_complete(
                list(state.conversation_history),
                payload_tokens=state._last_payload_tokens or None,
                turn_id=turn_id,
            )

        # Inject session marker into the response body so the client stores it
        conversation_id = state.engine.config.conversation_id
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        response_body = _inject_conversation_marker(response_body, marker, api_format)

    if metrics:
        metrics.record({
            "type": "response",
            "turn": turn,
            "turn_id": turn_id,
            "upstream_ms": upstream_ms,
            "total_ms": round(overhead_ms + upstream_ms, 1),
            "streaming": False,
            "conversation_id": conversation_id,
        })

    # Log upstream LLM call to telemetry ledger
    if state and hasattr(state.engine, '_telemetry'):
        _model = body.get("model", "unknown")
        _out_tok = len(assistant_text) // 4 if assistant_text else 0
        _in_tok = getattr(state, '_last_enriched_payload_tokens', 0) or 0
        state.engine._telemetry.log(
            component="proxy_upstream",
            model=_model,
            input_tokens=_in_tok,
            output_tokens=_out_tok,
            duration_ms=upstream_ms,
            detail="non_streaming",
        )

    logger.info(
        "T%d RESPONSE stream=False llm=%dms total=%dms chars=%d",
        turn, int(upstream_ms), int(round(overhead_ms + upstream_ms)),
        len(assistant_text or ""),
    )

    # Session state dump
    if session_log_path and state:
        _dump_session_state(state, session_log_path)

    # Forward response headers (filter hop-by-hop)
    resp_headers = _forward_headers(dict(resp.headers))

    # 4-to-client: response body after session marker injection
    if request_log_dir and log_prefix:
        try:
            Path(request_log_dir).joinpath(
                f"{log_prefix}.4-to-client.json",
            ).write_text(
                json.dumps(response_body, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("client response log write failed", exc_info=True)

    return JSONResponse(
        content=response_body,
        status_code=resp.status_code,
        headers=resp_headers,
    )
