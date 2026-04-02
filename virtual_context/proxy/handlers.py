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


def _restore_tool_stub_in_place(body: dict, fmt, ref: str, full_content: str) -> bool:
    """Replace a tool output stub containing *ref* with *full_content* in place.

    Walks all messages/items in *body* looking for tool output carriers whose
    content contains the ref string.  When found, replaces the stub content
    with the full stored tool output at the original position.

    Supports Anthropic (tool_result blocks), OpenAI Chat (role="tool" messages),
    and OpenAI Responses (function_call_output items).

    Returns True if a matching stub was found and replaced, False otherwise.
    """
    messages = fmt.get_messages(body)
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        # --- Anthropic: tool_result content blocks in user messages ---
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    # Check if this block's content contains the ref
                    block_content = block.get("content", "")
                    if isinstance(block_content, str) and ref in block_content:
                        block["content"] = full_content
                        return True
                    elif isinstance(block_content, list):
                        for sub in block_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                if ref in sub.get("text", ""):
                                    # Replace entire block content, not just the text sub-block
                                    block["content"] = full_content
                                    return True

        # --- OpenAI Chat: role="tool" messages ---
        if msg.get("role") == "tool":
            msg_content = msg.get("content", "")
            if isinstance(msg_content, str) and ref in msg_content:
                msg["content"] = full_content
                return True

        # --- OpenAI Responses: function_call_output items ---
        if msg.get("type") == "function_call_output":
            output = msg.get("output", "")
            if isinstance(output, str) and ref in output:
                msg["output"] = full_content
                return True

    return False


def _msg_key_for_format(fmt) -> str:
    """Return the body key that holds the message list for a given format."""
    name = fmt.name
    if name == "gemini":
        return "contents"
    if name == "openai_responses":
        return "input"
    # anthropic, openai (chat)
    return "messages"


def _restore_chain_in_place(
    body: dict,
    fmt,
    ref: str,
    chain_messages: list[dict],
) -> bool:
    """Replace a chain stub pair with the full chain messages in place.

    A chain stub pair is:
      1. A user message whose content contains ``[Compacted turn``
      2. Immediately followed by an assistant message whose content contains *ref*

    Both messages are removed and replaced with *chain_messages* (N messages).
    Returns True if the stub pair was found and replaced, False otherwise.
    """
    messages = fmt.get_messages(body)
    msg_key = _msg_key_for_format(fmt)

    new_messages: list[dict] = []
    i = 0
    found = False
    while i < len(messages):
        msg = messages[i]
        # Check if this is the stub user message and the next is the stub
        # assistant with our chain ref.
        if (
            not found
            and i + 1 < len(messages)
            and isinstance(msg, dict)
            and _msg_text_contains(msg, "[Compacted turn")
            and _msg_text_contains(messages[i + 1], ref)
        ):
            # Prefix first user message so model knows this was recovered
            _tagged = list(chain_messages)
            for _ci, _cm in enumerate(_tagged):
                if isinstance(_cm, dict) and _cm.get("role") in ("user", "human"):
                    _cm = dict(_cm)  # shallow copy
                    _cc = _cm.get("content", "")
                    _prefix = (
                        "[Previously compacted — restored by vc_restore_tool. "
                        "This content was NOT visible before this restore. "
                        "It was not there. Use the recovered content directly "
                        "to answer the user's question.]\n"
                    )
                    if isinstance(_cc, str):
                        _cm["content"] = _prefix + _cc
                    elif isinstance(_cc, list) and _cc:
                        _first = _cc[0]
                        if isinstance(_first, dict) and _first.get("type") == "text":
                            _cc = list(_cc)
                            _cc[0] = dict(_first)
                            _cc[0]["text"] = _prefix + _cc[0].get("text", "")
                            _cm["content"] = _cc
                    _tagged[_ci] = _cm
                    break
            new_messages.extend(_tagged)
            i += 2  # skip the stub pair
            found = True
        else:
            new_messages.append(msg)
            i += 1

    if found:
        body[msg_key] = new_messages
    return found


def _msg_text_contains(msg: dict, needle: str) -> bool:
    """Check whether any text content of *msg* contains *needle*.

    Handles string content, list-of-blocks content (Anthropic), and
    OpenAI Responses ``output`` fields.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return needle in content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if needle in block.get("text", ""):
                    return True
                # tool_result blocks can have nested content
                sub = block.get("content", "")
                if isinstance(sub, str) and needle in sub:
                    return True
            elif isinstance(block, str) and needle in block:
                return True
    # OpenAI Responses: output field
    output = msg.get("output", "")
    if isinstance(output, str) and needle in output:
        return True
    return False


class _ProxyToolRuntime:
    """Shared VC tool runtime backed by the proxy's mutable request body."""

    def __init__(
        self,
        engine,
        api_format: str,
        conversation_id: str,
        get_target_body,
    ) -> None:
        self._engine = engine
        self._api_format = api_format
        self._conversation_id = conversation_id
        self._get_target_body = get_target_body

    def has_restorable_stubs(self) -> bool:
        return True

    def restore_tool_output(self, ref: str) -> dict:
        if ref.startswith("media_"):
            return self._restore_media(ref)

        if ref.startswith("chain_"):
            return self._restore_chain(ref)

        full_content = self._engine._store.get_tool_output_by_ref(
            self._conversation_id, ref,
        )
        if full_content is None:
            return {"error": f"tool output ref {ref} not found"}

        target = self._get_target_body()
        if not isinstance(target, dict):
            return {"error": "no mutable payload available for vc_restore_tool"}

        fmt = get_format(self._api_format)
        restored = _restore_tool_stub_in_place(
            target, fmt, ref, full_content,
        )
        if not restored:
            return {"error": f"stub for ref {ref} not found in current payload"}
        return (
            "Restored. The tool output was compacted and has been recovered "
            "into your conversation history above. Use the recovered content "
            "to answer the user's question."
        )

    def _restore_media(self, ref: str) -> dict | list:
        """Restore a media stub by returning the image in the tool result."""
        import base64
        from .media import build_media_restore_result

        metadata = self._engine._store.get_media_output(self._conversation_id, ref)
        if metadata is None:
            return {"error": f"media ref {ref} not found"}

        file_path = metadata["file_path"]
        try:
            with open(file_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("ascii")
        except FileNotFoundError:
            return {"error": f"media file not found: {file_path}"}

        return build_media_restore_result(
            b64_data=b64_data,
            media_type=metadata["media_type"],
            width=metadata["width"],
            height=metadata["height"],
        )

    def _restore_chain(self, ref: str) -> dict:
        """Restore a chain stub pair to the full message chain."""
        snapshot = self._engine._store.get_chain_snapshot(
            self._conversation_id, ref,
        )
        if snapshot is None:
            return {"error": f"chain snapshot ref {ref} not found"}

        try:
            chain = json.loads(snapshot["chain_json"])
        except (json.JSONDecodeError, TypeError):
            return {"error": f"invalid chain_json for ref {ref}"}

        # Rehydrate any stubbed tool results within the chain.
        # The chain snapshot stores messages from pre_filter_body, so tool
        # results normally have their full content.  However, if stage 1
        # (tool result stubbing) ran before stage 2 (chain collapse) in
        # the same request, some tool_result blocks may contain stub text
        # with vc_restore_tool refs.  Rehydrate those from tool_outputs.
        tool_output_refs_str = snapshot.get("tool_output_refs", "")
        if tool_output_refs_str:
            tool_refs = [
                r.strip() for r in tool_output_refs_str.split(",") if r.strip()
            ]
            for msg in chain:
                if not isinstance(msg, dict):
                    continue
                self._rehydrate_tool_results_in_message(msg, tool_refs)

        # Strip trailing tool_use blocks from the last assistant message.
        # The chain may end with assistant[thinking, text, tool_use] where the
        # corresponding tool_result is in the NEXT turn chain (not stored in
        # this snapshot). Leaving the orphaned tool_use causes Anthropic 400.
        if chain:
            last = chain[-1]
            if isinstance(last, dict) and last.get("role") in ("assistant", "model"):
                content = last.get("content", [])
                if isinstance(content, list):
                    has_tool_use = any(
                        isinstance(b, dict) and b.get("type") == "tool_use"
                        for b in content
                    )
                    if has_tool_use:
                        cleaned = [
                            b for b in content
                            if not (isinstance(b, dict) and b.get("type") == "tool_use")
                        ]
                        if cleaned:
                            last = dict(last)
                            last["content"] = cleaned
                            chain[-1] = last
                        else:
                            # All content was tool_use — drop the message entirely
                            chain = chain[:-1]

        # Strip leading orphaned tool_result blocks from the first message.
        # When the previous chain's trailing tool_use was stripped above,
        # this chain's first user message may reference those gone IDs.
        # Collect tool_use IDs present in this chain to know which are valid.
        if chain:
            _chain_tool_use_ids: set[str] = set()
            for _cm in chain:
                if not isinstance(_cm, dict):
                    continue
                _cc = _cm.get("content", [])
                if isinstance(_cc, list):
                    for _cb in _cc:
                        if (
                            isinstance(_cb, dict)
                            and _cb.get("type") == "tool_use"
                        ):
                            _tid = _cb.get("id")
                            if _tid:
                                _chain_tool_use_ids.add(_tid)

            first = chain[0]
            if isinstance(first, dict) and first.get("role") in ("user", "human"):
                fc = first.get("content", [])
                if isinstance(fc, list):
                    orphaned = [
                        b for b in fc
                        if (
                            isinstance(b, dict)
                            and b.get("type") == "tool_result"
                            and b.get("tool_use_id") not in _chain_tool_use_ids
                        )
                    ]
                    if orphaned:
                        cleaned_first = [
                            b for b in fc
                            if not (
                                isinstance(b, dict)
                                and b.get("type") == "tool_result"
                                and b.get("tool_use_id") not in _chain_tool_use_ids
                            )
                        ]
                        if cleaned_first:
                            first = dict(first)
                            first["content"] = cleaned_first
                            chain[0] = first
                        else:
                            chain = chain[1:]

        target = self._get_target_body()
        if not isinstance(target, dict):
            return {"error": "no mutable payload available for chain restore"}

        fmt = get_format(self._api_format)
        restored = _restore_chain_in_place(target, fmt, ref, chain)
        if not restored:
            return {"error": f"stub pair for chain ref {ref} not found in payload"}
        return (
            f"Restored. {len(chain)} messages recovered from compacted storage "
            f"and spliced into your conversation history above. This content was "
            f"previously compacted — it was NOT visible before this restore. "
            f"Do not apologize for not seeing it earlier; it was not there. "
            f"Use the recovered content directly to answer the user's question."
        )

    def _rehydrate_tool_results_in_message(
        self, msg: dict, tool_refs: list[str],
    ) -> None:
        """Replace any stubbed tool_result content in *msg* with full content.

        Checks if text in tool_result blocks contains ``vc_restore_tool``
        and any of the known *tool_refs*.  If so, looks up the full content
        from the store and replaces in place.
        """
        content = msg.get("content", [])
        if not isinstance(content, list):
            # String content — check for stub text
            if isinstance(content, str) and "vc_restore_tool" in content:
                for ref in tool_refs:
                    if ref in content:
                        full = self._engine._store.get_tool_output_by_ref(
                            self._conversation_id, ref,
                        )
                        if full is not None:
                            msg["content"] = full
                        return
            return

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            block_content = block.get("content", "")
            if isinstance(block_content, str) and "vc_restore_tool" in block_content:
                for ref in tool_refs:
                    if ref in block_content:
                        full = self._engine._store.get_tool_output_by_ref(
                            self._conversation_id, ref,
                        )
                        if full is not None:
                            block["content"] = full
                        break
            elif isinstance(block_content, list):
                for sub in block_content:
                    if (
                        isinstance(sub, dict)
                        and sub.get("type") == "text"
                        and "vc_restore_tool" in sub.get("text", "")
                    ):
                        for ref in tool_refs:
                            if ref in sub.get("text", ""):
                                full = self._engine._store.get_tool_output_by_ref(
                                    self._conversation_id, ref,
                                )
                                if full is not None:
                                    block["content"] = full
                                break
                        break


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
    skip_marker_injection: bool = False,
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
                "input_tokens": _usage.get("input_tokens", 0),
                "upstream_input_tokens": _usage.get("input_tokens", 0),
                "cache_creation_input_tokens": _usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": _usage.get("cache_read_input_tokens", 0),
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
                conversation_id=conversation_id,
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

            _pg_stream_t0 = time.monotonic()
            _pg_first_chunk = True
            _pg_chunk_count = 0
            _pg_total_bytes = 0
            _pg_last_chunk_at = _pg_stream_t0
            _pg_max_gap = 0.0
            try:
                async for raw_chunk in upstream.aiter_bytes():
                    _pg_now = time.monotonic()
                    _pg_gap = _pg_now - _pg_last_chunk_at
                    if _pg_gap > _pg_max_gap:
                        _pg_max_gap = _pg_gap
                    _pg_last_chunk_at = _pg_now
                    _pg_chunk_count += 1
                    _pg_total_bytes += len(raw_chunk)
                    if _pg_first_chunk:
                        _pg_first_chunk = False
                        logger.info(
                            "STREAM_FIRST_BYTE conv=%s turn=%d after=%.1fs (paging)",
                            conversation_id[:12], turn, _pg_now - _pg_stream_t0,
                        )
                    if _pg_gap > 30.0:
                        logger.warning(
                            "STREAM_STALL conv=%s turn=%d gap=%.1fs chunks=%d bytes=%d elapsed=%.1fs (paging)",
                            conversation_id[:12], turn, _pg_gap,
                            _pg_chunk_count, _pg_total_bytes, _pg_now - _pg_stream_t0,
                        )
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
                                # Count ALL forwarded blocks (text, thinking, etc.)
                                # so continuation emits use correct indices.
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
                _pg_elapsed = time.monotonic() - _pg_stream_t0
                logger.info(
                    "STREAM_END conv=%s turn=%d elapsed=%.1fs chunks=%d bytes=%d max_gap=%.1fs (paging)",
                    conversation_id[:12], turn, _pg_elapsed,
                    _pg_chunk_count, _pg_total_bytes, _pg_max_gap,
                )
                if _pg_elapsed > 60.0:
                    logger.warning(
                        "STREAM_SLOW conv=%s turn=%d elapsed=%.1fs max_gap=%.1fs chunks=%d (paging) — investigate",
                        conversation_id[:12], turn, _pg_elapsed, _pg_max_gap, _pg_chunk_count,
                    )
                await upstream.aclose()

            # --- Continuation phase ---
            if need_continuation and vc_tools and state:
                cont_body: dict | None = None
                cont_data: dict | None = None
                loop_content_blocks: list[dict] = []
                tool_runtime = _ProxyToolRuntime(
                    engine=state.engine,
                    api_format=api_format,
                    conversation_id=conversation_id,
                    get_target_body=lambda: cont_body if cont_body is not None else body,
                )

                for loop_i in range(_MAX_CONTINUATION_LOOPS):
                    # Execute VC tools
                    import uuid as _uuid
                    _group_id = _uuid.uuid4().hex[:12]
                    tool_results: list[dict] = []
                    for tool in vc_tools:
                        t_tool = time.monotonic()
                        tool_name = tool["name"]
                        tool_input = tool["input"]
                        result_str = execute_vc_tool(
                            state.engine,
                            tool_name,
                            tool_input,
                            tool_runtime=tool_runtime,
                        )
                        tool_ms = round(
                            (time.monotonic() - t_tool) * 1000, 1,
                        )
                        _input_preview = json.dumps(tool_input)[:120]
                        _result_preview = result_str[:200].replace("\n", " ")
                        logger.info(
                            "TOOL_CALL %s %dms input=%s result_len=%d preview=%s",
                            tool_name, tool_ms, _input_preview,
                            len(result_str), _result_preview,
                        )
                        if metrics:
                            metrics.record({
                                "type": "tool_intercept",
                                "turn": turn,
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                                "result": result_str[:200],
                                "duration_ms": tool_ms,
                                "continuation_count": loop_i + 1,
                                "conversation_id": conversation_id,
                                "group_id": _group_id,
                            })
                        # Persist full tool call to store
                        try:
                            state.engine._store.save_tool_call({
                                "conversation_id": conversation_id,
                                "request_turn": turn,
                                "round": loop_i + 1,
                                "group_id": _group_id,
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                                "tool_result": result_str,
                                "result_length": len(result_str),
                                "duration_ms": tool_ms,
                                "found": "not found" not in result_str.lower()[:100] if result_str else None,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                        except Exception:
                            pass
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
                        _err_body = ""
                        try:
                            _err_body = (await cont_resp.aread()).decode("utf-8", errors="replace")[:500]
                        except Exception:
                            pass
                        logger.error(
                            "Continuation failed: %d body=%s",
                            cont_resp.status_code,
                            _err_body,
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
                            if not t:
                                continue
                            # Non-streaming continuations may embed
                            # <thinking>...</thinking> as literal text.
                            # Split into proper thinking + text blocks.
                            import re as _re
                            _think_match = _re.match(
                                r"<thinking>([\s\S]*?)</thinking>\s*",
                                t,
                            )
                            if _think_match:
                                t = t[_think_match.end():]
                            if t.strip():
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
            if state and assistant_text and not state.is_conversation_deleted():
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text,
                            timestamp=datetime.now(timezone.utc),
                            raw_content=all_content_blocks if all_content_blocks else None),
                )
                state.persist_completed_turn()
                if not passthrough:
                    state.fire_turn_complete(
                        list(state.conversation_history),
                        payload_tokens=state._last_enriched_payload_tokens or None,
                        turn_id=turn_id,
                    )

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
        _stream_t0 = time.monotonic()
        _stream_first_chunk = True
        _stream_chunk_count = 0
        _stream_total_bytes = 0
        _stream_last_chunk_at = _stream_t0
        _stream_max_gap = 0.0
        try:
            async for raw_chunk in upstream.aiter_bytes():
                _now = time.monotonic()
                _gap = _now - _stream_last_chunk_at
                if _gap > _stream_max_gap:
                    _stream_max_gap = _gap
                _stream_last_chunk_at = _now
                _stream_chunk_count += 1
                _stream_total_bytes += len(raw_chunk)

                if _stream_first_chunk:
                    _stream_first_chunk = False
                    logger.info(
                        "STREAM_FIRST_BYTE conv=%s turn=%d after=%.1fs",
                        conversation_id[:12], turn,
                        _now - _stream_t0,
                    )

                # Log if any gap between chunks exceeds 30 seconds
                if _gap > 30.0:
                    logger.warning(
                        "STREAM_STALL conv=%s turn=%d gap=%.1fs chunks=%d bytes=%d elapsed=%.1fs",
                        conversation_id[:12], turn, _gap,
                        _stream_chunk_count, _stream_total_bytes,
                        _now - _stream_t0,
                    )

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
            _stream_elapsed = time.monotonic() - _stream_t0
            logger.info(
                "STREAM_END conv=%s turn=%d elapsed=%.1fs chunks=%d bytes=%d max_gap=%.1fs",
                conversation_id[:12], turn, _stream_elapsed,
                _stream_chunk_count, _stream_total_bytes, _stream_max_gap,
            )
            if _stream_elapsed > 60.0:
                logger.warning(
                    "STREAM_SLOW conv=%s turn=%d elapsed=%.1fs max_gap=%.1fs chunks=%d — investigate",
                    conversation_id[:12], turn, _stream_elapsed, _stream_max_gap, _stream_chunk_count,
                )
            await upstream.aclose()
            assistant_text, _ = _post_stream(text_chunks, raw_events, usage=_raw_usage)
            if state and assistant_text and not state.is_conversation_deleted():
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text,
                            timestamp=datetime.now(timezone.utc),
                            raw_content=np_content_blocks if np_content_blocks else None)
                )
                state.persist_completed_turn()
                if not passthrough:
                    state.fire_turn_complete(
                        list(state.conversation_history),
                        payload_tokens=state._last_enriched_payload_tokens or None,
                        turn_id=turn_id,
                    )

                # Inject session marker as a final SSE delta so the client SDK
                # accumulates it into the stored assistant message.
                # Skip if the inbound already had a marker — never overwrite.
                if not skip_marker_injection:
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
    skip_marker_injection: bool = False,
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
            conversation_id=conversation_id,
        )

    # Extract and record assistant text
    assistant_text = _extract_assistant_text(response_body, api_format)
    if state and assistant_text and not state.is_conversation_deleted():
        state.conversation_history.append(
            Message(role="assistant", content=assistant_text,
                    timestamp=datetime.now(timezone.utc),
                    raw_content=_extract_assistant_raw_content(response_body, api_format))
        )
        state.persist_completed_turn()
        if not passthrough:
            state.fire_turn_complete(
                list(state.conversation_history),
                payload_tokens=state._last_enriched_payload_tokens or None,
                turn_id=turn_id,
            )

        # Inject session marker into the response body so the client stores it.
        # Skip if the inbound payload already had a marker — never overwrite
        # an established conversation identity with a new one.
        if not skip_marker_injection:
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
            "input_tokens": _ns_usage.get("input_tokens", 0),
            "upstream_input_tokens": _ns_usage.get("input_tokens", 0),
            "cache_creation_input_tokens": _ns_usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": _ns_usage.get("cache_read_input_tokens", 0),
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


# -------------------------------------------------------------------
# VCATTACH handler
# -------------------------------------------------------------------

async def _handle_vcattach(
    result,
    fmt,
    state,
    registry,
    *,
    labels: dict | None = None,
    conv_ids: list | None = None,
):
    """Handle VCATTACH command — resolve target, execute attach, return fake response."""
    from starlette.responses import StreamingResponse, JSONResponse
    from .vcattach import resolve_target, execute_attach

    target_raw = result.vcattach_label

    # In cloud mode, labels and conv_ids are passed from the tenant registry.
    # In standalone mode, fall back to in-memory sessions + store stats.
    if conv_ids is None:
        conv_ids = list(registry._conversations.keys()) if registry else []
        if state and state.engine._store:
            _stats = getattr(state.engine._store, "get_conversation_stats", None)
            if callable(_stats):
                try:
                    for s in _stats():
                        cid = getattr(s, "conversation_id", "")
                        if cid and cid not in conv_ids:
                            conv_ids.append(cid)
                except Exception:
                    pass
    target_id, target_label, error = resolve_target(
        target_raw, result.conversation_id, conv_ids, labels=labels or {},
    )

    if error:
        if result.is_streaming:
            return StreamingResponse(
                iter([fmt.emit_fake_response_sse(error, result.conversation_id)]),
                media_type="text/event-stream",
            )
        return JSONResponse(fmt.build_fake_response(error, result.conversation_id))

    # Execute attach with full authoritative delete
    _store = state.engine._store if state else None
    _inner = getattr(_store, '_store', _store) if _store else None

    def _full_delete(cid):
        if _inner:
            begin = getattr(_inner, "begin_conversation_deletion", None)
            if callable(begin):
                begin(cid)
        target_state = registry.remove_conversation(cid) if registry else None
        if target_state is not None:
            try:
                target_state.reset_for_conversation_deletion(cid, authoritative=True)
            except Exception:
                pass
            try:
                target_state.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        if _inner and hasattr(_inner, "delete_conversation"):
            _inner.delete_conversation(cid)
        if target_state and hasattr(target_state.engine, "_session_cache"):
            cache = target_state.engine._session_cache
            if cache:
                try:
                    cache.delete_conversation(cid)
                except Exception:
                    pass

    def _reset_target(tid):
        if not _inner:
            return
        _load = getattr(_inner, "load_engine_state", None)
        _save = getattr(_inner, "save_engine_state", None)
        if callable(_load) and callable(_save):
            existing = _load(tid)
            if existing:
                existing.compacted_through = 0
                existing.last_compacted_turn = -1
                existing.last_completed_turn = -1
                existing.last_indexed_turn = -1
                existing.turn_tag_entries = []
                _save(existing)

    execute_attach(
        old_id=result.conversation_id,
        target_id=target_id,
        store=_inner,
        # Core proxy: local eviction only. Cloud path: per-request Redis hydration
        # ensures all workers see the reset state on next request.
        registry_invalidate=registry.remove_conversation if registry else None,
        delete_conversation=_full_delete,
        reset_engine_state=_reset_target,
    )

    text = f"Conversation attached to {target_label} ({target_id}). History restored."
    if result.is_streaming:
        return StreamingResponse(
            iter([fmt.emit_fake_response_sse(text, target_id)]),
            media_type="text/event-stream",
        )
    return JSONResponse(fmt.build_fake_response(text, target_id))


async def _handle_vc_command(
    result,
    fmt,
    state,
    registry,
    *,
    tenant_registry=None,
    tenant_id: str | None = None,
):
    """Dispatch all VC commands (attach, label, status, recall, compact, list, forget)."""
    from starlette.responses import StreamingResponse, JSONResponse

    cmd = result.vc_command
    arg = result.vc_command_arg
    conv_id = result.conversation_id

    if cmd == "attach":
        labels = {}
        conv_ids = None
        if tenant_registry and tenant_id:
            labels = tenant_registry.get_conversation_labels(tenant_id)
            conv_ids = tenant_registry.list_persisted_conversation_ids(tenant_id)
        return await _handle_vcattach(
            result, fmt, state, registry,
            labels=labels, conv_ids=conv_ids,
        )

    if cmd == "label":
        text = _handle_vclabel(arg, conv_id, state, tenant_registry, tenant_id)
    elif cmd == "status":
        text = _handle_vcstatus(conv_id, state, tenant_registry, tenant_id)
    elif cmd == "recall":
        text = _handle_vcrecall(arg, state)
    elif cmd == "compact":
        text = _handle_vccompact(state)
    elif cmd == "list":
        text = _handle_vclist(tenant_registry, tenant_id)
    elif cmd == "forget":
        text = _handle_vcforget(arg, state)
    else:
        text = f"Unknown VC command: {cmd}"

    if result.is_streaming:
        return StreamingResponse(
            iter([fmt.emit_fake_response_sse(text, conv_id)]),
            media_type="text/event-stream",
        )
    return JSONResponse(fmt.build_fake_response(text, conv_id))


def _handle_vclabel(label: str, conv_id: str, state, tenant_registry, tenant_id):
    """Set or show the current conversation's label."""
    if not label:
        # Show current label
        if tenant_registry and tenant_id:
            labels = tenant_registry.get_conversation_labels(tenant_id)
            current = labels.get(conv_id, "")
            if current:
                return f"Current label: {current}"
            return f"No label set. Use VCLABEL <name> to set one."
        return "Labels not available (no tenant registry)."

    if not tenant_registry or not tenant_id:
        return "Labels not available (no tenant registry)."

    tenant_registry.set_conversation_label(tenant_id, conv_id, label)

    # Emit event so dashboard SSE updates
    if state and state.metrics:
        state.metrics.record({
            "type": "label_changed",
            "conversation_id": conv_id,
            "label": label,
        })

    return f"Label set to '{label}'"


def _handle_vcstatus(conv_id: str, state, tenant_registry, tenant_id):
    """Return conversation status summary."""
    if not state:
        return "No active conversation."

    engine = state.engine
    es = engine._engine_state
    tti = engine._turn_tag_index

    label = ""
    if tenant_registry and tenant_id:
        labels = tenant_registry.get_conversation_labels(tenant_id)
        label = labels.get(conv_id, "")

    turns = len(tti.entries) if tti else 0
    compacted = getattr(es, "compacted_through", 0)
    generation = getattr(es, "conversation_generation", 0)

    # Segment count from store
    segments = 0
    try:
        store = engine._store
        stats = getattr(store, "get_conversation_stats", None)
        if callable(stats):
            for s in stats():
                if getattr(s, "conversation_id", "") == conv_id:
                    segments = getattr(s, "segment_count", 0)
                    break
    except Exception:
        pass

    # Working set
    ws = getattr(engine, "_paging", None)
    ws_entries = list(ws.working_set.values()) if ws and hasattr(ws, "working_set") else []
    ws_tokens = sum(e.tokens for e in ws_entries)

    # Active tags
    active_tags = sorted(tti.get_active_tags(lookback=6)) if tti else []

    lines = [
        f"Conversation: {conv_id}",
    ]
    if label:
        lines.append(f"Label: {label}")
    lines.extend([
        f"Turns: {turns} (compacted through {compacted})",
        f"Segments: {segments}",
        f"Generation: {generation}",
        f"Working set: {len(ws_entries)} tags, {ws_tokens:,} tokens",
        f"Active tags: {', '.join(active_tags[:15]) if active_tags else 'none'}",
    ])
    return "\n".join(lines)


def _handle_vcrecall(query: str, state):
    """Search for content and promote matching tags to working set."""
    if not query:
        return "Usage: VCRECALL <query>"
    if not state:
        return "No active conversation."

    engine = state.engine
    store = engine._store

    # Search across all sources
    from ..core.quote_search import find_quote
    semantic = getattr(engine, "_semantic", None)
    results = find_quote(
        store, semantic, query, max_results=10,
        conversation_id=engine.config.conversation_id,
    )

    if not results.get("found"):
        return f"No matches found for '{query}'."

    # Extract unique tags from results — find_quote uses "topic" key,
    # which may be a single tag or comma-separated list of merged tags.
    matched_tags = set()
    for r in results.get("results", []):
        topic = r.get("topic", "")
        if topic:
            for t in topic.split(", "):
                t = t.strip()
                if t:
                    matched_tags.add(t)

    if not matched_tags:
        return f"Found content for '{query}' but no tags to promote."

    # Promote matching tags to working set at full depth
    promoted = []
    for tag in sorted(matched_tags)[:5]:  # cap at 5 to avoid budget explosion
        try:
            result = engine.expand_topic(tag=tag, depth="full")
            if result and not result.get("error"):
                tokens = result.get("tokens", 0)
                promoted.append(f"  {tag} ({tokens:,} tokens)")
        except Exception:
            pass

    if not promoted:
        return f"Found matches for '{query}' but could not promote any tags."

    lines = [
        f"Recalled {len(promoted)} topic(s) for '{query}':",
        *promoted,
        "",
        "These topics are now in the working set and will be included in context for subsequent turns.",
    ]
    return "\n".join(lines)


def _handle_vccompact(state):
    """Force compaction now."""
    if not state:
        return "No active conversation."

    engine = state.engine
    tti = engine._turn_tag_index
    es = engine._engine_state

    turns = len(tti.entries) if tti else 0
    compacted = getattr(es, "compacted_through", 0)
    uncompacted = turns - compacted

    if uncompacted < 2:
        return f"Nothing to compact ({turns} turns, all compacted through {compacted})."

    # Force compaction via _compact_after_ingestion pattern —
    # submits to background pool, doesn't block response.
    try:
        from ..types import CompactionSignal
        history = state.conversation_history if state.conversation_history else []
        signal = CompactionSignal(
            priority="soft",
            current_tokens=uncompacted * 100,
            budget_tokens=engine.config.monitor.context_window,
            overflow_tokens=uncompacted * 50,
        )
        state._compact_pool.submit(state._run_compact, history, signal, turns)
        return f"Compaction started for {uncompacted} uncompacted turns (turns {compacted + 1}\u2013{turns})."
    except Exception as e:
        return f"Could not trigger compaction: {e}"


def _handle_vclist(tenant_registry, tenant_id):
    """List all conversations with labels."""
    if not tenant_registry or not tenant_id:
        return "Conversation list not available (no tenant registry)."

    labels = tenant_registry.get_conversation_labels(tenant_id)
    conv_ids = tenant_registry.list_persisted_conversation_ids(tenant_id)

    if not conv_ids:
        return "No conversations found."

    lines = ["Conversations:"]
    for cid in conv_ids:
        label = labels.get(cid, "")
        # Try to get turn count from loaded state
        st = tenant_registry.get_state(tenant_id, cid)
        turns = "?"
        if st:
            tti = st.engine._turn_tag_index
            turns = str(len(tti.entries)) if tti else "0"
        label_str = f" ({label})" if label else ""
        lines.append(f"  {cid[:12]}{label_str} — {turns} turns")

    return "\n".join(lines)


def _handle_vc_command_rest(result, state, registry, tenant_id, vcconv):
    """REST endpoint handler for all VC commands. Returns JSONResponse."""
    from starlette.responses import JSONResponse
    cmd = result.vc_command
    arg = result.vc_command_arg
    conv_id = vcconv or result.conversation_id

    if cmd == "attach":
        # VCATTACH has special REST handling — alias, delete, reset
        from .vcattach import resolve_target, execute_attach

        labels = registry.get_conversation_labels(tenant_id)
        conv_ids = registry.list_persisted_conversation_ids(tenant_id)
        target_id, target_label, error = resolve_target(arg, conv_id, conv_ids, labels)

        if error:
            return JSONResponse({"conversation_id": conv_id, "vc_command": "attach", "error": error})

        _store = state.engine._store
        _inner = getattr(_store, '_store', _store)

        def _reset_target(tid):
            _load = getattr(_inner, 'load_engine_state', None)
            _save = getattr(_inner, 'save_engine_state', None)
            if callable(_load) and callable(_save):
                existing = _load(tid)
                if existing:
                    existing.compacted_through = 0
                    existing.last_compacted_turn = -1
                    existing.last_completed_turn = -1
                    existing.last_indexed_turn = -1
                    existing.turn_tag_entries = []
                    _save(existing)

        def _invalidate(tid):
            if registry._session_state_provider:
                try:
                    registry._session_state_provider.delete(tid)
                except Exception:
                    pass

        execute_attach(
            old_id=conv_id,
            target_id=target_id,
            store=_inner,
            registry_invalidate=_invalidate,
            delete_conversation=lambda cid: registry.delete_conversation(tenant_id, cid),
            reset_engine_state=_reset_target,
        )

        marker = f"\n<!-- vc:conversation={target_id} -->"
        message = f"Conversation attached to {target_label} ({target_id}). History restored."
        return JSONResponse({
            "conversation_id": target_id,
            "vc_command": "attach",
            "label": target_label,
            "message": message,
            "body": {"messages": [{"role": "assistant", "content": [{"type": "text", "text": message + marker}]}]},
        })

    # All other commands: dispatch to shared handlers, return JSON
    if cmd == "label":
        text = _handle_vclabel(arg, conv_id, state, registry, tenant_id)
    elif cmd == "status":
        text = _handle_vcstatus(conv_id, state, registry, tenant_id)
    elif cmd == "recall":
        text = _handle_vcrecall(arg, state)
    elif cmd == "compact":
        text = _handle_vccompact(state)
    elif cmd == "list":
        text = _handle_vclist(registry, tenant_id)
    elif cmd == "forget":
        text = _handle_vcforget(arg, state)
    else:
        text = f"Unknown VC command: {cmd}"

    marker = f"\n<!-- vc:conversation={conv_id} -->"
    return JSONResponse({
        "conversation_id": conv_id,
        "vc_command": cmd,
        "message": text,
        "body": {"messages": [{"role": "assistant", "content": [{"type": "text", "text": text + marker}]}]},
    })


def _handle_vcforget(tag: str, state):
    """Delete segments and summaries for a specific tag."""
    if not tag:
        return "Usage: VCFORGET <tag>"
    if not state:
        return "No active conversation."

    engine = state.engine
    store = engine._store
    conv_id = engine.config.conversation_id

    # Check if tag exists — get_all_tags returns TagStats objects
    all_tag_stats = store.get_all_tags(conversation_id=conv_id)
    all_tag_names = [ts.tag for ts in all_tag_stats if hasattr(ts, "tag")]
    if tag not in all_tag_names:
        # Try case-insensitive match
        tag_lower = tag.lower()
        matches = [t for t in all_tag_names if t.lower() == tag_lower]
        if not matches:
            available = ", ".join(sorted(all_tag_names)[:20])
            return f"Tag '{tag}' not found. Available tags: {available}"
        tag = matches[0]

    # Delete segments for this tag
    deleted = 0
    try:
        segments = store.get_segments_by_tags([tag], conversation_id=conv_id)
        for seg in segments:
            ref = getattr(seg, "ref", "") or getattr(seg, "segment_ref", "")
            if ref:
                store.delete_segment(ref)
                deleted += 1
    except Exception:
        pass

    # Delete tag summary
    try:
        if hasattr(store, "delete_tag_summary"):
            store.delete_tag_summary(tag, conversation_id=conv_id)
    except Exception:
        pass

    # Remove from working set if present
    paging = getattr(engine, "_paging", None)
    if paging and hasattr(paging, "working_set"):
        paging.working_set.pop(tag, None)

    # Emit event so dashboard updates
    if state and state.metrics:
        state.metrics.record({
            "type": "tag_forgotten",
            "conversation_id": conv_id,
            "tag": tag,
            "segments_removed": deleted,
        })

    return f"Forgot '{tag}': {deleted} segment(s) removed."
