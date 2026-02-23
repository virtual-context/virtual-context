"""Pure helper functions for the proxy server.

No ProxyState dependency. Extracted from proxy/server.py to reduce
that module's size. Constants, message processing, SSE construction,
and format delegation all live here.
"""

from __future__ import annotations

import json as _json
import re
from typing import TYPE_CHECKING

from ..types import Message
from .formats import (
    detect_format,
    get_format,
)

if TYPE_CHECKING:
    from ..engine import VirtualContextEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VC_PROMPT_MARKER = "[vc:prompt]\n"
# MemOS preamble: starts with "# Role", ends with this delimiter line (zero-width spaces)
_MEMOS_QUERY_DELIM = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b"

# Session marker: injected into assistant responses, extracted from inbound history
_VC_SESSION_RE = re.compile(r"<!-- vc:session=([a-f0-9-]+) -->")

# OpenClaw envelope patterns — consistent across all channels
_VC_USER_RE = re.compile(r"^\[vc:user\](.*?)\[/vc:user\]", re.DOTALL)
_SYSTEM_EVENT_RE = re.compile(r"^(?:System:\s*\[[^\]]*\][^\n]*\n+)+")
_CHANNEL_HEADER_RE = re.compile(r"^\[[A-Z][a-zA-Z]*\s[^\]]*\bid:-?\d+\b[^\]]*\]\s*")
_MESSAGE_ID_RE = re.compile(r"\n?\[message_id:\s*\d+\]\s*$")

_HOP_BY_HOP = frozenset({
    "host", "connection", "transfer-encoding", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "upgrade", "content-length",
})


# ---------------------------------------------------------------------------
# Message processing helpers
# ---------------------------------------------------------------------------

def _last_text_block(content: list) -> str:
    """Return the text of the last ``type: "text"`` block in *content*."""
    for block in reversed(content):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def _strip_vc_prompt(text: str) -> str:
    """Strip the ``[vc:prompt]`` marker injected by the OpenClaw plugin."""
    if text.startswith(_VC_PROMPT_MARKER):
        return text[len(_VC_PROMPT_MARKER):]
    return text


def _strip_openclaw_envelope(text: str) -> str:
    """Strip OpenClaw channel metadata from a message.

    Handles (in order):

    1. ``[vc:prompt]`` marker from the virtual-context-tagger plugin
    2. ``[vc:user]...[/vc:user]`` backward-compatible wrapper (extracts
       inner content and returns immediately — inner content is already clean)
    3. ``System: [TIMESTAMP] event`` lines prepended by OpenClaw
    4. ``[ChannelName ... id:NNN ...] `` header (Telegram, WhatsApp, etc.)
    5. ``[message_id: NNN]`` footer

    Returns the actual conversational content with all metadata removed.
    """
    if not text:
        return text

    # 1. Strip [vc:prompt] marker and any trailing whitespace
    if text.startswith(_VC_PROMPT_MARKER):
        text = text[len(_VC_PROMPT_MARKER):].lstrip()

    # 1b. Strip MemOS preamble: "# Role ... user原始query：" → keep only content after delimiter
    if text.startswith("# Role"):
        idx = text.find(_MEMOS_QUERY_DELIM)
        if idx != -1:
            text = text[idx + len(_MEMOS_QUERY_DELIM):].lstrip()

    # 2. Handle [vc:user]...[/vc:user] — inner content is already clean
    m = _VC_USER_RE.match(text)
    if m:
        return m.group(1).strip()

    # 3. Strip System: [...] event lines
    text = _SYSTEM_EVENT_RE.sub("", text)

    # 4. Strip channel header  [ChannelName ... id:NNN ...]
    text = _CHANNEL_HEADER_RE.sub("", text)

    # 5. Strip [message_id: NNN] footer
    text = _MESSAGE_ID_RE.sub("", text)

    return text.strip()


def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter out hop-by-hop headers for forwarding."""
    return {
        k: v for k, v in headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


# ---------------------------------------------------------------------------
# Format delegation helpers
# ---------------------------------------------------------------------------

def _detect_api_format(body: dict) -> str:
    """Detect whether this is an Anthropic, OpenAI, or Gemini request."""
    return detect_format(body).name


def _extract_session_id(body: dict) -> str | None:
    """Scan assistant messages for vc:session marker. Returns UUID or None."""
    fmt = detect_format(body)
    return fmt.extract_session_id(body)


def _strip_session_markers(body: dict) -> dict:
    """Strip vc:session markers from all assistant messages in the request body."""
    fmt = detect_format(body)
    return fmt.strip_session_markers(body)


def _extract_user_message(body: dict) -> str:
    """Extract the last user message text from a request body."""
    fmt = detect_format(body)
    return fmt.extract_user_message(body)


def _extract_message_text(msg: dict, api_format: str = "anthropic") -> str:
    """Extract text from a single message dict (string or content blocks)."""
    return get_format(api_format).extract_message_text(msg)


def _extract_history_pairs(body: dict) -> list[Message]:
    """Extract complete user+assistant pairs from request history."""
    fmt = detect_format(body)
    return fmt.extract_history_pairs(body)


def _inject_context(body: dict, prepend_text: str, api_format: str) -> dict:
    """Inject <virtual-context> block into a deep-copied request body."""
    return get_format(api_format).inject_context(body, prepend_text)


def _inject_session_marker(response_body: dict, marker: str, api_format: str) -> dict:
    """Append session marker text to the last text content block."""
    return get_format(api_format).inject_session_marker(response_body, marker)


def _extract_delta_text(data: dict, api_format: str) -> str:
    """Extract text delta from a streaming SSE event payload."""
    return get_format(api_format).extract_delta_text(data)


def _extract_assistant_text(response_body: dict, api_format: str) -> str:
    """Extract assistant text from a non-streaming response."""
    return get_format(api_format).extract_assistant_text(response_body)


def _inject_vc_tools(body: dict, engine: "VirtualContextEngine") -> dict:
    """Append VC paging tool definitions to the request body's tools array."""
    from ..core.tool_loop import vc_tool_definitions
    fmt = detect_format(body)
    return fmt.inject_tools(body, vc_tool_definitions())


def _build_continuation_request(
    original_body: dict,
    assistant_content: list[dict],
    tool_results: list[dict],
) -> dict:
    """Build a non-streaming continuation request after VC tool execution."""
    fmt = detect_format(original_body)
    return fmt.build_continuation_request(original_body, assistant_content, tool_results)


# ---------------------------------------------------------------------------
# SSE construction helpers
# ---------------------------------------------------------------------------

def _parse_sse_events(
    buf: bytes,
) -> tuple[list[tuple[str, str, bytes]], bytes]:
    """Split a byte buffer into complete SSE events.

    Returns ``(events, remainder)`` where each event is
    ``(event_type, data_str, raw_bytes)``.
    """
    events: list[tuple[str, str, bytes]] = []
    while True:
        idx_rn = buf.find(b"\r\n\r\n")
        idx_n = buf.find(b"\n\n")
        if idx_rn == -1 and idx_n == -1:
            break
        # Use whichever boundary comes first
        if idx_rn != -1 and (idx_n == -1 or idx_rn <= idx_n):
            end = idx_rn + 4
        else:
            end = idx_n + 2

        raw_event = buf[:end]
        buf = buf[end:]

        decoded = raw_event.decode("utf-8", errors="replace")
        event_type = ""
        data_str = ""
        for line in decoded.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:]

        events.append((event_type, data_str, raw_event))

    return events, buf


def _emit_text_as_sse(text: str, block_index: int) -> list[bytes]:
    """Convert *text* into Anthropic SSE events at *block_index*."""
    events: list[bytes] = []
    start = _json.dumps({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {"type": "text", "text": ""},
    })
    events.append(f"event: content_block_start\ndata: {start}\n\n".encode())

    delta = _json.dumps({
        "type": "content_block_delta",
        "index": block_index,
        "delta": {"type": "text_delta", "text": text},
    })
    events.append(f"event: content_block_delta\ndata: {delta}\n\n".encode())

    stop = _json.dumps({
        "type": "content_block_stop",
        "index": block_index,
    })
    events.append(f"event: content_block_stop\ndata: {stop}\n\n".encode())
    return events


def _emit_tool_use_as_sse(
    tool: dict, block_index: int,
) -> list[bytes]:
    """Convert a tool_use content block into Anthropic SSE events."""
    events: list[bytes] = []
    start = _json.dumps({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {
            "type": "tool_use",
            "id": tool.get("id", ""),
            "name": tool.get("name", ""),
            "input": {},
        },
    })
    events.append(f"event: content_block_start\ndata: {start}\n\n".encode())

    input_str = _json.dumps(tool.get("input", {}))
    delta = _json.dumps({
        "type": "content_block_delta",
        "index": block_index,
        "delta": {"type": "input_json_delta", "partial_json": input_str},
    })
    events.append(f"event: content_block_delta\ndata: {delta}\n\n".encode())

    stop = _json.dumps({
        "type": "content_block_stop",
        "index": block_index,
    })
    events.append(f"event: content_block_stop\ndata: {stop}\n\n".encode())
    return events


def _emit_text_as_responses_sse(text: str, item_index: int = 0) -> list[bytes]:
    """Convert *text* into OpenAI Responses API SSE events."""
    events: list[bytes] = []

    item_id = f"item_{item_index}"
    output_index = item_index

    # response.output_item.added — message item
    item_added = _json.dumps({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [],
        },
    })
    events.append(
        f"event: response.output_item.added\ndata: {item_added}\n\n".encode(),
    )

    # response.content_part.added — output_text part
    part_added = _json.dumps({
        "type": "response.content_part.added",
        "output_index": output_index,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    })
    events.append(
        f"event: response.content_part.added\ndata: {part_added}\n\n".encode(),
    )

    # response.output_text.delta — the text
    text_delta = _json.dumps({
        "type": "response.output_text.delta",
        "output_index": output_index,
        "content_index": 0,
        "delta": text,
    })
    events.append(
        f"event: response.output_text.delta\ndata: {text_delta}\n\n".encode(),
    )

    # response.output_text.done
    text_done = _json.dumps({
        "type": "response.output_text.done",
        "output_index": output_index,
        "content_index": 0,
        "text": text,
    })
    events.append(
        f"event: response.output_text.done\ndata: {text_done}\n\n".encode(),
    )

    # response.content_part.done
    part_done = _json.dumps({
        "type": "response.content_part.done",
        "output_index": output_index,
        "content_index": 0,
        "part": {"type": "output_text", "text": text},
    })
    events.append(
        f"event: response.content_part.done\ndata: {part_done}\n\n".encode(),
    )

    # response.output_item.done
    item_done = _json.dumps({
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    })
    events.append(
        f"event: response.output_item.done\ndata: {item_done}\n\n".encode(),
    )
    return events


def _emit_tool_use_as_responses_sse(
    tool: dict, item_index: int = 0,
) -> list[bytes]:
    """Convert a function_call into OpenAI Responses API SSE events."""
    events: list[bytes] = []
    output_index = item_index
    call_id = tool.get("id", tool.get("call_id", ""))
    name = tool.get("name", "")
    args = tool.get("input", {})
    args_str = _json.dumps(args) if isinstance(args, dict) else str(args)

    # response.output_item.added — function_call item
    item_added = _json.dumps({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": "",
        },
    })
    events.append(
        f"event: response.output_item.added\ndata: {item_added}\n\n".encode(),
    )

    # response.function_call_arguments.delta
    args_delta = _json.dumps({
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "delta": args_str,
    })
    events.append(
        f"event: response.function_call_arguments.delta\n"
        f"data: {args_delta}\n\n".encode(),
    )

    # response.function_call_arguments.done
    args_done = _json.dumps({
        "type": "response.function_call_arguments.done",
        "output_index": output_index,
        "name": name,
        "arguments": args_str,
    })
    events.append(
        f"event: response.function_call_arguments.done\n"
        f"data: {args_done}\n\n".encode(),
    )

    # response.output_item.done
    item_done = _json.dumps({
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": args_str,
        },
    })
    events.append(
        f"event: response.output_item.done\ndata: {item_done}\n\n".encode(),
    )
    return events


def _emit_response_done_sse(
    output_items: list[dict],
    usage: dict | None = None,
) -> list[bytes]:
    """Emit ``response.completed`` SSE event for Responses API."""
    events: list[bytes] = []
    completed = _json.dumps({
        "type": "response.completed",
        "response": {
            "output": output_items,
            "status": "completed",
            "usage": usage or {},
        },
    })
    events.append(
        f"event: response.completed\ndata: {completed}\n\n".encode(),
    )
    return events


def _emit_message_end_sse(
    stop_reason: str = "end_turn",
    usage: dict | None = None,
) -> list[bytes]:
    """Emit ``message_delta`` and ``message_stop`` SSE events."""
    events: list[bytes] = []
    md_payload: dict = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
    }
    if usage:
        md_payload["usage"] = usage
    md = _json.dumps(md_payload)
    events.append(f"event: message_delta\ndata: {md}\n\n".encode())

    ms = _json.dumps({"type": "message_stop"})
    events.append(f"event: message_stop\ndata: {ms}\n\n".encode())
    return events


# ---------------------------------------------------------------------------
# Debug / session state dump
# ---------------------------------------------------------------------------

def _dump_session_state(
    state: object,
    session_log_path: object,
) -> None:
    """Write full proxy memory dump to disk alongside request/response logs."""
    try:
        engine = state.engine  # type: ignore[attr-defined]
        idx = engine._turn_tag_index

        # TurnTagIndex entries
        entries = []
        for e in idx.entries:
            entries.append({
                "turn": e.turn_number,
                "tags": e.tags,
                "primary_tag": e.primary_tag,
                "message_hash": e.message_hash,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            })

        # Tag counts (how many turns per tag)
        tag_counts: dict[str, int] = {}
        for e in idx.entries:
            for t in e.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        # Tag summaries from store
        tag_summaries = []
        try:
            for ts in engine._store.get_all_tag_summaries():
                tag_summaries.append({
                    "tag": ts.tag,
                    "summary": ts.summary,
                    "summary_tokens": ts.summary_tokens,
                    "source_turn_numbers": ts.source_turn_numbers,
                    "covers_through_turn": ts.covers_through_turn,
                })
        except Exception:
            pass

        # Tag aliases
        aliases: dict[str, str] = {}
        try:
            aliases = engine._store.get_tag_aliases()
        except Exception:
            pass

        # Tag stats from store
        tag_stats = []
        try:
            for st in engine._store.get_all_tags():
                tag_stats.append({
                    "tag": st.tag,
                    "usage_count": st.usage_count,
                    "total_full_tokens": st.total_full_tokens,
                    "total_summary_tokens": st.total_summary_tokens,
                })
        except Exception:
            pass

        # Split processed tags
        split_tags = list(getattr(engine, "_split_processed_tags", set()))

        # Working set (paging state)
        working_set_dump: list[dict] = []
        ws = getattr(engine, "_working_set", None)
        if ws:
            for tag, entry in ws.items():
                working_set_dump.append({
                    "tag": tag,
                    "depth": entry.depth.value if hasattr(entry.depth, "value") else str(entry.depth),
                    "tokens": entry.tokens,
                    "last_accessed_turn": entry.last_accessed_turn,
                })

        dump = {
            "session_id": engine.config.session_id,
            "session_state": state._state.value if hasattr(state._state, "value") else str(state._state),  # type: ignore[attr-defined]
            "turn_count": len(state.conversation_history) // 2,  # type: ignore[attr-defined]
            "compacted_through": getattr(engine, "_compacted_through", 0),
            "turn_tag_index": entries,
            "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
            "tag_summaries": tag_summaries,
            "tag_aliases": aliases,
            "tag_stats": tag_stats,
            "split_processed_tags": split_tags,
            "working_set": working_set_dump,
            "conversation_history": [
                {"role": m.role, "content": m.content[:500]}
                for m in state.conversation_history  # type: ignore[attr-defined]
            ],
        }

        session_log_path.write_text(  # type: ignore[attr-defined]
            _json.dumps(dump, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass  # never let session dump break the request
