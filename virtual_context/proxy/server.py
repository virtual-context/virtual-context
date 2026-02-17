"""HTTP proxy server for virtual-context LLM enrichment.

Sits between any LLM client and an upstream provider (OpenAI or Anthropic),
transparently injecting <virtual-context> blocks into requests and capturing
assistant responses for on_turn_complete.

Usage:
    virtual-context -c config.yaml proxy --upstream https://api.anthropic.com
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import sys
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..engine import VirtualContextEngine
from ..core.turn_tag_index import TurnTagIndex
from ..types import Message

from .dashboard import register_dashboard_routes
from .metrics import ProxyMetrics

logger = logging.getLogger(__name__)

_VC_PROMPT_MARKER = "[vc:prompt]\n"
# MemOS preamble: starts with "# Role", ends with this delimiter line (zero-width spaces)
_MEMOS_QUERY_DELIM = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b"

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
# ProxyState — mirrors HeadlessRunner threading pattern
# ---------------------------------------------------------------------------

class ProxyState:
    """Shared mutable state for the proxy lifetime."""

    def __init__(
        self,
        engine: VirtualContextEngine,
        metrics: ProxyMetrics | None = None,
        upstream: str = "",
    ) -> None:
        self.engine = engine
        self.conversation_history: list[Message] = []
        self.metrics = metrics
        self.upstream = upstream
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending_complete: Future | None = None
        self._ingested_sessions: set[str] = set()
        self._ingestion_lock = threading.Lock()
        self._compaction_lock = threading.Lock()

    def wait_for_complete(self) -> None:
        """Block until the pending on_turn_complete finishes."""
        if self._pending_complete is not None:
            self._pending_complete.result()
            self._pending_complete = None

    def fire_turn_complete(self, history_snapshot: list[Message]) -> None:
        """Submit on_turn_complete to background thread."""
        self._pending_complete = self._pool.submit(
            self._run_turn_complete, history_snapshot
        )

    def _run_turn_complete(self, history: list[Message]) -> None:
        t0 = time.monotonic()
        turn = len(history) // 2 - 1
        session_id = self.engine.config.session_id
        try:
            report = self.engine.on_turn_complete(history)

            complete_ms = round((time.monotonic() - t0) * 1000, 1)
            entry = self.engine._turn_tag_index.get_tags_for_turn(turn)
            _tags = entry.tags if entry else []
            _primary = entry.primary_tag if entry else ""
            print(
                f"[T{turn}] COMPLETE {int(complete_ms)}ms "
                f"tags=[{', '.join(_tags)}] primary={_primary}"
                + (f" COMPACTION freed={report.tokens_freed}t" if report else "")
            )
            logger.info(
                "T%d complete (%dms) session=%s compacted_through=%d history=%d%s",
                turn, int(complete_ms), session_id[:12],
                getattr(self.engine, "_compacted_through", 0),
                len(history),
                " COMPACTION" if report else "",
            )

            if report is not None:
                logger.info(
                    "  compaction: %d segments, freed %d tokens, tags=%s, "
                    "summaries_built=%d",
                    report.segments_compacted,
                    report.tokens_freed,
                    report.tags,
                    report.tag_summaries_built,
                )

            # Emit turn_complete event
            if self.metrics:
                entry = self.engine._turn_tag_index.get_tags_for_turn(turn)
                active_tags = list(
                    self.engine._turn_tag_index.get_active_tags(lookback=6)
                )
                turn_pair_tokens = (
                    sum(len(m.content) for m in history[-2:]) // 4
                    if len(history) >= 2 else 0
                )
                # Write response tags to captured request
                response_tags = entry.tags if entry else []
                self.metrics.update_request_tags(
                    turn, response_tags=response_tags,
                )
                self.metrics.record({
                    "type": "turn_complete",
                    "turn": turn,
                    "tags": entry.tags if entry else [],
                    "primary_tag": entry.primary_tag if entry else "",
                    "complete_ms": complete_ms,
                    "active_tags": active_tags,
                    "store_tag_count": len(self.engine._store.get_all_tags()),
                    "turn_pair_tokens": turn_pair_tokens,
                })

                # Emit compaction event if compaction occurred
                if report is not None:
                    original_tokens = sum(
                        r.original_tokens for r in report.results
                    )
                    summary_tokens = sum(
                        r.summary_tokens for r in report.results
                    )
                    self.metrics.record({
                        "type": "compaction",
                        "turn": turn,
                        "segments": report.segments_compacted,
                        "tokens_freed": report.tokens_freed,
                        "original_tokens": original_tokens,
                        "summary_tokens": summary_tokens,
                        "tags": report.tags,
                        "tag_summaries_built": report.tag_summaries_built,
                        "compacted_through": getattr(
                            self.engine, "_compacted_through", 0
                        ),
                    })
        except Exception as e:
            logger.error("on_turn_complete error: %s", e, exc_info=True)

    @property
    def _history_ingested(self) -> bool:
        """Whether the current session's history has been ingested."""
        return self.engine.config.session_id in self._ingested_sessions

    def ingest_if_needed(self, history_pairs: list[Message]) -> None:
        """Bootstrap TurnTagIndex from pre-existing history (once per session).

        Double-checked locking: fast path skips the lock entirely.
        """
        session_id = self.engine.config.session_id
        if session_id in self._ingested_sessions:
            return
        with self._ingestion_lock:
            if session_id in self._ingested_sessions:
                return
            t0 = time.monotonic()
            turns = self.engine.ingest_history(history_pairs)
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            self._ingested_sessions.add(session_id)

            print(
                f"[INGEST] {turns} turns in {int(elapsed_ms)}ms "
                f"(session={session_id[:12]})"
            )
            logger.info(
                "History ingestion: %d turns in %dms (session=%s)",
                turns, int(elapsed_ms), session_id[:12],
            )

            if self.metrics:
                # Emit per-turn events so the dashboard grid shows history
                baseline_history_tokens = 0
                for i in range(0, len(history_pairs) - 1, 2):
                    turn_num = i // 2
                    entry = self.engine._turn_tag_index.get_tags_for_turn(
                        turn_num,
                    )
                    raw_content = history_pairs[i].content
                    preview = _strip_openclaw_envelope(raw_content)[:60]
                    # Estimate turn pair tokens for baseline calculation
                    pair_chars = len(history_pairs[i].content) + len(history_pairs[i + 1].content)
                    tpt = pair_chars // 4
                    baseline_history_tokens += tpt
                    self.metrics.record({
                        "type": "ingested_turn",
                        "turn": turn_num,
                        "tags": entry.tags if entry else [],
                        "primary_tag": entry.primary_tag if entry else "",
                        "message_preview": preview,
                        "turn_pair_tokens": tpt,
                    })
                self.metrics.record({
                    "type": "history_ingestion",
                    "turns_ingested": turns,
                    "pairs_received": len(history_pairs) // 2,
                    "elapsed_ms": elapsed_ms,
                    "session_id": session_id,
                    "baseline_history_tokens": baseline_history_tokens,
                })

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no side effects)
# ---------------------------------------------------------------------------

def _detect_api_format(body: dict) -> str:
    """Detect whether this is an Anthropic or OpenAI request.

    Anthropic requests have a top-level "system" field and/or a model name
    starting with "claude". OpenAI is the default.
    """
    if "system" in body:
        return "anthropic"
    model = body.get("model", "")
    if isinstance(model, str) and model.startswith("claude"):
        return "anthropic"
    return "openai"


def _last_text_block(content: list) -> str:
    """Return the text of the last ``type: "text"`` block in *content*.

    LLM clients often place extended-thinking or system-level content in
    earlier text blocks while the actual conversational content occupies the
    final text block.  Extracting only the last block filters out that noise
    without relying on client-specific markers or heuristics.
    """
    for block in reversed(content):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def _strip_vc_prompt(text: str) -> str:
    """Strip the ``[vc:prompt]`` marker injected by the OpenClaw plugin.

    Returns the text with the marker removed.  If no marker is present,
    returns the original text unchanged.
    """
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


def _extract_user_message(body: dict) -> str:
    """Extract the last user message text from a request body.

    Strips OpenClaw envelope metadata (channel headers, message footers,
    system events, plugin markers) and applies last-text-block extraction
    for content-block arrays.
    """
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return _strip_openclaw_envelope(content)
        if isinstance(content, list):
            return _strip_openclaw_envelope(_last_text_block(content))
    return ""


def _extract_message_text(msg: dict) -> str:
    """Extract text from a single message dict (string or content blocks).

    Strips OpenClaw envelope metadata, then uses last-text-block for arrays.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return _strip_openclaw_envelope(content)
    if isinstance(content, list):
        return _strip_openclaw_envelope(_last_text_block(content))
    return ""


def _extract_history_pairs(body: dict) -> list[Message]:
    """Extract complete user+assistant pairs from request history.

    Filters out system messages, drops the last user message (current turn),
    and drops trailing unpaired messages. Returns a flat list:
    [user_0, asst_0, user_1, asst_1, ...]
    """
    messages = body.get("messages", [])

    # Filter to user/assistant only
    chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]

    if not chat_msgs:
        return []

    # Drop the last user message (that's the current turn being sent to the LLM)
    if chat_msgs and chat_msgs[-1].get("role") == "user":
        chat_msgs = chat_msgs[:-1]

    if not chat_msgs:
        return []

    # Walk from start, collecting complete user+assistant pairs.
    # Skips misaligned messages (consecutive users, trailing unpaired, etc.).
    pairs: list[Message] = []
    i = 0
    while i + 1 < len(chat_msgs):
        if (chat_msgs[i].get("role") == "user"
                and chat_msgs[i + 1].get("role") == "assistant"):
            pairs.append(Message(
                role="user",
                content=_extract_message_text(chat_msgs[i]),
            ))
            pairs.append(Message(
                role="assistant",
                content=_extract_message_text(chat_msgs[i + 1]),
            ))
            i += 2
        else:
            # Skip misaligned messages
            i += 1
    return pairs


def _inject_context(body: dict, prepend_text: str, api_format: str) -> dict:
    """Inject <virtual-context> block into a shallow-copied request body.

    Does not mutate the original body.
    """
    if not prepend_text:
        return body

    body = copy.deepcopy(body)
    context_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"

    if api_format == "anthropic":
        existing = body.get("system", "")
        # Anthropic system can be a string or list of content blocks
        if isinstance(existing, list):
            # Prepend as a text block
            body["system"] = [{"type": "text", "text": context_block}] + existing
        else:
            body["system"] = f"{context_block}\n\n{existing}" if existing else context_block
    else:
        # OpenAI: system message in messages array
        messages = body.get("messages", [])
        if messages and messages[0].get("role") == "system":
            existing = messages[0].get("content", "")
            messages[0] = dict(messages[0])
            messages[0]["content"] = (
                f"{context_block}\n\n{existing}" if existing else context_block
            )
        else:
            messages.insert(0, {"role": "system", "content": context_block})
        body["messages"] = messages

    return body


def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter out hop-by-hop headers for forwarding."""
    return {
        k: v for k, v in headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


def _filter_body_messages(
    body: dict,
    turn_tag_index: TurnTagIndex,
    matched_tags: list[str],
    *,
    recent_turns: int = 3,
    broad: bool = False,
    temporal: bool = False,
) -> tuple[dict, int]:
    """Filter request body messages to remove irrelevant history turns.

    Operates on the raw API body, preserving original message format
    (content blocks, metadata, etc.).  Uses the TurnTagIndex to decide
    which user+assistant pairs to keep based on tag overlap.

    Returns (filtered_body, turns_dropped).
    """
    messages = body.get("messages", [])
    if not messages:
        return body, 0

    # Separate system messages (OpenAI format) and chat messages
    prefix: list[dict] = []
    chat_msgs: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and not chat_msgs:
            prefix.append(msg)
        else:
            chat_msgs.append(msg)

    if not chat_msgs:
        return body, 0

    # Split trailing user message (current turn) from history pairs
    current_user = None
    if chat_msgs and chat_msgs[-1].get("role") == "user":
        current_user = chat_msgs[-1]
        chat_msgs = chat_msgs[:-1]

    # Group into user+assistant pairs, tracking which message indices are paired.
    # Unpaired messages (tool_results between consecutive users, batched messages,
    # etc.) are always kept — they're structural and may be required by the API.
    pairs: list[tuple[int, int]] = []  # (msg_idx_user, msg_idx_assistant)
    paired_indices: set[int] = set()
    i = 0
    while i + 1 < len(chat_msgs):
        if (chat_msgs[i].get("role") == "user"
                and chat_msgs[i + 1].get("role") == "assistant"):
            pairs.append((i, i + 1))
            paired_indices.add(i)
            paired_indices.add(i + 1)
            i += 2
        else:
            i += 1

    total_pairs = len(pairs)
    protected = min(recent_turns, total_pairs)

    if total_pairs <= protected or not turn_tag_index.entries:
        return body, 0

    # Broad/temporal: keep everything
    if broad or temporal:
        return body, 0

    tag_set = set(matched_tags)

    # First pass: mark each pair as keep/drop based on tags
    keep_pair = [False] * total_pairs
    for pair_idx, (u_idx, a_idx) in enumerate(pairs):
        if pair_idx >= total_pairs - protected:
            keep_pair[pair_idx] = True
        else:
            entry = turn_tag_index.get_tags_for_turn(pair_idx)
            if entry is None:
                keep_pair[pair_idx] = True
            elif "rule" in entry.tags or set(entry.tags) & tag_set:
                keep_pair[pair_idx] = True

    # Second pass: fix tool_use/tool_result dependencies.
    # If assistant has tool_use, the next pair (with tool_result) must also be kept.
    # If user has tool_result, the previous pair (with tool_use) must also be kept.
    # Iterate until stable (handles multi-step tool chains).
    changed = True
    while changed:
        changed = False
        for pair_idx in range(total_pairs):
            if not keep_pair[pair_idx]:
                continue
            u_idx, a_idx = pairs[pair_idx]
            if _has_tool_use(chat_msgs[a_idx]) and pair_idx + 1 < total_pairs and not keep_pair[pair_idx + 1]:
                keep_pair[pair_idx + 1] = True
                changed = True
            if _has_tool_result(chat_msgs[u_idx]) and pair_idx > 0 and not keep_pair[pair_idx - 1]:
                keep_pair[pair_idx - 1] = True
                changed = True

    # Build per-message keep set: unpaired messages always kept, pairs based on filter
    keep_msg: set[int] = set()
    for msg_idx in range(len(chat_msgs)):
        if msg_idx not in paired_indices:
            keep_msg.add(msg_idx)  # always keep unpaired messages
    for pair_idx, (u_idx, a_idx) in enumerate(pairs):
        if keep_pair[pair_idx]:
            keep_msg.add(u_idx)
            keep_msg.add(a_idx)

    # Final tool chain safety: any kept assistant with tool_use must have its
    # tool_result in the immediately following message(s) also kept, and vice versa.
    changed = True
    while changed:
        changed = False
        for msg_idx in range(len(chat_msgs)):
            if msg_idx not in keep_msg:
                continue
            msg = chat_msgs[msg_idx]
            if msg.get("role") == "assistant" and _has_tool_use(msg):
                # Keep all following messages until we find the tool_result
                for j in range(msg_idx + 1, len(chat_msgs)):
                    if j not in keep_msg:
                        keep_msg.add(j)
                        changed = True
                    if _has_tool_result(chat_msgs[j]):
                        break
            if _has_tool_result(msg):
                # Keep all preceding messages back to the tool_use
                for j in range(msg_idx - 1, -1, -1):
                    if j not in keep_msg:
                        keep_msg.add(j)
                        changed = True
                    if chat_msgs[j].get("role") == "assistant" and _has_tool_use(chat_msgs[j]):
                        break

    # Build filtered message list preserving original order
    kept: list[dict] = list(prefix)
    dropped = 0
    for msg_idx in range(len(chat_msgs)):
        if msg_idx in keep_msg:
            kept.append(chat_msgs[msg_idx])
        elif msg_idx in paired_indices:
            dropped += 1  # only count paired message drops (half a pair = 0.5 turn)

    if current_user:
        kept.append(current_user)

    dropped = dropped // 2  # convert message drops to pair drops
    if dropped == 0:
        return body, 0

    body = dict(body)
    body["messages"] = kept
    return body, dropped


def _has_tool_use(msg: dict) -> bool:
    """Check if an assistant message contains tool_use blocks."""
    content = msg.get("content", [])
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
    return False


def _has_tool_result(msg: dict) -> bool:
    """Check if a user message contains tool_result blocks."""
    content = msg.get("content", [])
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _extract_delta_text(data: dict, api_format: str) -> str:
    """Extract text delta from a streaming SSE event payload."""
    if api_format == "openai":
        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "") or ""
    else:
        # Anthropic: content_block_delta event
        event_type = data.get("type", "")
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            return delta.get("text", "") or ""
    return ""


def _extract_assistant_text(response_body: dict, api_format: str) -> str:
    """Extract assistant text from a non-streaming response.

    Uses last-text-block extraction for Anthropic format to skip
    thinking/reasoning blocks that precede the actual response.
    """
    if api_format == "openai":
        choices = response_body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "") or ""
    else:
        # Anthropic: last text block (skips thinking blocks)
        content = response_body.get("content", [])
        return _last_text_block(content)
    return ""


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------

def create_app(upstream: str, config_path: str | None = None) -> FastAPI:
    """Create the FastAPI proxy application.

    Args:
        upstream: Upstream provider base URL (e.g. https://api.anthropic.com).
        config_path: Path to virtual-context config file.
    """
    upstream = upstream.rstrip("/")

    # Initialize engine
    try:
        engine = VirtualContextEngine(config_path=config_path)
        metrics = ProxyMetrics(
            context_window=engine.config.monitor.context_window,
        )
        state = ProxyState(engine, metrics=metrics, upstream=upstream)
        logger.info(
            "Engine ready — session_id=%s, window=%d, storage=%s",
            engine.config.session_id,
            engine.config.monitor.context_window,
            engine.config.storage.backend,
        )
    except Exception as e:
        print(f"Engine init failed: {e}", file=sys.stderr)
        metrics = ProxyMetrics()
        state = None

    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    shutdown_event = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        yield
        shutdown_event.set()
        await client.aclose()
        if state:
            state.shutdown()

    app = FastAPI(title="virtual-context proxy", lifespan=lifespan)

    # Register dashboard routes BEFORE the catch-all so /dashboard is not swallowed
    register_dashboard_routes(app, metrics, state, shutdown_event)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def catch_all(request: Request, path: str):
        url = f"{upstream}/{path}"
        raw_headers = dict(request.headers)
        fwd_headers = _forward_headers(raw_headers)

        # Non-POST or no body → passthrough
        if request.method != "POST":
            return await _passthrough(client, request, url, fwd_headers)

        body_bytes = await request.body()
        if not body_bytes:
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        import json as _json
        try:
            body = _json.loads(body_bytes)
        except _json.JSONDecodeError:
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        # Only intercept if it has a messages array (chat completion)
        if not isinstance(body.get("messages"), list):
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        api_format = _detect_api_format(body)
        user_message = _extract_user_message(body)
        is_streaming = body.get("stream", False)

        import datetime as _dt
        _now = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _msg_count = len(body.get("messages", []))
        print(f"[{_now}] POST /{path} msgs={_msg_count} stream={is_streaming}")

        if not user_message:
            # Tool-result or non-text turn — skip VC enrichment but
            # preserve streaming so the client SDK doesn't break.
            if is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=len(state.conversation_history) // 2 if state else 0,
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=len(state.conversation_history) // 2 if state else 0,
                )

        # Enrich with virtual-context
        prepend_text = ""
        assembled = None
        wait_ms = 0.0
        inbound_ms = 0.0
        if state:
            try:
                t0 = time.monotonic()
                await asyncio.to_thread(state.wait_for_complete)
                wait_ms = round((time.monotonic() - t0) * 1000, 1)

                # Bootstrap from client's pre-existing history on first request
                if not state._history_ingested:
                    history_pairs = _extract_history_pairs(body)
                    if history_pairs:
                        state.conversation_history = list(history_pairs)
                        await asyncio.to_thread(
                            state.ingest_if_needed, history_pairs
                        )

                state.conversation_history.append(
                    Message(role="user", content=user_message)
                )

                t1 = time.monotonic()
                assembled = await asyncio.to_thread(
                    state.engine.on_message_inbound,
                    user_message,
                    state.conversation_history,
                )
                inbound_ms = round((time.monotonic() - t1) * 1000, 1)

                prepend_text = assembled.prepend_text
            except Exception as e:
                logger.error("Engine error (forwarding unmodified): %s", e)

        # Filter irrelevant history turns from the request body
        _pre_filter_body = body  # preserve for request capture
        turns_dropped = 0
        _real_tags = [t for t in (assembled.matched_tags if assembled else []) if t != "_general"]
        if _real_tags and state:
            recent = state.engine.config.assembler.recent_turns_always_included
            body, turns_dropped = _filter_body_messages(
                body,
                state.engine._turn_tag_index,
                _real_tags,
                recent_turns=recent,
                broad=assembled.broad,
                temporal=assembled.temporal,
            )

        enriched_body = _inject_context(body, prepend_text, api_format)

        is_streaming = body.get("stream", False)

        # Estimate system prompt tokens from original body (before VC enrichment)
        _sys_len = 0
        if api_format == "anthropic":
            _sys_orig = body.get("system", "")
            if isinstance(_sys_orig, str):
                _sys_len = len(_sys_orig)
            elif isinstance(_sys_orig, list):
                _sys_len = sum(
                    len(b.get("text", "")) for b in _sys_orig
                    if isinstance(b, dict)
                )
        else:
            # OpenAI: system message is messages[0] with role=system
            msgs = body.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                sc = msgs[0].get("content", "")
                _sys_len = len(sc) if isinstance(sc, str) else sum(
                    len(b.get("text", "")) for b in sc
                    if isinstance(b, dict)
                )
        system_tokens = _sys_len // 4

        # Estimate input tokens from enriched body
        _input_text_len = 0
        for msg in enriched_body.get("messages", []):
            c = msg.get("content", "")
            _input_text_len += len(c) if isinstance(c, str) else sum(
                len(b.get("text", "")) for b in c if isinstance(b, dict)
            )
        sys_c = enriched_body.get("system", "")
        if isinstance(sys_c, str):
            _input_text_len += len(sys_c)
        elif isinstance(sys_c, list):
            _input_text_len += sum(
                len(b.get("text", "")) for b in sys_c if isinstance(b, dict)
            )
        input_tokens = _input_text_len // 4

        # Record request event
        turn = len(state.conversation_history) // 2 if state else 0
        context_tokens = len(prepend_text) // 4 if prepend_text else 0
        total_turns = len(state.conversation_history) // 2 if state else 0
        overhead_ms = round(wait_ms + inbound_ms, 1)
        metrics.record({
            "type": "request",
            "turn": turn,
            "message_preview": user_message[:60],
            "api_format": api_format,
            "streaming": is_streaming,
            "tags": assembled.matched_tags if assembled else [],
            "broad": assembled.broad if assembled else False,
            "temporal": assembled.temporal if assembled else False,
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
            "input_tokens": input_tokens,
            "system_tokens": system_tokens,
            "turns_dropped": turns_dropped,
        })

        # Log request to terminal for debugging
        _tags_str = ", ".join(assembled.matched_tags) if assembled else "none"
        _flags = []
        if assembled and assembled.broad:
            _flags.append("BROAD")
        if assembled and assembled.temporal:
            _flags.append("TEMPORAL")
        _flag_str = f" [{' '.join(_flags)}]" if _flags else ""
        print(
            f"[T{turn}] POST {api_format} stream={is_streaming} "
            f"tags=[{_tags_str}]{_flag_str} "
            f"msgs={len(body.get('messages', []))} "
            f"dropped={turns_dropped} "
            f"ctx={context_tokens}t input={input_tokens}t "
            f"vc={overhead_ms}ms | {user_message[:60]}"
        )

        # Capture pre-filter request body for dashboard inspection
        metrics.capture_request(
            turn, _pre_filter_body, api_format,
            inbound_tags=assembled.matched_tags if assembled else [],
        )

        if is_streaming:
            return await _handle_streaming(
                client, url, fwd_headers, enriched_body, api_format, state,
                metrics=metrics, turn=turn, overhead_ms=overhead_ms,
            )
        else:
            return await _handle_non_streaming(
                client, url, fwd_headers, enriched_body, api_format, state,
                metrics=metrics, turn=turn, overhead_ms=overhead_ms,
            )

    return app


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

async def _passthrough(
    client: httpx.AsyncClient,
    request: Request,
    url: str,
    headers: dict[str, str],
) -> StreamingResponse:
    """Transparent forwarding for non-chat requests."""
    body = await request.body()
    return await _passthrough_bytes(client, request.method, url, headers, body)


async def _passthrough_bytes(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> StreamingResponse:
    """Forward raw bytes to upstream and stream back."""
    resp = await client.request(method, url, headers=headers, content=body)
    return JSONResponse(
        content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
        status_code=resp.status_code,
        headers=dict(resp.headers),
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
    overhead_ms: float = 0.0,
) -> StreamingResponse | JSONResponse:
    """Forward SSE stream, accumulating assistant text for on_turn_complete.

    Forwards raw bytes from the upstream to preserve exact SSE framing.
    The Node.js Anthropic SDK is strict about SSE formatting — decoding
    and re-encoding via ``aiter_lines()`` can break its parser.

    Non-2xx upstream responses (rate limits, overloads) are returned as
    JSON errors instead of broken SSE streams.
    """
    import json as _json

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
                "upstream_ms": upstream_ms,
                "total_ms": round(overhead_ms + upstream_ms, 1),
                "streaming": True,
                "error": True,
            })
        print(
            f"[T{turn}] ERROR {upstream.status_code} "
            f"llm={int(upstream_ms)}ms | {error_bytes[:200].decode('utf-8', errors='replace')}"
        )
        try:
            error_body = _json.loads(error_bytes)
        except (ValueError, _json.JSONDecodeError):
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

    async def stream_generator():
        text_chunks: list[str] = []
        line_buf = ""
        try:
            async for raw_chunk in upstream.aiter_bytes():
                yield raw_chunk  # forward raw bytes unchanged

                # Side-channel: parse for text accumulation
                line_buf += raw_chunk.decode("utf-8", errors="replace")
                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        try:
                            data = _json.loads(data_str)
                            delta = _extract_delta_text(data, api_format)
                            if delta:
                                text_chunks.append(delta)
                        except _json.JSONDecodeError:
                            pass
        finally:
            await upstream.aclose()
            upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)
            if metrics:
                metrics.record({
                    "type": "response",
                    "turn": turn,
                    "upstream_ms": upstream_ms,
                    "total_ms": round(overhead_ms + upstream_ms, 1),
                    "streaming": True,
                })
            assistant_text = "".join(text_chunks)
            print(
                f"[T{turn}] RESPONSE stream={True} "
                f"llm={int(upstream_ms)}ms "
                f"total={int(round(overhead_ms + upstream_ms))}ms "
                f"chars={len(assistant_text)}"
            )
            if state and assistant_text:
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text)
                )
                state.fire_turn_complete(list(state.conversation_history))

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
    overhead_ms: float = 0.0,
) -> JSONResponse:
    """Forward JSON response, parse assistant text, fire on_turn_complete."""
    t_upstream = time.monotonic()
    resp = await client.request("POST", url, headers=headers, json=body)
    upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)

    try:
        response_body = resp.json()
    except Exception:
        return JSONResponse(content=resp.text, status_code=resp.status_code)

    # Extract and record assistant text
    assistant_text = _extract_assistant_text(response_body, api_format)
    if state and assistant_text:
        state.conversation_history.append(
            Message(role="assistant", content=assistant_text)
        )
        state.fire_turn_complete(list(state.conversation_history))

    if metrics:
        metrics.record({
            "type": "response",
            "turn": turn,
            "upstream_ms": upstream_ms,
            "total_ms": round(overhead_ms + upstream_ms, 1),
            "streaming": False,
        })

    # Forward response headers (filter hop-by-hop)
    resp_headers = _forward_headers(dict(resp.headers))

    return JSONResponse(
        content=response_body,
        status_code=resp.status_code,
        headers=resp_headers,
    )
