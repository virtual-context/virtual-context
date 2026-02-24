"""Payload format abstraction for multi-provider proxy support.

Each LLM provider (Anthropic, OpenAI, Gemini) has a distinct request/response
schema.  ``PayloadFormat`` is the strategy interface; concrete subclasses
implement provider-specific extraction, injection, and SSE parsing.

Usage:

    fmt = detect_format(body)
    user_msg = fmt.extract_user_message(body)
    enriched = fmt.inject_context(body, prepend_text)
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Shared helpers (provider-agnostic)
# ---------------------------------------------------------------------------

_VC_PROMPT_MARKER = "[vc:prompt]\n"
_MEMOS_QUERY_DELIM = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b"

_VC_SESSION_RE = re.compile(r"<!-- vc:session=([a-f0-9-]+) -->")

_VC_USER_RE = re.compile(r"^\[vc:user\](.*?)\[/vc:user\]", re.DOTALL)
_SYSTEM_EVENT_RE = re.compile(r"^(?:System:\s*\[[^\]]*\][^\n]*\n+)+")
_CHANNEL_HEADER_RE = re.compile(r"^\[[A-Z][a-zA-Z]*\s[^\]]*\bid:-?\d+\b[^\]]*\]\s*")
_MESSAGE_ID_RE = re.compile(r"\n?\[message_id:\s*\d+\]\s*$")


def _last_text_block(content: list) -> str:
    """Return the text of the last ``type: "text"`` block in *content*."""
    for block in reversed(content):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def _strip_openclaw_envelope(text: str) -> str:
    """Strip OpenClaw channel metadata from a message."""
    if not text:
        return text

    if text.startswith(_VC_PROMPT_MARKER):
        text = text[len(_VC_PROMPT_MARKER):].lstrip()

    if text.startswith("# Role"):
        idx = text.find(_MEMOS_QUERY_DELIM)
        if idx != -1:
            text = text[idx + len(_MEMOS_QUERY_DELIM):].lstrip()

    m = _VC_USER_RE.match(text)
    if m:
        return m.group(1).strip()

    text = _SYSTEM_EVENT_RE.sub("", text)
    text = _CHANNEL_HEADER_RE.sub("", text)
    text = _MESSAGE_ID_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class PayloadFormat(ABC):
    """Strategy interface for provider-specific request/response handling."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: ``"anthropic"``, ``"openai"``, ``"gemini"``."""

    # -- Message extraction --------------------------------------------------

    @abstractmethod
    def extract_user_message(self, body: dict) -> str:
        """Extract the last user message text from a request body."""

    @abstractmethod
    def extract_message_text(self, msg: dict) -> str:
        """Extract text from a single message dict."""

    @abstractmethod
    def extract_history_pairs(self, body: dict) -> list:
        """Extract complete user+assistant pairs from request history.

        Returns a flat list of Message objects:
        [user_0, asst_0, user_1, asst_1, ...]
        """

    @abstractmethod
    def get_messages(self, body: dict) -> list[dict]:
        """Return the messages array from the request body."""

    @abstractmethod
    def has_messages(self, body: dict) -> bool:
        """Return True if the body has a valid messages array."""

    # -- Context injection ---------------------------------------------------

    @abstractmethod
    def inject_context(self, body: dict, prepend_text: str) -> dict:
        """Inject <virtual-context> block into a deep-copied request body."""

    # -- Session markers -----------------------------------------------------

    @abstractmethod
    def extract_session_id(self, body: dict) -> str | None:
        """Scan assistant messages for vc:session marker."""

    @abstractmethod
    def strip_session_markers(self, body: dict) -> dict:
        """Strip vc:session markers from all assistant messages."""

    @abstractmethod
    def inject_session_marker(self, response_body: dict, marker: str) -> dict:
        """Append session marker to the last text block in a non-streaming response."""

    @abstractmethod
    def emit_session_marker_sse(self, session_id: str) -> bytes:
        """Return a single SSE event bytes that injects a session marker."""

    # -- SSE / response parsing ----------------------------------------------

    @abstractmethod
    def extract_delta_text(self, data: dict) -> str:
        """Extract text delta from a streaming SSE event payload."""

    @abstractmethod
    def extract_assistant_text(self, response_body: dict) -> str:
        """Extract assistant text from a non-streaming response."""

    # -- Payload token estimation --------------------------------------------

    def estimate_payload_tokens(self, body: dict) -> int:
        """Estimate total input tokens from a request body (chars // 4)."""
        total = 0
        for m in self.get_messages(body):
            c = m.get("content", "")
            if isinstance(c, str):
                total += len(c) // 4
            elif isinstance(c, list):
                total += sum(
                    len(b.get("text", "")) // 4
                    for b in c if isinstance(b, dict)
                )
        total += self._estimate_system_tokens(body)
        return total

    def _estimate_system_tokens(self, body: dict) -> int:
        """Estimate system prompt tokens."""
        return 0

    # -- Fingerprinting ------------------------------------------------------

    def compute_fingerprint(self, body: dict, offset: int = 0) -> str:
        """Trailing conversation fingerprint from the last N user messages."""
        messages = self.get_messages(body)
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if len(user_msgs) < 2:
            return ""
        history_user = user_msgs[:-1]

        n = 1  # _FINGERPRINT_SAMPLE_SIZE
        end = len(history_user) - offset
        start = end - n
        if start < 0 or end <= 0:
            return ""
        sample = history_user[start:end]
        if not sample:
            return ""

        texts = [self._msg_text(m) for m in sample]
        combined = "\n".join(texts)
        if not combined.strip():
            return ""
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @staticmethod
    def _msg_text(msg: dict) -> str:
        """Extract plain text from a message content (str or content blocks)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return ""

    # -- Paging tool support -------------------------------------------------

    @property
    def supports_tool_interception(self) -> bool:
        """Whether this format supports VC paging tool interception."""
        return False

    def inject_tools(
        self,
        body: dict,
        tool_defs: list,
        require_tool_use: bool | None = None,
    ) -> dict:
        """Inject tool definitions into the request body. Override in subclasses."""
        return body

    def is_tool_use_event(self, data: dict) -> bool:
        """Return True if *data* is a streaming event containing a tool call."""
        return False

    def extract_tool_calls(self, content: list) -> list[dict]:
        """Extract tool calls from response content blocks."""
        return []

    def build_tool_results(self, results: list[dict]) -> list[dict]:
        """Build tool_result content blocks for continuation."""
        return results

    def build_continuation_request(
        self,
        original_body: dict,
        assistant_content: list[dict],
        tool_results: list[dict],
    ) -> dict:
        """Build a non-streaming continuation request after VC tool execution.

        Override in subclasses for format-specific message structure.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicFormat(PayloadFormat):
    """Anthropic Messages API format."""

    @property
    def name(self) -> str:
        return "anthropic"

    def extract_user_message(self, body: dict) -> str:
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

    def extract_message_text(self, msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return _strip_openclaw_envelope(content)
        if isinstance(content, list):
            return _strip_openclaw_envelope(_last_text_block(content))
        return ""

    def extract_history_pairs(self, body: dict) -> list:
        from ..types import Message

        messages = body.get("messages", [])
        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        if not chat_msgs:
            return []
        if chat_msgs and chat_msgs[-1].get("role") == "user":
            chat_msgs = chat_msgs[:-1]
        if not chat_msgs:
            return []

        pairs: list[Message] = []
        i = 0
        while i + 1 < len(chat_msgs):
            if (chat_msgs[i].get("role") == "user"
                    and chat_msgs[i + 1].get("role") == "assistant"):
                pairs.append(Message(
                    role="user",
                    content=self.extract_message_text(chat_msgs[i]),
                ))
                pairs.append(Message(
                    role="assistant",
                    content=self.extract_message_text(chat_msgs[i + 1]),
                ))
                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        return body.get("messages", [])

    def has_messages(self, body: dict) -> bool:
        return isinstance(body.get("messages"), list)

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"
        existing = body.get("system", "")
        if isinstance(existing, list):
            body["system"] = [{"type": "text", "text": context_block}] + existing
        else:
            body["system"] = f"{context_block}\n\n{existing}" if existing else context_block
        return body

    def extract_session_id(self, body: dict) -> str | None:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                m = _VC_SESSION_RE.search(content)
                if m:
                    return m.group(1)
            elif isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        m = _VC_SESSION_RE.search(block.get("text", ""))
                        if m:
                            return m.group(1)
        return None

    def strip_session_markers(self, body: dict) -> dict:
        messages = body.get("messages")
        if not messages:
            return body

        modified = False
        new_messages = []
        for msg in messages:
            if msg.get("role") != "assistant":
                new_messages.append(msg)
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                cleaned = _VC_SESSION_RE.sub("", content).rstrip()
                if cleaned != content:
                    msg = dict(msg)
                    msg["content"] = cleaned
                    modified = True
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        cleaned = _VC_SESSION_RE.sub("", text).rstrip()
                        if cleaned != text:
                            block = dict(block)
                            block["text"] = cleaned
                            modified = True
                    new_blocks.append(block)
                if modified:
                    msg = dict(msg)
                    msg["content"] = new_blocks
            new_messages.append(msg)

        if not modified:
            return body

        body = dict(body)
        body["messages"] = new_messages
        return body

    def inject_session_marker(self, response_body: dict, marker: str) -> dict:
        response_body = copy.deepcopy(response_body)
        content = response_body.get("content", [])
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = (block.get("text", "") or "") + marker
                return response_body
        content.append({"type": "text", "text": marker})
        return response_body

    def emit_session_marker_sse(self, session_id: str) -> bytes:
        marker = f"\n<!-- vc:session={session_id} -->"
        marker_event = json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": marker},
        })
        return f"event: content_block_delta\ndata: {marker_event}\n\n".encode()

    def extract_delta_text(self, data: dict) -> str:
        event_type = data.get("type", "")
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            return delta.get("text", "") or ""
        return ""

    def extract_assistant_text(self, response_body: dict) -> str:
        content = response_body.get("content", [])
        return _last_text_block(content)

    def _estimate_system_tokens(self, body: dict) -> int:
        sys_raw = body.get("system", "")
        if isinstance(sys_raw, str):
            return len(sys_raw) // 4
        if isinstance(sys_raw, list):
            return sum(
                len(b.get("text", "")) // 4
                for b in sys_raw if isinstance(b, dict)
            ) // 1  # integer
        return 0

    @property
    def supports_tool_interception(self) -> bool:
        return True

    def inject_tools(
        self,
        body: dict,
        tool_defs: list,
        require_tool_use: bool | None = None,
    ) -> dict:
        tc = body.get("tool_choice")
        if isinstance(tc, dict) and tc.get("type") == "none":
            return body
        if tc == "none":
            return body
        body = dict(body)
        tools = list(body.get("tools") or [])
        tools.extend(tool_defs)
        body["tools"] = tools
        if require_tool_use and "tool_choice" not in body:
            body["tool_choice"] = {"type": "any"}
        return body

    def build_continuation_request(
        self,
        original_body: dict,
        assistant_content: list[dict],
        tool_results: list[dict],
    ) -> dict:
        body: dict = {
            "model": original_body.get("model"),
            "max_tokens": original_body.get("max_tokens", 4096),
            "stream": False,
            "messages": list(original_body.get("messages", [])),
        }
        if "system" in original_body:
            body["system"] = original_body["system"]
        if "tools" in original_body:
            body["tools"] = original_body["tools"]
        body["messages"].append({"role": "assistant", "content": assistant_content})
        body["messages"].append({"role": "user", "content": tool_results})
        return body


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIFormat(PayloadFormat):
    """OpenAI Chat Completions API format."""

    @property
    def name(self) -> str:
        return "openai"

    def extract_user_message(self, body: dict) -> str:
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

    def extract_message_text(self, msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return _strip_openclaw_envelope(content)
        if isinstance(content, list):
            return _strip_openclaw_envelope(_last_text_block(content))
        return ""

    def extract_history_pairs(self, body: dict) -> list:
        from ..types import Message

        messages = body.get("messages", [])
        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        if not chat_msgs:
            return []
        if chat_msgs and chat_msgs[-1].get("role") == "user":
            chat_msgs = chat_msgs[:-1]
        if not chat_msgs:
            return []

        pairs: list[Message] = []
        i = 0
        while i + 1 < len(chat_msgs):
            if (chat_msgs[i].get("role") == "user"
                    and chat_msgs[i + 1].get("role") == "assistant"):
                pairs.append(Message(
                    role="user",
                    content=self.extract_message_text(chat_msgs[i]),
                ))
                pairs.append(Message(
                    role="assistant",
                    content=self.extract_message_text(chat_msgs[i + 1]),
                ))
                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        return body.get("messages", [])

    def has_messages(self, body: dict) -> bool:
        return isinstance(body.get("messages"), list)

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"
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

    def extract_session_id(self, body: dict) -> str | None:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                m = _VC_SESSION_RE.search(content)
                if m:
                    return m.group(1)
            elif isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        m = _VC_SESSION_RE.search(block.get("text", ""))
                        if m:
                            return m.group(1)
        return None

    def strip_session_markers(self, body: dict) -> dict:
        messages = body.get("messages")
        if not messages:
            return body

        modified = False
        new_messages = []
        for msg in messages:
            if msg.get("role") != "assistant":
                new_messages.append(msg)
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                cleaned = _VC_SESSION_RE.sub("", content).rstrip()
                if cleaned != content:
                    msg = dict(msg)
                    msg["content"] = cleaned
                    modified = True
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        cleaned = _VC_SESSION_RE.sub("", text).rstrip()
                        if cleaned != text:
                            block = dict(block)
                            block["text"] = cleaned
                            modified = True
                    new_blocks.append(block)
                if modified:
                    msg = dict(msg)
                    msg["content"] = new_blocks
            new_messages.append(msg)

        if not modified:
            return body

        body = dict(body)
        body["messages"] = new_messages
        return body

    def inject_session_marker(self, response_body: dict, marker: str) -> dict:
        response_body = copy.deepcopy(response_body)
        choices = response_body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            existing = msg.get("content", "") or ""
            msg["content"] = existing + marker
        return response_body

    def emit_session_marker_sse(self, session_id: str) -> bytes:
        marker = f"\n<!-- vc:session={session_id} -->"
        marker_event = json.dumps({
            "choices": [{"index": 0, "delta": {"content": marker}}],
        })
        return f"data: {marker_event}\n\n".encode()

    def extract_delta_text(self, data: dict) -> str:
        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "") or ""
        return ""

    def extract_assistant_text(self, response_body: dict) -> str:
        choices = response_body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "") or ""
        return ""

    def _estimate_system_tokens(self, body: dict) -> int:
        msgs = body.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            sc = msgs[0].get("content", "")
            if isinstance(sc, str):
                return len(sc) // 4
            if isinstance(sc, list):
                return sum(
                    len(b.get("text", "")) // 4
                    for b in sc if isinstance(b, dict)
                )
        return 0

    def build_continuation_request(
        self,
        original_body: dict,
        assistant_content: list[dict],
        tool_results: list[dict],
    ) -> dict:
        """OpenAI continuation uses the same messages structure."""
        body: dict = {
            "model": original_body.get("model"),
            "max_tokens": original_body.get("max_tokens", 4096),
            "stream": False,
            "messages": list(original_body.get("messages", [])),
        }
        if "system" in original_body:
            body["system"] = original_body["system"]
        if "tools" in original_body:
            body["tools"] = original_body["tools"]
        body["messages"].append({"role": "assistant", "content": assistant_content})
        body["messages"].append({"role": "user", "content": tool_results})
        return body


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiFormat(PayloadFormat):
    """Google Gemini API format.

    Key differences:
    - Messages are in ``contents`` (not ``messages``)
    - System prompt is ``system_instruction.parts[]``
    - Assistant role is ``"model"`` (not ``"assistant"``)
    - Content is ``parts`` (list of ``{text: ...}``)
    - SSE delta: ``candidates[0].content.parts[0].text``
    - Tool calls: ``functionCall`` in ``parts[]``
    """

    @property
    def name(self) -> str:
        return "gemini"

    # -- helpers --

    @staticmethod
    def _extract_text_from_parts(parts: list) -> str:
        """Join text fields from a Gemini parts list."""
        texts = []
        for p in parts:
            if isinstance(p, dict) and "text" in p:
                texts.append(p["text"])
        return " ".join(texts)

    # -- Message extraction --

    def extract_user_message(self, body: dict) -> str:
        contents = body.get("contents", [])
        for msg in reversed(contents):
            if msg.get("role") != "user":
                continue
            parts = msg.get("parts", [])
            text = self._extract_text_from_parts(parts)
            return _strip_openclaw_envelope(text)
        return ""

    def extract_message_text(self, msg: dict) -> str:
        parts = msg.get("parts", [])
        text = self._extract_text_from_parts(parts)
        return _strip_openclaw_envelope(text)

    def extract_history_pairs(self, body: dict) -> list:
        from ..types import Message

        contents = body.get("contents", [])
        chat_msgs = [m for m in contents if m.get("role") in ("user", "model")]
        if not chat_msgs:
            return []
        if chat_msgs and chat_msgs[-1].get("role") == "user":
            chat_msgs = chat_msgs[:-1]
        if not chat_msgs:
            return []

        pairs: list[Message] = []
        i = 0
        while i + 1 < len(chat_msgs):
            if (chat_msgs[i].get("role") == "user"
                    and chat_msgs[i + 1].get("role") == "model"):
                pairs.append(Message(
                    role="user",
                    content=self.extract_message_text(chat_msgs[i]),
                ))
                pairs.append(Message(
                    role="assistant",
                    content=self.extract_message_text(chat_msgs[i + 1]),
                ))
                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        return body.get("contents", [])

    def has_messages(self, body: dict) -> bool:
        return isinstance(body.get("contents"), list)

    # -- Context injection --

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"

        # Gemini uses system_instruction.parts[] for system prompt
        si = body.get("system_instruction", {})
        existing_parts = si.get("parts", []) if isinstance(si, dict) else []
        new_parts = [{"text": context_block}] + list(existing_parts)
        body["system_instruction"] = {"parts": new_parts}
        return body

    # -- Session markers --

    def extract_session_id(self, body: dict) -> str | None:
        for msg in reversed(body.get("contents", [])):
            if msg.get("role") != "model":
                continue
            parts = msg.get("parts", [])
            for part in reversed(parts):
                if isinstance(part, dict) and "text" in part:
                    m = _VC_SESSION_RE.search(part["text"])
                    if m:
                        return m.group(1)
        return None

    def strip_session_markers(self, body: dict) -> dict:
        contents = body.get("contents")
        if not contents:
            return body

        modified = False
        new_contents = []
        for msg in contents:
            if msg.get("role") != "model":
                new_contents.append(msg)
                continue

            parts = msg.get("parts", [])
            new_parts = []
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    cleaned = _VC_SESSION_RE.sub("", part["text"]).rstrip()
                    if cleaned != part["text"]:
                        part = dict(part)
                        part["text"] = cleaned
                        modified = True
                new_parts.append(part)
            if modified:
                msg = dict(msg)
                msg["parts"] = new_parts
            new_contents.append(msg)

        if not modified:
            return body
        body = dict(body)
        body["contents"] = new_contents
        return body

    def inject_session_marker(self, response_body: dict, marker: str) -> dict:
        response_body = copy.deepcopy(response_body)
        candidates = response_body.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            for part in reversed(parts):
                if isinstance(part, dict) and "text" in part:
                    part["text"] = part["text"] + marker
                    return response_body
            parts.append({"text": marker})
        return response_body

    def emit_session_marker_sse(self, session_id: str) -> bytes:
        marker = f"\n<!-- vc:session={session_id} -->"
        # Gemini SSE format: candidates[0].content.parts[0].text
        event_data = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{"text": marker}],
                    "role": "model",
                },
            }],
        })
        return f"data: {event_data}\n\n".encode()

    # -- SSE / response parsing --

    def extract_delta_text(self, data: dict) -> str:
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "") or ""
        return ""

    def extract_assistant_text(self, response_body: dict) -> str:
        candidates = response_body.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
            return " ".join(texts)
        return ""

    def _estimate_system_tokens(self, body: dict) -> int:
        si = body.get("system_instruction", {})
        if isinstance(si, dict):
            parts = si.get("parts", [])
            return sum(
                len(p.get("text", "")) // 4
                for p in parts if isinstance(p, dict)
            )
        return 0

    # -- Fingerprinting (override for Gemini's different structure) --

    def compute_fingerprint(self, body: dict, offset: int = 0) -> str:
        contents = body.get("contents", [])
        user_msgs = [m for m in contents if m.get("role") == "user"]
        if len(user_msgs) < 2:
            return ""
        history_user = user_msgs[:-1]

        n = 1
        end = len(history_user) - offset
        start = end - n
        if start < 0 or end <= 0:
            return ""
        sample = history_user[start:end]
        if not sample:
            return ""

        texts = [self._extract_text_from_parts(m.get("parts", [])) for m in sample]
        combined = "\n".join(texts)
        if not combined.strip():
            return ""
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @property
    def supports_tool_interception(self) -> bool:
        return True

    def inject_tools(
        self,
        body: dict,
        tool_defs: list,
        require_tool_use: bool | None = None,
    ) -> dict:
        """Inject tool definitions using Gemini's functionDeclarations format."""
        body = dict(body)
        tools = list(body.get("tools") or [])
        # Gemini expects: tools: [{functionDeclarations: [...]}]
        # Convert from Anthropic tool format to Gemini format
        declarations = []
        for td in tool_defs:
            decl = {
                "name": td["name"],
                "description": td.get("description", ""),
            }
            schema = td.get("input_schema")
            if schema:
                decl["parameters"] = schema
            declarations.append(decl)

        if tools and isinstance(tools[0], dict) and "functionDeclarations" in tools[0]:
            tools[0]["functionDeclarations"].extend(declarations)
        else:
            tools.append({"functionDeclarations": declarations})
        body["tools"] = tools
        return body

    def build_continuation_request(
        self,
        original_body: dict,
        assistant_content: list[dict],
        tool_results: list[dict],
    ) -> dict:
        """Build Gemini continuation with functionCall/functionResponse parts."""
        body: dict = {
            "model": original_body.get("model"),
            "contents": list(original_body.get("contents", [])),
        }
        if "system_instruction" in original_body:
            body["system_instruction"] = original_body["system_instruction"]
        if "tools" in original_body:
            body["tools"] = original_body["tools"]
        if "generationConfig" in original_body:
            body["generationConfig"] = original_body["generationConfig"]

        # Gemini tool calls: model message with functionCall parts
        model_parts = []
        for block in assistant_content:
            if block.get("type") == "tool_use":
                model_parts.append({
                    "functionCall": {
                        "name": block["name"],
                        "args": block.get("input", {}),
                    }
                })
            elif block.get("type") == "text":
                model_parts.append({"text": block.get("text", "")})
        body["contents"].append({"role": "model", "parts": model_parts})

        # Tool results: user message with functionResponse parts
        user_parts = []
        for result in tool_results:
            user_parts.append({
                "functionResponse": {
                    "name": result.get("name", ""),
                    "response": {"content": result.get("content", "")},
                }
            })
        body["contents"].append({"role": "user", "parts": user_parts})
        return body


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------

class OpenAIResponsesFormat(PayloadFormat):
    """OpenAI Responses API format (used by Codex and newer OpenAI tools).

    Key differences from Chat Completions:
    - Messages are in ``input`` (not ``messages``)
    - System prompt is ``instructions`` (not ``system``)
    - Items have ``type`` (``message``) and ``role``
    - Content is ``content`` list with ``type: "input_text"`` / ``"output_text"``
    - SSE delta: ``response.output_text.delta`` events with ``delta`` field
    - Tool calls: ``function_call`` / ``function_call_output`` item types
    """

    @property
    def name(self) -> str:
        return "openai_responses"

    # -- helpers --

    @staticmethod
    def _extract_text_from_content(content) -> str:
        """Extract text from a Responses API content field.

        Content can be:
        - A plain string (simple user input)
        - A list of content blocks with ``type: "input_text"`` or ``"output_text"``
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("input_text", "output_text"):
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "text":
                        texts.append(block.get("text", ""))
            return " ".join(texts) if texts else ""
        return ""

    @staticmethod
    def _is_bare_item(item: dict) -> bool:
        """Return True if the item is a bare function_call or function_call_output."""
        item_type = item.get("type", "")
        return item_type in ("function_call", "function_call_output")

    # -- Message extraction --

    def extract_user_message(self, body: dict) -> str:
        items = body.get("input", [])
        if isinstance(items, str):
            return _strip_openclaw_envelope(items)
        if not isinstance(items, list):
            return ""
        for item in reversed(items):
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue
            content = item.get("content", "")
            text = self._extract_text_from_content(content)
            if text:
                return _strip_openclaw_envelope(text)
        return ""

    def extract_message_text(self, msg: dict) -> str:
        content = msg.get("content", "")
        text = self._extract_text_from_content(content)
        return _strip_openclaw_envelope(text)

    def extract_history_pairs(self, body: dict) -> list:
        from ..types import Message

        items = body.get("input", [])
        if not isinstance(items, list):
            return []
        # Filter to user/assistant message items, skip bare function_call items
        chat_msgs = [
            m for m in items
            if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and not self._is_bare_item(m)
        ]
        if not chat_msgs:
            return []
        if chat_msgs[-1].get("role") == "user":
            chat_msgs = chat_msgs[:-1]
        if not chat_msgs:
            return []

        pairs: list[Message] = []
        i = 0
        while i + 1 < len(chat_msgs):
            if (chat_msgs[i].get("role") == "user"
                    and chat_msgs[i + 1].get("role") == "assistant"):
                pairs.append(Message(
                    role="user",
                    content=self.extract_message_text(chat_msgs[i]),
                ))
                pairs.append(Message(
                    role="assistant",
                    content=self.extract_message_text(chat_msgs[i + 1]),
                ))
                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        items = body.get("input", [])
        return items if isinstance(items, list) else []

    def has_messages(self, body: dict) -> bool:
        return isinstance(body.get("input"), list)

    # -- Context injection --

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"
        existing = body.get("instructions", "")
        body["instructions"] = (
            f"{context_block}\n\n{existing}" if existing else context_block
        )
        return body

    # -- Session markers --

    def extract_session_id(self, body: dict) -> str | None:
        items = body.get("input", [])
        if not isinstance(items, list):
            return None
        for item in reversed(items):
            if not isinstance(item, dict) or item.get("role") != "assistant":
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                m = _VC_SESSION_RE.search(content)
                if m:
                    return m.group(1)
            elif isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text:
                            m = _VC_SESSION_RE.search(text)
                            if m:
                                return m.group(1)
        return None

    def strip_session_markers(self, body: dict) -> dict:
        items = body.get("input")
        if not isinstance(items, list) or not items:
            return body

        modified = False
        new_items = []
        for item in items:
            if not isinstance(item, dict) or item.get("role") != "assistant":
                new_items.append(item)
                continue

            content = item.get("content", "")
            if isinstance(content, str):
                cleaned = _VC_SESSION_RE.sub("", content).rstrip()
                if cleaned != content:
                    item = dict(item)
                    item["content"] = cleaned
                    modified = True
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        text = block["text"]
                        cleaned = _VC_SESSION_RE.sub("", text).rstrip()
                        if cleaned != text:
                            block = dict(block)
                            block["text"] = cleaned
                            modified = True
                    new_blocks.append(block)
                if modified:
                    item = dict(item)
                    item["content"] = new_blocks
            new_items.append(item)

        if not modified:
            return body

        body = dict(body)
        body["input"] = new_items
        return body

    def inject_session_marker(self, response_body: dict, marker: str) -> dict:
        response_body = copy.deepcopy(response_body)
        output = response_body.get("output", [])
        # Find last output_text block in output items
        for item in reversed(output):
            if not isinstance(item, dict):
                continue
            # Items with type "message" have content list
            content = item.get("content", [])
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        block["text"] = (block.get("text", "") or "") + marker
                        return response_body
        # Fallback: append a new output item
        output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": marker}],
        })
        return response_body

    def emit_session_marker_sse(self, session_id: str) -> bytes:
        marker = f"\n<!-- vc:session={session_id} -->"
        marker_event = json.dumps({
            "type": "response.output_text.delta",
            "delta": marker,
        })
        return f"event: response.output_text.delta\ndata: {marker_event}\n\n".encode()

    # -- SSE / response parsing --

    def extract_delta_text(self, data: dict) -> str:
        event_type = data.get("type", "")
        if event_type == "response.output_text.delta":
            return data.get("delta", "") or ""
        return ""

    def extract_assistant_text(self, response_body: dict) -> str:
        output = response_body.get("output", [])
        texts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        texts.append(block.get("text", ""))
        return " ".join(texts) if texts else ""

    # -- Token estimation --

    def _estimate_system_tokens(self, body: dict) -> int:
        instructions = body.get("instructions", "")
        if isinstance(instructions, str):
            return len(instructions) // 4
        return 0

    def estimate_payload_tokens(self, body: dict) -> int:
        """Override to handle bare function_call/function_call_output items."""
        total = 0
        for item in self.get_messages(body):
            if not isinstance(item, dict):
                continue
            # Bare function_call/function_call_output items have different structure
            if self._is_bare_item(item):
                # Estimate from name + arguments/output
                name = item.get("name", "") or item.get("call_id", "")
                args = item.get("arguments", "") or item.get("output", "")
                total += (len(str(name)) + len(str(args))) // 4
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                total += sum(
                    len(b.get("text", "")) // 4
                    for b in content if isinstance(b, dict)
                )
        total += self._estimate_system_tokens(body)
        return total

    # -- Fingerprinting --

    def compute_fingerprint(self, body: dict, offset: int = 0) -> str:
        """Override to filter input items by role=user (skip bare items)."""
        items = body.get("input", [])
        if not isinstance(items, list):
            return ""
        user_msgs = [
            m for m in items
            if isinstance(m, dict)
            and m.get("role") == "user"
            and not self._is_bare_item(m)
        ]
        if len(user_msgs) < 2:
            return ""
        history_user = user_msgs[:-1]

        n = 1  # _FINGERPRINT_SAMPLE_SIZE
        end = len(history_user) - offset
        start = end - n
        if start < 0 or end <= 0:
            return ""
        sample = history_user[start:end]
        if not sample:
            return ""

        texts = [self._extract_text_from_content(m.get("content", "")) for m in sample]
        combined = "\n".join(texts)
        if not combined.strip():
            return ""
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @property
    def supports_tool_interception(self) -> bool:
        return True

    def inject_tools(
        self,
        body: dict,
        tool_defs: list,
        require_tool_use: bool | None = None,
    ) -> dict:
        tc = body.get("tool_choice")
        if isinstance(tc, dict) and tc.get("type") == "none":
            return body
        if tc == "none":
            return body
        body = dict(body)
        tools = list(body.get("tools") or [])
        for td in tool_defs:
            tools.append({
                "type": "function",
                "name": td["name"],
                "description": td.get("description", ""),
                "parameters": td.get("input_schema", {}),
            })
        body["tools"] = tools
        if require_tool_use and "tool_choice" not in body:
            body["tool_choice"] = "required"
        return body

    def is_tool_use_event(self, data: dict) -> bool:
        dtype = data.get("type", "")
        if dtype == "response.output_item.added":
            return data.get("item", {}).get("type") == "function_call"
        return dtype.startswith("response.function_call_arguments")

    def extract_tool_calls(self, content: list) -> list[dict]:
        calls = []
        for item in content:
            if item.get("type") == "function_call":
                args_raw = item.get("arguments", "")
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "name": item.get("name", ""),
                    "input": args,
                })
        return calls

    def build_tool_results(self, results: list[dict]) -> list[dict]:
        return [{
            "type": "function_call_output",
            "call_id": r.get("tool_use_id", r.get("call_id", "")),
            "output": r.get("content", ""),
        } for r in results]

    def build_continuation_request(
        self,
        original_body: dict,
        assistant_content: list[dict],
        tool_results: list[dict],
    ) -> dict:
        body = copy.deepcopy(original_body)
        body["stream"] = False
        inp = list(body.get("input", []))
        for block in assistant_content:
            if block.get("type") == "function_call":
                inp.append(block)
        for r in tool_results:
            inp.append(r)
        body["input"] = inp
        return body


# ---------------------------------------------------------------------------
# Format registry + detection
# ---------------------------------------------------------------------------

_FORMAT_REGISTRY: dict[str, PayloadFormat] = {
    "anthropic": AnthropicFormat(),
    "openai": OpenAIFormat(),
    "openai_responses": OpenAIResponsesFormat(),
    "gemini": GeminiFormat(),
}


def detect_format(body: dict) -> PayloadFormat:
    """Auto-detect the API format from a request body.

    Detection order:
    1. ``contents`` or ``system_instruction`` → Gemini
    2. ``input`` (as list) or ``instructions`` → OpenAI Responses
    3. ``system`` (top-level) → Anthropic
    4. Model name starts with ``"claude"`` → Anthropic
    5. Default → OpenAI (Chat Completions)
    """
    if "contents" in body or "system_instruction" in body:
        return _FORMAT_REGISTRY["gemini"]
    if isinstance(body.get("input"), list) or "instructions" in body:
        return _FORMAT_REGISTRY["openai_responses"]
    if "system" in body:
        return _FORMAT_REGISTRY["anthropic"]
    model = body.get("model", "")
    if isinstance(model, str) and model.startswith("claude"):
        return _FORMAT_REGISTRY["anthropic"]
    return _FORMAT_REGISTRY["openai"]


def get_format(name: str) -> PayloadFormat:
    """Look up a format by name."""
    return _FORMAT_REGISTRY[name]
