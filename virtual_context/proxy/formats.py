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

import base64 as _b64_mod
import copy
import hashlib
import io
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Shared helpers (provider-agnostic) — canonical definitions in _envelope.py
# ---------------------------------------------------------------------------

from ._envelope import (  # noqa: E402
    _VC_CONVERSATION_RE,
    _extract_envelope_metadata,
    _last_text_block,
    _strip_envelope,
    extract_timestamp_from_metadata,
)


# ---------------------------------------------------------------------------
# Normalized tool-call / tool-output info
# ---------------------------------------------------------------------------

class ToolCallInfo(NamedTuple):
    """Normalized descriptor for a tool call across all provider formats."""
    call_id: str
    name: str
    arguments: object  # dict, str, or None depending on provider
    msg_index: int


class ToolOutputInfo(NamedTuple):
    """Normalized descriptor for a tool output across all provider formats.

    ``carrier`` is the mutable dict that owns the content in the request body.
    ``carrier_type`` identifies the replacement strategy:
      - ``"anthropic"``: carrier is a ``tool_result`` content block
      - ``"openai_chat"``: carrier is a ``role: "tool"`` message dict
      - ``"openai_responses"``: carrier is a bare ``function_call_output`` item
    """
    call_id: str
    content: str
    carrier: dict
    carrier_type: str
    msg_index: int


class TurnGroup(NamedTuple):
    """A logical turn — user message + assistant response + any tool rounds."""
    indices: list[int]       # raw message/item indices in this turn
    role: str                # "user" — the initiating role
    has_tool_activity: bool  # whether this turn includes tool calls/results


class MediaBlockInfo(NamedTuple):
    """Descriptor for a media block (image, audio, etc.) in the payload."""
    msg_index: int
    block_index: int
    media_type: str                    # "image", "audio", etc.
    setter: Callable                   # existing setter for raw data replacement
    replace_with_text: Callable[[str], None]  # replaces block with text placeholder
    carrier: dict                      # the parent message/item


# ---------------------------------------------------------------------------
# Media token estimation helpers
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


class MediaBlock(NamedTuple):
    """Base64 media found in a content block."""
    b64_data: str
    media_type: str  # "image/png", "application/pdf", etc.


@dataclass
class PayloadTokenCache:
    """Reusable token-count cache for append-mostly client payloads."""
    format_name: str
    message_key: str
    shell_fingerprint: str
    shell_tokens: int
    message_fingerprints: list[str]
    message_tokens: list[int]
    separator_tokens: int
    total_tokens: int


@dataclass
class PayloadTokenEstimate:
    """Structured result for segmented payload token estimation."""
    total_tokens: int
    cache: PayloadTokenCache
    reused_prefix_messages: int
    recounted_messages: int
    shell_cache_hit: bool


def _get_image_dimensions(b64_data: str) -> tuple[int, int] | None:
    """Read (width, height) from a base64-encoded image by parsing the header only."""
    try:
        from PIL import Image
        # Only decode first ~32 KB — enough for any image header
        prefix = b64_data[:43700]
        padding = (4 - len(prefix) % 4) % 4
        raw = _b64_mod.b64decode(prefix + "=" * padding)
        img = Image.open(io.BytesIO(raw))
        return img.size  # (width, height)
    except Exception:
        return None


def _anthropic_image_tokens(width: int, height: int) -> int:
    """Anthropic's image token formula: scale to fit 1568x1568, then (w*h)/750."""
    max_dim = max(width, height)
    if max_dim > 1568:
        scale = 1568 / max_dim
        width = int(width * scale)
        height = int(height * scale)
    return max(1, -(-width * height // 750))  # ceiling division


def _count_pdf_pages(b64_data: str) -> int:
    """Count pages in a base64-encoded PDF without external libraries."""
    try:
        raw = _b64_mod.b64decode(b64_data)
        # Look for /Type /Page (not /Pages) in the cross-reference objects.
        # This is the standard way pages are declared in PDF structure.
        import re as _re
        # Match /Type /Page with optional whitespace, but NOT /Type /Pages
        count = len(_re.findall(rb'/Type\s*/Page(?!s)', raw))
        return max(1, count)
    except Exception:
        # Fallback: estimate from base64 size (~100KB per page is typical)
        decoded_size = len(b64_data) * 3 // 4
        return max(1, decoded_size // 100_000)


def _estimate_media_tokens(media: MediaBlock) -> int:
    """Estimate tokens for a base64-encoded media block based on its type."""
    mt = media.media_type.lower()

    # Images: use Anthropic's (w*h)/750 formula
    if mt.startswith("image/"):
        dims = _get_image_dimensions(media.b64_data)
        if dims:
            return _anthropic_image_tokens(*dims)
        return 1049  # fallback: typical 1024x768

    # PDFs: Anthropic renders each page as ~1568x1196 image → ~2502 tokens/page
    if mt == "application/pdf" or mt.endswith("/pdf"):
        pages = _count_pdf_pages(media.b64_data)
        return pages * 2502

    # Audio: Anthropic charges ~1 token per 1.6 seconds of audio.
    # Average base64 size per second varies by codec, but ~16KB/s for mp3 is typical.
    # Decoded bytes / 16000 ≈ seconds, then seconds / 1.6 ≈ tokens.
    if mt.startswith("audio/"):
        decoded_size = len(media.b64_data) * 3 // 4
        seconds = max(1, decoded_size // 16000)
        return max(1, -(-seconds * 10 // 16))  # seconds / 1.6

    # Unknown media: use decoded byte size as conservative proxy.
    # Better than tokenizing the base64 as text (which is ~33% larger).
    decoded_size = len(media.b64_data) * 3 // 4
    return max(1, decoded_size // 40)  # ~40 bytes per token is conservative


# ---------------------------------------------------------------------------
# Non-standard message normalization (e.g. OpenClaw internal format)
# ---------------------------------------------------------------------------

def normalize_messages(messages: list) -> list:
    """Normalize non-standard message formats in-place.

    Handles OpenClaw's internal storage format and other non-standard schemas:
      - role: "toolResult" → role: "tool" + tool_call_id
      - content block type: "toolCall" → converted to tool_calls array
      - Anthropic-style tool_use content blocks → tool_calls array
      - User messages with only tool_result blocks → role: "tool" messages
      - Error/failed assistant messages (stopReason: "error", empty content) → removed
      - Non-standard metadata keys cleaned from all messages
    This runs once before any pipeline processing so that all downstream
    code (group_into_turns, chain collapse, trim, iter_tool_*) works
    correctly without per-consumer special cases.
    """
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict):
            i += 1
            continue

        # --- Remove error/failed messages entirely ---
        if (msg.get("role") == "assistant"
                and msg.get("stopReason") == "error"):
            messages.pop(i)
            continue
        # Also catch empty-content assistant messages with errorMessage
        if (msg.get("role") == "assistant"
                and msg.get("errorMessage")
                and not _has_real_content(msg)):
            messages.pop(i)
            continue

        # --- toolResult → role: "tool" ---
        if msg.get("role") == "toolResult":
            content = msg.get("content", "")
            # Flatten content blocks to string for OpenAI Chat tool format
            if isinstance(content, list):
                text_parts = []
                for b in content:
                    if isinstance(b, dict) and b.get("text"):
                        text_parts.append(b["text"])
                content = "\n".join(text_parts) if text_parts else str(content)
            msg["role"] = "tool"
            msg["tool_call_id"] = msg.pop("toolCallId", msg.get("tool_call_id", ""))
            msg["content"] = content
            # Clean up non-standard keys
            msg.pop("toolName", None)
            msg.pop("isError", None)
            msg.pop("timestamp", None)

        # --- assistant content blocks: only convert OpenClaw "toolCall" blocks ---
        # IMPORTANT: Do NOT convert standard Anthropic "tool_use" blocks —
        # those are the native format for Anthropic requests and must stay as-is.
        elif msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                tool_calls = []
                remaining = []
                for block in content:
                    if not isinstance(block, dict):
                        remaining.append(block)
                        continue
                    if block.get("type") == "toolCall":
                        # OpenClaw-specific format → OpenAI Chat tool_calls
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": block.get("arguments", ""),
                            },
                        })
                    else:
                        remaining.append(block)
                if tool_calls:
                    msg["tool_calls"] = msg.get("tool_calls", []) + tool_calls
                    msg["content"] = remaining if remaining else None

            # Clean up non-standard keys on assistant messages
            for k in ("api", "provider", "stopReason", "usage",
                       "responseId", "timestamp", "errorMessage",
                       "thinkingSignature"):
                msg.pop(k, None)

        # --- user messages: only clean non-standard keys ---
        # IMPORTANT: Do NOT convert standard Anthropic "tool_result" blocks
        # to role: "tool" messages — Anthropic requires them as user content.
        elif msg.get("role") == "user":
            msg.pop("timestamp", None)

        i += 1

    return messages


def _has_real_content(msg: dict) -> bool:
    """Check if a message has non-empty text content."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("text", "").strip()
            for b in content
        )
    return False


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class PayloadFormat(ABC):
    """Strategy interface for provider-specific request/response handling."""

    def __init__(self) -> None:
        self._count: Callable[[str], int] = lambda text: max(1, len(text) // 4) if text else 0

    def set_token_counter(self, counter: Callable[[str], int]) -> None:
        """Replace the default chars//4 estimator with an accurate counter (e.g. tiktoken)."""
        self._count = counter

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: ``"anthropic"``, ``"openai"``, ``"gemini"``."""

    # -- Message extraction --------------------------------------------------

    @abstractmethod
    def extract_user_message(self, body: dict) -> str:
        ...

    @abstractmethod
    def extract_message_text(self, msg: dict) -> str:
        ...

    @abstractmethod
    def extract_history_pairs(self, body: dict) -> list:
        """Extract complete user+assistant pairs from request history.

        Returns a flat list of Message objects:
        [user_0, asst_0, user_1, asst_1, ...]
        """

    @abstractmethod
    def get_messages(self, body: dict) -> list[dict]:
        ...

    @abstractmethod
    def has_messages(self, body: dict) -> bool:
        ...

    # -- Turn grouping -------------------------------------------------------

    @abstractmethod
    def group_into_turns(self, body: dict) -> list[TurnGroup]:
        """Group messages/items into logical turns.

        Each turn starts with a user message, includes the assistant response,
        and absorbs any tool call/result rounds that follow. Trailing user
        messages (no assistant response) form their own group.

        This is the SOLE source of truth for turn boundaries. All protected-window
        consumers must use this method.
        """
        ...

    # -- Tool-call / tool-output traversal -----------------------------------

    def iter_tool_calls(self, body: dict) -> Iterator[ToolCallInfo]:
        """Yield normalized tool-call descriptors from all messages/items.

        Handles Anthropic (``tool_use`` content blocks), OpenAI Chat
        (``tool_calls`` on assistant messages), and OpenAI Responses
        (bare ``function_call`` items).
        """
        messages = self.get_messages(body)
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            # --- Anthropic: tool_use content blocks in assistant messages ---
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            yield ToolCallInfo(
                                call_id=block.get("id", ""),
                                name=block.get("name", ""),
                                arguments=block.get("input"),
                                msg_index=i,
                            )
                # --- OpenAI Chat: tool_calls array on assistant messages ---
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        yield ToolCallInfo(
                            call_id=tc.get("id", ""),
                            name=fn.get("name", "") if isinstance(fn, dict) else "",
                            arguments=fn.get("arguments") if isinstance(fn, dict) else None,
                            msg_index=i,
                        )

            # --- OpenAI Responses: bare function_call items ---
            if msg.get("type") == "function_call":
                yield ToolCallInfo(
                    call_id=msg.get("call_id", ""),
                    name=msg.get("name", ""),
                    arguments=msg.get("arguments"),
                    msg_index=i,
                )

    def iter_tool_outputs(self, body: dict) -> Iterator[ToolOutputInfo]:
        """Yield normalized tool-output descriptors from all messages/items.

        Handles Anthropic (``tool_result`` content blocks in user messages),
        OpenAI Chat (``role: "tool"`` messages), and OpenAI Responses
        (bare ``function_call_output`` items).
        """
        messages = self.get_messages(body)
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            # --- Anthropic: tool_result content blocks in user messages ---
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            text = self._extract_tool_result_text(block)
                            yield ToolOutputInfo(
                                call_id=block.get("tool_use_id", ""),
                                content=text,
                                carrier=block,
                                carrier_type="anthropic",
                                msg_index=i,
                            )

            # --- OpenAI Chat: role: "tool" messages ---
            if msg.get("role") == "tool":
                yield ToolOutputInfo(
                    call_id=msg.get("tool_call_id", ""),
                    content=msg.get("content", "") or "",
                    carrier=msg,
                    carrier_type="openai_chat",
                    msg_index=i,
                )

            # --- OpenAI Responses: bare function_call_output items ---
            if msg.get("type") == "function_call_output":
                yield ToolOutputInfo(
                    call_id=msg.get("call_id", ""),
                    content=msg.get("output", "") or "",
                    carrier=msg,
                    carrier_type="openai_responses",
                    msg_index=i,
                )

    @staticmethod
    def _extract_tool_result_text(block: dict) -> str:
        """Extract plain text from an Anthropic tool_result content block."""
        content = block.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return ""

    # -- Context injection ---------------------------------------------------

    @abstractmethod
    def inject_context(self, body: dict, prepend_text: str) -> dict:
        ...

    # -- Conversation markers -----------------------------------------------------

    @abstractmethod
    def extract_conversation_id(self, body: dict) -> str | None:
        ...

    @abstractmethod
    def strip_conversation_markers(self, body: dict) -> dict:
        ...

    @abstractmethod
    def inject_conversation_marker(self, response_body: dict, marker: str) -> dict:
        ...

    @abstractmethod
    def emit_conversation_marker_sse(self, conversation_id: str) -> bytes:
        """Return a single SSE event bytes that injects a conversation marker."""

    @abstractmethod
    def emit_fake_response_sse(self, text: str, conversation_id: str) -> bytes:
        """Return SSE bytes for a complete fake LLM response with conversation marker."""

    @abstractmethod
    def build_fake_response(self, text: str, conversation_id: str) -> dict:
        """Return a format-correct non-streaming response body with conversation marker."""

    # -- Raw content extraction -----------------------------------------------

    def extract_user_raw_content(self, body: dict) -> list[dict] | None:
        """Extract raw content blocks from the last user message.

        Returns None if the format doesn't support structured content blocks.
        Subclasses override to provide format-specific extraction.
        """
        return None

    def extract_assistant_raw_content(self, response_body: dict) -> list[dict] | None:
        """Extract raw content blocks from an assistant response.

        Returns None if the format doesn't support structured content blocks.
        Subclasses override to provide format-specific extraction.
        """
        return None

    # -- SSE / response parsing ----------------------------------------------

    @abstractmethod
    def extract_delta_text(self, data: dict) -> str:
        ...

    @abstractmethod
    def extract_assistant_text(self, response_body: dict) -> str:
        ...

    # -- Image base64 extraction (for token adjustment) -----------------------

    def _extract_media_from_block(self, block: dict) -> MediaBlock | None:
        """Extract base64 media from a content block with its media type, or None.

        Handles all Anthropic base64 source blocks (image, document, audio, etc.)
        and OpenAI Chat data-URI image blocks.
        """
        # Anthropic: {"type": "image"|"document"|..., "source": {"type": "base64", "data": "...", "media_type": "..."}}
        source = block.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            data = source.get("data")
            if data:
                return MediaBlock(data, source.get("media_type", "application/octet-stream"))
        # OpenAI Chat: {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}
        if block.get("type") == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                header, b64 = url.split(";base64,", 1)
                mt = header.replace("data:", "") or "image/unknown"
                return MediaBlock(b64, mt)
        return None

    def _iter_nested_dicts(self, value: object) -> Iterator[dict]:
        """Yield every nested dict inside *value* depth-first."""
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from self._iter_nested_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._iter_nested_dicts(child)

    def _collect_media_from_value(self, value: object) -> list[MediaBlock]:
        """Return all media blocks reachable within *value*."""
        results: list[MediaBlock] = []
        for block in self._iter_nested_dicts(value):
            media = self._extract_media_from_block(block)
            if media:
                results.append(media)
        return results

    def _blank_non_token_fields_in_value(self, value: object) -> list[tuple[dict, str, str]]:
        """Blank fields that should not be counted as prompt text."""
        saved: list[tuple[dict, str, str]] = []
        for block in self._iter_nested_dicts(value):
            source = block.get("source")
            if isinstance(source, dict) and source.get("type") == "base64" and source.get("data"):
                saved.append((source, "data", source["data"]))
                source["data"] = ""
            elif block.get("type") == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if isinstance(url, str) and ";base64," in url:
                    saved.append((block["image_url"], "url", url))
                    block["image_url"]["url"] = url.split(";base64,")[0] + ";base64,"
            elif "inline_data" in block:
                inline = block.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    saved.append((inline, "data", inline["data"]))
                    inline["data"] = ""
            elif block.get("type") == "input_image":
                url = block.get("image_url", "")
                if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                    saved.append((block, "image_url", url))
                    block["image_url"] = url.split(";base64,")[0] + ";base64,"
            if block.get("type") == "thinking" and isinstance(block.get("signature"), str) and block.get("signature"):
                saved.append((block, "signature", block["signature"]))
                block["signature"] = ""
        return saved

    def _collect_media(self, body: dict) -> list[MediaBlock]:
        """Return all base64 media blocks found in the payload."""
        results: list[MediaBlock] = []
        for msg in self.get_messages(body):
            results.extend(self._collect_media_from_value(msg))
        return results

    # -- Payload token estimation --------------------------------------------

    def _message_key(self, body: dict) -> str:
        if "messages" in body:
            return "messages"
        if "input" in body:
            return "input"
        if "contents" in body:
            return "contents"
        return ""

    def _body_without_messages(self, body: dict) -> dict:
        shell = copy.deepcopy(body)
        key = self._message_key(shell)
        if key:
            shell[key] = []
        return shell

    @staticmethod
    def _fingerprint_payload_shell(body: dict) -> str:
        return hashlib.sha256(
            json.dumps(body, default=str, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _fingerprint_message_for_cache(msg: dict) -> str:
        return hashlib.sha256(
            json.dumps(msg, default=str, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _estimate_payload_tokens_with_media(
        self,
        body: dict,
        media_list: list[MediaBlock],
        *,
        serialized_json: str | None = None,
    ) -> int:
        saved = self._blank_media_data(body)
        if not media_list and not saved:
            if serialized_json is None:
                serialized_json = json.dumps(body, default=str)
            return self._count(serialized_json)

        stripped_json = json.dumps(body, default=str)
        self._restore_media_data(body, saved)

        text_tokens = self._count(stripped_json)
        media_tokens = sum(_estimate_media_tokens(m) for m in media_list)
        return max(1, text_tokens + media_tokens)

    def estimate_payload_tokens(self, body: dict) -> int:
        """Estimate total input tokens from a request body.

        Base64 media (images, PDFs, audio, etc.) is counted using
        provider-appropriate formulas instead of tokenizing the raw base64.
        """
        media_list = self._collect_media(body)
        return self._estimate_payload_tokens_with_media(body, media_list)

    def estimate_payload_tokens_from_serialized(
        self,
        body: dict,
        serialized_json: str,
    ) -> int:
        """Estimate payload tokens, reusing an already-serialized payload when possible."""
        media_list = self._collect_media(body)
        return self._estimate_payload_tokens_with_media(
            body,
            media_list,
            serialized_json=serialized_json,
        )

    def estimate_payload_tokens_segmented(
        self,
        body: dict,
        *,
        cache: PayloadTokenCache | None = None,
    ) -> PayloadTokenEstimate:
        """Estimate payload tokens via shell + per-message counts with prefix reuse.

        This is designed for append-mostly client payloads that resend a large
        stable history on every request. We fingerprint each message, reuse the
        cached prefix when it still matches, and only recount the changed tail.
        """
        messages = self.get_messages(body)
        message_key = self._message_key(body)
        shell = self._body_without_messages(body)
        shell_fingerprint = self._fingerprint_payload_shell(shell)

        shell_cache_hit = bool(
            cache
            and cache.format_name == self.name
            and cache.message_key == message_key
            and cache.shell_fingerprint == shell_fingerprint
        )
        if shell_cache_hit:
            shell_tokens = cache.shell_tokens
        else:
            shell_tokens = self.estimate_payload_tokens(shell)

        reusable_cache = None
        if (
            cache
            and cache.format_name == self.name
            and cache.message_key == message_key
        ):
            reusable_cache = cache

        message_fingerprints: list[str] = []
        message_tokens: list[int] = []
        total_message_tokens = 0
        reused_prefix_messages = 0

        if reusable_cache:
            reusable = min(
                len(messages),
                len(reusable_cache.message_fingerprints),
                len(reusable_cache.message_tokens),
            )
            for idx in range(reusable):
                fingerprint = self._fingerprint_message_for_cache(messages[idx])
                if fingerprint != reusable_cache.message_fingerprints[idx]:
                    break
                message_fingerprints.append(fingerprint)
                cached_tokens = reusable_cache.message_tokens[idx]
                message_tokens.append(cached_tokens)
                total_message_tokens += cached_tokens
                reused_prefix_messages += 1

        for msg in messages[reused_prefix_messages:]:
            fingerprint = self._fingerprint_message_for_cache(msg)
            token_count = self.estimate_message_tokens(msg)
            message_fingerprints.append(fingerprint)
            message_tokens.append(token_count)
            total_message_tokens += token_count

        separator_tokens = self._count("," * max(0, len(messages) - 1))
        total_tokens = shell_tokens + total_message_tokens + separator_tokens
        next_cache = PayloadTokenCache(
            format_name=self.name,
            message_key=message_key,
            shell_fingerprint=shell_fingerprint,
            shell_tokens=shell_tokens,
            message_fingerprints=message_fingerprints,
            message_tokens=message_tokens,
            separator_tokens=separator_tokens,
            total_tokens=total_tokens,
        )
        return PayloadTokenEstimate(
            total_tokens=total_tokens,
            cache=next_cache,
            reused_prefix_messages=reused_prefix_messages,
            recounted_messages=max(0, len(messages) - reused_prefix_messages),
            shell_cache_hit=shell_cache_hit,
        )

    def _blank_media_data(self, body: dict) -> list[tuple[dict, str, str]]:
        """Blank non-token payload fields before estimation.

        This strips raw base64 media and Anthropic thinking signatures anywhere
        inside message payloads, including nested tool_result content.
        """
        saved = []
        for msg in self.get_messages(body):
            saved.extend(self._blank_non_token_fields_in_value(msg))
        return saved

    @staticmethod
    def _restore_media_data(body: dict, saved: list[tuple[dict, str, str]]) -> None:
        """Restore blanked base64 data from save list."""
        for container, key, original in saved:
            container[key] = original

    def estimate_message_tokens(self, msg: dict) -> int:
        """Count tokens for a single message/item, using media formulas for base64 blocks."""
        raw_json = json.dumps(msg, default=str)
        media_list = self._collect_media_from_value(msg)
        saved = self._blank_non_token_fields_in_value(msg)
        if not media_list and not saved:
            return self._count(raw_json)
        stripped_json = json.dumps(msg, default=str)
        for container, key, original in saved:
            container[key] = original

        text_tokens = self._count(stripped_json)
        media_tokens = sum(_estimate_media_tokens(m) for m in media_list)
        return max(1, text_tokens + media_tokens)

    def _estimate_system_tokens(self, body: dict) -> int:
        return 0

    def estimate_tools_tokens(self, body: dict) -> int:
        tools = body.get("tools", [])
        if not tools:
            return 0
        return self._count(json.dumps(tools))

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
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return ""

    # -- Mutation methods ----------------------------------------------------

    def remove_items(self, body: dict, indices: list[int]) -> None:
        """Remove messages/items at the given indices. Handles index shifting."""
        messages = self.get_messages(body)
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(messages):
                messages.pop(idx)

    def insert_items(self, body: dict, position: int, items: list[dict]) -> None:
        """Insert new messages/items at the given position."""
        messages = self.get_messages(body)
        for i, item in enumerate(items):
            messages.insert(position + i, item)

    def remove_thinking_block(self, body: dict, msg_index: int, block_index: int) -> None:
        """Remove a thinking block from a message. Format-overridable.

        Anthropic: pops the block from the content list.
        All other formats: no-op (they don't have thinking blocks).
        """
        # Default: no-op. Anthropic overrides this.
        pass

    def replace_tool_output_content(self, body: dict, output_info: "ToolOutputInfo", new_content: str) -> None:
        """Replace a tool output's content using the carrier from iter_tool_outputs."""
        carrier = output_info.carrier
        ct = output_info.carrier_type
        if ct == "anthropic":
            carrier["content"] = new_content
        elif ct == "openai_chat":
            carrier["content"] = new_content
        elif ct == "openai_responses":
            carrier["output"] = new_content

    def iter_media_blocks(self, body: dict) -> Iterator["MediaBlockInfo"]:
        """Yield MediaBlockInfo for each media block in the payload.

        Each format overrides to detect its specific media block structure.
        Default implementation handles Anthropic and OpenAI Chat formats.
        """
        messages = self.get_messages(body)
        for mi, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                media_type, setter = self._detect_media_block(block)
                if media_type is not None:
                    def _make_replacer(_content, _bi):
                        def _replace(text: str) -> None:
                            _content[_bi] = {"type": "text", "text": text}
                        return _replace
                    yield MediaBlockInfo(
                        msg_index=mi,
                        block_index=bi,
                        media_type=media_type,
                        setter=setter,
                        replace_with_text=_make_replacer(content, bi),
                        carrier=msg,
                    )

    def _detect_media_block(self, block: dict) -> tuple[str | None, Callable | None]:
        """Detect if a content block is a media block. Returns (media_type, setter) or (None, None)."""
        # Anthropic: {"type": "image", "source": {"type": "base64", ...}}
        if block.get("type") == "image":
            source = block.get("source", {})
            if isinstance(source, dict) and source.get("type") == "base64":
                def _setter(blk, b64, mt):
                    blk["source"]["data"] = b64
                    blk["source"]["media_type"] = mt
                return source.get("media_type", "image"), _setter
        # OpenAI Chat: {"type": "image_url", "image_url": {"url": "data:..."}}
        if block.get("type") == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:") and ";base64," in url:
                header = url.split(";base64,", 1)[0]
                media_type = header.replace("data:", "")
                def _setter(blk, b64, mt):
                    blk["image_url"]["url"] = f"data:{mt};base64,{b64}"
                return media_type, _setter
        return None, None

    # -- Stub markers --------------------------------------------------------

    def mark_as_vc_stub(self, item: dict) -> None:
        """Add the VC stub marker to a message/item."""
        item["_vc_stub"] = True

    def is_vc_stub(self, body: dict, index: int) -> bool:
        """Check if a message/item at index has the VC stub marker."""
        messages = self.get_messages(body)
        if 0 <= index < len(messages):
            return bool(messages[index].get("_vc_stub"))
        return False

    def strip_vc_markers(self, body: dict) -> None:
        """Remove all _vc_stub markers from the payload."""
        messages = self.get_messages(body)
        for msg in messages:
            msg.pop("_vc_stub", None)

    # -- Conversational message detection and merging ------------------------

    def _is_conversational_message(self, msg: dict) -> bool:
        """Check if a message is purely conversational (no tool semantics).

        Conversational = user or assistant message with only text/image content.
        NOT conversational: tool messages, function_call items, tool_result carriers.
        """
        role = msg.get("role", "")
        if role not in ("user", "assistant", "human", "model"):
            return False
        # Check for tool semantics in content
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype in ("tool_use", "tool_result"):
                        return False
        # OpenAI Chat: assistant with tool_calls
        if msg.get("tool_calls"):
            return False
        # OpenAI Responses: bare items
        if msg.get("type") in ("function_call", "function_call_output", "reasoning"):
            return False
        # Gemini: function parts
        if isinstance(msg.get("parts"), list):
            for part in msg["parts"]:
                if isinstance(part, dict) and ("functionCall" in part or "functionResponse" in part):
                    return False
        return True

    def merge_consecutive_conversational(self, body: dict) -> None:
        """Merge adjacent same-role conversational messages.

        Only merges user/assistant messages that do NOT encode tool semantics.
        Tool messages, function items, reasoning items are never merged.
        """
        messages = self.get_messages(body)
        if len(messages) < 2:
            return

        merged = [messages[0]]
        for msg in messages[1:]:
            prev = merged[-1]
            prev_role = prev.get("role", prev.get("type", ""))
            curr_role = msg.get("role", msg.get("type", ""))

            if (prev_role == curr_role
                    and self._is_conversational_message(prev)
                    and self._is_conversational_message(msg)):
                # Merge content
                self._merge_message_content(prev, msg)
            else:
                merged.append(msg)

        # Replace messages in body
        msg_key = self._get_message_key(body)
        body[msg_key] = merged

    def _get_message_key(self, body: dict) -> str:
        """Return the key used for the message list in this format."""
        if "contents" in body:
            return "contents"
        if "input" in body and isinstance(body.get("input"), list):
            return "input"
        return "messages"

    @abstractmethod
    def _merge_message_content(self, target: dict, source: dict) -> None:
        """Merge source message content into target. Each format overrides."""
        ...

    @abstractmethod
    def extract_text_from_item(self, body: dict, index: int) -> str:
        """Extract all text content from the message/item at the given index.

        Returns a single string with all text content joined. Format-specific:
        each format knows its own content structure.
        """
        ...

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
        return body

    def is_tool_use_event(self, data: dict) -> bool:
        return False

    def extract_tool_calls(self, content: list) -> list[dict]:
        return []

    def build_tool_results(self, results: list[dict]) -> list[dict]:
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

    _SYSTEM_SCALE = 1.032
    _TOOLS_SCALE = 1.08
    _USER_TEXT_SCALE = 1.05
    _ASSISTANT_TEXT_SCALE = 1.33
    _TOOL_NAME_SCALE = 1.10
    _TOOL_USE_INPUT_SCALE = 1.255
    _TOOL_RESULT_TEXT_SCALE = 1.055
    _THINKING_SIGNATURE_SCALE = 0.226
    _TOOL_CALL_ID_TOKENS = 22
    _IMAGE_MEDIA_SCALE = 0.85
    _CACHE_BREAKPOINT = {"type": "ephemeral"}

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
                text = _strip_envelope(content)
                if text:
                    return text
            elif isinstance(content, list):
                text = _strip_envelope(_last_text_block(content))
                if text:
                    return text
            # No text in this user message (e.g. tool_result only) — keep looking
        return ""

    def extract_user_raw_content(self, body: dict) -> list[dict] | None:
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            if isinstance(content, list):
                return content
        return None

    def extract_assistant_raw_content(self, response_body: dict) -> list[dict] | None:
        content = response_body.get("content", [])
        if isinstance(content, list) and content:
            return content
        return None

    def extract_message_text(self, msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return _strip_envelope(content)
        if isinstance(content, list):
            return _strip_envelope(_last_text_block(content))
        return ""

    def extract_message_text_with_meta(self, msg: dict) -> tuple[str, dict]:
        content = msg.get("content", "")
        if isinstance(content, str):
            return _extract_envelope_metadata(content)
        if isinstance(content, list):
            text = _last_text_block(content)
            return _extract_envelope_metadata(text)
        return "", {}

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
        last_real_user: tuple[str, dict | None, "datetime | None"] | None = None
        i = 0
        while i + 1 < len(chat_msgs):
            if (chat_msgs[i].get("role") == "user"
                    and chat_msgs[i + 1].get("role") == "assistant"):
                # Check if user message is tool_result-only (API scaffolding)
                _user_raw = chat_msgs[i].get("content", "")
                _is_tool_result = False
                if isinstance(_user_raw, list):
                    _ctypes = {b.get("type") for b in _user_raw if isinstance(b, dict)}
                    _is_tool_result = bool(_ctypes and _ctypes <= {"tool_result"})

                if _is_tool_result:
                    # Skip tool_result user message, but if the next assistant
                    # has real text content, pair it with the last real user.
                    asst_text = self.extract_message_text(chat_msgs[i + 1])
                    if asst_text.strip() and last_real_user is not None:
                        _u_text, _u_meta, _u_ts = last_real_user
                        pairs.append(Message(
                            role="user", content=_u_text,
                            metadata=_u_meta, timestamp=_u_ts,
                        ))
                        pairs.append(Message(
                            role="assistant", content=asst_text,
                            timestamp=_u_ts,
                        ))
                        last_real_user = None
                    i += 2
                    continue

                text, meta = self.extract_message_text_with_meta(chat_msgs[i])
                ts = extract_timestamp_from_metadata(meta) if meta else None
                asst_text = self.extract_message_text(chat_msgs[i + 1])

                if asst_text.strip():
                    # Normal pair: real user + assistant with text
                    pairs.append(Message(
                        role="user", content=text,
                        metadata=meta or None, timestamp=ts,
                    ))
                    pairs.append(Message(
                        role="assistant", content=asst_text,
                        timestamp=ts,
                    ))
                    # If assistant also has tool_use, hold the user for the
                    # next tool_result→assistant text response. Otherwise clear.
                    _asst_raw = chat_msgs[i + 1].get("content", "")
                    _has_tool_use = False
                    if isinstance(_asst_raw, list):
                        _has_tool_use = any(
                            isinstance(b, dict) and b.get("type") == "tool_use"
                            for b in _asst_raw
                        )
                    if _has_tool_use:
                        last_real_user = (text, meta or None, ts)
                    else:
                        last_real_user = None
                else:
                    # Assistant is tool_use only — hold user for next text response
                    last_real_user = (text, meta or None, ts)

                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        return body.get("messages", [])

    def has_messages(self, body: dict) -> bool:
        return isinstance(body.get("messages"), list)

    def group_into_turns(self, body: dict) -> list[TurnGroup]:
        messages = body.get("messages", [])
        turns: list[TurnGroup] = []
        current_indices: list[int] = []
        has_tool = False

        def _is_real_user(msg: dict) -> bool:
            """True if user message has real content (not just tool_result blocks).

            A user message carrying tool_result blocks (even alongside media
            or text) is a tool_result carrier if any tool_result is present.
            This keeps the tool chain atomic with the preceding tool_use.
            """
            if msg.get("role") != "user":
                return False
            content = msg.get("content", "")
            if isinstance(content, str):
                return True
            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                return not has_tool_result
            return True

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if role == "user" and _is_real_user(msg):
                # Start of a new turn — flush the previous one
                if current_indices:
                    turns.append(TurnGroup(
                        indices=current_indices,
                        role="user",
                        has_tool_activity=has_tool,
                    ))
                current_indices = [i]
                has_tool = False
            else:
                if not current_indices:
                    # Orphaned message before any user message; start a group
                    current_indices = [i]
                else:
                    current_indices.append(i)
                # Detect tool activity
                if role == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") in ("tool_use",):
                                has_tool = True
                elif role == "user":
                    # tool_result-only user message
                    has_tool = True

        # Flush the last group
        if current_indices:
            turns.append(TurnGroup(
                indices=current_indices,
                role="user",
                has_tool_activity=has_tool,
            ))

        return turns

    # -- Anthropic-specific overrides ----------------------------------------

    def remove_thinking_block(self, body: dict, msg_index: int, block_index: int) -> None:
        """Anthropic: pop the thinking block from the content list."""
        messages = self.get_messages(body)
        msg = messages[msg_index]
        content = msg.get("content", [])
        if isinstance(content, list) and 0 <= block_index < len(content):
            content.pop(block_index)

    def _merge_message_content(self, target: dict, source: dict) -> None:
        """Anthropic: combine content block arrays."""
        tc = target.get("content", "")
        sc = source.get("content", "")
        if isinstance(tc, str):
            tc = [{"type": "text", "text": tc}] if tc else []
        if isinstance(sc, str):
            sc = [{"type": "text", "text": sc}] if sc else []
        target["content"] = list(tc) + list(sc)

    def extract_text_from_item(self, body: dict, index: int) -> str:
        msg = self.get_messages(body)[index]
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return ""

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_text = f"<system-reminder>\n{prepend_text}\n</system-reminder>"
        # Append to the last user message.  We append (not prepend) so that
        # tool_result blocks stay at the front of the content list — Anthropic
        # requires tool_result to immediately follow the preceding tool_use.
        # Anthropic prompt caching works best when the last stable block is the
        # explicit cache breakpoint, and the mutable VC reminder is isolated as
        # the trailing suffix after that breakpoint.
        messages = body.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            context_block = {"type": "text", "text": context_text}
            if isinstance(content, str):
                messages[i] = dict(msg)
                if content:
                    messages[i]["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": dict(self._CACHE_BREAKPOINT),
                        },
                        context_block,
                    ]
                else:
                    messages[i]["content"] = [context_block]
            elif isinstance(content, list):
                messages[i] = dict(msg)
                blocks = list(content)
                for j in range(len(blocks) - 1, -1, -1):
                    block = blocks[j]
                    if isinstance(block, dict):
                        updated = dict(block)
                        updated["cache_control"] = dict(self._CACHE_BREAKPOINT)
                        blocks[j] = updated
                        break
                blocks.append(context_block)
                messages[i]["content"] = blocks
            break
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": context_text}],
            })
        body["messages"] = messages
        return body

    def extract_conversation_id(self, body: dict) -> str | None:
        # Search BACKWARD — the most recent assistant marker is authoritative.
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                m = _VC_CONVERSATION_RE.search(content)
                if m:
                    return m.group(1)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        m = _VC_CONVERSATION_RE.search(block.get("text", ""))
                        if m:
                            return m.group(1)
        return None

    def strip_conversation_markers(self, body: dict) -> dict:
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
                cleaned = _VC_CONVERSATION_RE.sub("", content).rstrip()
                # Don't strip if it would leave the content empty —
                # the marker is harmless, an empty string is not.
                if cleaned != content and cleaned:
                    msg = dict(msg)
                    msg["content"] = cleaned
                    modified = True
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        cleaned = _VC_CONVERSATION_RE.sub("", text).rstrip()
                        # Don't strip if it would leave the text empty.
                        if cleaned != text and cleaned:
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

    def inject_conversation_marker(self, response_body: dict, marker: str) -> dict:
        response_body = copy.deepcopy(response_body)
        content = response_body.get("content", [])
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = (block.get("text", "") or "") + marker
                return response_body
        content.append({"type": "text", "text": marker})
        return response_body

    def emit_conversation_marker_sse(self, conversation_id: str) -> bytes:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        marker_event = json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": marker},
        })
        return f"event: content_block_delta\ndata: {marker_event}\n\n".encode()

    def emit_fake_response_sse(self, text: str, conversation_id: str) -> bytes:
        import uuid as _uuid
        msg_id = f"msg_vcattach_{_uuid.uuid4().hex[:12]}"
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        full_text = text + marker
        events = []
        events.append(f'event: message_start\ndata: {json.dumps({"type": "message_start", "message": {"id": msg_id, "type": "message", "role": "assistant", "content": [], "model": "vcattach", "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})}\n\n')
        events.append(f'event: content_block_start\ndata: {json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})}\n\n')
        events.append(f'event: content_block_delta\ndata: {json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": full_text}})}\n\n')
        events.append(f'event: content_block_stop\ndata: {json.dumps({"type": "content_block_stop", "index": 0})}\n\n')
        events.append(f'event: message_delta\ndata: {json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 0}})}\n\n')
        events.append(f'event: message_stop\ndata: {json.dumps({"type": "message_stop"})}\n\n')
        return "".join(events).encode()

    def build_fake_response(self, text: str, conversation_id: str) -> dict:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        return {
            "id": "msg_vcattach", "type": "message", "role": "assistant",
            "model": "vcattach",
            "content": [{"type": "text", "text": text + marker}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def extract_delta_text(self, data: dict) -> str:
        event_type = data.get("type", "")
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            return delta.get("text", "") or ""
        return ""

    def extract_assistant_text(self, response_body: dict) -> str:
        content = response_body.get("content", [])
        return _last_text_block(content)

    @staticmethod
    def _scale_tokens(tokens: int, factor: float) -> int:
        if tokens <= 0:
            return 0
        return max(1, int(math.ceil(tokens * factor)))

    def _estimate_scaled_text(self, text: str, factor: float) -> int:
        if not isinstance(text, str) or not text:
            return 0
        return self._scale_tokens(self._count(text), factor)

    def _estimate_signature_tokens(self, signature: str) -> int:
        if not isinstance(signature, str) or not signature:
            return 0
        return self._scale_tokens(self._count(signature), self._THINKING_SIGNATURE_SCALE)

    def _role_text_scale(self, role: str | None) -> float:
        if role == "assistant":
            return self._ASSISTANT_TEXT_SCALE
        if role == "user":
            return self._USER_TEXT_SCALE
        return 1.0

    def _estimate_media_tokens(self, media: MediaBlock) -> int:
        base = _estimate_media_tokens(media)
        if media.media_type.lower().startswith("image/"):
            return self._scale_tokens(base, self._IMAGE_MEDIA_SCALE)
        return base

    def _estimate_system_tokens(self, body: dict) -> int:
        sys_raw = body.get("system", "")
        if isinstance(sys_raw, str):
            return self._estimate_scaled_text(sys_raw, self._SYSTEM_SCALE)
        if isinstance(sys_raw, list):
            return sum(
                self._estimate_scaled_text(b.get("text", ""), self._SYSTEM_SCALE)
                for b in sys_raw if isinstance(b, dict)
            )
        return 0

    def _estimate_content_value_tokens(
        self,
        value: object,
        *,
        role: str | None = None,
        in_tool_result: bool = False,
    ) -> int:
        """Count Anthropic-visible semantic content, not wrapper JSON."""
        if isinstance(value, str):
            factor = self._TOOL_RESULT_TEXT_SCALE if in_tool_result else self._role_text_scale(role)
            return self._estimate_scaled_text(value, factor)
        if isinstance(value, list):
            total = 0
            for item in value:
                if isinstance(item, dict):
                    total += self._estimate_content_block_tokens(
                        item,
                        role=role,
                        in_tool_result=in_tool_result,
                    )
                elif isinstance(item, str):
                    factor = self._TOOL_RESULT_TEXT_SCALE if in_tool_result else self._role_text_scale(role)
                    total += self._estimate_scaled_text(item, factor)
            return total
        if isinstance(value, dict):
            return self._estimate_content_block_tokens(
                value,
                role=role,
                in_tool_result=in_tool_result,
            )
        return 0

    def _estimate_content_block_tokens(
        self,
        block: dict,
        *,
        role: str | None = None,
        in_tool_result: bool = False,
    ) -> int:
        """Count semantic payload fields for one Anthropic content block."""
        btype = block.get("type")
        if btype == "text":
            factor = self._TOOL_RESULT_TEXT_SCALE if in_tool_result else self._role_text_scale(role)
            return self._estimate_scaled_text(block.get("text", ""), factor)
        if btype == "thinking":
            return (
                self._estimate_scaled_text(block.get("text", ""), self._ASSISTANT_TEXT_SCALE)
                + self._estimate_scaled_text(block.get("thinking", ""), self._ASSISTANT_TEXT_SCALE)
                + self._estimate_signature_tokens(block.get("signature", ""))
            )
        if btype == "tool_use":
            total = 0
            if isinstance(block.get("id"), str):
                total += self._TOOL_CALL_ID_TOKENS
            if isinstance(block.get("name"), str):
                total += self._estimate_scaled_text(block["name"], self._TOOL_NAME_SCALE)
            if "input" in block:
                total += self._scale_tokens(
                    self._count(json.dumps(block.get("input"), sort_keys=True, default=str)),
                    self._TOOL_USE_INPUT_SCALE,
                )
            return total
        if btype == "tool_result":
            total = 0
            if isinstance(block.get("tool_use_id"), str):
                total += self._TOOL_CALL_ID_TOKENS
            total += self._estimate_content_value_tokens(
                block.get("content", ""),
                role=role,
                in_tool_result=True,
            )
            return total

        total = 0
        if isinstance(block.get("text"), str):
            factor = self._TOOL_RESULT_TEXT_SCALE if in_tool_result else self._role_text_scale(role)
            total += self._estimate_scaled_text(block["text"], factor)
        if "content" in block:
            total += self._estimate_content_value_tokens(
                block.get("content"),
                role=role,
                in_tool_result=in_tool_result,
            )
        if "input" in block:
            total += self._scale_tokens(
                self._count(json.dumps(block.get("input"), sort_keys=True, default=str)),
                self._TOOL_USE_INPUT_SCALE,
            )
        return total

    def estimate_message_tokens(self, msg: dict) -> int:
        """Anthropic tokens track semantic content more closely than request JSON wrappers."""
        content = msg.get("content", "")
        text_tokens = self._estimate_content_value_tokens(content, role=msg.get("role"))
        media_tokens = sum(self._estimate_media_tokens(m) for m in self._collect_media_from_value(msg))
        return max(1, text_tokens + media_tokens)

    def estimate_tools_tokens(self, body: dict) -> int:
        tools = body.get("tools", [])
        if not tools:
            return 0
        return self._scale_tokens(self._count(json.dumps(tools)), self._TOOLS_SCALE)

    def estimate_payload_tokens(self, body: dict) -> int:
        """Anthropic payload counting should follow visible system/tools/messages, not JSON scaffolding."""
        messages = self.get_messages(body)
        message_tokens = sum(self.estimate_message_tokens(msg) for msg in messages)
        separator_tokens = self._count("," * max(0, len(messages) - 1))
        return (
            self._estimate_system_tokens(body)
            + self.estimate_tools_tokens(body)
            + message_tokens
            + separator_tokens
        )

    def estimate_payload_tokens_from_serialized(
        self,
        body: dict,
        serialized_json: str,
    ) -> int:
        """Serialized JSON wrappers are not a good proxy for Anthropic prompt tokens."""
        return self.estimate_payload_tokens(body)

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
        existing_names = {t.get("name") for t in tools if isinstance(t, dict)}
        for td in tool_defs:
            if isinstance(td, dict) and td.get("name") not in existing_names:
                tools.append(td)
        body["tools"] = tools
        # Anthropic API rejects tool_choice=any/tool when thinking is enabled.
        # Downgrade any forcing tool_choice to auto in that case — tools are
        # still available, the model just isn't forced to call one.
        thinking = body.get("thinking")
        thinking_enabled = isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
        if thinking_enabled:
            existing_tc = body.get("tool_choice")
            if isinstance(existing_tc, dict) and existing_tc.get("type") in ("any", "tool"):
                body["tool_choice"] = {"type": "auto"}
        elif require_tool_use and "tool_choice" not in body:
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
                return _strip_envelope(content)
            if isinstance(content, list):
                return _strip_envelope(_last_text_block(content))
        return ""

    def extract_message_text(self, msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return _strip_envelope(content)
        if isinstance(content, list):
            return _strip_envelope(_last_text_block(content))
        return ""

    def extract_message_text_with_meta(self, msg: dict) -> tuple[str, dict]:
        content = msg.get("content", "")
        if isinstance(content, str):
            return _extract_envelope_metadata(content)
        if isinstance(content, list):
            text = _last_text_block(content)
            return _extract_envelope_metadata(text)
        return "", {}

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
        last_real_user: tuple[str, dict | None, "datetime | None"] | None = None
        i = 0
        while i + 1 < len(chat_msgs):
            if (chat_msgs[i].get("role") == "user"
                    and chat_msgs[i + 1].get("role") == "assistant"):
                # Check if user message is tool_result-only (API scaffolding)
                _user_raw = chat_msgs[i].get("content", "")
                _is_tool_result = False
                if isinstance(_user_raw, list):
                    _ctypes = {b.get("type") for b in _user_raw if isinstance(b, dict)}
                    _is_tool_result = bool(_ctypes and _ctypes <= {"tool_result"})

                if _is_tool_result:
                    # Skip tool_result user message, but if the next assistant
                    # has real text content, pair it with the last real user.
                    asst_text = self.extract_message_text(chat_msgs[i + 1])
                    if asst_text.strip() and last_real_user is not None:
                        _u_text, _u_meta, _u_ts = last_real_user
                        pairs.append(Message(
                            role="user", content=_u_text,
                            metadata=_u_meta, timestamp=_u_ts,
                        ))
                        pairs.append(Message(
                            role="assistant", content=asst_text,
                            timestamp=_u_ts,
                        ))
                        last_real_user = None
                    i += 2
                    continue

                text, meta = self.extract_message_text_with_meta(chat_msgs[i])
                ts = extract_timestamp_from_metadata(meta) if meta else None
                asst_text = self.extract_message_text(chat_msgs[i + 1])

                if asst_text.strip():
                    # Normal pair: real user + assistant with text
                    pairs.append(Message(
                        role="user", content=text,
                        metadata=meta or None, timestamp=ts,
                    ))
                    pairs.append(Message(
                        role="assistant", content=asst_text,
                        timestamp=ts,
                    ))
                    # If assistant also has tool_use, hold the user for the
                    # next tool_result→assistant text response.
                    _asst_raw = chat_msgs[i + 1].get("content", "")
                    _has_tool_use = False
                    if isinstance(_asst_raw, list):
                        _has_tool_use = any(
                            isinstance(b, dict) and b.get("type") == "tool_use"
                            for b in _asst_raw
                        )
                    if _has_tool_use:
                        last_real_user = (text, meta or None, ts)
                    else:
                        last_real_user = None
                else:
                    # Assistant is tool_use only — hold user for next text response
                    last_real_user = (text, meta or None, ts)

                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        return body.get("messages", [])

    def has_messages(self, body: dict) -> bool:
        return isinstance(body.get("messages"), list)

    def group_into_turns(self, body: dict) -> list[TurnGroup]:
        messages = body.get("messages", [])
        turns: list[TurnGroup] = []
        current_indices: list[int] = []
        has_tool = False

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if role == "user":
                # Start of a new turn — flush the previous one
                if current_indices:
                    turns.append(TurnGroup(
                        indices=current_indices,
                        role="user",
                        has_tool_activity=has_tool,
                    ))
                current_indices = [i]
                has_tool = False
            elif role == "tool":
                # Tool result — belongs to the current turn
                if not current_indices:
                    current_indices = [i]
                else:
                    current_indices.append(i)
                has_tool = True
            else:
                # assistant, system, or other
                if not current_indices:
                    current_indices = [i]
                else:
                    current_indices.append(i)
                # Detect tool calls on assistant messages
                if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                    has_tool = True

        # Flush the last group
        if current_indices:
            turns.append(TurnGroup(
                indices=current_indices,
                role="user",
                has_tool_activity=has_tool,
            ))

        return turns

    # -- OpenAI Chat-specific overrides --------------------------------------

    def _merge_message_content(self, target: dict, source: dict) -> None:
        """OpenAI Chat: concatenate content strings."""
        tc = target.get("content", "") or ""
        sc = source.get("content", "") or ""
        if isinstance(tc, str) and isinstance(sc, str):
            target["content"] = tc + "\n" + sc
            return
        # Fall back to list concatenation for array content
        if isinstance(tc, str):
            tc = [{"type": "text", "text": tc}] if tc else []
        if isinstance(sc, str):
            sc = [{"type": "text", "text": sc}] if sc else []
        target["content"] = list(tc) + list(sc)

    def extract_text_from_item(self, body: dict, index: int) -> str:
        msg = self.get_messages(body)[index]
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return ""

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_block = f"<system-reminder>\n{prepend_text}\n</system-reminder>"
        # Append to the LAST user message (current turn) so the conversation
        # prefix stays byte-identical between turns, maximising OpenAI prefix
        # caching.  Appending keeps tool-call content at the front.
        messages = body.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                messages[i] = dict(msg)
                messages[i]["content"] = f"{content}\n\n{context_block}"
            elif isinstance(content, list):
                messages[i] = dict(msg)
                messages[i]["content"] = list(content) + [{"type": "text", "text": context_block}]
            break
        else:
            messages.append({"role": "user", "content": context_block})
        body["messages"] = messages
        return body

    def extract_conversation_id(self, body: dict) -> str | None:
        # Search BACKWARD — the most recent assistant marker is authoritative.
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                m = _VC_CONVERSATION_RE.search(content)
                if m:
                    return m.group(1)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        m = _VC_CONVERSATION_RE.search(block.get("text", ""))
                        if m:
                            return m.group(1)
        return None

    def strip_conversation_markers(self, body: dict) -> dict:
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
                cleaned = _VC_CONVERSATION_RE.sub("", content).rstrip()
                # Don't strip if it would leave the content empty —
                # the marker is harmless, an empty string is not.
                if cleaned != content and cleaned:
                    msg = dict(msg)
                    msg["content"] = cleaned
                    modified = True
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        cleaned = _VC_CONVERSATION_RE.sub("", text).rstrip()
                        # Don't strip if it would leave the text empty.
                        if cleaned != text and cleaned:
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

    def inject_conversation_marker(self, response_body: dict, marker: str) -> dict:
        response_body = copy.deepcopy(response_body)
        choices = response_body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            existing = msg.get("content", "") or ""
            msg["content"] = existing + marker
        return response_body

    def emit_conversation_marker_sse(self, conversation_id: str) -> bytes:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        marker_event = json.dumps({
            "choices": [{"index": 0, "delta": {"content": marker}}],
        })
        return f"data: {marker_event}\n\n".encode()

    def emit_fake_response_sse(self, text: str, conversation_id: str) -> bytes:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        full_text = text + marker
        events = []
        events.append(f'data: {json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})}\n\n')
        events.append(f'data: {json.dumps({"choices": [{"index": 0, "delta": {"content": full_text}}]})}\n\n')
        events.append(f'data: {json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n')
        events.append("data: [DONE]\n\n")
        return "".join(events).encode()

    def build_fake_response(self, text: str, conversation_id: str) -> dict:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        return {
            "id": "chatcmpl-vcattach", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text + marker}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

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
                return self._count(sc)
            if isinstance(sc, list):
                return sum(
                    self._count(b.get("text", ""))
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
            return _strip_envelope(text)
        return ""

    def extract_message_text(self, msg: dict) -> str:
        parts = msg.get("parts", [])
        text = self._extract_text_from_parts(parts)
        return _strip_envelope(text)

    def extract_message_text_with_meta(self, msg: dict) -> tuple[str, dict]:
        parts = msg.get("parts", [])
        text = self._extract_text_from_parts(parts)
        return _extract_envelope_metadata(text)

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
                text, meta = self.extract_message_text_with_meta(chat_msgs[i])
                ts = extract_timestamp_from_metadata(meta) if meta else None
                pairs.append(Message(
                    role="user",
                    content=text,
                    metadata=meta or None,
                    timestamp=ts,
                ))
                pairs.append(Message(
                    role="assistant",
                    content=self.extract_message_text(chat_msgs[i + 1]),
                    timestamp=ts,
                ))
                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        return body.get("contents", [])

    def has_messages(self, body: dict) -> bool:
        return isinstance(body.get("contents"), list)

    def group_into_turns(self, body: dict) -> list[TurnGroup]:
        contents = body.get("contents", [])
        turns: list[TurnGroup] = []
        current_indices: list[int] = []
        has_tool = False

        def _is_real_user(msg: dict) -> bool:
            """True if user message has real content (not just functionResponse parts)."""
            if msg.get("role") != "user":
                return False
            parts = msg.get("parts", [])
            if not parts:
                return True
            types = set()
            for p in parts:
                if isinstance(p, dict):
                    if "functionResponse" in p:
                        types.add("functionResponse")
                    elif "text" in p:
                        types.add("text")
                    else:
                        types.add("other")
            return not (types and types <= {"functionResponse"})

        def _has_function_parts(msg: dict) -> bool:
            """True if message has functionCall or functionResponse parts."""
            parts = msg.get("parts", [])
            for p in parts:
                if isinstance(p, dict):
                    if "functionCall" in p or "functionResponse" in p:
                        return True
            return False

        for i, msg in enumerate(contents):
            role = msg.get("role", "")
            if role == "user" and _is_real_user(msg):
                # Start of a new turn — flush the previous one
                if current_indices:
                    turns.append(TurnGroup(
                        indices=current_indices,
                        role="user",
                        has_tool_activity=has_tool,
                    ))
                current_indices = [i]
                has_tool = False
            else:
                if not current_indices:
                    current_indices = [i]
                else:
                    current_indices.append(i)
                if _has_function_parts(msg):
                    has_tool = True

        # Flush the last group
        if current_indices:
            turns.append(TurnGroup(
                indices=current_indices,
                role="user",
                has_tool_activity=has_tool,
            ))

        return turns

    # -- Gemini-specific overrides -------------------------------------------

    def _merge_message_content(self, target: dict, source: dict) -> None:
        """Gemini: combine parts arrays."""
        target["parts"] = target.get("parts", []) + source.get("parts", [])

    def extract_text_from_item(self, body: dict, index: int) -> str:
        msg = self.get_messages(body)[index]
        parts = msg.get("parts", [])
        return " ".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p
        )

    def iter_media_blocks(self, body: dict) -> Iterator["MediaBlockInfo"]:
        """Gemini: media blocks are inline_data entries in parts."""
        messages = self.get_messages(body)
        for mi, msg in enumerate(messages):
            parts = msg.get("parts", [])
            if not isinstance(parts, list):
                continue
            for bi, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                if "inline_data" in part:
                    inline = part["inline_data"]
                    if isinstance(inline, dict):
                        media_type = inline.get("mime_type", "image")
                        def _setter(blk, b64, mt):
                            blk["inline_data"]["data"] = b64
                            blk["inline_data"]["mime_type"] = mt
                        def _make_replacer(_parts, _bi):
                            def _replace(text: str) -> None:
                                _parts[_bi] = {"text": text}
                            return _replace
                        yield MediaBlockInfo(
                            msg_index=mi,
                            block_index=bi,
                            media_type=media_type,
                            setter=_setter,
                            replace_with_text=_make_replacer(parts, bi),
                            carrier=msg,
                        )

    def _extract_media_from_block(self, block: dict) -> MediaBlock | None:
        """Gemini: inline_data blocks contain base64 media."""
        if "inline_data" in block:
            inline = block["inline_data"]
            if isinstance(inline, dict) and inline.get("data"):
                return MediaBlock(inline["data"], inline.get("mime_type", "application/octet-stream"))
        return None

    def _collect_media(self, body: dict) -> list[MediaBlock]:
        """Gemini: recurse through parts for inline_data blocks."""
        results: list[MediaBlock] = []
        for msg in self.get_messages(body):
            results.extend(self._collect_media_from_value(msg))
        return results

    def estimate_message_tokens(self, msg: dict) -> int:
        """Gemini: recurse through parts for media-aware token counting."""
        return super().estimate_message_tokens(msg)

    # -- Context injection --

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_block = f"<system-reminder>\n{prepend_text}\n</system-reminder>"
        # Append to the LAST user message (current turn) so the conversation
        # prefix stays byte-identical between turns, maximising Gemini context
        # caching.
        contents = body.get("contents", [])
        for i in range(len(contents) - 1, -1, -1):
            msg = contents[i]
            if msg.get("role") != "user":
                continue
            parts = msg.get("parts", [])
            contents[i] = dict(msg)
            contents[i]["parts"] = list(parts) + [{"text": context_block}]
            break
        else:
            # No user message — prepend as user message
            contents.insert(0, {"role": "user", "parts": [{"text": context_block}]})
        body["contents"] = contents
        return body

    # -- Conversation markers --

    def extract_conversation_id(self, body: dict) -> str | None:
        # Search BACKWARD — the most recent model marker is authoritative.
        for msg in reversed(body.get("contents", [])):
            if msg.get("role") != "model":
                continue
            parts = msg.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    m = _VC_CONVERSATION_RE.search(part["text"])
                    if m:
                        return m.group(1)
        return None

    def strip_conversation_markers(self, body: dict) -> dict:
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
                    cleaned = _VC_CONVERSATION_RE.sub("", part["text"]).rstrip()
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

    def inject_conversation_marker(self, response_body: dict, marker: str) -> dict:
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

    def emit_conversation_marker_sse(self, conversation_id: str) -> bytes:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
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

    def emit_fake_response_sse(self, text: str, conversation_id: str) -> bytes:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        full_text = text + marker
        event = json.dumps({"candidates": [{"content": {"parts": [{"text": full_text}], "role": "model"}, "finishReason": "STOP"}]})
        return f"data: {event}\n\n".encode()

    def build_fake_response(self, text: str, conversation_id: str) -> dict:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        return {
            "candidates": [{"content": {"parts": [{"text": text + marker}], "role": "model"}, "finishReason": "STOP"}],
        }

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
                self._count(p.get("text", ""))
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
            return _strip_envelope(items)
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
                return _strip_envelope(text)
        return ""

    def extract_message_text(self, msg: dict) -> str:
        content = msg.get("content", "")
        text = self._extract_text_from_content(content)
        return _strip_envelope(text)

    def extract_message_text_with_meta(self, msg: dict) -> tuple[str, dict]:
        content = msg.get("content", "")
        text = self._extract_text_from_content(content)
        return _extract_envelope_metadata(text)

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
        last_real_user: tuple[str, dict | None, "datetime | None"] | None = None
        i = 0
        while i + 1 < len(chat_msgs):
            if (chat_msgs[i].get("role") == "user"
                    and chat_msgs[i + 1].get("role") == "assistant"):
                # Check if user message is tool_result-only (API scaffolding)
                _user_raw = chat_msgs[i].get("content", "")
                _is_tool_result = False
                if isinstance(_user_raw, list):
                    _ctypes = {b.get("type") for b in _user_raw if isinstance(b, dict)}
                    _is_tool_result = bool(_ctypes and _ctypes <= {"tool_result"})

                if _is_tool_result:
                    # Skip tool_result user message, but if the next assistant
                    # has real text content, pair it with the last real user.
                    asst_text = self.extract_message_text(chat_msgs[i + 1])
                    if asst_text.strip() and last_real_user is not None:
                        _u_text, _u_meta, _u_ts = last_real_user
                        pairs.append(Message(
                            role="user", content=_u_text,
                            metadata=_u_meta, timestamp=_u_ts,
                        ))
                        pairs.append(Message(
                            role="assistant", content=asst_text,
                            timestamp=_u_ts,
                        ))
                        last_real_user = None
                    i += 2
                    continue

                text, meta = self.extract_message_text_with_meta(chat_msgs[i])
                ts = extract_timestamp_from_metadata(meta) if meta else None
                asst_text = self.extract_message_text(chat_msgs[i + 1])

                if asst_text.strip():
                    # Normal pair: real user + assistant with text
                    pairs.append(Message(
                        role="user", content=text,
                        metadata=meta or None, timestamp=ts,
                    ))
                    pairs.append(Message(
                        role="assistant", content=asst_text,
                        timestamp=ts,
                    ))
                    # If assistant also has tool_use, hold the user for the
                    # next tool_result→assistant text response.
                    _asst_raw = chat_msgs[i + 1].get("content", "")
                    _has_tool_use = False
                    if isinstance(_asst_raw, list):
                        _has_tool_use = any(
                            isinstance(b, dict) and b.get("type") == "tool_use"
                            for b in _asst_raw
                        )
                    if _has_tool_use:
                        last_real_user = (text, meta or None, ts)
                    else:
                        last_real_user = None
                else:
                    # Assistant is tool_use only — hold user for next text response
                    last_real_user = (text, meta or None, ts)

                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self, body: dict) -> list[dict]:
        items = body.get("input", [])
        if isinstance(items, list):
            return items
        if isinstance(items, str):
            return [{"role": "user", "content": items}]
        return []

    def has_messages(self, body: dict) -> bool:
        inp = body.get("input")
        return isinstance(inp, (list, str)) and bool(inp)

    def group_into_turns(self, body: dict) -> list[TurnGroup]:
        items = body.get("input", [])
        if not isinstance(items, list):
            return []
        turns: list[TurnGroup] = []
        current_indices: list[int] = []
        has_tool = False

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            item_type = item.get("type", "")

            if role == "user":
                # Start of a new turn — flush the previous one
                if current_indices:
                    turns.append(TurnGroup(
                        indices=current_indices,
                        role="user",
                        has_tool_activity=has_tool,
                    ))
                current_indices = [i]
                has_tool = False
            elif item_type in ("function_call", "function_call_output"):
                # Bare tool items belong to the current turn
                if not current_indices:
                    current_indices = [i]
                else:
                    current_indices.append(i)
                has_tool = True
            elif item_type == "reasoning":
                # Reasoning items belong to the current turn
                if not current_indices:
                    current_indices = [i]
                else:
                    current_indices.append(i)
            else:
                # assistant or other role-based items
                if not current_indices:
                    current_indices = [i]
                else:
                    current_indices.append(i)

        # Flush the last group
        if current_indices:
            turns.append(TurnGroup(
                indices=current_indices,
                role="user",
                has_tool_activity=has_tool,
            ))

        return turns

    # -- OpenAI Responses-specific overrides ---------------------------------

    def _merge_message_content(self, target: dict, source: dict) -> None:
        """OpenAI Responses: combine content appropriately by role.

        User messages use plain text strings (or ``input_text`` blocks).
        Assistant messages use ``output_text`` block arrays.
        """
        role = target.get("role", "")
        tc = target.get("content", [])
        sc = source.get("content", [])
        if role == "user":
            # User messages: preserve all content types (text, images, etc.)
            if isinstance(tc, str):
                tc = [{"type": "input_text", "text": tc}] if tc else []
            if isinstance(sc, str):
                sc = [{"type": "input_text", "text": sc}] if sc else []
            target["content"] = list(tc) + list(sc)
        else:
            # Assistant messages: combine output_text block arrays.
            if isinstance(tc, str):
                tc = [{"type": "output_text", "text": tc}] if tc else []
            if isinstance(sc, str):
                sc = [{"type": "output_text", "text": sc}] if sc else []
            target["content"] = list(tc) + list(sc)

    def extract_text_from_item(self, body: dict, index: int) -> str:
        item = self.get_messages(body)[index]
        # Bare items (function_call, function_call_output, reasoning)
        if "output" in item:
            return str(item["output"])
        content = item.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return ""

    def iter_media_blocks(self, body: dict) -> Iterator["MediaBlockInfo"]:
        """OpenAI Responses: media in content arrays with input_image type."""
        messages = self.get_messages(body)
        for mi, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            role = msg.get("role", "")
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                # Check for OpenAI Responses input_image blocks first
                if block.get("type") == "input_image":
                    image_url = block.get("image_url", "")
                    media_type = "image"
                    if isinstance(image_url, str) and image_url.startswith("data:") and ";base64," in image_url:
                        header = image_url.split(";base64,", 1)[0]
                        media_type = header.replace("data:", "")
                    def _make_input_image_setter(_block):
                        def _setter(blk, b64, mt):
                            blk["image_url"] = f"data:{mt};base64,{b64}"
                        return _setter
                    def _make_input_image_replacer(_content, _bi, _role):
                        def _replace(text: str) -> None:
                            # Use input_text for user messages, output_text for assistant
                            block_type = "input_text" if _role == "user" else "output_text"
                            _content[_bi] = {"type": block_type, "text": text}
                        return _replace
                    yield MediaBlockInfo(
                        msg_index=mi,
                        block_index=bi,
                        media_type=media_type,
                        setter=_make_input_image_setter(block),
                        replace_with_text=_make_input_image_replacer(content, bi, role),
                        carrier=msg,
                    )
                    continue
                # Fall back to base class detection (Anthropic/OpenAI Chat shapes)
                media_type, setter = self._detect_media_block(block)
                if media_type is not None:
                    def _make_replacer(_content, _bi, _role):
                        def _replace(text: str) -> None:
                            block_type = "input_text" if _role == "user" else "output_text"
                            _content[_bi] = {"type": block_type, "text": text}
                        return _replace
                    yield MediaBlockInfo(
                        msg_index=mi,
                        block_index=bi,
                        media_type=media_type,
                        setter=setter,
                        replace_with_text=_make_replacer(content, bi, role),
                        carrier=msg,
                    )

    # -- Context injection --

    def inject_context(self, body: dict, prepend_text: str) -> dict:
        if not prepend_text:
            return body
        body = copy.deepcopy(body)
        context_block = f"<system-reminder>\n{prepend_text}\n</system-reminder>"
        # Append to the LAST user item in input (current turn) so the
        # conversation prefix stays byte-identical between turns, maximising
        # OpenAI prefix caching.
        items = body.get("input", [])
        if isinstance(items, list):
            for i in range(len(items) - 1, -1, -1):
                item = items[i]
                if not isinstance(item, dict) or item.get("role") != "user":
                    continue
                content = item.get("content", "")
                if isinstance(content, str):
                    items[i] = dict(item)
                    items[i]["content"] = f"{content}\n\n{context_block}"
                elif isinstance(content, list):
                    items[i] = dict(item)
                    items[i]["content"] = list(content) + [{"type": "input_text", "text": context_block}]
                break
            else:
                items.append({"role": "user", "content": context_block})
            body["input"] = items
        elif isinstance(items, str):
            # input is a plain string — append context
            body["input"] = f"{items}\n\n{context_block}"
        return body

    # -- Conversation markers --

    def extract_conversation_id(self, body: dict) -> str | None:
        # Search BACKWARD — the most recent assistant marker is authoritative.
        items = body.get("input", [])
        if not isinstance(items, list):
            return None
        for item in reversed(items):
            if not isinstance(item, dict) or item.get("role") != "assistant":
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                m = _VC_CONVERSATION_RE.search(content)
                if m:
                    return m.group(1)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text:
                            m = _VC_CONVERSATION_RE.search(text)
                            if m:
                                return m.group(1)
        return None

    def strip_conversation_markers(self, body: dict) -> dict:
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
                cleaned = _VC_CONVERSATION_RE.sub("", content).rstrip()
                if cleaned != content:
                    item = dict(item)
                    item["content"] = cleaned
                    modified = True
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        text = block["text"]
                        cleaned = _VC_CONVERSATION_RE.sub("", text).rstrip()
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

    def inject_conversation_marker(self, response_body: dict, marker: str) -> dict:
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

    def emit_conversation_marker_sse(self, conversation_id: str) -> bytes:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        marker_event = json.dumps({
            "type": "response.output_text.delta",
            "delta": marker,
        })
        return f"event: response.output_text.delta\ndata: {marker_event}\n\n".encode()

    def emit_fake_response_sse(self, text: str, conversation_id: str) -> bytes:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        full_text = text + marker
        events = []
        events.append(f'data: {json.dumps({"type": "response.output_text.delta", "delta": full_text})}\n\n')
        events.append(f'data: {json.dumps({"type": "response.output_text.done", "text": full_text})}\n\n')
        events.append(f'data: {json.dumps({"type": "response.completed"})}\n\n')
        return "".join(events).encode()

    def build_fake_response(self, text: str, conversation_id: str) -> dict:
        marker = f"\n<!-- vc:conversation={conversation_id} -->"
        return {
            "id": "resp_vcattach", "object": "response",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text + marker}]}],
            "status": "completed",
        }

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
            return self._count(instructions)
        return 0

    def _extract_media_from_block(self, block: dict) -> MediaBlock | None:
        """Responses: input_image blocks contain base64 in image_url data URI."""
        if block.get("type") == "input_image":
            url = block.get("image_url", "")
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                header, b64 = url.split(";base64,", 1)
                mt = header.replace("data:", "") or "image/unknown"
                return MediaBlock(b64, mt)
        # Fall back to base class (Anthropic source blocks, OpenAI Chat data URIs)
        return super()._extract_media_from_block(block)

    def _collect_media(self, body: dict) -> list[MediaBlock]:
        """Responses: recurse through items for media blocks."""
        results: list[MediaBlock] = []
        for item in self.get_messages(body):
            results.extend(self._collect_media_from_value(item))
        return results

    def estimate_message_tokens(self, msg: dict) -> int:
        """Responses: recurse through content for media-aware token counting."""
        return super().estimate_message_tokens(msg)

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
        existing_names = {t.get("name") for t in tools if isinstance(t, dict)}
        for td in tool_defs:
            if td["name"] not in existing_names:
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
        # Responses API (Codex) requires stream=true; continuation handler
        # collects the SSE stream and extracts the completed response.
        body["stream"] = True
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
    # String input without a messages key: Responses API single-turn shorthand.
    # Exclude embedding/moderation models which also use "input" as a string.
    if isinstance(body.get("input"), str) and "messages" not in body:
        model = str(body.get("model", "")).lower()
        if "embedding" not in model and "moderation" not in model:
            return _FORMAT_REGISTRY["openai_responses"]
    if "system" in body:
        return _FORMAT_REGISTRY["anthropic"]
    model = body.get("model", "")
    if isinstance(model, str) and model.startswith("claude"):
        return _FORMAT_REGISTRY["anthropic"]
    return _FORMAT_REGISTRY["openai"]


def get_format(name: str) -> PayloadFormat:
    return _FORMAT_REGISTRY[name]
