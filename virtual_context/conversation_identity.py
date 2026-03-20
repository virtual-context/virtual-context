"""Conversation identity resolver: deterministic UUID from request body.

Walks the request body to find identity signals in priority order:
1. explicit_id (passed separately by caller)
2. conversation_label (from envelope metadata in first user message)
3. chat_id (from envelope metadata)
4. system_prompt (Anthropic top-level or OpenAI system message)

The body dict is received by reference — no copies, no intermediate dicts.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid as _uuid


_METADATA_BLOCK_RE = re.compile(
    r"^([A-Z][^\n(]{0,80})\s*\(([^\)]*)\)\s*:\s*\n"
    r"```(?:json)?\s*\n"
    r"(\{[^`]*?\}|\[[^`]*?\])\s*\n"
    r"```\s*\n?",
    re.DOTALL,
)

_VC_PROMPT_MARKER = "[vc:prompt]"


def _candidate_to_uuid(layer: str, value: str) -> str:
    raw = f"{layer}:{value}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return str(_uuid.UUID(digest[:32]))


def _extract_conversation_info(body: dict) -> dict:
    if not body:
        return {}

    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content = block.get("text", "")
                    break
            else:
                continue
        if not isinstance(content, str):
            continue

        text = content.lstrip()
        if text.startswith(_VC_PROMPT_MARKER):
            text = text[len(_VC_PROMPT_MARKER):].lstrip()

        while True:
            stripped = text.lstrip()
            m = _METADATA_BLOCK_RE.match(stripped)
            if m:
                label = m.group(1).strip().lower()
                if label == "conversation info":
                    try:
                        parsed = json.loads(m.group(3))
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                text = stripped[m.end():]
            else:
                break
        break

    return {}


def _extract_system_prompt_hash(body: dict) -> str:
    if not body:
        return ""

    system = body.get("system")
    if system is not None:
        h = hashlib.sha256()
        if isinstance(system, str):
            h.update(system.encode())
            return h.hexdigest()
        if isinstance(system, list):
            for b in system:
                if isinstance(b, dict) and b.get("type") == "text":
                    h.update(b.get("text", "").encode())
            return h.hexdigest()

    for msg in body.get("messages", []):
        role = msg.get("role", "")
        if role in ("system", "developer"):
            c = msg.get("content", "")
            h = hashlib.sha256()
            if isinstance(c, str):
                h.update(c.encode())
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        h.update(b.get("text", "").encode())
            return h.hexdigest()

    return ""


def resolve_conversation_id(
    body: dict | None,
    explicit_id: str | None = None,
) -> str:
    """Resolve a deterministic conversation ID from request body signals.

    Checks identity signals in priority order:
    1. explicit_id (caller-provided, e.g. from query string)
    2. conversation_label (from envelope metadata)
    3. chat_id (from envelope metadata)
    4. system_prompt hash (Anthropic or OpenAI format)

    The body dict is read by reference — no copies made.
    Special case: if explicit_id is already a valid UUID, returned as-is.
    Returns a random UUID if no signals are found.
    """
    if explicit_id and explicit_id.strip():
        explicit_id = explicit_id.strip()
        try:
            _uuid.UUID(explicit_id)
            return explicit_id
        except ValueError:
            return _candidate_to_uuid("explicit_id", explicit_id)

    if not body:
        return str(_uuid.uuid4())

    conv_info = _extract_conversation_info(body)
    if conv_info.get("conversation_label"):
        return _candidate_to_uuid("conversation_label", conv_info["conversation_label"])
    if conv_info.get("chat_id"):
        return _candidate_to_uuid("chat_id", conv_info["chat_id"])

    sys_hash = _extract_system_prompt_hash(body)
    if sys_hash:
        return _candidate_to_uuid("system_prompt", sys_hash)

    return str(_uuid.uuid4())
