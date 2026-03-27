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

_SYSTEM_JSON_BLOCK_RE = re.compile(
    r"```json\s*\n(.*?)\n```",
    re.DOTALL,
)

_VC_PROMPT_MARKER = "[vc:prompt]"


def _candidate_to_uuid(layer: str, value: str) -> str:
    raw = f"{layer}:{value}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return str(_uuid.UUID(digest[:32]))


def _get_messages(body: dict) -> list:
    """Return the message list from any format (messages, input, or contents)."""
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    inp = body.get("input")
    if isinstance(inp, list) and inp:
        return inp
    return []


def _extract_conversation_info(body: dict) -> dict:
    if not body:
        return {}

    for msg in _get_messages(body):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
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


def _extract_system_prompt_info(body: dict) -> dict:
    """Extract identity fields from JSON code blocks in the system prompt.

    Handles Anthropic (top-level ``system`` as str or list of text blocks)
    and OpenAI (first message with ``role`` in ``("system", "developer")``).
    Scans all ` ```json ` fenced blocks, parses each as JSON, and returns
    the first dict containing ``chat_id``, ``conversation_label``, or
    ``account_id``.
    """
    if not body:
        return {}

    system_text = ""

    # Anthropic: top-level "system" key
    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = "\n".join(
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )

    # OpenAI Chat: first message with role "system" or "developer"
    if not system_text:
        for msg in _get_messages(body):
            role = msg.get("role", "")
            if role in ("system", "developer"):
                c = msg.get("content", "")
                if isinstance(c, str):
                    system_text = c
                elif isinstance(c, list):
                    system_text = "\n".join(
                        b.get("text", "") for b in c
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                break

    # OpenAI Responses: "instructions" key
    if not system_text:
        instructions = body.get("instructions", "")
        if isinstance(instructions, str) and instructions:
            system_text = instructions

    if not system_text:
        return {}

    result: dict = {}
    identity_keys = {"chat_id", "conversation_label", "account_id"}
    for m in _SYSTEM_JSON_BLOCK_RE.finditer(system_text):
        try:
            parsed = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        for key in identity_keys:
            if key in parsed and isinstance(parsed[key], str) and parsed[key]:
                result[key] = parsed[key]
        if result:
            return result

    return {}


def _extract_system_prompt_hash(body: dict) -> str:
    if not body:
        return ""

    system_text = ""
    system_list: list | None = None

    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_list = system
            system_text = "".join(
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )

    if not system_text and not system_list:
        # No top-level system field — check OpenAI-style system/developer message
        for msg in _get_messages(body):
            role = msg.get("role", "")
            if role in ("system", "developer"):
                c = msg.get("content", "")
                if isinstance(c, str):
                    system_text = c
                elif isinstance(c, list):
                    system_list = c
                    system_text = "".join(
                        b.get("text", "") for b in c
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                break

    # OpenAI Responses: "instructions" key
    if not system_text and not system_list:
        instructions = body.get("instructions", "")
        if isinstance(instructions, str) and instructions:
            system_text = instructions

    # No extractable text — don't hash. A deterministic hash-of-nothing
    # would collapse unrelated requests (multimodal-only system, empty list)
    # into the same conversation ID.
    if not system_text:
        return ""

    # Claude Code special path: the billing header (block 0) contains a
    # per-request `cch=` value that changes every turn. Strip it and mix
    # in the first few user messages to get a stable per-session hash.
    if _is_claude_code_system(system, system_list):
        return _claude_code_hash(system, system_list, body)

    h = hashlib.sha256()
    h.update(system_text.encode())
    return h.hexdigest()


_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def _is_claude_code_system(
    system: object,
    system_list: list | None,
) -> bool:
    """Detect Claude Code by the billing header in the first system block."""
    if isinstance(system, list) and system:
        first = system[0]
        if isinstance(first, dict) and first.get("type") == "text":
            text = first.get("text", "")
            if text.startswith(_BILLING_HEADER_PREFIX):
                return True
    if isinstance(system, str) and system.startswith(_BILLING_HEADER_PREFIX):
        return True
    return False


def _claude_code_hash(
    system: object,
    system_list: list | None,
    body: dict,
) -> str:
    """Hash stable system blocks + first user messages for Claude Code sessions.

    Skips the billing header block (contains per-request ``cch=`` value)
    and mixes in the first 3 user message texts to disambiguate sessions
    in the same project.
    """
    h = hashlib.sha256()

    # Hash system blocks, skipping any that start with the billing header
    if isinstance(system, list):
        for block in system:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "")
            if text.startswith(_BILLING_HEADER_PREFIX):
                continue
            h.update(text.encode())
    elif isinstance(system, str):
        # Strip the billing header line from a plain string system prompt
        lines = system.split("\n")
        for line in lines:
            if line.startswith(_BILLING_HEADER_PREFIX):
                continue
            h.update(line.encode())

    # Mix in the first 3 user messages for per-session uniqueness
    user_count = 0
    for msg in _get_messages(body):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                    content = block.get("text", "")
                    break
            else:
                continue
        if isinstance(content, str) and content.strip():
            h.update(content[:500].encode())  # cap per-message to avoid huge hashes
            user_count += 1
            if user_count >= 3:
                break

    return h.hexdigest()


def resolve_conversation_id(
    body: dict | None,
    explicit_id: str | None = None,
    format_name: str = "",
) -> str:
    """Resolve a deterministic conversation ID from request body signals.

    Checks identity signals in priority order:
    1. explicit_id (caller-provided, e.g. from query string)
    2. conversation_label (from envelope metadata)
    3. chat_id (from envelope metadata)
    4. system_prompt hash (Anthropic or OpenAI format)

    When *format_name* is provided, it is included in the hash so that the
    same conversation on different provider formats (e.g. Anthropic vs OpenAI
    Responses) produces different IDs.  This prevents format-specific engine
    state (turn hashes, compaction watermark) from being applied to a
    different format's payload.

    The body dict is read by reference — no copies made.
    Special case: if explicit_id is already a valid UUID AND no format_name,
    returned as-is.  With format_name, it is re-hashed to include the format.
    Returns a random UUID if no signals are found.
    """
    if explicit_id and explicit_id.strip():
        explicit_id = explicit_id.strip()
        if format_name:
            return _candidate_to_uuid("explicit_id", f"{format_name}:{explicit_id}")
        try:
            _uuid.UUID(explicit_id)
            return explicit_id
        except ValueError:
            return _candidate_to_uuid("explicit_id", explicit_id)

    if not body:
        return str(_uuid.uuid4())

    _fmt_suffix = f":{format_name}" if format_name else ""

    conv_info = _extract_conversation_info(body)
    if conv_info.get("conversation_label"):
        return _candidate_to_uuid("conversation_label", conv_info["conversation_label"] + _fmt_suffix)
    if conv_info.get("chat_id"):
        return _candidate_to_uuid("chat_id", conv_info["chat_id"] + _fmt_suffix)

    # Also check system prompt for identity fields (e.g. OpenClaw embeds
    # chat_id in a ```json block inside the system prompt).  This is checked
    # BEFORE the system prompt hash so that a stable chat_id wins over an
    # unstable hash caused by dynamic prompt sections.
    sys_info = _extract_system_prompt_info(body)
    if sys_info.get("conversation_label"):
        return _candidate_to_uuid("conversation_label", sys_info["conversation_label"] + _fmt_suffix)
    if sys_info.get("chat_id"):
        return _candidate_to_uuid("chat_id", sys_info["chat_id"] + _fmt_suffix)

    sys_hash = _extract_system_prompt_hash(body)
    if sys_hash:
        return _candidate_to_uuid("system_prompt", sys_hash + _fmt_suffix)

    return str(_uuid.uuid4())
