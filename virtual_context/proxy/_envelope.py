"""Shared constants and helpers for OpenClaw envelope stripping.

Leaf module with no intra-package imports — safe for both formats.py
and helpers.py to depend on without circular import risk.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VC_PROMPT_MARKER = "[vc:prompt]\n"
# MemOS preamble: starts with "# Role", ends with this delimiter line (zero-width spaces)
_MEMOS_QUERY_DELIM = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b"

# Conversation marker: injected into assistant responses, extracted from inbound history
# Accepts both legacy "vc:session" and current "vc:conversation" for backward compat
_VC_CONVERSATION_RE = re.compile(r"<!-- vc:(?:session|conversation)=([a-f0-9-]+) -->")

# OpenClaw envelope patterns — consistent across all channels
_VC_USER_RE = re.compile(r"^\[vc:user\](.*?)\[/vc:user\]", re.DOTALL)
_SYSTEM_EVENT_RE = re.compile(r"^(?:System:\s*\[[^\]]*\][^\n]*\n+)+")
_CHANNEL_HEADER_RE = re.compile(r"^\[[A-Z][a-zA-Z]*\s[^\]]*\bid:-?\d+\b[^\]]*\]\s*")
_MESSAGE_ID_RE = re.compile(r"\n?\[message_id:\s*\d+\]\s*$")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _last_text_block(content: list) -> str:
    """Return the text of the last ``type: "text"`` block in *content*."""
    for block in reversed(content):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def _strip_vc_prompt(text: str) -> str:
    if text.startswith(_VC_PROMPT_MARKER):
        return text[len(_VC_PROMPT_MARKER):]
    return text


def _strip_openclaw_envelope(text: str) -> str:
    """Strip OpenClaw channel metadata from a message.

    Handles (in order):

    1. ``[vc:prompt]`` marker from the virtual-context-tagger plugin
    2. MemOS "# Role" preamble — strips everything before the query delimiter
    3. ``[vc:user]...[/vc:user]`` backward-compatible wrapper (extracts
       inner content and returns immediately — inner content is already clean)
    4. ``System: [TIMESTAMP] event`` lines prepended by OpenClaw
    5. ``[ChannelName ... id:NNN ...] `` header (Telegram, WhatsApp, etc.)
    6. ``[message_id: NNN]`` footer

    Returns the actual conversational content with all metadata removed.
    """
    if not text:
        return text

    # 1. Strip [vc:prompt] marker and any trailing whitespace
    if text.startswith(_VC_PROMPT_MARKER):
        text = text[len(_VC_PROMPT_MARKER):].lstrip()

    # 1b. Strip MemOS preamble
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
