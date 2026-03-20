"""Shared constants and helpers for message envelope stripping.

Leaf module with no intra-package imports — safe for both formats.py
and helpers.py to depend on without circular import risk.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VC_PROMPT_MARKER = "[vc:prompt]\n"
# MemOS preamble: starts with "# Role", ends with this delimiter line (zero-width spaces)
_MEMOS_QUERY_DELIM = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b"

# Conversation marker: injected into assistant responses, extracted from inbound history
# Accepts both legacy "vc:session" and current "vc:conversation"
_VC_CONVERSATION_RE = re.compile(r"<!-- vc:(?:session|conversation)=([a-f0-9-]+) -->")

# OpenClaw envelope patterns — consistent across all channels
_VC_USER_RE = re.compile(r"^\[vc:user\](.*?)\[/vc:user\]", re.DOTALL)
_SYSTEM_EVENT_RE = re.compile(r"^(?:System:\s*\[[^\]]*\][^\n]*\n+)+")
_CHANNEL_HEADER_RE = re.compile(r"^\[[A-Z][a-zA-Z]*\s[^\]]*\bid:-?\d+\b[^\]]*\]\s*")
_MESSAGE_ID_RE = re.compile(r"\n?\[message_id:\s*\d+\]\s*$")

# Labeled metadata block: "Label (qualifier):\n```json\n{...}\n```"
# Matches any client that wraps structured metadata in labeled fenced JSON.
# Only matches at string start (after lstrip) — won't eat user code fences.
_METADATA_BLOCK_RE = re.compile(
    r"^([A-Z][^\n(]{0,80})\s*\(([^\)]*)\)\s*:\s*\n"  # "Label (qualifier):\n"
    r"```(?:json)?\s*\n"                               # opening fence
    r"(\{[^`]*?\}|\[[^`]*?\])\s*\n"                    # JSON object or array
    r"```\s*\n?",                                       # closing fence
    re.DOTALL,
)


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


def _extract_envelope_metadata(text: str) -> tuple[str, dict]:
    """Strip envelope metadata and return both clean text and parsed metadata.

    Handles (in order):

    1. ``[vc:prompt]`` marker from the virtual-context-tagger plugin
    2. MemOS "# Role" preamble — strips everything before the query delimiter
    3. ``[vc:user]...[/vc:user]`` legacy wrapper (extracts
       inner content and returns immediately — inner content is already clean)
    4. Labeled metadata blocks — ``Label (qualifier):\\n```json\\n{...}\\n```\\n``
       Strips any number of fenced JSON blocks with labeled headers from the
       front of the message. These are injected by various clients (OpenClaw,
       Telegram adapters, etc.) and contain routing/sender metadata.
       Each block's parsed JSON is collected into the metadata dict, keyed by
       the lowercase label (e.g. ``"sender"``, ``"conversation info"``).
    5. ``System: [TIMESTAMP] event`` lines prepended by OpenClaw
    6. ``[ChannelName ... id:NNN ...] `` header (Telegram, WhatsApp, etc.)
    7. ``[message_id: NNN]`` footer

    Returns:
        A tuple of (stripped_text, metadata_dict).  The metadata dict maps
        lowercase labels to parsed JSON objects.  If no metadata blocks are
        found (or JSON is malformed), the dict is empty.
    """
    metadata: dict = {}

    if not text:
        return text, metadata

    # Strip [vc:prompt] marker and any trailing whitespace
    if text.startswith(_VC_PROMPT_MARKER):
        text = text[len(_VC_PROMPT_MARKER):].lstrip()

    # Strip MemOS preamble
    if text.startswith("# Role"):
        idx = text.find(_MEMOS_QUERY_DELIM)
        if idx != -1:
            text = text[idx + len(_MEMOS_QUERY_DELIM):].lstrip()

    # Handle [vc:user]...[/vc:user] — inner content is already clean
    m = _VC_USER_RE.match(text)
    if m:
        return m.group(1).strip(), metadata

    # Strip labeled metadata blocks from the front
    while True:
        stripped = text.lstrip()
        m = _METADATA_BLOCK_RE.match(stripped)
        if m:
            label = m.group(1).strip().lower()
            json_body = m.group(3)
            try:
                parsed = json.loads(json_body)
                if isinstance(parsed, (dict, list)):
                    metadata[label] = parsed
            except (json.JSONDecodeError, ValueError):
                pass  # malformed JSON — skip metadata but still strip the block
            text = stripped[m.end():]
        else:
            text = stripped
            break

    # Strip System: [...] event lines
    text = _SYSTEM_EVENT_RE.sub("", text)

    # Strip channel header  [ChannelName ... id:NNN ...]
    text = _CHANNEL_HEADER_RE.sub("", text)

    # Strip [message_id: NNN] footer
    text = _MESSAGE_ID_RE.sub("", text)

    return text.strip(), metadata


def parse_envelope_timestamp(ts_str: str) -> "datetime | None":
    """Parse an envelope timestamp like 'Tue 2026-03-17 00:35 EDT' into a datetime.

    Handles formats:
    - 'Day YYYY-MM-DD HH:MM TZ' (e.g., 'Tue 2026-03-17 00:35 EDT')
    - ISO 8601 (e.g., '2026-03-17T00:35:00Z')
    - 'YYYY-MM-DD HH:MM:SS' (no timezone)

    Returns None if parsing fails.
    """
    from datetime import datetime, timezone, timedelta

    if not ts_str or not isinstance(ts_str, str):
        return None
    ts_str = ts_str.strip()

    # Common US timezone offsets
    _tz_offsets = {
        "EDT": timedelta(hours=-4), "EST": timedelta(hours=-5),
        "CDT": timedelta(hours=-5), "CST": timedelta(hours=-6),
        "MDT": timedelta(hours=-6), "MST": timedelta(hours=-7),
        "PDT": timedelta(hours=-7), "PST": timedelta(hours=-8),
        "UTC": timedelta(hours=0), "GMT": timedelta(hours=0),
    }

    # Format: "Day YYYY-MM-DD HH:MM TZ"
    m = re.match(r"[A-Za-z]+\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(\w+)", ts_str)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M")
            tz_name = m.group(3).upper()
            offset = _tz_offsets.get(tz_name, timedelta(hours=0))
            return dt.replace(tzinfo=timezone(offset))
        except ValueError:
            pass

    # ISO 8601
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Plain datetime
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    return None


def extract_timestamp_from_metadata(metadata: dict | None) -> "datetime | None":
    """Extract and parse a timestamp from envelope metadata.

    Checks 'conversation info' block for a 'timestamp' field.
    """
    if not metadata:
        return None
    conv_info = metadata.get("conversation info")
    if not conv_info or not isinstance(conv_info, dict):
        return None
    ts_str = conv_info.get("timestamp")
    if ts_str:
        return parse_envelope_timestamp(ts_str)
    return None


def _strip_envelope(text: str) -> str:
    """Strip message envelope metadata to extract actual conversational content.

    Thin wrapper around :func:`_extract_envelope_metadata` that discards the
    metadata dict and returns only the cleaned text.  All 25+ existing callers
    are unaffected.
    """
    stripped, _ = _extract_envelope_metadata(text)
    return stripped


# Legacy alias — remove when OpenClaw callers are updated
_strip_openclaw_envelope = _strip_envelope
