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
# Accepts both legacy "vc:session" and current "vc:conversation".
# Ids are either bare UUIDs or `sk:`-namespace stable identities
# (caller-asserted, passed verbatim by resolve_conversation_id).
_VC_CONVERSATION_RE = re.compile(
    r"<!-- vc:(?:session|conversation)=(sk:[^\s<>]+|[a-f0-9-]+) -->"
)

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

# The same block shape, anchored to the END of the message. Adapters emit the
# canonical reply-target block as a trailing suffix, which the leading loop
# above cannot see, so it survives inside the requester's text today.
#
# This is deliberately NOT a whole-message scanner: broadening
# ``_METADATA_BLOCK_RE`` into one would reinterpret an ordinary user code
# fence in the middle of a message as engine metadata. Only the terminal
# position is parsed, and only for the accepted reply labels.
_TERMINAL_METADATA_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*([A-Z][^\n(]{0,80})\s*\(([^\)]*)\)\s*:\s*\n"
    r"```(?:json)?\s*\n"
    r"(\{[^`]*?\}|\[[^`]*?\])\s*\n"
    r"```[ \t]*\n?\s*$",
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


def _claim_actor_identity(metadata: dict, label: str, parsed: object) -> None:
    """Retain the FIRST syntactically identity-bearing block, across labels.

    The strip loop keeps consuming blocks from the front of the message, and
    ``metadata[label]`` is last-wins. A member whose own text begins with a
    well-formed ``Conversation info`` block naming somebody else's
    ``sender_id`` would therefore overwrite the adapter's block and repoint
    attribution at that member — cross-user contamination reachable from a
    chat message.

    The first candidate claims this reserved slot and no later block can take
    it, even when normalizing the winner ultimately yields no actor id. That
    fail-closed rule is the point: skipping a malformed first adapter block
    and accepting a later user-authored one would restore the spoof.

    Only identity is first-wins. Every other label keeps its shipped
    last-wins merge semantics, so sender and channel derivation are unchanged.
    """
    from ..types import (
        ACTOR_IDENTITY_KEY,
        is_actor_identity_block,
        is_conversation_info_identity_block,
    )

    if ACTOR_IDENTITY_KEY in metadata:
        return
    if label == "actor" and is_actor_identity_block(parsed):
        metadata[ACTOR_IDENTITY_KEY] = {"source": "actor", "value": parsed}
    elif label == "conversation info" and is_conversation_info_identity_block(parsed):
        metadata[ACTOR_IDENTITY_KEY] = {
            "source": "conversation info", "value": parsed,
        }


def _claim_current_conversation(metadata: dict, label: str, parsed: object) -> None:
    """Retain the FIRST ``conversation info`` block, ordered, not last-wins.

    Kept even when an explicit ``Actor`` block already claimed requester
    identity, because this block carries different things: the current
    ``message_id`` / ``reply_to_id`` and the audience channel.

    The audience channel is a privacy boundary, so it cannot be read from the
    last-wins merged dict — a member's own typed block would then choose the
    channel their memory is filtered against. The merged
    ``metadata["conversation info"]`` stays for backward-compatible display.
    """
    from ..types import CURRENT_CONVERSATION_KEY

    if label != "conversation info" or not isinstance(parsed, dict):
        return
    if CURRENT_CONVERSATION_KEY in metadata:
        return
    metadata[CURRENT_CONVERSATION_KEY] = parsed


def _reply_identity_tuple(value: dict) -> tuple:
    """Normalized comparison key for two candidate reply blocks."""
    def _clean(key: str) -> str:
        raw = value.get(key)
        return raw.strip() if isinstance(raw, str) else ""

    return (
        _clean("sender_id"),
        _clean("sender_label") or _clean("name"),
        _clean("body"),
        _clean("message_id"),
        _clean("platform").lower(),
    )


def _claim_reply_subject(
    metadata: dict, label: str, parsed: object, edge: str,
) -> None:
    """Retain the FIRST subject-bearing reply block, from either outer edge.

    Same discipline as requester identity, adapted to the two verified
    envelope positions: whichever outer edge the adapter owns is consumed
    before the user text adjacent to it.

    The two rejection modes are deliberately different, because the threats
    are different:

    * **Same edge, later candidate.** This is an inner block sitting against
      the adapter's outer one — i.e. text the member typed. It simply cannot
      replace the claim. Treating it as a contradiction would let any member
      erase their own reply's attribution by typing a block.
    * **Opposite edge, disagreeing candidate.** A deployment that emits both a
      leading and a trailing reply block must have them agree. Disagreement is
      not resolved by precedence — picking one of two contradictory actors is
      picking by coin flip — so it fails closed.
    """
    from ..types import REPLY_SUBJECT_KEY, REPLY_SUBJECT_LABELS, is_reply_subject_block

    if label not in REPLY_SUBJECT_LABELS:
        return

    existing = metadata.get(REPLY_SUBJECT_KEY)
    if isinstance(existing, dict):
        if existing.get("conflict") or existing.get("edge") == edge:
            return
        previous = existing.get("value")
        if isinstance(previous, dict) and isinstance(parsed, dict):
            if _reply_identity_tuple(previous) != _reply_identity_tuple(parsed):
                metadata[REPLY_SUBJECT_KEY] = {"conflict": True}
        return

    if is_reply_subject_block(parsed):
        metadata[REPLY_SUBJECT_KEY] = {"source": label, "edge": edge, "value": parsed}
    else:
        # A syntactically-empty, malformed, or non-dict first candidate still
        # claims the slot: letting a later block repair it with
        # attacker-controlled data is the spoof.
        metadata[REPLY_SUBJECT_KEY] = {"source": label, "edge": edge, "value": None}


def _strip_terminal_reply_blocks(text: str, metadata: dict) -> str:
    """Consume accepted reply blocks from the END of the message, outermost first.

    The outermost trailing block is the adapter's, so it is the first
    candidate seen and it claims the subject slot. An inner duplicate the
    member typed themselves is stripped but can never repoint attribution.

    The loop stops at the first terminal block whose label is not an accepted
    reply label, so ordinary trailing user content — including a JSON code
    fence — is never eaten.
    """
    from ..types import REPLY_SUBJECT_LABELS

    while True:
        m = _TERMINAL_METADATA_BLOCK_RE.search(text)
        if not m:
            return text
        label = m.group(1).strip().lower()
        if label not in REPLY_SUBJECT_LABELS:
            return text
        try:
            parsed = json.loads(m.group(3))
        except (json.JSONDecodeError, ValueError):
            # Malformed, but it IS an accepted reply label in the adapter's
            # position: strip it from requester content and fail closed.
            parsed = None
        if isinstance(parsed, dict) or parsed is None:
            _claim_reply_subject(metadata, label, parsed, edge="trailing")
        text = text[:m.start()]


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
                    _claim_actor_identity(metadata, label, parsed)
                    _claim_current_conversation(metadata, label, parsed)
                    _claim_reply_subject(metadata, label, parsed, edge="leading")
                else:
                    # A scalar under a reply label is not a subject, but it IS
                    # a candidate at the adapter's edge, so it must still claim
                    # the slot and fail closed.
                    _claim_reply_subject(metadata, label, None, edge="leading")
            except (json.JSONDecodeError, ValueError):
                # Malformed JSON — skip metadata but still strip the block.
                # A malformed block at an accepted reply label still CLAIMS the
                # subject slot: skipping it and letting the next candidate win
                # would let a member repair the adapter's block with their own
                # and repoint the quoted claim at somebody else.
                _claim_reply_subject(metadata, label, None, edge="leading")
            text = stripped[m.end():]
        else:
            text = stripped
            break

    # Consume the trailing reply-target envelope. The leading loop above
    # cannot see it, so without this it stays inside the requester's content
    # and a later distiller reads the quoted person's claim as the requester's
    # own. The body is retained separately, never appended back to the text.
    text = _strip_terminal_reply_blocks(text, metadata)

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
