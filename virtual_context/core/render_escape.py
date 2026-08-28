"""Escape host-attribution lookalikes at model-facing render boundaries.

Stored conversation content is exact-source: a member who types a
host-attribution wrapper lookalike keeps it verbatim in storage. Hosts
that assemble model payloads mark a small set of wrappers as trusted
attribution metadata, so member text that imitates one must not reach
the model unescaped through rendered context (summaries, transcripts,
excerpts, tool results) — otherwise one participant can forge another's
attribution.

The escape replaces the leading ``<`` of a recognized tag with the six
literal characters ``\\u003c``, leaving the rest of the text intact. It
is idempotent by construction (the escaped form no longer matches) and
consumers that render the text again may safely apply the same escape.

Apply it ONLY at model-facing render egress. Never at ingest, never in
storage, never in dashboards or exports that are not model-facing, and
never in the message lane: canonical replay content must stay
byte-exact because turn-hash alignment depends on it.
"""
from __future__ import annotations

import re

# Wrappers that hosts treat as trusted attribution metadata. Longest
# first where one name prefixes another so a match consumes the full
# tag name.
HOST_ATTRIBUTION_TAGS: tuple[str, ...] = (
    "message-speaker",
    "current-speaker-reminder",
    "current-speaker",
    "current-reply-target",
    "vc-prepared-context",
)

_TAG_ALTERNATION = "|".join(HOST_ATTRIBUTION_TAGS)

# Matches only the leading "<" of an opening or closing host tag; the
# replacement rewrites that single character, so tag body, attributes,
# and surrounding text survive untouched.
_HOST_TAG_OPEN = re.compile(
    rf"<(?=/?(?:{_TAG_ALTERNATION})\b)",
    re.IGNORECASE,
)


def escape_host_attribution_markup(text: str) -> str:
    """Escape host-attribution lookalikes in model-facing plain text."""
    if not text or "<" not in text:
        return text
    return _HOST_TAG_OPEN.sub("\\\\u003c", text)


def escape_host_attribution_in_serialized_json(serialized: str) -> str:
    """Escape lookalikes inside an already JSON-serialized string.

    The serialized form needs a doubled backslash so that decoding the
    JSON yields content carrying the literal ``\\u003c`` sequence; a
    single backslash would decode straight back to ``<`` and undo the
    escape for any consumer that re-parses the result.
    """
    if not serialized or "<" not in serialized:
        return serialized
    return _HOST_TAG_OPEN.sub("\\\\\\\\u003c", serialized)
