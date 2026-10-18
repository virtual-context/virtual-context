"""Compact raw request bodies before they are stored.

A raw body keeps the content parts the host sent with a turn. Hosts resend
recent images and their own rendering of the conversation on every request,
so stored bodies repeated the same base64 images and megabytes of replayed
context turn after turn.

Inline data (base64 images and files) moves to a blob store keyed by its
SHA-256 and the body keeps a ``vc-blob:sha256:<hex>`` reference in its place;
:func:`expand_raw_body` restores it. In a host-built prompt, the context
section between the adapter's leading metadata blocks and the request label
(replayed chat, assembled context, workspace files) is replaced by a short
labeled marker carrying its hash and length. Those sections are dropped from
the requester's text when a turn is normalized, so normalization of a
compacted body yields the same text and metadata as the original.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from ..proxy._envelope import (
    _HOST_CTX_JSON_BLOCK_RE,
    _HOST_REQUEST_LINE_RE,
    _is_host_carried_prompt,
)

BLOB_REF_PREFIX = "vc-blob:sha256:"
BLOB_REF_RE = re.compile(r"vc-blob:sha256:([0-9a-f]{64})")
HOST_CONTEXT_OMITTED_LABEL = "VC omitted host context"
# Values shorter than this stay inline: a reference would not save anything.
MIN_BLOB_CHARS = 1024
MIN_OMITTED_CONTEXT_CHARS = 4096


@dataclass
class CompactedRawBody:
    text: str | None
    blobs: dict[str, tuple[str, str]] = field(default_factory=dict)  # hash -> (media_type, data)


def compact_raw_body(raw: str | None) -> CompactedRawBody:
    """The stored form of *raw* plus the blobs it now references."""
    if not raw:
        return CompactedRawBody(raw)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        compacted = _omit_host_context(raw)
        return CompactedRawBody(compacted)
    blobs: dict[str, tuple[str, str]] = {}
    compacted = _compact_value(parsed, blobs)
    if compacted == parsed:
        return CompactedRawBody(raw)
    return CompactedRawBody(json.dumps(compacted), blobs)


def blob_refs(raw: str | None) -> set[str]:
    """Hashes of the blobs a stored body references."""
    return set(BLOB_REF_RE.findall(raw or ""))


def expand_raw_body(
    raw: str | None,
    load_blobs: Callable[[set[str]], dict[str, tuple[str, str]]],
) -> str | None:
    """*raw* with every blob reference replaced by its data.

    A reference whose blob cannot be loaded is left in place.
    """
    hashes = blob_refs(raw)
    if not hashes:
        return raw
    blobs = load_blobs(hashes)
    parsed = json.loads(raw)

    def expand(value):
        if isinstance(value, str) and value.startswith(BLOB_REF_PREFIX):
            stored = blobs.get(value[len(BLOB_REF_PREFIX):])
            return stored[1] if stored else value
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return json.dumps(expand(parsed))


def _compact_value(value, blobs: dict):
    if isinstance(value, dict):
        return {key: _compact_entry(key, item, value, blobs) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_value(item, blobs) for item in value]
    if isinstance(value, str):
        return _omit_host_context(value)
    return value


def _compact_entry(key: str, item, owner: dict, blobs: dict):
    if isinstance(item, str) and len(item) >= MIN_BLOB_CHARS:
        if item.startswith("data:"):
            media_type = item[5:].split(";", 1)[0].split(",", 1)[0]
            return _store_blob(item, media_type, blobs)
        if key in ("data", "file_data"):
            return _store_blob(item, str(owner.get("media_type") or ""), blobs)
    if isinstance(item, str) and key == "text":
        return _omit_host_context(item)
    return _compact_value(item, blobs)


def _store_blob(data: str, media_type: str, blobs: dict) -> str:
    digest = hashlib.sha256(data.encode()).hexdigest()
    blobs[digest] = (media_type, data)
    return BLOB_REF_PREFIX + digest


def _omit_host_context(text: str) -> str:
    """Replace a host prompt's replayed context section with a marker."""
    if "⟦openclaw:ctx⟧" not in text or not _is_host_carried_prompt(text):
        return text
    labels = list(_HOST_REQUEST_LINE_RE.finditer(text))
    request_start = labels[-1].start()
    head = text[:request_start]
    position = 0
    while True:
        match = _HOST_CTX_JSON_BLOCK_RE.match(head, position)
        if match is None:
            break
        position = match.end()
        while position < len(head) and head[position] == "\n":
            position += 1
    omitted = head[position:]
    if len(omitted) < MIN_OMITTED_CONTEXT_CHARS:
        return text
    digest = hashlib.sha256(omitted.encode()).hexdigest()
    marker = (
        f"{HOST_CONTEXT_OMITTED_LABEL}: ⟦openclaw:ctx⟧\n"
        f"sha256={digest} chars={len(omitted)}\n\n"
    )
    return head[:position] + marker + text[request_start:]
