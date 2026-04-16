"""Helpers for canonical turn identity, normalization, and anchor matching."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..types import CanonicalTurnRow, IngestBatchRecord

HASH_VERSION = 1
_WS_RE = re.compile(r"\s+")
_MEDIA_RE = re.compile(r"\[media attached:[^\]]+\]", re.IGNORECASE)
_SESSION_RE = re.compile(r"\[Session from [^\]]+\]", re.IGNORECASE)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_turn_text(text: str | None) -> str:
    text = str(text or "")
    text = _SESSION_RE.sub("", text)
    text = _MEDIA_RE.sub("[media attached]", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _WS_RE.sub(" ", text)
    return text.strip()


def compute_turn_hash(normalized_user: str, normalized_assistant: str, *, version: int = HASH_VERSION) -> str:
    payload = bytes([version]) + normalized_user.encode("utf-8") + b"\n---\n" + normalized_assistant.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_turn_hash_from_raw(user_text: str, assistant_text: str, *, version: int = HASH_VERSION) -> tuple[str, str, str]:
    normalized_user = normalize_turn_text(user_text)
    normalized_assistant = normalize_turn_text(assistant_text)
    return (
        compute_turn_hash(normalized_user, normalized_assistant, version=version),
        normalized_user,
        normalized_assistant,
    )


def compute_anchor_hash(rows: list[CanonicalTurnRow], start: int, size: int) -> str:
    digest = hashlib.sha256()
    digest.update(bytes([HASH_VERSION]))
    for row in rows[start:start + size]:
        digest.update((row.turn_hash or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_anchor_index(rows: list[CanonicalTurnRow], window_size: int) -> dict[str, list[int]]:
    anchors: dict[str, list[int]] = {}
    if window_size <= 0 or len(rows) < window_size:
        return anchors
    for start in range(0, len(rows) - window_size + 1):
        digest = compute_anchor_hash(rows, start, window_size)
        anchors.setdefault(digest, []).append(start)
    return anchors


def generate_canonical_turn_id() -> str:
    return str(uuid.uuid4())


def default_sort_key(existing: list[CanonicalTurnRow]) -> float:
    if not existing:
        return 1000.0
    return max(float(row.sort_key or 0.0) for row in existing) + 1000.0


def midpoint_sort_key(left: float | None, right: float | None) -> float:
    if left is None and right is None:
        return 1000.0
    if left is None:
        return right - 1000.0
    if right is None:
        return left + 1000.0
    return (left + right) / 2.0


@dataclass
class CanonicalIngestResult:
    merge_mode: str
    turns_written: int
    turns_matched: int
    turns_appended: int
    turns_prepended: int
    turns_inserted: int
    batch: IngestBatchRecord | None = None
    rows: list[CanonicalTurnRow] = field(default_factory=list)
