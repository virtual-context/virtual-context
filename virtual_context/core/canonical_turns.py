"""Helpers for canonical turn identity, normalization, and anchor matching."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import json

from ..types import CanonicalTurnIndexRow, CanonicalTurnRow, FactSignal, IngestBatchRecord

HASH_VERSION = 1


def pair_text_hash(user_text: str, assistant_text: str) -> str:
    """The turn index's key for a turn: sha256 over ``"<user> <assistant>"``, 16 hex digits."""
    return hashlib.sha256(f"{user_text} {assistant_text}".encode()).hexdigest()[:16]


def canonical_turn_index_row(row) -> CanonicalTurnIndexRow:
    """Build an index row from a mapping of the projected columns."""
    def _json_list(key: str) -> list:
        try:
            value = json.loads(row[key] or "[]")
        except Exception:
            return []
        return value if isinstance(value, list) else []

    fact_signals = [
        FactSignal(
            subject=item.get("subject", ""),
            verb=item.get("verb", ""),
            object=item.get("object", ""),
            status=item.get("status", ""),
            fact_type=item.get("fact_type", ""),
            what=item.get("what", ""),
        )
        for item in _json_list("fact_signals_json")
        if isinstance(item, dict)
    ]
    return CanonicalTurnIndexRow(
        canonical_turn_id=str(row["canonical_turn_id"] or ""),
        turn_number=int(row["turn_number"]) if row["turn_number"] is not None else -1,
        turn_group_number=int(row["turn_group_number"]) if row["turn_group_number"] is not None else -1,
        sort_key=float(row["sort_key"] or 0.0),
        primary_tag=row["primary_tag"] or "_general",
        tags=list(_json_list("tags_json")),
        session_date=row["session_date"] or "",
        sender=row["sender"] or "",
        fact_signals=fact_signals,
        code_refs=list(_json_list("code_refs_json")),
        tagged_at=row["tagged_at"] or None,
        compacted_at=row["compacted_at"] or None,
        first_seen_at=row["first_seen_at"] or None,
        last_seen_at=row["last_seen_at"] or None,
        created_at=row["created_at"] or None,
        updated_at=row["updated_at"] or None,
        has_user_content=bool(row["has_user_content"]),
        has_assistant_content=bool(row["has_assistant_content"]),
        pair_hash=row["pair_hash"] or "",
    )

# Every character ``str.strip()`` removes, as one string.
#
# Storage backends decide whether a stored row carries text by trimming it
# in SQL, while the rest of the engine asks ``(value or "").strip()``. Those
# two must answer identically or a row that holds only whitespace can be
# read as carrying content, which is what lets a row that is not the user
# half of a turn take speaker attribution.
#
# The set is passed to SQL as a bound parameter rather than written as an
# escaped literal per dialect. Hand-writing it invites exactly the mismatch
# it is meant to prevent: PostgreSQL's escape strings define ``\b \f \n \r
# \t`` and drop the backslash on anything else, so ``E'\v'`` is the letter
# ``v`` and not a vertical tab, while SQLite's ``char(11)`` is. Deriving it
# once, here, keeps both dialects on the same ruler as Python.
#
# ``test_canonical_turns.py`` regenerates this from ``str.strip()`` and
# fails if the two ever drift.
STRIP_WHITESPACE = (
    "\x09\x0a\x0b\x0c\x0d"
    "\x1c\x1d\x1e\x1f\x20"
    "\x85\xa0\u1680\u2000\u2001"
    "\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028"
    "\u2029\u202f\u205f\u3000"
)

_WS_RE = re.compile(r"\s+")
_MEDIA_RE = re.compile(r"\[media attached:[^\]]+\]", re.IGNORECASE)
_SESSION_RE = re.compile(r"\[Session from [^\]]+\]", re.IGNORECASE)
# A chat host may stamp the model-facing copy of a user message with its send
# time; the same message ingested from the host's own record carries no stamp.
_HOST_TIMESTAMP_RE = re.compile(r"^\s*\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]\s*")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_turn_text(text: str | None) -> str:
    text = str(text or "")
    # Strip every leading stamp: the host adds one, and a message that itself
    # begins with one must normalize the same way whichever copy arrives first.
    while True:
        stripped = _HOST_TIMESTAMP_RE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
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


def partition_canonical_rows_into_logical_turns(
    rows: list[CanonicalTurnRow],
) -> list[list[CanonicalTurnRow]]:
    """Partition chronologically ordered physical rows into logical turns.

    ``turn_group_number`` is useful only as an adjacency proof. Historical
    merge races can reuse a number for far-apart turns, so grouping every row
    with the same integer would conflate unrelated history. Canonical turn
    storage emits at most one user half and one assistant half per logical
    turn; legacy rows without group numbers fall back to the same adjacency
    pairing.
    """
    groups: list[list[CanonicalTurnRow]] = []
    pending_user: list[CanonicalTurnRow] = []

    def _flush_pending() -> None:
        nonlocal pending_user
        if pending_user:
            groups.append(list(pending_user))
            pending_user = []

    def _same_explicit_group(
        user_row: CanonicalTurnRow,
        assistant_row: CanonicalTurnRow,
    ) -> bool:
        def _group_number(row: CanonicalTurnRow) -> int:
            raw = getattr(row, "turn_group_number", -1)
            try:
                return int(raw) if raw is not None else -1
            except (TypeError, ValueError):
                return -1

        user_group = _group_number(user_row)
        assistant_group = _group_number(assistant_row)
        return (
            user_group < 0
            or assistant_group < 0
            or user_group == assistant_group
        )

    for row in rows:
        has_user = bool(row.user_content)
        has_assistant = bool(row.assistant_content)
        if has_user:
            _flush_pending()
            if has_assistant:
                groups.append([row])
            else:
                pending_user = [row]
            continue
        if has_assistant:
            if pending_user and _same_explicit_group(pending_user[0], row):
                pending_user.append(row)
                _flush_pending()
            else:
                _flush_pending()
                groups.append([row])
            continue

        # A malformed/tool-only physical row still counts as a singleton
        # boundary so it cannot bridge two otherwise unrelated turns.
        _flush_pending()
        groups.append([row])

    _flush_pending()
    return groups


def select_recent_logical_turn_rows(
    rows_desc: list[CanonicalTurnRow],
    *,
    limit: int,
) -> list[CanonicalTurnRow]:
    """Select the newest ``limit`` logical turns from DESC physical rows.

    Callers fetch ``2 * limit + 1`` rows: at most two physical rows belong to
    one canonical logical turn, and the extra row absorbs a boundary that
    begins on an assistant half. The returned rows retain DESC order.
    """
    if limit <= 0 or not rows_desc:
        return []
    groups = partition_canonical_rows_into_logical_turns(
        list(reversed(rows_desc))
    )
    selected = groups[-int(limit):]
    # ``groups`` is chronological; callers promise and expect DESC order.
    # Flatten and reverse rather than relying on Python object identity, which
    # would be fragile if a future backend materialized/cached equivalent rows.
    return list(reversed([
        row
        for group in selected
        for row in group
    ]))


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
