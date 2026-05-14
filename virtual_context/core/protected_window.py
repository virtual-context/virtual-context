"""Helpers for the cross-channel-mirror protected window.

This module hosts three pure helpers that participate in the three-tier
gate documented in ``docs/specs/cross-channel-mirror-engine-spec.md``:

* ``_stamp_canonical_turn_ids`` — runs at ingest time, propagates the
  canonical_turn_id (UUID) and turn_number (int) from each ingested
  canonical_turns row onto the matching payload ``Message.metadata``.
* ``_last_already_canonical_turn_number`` — Tier 2 staleness anchor.
  Walks the inbound payload and returns the most recent stamped
  ``metadata["turn_number"]`` as an int; returns ``None`` when no
  comparable anchor exists (legacy / partial-stamping path).
* ``_merge_protected_window`` — Tier 3 merge between the inbound payload
  and the most-recent-N rows fetched from canonical_turns. Dedups by
  ``canonical_turn_id`` first, falls back to ``turn_hash``. Token-budget
  enforcement is deferred to the downstream context-builder.

None of these helpers carry an engine, store, or proxy-state dependency.
They are unit-testable in isolation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Optional

from ..types import Message

if TYPE_CHECKING:  # pragma: no cover - import-only
    from ..types import CanonicalTurnRow


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------


def _stamp_canonical_turn_ids(
    messages: list[Message],
    ingest_rows: Iterable["CanonicalTurnRow"],
) -> None:
    """Stamp canonical_turn_id + turn_number from ingest rows onto messages.

    Walks ``messages`` and ``ingest_rows`` in parallel and writes two
    keys on each matching ``Message.metadata`` dict:

    * ``canonical_turn_id`` (str): Tier 3 dedup primary, populated from
      ``row.canonical_turn_id`` whenever the row carries one.
    * ``turn_number`` (int): Tier 2 staleness anchor, populated only
      when ``row.turn_number`` is a non-sentinel int (>= 0). The default
      sentinel ``-1`` is intentionally skipped so partially-populated
      rows do not seed a stale ``0`` anchor.

    Messages with ``metadata = None`` receive a fresh dict; existing
    metadata dicts are mutated in place. The function is otherwise
    side-effect-free with respect to the message ordering.

    Defensive suffix-tail rule: when ``len(messages) > len(rows)``, the
    stamping aligns to the suffix tail (the most-recent-N messages),
    because the protected-window dedup downstream only consults the
    recent prefix; older mis-stamped entries do not affect correctness.
    When ``len(messages) < len(rows)``, only the leading rows that fit
    are stamped.
    """
    rows = list(ingest_rows)
    if not rows or not messages:
        return

    # Suffix-tail alignment: stamp only the trailing ``len(rows)`` messages
    # in the common index-preserving case. When the payload list is
    # shorter than ``rows`` we still stamp every available message.
    take = min(len(messages), len(rows))
    if take == 0:
        return
    paired = zip(messages[-take:], rows[-take:] if take == len(rows) else rows[:take])
    for message, row in paired:
        canonical_id = getattr(row, "canonical_turn_id", "") or ""
        if not canonical_id:
            continue
        if message.metadata is None:
            message.metadata = {}
        message.metadata["canonical_turn_id"] = canonical_id
        turn_number = getattr(row, "turn_number", -1)
        if isinstance(turn_number, int) and turn_number >= 0:
            message.metadata["turn_number"] = turn_number


# ---------------------------------------------------------------------------
# Tier 2 anchor extraction
# ---------------------------------------------------------------------------


def _last_already_canonical_turn_number(
    conversation_history: list[Message],
) -> Optional[int]:
    """Return the Tier 2 payload-anchor turn-number, or ``None``.

    Walks ``conversation_history`` from the tail toward the head and
    returns the latest-position entry's stamped ``metadata["turn_number"]``
    (int). Returns ``None`` when:

    * No message carries a stamped ``turn_number`` (legacy / MCP /
      library-embedder path that bypassed the stamping helper).
    * The most-recent stamped entry is followed by a NEWER unstamped
      ingestible entry. Per spec §1.1 partial-stamping safety rule,
      equality against an older anchor is unsafe when a newer
      already-canonical payload entry failed to receive a stamp; falling
      through to Tier 3 in that case is the safe behavior.

    Tool-only / system-role / scaffold messages without stamped metadata
    after an anchor do NOT force fall-through. The safety rule only
    forces fall-through when the newer unstamped message is itself in
    the user/assistant role pair the ingestion path would normally
    stamp.

    Active-tail user messages produced by the current request have not
    yet been stamped (their canonical row is being written by the same
    request's ``handle_prepare_payload``); the proxy stamping path is
    therefore expected to drop the active-tail row before invoking
    ``_stamp_canonical_turn_ids``, and this helper naturally excludes
    them from anchor consideration.
    """
    if not conversation_history:
        return None

    # Walk from tail toward head looking for the most recent stamped
    # entry. Track whether we have already seen an unstamped ingestible
    # message after the last stamped one — that's the safety-rule signal.
    seen_unstamped_ingestible_after_anchor = False
    for message in reversed(conversation_history):
        metadata = message.metadata or {}
        turn_number = metadata.get("turn_number") if isinstance(metadata, dict) else None
        if isinstance(turn_number, int) and turn_number >= 0:
            if seen_unstamped_ingestible_after_anchor:
                return None
            return turn_number
        # Non-anchor message. Mark fall-through-required only when this
        # is an ingestible role/content shape (user or assistant text)
        # — those are the messages that would normally be stamped by
        # the ingestion path.
        if _is_ingestible_role(message):
            seen_unstamped_ingestible_after_anchor = True

    return None


def _is_ingestible_role(message: Message) -> bool:
    """Heuristic: does this message belong to the ingestible turn pair?

    A message is "ingestible" for stamping-anchor purposes when its
    role is ``user`` or ``assistant`` and it carries non-empty content.
    System messages and scaffolding messages without content are not
    anchors and do not trip the partial-stamping safety rule.
    """
    role = (getattr(message, "role", "") or "").lower()
    if role not in ("user", "assistant"):
        return False
    content = getattr(message, "content", "") or ""
    return bool(content)


# ---------------------------------------------------------------------------
# Tier 3 merge
# ---------------------------------------------------------------------------


def _merge_protected_window(
    payload_history: list[Message],
    db_recent_rows: list["CanonicalTurnRow"],
    mode: str = "merge",
) -> list[Message]:
    """Merge payload-history with DB-recent canonical_turns rows.

    Returns a fresh ``list[Message]``; ``payload_history`` is NOT
    mutated. Behavior is mode-dependent:

    * ``mode == "off"`` — identity merge. Returns a shallow copy of
      ``payload_history`` unchanged. Provided so callers can route
      through this helper without branching on the mode at call sites.
    * ``mode == "merge"`` — full Tier 3 merge:
        1. Convert each ``CanonicalTurnRow`` into the two-``Message``
           shape (user + assistant). Tool-only rows emit at most one
           ``Message``.
        2. Dedup against ``payload_history`` by ``canonical_turn_id``
           (preferred, populated by ``_stamp_canonical_turn_ids``) or
           ``turn_hash`` (fallback for legacy paths that bypassed
           stamping). Payload entries win ties; DB-source entries are
           dropped on conflict.
        3. Append every unique DB-source ``Message`` to the merged list
           in chronological order (``sort_key`` ascending; ties broken
           by ``created_at`` ASC then ``canonical_turn_id`` ASC for
           determinism).

    Token-budget enforcement is intentionally NOT performed here. The
    existing downstream context-builder (filter + assembler) enforces
    the overall assembled-context cap; introducing a second cap here
    would create artificial scarcity per spec §3.
    """
    if mode != "merge":
        return list(payload_history)

    if not db_recent_rows:
        return list(payload_history)

    # Build dedup index from payload-side metadata.
    payload_canonical_ids: set[str] = set()
    payload_turn_hashes: set[str] = set()
    for message in payload_history:
        metadata = message.metadata or {}
        if isinstance(metadata, dict):
            canonical_id = metadata.get("canonical_turn_id")
            if isinstance(canonical_id, str) and canonical_id:
                payload_canonical_ids.add(canonical_id)
            turn_hash = metadata.get("turn_hash")
            if isinstance(turn_hash, str) and turn_hash:
                payload_turn_hashes.add(turn_hash)

    # Determinism: sort the DB rows by sort_key ASC then created_at ASC then
    # canonical_turn_id ASC. The store returns DESC for fast LIMIT N; the
    # merge wants chronological ASC so the appended messages read
    # oldest-to-newest in the augmented history.
    def _sort_key_tuple(row: "CanonicalTurnRow") -> tuple:
        sk = getattr(row, "sort_key", 0.0) or 0.0
        ca = getattr(row, "created_at", "") or ""
        cid = getattr(row, "canonical_turn_id", "") or ""
        return (float(sk), str(ca), str(cid))

    sorted_rows = sorted(db_recent_rows, key=_sort_key_tuple)

    # Cache the turn-hash for legacy fallback when canonical_turn_id is
    # absent on the payload side but the row carries a hash. Importing
    # the canonical-turns hash helper lazily avoids a hard dependency
    # in this module for callers that only use the dedup index.
    augmented: list[Message] = list(payload_history)
    for row in sorted_rows:
        canonical_id = getattr(row, "canonical_turn_id", "") or ""
        turn_hash = getattr(row, "turn_hash", "") or ""
        if canonical_id and canonical_id in payload_canonical_ids:
            continue
        if turn_hash and turn_hash in payload_turn_hashes:
            continue

        # Convert row to message(s). Tool-only rows emit zero or one
        # messages; both-halves-present rows emit two.
        user_text = getattr(row, "user_content", "") or ""
        assistant_text = getattr(row, "assistant_content", "") or ""
        timestamp = _row_timestamp(row)
        base_metadata = {
            "canonical_turn_id": canonical_id,
            "turn_number": getattr(row, "turn_number", -1),
            "source": "db_recent",
        }
        if user_text:
            user_msg = Message(
                role="user",
                content=user_text,
                timestamp=timestamp,
                metadata=dict(base_metadata),
            )
            augmented.append(user_msg)
        if assistant_text:
            assistant_msg = Message(
                role="assistant",
                content=assistant_text,
                timestamp=timestamp,
                metadata=dict(base_metadata),
            )
            augmented.append(assistant_msg)
        # Record the canonical_id for downstream dedup against further
        # rows that might collide (defensive, sorted_rows should already
        # be unique).
        if canonical_id:
            payload_canonical_ids.add(canonical_id)
        if turn_hash:
            payload_turn_hashes.add(turn_hash)

    return augmented


def _row_timestamp(row: "CanonicalTurnRow") -> datetime | None:
    """Best-effort timestamp parse for a CanonicalTurnRow.

    Falls back through ``last_seen_at`` → ``first_seen_at`` →
    ``updated_at`` → ``created_at`` and returns ``None`` when none are
    parseable. The returned ``Message.timestamp`` is informational; the
    assembler does not currently slice by it for the protected window.
    """
    for attr in ("last_seen_at", "first_seen_at", "updated_at", "created_at"):
        value = getattr(row, attr, None)
        if not value:
            continue
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
    return None


__all__ = [
    "_stamp_canonical_turn_ids",
    "_last_already_canonical_turn_number",
    "_merge_protected_window",
]


# Imported for hashlib parity with future legacy turn_hash recompute paths.
_ = hashlib  # noqa: F841 - placeholder kept for future fallback paths
del _
