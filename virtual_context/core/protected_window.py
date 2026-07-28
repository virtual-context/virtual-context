"""Helpers for the cross-channel-mirror protected window.

This module hosts pure helpers that participate in the three-tier
gate documented in ``docs/specs/cross-channel-mirror-engine-spec.md``:

* ``_stamp_canonical_turn_ids`` — runs at ingest time, propagates the
  canonical_turn_id (UUID) and turn_number (int) from each ingested
  canonical_turns row onto the matching payload ``Message.metadata``.
* ``_last_already_canonical_turn_number`` — Tier 2 staleness anchor.
  Walks the inbound payload and returns the most recent stamped
  ``metadata["turn_number"]`` as an int; returns ``None`` when no
  comparable anchor exists (legacy / partial-stamping path).
* ``_merge_protected_window`` — Tier 3 merge between the inbound payload
  and the most-recent-N logical groups fetched from canonical_turns. Dedups
  whole logical groups by ``canonical_turn_id``, ``turn_hash``, or
  ``source_message_id``. Token-budget enforcement is deferred to the
  downstream context-builder.
* ``_slice_payload_prefix_preserving_db_recent`` — applies the payload-owned
  compaction watermark after Tier 3 without consuming recovered DB rows or
  the trailing active-user block.

None of these helpers carry an engine, store, or proxy-state dependency.
They are unit-testable in isolation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Optional

from ..types import Message, get_current_conversation_info, get_origin_channel
from .canonical_turns import partition_canonical_rows_into_logical_turns

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


def _drop_active_tail_ingest_rows(
    ingest_rows: Iterable["CanonicalTurnRow"],
    active_tail_messages: int,
) -> list["CanonicalTurnRow"]:
    """Return only ingest rows that can correspond to completed history.

    The active inbound tail is present in the request body and canonical
    ingest result but is deliberately absent from the completed-history list
    being stamped. When every returned row belongs to that active tail, the
    safe result is an empty list rather than suffix-stamping an older retained
    message with the current request's identity.
    """
    rows = list(ingest_rows)
    drop = max(0, int(active_tail_messages))
    if drop <= 0:
        return rows
    if drop >= len(rows):
        return []
    return rows[:-drop]


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
# Compaction-aware merged-history view
# ---------------------------------------------------------------------------


def _is_db_recent(message: Message) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return metadata.get("source") == "db_recent"


def _is_trailing_unstamped_user(message: Message) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return (
        message.role == "user"
        and not metadata.get("canonical_turn_id")
        and metadata.get("source") != "db_recent"
    )


def _slice_payload_prefix_preserving_db_recent(
    merged_history: list[Message],
    payload_offset: int,
) -> list[Message]:
    """Apply a payload watermark without consuming recovered DB rows.

    ``EngineState.history_offset`` describes payload-owned history. Tier 3
    inserts ``source=db_recent`` messages into that list before assembly, so a
    positional slice can split a recovered logical turn. Consume the offset
    from payload-owned rows only.

    The trailing unstamped-user block is the active request tail used by the
    Tier 3 insertion rule. It is never eligible for removal: unified guild
    watermarks are conversation-scoped while the payload is channel-local, so
    the requested offset can legitimately equal or exceed the local payload
    length.
    """
    if not merged_history:
        return []

    active_start = len(merged_history)
    while (
        active_start > 0
        and _is_trailing_unstamped_user(merged_history[active_start - 1])
    ):
        active_start -= 1

    eligible_payload = sum(
        1
        for message in merged_history[:active_start]
        if not _is_db_recent(message)
    )
    remaining = min(max(0, int(payload_offset)), eligible_payload)
    view: list[Message] = []
    for index, message in enumerate(merged_history):
        if index >= active_start or _is_db_recent(message):
            view.append(message)
            continue
        if remaining > 0:
            remaining -= 1
            continue
        view.append(message)
    return view


# ---------------------------------------------------------------------------
# Tier 3 merge
# ---------------------------------------------------------------------------


def _merge_protected_window(
    payload_history: list[Message],
    db_recent_rows: list["CanonicalTurnRow"],
    mode: str = "merge",
    *,
    dedup_origin_channel_id: str = "",
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
        2. Dedup logical DB groups against same-channel ``payload_history`` by
           ``canonical_turn_id`` (preferred, populated by
           ``_stamp_canonical_turn_ids``), ``turn_hash``, or
           ``source_message_id``. A complete payload pair (or the active
           request tail) wins the whole-group tie. An incomplete, non-active
           payload fragment is replaced by the complete canonical group.
        3. Insert every unique DB-source ``Message`` in chronological order
           immediately before a trailing unstamped active-user request
           (``sort_key`` ascending; ties broken by ``created_at`` ASC then
           ``canonical_turn_id`` ASC for determinism). Callers without such
           an active tail retain append behavior.

    Token-budget enforcement is intentionally NOT performed here. The
    existing downstream context-builder (filter + assembler) enforces
    the overall assembled-context cap; introducing a second cap here
    would create artificial scarcity per spec §3.
    """
    if mode != "merge":
        return list(payload_history)

    if not db_recent_rows:
        return list(payload_history)

    # Build dedup indexes from payload-side metadata.  Source-message ids are
    # load-bearing for the normal prepare race: the current inbound user row
    # can already exist in canonical_turns by the time Tier 3 reads.  Text is
    # deliberately never a key because short repeated messages are valid.
    payload_canonical_indexes: dict[str, set[int]] = {}
    payload_turn_hash_indexes: dict[str, set[int]] = {}
    payload_source_message_indexes: dict[str, set[int]] = {}

    def _index_value(
        index: dict[str, set[int]],
        value: object,
        payload_index: int,
    ) -> None:
        if isinstance(value, str) and value:
            index.setdefault(value, set()).add(payload_index)

    for payload_index, message in enumerate(payload_history):
        metadata = message.metadata or {}
        if isinstance(metadata, dict):
            # A unified guild engine retains history contributed by every
            # channel, while the upstream Discord payload remains
            # channel-local. A canonical twin in another channel's retained
            # engine history is therefore not proof that the model-visible
            # payload contains that turn. Let only exact same-channel rows
            # suppress the canonical copy. Unknown-channel legacy rows also
            # fail open to a duplicate rather than silently hiding the only
            # cross-channel copy from the model.
            if dedup_origin_channel_id:
                payload_origin, _ = get_origin_channel(metadata)
                if not payload_origin:
                    top_level_origin = metadata.get("origin_channel_id")
                    payload_origin = (
                        top_level_origin
                        if isinstance(top_level_origin, str)
                        else ""
                    )
                if payload_origin != dedup_origin_channel_id:
                    continue
            _index_value(
                payload_canonical_indexes,
                metadata.get("canonical_turn_id"),
                payload_index,
            )
            _index_value(
                payload_turn_hash_indexes,
                metadata.get("turn_hash"),
                payload_index,
            )
            source_message_id = metadata.get("source_message_id")
            if not (isinstance(source_message_id, str) and source_message_id):
                current = get_current_conversation_info(metadata)
                source_message_id = current.get("message_id")
            _index_value(
                payload_source_message_indexes,
                source_message_id,
                payload_index,
            )

    # Identify complete native payload pairs and the current active-user tail.
    # A match in either is sufficient to let the payload win the whole logical
    # group. A lone historical fragment is not sufficient: dropping its DB
    # sibling would recreate the assistant-only/user-only history corruption
    # this gate exists to repair.
    active_start = len(payload_history)
    while (
        active_start > 0
        and _is_trailing_unstamped_user(payload_history[active_start - 1])
    ):
        active_start -= 1
    active_payload_indexes = set(range(active_start, len(payload_history)))
    complete_payload_indexes: set[int] = set()
    pending_user_index: int | None = None
    for payload_index, message in enumerate(payload_history[:active_start]):
        role = (getattr(message, "role", "") or "").lower()
        if role == "user":
            pending_user_index = payload_index
        elif role == "assistant":
            if pending_user_index is not None:
                complete_payload_indexes.update(
                    (pending_user_index, payload_index)
                )
            pending_user_index = None
        else:
            # Transport/tool scaffolding can sit between the human request and
            # its assistant response. It does not end the candidate pair.
            continue

    # Determinism: sort the DB rows by sort_key ASC then created_at ASC then
    # canonical_turn_id ASC. The store returns DESC for fast LIMIT N; the
    # merge wants chronological ASC so the inserted messages read
    # oldest-to-newest in the augmented history.
    def _sort_key_tuple(row: "CanonicalTurnRow") -> tuple:
        sk = getattr(row, "sort_key", 0.0) or 0.0
        ca = getattr(row, "created_at", "") or ""
        cid = getattr(row, "canonical_turn_id", "") or ""
        return (float(sk), str(ca), str(cid))

    sorted_rows = sorted(db_recent_rows, key=_sort_key_tuple)

    # Raw turn_group_number values are not globally unique in repaired legacy
    # histories. Partition by chronological adjacency, then give every group a
    # request-local stable key. This prevents a payload match for an old group
    # from suppressing a newer group that happens to reuse the same integer.
    logical_groups = partition_canonical_rows_into_logical_turns(sorted_rows)

    def _matching_payload_indexes(row: "CanonicalTurnRow") -> set[int]:
        matches: set[int] = set()
        canonical_id = getattr(row, "canonical_turn_id", "") or ""
        turn_hash = getattr(row, "turn_hash", "") or ""
        source_message_id = getattr(row, "source_message_id", "") or ""
        if canonical_id:
            matches.update(payload_canonical_indexes.get(canonical_id, set()))
        if turn_hash:
            matches.update(payload_turn_hash_indexes.get(turn_hash, set()))
        if source_message_id:
            matches.update(
                payload_source_message_indexes.get(source_message_id, set())
            )
        return matches

    suppressed_group_indexes: set[int] = set()
    replaced_payload_indexes: set[int] = set()
    for group_index, group in enumerate(logical_groups):
        matches = {
            payload_index
            for row in group
            for payload_index in _matching_payload_indexes(row)
        }
        if not matches:
            continue
        if matches & (complete_payload_indexes | active_payload_indexes):
            suppressed_group_indexes.add(group_index)
        else:
            # The canonical group is the only complete representation. Remove
            # the matching native fragment and insert the canonical pair.
            replaced_payload_indexes.update(matches)

    payload_view = [
        message
        for payload_index, message in enumerate(payload_history)
        if payload_index not in replaced_payload_indexes
    ]

    # Cache the turn-hash for legacy fallback when canonical_turn_id is
    # absent on the payload side but the row carries a hash. Importing
    # the canonical-turns hash helper lazily avoids a hard dependency
    # in this module for callers that only use the dedup index.
    db_message_groups: list[list[Message]] = []
    for group_index, group in enumerate(logical_groups):
        if group_index in suppressed_group_indexes:
            continue
        first_canonical_id = next(
            (
                str(getattr(group_row, "canonical_turn_id", "") or "")
                for group_row in group
                if getattr(group_row, "canonical_turn_id", "") or ""
            ),
            "",
        )
        first_sort_key = float(
            getattr(group[0], "sort_key", 0.0) or 0.0
        )
        db_recent_group_key = (
            f"canonical:{first_canonical_id}"
            if first_canonical_id
            else f"window:{group_index}:{first_sort_key}"
        )
        group_provenance: dict[str, str] = {}
        provenance_group_number = -1
        for group_row in group:
            if not (getattr(group_row, "user_content", "") or ""):
                continue
            raw_group_number = getattr(
                group_row, "turn_group_number", -1
            )
            try:
                candidate_group_number = (
                    int(raw_group_number)
                    if raw_group_number is not None
                    else -1
                )
            except (TypeError, ValueError):
                candidate_group_number = -1
            if candidate_group_number >= 0:
                provenance_group_number = candidate_group_number
            for key in (
                "origin_channel_id",
                "origin_channel_label",
                "audience_conversation_id",
            ):
                value = getattr(group_row, key, "") or ""
                if isinstance(value, str) and value and not group_provenance.get(key):
                    group_provenance[key] = value

        group_messages: list[Message] = []
        for row in group:
            canonical_id = getattr(row, "canonical_turn_id", "") or ""
            source_message_id = getattr(row, "source_message_id", "") or ""

            # Convert row to message(s). Tool-only rows emit zero or one
            # messages; both-halves-present rows emit two.
            user_text = getattr(row, "user_content", "") or ""
            assistant_text = getattr(row, "assistant_content", "") or ""
            timestamp = _row_timestamp(row)
            group_number = getattr(row, "turn_group_number", -1)
            try:
                explicit_group_number = (
                    int(group_number) if group_number is not None else -1
                )
            except (TypeError, ValueError):
                explicit_group_number = -1
            # Adjacency may pair legacy (-1) rows for window completeness, but
            # it is not transport/audience proof. Only two explicit equal
            # group numbers authorize provenance inheritance onto a physical
            # assistant half. This is the DM/privacy fail-closed boundary.
            inherited = (
                group_provenance
                if (
                    not user_text
                    and provenance_group_number >= 0
                    and explicit_group_number == provenance_group_number
                )
                else {}
            )
            base_metadata: dict[str, object] = {
                "canonical_turn_id": canonical_id,
                "turn_number": getattr(row, "turn_number", -1),
                "turn_group_number": group_number,
                "db_recent_group_key": db_recent_group_key,
                "sort_key": getattr(row, "sort_key", 0.0) or 0.0,
                "source": "db_recent",
            }
            for key in (
                "origin_channel_id",
                "origin_channel_label",
                "audience_conversation_id",
            ):
                value = getattr(row, key, "") or inherited.get(key, "")
                if isinstance(value, str) and value:
                    base_metadata[key] = value
            if source_message_id:
                base_metadata["source_message_id"] = source_message_id
            if user_text:
                user_metadata = dict(base_metadata)
                # Sender attributes the human half only. A legacy row may
                # carry the logical-turn sender on both halves.
                sender = (getattr(row, "sender", "") or "").strip()
                if sender:
                    user_metadata["sender"] = {"name": sender}
                sender_actor_id = (
                    getattr(row, "sender_actor_id", "") or ""
                ).strip()
                if sender_actor_id:
                    user_metadata["sender_actor_id"] = sender_actor_id
                group_messages.append(Message(
                    role="user",
                    content=user_text,
                    timestamp=timestamp,
                    metadata=user_metadata,
                ))
            if assistant_text:
                group_messages.append(Message(
                    role="assistant",
                    content=assistant_text,
                    timestamp=timestamp,
                    metadata=dict(base_metadata),
                ))
        if group_messages:
            db_message_groups.append(group_messages)

    # The active inbound user is deliberately unstamped and belongs after the
    # recovered window. Interleave canonical groups with stamped payload
    # history by turn number so replacing an older sliced/partial group cannot
    # move it behind newer native turns. Unstamped historical payload entries
    # retain their native order; unnumbered DB groups remain at the end of the
    # historical prefix because no safe cross-source ordering proof exists.
    insert_at = len(payload_view)
    while insert_at > 0:
        tail = payload_view[insert_at - 1]
        if _is_trailing_unstamped_user(tail):
            insert_at -= 1
            continue
        break
    payload_prefix = list(payload_view[:insert_at])
    active_tail = list(payload_view[insert_at:])

    def _message_turn_number(message: Message) -> int | None:
        metadata = (
            message.metadata if isinstance(message.metadata, dict) else {}
        )
        value = metadata.get("turn_number")
        return value if isinstance(value, int) and value >= 0 else None

    def _group_last_turn_number(
        messages: list[Message],
    ) -> int | None:
        values = [
            value
            for message in messages
            if (value := _message_turn_number(message)) is not None
        ]
        return max(values) if values else None

    merged_prefix: list[Message] = []
    next_db_group = 0
    for payload_message in payload_prefix:
        payload_turn_number = _message_turn_number(payload_message)
        if payload_turn_number is not None:
            while next_db_group < len(db_message_groups):
                db_group = db_message_groups[next_db_group]
                db_last_turn_number = _group_last_turn_number(db_group)
                if (
                    db_last_turn_number is None
                    or db_last_turn_number >= payload_turn_number
                ):
                    break
                merged_prefix.extend(db_group)
                next_db_group += 1
        merged_prefix.append(payload_message)
    for db_group in db_message_groups[next_db_group:]:
        merged_prefix.extend(db_group)
    return merged_prefix + active_tail


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
    "_drop_active_tail_ingest_rows",
    "_last_already_canonical_turn_number",
    "_merge_protected_window",
    "_slice_payload_prefix_preserving_db_recent",
]


# Imported for hashlib parity with future legacy turn_hash recompute paths.
_ = hashlib  # noqa: F841 - placeholder kept for future fallback paths
del _
