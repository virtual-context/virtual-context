"""SessionState marker derivation from durable canonical_turns.

Used in two places:

* ``virtual_context/proxy/vcattach.py:execute_attach`` — at-VCATTACH-time
  derivation + provider.save so the target conv's Redis SessionState
  reflects durable state before sibling workers re-hydrate.
* ``virtual_context/cli/main.py`` admin
  ``backfill-session-state-markers`` subcommand — one-shot recovery
  for conversations whose Redis SessionState drifted from their
  canonical_turns truth.

The derivation is pure read-then-build: canonical_turns is
authoritative (commits visible across workers via SQL), and every
marker field that downstream paging / retrieval / passthrough gates
consult can be reconstructed from it.

See ``docs/specs/vcattach-redis-marker-write-and-cross-worker-invalidation.md``
for the surrounding design.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..proxy.session_state import SessionState
from ..types import TurnTagEntry

logger = logging.getLogger(__name__)


@dataclass
class _DerivedMarkers:
    """Internal carrier for the derivation pass."""

    compacted_prefix_messages: int = 0
    last_compacted_turn: int = -1
    last_completed_turn: int = -1
    last_indexed_turn: int = -1
    turn_tag_entries: list[TurnTagEntry] = field(default_factory=list)


def _parse_timestamp(value: Any) -> datetime:
    """Best-effort parse of one of the canonical_turn timestamp fields.

    Mirrors ``VirtualContextEngine._parse_turn_timestamp`` semantics so
    derived turn_tag_entries carry plausible timestamps when the row
    has any of last_seen_at / first_seen_at / updated_at populated.
    """
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _build_turn_tag_entries(
    paired_rows: list[tuple[int, list[Any]]],
) -> list[TurnTagEntry]:
    """Build a ``TurnTagEntry`` list from paired canonical_turn rows.

    Only includes pairs where every row carries a ``tagged_at`` value —
    those are the rows the tagging pipeline has fully processed. Mirrors
    the filter the engine's ``_restore_from_canonical_rows`` uses for
    the same purpose at non-provider load time.
    """
    entries: list[TurnTagEntry] = []
    for turn_number, pair_rows in paired_rows:
        if not pair_rows:
            continue
        if not all(getattr(row, "tagged_at", None) for row in pair_rows):
            continue

        primary_tag = "_general"
        sender = ""
        session_date = ""
        canonical_turn_id = ""
        user_content = ""
        assistant_content = ""
        tags: list[str] = []
        fact_signals: list[Any] = []
        code_refs: list[dict] = []
        timestamps: list[Any] = []

        for row in pair_rows:
            if not canonical_turn_id:
                canonical_turn_id = getattr(row, "canonical_turn_id", "") or ""
            row_primary = getattr(row, "primary_tag", "") or ""
            if row_primary not in ("", "_general"):
                primary_tag = row_primary
            if not sender:
                sender = getattr(row, "sender", "") or ""
            if not session_date:
                session_date = getattr(row, "session_date", "") or ""
            if not user_content and getattr(row, "user_content", ""):
                user_content = str(getattr(row, "user_content", "") or "")
            if not assistant_content and getattr(row, "assistant_content", ""):
                assistant_content = str(getattr(row, "assistant_content", "") or "")
            tags.extend(list(getattr(row, "tags", []) or []))
            fact_signals.extend(list(getattr(row, "fact_signals", []) or []))
            code_refs.extend(list(getattr(row, "code_refs", []) or []))
            timestamps.extend([
                getattr(row, "last_seen_at", None),
                getattr(row, "first_seen_at", None),
                getattr(row, "updated_at", None),
                getattr(row, "created_at", None),
            ])

        ts_value = next((t for t in timestamps if t), None)
        message_hash = hashlib.sha256(
            f"{user_content} {assistant_content}".encode(),
        ).hexdigest()[:16]
        entries.append(TurnTagEntry(
            turn_number=int(turn_number),
            message_hash=message_hash,
            canonical_turn_id=canonical_turn_id,
            tags=list(dict.fromkeys(tags)) or [primary_tag],
            primary_tag=primary_tag,
            timestamp=_parse_timestamp(ts_value),
            session_date=session_date,
            fact_signals=fact_signals,
            sender=sender,
            code_refs=code_refs,
        ))
    return entries


def derive_session_state_markers(
    store: Any,
    conversation_id: str,
    *,
    existing_state: SessionState | None = None,
) -> SessionState | None:
    """Derive SessionState markers from canonical_turns for ``conversation_id``.

    Args:
        store: unwrapped storage backend. Must expose
            ``get_all_canonical_turns(conversation_id)``. That storage
            API is conversation-scoped by contract; this helper trusts
            the store to filter by ``conversation_id`` rather than
            re-filtering row objects, because lightweight test/store
            adapters do not always carry a row-level ``conversation_id``
            attribute.
        conversation_id: target conversation.
        existing_state: optional SessionState that may already exist in
            Redis. Non-derivable fields (``checkpoint_version``,
            ``conversation_generation``, ``tool_tag_counter``,
            ``split_processed_tags``, ``trailing_fingerprint``,
            ``provider``, ``version``) are carried forward when supplied
            so the marker write doesn't clobber unrelated state.

    Returns:
        A ``SessionState`` with all derivable fields populated. Returns
        ``None`` when ``conversation_id`` has no canonical_turns rows
        (fresh conv) — caller should leave Redis untouched in that
        case.

    Derivation rules:

    * ``compacted_prefix_messages``: canonical-prefix watermark from
      paired rows where every row has ``compacted_at IS NOT NULL``.
    * ``last_compacted_turn``: the turn_number of the highest compacted
      pair (or -1 when no compacted pairs exist).
    * ``last_completed_turn``: max ``turn_number`` of any paired row
      (or -1 when no rows exist; we already return None earlier in
      that case).
    * ``last_indexed_turn``: max ``turn_number`` of any pair where
      every row carries ``tagged_at IS NOT NULL``.
    * ``flushed_prefix_messages``: equal to
      ``compacted_prefix_messages``. Matches the v0.4.5 clamp-inversion
      semantic — in steady state ``flushed`` follows ``compacted``;
      the per-session ``flushed`` value is a transient
      payload-mutation watermark that the proxy's ``prepare_payload``
      flush gate maintains separately at request time.
    * ``turn_tag_entries``: reconstructed list of ``TurnTagEntry``
      from rows where every row has ``tagged_at IS NOT NULL``.
    * ``working_set``: empty ``[]`` per the spec (paging working set
      is per-session ephemeral; retriever auto-populates on next
      inbound).
    """
    from ..engine import VirtualContextEngine

    try:
        rows = list(store.get_all_canonical_turns(conversation_id))
    except Exception:
        logger.warning(
            "derive_session_state_markers: failed to load canonical_turns "
            "for %s",
            conversation_id[:12],
            exc_info=True,
        )
        raise

    if not rows:
        return None

    paired_rows = VirtualContextEngine._group_canonical_rows_into_pairs(rows)
    if not paired_rows:
        return None

    compacted, last_compacted_turn = VirtualContextEngine._canonical_prefix_watermark(
        paired_rows,
    )

    last_completed_turn = -1
    last_indexed_turn = -1
    for turn_number, pair_rows in paired_rows:
        if not pair_rows:
            continue
        last_completed_turn = max(last_completed_turn, int(turn_number))
        if all(getattr(row, "tagged_at", None) for row in pair_rows):
            last_indexed_turn = max(last_indexed_turn, int(turn_number))

    turn_tag_entries = _build_turn_tag_entries(paired_rows)

    # Carry forward non-derivable fields when an existing SessionState was
    # supplied — these are session-scoped values (not DB-derivable) that
    # the marker write shouldn't clobber. When no existing state is
    # available, fall back to safe defaults; the next live request that
    # touches the conv will re-establish them naturally.
    if existing_state is not None:
        last_request_time = float(getattr(existing_state, "last_request_time", 0.0) or 0.0)
        checkpoint_version = int(getattr(existing_state, "checkpoint_version", 0) or 0) + 1
        conversation_generation = int(getattr(existing_state, "conversation_generation", 0) or 0)
        tool_tag_counter = int(getattr(existing_state, "tool_tag_counter", 0) or 0)
        split_processed_tags = set(getattr(existing_state, "split_processed_tags", set()) or set())
        trailing_fingerprint = str(getattr(existing_state, "trailing_fingerprint", "") or "")
        provider = str(getattr(existing_state, "provider", "") or "")
        session_state = str(getattr(existing_state, "session_state", "") or "")
        live_turn_count = int(getattr(existing_state, "live_turn_count", 0) or 0)
        history_message_count = int(getattr(existing_state, "history_message_count", 0) or 0)
        ingestion_done = int(getattr(existing_state, "ingestion_done", 0) or 0)
        ingestion_total = int(getattr(existing_state, "ingestion_total", 0) or 0)
        last_payload_kb = float(getattr(existing_state, "last_payload_kb", 0.0) or 0.0)
        last_payload_tokens = int(getattr(existing_state, "last_payload_tokens", 0) or 0)
        raw_payload_entry_count = int(getattr(existing_state, "raw_payload_entry_count", 0) or 0)
        ingestible_entry_count = int(getattr(existing_state, "ingestible_entry_count", 0) or 0)
        skipped_payload_entry_count = int(getattr(existing_state, "skipped_payload_entry_count", 0) or 0)
        telemetry_rollup = dict(getattr(existing_state, "telemetry_rollup", {}) or {})
        request_captures = list(getattr(existing_state, "request_captures", []) or [])
        deleted = bool(getattr(existing_state, "deleted", False))
        version = int(getattr(existing_state, "version", 0) or 0)
    else:
        last_request_time = 0.0
        checkpoint_version = 1
        conversation_generation = 0
        tool_tag_counter = 0
        split_processed_tags = set()
        trailing_fingerprint = ""
        provider = ""
        session_state = ""
        live_turn_count = 0
        history_message_count = 0
        ingestion_done = 0
        ingestion_total = 0
        last_payload_kb = 0.0
        last_payload_tokens = 0
        raw_payload_entry_count = 0
        ingestible_entry_count = 0
        skipped_payload_entry_count = 0
        telemetry_rollup = {}
        request_captures = []
        deleted = False
        version = 0

    serialized_entries = [
        {
            "turn_number": e.turn_number,
            "message_hash": e.message_hash,
            "canonical_turn_id": e.canonical_turn_id,
            "tags": e.tags,
            "primary_tag": e.primary_tag,
            "timestamp": e.timestamp.isoformat() if e.timestamp else "",
            "session_date": e.session_date,
            "sender": e.sender,
            "fact_signals": [
                {
                    "subject": fs.subject,
                    "verb": fs.verb,
                    "object": fs.object,
                    "status": fs.status,
                    "fact_type": getattr(fs, "fact_type", ""),
                    "what": getattr(fs, "what", ""),
                }
                for fs in (e.fact_signals or [])
            ],
            "code_refs": list(e.code_refs or []),
        }
        for e in turn_tag_entries
    ]

    return SessionState(
        compacted_prefix_messages=int(compacted),
        flushed_prefix_messages=int(compacted),
        flushed_prefix_messages_present=True,
        last_request_time=last_request_time,
        last_compacted_turn=int(last_compacted_turn),
        last_completed_turn=int(last_completed_turn),
        last_indexed_turn=int(last_indexed_turn),
        checkpoint_version=checkpoint_version,
        conversation_generation=conversation_generation,
        tool_tag_counter=tool_tag_counter,
        split_processed_tags=split_processed_tags,
        trailing_fingerprint=trailing_fingerprint,
        provider=provider,
        session_state=session_state,
        live_turn_count=live_turn_count,
        history_message_count=history_message_count,
        ingestion_done=ingestion_done,
        ingestion_total=ingestion_total,
        last_payload_kb=last_payload_kb,
        last_payload_tokens=last_payload_tokens,
        raw_payload_entry_count=raw_payload_entry_count,
        ingestible_entry_count=ingestible_entry_count,
        skipped_payload_entry_count=skipped_payload_entry_count,
        turn_tag_entries=serialized_entries,
        working_set=[],
        telemetry_rollup=telemetry_rollup,
        request_captures=request_captures,
        version=version,
        deleted=deleted,
    )
