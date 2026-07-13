"""Audience-scoped speaker roster construction and safe presentation.

This module owns the request-local roster: which participants a request may
see, in what order, under which handles, and how the block is rendered.

* **Membership** is derived from physical canonical USER rows under the
  alias-resolved owner — never by enumerating the handle-assignment table and
  never from tenant-global profiles. A row is admissible only when it is a
  human user lane with a non-empty ``sender_actor_id``, its validated
  pre-alias audience exactly equals the request audience, its audience
  attribution version is current, and — when the request carries a durable
  channel — its stored channel matches exactly (an empty stored channel fails
  closed). There is no cross-context exception: a DM-only actor never enters
  a guild roster even when both audiences share a VCMERGE owner.

* **Handles** come only from the durable per-audience assignment store.
  Construction fetches assignments for the already policy-derived actor set
  and requests allocation for unassigned actors in deterministic
  ``(first_seen_sort_key, actor_id)`` order. A store that cannot prove
  durable handles yields no roster at all — identity fails closed while
  retrieval is untouched.

* **Presentation** is a fixed, versioned wrapper around one standard-JSON
  payload. Every scalar is encoded atomically and angle brackets are emitted
  as ``\\u003c``/``\\u003e`` escapes, so a malicious display name cannot add
  an entry, close the wrapper, or open a new system section. No actor id
  appears anywhere in the rendered block.

* **The snapshot is immutable.** ``snapshot_id`` is created exactly once per
  request and assigned into the request's ``SpeakerRetrievalContext`` at that
  single construction point. Token-cap fitting and hard-cap eviction drop
  whole least-recent entries and produce a replacement snapshot with the SAME
  id, so a rendered roster and any schema built from the surviving entries
  can never disagree.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from typing import Callable

from ..types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    RESERVED_SPEAKER_HANDLES,
    SpeakerHandleCandidate,
    SpeakerRetrievalContext,
    SpeakerRosterEntry,
    SpeakerRosterSnapshot,
    is_valid_speaker_handle,
    normalize_speaker_handle_base,
)

logger = logging.getLogger(__name__)

SPEAKER_ROSTER_WRAPPER_VERSION = 1

# Fixed wrapper. Entry names are untrusted derived memory; they appear only
# as JSON scalars inside the payload line, never in these constants, tool
# descriptions, YAML, or any ad hoc prose.
_ROSTER_OPEN = f'<speaker-roster version="{SPEAKER_ROSTER_WRAPPER_VERSION}">'
_ROSTER_CLOSE = "</speaker-roster>"

# Default actor cap for one roster snapshot.
SPEAKER_ROSTER_ACTOR_CAP = 12

# How many most-recent physical rows one membership scan may read. Bounded so
# roster construction cannot become a full-table read; an actor whose latest
# admissible row is older than the window is simply not presented.
_DEFAULT_ROSTER_SCAN_LIMIT = 400


@dataclasses.dataclass(frozen=True)
class SpeakerRosterBuild:
    """Result of one roster construction.

    ``speaker_context`` is the request's context with ``roster_snapshot_id``
    assigned — the caller carries it forward as the request context. All
    fields are empty/None when no roster was emitted.
    """

    text: str = ""
    tokens: int = 0
    snapshot: SpeakerRosterSnapshot | None = None
    speaker_context: SpeakerRetrievalContext | None = None


_EMPTY_BUILD = SpeakerRosterBuild()


def _handle_presentable(handle: object) -> bool:
    """A handle a human roster entry may present.

    Grammar validity comes from the shared protocol check; the reserved
    engine identities are additionally rejected because a human entry
    carrying ``assistant`` would be an identity forgery even though the
    string is grammatically a handle.
    """
    if not isinstance(handle, str) or not handle:
        return False
    if handle in RESERVED_SPEAKER_HANDLES:
        return False
    return is_valid_speaker_handle(handle)


def render_speaker_roster(snapshot: SpeakerRosterSnapshot | None) -> str:
    """Render the fixed wrapper for *snapshot*, or ``""`` for no entries.

    The payload is one standard-JSON line: handles, audience-scoped names,
    and the truncation boolean — nothing else, and no actor ids. JSON
    escaping alone is NOT enough: the encoder leaves ``<`` and ``>``
    untouched, so a name containing a literal ``</speaker-roster>`` would
    close the wrapper. Angle brackets are therefore emitted as ``\\u003c`` /
    ``\\u003e`` escapes: a JSON parser round-trips the exact name while the
    rendered characters can no longer terminate the wrapper.
    """
    if snapshot is None or not snapshot.entries:
        return ""
    payload = json.dumps(
        {
            "speakers": [
                {"handle": e.handle, "name": e.name} for e in snapshot.entries
            ],
            "truncated": bool(snapshot.truncated),
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{_ROSTER_OPEN}\n{payload}\n{_ROSTER_CLOSE}"


def evict_least_recent(
    snapshot: SpeakerRosterSnapshot,
) -> SpeakerRosterSnapshot:
    """Drop the least-recent whole entry, keeping the SAME snapshot id.

    Entries are ordered most recent first, so the suffix entry goes. The
    replacement is marked truncated; scalars are never cut. Used by the
    hard-cap rebuild so the surviving snapshot stays the single source for
    both the rendered roster and any schema enum.
    """
    return dataclasses.replace(
        snapshot, entries=snapshot.entries[:-1], truncated=True,
    )


def fit_snapshot_to_tokens(
    snapshot: SpeakerRosterSnapshot,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> tuple[str, int, SpeakerRosterSnapshot | None]:
    """Enforce the wrapper-inclusive token cap by whole-entry eviction.

    Returns ``(text, tokens, surviving_snapshot)``. Entries are dropped
    least-recent first with the snapshot id preserved; no handle or name is
    ever truncated. If the fixed wrapper alone does not fit — no entries
    survive — nothing is emitted and the caller must expose no dynamic
    speaker parameter.
    """
    current: SpeakerRosterSnapshot | None = snapshot
    while current is not None and current.entries:
        text = render_speaker_roster(current)
        tokens = token_counter(text)
        if tokens <= max_tokens:
            return text, tokens, current
        current = evict_least_recent(current)
        if not current.entries:
            current = None
    return "", 0, None


def build_speaker_roster(
    store: object,
    *,
    speaker_context: SpeakerRetrievalContext | None,
    token_counter: Callable[[str], int],
    max_tokens: int,
    actor_cap: int = SPEAKER_ROSTER_ACTOR_CAP,
    scan_limit: int = _DEFAULT_ROSTER_SCAN_LIMIT,
) -> SpeakerRosterBuild:
    """Construct the request's immutable roster snapshot and rendered block.

    The gate check belongs to the caller: this function assumes the roster
    feature is enabled and performs the membership scan, handle resolution,
    snapshot creation, and token-cap fitting. Every failure mode — ineligible
    context, no store, scan failure, no admissible actors, missing or broken
    handle protocol, invalid handle, wrapper that cannot fit — returns the
    empty build: no roster, no snapshot, no context replacement. Identity
    fails closed; ordinary retrieval is untouched.
    """
    if speaker_context is None or not getattr(speaker_context, "eligible", False):
        return _EMPTY_BUILD
    owner = speaker_context.owner_conversation_id or ""
    audience = speaker_context.audience_conversation_id or ""
    if not owner or not audience or store is None:
        return _EMPTY_BUILD
    if max_tokens <= 0 or actor_cap <= 0:
        return _EMPTY_BUILD

    try:
        rows = store.get_recent_canonical_turns(owner, limit=int(scan_limit))
    except Exception:
        logger.debug(
            "speaker roster membership scan failed; no roster emitted",
            exc_info=True,
        )
        return _EMPTY_BUILD

    channel = speaker_context.audience_channel_id or ""
    # actor -> {last: (sort_key, ctid), first: (sort_key, ctid), label: str}
    members: dict[str, dict] = {}
    try:
        for row in rows or ():
            actor = getattr(row, "sender_actor_id", "") or ""
            if not actor:
                continue
            if not (getattr(row, "user_content", "") or "").strip():
                # Not a human user lane. An assistant row never contributes
                # membership even if a stored actor id survives on it.
                continue
            if (getattr(row, "audience_conversation_id", "") or "") != audience:
                continue
            version = int(getattr(row, "audience_attribution_version", 0) or 0)
            if version != AUDIENCE_ATTRIBUTION_VERSION:
                continue
            if channel:
                row_channel = getattr(row, "origin_channel_id", "") or ""
                if row_channel != channel:
                    # Includes the empty stored channel: fails closed.
                    continue
            key = (
                float(getattr(row, "sort_key", 0.0) or 0.0),
                getattr(row, "canonical_turn_id", "") or "",
            )
            entry = members.get(actor)
            if entry is None:
                members[actor] = {
                    "last": key,
                    "first": key,
                    "label": (getattr(row, "sender", "") or "").strip(),
                }
            else:
                if key > entry["last"]:
                    entry["last"] = key
                    entry["label"] = (getattr(row, "sender", "") or "").strip()
                if key < entry["first"]:
                    entry["first"] = key
    except Exception:
        logger.debug(
            "speaker roster admission failed; no roster emitted", exc_info=True,
        )
        return _EMPTY_BUILD

    if not members:
        return _EMPTY_BUILD

    # Most recent admissible physical row first; the physical row key and the
    # actor key are the deterministic tiebreakers (stable sort preserves the
    # actor-ascending pre-order for equal recency keys).
    ordered = sorted(members)
    ordered.sort(key=lambda actor: members[actor]["last"], reverse=True)

    truncated = len(ordered) > actor_cap
    ordered = ordered[:actor_cap]

    # Capture the audience lifecycle epoch BEFORE allocation so the store can
    # fence the insert against delete/resurrect, and record it on the
    # snapshot so a later lifecycle change is detectable as staleness.
    epoch = 0
    epoch_getter = getattr(store, "get_lifecycle_epoch", None)
    if callable(epoch_getter):
        try:
            epoch = int(epoch_getter(audience) or 0)
        except Exception:
            epoch = 0

    handles = _resolve_handles(
        store,
        tenant_id=speaker_context.tenant_id or "",
        audience_conversation_id=audience,
        owner_conversation_id=owner,
        members=members,
        ordered_actors=ordered,
        expected_lifecycle_epoch=epoch,
    )
    if handles is None:
        return _EMPTY_BUILD

    snapshot_id = uuid.uuid4().hex
    entries = tuple(
        SpeakerRosterEntry(
            handle=handles[actor],
            name=members[actor]["label"],
            actor_id=actor,
        )
        for actor in ordered
    )
    snapshot = SpeakerRosterSnapshot(
        snapshot_id=snapshot_id,
        entries=entries,
        truncated=truncated,
        tenant_id=speaker_context.tenant_id or "",
        audience_conversation_id=audience,
        lifecycle_epoch=epoch,
    )

    text, tokens, surviving = fit_snapshot_to_tokens(
        snapshot, max_tokens, token_counter,
    )
    if surviving is None:
        return _EMPTY_BUILD

    # Single construction point: the request's context carries the snapshot
    # id from here on. No second authority context is created — this is the
    # same frozen object with one field assigned.
    bound_context = dataclasses.replace(
        speaker_context, roster_snapshot_id=snapshot_id,
    )
    return SpeakerRosterBuild(
        text=text,
        tokens=tokens,
        snapshot=surviving,
        speaker_context=bound_context,
    )


def _resolve_handles(
    store: object,
    *,
    tenant_id: str,
    audience_conversation_id: str,
    owner_conversation_id: str,
    members: dict[str, dict],
    ordered_actors: list[str],
    expected_lifecycle_epoch: int = 0,
) -> dict[str, str] | None:
    """Fetch durable assignments and allocate the missing ones.

    Returns ``actor_id -> handle`` for every actor in *ordered_actors*, or
    ``None`` when the store cannot prove a durable, grammar-valid handle for
    each of them. Fetch and allocation operate only on the already
    policy-derived actor set; the assignment table is never enumerated to
    discover participants. A backend that does not claim durable handle
    support builds no roster: process-local handles would be unstable
    between workers and are forbidden.
    """
    supports = getattr(store, "supports_speaker_handles", None)
    try:
        if not callable(supports) or not supports():
            return None
    except Exception:
        return None
    fetch = getattr(store, "get_speaker_handles", None)
    if not callable(fetch):
        return None
    try:
        assignments = fetch(
            tenant_id, audience_conversation_id, list(ordered_actors),
        ) or []
    except Exception:
        logger.debug(
            "speaker handle fetch failed; no roster emitted", exc_info=True,
        )
        return None
    assigned: dict[str, str] = {
        getattr(a, "actor_id", ""): getattr(a, "handle", "")
        for a in assignments
    }

    missing = [actor for actor in ordered_actors if not assigned.get(actor)]
    if missing:
        allocate = getattr(store, "allocate_speaker_handles", None)
        if not callable(allocate):
            return None
        # Deterministic allocation order: (first_seen_sort_key, actor_id).
        missing.sort(key=lambda actor: (members[actor]["first"], actor))
        candidates = [
            SpeakerHandleCandidate(
                actor_id=actor,
                normalized_base=normalize_speaker_handle_base(
                    members[actor]["label"],
                ),
                first_seen_sort_key=members[actor]["first"][0],
            )
            for actor in missing
        ]
        try:
            allocated = allocate(
                tenant_id,
                audience_conversation_id,
                candidates,
                owner_conversation_id=owner_conversation_id,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
            ) or []
        except Exception:
            logger.debug(
                "speaker handle allocation failed; no roster emitted",
                exc_info=True,
            )
            return None
        for a in allocated:
            actor = getattr(a, "actor_id", "")
            if actor:
                assigned[actor] = getattr(a, "handle", "")

    resolved: dict[str, str] = {}
    for actor in ordered_actors:
        handle = assigned.get(actor, "")
        if not _handle_presentable(handle):
            # A missing, malformed, or reserved handle means the durable
            # assignment state cannot be trusted; no roster is safer than a
            # roster the validator would disagree with.
            return None
        resolved[actor] = handle
    if len(set(resolved.values())) != len(resolved):
        # Duplicate handles within one audience violate the assignment
        # relation's uniqueness; fail closed.
        return None
    return resolved
