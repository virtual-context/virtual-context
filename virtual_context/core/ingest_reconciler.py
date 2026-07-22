"""Canonical turn ingest reconciler for proxy and REST paths."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ..core.canonical_turns import (
    CanonicalIngestResult,
    HASH_VERSION,
    build_anchor_index,
    compute_anchor_hash,
    compute_turn_hash_from_raw,
    default_sort_key,
    generate_canonical_turn_id,
    utcnow_iso,
)
from ..core.semantic_search import SemanticSearchManager
from ..core.store import ContextStore
from ..types import (
    FactSignal,
    CanonicalTurnRow,
    IngestBatchRecord,
    TurnTagEntry,
    get_actor_id,
    get_origin_channel,
    get_sender_name,
)


@dataclass
class _Alignment:
    existing_start: int
    incoming_start: int
    overlap_len: int
    window_size: int
    merge_mode: str


logger = logging.getLogger(__name__)

# An assistant physical row never receives a human reply edge.
_EMPTY_REPLY_EDGE: dict = {
    "source_message_id": "",
    "reply_target_message_id": "",
    "reply_subject_actor_id": "",
    "reply_subject_label": "",
    "reply_target_body": "",
    "reply_attribution_version": 0,
    "audience_conversation_id": "",
    "audience_attribution_version": 0,
}

# The reply-edge columns, in one place, so the CAS and the full-row rewrites
# cannot drift apart. A full-row upsert defaults every omitted column away, so
# a rewrite that forgets one of these silently erases the edge.
_REPLY_EDGE_FIELDS: tuple[str, ...] = tuple(_EMPTY_REPLY_EDGE)


def _ordered_channel(metadata: dict | None) -> tuple[str, str]:
    """Audience channel from the ORDERED first-conversation-info snapshot.

    ``get_origin_channel`` reads the last-wins merged dict, which a member's
    own typed block can overwrite. That is tolerable for display and it is what
    the shipped channel derivation does, but it is not tolerable for the
    audience boundary that gates whose memory is surfaced. Policy-grade
    provenance reads the ordered snapshot only.
    """
    from ..types import get_current_conversation_info, get_origin_channel

    current = get_current_conversation_info(metadata)
    if not current:
        return "", ""
    return get_origin_channel({"conversation info": current})


def _has_reply_edge(edge: dict) -> bool:
    """Does this edge carry anything worth making durable?

    A version stamp alone counts: it is what distinguishes a row that WAS
    examined and honestly resolved to nothing from one that was never looked
    at, and that distinction is what keeps backfill idempotent.
    """
    if any(int(edge.get(name) or 0) > 0 for name in (
        "reply_attribution_version", "audience_attribution_version",
    )):
        return True
    return any(
        (edge.get(name) or "").strip()
        for name in _REPLY_EDGE_FIELDS
        if not name.endswith("_version")
    )


def _row_reply_edge(row: "CanonicalTurnRow") -> dict:
    """The stored edge, re-supplied verbatim on a full-row rewrite.

    Reads through the column default so a row double that predates the reply
    edge degrades to "no edge" instead of raising on an unrelated write path.
    """
    return {
        name: (
            getattr(row, name, default)
            if getattr(row, name, default) is not None
            else default
        )
        for name, default in _EMPTY_REPLY_EDGE.items()
    }


class IngestReconciler:
    """Merges inbound turns into canonical turn storage."""

    def __init__(self, store: ContextStore, semantic: SemanticSearchManager) -> None:
        self._store = store
        self._semantic = semantic

    def ingest_single(
        self,
        conversation_id: str,
        *,
        user_content: str,
        assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
        primary_tag: str = "_general",
        tags: list[str] | None = None,
        session_date: str = "",
        sender: str = "",
        user_sender: str | None = None,
        assistant_sender: str | None = None,
        user_origin_channel_id: str = "",
        user_origin_channel_label: str = "",
        assistant_origin_channel_id: str = "",
        assistant_origin_channel_label: str = "",
        # Separate per-role actor arguments, never one logical value: a single
        # ``sender_actor_id`` would smear the human speaker onto the assistant
        # row exactly as the shipped single ``sender`` argument does below.
        user_sender_actor_id: str = "",
        assistant_sender_actor_id: str = "",
        # The user half's durable reply edge (reply lanes + proved audience),
        # derived by the caller from the live message metadata. Role-local:
        # the assistant row never receives one. None degrades to the empty
        # edge, so a caller that cannot prove provenance writes exactly what
        # this path always wrote.
        user_reply_edge: dict | None = None,
        fact_signals: list[FactSignal] | None = None,
        code_refs: list[dict] | None = None,
        expected_lifecycle_epoch: int | None = None,
        ) -> CanonicalIngestResult:
        # ``sender`` is the legacy logical-turn argument and remains the
        # fallback for existing callers. New callers can preserve the physical
        # row contract by supplying role-local values; a human sender must not
        # be newly stamped onto the assistant half.
        resolved_user_sender = sender if user_sender is None else user_sender
        resolved_assistant_sender = sender if assistant_sender is None else assistant_sender
        edge = dict(_EMPTY_REPLY_EDGE)
        if user_reply_edge:
            edge.update({
                name: user_reply_edge[name]
                for name in _REPLY_EDGE_FIELDS
                if name in user_reply_edge
            })
        with self._conversation_merge_lock(conversation_id):
            existing = self._store.get_all_canonical_turns(conversation_id)
            prepared = [
                self._prepare_message_row(
                    conversation_id,
                    role="user",
                    content=user_content,
                    raw_content=user_raw_content,
                    primary_tag=primary_tag,
                    tags=tags,
                    session_date=session_date,
                    sender=resolved_user_sender,
                    origin_channel_id=user_origin_channel_id,
                    origin_channel_label=user_origin_channel_label,
                    sender_actor_id=user_sender_actor_id,
                    fact_signals=fact_signals,
                    code_refs=code_refs,
                    **edge,
                ),
                self._prepare_message_row(
                    conversation_id,
                    role="assistant",
                    content=assistant_content,
                    raw_content=assistant_raw_content,
                    primary_tag=primary_tag,
                    tags=tags,
                    session_date=session_date,
                    sender=resolved_assistant_sender,
                    origin_channel_id=assistant_origin_channel_id,
                    origin_channel_label=assistant_origin_channel_label,
                    sender_actor_id=assistant_sender_actor_id,
                    fact_signals=fact_signals,
                    code_refs=code_refs,
                ),
            ]
            # Same resolution the batch path runs: a reply target named by the
            # edge may resolve to a subject actor already stored in this
            # conversation. Rows without a target skip in O(1), so the empty
            # edge costs nothing here.
            if _has_reply_edge(edge):
                self._resolve_reply_subjects(conversation_id, prepared)
            if len(existing) >= len(prepared):
                recent_window = existing[-min(5, len(existing)):]
                for window_start in range(0, len(recent_window) - len(prepared) + 1):
                    recent = recent_window[window_start:window_start + len(prepared)]
                    if (
                        [row.turn_hash for row in recent] == [row.turn_hash for row in prepared]
                        and all(self._seen_recently(row.last_seen_at or "") for row in recent)
                    ):
                        for idx, row in enumerate(prepared):
                            existing_row = recent[idx]
                            row.canonical_turn_id = existing_row.canonical_turn_id
                            row.sort_key = existing_row.sort_key
                            row.first_seen_at = existing_row.first_seen_at or row.first_seen_at
                            row.last_seen_at = utcnow_iso()
                            self._preserve_existing_enrichment(row, existing_row)
                            self._write_turn(
                                row,
                                turn_number=self._ordinal_for_row(existing, existing_row.canonical_turn_id),
                                first_seen_at=row.first_seen_at,
                                last_seen_at=row.last_seen_at,
                            )
                        self._refresh_persisted_anchors(conversation_id)
                        return CanonicalIngestResult(
                            merge_mode="exact_resend",
                            turns_written=0,
                            turns_matched=len(prepared),
                            turns_appended=0,
                            turns_prepended=0,
                            turns_inserted=0,
                            rows=recent,
                        )
            # Prepare-then-ingest flow: the pair's user half is normally
            # already persisted as the conversation's LAST row by the
            # preceding payload reconcile — only the assistant half is
            # new. The general aligner cannot see this: a 2-row incoming
            # fragment has no ≥3-row anchor window and short-overlap
            # matching is disallowed here, so alignment failed and
            # ``no_overlap_append`` duplicated the user row. The duplicate
            # then scrambled every later payload alignment (mid-insertions
            # of already-present content), broke strict pair tagging, and
            # progressively exhausted sort-key gaps at the insertion
            # point. Anchor on the tail row's hash instead: mirror its
            # identity (no rewrite — the fast-skip contract) and append
            # only the assistant row.
            if existing:
                user_row, assistant_row = prepared
                tail = existing[-1]
                if tail.turn_hash == user_row.turn_hash:
                    user_row.canonical_turn_id = tail.canonical_turn_id
                    user_row.sort_key = tail.sort_key
                    user_row.source_batch_id = tail.source_batch_id
                    user_row.first_seen_at = tail.first_seen_at or user_row.first_seen_at
                    user_row.last_seen_at = tail.last_seen_at or user_row.last_seen_at
                    # The tail row is mirrored, never rewritten, so a channel
                    # value that only became derivable on this call would live
                    # in memory alone. Fence the CAS on the epoch the CALLER
                    # entered with: reading the epoch here would observe a
                    # conversation resurrected mid-flight and happily write
                    # this stale provenance into its new lifecycle.
                    tail_candidate_id = (user_row.origin_channel_id or "").strip()
                    tail_candidate_label = (user_row.origin_channel_label or "").strip()
                    tail_fills_id = (
                        tail_candidate_id and not (tail.origin_channel_id or "").strip()
                    )
                    tail_fills_label = (
                        tail_candidate_label
                        and not (tail.origin_channel_label or "").strip()
                    )
                    if tail.canonical_turn_id and (tail_fills_id or tail_fills_label):
                        self._upgrade_empty_channels(
                            conversation_id,
                            {
                                tail.canonical_turn_id: (
                                    tail_candidate_id if tail_fills_id else "",
                                    tail_candidate_label if tail_fills_label else "",
                                ),
                            },
                            expected_lifecycle_epoch=expected_lifecycle_epoch,
                        )
                    tail_candidate_sender = (user_row.sender or "").strip()
                    if (
                        tail.canonical_turn_id
                        and tail_candidate_sender
                        and not (tail.sender or "").strip()
                        and (tail.user_content or "").strip()
                    ):
                        self._upgrade_empty_senders(
                            conversation_id,
                            {tail.canonical_turn_id: tail_candidate_sender},
                            expected_lifecycle_epoch=expected_lifecycle_epoch,
                        )
                    # Same reasoning for the actor id, but with the sender role
                    # rule: the tail row here is the USER half, so it may carry
                    # one. The caller's entry epoch fences it for the same
                    # mid-flight-resurrection reason as the channel CAS above.
                    tail_candidate_actor = (user_row.sender_actor_id or "").strip()
                    if (
                        tail.canonical_turn_id
                        and tail_candidate_actor
                        and not (tail.sender_actor_id or "").strip()
                        and (tail.user_content or "").strip()
                    ):
                        self._upgrade_empty_actors(
                            conversation_id,
                            {tail.canonical_turn_id: tail_candidate_actor},
                            expected_lifecycle_epoch=expected_lifecycle_epoch,
                        )
                    tail_candidate_edge = {
                        name: getattr(user_row, name)
                        for name in _REPLY_EDGE_FIELDS
                    }
                    if tail.canonical_turn_id and _has_reply_edge(tail_candidate_edge):
                        self._upgrade_empty_reply_roles(
                            conversation_id,
                            {tail.canonical_turn_id: tail_candidate_edge},
                            expected_lifecycle_epoch=expected_lifecycle_epoch,
                        )
                    self._preserve_existing_enrichment(user_row, tail)
                    now = utcnow_iso()
                    batch_id = generate_canonical_turn_id()
                    assistant_row.canonical_turn_id = generate_canonical_turn_id()
                    assistant_row.sort_key = float(tail.sort_key) + 1000.0
                    assistant_row.source_batch_id = batch_id
                    assistant_row.last_seen_at = now
                    self._write_turn(assistant_row, turn_number=len(existing))
                    recompute_groups = getattr(
                        self._store, "recompute_canonical_turn_groups", None,
                    )
                    if callable(recompute_groups):
                        try:
                            recompute_groups(conversation_id)
                        except Exception:
                            logger.warning(
                                "CANONICAL_TURN_GROUP_RECOMPUTE_FAILED: conv=%s",
                                conversation_id[:12],
                                exc_info=True,
                            )
                    batch = self._save_batch(
                        conversation_id,
                        raw_turn_count=len(prepared),
                        merge_mode="tail_append",
                        first_turn_hash=prepared[0].turn_hash,
                        last_turn_hash=prepared[-1].turn_hash,
                        turns_matched=1,
                        turns_appended=1,
                        turns_prepended=0,
                        turns_inserted=0,
                        batch_id=batch_id,
                    )
                    self._refresh_persisted_anchors(conversation_id)
                    return CanonicalIngestResult(
                        merge_mode="tail_append",
                        turns_written=1,
                        turns_matched=1,
                        turns_appended=1,
                        turns_prepended=0,
                        turns_inserted=0,
                        batch=batch,
                        rows=[user_row, assistant_row],
                    )
            return self._ingest_prepared_turns_locked(
                conversation_id,
                prepared_turns=prepared,
                raw_turn_count=len(prepared),
                existing=existing,
                allow_short_overlap=False,
            )

    def ingest_batch(
        self,
        conversation_id: str,
        *,
        body: dict,
        fmt: Any,
        expected_lifecycle_epoch: int,
        source_conversation_key: str = "",
        source_audience_conversation_id: str = "",
        current_user_metadata: dict | None = None,
    ) -> CanonicalIngestResult:
        from ..proxy.formats import extract_ingestible_messages

        entries, _stats = extract_ingestible_messages(
            body,
            fmt,
            mode="ingest",
            current_user_metadata=current_user_metadata,
        )
        # The platform segment of an actor id lives only in the RAW caller key.
        # ``conversation_id`` here is already alias-resolved, so after VCATTACH
        # it can be a UUID that names no platform. A caller that supplied a raw
        # key is trusted with it even when it is unparseable: silently falling
        # back to the resolved id would misattribute the platform.
        actor_key = (source_conversation_key or "").strip() or conversation_id
        # The AUDIENCE is a different question from the actor platform, and it
        # is deliberately not derived from the raw caller key. A caller-supplied
        # route is only a claim; it becomes the audience only after the
        # tenant-scoped resolver has PROVED it is the owner itself or a retained
        # alias to the owner. An unproved route stays empty, which makes the row
        # policy-ineligible — the honest outcome. Substituting the resolved owner
        # instead is what would let a DM request through a merged alias inherit
        # guild-origin influence.
        audience_id = (source_audience_conversation_id or "").strip()
        prepared: list[CanonicalTurnRow] = []
        for message in entries:
            # Channel is source provenance, not speaker attribution, so it is
            # derived independently per physical entry and BOTH roles may
            # carry it. An assistant entry gets values only when its own
            # selected text exposed the envelope; the paired user's values are
            # never copied onto it.
            channel_id, channel_label = get_origin_channel(message.metadata)
            is_user = message.role == "user"
            edge = (
                self._derive_reply_edge(message, actor_key, audience_id)
                if is_user
                else _EMPTY_REPLY_EDGE
            )
            if is_user:
                # Audience provenance for a user row comes from the ORDERED
                # first-conversation-info snapshot, not the last-wins merged
                # dict. The channel is a privacy boundary, so a member's own
                # typed block must not be able to choose the channel their
                # memory is filtered against.
                snapshot_id, snapshot_label = _ordered_channel(message.metadata)
                if snapshot_id or snapshot_label:
                    channel_id = snapshot_id or channel_id
                    channel_label = snapshot_label or channel_label
            prepared.append(
                self._prepare_message_row(
                    conversation_id,
                    role=message.role,
                    content=message.content,
                    raw_content=message.raw_content,
                    # Sender is speaker attribution, so only a user entry may
                    # carry it. ``extract_ingestible_messages`` has already
                    # parsed the leading labeled-JSON envelope into
                    # ``Message.metadata``; the name exists nowhere else once
                    # the envelope is stripped.
                    sender=(
                        (get_sender_name(message.metadata) or "")
                        if is_user
                        else ""
                    ),
                    origin_channel_id=channel_id,
                    origin_channel_label=channel_label,
                    # Actor follows the SENDER role rule, not the channel one:
                    # it is speaker attribution, so an assistant row is never
                    # newly labeled with a human actor.
                    sender_actor_id=(
                        get_actor_id(message.metadata, actor_key)
                        if is_user
                        else ""
                    ),
                    # The reply edge is role-local for the same reason: an
                    # assistant row never receives a human reply edge.
                    **edge,
                )
            )
        # Resolution runs after every physical row in this batch is prepared,
        # so a reply to a message that arrived in the SAME payload can still
        # link by its source id.
        self._resolve_reply_subjects(conversation_id, prepared)
        result = self.ingest_prepared_turns(
            conversation_id,
            prepared_turns=prepared,
            raw_turn_count=len(prepared),
            expected_lifecycle_epoch=expected_lifecycle_epoch,
        )
        # A profile is an observation cache, not a side effect of filling an
        # empty canonical column. Repeat sightings must still advance
        # last_seen_at and may refresh the presentation name, including exact
        # resends and overlap fast-skips whose actor was already durable.
        upsert_profile = getattr(
            self._store, "upsert_actor_profile_from_turn", None,
        )
        if callable(upsert_profile):
            from ..types import get_actor_display_name

            for message, row in zip(entries, prepared, strict=False):
                actor_id = (row.sender_actor_id or "").strip()
                if message.role != "user" or not actor_id:
                    continue
                try:
                    upsert_profile(
                        conversation_id,
                        actor_id,
                        get_actor_display_name(message.metadata) or row.sender,
                        seen_at=row.last_seen_at or utcnow_iso(),
                        expected_lifecycle_epoch=expected_lifecycle_epoch,
                    )
                except Exception:
                    logger.warning(
                        "ACTOR_PROFILE_UPSERT_FAILED: conv=%s actor=%s",
                        conversation_id[:12], actor_id[:24], exc_info=True,
                    )
        return result

    @staticmethod
    def _derive_reply_edge(
        message: Any, actor_key: str, audience_conversation_id: str = "",
    ) -> dict:
        """The durable reply edge for one physical user message.

        ``reply_target_body`` is deliberately NOT merged into ``content``: it
        is the quoted person's words, and letting it into requester content is
        what makes a later distiller read their claim as the requester's own
        belief.

        A non-empty current ``reply_to_id`` is itself a reply-bearing
        observation, so it stamps the version and drives an exact row lookup
        even when no reply-target BLOCK was present. The fixtures prove
        ``reply_to_id`` is the only durable target handle the platform actually
        sends; suppressing it because a label/body block is absent would throw
        away the one signal that resolves reliably.
        """
        from ..types import (
            AUDIENCE_ATTRIBUTION_VERSION,
            REPLY_ATTRIBUTION_VERSION,
            get_current_conversation_info,
            get_reply_subject,
        )

        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        current = get_current_conversation_info(metadata)
        subject = get_reply_subject(metadata, actor_key)

        def _clean(value: object) -> str:
            return value.strip() if isinstance(value, str) else ""

        reply_to_id = _clean(current.get("reply_to_id"))
        audience = (audience_conversation_id or "").strip()

        return {
            "source_message_id": _clean(current.get("message_id")),
            "reply_target_message_id": reply_to_id or subject.target_message_id,
            "reply_subject_actor_id": subject.subject_actor_id,
            "reply_subject_label": subject.subject_label,
            "reply_target_body": subject.target_body,
            # Version the row when EITHER a reply block claimed the slot or the
            # current conversation info carries a target id.
            "reply_attribution_version": (
                subject.version
                or (REPLY_ATTRIBUTION_VERSION if reply_to_id else 0)
            ),
            # Audience provenance is only versioned when the route was actually
            # PROVED. An ordered current-conversation snapshot alone is not
            # proof of route: it is untrusted JSON from the payload.
            "audience_conversation_id": audience,
            "audience_attribution_version": (
                AUDIENCE_ATTRIBUTION_VERSION if audience else 0
            ),
        }

    def _resolve_reply_subjects(
        self,
        conversation_id: str,
        prepared: list[CanonicalTurnRow],
    ) -> None:
        """Fill ``reply_subject_actor_id`` from the referenced row, then a label.

        Deterministic and fail-closed, in the spec's order:

        1. The exact referenced physical row in the same conversation wins. It
           is checked against this batch first, then the store.
        2. A direct target ``sender_id`` with a valid platform was already
           resolved in ``get_reply_subject``.
        3. A display label resolves only when it maps to exactly ONE durable
           actor in the same audience.

        Contradiction between the row and an already-derived id is a hard
        conflict: the subject goes unresolved rather than picking one of two
        actors. Zero, multiple, or fuzzy label matches also stay unresolved.
        """
        in_batch: dict[str, list[CanonicalTurnRow]] = {}
        for candidate in prepared:
            if candidate.source_message_id and (candidate.user_content or "").strip():
                in_batch.setdefault(candidate.source_message_id, []).append(candidate)
        for row in prepared:
            if row.reply_attribution_version <= 0:
                continue
            if not (row.user_content or "").strip():
                continue

            target_id = (row.reply_target_message_id or "").strip()
            row_actor = ""
            if target_id:
                # Retain every same-batch candidate. A last-wins dict silently
                # chooses one row when a merge legitimately put duplicate
                # opaque message ids under the same owner.
                candidates = [
                    candidate for candidate in in_batch.get(target_id, [])
                    if (row.audience_conversation_id or "").strip()
                    and (candidate.audience_conversation_id or "").strip()
                    == (row.audience_conversation_id or "").strip()
                    and (
                        not (row.origin_channel_id or "").strip()
                        or (candidate.origin_channel_id or "").strip()
                        == (row.origin_channel_id or "").strip()
                    )
                ]
                target = candidates[0] if len(candidates) == 1 else None
                if target is None and not candidates:
                    target = self._find_row_by_source_message_id(
                        conversation_id,
                        target_id,
                        audience_conversation_id=row.audience_conversation_id,
                        origin_channel_id=row.origin_channel_id,
                    )
                if target is not None:
                    row_actor = (target.sender_actor_id or "").strip()
                    if not row.reply_subject_label and target.sender:
                        row.reply_subject_label = target.sender

            existing = (row.reply_subject_actor_id or "").strip()
            if row_actor and existing and row_actor != existing:
                # The referenced row and the block's own id name different
                # people. Never pick by precedence across contradictory actors.
                row.reply_subject_actor_id = ""
                continue
            if row_actor:
                row.reply_subject_actor_id = row_actor
                continue
            if existing:
                continue

            label = (row.reply_subject_label or "").strip()
            if not label:
                continue
            matches = self._find_actor_ids_by_label(
                conversation_id, label, row.audience_conversation_id,
                row.origin_channel_id,
            )
            # Exactly one durable same-audience actor, or nothing. A label that
            # names two members is not an identity, and "most recent" would be
            # a guess with a real member's name on it.
            if len(matches) == 1:
                row.reply_subject_actor_id = matches[0]

    def _find_row_by_source_message_id(
        self,
        conversation_id: str,
        source_message_id: str,
        *,
        audience_conversation_id: str = "",
        origin_channel_id: str = "",
    ) -> "CanonicalTurnRow | None":
        if not (audience_conversation_id or "").strip():
            return None
        fn = getattr(self._store, "find_canonical_turn_by_source_message_id", None)
        if not callable(fn):
            return None
        try:
            return fn(
                conversation_id,
                source_message_id,
                audience_conversation_id=audience_conversation_id,
                origin_channel_id=origin_channel_id,
            )
        except Exception:
            logger.warning(
                "REPLY_TARGET_LOOKUP_FAILED: conv=%s", conversation_id[:12],
                exc_info=True,
            )
            return None

    def _find_actor_ids_by_label(
        self,
        conversation_id: str,
        label: str,
        audience_conversation_id: str,
        origin_channel_id: str,
    ) -> list[str]:
        if not (audience_conversation_id or "").strip():
            return []
        fn = getattr(self._store, "find_actor_ids_by_display_label", None)
        if not callable(fn):
            return []
        try:
            return list(fn(
                conversation_id, label,
                audience_conversation_id=audience_conversation_id,
                origin_channel_id=origin_channel_id,
            ))
        except Exception:
            logger.warning(
                "REPLY_LABEL_LOOKUP_FAILED: conv=%s", conversation_id[:12],
                exc_info=True,
            )
            return []

    def ingest_prepared_turns(
        self,
        conversation_id: str,
        *,
        prepared_turns: list[CanonicalTurnRow],
        raw_turn_count: int,
        expected_lifecycle_epoch: int,
        allow_short_overlap: bool = True,
    ) -> CanonicalIngestResult:
        from .lifecycle_epoch import verify_epoch
        # Entry check — fail fast before acquiring the per-conversation lock.
        verify_epoch(
            conversation_id=conversation_id,
            expected=expected_lifecycle_epoch,
            observed=self._store.get_lifecycle_epoch(conversation_id),
        )
        with self._conversation_merge_lock(conversation_id):
            # Re-check inside the lock: another thread may have resurrected
            # the conversation between our entry check and lock acquisition.
            verify_epoch(
                conversation_id=conversation_id,
                expected=expected_lifecycle_epoch,
                observed=self._store.get_lifecycle_epoch(conversation_id),
            )
            existing = self._store.get_all_canonical_turns(conversation_id)
            result = self._ingest_prepared_turns_locked(
                conversation_id,
                prepared_turns=prepared_turns,
                raw_turn_count=raw_turn_count,
                existing=existing,
                allow_short_overlap=allow_short_overlap,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
            )
            # Commit-time check: a resurrect could have raced DURING our writes
            # (the Python-level merge lock doesn't serialize against external
            # delete+resurrect flows). If the epoch moved, purge the rows we
            # just wrote — they belong to the old lifecycle and must not leak
            # into the new one. We use source_batch_id (stamped on every row
            # in this call) so the rollback is surgical: only THIS call's
            # rows, no concurrent new-lifecycle writes affected.
            observed_now = self._store.get_lifecycle_epoch(conversation_id)
            if observed_now != expected_lifecycle_epoch:
                batch_id = result.batch.batch_id if result.batch else None
                if batch_id:
                    self._store.delete_canonical_turns_by_batch_id(
                        conversation_id=conversation_id,
                        batch_id=batch_id,
                    )
                from .lifecycle_epoch import LifecycleEpochMismatch
                raise LifecycleEpochMismatch(
                    conversation_id=conversation_id,
                    expected=expected_lifecycle_epoch,
                    observed=observed_now,
                )
            return result

    def _ingest_prepared_turns_locked(
        self,
        conversation_id: str,
        *,
        prepared_turns: list[CanonicalTurnRow],
        raw_turn_count: int,
        existing: list[CanonicalTurnRow],
        allow_short_overlap: bool = True,
        expected_lifecycle_epoch: int | None = None,
    ) -> CanonicalIngestResult:
        if not prepared_turns:
            logger.info(
                "INGEST_EMPTY_PAYLOAD: conv=%s raw_turn_count=%d",
                conversation_id[:12],
                raw_turn_count,
            )
            batch = self._save_batch(
                conversation_id,
                raw_turn_count=0,
                merge_mode="empty_payload",
                first_turn_hash="",
                last_turn_hash="",
                turns_matched=0,
                turns_appended=0,
                turns_prepended=0,
                turns_inserted=0,
            )
            return CanonicalIngestResult("empty_payload", 0, 0, 0, 0, 0, batch=batch, rows=[])

        alignment = self._find_alignment(
            conversation_id,
            existing,
            prepared_turns,
            allow_short_overlap=allow_short_overlap,
        )
        merge_mode = alignment.merge_mode if alignment else "no_overlap_append"
        if alignment is None and existing and prepared_turns:
            logger.warning(
                "CANONICAL_TURN_NO_ALIGNMENT: conv=%s existing=%d incoming=%d -> no_overlap_append",
                conversation_id[:12],
                len(existing),
                len(prepared_turns),
            )
        turns_written = 0
        turns_matched = 0
        turns_appended = 0
        turns_prepended = 0
        turns_inserted = 0
        batch_id = generate_canonical_turn_id()
        now = utcnow_iso()
        rows_touched: list[CanonicalTurnRow] = []

        if not existing:
            merge_mode = "no_overlap_append"
            for idx, row in enumerate(prepared_turns):
                row.canonical_turn_id = generate_canonical_turn_id()
                row.sort_key = float((idx + 1) * 1000.0)
                row.source_batch_id = batch_id
                row.last_seen_at = now
                self._write_turn(row, turn_number=idx)
                rows_touched.append(row)
                turns_written += 1
                turns_appended += 1
        elif alignment is None:
            # Fragment guard: a no-alignment append into a non-empty
            # conversation is legitimate for genuine new turns, but tiny
            # windowed-payload fragments (an 'testing' user turn without its
            # assistant reply, an orphan '[[reply_to_current]]' assistant
            # half, a system-generated 'NO_REPLY' emission) produce
            # canonical rows the tagger can never tag — the tagger matches
            # user+assistant pairs and orphan halves have no partner. Left
            # in place, these rows keep ``done < total`` forever, so phase
            # never transitions out of ``ingesting`` and the dashboard
            # badge stalls at ~99% permanently (2026-04-18 production
            # incident on conv 77f110fc — 13 orphan rows left the
            # ingestion progress wedged at 2419/2432).
            #
            # Rule: reject ``no_overlap_append`` writes when the incoming
            # payload has no complete user+assistant pair. "Complete pair"
            # is evaluated at the payload level (at least one row with
            # user content AND at least one row with assistant content),
            # not per-row, so the rule passes for any payload that
            # contains a real round-trip turn even if it also carries
            # orphan halves alongside it.
            #
            # Fresh conversations (``not existing``) skip this guard —
            # the first write of a new conversation can legitimately be a
            # lone user message with the assistant reply arriving later.
            has_user = any(
                (row.user_content or "").strip() for row in prepared_turns
            )
            has_asst = any(
                (row.assistant_content or "").strip() for row in prepared_turns
            )
            if not (has_user and has_asst):
                logger.warning(
                    "CANONICAL_TURN_FRAGMENT_REJECTED: conv=%s existing=%d "
                    "incoming=%d has_user=%s has_asst=%s — no complete "
                    "user+assistant pair, skipping persist to avoid "
                    "permanently-untag-able orphan rows.",
                    conversation_id[:12],
                    len(existing),
                    len(prepared_turns),
                    has_user,
                    has_asst,
                )
                batch = self._save_batch(
                    conversation_id,
                    raw_turn_count=raw_turn_count,
                    merge_mode="fragment_rejected",
                    first_turn_hash=prepared_turns[0].turn_hash,
                    last_turn_hash=prepared_turns[-1].turn_hash,
                    turns_matched=0,
                    turns_appended=0,
                    turns_prepended=0,
                    turns_inserted=0,
                    batch_id=batch_id,
                )
                return CanonicalIngestResult(
                    merge_mode="fragment_rejected",
                    turns_written=0,
                    turns_matched=0,
                    turns_appended=0,
                    turns_prepended=0,
                    turns_inserted=0,
                    batch=batch,
                    rows=[],
                )
            start_key = default_sort_key(existing)
            for idx, row in enumerate(prepared_turns):
                row.canonical_turn_id = generate_canonical_turn_id()
                row.sort_key = start_key + (1000.0 * idx)
                row.source_batch_id = batch_id
                row.last_seen_at = now
                self._write_turn(row, turn_number=len(existing) + idx)
                rows_touched.append(row)
                turns_written += 1
                turns_appended += 1
        else:
            overlap_existing = existing[alignment.existing_start:alignment.existing_start + alignment.overlap_len]
            overlap_incoming = prepared_turns[alignment.incoming_start:alignment.incoming_start + alignment.overlap_len]
            # Append-path fast-skip: overlap rows are already persisted with
            # canonically-correct content (``_find_alignment`` proved the
            # turn_hash matches). Re-writing them was a 164 s bottleneck on
            # append-only payloads in production (5777 redundant UPDATEs for
            # an incoming payload whose real novelty was 3 tail turns).
            #
            # We instead mirror the existing row's identity fields into the
            # in-memory ``row`` so downstream consumers (batch record,
            # rows_touched) see a consistent canonical set, and skip the DB
            # write entirely. Trade-offs:
            #
            #  * ``source_batch_id`` on the stored overlap row stays on the
            #    batch that originally wrote it. This is DESIRED: rollback
            #    on lifecycle-epoch mismatch scopes to
            #    ``source_batch_id = <this batch_id>``, so overlap rows
            #    correctly do NOT get purged when a resurrect races this
            #    batch — they belong to the prior (still-valid) batch.
            #  * ``last_seen_at`` on the stored overlap row stays at its
            #    original timestamp. It's a diagnostic field with no
            #    invariant-enforcing reader; dashboards that surface it
            #    should display ``updated_at`` (refreshed by the tagger)
            #    instead.
            #  * Enrichment upgrades (e.g. a late-arriving ``session_date``
            #    replacing an empty one) on an overlap row are skipped
            #    here. The background tagger's
            #    ``_persist_existing_canonical_rows`` path still writes
            #    fresh enrichment when it re-tags the row, so the upgrade
            #    lands on the first tag pass — not lost, just deferred.
            #
            # Prefix / suffix writes below are genuinely new rows and must
            # still happen.
            #
            # One enrichment field cannot wait for the tagger: ``sender``.
            # An already-tagged overlap row takes the tagger's hydration fast
            # path, which never rewrites the canonical row, so a sender that
            # only became derivable on this payload would never become
            # durable. Collect those rows and issue one narrow,
            # epoch-guarded compare-and-set batch instead of dropping back to
            # ``_write_turn`` for every overlap row.
            sender_upgrades: dict[str, str] = {}
            channel_upgrades: dict[str, tuple[str, str]] = {}
            actor_upgrades: dict[str, str] = {}
            reply_upgrades: dict[str, dict] = {}
            for offset, row in enumerate(overlap_incoming):
                existing_row = overlap_existing[offset]
                row.canonical_turn_id = existing_row.canonical_turn_id
                row.sort_key = existing_row.sort_key
                row.source_batch_id = existing_row.source_batch_id
                row.first_seen_at = existing_row.first_seen_at or row.first_seen_at
                row.last_seen_at = existing_row.last_seen_at or row.last_seen_at
                incoming_sender = (row.sender or "").strip()
                if (
                    incoming_sender
                    and not (existing_row.sender or "").strip()
                    and (row.user_content or "").strip()
                    and existing_row.canonical_turn_id
                ):
                    # I5: never attribute a human speaker to an assistant-only
                    # row, even when a legacy sibling row carries one.
                    sender_upgrades[existing_row.canonical_turn_id] = incoming_sender
                # Channel has no user-half restriction: an assistant row that
                # carried its own envelope legitimately owns the provenance.
                # Each field is offered only when it is newly derivable.
                candidate_id = (row.origin_channel_id or "").strip()
                candidate_label = (row.origin_channel_label or "").strip()
                fills_id = candidate_id and not (existing_row.origin_channel_id or "").strip()
                fills_label = (
                    candidate_label
                    and not (existing_row.origin_channel_label or "").strip()
                )
                if existing_row.canonical_turn_id and (fills_id or fills_label):
                    channel_upgrades[existing_row.canonical_turn_id] = (
                        candidate_id if fills_id else "",
                        candidate_label if fills_label else "",
                    )
                incoming_actor = (row.sender_actor_id or "").strip()
                if (
                    incoming_actor
                    and not (existing_row.sender_actor_id or "").strip()
                    and (row.user_content or "").strip()
                    and existing_row.canonical_turn_id
                ):
                    # Same role rule as ``sender``: never label an
                    # assistant-only row with a human actor, even when a legacy
                    # sibling row carries one.
                    actor_upgrades[existing_row.canonical_turn_id] = incoming_actor
                # The reply edge needs the same treatment for the same reason:
                # a fast-skipped overlap row is never rewritten, so an edge
                # that only became derivable on this payload would never become
                # durable. Role-local to the user half, and the store-side CAS
                # fills each column only when it is empty, so a stored edge is
                # never overwritten by a contradictory one.
                incoming_edge = _row_reply_edge(row)
                if (
                    existing_row.canonical_turn_id
                    and (row.user_content or "").strip()
                    and _has_reply_edge(incoming_edge)
                ):
                    reply_upgrades[existing_row.canonical_turn_id] = incoming_edge
                self._preserve_existing_enrichment(row, existing_row)
                rows_touched.append(row)
                turns_matched += 1

            if sender_upgrades:
                self._upgrade_empty_senders(
                    conversation_id,
                    sender_upgrades,
                    expected_lifecycle_epoch=expected_lifecycle_epoch,
                )

            if channel_upgrades:
                self._upgrade_empty_channels(
                    conversation_id,
                    channel_upgrades,
                    expected_lifecycle_epoch=expected_lifecycle_epoch,
                )

            if actor_upgrades:
                self._upgrade_empty_actors(
                    conversation_id,
                    actor_upgrades,
                    expected_lifecycle_epoch=expected_lifecycle_epoch,
                )

            if reply_upgrades:
                self._upgrade_empty_reply_roles(
                    conversation_id,
                    reply_upgrades,
                    expected_lifecycle_epoch=expected_lifecycle_epoch,
                )

            prefix = prepared_turns[:alignment.incoming_start]
            if prefix:
                left_key = existing[alignment.existing_start - 1].sort_key if alignment.existing_start > 0 else None
                right_key = existing[alignment.existing_start].sort_key
                prefix_keys = self._allocate_bounded_sort_keys(
                    conversation_id,
                    existing=existing,
                    rows_touched=rows_touched,
                    left_key=left_key,
                    right_key=right_key,
                    count=len(prefix),
                )
                for row, key in zip(prefix, prefix_keys):
                    row.canonical_turn_id = generate_canonical_turn_id()
                    row.sort_key = key
                    row.source_batch_id = batch_id
                    row.last_seen_at = now
                    self._write_turn(row, turn_number=-1)
                    rows_touched.append(row)
                    turns_written += 1
                    if merge_mode == "prefix_widening":
                        turns_prepended += 1
                    else:
                        turns_inserted += 1

            suffix = prepared_turns[alignment.incoming_start + alignment.overlap_len:]
            if suffix:
                left_idx = alignment.existing_start + alignment.overlap_len - 1
                left_key = existing[left_idx].sort_key if left_idx >= 0 else None
                next_existing_idx = alignment.existing_start + alignment.overlap_len
                right_key = existing[next_existing_idx].sort_key if next_existing_idx < len(existing) else None
                suffix_keys = self._allocate_bounded_sort_keys(
                    conversation_id,
                    existing=existing,
                    rows_touched=rows_touched,
                    left_key=left_key,
                    right_key=right_key,
                    count=len(suffix),
                )
                for row, key in zip(suffix, suffix_keys):
                    row.canonical_turn_id = generate_canonical_turn_id()
                    row.sort_key = key
                    row.source_batch_id = batch_id
                    row.last_seen_at = now
                    self._write_turn(row, turn_number=-1)
                    rows_touched.append(row)
                    turns_written += 1
                    if merge_mode == "tail_append":
                        turns_appended += 1
                    else:
                        turns_inserted += 1

        batch = self._save_batch(
            conversation_id,
            raw_turn_count=raw_turn_count,
            merge_mode=merge_mode,
            first_turn_hash=prepared_turns[0].turn_hash,
            last_turn_hash=prepared_turns[-1].turn_hash,
            turns_matched=turns_matched,
            turns_appended=turns_appended,
            turns_prepended=turns_prepended,
            turns_inserted=turns_inserted,
            batch_id=batch_id,
        )
        recompute_groups = getattr(self._store, "recompute_canonical_turn_groups", None)
        if callable(recompute_groups):
            try:
                recompute_groups(conversation_id)
            except Exception:
                logger.warning(
                    "CANONICAL_TURN_GROUP_RECOMPUTE_FAILED: conv=%s",
                    conversation_id[:12],
                    exc_info=True,
                )
        try:
            self._refresh_persisted_anchors(conversation_id)
        except Exception:
            logger.warning(
                "CANONICAL_TURN_ANCHOR_REFRESH_FAILED: conv=%s",
                conversation_id[:12],
                exc_info=True,
            )
        return CanonicalIngestResult(
            merge_mode=merge_mode,
            turns_written=turns_written,
            turns_matched=turns_matched,
            turns_appended=turns_appended,
            turns_prepended=turns_prepended,
            turns_inserted=turns_inserted,
            batch=batch,
            rows=rows_touched,
        )

    def _upgrade_empty_senders(
        self,
        conversation_id: str,
        upgrades: dict[str, str],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int:
        """Make a newly-derived sender durable on fast-skipped overlap rows.

        The store-level write is a compare-and-set on ``sender = ''`` keyed by
        ``canonical_turn_id``, so it can never overwrite a stored attribution
        and a re-run is a no-op. A store without the surface (or a failure)
        degrades to "sender not yet durable" rather than failing the ingest.
        """
        updater = getattr(self._store, "update_canonical_turn_senders_if_empty", None)
        if not callable(updater):
            return 0
        try:
            return int(updater(
                conversation_id,
                upgrades,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
            ))
        except Exception:
            logger.warning(
                "CANONICAL_TURN_SENDER_UPGRADE_FAILED: conv=%s rows=%d",
                conversation_id[:12],
                len(upgrades),
                exc_info=True,
            )
            return 0

    def _upgrade_empty_channels(
        self,
        conversation_id: str,
        upgrades: dict[str, tuple[str, str]],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int:
        """Make newly-derived channel provenance durable on fast-skipped rows.

        An already-tagged overlap row takes the tagger's hydration fast path,
        which never rewrites the canonical row, so provenance that only became
        derivable on this payload would never become durable. The store-level
        write is a per-column compare-and-set keyed by ``canonical_turn_id``,
        so it can never overwrite either stored value and a re-run is a no-op.
        A store without the surface (or a failure) degrades to "channel not
        yet durable" rather than failing the ingest.
        """
        updater = getattr(self._store, "update_canonical_turn_channels_if_empty", None)
        if not callable(updater):
            return 0
        try:
            return int(updater(
                conversation_id,
                upgrades,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
            ))
        except Exception:
            logger.warning(
                "CANONICAL_TURN_CHANNEL_UPGRADE_FAILED: conv=%s rows=%d",
                conversation_id[:12],
                len(upgrades),
                exc_info=True,
            )
            return 0

    def _upgrade_empty_actors(
        self,
        conversation_id: str,
        upgrades: dict[str, str],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int:
        """Make a newly-derived actor id durable on fast-skipped rows.

        An already-tagged overlap row takes the tagger's hydration fast path,
        which never rewrites the canonical row, so an identity that only became
        derivable on this payload would never become durable. The store-level
        write is a compare-and-set on an empty ``sender_actor_id`` keyed by
        ``canonical_turn_id``, so it can never overwrite a stored identity and
        a re-run is a no-op. A store without the surface (or a failure)
        degrades to "actor not yet durable" rather than failing the ingest.
        """
        updater = getattr(self._store, "update_canonical_turn_actors_if_empty", None)
        if not callable(updater):
            return 0
        try:
            return int(updater(
                conversation_id,
                upgrades,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
            ))
        except Exception:
            logger.warning(
                "CANONICAL_TURN_ACTOR_UPGRADE_FAILED: conv=%s rows=%d",
                conversation_id[:12],
                len(upgrades),
                exc_info=True,
            )
            return 0

    def _upgrade_empty_reply_roles(
        self,
        conversation_id: str,
        upgrades: dict[str, dict],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int:
        """Make a newly-derived reply edge durable on fast-skipped rows.

        Same shape and same reason as the actor upgrade: an already-tagged
        overlap row is never rewritten, so an edge that only became derivable
        on this payload would never land. The store-level write fills each
        column only when it is empty, so a stored edge is never overwritten and
        a re-run is a no-op. A store without the surface (or a failure)
        degrades to "edge not yet durable" rather than failing the ingest.
        """
        updater = getattr(
            self._store, "update_canonical_turn_reply_roles_if_empty", None,
        )
        if not callable(updater):
            return 0
        try:
            return int(updater(
                conversation_id,
                upgrades,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
            ))
        except Exception:
            logger.warning(
                "CANONICAL_TURN_REPLY_UPGRADE_FAILED: conv=%s rows=%d",
                conversation_id[:12],
                len(upgrades),
                exc_info=True,
            )
            return 0

    @staticmethod
    def _preserve_existing_enrichment(
        row: CanonicalTurnRow, existing_row: CanonicalTurnRow,
    ) -> None:
        # Re-ingest of an already-stored turn constructs a new prepared row
        # from the incoming payload and overwrites the stored row. When the
        # incoming payload lacks metadata that the stored row already has —
        # for example a resync batch that doesn't carry the session_date the
        # original ingestion extracted from a "[Session from ...]" header —
        # the naive overwrite clobbers good data with empty. Merge alignment
        # is content-hash only (see _find_alignment), so preserving the
        # existing enrichment here does not affect row identity or ordering.
        if not (row.session_date or "").strip() and (existing_row.session_date or "").strip():
            row.session_date = existing_row.session_date
        # turn_group_number is derived by recompute_canonical_turn_groups()
        # after writes settle. On the exact_resend path (ingest_single) we
        # return early without calling recompute, so a prepared row with the
        # default -1 would clobber an already-assigned group on the existing
        # row. Concurrent writers (proxy single + REST batch) race on this:
        # if REST wrote groups 0,0,1,1 then PROXY exact_resends rows 0,1,
        # the overwrite leaves those rows at -1 while rows 2,3 keep group 1.
        # Inheriting the existing group closes that race.
        if row.turn_group_number < 0 and existing_row.turn_group_number >= 0:
            row.turn_group_number = existing_row.turn_group_number
        # Sender follows the same one-way rule: a resend whose payload no
        # longer carries the sender envelope must never blank an attribution
        # that a previous ingest (or the tagger) already derived.
        if not (row.sender or "").strip() and (existing_row.sender or "").strip():
            row.sender = existing_row.sender
        # The two channel columns merge one-way and INDEPENDENTLY: a row can
        # hold a stored id with no label (recovered from a stable conversation
        # key) and later gain the label from a payload that carries the
        # envelope, or vice versa. Merging them as a pair would let an
        # incoming label-only derivation blank a stored id.
        if (
            not (row.origin_channel_id or "").strip()
            and (existing_row.origin_channel_id or "").strip()
        ):
            row.origin_channel_id = existing_row.origin_channel_id
        if (
            not (row.origin_channel_label or "").strip()
            and (existing_row.origin_channel_label or "").strip()
        ):
            row.origin_channel_label = existing_row.origin_channel_label
        # One-way merge, exactly like ``sender``: a resend whose payload no
        # longer carries the identity envelope cannot blank a stored actor id.
        # This also preserves a value already on an assistant row (from a
        # legacy combined row or a partially deployed database) without ever
        # newly deriving one onto it.
        if (existing_row.sender_actor_id or "").strip():
            if (
                (row.sender_actor_id or "").strip()
                and (row.sender_actor_id or "").strip()
                != (existing_row.sender_actor_id or "").strip()
            ):
                logger.warning(
                    "CANONICAL_TURN_ACTOR_CONFLICT: turn_id=%s; preserving stored actor",
                    (existing_row.canonical_turn_id or "")[:12],
                )
            row.sender_actor_id = existing_row.sender_actor_id
        # Same one-way merge for every reply-edge column, per column. A resend
        # that no longer carries the reply envelope cannot blank a stored edge,
        # and a CONTRADICTORY new value never rewrites a stored one: moving a
        # quoted claim from one member to another is the contamination this
        # design exists to prevent.
        edge_names = (
            "source_message_id", "reply_target_message_id",
            "reply_subject_actor_id", "reply_subject_label",
            "reply_target_body", "audience_conversation_id",
        )
        edge_conflict = any(
            (getattr(existing_row, name) or "").strip()
            and (getattr(row, name) or "").strip()
            and (getattr(existing_row, name) or "").strip()
            != (getattr(row, name) or "").strip()
            for name in edge_names
        )
        if edge_conflict:
            logger.warning(
                "CANONICAL_TURN_REPLY_CONFLICT: turn_id=%s; preserving stored edge",
                (existing_row.canonical_turn_id or "")[:12],
            )
        for name in edge_names:
            stored = (getattr(existing_row, name) or "").strip()
            if edge_conflict:
                setattr(row, name, getattr(existing_row, name))
            elif stored and not (getattr(row, name) or "").strip():
                setattr(row, name, getattr(existing_row, name))
        # Versions are a high-water mark, so an observed-but-unresolved row
        # stays distinguishable from one that was never looked at.
        for name in ("reply_attribution_version", "audience_attribution_version"):
            if edge_conflict:
                setattr(row, name, int(getattr(existing_row, name) or 0))
            else:
                setattr(row, name, max(
                    int(getattr(row, name) or 0),
                    int(getattr(existing_row, name) or 0),
                ))

    def _prepare_message_row(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        raw_content: str | list[dict] | None = None,
        primary_tag: str = "_general",
        tags: list[str] | None = None,
        session_date: str = "",
        sender: str = "",
        origin_channel_id: str = "",
        origin_channel_label: str = "",
        sender_actor_id: str = "",
        source_message_id: str = "",
        reply_target_message_id: str = "",
        reply_subject_actor_id: str = "",
        reply_subject_label: str = "",
        reply_target_body: str = "",
        reply_attribution_version: int = 0,
        audience_conversation_id: str = "",
        audience_attribution_version: int = 0,
        fact_signals: list[FactSignal] | None = None,
        code_refs: list[dict] | None = None,
        entry: TurnTagEntry | None = None,
    ) -> CanonicalTurnRow:
        if role == "assistant":
            user_content = ""
            assistant_content = content
        else:
            user_content = content
            assistant_content = ""
        turn_hash, norm_user, norm_asst = compute_turn_hash_from_raw(user_content, assistant_content, version=HASH_VERSION)
        if isinstance(raw_content, list):
            import json

            raw_payload = json.dumps(raw_content)
        else:
            raw_payload = raw_content
        now = utcnow_iso()
        if entry is not None:
            primary_tag = entry.primary_tag or primary_tag
            tags = list(entry.tags or [])
            session_date = entry.session_date or session_date
            sender = entry.sender or sender
            fact_signals = list(entry.fact_signals or [])
            code_refs = list(entry.code_refs or [])
        # IngestReconciler never sets ``tagged_at`` — the tagger (A27) owns
        # that write as part of the progress-bar redesign. A row starts
        # untagged and the tagger stamps it later.
        # ``covered_ingestible_entries`` defaults to 1: each canonical row
        # represents one ingestible payload entry (user/assistant side).
        # Grouped-turn ingestion in later tasks may set this higher.
        return CanonicalTurnRow(
            conversation_id=conversation_id,
            turn_number=-1,
            turn_group_number=-1,
            sort_key=0.0,
            turn_hash=turn_hash,
            hash_version=HASH_VERSION,
            normalized_user_text=norm_user,
            normalized_assistant_text=norm_asst,
            user_content=user_content,
            assistant_content=assistant_content,
            user_raw_content=raw_payload if role == "user" else None,
            assistant_raw_content=raw_payload if role == "assistant" else None,
            primary_tag=primary_tag or "_general",
            tags=list(tags or []),
            session_date=session_date or "",
            sender=sender or "",
            # Deliberately NOT sourced from ``entry``: a TurnTagEntry
            # represents a logical turn, so taking channel from it would smear
            # the user-derived value onto the assistant physical row.
            origin_channel_id=origin_channel_id or "",
            origin_channel_label=origin_channel_label or "",
            # Also deliberately NOT sourced from ``entry``: a TurnTagEntry is a
            # logical turn, so taking the actor from it would smear the human
            # speaker onto the assistant physical row.
            sender_actor_id=sender_actor_id or "",
            source_message_id=source_message_id or "",
            reply_target_message_id=reply_target_message_id or "",
            reply_subject_actor_id=reply_subject_actor_id or "",
            reply_subject_label=reply_subject_label or "",
            reply_target_body=reply_target_body or "",
            reply_attribution_version=int(reply_attribution_version or 0),
            audience_conversation_id=audience_conversation_id or "",
            audience_attribution_version=int(audience_attribution_version or 0),
            fact_signals=list(fact_signals or []),
            code_refs=list(code_refs or []),
            tagged_at=None,
            covered_ingestible_entries=1,
            first_seen_at=now,
            last_seen_at=now,
            created_at=now,
            updated_at=now,
        )

    def _find_alignment(
        self,
        conversation_id: str,
        existing: list[CanonicalTurnRow],
        incoming: list[CanonicalTurnRow],
        *,
        allow_short_overlap: bool = True,
    ) -> _Alignment | None:
        if not existing or not incoming:
            return None
        existing_hashes = [row.turn_hash for row in existing]
        incoming_hashes = [row.turn_hash for row in incoming]

        if existing_hashes == incoming_hashes:
            return _Alignment(0, 0, len(existing), 0, "exact_resend")

        if len(existing_hashes) <= len(incoming_hashes) and incoming_hashes[:len(existing_hashes)] == existing_hashes:
            return _Alignment(0, 0, len(existing_hashes), 0, "tail_append")

        if len(existing_hashes) <= len(incoming_hashes) and incoming_hashes[-len(existing_hashes):] == existing_hashes:
            return _Alignment(0, len(incoming_hashes) - len(existing_hashes), len(existing_hashes), 0, "prefix_widening")

        def _alignment_mode(existing_start: int, incoming_start: int, overlap_len: int) -> str:
            if overlap_len == len(incoming_hashes):
                return "exact_resend"
            if incoming_start == 0 and existing_start == 0:
                return "tail_append" if len(incoming_hashes) > overlap_len else "exact_resend"
            if (
                existing_start == 0
                and incoming_start > 0
                and incoming_start + overlap_len == len(incoming_hashes)
            ):
                return "prefix_widening"
            return "interior_overlap"

        if allow_short_overlap and min(len(existing_hashes), len(incoming_hashes)) < 3:
            max_overlap = min(len(existing_hashes), len(incoming_hashes))
            for overlap_len in range(max_overlap, 0, -1):
                for incoming_start in range(0, len(incoming_hashes) - overlap_len + 1):
                    incoming_slice = incoming_hashes[incoming_start:incoming_start + overlap_len]
                    for existing_start in range(0, len(existing_hashes) - overlap_len + 1):
                        if existing_hashes[existing_start:existing_start + overlap_len] != incoming_slice:
                            continue
                        return _Alignment(
                            existing_start=existing_start,
                            incoming_start=incoming_start,
                            overlap_len=overlap_len,
                            window_size=overlap_len,
                            merge_mode=_alignment_mode(existing_start, incoming_start, overlap_len),
                        )

        best: _Alignment | None = None
        for window_size in (5, 4, 3):
            existing_index = self._load_existing_anchor_index(
                conversation_id,
                existing,
                window_size,
            )
            if not existing_index:
                continue
            incoming_index = build_anchor_index(incoming, window_size)
            for digest, incoming_positions in incoming_index.items():
                existing_positions = existing_index.get(digest, [])
                if not existing_positions:
                    continue
                for incoming_start in incoming_positions:
                    for existing_start in existing_positions:
                        left = 0
                        while (
                            incoming_start - left - 1 >= 0
                            and existing_start - left - 1 >= 0
                            and incoming_hashes[incoming_start - left - 1] == existing_hashes[existing_start - left - 1]
                        ):
                            left += 1
                        right = window_size
                        while (
                            incoming_start + right < len(incoming_hashes)
                            and existing_start + right < len(existing_hashes)
                            and incoming_hashes[incoming_start + right] == existing_hashes[existing_start + right]
                        ):
                            right += 1
                        overlap_len = left + right
                        normalized_incoming_start = incoming_start - left
                        normalized_existing_start = existing_start - left
                        candidate = _Alignment(
                            existing_start=normalized_existing_start,
                            incoming_start=normalized_incoming_start,
                            overlap_len=overlap_len,
                            window_size=window_size,
                            merge_mode=_alignment_mode(
                                normalized_existing_start,
                                normalized_incoming_start,
                                overlap_len,
                            ),
                        )
                        if best is None or candidate.overlap_len > best.overlap_len or (
                            candidate.overlap_len == best.overlap_len and candidate.window_size > best.window_size
                        ):
                            best = candidate
            if best is not None:
                return best
        return None

    def _write_turn(
        self,
        row: CanonicalTurnRow,
        *,
        turn_number: int,
        first_seen_at: str | None = None,
        last_seen_at: str | None = None,
    ) -> None:
        self._store.save_canonical_turn(
            row.conversation_id,
            turn_number,
            row.user_content,
            row.assistant_content,
            user_raw_content=row.user_raw_content,
            assistant_raw_content=row.assistant_raw_content,
            primary_tag=row.primary_tag,
            tags=list(row.tags or []),
            session_date=row.session_date,
            sender=row.sender,
            fact_signals=list(row.fact_signals or []),
            code_refs=list(row.code_refs or []),
            created_at=row.created_at or first_seen_at or utcnow_iso(),
            updated_at=utcnow_iso(),
            canonical_turn_id=row.canonical_turn_id or None,
            sort_key=row.sort_key,
            turn_hash=row.turn_hash,
            hash_version=row.hash_version or HASH_VERSION,
            normalized_user_text=row.normalized_user_text,
            normalized_assistant_text=row.normalized_assistant_text,
            tagged_at=row.tagged_at,
            compacted_at=row.compacted_at,
            first_seen_at=first_seen_at or row.first_seen_at,
            last_seen_at=last_seen_at or row.last_seen_at,
            source_batch_id=row.source_batch_id or None,
            turn_group_number=row.turn_group_number,
            # The upsert overwrites omitted fields with defaults, so every
            # full-row rewrite must re-supply both channel columns and the
            # actor column.
            origin_channel_id=row.origin_channel_id,
            origin_channel_label=row.origin_channel_label,
            sender_actor_id=row.sender_actor_id,
            **_row_reply_edge(row),
        )
        resolved_turn_number = turn_number
        if turn_number < 0 and row.canonical_turn_id:
            lookup = getattr(self._store, "_lookup_ordinal_for_canonical_turn_id", None)
            if callable(lookup):
                resolved_turn_number = int(lookup(row.conversation_id, row.canonical_turn_id))
        if resolved_turn_number >= 0:
            try:
                self._semantic.embed_and_store_turn(
                    row.conversation_id,
                    resolved_turn_number,
                    canonical_turn_id=row.canonical_turn_id or None,
                    user_text=row.user_content,
                    assistant_text=row.assistant_content,
                    user_raw_content=row.user_raw_content,
                    assistant_raw_content=row.assistant_raw_content,
                    reply_target_body=row.reply_target_body or "",
                )
            except Exception:
                logger.warning(
                    "CANONICAL_TURN_EMBED_FAILED side=both conv=%s turn=%d",
                    row.conversation_id[:12],
                    resolved_turn_number,
                    exc_info=True,
                )

    def _conversation_merge_lock(self, conversation_id: str):
        locker = getattr(self._store, "conversation_reconcile", None)
        if callable(locker):
            return locker(conversation_id)
        return nullcontext()

    def _save_batch(
        self,
        conversation_id: str,
        *,
        raw_turn_count: int,
        merge_mode: str,
        first_turn_hash: str,
        last_turn_hash: str,
        turns_matched: int,
        turns_appended: int,
        turns_prepended: int,
        turns_inserted: int,
        batch_id: str | None = None,
    ) -> IngestBatchRecord:
        batch_payload = {
            "batch_id": batch_id or generate_canonical_turn_id(),
            "conversation_id": conversation_id,
            "received_at": utcnow_iso(),
            "raw_turn_count": raw_turn_count,
            "merge_mode": merge_mode,
            "turns_matched": turns_matched,
            "turns_appended": turns_appended,
            "turns_prepended": turns_prepended,
            "turns_inserted": turns_inserted,
            "first_turn_hash": first_turn_hash,
            "last_turn_hash": last_turn_hash,
        }
        batch_id = self._store.save_ingest_batch(batch_payload)
        return IngestBatchRecord(
            batch_id=batch_id,
            conversation_id=conversation_id,
            received_at=batch_payload["received_at"],
            raw_turn_count=raw_turn_count,
            merge_mode=merge_mode,
            turns_matched=turns_matched,
            turns_appended=turns_appended,
            turns_prepended=turns_prepended,
            turns_inserted=turns_inserted,
            first_turn_hash=first_turn_hash,
            last_turn_hash=last_turn_hash,
        )

    #: Minimum spacing between allocated sort keys in a bounded gap. A
    #: bounded allocation that cannot fit ``count`` keys strictly inside
    #: ``(left_key, right_key)`` at this spacing signals exhaustion so the
    #: caller can rebalance instead of colliding with the boundary rows.
    _MIN_SORT_KEY_STEP = 0.001

    def _allocate_sort_keys(
        self,
        left_key: float | None,
        right_key: float | None,
        count: int,
    ) -> list[float] | None:
        """Allocate ``count`` sort keys between the given boundaries.

        Returns ``None`` when both boundaries are set and the gap cannot
        host ``count`` keys strictly inside the open interval. The old
        behavior clamped the step to 0.001, which let allocated keys land
        ON or PAST ``right_key`` — a guaranteed violation of the
        ``UNIQUE (conversation_id, sort_key)`` constraint (when a key hit
        an existing row) or a silent ordering corruption (when a key
        overshot the boundary without colliding).
        """
        if count <= 0:
            return []
        if left_key is None and right_key is None:
            return [float((idx + 1) * 1000.0) for idx in range(count)]
        if left_key is None:
            start = float(right_key - (1000.0 * count))
            return [start + (1000.0 * idx) for idx in range(count)]
        if right_key is None:
            return [float(left_key + (1000.0 * (idx + 1))) for idx in range(count)]
        step = (right_key - left_key) / float(count + 1)
        if step < self._MIN_SORT_KEY_STEP:
            return None
        keys = [float(left_key + (step * (idx + 1))) for idx in range(count)]
        # Float-precision guard: at extreme magnitudes ``left + step`` can
        # round onto a boundary or collapse adjacent keys. Any violation of
        # strict interior ordering is exhaustion, same as a too-small gap.
        prev = float(left_key)
        for key in keys:
            if not (prev < key < right_key):
                return None
            prev = key
        return keys

    def _allocate_bounded_sort_keys(
        self,
        conversation_id: str,
        *,
        existing: list[CanonicalTurnRow],
        rows_touched: list[CanonicalTurnRow],
        left_key: float | None,
        right_key: float | None,
        count: int,
    ) -> list[float]:
        """Allocate keys, rebalancing the conversation when the gap is full.

        On exhaustion (only possible when both boundaries are set), every
        row at or beyond ``right_key`` is shifted upward to open room, the
        in-memory mirrors in ``existing`` / ``rows_touched`` are kept in
        lockstep with the DB, and allocation is retried in the widened gap.
        """
        keys = self._allocate_sort_keys(left_key, right_key, count)
        if keys is not None:
            return keys
        delta = self._open_sort_key_gap(
            conversation_id,
            existing=existing,
            rows_touched=rows_touched,
            right_key=float(right_key),
            count=count,
        )
        keys = self._allocate_sort_keys(left_key, float(right_key) + delta, count)
        if keys is None:
            raise RuntimeError(
                "sort-key allocation failed after rebalance: "
                f"conv={conversation_id[:12]} left={left_key} "
                f"right={right_key} delta={delta} count={count}"
            )
        return keys

    def _open_sort_key_gap(
        self,
        conversation_id: str,
        *,
        existing: list[CanonicalTurnRow],
        rows_touched: list[CanonicalTurnRow],
        right_key: float,
        count: int,
    ) -> float:
        """Shift every row at or beyond ``right_key`` upward by a delta.

        The delta exceeds the sort-key spread being shifted, so a
        single-statement UPDATE can never transiently collide on the
        ``UNIQUE (conversation_id, sort_key)`` constraint regardless of
        row visit order. In-memory row objects whose keys mirror the
        shifted DB rows are updated in lockstep so later allocation in
        this reconcile pass sees current boundaries.
        """
        max_key = max(row.sort_key for row in existing)
        delta = (max_key - right_key) + 1000.0 * (count + 1)
        shifter = getattr(self._store, "shift_canonical_turn_sort_keys", None)
        shifted = -1
        if callable(shifter):
            try:
                shifted = int(
                    shifter(conversation_id, min_sort_key=right_key, delta=delta)
                )
            except NotImplementedError:
                shifted = -1
        if shifted < 0:
            # Store lacks the bulk shift: per-row upserts in descending key
            # order. The delta puts every shifted key above the current
            # maximum, so each single-row write lands in free space.
            for row in sorted(
                (r for r in existing if r.sort_key >= right_key),
                key=lambda r: r.sort_key,
                reverse=True,
            ):
                row.sort_key = float(row.sort_key) + delta
                # Direct save (not _write_turn): content is unchanged, so
                # re-embedding the shifted rows would be pure waste.
                self._store.save_canonical_turn(
                    row.conversation_id or conversation_id,
                    row.turn_number,
                    row.user_content,
                    row.assistant_content,
                    user_raw_content=row.user_raw_content,
                    assistant_raw_content=row.assistant_raw_content,
                    primary_tag=row.primary_tag,
                    tags=list(row.tags or []),
                    session_date=row.session_date,
                    sender=row.sender,
                    fact_signals=list(row.fact_signals or []),
                    code_refs=list(row.code_refs or []),
                    created_at=row.created_at,
                    updated_at=utcnow_iso(),
                    canonical_turn_id=row.canonical_turn_id or None,
                    sort_key=row.sort_key,
                    turn_hash=row.turn_hash,
                    hash_version=row.hash_version or HASH_VERSION,
                    normalized_user_text=row.normalized_user_text,
                    normalized_assistant_text=row.normalized_assistant_text,
                    tagged_at=row.tagged_at,
                    compacted_at=row.compacted_at,
                    first_seen_at=row.first_seen_at,
                    last_seen_at=row.last_seen_at,
                    source_batch_id=row.source_batch_id or None,
                    turn_group_number=row.turn_group_number,
                    origin_channel_id=row.origin_channel_id,
                    origin_channel_label=row.origin_channel_label,
                    sender_actor_id=row.sender_actor_id,
                    **_row_reply_edge(row),
                )
            # Per-row path already updated ``existing`` in place; only the
            # ``rows_touched`` mirrors remain.
            for row in rows_touched:
                if row.sort_key >= right_key:
                    row.sort_key = float(row.sort_key) + delta
            logger.info(
                "SORT_KEY_GAP_REBALANCE conv=%s right_key=%s delta=%s path=per-row",
                conversation_id[:12], right_key, delta,
            )
            return delta
        for row in existing:
            if row.sort_key >= right_key:
                row.sort_key = float(row.sort_key) + delta
        for row in rows_touched:
            if row.sort_key >= right_key:
                row.sort_key = float(row.sort_key) + delta
        logger.info(
            "SORT_KEY_GAP_REBALANCE conv=%s right_key=%s delta=%s rows=%d",
            conversation_id[:12], right_key, delta, shifted,
        )
        return delta

    def _seen_recently(self, last_seen_at: str) -> bool:
        try:
            seen_at = datetime.fromisoformat(str(last_seen_at).replace("Z", "+00:00"))
        except Exception:
            if last_seen_at:
                logger.warning(
                    "CANONICAL_TURN_DEDUP_TIMESTAMP_INVALID: value=%r",
                    last_seen_at,
                )
            return False
        return datetime.now(timezone.utc) - seen_at <= timedelta(minutes=10)

    def _ordinal_for_row(self, rows: list[CanonicalTurnRow], canonical_turn_id: str) -> int:
        for idx, row in enumerate(rows):
            if row.canonical_turn_id == canonical_turn_id:
                return idx
        return -1

    def _load_existing_anchor_index(
        self,
        conversation_id: str,
        existing: list[CanonicalTurnRow],
        window_size: int,
    ) -> dict[str, list[int]]:
        loader = getattr(self._store, "get_canonical_turn_anchor_positions", None)
        if callable(loader):
            anchors = loader(conversation_id, window_size)
            if anchors:
                return anchors
        return build_anchor_index(existing, window_size)

    def _refresh_persisted_anchors(self, conversation_id: str) -> None:
        saver = getattr(self._store, "replace_canonical_turn_anchors", None)
        if not callable(saver):
            return
        rows = self._store.get_all_canonical_turns(conversation_id)
        anchors: list[tuple[int, str, str]] = []
        for window_size in (3, 4, 5):
            if len(rows) < window_size:
                continue
            for start in range(0, len(rows) - window_size + 1):
                start_turn_id = rows[start].canonical_turn_id
                if not start_turn_id:
                    continue
                anchors.append(
                    (
                        window_size,
                        compute_anchor_hash(rows, start, window_size),
                        start_turn_id,
                    )
                )
        saver(conversation_id, anchors)
