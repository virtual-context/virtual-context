"""CommunityAttributionService: explicit dependencies for community memory work."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import ActorRoster, CanonicalTurnRow

# Keep the existing operator log channel stable across the extraction.
logger = logging.getLogger("virtual_context.core.compaction_pipeline")


class CommunityAttributionService:
    QUOTE_AGENT_AUTHORED = "agent_authored"
    QUOTE_NOT_AGENT = "not_agent"
    QUOTE_IDENTITY_UNKNOWN = "agent_identity_unknown"

    def __init__(
        self,
        *,
        store,
        config,
        quote_outcomes: dict,
        quote_is_agent_output: Callable,
        record_quote_outcome: Callable,
        segment_source_ids: Callable,
    ) -> None:
        self._store = store
        self._config = config
        self._agent_quote_counts = quote_outcomes
        self._quote_is_agent_output = quote_is_agent_output
        self._record_quote_outcome = record_quote_outcome
        self._segment_source_ids = segment_source_ids

    @staticmethod
    def segment_source_ids(segment) -> tuple[list[str], bool]:
        """First-seen deduplicated source ids of a segment, and completeness.

        Completeness requires every non-empty message to carry at least one
        source id. It is deliberately not derived from ``turn_count``: topic
        grouping is noncontiguous and the session splitter can turn one source
        message into two, so a positional slice is not a row mapping.
        """
        from ...types import SOURCE_CANONICAL_TURN_IDS_KEY

        ordered: list[str] = []
        seen: set[str] = set()
        complete = True
        for message in segment.messages:
            if not (getattr(message, "content", "") or "").strip():
                continue
            ids = (getattr(message, "metadata", None) or {}).get(
                SOURCE_CANONICAL_TURN_IDS_KEY
            ) or []
            if not ids:
                complete = False
                continue
            for cid in ids:
                if cid and cid not in seen:
                    seen.add(cid)
                    ordered.append(cid)
        return ordered, complete

    @staticmethod
    def source_human_identity_keys(
        source_ids: list[str],
        physical_by_id: dict[str, "CanonicalTurnRow"],
    ) -> set[tuple[str, ...]] | None:
        """Exact human identities for a fully resolved canonical source map.

        A human row must carry its durable actor id. Display labels are
        presentation data: two people can share one and one person can change
        theirs, so a normalized-name fallback is not identity proof for a
        destructive stored-segment merge. Assistant-only sources return an
        empty set and are therefore ineligible for a merge that requires one
        exact actor+audience+channel key.
        """
        if not source_ids:
            return None

        from ...types import AUDIENCE_ATTRIBUTION_VERSION

        identities: set[tuple[str, ...]] = set()
        for canonical_id in source_ids:
            row = physical_by_id.get(canonical_id)
            if row is None:
                return None

            user_text = (getattr(row, "user_content", "") or "").strip()
            assistant_text = (getattr(row, "assistant_content", "") or "").strip()
            if not user_text and not assistant_text:
                return None
            if not user_text:
                continue

            actor_id = (getattr(row, "sender_actor_id", "") or "").strip()
            audience = (getattr(row, "audience_conversation_id", "") or "").strip()
            channel = (getattr(row, "origin_channel_id", "") or "").strip()
            attribution_version = int(getattr(row, "audience_attribution_version", 0) or 0)
            if not actor_id or not audience or attribution_version != AUDIENCE_ATTRIBUTION_VERSION:
                return None
            identities.add(("actor_scope", actor_id, audience, channel))

            # A quoted physical human is a source speaker too.  Counting the
            # reply subject prevents a requester-only key from authorizing a
            # destructive merge that silently folds another person's words
            # into the same summary.  An unresolved quoted subject fails
            # closed; assistant-quote suppression happens later at the roster
            # boundary where configured/ledger identity is available.
            quote = (getattr(row, "reply_target_body", "") or "").strip()
            if quote:
                subject_actor = (getattr(row, "reply_subject_actor_id", "") or "").strip()
                if not subject_actor:
                    return None
                identities.add(
                    (
                        "actor_scope",
                        subject_actor,
                        audience,
                        channel,
                    )
                )

        return identities

    def validated_agent_actor_ids(self, physical_by_id: dict) -> dict:
        """Configured agent identities by platform, minus any the rows refute.

        An id that has ever been an INBOUND sender is not the agent. Checked
        against stored history rather than waiting for a wrong value to speak,
        so a misconfiguration is caught before it suppresses anything.

        Rejection is PER PLATFORM: a bad discord entry must not disable a good
        telegram one. Malformed entries are dropped and never repaired -- a
        repaired identity is a guess, and a guess here deletes a person's
        words. An id too narrow leaves ghosts; an id too broad destroys a
        member's quoted text and leaves nothing behind. Those costs are not
        comparable, so anything unverifiable is discarded.
        """
        from ...types import _normalize_actor_id

        raw = getattr(self._config, "agent_actor_ids", None) or {}
        configured = {}
        for platform, user_id in raw.items():
            actor_id = _normalize_actor_id(str(platform or ""), str(user_id or ""))
            if actor_id:
                configured[str(platform or "").strip().lower()] = actor_id
        if not configured:
            return {}
        # The rows this run holds are NOT the population that can refute an
        # id. A member quoted in this batch may have spoken only in an earlier
        # one, and that member is exactly whose words a wrong identity would
        # destroy. Ask the store for every sender the conversation has ever
        # had; fall back to the loaded rows only when the backend cannot
        # answer, and say so, because a narrower check is weaker assurance and
        # must not look identical to the full one.
        senders = set()
        conversation_id = getattr(self._config, "conversation_id", "") or ""
        fn = getattr(self._store, "distinct_sender_actor_ids", None)
        if callable(fn) and conversation_id:
            try:
                senders = {str(x or "").strip() for x in (fn(conversation_id) or set())}
            except Exception:
                # Verification was attempted and failed. Partial assurance is
                # not assurance: refuse every identity rather than suppress on
                # a check that did not complete.
                logger.warning(
                    "AGENT_ACTOR_ID_UNVERIFIED conv=%s — the sender set could "
                    "not be read, so no configured identity is trusted and "
                    "nothing will be suppressed on this run.",
                    conversation_id,
                    exc_info=True,
                )
                return {}
        else:
            logger.info(
                "AGENT_ACTOR_ID_NARROW_CHECK conv=%s — backend cannot list "
                "senders; the configured identity is checked only against the "
                "rows this run holds, which is weaker.",
                conversation_id,
            )
            senders = {
                (getattr(row, "sender_actor_id", "") or "").strip()
                for row in (physical_by_id or {}).values()
            }
        senders.discard("")
        kept = {}
        for platform, actor_id in configured.items():
            if actor_id in senders:
                logger.error(
                    "AGENT_ACTOR_ID_REJECTED conv=%s platform=%s actor_id=%s — "
                    "this id appears as an INBOUND SENDER, so it is not the "
                    "agent. The comparison is disabled for this platform and "
                    "prior behaviour kept; nothing will be suppressed by it.",
                    (getattr(self._config, "conversation_id", "") or ""),
                    platform,
                    actor_id,
                )
                continue
            kept[platform] = actor_id
        return kept

    def record_quote_outcome(self, outcome: str) -> None:
        counts = getattr(self, "_agent_quote_counts", None)
        if counts is None:
            counts = {
                self.QUOTE_AGENT_AUTHORED: 0,
                self.QUOTE_NOT_AGENT: 0,
                self.QUOTE_IDENTITY_UNKNOWN: 0,
            }
            self._agent_quote_counts = counts
        counts[outcome] = counts.get(outcome, 0) + 1

    def log_quote_outcomes(self) -> None:
        """Emit the tally, INCLUDING when every count is zero.

        A guard that never ran and a guard that ran and matched nothing are
        otherwise identical in the record, and the difference is exactly what
        anyone verifying this needs.
        """
        counts = getattr(self, "_agent_quote_counts", None) or {}
        logger.info(
            "AGENT_QUOTE_OUTCOMES conv=%s agent_authored=%d not_agent=%d agent_identity_unknown=%d",
            (getattr(self._config, "conversation_id", "") or ""),
            counts.get(self.QUOTE_AGENT_AUTHORED, 0),
            counts.get(self.QUOTE_NOT_AGENT, 0),
            counts.get(self.QUOTE_IDENTITY_UNKNOWN, 0),
        )

    def quote_is_agent_output(
        self,
        *,
        channel_id: str,
        target_message_id: str,
        reply_subject_actor_id: str = "",
        agent_actor_ids: dict | None = None,
    ) -> str:
        """Classify a quoted message's authorship. Returns one of three states.

        CONTRACT CHANGE: this returned ``bool`` and documented False as
        "unknown, which includes every case where the identity was never
        reported". That conflated "not the agent" with "cannot tell", and the
        two need different remedies. Callers must now compare against
        ``QUOTE_AGENT_AUTHORED`` explicitly; anything else declines to
        suppress, but only ``QUOTE_IDENTITY_UNKNOWN`` means unevaluable.

        Two independent signals, in order of what they can reach:

        1. ``reply_subject_actor_id`` on the canonical row -- who the platform
           said authored the quoted message. Recorded at ingest, so it is
           present on HISTORICAL rows and needs no ledger.
        2. The outbound ledger -- an exact namespaced identity match. Only
           covers messages recorded since the ledger began, so it cannot
           answer for history.

        Signal 1 answers both directions; signal 2 can only ever confirm, and
        its absence is never a denial because the recorded set is partial by
        construction.
        """
        subject = (reply_subject_actor_id or "").strip()
        agents = agent_actor_ids or {}
        if agents and subject.startswith("actor:"):
            # The platform is inside the id, so no store lookup is needed. An
            # id whose platform is NOT configured is unevaluable, not a
            # mismatch -- reporting "not the agent" there would claim a
            # negative we never checked.
            parts = subject.split(":", 2)
            platform = parts[1].strip().lower() if len(parts) == 3 else ""
            expected = agents.get(platform, "")
            if expected:
                return self.QUOTE_AGENT_AUTHORED if subject == expected else self.QUOTE_NOT_AGENT
        # No subject actor recorded, or no configured agent identity. The
        # ledger can still positively confirm, but it cannot deny.
        if not channel_id or not target_message_id:
            return self.QUOTE_IDENTITY_UNKNOWN
        conversation_id = getattr(self._config, "conversation_id", "") or ""
        if not conversation_id:
            return self.QUOTE_IDENTITY_UNKNOWN
        try:
            namespace = self._store.resolve_channel_namespace(
                conversation_id=conversation_id,
                channel_id=channel_id,
            )
            if not namespace:
                return self.QUOTE_IDENTITY_UNKNOWN
            agent_scope_id, platform, account_id = namespace
            matched = bool(
                self._store.is_bot_authored_message(
                    tenant_id=getattr(self._config, "tenant_id", "") or "",
                    agent_scope_id=agent_scope_id,
                    conversation_id=conversation_id,
                    platform=platform,
                    account_id=account_id,
                    channel_id=channel_id,
                    message_id=target_message_id,
                )
            )
            return self.QUOTE_AGENT_AUTHORED if matched else self.QUOTE_IDENTITY_UNKNOWN
        except Exception:
            # An enhancement must never be able to change how a turn is filed
            # by failing. Unevaluable is the safe answer.
            logger.warning("agent-authored quote check failed", exc_info=True)
            return self.QUOTE_IDENTITY_UNKNOWN

    def build_actor_roster(
        self,
        segment,
        physical_by_id: dict,
        agent_actor_ids: dict | None = None,
    ) -> "ActorRoster":
        """Build one segment's actor roster and fact lanes from physical rows.

        Everything here comes from stored rows, never from model text or a
        positional cursor. A segment whose mapping is incomplete, or that spans
        more than one human, will attribute no fact author at all.
        """
        from ...types import (
            AUTHOR_ROLE_ASSISTANT,
            AUTHOR_ROLE_REQUESTER,
            AUTHOR_ROLE_SUBJECT,
            ActorRoster,
            FactLane,
        )

        ids, complete = self._segment_source_ids(segment)
        roster = ActorRoster(complete=complete)
        if not ids:
            roster.complete = False
            return roster

        # Index the bounded physical map once per roster. Preserve every
        # match: duplicate targets cannot establish a unique source speaker.
        targets_by_scope: dict[tuple[str, str, str], list] = {}
        for candidate in physical_by_id.values():
            message_id = (getattr(candidate, "source_message_id", "") or "").strip()
            audience = (getattr(candidate, "audience_conversation_id", "") or "").strip()
            channel = (getattr(candidate, "origin_channel_id", "") or "").strip()
            if message_id and audience and (getattr(candidate, "user_content", "") or "").strip():
                targets_by_scope.setdefault((message_id, audience, channel), []).append(candidate)

        for cid in ids:
            row = physical_by_id.get(cid)
            if row is None:
                # A source id that no longer resolves to a physical row makes
                # the mapping incomplete; it must not silently narrow a roster.
                roster.complete = False
                continue

            user_text = (row.user_content or "").strip()
            assistant_text = (row.assistant_content or "").strip()
            actor = (getattr(row, "sender_actor_id", "") or "").strip()
            label = (getattr(row, "sender", "") or "").strip()

            if user_text:
                if actor:
                    roster.actor_ids.add(actor)
                    if label:
                        roster.labels.setdefault(label.casefold(), set()).add(actor)
                else:
                    roster.has_unidentified_user_row = True

                # One requester lane per physical user row. It carries that
                # row's own words and NEVER its quote block.
                roster.lanes.append(
                    FactLane(
                        role=AUTHOR_ROLE_REQUESTER,
                        text=user_text,
                        actor_id=actor,
                        source_message_id=(getattr(row, "source_message_id", "") or ""),
                        canonical_turn_id=row.canonical_turn_id,
                        speaker_label=label,
                        session_date=(getattr(row, "session_date", "") or "").strip(),
                        audience_conversation_id=(
                            getattr(row, "audience_conversation_id", "") or ""
                        ).strip(),
                        origin_channel_id=(getattr(row, "origin_channel_id", "") or "").strip(),
                        audience_attribution_version=int(
                            getattr(row, "audience_attribution_version", 0) or 0,
                        ),
                    )
                )

                quote = (getattr(row, "reply_target_body", "") or "").strip()
                target_id = (getattr(row, "reply_target_message_id", "") or "").strip()
                if int(getattr(row, "reply_attribution_version", 0) or 0) > 0:
                    roster.reply_bearing = True
                if quote:
                    # When the reply target resolves to a row we already hold,
                    # create NO subject lane: that row's own requester lane is
                    # the source of truth and will produce (or already produced)
                    # its facts. The quote is current-request context, not a
                    # second disclosure.
                    audience = (getattr(row, "audience_conversation_id", "") or "").strip()
                    channel = (getattr(row, "origin_channel_id", "") or "").strip()
                    target_candidates = targets_by_scope.get(
                        (target_id, audience, channel),
                        [],
                    )
                    target_present = len(target_candidates) == 1
                    # A reply can quote the agent's own earlier message. That
                    # text is the agent's output, not a disclosure by the
                    # person being addressed, and filing it as one is how the
                    # agent's own words become evidence about a named human.
                    # Two signals answer this, and only a positive one
                    # suppresses. The row's own reply_subject_actor_id is what
                    # the platform said, so it reaches HISTORICAL rows and can
                    # answer in both directions. The ledger can only ever
                    # confirm: its absence is not a denial, because the
                    # recorded set is partial by construction -- a reply split
                    # across several platform messages reports at most one of
                    # them. Anything short of a positive answer falls through
                    # to the behaviour below, never to suppression.
                    subject_actor = (getattr(row, "reply_subject_actor_id", "") or "").strip()
                    if not target_present:
                        outcome = self._quote_is_agent_output(
                            channel_id=channel,
                            target_message_id=target_id,
                            reply_subject_actor_id=subject_actor,
                            agent_actor_ids=agent_actor_ids,
                        )
                        self._record_quote_outcome(outcome)
                        if outcome == self.QUOTE_AGENT_AUTHORED:
                            # Name the identity that caused this. Suppression
                            # destroys a quoted block, and an id that is too
                            # broad would delete a real member's words while
                            # looking exactly like a working guard. Without the
                            # matched id the record shows that something was
                            # destroyed and never what destroyed it.
                            logger.info(
                                "AGENT_QUOTE_SUPPRESSED conv=%s channel=%s "
                                "target_message_id=%s canonical_turn_id=%s "
                                "matched_actor_id=%s",
                                (getattr(self._config, "conversation_id", "") or ""),
                                channel,
                                target_id,
                                row.canonical_turn_id,
                                subject_actor or "(ledger)",
                            )
                            target_present = True
                    if not target_present:
                        roster.lanes.append(
                            FactLane(
                                role=AUTHOR_ROLE_SUBJECT,
                                text=quote,
                                # ONLY the resolved subject. Never the requester's
                                # id: that is the reply-chain contamination path.
                                actor_id=subject_actor,
                                source_message_id=target_id,
                                canonical_turn_id=row.canonical_turn_id,
                                speaker_label=(
                                    getattr(row, "reply_subject_label", "") or ""
                                ).strip(),
                            )
                        )

            if assistant_text:
                roster.lanes.append(
                    FactLane(
                        role=AUTHOR_ROLE_ASSISTANT,
                        text=assistant_text,
                        actor_id="",
                        canonical_turn_id=row.canonical_turn_id,
                    )
                )
        return roster
