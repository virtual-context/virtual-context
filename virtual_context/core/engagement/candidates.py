"""Turn retrieval results into verifiable candidates, or rejections.

Two rules govern this stage and neither has an exception.

Authorship, not involvement. Only a ``requester`` row is a candidate. A
``subject`` row is content authored by somebody else on which this member
was replied to; presenting it as their words is precisely the
misattribution this system exists to prevent.

Provable timing. Send time is decoded from the message id, which encodes
it. A row without a usable message id cannot be timed, cannot be verified
against its source record, and cannot be deduplicated — so it is rejected
rather than carried forward on the strength of the fields it does have.

Every discarded row produces a ``Rejection`` naming the stage and reason. A
silent drop is indistinguishable from a bug, and a run that reports nothing
is indistinguishable from a run that found nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..discord_snowflake import snowflake_to_datetime
from .channels import ChannelAllowlist


@dataclass(frozen=True)
class Candidate:
    """One member's own message, eligible for verification."""

    canonical_turn_id: str
    source_message_id: str
    actor_id: str
    channel_id: str
    text: str
    sent_at: datetime


@dataclass(frozen=True)
class Rejection:
    """Why one candidate was discarded, and by which stage."""

    canonical_turn_id: str
    stage: str
    reason: str
    detail: str = ""


def collect_candidates(
    results,
    *,
    allowlist: ChannelAllowlist,
    dedupe: bool = True,
) -> tuple[list[Candidate], list[Rejection]]:
    """Filter retrieval results down to authored, timeable, sourceable rows.

    ``dedupe`` collapses repeats of one message id, keeping the first. It is
    disabled only by callers that must inspect every record for one id — the
    contradiction check needs to see all of them before any are dropped.
    """
    kept: list[Candidate] = []
    rejected: list[Rejection] = []
    seen: set[str] = set()

    for result in results or []:
        provenance = getattr(result, "provenance", None)
        turn_id = str(getattr(provenance, "canonical_turn_id", "") or "")

        def _reject(reason: str, detail: str = "") -> None:
            rejected.append(Rejection(turn_id, "collect", reason, detail))

        if provenance is None:
            _reject("no_provenance")
            continue
        if (getattr(provenance, "source_role", "") or "") != "requester":
            _reject(
                "not_authored_by_actor",
                f"source_role={getattr(provenance, 'source_role', '') or ''}",
            )
            continue
        channel_id = str(getattr(provenance, "origin_channel_id", "") or "")
        if not allowlist.may_source(channel_id):
            _reject("channel_not_sourceable", f"channel_id={channel_id}")
            continue
        message_id = str(getattr(provenance, "source_message_id", "") or "")
        sent_at = snowflake_to_datetime(message_id)
        if sent_at is None:
            _reject("no_source_message_id")
            continue
        if dedupe and message_id in seen:
            _reject("duplicate_source_message_id", f"message_id={message_id}")
            continue
        seen.add(message_id)
        kept.append(Candidate(
            canonical_turn_id=turn_id,
            source_message_id=message_id,
            actor_id=str(getattr(provenance, "actor_id", "") or ""),
            channel_id=channel_id,
            text=str(getattr(result, "text", "") or ""),
            sent_at=sent_at,
        ))
    return kept, rejected
