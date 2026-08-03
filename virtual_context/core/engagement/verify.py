"""Prove a candidate's attribution twice, or discard it.

A canonical row already carries an actor, a channel and a message id. That
is one record, and a derived speaker label on its own is exactly what this
system learned not to trust. The trusted adapter also writes an independent
source record at ingest, from the transport's own evidence, keyed on the
message id. Agreement between the two is what makes an attribution safe to
act on.

Disagreement is always a rejection and never a repair. A row whose stored
actor contradicts its attested author is not evidence that the row needs
fixing here; it is evidence that nobody knows who spoke, and quoting it
would put one member's words under another's name.

The contradiction rule is stronger than a per-candidate check: if two
records claim the same message id for different speakers, BOTH are rejected.
Choosing between them would mean guessing, and guessing is what produced the
incident this pipeline was written for.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from .candidates import Candidate, Rejection


@dataclass(frozen=True)
class MessageSourceRecord:
    """The adapter-attested identity of one source message."""

    canonical_turn_id: str
    message_id: str
    channel_id: str
    guild_id: str
    author_id: str
    source_actor_id: str


def _actor_matches_author(actor_id: str, author_id: str) -> bool:
    """An actor id ends in the platform's own immutable user id."""
    actor = (actor_id or "").strip()
    author = (author_id or "").strip()
    if not actor or not author:
        return False
    return actor.rsplit(":", 1)[-1] == author


def verify_candidates(
    candidates: list[Candidate],
    sources: Mapping[str, MessageSourceRecord],
) -> tuple[list[Candidate], list[Rejection]]:
    """Cross-check each candidate against its attested source record."""
    verified: list[Candidate] = []
    rejected: list[Rejection] = []

    # Contradiction sweep first: when SEPARATE records claim one message id
    # for different speakers, every record of it is disqualified, because
    # choosing between them would be a guess. This is distinct from a single
    # row disagreeing with its own attestation — that is an author mismatch,
    # diagnosed per candidate below — and the two must stay distinguishable
    # or the dry-run report cannot say which failure actually occurred.
    speakers_by_message: dict[str, set[str]] = defaultdict(set)
    record_actors_by_message: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        speakers_by_message[candidate.source_message_id].add(candidate.actor_id)
        record = sources.get(candidate.canonical_turn_id)
        if record is not None and record.source_actor_id:
            record_actors_by_message[record.message_id].add(
                record.source_actor_id,
            )
    contradicted = {
        message_id
        for message_id, actors in speakers_by_message.items()
        if len(actors) > 1
    } | {
        message_id
        for message_id, actors in record_actors_by_message.items()
        if len(actors) > 1
    }

    for candidate in candidates:
        def _reject(reason: str, detail: str = "") -> None:
            rejected.append(Rejection(
                candidate.canonical_turn_id, "verify", reason, detail,
            ))

        if candidate.source_message_id in contradicted:
            _reject(
                "contradictory_speaker",
                f"message_id={candidate.source_message_id} claimed for "
                f"{sorted(speakers_by_message[candidate.source_message_id])}",
            )
            continue
        record = sources.get(candidate.canonical_turn_id)
        if record is None:
            _reject("no_attested_source")
            continue
        if record.message_id != candidate.source_message_id:
            _reject(
                "message_id_mismatch",
                f"row={candidate.source_message_id} attested={record.message_id}",
            )
            continue
        if not _actor_matches_author(candidate.actor_id, record.author_id):
            _reject(
                "author_mismatch",
                f"row_actor={candidate.actor_id} attested_author={record.author_id}",
            )
            continue
        if record.channel_id != candidate.channel_id:
            _reject(
                "channel_mismatch",
                f"row={candidate.channel_id} attested={record.channel_id}",
            )
            continue
        verified.append(candidate)
    return verified, rejected
