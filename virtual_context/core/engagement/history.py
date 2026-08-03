"""What was asked, so it is not asked again.

The corpus this job draws from is not a rolling window: it holds every
message the server has produced, and widening a time window cannot refresh
it. So the defence against sounding repetitive is memory of what was already
asked, and that memory matters most at launch, when there is least of it.

The record is a repetition ledger, not a profile. It keeps an actor id
because "do not tag the same member twice this week" needs one, the source
message ids because "do not mine the same thread twice" needs those, and a
fixed-width topic fingerprint because "do not ask this again in other words"
needs a similarity signal. It does NOT keep the member's words, display
name, or anything else that would let someone reconstruct a picture of him
from this table. Deliberately absent, so the absence is a decision rather
than an oversight:

  * the member's original message text — the id identifies the thread; a
    second copy of what he said is a second place it can leak from
  * his handle or display name — presentation, re-derivable, and the actor
    id is what dedup actually compares
  * channel labels — ids decide everything; labels drift
  * anything about him not needed to prevent a repeat

The topic fingerprint is a 64-bit simhash over normalised tokens. It answers
"is this close to something already asked" and cannot be read back into the
question it came from, so similarity costs no stored content.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .candidates import Rejection

# Cooldowns. Deliberately conservative at launch, when the pool is thinnest
# and a repeat is most visible.
MEMBER_COOLDOWN = timedelta(days=14)
QUESTION_SIMILARITY_WINDOW = timedelta(days=60)
CHANNEL_WINDOW = timedelta(days=7)
CHANNEL_MAX_IN_WINDOW = 3
SIMILARITY_DISTANCE = 12  # bits of a 64-bit simhash

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do",
    "for", "from", "have", "how", "in", "is", "it", "of", "on", "or", "the",
    "to", "was", "were", "what", "when", "which", "with", "you", "your",
})


@dataclass(frozen=True)
class PostRecord:
    """One question that was asked, reduced to what prevents a repeat."""

    posted_at: datetime
    channel_id: str
    question_type: str                 # "personal" | "timed" | "broader"
    tagged_actor_id: str               # "" when no member was tagged
    source_message_ids: tuple[str, ...] = ()
    topic_fingerprint: int = 0
    question_text: str = ""            # Vast's own words, for near-dup review
    discord_message_id: str = ""
    resolution: str = ""               # "" | "answered" | "ignored"


class InMemoryPostHistory:
    """Reference implementation. The durable backend is gated on approval."""

    def __init__(self) -> None:
        self._records: list[PostRecord] = []

    def record(self, entry: PostRecord) -> None:
        self._records.append(entry)

    def since(self, moment: datetime) -> list[PostRecord]:
        return [r for r in self._records if r.posted_at >= moment]

    def all(self) -> list[PostRecord]:
        return list(self._records)


def topic_fingerprint(text: str) -> int:
    """64-bit simhash over normalised tokens; not reversible to the text."""
    tokens = [
        t for t in _TOKEN.findall((text or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return 0
    vector = [0] * 64
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


def fingerprint_distance(left: int, right: int) -> int:
    """Hamming distance between two fingerprints."""
    return bin(int(left) ^ int(right)).count("1")


def check_repetition(
    *,
    history,
    now: datetime,
    actor_id: str,
    channel_id: str,
    source_message_ids: tuple[str, ...] | list[str],
    question_text: str,
) -> Rejection | None:
    """The first repetition rule this candidate breaks, or ``None``."""
    incoming = {str(m) for m in (source_message_ids or []) if str(m)}
    fingerprint = topic_fingerprint(question_text)
    records = history.all()

    for record in records:
        if incoming & set(record.source_message_ids):
            reason = (
                "thread_previously_ignored"
                if record.resolution == "ignored"
                else "thread_already_used"
            )
            return Rejection("", "history", reason, f"posted {record.posted_at:%Y-%m-%d}")

    if actor_id:
        for record in records:
            if (
                record.tagged_actor_id == actor_id
                and now - record.posted_at < MEMBER_COOLDOWN
            ):
                return Rejection(
                    "", "history", "member_recently_tagged",
                    f"last tagged {record.posted_at:%Y-%m-%d}",
                )

    if fingerprint:
        for record in records:
            if now - record.posted_at > QUESTION_SIMILARITY_WINDOW:
                continue
            if not record.topic_fingerprint:
                continue
            distance = fingerprint_distance(fingerprint, record.topic_fingerprint)
            if distance <= SIMILARITY_DISTANCE:
                return Rejection(
                    "", "history", "question_recently_asked",
                    f"distance={distance} from {record.posted_at:%Y-%m-%d}",
                )

    recent_in_channel = [
        r for r in records
        if r.channel_id == channel_id and now - r.posted_at <= CHANNEL_WINDOW
    ]
    if len(recent_in_channel) >= CHANNEL_MAX_IN_WINDOW:
        return Rejection(
            "", "history", "channel_recently_overused",
            f"{len(recent_in_channel)} posts in {CHANNEL_WINDOW.days}d",
        )
    return None


# Designed, NOT applied. The durable table is gated on explicit approval and
# a proven backup; nothing in this module creates or migrates it.
ENGAGEMENT_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS engagement_post_history (
    id                  TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    conversation_id     TEXT NOT NULL,
    posted_at           TEXT NOT NULL,
    channel_id          TEXT NOT NULL,
    question_type       TEXT NOT NULL,
    tagged_actor_id     TEXT NOT NULL DEFAULT '',
    source_message_ids  TEXT NOT NULL DEFAULT '',
    -- NUMERIC(20,0), not BIGINT. The fingerprint is an unsigned 64-bit
    -- simhash spanning 0..2^64-1, and Postgres BIGINT is signed with a
    -- ceiling of 2^63-1, so about half of all values would fail to insert.
    -- SQLite accepts them either way, which is why a test on SQLite alone
    -- would not have caught it.
    topic_fingerprint   NUMERIC(20,0) NOT NULL DEFAULT 0,
    question_text       TEXT NOT NULL DEFAULT '',
    discord_message_id  TEXT NOT NULL DEFAULT '',
    resolution          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS engagement_post_history_actor
    ON engagement_post_history (tenant_id, tagged_actor_id, posted_at);
CREATE INDEX IF NOT EXISTS engagement_post_history_channel
    ON engagement_post_history (tenant_id, channel_id, posted_at);
"""
