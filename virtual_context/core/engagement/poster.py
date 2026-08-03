"""The only place this package can write anything.

Everything else here reads. This module is the single point where a question
leaves the system, so its guards are refusals rather than checks: each one
raises, because a guard that returns a value can be ignored by a caller that
forgets to look at it, and there is no safe default for "post anyway".

Four things must all be true, and none of them has a fallback:

  the destination is on the shipped post list, compared by id;
  this exact message was confirmed against its source in this run;
  posting was explicitly asked for, never assumed;
  no question has already gone out for this Eastern day.

The credential never reaches this module. As with the source check, the
caller supplies a sender and owns the transport, the token and the egress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .allowlist import POST_CHANNEL_IDS
from .history import PostRecord, topic_fingerprint

# Ships dark, and deliberately NOT as a parameter.
#
# Permission and intent are different things. A caller says whether it wants
# to post today (``post=`` on the runner); it does not get to say whether
# posting is allowed at all. As a parameter with a safe default, the only
# property a test could establish was that today's caller chooses not to pass
# ``enabled=True`` — a convention that holds until someone writes a second
# caller. Read from the module instead, enabling posting is a committed,
# reviewable edit to this line, and no caller can reach it.
#
# This is the same shape as POST_CHANNEL_IDS below, which is imported rather
# than injected and is why a caller cannot post to a community channel.
POSTING_ENABLED = False

POSTING_ZONE = "America/New_York"


class PostRefused(RuntimeError):
    """Raised when any precondition for sending is not met."""


@dataclass(frozen=True)
class PostResult:
    message_id: str
    channel_id: str
    day: str


def pending_claims(history) -> list:
    """Days claimed whose send was never confirmed.

    A pending row means we sent, or possibly did not, and cannot tell. It is
    deliberately NOT retried: retrying is exactly how the one irreversible
    mistake happens. It stays claimed and is surfaced here for a person to
    resolve.
    """
    query = getattr(history, "pending", None)
    if callable(query):
        return list(query())
    return [r for r in history.all() if getattr(r, "status", "") == "pending"]


def already_posted_today(history, *, now: datetime) -> bool:
    """Whether this Eastern calendar day is already claimed.

    A claim counts whether or not the send was confirmed. An unconfirmed
    claim is the case where we cannot tell what happened, and the safe
    reading of "cannot tell" is that the day is spent.

    Keyed on the civil day rather than an elapsed interval, so a restart, a
    manual re-run or a duplicate wake-up all resolve to the same day and
    cannot produce a second post.
    """
    today = now.astimezone(ZoneInfo(POSTING_ZONE)).date()
    # A durable backend answers this in the database, so a row committed by
    # another process counts immediately and no growing table is loaded to
    # answer a yes/no question. The scan is the reference path only.
    claimed = getattr(history, "day_is_claimed", None)
    if callable(claimed):
        return bool(claimed(today))
    for record in history.all():
        posted = getattr(record, "posted_at", None)
        if posted is None:
            continue
        if posted.astimezone(ZoneInfo(POSTING_ZONE)).date() == today:
            return True
    return False


def post_question(
    *,
    candidate,
    question: str,
    channel_id: str,
    verification,
    history,
    sender: Callable[..., Any],
    now: datetime,
    question_type: str,
) -> PostResult:
    """Send one question, or refuse and say which precondition failed."""
    if not POSTING_ENABLED:
        raise PostRefused(
            "posting is disabled in this build; it is shipped configuration "
            "and cannot be enabled by a caller"
        )
    if channel_id not in POST_CHANNEL_IDS:
        raise PostRefused(
            f"channel {channel_id} is not a permitted destination; posting "
            "is restricted to the shipped post list"
        )
    if verification is None or not getattr(verification, "ok", False):
        raise PostRefused(
            "the source was not confirmed live; a question is never sent on "
            "an unverified quote"
        )
    expected = str(getattr(candidate, "source_message_id", "") or "")
    if str(getattr(verification, "verified_message_id", "")) != expected:
        raise PostRefused(
            "the live verification names a different message than the "
            "candidate being quoted; a pass is not transferable between "
            "candidates or runs"
        )
    if already_posted_today(history, now=now):
        raise PostRefused(
            "a question has already gone out for this Eastern day"
        )
    body = (question or "").strip()
    if not body:
        raise PostRefused("refusing to send an empty question")

    # Claim the day BEFORE sending, and address the confirmation by the
    # handle this claim returned. "The last row" is not an address: a second
    # process computes the same one and confirms a claim it did not make.
    #
    # A Discord send and a Postgres write
    # cannot be made atomic, so the ordering decides which way a failure
    # between them breaks. Claiming first means a crash costs a post; sending
    # first means it costs the record, and the next run then sees an unclaimed
    # day and posts again. Skipping a day is recoverable; posting twice into a
    # community channel is not.
    handle = history.record(PostRecord(
        posted_at=now,
        channel_id=channel_id,
        question_type=question_type,
        tagged_actor_id=str(getattr(candidate, "actor_id", "") or ""),
        source_message_ids=(expected,) if expected else (),
        topic_fingerprint=topic_fingerprint(body),
        question_text=body,
        discord_message_id="",
        status="pending",
    ))

    message_id = str(sender(channel_id=channel_id, content=body) or "")
    if not message_id:
        # The claim stands. The send may or may not have landed, and the day
        # stays taken precisely because we cannot tell.
        raise PostRefused("the send returned no message id; treating as failed")

    history.update(handle, discord_message_id=message_id, status="posted")
    day = now.astimezone(ZoneInfo(POSTING_ZONE)).date().isoformat()
    return PostResult(message_id=message_id, channel_id=channel_id, day=day)
