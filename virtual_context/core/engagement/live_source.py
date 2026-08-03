"""Re-fetch a message from its source before quoting it publicly.

The attested source record proves what a message was at ingest. It cannot
prove what it is now, so a message edited or deleted since then would still
verify — and quoting someone's deleted words back at the whole server is the
failure this closes.

CONTENT IS DELIBERATELY NOT COMPARED, and this is not an oversight to
improve on later. The stored ``canonical_body_sha256`` covers the
speaker-prefixed body written at ingest, while the source returns the raw
text; the two agree on 19 of 1645 stored rows. Stripping the prefix does not
reproduce ``transport_body_sha256`` either, because that digest is the
adapter's own projection and lives in the adapter. So a body comparison
would report drift on nearly every row while looking exactly like a working
verifier. The four signals used instead — the HTTP status, the edited
marker, the author id and the channel id — are projection-independent and
answer both harms exactly.

The credential never reaches this module. The caller supplies a fetcher and
owns the transport, the token and the egress; this code owns only the
question being asked and what each answer means.

What is measured, and what is not, because the two should not blur together
just by sitting in the same file:

MEASURED against the live source — a present message returns 200 with a null
edited marker and an author id matching the stored actor; an absent one
returns 404; a selection run costs one request.

NOT MEASURED, assumed from documentation — that this read requires View
Channel and Read Message History and no privileged intent, and that the
per-route budget comfortably absorbs a handful of daily requests. The
successful reads show the bot can read the channels it is pointed at; they
do not establish which permission bits are required in general, nor where
the rate ceiling sits. Neither assumption is load-bearing for correctness:
a permission that turns out to be missing surfaces as 403 and a budget that
turns out to be tighter surfaces as 429, and both already block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Every outcome is named so a block lands as a counted row rather than a
# sentence, and so an operator can tell "he deleted it" from "we were rate
# limited" without reading a log.
LIVE_OK = ""
LIVE_DELETED = "source_message_deleted"
LIVE_EDITED = "source_message_edited"
LIVE_AUTHOR_MISMATCH = "source_author_mismatch"
LIVE_CHANNEL_MISMATCH = "source_channel_mismatch"
LIVE_FORBIDDEN = "source_access_forbidden"
LIVE_RATE_LIMITED = "source_rate_limited"
LIVE_UNREACHABLE = "source_unreachable"
LIVE_UNREADABLE = "source_response_unreadable"

# Everything except a clean match blocks. Membership is deliberately absent:
# an author who left the guild still wrote the message, it is still public,
# and treating departure as a verification failure would erase former
# members from the corpus — a different harm than the one guarded here.
BLOCKING_REASONS = frozenset({
    LIVE_DELETED, LIVE_EDITED, LIVE_AUTHOR_MISMATCH, LIVE_CHANNEL_MISMATCH,
    LIVE_FORBIDDEN, LIVE_RATE_LIMITED, LIVE_UNREACHABLE, LIVE_UNREADABLE,
})


@dataclass(frozen=True)
class LiveVerification:
    ok: bool
    reason: str = ""
    detail: str = ""


def _actor_tail(actor_id: str) -> str:
    return (actor_id or "").rsplit(":", 1)[-1].strip()


def verify_source_live(
    candidate,
    *,
    fetcher: Callable[..., Any],
) -> LiveVerification:
    """Confirm the message still exists, unedited, from the same author.

    ``fetcher(channel_id, message_id)`` returns ``(status, payload)``. It is
    injected so this module never holds a credential and never opens a
    socket.
    """
    channel_id = str(getattr(candidate, "channel_id", "") or "")
    message_id = str(getattr(candidate, "source_message_id", "") or "")
    try:
        status, payload = fetcher(channel_id=channel_id, message_id=message_id)
    except Exception:
        return LiveVerification(False, LIVE_UNREACHABLE)

    if status == 404:
        return LiveVerification(False, LIVE_DELETED, f"message {message_id}")
    if status == 403:
        return LiveVerification(False, LIVE_FORBIDDEN, f"channel {channel_id}")
    if status == 429:
        return LiveVerification(False, LIVE_RATE_LIMITED)
    if status != 200 or not isinstance(payload, dict):
        return LiveVerification(False, LIVE_UNREACHABLE, f"status {status}")

    if payload.get("edited_timestamp"):
        return LiveVerification(
            False, LIVE_EDITED, str(payload.get("edited_timestamp")),
        )
    author = payload.get("author")
    author_id = str((author or {}).get("id") or "")
    if not author_id:
        return LiveVerification(False, LIVE_UNREADABLE, "no author id")
    if author_id != _actor_tail(getattr(candidate, "actor_id", "")):
        return LiveVerification(
            False, LIVE_AUTHOR_MISMATCH,
            f"stored {_actor_tail(getattr(candidate, 'actor_id', ''))} "
            f"live {author_id}",
        )
    live_channel = str(payload.get("channel_id") or "")
    if live_channel and live_channel != channel_id:
        return LiveVerification(
            False, LIVE_CHANNEL_MISMATCH, f"stored {channel_id} live {live_channel}",
        )
    return LiveVerification(True, LIVE_OK)


def select_live_verified(
    ranked: list,
    *,
    fetcher: Callable[..., Any],
    max_attempts: int = 3,
) -> tuple[Any | None, list]:
    """Walk the ranking until one candidate verifies against its source.

    Verification runs at selection time on the candidate about to be used,
    not across the corpus: one post a day means a handful of calls. Each
    block becomes a named rejection so it lands as a counted row in the
    ladder rather than disappearing into a log line, and the walk stops
    after ``max_attempts`` so a rate limit cannot turn one post into a
    hundred requests.
    """
    from .candidates import Rejection

    rejections: list = []
    for candidate in list(ranked or [])[:max(0, max_attempts)]:
        verdict = verify_source_live(candidate, fetcher=fetcher)
        if verdict.ok:
            return candidate, rejections
        rejections.append(Rejection(
            getattr(candidate, "canonical_turn_id", ""),
            "live_source",
            verdict.reason,
            verdict.detail,
        ))
        # A rate limit or an unreachable source is about us, not about this
        # candidate, so trying the next one would repeat the same failure.
        if verdict.reason in (LIVE_RATE_LIMITED, LIVE_UNREACHABLE):
            break
    return None, rejections
