"""Answering a few replies after the daily question, then stopping.

A question posted and then ignored reads worse than no question at all: it
looks like something fired and left. So the window exists to make Vast a
participant rather than a publisher. Its whole design is restraint — reply
to a couple of the best responses, prefer a short question to a lecture,
and stop.

Every bound is a shipped constant, because "bounded" that a caller can
widen is not bounded. Where the specification gives a range this takes the
conservative end and says so; where it gives nothing measurable, the value
is REQUIRED from the caller rather than invented here, so an unset bound
stops the window instead of quietly defaulting to something nobody chose.

Replies are judged exactly as questions are. A reply that puts words in a
member's mouth is the same harm as a question that does, and a reply is the
likeliest place for a general medical claim in Vast's own voice, so both
guards apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

# Ships dark, like the scheduler. Enabling belongs with live posting.
RESPONSE_WINDOW_ENABLED_BY_DEFAULT = False

# Spec: "approximately 6-12 hours after the post". Conservative end chosen
# deliberately — a shorter window can be widened after observation, while a
# long one cannot be narrowed after Vast has already replied late.
WINDOW_HOURS = 6

# Spec: "Reply to at most 2-4 responses total". Conservative end again, and
# "total" is per posted question, of which there is one per day.
MAX_REPLIES_PER_QUESTION = 2

# Spec priorities, in the order given. A response matching none of these is
# not replied to, which is how "Vast should not answer everyone" is enforced
# rather than hoped for.
REPLY_PRIORITIES: tuple[str, ...] = (
    "unexpectedly_useful_answer",
    "strong_disagreement",
    "concrete_personal_result",
    "claim_worth_clarifying",
    "invites_a_good_joke",
)

REPLY_GUIDANCE = (
    "Reply as one participant among others. Prefer a short follow-up "
    "question to any kind of lecture. Do not explain, do not summarise, do "
    "not correct at length. Never state a general claim as fact, and never "
    "attribute anything to the person you are replying to that they did not "
    "say. Never mention scheduling, tooling, access or that any of this is "
    "automated. One or two sentences."
)


class ResponseWindowNotConfigured(RuntimeError):
    """Raised when a bound the specification leaves open has no value."""


@dataclass(frozen=True)
class WindowState:
    """Everything needed to decide whether one reply may be sent."""

    posted_at: datetime
    replies_sent: int = 0
    peers_talking: bool = False


@dataclass(frozen=True)
class ReplyDecision:
    reply: bool
    reason: str = ""
    priority: str = ""


def window_closes_at(posted_at: datetime) -> datetime:
    return posted_at + timedelta(hours=WINDOW_HOURS)


def should_reply(
    *,
    state: WindowState,
    now: datetime,
    priority: str,
    enabled: bool = RESPONSE_WINDOW_ENABLED_BY_DEFAULT,
    active_hours_end=None,
) -> ReplyDecision:
    """Whether this response earns one of the few available replies.

    ``active_hours_end`` is the specification's "no later than reasonable
    active-channel hours", which it does not define. It is required rather
    than guessed: passing ``None`` refuses, so the missing decision surfaces
    instead of being silently made here.
    """
    if not enabled:
        return ReplyDecision(False, "window_disabled")
    if active_hours_end is None:
        raise ResponseWindowNotConfigured(
            "active-channel end hour is not configured; the specification "
            "requires the window to end no later than reasonable active "
            "hours but does not say what they are, so it must be supplied."
        )
    if state.peers_talking:
        # "Stop participating once members are productively talking among
        # themselves." Supplied by the caller; see the module note.
        return ReplyDecision(False, "peers_talking_among_themselves")
    if state.replies_sent >= MAX_REPLIES_PER_QUESTION:
        return ReplyDecision(False, "reply_budget_spent")
    if now > window_closes_at(state.posted_at):
        return ReplyDecision(False, "window_closed")
    if now.astimezone(active_hours_end.tzinfo).hour >= active_hours_end.hour:
        return ReplyDecision(False, "outside_active_hours")
    if priority not in REPLY_PRIORITIES:
        return ReplyDecision(False, "not_worth_replying_to")
    return ReplyDecision(True, "", priority)


def compose_reply(
    *,
    member_words: str,
    priority: str,
    composer,
    judge,
    claim_checker=None,
) -> tuple[str, str]:
    """Draft one reply and put it through both guards. Returns (text, reason).

    The same attribution judge the questions use, because a reply that puts
    words in a member's mouth is the same harm; and the same generality
    check the generated questions use, because a reply is the likeliest
    place for a medical claim in Vast's own voice. Either guard failing
    yields no reply — there is no partial pass here.
    """
    from .broader import validate_broader_question
    from .fidelity import run_fidelity_gate

    if composer is None:
        return "", "composer_not_configured"
    try:
        raw = composer(
            member_words=member_words,
            priority=priority,
            guidance=REPLY_GUIDANCE,
        )
    except Exception:
        return "", "composer_error"
    text = str(raw or "").strip()
    if not text:
        return "", "empty_reply"

    verdict = run_fidelity_gate(quote=member_words, draft=text, judge=judge)
    if not verdict.faithful:
        return "", f"attribution:{verdict.reason or 'asserts'}"

    # Reuse the generated-question validator for Vast's own voice. Its form
    # rules apply verbatim: a reply is also one or two sentences, also never
    # a diagnosis, also never "Question of the day".
    checked = validate_broader_question(text, claim_checker=claim_checker)
    if not checked.usable:
        return "", f"own_voice:{checked.reason}"
    return text, ""


def handle_response(
    *,
    state: WindowState,
    now: datetime,
    priority: str,
    member_words: str,
    composer,
    judge,
    claim_checker=None,
    enabled: bool = RESPONSE_WINDOW_ENABLED_BY_DEFAULT,
    active_hours_end=None,
) -> tuple[str, str]:
    """Decide and draft in one call: the unit a poster would invoke.

    The two halves are joined here rather than left for a caller to combine,
    because a caller that forgets the decision would draft replies the
    window had already refused, and a caller that forgets the guards would
    send text nothing checked. Returns ``(text, reason)`` with an empty text
    whenever anything refused.

    Nothing here posts, watches, or waits; the outermost caller — a poster
    reading replies — does not exist yet and is gated behind live posting.
    """
    decision = should_reply(
        state=state, now=now, priority=priority, enabled=enabled,
        active_hours_end=active_hours_end,
    )
    if not decision.reply:
        return "", decision.reason
    return compose_reply(
        member_words=member_words, priority=priority, composer=composer,
        judge=judge, claim_checker=claim_checker,
    )
