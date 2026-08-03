"""When a thread is worth revisiting, and what question it can still carry.

A timed follow-up needs enough elapsed time that an update exists and not so
much that the thread is cold. Elapsed time is measured from the send time
encoded in the message id, never from an ingest timestamp.

Resolution is deliberately not a veto. An answered thread is usually a
BETTER thread, because there is now an outcome to ask about; what expires is
the anticipatory question, not the subject. So a later answer moves the
thread from an "anticipatory" stance to an "outcome" stance rather than
discarding it. Only one thing blocks a candidate outright: the member is
talking about it right now, where a follow-up would just ask him to repeat
the conversation he is already having.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

_MIN_AGE = timedelta(days=2)
_MAX_AGE = timedelta(days=7)
_SAME_DAY = timedelta(days=1)
_REOPENED_WINDOW = timedelta(hours=12)


def timed_followup_eligibility(candidate, *, now: datetime) -> tuple[bool, str]:
    """Whether *candidate* can carry a timed follow-up, and why not."""
    age = now - candidate.sent_at
    if age < timedelta(0):
        return False, "future_dated"
    if age < _SAME_DAY:
        return False, "same_day"
    if age < _MIN_AGE:
        return False, "too_recent"
    if age > _MAX_AGE:
        return False, "too_old_for_timed_followup"
    return True, ""


@dataclass(frozen=True)
class ThreadState:
    """What later conversation did to this thread."""

    resolved: bool
    stance: str            # "anticipatory" | "outcome"
    blocks_candidate: bool
    reason: str = ""
    resolving_text: str = ""


def _mentions(text: str, terms: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(
        re.search(rf"\b{re.escape(t.lower())}\b", lowered) for t in terms if t
    )


def assess_thread(
    candidate,
    *,
    later_texts: list[tuple[str, datetime]],
    now: datetime,
    subject_terms: list[str] | None = None,
) -> ThreadState:
    """Classify a thread from the member's own later messages."""
    terms = [t for t in (subject_terms or []) if t]
    on_subject = [
        (text, when) for text, when in (later_texts or [])
        if not terms or _mentions(text, terms)
    ]
    if not on_subject:
        return ThreadState(False, "anticipatory", False)

    newest_at = max(when for _text, when in on_subject)
    if now - newest_at <= _REOPENED_WINDOW:
        return ThreadState(
            True, "outcome", True, "member_posted_today",
        )
    newest_text = max(on_subject, key=lambda item: item[1])[0]
    return ThreadState(True, "outcome", False, "", newest_text)
