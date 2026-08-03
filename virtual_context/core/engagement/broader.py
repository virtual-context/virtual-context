"""Generate a channel question when no member thread survives.

The specification asks for a question to be GENERATED and validated, not
chosen from a list. A fixed rotation is what it explicitly rules out — the
worked examples are described as shape and tone rather than content to
recycle — so there is no pool here and none is expected.

It runs only after every personal and timed candidate has failed, and it is
allowed to fail: skipping beats posting filler.

Validation is deliberately separate from the attribution judge. That judge
asks whether a draft puts words in a member's mouth, which is meaningless
for a question addressed to nobody in particular. What matters here is
different: a generated question has no source, so nothing stops it asserting
a general claim as fact, and the specification forbids exactly that —
"can happen in these cases" must never become "usually". Folding this into
the attribution judge is what produced the over-strict gate that could not
pass a question at all; the guards stay apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

# Spec §Broader-question requirements and §Tone, carried to the generator.
BROADER_GENERATOR_GUIDANCE = (
    "Ask one practical, experience-driven or adversarial question that fits "
    "this channel. Prefer questions that expose the gap between biomarkers "
    "and lived outcomes, between protocol theory and actual response, or "
    "between a benefit and its cost. It must be answerable without an essay. "
    "Never ask for a medical diagnosis. Never open a bland prompt like 'what "
    "are your thoughts on X'. One clean question, one or two sentences, in a "
    "warm curious voice, one man among men. Never write 'Question of the "
    "day'. Never mention that anything was scheduled or generated. State no "
    "general claim as fact — ask, do not assert."
)

_BANNED = (
    "question of the day",
    "what are your thoughts on",
    "let's discuss",
    "engagement",
)
_DIAGNOSIS = re.compile(
    r"\b(should i (take|stop|start)|is (this|it) (safe|normal) for me|"
    r"do i have|diagnos)", re.IGNORECASE,
)
_MAX_SENTENCES = 2
_MAX_CHARS = 320


class BroaderGeneratorNotConfigured(RuntimeError):
    """Raised when generation is attempted with no model configured."""


@dataclass(frozen=True)
class BroaderQuestion:
    text: str = ""
    reason: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.text.strip()) and not self.reason


def validate_broader_question(
    text: str, *, claim_checker: Callable[..., Any] | None = None,
) -> BroaderQuestion:
    """Form checks, then an optional model check for asserted generalities.

    The form rules are deterministic because they are crisp — a banned
    phrase or a fourth sentence is not a judgement call. The generalisation
    rule is not crisp, so it is asked of a model when one is supplied; with
    none, the question fails rather than passing unchecked, for the same
    reason the fidelity gate refuses to run without a judge.
    """
    body = (text or "").strip()
    if not body:
        return BroaderQuestion("", "empty_question")
    lowered = body.lower()
    for phrase in _BANNED:
        if phrase in lowered:
            return BroaderQuestion("", "banned_phrasing")
    if "?" not in body:
        return BroaderQuestion("", "not_a_question")
    if len(body) > _MAX_CHARS:
        return BroaderQuestion("", "too_long")
    if len([s for s in re.split(r"[.!?]+", body) if s.strip()]) > _MAX_SENTENCES:
        return BroaderQuestion("", "too_many_sentences")
    if _DIAGNOSIS.search(body):
        return BroaderQuestion("", "solicits_diagnosis")

    if claim_checker is None:
        return BroaderQuestion("", "claim_check_not_configured")
    try:
        verdict = claim_checker(question=body)
    except Exception:
        return BroaderQuestion("", "claim_check_error")
    if not isinstance(verdict, dict) or "asserts_generality" not in verdict:
        return BroaderQuestion("", "unreadable_claim_verdict")
    asserts = verdict.get("asserts_generality")
    if not isinstance(asserts, bool):
        return BroaderQuestion("", "unreadable_claim_verdict")
    if asserts:
        return BroaderQuestion("", "asserts_a_generality")
    return BroaderQuestion(body, "")


CLAIM_CHECKER_SYSTEM_PROMPT = (
    "You are given one question intended for a discussion channel.\n\n"
    "Decide ONE thing: does it STATE a general claim as fact, rather than "
    "asking? Stating that something usually, mostly, always or never happens "
    "is asserting. Asking whether, when, or for whom it happens is not.\n\n"
    'Reply ONLY compact JSON: {"asserts_generality": true|false, '
    '"reason": "<short>"}.'
)


def generate_broader_question(
    *,
    channel_label: str,
    generator: Callable[..., Any] | None,
    claim_checker: Callable[..., Any] | None = None,
    recent_questions: list[str] | None = None,
) -> BroaderQuestion:
    """Generate one channel question, or fail with a named reason."""
    if generator is None:
        raise BroaderGeneratorNotConfigured(
            "no broader-question generator is configured; the specification "
            "requires this question to be generated rather than chosen from "
            "a list, so there is nothing to fall back to."
        )
    try:
        raw = generator(
            channel_label=channel_label,
            guidance=BROADER_GENERATOR_GUIDANCE,
            avoid=list(recent_questions or []),
        )
    except Exception:
        return BroaderQuestion("", "generator_error")
    candidate = str(raw or "").strip()
    validated = validate_broader_question(candidate, claim_checker=claim_checker)
    if validated.usable and candidate in set(recent_questions or []):
        return BroaderQuestion("", "repeats_a_recent_question")
    return validated
