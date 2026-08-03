"""The entailment gate: a draft may not say more than its source did.

Judged by a model, because the failure is semantic — "planned to add"
becoming "started", "can happen in these cases" becoming "usually". Those
are not detectable by pattern matching without either missing paraphrases or
rejecting faithful drafts.

Two rules make the gate trustworthy rather than decorative:

Fail closed on configuration. A gate with no judge RAISES. Skipping the
check because a config key was blank would let unchecked drafts through
silently, which is the exact defect shape this pipeline exists to prevent —
a check accepted, dropped, and never mentioned.

Fail closed on failure. A judge that errors, or answers in a shape that
cannot be read, produces an UNFAITHFUL verdict. An unreadable answer is not
permission to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class FidelityGateNotConfigured(RuntimeError):
    """Raised when the gate is asked to run with no judge configured."""


@dataclass(frozen=True)
class FidelityVerdict:
    faithful: bool
    reason: str = ""


@dataclass(frozen=True)
class FidelityFixture:
    """One adversarial pair with a known-correct verdict."""

    name: str
    quote: str
    draft: str
    expected_faithful: bool
    must_reject: bool = True


# Drawn from the specification's own worked examples, so each carries a
# verdict that is correct independently of any model's judgement. These
# exist to test the GATE, not the model's mood on a given day.
ADVERSARIAL_FIDELITY_FIXTURES: tuple[FidelityFixture, ...] = (
    FidelityFixture(
        name="planned_became_started",
        quote="Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        draft="You started a four-week SS-31 run. How is recovery?",
        expected_faithful=False,
    ),
    FidelityFixture(
        name="planned_stays_planned",
        quote="Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        draft=(
            "On July 30 you said you planned to add SS-31 at 5 mg for four "
            "weeks, then move to MOTS-c. Have you started the SS-31 yet?"
        ),
        expected_faithful=True,
        must_reject=False,
    ),
    FidelityFixture(
        name="scope_widened_to_usually",
        quote="Elevated ApoB can happen in these cases.",
        draft="You said elevated ApoB usually happens. What drives that?",
        expected_faithful=False,
    ),
    FidelityFixture(
        name="causality_invented",
        quote=(
            "These were my previous labs on 6/18 when I was taking cardarine."
        ),
        draft="Since Cardarine raised your ApoB, are you dropping it?",
        expected_faithful=False,
    ),
    FidelityFixture(
        name="uncertainty_preserved",
        quote=(
            "These were my previous labs on 6/18 when I was taking cardarine."
        ),
        draft=(
            "When you supplied the June labs and noted you were taking "
            "Cardarine, were you suggesting it likely caused the ApoB "
            "difference, or simply identifying one changed variable?"
        ),
        expected_faithful=True,
        must_reject=False,
    ),
)


def run_fidelity_gate(
    *,
    quote: str,
    draft: str,
    judge: Callable[..., Any] | None,
) -> FidelityVerdict:
    """Judge whether *draft* is entailed by *quote*."""
    if judge is None:
        raise FidelityGateNotConfigured(
            "the fidelity gate has no judge configured; configure the "
            "engagement fidelity model or disable the pipeline. Running "
            "without the gate would post drafts no one checked."
        )
    try:
        raw = judge(quote=quote, draft=draft)
    except Exception:
        return FidelityVerdict(False, "judge_error")
    if not isinstance(raw, dict) or "faithful" not in raw:
        return FidelityVerdict(False, "unreadable_verdict")
    faithful = raw.get("faithful")
    if not isinstance(faithful, bool):
        return FidelityVerdict(False, "unreadable_verdict")
    return FidelityVerdict(faithful, str(raw.get("reason") or ""))
