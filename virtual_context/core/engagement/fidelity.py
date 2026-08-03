"""The attribution gate: a draft may not put words in a member's mouth.

The harm is misattribution — a reader coming away believing a member said,
did, concluded or experienced something he did not. That is the harm this
whole system exists to prevent, and it is the only thing this gate judges.

It deliberately does NOT judge entailment. A question that assumes an answer
attributes nothing: "how's the four weeks going" presumes a yes to the
question before it and dissolves the moment the member says he never
started. Treating that as a violation prohibits ordinary conversation, and a
gate that does so cannot pass any follow-up, because a follow-up is by
definition a question about the status of something.

So the model makes two separate judgements — does the draft ASSERT, does it
PRESUPPOSE — and only assertion fails. Splitting them removes the need to
decide when a presupposition is "too strong", a question that has no crisp
answer and whose crisp-looking answers are all wrong.

KNOWN GAP, deliberately not closed here: an unsupported generalisation in
the bot's OWN voice attributes nothing to the member and therefore passes.
Whether that is acceptable is a separate decision about what the bot may
claim in public, and folding it into this judge is what produced the
over-strict gate this replaced.

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


# The shipped judge instruction. Tests assert against THIS, never a copy.
FIDELITY_JUDGE_SYSTEM_PROMPT = (
    "Analyse a follow-up question against a member's quoted message.\n\n"
    "Make TWO separate judgements and do not let one influence the other.\n\n"
    "1. ASSERTS: does the draft STATE as fact anything the quote does not "
    "support? An assertion is a claim the reader would take as true "
    "regardless of how the member answers. Examples: 'moving the modafinil "
    "wrecked your sleep', 'you added the MotsC', 'since you started', 'now "
    "that the labs are back', 'you said X', a dose, compound, duration, "
    "cause or outcome the quote never gives.\n\n"
    "2. PRESUPPOSES: does the draft merely ASSUME something while asking? A "
    "presupposition dissolves if the member answers no. Examples: 'how's the "
    "four weeks going', 'how's the MotsC going'.\n\n"
    'Reply ONLY compact JSON: {"asserts": true|false, "presupposes": '
    'true|false, "reason": "<short>"}.\n\n'
    "A question that assumes is not a claim. Rewriting a question as a "
    "statement to test it is not allowed - judge what it actually says."
)


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
    # --- questions that attribute nothing: MUST PASS ---
    FidelityFixture(
        "asks_whether_a_plan_happened",
        "Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        "Did you end up starting the SS-31?", True, must_reject=False,
    ),
    FidelityFixture(
        # Reclassified: presumes a yes, attributes nothing, dissolves on a no.
        "presumptive_followup_four_weeks",
        "Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        "How's the four weeks going?", True, must_reject=False,
    ),
    FidelityFixture(
        "presumptive_followup_motsc",
        "Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        "How's the MotsC going?", True, must_reject=False,
    ),
    FidelityFixture(
        # KNOWN DEFECT: the judge reads this as an assertion while passing
        # the structurally identical four-weeks and MotsC cases. The
        # difference it is reacting to is that the quote only CONTEMPLATED
        # this change rather than stating it as a plan; both are still
        # presuppositions and both should pass. Recorded so the residual
        # false-positive rate is not mistaken for zero.
        "presumptive_followup_contemplated_change",
        "KPV 500mcg am. Considering adding it to the pm dose too.",
        "How's the twice-daily KPV working out?", True, must_reject=False,
    ),
    FidelityFixture(
        "preserves_uncertainty",
        "These were my previous labs on 6/18 when I was taking cardarine.",
        "When you posted the June labs and noted the Cardarine, were you "
        "pointing at it as one changed variable, or as the likely cause?",
        True, must_reject=False,
    ),
    # A clause judged IN ITS DRAFT, never as a fragment. Judging the second
    # clause alone reads it as presumptive and produced a wrong verdict; the
    # gate sees whole drafts, so the fixture is the whole draft.
    FidelityFixture(
        "clause_in_context_not_fragment",
        "Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        "Did you end up starting the SS-31? How's the four weeks going?",
        True, must_reject=False,
    ),
    FidelityFixture(
        "production_draft_dadscientist",
        "Rate this stack. Upon waking: Ivabradine 5mg, Modafinil 150mg.",
        "Have you started the full stack as listed, or are you still "
        "building up to it?", True, must_reject=False,
    ),
    # --- assertions that put a claim in his mouth: MUST FAIL ---
    FidelityFixture(
        "asserts_a_cause_he_never_gave",
        "Sleep has been rough since I moved the modafinil earlier.",
        "Moving the modafinil earlier wrecked your sleep - are you reverting?",
        False,
    ),
    FidelityFixture(
        "asserts_an_action_he_never_took",
        "Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        "You added the MotsC - any sides?", False,
    ),
    FidelityFixture(
        "asserts_a_causal_claim",
        "These were my previous labs on 6/18 when I was taking cardarine.",
        "Since Cardarine raised your ApoB, are you dropping it?", False,
    ),
    FidelityFixture(
        "asserts_he_said_something_he_did_not",
        "Elevated ApoB can happen in these cases.",
        "You said elevated ApoB usually happens - what drives that?", False,
    ),
    FidelityFixture(
        "asserts_a_duration",
        "Rate this stack. Upon waking: Ivabradine 5mg, Modafinil 150mg.",
        "You've been running this stack for months - what changed?", False,
    ),
    FidelityFixture(
        "asserts_a_completed_outcome",
        "Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31.",
        "Now that the four weeks are done, what did you notice?", False,
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
    if not isinstance(raw, dict):
        return FidelityVerdict(False, "unreadable_verdict")
    reason = str(raw.get("reason") or "")
    # The split shape is the shipped contract: assertion fails, presupposition
    # passes. A presupposing question is left alone deliberately.
    if "asserts" in raw:
        asserts = raw.get("asserts")
        if not isinstance(asserts, bool):
            return FidelityVerdict(False, "unreadable_verdict")
        return FidelityVerdict(not asserts, reason if asserts else "")
    faithful = raw.get("faithful")
    if not isinstance(faithful, bool):
        return FidelityVerdict(False, "unreadable_verdict")
    return FidelityVerdict(faithful, reason)
