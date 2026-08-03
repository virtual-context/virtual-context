"""Personal continuation: the tag has to be earned.

A timed follow-up is bounded by the clock — it revisits something said two
to seven days ago. A personal continuation is bounded by evidence instead:
it may reach back to anything the member said, but only when his own words
contain something specific to continue. That is the entire distinction, and
it is why most of the corpus is invisible without this path: material older
than a week is not stale, it is simply outside the only window that was
implemented.

The connection test is the feature. Without it a name is stapled to a
question that could be asked of anyone, which the specification names as the
failure and illustrates: "what compound gave you the biggest benefit?" is
answerable by every member and teaches nothing about this one. So a
candidate qualifies only when a hook is found, the hook names which kind of
item it is, and the evidence is the member's own words rather than a
paraphrase.

Detection is model-judged because the nine kinds are semantic. With no
detector configured this RAISES: silently admitting every candidate would
turn the guard into a formality and staple names to generic questions at
scale.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

# The nine item kinds a personal question may rest on, from the spec.
CONTINUATION_HOOK_KINDS: tuple[str, ...] = (
    "unresolved_experiment_or_protocol",
    "symptom_or_side_effect_tracked",
    "pending_or_surprising_labs",
    "dose_or_compound_change",
    "stated_decision_rule",
    "personal_preference_or_tradeoff",
    "contradiction_or_change_of_view",
    "specific_practical_concern",
    "prior_result_unclear",
)

HOOK_DETECTOR_SYSTEM_PROMPT = (
    "You are given one message a member wrote. Decide whether it contains "
    "something specific enough to continue a conversation about, and if so "
    "which kind.\n\n"
    "The kinds are exactly:\n"
    + "\n".join(f"  - {k}" for k in CONTINUATION_HOOK_KINDS)
    + "\n\nA hook must be specific to THIS member. If the message would "
    "support only a question that could be asked of anyone, there is no "
    "hook.\n\n"
    'Reply ONLY compact JSON: {"kind": "<one of the kinds, or empty>", '
    '"evidence": "<the member\'s exact words carrying it, verbatim>", '
    '"reason": "<short>"}.'
)


class HookDetectorNotConfigured(RuntimeError):
    """Raised when the connection test is attempted with no detector."""


@dataclass(frozen=True)
class ContinuationHook:
    kind: str
    evidence: str

    @property
    def usable(self) -> bool:
        return bool(self.kind) and self.kind in CONTINUATION_HOOK_KINDS


def find_continuation_hook(
    *, quote: str, detector: Callable[..., Any] | None,
) -> tuple[ContinuationHook | None, str]:
    """The verified item this member's words offer, or ``None`` and a reason."""
    if detector is None:
        raise HookDetectorNotConfigured(
            "no continuation hook detector is configured; without the "
            "connection test a member's name would be attached to questions "
            "that could be asked of anyone."
        )
    body = (quote or "").strip()
    if not body:
        return None, "empty_quote"
    try:
        raw = detector(quote=body)
    except Exception:
        return None, "detector_error"
    if not isinstance(raw, dict):
        return None, "unreadable_hook_verdict"
    kind = str(raw.get("kind") or "").strip()
    if not kind:
        return None, "no_specific_hook"
    if kind not in CONTINUATION_HOOK_KINDS:
        return None, "unknown_hook_kind"
    evidence = str(raw.get("evidence") or "").strip()
    if not evidence:
        return None, "hook_without_evidence"
    # The evidence must be the member's own words, not a paraphrase: a
    # rewritten "quote" is how a claim drifts from what he actually said.
    if evidence.lower() not in body.lower():
        return None, "evidence_not_verbatim"
    return ContinuationHook(kind, evidence), ""


def qualify_candidates(
    verified: list,
    *,
    now,
    detector: Callable[..., Any] | None,
    later_texts_for=None,
    subject_terms_for=None,
) -> tuple[list, list]:
    """Label every verified candidate with the type it can actually carry.

    One pool, not two tiers. A timed follow-up and a continuation compete on
    the same ranking because the specification orders selection by verified
    personal relevance; only the generated channel question sits beneath
    them. Splitting them into separate pools would have made the older,
    larger set a fallback by construction.

    Both types get the same later-context treatment, because a thread the
    member has already carried forward is equally stale whether it is three
    days old or three months. Returns ``(qualified, rejections)``.
    """
    from .candidates import Rejection
    from .timing import assess_thread, timed_followup_eligibility

    qualified: list = []
    rejections: list[Rejection] = []
    for candidate in verified or []:
        turn_id = getattr(candidate, "canonical_turn_id", "")
        timed_ok, _why = timed_followup_eligibility(candidate, now=now)
        question_type = "timed"
        hook_kind = ""
        if not timed_ok:
            # Outside the clock window it can still qualify on evidence, and
            # that is the only thing separating the two types.
            hook, reason = find_continuation_hook(
                quote=getattr(candidate, "text", ""), detector=detector,
            )
            if hook is None:
                rejections.append(Rejection(turn_id, "continuation", reason))
                continue
            question_type, hook_kind = "personal", hook.kind

        later = (later_texts_for or (lambda c: []))(candidate)
        terms = (subject_terms_for or (lambda c: []))(candidate)
        state = assess_thread(
            candidate, later_texts=later, now=now, subject_terms=terms,
        )
        if state.blocks_candidate:
            rejections.append(
                Rejection(turn_id, "resolution", state.reason or "blocked"),
            )
            continue
        qualified.append(
            dataclasses.replace(
                candidate, question_type=question_type, hook_kind=hook_kind,
            )
        )
    return qualified, rejections
