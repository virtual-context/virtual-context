"""Draft one question from one verified quote, and keep them separate.

The member's words and the claim about who spoke them travel as different
arguments, never concatenated. Stored guild bodies carry a ``"<sender>: "``
prefix written at ingest — 387 of 400 sampled production rows begin with one
— so a composer handed the raw body would receive the speaker label as part
of the message and could echo it, or reason about it as something the member
wrote. Splitting body from attribution here is the same separation the rest
of this pipeline enforces, applied at the point where text is generated.

A composer that cannot run is not permission to post something unchecked:
with no model configured this RAISES, and a composer that errors or returns
nothing yields an unusable draft rather than a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class DraftComposerNotConfigured(RuntimeError):
    """Raised when composition is attempted with no model configured."""


@dataclass(frozen=True)
class Draft:
    text: str = ""
    reason: str = ""
    # What actually gets sent, when presentation differs from the question —
    # review framing, a quoted original, a mention. Empty means the question
    # is sent as-is. The repetition fingerprint always keys on ``text``, never
    # on this, so how a question is presented cannot change whether it counts
    # as a repeat.
    delivery_body: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.text.strip()) and not self.reason


# Spec §Tone, carried to the composer so the owner reviews the real product.
TONE_CONSTRAINTS = (
    "Write as one man among men: concise, curious, warm. One clean question, "
    "normally one or two sentences. No corporate engagement language. No "
    "'Question of the day'. Never explain that the post was scheduled or "
    "generated. Never mention tools, memory, routing, models, prompts, or "
    "infrastructure. Light wit is welcome; do not force it. Ask only about "
    "what the quoted words actually say — add no claim, intention, "
    "causality, action, outcome, or degree of certainty that is not in "
    "them.\n\n"
    "WHEN he wrote it is an exception, and the only one. The message's own "
    "timestamp is verified metadata, re-checked against the live source "
    "before anything is sent, so referring to it puts nothing in his mouth: "
    "'you mentioned this last week' is a claim about when he spoke. A claim "
    "about what he DID in that time is not — 'you've been running it a "
    "week' says something about him the message never said. The permission "
    "is about the SOURCE of the fact, not the kind of fact."
)

_STANCE_GUIDANCE = {
    "anticipatory": (
        "The member described something planned or underway and has not "
        "reported back. Ask whether it happened or how it is going, without "
        "assuming it started."
    ),
    "outcome": (
        "The member has since reported movement on this. Do not ask whether "
        "it began; ask about the result, what changed, or what surprised him."
    ),
}


def strip_speaker_prefix(body: str, sender: str) -> str:
    """Remove a leading ``"<sender>: "`` written by ingest, and nothing else.

    Only this row's own sender is stripped. A generic strip-to-first-colon
    would mangle ordinary text — "Note: I stopped the SS-31" is the member's
    sentence, not an attribution header.
    """
    text = body or ""
    name = (sender or "").strip()
    if not name:
        return text
    prefix = f"{name}:"
    if text.startswith(prefix):
        return text[len(prefix):].lstrip()
    return text


def compose_draft(
    *,
    candidate,
    composer: Callable[..., Any] | None,
    sender: str = "",
    channel_label: str = "",
) -> Draft:
    """Ask the configured model for one question about *candidate*.

    The stance is read from the candidate rather than passed in. Thread
    assessment during qualification already decided it, and recomputing it
    here would be a second ruler for the same measurement.
    """
    stance = str(getattr(candidate, "stance", "") or "")
    if composer is None:
        raise DraftComposerNotConfigured(
            "no draft composer model is configured. No such setting exists "
            "yet — the judge is assembly.engagement_fidelity_judge_model and "
            "its composition counterpart has not been decided. Until it is, "
            "the caller must inject a composer explicitly, or "
            "disable the pipeline. Composing without a model would mean "
            "posting text nobody generated from the verified quote."
        )
    quote = strip_speaker_prefix(
        getattr(candidate, "text", ""), sender or getattr(candidate, "sender", ""),
    )
    try:
        raw = composer(
            quote=quote,
            handle=sender or getattr(candidate, "sender", ""),
            stance=stance,
            stance_guidance=_STANCE_GUIDANCE.get(stance, ""),
            channel_label=channel_label,
            sent_at=getattr(candidate, "sent_at", None),
            tone=TONE_CONSTRAINTS,
        )
    except Exception:
        return Draft("", "composer_error")
    text = str(raw or "").strip()
    if not text:
        return Draft("", "empty_draft")
    return Draft(text, "")


# A continuation's own guidance. The timed path's prompt is deliberately
# untouched: that one asks about the status of something recent, while this
# one reopens a specific thing the member said at any distance, and the two
# need different framing. The hook is passed as structure, and the evidence
# is his verbatim words, so the draft is built from what he actually wrote
# rather than from a summary of it.
CONTINUATION_GUIDANCE = (
    "This is not a recent-status check. The member said something specific, "
    "possibly a while ago, that was never resolved or never explained. Ask "
    "about THAT. Quote or closely echo his own words so he recognises what "
    "you are referring to, then ask the one thing a curious reader would "
    "still want to know: what happened, what he decided, or what he meant. "
    "Do not state that anything happened, changed, or was caused - ask. If "
    "the question you are about to write could be asked of any member, it is "
    "the wrong question; the tag is earned by his specific words or not at all."
)

_HOOK_FRAMING = {
    "unresolved_experiment_or_protocol": "an experiment he never reported back on",
    "symptom_or_side_effect_tracked": "a symptom he was tracking",
    "pending_or_surprising_labs": "lab work that was pending or surprised him",
    "dose_or_compound_change": "a dose or compound he was changing",
    "stated_decision_rule": "a rule he gave for how he would decide",
    "personal_preference_or_tradeoff": "a preference or tradeoff he stated",
    "contradiction_or_change_of_view": "a view he appeared to change",
    "specific_practical_concern": "a specific practical concern he raised",
    "prior_result_unclear": "a result whose meaning he left unclear",
}


def compose_continuation_draft(
    *,
    candidate,
    composer: Callable[..., Any] | None,
    sender: str = "",
    channel_label: str = "",
) -> Draft:
    """Draft a continuation from the hook the candidate already carries.

    The hook and its evidence are read from the candidate, not accepted as
    arguments. They were computed during qualification, and that computation
    is what let this candidate through the gate; taking them as parameters
    would let a caller supply a freshly recomputed hook that never passed it.
    A recomputation can disagree with the original, and the disagreement is
    invisible afterwards because both values look equally plausible.
    """
    if composer is None:
        raise DraftComposerNotConfigured(
            "no draft composer model is configured; a continuation cannot be "
            "written without one, and posting an unwritten question is not an "
            "available outcome."
        )
    quote = strip_speaker_prefix(
        getattr(candidate, "text", ""), sender or getattr(candidate, "sender", ""),
    )
    hook_kind = str(getattr(candidate, "hook_kind", "") or "")
    verbatim = str(getattr(candidate, "hook_evidence", "") or "").strip()
    if not hook_kind:
        return Draft("", "no_qualified_hook")
    if not verbatim or verbatim.lower() not in quote.lower():
        return Draft("", "evidence_not_in_quote")
    try:
        raw = composer(
            quote=quote,
            evidence=verbatim,
            hook_kind=hook_kind,
            hook_framing=_HOOK_FRAMING.get(hook_kind, "something he said"),
            handle=sender or getattr(candidate, "sender", ""),
            stance="continuation",
            stance_guidance=CONTINUATION_GUIDANCE,
            channel_label=channel_label,
            sent_at=getattr(candidate, "sent_at", None),
            tone=TONE_CONSTRAINTS,
        )
    except Exception:
        return Draft("", "composer_error")
    text = str(raw or "").strip()
    if not text:
        return Draft("", "empty_draft")
    return Draft(text, "")
