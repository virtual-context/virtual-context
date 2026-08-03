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
    "what the quoted words actually say — add no claim, timing, intention, "
    "causality, action, outcome, or degree of certainty that is not in them."
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
    stance: str,
    composer: Callable[..., Any] | None,
    sender: str = "",
    channel_label: str = "",
) -> Draft:
    """Ask the configured model for one question about *candidate*."""
    if composer is None:
        raise DraftComposerNotConfigured(
            "no draft composer model is configured; configure "
            "engagement.fidelity_judge_model's composition counterpart or "
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
