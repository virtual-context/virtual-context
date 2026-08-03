"""Rank surviving candidates, fall back, or skip with a reason.

Ranking prefers material that produces a question only this member could
answer: specific, recent enough to be live, and not from someone who was
asked yesterday. It is a cheap ordering over an already-verified set, not a
second gate — nothing here may admit a candidate that verification rejected.

Skipping is a real cost, so it is the last option rather than the safe
default. When it happens the outcome names the stage that rejected
everything and the reason that dominated, because a silent skip cannot be
told apart from a broken job.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectionOutcome:
    kind: str                 # "personal" | "broader" | "skip"
    question: str = ""
    candidate: object | None = None
    reason: str = ""
    skip_stage: str = ""
    considered: int = 0


def _specificity(text: str) -> float:
    """Rough proxy: concrete detail, not length alone."""
    body = (text or "").strip()
    if not body:
        return 0.0
    digits = sum(ch.isdigit() for ch in body)
    words = len(body.split())
    return min(words / 25.0, 1.0) + min(digits / 6.0, 1.0)


def rank_candidates(
    candidates: list,
    *,
    recent_actor_ids: list[str] | None = None,
    recent_channel_ids: list[str] | None = None,
) -> list:
    """Order verified candidates best-first. Never admits or rejects."""
    recent_actors = set(recent_actor_ids or [])
    recent_channels = set(recent_channel_ids or [])

    def _score(candidate) -> tuple:
        score = _specificity(getattr(candidate, "text", ""))
        if getattr(candidate, "actor_id", "") in recent_actors:
            score -= 1.5
        if getattr(candidate, "channel_id", "") in recent_channels:
            score -= 0.5
        # Stable tiebreak so a run is reproducible for review.
        return (-score, getattr(candidate, "canonical_turn_id", ""))

    return sorted(candidates, key=_score)


def _dominant_rejection(rejections: list) -> tuple[str, str]:
    if not rejections:
        return "", ""
    stages = Counter(getattr(r, "stage", "") for r in rejections)
    stage = stages.most_common(1)[0][0]
    reasons = Counter(
        getattr(r, "reason", "") for r in rejections
        if getattr(r, "stage", "") == stage
    )
    return stage, reasons.most_common(1)[0][0]


def select_question(
    *,
    verified: list,
    rejections: list,
    channel_id: str,
    broader_questions: dict[str, list[str]] | None = None,
    recent_questions: list[str] | None = None,
    rng: random.Random | None = None,
) -> SelectionOutcome:
    """Choose a personal question, else a broader one, else skip."""
    considered = len(verified) + len(rejections)
    if verified:
        best = verified[0]
        return SelectionOutcome(
            kind="personal", candidate=best, considered=considered,
        )

    pool = list((broader_questions or {}).get(channel_id, []))
    already = set(recent_questions or [])
    fresh = [q for q in pool if q not in already]
    if fresh:
        chooser = rng or random.Random(0)
        return SelectionOutcome(
            kind="broader",
            question=chooser.choice(sorted(fresh)),
            considered=considered,
        )

    stage, reason = _dominant_rejection(rejections)
    if not stage:
        stage = "collect" if not pool else "broader"
        reason = "no_candidates" if not pool else "all_broader_questions_recent"
    return SelectionOutcome(
        kind="skip",
        reason=(
            f"nothing survived; dominant rejection at '{stage}': {reason}"
        ),
        skip_stage=stage,
        considered=considered,
    )


def apply_fidelity_outcome(outcome: SelectionOutcome, *, verdicts: list):
    """Downgrade a selection whose drafts all failed the fidelity gate.

    Reporting a run where every draft was rejected as a "personal" outcome
    states the question TYPE where the RESULT belongs, and reads as a working
    selection. A run that produced nothing postable is a skip, and the stage
    that rejected it is the fidelity gate.
    """
    if outcome.kind != "personal":
        return outcome
    if any(getattr(v, "faithful", False) for v in (verdicts or [])):
        return outcome
    reasons = [
        getattr(v, "reason", "") for v in (verdicts or [])
        if getattr(v, "reason", "")
    ]
    detail = "; ".join(reasons) if reasons else "no draft survived the gate"
    return SelectionOutcome(
        kind="skip",
        reason=f"every draft was rejected by the fidelity gate: {detail}",
        skip_stage="fidelity",
        considered=outcome.considered,
    )
