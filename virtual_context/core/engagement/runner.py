"""One run of the daily job, from candidates to a posted question.

This is the entrypoint something outside the process invokes once a day. It
installs nothing, schedules nothing and holds no credential: every capability
that reaches the world — the source fetcher, the sender, the models — arrives
as a callable the caller supplies, so this module can be read end to end
without wondering what it can reach.

Dry run is the default and posting is a separate, explicit argument. A caller
that forgets the flag produces a report and sends nothing, which is the
behaviour you want from the mistake that is easiest to make.

``post`` is intent and belongs to the caller. Permission does not: whether
this build may post at all is shipped configuration read by the poster, and
there is deliberately no parameter here that reaches it. A caller can decline
to post; it cannot grant itself the right to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .candidates import Rejection, collect_candidates
from .live_source import select_live_verified
from .poster import PostRefused, post_question
from .report import DryRunReport
from .select import apply_fidelity_outcome, rank_candidates, select_question
from .verify import verify_candidates


@dataclass
class RunResult:
    report: DryRunReport
    posted_message_id: str = ""
    refused: str = ""
    rejections: list = field(default_factory=list)


def run_once(
    *,
    results,
    sources,
    senders,
    allowlist,
    history,
    now: datetime,
    conversation_id: str,
    qualifier: Callable[..., Any],
    drafter: Callable[..., Any],
    source_fetcher: Callable[..., Any],
    message_sender: Callable[..., Any] | None = None,
    post: bool = False,
) -> RunResult:
    """Select, verify, draft, and optionally send one question.

    ``qualifier`` labels candidates with the type they can carry and
    ``drafter`` turns one into a gated question; both are injected so this
    function owns the order of operations and nothing else. Returns the
    report whether or not anything was sent.
    """
    kept, rejections = collect_candidates(
        results, allowlist=allowlist, senders=senders,
    )
    verified, verify_rejections = verify_candidates(kept, sources)
    rejections = list(rejections) + list(verify_rejections)

    qualified, qualify_rejections = qualifier(verified, now=now)
    rejections += list(qualify_rejections)

    ranked = rank_candidates(qualified)
    # The source is re-checked here, immediately before a question is drafted
    # from it, so the verdict a post relies on belongs to this run.
    chosen, live_rejections = select_live_verified(
        ranked, fetcher=source_fetcher,
    )
    rejections += list(live_rejections)

    report = DryRunReport(
        generated_at=now, conversation_id=conversation_id,
        channel_id=getattr(chosen, "channel_id", "") if chosen else "",
        channel_label=(
            allowlist.label_for(chosen.channel_id) if chosen else ""
        ),
        considered=len(list(results or [])),
        rejections=rejections,
        live_verified=chosen is not None,
    )
    if chosen is None:
        report.apply_outcome(select_question(
            verified=[], rejections=rejections,
            channel_id=getattr(allowlist, "post_channel_ids", ("",)) and "",
        ))
        return RunResult(report=report, refused="no_verified_candidate",
                         rejections=rejections)

    draft, verdict = drafter(chosen)
    outcome = apply_fidelity_outcome(
        select_question(verified=[chosen], rejections=rejections,
                        channel_id=chosen.channel_id),
        verdicts=[verdict],
    )
    report.apply_outcome(outcome)
    report.quote = chosen.text[:500]
    report.question = draft.text if outcome.kind != "skip" else ""
    report.fidelity = {"faithful": verdict.faithful, "reason": verdict.reason}

    if not post or outcome.kind == "skip":
        return RunResult(report=report, rejections=rejections,
                         refused="" if post else "dry_run")

    # Posting is deliberately the last thing, after every guard, and it
    # refuses loudly rather than degrading to a dry run.
    try:
        sent = post_question(
            candidate=chosen, question=draft.text,
            channel_id=_post_target(allowlist),
            verification=_verification_for(chosen, live_rejections),
            history=history, sender=message_sender, now=now,
            question_type=outcome.kind,
        )
    except PostRefused as exc:
        return RunResult(report=report, refused=str(exc),
                         rejections=rejections)
    return RunResult(report=report, posted_message_id=sent.message_id,
                     rejections=rejections)


def _post_target(allowlist) -> str:
    """The single permitted destination, taken from the shipped list."""
    targets = sorted(getattr(allowlist, "post_channel_ids", ()) or ())
    return targets[0] if len(targets) == 1 else ""


def _verification_for(candidate, live_rejections):
    """The pass belonging to this candidate in this run.

    ``select_live_verified`` returns the chosen candidate only after a clean
    verdict, so reaching here means one exists for exactly this message.
    """
    from .live_source import LiveVerification

    return LiveVerification(
        True, "", "", str(getattr(candidate, "source_message_id", "") or ""),
    )
