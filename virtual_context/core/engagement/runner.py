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
from .history import check_repetition
from .live_source import select_live_verified
from .poster import PostRefused, post_question
from .report import DryRunReport
from .select import apply_fidelity_outcome, rank_candidates, select_question
from .verify import verify_candidates


# How many candidates a run will draft before giving the day up. Small on
# purpose: each attempt costs a composer call, a judge call and a live source
# request, and the failures worth surviving are single unlucky calls rather
# than a systematically unusable pool.
DRAFT_ATTEMPT_CAP = 3


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

    # Repetition is checked BEFORE ranking so a repeat never costs a rank
    # slot, a live source request or a model call. These three rules need
    # only the candidate: who was tagged recently, which threads have been
    # mined, how busy the channel has been.
    #
    # The fourth rule — question similarity — cannot run here. It fingerprints
    # the DRAFT, which does not exist yet, and check_repetition skips that
    # rule when the text is empty. Calling once here would ship
    # question_recently_asked as a rule that can never fire, so it is checked
    # again after each draft below.
    fresh: list = []
    for candidate in qualified:
        # The cap counts what has landed in a channel, so it must be given
        # the DESTINATION. Passing the source channel compared a candidate's
        # origin against the ledger's destination and the rule never matched.
        #
        # A destination that is not the source channel is the rehearsal
        # fallback: private, one watcher, nothing to protect. The cap is
        # scoped out there and re-arms by itself the moment the allowlist
        # widens, because then the destination IS the source channel and the
        # condition below is false. No second list, nothing to remember.
        destination = _post_target(allowlist, candidate)
        posting_to_community = bool(destination) and destination == getattr(
            candidate, "channel_id", "",
        )
        repeat = check_repetition(
            history=history, now=now,
            actor_id=getattr(candidate, "actor_id", ""),
            channel_id=destination,
            source_message_ids=(
                (candidate.source_message_id,)
                if getattr(candidate, "source_message_id", "") else ()
            ),
            question_text="",
            apply_channel_cap=posting_to_community,
        )
        if repeat is not None:
            rejections.append(Rejection(
                getattr(candidate, "canonical_turn_id", ""),
                "history", repeat.reason, repeat.detail,
            ))
            continue
        fresh.append(candidate)

    ranked = rank_candidates(fresh)

    # A failed draft costs a candidate, not the day. One flaky model call
    # against a pool of qualified candidates should not decide that there was
    # nothing worth asking; the run advances to the next-ranked one.
    #
    # Bounded deliberately. Every attempt spends model calls and a live source
    # request, so this walks a few candidates rather than the ranking: the
    # difference worth having is between "one unlucky call" and "no good
    # question today", and that is settled within a handful of attempts.
    chosen = None
    draft = verdict = outcome = None
    remaining = list(ranked)
    attempts = 0
    while remaining and attempts < DRAFT_ATTEMPT_CAP:
        # Live verification runs per attempt, inside the loop. A pass belongs
        # to one message in one run and is not transferable between
        # candidates, so falling through must re-verify rather than carry the
        # previous candidate's verdict forward.
        candidate, live_rejections = select_live_verified(
            remaining, fetcher=source_fetcher,
        )
        rejections += list(live_rejections)
        if candidate is None:
            break
        remaining = remaining[remaining.index(candidate) + 1:]
        attempts += 1

        attempt_draft, attempt_verdict = drafter(candidate)

        # Now the draft exists, so the question-similarity rule can run. A
        # near-duplicate of something already asked is rejected here and the
        # walk moves to the next candidate rather than losing the day.
        repeat = check_repetition(
            history=history, now=now,
            actor_id=getattr(candidate, "actor_id", ""),
            channel_id=_post_target(allowlist, candidate),
            source_message_ids=(),
            question_text=attempt_draft.text or "",
            # Already decided before ranking; this pass is for similarity.
            apply_channel_cap=False,
        )
        if repeat is not None:
            rejections.append(Rejection(
                getattr(candidate, "canonical_turn_id", ""),
                "history", repeat.reason, repeat.detail,
            ))
            continue

        attempt_outcome = apply_fidelity_outcome(
            select_question(verified=[candidate], rejections=rejections,
                            channel_id=candidate.channel_id),
            verdicts=[attempt_verdict],
        )
        if attempt_outcome.kind != "skip" and (attempt_draft.text or "").strip():
            chosen = candidate
            draft, verdict, outcome = (
                attempt_draft, attempt_verdict, attempt_outcome,
            )
            break
        # A rejected attempt is a named, counted row. Without this, a run that
        # tried three candidates and rejected all three reads identically to
        # one where nothing qualified — and those need different responses.
        rejections.append(Rejection(
            getattr(candidate, "canonical_turn_id", ""),
            "draft",
            attempt_outcome.skip_stage or "draft_rejected",
            attempt_draft.reason
            or getattr(attempt_verdict, "reason", "")
            or attempt_outcome.reason,
        ))

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
        return RunResult(
            report=report,
            refused=("every_draft_rejected" if attempts
                     else "no_verified_candidate"),
            rejections=rejections,
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
            delivery_body=getattr(draft, "delivery_body", "") or "",
            channel_id=_post_target(allowlist, chosen),
            verification=_verification_for(chosen, live_rejections),
            history=history, sender=message_sender, now=now,
            question_type=outcome.kind,
        )
    except PostRefused as exc:
        return RunResult(report=report, refused=str(exc),
                         rejections=rejections)
    return RunResult(report=report, posted_message_id=sent.message_id,
                     rejections=rejections)


def _post_target(allowlist, candidate=None) -> str:
    """Where this question goes: the source channel if that is permitted.

    A follow-up belongs under the message it follows up on. When the source
    channel is on the shipped post list, that is the destination and the
    send can carry a real reply reference. Otherwise it falls back to the
    single rehearsal destination, which is a testing artifact and must not
    decide what the feature is.

    So widening POST_CHANNEL_IDS to include the community channels is
    sufficient on its own to turn every post into a true in-channel reply,
    with no further code change. An ambiguous list still resolves to "",
    which the poster refuses.
    """
    permitted = tuple(getattr(allowlist, "post_channel_ids", ()) or ())
    source = str(getattr(candidate, "channel_id", "") or "")
    if source and source in permitted:
        return source
    targets = sorted(permitted)
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
