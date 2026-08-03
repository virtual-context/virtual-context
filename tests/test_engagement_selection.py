"""Timing, resolution, fidelity, and selection for the engagement pipeline.

Timing and resolution encode a judgement the owner made explicitly: an
answered thread is not a dead thread. What dies is the *anticipatory*
question — "did you start?" stops making sense once he says he started —
while the outcome question becomes newly available. Rejecting the whole
thread on the first sign of an answer would throw away the best material,
and skipping is itself a cost, because an absent participant breaks the
illusion as badly as an inattentive one.

The fidelity gate fails closed on configuration: a gate with no judge
refuses to run rather than passing drafts through unchecked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.core.discord_snowflake import datetime_to_snowflake_floor
from virtual_context.core.engagement import (
    ADVERSARIAL_FIDELITY_FIXTURES,
    Candidate,
    FidelityGateNotConfigured,
    assess_thread,
    rank_candidates,
    run_fidelity_gate,
    select_question,
    timed_followup_eligibility,
)

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
P3 = "1524917968440524990"
BIGTEX = "actor:discord:1338726888809697364"


def _mid(moment):
    return str(datetime_to_snowflake_floor(moment) + 7)


def _cand(days_ago: float, text="Adding ss31 (5mg) for 4 weeks.", turn="ct-1"):
    sent = NOW - timedelta(days=days_ago)
    return Candidate(
        canonical_turn_id=turn, source_message_id=_mid(sent),
        actor_id=BIGTEX, channel_id=P3, text=text, sent_at=sent,
    )


# ------------------------------------------------------------------ Task 8


class TestTimedFollowupTiming:
    def test_same_day_material_is_rejected(self):
        ok, reason = timed_followup_eligibility(_cand(0.2), now=NOW)
        assert ok is False
        assert reason == "same_day"

    def test_one_day_old_is_still_too_soon(self):
        ok, reason = timed_followup_eligibility(_cand(1.5), now=NOW)
        assert ok is False
        assert reason == "too_recent"

    @pytest.mark.parametrize("days", [2.0, 3.5, 7.0])
    def test_two_to_seven_days_is_eligible(self, days):
        ok, reason = timed_followup_eligibility(_cand(days), now=NOW)
        assert ok is True
        assert reason == ""

    def test_older_than_seven_days_is_not_a_timed_followup(self):
        ok, reason = timed_followup_eligibility(_cand(9), now=NOW)
        assert ok is False
        assert reason == "too_old_for_timed_followup"

    def test_future_dated_material_is_rejected(self):
        ok, reason = timed_followup_eligibility(_cand(-1), now=NOW)
        assert ok is False
        assert reason == "future_dated"


class TestResolution:
    def test_an_unanswered_thread_stays_anticipatory(self):
        state = assess_thread(_cand(3), later_texts=[], now=NOW)
        assert state.resolved is False
        assert state.stance == "anticipatory"

    def test_an_answered_thread_is_not_dead_it_turns_into_an_outcome(self):
        """The owner's override: reject the stale question, not the thread."""
        state = assess_thread(
            _cand(5),
            later_texts=[("I started the ss31 on monday", NOW - timedelta(days=4))],
            now=NOW,
            subject_terms=["ss31"],
        )
        assert state.resolved is True
        assert state.stance == "outcome"
        assert state.blocks_candidate is False

    def test_material_the_member_reopened_today_is_blocked(self):
        """Asking about something he is talking about right now is noise."""
        state = assess_thread(
            _cand(4),
            later_texts=[("ss31 update: week two", NOW - timedelta(hours=2))],
            now=NOW,
            subject_terms=["ss31"],
        )
        assert state.blocks_candidate is True
        assert state.reason == "member_posted_today"

    def test_unrelated_later_messages_do_not_resolve_the_thread(self):
        state = assess_thread(
            _cand(3),
            later_texts=[("anyone lifting today", NOW - timedelta(days=1))],
            now=NOW,
            subject_terms=["ss31"],
        )
        assert state.resolved is False
        assert state.stance == "anticipatory"


# ------------------------------------------------------------------ Task 9


class TestFidelityGate:
    def test_a_gate_with_no_judge_refuses_to_run(self):
        """Config failure must stop the pipeline, never skip the check."""
        with pytest.raises(FidelityGateNotConfigured):
            run_fidelity_gate(
                quote="Adding ss31 (5mg) for 4 weeks.",
                draft="Have you started the SS-31 yet?",
                judge=None,
            )

    def test_a_faithful_draft_passes(self):
        verdict = run_fidelity_gate(
            quote="Adding ss31 (5mg) for 4 weeks.",
            draft="Have you started the SS-31 yet?",
            judge=lambda **kw: {"faithful": True, "reason": ""},
        )
        assert verdict.faithful is True

    def test_an_unfaithful_draft_is_rejected_with_the_judges_reason(self):
        verdict = run_fidelity_gate(
            quote="Adding ss31 (5mg) for 4 weeks.",
            draft="You started a four-week SS-31 run. How is recovery?",
            judge=lambda **kw: {
                "faithful": False, "reason": "planned_became_started",
            },
        )
        assert verdict.faithful is False
        assert verdict.reason == "planned_became_started"

    def test_a_malformed_judge_response_fails_closed(self):
        verdict = run_fidelity_gate(
            quote="q", draft="d", judge=lambda **kw: "not a verdict",
        )
        assert verdict.faithful is False
        assert verdict.reason == "unreadable_verdict"

    def test_a_judge_that_raises_fails_closed(self):
        def _boom(**kw):
            raise RuntimeError("provider down")

        verdict = run_fidelity_gate(quote="q", draft="d", judge=_boom)
        assert verdict.faithful is False
        assert verdict.reason == "judge_error"

    def test_the_adversarial_fixtures_carry_the_specs_own_pairs(self):
        bodies = [f.draft.lower() for f in ADVERSARIAL_FIDELITY_FIXTURES]
        assert any("you started" in b for b in bodies)
        assert any("usually" in b for b in bodies)
        # Every fixture states the verdict it must produce.
        assert all(f.expected_faithful is False for f in
                   ADVERSARIAL_FIDELITY_FIXTURES if f.must_reject)
        assert any(f.expected_faithful for f in ADVERSARIAL_FIDELITY_FIXTURES)


# ----------------------------------------------------------------- Task 10


class TestRankingAndSelection:
    def test_unresolved_and_specific_outranks_generic(self):
        a = _cand(3, text="Adding ss31 5mg for 4 weeks", turn="ct-a")
        b = _cand(3, text="ok", turn="ct-b")
        ranked = rank_candidates([b, a])
        assert ranked[0].canonical_turn_id == "ct-a"

    def test_a_recently_used_actor_is_penalised(self):
        """A member asked yesterday loses to an equally good stranger."""
        import dataclasses

        a = _cand(3, text="Adding ss31 5mg for 4 weeks", turn="ct-a")
        b = dataclasses.replace(
            _cand(3, text="Adding ss31 5mg for 4 weeks", turn="ct-b"),
            actor_id="actor:discord:999",
        )
        # Identical text, so only the recency penalty can separate them.
        assert rank_candidates([a, b])[0].canonical_turn_id == "ct-a"
        ranked = rank_candidates([a, b], recent_actor_ids=[a.actor_id])
        assert ranked[0].canonical_turn_id == "ct-b"

    def test_selection_falls_back_to_a_broader_question(self):
        outcome = select_question(
            verified=[], rejections=[], channel_id=P3,
            broader_questions={P3: ["What still argues for secretagogues?"]},
        )
        assert outcome.kind == "broader"
        assert outcome.question

    def test_selection_skips_and_names_the_stage_when_nothing_survives(self):
        outcome = select_question(
            verified=[], rejections=[], channel_id=P3, broader_questions={},
        )
        assert outcome.kind == "skip"
        assert outcome.reason
        assert outcome.skip_stage

    def test_a_skip_reports_the_stage_that_rejected_everything(self):
        from virtual_context.core.engagement import Rejection

        outcome = select_question(
            verified=[], rejections=[
                Rejection("ct-1", "verify", "author_mismatch"),
                Rejection("ct-2", "verify", "author_mismatch"),
                Rejection("ct-3", "collect", "channel_not_sourceable"),
            ],
            channel_id=P3, broader_questions={},
        )
        assert outcome.kind == "skip"
        assert outcome.skip_stage == "verify"
        assert "author_mismatch" in outcome.reason
