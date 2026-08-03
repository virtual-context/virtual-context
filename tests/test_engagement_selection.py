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

# The rehearsal configuration: community channels may be SOURCED from, and
# only the approved private rehearsal channel may be POSTED to, until the
# owner approves live posting.
REHEARSAL_ALLOWLIST = {
    "source_channel_ids": [
        "1524917968440524990", "1524917037787250834", "1524964360030785686",
        "1530567788949798963", "1524918613008580768",
    ],
    "post_channel_ids": ["1524946242499514418"],
    "labels": {"1524946242499514418": "#vasttest"},
}


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


# ------------------------------------------------- speaker-prefix stripping


class TestSpeakerPrefix:
    """Stored guild bodies carry a "<sender>: " prefix from ingest.

    387 of 400 sampled production rows begin with it. It is durable data,
    not a rendering artifact, so a composer fed the raw body would receive
    the speaker label as part of the member's words and could reproduce it
    or reason about it as content.
    """

    def test_the_rows_own_sender_prefix_is_removed(self):
        from virtual_context.core.engagement import strip_speaker_prefix

        assert strip_speaker_prefix(
            "Cashew King: @Vast Pinned my 80mg Test this morning",
            "Cashew King",
        ) == "@Vast Pinned my 80mg Test this morning"

    def test_a_body_without_the_prefix_is_untouched(self):
        from virtual_context.core.engagement import strip_speaker_prefix

        body = "@Vast Pinned my 80mg Test this morning"
        assert strip_speaker_prefix(body, "Cashew King") == body

    def test_only_the_rows_own_sender_is_stripped(self):
        """Never a generic strip-to-first-colon; that mangles real text."""
        from virtual_context.core.engagement import strip_speaker_prefix

        assert strip_speaker_prefix(
            "Note: I stopped the SS-31", "Cashew King",
        ) == "Note: I stopped the SS-31"
        assert strip_speaker_prefix(
            "Roo: something", "Cashew King",
        ) == "Roo: something"

    def test_an_empty_sender_strips_nothing(self):
        from virtual_context.core.engagement import strip_speaker_prefix

        assert strip_speaker_prefix("Roo: hi", "") == "Roo: hi"

    def test_the_collector_stores_the_members_words_alone(self):
        from virtual_context.core.engagement import collect_candidates, load_channel_allowlist
        from virtual_context.types import QuoteResult, SourceProvenance

        result = QuoteResult(
            text="Cashew King: Adding ss31 (5mg) for 4 weeks.",
            tag="", segment_ref="ct-1", source_scope="turn",
            matched_side="user",
            provenance=SourceProvenance(
                conversation_id="c", canonical_turn_id="ct-1",
                source_role="requester", actor_id=BIGTEX,
                audience_conversation_id="c", audience_attribution_version=1,
                origin_channel_id=P3, source_message_id=_mid(NOW),
            ),
        )
        allow = load_channel_allowlist(
            {"source_channel_ids": [P3], "post_channel_ids": [P3]},
        )
        kept, _ = collect_candidates(
            [result], allowlist=allow, senders={"ct-1": "Cashew King"},
        )
        assert kept[0].text == "Adding ss31 (5mg) for 4 weeks."
        assert kept[0].sender == "Cashew King"


# ----------------------------------------------------- draft composition


class TestDraftComposition:
    def test_a_composer_with_no_model_refuses_to_run(self):
        from virtual_context.core.engagement import (
            DraftComposerNotConfigured, compose_draft,
        )

        with pytest.raises(DraftComposerNotConfigured):
            compose_draft(candidate=_cand(3), stance="anticipatory", composer=None)

    def test_the_composer_never_receives_the_speaker_label_in_the_body(self):
        from virtual_context.core.engagement import compose_draft

        seen = {}

        def _composer(**kw):
            seen.update(kw)
            return "@BigTex have you started the SS-31 yet?"

        candidate = _cand(3, text="Adding ss31 (5mg) for 4 weeks.")
        compose_draft(
            candidate=candidate, stance="anticipatory", composer=_composer,
            sender="BigTex",
        )
        assert "BigTex:" not in seen["quote"]
        assert seen["quote"] == "Adding ss31 (5mg) for 4 weeks."
        # Attribution travels separately, as structure, never inside the body.
        assert seen["handle"] == "BigTex"

    def test_a_composer_that_raises_yields_no_draft(self):
        from virtual_context.core.engagement import compose_draft

        def _boom(**kw):
            raise RuntimeError("provider down")

        draft = compose_draft(
            candidate=_cand(3), stance="anticipatory", composer=_boom,
        )
        assert draft.text == ""
        assert draft.reason == "composer_error"

    def test_an_empty_draft_is_not_usable(self):
        from virtual_context.core.engagement import compose_draft

        draft = compose_draft(
            candidate=_cand(3), stance="anticipatory",
            composer=lambda **kw: "   ",
        )
        assert draft.usable is False
        assert draft.reason == "empty_draft"

    def test_the_stance_reaches_the_composer(self):
        from virtual_context.core.engagement import compose_draft

        seen = {}
        compose_draft(
            candidate=_cand(5), stance="outcome",
            composer=lambda **kw: seen.update(kw) or "how did it go?",
        )
        assert seen["stance"] == "outcome"


# --------------------------------------------- artifact honesty (a) and (b)


class TestRejectionAccounting:
    """Truncating the list made the commonest reason look like the only one."""

    def test_the_report_counts_every_reason_not_just_the_shown_ones(self):
        from virtual_context.core.engagement import DryRunReport, Rejection

        rejections = (
            [Rejection(f"ct-{i}", "collect", "channel_not_sourceable")
             for i in range(142)]
            + [Rejection(f"ct-r{i}", "timing", "too_recent") for i in range(51)]
            + [Rejection(f"ct-s{i}", "timing", "same_day") for i in range(37)]
            + [Rejection(f"ct-o{i}", "timing", "too_old_for_timed_followup")
               for i in range(3)]
        )
        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=P3,
            rejections=rejections, outcome_kind="skip",
        )
        rendered = report.render()
        assert "channel_not_sourceable" in rendered and "142" in rendered
        assert "too_recent" in rendered and "51" in rendered
        assert "same_day" in rendered and "37" in rendered
        assert "too_old_for_timed_followup" in rendered and "3" in rendered
        assert "233" in rendered  # the total, so nothing hides behind a cap

    def test_examples_are_capped_but_counts_are_not(self):
        from virtual_context.core.engagement import DryRunReport, Rejection

        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=P3,
            outcome_kind="skip",
            rejections=[
                Rejection(f"ct-{i}", "collect", "channel_not_sourceable")
                for i in range(50)
            ],
        )
        rendered = report.render()
        assert "50" in rendered
        assert rendered.count("ct-") <= 6

    def test_the_ladder_is_printed(self):
        from virtual_context.core.engagement import DryRunReport

        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=P3,
            outcome_kind="skip",
            ladder=[("input", 400), ("collected", 258), ("verified", 258),
                    ("timed_eligible", 167), ("composed", 2), ("postable", 0)],
        )
        rendered = report.render()
        for label in ("input", "collected", "verified", "timed_eligible",
                      "composed", "postable"):
            assert label in rendered


class TestFidelityDowngradesToSkip:
    def test_all_drafts_rejected_becomes_a_skip_not_a_personal(self):
        """Reporting the question TYPE where the RESULT belongs is a lie."""
        from virtual_context.core.engagement import (
            FidelityVerdict, SelectionOutcome, apply_fidelity_outcome,
        )

        chosen = SelectionOutcome(kind="personal", candidate=_cand(3),
                                  considered=400)
        final = apply_fidelity_outcome(
            chosen,
            verdicts=[FidelityVerdict(False, "planned_became_started")],
        )
        assert final.kind == "skip"
        assert final.skip_stage == "fidelity"
        assert "planned_became_started" in final.reason

    def test_a_surviving_draft_keeps_the_personal_outcome(self):
        from virtual_context.core.engagement import (
            FidelityVerdict, SelectionOutcome, apply_fidelity_outcome,
        )

        chosen = SelectionOutcome(kind="personal", candidate=_cand(3))
        final = apply_fidelity_outcome(
            chosen, verdicts=[FidelityVerdict(True, "")],
        )
        assert final.kind == "personal"

    def test_a_skip_is_left_alone(self):
        from virtual_context.core.engagement import (
            SelectionOutcome, apply_fidelity_outcome,
        )

        chosen = SelectionOutcome(kind="skip", reason="nothing", skip_stage="collect")
        assert apply_fidelity_outcome(chosen, verdicts=[]).skip_stage == "collect"


class TestPostChannelRestriction:
    def test_only_the_rehearsal_channel_is_postable(self):
        """Absence of posting code is not a safety boundary; config is."""
        from virtual_context.core.engagement import load_channel_allowlist

        allow = load_channel_allowlist(REHEARSAL_ALLOWLIST)
        assert allow.may_post("1524946242499514418") is True
        for community in (P3, "1524917037787250834", "1530567788949798963"):
            assert allow.may_post(community) is False
            assert allow.may_source(community) is True


class TestShippedAllowlist:
    """The boundary must be a shipped artifact, not each caller's argument."""

    def test_the_shipped_allowlist_posts_only_to_the_rehearsal_channel(self):
        from virtual_context.core.engagement import (
            POST_CHANNEL_IDS, rehearsal_allowlist,
        )

        allow = rehearsal_allowlist()
        assert POST_CHANNEL_IDS == ("1524946242499514418",)
        assert allow.may_post("1524946242499514418") is True

    def test_no_community_channel_is_postable_in_the_shipped_config(self):
        from virtual_context.core.engagement import (
            SOURCE_CHANNEL_IDS, rehearsal_allowlist,
        )

        allow = rehearsal_allowlist()
        for channel_id in SOURCE_CHANNEL_IDS:
            assert allow.may_source(channel_id) is True
            assert allow.may_post(channel_id) is False, (
                f"{channel_id} is postable in the shipped config; widening "
                "this must be a deliberate, reviewed edit"
            )

    def test_the_rehearsal_channel_is_never_a_source(self):
        from virtual_context.core.engagement import rehearsal_allowlist

        assert rehearsal_allowlist().may_source("1524946242499514418") is False


class TestAttributionStandard:
    """The gate judges attribution, not entailment.

    A question that assumes an answer attributes nothing and must pass; a
    draft that states something the quote does not support must fail. Every
    assertion here is made against the SHIPPED prompt and the SHIPPED
    fixtures, never a copy declared in this file.
    """

    def test_the_shipped_prompt_asks_for_two_separate_judgements(self):
        from virtual_context.core.engagement import (
            FIDELITY_JUDGE_SYSTEM_PROMPT as PROMPT,
        )

        assert "ASSERTS" in PROMPT and "PRESUPPOSES" in PROMPT
        assert "separate judgements" in PROMPT
        assert '"asserts"' in PROMPT and '"presupposes"' in PROMPT

    def test_an_assertion_fails(self):
        from virtual_context.core.engagement import run_fidelity_gate

        verdict = run_fidelity_gate(
            quote="Sleep has been rough since I moved the modafinil earlier.",
            draft="Moving the modafinil earlier wrecked your sleep.",
            judge=lambda **kw: {
                "asserts": True, "presupposes": False,
                "reason": "states a cause the quote does not give",
            },
        )
        assert verdict.faithful is False
        assert "cause" in verdict.reason

    def test_a_presupposition_passes(self):
        """The owner's ruling, encoded as a mechanism rather than a case."""
        from virtual_context.core.engagement import run_fidelity_gate

        verdict = run_fidelity_gate(
            quote="Adding ss31 (5mg) for 4 weeks.",
            draft="How's the four weeks going?",
            judge=lambda **kw: {"asserts": False, "presupposes": True},
        )
        assert verdict.faithful is True

    def test_an_unreadable_split_verdict_still_fails_closed(self):
        from virtual_context.core.engagement import run_fidelity_gate

        verdict = run_fidelity_gate(
            quote="q", draft="d",
            judge=lambda **kw: {"asserts": "maybe", "presupposes": False},
        )
        assert verdict.faithful is False
        assert verdict.reason == "unreadable_verdict"

    def test_the_reclassified_followups_are_pass_fixtures(self):
        from virtual_context.core.engagement import (
            ADVERSARIAL_FIDELITY_FIXTURES as FIXTURES,
        )

        by_name = {f.name: f for f in FIXTURES}
        for name in (
            "presumptive_followup_four_weeks",
            "presumptive_followup_motsc",
            "presumptive_followup_contemplated_change",
        ):
            assert by_name[name].expected_faithful is True, name

    def test_the_assertion_fixtures_must_still_fail(self):
        from virtual_context.core.engagement import (
            ADVERSARIAL_FIDELITY_FIXTURES as FIXTURES,
        )

        by_name = {f.name: f for f in FIXTURES}
        for name in (
            "asserts_a_cause_he_never_gave",
            "asserts_an_action_he_never_took",
            "asserts_a_causal_claim",
            "asserts_he_said_something_he_did_not",
        ):
            assert by_name[name].expected_faithful is False, name

    def test_a_clause_is_pinned_in_its_draft_not_as_a_fragment(self):
        """Judging the second clause alone produced a wrong verdict twice.

        The gate sees whole drafts, so the fixture is the whole draft. A
        fixture holding only the trailing clause would measure a unit the
        runtime never evaluates.
        """
        from virtual_context.core.engagement import (
            ADVERSARIAL_FIDELITY_FIXTURES as FIXTURES,
        )

        fixture = next(
            f for f in FIXTURES if f.name == "clause_in_context_not_fragment"
        )
        assert fixture.draft.count("?") == 2, "the fixture lost its antecedent"
        assert fixture.draft.startswith("Did you end up starting")
        assert "How's the four weeks going?" in fixture.draft
        assert fixture.expected_faithful is True

    def test_both_production_drafts_are_pass_regressions(self):
        from virtual_context.core.engagement import (
            ADVERSARIAL_FIDELITY_FIXTURES as FIXTURES,
        )

        names = {f.name for f in FIXTURES if f.expected_faithful}
        assert "clause_in_context_not_fragment" in names
        assert "production_draft_dadscientist" in names

    def test_the_suite_cannot_be_passed_by_rejecting_everything(self):
        from virtual_context.core.engagement import (
            ADVERSARIAL_FIDELITY_FIXTURES as FIXTURES,
        )

        assert sum(1 for f in FIXTURES if f.expected_faithful) >= 5
        assert sum(1 for f in FIXTURES if not f.expected_faithful) >= 5
