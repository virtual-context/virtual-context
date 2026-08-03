"""Personal continuation: unbounded in time, bounded by evidence.

A timed follow-up is gated by the clock. A continuation may reach back to
anything the member said, and is gated instead by whether his own words
contain something specific to continue. Without that test a name is stapled
to a question anyone could answer, which is the failure the specification
names and illustrates.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.core.engagement import (
    CONTINUATION_HOOK_KINDS,
    HOOK_DETECTOR_SYSTEM_PROMPT,
    Candidate,
    HookDetectorNotConfigured,
    SelectionOutcome,
    find_continuation_hook,
    select_question,
)

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
P3 = "1524917968440524990"
QUOTE = "Adding ss31 (5mg) for 4 weeks. Labs pending, should have them soon."


def _cand(**kw):
    base = dict(
        canonical_turn_id="ct-1", source_message_id="1532400954878595094",
        actor_id="actor:discord:1", channel_id=P3, text=QUOTE,
        sent_at=NOW - timedelta(days=30), sender="BigTex",
    )
    base.update(kw)
    return Candidate(**base)


class TestConnectionTestFailsClosed:
    def test_no_detector_configured_raises(self):
        with pytest.raises(HookDetectorNotConfigured):
            find_continuation_hook(quote=QUOTE, detector=None)

    def test_a_detector_that_raises_finds_no_hook(self):
        def _boom(**kw):
            raise RuntimeError("down")

        hook, reason = find_continuation_hook(quote=QUOTE, detector=_boom)
        assert hook is None and reason == "detector_error"

    def test_an_unreadable_verdict_finds_no_hook(self):
        hook, reason = find_continuation_hook(
            quote=QUOTE, detector=lambda **kw: "nope",
        )
        assert hook is None and reason == "unreadable_hook_verdict"


class TestTheTagMustBeEarned:
    def test_a_generic_message_yields_no_hook(self):
        """The spec's failure: a question anyone could answer, with a name."""
        hook, reason = find_continuation_hook(
            quote="morning all",
            detector=lambda **kw: {"kind": "", "evidence": "", "reason": "generic"},
        )
        assert hook is None
        assert reason == "no_specific_hook"

    def test_a_specific_item_yields_a_named_hook(self):
        hook, _ = find_continuation_hook(
            quote=QUOTE,
            detector=lambda **kw: {
                "kind": "pending_or_surprising_labs",
                "evidence": "Labs pending, should have them soon.",
            },
        )
        assert hook.usable is True
        assert hook.kind == "pending_or_surprising_labs"

    def test_an_unknown_kind_is_refused(self):
        hook, reason = find_continuation_hook(
            quote=QUOTE,
            detector=lambda **kw: {"kind": "vibes", "evidence": QUOTE},
        )
        assert hook is None and reason == "unknown_hook_kind"

    def test_a_paraphrased_evidence_is_refused(self):
        """A rewritten quote is how a claim drifts from what he said."""
        hook, reason = find_continuation_hook(
            quote=QUOTE,
            detector=lambda **kw: {
                "kind": "pending_or_surprising_labs",
                "evidence": "he is waiting on bloodwork",
            },
        )
        assert hook is None and reason == "evidence_not_verbatim"

    def test_a_hook_without_evidence_is_refused(self):
        hook, reason = find_continuation_hook(
            quote=QUOTE,
            detector=lambda **kw: {
                "kind": "pending_or_surprising_labs", "evidence": "",
            },
        )
        assert hook is None and reason == "hook_without_evidence"

    def test_the_nine_spec_kinds_are_shipped(self):
        assert len(CONTINUATION_HOOK_KINDS) == 9
        for expected in (
            "unresolved_experiment_or_protocol",
            "symptom_or_side_effect_tracked",
            "pending_or_surprising_labs",
            "dose_or_compound_change",
            "stated_decision_rule",
            "personal_preference_or_tradeoff",
            "contradiction_or_change_of_view",
            "specific_practical_concern",
            "prior_result_unclear",
        ):
            assert expected in CONTINUATION_HOOK_KINDS

    def test_the_detector_prompt_demands_verbatim_evidence(self):
        assert "verbatim" in HOOK_DETECTOR_SYSTEM_PROMPT
        assert "could be asked of anyone" in HOOK_DETECTOR_SYSTEM_PROMPT


class TestUnboundedInTime:
    def test_material_far_older_than_the_timed_window_still_qualifies(self):
        """The distinguishing property: no clock gate."""
        old = _cand(sent_at=NOW - timedelta(days=180))
        hook, _ = find_continuation_hook(
            quote=old.text,
            detector=lambda **kw: {
                "kind": "dose_or_compound_change",
                "evidence": "Adding ss31 (5mg) for 4 weeks.",
            },
        )
        assert hook.usable is True


class TestLabelsAreTruthful:
    def test_the_selector_takes_the_type_from_the_candidate(self):
        outcome = select_question(
            verified=[_cand(question_type="personal")], rejections=[],
            channel_id=P3,
        )
        assert outcome.kind == "personal"

    def test_a_timed_candidate_is_labelled_timed(self):
        outcome = select_question(
            verified=[_cand(question_type="timed")], rejections=[],
            channel_id=P3,
        )
        assert outcome.kind == "timed"

    def test_an_unlabelled_candidate_does_not_claim_to_be_personal(self):
        """The old hardcode called every timed follow-up a continuation."""
        outcome = select_question(
            verified=[_cand()], rejections=[], channel_id=P3,
        )
        assert outcome.kind != "personal"
        assert outcome.kind == "timed"


class TestOnePoolNotTwoTiers:
    def _detector(self, kind="dose_or_compound_change"):
        return lambda **kw: {
            "kind": kind, "evidence": "Adding ss31 (5mg) for 4 weeks.",
        }

    def test_both_types_land_in_one_pool(self):
        from virtual_context.core.engagement import qualify_candidates

        recent = _cand(canonical_turn_id="ct-recent",
                       sent_at=NOW - timedelta(days=3))
        old = _cand(canonical_turn_id="ct-old",
                    sent_at=NOW - timedelta(days=90))
        qualified, _ = qualify_candidates(
            [recent, old], now=NOW, detector=self._detector(),
        )
        types = {c.canonical_turn_id: c.question_type for c in qualified}
        assert types == {"ct-recent": "timed", "ct-old": "personal"}

    def test_a_continuation_can_outrank_a_timed_followup(self):
        """Not a fallback tier: they compete on the same ranking."""
        from virtual_context.core.engagement import (
            qualify_candidates, rank_candidates,
        )

        thin_timed = _cand(canonical_turn_id="ct-timed", text="ok",
                           sent_at=NOW - timedelta(days=3))
        rich_old = _cand(canonical_turn_id="ct-old",
                         text="Adding ss31 (5mg) for 4 weeks. Labs pending.",
                         sent_at=NOW - timedelta(days=90))
        qualified, _ = qualify_candidates(
            [thin_timed, rich_old], now=NOW, detector=self._detector(),
        )
        assert rank_candidates(qualified)[0].canonical_turn_id == "ct-old"

    def test_a_continuation_without_a_hook_is_rejected_by_name(self):
        from virtual_context.core.engagement import qualify_candidates

        old = _cand(sent_at=NOW - timedelta(days=90))
        qualified, rejections = qualify_candidates(
            [old], now=NOW,
            detector=lambda **kw: {"kind": "", "evidence": ""},
        )
        assert qualified == []
        assert rejections[0].stage == "continuation"
        assert rejections[0].reason == "no_specific_hook"

    def test_the_resolution_check_applies_to_continuations_too(self):
        """A thread he is talking about today is stale at any age."""
        from virtual_context.core.engagement import qualify_candidates

        old = _cand(sent_at=NOW - timedelta(days=90))
        qualified, rejections = qualify_candidates(
            [old], now=NOW, detector=self._detector(),
            later_texts_for=lambda c: [("ss31 update", NOW - timedelta(hours=2))],
            subject_terms_for=lambda c: ["ss31"],
        )
        assert qualified == []
        assert rejections[0].stage == "resolution"
        assert rejections[0].reason == "member_posted_today"

    def test_the_resolution_check_applies_to_timed_too(self):
        from virtual_context.core.engagement import qualify_candidates

        recent = _cand(sent_at=NOW - timedelta(days=3))
        qualified, rejections = qualify_candidates(
            [recent], now=NOW, detector=self._detector(),
            later_texts_for=lambda c: [("ss31 update", NOW - timedelta(hours=1))],
            subject_terms_for=lambda c: ["ss31"],
        )
        assert qualified == []
        assert rejections[0].stage == "resolution"


class TestContinuationComposition:
    def _cand_with_hook(self, evidence="Adding ss31 (5mg) for 4 weeks."):
        return _cand(question_type="personal",
                     hook_kind="dose_or_compound_change",
                     hook_evidence=evidence)

    def test_no_composer_configured_raises(self):
        from virtual_context.core.engagement import (
            DraftComposerNotConfigured, compose_continuation_draft,
        )

        with pytest.raises(DraftComposerNotConfigured):
            compose_continuation_draft(
                candidate=self._cand_with_hook(), composer=None,
            )

    def test_evidence_must_appear_in_the_quote(self):
        """A hook whose evidence is not his words cannot ground a draft."""
        from virtual_context.core.engagement import compose_continuation_draft

        draft = compose_continuation_draft(
            candidate=self._cand_with_hook(evidence="words he never wrote"),
            composer=lambda **kw: "anything",
        )
        assert draft.usable is False
        assert draft.reason == "evidence_not_in_quote"

    def test_the_composer_receives_the_hook_and_verbatim_evidence(self):
        from virtual_context.core.engagement import compose_continuation_draft

        seen = {}
        compose_continuation_draft(
            candidate=self._cand_with_hook(),
            composer=lambda **kw: seen.update(kw) or "q?",
            sender="BigTex",
        )
        assert seen["evidence"] == "Adding ss31 (5mg) for 4 weeks."
        assert seen["hook_kind"] == "dose_or_compound_change"
        assert "dose or compound" in seen["hook_framing"]
        assert seen["stance"] == "continuation"
        assert "BigTex:" not in seen["quote"]

    def test_the_timed_prompt_is_not_reused(self):
        """Continuation has its own guidance; the timed path is untouched."""
        from virtual_context.core.engagement import CONTINUATION_GUIDANCE

        assert "not a recent-status check" in CONTINUATION_GUIDANCE
        assert "could be asked of any member" in CONTINUATION_GUIDANCE


class TestTheArtifactNamesTheType:
    """A label that stays inside the candidate never reaches the reviewer."""

    def test_a_continuation_is_named_as_one_with_its_hook(self):
        from virtual_context.core.engagement import (
            DryRunReport, SelectionOutcome,
        )

        candidate = _cand(question_type="personal",
                          hook_kind="pending_or_surprising_labs")
        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=P3,
        ).apply_outcome(
            SelectionOutcome(kind="personal", candidate=candidate),
        )
        rendered = report.render()
        assert "question type: personal" in rendered
        assert "hook: pending_or_surprising_labs" in rendered

    def test_a_timed_followup_is_named_as_timed_and_shows_no_hook(self):
        from virtual_context.core.engagement import (
            DryRunReport, SelectionOutcome,
        )

        candidate = _cand(question_type="timed")
        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=P3,
        ).apply_outcome(SelectionOutcome(kind="timed", candidate=candidate))
        rendered = report.render()
        assert "question type: timed" in rendered
        assert "hook:" not in rendered

    def test_a_broader_question_is_named_as_broader(self):
        from virtual_context.core.engagement import (
            DryRunReport, SelectionOutcome,
        )

        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=P3,
        ).apply_outcome(SelectionOutcome(kind="broader", question="q?"))
        assert "question type: broader" in report.render()


class TestTheHookCannotBeRecomputed:
    """The draft must rest on the hook that passed the gate, not a new one.

    Qualification computes the hook and its evidence, and that computation is
    what admitted the candidate. If the drafter recomputes, a detector that
    answers differently the second time produces a draft grounded in evidence
    no gate ever saw — and the divergence is invisible afterwards, because
    both hooks look equally plausible in a report.
    """

    def _drifting_detector(self, calls):
        """Answers one way first, then differently. A real model can do this."""
        answers = [
            {"kind": "dose_or_compound_change",
             "evidence": "Adding ss31 (5mg) for 4 weeks."},
            {"kind": "prior_result_unclear", "evidence": "something else"},
        ]

        def _detector(**kw):
            calls.append(kw)
            return answers[min(len(calls) - 1, len(answers) - 1)]

        return _detector

    def test_the_drafter_never_calls_the_detector(self):
        """The strongest form: there is no second computation to disagree."""
        from virtual_context.core.engagement import (
            compose_continuation_draft, qualify_candidates,
        )

        calls: list = []
        candidate = _cand(sent_at=NOW - timedelta(days=90))
        qualified, _ = qualify_candidates(
            [candidate], now=NOW, detector=self._drifting_detector(calls),
        )
        assert len(calls) == 1, "qualification should compute the hook once"

        compose_continuation_draft(
            candidate=qualified[0], composer=lambda **kw: "q?", sender="BigTex",
        )
        assert len(calls) == 1, "the drafter recomputed the hook"

    def test_the_draft_uses_the_hook_that_qualified_it(self):
        from virtual_context.core.engagement import (
            compose_continuation_draft, qualify_candidates,
        )

        calls: list = []
        qualified, _ = qualify_candidates(
            [_cand(sent_at=NOW - timedelta(days=90))], now=NOW,
            detector=self._drifting_detector(calls),
        )
        seen = {}
        compose_continuation_draft(
            candidate=qualified[0],
            composer=lambda **kw: seen.update(kw) or "q?", sender="BigTex",
        )
        # The second detector answer must not appear anywhere in the draft.
        assert seen["hook_kind"] == "dose_or_compound_change"
        assert seen["evidence"] == "Adding ss31 (5mg) for 4 weeks."

    def test_the_carried_hook_survives_the_whole_pipeline(self):
        from virtual_context.core.engagement import qualify_candidates

        qualified, _ = qualify_candidates(
            [_cand(sent_at=NOW - timedelta(days=90))], now=NOW,
            detector=self._drifting_detector([]),
        )
        assert qualified[0].hook_evidence == "Adding ss31 (5mg) for 4 weeks."
        assert qualified[0].hook_kind == "dose_or_compound_change"

    def test_a_candidate_with_no_hook_cannot_be_drafted(self):
        """Reading from the candidate must not silently draft on a blank."""
        from virtual_context.core.engagement import compose_continuation_draft

        draft = compose_continuation_draft(
            candidate=_cand(), composer=lambda **kw: "q?",
        )
        assert draft.usable is False
        assert draft.reason == "no_qualified_hook"

    def test_no_argument_can_inject_a_hook(self):
        """Same shape as the posting flag: it isn't a parameter at all."""
        import inspect

        from virtual_context.core.engagement import compose_continuation_draft

        params = set(inspect.signature(compose_continuation_draft).parameters)
        assert not params & {"hook_kind", "evidence", "hook_evidence"}

    def test_the_stance_is_carried_not_recomputed(self):
        import inspect

        from virtual_context.core.engagement import compose_draft, qualify_candidates

        assert "stance" not in inspect.signature(compose_draft).parameters
        qualified, _ = qualify_candidates(
            [_cand(sent_at=NOW - timedelta(days=3))], now=NOW,
            detector=lambda **kw: {"kind": "x", "evidence": "y"},
        )
        assert qualified[0].stance in {"anticipatory", "outcome"}
