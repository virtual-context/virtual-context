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
