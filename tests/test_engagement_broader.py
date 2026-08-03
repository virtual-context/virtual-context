"""Generating a channel question, and refusing to post filler.

The specification requires this question to be generated and validated, not
selected from a list — the worked examples are shape and tone, explicitly
not a rotation to recycle. So there is no pool, and the absence of one is
the design rather than a gap.
"""

from __future__ import annotations

import pytest

from virtual_context.core.engagement import (
    BROADER_GENERATOR_GUIDANCE,
    CLAIM_CHECKER_SYSTEM_PROMPT,
    BroaderGeneratorNotConfigured,
    generate_broader_question,
    validate_broader_question,
)

GOOD = "Which marker changed your protocol, and which did you overreact to?"


def _ok(**kw):
    return {"asserts_generality": False, "reason": ""}


class TestGeneratorFailsClosed:
    def test_no_generator_configured_raises(self):
        with pytest.raises(BroaderGeneratorNotConfigured):
            generate_broader_question(channel_label="#p3ptides", generator=None)

    def test_a_generator_that_raises_yields_a_named_reason(self):
        def _boom(**kw):
            raise RuntimeError("provider down")

        out = generate_broader_question(
            channel_label="#p3ptides", generator=_boom, claim_checker=_ok,
        )
        assert out.usable is False
        assert out.reason == "generator_error"

    def test_an_empty_generation_is_not_usable(self):
        out = generate_broader_question(
            channel_label="#p3ptides", generator=lambda **kw: "   ",
            claim_checker=_ok,
        )
        assert out.reason == "empty_question"

    def test_validation_without_a_claim_checker_fails_closed(self):
        """Unvalidated is not permission to post, as with the fidelity gate."""
        out = generate_broader_question(
            channel_label="#p3ptides", generator=lambda **kw: GOOD,
        )
        assert out.usable is False
        assert out.reason == "claim_check_not_configured"


class TestSpecFormRules:
    def test_a_clean_question_passes(self):
        out = generate_broader_question(
            channel_label="#p3ptides", generator=lambda **kw: GOOD,
            claim_checker=_ok,
        )
        assert out.usable is True
        assert out.text == GOOD

    @pytest.mark.parametrize("bad,reason", [
        ("Question of the day: what is your stack?", "banned_phrasing"),
        ("What are your thoughts on peptides?", "banned_phrasing"),
        ("Tell me about your stack.", "not_a_question"),
        ("Do I have low testosterone?", "solicits_diagnosis"),
        ("Should I take more tren?", "solicits_diagnosis"),
        ("One. Two. Three. Which is it?", "too_many_sentences"),
        ("x" * 400 + "?", "too_long"),
    ])
    def test_the_spec_rules_are_enforced(self, bad, reason):
        assert validate_broader_question(bad, claim_checker=_ok).reason == reason


class TestScopeGuard:
    def test_a_question_asserting_a_generality_is_rejected(self):
        """'can happen in these cases' must never become 'usually'."""
        out = validate_broader_question(
            "Since tren usually wrecks lipids, how do you manage it?",
            claim_checker=lambda **kw: {
                "asserts_generality": True, "reason": "states 'usually'",
            },
        )
        assert out.usable is False
        assert out.reason == "asserts_a_generality"

    def test_an_unreadable_claim_verdict_fails_closed(self):
        out = validate_broader_question(
            GOOD, claim_checker=lambda **kw: {"asserts_generality": "maybe"},
        )
        assert out.reason == "unreadable_claim_verdict"

    def test_a_claim_checker_that_raises_fails_closed(self):
        def _boom(**kw):
            raise RuntimeError("down")

        assert validate_broader_question(
            GOOD, claim_checker=_boom,
        ).reason == "claim_check_error"

    def test_the_shipped_checker_prompt_asks_only_about_assertion(self):
        assert "asserts_generality" in CLAIM_CHECKER_SYSTEM_PROMPT
        assert "usually" in CLAIM_CHECKER_SYSTEM_PROMPT
        assert "Asking whether" in CLAIM_CHECKER_SYSTEM_PROMPT


class TestNoRecycling:
    def test_a_repeat_of_a_recent_question_is_rejected(self):
        out = generate_broader_question(
            channel_label="#p3ptides", generator=lambda **kw: GOOD,
            claim_checker=_ok, recent_questions=[GOOD],
        )
        assert out.usable is False
        assert out.reason == "repeats_a_recent_question"

    def test_recent_questions_are_shown_to_the_generator(self):
        seen = {}
        generate_broader_question(
            channel_label="#p3ptides",
            generator=lambda **kw: seen.update(kw) or GOOD,
            claim_checker=_ok, recent_questions=["old one"],
        )
        assert seen["avoid"] == ["old one"]
        assert seen["channel_label"] == "#p3ptides"


class TestGuidanceCarriesTheSpec:
    def test_the_shipped_guidance_states_the_spec_preferences(self):
        g = BROADER_GENERATOR_GUIDANCE.lower()
        assert "biomarkers" in g and "lived outcomes" in g
        assert "protocol theory" in g and "actual response" in g
        assert "benefit" in g and "cost" in g
        assert "medical diagnosis" in g
        assert "question of the day" in g
        assert "one or two sentences" in g
