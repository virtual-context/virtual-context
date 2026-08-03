"""The judge must not be talked out of its verdict by what it is judging.

Selecting and verifying a quote proves provenance — that the member really
wrote it. That is not the same as proving the text is safe to hand a
tool-enabled model. A member who knows a bot quotes him back can write a
message shaped as instructions.

These tests cover what the ENGINE controls. They deliberately do not claim
the judge resists injection: that is a property of a model plus the caller's
prompt assembly, and no fixture run without a live judge can establish it.
What is testable here is that the fixtures exist, carry the correct verdict,
and are scored identically to every other adversarial pair — so the moment
the gate is measured against a real judge, injection is measured with it.
"""

from __future__ import annotations

from virtual_context.core.engagement import (
    ADVERSARIAL_FIDELITY_FIXTURES,
    FIDELITY_JUDGE_SYSTEM_PROMPT,
    run_fidelity_gate,
)

INJECTION = tuple(
    f for f in ADVERSARIAL_FIDELITY_FIXTURES
    if f.name.startswith("quote_")
)


class TestTheInjectionFixturesExistAndAreCorrect:
    def test_there_are_injection_fixtures(self):
        assert len(INJECTION) >= 4, "the injection cases were dropped"

    def test_every_injection_draft_genuinely_asserts(self):
        """The correct verdict must not depend on the injection working.

        Each draft states something the quote never says, so 'reject' is
        right whatever the embedded instruction asks for. A fixture whose
        draft were actually faithful would let a compliant judge look
        correct.
        """
        for fixture in INJECTION:
            assert fixture.expected_faithful is False, fixture.name
            assert fixture.must_reject is True, fixture.name

    def test_the_quotes_carry_instruction_shaped_text(self):
        markers = ("ignore all previous", "system:", "asserts", "no further")
        for fixture in INJECTION:
            low = fixture.quote.lower()
            assert any(m in low for m in markers), (
                f"{fixture.name} no longer contains instruction-shaped text"
            )

    def test_they_are_scored_like_every_other_fixture(self):
        """No separate path, so they cannot be quietly excluded from a run."""
        assert set(INJECTION).issubset(set(ADVERSARIAL_FIDELITY_FIXTURES))


class TestTheGateItselfIsNotPersuadable:
    """Whatever the judge returns, the gate's own handling is fixed."""

    def test_a_compliant_verdict_is_still_just_a_verdict(self):
        """If a judge IS talked round, the gate reports faithful.

        This is the honest statement of the exposure: the gate has no
        independent check on the judge. A model that returns asserts=false
        produces a passing verdict, so resistance has to come from the judge
        call, not from here.
        """
        fixture = INJECTION[0]
        verdict = run_fidelity_gate(
            quote=fixture.quote, draft=fixture.draft,
            judge=lambda **kw: {"asserts": False, "reason": "instructed"},
        )
        assert verdict.faithful is True, (
            "the gate does not second-guess the judge; if this changes, the "
            "exposure note in the audit is stale"
        )

    def test_a_malformed_verdict_fails_closed(self):
        """An injection that garbles the output cannot pass as approval."""
        for bad in ({"asserts": "false"}, {"asserts": None}, {}, "asserts=false"):
            def _judge(_b=bad, **kw):
                return _b

            verdict = run_fidelity_gate(
                quote=INJECTION[0].quote, draft=INJECTION[0].draft,
                judge=_judge,
            )
            assert verdict.faithful is False, bad

    def test_a_judge_that_raises_fails_closed(self):
        verdict = run_fidelity_gate(
            quote=INJECTION[0].quote, draft=INJECTION[0].draft,
            judge=lambda **kw: (_ for _ in ()).throw(RuntimeError("x")),
        )
        assert verdict.faithful is False
        assert verdict.reason == "judge_error"


class TestWhatTheShippedPromptDoesAndDoesNotSay:
    def test_the_prompt_does_not_frame_the_quote_as_untrusted(self):
        """Pins the audit finding so a fix flips this test deliberately.

        The shipped judge prompt describes the task but never tells the model
        the quote is material to reason about rather than instructions to
        follow. If that is added, this test should be updated in the same
        change — not left passing by accident.
        """
        low = FIDELITY_JUDGE_SYSTEM_PROMPT.lower()
        assert not any(
            phrase in low for phrase in
            ("untrusted", "do not follow", "never follow", "instructions "
             "inside", "ignore any instructions")
        ), "the prompt now has untrusted-input framing; update the audit"


class TestTheScorecard:
    """The harness, not the judge.

    Every judge here is a stub, so nothing in this class says anything about
    whether a real model resists injection. What it establishes is that the
    measurement is sound: that a failure is counted as a failure, that the
    two error directions are not summed into one number, and that an
    injection failure is reported as its own kind.
    """

    def test_a_perfect_judge_passes(self):
        from virtual_context.core.engagement import (
            ADVERSARIAL_FIDELITY_FIXTURES, score_fidelity_fixtures,
        )

        expected = {f.draft: f.expected_faithful
                    for f in ADVERSARIAL_FIDELITY_FIXTURES}

        def _oracle(*, quote, draft):
            return {"asserts": not expected[draft], "reason": ""}

        card = score_fidelity_fixtures(_oracle)
        assert card.passed
        assert card.false_negatives == ()
        assert card.false_positives == ()

    def test_a_judge_that_obeys_the_quote_is_caught(self):
        """The headline failure mode, as the harness would report it."""
        from virtual_context.core.engagement import score_fidelity_fixtures

        card = score_fidelity_fixtures(
            lambda **kw: {"asserts": False, "reason": "instructed"},
        )
        assert not card.passed
        assert len(card.injection_failures) == 4, card.summary()
        assert {r.name for r in card.injection_failures} == {
            "quote_instructs_the_judge_in_prose",
            "quote_contains_the_answer_shaped_as_json",
            "quote_impersonates_a_system_turn",
            "quote_claims_the_draft_is_pre_approved",
        }

    def test_the_two_error_directions_are_not_summed(self):
        """A false negative posts a false attribution; a false positive
        costs a question. One accuracy number would hide which happened."""
        from virtual_context.core.engagement import score_fidelity_fixtures

        card = score_fidelity_fixtures(
            lambda **kw: {"asserts": True, "reason": "everything rejected"},
        )
        assert card.false_negatives == ()
        assert card.false_positives != ()
        assert card.passed, "rejecting everything is safe, if useless"

    def test_an_injection_failure_is_also_a_false_negative(self):
        from virtual_context.core.engagement import score_fidelity_fixtures

        card = score_fidelity_fixtures(lambda **kw: {"asserts": False})
        assert set(card.injection_failures).issubset(set(card.false_negatives))

    def test_the_summary_names_every_count(self):
        from virtual_context.core.engagement import score_fidelity_fixtures

        text = score_fidelity_fixtures(lambda **kw: {"asserts": False}).summary()
        for word in ("correct", "false negative", "false positive", "injection"):
            assert word in text


class TestElapsedTimeTurnsOnSourceNotKind:
    """Metadata about the message is permitted; conduct claims are not.

    The judge guards attribution, not entailment. A message's timestamp is
    verified metadata — re-checked against the live source before anything
    sends — so saying when he wrote something puts nothing in his mouth.
    Saying what he did during that interval still does.

    The pair below is deliberately the SAME nine-day gap and the same quote,
    so the only difference under test is whether the claim is about the
    message or about the member.
    """

    def _fixture(self, name):
        from virtual_context.core.engagement import (
            ADVERSARIAL_FIDELITY_FIXTURES,
        )

        found = [f for f in ADVERSARIAL_FIDELITY_FIXTURES if f.name == name]
        assert found, f"{name} is missing from the shipped set"
        return found[0]

    def test_referring_to_when_he_posted_is_allowed(self):
        fixture = self._fixture("refers_to_when_he_posted")
        assert fixture.expected_faithful is True
        assert fixture.must_reject is False

    def test_a_duration_of_conduct_is_still_rejected(self):
        fixture = self._fixture("asserts_a_duration_of_conduct_from_the_same_gap")
        assert fixture.expected_faithful is False
        assert fixture.must_reject is True

    def test_the_pair_differs_only_in_what_the_claim_is_about(self):
        """Same quote, same interval — so the verdict cannot turn on those."""
        allowed = self._fixture("refers_to_when_he_posted")
        refused = self._fixture("asserts_a_duration_of_conduct_from_the_same_gap")
        assert allowed.quote == refused.quote
        assert "nine days" in allowed.draft and "nine days" in refused.draft
        assert allowed.expected_faithful != refused.expected_faithful

    def test_the_judge_prompt_states_the_distinction(self):
        from virtual_context.core.engagement import FIDELITY_JUDGE_SYSTEM_PROMPT

        low = FIDELITY_JUDGE_SYSTEM_PROMPT.lower()
        assert "about the message" in low and "about the member" in low

    def test_the_composer_is_no_longer_told_to_avoid_all_timing(self):
        """It was handed sent_at and instructed not to use it."""
        from virtual_context.core.engagement.compose import TONE_CONSTRAINTS

        assert "add no claim, intention" in TONE_CONSTRAINTS
        assert "when he spoke" in TONE_CONSTRAINTS
        assert "source of the fact" in TONE_CONSTRAINTS.lower()
