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
