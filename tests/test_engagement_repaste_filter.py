"""A question asking a member to re-send data is not a question.

Two of the first three questions this job produced were this shape:
grammatical, faithful, on-topic and worthless, because the answer is data
entry. The corpus is largely people handing Vast tasks, so "the unresolved
thing" is often a missing field rather than an open question.

The signal was measured, not chosen. The obvious one — "the rest" — rejects
good questions, so it is deliberately not used.
"""

from __future__ import annotations

import pytest

from virtual_context.core.engagement import (
    REPASTE_REJECTION, asks_for_a_repaste,
)

# The two the owner objected to, verbatim, plus a third of the same shape.
REAL_BAD = [
    "Can you paste the rest of the Maximus Building Blocks label? "
    "It cuts off at vitamin B6.",
    "...your list cut off at retatrutide—can you paste the rest?",
    "Your stack list got truncated - can you repost the full thing?",
]

# The one good question this job has produced, plus innocent questions that
# use the same vocabulary. All of these must pass.
REAL_GOOD = [
    'When you say "equivalent," do you mean matching the average IGF-1 '
    "increase from 1 mg tesamorelin, or matching its broader effects?",
    "How did the rest of the week go on the enclo?",
    "How's the rest of the protocol treating you?",
    "Did you rest between the two blocks or run them back to back?",
    "Did the cut go the way you wanted, or did you hold weight?",
    "Are you resting the shoulder or training through it?",
    "Did you end up starting the SS-31?",
    "What made you pick the morning dose over the evening one?",
]


class TestAgainstTheRealQuestions:
    @pytest.mark.parametrize("question", REAL_BAD)
    def test_a_repaste_request_is_caught(self, question):
        assert asks_for_a_repaste(question) is True

    @pytest.mark.parametrize("question", REAL_GOOD)
    def test_a_real_question_is_not_caught(self, question):
        assert asks_for_a_repaste(question) is False

    def test_the_one_good_question_this_job_produced_survives(self):
        """Named separately: if this fails, the filter is aimed wrong."""
        assert asks_for_a_repaste(REAL_GOOD[0]) is False


class TestTheSignalIsTheVerbNotTheWordRest:
    """`the rest` was the obvious signal and it is wrong.

    Measured: keying on "paste|the rest" catches 2 of 3 bad questions and
    falsely rejects 2 of 6 innocent ones. Keying on the re-transmission verb
    catches 3 of 3 with none. This pins the distinction so nobody
    "simplifies" it back.
    """

    def test_asking_about_the_rest_of_something_is_fine(self):
        for question in ("How did the rest of the week go on the enclo?",
                         "How's the rest of the protocol treating you?"):
            assert asks_for_a_repaste(question) is False, question

    def test_asking_someone_to_paste_is_not(self):
        assert asks_for_a_repaste("Could you paste that again?") is True

    def test_repost_and_resend_are_caught_too(self):
        assert asks_for_a_repaste("Mind reposting the list?") is True
        assert asks_for_a_repaste("Can you resend the full stack?") is True

    def test_explicit_truncation_language_is_caught(self):
        """`truncat` stays; `cuts off` does not. See the class below."""
        assert asks_for_a_repaste("Looks truncated, what was the last one?")

    def test_a_word_that_merely_starts_with_paste_is_not_caught(self):
        """Constructed, not observed — but a false positive costs a question.

        The stem pattern matched "paste-y" because a hyphen ends a word
        boundary. No real question has done this; the guard is defensive and
        labelled as such rather than presented as a measured case.
        """
        assert asks_for_a_repaste(
            "How's the paste-y texture of the reconstituted stuff?"
        ) is False
        assert asks_for_a_repaste("Did the pasty look go away?") is False

    def test_empty_input_is_not_a_repaste_request(self):
        assert asks_for_a_repaste("") is False
        assert asks_for_a_repaste(None) is False


class TestCutOffIsOrdinaryVocabularyHere:
    """`cuts off` was shipped and had to come out.

    It caught zero true positives the re-transmission verbs did not already
    catch, and caused every false positive found: 3 of 10. In a biohacking
    channel "cut off your appetite" and "cuts off after a few days" are
    everyday phrasings.

    The reason it survived the first round is the sharper lesson: the innocent
    set used to validate the filter contained no non-re-transmission use of
    "cut off", so a measured zero-false-positive result was taken from a
    population that could not contain the failure.
    """

    CUT_OFF_BUT_FINE = [
        "Did moving the modafinil earlier cut off your deep sleep?",
        "Has the tirzepatide cut off your appetite entirely, or just "
        "blunted it?",
        "You said the nausea cuts off after a few days — did that hold?",
    ]

    @pytest.mark.parametrize("question", CUT_OFF_BUT_FINE)
    def test_a_physiological_cut_off_is_not_a_repaste_request(self, question):
        assert asks_for_a_repaste(question) is False

    def test_the_pattern_does_not_mention_cut_off(self):
        """Pins the removal, so it cannot be helpfully re-added."""
        from virtual_context.core.engagement.select import _REPASTE_REQUEST

        assert "cut" not in _REPASTE_REQUEST.pattern

    def test_the_first_real_bad_question_is_still_caught_without_it(self):
        """It contains "cuts off" — but the verb is what catches it."""
        assert asks_for_a_repaste(REAL_BAD[0]) is True


class TestItIsWiredIntoTheRun:
    def test_the_rejection_reason_is_named(self):
        assert REPASTE_REJECTION == "asks_for_a_repaste"

    def test_the_runner_consults_it(self):
        """Grep for a consumer, not a definition."""
        import inspect

        from virtual_context.core.engagement import runner

        source = inspect.getsource(runner)
        assert "asks_for_a_repaste(" in source, "defined but never called"
        assert "REPASTE_REJECTION" in source
