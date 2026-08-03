"""The bounded window after the daily question.

A question posted and then ignored reads worse than no question: it looks
like something fired and left. The window makes Vast a participant — but its
whole design is restraint, so every bound is a shipped constant rather than
a caller's argument.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from virtual_context.core.engagement import (
    MAX_REPLIES_PER_QUESTION,
    REPLY_PRIORITIES,
    RESPONSE_WINDOW_ENABLED_BY_DEFAULT,
    WINDOW_HOURS,
    ResponseWindowNotConfigured,
    WindowState,
    compose_reply,
    should_reply,
    window_closes_at,
)

EASTERN = ZoneInfo("America/New_York")
POSTED = datetime(2026, 8, 3, 13, 0, tzinfo=EASTERN)
ACTIVE_END = datetime(2026, 8, 3, 23, 0, tzinfo=EASTERN)


def _ok(**kw):
    return {"asserts": False, "presupposes": False}


def _state(**kw):
    base = dict(posted_at=POSTED, replies_sent=0, peers_talking=False)
    base.update(kw)
    return WindowState(**base)


def _reply(now=None, priority="strong_disagreement", **kw):
    return should_reply(
        state=kw.pop("state", _state()), now=now or POSTED + timedelta(hours=1),
        priority=priority, enabled=kw.pop("enabled", True),
        active_hours_end=kw.pop("active_hours_end", ACTIVE_END),
    )


class TestShipsDisabled:
    def test_the_shipped_default_is_off(self):
        assert RESPONSE_WINDOW_ENABLED_BY_DEFAULT is False

    def test_a_disabled_window_replies_to_nothing(self):
        assert _reply(enabled=False).reply is False


class TestBoundsAreShippedConstants:
    def test_the_window_is_the_conservative_end_of_the_range(self):
        assert WINDOW_HOURS == 6
        assert window_closes_at(POSTED) == POSTED + timedelta(hours=6)

    def test_the_reply_cap_is_the_conservative_end_of_the_range(self):
        assert MAX_REPLIES_PER_QUESTION == 2

    def test_the_budget_is_spent_not_merely_advisory(self):
        state = _state(replies_sent=MAX_REPLIES_PER_QUESTION)
        decision = _reply(state=state)
        assert decision.reply is False
        assert decision.reason == "reply_budget_spent"

    def test_a_reply_after_the_window_is_refused(self):
        decision = _reply(now=POSTED + timedelta(hours=WINDOW_HOURS, minutes=1))
        assert decision.reply is False
        assert decision.reason == "window_closed"


class TestVastDoesNotAnswerEveryone:
    def test_an_ordinary_response_earns_no_reply(self):
        decision = _reply(priority="")
        assert decision.reply is False
        assert decision.reason == "not_worth_replying_to"

    @pytest.mark.parametrize("priority", REPLY_PRIORITIES)
    def test_each_spec_priority_earns_one(self, priority):
        assert _reply(priority=priority).reply is True

    def test_the_five_spec_priorities_are_shipped(self):
        assert set(REPLY_PRIORITIES) == {
            "unexpectedly_useful_answer", "strong_disagreement",
            "concrete_personal_result", "claim_worth_clarifying",
            "invites_a_good_joke",
        }

    def test_it_stops_once_members_talk_among_themselves(self):
        decision = _reply(state=_state(peers_talking=True))
        assert decision.reply is False
        assert decision.reason == "peers_talking_among_themselves"


class TestUndefinedBoundsAreRequiredNotGuessed:
    def test_an_unset_active_hours_end_refuses_rather_than_defaulting(self):
        """The spec never says what 'reasonable active hours' are."""
        with pytest.raises(ResponseWindowNotConfigured):
            should_reply(
                state=_state(), now=POSTED + timedelta(hours=1),
                priority="strong_disagreement", enabled=True,
                active_hours_end=None,
            )

    def test_a_reply_past_active_hours_is_refused(self):
        late = datetime(2026, 8, 3, 23, 30, tzinfo=EASTERN)
        decision = should_reply(
            state=_state(posted_at=datetime(2026, 8, 3, 19, 0, tzinfo=EASTERN)),
            now=late, priority="strong_disagreement", enabled=True,
            active_hours_end=ACTIVE_END,
        )
        assert decision.reply is False
        assert decision.reason == "outside_active_hours"


class TestRepliesGoThroughBothGuards:
    def test_a_reply_that_asserts_is_refused(self):
        text, reason = compose_reply(
            member_words="I might drop to 7.5 if the nausea keeps up.",
            priority="concrete_personal_result",
            composer=lambda **kw: "Since you dropped to 7.5, how is it?",
            judge=lambda **kw: {"asserts": True, "reason": "claims he dropped"},
            claim_checker=lambda **kw: {"asserts_generality": False},
        )
        assert text == ""
        assert reason.startswith("attribution:")

    def test_a_reply_asserting_a_generality_is_refused(self):
        text, reason = compose_reply(
            member_words="Nausea has been rough.",
            priority="claim_worth_clarifying",
            composer=lambda **kw: "Tirzepatide usually does that - how bad?",
            judge=_ok,
            claim_checker=lambda **kw: {"asserts_generality": True},
        )
        assert text == ""
        assert reason == "own_voice:asserts_a_generality"

    def test_a_clean_short_reply_passes_both(self):
        text, reason = compose_reply(
            member_words="Nausea has been rough.",
            priority="claim_worth_clarifying",
            composer=lambda **kw: "Did it settle after the first few weeks?",
            judge=_ok,
            claim_checker=lambda **kw: {"asserts_generality": False},
        )
        assert reason == ""
        assert text == "Did it settle after the first few weeks?"

    def test_an_unchecked_reply_never_passes(self):
        """No claim checker is not permission, as everywhere else."""
        text, reason = compose_reply(
            member_words="x", priority="strong_disagreement",
            composer=lambda **kw: "Short question?", judge=_ok,
            claim_checker=None,
        )
        assert text == ""
        assert reason == "own_voice:claim_check_not_configured"

    def test_a_lecture_is_refused_by_the_form_rules(self):
        text, reason = compose_reply(
            member_words="x", priority="strong_disagreement",
            composer=lambda **kw: "One. Two. Three. Four. So what do you think?",
            judge=_ok, claim_checker=lambda **kw: {"asserts_generality": False},
        )
        assert text == ""
        assert reason == "own_voice:too_many_sentences"


class TestNothingIsInstalled:
    def test_the_module_starts_no_loop_timer_or_listener(self):
        """Checks the code, not the prose.

        A substring sweep matched the word "scheduler" inside a comment,
        which is the same class of error as a test carrying its own ruler:
        it measured the file's text rather than what the file does.
        """
        import ast
        import inspect

        from virtual_context.core.engagement import response_window

        tree = ast.parse(inspect.getsource(response_window))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        for forbidden in ("threading", "asyncio", "sched", "signal", "socket"):
            assert forbidden not in imported, forbidden
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)]
        called = {
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "sleep" not in called


class TestTheTwoHalvesAreJoined:
    """A caller that forgets either half is a caller that ships a defect."""

    def test_a_refused_window_never_reaches_the_composer(self):
        from virtual_context.core.engagement import handle_response

        called = {"n": 0}

        def _composer(**kw):
            called["n"] += 1
            return "should never run"

        text, reason = handle_response(
            state=_state(replies_sent=MAX_REPLIES_PER_QUESTION),
            now=POSTED + timedelta(hours=1), priority="strong_disagreement",
            member_words="x", composer=_composer, judge=_ok,
            claim_checker=lambda **kw: {"asserts_generality": False},
            enabled=True, active_hours_end=ACTIVE_END,
        )
        assert text == ""
        assert reason == "reply_budget_spent"
        assert called["n"] == 0, "drafted a reply the window had refused"

    def test_an_allowed_response_is_drafted_and_guarded(self):
        from virtual_context.core.engagement import handle_response

        text, reason = handle_response(
            state=_state(), now=POSTED + timedelta(hours=1),
            priority="concrete_personal_result",
            member_words="Dropped to 7.5 and the nausea eased.",
            composer=lambda **kw: "How long did that take to settle?",
            judge=_ok,
            claim_checker=lambda **kw: {"asserts_generality": False},
            enabled=True, active_hours_end=ACTIVE_END,
        )
        assert reason == ""
        assert text == "How long did that take to settle?"

    def test_the_guards_still_apply_through_the_joined_path(self):
        from virtual_context.core.engagement import handle_response

        text, reason = handle_response(
            state=_state(), now=POSTED + timedelta(hours=1),
            priority="concrete_personal_result", member_words="x",
            composer=lambda **kw: "Since you stopped, how is it?",
            judge=lambda **kw: {"asserts": True, "reason": "claims he stopped"},
            claim_checker=lambda **kw: {"asserts_generality": False},
            enabled=True, active_hours_end=ACTIVE_END,
        )
        assert text == ""
        assert reason.startswith("attribution:")


class TestTheDailyCapIsItsOwnBound:
    """Equal to the per-question cap today, but not derived from it."""

    def test_the_daily_cap_is_a_shipped_constant(self):
        from virtual_context.core.engagement import MAX_REPLIES_PER_DAY

        assert MAX_REPLIES_PER_DAY == 2

    def test_the_daily_budget_refuses_even_with_a_fresh_question(self):
        """A second post in one day must not reset the daily total."""
        from virtual_context.core.engagement import MAX_REPLIES_PER_DAY

        decision = _reply(
            state=_state(replies_sent=0,
                         replies_sent_today=MAX_REPLIES_PER_DAY),
        )
        assert decision.reply is False
        assert decision.reason == "daily_reply_budget_spent"

    def test_the_two_bounds_are_independent(self):
        """Neither is computed from the other, so changing one is visible."""
        import inspect

        from virtual_context.core.engagement import response_window

        source = inspect.getsource(response_window)
        assert "MAX_REPLIES_PER_DAY = 2" in source
        assert "MAX_REPLIES_PER_DAY = MAX_REPLIES_PER_QUESTION" not in source
