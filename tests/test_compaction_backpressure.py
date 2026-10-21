"""A request waits on a hard compaction only when that compaction can relieve the overflow."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.compaction_backpressure import (
    CompactionWaitDecision,
    decide_compaction_wait,
    hard_tokens_for,
)
from virtual_context.proxy.state import ProxyState
from virtual_context.types import EngineState, Message


def _signal(priority: str, budget: int = 200_000, current: int = 348_842):
    return SimpleNamespace(priority=priority, budget_tokens=budget, current_tokens=current,
                           overflow_tokens=max(0, current - int(budget * 0.35)))


@pytest.mark.regression("BUG-140")
@pytest.mark.parametrize(
    "priority, floor, expected",
    [
        ("hard", 0, True),
        ("hard", 199_999, True),
        ("hard", 200_000, False),
        ("hard", 278_000, False),
        ("soft", 0, False),
        ("soft", 278_000, False),
        ("", 0, False),
    ],
)
def test_wait_only_for_a_hard_compaction_that_can_relieve(priority, floor, expected):
    decision = decide_compaction_wait(_signal(priority), floor_tokens=floor, hard_threshold=1.0)
    assert decision.should_wait() is expected
    assert decision.hard_tokens == 200_000


@pytest.mark.regression("BUG-140")
def test_no_signal_means_no_wait():
    decision = decide_compaction_wait(None, floor_tokens=10**9, hard_threshold=0.85)
    assert decision == CompactionWaitDecision()
    assert decision.should_wait() is False


@pytest.mark.regression("BUG-140")
def test_hard_tokens_follow_the_configured_threshold():
    assert hard_tokens_for(200_000, 0.85) == 170_000
    assert hard_tokens_for(120_000, 1.0) == 120_000
    assert hard_tokens_for("bad", 0.85) == 0
    assert decide_compaction_wait(_signal("hard", budget=0), floor_tokens=0, hard_threshold=0.85).should_wait() is False


def _engine(protected_recent_turns: int, hard_threshold: float, signal):
    engine = MagicMock()
    engine._turn_tag_index = TurnTagIndex()
    engine._engine_state = EngineState()
    engine.config.conversation_id = "conv-backpressure"
    engine.config.monitor.protected_recent_turns = protected_recent_turns
    engine.config.monitor.hard_threshold = hard_threshold
    engine._token_counter = lambda text: len(text) // 4
    engine.tag_turn.return_value = signal
    engine.compact_if_needed.return_value = None
    engine.process_broad_tag_split.return_value = None
    return engine


@pytest.mark.regression("BUG-140")
def test_tag_pass_records_that_an_oversized_protected_window_cannot_be_relieved():
    # The newest turn alone is ~250k tokens: whatever compaction removes
    # before the protected window, the payload stays over the 200k hard line.
    engine = _engine(1, 1.0, _signal("hard"))
    state = ProxyState(engine)
    history = [
        Message(role="user", content="older question"),
        Message(role="assistant", content="older answer"),
        Message(role="user", content="x" * 1_000_000),
        Message(role="assistant", content="Queued 1, then 3."),
    ]
    state.fire_turn_complete(history)
    state.wait_for_complete()

    decision = state._compact_wait_decision
    assert decision.priority == "hard"
    assert decision.hard_tokens == 200_000
    assert decision.floor_tokens == 250_000 + len("Queued 1, then 3.") // 4
    assert decision.should_wait() is False
    assert state._last_compact_priority == "hard"


@pytest.mark.regression("BUG-140")
def test_tag_pass_keeps_the_wait_when_compaction_can_relieve():
    engine = _engine(1, 1.0, _signal("hard"))
    state = ProxyState(engine)
    history = [
        Message(role="user", content="y" * 1_000_000),
        Message(role="assistant", content="long earlier answer"),
        Message(role="user", content="short follow-up"),
        Message(role="assistant", content="short reply"),
    ]
    state.fire_turn_complete(history)
    state.wait_for_complete()

    decision = state._compact_wait_decision
    assert decision.priority == "hard"
    assert decision.floor_tokens < 100
    assert decision.should_wait() is True


@pytest.mark.regression("BUG-140")
def test_soft_signal_never_waits_and_skips_the_floor_count():
    counted = []
    engine = _engine(1, 1.0, _signal("soft", current=90_000))
    engine._token_counter = lambda text: counted.append(len(text)) or len(text) // 4
    state = ProxyState(engine)
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
    state.fire_turn_complete(history)
    state.wait_for_complete()

    assert state._compact_wait_decision.priority == "soft"
    assert state._compact_wait_decision.should_wait() is False
    assert counted == []
