"""Tests for ContextMonitor."""

from virtual_context.core.monitor import ContextMonitor
from virtual_context.types import ContextSnapshot, MonitorConfig


def make_snapshot(total: int, budget: int = 10000) -> ContextSnapshot:
    return ContextSnapshot(
        system_tokens=0,
        core_context_tokens=0,
        retrieved_domain_tokens=0,
        conversation_tokens=total,
        total_tokens=total,
        budget_tokens=budget,
        turn_count=10,
    )


def test_no_signal_under_soft():
    monitor = ContextMonitor(MonitorConfig(context_window=10000))
    signal = monitor.check(make_snapshot(5000))
    assert signal is None


def test_soft_signal():
    monitor = ContextMonitor(MonitorConfig(context_window=10000))
    signal = monitor.check(make_snapshot(7500))
    assert signal is not None
    assert signal.priority == "soft"


def test_hard_signal():
    monitor = ContextMonitor(MonitorConfig(context_window=10000))
    signal = monitor.check(make_snapshot(9000))
    assert signal is not None
    assert signal.priority == "hard"


def test_force_compact():
    monitor = ContextMonitor(MonitorConfig(context_window=10000))
    monitor.check(make_snapshot(5000))
    signal = monitor.force_compact()
    assert signal.priority == "hard"
    assert signal.overflow_tokens >= 1000


def test_build_snapshot():
    from virtual_context.types import Message
    monitor = ContextMonitor(MonitorConfig(context_window=10000))
    history = [
        Message(role="user", content="Hello world"),
        Message(role="assistant", content="Hi there"),
    ]
    snapshot = monitor.build_snapshot(history)
    assert snapshot.total_tokens > 0
    assert snapshot.budget_tokens == 10000
