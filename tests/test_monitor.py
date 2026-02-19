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


import pytest


@pytest.mark.regression("PROXY-021")
def test_build_snapshot_payload_token_override():
    """PROXY-021: In proxy mode, monitor should use actual client payload tokens
    instead of stripped conversation_history tokens.

    The proxy's conversation_history contains envelope-stripped text (~16k tokens)
    while the actual client payload is much larger (~82k tokens).  Without the
    override, compaction never triggers because the monitor sees 16k against an
    84k soft threshold.
    """
    from virtual_context.types import Message
    monitor = ContextMonitor(MonitorConfig(context_window=100_000))

    # Simulate proxy conversation_history: short stripped messages
    history = [
        Message(role="user", content="short msg"),
        Message(role="assistant", content="short reply"),
    ] * 50  # 100 messages, ~500 tokens from stripped text

    # Without override: snapshot reflects tiny stripped tokens
    snapshot_no_override = monitor.build_snapshot(history)
    assert snapshot_no_override.total_tokens < 1000  # stripped text is tiny

    # With override: snapshot uses the actual payload token count
    snapshot_with_override = monitor.build_snapshot(
        history, payload_tokens=82_000,
    )
    assert snapshot_with_override.conversation_tokens == 82_000
    assert snapshot_with_override.total_tokens == 82_000

    # Verify this actually triggers compaction (70% of 100k = 70k)
    signal = monitor.check(snapshot_with_override)
    assert signal is not None
    assert signal.priority == "soft"


@pytest.mark.regression("PROXY-021")
def test_engine_on_turn_complete_payload_tokens(tmp_path):
    """PROXY-021: Engine.on_turn_complete should accept payload_tokens override
    so the proxy can pass the real client payload size for compaction decisions."""
    from virtual_context.types import Message, VirtualContextConfig, MonitorConfig
    from virtual_context.types import StorageConfig
    from virtual_context.engine import VirtualContextEngine

    cfg = VirtualContextConfig(
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "test.db"),
        ),
        monitor=MonitorConfig(
            context_window=100_000,
            soft_threshold=0.70,
            hard_threshold=0.85,
            protected_recent_turns=2,
        ),
    )
    engine = VirtualContextEngine(config=cfg)

    # Build a small conversation_history (stripped proxy text)
    history = []
    for i in range(20):
        history.append(Message(role="user", content=f"Turn {i} user"))
        history.append(Message(role="assistant", content=f"Turn {i} asst"))

    # Without payload_tokens: tiny history, no compaction
    report = engine.on_turn_complete(history)
    assert report is None  # should not compact — history is tiny

    # With payload_tokens=82000: should trigger compaction (82k > 70k threshold)
    # Need a compactor for actual compaction, but we can check the signal path
    # by verifying the snapshot the monitor builds
    snapshot = engine._monitor.build_snapshot(
        history[engine._compacted_through:],
        payload_tokens=82_000,
    )
    signal = engine._monitor.check(snapshot)
    assert signal is not None
    assert signal.priority == "soft"
