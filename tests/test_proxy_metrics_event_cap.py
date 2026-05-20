"""Tests for ``ProxyMetrics._events`` count cap.

The cap is enforced on every ``record()`` call via the hard-cap branch
that runs after the optional 24h time-based prune.  These tests pin
the post-append bound, chronological-preservation across truncation,
and the seq-cursor read path for SSE dashboard consumers.
"""

from __future__ import annotations

from virtual_context.proxy.metrics import ProxyMetrics


def test_max_events_constant_value() -> None:
    """Defensive assertion: surface accidental cap regressions."""
    assert ProxyMetrics.MAX_EVENTS == 5_000


def test_record_above_cap_truncates_to_max_events() -> None:
    metrics = ProxyMetrics()
    for i in range(6_000):
        metrics.record({"type": "request", "turn": i})
    assert len(metrics._events) == ProxyMetrics.MAX_EVENTS


def test_record_at_cap_holds_steady() -> None:
    metrics = ProxyMetrics()
    for i in range(ProxyMetrics.MAX_EVENTS):
        metrics.record({"type": "request", "turn": i})
    assert len(metrics._events) == ProxyMetrics.MAX_EVENTS
    # One more append must not push beyond the cap.
    metrics.record({"type": "request", "turn": ProxyMetrics.MAX_EVENTS})
    assert len(metrics._events) == ProxyMetrics.MAX_EVENTS


def test_events_since_returns_recent_events_after_truncation() -> None:
    """Truncation drops the OLDEST events.  The list tail should always
    carry the most recent events; chronological order within the
    surviving window is preserved."""
    metrics = ProxyMetrics()
    for i in range(ProxyMetrics.MAX_EVENTS + 500):
        metrics.record({"type": "request", "turn": i})
    # events_since(seq=0) returns every event currently held in memory.
    # The newest event is the one we just appended (turn == 5499).
    events = metrics.events_since(0)
    assert len(events) == ProxyMetrics.MAX_EVENTS
    assert events[-1]["turn"] == ProxyMetrics.MAX_EVENTS + 500 - 1
    # Surviving turn-numbers form a contiguous tail of the appended series.
    turns = [e["turn"] for e in events]
    assert turns == list(range(500, ProxyMetrics.MAX_EVENTS + 500))


def test_events_since_seq_cursor_returns_only_newer_events() -> None:
    """SSE dashboard pattern: client tracks ``_seq`` and re-queries for
    new events.  Cap reduction must not break the cursor semantic."""
    metrics = ProxyMetrics()
    for i in range(100):
        metrics.record({"type": "request", "turn": i})
    snapshot_seq = metrics._seq
    for i in range(100, 200):
        metrics.record({"type": "request", "turn": i})
    new_events = metrics.events_since(snapshot_seq - 1)
    assert len(new_events) == 100
    assert new_events[0]["turn"] == 100
    assert new_events[-1]["turn"] == 199
