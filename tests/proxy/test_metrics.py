"""Tests for ProxyMetrics 24h rolling event buffer."""

from __future__ import annotations

import time

from virtual_context.proxy.metrics import ProxyMetrics


def test_events_older_than_24h_are_evicted():
    """Events older than 24h should be pruned from snapshot."""
    m = ProxyMetrics()
    # Inject an old event by manipulating _recorded_at
    with m._lock:
        m._events.append({
            "type": "request", "input_tokens": 100, "_seq": 0,
            "_recorded_at": time.time() - 86401,
            "ts": "old",
        })
        m._seq = 1
    m.record({"type": "request", "input_tokens": 200})
    # Force eviction
    snap = m.snapshot()
    assert snap["total_requests"] == 1
    reqs = snap["recent_requests"]
    assert len(reqs) == 1
    assert reqs[0]["input_tokens"] == 200


def test_events_within_24h_are_retained():
    m = ProxyMetrics()
    m.record({"type": "request", "input_tokens": 100})
    m.record({"type": "request", "input_tokens": 200})
    snap = m.snapshot()
    assert snap["total_requests"] == 2


def test_event_buffer_capped_at_max():
    m = ProxyMetrics()
    for i in range(50_010):
        m.record({"type": "request", "input_tokens": i})
    with m._lock:
        assert len(m._events) <= 50_000
