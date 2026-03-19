"""Tests for SQLite-backed metrics event persistence."""

import json
import time

import pytest

from virtual_context.proxy.metrics import ProxyMetrics


class TestMetricsSQLitePersistence:
    """Events should survive a ProxyMetrics restart when db_path is provided."""

    def test_events_persist_across_restart(self, tmp_path):
        db = str(tmp_path / "metrics.db")

        # Instance 1: record 10 events
        m1 = ProxyMetrics(db_path=db)
        for i in range(10):
            m1.record({"type": "request", "turn": i, "conversation_id": "c1"})

        events1 = m1.events_since(-1)
        assert len(events1) == 10

        # Instance 2: same DB, simulating restart
        m2 = ProxyMetrics(db_path=db)
        events2 = m2.events_since(-1)
        assert len(events2) == 10, (
            f"Expected 10 events after restart, got {len(events2)}"
        )

    def test_seq_continues_after_restart(self, tmp_path):
        db = str(tmp_path / "metrics.db")

        m1 = ProxyMetrics(db_path=db)
        for i in range(5):
            m1.record({"type": "request", "turn": i})

        m2 = ProxyMetrics(db_path=db)
        m2.record({"type": "request", "turn": 5})

        events = m2.events_since(-1)
        seqs = [e["_seq"] for e in events]
        assert seqs == list(range(6)), f"Seqs should be 0-5, got {seqs}"

    def test_eviction_removes_old_events(self, tmp_path):
        db = str(tmp_path / "metrics.db")
        m = ProxyMetrics(db_path=db)

        # Record an event with a fake old timestamp
        old_event = {
            "type": "request",
            "turn": 0,
            "_recorded_at": time.time() - 90_000,  # 25 hours ago
        }
        m.record(old_event)

        # Record 100 more to trigger eviction (every 100 calls)
        for i in range(100):
            m.record({"type": "request", "turn": i + 1})

        events = m.events_since(-1)
        # The old event should be evicted
        turns = [e.get("turn") for e in events]
        assert 0 not in turns, "Old event (turn=0) should have been evicted"
        assert len(events) == 100  # only the 100 new ones

    def test_no_db_path_works_in_memory_only(self):
        m = ProxyMetrics()
        m.record({"type": "request", "turn": 0})
        events = m.events_since(-1)
        assert len(events) == 1

    def test_snapshot_reconstructs_from_sqlite(self, tmp_path):
        db = str(tmp_path / "metrics.db")

        m1 = ProxyMetrics(db_path=db, context_window=100_000)
        for i in range(3):
            m1.record({
                "type": "request",
                "turn": i,
                "input_tokens": 1000,
                "context_tokens": 200,
                "raw_input_tokens": 5000,
                "conversation_id": "c1",
            })

        # Restart
        m2 = ProxyMetrics(db_path=db, context_window=100_000)
        snap = m2.snapshot()
        assert snap["total_requests"] == 3
        assert snap["total_actual_input"] == 3000
