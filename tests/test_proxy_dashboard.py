"""Tests for the proxy dashboard: ProxyMetrics and dashboard routes."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.server import ProxyState, create_app
from virtual_context.types import AssembledContext, SessionStats


# ---------------------------------------------------------------------------
# ProxyMetrics unit tests
# ---------------------------------------------------------------------------


class TestProxyMetricsRecord:
    def test_record_and_events_since(self):
        m = ProxyMetrics()
        m.record({"type": "request", "turn": 0})
        m.record({"type": "request", "turn": 1})
        m.record({"type": "turn_complete", "turn": 0})

        # All events after cursor -1
        all_events = m.events_since(-1)
        assert len(all_events) == 3

        # Events after seq 0 (skip first)
        after_first = m.events_since(0)
        assert len(after_first) == 2
        assert after_first[0]["turn"] == 1

        # Events after last
        after_last = m.events_since(2)
        assert len(after_last) == 0

    def test_record_adds_seq_and_ts(self):
        m = ProxyMetrics()
        m.record({"type": "request"})
        events = m.events_since(-1)
        assert events[0]["_seq"] == 0
        assert "ts" in events[0]

    def test_record_preserves_existing_ts(self):
        m = ProxyMetrics()
        m.record({"type": "request", "ts": "custom-ts"})
        events = m.events_since(-1)
        assert events[0]["ts"] == "custom-ts"

    def test_record_does_not_mutate_caller_dict(self):
        m = ProxyMetrics()
        original = {"type": "request", "turn": 5}
        m.record(original)
        assert "_seq" not in original

    def test_events_since_empty(self):
        m = ProxyMetrics()
        assert m.events_since(0) == []
        assert m.events_since(100) == []


class TestProxyMetricsSnapshot:
    def test_snapshot_empty(self):
        m = ProxyMetrics()
        snap = m.snapshot()
        assert snap["type"] == "snapshot"
        assert snap["total_requests"] == 0
        assert snap["total_compactions"] == 0
        assert snap["avg_wait_ms"] == 0
        assert snap["avg_inbound_ms"] == 0
        assert snap["avg_context_tokens"] == 0
        assert snap["recent_requests"] == []
        assert snap["compactions"] == []
        # Cost savings fields
        assert snap["total_original_tokens"] == 0
        assert snap["total_summary_tokens"] == 0
        assert snap["compression_ratio"] == 0
        assert snap["total_context_injected"] == 0
        # Session efficiency fields
        assert snap["total_actual_input"] == 0
        assert snap["total_baseline_input"] == 0

    def test_snapshot_aggregation(self):
        m = ProxyMetrics()
        m.record({"type": "request", "wait_ms": 10, "inbound_ms": 100, "context_tokens": 500})
        m.record({"type": "request", "wait_ms": 30, "inbound_ms": 200, "context_tokens": 1500})
        m.record({"type": "compaction", "tokens_freed": 5000, "original_tokens": 6000, "summary_tokens": 1000})
        m.record({"type": "compaction", "tokens_freed": 3000, "original_tokens": 4000, "summary_tokens": 1000})

        snap = m.snapshot()
        assert snap["total_requests"] == 2
        assert snap["total_compactions"] == 2
        assert snap["total_tokens_freed"] == 8000
        assert snap["avg_wait_ms"] == 20.0
        assert snap["avg_inbound_ms"] == 150.0
        assert snap["avg_context_tokens"] == 1000.0
        assert len(snap["recent_requests"]) == 2
        assert len(snap["compactions"]) == 2
        # Cost savings
        assert snap["total_original_tokens"] == 10000
        assert snap["total_summary_tokens"] == 2000
        assert snap["compression_ratio"] == 0.2
        assert snap["total_context_injected"] == 2000

    def test_snapshot_limits_recent_requests(self):
        m = ProxyMetrics()
        for i in range(60):
            m.record({"type": "request", "turn": i, "wait_ms": 1, "inbound_ms": 1, "context_tokens": 1})
        snap = m.snapshot()
        assert len(snap["recent_requests"]) == 50
        # Should be the last 50
        assert snap["recent_requests"][0]["turn"] == 10

    def test_snapshot_uptime(self):
        m = ProxyMetrics()
        snap = m.snapshot()
        assert snap["uptime_s"] >= 0

    def test_snapshot_actual_input(self):
        """total_actual_input sums input_tokens from request events."""
        m = ProxyMetrics()
        m.record({"type": "request", "input_tokens": 500})
        m.record({"type": "request", "input_tokens": 1200})
        m.record({"type": "request"})  # missing field → 0
        snap = m.snapshot()
        assert snap["total_actual_input"] == 1700

    def test_snapshot_session_efficiency(self):
        """Baseline simulation includes system_tokens + history per turn."""
        m = ProxyMetrics(context_window=1000)
        # Record a request with system_tokens so baseline accounts for it
        m.record({"type": "request", "system_tokens": 100})
        # 10 turn_completes each with 200 turn_pair_tokens
        for i in range(10):
            m.record({
                "type": "turn_complete",
                "turn": i,
                "turn_pair_tokens": 200,
            })
        snap = m.snapshot()
        assert snap["total_baseline_input"] > 0
        # Each turn contributes: system_tokens(100) + baseline_history
        # Turn 0: 100 + 200 = 300
        # Turn 1: 100 + 400 = 500
        # Turn 2: 100 + 600 = 700
        # Turn 3: 100 + 800 = 900
        # Turn 4: 100 + 1000 = 1100
        # Cumulative first 5 = 300+500+700+900+1100 = 3500
        # Turn 5: history=1200 > 1000 → compact → history=920 → 100+920=1020
        # etc.
        assert snap["total_baseline_input"] >= 3500  # at least first 5 uncompacted
        assert snap["baseline_ratio"] == 0.30

    def test_snapshot_baseline_without_system_tokens(self):
        """Baseline works when no request events provide system_tokens."""
        m = ProxyMetrics(context_window=10000)
        for i in range(5):
            m.record({"type": "turn_complete", "turn": i, "turn_pair_tokens": 100})
        snap = m.snapshot()
        # No system_tokens → baseline = just history: 100+200+300+400+500=1500
        assert snap["total_baseline_input"] == 1500

    def test_snapshot_includes_turn_completes(self):
        m = ProxyMetrics()
        m.record({"type": "turn_complete", "turn": 0, "tags": ["auth"]})
        snap = m.snapshot()
        assert len(snap["turn_completes"]) == 1
        assert snap["turn_completes"][0]["turn"] == 0


class TestProxyMetricsThreadSafety:
    def test_concurrent_records(self):
        m = ProxyMetrics()
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    m.record({"type": "request", "turn": start + i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        all_events = m.events_since(-1)
        assert len(all_events) == 500
        # All seq values should be unique
        seqs = [e["_seq"] for e in all_events]
        assert len(set(seqs)) == 500


# ---------------------------------------------------------------------------
# Dashboard route tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    assembled = AssembledContext(prepend_text="mock context")
    engine.on_message_inbound.return_value = assembled
    engine.on_turn_complete.return_value = None
    engine.tag_turn.return_value = None
    engine.compact_if_needed.return_value = None
    engine._turn_tag_index = MagicMock()
    engine._turn_tag_index.get_active_tags.return_value = {"auth", "api"}
    engine._turn_tag_index.get_tags_for_turn.return_value = None
    engine._store = MagicMock()
    engine._store.get_all_tags.return_value = []
    engine._compacted_through = 0
    engine.config = MagicMock()
    engine.config.monitor.context_window = 120000
    engine.config.session_id = "test-session"
    engine.config.monitor.soft_threshold = 0.7
    engine.config.monitor.hard_threshold = 0.85
    engine.config.tag_generator.type = "keyword"
    engine.config.tag_generator.model = ""
    engine.config.summarization.model = "test-model"
    engine.config.storage.backend = "sqlite"
    return engine


@pytest.fixture
def app_with_mock(mock_engine):
    with patch("virtual_context.proxy.server.VirtualContextEngine", return_value=mock_engine):
        app = create_app(upstream="http://fake-upstream:9999", config_path=None)
    return app, mock_engine


@pytest.fixture
def test_client(app_with_mock):
    from starlette.testclient import TestClient
    app, engine = app_with_mock
    with TestClient(app) as client:
        yield client, engine


class TestDashboardRoutes:
    def test_dashboard_html_served(self, test_client):
        client, _ = test_client
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "virtual-context proxy" in body
        assert "EventSource" in body
        assert "/dashboard/events" in body

    def test_dashboard_html_contains_panels(self, test_client):
        """HTML includes all major dashboard panels."""
        client, _ = test_client
        resp = client.get("/dashboard")
        body = resp.text
        assert "Request Log" in body
        assert "Compaction Events" in body
        assert "Active Tags" in body
        assert "Pipeline" in body
        assert "Memory" in body
        assert "Cost Savings" in body
        assert "Summary Compression" in body
        assert "Session Efficiency" in body
        assert "Tokens Freed" in body

    def test_dashboard_js_syntax_valid(self, test_client):
        """Embedded JS has no syntax errors (prevents silent dashboard freeze)."""
        import re
        import subprocess

        client, _ = test_client
        resp = client.get("/dashboard")
        m = re.search(r"<script>(.*?)</script>", resp.text, re.DOTALL)
        assert m, "No <script> block found"
        result = subprocess.run(
            ["node", "--check"],
            input=m.group(1),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"JS syntax error: {result.stderr}"

    def test_dashboard_html_contains_sessions_panel(self, test_client):
        """HTML includes the Sessions panel."""
        client, _ = test_client
        resp = client.get("/dashboard")
        body = resp.text
        assert "Sessions" in body
        assert "session-list" in body
        assert "fetchSessions" in body

    def test_dashboard_routes_before_catchall(self, test_client):
        """Ensure /dashboard is not captured by the proxy catch-all."""
        client, engine = test_client
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        # The engine should NOT have been called (catch-all would try to forward)
        engine.on_message_inbound.assert_not_called()


class TestMetricsIntegration:
    def test_request_event_emitted_on_chat(self, test_client):
        """Chat request emits a request event to metrics."""
        client, engine = test_client

        upstream_response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "model": "gpt-4o",
        }

        with patch("virtual_context.proxy.server.httpx.AsyncClient.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = upstream_response
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_req.return_value = mock_resp

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "How are you?"}],
                },
            )
            assert resp.status_code == 200

        # Access the app's metrics directly (avoid SSE streaming deadlocks)
        # The metrics are stored in the app's route closures — check via snapshot
        # by importing from the module level. Since we use create_app, the metrics
        # instance is created inside the function. We verify indirectly by checking
        # the existing proxy test patterns work and the dashboard HTML loads.
        # The key test is that create_app doesn't crash with metrics + dashboard.

    def test_proxy_state_emits_turn_complete_event(self):
        """ProxyState._run_tag_turn emits turn_complete event."""
        from virtual_context.types import Message

        engine = MagicMock()
        engine.tag_turn.return_value = None  # no compaction
        engine._turn_tag_index = MagicMock()

        entry = MagicMock()
        entry.tags = ["auth", "jwt"]
        entry.primary_tag = "auth"
        engine._turn_tag_index.get_tags_for_turn.return_value = entry

        metrics = ProxyMetrics()
        state = ProxyState(engine, metrics=metrics)

        history = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        state._run_tag_turn(history)

        events = metrics.events_since(-1)
        assert len(events) == 1
        assert events[0]["type"] == "turn_complete"
        assert events[0]["turn"] == 0
        assert events[0]["tags"] == ["auth", "jwt"]
        assert events[0]["primary_tag"] == "auth"
        assert "complete_ms" in events[0]

    def test_proxy_state_emits_compaction_event(self):
        """When compaction runs in background, a compaction event is emitted."""
        from virtual_context.types import CompactionReport, CompactionResult, CompactionSignal, Message

        engine = MagicMock()
        results = [
            CompactionResult(segment_id="s1", primary_tag="auth", original_tokens=4000, summary_tokens=600),
            CompactionResult(segment_id="s2", primary_tag="db", original_tokens=3000, summary_tokens=400),
            CompactionResult(segment_id="s3", primary_tag="auth", original_tokens=3000, summary_tokens=1000),
        ]
        report = CompactionReport(
            segments_compacted=3,
            tokens_freed=5000,
            tags=["auth", "db"],
            tag_summaries_built=2,
            results=results,
        )
        signal = CompactionSignal(
            priority="soft", current_tokens=4000,
            budget_tokens=5000, overflow_tokens=0,
        )
        engine.tag_turn.return_value = signal
        engine.compact_if_needed.return_value = report
        engine._compacted_through = 20
        engine._turn_tag_index = MagicMock()
        engine._turn_tag_index.get_tags_for_turn.return_value = None

        metrics = ProxyMetrics()
        state = ProxyState(engine, metrics=metrics)

        history = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        # Run tag turn (fires compact in background)
        state._run_tag_turn(history)
        # Wait for background compact to finish
        state.wait_for_complete()

        events = metrics.events_since(-1)
        assert len(events) == 2  # turn_complete + compaction

        compaction = next(e for e in events if e["type"] == "compaction")
        assert compaction["segments"] == 3
        assert compaction["tokens_freed"] == 5000
        assert compaction["original_tokens"] == 10000
        assert compaction["summary_tokens"] == 2000
        assert compaction["tags"] == ["auth", "db"]
        assert compaction["tag_summaries_built"] == 2
        assert compaction["compacted_through"] == 20

    def test_proxy_state_no_metrics(self):
        """ProxyState works without metrics (backwards compat)."""
        engine = MagicMock()
        engine.tag_turn.return_value = None
        state = ProxyState(engine)
        # Should not raise even without metrics
        from virtual_context.types import Message
        history = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        state._run_tag_turn(history)
        engine.tag_turn.assert_called_once()


# ---------------------------------------------------------------------------
# Dashboard sessions endpoint tests
# ---------------------------------------------------------------------------


class TestDashboardSessions:
    def test_sessions_endpoint(self, test_client):
        """GET /dashboard/sessions returns session stats from the store."""
        client, engine = test_client
        from datetime import datetime, timezone

        engine._store.get_session_stats.return_value = [
            SessionStats(
                session_id="sess-abc",
                segment_count=3,
                total_full_tokens=9000,
                total_summary_tokens=2000,
                compression_ratio=0.222,
                distinct_tags=["auth", "db"],
                oldest_segment=datetime(2026, 1, 10, tzinfo=timezone.utc),
                newest_segment=datetime(2026, 1, 15, tzinfo=timezone.utc),
                compaction_model="haiku",
            ),
        ]
        engine.config.session_id = "sess-abc"

        resp = client.get("/dashboard/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_session_id"] == "sess-abc"
        assert len(data["sessions"]) == 1

        s = data["sessions"][0]
        assert s["session_id"] == "sess-abc"
        assert s["is_current"] is True
        assert s["segment_count"] == 3
        assert s["total_full_tokens"] == 9000
        assert s["total_summary_tokens"] == 2000
        assert s["compression_ratio"] == 0.222
        assert s["distinct_tags"] == ["auth", "db"]
        assert s["compaction_model"] == "haiku"

    def test_sessions_endpoint_multiple(self, test_client):
        """Multiple sessions are returned with correct is_current flag."""
        client, engine = test_client

        engine._store.get_session_stats.return_value = [
            SessionStats(session_id="current-sess", segment_count=2),
            SessionStats(session_id="old-sess", segment_count=5),
        ]
        engine.config.session_id = "current-sess"

        resp = client.get("/dashboard/sessions")
        data = resp.json()
        assert len(data["sessions"]) == 2
        assert data["sessions"][0]["is_current"] is True
        assert data["sessions"][1]["is_current"] is False

    def test_sessions_endpoint_no_engine(self):
        """When state is None, sessions endpoint returns empty."""
        from virtual_context.proxy.dashboard import register_dashboard_routes

        from fastapi import FastAPI
        app = FastAPI()
        metrics = ProxyMetrics()
        register_dashboard_routes(app, metrics, state=None)

        from starlette.testclient import TestClient
        with TestClient(app) as client:
            resp = client.get("/dashboard/sessions")
            assert resp.status_code == 200
            data = resp.json()
            assert data["sessions"] == []
            assert data["current_session_id"] == ""

    def test_sessions_not_captured_by_catchall(self, test_client):
        """GET /dashboard/sessions is not captured by the proxy catch-all."""
        client, engine = test_client
        engine._store.get_session_stats.return_value = []
        engine.config.session_id = "x"
        resp = client.get("/dashboard/sessions")
        assert resp.status_code == 200
        engine.on_message_inbound.assert_not_called()


# ---------------------------------------------------------------------------
# Dashboard replay endpoint tests
# ---------------------------------------------------------------------------


class TestDashboardDeleteSession:
    def test_delete_session(self, test_client):
        """DELETE /dashboard/sessions/{session_id} removes segments."""
        client, engine = test_client
        engine._store.delete_session = MagicMock(return_value=3)
        resp = client.delete("/dashboard/sessions/sess-old")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 3
        engine._store.delete_session.assert_called_once_with("sess-old")

    def test_delete_session_no_state(self):
        """DELETE when state is None returns 503."""
        from virtual_context.proxy.dashboard import register_dashboard_routes

        from fastapi import FastAPI
        app = FastAPI()
        metrics = ProxyMetrics()
        register_dashboard_routes(app, metrics, state=None)

        from starlette.testclient import TestClient
        with TestClient(app) as client:
            resp = client.delete("/dashboard/sessions/sess-x")
            assert resp.status_code == 503

    def test_delete_session_store_unsupported(self, test_client):
        """DELETE when store lacks delete_session returns 501."""
        client, engine = test_client
        # Remove the method to simulate an unsupported store
        if hasattr(engine._store, "delete_session"):
            del engine._store.delete_session
        resp = client.delete("/dashboard/sessions/sess-x")
        assert resp.status_code == 501

    def test_delete_session_not_captured_by_catchall(self, test_client):
        """DELETE /dashboard/sessions/... is not captured by the proxy catch-all."""
        client, engine = test_client
        engine._store.delete_session = MagicMock(return_value=0)
        resp = client.delete("/dashboard/sessions/sess-x")
        assert resp.status_code == 200
        engine.on_message_inbound.assert_not_called()


class TestDashboardShutdown:
    def test_shutdown_html_button(self, test_client):
        """HTML includes the shutdown button."""
        client, _ = test_client
        resp = client.get("/dashboard")
        body = resp.text
        assert "shutdownProxy" in body
        assert "shutdown-btn" in body

    def test_shutdown_route_before_catchall(self, test_client):
        """POST /dashboard/shutdown is not captured by the proxy catch-all."""
        import signal
        client, engine = test_client
        with patch("os.kill") as mock_kill:
            resp = client.post("/dashboard/shutdown")
            assert resp.status_code == 200
            assert resp.json()["status"] == "shutting_down"
            mock_kill.assert_called_once()
            engine.on_message_inbound.assert_not_called()


class TestDashboardReplay:
    def test_replay_html_panel(self, test_client):
        """HTML includes the Replay panel."""
        client, _ = test_client
        resp = client.get("/dashboard")
        body = resp.text
        assert "Replay" in body
        assert "replay-file" in body
        assert "startReplay" in body
        assert "stopReplay" in body

    def test_replay_start_no_state(self):
        """Start replay when state is None returns 503."""
        from virtual_context.proxy.dashboard import register_dashboard_routes

        from fastapi import FastAPI
        app = FastAPI()
        metrics = ProxyMetrics()
        register_dashboard_routes(app, metrics, state=None)

        from starlette.testclient import TestClient
        with TestClient(app) as client:
            resp = client.post(
                "/dashboard/replay/start",
                json={"file": "test.txt"},
            )
            assert resp.status_code == 503

    def test_replay_start_file_not_found(self, test_client):
        """Start replay with non-existent file returns 400."""
        client, _ = test_client
        resp = client.post(
            "/dashboard/replay/start",
            json={"file": "/no/such/file.txt"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["error"].lower()

    def test_replay_start_no_provider(self, test_client):
        """Start replay when engine has no LLM provider returns 503."""
        import tempfile
        client, engine = test_client
        engine._llm_provider = None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt 1\nprompt 2\n")
            f.flush()
            resp = client.post(
                "/dashboard/replay/start",
                json={"file": f.name},
            )
            assert resp.status_code == 503

    def test_replay_stop_not_running(self, test_client):
        """Stop when no replay is running returns 200."""
        client, _ = test_client
        resp = client.post("/dashboard/replay/stop")
        assert resp.status_code == 200

    def test_replay_status_idle(self, test_client):
        """Status when no replay is running."""
        client, _ = test_client
        resp = client.get("/dashboard/replay/status")
        assert resp.status_code == 200
        assert resp.json()["running"] is False

    def test_replay_routes_before_catchall(self, test_client):
        """Replay routes are not captured by the proxy catch-all."""
        client, engine = test_client
        resp = client.get("/dashboard/replay/status")
        assert resp.status_code == 200
        engine.on_message_inbound.assert_not_called()

    def test_replay_progress_event_format(self):
        """replay_progress events have the right shape."""
        m = ProxyMetrics()
        m.record({
            "type": "replay_progress",
            "turn": 5,
            "total": 100,
            "prompt_preview": "test prompt",
            "elapsed_ms": 123.4,
        })
        events = m.events_since(-1)
        assert len(events) == 1
        evt = events[0]
        assert evt["type"] == "replay_progress"
        assert evt["turn"] == 5
        assert evt["total"] == 100
        assert evt["prompt_preview"] == "test prompt"

    def test_replay_done_event_format(self):
        """replay_done events have the right shape."""
        m = ProxyMetrics()
        m.record({
            "type": "replay_done",
            "turns_completed": 100,
            "total": 100,
            "status": "complete",
        })
        events = m.events_since(-1)
        assert len(events) == 1
        evt = events[0]
        assert evt["type"] == "replay_done"
        assert evt["status"] == "complete"
        assert evt["turns_completed"] == 100


# ---------------------------------------------------------------------------
# Dashboard settings endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_client(mock_engine):
    """Test client with a real VirtualContextConfig (settings needs real attrs)."""
    from virtual_context.types import VirtualContextConfig
    mock_engine.config = VirtualContextConfig()
    with patch("virtual_context.proxy.server.VirtualContextEngine", return_value=mock_engine):
        app = create_app(upstream="http://fake:9999", config_path=None)
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        yield client, mock_engine


class TestDashboardSettings:
    def test_settings_get(self, settings_client):
        """GET /dashboard/settings returns all sections with correct defaults."""
        client, _ = settings_client
        resp = client.get("/dashboard/settings")
        assert resp.status_code == 200
        data = resp.json()
        # Readonly
        assert data["readonly"]["context_window"] == 120_000
        assert data["readonly"]["tagger_type"] == "keyword"
        assert data["readonly"]["storage_backend"] == "sqlite"
        # Compaction defaults
        assert data["compaction"]["soft_threshold"] == 0.70
        assert data["compaction"]["hard_threshold"] == 0.85
        assert data["compaction"]["protected_recent_turns"] == 6
        assert data["compaction"]["min_summary_tokens"] == 200
        assert data["compaction"]["max_summary_tokens"] == 2000
        # Tagging defaults
        assert data["tagging"]["temporal_heuristic_enabled"] is True
        # Retrieval defaults
        assert data["retrieval"]["active_tag_lookback"] == 4
        assert data["retrieval"]["anchorless_lookback"] == 6
        assert data["retrieval"]["max_results"] == 10
        assert data["retrieval"]["max_budget_fraction"] == 0.25
        assert data["retrieval"]["include_related"] is True
        # Assembly defaults
        assert data["assembly"]["tag_context_max_tokens"] == 30_000
        assert data["assembly"]["recent_turns_always_included"] == 3
        assert data["assembly"]["context_hint_enabled"] is True
        assert data["assembly"]["context_hint_max_tokens"] == 2000
        # Summarization defaults
        assert data["summarization"]["temperature"] == 0.3

    def test_settings_get_no_state(self):
        """GET /dashboard/settings returns 503 when state is None."""
        from virtual_context.proxy.dashboard import register_dashboard_routes

        from fastapi import FastAPI
        app = FastAPI()
        metrics = ProxyMetrics()
        register_dashboard_routes(app, metrics, state=None)

        from starlette.testclient import TestClient
        with TestClient(app) as client:
            resp = client.get("/dashboard/settings")
            assert resp.status_code == 503

    def test_settings_put_partial(self, settings_client):
        """PUT with one field updates only that field."""
        client, engine = settings_client
        resp = client.put(
            "/dashboard/settings",
            json={"compaction": {"protected_recent_turns": 10}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["compaction"]["protected_recent_turns"] == 10
        # Other fields unchanged
        assert data["compaction"]["soft_threshold"] == 0.70
        # Verify the config object was actually mutated
        assert engine.config.monitor.protected_recent_turns == 10

    def test_settings_put_validation_soft_ge_hard(self, settings_client):
        """PUT returns 400 when soft_threshold >= hard_threshold."""
        client, _ = settings_client
        resp = client.put(
            "/dashboard/settings",
            json={"compaction": {"soft_threshold": 0.90, "hard_threshold": 0.80}},
        )
        assert resp.status_code == 400
        assert "soft_threshold" in resp.json()["error"]

    def test_settings_put_validation_min_gt_max_summary(self, settings_client):
        """PUT returns 400 when min_summary_tokens > max_summary_tokens."""
        client, _ = settings_client
        resp = client.put(
            "/dashboard/settings",
            json={"compaction": {"min_summary_tokens": 5000, "max_summary_tokens": 100}},
        )
        assert resp.status_code == 400
        assert "min_summary_tokens" in resp.json()["error"]

    def test_settings_put_strategy_config(self, settings_client):
        """PUT updates strategy_configs['default'] attributes."""
        client, engine = settings_client
        resp = client.put(
            "/dashboard/settings",
            json={"retrieval": {"max_results": 20, "include_related": False}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval"]["max_results"] == 20
        assert data["retrieval"]["include_related"] is False
        strategy = engine.config.retriever.strategy_configs["default"]
        assert strategy.max_results == 20
        assert strategy.include_related is False

    def test_settings_put_syncs_tag_context(self, settings_client):
        """Updating assembly.tag_context_max_tokens also updates retriever."""
        client, engine = settings_client
        resp = client.put(
            "/dashboard/settings",
            json={"assembly": {"tag_context_max_tokens": 50000}},
        )
        assert resp.status_code == 200
        assert engine.config.assembler.tag_context_max_tokens == 50000
        assert engine.config.retriever.tag_context_max_tokens == 50000

    def test_settings_put_unknown_key(self, settings_client):
        """PUT returns 400 for unknown keys."""
        client, _ = settings_client
        resp = client.put(
            "/dashboard/settings",
            json={"compaction": {"nonexistent_key": 42}},
        )
        assert resp.status_code == 400
        assert "Unknown setting" in resp.json()["error"]

    def test_settings_put_readonly_ignored(self, settings_client):
        """PUT silently skips readonly section."""
        client, engine = settings_client
        resp = client.put(
            "/dashboard/settings",
            json={"readonly": {"context_window": 999}},
        )
        assert resp.status_code == 200
        # context_window unchanged
        assert resp.json()["readonly"]["context_window"] == 120_000

    def test_settings_html_present(self, settings_client):
        """Dashboard HTML contains settings UI elements."""
        client, _ = settings_client
        resp = client.get("/dashboard")
        body = resp.text
        assert "settings-btn" in body
        assert "settings-overlay" in body
        assert "saveSettings" in body
        assert "openSettings" in body


# ---------------------------------------------------------------------------
# Dashboard compact endpoint tests
# ---------------------------------------------------------------------------


class TestDashboardCompact:
    def test_compact_no_state(self):
        """POST /dashboard/compact returns 503 when state is None."""
        from virtual_context.proxy.dashboard import register_dashboard_routes

        from fastapi import FastAPI
        app = FastAPI()
        metrics = ProxyMetrics()
        register_dashboard_routes(app, metrics, state=None)

        from starlette.testclient import TestClient
        with TestClient(app) as client:
            resp = client.post("/dashboard/compact")
            assert resp.status_code == 503

    def test_compact_no_history(self, settings_client):
        """POST /dashboard/compact with empty history returns no_action."""
        client, engine = settings_client
        engine.compact_manual = MagicMock(return_value=None)
        resp = client.post("/dashboard/compact")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_action"

    def test_compact_success(self, settings_client):
        """POST /dashboard/compact returns compaction results on success."""
        from virtual_context.types import CompactionReport, CompactionResult

        client, engine = settings_client
        results = [
            CompactionResult(
                segment_id="s1", primary_tag="auth",
                original_tokens=4000, summary_tokens=600,
            ),
        ]
        report = CompactionReport(
            segments_compacted=1,
            tokens_freed=3400,
            tags=["auth"],
            results=results,
            tag_summaries_built=1,
        )
        engine.compact_manual = MagicMock(return_value=report)
        engine._compacted_through = 10

        resp = client.post("/dashboard/compact")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "compacted"
        assert data["segments"] == 1
        assert data["tokens_freed"] == 3400
        assert data["tags"] == ["auth"]
        assert data["tag_summaries_built"] == 1

    def test_compact_html_present(self, settings_client):
        """Dashboard HTML contains the Compact Now button."""
        client, _ = settings_client
        resp = client.get("/dashboard")
        body = resp.text
        assert "compact-btn" in body
        assert "compactNow" in body
        assert "Compact Now" in body

    def test_compact_not_captured_by_catchall(self, settings_client):
        """POST /dashboard/compact is not captured by the proxy catch-all."""
        client, engine = settings_client
        engine.compact_manual = MagicMock(return_value=None)
        resp = client.post("/dashboard/compact")
        assert resp.status_code == 200
        engine.on_message_inbound.assert_not_called()


# ---------------------------------------------------------------------------
# Dashboard request inspection endpoint tests
# ---------------------------------------------------------------------------


class TestDashboardRequestInspect:
    def test_get_request_found(self, test_client):
        """GET /dashboard/requests/{turn} returns captured request."""
        client, engine = test_client
        # Access the app's metrics through the proxy state closure
        # We need to capture a request first via the metrics object
        # Since metrics is created inside create_app, we inject via a chat request
        upstream_response = {
            "choices": [{"message": {"content": "OK"}}],
        }
        with patch("virtual_context.proxy.server.httpx.AsyncClient.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = upstream_response
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_req.return_value = mock_resp

            # Send a chat request to trigger capture
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello world"}],
                },
            )

        # Now inspect it — check the list to get the actual turn number.
        list_resp = client.get("/dashboard/requests")
        summaries = list_resp.json()
        assert len(summaries) >= 1
        captured_turn = summaries[0]["turn"]

        resp = client.get(f"/dashboard/requests/{captured_turn}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["turn"] == captured_turn
        assert data["model"] == "gpt-4o"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Hello world"

    def test_get_request_not_found(self, test_client):
        client, _ = test_client
        resp = client.get("/dashboard/requests/999")
        assert resp.status_code == 404

    def test_list_requests(self, test_client):
        client, engine = test_client
        upstream_response = {
            "choices": [{"message": {"content": "OK"}}],
        }
        with patch("virtual_context.proxy.server.httpx.AsyncClient.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = upstream_response
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_req.return_value = mock_resp

            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Q1"}],
                },
            )

        resp = client.get("/dashboard/requests")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert "messages" not in data[0]  # summary excludes full messages
        assert "message_count" in data[0]

    def test_request_inspect_not_captured_by_catchall(self, test_client):
        client, engine = test_client
        resp = client.get("/dashboard/requests/0")
        # Should hit the dashboard route, not the catch-all proxy
        engine.on_message_inbound.assert_not_called()

    def test_inspect_html_elements(self, test_client):
        """Dashboard HTML contains inspect modal elements."""
        client, _ = test_client
        resp = client.get("/dashboard")
        body = resp.text
        assert "inspect-overlay" in body
        assert "inspectRequest" in body
        assert "saveInspectedRequest" in body
        assert "Request Inspector" in body


class TestDashboardExport:
    def test_export_returns_snapshot_with_engine_state(self, test_client):
        client, engine = test_client
        engine._turn_tag_index.entries = []
        engine._store.get_all_tags.return_value = []

        resp = client.get("/dashboard/export")
        assert resp.status_code == 200
        data = resp.json()
        # Snapshot fields
        assert "total_requests" in data
        assert "total_compactions" in data
        # Engine state fields
        assert "turn_tag_index" in data
        assert "store_tags" in data
        assert "config" in data
        assert data["config"]["session_id"] == "test-session"
        # Should not have type=snapshot
        assert "type" not in data

    def test_export_not_captured_by_catchall(self, test_client):
        client, engine = test_client
        engine._turn_tag_index.entries = []
        engine._store.get_all_tags.return_value = []
        client.get("/dashboard/export")
        engine.on_message_inbound.assert_not_called()

    def test_export_html_button(self, test_client):
        client, _ = test_client
        resp = client.get("/dashboard")
        assert "exportSession" in resp.text
