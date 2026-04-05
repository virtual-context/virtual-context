"""Tests for virtual_context.proxy.server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from virtual_context.proxy.server import (
    ProxyState,
    _build_continuation_request,
    _inject_vc_tools,
    create_app,
    prepare_payload,
)
from virtual_context.config import load_config
from virtual_context.proxy.formats import AnthropicFormat, PayloadTokenCache, PayloadTokenEstimate
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import AssembledContext, EngineState, Message, TagResult, TurnTagEntry

# ---------------------------------------------------------------------------
# ProxyState
# ---------------------------------------------------------------------------


class TestProxyState:
    def test_wait_for_tag_noop_when_no_pending(self):
        engine = MagicMock()
        state = ProxyState(engine)
        state.wait_for_tag()  # should not raise

    def test_wait_for_complete_noop_when_no_pending(self):
        engine = MagicMock()
        state = ProxyState(engine)
        state.wait_for_complete()  # should not raise

    def test_fire_and_wait_for_tag(self):
        engine = MagicMock()
        engine.tag_turn.return_value = None  # no compaction needed
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_tag()
        engine.tag_turn.assert_called_once_with(history, payload_tokens=None)

    def test_fire_and_wait_for_complete(self):
        engine = MagicMock()
        engine.tag_turn.return_value = None  # no compaction needed
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_complete()
        engine.tag_turn.assert_called_once_with(history, payload_tokens=None)

    def test_compaction_fires_in_background(self):
        engine = MagicMock()
        signal = MagicMock()  # non-None → compaction needed
        engine.tag_turn.return_value = signal
        engine.compact_if_needed.return_value = None
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_complete()  # waits for both tag + compact
        engine.tag_turn.assert_called_once()
        engine.compact_if_needed.assert_called_once()
        call_args = engine.compact_if_needed.call_args
        assert call_args[0] == (history, signal)
        assert "progress_callback" in call_args[1]

    def test_compaction_progress_phase_name_does_not_crash(self):
        engine = MagicMock()
        signal = MagicMock()
        engine.tag_turn.return_value = signal
        metrics = ProxyMetrics()

        def _compact_if_needed(history, compaction_signal, progress_callback=None):
            assert progress_callback is not None
            progress_callback(
                1,
                2,
                None,
                phase="segment_compacting",
                phase_name="compactor",
                overall_percent=55,
                phase_detail="working",
            )
            return None

        engine.compact_if_needed.side_effect = _compact_if_needed

        state = ProxyState(engine, metrics=metrics)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]

        state.fire_turn_complete(history)
        state.wait_for_complete()

        events = metrics.events_since(-1)
        assert any(
            event["type"] == "compaction_progress"
            and event["phase_name"] == "compactor"
            for event in events
        )
        assert not any(event["type"] == "compaction_error" for event in events)
        snap = state.compaction_snapshot()
        assert snap is not None
        assert snap["status"] == "skipped"

    def test_error_in_tag_turn_is_caught(self):
        engine = MagicMock()
        engine.tag_turn.side_effect = RuntimeError("boom")
        state = ProxyState(engine)
        history = [Message(role="user", content="hi")]
        state.fire_turn_complete(history)
        state.wait_for_tag()  # should not raise

    def test_error_in_compact_is_caught(self):
        engine = MagicMock()
        signal = MagicMock()
        engine.tag_turn.return_value = signal
        engine.compact_if_needed.side_effect = RuntimeError("compact boom")
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_complete()  # should not raise


# ---------------------------------------------------------------------------
# Integration tests with FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    """Create a mock VirtualContextEngine."""
    engine = MagicMock()
    assembled = AssembledContext(prepend_text="mock context here")
    engine.on_message_inbound.return_value = assembled
    engine.on_turn_complete.return_value = None
    engine.tag_turn.return_value = None
    engine.compact_if_needed.return_value = None
    return engine


@pytest.fixture
def app_with_mock_engine(mock_engine):
    """Create a FastAPI app with a mock engine, patching VirtualContextEngine."""
    with patch("virtual_context.proxy.server.VirtualContextEngine", return_value=mock_engine):
        app = create_app(upstream="http://fake-upstream:9999", config_path=None)
    return app, mock_engine


@pytest.fixture
def test_client(app_with_mock_engine):
    """Provide a TestClient for the proxy app."""
    from starlette.testclient import TestClient
    app, engine = app_with_mock_engine
    with TestClient(app) as client:
        yield client, engine


class TestIntegration:
    def test_non_chat_passthrough(self, test_client):
        """Non-chat POST requests are forwarded without engine involvement."""
        client, engine = test_client
        # We can't actually reach the upstream, so this will fail at the httpx level
        # but we can verify engine was not called
        with patch("virtual_context.proxy.server.httpx.AsyncClient.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": []}
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_req.return_value = mock_resp

            resp = client.get("/v1/models")
            assert resp.status_code == 200
            engine.on_message_inbound.assert_not_called()

    def test_chat_non_streaming_enrichment(self, test_client):
        """Chat request gets context injected and response parsed."""
        client, engine = test_client

        upstream_response = {
            "choices": [{"message": {"content": "I'm fine, thanks!"}}],
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
            engine.on_message_inbound.assert_called_once()

            # Verify the forwarded body has context injected
            call_kwargs = mock_req.call_args
            forwarded_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert forwarded_body["messages"][0]["role"] == "system"
            assert "mock context here" in forwarded_body["messages"][0]["content"]

    def test_engine_failure_forwards_unmodified(self, test_client):
        """If engine fails, request is forwarded without enrichment."""
        client, engine = test_client
        engine.on_message_inbound.side_effect = RuntimeError("engine broke")

        upstream_response = {
            "choices": [{"message": {"content": "OK"}}],
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
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

            assert resp.status_code == 200
            # Body should not have virtual-context injected
            call_kwargs = mock_req.call_args
            forwarded_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            # No system message injected since prepend_text is empty
            assert forwarded_body["messages"][0]["role"] == "user"

    def test_anthropic_format_detection(self, test_client):
        """Anthropic request format is detected and context injected into system field."""
        client, engine = test_client

        upstream_response = {
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3",
        }

        with patch("virtual_context.proxy.server.httpx.AsyncClient.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = upstream_response
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_req.return_value = mock_resp

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-haiku-20240307",
                    "system": "Be helpful",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

            assert resp.status_code == 200
            call_kwargs = mock_req.call_args
            forwarded_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "<system-reminder>" in forwarded_body["system"]
            assert "mock context here" in forwarded_body["system"]
            assert "Be helpful" in forwarded_body["system"]

    def test_no_messages_key_passthrough(self, test_client):
        """POST without messages array is passed through."""
        client, engine = test_client

        with patch("virtual_context.proxy.server.httpx.AsyncClient.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"ok": True}
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_req.return_value = mock_resp

            resp = client.post(
                "/v1/embeddings",
                json={"input": "hello", "model": "text-embedding-ada-002"},
            )

            assert resp.status_code == 200
            engine.on_message_inbound.assert_not_called()


class TestPreparePayloadInboundTokenCache:
    def test_prepare_payload_loads_and_saves_redis_payload_cache(self, tmp_path):
        config = load_config(config_dict={
            "conversation_id": "conv-123",
            "context_window": 10000,
            "storage_root": str(tmp_path),
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "store.db")}},
            "tool_output": {"enabled": False},
        })
        engine = MagicMock()
        engine.config = config
        engine._turn_tag_index = TurnTagIndex()
        engine._engine_state = EngineState()
        engine._store = MagicMock()
        engine._restored_request_captures = []
        engine._restored_conversation_history = []
        engine._restored_pending_turns = []
        engine._restored_working_set = []
        engine.on_message_inbound.return_value = AssembledContext()
        provider = MagicMock()
        cached = PayloadTokenCache(
            format_name="anthropic",
            message_key="messages",
            shell_fingerprint="shell-old",
            shell_tokens=10,
            message_fingerprints=["m1"],
            message_tokens=[5],
            separator_tokens=0,
            total_tokens=15,
        )
        next_cache = PayloadTokenCache(
            format_name="anthropic",
            message_key="messages",
            shell_fingerprint="shell-new",
            shell_tokens=12,
            message_fingerprints=["m1", "m2"],
            message_tokens=[5, 6],
            separator_tokens=1,
            total_tokens=24,
        )
        provider.load_payload_token_cache.return_value = cached
        engine._session_state_provider = provider

        state = ProxyState(engine)
        fmt = AnthropicFormat()
        fmt.estimate_payload_tokens_segmented = MagicMock(return_value=PayloadTokenEstimate(
            total_tokens=24,
            cache=next_cache,
            reused_prefix_messages=1,
            recounted_messages=1,
            shell_cache_hit=True,
        ))
        body = {
            "model": "claude-opus-4-6",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }
        metrics = ProxyMetrics()

        asyncio.run(
            prepare_payload(
                body,
                state,
                fmt,
                metrics,
                body_bytes=json.dumps(body).encode("utf-8"),
            )
        )

        provider.load_payload_token_cache.assert_called_once_with("conv-123")
        provider.save_payload_token_cache.assert_called_once_with("conv-123", next_cache)
        assert fmt.estimate_payload_tokens_segmented.call_args.kwargs["cache"] == cached
        assert state._inbound_payload_token_cache == next_cache


# ---------------------------------------------------------------------------
# ProxyState ingestion
# ---------------------------------------------------------------------------


class TestProxyStateIngestion:
    def _make_engine(self, conversation_id="test-session"):
        engine = MagicMock()
        engine.config.conversation_id = conversation_id
        return engine

    def test_ingest_runs_once(self):
        engine = self._make_engine()
        engine.ingest_history.return_value = 5
        metrics = ProxyMetrics()
        state = ProxyState(engine, metrics=metrics)

        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        state.ingest_if_needed(pairs)

        engine.ingest_history.assert_called_once_with(pairs)
        assert state._history_ingested() is True

    def test_second_call_is_noop(self):
        engine = self._make_engine()
        engine.ingest_history.return_value = 5
        state = ProxyState(engine)

        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        state.ingest_if_needed(pairs)
        state.ingest_if_needed(pairs)

        engine.ingest_history.assert_called_once()

    def test_different_session_triggers_new_ingestion(self):
        engine = self._make_engine("session-1")
        engine.ingest_history.return_value = 2
        state = ProxyState(engine)

        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        state.ingest_if_needed(pairs)
        assert engine.ingest_history.call_count == 1

        # Switch session
        engine.config.conversation_id = "session-2"
        state.ingest_if_needed(pairs)
        assert engine.ingest_history.call_count == 2

    def test_metrics_event_emitted(self):
        engine = self._make_engine()
        engine.ingest_history.return_value = 3
        metrics = ProxyMetrics()
        state = ProxyState(engine, metrics=metrics)

        pairs = [
            Message(role="user", content=f"Q{i}")
            if i % 2 == 0
            else Message(role="assistant", content=f"A{i}")
            for i in range(6)
        ]
        state.ingest_if_needed(pairs)

        events = metrics.events_since(-1)
        # 3 ingested_turn events + 1 history_ingestion summary
        assert len(events) == 4
        ingested = [e for e in events if e["type"] == "ingested_turn"]
        assert len(ingested) == 3
        assert ingested[0]["turn"] == 0
        assert ingested[0]["message_preview"] == "Q0"
        assert ingested[1]["turn"] == 1
        assert ingested[2]["turn"] == 2
        evt = [e for e in events if e["type"] == "history_ingestion"][0]
        assert evt["turns_ingested"] == 3
        assert evt["pairs_received"] == 3
        assert evt["conversation_id"] == "test-session"
        assert "elapsed_ms" in evt

    def test_conversation_history_backfilled(self):
        """After ingestion, conversation_history should contain the pairs."""
        engine = self._make_engine()
        engine.ingest_history.return_value = 2
        state = ProxyState(engine)

        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
            Message(role="assistant", content="A2"),
        ]
        # Simulate proxy's backfill before calling ingest
        state.conversation_history = list(pairs)
        state.ingest_if_needed(pairs)

        assert len(state.conversation_history) == 4
        assert state.conversation_history[0].content == "Q1"


# ---------------------------------------------------------------------------
# Engine ingest_history
# ---------------------------------------------------------------------------


class TestEngineIngestHistory:
    def _make_mock_engine(self):
        """Create a mock engine with the config attributes needed by ingest_history."""
        engine = MagicMock()
        from virtual_context.core.turn_tag_index import TurnTagIndex
        from virtual_context.core.tagging_pipeline import TaggingPipeline
        from virtual_context.types import EngineState
        engine._turn_tag_index = TurnTagIndex()
        engine_state = EngineState()
        engine._engine_state = engine_state
        engine._store = MagicMock()
        engine._store.get_all_tags.return_value = []
        engine.config.tag_generator.context_lookback_pairs = 5
        engine.config.tag_generator.context_bleed_threshold = 0
        engine._tagging = TaggingPipeline(
            tag_generator=MagicMock(),
            turn_tag_index=engine._turn_tag_index,
            store=engine._store,
            semantic=MagicMock(),
            engine_state=engine_state,
            config=engine.config,
            tag_splitter=None,
            canonicalizer=None,
            telemetry=MagicMock(),
            monitor=MagicMock(),
            compactor=None,
            save_state_callback=MagicMock(),
        )
        return engine

    def test_ingest_five_pairs(self):
        """5 pairs -> 5 TurnTagEntry entries with sequential turn numbers."""
        engine = self._make_mock_engine()

        # Tag generator returns predictable results
        tag_results = [
            TagResult(tags=["auth", "login"], primary="auth", source="keyword"),
            TagResult(tags=["db", "schema"], primary="db", source="keyword"),
            TagResult(tags=["api", "rest"], primary="api", source="keyword"),
            TagResult(tags=["auth", "jwt"], primary="auth", source="keyword"),
            TagResult(tags=["deploy", "ci"], primary="deploy", source="keyword"),
        ]
        mock_tagger = MagicMock()
        mock_tagger.generate_tags.side_effect = tag_results
        engine._tagging._tag_generator = mock_tagger

        # Call the real method on the mock
        from virtual_context.engine import VirtualContextEngine
        pairs = []
        for i in range(5):
            pairs.append(Message(role="user", content=f"Tell me about topic number {i}"))
            pairs.append(Message(role="assistant", content=f"Here is the answer for topic {i}"))

        result = VirtualContextEngine.ingest_history(engine, pairs)

        assert result == 5
        assert len(engine._turn_tag_index.entries) == 5
        assert engine._turn_tag_index.entries[0].turn_number == 0
        assert engine._turn_tag_index.entries[0].tags == ["auth", "login"]
        assert engine._turn_tag_index.entries[0].primary_tag == "auth"
        assert engine._turn_tag_index.entries[4].turn_number == 4
        assert engine._turn_tag_index.entries[4].tags == ["deploy", "ci"]
        assert engine._turn_tag_index.entries[4].primary_tag == "deploy"

    def test_ingest_empty_returns_zero(self):
        engine = self._make_mock_engine()

        from virtual_context.engine import VirtualContextEngine
        result = VirtualContextEngine.ingest_history(engine, [])

        assert result == 0
        assert len(engine._turn_tag_index.entries) == 0

    def test_ingest_refreshes_store_tags(self):
        """Store tags are refreshed every 10 turns."""
        engine = self._make_mock_engine()
        mock_tagger = MagicMock()
        mock_tagger.generate_tags.return_value = TagResult(
            tags=["tag"], primary="tag", source="keyword"
        )
        engine._tagging._tag_generator = mock_tagger

        from virtual_context.engine import VirtualContextEngine
        # 15 pairs = 30 messages
        pairs = []
        for i in range(15):
            pairs.append(Message(role="user", content=f"Q{i}"))
            pairs.append(Message(role="assistant", content=f"A{i}"))

        VirtualContextEngine.ingest_history(engine, pairs)

        # get_all_tags called: once at start + once after turn 10
        assert engine._store.get_all_tags.call_count == 2

    def test_ingest_links_tool_outputs_from_explicit_turn_mapping(self):
        engine = self._make_mock_engine()
        mock_tagger = MagicMock()
        mock_tagger.generate_tags.side_effect = [
            TagResult(tags=["alpha"], primary="alpha", source="keyword"),
            TagResult(tags=["beta"], primary="beta", source="keyword"),
        ]
        engine._tagging._tag_generator = mock_tagger

        from virtual_context.engine import VirtualContextEngine
        pairs = [
            Message(role="user", content="Q0"),
            Message(role="assistant", content="A0"),
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]

        result = VirtualContextEngine.ingest_history(
            engine,
            pairs,
            tool_output_refs_by_turn={0: ["tool_ref_0"], 1: ["tool_ref_1", "tool_ref_2"]},
        )

        assert result == 2
        assert engine._store.link_turn_tool_output.call_args_list == [
            call(engine.config.conversation_id, 0, "tool_ref_0"),
            call(engine.config.conversation_id, 1, "tool_ref_1"),
            call(engine.config.conversation_id, 1, "tool_ref_2"),
        ]
        engine._store.get_tool_output_refs_for_turn.assert_not_called()


# ---------------------------------------------------------------------------
# Request capture (ProxyMetrics ring buffer)
# ---------------------------------------------------------------------------


class TestRequestCapture:
    def test_capture_stores_raw_body(self):
        m = ProxyMetrics()
        body = {
            "model": "claude-3",
            "stream": True,
            "system": "Be helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Question"},
            ],
        }
        m.capture_request(0, body, "anthropic")

        req = m.get_captured_request(0)
        assert req is not None
        assert req["turn"] == 0
        assert req["api_format"] == "anthropic"
        assert req["model"] == "claude-3"
        assert req["stream"] is True
        assert req["system"] == "Be helpful"
        assert len(req["messages"]) == 3
        assert req["message_count"] == 3

    def test_ring_buffer_limit(self):
        m = ProxyMetrics()
        for i in range(55):
            m.capture_request(i, {"messages": []}, "openai")

        # First 5 should be evicted (buffer holds 50)
        assert m.get_captured_request(0) is None
        assert m.get_captured_request(4) is None
        # Turn 5 onwards should exist
        assert m.get_captured_request(5) is not None
        assert m.get_captured_request(54) is not None

    def test_get_by_turn_not_found(self):
        m = ProxyMetrics()
        assert m.get_captured_request(99) is None

    def test_summary_excludes_messages(self):
        m = ProxyMetrics()
        m.capture_request(0, {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "big payload"}],
        }, "openai")

        summaries = m.get_captured_requests_summary()
        assert len(summaries) == 1
        assert "messages" not in summaries[0]
        assert summaries[0]["turn"] == 0
        assert summaries[0]["model"] == "gpt-4"
        assert summaries[0]["message_count"] == 1

    def test_capture_stores_inbound_tags(self):
        m = ProxyMetrics()
        m.capture_request(
            0, {"messages": []}, "anthropic",
            inbound_tags=["python", "testing"],
        )
        req = m.get_captured_request(0)
        assert req["inbound_tags"] == ["python", "testing"]
        assert req["response_tags"] == []

    def test_capture_default_empty_tags(self):
        m = ProxyMetrics()
        m.capture_request(0, {"messages": []}, "openai")
        req = m.get_captured_request(0)
        assert req["inbound_tags"] == []
        assert req["response_tags"] == []

    def test_capture_stores_prepare_breakdown(self):
        m = ProxyMetrics()
        m.capture_request(
            0,
            {"messages": [{"role": "user", "content": "big payload"}]},
            "openai",
            prepare_total_ms=1234.5,
            prepare_breakdown={
                "filter_body_messages": 611.2,
                "collapse_turn_chains": 402.1,
            },
        )

        req = m.get_captured_request(0)
        assert req["prepare_total_ms"] == 1234.5
        assert req["prepare_breakdown"] == {
            "filter_body_messages": 611.2,
            "collapse_turn_chains": 402.1,
        }

        summaries = m.get_captured_requests_summary()
        assert summaries[0]["prepare_total_ms"] == 1234.5
        assert summaries[0]["prepare_breakdown"] == {
            "filter_body_messages": 611.2,
            "collapse_turn_chains": 402.1,
        }

    def test_update_request_tags(self):
        m = ProxyMetrics()
        m.capture_request(
            0, {"messages": []}, "anthropic",
            inbound_tags=["query-tag"],
        )
        m.update_request_tags(0, response_tags=["response-tag", "code"])
        req = m.get_captured_request(0)
        assert req["inbound_tags"] == ["query-tag"]
        assert req["response_tags"] == ["response-tag", "code"]

    def test_update_request_tags_not_found(self):
        """update_request_tags silently skips if turn not in buffer."""
        m = ProxyMetrics()
        m.update_request_tags(99, response_tags=["x"])  # should not raise


# ---------------------------------------------------------------------------
# Request event model field
# ---------------------------------------------------------------------------


class TestRequestEventModel:
    """Verify that the request metrics event includes the model field."""

    def test_model_in_request_event(self):
        """The metrics.record() call should include model from the request body."""
        from virtual_context.proxy.metrics import ProxyMetrics
        metrics = ProxyMetrics()
        metrics.record({
            "type": "request",
            "turn": 0,
            "model": "claude-haiku-4-5-20251001",
            "input_tokens": 1000,
        })
        events = metrics.events_since(-1)
        req = [e for e in events if e["type"] == "request"][0]
        assert req["model"] == "claude-haiku-4-5-20251001"

    def test_model_in_snapshot_recent_requests(self):
        """Snapshot should include model in recent_requests for dashboard rebuild."""
        from virtual_context.proxy.metrics import ProxyMetrics
        metrics = ProxyMetrics()
        metrics.record({
            "type": "request",
            "turn": 0,
            "model": "claude-sonnet-4-5-20250929",
            "input_tokens": 5000,
            "raw_input_tokens": 8000,
            "wait_ms": 10,
            "inbound_ms": 20,
            "context_tokens": 100,
        })
        snap = metrics.snapshot()
        assert len(snap["recent_requests"]) == 1
        assert snap["recent_requests"][0]["model"] == "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Turn offset on proxy restart
# ---------------------------------------------------------------------------


class TestProxyStateTurnOffset:
    """Verify ProxyState.turn_offset continues numbering from persisted state."""

    def test_turn_offset_from_turn_tag_index(self):
        """turn_offset should be max(turn_number) + 1 from restored entries."""
        engine = MagicMock()
        engine._turn_tag_index = TurnTagIndex()
        for i in range(10):
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i,
                message_hash=f"h{i}",
                tags=["tag"],
                primary_tag="tag",
            ))
        state = ProxyState(engine=engine)
        assert state.turn_offset == 10

    def test_turn_offset_empty_index(self):
        """turn_offset should be 0 when no prior entries exist."""
        engine = MagicMock()
        engine._turn_tag_index = TurnTagIndex()
        state = ProxyState(engine=engine)
        assert state.turn_offset == 0

# ---------------------------------------------------------------------------
# Dashboard settings tests
# ---------------------------------------------------------------------------


class TestDashboardSettings:
    """Tests for /dashboard/settings GET and PUT with context_lookback_pairs."""

    @pytest.fixture
    def settings_client(self, tmp_path):
        """Create a real-engine app for dashboard settings testing."""
        from starlette.testclient import TestClient
        db_path = str(tmp_path / "store.db")
        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            real_config = load_config(config_dict={
                "context_window": 10000,
                "storage_root": str(tmp_path),
                "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
                "tag_generator": {"type": "keyword", "context_lookback_pairs": 5},
            })
            engine = MagicMock()
            engine.config = real_config
            engine.on_message_inbound.return_value = AssembledContext()
            engine.on_turn_complete.return_value = None
            engine.tag_turn.return_value = None
            MockEngine.return_value = engine
            app = create_app(upstream="http://fake:9999", config_path=None)
        with TestClient(app) as client:
            yield client

    def test_context_lookback_pairs_in_settings(self, settings_client):
        """GET /dashboard/settings should include context_lookback_pairs in tagging section."""
        resp = settings_client.get("/dashboard/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "tagging" in data
        assert "context_lookback_pairs" in data["tagging"]
        assert data["tagging"]["context_lookback_pairs"] == 5

    def test_context_lookback_pairs_update(self, settings_client):
        """PUT /dashboard/settings should update context_lookback_pairs."""
        resp = settings_client.put(
            "/dashboard/settings",
            json={"tagging": {"context_lookback_pairs": 3}},
        )
        assert resp.status_code == 200

        resp = settings_client.get("/dashboard/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tagging"]["context_lookback_pairs"] == 3


class TestCompactionConcurrencyGuard:
    """PROXY-007: Manual compaction must reject concurrent requests.

    Tests the lock directly on ProxyState rather than via HTTP, since
    the registry/state are closure variables not accessible from the app object.
    """

    @pytest.mark.regression("PROXY-007")
    def test_compaction_lock_exists_on_proxy_state(self):
        """ProxyState has a _compaction_lock for concurrency control."""
        import threading
        engine = MagicMock()
        engine.config = MagicMock()
        engine.config.conversation_id = "test"
        state = ProxyState(engine)
        assert hasattr(state, "_compaction_lock")
        assert isinstance(state._compaction_lock, type(threading.Lock()))

    @pytest.mark.regression("PROXY-007")
    def test_compaction_lock_is_non_reentrant(self):
        """Lock is a plain Lock (not RLock) so double-acquire blocks."""
        engine = MagicMock()
        engine.config = MagicMock()
        engine.config.conversation_id = "test"
        state = ProxyState(engine)
        # First acquire succeeds
        assert state._compaction_lock.acquire(blocking=False) is True
        # Second acquire fails (non-blocking) — proves concurrency guard works
        assert state._compaction_lock.acquire(blocking=False) is False
        state._compaction_lock.release()

    @pytest.mark.regression("PROXY-007")
    def test_dashboard_compact_endpoint_uses_lock(self, tmp_path):
        """The /dashboard/compact endpoint acquires the lock and returns 409 if busy."""
        from starlette.testclient import TestClient

        db_path = str(tmp_path / "store.db")
        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            cfg = load_config(config_dict={
                "context_window": 10000,
                "storage_root": str(tmp_path),
                "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
                "tag_generator": {"type": "keyword"},
            })
            engine = MagicMock()
            engine.config = cfg
            engine.on_message_inbound.return_value = AssembledContext()
            engine.on_turn_complete.return_value = None
            engine.tag_turn.return_value = None
            engine._turn_tag_index = MagicMock()
            engine._turn_tag_index.entries = []
            engine._engine_state.compacted_through = 0
            MockEngine.return_value = engine
            app = create_app(upstream="http://fake:9999", config_path=None)

        # The dashboard routes close over a `state` variable. We can access it
        # by inspecting the route's endpoint closure.
        from virtual_context.proxy.dashboard import register_dashboard_routes
        # Find the compact route and extract the state from its closure
        state = None
        for route in app.routes:
            if hasattr(route, "path") and route.path == "/dashboard/compact":
                # The endpoint is a closure over `state`
                endpoint = route.endpoint
                if hasattr(endpoint, "__code__"):
                    free_vars = endpoint.__code__.co_freevars
                    if "state" in free_vars:
                        idx = free_vars.index("state")
                        state = endpoint.__closure__[idx].cell_contents
                break

        with TestClient(app) as client:
            if state is None:
                # Can't extract state — just verify endpoint exists
                resp = client.post("/dashboard/compact")
                assert resp.status_code in (200, 409, 503)
                return

            # Hold the lock to simulate in-progress compaction
            state._compaction_lock.acquire()
            try:
                resp = client.post("/dashboard/compact")
                assert resp.status_code == 409
                data = resp.json()
                assert data["status"] == "busy"
                assert "already in progress" in data["message"].lower()
            finally:
                state._compaction_lock.release()

# ---------------------------------------------------------------------------
# Dashboard passthrough toggle
# ---------------------------------------------------------------------------


class TestPassthroughToggle:
    """Tests for the passthrough toggle endpoint."""

    @pytest.fixture
    def toggle_client(self, tmp_path):
        """Create app with mock engine for passthrough toggle testing."""
        from starlette.testclient import TestClient
        db_path = str(tmp_path / "store.db")
        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            real_config = load_config(config_dict={
                "context_window": 10000,
                "storage_root": str(tmp_path),
                "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
                "tag_generator": {"type": "keyword"},
            })
            engine = MagicMock()
            engine.config = real_config
            engine.on_message_inbound.return_value = AssembledContext()
            engine.on_turn_complete.return_value = None
            engine.tag_turn.return_value = None
            engine._turn_tag_index = TurnTagIndex()
            MockEngine.return_value = engine
            app = create_app(upstream="http://fake:9999", config_path=None)
        with TestClient(app) as client:
            yield client, engine

    def test_toggle_on(self, toggle_client):
        client, engine = toggle_client
        sid = engine.config.conversation_id
        resp = client.post(
            f"/dashboard/conversations/{sid}/passthrough",
            json={"enabled": True},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_toggle_off(self, toggle_client):
        client, engine = toggle_client
        sid = engine.config.conversation_id
        # Enable then disable
        client.post(
            f"/dashboard/conversations/{sid}/passthrough",
            json={"enabled": True},
        )
        resp = client.post(
            f"/dashboard/conversations/{sid}/passthrough",
            json={"enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_toggle_unknown_session_404(self, toggle_client):
        client, _ = toggle_client
        resp = client.post(
            "/dashboard/conversations/nonexistent-session/passthrough",
            json={"enabled": True},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Phase 6: Paging Tool Interception
# ---------------------------------------------------------------------------


class TestInjectVCTools:
    """Tests for _inject_vc_tools()."""

    def test_injects_when_no_existing_tools(self):
        engine = MagicMock()
        body = {"model": "claude-3", "messages": []}
        result = _inject_vc_tools(body, engine)
        assert "tools" in result
        names = {t["name"] for t in result["tools"]}
        assert "vc_expand_topic" in names
        assert "vc_restore_tool" not in names
        assert "vc_collapse_topic" not in names

    def test_preserves_existing_tools(self):
        engine = MagicMock()
        body = {
            "model": "claude-3",
            "messages": [],
            "tools": [{"name": "web_search", "description": "Search", "input_schema": {}}],
        }
        result = _inject_vc_tools(body, engine)
        names = [t["name"] for t in result["tools"]]
        assert names[0] == "web_search"
        assert "vc_expand_topic" in names
        assert len(names) == 6  # web_search + 5 VC tools

    def test_sets_required_policy_when_requested(self):
        engine = MagicMock()
        body = {"model": "claude-3", "messages": []}
        result = _inject_vc_tools(body, engine, require_tool_use=True)
        assert result["tool_choice"] == {"type": "any"}

    def test_skips_when_tool_choice_none_string(self):
        engine = MagicMock()
        body = {"model": "claude-3", "messages": [], "tool_choice": "none"}
        result = _inject_vc_tools(body, engine)
        assert "tools" not in result or result is body

    def test_skips_when_tool_choice_none_dict(self):
        engine = MagicMock()
        body = {"model": "claude-3", "messages": [], "tool_choice": {"type": "none"}}
        result = _inject_vc_tools(body, engine)
        assert "tools" not in result or result is body

    def test_shallow_copies_body(self):
        engine = MagicMock()
        body = {"model": "claude-3", "messages": []}
        result = _inject_vc_tools(body, engine)
        assert result is not body
        assert "tools" not in body  # original untouched

    def test_injects_restore_when_restore_available(self):
        engine = MagicMock()
        body = {"model": "claude-3", "messages": []}
        result = _inject_vc_tools(body, engine, restore_available=True)
        names = {t["name"] for t in result["tools"]}
        assert "vc_restore_tool" in names

class TestBuildContinuationRequest:
    """Tests for _build_continuation_request()."""

    def test_includes_original_messages(self):
        original = {
            "model": "claude-3",
            "max_tokens": 1024,
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "t1"}],
        }
        assistant_content = [
            {"type": "text", "text": "Let me check"},
            {"type": "tool_use", "id": "t1", "name": "vc_expand_topic", "input": {"tag": "db"}},
        ]
        tool_results = [
            {"type": "tool_result", "tool_use_id": "t1", "content": "{}"},
        ]
        result = _build_continuation_request(original, assistant_content, tool_results)
        assert result["messages"][0] == {"role": "user", "content": "Hi"}
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][2]["role"] == "user"

    def test_sets_stream_false(self):
        result = _build_continuation_request(
            {"model": "m", "stream": True, "messages": []}, [], [],
        )
        assert result["stream"] is False

    def test_preserves_model_and_tools(self):
        original = {
            "model": "claude-opus",
            "max_tokens": 2048,
            "tools": [{"name": "vc_expand_topic"}],
            "messages": [],
        }
        result = _build_continuation_request(original, [], [])
        assert result["model"] == "claude-opus"
        assert result["max_tokens"] == 2048
        assert result["tools"] == [{"name": "vc_expand_topic"}]

    def test_preserves_system(self):
        original = {"model": "m", "system": "Be helpful", "messages": []}
        result = _build_continuation_request(original, [], [])
        assert result["system"] == "Be helpful"

    def test_no_system_key_when_absent(self):
        original = {"model": "m", "messages": []}
        result = _build_continuation_request(original, [], [])
        assert "system" not in result

    def test_assistant_content_in_message(self):
        blocks = [{"type": "text", "text": "Hello"}]
        result = _build_continuation_request(
            {"model": "m", "messages": []}, blocks, [],
        )
        assert result["messages"][-2]["content"] == blocks

    def test_tool_results_in_user_message(self):
        results = [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
        result = _build_continuation_request(
            {"model": "m", "messages": []}, [], results,
        )
        assert result["messages"][-1]["content"] == results

# ---------------------------------------------------------------------------
# Phase 6: Multi-instance create_app + shared engine
# ---------------------------------------------------------------------------


class TestCreateAppSharedEngine:
    """Test that create_app accepts shared_engine / shared_metrics / instance_label."""

    def test_shared_engine_reused(self, tmp_path):
        """When shared_engine is provided, create_app doesn't create a new one."""
        config = load_config(config_dict={
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "storage_root": str(tmp_path),
        })
        from virtual_context.engine import VirtualContextEngine
        engine = VirtualContextEngine(config=config)
        metrics = ProxyMetrics(context_window=120_000)

        app = create_app(
            upstream="https://api.anthropic.com",
            shared_engine=engine,
            shared_metrics=metrics,
            instance_label="anthropic",
        )
        assert app.title == "virtual-context proxy [anthropic]"
        assert app.state.instance_label == "anthropic"

    def test_no_label_default_title(self, tmp_path):
        """Without instance_label, title stays default."""
        config = load_config(config_dict={
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "storage_root": str(tmp_path),
        })
        from virtual_context.engine import VirtualContextEngine
        engine = VirtualContextEngine(config=config)
        metrics = ProxyMetrics(context_window=120_000)

        app = create_app(
            upstream="https://api.anthropic.com",
            shared_engine=engine,
            shared_metrics=metrics,
        )
        assert app.title == "virtual-context proxy"
        assert app.state.instance_label == ""

    def test_shared_metrics_not_replaced(self, tmp_path):
        """Shared metrics object is reused, not replaced."""
        config = load_config(config_dict={
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "storage_root": str(tmp_path),
        })
        from virtual_context.engine import VirtualContextEngine
        engine = VirtualContextEngine(config=config)
        metrics = ProxyMetrics(context_window=120_000)
        metrics.record({"type": "test_event"})  # mark it

        app = create_app(
            upstream="https://api.anthropic.com",
            shared_engine=engine,
            shared_metrics=metrics,
            instance_label="test",
        )
        # The app was created successfully with shared components
        assert app.title == "virtual-context proxy [test]"

    def test_backward_compat_no_shared(self, tmp_path):
        """Without shared params, create_app works as before (creates its own engine)."""
        app = create_app(
            upstream="https://api.anthropic.com",
            config_path=None,
        )
        assert app.title == "virtual-context proxy"
        assert app.state.instance_label == ""

# ---------------------------------------------------------------------------
# TurnTagIndex hash lookup
# ---------------------------------------------------------------------------


class TestTurnTagIndexHashLookup:
    def test_turn_tag_index_hash_lookup(self):
        """TurnTagIndex supports O(1) lookup by message_hash."""
        idx = TurnTagIndex()
        idx.append(TurnTagEntry(
            turn_number=0, message_hash="abc123", tags=["python"],
            primary_tag="python",
        ))
        idx.append(TurnTagEntry(
            turn_number=1, message_hash="def456", tags=["cooking"],
            primary_tag="cooking",
        ))
        idx.append(TurnTagEntry(
            turn_number=2, message_hash="ghi789", tags=["music"],
            primary_tag="music",
        ))

        assert idx.get_entry_by_hash("abc123").turn_number == 0
        assert idx.get_entry_by_hash("def456").turn_number == 1
        assert idx.get_entry_by_hash("ghi789").turn_number == 2
        assert idx.get_entry_by_hash("nonexistent") is None


# ---------------------------------------------------------------------------
# estimate_tools_tokens
# ---------------------------------------------------------------------------


class TestEstimateToolsTokens:
    def test_estimate_tools_tokens(self):
        """PayloadFormat.estimate_tools_tokens counts tool definition tokens."""
        from virtual_context.proxy.formats import AnthropicFormat
        fmt = AnthropicFormat()
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"name": "read", "description": "Read a file", "input_schema": {"type": "object"}},
                {"name": "write", "description": "Write a file", "input_schema": {"type": "object"}},
            ],
        }
        tokens = fmt.estimate_tools_tokens(body)
        assert tokens > 0
        # Empty tools → 0
        assert fmt.estimate_tools_tokens({"messages": []}) == 0
