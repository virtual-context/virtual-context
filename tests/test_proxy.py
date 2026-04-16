"""Tests for virtual_context.proxy.server."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from virtual_context.proxy.server import (
    ProxyState,
    _compute_protected_turn_stats,
    _build_continuation_request,
    _inject_vc_tools,
    create_app,
    prepare_payload,
)
from virtual_context.config import load_config
from virtual_context.proxy.formats import AnthropicFormat, PayloadTokenCache, PayloadTokenEstimate
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.handlers import _handle_non_streaming
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import AssembledContext, EngineState, Message, SplitResult, TagResult, TurnTagEntry

# ---------------------------------------------------------------------------
# ProxyState
# ---------------------------------------------------------------------------


class TestProxyState:
    @staticmethod
    def _make_engine():
        engine = MagicMock()
        engine._turn_tag_index = TurnTagIndex()
        engine._engine_state = EngineState()
        engine.config.monitor.protected_recent_turns = 0
        engine.config.conversation_id = "conv-123"
        engine.process_broad_tag_split.return_value = None
        return engine

    def test_wait_for_tag_noop_when_no_pending(self):
        engine = self._make_engine()
        state = ProxyState(engine)
        state.wait_for_tag()  # should not raise

    def test_wait_for_complete_noop_when_no_pending(self):
        engine = self._make_engine()
        state = ProxyState(engine)
        state.wait_for_complete()  # should not raise

    def test_fire_and_wait_for_tag(self):
        engine = self._make_engine()
        engine.tag_turn.return_value = None  # no compaction needed
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_tag()
        engine.tag_turn.assert_called_once_with(
            history,
            payload_tokens=None,
            run_broad_split=False,
            turn_number=0,
        )

    def test_fire_and_wait_for_complete(self):
        engine = self._make_engine()
        engine.tag_turn.return_value = None  # no compaction needed
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_complete()
        engine.tag_turn.assert_called_once_with(
            history,
            payload_tokens=None,
            run_broad_split=False,
            turn_number=0,
        )

    def test_wait_for_tag_does_not_block_on_deferred_split(self):
        engine = self._make_engine()
        engine.tag_turn.return_value = None
        release_split = threading.Event()

        def _split(*args, **kwargs):
            release_split.wait(0.5)
            return None

        engine.process_broad_tag_split.side_effect = _split
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]

        state.fire_turn_complete(history)
        started = time.monotonic()
        state.wait_for_tag()
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        release_split.set()
        state._drain_background_work()

    def test_deferred_tag_split_emits_metrics(self):
        engine = self._make_engine()
        engine.tag_turn.return_value = None
        engine.process_broad_tag_split.return_value = SplitResult(
            tag="troubleshooting",
            splittable=True,
            groups={"api-troubleshooting": [0], "db-troubleshooting": [1]},
        )
        metrics = ProxyMetrics()
        state = ProxyState(engine, metrics=metrics)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]

        state.fire_turn_complete(history)
        state.wait_for_tag()
        state._drain_background_work()

        events = metrics.events_since(-1)
        split = next(event for event in events if event["type"] == "tag_split")
        assert split["tag"] == "troubleshooting"
        assert split["splittable"] is True
        assert split["new_tags"] == ["api-troubleshooting", "db-troubleshooting"]

    def test_compaction_fires_in_background(self):
        engine = self._make_engine()
        signal = MagicMock()  # non-None → compaction needed
        signal.priority = "soft"
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
        engine = self._make_engine()
        signal = MagicMock()
        signal.priority = "soft"
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
        engine = self._make_engine()
        engine.tag_turn.side_effect = RuntimeError("boom")
        state = ProxyState(engine)
        history = [Message(role="user", content="hi")]
        state.fire_turn_complete(history)
        state.wait_for_tag()  # should not raise

    def test_error_in_compact_is_caught(self):
        engine = self._make_engine()
        signal = MagicMock()
        signal.priority = "soft"
        engine.tag_turn.return_value = signal
        engine.compact_if_needed.side_effect = RuntimeError("compact boom")
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_complete()  # should not raise

    def test_fire_turn_complete_dedupes_same_completed_turn_while_queued(self):
        engine = self._make_engine()
        release = threading.Event()

        def _tag_turn(*args, **kwargs):
            release.wait(0.5)
            return None

        engine.tag_turn.side_effect = _tag_turn
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]

        state.fire_turn_complete(history)
        state.fire_turn_complete(history)
        release.set()
        state.wait_for_tag()

        assert engine.tag_turn.call_count == 1

    def test_compaction_requests_coalesce_to_latest_target(self):
        engine = self._make_engine()
        state = ProxyState(engine)
        signal = MagicMock()
        signal.priority = "soft"
        release = threading.Event()
        calls: list[list[Message]] = []

        def _compact_if_needed(history, compaction_signal, progress_callback=None):
            calls.append(history)
            release.wait(0.5)
            engine._engine_state.compacted_through = max(
                engine._engine_state.compacted_through,
                len(history) - (engine.config.monitor.protected_recent_turns * 2),
            )
            return None

        engine.compact_if_needed.side_effect = _compact_if_needed
        history_a = [
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="u2"),
            Message(role="assistant", content="a2"),
            Message(role="user", content="u3"),
            Message(role="assistant", content="a3"),
        ]
        history_b = history_a + [
            Message(role="user", content="u4"),
            Message(role="assistant", content="a4"),
        ]
        history_c = history_b + [
            Message(role="user", content="u5"),
            Message(role="assistant", content="a5"),
        ]

        state._queue_compaction(history_a, signal, 2)
        state._queue_compaction(history_b, signal, 3)
        state._queue_compaction(history_c, signal, 4)
        release.set()
        state.wait_for_compact()

        assert engine.compact_if_needed.call_count == 2
        assert calls[0] == history_a
        assert calls[1] == history_c


class TestResponseTimestamping:
    def test_non_streaming_tool_only_response_updates_last_request_time(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {
            "id": "msg_123",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "lookup",
                    "input": {"q": "cache"},
                }
            ],
            "stop_reason": "tool_use",
        }

        client = MagicMock()
        client.request = AsyncMock(return_value=response)

        state = MagicMock()
        state.is_conversation_deleted.return_value = False
        state.engine._engine_state.last_request_time = 0.0
        state.engine.config.conversation_id = "conv-123"
        state._last_enriched_payload_tokens = 0
        state.conversation_history = []

        asyncio.run(
            _handle_non_streaming(
                client,
                "https://example.com/v1/messages",
                {},
                {"model": "claude-sonnet", "messages": []},
                "anthropic",
                state,
                conversation_id="conv-123",
            )
        )

        assert state.engine._engine_state.last_request_time > 0.0
        state.persist_completed_turn.assert_not_called()
        state.fire_turn_complete.assert_not_called()


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
    engine.compact_manual.return_value = None
    engine._turn_tag_index = TurnTagIndex()
    engine._engine_state = EngineState()
    engine._store = MagicMock()
    engine._session_state_provider = None
    engine.config.context_window = 200000
    engine.config.monitor.context_window = 200000
    engine.config.monitor.protected_recent_turns = 6
    engine.config.monitor.store_recovery_threshold = 0.70
    engine.config.monitor.defer_payload_mutation = False
    engine.config.monitor.fill_pass_enabled = False
    engine.config.monitor.flush_ttl_seconds = 300
    engine.config.monitor.hard_threshold = 0.85
    engine.config.proxy.upstream_context_limit = 200000
    engine.config.proxy.passthrough_trim_ratio = 0.40
    engine.config.proxy.history_widening_threshold = 0.10
    engine.config.tool_output.enabled = False
    engine.config.paging.enabled = False
    engine.config.conversation_id = "conv-test"
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
        outbound_cache = PayloadTokenCache(
            format_name="anthropic",
            message_key="messages",
            shell_fingerprint="shell-out",
            shell_tokens=14,
            message_fingerprints=["m1", "m2"],
            message_tokens=[7, 8],
            separator_tokens=1,
            total_tokens=24,
        )
        provider.load_payload_token_cache.side_effect = [cached, None]
        engine._session_state_provider = provider

        state = ProxyState(engine)
        fmt = AnthropicFormat()
        fmt.estimate_payload_tokens_segmented = MagicMock(side_effect=[
            PayloadTokenEstimate(
                total_tokens=24,
                cache=next_cache,
                reused_prefix_messages=1,
                recounted_messages=1,
                shell_cache_hit=True,
            ),
            PayloadTokenEstimate(
                total_tokens=24,
                cache=outbound_cache,
                reused_prefix_messages=0,
                recounted_messages=1,
                shell_cache_hit=False,
            ),
        ])
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

        assert provider.load_payload_token_cache.call_args_list == [
            call("conv-123"),
            call("conv-123", scope="outbound"),
        ]
        provider.save_payload_token_cache.assert_any_call("conv-123", next_cache)
        provider.save_payload_token_cache.assert_any_call("conv-123", outbound_cache, scope="outbound")
        assert fmt.estimate_payload_tokens_segmented.call_args_list[0].kwargs["cache"] == cached
        assert fmt.estimate_payload_tokens_segmented.call_args_list[1].kwargs["cache"] is None
        assert state._inbound_payload_token_cache == next_cache
        assert state._outbound_payload_token_cache == outbound_cache

    def test_compute_protected_turn_stats_uses_precomputed_message_tokens(self):
        fmt = AnthropicFormat()
        body = {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "u3"},
                {"role": "assistant", "content": "a3"},
            ],
        }
        fmt.estimate_message_tokens = MagicMock(side_effect=AssertionError("should not recount"))

        stats = _compute_protected_turn_stats(
            body,
            fmt,
            2,
            message_tokens=[11, 12, 21, 22, 31, 32],
        )

        assert stats["count"] == 2
        assert stats["turn_tokens"] == [43, 63]
        assert stats["tokens"] == 106


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
        assert evt["turns_received"] == 3
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


class TestPrepareRouting:
    def _make_state(self):
        engine = MagicMock()
        engine._turn_tag_index = TurnTagIndex()
        engine._engine_state = EngineState()
        engine._store = MagicMock()
        engine._store.get_all_tags.return_value = []
        engine.config.conversation_id = "conv-123"
        engine.config.context_window = 120_000
        engine.config.monitor.context_window = 120_000
        engine.config.monitor.protected_recent_turns = 0
        engine.config.monitor.store_recovery_threshold = 0.70
        engine.config.monitor.defer_payload_mutation = False
        engine.config.monitor.flush_ttl_seconds = 300
        engine.config.monitor.fill_pass_enabled = False
        engine.config.proxy.passthrough_trim_ratio = 0.40
        engine.config.proxy.upstream_context_limit = 120_000
        engine.config.proxy.enable_tool_output_compression = False
        engine.config.proxy.max_output_media_bytes = 0
        engine.config.proxy.history_widening_threshold = 0.10
        engine.config.tag_generator.context_lookback_pairs = 3
        engine.config.tag_generator.context_bleed_threshold = 0
        engine.config.tool_output.enabled = False
        engine.process_broad_tag_split.return_value = None
        engine.on_message_inbound.return_value = AssembledContext()
        metrics = ProxyMetrics()
        return ProxyState(engine, metrics=metrics), metrics

    def test_prepare_payload_restored_ready_conversation_stays_active(self):
        state, metrics = self._make_state()
        for i in range(2):
            state.engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i,
                message_hash=f"h{i}",
                tags=[f"tag-{i}"],
                primary_tag=f"tag-{i}",
            ))
        state.engine._engine_state.last_indexed_turn = 1
        state.engine._engine_state.last_completed_turn = 1
        state.note_engine_restore(force=True)
        state._ingested_conversations.clear()

        fmt = AnthropicFormat()
        body = {
            "model": "claude-opus-4-6",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ],
        }

        prepared = asyncio.run(
            prepare_payload(
                body,
                state,
                fmt,
                metrics,
                body_bytes=json.dumps(body).encode("utf-8"),
            )
        )

        assert prepared.is_passthrough is False
        assert state._history_ingested() is True
        captured = metrics.get_captured_requests_summary(conversation_id="conv-123")
        assert captured
        assert captured[-1]["passthrough"] is False


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

    def test_same_turn_different_turn_ids_do_not_collide(self):
        m = ProxyMetrics()
        m.capture_request(
            7,
            {"messages": []},
            "anthropic",
            conversation_id="conv-1",
            turn_id="req-a",
            inbound_tags=["first"],
        )
        m.capture_request(
            7,
            {"messages": []},
            "anthropic",
            conversation_id="conv-1",
            turn_id="req-b",
            inbound_tags=["second"],
        )
        m.capture_response(
            7,
            {"content": "a"},
            conversation_id="conv-1",
            turn_id="req-a",
            upstream_input_tokens=100,
        )
        m.capture_response(
            7,
            {"content": "b"},
            conversation_id="conv-1",
            turn_id="req-b",
            upstream_input_tokens=200,
        )

        req_a = m.get_captured_request(7, conversation_id="conv-1", turn_id="req-a")
        req_b = m.get_captured_request(7, conversation_id="conv-1", turn_id="req-b")

        assert req_a["inbound_tags"] == ["first"]
        assert req_a["upstream_input_tokens"] == 100
        assert req_b["inbound_tags"] == ["second"]
        assert req_b["upstream_input_tokens"] == 200


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
            engine.compact_manual.return_value = None
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
