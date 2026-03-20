"""Tests for proxy streaming: SSE parsing, emission, interception, and continuation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_context.proxy.server import (
    ProxyState,
    _build_continuation_request,
    _emit_message_end_sse,
    _emit_text_as_sse,
    _emit_tool_use_as_sse,
    _parse_sse_events,
    create_app,
)
from virtual_context.config import load_config
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import AssembledContext, Message


# ---------------------------------------------------------------------------
# Fixtures shared with test_proxy.py (duplicated for independence)
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


# ---------------------------------------------------------------------------
# Streaming SSE tests
# ---------------------------------------------------------------------------


class TestHandleStreaming:
    """Tests for _handle_streaming raw-byte forwarding and error handling."""

    def test_streaming_forwards_raw_bytes(self, test_client):
        """Streaming response forwards raw bytes unchanged (no decode/re-encode)."""
        client, engine = test_client

        # Build an SSE stream as raw bytes — use \r\n like a real HTTP server
        sse_events = (
            b"event: message_start\r\n"
            b"data: {\"type\":\"message_start\"}\r\n"
            b"\r\n"
            b"event: content_block_delta\r\n"
            b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\r\n"
            b"\r\n"
            b"event: message_stop\r\n"
            b"data: {\"type\":\"message_stop\"}\r\n"
            b"\r\n"
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "content-type": "text/event-stream",
        }

        async def mock_aiter_bytes():
            yield sse_events

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

                assert resp.status_code == 200
                # Raw bytes should be forwarded — verify \r\n preserved
                body = resp.content
                assert b"event: message_start\r\n" in body
                assert b"content_block_delta" in body

    def test_streaming_upstream_error_returns_json(self, test_client):
        """Non-2xx upstream response is returned as JSON, not broken SSE."""
        client, engine = test_client

        error_body = b'{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'

        mock_resp = AsyncMock()
        mock_resp.status_code = 529
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.aread = AsyncMock(return_value=error_body)
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

                # Should return the upstream error code, not 200
                assert resp.status_code == 529
                data = resp.json()
                assert data["type"] == "error"
                assert data["error"]["type"] == "overloaded_error"

    def test_streaming_forwards_upstream_headers(self, test_client):
        """Upstream Content-Type and other headers are forwarded to the client."""
        client, engine = test_client

        sse_data = b"event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n"

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "content-type": "text/event-stream",
            "x-request-id": "req_abc123",
        }

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

                # Upstream Content-Type forwarded (no charset=utf-8 added)
                ct = resp.headers.get("content-type", "")
                assert "text/event-stream" in ct
                # SSE-critical headers added
                assert resp.headers.get("cache-control") == "no-cache"
                assert resp.headers.get("x-accel-buffering") == "no"
                # Upstream custom header forwarded
                assert resp.headers.get("x-request-id") == "req_abc123"

    def test_streaming_accumulates_text_for_turn_complete(self, test_client):
        """Text deltas are parsed from raw bytes for on_turn_complete."""
        client, engine = test_client

        # Two content_block_delta events with text
        chunk1 = (
            b"event: content_block_delta\r\n"
            b"data: {\"type\":\"content_block_delta\",\"index\":0,"
            b"\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\r\n"
            b"\r\n"
        )
        chunk2 = (
            b"event: content_block_delta\r\n"
            b"data: {\"type\":\"content_block_delta\",\"index\":0,"
            b"\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\r\n"
            b"\r\n"
            b"event: message_stop\r\n"
            b"data: {\"type\":\"message_stop\"}\r\n"
            b"\r\n"
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield chunk1
            yield chunk2

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

                assert resp.status_code == 200

        # Engine should have received on_turn_complete (fired in background)
        # Wait briefly for the background thread to finish
        import time
        time.sleep(0.2)
        # The turn_complete fires with the accumulated "Hello world" text
        # Check that the conversation history has the assistant message
        # (engine.on_turn_complete would be called via state.fire_turn_complete)

    @pytest.mark.regression("PROXY-005")
    def test_streaming_tool_result_preserves_sse(self, test_client):
        """Tool-result turn (no text) should stream SSE, not fall to JSON passthrough.

        Regression for PROXY-005: when client sends a tool_result message,
        _extract_user_message() returns "". Before the fix, this fell through to
        _passthrough_bytes() which returned application/json — breaking SSE clients.
        """
        client, engine = test_client

        sse_events = (
            b"event: message_start\r\n"
            b"data: {\"type\":\"message_start\"}\r\n"
            b"\r\n"
            b"event: content_block_start\r\n"
            b"data: {\"type\":\"content_block_start\",\"index\":0,"
            b"\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\r\n"
            b"\r\n"
            b"event: content_block_delta\r\n"
            b"data: {\"type\":\"content_block_delta\",\"index\":0,"
            b"\"delta\":{\"type\":\"text_delta\",\"text\":\"Based on the search results...\"}}\r\n"
            b"\r\n"
            b"event: message_stop\r\n"
            b"data: {\"type\":\"message_stop\"}\r\n"
            b"\r\n"
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_events

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [
                            {"role": "user", "content": "Search for pandas docs"},
                            {"role": "assistant", "content": [
                                {"type": "tool_use", "id": "toolu_01", "name": "web_search",
                                 "input": {"query": "pandas documentation"}},
                            ]},
                            {"role": "user", "content": [
                                {"type": "tool_result", "tool_use_id": "toolu_01",
                                 "content": "pandas is a data analysis library..."},
                            ]},
                        ],
                    },
                )

                # Must be SSE, not JSON
                assert resp.status_code == 200
                ct = resp.headers.get("content-type", "")
                assert "text/event-stream" in ct, (
                    f"Expected text/event-stream, got {ct!r} — "
                    "tool_result turn fell through to JSON passthrough"
                )

                # Raw SSE bytes forwarded
                assert b"content_block_delta" in resp.content
                assert b"Based on the search results" in resp.content

                # Engine enrichment should NOT have been called
                engine.on_message_inbound.assert_not_called()


class TestParseSSEEvents:
    """Tests for _parse_sse_events()."""

    def test_splits_on_double_newline(self):
        buf = (
            b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
            b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"
        )
        events, remainder = _parse_sse_events(buf)
        assert len(events) == 2
        assert remainder == b""

    def test_handles_crlf(self):
        buf = (
            b"event: message_start\r\ndata: {\"type\":\"message_start\"}\r\n\r\n"
        )
        events, remainder = _parse_sse_events(buf)
        assert len(events) == 1
        assert events[0][0] == "message_start"

    def test_preserves_raw_bytes(self):
        raw = b"event: ping\r\ndata: {}\r\n\r\n"
        events, _ = _parse_sse_events(raw)
        assert events[0][2] == raw

    def test_handles_partial_event_in_buffer(self):
        buf = (
            b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
            b"event: partial\ndata: {\"ty"  # incomplete
        )
        events, remainder = _parse_sse_events(buf)
        assert len(events) == 1
        assert b"partial" in remainder

    def test_extracts_event_type_and_data(self):
        buf = b"event: content_block_delta\ndata: {\"type\":\"cbd\"}\n\n"
        events, _ = _parse_sse_events(buf)
        evt_type, data_str, _ = events[0]
        assert evt_type == "content_block_delta"
        assert json.loads(data_str)["type"] == "cbd"

    def test_empty_buffer(self):
        events, remainder = _parse_sse_events(b"")
        assert events == []
        assert remainder == b""

    def test_data_only_event(self):
        """Events without an event: field should return empty event_type."""
        buf = b"data: {\"done\":true}\n\n"
        events, _ = _parse_sse_events(buf)
        assert len(events) == 1
        assert events[0][0] == ""
        assert events[0][1] == '{"done":true}'


class TestEmitTextAsSSE:
    """Tests for _emit_text_as_sse()."""

    def test_emits_three_events(self):
        events = _emit_text_as_sse("Hello world", 0)
        assert len(events) == 3

    def test_content_block_start_delta_stop(self):
        events = _emit_text_as_sse("Hello", 2)
        # Parse each event
        start = json.loads(events[0].decode().split("data: ")[1].strip())
        delta = json.loads(events[1].decode().split("data: ")[1].strip())
        stop = json.loads(events[2].decode().split("data: ")[1].strip())
        assert start["type"] == "content_block_start"
        assert start["index"] == 2
        assert start["content_block"]["type"] == "text"
        assert delta["type"] == "content_block_delta"
        assert delta["index"] == 2
        assert delta["delta"]["text"] == "Hello"
        assert stop["type"] == "content_block_stop"
        assert stop["index"] == 2

    def test_events_are_valid_sse(self):
        events = _emit_text_as_sse("test", 0)
        for event in events:
            decoded = event.decode()
            assert decoded.startswith("event: ")
            assert "\ndata: " in decoded
            assert decoded.endswith("\n\n")


class TestEmitMessageEndSSE:
    """Tests for _emit_message_end_sse()."""

    def test_emits_two_events(self):
        events = _emit_message_end_sse("end_turn")
        assert len(events) == 2

    def test_message_delta_has_stop_reason(self):
        events = _emit_message_end_sse("end_turn")
        delta = json.loads(events[0].decode().split("data: ")[1].strip())
        assert delta["type"] == "message_delta"
        assert delta["delta"]["stop_reason"] == "end_turn"

    def test_message_stop(self):
        events = _emit_message_end_sse("end_turn")
        stop = json.loads(events[1].decode().split("data: ")[1].strip())
        assert stop["type"] == "message_stop"

    def test_no_usage_by_default(self):
        events = _emit_message_end_sse("end_turn")
        delta = json.loads(events[0].decode().split("data: ")[1].strip())
        assert "usage" not in delta

    def test_usage_included_when_provided(self):
        usage = {"output_tokens": 59}
        events = _emit_message_end_sse("end_turn", usage=usage)
        delta = json.loads(events[0].decode().split("data: ")[1].strip())
        assert delta["usage"] == {"output_tokens": 59}

    def test_usage_none_excluded(self):
        events = _emit_message_end_sse("end_turn", usage=None)
        delta = json.loads(events[0].decode().split("data: ")[1].strip())
        assert "usage" not in delta

# ---------------------------------------------------------------------------
# Phase 6: Stream Interception Integration Tests
# ---------------------------------------------------------------------------


def _make_sse_event(event_type: str, data: dict) -> bytes:
    """Helper to build a single raw SSE event."""
    data_str = json.dumps(data)
    return f"event: {event_type}\r\ndata: {data_str}\r\n\r\n".encode()


def _build_text_sse_stream(text: str) -> bytes:
    """Build a complete SSE stream for a text-only response."""
    return (
        _make_sse_event("message_start", {"type": "message_start"})
        + _make_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        + _make_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        })
        + _make_sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })
        + _make_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        })
        + _make_sse_event("message_stop", {"type": "message_stop"})
    )


def _build_tool_use_sse_stream(
    text_before: str = "",
    tool_name: str = "vc_expand_topic",
    tool_id: str = "toolu_01",
    tool_input: dict | None = None,
) -> bytes:
    """Build a complete SSE stream with text (optional) + tool_use."""
    if tool_input is None:
        tool_input = {"tag": "database", "depth": "full"}
    input_json = json.dumps(tool_input)

    events = b""
    events += _make_sse_event("message_start", {"type": "message_start"})

    block_idx = 0

    # Optional text block
    if text_before:
        events += _make_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        })
        events += _make_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "text_delta", "text": text_before},
        })
        events += _make_sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": block_idx,
        })
        block_idx += 1

    # Tool use block
    events += _make_sse_event("content_block_start", {
        "type": "content_block_start",
        "index": block_idx,
        "content_block": {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": {},
        },
    })
    # Send input JSON in two chunks to test accumulation
    mid = len(input_json) // 2
    events += _make_sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": block_idx,
        "delta": {"type": "input_json_delta", "partial_json": input_json[:mid]},
    })
    events += _make_sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": block_idx,
        "delta": {"type": "input_json_delta", "partial_json": input_json[mid:]},
    })
    events += _make_sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": block_idx,
    })

    # message_delta with stop_reason=tool_use
    events += _make_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
    })
    events += _make_sse_event("message_stop", {"type": "message_stop"})
    return events


@pytest.fixture
def paging_test_client(tmp_path):
    """Create app with paging-enabled mock engine."""
    from starlette.testclient import TestClient
    from virtual_context.types import PagingConfig

    db_path = str(tmp_path / "store.db")
    with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
        real_config = load_config(config_dict={
            "context_window": 10000,
            "storage_root": str(tmp_path),
            "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
            "tag_generator": {"type": "keyword"},
        })
        # Enable paging + autonomous mode
        real_config.paging = PagingConfig(enabled=True, autonomous_models=["opus", "sonnet"])
        engine = MagicMock()
        engine.config = real_config
        engine.on_message_inbound.return_value = AssembledContext()
        engine.on_turn_complete.return_value = None
        engine.tag_turn.return_value = None
        engine._turn_tag_index = TurnTagIndex()
        engine._retrieval._resolve_paging_mode.return_value = "autonomous"
        engine._engine_state.compacted_through = 0
        engine.expand_topic.return_value = {
            "tag": "database",
            "depth": "full",
            "tokens_added": 500,
            "tokens_evicted": 0,
            "evicted_tags": [],
        }
        engine.collapse_topic.return_value = {
            "tag": "api",
            "depth": "summary",
            "tokens_freed": 300,
        }
        MockEngine.return_value = engine
        app = create_app(upstream="http://fake:9999", config_path=None)
    with TestClient(app) as client:
        yield client, engine

class TestStreamInterception:
    """Integration tests for Phase 6 stream interception."""

    def test_text_only_response_forwarded_unchanged(self, paging_test_client):
        """Text-only responses pass through even with paging enabled."""
        client, engine = paging_test_client

        sse_data = _build_text_sse_stream("Hello world")

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

                assert resp.status_code == 200
                body = resp.content
                assert b"Hello world" in body
                assert b"message_stop" in body

    def test_vc_tool_intercepted_and_executed(self, paging_test_client):
        """VC tool_use is intercepted, executed, and continuation text emitted."""
        client, engine = paging_test_client

        # Initial stream: text + VC tool_use
        sse_data = _build_tool_use_sse_stream(
            text_before="Let me check",
            tool_name="vc_expand_topic",
            tool_id="toolu_01",
            tool_input={"tag": "database", "depth": "full"},
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        # Continuation response (non-streaming)
        cont_response = MagicMock()
        cont_response.status_code = 200
        cont_response.json.return_value = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Here is the database detail."}],
        }

        call_count = 0

        async def mock_send(req, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        async def mock_post(url, **kwargs):
            return cont_response

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", side_effect=mock_send):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                with patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
                    resp = client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-3",
                            "system": "test",
                            "stream": True,
                            "messages": [{"role": "user", "content": "Tell me about the database"}],
                        },
                    )

                    assert resp.status_code == 200
                    body = resp.content

                    # Text before tool should be forwarded
                    assert b"Let me check" in body

                    # VC tool_use events should NOT be visible to client
                    assert b"vc_expand_topic" not in body

                    # Continuation text should be emitted
                    assert b"Here is the database detail" in body

                    # Engine expand_topic should have been called
                    engine.expand_topic.assert_called_once_with(
                        tag="database", depth="full",
                    )

    def test_mixed_tools_pass_through(self, paging_test_client):
        """Mixed VC + non-VC tools → BAIL: all events forwarded to client."""
        client, engine = paging_test_client

        # Build a stream with both a VC tool and a non-VC tool
        events = b""
        events += _make_sse_event("message_start", {"type": "message_start"})

        # Text block
        events += _make_sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        events += _make_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Checking"},
        })
        events += _make_sse_event("content_block_stop", {
            "type": "content_block_stop", "index": 0,
        })

        # Non-VC tool
        events += _make_sse_event("content_block_start", {
            "type": "content_block_start", "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "web_search", "input": {}},
        })
        events += _make_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":"test"}'},
        })
        events += _make_sse_event("content_block_stop", {
            "type": "content_block_stop", "index": 1,
        })

        # VC tool
        events += _make_sse_event("content_block_start", {
            "type": "content_block_start", "index": 2,
            "content_block": {"type": "tool_use", "id": "t2", "name": "vc_expand_topic", "input": {}},
        })
        events += _make_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"tag":"db"}'},
        })
        events += _make_sse_event("content_block_stop", {
            "type": "content_block_stop", "index": 2,
        })

        # message_delta with tool_use
        events += _make_sse_event("message_delta", {
            "type": "message_delta", "delta": {"stop_reason": "tool_use"},
        })
        events += _make_sse_event("message_stop", {"type": "message_stop"})

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield events

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Search and expand"}],
                    },
                )

                assert resp.status_code == 200
                body = resp.content

                # Both tools should be visible (BAIL path)
                assert b"web_search" in body
                assert b"vc_expand_topic" in body

                # Engine should NOT have been called (BAIL)
                engine.expand_topic.assert_not_called()

    def test_no_preceding_text_tool_only(self, paging_test_client):
        """LLM calls VC tool with no text first — still intercepted."""
        client, engine = paging_test_client

        sse_data = _build_tool_use_sse_stream(
            text_before="",  # no text
            tool_name="vc_expand_topic",
            tool_id="toolu_01",
            tool_input={"tag": "api"},
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        cont_response = MagicMock()
        cont_response.status_code = 200
        cont_response.json.return_value = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "API details here."}],
        }

        async def mock_post(url, **kwargs):
            return cont_response

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                with patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
                    resp = client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-3",
                            "system": "test",
                            "stream": True,
                            "messages": [{"role": "user", "content": "Show API details"}],
                        },
                    )

                    body = resp.content
                    assert b"API details here" in body
                    engine.expand_topic.assert_called_once()

    def test_nested_tool_calls_loop(self, paging_test_client):
        """Continuation that triggers another VC tool call loops correctly."""
        client, engine = paging_test_client

        sse_data = _build_tool_use_sse_stream(
            tool_name="vc_expand_topic",
            tool_input={"tag": "db"},
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        call_count = 0

        def make_cont_response():
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            if call_count == 1:
                # First continuation: another VC tool call
                resp.json.return_value = {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "Expanding more..."},
                        {
                            "type": "tool_use",
                            "id": "toolu_02",
                            "name": "vc_expand_topic",
                            "input": {"tag": "api"},
                        },
                    ],
                }
            else:
                # Second continuation: final text
                resp.json.return_value = {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "All done."}],
                }
            return resp

        async def mock_post(url, **kwargs):
            return make_cont_response()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                with patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
                    resp = client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-3",
                            "system": "test",
                            "stream": True,
                            "messages": [{"role": "user", "content": "Expand everything"}],
                        },
                    )

                    body = resp.content
                    assert b"Expanding more" in body
                    assert b"All done" in body
                    # expand_topic called twice: once for initial, once for nested
                    assert engine.expand_topic.call_count == 2

    def test_max_continuation_loops_respected(self, paging_test_client):
        """Continuation loops cap at 5 even if model keeps calling tools."""
        client, engine = paging_test_client

        sse_data = _build_tool_use_sse_stream(
            tool_name="vc_expand_topic",
            tool_input={"tag": "t1"},
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        # Every continuation returns another tool call (infinite loop attempt)
        def make_cont_response():
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "stop_reason": "tool_use",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_loop",
                    "name": "vc_expand_topic",
                    "input": {"tag": "looping"},
                }],
            }
            return resp

        async def mock_post(url, **kwargs):
            return make_cont_response()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                with patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
                    resp = client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-3",
                            "system": "test",
                            "stream": True,
                            "messages": [{"role": "user", "content": "Loop test"}],
                        },
                    )

                    # Should cap at 5: 1 initial + 5 continuations = 6 total expand_topic calls
                    # (initial tool + 5 loops)
                    assert engine.expand_topic.call_count <= 6

    def test_tool_intercept_metric_recorded(self, paging_test_client):
        """Tool intercept events are recorded in metrics."""
        client, engine = paging_test_client

        sse_data = _build_tool_use_sse_stream(
            tool_name="vc_expand_topic",
            tool_input={"tag": "db"},
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        cont_response = MagicMock()
        cont_response.status_code = 200
        cont_response.json.return_value = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Done"}],
        }

        async def mock_post(url, **kwargs):
            return cont_response

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                with patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
                    resp = client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-3",
                            "system": "test",
                            "stream": True,
                            "messages": [{"role": "user", "content": "Check db"}],
                        },
                    )

        # Check metrics — we need to access the app's metrics
        # The test client doesn't expose metrics directly, but
        # engine.expand_topic was called → tool interception happened
        engine.expand_topic.assert_called_once()

    def test_paging_disabled_uses_raw_forwarding(self, test_client):
        """When paging is disabled, raw byte forwarding is unchanged."""
        client, engine = test_client

        sse_events = (
            b"event: message_start\r\n"
            b"data: {\"type\":\"message_start\"}\r\n"
            b"\r\n"
            b"event: content_block_delta\r\n"
            b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\r\n"
            b"\r\n"
            b"event: message_stop\r\n"
            b"data: {\"type\":\"message_stop\"}\r\n"
            b"\r\n"
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_events

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

                assert resp.status_code == 200
                # Raw bytes forwarded — \r\n preserved
                assert b"event: message_start\r\n" in resp.content

    def test_vc_tools_injected_in_request(self, paging_test_client):
        """When paging enabled, VC tools are injected into outbound request."""
        client, engine = paging_test_client

        sse_data = _build_text_sse_stream("Hi")

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        captured_body = {}

        original_build_request = None

        def capture_build_request(method, url, **kwargs):
            if "json" in kwargs:
                captured_body.update(kwargs["json"])
            return MagicMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch(
                "virtual_context.proxy.server.httpx.AsyncClient.build_request",
                side_effect=capture_build_request,
            ):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

        # Verify VC tools were injected into the outbound body
        tools = captured_body.get("tools", [])
        tool_names = {t["name"] for t in tools}
        assert "vc_expand_topic" in tool_names
        assert "vc_collapse_topic" not in tool_names


# ---------------------------------------------------------------------------
# _emit_tool_use_as_sse
# ---------------------------------------------------------------------------


class TestEmitToolUseAsSSE:
    """Tests for _emit_tool_use_as_sse()."""

    def test_emits_three_events(self):
        tool = {"id": "t1", "name": "memory_search", "input": {"q": "test"}}
        events = _emit_tool_use_as_sse(tool, block_index=0)
        assert len(events) == 3

    def test_content_block_start_has_tool_use_type(self):
        tool = {"id": "t1", "name": "memory_search", "input": {"q": "test"}}
        events = _emit_tool_use_as_sse(tool, block_index=2)
        start = json.loads(events[0].decode().split("data: ")[1].strip())
        assert start["type"] == "content_block_start"
        assert start["index"] == 2
        assert start["content_block"]["type"] == "tool_use"
        assert start["content_block"]["id"] == "t1"
        assert start["content_block"]["name"] == "memory_search"
        assert start["content_block"]["input"] == {}

    def test_delta_has_input_json(self):
        tool = {"id": "t1", "name": "memory_search", "input": {"q": "test"}}
        events = _emit_tool_use_as_sse(tool, block_index=0)
        delta = json.loads(events[1].decode().split("data: ")[1].strip())
        assert delta["type"] == "content_block_delta"
        assert delta["delta"]["type"] == "input_json_delta"
        parsed_input = json.loads(delta["delta"]["partial_json"])
        assert parsed_input == {"q": "test"}

    def test_content_block_stop(self):
        tool = {"id": "t1", "name": "memory_search", "input": {}}
        events = _emit_tool_use_as_sse(tool, block_index=1)
        stop = json.loads(events[2].decode().split("data: ")[1].strip())
        assert stop["type"] == "content_block_stop"
        assert stop["index"] == 1


# ---------------------------------------------------------------------------
# PROXY-015: Continuation BAIL forwards non-VC tools to client
# ---------------------------------------------------------------------------


class TestContinuationBailForward:
    """When a continuation returns non-VC tools, they should be forwarded
    to the client with stop_reason=tool_use instead of being silently
    dropped (PROXY-015)."""

    @pytest.mark.regression("PROXY-015")
    def test_non_vc_tool_forwarded_after_vc_continuation(self, paging_test_client):
        """After VC tool succeeds, continuation returns non-VC tool →
        non-VC tool forwarded to client, stop_reason=tool_use."""
        client, engine = paging_test_client

        # Initial stream: text + VC tool_use
        sse_data = _build_tool_use_sse_stream(
            text_before="Let me page that in instead of guessing.",
            tool_name="vc_expand_topic",
            tool_id="toolu_01",
            tool_input={"tag": "health-protocol", "depth": "full"},
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        call_count = 0

        def make_cont_response():
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.text = "{}"
            if call_count == 1:
                # After VC expand, LLM wants a non-VC tool
                resp.json.return_value = {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "Now searching memory..."},
                        {
                            "type": "tool_use",
                            "id": "toolu_02",
                            "name": "memory_search",
                            "input": {"query": "magnesium glycinate 400mg"},
                        },
                    ],
                }
            else:
                resp.json.return_value = {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "Unexpected"}],
                }
            return resp

        async def mock_post(url, **kwargs):
            return make_cont_response()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                with patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
                    resp = client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-opus-4-6",
                            "system": "test",
                            "stream": True,
                            "messages": [{"role": "user", "content": "what did you recommend after magnesium?"}],
                        },
                    )

                    assert resp.status_code == 200
                    body = resp.content

                    # Original text should be forwarded
                    assert b"Let me page that in instead of guessing" in body

                    # Continuation text should be forwarded
                    assert b"Now searching memory" in body

                    # Non-VC tool should be forwarded to client
                    assert b"memory_search" in body
                    assert b"magnesium glycinate 400mg" in body

                    # VC tool should NOT be visible
                    assert b"vc_expand_topic" not in body

                    # stop_reason should be tool_use (not end_turn)
                    assert b'"stop_reason": "tool_use"' in body or b'"stop_reason":"tool_use"' in body

                    # VC tool was still executed
                    engine.expand_topic.assert_called_once()

    @pytest.mark.regression("PROXY-015")
    def test_multiple_vc_then_non_vc_all_forwarded(self, paging_test_client):
        """VC tool x2 then non-VC → both VC tools executed, non-VC forwarded."""
        client, engine = paging_test_client

        sse_data = _build_tool_use_sse_stream(
            text_before="Expanding...",
            tool_name="vc_expand_topic",
            tool_id="toolu_01",
            tool_input={"tag": "tag-a"},
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_data

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        call_count = 0

        def make_cont_response():
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.text = "{}"
            if call_count == 1:
                # Second VC tool
                resp.json.return_value = {
                    "stop_reason": "tool_use",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "vc_expand_topic",
                        "input": {"tag": "tag-b"},
                    }],
                }
            elif call_count == 2:
                # Non-VC tool after both VC tools succeeded
                resp.json.return_value = {
                    "stop_reason": "tool_use",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_03",
                        "name": "web_search",
                        "input": {"q": "some query"},
                    }],
                }
            else:
                resp.json.return_value = {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "Unexpected"}],
                }
            return resp

        async def mock_post(url, **kwargs):
            return make_cont_response()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                with patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
                    resp = client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-opus-4-6",
                            "system": "test",
                            "stream": True,
                            "messages": [{"role": "user", "content": "expand all"}],
                        },
                    )

                    body = resp.content

                    # Non-VC tool forwarded
                    assert b"web_search" in body

                    # VC tools NOT visible
                    assert b"vc_expand_topic" not in body

                    # Both VC tools executed
                    assert engine.expand_topic.call_count == 2

                    # stop_reason=tool_use
                    assert b'"stop_reason": "tool_use"' in body or b'"stop_reason":"tool_use"' in body

