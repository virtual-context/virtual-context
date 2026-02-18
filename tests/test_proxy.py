"""Tests for virtual_context.proxy.server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_context.proxy.server import (
    ProxyState,
    SessionRegistry,
    SessionState,
    _detect_api_format,
    _extract_assistant_text,
    _extract_delta_text,
    _extract_history_pairs,
    _extract_session_id,
    _extract_user_message,
    _filter_body_messages,
    _forward_headers,
    _inject_context,
    _inject_session_marker,
    _last_text_block,
    _strip_openclaw_envelope,
    _strip_session_markers,
    _strip_vc_prompt,
    create_app,
)
from virtual_context.config import load_config
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import AssembledContext, Message, TagResult, TurnTagEntry


# ---------------------------------------------------------------------------
# _detect_api_format
# ---------------------------------------------------------------------------


class TestDetectApiFormat:
    def test_anthropic_system_field(self):
        body = {"system": "You are helpful", "messages": [], "model": "claude-3"}
        assert _detect_api_format(body) == "anthropic"

    def test_anthropic_claude_model(self):
        body = {"messages": [], "model": "claude-haiku-4-5-20251001"}
        assert _detect_api_format(body) == "anthropic"

    def test_openai_default(self):
        body = {"messages": [], "model": "gpt-4o"}
        assert _detect_api_format(body) == "openai"

    def test_openai_no_model(self):
        body = {"messages": []}
        assert _detect_api_format(body) == "openai"

    def test_anthropic_empty_system(self):
        """Even empty string system field → anthropic."""
        body = {"system": "", "messages": []}
        assert _detect_api_format(body) == "anthropic"


# ---------------------------------------------------------------------------
# _extract_user_message
# ---------------------------------------------------------------------------


class TestExtractUserMessage:
    def test_string_content(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]}
        assert _extract_user_message(body) == "How are you?"

    def test_content_blocks(self):
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "First part"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "second part"},
            ]},
        ]}
        # Last text block extraction — returns only the final text block
        assert _extract_user_message(body) == "second part"

    def test_returns_last_user(self):
        body = {"messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]}
        assert _extract_user_message(body) == "Second"

    def test_no_user_message(self):
        body = {"messages": [{"role": "assistant", "content": "Hi"}]}
        assert _extract_user_message(body) == ""

    def test_empty_messages(self):
        body = {"messages": []}
        assert _extract_user_message(body) == ""

    def test_no_messages_key(self):
        body = {}
        assert _extract_user_message(body) == ""

    def test_content_blocks_returns_last_text(self):
        """With multiple text blocks, returns only the last one (skips thinking)."""
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "**Thinking about the question**\n\nLet me consider..."},
                {"type": "text", "text": "What is the weather?"},
            ]},
        ]}
        assert _extract_user_message(body) == "What is the weather?"

    def test_skips_non_text_blocks(self):
        """Non-text blocks are ignored; last text block is returned."""
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "First text"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "Actual question"},
                {"type": "tool_result", "content": "result"},
            ]},
        ]}
        assert _extract_user_message(body) == "Actual question"

    @pytest.mark.regression("PROXY-005")
    def test_tool_result_only_returns_empty(self):
        """When last user message is pure tool_result, returns empty string."""
        body = {"messages": [
            {"role": "user", "content": "Search for pandas docs"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"q": "pandas"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "search results here"},
            ]},
        ]}
        assert _extract_user_message(body) == ""


# ---------------------------------------------------------------------------
# _last_text_block
# ---------------------------------------------------------------------------


class TestLastTextBlock:
    def test_single_text_block(self):
        content = [{"type": "text", "text": "Hello"}]
        assert _last_text_block(content) == "Hello"

    def test_multiple_text_blocks_returns_last(self):
        content = [
            {"type": "text", "text": "**Thinking**\n\nI need to consider..."},
            {"type": "text", "text": "Here is my answer."},
        ]
        assert _last_text_block(content) == "Here is my answer."

    def test_skips_non_text_blocks(self):
        content = [
            {"type": "text", "text": "Thinking..."},
            {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
            {"type": "text", "text": "Final answer"},
            {"type": "tool_use", "id": "t2", "name": "save", "input": {}},
        ]
        assert _last_text_block(content) == "Final answer"

    def test_thinking_then_tool_use_then_response(self):
        """Realistic OpenClaw pattern: thinking + tool_use + response."""
        content = [
            {"type": "text", "text": "**Crafting a response**\n\nLet me think..."},
            {"type": "tool_use", "id": "t1", "name": "message", "input": {}},
            {"type": "text", "text": "Good. I already opened the door."},
        ]
        assert _last_text_block(content) == "Good. I already opened the door."

    def test_no_text_blocks(self):
        content = [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
            {"type": "image", "source": {}},
        ]
        assert _last_text_block(content) == ""

    def test_empty_content(self):
        assert _last_text_block([]) == ""

    def test_single_block_with_thinking(self):
        """Single text block that IS the actual content (no thinking)."""
        content = [{"type": "text", "text": "Ack."}]
        assert _last_text_block(content) == "Ack."


# ---------------------------------------------------------------------------
# _strip_vc_prompt
# ---------------------------------------------------------------------------


class TestStripVcPrompt:
    def test_strips_marker(self):
        text = "[vc:prompt]\n[Telegram Y (@y) id:123] hello world\n[message_id: 456]"
        result = _strip_vc_prompt(text)
        assert result == "[Telegram Y (@y) id:123] hello world\n[message_id: 456]"

    def test_no_marker_returns_unchanged(self):
        text = "just plain text"
        assert _strip_vc_prompt(text) == text

    def test_empty_string(self):
        assert _strip_vc_prompt("") == ""

    def test_marker_only(self):
        assert _strip_vc_prompt("[vc:prompt]\n") == ""

    def test_marker_not_at_start(self):
        """Marker must be at the start to be stripped."""
        text = "prefix [vc:prompt]\nsome content"
        assert _strip_vc_prompt(text) == text


# ---------------------------------------------------------------------------
# _strip_openclaw_envelope
# ---------------------------------------------------------------------------


class TestStripOpenclawEnvelope:
    def test_empty_string(self):
        assert _strip_openclaw_envelope("") == ""

    def test_plain_text_passthrough(self):
        assert _strip_openclaw_envelope("just a question") == "just a question"

    def test_strips_vc_prompt_marker(self):
        text = "[vc:prompt]\nhello world"
        assert _strip_openclaw_envelope(text) == "hello world"

    def test_strips_channel_header(self):
        text = "[Telegram Y (@yursilk) id:8049932331 +17m Sun 2026-02-15 22:07 EST] what time is it"
        assert _strip_openclaw_envelope(text) == "what time is it"

    def test_strips_message_id_footer(self):
        text = "some content\n[message_id: 8663]"
        assert _strip_openclaw_envelope(text) == "some content"

    @pytest.mark.regression("PROXY-003")
    def test_strips_full_envelope(self):
        """Full OpenClaw message: [vc:prompt] + channel header + content + footer."""
        text = (
            "[vc:prompt]\n"
            "[Telegram Y (@yursilk) id:8049932331 +17m Sun 2026-02-15 22:07 EST] "
            "what time is it\n"
            "[message_id: 8663]"
        )
        assert _strip_openclaw_envelope(text) == "what time is it"

    def test_strips_system_event(self):
        text = (
            "System: [2026-02-15T22:00:00Z] Model switched to claude-opus-4-6\n\n"
            "[Telegram Y (@yursilk) id:123 +5m Sun] hello\n"
            "[message_id: 456]"
        )
        assert _strip_openclaw_envelope(text) == "hello"

    def test_strips_system_event_with_vc_prompt(self):
        text = (
            "[vc:prompt]\n"
            "System: [2026-02-15T22:00:00Z] Model switched\n\n"
            "[Telegram Y (@yursilk) id:123 +5m Sun] question\n"
            "[message_id: 456]"
        )
        assert _strip_openclaw_envelope(text) == "question"

    def test_vc_user_backward_compat(self):
        """Old-format [vc:user]...[/vc:user] extracts inner content."""
        text = "[vc:user]clean content here[/vc:user]\n[Telegram ...] more stuff"
        assert _strip_openclaw_envelope(text) == "clean content here"

    def test_vc_prompt_then_vc_user(self):
        """[vc:prompt] stripped first, then [vc:user] inner content extracted."""
        text = "[vc:prompt]\n[vc:user]the question[/vc:user]\ngarbage"
        assert _strip_openclaw_envelope(text) == "the question"

    def test_whatsapp_channel(self):
        text = "[WhatsApp User id:12345 +2m Mon] how is the weather\n[message_id: 99]"
        assert _strip_openclaw_envelope(text) == "how is the weather"

    def test_discord_channel(self):
        text = "[Discord Bob id:777 +1m Tue] hello there\n[message_id: 42]"
        assert _strip_openclaw_envelope(text) == "hello there"

    def test_no_id_in_header_not_stripped(self):
        """Bracketed text without id:NNN is not treated as channel header."""
        text = "[Some random bracket] content here"
        assert _strip_openclaw_envelope(text) == "[Some random bracket] content here"

    def test_multiline_content_preserved(self):
        text = (
            "[Telegram Y id:123 +5m Sun] line one\n"
            "line two\n"
            "line three\n"
            "[message_id: 456]"
        )
        assert _strip_openclaw_envelope(text) == "line one\nline two\nline three"

    def test_short_message_extraction(self):
        """Short messages (where metadata dominates) are correctly extracted."""
        text = (
            "[vc:prompt]\n"
            "[Telegram Y (@yursilk) id:8049932331 +3m Sun 2026-02-15 22:10 EST] "
            "ok\n"
            "[message_id: 8670]"
        )
        assert _strip_openclaw_envelope(text) == "ok"

    # Integration tests with _extract_user_message

    def test_extract_user_message_full_envelope(self):
        """_extract_user_message strips full OpenClaw envelope."""
        body = {"messages": [
            {"role": "user", "content": (
                "[vc:prompt]\n"
                "[Telegram Y (@yursilk) id:123 +5m Sun] what time is it\n"
                "[message_id: 789]"
            )},
        ]}
        result = _extract_user_message(body)
        assert result == "what time is it"

    def test_extract_user_message_content_blocks(self):
        """Content block array: envelope stripped from last text block."""
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "thinking overhead..."},
                {"type": "text", "text": (
                    "[vc:prompt]\n"
                    "[Telegram Y id:123] actual question\n[message_id: 1]"
                )},
            ]},
        ]}
        result = _extract_user_message(body)
        assert result == "actual question"

    def test_extract_user_message_no_envelope_passthrough(self):
        """Without OpenClaw envelope, message passes through unchanged."""
        body = {"messages": [
            {"role": "user", "content": "plain message no envelope"},
        ]}
        assert _extract_user_message(body) == "plain message no envelope"

    def test_extract_message_text_strips_envelope(self):
        """_extract_message_text strips full envelope."""
        from virtual_context.proxy.server import _extract_message_text
        msg = {"role": "user", "content": (
            "[vc:prompt]\n"
            "[Telegram Y id:99 +1m] the real content\n[message_id: 99]"
        )}
        result = _extract_message_text(msg)
        assert result == "the real content"

    @pytest.mark.regression("PROXY-003")
    def test_history_pairs_strip_envelope(self):
        """Historical user messages get full envelope stripped."""
        body = {"messages": [
            {"role": "user", "content": (
                "[vc:prompt]\n"
                "[Telegram Y id:1 +1m] first question\n[message_id: 1]"
            )},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": (
                "[vc:prompt]\n"
                "[Telegram Y id:2 +2m] second question\n[message_id: 2]"
            )},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 2
        assert pairs[0].content == "first question"
        assert pairs[1].content == "first answer"


# ---------------------------------------------------------------------------
# _inject_context
# ---------------------------------------------------------------------------


class TestInjectContext:
    def test_empty_prepend_returns_body(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        result = _inject_context(body, "", "openai")
        assert result is body  # no copy when empty

    def test_openai_with_system_message(self):
        body = {"messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]}
        result = _inject_context(body, "context here", "openai")
        assert result["messages"][0]["content"].startswith("<virtual-context>")
        assert "context here" in result["messages"][0]["content"]
        assert "Be helpful" in result["messages"][0]["content"]

    def test_openai_without_system_message(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        result = _inject_context(body, "context here", "openai")
        assert result["messages"][0]["role"] == "system"
        assert "context here" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"

    def test_anthropic_string_system(self):
        body = {"system": "Be helpful", "messages": []}
        result = _inject_context(body, "context here", "anthropic")
        assert result["system"].startswith("<virtual-context>")
        assert "context here" in result["system"]
        assert "Be helpful" in result["system"]

    def test_anthropic_no_system(self):
        body = {"messages": []}
        result = _inject_context(body, "context here", "anthropic")
        assert "<virtual-context>" in result["system"]
        assert "context here" in result["system"]

    def test_anthropic_list_system(self):
        body = {
            "system": [{"type": "text", "text": "Existing system"}],
            "messages": [],
        }
        result = _inject_context(body, "context here", "anthropic")
        assert isinstance(result["system"], list)
        assert result["system"][0]["type"] == "text"
        assert "context here" in result["system"][0]["text"]
        assert result["system"][1]["text"] == "Existing system"

    def test_does_not_mutate_original(self):
        body = {"messages": [
            {"role": "system", "content": "Original"},
            {"role": "user", "content": "Hi"},
        ]}
        original_content = body["messages"][0]["content"]
        _inject_context(body, "injected", "openai")
        assert body["messages"][0]["content"] == original_content


# ---------------------------------------------------------------------------
# _forward_headers
# ---------------------------------------------------------------------------


class TestForwardHeaders:
    def test_strips_hop_by_hop(self):
        headers = {
            "Authorization": "Bearer sk-xxx",
            "Content-Type": "application/json",
            "Host": "api.openai.com",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }
        result = _forward_headers(headers)
        assert "Authorization" in result
        assert "Content-Type" in result
        assert "Host" not in result
        assert "Connection" not in result
        assert "Transfer-Encoding" not in result

    def test_case_insensitive(self):
        headers = {"host": "example.com", "connection": "close"}
        result = _forward_headers(headers)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _extract_delta_text
# ---------------------------------------------------------------------------


class TestExtractDeltaText:
    def test_openai_delta(self):
        data = {"choices": [{"delta": {"content": "Hello"}}]}
        assert _extract_delta_text(data, "openai") == "Hello"

    def test_openai_empty_delta(self):
        data = {"choices": [{"delta": {}}]}
        assert _extract_delta_text(data, "openai") == ""

    def test_openai_no_choices(self):
        data = {}
        assert _extract_delta_text(data, "openai") == ""

    def test_anthropic_content_block_delta(self):
        data = {"type": "content_block_delta", "delta": {"text": "World"}}
        assert _extract_delta_text(data, "anthropic") == "World"

    def test_anthropic_non_delta_event(self):
        data = {"type": "message_start", "message": {}}
        assert _extract_delta_text(data, "anthropic") == ""

    def test_openai_none_content(self):
        data = {"choices": [{"delta": {"content": None}}]}
        assert _extract_delta_text(data, "openai") == ""


# ---------------------------------------------------------------------------
# _extract_assistant_text
# ---------------------------------------------------------------------------


class TestExtractAssistantText:
    def test_openai_response(self):
        body = {"choices": [{"message": {"content": "Hello there"}}]}
        assert _extract_assistant_text(body, "openai") == "Hello there"

    def test_openai_no_choices(self):
        body = {"choices": []}
        assert _extract_assistant_text(body, "openai") == ""

    def test_anthropic_response(self):
        body = {"content": [
            {"type": "text", "text": "Hello world"},
        ]}
        assert _extract_assistant_text(body, "anthropic") == "Hello world"

    def test_anthropic_skips_thinking_blocks(self):
        """Last text block extraction skips earlier thinking blocks."""
        body = {"content": [
            {"type": "text", "text": "**Planning**\n\nI need to think about this..."},
            {"type": "text", "text": "Here is my actual answer."},
        ]}
        assert _extract_assistant_text(body, "anthropic") == "Here is my actual answer."

    def test_anthropic_empty_content(self):
        body = {"content": []}
        assert _extract_assistant_text(body, "anthropic") == ""

    def test_openai_none_content(self):
        body = {"choices": [{"message": {"content": None}}]}
        assert _extract_assistant_text(body, "openai") == ""


# ---------------------------------------------------------------------------
# ProxyState
# ---------------------------------------------------------------------------


class TestProxyState:
    def test_wait_for_complete_noop_when_no_pending(self):
        engine = MagicMock()
        state = ProxyState(engine)
        state.wait_for_complete()  # should not raise

    def test_fire_and_wait(self):
        engine = MagicMock()
        state = ProxyState(engine)
        history = [Message(role="user", content="hi"), Message(role="assistant", content="hey")]
        state.fire_turn_complete(history)
        state.wait_for_complete()
        engine.on_turn_complete.assert_called_once_with(history)

    def test_error_in_turn_complete_is_caught(self):
        engine = MagicMock()
        engine.on_turn_complete.side_effect = RuntimeError("boom")
        state = ProxyState(engine)
        history = [Message(role="user", content="hi")]
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
            assert "<virtual-context>" in forwarded_body["system"]
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


# ---------------------------------------------------------------------------
# _extract_history_pairs
# ---------------------------------------------------------------------------


class TestExtractHistoryPairs:
    def test_string_content(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Current question"},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 2
        assert pairs[0].role == "user"
        assert pairs[0].content == "Hello"
        assert pairs[1].role == "assistant"
        assert pairs[1].content == "Hi there"

    def test_content_blocks(self):
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "Part 2"},
            ]},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Current"},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 2
        # Last text block extraction — returns only final text block
        assert pairs[0].content == "Part 2"
        assert pairs[1].content == "Response"

    def test_system_messages_excluded(self):
        body = {"messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Current"},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 2
        assert pairs[0].role == "user"
        assert pairs[0].content == "Hello"

    def test_single_user_message_no_history(self):
        """First message ever — no history to extract."""
        body = {"messages": [
            {"role": "user", "content": "Hello"},
        ]}
        pairs = _extract_history_pairs(body)
        assert pairs == []

    def test_odd_message_count(self):
        """Multiple complete pairs with trailing current user message."""
        body = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Current"},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 4  # 2 complete pairs
        assert pairs[0].content == "Q1"
        assert pairs[1].content == "A1"
        assert pairs[2].content == "Q2"
        assert pairs[3].content == "A2"

    def test_empty_messages(self):
        body = {"messages": []}
        assert _extract_history_pairs(body) == []

    def test_no_messages_key(self):
        body = {}
        assert _extract_history_pairs(body) == []

    def test_multiple_pairs(self):
        body = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Current"},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 6  # 3 complete pairs
        assert pairs[4].content == "Q3"
        assert pairs[5].content == "A3"

    @pytest.mark.regression("PROXY-001")
    def test_consecutive_user_messages_at_end(self):
        """OpenClaw batches multiple Telegram messages as consecutive user turns.

        When the trailing messages are [user, user], after dropping the last
        user (current turn), the list ends with a user — history must still
        be extracted from earlier pairs.
        """
        body = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Batched msg 1"},
            {"role": "user", "content": "Batched msg 2"},
            {"role": "user", "content": "Current"},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 4  # 2 complete pairs, batched users skipped
        assert pairs[0].content == "Q1"
        assert pairs[1].content == "A1"
        assert pairs[2].content == "Q2"
        assert pairs[3].content == "A2"

    @pytest.mark.regression("PROXY-001")
    def test_consecutive_user_messages_mid_conversation(self):
        """Consecutive user messages in the middle should be skipped,
        but pairs on both sides should be extracted."""
        body = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Batch 1"},
            {"role": "user", "content": "Batch 2"},
            {"role": "user", "content": "Batch 3"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Current"},
        ]}
        pairs = _extract_history_pairs(body)
        # Pair walking: Q1+A1, then Batch1 (user) + Batch2 (user) → skip,
        # Batch2 (user) + Batch3 (user) → skip, Batch3 (user) + A2 → skip
        # (misaligned: Batch3 is user, A2 is assistant but Batch3 was already
        # advanced past)... Actually the walker advances 1 at a time on mismatch:
        # i=2: Batch1(user) + Batch2(user) → skip, i=3
        # i=3: Batch2(user) + Batch3(user) → skip, i=4
        # i=4: Batch3(user) + A2(assistant) → pair! i=6
        # i=6: Q3(user) + A3(assistant) → pair! i=8
        assert len(pairs) == 6  # 3 complete pairs
        assert pairs[0].content == "Q1"
        assert pairs[2].content == "Batch 3"
        assert pairs[4].content == "Q3"


# ---------------------------------------------------------------------------
# ProxyState ingestion
# ---------------------------------------------------------------------------


class TestProxyStateIngestion:
    def _make_engine(self, session_id="test-session"):
        engine = MagicMock()
        engine.config.session_id = session_id
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
        engine.config.session_id = "session-2"
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
        assert evt["session_id"] == "test-session"
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
        engine._turn_tag_index = TurnTagIndex()
        engine._store = MagicMock()
        engine._store.get_all_tags.return_value = []
        engine.config.tag_generator.context_lookback_pairs = 5
        engine.config.tag_generator.context_bleed_threshold = 0
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
        engine._tag_generator = MagicMock()
        engine._tag_generator.generate_tags.side_effect = tag_results

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
        engine._tag_generator = MagicMock()

        from virtual_context.engine import VirtualContextEngine
        result = VirtualContextEngine.ingest_history(engine, [])

        assert result == 0
        assert len(engine._turn_tag_index.entries) == 0

    def test_ingest_refreshes_store_tags(self):
        """Store tags are refreshed every 10 turns."""
        engine = self._make_mock_engine()
        engine._tag_generator = MagicMock()
        engine._tag_generator.generate_tags.return_value = TagResult(
            tags=["tag"], primary="tag", source="keyword"
        )

        from virtual_context.engine import VirtualContextEngine
        # 15 pairs = 30 messages
        pairs = []
        for i in range(15):
            pairs.append(Message(role="user", content=f"Q{i}"))
            pairs.append(Message(role="assistant", content=f"A{i}"))

        VirtualContextEngine.ingest_history(engine, pairs)

        # get_all_tags called: once at start + once after turn 10
        assert engine._store.get_all_tags.call_count == 2


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
# _filter_body_messages
# ---------------------------------------------------------------------------


class TestFilterBodyMessages:
    """Test history filtering of raw API request bodies."""

    def _build_index(self, turn_tags: list[list[str]]) -> TurnTagIndex:
        """Build a TurnTagIndex with the given per-turn tag lists."""
        idx = TurnTagIndex()
        for i, tags in enumerate(turn_tags):
            idx.append(TurnTagEntry(
                turn_number=i,
                message_hash=f"h{i}",
                tags=tags,
                primary_tag=tags[0] if tags else "_general",
            ))
        return idx

    def _build_body(self, n_pairs: int, current_user: bool = True) -> dict:
        """Build a request body with n user+assistant pairs + optional trailing user."""
        messages = []
        for i in range(n_pairs):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})
        if current_user:
            messages.append({"role": "user", "content": "Current question"})
        return {"messages": messages}

    def test_drops_irrelevant_turns(self):
        """Turns with no tag overlap are dropped."""
        # 5 history pairs + current user message
        body = self._build_body(5)
        idx = self._build_index([
            ["python", "testing"],   # turn 0 — matches
            ["cooking", "recipes"],  # turn 1 — no match
            ["music", "guitar"],     # turn 2 — no match
            ["python", "api"],       # turn 3 — matches
            ["weather"],             # turn 4 — no match (but protected)
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # Keep: turn 0, turn 3 (tag match), turn 4 (protected), current user
        assert dropped == 2  # turns 1 and 2 dropped
        msgs = filtered["messages"]
        assert len(msgs) == 7  # 3 kept pairs * 2 + current user

    def test_protects_recent_turns(self):
        """Recent N turns are always kept regardless of tags."""
        body = self._build_body(4)
        idx = self._build_index([
            ["python"],     # turn 0
            ["cooking"],    # turn 1
            ["music"],      # turn 2
            ["weather"],    # turn 3
        ])
        # No tag overlap at all, but recent_turns=2 protects turns 2 and 3
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated-tag"], recent_turns=2,
        )
        assert dropped == 2  # turns 0 and 1 dropped
        msgs = filtered["messages"]
        assert len(msgs) == 5  # 2 protected pairs * 2 + current user

    @pytest.mark.regression("BUG-007")
    def test_broad_keeps_everything(self):
        """Broad queries keep all turns."""
        body = self._build_body(5)
        idx = self._build_index([
            ["python"],
            ["cooking"],
            ["music"],
            ["weather"],
            ["cars"],
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated"], recent_turns=1, broad=True,
        )
        assert dropped == 0
        assert filtered is body  # unchanged

    @pytest.mark.regression("BUG-008")
    def test_temporal_keeps_everything(self):
        """Temporal queries keep all turns."""
        body = self._build_body(3)
        idx = self._build_index([["a"], ["b"], ["c"]])
        filtered, dropped = _filter_body_messages(
            body, idx, ["x"], recent_turns=1, temporal=True,
        )
        assert dropped == 0

    def test_rule_tag_always_kept(self):
        """Turns tagged with 'rule' are always kept."""
        body = self._build_body(4)
        idx = self._build_index([
            ["rule", "style"],   # turn 0 — rule tag
            ["cooking"],         # turn 1 — no match
            ["music"],           # turn 2 — no match
            ["weather"],         # turn 3 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # turn 0 kept (rule), turn 3 kept (protected), turns 1+2 dropped
        assert dropped == 2
        msgs = filtered["messages"]
        assert len(msgs) == 5  # 2 pairs * 2 + current user

    @pytest.mark.regression("PROXY-002")
    def test_no_index_entries_skips_filtering(self):
        """If TurnTagIndex is empty, no filtering occurs."""
        body = self._build_body(5)
        idx = TurnTagIndex()  # empty
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        assert dropped == 0
        assert filtered is body

    def test_too_few_turns_skips_filtering(self):
        """If total turns <= recent_turns, no filtering occurs."""
        body = self._build_body(2)
        idx = self._build_index([["a"], ["b"]])
        filtered, dropped = _filter_body_messages(
            body, idx, ["x"], recent_turns=3,
        )
        assert dropped == 0

    def test_preserves_system_message(self):
        """OpenAI-style system message at position 0 is preserved."""
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["python"],    # turn 0
            ["cooking"],   # turn 1 — no match
            ["music"],     # turn 2 — no match
            ["python"],    # turn 3 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        assert dropped == 2
        msgs = filtered["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"
        # system + turn 0 pair + turn 3 pair + current user = 6
        assert len(msgs) == 6

    def test_no_tag_overlap_drops_all_older(self):
        """When nothing matches, only protected turns + current remain."""
        body = self._build_body(6)
        idx = self._build_index([
            ["a"], ["b"], ["c"], ["d"], ["e"], ["f"],
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["zzz-nonexistent"], recent_turns=2,
        )
        assert dropped == 4
        msgs = filtered["messages"]
        # 2 protected pairs * 2 + current user = 5
        assert len(msgs) == 5

    @pytest.mark.regression("PROXY-004")
    def test_tool_use_keeps_tool_result_pair(self):
        """If an assistant uses tool_use, the next pair (tool_result) must also be kept."""
        body = {
            "messages": [
                # Pair 0: normal turn (no match → would be dropped)
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: assistant uses tool (matched → kept)
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {}},
                ]},
                # Pair 2: tool_result (no match → but must be kept because pair 1 has tool_use)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "72F"},
                ]},
                {"role": "assistant", "content": "It's 72F"},
                # Pair 3: normal (no match → dropped)
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                # Pair 4: protected
                {"role": "user", "content": "Q4"},
                {"role": "assistant", "content": "A4"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],    # pair 0 — no match
            ["python"],     # pair 1 — matches
            ["cooking"],    # pair 2 — no match but forced by tool_use chain
            ["cooking"],    # pair 3 — no match
            ["cooking"],    # pair 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # pair 0 dropped, pair 1 kept (tag match), pair 2 force-kept (tool_result),
        # pair 3 dropped, pair 4 protected
        assert dropped == 2
        msgs = filtered["messages"]
        # 3 kept pairs * 2 + current user = 7
        assert len(msgs) == 7

    @pytest.mark.regression("PROXY-004")
    def test_tool_result_keeps_preceding_tool_use_pair(self):
        """If we keep a pair with tool_result, the preceding pair must also be kept."""
        body = {
            "messages": [
                # Pair 0: assistant uses tool (no tag match → would be dropped)
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool_1", "name": "search", "input": {}},
                ]},
                # Pair 1: tool_result + response (tag match → kept)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "results"},
                ]},
                {"role": "assistant", "content": "Found it"},
                # Pair 2: protected
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],    # pair 0 — no match but forced by pair 1's tool_result
            ["python"],     # pair 1 — matches
            ["cooking"],    # pair 2 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # All 3 pairs kept (pair 0 force-kept due to tool chain)
        assert dropped == 0


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


# ---------------------------------------------------------------------------
# Session marker: extraction
# ---------------------------------------------------------------------------


class TestExtractSessionId:
    def test_extracts_from_string_content(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there\n<!-- vc:session=a3f8b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c -->"},
            {"role": "user", "content": "Next question"},
        ]}
        assert _extract_session_id(body) == "a3f8b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"

    def test_extracts_from_content_blocks(self):
        body = {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "thinking..."},
                {"type": "text", "text": "answer\n<!-- vc:session=dead-beef-1234 -->"},
            ]},
            {"role": "user", "content": "Next"},
        ]}
        assert _extract_session_id(body) == "dead-beef-1234"

    def test_returns_none_when_no_marker(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Next"},
        ]}
        assert _extract_session_id(body) is None

    def test_returns_none_for_empty_messages(self):
        assert _extract_session_id({"messages": []}) is None
        assert _extract_session_id({}) is None

    def test_finds_most_recent_marker(self):
        """When multiple assistant messages have markers, returns the most recent."""
        body = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1\n<!-- vc:session=aaa00000-0000-0000-0000-000000000001 -->"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2\n<!-- vc:session=bbb00000-0000-0000-0000-000000000002 -->"},
            {"role": "user", "content": "Q3"},
        ]}
        assert _extract_session_id(body) == "bbb00000-0000-0000-0000-000000000002"

    def test_ignores_user_messages(self):
        """Markers in user messages are not extracted."""
        body = {"messages": [
            {"role": "user", "content": "<!-- vc:session=fake-id --> hello"},
            {"role": "assistant", "content": "Hi"},
        ]}
        assert _extract_session_id(body) is None


# ---------------------------------------------------------------------------
# Session marker: stripping
# ---------------------------------------------------------------------------


class TestStripSessionMarkers:
    def test_strips_from_string_content(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there\n<!-- vc:session=abc00000-0000-0000-0000-000000000123 -->"},
        ]}
        result = _strip_session_markers(body)
        assert result["messages"][1]["content"] == "Hi there"

    def test_strips_from_content_blocks(self):
        body = {"messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Answer\n<!-- vc:session=abc00000-0000-0000-0000-000000000123 -->"},
            ]},
        ]}
        result = _strip_session_markers(body)
        assert result["messages"][0]["content"][0]["text"] == "Answer"

    def test_no_marker_returns_same_body(self):
        body = {"messages": [
            {"role": "assistant", "content": "clean text"},
        ]}
        result = _strip_session_markers(body)
        assert result is body  # no copy made

    def test_preserves_user_messages(self):
        body = {"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer\n<!-- vc:session=aaa00000-0000-0000-0000-000000000001 -->"},
        ]}
        result = _strip_session_markers(body)
        assert result["messages"][0]["content"] == "question"
        assert result["messages"][1]["content"] == "answer"

    def test_strips_multiple_assistant_messages(self):
        body = {"messages": [
            {"role": "assistant", "content": "A1\n<!-- vc:session=aaa00000-0000-0000-0000-000000000001 -->"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2\n<!-- vc:session=bbb00000-0000-0000-0000-000000000002 -->"},
        ]}
        result = _strip_session_markers(body)
        assert result["messages"][0]["content"] == "A1"
        assert result["messages"][2]["content"] == "A2"


# ---------------------------------------------------------------------------
# Session marker: injection into non-streaming response
# ---------------------------------------------------------------------------


class TestInjectSessionMarker:
    def test_openai_format(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        result = _inject_session_marker(resp, "\n<!-- vc:session=abc -->", "openai")
        assert result["choices"][0]["message"]["content"] == "Hello\n<!-- vc:session=abc -->"

    def test_anthropic_format(self):
        resp = {"content": [{"type": "text", "text": "Hello"}]}
        result = _inject_session_marker(resp, "\n<!-- vc:session=abc -->", "anthropic")
        assert result["content"][0]["text"] == "Hello\n<!-- vc:session=abc -->"

    def test_anthropic_multiple_blocks_appends_to_last(self):
        resp = {"content": [
            {"type": "text", "text": "Thinking..."},
            {"type": "text", "text": "Answer"},
        ]}
        result = _inject_session_marker(resp, "\n<!-- vc:session=abc -->", "anthropic")
        assert result["content"][0]["text"] == "Thinking..."
        assert result["content"][1]["text"] == "Answer\n<!-- vc:session=abc -->"

    def test_does_not_mutate_original(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        original_content = resp["choices"][0]["message"]["content"]
        _inject_session_marker(resp, "\n<!-- vc:session=abc -->", "openai")
        assert resp["choices"][0]["message"]["content"] == original_content


# ---------------------------------------------------------------------------
# Session marker: streaming integration
# ---------------------------------------------------------------------------


class TestStreamingSessionMarker:
    def test_streaming_response_includes_session_marker(self, test_client):
        """Streaming response should include a session marker SSE event."""
        client, engine = test_client

        sse_events = (
            b"event: content_block_delta\r\n"
            b"data: {\"type\":\"content_block_delta\",\"index\":0,"
            b"\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\r\n"
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
                body = resp.content.decode("utf-8", errors="replace")
                assert "vc:session=" in body
                # The session ID should be a UUID from the engine
                assert "content_block_delta" in body

    def test_non_streaming_response_includes_session_marker(self, test_client):
        """Non-streaming response body should include session marker."""
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
                    "model": "claude-3",
                    "system": "test",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            # The last text block should contain the session marker
            text = data["content"][-1]["text"]
            assert "<!-- vc:session=" in text


# ---------------------------------------------------------------------------
# SessionRegistry
# ---------------------------------------------------------------------------


class TestSessionRegistry:
    def test_creates_new_session_when_no_id_and_empty(self, tmp_path):
        """No session ID + empty registry → creates a new session."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            engine = MagicMock()
            engine.config.session_id = "new-uuid-1234"
            MockEngine.return_value = engine

            state, is_new = registry.get_or_create(None)

        assert is_new is True
        assert state is not None
        assert state.engine.config.session_id == "new-uuid-1234"
        assert registry.session_count == 1

    def test_reuses_session_via_fingerprint(self, tmp_path):
        """No session ID + matching fingerprint → reuses existing session."""
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.session_id = "default-session"
        default = ProxyState(engine, metrics=metrics)

        body = {"messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]}

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._sessions["default-session"] = default
        # Register the fingerprint for this body
        fp = SessionRegistry._compute_fingerprint(body)
        registry._fingerprints[fp] = "default-session"

        result, is_new = registry.get_or_create(None, body=body)
        assert is_new is False
        assert result is default

    def test_resumes_existing_session(self, tmp_path):
        """Known session ID in memory → returns existing state."""
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.session_id = "existing-session"
        state = ProxyState(engine, metrics=metrics)

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._sessions["existing-session"] = state

        result, is_new = registry.get_or_create("existing-session")
        assert is_new is False
        assert result is state

    def test_loads_state_from_store_on_restart(self, tmp_path):
        """Known session ID but not in memory → creates engine, loads state."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            engine = MagicMock()
            engine.config.session_id = "restored-session"
            MockEngine.return_value = engine

            state, is_new = registry.get_or_create("restored-session")

        assert is_new is True
        assert state.engine.config.session_id == "restored-session"
        # Engine should have had _load_persisted_state called
        engine._load_persisted_state.assert_called_once()

    def test_multiple_sessions(self, tmp_path):
        """Multiple sessions can coexist."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        engines = []
        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            for sid in ["session-a", "session-b", "session-c"]:
                engine = MagicMock()
                engine.config.session_id = sid
                MockEngine.return_value = engine
                engines.append(engine)
                registry.get_or_create(None if sid == "session-a" else sid)

        assert registry.session_count == 3

    def test_shutdown_all_clears_sessions(self, tmp_path):
        """shutdown_all() cleans up all sessions."""
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.session_id = "s1"
        state = ProxyState(engine, metrics=metrics)

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._sessions["s1"] = state

        registry.shutdown_all()
        assert registry.session_count == 0


# ---------------------------------------------------------------------------
# Session awareness: request events include session_id
# ---------------------------------------------------------------------------


class TestRequestEventSessionId:
    def test_request_event_includes_session_id(self):
        """Request event dict should carry session_id."""
        m = ProxyMetrics()
        m.record({
            "type": "request",
            "turn": 0,
            "session_id": "abc-123",
            "tags": ["test"],
        })
        events = m.events_since(-1)
        assert len(events) == 1
        assert events[0]["session_id"] == "abc-123"

    def test_captured_request_includes_session_id(self):
        """capture_request() with session_id appears in summary."""
        m = ProxyMetrics()
        m.capture_request(
            0, {"messages": []}, "anthropic",
            session_id="sess-xyz",
        )
        summaries = m.get_captured_requests_summary()
        assert len(summaries) == 1
        assert summaries[0]["session_id"] == "sess-xyz"

        req = m.get_captured_request(0)
        assert req["session_id"] == "sess-xyz"


# ---------------------------------------------------------------------------
# Live sessions in snapshot / HTML
# ---------------------------------------------------------------------------


class TestLiveSessions:
    def test_dashboard_html_has_session_column(self):
        """The dashboard HTML template should include a Session column header."""
        from virtual_context.proxy.dashboard import get_dashboard_html
        html = get_dashboard_html()
        assert "<th>Session</th>" in html

    def test_live_sessions_in_snapshot(self):
        """SSE snapshot should include live_sessions when registry has sessions."""
        # This tests the data shape: registry provides session data
        # that gets included in the SSE snapshot.
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.session_id = "live-session-1"
        engine._compacted_through = 0
        engine._turn_tag_index = MagicMock()
        engine._turn_tag_index.entries = []
        engine._turn_tag_index.get_active_tags.return_value = ["tag-a"]

        state = ProxyState(engine, metrics=metrics)
        state.conversation_history = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._sessions["live-session-1"] = state

        # Simulate what the SSE snapshot builder does
        live_sessions = []
        for sid, s in registry._sessions.items():
            live_sessions.append({
                "session_id": sid,
                "turn_count": len(s.conversation_history) // 2,
                "compacted_through": getattr(s.engine, "_compacted_through", 0),
                "tag_count": len(s.engine._turn_tag_index.entries),
                "active_tags": list(
                    s.engine._turn_tag_index.get_active_tags(lookback=6)
                ),
            })

        assert len(live_sessions) == 1
        ls = live_sessions[0]
        assert ls["session_id"] == "live-session-1"
        assert ls["turn_count"] == 1
        assert ls["active_tags"] == ["tag-a"]


# ---------------------------------------------------------------------------
# PROXY-010: Content fingerprint session routing
# ---------------------------------------------------------------------------


class TestContentFingerprintRouting:
    """Different conversations (no session marker) must get separate sessions."""

    def _make_body(self, user_messages: list[str]) -> dict:
        """Build a minimal Anthropic-format request body."""
        msgs = []
        for i, text in enumerate(user_messages):
            msgs.append({"role": "user", "content": text})
            if i < len(user_messages) - 1:
                msgs.append({"role": "assistant", "content": f"Response {i}"})
        return {"messages": msgs, "model": "test"}

    @pytest.mark.regression("PROXY-010")
    def test_different_conversations_get_different_sessions(self):
        """Two conversations with different history must not share a session."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        body_a = self._make_body([
            "[Telegram Y id:111] i am the one who knocks",
            "[Telegram Y id:111] lets build an arduino alarm",
            "[Telegram Y id:111] what resistor do i need",
        ])
        body_b = self._make_body([
            "The conversation history was compacted: health discussion",
            "[Telegram G id:-555] check my progesterone levels",
            "[Telegram G id:-555] what about the cycle tracking",
        ])

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            # First call: conversation A
            engine_a = MagicMock()
            engine_a.config.session_id = "session-aaa"
            MockEngine.return_value = engine_a
            state_a, is_new_a = registry.get_or_create(None, body=body_a)

            # Second call: conversation B — must get a DIFFERENT session
            engine_b = MagicMock()
            engine_b.config.session_id = "session-bbb"
            MockEngine.return_value = engine_b
            state_b, is_new_b = registry.get_or_create(None, body=body_b)

        assert is_new_a is True
        assert is_new_b is True
        assert state_a is not state_b, (
            "Different conversations must get different sessions"
        )
        assert registry.session_count == 2

    @pytest.mark.regression("PROXY-010")
    def test_same_conversation_reuses_session_via_fingerprint(self):
        """Same conversation (same first messages) reuses the existing session.

        Uses 6+ user messages so the first 5 (the fingerprint sample) are
        stable across both request bodies.
        """
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        base_msgs = [
            "[Telegram Y id:111] hello world",
            "[Telegram Y id:111] tell me about cars",
            "[Telegram Y id:111] what about trucks",
            "[Telegram Y id:111] and planes too",
            "[Telegram Y id:111] also ships",
            "[Telegram Y id:111] what about trains",
        ]
        body_v1 = self._make_body(base_msgs)
        # Same conversation, one more message appended — first 5 unchanged
        body_v2 = self._make_body(base_msgs + [
            "[Telegram Y id:111] and buses too",
        ])

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            engine = MagicMock()
            engine.config.session_id = "session-xxx"
            MockEngine.return_value = engine
            state_1, is_new_1 = registry.get_or_create(None, body=body_v1)

            state_2, is_new_2 = registry.get_or_create(None, body=body_v2)

        assert is_new_1 is True
        assert is_new_2 is False
        assert state_1 is state_2, (
            "Same conversation with appended messages must reuse session"
        )
        assert registry.session_count == 1

    @pytest.mark.regression("PROXY-010")
    def test_marker_takes_priority_over_fingerprint(self):
        """When a session marker exists, it takes priority over fingerprint."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        engine = MagicMock()
        engine.config.session_id = "marker-session"
        state = ProxyState(engine, metrics=metrics)
        registry._sessions["marker-session"] = state

        # Body from a different conversation — but marker overrides
        body = self._make_body(["completely different conversation"])

        result, is_new = registry.get_or_create("marker-session", body=body)
        assert is_new is False
        assert result is state


# ---------------------------------------------------------------------------
# SessionState machine
# ---------------------------------------------------------------------------


class TestSessionStateMachine:
    """Tests for the non-blocking ingestion state machine."""

    def _make_state(self, *, session_id="test-session", metrics=None):
        engine = MagicMock()
        engine.config.session_id = session_id
        engine._turn_tag_index = TurnTagIndex()
        engine._store = MagicMock()
        engine._store.get_all_tags.return_value = []
        engine.config.tag_generator.context_lookback_pairs = 3
        engine.config.tag_generator.context_bleed_threshold = 0
        if metrics is None:
            metrics = ProxyMetrics()
        return ProxyState(engine, metrics=metrics)

    def test_initial_state_is_active(self):
        state = self._make_state()
        assert state.session_state == SessionState.ACTIVE

    def test_start_ingestion_transitions_to_ingesting(self):
        state = self._make_state()
        state.engine.ingest_history.return_value = 2
        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
            Message(role="assistant", content="A2"),
        ]
        state.start_ingestion_if_needed(pairs)
        # After starting, state should be INGESTING (thread still running)
        assert state._state in (SessionState.INGESTING, SessionState.ACTIVE)

    def test_ingestion_completes_transitions_to_active(self):
        state = self._make_state()
        state.engine.ingest_history.return_value = 1
        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        state.start_ingestion_if_needed(pairs)
        # Wait for background thread
        import time
        time.sleep(0.5)
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

    def test_manual_passthrough_overrides_state(self):
        state = self._make_state()
        assert state.session_state == SessionState.ACTIVE
        state.set_manual_passthrough(True)
        assert state.session_state == SessionState.PASSTHROUGH
        # Internal state unchanged
        assert state._state == SessionState.ACTIVE

    def test_manual_passthrough_disable_restores(self):
        state = self._make_state()
        state.set_manual_passthrough(True)
        assert state.session_state == SessionState.PASSTHROUGH
        state.set_manual_passthrough(False)
        assert state.session_state == SessionState.ACTIVE

    def test_empty_history_stays_active(self):
        state = self._make_state()
        state.start_ingestion_if_needed([])
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

    def test_state_change_event_emitted(self):
        metrics = ProxyMetrics()
        state = self._make_state(metrics=metrics)
        state._transition_to(SessionState.INGESTING)
        events = metrics.events_since(-1)
        state_events = [e for e in events if e["type"] == "session_state_change"]
        assert len(state_events) == 1
        assert state_events[0]["from"] == "active"
        assert state_events[0]["to"] == "ingesting"

    def test_persisted_state_skips_ingestion(self):
        state = self._make_state()
        # Simulate persisted index covering the history
        for i in range(3):
            state.engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i}",
                tags=["tag"], primary_tag="tag",
            ))
        pairs = [
            Message(role="user", content=f"Q{i}")
            if i % 2 == 0
            else Message(role="assistant", content=f"A{i}")
            for i in range(6)  # 3 pairs
        ]
        state.start_ingestion_if_needed(pairs)
        # Should skip ingestion, stay ACTIVE
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True
        state.engine.ingest_history.assert_not_called()

    def test_ingestion_error_transitions_to_active(self):
        state = self._make_state()
        state.engine.ingest_history.side_effect = RuntimeError("boom")
        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        state.start_ingestion_if_needed(pairs)
        import time
        time.sleep(0.5)
        # Even on error, should end up ACTIVE (not stuck in INGESTING)
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

    @pytest.mark.regression("PROXY-013")
    def test_second_call_during_ingestion_does_not_restart(self):
        """Second start_ingestion_if_needed during INGESTING should not
        spawn a duplicate thread or reset progress to turn 0."""
        import hashlib
        import time

        ingestion_calls = []
        progress_events = []

        def slow_ingest(pairs, progress_callback=None):
            """Simulate slow ingestion — record each call's pair count."""
            ingestion_calls.append(len(pairs) // 2)
            # Simulate tagging each pair
            n = len(pairs) // 2
            for i in range(n):
                combined = f"{pairs[i*2].content} {pairs[i*2+1].content}"
                entry = TurnTagEntry(
                    turn_number=len(state.engine._turn_tag_index.entries),
                    message_hash=hashlib.sha256(combined.encode()).hexdigest()[:16],
                    tags=[f"tag-{i}"],
                    primary_tag=f"tag-{i}",
                )
                state.engine._turn_tag_index.append(entry)
                if progress_callback:
                    progress_callback(i + 1, n, entry)
                time.sleep(0.05)  # simulate work
            return n

        state = self._make_state()
        state.engine.ingest_history.side_effect = slow_ingest

        # 6 pairs = 12 messages
        pairs = [
            Message(role=("user" if j % 2 == 0 else "assistant"), content=f"msg-{j}")
            for j in range(12)
        ]

        # First call — starts background ingestion
        state.start_ingestion_if_needed(pairs)
        assert state._state == SessionState.INGESTING

        # Wait for partial progress (should be mid-ingestion)
        time.sleep(0.15)
        progress_at_second_call = state._ingestion_progress[0]
        assert progress_at_second_call > 0, "Should have made some progress"

        # Second call — should NOT restart from 0
        state.start_ingestion_if_needed(pairs)

        # Wait for completion
        time.sleep(2.0)
        assert state.session_state == SessionState.ACTIVE

        # KEY ASSERTIONS:
        # 1. Two ingestion calls: first was cancelled, second picked up remainder
        assert len(ingestion_calls) == 2, f"Expected 2 calls, got {ingestion_calls}"

        # 2. The second call should only ingest the REMAINING turns,
        #    not restart from 0. Total across both should equal 6.
        assert sum(ingestion_calls) < 12, (
            f"Second call restarted from 0: {ingestion_calls}"
        )
        assert ingestion_calls[1] < ingestion_calls[0], (
            f"Second call should be smaller (resumed): {ingestion_calls}"
        )

        # 3. All 6 turns should be tagged (no gaps, no duplicates)
        assert len(state.engine._turn_tag_index.entries) == 6

    @pytest.mark.regression("PROXY-014")
    def test_third_call_during_ingestion_cancels_second(self):
        """Third start_ingestion_if_needed should cancel the second
        thread and resume, not silently return because the session
        was erroneously marked as ingested by the first thread's
        finally block."""
        import hashlib
        import time

        ingestion_calls = []

        def slow_ingest(pairs, progress_callback=None):
            """Simulate slow ingestion — record each call's pair count."""
            ingestion_calls.append(len(pairs) // 2)
            n = len(pairs) // 2
            for i in range(n):
                combined = f"{pairs[i*2].content} {pairs[i*2+1].content}"
                entry = TurnTagEntry(
                    turn_number=len(state.engine._turn_tag_index.entries),
                    message_hash=hashlib.sha256(combined.encode()).hexdigest()[:16],
                    tags=[f"tag-{i}"],
                    primary_tag=f"tag-{i}",
                )
                state.engine._turn_tag_index.append(entry)
                if progress_callback:
                    progress_callback(i + 1, n, entry)
                time.sleep(0.08)  # slow enough to ensure cancel mid-flight
            return n

        state = self._make_state()
        state.engine.ingest_history.side_effect = slow_ingest

        # 20 pairs = 40 messages (takes ~1.6s per full run, plenty of time)
        pairs = [
            Message(role=("user" if j % 2 == 0 else "assistant"), content=f"msg-{j}")
            for j in range(40)
        ]

        # First call — starts background ingestion
        state.start_ingestion_if_needed(pairs)
        assert state._state == SessionState.INGESTING

        # Wait for partial progress
        time.sleep(0.3)
        p1 = state._ingestion_progress[0]
        assert p1 > 0, "Should have made some progress after call 1"

        # Second call — cancels first, resumes
        state.start_ingestion_if_needed(pairs)
        thread_after_second = state._ingestion_thread

        # Wait for partial progress from second thread
        time.sleep(0.3)

        # KEY CHECK: the cancelled first thread's finally block should
        # NOT have marked the session as ingested.  If it did, this call
        # will silently return without cancelling the second thread.
        session_id = state.engine.config.session_id
        was_marked = session_id in state._ingested_sessions
        state.start_ingestion_if_needed(pairs)
        thread_after_third = state._ingestion_thread

        # The third call should have started a NEW thread
        assert thread_after_third is not thread_after_second, (
            "Third call did not start a new thread — cancelled thread's "
            "finally block erroneously marked session as ingested "
            f"(was_marked={was_marked})"
        )

        # Wait for completion
        time.sleep(5.0)
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

        # Should have had 3 ingestion calls (first two cancelled, third completes)
        assert len(ingestion_calls) == 3, (
            f"Expected 3 ingestion calls, got {ingestion_calls}"
        )

        # All 20 turns should eventually be tagged
        assert len(state.engine._turn_tag_index.entries) == 20

    def test_passthrough_request_has_flag(self):
        """Captured request in passthrough should have passthrough=True."""
        metrics = ProxyMetrics()
        metrics.capture_request(
            0, {"messages": []}, "anthropic",
            passthrough=True,
        )
        req = metrics.get_captured_request(0)
        assert req["passthrough"] is True

    def test_non_passthrough_request_default(self):
        """Default captured request should have passthrough=False."""
        metrics = ProxyMetrics()
        metrics.capture_request(0, {"messages": []}, "anthropic")
        req = metrics.get_captured_request(0)
        assert req["passthrough"] is False


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
            engine._turn_tag_index = TurnTagIndex()
            MockEngine.return_value = engine
            app = create_app(upstream="http://fake:9999", config_path=None)
        with TestClient(app) as client:
            yield client, engine

    def test_toggle_on(self, toggle_client):
        client, engine = toggle_client
        sid = engine.config.session_id
        resp = client.post(
            f"/dashboard/sessions/{sid}/passthrough",
            json={"enabled": True},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_toggle_off(self, toggle_client):
        client, engine = toggle_client
        sid = engine.config.session_id
        # Enable then disable
        client.post(
            f"/dashboard/sessions/{sid}/passthrough",
            json={"enabled": True},
        )
        resp = client.post(
            f"/dashboard/sessions/{sid}/passthrough",
            json={"enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_toggle_unknown_session_404(self, toggle_client):
        client, _ = toggle_client
        resp = client.post(
            "/dashboard/sessions/nonexistent-session/passthrough",
            json={"enabled": True},
        )
        assert resp.status_code == 404

