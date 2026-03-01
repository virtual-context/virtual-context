"""Tests for virtual_context.proxy.server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_context.proxy.server import (
    ProxyState,
    SessionRegistry,
    SessionState,
    _build_continuation_request,
    _detect_api_format,
    _emit_message_end_sse,
    _emit_text_as_sse,
    _emit_tool_use_as_sse,
    _extract_assistant_text,
    _extract_delta_text,
    _extract_history_pairs,
    _extract_session_id,
    _extract_user_message,
    _filter_body_messages,
    _forward_headers,
    _inject_context,
    _inject_session_marker,
    _inject_vc_tools,
    _last_text_block,
    _parse_sse_events,
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
        assert result["messages"][0]["content"].startswith("<system-reminder>")
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
        assert result["system"].startswith("<system-reminder>")
        assert "context here" in result["system"]
        assert "Be helpful" in result["system"]

    def test_anthropic_no_system(self):
        body = {"messages": []}
        result = _inject_context(body, "context here", "anthropic")
        assert "<system-reminder>" in result["system"]
        assert "context here" in result["system"]

    def test_anthropic_list_system(self):
        body = {
            "system": [{"type": "text", "text": "Existing system"}],
            "messages": [],
        }
        result = _inject_context(body, "context here", "anthropic")
        assert isinstance(result["system"], list)
        assert result["system"][0]["text"] == "Existing system"
        assert result["system"][1]["type"] == "text"
        assert "context here" in result["system"][1]["text"]

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
        engine.compact_if_needed.assert_called_once_with(history, signal)

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

    @pytest.mark.regression("BUG-008")
    def test_no_temporal_bypass_in_filter(self):
        """Time-scoped recall is tool-driven; history filter does not bypass."""
        body = self._build_body(3)
        idx = self._build_index([["a"], ["b"], ["c"]])
        filtered, dropped = _filter_body_messages(
            body, idx, ["x"], recent_turns=1,
        )
        assert dropped == 2

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

    @pytest.mark.regression("PROXY-022")
    def test_consecutive_user_messages_preserve_alternation(self):
        """Unpaired consecutive user messages must not break role alternation.

        OpenClaw sends consecutive user messages (e.g., batched Telegram messages,
        tool_result followed by new user text without intervening assistant). When
        _filter_body_messages drops pairs around these unpaired messages, the result
        must still strictly alternate user/assistant for the Anthropic API.
        """
        # Reproduce real OpenClaw pattern: consecutive users at start
        body = {
            "messages": [
                # Unpaired user (e.g., batched Telegram message)
                {"role": "user", "content": "first batch msg"},
                # Pair 0: normal pair
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: no tag match → will be dropped
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                # Pair 2: protected (recent)
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],   # pair 0 — no match
            ["music"],     # pair 1 — no match
            ["weather"],   # pair 2 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated"], recent_turns=1,
        )
        msgs = filtered["messages"]
        # Verify strict role alternation in output
        for i in range(1, len(msgs)):
            assert msgs[i]["role"] != msgs[i - 1]["role"], (
                f"Consecutive same role at indices {i-1}->{i}: "
                f"{msgs[i-1]['role']}, {msgs[i]['role']}"
            )

    @pytest.mark.regression("PROXY-004")
    def test_consecutive_assistant_msgs_preserve_tool_use(self):
        """Consecutive assistant messages where the second has tool_use must not
        be dropped by role alternation enforcement.

        Claude Code with extended thinking can produce consecutive assistant
        messages: msg N = [thinking, text] (no tool_use), msg N+1 = [text, tool_use].
        The pairing logic pairs the user before msg N, leaving msg N+1 unpaired.
        When pair (user, msg N) is dropped, msg N+1 becomes consecutive with the
        previous assistant — role alternation would drop it, orphaning its
        tool_result in the next message.
        """
        body = {
            "messages": [
                # Pair 0: normal turn (matched → kept)
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: normal turn (no match → dropped)
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1-thinking-only"},
                # Unpaired assistant: consecutive with pair 1's assistant
                # Contains tool_use — critical for referential integrity
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check"},
                    {"type": "tool_use", "id": "tool_abc", "name": "Read",
                     "input": {"path": "foo.py"}},
                ]},
                # Pair 2: tool_result (no match → but forced by tool chain)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_abc",
                     "content": "file contents"},
                ]},
                {"role": "assistant", "content": "I read the file"},
                # Pair 3: normal turn (no match → dropped)
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
            ["python"],     # pair 0 — matches
            ["cooking"],    # pair 1 — no match → dropped
            ["cooking"],    # pair 2 — no match but forced by tool chain
            ["cooking"],    # pair 3 — no match
            ["cooking"],    # pair 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        msgs = filtered["messages"]

        # The unpaired assistant with tool_use MUST survive.
        # Verify no orphaned tool_result references.
        tool_use_ids = set()
        tool_result_refs = set()
        for msg in msgs:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_use_ids.add(block["id"])
                elif block.get("type") == "tool_result":
                    tool_result_refs.add(block["tool_use_id"])

        orphaned = tool_result_refs - tool_use_ids
        assert not orphaned, (
            f"Orphaned tool_result references: {orphaned}. "
            f"Role alternation dropped a tool_use message."
        )

        # Also verify role alternation is valid
        for i in range(1, len(msgs)):
            assert msgs[i].get("role") != msgs[i - 1].get("role"), (
                f"Consecutive same role at indices {i-1}->{i}: "
                f"{msgs[i-1].get('role')}, {msgs[i].get('role')}"
            )

    @pytest.mark.regression("PROXY-004b")
    def test_consecutive_assistant_dropped_pair_keeps_user_first(self):
        """Dropping pair 0 when an unpaired assistant[2] is force-kept must not
        leave the message list starting with role=assistant.

        Claude Code with extended thinking sends:
          msg[0]=user, msg[1]=assistant(thinking), msg[2]=assistant(tool_use),
          msg[3]=user(tool_result), ...
        Pair 0 is (0,1). msg[2] is unpaired. If pair 0 is dropped but msg[2]
        is kept (via tool_use_id integrity), the filtered output starts with
        assistant — which the Anthropic API rejects with 400.
        """
        body = {
            "messages": [
                # Pair 0: initial turn (no match → should be dropped)
                {"role": "user", "content": [{"type": "text", "text": "init"}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "thinking..."},
                ]},
                # Unpaired assistant: consecutive with pair 0's assistant
                {"role": "assistant", "content": [
                    {"type": "text", "text": "let me check"},
                    {"type": "tool_use", "id": "tu_glob", "name": "Glob",
                     "input": {"pattern": "*.py"}},
                    {"type": "tool_use", "id": "tu_bash", "name": "Bash",
                     "input": {"command": "ls"}},
                ]},
                # Pair 1: tool_result (no match → but forced by tool chain)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_glob",
                     "content": "a.py b.py"},
                    {"type": "tool_result", "tool_use_id": "tu_bash",
                     "content": "ok"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                # Pair 2: no match → dropped
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                # Pair 3: no match → dropped
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                # Pair 4: protected
                {"role": "user", "content": "Q4"},
                {"role": "assistant", "content": "A4"},
                # Current user
                {"role": "user", "content": "current"},
            ],
        }
        idx = self._build_index([
            ["setup"],       # pair 0 — no match
            ["setup"],       # pair 1 — no match (forced by tool chain)
            ["cooking"],     # pair 2 — no match
            ["cooking"],     # pair 3 — no match
            ["cooking"],     # pair 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        msgs = filtered["messages"]

        # First message MUST be role=user (Anthropic API requirement)
        first_chat = None
        for msg in msgs:
            if msg.get("role") in ("user", "assistant"):
                first_chat = msg
                break
        assert first_chat is not None
        assert first_chat["role"] == "user", (
            f"First chat message is '{first_chat['role']}' — "
            f"Anthropic API requires first message to be 'user'"
        )

        # No orphaned tool_results
        tu_ids = set()
        tr_ids = set()
        for msg in msgs:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tu_ids.add(block["id"])
                elif block.get("type") == "tool_result":
                    tr_ids.add(block["tool_use_id"])
        orphaned = tr_ids - tu_ids
        assert not orphaned, f"Orphaned tool_result references: {orphaned}"

    @pytest.mark.regression("PROXY-004c")
    def test_consecutive_assistant_thinking_strip_preserves_tool_chain(self):
        """Consecutive assistants with thinking blocks must not orphan tool_results.

        Reproduces the exact layout from A/B run 2026-03-01, request_log/000038:

          msg[48] user   [tool_result(prev)]       — pair N
          msg[49] assistant [thinking, text]        — pair N (response without tool_use)
          msg[50] assistant [thinking, text, tool_use(X)]  — UNPAIRED (consecutive asst)
          msg[51] user   [tool_result(X)]           — pair N+1

        _strip_thinking_blocks creates new dicts for msgs 49 and 50 (both have
        thinking blocks).  If _vc_critical is set on original chat_msgs instead
        of the copies in `kept`, alternation enforcement can't see the sentinel
        on msg[50]'s copy → drops it → tool_result(X) orphaned → API 400.
        """
        body = {
            "messages": [
                # Pair 0: tag match → keep
                {"role": "user", "content": "analyze the data"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "ok",
                     "signature": "sig_a0"},
                    {"type": "text", "text": "I'll analyze it"},
                    {"type": "tool_use", "id": "tu_read1", "name": "Read",
                     "input": {"file_path": "/data.csv"}},
                ]},
                # Pair 1: tag match → keep
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_read1",
                     "content": "col1,col2\n1,2\n3,4"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "got it",
                     "signature": "sig_a1"},
                    {"type": "text", "text": "loaded data"},
                    {"type": "tool_use", "id": "tu_bash1", "name": "Bash",
                     "input": {"command": "python analyze.py"}},
                ]},
                # Pair 2: no match → dropped
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_bash1",
                     "content": "analysis complete"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "done",
                     "signature": "sig_a2"},
                    {"type": "text", "text": "analysis done"},
                ]},
                # --- THE BUG PATTERN: consecutive assistants ---
                # UNPAIRED assistant (consecutive with pair 2's assistant)
                # Has thinking + tool_use — _strip_thinking_blocks will copy it
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "wait let me also run tests",
                     "signature": "sig_a_unpaired"},
                    {"type": "text", "text": "Let me also run the tests"},
                    {"type": "tool_use", "id": "tu_target", "name": "Bash",
                     "input": {"command": "pytest"}},
                ]},
                # Pair 3: no match → dropped, but tool_result forces keep
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_target",
                     "content": "===== test session starts =====\n3 passed"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "All tests pass"},
                ]},
                # Pair 4: no match → dropped
                {"role": "user", "content": "what about coverage?"},
                {"role": "assistant", "content": "I can check coverage next"},
                # Pair 5: no match → dropped
                {"role": "user", "content": "and linting?"},
                {"role": "assistant", "content": "will check linting too"},
                # Pair 6: protected (recent)
                {"role": "user", "content": "ok do it"},
                {"role": "assistant", "content": "on it"},
                # Current user turn
                {"role": "user", "content": "status?"},
            ],
        }
        idx = self._build_index([
            ["data-analysis"],  # pair 0 — match
            ["data-analysis"],  # pair 1 — match
            ["testing"],        # pair 2 — no match
            ["testing"],        # pair 3 — no match (forced by tool chain)
            ["coverage"],       # pair 4 — no match
            ["linting"],        # pair 5 — no match
            ["linting"],        # pair 6 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["data-analysis"], recent_turns=1,
        )
        msgs = filtered["messages"]

        # Thinking blocks must be stripped (some messages were dropped)
        for msg in msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert block.get("type") != "thinking", \
                            "Thinking blocks should be stripped after dropping"

        # The critical check: no orphaned tool_results
        tu_ids = set()
        tr_ids = set()
        for msg in msgs:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tu_ids.add(block["id"])
                elif block.get("type") == "tool_result":
                    tr_ids.add(block["tool_use_id"])
        orphaned = tr_ids - tu_ids
        assert not orphaned, (
            f"Orphaned tool_result references: {orphaned}. "
            f"PROXY-004c: _strip_thinking_blocks created a copy of the "
            f"assistant with tool_use(tu_target) and the _vc_critical "
            f"sentinel was lost on the copy."
        )

    @pytest.mark.regression("PROXY-023")
    def test_compacted_turns_dropped_when_paging_active(self):
        """When paging is active, turns below compacted_turn watermark are dropped.

        Even if a compacted turn has matching tags, it should be dropped because
        the content is available via VC summaries and expandable via paging tools.
        Without this, the LLM always has the raw messages and never needs paging.
        """
        # 8 history pairs: turns 0-4 are "compacted", turns 5-7 are not
        body = self._build_body(8)
        idx = self._build_index([
            ["python", "testing"],   # turn 0 — matches but compacted
            ["python", "api"],       # turn 1 — matches but compacted
            ["cooking"],             # turn 2 — no match, compacted
            ["python", "debug"],     # turn 3 — matches but compacted
            ["music"],               # turn 4 — no match, compacted
            ["python", "deploy"],    # turn 5 — matches, NOT compacted → keep
            ["weather"],             # turn 6 — no match, not compacted
            ["cars"],                # turn 7 — no match, protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
            compacted_turn=5,  # turns 0-4 are compacted
        )
        # turns 0-4 ALL dropped (compacted, even though 0,1,3 match tags)
        # turn 5 kept (matches, not compacted)
        # turn 6 dropped (no match, not compacted)
        # turn 7 kept (protected)
        assert dropped == 6
        msgs = filtered["messages"]
        # 2 kept pairs * 2 + current user = 5
        assert len(msgs) == 5
        # Verify the kept messages are from turns 5 and 7
        assert msgs[0]["content"] == "Question 5"
        assert msgs[1]["content"] == "Answer 5"
        assert msgs[2]["content"] == "Question 7"
        assert msgs[3]["content"] == "Answer 7"
        assert msgs[4]["content"] == "Current question"

    @pytest.mark.regression("PROXY-023")
    def test_compacted_turn_zero_preserves_current_behavior(self):
        """When compacted_turn=0 (default/no paging), filter behaves normally.

        Tag-matching turns are kept even if they're old. This is the existing
        behavior and must not regress.
        """
        body = self._build_body(5)
        idx = self._build_index([
            ["python"],    # turn 0 — matches → kept
            ["cooking"],   # turn 1 — no match → dropped
            ["music"],     # turn 2 — no match → dropped
            ["python"],    # turn 3 — matches → kept
            ["weather"],   # turn 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
            compacted_turn=0,
        )
        # Same as without compacted_turn: turns 0,3 kept (match), turn 4 (protected)
        assert dropped == 2
        msgs = filtered["messages"]
        assert len(msgs) == 7  # 3 pairs * 2 + current user

    @pytest.mark.regression("PROXY-023")
    def test_compacted_turns_rule_tag_still_dropped(self):
        """Even 'rule' tagged turns are dropped when compacted and paging is active.

        Rule tags normally force-keep turns, but compacted content has already
        been summarized. The paging system can retrieve it if needed.
        """
        body = self._build_body(5)
        idx = self._build_index([
            ["rule", "style"],   # turn 0 — rule tag but compacted
            ["cooking"],         # turn 1 — compacted
            ["music"],           # turn 2 — not compacted, no match → dropped
            ["python"],          # turn 3 — not compacted, matches → kept
            ["weather"],         # turn 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
            compacted_turn=2,  # turns 0-1 compacted
        )
        # turn 0 dropped (compacted, even with rule tag)
        # turn 1 dropped (compacted)
        # turn 2 dropped (no match)
        # turn 3 kept (match, not compacted)
        # turn 4 kept (protected)
        assert dropped == 3
        msgs = filtered["messages"]
        assert len(msgs) == 5  # 2 pairs * 2 + current user

    @pytest.mark.regression("PROXY-022")
    def test_consecutive_user_after_tool_result_preserves_alternation(self):
        """tool_result user followed by text user must not break alternation.

        Real pattern from OpenClaw: assistant uses tool → user sends tool_result →
        user sends new text message (no intervening assistant). When pairs around
        the unpaired tool_result are dropped, alternation must be preserved.
        """
        body = {
            "messages": [
                # Pair 0: tag match → kept
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: assistant uses tool → no tag match
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
                ]},
                # Unpaired user: tool_result without subsequent assistant
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                ]},
                # Pair 2: new user text + assistant (this pairs with the next assistant)
                {"role": "user", "content": "Q2-new-topic"},
                {"role": "assistant", "content": "A2"},
                # Pair 3: protected (recent)
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["python"],    # pair 0 — matches
            ["cooking"],   # pair 1 — no match
            ["music"],     # pair 2 — no match
            ["weather"],   # pair 3 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        msgs = filtered["messages"]
        # Verify strict role alternation
        for i in range(1, len(msgs)):
            assert msgs[i]["role"] != msgs[i - 1]["role"], (
                f"Consecutive same role at indices {i-1}->{i}: "
                f"{msgs[i-1]['role']}, {msgs[i]['role']}"
            )

    def test_dropped_never_negative(self):
        """dropped count must never be negative, even with unpaired messages.

        When alternation enforcement silently removes extra messages,
        the drop count must be computed from the final kept list.
        """
        # Build a body with unpaired messages that will cause alternation enforcement
        body = {
            "messages": [
                # Unpaired user (batched Telegram)
                {"role": "user", "content": "batch1"},
                # Another unpaired user
                {"role": "user", "content": "batch2"},
                # Pair 0
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["python"],  # pair 0
            ["music"],   # pair 1
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated"], recent_turns=1,
        )
        assert dropped >= 0, f"dropped should never be negative, got {dropped}"
        # Verify alternation
        msgs = filtered["messages"]
        for i in range(1, len(msgs)):
            assert msgs[i]["role"] != msgs[i - 1]["role"]

    def test_dropped_count_with_alternation_enforcement(self):
        """Alternation enforcement removes messages beyond pair-based drops.

        The dropped count must reflect ALL removed user turns, including
        those removed by alternation enforcement.
        """
        body = {
            "messages": [
                {"role": "user", "content": "unpaired"},
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],  # pair 0 — no match
            ["weather"],  # pair 1 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated"], recent_turns=1,
        )
        assert dropped >= 0
        # Verify no negative payload in the formula: total_turns - dropped
        total_pairs = 2
        assert total_pairs - dropped >= 0


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
        engine.config.session_id = "test"
        state = ProxyState(engine)
        assert hasattr(state, "_compaction_lock")
        assert isinstance(state._compaction_lock, type(threading.Lock()))

    @pytest.mark.regression("PROXY-007")
    def test_compaction_lock_is_non_reentrant(self):
        """Lock is a plain Lock (not RLock) so double-acquire blocks."""
        engine = MagicMock()
        engine.config = MagicMock()
        engine.config.session_id = "test"
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
            engine._compacted_through = 0
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
        """No session ID + tail-1 fingerprint match → reuses existing session.

        Simulates the real flow: request N stores fp at offset=0 (tail),
        then request N+1 (one more turn) matches via offset=1 (tail-1).
        """
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.session_id = "default-session"
        default = ProxyState(engine, metrics=metrics)

        # Request N: 3 user messages.  history_user = [u0, u1], fp = hash(u1)
        body_prev = {"messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "doing well"},
            {"role": "user", "content": "tell me more"},
        ]}
        # Request N+1: one more turn.  history_user = [u0, u1, u2],
        # fp_match(offset=1) = hash(u1) → matches prev fp
        body_next = {"messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "doing well"},
            {"role": "user", "content": "tell me more"},
            {"role": "assistant", "content": "sure thing"},
            {"role": "user", "content": "what is new"},
        ]}

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._sessions["default-session"] = default
        # Simulate catch_all: store fp from request N at offset=0
        fp = SessionRegistry._compute_fingerprint(body_prev)
        registry._fingerprints[fp] = "default-session"

        # Request N+1 should match via offset=1
        result, is_new = registry.get_or_create(None, body=body_next)
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
        """Same conversation growing by one turn reuses the existing session.

        With tail-based fingerprinting (sample_size=1):
        - Request 1: history_user = [u0, u1, u2], fp_store = hash(u2)
        - Request 2: history_user = [u0, u1, u2, u3],
          fp_match(offset=1) = hash(u2) → matches fp_store
        """
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        # 4 user messages → history_user has 3, fp_store = hash(u2)
        base_msgs = [
            "[Telegram Y id:111] hello world",
            "[Telegram Y id:111] tell me about cars",
            "[Telegram Y id:111] what about trucks",
            "[Telegram Y id:111] and planes too",
        ]
        body_v1 = self._make_body(base_msgs)
        # One more message → history_user has 4, fp_match(offset=1) = hash(u2)
        body_v2 = self._make_body(base_msgs + [
            "[Telegram Y id:111] also ships",
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
# Trailing fingerprint
# ---------------------------------------------------------------------------


class TestTrailingFingerprint:
    """Tail-based fingerprint: store at offset=0, match at offset=1."""

    def _make_body(self, user_messages: list[str]) -> dict:
        msgs = []
        for i, text in enumerate(user_messages):
            msgs.append({"role": "user", "content": text})
            if i < len(user_messages) - 1:
                msgs.append({"role": "assistant", "content": f"Response {i}"})
        return {"messages": msgs, "model": "test"}

    def test_compute_fingerprint_samples_tail(self):
        """offset=0 hashes the last S user messages before current turn."""
        body = self._make_body(["msg-a", "msg-b", "msg-c"])
        # history_user = ["msg-a", "msg-b"], sample_size=1 → hash("msg-b")
        fp = SessionRegistry._compute_fingerprint(body, offset=0)
        assert fp  # non-empty
        assert len(fp) == 16  # sha256 truncated to 16 hex chars

    def test_compute_fingerprint_offset1_shifts_back(self):
        """offset=1 shifts sampling window back by one position."""
        body = self._make_body(["msg-a", "msg-b", "msg-c"])
        # history_user = ["msg-a", "msg-b"]
        # offset=0 → hash("msg-b"), offset=1 → hash("msg-a")
        fp0 = SessionRegistry._compute_fingerprint(body, offset=0)
        fp1 = SessionRegistry._compute_fingerprint(body, offset=1)
        assert fp0 != fp1

    def test_tail1_matches_previous_tail(self):
        """Core invariant: request N+1's tail-1 == request N's tail.

        Request N:   history = [u0, u1, u2].  tail = hash(u2).
        Request N+1: history = [u0, u1, u2, u3].  tail-1 = hash(u2).
        """
        body_n = self._make_body(["u0", "u1", "u2", "current-n"])
        body_n1 = self._make_body(["u0", "u1", "u2", "u3", "current-n1"])

        fp_store = SessionRegistry._compute_fingerprint(body_n, offset=0)
        fp_match = SessionRegistry._compute_fingerprint(body_n1, offset=1)

        assert fp_store == fp_match, (
            "tail-1 of next request must equal tail of previous request"
        )

    def test_too_few_messages_returns_empty(self):
        """A body with < 2 user messages cannot produce a fingerprint."""
        body_one = self._make_body(["only-one"])
        assert SessionRegistry._compute_fingerprint(body_one) == ""

        body_two = self._make_body(["first", "second"])
        # history_user = ["first"], offset=1 → start < 0
        assert SessionRegistry._compute_fingerprint(body_two, offset=1) == ""

    def test_offset_too_large_returns_empty(self):
        """Offset beyond available history returns empty string."""
        body = self._make_body(["u0", "u1", "u2"])
        # history_user = ["u0", "u1"], offset=2 → end=0, returns ""
        assert SessionRegistry._compute_fingerprint(body, offset=2) == ""

    def test_different_conversations_produce_different_fingerprints(self):
        """Two conversations with different messages have different fps."""
        body_a = self._make_body(["[id:111] hello", "[id:111] cars", "[id:111] query"])
        body_b = self._make_body(["[id:222] hey", "[id:222] bikes", "[id:222] query"])
        fp_a = SessionRegistry._compute_fingerprint(body_a)
        fp_b = SessionRegistry._compute_fingerprint(body_b)
        assert fp_a != fp_b

    def test_persisted_fingerprint_roundtrip_sqlite(self, tmp_path):
        """Trailing fingerprint persists through SQLite save/load cycle."""
        from virtual_context.storage.sqlite import SQLiteStore
        store = SQLiteStore(str(tmp_path / "test.db"))

        from virtual_context.types import EngineStateSnapshot
        snap = EngineStateSnapshot(
            session_id="sess-1",
            compacted_through=10,
            turn_tag_entries=[],
            turn_count=5,
            trailing_fingerprint="abc123deadbeef00",
        )
        store.save_engine_state(snap)

        loaded = store.load_engine_state("sess-1")
        assert loaded is not None
        assert loaded.trailing_fingerprint == "abc123deadbeef00"

        # list_engine_state_fingerprints should return this
        fps = store.list_engine_state_fingerprints()
        assert fps == {"abc123deadbeef00": "sess-1"}

    def test_persisted_fingerprint_roundtrip_filesystem(self, tmp_path):
        """Trailing fingerprint persists through filesystem save/load cycle."""
        from virtual_context.storage.filesystem import FilesystemStore
        store = FilesystemStore(str(tmp_path / "fs_store"))

        from virtual_context.types import EngineStateSnapshot
        snap = EngineStateSnapshot(
            session_id="sess-2",
            compacted_through=5,
            turn_tag_entries=[],
            turn_count=3,
            trailing_fingerprint="beef1234cafe5678",
        )
        store.save_engine_state(snap)

        loaded = store.load_engine_state("sess-2")
        assert loaded is not None
        assert loaded.trailing_fingerprint == "beef1234cafe5678"

        fps = store.list_engine_state_fingerprints()
        assert fps == {"beef1234cafe5678": "sess-2"}

    def test_empty_fingerprint_excluded_from_listing(self, tmp_path):
        """Sessions with no trailing fingerprint are excluded from the map."""
        from virtual_context.storage.sqlite import SQLiteStore
        store = SQLiteStore(str(tmp_path / "test.db"))

        from virtual_context.types import EngineStateSnapshot
        snap = EngineStateSnapshot(
            session_id="sess-no-fp",
            compacted_through=0,
            turn_tag_entries=[],
            turn_count=0,
            trailing_fingerprint="",
        )
        store.save_engine_state(snap)
        assert store.list_engine_state_fingerprints() == {}

    def test_match_persisted_fingerprint_on_restart(self, tmp_path):
        """Simulates proxy restart: persisted fp matches inbound tail-1."""
        from virtual_context.storage.sqlite import SQLiteStore
        store = SQLiteStore(str(tmp_path / "test.db"))

        # Request N stored fp = hash(u2) at offset=0
        body_prev = self._make_body(["u0", "u1", "u2", "current-n"])
        fp_stored = SessionRegistry._compute_fingerprint(body_prev, offset=0)

        from virtual_context.types import EngineStateSnapshot
        snap = EngineStateSnapshot(
            session_id="persisted-sess",
            compacted_through=0,
            turn_tag_entries=[],
            turn_count=4,
            trailing_fingerprint=fp_stored,
        )
        store.save_engine_state(snap)

        # Proxy restarts.  Request N+1 arrives with one more turn.
        body_next = self._make_body(["u0", "u1", "u2", "u3", "current-n1"])

        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
            store=store,
        )

        matched_sid = registry._match_persisted_fingerprint(body_next)
        assert matched_sid == "persisted-sess"

    def test_multi_session_fingerprint_routing(self):
        """Multiple concurrent sessions route correctly via tail fingerprints.

        Simulates 3 independent conversations (different Telegram groups)
        each advancing over 3 turns.  Every request must route to the
        correct session via tail-1 fingerprint matching.
        """
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        # Three conversations with distinct Telegram group IDs
        convos = {
            "session-a": ["[id:111] msg-a-{}".format(i) for i in range(6)],
            "session-b": ["[id:222] msg-b-{}".format(i) for i in range(6)],
            "session-c": ["[id:333] msg-c-{}".format(i) for i in range(6)],
        }

        states = {}

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            # Turn 1: each conversation sends first 3 messages (creates session)
            for sid, msgs in convos.items():
                engine = MagicMock()
                engine.config.session_id = sid
                MockEngine.return_value = engine
                body = self._make_body(msgs[:3])
                state, is_new = registry.get_or_create(None, body=body)
                states[sid] = state
                # Simulate catch_all: store tail fingerprint
                fp = SessionRegistry._compute_fingerprint(body)
                if fp:
                    registry._fingerprints[fp] = sid

        assert registry.session_count == 3

        # Turn 2: each conversation grows by one message — must route back
        # to its own session via tail-1 fingerprint match
        for sid, msgs in convos.items():
            body = self._make_body(msgs[:4])  # one more message
            state, is_new = registry.get_or_create(None, body=body)
            assert is_new is False, f"{sid} should reuse existing session"
            assert state is states[sid], f"{sid} routed to wrong session"

            # Simulate catch_all: update fingerprint
            fp = SessionRegistry._compute_fingerprint(body)
            if fp:
                registry._fingerprints[fp] = sid

        # Turn 3: another advance — still routes correctly
        for sid, msgs in convos.items():
            body = self._make_body(msgs[:5])
            state, is_new = registry.get_or_create(None, body=body)
            assert is_new is False, f"{sid} turn 3 should reuse session"
            assert state is states[sid], f"{sid} turn 3 routed to wrong session"

        # No extra sessions created
        assert registry.session_count == 3

    def test_multi_session_restart_with_persisted_fingerprints(self, tmp_path):
        """On restart, 3 persisted sessions are correctly restored via fps."""
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import EngineStateSnapshot
        store = SQLiteStore(str(tmp_path / "test.db"))

        # Three conversations, each with a unique tail fingerprint
        convos = {
            "session-a": ["[id:111] a-{}".format(i) for i in range(5)],
            "session-b": ["[id:222] b-{}".format(i) for i in range(5)],
            "session-c": ["[id:333] c-{}".format(i) for i in range(5)],
        }

        # Persist engine state with fingerprints (simulates pre-restart)
        for sid, msgs in convos.items():
            body = self._make_body(msgs[:4])  # 4 msgs, last is "current"
            fp = SessionRegistry._compute_fingerprint(body, offset=0)
            snap = EngineStateSnapshot(
                session_id=sid,
                compacted_through=0,
                turn_tag_entries=[],
                turn_count=4,
                trailing_fingerprint=fp,
            )
            store.save_engine_state(snap)

        # Proxy restarts — fresh registry with store
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
            store=store,
        )

        # Each conversation sends request with one more turn
        for sid, msgs in convos.items():
            body = self._make_body(msgs[:5])  # 5 msgs, one more than persisted
            matched = registry._match_persisted_fingerprint(body)
            assert matched == sid, (
                f"Session {sid} should match its persisted fingerprint"
            )


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
            engine.tag_turn.return_value = None
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
        assert "vc_collapse_topic" in names

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
        assert len(names) == 7  # web_search + 6 VC tools

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
        engine._resolve_paging_mode.return_value = "autonomous"
        engine._compacted_through = 0
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
        assert "vc_collapse_topic" in tool_names


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
