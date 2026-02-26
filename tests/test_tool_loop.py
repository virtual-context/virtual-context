"""Tests for virtual_context.core.tool_loop."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.core.tool_loop import (
    VC_TOOL_NAMES,
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAICodexAdapter,
    execute_vc_tool,
    get_adapter,
    is_vc_tool,
    run_tool_loop,
    vc_tool_definitions,
)


# ---------------------------------------------------------------------------
# TestVCToolDefinitions (migrated from test_proxy.py)
# ---------------------------------------------------------------------------

class TestVCToolDefinitions:
    """Tests for vc_tool_definitions()."""

    def test_returns_six_tools(self):
        defs = vc_tool_definitions()
        assert len(defs) == 6

    def test_tool_names_have_vc_prefix(self):
        defs = vc_tool_definitions()
        names = {d["name"] for d in defs}
        assert names == {
            "vc_expand_topic",
            "vc_collapse_topic",
            "vc_find_quote",
            "vc_query_facts",
            "vc_recall_all",
            "vc_remember_when",
        }

    def test_tools_have_input_schema(self):
        defs = vc_tool_definitions()
        for d in defs:
            assert "input_schema" in d
            schema = d["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_expand_has_depth_enum(self):
        defs = vc_tool_definitions()
        expand = [d for d in defs if d["name"] == "vc_expand_topic"][0]
        depth = expand["input_schema"]["properties"]["depth"]
        assert set(depth["enum"]) == {"segments", "full"}

    def test_collapse_has_depth_enum(self):
        defs = vc_tool_definitions()
        collapse = [d for d in defs if d["name"] == "vc_collapse_topic"][0]
        depth = collapse["input_schema"]["properties"]["depth"]
        assert set(depth["enum"]) == {"summary", "none"}


# ---------------------------------------------------------------------------
# TestIsVCTool (migrated from test_proxy.py)
# ---------------------------------------------------------------------------

class TestIsVCTool:
    """Tests for is_vc_tool()."""

    def test_recognizes_expand(self):
        assert is_vc_tool("vc_expand_topic") is True

    def test_recognizes_collapse(self):
        assert is_vc_tool("vc_collapse_topic") is True

    def test_recognizes_find_quote(self):
        assert is_vc_tool("vc_find_quote") is True

    def test_rejects_unknown(self):
        assert is_vc_tool("vc_unknown") is False

    def test_rejects_client_tool(self):
        assert is_vc_tool("web_search") is False

    def test_rejects_empty(self):
        assert is_vc_tool("") is False


# ---------------------------------------------------------------------------
# TestExecuteVCTool (migrated from test_proxy.py)
# ---------------------------------------------------------------------------

class TestExecuteVCTool:
    """Tests for execute_vc_tool()."""

    def test_expand_calls_engine(self):
        engine = MagicMock()
        engine.expand_topic.return_value = {"tag": "db", "depth": "full", "tokens_added": 500}
        result = execute_vc_tool(engine, "vc_expand_topic", {"tag": "db", "depth": "full"})
        engine.expand_topic.assert_called_once_with(tag="db", depth="full")
        parsed = json.loads(result)
        assert parsed["tag"] == "db"
        assert parsed["tokens_added"] == 500

    def test_collapse_calls_engine(self):
        engine = MagicMock()
        engine.collapse_topic.return_value = {"tag": "api", "depth": "summary", "tokens_freed": 300}
        result = execute_vc_tool(engine, "vc_collapse_topic", {"tag": "api"})
        engine.collapse_topic.assert_called_once_with(tag="api", depth="summary")
        parsed = json.loads(result)
        assert parsed["tokens_freed"] == 300

    def test_find_quote_calls_engine(self):
        engine = MagicMock()
        engine.find_quote.return_value = {
            "query": "test",
            "query_intent": "current_state",
            "found": True,
            "results": [
                {
                    "excerpt": "found it",
                    "topic": "health",
                    "segment_ref": "seg-1",
                    "segment_refs": ["seg-1", "seg-2"],
                }
            ],
        }
        result = execute_vc_tool(engine, "vc_find_quote", {"query": "test"})
        engine.find_quote.assert_called_once_with(
            query="test",
            max_results=20,
            intent_context="",
        )
        parsed = json.loads(result)
        assert parsed["found"] is True
        assert "query" not in parsed
        assert "query_intent" not in parsed
        assert "segment_ref" not in parsed["results"][0]
        assert "segment_refs" not in parsed["results"][0]

    def test_find_session_calls_engine_with_session_filter(self):
        engine = MagicMock()
        engine.find_quote.return_value = {
            "found": True,
            "results": [{"excerpt": "found it", "topic": "sneakers"}],
        }
        result = execute_vc_tool(
            engine, "vc_find_session",
            {"query": "shoe rack", "session": "2023/05/29"},
        )
        engine.find_quote.assert_called_once_with(
            query="shoe rack",
            max_results=20,
            intent_context="",
            session_filter="2023/05/29",
        )
        parsed = json.loads(result)
        assert parsed["found"] is True

    def test_find_quote_does_not_pass_session_filter(self):
        """vc_find_quote must NOT pass session_filter to engine."""
        engine = MagicMock()
        engine.find_quote.return_value = {"found": False, "results": []}
        execute_vc_tool(engine, "vc_find_quote", {"query": "test"})
        call_kwargs = engine.find_quote.call_args[1]
        assert "session_filter" not in call_kwargs

    def test_remember_when_calls_engine(self):
        engine = MagicMock()
        engine.remember_when.return_value = {"query": "auth", "found": True, "results": []}
        result = execute_vc_tool(
            engine,
            "vc_remember_when",
            {
                "query": "auth",
                "time_range": {"kind": "relative", "preset": "last_7_days"},
            },
        )
        engine.remember_when.assert_called_once_with(
            query="auth",
            time_range={"kind": "relative", "preset": "last_7_days"},
            max_results=5,
        )
        parsed = json.loads(result)
        assert parsed["found"] is True

    def test_unknown_tool_returns_error(self):
        engine = MagicMock()
        result = execute_vc_tool(engine, "vc_unknown", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_engine_error_returns_error_json(self):
        engine = MagicMock()
        engine.expand_topic.side_effect = RuntimeError("boom")
        result = execute_vc_tool(engine, "vc_expand_topic", {"tag": "x"})
        parsed = json.loads(result)
        assert parsed["is_error"] is True
        assert "boom" in parsed["content"]

    def test_expand_default_depth(self):
        engine = MagicMock()
        engine.expand_topic.return_value = {}
        execute_vc_tool(engine, "vc_expand_topic", {"tag": "t"})
        engine.expand_topic.assert_called_once_with(tag="t", depth="full")

    def test_collapse_default_depth(self):
        engine = MagicMock()
        engine.collapse_topic.return_value = {}
        execute_vc_tool(engine, "vc_collapse_topic", {"tag": "t"})
        engine.collapse_topic.assert_called_once_with(tag="t", depth="summary")


# ---------------------------------------------------------------------------
# TestGetAdapter
# ---------------------------------------------------------------------------

class TestGetAdapter:
    """Tests for the get_adapter() factory."""

    def test_anthropic(self):
        a = get_adapter("anthropic", "key123")
        assert isinstance(a, AnthropicAdapter)
        assert a.api_key == "key123"

    def test_openai(self):
        a = get_adapter("openai", "sk-xxx")
        assert isinstance(a, OpenAIAdapter)

    def test_gemini(self):
        a = get_adapter("gemini", "AIza...")
        assert isinstance(a, GeminiAdapter)

    def test_openai_codex(self):
        a = get_adapter("openai-codex", "oauth-token")
        assert isinstance(a, OpenAICodexAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_adapter("mistral", "key")

    def test_custom_url(self):
        a = get_adapter("openai", "key", api_url="http://localhost:8000/v1/chat")
        assert a._base_url == "http://localhost:8000/v1/chat"


# ---------------------------------------------------------------------------
# TestAnthropicAdapter
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:
    """Tests for AnthropicAdapter."""

    def setup_method(self):
        self.adapter = AnthropicAdapter("test-key")

    def test_headers(self):
        h = self.adapter.get_headers()
        assert h["x-api-key"] == "test-key"
        assert "anthropic-version" in h

    def test_url_default(self):
        assert "anthropic.com" in self.adapter.get_url()

    def test_build_request_body(self):
        body = self.adapter.build_request_body(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.",
            max_tokens=1024, temperature=0.5,
            tools=[{"name": "t"}],
        )
        assert body["model"] == "claude-sonnet-4-5-20250929"
        assert body["system"] == "Be helpful."
        assert body["tools"] == [{"name": "t"}]
        assert body["tool_choice"] == {"type": "any"}
        assert body["stream"] is False

    def test_convert_tool_defs_passthrough(self):
        defs = vc_tool_definitions()
        assert self.adapter.convert_tool_defs(defs) is defs

    def test_extract_text(self):
        resp = {"content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]}
        assert self.adapter.extract_text(resp) == "Hello world"

    def test_extract_tool_calls(self):
        resp = {"content": [
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
            {"type": "text", "text": "checking..."},
        ]}
        calls = self.adapter.extract_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0] == {"id": "t1", "name": "vc_find_quote", "input": {"query": "x"}}

    def test_extract_usage(self):
        resp = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        assert self.adapter.extract_usage(resp) == (100, 50)

    def test_stop_reasons(self):
        assert self.adapter.is_tool_use_stop({"stop_reason": "tool_use"})
        assert not self.adapter.is_tool_use_stop({"stop_reason": "end_turn"})
        assert self.adapter.get_stop_reason({"stop_reason": "end_turn"}) == "end_turn"

    def test_build_tool_result(self):
        tr = self.adapter.build_tool_result("t1", "vc_find_quote", "found it")
        assert tr == {"type": "tool_result", "tool_use_id": "t1", "content": "found it"}

    def test_build_continuation_fresh(self):
        original = {
            "model": "claude-sonnet-4-5-20250929", "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are helpful.", "tools": [{"name": "web_search"}],
            "tool_choice": {"type": "any"},
        }
        raw_resp = {"content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
        ]}
        tool_results = [{"type": "tool_result", "tool_use_id": "t1", "content": "found it"}]

        body = self.adapter.build_continuation(None, original, raw_resp, tool_results)

        assert body["model"] == "claude-sonnet-4-5-20250929"
        assert body["max_tokens"] == 1024
        assert body["stream"] is False
        assert body["system"] == "You are helpful."
        assert body["tools"] == [{"name": "web_search"}]
        assert body["tool_choice"] == {"type": "any"}
        assert len(body["messages"]) == 3
        assert body["messages"][0] == {"role": "user", "content": "hi"}
        assert body["messages"][1]["role"] == "assistant"
        assert body["messages"][2]["role"] == "user"

    def test_build_continuation_does_not_mutate_original(self):
        original = {
            "model": "m", "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        original_len = len(original["messages"])
        raw_resp = {"content": [{"type": "text", "text": "ok"}]}
        self.adapter.build_continuation(None, original, raw_resp, [])
        assert len(original["messages"]) == original_len

    def test_strip_tools(self):
        body = {"model": "m", "tools": [{"name": "t"}], "tool_choice": {"type": "any"}}
        self.adapter.strip_tools(body)
        assert "tools" not in body
        assert "tool_choice" not in body


# ---------------------------------------------------------------------------
# TestOpenAIAdapter
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    """Tests for OpenAIAdapter."""

    def setup_method(self):
        self.adapter = OpenAIAdapter("sk-test")

    def test_headers(self):
        h = self.adapter.get_headers()
        assert h["Authorization"] == "Bearer sk-test"

    def test_url_default(self):
        assert "openai.com" in self.adapter.get_url()

    def test_build_request_body_with_system(self):
        body = self.adapter.build_request_body(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.", max_tokens=1024, temperature=0.5, tools=None,
        )
        assert body["model"] == "gpt-4o"
        assert body["max_completion_tokens"] == 1024
        assert body["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert body["messages"][1] == {"role": "user", "content": "hi"}
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_build_request_body_with_tools_sets_required_policy(self):
        body = self.adapter.build_request_body(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.",
            max_tokens=1024,
            temperature=0.5,
            tools=[{"type": "function", "function": {"name": "t"}}],
        )
        assert body["tool_choice"] == "required"

    def test_convert_tool_defs(self):
        defs = [{"name": "vc_find_quote", "description": "Search", "input_schema": {"type": "object", "properties": {}}}]
        converted = self.adapter.convert_tool_defs(defs)
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "vc_find_quote"
        assert converted[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_extract_text(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
        assert self.adapter.extract_text(resp) == "Hello"

    def test_extract_text_null_content(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": []}}]}
        assert self.adapter.extract_text(resp) == ""

    def test_extract_tool_calls(self):
        resp = {"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "vc_find_quote", "arguments": '{"query": "test"}'},
            }],
        }}]}
        calls = self.adapter.extract_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0]["id"] == "call_1"
        assert calls[0]["name"] == "vc_find_quote"
        assert calls[0]["input"] == {"query": "test"}

    def test_extract_tool_calls_bad_json(self):
        resp = {"choices": [{"message": {
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "vc_find_quote", "arguments": "not json"},
            }],
        }}]}
        calls = self.adapter.extract_tool_calls(resp)
        assert calls[0]["input"] == {}

    def test_extract_usage(self):
        resp = {"usage": {"prompt_tokens": 200, "completion_tokens": 80}}
        assert self.adapter.extract_usage(resp) == (200, 80)

    def test_stop_reasons(self):
        assert self.adapter.is_tool_use_stop({"choices": [{"finish_reason": "tool_calls"}]})
        assert not self.adapter.is_tool_use_stop({"choices": [{"finish_reason": "stop"}]})
        assert self.adapter.get_stop_reason({"choices": [{"finish_reason": "stop"}]}) == "end_turn"
        assert self.adapter.get_stop_reason({"choices": [{"finish_reason": "tool_calls"}]}) == "tool_use"

    def test_build_tool_result(self):
        tr = self.adapter.build_tool_result("call_1", "vc_find_quote", "result")
        assert tr == {"role": "tool", "tool_call_id": "call_1", "content": "result"}

    def test_build_continuation(self):
        original = {
            "model": "gpt-4o", "max_completion_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "tool_choice": "required",
        }
        raw_resp = {"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "vc_find_quote", "arguments": "{}"}}],
        }}]}
        tool_results = [{"role": "tool", "tool_call_id": "c1", "content": "found"}]

        body = self.adapter.build_continuation(None, original, raw_resp, tool_results)
        assert body["model"] == "gpt-4o"
        assert body["tool_choice"] == "required"
        assert len(body["messages"]) == 3  # user + assistant + tool
        assert body["messages"][1]["role"] == "assistant"
        assert body["messages"][2]["role"] == "tool"

    def test_strip_tools(self):
        body = {
            "model": "gpt-4o",
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "tool_choice": "required",
        }
        self.adapter.strip_tools(body)
        assert "tools" not in body
        assert "tool_choice" not in body


# ---------------------------------------------------------------------------
# TestGeminiAdapter
# ---------------------------------------------------------------------------

class TestGeminiAdapter:
    """Tests for GeminiAdapter."""

    def setup_method(self):
        self.adapter = GeminiAdapter("AIza-test-key")

    def test_headers(self):
        h = self.adapter.get_headers()
        assert h["Content-Type"] == "application/json"
        assert "Authorization" not in h

    def test_url_includes_model_and_key(self):
        url = self.adapter.get_url("gemini-2.0-flash")
        assert "gemini-2.0-flash" in url
        assert "AIza-test-key" in url
        assert "generateContent" in url

    def test_build_request_body(self):
        body = self.adapter.build_request_body(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.", max_tokens=1024, temperature=0.5, tools=None,
        )
        assert "contents" in body
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][0]["parts"] == [{"text": "hi"}]
        assert body["system_instruction"]["parts"] == [{"text": "Be helpful."}]
        assert body["generationConfig"]["maxOutputTokens"] == 1024

    def test_convert_tool_defs(self):
        defs = [{"name": "vc_find_quote", "description": "Search", "input_schema": {"type": "object"}}]
        converted = self.adapter.convert_tool_defs(defs)
        assert len(converted) == 1
        assert "functionDeclarations" in converted[0]
        assert converted[0]["functionDeclarations"][0]["name"] == "vc_find_quote"

    def test_extract_text(self):
        resp = {"candidates": [{"content": {"parts": [
            {"text": "Hello"}, {"text": " world"},
        ]}}]}
        assert self.adapter.extract_text(resp) == "Hello world"

    def test_extract_tool_calls(self):
        resp = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "vc_find_quote", "args": {"query": "test"}}},
        ]}}]}
        calls = self.adapter.extract_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0]["name"] == "vc_find_quote"
        assert calls[0]["input"] == {"query": "test"}
        assert "id" in calls[0]  # auto-generated UUID

    def test_extract_usage(self):
        resp = {"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50}}
        assert self.adapter.extract_usage(resp) == (100, 50)

    def test_stop_reasons(self):
        tool_resp = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "vc_find_quote", "args": {}}},
        ]}}]}
        text_resp = {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}

        assert self.adapter.is_tool_use_stop(tool_resp)
        assert not self.adapter.is_tool_use_stop(text_resp)
        assert self.adapter.get_stop_reason(tool_resp) == "tool_use"
        assert self.adapter.get_stop_reason(text_resp) == "end_turn"

    def test_build_tool_result(self):
        tr = self.adapter.build_tool_result("id", "vc_find_quote", "result")
        assert tr == {"functionResponse": {"name": "vc_find_quote", "response": {"content": "result"}}}

    def test_build_continuation(self):
        original = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 1024},
            "tools": [{"functionDeclarations": []}],
        }
        raw_resp = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "vc_find_quote", "args": {"query": "x"}}},
        ]}}]}
        tool_results = [{"functionResponse": {"name": "vc_find_quote", "response": {"content": "found"}}}]

        body = self.adapter.build_continuation(None, original, raw_resp, tool_results)
        assert len(body["contents"]) == 3  # user + model + user (function response)
        assert body["contents"][1]["role"] == "model"
        assert body["contents"][2]["role"] == "user"


# ---------------------------------------------------------------------------
# TestRunToolLoop
# ---------------------------------------------------------------------------

def _make_response(content, stop_reason="end_turn", usage=None):
    """Helper to create a mock Anthropic response dict."""
    return {
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 100, "output_tokens": 50},
    }


class _MockHTTPResponse:
    """Minimal mock for httpx.Response."""
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._data


def _anthropic_adapter():
    """Create an AnthropicAdapter for tests."""
    return AnthropicAdapter("test-key")


class TestRunToolLoop:
    """Tests for run_tool_loop()."""

    def test_no_tool_calls_returns_text(self):
        engine = MagicMock()
        response = _make_response([{"type": "text", "text": "Hello world"}])
        result = run_tool_loop(engine, response, {}, _anthropic_adapter())

        assert result.text == "Hello world"
        assert result.tool_calls == []
        assert result.continuation_count == 0
        assert result.stop_reason == "end_turn"

    def test_mixed_vc_and_non_vc_bails(self):
        engine = MagicMock()
        response = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
            {"type": "tool_use", "id": "t2", "name": "web_search", "input": {"q": "y"}},
        ], stop_reason="tool_use")
        result = run_tool_loop(engine, response, {}, _anthropic_adapter())

        assert result.tool_calls == []
        assert result.continuation_count == 0
        assert result.stop_reason == "tool_use"

    def test_single_tool_call_with_continuation(self):
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
        ], stop_reason="tool_use")

        continuation = _make_response(
            [{"type": "text", "text": "Found the answer."}],
            usage={"input_tokens": 200, "output_tokens": 80},
        )

        original_request = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}],
        }

        mock_resp = _MockHTTPResponse(continuation)
        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, original_request, _anthropic_adapter())

        assert result.text == "Found the answer."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "vc_find_quote"
        assert result.continuation_count == 1
        assert result.input_tokens == 300  # 100 + 200
        assert result.output_tokens == 130  # 50 + 80
        assert result.stop_reason == "end_turn"
        # find_quote is read-only — no working-set mutation, so no reassemble
        engine.reassemble_context.assert_not_called()
        engine.find_quote.assert_called_once_with(
            query="x",
            max_results=20,
            intent_context="test",
        )

    def test_passes_last_user_text_as_intent_context(self):
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "shoe rack"}},
        ], stop_reason="tool_use")
        continuation = _make_response([{"type": "text", "text": "answer"}])

        original_request = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "Where do I currently keep my old sneakers?"},
            ],
        }

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(continuation)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, original_request, _anthropic_adapter())

        assert result.text == "answer"
        engine.find_quote.assert_called_once_with(
            query="shoe rack",
            max_results=20,
            intent_context="Where do I currently keep my old sneakers?",
        )

    def test_multi_tool_single_loop(self):
        engine = MagicMock()
        engine.expand_topic.return_value = {"tag": "db", "tokens_added": 500}
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_expand_topic", "input": {"tag": "db"}},
            {"type": "tool_use", "id": "t2", "name": "vc_find_quote", "input": {"query": "x"}},
        ], stop_reason="tool_use")

        continuation = _make_response([{"type": "text", "text": "Done."}])
        mock_resp = _MockHTTPResponse(continuation)

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, {"model": "m", "max_tokens": 100, "messages": []}, _anthropic_adapter())

        assert len(result.tool_calls) == 2
        assert result.continuation_count == 1

    def test_chained_tool_calls(self):
        """LLM responds to first continuation with another tool call."""
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.expand_topic.return_value = {"tag": "t", "tokens_added": 100}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
        ], stop_reason="tool_use")

        cont1 = _make_response([
            {"type": "text", "text": "Searching... "},
            {"type": "tool_use", "id": "t2", "name": "vc_expand_topic", "input": {"tag": "t"}},
        ], stop_reason="tool_use")

        cont2 = _make_response([{"type": "text", "text": "Final answer."}])

        responses = [_MockHTTPResponse(cont1), _MockHTTPResponse(cont2)]

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = responses
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, {"model": "m", "max_tokens": 100, "messages": []}, _anthropic_adapter())

        assert result.text == "Searching... Final answer."
        assert len(result.tool_calls) == 2
        assert result.continuation_count == 2

    def test_max_loops_respected(self):
        """Loop terminates after max_loops even if LLM keeps calling tools.
        After exhausting loops with no text, a forced continuation fires."""
        engine = MagicMock()
        engine.find_quote.return_value = {"found": False, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
        ], stop_reason="tool_use")

        # Every continuation also returns a tool call
        tool_resp = _make_response([
            {"type": "tool_use", "id": "t2", "name": "vc_find_quote", "input": {"query": "y"}},
        ], stop_reason="tool_use")

        # Forced text continuation (tools stripped) returns actual text
        forced_resp = _make_response([
            {"type": "text", "text": "I could not find the answer."},
        ])

        responses = [
            _MockHTTPResponse(tool_resp),    # loop 1
            _MockHTTPResponse(tool_resp),    # loop 2 (max_loops exhausted)
            _MockHTTPResponse(forced_resp),  # forced text (BUG-017 fix)
        ]

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = responses
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(
                engine, initial,
                {"model": "m", "max_tokens": 100, "messages": []},
                _anthropic_adapter(), max_loops=2,
            )

        # 2 regular + 1 forced = 3 continuations
        assert result.continuation_count == 3
        # 1 per regular loop + 1 from forced = 3 tool executions
        assert len(result.tool_calls) == 3
        assert result.text == "I could not find the answer."

    def test_http_error_stops_loop(self):
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
        ], stop_reason="tool_use")

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse({}, status_code=500)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, {"model": "m", "max_tokens": 100, "messages": []}, _anthropic_adapter())

        assert result.stop_reason == "error"
        assert result.continuation_count == 1
        assert len(result.tool_calls) == 1

    def test_token_accumulation(self):
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response(
            [{"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}}],
            stop_reason="tool_use",
            usage={"input_tokens": 500, "output_tokens": 100},
        )

        cont = _make_response(
            [{"type": "text", "text": "ok"}],
            usage={"input_tokens": 600, "output_tokens": 200},
        )

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(cont)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, {"model": "m", "max_tokens": 100, "messages": []}, _anthropic_adapter())

        assert result.input_tokens == 1100
        assert result.output_tokens == 300

    def test_raw_responses_collected(self):
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response(
            [{"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}}],
            stop_reason="tool_use",
        )
        cont = _make_response([{"type": "text", "text": "done"}])

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(cont)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, {"model": "m", "max_tokens": 100, "messages": []}, _anthropic_adapter())

        assert len(result.raw_responses) == 2

    def test_text_before_tool_call_preserved(self):
        """Text emitted before a tool_use block is captured."""
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "text", "text": "Let me search. "},
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
        ], stop_reason="tool_use")

        cont = _make_response([{"type": "text", "text": "Found it."}])

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(cont)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, {"model": "m", "max_tokens": 100, "messages": []}, _anthropic_adapter())

        assert result.text == "Let me search. Found it."

    @pytest.mark.regression("BUG-017")
    def test_forced_text_after_max_loops_exhausted(self):
        """BUG-017: When max_loops is exhausted with only tool_use responses
        (no text), a forced continuation without tools should extract a text answer."""
        engine = MagicMock()
        engine.find_quote.return_value = {"found": False, "results": [], "message": "No matches"}
        engine.reassemble_context.return_value = ""

        # Initial response: tool_use only, no text
        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "painting"}},
        ], stop_reason="tool_use")

        # Every continuation returns another tool call (model keeps searching)
        tool_resp = _make_response([
            {"type": "tool_use", "id": "t2", "name": "vc_find_quote", "input": {"query": "sunset"}},
        ], stop_reason="tool_use")

        # The forced text continuation (no tools) produces actual text
        forced_text_resp = _make_response(
            [{"type": "text", "text": "I couldn't find information about a painting."}],
            usage={"input_tokens": 300, "output_tokens": 30},
        )

        responses = [
            _MockHTTPResponse(tool_resp),   # loop 1
            _MockHTTPResponse(tool_resp),   # loop 2 (max_loops=2 exhausted)
            _MockHTTPResponse(forced_text_resp),  # forced text continuation
        ]

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = responses
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(
                engine, initial,
                {"model": "m", "max_tokens": 100, "messages": []},
                _anthropic_adapter(), max_loops=2,
            )

        assert result.text != "", "BUG-017: text must not be empty after forced continuation"
        assert "couldn't find" in result.text

    def test_find_session_injected_on_suppression(self):
        """vc_find_session tool is dynamically added when suppression occurs."""
        engine = MagicMock()
        # find_quote returns a result that will trigger suppression
        engine.find_quote.return_value = {
            "found": True,
            "current_state_multi_session": True,
            "results": [
                {
                    "excerpt": "shoe rack",
                    "session": "2023/05/29",
                    "session_recency_rank": 1,
                },
                {
                    "excerpt": "under the bed",
                    "session": "2023/05/25",
                    "session_recency_rank": 2,
                },
            ],
        }
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote",
             "input": {"query": "sneakers"}},
        ], stop_reason="tool_use")

        continuation = _make_response([{"type": "text", "text": "shoe rack"}])
        mock_resp = _MockHTTPResponse(continuation)

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(
                engine, initial,
                {"model": "m", "max_tokens": 100, "messages": [],
                 "tools": [{"name": "vc_find_quote", "input_schema": {}}]},
                _anthropic_adapter(),
            )

        # The continuation body should contain vc_find_session tool
        sent_body = mock_client.post.call_args[1]["json"]
        tool_names = {t["name"] for t in sent_body.get("tools", [])}
        assert "vc_find_session" in tool_names

        # The suppressed excerpt should reference vc_find_session
        suppressed = json.loads(result.tool_calls[0].result_json)
        rank2 = [r for r in suppressed["results"]
                 if r.get("session_recency_rank") == 2][0]
        assert "vc_find_session" in rank2["excerpt"]
        assert "vc_find_quote" not in rank2["excerpt"]

    def test_find_session_not_injected_without_suppression(self):
        """vc_find_session is NOT added when no suppression occurs."""
        engine = MagicMock()
        engine.find_quote.return_value = {
            "found": True,
            "results": [{"excerpt": "data", "topic": "t"}],
        }
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote",
             "input": {"query": "test"}},
        ], stop_reason="tool_use")

        continuation = _make_response([{"type": "text", "text": "answer"}])
        mock_resp = _MockHTTPResponse(continuation)

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            run_tool_loop(
                engine, initial,
                {"model": "m", "max_tokens": 100, "messages": [],
                 "tools": [{"name": "vc_find_quote", "input_schema": {}}]},
                _anthropic_adapter(),
            )

        sent_body = mock_client.post.call_args[1]["json"]
        tool_names = {t["name"] for t in sent_body.get("tools", [])}
        assert "vc_find_session" not in tool_names


# ---------------------------------------------------------------------------
# TestRunToolLoopOpenAI
# ---------------------------------------------------------------------------

def _make_openai_response(content=None, tool_calls=None, finish_reason="stop", usage=None):
    """Helper to create a mock OpenAI response dict."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50},
    }


class TestRunToolLoopOpenAI:
    """Tests for run_tool_loop() with OpenAI adapter."""

    def test_no_tool_calls_returns_text(self):
        engine = MagicMock()
        response = _make_openai_response(content="Hello world")
        result = run_tool_loop(engine, response, {}, OpenAIAdapter("sk-test"))

        assert result.text == "Hello world"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"

    def test_single_tool_call_with_continuation(self):
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_openai_response(
            content=None,
            tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "vc_find_quote", "arguments": '{"query": "x"}'},
            }],
            finish_reason="tool_calls",
        )

        continuation = _make_openai_response(
            content="Found it.", finish_reason="stop",
            usage={"prompt_tokens": 200, "completion_tokens": 80},
        )

        original = {"model": "gpt-4o", "max_completion_tokens": 1024,
                     "messages": [{"role": "user", "content": "test"}]}

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(continuation)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, original, OpenAIAdapter("sk-test"))

        assert result.text == "Found it."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "vc_find_quote"
        assert result.continuation_count == 1
        assert result.input_tokens == 300
        assert result.output_tokens == 130


# ---------------------------------------------------------------------------
# TestVCToolNames
# ---------------------------------------------------------------------------

class TestVCToolNames:
    """Tests for VC_TOOL_NAMES constant."""

    def test_is_frozenset(self):
        assert isinstance(VC_TOOL_NAMES, frozenset)

    def test_contains_all_tools(self):
        assert VC_TOOL_NAMES == {
            "vc_expand_topic",
            "vc_collapse_topic",
            "vc_find_quote",
            "vc_find_session",
            "vc_query_facts",
            "vc_recall_all",
            "vc_remember_when",
        }


# ---------------------------------------------------------------------------
# TestAnthropicAdapter.inject_context
# ---------------------------------------------------------------------------

class TestAnthropicAdapterInjectContext:
    """Tests for AnthropicAdapter.inject_context()."""

    def setup_method(self):
        self.adapter = AnthropicAdapter("test-key")

    def test_replaces_existing_block_string_system(self):
        body = {
            "system": "<virtual-context>\nold summaries\n</virtual-context>\n\nBe helpful.",
            "messages": [],
        }
        self.adapter.inject_context(body, "NEW expanded text")
        assert "<virtual-context>\nNEW expanded text\n</virtual-context>" in body["system"]
        assert "old summaries" not in body["system"]
        assert "Be helpful." in body["system"]

    def test_no_existing_block_prepends(self):
        body = {"system": "Be helpful.", "messages": []}
        self.adapter.inject_context(body, "injected text")
        assert body["system"].startswith("<virtual-context>\ninjected text\n</virtual-context>")
        assert "Be helpful." in body["system"]

    def test_empty_system_sets_block(self):
        body = {"system": "", "messages": []}
        self.adapter.inject_context(body, "content")
        assert body["system"] == "<virtual-context>\ncontent\n</virtual-context>"

    def test_list_system_replaces_existing(self):
        body = {
            "system": [
                {"type": "text", "text": "<virtual-context>\nold\n</virtual-context>"},
                {"type": "text", "text": "Other instructions"},
            ],
            "messages": [],
        }
        self.adapter.inject_context(body, "new content")
        assert "<virtual-context>\nnew content\n</virtual-context>" in body["system"][0]["text"]
        assert "old" not in body["system"][0]["text"]

    def test_list_system_no_existing_inserts(self):
        body = {
            "system": [{"type": "text", "text": "Other instructions"}],
            "messages": [],
        }
        self.adapter.inject_context(body, "new content")
        assert body["system"][0]["text"] == "<virtual-context>\nnew content\n</virtual-context>"
        assert body["system"][1]["text"] == "Other instructions"

    def test_no_system_key(self):
        body = {"messages": []}
        self.adapter.inject_context(body, "content")
        assert body["system"] == "<virtual-context>\ncontent\n</virtual-context>"


# ---------------------------------------------------------------------------
# TestOpenAIAdapter.inject_context
# ---------------------------------------------------------------------------

class TestOpenAIAdapterInjectContext:
    """Tests for OpenAIAdapter.inject_context()."""

    def setup_method(self):
        self.adapter = OpenAIAdapter("sk-test")

    def test_replaces_existing_block(self):
        body = {
            "messages": [
                {"role": "system", "content": "<virtual-context>\nold\n</virtual-context>\n\nBe helpful."},
                {"role": "user", "content": "hi"},
            ],
        }
        self.adapter.inject_context(body, "new text")
        assert "<virtual-context>\nnew text\n</virtual-context>" in body["messages"][0]["content"]
        assert "old" not in body["messages"][0]["content"]
        assert "Be helpful." in body["messages"][0]["content"]

    def test_no_existing_block_prepends(self):
        body = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "hi"},
            ],
        }
        self.adapter.inject_context(body, "new text")
        assert body["messages"][0]["content"].startswith("<virtual-context>\nnew text\n</virtual-context>")
        assert "Be helpful." in body["messages"][0]["content"]

    def test_no_system_message_inserts(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        self.adapter.inject_context(body, "new text")
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "<virtual-context>\nnew text\n</virtual-context>"
        assert body["messages"][1]["role"] == "user"


# ---------------------------------------------------------------------------
# TestGeminiAdapter.inject_context
# ---------------------------------------------------------------------------

class TestGeminiAdapterInjectContext:
    """Tests for GeminiAdapter.inject_context()."""

    def setup_method(self):
        self.adapter = GeminiAdapter("AIza-test")

    def test_replaces_existing_block(self):
        body = {
            "system_instruction": {
                "parts": [{"text": "<virtual-context>\nold\n</virtual-context>\n\nBe helpful."}],
            },
            "contents": [],
        }
        self.adapter.inject_context(body, "new text")
        text = body["system_instruction"]["parts"][0]["text"]
        assert "<virtual-context>\nnew text\n</virtual-context>" in text
        assert "old" not in text
        assert "Be helpful." in text

    def test_no_existing_block_prepends(self):
        body = {
            "system_instruction": {"parts": [{"text": "Be helpful."}]},
            "contents": [],
        }
        self.adapter.inject_context(body, "new text")
        assert body["system_instruction"]["parts"][0]["text"] == "<virtual-context>\nnew text\n</virtual-context>"
        assert body["system_instruction"]["parts"][1]["text"] == "Be helpful."

    def test_no_system_instruction_creates(self):
        body = {"contents": []}
        self.adapter.inject_context(body, "new text")
        assert body["system_instruction"]["parts"][0]["text"] == "<virtual-context>\nnew text\n</virtual-context>"


# ---------------------------------------------------------------------------
# TestToolLoopInjectsReassembledContext
# ---------------------------------------------------------------------------

class TestToolLoopInjectsReassembledContext:
    """Integration test: verify run_tool_loop injects reassembled context."""

    def test_continuation_body_has_updated_context(self):
        """After expand_topic, the continuation body's system prompt should
        contain the reassembled text, not the original summaries."""
        engine = MagicMock()
        engine.expand_topic.return_value = {"tag": "db", "tokens_added": 500}
        engine.reassemble_context.return_value = "EXPANDED full DB conversation"

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_expand_topic", "input": {"tag": "db"}},
        ], stop_reason="tool_use")

        continuation = _make_response([{"type": "text", "text": "The DB was PostgreSQL."}])

        original_request = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What DB did we discuss?"}],
            "system": "<virtual-context>\ndb: summary only\n</virtual-context>\n\nBe helpful.",
        }

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(continuation)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, original_request, _anthropic_adapter())

        assert result.text == "The DB was PostgreSQL."
        engine.reassemble_context.assert_called_once()

        # Verify the captured continuation request has updated context
        assert len(result.raw_requests) == 1
        sent_system = result.raw_requests[0]["system"]
        assert "EXPANDED full DB conversation" in sent_system
        assert "summary only" not in sent_system

    def test_openai_continuation_has_updated_context(self):
        """OpenAI path: system message in messages[0] is updated."""
        engine = MagicMock()
        engine.expand_topic.return_value = {"tag": "t", "tokens_added": 100}
        engine.reassemble_context.return_value = "EXPANDED content"

        initial = _make_openai_response(
            content=None,
            tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "vc_expand_topic", "arguments": '{"tag": "t"}'},
            }],
            finish_reason="tool_calls",
        )

        continuation = _make_openai_response(content="Answer.", finish_reason="stop")

        original = {
            "model": "gpt-4o",
            "max_completion_tokens": 1024,
            "messages": [
                {"role": "system", "content": "<virtual-context>\nold\n</virtual-context>\n\nBe helpful."},
                {"role": "user", "content": "test"},
            ],
        }

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(continuation)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, original, OpenAIAdapter("sk-test"))

        sent_body = result.raw_requests[0]
        sys_content = sent_body["messages"][0]["content"]
        assert "EXPANDED content" in sys_content
        assert "old" not in sys_content

    def test_no_injection_when_reassemble_returns_empty(self):
        """When reassemble_context returns empty string, system is unchanged."""
        engine = MagicMock()
        engine.find_quote.return_value = {"found": True, "results": []}
        engine.reassemble_context.return_value = ""

        initial = _make_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote", "input": {"query": "x"}},
        ], stop_reason="tool_use")

        continuation = _make_response([{"type": "text", "text": "done"}])

        original_request = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}],
            "system": "original system",
        }

        with patch("virtual_context.core.tool_loop.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = _MockHTTPResponse(continuation)
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = run_tool_loop(engine, initial, original_request, _anthropic_adapter())

        sent_system = result.raw_requests[0]["system"]
        assert sent_system == "original system"
