"""Tests for virtual_context.core.provider_adapters — multi-provider tool loop adapters."""

from __future__ import annotations

import json
import uuid

import pytest

from virtual_context.core.provider_adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAICodexAdapter,
    ProviderAdapter,
    get_adapter,
)


# ── factory tests ────────────────────────────────────────────────────────

class TestGetAdapter:

    def test_anthropic(self):
        a = get_adapter("anthropic", "key-a")
        assert isinstance(a, AnthropicAdapter)

    def test_openai(self):
        a = get_adapter("openai", "key-o")
        assert isinstance(a, OpenAIAdapter)

    def test_openai_codex(self):
        a = get_adapter("openai-codex", "key-c")
        assert isinstance(a, OpenAICodexAdapter)

    def test_openai_codex_underscore(self):
        a = get_adapter("openai_codex", "key-c")
        assert isinstance(a, OpenAICodexAdapter)

    def test_gemini(self):
        a = get_adapter("gemini", "key-g")
        assert isinstance(a, GeminiAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_adapter("xyzzy", "key")


# ── Anthropic adapter tests ─────────────────────────────────────────────

class TestAnthropicAdapter:

    @pytest.fixture
    def adapter(self) -> AnthropicAdapter:
        return AnthropicAdapter(api_key="test-key")

    def test_headers(self, adapter):
        headers = adapter.get_headers()
        assert headers["x-api-key"] == "test-key"
        assert "anthropic-version" in headers

    def test_get_url(self, adapter):
        assert "anthropic.com" in adapter.get_url()

    def test_build_request_body(self, adapter):
        body = adapter.build_request_body(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.",
            max_tokens=1024,
            temperature=0.7,
            tools=None,
        )
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["system"] == "Be helpful."
        assert "tools" not in body

    def test_build_request_body_with_tools(self, adapter):
        tools = [{"name": "search", "description": "search", "input_schema": {}}]
        body = adapter.build_request_body(
            model="claude-sonnet-4-20250514", messages=[], system="",
            max_tokens=1024, temperature=0, tools=tools,
        )
        assert "tools" in body
        assert body["tool_choice"] == {"type": "any"}

    def test_convert_tool_defs_passthrough(self, adapter):
        defs = [{"name": "t1", "input_schema": {}}]
        assert adapter.convert_tool_defs(defs) is defs

    def test_extract_text(self, adapter):
        response = {"content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]}
        assert adapter.extract_text(response) == "Hello world"

    def test_extract_text_empty(self, adapter):
        assert adapter.extract_text({"content": []}) == ""

    def test_extract_tool_calls(self, adapter):
        response = {"content": [
            {"type": "tool_use", "id": "tu-1", "name": "search", "input": {"q": "hello"}},
            {"type": "text", "text": "I'll search for that."},
        ]}
        calls = adapter.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["id"] == "tu-1"
        assert calls[0]["name"] == "search"
        assert calls[0]["input"] == {"q": "hello"}

    def test_extract_tool_calls_empty(self, adapter):
        response = {"content": [{"type": "text", "text": "no tools"}]}
        assert adapter.extract_tool_calls(response) == []

    def test_extract_usage(self, adapter):
        response = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        assert adapter.extract_usage(response) == (100, 50)

    def test_is_tool_use_stop(self, adapter):
        assert adapter.is_tool_use_stop({"stop_reason": "tool_use"}) is True
        assert adapter.is_tool_use_stop({"stop_reason": "end_turn"}) is False

    def test_get_stop_reason(self, adapter):
        assert adapter.get_stop_reason({"stop_reason": "end_turn"}) == "end_turn"
        assert adapter.get_stop_reason({}) == "end_turn"

    def test_build_tool_result(self, adapter):
        result = adapter.build_tool_result("tu-1", "search", "found it")
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tu-1"
        assert result["content"] == "found it"

    def test_build_continuation_fresh(self, adapter):
        original = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": "Be helpful.",
            "tools": [{"name": "search"}],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "hello"}],
        }
        raw_response = {"content": [{"type": "text", "text": "thinking..."}]}
        tool_results = [{"type": "tool_result", "tool_use_id": "tu-1", "content": "ok"}]

        cont = adapter.build_continuation(None, original, raw_response, tool_results)
        assert cont["model"] == "claude-sonnet-4-20250514"
        assert cont["system"] == "Be helpful."
        # original user + assistant response + tool results
        assert len(cont["messages"]) == 3

    def test_build_continuation_append(self, adapter):
        cont_body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hello"}],
        }
        raw_response = {"content": [{"type": "text", "text": "more"}]}
        cont = adapter.build_continuation(cont_body, {}, raw_response, [])
        assert len(cont["messages"]) == 3  # original + assistant + empty user

    def test_inject_context_no_existing_block(self, adapter):
        body = {"system": "<system-reminder>\nOriginal VC context.\n</system-reminder>"}
        adapter.inject_context(body, "New VC context")
        assert "<system-reminder>" in body["system"]
        assert "New VC context" in body["system"]
        assert "Original VC context." not in body["system"]

    def test_inject_context_preserves_stable_prefix(self, adapter):
        body = {
            "system": "You are a helpful assistant.\n\n<system-reminder>\nOld VC context\n</system-reminder>"
        }
        adapter.inject_context(body, "Updated VC context")
        assert "You are a helpful assistant." in body["system"]
        assert "Updated VC context" in body["system"]
        assert "Old VC context" not in body["system"]

    def test_inject_context_replaces_existing_block(self, adapter):
        body = {"system": "<virtual-context>\nold context\n</virtual-context>\nRest of prompt."}
        adapter.inject_context(body, "new context")
        assert "old context" not in body["system"]
        assert "new context" in body["system"]
        assert "Rest of prompt." in body["system"]

    def test_inject_context_empty_system(self, adapter):
        body = {"system": ""}
        adapter.inject_context(body, "context")
        assert "<system-reminder>" in body["system"]

    def test_inject_context_list_system(self, adapter):
        body = {"system": [{"type": "text", "text": "Original text."}]}
        adapter.inject_context(body, "VC context")
        # Should prepend a new entry
        assert len(body["system"]) == 2
        assert "<system-reminder>" in body["system"][0]["text"]

    def test_strip_tools(self, adapter):
        body = {"tools": [{"name": "t"}], "tool_choice": {"type": "any"}, "model": "m"}
        adapter.strip_tools(body)
        assert "tools" not in body
        assert "tool_choice" not in body
        assert body["model"] == "m"


# ── OpenAI adapter tests ────────────────────────────────────────────────

class TestOpenAIAdapter:

    @pytest.fixture
    def adapter(self) -> OpenAIAdapter:
        return OpenAIAdapter(api_key="test-key")

    def test_headers(self, adapter):
        headers = adapter.get_headers()
        assert headers["Authorization"] == "Bearer test-key"

    def test_build_request_body_system_message(self, adapter):
        body = adapter.build_request_body(
            model="gpt-4.1", messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.", max_tokens=1024, temperature=0.5, tools=None,
        )
        # System becomes the first message
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "Be helpful."
        assert body["messages"][1]["role"] == "user"

    def test_build_request_body_no_system(self, adapter):
        body = adapter.build_request_body(
            model="gpt-4.1", messages=[{"role": "user", "content": "hi"}],
            system="", max_tokens=1024, temperature=0.5, tools=None,
        )
        assert body["messages"][0]["role"] == "user"

    def test_convert_tool_defs(self, adapter):
        anthropic_defs = [{
            "name": "search",
            "description": "Search for things",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]
        oai = adapter.convert_tool_defs(anthropic_defs)
        assert len(oai) == 1
        assert oai[0]["type"] == "function"
        assert oai[0]["function"]["name"] == "search"
        assert oai[0]["function"]["parameters"]["type"] == "object"

    def test_extract_text(self, adapter):
        response = {"choices": [{"message": {"content": "Hello!"}}]}
        assert adapter.extract_text(response) == "Hello!"

    def test_extract_text_empty(self, adapter):
        assert adapter.extract_text({"choices": []}) == ""

    def test_extract_text_null_content(self, adapter):
        response = {"choices": [{"message": {"content": None}}]}
        assert adapter.extract_text(response) == ""

    def test_extract_tool_calls(self, adapter):
        response = {"choices": [{"message": {
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "search", "arguments": '{"q": "hello"}'},
            }],
        }}]}
        calls = adapter.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["input"] == {"q": "hello"}

    def test_extract_tool_calls_malformed_json_args(self, adapter):
        response = {"choices": [{"message": {
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "search", "arguments": "not-json"},
            }],
        }}]}
        calls = adapter.extract_tool_calls(response)
        assert calls[0]["input"] == {}  # graceful fallback

    def test_extract_tool_calls_empty(self, adapter):
        response = {"choices": [{"message": {"content": "text", "tool_calls": None}}]}
        assert adapter.extract_tool_calls(response) == []

    def test_extract_usage(self, adapter):
        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        assert adapter.extract_usage(response) == (100, 50)

    def test_is_tool_use_stop(self, adapter):
        assert adapter.is_tool_use_stop({"choices": [{"finish_reason": "tool_calls"}]}) is True
        assert adapter.is_tool_use_stop({"choices": [{"finish_reason": "stop"}]}) is False

    def test_get_stop_reason_normalizes(self, adapter):
        assert adapter.get_stop_reason({"choices": [{"finish_reason": "stop"}]}) == "end_turn"
        assert adapter.get_stop_reason({"choices": [{"finish_reason": "tool_calls"}]}) == "tool_use"
        assert adapter.get_stop_reason({"choices": []}) == "error"

    def test_build_tool_result(self, adapter):
        result = adapter.build_tool_result("call-1", "search", "found it")
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call-1"

    def test_inject_context_system_message(self, adapter):
        body = {"messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]}
        adapter.inject_context(body, "VC context")
        assert "<system-reminder>" in body["messages"][0]["content"]
        assert "VC context" in body["messages"][0]["content"]

    def test_inject_context_no_system_message(self, adapter):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        adapter.inject_context(body, "VC context")
        assert body["messages"][0]["role"] == "system"
        assert "VC context" in body["messages"][0]["content"]

    def test_inject_context_replaces_existing(self, adapter):
        body = {"messages": [
            {"role": "system", "content": "<virtual-context>\nold\n</virtual-context>\nrest"},
            {"role": "user", "content": "hi"},
        ]}
        adapter.inject_context(body, "new context")
        assert "old" not in body["messages"][0]["content"]
        assert "new context" in body["messages"][0]["content"]

    def test_strip_tools(self, adapter):
        body = {"tools": [{}], "tool_choice": "auto"}
        adapter.strip_tools(body)
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_build_continuation_fresh(self, adapter):
        original = {
            "model": "gpt-4.1",
            "max_completion_tokens": 4096,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        }
        raw_response = {"choices": [{"message": {"role": "assistant", "content": "thinking"}}]}
        tool_results = [{"role": "tool", "tool_call_id": "c1", "content": "result"}]
        cont = adapter.build_continuation(None, original, raw_response, tool_results)
        # sys + user + assistant + tool result
        assert len(cont["messages"]) == 4


# ── OpenAI Codex adapter tests ──────────────────────────────────────────

class TestOpenAICodexAdapter:

    @pytest.fixture
    def adapter(self) -> OpenAICodexAdapter:
        return OpenAICodexAdapter(api_key="test-key")

    def test_build_request_body(self, adapter):
        body = adapter.build_request_body(
            model="codex-mini",
            messages=[{"role": "user", "content": "hello"}],
            system="Be helpful.",
            max_tokens=1024,
            temperature=0,
            tools=None,
        )
        assert body["instructions"] == "Be helpful."
        assert body["stream"] is True
        assert len(body["input"]) == 1
        assert body["input"][0]["role"] == "user"

    def test_extract_text(self, adapter):
        response = {"output": [
            {"type": "message", "content": [
                {"type": "output_text", "text": "hello"},
            ]},
        ]}
        assert adapter.extract_text(response) == "hello"

    def test_extract_tool_calls(self, adapter):
        response = {"output": [
            {"type": "function_call", "call_id": "fc-1", "name": "search",
             "arguments": '{"q": "test"}'},
        ]}
        calls = adapter.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["input"] == {"q": "test"}

    def test_extract_tool_calls_invalid_json(self, adapter):
        response = {"output": [
            {"type": "function_call", "call_id": "fc-1", "name": "search",
             "arguments": "not json"},
        ]}
        calls = adapter.extract_tool_calls(response)
        assert calls[0]["input"] == {}

    def test_extract_tool_calls_empty_args(self, adapter):
        response = {"output": [
            {"type": "function_call", "call_id": "fc-1", "name": "noop",
             "arguments": ""},
        ]}
        calls = adapter.extract_tool_calls(response)
        assert calls[0]["input"] == {}

    def test_is_tool_use_stop(self, adapter):
        assert adapter.is_tool_use_stop({"output": [{"type": "function_call"}]}) is True
        assert adapter.is_tool_use_stop({"output": [{"type": "message"}]}) is False

    def test_get_stop_reason(self, adapter):
        assert adapter.get_stop_reason({"output": [{"type": "function_call"}]}) == "tool_use"
        assert adapter.get_stop_reason({"error": "bad"}) == "error"
        assert adapter.get_stop_reason({"output": [{"type": "message"}]}) == "end_turn"

    def test_build_tool_result(self, adapter):
        result = adapter.build_tool_result("fc-1", "search", "data")
        assert result["type"] == "function_call_output"
        assert result["call_id"] == "fc-1"
        assert result["output"] == "data"

    def test_inject_context(self, adapter):
        body = {"instructions": "Original."}
        adapter.inject_context(body, "VC context")
        assert "<system-reminder>" in body["instructions"]
        assert "VC context" in body["instructions"]

    def test_inject_context_replaces_existing(self, adapter):
        body = {"instructions": "<system-reminder>\nold\n</system-reminder>\nOriginal."}
        adapter.inject_context(body, "new context")
        assert "old" not in body["instructions"]
        assert "new context" in body["instructions"]

    def test_convert_tool_defs(self, adapter):
        defs = [{"name": "search", "description": "s", "input_schema": {"type": "object"}}]
        converted = adapter.convert_tool_defs(defs)
        assert converted[0]["type"] == "function"
        assert converted[0]["name"] == "search"


# ── Gemini adapter tests ────────────────────────────────────────────────

class TestGeminiAdapter:

    @pytest.fixture
    def adapter(self) -> GeminiAdapter:
        return GeminiAdapter(api_key="test-key")

    def test_headers(self, adapter):
        headers = adapter.get_headers()
        assert headers["x-goog-api-key"] == "test-key"

    def test_get_url_default(self, adapter):
        url = adapter.get_url(model="gemini-2.5-flash")
        assert "gemini-2.5-flash" in url
        assert "generateContent" in url

    def test_get_url_custom(self):
        a = GeminiAdapter(api_key="k", api_url="https://custom.api/v1")
        assert a.get_url(model="m") == "https://custom.api/v1"

    def test_build_request_body(self, adapter):
        body = adapter.build_request_body(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.",
            max_tokens=1024,
            temperature=0.5,
            tools=None,
        )
        assert "contents" in body
        assert body["contents"][0]["role"] == "user"
        assert body["system_instruction"]["parts"][0]["text"] == "Be helpful."

    def test_build_request_body_assistant_role_mapped(self, adapter):
        body = adapter.build_request_body(
            model="gemini-2.0-flash",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            system="", max_tokens=1024, temperature=0, tools=None,
        )
        assert body["contents"][1]["role"] == "model"

    def test_build_request_body_thinking_model(self, adapter):
        body = adapter.build_request_body(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            system="", max_tokens=1024, temperature=0, tools=None,
        )
        # Thinking models get elevated max tokens and thinking config
        assert body["generationConfig"]["maxOutputTokens"] >= 8192
        assert "thinkingConfig" in body["generationConfig"]

    def test_convert_tool_defs(self, adapter):
        defs = [{"name": "search", "description": "s", "input_schema": {"type": "object"}}]
        converted = adapter.convert_tool_defs(defs)
        assert len(converted) == 1
        assert "functionDeclarations" in converted[0]
        assert converted[0]["functionDeclarations"][0]["name"] == "search"

    def test_extract_text(self, adapter):
        response = {"candidates": [{"content": {"parts": [
            {"text": "Hello "},
            {"text": "world"},
        ]}}]}
        assert adapter.extract_text(response) == "Hello world"

    def test_extract_text_empty(self, adapter):
        assert adapter.extract_text({"candidates": []}) == ""

    def test_extract_tool_calls(self, adapter):
        response = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "search", "args": {"q": "test"}}},
        ]}}]}
        calls = adapter.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["input"] == {"q": "test"}

    def test_extract_tool_calls_empty(self, adapter):
        response = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        assert adapter.extract_tool_calls(response) == []

    def test_extract_usage(self, adapter):
        response = {"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50}}
        assert adapter.extract_usage(response) == (100, 50)

    def test_is_tool_use_stop(self, adapter):
        response_yes = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "x", "args": {}}},
        ]}}]}
        response_no = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        assert adapter.is_tool_use_stop(response_yes) is True
        assert adapter.is_tool_use_stop(response_no) is False

    def test_get_stop_reason(self, adapter):
        assert adapter.get_stop_reason({"candidates": [{"finishReason": "STOP"}]}) == "end_turn"
        assert adapter.get_stop_reason({"candidates": []}) == "error"

    def test_build_tool_result_json_content(self, adapter):
        content = '{"results": ["a", "b"]}'
        result = adapter.build_tool_result("id-1", "search", content)
        assert "functionResponse" in result
        assert result["functionResponse"]["name"] == "search"
        assert result["functionResponse"]["response"]["results"] == ["a", "b"]

    def test_build_tool_result_plain_string(self, adapter):
        result = adapter.build_tool_result("id-1", "search", "plain text result")
        assert result["functionResponse"]["response"] == {"content": "plain text result"}

    def test_inject_context_with_existing_system(self, adapter):
        body = {"system_instruction": {"parts": [{"text": "Original."}]}}
        adapter.inject_context(body, "VC context")
        # Should prepend a new part
        assert "<system-reminder>" in body["system_instruction"]["parts"][0]["text"]

    def test_inject_context_no_system(self, adapter):
        body = {}
        adapter.inject_context(body, "VC context")
        assert "system_instruction" in body
        assert "<system-reminder>" in body["system_instruction"]["parts"][0]["text"]

    def test_inject_context_replaces_existing(self, adapter):
        body = {"system_instruction": {"parts": [
            {"text": "<virtual-context>\nold\n</virtual-context>\nRest."},
        ]}}
        adapter.inject_context(body, "new context")
        assert "old" not in body["system_instruction"]["parts"][0]["text"]
        assert "new context" in body["system_instruction"]["parts"][0]["text"]

    def test_strip_tools(self, adapter):
        body = {"tools": [{}], "tool_config": {"mode": "any"}}
        adapter.strip_tools(body)
        assert "tools" not in body
        assert "tool_config" not in body

    def test_add_tool_defs_fresh(self, adapter):
        body = {}
        defs = [{"name": "search", "description": "s", "input_schema": {}}]
        adapter.add_tool_defs(body, defs)
        assert "tools" in body
        assert body["tools"][0]["functionDeclarations"][0]["name"] == "search"

    def test_add_tool_defs_merge(self, adapter):
        body = {"tools": [{"functionDeclarations": [
            {"name": "existing", "description": "e", "parameters": {}},
        ]}]}
        defs = [{"name": "new_tool", "description": "n", "input_schema": {}}]
        adapter.add_tool_defs(body, defs)
        # Should merge into existing
        decls = body["tools"][0]["functionDeclarations"]
        assert len(decls) == 2
        assert {d["name"] for d in decls} == {"existing", "new_tool"}

    def test_build_continuation_fresh(self, adapter):
        original = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 1024},
            "tools": [{"functionDeclarations": []}],
            "tool_config": {"mode": "any"},
        }
        raw_response = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "search", "args": {"q": "test"}}},
        ]}}]}
        tool_results = [{"functionResponse": {"name": "search", "response": {"r": "ok"}}}]
        cont = adapter.build_continuation(None, original, raw_response, tool_results)
        # original user + model response + tool results
        assert len(cont["contents"]) == 3
        # tool_config should be removed for free-form response
        assert "tool_config" not in cont


# ── compress_previous_results tests ──────────────────────────────────────

class TestCompressPreviousResults:

    def test_anthropic_compress(self):
        adapter = AnthropicAdapter(api_key="k")
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "old-1", "content": "old result"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I found X based on the search."},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "current-1", "content": "current result"},
            ]},
        ]}
        adapter.compress_previous_results(body, {"current-1"})
        # Old assistant text should be compressed
        assert body["messages"][1]["content"][0]["text"] == "[Previous reasoning compressed]"
        # Tool results are NOT compressed

    def test_openai_compress(self):
        adapter = OpenAIAdapter(api_key="k")
        body = {"messages": [
            {"role": "assistant", "content": "I'll search for that."},
            {"role": "tool", "tool_call_id": "old-1", "content": "old data"},
            {"role": "assistant", "content": "Based on old data..."},
            {"role": "tool", "tool_call_id": "current-1", "content": "new data"},
        ]}
        adapter.compress_previous_results(body, {"current-1"})
        assert body["messages"][0]["content"] == "[Previous reasoning compressed]"
        assert body["messages"][2]["content"] == "[Previous reasoning compressed]"

    def test_gemini_compress(self):
        adapter = GeminiAdapter(api_key="k")
        body = {"contents": [
            {"role": "user", "parts": [{"text": "q1"}]},
            {"role": "model", "parts": [{"text": "I'll search."}]},
            {"role": "user", "parts": [{"text": "q2"}]},
        ]}
        adapter.compress_previous_results(body, set())
        # Model text in all but last entry should be compressed
        assert body["contents"][1]["parts"][0]["text"] == "[Previous reasoning compressed]"
