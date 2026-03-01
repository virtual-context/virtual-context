"""Tests for virtual_context.proxy.formats — PayloadFormat ABC + implementations."""

import json
import pytest

from virtual_context.proxy.formats import (
    PayloadFormat,
    AnthropicFormat,
    OpenAIFormat,
    OpenAIResponsesFormat,
    GeminiFormat,
    detect_format,
    get_format,
)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_anthropic_via_system_field(self):
        body = {"system": "You are helpful.", "messages": []}
        fmt = detect_format(body)
        assert fmt.name == "anthropic"
        assert isinstance(fmt, AnthropicFormat)

    def test_anthropic_via_claude_model(self):
        body = {"model": "claude-haiku-4-5-20251001", "messages": []}
        fmt = detect_format(body)
        assert fmt.name == "anthropic"

    def test_openai_default(self):
        body = {"model": "gpt-4o", "messages": []}
        fmt = detect_format(body)
        assert fmt.name == "openai"
        assert isinstance(fmt, OpenAIFormat)

    def test_openai_no_model(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        fmt = detect_format(body)
        assert fmt.name == "openai"

    def test_gemini_via_contents(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        fmt = detect_format(body)
        assert fmt.name == "gemini"
        assert isinstance(fmt, GeminiFormat)

    def test_gemini_via_system_instruction(self):
        body = {"system_instruction": {"parts": [{"text": "Be helpful"}]}, "contents": []}
        fmt = detect_format(body)
        assert fmt.name == "gemini"

    def test_openai_responses_via_input_list(self):
        body = {"model": "gpt-4.1", "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]}
        fmt = detect_format(body)
        assert fmt.name == "openai_responses"
        assert isinstance(fmt, OpenAIResponsesFormat)

    def test_openai_responses_via_instructions(self):
        body = {"model": "gpt-4.1", "instructions": "Be helpful."}
        fmt = detect_format(body)
        assert fmt.name == "openai_responses"

    def test_openai_responses_does_not_interfere_with_gemini(self):
        # Gemini detection should take priority
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        assert detect_format(body).name == "gemini"

    def test_openai_responses_does_not_interfere_with_anthropic(self):
        # Anthropic with system field should still be detected correctly
        body = {"system": "You are helpful.", "messages": []}
        assert detect_format(body).name == "anthropic"

    def test_openai_chat_completions_still_detected(self):
        # Standard chat completions without input/instructions
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        assert detect_format(body).name == "openai"

    def test_get_format_by_name(self):
        assert get_format("anthropic").name == "anthropic"
        assert get_format("openai").name == "openai"
        assert get_format("openai_responses").name == "openai_responses"
        assert get_format("gemini").name == "gemini"


# ---------------------------------------------------------------------------
# AnthropicFormat
# ---------------------------------------------------------------------------

class TestAnthropicFormat:
    fmt = AnthropicFormat()

    def test_extract_user_message_string(self):
        body = {"messages": [
            {"role": "user", "content": "hello world"},
        ]}
        assert self.fmt.extract_user_message(body) == "hello world"

    def test_extract_user_message_content_blocks(self):
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]},
        ]}
        # Should get last text block
        assert self.fmt.extract_user_message(body) == "second"

    def test_extract_user_message_strips_openclaw(self):
        body = {"messages": [
            {"role": "user", "content": "[Telegram id:123] actual question"},
        ]}
        assert self.fmt.extract_user_message(body) == "actual question"

    def test_extract_user_message_empty(self):
        body = {"messages": [{"role": "assistant", "content": "hi"}]}
        assert self.fmt.extract_user_message(body) == ""

    def test_extract_message_text_string(self):
        msg = {"role": "user", "content": "hello"}
        assert self.fmt.extract_message_text(msg) == "hello"

    def test_extract_message_text_blocks(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "block1"},
            {"type": "text", "text": "block2"},
        ]}
        assert self.fmt.extract_message_text(msg) == "block2"

    def test_extract_history_pairs(self):
        body = {"messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "current"},  # current turn, dropped
        ]}
        pairs = self.fmt.extract_history_pairs(body)
        assert len(pairs) == 4
        assert pairs[0].role == "user"
        assert pairs[0].content == "q1"
        assert pairs[1].role == "assistant"
        assert pairs[1].content == "a1"

    def test_extract_history_pairs_empty(self):
        body = {"messages": [{"role": "user", "content": "only one"}]}
        assert self.fmt.extract_history_pairs(body) == []

    def test_has_messages(self):
        assert self.fmt.has_messages({"messages": []}) is True
        assert self.fmt.has_messages({"contents": []}) is False
        assert self.fmt.has_messages({}) is False

    def test_get_messages(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert len(self.fmt.get_messages(body)) == 1

    def test_inject_context_string_system(self):
        body = {"system": "Be helpful.", "messages": []}
        result = self.fmt.inject_context(body, "topic summary")
        assert "<system-reminder>" in result["system"]
        assert "topic summary" in result["system"]
        assert "Be helpful." in result["system"]

    def test_inject_context_list_system(self):
        body = {"system": [{"type": "text", "text": "Be helpful."}], "messages": []}
        result = self.fmt.inject_context(body, "topic summary")
        assert isinstance(result["system"], list)
        assert result["system"][0]["text"] == "Be helpful."
        assert result["system"][-1]["text"].startswith("<system-reminder>")

    def test_inject_context_no_system(self):
        body = {"messages": []}
        result = self.fmt.inject_context(body, "ctx")
        assert "<system-reminder>" in result.get("system", "")

    def test_inject_context_empty_prepend(self):
        body = {"system": "original", "messages": []}
        result = self.fmt.inject_context(body, "")
        assert result is body  # no copy

    def test_inject_context_does_not_mutate(self):
        body = {"system": "original", "messages": [{"role": "user", "content": "hi"}]}
        result = self.fmt.inject_context(body, "ctx")
        assert body["system"] == "original"

    def test_extract_session_id(self):
        body = {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello <!-- vc:session=abc-123-def -->"},
        ]}
        assert self.fmt.extract_session_id(body) == "abc-123-def"

    def test_extract_session_id_content_blocks(self):
        body = {"messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello <!-- vc:session=abc-123 -->"},
            ]},
        ]}
        assert self.fmt.extract_session_id(body) == "abc-123"

    def test_extract_session_id_none(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert self.fmt.extract_session_id(body) is None

    def test_strip_session_markers_string(self):
        body = {"messages": [
            {"role": "assistant", "content": "hello <!-- vc:session=abc-123 -->"},
        ]}
        result = self.fmt.strip_session_markers(body)
        assert "vc:session" not in result["messages"][0]["content"]

    def test_strip_session_markers_blocks(self):
        body = {"messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello <!-- vc:session=abc-123 -->"},
            ]},
        ]}
        result = self.fmt.strip_session_markers(body)
        assert "vc:session" not in result["messages"][0]["content"][0]["text"]

    def test_strip_session_markers_no_markers(self):
        body = {"messages": [
            {"role": "assistant", "content": "hello"},
        ]}
        result = self.fmt.strip_session_markers(body)
        assert result is body  # no copy needed

    def test_inject_session_marker(self):
        response = {"content": [{"type": "text", "text": "hello"}]}
        marker = "\n<!-- vc:session=test-123 -->"
        result = self.fmt.inject_session_marker(response, marker)
        assert result["content"][0]["text"].endswith(marker)
        # original not mutated
        assert response["content"][0]["text"] == "hello"

    def test_inject_session_marker_no_text_block(self):
        response = {"content": []}
        marker = "\n<!-- vc:session=test -->"
        result = self.fmt.inject_session_marker(response, marker)
        assert result["content"][-1]["text"] == marker

    def test_emit_session_marker_sse(self):
        data = self.fmt.emit_session_marker_sse("test-session-id")
        assert b"content_block_delta" in data
        assert b"vc:session=test-session-id" in data

    def test_extract_delta_text(self):
        data = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        }
        assert self.fmt.extract_delta_text(data) == "hello"

    def test_extract_delta_text_non_delta(self):
        data = {"type": "message_start"}
        assert self.fmt.extract_delta_text(data) == ""

    def test_extract_assistant_text(self):
        response = {"content": [
            {"type": "thinking", "text": "hmm..."},
            {"type": "text", "text": "the answer"},
        ]}
        assert self.fmt.extract_assistant_text(response) == "the answer"

    def test_supports_tool_interception(self):
        assert self.fmt.supports_tool_interception is True

    def test_inject_tools(self):
        body = {"tools": [{"name": "existing"}], "messages": []}
        result = self.fmt.inject_tools(body, [{"name": "vc_expand_topic"}])
        assert len(result["tools"]) == 2

    def test_inject_tools_sets_required_when_requested(self):
        body = {"messages": []}
        result = self.fmt.inject_tools(
            body, [{"name": "vc_expand_topic"}], require_tool_use=True,
        )
        assert result["tool_choice"] == {"type": "any"}

    def test_inject_tools_respects_none_choice(self):
        body = {"tool_choice": "none", "messages": []}
        result = self.fmt.inject_tools(body, [{"name": "test"}])
        assert result is body


# ---------------------------------------------------------------------------
# OpenAIFormat
# ---------------------------------------------------------------------------

class TestOpenAIFormat:
    fmt = OpenAIFormat()

    def test_extract_user_message(self):
        body = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]}
        assert self.fmt.extract_user_message(body) == "hello"

    def test_extract_history_pairs(self):
        body = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "current"},
        ]}
        pairs = self.fmt.extract_history_pairs(body)
        assert len(pairs) == 2
        assert pairs[0].content == "q1"
        assert pairs[1].content == "a1"

    def test_inject_context_with_system(self):
        body = {"messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]}
        result = self.fmt.inject_context(body, "ctx text")
        assert "<system-reminder>" in result["messages"][0]["content"]
        assert "Be helpful." in result["messages"][0]["content"]

    def test_inject_context_without_system(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = self.fmt.inject_context(body, "ctx text")
        assert result["messages"][0]["role"] == "system"
        assert "<system-reminder>" in result["messages"][0]["content"]

    def test_extract_session_id(self):
        body = {"messages": [
            {"role": "assistant", "content": "hi <!-- vc:session=abc-123 -->"},
        ]}
        assert self.fmt.extract_session_id(body) == "abc-123"

    def test_inject_session_marker(self):
        response = {"choices": [{"message": {"content": "hello"}}]}
        marker = "\n<!-- vc:session=test -->"
        result = self.fmt.inject_session_marker(response, marker)
        assert result["choices"][0]["message"]["content"].endswith(marker)

    def test_emit_session_marker_sse(self):
        data = self.fmt.emit_session_marker_sse("test-id")
        assert b"vc:session=test-id" in data
        assert data.startswith(b"data: ")

    def test_extract_delta_text(self):
        data = {"choices": [{"delta": {"content": "hello"}}]}
        assert self.fmt.extract_delta_text(data) == "hello"

    def test_extract_assistant_text(self):
        response = {"choices": [{"message": {"content": "the answer"}}]}
        assert self.fmt.extract_assistant_text(response) == "the answer"

    def test_supports_tool_interception(self):
        assert self.fmt.supports_tool_interception is False


# ---------------------------------------------------------------------------
# GeminiFormat
# ---------------------------------------------------------------------------

class TestGeminiFormat:
    fmt = GeminiFormat()

    def test_extract_user_message(self):
        body = {"contents": [
            {"role": "user", "parts": [{"text": "hello world"}]},
        ]}
        assert self.fmt.extract_user_message(body) == "hello world"

    def test_extract_user_message_multi_parts(self):
        body = {"contents": [
            {"role": "user", "parts": [{"text": "part1"}, {"text": "part2"}]},
        ]}
        assert "part1" in self.fmt.extract_user_message(body)
        assert "part2" in self.fmt.extract_user_message(body)

    def test_extract_user_message_empty(self):
        body = {"contents": [{"role": "model", "parts": [{"text": "hi"}]}]}
        assert self.fmt.extract_user_message(body) == ""

    def test_extract_message_text(self):
        msg = {"role": "user", "parts": [{"text": "hello"}]}
        assert self.fmt.extract_message_text(msg) == "hello"

    def test_extract_history_pairs(self):
        body = {"contents": [
            {"role": "user", "parts": [{"text": "q1"}]},
            {"role": "model", "parts": [{"text": "a1"}]},
            {"role": "user", "parts": [{"text": "q2"}]},
            {"role": "model", "parts": [{"text": "a2"}]},
            {"role": "user", "parts": [{"text": "current"}]},
        ]}
        pairs = self.fmt.extract_history_pairs(body)
        assert len(pairs) == 4
        assert pairs[0].content == "q1"
        assert pairs[1].role == "assistant"  # normalized from "model"
        assert pairs[1].content == "a1"

    def test_extract_history_pairs_empty(self):
        body = {"contents": []}
        assert self.fmt.extract_history_pairs(body) == []

    def test_has_messages(self):
        assert self.fmt.has_messages({"contents": []}) is True
        assert self.fmt.has_messages({"messages": []}) is False
        assert self.fmt.has_messages({}) is False

    def test_get_messages(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        assert len(self.fmt.get_messages(body)) == 1

    def test_inject_context(self):
        body = {"contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
        ]}
        result = self.fmt.inject_context(body, "ctx text")
        si = result["system_instruction"]
        assert si["parts"][0]["text"].startswith("<system-reminder>")

    def test_inject_context_with_existing_system_instruction(self):
        body = {
            "system_instruction": {"parts": [{"text": "Be helpful."}]},
            "contents": [],
        }
        result = self.fmt.inject_context(body, "ctx")
        parts = result["system_instruction"]["parts"]
        assert len(parts) == 2
        assert "<system-reminder>" in parts[0]["text"]
        assert parts[1]["text"] == "Be helpful."

    def test_inject_context_empty(self):
        body = {"contents": []}
        result = self.fmt.inject_context(body, "")
        assert result is body

    def test_extract_session_id(self):
        body = {"contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello <!-- vc:session=abc-123 -->"}]},
        ]}
        assert self.fmt.extract_session_id(body) == "abc-123"

    def test_extract_session_id_none(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        assert self.fmt.extract_session_id(body) is None

    def test_strip_session_markers(self):
        body = {"contents": [
            {"role": "model", "parts": [
                {"text": "hello <!-- vc:session=abc-123 -->"},
            ]},
        ]}
        result = self.fmt.strip_session_markers(body)
        assert "vc:session" not in result["contents"][0]["parts"][0]["text"]

    def test_strip_session_markers_no_markers(self):
        body = {"contents": [{"role": "model", "parts": [{"text": "hello"}]}]}
        result = self.fmt.strip_session_markers(body)
        assert result is body

    def test_inject_session_marker(self):
        response = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        marker = "\n<!-- vc:session=test -->"
        result = self.fmt.inject_session_marker(response, marker)
        assert result["candidates"][0]["content"]["parts"][0]["text"].endswith(marker)

    def test_emit_session_marker_sse(self):
        data = self.fmt.emit_session_marker_sse("test-id")
        assert b"vc:session=test-id" in data
        decoded = json.loads(data.decode().split("data: ")[1])
        assert decoded["candidates"][0]["content"]["role"] == "model"

    def test_extract_delta_text(self):
        data = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        assert self.fmt.extract_delta_text(data) == "hi"

    def test_extract_delta_text_empty(self):
        data = {"candidates": []}
        assert self.fmt.extract_delta_text(data) == ""

    def test_extract_assistant_text(self):
        response = {"candidates": [{"content": {"parts": [
            {"text": "part1"},
            {"text": "part2"},
        ]}}]}
        text = self.fmt.extract_assistant_text(response)
        assert "part1" in text
        assert "part2" in text

    def test_extract_assistant_text_empty(self):
        response = {"candidates": []}
        assert self.fmt.extract_assistant_text(response) == ""

    def test_supports_tool_interception(self):
        assert self.fmt.supports_tool_interception is True

    def test_inject_tools(self):
        body = {"contents": [], "tools": []}
        tool_defs = [
            {"name": "vc_expand_topic", "description": "Expand", "input_schema": {
                "type": "object", "properties": {"tag": {"type": "string"}},
            }},
        ]
        result = self.fmt.inject_tools(body, tool_defs)
        assert len(result["tools"]) == 1
        decls = result["tools"][0]["functionDeclarations"]
        assert decls[0]["name"] == "vc_expand_topic"

    def test_inject_tools_existing_declarations(self):
        body = {"contents": [], "tools": [{"functionDeclarations": [
            {"name": "existing_tool"},
        ]}]}
        tool_defs = [{"name": "vc_expand_topic", "description": "Expand"}]
        result = self.fmt.inject_tools(body, tool_defs)
        decls = result["tools"][0]["functionDeclarations"]
        assert len(decls) == 2

    def test_compute_fingerprint(self):
        body = {"contents": [
            {"role": "user", "parts": [{"text": "q1"}]},
            {"role": "model", "parts": [{"text": "a1"}]},
            {"role": "user", "parts": [{"text": "q2"}]},
            {"role": "model", "parts": [{"text": "a2"}]},
            {"role": "user", "parts": [{"text": "current"}]},
        ]}
        fp = self.fmt.compute_fingerprint(body)
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_compute_fingerprint_too_few_messages(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "only one"}]}]}
        assert self.fmt.compute_fingerprint(body) == ""

    def test_estimate_system_tokens(self):
        body = {"system_instruction": {"parts": [{"text": "x" * 400}]}}
        tokens = self.fmt._estimate_system_tokens(body)
        assert tokens == 100  # 400 chars // 4


# ---------------------------------------------------------------------------
# Cross-format consistency
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Server wrapper delegation tests
# ---------------------------------------------------------------------------

class TestServerWrappers:
    """Verify server.py wrapper functions delegate to format objects."""

    def test_detect_api_format_returns_string(self):
        from virtual_context.proxy.server import _detect_api_format
        assert _detect_api_format({"system": "x", "messages": []}) == "anthropic"
        assert _detect_api_format({"messages": []}) == "openai"
        assert _detect_api_format({"contents": []}) == "gemini"
        assert _detect_api_format({"input": [{"role": "user", "content": "hi"}]}) == "openai_responses"

    def test_extract_user_message_anthropic(self):
        from virtual_context.proxy.server import _extract_user_message
        body = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}
        assert _extract_user_message(body) == "hi"

    def test_extract_user_message_gemini(self):
        from virtual_context.proxy.server import _extract_user_message
        body = {"contents": [{"role": "user", "parts": [{"text": "hello gemini"}]}]}
        assert _extract_user_message(body) == "hello gemini"

    def test_extract_history_pairs_gemini(self):
        from virtual_context.proxy.server import _extract_history_pairs
        body = {"contents": [
            {"role": "user", "parts": [{"text": "q1"}]},
            {"role": "model", "parts": [{"text": "a1"}]},
            {"role": "user", "parts": [{"text": "current"}]},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 2
        assert pairs[0].content == "q1"
        assert pairs[1].role == "assistant"

    def test_inject_context_gemini(self):
        from virtual_context.proxy.server import _inject_context
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        result = _inject_context(body, "context text", "gemini")
        assert "system_instruction" in result
        assert "<system-reminder>" in result["system_instruction"]["parts"][0]["text"]

    def test_extract_session_id_gemini(self):
        from virtual_context.proxy.server import _extract_session_id
        body = {"contents": [
            {"role": "model", "parts": [{"text": "hi <!-- vc:session=abc-123 -->"}]},
        ]}
        assert _extract_session_id(body) == "abc-123"

    def test_strip_session_markers_gemini(self):
        from virtual_context.proxy.server import _strip_session_markers
        body = {"contents": [
            {"role": "model", "parts": [{"text": "hi <!-- vc:session=abc -->"}]},
        ]}
        result = _strip_session_markers(body)
        assert "vc:session" not in result["contents"][0]["parts"][0]["text"]

    def test_extract_delta_text_gemini(self):
        from virtual_context.proxy.server import _extract_delta_text
        data = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        assert _extract_delta_text(data, "gemini") == "hello"

    def test_extract_assistant_text_gemini(self):
        from virtual_context.proxy.server import _extract_assistant_text
        response = {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]}
        assert _extract_assistant_text(response, "gemini") == "answer"

    def test_inject_session_marker_gemini(self):
        from virtual_context.proxy.server import _inject_session_marker
        response = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        marker = "\n<!-- vc:session=test -->"
        result = _inject_session_marker(response, marker, "gemini")
        assert result["candidates"][0]["content"]["parts"][0]["text"].endswith(marker)

    def test_extract_user_message_responses(self):
        from virtual_context.proxy.server import _extract_user_message
        body = {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello responses"}]},
        ]}
        assert _extract_user_message(body) == "hello responses"

    def test_extract_history_pairs_responses(self):
        from virtual_context.proxy.server import _extract_history_pairs
        body = {"input": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
            {"role": "user", "content": "current"},
        ]}
        pairs = _extract_history_pairs(body)
        assert len(pairs) == 2
        assert pairs[0].content == "q1"
        assert pairs[1].role == "assistant"

    def test_inject_context_responses(self):
        from virtual_context.proxy.server import _inject_context
        body = {"instructions": "Be helpful.", "input": []}
        result = _inject_context(body, "context text", "openai_responses")
        assert "<system-reminder>" in result["instructions"]

    def test_extract_session_id_responses(self):
        from virtual_context.proxy.server import _extract_session_id
        body = {"input": [
            {"role": "assistant", "content": "hi <!-- vc:session=abc-123 -->"},
        ]}
        assert _extract_session_id(body) == "abc-123"

    def test_strip_session_markers_responses(self):
        from virtual_context.proxy.server import _strip_session_markers
        body = {"input": [
            {"role": "assistant", "content": "hi <!-- vc:session=abc -->"},
        ]}
        result = _strip_session_markers(body)
        assert "vc:session" not in result["input"][0]["content"]

    def test_extract_delta_text_responses(self):
        from virtual_context.proxy.server import _extract_delta_text
        data = {"type": "response.output_text.delta", "delta": "hello"}
        assert _extract_delta_text(data, "openai_responses") == "hello"

    def test_extract_assistant_text_responses(self):
        from virtual_context.proxy.server import _extract_assistant_text
        response = {"output": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "answer"},
            ]},
        ]}
        assert _extract_assistant_text(response, "openai_responses") == "answer"

    def test_inject_session_marker_responses(self):
        from virtual_context.proxy.server import _inject_session_marker
        response = {"output": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "hello"},
            ]},
        ]}
        marker = "\n<!-- vc:session=test -->"
        result = _inject_session_marker(response, marker, "openai_responses")
        assert result["output"][0]["content"][0]["text"].endswith(marker)


# ---------------------------------------------------------------------------
# OpenAIResponsesFormat
# ---------------------------------------------------------------------------

class TestOpenAIResponsesFormat:
    fmt = OpenAIResponsesFormat()

    # -- User message extraction --

    def test_extract_user_message_input_text_blocks(self):
        body = {"input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "hello world"},
            ]},
        ]}
        assert self.fmt.extract_user_message(body) == "hello world"

    def test_extract_user_message_string_content(self):
        body = {"input": [
            {"role": "user", "content": "plain string"},
        ]}
        assert self.fmt.extract_user_message(body) == "plain string"

    def test_extract_user_message_last_user(self):
        body = {"input": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "reply"}]},
            {"role": "user", "content": "second"},
        ]}
        assert self.fmt.extract_user_message(body) == "second"

    def test_extract_user_message_skips_function_call(self):
        body = {"input": [
            {"role": "user", "content": "question"},
            {"type": "function_call", "name": "get_weather", "arguments": "{}"},
            {"type": "function_call_output", "output": "sunny"},
        ]}
        assert self.fmt.extract_user_message(body) == "question"

    def test_extract_user_message_strips_openclaw(self):
        body = {"input": [
            {"role": "user", "content": "[Telegram id:123] actual question"},
        ]}
        assert self.fmt.extract_user_message(body) == "actual question"

    def test_extract_user_message_empty(self):
        body = {"input": [{"role": "assistant", "content": "hi"}]}
        assert self.fmt.extract_user_message(body) == ""

    def test_extract_user_message_no_input(self):
        body = {}
        assert self.fmt.extract_user_message(body) == ""

    def test_extract_user_message_string_input(self):
        body = {"input": "just a string"}
        assert self.fmt.extract_user_message(body) == "just a string"

    # -- Message text extraction --

    def test_extract_message_text_input_text(self):
        msg = {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        assert self.fmt.extract_message_text(msg) == "hello"

    def test_extract_message_text_output_text(self):
        msg = {"role": "assistant", "content": [{"type": "output_text", "text": "reply"}]}
        assert self.fmt.extract_message_text(msg) == "reply"

    def test_extract_message_text_string(self):
        msg = {"role": "user", "content": "plain string"}
        assert self.fmt.extract_message_text(msg) == "plain string"

    # -- History pairs --

    def test_extract_history_pairs(self):
        body = {"input": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a2"}]},
            {"role": "user", "content": "current"},
        ]}
        pairs = self.fmt.extract_history_pairs(body)
        assert len(pairs) == 4
        assert pairs[0].role == "user"
        assert pairs[0].content == "q1"
        assert pairs[1].role == "assistant"
        assert pairs[1].content == "a1"
        assert pairs[2].content == "q2"
        assert pairs[3].content == "a2"

    def test_extract_history_pairs_with_tool_calls(self):
        """Tool call items interspersed should be skipped when pairing."""
        body = {"input": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
            {"type": "function_call", "name": "tool", "arguments": "{}"},
            {"type": "function_call_output", "output": "result"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a2"}]},
            {"role": "user", "content": "current"},
        ]}
        pairs = self.fmt.extract_history_pairs(body)
        assert len(pairs) == 4
        assert pairs[0].content == "q1"
        assert pairs[2].content == "q2"

    def test_extract_history_pairs_empty(self):
        body = {"input": [{"role": "user", "content": "only one"}]}
        assert self.fmt.extract_history_pairs(body) == []

    def test_extract_history_pairs_non_list(self):
        body = {"input": "just a string"}
        assert self.fmt.extract_history_pairs(body) == []

    # -- has_messages / get_messages --

    def test_has_messages(self):
        assert self.fmt.has_messages({"input": []}) is True
        assert self.fmt.has_messages({"input": "string"}) is False
        assert self.fmt.has_messages({"messages": []}) is False
        assert self.fmt.has_messages({}) is False

    def test_get_messages(self):
        body = {"input": [{"role": "user", "content": "hi"}]}
        assert len(self.fmt.get_messages(body)) == 1

    def test_get_messages_non_list(self):
        body = {"input": "string"}
        assert self.fmt.get_messages(body) == []

    # -- Context injection --

    def test_inject_context_into_instructions(self):
        body = {"instructions": "Be helpful.", "input": []}
        result = self.fmt.inject_context(body, "topic summary")
        assert "<system-reminder>" in result["instructions"]
        assert "topic summary" in result["instructions"]
        assert "Be helpful." in result["instructions"]

    def test_inject_context_no_instructions(self):
        body = {"input": [{"role": "user", "content": "hi"}]}
        result = self.fmt.inject_context(body, "ctx")
        assert "<system-reminder>" in result["instructions"]
        assert "ctx" in result["instructions"]

    def test_inject_context_empty_prepend(self):
        body = {"instructions": "original", "input": []}
        result = self.fmt.inject_context(body, "")
        assert result is body  # no copy

    def test_inject_context_does_not_mutate(self):
        body = {"instructions": "original", "input": [{"role": "user", "content": "hi"}]}
        result = self.fmt.inject_context(body, "ctx")
        assert body["instructions"] == "original"
        assert result["instructions"] != "original"

    # -- Session markers --

    def test_extract_session_id(self):
        body = {"input": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "hello <!-- vc:session=abc-123-def -->"},
            ]},
        ]}
        assert self.fmt.extract_session_id(body) == "abc-123-def"

    def test_extract_session_id_string_content(self):
        body = {"input": [
            {"role": "assistant", "content": "hello <!-- vc:session=abc-123 -->"},
        ]}
        assert self.fmt.extract_session_id(body) == "abc-123"

    def test_extract_session_id_none(self):
        body = {"input": [{"role": "user", "content": "hi"}]}
        assert self.fmt.extract_session_id(body) is None

    def test_extract_session_id_no_input(self):
        body = {}
        assert self.fmt.extract_session_id(body) is None

    def test_strip_session_markers_content_blocks(self):
        body = {"input": [
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "hello <!-- vc:session=abc-123 -->"},
            ]},
        ]}
        result = self.fmt.strip_session_markers(body)
        assert "vc:session" not in result["input"][0]["content"][0]["text"]

    def test_strip_session_markers_string(self):
        body = {"input": [
            {"role": "assistant", "content": "hello <!-- vc:session=abc-123 -->"},
        ]}
        result = self.fmt.strip_session_markers(body)
        assert "vc:session" not in result["input"][0]["content"]

    def test_strip_session_markers_no_markers(self):
        body = {"input": [{"role": "assistant", "content": "hello"}]}
        result = self.fmt.strip_session_markers(body)
        assert result is body

    def test_inject_session_marker(self):
        response = {"output": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "hello"},
            ]},
        ]}
        marker = "\n<!-- vc:session=test-123 -->"
        result = self.fmt.inject_session_marker(response, marker)
        assert result["output"][0]["content"][0]["text"].endswith(marker)
        # original not mutated
        assert response["output"][0]["content"][0]["text"] == "hello"

    def test_inject_session_marker_no_output_text(self):
        response = {"output": []}
        marker = "\n<!-- vc:session=test -->"
        result = self.fmt.inject_session_marker(response, marker)
        assert len(result["output"]) == 1
        assert result["output"][0]["content"][0]["text"] == marker

    # -- SSE emission --

    def test_emit_session_marker_sse(self):
        data = self.fmt.emit_session_marker_sse("test-session-id")
        assert b"response.output_text.delta" in data
        assert b"vc:session=test-session-id" in data
        # Verify it's valid SSE
        assert data.startswith(b"event: response.output_text.delta\n")

    # -- Delta text extraction --

    def test_extract_delta_text(self):
        data = {
            "type": "response.output_text.delta",
            "delta": "hello",
        }
        assert self.fmt.extract_delta_text(data) == "hello"

    def test_extract_delta_text_other_event(self):
        data = {"type": "response.created"}
        assert self.fmt.extract_delta_text(data) == ""

    def test_extract_delta_text_empty_delta(self):
        data = {"type": "response.output_text.delta", "delta": ""}
        assert self.fmt.extract_delta_text(data) == ""

    # -- Assistant text extraction --

    def test_extract_assistant_text(self):
        response = {"output": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "the answer"},
            ]},
        ]}
        assert self.fmt.extract_assistant_text(response) == "the answer"

    def test_extract_assistant_text_multiple_items(self):
        response = {"output": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "part1"},
            ]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "part2"},
            ]},
        ]}
        text = self.fmt.extract_assistant_text(response)
        assert "part1" in text
        assert "part2" in text

    def test_extract_assistant_text_empty(self):
        response = {"output": []}
        assert self.fmt.extract_assistant_text(response) == ""

    # -- Token estimation --

    def test_estimate_system_tokens(self):
        body = {"instructions": "x" * 400}
        tokens = self.fmt._estimate_system_tokens(body)
        assert tokens == 100  # 400 chars // 4

    def test_estimate_payload_tokens_with_bare_items(self):
        body = {"input": [
            {"role": "user", "content": "hello"},  # 5 chars -> 1 token
            {"type": "function_call", "name": "get_weather", "arguments": '{"city": "NYC"}'},
            {"type": "function_call_output", "output": "sunny and warm"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "The weather is nice."}]},
        ], "instructions": ""}
        tokens = self.fmt.estimate_payload_tokens(body)
        assert tokens > 0
        # Should include tokens from bare items too
        # function_call: "get_weather" + '{"city": "NYC"}' = 11 + 15 = 26 chars -> 6 tokens
        # function_call_output: "" + "sunny and warm" = 14 chars -> 3 tokens
        assert tokens >= 1 + 6 + 3  # user + fn_call + fn_output (at minimum)

    def test_estimate_payload_tokens_basic(self):
        body = {"input": [
            {"role": "user", "content": "x" * 100},
        ], "instructions": "y" * 200}
        tokens = self.fmt.estimate_payload_tokens(body)
        assert tokens == 25 + 50  # 100//4 + 200//4

    # -- Fingerprinting --

    def test_compute_fingerprint(self):
        body = {"input": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a2"}]},
            {"role": "user", "content": "current"},
        ]}
        fp = self.fmt.compute_fingerprint(body)
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_compute_fingerprint_too_few(self):
        body = {"input": [{"role": "user", "content": "only one"}]}
        assert self.fmt.compute_fingerprint(body) == ""

    def test_compute_fingerprint_skips_bare_items(self):
        body = {"input": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"type": "function_call", "name": "tool", "arguments": "{}"},
            {"type": "function_call_output", "output": "result"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "current"},
        ]}
        fp = self.fmt.compute_fingerprint(body)
        assert isinstance(fp, str)
        assert len(fp) == 16

    # -- Tool interception --

    def test_supports_tool_interception(self):
        assert self.fmt.supports_tool_interception is True

    def test_inject_tools(self):
        body = {"input": [{"role": "user", "content": "hi"}], "tools": []}
        tool_defs = [
            {"name": "vc_expand_topic", "description": "Expand", "input_schema": {
                "type": "object", "properties": {"tag": {"type": "string"}},
            }},
        ]
        result = self.fmt.inject_tools(body, tool_defs)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["name"] == "vc_expand_topic"
        assert result["tools"][0]["parameters"]["type"] == "object"

    def test_inject_tools_existing(self):
        body = {"input": [], "tools": [
            {"type": "function", "name": "existing_tool"},
        ]}
        tool_defs = [{"name": "vc_expand_topic", "description": "Expand"}]
        result = self.fmt.inject_tools(body, tool_defs)
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "existing_tool"
        assert result["tools"][1]["name"] == "vc_expand_topic"

    def test_inject_tools_no_existing(self):
        body = {"input": []}
        tool_defs = [{"name": "vc_expand_topic", "description": "Expand"}]
        result = self.fmt.inject_tools(body, tool_defs)
        assert len(result["tools"]) == 1

    def test_inject_tools_sets_required_when_requested(self):
        body = {"input": []}
        tool_defs = [{"name": "vc_expand_topic", "description": "Expand"}]
        result = self.fmt.inject_tools(body, tool_defs, require_tool_use=True)
        assert result["tool_choice"] == "required"

    def test_inject_tools_respects_none_choice(self):
        body = {"input": [], "tool_choice": "none"}
        tool_defs = [{"name": "vc_expand_topic", "description": "Expand"}]
        result = self.fmt.inject_tools(body, tool_defs, require_tool_use=True)
        assert result is body

    def test_is_tool_use_event_output_item_added(self):
        data = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic"},
        }
        assert self.fmt.is_tool_use_event(data) is True

    def test_is_tool_use_event_arguments_delta(self):
        data = {
            "type": "response.function_call_arguments.delta",
            "delta": '{"tag":',
        }
        assert self.fmt.is_tool_use_event(data) is True

    def test_is_tool_use_event_arguments_done(self):
        data = {
            "type": "response.function_call_arguments.done",
            "arguments": '{"tag": "cooking"}',
        }
        assert self.fmt.is_tool_use_event(data) is True

    def test_is_tool_use_event_text_delta(self):
        data = {"type": "response.output_text.delta", "delta": "hello"}
        assert self.fmt.is_tool_use_event(data) is False

    def test_is_tool_use_event_message_item(self):
        data = {
            "type": "response.output_item.added",
            "item": {"type": "message", "role": "assistant"},
        }
        assert self.fmt.is_tool_use_event(data) is False

    def test_extract_tool_calls(self):
        output = [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Let me check."},
            ]},
            {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic",
             "arguments": '{"tag": "cooking"}'},
            {"type": "function_call", "call_id": "c2", "name": "vc_find_quote",
             "arguments": '{"query": "recipe"}'},
        ]
        calls = self.fmt.extract_tool_calls(output)
        assert len(calls) == 2
        assert calls[0]["id"] == "c1"
        assert calls[0]["name"] == "vc_expand_topic"
        assert calls[0]["input"] == {"tag": "cooking"}
        assert calls[1]["id"] == "c2"
        assert calls[1]["name"] == "vc_find_quote"
        assert calls[1]["input"] == {"query": "recipe"}

    def test_extract_tool_calls_bad_json(self):
        output = [
            {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic",
             "arguments": "not json"},
        ]
        calls = self.fmt.extract_tool_calls(output)
        assert len(calls) == 1
        assert calls[0]["input"] == {}

    def test_extract_tool_calls_empty_args(self):
        output = [
            {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic",
             "arguments": ""},
        ]
        calls = self.fmt.extract_tool_calls(output)
        assert len(calls) == 1
        assert calls[0]["input"] == {}

    def test_build_tool_results(self):
        results = [
            {"tool_use_id": "c1", "content": "expanded cooking content"},
            {"call_id": "c2", "content": "found quote"},
        ]
        formatted = self.fmt.build_tool_results(results)
        assert len(formatted) == 2
        assert formatted[0]["type"] == "function_call_output"
        assert formatted[0]["call_id"] == "c1"
        assert formatted[0]["output"] == "expanded cooking content"
        assert formatted[1]["call_id"] == "c2"

    def test_build_continuation_request(self):
        original = {
            "model": "codex-mini",
            "instructions": "Be helpful.",
            "input": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "vc_expand_topic"}],
        }
        assistant_content = [
            {"type": "message", "content": [{"type": "output_text", "text": "Let me check."}]},
            {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic",
             "arguments": '{"tag": "cooking"}'},
        ]
        tool_results = [
            {"type": "function_call_output", "call_id": "c1", "output": "expanded content"},
        ]
        result = self.fmt.build_continuation_request(original, assistant_content, tool_results)
        assert result["model"] == "codex-mini"
        assert result["stream"] is True  # Codex Responses API requires streaming
        assert result["instructions"] == "Be helpful."
        # Original input + function_call + function_call_output
        assert len(result["input"]) == 3
        assert result["input"][0]["role"] == "user"
        assert result["input"][1]["type"] == "function_call"
        assert result["input"][2]["type"] == "function_call_output"

    def test_build_continuation_does_not_mutate_original(self):
        original = {
            "model": "codex-mini",
            "input": [{"role": "user", "content": "hi"}],
        }
        import copy
        saved = copy.deepcopy(original)
        self.fmt.build_continuation_request(
            original,
            [{"type": "function_call", "call_id": "c1", "name": "test", "arguments": "{}"}],
            [{"type": "function_call_output", "call_id": "c1", "output": "result"}],
        )
        assert original == saved


# ---------------------------------------------------------------------------
# Filter body messages with OpenAI Responses format
# ---------------------------------------------------------------------------

class TestFilterBodyMessagesResponses:
    """Test filter_body_messages with OpenAI Responses API payloads."""

    def test_filter_responses_basic(self):
        from virtual_context.proxy.message_filter import filter_body_messages
        from virtual_context.core.turn_tag_index import TurnTagIndex
        from virtual_context.types import TurnTagEntry

        tti = TurnTagIndex()
        tti.append(TurnTagEntry(turn_number=0, message_hash="h0", tags=["cooking"], primary_tag="cooking"))
        tti.append(TurnTagEntry(turn_number=1, message_hash="h1", tags=["music"], primary_tag="music"))
        tti.append(TurnTagEntry(turn_number=2, message_hash="h2", tags=["cooking"], primary_tag="cooking"))
        tti.append(TurnTagEntry(turn_number=3, message_hash="h3", tags=["cooking"], primary_tag="cooking"))

        body = {"input": [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "current"},
        ]}

        result, dropped = filter_body_messages(
            body, tti, ["cooking"], recent_turns=1, fmt=OpenAIResponsesFormat(),
        )
        assert dropped > 0
        assert "input" in result
        assert "messages" not in result

    def test_filter_responses_with_bare_items(self):
        """Bare function_call items should always be kept."""
        from virtual_context.proxy.message_filter import filter_body_messages
        from virtual_context.core.turn_tag_index import TurnTagIndex
        from virtual_context.types import TurnTagEntry

        tti = TurnTagIndex()
        tti.append(TurnTagEntry(turn_number=0, message_hash="h0", tags=["cooking"], primary_tag="cooking"))
        tti.append(TurnTagEntry(turn_number=1, message_hash="h1", tags=["music"], primary_tag="music"))
        tti.append(TurnTagEntry(turn_number=2, message_hash="h2", tags=["cooking"], primary_tag="cooking"))
        tti.append(TurnTagEntry(turn_number=3, message_hash="h3", tags=["cooking"], primary_tag="cooking"))

        body = {"input": [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"type": "function_call", "name": "tool", "arguments": "{}"},  # bare item
            {"type": "function_call_output", "output": "result"},  # bare item
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "current"},
        ]}

        result, dropped = filter_body_messages(
            body, tti, ["cooking"], recent_turns=1, fmt=OpenAIResponsesFormat(),
        )
        # Bare items (no role) should be preserved in the output
        bare_items = [m for m in result["input"] if m.get("type") in ("function_call", "function_call_output")]
        assert len(bare_items) == 2


# ---------------------------------------------------------------------------
# Filter body messages with Gemini format
# ---------------------------------------------------------------------------

class TestFilterBodyMessagesGemini:
    """Test _filter_body_messages with Gemini-format payloads."""

    def test_filter_gemini_basic(self):
        from virtual_context.proxy.server import _filter_body_messages
        from virtual_context.core.turn_tag_index import TurnTagIndex
        from virtual_context.types import TurnTagEntry
        from virtual_context.proxy.formats import GeminiFormat

        tti = TurnTagIndex()
        tti.append(TurnTagEntry(turn_number=0, message_hash="h0", tags=["cooking"], primary_tag="cooking"))
        tti.append(TurnTagEntry(turn_number=1, message_hash="h1", tags=["music"], primary_tag="music"))
        tti.append(TurnTagEntry(turn_number=2, message_hash="h2", tags=["cooking"], primary_tag="cooking"))
        tti.append(TurnTagEntry(turn_number=3, message_hash="h3", tags=["cooking"], primary_tag="cooking"))

        body = {"contents": [
            {"role": "user", "parts": [{"text": "q0"}]},
            {"role": "model", "parts": [{"text": "a0"}]},
            {"role": "user", "parts": [{"text": "q1"}]},
            {"role": "model", "parts": [{"text": "a1"}]},
            {"role": "user", "parts": [{"text": "q2"}]},
            {"role": "model", "parts": [{"text": "a2"}]},
            {"role": "user", "parts": [{"text": "q3"}]},
            {"role": "model", "parts": [{"text": "a3"}]},
            {"role": "user", "parts": [{"text": "current"}]},
        ]}

        result, dropped = _filter_body_messages(
            body, tti, ["cooking"], recent_turns=1, fmt=GeminiFormat(),
        )
        assert dropped > 0
        # The result should use 'contents' key, not 'messages'
        assert "contents" in result
        assert "messages" not in result



# ---------------------------------------------------------------------------
# Paging tool support (Phase 4)
# ---------------------------------------------------------------------------

class TestPagingToolSupport:
    """Test format-specific paging tool injection and continuation building."""

    def test_anthropic_build_continuation(self):
        fmt = AnthropicFormat()
        original = {
            "model": "claude-3-opus",
            "max_tokens": 4096,
            "system": "Be helpful.",
            "tools": [{"name": "vc_expand_topic"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        assistant_content = [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t1", "name": "vc_expand_topic", "input": {"tag": "cooking"}},
        ]
        tool_results = [
            {"type": "tool_result", "tool_use_id": "t1", "content": "expanded content"},
        ]
        result = fmt.build_continuation_request(original, assistant_content, tool_results)
        assert result["model"] == "claude-3-opus"
        assert result["stream"] is False
        assert result["system"] == "Be helpful."
        assert len(result["messages"]) == 3
        assert result["messages"][-2]["role"] == "assistant"
        assert result["messages"][-1]["role"] == "user"

    def test_gemini_build_continuation(self):
        fmt = GeminiFormat()
        original = {
            "model": "gemini-2.0-flash",
            "system_instruction": {"parts": [{"text": "Be helpful."}]},
            "tools": [{"functionDeclarations": [{"name": "vc_expand_topic"}]}],
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        assistant_content = [
            {"type": "text", "text": "Let me expand that."},
            {"type": "tool_use", "id": "t1", "name": "vc_expand_topic", "input": {"tag": "cooking"}},
        ]
        tool_results = [
            {"name": "vc_expand_topic", "content": "expanded content"},
        ]
        result = fmt.build_continuation_request(original, assistant_content, tool_results)
        assert result["model"] == "gemini-2.0-flash"
        assert "system_instruction" in result
        assert len(result["contents"]) == 3  # original + model + user
        # Model message should have functionCall
        model_msg = result["contents"][-2]
        assert model_msg["role"] == "model"
        assert any("functionCall" in p for p in model_msg["parts"])
        # User message should have functionResponse
        user_msg = result["contents"][-1]
        assert user_msg["role"] == "user"
        assert any("functionResponse" in p for p in user_msg["parts"])

    def test_openai_does_not_support_tool_interception(self):
        fmt = OpenAIFormat()
        assert fmt.supports_tool_interception is False

    def test_server_inject_vc_tools_gemini(self):
        from virtual_context.proxy.server import _inject_vc_tools

        class FakeEngine:
            pass

        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": []}
        result = _inject_vc_tools(body, FakeEngine())
        # Should use Gemini's functionDeclarations format
        assert "tools" in result
        decls = result["tools"][0]["functionDeclarations"]
        names = [d["name"] for d in decls]
        assert "vc_expand_topic" in names
        assert "vc_collapse_topic" in names
        assert "vc_find_quote" in names

    def test_server_inject_vc_tools_anthropic(self):
        from virtual_context.proxy.server import _inject_vc_tools

        class FakeEngine:
            pass

        body = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}
        result = _inject_vc_tools(body, FakeEngine())
        # Should use Anthropic's flat tools format
        tool_names = [t["name"] for t in result["tools"]]
        assert "vc_expand_topic" in tool_names
        assert "vc_collapse_topic" in tool_names

    def test_server_build_continuation_gemini(self):
        from virtual_context.proxy.server import _build_continuation_request
        body = {
            "model": "gemini-pro",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        assistant_content = [
            {"type": "tool_use", "name": "vc_expand_topic", "input": {"tag": "test"}},
        ]
        tool_results = [{"name": "vc_expand_topic", "content": "result"}]
        result = _build_continuation_request(body, assistant_content, tool_results)
        assert "contents" in result
        assert result["contents"][-2]["role"] == "model"

    def test_server_build_continuation_anthropic(self):
        from virtual_context.proxy.server import _build_continuation_request
        body = {
            "model": "claude-3-opus",
            "system": "sys",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assistant_content = [
            {"type": "tool_use", "id": "t1", "name": "vc_expand_topic", "input": {"tag": "test"}},
        ]
        tool_results = [
            {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
        ]
        result = _build_continuation_request(body, assistant_content, tool_results)
        assert "messages" in result
        assert result["messages"][-2]["role"] == "assistant"
        assert result["messages"][-1]["role"] == "user"

    def test_responses_build_continuation(self):
        fmt = OpenAIResponsesFormat()
        original = {
            "model": "codex-mini",
            "instructions": "Be helpful.",
            "input": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "vc_expand_topic"}],
        }
        assistant_content = [
            {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic",
             "arguments": '{"tag": "cooking"}'},
        ]
        tool_results = [
            {"type": "function_call_output", "call_id": "c1", "output": "expanded"},
        ]
        result = fmt.build_continuation_request(original, assistant_content, tool_results)
        assert result["model"] == "codex-mini"
        assert result["stream"] is True  # Codex Responses API requires streaming
        assert "input" in result
        assert result["input"][-2]["type"] == "function_call"
        assert result["input"][-1]["type"] == "function_call_output"

    def test_server_inject_vc_tools_responses(self):
        from virtual_context.proxy.server import _inject_vc_tools

        class FakeEngine:
            pass

        body = {"input": [{"role": "user", "content": "hi"}]}
        result = _inject_vc_tools(body, FakeEngine())
        assert "tools" in result
        names = [t["name"] for t in result["tools"]]
        assert "vc_expand_topic" in names
        assert "vc_collapse_topic" in names
        assert "vc_find_quote" in names
        # Verify Responses API format (type: "function")
        assert all(t["type"] == "function" for t in result["tools"])

    def test_server_build_continuation_responses(self):
        from virtual_context.proxy.server import _build_continuation_request
        body = {
            "model": "codex-mini",
            "instructions": "Be helpful.",
            "input": [{"role": "user", "content": "hi"}],
        }
        assistant_content = [
            {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic",
             "arguments": '{"tag": "test"}'},
        ]
        tool_results = [
            {"type": "function_call_output", "call_id": "c1", "output": "result"},
        ]
        result = _build_continuation_request(body, assistant_content, tool_results)
        assert "input" in result
        assert result["stream"] is True  # Codex Responses API requires streaming
        assert result["input"][-2]["type"] == "function_call"
        assert result["input"][-1]["type"] == "function_call_output"


class TestCrossFormat:
    """Verify that equivalent payloads produce consistent results across formats."""

    def test_inject_context_does_not_mutate_any_format(self):
        for fmt_name in ("anthropic", "openai", "openai_responses", "gemini"):
            fmt = get_format(fmt_name)
            if fmt_name == "gemini":
                body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
            elif fmt_name == "anthropic":
                body = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}
            elif fmt_name == "openai_responses":
                body = {"instructions": "sys", "input": [{"role": "user", "content": "hi"}]}
            else:
                body = {"messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ]}
            import copy
            original = copy.deepcopy(body)
            fmt.inject_context(body, "test context")
            assert body == original, f"{fmt_name} mutated the body"

    def test_all_formats_return_empty_for_no_user(self):
        for fmt_name in ("anthropic", "openai", "openai_responses", "gemini"):
            fmt = get_format(fmt_name)
            if fmt_name == "gemini":
                body = {"contents": []}
            elif fmt_name == "openai_responses":
                body = {"input": []}
            else:
                body = {"messages": []}
            assert fmt.extract_user_message(body) == "", f"{fmt_name} failed"


# ---------------------------------------------------------------------------
# Responses API SSE emission helpers
# ---------------------------------------------------------------------------

class TestResponsesSSEEmission:
    """Test the Responses API SSE emission helper functions."""

    def test_emit_text_as_responses_sse(self):
        from virtual_context.proxy.helpers import _emit_text_as_responses_sse
        events = _emit_text_as_responses_sse("Hello world", item_index=0)
        # Should produce 6 events
        assert len(events) == 6
        # Check event types in order
        decoded = [e.decode() for e in events]
        assert "response.output_item.added" in decoded[0]
        assert "response.content_part.added" in decoded[1]
        assert "response.output_text.delta" in decoded[2]
        assert "response.output_text.done" in decoded[3]
        assert "response.content_part.done" in decoded[4]
        assert "response.output_item.done" in decoded[5]
        # Verify the text content is present
        assert b"Hello world" in events[2]
        assert b"Hello world" in events[3]
        # Each event should be valid SSE (ends with \n\n)
        for evt in events:
            assert evt.endswith(b"\n\n")
        # Verify JSON parseable
        for evt in events:
            lines = evt.decode().strip().split("\n")
            data_line = [l for l in lines if l.startswith("data: ")][0]
            parsed = json.loads(data_line[6:])
            assert "type" in parsed

    def test_emit_text_as_responses_sse_item_index(self):
        from virtual_context.proxy.helpers import _emit_text_as_responses_sse
        events = _emit_text_as_responses_sse("test", item_index=3)
        data_line = [l for l in events[0].decode().strip().split("\n") if l.startswith("data: ")][0]
        parsed = json.loads(data_line[6:])
        assert parsed["output_index"] == 3

    def test_emit_tool_use_as_responses_sse(self):
        from virtual_context.proxy.helpers import _emit_tool_use_as_responses_sse
        tool = {"id": "c1", "name": "vc_expand_topic", "input": {"tag": "cooking"}}
        events = _emit_tool_use_as_responses_sse(tool, item_index=1)
        # Should produce 4 events
        assert len(events) == 4
        decoded = [e.decode() for e in events]
        assert "response.output_item.added" in decoded[0]
        assert "response.function_call_arguments.delta" in decoded[1]
        assert "response.function_call_arguments.done" in decoded[2]
        assert "response.output_item.done" in decoded[3]
        # Verify tool name and args are present
        assert b"vc_expand_topic" in events[0]
        assert b"cooking" in events[1]
        # Verify JSON parseable
        for evt in events:
            lines = evt.decode().strip().split("\n")
            data_line = [l for l in lines if l.startswith("data: ")][0]
            parsed = json.loads(data_line[6:])
            assert "type" in parsed

    def test_emit_tool_use_as_responses_sse_call_id(self):
        from virtual_context.proxy.helpers import _emit_tool_use_as_responses_sse
        tool = {"call_id": "call_abc", "name": "test_tool", "input": {}}
        events = _emit_tool_use_as_responses_sse(tool, item_index=0)
        data_line = [l for l in events[0].decode().strip().split("\n") if l.startswith("data: ")][0]
        parsed = json.loads(data_line[6:])
        assert parsed["item"]["call_id"] == "call_abc"

    def test_emit_response_done_sse(self):
        from virtual_context.proxy.helpers import _emit_response_done_sse
        output_items = [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Here is the answer."},
            ]},
        ]
        usage = {"input_tokens": 100, "output_tokens": 50}
        events = _emit_response_done_sse(output_items, usage=usage)
        assert len(events) == 1
        decoded = events[0].decode()
        assert "response.completed" in decoded
        data_line = [l for l in decoded.strip().split("\n") if l.startswith("data: ")][0]
        parsed = json.loads(data_line[6:])
        assert parsed["type"] == "response.completed"
        assert parsed["response"]["status"] == "completed"
        assert len(parsed["response"]["output"]) == 1
        assert parsed["response"]["usage"]["input_tokens"] == 100

    def test_emit_response_done_sse_no_usage(self):
        from virtual_context.proxy.helpers import _emit_response_done_sse
        events = _emit_response_done_sse([], usage=None)
        assert len(events) == 1
        data_line = [l for l in events[0].decode().strip().split("\n") if l.startswith("data: ")][0]
        parsed = json.loads(data_line[6:])
        assert parsed["response"]["usage"] == {}


# ---------------------------------------------------------------------------
# Responses API stream interception integration
# ---------------------------------------------------------------------------

class TestResponsesStreamInterception:
    """Integration tests for Responses API streaming tool interception."""

    def test_parse_sse_events_responses_format(self):
        """Verify _parse_sse_events works with Responses API event format."""
        from virtual_context.proxy.helpers import _parse_sse_events
        raw = (
            b"event: response.output_item.added\n"
            b"data: {\"type\":\"response.output_item.added\","
            b"\"item\":{\"type\":\"function_call\",\"name\":\"vc_expand_topic\"}}\n\n"
            b"event: response.function_call_arguments.delta\n"
            b"data: {\"type\":\"response.function_call_arguments.delta\","
            b"\"delta\":\"{\\\"tag\\\": \\\"cooking\\\"}\"}\n\n"
        )
        events, remainder = _parse_sse_events(raw)
        assert len(events) == 2
        assert events[0][0] == "response.output_item.added"
        assert events[1][0] == "response.function_call_arguments.delta"
        assert remainder == b""

    def test_is_tool_use_event_integration(self):
        """Verify format.is_tool_use_event works for all Responses API tool events."""
        fmt = OpenAIResponsesFormat()
        # All tool-related events should return True
        assert fmt.is_tool_use_event({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "vc_expand_topic"},
        })
        assert fmt.is_tool_use_event({
            "type": "response.function_call_arguments.delta",
            "delta": '{"tag":',
        })
        assert fmt.is_tool_use_event({
            "type": "response.function_call_arguments.done",
            "arguments": '{"tag": "cooking"}',
        })
        # Non-tool events should return False
        assert not fmt.is_tool_use_event({
            "type": "response.output_text.delta",
            "delta": "hello",
        })
        assert not fmt.is_tool_use_event({
            "type": "response.completed",
        })
        assert not fmt.is_tool_use_event({
            "type": "response.output_item.added",
            "item": {"type": "message"},
        })

    def test_extract_and_build_roundtrip(self):
        """Verify extract_tool_calls + build_tool_results + build_continuation roundtrips."""
        fmt = OpenAIResponsesFormat()
        # Simulated upstream output
        output = [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Let me check."},
            ]},
            {"type": "function_call", "call_id": "c1", "name": "vc_expand_topic",
             "arguments": '{"tag": "cooking"}'},
        ]
        # Extract tool calls
        calls = fmt.extract_tool_calls(output)
        assert len(calls) == 1
        assert calls[0]["name"] == "vc_expand_topic"

        # Build tool results (simulating execute_vc_tool)
        raw_results = [{"tool_use_id": calls[0]["id"], "content": "expanded cooking content"}]
        formatted_results = fmt.build_tool_results(raw_results)
        assert formatted_results[0]["type"] == "function_call_output"
        assert formatted_results[0]["call_id"] == "c1"

        # Build continuation
        original = {
            "model": "codex-mini",
            "instructions": "Be helpful.",
            "input": [{"role": "user", "content": "What did we discuss about cooking?"}],
        }
        cont = fmt.build_continuation_request(original, output, formatted_results)
        assert cont["stream"] is True  # Codex Responses API requires streaming
        # Should have: user + function_call + function_call_output
        assert len(cont["input"]) == 3
        assert cont["input"][1]["type"] == "function_call"
        assert cont["input"][2]["type"] == "function_call_output"
        assert cont["input"][2]["output"] == "expanded cooking content"
