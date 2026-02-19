"""Tests for virtual_context.proxy.formats — PayloadFormat ABC + implementations."""

import json
import pytest

from virtual_context.proxy.formats import (
    PayloadFormat,
    AnthropicFormat,
    OpenAIFormat,
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

    def test_get_format_by_name(self):
        assert get_format("anthropic").name == "anthropic"
        assert get_format("openai").name == "openai"
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
        assert "<virtual-context>" in result["system"]
        assert "topic summary" in result["system"]
        assert "Be helpful." in result["system"]

    def test_inject_context_list_system(self):
        body = {"system": [{"type": "text", "text": "Be helpful."}], "messages": []}
        result = self.fmt.inject_context(body, "topic summary")
        assert isinstance(result["system"], list)
        assert result["system"][0]["text"].startswith("<virtual-context>")

    def test_inject_context_no_system(self):
        body = {"messages": []}
        result = self.fmt.inject_context(body, "ctx")
        assert "<virtual-context>" in result.get("system", "")

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
        assert "<virtual-context>" in result["messages"][0]["content"]
        assert "Be helpful." in result["messages"][0]["content"]

    def test_inject_context_without_system(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = self.fmt.inject_context(body, "ctx text")
        assert result["messages"][0]["role"] == "system"
        assert "<virtual-context>" in result["messages"][0]["content"]

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
        assert si["parts"][0]["text"].startswith("<virtual-context>")

    def test_inject_context_with_existing_system_instruction(self):
        body = {
            "system_instruction": {"parts": [{"text": "Be helpful."}]},
            "contents": [],
        }
        result = self.fmt.inject_context(body, "ctx")
        parts = result["system_instruction"]["parts"]
        assert len(parts) == 2
        assert "<virtual-context>" in parts[0]["text"]
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
        assert "<virtual-context>" in result["system_instruction"]["parts"][0]["text"]

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

    def test_filter_gemini_no_drop_when_broad(self):
        from virtual_context.proxy.server import _filter_body_messages
        from virtual_context.core.turn_tag_index import TurnTagIndex
        from virtual_context.types import TurnTagEntry
        from virtual_context.proxy.formats import GeminiFormat

        tti = TurnTagIndex()
        tti.append(TurnTagEntry(turn_number=0, message_hash="h0", tags=["a"], primary_tag="a"))

        body = {"contents": [
            {"role": "user", "parts": [{"text": "q0"}]},
            {"role": "model", "parts": [{"text": "a0"}]},
            {"role": "user", "parts": [{"text": "current"}]},
        ]}

        result, dropped = _filter_body_messages(
            body, tti, ["a"], broad=True, fmt=GeminiFormat(),
        )
        assert dropped == 0


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


class TestCrossFormat:
    """Verify that equivalent payloads produce consistent results across formats."""

    def test_inject_context_does_not_mutate_any_format(self):
        for fmt_name in ("anthropic", "openai", "gemini"):
            fmt = get_format(fmt_name)
            if fmt_name == "gemini":
                body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
            elif fmt_name == "anthropic":
                body = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}
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
        for fmt_name in ("anthropic", "openai", "gemini"):
            fmt = get_format(fmt_name)
            if fmt_name == "gemini":
                body = {"contents": []}
            else:
                body = {"messages": []}
            assert fmt.extract_user_message(body) == "", f"{fmt_name} failed"
