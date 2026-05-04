"""Tests for conversation export adapters."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from virtual_context.import_adapters import get_adapter, ADAPTERS
from virtual_context.import_adapters.chatgpt import ChatGPTAdapter
from virtual_context.import_adapters.claude import ClaudeAdapter
from virtual_context.import_adapters.grok import GrokAdapter
from virtual_context.import_adapters.loader import load_from_path


class TestAdapterRegistry:
    """Tests for adapter registry and factory."""

    def test_get_adapter_chatgpt(self):
        adapter = get_adapter("chatgpt")
        assert isinstance(adapter, ChatGPTAdapter)
        assert adapter.name == "chatgpt"

    def test_get_adapter_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_adapter("unknown")


class TestChatGPTAdapter:
    """Tests for ChatGPT export adapter."""

    def test_extract_messages_basic(self):
        adapter = ChatGPTAdapter()
        data = {
            "conversation_id": "abc-123",
            "title": "Test Chat",
            "messages": [
                {"role": "user", "text": "Hello", "create_time": 1715848606.0},
                {"role": "assistant", "text": "Hi there!", "create_time": 1715848607.0},
            ],
        }
        messages = adapter.extract_messages(data)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[0].timestamp == datetime.fromtimestamp(1715848606.0)
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"

    def test_extract_conversation_id(self):
        adapter = ChatGPTAdapter()
        data = {"conversation_id": "abc-123"}
        assert adapter.extract_conversation_id(data) == "abc-123"

    def test_extract_messages_empty(self):
        adapter = ChatGPTAdapter()
        data = {"conversation_id": "abc-123", "messages": []}
        messages = adapter.extract_messages(data)
        assert messages == []

    def test_extract_messages_missing_timestamp(self):
        adapter = ChatGPTAdapter()
        data = {
            "conversation_id": "abc-123",
            "messages": [{"role": "user", "text": "No timestamp"}],
        }
        messages = adapter.extract_messages(data)
        assert len(messages) == 1
        assert messages[0].timestamp is None


class TestClaudeAdapter:
    """Tests for Claude export adapter."""

    def test_extract_messages_normalizes_human_role(self):
        adapter = ClaudeAdapter()
        data = {
            "uuid": "abc-123",
            "messages": [
                {"role": "human", "text": "Hello", "created_at": "2025-09-28T16:41:19.245108Z"},
                {"role": "assistant", "text": "Hi!", "created_at": "2025-09-28T16:41:20.000000Z"},
            ],
        }
        messages = adapter.extract_messages(data)
        assert messages[0].role == "user"  # normalized from "human"
        assert messages[1].role == "assistant"

    def test_extract_messages_parses_iso_timestamp(self):
        adapter = ClaudeAdapter()
        data = {
            "uuid": "abc-123",
            "messages": [{"role": "human", "text": "Hi", "created_at": "2025-09-28T16:41:19.245108Z"}],
        }
        messages = adapter.extract_messages(data)
        assert messages[0].timestamp is not None
        assert messages[0].timestamp.year == 2025

    def test_extract_conversation_id(self):
        adapter = ClaudeAdapter()
        data = {"uuid": "abc-123-uuid"}
        assert adapter.extract_conversation_id(data) == "abc-123-uuid"


class TestGrokAdapter:
    """Tests for Grok export adapter."""

    def test_extract_messages_two_key_envelope(self):
        adapter = GrokAdapter()
        data = {
            "conversation": {"id": "conv-uuid"},
            "responses": [
                {
                    "response": {
                        "sender": "human",
                        "message": "Hello",
                        "create_time": {"$date": {"$numberLong": "1753841416257"}},
                    }
                },
                {
                    "response": {
                        "sender": "ASSISTANT",
                        "message": "Hi there!",
                        "create_time": {"$date": {"$numberLong": "1753841417000"}},
                    }
                },
            ],
        }
        messages = adapter.extract_messages(data)
        assert len(messages) == 2
        assert messages[0].role == "user"  # normalized from "human"
        assert messages[1].role == "assistant"  # normalized from "ASSISTANT"

    def test_extract_messages_mongodb_timestamp(self):
        adapter = GrokAdapter()
        data = {
            "conversation": {"id": "conv-uuid"},
            "responses": [
                {
                    "response": {
                        "sender": "human",
                        "message": "Hi",
                        "create_time": {"$date": {"$numberLong": "1753841416257"}},
                    }
                }
            ],
        }
        messages = adapter.extract_messages(data)
        assert messages[0].timestamp == datetime.fromtimestamp(1753841416257 / 1000)

    def test_extract_conversation_id(self):
        adapter = GrokAdapter()
        data = {"conversation": {"id": "conv-uuid"}}
        assert adapter.extract_conversation_id(data) == "conv-uuid"

    def test_thinking_trace_not_in_content(self):
        adapter = GrokAdapter()
        data = {
            "conversation": {"id": "conv-uuid"},
            "responses": [
                {
                    "response": {
                        "sender": "ASSISTANT",
                        "message": "The answer is 42",
                        "thinking_trace": "Let me reason through this...",
                        "steps": [{"type": "reasoning", "text": "First..."}],
                    }
                }
            ],
        }
        messages = adapter.extract_messages(data)
        assert messages[0].content == "The answer is 42"
        assert "thinking" not in messages[0].content.lower()


class TestLoader:
    """Tests for file/directory loading utilities."""

    def test_load_single_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "chat.json"
        test_file.write_text(json.dumps({
            "conversation_id": "test-123",
            "messages": [
                {"role": "user", "text": "Hello", "create_time": 1715848606.0},
            ],
        }))

        adapter = ChatGPTAdapter()
        results = list(load_from_path(test_file, adapter))

        assert len(results) == 1
        conv_id, messages = results[0]
        assert conv_id == "test-123"
        assert len(messages) == 1

    def test_load_directory(self, tmp_path: Path) -> None:
        for i in range(3):
            test_file = tmp_path / f"chat_{i}.json"
            test_file.write_text(json.dumps({
                "conversation_id": f"conv-{i}",
                "messages": [
                    {"role": "user", "text": f"Hello {i}", "create_time": 1715848606.0},
                ],
            }))

        adapter = ChatGPTAdapter()
        results = list(load_from_path(tmp_path, adapter))

        assert len(results) == 3

    def test_load_directory_skips_invalid_json(self, tmp_path: Path) -> None:
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(json.dumps({
            "conversation_id": "valid",
            "messages": [{"role": "user", "text": "Hi", "create_time": 1.0}],
        }))

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json{{{")

        adapter = ChatGPTAdapter()
        results = list(load_from_path(tmp_path, adapter))

        assert len(results) == 1
        assert results[0][0] == "valid"

    def test_load_directory_skips_empty_messages(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(json.dumps({
            "conversation_id": "empty",
            "messages": [],
        }))

        adapter = ChatGPTAdapter()
        results = list(load_from_path(tmp_path, adapter))

        assert len(results) == 0
