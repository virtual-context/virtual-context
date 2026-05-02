"""Tests for conversation export adapters."""

from datetime import datetime

import pytest

from virtual_context.import_adapters import get_adapter, ADAPTERS
from virtual_context.import_adapters.chatgpt import ChatGPTAdapter


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
