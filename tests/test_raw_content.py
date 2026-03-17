"""Tests for lossless raw_content on Message."""

import json
import pytest
from virtual_context.types import Message
from virtual_context.storage.sqlite import SQLiteStore


def test_message_raw_content_default_none():
    m = Message(role="user", content="hello")
    assert m.raw_content is None


def test_message_raw_content_stores_blocks():
    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "tool_result", "tool_use_id": "abc", "content": "world"},
    ]
    m = Message(role="user", content="hello", raw_content=blocks)
    assert m.raw_content == blocks
    assert len(m.raw_content) == 2
    assert m.raw_content[0]["type"] == "text"


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def test_save_and_get_turn_message_with_raw_content(store):
    user_raw = json.dumps([{"type": "text", "text": "hello"}])
    asst_raw = json.dumps([
        {"type": "text", "text": "Hi!"},
        {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}},
    ])
    store.save_turn_message(
        "conv-1", 0, "hello", "Hi!",
        user_raw_content=user_raw, assistant_raw_content=asst_raw,
    )
    result = store.get_turn_messages("conv-1", [0])
    assert 0 in result
    user_content, asst_content, user_rc, asst_rc = result[0]
    assert user_content == "hello"
    assert asst_content == "Hi!"
    assert json.loads(user_rc) == [{"type": "text", "text": "hello"}]
    assert json.loads(asst_rc)[1]["name"] == "bash"


def test_extract_user_raw_content_anthropic():
    from virtual_context.proxy.formats import AnthropicFormat

    fmt = AnthropicFormat()
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Check this file"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents here"},
            ]},
        ],
    }
    raw = fmt.extract_user_raw_content(body)
    assert raw is not None
    assert len(raw) == 2
    assert raw[0]["type"] == "text"
    assert raw[1]["type"] == "tool_result"


def test_extract_user_raw_content_string_content():
    from virtual_context.proxy.formats import AnthropicFormat

    fmt = AnthropicFormat()
    body = {"messages": [{"role": "user", "content": "hello"}]}
    raw = fmt.extract_user_raw_content(body)
    assert raw == [{"type": "text", "text": "hello"}]


def test_extract_assistant_raw_content_anthropic():
    from virtual_context.proxy.formats import AnthropicFormat

    fmt = AnthropicFormat()
    response_body = {
        "content": [
            {"type": "text", "text": "Here's the result"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}},
        ]
    }
    raw = fmt.extract_assistant_raw_content(response_body)
    assert raw is not None
    assert len(raw) == 2
    assert raw[0]["type"] == "text"
    assert raw[1]["type"] == "tool_use"
    assert raw[1]["name"] == "bash"


def test_save_turn_message_without_raw_content(store):
    store.save_turn_message("conv-1", 0, "hello", "Hi!")
    result = store.get_turn_messages("conv-1", [0])
    user_content, asst_content, user_rc, asst_rc = result[0]
    assert user_content == "hello"
    assert asst_content == "Hi!"
    assert user_rc is None
    assert asst_rc is None


# ---------------------------------------------------------------------------
# Task 10: Compactor renders raw_content into full_text
# ---------------------------------------------------------------------------

from virtual_context.core.compactor import DomainCompactor


def test_format_conversation_with_raw_content():
    """When raw_content is present, _format_conversation renders it."""
    messages = [
        Message(role="user", content="check the file", raw_content=[
            {"type": "text", "text": "check the file"},
        ]),
        Message(role="assistant", content="Here it is", raw_content=[
            {"type": "text", "text": "Here it is"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "cat foo.py"}},
        ]),
        Message(role="user", content="", raw_content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "print('hello')"},
        ]),
    ]
    compactor = DomainCompactor.__new__(DomainCompactor)
    result = compactor._format_conversation(messages)
    assert "bash" in result
    assert "cat foo.py" in result
    assert "print('hello')" in result
    assert "tool_use_id" not in result  # should be rendered, not raw JSON


def test_format_conversation_tool_result_name_resolution():
    """tool_result blocks resolve tool name from preceding tool_use."""
    messages = [
        Message(role="assistant", content="", raw_content=[
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/tmp/x"}},
        ]),
        Message(role="user", content="", raw_content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"},
        ]),
    ]
    compactor = DomainCompactor.__new__(DomainCompactor)
    result = compactor._format_conversation(messages)
    assert "read_file" in result
    assert "file contents" in result


def test_format_conversation_falls_back_to_content():
    """When raw_content is None, use content as before."""
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]
    compactor = DomainCompactor.__new__(DomainCompactor)
    result = compactor._format_conversation(messages)
    assert "User: hello" in result
    assert "Assistant: world" in result
