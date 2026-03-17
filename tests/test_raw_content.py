"""Tests for lossless raw_content on Message."""

from virtual_context.types import Message


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
