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


def test_save_turn_message_without_raw_content(store):
    store.save_turn_message("conv-1", 0, "hello", "Hi!")
    result = store.get_turn_messages("conv-1", [0])
    user_content, asst_content, user_rc, asst_rc = result[0]
    assert user_content == "hello"
    assert asst_content == "Hi!"
    assert user_rc is None
    assert asst_rc is None
