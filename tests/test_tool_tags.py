"""Tests for tool tag assignment and expansion hints."""

from virtual_context.types import EngineStateSnapshot, Message, TurnTagEntry
from virtual_context.engine import VirtualContextEngine


def test_engine_state_snapshot_has_tool_tag_counter():
    snap = EngineStateSnapshot(
        conversation_id="test",
        compacted_through=0,
        turn_tag_entries=[],
        turn_count=0,
        tool_tag_counter=5,
    )
    assert snap.tool_tag_counter == 5


def test_engine_state_snapshot_tool_tag_counter_default():
    snap = EngineStateSnapshot(
        conversation_id="test",
        compacted_through=0,
        turn_tag_entries=[],
        turn_count=0,
    )
    assert snap.tool_tag_counter == 0


# --- Task 2: Tool-turn detection helper ---

def test_is_tool_turn_with_tool_use():
    msgs = [
        Message(role="user", content="", raw_content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "output"},
        ]),
        Message(role="assistant", content="", raw_content=[
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
        ]),
    ]
    assert VirtualContextEngine._is_tool_turn(msgs) is True


def test_is_tool_turn_without_tools():
    msgs = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]
    assert VirtualContextEngine._is_tool_turn(msgs) is False


def test_is_tool_turn_mixed_content():
    """Turn with both text content and tool blocks — not a tool-only turn."""
    msgs = [
        Message(role="user", content="check this file", raw_content=[
            {"type": "text", "text": "check this file"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "file data"},
        ]),
        Message(role="assistant", content="Here's what I found"),
    ]
    # content is non-empty, so this is NOT a tool-only turn
    assert VirtualContextEngine._is_tool_turn(msgs) is False


def test_is_tool_turn_empty_content_with_tools():
    """Turn with empty content but tool blocks — IS a tool-only turn."""
    msgs = [
        Message(role="user", content="  ", raw_content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "data"},
        ]),
        Message(role="assistant", content="", raw_content=[
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": "t2", "name": "read", "input": {}},
        ]),
    ]
    assert VirtualContextEngine._is_tool_turn(msgs) is True
