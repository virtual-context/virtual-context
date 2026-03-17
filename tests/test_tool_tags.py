"""Tests for tool tag assignment and expansion hints."""

from virtual_context.types import EngineStateSnapshot, TurnTagEntry


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
