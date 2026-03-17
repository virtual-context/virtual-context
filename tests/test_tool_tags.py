"""Tests for tool tag assignment and expansion hints."""

from unittest.mock import MagicMock

from virtual_context.types import EngineStateSnapshot, Message, TagResult, TurnTagEntry
from virtual_context.engine import VirtualContextEngine
from virtual_context.core.turn_tag_index import TurnTagIndex


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


# --- Task 3: Wire tool tag assignment into tag_turn() ---

def _build_tag_turn_engine_mock():
    """Build a minimal engine mock suitable for calling tag_turn()."""
    engine = MagicMock(spec=VirtualContextEngine)
    engine._turn_tag_index = TurnTagIndex()
    engine._tool_tag_counter = 0
    engine._is_tool_turn = VirtualContextEngine._is_tool_turn
    engine._get_latest_turn_pair = VirtualContextEngine._get_latest_turn_pair.__get__(engine)
    engine._tag_splitter = None
    engine._monitor = MagicMock()
    engine._monitor.check.return_value = None
    engine._store = MagicMock()
    engine._store.get_all_tags.return_value = []
    engine.config = MagicMock()
    engine.config.conversation_id = "test"
    engine._last_tag_ms = 0
    engine._last_compact_ms = 0
    engine._compacted_through = 0
    engine._split_processed_tags = set()
    engine._working_set = {}
    engine._trailing_fingerprint = ""
    engine._telemetry = MagicMock()
    engine._request_captures_provider = None
    engine._provider = ""
    return engine


def test_tag_turn_assigns_tool_tag_for_tool_only_turn():
    """Tool-only turns get tool_N tags, skipping the LLM tagger."""
    engine = _build_tag_turn_engine_mock()

    history = [
        Message(role="user", content="", raw_content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "output"},
        ]),
        Message(role="assistant", content="", raw_content=[
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        ]),
    ]

    # Call the real tag_turn on the mock
    VirtualContextEngine.tag_turn(engine, history)

    # Verify tool tag was assigned
    assert len(engine._turn_tag_index.entries) == 1
    entry = engine._turn_tag_index.entries[0]
    assert entry.tags == ["tool_1"]
    assert entry.primary_tag == "tool_1"
    assert engine._tool_tag_counter == 1


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


# --- Task 5: Persist and restore tool_tag_counter ---

def test_tool_tag_counter_persists_in_snapshot():
    """tool_tag_counter round-trips through EngineStateSnapshot."""
    snap = EngineStateSnapshot(
        conversation_id="test",
        compacted_through=0,
        turn_tag_entries=[],
        turn_count=0,
        tool_tag_counter=7,
    )
    assert snap.tool_tag_counter == 7

    # Simulate save/restore
    snap_dict = {
        "conversation_id": snap.conversation_id,
        "compacted_through": snap.compacted_through,
        "turn_tag_entries": snap.turn_tag_entries,
        "turn_count": snap.turn_count,
        "tool_tag_counter": snap.tool_tag_counter,
    }
    restored = EngineStateSnapshot(**snap_dict)
    assert restored.tool_tag_counter == 7
