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
    from virtual_context.types import EngineState
    engine._engine_state = EngineState()
    engine._is_tool_turn = VirtualContextEngine._is_tool_turn
    engine._get_latest_turn_pair = VirtualContextEngine._get_latest_turn_pair.__get__(engine)
    engine._tag_splitter = None
    engine._monitor = MagicMock()
    engine._monitor.check.return_value = None
    engine._store = MagicMock()
    engine._store.get_all_tags.return_value = []
    engine.config = MagicMock()
    engine.config.conversation_id = "test"
    engine._working_set = {}
    engine._telemetry = MagicMock()
    engine._request_captures_provider = None
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
    assert engine._engine_state.tool_tag_counter == 1


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


# --- Task 6: Assembler expansion hints for tool segments ---

from virtual_context.core.assembler import ContextAssembler as Assembler
from virtual_context.types import StoredSummary, SegmentMetadata
from datetime import datetime, timezone


def _make_summary(tag: str, tags: list[str], summary: str) -> StoredSummary:
    return StoredSummary(
        ref="ref-1",
        primary_tag=tag,
        tags=tags,
        summary=summary,
        summary_tokens=50,
        start_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        metadata=SegmentMetadata(turn_count=1),
    )


def test_expansion_hint_for_tool_tag():
    """Segments with tool_N tags get expansion hints at SUMMARY depth."""
    assembler = Assembler.__new__(Assembler)
    summaries = [_make_summary("tool_3", ["tool_3"], "User ran bash to list files.")]
    result = assembler._format_tag_section("tool_3", summaries)
    assert 'vc_expand_topic("tool_3")' in result
    assert "tool output truncated" in result


def test_no_expansion_hint_for_regular_tag():
    """Regular tags don't get expansion hints."""
    assembler = Assembler.__new__(Assembler)
    summaries = [_make_summary("coding", ["coding"], "Discussed Python patterns.")]
    result = assembler._format_tag_section("coding", summaries)
    assert "vc_expand_topic" not in result
    assert "tool output truncated" not in result


# --- Task 7: End-to-end verification ---

def test_tool_tag_counter_increments():
    """Multiple tool turns get sequential tool_1, tool_2, etc."""
    engine = _build_tag_turn_engine_mock()

    # First tool turn
    history1 = [
        Message(role="user", content="", raw_content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "out1"},
        ]),
        Message(role="assistant", content="", raw_content=[
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        ]),
    ]
    VirtualContextEngine.tag_turn(engine, history1)

    # Second tool turn
    history2 = history1 + [
        Message(role="user", content="", raw_content=[
            {"type": "tool_result", "tool_use_id": "t2", "content": "out2"},
        ]),
        Message(role="assistant", content="", raw_content=[
            {"type": "tool_use", "id": "t2", "name": "read", "input": {}},
        ]),
    ]
    VirtualContextEngine.tag_turn(engine, history2)

    assert engine._engine_state.tool_tag_counter == 2
    assert engine._turn_tag_index.entries[0].tags == ["tool_1"]
    assert engine._turn_tag_index.entries[1].tags == ["tool_2"]
