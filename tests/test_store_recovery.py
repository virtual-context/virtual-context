"""Tests for store-backed pipeline recovery."""

from virtual_context.types import MonitorConfig


def test_monitor_config_has_store_recovery_threshold():
    mc = MonitorConfig()
    assert hasattr(mc, "store_recovery_threshold")
    assert mc.store_recovery_threshold == 0.70


def _make_dummy_store():
    """Create a minimal concrete ContextStore for testing default method implementations."""
    from virtual_context.core.store import ContextStore

    class DummyStore(ContextStore):
        def store_segment(self, segment): return ""
        def get_segment(self, ref, *, conversation_id=None): return None
        def get_summary(self, ref, *, conversation_id=None): return None
        def get_summaries_by_tags(self, tags, min_overlap=1, limit=10, before=None, after=None, conversation_id=None): return []
        def search(self, query, tags=None, limit=5, conversation_id=None): return []
        def get_all_tags(self, conversation_id=None): return []
        def get_conversation_stats(self): return []
        def get_tag_aliases(self, conversation_id=None): return {}
        def set_tag_alias(self, alias, canonical, conversation_id=""): pass
        def delete_segment(self, ref): return False
        def cleanup(self, max_age=None, max_total_tokens=None): return 0
        def save_tag_summary(self, tag_summary, conversation_id=""): pass
        def get_tag_summary(self, tag, conversation_id=""): return None
        def get_all_tag_summaries(self, *, conversation_id=None): return []
        def search_full_text(self, query, limit=5, conversation_id=None): return []
        def get_segments_by_tags(self, tags, min_overlap=1, limit=20, conversation_id=None): return []

    return DummyStore()


def test_get_chain_snapshots_for_conversation_abstract():
    s = _make_dummy_store()
    assert s.get_chain_snapshots_for_conversation("conv1") == []
    assert s.get_chain_snapshots_for_conversation("conv1", min_turn=100) == []


def test_get_tool_names_for_refs_abstract():
    s = _make_dummy_store()
    assert s.get_tool_names_for_refs(["ref1", "ref2"]) == []


def test_collapse_turn_chains_recovers_from_store():
    from unittest.mock import MagicMock
    from virtual_context.proxy.message_filter import collapse_turn_chains
    from virtual_context.proxy.formats import detect_format
    from virtual_context.core.turn_tag_index import TurnTagIndex, TurnTagEntry

    body = {
        "system": "test",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "current question"},
        ],
        "model": "claude-opus-4-6",
    }
    fmt = detect_format(body)
    tti = TurnTagIndex()
    for i in range(100):
        tti.entries.append(TurnTagEntry(turn_number=i, message_hash=f"hash_{i}", tags=[f"tag_{i}"]))

    mock_store = MagicMock()
    mock_store.get_chain_snapshots_for_conversation.return_value = [
        {"ref": "chain_50_abc123", "turn_number": 50, "tool_output_refs": "tool_ref1,tool_ref2", "message_count": 4},
        {"ref": "chain_80_def456", "turn_number": 80, "tool_output_refs": "tool_ref3", "message_count": 3},
    ]
    mock_store.get_tool_names_for_refs.return_value = ["web_search", "Read"]

    result, count, refs, recovered = collapse_turn_chains(
        body, fmt,
        protected_recent_turns=6,
        turn_tag_index=tti,
        store=mock_store,
        conversation_id="test-conv",
        client_truncated=True,
    )

    mock_store.get_chain_snapshots_for_conversation.assert_called_once()
    assert recovered == 2
    assert "chain_50_abc123" in refs
    assert "chain_80_def456" in refs

    import json
    result_json = json.dumps(result)
    assert "Compacted turn 50" in result_json
    assert "Compacted turn 80" in result_json
    assert "vc_restore_tool" in result_json


def test_fill_pass_restores_from_store_on_truncation():
    from unittest.mock import MagicMock
    from virtual_context.proxy.message_filter import fill_pass
    from virtual_context.proxy.formats import detect_format
    from virtual_context.types import AssembledContext, RetrievalResult
    from virtual_context.core.turn_tag_index import TurnTagIndex, TurnTagEntry
    import copy

    body = {
        "system": "test",
        "messages": [
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": [{"type": "text", "text": "recent reply"}]},
            {"role": "user", "content": "current question"},
        ],
        "model": "claude-opus-4-6",
    }
    fmt = detect_format(body)

    mock_store = MagicMock()
    mock_store.get_all_tag_summaries.return_value = []
    mock_store.load_recent_turn_messages.return_value = [
        (10, "older question about cooking", "I explained Italian techniques"),
        (11, "what about baking?", "Bread baking involves..."),
    ]
    mock_store.get_tool_outputs_for_turn.return_value = []

    assembled = AssembledContext(
        presented_segment_refs=set(),
        presented_tags=set(),
        tag_sections={},
        retrieval_result=RetrievalResult(),
    )

    tti = TurnTagIndex()
    for i in range(50):
        tti.entries.append(TurnTagEntry(turn_number=i, message_hash=f"hash_{i}", tags=[f"tag_{i}"]))

    result_body, summaries, turns = fill_pass(
        body=body, fmt=fmt,
        outbound_tokens=70000, target_tokens=90000,
        assembled=assembled, pre_filter_body=copy.deepcopy(body),
        store=mock_store, conversation_id="test-conv",
        summary_ratio=0.0,
        client_truncated=True,
        turn_tag_index=tti,
    )

    mock_store.load_recent_turn_messages.assert_called_once()
    assert turns >= 1


def test_fill_pass_skips_tool_turns_from_store():
    from unittest.mock import MagicMock
    from virtual_context.proxy.message_filter import fill_pass
    from virtual_context.proxy.formats import detect_format
    from virtual_context.types import AssembledContext, RetrievalResult
    from virtual_context.core.turn_tag_index import TurnTagIndex, TurnTagEntry
    import copy, json

    body = {
        "system": "test",
        "messages": [{"role": "user", "content": "current question"}],
        "model": "claude-opus-4-6",
    }
    fmt = detect_format(body)

    mock_store = MagicMock()
    mock_store.get_all_tag_summaries.return_value = []
    mock_store.load_recent_turn_messages.return_value = [
        (10, "run the tests", "Here are the results"),
        (11, "what about baking?", "Bread baking involves..."),
    ]
    mock_store.get_tool_outputs_for_turn.side_effect = lambda cid, tn: ["ref1"] if tn == 10 else []

    assembled = AssembledContext(
        presented_segment_refs=set(), presented_tags=set(),
        tag_sections={}, retrieval_result=RetrievalResult(),
    )

    tti = TurnTagIndex()
    for i in range(50):
        tti.entries.append(TurnTagEntry(turn_number=i, message_hash=f"hash_{i}", tags=[f"tag_{i}"]))

    result_body, summaries, turns = fill_pass(
        body=body, fmt=fmt,
        outbound_tokens=70000, target_tokens=90000,
        assembled=assembled, pre_filter_body=copy.deepcopy(body),
        store=mock_store, conversation_id="test-conv",
        summary_ratio=0.0,
        client_truncated=True,
        turn_tag_index=tti,
    )

    result_json = json.dumps(result_body)
    assert "Bread baking" in result_json
    assert "run the tests" not in result_json
