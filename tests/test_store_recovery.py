"""Tests for store-backed pipeline recovery."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_get_chain_recovery_manifest_abstract():
    s = _make_dummy_store()
    assert s.get_chain_recovery_manifest("conv1") == []
    assert s.get_chain_recovery_manifest("conv1", min_turn=100) == []


def test_get_tool_names_for_refs_abstract():
    s = _make_dummy_store()
    assert s.get_tool_names_for_refs(["ref1", "ref2"]) == []


def test_chain_snapshot_sanitizer_contains_every_vc_tool_as_source_only():
    """Every VC tool is request-scoped scaffolding, not replay evidence."""
    from virtual_context.core.tool_loop import VC_TOOL_NAMES
    from virtual_context.proxy.message_filter import sanitize_chain_snapshot_messages

    poison = "BigTex started tesamorelin"
    for index, tool_name in enumerate(sorted(VC_TOOL_NAMES)):
        call_id = f"vc-{index}"
        chain = [
            {"role": "user", "content": "What did BigTex say?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": call_id, "name": tool_name, "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": poison},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": poison}]},
        ]

        sanitized = sanitize_chain_snapshot_messages(chain)

        assert sanitized == [{"role": "user", "content": "What did BigTex say?"}]


def test_collapse_turn_chains_recovers_from_store():
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
    mock_store.get_chain_recovery_manifest.return_value = [
        {"ref": "chain_50_abc123", "turn_number": 50, "tool_output_refs": "tool_ref1,tool_ref2", "message_count": 4, "tool_names": "Read, web_search"},
        {"ref": "chain_80_def456", "turn_number": 80, "tool_output_refs": "tool_ref3", "message_count": 3, "tool_names": "Read"},
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

    mock_store.get_chain_recovery_manifest.assert_called_once_with("test-conv", min_turn=0)
    assert mock_store.get_tool_names_for_refs.call_count == 0
    assert recovered == 2
    assert "chain_50_abc123" in refs
    assert "chain_80_def456" in refs

    import json
    result_json = json.dumps(result)
    assert "Compacted turn 50" in result_json
    assert "Compacted turn 80" in result_json
    assert "vc_restore_tool" in result_json


def test_collapse_turn_chains_reuses_existing_snapshot_for_canonical_turn():
    import json
    from unittest.mock import MagicMock

    from virtual_context.core.turn_tag_index import TurnTagIndex
    from virtual_context.proxy.formats import detect_format
    from virtual_context.proxy.message_filter import collapse_turn_chains
    from virtual_context.types import TurnTagEntry

    body = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/tmp/x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "file contents shown"}]},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "current"},
        ],
    }
    fmt = detect_format(body)

    combined = "read file file contents shown"
    msg_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
    tti = TurnTagIndex()
    tti.append(TurnTagEntry(turn_number=10, message_hash=msg_hash, tags=["file-ops"]))

    mock_store = MagicMock()
    mock_store.get_chain_snapshots_for_conversation.return_value = [
        {"ref": "chain_existing_ref", "turn_number": 10, "tool_output_refs": "tool_ref", "message_count": 4},
    ]
    mock_store.get_chain_recovery_manifest.return_value = []

    result, count, refs, recovered = collapse_turn_chains(
        body=body,
        fmt=fmt,
        protected_recent_turns=3,
        turn_tag_index=tti,
        store=mock_store,
        conversation_id="test-conv",
        pre_filter_body={"messages": []},
    )

    assert count == 1
    assert recovered == 0
    assert refs == ["chain_existing_ref"]
    assert "chain_existing_ref" in json.dumps(result)
    mock_store.get_chain_snapshots_for_conversation.assert_called_once_with("test-conv", min_turn=0)
    mock_store.store_tool_output.assert_not_called()
    mock_store.store_chain_snapshot.assert_not_called()


def test_collapse_turn_chains_stores_vc_chain_as_exact_user_source_only():
    """New snapshots cannot persist VC result prose or its assistant echo."""
    import copy
    import json

    from virtual_context.core.turn_tag_index import TurnTagIndex
    from virtual_context.proxy.formats import detect_format
    from virtual_context.proxy.message_filter import collapse_turn_chains

    poison = "BigTex started tesamorelin"
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "What did BigTex say?"},
            {"role": "assistant", "content": [
                {
                    "type": "tool_use",
                    "id": "vc-1",
                    "name": "vc_search_summaries",
                    "input": {"query": "BigTex"},
                },
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "vc-1", "content": poison},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": poison}]},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "current"},
        ],
    }
    pre_filter_body = copy.deepcopy(body)
    fmt = detect_format(body)
    mock_store = MagicMock()
    mock_store.get_chain_snapshots_for_conversation.return_value = []

    _, collapsed, _, _ = collapse_turn_chains(
        body=body,
        fmt=fmt,
        protected_recent_turns=1,
        turn_tag_index=TurnTagIndex(),
        store=mock_store,
        conversation_id="test-conv",
        pre_filter_body=pre_filter_body,
        deep_compaction_ratio=0,
    )

    assert collapsed == 1
    snapshot_call = mock_store.store_chain_snapshot.call_args.kwargs
    assert snapshot_call["message_count"] == 1
    assert json.loads(snapshot_call["chain_json"]) == [
        {"role": "user", "content": "What did BigTex say?"},
    ]
    mock_store.store_tool_output.assert_not_called()


def test_restore_legacy_vc_chain_cannot_reinsert_result_or_assistant_poison():
    """The read boundary contains poisoned snapshots written by old builds."""
    import json

    from virtual_context.proxy.handlers import _ProxyToolRuntime

    ref = "chain_7_legacy"
    poison = "BigTex started tesamorelin"
    chain = [
        {"role": "user", "content": "What did BigTex say?"},
        {"role": "assistant", "content": [
            {
                "type": "tool_use",
                "id": "vc-legacy",
                "name": "vc_remember_when",
                "input": {"query": "BigTex", "time_range": "last month"},
            },
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "vc-legacy", "content": poison},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": poison}]},
    ]
    target = {"messages": [
        {"role": "user", "content": "[Compacted turn 7]"},
        {"role": "assistant", "content": f'vc_restore_tool(ref="{ref}")'},
    ]}
    store = MagicMock()
    store.get_chain_snapshot.return_value = {
        "chain_json": json.dumps(chain),
        "tool_output_refs": "",
    }
    runtime = _ProxyToolRuntime(
        engine=SimpleNamespace(_store=store),
        api_format="anthropic",
        conversation_id="test-conv",
        get_target_body=lambda: target,
    )

    result = runtime.restore_tool_output(ref)
    restored_json = json.dumps(target)

    assert isinstance(result, str) and result.startswith("Restored.")
    assert "What did BigTex say?" in restored_json
    assert poison not in restored_json
    assert "vc_remember_when" not in restored_json
    assert "tool_use" not in restored_json
    assert "tool_result" not in restored_json


def test_restore_non_vc_chain_preserves_existing_assistant_behavior():
    """Containment does not change ordinary non-VC chain restoration."""
    import json

    from virtual_context.proxy.handlers import _ProxyToolRuntime

    ref = "chain_8_nonvc"
    chain = [
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": "The file contains safe output"},
    ]
    target = {"messages": [
        {"role": "user", "content": "[Compacted turn 8]"},
        {"role": "assistant", "content": f'vc_restore_tool(ref="{ref}")'},
    ]}
    store = MagicMock()
    store.get_chain_snapshot.return_value = {
        "chain_json": json.dumps(chain),
        "tool_output_refs": "",
    }
    runtime = _ProxyToolRuntime(
        engine=SimpleNamespace(_store=store),
        api_format="anthropic",
        conversation_id="test-conv",
        get_target_body=lambda: target,
    )

    result = runtime.restore_tool_output(ref)

    assert isinstance(result, str) and result.startswith("Restored.")
    assert "The file contains safe output" in json.dumps(target)


def test_collapse_turn_chains_reuses_runtime_snapshot_cache():
    from unittest.mock import MagicMock

    from virtual_context.core.turn_tag_index import TurnTagIndex
    from virtual_context.proxy.formats import detect_format
    from virtual_context.proxy.message_filter import collapse_turn_chains
    from virtual_context.types import TurnTagEntry

    body = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/tmp/x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "file contents shown"}]},
            {"role": "user", "content": "current"},
        ],
    }
    fmt = detect_format(body)

    combined = "read file file contents shown"
    msg_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
    tti = TurnTagIndex()
    tti.append(TurnTagEntry(turn_number=10, message_hash=msg_hash, tags=["file-ops"]))

    mock_store = MagicMock()
    mock_store.get_chain_snapshots_for_conversation.return_value = [
        {"ref": "chain_existing_ref", "turn_number": 10, "tool_output_refs": "tool_ref", "message_count": 4},
    ]
    runtime_cache = {"loaded": False, "refs_by_turn": {}}
    runtime_cache["recovery_loaded"] = False
    runtime_cache["recovery_manifest"] = []

    collapse_turn_chains(
        body=body,
        fmt=fmt,
        protected_recent_turns=1,
        turn_tag_index=tti,
        store=mock_store,
        conversation_id="test-conv",
        pre_filter_body={"messages": []},
        collapse_runtime_cache=runtime_cache,
    )
    collapse_turn_chains(
        body=body,
        fmt=fmt,
        protected_recent_turns=1,
        turn_tag_index=tti,
        store=mock_store,
        conversation_id="test-conv",
        pre_filter_body={"messages": []},
        collapse_runtime_cache=runtime_cache,
    )

    assert runtime_cache["loaded"] is True
    assert runtime_cache["refs_by_turn"] == {10: "chain_existing_ref"}
    assert runtime_cache["recovery_loaded"] is False
    mock_store.get_chain_snapshots_for_conversation.assert_called_once_with("test-conv", min_turn=0)


def test_collapse_turn_chains_reuses_runtime_recovery_manifest():
    from unittest.mock import MagicMock

    from virtual_context.core.turn_tag_index import TurnTagIndex, TurnTagEntry
    from virtual_context.proxy.formats import detect_format
    from virtual_context.proxy.message_filter import collapse_turn_chains

    body = {
        "model": "claude-opus-4-6",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "current question"},
        ],
    }
    fmt = detect_format(body)
    tti = TurnTagIndex()
    for i in range(100):
        tti.entries.append(TurnTagEntry(turn_number=i, message_hash=f"hash_{i}", tags=[f"tag_{i}"]))

    mock_store = MagicMock()
    mock_store.get_chain_recovery_manifest.return_value = [
        {"ref": "chain_50_abc123", "turn_number": 50, "tool_output_refs": "tool_ref1,tool_ref2", "message_count": 4, "tool_names": "Read, web_search"},
    ]
    runtime_cache = {
        "loaded": False,
        "refs_by_turn": {},
        "recovery_loaded": False,
        "recovery_manifest": [],
    }

    collapse_turn_chains(
        body=body,
        fmt=fmt,
        protected_recent_turns=6,
        turn_tag_index=tti,
        store=mock_store,
        conversation_id="test-conv",
        client_truncated=True,
        collapse_runtime_cache=runtime_cache,
    )
    collapse_turn_chains(
        body=body,
        fmt=fmt,
        protected_recent_turns=6,
        turn_tag_index=tti,
        store=mock_store,
        conversation_id="test-conv",
        client_truncated=True,
        collapse_runtime_cache=runtime_cache,
    )

    assert runtime_cache["recovery_loaded"] is True
    assert runtime_cache["recovery_manifest"] == [
        {"ref": "chain_50_abc123", "turn_number": 50, "tool_output_refs": "tool_ref1,tool_ref2", "message_count": 4, "tool_names": "Read, web_search"},
    ]
    mock_store.get_chain_recovery_manifest.assert_called_once_with("test-conv", min_turn=0)


_FILL_OWNER = "test-conv"
_FILL_GUILD = "guild:one"
_FILL_CHANNEL = "channel:one"
_FILL_REQUESTER = "actor:discord:requester"


def _fill_row(
    canonical_id: str,
    turn_number: int,
    text: str,
    *,
    speaker: str,
    actor: str,
    audience: str = _FILL_GUILD,
    channel: str = _FILL_CHANNEL,
    assistant: str = "",
    reply_body: str = "",
):
    from virtual_context.types import (
        AUDIENCE_ATTRIBUTION_VERSION,
        CanonicalTurnRow,
    )

    return CanonicalTurnRow(
        conversation_id=_FILL_OWNER,
        canonical_turn_id=canonical_id,
        turn_number=turn_number,
        sort_key=float(turn_number),
        user_content=text,
        assistant_content=assistant,
        sender=speaker,
        sender_actor_id=actor,
        audience_conversation_id=audience,
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        origin_channel_id=channel,
        session_date="2026-08-22",
        reply_target_body=reply_body,
    )


def _fill_context(
    *,
    audience: str = _FILL_GUILD,
    channel: str = _FILL_CHANNEL,
    owner: str = _FILL_OWNER,
    roster=(),
):
    from virtual_context.types import SpeakerRetrievalContext

    return SpeakerRetrievalContext(
        owner_conversation_id=owner,
        audience_conversation_id=audience,
        audience_channel_id=channel,
        request_origin_channel_id=channel,
        requester_actor_id=_FILL_REQUESTER,
        roster_label_actor_pairs=tuple(roster),
    )


def _fill_store(rows):
    store = MagicMock()
    materialized = list(rows)
    by_key = {
        (row.conversation_id, row.canonical_turn_id): row
        for row in materialized
    }
    store.get_all_tag_summaries.return_value = []
    store.get_all_canonical_turns.return_value = materialized
    store.get_canonical_turn_rows_by_id.side_effect = (
        lambda keys, *, speaker_context: {
            key: by_key[key] for key in keys if key in by_key
        }
    )
    store.get_recent_canonical_turns.return_value = materialized
    store.get_tool_outputs_for_turn.return_value = []
    return store


def _run_store_fill(rows, context, *, store=None):
    import json

    from virtual_context.core.turn_tag_index import TurnTagIndex
    from virtual_context.proxy.formats import detect_format
    from virtual_context.proxy.message_filter import fill_pass
    from virtual_context.types import AssembledContext, RetrievalResult

    body = {
        "model": "claude-opus-4-6",
        "messages": [{"role": "user", "content": "current question"}],
    }
    resolved_store = store or _fill_store(rows)
    assembled = (
        AssembledContext(
            presented_segment_refs=set(),
            presented_tags=set(),
            tag_sections={},
            retrieval_result=RetrievalResult(),
            speaker_context=context,
        )
        if context is not None
        else None
    )
    result, _, turns = fill_pass(
        body=body,
        fmt=detect_format(body),
        outbound_tokens=100,
        target_tokens=10_000,
        assembled=assembled,
        pre_filter_body={"messages": []},
        store=resolved_store,
        conversation_id=_FILL_OWNER,
        summary_ratio=0,
        client_truncated=True,
        turn_tag_index=TurnTagIndex(),
    )
    return result, json.dumps(result), turns, resolved_store


def test_fill_pass_store_recovery_injects_proved_source_as_context_not_user():
    exact = "I only planned to research tesamorelin"
    generated = "BigTex started tesamorelin"
    copied_reply = "A third party said they started it"
    row = _fill_row(
        "turn-12",
        12,
        exact,
        speaker="BigTex",
        actor="actor:discord:bigtex",
        assistant=generated,
        reply_body=copied_reply,
    )

    result, rendered, turns, store = _run_store_fill(
        [row],
        _fill_context(roster=(("BigTex", "actor:discord:bigtex"),)),
    )

    assert turns == 1
    assert exact in rendered
    assert generated not in rendered
    assert copied_reply not in rendered
    assert "<historical-source-transcript>" in rendered
    context_text = result["messages"][0]["content"][-1]["text"]
    assert '"display_name":"BigTex"' in context_text
    assert '"role":"historical_human"' in context_text
    assert "actor:discord:bigtex" not in rendered
    assert [message["role"] for message in result["messages"]] == ["user"]
    store.get_recent_canonical_turns.assert_any_call(_FILL_OWNER, limit=200)
    store.get_all_canonical_turns.assert_not_called()


def test_fill_pass_store_recovery_keeps_two_historical_humans_separate():
    rows = [
        _fill_row(
            "turn-20", 20, "Alice chose tea", speaker="Alice",
            actor="actor:discord:alice",
        ),
        _fill_row(
            "turn-21", 21, "Bob chose coffee", speaker="Bob",
            actor="actor:discord:bob",
        ),
    ]
    context = _fill_context(roster=(
        ("Alice", "actor:discord:alice"),
        ("Bob", "actor:discord:bob"),
    ))

    result, rendered, turns, _ = _run_store_fill(rows, context)

    assert turns == 2
    assert "Alice chose tea" in rendered
    assert "Bob chose coffee" in rendered
    context_text = result["messages"][0]["content"][-1]["text"]
    assert '"display_name":"Alice"' in context_text
    assert '"display_name":"Bob"' in context_text
    assert context_text.count('"role":"historical_human"') == 2
    assert [message["role"] for message in result["messages"]] == ["user"]


def test_fill_pass_store_recovery_excludes_dm_row_under_guild_owner_alias():
    guild_text = "Guild-only source statement"
    dm_text = "Private DM source statement"
    rows = [
        _fill_row(
            "guild-turn", 30, guild_text, speaker="GuildName",
            actor="actor:discord:member",
        ),
        _fill_row(
            "dm-turn", 31, dm_text, speaker="PrivateAlias",
            actor="actor:discord:member", audience="dm:private", channel="",
        ),
    ]

    _, rendered, turns, _ = _run_store_fill(
        rows,
        _fill_context(roster=(("GuildName", "actor:discord:member"),)),
    )

    assert turns == 1
    assert guild_text in rendered
    assert "GuildName" in rendered
    assert dm_text not in rendered
    assert "PrivateAlias" not in rendered


def test_fill_pass_store_recovery_adds_nothing_without_request_authority():
    from virtual_context.types import SpeakerRetrievalContext

    secret = "historical source must stay absent"
    row = _fill_row(
        "turn-40", 40, secret, speaker="Alice",
        actor="actor:discord:alice",
    )
    for context in (None, SpeakerRetrievalContext.ineligible()):
        _, rendered, turns, store = _run_store_fill([row], context)

        assert turns == 0
        assert secret not in rendered
        store.get_recent_canonical_turns.assert_not_called()
        store.get_all_canonical_turns.assert_not_called()


def test_fill_pass_store_recovery_rejects_label_collision():
    rows = [
        _fill_row(
            "turn-50", 50, "first Alex source", speaker="Alex",
            actor="actor:discord:alex-one",
        ),
        _fill_row(
            "turn-51", 51, "second Alex source", speaker="Alex",
            actor="actor:discord:alex-two",
        ),
    ]

    _, rendered, turns, _ = _run_store_fill(rows, _fill_context())

    assert turns == 0
    assert "first Alex source" not in rendered
    assert "second Alex source" not in rendered


def test_fill_pass_store_recovery_adds_nothing_when_hydration_unavailable():
    row = _fill_row(
        "turn-60", 60, "unhydrated source", speaker="Alice",
        actor="actor:discord:alice",
    )
    store = _fill_store([row])
    store.get_canonical_turn_rows_by_id.side_effect = RuntimeError("unavailable")

    _, rendered, turns, _ = _run_store_fill(
        [row], _fill_context(), store=store,
    )

    assert turns == 0
    assert "unhydrated source" not in rendered


def test_fill_pass_skips_tool_turns_from_store():
    rows = [
        _fill_row(
            "turn-70", 70, "run the tests", speaker="Alice",
            actor="actor:discord:alice",
        ),
        _fill_row(
            "turn-71", 71, "what about baking?", speaker="Bob",
            actor="actor:discord:bob",
        ),
    ]
    store = _fill_store(rows)
    store.get_tool_outputs_for_turn.side_effect = (
        lambda _conversation_id, turn_number: ["ref1"]
        if turn_number == 70
        else []
    )

    _, rendered, turns, _ = _run_store_fill(
        rows,
        _fill_context(roster=(
            ("Alice", "actor:discord:alice"),
            ("Bob", "actor:discord:bob"),
        )),
        store=store,
    )

    assert turns == 1
    assert "what about baking?" in rendered
    assert "run the tests" not in rendered


def test_fill_pass_prefilter_replay_omits_assistant_poison():
    import json

    from virtual_context.proxy.formats import detect_format
    from virtual_context.proxy.message_filter import fill_pass

    poison = "BigTex started tesamorelin"
    body = {
        "model": "claude-opus-4-6",
        "messages": [{"role": "user", "content": "current question"}],
    }
    pre_filter_body = {
        "model": "claude-opus-4-6",
        "messages": [
            {"role": "user", "content": "I only planned to research tesamorelin"},
            {"role": "assistant", "content": poison},
            {"role": "user", "content": "current question"},
        ],
    }

    result, _, turns = fill_pass(
        body=body,
        fmt=detect_format(body),
        outbound_tokens=100,
        target_tokens=2000,
        assembled=None,
        pre_filter_body=pre_filter_body,
        store=None,
        conversation_id="test-conv",
        summary_ratio=0,
    )
    rendered = json.dumps(result)

    assert turns == 1
    assert "I only planned to research tesamorelin" in rendered
    assert poison not in rendered
