"""Instruction carriers survive history transforms without preserving old chat."""

import copy
import json
from unittest.mock import Mock

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.formats import OpenAIResponsesFormat
from virtual_context.proxy.message_filter import (
    PayloadBudgetExceeded,
    admit_provider_payload,
    collapse_turn_chains,
    drop_compacted_turns,
    enforce_payload_budget,
    filter_body_messages,
    scan_reducible_items,
    trim_to_upstream_limit,
)
from virtual_context.types import TurnTagEntry


def project_instructions(*, string_content=False):
    text = (
        "# AGENTS.md instructions for /workspace\n\n<INSTRUCTIONS>\n"
        "Use BrowserOS when direct web_fetch is blocked.\n"
        + "Keep the original instruction content intact.\n" * 20
        + "</INSTRUCTIONS>"
    )
    return {
        "type": "message", "role": "user", "id": "project-carrier",
        "content": text if string_content else [
            {"type": "input_text", "text": text},
            {"type": "input_text", "text": "<environment_context>\n/workspace\n</environment_context>"},
        ],
    }


def message(role, text):
    return {"type": "message", "role": role, "content": text}


def format_for_test():
    fmt = OpenAIResponsesFormat()
    fmt.set_token_counter(lambda text: max(1, len(text) // 4))
    return fmt


def test_trim_preserves_prefix_and_interleaved_instructions_without_old_chat():
    fmt = format_for_test()
    host = project_instructions()
    developer = message("developer", "Runtime policy. " * 40)
    system = message("system", "Original system policy.")
    body = {"model": "test", "input": [system, developer, host]}
    for i in range(5):
        body["input"].extend([
            message("user", f"old question {i} " * 100),
            message("assistant", f"old answer {i} " * 100),
        ])
    late = message("developer", "Updated developer instruction. " * 20)
    body["input"].insert(5, late)
    original = copy.deepcopy(body)

    result, removed = trim_to_upstream_limit(body, 1, fmt)

    assert removed == 3
    assert [item for item in result["input"] if item in [system, developer, host, late]] == [system, developer, host, late]
    assert "old question 0" not in json.dumps(result)
    assert "old question 3" in json.dumps(result)
    assert "old question 4" in json.dumps(result)
    assert body == original


def test_instruction_tokens_consume_budget_and_overflow_is_rejected():
    fmt = format_for_test()
    host = project_instructions(string_content=True)
    body = {"input": [host]}
    for i in range(4):
        body["input"].extend([message("user", f"question {i}"), message("assistant", f"answer {i}")])
    before = copy.deepcopy(body)

    with pytest.raises(PayloadBudgetExceeded):
        admit_provider_payload(body, 50, fmt)

    assert body == before
    assert body["input"][0] == host


@pytest.mark.parametrize("string_content", [False, True])
def test_compacted_history_drop_keeps_native_project_carrier(string_content):
    fmt = format_for_test()
    host = project_instructions(string_content=string_content)
    body = {"input": [
        message("developer", "Runtime policy."), host,
        message("user", "old question"), message("assistant", "old answer"),
        message("user", "current question"),
    ]}

    result, removed = drop_compacted_turns(body, TurnTagIndex(), 2, fmt=fmt, protected_recent_turns=0)

    assert removed == 1
    assert result["input"] == [message("developer", "Runtime policy."), host, message("user", "current question")]


def test_tag_filter_does_not_drop_or_merge_mid_history_project_carrier():
    fmt = format_for_test()
    host = project_instructions()
    index = TurnTagIndex()
    for i in range(2):
        index.append(TurnTagEntry(turn_number=i, message_hash=f"h{i}", tags=["old"], primary_tag="old"))
    body = {"input": [
        message("user", "old 0"), message("assistant", "answer 0"), host,
        message("user", "old 1"), message("assistant", "answer 1"),
        message("user", "current question"),
    ]}

    result, removed = filter_body_messages(body, index, [], recent_turns=0, compacted_turn=2, fmt=fmt)

    assert removed == 2
    assert result["input"] == [host, message("user", "current question")]


@pytest.mark.parametrize("host_after_current", [False, True])
def test_tag_filter_keeps_logical_turn_tags_with_interleaved_host(host_after_current):
    fmt = format_for_test()
    host = project_instructions()
    trailing_host = message("user", "<environment_context>\n/workspace\n</environment_context>")
    current = message("user", "current question")
    index = TurnTagIndex()
    body = {"input": []}
    for i, tag in enumerate(["old", "wanted", "old", "old"]):
        index.append(TurnTagEntry(turn_number=i, message_hash=f"h{i}", tags=[tag], primary_tag=tag))
        body["input"].append(message("user", f"question {i}"))
        if i == 0:
            body["input"].append(host)
        body["input"].append(message("assistant", f"answer {i}"))
    body["input"].append(current)
    if host_after_current:
        body["input"].append(trailing_host)
    before = copy.deepcopy(body)

    result, removed = filter_body_messages(body, index, ["wanted"], recent_turns=0, fmt=fmt)

    expected = [host, message("user", "question 1"), message("assistant", "answer 1"), current]
    if host_after_current:
        expected.append(trailing_host)
    assert result["input"] == expected
    assert removed == 3
    assert body == before


def test_chain_collapse_preserves_interleaved_instructions_and_excludes_snapshot():
    fmt = format_for_test()
    host = project_instructions()
    developer = message("developer", "An instruction inside the tool turn.")
    body = {"input": [
        message("user", "old query"), host,
        message("assistant", "looking now"),
        {"type": "function_call", "call_id": "call1", "name": "read", "arguments": "{}"},
        developer,
        {"type": "function_call_output", "call_id": "call1", "output": "old tool output"},
        message("assistant", "old final answer"),
        message("user", "current query"),
    ]}
    store = Mock()
    store.get_chain_snapshots_for_conversation.return_value = []

    result, collapsed, refs, _ = collapse_turn_chains(
        body, fmt, protected_recent_turns=0, turn_tag_index=TurnTagIndex(),
        store=store, conversation_id="instructions-test", deep_compaction_ratio=0,
    )

    assert collapsed == 1 and len(refs) == 1
    assert [item for item in result["input"] if item in [host, developer]] == [host, developer]
    assert not any(item.get("call_id") == "call1" for item in result["input"])
    snapshot = json.loads(store.store_chain_snapshot.call_args.kwargs["chain_json"])
    assert snapshot[0] == message("user", "old query")
    assert host not in snapshot and developer not in snapshot
    assert any(item.get("call_id") == "call1" for item in snapshot)


@pytest.mark.parametrize("string_content", [False, True])
def test_byte_budget_does_not_truncate_instruction_content(string_content):
    fmt = format_for_test()
    protected = [
        message("system", "system policy " * 100),
        message("developer", "developer policy " * 100),
        project_instructions(string_content=string_content),
    ]
    body = {"input": [*copy.deepcopy(protected), message("assistant", "reducible answer " * 600)]}

    candidates = scan_reducible_items(body, fmt)
    assert candidates and all(item.msg_index == 3 for item in candidates)
    result, reductions, _ = enforce_payload_budget(body, fmt, context_window=100)

    assert reductions > 0
    assert result["input"][:3] == protected
