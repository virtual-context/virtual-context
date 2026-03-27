# tests/test_format_agnostic.py
import json
import copy
import pytest
from virtual_context.proxy.formats import detect_format
from virtual_context.proxy.message_filter import collapse_turn_chains, drop_topic_only_stubs, stub_tool_outputs_by_position
from virtual_context.core.turn_tag_index import TurnTagIndex


class TestMutationMethods:
    def _anthropic_body(self):
        return {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "A" * 1000},
                {"type": "text", "text": "hi there"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "big result here"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]}

    def test_remove_items(self):
        body = self._anthropic_body()
        fmt = detect_format(body)
        fmt.remove_items(body, [0, 1])  # remove first two messages
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "user"  # was index 2

    def test_insert_items(self):
        body = self._anthropic_body()
        fmt = detect_format(body)
        new_items = [
            {"role": "user", "content": "stub user"},
            {"role": "assistant", "content": "stub asst"},
        ]
        fmt.insert_items(body, 2, new_items)
        assert len(body["messages"]) == 6
        assert body["messages"][2]["content"] == "stub user"

    def test_remove_thinking_block_anthropic(self):
        body = self._anthropic_body()
        fmt = detect_format(body)
        # Remove thinking block from assistant message at index 1, block 0
        fmt.remove_thinking_block(body, msg_index=1, block_index=0)
        assert len(body["messages"][1]["content"]) == 1
        assert body["messages"][1]["content"][0]["type"] == "text"

    def test_remove_thinking_block_noop_openai(self):
        """Non-Anthropic formats don't have thinking blocks — should be a no-op."""
        body = {"model": "gpt-4", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]}
        fmt = detect_format(body)
        # Should not raise, should be a no-op
        fmt.remove_thinking_block(body, msg_index=1, block_index=0)
        assert body["messages"][1]["content"] == "hi there"

    def test_iter_media_blocks_replace_with_text(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
                {"type": "text", "text": "describe this"},
            ]},
            {"role": "assistant", "content": "it is a photo"},
        ]}
        fmt = detect_format(body)
        for media_info in fmt.iter_media_blocks(body):
            media_info.replace_with_text("[image removed]")
            break
        assert body["messages"][0]["content"][0] == {"type": "text", "text": "[image removed]"}

    def test_replace_tool_output_content(self):
        body = self._anthropic_body()
        fmt = detect_format(body)
        outputs = list(fmt.iter_tool_outputs(body))
        assert len(outputs) == 1
        fmt.replace_tool_output_content(body, outputs[0], "stubbed content")
        assert body["messages"][2]["content"][0]["content"] == "stubbed content"


class TestStubMarkers:
    def test_mark_and_detect(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "real message"},
            {"role": "assistant", "content": "real response"},
        ]}
        fmt = detect_format(body)
        fmt.mark_as_vc_stub(body["messages"][0])
        assert fmt.is_vc_stub(body, 0)
        assert not fmt.is_vc_stub(body, 1)

    def test_strip_markers(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "msg", "_vc_stub": True},
            {"role": "assistant", "content": "resp"},
        ]}
        fmt = detect_format(body)
        assert fmt.is_vc_stub(body, 0)
        fmt.strip_vc_markers(body)
        assert "_vc_stub" not in body["messages"][0]


class TestOpenAIResponsesMutations:
    def _responses_body(self):
        return {"model": "gpt-5", "input": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "fc1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc1", "output": "big result"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ]}

    def test_remove_items(self):
        body = self._responses_body()
        fmt = detect_format(body)
        fmt.remove_items(body, [2, 3])  # remove function_call pair
        assert len(body["input"]) == 3

    def test_replace_tool_output_content(self):
        body = self._responses_body()
        fmt = detect_format(body)
        outputs = list(fmt.iter_tool_outputs(body))
        assert len(outputs) == 1
        fmt.replace_tool_output_content(body, outputs[0], "stubbed")
        assert body["input"][3]["output"] == "stubbed"


class TestMergeConsecutiveConversational:
    def test_merges_consecutive_users_anthropic(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response"},
        ]}
        fmt = detect_format(body)
        fmt.merge_consecutive_conversational(body)
        assert len(body["messages"]) == 2
        # Content should be combined
        assert "first" in str(body["messages"][0]["content"])
        assert "second" in str(body["messages"][0]["content"])

    def test_does_not_merge_tool_messages(self):
        body = {"model": "gpt-4", "messages": [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "t2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "result1"},
            {"role": "tool", "tool_call_id": "t2", "content": "result2"},
            {"role": "assistant", "content": "done"},
        ]}
        fmt = detect_format(body)
        fmt.merge_consecutive_conversational(body)
        # Tool messages should NOT be merged
        assert len(body["messages"]) == 5

    def test_does_not_merge_function_items(self):
        body = {"model": "gpt-5", "input": [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "calling"}]},
            {"type": "function_call", "call_id": "fc1", "name": "a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc1", "output": "result1"},
            {"type": "function_call", "call_id": "fc2", "name": "b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc2", "output": "result2"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ]}
        fmt = detect_format(body)
        fmt.merge_consecutive_conversational(body)
        # function_call and function_call_output items must NOT be merged
        assert len(body["input"]) == 7

    def test_merges_consecutive_assistant_text_only(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "first part"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "second part"}]},
            {"role": "user", "content": "bye"},
        ]}
        fmt = detect_format(body)
        fmt.merge_consecutive_conversational(body)
        assert len(body["messages"]) == 3
        # Assistant messages merged
        merged = body["messages"][1]["content"]
        assert len(merged) == 2  # two text blocks combined


class TestStubToolOutputsCrossFormat:
    def test_stubs_anthropic_tool_results(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            # Recent turns (protected)
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "current"},
        ]}
        fmt = detect_format(body)
        tti = TurnTagIndex()
        result, count, refs = stub_tool_outputs_by_position(
            body, fmt, protected_recent_turns=3, turn_tag_index=tti,
            store=None, conversation_id="test",
        )
        assert count >= 1

    def test_stubs_openai_responses_function_outputs(self):
        body = {"model": "gpt-5", "input": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "calling"}]},
            {"type": "function_call", "call_id": "fc1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc1", "output": "x" * 10000},
            {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
            # Recent turns (protected)
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a2"}]},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a3"}]},
            {"role": "user", "content": "current"},
        ]}
        fmt = detect_format(body)
        tti = TurnTagIndex()
        result, count, refs = stub_tool_outputs_by_position(
            body, fmt, protected_recent_turns=3, turn_tag_index=tti,
            store=None, conversation_id="test",
        )
        assert count >= 1
        # The function_call_output should be stubbed
        item = body["input"][3]
        assert len(item.get("output", "")) < 10000

    def test_stubs_openai_chat_tool_messages(self):
        body = {"model": "gpt-4", "messages": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 10000},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "current"},
        ]}
        fmt = detect_format(body)
        tti = TurnTagIndex()
        result, count, refs = stub_tool_outputs_by_position(
            body, fmt, protected_recent_turns=3, turn_tag_index=tti,
            store=None, conversation_id="test",
        )
        assert count >= 1


class TestDropTopicOnlyStubs:
    def test_drops_stubs_without_restore_ref(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "stub user", "_vc_stub": True},
            {"role": "assistant", "content": "stub asst with no ref", "_vc_stub": True},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "real answer"},
        ]}
        fmt = detect_format(body)
        result, dropped = drop_topic_only_stubs(body, fmt)
        assert dropped == 2
        assert len(body["messages"]) == 2

    def test_keeps_stubs_with_restore_ref(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "[Compacted turn 0]", "_vc_stub": True},
            {"role": "assistant", "content": 'To restore: vc_restore_tool(ref="chain_0_abc")', "_vc_stub": True},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "real answer"},
        ]}
        fmt = detect_format(body)
        result, dropped = drop_topic_only_stubs(body, fmt)
        assert dropped == 0
        assert len(body["messages"]) == 4

    def test_keeps_non_stub_messages(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "real message"},
            {"role": "assistant", "content": "real response"},
        ]}
        fmt = detect_format(body)
        result, dropped = drop_topic_only_stubs(body, fmt)
        assert dropped == 0

    def test_drops_openai_stubs_without_restore_ref(self):
        body = {"model": "gpt-4", "messages": [
            {"role": "user", "content": "stub user", "_vc_stub": True},
            {"role": "assistant", "content": "stub asst", "_vc_stub": True},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "real answer"},
        ]}
        fmt = detect_format(body)
        result, dropped = drop_topic_only_stubs(body, fmt)
        assert dropped == 2
        assert len(body["messages"]) == 2

    def test_drops_openai_responses_stubs_without_restore_ref(self):
        body = {"model": "gpt-5", "input": [
            {"role": "user", "content": "stub user", "_vc_stub": True},
            {"role": "assistant", "content": [{"type": "output_text", "text": "stub asst"}], "_vc_stub": True},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "real answer"}]},
        ]}
        fmt = detect_format(body)
        result, dropped = drop_topic_only_stubs(body, fmt)
        assert dropped == 2
        assert len(body["input"]) == 2

    def test_mixed_stubs_drops_only_topic_only(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "topic stub user", "_vc_stub": True},
            {"role": "assistant", "content": "topic stub asst", "_vc_stub": True},
            {"role": "user", "content": "[Compacted turn 1]", "_vc_stub": True},
            {"role": "assistant", "content": 'Use vc_restore_tool(ref="chain_1_xyz") to expand', "_vc_stub": True},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "real answer"},
        ]}
        fmt = detect_format(body)
        result, dropped = drop_topic_only_stubs(body, fmt)
        assert dropped == 2
        assert len(body["messages"]) == 4
        # The restore-ref stubs and real messages remain
        assert body["messages"][0]["content"] == "[Compacted turn 1]"
        assert body["messages"][2]["content"] == "real question"

    def test_anthropic_list_content_stubs(self):
        """Stubs with list content (block-style) should also be inspected."""
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "stub user"}], "_vc_stub": True},
            {"role": "assistant", "content": [{"type": "text", "text": "stub asst no ref"}], "_vc_stub": True},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "real answer"},
        ]}
        fmt = detect_format(body)
        result, dropped = drop_topic_only_stubs(body, fmt)
        assert dropped == 2
        assert len(body["messages"]) == 2


class TestCollapseChainsCrossFormat:
    def test_collapses_anthropic_chains(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            # Old turn with tool activity -- should be collapsed
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/tmp/x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "file contents shown"}]},
            # Recent turns (protected)
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "current"},
        ]}
        fmt = detect_format(body)
        tti = TurnTagIndex()
        original_count = len(body["messages"])
        result, collapsed, refs = collapse_turn_chains(
            body, fmt, protected_recent_turns=3, turn_tag_index=tti,
            store=None, conversation_id="test",
        )
        # The old tool turn (4 messages) should be collapsed into a stub pair (2 messages)
        assert len(result["messages"]) < original_count
        # The last message should still be the current user question
        assert result["messages"][-1]["content"] == "current"
        assert collapsed >= 1

    def test_collapses_openai_responses_chains(self):
        body = {"model": "gpt-5", "input": [
            # Old turn with tool activity -- should be collapsed
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "calling read"}]},
            {"type": "function_call", "call_id": "fc1", "name": "read", "arguments": '{"path": "/tmp/x"}'},
            {"type": "function_call_output", "call_id": "fc1", "output": "x" * 10000},
            {"role": "assistant", "content": [{"type": "output_text", "text": "file contents shown"}]},
            # Recent turns (protected)
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a2"}]},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "a3"}]},
            {"role": "user", "content": "current"},
        ]}
        fmt = detect_format(body)
        tti = TurnTagIndex()
        original_count = len(body["input"])
        result, collapsed, refs = collapse_turn_chains(
            body, fmt, protected_recent_turns=3, turn_tag_index=tti,
            store=None, conversation_id="test",
        )
        # The old tool turn (5 items) should be collapsed into a stub pair (2 items)
        assert len(result["input"]) < original_count
        # The last item should still be the current user question
        assert result["input"][-1]["content"] == "current"
        assert collapsed >= 1

    def test_protects_recent_turns(self):
        """Turns inside the protected window should NOT be collapsed."""
        body = {"model": "claude-sonnet-4-6", "messages": [
            # Only two turns -- both within protected_recent_turns=3
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/tmp/x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": "current"},
        ]}
        fmt = detect_format(body)
        tti = TurnTagIndex()
        original_count = len(body["messages"])
        result, collapsed, refs = collapse_turn_chains(
            body, fmt, protected_recent_turns=3, turn_tag_index=tti,
            store=None, conversation_id="test",
        )
        # Everything is within the protected window -- nothing should be collapsed
        assert len(result["messages"]) == original_count
        assert collapsed == 0
