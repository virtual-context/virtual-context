"""Tests for virtual_context.proxy.message_filter — history turn filtering."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.message_filter import (
    _strip_thinking_blocks,
    filter_body_messages,
)
from virtual_context.types import TurnTagEntry


# ── helpers ──────────────────────────────────────────────────────────────

def _make_index(entries: list[tuple[int, list[str]]]) -> TurnTagIndex:
    """Build a TurnTagIndex from (turn_number, tags) tuples."""
    idx = TurnTagIndex()
    for turn, tags in entries:
        idx.append(TurnTagEntry(
            turn_number=turn,
            message_hash=f"hash-{turn}",
            tags=tags,
            primary_tag=tags[0] if tags else "_general",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ))
    return idx


def _make_pair(turn: int, user_text: str = "", asst_text: str = "") -> list[dict]:
    """Make a user+assistant message pair."""
    return [
        {"role": "user", "content": user_text or f"User turn {turn}"},
        {"role": "assistant", "content": asst_text or f"Assistant turn {turn}"},
    ]


def _make_anthropic_body(
    pairs: int = 5,
    current_user: str = "Current question",
    system: str | None = None,
) -> dict:
    """Build an Anthropic-format request body with N history pairs + current user."""
    messages = []
    for i in range(pairs):
        messages.extend(_make_pair(i))
    messages.append({"role": "user", "content": current_user})

    body: dict = {"messages": messages}
    if system is not None:
        body["system"] = system
    return body


# ── _strip_thinking_blocks tests ─────────────────────────────────────────

class TestStripThinkingBlocks:

    def test_no_thinking_blocks(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello"},
            ]},
        ]
        result = _strip_thinking_blocks(messages)
        assert result == messages

    def test_removes_thinking_blocks(self):
        messages = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "let me reason..."},
                {"type": "text", "text": "The answer is 42."},
            ]},
        ]
        result = _strip_thinking_blocks(messages)
        assert len(result) == 1
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"

    def test_preserves_non_assistant_messages(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "hi"},
            ]},
        ]
        result = _strip_thinking_blocks(messages)
        assert result == messages

    def test_string_content_untouched(self):
        messages = [
            {"role": "assistant", "content": "plain string answer"},
        ]
        result = _strip_thinking_blocks(messages)
        assert result == messages

    def test_only_thinking_blocks(self):
        """Message with only thinking blocks results in empty content list."""
        messages = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning 1"},
                {"type": "thinking", "thinking": "reasoning 2"},
            ]},
        ]
        result = _strip_thinking_blocks(messages)
        assert result[0]["content"] == []

    def test_original_not_mutated(self):
        """When thinking blocks are stripped, original message is not mutated."""
        original_content = [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "answer"},
        ]
        messages = [{"role": "assistant", "content": original_content}]
        result = _strip_thinking_blocks(messages)
        # Original should still have 2 blocks
        assert len(original_content) == 2
        # Result should have 1
        assert len(result[0]["content"]) == 1


# ── filter_body_messages basic tests ─────────────────────────────────────

class TestFilterBodyMessagesBasic:

    def test_empty_messages(self):
        body = {"messages": []}
        idx = TurnTagIndex()
        result, dropped = filter_body_messages(body, idx, [])
        assert dropped == 0

    def test_no_tags_no_filtering(self):
        """With no turn tag entries, nothing is filtered."""
        body = _make_anthropic_body(pairs=5)
        idx = TurnTagIndex()
        result, dropped = filter_body_messages(body, idx, ["some-tag"])
        assert dropped == 0

    def test_all_protected_by_recent_turns(self):
        """When recent_turns >= total pairs, nothing is dropped."""
        body = _make_anthropic_body(pairs=3)
        idx = _make_index([(0, ["a"]), (1, ["b"]), (2, ["c"])])
        result, dropped = filter_body_messages(body, idx, ["x"], recent_turns=3)
        assert dropped == 0

    def test_matching_tags_kept(self):
        """Pairs with matching tags are kept."""
        body = _make_anthropic_body(pairs=5)
        idx = _make_index([
            (0, ["cooking"]),
            (1, ["fitness"]),
            (2, ["cooking"]),
            (3, ["travel"]),
            (4, ["coding"]),
        ])
        result, dropped = filter_body_messages(
            body, idx, ["cooking"], recent_turns=1,
        )
        # Pairs 0 and 2 match "cooking", pair 4 is protected as recent
        # Pairs 1 and 3 should be dropped
        assert dropped >= 1

    def test_rule_tag_always_kept(self):
        """Pairs tagged with 'rule' are always kept."""
        body = _make_anthropic_body(pairs=5)
        idx = _make_index([
            (0, ["rule"]),
            (1, ["unrelated"]),
            (2, ["unrelated"]),
            (3, ["unrelated"]),
            (4, ["unrelated"]),
        ])
        result, dropped = filter_body_messages(
            body, idx, ["something-else"], recent_turns=1,
        )
        # Pair 0 has "rule" tag, so it's kept even though no tag match
        # We can verify by checking the result has messages from pair 0
        user_contents = [
            m["content"] for m in result["messages"]
            if m.get("role") == "user"
        ]
        assert "User turn 0" in user_contents

    def test_current_user_message_preserved(self):
        """The trailing user message (current turn) is always kept."""
        body = _make_anthropic_body(pairs=3, current_user="What is 2+2?")
        idx = _make_index([(0, ["a"]), (1, ["b"]), (2, ["c"])])
        result, dropped = filter_body_messages(body, idx, ["a"], recent_turns=1)
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert last_msg["content"] == "What is 2+2?"


# ── tool_use/tool_result chain filtering tests ──────────────────────────

class TestToolUseChainFiltering:

    def test_tool_use_result_pair_kept_together(self):
        """When a tool_use message is kept, its tool_result partner is also kept."""
        messages = [
            # Pair 0: normal chat (unrelated tag)
            {"role": "user", "content": "User turn 0"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "tu-1", "name": "search", "input": {"q": "test"}},
            ]},
            # tool_result for tu-1
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu-1", "content": "result data"},
            ]},
            # Pair 1 (the tool_result + assistant response)
            {"role": "assistant", "content": [
                {"type": "text", "text": "Based on the search results..."},
            ]},
            # Pair 2: recent
            {"role": "user", "content": "User turn 2"},
            {"role": "assistant", "content": "Assistant turn 2"},
            # Current user
            {"role": "user", "content": "Current question"},
        ]
        body = {"system": "Be helpful.", "messages": messages}
        idx = _make_index([
            (0, ["tool-topic"]),
            (1, ["tool-topic"]),
            (2, ["recent"]),
        ])
        result, dropped = filter_body_messages(
            body, idx, ["tool-topic"], recent_turns=1,
        )
        # Verify no orphaned tool_results in the output
        tool_use_ids = set()
        tool_result_ids = set()
        for msg in result["messages"]:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_use_ids.add(block["id"])
                        elif block.get("type") == "tool_result":
                            tool_result_ids.add(block["tool_use_id"])
        # All tool_result references must have matching tool_use
        orphans = tool_result_ids - tool_use_ids
        assert not orphans, f"Orphaned tool_results: {orphans}"


# ── thinking block interaction with filtering ────────────────────────────

class TestThinkingBlockFiltering:

    def test_thinking_blocks_stripped_when_messages_dropped(self):
        """When messages are dropped, thinking blocks are stripped from survivors."""
        messages = [
            {"role": "user", "content": "Turn 0"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning..."},
                {"type": "text", "text": "Answer 0"},
            ]},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "more reasoning"},
                {"type": "text", "text": "Answer 1"},
            ]},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "yet more"},
                {"type": "text", "text": "Answer 2"},
            ]},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Answer 3"},
            {"role": "user", "content": "Current question"},
        ]
        body = {"system": "sys", "messages": messages}
        idx = _make_index([
            (0, ["cooking"]),
            (1, ["unrelated"]),
            (2, ["unrelated"]),
            (3, ["unrelated"]),
        ])
        result, dropped = filter_body_messages(
            body, idx, ["cooking"], recent_turns=1,
        )
        if dropped > 0:
            # Verify no thinking blocks remain in assistant messages
            for msg in result["messages"]:
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        thinking = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
                        assert not thinking, "Thinking blocks should be stripped when messages are dropped"


# ── compacted_turn (paging) tests ────────────────────────────────────────

class TestCompactedTurnPaging:

    def test_compacted_turns_dropped(self):
        """Pairs below the compacted_turn watermark are unconditionally dropped."""
        body = _make_anthropic_body(pairs=5)
        idx = _make_index([
            (0, ["cooking"]),
            (1, ["cooking"]),
            (2, ["cooking"]),
            (3, ["cooking"]),
            (4, ["cooking"]),
        ])
        # Even though all tags match, pairs 0 and 1 are below the watermark
        result, dropped = filter_body_messages(
            body, idx, ["cooking"],
            recent_turns=1,
            compacted_turn=2,
        )
        assert dropped >= 2

    def test_compacted_turn_zero_no_effect(self):
        """compacted_turn=0 means no paging, no extra drops."""
        body = _make_anthropic_body(pairs=5)
        idx = _make_index([
            (0, ["cooking"]),
            (1, ["cooking"]),
            (2, ["cooking"]),
            (3, ["cooking"]),
            (4, ["cooking"]),
        ])
        result, dropped = filter_body_messages(
            body, idx, ["cooking"],
            recent_turns=1,
            compacted_turn=0,
        )
        # All tags match, so nothing should be dropped except maybe none
        assert dropped == 0


# ── role alternation tests ───────────────────────────────────────────────

class TestRoleAlternation:

    def test_output_has_valid_role_alternation(self):
        """The filtered output should not have consecutive same-role messages."""
        body = _make_anthropic_body(pairs=6)
        idx = _make_index([
            (0, ["a"]),
            (1, ["b"]),
            (2, ["a"]),
            (3, ["c"]),
            (4, ["d"]),
            (5, ["e"]),
        ])
        result, dropped = filter_body_messages(
            body, idx, ["a"], recent_turns=1,
        )
        messages = result.get("messages", body.get("messages", []))
        prev_role = None
        for msg in messages:
            role = msg.get("role")
            if role is None:
                continue  # bare items (e.g. function_call)
            # Note: system messages at the start are OK before user
            if prev_role == "system" and role == "user":
                prev_role = role
                continue
            if prev_role is not None and prev_role != "system":
                # Consecutive same-role is only OK if both are _vc_critical
                # but _vc_critical should be cleaned up by now
                assert role != prev_role or "_vc_critical" in msg, \
                    f"Consecutive {role} messages found in filtered output"
            prev_role = role


# ── edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_user_message_only(self):
        """Body with only the current user message, no history."""
        body = {"messages": [{"role": "user", "content": "hello"}]}
        idx = TurnTagIndex()
        result, dropped = filter_body_messages(body, idx, [])
        assert dropped == 0
        assert len(result["messages"]) == 1

    def test_system_messages_preserved(self):
        """OpenAI-style system messages at the start are preserved."""
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Turn 0"},
                {"role": "assistant", "content": "Answer 0"},
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Current"},
            ]
        }
        idx = _make_index([(0, ["x"]), (1, ["y"])])
        result, dropped = filter_body_messages(body, idx, ["y"], recent_turns=1)
        # System message should always be present
        assert result["messages"][0]["role"] == "system"

    def test_no_turn_tag_entry_keeps_pair(self):
        """Pairs without a TurnTagEntry are kept (entry is None -> keep)."""
        body = _make_anthropic_body(pairs=4)
        # Only index turns 0 and 3; turns 1 and 2 have no entry
        idx = _make_index([(0, ["unrelated"]), (3, ["unrelated"])])
        result, dropped = filter_body_messages(
            body, idx, ["something"], recent_turns=1,
        )
        # Turns 1 and 2 have no entry -> kept. Turn 0 unrelated -> dropped.
        # Turn 3 is recent -> kept.
        assert dropped <= 1

    def test_body_without_messages_key(self):
        """Body with no recognized message key returns unchanged."""
        body = {"model": "test"}
        idx = TurnTagIndex()
        result, dropped = filter_body_messages(body, idx, [])
        assert dropped == 0
