"""Tests for proxy message filtering, stub compaction, and effective budget."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from virtual_context.proxy.server import (
    _filter_body_messages,
)
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import TurnTagEntry

class TestFilterBodyMessages:
    """Test history filtering of raw API request bodies."""

    def _build_index(self, turn_tags: list[list[str]]) -> TurnTagIndex:
        """Build a TurnTagIndex with the given per-turn tag lists."""
        idx = TurnTagIndex()
        for i, tags in enumerate(turn_tags):
            idx.append(TurnTagEntry(
                turn_number=i,
                message_hash=f"h{i}",
                tags=tags,
                primary_tag=tags[0] if tags else "_general",
            ))
        return idx

    def _build_body(self, n_pairs: int, current_user: bool = True) -> dict:
        """Build a request body with n user+assistant pairs + optional trailing user."""
        messages = []
        for i in range(n_pairs):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})
        if current_user:
            messages.append({"role": "user", "content": "Current question"})
        return {"messages": messages}

    def test_drops_irrelevant_turns(self):
        """Turns with no tag overlap are dropped."""
        # 5 history pairs + current user message
        body = self._build_body(5)
        idx = self._build_index([
            ["python", "testing"],   # turn 0 — matches
            ["cooking", "recipes"],  # turn 1 — no match
            ["music", "guitar"],     # turn 2 — no match
            ["python", "api"],       # turn 3 — matches
            ["weather"],             # turn 4 — no match (but protected)
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # Keep: turn 0, turn 3 (tag match), turn 4 (protected), current user
        assert dropped == 2  # turns 1 and 2 dropped
        msgs = filtered["messages"]
        assert len(msgs) == 7  # 3 kept pairs * 2 + current user

    def test_protects_recent_turns(self):
        """Recent N turns are always kept regardless of tags."""
        body = self._build_body(4)
        idx = self._build_index([
            ["python"],     # turn 0
            ["cooking"],    # turn 1
            ["music"],      # turn 2
            ["weather"],    # turn 3
        ])
        # No tag overlap at all, but recent_turns=2 protects turns 2 and 3
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated-tag"], recent_turns=2,
        )
        assert dropped == 2  # turns 0 and 1 dropped
        msgs = filtered["messages"]
        assert len(msgs) == 5  # 2 protected pairs * 2 + current user

    @pytest.mark.regression("BUG-008")
    def test_no_temporal_bypass_in_filter(self):
        """Time-scoped recall is tool-driven; history filter does not bypass."""
        body = self._build_body(3)
        idx = self._build_index([["a"], ["b"], ["c"]])
        filtered, dropped = _filter_body_messages(
            body, idx, ["x"], recent_turns=1,
        )
        assert dropped == 2

    def test_rule_tag_always_kept(self):
        """Turns tagged with 'rule' are always kept."""
        body = self._build_body(4)
        idx = self._build_index([
            ["rule", "style"],   # turn 0 — rule tag
            ["cooking"],         # turn 1 — no match
            ["music"],           # turn 2 — no match
            ["weather"],         # turn 3 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # turn 0 kept (rule), turn 3 kept (protected), turns 1+2 dropped
        assert dropped == 2
        msgs = filtered["messages"]
        assert len(msgs) == 5  # 2 pairs * 2 + current user

    @pytest.mark.regression("PROXY-002")
    def test_no_index_entries_skips_filtering(self):
        """If TurnTagIndex is empty, no filtering occurs."""
        body = self._build_body(5)
        idx = TurnTagIndex()  # empty
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        assert dropped == 0
        assert filtered is body

    def test_too_few_turns_skips_filtering(self):
        """If total turns <= recent_turns, no filtering occurs."""
        body = self._build_body(2)
        idx = self._build_index([["a"], ["b"]])
        filtered, dropped = _filter_body_messages(
            body, idx, ["x"], recent_turns=3,
        )
        assert dropped == 0

    def test_preserves_system_message(self):
        """OpenAI-style system message at position 0 is preserved."""
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["python"],    # turn 0
            ["cooking"],   # turn 1 — no match
            ["music"],     # turn 2 — no match
            ["python"],    # turn 3 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        assert dropped == 2
        msgs = filtered["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"
        # system + turn 0 pair + turn 3 pair + current user = 6
        assert len(msgs) == 6

    def test_no_tag_overlap_drops_all_older(self):
        """When nothing matches, only protected turns + current remain."""
        body = self._build_body(6)
        idx = self._build_index([
            ["a"], ["b"], ["c"], ["d"], ["e"], ["f"],
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["zzz-nonexistent"], recent_turns=2,
        )
        assert dropped == 4
        msgs = filtered["messages"]
        # 2 protected pairs * 2 + current user = 5
        assert len(msgs) == 5

    @pytest.mark.regression("PROXY-004")
    def test_tool_use_keeps_tool_result_pair(self):
        """If an assistant uses tool_use, the next pair (tool_result) must also be kept."""
        body = {
            "messages": [
                # Pair 0: normal turn (no match → would be dropped)
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: assistant uses tool (matched → kept)
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {}},
                ]},
                # Pair 2: tool_result (no match → but must be kept because pair 1 has tool_use)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "72F"},
                ]},
                {"role": "assistant", "content": "It's 72F"},
                # Pair 3: normal (no match → dropped)
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                # Pair 4: protected
                {"role": "user", "content": "Q4"},
                {"role": "assistant", "content": "A4"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],    # pair 0 — no match
            ["python"],     # pair 1 — matches
            ["cooking"],    # pair 2 — no match but forced by tool_use chain
            ["cooking"],    # pair 3 — no match
            ["cooking"],    # pair 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # pair 0 dropped, pair 1 kept (tag match), pair 2 force-kept (tool_result),
        # pair 3 dropped, pair 4 protected
        assert dropped == 2
        msgs = filtered["messages"]
        # 3 kept pairs * 2 + current user = 7
        assert len(msgs) == 7

    @pytest.mark.regression("PROXY-004")
    def test_tool_result_keeps_preceding_tool_use_pair(self):
        """If we keep a pair with tool_result, the preceding pair must also be kept."""
        body = {
            "messages": [
                # Pair 0: assistant uses tool (no tag match → would be dropped)
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool_1", "name": "search", "input": {}},
                ]},
                # Pair 1: tool_result + response (tag match → kept)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "results"},
                ]},
                {"role": "assistant", "content": "Found it"},
                # Pair 2: protected
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],    # pair 0 — no match but forced by pair 1's tool_result
            ["python"],     # pair 1 — matches
            ["cooking"],    # pair 2 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        # All 3 pairs kept (pair 0 force-kept due to tool chain)
        assert dropped == 0

    @pytest.mark.regression("PROXY-022")
    def test_consecutive_user_messages_preserve_alternation(self):
        """Unpaired consecutive user messages must not break role alternation.

        OpenClaw sends consecutive user messages (e.g., batched Telegram messages,
        tool_result followed by new user text without intervening assistant). When
        _filter_body_messages drops pairs around these unpaired messages, the result
        must still strictly alternate user/assistant for the Anthropic API.
        """
        # Reproduce real OpenClaw pattern: consecutive users at start
        body = {
            "messages": [
                # Unpaired user (e.g., batched Telegram message)
                {"role": "user", "content": "first batch msg"},
                # Pair 0: normal pair
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: no tag match → will be dropped
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                # Pair 2: protected (recent)
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],   # pair 0 — no match
            ["music"],     # pair 1 — no match
            ["weather"],   # pair 2 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated"], recent_turns=1,
        )
        msgs = filtered["messages"]
        # Verify strict role alternation in output
        for i in range(1, len(msgs)):
            assert msgs[i]["role"] != msgs[i - 1]["role"], (
                f"Consecutive same role at indices {i-1}->{i}: "
                f"{msgs[i-1]['role']}, {msgs[i]['role']}"
            )

    @pytest.mark.regression("PROXY-004")
    def test_consecutive_assistant_msgs_preserve_tool_use(self):
        """Consecutive assistant messages where the second has tool_use must not
        be dropped by role alternation enforcement.

        Claude Code with extended thinking can produce consecutive assistant
        messages: msg N = [thinking, text] (no tool_use), msg N+1 = [text, tool_use].
        The pairing logic pairs the user before msg N, leaving msg N+1 unpaired.
        When pair (user, msg N) is dropped, msg N+1 becomes consecutive with the
        previous assistant — role alternation would drop it, orphaning its
        tool_result in the next message.
        """
        body = {
            "messages": [
                # Pair 0: normal turn (matched → kept)
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: normal turn (no match → dropped)
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1-thinking-only"},
                # Unpaired assistant: consecutive with pair 1's assistant
                # Contains tool_use — critical for referential integrity
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check"},
                    {"type": "tool_use", "id": "tool_abc", "name": "Read",
                     "input": {"path": "foo.py"}},
                ]},
                # Pair 2: tool_result (no match → but forced by tool chain)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_abc",
                     "content": "file contents"},
                ]},
                {"role": "assistant", "content": "I read the file"},
                # Pair 3: normal turn (no match → dropped)
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                # Pair 4: protected
                {"role": "user", "content": "Q4"},
                {"role": "assistant", "content": "A4"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["python"],     # pair 0 — matches
            ["cooking"],    # pair 1 — no match → dropped
            ["cooking"],    # pair 2 — no match but forced by tool chain
            ["cooking"],    # pair 3 — no match
            ["cooking"],    # pair 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        msgs = filtered["messages"]

        # The unpaired assistant with tool_use MUST survive.
        # Verify no orphaned tool_result references.
        tool_use_ids = set()
        tool_result_refs = set()
        for msg in msgs:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_use_ids.add(block["id"])
                elif block.get("type") == "tool_result":
                    tool_result_refs.add(block["tool_use_id"])

        orphaned = tool_result_refs - tool_use_ids
        assert not orphaned, (
            f"Orphaned tool_result references: {orphaned}. "
            f"Role alternation dropped a tool_use message."
        )

        # Also verify role alternation is valid
        for i in range(1, len(msgs)):
            assert msgs[i].get("role") != msgs[i - 1].get("role"), (
                f"Consecutive same role at indices {i-1}->{i}: "
                f"{msgs[i-1].get('role')}, {msgs[i].get('role')}"
            )

    @pytest.mark.regression("PROXY-004b")
    def test_consecutive_assistant_dropped_pair_keeps_user_first(self):
        """Dropping pair 0 when an unpaired assistant[2] is force-kept must not
        leave the message list starting with role=assistant.

        Claude Code with extended thinking sends:
          msg[0]=user, msg[1]=assistant(thinking), msg[2]=assistant(tool_use),
          msg[3]=user(tool_result), ...
        Pair 0 is (0,1). msg[2] is unpaired. If pair 0 is dropped but msg[2]
        is kept (via tool_use_id integrity), the filtered output starts with
        assistant — which the Anthropic API rejects with 400.
        """
        body = {
            "messages": [
                # Pair 0: initial turn (no match → should be dropped)
                {"role": "user", "content": [{"type": "text", "text": "init"}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "thinking..."},
                ]},
                # Unpaired assistant: consecutive with pair 0's assistant
                {"role": "assistant", "content": [
                    {"type": "text", "text": "let me check"},
                    {"type": "tool_use", "id": "tu_glob", "name": "Glob",
                     "input": {"pattern": "*.py"}},
                    {"type": "tool_use", "id": "tu_bash", "name": "Bash",
                     "input": {"command": "ls"}},
                ]},
                # Pair 1: tool_result (no match → but forced by tool chain)
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_glob",
                     "content": "a.py b.py"},
                    {"type": "tool_result", "tool_use_id": "tu_bash",
                     "content": "ok"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                # Pair 2: no match → dropped
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                # Pair 3: no match → dropped
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                # Pair 4: protected
                {"role": "user", "content": "Q4"},
                {"role": "assistant", "content": "A4"},
                # Current user
                {"role": "user", "content": "current"},
            ],
        }
        idx = self._build_index([
            ["setup"],       # pair 0 — no match
            ["setup"],       # pair 1 — no match (forced by tool chain)
            ["cooking"],     # pair 2 — no match
            ["cooking"],     # pair 3 — no match
            ["cooking"],     # pair 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        msgs = filtered["messages"]

        # First message MUST be role=user (Anthropic API requirement)
        first_chat = None
        for msg in msgs:
            if msg.get("role") in ("user", "assistant"):
                first_chat = msg
                break
        assert first_chat is not None
        assert first_chat["role"] == "user", (
            f"First chat message is '{first_chat['role']}' — "
            f"Anthropic API requires first message to be 'user'"
        )

        # No orphaned tool_results
        tu_ids = set()
        tr_ids = set()
        for msg in msgs:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tu_ids.add(block["id"])
                elif block.get("type") == "tool_result":
                    tr_ids.add(block["tool_use_id"])
        orphaned = tr_ids - tu_ids
        assert not orphaned, f"Orphaned tool_result references: {orphaned}"

    @pytest.mark.regression("PROXY-004c")
    def test_consecutive_assistant_thinking_strip_preserves_tool_chain(self):
        """Consecutive assistants with thinking blocks must not orphan tool_results.

        Reproduces the exact layout from A/B run 2026-03-01, request_log/000038:

          msg[48] user   [tool_result(prev)]       — pair N
          msg[49] assistant [thinking, text]        — pair N (response without tool_use)
          msg[50] assistant [thinking, text, tool_use(X)]  — UNPAIRED (consecutive asst)
          msg[51] user   [tool_result(X)]           — pair N+1

        _strip_thinking_blocks creates new dicts for msgs 49 and 50 (both have
        thinking blocks).  If _vc_critical is set on original chat_msgs instead
        of the copies in `kept`, alternation enforcement can't see the sentinel
        on msg[50]'s copy → drops it → tool_result(X) orphaned → API 400.
        """
        body = {
            "messages": [
                # Pair 0: tag match → keep
                {"role": "user", "content": "analyze the data"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "ok",
                     "signature": "sig_a0"},
                    {"type": "text", "text": "I'll analyze it"},
                    {"type": "tool_use", "id": "tu_read1", "name": "Read",
                     "input": {"file_path": "/data.csv"}},
                ]},
                # Pair 1: tag match → keep
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_read1",
                     "content": "col1,col2\n1,2\n3,4"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "got it",
                     "signature": "sig_a1"},
                    {"type": "text", "text": "loaded data"},
                    {"type": "tool_use", "id": "tu_bash1", "name": "Bash",
                     "input": {"command": "python analyze.py"}},
                ]},
                # Pair 2: no match → dropped
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_bash1",
                     "content": "analysis complete"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "done",
                     "signature": "sig_a2"},
                    {"type": "text", "text": "analysis done"},
                ]},
                # --- THE BUG PATTERN: consecutive assistants ---
                # UNPAIRED assistant (consecutive with pair 2's assistant)
                # Has thinking + tool_use — _strip_thinking_blocks will copy it
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "wait let me also run tests",
                     "signature": "sig_a_unpaired"},
                    {"type": "text", "text": "Let me also run the tests"},
                    {"type": "tool_use", "id": "tu_target", "name": "Bash",
                     "input": {"command": "pytest"}},
                ]},
                # Pair 3: no match → dropped, but tool_result forces keep
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_target",
                     "content": "===== test session starts =====\n3 passed"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "All tests pass"},
                ]},
                # Pair 4: no match → dropped
                {"role": "user", "content": "what about coverage?"},
                {"role": "assistant", "content": "I can check coverage next"},
                # Pair 5: no match → dropped
                {"role": "user", "content": "and linting?"},
                {"role": "assistant", "content": "will check linting too"},
                # Pair 6: protected (recent)
                {"role": "user", "content": "ok do it"},
                {"role": "assistant", "content": "on it"},
                # Current user turn
                {"role": "user", "content": "status?"},
            ],
        }
        idx = self._build_index([
            ["data-analysis"],  # pair 0 — match
            ["data-analysis"],  # pair 1 — match
            ["testing"],        # pair 2 — no match
            ["testing"],        # pair 3 — no match (forced by tool chain)
            ["coverage"],       # pair 4 — no match
            ["linting"],        # pair 5 — no match
            ["linting"],        # pair 6 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["data-analysis"], recent_turns=1,
        )
        msgs = filtered["messages"]

        # Thinking blocks must be stripped (some messages were dropped)
        for msg in msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert block.get("type") != "thinking", \
                            "Thinking blocks should be stripped after dropping"

        # The critical check: no orphaned tool_results
        tu_ids = set()
        tr_ids = set()
        for msg in msgs:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tu_ids.add(block["id"])
                elif block.get("type") == "tool_result":
                    tr_ids.add(block["tool_use_id"])
        orphaned = tr_ids - tu_ids
        assert not orphaned, (
            f"Orphaned tool_result references: {orphaned}. "
            f"PROXY-004c: _strip_thinking_blocks created a copy of the "
            f"assistant with tool_use(tu_target) and the _vc_critical "
            f"sentinel was lost on the copy."
        )

    @pytest.mark.regression("PROXY-023")
    def test_compacted_turns_dropped_when_paging_active(self):
        """When paging is active, turns below compacted_turn watermark are dropped.

        Even if a compacted turn has matching tags, it should be dropped because
        the content is available via VC summaries and expandable via paging tools.
        Without this, the LLM always has the raw messages and never needs paging.
        """
        # 8 history pairs: turns 0-4 are "compacted", turns 5-7 are not
        body = self._build_body(8)
        idx = self._build_index([
            ["python", "testing"],   # turn 0 — matches but compacted
            ["python", "api"],       # turn 1 — matches but compacted
            ["cooking"],             # turn 2 — no match, compacted
            ["python", "debug"],     # turn 3 — matches but compacted
            ["music"],               # turn 4 — no match, compacted
            ["python", "deploy"],    # turn 5 — matches, NOT compacted → keep
            ["weather"],             # turn 6 — no match, not compacted
            ["cars"],                # turn 7 — no match, protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
            compacted_turn=5,  # turns 0-4 are compacted
        )
        # turns 0-4 ALL dropped (compacted, even though 0,1,3 match tags)
        # turn 5 kept (matches, not compacted)
        # turn 6 dropped (no match, not compacted)
        # turn 7 kept (protected)
        assert dropped == 6
        msgs = filtered["messages"]
        # 2 kept pairs * 2 + current user = 5
        assert len(msgs) == 5
        # Verify the kept messages are from turns 5 and 7
        assert msgs[0]["content"] == "Question 5"
        assert msgs[1]["content"] == "Answer 5"
        assert msgs[2]["content"] == "Question 7"
        assert msgs[3]["content"] == "Answer 7"
        assert msgs[4]["content"] == "Current question"

    @pytest.mark.regression("PROXY-023")
    def test_compacted_turn_zero_preserves_current_behavior(self):
        """When compacted_turn=0 (default/no paging), filter behaves normally.

        Tag-matching turns are kept even if they're old. This is the existing
        behavior and must not regress.
        """
        body = self._build_body(5)
        idx = self._build_index([
            ["python"],    # turn 0 — matches → kept
            ["cooking"],   # turn 1 — no match → dropped
            ["music"],     # turn 2 — no match → dropped
            ["python"],    # turn 3 — matches → kept
            ["weather"],   # turn 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
            compacted_turn=0,
        )
        # Same as without compacted_turn: turns 0,3 kept (match), turn 4 (protected)
        assert dropped == 2
        msgs = filtered["messages"]
        assert len(msgs) == 7  # 3 pairs * 2 + current user

    @pytest.mark.regression("PROXY-023")
    def test_compacted_turns_rule_tag_still_dropped(self):
        """Even 'rule' tagged turns are dropped when compacted and paging is active.

        Rule tags normally force-keep turns, but compacted content has already
        been summarized. The paging system can retrieve it if needed.
        """
        body = self._build_body(5)
        idx = self._build_index([
            ["rule", "style"],   # turn 0 — rule tag but compacted
            ["cooking"],         # turn 1 — compacted
            ["music"],           # turn 2 — not compacted, no match → dropped
            ["python"],          # turn 3 — not compacted, matches → kept
            ["weather"],         # turn 4 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
            compacted_turn=2,  # turns 0-1 compacted
        )
        # turn 0 dropped (compacted, even with rule tag)
        # turn 1 dropped (compacted)
        # turn 2 dropped (no match)
        # turn 3 kept (match, not compacted)
        # turn 4 kept (protected)
        assert dropped == 3
        msgs = filtered["messages"]
        assert len(msgs) == 5  # 2 pairs * 2 + current user

    @pytest.mark.regression("PROXY-022")
    def test_consecutive_user_after_tool_result_preserves_alternation(self):
        """tool_result user followed by text user must not break alternation.

        Real pattern from OpenClaw: assistant uses tool → user sends tool_result →
        user sends new text message (no intervening assistant). When pairs around
        the unpaired tool_result are dropped, alternation must be preserved.
        """
        body = {
            "messages": [
                # Pair 0: tag match → kept
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1: assistant uses tool → no tag match
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
                ]},
                # Unpaired user: tool_result without subsequent assistant
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                ]},
                # Pair 2: new user text + assistant (this pairs with the next assistant)
                {"role": "user", "content": "Q2-new-topic"},
                {"role": "assistant", "content": "A2"},
                # Pair 3: protected (recent)
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["python"],    # pair 0 — matches
            ["cooking"],   # pair 1 — no match
            ["music"],     # pair 2 — no match
            ["weather"],   # pair 3 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["python"], recent_turns=1,
        )
        msgs = filtered["messages"]
        # Verify strict role alternation
        for i in range(1, len(msgs)):
            assert msgs[i]["role"] != msgs[i - 1]["role"], (
                f"Consecutive same role at indices {i-1}->{i}: "
                f"{msgs[i-1]['role']}, {msgs[i]['role']}"
            )

    def test_dropped_never_negative(self):
        """dropped count must never be negative, even with unpaired messages.

        When alternation enforcement silently removes extra messages,
        the drop count must be computed from the final kept list.
        """
        # Build a body with unpaired messages that will cause alternation enforcement
        body = {
            "messages": [
                # Unpaired user (batched Telegram)
                {"role": "user", "content": "batch1"},
                # Another unpaired user
                {"role": "user", "content": "batch2"},
                # Pair 0
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                # Pair 1
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                # Current user
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["python"],  # pair 0
            ["music"],   # pair 1
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated"], recent_turns=1,
        )
        assert dropped >= 0, f"dropped should never be negative, got {dropped}"
        # Verify alternation
        msgs = filtered["messages"]
        for i in range(1, len(msgs)):
            assert msgs[i]["role"] != msgs[i - 1]["role"]

    def test_dropped_count_with_alternation_enforcement(self):
        """Alternation enforcement removes messages beyond pair-based drops.

        The dropped count must reflect ALL removed user turns, including
        those removed by alternation enforcement.
        """
        body = {
            "messages": [
                {"role": "user", "content": "unpaired"},
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Current"},
            ],
        }
        idx = self._build_index([
            ["cooking"],  # pair 0 — no match
            ["weather"],  # pair 1 — protected
        ])
        filtered, dropped = _filter_body_messages(
            body, idx, ["unrelated"], recent_turns=1,
        )
        assert dropped >= 0
        # Verify no negative payload in the formula: total_turns - dropped
        total_pairs = 2
        assert total_pairs - dropped >= 0



# ---------------------------------------------------------------------------
# stub_compacted_messages
# ---------------------------------------------------------------------------


class TestStubCompactedMessages:
    """Test hash-based stub replacement for compacted turns."""

    def _build_index_with_hashes(self, entries: list[tuple[int, str, list[str]]]) -> TurnTagIndex:
        """Build index: list of (turn_number, message_hash, tags)."""
        idx = TurnTagIndex()
        for turn, hash_, tags in entries:
            idx.append(TurnTagEntry(
                turn_number=turn, message_hash=hash_, tags=tags,
                primary_tag=tags[0] if tags else "_general",
            ))
        return idx

    @pytest.mark.regression("PROXY-025")
    def test_stubs_compacted_turn_simple_pair(self):
        """A simple user+assistant pair is stubbed when its hash matches a compacted turn."""
        import hashlib
        user_text = "What is pregnancy testing?"
        asst_text = "Pregnancy testing involves..."
        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        body = {"messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
            {"role": "user", "content": "Follow up question"},
        ]}
        idx = self._build_index_with_hashes([
            (0, h, ["pregnancy-test"]),
            (1, "other_hash", ["general"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_prefix_messages=2,  # turn 0 is compacted
        )
        msgs = result["messages"]
        assert stub_count == 1
        # User message stubbed
        assert "[Compacted turn 0" in msgs[0]["content"]
        # Assistant message stubbed
        assert isinstance(msgs[1]["content"], list)
        assert "[Compacted turn 0" in msgs[1]["content"][0]["text"]
        # Follow-up preserved
        assert msgs[2]["content"] == "Follow up question"

    @pytest.mark.regression("PROXY-025")
    def test_stubs_turn_with_tool_chain(self):
        """A turn with tool_use/tool_result chain is collapsed to user+assistant stubs."""
        import hashlib
        user_text = "Read the config file"
        asst_text = "Let me read that for you."
        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        body = {"messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": asst_text},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "/etc/config"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents here..."},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "The config contains..."},
            ]},
            # Next turn (not compacted)
            {"role": "user", "content": "What does it mean?"},
            {"role": "assistant", "content": "It means..."},
            {"role": "user", "content": "Current question"},
        ]}
        idx = self._build_index_with_hashes([
            (0, h, ["config", "file-reading"]),
            (1, "hash_turn1", ["interpretation"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_prefix_messages=2,  # turn 0 compacted
        )
        msgs = result["messages"]
        assert stub_count == 1
        # Tool chain collapsed: 4 messages → 2 stubs
        # First msg: user stub
        assert msgs[0]["role"] == "user"
        assert "[Compacted turn 0" in str(msgs[0]["content"])
        # Second msg: assistant stub
        assert msgs[1]["role"] == "assistant"
        assert "[Compacted turn 0" in str(msgs[1]["content"])
        # No tool_use_id or tool_result blocks survive
        all_content = json.dumps(msgs[:2])
        assert "tool_use" not in all_content
        assert "tool_result" not in all_content
        # Uncompacted turn preserved
        assert msgs[2]["content"] == "What does it mean?"

    def test_uncompacted_turn_hash_match_preserved(self):
        """A turn whose hash matches but is above watermark is NOT stubbed."""
        import hashlib
        user_text = "Hello"
        asst_text = "Hi there"
        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        body = {"messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
            {"role": "user", "content": "Current"},
        ]}
        idx = self._build_index_with_hashes([
            (0, h, ["greeting"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        # compacted_prefix_messages=0 means nothing is compacted
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_prefix_messages=0,
        )
        assert stub_count == 0
        assert result["messages"][0]["content"] == user_text

    def test_hash_miss_preserves_message(self):
        """Messages whose hash doesn't match any entry are preserved."""
        body = {"messages": [
            {"role": "user", "content": "Unique message not in index"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Current"},
        ]}
        idx = self._build_index_with_hashes([
            (0, "some_other_hash", ["topic"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_prefix_messages=2,
        )
        assert stub_count == 0
        assert result["messages"][0]["content"] == "Unique message not in index"

    @pytest.mark.regression("PROXY-025")
    def test_stub_eliminates_tool_chain_integrity_issue(self):
        """Stubbing compacted turns eliminates tool_use_id references.

        Simulates PROXY-025: compacted turns with tool_use/tool_result chains
        that would normally be force-kept by referential integrity. After stubbing,
        no tool_use_id survives, so the integrity loop has nothing to chase.
        """
        import hashlib

        # Turn 0: user asks, assistant uses a tool, tool returns result
        user_text_0 = "Read the config file"
        asst_text_0 = "Let me read that."
        h0 = hashlib.sha256(f"{user_text_0} {asst_text_0}".encode()).hexdigest()[:16]

        # Turn 1: user asks about result, assistant responds with another tool
        user_text_1 = "Now edit it"
        asst_text_1 = "I'll edit the file."
        h1 = hashlib.sha256(f"{user_text_1} {asst_text_1}".encode()).hexdigest()[:16]

        body = {"messages": [
            # Turn 0 (compacted)
            {"role": "user", "content": [{"type": "text", "text": user_text_0}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": asst_text_0},
                {"type": "tool_use", "id": "toolu_001", "name": "Read", "input": {"path": "/config"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_001", "content": "config data here"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Config contains X."}]},
            # Turn 1 (compacted)
            {"role": "user", "content": [{"type": "text", "text": user_text_1}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": asst_text_1},
                {"type": "tool_use", "id": "toolu_002", "name": "Edit", "input": {"path": "/config"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_002", "content": "edited"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done editing."}]},
            # Turn 2 (not compacted — current)
            {"role": "user", "content": "What did we change?"},
        ]}

        idx = TurnTagIndex()
        idx.append(TurnTagEntry(turn_number=0, message_hash=h0,
                                tags=["config", "file-reading"], primary_tag="config"))
        idx.append(TurnTagEntry(turn_number=1, message_hash=h1,
                                tags=["config", "file-editing"], primary_tag="config"))

        from virtual_context.proxy.message_filter import stub_compacted_messages
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_prefix_messages=4,  # turns 0-1 compacted (4 internal messages)
        )
        msgs = result["messages"]

        assert stub_count == 2

        # Verify NO tool_use or tool_result blocks anywhere in stubbed messages
        for m in msgs[:-1]:  # exclude current user message
            content = m.get("content", "")
            if isinstance(content, list):
                for block in content:
                    assert block.get("type") != "tool_use", f"tool_use survived: {block}"
                    assert block.get("type") != "tool_result", f"tool_result survived: {block}"

        # Verify role alternation
        roles = [m["role"] for m in msgs]
        expected_roles = ["user", "assistant", "user", "assistant", "user"]
        assert roles == expected_roles, f"Role alternation broken: {roles}"

        # Verify current turn preserved
        assert msgs[-1]["content"] == "What did we change?"



# ---------------------------------------------------------------------------
# _compute_effective_budget
# ---------------------------------------------------------------------------


class TestComputeEffectiveBudget:
    def test_compute_effective_budget_within_limit(self):
        """When overhead < context_window, budget is unchanged."""
        from virtual_context.proxy.server import _compute_effective_budget
        budget, promoted = _compute_effective_budget(
            context_window=120_000,
            system_tokens=19_000,
            tools_tokens=7_500,
        )
        assert budget == 120_000
        assert promoted is False

    @pytest.mark.regression("PROXY-025")
    def test_compute_effective_budget_auto_promotes(self):
        """When overhead >= context_window, budget auto-promotes to overhead + 10k."""
        from virtual_context.proxy.server import _compute_effective_budget
        budget, promoted = _compute_effective_budget(
            context_window=5_000,
            system_tokens=19_000,
            tools_tokens=7_500,
        )
        assert budget == 19_000 + 7_500 + 10_000  # 36,500
        assert promoted is True
