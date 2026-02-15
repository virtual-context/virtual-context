"""Tests for the headless replay runner."""

from __future__ import annotations

import json

import pytest

from conftest import FakeChatProvider
from virtual_context.tui.headless import HeadlessRunner
from virtual_context.tui.state import TurnRecord, load_replay_prompts
from virtual_context.types import Message, TurnTagEntry


def _make_runner(responses: list[str] | None = None, **kwargs) -> HeadlessRunner:
    """Build a HeadlessRunner with a fake provider and no engine."""
    runner = HeadlessRunner.__new__(HeadlessRunner)
    runner.engine = None
    runner.provider = FakeChatProvider(responses)
    runner._conversation_history = []
    runner._turns = []
    runner._turn_counter = 0
    runner._brief_mode = kwargs.get("brief_mode", True)
    runner._config_path = None
    runner._pending_complete = None
    from concurrent.futures import ThreadPoolExecutor
    runner._pool = ThreadPoolExecutor(max_workers=1)
    return runner


class TestHeadlessBasic:
    def test_three_prompts(self, tmp_path):
        runner = _make_runner(["Reply A.", "Reply B.", "Reply C."])
        turns = runner.run(
            ["hello", "how are you", "goodbye"], output=tmp_path
        )

        assert len(turns) == 3
        assert turns[0].user_message == "hello"
        assert turns[0].assistant_message == "Reply A."
        assert turns[1].user_message == "how are you"
        assert turns[1].assistant_message == "Reply B."
        assert turns[2].user_message == "goodbye"
        assert turns[2].assistant_message == "Reply C."

    def test_turn_numbers_sequential(self, tmp_path):
        runner = _make_runner(["a", "b"])
        turns = runner.run(["x", "y"], output=tmp_path)
        assert turns[0].turn_number == 1
        assert turns[1].turn_number == 2

    def test_conversation_history_built(self, tmp_path):
        runner = _make_runner(["ok"])
        runner.run(["test"], output=tmp_path)
        assert len(runner._conversation_history) == 2
        assert runner._conversation_history[0].role == "user"
        assert runner._conversation_history[1].role == "assistant"


class TestHeadlessSaveSession:
    def test_saves_json(self, tmp_path):
        runner = _make_runner(["Reply."])
        runner.run(["hello"], output=tmp_path)

        path = tmp_path / "vc-session.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["total_turns"] == 1
        assert data["turns"][0]["user_message"] == "hello"
        assert data["turns"][0]["assistant_message"] == "Reply."

    def test_api_payload_captured(self, tmp_path):
        runner = _make_runner(["Reply."])
        runner.run(["hello"], output=tmp_path)

        data = json.loads((tmp_path / "vc-session.json").read_text())
        payload = data["turns"][0]["api_payload"]
        assert "system" in payload
        assert "messages" in payload
        assert "total_history" in payload
        assert "filtered_history" in payload


class TestHeadlessBriefMode:
    def test_brief_mode_on(self, tmp_path):
        runner = _make_runner(["ok"], brief_mode=True)
        turns = runner.run(["test"], output=tmp_path)

        last_msg = turns[0].api_payload["messages"][-1]["content"]
        assert last_msg.endswith("(Answer in 2 lines.)")

    def test_brief_mode_off(self, tmp_path):
        runner = _make_runner(["ok"], brief_mode=False)
        turns = runner.run(["test"], output=tmp_path)

        last_msg = turns[0].api_payload["messages"][-1]["content"]
        assert "(Answer in 2 lines.)" not in last_msg


class TestHeadlessWithEngine:
    def test_engine_methods_called(self, tmp_path):
        """Verify engine integration points are exercised."""
        runner = _make_runner(["ok"])

        # Create a minimal mock engine
        class MockEngine:
            def __init__(self):
                self.inbound_calls = []
                self.complete_calls = []
                self.filter_calls = []
                self._turn_tag_index = MockIndex()
                self.config = type("C", (), {"context_window": 10000})()

            def on_message_inbound(self, msg, history):
                self.inbound_calls.append(msg)
                from virtual_context.types import AssembledContext
                return AssembledContext()

            def filter_history(self, history, current_tags=None, broad=False, temporal=False):
                self.filter_calls.append(current_tags)
                return history

            def on_turn_complete(self, history):
                self.complete_calls.append(len(history))
                return None

        class MockIndex:
            @property
            def entries(self):
                return []

        engine = MockEngine()
        runner.engine = engine
        runner.run(["question"], output=tmp_path)

        assert engine.inbound_calls == ["question"]
        assert len(engine.filter_calls) == 1
        assert engine.complete_calls == [2]  # 1 user + 1 assistant


class TestHeadlessTokenEstimate:
    def test_tokens_nonzero(self, tmp_path):
        runner = _make_runner(["A long response with many words."])
        turns = runner.run(["Tell me something"], output=tmp_path)
        assert turns[0].input_tokens > 0

    def test_turns_in_payload(self, tmp_path):
        runner = _make_runner(["a", "b", "c"])
        turns = runner.run(["x", "y", "z"], output=tmp_path)
        # First turn: 1 message → 1 bundled
        assert turns[0].turns_in_payload == 1
        # Third turn: 5 messages (3 user + 2 assistant) → 3 bundled
        assert turns[2].turns_in_payload == 3


class TestRuleTagFiltering:
    """Turns tagged 'rule' must always survive history filtering."""

    def test_rule_tag_always_included(self, sample_config):
        """A turn with 'rule' tag stays in filtered history even without tag overlap."""
        from virtual_context.engine import VirtualContextEngine

        engine = VirtualContextEngine.__new__(VirtualContextEngine)
        engine.config = sample_config

        # Manually set up the turn tag index
        from virtual_context.core.turn_tag_index import TurnTagIndex

        engine._turn_tag_index = TurnTagIndex()

        # Turn 0: teeth + rule (a behavioral instruction alongside teeth topic)
        engine._turn_tag_index.append(TurnTagEntry(
            turn_number=0, message_hash="a", tags=["teeth", "rule"],
            primary_tag="teeth",
        ))
        # Turn 1: pure database topic
        engine._turn_tag_index.append(TurnTagEntry(
            turn_number=1, message_hash="b", tags=["database"],
            primary_tag="database",
        ))

        history = [
            Message(role="user", content="Tell me about teeth, and always be honest"),
            Message(role="assistant", content="Teeth are..."),
            Message(role="user", content="What about databases?"),
            Message(role="assistant", content="Databases are..."),
            # Current turn (unpaired user message)
            Message(role="user", content="How do I create a table?"),
        ]

        # Filter with only "database" as current tags — teeth+rule turn should survive
        filtered = engine.filter_history(history, current_tags=["database"], recent_turns=1)

        # Recent turn (last 2 msgs) always included
        # Older: turn 0 (teeth+rule) included because of "rule" tag
        # Older: turn 1 (database) included because of tag overlap
        contents = [m.content for m in filtered]
        assert "Tell me about teeth, and always be honest" in contents
        assert "What about databases?" in contents

    def test_non_rule_turn_dropped_without_overlap(self, sample_config):
        """A turn WITHOUT 'rule' is correctly dropped when tags don't overlap."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.core.turn_tag_index import TurnTagIndex

        engine = VirtualContextEngine.__new__(VirtualContextEngine)
        engine.config = sample_config
        engine._turn_tag_index = TurnTagIndex()

        # Turn 0: pure teeth (no rule)
        engine._turn_tag_index.append(TurnTagEntry(
            turn_number=0, message_hash="a", tags=["teeth"],
            primary_tag="teeth",
        ))
        # Turn 1: database
        engine._turn_tag_index.append(TurnTagEntry(
            turn_number=1, message_hash="b", tags=["database"],
            primary_tag="database",
        ))

        history = [
            Message(role="user", content="Tell me about teeth"),
            Message(role="assistant", content="Teeth are..."),
            Message(role="user", content="What about databases?"),
            Message(role="assistant", content="Databases are..."),
            Message(role="user", content="How do I create a table?"),
        ]

        filtered = engine.filter_history(history, current_tags=["database"], recent_turns=1)

        contents = [m.content for m in filtered]
        # Teeth turn should be DROPPED (no rule tag, no overlap with "database")
        assert "Tell me about teeth" not in contents
        # Database turn stays (tag overlap)
        assert "What about databases?" in contents


class TestSlashCompact:
    """Tests for /compact slash command."""

    def test_compact_not_added_to_history(self, tmp_path):
        """/compact should NOT appear in conversation history."""
        runner = _make_runner(["Reply."])
        runner.run(["hello", "/compact"], output=tmp_path)
        # Only "hello" + its reply should be in conversation history
        roles = [m.role for m in runner._conversation_history]
        contents = [m.content for m in runner._conversation_history]
        assert roles == ["user", "assistant"]
        assert contents[0] == "hello"

    def test_compact_records_system_turn(self, tmp_path):
        """/compact should record a TurnRecord with _system tag."""
        runner = _make_runner(["Reply."])
        turns = runner.run(["hello", "/compact"], output=tmp_path)
        assert len(turns) == 2
        compact_turn = turns[1]
        assert compact_turn.user_message == "/compact"
        assert compact_turn.primary_tag == "_system"
        assert compact_turn.input_tokens == 0

    def test_compact_nothing_to_compact(self, tmp_path):
        """Without engine, /compact reports nothing."""
        runner = _make_runner(["Reply."])
        turns = runner.run(["/compact"], output=tmp_path)
        assert turns[0].assistant_message == "Nothing to compact."

    def test_compact_calls_engine(self, tmp_path):
        """/compact should call engine.compact_manual."""
        runner = _make_runner(["Reply."])

        class MockEngine:
            def __init__(self):
                self.compact_calls = 0
                self._turn_tag_index = type("I", (), {"entries": []})()

            def compact_manual(self, history):
                self.compact_calls += 1
                return None

        engine = MockEngine()
        runner.engine = engine
        runner.run(["/compact"], output=tmp_path)
        assert engine.compact_calls == 1

    def test_compact_in_session_json(self, tmp_path):
        """/compact turn should appear in session JSON."""
        runner = _make_runner(["Reply."])
        runner.run(["hello", "/compact"], output=tmp_path)

        data = json.loads((tmp_path / "vc-session.json").read_text())
        assert data["total_turns"] == 2
        assert data["turns"][1]["user_message"] == "/compact"
        assert data["turns"][1]["primary_tag"] == "_system"

    def test_compact_round_trips_via_replay(self, tmp_path):
        """/compact should be extractable from session JSON for replay."""
        runner = _make_runner(["Reply."])
        runner.run(["hello", "/compact", "goodbye"], output=tmp_path)

        prompts = load_replay_prompts(tmp_path / "vc-session.json")
        assert prompts == ["hello", "/compact", "goodbye"]

    def test_compact_between_normal_turns(self, tmp_path):
        """Normal turns before and after /compact should work correctly."""
        runner = _make_runner(["Reply A.", "Reply B."])
        turns = runner.run(["hello", "/compact", "goodbye"], output=tmp_path)
        assert len(turns) == 3
        assert turns[0].assistant_message == "Reply A."
        assert turns[1].user_message == "/compact"
        assert turns[2].assistant_message == "Reply B."
        # Conversation history should only have hello/reply + goodbye/reply
        assert len(runner._conversation_history) == 4
