"""Tests for tag splitting: TagSplitter, TurnTagIndex additions, engine integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from virtual_context.core.tag_splitter import TagSplitter
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import (
    EngineStateSnapshot,
    Message,
    SplitResult,
    TagSplittingConfig,
    TurnTagEntry,
)


# ---------------------------------------------------------------------------
# TagSplitter unit tests
# ---------------------------------------------------------------------------


class TestTagSplitter:
    """Unit tests for the TagSplitter class."""

    def _make_splitter(self, llm_response: str) -> TagSplitter:
        """Create a TagSplitter with a mock LLM that returns the given response."""
        llm = MagicMock()
        llm.complete.return_value = llm_response
        config = TagSplittingConfig(enabled=True)
        return TagSplitter(llm=llm, config=config)

    def _make_turn_contents(self, turn_nums: list[int]) -> list[tuple[int, str]]:
        """Create turn contents for the given turn numbers."""
        return [(n, f"User message for turn {n}") for n in turn_nums]

    def test_split_produces_valid_groups(self):
        """Mock LLM returns valid split JSON — verify groups are parsed correctly."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "resy-troubleshooting": [0, 1, 5],
                "browser-debugging": [2, 3, 4],
            },
        })
        splitter = self._make_splitter(response)
        turn_contents = self._make_turn_contents([0, 1, 2, 3, 4, 5])

        result = splitter.split(
            tag="troubleshooting",
            turn_contents=turn_contents,
            existing_tags={"database", "fitness"},
            total_turns=20,
        )

        assert result.splittable is True
        assert "resy-troubleshooting" in result.groups
        assert "browser-debugging" in result.groups
        assert result.groups["resy-troubleshooting"] == [0, 1, 5]
        assert result.groups["browser-debugging"] == [2, 3, 4]

    def test_unsplittable_returns_false(self):
        """Mock LLM says not splittable — verify result."""
        response = json.dumps({
            "splittable": False,
            "reason": "All turns discuss the same database troubleshooting topic",
        })
        splitter = self._make_splitter(response)
        turn_contents = self._make_turn_contents([0, 1, 2])

        result = splitter.split(
            tag="troubleshooting",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=10,
        )

        assert result.splittable is False
        assert "same" in result.reason.lower() or "database" in result.reason.lower()

    def test_malformed_response_fallback(self):
        """Parse error → treated as unsplittable."""
        splitter = self._make_splitter("This is not JSON at all!")
        turn_contents = self._make_turn_contents([0, 1])

        result = splitter.split(
            tag="troubleshooting",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=10,
        )

        assert result.splittable is False
        assert "parse" in result.reason.lower() or "json" in result.reason.lower()

    def test_single_group_treated_as_unsplittable(self):
        """1 group → fallback to unsplittable."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "only-group": [0, 1, 2],
            },
        })
        splitter = self._make_splitter(response)
        turn_contents = self._make_turn_contents([0, 1, 2])

        result = splitter.split(
            tag="troubleshooting",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=10,
        )

        assert result.splittable is False
        assert "fewer than 2" in result.reason.lower()

    def test_turn_text_truncated_in_prompt(self):
        """Verify long text is truncated to 200 chars in the prompt."""
        llm = MagicMock()
        llm.complete.return_value = json.dumps({
            "splittable": False,
            "reason": "uniform topic",
        })
        config = TagSplittingConfig(enabled=True)
        splitter = TagSplitter(llm=llm, config=config)

        long_text = "x" * 500
        turn_contents = [(0, long_text)]

        splitter.split(
            tag="test",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=5,
        )

        # Check that the prompt sent to LLM has truncated text
        call_args = llm.complete.call_args
        user_prompt = call_args.kwargs.get("user") or call_args[1] if len(call_args) > 1 else call_args.kwargs.get("user", "")
        # The turn text in the prompt should be at most 200 chars
        assert "x" * 201 not in user_prompt

    def test_collision_detection_appends_suffix(self):
        """If a proposed new tag already exists, a suffix is appended."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "database": [0, 1],  # collides with existing
                "api-errors": [2, 3],
            },
        })
        splitter = self._make_splitter(response)
        turn_contents = self._make_turn_contents([0, 1, 2, 3])

        result = splitter.split(
            tag="troubleshooting",
            turn_contents=turn_contents,
            existing_tags={"database", "fitness"},
            total_turns=10,
        )

        assert result.splittable is True
        # "database" should have been renamed to avoid collision
        tag_names = set(result.groups.keys())
        assert "database" not in tag_names
        assert any("database" in t for t in tag_names)  # e.g. "database-split"

    def test_llm_error_returns_unsplittable(self):
        """If the LLM call raises an exception, result is unsplittable."""
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("API timeout")
        config = TagSplittingConfig(enabled=True)
        splitter = TagSplitter(llm=llm, config=config)
        turn_contents = self._make_turn_contents([0, 1])

        result = splitter.split(
            tag="test",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=5,
        )

        assert result.splittable is False
        assert "LLM error" in result.reason

    def test_markdown_fences_stripped(self):
        """Response wrapped in markdown fences should still parse."""
        inner = json.dumps({
            "splittable": True,
            "groups": {
                "group-a": [0, 1],
                "group-b": [2, 3],
            },
        })
        response = f"```json\n{inner}\n```"
        splitter = self._make_splitter(response)
        turn_contents = self._make_turn_contents([0, 1, 2, 3])

        result = splitter.split(
            tag="test",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=10,
        )

        assert result.splittable is True
        assert len(result.groups) == 2

    @pytest.mark.regression("BUG-011")
    def test_string_turn_numbers_with_t_prefix(self):
        """LLM returns turn numbers as 'T9' strings — should parse correctly."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "browser-debugging": ["T0", "T1"],
                "api-errors": ["T2", "T3"],
            },
        })
        splitter = self._make_splitter(response)
        turn_contents = self._make_turn_contents([0, 1, 2, 3])

        result = splitter.split(
            tag="troubleshooting",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=10,
        )

        assert result.splittable is True
        assert result.groups["browser-debugging"] == [0, 1]
        assert result.groups["api-errors"] == [2, 3]

    @pytest.mark.regression("BUG-011")
    def test_string_turn_numbers_plain_digits(self):
        """LLM returns turn numbers as string digits '9' — should parse correctly."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "group-a": ["0", "1"],
                "group-b": ["2", "3"],
            },
        })
        splitter = self._make_splitter(response)
        turn_contents = self._make_turn_contents([0, 1, 2, 3])

        result = splitter.split(
            tag="test",
            turn_contents=turn_contents,
            existing_tags=set(),
            total_turns=10,
        )

        assert result.splittable is True
        assert len(result.groups) == 2


# ---------------------------------------------------------------------------
# TurnTagIndex additions
# ---------------------------------------------------------------------------


class TestTurnTagIndexSplitting:
    """Tests for replace_tag() and get_tag_counts() methods."""

    def _make_index(self) -> TurnTagIndex:
        """Create an index with known entries for testing."""
        index = TurnTagIndex()
        index.append(TurnTagEntry(
            turn_number=0, message_hash="h0",
            tags=["troubleshooting", "database"], primary_tag="troubleshooting",
        ))
        index.append(TurnTagEntry(
            turn_number=1, message_hash="h1",
            tags=["troubleshooting", "api"], primary_tag="troubleshooting",
        ))
        index.append(TurnTagEntry(
            turn_number=2, message_hash="h2",
            tags=["database", "optimization"], primary_tag="database",
        ))
        index.append(TurnTagEntry(
            turn_number=3, message_hash="h3",
            tags=["troubleshooting", "browser"], primary_tag="browser",
        ))
        return index

    def test_replace_tag_updates_entries(self):
        """replace_tag removes old tag and adds new sub-tags."""
        index = self._make_index()

        modified = index.replace_tag("troubleshooting", {
            0: ["db-troubleshooting"],
            1: ["api-troubleshooting"],
            3: ["browser-troubleshooting"],
        })

        assert modified == 3
        assert "troubleshooting" not in index.entries[0].tags
        assert "db-troubleshooting" in index.entries[0].tags
        assert "database" in index.entries[0].tags  # other tags preserved

        assert "api-troubleshooting" in index.entries[1].tags
        assert "browser-troubleshooting" in index.entries[3].tags

        # Entry 2 should be unchanged (didn't have "troubleshooting")
        assert index.entries[2].tags == ["database", "optimization"]

    def test_replace_tag_updates_primary(self):
        """When primary_tag matches old_tag, it's updated to first new tag."""
        index = self._make_index()

        index.replace_tag("troubleshooting", {
            0: ["db-troubleshooting"],
            1: ["api-troubleshooting"],
            3: ["browser-troubleshooting"],
        })

        assert index.entries[0].primary_tag == "db-troubleshooting"
        assert index.entries[1].primary_tag == "api-troubleshooting"
        # Entry 3: primary was "browser", not "troubleshooting" → unchanged
        assert index.entries[3].primary_tag == "browser"

    def test_replace_tag_no_mapping_skips(self):
        """Entries with the old tag but no mapping in turn_to_new_tags are not modified."""
        index = self._make_index()

        # Only provide mapping for turn 0
        modified = index.replace_tag("troubleshooting", {
            0: ["db-troubleshooting"],
        })

        assert modified == 1
        # Turns 1 and 3 still have "troubleshooting"
        assert "troubleshooting" in index.entries[1].tags
        assert "troubleshooting" in index.entries[3].tags

    def test_get_tag_counts(self):
        """get_tag_counts returns accurate counts for all tags."""
        index = self._make_index()
        counts = index.get_tag_counts()

        assert counts["troubleshooting"] == 3  # turns 0, 1, 3
        assert counts["database"] == 2  # turns 0, 2
        assert counts["api"] == 1
        assert counts["browser"] == 1
        assert counts["optimization"] == 1


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class TestEngineTagSplitting:
    """Integration tests for tag splitting in the engine."""

    def _make_engine(self, tmp_path, llm_response: str | None = None, splitting_enabled: bool = True):
        """Create an engine with tag splitting configured."""
        from virtual_context.config import load_config
        from virtual_context.engine import VirtualContextEngine

        db_path = str(tmp_path / "store.db")
        config_dict = {
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
            "tag_generator": {
                "type": "keyword",
                "tag_splitting": {
                    "enabled": splitting_enabled,
                    "frequency_threshold": 5,
                    "frequency_pct_threshold": 0.1,
                },
            },
        }

        config = load_config(config_dict=config_dict)
        engine = VirtualContextEngine(config=config)

        if llm_response is not None and splitting_enabled:
            # Inject a mock tag splitter
            from virtual_context.core.tag_splitter import TagSplitter

            mock_llm = MagicMock()
            mock_llm.complete.return_value = llm_response
            engine._tag_splitter = TagSplitter(
                llm=mock_llm,
                config=config.tag_generator.tag_splitting,
            )

        return engine

    def _populate_index(self, engine, tag: str, count: int, other_tags: list[str] | None = None):
        """Add entries to the turn tag index with the given tag."""
        for i in range(count):
            tags = [tag]
            if other_tags:
                tags.extend(other_tags)
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i,
                message_hash=f"h{i:04d}",
                tags=tags,
                primary_tag=tag,
            ))

    def _make_history(self, count: int) -> list[Message]:
        """Create a fake conversation history with count turns."""
        history = []
        for i in range(count):
            history.append(Message(role="user", content=f"User message {i} about topic"))
            history.append(Message(role="assistant", content=f"Assistant response {i}"))
        return history

    def test_broad_tag_triggers_split(self, tmp_path):
        """A tag on 20/50 turns should trigger splitting."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "resy-troubleshooting": [0, 1, 2, 3, 4],
                "browser-debugging": [5, 6, 7, 8, 9],
                "api-errors": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            },
        })
        engine = self._make_engine(tmp_path, llm_response=response)
        self._populate_index(engine, "troubleshooting", 20)
        # Add more entries with diverse tags (each unique, so none crosses threshold)
        for i in range(20, 50):
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i:04d}",
                tags=[f"topic-{i}"], primary_tag=f"topic-{i}",
            ))

        history = self._make_history(50)
        result = engine._check_and_split_broad_tags(history)

        assert result is not None
        assert result.splittable is True
        assert "troubleshooting" in engine._split_processed_tags

    def test_below_threshold_no_split(self, tmp_path):
        """10/100 turns (10%) below 15 threshold — no split triggered."""
        engine = self._make_engine(tmp_path, llm_response=None)
        # 10 turns of "troubleshooting" — below frequency_threshold of 15
        self._populate_index(engine, "troubleshooting", 10)
        for i in range(10, 100):
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i:04d}",
                tags=["other"], primary_tag="other",
            ))

        history = self._make_history(100)
        result = engine._check_and_split_broad_tags(history)

        assert result is None

    def test_split_updates_turn_tag_index(self, tmp_path):
        """After split, entries should have new sub-tags and old tag removed."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "db-troubleshooting": [0, 1, 2],
                "api-troubleshooting": [3, 4],
            },
        })
        engine = self._make_engine(tmp_path, llm_response=response)
        self._populate_index(engine, "troubleshooting", 5)
        # Need enough other entries with diverse tags (each unique)
        for i in range(5, 50):
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i:04d}",
                tags=[f"topic-{i}"], primary_tag=f"topic-{i}",
            ))

        history = self._make_history(50)
        engine._check_and_split_broad_tags(history)

        # Verify old tag removed from split turns
        for i in range(5):
            entry = engine._turn_tag_index.get_tags_for_turn(i)
            assert "troubleshooting" not in entry.tags

        # Verify new tags present
        assert "db-troubleshooting" in engine._turn_tag_index.entries[0].tags
        assert "api-troubleshooting" in engine._turn_tag_index.entries[3].tags

    def test_unsplittable_does_not_modify_index(self, tmp_path):
        """An unsplittable result should not modify the TurnTagIndex entries."""
        response = json.dumps({
            "splittable": False,
            "reason": "All turns are about the same topic",
        })
        engine = self._make_engine(tmp_path, llm_response=response)
        self._populate_index(engine, "cycle-tracking", 10)
        # Use diverse tags so "cycle-tracking" is the highest-frequency candidate
        for i in range(10, 50):
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i:04d}",
                tags=[f"topic-{i}"], primary_tag=f"topic-{i}",
            ))

        history = self._make_history(50)
        result = engine._check_and_split_broad_tags(history)

        assert result is not None
        assert result.splittable is False
        # Tag should still be in entries
        assert "cycle-tracking" in engine._turn_tag_index.entries[0].tags
        # But should be marked as processed
        assert "cycle-tracking" in engine._split_processed_tags

    def test_split_not_retriggered(self, tmp_path):
        """Once processed, a tag is not split again."""
        response = json.dumps({
            "splittable": True,
            "groups": {
                "a-troubleshooting": [0, 1, 2],
                "b-troubleshooting": [3, 4],
            },
        })
        engine = self._make_engine(tmp_path, llm_response=response)
        self._populate_index(engine, "troubleshooting", 5)
        # Use diverse tags so only "troubleshooting" crosses threshold
        for i in range(5, 50):
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i:04d}",
                tags=[f"topic-{i}"], primary_tag=f"topic-{i}",
            ))

        history = self._make_history(50)

        # First call: should split
        result1 = engine._check_and_split_broad_tags(history)
        assert result1 is not None

        # Second call: should skip (already processed)
        result2 = engine._check_and_split_broad_tags(history)
        assert result2 is None

    def test_general_tag_excluded(self, tmp_path):
        """_general should never be a splitting candidate."""
        engine = self._make_engine(tmp_path, llm_response=None)
        self._populate_index(engine, "_general", 30)

        history = self._make_history(30)
        result = engine._check_and_split_broad_tags(history)

        assert result is None

    @pytest.mark.regression("BUG-012")
    def test_collect_turn_text_with_preamble_messages(self, tmp_path):
        """_collect_turn_text should find real content even when history has
        extra messages (MemOS preambles) that break turn_number * 2 indexing.

        In proxy mode, OpenClaw injects preamble user messages like
        '# Role\\nYou are an intelligent assistant...' before the real user
        content, so the history has extra user messages that shift indices.
        """
        engine = self._make_engine(tmp_path, llm_response=None, splitting_enabled=False)

        # Add index entries for turns 0, 1, 2
        for i in range(3):
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i}",
                tags=["troubleshooting"], primary_tag="troubleshooting",
            ))

        # Build a history with MemOS preamble messages injected:
        # The real content is in the preamble-wrapped user messages,
        # but turn_number * 2 would point to the wrong messages.
        history = [
            # Turn 0: preamble + real content are separate user messages
            Message(role="user", content="# Role\nYou are an intelligent assistant..."),
            Message(role="user", content="How do I fix this error?"),
            Message(role="assistant", content="Let me help you debug."),
            # Turn 1: same pattern
            Message(role="user", content="# Role\nYou are an intelligent assistant..."),
            Message(role="user", content="The browser tab disconnected"),
            Message(role="assistant", content="Try reconnecting."),
            # Turn 2: same pattern
            Message(role="user", content="# Role\nYou are an intelligent assistant..."),
            Message(role="user", content="OpenTable is blocking the sandbox"),
            Message(role="assistant", content="Let me try a different approach."),
        ]

        turn_contents = engine._collect_turn_text("troubleshooting", history)

        # All 3 turns should have non-empty real content (not preambles).
        # BUG-012: With turn_number * 2 indexing, turn 0 gets the preamble,
        # turn 1 gets "How do I fix this error?" (wrong turn), turn 2 gets
        # another preamble. At least some will be wrong.
        assert len(turn_contents) == 3
        texts = [text for _, text in turn_contents]
        # Every collected text should contain actual user content, not preambles
        for text in texts:
            assert text.strip(), f"Empty text collected for turn"
            assert "# Role" not in text, f"Preamble content leaked into turn text: {text[:50]}"


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestSplitPersistence:
    """Tests for split_processed_tags persistence in snapshots."""

    def test_split_persisted_in_sqlite(self, tmp_path):
        """split_processed_tags survive save/load in SQLite."""
        from virtual_context.storage.sqlite import SQLiteStore

        db = tmp_path / "test.db"
        store = SQLiteStore(db_path=db)

        snap = EngineStateSnapshot(
            session_id="test-session",
            compacted_through=4,
            turn_tag_entries=[
                TurnTagEntry(
                    turn_number=0, message_hash="h0",
                    tags=["db-troubleshooting"], primary_tag="db-troubleshooting",
                    timestamp=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                ),
            ],
            turn_count=1,
            split_processed_tags=["troubleshooting", "debugging"],
        )
        store.save_engine_state(snap)

        loaded = store.load_engine_state("test-session")
        assert loaded is not None
        assert loaded.split_processed_tags == ["troubleshooting", "debugging"]

    def test_split_persisted_in_filesystem(self, tmp_path):
        """split_processed_tags survive save/load in Filesystem."""
        from virtual_context.storage.filesystem import FilesystemStore

        store = FilesystemStore(root=tmp_path / "store")

        snap = EngineStateSnapshot(
            session_id="test-session",
            compacted_through=4,
            turn_tag_entries=[
                TurnTagEntry(
                    turn_number=0, message_hash="h0",
                    tags=["db-troubleshooting"], primary_tag="db-troubleshooting",
                    timestamp=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                ),
            ],
            turn_count=1,
            split_processed_tags=["troubleshooting", "debugging"],
        )
        store.save_engine_state(snap)

        loaded = store.load_engine_state("test-session")
        assert loaded is not None
        assert loaded.split_processed_tags == ["troubleshooting", "debugging"]

    def test_backward_compat_no_split_field(self, tmp_path):
        """Loading old state without split_processed_tags defaults to empty list."""
        from virtual_context.storage.sqlite import SQLiteStore

        db = tmp_path / "test.db"
        store = SQLiteStore(db_path=db)

        # Save in old format (plain list, not dict)
        conn = store._get_conn()
        entries_json = json.dumps([
            {
                "turn_number": 0,
                "message_hash": "h0",
                "tags": ["test"],
                "primary_tag": "test",
                "timestamp": "2026-01-15T10:00:00+00:00",
            },
        ])
        conn.execute(
            """INSERT OR REPLACE INTO engine_state
            (session_id, compacted_through, turn_count, turn_tag_entries, saved_at)
            VALUES (?, ?, ?, ?, ?)""",
            ("old-session", 0, 1, entries_json, "2026-01-15T10:00:00+00:00"),
        )
        conn.commit()

        loaded = store.load_engine_state("old-session")
        assert loaded is not None
        assert loaded.split_processed_tags == []
        assert len(loaded.turn_tag_entries) == 1
