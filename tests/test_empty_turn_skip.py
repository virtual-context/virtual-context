"""BUG-013: Empty turn skipping — tool_use/tool_result pairs with no text content
should not produce TurnTagIndex entries or tagger calls."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tests.conftest import MockTagGenerator
from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message, TagResult


def _make_engine(tmp_path: Path, tagger: MockTagGenerator | None = None):
    """Create a minimal engine with a mock tagger."""
    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"type": "sqlite", "path": str(tmp_path / "vc.db")},
        "tag_generator": {"type": "keyword"},
    })
    engine = VirtualContextEngine(config=config)
    if tagger:
        engine._tag_generator = tagger
    return engine


class TestIngestHistorySkipsEmptyTurns:
    """ingest_history() should skip tagging when both user and assistant are empty."""

    @pytest.mark.regression("BUG-013")
    def test_empty_pair_not_tagged(self, tmp_path):
        """A turn where both user and assistant content are empty should produce
        no TurnTagIndex entry and no tagger call."""
        tagger = MockTagGenerator(default_tag="phantom")
        engine = _make_engine(tmp_path, tagger)

        # Simulate: real turn, empty tool-use turn, real turn
        pairs = [
            Message(role="user", content="Tell me about cats"),
            Message(role="assistant", content="Cats are great pets"),
            Message(role="user", content=""),      # tool_result (empty)
            Message(role="assistant", content=""),  # tool_use (empty)
            Message(role="user", content="What about dogs?"),
            Message(role="assistant", content="Dogs are loyal companions"),
        ]

        ingested = engine.ingest_history(pairs)

        # Should only have 2 entries (skipping the empty pair)
        assert len(engine._turn_tag_index.entries) == 2
        # Tagger should only have been called twice
        assert len(tagger.calls) == 2
        # Turn numbers should preserve pair index mapping
        assert engine._turn_tag_index.entries[0].turn_number == 0
        assert engine._turn_tag_index.entries[1].turn_number == 2

    @pytest.mark.regression("BUG-013")
    def test_whitespace_only_pair_not_tagged(self, tmp_path):
        """Whitespace-only content should also be treated as empty."""
        tagger = MockTagGenerator(default_tag="phantom")
        engine = _make_engine(tmp_path, tagger)

        pairs = [
            Message(role="user", content="Real content here"),
            Message(role="assistant", content="Real response"),
            Message(role="user", content="   "),    # whitespace only
            Message(role="assistant", content="\n"),  # whitespace only
        ]

        engine.ingest_history(pairs)

        assert len(engine._turn_tag_index.entries) == 1
        assert len(tagger.calls) == 1

    @pytest.mark.regression("BUG-013")
    def test_empty_user_nonempty_assistant_still_tagged(self, tmp_path):
        """When user is empty but assistant has content (e.g., multi-step tool
        chain continuation), the turn should still be tagged."""
        tagger = MockTagGenerator(default_tag="continued")
        engine = _make_engine(tmp_path, tagger)

        pairs = [
            Message(role="user", content=""),
            Message(role="assistant", content="Here are the search results for your query"),
        ]

        engine.ingest_history(pairs)

        # Should still be tagged (assistant has content)
        assert len(engine._turn_tag_index.entries) == 1
        assert len(tagger.calls) == 1

    @pytest.mark.regression("BUG-013")
    def test_nonempty_user_empty_assistant_still_tagged(self, tmp_path):
        """When user has content but assistant is empty (e.g., tool_use only
        response), the turn should still be tagged."""
        tagger = MockTagGenerator(default_tag="tool-init")
        engine = _make_engine(tmp_path, tagger)

        pairs = [
            Message(role="user", content="Search for the latest news"),
            Message(role="assistant", content=""),
        ]

        engine.ingest_history(pairs)

        assert len(engine._turn_tag_index.entries) == 1
        assert len(tagger.calls) == 1

    @pytest.mark.regression("BUG-013")
    def test_multiple_consecutive_empty_turns_skipped(self, tmp_path):
        """Multiple consecutive empty turns (chained tool calls) should all be skipped."""
        tagger = MockTagGenerator(default_tag="real")
        engine = _make_engine(tmp_path, tagger)

        pairs = [
            Message(role="user", content="Do a complex analysis"),
            Message(role="assistant", content="Starting analysis"),
            # 3 consecutive empty tool turns
            Message(role="user", content=""), Message(role="assistant", content=""),
            Message(role="user", content=""), Message(role="assistant", content=""),
            Message(role="user", content=""), Message(role="assistant", content=""),
            # Final real turn
            Message(role="user", content="Thanks for the results"),
            Message(role="assistant", content="You're welcome"),
        ]

        engine.ingest_history(pairs)

        assert len(engine._turn_tag_index.entries) == 2
        assert len(tagger.calls) == 2
        assert engine._turn_tag_index.entries[0].turn_number == 0
        assert engine._turn_tag_index.entries[1].turn_number == 4

    @pytest.mark.regression("BUG-013")
    def test_ingested_count_excludes_skipped(self, tmp_path):
        """Return value of ingest_history should count only tagged turns."""
        tagger = MockTagGenerator(default_tag="real")
        engine = _make_engine(tmp_path, tagger)

        pairs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content=""),
            Message(role="assistant", content=""),
            Message(role="user", content="Bye"),
            Message(role="assistant", content="Goodbye"),
        ]

        ingested = engine.ingest_history(pairs)

        # Only 2 turns were actually tagged
        assert ingested == 2


class TestOnTurnCompleteSkipsEmptyTurns:
    """on_turn_complete() should skip tagging when latest pair is empty."""

    @pytest.mark.regression("BUG-013")
    def test_empty_latest_pair_not_tagged(self, tmp_path):
        """When the latest user+assistant pair are both empty, no TurnTagIndex
        entry should be created."""
        tagger = MockTagGenerator(default_tag="phantom")
        engine = _make_engine(tmp_path, tagger)

        history = [
            Message(role="user", content="Tell me about cats"),
            Message(role="assistant", content="Cats are great"),
            # Tool-use turn: both empty
            Message(role="user", content=""),
            Message(role="assistant", content=""),
        ]

        engine.on_turn_complete(history)

        # Only the empty turn was processed — should be skipped
        # (No entries since we only called on_turn_complete, not ingest)
        assert len(engine._turn_tag_index.entries) == 0
        assert len(tagger.calls) == 0

    @pytest.mark.regression("BUG-013")
    def test_nonempty_latest_pair_still_tagged(self, tmp_path):
        """Normal turns should still be tagged as before."""
        tagger = MockTagGenerator(default_tag="cats")
        engine = _make_engine(tmp_path, tagger)

        history = [
            Message(role="user", content="Tell me about cats"),
            Message(role="assistant", content="Cats are great pets"),
        ]

        engine.on_turn_complete(history)

        assert len(engine._turn_tag_index.entries) == 1
        assert len(tagger.calls) == 1
