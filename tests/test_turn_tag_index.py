"""Tests for TurnTagIndex."""
import pytest
from datetime import datetime, timezone, timedelta
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import TurnTagEntry


class TestTurnTagIndex:
    def test_append_and_retrieve(self):
        index = TurnTagIndex()
        entry = TurnTagEntry(turn_number=0, message_hash="abc123", tags=["database", "api"], primary_tag="database")
        index.append(entry)
        assert len(index.entries) == 1
        assert index.get_tags_for_turn(0) == entry

    def test_get_tags_for_turn_not_found(self):
        index = TurnTagIndex()
        assert index.get_tags_for_turn(99) is None

    def test_get_active_tags(self):
        index = TurnTagIndex()
        index.append(TurnTagEntry(turn_number=0, message_hash="a", tags=["database"], primary_tag="database"))
        index.append(TurnTagEntry(turn_number=1, message_hash="b", tags=["api"], primary_tag="api"))
        index.append(TurnTagEntry(turn_number=2, message_hash="c", tags=["frontend"], primary_tag="frontend"))
        active = index.get_active_tags(lookback=2)
        assert active == {"api", "frontend"}

    def test_get_active_tags_lookback_exceeds_entries(self):
        index = TurnTagIndex()
        index.append(TurnTagEntry(turn_number=0, message_hash="a", tags=["database"], primary_tag="database"))
        active = index.get_active_tags(lookback=10)
        assert active == {"database"}

    def test_get_active_tags_empty(self):
        index = TurnTagIndex()
        assert index.get_active_tags() == set()

    def test_get_tag_velocity(self):
        index = TurnTagIndex()
        now = datetime.now(timezone.utc)
        for i in range(5):
            index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i}", tags=["database"],
                primary_tag="database",
                timestamp=now - timedelta(hours=i),
            ))
        velocity = index.get_tag_velocity("database", window_hours=72)
        assert velocity > 0

    def test_get_tag_velocity_no_entries(self):
        index = TurnTagIndex()
        assert index.get_tag_velocity("nonexistent") == 0.0

    def test_get_tag_velocity_outside_window(self):
        index = TurnTagIndex()
        old = datetime.now(timezone.utc) - timedelta(hours=100)
        index.append(TurnTagEntry(turn_number=0, message_hash="a", tags=["database"], primary_tag="database", timestamp=old))
        assert index.get_tag_velocity("database", window_hours=72) == 0.0

    def test_compute_cover_set_covers_all_turns(self):
        index = TurnTagIndex()
        index.append(TurnTagEntry(turn_number=0, message_hash="a", tags=["database", "api"], primary_tag="database"))
        index.append(TurnTagEntry(turn_number=1, message_hash="b", tags=["api", "auth"], primary_tag="api"))
        index.append(TurnTagEntry(turn_number=2, message_hash="c", tags=["frontend"], primary_tag="frontend"))
        cover = index.compute_cover_set()
        # Cover must touch all turns
        covered_turns = set()
        for entry in index.entries:
            if any(t in cover for t in entry.tags):
                covered_turns.add(entry.turn_number)
        assert covered_turns == {0, 1, 2}

    @pytest.mark.regression("PROXY-002")
    def test_compute_cover_set_excludes_general(self):
        index = TurnTagIndex()
        index.append(TurnTagEntry(turn_number=0, message_hash="a", tags=["_general", "database"], primary_tag="database"))
        index.append(TurnTagEntry(turn_number=1, message_hash="b", tags=["_general"], primary_tag="_general"))
        cover = index.compute_cover_set()
        assert "_general" not in cover

    def test_compute_cover_set_greedy_order(self):
        """Most-covering tag should come first."""
        index = TurnTagIndex()
        # "api" covers turns 0,1,2 — "frontend" covers only turn 3
        index.append(TurnTagEntry(turn_number=0, message_hash="a", tags=["api"], primary_tag="api"))
        index.append(TurnTagEntry(turn_number=1, message_hash="b", tags=["api"], primary_tag="api"))
        index.append(TurnTagEntry(turn_number=2, message_hash="c", tags=["api"], primary_tag="api"))
        index.append(TurnTagEntry(turn_number=3, message_hash="d", tags=["frontend"], primary_tag="frontend"))
        cover = index.compute_cover_set()
        assert cover[0] == "api"  # Most covering first
        assert "frontend" in cover

    def test_compute_cover_set_empty(self):
        index = TurnTagIndex()
        assert index.compute_cover_set() == []

    @pytest.mark.regression("PROXY-002")
    def test_compute_cover_set_only_general(self):
        """When all turns have only _general, cover should be empty."""
        index = TurnTagIndex()
        index.append(TurnTagEntry(turn_number=0, message_hash="a", tags=["_general"], primary_tag="_general"))
        cover = index.compute_cover_set()
        assert cover == []
