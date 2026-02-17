"""Tests for context bleed prevention (BUG-010).

The embedding similarity gate in _get_recent_context() prevents stale
context from a previous topic block from being fed to the tagger when
the topic shifts.  Compares the current turn's combined text against the
most recent pair in the collected context using cosine similarity.
"""

import pytest

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message, TurnTagEntry


def _make_history(*pairs):
    """Build a flat message list from (user_text, asst_text) pairs."""
    msgs = []
    for user_text, asst_text in pairs:
        msgs.append(Message(role="user", content=user_text))
        msgs.append(Message(role="assistant", content=asst_text))
    return msgs


def _mock_embed_fn(texts):
    """Deterministic embedding based on keyword detection.

    transit → [1,0,0], identity → [0,1,0], database → [0,0,1].
    """
    result = []
    for text in texts:
        t = text.lower()
        if any(w in t for w in ["transit", "train", "bus", "station", "penn", "river edge"]):
            result.append([1.0, 0.0, 0.0])
        elif any(w in t for w in ["sania", "love", "remember", "care", "means a lot"]):
            result.append([0.0, 1.0, 0.0])
        elif any(w in t for w in ["index", "database", "column", "migration", "b-tree", "brin"]):
            result.append([0.0, 0.0, 1.0])
        elif any(w in t for w in ["rush hour", "faster", "cost difference", "saves you"]):
            result.append([0.8, 0.2, 0.0])  # transit-adjacent
        elif any(w in t for w in ["of course", "she means"]):
            result.append([0.1, 0.9, 0.0])  # identity-adjacent
        else:
            result.append([0.33, 0.33, 0.34])  # generic
    return result


def _make_engine(threshold=0.1):
    """Create engine with keyword tagger and mock embeddings."""
    config = load_config(config_dict={
        "context_window": 10000,
        "tag_generator": {
            "type": "keyword",
            "context_lookback_pairs": 5,
            "context_bleed_threshold": threshold,
            "keyword_fallback": {"tag_keywords": {}},
        },
    })
    engine = VirtualContextEngine(config=config)
    engine._embed_fn = _mock_embed_fn
    return engine


class TestContextBleedGate:
    """Embedding similarity gate prevents stale context on topic shifts."""

    @pytest.mark.regression("BUG-010")
    def test_topic_shift_strips_context(self):
        """Identity message after transit block should NOT get transit context."""
        engine = _make_engine()
        history = _make_history(
            (
                "if I need to go to NYC from river edge on NJ transit train, what time?",
                "The NJ Transit train from River Edge station runs every 30 minutes.",
            ),
            (
                "what about the bus schedule from there?",
                "The 165 bus from River Edge runs every 20 minutes.",
            ),
            (
                "what do you love bast?",
                "I love helping you navigate complex problems.",
            ),
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=0, message_hash="a", tags=["transit-schedule"], primary_tag="transit-schedule")
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=1, message_hash="b", tags=["transit-schedule"], primary_tag="transit-schedule")
        )

        ctx = engine._get_recent_context(
            history, 5,
            current_text="what do you love bast? I love helping you navigate complex problems.",
        )
        assert ctx is None, "Transit context should be blocked for identity message"

    @pytest.mark.regression("BUG-010")
    def test_continuation_keeps_context(self):
        """Transit follow-up should keep transit context."""
        engine = _make_engine()
        history = _make_history(
            (
                "if I need to go to NYC from river edge on NJ transit train, what time?",
                "The NJ Transit train from River Edge station runs every 30 minutes.",
            ),
            (
                "what about the bus schedule from there?",
                "The 165 bus from River Edge runs every 20 minutes.",
            ),
            (
                "which is faster during rush hour?",
                "The train is faster, about 45 minutes door to door.",
            ),
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=0, message_hash="a", tags=["transit-schedule"], primary_tag="transit-schedule")
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=1, message_hash="b", tags=["transit-schedule"], primary_tag="transit-schedule")
        )

        ctx = engine._get_recent_context(
            history, 5,
            current_text="which is faster during rush hour? The train is faster, about 45 minutes door to door.",
        )
        assert ctx is not None, "Transit context should be kept for transit follow-up"

    @pytest.mark.regression("BUG-010")
    def test_short_msg_after_shift_keeps_new_topic(self):
        """'of course' after identity turn should keep identity context, not transit."""
        engine = _make_engine()
        history = _make_history(
            (
                "NJ transit schedule from river edge",
                "Trains run every 30 minutes from River Edge station.",
            ),
            (
                "what about the bus?",
                "The 165 bus runs every 20 minutes.",
            ),
            (
                "do you remember sania?",
                "Yes, Sania is someone you care about deeply.",
            ),
            (
                "of course",
                "It's clear she means a lot to you.",
            ),
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=0, message_hash="a", tags=["transit-schedule"], primary_tag="transit-schedule")
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=1, message_hash="b", tags=["transit-schedule"], primary_tag="transit-schedule")
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=2, message_hash="c", tags=["identity", "relationship"], primary_tag="identity")
        )

        # Most recent context pair is T2 (identity), which is relevant to T3
        ctx = engine._get_recent_context(
            history, 5,
            current_text="of course. It's clear she means a lot to you.",
        )
        assert ctx is not None, "Identity context should be kept for 'of course' after identity turn"

    @pytest.mark.regression("BUG-010")
    def test_low_overlap_continuation_keeps_context(self):
        """'cost difference over a month?' should keep transit context."""
        engine = _make_engine()
        history = _make_history(
            (
                "if I need to go to NYC from river edge on NJ transit train, what time?",
                "The NJ Transit train from River Edge station runs every 30 minutes.",
            ),
            (
                "what about the bus schedule from there?",
                "Bus costs $3.50 per ride versus $5.75 for the train.",
            ),
            (
                "what about the cost difference over a month?",
                "At 20 workdays, the bus saves you $45/month compared to the train.",
            ),
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=0, message_hash="a", tags=["transit-schedule"], primary_tag="transit-schedule")
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=1, message_hash="b", tags=["transit-schedule"], primary_tag="transit-schedule")
        )

        ctx = engine._get_recent_context(
            history, 5,
            current_text="what about the cost difference over a month? At 20 workdays, the bus saves you $45/month compared to the train.",
        )
        assert ctx is not None, "Transit context should be kept for low-overlap continuation"

    @pytest.mark.regression("BUG-010")
    def test_single_word_continuation_keeps_context(self):
        """'yes' continuing database discussion should keep context."""
        engine = _make_engine()
        history = _make_history(
            (
                "should I add an index on the email column?",
                "Yes, a B-tree index on email would speed up lookups significantly.",
            ),
            (
                "what about the created_at column?",
                "A BRIN index would be more space-efficient for timestamp columns.",
            ),
            (
                "yes",
                "I'll add both indexes to the migration.",
            ),
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=0, message_hash="a", tags=["database", "indexing"], primary_tag="database")
        )
        engine._turn_tag_index.append(
            TurnTagEntry(turn_number=1, message_hash="b", tags=["database", "indexing"], primary_tag="database")
        )

        ctx = engine._get_recent_context(
            history, 5,
            current_text="yes. I'll add both indexes to the migration.",
        )
        assert ctx is not None, "Database context should be kept for single-word continuation"


class TestContextBleedConfig:
    """Gate configuration edge cases."""

    def test_gate_disabled_when_threshold_zero(self):
        """When threshold is 0, gate should not block anything."""
        engine = _make_engine(threshold=0.0)
        history = _make_history(
            (
                "NJ transit from River Edge, what time?",
                "Train runs every 30 minutes from River Edge station.",
            ),
            (
                "what do you love bast?",
                "I love helping you navigate complex problems.",
            ),
        )

        ctx = engine._get_recent_context(
            history, 5,
            current_text="what do you love bast? I love helping you navigate complex problems.",
        )
        assert ctx is not None, "Context should not be blocked when threshold is 0"

    def test_graceful_degradation_no_embeddings(self):
        """When no embed function is available, gate should pass through."""
        engine = _make_engine()
        engine._embed_fn = None  # simulate no sentence-transformers
        history = _make_history(
            (
                "NJ transit from River Edge, what time?",
                "Train runs every 30 minutes from River Edge station.",
            ),
            (
                "what do you love bast?",
                "I love helping you navigate complex problems.",
            ),
        )

        ctx = engine._get_recent_context(
            history, 5,
            current_text="what do you love bast? I love helping you navigate complex problems.",
        )
        assert ctx is not None, "Context should not be blocked when embeddings unavailable"

    def test_no_current_text_skips_gate(self):
        """Without current_text, gate should not apply (backward compat)."""
        engine = _make_engine()
        history = _make_history(
            (
                "NJ transit from River Edge, what time?",
                "Train runs every 30 minutes from River Edge station.",
            ),
            (
                "what do you love bast?",
                "I love helping you navigate complex problems.",
            ),
        )

        ctx = engine._get_recent_context(history, 5)
        assert ctx is not None, "Context should not be blocked without current_text"
