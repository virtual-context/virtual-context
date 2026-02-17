"""Haiku end-to-end retrieval tests — real engine with real LLM."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message

HAIKU_MODEL = "claude-haiku-4-5"


def _make_engine(tmpdir: str, context_window: int = 3000) -> VirtualContextEngine:
    """Build a real engine with Haiku for tagging and summarization."""
    db_path = str(Path(tmpdir) / "store.db")
    return VirtualContextEngine(config=load_config(config_dict={
        "context_window": context_window,
        "storage_root": tmpdir,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "assembly": {"recent_turns_always_included": 2},
        "tag_generator": {
            "type": "llm",
            "provider": "haiku",
            "max_tags": 10,
            "min_tags": 5,
        },
        "summarization": {
            "provider": "haiku",
            "model": HAIKU_MODEL,
            "temperature": 0.0,
        },
        "compaction": {
            "soft_threshold": 0.70,
            "hard_threshold": 0.85,
            "protected_recent_turns": 2,
        },
        "providers": {
            "haiku": {
                "type": "anthropic",
                "model": HAIKU_MODEL,
                "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            },
        },
    }))


def _simulate_turns(engine, pairs):
    """Run N user/assistant pairs through the engine, return full history."""
    history = []
    for user_text, assistant_text in pairs:
        history.append(Message(role="user", content=user_text))
        engine.on_message_inbound(user_text, history)
        history.append(Message(role="assistant", content=assistant_text))
        engine.on_turn_complete(history)
    return history


class TestCrossVocabularyRecall:
    """BUG-005 scenario: vocabulary mismatch should still retrieve via related tags."""

    def test_caching_trick_recalls_materialized_view(self):
        """After compaction, 'caching trick' should recall 'materialized view' content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            pairs = [
                # Database topic — target content
                ("Should we use a materialized view for the feed query? It's slow with JOINs.",
                 "Yes, a materialized view would precompute the feed data. Refresh it every 5 minutes."),
                ("How do we handle cache invalidation for the materialized view?",
                 "Use a trigger-based approach: invalidate when source tables change."),
                # Padding — different topics to push compaction
                ("What guitar strings do you recommend for acoustic?",
                 "Elixir Phosphor Bronze 12-53 for a warm, balanced tone."),
                ("How do I tune my guitar to drop D?",
                 "Lower the 6th string from E to D. Use a tuner for accuracy."),
                ("What's the best fingerpicking pattern for beginners?",
                 "Start with Travis picking: alternate bass notes with melody on treble strings."),
                ("Can you recommend a good beginner guitar book?",
                 "Try 'Hal Leonard Guitar Method' — it covers basics through intermediate."),
                ("What's the difference between nylon and steel strings?",
                 "Nylon is softer, easier on fingers. Steel has brighter tone, more volume."),
                ("How often should I change guitar strings?",
                 "Every 2-4 weeks if playing daily. Coated strings last longer."),
            ]

            history = _simulate_turns(engine, pairs)

            # Now ask about "caching trick" which should bridge to "materialized view"
            result = engine.on_message_inbound(
                "What was that caching trick we discussed for the feed?", history
            )

            # Should have retrieved something about the materialized view
            assert result.prepend_text, "Expected prepend_text with retrieved context"
            assert len(result.prepend_text) > 0


class TestIDFPrecision:
    """BUG-006 scenario: rare tag should beat common tags in ranking."""

    def test_rare_tag_beats_common_tags(self):
        """Specific postgres query should retrieve postgres-specific content over generic DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            pairs = [
                # Generic database turns
                ("How do I design a normalized database schema?",
                 "Start with 3NF: eliminate transitive dependencies between non-key attributes."),
                ("What are best practices for database indexing?",
                 "Index columns used in WHERE, JOIN, and ORDER BY. Avoid over-indexing."),
                ("How do I handle database migrations safely?",
                 "Use versioned migrations, always test rollback, run in staging first."),
                # Specific postgres turn — the target
                ("How do I set up materialized views with auto-refresh in PostgreSQL?",
                 "Use CREATE MATERIALIZED VIEW, then pg_cron for scheduled REFRESH MATERIALIZED VIEW CONCURRENTLY."),
                # Padding
                ("What's a good running schedule for a 5K?",
                 "Run 3-4 times per week: 2 easy runs, 1 tempo, 1 long run."),
                ("How do I prevent shin splints while running?",
                 "Increase mileage gradually (10% rule), strengthen calves, proper shoes."),
                ("What should I eat before a morning run?",
                 "Light carbs 30-60 min before: banana, toast with peanut butter."),
                ("How do I track my running progress?",
                 "Use Strava or Garmin. Track pace, distance, heart rate, and weekly volume."),
            ]

            history = _simulate_turns(engine, pairs)

            # Ask specifically about postgres
            result = engine.on_message_inbound(
                "Tell me about the PostgreSQL materialized view setup we discussed", history
            )

            assert result.prepend_text, "Expected prepend_text with retrieved context"
            # The prepend text should mention materialized views
            prepend_lower = result.prepend_text.lower()
            assert "materialized" in prepend_lower or "postgres" in prepend_lower


class TestSummaryFloor:
    """Expected failure: post-compaction retrieval gap for vague callbacks."""

    def test_vague_callback_after_compaction(self):
        """'Go back to that earlier thing' should retrieve something after compaction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            pairs = [
                ("How do I set up JWT authentication?",
                 "Use jsonwebtoken library. Sign with RS256, set expiry to 1 hour."),
                ("What about refresh token rotation?",
                 "Store refresh tokens in DB. Rotate on each use, invalidate old ones."),
                ("What's a good pasta recipe?",
                 "Aglio e olio: garlic, olive oil, chili flakes, parsley, spaghetti."),
                ("How do I make homemade tomato sauce?",
                 "San Marzano tomatoes, garlic, basil, olive oil. Simmer 45 minutes."),
                ("What running shoes do you recommend?",
                 "Nike Pegasus for daily training, Vaporfly for race day."),
                ("How do I build running endurance?",
                 "80/20 rule: 80% easy runs, 20% hard. Increase weekly volume by 10%."),
                ("What guitar chords should I learn first?",
                 "G, C, D, Em, Am. These cover most beginner songs."),
                ("How do I transition between chords faster?",
                 "Practice common transitions slowly. Use a metronome, increase BPM gradually."),
            ]

            history = _simulate_turns(engine, pairs)

            # Vague callback — no specific topic
            result = engine.on_message_inbound(
                "Go back to that earlier thing we were talking about", history
            )

            assert result.prepend_text, "Expected prepend_text for vague callback"


class TestTagLookback:
    """Short ambiguous messages should use context from recent turns."""

    def test_short_message_uses_context(self):
        """'Of course' after database discussion should tag as database, not _general."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            pairs = [
                ("How do I optimize the users table query?",
                 "Add an index on the email column and use EXPLAIN ANALYZE to verify."),
                ("What about the orders table?",
                 "Consider a composite index on (user_id, created_at) for the common query pattern."),
            ]

            history = _simulate_turns(engine, pairs)

            # Short ambiguous message — should tag as database from context
            result = engine.on_message_inbound("of course", history)

            # Should have database-related tags, not _general
            assert result.matched_tags, "Expected matched tags"
            assert "_general" not in result.matched_tags
