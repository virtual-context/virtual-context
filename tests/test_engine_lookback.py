"""Tests for tag context lookback in engine (on_message_inbound + on_turn_complete + ingest_history)."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message, TagResult, TurnTagEntry


def _make_engine(tmpdir: str, context_lookback_pairs: int = 5) -> VirtualContextEngine:
    """Build an engine with a mock tagger we can inspect."""
    db_path = str(Path(tmpdir) / "store.db")
    return VirtualContextEngine(config=load_config(config_dict={
        "context_window": 120_000,
        "storage_root": tmpdir,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "tag_generator": {
            "type": "keyword",
            "context_lookback_pairs": context_lookback_pairs,
            "context_bleed_threshold": 0,  # disable embedding gate for unit tests
            "keyword_fallback": {"tag_keywords": {}, "tag_patterns": {}},
        },
        "compaction": {
            "soft_threshold": 0.70,
            "hard_threshold": 0.85,
            "protected_recent_turns": 2,
        },
    }))


def _build_history(*pairs) -> list[Message]:
    """Build flat history from (user_text, assistant_text) pairs."""
    msgs = []
    for user_text, asst_text in pairs:
        msgs.append(Message(role="user", content=user_text))
        msgs.append(Message(role="assistant", content=asst_text))
    return msgs


class TestOnMessageInboundLookback:
    """Test that on_message_inbound passes context through the retriever to the tagger."""

    def _patch_tagger(self, engine, mock_tagger):
        """Replace the tagger in both the engine and the retriever."""
        engine._tag_generator = mock_tagger
        engine._retriever.tag_generator = mock_tagger
        # Null out the embedding-based inbound tagger so the retriever
        # falls back to tag_generator (which we just mocked).
        engine._retriever._inbound_tagger = None

    def test_inbound_retries_on_general(self):
        """When inbound tagger returns _general, engine should retry with expanded context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            call_count = 0
            def mock_generate(text, existing_tags=None, context_turns=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call: no/short context → _general
                    return TagResult(tags=["_general"], primary="_general", source="llm")
                # Retry with expanded context → meaningful tags
                return TagResult(tags=["database"], primary="database", source="llm")

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(side_effect=mock_generate)
            self._patch_tagger(engine, mock_tagger)

            # Prior turns about databases
            history = _build_history(
                ("How do I optimize the users table query?",
                 "Add an index on the email column."),
                ("What about the orders table?",
                 "Consider a composite index on (user_id, created_at)."),
            )

            assembled = engine.on_message_inbound("of course", history)

            # Should have retried and ended up with meaningful tags
            assert mock_tagger.generate_tags.call_count >= 2, (
                f"Expected retry, got {mock_tagger.generate_tags.call_count} call(s)"
            )
            assert "_general" not in assembled.matched_tags, (
                f"matched_tags should not contain _general, got {assembled.matched_tags}"
            )

    def test_inbound_inherits_on_persistent_general(self):
        """When retry still returns _general, inbound should fall back to inheritance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            # Always return _general regardless of context
            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["_general"], primary="_general", source="fallback",
            ))
            self._patch_tagger(engine, mock_tagger)

            # Populate TurnTagIndex with a prior database turn
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=0, message_hash="h0",
                tags=["database"], primary_tag="database",
            ))

            history = _build_history(
                ("How do I optimize queries?",
                 "Add an index on the email column."),
            )

            assembled = engine.on_message_inbound("ok", history)

            # Should inherit from the index rather than pass through _general
            assert "database" in assembled.matched_tags, (
                f"Expected inherited 'database', got {assembled.matched_tags}"
            )
            assert "_general" not in assembled.matched_tags

    def test_inbound_tagger_receives_context(self):
        """Inbound message after prior turns should send context_turns to tagger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["database"], primary="database", source="mock",
            ))
            self._patch_tagger(engine, mock_tagger)

            # History has 2 prior turns (user+assistant pairs already in context)
            history = _build_history(
                ("How do I optimize the users table query?",
                 "Add an index on the email column."),
                ("What about the orders table?",
                 "Consider a composite index on (user_id, created_at)."),
            )

            # Inbound message — "of course" is short/ambiguous
            engine.on_message_inbound("of course", history)

            # The tagger should have been called with context_turns
            assert mock_tagger.generate_tags.call_count >= 1
            call_kwargs = mock_tagger.generate_tags.call_args
            context = call_kwargs.kwargs.get("context_turns")
            assert context is not None, "Inbound tagger should receive context_turns"
            # Should contain text from the prior turns
            assert any("optimize" in c or "users table" in c for c in context)

    def test_inbound_no_history_no_context(self):
        """First message with empty history should not crash and should pass no context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["database"], primary="database", source="mock",
            ))
            self._patch_tagger(engine, mock_tagger)

            engine.on_message_inbound("How do I optimize queries?", [])

            call_kwargs = mock_tagger.generate_tags.call_args
            context = call_kwargs.kwargs.get("context_turns")
            assert context is None, "No history means no context"

    def test_inbound_context_respects_config(self):
        """context_lookback_pairs=1 should pass only 1 pair even with more history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, context_lookback_pairs=1)

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["database"], primary="database", source="mock",
            ))
            self._patch_tagger(engine, mock_tagger)

            history = _build_history(
                ("Turn 1 about cooking", "Recipe for pasta."),
                ("Turn 2 about music", "Try jazz."),
                ("Turn 3 about databases", "Use PostgreSQL."),
            )

            engine.on_message_inbound("tell me more", history)

            call_kwargs = mock_tagger.generate_tags.call_args
            context = call_kwargs.kwargs.get("context_turns")
            assert context is not None
            # 1 pair = 2 strings
            assert len(context) == 2, f"Expected 2 context strings (1 pair), got {len(context)}"


class TestOnTurnCompleteLookback:
    """Test that on_turn_complete passes context to tagger and retries on _general."""

    def test_short_message_gets_context(self):
        """Short message ('of course') should trigger tagger with context from prior turns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            # Install a mock tagger that records calls
            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["database"], primary="database", source="mock",
            ))
            engine._tag_generator = mock_tagger

            # Build history with 2 database turns + short response
            history = _build_history(
                ("How do I optimize the users table query?",
                 "Add an index on the email column."),
                ("What about the orders table?",
                 "Consider a composite index on (user_id, created_at)."),
                ("of course",
                 "Great, let me know if you need help with the implementation."),
            )

            # Process first two turns to populate index
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=0,
                message_hash="hash0",
                tags=["database"], primary_tag="database",
            ))
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=1,
                message_hash="hash1",
                tags=["database", "indexing"], primary_tag="database",
            ))

            # Now call on_turn_complete for the third turn
            engine.on_turn_complete(history)

            # Tagger should have been called with context_turns
            call_kwargs = mock_tagger.generate_tags.call_args
            assert call_kwargs is not None
            assert "context_turns" in call_kwargs.kwargs or (
                len(call_kwargs.args) > 2 and call_kwargs.args[2] is not None
            ), "Expected context_turns to be passed to generate_tags"

    def test_retry_on_general(self):
        """When tagger returns _general, engine should retry with expanded context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            call_count = 0
            def mock_generate(text, existing_tags=None, context_turns=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return TagResult(tags=["_general"], primary="_general", source="llm")
                return TagResult(tags=["database"], primary="database", source="llm")

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(side_effect=mock_generate)
            engine._tag_generator = mock_tagger

            history = _build_history(
                ("How do I optimize the users table?",
                 "Add an index on the email column."),
                ("of course",
                 "Great, glad you agree."),
            )
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=0, message_hash="h0",
                tags=["database"], primary_tag="database",
            ))

            engine.on_turn_complete(history)

            # Should have been called twice (initial + retry)
            assert mock_tagger.generate_tags.call_count == 2
            # Final entry should have "database" not "_general"
            last_entry = engine._turn_tag_index.entries[-1]
            assert "database" in last_entry.tags
            assert "_general" not in last_entry.tags

    def test_no_retry_when_specific_tags(self):
        """When tagger returns specific tags on first call, no retry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["database", "indexing"], primary="database", source="llm",
            ))
            engine._tag_generator = mock_tagger

            history = _build_history(
                ("How do I optimize the users table?",
                 "Add an index on the email column."),
                ("What about composite indexes?",
                 "Use (user_id, created_at) for common queries."),
            )
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=0, message_hash="h0",
                tags=["database"], primary_tag="database",
            ))

            engine.on_turn_complete(history)

            assert mock_tagger.generate_tags.call_count == 1

    def test_context_pairs_respects_config(self):
        """context_lookback_pairs=2 should only pass 2 pairs even with more available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, context_lookback_pairs=2)

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["database"], primary="database", source="llm",
            ))
            engine._tag_generator = mock_tagger

            # 5 turns of history
            history = _build_history(
                ("Turn 1 about cooking", "Recipe for pasta."),
                ("Turn 2 about music", "Try jazz."),
                ("Turn 3 about legal stuff", "File a motion."),
                ("Turn 4 about databases", "Use PostgreSQL."),
                ("Short response",
                 "Acknowledged."),
            )
            for i in range(4):
                engine._turn_tag_index.append(TurnTagEntry(
                    turn_number=i, message_hash=f"h{i}",
                    tags=[f"tag-{i}"], primary_tag=f"tag-{i}",
                ))

            engine.on_turn_complete(history)

            call_kwargs = mock_tagger.generate_tags.call_args
            context = call_kwargs.kwargs.get("context_turns") or (
                call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
            )
            assert context is not None
            # 2 pairs = 4 strings (user, assistant, user, assistant)
            assert len(context) <= 4, f"Expected max 4 context strings (2 pairs), got {len(context)}"


class TestIngestHistoryLookback:
    """Test that ingest_history passes context and retries instead of blind inheritance."""

    def test_short_message_not_inherited_blindly(self):
        """Short message after topic shift should get new tags from context, not inherited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            # Track what the tagger returns per call
            call_count = 0
            def mock_generate(text, existing_tags=None, context_turns=None):
                nonlocal call_count
                call_count += 1
                if "cooking" in text.lower() or "pasta" in text.lower():
                    return TagResult(tags=["cooking"], primary="cooking", source="mock")
                if "database" in text.lower() or "postgresql" in text.lower():
                    return TagResult(tags=["database"], primary="database", source="mock")
                # For "ok" with context about databases, return database
                if context_turns and any("database" in c.lower() or "postgresql" in c.lower() for c in context_turns):
                    return TagResult(tags=["database"], primary="database", source="mock")
                return TagResult(tags=["_general"], primary="_general", source="fallback")

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(side_effect=mock_generate)
            engine._tag_generator = mock_tagger

            history = _build_history(
                ("How do I make pasta?", "Boil water, add pasta, cook 8 minutes."),
                ("How do I set up PostgreSQL?", "Install via brew, create a database."),
                ("ok", "Let me know if you need help."),
            )

            engine.ingest_history(history)

            # The short "ok" turn should have database tags (from context), not cooking (inherited)
            last_entry = engine._turn_tag_index.entries[-1]
            assert "cooking" not in last_entry.tags, "Should not blindly inherit old topic"

    def test_context_passed_during_ingest(self):
        """Tagger calls during ingest should include context from preceding pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(return_value=TagResult(
                tags=["database"], primary="database", source="mock",
            ))
            engine._tag_generator = mock_tagger

            history = _build_history(
                ("First about databases", "Use PostgreSQL."),
                ("Then about indexing", "Add a B-tree index."),
                ("Short message", "Ok."),
            )

            engine.ingest_history(history)

            # Third call (turn index 2) should have context_turns from preceding pairs
            assert mock_tagger.generate_tags.call_count >= 3
            third_call = mock_tagger.generate_tags.call_args_list[2]
            context = third_call.kwargs.get("context_turns") or (
                third_call.args[2] if len(third_call.args) > 2 else None
            )
            assert context is not None, "Third call should have context_turns"
            assert len(context) >= 2, "Should have at least one preceding pair as context"

    def test_final_fallback_inherits(self):
        """If tagger returns _general even after retry, inheritance kicks in as last resort."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            def mock_generate(text, existing_tags=None, context_turns=None):
                if "database" in text.lower():
                    return TagResult(tags=["database"], primary="database", source="mock")
                # Always return _general for short messages
                return TagResult(tags=["_general"], primary="_general", source="fallback")

            mock_tagger = MagicMock()
            mock_tagger.generate_tags = MagicMock(side_effect=mock_generate)
            engine._tag_generator = mock_tagger

            history = _build_history(
                ("How do I set up a database?", "Use PostgreSQL."),
                ("ok", "Sure thing."),
            )

            engine.ingest_history(history)

            # The short turn should inherit from the database turn as last resort
            last_entry = engine._turn_tag_index.entries[-1]
            assert "database" in last_entry.tags, "Should inherit from previous meaningful turn"
            assert "_general" not in last_entry.tags
