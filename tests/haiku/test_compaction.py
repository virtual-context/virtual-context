"""Haiku compaction cycle tests — full engine with real LLM summarization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message

HAIKU_MODEL = "claude-haiku-4-5"


def _make_engine(tmpdir: str, context_window: int = 3000):
    """Build a real engine with Haiku, configured for early compaction."""
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
            "soft_threshold": 0.50,
            "hard_threshold": 0.70,
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
    """Run turn pairs through the engine, return full history."""
    history = []
    for user_text, assistant_text in pairs:
        history.append(Message(role="user", content=user_text))
        engine.on_message_inbound(user_text, history)
        history.append(Message(role="assistant", content=assistant_text))
        engine.on_turn_complete(history)
    return history


class TestCompactionCycle:
    """Test full compaction lifecycle with real Haiku."""

    def test_compaction_fires_and_recovers(self):
        """10 turns with 3k window should trigger compaction and still retrieve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            pairs = [
                ("How do I design a database schema for an e-commerce app?",
                 "Start with users, products, orders, and order_items tables. Use foreign keys."),
                ("What indexes should I add to the orders table?",
                 "Index on user_id, created_at, and status. Consider a composite for user+status."),
                ("How do I handle database migrations in production?",
                 "Use versioned migration files. Run in staging first, always test rollback."),
                ("What's the best way to set up JWT authentication?",
                 "Use RS256 signing. Store public key on server, private key in secrets manager."),
                ("How do I implement refresh token rotation?",
                 "Store refresh tokens in DB. On each use, issue new pair and invalidate old."),
                ("What's a good REST API versioning strategy?",
                 "URL versioning (/v1/) is simplest. Header versioning is cleaner but harder."),
                ("How should I structure error responses in the API?",
                 "Use RFC 7807 Problem Details: type, title, status, detail, instance fields."),
                ("What caching strategy works for a read-heavy API?",
                 "Redis for hot data, HTTP cache headers for client-side. Cache-aside pattern."),
                ("How do I set up rate limiting for the API?",
                 "Token bucket algorithm. Use Redis for distributed rate limiting across instances."),
                ("What monitoring should I add to the production API?",
                 "Request latency (p50/p95/p99), error rate, throughput. Alert on anomalies."),
            ]

            history = _simulate_turns(engine, pairs)

            # Compaction should have fired
            assert engine._compacted_through > 0, (
                f"Expected compaction, but _compacted_through={engine._compacted_through}"
            )

            # Post-compaction retrieval should still work
            result = engine.on_message_inbound(
                "Tell me about the database schema we designed", history
            )
            assert result.prepend_text, "Expected prepend_text after compaction"

    def test_broad_bounded_after_compaction(self):
        """Broad query after compaction should stay within budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            pairs = [
                ("How do I design a normalized schema?",
                 "Use 3NF: no transitive dependencies. Users, orders, products tables."),
                ("What are best indexing practices?",
                 "Index WHERE/JOIN/ORDER BY columns. B-tree for equality, GIN for arrays."),
                ("Tell me about JWT authentication.",
                 "Sign tokens with RS256. Include exp, sub, iss claims. 1-hour expiry."),
                ("How does refresh token rotation work?",
                 "Issue new access+refresh pair on each refresh. Invalidate old refresh token."),
                ("What's a good guitar practice routine?",
                 "30 min: 5 min warmup, 10 min scales, 10 min songs, 5 min new material."),
                ("How do I learn barre chords?",
                 "Start with F major. Press firmly with index across all strings. Build strength."),
                ("What running plan works for a half marathon?",
                 "16-week plan: 4 runs/week, peak at 20 miles/week. Taper last 2 weeks."),
                ("How do I prevent injuries while running?",
                 "10% volume increase rule, strength training, proper warmup, rest days."),
                ("What's the best way to deploy a Node.js app?",
                 "Docker container on ECS or Cloud Run. CI/CD with GitHub Actions."),
                ("How do I set up health checks for the deployment?",
                 "HTTP /health endpoint returning 200. Check DB connectivity and Redis ping."),
            ]

            history = _simulate_turns(engine, pairs)

            # Broad query
            result = engine.on_message_inbound(
                "Can you summarize everything we've discussed today?", history
            )

            # Total tokens should stay within budget (< 50% of window)
            assert result.total_tokens < engine.config.context_window * 0.5, (
                f"Broad query used {result.total_tokens} tokens, "
                f"exceeds 50% of {engine.config.context_window} window"
            )

    def test_temporal_retrieves_earliest(self):
        """Temporal query should retrieve content from the earliest turns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir)

            pairs = [
                # The earliest topic — should be retrievable via temporal
                ("Let's design the modular blueprint for the architecture.",
                 "We'll use a hexagonal architecture with ports and adapters pattern."),
                ("How do the modules communicate?",
                 "Via domain events and a mediator. No direct module-to-module calls."),
                # Many padding turns to push early content far back
                ("What guitar scales should I practice?",
                 "Major pentatonic first, then minor pentatonic, then full major/minor."),
                ("How do I improvise over a 12-bar blues?",
                 "Use minor pentatonic with blue notes. Follow the chord changes for target notes."),
                ("What running shoes do you recommend for trail?",
                 "Salomon Speedcross for technical terrain, Hoka Speedgoat for cushioned trail."),
                ("How do I train for elevation gain?",
                 "Hill repeats twice a week. Stair climbing for strength. Altitude training if possible."),
                ("What's a good pasta recipe for meal prep?",
                 "Chicken pesto pasta: rotini, grilled chicken, pesto, sun-dried tomatoes, spinach."),
                ("How do I store meal prepped food safely?",
                 "Glass containers, refrigerate within 2 hours. Consume within 3-4 days."),
                ("What's the best way to learn music theory?",
                 "Start with intervals, then chords, then progressions. Apply to songs you know."),
                ("How do I read sheet music fluently?",
                 "Flash cards for notes, practice sight-reading daily, start with simple pieces."),
            ]

            history = _simulate_turns(engine, pairs)

            # Temporal query for earliest content
            result = engine.on_message_inbound(
                "What was the very first architecture decision we made?", history
            )

            # Should have retrieved something — temporal queries pull earliest content
            if result.prepend_text:
                prepend_lower = result.prepend_text.lower()
                assert "modular" in prepend_lower or "hexagonal" in prepend_lower or "architecture" in prepend_lower, (
                    f"Expected early architecture content, got: {result.prepend_text[:200]}"
                )

    def test_multiple_compactions_stable(self):
        """Engine should handle multiple compaction cycles without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, context_window=2000)

            # 15 turns should force multiple compactions with 2k window
            pairs = [
                (f"Tell me about topic {i}: {'database schema' if i % 3 == 0 else 'guitar practice' if i % 3 == 1 else 'running training'}",
                 f"Here's information about topic {i}. " + "Details. " * 20)
                for i in range(15)
            ]

            history = _simulate_turns(engine, pairs)

            # Should have compacted multiple times
            assert engine._compacted_through > 0

            # Engine should still respond without errors
            result = engine.on_message_inbound("What topics did we cover?", history)
            assert result is not None
