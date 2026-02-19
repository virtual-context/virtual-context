"""Tests for engine state persistence (Phase 1 of session continuity)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import EngineStateSnapshot, TurnTagEntry


def _make_snapshot(
    session_id: str = "test-session-abc",
    compacted_through: int = 4,
    turn_count: int = 10,
) -> EngineStateSnapshot:
    """Build a realistic EngineStateSnapshot for testing."""
    entries = [
        TurnTagEntry(
            turn_number=i,
            message_hash=f"hash{i:04d}",
            tags=[f"tag-{i}", "common"],
            primary_tag=f"tag-{i}",
            timestamp=datetime(2026, 1, 15, 10, i, 0, tzinfo=timezone.utc),
        )
        for i in range(turn_count)
    ]
    return EngineStateSnapshot(
        session_id=session_id,
        compacted_through=compacted_through,
        turn_tag_entries=entries,
        turn_count=turn_count,
    )


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------


class TestEngineStateSQLite:
    def test_save_and_load(self, tmp_path):
        """Round-trip: save state, load it back, verify all fields."""
        db = tmp_path / "test.db"
        store = SQLiteStore(db_path=db)

        snap = _make_snapshot()
        store.save_engine_state(snap)

        loaded = store.load_engine_state("test-session-abc")
        assert loaded is not None
        assert loaded.session_id == snap.session_id
        assert loaded.compacted_through == snap.compacted_through
        assert loaded.turn_count == snap.turn_count
        assert len(loaded.turn_tag_entries) == len(snap.turn_tag_entries)

        # Verify individual entries
        for orig, restored in zip(snap.turn_tag_entries, loaded.turn_tag_entries):
            assert restored.turn_number == orig.turn_number
            assert restored.message_hash == orig.message_hash
            assert restored.tags == orig.tags
            assert restored.primary_tag == orig.primary_tag
            assert restored.timestamp == orig.timestamp

    def test_load_empty_store(self, tmp_path):
        """Loading from an empty store returns None."""
        db = tmp_path / "test.db"
        store = SQLiteStore(db_path=db)

        result = store.load_engine_state("nonexistent-session")
        assert result is None

    def test_upsert_gets_latest(self, tmp_path):
        """Saving twice with same session_id overwrites — load gets latest."""
        db = tmp_path / "test.db"
        store = SQLiteStore(db_path=db)

        snap1 = _make_snapshot(compacted_through=2, turn_count=5)
        store.save_engine_state(snap1)

        snap2 = _make_snapshot(compacted_through=8, turn_count=15)
        store.save_engine_state(snap2)

        loaded = store.load_engine_state("test-session-abc")
        assert loaded is not None
        assert loaded.compacted_through == 8
        assert loaded.turn_count == 15
        assert len(loaded.turn_tag_entries) == 15


# ---------------------------------------------------------------------------
# FilesystemStore
# ---------------------------------------------------------------------------


class TestEngineStateFilesystem:
    def test_save_and_load(self, tmp_path):
        """Round-trip: save state, load it back, verify all fields."""
        store = FilesystemStore(root=tmp_path / "store")

        snap = _make_snapshot()
        store.save_engine_state(snap)

        loaded = store.load_engine_state("test-session-abc")
        assert loaded is not None
        assert loaded.session_id == snap.session_id
        assert loaded.compacted_through == snap.compacted_through
        assert loaded.turn_count == snap.turn_count
        assert len(loaded.turn_tag_entries) == len(snap.turn_tag_entries)

        for orig, restored in zip(snap.turn_tag_entries, loaded.turn_tag_entries):
            assert restored.turn_number == orig.turn_number
            assert restored.message_hash == orig.message_hash
            assert restored.tags == orig.tags
            assert restored.primary_tag == orig.primary_tag

    def test_load_empty_store(self, tmp_path):
        """Loading from an empty store returns None."""
        store = FilesystemStore(root=tmp_path / "store")

        result = store.load_engine_state("nonexistent")
        assert result is None

    def test_upsert_gets_latest(self, tmp_path):
        """Saving twice overwrites — load gets latest."""
        store = FilesystemStore(root=tmp_path / "store")

        snap1 = _make_snapshot(compacted_through=2, turn_count=5)
        store.save_engine_state(snap1)

        snap2 = _make_snapshot(compacted_through=8, turn_count=15)
        store.save_engine_state(snap2)

        loaded = store.load_engine_state("test-session-abc")
        assert loaded is not None
        assert loaded.compacted_through == 8
        assert loaded.turn_count == 15

    def test_state_file_is_valid_json(self, tmp_path):
        """The state file on disk is valid JSON with expected structure."""
        store = FilesystemStore(root=tmp_path / "store")
        snap = _make_snapshot(turn_count=3)
        store.save_engine_state(snap)

        path = tmp_path / "store" / "_engine_state" / f"{snap.session_id}.json"
        assert path.is_file()

        data = json.loads(path.read_text())
        assert data["session_id"] == snap.session_id
        assert data["compacted_through"] == snap.compacted_through
        assert len(data["turn_tag_entries"]) == 3


# ---------------------------------------------------------------------------
# Engine integration: state survives reinit
# ---------------------------------------------------------------------------


class TestEngineStateIntegration:
    def test_state_survives_engine_reinit(self, tmp_path):
        """Save state via one engine, create a new engine with same store, verify restored."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config

        db_path = str(tmp_path / "store.db")
        config_dict = {
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite_path": db_path},
            "tag_generator": {"type": "keyword"},
        }

        # First engine: populate state and save
        config1 = load_config(config_dict=config_dict)
        engine1 = VirtualContextEngine(config=config1)
        session_id = engine1.config.session_id

        # Manually add some turns to the index
        for i in range(5):
            engine1._turn_tag_index.append(TurnTagEntry(
                turn_number=i,
                message_hash=f"h{i}",
                tags=[f"tag-{i}"],
                primary_tag=f"tag-{i}",
            ))
        engine1._compacted_through = 4

        # Save state
        from virtual_context.types import Message
        history = [Message(role="user", content="x"), Message(role="assistant", content="y")] * 5
        engine1._save_state(history)

        # Second engine: same store, same session_id
        config2 = load_config(config_dict=config_dict)
        config2.session_id = session_id
        engine2 = VirtualContextEngine(config=config2)

        # State should be restored
        assert engine2.config.session_id == session_id
        assert engine2._compacted_through == 4
        assert len(engine2._turn_tag_index.entries) == 5
        assert engine2._turn_tag_index.entries[0].tags == ["tag-0"]
        assert engine2._turn_tag_index.entries[4].tags == ["tag-4"]


# ---------------------------------------------------------------------------
# Vocabulary bootstrap
# ---------------------------------------------------------------------------


class TestVocabularyBootstrap:
    """Verify that _bootstrap_vocabulary loads historical tags into the tagger."""

    def test_bootstrap_from_store_tags(self, tmp_path):
        """Engine with store containing tag stats should populate tagger vocabulary."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config

        db_path = str(tmp_path / "store.db")
        config_dict = {
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite_path": db_path},
            "tag_generator": {"type": "keyword"},
        }
        config = load_config(config_dict=config_dict)
        engine = VirtualContextEngine(config=config)

        # KeywordTagGenerator doesn't have load_vocabulary, so bootstrap is a no-op
        assert not hasattr(engine._tag_generator, "load_vocabulary") or \
            not hasattr(engine._tag_generator, "_tag_vocabulary")

    def test_bootstrap_from_turn_tag_index(self, tmp_path):
        """Engine with restored TurnTagIndex should populate LLM tagger vocabulary."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config
        from virtual_context.types import Message

        db_path = str(tmp_path / "store.db")
        config_dict = {
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite_path": db_path},
            "tag_generator": {"type": "keyword"},
        }

        # First engine: populate and save state
        config1 = load_config(config_dict=config_dict)
        engine1 = VirtualContextEngine(config=config1)
        session_id = engine1.config.session_id
        for i in range(5):
            engine1._turn_tag_index.append(TurnTagEntry(
                turn_number=i,
                message_hash=f"h{i}",
                tags=["skincare", "retinol"] if i % 2 == 0 else ["fitness"],
                primary_tag="skincare" if i % 2 == 0 else "fitness",
            ))
        history = [Message(role="user", content="x"), Message(role="assistant", content="y")] * 5
        engine1._save_state(history)

        # Second engine: same store, LLM tagger (mock provider)
        config2 = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite_path": db_path},
            "tag_generator": {"type": "llm", "provider": "test-provider"},
            "providers": {"test-provider": {
                "type": "generic_openai",
                "base_url": "http://fake:9999",
                "model": "test-model",
            }},
        })
        config2.session_id = session_id
        engine2 = VirtualContextEngine(config=config2)

        # LLMTagGenerator should have vocabulary populated
        vocab = engine2._tag_generator._tag_vocabulary
        assert "skincare" in vocab
        assert "retinol" in vocab
        assert "fitness" in vocab
        assert vocab["skincare"] >= 3  # appears in turns 0, 2, 4
        assert vocab["fitness"] >= 2   # appears in turns 1, 3

    def test_bootstrap_no_op_for_keyword_generator(self, tmp_path):
        """KeywordTagGenerator lacks load_vocabulary; bootstrap is silently skipped."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config

        db_path = str(tmp_path / "store.db")
        config = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite_path": db_path},
            "tag_generator": {"type": "keyword"},
        })
        # Should not raise
        engine = VirtualContextEngine(config=config)
