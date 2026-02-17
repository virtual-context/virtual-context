"""Tests for broad query detection and handling."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.conftest import MockLLMProvider, MockTagGenerator
from virtual_context.core.retriever import ContextRetriever
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    AssembledContext,
    Message,
    RetrieverConfig,
    SegmentMetadata,
    StoredSegment,
    StrategyConfig,
    TagResult,
    TagSummary,
    TurnTagEntry,
)


class TestTagResultBroad:
    def test_broad_defaults_false(self):
        result = TagResult(tags=["test"], primary="test", source="mock")
        assert result.broad is False

    def test_broad_explicit_true(self):
        result = TagResult(tags=["test"], primary="test", source="mock", broad=True)
        assert result.broad is True


class TestAssembledContextBroad:
    def test_broad_defaults_false(self):
        ctx = AssembledContext()
        assert ctx.broad is False


class TestFilterHistoryBroad:
    """Test engine.filter_history with broad flag."""

    def _make_engine_with_index(self, tmp_path):
        """Build a minimal engine for filter_history testing."""
        from virtual_context.config import load_config
        from virtual_context.engine import VirtualContextEngine

        config = load_config(config_dict={
            "context_window": 50000,
            "storage_root": str(tmp_path),
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "store.db")}},
            "tag_generator": {"type": "keyword"},
        })
        engine = VirtualContextEngine(config=config)

        # Populate turn tag index
        for i in range(10):
            tag = "legal" if i % 2 == 0 else "medical"
            engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i}",
                tags=[tag], primary_tag=tag,
            ))
        return engine

    def test_broad_true_includes_all_pre_compaction(self, tmp_path):
        """Pre-compaction, broad=True returns all history."""
        engine = self._make_engine_with_index(tmp_path)
        ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

        history = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            history.append(Message(role=role, content=f"msg {i}", timestamp=ts + timedelta(minutes=i)))

        filtered = engine.filter_history(history, current_tags=["legal"], broad=True)
        assert len(filtered) == len(history)

    @pytest.mark.regression("BUG-003")
    def test_broad_true_skips_compacted_post_compaction(self, tmp_path):
        """Post-compaction, broad=True skips compacted messages."""
        engine = self._make_engine_with_index(tmp_path)
        engine._compacted_through = 10  # 5 turns compacted
        ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

        history = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            history.append(Message(role=role, content=f"msg {i}", timestamp=ts + timedelta(minutes=i)))

        filtered = engine.filter_history(history, current_tags=["legal"], broad=True)
        # Should skip first 10 messages (compacted), return remaining 10
        assert len(filtered) == 10
        assert filtered[0].content == "msg 10"

    def test_broad_false_filters_normally(self, tmp_path):
        engine = self._make_engine_with_index(tmp_path)
        ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

        history = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            history.append(Message(role=role, content=f"msg {i}", timestamp=ts + timedelta(minutes=i)))

        filtered = engine.filter_history(history, current_tags=["legal"], broad=False)
        # Should drop some medical turns from older portion
        assert len(filtered) < len(history)


class TestBroadRetrieval:
    """Test retriever broad query branch."""

    @pytest.fixture
    def store_with_tag_summaries(self, tmp_sqlite_db):
        store = SQLiteStore(db_path=tmp_sqlite_db)
        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

        # Add segment summaries
        store.store_segment(StoredSegment(
            ref="legal-1", primary_tag="legal", tags=["legal"],
            summary="Legal case discussion", summary_tokens=20,
            full_tokens=100, metadata=SegmentMetadata(),
            created_at=now, start_timestamp=now, end_timestamp=now,
        ))

        # Add tag summaries
        store.save_tag_summary(TagSummary(
            tag="legal", summary="Comprehensive legal overview",
            summary_tokens=50, source_segment_refs=["legal-1"],
            covers_through_turn=5, created_at=now, updated_at=now,
        ))
        store.save_tag_summary(TagSummary(
            tag="medical", summary="Medical discussion overview",
            summary_tokens=40, source_segment_refs=["medical-1"],
            covers_through_turn=5, created_at=now, updated_at=now,
        ))
        yield store
        store.close()

    @pytest.mark.regression("BUG-007")
    def test_broad_retrieval_loads_tag_summaries(self, store_with_tag_summaries):
        """Broad query should load all tag summaries."""
        tag_gen = MockTagGenerator(default_tag="broad-query")
        tag_gen.set_override(
            "earlier",
            TagResult(tags=["broad-query"], primary="broad-query", source="mock", broad=True),
        )

        retriever = ContextRetriever(
            tag_generator=tag_gen,
            store=store_with_tag_summaries,
            config=RetrieverConfig(
                tag_context_max_tokens=30000,
                strategy_configs={"default": StrategyConfig()},
            ),
        )

        result = retriever.retrieve("What did you say earlier about that?")
        assert result.broad is True
        assert len(result.summaries) == 2  # Both tag summaries
        assert result.retrieval_metadata.get("broad") is True

    def test_broad_falls_through_when_no_summaries(self, tmp_sqlite_db):
        """Broad query with no tag summaries falls through to normal retrieval."""
        store = SQLiteStore(db_path=tmp_sqlite_db)
        tag_gen = MockTagGenerator(default_tag="legal")
        tag_gen.set_override(
            "earlier",
            TagResult(tags=["legal"], primary="legal", source="mock", broad=True),
        )

        # Add a segment but NO tag summaries
        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        store.store_segment(StoredSegment(
            ref="legal-1", primary_tag="legal", tags=["legal"],
            summary="Legal discussion", summary_tokens=20,
            full_tokens=100, metadata=SegmentMetadata(),
            created_at=now, start_timestamp=now, end_timestamp=now,
        ))

        retriever = ContextRetriever(
            tag_generator=tag_gen,
            store=store,
            config=RetrieverConfig(
                tag_context_max_tokens=30000,
                strategy_configs={"default": StrategyConfig()},
            ),
        )

        result = retriever.retrieve("What did you say earlier?")
        assert result.broad is True
        # Falls through to normal retrieval — may find segments by tag
        store.close()

    def test_non_broad_query_skips_tag_summaries(self, store_with_tag_summaries):
        """Non-broad queries should NOT load tag summaries."""
        tag_gen = MockTagGenerator(default_tag="legal")
        tag_gen.set_override(
            "court",
            TagResult(tags=["legal"], primary="legal", source="mock", broad=False),
        )

        retriever = ContextRetriever(
            tag_generator=tag_gen,
            store=store_with_tag_summaries,
            config=RetrieverConfig(
                tag_context_max_tokens=30000,
                strategy_configs={"default": StrategyConfig()},
            ),
        )

        result = retriever.retrieve("What about the court filing?")
        assert result.broad is False


class TestContextHint:
    """Test engine._build_context_hint()."""

    def _make_engine(self, tmp_path, hint_enabled=True):
        from virtual_context.config import load_config
        from virtual_context.engine import VirtualContextEngine

        config = load_config(config_dict={
            "context_window": 50000,
            "storage_root": str(tmp_path),
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "store.db")}},
            "tag_generator": {"type": "keyword"},
            "assembly": {"context_hint_enabled": hint_enabled},
        })
        return VirtualContextEngine(config=config)

    def test_hint_empty_pre_compaction(self, tmp_path):
        """No hint before any compaction occurs."""
        engine = self._make_engine(tmp_path)
        assert engine._build_context_hint() == ""

    def test_hint_empty_when_disabled(self, tmp_path):
        """No hint when feature is disabled."""
        engine = self._make_engine(tmp_path, hint_enabled=False)
        engine._compacted_through = 10
        assert engine._build_context_hint() == ""

    def test_hint_generated_post_compaction(self, tmp_path):
        """Hint generated after compaction with tag summaries."""
        engine = self._make_engine(tmp_path)
        engine._compacted_through = 10

        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        engine._store.save_tag_summary(TagSummary(
            tag="legal", summary="Legal case 24-cv-1234 discussed, filing due Jan 30",
            summary_tokens=50, source_segment_refs=["seg-1"],
            source_turn_numbers=[0, 1, 2],
            covers_through_turn=5, created_at=now, updated_at=now,
        ))
        engine._store.save_tag_summary(TagSummary(
            tag="medical", summary="Blood glucose monitoring and insulin adjustments",
            summary_tokens=40, source_segment_refs=["seg-2"],
            source_turn_numbers=[3, 4],
            covers_through_turn=5, created_at=now, updated_at=now,
        ))

        hint = engine._build_context_hint()
        assert "<context-topics>" in hint
        assert "</context-topics>" in hint
        assert "legal (3 turns)" in hint
        assert "medical (2 turns)" in hint

    def test_hint_empty_when_no_tag_summaries(self, tmp_path):
        """No hint when store has no tag summaries."""
        engine = self._make_engine(tmp_path)
        engine._compacted_through = 10
        assert engine._build_context_hint() == ""
