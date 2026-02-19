"""Tests for LLM-Driven Context Navigation (Virtual Memory Paging) — Phases 1-5."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from virtual_context.config import _build_config, load_config
from virtual_context.core.assembler import ContextAssembler
from virtual_context.types import (
    AssemblerConfig,
    DepthLevel,
    EngineStateSnapshot,
    Message,
    PagingConfig,
    RetrievalResult,
    StoredSegment,
    StoredSummary,
    TagSummary,
    TurnTagEntry,
    VirtualContextConfig,
    WorkingSetEntry,
)


# ---------------------------------------------------------------------------
# Phase 1: Types & State
# ---------------------------------------------------------------------------

class TestDepthLevel:
    def test_enum_values(self):
        assert DepthLevel.NONE.value == "none"
        assert DepthLevel.SUMMARY.value == "summary"
        assert DepthLevel.SEGMENTS.value == "segments"
        assert DepthLevel.FULL.value == "full"

    def test_from_string(self):
        assert DepthLevel("none") == DepthLevel.NONE
        assert DepthLevel("summary") == DepthLevel.SUMMARY
        assert DepthLevel("segments") == DepthLevel.SEGMENTS
        assert DepthLevel("full") == DepthLevel.FULL

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError):
            DepthLevel("invalid")

    def test_str_enum_comparison(self):
        """DepthLevel is a str enum, so it compares with strings."""
        assert DepthLevel.SUMMARY == "summary"
        assert DepthLevel.FULL == "full"


class TestWorkingSetEntry:
    def test_defaults(self):
        ws = WorkingSetEntry(tag="database")
        assert ws.depth == DepthLevel.SUMMARY
        assert ws.tokens == 0
        assert ws.last_accessed_turn == 0

    def test_custom_values(self):
        ws = WorkingSetEntry(
            tag="api",
            depth=DepthLevel.FULL,
            tokens=8000,
            last_accessed_turn=42,
        )
        assert ws.tag == "api"
        assert ws.depth == DepthLevel.FULL
        assert ws.tokens == 8000
        assert ws.last_accessed_turn == 42


class TestPagingConfig:
    def test_defaults(self):
        cfg = PagingConfig()
        assert cfg.enabled is False
        assert cfg.autonomous_models == ["opus", "sonnet", "gpt-4", "gpt-4o"]
        assert cfg.auto_promote is True
        assert cfg.auto_evict is True

    def test_custom_values(self):
        cfg = PagingConfig(enabled=True, autonomous_models=["opus"], auto_evict=False)
        assert cfg.enabled is True
        assert cfg.autonomous_models == ["opus"]
        assert cfg.auto_evict is False


class TestEngineStateSnapshotPaging:
    def test_working_set_defaults_to_empty(self):
        snap = EngineStateSnapshot(
            session_id="test",
            compacted_through=0,
            turn_tag_entries=[],
            turn_count=0,
        )
        assert snap.working_set == []

    def test_working_set_roundtrip(self):
        ws_entries = [
            WorkingSetEntry(tag="db", depth=DepthLevel.FULL, tokens=5000, last_accessed_turn=10),
            WorkingSetEntry(tag="api", depth=DepthLevel.SUMMARY, tokens=200, last_accessed_turn=8),
        ]
        snap = EngineStateSnapshot(
            session_id="test",
            compacted_through=4,
            turn_tag_entries=[],
            turn_count=5,
            working_set=ws_entries,
        )
        assert len(snap.working_set) == 2
        assert snap.working_set[0].tag == "db"
        assert snap.working_set[0].depth == DepthLevel.FULL

    def test_backward_compat_missing_working_set(self):
        """Old snapshots without working_set should load with empty list."""
        snap = EngineStateSnapshot(
            session_id="old-session",
            compacted_through=2,
            turn_tag_entries=[],
            turn_count=3,
        )
        # Simulate the engine's restoration: (saved.working_set or [])
        restored = {ws.tag: ws for ws in (snap.working_set or [])}
        assert restored == {}


# ---------------------------------------------------------------------------
# Phase 2: Store get_segments_by_tags + Assembler depth
# ---------------------------------------------------------------------------

class TestAssemblerPagingDepths:
    """Test that assembler correctly renders content at different depth levels."""

    def _make_assembler(self, tag_budget=30_000):
        return ContextAssembler(
            config=AssemblerConfig(tag_context_max_tokens=tag_budget),
            token_counter=lambda text: len(text) // 4,
        )

    def _make_retrieval_result(self, summaries):
        return RetrievalResult(
            tags_matched=[s.primary_tag for s in summaries],
            summaries=summaries,
        )

    def _make_summary(self, tag, text="Summary text for testing.", tokens=50):
        return StoredSummary(
            ref=f"seg-{tag}",
            primary_tag=tag,
            tags=[tag],
            summary=text,
            summary_tokens=tokens,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

    def _make_segment(self, tag, summary="Segment summary.", full_text="Full original text of the conversation segment."):
        return StoredSegment(
            ref=f"seg-{tag}-full",
            primary_tag=tag,
            tags=[tag],
            summary=summary,
            summary_tokens=len(summary) // 4,
            full_text=full_text,
            full_tokens=len(full_text) // 4,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

    def test_summary_depth_default(self):
        """Without working_set, tags render at SUMMARY depth."""
        asm = self._make_assembler()
        summary = self._make_summary("database")
        result = asm.assemble(
            core_context="",
            retrieval_result=self._make_retrieval_result([summary]),
            conversation_history=[],
            token_budget=100_000,
        )
        assert "database" in result.tag_sections
        assert "virtual-context" in result.tag_sections["database"]
        assert "Summary text" in result.tag_sections["database"]

    def test_segments_depth_with_working_set(self):
        """SEGMENTS depth renders individual segment summaries."""
        asm = self._make_assembler()
        summary = self._make_summary("api")
        segment = self._make_segment("api", summary="API design patterns discussed.")

        working_set = {
            "api": WorkingSetEntry(tag="api", depth=DepthLevel.SEGMENTS, tokens=500),
        }

        result = asm.assemble(
            core_context="",
            retrieval_result=self._make_retrieval_result([summary]),
            conversation_history=[],
            token_budget=100_000,
            working_set=working_set,
            full_segments={"api": [segment]},
        )
        assert "api" in result.tag_sections
        assert 'depth="segments"' in result.tag_sections["api"]

    def test_full_depth_with_working_set(self):
        """FULL depth renders full_text from StoredSegment."""
        asm = self._make_assembler()
        summary = self._make_summary("auth")
        full_text = "Complete authentication implementation discussion with all details."
        segment = self._make_segment("auth", full_text=full_text)

        working_set = {
            "auth": WorkingSetEntry(tag="auth", depth=DepthLevel.FULL, tokens=2000),
        }

        result = asm.assemble(
            core_context="",
            retrieval_result=self._make_retrieval_result([summary]),
            conversation_history=[],
            token_budget=100_000,
            working_set=working_set,
            full_segments={"auth": [segment]},
        )
        assert "auth" in result.tag_sections
        assert 'depth="full"' in result.tag_sections["auth"]
        assert full_text in result.tag_sections["auth"]

    def test_none_depth_skips_tag(self):
        """NONE depth skips the tag entirely (hint only)."""
        asm = self._make_assembler()
        summary = self._make_summary("old-topic")

        working_set = {
            "old-topic": WorkingSetEntry(tag="old-topic", depth=DepthLevel.NONE, tokens=0),
        }

        result = asm.assemble(
            core_context="",
            retrieval_result=self._make_retrieval_result([summary]),
            conversation_history=[],
            token_budget=100_000,
            working_set=working_set,
        )
        assert "old-topic" not in result.tag_sections

    def test_mixed_depths(self):
        """Different tags at different depths in same assembly."""
        asm = self._make_assembler()
        s1 = self._make_summary("database", "DB summary.")
        s2 = self._make_summary("api", "API summary.")
        seg2 = self._make_segment("api", full_text="Full API discussion text.")

        working_set = {
            "database": WorkingSetEntry(tag="database", depth=DepthLevel.SUMMARY, tokens=50),
            "api": WorkingSetEntry(tag="api", depth=DepthLevel.FULL, tokens=500),
        }

        result = asm.assemble(
            core_context="",
            retrieval_result=self._make_retrieval_result([s1, s2]),
            conversation_history=[],
            token_budget=100_000,
            working_set=working_set,
            full_segments={"api": [seg2]},
        )
        assert "database" in result.tag_sections
        assert "api" in result.tag_sections
        # database should be at default (SUMMARY), no depth attr
        assert 'depth="full"' not in result.tag_sections["database"]
        assert 'depth="full"' in result.tag_sections["api"]


class TestAssemblerHeadroom:
    """Tests for max_context_tokens headroom-aware assembly."""

    def _make_assembler(self, tag_budget=30_000, core_budget=18_000):
        return ContextAssembler(
            config=AssemblerConfig(
                tag_context_max_tokens=tag_budget,
                core_context_max_tokens=core_budget,
            ),
            token_counter=lambda text: len(text) // 4,
        )

    def _make_retrieval_result(self, tags_and_summaries):
        """Create RetrievalResult with multiple StoredSummary objects."""
        summaries = []
        for tag, text in tags_and_summaries:
            summaries.append(StoredSummary(
                ref=f"seg-{tag}",
                primary_tag=tag,
                tags=[tag],
                summary=text,
                summary_tokens=len(text) // 4,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ))
        return RetrievalResult(
            tags_matched=[s.primary_tag for s in summaries],
            summaries=summaries,
        )

    def test_assembler_headroom_caps_tag_budget(self):
        """With max_context_tokens=5000, tag sections are capped by available headroom."""
        asm = self._make_assembler(tag_budget=30_000, core_budget=18_000)
        # Create many summaries with substantial text
        tags_and_summaries = [
            (f"topic-{i}", f"Summary text for topic {i}. " * 50)
            for i in range(10)
        ]
        result = asm.assemble(
            core_context="Core context text.",
            retrieval_result=self._make_retrieval_result(tags_and_summaries),
            conversation_history=[],
            token_budget=100_000,
            context_hint="<context-topics>hint text</context-topics>",
            max_context_tokens=200,  # Very tight headroom
        )
        # With only 200 tokens total and core + hint eating most of it,
        # tag_sections should be severely limited or empty
        total_tag_tokens = sum(
            len(section) // 4 for section in result.tag_sections.values()
        )
        # Core text is "Core context text." = ~5 tokens
        # Hint is ~10 tokens
        # So available for tags is 200 - 5 - 10 = 185 tokens max
        assert total_tag_tokens <= 200  # Must be within headroom

    def test_assembler_headroom_none_uses_default(self):
        """With max_context_tokens=None, normal tag_budget applies."""
        asm = self._make_assembler(tag_budget=30_000)
        tags_and_summaries = [
            ("database", "Database schema design with PostgreSQL."),
            ("api", "REST API design patterns discussed."),
        ]
        result = asm.assemble(
            core_context="",
            retrieval_result=self._make_retrieval_result(tags_and_summaries),
            conversation_history=[],
            token_budget=100_000,
            max_context_tokens=None,
        )
        # Both tags should be present since 30k budget is plenty
        assert len(result.tag_sections) == 2
        assert "database" in result.tag_sections
        assert "api" in result.tag_sections

    def test_assembler_headroom_zero_hint_only(self):
        """With max_context_tokens equal to core+hint, tag budget is 0 but hint is present."""
        asm = self._make_assembler(tag_budget=30_000, core_budget=18_000)
        core = "Core context."  # ~3 tokens with //4
        hint = "<context-topics>Topics here</context-topics>"  # ~10 tokens with //4
        core_tokens = len(core) // 4
        hint_tokens = len(hint) // 4
        headroom = core_tokens + hint_tokens  # exactly enough for core + hint

        tags_and_summaries = [
            ("database", "Database schema design."),
        ]
        result = asm.assemble(
            core_context=core,
            retrieval_result=self._make_retrieval_result(tags_and_summaries),
            conversation_history=[],
            token_budget=100_000,
            context_hint=hint,
            max_context_tokens=headroom,
        )
        # Tag sections should be empty (0 budget for tags)
        assert len(result.tag_sections) == 0
        # But the hint should still be in prepend_text
        assert "Topics here" in result.prepend_text


class TestStoreSQLiteGetSegments:
    """Test get_segments_by_tags on SQLite store."""

    def _make_store(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore
        return SQLiteStore(db_path=str(tmp_path / "test.db"))

    def test_get_segments_by_tags_returns_stored(self, tmp_path):
        store = self._make_store(tmp_path)
        seg = StoredSegment(
            ref="seg-1",
            primary_tag="database",
            tags=["database", "sql"],
            summary="Database discussion.",
            summary_tokens=10,
            full_text="Full text of database conversation.",
            full_tokens=20,
        )
        store.store_segment(seg)

        results = store.get_segments_by_tags(tags=["database"])
        assert len(results) >= 1
        found = [r for r in results if r.ref == "seg-1"]
        assert len(found) == 1
        assert found[0].full_text == "Full text of database conversation."

    def test_get_segments_by_tags_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        results = store.get_segments_by_tags(tags=["nonexistent"])
        assert results == []

    def test_get_segments_by_tags_limit(self, tmp_path):
        store = self._make_store(tmp_path)
        for i in range(5):
            store.store_segment(StoredSegment(
                ref=f"seg-{i}", primary_tag="api", tags=["api"],
                summary=f"Summary {i}", summary_tokens=10,
                full_text=f"Full {i}", full_tokens=20,
            ))
        results = store.get_segments_by_tags(tags=["api"], limit=3)
        assert len(results) == 3


class TestStoreFilesystemGetSegments:
    """Test get_segments_by_tags on filesystem store."""

    def _make_store(self, tmp_path):
        from virtual_context.storage.filesystem import FilesystemStore
        return FilesystemStore(root=str(tmp_path / "store"))

    def test_get_segments_by_tags_returns_stored(self, tmp_path):
        store = self._make_store(tmp_path)
        seg = StoredSegment(
            ref="seg-fs-1",
            primary_tag="auth",
            tags=["auth", "security"],
            summary="Auth discussion.",
            summary_tokens=10,
            full_text="Full auth conversation text.",
            full_tokens=20,
        )
        store.store_segment(seg)

        results = store.get_segments_by_tags(tags=["auth"])
        assert len(results) >= 1
        found = [r for r in results if r.ref == "seg-fs-1"]
        assert len(found) == 1
        assert found[0].full_text == "Full auth conversation text."

    def test_get_segments_by_tags_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        results = store.get_segments_by_tags(tags=["nonexistent"])
        assert results == []


# ---------------------------------------------------------------------------
# Phase 3: Engine expand / collapse / working set / eviction
# ---------------------------------------------------------------------------

class TestEnginePagingAPI:
    """Test engine paging API with a real engine using keyword tagger."""

    def _make_engine(self, tmp_path, paging_enabled=True, autonomous_models=None,
                     auto_evict=True, tag_budget=1000):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(
                enabled=paging_enabled,
                autonomous_models=autonomous_models or [],
                auto_evict=auto_evict,
            ),
            assembler=AssemblerConfig(tag_context_max_tokens=tag_budget),
        )
        return VirtualContextEngine(config=cfg)

    def _seed_segments(self, engine, tag, n=3, tokens_per=100):
        """Store segments so expand_topic has content to serve."""
        for i in range(n):
            text = "x" * (tokens_per * 4)  # ~tokens_per tokens with //4 counter
            engine._store.store_segment(StoredSegment(
                ref=f"{tag}-seg-{i}",
                primary_tag=tag,
                tags=[tag],
                summary=f"Summary for {tag} segment {i}.",
                summary_tokens=20,
                full_text=text,
                full_tokens=tokens_per,
            ))
        # Also store a tag summary so SUMMARY depth works
        engine._store.save_tag_summary(TagSummary(
            tag=tag,
            summary=f"Tag summary for {tag}.",
            summary_tokens=30,
            source_segment_refs=[f"{tag}-seg-{i}" for i in range(n)],
            source_turn_numbers=list(range(n)),
        ))

    def test_expand_disabled_returns_error(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=False)
        result = engine.expand_topic("database")
        assert "error" in result
        assert "not enabled" in result["error"]

    def test_collapse_disabled_returns_error(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=False)
        result = engine.collapse_topic("database")
        assert "error" in result

    def test_expand_invalid_depth(self, tmp_path):
        engine = self._make_engine(tmp_path)
        result = engine.expand_topic("database", depth="invalid")
        assert "error" in result
        assert "invalid depth" in result["error"]

    def test_expand_no_content(self, tmp_path):
        engine = self._make_engine(tmp_path)
        result = engine.expand_topic("nonexistent", depth="full")
        assert "error" in result
        assert "no stored content" in result["error"]

    def test_expand_to_full(self, tmp_path):
        engine = self._make_engine(tmp_path, tag_budget=50_000)
        self._seed_segments(engine, "database", n=2, tokens_per=100)

        result = engine.expand_topic("database", depth="full")
        assert "error" not in result
        assert result["tag"] == "database"
        assert result["depth"] == "full"
        assert result["tokens_added"] > 0
        assert "database" in engine._working_set
        assert engine._working_set["database"].depth == DepthLevel.FULL

    def test_expand_to_segments(self, tmp_path):
        engine = self._make_engine(tmp_path, tag_budget=50_000)
        self._seed_segments(engine, "api", n=2, tokens_per=100)

        result = engine.expand_topic("api", depth="segments")
        assert result["depth"] == "segments"
        assert engine._working_set["api"].depth == DepthLevel.SEGMENTS

    def test_expand_to_summary(self, tmp_path):
        engine = self._make_engine(tmp_path, tag_budget=50_000)
        self._seed_segments(engine, "auth", n=1, tokens_per=50)

        result = engine.expand_topic("auth", depth="summary")
        assert result["depth"] == "summary"
        assert engine._working_set["auth"].depth == DepthLevel.SUMMARY

    def test_expand_none_delegates_to_collapse(self, tmp_path):
        engine = self._make_engine(tmp_path, tag_budget=50_000)
        self._seed_segments(engine, "db", n=1, tokens_per=50)

        # First expand to full
        engine.expand_topic("db", depth="full")
        assert "db" in engine._working_set

        # Expand with "none" should collapse
        result = engine.expand_topic("db", depth="none")
        assert result["depth"] == "none"
        assert "db" not in engine._working_set

    def test_collapse_to_summary(self, tmp_path):
        engine = self._make_engine(tmp_path, tag_budget=50_000)
        self._seed_segments(engine, "database", n=2, tokens_per=100)

        # Expand to full first
        engine.expand_topic("database", depth="full")
        old_tokens = engine._working_set["database"].tokens

        # Collapse to summary
        result = engine.collapse_topic("database", depth="summary")
        assert result["tag"] == "database"
        assert result["depth"] == "summary"
        assert result["tokens_freed"] > 0
        assert engine._working_set["database"].depth == DepthLevel.SUMMARY

    def test_collapse_to_none_removes(self, tmp_path):
        engine = self._make_engine(tmp_path, tag_budget=50_000)
        self._seed_segments(engine, "auth", n=1, tokens_per=50)
        engine.expand_topic("auth", depth="full")

        result = engine.collapse_topic("auth", depth="none")
        assert result["depth"] == "none"
        assert "auth" not in engine._working_set

    def test_collapse_nonexistent_tag(self, tmp_path):
        engine = self._make_engine(tmp_path)
        result = engine.collapse_topic("nonexistent", depth="summary")
        assert result["tokens_freed"] == 0

    def test_get_working_set_summary(self, tmp_path):
        engine = self._make_engine(tmp_path, tag_budget=5000)
        self._seed_segments(engine, "db", n=1, tokens_per=50)
        engine.expand_topic("db", depth="summary")

        summary = engine.get_working_set_summary()
        assert "budget" in summary
        assert summary["budget"] == 5000
        assert summary["used"] > 0
        assert summary["available"] < 5000
        assert len(summary["entries"]) == 1
        assert summary["entries"][0]["tag"] == "db"

    def test_auto_evict_on_budget_overflow(self, tmp_path):
        """When expanding over budget, coldest topic gets evicted."""
        engine = self._make_engine(tmp_path, tag_budget=500, auto_evict=True)
        self._seed_segments(engine, "old-topic", n=2, tokens_per=100)
        self._seed_segments(engine, "new-topic", n=2, tokens_per=200)

        # Expand old topic first (lower last_accessed_turn)
        engine.expand_topic("old-topic", depth="full")
        # Now expand new topic which should trigger eviction of old
        result = engine.expand_topic("new-topic", depth="full")

        # old-topic should have been evicted (collapsed or removed)
        if "old-topic" in engine._working_set:
            # Collapsed to summary
            assert engine._working_set["old-topic"].depth == DepthLevel.SUMMARY
        assert result.get("evicted_tags", []) != [] or "error" in result

    def test_no_evict_when_disabled(self, tmp_path):
        """With auto_evict=False, over-budget returns error."""
        engine = self._make_engine(tmp_path, tag_budget=100, auto_evict=False)
        self._seed_segments(engine, "big-topic", n=5, tokens_per=200)

        result = engine.expand_topic("big-topic", depth="full")
        assert "error" in result
        assert "insufficient budget" in result["error"]


class TestEngineAutoEvict:
    """Test the _auto_evict helper."""

    def _make_engine(self, tmp_path, tag_budget=1000):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(enabled=True, auto_evict=True),
            assembler=AssemblerConfig(tag_context_max_tokens=tag_budget),
        )
        return VirtualContextEngine(config=cfg)

    def test_evicts_coldest_first(self, tmp_path):
        engine = self._make_engine(tmp_path)

        # Manually set working set with different access turns
        engine._working_set = {
            "cold": WorkingSetEntry(tag="cold", depth=DepthLevel.FULL, tokens=300, last_accessed_turn=1),
            "warm": WorkingSetEntry(tag="warm", depth=DepthLevel.FULL, tokens=300, last_accessed_turn=5),
            "hot": WorkingSetEntry(tag="hot", depth=DepthLevel.FULL, tokens=300, last_accessed_turn=10),
        }

        evicted, freed = engine._auto_evict(needed=200, exclude_tag="hot")
        # Should evict "cold" first (lowest last_accessed_turn)
        assert "cold" in evicted

    def test_excludes_specified_tag(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine._working_set = {
            "target": WorkingSetEntry(tag="target", depth=DepthLevel.FULL, tokens=500, last_accessed_turn=1),
            "other": WorkingSetEntry(tag="other", depth=DepthLevel.FULL, tokens=300, last_accessed_turn=2),
        }

        evicted, freed = engine._auto_evict(needed=200, exclude_tag="target")
        assert "target" not in evicted
        assert "other" in evicted


class TestCalculateDepthTokens:
    """Test _calculate_depth_tokens helper."""

    def _make_engine(self, tmp_path):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(enabled=True),
        )
        return VirtualContextEngine(config=cfg)

    def test_none_returns_zero(self, tmp_path):
        engine = self._make_engine(tmp_path)
        assert engine._calculate_depth_tokens("any", DepthLevel.NONE) == 0

    def test_summary_uses_tag_summary(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine._store.save_tag_summary(TagSummary(
            tag="db", summary="Database summary.", summary_tokens=42,
        ))
        assert engine._calculate_depth_tokens("db", DepthLevel.SUMMARY) == 42

    def test_summary_missing_returns_zero(self, tmp_path):
        engine = self._make_engine(tmp_path)
        assert engine._calculate_depth_tokens("missing", DepthLevel.SUMMARY) == 0

    def test_segments_sums_segment_summaries(self, tmp_path):
        engine = self._make_engine(tmp_path)
        for i in range(3):
            engine._store.store_segment(StoredSegment(
                ref=f"s{i}", primary_tag="api", tags=["api"],
                summary=f"Summary {i}", summary_tokens=10 * (i + 1),
                full_text="text", full_tokens=100,
            ))
        # Should sum summary_tokens: 10 + 20 + 30 = 60
        assert engine._calculate_depth_tokens("api", DepthLevel.SEGMENTS) == 60

    def test_full_sums_full_tokens(self, tmp_path):
        engine = self._make_engine(tmp_path)
        for i in range(2):
            engine._store.store_segment(StoredSegment(
                ref=f"f{i}", primary_tag="auth", tags=["auth"],
                summary="s", summary_tokens=5,
                full_text="x" * 400, full_tokens=100,
            ))
        assert engine._calculate_depth_tokens("auth", DepthLevel.FULL) == 200


# ---------------------------------------------------------------------------
# Phase 4: Config parsing + Hint modes
# ---------------------------------------------------------------------------

class TestPagingConfigParsing:
    def test_default_config(self):
        cfg = _build_config({})
        assert cfg.paging.enabled is False
        assert cfg.paging.autonomous_models == ["opus", "sonnet", "gpt-4", "gpt-4o"]
        assert cfg.paging.auto_promote is True
        assert cfg.paging.auto_evict is True

    def test_custom_paging_config(self):
        cfg = _build_config({
            "paging": {
                "enabled": True,
                "autonomous_models": ["opus"],
                "auto_promote": False,
                "auto_evict": False,
            }
        })
        assert cfg.paging.enabled is True
        assert cfg.paging.autonomous_models == ["opus"]
        assert cfg.paging.auto_promote is False
        assert cfg.paging.auto_evict is False

    def test_autonomous_models_validation(self):
        from virtual_context.config import validate_config
        cfg = _build_config({"paging": {"autonomous_models": "not-a-list"}})
        errors = validate_config(cfg)
        model_errors = [e for e in errors if "autonomous_models" in e]
        assert len(model_errors) == 1


class TestResolvePagingMode:
    """Test _resolve_paging_mode with autonomous_models list."""

    def _make_engine(self, tmp_path, autonomous_models=None):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(
                enabled=True,
                autonomous_models=autonomous_models if autonomous_models is not None else [],
            ),
        )
        return VirtualContextEngine(config=cfg)

    def test_empty_list_always_supervised(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=[])
        assert engine._resolve_paging_mode("claude-opus-4") == "supervised"

    def test_opus_matches(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=["opus", "sonnet"])
        assert engine._resolve_paging_mode("claude-opus-4") == "autonomous"

    def test_sonnet_matches(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=["opus", "sonnet"])
        assert engine._resolve_paging_mode("claude-sonnet-4") == "autonomous"

    def test_gpt4_matches(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=["gpt-4"])
        assert engine._resolve_paging_mode("gpt-4-turbo") == "autonomous"

    def test_haiku_not_in_list(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=["opus", "sonnet"])
        assert engine._resolve_paging_mode("claude-haiku-3") == "supervised"

    def test_unknown_model_supervised(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=["opus", "sonnet"])
        assert engine._resolve_paging_mode("qwen3:4b") == "supervised"

    def test_empty_model_name_supervised(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=["opus", "sonnet"])
        assert engine._resolve_paging_mode("") == "supervised"

    def test_case_insensitive(self, tmp_path):
        engine = self._make_engine(tmp_path, autonomous_models=["Sonnet"])
        assert engine._resolve_paging_mode("claude-sonnet-4") == "autonomous"


class TestContextHintModes:
    """Test _build_context_hint with different paging modes."""

    def _make_engine(self, tmp_path, paging_enabled=False):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(enabled=paging_enabled),
            assembler=AssemblerConfig(
                context_hint_enabled=True,
                context_hint_max_tokens=500,
                tag_context_max_tokens=10_000,
            ),
        )
        engine = VirtualContextEngine(config=cfg)
        # Simulate post-compaction state
        engine._compacted_through = 4
        return engine

    def _seed_tag_summary(self, engine, tag, summary="Discussion about topic."):
        engine._store.save_tag_summary(TagSummary(
            tag=tag,
            summary=summary,
            summary_tokens=len(summary) // 4,
            source_turn_numbers=[0, 1, 2],
        ))

    def test_default_hint_no_paging(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=False)
        self._seed_tag_summary(engine, "database")
        hint = engine._build_context_hint()
        assert "<context-topics>" in hint
        assert "database" in hint
        # Default hint should NOT mention expand_topic
        assert "expand_topic" not in hint

    def test_supervised_hint(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=True)
        self._seed_tag_summary(engine, "api")
        hint = engine._build_context_hint(paging_mode="supervised")
        assert "expand_topic" in hint
        assert "api" in hint

    def test_autonomous_hint_has_budget(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=True)
        self._seed_tag_summary(engine, "auth")
        hint = engine._build_context_hint(paging_mode="autonomous")
        assert "budget=" in hint
        assert "available=" in hint
        assert "expand_topic" in hint
        assert "collapse_topic" in hint

    def test_hint_empty_before_compaction(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=True)
        engine._compacted_through = 0  # no compaction yet
        self._seed_tag_summary(engine, "api")
        hint = engine._build_context_hint(paging_mode="supervised")
        assert hint == ""

    @pytest.mark.regression("PROXY-024")
    def test_autonomous_hint_expanded_tags_listed_first(self, tmp_path):
        """Tags at summary depth (in working set) must appear before depth:none tags."""
        engine = self._make_engine(tmp_path, paging_enabled=True)
        # Seed 30 tags alphabetically — a-tag-01 through a-tag-30
        for i in range(1, 31):
            self._seed_tag_summary(engine, f"a-tag-{i:02d}")
        # Put z-fragrance in working set at summary depth
        self._seed_tag_summary(engine, "z-fragrance")
        engine._working_set["z-fragrance"] = WorkingSetEntry(
            tag="z-fragrance", depth=DepthLevel.SUMMARY,
            tokens=200, last_accessed_turn=5,
        )
        hint = engine._build_context_hint(paging_mode="autonomous")
        # z-fragrance should appear despite being last alphabetically
        assert "z-fragrance" in hint
        # And it should appear BEFORE the depth:none tags
        zf_pos = hint.index("z-fragrance")
        # At least one a-tag should also appear — check a-tag-01
        if "a-tag-01" in hint:
            at_pos = hint.index("a-tag-01")
            assert zf_pos < at_pos, "Expanded tag must appear before depth:none tags"

    @pytest.mark.regression("PROXY-024")
    def test_autonomous_hint_compact_format_fits_more_tags(self, tmp_path):
        """With compact format, 50+ tags should fit in 500 token budget."""
        engine = self._make_engine(tmp_path, paging_enabled=True)
        # Seed 50 tags
        for i in range(50):
            self._seed_tag_summary(engine, f"topic-{i:03d}")
        hint = engine._build_context_hint(paging_mode="autonomous")
        # Count how many topic- tags appear in the hint
        count = sum(1 for i in range(50) if f"topic-{i:03d}" in hint)
        # With compact format, we should fit significantly more than 9
        assert count >= 25, f"Only {count}/50 tags fit — compact format not working"

    @pytest.mark.regression("PROXY-024")
    def test_autonomous_hint_truncation_drops_none_first(self, tmp_path):
        """When truncating, depth:none tags are dropped before expanded tags."""
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(enabled=True),
            assembler=AssemblerConfig(
                context_hint_enabled=True,
                context_hint_max_tokens=200,  # Very tight budget
                tag_context_max_tokens=10_000,
            ),
        )
        engine = VirtualContextEngine(config=cfg)
        engine._compacted_through = 4
        # Seed 80 tags at depth:none
        for i in range(80):
            self._seed_tag_summary(engine, f"filler-{i:03d}")
        # Put 3 tags in working set at summary depth
        for name in ["fragrance-selection", "karak-chai", "whales"]:
            self._seed_tag_summary(engine, name)
            engine._working_set[name] = WorkingSetEntry(
                tag=name, depth=DepthLevel.SUMMARY,
                tokens=100, last_accessed_turn=5,
            )
        hint = engine._build_context_hint(paging_mode="autonomous")
        # All 3 expanded tags MUST survive truncation
        assert "fragrance-selection" in hint
        assert "karak-chai" in hint
        assert "whales" in hint

    @pytest.mark.regression("PROXY-024")
    def test_supervised_hint_compact_format(self, tmp_path):
        """Supervised mode also uses compact format."""
        engine = self._make_engine(tmp_path, paging_enabled=True)
        for i in range(50):
            self._seed_tag_summary(engine, f"topic-{i:03d}")
        hint = engine._build_context_hint(paging_mode="supervised")
        count = sum(1 for i in range(50) if f"topic-{i:03d}" in hint)
        assert count >= 25, f"Only {count}/50 tags fit in supervised mode"

    def test_autonomous_hint_includes_description(self, tmp_path):
        """Autonomous hint includes ts.description when available."""
        engine = self._make_engine(tmp_path, paging_enabled=True)
        engine._store.save_tag_summary(TagSummary(
            tag="cycle-tracking",
            summary="Discussed Mira device for cycle tracking.",
            description="Sania's cycle tracking via Mira",
            summary_tokens=30,
            source_turn_numbers=[0, 1, 2],
        ))
        hint = engine._build_context_hint(paging_mode="autonomous")
        assert "Sania's cycle tracking via Mira" in hint

    def test_autonomous_hint_omits_description_when_empty(self, tmp_path):
        """Autonomous hint does not include ' — ' when description is empty."""
        engine = self._make_engine(tmp_path, paging_enabled=True)
        engine._store.save_tag_summary(TagSummary(
            tag="database",
            summary="Database discussion.",
            description="",
            summary_tokens=20,
            source_turn_numbers=[0, 1],
        ))
        hint = engine._build_context_hint(paging_mode="autonomous")
        assert "database" in hint
        # The " — " separator should not appear since description is empty
        # The tag should appear without a trailing description
        db_line = [line for line in hint.split("\n") if "database" in line][0]
        assert " — " not in db_line

    def test_supervised_hint_includes_description(self, tmp_path):
        """Supervised hint includes ts.description for available tags."""
        engine = self._make_engine(tmp_path, paging_enabled=True)
        engine._store.save_tag_summary(TagSummary(
            tag="meal-planning",
            summary="Discussed weekly meal prep and grocery lists.",
            description="Weekly meal prep and grocery optimization",
            summary_tokens=25,
            source_turn_numbers=[0, 1, 2],
        ))
        hint = engine._build_context_hint(paging_mode="supervised")
        assert "Weekly meal prep and grocery optimization" in hint

    def test_default_hint_uses_description(self, tmp_path):
        """Default hint (no paging) uses ts.description when available."""
        engine = self._make_engine(tmp_path, paging_enabled=False)
        engine._store.save_tag_summary(TagSummary(
            tag="fitness",
            summary="Running program with interval training and heart rate zones discussed at length.",
            description="Interval training and HR zone programming",
            summary_tokens=30,
            source_turn_numbers=[0, 1, 2, 3],
        ))
        hint = engine._build_context_hint()
        # Default hint uses description when available instead of summary[:60]
        assert "Interval training and HR zone programming" in hint


# ---------------------------------------------------------------------------
# Phase 5: MCP Tools
# ---------------------------------------------------------------------------

class TestMCPPagingTools:
    """Test MCP expand_topic and collapse_topic tools."""

    def _mock_engine(self):
        engine = MagicMock()
        engine.expand_topic.return_value = {
            "tag": "database",
            "depth": "full",
            "tokens_added": 5000,
            "tokens_evicted": 0,
            "evicted_tags": [],
        }
        engine.collapse_topic.return_value = {
            "tag": "database",
            "depth": "summary",
            "tokens_freed": 4500,
        }
        return engine

    @patch("virtual_context.mcp.server._get_engine")
    def test_expand_topic_tool(self, mock_get_engine):
        mock_get_engine.return_value = self._mock_engine()

        from virtual_context.mcp.server import expand_topic
        result = expand_topic("database", "full")
        data = json.loads(result)
        assert data["tag"] == "database"
        assert data["depth"] == "full"
        assert data["tokens_added"] == 5000

    @patch("virtual_context.mcp.server._get_engine")
    def test_collapse_topic_tool(self, mock_get_engine):
        mock_get_engine.return_value = self._mock_engine()

        from virtual_context.mcp.server import collapse_topic
        result = collapse_topic("database", "summary")
        data = json.loads(result)
        assert data["tag"] == "database"
        assert data["depth"] == "summary"
        assert data["tokens_freed"] == 4500

    @patch("virtual_context.mcp.server._get_engine")
    def test_expand_calls_engine(self, mock_get_engine):
        engine = self._mock_engine()
        mock_get_engine.return_value = engine

        from virtual_context.mcp.server import expand_topic
        expand_topic("api", "segments")
        engine.expand_topic.assert_called_once_with("api", "segments")

    @patch("virtual_context.mcp.server._get_engine")
    def test_collapse_calls_engine(self, mock_get_engine):
        engine = self._mock_engine()
        mock_get_engine.return_value = engine

        from virtual_context.mcp.server import collapse_topic
        collapse_topic("api", "none")
        engine.collapse_topic.assert_called_once_with("api", "none")

    @patch("virtual_context.mcp.server._get_engine")
    def test_domain_status_still_works(self, mock_get_engine):
        """Existing domain_status tool should still work after paging additions."""
        engine = MagicMock()
        engine._store.get_all_tags.return_value = [
            __import__("virtual_context.types", fromlist=["TagStats"]).TagStats(
                tag="database", usage_count=5,
            ),
        ]
        mock_get_engine.return_value = engine

        from virtual_context.mcp.server import domain_status
        result = domain_status()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["tag"] == "database"


# ---------------------------------------------------------------------------
# Integration: state persistence roundtrip
# ---------------------------------------------------------------------------

class TestPagingStatePersistence:
    """Test that working set survives engine state save/load."""

    def _make_engine(self, tmp_path):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(enabled=True),
        )
        return VirtualContextEngine(config=cfg)

    def test_working_set_persists_through_save_load(self, tmp_path):
        engine = self._make_engine(tmp_path)

        # Seed content and expand
        engine._store.store_segment(StoredSegment(
            ref="seg-1", primary_tag="db", tags=["db"],
            summary="DB summary", summary_tokens=20,
            full_text="Full text", full_tokens=50,
        ))
        engine._store.save_tag_summary(TagSummary(
            tag="db", summary="Tag summary", summary_tokens=20,
        ))
        engine.expand_topic("db", depth="full")
        assert "db" in engine._working_set

        # Save state
        engine._save_state([])

        # Create new engine from same DB — should restore working set
        from virtual_context.engine import VirtualContextEngine
        engine2 = VirtualContextEngine(config=engine.config)
        assert "db" in engine2._working_set
        assert engine2._working_set["db"].depth == DepthLevel.FULL
        assert engine2._working_set["db"].tokens > 0


# ---------------------------------------------------------------------------
# Phase 6: reassemble_context
# ---------------------------------------------------------------------------


class TestReassembleContext:
    """Tests for engine.reassemble_context()."""

    def _make_engine(self, tmp_path):
        """Create a paging-enabled engine with stored content."""
        cfg = load_config(config_dict={
            "context_window": 50000,
            "storage_root": str(tmp_path),
            "storage": {
                "backend": "sqlite",
                "sqlite": {"path": str(tmp_path / "store.db")},
            },
            "tag_generator": {"type": "keyword"},
            "paging": {"enabled": True, "autonomous_models": ["opus", "sonnet"]},
            "assembly": {"context_hint_enabled": True},
        })
        from virtual_context.engine import VirtualContextEngine
        engine = VirtualContextEngine(config=cfg)

        # Populate store with a tag summary + segments
        engine._store.save_tag_summary(TagSummary(
            tag="database",
            summary="Discussed PostgreSQL schema design and indexing.",
            summary_tokens=30,
        ))
        engine._store.store_segment(StoredSegment(
            ref="seg1",
            primary_tag="database",
            tags=["database"],
            summary="PostgreSQL schema discussion.",
            summary_tokens=20,
            full_text="User asked about database indexing. Assistant explained B-tree indexes.",
            full_tokens=50,
        ))

        # Simulate post-compaction state so context hint is generated
        engine._compacted_through = 2
        engine._turn_tag_index.append(TurnTagEntry(
            turn_number=1, message_hash="abc123", tags=["database"], primary_tag="database",
        ))

        return engine

    def test_returns_empty_before_any_inbound(self, tmp_path):
        """Before on_message_inbound, reassemble returns empty."""
        engine = self._make_engine(tmp_path)
        assert engine.reassemble_context() == ""

    def test_returns_content_after_inbound(self, tmp_path):
        """After on_message_inbound, reassemble returns prepend_text."""
        engine = self._make_engine(tmp_path)
        history = [
            Message(role="user", content="Tell me about databases"),
            Message(role="assistant", content="Sure, PostgreSQL..."),
        ]
        assembled = engine.on_message_inbound("What about indexing?", history)
        assert assembled.prepend_text  # has content

        text = engine.reassemble_context()
        assert text  # also has content
        assert "database" in text.lower() or "PostgreSQL" in text.lower()

    def test_reflects_expanded_depth(self, tmp_path):
        """After expand_topic, reassemble includes expanded content."""
        engine = self._make_engine(tmp_path)
        history = [
            Message(role="user", content="Tell me about databases"),
            Message(role="assistant", content="Sure, PostgreSQL..."),
        ]
        engine.on_message_inbound("What about indexing?", history)

        # Working set starts at SUMMARY depth
        initial = engine.reassemble_context()

        # Expand to FULL
        engine.expand_topic("database", "full")

        # Re-assemble should now include full text
        expanded = engine.reassemble_context()
        assert "B-tree indexes" in expanded  # full_text content
        # Initial (summary) should NOT have had the full text
        assert "B-tree indexes" not in initial


# ---------------------------------------------------------------------------
# BUG-014 / BUG-015 / BUG-016: Broad & temporal queries must bypass working set
# ---------------------------------------------------------------------------


class TestBroadBypassesWorkingSet:
    """BUG-014: Broad queries must flatten working set to SUMMARY depth.

    When a user asks 'summarize everything' and 2 tags are at FULL depth,
    those tags eat the budget and 15+ topic summaries get dropped.
    Fix: pass working_set=None to assembler when retrieval_result.broad is True.
    """

    def _make_engine(self, tmp_path, tag_budget=2000):
        cfg = load_config(config_dict={
            "context_window": 100_000,
            "storage_root": str(tmp_path),
            "storage": {
                "backend": "sqlite",
                "sqlite": {"path": str(tmp_path / "store.db")},
            },
            "tag_generator": {"type": "keyword"},
            "paging": {"enabled": True, "autonomous_models": []},
            "assembly": {
                "tag_context_max_tokens": tag_budget,
                "context_hint_enabled": False,
            },
        })
        from virtual_context.engine import VirtualContextEngine
        engine = VirtualContextEngine(config=cfg)
        engine._compacted_through = 2
        return engine

    def _seed_tag(self, engine, tag, summary_text="Summary.", full_text=None):
        """Store a tag summary and optionally a segment with full_text."""
        engine._store.save_tag_summary(TagSummary(
            tag=tag,
            summary=summary_text,
            summary_tokens=len(summary_text) // 4,
            source_turn_numbers=[0],
        ))
        if full_text:
            engine._store.store_segment(StoredSegment(
                ref=f"seg-{tag}",
                primary_tag=tag,
                tags=[tag],
                summary=summary_text,
                summary_tokens=len(summary_text) // 4,
                full_text=full_text,
                full_tokens=len(full_text) // 4,
            ))

    @pytest.mark.regression("BUG-014")
    def test_broad_query_gets_all_summaries_despite_expanded_tags(self, tmp_path):
        """With 2 tags at FULL depth eating budget, broad must still show all topics."""
        # Tight budget: 2000 tokens. FULL-depth tags would eat it all.
        engine = self._make_engine(tmp_path, tag_budget=2000)

        # Seed 10 tags with summaries (~50 tokens each = 500t total at SUMMARY)
        for i in range(10):
            self._seed_tag(
                engine, f"topic-{i}",
                summary_text=f"Summary of topic {i} with enough text to be meaningful. " * 2,
                full_text=f"Full detailed text for topic {i}. " * 50,  # ~600t each
            )

        # Expand 2 tags to FULL depth — each ~600t, total ~1200t, leaving only 800t
        engine._working_set = {
            "topic-0": WorkingSetEntry(tag="topic-0", depth=DepthLevel.FULL, tokens=600),
            "topic-1": WorkingSetEntry(tag="topic-1", depth=DepthLevel.FULL, tokens=600),
        }

        # Mock retriever to return broad=True with all 10 tag summaries
        all_summaries = []
        for i in range(10):
            ts = engine._store.get_tag_summary(f"topic-{i}")
            all_summaries.append(StoredSummary(
                ref=f"seg-topic-{i}",
                primary_tag=f"topic-{i}",
                tags=[f"topic-{i}"],
                summary=ts.summary,
                summary_tokens=ts.summary_tokens,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ))

        broad_result = RetrievalResult(
            tags_matched=[f"topic-{i}" for i in range(10)],
            summaries=all_summaries,
            broad=True,
        )

        with patch.object(engine._retriever, "retrieve", return_value=broad_result):
            assembled = engine.on_message_inbound(
                "Summarize everything we've discussed",
                [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
            )

        # ALL 10 topics should appear as summaries — not just the 2 expanded ones
        assert len(assembled.tag_sections) >= 8, (
            f"Broad query should show most topics but only got {len(assembled.tag_sections)}: "
            f"{list(assembled.tag_sections.keys())}"
        )

    @pytest.mark.regression("BUG-014")
    def test_broad_query_does_not_render_full_depth(self, tmp_path):
        """Broad queries must not render any tag at FULL depth."""
        engine = self._make_engine(tmp_path, tag_budget=5000)
        self._seed_tag(
            engine, "recipes",
            summary_text="Discussed various recipe ideas.",
            full_text="Very long recipe discussion. " * 100,  # ~750t
        )
        engine._working_set = {
            "recipes": WorkingSetEntry(tag="recipes", depth=DepthLevel.FULL, tokens=750),
        }

        broad_result = RetrievalResult(
            tags_matched=["recipes"],
            summaries=[StoredSummary(
                ref="seg-recipes", primary_tag="recipes", tags=["recipes"],
                summary="Discussed various recipe ideas.",
                summary_tokens=20,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )],
            broad=True,
        )

        with patch.object(engine._retriever, "retrieve", return_value=broad_result):
            assembled = engine.on_message_inbound(
                "What have we covered so far?",
                [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
            )

        # Should NOT contain depth="full" — broad flattens to SUMMARY
        if "recipes" in assembled.tag_sections:
            assert 'depth="full"' not in assembled.tag_sections["recipes"], (
                "Broad query rendered a tag at FULL depth — should be SUMMARY"
            )

    @pytest.mark.regression("BUG-014")
    def test_working_set_preserved_after_broad_query(self, tmp_path):
        """Broad query flattens assembly but must NOT modify the working set."""
        engine = self._make_engine(tmp_path, tag_budget=5000)
        self._seed_tag(engine, "auth", summary_text="Auth discussion.", full_text="Full auth text.")
        engine._working_set = {
            "auth": WorkingSetEntry(tag="auth", depth=DepthLevel.FULL, tokens=500, last_accessed_turn=5),
        }

        broad_result = RetrievalResult(
            tags_matched=["auth"],
            summaries=[StoredSummary(
                ref="seg-auth", primary_tag="auth", tags=["auth"],
                summary="Auth discussion.", summary_tokens=10,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )],
            broad=True,
        )

        with patch.object(engine._retriever, "retrieve", return_value=broad_result):
            engine.on_message_inbound(
                "Summarize everything",
                [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
            )

        # Working set must be unmodified — still FULL depth
        assert "auth" in engine._working_set
        assert engine._working_set["auth"].depth == DepthLevel.FULL


class TestTemporalBypassesWorkingSet:
    """BUG-015: Temporal queries must skip working set depth override.

    Temporal queries retrieve chronologically-sorted segment summaries,
    but paging overrides with unsorted full_segments from get_segments_by_tags().
    """

    def _make_engine(self, tmp_path):
        cfg = load_config(config_dict={
            "context_window": 100_000,
            "storage_root": str(tmp_path),
            "storage": {
                "backend": "sqlite",
                "sqlite": {"path": str(tmp_path / "store.db")},
            },
            "tag_generator": {"type": "keyword"},
            "paging": {"enabled": True, "autonomous_models": []},
            "assembly": {
                "tag_context_max_tokens": 5000,
                "context_hint_enabled": False,
            },
        })
        from virtual_context.engine import VirtualContextEngine
        engine = VirtualContextEngine(config=cfg)
        engine._compacted_through = 2
        return engine

    @pytest.mark.regression("BUG-015")
    def test_temporal_query_ignores_working_set_depth(self, tmp_path):
        """Temporal query must render at SUMMARY depth even if working set says FULL."""
        engine = self._make_engine(tmp_path)

        # Tag in working set at FULL depth
        engine._store.save_tag_summary(TagSummary(
            tag="architecture",
            summary="Discussed system architecture.",
            summary_tokens=20,
            source_turn_numbers=[0, 1],
        ))
        engine._store.store_segment(StoredSegment(
            ref="seg-arch",
            primary_tag="architecture",
            tags=["architecture"],
            summary="Architecture discussion.",
            summary_tokens=15,
            full_text="Detailed architecture text with all implementation specifics. " * 20,
            full_tokens=300,
        ))
        engine._working_set = {
            "architecture": WorkingSetEntry(
                tag="architecture", depth=DepthLevel.FULL, tokens=300, last_accessed_turn=3,
            ),
        }

        temporal_result = RetrievalResult(
            tags_matched=["architecture"],
            summaries=[StoredSummary(
                ref="seg-arch", primary_tag="architecture", tags=["architecture"],
                summary="Architecture discussion.",
                summary_tokens=15,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )],
            temporal=True,
        )

        with patch.object(engine._retriever, "retrieve", return_value=temporal_result):
            assembled = engine.on_message_inbound(
                "What did we first discuss about architecture?",
                [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
            )

        # Should NOT be at FULL depth — temporal preserves retriever's chronological order
        if "architecture" in assembled.tag_sections:
            assert 'depth="full"' not in assembled.tag_sections["architecture"], (
                "Temporal query rendered at FULL depth — should preserve retriever's sort order"
            )


class TestSegmentLoadingGate:
    """BUG-016: Segment loading should be skipped for broad/temporal queries."""

    def _make_engine(self, tmp_path):
        cfg = load_config(config_dict={
            "context_window": 100_000,
            "storage_root": str(tmp_path),
            "storage": {
                "backend": "sqlite",
                "sqlite": {"path": str(tmp_path / "store.db")},
            },
            "tag_generator": {"type": "keyword"},
            "paging": {"enabled": True, "autonomous_models": []},
            "assembly": {
                "tag_context_max_tokens": 5000,
                "context_hint_enabled": False,
            },
        })
        from virtual_context.engine import VirtualContextEngine
        engine = VirtualContextEngine(config=cfg)
        engine._compacted_through = 2
        return engine

    @pytest.mark.regression("BUG-016")
    def test_broad_query_skips_segment_loading(self, tmp_path):
        """Broad query must not call get_segments_by_tags for working set tags."""
        engine = self._make_engine(tmp_path)

        engine._store.save_tag_summary(TagSummary(
            tag="db", summary="Database discussion.", summary_tokens=15,
        ))
        engine._working_set = {
            "db": WorkingSetEntry(tag="db", depth=DepthLevel.FULL, tokens=500),
        }

        broad_result = RetrievalResult(
            tags_matched=["db"],
            summaries=[StoredSummary(
                ref="seg-db", primary_tag="db", tags=["db"],
                summary="Database discussion.", summary_tokens=15,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )],
            broad=True,
        )

        with patch.object(engine._retriever, "retrieve", return_value=broad_result), \
             patch.object(engine._store, "get_segments_by_tags", wraps=engine._store.get_segments_by_tags) as mock_get:
            engine.on_message_inbound(
                "Summarize everything",
                [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
            )

        # get_segments_by_tags should NOT have been called for working set loading
        mock_get.assert_not_called()

    @pytest.mark.regression("BUG-016")
    def test_temporal_query_skips_segment_loading(self, tmp_path):
        """Temporal query must not call get_segments_by_tags for working set tags."""
        engine = self._make_engine(tmp_path)

        engine._store.save_tag_summary(TagSummary(
            tag="api", summary="API discussion.", summary_tokens=15,
        ))
        engine._working_set = {
            "api": WorkingSetEntry(tag="api", depth=DepthLevel.FULL, tokens=500),
        }

        temporal_result = RetrievalResult(
            tags_matched=["api"],
            summaries=[StoredSummary(
                ref="seg-api", primary_tag="api", tags=["api"],
                summary="API discussion.", summary_tokens=15,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )],
            temporal=True,
        )

        with patch.object(engine._retriever, "retrieve", return_value=temporal_result), \
             patch.object(engine._store, "get_segments_by_tags", wraps=engine._store.get_segments_by_tags) as mock_get:
            engine.on_message_inbound(
                "What did we discuss first?",
                [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
            )

        mock_get.assert_not_called()

    @pytest.mark.regression("BUG-016")
    def test_normal_query_still_loads_segments(self, tmp_path):
        """Normal (non-broad, non-temporal) queries must still load segments."""
        engine = self._make_engine(tmp_path)

        engine._store.save_tag_summary(TagSummary(
            tag="auth", summary="Auth discussion.", summary_tokens=15,
        ))
        engine._store.store_segment(StoredSegment(
            ref="seg-auth", primary_tag="auth", tags=["auth"],
            summary="Auth segment.", summary_tokens=10,
            full_text="Full auth text.", full_tokens=20,
        ))
        engine._working_set = {
            "auth": WorkingSetEntry(tag="auth", depth=DepthLevel.FULL, tokens=500),
        }

        normal_result = RetrievalResult(
            tags_matched=["auth"],
            summaries=[StoredSummary(
                ref="seg-auth", primary_tag="auth", tags=["auth"],
                summary="Auth discussion.", summary_tokens=15,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )],
            broad=False,
            temporal=False,
        )

        with patch.object(engine._retriever, "retrieve", return_value=normal_result), \
             patch.object(engine._store, "get_segments_by_tags", wraps=engine._store.get_segments_by_tags) as mock_get:
            engine.on_message_inbound(
                "Tell me about auth",
                [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
            )

        # Normal query SHOULD load segments for FULL-depth working set tags
        mock_get.assert_called()
