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
        assert cfg.mode == "auto"
        assert cfg.auto_promote is True
        assert cfg.auto_evict is True

    def test_custom_values(self):
        cfg = PagingConfig(enabled=True, mode="autonomous", auto_evict=False)
        assert cfg.enabled is True
        assert cfg.mode == "autonomous"
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

    def _make_engine(self, tmp_path, paging_enabled=True, paging_mode="supervised",
                     auto_evict=True, tag_budget=1000):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(
                enabled=paging_enabled,
                mode=paging_mode,
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
        assert cfg.paging.mode == "auto"
        assert cfg.paging.auto_promote is True
        assert cfg.paging.auto_evict is True

    def test_custom_paging_config(self):
        cfg = _build_config({
            "paging": {
                "enabled": True,
                "mode": "autonomous",
                "auto_promote": False,
                "auto_evict": False,
            }
        })
        assert cfg.paging.enabled is True
        assert cfg.paging.mode == "autonomous"
        assert cfg.paging.auto_promote is False
        assert cfg.paging.auto_evict is False

    def test_paging_mode_validation(self):
        from virtual_context.config import validate_config
        cfg = _build_config({"paging": {"mode": "invalid_mode"}})
        errors = validate_config(cfg)
        mode_errors = [e for e in errors if "paging.mode" in e]
        assert len(mode_errors) == 1


class TestResolvePagingMode:
    """Test _resolve_paging_mode model-name mapping."""

    def _make_engine(self, tmp_path, mode="auto"):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(enabled=True, mode=mode),
        )
        return VirtualContextEngine(config=cfg)

    def test_explicit_supervised(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="supervised")
        assert engine._resolve_paging_mode() == "supervised"

    def test_explicit_autonomous(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="autonomous")
        assert engine._resolve_paging_mode() == "autonomous"

    def test_auto_opus_is_autonomous(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="auto")
        assert engine._resolve_paging_mode("claude-opus-4") == "autonomous"

    def test_auto_sonnet_is_autonomous(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="auto")
        assert engine._resolve_paging_mode("claude-sonnet-4") == "autonomous"

    def test_auto_gpt4_is_autonomous(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="auto")
        assert engine._resolve_paging_mode("gpt-4-turbo") == "autonomous"

    def test_auto_haiku_is_supervised(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="auto")
        assert engine._resolve_paging_mode("claude-haiku-3") == "supervised"

    def test_auto_unknown_is_supervised(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="auto")
        assert engine._resolve_paging_mode("qwen3:4b") == "supervised"

    def test_auto_empty_string_is_supervised(self, tmp_path):
        engine = self._make_engine(tmp_path, mode="auto")
        assert engine._resolve_paging_mode("") == "supervised"


class TestContextHintModes:
    """Test _build_context_hint with different paging modes."""

    def _make_engine(self, tmp_path, paging_enabled=False, mode="auto"):
        from virtual_context.engine import VirtualContextEngine
        cfg = VirtualContextConfig(
            storage=__import__("virtual_context.types", fromlist=["StorageConfig"]).StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
            ),
            paging=PagingConfig(enabled=paging_enabled, mode=mode),
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
        engine = self._make_engine(tmp_path, paging_enabled=True, mode="supervised")
        self._seed_tag_summary(engine, "api")
        hint = engine._build_context_hint()
        assert "expand_topic" in hint
        assert "api" in hint

    def test_autonomous_hint_has_budget(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=True, mode="autonomous")
        self._seed_tag_summary(engine, "auth")
        hint = engine._build_context_hint()
        assert "budget=" in hint
        assert "available=" in hint
        assert "expand_topic" in hint
        assert "collapse_topic" in hint

    def test_hint_empty_before_compaction(self, tmp_path):
        engine = self._make_engine(tmp_path, paging_enabled=True, mode="supervised")
        engine._compacted_through = 0  # no compaction yet
        self._seed_tag_summary(engine, "api")
        hint = engine._build_context_hint()
        assert hint == ""


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
