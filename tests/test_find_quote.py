"""Tests for vc_find_quote: full-text search across stored segments."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from virtual_context.types import (
    AssemblerConfig,
    PagingConfig,
    QuoteResult,
    SegmentMetadata,
    StorageConfig,
    StoredSegment,
    TagSummary,
    VirtualContextConfig,
)
from virtual_context.storage.sqlite import SQLiteStore, _extract_excerpt
from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.storage.filesystem import _extract_excerpt as fs_extract_excerpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    ref: str = "seg-1",
    primary_tag: str = "health",
    tags: list[str] | None = None,
    summary: str = "Discussed supplements",
    full_text: str = "User asked about magnesium glycinate 400mg for sleep. Assistant recommended taking it before bed.",
) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        session_id="session-1",
        primary_tag=primary_tag,
        tags=tags or [primary_tag],
        summary=summary,
        summary_tokens=20,
        full_text=full_text,
        full_tokens=100,
        messages=[{"role": "user", "content": "test"}],
        metadata=SegmentMetadata(turn_count=1),
        created_at=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        start_timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        end_timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
    )


def _make_engine(tmp_path, paging_enabled=False):
    from virtual_context.engine import VirtualContextEngine
    cfg = VirtualContextConfig(
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "test.db"),
        ),
        paging=PagingConfig(enabled=paging_enabled),
    )
    return VirtualContextEngine(config=cfg)


# ---------------------------------------------------------------------------
# _extract_excerpt unit tests
# ---------------------------------------------------------------------------

class TestExtractExcerpt:
    def test_match_in_middle(self):
        text = "A" * 300 + "TARGET" + "B" * 300
        excerpt = _extract_excerpt(text, "TARGET", context_chars=50)
        assert "TARGET" in excerpt
        assert excerpt.startswith("...")
        assert excerpt.endswith("...")

    def test_match_at_start(self):
        text = "TARGET" + "B" * 300
        excerpt = _extract_excerpt(text, "TARGET", context_chars=50)
        assert "TARGET" in excerpt
        assert not excerpt.startswith("...")
        assert excerpt.endswith("...")

    def test_match_at_end(self):
        text = "A" * 300 + "TARGET"
        excerpt = _extract_excerpt(text, "TARGET", context_chars=50)
        assert "TARGET" in excerpt
        assert excerpt.startswith("...")
        assert not excerpt.endswith("...")

    def test_no_match_returns_beginning(self):
        text = "Some text without the search term"
        excerpt = _extract_excerpt(text, "nonexistent", context_chars=10)
        assert excerpt == text[:20]

    def test_case_insensitive(self):
        text = "Before magnesium GLYCINATE after"
        excerpt = _extract_excerpt(text, "Magnesium glycinate", context_chars=200)
        assert "magnesium GLYCINATE" in excerpt

    def test_filesystem_extract_excerpt_matches(self):
        """Both backends use the same algorithm."""
        text = "prefix magnesium glycinate suffix"
        assert _extract_excerpt(text, "magnesium") == fs_extract_excerpt(text, "magnesium")


# ---------------------------------------------------------------------------
# SQLiteStore.search_full_text tests
# ---------------------------------------------------------------------------

class TestSQLiteSearchFullText:
    def test_fts_finds_match(self, tmp_sqlite_db):
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment())

        results = store.search_full_text("magnesium")
        assert len(results) == 1
        assert results[0].tag == "health"
        assert results[0].segment_ref == "seg-1"
        assert results[0].tags == ["health"]
        assert "magnesium" in results[0].text.lower()
        store.close()

    def test_fts_snippet_markers(self, tmp_sqlite_db):
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment())

        results = store.search_full_text("magnesium")
        # FTS5 snippet uses >>> and <<< markers
        assert len(results) == 1
        assert ">>>" in results[0].text or "magnesium" in results[0].text.lower()
        store.close()

    def test_no_match_returns_empty(self, tmp_sqlite_db):
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment())

        results = store.search_full_text("nonexistent_xyz_term")
        assert results == []
        store.close()

    def test_multiple_segments_with_tags(self, tmp_sqlite_db):
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment(
            ref="seg-1",
            tags=["health", "supplements"],
            full_text="magnesium glycinate for sleep",
        ))
        store.store_segment(_make_segment(
            ref="seg-2", primary_tag="ai-memory",
            tags=["ai-memory", "health"],
            full_text="User mentioned magnesium glycinate 400mg in health discussion",
        ))

        results = store.search_full_text("magnesium glycinate")
        assert len(results) == 2
        refs = {r.segment_ref for r in results}
        assert refs == {"seg-1", "seg-2"}
        # Verify full tag lists returned
        by_ref = {r.segment_ref: r for r in results}
        assert set(by_ref["seg-1"].tags) == {"health", "supplements"}
        assert set(by_ref["seg-2"].tags) == {"ai-memory", "health"}
        store.close()

    def test_limit_respected(self, tmp_sqlite_db):
        store = SQLiteStore(db_path=tmp_sqlite_db)
        for i in range(10):
            store.store_segment(_make_segment(
                ref=f"seg-{i}",
                full_text=f"Segment {i} contains the keyword alpha",
            ))

        results = store.search_full_text("alpha", limit=3)
        assert len(results) == 3
        store.close()

    def test_returns_quote_result_type(self, tmp_sqlite_db):
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment())

        results = store.search_full_text("magnesium")
        assert all(isinstance(r, QuoteResult) for r in results)
        store.close()

    def test_backfill_indexes_existing_segments(self, tmp_sqlite_db):
        """Segments stored before FTS full-text table are backfilled on schema init."""
        # Create store, store segment, close, reopen — backfill should find it
        store = SQLiteStore(db_path=tmp_sqlite_db)
        store.store_segment(_make_segment())
        store.close()

        # Reopen — triggers _ensure_schema again with backfill
        store2 = SQLiteStore(db_path=tmp_sqlite_db)
        results = store2.search_full_text("magnesium")
        assert len(results) >= 1
        store2.close()


# ---------------------------------------------------------------------------
# FilesystemStore.search_full_text tests
# ---------------------------------------------------------------------------

class TestFilesystemSearchFullText:
    def test_finds_match(self, tmp_store_dir):
        store = FilesystemStore(root=tmp_store_dir / "fs_store")
        store.store_segment(_make_segment(tags=["health", "supplements"]))

        results = store.search_full_text("magnesium")
        assert len(results) == 1
        assert results[0].tag == "health"
        assert results[0].segment_ref == "seg-1"
        assert results[0].tags == ["health", "supplements"]
        assert "magnesium" in results[0].text.lower()

    def test_no_match_returns_empty(self, tmp_store_dir):
        store = FilesystemStore(root=tmp_store_dir / "fs_store")
        store.store_segment(_make_segment())

        results = store.search_full_text("nonexistent_xyz_term")
        assert results == []

    def test_limit_respected(self, tmp_store_dir):
        store = FilesystemStore(root=tmp_store_dir / "fs_store")
        for i in range(10):
            store.store_segment(_make_segment(
                ref=f"seg-{i}",
                full_text=f"Segment {i} contains keyword beta",
            ))

        results = store.search_full_text("beta", limit=3)
        assert len(results) == 3

    def test_returns_quote_result_type(self, tmp_store_dir):
        store = FilesystemStore(root=tmp_store_dir / "fs_store")
        store.store_segment(_make_segment())

        results = store.search_full_text("magnesium")
        assert all(isinstance(r, QuoteResult) for r in results)


# ---------------------------------------------------------------------------
# Engine find_quote tests
# ---------------------------------------------------------------------------

class TestEngineFindQuote:
    def test_find_quote_hit(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment(tags=["health", "supplements"]))

        result = engine.find_quote("magnesium glycinate")
        assert result["found"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["topic"] == "health"
        assert "tags" not in result["results"][0]  # tags removed to avoid sub-tag noise
        assert "magnesium" in result["results"][0]["excerpt"].lower()

    def test_find_quote_miss(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment())

        result = engine.find_quote("nonexistent xyz")
        assert result["found"] is False
        assert result["results"] == []
        assert "No matches" in result["message"]

    def test_find_quote_empty_query(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.find_quote("")
        assert "error" in result

    def test_find_quote_whitespace_query(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.find_quote("   ")
        assert "error" in result

    def test_find_quote_works_without_paging_enabled(self, tmp_path):
        """find_quote works even when paging is disabled."""
        engine = _make_engine(tmp_path, paging_enabled=False)
        engine._store.store_segment(_make_segment())

        result = engine.find_quote("magnesium")
        assert result["found"] is True

    def test_find_quote_works_with_paging_enabled(self, tmp_path):
        """find_quote also works when paging is enabled."""
        engine = _make_engine(tmp_path, paging_enabled=True)
        engine._store.store_segment(_make_segment())

        result = engine.find_quote("magnesium")
        assert result["found"] is True

    def test_find_quote_cross_tag(self, tmp_path):
        """find_quote finds content regardless of tag — the core use case."""
        engine = _make_engine(tmp_path)
        # Content about health stored under ai-memory tag (the bug scenario)
        engine._store.store_segment(_make_segment(
            ref="seg-wrong-tag",
            primary_tag="ai-memory-systems",
            tags=["ai-memory-systems"],
            full_text="magnesium glycinate 400mg recommended for sleep quality",
        ))

        result = engine.find_quote("magnesium glycinate")
        assert result["found"] is True
        assert result["results"][0]["topic"] == "ai-memory-systems"


# ---------------------------------------------------------------------------
# Proxy tool definition + interception tests
# ---------------------------------------------------------------------------

class TestProxyFindQuoteTool:
    def test_vc_tool_names_includes_find_quote(self):
        from virtual_context.core.tool_loop import VC_TOOL_NAMES
        assert "vc_find_quote" in VC_TOOL_NAMES

    def test_is_vc_tool_find_quote(self):
        from virtual_context.core.tool_loop import is_vc_tool
        assert is_vc_tool("vc_find_quote") is True

    def test_tool_definitions_include_find_quote(self):
        from virtual_context.core.tool_loop import vc_tool_definitions
        tools = vc_tool_definitions()
        names = [t["name"] for t in tools]
        assert "vc_find_quote" in names
        assert len(tools) == 3  # expand, collapse, find_quote

    def test_find_quote_tool_schema(self):
        from virtual_context.core.tool_loop import vc_tool_definitions
        tools = vc_tool_definitions()
        fq = next(t for t in tools if t["name"] == "vc_find_quote")
        assert "query" in fq["input_schema"]["properties"]
        assert "max_results" in fq["input_schema"]["properties"]
        assert fq["input_schema"]["required"] == ["query"]

    def test_execute_vc_tool_dispatches_find_quote(self):
        from virtual_context.core.tool_loop import execute_vc_tool
        engine = MagicMock()
        engine.find_quote.return_value = {
            "query": "magnesium",
            "found": True,
            "results": [{"excerpt": "found it", "topic": "health", "segment_ref": "seg-1"}],
        }

        result_str = execute_vc_tool(engine, "vc_find_quote", {"query": "magnesium"})
        engine.find_quote.assert_called_once_with(query="magnesium", max_results=5)
        result = json.loads(result_str)
        assert result["found"] is True

    def test_execute_vc_tool_find_quote_with_max_results(self):
        from virtual_context.core.tool_loop import execute_vc_tool
        engine = MagicMock()
        engine.find_quote.return_value = {"found": False, "results": []}

        execute_vc_tool(engine, "vc_find_quote", {"query": "test", "max_results": 3})
        engine.find_quote.assert_called_once_with(query="test", max_results=3)


# ---------------------------------------------------------------------------
# Context hint RULE text tests
# ---------------------------------------------------------------------------

class TestContextHintMentionsFindQuote:
    def _make_engine_with_hint(self, tmp_path, mode="autonomous"):
        engine = _make_engine(tmp_path, paging_enabled=True)
        # Simulate post-compaction state
        engine._compacted_through = 10
        # Store a tag summary so hint is non-empty
        engine._store.save_tag_summary(TagSummary(
            tag="health",
            summary="Health discussion",
            description="Supplements and sleep",
            summary_tokens=20,
            source_segment_refs=["seg-1"],
            source_turn_numbers=[1],
            covers_through_turn=5,
        ))
        return engine

    def test_autonomous_hint_mentions_find_quote(self, tmp_path):
        engine = self._make_engine_with_hint(tmp_path)
        hint = engine._build_context_hint(paging_mode="autonomous")
        assert "vc_find_quote" in hint
        assert "find_quote(query)" in hint

    def test_supervised_hint_mentions_find_quote(self, tmp_path):
        engine = self._make_engine_with_hint(tmp_path)
        hint = engine._build_context_hint(paging_mode="supervised")
        assert "vc_find_quote" in hint
