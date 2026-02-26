"""Tests for vc_find_quote: full-text search across stored segments."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from virtual_context.types import (
    PagingConfig,
    QuoteResult,
    SegmentMetadata,
    StorageConfig,
    StoredSegment,
    TagSummary,
    VirtualContextConfig,
)
from virtual_context.core.quote_search import (
    _detect_query_intent,
    _locate_excerpt,
    _merge_segment_excerpts,
    _normalize_session_date,
    _parse_session_date,
    _union_spans,
    find_quote as core_find_quote,
    supplement_from_descriptions,
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
        assert "vc_remember_when" in names
        assert len(tools) == 6  # expand, collapse, find_quote, query_facts, recall_all, remember_when

    def test_find_quote_tool_schema(self):
        from virtual_context.core.tool_loop import vc_tool_definitions
        tools = vc_tool_definitions()
        fq = next(t for t in tools if t["name"] == "vc_find_quote")
        assert "query" in fq["input_schema"]["properties"]
        assert "max_results" not in fq["input_schema"]["properties"]
        assert fq["input_schema"]["required"] == ["query"]

    def test_execute_vc_tool_dispatches_find_quote(self):
        from virtual_context.core.tool_loop import execute_vc_tool
        engine = MagicMock()
        engine.find_quote.return_value = {
            "query": "magnesium",
            "query_intent": "default",
            "found": True,
            "results": [{
                "excerpt": "found it",
                "topic": "health",
                "segment_ref": "seg-1",
                "segment_refs": ["seg-1", "seg-2"],
            }],
        }

        result_str = execute_vc_tool(engine, "vc_find_quote", {"query": "magnesium"})
        engine.find_quote.assert_called_once_with(
            query="magnesium",
            max_results=20,
            intent_context="",
        )
        result = json.loads(result_str)
        assert result["found"] is True
        assert "query" not in result
        assert "query_intent" not in result
        assert "segment_ref" not in result["results"][0]
        assert "segment_refs" not in result["results"][0]

    def test_execute_vc_tool_find_quote_ignores_input_max_results(self):
        from virtual_context.core.tool_loop import execute_vc_tool
        engine = MagicMock()
        engine.find_quote.return_value = {"found": False, "results": []}

        execute_vc_tool(engine, "vc_find_quote", {"query": "test", "max_results": 3})
        engine.find_quote.assert_called_once_with(
            query="test",
            max_results=20,
            intent_context="",
        )


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


class TestRememberWhenTool:
    def test_execute_vc_tool_dispatches_remember_when(self):
        from virtual_context.core.tool_loop import execute_vc_tool

        engine = MagicMock()
        engine.remember_when.return_value = {
            "query": "auth",
            "found": True,
            "results": [],
        }
        result_str = execute_vc_tool(
            engine,
            "vc_remember_when",
            {
                "query": "auth",
                "time_range": {"kind": "relative", "preset": "last_7_days"},
            },
        )
        engine.remember_when.assert_called_once_with(
            query="auth",
            time_range={"kind": "relative", "preset": "last_7_days"},
            max_results=5,
        )
        result = json.loads(result_str)
        assert result["found"] is True


# ---------------------------------------------------------------------------
# Span-union merge tests
# ---------------------------------------------------------------------------

LONG_TEXT = (
    "Alice started the conversation about travel planning. "
    "She mentioned wanting to visit Tokyo in March for the cherry blossoms. "
    "Bob suggested they also consider Kyoto for temples and traditional culture. "
    "They discussed budget options including hostels and rail passes. "
    "Alice noted that a JR Pass would save money on bullet trains between cities. "
    "Bob recommended visiting Fushimi Inari shrine early in the morning to avoid crowds. "
    "They agreed to book flights by the end of the week."
)


class TestUnionSpans:
    def test_empty(self):
        assert _union_spans([]) == []

    def test_single(self):
        assert _union_spans([(10, 20)]) == [(10, 20)]

    def test_disjoint(self):
        assert _union_spans([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]

    def test_overlapping(self):
        assert _union_spans([(0, 20), (10, 30)]) == [(0, 30)]

    def test_adjacent(self):
        assert _union_spans([(0, 10), (10, 20)]) == [(0, 20)]

    def test_nested(self):
        assert _union_spans([(0, 50), (10, 30)]) == [(0, 50)]

    def test_unsorted_input(self):
        assert _union_spans([(20, 30), (0, 10), (5, 25)]) == [(0, 30)]

    def test_multiple_groups(self):
        result = _union_spans([(0, 10), (5, 15), (30, 40), (35, 50)])
        assert result == [(0, 15), (30, 50)]


class TestLocateExcerpt:
    def test_plain_text(self):
        full = "Hello world, this is a test."
        assert _locate_excerpt(full, "world, this") == (6, 17)

    def test_with_leading_ellipsis(self):
        full = "prefix some text suffix"
        assert _locate_excerpt(full, "...some text") == (7, 16)

    def test_with_trailing_ellipsis(self):
        full = "prefix some text suffix"
        assert _locate_excerpt(full, "some text...") == (7, 16)

    def test_with_both_ellipsis(self):
        full = "prefix some text suffix"
        assert _locate_excerpt(full, "...some text...") == (7, 16)

    def test_with_fts_markers(self):
        full = "User asked about magnesium glycinate for sleep."
        excerpt = "User asked about >>>magnesium glycinate<<< for sleep."
        span = _locate_excerpt(full, excerpt)
        assert span is not None
        assert full[span[0]:span[1]] == "User asked about magnesium glycinate for sleep."

    def test_case_insensitive_fallback(self):
        full = "Tokyo cherry blossoms"
        assert _locate_excerpt(full, "TOKYO CHERRY BLOSSOMS") is not None

    def test_not_found(self):
        assert _locate_excerpt("hello world", "xyz not here") is None

    def test_empty_excerpt(self):
        assert _locate_excerpt("hello", "") is None
        assert _locate_excerpt("hello", "...") is None


class TestMergeSegmentExcerpts:
    def _make_qr(
        self,
        text: str,
        ref: str = "seg-1",
        tag: str = "travel",
        match_type: str = "fts",
        similarity: float = 0.0,
        session_date: str = "2026-01-15",
    ) -> QuoteResult:
        return QuoteResult(
            text=text,
            tag=tag,
            segment_ref=ref,
            tags=[tag],
            match_type=match_type,
            similarity=similarity,
            session_date=session_date,
        )

    def _mock_store(self, segments: dict[str, str | None] | None = None):
        """Return a mock store. segments maps ref -> full_text (or None)."""
        store = MagicMock()
        segs = segments or {}

        def get_segment(ref):
            if ref not in segs or segs[ref] is None:
                return None
            mock_seg = MagicMock()
            mock_seg.full_text = segs[ref]
            return mock_seg

        store.get_segment.side_effect = get_segment
        return store

    def test_single_result_unchanged(self):
        qr = self._make_qr("some text")
        store = self._mock_store()
        result = _merge_segment_excerpts(store, [qr])
        assert len(result) == 1
        assert result[0] is qr
        store.get_segment.assert_not_called()

    def test_different_segments_unchanged(self):
        qr1 = self._make_qr("text one", ref="seg-1")
        qr2 = self._make_qr("text two", ref="seg-2")
        store = self._mock_store()
        result = _merge_segment_excerpts(store, [qr1, qr2])
        assert len(result) == 2
        assert result[0] is qr1
        assert result[1] is qr2

    def test_overlapping_same_segment_merged(self):
        # Two excerpts from the same segment that overlap in LONG_TEXT
        excerpt1 = "..." + LONG_TEXT[50:250] + "..."
        excerpt2 = "..." + LONG_TEXT[200:400] + "..."
        qr1 = self._make_qr(excerpt1, ref="seg-1", match_type="fts")
        qr2 = self._make_qr(excerpt2, ref="seg-1", match_type="semantic", similarity=0.85)
        store = self._mock_store({"seg-1": LONG_TEXT})

        result = _merge_segment_excerpts(store, [qr1, qr2])
        assert len(result) == 1
        merged = result[0]
        # The merged text should cover 50..400 (union of overlapping spans)
        assert LONG_TEXT[50:250] in merged.text
        assert LONG_TEXT[200:400] in merged.text
        # No duplication: the merged text should be shorter than both excerpts combined
        assert len(merged.text) < len(excerpt1) + len(excerpt2)
        assert merged.segment_ref == "seg-1"
        assert merged.similarity == 0.85  # best similarity kept

    def test_disjoint_same_segment_both_kept(self):
        # Two non-overlapping excerpts from the same segment
        excerpt1 = LONG_TEXT[0:50]
        excerpt2 = "..." + LONG_TEXT[350:] + "..."
        qr1 = self._make_qr(excerpt1, ref="seg-1")
        qr2 = self._make_qr(excerpt2, ref="seg-1")
        store = self._mock_store({"seg-1": LONG_TEXT})

        result = _merge_segment_excerpts(store, [qr1, qr2])
        assert len(result) == 1
        # Both spans present, separated by ---
        assert "---" in result[0].text

    def test_unfindable_excerpt_kept(self):
        qr1 = self._make_qr("findable text", ref="seg-1")
        qr2 = self._make_qr("totally different not in segment", ref="seg-1")
        store = self._mock_store({"seg-1": "prefix findable text suffix"})

        result = _merge_segment_excerpts(store, [qr1, qr2])
        # One merged from locatable spans + one unlocatable passed through
        assert len(result) == 2
        texts = [r.text for r in result]
        assert any("findable text" in t for t in texts)
        assert any("totally different" in t for t in texts)

    def test_segment_not_in_store(self):
        qr1 = self._make_qr("text A", ref="seg-missing")
        qr2 = self._make_qr("text B", ref="seg-missing")
        store = self._mock_store({})

        result = _merge_segment_excerpts(store, [qr1, qr2])
        # Falls back to keeping first result
        assert len(result) == 1
        assert result[0].text == "text A"

    def test_tags_combined(self):
        qr1 = self._make_qr("...Tokyo...", ref="seg-1", tag="travel")
        qr1.tags = ["travel", "japan"]
        qr2 = self._make_qr("...Tokyo...", ref="seg-1", tag="planning")
        qr2.tags = ["planning", "japan"]
        store = self._mock_store({"seg-1": "Visit Tokyo for cherry blossoms"})

        result = _merge_segment_excerpts(store, [qr1, qr2])
        assert len(result) == 1
        assert set(result[0].tags) == {"travel", "japan", "planning"}

    def test_preserves_order_across_segments(self):
        qr1 = self._make_qr("AAA", ref="seg-1")
        qr2 = self._make_qr("BBB", ref="seg-2")
        qr3 = self._make_qr("CCC", ref="seg-1")
        store = self._mock_store({"seg-1": "prefix AAA middle CCC suffix"})

        result = _merge_segment_excerpts(store, [qr1, qr2, qr3])
        # seg-1 results merged first (in insertion order), then seg-2
        refs = [r.segment_ref for r in result]
        assert refs[0] == "seg-1"
        assert refs[1] == "seg-2"


# ---------------------------------------------------------------------------
# Query intent + session-date recency ordering tests
# ---------------------------------------------------------------------------


class TestFindQuoteIntentAndRecency:
    @pytest.mark.parametrize(
        "query",
        [
            "what are we doing currently",
            "what is the status now",
            "latest architecture decision",
            "what's the plan at the moment",
            "how is this working these days",
        ],
    )
    def test_detect_query_intent_current_state(self, query):
        assert _detect_query_intent(query) == "current_state"

    def test_detect_query_intent_default(self):
        assert _detect_query_intent("find quote about magnesium") == "default"

    @pytest.mark.parametrize(
        ("raw", "normalized"),
        [
            ("2026/02/20 (Thu) 10:04", "2026-02-20"),
            ("2026-02-19", "2026-02-19"),
            ("2026/02", "2026-02-01"),
            ("no-date", ""),
        ],
    )
    def test_session_date_normalization(self, raw, normalized):
        parsed = _parse_session_date(raw)
        if normalized:
            assert parsed is not None
        assert _normalize_session_date(raw) == normalized

    def test_current_state_queries_rank_newest_session_first(self):
        store = MagicMock()
        store.search_full_text.return_value = [
            QuoteResult(
                text="older result",
                tag="status",
                segment_ref="seg-old",
                tags=["status"],
                match_type="fts",
                session_date="2025/12/01 (Mon) 09:00",
            ),
            QuoteResult(
                text="newest result",
                tag="status",
                segment_ref="seg-new",
                tags=["status"],
                match_type="fts",
                session_date="2026/02/20 (Thu) 10:04",
            ),
            QuoteResult(
                text="middle result",
                tag="status",
                segment_ref="seg-mid",
                tags=["status"],
                match_type="fts",
                session_date="2026-01-15",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="what is the latest status now",
            max_results=5,
        )

        assert out["query_intent"] == "current_state"
        sessions = [row["session"] for row in out["results"]]
        assert sessions == [
            "2026/02/20 (Thu) 10:04",
            "2026-01-15",
            "2025/12/01 (Mon) 09:00",
        ]
        normalized = [row["session_date_normalized"] for row in out["results"]]
        assert normalized == ["2026-02-20", "2026-01-15", "2025-12-01"]

    def test_default_queries_keep_original_session_order(self):
        store = MagicMock()
        store.search_full_text.return_value = [
            QuoteResult(
                text="older result",
                tag="status",
                segment_ref="seg-old",
                tags=["status"],
                match_type="fts",
                session_date="2025/12/01",
            ),
            QuoteResult(
                text="newer result",
                tag="status",
                segment_ref="seg-new",
                tags=["status"],
                match_type="fts",
                session_date="2026/02/20",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="find quote about status",
            max_results=5,
        )

        assert out["query_intent"] == "default"
        sessions = [row["session"] for row in out["results"]]
        assert sessions == ["2025/12/01", "2026/02/20"]

    def test_intent_context_can_trigger_current_state_ordering(self):
        store = MagicMock()
        store.search_full_text.return_value = [
            QuoteResult(
                text="older result",
                tag="status",
                segment_ref="seg-old",
                tags=["status"],
                match_type="fts",
                session_date="2025/12/01",
            ),
            QuoteResult(
                text="newer result",
                tag="status",
                segment_ref="seg-new",
                tags=["status"],
                match_type="fts",
                session_date="2026/02/20",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="shoe rack",
            max_results=5,
            intent_context="Where do I currently keep my old sneakers?",
        )

        assert out["query_intent"] == "current_state"
        sessions = [row["session"] for row in out["results"]]
        assert sessions == ["2026/02/20", "2025/12/01"]

    @pytest.mark.regression("BUG-031")
    def test_weak_semantic_newest_session_does_not_suppress(self):
        """An unrelated newest session matched via weak semantic similarity
        must NOT trigger current-state suppression of topically relevant
        older sessions.  Regression: 07741c45 sneakers question — gaming
        keyboard session (sim=0.26) suppressed shoe-storage sessions."""
        store = MagicMock()
        # FTS returns the topically relevant older sessions
        store.search_full_text.return_value = [
            QuoteResult(
                text="I keep my old sneakers under my bed",
                tag="sneaker-care",
                segment_ref="seg-may25",
                tags=["sneaker-care"],
                match_type="fts",
                session_date="2023/05/25 (Thu) 10:04",
            ),
            QuoteResult(
                text="storing my old sneakers in a shoe rack",
                tag="closet-organization",
                segment_ref="seg-may29",
                tags=["closet-organization"],
                match_type="fts",
                session_date="2023/05/29 (Mon) 15:01",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        # Semantic search returns a weak, unrelated match from a newer session
        semantic = MagicMock()
        semantic.semantic_search.return_value = [
            QuoteResult(
                text="I got a new cherry-mx-brown keyboard",
                tag="cherry-mx-brown",
                segment_ref="seg-jun9",
                tags=["cherry-mx-brown"],
                match_type="semantic",
                similarity=0.26,
                session_date="2023/06/09 (Fri) 12:00",
            ),
        ]

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="Where do I currently keep my old sneakers",
            max_results=5,
        )

        assert out["query_intent"] == "current_state"
        # The newest session (Jun 9) has only weak semantic match —
        # suppression must NOT activate.
        assert out.get("current_state_multi_session") is not True
        # All excerpts must be fully visible, not replaced with suppression text
        for row in out["results"]:
            assert "[Older session" not in row["excerpt"]


@pytest.mark.regression("BUG-029")
class TestSupplementFromDescriptionsWordBoundary:
    """Regression: 'old' must NOT match 'gold' as a substring."""

    def test_old_does_not_match_gold(self):
        """Query word 'old' should not match 'gold' in tag description."""
        store = MagicMock()
        store.get_all_tag_summaries.return_value = [
            TagSummary(
                tag="asian-games",
                description="China won 150 gold medals at the 2002 Asian Games.",
            ),
        ]
        store.get_segments_by_tags.return_value = []

        results = supplement_from_descriptions(
            store=store,
            query="storing old sneakers",
            results=[],
            max_results=5,
        )
        tags = [r.tag for r in results]
        assert "asian-games" not in tags

    def test_old_matches_whole_word_old(self):
        """Query word 'old' should match 'old' as a whole word."""
        store = MagicMock()
        store.get_all_tag_summaries.return_value = [
            TagSummary(
                tag="messenger-bag",
                description="User wants to replace their old messenger bag.",
            ),
        ]
        seg = StoredSegment(
            ref="seg1",
            session_id="s1",
            primary_tag="messenger-bag",
            tags=["messenger-bag"],
            summary="",
            full_text="My old bag is falling apart",
            messages=[],
            metadata=SegmentMetadata(
                turn_count=1, session_date="2023/05/27"
            ),
        )
        store.get_segments_by_tags.return_value = [seg]

        results = supplement_from_descriptions(
            store=store,
            query="storing old sneakers",
            results=[],
            max_results=5,
        )
        tags = [r.tag for r in results]
        assert "messenger-bag" in tags
