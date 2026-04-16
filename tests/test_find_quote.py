"""Tests for vc_find_quote: full-text search across stored segments."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from virtual_context.types import SearchConfig


def _mock_engine(**overrides):
    """Create a MagicMock engine with a real SearchConfig on config.search."""
    engine = MagicMock()
    engine.config.search = SearchConfig()
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine

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
    search_summaries as core_search_summaries,
    supplement_from_descriptions,
)
from virtual_context.core.semantic_search import persist_turn_with_embeddings
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
    conversation_id: str = "session-1",
    session_date: str = "",
) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        conversation_id=conversation_id,
        primary_tag=primary_tag,
        tags=tags or [primary_tag],
        summary=summary,
        summary_tokens=20,
        full_text=full_text,
        full_tokens=100,
        messages=[{"role": "user", "content": "test"}],
        metadata=SegmentMetadata(turn_count=1, session_date=session_date),
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


def _persist_find_quote_turn(engine, *, turn_number: int = 0, user_content: str | None = None, assistant_content: str | None = None):
    persist_turn_with_embeddings(
        engine._store,
        engine._semantic,
        conversation_id=engine.config.conversation_id,
        turn_number=turn_number,
        user_content=user_content or "I take magnesium glycinate 400mg for sleep.",
        assistant_content=assistant_content or "That magnesium glycinate routine sounds sensible.",
    )


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
        _persist_find_quote_turn(engine)

        result = engine.find_quote("magnesium glycinate")
        assert result["found"] is True
        assert len(result["results"]) >= 1
        assert result["results"][0]["topic"] == "_general"
        assert result["results"][0]["source_scope"] == "turn"
        assert "tags" not in result["results"][0]  # tags removed to avoid sub-tag noise
        assert any("magnesium" in row["excerpt"].lower() for row in result["results"])

    def test_find_quote_miss(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._store.store_segment(_make_segment(conversation_id=engine.config.conversation_id))

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
        _persist_find_quote_turn(engine)

        result = engine.find_quote("magnesium")
        assert result["found"] is True

    def test_find_quote_works_with_paging_enabled(self, tmp_path):
        """find_quote also works when paging is enabled."""
        engine = _make_engine(tmp_path, paging_enabled=True)
        _persist_find_quote_turn(engine)

        result = engine.find_quote("magnesium")
        assert result["found"] is True

    def test_find_quote_cross_tag(self, tmp_path):
        """find_quote searches canonical turn text instead of relying on segment tags."""
        engine = _make_engine(tmp_path)
        _persist_find_quote_turn(
            engine,
            assistant_content="magnesium glycinate 400mg recommended for sleep quality",
        )

        result = engine.find_quote("magnesium glycinate")
        assert result["found"] is True
        assert result["results"][0]["topic"] == "_general"


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
        assert "vc_search_summaries" in names
        assert "vc_remember_when" in names
        assert len(tools) == 7  # shared VC catalogue also includes vc_restore_tool

    def test_find_quote_tool_schema(self):
        from virtual_context.core.tool_loop import vc_tool_definitions
        tools = vc_tool_definitions()
        fq = next(t for t in tools if t["name"] == "vc_find_quote")
        assert "query" in fq["input_schema"]["properties"]
        assert "mode" in fq["input_schema"]["properties"]
        assert fq["input_schema"]["properties"]["mode"]["enum"] == ["lookup", "exact_value"]
        assert "max_results" not in fq["input_schema"]["properties"]
        assert fq["input_schema"]["required"] == ["query", "mode"]

    def test_search_summaries_tool_schema(self):
        from virtual_context.core.tool_loop import vc_tool_definitions
        tools = vc_tool_definitions()
        tool = next(t for t in tools if t["name"] == "vc_search_summaries")
        assert tool["input_schema"]["properties"]["mode"]["enum"] == [
            "lookup",
            "aggregate_total",
            "coverage",
        ]

    def test_execute_vc_tool_dispatches_find_quote(self):
        from virtual_context.core.tool_loop import execute_vc_tool
        engine = _mock_engine()
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
            mode="lookup",
        )
        result = json.loads(result_str)
        assert result["found"] is True
        assert "query" not in result
        assert "query_intent" not in result
        assert "segment_ref" not in result["results"][0]
        assert "segment_refs" not in result["results"][0]

    def test_execute_vc_tool_find_quote_ignores_input_max_results(self):
        from virtual_context.core.tool_loop import execute_vc_tool
        engine = _mock_engine()
        engine.find_quote.return_value = {"found": False, "results": []}

        execute_vc_tool(engine, "vc_find_quote", {"query": "test", "max_results": 3})
        engine.find_quote.assert_called_once_with(
            query="test",
            max_results=20,
            intent_context="",
            mode="lookup",
        )

    def test_execute_vc_tool_dispatches_find_quote_with_explicit_mode(self):
        from virtual_context.core.tool_loop import execute_vc_tool
        engine = _mock_engine()
        engine.find_quote.return_value = {"found": False, "results": [], "mode": "exact_value"}

        execute_vc_tool(engine, "vc_find_quote", {"query": "test", "mode": "exact_value"})
        engine.find_quote.assert_called_once_with(
            query="test",
            max_results=20,
            intent_context="",
            mode="exact_value",
        )


# ---------------------------------------------------------------------------
# Context hint RULE text tests
# ---------------------------------------------------------------------------

class TestContextHintMentionsFindQuote:
    def _make_engine_with_hint(self, tmp_path, mode="autonomous"):
        engine = _make_engine(tmp_path, paging_enabled=True)
        # Simulate post-compaction state
        engine._engine_state.compacted_prefix_messages = 10
        engine._engine_state.flushed_prefix_messages = 10
        # Store a tag summary so hint is non-empty
        engine._store.save_tag_summary(TagSummary(
            tag="health",
            summary="Health discussion",
            description="Supplements and sleep",
            summary_tokens=20,
            source_segment_refs=["seg-1"],
            source_turn_numbers=[1],
            covers_through_turn=5,
        ), conversation_id=engine.config.conversation_id)
        return engine

    def test_autonomous_hint_mentions_find_quote(self, tmp_path):
        engine = self._make_engine_with_hint(tmp_path)
        hint = engine._retrieval._build_context_hint(paging_mode="autonomous")
        assert "vc_find_quote" in hint
        assert "find_quote(query)" in hint

    def test_supervised_hint_mentions_find_quote(self, tmp_path):
        engine = self._make_engine_with_hint(tmp_path)
        hint = engine._retrieval._build_context_hint(paging_mode="supervised")
        assert "vc_find_quote" in hint


class TestCoverageMode:
    def test_find_quote_coverage_extracts_named_components(self, tmp_path):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-sharding",
            primary_tag="sharding",
            tags=["sharding"],
            full_text=(
                "User is scaling sharding to support 5000 queries per second "
                "with balanced routing."
            ),
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-load-balancing",
            primary_tag="load-balancing",
            tags=["load-balancing"],
            full_text=(
                "Load balancing configuration is being updated for 5000 queries "
                "per second with health checks."
            ),
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-partitioning",
            primary_tag="partitioning",
            tags=["partitioning"],
            full_text=(
                "Partitioning work now targets 5000 queries per second with "
                "consistent shard distribution."
            ),
            conversation_id=conv_id,
        ))

        result = engine.search_summaries(
            "queries per second sharding load balancing partitioning",
            max_results=5,
            intent_context=(
                "How many queries per second am I aiming to support across "
                "sharding, load balancing, and partitioning efforts combined?"
            ),
            mode="coverage",
        )

        assert result["found"] is True
        summary = result["coverage_summary"]
        assert summary["requested_components"] == [
            "sharding",
            "load balancing",
            "partitioning",
        ]
        assert set(summary["covered_components"]) == {
            "sharding",
            "load balancing",
            "partitioning",
        }
        assert summary["missing_components"] == []
        assert result["coverage_value_candidates"][0]["value"] == "5000 queries/second"
        matched = {
            component
            for row in result["results"]
            for component in row.get("matched_components", [])
        }
        assert {"sharding", "load balancing", "partitioning"} <= matched

    def test_find_quote_coverage_extracts_combining_projects(self, tmp_path):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch",
            primary_tag="elasticsearch",
            tags=["elasticsearch"],
            full_text=(
                "Elasticsearch project planning currently targets 1 million "
                "documents with 98% uptime."
            ),
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr",
            primary_tag="solr",
            tags=["solr"],
            full_text=(
                "Solr optimization work focuses on 800K documents with index "
                "size tuning."
            ),
            conversation_id=conv_id,
        ))

        result = engine.search_summaries(
            "documents planning Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="coverage",
        )

        assert result["found"] is True
        summary = result["coverage_summary"]
        assert summary["requested_components"] == ["elasticsearch", "solr"]
        assert set(summary["covered_components"]) == {"elasticsearch", "solr"}
        assert summary["missing_components"] == []
        matched = {
            component
            for row in result["results"]
            for component in row.get("matched_components", [])
        }
        assert {"elasticsearch", "solr"} <= matched

    def test_find_quote_aggregate_total_computes_total(self, tmp_path):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-total",
            primary_tag="elasticsearch",
            tags=["elasticsearch"],
            full_text=(
                "I am evaluating Elasticsearch 8.8.0 for 1 million documents "
                "with 98% uptime."
            ),
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-total",
            primary_tag="solr",
            tags=["solr"],
            full_text=(
                "I am optimizing Solr 9.2.0 for 800K documents with lower "
                "search latency."
            ),
            conversation_id=conv_id,
        ))

        result = engine.search_summaries(
            "documents planning Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="aggregate_total",
        )

        assert result["found"] is True
        assert result["mode"] == "aggregate_total"
        assert result["coverage_summary"]["requested_components"] == [
            "elasticsearch",
            "solr",
        ]
        assert result["chosen_aggregate_total"]["value"] == "1.8 million documents"
        assert result["aggregate_total_candidates"][0]["value"] == "1.8 million documents"
        assert result["aggregate_total_candidates"][0]["covered_components"] == [
            "elasticsearch",
            "solr",
        ]
        assert "AGGREGATE-TOTAL MODE" in result["reader_hint"]
        assert "ambiguity_detected" not in result

    def test_find_quote_aggregate_total_prefers_component_specific_summary_values(self, tmp_path):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-summary",
            primary_tag="elasticsearch",
            tags=["elasticsearch"],
            summary=(
                "User evaluated Elasticsearch 8.8.0 performance targeting "
                "98% uptime on 1 million documents."
            ),
            full_text="This Elasticsearch project focuses on uptime tuning.",
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-summary",
            primary_tag="solr",
            tags=["solr"],
            summary=(
                "User reported 300ms delays on a Solr 9.2.0 deployment "
                "handling 800K documents and asked how to optimize it."
            ),
            full_text="This Solr project focuses on search-query optimization.",
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-shared-comparison",
            primary_tag="comparison",
            tags=["comparison", "elasticsearch", "solr"],
            summary=(
                "Comparison notes that Solr can handle 1M documents while "
                "Elasticsearch remains an alternative under evaluation."
            ),
            full_text=(
                "Stakeholder comparison mentions both Solr and Elasticsearch. "
                "The Solr option can handle 1M documents."
            ),
            conversation_id=conv_id,
        ))

        result = engine.search_summaries(
            "documents total combining Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="aggregate_total",
        )

        assert result["found"] is True
        assert result["chosen_aggregate_total"]["value"] == "1.8 million documents"
        assert result["aggregate_total_candidates"][0]["value"] == "1.8 million documents"
        component_values = result["aggregate_total_candidates"][0]["component_values"]
        assert [(item["component"], item["value"]) for item in component_values] == [
            ("elasticsearch", "1 million documents"),
            ("solr", "800K documents"),
        ]

    def test_find_quote_aggregate_total_scans_all_summaries_for_older_components(
        self, tmp_path, monkeypatch,
    ):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-older",
            primary_tag="elasticsearch",
            tags=["elasticsearch"],
            summary=(
                "User evaluated Elasticsearch 8.8.0 performance targeting "
                "98% uptime on 1 million documents."
            ),
            full_text="This Elasticsearch project focuses on uptime tuning.",
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-older",
            primary_tag="solr",
            tags=["solr"],
            summary=(
                "User reported 300ms delays on a Solr 9.2.0 deployment "
                "handling 800K documents and asked how to optimize it."
            ),
            full_text="This Solr project focuses on search-query optimization.",
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-filler",
            primary_tag="comparison",
            tags=["comparison"],
            summary="Comparison notes mention 1M documents but not the target projects.",
            full_text="This is only filler and should not determine the aggregate total.",
            conversation_id=conv_id,
        ))

        original_get_all_segments = engine._store.get_all_segments
        observed_limits: list[int | None] = []

        def spy_get_all_segments(*, conversation_id=None, limit=None):
            observed_limits.append(limit)
            if limit is not None:
                return [
                    segment
                    for segment in original_get_all_segments(
                        conversation_id=conversation_id, limit=None,
                    )
                    if segment.ref == "seg-filler"
                ][:limit]
            return original_get_all_segments(conversation_id=conversation_id, limit=limit)

        monkeypatch.setattr(engine._store, "get_all_segments", spy_get_all_segments)

        result = engine.search_summaries(
            "documents total combining Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="aggregate_total",
        )

        assert result["found"] is True
        assert result["chosen_aggregate_total"]["value"] == "1.8 million documents"
        assert result["aggregate_total_candidates"][0]["value"] == "1.8 million documents"
        assert observed_limits == [None]

    def test_find_quote_aggregate_total_ignores_far_single_component_quantity(self, tmp_path):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-near",
            primary_tag="elasticsearch",
            tags=["elasticsearch"],
            summary=(
                "User evaluated Elasticsearch 8.8.0 for 1 million documents "
                "with 98% uptime."
            ),
            full_text="Direct Elasticsearch capacity planning for 1 million documents.",
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-far",
            primary_tag="comparison",
            tags=["comparison", "elasticsearch"],
            summary=(
                "The team discussed Elasticsearch experience and migration tradeoffs. "
                "Additional operational notes covered batching, alerts, dashboards, "
                "runbooks, and incident reviews across multiple services before the "
                "document ingestion pipeline target reached 2 million documents."
            ),
            full_text=(
                "Elasticsearch came up early in the conversation, but the later "
                "2 million documents figure refers to a separate ingestion pipeline."
            ),
            conversation_id=conv_id,
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-near",
            primary_tag="solr",
            tags=["solr"],
            summary=(
                "User planned Solr 9.2.0 scalability work for 800K documents."
            ),
            full_text="Direct Solr capacity planning for 800K documents.",
            conversation_id=conv_id,
        ))

        result = engine.search_summaries(
            "documents total combining Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="aggregate_total",
        )

        assert result["found"] is True
        assert result["chosen_aggregate_total"]["value"] == "1.8 million documents"
        assert result["aggregate_total_candidates"][0]["value"] == "1.8 million documents"

    def test_find_quote_aggregate_total_prefers_clean_component_anchors_over_recent_smaller_pairs(
        self, tmp_path,
    ):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-intended",
            primary_tag="agile-methodologies",
            tags=["agile-methodologies", "elasticsearch"],
            summary=(
                "User evaluated Elasticsearch 8.8.0 performance targeting "
                "98% uptime on 1 million documents for a search project."
            ),
            full_text=(
                "User evaluated Elasticsearch 8.8.0 performance targeting "
                "98% uptime on 1 million documents for a search project."
            ),
            conversation_id=conv_id,
            session_date="July-16-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-intended",
            primary_tag="solr-clustering",
            tags=["solr-clustering", "solr"],
            summary=(
                "User planned Solr 9.2.0 scalability work for 800K "
                "documents with lower search latency."
            ),
            full_text=(
                "User planned Solr 9.2.0 scalability work for 800K "
                "documents with lower search latency."
            ),
            conversation_id=conv_id,
            session_date="July-18-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-recent-small",
            primary_tag="elasticsearch-integration",
            tags=["elasticsearch-integration", "elasticsearch"],
            summary=(
                "User is implementing an Apache Lucene indexing system "
                "integrated with Elasticsearch for handling 100K documents "
                "with compliance and deployment planning."
            ),
            full_text=(
                "User is implementing an Apache Lucene indexing system "
                "integrated with Elasticsearch for handling 100K documents "
                "with compliance and deployment planning."
            ),
            conversation_id=conv_id,
            session_date="August-21-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-recent-small",
            primary_tag="kafka-error-handling",
            tags=["kafka-error-handling", "solr"],
            summary=(
                "The conversation covers NiFi tuning and later Solr 9.3.0 "
                "search latency optimization for 150K documents."
            ),
            full_text=(
                "The conversation covers NiFi tuning and later Solr 9.3.0 "
                "search latency optimization for 150K documents."
            ),
            conversation_id=conv_id,
            session_date="August-21-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-recent-medium",
            primary_tag="oauth-authentication",
            tags=["oauth-authentication", "elasticsearch"],
            summary=(
                "User requested authentication logging improvements that "
                "send events to Elasticsearch while handling 200K documents."
            ),
            full_text=(
                "User requested authentication logging improvements that "
                "send events to Elasticsearch while handling 200K documents."
            ),
            conversation_id=conv_id,
            session_date="August-29-2024",
        ))

        result = engine.search_summaries(
            "documents total combining Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="aggregate_total",
        )

        assert result["found"] is True
        assert result["chosen_aggregate_total"]["value"] == "1.8 million documents"
        assert result["aggregate_total_candidates"][0]["value"] == "1.8 million documents"
        component_values = result["aggregate_total_candidates"][0]["component_values"]
        assert [(item["component"], item["value"]) for item in component_values] == [
            ("elasticsearch", "1 million documents"),
            ("solr", "800K documents"),
        ]

    def test_find_quote_aggregate_total_surfaces_ambiguity_when_pairs_tie(self, tmp_path):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-2m",
            primary_tag="elasticsearch",
            tags=["elasticsearch"],
            summary=(
                "User planned an Elasticsearch project sized for 2 million "
                "documents with operational tuning."
            ),
            full_text="Elasticsearch planning for 2 million documents.",
            conversation_id=conv_id,
            session_date="August-01-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-800k",
            primary_tag="solr",
            tags=["solr"],
            summary=(
                "User planned a Solr project sized for 800K documents with "
                "index optimization."
            ),
            full_text="Solr planning for 800K documents.",
            conversation_id=conv_id,
            session_date="July-18-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-1m",
            primary_tag="solr",
            tags=["solr"],
            summary=(
                "User planned another Solr project sized for 1,000,000 "
                "documents with clustering work."
            ),
            full_text="Solr planning for 1,000,000 documents.",
            conversation_id=conv_id,
            session_date="July-11-2024",
        ))

        result = engine.search_summaries(
            "documents total combining Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="aggregate_total",
        )

        assert result["found"] is True
        assert result["ambiguity_detected"] is True
        assert "chosen_aggregate_total" not in result
        competing_values = [
            candidate["value"]
            for candidate in result["competing_aggregate_totals"]
        ]
        assert "2.8 million documents" in competing_values
        assert "3 million documents" in competing_values
        assert "Do not choose one confidently" in result["reader_hint"]

    def test_find_quote_aggregate_total_ambiguity_lists_full_top_anchor_matrix(self, tmp_path):
        engine = _make_engine(tmp_path)
        conv_id = engine.config.conversation_id
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-18m",
            primary_tag="elasticsearch-indexing",
            tags=["elasticsearch-indexing", "elasticsearch"],
            summary=(
                "User is optimizing Elasticsearch for sparse retrieval on "
                "1.8 million documents with cluster tuning."
            ),
            full_text=(
                "User is optimizing Elasticsearch for sparse retrieval on "
                "1.8 million documents with cluster tuning."
            ),
            conversation_id=conv_id,
            session_date="August-21-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-elasticsearch-1m",
            primary_tag="agile-methodologies",
            tags=["agile-methodologies", "elasticsearch"],
            summary=(
                "User evaluated Elasticsearch 8.8.0 performance targeting "
                "98% uptime on 1 million documents."
            ),
            full_text=(
                "User evaluated Elasticsearch 8.8.0 performance targeting "
                "98% uptime on 1 million documents."
            ),
            conversation_id=conv_id,
            session_date="July-16-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-1m",
            primary_tag="latency-optimization",
            tags=["latency-optimization", "solr"],
            summary=(
                "User is designing a Solr architecture that handles 1M "
                "documents with low search latency."
            ),
            full_text=(
                "User is designing a Solr architecture that handles 1M "
                "documents with low search latency."
            ),
            conversation_id=conv_id,
            session_date="July-11-2024",
        ))
        engine._store.store_segment(_make_segment(
            ref="seg-solr-800k",
            primary_tag="solr-clustering",
            tags=["solr-clustering", "solr"],
            summary=(
                "User planned Solr 9.2.0 scalability work for 800K "
                "documents with lower search latency."
            ),
            full_text=(
                "User planned Solr 9.2.0 scalability work for 800K "
                "documents with lower search latency."
            ),
            conversation_id=conv_id,
            session_date="July-18-2024",
        ))

        result = engine.search_summaries(
            "documents total combining Elasticsearch Solr projects",
            max_results=5,
            intent_context=(
                "How many documents am I planning to handle in total when "
                "combining my Elasticsearch and Solr projects?"
            ),
            mode="aggregate_total",
        )

        assert result["found"] is True
        assert result["ambiguity_detected"] is True
        competing_values = [
            candidate["value"]
            for candidate in result["competing_aggregate_totals"]
        ]
        assert set(competing_values) == {
            "2.8 million documents",
            "2 million documents",
            "2.6 million documents",
            "1.8 million documents",
        }
        assert "supported possibility" in result["reader_hint"]


class TestRememberWhenTool:
    def test_execute_vc_tool_dispatches_remember_when(self):
        from virtual_context.core.tool_loop import execute_vc_tool

        engine = _mock_engine()
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
            intent_context="What auth issues came up recently?",
        )
        engine.remember_when.assert_called_once_with(
            query="auth",
            time_range={"kind": "relative", "preset": "last_7_days"},
            max_results=None,
            mode="auto",
            intent_context="What auth issues came up recently?",
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

        def get_segment(ref, conversation_id=None):
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

        out = core_search_summaries(
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
        store.search_canonical_turn_text.return_value = [
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
        semantic.semantic_canonical_turn_search.return_value = []

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

        out = core_search_summaries(
            store=store,
            semantic=semantic,
            query="shoe rack",
            max_results=5,
            intent_context="Where do I currently keep my old sneakers?",
        )

        assert out["query_intent"] == "current_state"
        sessions = [row["session"] for row in out["results"]]
        assert sessions == ["2026/02/20", "2025/12/01"]

    def test_search_summaries_lookup_returns_intact_segment_summaries(self):
        store = MagicMock()
        store.search_full_text.return_value = []
        store.get_all_tag_summaries.return_value = []
        store.search_tool_outputs.return_value = []
        store.get_segment.side_effect = lambda ref, conversation_id=None: {
            "seg-anchor": _make_segment(
                ref="seg-anchor",
                primary_tag="cost-analysis",
                tags=["cost-analysis"],
                summary=(
                    "User specified an initial requirement of 500 EC2 instances "
                    "at $0.11/hour and asked for a cost estimation tool across providers."
                ),
                full_text="generic chunk body",
                session_date="July-13-2024",
            ),
            "seg-support": _make_segment(
                ref="seg-support",
                primary_tag="cost-modeling",
                tags=["cost-modeling"],
                summary=(
                    "The tool should also support instance type selection, "
                    "region selection, and cost optimization."
                ),
                full_text="generic support body",
                session_date="July-13-2024",
            ),
        }.get(ref)
        semantic = MagicMock()
        semantic.semantic_search.return_value = [
            QuoteResult(
                text="2. Calculate Total Cost ... generic chunk",
                tag="cost-analysis",
                segment_ref="seg-anchor",
                tags=["cost-analysis"],
                match_type="semantic",
                similarity=0.76,
                session_date="July-13-2024",
            ),
            QuoteResult(
                text="Optimize resource utilization ... generic chunk",
                tag="cost-modeling",
                segment_ref="seg-support",
                tags=["cost-modeling"],
                match_type="semantic",
                similarity=0.66,
                session_date="July-13-2024",
            ),
        ]

        out = core_search_summaries(
            store=store,
            semantic=semantic,
            query="cost estimation cloud instances multiple providers",
            max_results=5,
            mode="lookup",
            conversation_id="beam-10M_1",
        )

        assert out["found"] is True
        assert len(out["results"]) == 2
        assert out["results"][0]["topic"] == "cost-analysis"
        assert "$0.11/hour" in out["results"][0]["excerpt"]
        assert "500 EC2 instances" in out["results"][0]["excerpt"]
        assert "merged_count" not in out["results"][0]
        assert out["results"][1]["topic"] == "cost-modeling"
        assert out["chosen_preference_anchor"]["provider"] == "AWS EC2"
        assert out["chosen_preference_anchor"]["hourly_rate"] == "$0.11/hour"
        assert out["chosen_preference_anchor"]["instance_count"] == "500"
        assert out["anchor_example_calculation"]["formula"] == "$0.11/hour * 500 instances = $55/hour"
        assert "Do not substitute alternate illustrative rates or counts" in out["reader_hint"]

    def test_exact_value_mode_prioritizes_explicit_value_hits(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "The pipeline targeted 96% detection across 3,000 query "
                    "simulations for QueryResponseError."
                ),
                tag="testing",
                segment_ref="seg-approx",
                tags=["testing"],
                match_type="semantic",
                similarity=0.81,
                session_date="2025-02-16",
            ),
            QuoteResult(
                text=(
                    "The logging setup targeted 98% detection across 10,000 "
                    "test records for IngestionParseError."
                ),
                tag="ingestion-error-handling",
                segment_ref="seg-exact",
                tags=["ingestion-error-handling"],
                match_type="fts",
                session_date="2025-02-15",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="detection rate total number test records",
            max_results=2,
            intent_context=(
                "What detection rate and total number of test records did I "
                "mention when setting up logs to catch that specific error?"
            ),
            mode="exact_value",
        )

        assert out["mode"] == "exact_value"
        assert out["results"][0]["topic"] == "ingestion-error-handling"
        assert out["value_candidates"][0]["values"][:2] == [
            "98%",
            "10,000 test records",
        ]
        assert out["chosen_exact_value_candidate"]["values"][:2] == [
            "98%",
            "10,000 test records",
        ]
        assert out["chosen_exact_value_candidate"]["topic"] == "ingestion-error-handling"
        assert "chosen_exact_value_candidate" in out["reader_hint"]
        assert len(out["results"]) == 1
        assert "10,000 test records" in out["results"][0]["excerpt"]

    def test_exact_value_mode_prioritizes_clean_version_candidates(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "Assistant: For indexing over 1 million documents, Milvus "
                    "2.2.0 is a solid option and integrates well with the "
                    "rest of the retrieval stack."
                ),
                tag="search-infra",
                segment_ref="seg-noisy",
                tags=["search-infra"],
                match_type="fts",
                session_date="2025-02-21",
            ),
            QuoteResult(
                text=(
                    "User: I'm evaluating Milvus 2.3.1 for the vector "
                    "database cluster setup and indexing over 1 million "
                    "vectors."
                ),
                tag="milvus-cluster-setup",
                segment_ref="seg-version",
                tags=["milvus-cluster-setup"],
                match_type="semantic",
                similarity=0.79,
                session_date="2025-02-20",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="Milvus version evaluating indexing 1 million documents",
            max_results=2,
            intent_context=(
                "What version of the vector database am I evaluating for "
                "indexing over 1 million documents?"
            ),
            mode="exact_value",
        )

        assert out["mode"] == "exact_value"
        assert out["chosen_exact_value_candidate"]["values"][0] == "2.3.1"
        assert out["chosen_exact_value_candidate"]["version_values"] == ["2.3.1"]
        assert out["chosen_exact_value_candidate"]["topic"] == "milvus-cluster-setup"
        assert out["chosen_exact_value_candidate"]["user_statement"] is True
        assert "exact_value_candidates" not in out
        assert len(out["results"]) == 1
        assert "2.3.1" in out["results"][0]["excerpt"]

    def test_exact_value_mode_surfaces_shared_rate_candidates_for_quote_evidence(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "User: I'm trying to implement a system that can handle "
                    "2,500 queries/sec with 99.9% uptime using parallel "
                    "processing."
                ),
                tag="terraform-iac",
                segment_ref="seg-2500",
                tags=["terraform-iac"],
                match_type="fts",
                session_date="2024-09-06",
                source_scope="turn",
                turn_number=3022,
                matched_side="both",
            ),
            QuoteResult(
                text=(
                    "User: I'm trying to implement the partitioning logic to "
                    "handle 1,500 queries/sec across 4 replicated zones."
                ),
                tag="query-parallelization",
                segment_ref="seg-1500",
                tags=["query-parallelization"],
                match_type="fts",
                session_date="2025-02-10",
                source_scope="turn",
                turn_number=6227,
                matched_side="both",
            ),
            QuoteResult(
                text=(
                    "User: I've been working on designing a distributed "
                    "system architecture to support 5,000 queries/sec with "
                    "99.9% uptime, using a combination of load balancing "
                    "and sharding to distribute the queries across nodes.\n\n"
                    "Assistant: Certainly! Designing a distributed system to "
                    "handle 5,000 queries per second with 99.9% uptime "
                    "requires careful consideration of load balancing, "
                    "sharding, and fault tolerance."
                ),
                tag="api-rate-limiting",
                segment_ref="seg-5000",
                tags=["api-rate-limiting"],
                match_type="fts",
                session_date="2025-02-25",
                source_scope="turn",
                turn_number=6718,
                matched_side="both",
            ),
            QuoteResult(
                text=(
                    "User: I need to design a distributed system "
                    "architecture to handle 3,000 queries per second with "
                    "99.9% uptime, using load balancing and sharding."
                ),
                tag="effort-estimation",
                segment_ref="seg-3000",
                tags=["effort-estimation"],
                match_type="fts",
                session_date="2025-02-10",
                source_scope="turn",
                turn_number=6271,
                matched_side="both",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="queries per second sharding load balancing partitioning",
            max_results=4,
            intent_context=(
                "How many queries per second am I aiming to support across "
                "sharding, load balancing, and partitioning efforts combined?"
            ),
            mode="exact_value",
        )

        assert out["chosen_exact_value_candidate"]["values"][:2] == [
            "99.9%",
            "5,000 queries",
        ]
        assert "exact_value_candidates" not in out
        assert out["shared_value_candidates"][0]["value"] == "5,000 queries/second"
        assert out["shared_value_candidates"][0]["occurrences"] == 2
        assert out["shared_value_candidates"][0]["matched_components"] == [
            "sharding",
            "load balancing",
            "partitioning",
        ]
        assert "shared_value_candidates" in out["reader_hint"]
        assert len(out["results"]) == 1
        assert "5,000 queries/sec" in out["results"][0]["excerpt"]

    def test_exact_value_mode_prefers_latest_user_version_self_report(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "User: I'm evaluating Milvus 2.2.0 for indexing over "
                    "1 million documents."
                ),
                tag="vector-database-evaluation",
                segment_ref="seg-older",
                tags=["vector-database-evaluation"],
                match_type="fts",
                session_date="2024-07-18",
            ),
            QuoteResult(
                text=(
                    "User: I'm evaluating Milvus 2.3.1 for the vector "
                    "database cluster setup and indexing over 1 million "
                    "vectors."
                ),
                tag="milvus-cluster-setup",
                segment_ref="seg-newer",
                tags=["milvus-cluster-setup"],
                match_type="fts",
                session_date="2024-08-17",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="vector database version indexing 1 million documents",
            max_results=2,
            intent_context=(
                "What version of the vector database am I evaluating for "
                "indexing over 1 million documents?"
            ),
            mode="exact_value",
        )

        assert out["chosen_exact_value_candidate"]["values"][0] == "2.3.1"
        assert out["chosen_exact_value_candidate"]["session_date_normalized"] == "2024-08-17"

    def test_lookup_mode_adds_reader_hint_for_conflicting_user_self_state_quotes(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "User: I'm setting up diagnostic logs to capture shard "
                    "distribution errors, targeting 98% detection across "
                    "100,000 test vectors."
                ),
                tag="sharding-logs",
                segment_ref="seg-yes",
                tags=["sharding-logs"],
                match_type="fts",
                session_date="2025-02-12",
                source_scope="turn",
                turn_number=6401,
                matched_side="user",
            ),
            QuoteResult(
                text=(
                    "User: I've never set up diagnostic logs to capture "
                    "shard distribution errors, and I'm worried this might "
                    "impact my ability to debug issues."
                ),
                tag="sharding-logs",
                segment_ref="seg-no",
                tags=["sharding-logs"],
                match_type="fts",
                session_date="2025-02-14",
                source_scope="turn",
                turn_number=6409,
                matched_side="user",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="diagnostic logs capture shard distribution errors",
            max_results=4,
            intent_context=(
                "Have I set up diagnostic logs to capture shard distribution "
                "errors in my sharding implementation?"
            ),
            mode="lookup",
        )

        assert out["mode"] == "lookup"
        assert "CONTRADICTION CHECK" in out["reader_hint"]
        assert "ask which is correct" in out["reader_hint"].lower()
        assert len(out["results"]) == 2

    def test_lookup_mode_adds_soft_compare_hint_for_non_conflicting_quotes(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "User: I'm working with Johnny on securing tuning logic "
                    "for a 25% protection boost."
                ),
                tag="secure-tuning-logic",
                segment_ref="seg-1",
                tags=["secure-tuning-logic"],
                match_type="fts",
                session_date="2024-08-10",
                source_scope="turn",
                turn_number=2101,
                matched_side="user",
            )
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="Johnny code review tuning logic qualifications expertise",
            max_results=3,
            mode="lookup",
        )

        assert out["mode"] == "lookup"
        assert "Compare the returned quotes before answering" in out["reader_hint"]
        assert "If the evidence does not conflict, answer directly" in out["reader_hint"]
        assert len(out["results"]) == 1

    def test_lookup_mode_reranks_raw_turns_by_distinct_query_phrase_coverage(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "User: I've been getting some issues with vector lookups "
                    "during dense search integration, but I've never "
                    "actually logged any errors for this, so I'm not sure "
                    "where to start debugging."
                ),
                tag="elasticsearch-indexing",
                segment_ref="turn_2552",
                tags=["elasticsearch-indexing"],
                match_type="full_text_search",
                session_date="August-21-2024",
                source_scope="turn",
                turn_number=2552,
                matched_side="user",
            ),
            QuoteResult(
                text=(
                    "User: I'm working on a project that involves "
                    "integrating dense vector search with approximate nearest "
                    "neighbors using FAISS 1.7.4. I've been experiencing "
                    "some issues with the integration."
                ),
                tag="pipeline-routing-optimization",
                segment_ref="turn_3372",
                tags=["pipeline-routing-optimization"],
                match_type="full_text_search",
                session_date="September-28-2024",
                source_scope="turn",
                turn_number=3372,
                matched_side="user",
            ),
            QuoteResult(
                text=(
                    "User: Always provide detailed error codes when I ask "
                    "about debugging strategies."
                ),
                tag="query-throughput-optimization",
                segment_ref="turn_6127",
                tags=["query-throughput-optimization"],
                match_type="full_text_search",
                session_date="February-07-2025",
                source_scope="turn",
                turn_number=6127,
                matched_side="user",
            ),
            QuoteResult(
                text=(
                    "User: Always include exact error messages when I ask "
                    "about debugging strategies."
                ),
                tag="hybrid-retrieval",
                segment_ref="turn_3329",
                tags=["hybrid-retrieval"],
                match_type="full_text_search",
                session_date="September-28-2024",
                source_scope="turn",
                turn_number=3329,
                matched_side="user",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="exact error messages debugging strategies vector lookups dense search integration",
            max_results=4,
            mode="lookup",
        )

        assert [row["turn_number"] for row in out["results"][:2]] == [2552, 3329]

    def test_exact_value_mode_reserves_semantic_budget_when_lexical_is_full(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text=(
                    "User: I'm evaluating Elasticsearch 8.9.0 for sparse "
                    "retrieval and indexing 1.8 million documents."
                ),
                tag="fastapi-integration",
                segment_ref=f"turn_{idx}",
                tags=["fastapi-integration"],
                match_type="full_text_search",
                session_date="2024-08-25",
                turn_number=idx,
                source_scope="turn",
                matched_side="user",
            )
            for idx in range(1, 21)
        ]
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = [
            QuoteResult(
                text=(
                    "User: I'm trying to optimize the performance of my "
                    "Milvus 2.3.1 vector database for dense result retrieval."
                ),
                tag="elasticsearch-8-9-0",
                segment_ref="turn_999",
                tags=["elasticsearch-8-9-0"],
                match_type="full_text_semantic",
                session_date="2024-10-02",
                turn_number=999,
                source_scope="turn",
                matched_side="user",
                similarity=0.69,
            )
        ]

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="vector database evaluation indexing 1 million documents",
            max_results=5,
            intent_context=(
                "What version of the vector database am I evaluating for "
                "indexing over 1 million documents?"
            ),
            mode="exact_value",
        )

        semantic.semantic_canonical_turn_search.assert_called_once()
        assert out["chosen_exact_value_candidate"]["values"][0] == "2.3.1"
        assert "2.3.1" in out["results"][0]["excerpt"]

    def test_find_quote_uses_canonical_turn_context_without_segment_join(self):
        store = MagicMock()
        store.search_canonical_turn_text.return_value = [
            QuoteResult(
                text="User: I'm setting up logs to catch IngestionParseError.",
                tag="ingestion-error-handling",
                segment_ref="turn_42",
                tags=["ingestion-error-handling", "logging"],
                match_type="full_text_search",
                session_date="February-17-2025",
                source_scope="turn",
                turn_number=42,
                matched_side="user",
            )
        ]
        store.get_all_segments.side_effect = AssertionError("segment join should not be used")
        store.get_all_tag_summaries.return_value = []
        semantic = MagicMock()
        semantic.semantic_canonical_turn_search.return_value = []

        out = core_find_quote(
            store=store,
            semantic=semantic,
            query="IngestionParseError logs",
            max_results=1,
            mode="lookup",
            conversation_id="beam-10M_1",
        )

        assert out["results"][0]["topic"] == "ingestion-error-handling"
        assert out["results"][0]["segment_ref"] == "turn_42"
        assert out["results"][0]["session"] == "February-17-2025"
        assert out["results"][0]["session_date_normalized"] == "2025-02-17"

    def test_coverage_mode_diversifies_across_sessions(self):
        store = MagicMock()
        store.search_full_text.return_value = [
            QuoteResult(
                text="System A handled 5,000 queries per hour with sharding.",
                tag="sharding",
                segment_ref="seg-a-1",
                tags=["sharding"],
                match_type="fts",
                session_date="2024-07-18",
            ),
            QuoteResult(
                text="Load balancing details for the same sharding project.",
                tag="sharding",
                segment_ref="seg-a-2",
                tags=["sharding"],
                match_type="fts",
                session_date="2024-07-18",
            ),
            QuoteResult(
                text="Partitioning work targeted 6,000 queries per hour.",
                tag="partitioning",
                segment_ref="seg-b-1",
                tags=["partitioning"],
                match_type="fts",
                session_date="2024-08-02",
            ),
        ]
        store.get_all_tag_summaries.return_value = []
        store.search_tool_outputs.return_value = []
        semantic = MagicMock()
        semantic.semantic_search.return_value = []

        out = core_search_summaries(
            store=store,
            semantic=semantic,
            query="queries per second sharding load balancing partitioning",
            max_results=2,
            mode="coverage",
        )

        sessions = [row["session"] for row in out["results"]]
        assert len(set(sessions)) == 2
        assert out["mode"] == "coverage"
        assert out["coverage_summary"]["distinct_sessions"] == 2
        assert "COVERAGE MODE" in out["reader_hint"]

    @pytest.mark.regression("BUG-031")
    def test_weak_semantic_newest_session_does_not_suppress(self):
        """An unrelated newest session matched via weak semantic similarity
        must NOT trigger current-state suppression of topically relevant
        older sessions.  Regression: 07741c45 sneakers question — gaming
        keyboard session (sim=0.26) suppressed shoe-storage sessions."""
        store = MagicMock()
        # FTS returns the topically relevant older sessions
        store.search_canonical_turn_text.return_value = [
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
        semantic.semantic_canonical_turn_search.return_value = [
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
            conversation_id="s1",
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
