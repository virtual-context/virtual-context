"""Tests for conversation-scoped retrieval across all store methods."""

import inspect

import pytest

from virtual_context.core.store import ContextStore


class TestABCSignatures:
    """Every retrieval method on ContextStore must accept conversation_id."""

    SCOPED_METHODS = [
        "get_summaries_by_tags",
        "get_segments_by_tags",
        "search",
        "search_full_text",
        "get_segment",
        "get_summary",
        "query_facts",
        "search_facts",
        "search_tool_outputs",
        "get_fact_count_by_tags",
        "get_unique_fact_verbs",
        "get_all_tags",
        "get_all_tag_summaries",
    ]

    @pytest.mark.parametrize("method_name", SCOPED_METHODS)
    def test_method_accepts_conversation_id(self, method_name):
        method = getattr(ContextStore, method_name)
        sig = inspect.signature(method)
        assert "conversation_id" in sig.parameters, (
            f"{method_name} missing conversation_id parameter"
        )
        param = sig.parameters["conversation_id"]
        assert param.default is None, (
            f"{method_name} conversation_id should default to None"
        )


from datetime import datetime, timezone

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import StoredSegment, SegmentMetadata


def _make_two_conversation_store(tmp_path):
    """Create a SQLite store with segments in two different conversations."""
    store = SQLiteStore(db_path=str(tmp_path / "test.db"))
    now = datetime.now(timezone.utc)
    base = dict(
        summary_tokens=10, full_tokens=50, full_text="full text here",
        metadata=SegmentMetadata(), created_at=now,
        start_timestamp=now, end_timestamp=now,
    )
    store.store_segment(StoredSegment(
        ref="conv-a-1", conversation_id="conv-a",
        primary_tag="cooking", tags=["cooking", "recipes"],
        summary="How to make pasta.", **base,
    ))
    store.store_segment(StoredSegment(
        ref="conv-a-2", conversation_id="conv-a",
        primary_tag="travel", tags=["travel", "europe"],
        summary="Trip to Italy planned.", **base,
    ))
    store.store_segment(StoredSegment(
        ref="conv-b-1", conversation_id="conv-b",
        primary_tag="cooking", tags=["cooking", "baking"],
        summary="Sourdough bread recipe.", **base,
    ))
    store.store_segment(StoredSegment(
        ref="conv-b-2", conversation_id="conv-b",
        primary_tag="fitness", tags=["fitness", "running"],
        summary="Marathon training plan.", **base,
    ))
    return store


class TestSQLiteSegmentScoping:
    def test_get_summaries_by_tags_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.get_summaries_by_tags(
            tags=["cooking"], conversation_id="conv-a",
        )
        refs = [r.ref for r in results]
        assert "conv-a-1" in refs
        assert "conv-b-1" not in refs

    def test_get_summaries_by_tags_unscoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.get_summaries_by_tags(tags=["cooking"])
        refs = [r.ref for r in results]
        assert "conv-a-1" in refs
        assert "conv-b-1" in refs

    def test_get_segments_by_tags_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.get_segments_by_tags(
            tags=["cooking"], conversation_id="conv-a",
        )
        refs = [r.ref for r in results]
        assert "conv-a-1" in refs
        assert "conv-b-1" not in refs

    def test_search_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.search(query="pasta sourdough", conversation_id="conv-a")
        refs = [r.ref for r in results]
        for ref in refs:
            assert ref.startswith("conv-a")

    def test_search_full_text_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.search_full_text(query="full text", conversation_id="conv-a")
        refs = [r.segment_ref for r in results]
        for ref in refs:
            assert ref.startswith("conv-a")

    def test_get_segment_scoped_returns_own(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        seg = store.get_segment("conv-a-1", conversation_id="conv-a")
        assert seg is not None
        assert seg.ref == "conv-a-1"

    def test_get_segment_scoped_rejects_other(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        seg = store.get_segment("conv-b-1", conversation_id="conv-a")
        assert seg is None

    def test_get_summary_scoped_rejects_other(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        summary = store.get_summary("conv-b-1", conversation_id="conv-a")
        assert summary is None


from virtual_context.types import Fact


def _make_store_with_facts(tmp_path):
    """Create a store with segments and facts in two conversations."""
    store = _make_two_conversation_store(tmp_path)
    store.store_facts([
        Fact(
            subject="user", verb="likes", object="pasta",
            segment_ref="conv-a-1", conversation_id="conv-a",
            tags=["cooking"],
        ),
        Fact(
            subject="user", verb="likes", object="sourdough",
            segment_ref="conv-b-1", conversation_id="conv-b",
            tags=["cooking"],
        ),
    ])
    return store


class TestSQLiteFactScoping:
    def test_query_facts_scoped(self, tmp_path):
        store = _make_store_with_facts(tmp_path)
        facts = store.query_facts(verb="likes", conversation_id="conv-a")
        objects = [f.object for f in facts]
        assert "pasta" in objects
        assert "sourdough" not in objects

    def test_query_facts_unscoped(self, tmp_path):
        store = _make_store_with_facts(tmp_path)
        facts = store.query_facts(verb="likes")
        assert len(facts) == 2

    def test_search_facts_scoped(self, tmp_path):
        store = _make_store_with_facts(tmp_path)
        facts = store.search_facts("pasta sourdough", conversation_id="conv-a")
        for f in facts:
            assert f.conversation_id == "conv-a"

    def test_get_unique_fact_verbs_scoped(self, tmp_path):
        store = _make_store_with_facts(tmp_path)
        store.store_facts([
            Fact(
                subject="user", verb="trains_for", object="marathon",
                segment_ref="conv-b-2", conversation_id="conv-b",
                tags=["fitness"],
            ),
        ])
        verbs_a = store.get_unique_fact_verbs(conversation_id="conv-a")
        assert "likes" in verbs_a
        assert "trains_for" not in verbs_a

    def test_get_fact_count_by_tags_scoped(self, tmp_path):
        store = _make_store_with_facts(tmp_path)
        counts = store.get_fact_count_by_tags(conversation_id="conv-a")
        assert counts.get("cooking", 0) == 1

    def test_search_tool_outputs_scoped(self, tmp_path):
        store = _make_store_with_facts(tmp_path)
        store.store_tool_output(
            ref="conv-a-1", tool_name="web_search",
            content="Italian pasta recipes found", conversation_id="conv-a",
            command="search pasta", turn=1, original_bytes=100,
        )
        store.store_tool_output(
            ref="conv-b-1", tool_name="web_search",
            content="Sourdough bread recipes found", conversation_id="conv-b",
            command="search sourdough", turn=1, original_bytes=100,
        )
        results = store.search_tool_outputs("recipes", conversation_id="conv-a")
        for r in results:
            assert "conv-a" in r.segment_ref or "pasta" in str(r.text)


class TestEmptyStringSemantics:
    def test_sqlite_get_all_tags_empty_string_not_unscoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        tags = store.get_all_tags(conversation_id="")
        assert len(tags) == 0

    def test_sqlite_get_all_tags_none_is_unscoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        tags = store.get_all_tags(conversation_id=None)
        tag_names = [t.tag for t in tags]
        assert "cooking" in tag_names


from virtual_context.core.retriever import ContextRetriever
from virtual_context.types import RetrieverConfig, StrategyConfig

from conftest import MockTagGenerator


class TestEndToEndScoping:
    @pytest.mark.regression("BUG-035")
    def test_new_conversation_gets_no_context_from_other(self, tmp_path):
        """Conversation B should not get context from conversation A's segments."""
        store = SQLiteStore(db_path=str(tmp_path / "shared.db"))
        now = datetime.now(timezone.utc)
        base = dict(
            summary_tokens=10, full_tokens=50,
            full_text="Discussion about crochet patterns and yarn.",
            metadata=SegmentMetadata(), created_at=now,
            start_timestamp=now, end_timestamp=now,
        )
        store.store_segment(StoredSegment(
            ref="a-1", conversation_id="conv-a",
            primary_tag="crochet", tags=["crochet", "crafts"],
            summary="Crochet pattern discussion.", **base,
        ))
        store.store_segment(StoredSegment(
            ref="a-2", conversation_id="conv-a",
            primary_tag="tv_shows", tags=["tv_shows", "bridgerton"],
            summary="Bridgerton season 4 review.", **base,
        ))

        tag_gen = MockTagGenerator(default_tag="crochet", default_tags=["crochet"])
        config = RetrieverConfig(
            tag_context_max_tokens=30000,
            strategy_configs={
                "default": StrategyConfig(min_overlap=1, max_results=10),
            },
        )
        retriever_b = ContextRetriever(
            tag_generator=tag_gen, store=store, config=config,
            conversation_id="conv-b",
        )
        result = retriever_b.retrieve("tell me about crochet")
        assert len(result.summaries) == 0, (
            f"Conv-B leaked {len(result.summaries)} segments from conv-A: "
            f"{[s.ref for s in result.summaries]}"
        )

    def test_find_quote_scoped_no_cross_conversation_leaks(self, tmp_path):
        """find_quote must not leak across conversations via FTS."""
        store = _make_store_with_facts(tmp_path)
        # Test FTS scoping directly — search_full_text is the core path
        results = store.search_full_text("pasta sourdough", conversation_id="conv-a")
        for r in results:
            assert r.segment_ref.startswith("conv-a"), (
                f"search_full_text leaked cross-conversation ref: {r.segment_ref}"
            )

    def test_alias_ride_along_scoped(self, tmp_path):
        """Alias-expanded retrieval is conversation-scoped."""
        store = SQLiteStore(db_path=str(tmp_path / "alias.db"))
        now = datetime.now(timezone.utc)
        base = dict(
            summary_tokens=10, full_tokens=50, full_text="text",
            metadata=SegmentMetadata(), created_at=now,
            start_timestamp=now, end_timestamp=now,
        )
        store.set_tag_alias("food", "cooking")
        store.store_segment(StoredSegment(
            ref="a-food", conversation_id="conv-a",
            primary_tag="cooking", tags=["cooking"],
            summary="Conv-A cooking segment.", **base,
        ))
        store.store_segment(StoredSegment(
            ref="b-food", conversation_id="conv-b",
            primary_tag="cooking", tags=["cooking"],
            summary="Conv-B cooking segment.", **base,
        ))
        tag_gen = MockTagGenerator(default_tag="food", default_tags=["food"])
        config = RetrieverConfig(
            tag_context_max_tokens=30000,
            strategy_configs={
                "default": StrategyConfig(min_overlap=1, max_results=10),
            },
        )
        retriever = ContextRetriever(
            tag_generator=tag_gen, store=store, config=config,
            conversation_id="conv-a",
        )
        result = retriever.retrieve("tell me about food")
        refs = [s.ref for s in result.summaries]
        assert "b-food" not in refs, (
            f"Alias ride-along leaked cross-conversation ref: {refs}"
        )


from virtual_context.types import TagSummary


class TestScopingGapFixes:
    """Tests for the four conversation_id scoping gaps."""

    def test_composite_store_get_summary_forwards_conversation_id(self, tmp_path):
        """CompositeStore.get_summary() should respect conversation_id."""
        store = _make_two_conversation_store(tmp_path)
        from virtual_context.core.composite_store import CompositeStore

        comp = CompositeStore(
            segments=store, facts=store, fact_links=store,
            state=store, search=store,
        )
        # conv-b-1 exists but should be invisible when scoped to conv-a
        result = comp.get_summary("conv-b-1", conversation_id="conv-a")
        assert result is None, "get_summary should return None for wrong conversation"
        # conv-a-1 should be visible when scoped to conv-a
        result = comp.get_summary("conv-a-1", conversation_id="conv-a")
        assert result is not None

    def test_get_all_tag_summaries_scoped(self, tmp_path):
        """Tag summaries should be scoped when conversation_id is provided."""
        store = SQLiteStore(db_path=str(tmp_path / "ts.db"))
        now = datetime.now(timezone.utc)
        base = dict(
            summary_tokens=10, full_tokens=50, full_text="text",
            metadata=SegmentMetadata(), created_at=now,
            start_timestamp=now, end_timestamp=now,
        )
        store.store_segment(StoredSegment(
            ref="a-1", conversation_id="conv-a",
            primary_tag="cooking", tags=["cooking"],
            summary="Conv-A cooking.", **base,
        ))
        store.store_segment(StoredSegment(
            ref="b-1", conversation_id="conv-b",
            primary_tag="fitness", tags=["fitness"],
            summary="Conv-B fitness.", **base,
        ))
        store.save_tag_summary(TagSummary(
            tag="cooking", summary="Cooking stuff.",
            summary_tokens=5, source_segment_refs=["a-1"],
        ), conversation_id="conv-a")
        store.save_tag_summary(TagSummary(
            tag="fitness", summary="Fitness stuff.",
            summary_tokens=5, source_segment_refs=["b-1"],
        ), conversation_id="conv-b")
        # Scoped to conv-a: should only see cooking
        summaries = store.get_all_tag_summaries(conversation_id="conv-a")
        tags = [ts.tag for ts in summaries]
        assert "cooking" in tags
        assert "fitness" not in tags

    def test_get_all_tag_summaries_unscoped_returns_all(self, tmp_path):
        """Without conversation_id, all tag summaries should be returned."""
        store = SQLiteStore(db_path=str(tmp_path / "ts2.db"))
        now = datetime.now(timezone.utc)
        base = dict(
            summary_tokens=10, full_tokens=50, full_text="text",
            metadata=SegmentMetadata(), created_at=now,
            start_timestamp=now, end_timestamp=now,
        )
        store.store_segment(StoredSegment(
            ref="a-1", conversation_id="conv-a",
            primary_tag="cooking", tags=["cooking"],
            summary="Conv-A cooking.", **base,
        ))
        store.store_segment(StoredSegment(
            ref="b-1", conversation_id="conv-b",
            primary_tag="fitness", tags=["fitness"],
            summary="Conv-B fitness.", **base,
        ))
        store.save_tag_summary(TagSummary(
            tag="cooking", summary="Cooking stuff.",
            summary_tokens=5, source_segment_refs=["a-1"],
        ), conversation_id="conv-a")
        store.save_tag_summary(TagSummary(
            tag="fitness", summary="Fitness stuff.",
            summary_tokens=5, source_segment_refs=["b-1"],
        ), conversation_id="conv-b")
        summaries = store.get_all_tag_summaries()
        tags = [ts.tag for ts in summaries]
        assert "cooking" in tags
        assert "fitness" in tags

    def test_composite_store_get_all_tag_summaries_forwards(self, tmp_path):
        """CompositeStore.get_all_tag_summaries() should forward conversation_id."""
        store = SQLiteStore(db_path=str(tmp_path / "comp_ts.db"))
        now = datetime.now(timezone.utc)
        base = dict(
            summary_tokens=10, full_tokens=50, full_text="text",
            metadata=SegmentMetadata(), created_at=now,
            start_timestamp=now, end_timestamp=now,
        )
        store.store_segment(StoredSegment(
            ref="a-1", conversation_id="conv-a",
            primary_tag="cooking", tags=["cooking"],
            summary="Conv-A cooking.", **base,
        ))
        store.store_segment(StoredSegment(
            ref="b-1", conversation_id="conv-b",
            primary_tag="fitness", tags=["fitness"],
            summary="Conv-B fitness.", **base,
        ))
        store.save_tag_summary(TagSummary(
            tag="cooking", summary="Cooking stuff.",
            summary_tokens=5, source_segment_refs=["a-1"],
        ), conversation_id="conv-a")
        store.save_tag_summary(TagSummary(
            tag="fitness", summary="Fitness stuff.",
            summary_tokens=5, source_segment_refs=["b-1"],
        ), conversation_id="conv-b")
        from virtual_context.core.composite_store import CompositeStore

        comp = CompositeStore(
            segments=store, facts=store, fact_links=store,
            state=store, search=store,
        )
        summaries = comp.get_all_tag_summaries(conversation_id="conv-a")
        tags = [ts.tag for ts in summaries]
        assert "cooking" in tags
        assert "fitness" not in tags

    def test_supplement_from_descriptions_scoped(self, tmp_path):
        """supplement_from_descriptions should only scan tag summaries for target conversation."""
        store = SQLiteStore(db_path=str(tmp_path / "desc.db"))
        now = datetime.now(timezone.utc)
        base = dict(
            summary_tokens=10, full_tokens=50,
            full_text="chocolate cake recipe details here",
            metadata=SegmentMetadata(), created_at=now,
            start_timestamp=now, end_timestamp=now,
        )
        store.store_segment(StoredSegment(
            ref="a-1", conversation_id="conv-a",
            primary_tag="baking", tags=["baking"],
            summary="Conv-A baking.", **base,
        ))
        store.store_segment(StoredSegment(
            ref="b-1", conversation_id="conv-b",
            primary_tag="woodworking", tags=["woodworking"],
            summary="Conv-B woodworking.",
            full_text="building a chocolate colored desk",
            summary_tokens=10, full_tokens=50,
            metadata=SegmentMetadata(), created_at=now,
            start_timestamp=now, end_timestamp=now,
        ))
        store.save_tag_summary(TagSummary(
            tag="baking", summary="Baking.",
            description="chocolate cake baking",
            summary_tokens=5, source_segment_refs=["a-1"],
        ), conversation_id="conv-a")
        store.save_tag_summary(TagSummary(
            tag="woodworking", summary="Woodworking.",
            description="chocolate colored desk project",
            summary_tokens=5, source_segment_refs=["b-1"],
        ), conversation_id="conv-b")
        from virtual_context.core.quote_search import supplement_from_descriptions

        results = supplement_from_descriptions(
            store, "chocolate", [], max_results=5, conversation_id="conv-a",
        )
        tags = [r.tag for r in results]
        assert "woodworking" not in tags
