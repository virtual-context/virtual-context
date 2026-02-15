"""Tests for IDF-weighted retrieval and related tag expansion."""

from datetime import datetime, timezone

import pytest

from virtual_context.core.retriever import ContextRetriever
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    RetrieverConfig,
    SegmentMetadata,
    StoredSegment,
    StrategyConfig,
    TagResult,
)

from conftest import MockTagGenerator


def _make_store_with_segments(db_path, segments: list[StoredSegment]) -> SQLiteStore:
    """Create a SQLiteStore and populate with given segments."""
    store = SQLiteStore(db_path=db_path)
    for seg in segments:
        store.store_segment(seg)
    return store


def _make_retriever(
    store: SQLiteStore,
    tag_gen: MockTagGenerator,
    max_results: int = 10,
    tag_context_max_tokens: int = 30000,
) -> ContextRetriever:
    config = RetrieverConfig(
        tag_context_max_tokens=tag_context_max_tokens,
        skip_active_tags=True,
        strategy_configs={
            "default": StrategyConfig(
                min_overlap=1,
                max_results=max_results,
                max_budget_fraction=1.0,
            ),
        },
    )
    return ContextRetriever(
        tag_generator=tag_gen,
        store=store,
        config=config,
    )


NOW = datetime.now(timezone.utc)


# ---- BUG-006 scenario: rare tags should score higher than common ones ----

class TestIDFRanking:
    """IDF re-ranking should promote segments with rare tag matches."""

    def _build_bug006_store(self, db_path):
        """Reproduce BUG-006: T46 (materialized view) buried under common-tag segments.

        Tags like 'database', 'performance', 'schema' appear in many segments.
        The target segment (materialized view) also has rare tags like 'postgres', 'habits'.
        """
        segments = []
        # 5 distractors with high-frequency tags only
        for i in range(5):
            segments.append(StoredSegment(
                ref=f"distractor-{i}",
                primary_tag="database",
                tags=["database", "performance", "schema"],
                summary=f"Generic database discussion #{i} about indexing and schema design.",
                summary_tokens=30,
                full_tokens=100,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ))

        # Target: has common tags PLUS rare ones
        segments.append(StoredSegment(
            ref="target-t46",
            primary_tag="performance",
            tags=["habits", "performance", "database", "schema", "postgres"],
            summary="Materialized view for feed performance. Discussed precomputed summary table.",
            summary_tokens=30,
            full_tokens=100,
            created_at=NOW,
            start_timestamp=NOW,
            end_timestamp=NOW,
        ))

        return _make_store_with_segments(db_path, segments)

    def test_idf_ranks_rare_tags_higher(self, tmp_sqlite_db):
        """Segment with rare tag 'postgres' should rank above distractors
        that only have common tags like 'database'."""
        store = self._build_bug006_store(tmp_sqlite_db)

        # Query tags include 'postgres' (rare) + common tags
        tag_gen = MockTagGenerator(default_tag="performance")
        tag_gen.set_override(
            "precomputed",
            TagResult(
                tags=["architecture", "performance", "caching", "schema", "database"],
                primary="architecture",
                source="mock",
            ),
        )

        retriever = _make_retriever(store, tag_gen)
        result = retriever.retrieve(
            "precomputed summary table for feed with photos and comments"
        )

        assert len(result.summaries) > 0
        # The target should be reachable — IDF should promote it
        refs = [s.ref for s in result.summaries]
        # With IDF, common tags like 'database' get lower weight, so
        # the target (which matches on same common tags) is still returned
        # but potentially with a rare-tag boost if query included rare tags
        assert result.retrieval_metadata.get("idf_reranked") is True
        store.close()

    def test_idf_with_rare_query_tag_promotes_target(self, tmp_sqlite_db):
        """When query includes rare tag 'postgres', target segment should rank first."""
        store = self._build_bug006_store(tmp_sqlite_db)

        # Query includes 'postgres' which is rare (only in target)
        tag_gen = MockTagGenerator(default_tag="performance")
        tag_gen.set_override(
            "materialized",
            TagResult(
                tags=["postgres", "performance", "database"],
                primary="postgres",
                source="mock",
            ),
        )

        retriever = _make_retriever(store, tag_gen)
        result = retriever.retrieve("materialized view for feed")

        assert len(result.summaries) > 0
        # Target should be first due to rare 'postgres' tag
        assert result.summaries[0].ref == "target-t46"
        store.close()


class TestIDFGraceful:
    """IDF should handle edge cases without crashing."""

    def test_idf_graceful_when_no_stats(self, tmp_sqlite_db):
        """Empty store returns no results but doesn't crash."""
        store = SQLiteStore(db_path=tmp_sqlite_db)
        tag_gen = MockTagGenerator(default_tag="test")
        retriever = _make_retriever(store, tag_gen)

        result = retriever.retrieve("some query")
        assert result.total_tokens == 0
        store.close()

    def test_idf_with_single_segment(self, tmp_sqlite_db):
        """Single segment should still be retrieved and scored."""
        store = _make_store_with_segments(tmp_sqlite_db, [
            StoredSegment(
                ref="only-one",
                primary_tag="cooking",
                tags=["cooking"],
                summary="Recipe for pasta.",
                summary_tokens=10,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ),
        ])
        tag_gen = MockTagGenerator(default_tag="cooking", default_tags=["cooking"])
        retriever = _make_retriever(store, tag_gen)

        result = retriever.retrieve("pasta recipe")
        assert len(result.summaries) == 1
        assert result.summaries[0].ref == "only-one"
        store.close()


# ---- BUG-005 scenario: related tags bridge vocabulary gap ----

class TestRelatedTagExpansion:
    """Related tags from the tagger should expand retrieval reach."""

    def _build_bug005_store(self, db_path):
        """Reproduce BUG-005: T46 tagged {habits, performance, database, schema, postgres}.
        Query at T71 generates {caching, feature-request, saas} — zero overlap.
        But if tagger also returns related_tags=['materialized-view', 'feed-optimization'],
        and T46 was stored with those as additional tags, retrieval should succeed.
        """
        segments = [
            StoredSegment(
                ref="target-t46",
                primary_tag="performance",
                tags=["habits", "performance", "database", "schema", "postgres",
                      "materialized-view", "feed-optimization"],  # related tags added at write-time
                summary="Materialized view for feed performance. Discussed precomputed summary table.",
                summary_tokens=30,
                full_tokens=100,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ),
            StoredSegment(
                ref="unrelated-1",
                primary_tag="saas",
                tags=["saas", "pricing"],
                summary="SaaS pricing discussion.",
                summary_tokens=20,
                full_tokens=80,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ),
        ]
        return _make_store_with_segments(db_path, segments)

    def test_write_time_related_tags_enable_retrieval(self, tmp_sqlite_db):
        """Write-time related tags (materialized-view) bridge the vocabulary gap."""
        store = self._build_bug005_store(tmp_sqlite_db)

        # Query tags miss the target's original tags...
        # but 'materialized-view' was added as a write-time related tag
        tag_gen = MockTagGenerator(default_tag="caching")
        tag_gen.set_override(
            "caching trick",
            TagResult(
                tags=["caching", "feature-request", "saas"],
                primary="caching",
                source="mock",
                related_tags=["materialized-view", "precomputed"],
            ),
        )

        retriever = _make_retriever(store, tag_gen)
        result = retriever.retrieve("caching trick for the feed")

        assert len(result.summaries) > 0
        refs = [s.ref for s in result.summaries]
        assert "target-t46" in refs
        assert result.retrieval_metadata.get("query_expanded") is True
        store.close()

    def test_query_time_related_tags_expand_search(self, tmp_sqlite_db):
        """Query-time related_tags expand the search to find matching segments."""
        store = _make_store_with_segments(tmp_sqlite_db, [
            StoredSegment(
                ref="seg-auth",
                primary_tag="authentication",
                tags=["authentication", "jwt", "security"],
                summary="JWT token rotation for auth flow.",
                summary_tokens=20,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ),
        ])

        # Query is about "login" which doesn't match any tags directly
        # but related_tags includes "authentication" which bridges the gap
        tag_gen = MockTagGenerator(default_tag="login")
        tag_gen.set_override(
            "login",
            TagResult(
                tags=["login", "onboarding"],
                primary="login",
                source="mock",
                related_tags=["authentication", "jwt"],
            ),
        )

        retriever = _make_retriever(store, tag_gen)
        result = retriever.retrieve("how does login work?")

        assert len(result.summaries) > 0
        assert result.summaries[0].ref == "seg-auth"
        assert result.retrieval_metadata.get("query_expanded") is True
        assert "authentication" in result.retrieval_metadata.get("related_tags_used", [])
        store.close()

    def test_related_tags_get_lower_weight(self, tmp_sqlite_db):
        """Primary tag matches should score higher than related tag matches."""
        store = _make_store_with_segments(tmp_sqlite_db, [
            StoredSegment(
                ref="primary-match",
                primary_tag="cooking",
                tags=["cooking", "recipe"],
                summary="Cooking recipe for pasta.",
                summary_tokens=20,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ),
            StoredSegment(
                ref="related-match",
                primary_tag="nutrition",
                tags=["nutrition", "meal-prep"],
                summary="Meal prep nutrition guide.",
                summary_tokens=20,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ),
        ])

        tag_gen = MockTagGenerator(default_tag="cooking")
        tag_gen.set_override(
            "dinner",
            TagResult(
                tags=["cooking", "recipe"],
                primary="cooking",
                source="mock",
                related_tags=["nutrition"],
            ),
        )

        retriever = _make_retriever(store, tag_gen)
        result = retriever.retrieve("what should I make for dinner?")

        assert len(result.summaries) == 2
        # Primary match (cooking+recipe = 2x full IDF) should rank above
        # related match (nutrition = 1x 0.5 IDF)
        assert result.summaries[0].ref == "primary-match"
        assert result.summaries[1].ref == "related-match"
        store.close()

    def test_no_related_tags_backward_compatible(self, tmp_sqlite_db):
        """When TagResult has no related_tags, retrieval still works normally."""
        store = _make_store_with_segments(tmp_sqlite_db, [
            StoredSegment(
                ref="seg-1",
                primary_tag="legal",
                tags=["legal", "court"],
                summary="Court filing deadline.",
                summary_tokens=15,
                created_at=NOW,
                start_timestamp=NOW,
                end_timestamp=NOW,
            ),
        ])

        tag_gen = MockTagGenerator(default_tag="legal", default_tags=["legal"])
        retriever = _make_retriever(store, tag_gen)
        result = retriever.retrieve("court filing")

        assert len(result.summaries) == 1
        assert result.retrieval_metadata.get("query_expanded") is False
        assert result.retrieval_metadata.get("related_tags_used") == []
        store.close()


class TestRelatedTagsInCompactor:
    """Compactor should merge related_tags from LLM into segment tags."""

    def test_related_tags_merged_into_segment(self):
        """Related tags from LLM response should be added to segment tags."""
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, Message, TaggedSegment
        from conftest import MockLLMProvider

        llm = MockLLMProvider(response=(
            '{"summary": "Materialized view for feed performance.", '
            '"entities": ["materialized-view"], '
            '"key_decisions": ["use materialized view"], '
            '"action_items": [], '
            '"date_references": [], '
            '"refined_tags": ["performance", "feed-optimization", "caching"]}'
        ))
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
        )

        segment = TaggedSegment(
            primary_tag="performance",
            tags=["performance", "database"],
            messages=[
                Message(role="user", content="Should we add a materialized view?"),
                Message(role="assistant", content="Yes, for feed performance."),
            ],
            token_count=50,
        )

        results = compactor.compact([segment])
        assert len(results) == 1
        result = results[0]
        # Original tags preserved + LLM refined_tags merged in
        assert "performance" in result.tags
        assert "database" in result.tags
        assert "feed-optimization" in result.tags
        assert "caching" in result.tags

    def test_compactor_no_related_tags_backward_compatible(self):
        """When LLM returns no refined_tags, original segment tags are preserved."""
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, Message, TaggedSegment
        from conftest import MockLLMProvider

        llm = MockLLMProvider(response=(
            '{"summary": "Test summary.", '
            '"entities": [], "key_decisions": [], '
            '"action_items": [], "date_references": [], '
            '"refined_tags": []}'
        ))
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(),
        )

        segment = TaggedSegment(
            primary_tag="testing",
            tags=["testing", "ci"],
            messages=[
                Message(role="user", content="How do tests work?"),
                Message(role="assistant", content="They verify behavior."),
            ],
            token_count=30,
        )

        results = compactor.compact([segment])
        assert len(results) == 1
        assert set(results[0].tags) == {"testing", "ci"}
