"""Tests for ContextRetriever (tag-based)."""

from datetime import datetime, timezone

import pytest

from virtual_context.core.retriever import ContextRetriever
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    Message,
    RetrieverConfig,
    SegmentMetadata,
    StoredSegment,
    StrategyConfig,
    TagGeneratorConfig,
    TagResult,
    TagSummary,
)

from conftest import MockTagGenerator


def _make_retriever(
    db_path,
    default_tag="legal",
    skip_active=True,
    max_budget_fraction=0.25,
):
    tag_gen = MockTagGenerator(default_tag=default_tag, default_tags=[default_tag])
    tag_gen.set_override("insulin", TagResult(tags=["medical"], primary="medical", source="mock"))
    tag_gen.set_override("glucose", TagResult(tags=["medical"], primary="medical", source="mock"))
    tag_gen.set_override("court", TagResult(tags=["legal"], primary="legal", source="mock"))
    tag_gen.set_override("weather", TagResult(tags=["_general"], primary="_general", source="fallback"))

    store = SQLiteStore(db_path=db_path)

    # Pre-populate store
    now = datetime.now(timezone.utc)
    store.store_segment(StoredSegment(
        ref="legal-1",
        primary_tag="legal",
        tags=["legal", "court"],
        summary="Discussion about case 24-cv-1234 filing deadline.",
        summary_tokens=20,
        full_tokens=100,
        metadata=SegmentMetadata(entities=["Case 24-cv-1234"]),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    ))
    store.store_segment(StoredSegment(
        ref="medical-1",
        primary_tag="medical",
        tags=["medical", "health"],
        summary="Patient glucose levels elevated. Insulin adjustment discussed.",
        summary_tokens=25,
        full_tokens=120,
        metadata=SegmentMetadata(entities=["glucose", "insulin"]),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    ))

    config = RetrieverConfig(
        tag_context_max_tokens=30000,
        skip_active_tags=skip_active,
        strategy_configs={
            "default": StrategyConfig(
                min_overlap=1,
                max_results=10,
                max_budget_fraction=max_budget_fraction,
            ),
        },
    )
    return ContextRetriever(
        tag_generator=tag_gen,
        store=store,
        config=config,
    ), store


@pytest.fixture
def retriever_and_store(tmp_sqlite_db):
    retriever, store = _make_retriever(tmp_sqlite_db)
    yield retriever, store
    store.close()


def test_retrieve_legal(retriever_and_store):
    retriever, _ = retriever_and_store
    result = retriever.retrieve("What about the court filing?")
    assert "legal" in result.tags_matched
    assert len(result.summaries) > 0


def test_retrieve_medical(tmp_sqlite_db):
    retriever, store = _make_retriever(tmp_sqlite_db, default_tag="medical")
    result = retriever.retrieve("How is my insulin dosage?")
    assert "medical" in result.tags_matched
    store.close()


def test_skip_active_tags(retriever_and_store):
    retriever, _ = retriever_and_store
    result = retriever.retrieve(
        "What about the court filing?",
        current_active_tags=["legal"],
    )
    # legal tag should be skipped since it's active
    assert len(result.summaries) == 0


def test_general_returns_empty(retriever_and_store):
    retriever, _ = retriever_and_store
    result = retriever.retrieve("The weather is nice today")
    assert result.total_tokens == 0


def test_cost_report_populated(retriever_and_store):
    retriever, _ = retriever_and_store
    result = retriever.retrieve("Court filing deadline?")
    assert result.cost_report is not None
    assert result.cost_report.strategy_active == "default"
    assert len(result.cost_report.tags_queried) > 0


def test_budget_scaling_with_utilization(tmp_sqlite_db):
    retriever, store = _make_retriever(tmp_sqlite_db, max_budget_fraction=1.0)
    # High utilization should reduce effective budget
    result = retriever.retrieve(
        "Court filing?",
        current_utilization=0.9,
    )
    assert result.cost_report.budget_fraction_used >= 0
    store.close()


def test_retrieval_metadata(retriever_and_store):
    retriever, _ = retriever_and_store
    result = retriever.retrieve("Court case update")
    meta = result.retrieval_metadata
    assert "elapsed_ms" in meta
    assert "tags_from_message" in meta
    assert "tags_queried" in meta


def test_no_active_tag_skip_when_disabled(tmp_sqlite_db):
    retriever, store = _make_retriever(tmp_sqlite_db, skip_active=False)
    result = retriever.retrieve(
        "Court filing?",
        current_active_tags=["legal"],
    )
    # Should still retrieve even though legal is active
    assert len(result.summaries) > 0
    store.close()


def test_fts_fallback_on_tag_miss(tmp_sqlite_db):
    """When tags don't overlap but stored text matches, FTS fallback finds it."""
    tag_gen = MockTagGenerator(default_tag="cook-mode", default_tags=["cook-mode"])
    store = SQLiteStore(db_path=tmp_sqlite_db)

    now = datetime.now(timezone.utc)
    # Stored segment was tagged with descriptive terms, not "cook-mode"
    store.store_segment(StoredSegment(
        ref="ux-1",
        primary_tag="ux",
        tags=["ux", "frontend", "timers"],
        summary="Cook mode feature: full-screen step cards with large typography and floating timer widget.",
        summary_tokens=30,
        full_text="User asked about cook mode for the frontend showing one step at a time with large text and timers.",
        full_tokens=100,
        metadata=SegmentMetadata(),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    ))

    config = RetrieverConfig(
        tag_context_max_tokens=30000,
        strategy_configs={"default": StrategyConfig()},
    )
    retriever = ContextRetriever(
        tag_generator=tag_gen,
        store=store,
        config=config,
    )

    # "cook-mode" tag has zero overlap with ["ux", "frontend", "timers"]
    result = retriever.retrieve("How does the cook mode feature we discussed work?")
    assert len(result.summaries) > 0, "FTS fallback should find the cook mode segment"
    assert result.retrieval_metadata.get("fts_fallback") is True
    assert "cook mode" in result.summaries[0].summary.lower() or "cook mode" in result.summaries[0].ref
    store.close()


def test_fts_fallback_not_used_when_tags_match(tmp_sqlite_db):
    """FTS fallback should NOT fire when tag overlap already found results."""
    retriever, store = _make_retriever(tmp_sqlite_db)
    result = retriever.retrieve("What about the court filing?")
    assert len(result.summaries) > 0
    assert result.retrieval_metadata.get("fts_fallback") is None
    store.close()


# ---------------------------------------------------------------------------
# Inbound Matching (embedding-based vocabulary matching)
# ---------------------------------------------------------------------------

class TestInboundMatching:
    """Test embedding-based inbound matching replaces LLM tagger for inbound."""

    def _make_embed_fn(self):
        """Fake embed function: returns a fixed vector per known string."""
        # Simple deterministic embeddings for testing:
        # Represent tags/text as one-hot-ish vectors for cosine similarity control
        vectors = {
            "electronics": [1.0, 0.0, 0.0, 0.0, 0.0],
            "arduino":     [0.9, 0.1, 0.0, 0.0, 0.0],  # close to electronics
            "planes":      [0.0, 1.0, 0.0, 0.0, 0.0],
            "jiu-jitsu":   [0.0, 0.0, 1.0, 0.0, 0.0],
            "cars":        [0.0, 0.0, 0.0, 1.0, 0.0],
            "engineering": [0.3, 0.3, 0.0, 0.3, 0.1],  # cross-cutting
            "design":      [0.2, 0.3, 0.0, 0.3, 0.2],  # cross-cutting
            "craftsmanship": [0.1, 0.2, 0.2, 0.2, 0.3],  # very cross-cutting
            "_general":    [0.0, 0.0, 0.0, 0.0, 1.0],
        }
        # Text vectors — simulate what a real model would return
        # Keys must be lowercase (embed fn lowercases input)
        text_vectors = {
            "tell me about arduino circuits": [0.95, 0.05, 0.0, 0.0, 0.0],  # very close to electronics
            "i want to talk about planes": [0.05, 0.95, 0.0, 0.0, 0.0],  # very close to planes
            "tell me about jjiujitu": [0.05, 0.0, 0.85, 0.0, 0.1],  # close to jiu-jitsu (typo!)
            "tell me about cars please": [0.0, 0.05, 0.0, 0.9, 0.05],  # close to cars
            "summarize everything we discussed": [0.2, 0.2, 0.2, 0.2, 0.2],  # broad
            "what did we first talk about": [0.2, 0.2, 0.2, 0.2, 0.2],  # temporal
        }

        def embed(texts: list[str]) -> list[list[float]]:
            result = []
            for t in texts:
                t_lower = t.lower()
                if t_lower in vectors:
                    result.append(vectors[t_lower])
                elif t_lower in text_vectors:
                    result.append(text_vectors[t_lower])
                else:
                    # Default: slight general bias
                    result.append([0.1, 0.1, 0.1, 0.1, 0.6])
            return result

        return embed

    def _make_retriever_with_inbound(self, db_path, populate_vocab=True):
        """Build a retriever with an embedding-based inbound tagger."""
        from virtual_context.core.embedding_tag_generator import EmbeddingTagGenerator
        from virtual_context.core.turn_tag_index import TurnTagIndex
        from virtual_context.types import TurnTagEntry

        # Main tagger (LLM mock) for on_turn_complete
        main_tagger = MockTagGenerator(default_tag="test", default_tags=["test"])

        # Inbound tagger (embedding-based)
        tag_gen_config = TagGeneratorConfig(max_tags=5, min_tags=1)
        inbound = EmbeddingTagGenerator(
            config=tag_gen_config,
            similarity_threshold=0.3,
            embed_fn=self._make_embed_fn(),
        )

        store = SQLiteStore(db_path=db_path)
        turn_tag_index = TurnTagIndex()

        # Pre-populate vocabulary (inbound matching needs existing tags to match against)
        if populate_vocab:
            turn_tag_index.append(TurnTagEntry(
                turn_number=0, message_hash="h0",
                tags=["electronics", "arduino"], primary_tag="electronics",
            ))
            turn_tag_index.append(TurnTagEntry(
                turn_number=1, message_hash="h1",
                tags=["planes", "engineering", "design"], primary_tag="planes",
            ))
            turn_tag_index.append(TurnTagEntry(
                turn_number=2, message_hash="h2",
                tags=["jiu-jitsu", "craftsmanship"], primary_tag="jiu-jitsu",
            ))
            turn_tag_index.append(TurnTagEntry(
                turn_number=3, message_hash="h3",
                tags=["cars", "design", "engineering"], primary_tag="cars",
            ))

        config = RetrieverConfig(
            tag_context_max_tokens=30000,
            skip_active_tags=False,
            strategy_configs={
                "default": StrategyConfig(min_overlap=1, max_results=10),
            },
        )
        retriever = ContextRetriever(
            tag_generator=main_tagger,
            store=store,
            config=config,
            turn_tag_index=turn_tag_index,
            inbound_tagger=inbound,
        )
        return retriever, store, turn_tag_index

    def test_electronics_does_not_return_planes(self, tmp_sqlite_db):
        """Embedding match for 'arduino circuits' should NOT include 'planes'."""
        retriever, store, _ = self._make_retriever_with_inbound(tmp_sqlite_db)

        result = retriever.retrieve("tell me about arduino circuits")
        tags = result.retrieval_metadata.get("tags_from_message", [])

        assert "electronics" in tags or "arduino" in tags
        assert "planes" not in tags
        store.close()

    def test_topic_shift_detected(self, tmp_sqlite_db):
        """Switching from electronics to planes should pick up planes tag."""
        retriever, store, _ = self._make_retriever_with_inbound(tmp_sqlite_db)

        result = retriever.retrieve("I want to talk about planes")
        tags = result.retrieval_metadata.get("tags_from_message", [])

        assert "planes" in tags
        assert "electronics" not in tags or tags.index("planes") < tags.index("electronics")
        store.close()

    def test_typo_handled_via_embedding(self, tmp_sqlite_db):
        """Misspelled 'jjiujitu' should match 'jiu-jitsu' via embedding similarity."""
        retriever, store, _ = self._make_retriever_with_inbound(tmp_sqlite_db)

        # Pre-populate vocabulary by passing existing tags
        result = retriever.retrieve("tell me about jjiujitu")
        tags = result.retrieval_metadata.get("tags_from_message", [])

        assert "jiu-jitsu" in tags
        store.close()

    @pytest.mark.regression("BUG-008")
    def test_temporal_heuristic_applied(self, tmp_sqlite_db):
        """Embedding tagger doesn't detect temporal — heuristic should catch it."""
        retriever, store, _ = self._make_retriever_with_inbound(tmp_sqlite_db)

        result = retriever.retrieve("what did we first talk about")
        # "first talk about" matches temporal pattern
        assert result.retrieval_metadata.get("tags_from_message") is not None
        store.close()

    def test_inbound_tagger_not_used_for_main_tagger(self, tmp_sqlite_db):
        """The main tag_generator should not be affected by inbound_tagger."""
        retriever, store, _ = self._make_retriever_with_inbound(tmp_sqlite_db)

        # The main tagger is a MockTagGenerator that always returns "test"
        main_result = retriever.tag_generator.generate_tags("anything")
        assert main_result.tags == ["test"]
        store.close()

    def test_empty_vocabulary_returns_general(self, tmp_sqlite_db):
        """With no vocabulary, inbound matching returns _general (fallback)."""
        retriever, store, _ = self._make_retriever_with_inbound(
            tmp_sqlite_db, populate_vocab=False,
        )

        result = retriever.retrieve("tell me about arduino circuits")
        tags = result.retrieval_metadata.get("tags_from_message", [])

        assert tags == ["_general"]
        store.close()

    @pytest.mark.regression("BUG-009")
    def test_early_tag_visible_after_many_turns(self, tmp_sqlite_db):
        """Tags from early turns must remain in inbound vocabulary after >4 later turns.

        BUG-009: Retriever passed get_active_tags(lookback=4) as the inbound
        tagger vocabulary, so tags older than 4 turns were invisible. After
        history ingestion of 47 turns, the embedding tagger could only match
        against the last 4 turns' tags — everything else returned _general.
        """
        from virtual_context.core.embedding_tag_generator import EmbeddingTagGenerator
        from virtual_context.core.turn_tag_index import TurnTagIndex
        from virtual_context.types import TurnTagEntry

        main_tagger = MockTagGenerator(default_tag="test", default_tags=["test"])
        tag_gen_config = TagGeneratorConfig(max_tags=5, min_tags=1)
        inbound = EmbeddingTagGenerator(
            config=tag_gen_config,
            similarity_threshold=0.3,
            embed_fn=self._make_embed_fn(),
        )

        store = SQLiteStore(db_path=tmp_sqlite_db)
        turn_tag_index = TurnTagIndex()

        # "cars" at turn 0 — the target early tag
        turn_tag_index.append(TurnTagEntry(
            turn_number=0, message_hash="h0",
            tags=["cars", "design"], primary_tag="cars",
        ))
        # 10 filler turns with unrelated tags — pushes "cars" well outside lookback=4
        for i in range(1, 11):
            turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i}",
                tags=["planes", "engineering"], primary_tag="planes",
            ))

        config = RetrieverConfig(
            tag_context_max_tokens=30000,
            skip_active_tags=False,
            strategy_configs={"default": StrategyConfig(min_overlap=1, max_results=10)},
        )
        retriever = ContextRetriever(
            tag_generator=main_tagger,
            store=store,
            config=config,
            turn_tag_index=turn_tag_index,
            inbound_tagger=inbound,
        )

        # "cars" is 10 turns back — must still be in vocabulary
        result = retriever.retrieve("tell me about cars please")
        tags = result.retrieval_metadata.get("tags_from_message", [])

        assert "cars" in tags, (
            f"Expected 'cars' in vocabulary but got {tags}. "
            "Inbound tagger must see ALL TurnTagIndex tags, not just recent lookback."
        )
        store.close()


# ---------------------------------------------------------------------------
# Summary Floor (post-compaction fallback)
# ---------------------------------------------------------------------------

class TestSummaryFloor:
    """Post-compaction summary floor: inject tag summaries when retrieval is empty."""

    def _make_retriever_with_tag_summaries(self, db_path):
        """Build a retriever whose tagger always returns _general, with tag summaries in the store."""
        from virtual_context.types import TagSummary

        tag_gen = MockTagGenerator(default_tag="_general", default_tags=["_general"])
        tag_gen.set_override("_general", TagResult(tags=["_general"], primary="_general", source="fallback"))

        store = SQLiteStore(db_path=db_path)

        # Populate tag summaries (what compaction would produce)
        now = datetime.now(timezone.utc)
        store.save_tag_summary(TagSummary(
            tag="authentication",
            summary="JWT setup with RS256 signing and 1-hour expiry. Refresh token rotation.",
            summary_tokens=25,
            source_turn_numbers=[0, 1],
            created_at=now,
            updated_at=now,
        ))
        store.save_tag_summary(TagSummary(
            tag="cooking",
            summary="Aglio e olio recipe. Homemade tomato sauce with San Marzano tomatoes.",
            summary_tokens=20,
            source_turn_numbers=[2, 3],
            created_at=now,
            updated_at=now,
        ))

        config = RetrieverConfig(
            tag_context_max_tokens=30000,
            strategy_configs={"default": StrategyConfig()},
        )
        retriever = ContextRetriever(
            tag_generator=tag_gen,
            store=store,
            config=config,
        )
        return retriever, store

    def test_floor_activates_post_compaction(self, tmp_sqlite_db):
        """post_compaction=True + empty query tags → tag summaries returned."""
        retriever, store = self._make_retriever_with_tag_summaries(tmp_sqlite_db)

        result = retriever.retrieve("go back to that earlier thing", post_compaction=True)

        assert len(result.summaries) == 2
        assert result.total_tokens > 0
        assert result.retrieval_metadata.get("summary_floor") is True
        store.close()

    def test_floor_inactive_pre_compaction(self, tmp_sqlite_db):
        """post_compaction=False + empty query tags → no summaries (existing behavior)."""
        retriever, store = self._make_retriever_with_tag_summaries(tmp_sqlite_db)

        result = retriever.retrieve("go back to that earlier thing", post_compaction=False)

        assert len(result.summaries) == 0
        assert result.total_tokens == 0
        assert result.retrieval_metadata.get("summary_floor") is None
        store.close()

    def test_floor_skipped_when_normal_retrieval_succeeds(self, tmp_sqlite_db):
        """Specific tags + post_compaction=True → normal results, no floor."""
        tag_gen = MockTagGenerator(default_tag="legal", default_tags=["legal"])
        store = SQLiteStore(db_path=tmp_sqlite_db)

        now = datetime.now(timezone.utc)
        store.store_segment(StoredSegment(
            ref="legal-1",
            primary_tag="legal",
            tags=["legal"],
            summary="Court filing deadline discussion.",
            summary_tokens=15,
            full_tokens=80,
            metadata=SegmentMetadata(entities=[]),
            created_at=now,
            start_timestamp=now,
            end_timestamp=now,
        ))

        config = RetrieverConfig(
            tag_context_max_tokens=30000,
            strategy_configs={"default": StrategyConfig()},
        )
        retriever = ContextRetriever(
            tag_generator=tag_gen,
            store=store,
            config=config,
        )

        result = retriever.retrieve("Tell me about the court case", post_compaction=True)

        assert len(result.summaries) > 0
        assert result.retrieval_metadata.get("summary_floor") is None
        store.close()
