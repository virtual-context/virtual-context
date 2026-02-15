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
    TagResult,
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
