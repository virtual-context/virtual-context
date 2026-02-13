"""Tests for ContextRetriever."""

from datetime import datetime, timezone

import pytest

from virtual_context.classifiers.base import ClassifierPipeline
from virtual_context.classifiers.keyword import KeywordClassifier
from virtual_context.core.retriever import ContextRetriever
from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.types import (
    DomainDef,
    Message,
    RetrieverConfig,
    SegmentMetadata,
    StoredSegment,
)


@pytest.fixture
def domains():
    return [
        DomainDef(name="legal", keywords=["court", "filing", "attorney"], retrieval_limit=3, retrieval_max_tokens=5000),
        DomainDef(name="medical", keywords=["insulin", "doctor", "glucose"], retrieval_limit=3, retrieval_max_tokens=5000),
        DomainDef(name="_general", retrieval_limit=2, retrieval_max_tokens=2000),
    ]


async def _make_retriever(tmp_store_dir, domains, velocity_fallback=True, velocity_threshold=0.3):
    pipeline = ClassifierPipeline([KeywordClassifier()], min_confidence=0.3)
    await pipeline.initialize(domains)

    store = FilesystemStore(tmp_store_dir)

    # Pre-populate store with some segments
    now = datetime.now(timezone.utc)
    await store.store_segment(StoredSegment(
        ref="legal-1",
        domain="legal",
        summary="Discussion about case 24-cv-1234 filing deadline.",
        summary_tokens=20,
        full_tokens=100,
        metadata=SegmentMetadata(entities=["Case 24-cv-1234"]),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    ))
    await store.store_segment(StoredSegment(
        ref="medical-1",
        domain="medical",
        summary="Patient glucose levels elevated. Insulin adjustment discussed.",
        summary_tokens=25,
        full_tokens=120,
        metadata=SegmentMetadata(entities=["glucose", "insulin"]),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    ))

    config = RetrieverConfig(
        domains=domains,
        domain_context_max_tokens=30000,
        velocity_fallback=velocity_fallback,
        velocity_lookback=10,
        velocity_threshold=velocity_threshold,
    )
    return ContextRetriever(
        classifier_pipeline=pipeline,
        store=store,
        config=config,
    )


@pytest.fixture
async def retriever(tmp_store_dir, domains):
    return await _make_retriever(tmp_store_dir, domains)


@pytest.mark.asyncio
async def test_retrieve_legal(retriever):
    result = await retriever.retrieve("What about the court filing?")
    assert "legal" in result.domains_matched
    assert len(result.summaries) > 0
    assert result.summaries[0].domain == "legal"


@pytest.mark.asyncio
async def test_retrieve_medical(retriever):
    result = await retriever.retrieve("How is my insulin dosage?")
    assert "medical" in result.domains_matched


@pytest.mark.asyncio
async def test_skip_active_domains(retriever):
    result = await retriever.retrieve(
        "What about the court filing?",
        current_domains_in_context=["legal"],
    )
    assert "legal" not in result.domains_matched


@pytest.mark.asyncio
async def test_no_match_no_history_returns_empty(retriever):
    """No keyword match, no conversation history → nothing to compute velocity from."""
    result = await retriever.retrieve("The weather is nice today")
    assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# Velocity fallback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_velocity_fallback_legal(retriever):
    """8 legal turns then a vague message → velocity should retrieve legal."""
    history = [
        Message(role="user", content="What's the court filing deadline?"),
        Message(role="assistant", content="The filing is due January 30."),
        Message(role="user", content="Has the attorney reviewed the motion?"),
        Message(role="assistant", content="Yes, the attorney approved it."),
        Message(role="user", content="What about the court order on discovery?"),
        Message(role="assistant", content="The court granted the discovery motion."),
        Message(role="user", content="Did the attorney file the brief?"),
        Message(role="assistant", content="The brief was filed yesterday."),
    ]
    result = await retriever.retrieve(
        "What do you think about that?",
        conversation_history=history,
    )
    assert "legal" in result.domains_matched
    assert result.retrieval_metadata["used_velocity_fallback"] is True
    assert len(result.summaries) > 0


@pytest.mark.asyncio
async def test_velocity_fallback_medical(retriever):
    """Concentrated medical discussion → vague follow-up retrieves medical."""
    history = [
        Message(role="user", content="My glucose was 190 this morning."),
        Message(role="assistant", content="That's above target. Adjust insulin."),
        Message(role="user", content="Should I call my doctor?"),
        Message(role="assistant", content="Yes, schedule an appointment."),
        Message(role="user", content="What about my insulin dosage?"),
        Message(role="assistant", content="Consider increasing by 1 unit."),
    ]
    result = await retriever.retrieve(
        "Can you elaborate on that?",
        conversation_history=history,
    )
    assert "medical" in result.domains_matched
    assert result.retrieval_metadata["used_velocity_fallback"] is True


@pytest.mark.asyncio
async def test_velocity_decays_with_topic_shift(retriever):
    """Legal conversation followed by many general turns → velocity drops, no fallback."""
    history = [
        # 2 legal turns
        Message(role="user", content="What's the court filing deadline?"),
        Message(role="assistant", content="January 30."),
        # 8 general turns (no domain keywords)
        Message(role="user", content="What's the weather?"),
        Message(role="assistant", content="Sunny."),
        Message(role="user", content="Tell me a joke."),
        Message(role="assistant", content="Why did the chicken cross the road?"),
        Message(role="user", content="What time is it?"),
        Message(role="assistant", content="3pm."),
        Message(role="user", content="What's for dinner?"),
        Message(role="assistant", content="How about pasta?"),
    ]
    result = await retriever.retrieve(
        "What do you think?",
        conversation_history=history,
    )
    # Legal velocity = 1/5 = 0.2 which is below 0.3 threshold
    # So no fallback should fire
    assert "legal" not in result.domains_matched


@pytest.mark.asyncio
async def test_velocity_disabled(tmp_store_dir, domains):
    """When velocity_fallback=False, vague messages get nothing."""
    retriever = await _make_retriever(tmp_store_dir, domains, velocity_fallback=False)
    history = [
        Message(role="user", content="What's the court filing deadline?"),
        Message(role="assistant", content="January 30."),
        Message(role="user", content="Has the attorney reviewed the motion?"),
        Message(role="assistant", content="Yes."),
    ]
    result = await retriever.retrieve(
        "What do you think about that?",
        conversation_history=history,
    )
    assert result.total_tokens == 0
    assert result.retrieval_metadata["used_velocity_fallback"] is False


@pytest.mark.asyncio
async def test_velocity_scores_in_metadata(retriever):
    """Velocity scores should be reported in retrieval metadata."""
    history = [
        Message(role="user", content="Court filing motion attorney case"),
        Message(role="assistant", content="The court approved it."),
    ]
    result = await retriever.retrieve(
        "Tell me more.",
        conversation_history=history,
    )
    assert "velocity_scores" in result.retrieval_metadata
    scores = result.retrieval_metadata["velocity_scores"]
    if scores:
        assert "legal" in scores


@pytest.mark.asyncio
async def test_velocity_high_threshold(tmp_store_dir, domains):
    """High threshold requires very concentrated conversation."""
    retriever = await _make_retriever(tmp_store_dir, domains, velocity_threshold=0.9)
    history = [
        # 3 legal, 1 medical → legal velocity = 0.75, below 0.9
        Message(role="user", content="Court filing deadline?"),
        Message(role="assistant", content="January 30."),
        Message(role="user", content="Attorney reviewed motion?"),
        Message(role="assistant", content="Yes."),
        Message(role="user", content="What about the court order?"),
        Message(role="assistant", content="Granted."),
        Message(role="user", content="My glucose is high."),
        Message(role="assistant", content="Adjust insulin."),
    ]
    result = await retriever.retrieve(
        "What about that?",
        conversation_history=history,
    )
    # 0.75 < 0.9 threshold → no fallback
    assert result.total_tokens == 0
