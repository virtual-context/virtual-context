"""Tests for DomainCompactor (tag-based)."""

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import MockLLMProvider
from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import CompactorConfig, Message, TaggedSegment, TagPromptRule


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def compactor(mock_llm):
    return DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )


@pytest.fixture
def legal_segment(ts):
    return TaggedSegment(
        primary_tag="legal",
        tags=["legal", "court"],
        messages=[
            Message(role="user", content="What's the court filing deadline?", timestamp=ts),
            Message(role="assistant", content="The filing is due January 30.", timestamp=ts + timedelta(seconds=30)),
        ],
        token_count=50,
        start_timestamp=ts,
        end_timestamp=ts + timedelta(seconds=30),
        turn_count=1,
    )


def test_compact_single(compactor, legal_segment, mock_llm):
    results = compactor.compact([legal_segment])
    assert len(results) == 1
    assert results[0].primary_tag == "legal"
    assert results[0].summary == "Test summary"
    assert len(mock_llm.calls) == 1


def test_compact_preserves_metadata(compactor, legal_segment):
    results = compactor.compact([legal_segment])
    assert results[0].metadata.entities == ["entity1"]
    assert results[0].metadata.key_decisions == ["decision1"]


@pytest.mark.regression("BUG-004")
def test_compact_refined_tags(compactor, legal_segment):
    results = compactor.compact([legal_segment])
    assert "test-tag" in results[0].tags  # From mock LLM response


def test_compact_multiple(compactor):
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    segments = [
        TaggedSegment(
            primary_tag="legal",
            tags=["legal"],
            messages=[Message(role="user", content="Court case update")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
        TaggedSegment(
            primary_tag="medical",
            tags=["medical"],
            messages=[Message(role="user", content="Blood test results")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
    ]
    results = compactor.compact(segments)
    assert len(results) == 2


def test_format_conversation(compactor):
    ts = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
    messages = [
        Message(role="user", content="Hello", timestamp=ts),
        Message(role="assistant", content="Hi there", timestamp=ts),
    ]
    text = compactor._format_conversation(messages)
    assert "User (10:30): Hello" in text
    assert "Assistant (10:30): Hi there" in text


def test_parse_response_valid(compactor):
    result = compactor._parse_response('{"summary": "test", "entities": ["a"]}')
    assert result["summary"] == "test"


def test_parse_response_with_fences(compactor):
    result = compactor._parse_response('```json\n{"summary": "test"}\n```')
    assert result["summary"] == "test"


def test_parse_response_with_thinking(compactor):
    result = compactor._parse_response('<think>analyzing...</think>{"summary": "test"}')
    assert result["summary"] == "test"


def test_parse_response_fallback(compactor):
    result = compactor._parse_response("Just plain text summary")
    assert result["summary"] == "Just plain text summary"


def test_custom_prompt_from_tag_rules():
    """Custom summary prompt should be used when tag matches a rule."""
    rules = [
        TagPromptRule(match="legal*", summary_prompt="Summarize legal matters carefully."),
    ]
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(),
        tag_rules=rules,
    )
    prompt = compactor._get_prompt_for_tags(["legal-case", "court"])
    assert prompt == "Summarize legal matters carefully."


def test_no_custom_prompt_for_unmatched_tags():
    rules = [
        TagPromptRule(match="legal*", summary_prompt="Summarize legal matters."),
    ]
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(),
        tag_rules=rules,
    )
    prompt = compactor._get_prompt_for_tags(["medical", "health"])
    assert prompt is None


def test_compact_tag_summaries_builds(mock_llm):
    """compact_tag_summaries builds summaries for tags with segments."""
    from virtual_context.types import StoredSummary, TagSummary

    compactor = DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(),
    )

    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    summaries = [
        StoredSummary(
            ref="seg-1", primary_tag="legal", tags=["legal"],
            summary="Case discussion", summary_tokens=20,
            created_at=now, start_timestamp=now, end_timestamp=now,
        ),
    ]

    result = compactor.compact_tag_summaries(
        cover_tags=["legal"],
        tag_to_summaries={"legal": summaries},
        tag_to_turns={"legal": [0, 1]},
        existing_tag_summaries={},
        max_turn=5,
    )
    assert len(result) == 1
    assert result[0].tag == "legal"
    assert result[0].covers_through_turn == 5
    assert len(mock_llm.calls) == 1


@pytest.mark.regression("BUG-003")
def test_compact_tag_summaries_skips_fresh(mock_llm):
    """compact_tag_summaries skips tags where existing summary is fresh."""
    from virtual_context.types import StoredSummary, TagSummary

    compactor = DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(),
    )

    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    summaries = [
        StoredSummary(
            ref="seg-1", primary_tag="legal", tags=["legal"],
            summary="Case discussion", summary_tokens=20,
            created_at=now, start_timestamp=now, end_timestamp=now,
        ),
    ]
    existing = TagSummary(
        tag="legal", summary="Already fresh", summary_tokens=20,
        covers_through_turn=10,  # >= max_turn
        created_at=now, updated_at=now,
    )

    result = compactor.compact_tag_summaries(
        cover_tags=["legal"],
        tag_to_summaries={"legal": summaries},
        tag_to_turns={"legal": [0, 1]},
        existing_tag_summaries={"legal": existing},
        max_turn=5,  # existing covers_through_turn (10) >= max_turn (5)
    )
    assert len(result) == 0  # Nothing to build
    assert len(mock_llm.calls) == 0  # No LLM calls
