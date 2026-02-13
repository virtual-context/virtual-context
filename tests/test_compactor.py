"""Tests for DomainCompactor."""

from datetime import datetime, timezone

import pytest

from tests.conftest import MockLLMProvider
from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import CompactorConfig, DomainSegment, Message


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
    from datetime import timedelta
    return DomainSegment(
        domain="legal",
        messages=[
            Message(role="user", content="What's the court filing deadline?", timestamp=ts),
            Message(role="assistant", content="The filing is due January 30.", timestamp=ts + timedelta(seconds=30)),
        ],
        token_count=50,
        start_timestamp=ts,
        end_timestamp=ts + timedelta(seconds=30),
        turn_count=1,
        confidence=0.8,
    )


@pytest.mark.asyncio
async def test_compact_single(compactor, legal_segment, mock_llm):
    results = await compactor.compact([legal_segment])
    assert len(results) == 1
    assert results[0].domain == "legal"
    assert results[0].summary == "Test summary"
    assert len(mock_llm.calls) == 1


@pytest.mark.asyncio
async def test_compact_preserves_metadata(compactor, legal_segment):
    results = await compactor.compact([legal_segment])
    assert results[0].metadata.entities == ["entity1"]
    assert results[0].metadata.key_decisions == ["decision1"]


@pytest.mark.asyncio
async def test_compact_multiple(compactor):
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    segments = [
        DomainSegment(
            domain="legal",
            messages=[Message(role="user", content="Court case update")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
        DomainSegment(
            domain="medical",
            messages=[Message(role="user", content="Blood test results")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
    ]
    results = await compactor.compact(segments)
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


def test_parse_response_fallback(compactor):
    result = compactor._parse_response("Just plain text summary")
    assert result["summary"] == "Just plain text summary"
