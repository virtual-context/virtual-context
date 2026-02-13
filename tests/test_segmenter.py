"""Tests for TopicSegmenter."""

import pytest

from virtual_context.classifiers.base import ClassifierPipeline
from virtual_context.classifiers.keyword import KeywordClassifier
from virtual_context.core.segmenter import TopicSegmenter
from virtual_context.types import DomainDef, Message, SegmenterConfig


@pytest.fixture
def domains():
    return [
        DomainDef(name="legal", keywords=["court", "filing", "attorney", "motion", "case"]),
        DomainDef(name="medical", keywords=["insulin", "medication", "doctor", "glucose", "blood"]),
        DomainDef(name="_general"),
    ]


@pytest.fixture
async def segmenter(domains):
    pipeline = ClassifierPipeline([KeywordClassifier()], min_confidence=0.3)
    await pipeline.initialize(domains)
    return TopicSegmenter(
        classifier_pipeline=pipeline,
        config=SegmenterConfig(min_confidence=0.3),
        domains=domains,
    )


@pytest.mark.asyncio
async def test_segment_single_domain(segmenter):
    messages = [
        Message(role="user", content="What's the court filing deadline?"),
        Message(role="assistant", content="The filing is due January 30."),
        Message(role="user", content="Has the attorney reviewed the motion?"),
        Message(role="assistant", content="Yes, the attorney approved the motion."),
    ]
    segments = await segmenter.segment(messages)
    assert len(segments) == 1
    assert segments[0].domain == "legal"
    assert segments[0].turn_count == 2


@pytest.mark.asyncio
async def test_segment_two_domains(segmenter, mixed_messages):
    segments = await segmenter.segment(mixed_messages)
    assert len(segments) >= 2
    domains = {s.domain for s in segments}
    assert "legal" in domains
    assert "medical" in domains


@pytest.mark.asyncio
async def test_segment_empty(segmenter):
    segments = await segmenter.segment([])
    assert segments == []


@pytest.mark.asyncio
async def test_turn_pairing(segmenter):
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]
    pairs = segmenter._pair_turns(messages)
    assert len(pairs) == 1
    assert len(pairs[0].messages) == 2


@pytest.mark.asyncio
async def test_system_message_attachment(segmenter):
    messages = [
        Message(role="user", content="Hello"),
        Message(role="system", content="Tool result here"),
        Message(role="assistant", content="Hi"),
    ]
    pairs = segmenter._pair_turns(messages)
    # system/tool attaches to the current pair
    assert len(pairs) == 1
    assert len(pairs[0].messages) == 3
