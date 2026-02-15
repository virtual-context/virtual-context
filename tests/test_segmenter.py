"""Tests for TopicSegmenter (tag-based)."""

import pytest

from virtual_context.core.segmenter import TopicSegmenter
from virtual_context.types import Message, SegmenterConfig, TagResult

from conftest import MockTagGenerator


@pytest.fixture
def tag_generator():
    gen = MockTagGenerator(default_tag="legal", default_tags=["legal"])
    gen.set_override("insulin", TagResult(tags=["medical"], primary="medical", source="mock"))
    gen.set_override("glucose", TagResult(tags=["medical"], primary="medical", source="mock"))
    gen.set_override("doctor", TagResult(tags=["medical"], primary="medical", source="mock"))
    return gen


@pytest.fixture
def segmenter(tag_generator):
    return TopicSegmenter(
        tag_generator=tag_generator,
        config=SegmenterConfig(),
    )


def test_segment_single_tag(segmenter):
    messages = [
        Message(role="user", content="What's the court filing deadline?"),
        Message(role="assistant", content="The filing is due January 30."),
        Message(role="user", content="Has the attorney reviewed the motion?"),
        Message(role="assistant", content="Yes, the attorney approved the motion."),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 1
    assert segments[0].primary_tag == "legal"
    assert segments[0].turn_count == 2


def test_segment_two_tags(segmenter, mixed_messages):
    segments = segmenter.segment(mixed_messages)
    assert len(segments) >= 2
    tags = {s.primary_tag for s in segments}
    assert "legal" in tags
    assert "medical" in tags


def test_segment_empty(segmenter):
    segments = segmenter.segment([])
    assert segments == []


def test_turn_pairing(segmenter):
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]
    pairs = segmenter._pair_turns(messages)
    assert len(pairs) == 1
    assert len(pairs[0].messages) == 2


def test_system_message_attachment(segmenter):
    messages = [
        Message(role="user", content="Hello"),
        Message(role="system", content="Tool result here"),
        Message(role="assistant", content="Hi"),
    ]
    pairs = segmenter._pair_turns(messages)
    assert len(pairs) == 1
    assert len(pairs[0].messages) == 3


def test_segment_tags_union(segmenter):
    """Tags from all turn pairs should be unioned in the segment."""
    # All messages will get "legal" tag from mock
    messages = [
        Message(role="user", content="Court filing"),
        Message(role="assistant", content="Done"),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 1
    assert "legal" in segments[0].tags
