"""Tests for TopicSegmenter (tag-based)."""

import pytest

from virtual_context.config import load_config, validate_config
from virtual_context.core.segmenter import TopicSegmenter
from virtual_context.types import Message, SegmenterConfig, TagResult, VirtualContextConfig

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


def test_segmenter_config_defaults():
    """SegmenterConfig exposes tag_overlap_threshold and max_segment_turns with defaults."""
    cfg = SegmenterConfig()
    assert cfg.tag_overlap_threshold == 0.5
    assert cfg.max_segment_turns == 20
    assert cfg.session_gap_minutes == 30  # existing field unchanged


def test_config_parses_tag_overlap_threshold(tmp_path):
    """tag_overlap_threshold and max_segment_turns parsed from compaction: section."""
    cfg_file = tmp_path / "vc.yaml"
    cfg_file.write_text(
        "compaction:\n"
        "  tag_overlap_threshold: 0.7\n"
        "  max_segment_turns: 15\n"
    )
    config = load_config(str(cfg_file))
    assert config.segmenter.tag_overlap_threshold == 0.7
    assert config.segmenter.max_segment_turns == 15


def test_config_defaults_without_tag_overlap(tmp_path):
    """Omitted fields get defaults."""
    cfg_file = tmp_path / "vc.yaml"
    cfg_file.write_text("compaction:\n  soft_threshold: 0.70\n")
    config = load_config(str(cfg_file))
    assert config.segmenter.tag_overlap_threshold == 0.5
    assert config.segmenter.max_segment_turns == 20


def test_validate_rejects_threshold_above_1():
    config = VirtualContextConfig()
    config.segmenter = SegmenterConfig(tag_overlap_threshold=1.5)
    errors = validate_config(config)
    assert any("tag_overlap_threshold" in e for e in errors)


def test_validate_rejects_threshold_below_0():
    config = VirtualContextConfig()
    config.segmenter = SegmenterConfig(tag_overlap_threshold=-0.1)
    errors = validate_config(config)
    assert any("tag_overlap_threshold" in e for e in errors)


def test_validate_rejects_negative_max_segment_turns():
    config = VirtualContextConfig()
    config.segmenter = SegmenterConfig(max_segment_turns=-1)
    errors = validate_config(config)
    assert any("max_segment_turns" in e for e in errors)


def test_validate_accepts_valid_segmenter_config():
    config = VirtualContextConfig()
    config.segmenter = SegmenterConfig(tag_overlap_threshold=0.7, max_segment_turns=10)
    errors = validate_config(config)
    assert not any("tag_overlap_threshold" in e for e in errors)
    assert not any("max_segment_turns" in e for e in errors)


def test_overlap_above_threshold_merges():
    """Turns sharing enough tags stay in the same segment even with different primaries."""
    gen = MockTagGenerator()
    # Turn 1: primary=saas-version, tags=[saas-version, pricing, product]
    gen.set_override("pricing tiers", TagResult(
        tags=["saas-version", "pricing", "product"], primary="saas-version", source="mock",
    ))
    # Turn 2: primary=agreement, tags=[agreement, saas-version]
    # Overlap with turn 1: {saas-version} / min(3, 2) = 1/2 = 0.5
    gen.set_override("yes exactly", TagResult(
        tags=["agreement", "saas-version"], primary="agreement", source="mock",
    ))

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5),
    )
    messages = [
        Message(role="user", content="What are the pricing tiers?"),
        Message(role="assistant", content="We have three tiers."),
        Message(role="user", content="yes exactly"),
        Message(role="assistant", content="Great."),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 1
    assert segments[0].turn_count == 2
