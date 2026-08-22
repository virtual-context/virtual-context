"""Tests for TopicSegmenter (tag-based)."""

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.config import load_config, validate_config
from virtual_context.core.segmenter import TopicSegmenter
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    Message,
    SegmenterConfig,
    TagResult,
    TurnTagEntry,
    VirtualContextConfig,
)
from virtual_context.core.turn_tag_index import TurnTagIndex

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


def _physical_user(
    content: str,
    canonical_id: str,
    *,
    actor: str = "actor:discord:111",
    audience: str = "audience:guild:1",
    channel: str = "channel:legal",
    label: str = "Alex",
) -> Message:
    return Message(
        role="user",
        content=content,
        metadata={
            "sender": {"name": label},
            "_vc_source_canonical_turn_ids": [canonical_id],
        },
        source_actor_id=actor,
        source_audience_conversation_id=audience,
        source_origin_channel_id=channel,
        source_audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
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


def test_same_tag_different_known_senders_stay_separate(segmenter):
    messages = [
        Message(
            role="user",
            content="First legal question",
            metadata={"sender": {"name": "BigTex"}},
        ),
        Message(role="assistant", content="First answer"),
        Message(
            role="user",
            content="Second legal question",
            metadata={"sender": {"name": "Vast"}},
        ),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 2
    assert [segment.turn_count for segment in segments] == [1, 1]


def test_same_display_name_different_actor_ids_stay_separate(segmenter):
    def metadata(actor_id: str) -> dict:
        return {
            "sender": {"name": "Alex"},
            "actor": {"platform": "discord", "user_id": actor_id},
        }

    messages = [
        Message(
            role="user",
            content="First legal question",
            metadata=metadata("111"),
        ),
        Message(role="assistant", content="First answer"),
        Message(
            role="user",
            content="Second legal question",
            metadata=metadata("222"),
        ),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 2
    assert [segment.turn_count for segment in segments] == [1, 1]


def test_actorless_physical_rows_with_same_display_name_stay_separate(segmenter):
    def metadata(source_id: str) -> dict:
        return {
            "sender": {"name": "Alex"},
            "_vc_source_canonical_turn_ids": [source_id],
        }

    messages = [
        Message(role="user", content="First legal question", metadata=metadata("ct-1")),
        Message(role="assistant", content="First answer"),
        Message(role="user", content="Second legal question", metadata=metadata("ct-2")),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 2
    assert [segment.turn_count for segment in segments] == [1, 1]


def test_same_actor_id_can_rejoin_after_display_name_change(segmenter):
    def metadata(name: str) -> dict:
        return {
            "sender": {"name": name},
            "actor": {"platform": "discord", "user_id": "111"},
        }

    messages = [
        Message(role="user", content="First legal question", metadata=metadata("Alex")),
        Message(role="assistant", content="First answer"),
        Message(role="user", content="Second legal question", metadata=metadata("A. Lex")),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 1
    assert segments[0].turn_count == 2


def test_same_actor_audience_and_channel_can_rejoin(segmenter):
    messages = [
        _physical_user("First legal question", "ct-1", label="Alex"),
        Message(role="assistant", content="First answer"),
        _physical_user("Second legal question", "ct-2", label="A. Lex"),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 1
    assert segments[0].turn_count == 2


@pytest.mark.parametrize(
    ("second_audience", "second_channel"),
    [
        ("audience:guild:2", "channel:legal"),
        ("audience:guild:1", "channel:medical"),
    ],
    ids=["different-audience", "different-channel"],
)
def test_same_actor_cross_scope_stays_separate(
    segmenter,
    second_audience,
    second_channel,
):
    messages = [
        _physical_user("First legal question", "ct-1"),
        Message(role="assistant", content="First answer"),
        _physical_user(
            "Second legal question",
            "ct-2",
            audience=second_audience,
            channel=second_channel,
        ),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 2
    assert [segment.turn_count for segment in segments] == [1, 1]


def test_same_actor_same_dm_audience_with_empty_channel_can_rejoin(segmenter):
    messages = [
        _physical_user("First legal question", "ct-1", channel=""),
        Message(role="assistant", content="First answer"),
        _physical_user("Second legal question", "ct-2", channel=""),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 1
    assert segments[0].turn_count == 2


def test_same_sender_can_rejoin_noncontiguous_topic_segment():
    gen = MockTagGenerator()
    gen.set_override("alpha-one", TagResult(
        tags=["alpha"], primary="alpha", source="mock",
    ))
    gen.set_override("beta", TagResult(
        tags=["beta"], primary="beta", source="mock",
    ))
    gen.set_override("alpha-two", TagResult(
        tags=["alpha"], primary="alpha", source="mock",
    ))
    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5),
    )
    messages = [
        Message(
            role="user",
            content="alpha-one question",
            metadata={"sender": {"name": " BigTex "}},
        ),
        Message(role="assistant", content="alpha-one answer"),
        Message(
            role="user",
            content="beta question",
            metadata={"sender": {"name": "BIGTEX"}},
        ),
        Message(role="assistant", content="beta answer"),
        Message(
            role="user",
            content="alpha-two question",
            metadata={"sender": {"name": "bigtex"}},
        ),
        Message(role="assistant", content="alpha-two answer"),
    ]

    segments = segmenter.segment(messages)

    assert [segment.primary_tag for segment in segments] == ["alpha", "beta"]
    assert [segment.turn_count for segment in segments] == [2, 1]


def test_unknown_sender_isolated_from_known_sender(segmenter):
    messages = [
        Message(
            role="user",
            content="Known sender question",
            metadata={"sender": {"name": "BigTex"}},
            source_actor_id="actor:discord:1",
        ),
        Message(role="assistant", content="First answer"),
        Message(
            role="user",
            content="Unknown sender follow-up",
            metadata={"_vc_source_canonical_turn_ids": ["ct-unknown"]},
        ),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 2
    assert [segment.turn_count for segment in segments] == [1, 1]


def test_two_unknown_senders_are_isolated(segmenter):
    messages = [
        Message(
            role="user",
            content="First unresolved question",
            metadata={"_vc_source_canonical_turn_ids": ["ct-1"]},
        ),
        Message(role="assistant", content="First answer"),
        Message(
            role="user",
            content="Second unresolved question",
            metadata={"_vc_source_canonical_turn_ids": ["ct-2"]},
        ),
        Message(role="assistant", content="Second answer"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 2
    assert [segment.turn_count for segment in segments] == [1, 1]


def test_split_physical_humans_reuse_logical_tag_index_without_shifting():
    index = TurnTagIndex()
    index.append(TurnTagEntry(
        turn_number=10,
        tags=["medical"],
        primary_tag="medical",
    ))
    index.append(TurnTagEntry(
        turn_number=11,
        tags=["legal"],
        primary_tag="legal",
    ))
    segmenter = TopicSegmenter(
        tag_generator=MockTagGenerator(
            default_tag="wrong",
            default_tags=["wrong"],
        ),
        config=SegmenterConfig(),
        turn_tag_index=index,
    )
    messages = [
        Message(
            role="user",
            content="BigTex physical words",
            source_actor_id="actor:discord:1",
            source_logical_turn_number=10,
        ),
        Message(
            role="user",
            content="Kuw physical words",
            source_actor_id="actor:discord:2",
            source_logical_turn_number=10,
        ),
        Message(
            role="assistant",
            content="shared response",
            source_logical_turn_number=10,
        ),
        Message(
            role="user",
            content="next logical turn",
            source_actor_id="actor:discord:1",
            source_logical_turn_number=11,
        ),
        Message(role="assistant", content="next response", source_logical_turn_number=11),
    ]

    segments = segmenter.segment(messages, turn_offset=10)

    assert [segment.primary_tag for segment in segments] == [
        "medical", "medical", "legal",
    ]


def test_segment_two_tags(segmenter, mixed_messages):
    segments = segmenter.segment(mixed_messages)
    assert len(segments) >= 2
    tags = {s.primary_tag for s in segments}
    assert "legal" in tags
    assert "medical" in tags


def test_segment_empty(segmenter):
    segments = segmenter.segment([])
    assert segments == []


def test_segment_progress_reports_monotonic_turns(segmenter):
    messages = []
    for i in range(6):
        messages.append(Message(role="user", content=f"Question {i}"))
        messages.append(Message(role="assistant", content=f"Answer {i}"))

    events = []

    def on_progress(done, total, result, **kwargs):
        events.append({
            "done": done,
            "total": total,
            "segments": kwargs.get("segments"),
            "phase_name": kwargs.get("phase_name"),
            "elapsed_ms": kwargs.get("elapsed_ms"),
        })

    segments = segmenter.segment(messages, progress_callback=on_progress)

    assert len(segments) == 1
    assert events
    assert events[0]["done"] == 0
    assert events[0]["total"] == 6
    assert events[0]["phase_name"] == "segment_tagging"
    assert events[-1]["phase_name"] == "segment_postprocess"
    assert events[-1]["done"] == events[-1]["total"]
    assert events[-1]["segments"] == len(segments)
    assert {"segment_tagging", "segment_grouping", "segment_postprocess"}.issubset(
        {evt["phase_name"] for evt in events},
    )

    events_by_phase: dict[str, list[dict]] = {}
    for evt in events:
        events_by_phase.setdefault(str(evt["phase_name"]), []).append(evt)
    for phase_events in events_by_phase.values():
        assert all(
            later["done"] >= earlier["done"]
            for earlier, later in zip(phase_events, phase_events[1:], strict=False)
        )


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


def test_overlap_below_threshold_splits():
    """Turns with no tag overlap split into separate segments."""
    gen = MockTagGenerator()
    gen.set_override("pricing", TagResult(
        tags=["saas-version", "pricing"], primary="saas-version", source="mock",
    ))
    gen.set_override("doctor", TagResult(
        tags=["medical", "health"], primary="medical", source="mock",
    ))

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5),
    )
    messages = [
        Message(role="user", content="What about pricing?"),
        Message(role="assistant", content="Three tiers."),
        Message(role="user", content="I need a doctor"),
        Message(role="assistant", content="Let me help."),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 2
    assert segments[0].primary_tag == "saas-version"
    assert segments[1].primary_tag == "medical"


def test_general_only_turn_merges():
    """A turn tagged only with _general merges into the preceding segment."""
    gen = MockTagGenerator()
    gen.set_override("pricing", TagResult(
        tags=["saas-version", "pricing"], primary="saas-version", source="mock",
    ))
    gen.set_override("ok", TagResult(
        tags=["_general"], primary="_general", source="mock",
    ))

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5),
    )
    messages = [
        Message(role="user", content="What about pricing?"),
        Message(role="assistant", content="Three tiers."),
        Message(role="user", content="ok"),
        Message(role="assistant", content="Sure."),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 1
    assert segments[0].turn_count == 2


def test_max_segment_turns_cap():
    """Segment splits when max_segment_turns is reached, even with full overlap."""
    gen = MockTagGenerator(default_tag="topic-a", default_tags=["topic-a", "topic-b"])

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5, max_segment_turns=3),
    )
    messages = []
    for i in range(5):
        messages.append(Message(role="user", content=f"Question {i}"))
        messages.append(Message(role="assistant", content=f"Answer {i}"))

    segments = segmenter.segment(messages)
    assert len(segments) == 2
    assert segments[0].turn_count == 3
    assert segments[1].turn_count == 2


def test_max_segment_turns_zero_unlimited():
    """max_segment_turns=0 disables the cap."""
    gen = MockTagGenerator(default_tag="topic-a", default_tags=["topic-a"])

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5, max_segment_turns=0),
    )
    messages = []
    for i in range(25):
        messages.append(Message(role="user", content=f"Question {i}"))
        messages.append(Message(role="assistant", content=f"Answer {i}"))

    segments = segmenter.segment(messages)
    assert len(segments) == 1
    assert segments[0].turn_count == 25


def test_threshold_zero_never_splits_on_tags():
    """threshold=0.0 means only session/temporal/cap splits apply."""
    gen = MockTagGenerator()
    gen.set_override("pricing", TagResult(
        tags=["saas-version"], primary="saas-version", source="mock",
    ))
    gen.set_override("doctor", TagResult(
        tags=["medical"], primary="medical", source="mock",
    ))

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.0, max_segment_turns=0),
    )
    messages = [
        Message(role="user", content="What about pricing?"),
        Message(role="assistant", content="Three tiers."),
        Message(role="user", content="I need a doctor"),
        Message(role="assistant", content="Help."),
    ]
    segments = segmenter.segment(messages)
    # overlap=0.0, threshold=0.0 → 0.0 < 0.0 is False → no split
    assert len(segments) == 1


def test_threshold_one_splits_unless_identical():
    """threshold=1.0 splits unless tags are identical sets."""
    gen = MockTagGenerator()
    gen.set_override("pricing", TagResult(
        tags=["saas-version", "pricing"], primary="saas-version", source="mock",
    ))
    gen.set_override("sounds good", TagResult(
        tags=["saas-version", "agreement"], primary="agreement", source="mock",
    ))

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=1.0),
    )
    messages = [
        Message(role="user", content="What about pricing?"),
        Message(role="assistant", content="Three tiers."),
        Message(role="user", content="sounds good"),
        Message(role="assistant", content="Great."),
    ]
    segments = segmenter.segment(messages)
    # {saas-version, pricing} vs {saas-version, agreement}
    # intersection={saas-version}, min(2,2)=2, overlap=0.5 < 1.0 → splits
    assert len(segments) == 2


def test_gradual_topic_drift_stops_at_anchor_boundary():
    """Bridge tags cannot walk a segment away from its anchor topic.

    A=[alpha, beta], B=[beta, gamma], C=[gamma, delta]
    A→B overlap: {beta}/min(2,2) = 0.5 → merge
    A→C overlap: none → split.  The accumulated B tag must not turn the
    segment into a semantic grab bag.
    """
    gen = MockTagGenerator()
    gen.set_override("turn-a", TagResult(
        tags=["alpha", "beta"], primary="alpha", source="mock",
    ))
    gen.set_override("turn-b", TagResult(
        tags=["beta", "gamma"], primary="beta", source="mock",
    ))
    gen.set_override("turn-c", TagResult(
        tags=["gamma", "delta"], primary="gamma", source="mock",
    ))

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5, max_segment_turns=0),
    )
    messages = [
        Message(role="user", content="turn-a question"),
        Message(role="assistant", content="turn-a answer"),
        Message(role="user", content="turn-b question"),
        Message(role="assistant", content="turn-b answer"),
        Message(role="user", content="turn-c question"),
        Message(role="assistant", content="turn-c answer"),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 2
    assert segments[0].turn_count == 2
    assert segments[1].turn_count == 1


def test_general_turn_attaches_to_immediately_preceding_topic():
    """A general acknowledgement cannot jump to the first library entry."""
    gen = MockTagGenerator()
    gen.set_override("pricing", TagResult(
        tags=["pricing"], primary="pricing", source="mock",
    ))
    gen.set_override("doctor", TagResult(
        tags=["medical"], primary="medical", source="mock",
    ))
    gen.set_override("ok", TagResult(
        tags=["_general"], primary="_general", source="mock",
    ))
    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5),
    )
    messages = [
        Message(role="user", content="pricing question"),
        Message(role="assistant", content="pricing answer"),
        Message(role="user", content="doctor question"),
        Message(role="assistant", content="medical answer"),
        Message(role="user", content="ok"),
        Message(role="assistant", content="sure"),
    ]

    segments = segmenter.segment(messages)

    assert len(segments) == 2
    assert segments[0].primary_tag == "pricing"
    assert segments[0].turn_count == 1
    assert segments[1].primary_tag == "medical"
    assert segments[1].turn_count == 2


def test_initial_general_turn_cannot_anchor_following_substantive_topic():
    """A greeting remains isolated instead of becoming the topic anchor."""
    gen = MockTagGenerator()
    gen.set_override("hello", TagResult(
        tags=["_general"], primary="_general", source="mock",
    ))
    gen.set_override("medical", TagResult(
        tags=["medical"], primary="medical", source="mock",
    ))
    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5),
    )
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
        Message(role="user", content="medical question"),
        Message(role="assistant", content="medical answer"),
    ]

    segments = segmenter.segment(messages)

    assert [segment.primary_tag for segment in segments] == ["_general", "medical"]
    assert [segment.turn_count for segment in segments] == [1, 1]


def test_last_appended_primary_cannot_move_segment_anchor():
    """A bridge turn's primary tag does not become the segment identity."""
    gen = MockTagGenerator()
    gen.set_override("turn-a", TagResult(
        tags=["alpha", "beta"], primary="alpha", source="mock",
    ))
    gen.set_override("turn-b", TagResult(
        tags=["beta", "gamma"], primary="beta", source="mock",
    ))
    gen.set_override("turn-c", TagResult(
        tags=["gamma", "delta"], primary="beta", source="mock",
    ))
    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5, max_segment_turns=0),
    )
    messages = [
        Message(role="user", content="turn-a question"),
        Message(role="assistant", content="turn-a answer"),
        Message(role="user", content="turn-b question"),
        Message(role="assistant", content="turn-b answer"),
        Message(role="user", content="turn-c question"),
        Message(role="assistant", content="turn-c answer"),
    ]

    segments = segmenter.segment(messages)

    assert [segment.turn_count for segment in segments] == [2, 1]


def test_gradual_drift_capped_by_max_segment_turns():
    """max_segment_turns breaks a drifting segment."""
    gen = MockTagGenerator()
    gen.set_override("turn-a", TagResult(
        tags=["alpha", "beta"], primary="alpha", source="mock",
    ))
    gen.set_override("turn-b", TagResult(
        tags=["beta", "gamma"], primary="beta", source="mock",
    ))
    gen.set_override("turn-c", TagResult(
        tags=["gamma", "delta"], primary="gamma", source="mock",
    ))

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5, max_segment_turns=2),
    )
    messages = [
        Message(role="user", content="turn-a question"),
        Message(role="assistant", content="turn-a answer"),
        Message(role="user", content="turn-b question"),
        Message(role="assistant", content="turn-b answer"),
        Message(role="user", content="turn-c question"),
        Message(role="assistant", content="turn-c answer"),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 2
    assert segments[0].turn_count == 2
    assert segments[1].turn_count == 1


def test_session_date_change_still_splits():
    """Session date header forces a split regardless of tag overlap."""
    gen = MockTagGenerator(default_tag="topic-a", default_tags=["topic-a", "topic-b"])

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(tag_overlap_threshold=0.5),
    )
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
        Message(role="user", content="[Session from 2026-03-16] Hello again"),
        Message(role="assistant", content="Welcome back"),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 2


def test_temporal_gap_still_splits():
    """Temporal gap forces a split regardless of tag overlap."""
    gen = MockTagGenerator(default_tag="topic-a", default_tags=["topic-a", "topic-b"])
    now = datetime.now(timezone.utc)

    segmenter = TopicSegmenter(
        tag_generator=gen,
        config=SegmenterConfig(
            tag_overlap_threshold=0.5,
            session_gap_minutes=30,
        ),
    )
    messages = [
        Message(role="user", content="Hello", timestamp=now),
        Message(role="assistant", content="Hi", timestamp=now + timedelta(minutes=1)),
        Message(role="user", content="Hello again", timestamp=now + timedelta(minutes=60)),
        Message(role="assistant", content="Welcome back", timestamp=now + timedelta(minutes=61)),
    ]
    segments = segmenter.segment(messages)
    assert len(segments) == 2
