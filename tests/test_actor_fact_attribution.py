from types import SimpleNamespace

from virtual_context.core.compaction_pipeline import CompactionPipeline
from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import (
    AUTHOR_ROLE_REQUESTER,
    AUTHOR_ROLE_SUBJECT,
    ActorRoster,
    CanonicalTurnRow,
    Fact,
    FactLane,
    Message,
    SOURCE_CANONICAL_TURN_IDS_KEY,
    TaggedSegment,
)


def _pipeline() -> CompactionPipeline:
    return object.__new__(CompactionPipeline)


def test_turn_group_zero_does_not_collapse_into_unset_provenance():
    pipeline = _pipeline()
    rows = [
        CanonicalTurnRow(
            conversation_id="c", canonical_turn_id="zero", turn_group_number=0,
        ),
        CanonicalTurnRow(
            conversation_id="c", canonical_turn_id="unset", turn_group_number=-1,
        ),
    ]
    pipeline._store = SimpleNamespace(get_all_canonical_turns=lambda _cid: rows)
    pipeline._config = SimpleNamespace(conversation_id="c")

    grouped = pipeline._physical_rows_by_group()

    assert [row.canonical_turn_id for row in grouped[0]] == ["zero"]
    assert [row.canonical_turn_id for row in grouped[-1]] == ["unset"]


def test_resolved_target_outside_segment_suppresses_copied_subject_lane():
    pipeline = _pipeline()
    target = CanonicalTurnRow(
        conversation_id="c", canonical_turn_id="target", user_content="claim",
        source_message_id="m1", sender_actor_id="actor:discord:bigtex",
        audience_conversation_id="guild", origin_channel_id="chan",
    )
    reply = CanonicalTurnRow(
        conversation_id="c", canonical_turn_id="reply", user_content="thoughts?",
        sender_actor_id="actor:discord:optics", reply_target_message_id="m1",
        reply_subject_actor_id="actor:discord:bigtex",
        reply_target_body="claim", reply_attribution_version=1,
        audience_conversation_id="guild", origin_channel_id="chan",
    )
    segment = TaggedSegment(messages=[
        Message(
            role="user", content="thoughts?",
            metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["reply"]},
        ),
    ])

    roster = pipeline._build_actor_roster(
        segment, {"target": target, "reply": reply},
    )

    assert roster.reply_bearing is True
    assert not any(lane.role == AUTHOR_ROLE_SUBJECT for lane in roster.lanes)


def test_incomplete_reply_mapping_never_stamps_a_lane_actor():
    compactor = object.__new__(DomainCompactor)
    compactor._extract_facts_for_text = lambda *_args, **_kwargs: [
        Fact(what="quoted claim"),
    ]
    roster = ActorRoster(
        complete=False,
        reply_bearing=True,
        actor_ids={"actor:discord:optics"},
        lanes=[FactLane(
            role=AUTHOR_ROLE_REQUESTER,
            text="claim",
            actor_id="actor:discord:optics",
            source_message_id="m1",
        )],
    )

    facts = compactor._extract_lane_facts(roster, segment=TaggedSegment())

    assert len(facts) == 1
    assert facts[0].author_actor_id == ""
    assert facts[0].author_source_role == AUTHOR_ROLE_REQUESTER
