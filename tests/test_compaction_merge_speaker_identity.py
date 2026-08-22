"""Speaker-identity admission for stored-segment merge compaction."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.config import load_config
from virtual_context.core.compaction_pipeline import CompactionPipeline
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    AUTHOR_ROLE_SUBJECT,
    SOURCE_CANONICAL_TURN_IDS_KEY,
    CanonicalTurnRow,
    CompactionResult,
    Message,
    SegmentMetadata,
    StoredSegment,
    TaggedSegment,
)


NEW_ACTOR = "actor:discord:new-speaker"
OTHER_ACTOR = "actor:discord:other-speaker"


def _source_message(
    role: str,
    content: str,
    canonical_id: str,
    *,
    channel: str = "channel:shared",
) -> Message:
    return Message(
        role=role,
        content=content,
        metadata={SOURCE_CANONICAL_TURN_IDS_KEY: [canonical_id]},
        source_actor_id=NEW_ACTOR if role == "user" else "",
        source_audience_conversation_id="audience:guild:1",
        source_origin_channel_id=channel,
        source_audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
    )


def _engine(tmp_path) -> VirtualContextEngine:
    config = load_config(config_dict={
        "tenant_id": "tenant-merge-speaker",
        "conversation_id": "sk:agent:test:discord:group:merge-speaker",
        "context_window": 10000,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / "merge-speaker.db")},
        },
        "tag_generator": {"type": "keyword"},
        "compaction": {
            "merge_lookback": 10,
            "merge_overlap_threshold": 0.1,
        },
    })
    engine = VirtualContextEngine(config=config)
    semantic = MagicMock()
    semantic.get_embed_fn.return_value = None
    engine._compaction._semantic = semantic
    return engine


def _seed_row(
    engine: VirtualContextEngine,
    *,
    canonical_id: str,
    actor_id: str,
    sender: str,
    group: int,
    audience: str = "audience:guild:1",
    channel: str = "channel:shared",
) -> None:
    engine._store.save_canonical_turn(
        engine.config.conversation_id,
        -1,
        f"shared topic question {canonical_id}",
        f"shared topic answer {canonical_id}",
        canonical_turn_id=canonical_id,
        turn_group_number=group,
        sort_key=float((group + 1) * 1000),
        turn_hash=f"hash-{canonical_id}",
        primary_tag="shared-topic",
        tags=["shared-topic"],
        sender=sender,
        sender_actor_id=actor_id,
        audience_conversation_id=audience,
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        origin_channel_id=channel,
    )


def _stored_candidate(
    engine: VirtualContextEngine,
    *,
    canonical_ids: list[str],
    mapping_complete: bool,
) -> None:
    now = datetime.now(timezone.utc)
    engine._store.store_segment(StoredSegment(
        ref="stored-old",
        conversation_id=engine.config.conversation_id,
        primary_tag="shared-topic",
        tags=["shared-topic"],
        summary="Earlier discussion of the shared topic.",
        summary_tokens=7,
        full_text="Earlier discussion of the shared topic in detail.",
        full_tokens=20,
        messages=[{"role": "user", "content": "legacy derived message"}],
        metadata=SegmentMetadata(
            turn_count=len(canonical_ids),
            canonical_turn_ids=canonical_ids,
            start_turn_number=0,
            end_turn_number=max(0, len(canonical_ids) - 1),
            source_mapping_complete=mapping_complete,
        ),
        created_at=now - timedelta(days=1),
        start_timestamp=now - timedelta(days=1),
        end_timestamp=now - timedelta(days=1),
    ))


def _new_segment(*, channel: str = "channel:shared") -> TaggedSegment:
    now = datetime.now(timezone.utc)
    return TaggedSegment(
        id="seg-new",
        primary_tag="shared-topic",
        tags=["shared-topic"],
        messages=[
            _source_message(
                "user", "new shared topic question", "ct-new", channel=channel,
            ),
            _source_message(
                "assistant", "new shared topic answer", "ct-new", channel=channel,
            ),
        ],
        token_count=12,
        start_timestamp=now,
        end_timestamp=now,
        turn_count=1,
    )


def _install_compactor(engine: VirtualContextEngine, captured: dict) -> None:
    def compact(segments, **_kwargs):
        captured["segments"] = list(segments)
        results = []
        for segment in segments:
            results.append(CompactionResult(
                segment_id=segment.id,
                primary_tag=segment.primary_tag,
                tags=list(segment.tags),
                summary="Fresh summary of the shared topic.",
                summary_tokens=7,
                full_text=" ".join(message.content for message in segment.messages),
                original_tokens=segment.token_count,
                messages=[{
                    "role": message.role,
                    "content": message.content,
                    "metadata": message.metadata,
                } for message in segment.messages],
                metadata=SegmentMetadata(turn_count=segment.turn_count),
                compression_ratio=0.5,
                timestamp=segment.start_timestamp,
                facts=[],
            ))
        return results

    compactor = MagicMock()
    compactor.compact.side_effect = compact
    compactor.model_name = "test-model"
    engine._compaction._compactor = compactor


def test_source_identity_keys_require_durable_actor_identity():
    rows = {
        "actor-a": CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id="actor-a",
            user_content="one",
            sender="Display One",
            sender_actor_id=NEW_ACTOR,
            audience_conversation_id="audience:guild:1",
            audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
            origin_channel_id="channel:shared",
        ),
        "actor-b": CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id="actor-b",
            user_content="two",
            sender="Changed Display",
            sender_actor_id=NEW_ACTOR,
            audience_conversation_id="audience:guild:1",
            audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
            origin_channel_id="channel:shared",
        ),
        "fallback-a": CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id="fallback-a",
            user_content="three",
            sender=" BigTex ",
        ),
        "fallback-b": CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id="fallback-b",
            user_content="four",
            sender="BIGTEX",
        ),
        "assistant": CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id="assistant",
            assistant_content="assistant only",
        ),
        "unknown": CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id="unknown",
            user_content="unattributed human",
        ),
    }
    assert CompactionPipeline._source_human_identity_keys(
        ["actor-a", "actor-b"], rows,
    ) == {(
        "actor_scope", NEW_ACTOR, "audience:guild:1", "channel:shared",
    )}
    assert CompactionPipeline._source_human_identity_keys(
        ["fallback-a", "fallback-b"], rows,
    ) is None
    assert CompactionPipeline._source_human_identity_keys(
        ["assistant"], rows,
    ) == set()
    assert CompactionPipeline._source_human_identity_keys(
        ["unknown"], rows,
    ) is None


def test_source_identity_keys_count_reply_subject_as_a_human():
    rows = {
        "reply": CanonicalTurnRow(
            conversation_id="c",
            canonical_turn_id="reply",
            user_content="What do you think?",
            sender="New Speaker",
            sender_actor_id=NEW_ACTOR,
            reply_target_body="I stopped the medication.",
            reply_subject_actor_id=OTHER_ACTOR,
            reply_subject_label="Other Speaker",
            audience_conversation_id="audience:guild:1",
            audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
            origin_channel_id="channel:shared",
        ),
    }
    assert CompactionPipeline._source_human_identity_keys(
        ["reply"], rows,
    ) == {
        ("actor_scope", NEW_ACTOR, "audience:guild:1", "channel:shared"),
        ("actor_scope", OTHER_ACTOR, "audience:guild:1", "channel:shared"),
    }


def test_dm_reply_target_does_not_wildcard_match_a_channel_row():
    target = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="target",
        user_content="Quoted claim.",
        source_message_id="message-1",
        sender_actor_id=OTHER_ACTOR,
        audience_conversation_id="audience:shared",
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        origin_channel_id="channel:guild",
    )
    reply = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="reply",
        user_content="What do you think?",
        sender_actor_id=NEW_ACTOR,
        reply_target_message_id="message-1",
        reply_target_body="Quoted claim.",
        reply_subject_actor_id=OTHER_ACTOR,
        reply_subject_label="Other Speaker",
        reply_attribution_version=1,
        audience_conversation_id="audience:shared",
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        origin_channel_id="",
    )
    segment = TaggedSegment(messages=[Message(
        role="user",
        content="What do you think?",
        metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["reply"]},
    )])
    pipeline = object.__new__(CompactionPipeline)

    roster = pipeline._build_actor_roster(
        segment,
        {"target": target, "reply": reply},
    )

    subject_lanes = [
        lane for lane in roster.lanes if lane.role == AUTHOR_ROLE_SUBJECT
    ]
    assert len(subject_lanes) == 1
    assert subject_lanes[0].actor_id == OTHER_ACTOR
    assert subject_lanes[0].speaker_label == "Other Speaker"

@pytest.mark.parametrize(
    "candidate_mode",
    [
        "different-speaker",
        "different-audience",
        "different-channel",
        "dm-vs-channel",
        "incomplete-flag",
        "unresolved-source",
        "mixed",
    ],
)
def test_pipeline_rejects_unproven_or_mismatched_merge_candidates(
    tmp_path,
    candidate_mode,
):
    engine = _engine(tmp_path)
    _seed_row(
        engine,
        canonical_id="ct-new",
        actor_id=NEW_ACTOR,
        sender="New Speaker",
        group=10,
    )

    candidate_ids = ["ct-old"]
    candidate_complete = True
    old_actor = NEW_ACTOR
    old_audience = "audience:guild:1"
    old_channel = "channel:shared"
    if candidate_mode == "different-speaker":
        old_actor = OTHER_ACTOR
    elif candidate_mode == "different-audience":
        old_audience = "audience:guild:2"
    elif candidate_mode == "different-channel":
        old_channel = "channel:other"
    elif candidate_mode == "dm-vs-channel":
        old_channel = ""
    elif candidate_mode == "incomplete-flag":
        candidate_complete = False
    elif candidate_mode == "unresolved-source":
        candidate_ids = ["ct-missing"]
    elif candidate_mode == "mixed":
        candidate_ids = ["ct-old", "ct-other"]

    _seed_row(
        engine,
        canonical_id="ct-old",
        actor_id=old_actor,
        sender="Old Speaker",
        group=0,
        audience=old_audience,
        channel=old_channel,
    )
    if candidate_mode == "mixed":
        _seed_row(
            engine,
            canonical_id="ct-other",
            actor_id=OTHER_ACTOR,
            sender="Other Speaker",
            group=1,
        )
    _stored_candidate(
        engine,
        canonical_ids=candidate_ids,
        mapping_complete=candidate_complete,
    )
    segment = _new_segment()
    captured = {}
    _install_compactor(engine, captured)

    with patch(
        "virtual_context.core.tag_scoring.compute_relatedness",
        return_value=1.0,
    ) as relatedness:
        results = engine._compaction._compact_and_store(
            [segment], 2, compact_rows=[],
        )

    assert not relatedness.called
    assert segment.merge_ref == ""
    assert results[0].segment_id == "seg-new"
    assert [
        message.content for message in captured["segments"][0].messages
    ] == ["new shared topic question", "new shared topic answer"]
    assert engine._store.get_segment(
        "stored-old", conversation_id=engine.config.conversation_id,
    ).summary == "Earlier discussion of the shared topic."


@pytest.mark.parametrize("channel", ["channel:shared", ""], ids=["guild", "dm"])
def test_pipeline_merges_exact_single_actor_match(tmp_path, channel):
    engine = _engine(tmp_path)
    _seed_row(
        engine,
        canonical_id="ct-old",
        actor_id=NEW_ACTOR,
        sender="Old Display",
        group=0,
        channel=channel,
    )
    _seed_row(
        engine,
        canonical_id="ct-new",
        actor_id=NEW_ACTOR,
        sender="New Display",
        group=10,
        channel=channel,
    )
    _stored_candidate(
        engine,
        canonical_ids=["ct-old"],
        mapping_complete=True,
    )
    segment = _new_segment(channel=channel)
    captured = {}
    _install_compactor(engine, captured)

    with patch(
        "virtual_context.core.tag_scoring.compute_relatedness",
        return_value=1.0,
    ) as relatedness:
        results = engine._compaction._compact_and_store(
            [segment], 2, compact_rows=[],
        )

    assert relatedness.call_count == 1
    assert segment.merge_ref == "stored-old"
    assert results[0].segment_id == "stored-old"
    assert results[0].metadata.canonical_turn_ids == ["ct-old", "ct-new"]
    assert results[0].metadata.source_mapping_complete is True
    assert [
        message.content for message in captured["segments"][0].messages
    ] == [
        "shared topic question ct-old",
        "shared topic answer ct-old",
        "new shared topic question",
        "new shared topic answer",
    ]
