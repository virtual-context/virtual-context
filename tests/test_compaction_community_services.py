"""Exact-source and correction contracts for extracted community services."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from virtual_context.core.compaction_pipeline import CompactionPipeline
from virtual_context.core.community.actor_card_evidence import ActorCardEvidenceService
from virtual_context.core.community.canonical_sources import (
    physical_rows_by_id,
    reply_target_rows,
)
from virtual_context.core.community.evidence_manifest import evidence_digest
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    AUTHOR_ROLE_SUBJECT,
    CanonicalTurnRow,
    Message,
    SegmentMetadata,
    SOURCE_CANONICAL_TURN_IDS_KEY,
    StoredSegment,
    TaggedSegment,
)


def test_source_hydration_batches_literal_keys_and_rejects_foreign_rows():
    calls = []

    def lookup(keys, *, internal_validation):
        assert internal_validation is True
        calls.append(keys)
        result = {
            key: CanonicalTurnRow(conversation_id=key[0], canonical_turn_id=key[1]) for key in keys
        }
        result[("foreign", "unexpected")] = CanonicalTurnRow(
            conversation_id="foreign",
            canonical_turn_id="unexpected",
        )
        result[keys[0]] = CanonicalTurnRow(conversation_id="foreign", canonical_turn_id=keys[0][1])
        return result

    store = SimpleNamespace(get_canonical_turn_rows_by_id=lookup)
    rows = physical_rows_by_id(store, (("owner", str(i)) for i in range(513)))
    assert [len(call) for call in calls] == [256, 256, 1]
    assert len(rows) == 510
    assert all(owner == "owner" for owner, _id in rows)


def test_compaction_loads_only_selected_physical_siblings():
    logical = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="u",
        turn_group_number=0,
        turn_number=0,
        user_content="request",
        assistant_content="reply",
    )
    user = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="u",
        turn_group_number=0,
        sort_key=1,
        user_content="request",
    )
    assistant = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="a",
        turn_group_number=0,
        sort_key=2,
        assistant_content="reply",
    )
    calls = []

    def groups(owner, ids, *, internal_validation):
        calls.append((owner, ids, internal_validation))
        return [user, assistant]

    pipeline = object.__new__(CompactionPipeline)
    pipeline._config = SimpleNamespace(
        conversation_id="c", monitor=SimpleNamespace(protected_recent_turns=0)
    )
    pipeline._store = SimpleNamespace(
        get_uncompacted_canonical_turns=lambda *_a, **_kw: [logical],
        get_canonical_turn_rows_by_group=groups,
    )
    rows, messages = pipeline._load_compactable_rows()
    assert rows == [logical]
    assert [message.metadata[SOURCE_CANONICAL_TURN_IDS_KEY] for message in messages] == [
        ["u"],
        ["a"],
    ]
    assert calls == [("c", [0], True)]


def test_incremental_manifest_matches_canonical_json_and_sees_old_corrections():
    metadata = {"version": 1, "policy": "actor\u200bproof"}
    items = [{"id": "old", "content": "Zoë: planned"}, {"id": "new", "content": "continue"}]
    expected = hashlib.sha256(
        json.dumps(
            {**metadata, "turns": items},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert evidence_digest(metadata, records={"turns": iter(items)}) == expected
    items[0]["content"] = "Zoë: canceled"
    assert evidence_digest(metadata, records={"turns": iter(items)}) != expected


def test_evidence_manifest_reads_exact_old_sources_and_corrected_agent_reply(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "evidence.db"))
    store.save_canonical_turn(
        "c",
        0,
        "Please use bullets",
        "",
        canonical_turn_id="old-user",
        turn_group_number=0,
        sort_key=1,
        sender_actor_id="actor:discord:one",
        audience_conversation_id="guild",
        audience_attribution_version=1,
    )
    store.save_canonical_turn(
        "c",
        0,
        "",
        "I will use bullets",
        canonical_turn_id="old-reply",
        turn_group_number=0,
        sort_key=2,
    )
    store.save_canonical_turn(
        "c",
        99,
        "unrelated private history",
        "irrelevant",
        canonical_turn_id="unrelated",
        turn_group_number=99,
        sort_key=99,
    )
    store.store_segment(
        StoredSegment(
            ref="old-segment",
            conversation_id="c",
            metadata=SegmentMetadata(canonical_turn_ids=["old-user"]),
        )
    )
    user = store.get_canonical_turn_rows_by_id([("c", "old-user")], internal_validation=True)[
        ("c", "old-user")
    ]
    turns = [SimpleNamespace(turn=user)]
    facts = [
        SimpleNamespace(owner_conversation_id="c", fact=SimpleNamespace(segment_ref="old-segment"))
    ]
    service = ActorCardEvidenceService(
        store=store, paired_agent_replies=lambda sources: service.paired_agent_replies(sources)
    )
    store.get_all_canonical_turns = lambda *_a, **_kw: (_ for _ in ()).throw(
        AssertionError("unbounded history")
    )
    records = list(service.fingerprint_records(facts, turns))
    assert "unrelated private history" not in json.dumps(records)
    before = evidence_digest({}, records={"evidence": iter(records)})
    store._get_conn().execute(
        "UPDATE canonical_turns SET assistant_content='I refuse that change' WHERE canonical_turn_id='old-reply'"
    )
    store._get_conn().commit()
    after_reply = evidence_digest(
        {}, records={"evidence": service.fingerprint_records(facts, turns)}
    )
    assert after_reply != before
    store._get_conn().execute(
        "UPDATE canonical_turns SET user_content='Actually, use prose' WHERE canonical_turn_id='old-user'"
    )
    store._get_conn().commit()
    assert (
        evidence_digest({}, records={"evidence": service.fingerprint_records(facts, turns)})
        != after_reply
    )
    store.close()


def test_historical_reply_lookup_preserves_target_ambiguity(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "replies.db"))
    for i in range(2):
        store.save_canonical_turn(
            "c",
            i,
            "target",
            "",
            canonical_turn_id=f"target-{i}",
            turn_group_number=i,
            sort_key=i,
            source_message_id="same-message",
            audience_conversation_id="guild",
            origin_channel_id="channel",
            sender_actor_id="actor:discord:one",
        )
    reply = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="reply",
        user_content="thoughts?",
        reply_target_body="copied quote",
        reply_target_message_id="same-message",
        reply_subject_actor_id="actor:discord:one",
        audience_conversation_id="guild",
        origin_channel_id="channel",
    )
    found = reply_target_rows(store, "c", [reply])
    assert set(found) == {"target-0", "target-1"}
    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(conversation_id="c", tenant_id="t")
    pipeline._quote_is_agent_output = lambda **_kwargs: pipeline.QUOTE_NOT_AGENT
    segment = TaggedSegment(
        messages=[
            Message(
                role="user",
                content="thoughts?",
                metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["reply"]},
            )
        ]
    )
    assert any(
        lane.role == AUTHOR_ROLE_SUBJECT
        for lane in pipeline._build_actor_roster(
            segment,
            {"reply": reply, **found},
            {},
        ).lanes
    )
    found.pop("target-1")
    assert not any(
        lane.role == AUTHOR_ROLE_SUBJECT
        for lane in pipeline._build_actor_roster(
            segment,
            {"reply": reply, **found},
            {},
        ).lanes
    )
    store.close()
