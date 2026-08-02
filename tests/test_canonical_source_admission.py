from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.proxy.formats import detect_format
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    SegmentMetadata,
    StorageConfig,
    StoredSegment,
    TagGeneratorConfig,
    build_user_turn_metadata,
)


GUILD_ID = "1524917037191925871"
CHANNEL_A = "1524946242499514418"
CHANNEL_B = "1524974537458974851"
ACTOR_A = "387316537012518913"
ACTOR_B = "940968368398270464"


def _reconciler(store: SQLiteStore) -> IngestReconciler:
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "source-admission.db")
    store.activate_conversation("c")
    store.upsert_conversation(tenant_id="tenant", conversation_id="c")
    return store


def _claim(
    message_id: str,
    body: str,
    *,
    channel_id: str = CHANNEL_A,
    author_id: str = ACTOR_A,
    reply_target_message_id: str = "",
) -> dict:
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return {
        "version": 1,
        "agent_scope_id": "vast",
        "platform": "discord",
        "account_id": "vast",
        "message_id": message_id,
        "channel_id": channel_id,
        "guild_id": GUILD_ID,
        "author_id": author_id,
        "transport_body_sha256": digest,
        "canonical_body_sha256": digest,
        "projection_version": "openclaw-current-user-v1",
        "reply_target_message_id": reply_target_message_id,
    }


def _edge(
    message_id: str,
    *,
    audience: str,
    reply_target_message_id: str = "",
) -> dict:
    return {
        "source_message_id": message_id,
        "reply_target_message_id": reply_target_message_id,
        "reply_subject_actor_id": "",
        "reply_subject_label": "",
        "reply_target_body": "",
        "reply_attribution_version": 1 if reply_target_message_id else 0,
        "audience_conversation_id": audience,
        "audience_attribution_version": 1,
    }


def _ingest_pair(
    rec: IngestReconciler,
    *,
    message_id: str,
    body: str,
    answer: str,
    channel_id: str = CHANNEL_A,
    author_id: str = ACTOR_A,
    audience: str = "aud-a",
):
    return rec.ingest_single(
        "c",
        user_content=body,
        assistant_content=answer,
        user_origin_channel_id=channel_id,
        user_sender_actor_id=f"actor:discord:{author_id}",
        user_reply_edge=_edge(message_id, audience=audience),
        user_source_claim=_claim(
            message_id,
            body,
            channel_id=channel_id,
            author_id=author_id,
        ),
        expected_conversation_generation=(
            rec._store.get_conversation_generation("c")
        ),
        expected_lifecycle_epoch=1,
    )


def _prepare_attested(
    rec: IngestReconciler,
    *,
    message_id: str,
    body: str,
    channel_id: str = CHANNEL_A,
    author_id: str = ACTOR_A,
    audience: str = "aud-a",
):
    metadata = build_user_turn_metadata(
        sender_name="actor",
        sender_actor_id=f"actor:discord:{author_id}",
        source_message_id=message_id,
        origin_channel_id=channel_id,
        source_conversation_key=f"agent:vast:discord:guild:{GUILD_ID}",
        source_attestation=_claim(
            message_id,
            body,
            channel_id=channel_id,
            author_id=author_id,
        ),
    )
    payload = {"messages": [{"role": "user", "content": body}]}
    return rec.ingest_batch(
        "c",
        body=payload,
        fmt=detect_format(payload),
        expected_lifecycle_epoch=1,
        source_conversation_key=f"agent:vast:discord:guild:{GUILD_ID}",
        source_audience_conversation_id=audience,
        current_user_metadata=metadata,
    )


def _seed_legacy_current_user(store: SQLiteStore, row) -> None:
    """Mirror the exact user-only row an older prepare implementation wrote."""
    store.save_canonical_turn(
        row.conversation_id,
        0,
        row.user_content,
        "",
        user_raw_content=row.user_raw_content,
        primary_tag=row.primary_tag,
        tags=list(row.tags or []),
        session_date=row.session_date,
        sender=row.sender,
        canonical_turn_id="legacy-current-user",
        sort_key=1000.0,
        turn_hash=row.turn_hash,
        hash_version=row.hash_version,
        normalized_user_text=row.normalized_user_text,
        normalized_assistant_text=row.normalized_assistant_text,
        turn_group_number=0,
        origin_channel_id=row.origin_channel_id,
        origin_channel_label=row.origin_channel_label,
        sender_actor_id=row.sender_actor_id,
        source_message_id=row.source_message_id,
        reply_target_message_id=row.reply_target_message_id,
        reply_subject_actor_id=row.reply_subject_actor_id,
        reply_subject_label=row.reply_subject_label,
        reply_target_body=row.reply_target_body,
        reply_attribution_version=row.reply_attribution_version,
        audience_conversation_id=row.audience_conversation_id,
        audience_attribution_version=row.audience_attribution_version,
    )


def test_identical_text_from_two_source_ids_stays_two_speakers(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)

    _ingest_pair(
        rec,
        message_id="1533000000000000001",
        body="same words",
        answer="first",
        author_id=ACTOR_A,
    )
    _ingest_pair(
        rec,
        message_id="1533000000000000002",
        body="same words",
        answer="second",
        author_id=ACTOR_B,
    )

    users = [row for row in store.get_all_canonical_turns("c") if row.user_content]
    assert [(row.source_message_id, row.sender_actor_id) for row in users] == [
        ("1533000000000000001", f"actor:discord:{ACTOR_A}"),
        ("1533000000000000002", f"actor:discord:{ACTOR_B}"),
    ]
    memberships = store._get_conn().execute(
        "SELECT message_id, canonical_turn_id, assistant_canonical_turn_id, "
        "assistant_turn_hash, turn_group_number, pair_version "
        "FROM canonical_message_sources "
        "ORDER BY message_id"
    ).fetchall()
    assert len(memberships) == 2
    for membership in memberships:
        assert membership["assistant_canonical_turn_id"]
        assert len(membership["assistant_turn_hash"]) == 64
        # Deprecated compatibility column: group/owner are mutable placement
        # and are validated dynamically from the two canonical rows.
        assert membership["turn_group_number"] == -1
        assert membership["pair_version"] == 1
    assert memberships[0]["canonical_turn_id"] != memberships[1]["canonical_turn_id"]


def test_delayed_exact_completion_cannot_cross_delete_recreate_generation(
    tmp_path: Path,
):
    store = _store(tmp_path)
    rec = _reconciler(store)
    prepared_generation = store.get_conversation_generation("c")
    prepared_epoch = store.get_lifecycle_epoch("c")

    deletion_generation = store.begin_conversation_deletion("c")
    successor_generation = store.activate_conversation(
        "c",
        recreate_deleted=True,
    )
    assert successor_generation > deletion_generation > prepared_generation
    # The progress-row epoch did not advance, proving why it is only a
    # secondary fence and cannot authorize a delayed provider completion.
    assert store.get_lifecycle_epoch("c") == prepared_epoch

    body = "answer from the deleted generation must not cross over"
    with pytest.raises(CanonicalSourceConflict):
        rec.ingest_single(
            "c",
            user_content=body,
            assistant_content="late answer",
            user_origin_channel_id=CHANNEL_A,
            user_sender_actor_id=f"actor:discord:{ACTOR_A}",
            user_reply_edge=_edge(
                "1533000000000000998",
                audience="aud-a",
            ),
            user_source_claim=_claim(
                "1533000000000000998",
                body,
            ),
            expected_conversation_generation=prepared_generation,
            expected_lifecycle_epoch=prepared_epoch,
        )
    assert store.get_all_canonical_turns("c") == []


def test_source_hash_contract_is_exact_utf8_not_trimmed(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    body = "  actor: café\n"
    expected = "a245d01543a433183954baac6b012a6dc5669f08eba1e1f3adfb8daa951d9e69"
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == expected

    _ingest_pair(
        rec,
        message_id="1533000000000000003",
        body=body,
        answer="exact bytes admitted",
    )
    membership = store._get_conn().execute(
        "SELECT canonical_body_sha256 FROM canonical_message_sources "
        "WHERE message_id = ?",
        ("1533000000000000003",),
    ).fetchone()
    assert membership["canonical_body_sha256"] == expected

    bad_claim = _claim("1533000000000000004", body)
    trimmed = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
    bad_claim["canonical_body_sha256"] = trimmed
    with pytest.raises(CanonicalSourceConflict):
        rec.ingest_single(
            "c",
            user_content=body,
            assistant_content="must not admit a trimmed digest",
            user_origin_channel_id=CHANNEL_A,
            user_sender_actor_id=f"actor:discord:{ACTOR_A}",
            user_reply_edge=_edge(
                "1533000000000000004",
                audience="aud-a",
            ),
            user_source_claim=bad_claim,
            expected_conversation_generation=(
                store.get_conversation_generation("c")
            ),
            expected_lifecycle_epoch=1,
        )


def test_same_source_replay_is_idempotent_but_conflict_aborts(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    message_id = "1533000000000000010"

    first = _ingest_pair(
        rec, message_id=message_id, body="stable body", answer="stable answer",
    )
    replay = _ingest_pair(
        rec, message_id=message_id, body="stable body", answer="stable answer",
    )
    assert first.turns_written == 2
    assert replay.turns_written == 0
    assert len(store.get_all_canonical_turns("c")) == 2

    with pytest.raises(CanonicalSourceConflict):
        _ingest_pair(
            rec, message_id=message_id, body="changed body", answer="stable answer",
        )
    with pytest.raises(CanonicalSourceConflict):
        _ingest_pair(
            rec,
            message_id=message_id,
            body="stable body",
            answer="stable answer",
            author_id=ACTOR_B,
        )
    with pytest.raises(CanonicalSourceConflict):
        _ingest_pair(
            rec,
            message_id=message_id,
            body="stable body",
            answer="stable answer",
            channel_id=CHANNEL_B,
            audience="aud-b",
        )
    assert len(store.get_all_canonical_turns("c")) == 2


def test_source_ledger_cannot_admit_an_unpaired_user_row(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    _ingest_pair(
        rec,
        message_id="1533000000000000011",
        body="pair me atomically",
        answer="paired answer",
    )
    stored_user = next(
        row for row in store.get_all_canonical_turns("c") if row.user_content
    )
    user = replace(
        stored_user,
        source_claim=_claim(
            "1533000000000000011",
            "pair me atomically",
        ),
    )

    with pytest.raises(
        CanonicalSourceConflict,
        match="requires an immutable exact assistant pair",
    ):
        store.attest_canonical_user_source(
            user,
            observed_at=datetime.now(timezone.utc).isoformat(),
            assistant_row=None,
        )

    membership = store._get_conn().execute(
        "SELECT pair_version, assistant_canonical_turn_id, assistant_turn_hash "
        "FROM canonical_message_sources WHERE message_id = ?",
        ("1533000000000000011",),
    ).fetchone()
    assert membership["pair_version"] == 1
    assert membership["assistant_canonical_turn_id"]
    assert membership["assistant_turn_hash"]


def test_interleaved_channel_completion_anchors_exact_user_source(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    message_a = "1533000000000000020"
    message_b = "1533000000000000021"
    body_a = "question from channel a"

    metadata_a = build_user_turn_metadata(
        sender_name="A",
        sender_actor_id=f"actor:discord:{ACTOR_A}",
        source_message_id=message_a,
        origin_channel_id=CHANNEL_A,
        origin_channel_label="#a",
        source_conversation_key=f"agent:vast:discord:guild:{GUILD_ID}",
        source_attestation=_claim(message_a, body_a),
    )
    prepare_body = {"messages": [{"role": "user", "content": body_a}]}
    rec.ingest_batch(
        "c",
        body=prepare_body,
        fmt=detect_format(prepare_body),
        expected_lifecycle_epoch=1,
        source_conversation_key=f"agent:vast:discord:channel:{CHANNEL_A}",
        source_audience_conversation_id="aud-a",
        current_user_metadata=metadata_a,
    )

    _ingest_pair(
        rec,
        message_id=message_b,
        body="question from channel b",
        answer="answer b",
        channel_id=CHANNEL_B,
        author_id=ACTOR_B,
        audience="aud-b",
    )
    _ingest_pair(
        rec,
        message_id=message_a,
        body=body_a,
        answer="answer a",
        channel_id=CHANNEL_A,
        author_id=ACTOR_A,
        audience="aud-a",
    )

    rows = store.get_all_canonical_turns("c")
    assert [row.user_content or row.assistant_content for row in rows] == [
        "question from channel b",
        "answer b",
        body_a,
        "answer a",
    ]
    assert rows[0].turn_group_number == rows[1].turn_group_number
    assert rows[2].turn_group_number == rows[3].turn_group_number
    assert rows[0].turn_group_number != rows[2].turn_group_number


def test_late_lower_snowflake_never_resequences_derived_group(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)

    # B completes first even though A has the lower source snowflake.
    _ingest_pair(
        rec,
        message_id="1533000000000000022",
        body="completed first",
        answer="answer b",
        channel_id=CHANNEL_B,
        author_id=ACTOR_B,
        audience="aud-b",
    )
    before = store.get_all_canonical_turns("c")
    assert {row.turn_group_number for row in before} == {0}
    before_ids = [row.canonical_turn_id for row in before]
    tagged_at = "2026-08-02T04:00:00+00:00"
    assert store.mark_canonical_turns_tagged(
        "c", before_ids, tagged_at=tagged_at,
    ) == 2
    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET compacted_at = ? "
        "WHERE canonical_turn_id IN (?, ?)",
        (tagged_at, *before_ids),
    )
    conn.commit()
    now = datetime.now(timezone.utc)
    store.store_segment(StoredSegment(
        ref="seg-b",
        conversation_id="c",
        primary_tag="stable",
        tags=["stable"],
        summary="derived from B",
        full_text="completed first\nanswer b",
        metadata=SegmentMetadata(
            canonical_turn_ids=before_ids,
            start_turn_number=0,
            end_turn_number=0,
            turn_count=1,
            source_mapping_complete=True,
        ),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    ))

    _ingest_pair(
        rec,
        message_id="1533000000000000021",
        body="lower snowflake completed later",
        answer="answer a",
        channel_id=CHANNEL_A,
        author_id=ACTOR_A,
        audience="aud-a",
    )

    after = store.get_all_canonical_turns("c")
    original = [row for row in after if row.canonical_turn_id in before_ids]
    assert [row.turn_group_number for row in original] == [0, 0]
    assert [row.tagged_at for row in original] == [tagged_at, tagged_at]
    assert [row.compacted_at for row in original] == [tagged_at, tagged_at]
    assert [row.canonical_turn_id for row in after[:2]] == before_ids
    assert {row.turn_group_number for row in after[2:]} == {1}
    derived = store.get_segment("seg-b", conversation_id="c")
    assert derived is not None
    assert derived.metadata.canonical_turn_ids == before_ids
    assert (
        derived.metadata.start_turn_number,
        derived.metadata.end_turn_number,
    ) == (0, 0)


def test_attested_discord_prepare_is_validated_but_canonical_read_only(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    message_id = "1533000000000000030"
    current = "optics: current native turn"
    metadata = build_user_turn_metadata(
        sender_name="optics",
        sender_actor_id=f"actor:discord:{ACTOR_A}",
        source_message_id=message_id,
        origin_channel_id=CHANNEL_A,
        source_conversation_key=f"agent:vast:discord:guild:{GUILD_ID}",
        source_attestation=_claim(message_id, current),
    )
    body = {
        "messages": [
            {"role": "user", "content": "untrusted historical user"},
            {"role": "assistant", "content": "untrusted historical answer"},
            {"role": "user", "content": current},
        ]
    }
    result = rec.ingest_batch(
        "c",
        body=body,
        fmt=detect_format(body),
        expected_lifecycle_epoch=1,
        source_conversation_key=f"agent:vast:discord:guild:{GUILD_ID}",
        source_audience_conversation_id="aud-a",
        current_user_metadata=metadata,
    )
    rows = store.get_all_canonical_turns("c")
    assert result.merge_mode == "source_exact_prepare_read_only"
    assert result.turns_written == 0
    assert rows == []


def test_attested_current_user_appends_after_existing_history_and_completes(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    _ingest_pair(
        rec,
        message_id="1533000000000000040",
        body="existing question",
        answer="existing answer",
    )

    message_id = "1533000000000000041"
    prepared = _prepare_attested(
        rec,
        message_id=message_id,
        body="new current question",
    )
    assert prepared.merge_mode == "source_exact_prepare_read_only"
    assert prepared.turns_written == 0
    completed = _ingest_pair(
        rec,
        message_id=message_id,
        body="new current question",
        answer="new current answer",
    )
    assert completed.merge_mode == "source_exact_pair_append"
    rows = store.get_all_canonical_turns("c")
    assert [row.user_content or row.assistant_content for row in rows] == [
        "existing question",
        "existing answer",
        "new current question",
        "new current answer",
    ]


def test_same_text_new_source_id_is_distinct_current_user(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    first_id = "1533000000000000050"
    second_id = "1533000000000000051"
    _ingest_pair(
        rec,
        message_id=first_id,
        body="repeatable words",
        answer="first answer",
    )

    prepared = _prepare_attested(
        rec,
        message_id=second_id,
        body="repeatable words",
    )
    assert prepared.turns_written == 0
    _ingest_pair(
        rec,
        message_id=second_id,
        body="repeatable words",
        answer="second answer",
    )
    users = [row for row in store.get_all_canonical_turns("c") if row.user_content]
    assert [row.source_message_id for row in users] == [first_id, second_id]
    assert users[0].canonical_turn_id != users[1].canonical_turn_id


def test_abandoned_prepare_cannot_block_later_complete_group(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)

    abandoned = _prepare_attested(
        rec,
        message_id="1533000000000000060",
        body="provider never returns for this request",
    )
    assert abandoned.merge_mode == "source_exact_prepare_read_only"
    assert store.get_all_canonical_turns("c") == []

    completed = _ingest_pair(
        rec,
        message_id="1533000000000000061",
        body="later complete request",
        answer="later complete answer",
        author_id=ACTOR_B,
    )
    assert completed.merge_mode == "source_exact_pair_append"
    rows = store.get_all_canonical_turns("c")
    assert [row.user_content or row.assistant_content for row in rows] == [
        "later complete request",
        "later complete answer",
    ]
    groups = store.iter_complete_untagged_canonical_groups(
        conversation_id="c",
        expected_lifecycle_epoch=1,
        batch_size=10,
    )
    assert len(groups) == 1
    assert [row.canonical_turn_id for row in groups[0]] == [
        row.canonical_turn_id for row in rows
    ]


def test_exact_legacy_user_only_row_is_atomically_adopted(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    message_id = "1533000000000000064"
    prepared = _prepare_attested(
        rec,
        message_id=message_id,
        body="rolling deployment current turn",
    )
    _seed_legacy_current_user(store, prepared.rows[0])
    before = store.get_all_canonical_turns("c")[0]

    result = _ingest_pair(
        rec,
        message_id=message_id,
        body="rolling deployment current turn",
        answer="exact completion",
    )

    assert result.merge_mode == "source_exact_legacy_user_adopt"
    rows = store.get_all_canonical_turns("c")
    assert len(rows) == 2
    assert rows[0].canonical_turn_id == before.canonical_turn_id
    assert rows[0].user_content == before.user_content
    assert rows[0].sender_actor_id == before.sender_actor_id
    assert rows[0].turn_group_number == before.turn_group_number
    assert rows[1].assistant_content == "exact completion"
    assert rows[1].turn_group_number == before.turn_group_number
    membership = store._get_conn().execute(
        "SELECT canonical_turn_id, assistant_canonical_turn_id "
        "FROM canonical_message_sources WHERE message_id = ?",
        (message_id,),
    ).fetchone()
    assert membership["canonical_turn_id"] == before.canonical_turn_id
    assert membership["assistant_canonical_turn_id"] == rows[1].canonical_turn_id


def test_legacy_source_id_with_different_identity_fails_without_writes(
    tmp_path: Path,
):
    store = _store(tmp_path)
    rec = _reconciler(store)
    message_id = "1533000000000000065"
    prepared = _prepare_attested(
        rec,
        message_id=message_id,
        body="identity must remain exact",
    )
    wrong = replace(
        prepared.rows[0],
        sender_actor_id=f"actor:discord:{ACTOR_B}",
    )
    _seed_legacy_current_user(store, wrong)
    before = store.get_all_canonical_turns("c")

    with pytest.raises(CanonicalSourceConflict, match="body, role, or identity"):
        _ingest_pair(
            rec,
            message_id=message_id,
            body="identity must remain exact",
            answer="must not persist",
        )

    assert store.get_all_canonical_turns("c") == before
    assert store._get_conn().execute(
        "SELECT COUNT(*) AS n FROM canonical_message_sources"
    ).fetchone()["n"] == 0


def test_attested_pair_second_write_failure_rolls_back_user_and_membership(
    tmp_path: Path,
    monkeypatch,
):
    store = _store(tmp_path)
    rec = _reconciler(store)
    original = store.save_canonical_turn
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected assistant write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "save_canonical_turn", fail_second)
    with pytest.raises(RuntimeError, match="injected assistant write failure"):
        _ingest_pair(
            rec,
            message_id="1533000000000000062",
            body="must be atomic",
            answer="must roll back",
        )
    assert store.get_all_canonical_turns("c") == []
    count = store._get_conn().execute(
        "SELECT COUNT(*) AS n FROM canonical_message_sources"
    ).fetchone()["n"]
    assert count == 0


def test_stale_exact_completion_cannot_cross_lifecycle_epoch(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    prepared = _prepare_attested(
        rec,
        message_id="1533000000000000063",
        body="prepared before lifecycle reset",
    )
    assert prepared.turns_written == 0
    conn = store._get_conn()
    conn.execute(
        "UPDATE conversations SET lifecycle_epoch = 2 "
        "WHERE conversation_id = 'c'",
    )
    conn.commit()

    with pytest.raises(LifecycleEpochMismatch):
        _ingest_pair(
            rec,
            message_id="1533000000000000063",
            body="prepared before lifecycle reset",
            answer="must not cross cutover",
        )
    assert store.get_all_canonical_turns("c") == []
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM canonical_message_sources"
    ).fetchone()["n"]
    assert count == 0


def test_unattested_discord_group_window_is_context_only(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    body = {
        "messages": [
            {"role": "user", "content": "must not become legacy history"},
            {"role": "assistant", "content": "also must not persist"},
        ]
    }
    result = rec.ingest_batch(
        "c",
        body=body,
        fmt=detect_format(body),
        expected_lifecycle_epoch=1,
        source_conversation_key=f"agent:vast:discord:guild:{GUILD_ID}",
        source_audience_conversation_id="aud-a",
    )
    assert result.merge_mode == "source_attestation_required_noop"
    assert result.turns_written == 0
    assert store.get_all_canonical_turns("c") == []


def test_discord_group_dm_remains_on_legacy_non_attested_lane(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    body = {
        "messages": [
            {"role": "user", "content": "group DM request"},
            {"role": "assistant", "content": "group DM response"},
        ]
    }
    result = rec.ingest_batch(
        "c",
        body=body,
        fmt=detect_format(body),
        expected_lifecycle_epoch=1,
        source_conversation_key="sk:agent:vast:discord:group:123456789",
        source_audience_conversation_id="group-dm-audience",
    )
    assert result.turns_written == 2
    rows = store.get_all_canonical_turns("c")
    assert [row.user_content or row.assistant_content for row in rows] == [
        "group DM request",
        "group DM response",
    ]


def test_verified_source_columns_are_database_immutable(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    message_id = "1533000000000000030"
    _ingest_pair(
        rec, message_id=message_id, body="do not rewrite", answer="answer",
    )
    rows = store.get_all_canonical_turns("c")
    user = next(row for row in rows if row.user_content)
    assistant = next(row for row in rows if row.assistant_content)

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        store._get_conn().execute(
            "UPDATE canonical_turns SET user_content = ? "
            "WHERE canonical_turn_id = ?",
            ("rewritten", user.canonical_turn_id),
        )
    store._get_conn().rollback()

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        store._get_conn().execute(
            "UPDATE canonical_turns SET assistant_content = ? "
            "WHERE canonical_turn_id = ?",
            ("wrong answer", assistant.canonical_turn_id),
        )
    store._get_conn().rollback()

    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET turn_group_number = ? "
        "WHERE canonical_turn_id = ?",
        (99, assistant.canonical_turn_id),
    )
    conn.commit()
    with pytest.raises(CanonicalSourceConflict):
        _ingest_pair(
            rec, message_id=message_id,
            body="do not rewrite", answer="answer",
        )
    conn.execute(
        "UPDATE canonical_turns SET turn_group_number = ? "
        "WHERE canonical_turn_id = ?",
        (99, user.canonical_turn_id),
    )
    conn.commit()
    assert _ingest_pair(
        rec, message_id=message_id,
        body="do not rewrite", answer="answer",
    ).turns_written == 0

    unchanged = next(row for row in store.get_all_canonical_turns("c") if row.user_content)
    assert unchanged.user_content == "do not rewrite"


def test_boolean_source_version_is_not_integer_version_one(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    body = "boolean is not protocol version one"
    claim = _claim("1533000000000000031", body)
    claim["version"] = True
    with pytest.raises(CanonicalSourceConflict):
        rec.ingest_single(
            "c",
            user_content=body,
            assistant_content="must reject",
            user_origin_channel_id=CHANNEL_A,
            user_sender_actor_id=f"actor:discord:{ACTOR_A}",
            user_reply_edge=_edge(
                "1533000000000000031", audience="aud-a",
            ),
            user_source_claim=claim,
            expected_conversation_generation=(
                store.get_conversation_generation("c")
            ),
            expected_lifecycle_epoch=1,
        )


def test_sqlite_upgrade_rebuilds_missing_assistant_foreign_key(tmp_path: Path):
    db_path = tmp_path / "source-admission.db"
    store = _store(tmp_path)
    rec = _reconciler(store)
    _ingest_pair(
        rec,
        message_id="1533000000000000070",
        body="preserve this ledger row",
        answer="and its exact assistant",
    )
    conn = store._get_conn()
    assistant_id = conn.execute(
        "SELECT assistant_canonical_turn_id FROM canonical_message_sources"
    ).fetchone()[0]
    for trigger in (
        "trg_guard_attested_canonical_turn_update",
        "trg_validate_canonical_message_source_pair_insert",
        "trg_validate_canonical_message_source_pair_update",
        "trg_guard_canonical_message_source_update",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        "ALTER TABLE canonical_message_sources RENAME TO "
        "canonical_message_sources_without_assistant_fk"
    )
    conn.execute(
        """CREATE TABLE canonical_message_sources (
               tenant_id TEXT NOT NULL,
               agent_scope_id TEXT NOT NULL,
               platform TEXT NOT NULL,
               account_id TEXT NOT NULL,
               message_id TEXT NOT NULL,
               canonical_turn_id TEXT NOT NULL UNIQUE
                   REFERENCES canonical_turns(canonical_turn_id)
                   ON DELETE CASCADE,
               assistant_canonical_turn_id TEXT,
               assistant_turn_hash TEXT NOT NULL DEFAULT '',
               turn_group_number INTEGER NOT NULL DEFAULT -1,
               pair_version INTEGER NOT NULL DEFAULT 1,
               audience_conversation_id TEXT NOT NULL,
               channel_id TEXT NOT NULL,
               guild_id TEXT NOT NULL,
               author_id TEXT NOT NULL,
               source_actor_id TEXT NOT NULL,
               transport_body_sha256 TEXT NOT NULL,
               canonical_body_sha256 TEXT NOT NULL,
               projection_version TEXT NOT NULL,
               canonical_turn_hash TEXT NOT NULL,
               reply_target_message_id TEXT NOT NULL DEFAULT '',
               observed_at TEXT NOT NULL,
               created_at TEXT NOT NULL,
               PRIMARY KEY (
                   tenant_id, agent_scope_id, platform, account_id, message_id
               )
           )"""
    )
    columns = (
        "tenant_id, agent_scope_id, platform, account_id, message_id, "
        "canonical_turn_id, assistant_canonical_turn_id, assistant_turn_hash, "
        "turn_group_number, pair_version, audience_conversation_id, channel_id, "
        "guild_id, author_id, source_actor_id, transport_body_sha256, "
        "canonical_body_sha256, projection_version, canonical_turn_hash, "
        "reply_target_message_id, observed_at, created_at"
    )
    conn.execute(
        f"INSERT INTO canonical_message_sources ({columns}) "
        f"SELECT {columns} FROM canonical_message_sources_without_assistant_fk"
    )
    conn.execute("DROP TABLE canonical_message_sources_without_assistant_fk")
    conn.execute("PRAGMA foreign_keys=ON")
    store.close()

    reopened = SQLiteStore(db_path)
    upgraded = reopened._get_conn()
    assert upgraded.execute(
        "SELECT COUNT(*) AS n FROM canonical_message_sources"
    ).fetchone()["n"] == 1
    foreign_keys = {
        (row["from"], row["table"], row["to"], row["on_delete"])
        for row in upgraded.execute(
            "PRAGMA foreign_key_list(canonical_message_sources)"
        ).fetchall()
    }
    assert (
        "assistant_canonical_turn_id",
        "canonical_turns",
        "canonical_turn_id",
        "CASCADE",
    ) in foreign_keys
    upgraded.execute(
        "DELETE FROM canonical_turns WHERE canonical_turn_id = ?",
        (assistant_id,),
    )
    assert upgraded.execute(
        "SELECT COUNT(*) AS n FROM canonical_message_sources"
    ).fetchone()["n"] == 0


def test_sqlite_conversation_claim_never_reassigns_tenant(tmp_path: Path):
    store = _store(tmp_path)
    store.claim_or_verify_conversation(tenant_id="tenant", conversation_id="c")
    with pytest.raises(PermissionError, match="different tenant"):
        store.claim_or_verify_conversation(
            tenant_id="other-tenant",
            conversation_id="c",
        )
    assert store.conversation_belongs_to_tenant(
        tenant_id="tenant",
        conversation_id="c",
    )
    assert not store.conversation_belongs_to_tenant(
        tenant_id="other-tenant",
        conversation_id="c",
    )
