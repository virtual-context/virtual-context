"""Focused Postgres proof for immutable Discord source admission."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
import threading
import uuid

import psycopg
import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.postgres import PostgresStore
from virtual_context.types import StorageConfig, TagGeneratorConfig


from tests.pg_helpers import pg_dsn

PG_URL = pg_dsn()


def _is_disposable_database(url: str | None) -> bool:
    """This file is destructive; it runs ONLY against a database whose name
    declares disposability. The ``vc_disposable_`` prefix is the operator's
    explicit opt-in — the shared fleet database never carries it."""
    if not url:
        return False
    try:
        dbname = str(psycopg.conninfo.conninfo_to_dict(url).get("dbname") or "")
    except Exception:
        return False
    return dbname.startswith("vc_disposable_")


pytestmark = pytest.mark.skipif(
    not _is_disposable_database(PG_URL),
    reason=(
        "destructive source-admission tests require a vc_disposable_* "
        "database (DATABASE_URL / VC_TEST_POSTGRES_URL via pg_dsn())"
    ),
)


def _assert_disposable_database() -> None:
    """Never let this destructive integration file point at a normal DB."""
    if not _is_disposable_database(PG_URL):
        raise RuntimeError(
            "Postgres source-admission tests require a vc_disposable_* database"
        )


def _stack(conversation_id: str) -> tuple[PostgresStore, IngestReconciler]:
    _assert_disposable_database()
    store = PostgresStore(PG_URL)
    store.activate_conversation(conversation_id)
    store.upsert_conversation(
        tenant_id="source-test-tenant",
        conversation_id=conversation_id,
    )
    config = VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(backend="postgres", postgres_dsn=PG_URL),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return store, IngestReconciler(store=store, semantic=semantic)


def _exact_ingest(
    reconciler: IngestReconciler,
    conversation_id: str,
    *,
    message_id: str,
    actor_id: str,
    body: str,
    answer: str,
    expected_lifecycle_epoch: int = 1,
    channel_id: str = "1524946242499514418",
    guild_id: str = "1524917037191925871",
    audience: str = "men-guild-audience",
    reply_target_message_id: str = "",
):
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    claim = {
        "version": 1,
        "agent_scope_id": "vast",
        "platform": "discord",
        "account_id": "vast",
        "message_id": message_id,
        "channel_id": channel_id,
        "guild_id": guild_id,
        "author_id": actor_id,
        "transport_body_sha256": digest,
        "canonical_body_sha256": digest,
        "projection_version": "openclaw-current-user-v1",
        "reply_target_message_id": reply_target_message_id,
    }
    edge = {
        "source_message_id": message_id,
        "reply_target_message_id": reply_target_message_id,
        "reply_subject_actor_id": "",
        "reply_subject_label": "",
        "reply_target_body": "",
        "reply_attribution_version": 0,
        "audience_conversation_id": audience,
        "audience_attribution_version": 1,
    }
    return reconciler.ingest_single(
        conversation_id,
        user_content=body,
        assistant_content=answer,
        user_origin_channel_id=channel_id,
        user_sender_actor_id=f"actor:discord:{actor_id}",
        user_reply_edge=edge,
        user_source_claim=claim,
        expected_conversation_generation=(
            reconciler._store.get_conversation_generation(conversation_id)
        ),
        expected_lifecycle_epoch=expected_lifecycle_epoch,
    )


def test_attested_source_is_transactional_idempotent_and_database_immutable():
    conversation_id = f"source-admission-{uuid.uuid4()}"
    message_id = "1533000000000000200"
    channel_id = "1524946242499514418"
    guild_id = "1524917037191925871"
    author_id = "387316537012518913"
    body = "postgres source-bound body"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    claim = {
        "version": 1,
        "agent_scope_id": "vast",
        "platform": "discord",
        "account_id": "vast",
        "message_id": message_id,
        "channel_id": channel_id,
        "guild_id": guild_id,
        "author_id": author_id,
        "transport_body_sha256": digest,
        "canonical_body_sha256": digest,
        "projection_version": "openclaw-current-user-v1",
        "reply_target_message_id": "",
    }
    edge = {
        "source_message_id": message_id,
        "reply_target_message_id": "",
        "reply_subject_actor_id": "",
        "reply_subject_label": "",
        "reply_target_body": "",
        "reply_attribution_version": 0,
        "audience_conversation_id": "men-guild-audience",
        "audience_attribution_version": 1,
    }

    store, reconciler = _stack(conversation_id)

    def ingest(*, actor_id: str = author_id):
        source_claim = {**claim, "author_id": actor_id}
        return reconciler.ingest_single(
            conversation_id,
            user_content=body,
            assistant_content="postgres source-bound answer",
            user_origin_channel_id=channel_id,
            user_sender_actor_id=f"actor:discord:{actor_id}",
            user_reply_edge=edge,
            user_source_claim=source_claim,
            expected_conversation_generation=(
                store.get_conversation_generation(conversation_id)
            ),
            expected_lifecycle_epoch=1,
        )

    first = ingest()
    replay = ingest()
    assert first.turns_written == 2
    assert replay.turns_written == 0

    with store.pool.connection() as conn:
        membership = conn.execute(
            "SELECT * FROM canonical_message_sources WHERE message_id = %s",
            (message_id,),
        ).fetchall()
    assert len(membership) == 1
    assert membership[0]["author_id"] == author_id

    with pytest.raises(CanonicalSourceConflict):
        ingest(actor_id="940968368398270464")
    with pytest.raises(CanonicalSourceConflict):
        _exact_ingest(
            reconciler, conversation_id,
            message_id=message_id, actor_id=author_id, body=body,
            answer="postgres source-bound answer",
            guild_id="999999999999999999",
        )
    with pytest.raises(CanonicalSourceConflict):
        _exact_ingest(
            reconciler, conversation_id,
            message_id=message_id, actor_id=author_id, body=body,
            answer="postgres source-bound answer",
            reply_target_message_id="1532999999999999999",
        )

    user_row = next(
        row for row in store.get_all_canonical_turns(conversation_id)
        if row.user_content
    )
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        with store.pool.connection() as conn, conn.transaction():
            conn.execute(
                "UPDATE canonical_turns SET sender_actor_id = %s "
                "WHERE canonical_turn_id = %s",
                ("actor:discord:wrong", user_row.canonical_turn_id),
            )

    unchanged = next(
        row for row in store.get_all_canonical_turns(conversation_id)
        if row.user_content
    )
    assert unchanged.sender_actor_id == f"actor:discord:{author_id}"


def test_postgres_atomically_adopts_exact_legacy_user_only_row():
    conversation_id = f"source-legacy-adopt-{uuid.uuid4()}"
    message_id = str(
        1_533_000_000_000_000_000 + uuid.uuid4().int % 1_000_000_000_000_000
    )
    actor_id = "387316537012518913"
    channel_id = "1524946242499514418"
    audience = "men-guild-audience"
    body = "legacy rolling deploy current user"
    legacy_user_id = str(uuid.uuid4())
    store, reconciler = _stack(conversation_id)
    prepared = reconciler._prepare_message_row(
        conversation_id,
        role="user",
        content=body,
        origin_channel_id=channel_id,
        sender_actor_id=f"actor:discord:{actor_id}",
        source_message_id=message_id,
        audience_conversation_id=audience,
        audience_attribution_version=1,
    )
    store.save_canonical_turn(
        conversation_id,
        0,
        body,
        "",
        canonical_turn_id=legacy_user_id,
        sort_key=1000.0,
        turn_hash=prepared.turn_hash,
        hash_version=prepared.hash_version,
        normalized_user_text=prepared.normalized_user_text,
        normalized_assistant_text=prepared.normalized_assistant_text,
        turn_group_number=0,
        origin_channel_id=channel_id,
        sender_actor_id=f"actor:discord:{actor_id}",
        source_message_id=message_id,
        audience_conversation_id=audience,
        audience_attribution_version=1,
    )

    result = _exact_ingest(
        reconciler,
        conversation_id,
        message_id=message_id,
        actor_id=actor_id,
        body=body,
        answer="atomic exact completion",
        channel_id=channel_id,
        audience=audience,
    )

    assert result.merge_mode == "source_exact_legacy_user_adopt"
    rows = store.get_all_canonical_turns(conversation_id)
    assert len(rows) == 2
    assert rows[0].canonical_turn_id == legacy_user_id
    assert rows[0].user_content == body
    assert rows[0].sender_actor_id == f"actor:discord:{actor_id}"
    assert rows[1].assistant_content == "atomic exact completion"
    assert rows[0].turn_group_number == rows[1].turn_group_number == 0
    with store.pool.connection() as conn:
        membership = conn.execute(
            "SELECT canonical_turn_id, assistant_canonical_turn_id "
            "FROM canonical_message_sources WHERE message_id = %s",
            (message_id,),
        ).fetchone()
    assert str(membership["canonical_turn_id"]) == legacy_user_id
    assert str(membership["assistant_canonical_turn_id"]) == rows[1].canonical_turn_id


def test_postgres_conversation_claim_never_reassigns_tenant():
    conversation_id = f"source-owner-{uuid.uuid4()}"
    store, _reconciler = _stack(conversation_id)
    store.claim_or_verify_conversation(
        tenant_id="source-test-tenant",
        conversation_id=conversation_id,
    )
    with pytest.raises(PermissionError, match="different tenant"):
        store.claim_or_verify_conversation(
            tenant_id="hostile-tenant",
            conversation_id=conversation_id,
        )
    assert store.conversation_belongs_to_tenant(
        tenant_id="source-test-tenant",
        conversation_id=conversation_id,
    )
    assert not store.conversation_belongs_to_tenant(
        tenant_id="hostile-tenant",
        conversation_id=conversation_id,
    )


def test_two_connections_append_reverse_chronology_without_cross_pairing():
    conversation_id = f"source-race-{uuid.uuid4()}"
    store_a, reconciler_a = _stack(conversation_id)
    store_b, reconciler_b = _stack(conversation_id)
    b_holds_reconcile = threading.Event()
    a_attempted_reconcile = threading.Event()
    original_reconcile_a = store_a.conversation_reconcile
    original_reconcile_b = store_b.conversation_reconcile

    @contextmanager
    def observed_a_reconcile(
        candidate: str,
        *,
        expected_generation: int | None = None,
    ):
        a_attempted_reconcile.set()
        with original_reconcile_a(
            candidate,
            expected_generation=expected_generation,
        ):
            yield

    @contextmanager
    def held_b_reconcile(
        candidate: str,
        *,
        expected_generation: int | None = None,
    ):
        with original_reconcile_b(
            candidate,
            expected_generation=expected_generation,
        ):
            b_holds_reconcile.set()
            if not a_attempted_reconcile.wait(15):
                raise AssertionError("A never attempted the held reconcile lock")
            yield

    store_a.conversation_reconcile = observed_a_reconcile
    store_b.conversation_reconcile = held_b_reconcile
    older = {
        "message_id": "1533000000000000400",
        "actor_id": "387316537012518913",
        "body": "older source completed second",
        "answer": "answer bound only to older source",
    }
    newer = {
        "message_id": "1533000000000000401",
        "actor_id": "940968368398270464",
        "body": "newer source completed first",
        "answer": "answer bound only to newer source",
    }

    def complete_a():
        if not b_holds_reconcile.wait(15):
            raise AssertionError("B never acquired the reconcile lock")
        return _exact_ingest(
            reconciler_a, conversation_id, **older,
        )

    def complete_b():
        return _exact_ingest(reconciler_b, conversation_id, **newer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_b = executor.submit(complete_b)
        assert b_holds_reconcile.wait(15)
        future_a = executor.submit(complete_a)
        result_a = future_a.result(timeout=30)
        result_b = future_b.result(timeout=30)
    assert result_a.turns_written == result_b.turns_written == 2

    rows = store_a.get_all_canonical_turns(conversation_id)
    assert len(rows) == 4
    users = {row.source_message_id: row for row in rows if row.user_content}
    assert set(users) == {older["message_id"], newer["message_id"]}
    assert users[newer["message_id"]].turn_group_number == 0
    assert users[older["message_id"]].turn_group_number == 1
    original_groups = {
        source: row.turn_group_number for source, row in users.items()
    }
    for expected in (older, newer):
        user = users[expected["message_id"]]
        assert user.user_content == expected["body"]
        assert user.sender_actor_id == f"actor:discord:{expected['actor_id']}"
        siblings = [
            row for row in rows
            if row.turn_group_number == user.turn_group_number
        ]
        assert len(siblings) == 2
        assistant = next(row for row in siblings if row.assistant_content)
        assert assistant.assistant_content == expected["answer"]
        assert assistant.sender_actor_id == ""

    with store_a.pool.connection() as conn:
        memberships = conn.execute(
            """SELECT message_id, canonical_turn_id, author_id
                 FROM canonical_message_sources
                WHERE canonical_turn_id IN (
                      SELECT canonical_turn_id FROM canonical_turns
                       WHERE conversation_id = %s)
                ORDER BY message_id""",
            (conversation_id,),
        ).fetchall()
    assert len(memberships) == 2
    assert {
        (row["message_id"], row["author_id"])
        for row in memberships
    } == {
        (older["message_id"], older["actor_id"]),
        (newer["message_id"], newer["actor_id"]),
    }

    replay = _exact_ingest(reconciler_a, conversation_id, **older)
    assert replay.turns_written == 0
    replay_rows = store_a.get_all_canonical_turns(conversation_id)
    assert len(replay_rows) == 4
    assert {
        row.source_message_id: row.turn_group_number
        for row in replay_rows if row.user_content
    } == original_groups


def test_attested_pair_survives_vcmerge_and_replays_under_new_owner():
    source_id = f"source-merge-from-{uuid.uuid4()}"
    target_id = f"source-merge-into-{uuid.uuid4()}"
    source_store, source_rec = _stack(source_id)
    target_store, target_rec = _stack(target_id)
    target_payload = {
        "message_id": "1533000000000000298",
        "actor_id": "940968368398270464",
        "body": "target already has a complete pair",
        "answer": "target answer",
    }
    source_payload = {
        "message_id": "1533000000000000299",
        "actor_id": "387316537012518913",
        "body": "source pair must keep its exact identity",
        "answer": "source answer",
    }
    _exact_ingest(target_rec, target_id, **target_payload)
    _exact_ingest(source_rec, source_id, **source_payload)
    before = source_store.get_all_canonical_turns(source_id)
    source_ids = {row.canonical_turn_id for row in before}

    merge_id = str(uuid.uuid4())
    reservation = source_store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id,
        tenant_id="source-test-tenant",
        source_conversation_id=source_id,
        target_conversation_id=target_id,
        source_label_at_merge="attested source",
    )
    assert reservation.status == "reserved"
    source_store.merge_conversation_data(
        merge_id=merge_id,
        tenant_id="source-test-tenant",
        source_conversation_id=source_id,
        target_conversation_id=target_id,
        expected_source_lifecycle_epoch=1,
        expected_target_lifecycle_epoch=1,
        source_label_at_merge="attested source",
    )

    merged = target_store.get_all_canonical_turns(target_id)
    moved = [row for row in merged if row.canonical_turn_id in source_ids]
    assert len(moved) == 2
    assert len({row.turn_group_number for row in moved}) == 1
    assert _exact_ingest(
        target_rec, target_id, **source_payload,
    ).turns_written == 0
    assert len(target_store.get_all_canonical_turns(target_id)) == 4


def test_completion_wins_epoch_lock_then_cutover_removes_whole_pair():
    conversation_id = f"source-epoch-completion-wins-{uuid.uuid4()}"
    store, reconciler = _stack(conversation_id)
    completion_has_epoch_lock = threading.Event()
    cutover_attempted = threading.Event()
    original_save = store.save_canonical_turn
    first = {"seen": False}

    def held_first_save(*args, **kwargs):
        if not first["seen"]:
            first["seen"] = True
            completion_has_epoch_lock.set()
            if not cutover_attempted.wait(15):
                raise AssertionError("cutover never attempted the epoch lock")
        return original_save(*args, **kwargs)

    store.save_canonical_turn = held_first_save
    payload = {
        "message_id": "1533000000000000300",
        "actor_id": "387316537012518913",
        "body": "completion commits before lifecycle cutover",
        "answer": "cutover must remove both halves",
    }

    def cutover():
        if not completion_has_epoch_lock.wait(15):
            raise AssertionError("completion never acquired epoch lock")
        cutover_attempted.set()
        with store.pool.connection() as conn, conn.transaction():
            conn.execute(
                "SELECT lifecycle_epoch FROM conversations "
                "WHERE conversation_id = %s FOR UPDATE",
                (conversation_id,),
            ).fetchone()
            conn.execute(
                "UPDATE conversations SET lifecycle_epoch = 2 "
                "WHERE conversation_id = %s",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM canonical_turns WHERE conversation_id = %s",
                (conversation_id,),
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        completion = executor.submit(
            _exact_ingest, reconciler, conversation_id, **payload,
        )
        cutover_future = executor.submit(cutover)
        assert completion.result(timeout=30).turns_written == 2
        cutover_future.result(timeout=30)
    assert store.get_all_canonical_turns(conversation_id) == []
    with store.pool.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM canonical_message_sources "
            "WHERE message_id = %s",
            (payload["message_id"],),
        ).fetchone()["n"] == 0


def test_cutover_wins_epoch_lock_then_stale_completion_writes_nothing():
    conversation_id = f"source-epoch-cutover-wins-{uuid.uuid4()}"
    store, reconciler = _stack(conversation_id)
    saver_entered = threading.Event()
    original_saver = store.save_attested_canonical_pair

    def observed_saver(*args, **kwargs):
        saver_entered.set()
        return original_saver(*args, **kwargs)

    store.save_attested_canonical_pair = observed_saver
    payload = {
        "message_id": "1533000000000000301",
        "actor_id": "387316537012518913",
        "body": "cutover commits before stale completion",
        "answer": "this pair must never be admitted",
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        with store.pool.connection() as conn, conn.transaction():
            conn.execute(
                "SELECT lifecycle_epoch FROM conversations "
                "WHERE conversation_id = %s FOR UPDATE",
                (conversation_id,),
            ).fetchone()
            future = executor.submit(
                _exact_ingest, reconciler, conversation_id, **payload,
            )
            assert saver_entered.wait(15)
            conn.execute(
                "UPDATE conversations SET lifecycle_epoch = 2 "
                "WHERE conversation_id = %s",
                (conversation_id,),
            )
        with pytest.raises(LifecycleEpochMismatch):
            future.result(timeout=30)

    assert store.get_all_canonical_turns(conversation_id) == []
    with store.pool.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM canonical_message_sources "
            "WHERE message_id = %s",
            (payload["message_id"],),
        ).fetchone()["n"] == 0
