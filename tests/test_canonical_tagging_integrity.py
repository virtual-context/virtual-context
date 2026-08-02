"""High-signal guards for canonical tagging source integrity."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from virtual_context.core.canonical_turns import compute_turn_hash_from_raw
from virtual_context.core.conversation_store import (
    ConversationStoreView,
    StaleConversationWriteError,
)
from virtual_context.core.store import canonical_rows_to_history
from virtual_context.core.tagging_pipeline import TaggingPipeline
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CANONICAL_TAGGING_IDENTITY_KEY,
    FactSignal,
    Message,
    TurnTagEntry,
)


_TAG_FIELDS = {
    "primary_tag",
    "tags",
    "session_date",
    "fact_signals",
    "code_refs",
    "tagged_at",
    "updated_at",
}


def _immutable_projection(row) -> dict:
    return {
        key: value
        for key, value in asdict(row).items()
        if key not in _TAG_FIELDS
    }


def test_tag_only_cas_cannot_change_content_or_discord_provenance(tmp_path):
    store = SQLiteStore(tmp_path / "tag-integrity.db")
    conversation_id = "sk:agent:vast:discord:guild:men"
    store.activate_conversation(conversation_id)
    store.upsert_conversation(tenant_id="tenant", conversation_id=conversation_id)
    turn_hash, normalized_user, normalized_assistant = (
        compute_turn_hash_from_raw("Cashew's actual message", "", version=1)
    )
    store.save_canonical_turn(
        conversation_id,
        7,
        "Cashew's actual message",
        "",
        canonical_turn_id="11111111-1111-4111-8111-111111111111",
        turn_hash=turn_hash,
        hash_version=1,
        normalized_user_text=normalized_user,
        normalized_assistant_text=normalized_assistant,
        turn_group_number=7,
        sender="Cashew King",
        sender_actor_id="actor:discord:cashew",
        source_message_id="1532787099986821351",
        origin_channel_id="1524946242499514418",
        origin_channel_label="vasttest",
        audience_conversation_id=conversation_id,
        audience_attribution_version=1,
    )
    before = store.get_all_canonical_turns(conversation_id)[0]

    updated = store.update_canonical_row_tagging_if_unchanged(
        canonical_turn_id=before.canonical_turn_id,
        conversation_id=conversation_id,
        expected_turn_hash=before.turn_hash,
        expected_lifecycle_epoch=1,
        primary_tag="supplements",
        tags=["supplements", "sleep"],
        session_date="2026-08-01T12:00:00",
        fact_signals=[FactSignal(subject="Cashew", verb="asked", object="x")],
        code_refs=[],
        require_untagged=True,
    )
    assert updated is True

    after = store.get_all_canonical_turns(conversation_id)[0]
    assert _immutable_projection(after) == _immutable_projection(before)
    assert after.primary_tag == "supplements"
    assert after.tags == ["supplements", "sleep"]
    assert after.tagged_at is not None


def test_tag_only_cas_rejects_stale_hash_and_epoch(tmp_path):
    store = SQLiteStore(tmp_path / "tag-stale.db")
    conversation_id = "conv"
    store.activate_conversation(conversation_id)
    store.upsert_conversation(tenant_id="tenant", conversation_id=conversation_id)
    old_hash, old_user, old_assistant = compute_turn_hash_from_raw(
        "old body", "", version=1,
    )
    store.save_canonical_turn(
        conversation_id,
        0,
        "old body",
        "",
        canonical_turn_id="22222222-2222-4222-8222-222222222222",
        turn_hash=old_hash,
        hash_version=1,
        normalized_user_text=old_user,
        normalized_assistant_text=old_assistant,
        turn_group_number=0,
    )
    new_hash, new_user, _ = compute_turn_hash_from_raw(
        "new body", "", version=1,
    )
    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET user_content = ?, "
        "normalized_user_text = ?, turn_hash = ? "
        "WHERE conversation_id = ?",
        ("new body", new_user, new_hash, conversation_id),
    )
    conn.commit()

    stale_hash = store.update_canonical_row_tagging_if_unchanged(
        canonical_turn_id="22222222-2222-4222-8222-222222222222",
        conversation_id=conversation_id,
        expected_turn_hash=old_hash,
        expected_lifecycle_epoch=1,
        primary_tag="wrong",
        tags=["wrong"],
        session_date="",
        fact_signals=[],
        code_refs=[],
    )
    stale_epoch = store.update_canonical_row_tagging_if_unchanged(
        canonical_turn_id="22222222-2222-4222-8222-222222222222",
        conversation_id=conversation_id,
        expected_turn_hash=new_hash,
        expected_lifecycle_epoch=999,
        primary_tag="wrong",
        tags=["wrong"],
        session_date="",
        fact_signals=[],
        code_refs=[],
    )
    row = store.get_all_canonical_turns(conversation_id)[0]
    assert stale_hash is False
    assert stale_epoch is False
    assert row.user_content == "new body"
    assert row.primary_tag == "_general"
    assert row.tagged_at is None


def test_non_strict_pair_cas_race_never_reconciles_stale_snapshot(
    tmp_path,
    monkeypatch,
):
    """One successful half-CAS plus one stale half must not append a pair."""
    from virtual_context.core.ingest_reconciler import IngestReconciler

    store = SQLiteStore(tmp_path / "pair-cas-race.db")
    conversation_id = "pair-cas-race"
    store.activate_conversation(conversation_id)
    store.upsert_conversation(tenant_id="tenant", conversation_id=conversation_id)
    user_hash, user_norm, _ = compute_turn_hash_from_raw(
        "user-old", "", version=1,
    )
    assistant_hash, _, assistant_norm = compute_turn_hash_from_raw(
        "", "assistant-old", version=1,
    )
    user_id = "99999999-9999-4999-8999-999999999991"
    assistant_id = "99999999-9999-4999-8999-999999999992"
    store.save_canonical_turn(
        conversation_id,
        0,
        "user-old",
        "",
        canonical_turn_id=user_id,
        turn_hash=user_hash,
        hash_version=1,
        normalized_user_text=user_norm,
        turn_group_number=0,
        sender="Cashew King",
        sender_actor_id="actor:discord:cashew",
        source_message_id="discord-user-old",
        origin_channel_id="discord-channel",
        audience_conversation_id=conversation_id,
        audience_attribution_version=1,
    )
    store.save_canonical_turn(
        conversation_id,
        1,
        "",
        "assistant-old",
        canonical_turn_id=assistant_id,
        turn_hash=assistant_hash,
        hash_version=1,
        normalized_assistant_text=assistant_norm,
        turn_group_number=0,
        sender="Vast",
        sender_actor_id="actor:vast",
        source_message_id="discord-assistant-old",
        origin_channel_id="discord-channel",
        reply_target_message_id="discord-user-old",
        reply_subject_actor_id="actor:discord:cashew",
        audience_conversation_id=conversation_id,
        audience_attribution_version=1,
    )

    pipeline = TaggingPipeline.__new__(TaggingPipeline)
    pipeline._store = store
    pipeline._semantic = None
    pipeline.config = SimpleNamespace(conversation_id=conversation_id)
    pipeline._engine_state = SimpleNamespace(lifecycle_epoch=1)

    original_cas = store.update_canonical_row_tagging_if_unchanged
    calls = {"count": 0}
    new_assistant_hash, _, new_assistant_norm = compute_turn_hash_from_raw(
        "", "assistant-new", version=1,
    )

    def racing_cas(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            conn = store._get_conn()
            conn.execute(
                "UPDATE canonical_turns SET assistant_content = ?, "
                "normalized_assistant_text = ?, turn_hash = ? "
                "WHERE canonical_turn_id = ?",
                (
                    "assistant-new",
                    new_assistant_norm,
                    new_assistant_hash,
                    assistant_id,
                ),
            )
            conn.commit()
        return original_cas(**kwargs)

    store.update_canonical_row_tagging_if_unchanged = racing_cas

    def stale_reconciliation_is_forbidden(*args, **kwargs):
        raise AssertionError("stale pair reached reconciliation")

    monkeypatch.setattr(
        IngestReconciler,
        "ingest_single",
        stale_reconciliation_is_forbidden,
    )
    user_message = Message(role="user", content="user-old")
    assistant_message = Message(role="assistant", content="assistant-old")
    consumed = pipeline._persist_canonical_turn(
        TurnTagEntry(
            turn_number=0,
            message_hash="entry",
            tags=["chat"],
            primary_tag="chat",
        ),
        user_message,
        assistant_message,
        pair_messages=[user_message, assistant_message],
        append_missing=True,
        expected_lifecycle_epoch=1,
    )

    assert consumed == 0
    rows = store.get_all_canonical_turns(conversation_id)
    assert len(rows) == 2
    assert [(row.user_content, row.assistant_content) for row in rows] == [
        ("user-old", ""),
        ("", "assistant-new"),
    ]
    assert rows[0].source_message_id == "discord-user-old"
    assert rows[0].sender_actor_id == "actor:discord:cashew"
    assert rows[1].source_message_id == "discord-assistant-old"
    assert rows[1].reply_target_message_id == "discord-user-old"


def test_canonical_resume_history_carries_each_physical_row_identity(tmp_path):
    store = SQLiteStore(tmp_path / "resume-identity.db")
    conversation_id = "conv"
    store.activate_conversation(conversation_id)
    store.upsert_conversation(tenant_id="tenant", conversation_id=conversation_id)
    user_hash, user_norm, _ = compute_turn_hash_from_raw("user", "", version=1)
    assistant_hash, _, assistant_norm = compute_turn_hash_from_raw(
        "", "assistant", version=1,
    )
    store.save_canonical_turn(
        conversation_id,
        0,
        "user",
        "",
        canonical_turn_id="33333333-3333-4333-8333-333333333333",
        turn_hash=user_hash,
        hash_version=1,
        normalized_user_text=user_norm,
        turn_group_number=0,
        source_message_id="discord-user-message",
    )
    store.save_canonical_turn(
        conversation_id,
        1,
        "",
        "assistant",
        canonical_turn_id="44444444-4444-4444-8444-444444444444",
        turn_hash=assistant_hash,
        hash_version=1,
        normalized_assistant_text=assistant_norm,
        turn_group_number=0,
        source_message_id="discord-bot-message",
    )

    history = canonical_rows_to_history(
        store.get_all_canonical_turns(conversation_id),
        include_tagging_identity=True,
    )
    assert [message.role for message in history] == ["user", "assistant"]
    user_identity = history[0].metadata[CANONICAL_TAGGING_IDENTITY_KEY]
    assistant_identity = history[1].metadata[CANONICAL_TAGGING_IDENTITY_KEY]
    assert user_identity == {
        "canonical_turn_id": "33333333-3333-4333-8333-333333333333",
        "turn_hash": user_hash,
        "turn_group_number": 0,
        "source_message_id": "discord-user-message",
    }
    assert assistant_identity == {
        "canonical_turn_id": "44444444-4444-4444-8444-444444444444",
        "turn_hash": assistant_hash,
        "turn_group_number": 0,
        "source_message_id": "discord-bot-message",
    }


def test_tag_only_cas_is_also_guarded_by_conversation_generation(tmp_path):
    store = SQLiteStore(tmp_path / "tag-generation.db")
    conversation_id = "conv"
    generation = store.activate_conversation(conversation_id)
    store.upsert_conversation(tenant_id="tenant", conversation_id=conversation_id)
    view = ConversationStoreView(store, conversation_id, generation)
    view.save_canonical_turn(
        conversation_id,
        0,
        "body",
        "",
        canonical_turn_id="55555555-5555-4555-8555-555555555555",
        turn_group_number=0,
    )
    row = store.get_all_canonical_turns(conversation_id)[0]
    store.begin_conversation_deletion(conversation_id)
    store.mark_conversation_deleted(conversation_id)

    with pytest.raises(StaleConversationWriteError):
        view.update_canonical_row_tagging_if_unchanged(
            canonical_turn_id=row.canonical_turn_id,
            conversation_id=conversation_id,
            expected_turn_hash=row.turn_hash,
            expected_lifecycle_epoch=1,
            primary_tag="must-not-write",
            tags=["must-not-write"],
            session_date="",
            fact_signals=[],
            code_refs=[],
        )

    # The store predicate must independently reject the write.  Deletion does
    # not advance lifecycle_epoch, so an epoch-only CAS would still succeed in
    # the check-then-delete race after the view's generation check.
    assert store.update_canonical_row_tagging_if_unchanged(
        canonical_turn_id=row.canonical_turn_id,
        conversation_id=conversation_id,
        expected_turn_hash=row.turn_hash,
        expected_lifecycle_epoch=1,
        primary_tag="must-not-write",
        tags=["must-not-write"],
        session_date="",
        fact_signals=[],
        code_refs=[],
    ) is False
    after = store.get_all_canonical_turns(conversation_id)[0]
    assert after.primary_tag == "_general"
    assert after.tagged_at is None


def test_durable_resume_tags_exact_physical_rows_without_tail_graft(tmp_path):
    """Exercise the production resume chain against a real SQLite engine.

    The already-indexed group is deliberately stored at the physical tail.
    The incident writer selected a physical tail by coverage count and could
    therefore graft pending bodies onto those victim rows.  Durable resume
    must instead reconstruct groups 1 and 2 with stamped physical identity,
    run real strict ingestion at a nonzero offset, and change tag fields only.
    """
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.proxy.state import ProxyState, SessionState
    from virtual_context.types import (
        KeywordTagConfig,
        StorageConfig,
        TagGeneratorConfig,
        TurnTagEntry,
        VirtualContextConfig,
    )

    conversation_id = "resume-physical-identity"
    config = VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "resume-chain.db"),
        ),
        tag_generator=TagGeneratorConfig(
            type="keyword",
            keyword_fallback=KeywordTagConfig(
                tag_keywords={
                    "legal": ["court", "motion"],
                    "biology": ["enzyme", "protein"],
                },
            ),
        ),
    )
    state = ProxyState(VirtualContextEngine(config=config))
    store = state.engine._store
    now = datetime.now(timezone.utc).isoformat()

    def save_half(
        *,
        row_id: str,
        group: int,
        sort_key: float,
        body: str,
        role: str,
        source_message_id: str,
        actor_id: str,
        sender: str,
        tagged: bool = False,
    ) -> None:
        user_content = body if role == "user" else ""
        assistant_content = body if role == "assistant" else ""
        turn_hash, normalized_user, normalized_assistant = (
            compute_turn_hash_from_raw(
                user_content,
                assistant_content,
                version=1,
            )
        )
        store.save_canonical_turn(
            conversation_id,
            int(sort_key),
            user_content,
            assistant_content,
            user_raw_content=user_content or None,
            assistant_raw_content=assistant_content or None,
            primary_tag="baseline" if tagged else "_general",
            tags=["baseline"] if tagged else [],
            session_date="2026-08-01T20:00:00",
            sender=sender,
            created_at=now,
            updated_at=now,
            canonical_turn_id=row_id,
            sort_key=sort_key,
            turn_hash=turn_hash,
            hash_version=1,
            normalized_user_text=normalized_user,
            normalized_assistant_text=normalized_assistant,
            tagged_at=now if tagged else None,
            first_seen_at=now,
            last_seen_at=now,
            source_batch_id=f"batch-{group}",
            turn_group_number=group,
            origin_channel_id="discord-channel",
            origin_channel_label="vasttest",
            sender_actor_id=actor_id,
            source_message_id=source_message_id,
            reply_target_message_id=(
                f"discord-user-{group}" if role == "assistant" else ""
            ),
            reply_subject_actor_id=(
                f"actor:user:{group}" if role == "assistant" else ""
            ),
            reply_subject_label=(
                f"Member {group}" if role == "assistant" else ""
            ),
            reply_target_body=(
                f"question-{group}" if role == "assistant" else ""
            ),
            reply_attribution_version=2 if role == "assistant" else 0,
            audience_conversation_id=conversation_id,
            audience_attribution_version=1,
        )

    try:
        # Pending logical groups are physically earlier than the indexed
        # victim group.  A count-based physical-tail resolver targets the
        # wrong rows; group/id/hash resolution does not.
        save_half(
            row_id="11111111-0000-4000-8000-000000000001",
            group=1,
            sort_key=3000.0,
            body="Please file the motion in court",
            role="user",
            source_message_id="discord-user-1",
            actor_id="actor:user:1",
            sender="Cashew King",
        )
        save_half(
            row_id="11111111-0000-4000-8000-000000000002",
            group=1,
            sort_key=4000.0,
            body="The court motion is ready",
            role="assistant",
            source_message_id="discord-bot-1",
            actor_id="actor:vast",
            sender="Vast",
        )
        save_half(
            row_id="22222222-0000-4000-8000-000000000001",
            group=2,
            sort_key=5000.0,
            body="Which enzyme changes this protein?",
            role="user",
            source_message_id="discord-user-2",
            actor_id="actor:user:2",
            sender="Kuw9239",
        )
        save_half(
            row_id="22222222-0000-4000-8000-000000000002",
            group=2,
            sort_key=6000.0,
            body="That enzyme modifies the protein",
            role="assistant",
            source_message_id="discord-bot-2",
            actor_id="actor:vast",
            sender="Vast",
        )
        save_half(
            row_id="00000000-0000-4000-8000-000000000001",
            group=0,
            sort_key=9000.0,
            body="victim user body must remain byte-identical",
            role="user",
            source_message_id="discord-user-0",
            actor_id="actor:user:0",
            sender="Roo",
            tagged=True,
        )
        save_half(
            row_id="00000000-0000-4000-8000-000000000002",
            group=0,
            sort_key=10000.0,
            body="victim assistant body must remain byte-identical",
            role="assistant",
            source_message_id="discord-bot-0",
            actor_id="actor:vast",
            sender="Vast",
            tagged=True,
        )

        before_rows = store.get_all_canonical_turns(conversation_id)
        before = {row.canonical_turn_id: asdict(row) for row in before_rows}
        victim_ids = {
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
        }

        state.engine._turn_tag_index.append(TurnTagEntry(
            turn_number=0,
            message_hash="baseline",
            tags=["baseline"],
            primary_tag="baseline",
        ))
        state.engine._engine_state.last_indexed_turn = 0
        state.engine._engine_state.last_completed_turn = 2
        store.upsert_ingestion_episode(
            conversation_id=conversation_id,
            lifecycle_epoch=1,
            worker_id=state._worker_id,
            raw_payload_entries=6,
        )
        assert store.set_phase(
            conversation_id=conversation_id,
            lifecycle_epoch=1,
            phase="ingesting",
        ) is True

        assert state.resume_pending_ingestion_if_needed() is True
        worker = state._ingestion_thread
        assert worker is not None
        worker.join(timeout=15.0)
        assert not worker.is_alive(), "durable resume did not finish"
        assert state.session_state == SessionState.ACTIVE

        after_rows = store.get_all_canonical_turns(conversation_id)
        after = {row.canonical_turn_id: asdict(row) for row in after_rows}
        assert set(after) == set(before)
        for row_id in victim_ids:
            assert after[row_id] == before[row_id]
        for row_id in set(after) - victim_ids:
            assert {
                key: value
                for key, value in after[row_id].items()
                if key not in _TAG_FIELDS
            } == {
                key: value
                for key, value in before[row_id].items()
                if key not in _TAG_FIELDS
            }
            assert after[row_id]["tagged_at"] is not None
            assert after[row_id]["primary_tag"] in {"legal", "biology"}
    finally:
        state.shutdown(wait=True)


def test_row_sweep_repairs_hashless_legacy_row_without_provenance_change(
    tmp_path,
):
    """A hashless legacy row must self-heal instead of blocking the queue."""
    from tests.test_handle_prepare_payload import _inner_store
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.proxy.state import ProxyState
    from virtual_context.types import (
        KeywordTagConfig,
        StorageConfig,
        TagGeneratorConfig,
        VirtualContextConfig,
    )

    conversation_id = "hashless-legacy-row"
    state = ProxyState(VirtualContextEngine(config=VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "hashless.db"),
        ),
        tag_generator=TagGeneratorConfig(
            type="keyword",
            keyword_fallback=KeywordTagConfig(
                tag_keywords={"legal": ["court", "motion"]},
            ),
        ),
    )))
    store = state.engine._store
    inner = _inner_store(state.engine)
    row_id = "88888888-8888-4888-8888-888888888888"
    try:
        store.save_canonical_turn(
            conversation_id,
            0,
            "Please file the motion in court",
            "",
            user_raw_content="raw Discord payload",
            canonical_turn_id=row_id,
            turn_group_number=0,
            sender="Cashew King",
            sender_actor_id="actor:discord:cashew",
            source_message_id="discord-hashless-source",
            origin_channel_id="discord-channel",
            origin_channel_label="vasttest",
            audience_conversation_id=conversation_id,
            audience_attribution_version=1,
        )
        conn = inner._get_conn()
        conn.execute(
            "UPDATE canonical_turns SET turn_hash = '', hash_version = 0, "
            "normalized_user_text = '', normalized_assistant_text = '' "
            "WHERE canonical_turn_id = ?",
            (row_id,),
        )
        conn.commit()
        before = store.get_all_canonical_turns(conversation_id)[0]

        store.upsert_ingestion_episode(
            conversation_id=conversation_id,
            lifecycle_epoch=1,
            worker_id=state._worker_id,
            raw_payload_entries=1,
        )
        assert store.claim_ingestion_lease(
            conversation_id=conversation_id,
            lifecycle_epoch=1,
            worker_id=state._worker_id,
            lease_ttl_s=30.0,
        ) is True
        assert store.set_phase(
            conversation_id=conversation_id,
            lifecycle_epoch=1,
            phase="ingesting",
        ) is True

        assert state._tagger_run() is True
        after = store.get_all_canonical_turns(conversation_id)[0]
        repaired_hash, normalized_user, normalized_assistant = (
            compute_turn_hash_from_raw(
                before.user_content,
                before.assistant_content,
                version=1,
            )
        )
        assert after.turn_hash == repaired_hash
        assert after.hash_version == 1
        assert after.normalized_user_text == normalized_user
        assert after.normalized_assistant_text == normalized_assistant
        assert after.primary_tag == "legal"
        assert after.tagged_at is not None

        derived = _TAG_FIELDS | {
            "turn_hash",
            "hash_version",
            "normalized_user_text",
            "normalized_assistant_text",
        }
        assert {
            key: value for key, value in asdict(after).items()
            if key not in derived
        } == {
            key: value for key, value in asdict(before).items()
            if key not in derived
        }
    finally:
        state.shutdown(wait=True)


def test_stamped_identity_cannot_be_reindexed_under_another_turn_number(tmp_path):
    store = SQLiteStore(tmp_path / "identity-turn.db")
    conversation_id = "conv"
    store.activate_conversation(conversation_id)
    store.upsert_conversation(tenant_id="tenant", conversation_id=conversation_id)
    user_hash, user_norm, _ = compute_turn_hash_from_raw("group one", "", version=1)
    assistant_hash, _, assistant_norm = compute_turn_hash_from_raw(
        "", "reply one", version=1,
    )
    store.save_canonical_turn(
        conversation_id,
        2,
        "group one",
        "",
        canonical_turn_id="66666666-6666-4666-8666-666666666666",
        turn_hash=user_hash,
        hash_version=1,
        normalized_user_text=user_norm,
        turn_group_number=1,
    )
    store.save_canonical_turn(
        conversation_id,
        3,
        "",
        "reply one",
        canonical_turn_id="77777777-7777-4777-8777-777777777777",
        turn_hash=assistant_hash,
        hash_version=1,
        normalized_assistant_text=assistant_norm,
        turn_group_number=1,
    )
    rows = store.get_all_canonical_turns(conversation_id)
    messages = canonical_rows_to_history(
        rows,
        include_tagging_identity=True,
    )
    pipeline = TaggingPipeline.__new__(TaggingPipeline)

    resolved = pipeline._resolve_strict_pair_rows(
        messages,
        turn_number=0,
        rows_by_id={row.canonical_turn_id: row for row in rows},
        rows_by_group={1: rows},
    )
    assert resolved is None
