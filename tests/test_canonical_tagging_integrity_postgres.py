"""PostgreSQL parity for the canonical tag-only content/provenance CAS."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import asdict
import uuid

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn
from virtual_context.core.canonical_turns import compute_turn_hash_from_raw
from virtual_context.types import FactSignal


pytestmark = pytest.mark.skipif(
    not pg_dsn(),
    reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set",
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


def test_postgres_tag_only_cas_guards_content_epoch_and_deletion():
    from virtual_context.storage.postgres import PostgresStore

    store = PostgresStore(pg_dsn())
    conversation_id = f"tag-integrity-{uuid.uuid4().hex}"
    canonical_turn_id = str(uuid.uuid4())
    hashless_turn_id = str(uuid.uuid4())
    store.upsert_conversation(
        tenant_id="canonical-tag-integrity-test",
        conversation_id=conversation_id,
    )
    turn_hash, normalized_user, normalized_assistant = (
        compute_turn_hash_from_raw(
            "Cashew's immutable Discord body",
            "",
            version=1,
        )
    )
    try:
        store.save_canonical_turn(
            conversation_id,
            4,
            "Cashew's immutable Discord body",
            "",
            user_raw_content="raw Discord body",
            canonical_turn_id=canonical_turn_id,
            turn_hash=turn_hash,
            hash_version=1,
            normalized_user_text=normalized_user,
            normalized_assistant_text=normalized_assistant,
            turn_group_number=4,
            sender="Cashew King",
            sender_actor_id="actor:discord:cashew",
            source_message_id="1532787099986821351",
            origin_channel_id="1524946242499514418",
            origin_channel_label="vasttest",
            reply_target_message_id="reply-target",
            reply_subject_actor_id="actor:discord:roo",
            reply_subject_label="Roo",
            reply_target_body="quoted body",
            reply_attribution_version=2,
            audience_conversation_id=conversation_id,
            audience_attribution_version=1,
        )
        before = store.get_all_canonical_turns(conversation_id)[0]

        assert store.update_canonical_row_tagging_if_unchanged(
            canonical_turn_id=canonical_turn_id,
            conversation_id=conversation_id,
            expected_turn_hash=turn_hash,
            expected_lifecycle_epoch=1,
            primary_tag="supplements",
            tags=["supplements", "sleep"],
            session_date="2026-08-01T20:00:00",
            fact_signals=[
                FactSignal(subject="Cashew", verb="asked", object="sleep"),
            ],
            code_refs=[],
            require_untagged=True,
        ) is True
        after = store.get_all_canonical_turns(conversation_id)[0]
        assert _immutable_projection(after) == _immutable_projection(before)
        assert after.primary_tag == "supplements"
        assert after.tags == ["supplements", "sleep"]

        assert store.update_canonical_row_tagging_if_unchanged(
            canonical_turn_id=canonical_turn_id,
            conversation_id=conversation_id,
            expected_turn_hash="stale-hash",
            expected_lifecycle_epoch=1,
            primary_tag="wrong",
            tags=["wrong"],
            session_date="",
            fact_signals=[],
            code_refs=[],
        ) is False
        assert store.update_canonical_row_tagging_if_unchanged(
            canonical_turn_id=canonical_turn_id,
            conversation_id=conversation_id,
            expected_turn_hash=turn_hash,
            expected_lifecycle_epoch=999,
            primary_tag="wrong",
            tags=["wrong"],
            session_date="",
            fact_signals=[],
            code_refs=[],
        ) is False

        store.save_canonical_turn(
            conversation_id,
            5,
            "legacy hashless body",
            "",
            user_raw_content="legacy raw body",
            canonical_turn_id=hashless_turn_id,
            turn_group_number=5,
            sender="Roo",
            sender_actor_id="actor:discord:roo",
            source_message_id="legacy-source-message",
            origin_channel_id="legacy-channel",
            origin_channel_label="vasttest",
            audience_conversation_id=conversation_id,
            audience_attribution_version=1,
        )
        conn = pg_test_conn()
        conn.execute(
            "UPDATE canonical_turns SET turn_hash = '', hash_version = 0, "
            "normalized_user_text = '', normalized_assistant_text = '' "
            "WHERE canonical_turn_id = %s",
            (hashless_turn_id,),
        )
        hashless_before = [
            row for row in store.get_all_canonical_turns(conversation_id)
            if row.canonical_turn_id == hashless_turn_id
        ][0]
        repaired_hash, repaired_user, repaired_assistant = (
            compute_turn_hash_from_raw(
                hashless_before.user_content,
                hashless_before.assistant_content,
                version=1,
            )
        )
        assert store.backfill_canonical_row_hash_if_empty(
            canonical_turn_id=hashless_turn_id,
            conversation_id=conversation_id,
            expected_lifecycle_epoch=1,
            expected_user_content=hashless_before.user_content,
            expected_assistant_content=hashless_before.assistant_content,
            turn_hash=repaired_hash,
            hash_version=1,
            normalized_user_text=repaired_user,
            normalized_assistant_text=repaired_assistant,
        ) is True
        hashless_after = [
            row for row in store.get_all_canonical_turns(conversation_id)
            if row.canonical_turn_id == hashless_turn_id
        ][0]
        assert hashless_after.turn_hash == repaired_hash
        assert hashless_after.hash_version == 1
        assert hashless_after.normalized_user_text == repaired_user
        hash_fields = {
            "turn_hash",
            "hash_version",
            "normalized_user_text",
            "normalized_assistant_text",
            "updated_at",
        }
        assert {
            key: value for key, value in asdict(hashless_after).items()
            if key not in hash_fields
        } == {
            key: value for key, value in asdict(hashless_before).items()
            if key not in hash_fields
        }

        # Deletion does not advance lifecycle_epoch.  The SQL statement must
        # still reject a tag write that races after the deletion commit.
        store.mark_conversation_deleted(conversation_id)
        assert store.update_canonical_row_tagging_if_unchanged(
            canonical_turn_id=canonical_turn_id,
            conversation_id=conversation_id,
            expected_turn_hash=turn_hash,
            expected_lifecycle_epoch=1,
            primary_tag="wrong",
            tags=["wrong"],
            session_date="",
            fact_signals=[],
            code_refs=[],
        ) is False
        final = store.get_all_canonical_turns(conversation_id)[0]
        assert final.primary_tag == "supplements"
        assert _immutable_projection(final) == _immutable_projection(before)
    finally:
        conn = pg_test_conn()
        conn.execute(
            "DELETE FROM canonical_turns WHERE conversation_id = %s",
            (conversation_id,),
        )
        conn.execute(
            "DELETE FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        )
        store.close()
