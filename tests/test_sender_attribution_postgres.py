"""Postgres mirror of the sender store contracts.

Skipped unless a Postgres DSN is configured. Keeps the two SQL backends in
lockstep on:

* ``update_canonical_turn_senders_if_empty`` — compare-and-set on an empty
  sender, epoch-guarded, idempotent.
* ``list_canonical_conversation_ids`` — canonical-row enumeration with tenant
  scoping through ``conversations``.
* ``search_canonical_turn_text`` — sender-only lexical match with a labeled
  user excerpt, assistant excerpts unlabeled, empty-sender rows unchanged.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid

import pytest
from tests.pg_helpers import pg_dsn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


@pytest.fixture()
def store():
    from virtual_context.storage.postgres import PostgresStore  # deferred

    return PostgresStore(PG_URL)


def _conv(prefix: str = "conv") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _row(
    store,
    conv: str,
    *,
    ct_id: str,
    sort_key: float,
    user_content: str = "",
    assistant_content: str = "",
    sender: str = "",
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        sender=sender,
        primary_tag="chat",
        tags=["chat"],
    )


def _senders(store, conv: str) -> list[str]:
    return [r.sender for r in store.get_all_canonical_turns(conv)]


class TestUpdateCanonicalTurnSendersIfEmptyPg:
    def test_cas_sets_empty_sender(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="hello")
        ct_id = store.get_all_canonical_turns(conv)[0].canonical_turn_id
        assert store.update_canonical_turn_senders_if_empty(
            conv, {ct_id: "BigTex"}, expected_lifecycle_epoch=1,
        ) == 1
        assert _senders(store, conv) == ["BigTex"]

    def test_cas_never_overwrites_and_rerun_is_noop(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="hello", sender="BigTex")
        ct_id = store.get_all_canonical_turns(conv)[0].canonical_turn_id
        assert store.update_canonical_turn_senders_if_empty(
            conv, {ct_id: "Impostor"}, expected_lifecycle_epoch=1,
        ) == 0
        assert _senders(store, conv) == ["BigTex"]

    def test_stale_epoch_blocks_the_write(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="hello")
        ct_id = store.get_all_canonical_turns(conv)[0].canonical_turn_id
        assert store.update_canonical_turn_senders_if_empty(
            conv, {ct_id: "BigTex"}, expected_lifecycle_epoch=99,
        ) == 0
        assert _senders(store, conv) == [""]

    def test_empty_updates_is_a_noop(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        assert store.update_canonical_turn_senders_if_empty(conv, {}) == 0


class TestListCanonicalConversationIdsPg:
    def test_tenant_scope_and_canonical_only_enumeration(self, store):
        tenant = f"t-{uuid.uuid4().hex[:8]}"
        conv_a, conv_b, conv_empty = _conv("a"), _conv("b"), _conv("e")
        store.upsert_conversation(tenant_id=tenant, conversation_id=conv_a)
        store.upsert_conversation(tenant_id=tenant, conversation_id=conv_b)
        store.upsert_conversation(tenant_id=tenant, conversation_id=conv_empty)
        _row(store, conv_a, ct_id=str(uuid.uuid4()),
             sort_key=1000.0, user_content="u")
        _row(store, conv_b, ct_id=str(uuid.uuid4()),
             sort_key=1000.0, user_content="u")

        listed = store.list_canonical_conversation_ids(tenant_id=tenant)
        assert sorted(listed) == sorted([conv_a, conv_b])
        assert conv_empty not in listed

    def test_other_tenant_is_excluded(self, store):
        t1, t2 = f"t-{uuid.uuid4().hex[:8]}", f"t-{uuid.uuid4().hex[:8]}"
        conv1, conv2 = _conv("x"), _conv("y")
        store.upsert_conversation(tenant_id=t1, conversation_id=conv1)
        store.upsert_conversation(tenant_id=t2, conversation_id=conv2)
        _row(store, conv1, ct_id=str(uuid.uuid4()),
             sort_key=1000.0, user_content="u")
        _row(store, conv2, ct_id=str(uuid.uuid4()),
             sort_key=1000.0, user_content="u")
        assert store.list_canonical_conversation_ids(tenant_id=t1) == [conv1]


class TestSearchCanonicalTurnTextSenderPg:
    def test_sender_only_match_returns_labeled_user_excerpt(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex")
        results = store.search_canonical_turn_text("bigtex", conversation_id=conv)
        assert len(results) == 1
        assert results[0].matched_side == "user"
        assert results[0].text.startswith("BigTex: ")
        assert "bigtex" in results[0].text.lower()

    def test_sender_only_match_requires_user_content(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             assistant_content="neurological", sender="BigTex")
        assert store.search_canonical_turn_text("bigtex", conversation_id=conv) == []

    def test_assistant_excerpt_is_never_sender_labeled(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             assistant_content="neurological perhaps", sender="BigTex")
        qr = store.search_canonical_turn_text("neurological", conversation_id=conv)[0]
        assert qr.matched_side == "assistant"
        assert qr.text.startswith("Assistant: ")
        assert "BigTex" not in qr.text

    def test_user_text_match_on_sender_row_uses_sender_label(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex")
        qr = store.search_canonical_turn_text("tingling", conversation_id=conv)[0]
        assert qr.text.startswith("BigTex: ")

    def test_empty_sender_rows_render_as_before(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="toes tingling")
        qr = store.search_canonical_turn_text("tingling", conversation_id=conv)[0]
        assert qr.matched_side == "user"
        assert qr.text.startswith("User: ")
