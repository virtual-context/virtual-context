"""Postgres mirror of the person-card store contracts.

Skipped unless a Postgres DSN is configured. Keeps the two SQL backends in
lockstep on the invariants that are privacy boundaries rather than features:
tenant scoping, audience scoping (a private DM must not shape a public answer),
cross-actor isolation, dirty-card unreadability, and delete invalidation.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import uuid
from datetime import datetime, timezone

import pytest
from tests.pg_helpers import pg_dsn

from virtual_context.types import (
    CARD_KIND_ACTIVE_GOAL,
    CARD_KIND_COMMUNICATION_PREF,
    CARD_SCOPE_CROSS_CONTEXT,
    CARD_SCOPE_SAME_CONVERSATION,
    ActorCardEntry,
    ActorCardEntrySource,
)

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


@pytest.fixture()
def store():
    from virtual_context.storage.postgres import PostgresStore  # deferred

    return PostgresStore(PG_URL)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class World:
    """One actor speaking in a private DM and in a public guild channel.

    ``canonical_turn_id`` is a real ``uuid`` column on Postgres and ``sort_key``
    is a float, so both are generated rather than borrowed from the name.
    """

    def __init__(self, store):
        self.store = store
        self.tenant = _uid("tenant")
        self.dm = _uid("dm")
        self.guild = _uid("guild")
        self.optics = f"actor:discord:{uuid.uuid4().hex[:10]}"
        self.ct_dm = str(uuid.uuid4())
        self.ct_guild = str(uuid.uuid4())
        self.seg_dm = _uid("seg")
        self.seg_guild = _uid("seg")
        self.f_dm = _uid("fact")
        self.f_guild = _uid("fact")

        with store.pool.connection() as conn:
            for cid in (self.dm, self.guild):
                conn.execute(
                    """INSERT INTO conversations
                           (conversation_id, tenant_id, lifecycle_epoch, phase,
                            created_at, updated_at)
                       VALUES (%s, %s, 1, 'active', %s, %s)""",
                    (cid, self.tenant, _now(), _now()),
                )
            self._turn(conn, self.ct_dm, self.dm, self.dm, "chan-dm", 1.0)
            self._turn(conn, self.ct_guild, self.guild, self.guild,
                       "chan-guild", 2.0)
            self._segment(conn, self.seg_dm, self.dm, [self.ct_dm])
            self._segment(conn, self.seg_guild, self.guild, [self.ct_guild])
            self._fact(conn, self.f_dm, self.dm, self.seg_dm)
            self._fact(conn, self.f_guild, self.guild, self.seg_guild)

        store.upsert_actor_profile_from_turn(
            self.dm, self.optics, "Optics", seen_at=_now(),
        )

    def _turn(self, conn, ctid, cid, audience, channel, sort_key):
        conn.execute(
            """INSERT INTO canonical_turns
                   (canonical_turn_id, conversation_id, turn_hash, sort_key,
                    user_content, assistant_content, sender_actor_id,
                    audience_conversation_id, audience_attribution_version,
                    origin_channel_id)
               VALUES (%s, %s, %s, %s, 'hello', '', %s, %s, 1, %s)""",
            (ctid, cid, uuid.uuid4().hex, sort_key, self.optics, audience,
             channel),
        )

    def _segment(self, conn, ref, cid, ctids):
        meta = {
            "canonical_turn_ids": list(ctids),
            "source_mapping_complete": True,
        }
        conn.execute(
            """INSERT INTO segments
                   (ref, conversation_id, primary_tag, summary, full_text,
                    messages_json, metadata_json, created_at,
                    start_timestamp, end_timestamp)
               VALUES (%s, %s, 'tag', 's', 'f', '[]', %s, %s, %s, %s)""",
            (ref, cid, json.dumps(meta), _now(), _now(), _now()),
        )

    def _fact(self, conn, fid, cid, ref):
        conn.execute(
            """INSERT INTO facts
                   (id, subject, verb, object, status, what, who, when_date,
                    "where", why, fact_type, tags_json, segment_ref,
                    conversation_id, turn_numbers_json, mentioned_at,
                    session_date, superseded_by, author_actor_id,
                    author_attribution_version, author_source_role,
                    author_source_message_id)
               VALUES (%s, 's', 'v', 'o', 'active', 'w', '', '', '', '',
                       'personal', '[]', %s, %s, '[]', %s, '', NULL, %s, 1,
                       'requester', '')""",
            (fid, ref, cid, _now(), self.optics),
        )


def _entry(entry_id, kind, body, *, scope=CARD_SCOPE_SAME_CONVERSATION):
    return ActorCardEntry(
        id=entry_id, kind=kind, body=body, confidence=0.9,
        sensitivity="normal", audience_scope=scope,
    )


def _source(entry_id, tenant, owner, audience, fact_id, channel=""):
    return ActorCardEntrySource(
        entry_id=entry_id, tenant_id=tenant, owner_conversation_id=owner,
        audience_conversation_id=audience, audience_channel_id=channel,
        fact_id=fact_id,
    )


def _build(w):
    goal_id, pref_id = _uid("e"), _uid("e")
    w.store.replace_actor_card(
        w.tenant, w.optics,
        [
            (_entry(goal_id, CARD_KIND_ACTIVE_GOAL, "private DM goal"),
             [_source(goal_id, w.tenant, w.dm, w.dm, w.f_dm, "chan-dm")]),
            (_entry(pref_id, CARD_KIND_COMMUNICATION_PREF, "prefers terse",
                    scope=CARD_SCOPE_CROSS_CONTEXT),
             [_source(pref_id, w.tenant, w.guild, w.guild, w.f_guild,
                      "chan-guild")]),
        ],
        input_hash="h1",
        expected_source_epochs={w.dm: 1, w.guild: 1},
    )
    return goal_id, pref_id


def _bodies(card):
    return sorted(e.body for e in card.entries) if card else None


def test_pg_list_actor_facts_derives_audience_and_is_tenant_scoped(store):
    w = World(store)
    got = {
        s.fact.id: (s.owner_conversation_id, s.audience_conversation_id,
                    s.audience_channel_id)
        for s in store.list_actor_facts(w.tenant, w.optics, limit=10)
    }
    assert got == {
        w.f_dm: (w.dm, w.dm, "chan-dm"),
        w.f_guild: (w.guild, w.guild, "chan-guild"),
    }
    assert store.list_actor_facts("someone-else", w.optics, limit=10) == []


def test_pg_dm_entry_is_not_served_in_the_guild(store):
    w = World(store)
    _build(w)

    in_dm = store.get_actor_card(
        w.tenant, w.optics, owner_conversation_id=w.dm,
        audience_conversation_id=w.dm, audience_channel_id="chan-dm",
    )
    assert _bodies(in_dm) == ["prefers terse", "private DM goal"]

    in_guild = store.get_actor_card(
        w.tenant, w.optics, owner_conversation_id=w.guild,
        audience_conversation_id=w.guild, audience_channel_id="chan-guild",
    )
    assert _bodies(in_guild) == ["prefers terse"]


def test_pg_cross_tenant_card_read_is_refused(store):
    w = World(store)
    _build(w)
    assert store.get_actor_card(
        "other-tenant", w.optics, owner_conversation_id=w.dm,
        audience_conversation_id=w.dm, audience_channel_id="chan-dm",
    ) is None


def test_pg_channel_mismatch_fails_closed(store):
    w = World(store)
    _build(w)
    card = store.get_actor_card(
        w.tenant, w.optics, owner_conversation_id=w.dm,
        audience_conversation_id=w.dm, audience_channel_id="chan-other",
    )
    assert "private DM goal" not in (_bodies(card) or [])


def test_pg_delete_conversation_removes_its_contribution(store):
    w = World(store)
    goal_id, pref_id = _build(w)

    store.delete_conversation(w.dm)

    with store.pool.connection() as conn:
        surviving = {
            r["id"] for r in conn.execute(
                "SELECT id FROM actor_card_entries WHERE tenant_id = %s",
                (w.tenant,),
            ).fetchall()
        }
    assert goal_id not in surviving
    assert surviving == {pref_id}

    # ...and the profile is dirty, so nothing is served until a clean rebuild.
    assert store.get_actor_card(
        w.tenant, w.optics, owner_conversation_id=w.guild,
        audience_conversation_id=w.guild, audience_channel_id="chan-guild",
    ) is None


def test_pg_stale_builder_cannot_write_after_epoch_change(store):
    w = World(store)
    with store.pool.connection() as conn:
        conn.execute(
            "UPDATE conversations SET lifecycle_epoch = 2 "
            "WHERE conversation_id = %s",
            (w.dm,),
        )
    eid = _uid("e")
    written = store.replace_actor_card(
        w.tenant, w.optics,
        [(_entry(eid, CARD_KIND_ACTIVE_GOAL, "stale"),
          [_source(eid, w.tenant, w.dm, w.dm, w.f_dm, "chan-dm")])],
        input_hash="h", expected_source_epochs={w.dm: 1},  # stale epoch
    )
    assert written == 0


def test_pg_replace_facts_for_segment_dirties_the_author_card(store):
    w = World(store)
    _build(w)
    assert store.get_actor_card(
        w.tenant, w.optics, owner_conversation_id=w.dm,
        audience_conversation_id=w.dm, audience_channel_id="chan-dm",
    ) is not None

    store.replace_facts_for_segment(w.dm, w.seg_dm, [])

    assert store.get_actor_card(
        w.tenant, w.optics, owner_conversation_id=w.dm,
        audience_conversation_id=w.dm, audience_channel_id="chan-dm",
    ) is None
