"""Person-card storage, audience policy, and lifecycle fences.

The worst failure this feature can have is cross-user contamination: one
member's prepare containing another member's card material. Cross-context
leakage (a private DM shaping a public answer) is the same severity. These
tests pin both structurally, at the store, so no prompt wording is load-bearing.
"""
import json
import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CARD_KIND_ACTIVE_GOAL,
    CARD_KIND_COMMUNICATION_PREF,
    CARD_SCOPE_CROSS_CONTEXT,
    CARD_SCOPE_SAME_CONVERSATION,
    CARD_SENSITIVITY_HIGH,
    ActorCardEntry,
    ActorCardEntrySource,
)

OPTICS = "actor:discord:optics"
BIGTEX = "actor:discord:bigtex"


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "cards.db"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _conversation(store, cid, tenant="t1", phase="active", epoch=1):
    conn = store._get_conn()
    now = _now()
    conn.execute(
        """INSERT INTO conversations
               (conversation_id, tenant_id, lifecycle_epoch, phase,
                created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (cid, tenant, epoch, phase, now, now),
    )
    conn.commit()


def _turn(store, ctid, cid, actor, audience, channel="", content="hello"):
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO canonical_turns
               (canonical_turn_id, conversation_id, turn_hash, sort_key,
                user_content, assistant_content, sender_actor_id,
                audience_conversation_id, audience_attribution_version,
                origin_channel_id)
           VALUES (?, ?, ?, ?, ?, '', ?, ?, 1, ?)""",
        (ctid, cid, ctid, ctid, content, actor, audience, channel),
    )
    conn.commit()


def _segment(store, ref, cid, ctids, complete=True):
    conn = store._get_conn()
    now = _now()
    meta = {"canonical_turn_ids": list(ctids), "source_mapping_complete": complete}
    conn.execute(
        """INSERT INTO segments
               (ref, conversation_id, primary_tag, summary, full_text,
                messages_json, metadata_json, created_at,
                start_timestamp, end_timestamp)
           VALUES (?, ?, 'tag', 's', 'f', '[]', ?, ?, ?, ?)""",
        (ref, cid, json.dumps(meta), now, now, now),
    )
    conn.commit()


def _fact(store, fid, cid, ref, actor, version=1):
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO facts
               (id, subject, verb, object, status, what, who, when_date,
                "where", why, fact_type, tags_json, segment_ref,
                conversation_id, turn_numbers_json, mentioned_at, session_date,
                superseded_by, author_actor_id, author_attribution_version,
                author_source_role, author_source_message_id)
           VALUES (?, 's', 'v', 'o', 'active', 'w', '', '', '', '',
                   'personal', '[]', ?, ?, '[]', ?, '', NULL, ?, ?,
                   'requester', '')""",
        (fid, ref, cid, _now(), actor, version),
    )
    conn.commit()


def _entry(entry_id, kind, body, *, scope=CARD_SCOPE_SAME_CONVERSATION,
           sensitivity="normal", confidence=0.9):
    return ActorCardEntry(
        id=entry_id, kind=kind, body=body, confidence=confidence,
        sensitivity=sensitivity, audience_scope=scope,
    )


def _source(entry_id, owner, audience, fact_id, channel="", tenant="t1"):
    return ActorCardEntrySource(
        entry_id=entry_id, tenant_id=tenant, owner_conversation_id=owner,
        audience_conversation_id=audience, audience_channel_id=channel,
        fact_id=fact_id,
    )


def _dm_and_guild(store):
    """One actor speaking in a private DM and in a public guild channel."""
    _conversation(store, "dm")
    _conversation(store, "guild")
    _turn(store, "ct-dm", "dm", OPTICS, "dm", "chan-dm")
    _turn(store, "ct-guild", "guild", OPTICS, "guild", "chan-guild")
    _segment(store, "seg-dm", "dm", ["ct-dm"])
    _segment(store, "seg-guild", "guild", ["ct-guild"])
    _fact(store, "f-dm", "dm", "seg-dm", OPTICS)
    _fact(store, "f-guild", "guild", "seg-guild", OPTICS)
    store.upsert_actor_profile_from_turn("dm", OPTICS, "Optics", seen_at=_now())


def _build_dm_goal_and_cross_pref(store):
    goal = _entry("e-goal", CARD_KIND_ACTIVE_GOAL, "private DM goal")
    pref = _entry("e-pref", CARD_KIND_COMMUNICATION_PREF, "prefers terse answers",
                  scope=CARD_SCOPE_CROSS_CONTEXT)
    return store.replace_actor_card(
        "t1", OPTICS,
        [
            (goal, [_source("e-goal", "dm", "dm", "f-dm", "chan-dm")]),
            (pref, [_source("e-pref", "guild", "guild", "f-guild", "chan-guild")]),
        ],
        input_hash="h1",
        expected_source_epochs={"dm": 1, "guild": 1},
    )


def _bodies(card):
    return sorted(e.body for e in card.entries) if card else None


# ---------------------------------------------------------------------------
# Fact enumeration
# ---------------------------------------------------------------------------

def test_list_actor_facts_derives_audience_from_canonical_rows(store):
    _dm_and_guild(store)
    sources = store.list_actor_facts("t1", OPTICS, limit=10)
    got = {s.fact.id: (s.owner_conversation_id, s.audience_conversation_id,
                       s.audience_channel_id) for s in sources}
    assert got == {
        "f-dm": ("dm", "dm", "chan-dm"),
        "f-guild": ("guild", "guild", "chan-guild"),
    }


def test_list_actor_facts_is_tenant_scoped(store):
    """An actor id shared by two tenants must never cross the fact query."""
    _dm_and_guild(store)
    assert store.list_actor_facts("other-tenant", OPTICS, limit=10) == []


def test_incomplete_source_mapping_is_card_ineligible(store):
    _conversation(store, "c1")
    _turn(store, "ct1", "c1", OPTICS, "c1", "chan")
    _segment(store, "seg1", "c1", ["ct1"], complete=False)
    _fact(store, "f1", "c1", "seg1", OPTICS)
    store.upsert_actor_profile_from_turn("c1", OPTICS, "Optics", seen_at=_now())
    assert store.list_actor_facts("t1", OPTICS, limit=10) == []


def test_legacy_unversioned_audience_is_card_ineligible(store):
    """A row with no proven audience cannot fall back to the owner."""
    _conversation(store, "c1")
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO canonical_turns
               (canonical_turn_id, conversation_id, turn_hash, sort_key,
                user_content, assistant_content, sender_actor_id,
                audience_conversation_id, audience_attribution_version)
           VALUES ('ct1', 'c1', 'h', 's', 'hi', '', ?, '', 0)""",
        (OPTICS,),
    )
    conn.commit()
    _segment(store, "seg1", "c1", ["ct1"])
    _fact(store, "f1", "c1", "seg1", OPTICS)
    store.upsert_actor_profile_from_turn("c1", OPTICS, "Optics", seen_at=_now())
    assert store.list_actor_facts("t1", OPTICS, limit=10) == []


# ---------------------------------------------------------------------------
# Audience policy — the private-to-public leakage boundary
# ---------------------------------------------------------------------------

def test_dm_entry_is_not_served_in_the_guild(store):
    """Influence-only use is NOT an audience boundary: a DM goal can leak by
    shaping a public answer without ever being quoted. It must be structurally
    absent, not merely un-quoted."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)

    in_dm = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    )
    assert _bodies(in_dm) == ["prefers terse answers", "private DM goal"]

    in_guild = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    )
    assert _bodies(in_guild) == ["prefers terse answers"]
    assert "private DM goal" not in (_bodies(in_guild) or [])


def test_channel_mismatch_fails_closed(store):
    """A source from channel A is not served to a same_conversation request in
    channel B."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-other",
    )
    assert "private DM goal" not in (_bodies(card) or [])


def test_unknown_source_channel_fails_closed(store):
    """An empty source channel is unknown, not wildcard."""
    _conversation(store, "c1")
    _turn(store, "ct1", "c1", OPTICS, "c1", channel="")  # no durable channel
    _segment(store, "seg1", "c1", ["ct1"])
    _fact(store, "f1", "c1", "seg1", OPTICS)
    store.upsert_actor_profile_from_turn("c1", OPTICS, "Optics", seen_at=_now())
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "goal"),
          [_source("e1", "c1", "c1", "f1", channel="")])],
        input_hash="h", expected_source_epochs={"c1": 1},
    )
    # Request HAS a durable channel; the source's is unknown -> excluded.
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="c1",
        audience_conversation_id="c1", audience_channel_id="chan-real",
    )
    assert card is None


def test_empty_audience_reads_no_card(store):
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="", audience_channel_id="chan-dm",
    ) is None


def test_high_sensitivity_entries_are_never_served(store):
    _dm_and_guild(store)
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "secret", sensitivity=CARD_SENSITIVITY_HIGH),
          [_source("e1", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h", expected_source_epochs={"dm": 1},
    )
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    )
    assert card is None


# ---------------------------------------------------------------------------
# Cross-actor and cross-tenant isolation — the rubric's worst failures
# ---------------------------------------------------------------------------

def test_no_cross_actor_leakage(store):
    """A prepare for actor A never contains any entry belonging to actor B."""
    _conversation(store, "guild")
    _turn(store, "ct-o", "guild", OPTICS, "guild", "chan")
    _turn(store, "ct-b", "guild", BIGTEX, "guild", "chan")
    _segment(store, "seg-o", "guild", ["ct-o"])
    _segment(store, "seg-b", "guild", ["ct-b"])
    _fact(store, "f-o", "guild", "seg-o", OPTICS)
    _fact(store, "f-b", "guild", "seg-b", BIGTEX)
    store.upsert_actor_profile_from_turn("guild", OPTICS, "Optics", seen_at=_now())
    store.upsert_actor_profile_from_turn("guild", BIGTEX, "BigTex", seen_at=_now())

    store.replace_actor_card(
        "t1", BIGTEX,
        [(_entry("e-b", CARD_KIND_ACTIVE_GOAL, "bigtex protocol claim"),
          [_source("e-b", "guild", "guild", "f-b", "chan")])],
        input_hash="hb", expected_source_epochs={"guild": 1},
    )
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e-o", CARD_KIND_ACTIVE_GOAL, "optics own goal"),
          [_source("e-o", "guild", "guild", "f-o", "chan")])],
        input_hash="ho", expected_source_epochs={"guild": 1},
    )

    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan",
    )
    assert _bodies(card) == ["optics own goal"]
    assert "bigtex protocol claim" not in (_bodies(card) or [])


def test_no_cross_tenant_card_read(store):
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    assert store.get_actor_card(
        "other-tenant", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


# ---------------------------------------------------------------------------
# Cache semantics and lifecycle fences
# ---------------------------------------------------------------------------

def test_entries_supersede_rather_than_duplicate(store):
    _dm_and_guild(store)
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "first"),
          [_source("e1", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h1", expected_source_epochs={"dm": 1},
    )
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e2", CARD_KIND_ACTIVE_GOAL, "second"),
          [_source("e2", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h2", expected_source_epochs={"dm": 1},
    )
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    )
    assert _bodies(card) == ["second"]

    conn = store._get_conn()
    old = conn.execute(
        "SELECT superseded_by FROM actor_card_entries WHERE id = 'e1'"
    ).fetchone()
    assert old[0] == "e2"  # superseded by its same-kind replacement, not deleted


def test_dirty_card_is_not_served(store):
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    conn = store._get_conn()
    conn.execute(
        "UPDATE actor_profiles SET card_dirty = 1 WHERE actor_id = ?", (OPTICS,)
    )
    conn.commit()
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


def test_delete_conversation_removes_its_contribution_to_every_card(store):
    """No entry whose fact owner or audience origin is deleted may survive."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)

    store.delete_conversation("dm")

    conn = store._get_conn()
    surviving = {r[0] for r in conn.execute("SELECT id FROM actor_card_entries")}
    assert "e-goal" not in surviving          # the DM-sourced entry is gone
    assert surviving == {"e-pref"}

    # ...and the profile is dirty, so nothing is served until a clean rebuild.
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    ) is None


def test_stale_builder_cannot_resurrect_after_epoch_change(store):
    """A builder that enumerated at epoch 1 cannot write after a resurrect."""
    _dm_and_guild(store)
    conn = store._get_conn()
    conn.execute(
        "UPDATE conversations SET lifecycle_epoch = 2 WHERE conversation_id = 'dm'"
    )
    conn.commit()
    written = store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "stale"),
          [_source("e1", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h", expected_source_epochs={"dm": 1},  # stale epoch
    )
    assert written == 0
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


def test_replace_facts_for_segment_dirties_the_author_card(store):
    """Fact replacement must dirty the card in the same transaction, or stale
    card content stays readable through the crash window."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    # Card is clean and readable to begin with.
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is not None

    store.replace_facts_for_segment("dm", "seg-dm", [])

    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


def test_card_read_requires_route_to_resolve_to_the_owner(store):
    """An audience route that does not prove owner-or-alias-to-owner reads no
    card, rather than silently falling back to the resolved owner."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="dm",   # unrelated route, not an alias of guild
        audience_channel_id="chan-guild",
    ) is None
