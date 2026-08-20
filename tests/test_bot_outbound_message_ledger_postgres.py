"""The ledger of message ids the agent itself authored.

Skipped unless a Postgres DSN is configured.

A quote-reply is matched against ingested transport messages to decide whether
the quoted text is already on record. The agent's own outbound ids are recorded
nowhere, so a reply quoting the agent never matches and its own words are
re-filed as a quoted person's disclosure.

The property every test here defends: a real person's disclosure must never
disappear because this feature guessed. A recorded identity is positive
evidence. Everything else is unknown, and unknown must fall back to the
existing behaviour rather than to suppression.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)

TENANT = "t"
SCOPE = "scope"


def _store():
    from virtual_context.storage.postgres import PostgresStore
    return PostgresStore(PG_URL)


def _conv() -> str:
    return f"outbound-{uuid.uuid4().hex[:12]}"


def _identity(ns: str = "", **over) -> dict:
    """An identity in a namespace unique to the calling test.

    The ledger's key is deliberately global: a platform message belongs to one
    channel on one account, not to whichever conversation reports it first. So
    two tests reusing a literal channel and message id would collide on the
    primary key and the second would see a duplicate. The namespace keeps each
    test's identities its own without weakening the key under test.
    """
    tag = ns or uuid.uuid4().hex[:10]
    base = {
        "platform": "discord",
        "account_id": f"acct-{tag}",
        "channel_id": f"chan-{tag}",
        "message_id": f"msg-{tag}",
        "observed_at": datetime.now(timezone.utc),
    }
    base.update(over)
    return base


def _live(store) -> str:
    conv = _conv()
    store.upsert_conversation(tenant_id=TENANT, conversation_id=conv)
    return conv


def _ask(store, conv, ident) -> bool:
    return store.is_bot_authored_message(
        tenant_id=TENANT, agent_scope_id=SCOPE, conversation_id=conv,
        platform=ident["platform"], account_id=ident["account_id"],
        channel_id=ident["channel_id"], message_id=ident["message_id"],
    )


def _record(store, conv, *idents, **kw):
    return store.record_bot_outbound_messages(
        tenant_id=TENANT, agent_scope_id=SCOPE, conversation_id=conv,
        observed=list(idents), **kw,
    )


def test_a1_a_witnessed_identity_is_positive_evidence():
    store = _store()
    conv = _live(store)
    ident = _identity()

    assert _record(store, conv, ident)["accepted"] == 1
    assert _ask(store, conv, ident) is True


def test_a2_an_identity_that_never_arrives_stays_unknown():
    """The load-bearing negative. Nothing recorded must never read as
    'someone else authored this'."""
    store = _store()
    conv = _live(store)

    assert _ask(store, conv, _identity()) is False


def test_a3_the_same_identity_twice_is_a_no_op():
    store = _store()
    conv = _live(store)
    ident = _identity()

    first = _record(store, conv, ident)
    second = _record(store, conv, ident)

    assert first["accepted"] == 1
    assert second["accepted"] == 0
    assert second["duplicate"] == 1
    assert _ask(store, conv, ident) is True


def test_a4_late_and_out_of_order_identities_are_both_accepted():
    """Order carries no meaning; arrival after the turn is normal operation."""
    store = _store()
    conv = _live(store)
    ns = uuid.uuid4().hex[:10]
    now = datetime.now(timezone.utc)
    older = _identity(ns, message_id=f"{ns}-1", observed_at=now - timedelta(minutes=5))
    newer = _identity(ns, message_id=f"{ns}-2", observed_at=now - timedelta(minutes=1))

    _record(store, conv, newer)
    _record(store, conv, older)

    assert _ask(store, conv, older) is True
    assert _ask(store, conv, newer) is True


def test_a5_a_non_member_of_a_partial_set_is_unknown():
    """THE test. A recorded set is always partial, so non-membership proves
    nothing and must not read as 'authored by a person'."""
    store = _store()
    conv = _live(store)
    ns = uuid.uuid4().hex[:10]
    member = _identity(ns, message_id=f"{ns}-known")
    _record(store, conv, member)

    non_member = _identity(ns, message_id=f"{ns}-never-reported")

    assert _ask(store, conv, member) is True
    assert _ask(store, conv, non_member) is False, (
        "a message absent from a partial set was treated as known-not-ours"
    )


def test_a6_malformed_metadata_is_dropped_without_raising():
    """The turn is the product; the metadata is an enhancement. A bad identity
    must never be able to fail the caller."""
    store = _store()
    conv = _live(store)

    ns = uuid.uuid4().hex[:10]
    outcome = _record(
        store, conv,
        _identity(ns, message_id=""),
        _identity(ns, platform=""),
        _identity(ns, channel_id=" 111 "),
        _identity(ns, message_id="x" * 300),
        _identity(ns, observed_at="not-a-timestamp"),
    )

    assert outcome["accepted"] == 0
    assert outcome["malformed_identity"] == 5


def test_a7_an_identity_for_an_unknown_conversation_is_declined():
    store = _store()

    outcome = _record(store, _conv(), _identity())

    assert outcome["accepted"] == 0
    assert outcome["conversation_deleted"] == 1


def test_a8_the_same_bare_id_in_another_namespace_does_not_match():
    """Ids from unrelated platforms share a value space. Matching on a bare id
    would attribute one platform's message to another's."""
    store = _store()
    conv = _live(store)
    ns = uuid.uuid4().hex[:10]
    bare = f"{ns}-12345"
    _record(store, conv, _identity(ns, platform="discord", message_id=bare))

    assert _ask(store, conv, _identity(ns, platform="discord", message_id=bare)) is True
    for other in (
        _identity(ns, platform="telegram", message_id=bare),
        _identity(ns, account_id=f"other-{ns}", message_id=bare),
        _identity(ns, channel_id=f"other-{ns}", message_id=bare),
    ):
        assert _ask(store, conv, other) is False, (
            f"a bare id matched across namespaces: {other}"
        )


def test_an_identity_observed_before_the_current_epoch_is_declined():
    """The lifecycle fence at write. An observation from before this
    incarnation began describes the conversation that was deleted."""
    store = _store()
    conv = _live(store)
    started = store.get_lifecycle_epoch_started_at(conv)

    outcome = _record(
        store, conv,
        _identity(observed_at=started - timedelta(hours=1)),
        clock_skew_seconds=0,
    )

    assert outcome["accepted"] == 0
    assert outcome["fence_rejection"] == 1


def test_a_row_from_a_previous_epoch_does_not_speak_for_its_successor():
    """The lifecycle fence at read. Even a row admitted legitimately must stop
    counting once the conversation is deleted and recreated under the same id,
    or a queued identity becomes evidence about a different conversation."""
    store = _store()
    conv = _live(store)
    ident = _identity()
    _record(store, conv, ident)
    assert _ask(store, conv, ident) is True

    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET phase = 'deleted' WHERE conversation_id = %s",
            (conv,),
        )
    store.increment_lifecycle_epoch_on_resurrect(conv)

    assert _ask(store, conv, ident) is False, (
        "an identity recorded under the previous incarnation still reads as "
        "evidence about the conversation that replaced it"
    )


def test_an_unknown_epoch_start_declines_rather_than_assumes():
    """NULL start means unknown, not old. Nothing can be shown to have happened
    after a boundary the row does not know."""
    store = _store()
    conv = _live(store)
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET lifecycle_epoch_started_at = NULL "
            "WHERE conversation_id = %s",
            (conv,),
        )

    outcome = _record(store, conv, _identity())

    assert outcome["accepted"] == 0
    assert outcome["epoch_start_unknown"] == 1, (
        "an unrecorded epoch start must be named distinctly from a fence "
        f"firing, or a permanently inert conversation reads as a healthy "
        f"one: {outcome}"
    )
    assert "fence_rejection" not in outcome


def test_the_skew_allowance_declines_rather_than_admits():
    """The allowance exists because two clocks disagree. It must be spent on
    declining more, never on admitting something from before the boundary."""
    store = _store()
    conv = _live(store)
    started = store.get_lifecycle_epoch_started_at(conv)

    ns = uuid.uuid4().hex[:10]
    just_before = _record(
        store, conv,
        _identity(ns, message_id=f"{ns}-a", observed_at=started - timedelta(seconds=30)),
        clock_skew_seconds=0,
    )
    within_allowance = _record(
        store, conv,
        _identity(ns, message_id=f"{ns}-b", observed_at=started - timedelta(seconds=30)),
        clock_skew_seconds=300,
    )

    assert just_before["fence_rejection"] == 1
    assert within_allowance["accepted"] == 1


def test_an_identity_already_held_for_one_conversation_is_not_evidence_for_another():
    """A platform message belongs to one conversation. If the same identity is
    later reported under a different conversation, the first record stands and
    the second conversation gains no evidence, rather than the row being moved
    and the original quietly losing it.
    """
    store = _store()
    first, second = _live(store), _live(store)
    ident = _identity()

    assert _record(store, first, ident)["accepted"] == 1
    outcome = _record(store, second, ident)

    assert outcome["accepted"] == 0
    assert outcome["duplicate"] == 1
    assert _ask(store, first, ident) is True
    assert _ask(store, second, ident) is False, (
        "an identity recorded for one conversation became evidence for another"
    )


def test_a7_an_identity_for_a_merged_conversation_follows_the_alias():
    """A7. The messages moved to the survivor, so the identity should too,
    rather than being declined for a conversation that no longer answers."""
    store = _store()
    survivor = _live(store)
    merged_away = _conv()
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "INSERT INTO conversation_aliases (alias_id, target_id) VALUES (%s, %s) "
            "ON CONFLICT (alias_id) DO UPDATE SET target_id = EXCLUDED.target_id",
            (merged_away, survivor),
        )
    ident = _identity()

    outcome = _record(store, merged_away, ident)

    assert outcome["accepted"] == 1
    assert _ask(store, survivor, ident) is True, (
        "the identity was not recorded against the surviving conversation"
    )


def test_an_alias_chain_that_leads_nowhere_is_declined():
    """Unresolvable is unknown. The last id in a broken chain is not a
    fallback target."""
    store = _store()
    dangling, nowhere = _conv(), _conv()
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "INSERT INTO conversation_aliases (alias_id, target_id) VALUES (%s, %s) "
            "ON CONFLICT (alias_id) DO UPDATE SET target_id = EXCLUDED.target_id",
            (dangling, nowhere),
        )

    outcome = _record(store, dangling, _identity())

    assert outcome["accepted"] == 0
    assert outcome["conversation_deleted"] == 1


def test_an_alias_cycle_terminates_and_declines():
    store = _store()
    a, b = _conv(), _conv()
    with pg_test_conn().cursor() as cur:
        for alias, target in ((a, b), (b, a)):
            cur.execute(
                "INSERT INTO conversation_aliases (alias_id, target_id) VALUES (%s, %s) "
                "ON CONFLICT (alias_id) DO UPDATE SET target_id = EXCLUDED.target_id",
                (alias, target),
            )

    outcome = _record(store, a, _identity())

    assert outcome["accepted"] == 0
    assert outcome["ambiguous_alias_resolution"] == 1


def test_an_alias_cycle_is_bounded_rather_than_walked_to_the_recursion_limit():
    """A cycle must cost a handful of lookups, not a thousand.

    Terminating only because Python's recursion limit is reached would still
    decline the identity, so the outcome assertion above cannot tell the two
    apart. What separates them is how many database round-trips a single
    malformed alias chain costs.
    """
    store = _store()
    a, b = _conv(), _conv()
    with pg_test_conn().cursor() as cur:
        for alias, target in ((a, b), (b, a)):
            cur.execute(
                "INSERT INTO conversation_aliases (alias_id, target_id) VALUES (%s, %s) "
                "ON CONFLICT (alias_id) DO UPDATE SET target_id = EXCLUDED.target_id",
                (alias, target),
            )

    calls: list[str] = []
    real = store.resolve_conversation_alias

    def counting(alias_id):
        calls.append(alias_id)
        return real(alias_id)

    store.resolve_conversation_alias = counting
    outcome = _record(store, a, _identity())

    assert outcome["accepted"] == 0
    assert len(calls) <= 10, (
        f"a two-node alias cycle cost {len(calls)} lookups; the walk is not bounded"
    )


def test_a_missing_tenant_or_scope_is_declined_by_name():
    """Half an identity is not an identity. Recording under an empty tenant or
    scope would put the row in a namespace no reader ever builds."""
    store = _store()
    conv = _live(store)

    for kwargs in ({"tenant_id": ""}, {"agent_scope_id": ""}):
        outcome = store.record_bot_outbound_messages(
            tenant_id=kwargs.get("tenant_id", TENANT),
            agent_scope_id=kwargs.get("agent_scope_id", SCOPE),
            conversation_id=conv, observed=[_identity()],
        )
        assert outcome["accepted"] == 0
        assert outcome["unresolvable_tenant_scope"] == 1


def test_an_unreadable_or_ancient_timestamp_cannot_evade_the_fence():
    """The fence is only as good as the value it compares.

    A record whose observation time cannot be read must not be admitted as
    though it were fresh, and one stamped at the epoch must not slip past as
    though it were recent. Either would produce a record that outlives the
    conversation the fence exists to protect.
    """
    store = _store()
    conv = _live(store)
    ns = uuid.uuid4().hex[:10]

    unreadable = _record(
        store, conv, _identity(ns, message_id=f"{ns}-a", observed_at="whenever"),
    )
    missing = _record(
        store, conv, _identity(ns, message_id=f"{ns}-b", observed_at=None),
    )
    empty = _record(
        store, conv, _identity(ns, message_id=f"{ns}-c", observed_at=""),
    )
    epoch_zero = _record(
        store, conv,
        _identity(ns, message_id=f"{ns}-d",
                  observed_at=datetime(1970, 1, 1, tzinfo=timezone.utc)),
    )

    assert unreadable["malformed_identity"] == 1
    assert missing["malformed_identity"] == 1
    assert empty["malformed_identity"] == 1
    assert epoch_zero["fence_rejection"] == 1, (
        "a record stamped at the epoch was admitted as recent"
    )
    for ident in ("a", "b", "c", "d"):
        assert _ask(store, conv, _identity(ns, message_id=f"{ns}-{ident}")) is False


def _ns_conv(store, scope="vast", platform="discord", account="vast", channel=None):
    """A conversation whose channel has a RECORDED namespace.

    The reader builds its key from what the inbound path stored, so a writer
    test that never seeds that table is testing a different world from the one
    the guard reads in.
    """
    conv = _live(store)
    channel = channel or f"chan-{uuid.uuid4().hex[:10]}"
    now = datetime.now(timezone.utc).isoformat()
    with pg_test_conn().cursor() as cur:
        cur.execute(
            """INSERT INTO canonical_turns (
                   canonical_turn_id, conversation_id, turn_hash, hash_version,
                   normalized_user_text, normalized_assistant_text,
                   user_content, assistant_content, sort_key, source_batch_id,
                   first_seen_at, last_seen_at, covered_ingestible_entries,
                   tagged_at, created_at, updated_at
               ) VALUES (gen_random_uuid(), %s, %s, 1, 'u','a','u','a', 1000.0,
                         gen_random_uuid(), %s, %s, 1, NULL, %s, %s)
               RETURNING canonical_turn_id""",
            (conv, f"h-{uuid.uuid4().hex[:8]}", now, now, now, now),
        )
        turn_id = cur.fetchone()["canonical_turn_id"]
        cur.execute(
            """INSERT INTO canonical_message_sources (
                   tenant_id, agent_scope_id, platform, account_id, message_id,
                   canonical_turn_id, assistant_canonical_turn_id,
                   assistant_turn_hash, audience_conversation_id, channel_id,
                   guild_id, author_id, source_actor_id,
                   transport_body_sha256, canonical_body_sha256,
                   projection_version, canonical_turn_hash, observed_at,
                   created_at
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,'h',%s,%s,'g','author','actor',
                         'sha','sha','1','hash',%s,%s)""",
            (TENANT, scope, platform, account, f"in-{uuid.uuid4().hex[:8]}",
             turn_id, turn_id, conv, channel, now, now),
        )
    return conv, channel


def test_an_entry_carrying_its_own_scope_is_accepted():
    """The defect that made the ledger unwritable: the scope was read from a
    config field that does not exist, so every entry was declined for an
    unresolvable scope before any insert was attempted."""
    store = _store()
    conv, channel = _ns_conv(store)
    ident = _identity(channel_id=channel, account_id="vast")
    ident["agent_scope_id"] = "vast"

    outcome = store.record_bot_outbound_messages(
        tenant_id=TENANT, agent_scope_id="", conversation_id=conv,
        observed=[ident],
    )

    assert outcome["accepted"] == 1, f"declined: {outcome}"
    assert store.is_bot_authored_message(
        tenant_id=TENANT, agent_scope_id="vast", conversation_id=conv,
        platform="discord", account_id="vast",
        channel_id=channel, message_id=ident["message_id"],
    ) is True


def test_a_batch_with_no_scope_falls_back_to_the_channel_namespace():
    """When the sender names none, the channel's RECORDED namespace is the
    fallback, because that is the ruler the reader builds its key with. A
    scope invented from local configuration would file the identity under
    whichever agent this process happens to be."""
    store = _store()
    conv, channel = _ns_conv(store)
    ident = _identity(channel_id=channel, account_id="vast")

    outcome = store.record_bot_outbound_messages(
        tenant_id=TENANT, agent_scope_id="", conversation_id=conv,
        observed=[ident],
    )

    assert outcome["accepted"] == 1, f"declined: {outcome}"


def test_a_scope_the_reader_will_not_use_is_declined_by_name():
    """Writing under a namespace the reader never builds is not merely
    unmatched, it is unmatchable — and a set that never matches looks exactly
    like an empty one. It must be counted separately from malformed."""
    store = _store()
    conv, channel = _ns_conv(store, scope="vast", account="vast")
    ident = _identity(channel_id=channel, account_id="vast")
    ident["agent_scope_id"] = "some-other-agent"

    outcome = store.record_bot_outbound_messages(
        tenant_id=TENANT, agent_scope_id="", conversation_id=conv,
        observed=[ident],
    )

    assert outcome["accepted"] == 0
    assert outcome["namespace_mismatch"] == 1
    assert "malformed_identity" not in outcome


def test_an_account_id_that_is_not_the_recorded_one_is_declined():
    """The recorded account is an alias, not a platform snowflake. A sender
    that helpfully substitutes the snowflake writes an unmatchable row."""
    store = _store()
    conv, channel = _ns_conv(store, scope="vast", account="vast")
    ident = _identity(channel_id=channel, account_id="1540026970606403645")
    ident["agent_scope_id"] = "vast"

    outcome = store.record_bot_outbound_messages(
        tenant_id=TENANT, agent_scope_id="", conversation_id=conv,
        observed=[ident],
    )

    assert outcome["accepted"] == 0
    assert outcome["namespace_mismatch"] == 1


def test_a_missing_tenant_is_still_declined():
    store = _store()
    conv, channel = _ns_conv(store)
    outcome = store.record_bot_outbound_messages(
        tenant_id="", agent_scope_id="vast", conversation_id=conv,
        observed=[_identity(channel_id=channel)],
    )
    assert outcome["unresolvable_tenant_scope"] == 1
def test_sealing_never_overwrites_a_recorded_start():
    """A row that already knows when its epoch began keeps that answer, so a
    seal can neither move a real boundary nor erase one a resurrect wrote."""
    store = _store()
    conv, channel = _ns_conv(store)
    before = store.get_lifecycle_epoch_started_at(conv)
    assert before is not None

    ident = _identity(channel_id=channel, account_id="vast")
    ident["agent_scope_id"] = "vast"
    _record(store, conv, ident)

    assert store.get_lifecycle_epoch_started_at(conv) == before


def test_an_unknown_start_that_cannot_be_sealed_is_named_distinctly():
    """When the seal itself fails, the reason must say the start is unknown
    rather than that a fence fired. They need different remedies."""
    store = _store()
    conv, channel = _ns_conv(store)
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET lifecycle_epoch = 2, "
            "lifecycle_epoch_started_at = NULL WHERE conversation_id = %s",
            (conv,),
        )
    store._seal_lifecycle_epoch_start = lambda _cid: None

    outcome = _record(store, conv, _identity(channel_id=channel))

    assert outcome["accepted"] == 0
    assert outcome["epoch_start_unknown"] == 1
    assert "fence_rejection" not in outcome, (
        "an unknown epoch start was reported as a fence firing; a permanently "
        "inert conversation then reads as one correctly-declined id"
    )
def test_a_merged_conversation_forwards_to_its_survivor():
    """A merged conversation keeps its row while its turns move to the
    survivor. Recording against the husk files the evidence under a
    conversation the guard never asks about, so it is invisible rather than
    wrong — the harder of the two to notice.
    """
    store = _store()
    survivor = _live(store)
    husk = _live(store)
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET phase = 'merged' WHERE conversation_id = %s",
            (husk,),
        )
        cur.execute(
            "INSERT INTO conversation_aliases (alias_id, target_id) VALUES (%s, %s) "
            "ON CONFLICT (alias_id) DO UPDATE SET target_id = EXCLUDED.target_id",
            (husk, survivor),
        )
    ident = _identity()

    outcome = _record(store, husk, ident)

    assert outcome["accepted"] == 1, f"declined: {outcome}"
    assert _ask(store, survivor, ident) is True, (
        "the identity was filed against the merged husk, where the guard "
        "looking at the surviving conversation will never find it"
    )
    assert _ask(store, husk, ident) is False


def test_a_deleted_conversation_is_declined_not_forwarded():
    """Merged means the turns moved. Deleted means they are gone, and an
    identity for them belongs nowhere."""
    store = _store()
    conv = _live(store)
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET phase = 'deleted' WHERE conversation_id = %s",
            (conv,),
        )

    outcome = _record(store, conv, _identity())

    assert outcome["accepted"] == 0
    assert outcome["conversation_deleted"] == 1
