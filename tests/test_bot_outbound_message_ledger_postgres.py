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
    assert outcome.get("malformed", 0) + outcome.get("not_canonical", 0) == 5


def test_a7_an_identity_for_an_unknown_conversation_is_declined():
    store = _store()

    outcome = _record(store, _conv(), _identity())

    assert outcome["accepted"] == 0
    assert outcome["unknown_conversation"] == 1


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
    assert outcome["predates_epoch"] == 1


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
    assert outcome["epoch_start_unknown"] == 1


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

    assert just_before["predates_epoch"] == 1
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
    assert outcome["unknown_conversation"] == 1


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
    assert outcome["unknown_conversation"] == 1


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
