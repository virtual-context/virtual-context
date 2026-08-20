"""When the current lifecycle epoch began, recorded rather than inferred.

Skipped unless a Postgres DSN is configured.

A conversation can be deleted and recreated under the same id. Evidence
observed before the current incarnation began describes its predecessor, and
attributing it here promotes a thing that is unknown about this conversation
into positive evidence about it. Deciding that requires knowing when the
current epoch started, which nothing recorded.

``lifecycle_epoch_started_at`` records it. NULL means unknown, and unknown is
not old: a caller cannot show that anything happened after a start it does not
know, so it must decline rather than assume.
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


def _store():
    from virtual_context.storage.postgres import PostgresStore
    return PostgresStore(PG_URL)


def _conv() -> str:
    return f"epoch-start-{uuid.uuid4().hex[:12]}"


def test_new_conversation_records_when_its_first_epoch_began():
    store = _store()
    conv = _conv()
    before = datetime.now(timezone.utc) - timedelta(seconds=5)

    store.upsert_conversation(tenant_id="t", conversation_id=conv)

    started = store.get_lifecycle_epoch_started_at(conv)
    assert started is not None, "a conversation was created with no epoch start"
    assert started >= before
    assert store.get_lifecycle_epoch(conv) == 1


def test_resurrect_moves_the_epoch_start_forward():
    """The bump is the whole point: after a delete and recreate, evidence from
    before the recreate must be distinguishable from evidence after it."""
    store = _store()
    conv = _conv()
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    first_start = store.get_lifecycle_epoch_started_at(conv)

    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET phase = 'deleted' WHERE conversation_id = %s",
            (conv,),
        )
    new_epoch = store.increment_lifecycle_epoch_on_resurrect(conv)

    assert new_epoch == 2
    second_start = store.get_lifecycle_epoch_started_at(conv)
    assert second_start is not None
    assert second_start > first_start, (
        "the epoch was bumped but its start was left at the previous "
        "incarnation's, so evidence from before the recreate still reads as "
        "belonging to the successor"
    )


def test_a_failed_resurrect_does_not_move_the_epoch_start():
    """The bump is guarded on phase='deleted'. A no-op bump must not restamp
    the start, or a live conversation's window would keep sliding forward."""
    store = _store()
    conv = _conv()
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    first_start = store.get_lifecycle_epoch_started_at(conv)

    epoch = store.increment_lifecycle_epoch_on_resurrect(conv)

    assert epoch == 1
    assert store.get_lifecycle_epoch_started_at(conv) == first_start


def test_backfill_fills_never_resurrected_rows_and_leaves_the_rest_unknown():
    """The backfill must only write a value it can derive.

    ``lifecycle_epoch = 1`` means the row was never resurrected, so its epoch
    began at ``created_at``. A row past epoch 1 was resurrected at a time
    nothing recorded, and inventing one would be exactly the guess this column
    exists to avoid.

    The rows are seeded and then a NEW store is constructed, so the assertion
    runs against the migration in the schema bootstrap. Running a copy of the
    backfill statement from inside the test would assert on a second
    implementation and pass while the real one was wrong.
    """
    _store()  # ensure the column exists before seeding it
    virgin, resurrected = _conv(), _conv()
    now = datetime.now(timezone.utc)
    with pg_test_conn().cursor() as cur:
        for conv, epoch in ((virgin, 1), (resurrected, 3)):
            cur.execute(
                """INSERT INTO conversations (
                       conversation_id, tenant_id, lifecycle_epoch,
                       created_at, updated_at, lifecycle_epoch_started_at
                   ) VALUES (%s, 't', %s, %s, %s, NULL)""",
                (conv, epoch, now, now),
            )

    store = _store()  # runs _ensure_schema, and with it the real backfill

    assert store.get_lifecycle_epoch_started_at(virgin) is not None, (
        "the backfill left a never-resurrected row without an epoch start"
    )
    assert store.get_lifecycle_epoch_started_at(resurrected) is None, (
        "a resurrected row was given an invented epoch start; unknown must "
        "stay unknown rather than become a usable timestamp"
    )


def test_backfill_does_not_restamp_an_already_recorded_start():
    """It must be idempotent. A row that already has a start keeps it, or every
    bootstrap would slide a live conversation's window forward to created_at
    and undo the bump a resurrect recorded.
    """
    store = _store()
    conv = _conv()
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET phase = 'deleted' WHERE conversation_id = %s",
            (conv,),
        )
    store.increment_lifecycle_epoch_on_resurrect(conv)
    # Force the row back to epoch 1 so it matches the backfill's WHERE clause.
    # Only the IS NULL guard should keep the backfill off it.
    with pg_test_conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET lifecycle_epoch = 1 WHERE conversation_id = %s",
            (conv,),
        )
    recorded = store.get_lifecycle_epoch_started_at(conv)

    _store()  # bootstrap again; the backfill must skip this populated row

    assert store.get_lifecycle_epoch_started_at(conv) == recorded


def test_missing_conversation_raises_rather_than_reporting_unknown():
    """A conversation that does not exist and one whose start is unrecorded are
    different answers and must not collapse into the same None."""
    store = _store()
    with pytest.raises(KeyError):
        store.get_lifecycle_epoch_started_at(_conv())
