"""The durable post history, against a real Postgres.

Every scenario here runs against BOTH backends through the same body. That
is the point of the file: the in-memory reference exists so tests are cheap,
and it is only worth having if it cannot quietly disagree with the durable
one. A behaviour asserted against the reference alone is asserted against a
mock of the thing that matters.

SQLite is deliberately absent. This table's two hazards — an unsigned 64-bit
fingerprint and a civil-day comparison over a stored timestamp — are both
places SQLite is more permissive than Postgres, so a green SQLite run would
mean strictly less than it appears to. The fingerprint case is not
hypothetical: BIGINT was measured rejecting values SQLite accepted.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from tests.pg_helpers import pg_dsn
from virtual_context.core.engagement import (
    InMemoryPostHistory,
    PostgresPostHistory,
    PostRecord,
    apply_engagement_history_schema,
)
from virtual_context.core.engagement.poster import (
    POSTING_ZONE,
    already_posted_today,
    pending_claims,
)

PG_URL = pg_dsn()
pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)

EASTERN = ZoneInfo(POSTING_ZONE)
CHANNEL = "1524946242499514418"
ACTOR = "actor:discord:1338726888809697364"


def _record(**kw) -> PostRecord:
    base = dict(
        posted_at=datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc),
        channel_id=CHANNEL,
        question_type="personal",
        tagged_actor_id=ACTOR,
        source_message_ids=("1524917968440524991",),
        topic_fingerprint=12345,
        question_text="How did the ss31 run go?",
        status="posted",
    )
    base.update(kw)
    return PostRecord(**base)


class _Pool:
    """A pool shape matching what the storage backends expose."""

    def __init__(self, dsn):
        self._dsn = dsn

    def connection(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)


class _Store:
    def __init__(self, dsn):
        self.pool = _Pool(dsn)


@pytest.fixture(scope="module")
def pg_store():
    store = _Store(PG_URL)
    apply_engagement_history_schema(store)
    return store


@pytest.fixture
def durable(pg_store):
    """A fresh tenant per test, so no test can see another's rows."""
    tenant = f"t-{uuid.uuid4().hex[:12]}"
    history = PostgresPostHistory(
        pg_store, tenant_id=tenant, conversation_id="sk:agent:vast:discord",
    )
    yield history
    with pg_store.pool.connection() as conn:
        conn.execute(
            "DELETE FROM engagement_post_history WHERE tenant_id = %s",
            (tenant,),
        )


@pytest.fixture(params=["memory", "postgres"])
def history(request, durable):
    """Both backends, same scenarios. The whole reason this file exists."""
    return InMemoryPostHistory() if request.param == "memory" else durable


# ------------------------------------------------------- same-interface


class TestBothBackendsAgree:
    def test_record_returns_a_handle_that_addresses_that_row(self, history):
        first = history.record(_record(question_text="one"))
        second = history.record(_record(question_text="two"))
        assert first != second
        history.update(first, resolution="answered")
        by_text = {r.question_text: r.resolution for r in history.all()}
        assert by_text == {"one": "answered", "two": ""}

    def test_a_handle_is_not_a_position(self, history):
        """The defect this interface replaced.

        Under the old index-based form, ``len(all()) - 1`` after two claims
        addressed the second row, so a process confirming its own earlier
        claim silently confirmed someone else's.
        """
        first = history.record(_record(question_text="mine", status="pending"))
        history.record(_record(question_text="theirs", status="pending"))
        history.update(first, status="posted", discord_message_id="111")
        states = {r.question_text: (r.status, r.discord_message_id)
                  for r in history.all()}
        assert states["mine"] == ("posted", "111")
        assert states["theirs"] == ("pending", "")

    def test_an_unknown_handle_raises_rather_than_writing(self, history):
        history.record(_record())
        with pytest.raises(KeyError):
            history.update("nope-not-a-handle", status="posted")
        assert [r.status for r in history.all()] == ["posted"]

    def test_the_full_unsigned_fingerprint_survives_a_round_trip(self, history):
        """2^64-1 is a legal simhash. BIGINT would have rejected it."""
        top = 2**64 - 1
        history.record(_record(topic_fingerprint=top))
        assert history.all()[0].topic_fingerprint == top

    def test_source_message_ids_round_trip_as_a_tuple(self, history):
        history.record(_record(source_message_ids=("111", "222")))
        assert history.all()[0].source_message_ids == ("111", "222")

    def test_empty_source_message_ids_round_trip(self, history):
        history.record(_record(source_message_ids=()))
        assert history.all()[0].source_message_ids == ()

    def test_pending_claims_sees_the_unconfirmed_row(self, history):
        history.record(_record(question_text="done", status="posted"))
        history.record(_record(question_text="stuck", status="pending"))
        assert [r.question_text for r in pending_claims(history)] == ["stuck"]

    def test_a_confirmed_claim_leaves_pending(self, history):
        handle = history.record(_record(status="pending"))
        history.update(handle, status="posted", discord_message_id="99")
        assert pending_claims(history) == []

    def test_since_filters_by_moment(self, history):
        old = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
        history.record(_record(posted_at=old, question_text="old"))
        history.record(_record(question_text="new"))
        cut = datetime(2026, 8, 1, tzinfo=timezone.utc)
        assert [r.question_text for r in history.since(cut)] == ["new"]


# ------------------------------------------------------- the day claim


class TestTheDayClaim:
    def test_a_claim_takes_the_day(self, history):
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        assert not already_posted_today(history, now=now)
        history.record(_record(posted_at=now))
        assert already_posted_today(history, now=now)

    def test_a_pending_claim_also_takes_the_day(self, history):
        """'We cannot tell whether it sent' must read as spent, not free."""
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        history.record(_record(posted_at=now, status="pending"))
        assert already_posted_today(history, now=now)

    def test_yesterdays_claim_does_not_take_today(self, history):
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        history.record(_record(posted_at=now - timedelta(days=1)))
        assert not already_posted_today(history, now=now)

    def test_the_day_is_eastern_not_utc(self, history):
        """The boundary that a UTC-day implementation gets wrong.

        03:00 UTC on the 3rd is 23:00 Eastern on the 2nd. A post then and a
        post at 16:00 UTC on the 2nd are the same Eastern day, and treating
        them as two days would allow two posts.
        """
        eastern_evening = datetime(2026, 8, 3, 3, 0, tzinfo=timezone.utc)
        assert eastern_evening.astimezone(EASTERN).date().day == 2
        history.record(_record(posted_at=eastern_evening))
        same_day = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        assert already_posted_today(history, now=same_day)

    def test_a_utc_day_rollover_that_is_not_an_eastern_one(self, history):
        """01:00 UTC on the 3rd is still the 2nd in Eastern."""
        history.record(_record(
            posted_at=datetime(2026, 8, 2, 20, 0, tzinfo=timezone.utc),
        ))
        just_after_utc_midnight = datetime(2026, 8, 3, 1, 0, tzinfo=timezone.utc)
        assert already_posted_today(history, now=just_after_utc_midnight)


# --------------------------------------------- durable-only properties


class TestOnlyTheDurableBackendCanBeAskedThis:
    def test_a_second_connection_sees_the_claim_immediately(self, durable, pg_store):
        """The property in-memory cannot have, and the job depends on."""
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        durable.record(_record(posted_at=now, status="pending"))
        other_process = PostgresPostHistory(
            pg_store,
            tenant_id=durable._tenant_id,
            conversation_id="sk:agent:vast:discord",
        )
        assert already_posted_today(other_process, now=now)

    def test_the_day_query_does_not_load_the_table(self, durable, monkeypatch):
        """`already_posted_today` must not become a scan as rows accumulate."""
        durable.record(_record())

        def _boom():
            raise AssertionError("day check loaded every row")

        monkeypatch.setattr(durable, "all", _boom)
        already_posted_today(
            durable, now=datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc),
        )

    def test_rows_are_scoped_to_the_tenant(self, pg_store, durable):
        durable.record(_record(question_text="mine"))
        stranger = PostgresPostHistory(
            pg_store, tenant_id=f"t-{uuid.uuid4().hex[:12]}",
            conversation_id="sk:agent:vast:discord",
        )
        assert stranger.all() == []
        assert already_posted_today(
            stranger, now=datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc),
        ) is False

    def test_another_tenants_handle_cannot_be_updated(self, pg_store, durable):
        handle = durable.record(_record())
        stranger = PostgresPostHistory(
            pg_store, tenant_id=f"t-{uuid.uuid4().hex[:12]}",
            conversation_id="sk:agent:vast:discord",
        )
        with pytest.raises(KeyError):
            stranger.update(handle, status="posted")
        assert durable.all()[0].status == "posted"

    def test_only_named_columns_are_updatable(self, durable):
        handle = durable.record(_record())
        with pytest.raises(ValueError, match="not updatable"):
            durable.update(handle, tenant_id="somebody-else")

    def test_a_store_without_a_pool_refuses(self):
        with pytest.raises(RuntimeError, match="no connection pool"):
            PostgresPostHistory(object(), tenant_id="t", conversation_id="c")
