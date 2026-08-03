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
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tests.pg_helpers import pg_dsn
from virtual_context.core.engagement import (
    DayAlreadyClaimed,
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
    """A pool shape matching what the storage backends expose.

    ``row_factory`` is a constructor argument because it is exactly what this
    test got wrong. Production builds its pool with ``dict_row``; this pool
    defaulted to psycopg's tuple rows, so every assertion passed against a
    connection configured differently from the one production hands the
    class — and the class could not read a single row in production.
    """

    def __init__(self, dsn, row_factory=None):
        self._dsn = dsn
        self._row_factory = row_factory

    def connection(self):
        import psycopg

        kwargs = {"autocommit": True}
        if self._row_factory is not None:
            kwargs["row_factory"] = self._row_factory
        return psycopg.connect(self._dsn, **kwargs)


class _Store:
    def __init__(self, dsn, row_factory=None):
        self.pool = _Pool(dsn, row_factory)


def _row_factories():
    """Both shapes a driver can hand back. Production uses dict_row."""
    from psycopg.rows import dict_row, tuple_row

    return [("dict_row", dict_row), ("tuple_row", tuple_row)]


@pytest.fixture(
    scope="module",
    params=_row_factories() if PG_URL else [],
    ids=lambda p: p[0],
)
def pg_store(request):
    """Every durable test runs under BOTH row factories.

    Parameterised rather than fixed, because a fix verified on the wrong
    factory passes and changes nothing — which is how the positional access
    survived a green suite.
    """
    store = _Store(PG_URL, request.param[1])
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
        second = history.record(_record(
            question_text="two",
            posted_at=datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc),
        ))
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
        history.record(_record(
            question_text="theirs", status="pending",
            posted_at=datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc),
        ))
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
        history.record(_record(
            question_text="stuck", status="pending",
            posted_at=datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc),
        ))
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


# ------------------------------------------------- the day is a constraint


class TestTheDayClaimIsEnforcedByTheDatabase:
    """A read-then-write cannot hold a claim under two processes.

    ``already_posted_today`` checks, then ``record`` writes. Between those
    two, another process can do the same. The unique index is what makes the
    second one impossible rather than unlikely.
    """

    def test_a_second_claim_on_the_same_day_is_refused(self, history):
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        history.record(_record(posted_at=now))
        with pytest.raises(DayAlreadyClaimed):
            history.record(_record(posted_at=now, question_text="second"))

    def test_the_refusal_does_not_depend_on_the_backend(self, history):
        """Both raise the same domain error, not a driver exception."""
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        history.record(_record(posted_at=now))
        try:
            history.record(_record(posted_at=now))
        except DayAlreadyClaimed as exc:
            assert "2026-08-02" in str(exc)
        else:
            pytest.fail("expected DayAlreadyClaimed")

    def test_a_claim_that_loses_the_race_becomes_a_refusal_not_a_crash(
        self, history, monkeypatch,
    ):
        """The reason the constraint and the handler ship together.

        A constraint whose violation escapes as a database error turns a
        cleanly skipped day into a failed unit — the state that invites
        someone to fix it by restarting, which is how a double post happens.
        """
        import virtual_context.core.engagement.poster as poster_module

        monkeypatch.setattr(poster_module, "POSTING_ENABLED", True)
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        history.record(_record(posted_at=now))

        candidate = SimpleNamespace(
            actor_id=ACTOR, source_message_id="1524917968440524991",
            channel_id=CHANNEL,
        )
        verification = SimpleNamespace(
            ok=True, verified_message_id="1524917968440524991",
        )
        # Bypass the read-then-write check so the constraint is what refuses.
        monkeypatch.setattr(poster_module, "already_posted_today",
                            lambda *a, **k: False)
        with pytest.raises(poster_module.PostRefused, match="already claimed"):
            poster_module.post_question(
                candidate=candidate, question="Did the ss31 run go well?",
                channel_id=CHANNEL, verification=verification,
                history=history, sender=lambda **kw: "999", now=now,
                question_type="timed",
            )

    def test_the_next_day_is_still_claimable(self, history):
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        history.record(_record(posted_at=now))
        history.record(_record(posted_at=now + timedelta(days=1)))
        assert len(history.all()) == 2

    def test_the_stored_day_matches_the_sql_backfill_expression(
        self, durable, pg_store,
    ):
        """Python writes the column; SQL backfills it. They must agree.

        A disagreement would put a row's claim on one day and the migration's
        recomputation on another, so the constraint would guard a different
        day than the one the code checks.
        """
        for hour in (0, 3, 4, 5, 12, 23):
            moment = datetime(2026, 8, 3, hour, 30, tzinfo=timezone.utc)
            durable.record(_record(posted_at=moment, question_text=f"h{hour}"))
            with pg_store.pool.connection() as conn:
                row = conn.execute(
                    """SELECT eastern_day,
                              (posted_at::timestamptz
                               AT TIME ZONE 'America/New_York')::date AS sql_day
                         FROM engagement_post_history
                        WHERE tenant_id = %s AND question_text = %s""",
                    (durable._tenant_id, f"h{hour}"),
                ).fetchone()
            stored, sql_day = (
                (row["eastern_day"], row["sql_day"])
                if isinstance(row, dict) else (row[0], row[1])
            )
            assert stored == sql_day, f"disagreed at {hour}:30 UTC"
            with pg_store.pool.connection() as conn:
                conn.execute(
                    "DELETE FROM engagement_post_history WHERE tenant_id = %s",
                    (durable._tenant_id,),
                )

    def test_one_tenants_claim_does_not_block_another(self, pg_store, durable):
        """The index is on (tenant_id, eastern_day), not the day alone."""
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        durable.record(_record(posted_at=now))
        other = PostgresPostHistory(
            pg_store, tenant_id=f"t-{uuid.uuid4().hex[:12]}",
            conversation_id="sk:agent:vast:discord",
        )
        other.record(_record(posted_at=now))
        assert len(other.all()) == 1


class TestAnEmptyResultIsNotEvidence:
    """`[]` from an empty table looks exactly like a working reader.

    Every read here returned `[]` in production and the suite was green,
    because the table had no rows and the tests used a connection whose row
    factory did not match production's. These assert a row goes in, comes
    back, and carries its values — so an empty list can never again be
    mistaken for a working read.
    """

    def test_all_returns_the_record_with_its_fields(self, durable):
        durable.record(_record(
            question_text="How did the ss31 run go?",
            tagged_actor_id=ACTOR,
            source_message_ids=("1524917968440524991",),
            topic_fingerprint=2**64 - 1,
            status="posted",
        ))
        rows = durable.all()
        assert len(rows) == 1, "a written row did not come back"
        row = rows[0]
        assert row.question_text == "How did the ss31 run go?"
        assert row.tagged_actor_id == ACTOR
        assert row.source_message_ids == ("1524917968440524991",)
        assert row.topic_fingerprint == 2**64 - 1
        assert row.status == "posted"
        assert row.posted_at.tzinfo is not None, "timestamp lost its zone"

    def test_pending_claims_returns_the_record_not_an_empty_list(self, durable):
        durable.record(_record(status="pending", question_text="stuck"))
        claims = pending_claims(durable)
        assert len(claims) == 1, "a pending claim was invisible"
        assert claims[0].question_text == "stuck"
        assert claims[0].status == "pending"

    def test_day_is_claimed_answers_true_for_a_written_row(self, durable):
        """The read that was raising KeyError on every single run."""
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        assert durable.day_is_claimed(now.astimezone(EASTERN).date()) is False
        durable.record(_record(posted_at=now))
        assert durable.day_is_claimed(now.astimezone(EASTERN).date()) is True

    def test_the_guard_actually_blocks_a_second_post_end_to_end(
        self, durable, monkeypatch,
    ):
        """The property the whole ledger exists for.

        This is the assertion that would have failed in production: the
        duplicate guard was not degraded, it was inoperative, and only the
        ordering of the checks kept it from posting twice.
        """
        import virtual_context.core.engagement.poster as poster_module

        monkeypatch.setattr(poster_module, "POSTING_ENABLED", True)
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        candidate = SimpleNamespace(
            actor_id=ACTOR, source_message_id="1524917968440524991",
            channel_id=CHANNEL,
        )
        verification = SimpleNamespace(
            ok=True, verified_message_id="1524917968440524991",
        )
        kwargs = dict(
            candidate=candidate, question="Did the ss31 run go well?",
            channel_id=CHANNEL, verification=verification, history=durable,
            sender=lambda **kw: "9001", now=now, question_type="timed",
        )
        first = poster_module.post_question(**kwargs)
        assert first.message_id == "9001"
        with pytest.raises(poster_module.PostRefused):
            poster_module.post_question(**kwargs)
        assert len(durable.all()) == 1, "the day was claimed twice"
