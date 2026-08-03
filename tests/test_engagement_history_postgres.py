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
    check_repetition,
    topic_fingerprint,
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


class TestThroughARealStore:
    """Not a double. An actual PostgresStore, built the way production builds it.

    The parameterised fixtures above cover both row factories, but they still
    use a hand-written pool. That is the same class of thing that hid the
    defect: a double configured by the person who also wrote the assertions.
    This constructs the real store, so the pool, its row factory and its
    connection kwargs are whatever PostgresStore chooses — not what a test
    chose on its behalf.
    """

    @pytest.fixture
    def real_store(self):
        from virtual_context.storage.postgres import PostgresStore

        store = PostgresStore(PG_URL)
        apply_engagement_history_schema(store)
        return store

    def test_the_pool_really_does_return_mappings(self, real_store):
        """Pins the production fact the fix depends on.

        If this ever returns tuples, the by-name reads still work — but the
        premise in the fix's comment would be stale, and a stale premise is
        how the next person justifies going back to positional access.
        """
        with real_store.pool.connection() as conn:
            row = conn.execute("SELECT 1 AS n").fetchone()
        assert isinstance(row, dict), (
            f"production pool returned {type(row).__name__}, not a mapping"
        )

    def test_day_is_claimed_answers_both_ways_through_the_real_store(
        self, real_store,
    ):
        """Cloud's post-fix ask, and the exact line that raised KeyError."""
        tenant = f"t-{uuid.uuid4().hex[:12]}"
        history = PostgresPostHistory(
            real_store, tenant_id=tenant,
            conversation_id="sk:agent:vast:discord",
        )
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        day = now.astimezone(EASTERN).date()
        try:
            assert history.day_is_claimed(day) is False, "unclaimed day"
            history.record(_record(posted_at=now))
            assert history.day_is_claimed(day) is True, "claimed day"
            assert already_posted_today(history, now=now) is True
        finally:
            with real_store.pool.connection() as conn:
                conn.execute(
                    "DELETE FROM engagement_post_history WHERE tenant_id = %s",
                    (tenant,),
                )

    def test_a_written_row_reads_back_through_the_real_store(self, real_store):
        """all() and pending_claims() against the real pool, not [] from empty."""
        tenant = f"t-{uuid.uuid4().hex[:12]}"
        history = PostgresPostHistory(
            real_store, tenant_id=tenant,
            conversation_id="sk:agent:vast:discord",
        )
        try:
            history.record(_record(question_text="real", status="pending"))
            rows = history.all()
            assert len(rows) == 1
            assert rows[0].question_text == "real"
            assert [r.question_text for r in pending_claims(history)] == ["real"]
        finally:
            with real_store.pool.connection() as conn:
                conn.execute(
                    "DELETE FROM engagement_post_history WHERE tenant_id = %s",
                    (tenant,),
                )


class TestRepetitionIsEnforcedAgainstTheDurableLedger:
    """The rules exist, are tested, and were never invoked by the runner.

    `check_repetition` was defined, exported, and absent from the run path.
    Every Phase 3 rule was therefore unenforced, and `topic_fingerprint` was
    computed and stored on every post and compared to nothing. A second run
    reproduced a posted question word for word.

    These run against the durable ledger with a real posted row, because the
    in-memory store proved nothing about what the live job would read.
    """

    @pytest.fixture
    def durable_with_a_post(self, durable):
        """One posted question, as production has today."""
        durable.record(_record(
            posted_at=datetime(2026, 8, 3, 18, 4, tzinfo=timezone.utc),
            tagged_actor_id=ACTOR,
            source_message_ids=("1533835931390443521",),
            question_text="Can you paste the rest of the label? It cuts off "
                          "at vitamin B6.",
            topic_fingerprint=topic_fingerprint(
                "Can you paste the rest of the label? It cuts off at "
                "vitamin B6."
            ),
            status="posted",
        ))
        return durable

    def test_the_same_thread_cannot_be_mined_twice(self, durable_with_a_post):
        rejection = check_repetition(
            history=durable_with_a_post,
            now=datetime(2026, 8, 4, 18, 4, tzinfo=timezone.utc),
            actor_id="actor:discord:0000", channel_id="9999",
            source_message_ids=("1533835931390443521",),
            question_text="Something completely different?",
        )
        assert rejection is not None
        assert rejection.reason == "thread_already_used"

    def test_the_same_member_is_not_tagged_again_inside_the_cooldown(
        self, durable_with_a_post,
    ):
        rejection = check_repetition(
            history=durable_with_a_post,
            now=datetime(2026, 8, 4, 18, 4, tzinfo=timezone.utc),
            actor_id=ACTOR, channel_id="9999",
            source_message_ids=("a-different-message",),
            question_text="Something completely different?",
        )
        assert rejection is not None
        assert rejection.reason == "member_recently_tagged"

    def test_the_posted_question_cannot_be_asked_again(
        self, durable_with_a_post,
    ):
        """The exact incident: same question, word for word, next day."""
        rejection = check_repetition(
            history=durable_with_a_post,
            now=datetime(2026, 8, 4, 18, 4, tzinfo=timezone.utc),
            actor_id="actor:discord:0000", channel_id="9999",
            source_message_ids=("a-different-message",),
            question_text="Can you paste the rest of the label? It cuts off "
                          "at vitamin B6.",
        )
        assert rejection is not None
        assert rejection.reason == "question_recently_asked"

    def test_an_unrelated_question_from_a_new_member_passes(
        self, durable_with_a_post,
    ):
        """The rules must not reject everything — that would also 'work'."""
        assert check_repetition(
            history=durable_with_a_post,
            now=datetime(2026, 8, 4, 18, 4, tzinfo=timezone.utc),
            actor_id="actor:discord:0000", channel_id="9999",
            source_message_ids=("a-different-message",),
            question_text="How did the sleep protocol end up going?",
        ) is None

    def test_the_fingerprint_stored_on_a_post_is_what_similarity_reads(
        self, durable_with_a_post,
    ):
        """It was written on every post and compared to nothing."""
        row = durable_with_a_post.all()[0]
        assert row.topic_fingerprint == topic_fingerprint(row.question_text)
        assert row.topic_fingerprint != 0


class TestTheGuardRejectsOneAndLetsAnotherThrough:
    """A guard that rejects the whole pool is indistinguishable from one that
    works: the artifact reads no_question_selected either way.

    This is the re-arm gate. It runs the real runner against the durable
    ledger, with the member who was already posted about and a second member
    who was not, and asserts BOTH halves in a single run — the repeat is
    rejected by name, and the other candidate is the one that gets drafted.
    """

    OTHER_ACTOR = "actor:discord:1485681229608259666"
    OTHER_AUTHOR = "1485681229608259666"
    GUILD = "sk:agent:vast:discord:guild:1524917037191925871"
    P3 = "1524917968440524990"

    def _row(self, *, turn, actor, message_id, text):
        from virtual_context.types import QuoteResult, SourceProvenance

        return QuoteResult(
            text=text, tag="", segment_ref=turn, source_scope="turn",
            matched_side="user",
            provenance=SourceProvenance(
                conversation_id=self.GUILD, canonical_turn_id=turn,
                source_role="requester", actor_id=actor,
                audience_conversation_id=self.GUILD,
                audience_attribution_version=1,
                origin_channel_id=self.P3, source_message_id=message_id,
            ),
        )

    def test_the_repeat_is_rejected_and_the_other_candidate_survives(
        self, durable,
    ):
        import dataclasses

        from virtual_context.core.discord_snowflake import (
            datetime_to_snowflake_floor,
        )
        from virtual_context.core.engagement import (
            Draft, FidelityVerdict, MessageSourceRecord, load_channel_allowlist,
            run_once,
        )

        now = datetime(2026, 8, 4, 18, 4, tzinfo=timezone.utc)
        posted_message = "1533835931390443521"
        # The row production already has: Rob, posted about yesterday.
        durable.record(_record(
            posted_at=now - timedelta(days=1),
            channel_id=self.P3, tagged_actor_id=ACTOR,
            source_message_ids=(posted_message,),
            question_text="Can you paste the rest of the label?",
            topic_fingerprint=topic_fingerprint(
                "Can you paste the rest of the label?"
            ),
            status="posted",
        ))

        sent = now - timedelta(days=4)
        repeat = self._row(turn="ct-rob", actor=ACTOR,
                           message_id=posted_message,
                           text="Rob: Maximus Building Blocks label.")
        other_message = str(datetime_to_snowflake_floor(sent) + 11)
        fresh = self._row(turn="ct-other", actor=self.OTHER_ACTOR,
                          message_id=other_message,
                          text="Roo: Started KPV 500mcg in the mornings.")
        sources = {
            "ct-rob": MessageSourceRecord(
                canonical_turn_id="ct-rob", message_id=posted_message,
                channel_id=self.P3, guild_id="1524917037191925871",
                author_id="1338726888809697364", source_actor_id=ACTOR,
            ),
            "ct-other": MessageSourceRecord(
                canonical_turn_id="ct-other", message_id=other_message,
                channel_id=self.P3, guild_id="1524917037191925871",
                author_id=self.OTHER_AUTHOR,
                source_actor_id=self.OTHER_ACTOR,
            ),
        }
        authors = {posted_message: "1338726888809697364",
                   other_message: self.OTHER_AUTHOR}

        def _fetcher(*, channel_id, message_id):
            return 200, {"channel_id": channel_id, "edited_timestamp": None,
                         "author": {"id": authors[message_id]}}

        def _qualifier(verified, *, now):
            return [dataclasses.replace(c, question_type="timed",
                                        stance="anticipatory")
                    for c in verified], []

        drafted: list = []

        def _drafter(candidate):
            drafted.append(candidate.canonical_turn_id)
            return Draft("How are the mornings treating you?", ""), \
                FidelityVerdict(True)

        result = run_once(
            results=[repeat, fresh], sources=sources,
            senders={"ct-rob": "Rob", "ct-other": "Roo"},
            allowlist=load_channel_allowlist({
                "source_channel_ids": [self.P3],
                "post_channel_ids": [CHANNEL],
            }),
            history=durable, now=now, conversation_id=self.GUILD,
            qualifier=_qualifier, drafter=_drafter, source_fetcher=_fetcher,
        )

        # Half one: the repeat is rejected, by name, against the real ledger.
        history_rejections = {
            r.canonical_turn_id: r.reason
            for r in result.rejections if r.stage == "history"
        }
        assert "ct-rob" in history_rejections, (
            f"the already-posted thread was not rejected: {result.rejections}"
        )
        assert history_rejections["ct-rob"] in {
            "thread_already_used", "member_recently_tagged",
        }, history_rejections

        # Half two: something else still got through. Without this, a guard
        # that rejects everything passes half one and looks correct.
        assert drafted == ["ct-other"], (
            f"the surviving candidate was not drafted: {drafted}"
        )
        assert result.report.question == "How are the mornings treating you?"


class TestARecordCanBeAddressed:
    """A record read back must be usable with update().

    Filed as cosmetic and it wasn't: pending_claims() reported rows "for a
    person to resolve" while update() took a handle the record didn't carry,
    so the only method that could fix them could not be told which one. The
    approval loop needs the same round trip — read a staged row, update it.
    """

    def test_the_record_carries_the_handle_record_returned(self, history):
        handle = history.record(_record(status="pending"))
        assert history.all()[0].id == handle

    def test_a_record_read_back_can_be_updated_by_its_own_id(self, history):
        """The exact round trip the approval loop performs."""
        history.record(_record(status="pending", question_text="staged"))
        row = history.all()[0]
        history.update(row.id, status="posted", discord_message_id="123")
        after = history.all()[0]
        assert after.status == "posted"
        assert after.discord_message_id == "123"
        assert after.id == row.id

    def test_pending_claims_returns_addressable_rows(self, history):
        """The operator surface, now actually operable."""
        history.record(_record(status="pending"))
        claims = pending_claims(history)
        assert claims and claims[0].id
        history.update(claims[0].id, status="posted")
        assert pending_claims(history) == []

    def test_ids_are_distinct_per_row(self, history):
        first = history.record(_record(question_text="one"))
        second = history.record(_record(
            question_text="two",
            posted_at=datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc),
        ))
        ids = {r.id for r in history.all()}
        assert ids == {first, second}
        assert len(ids) == 2

    def test_both_backends_agree_on_the_field(self, history):
        """Neither backend may leave it empty on a row read back."""
        history.record(_record())
        assert history.all()[0].id != ""


class TestApprovalIsExactlyOnce:
    """Two pollers seeing one approval must not both publish.

    Tested with two callers rather than one call twice, because the failure
    needs both to read the row before either writes — the same shape as the
    schema-executor defect, which one sequential call could never show.
    """

    def test_only_one_of_two_callers_wins(self, durable):
        handle = durable.record(_record(status="staged"))
        first = durable.claim_for_publish(handle)
        second = durable.claim_for_publish(handle)
        assert [first, second] == [True, False]
        assert durable.all()[0].status == "approved"

    def test_concurrent_callers_still_yield_one_winner(self, pg_store, durable):
        """Real concurrency, separate connections, no sequencing."""
        import threading

        handle = durable.record(_record(status="staged"))
        wins: list = []
        barrier = threading.Barrier(4, timeout=10)

        def attempt():
            history = PostgresPostHistory(
                pg_store, tenant_id=durable._tenant_id,
                conversation_id="sk:agent:vast:discord",
            )
            barrier.wait()
            wins.append(history.claim_for_publish(handle))

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert wins.count(True) == 1, f"{wins.count(True)} callers would publish"

    def test_an_approved_row_cannot_be_claimed_again(self, durable):
        handle = durable.record(_record(status="staged"))
        durable.claim_for_publish(handle)
        assert durable.claim_for_publish(handle) is False


class TestDecliningIsAtomicAndIdempotent:
    """Declining frees the day in the same statement that declines.

    The orderings are not symmetric. Declining without releasing costs a day.
    Releasing without declining leaves the row reading `staged` with the day
    free, so the next run stages a second question while the first still
    awaits approval — two staged messages, either publishable.
    """

    def test_declining_frees_the_day(self, history):
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        handle = history.record(_record(posted_at=now, status="staged"))
        assert already_posted_today(history, now=now) is True
        assert history.decline(handle) is True
        assert already_posted_today(history, now=now) is False

    def test_the_declined_row_is_kept_for_audit(self, history):
        handle = history.record(_record(
            status="staged", question_text="rejected question",
        ))
        history.decline(handle)
        row = history.all()[0]
        assert row.status == "declined"
        assert row.question_text == "rejected question"
        assert row.source_message_ids == ("1524917968440524991",)

    def test_a_second_decline_is_a_no_op_not_an_error(self, history):
        """A poller can see the same reply twice."""
        handle = history.record(_record(status="staged"))
        assert history.decline(handle) is True
        assert history.decline(handle) is False

    def test_a_published_row_cannot_be_declined(self, history):
        """A late decline must not release the day or imply an unpublish."""
        handle = history.record(_record(
            status="posted", discord_message_id="1533835931390443521",
        ))
        with pytest.raises(ValueError, match="only a staged row"):
            history.decline(handle)
        assert history.all()[0].status == "posted"

    def test_a_freed_day_can_be_claimed_again(self, history):
        now = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)
        first = history.record(_record(posted_at=now, status="staged"))
        history.decline(first)
        history.record(_record(posted_at=now, question_text="second attempt"))
        assert already_posted_today(history, now=now) is True

    def test_the_two_message_ids_are_separate_fields(self, history):
        handle = history.record(_record(
            status="staged", staged_message_id="1111111111111111111",
        ))
        history.update(handle, discord_message_id="2222222222222222222",
                       status="posted")
        row = history.all()[0]
        assert row.staged_message_id == "1111111111111111111"
        assert row.discord_message_id == "2222222222222222222"
