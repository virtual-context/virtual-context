"""Durable posting history and the repetition checks it powers.

The corpus is not a rolling window. History begins 2026-07-10 and every
channel's thirty-day count equals its total, so widening the time window
cannot refresh the pool — the job draws from the whole corpus every day.
That makes repetition control the primary defence against looking
repetitive, and it is thinnest at launch, when there is least history to
draw from and least history to check against.

The record stores what prevents a repeat and nothing that builds a picture
of a member. Topic similarity is carried as a fixed-width fingerprint rather
than text: it answers "have I asked something like this" without keeping a
second copy of what anyone said.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.core.engagement import (
    InMemoryPostHistory,
    PostRecord,
    check_repetition,
    topic_fingerprint,
)

# Assert against the SHIPPED thresholds, never a number retyped here. A test
# carrying its own ruler stays green while the rule it claims to pin moves
# out from under it.
from virtual_context.core.engagement.history import (
    CHANNEL_MAX_IN_WINDOW,
    CHANNEL_WINDOW,
    MEMBER_COOLDOWN,
    SIMILARITY_DISTANCE,
)

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
P3 = "1524917968440524990"
RMS = "1530567788949798963"
BIGTEX = "actor:discord:1338726888809697364"
ROO = "actor:discord:1485681229608259666"


def _record(**kw) -> PostRecord:
    base = dict(
        posted_at=NOW - timedelta(days=1),
        channel_id=P3,
        question_type="personal",
        tagged_actor_id=BIGTEX,
        source_message_ids=("1532400954878595094",),
        topic_fingerprint=topic_fingerprint("have you started the SS-31 yet"),
        question_text="Have you started the SS-31 yet?",
        discord_message_id="9990001",
        resolution="",
    )
    base.update(kw)
    return PostRecord(**base)


class TestTopicFingerprint:
    def test_it_is_stable_for_the_same_text(self):
        assert topic_fingerprint("how did the four weeks go") == (
            topic_fingerprint("how did the four weeks go")
        )

    def test_a_reworded_question_is_similar_by_the_shipped_rule(self):
        from virtual_context.core.engagement import fingerprint_distance

        a = topic_fingerprint("have you started the SS-31 yet")
        b = topic_fingerprint("did you start the SS-31 yet")
        distance = fingerprint_distance(a, b)
        # The bound is the shipped threshold, so tightening it below this
        # pair's distance fails here instead of silently letting a reworded
        # repeat through.
        assert distance <= SIMILARITY_DISTANCE, (
            f"reworded question at distance {distance} is no longer caught "
            f"by SIMILARITY_DISTANCE={SIMILARITY_DISTANCE}"
        )

    def test_a_different_topic_is_not_similar_by_the_shipped_rule(self):
        from virtual_context.core.engagement import fingerprint_distance

        a = topic_fingerprint("have you started the SS-31 yet")
        b = topic_fingerprint("which marker changed your entire protocol")
        distance = fingerprint_distance(a, b)
        assert distance > SIMILARITY_DISTANCE, (
            f"unrelated topics at distance {distance} would be suppressed as "
            f"duplicates by SIMILARITY_DISTANCE={SIMILARITY_DISTANCE}"
        )

    def test_it_keeps_no_recoverable_text(self):
        """A fingerprint answers similarity without storing what was said."""
        fp = topic_fingerprint("BigTex is running SS-31 at 5mg for four weeks")
        assert isinstance(fp, int)
        assert 0 <= fp < (1 << 64)
        assert "SS-31" not in str(fp)


class TestRecordCarriesOnlyWhatPreventsRepeats:
    def test_the_record_has_no_member_content_or_display_name(self):
        record = _record()
        fields = set(record.__dataclass_fields__)
        for forbidden in (
            "member_text", "quote", "handle", "display_name", "sender",
            "author_name", "body",
        ):
            assert forbidden not in fields, (
                f"{forbidden} would make this a dossier rather than a "
                "repetition ledger"
            )

    def test_identity_is_kept_as_an_actor_id_only(self):
        assert _record().tagged_actor_id.startswith("actor:")


class TestRepetitionChecks:
    def _history(self, *records) -> InMemoryPostHistory:
        history = InMemoryPostHistory()
        for record in records:
            history.record(record)
        return history

    def test_no_history_rejects_nothing(self):
        rejection = check_repetition(
            history=InMemoryPostHistory(), now=NOW,
            actor_id=BIGTEX, channel_id=P3,
            source_message_ids=("1532400954878595094",),
            question_text="Have you started the SS-31 yet?",
        )
        assert rejection is None

    def test_a_thread_already_drawn_from_is_rejected(self):
        history = self._history(_record())
        rejection = check_repetition(
            history=history, now=NOW, actor_id=BIGTEX, channel_id=RMS,
            source_message_ids=("1532400954878595094",),
            question_text="Something completely different about labs",
        )
        assert rejection is not None
        assert rejection.reason == "thread_already_used"

    def test_a_recently_tagged_member_is_rejected(self):
        history = self._history(_record(source_message_ids=("other",)))
        rejection = check_repetition(
            history=history, now=NOW, actor_id=BIGTEX, channel_id=RMS,
            source_message_ids=("1111",),
            question_text="Something completely different about labs",
        )
        assert rejection is not None
        assert rejection.reason == "member_recently_tagged"

    def test_the_same_member_is_available_again_after_the_cooldown(self):
        history = self._history(
            _record(posted_at=NOW - MEMBER_COOLDOWN - timedelta(days=1),
                    source_message_ids=("other",)),
        )
        rejection = check_repetition(
            history=history, now=NOW, actor_id=BIGTEX, channel_id=RMS,
            source_message_ids=("1111",),
            question_text="Something completely different about labs",
        )
        assert rejection is None

    def test_a_near_duplicate_question_is_rejected(self):
        history = self._history(
            _record(tagged_actor_id=ROO, source_message_ids=("other",)),
        )
        rejection = check_repetition(
            history=history, now=NOW, actor_id=BIGTEX, channel_id=RMS,
            source_message_ids=("1111",),
            question_text="Have you started the SS-31 yet?",
        )
        assert rejection is not None
        assert rejection.reason == "question_recently_asked"

    def test_an_over_used_channel_is_rejected(self):
        # Exactly the shipped limit, derived rather than retyped.
        history = self._history(*[
            _record(posted_at=NOW - timedelta(hours=6 * (d + 1)),
                    channel_id=P3,
                    tagged_actor_id=f"actor:discord:{d}",
                    source_message_ids=(f"m{d}",),
                    topic_fingerprint=topic_fingerprint(f"unrelated topic {d}"))
            for d in range(CHANNEL_MAX_IN_WINDOW)
        ])
        rejection = check_repetition(
            history=history, now=NOW, actor_id=BIGTEX, channel_id=P3,
            source_message_ids=("1111",),
            question_text="An entirely unrelated question about sleep",
        )
        assert rejection is not None
        assert rejection.reason == "channel_recently_overused"

    def test_a_thread_the_member_ignored_is_not_reopened(self):
        history = self._history(
            _record(posted_at=NOW - timedelta(days=20),
                    resolution="ignored"),
        )
        rejection = check_repetition(
            history=history, now=NOW, actor_id=BIGTEX, channel_id=RMS,
            source_message_ids=("1532400954878595094",),
            question_text="Have you started the SS-31 yet?",
        )
        assert rejection is not None
        assert rejection.reason in (
            "thread_already_used", "thread_previously_ignored",
        )

    def test_every_rejection_names_the_history_stage(self):
        history = self._history(_record())
        rejection = check_repetition(
            history=history, now=NOW, actor_id=BIGTEX, channel_id=RMS,
            source_message_ids=("1532400954878595094",),
            question_text="x",
        )
        assert rejection.stage == "history"


class TestSchemaIsDesignedNotMigrated:
    def test_the_ddl_is_available_for_review(self):
        from virtual_context.core.engagement import ENGAGEMENT_HISTORY_DDL

        ddl = ENGAGEMENT_HISTORY_DDL.lower()
        assert "create table" in ddl
        for column in (
            "posted_at", "channel_id", "question_type", "tagged_actor_id",
            "source_message_ids", "topic_fingerprint", "question_text",
            "discord_message_id", "resolution",
        ):
            assert column in ddl

    def test_the_ddl_stores_no_member_content(self):
        from virtual_context.core.engagement import ENGAGEMENT_HISTORY_DDL

        ddl = ENGAGEMENT_HISTORY_DDL.lower()
        for forbidden in ("member_text", "quote", "handle", "display_name"):
            assert forbidden not in ddl


class TestThresholdsHaveMargin:
    """A rule that only just holds is one wording change from not holding."""

    def test_the_reworded_pair_sits_at_the_shipped_boundary(self):
        from virtual_context.core.engagement import fingerprint_distance

        distance = fingerprint_distance(
            topic_fingerprint("have you started the SS-31 yet"),
            topic_fingerprint("did you start the SS-31 yet"),
        )
        # Recorded, not asserted as good: the only worked example we have
        # sits exactly ON the threshold, so the near-duplicate rule has zero
        # margin here. Documented so a future tuning pass starts from a
        # measurement rather than from the constant's round number.
        assert distance == SIMILARITY_DISTANCE

    def test_the_channel_window_and_limit_are_consistent(self):
        assert CHANNEL_MAX_IN_WINDOW >= 1
        assert CHANNEL_WINDOW.days >= 1


class TestTheFingerprintColumnHoldsTheWholeRange:
    """The simhash is unsigned 64-bit; a signed column loses half of it."""

    def test_fingerprints_exceed_signed_64_bit(self):
        """Non-vacuous: prove the range actually goes above the ceiling."""
        over = [
            s for s in (
                "have you started the SS-31 yet",
                "did you add the KPV to the pm dose",
                "how did the four weeks go",
                "which marker changed your protocol",
                "what tells you that you are recovered",
                "rate my stack question",
                "is the am dose still 500mcg",
                "what would make you drop the dose",
            ) if topic_fingerprint(s) > 2**63 - 1
        ]
        assert over, "no sample exceeded the signed ceiling; sample is unfit"

    def test_every_fingerprint_fits_the_declared_column(self):
        from virtual_context.core.engagement import ENGAGEMENT_HISTORY_DDL

        assert "NUMERIC(20,0)" in ENGAGEMENT_HISTORY_DDL
        assert "topic_fingerprint   BIGINT" not in ENGAGEMENT_HISTORY_DDL
        # NUMERIC(20,0) spans 20 digits; 2^64-1 has 20.
        assert len(str(2**64 - 1)) == 20

    def test_the_ddl_says_why_the_type_is_what_it_is(self):
        from virtual_context.core.engagement import ENGAGEMENT_HISTORY_DDL

        assert "unsigned 64-bit" in ENGAGEMENT_HISTORY_DDL
        assert "2^63-1" in ENGAGEMENT_HISTORY_DDL

    def test_distance_still_works_across_the_boundary(self):
        from virtual_context.core.engagement import fingerprint_distance

        low = topic_fingerprint("which marker changed your protocol")
        high = topic_fingerprint("have you started the SS-31 yet")
        assert high > 2**63 - 1 > low
        assert 0 <= fingerprint_distance(low, high) <= 64


class TestTheSchemaExecutor:
    """The applied text and the asserted text must be the same object."""

    class _FakeConn:
        def __init__(self, log):
            self.log = log

        def execute(self, sql, params=None):
            self.log.append(sql)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakePool:
        def __init__(self, log):
            self.log = log

        def connection(self):
            return TestTheSchemaExecutor._FakeConn(self.log)

    class _FakeStore:
        def __init__(self, log):
            self.pool = TestTheSchemaExecutor._FakePool(log)

    def test_it_executes_the_shipped_constant_not_a_copy(self):
        from virtual_context.core.engagement import (
            ENGAGEMENT_HISTORY_DDL, apply_engagement_history_schema,
        )

        log: list[str] = []
        returned = apply_engagement_history_schema(self._FakeStore(log))
        assert returned is ENGAGEMENT_HISTORY_DDL
        joined = " ".join(log)
        assert "engagement_post_history" in joined
        assert "NUMERIC(20,0)" in joined
        assert "status" in joined

    def test_the_script_is_serialized_by_an_advisory_lock(self):
        """IF NOT EXISTS is idempotent over time, not over concurrency.

        Two sessions can both find the table absent and both create it;
        Postgres fails the loser in its catalogue rather than treating it as
        the no-op the clause implies. Measured: 8 of 12 concurrent applies
        failed without the lock, 0 with it.
        """
        from virtual_context.core.engagement import apply_engagement_history_schema
        from virtual_context.core.engagement.history import (
            ENGAGEMENT_SCHEMA_LOCK,
        )

        log: list[str] = []
        apply_engagement_history_schema(self._FakeStore(log))
        assert "pg_advisory_lock" in log[0]
        assert "pg_advisory_unlock" in log[-1]
        assert isinstance(ENGAGEMENT_SCHEMA_LOCK, int)
        # Must fit a signed bigint, which is what pg_advisory_lock takes.
        assert -(2**63) <= ENGAGEMENT_SCHEMA_LOCK < 2**63

    def test_the_lock_is_released_even_if_the_ddl_fails(self):
        from virtual_context.core.engagement import apply_engagement_history_schema

        class _Boom(TestTheSchemaExecutor._FakeConn):
            def execute(self, sql, params=None):
                self.log.append(sql)
                if "CREATE" in sql:
                    raise RuntimeError("ddl exploded")
                return self

        log: list[str] = []
        store = TestTheSchemaExecutor._FakeStore(log)
        store.pool.connection = lambda: _Boom(log)
        with pytest.raises(RuntimeError, match="ddl exploded"):
            apply_engagement_history_schema(store)
        assert "pg_advisory_unlock" in log[-1], "lock leaked on failure"

    def test_the_whole_script_is_sent_in_one_call(self):
        """Splitting on semicolons cuts the table in half.

        The explanatory comment contains a semicolon, so a naive split sends
        a truncated CREATE TABLE — automated-looking and broken, which is
        worse than the hand-application this replaces.
        """
        from virtual_context.core.engagement import apply_engagement_history_schema

        log: list[str] = []
        apply_engagement_history_schema(self._FakeStore(log))
        ddl = [s for s in log if "CREATE" in s]
        assert len(ddl) == 1, "the script was split"
        sent = ddl[0]
        assert sent.count("CREATE TABLE") == 1
        assert sent.count("CREATE INDEX") == 2
        assert sent.count("IF NOT EXISTS") == 3

    def test_a_naive_split_would_have_truncated_the_table(self):
        """Pins the hazard so nobody reintroduces the obvious version."""
        from virtual_context.core.engagement import ENGAGEMENT_HISTORY_DDL

        naive = [p for p in ENGAGEMENT_HISTORY_DDL.split(";") if p.strip()]
        assert len(naive) == 4, (
            "the comment semicolon is gone; if the DDL no longer contains "
            "one, this hazard has changed and the guard should be revisited"
        )
        assert "status" not in naive[0], (
            "a naive split drops the final column from the CREATE TABLE"
        )

    def test_a_store_without_a_pool_refuses(self):
        from virtual_context.core.engagement import apply_engagement_history_schema

        class _NoPool:
            pass

        with pytest.raises(RuntimeError, match="no connection pool"):
            apply_engagement_history_schema(_NoPool())
