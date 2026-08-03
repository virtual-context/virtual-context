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

    def test_wording_changes_stay_close(self):
        from virtual_context.core.engagement import fingerprint_distance

        a = topic_fingerprint("have you started the SS-31 yet")
        b = topic_fingerprint("did you start the SS-31 yet")
        assert fingerprint_distance(a, b) <= 16

    def test_different_topics_are_far_apart(self):
        from virtual_context.core.engagement import fingerprint_distance

        a = topic_fingerprint("have you started the SS-31 yet")
        b = topic_fingerprint("which marker changed your entire protocol")
        assert fingerprint_distance(a, b) > 16

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
            _record(posted_at=NOW - timedelta(days=30),
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
        history = self._history(*[
            _record(posted_at=NOW - timedelta(days=d), channel_id=P3,
                    tagged_actor_id=f"actor:discord:{d}",
                    source_message_ids=(f"m{d}",),
                    topic_fingerprint=topic_fingerprint(f"topic number {d}"))
            for d in (1, 2, 3)
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
