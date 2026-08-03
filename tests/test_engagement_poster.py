"""The only write path in the package.

Every guard is a refusal rather than a check, because a returned value can
be ignored by a caller that forgets to look at it, and there is no safe
default for sending anyway. No test here performs a real send; the sender is
always a mock, and a test asserts the module cannot open a socket at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import virtual_context.core.engagement.poster as poster_module
from virtual_context.core.engagement import run_once

from virtual_context.core.engagement import (
    POST_CHANNEL_IDS,
    POSTING_ENABLED,
    SOURCE_CHANNEL_IDS,
    Candidate,
    InMemoryPostHistory,
    LiveVerification,
    PostRefused,
    already_posted_today,
    post_question,
)

EASTERN = ZoneInfo("America/New_York")
NOW = datetime(2026, 8, 3, 14, 0, tzinfo=EASTERN)
VASTTEST = "1524946242499514418"
MSG = "1532400954878595094"


def _cand():
    return Candidate(
        canonical_turn_id="ct-1", source_message_id=MSG,
        actor_id="actor:discord:1327457861143494767", channel_id=VASTTEST,
        text="Adding ss31 (5mg) for 4 weeks.", sent_at=NOW - timedelta(days=4),
        sender="BigTex", question_type="timed",
    )


def _verified():
    return LiveVerification(True, "", "", MSG)


def _send(**kw):
    return "9990001"


def _post(**over):
    kw = dict(
        candidate=_cand(), question="Did you start the SS-31?",
        channel_id=VASTTEST, verification=_verified(),
        history=InMemoryPostHistory(), sender=_send, now=NOW,
        question_type="timed",
    )
    kw.update(over)
    return post_question(**kw)


@pytest.fixture
def posting_permitted(monkeypatch):
    """Grant permission the only way it can be granted: patch the module.

    Permission is shipped configuration, not a parameter, so exercising the
    send path costs an explicit patch of a named constant. That is the price
    of the guarantee — a caller has no argument that reaches this, so the
    only way in is one that is obvious in a diff and absent from every
    production caller.
    """
    monkeypatch.setattr(poster_module, "POSTING_ENABLED", True)


@pytest.fixture
def posting_disabled(monkeypatch):
    monkeypatch.setattr(poster_module, "POSTING_ENABLED", False)


class TestPostingIsLiveAndWhatNowConstrainsIt:
    """Posting is enabled. These pin what is left holding it.

    While this shipped False, "nothing can post" was the guarantee and the
    channel list was a second line. That is no longer true: the flag is
    global to the build and says nothing about WHERE. POST_CHANNEL_IDS is now
    the only thing keeping posts to one private channel, so it is asserted
    here rather than left to the channel-refusal class alone.
    """

    def test_the_shipped_value_is_on(self):
        assert POSTING_ENABLED is True

    def test_the_only_destination_is_the_rehearsal_channel(self):
        """The load-bearing guard, now that permission is granted."""
        assert POST_CHANNEL_IDS == (VASTTEST,)

    def test_no_community_channel_is_a_destination(self):
        assert not set(POST_CHANNEL_IDS) & set(SOURCE_CHANNEL_IDS)

    def test_the_refusal_path_still_works_when_disabled(self, posting_disabled):
        """The mechanism must survive the flip, not just the default.

        If this build is ever turned back off, the guard has to still refuse
        — a switch that only worked while it was already closed would be
        discovered at exactly the wrong moment.
        """
        with pytest.raises(PostRefused, match="disabled in this build"):
            _post()

    def test_no_argument_can_enable_posting(self):
        """The boundary, asserted on the shipped signature.

        A caller cannot pass permission because there is no parameter to
        pass it through. This is what makes the guarantee testable as
        impossibility rather than as the current caller's restraint.
        """
        import inspect

        for func in (post_question, run_once):
            params = set(inspect.signature(func).parameters)
            assert "enabled" not in params, func.__name__
            assert not params & {"posting_enabled", "allow_post", "force"}

    def test_enabling_requires_editing_the_module(self, posting_permitted):
        """The inverse: with the constant patched, the guard passes."""
        assert _post().message_id == "9990001"


class TestChannelRefusal:
    def test_the_rehearsal_channel_is_the_only_destination(self, posting_permitted):
        assert POST_CHANNEL_IDS == (VASTTEST,)

    @pytest.mark.parametrize("community", SOURCE_CHANNEL_IDS)
    def test_every_source_channel_is_refused_by_id(self, posting_permitted, community):
        """Refused, not merely checked — by id, against the shipped tuple."""
        with pytest.raises(PostRefused, match="not a permitted staging destination"):
            _post(channel_id=community)

    def test_an_unknown_channel_is_refused(self, posting_permitted):
        with pytest.raises(PostRefused):
            _post(channel_id="999999999999999999")

    def test_nothing_is_sent_when_the_channel_is_refused(self, posting_permitted):
        calls = {"n": 0}

        def _counting(**kw):
            calls["n"] += 1
            return "x"

        with pytest.raises(PostRefused):
            _post(channel_id=SOURCE_CHANNEL_IDS[0], sender=_counting)
        assert calls["n"] == 0


class TestVerificationGate:
    def test_no_verification_refuses(self, posting_permitted):
        with pytest.raises(PostRefused, match="not confirmed live"):
            _post(verification=None)

    def test_a_failed_verification_refuses(self, posting_permitted):
        with pytest.raises(PostRefused, match="not confirmed live"):
            _post(verification=LiveVerification(False, "source_message_deleted"))

    def test_a_verification_for_another_message_refuses(self, posting_permitted):
        """A pass is not transferable between candidates or runs."""
        with pytest.raises(PostRefused, match="different message"):
            _post(verification=LiveVerification(True, "", "", "111111111111"))

    def test_a_verification_with_no_message_id_refuses(self, posting_permitted):
        """Guards against a stale verdict shape carrying an implicit pass."""
        with pytest.raises(PostRefused, match="different message"):
            _post(verification=LiveVerification(True))


class TestIdempotentPerEasternDay:
    def test_a_second_post_the_same_day_refuses(self, posting_permitted):
        history = InMemoryPostHistory()
        _post(history=history)
        with pytest.raises(PostRefused, match="already gone out"):
            _post(history=history)

    def test_the_next_eastern_day_is_allowed(self, posting_permitted):
        history = InMemoryPostHistory()
        _post(history=history)
        result = _post(history=history, now=NOW + timedelta(days=1))
        assert result.day == "2026-08-04"

    def test_the_key_is_the_civil_day_not_an_elapsed_interval(self, posting_permitted):
        """23:30 and 00:30 Eastern are different days, 1 hour apart."""
        history = InMemoryPostHistory()
        late = datetime(2026, 8, 3, 23, 30, tzinfo=EASTERN)
        _post(history=history, now=late)
        result = _post(history=history, now=late + timedelta(hours=1))
        assert result.day == "2026-08-04"

    def test_a_utc_timestamped_record_still_matches_its_eastern_day(self, posting_permitted):
        history = InMemoryPostHistory()
        _post(history=history, now=NOW)
        # Same instant expressed in UTC must not read as a different day.
        assert already_posted_today(
            history, now=NOW.astimezone(timezone.utc),
        ) is True


class TestTheSendItself:
    def test_an_empty_question_is_refused(self, posting_permitted):
        with pytest.raises(PostRefused, match="empty question"):
            _post(question="   ")

    def test_a_send_returning_no_id_is_treated_as_failed(self, posting_permitted):
        with pytest.raises(PostRefused, match="no message id"):
            _post(sender=lambda **kw: "")

    def test_a_successful_post_is_recorded_in_history(self, posting_permitted):
        history = InMemoryPostHistory()
        result = _post(history=history)
        assert result.message_id == "9990001"
        record = history.all()[0]
        # The staging channel's message id belongs in staged_message_id.
        # discord_message_id is reserved for a published reply, so an
        # operator can tell what actually reached a community channel.
        assert record.staged_message_id == "9990001"
        assert record.discord_message_id == ""
        assert record.status == "staged"
        assert record.channel_id == VASTTEST
        assert record.source_message_ids == (MSG,)
        assert record.question_type == "timed"
        assert record.topic_fingerprint != 0

    def test_the_history_record_carries_no_member_content(self, posting_permitted):
        history = InMemoryPostHistory()
        _post(history=history)
        record = history.all()[0]
        assert "Adding ss31" not in record.question_text


class TestTheModuleHoldsNoCredential:
    def test_it_imports_no_http_client_and_opens_no_socket(self, posting_permitted):
        import ast
        import inspect

        from virtual_context.core.engagement import poster

        tree = ast.parse(inspect.getsource(poster))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        for forbidden in ("httpx", "requests", "urllib", "socket", "http"):
            assert forbidden not in imported, forbidden

    def test_no_credential_is_bound_or_embedded(self, posting_permitted):
        """Inspects the code, not the prose.

        A word sweep flagged the docstring sentence explaining that this
        module does NOT hold a token — measuring the file's text rather than
        what it does, which is the same error as a test carrying its own
        ruler. This checks bindings and literals instead.
        """
        import ast
        import inspect

        from virtual_context.core.engagement import poster

        tree = ast.parse(inspect.getsource(poster))
        bound = {
            t.id.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        }
        for forbidden in ("token", "secret", "authorization", "bearer"):
            assert not any(forbidden in name for name in bound), forbidden

        docstrings = {
            ast.get_docstring(n)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                # No literal long enough to be a credential.
                assert len(node.value) < 60 or " " in node.value, node.value[:40]


class TestTheDayIsClaimedBeforeTheSend:
    """A send and a write cannot be atomic, so the ordering picks the failure.

    Claiming first means a crash costs a post. Sending first means it costs
    the record, and the next run then sees an unclaimed day and posts again.
    Skipping is recoverable; a duplicate in a community channel is not.
    """

    def test_a_send_that_raises_leaves_the_day_claimed(self, posting_permitted):
        from virtual_context.core.engagement import pending_claims

        history = InMemoryPostHistory()

        def _explodes(**kw):
            raise OSError("connection reset")

        with pytest.raises(OSError):
            _post(history=history, sender=_explodes)

        assert already_posted_today(history, now=NOW) is True
        assert len(pending_claims(history)) == 1
        assert history.all()[0].status == "pending"
        assert history.all()[0].discord_message_id == ""

    def test_a_claimed_day_refuses_a_second_attempt(self, posting_permitted):
        """The whole point: a crash must not licence a retry."""
        history = InMemoryPostHistory()

        def _explodes(**kw):
            raise OSError("connection reset")

        with pytest.raises(OSError):
            _post(history=history, sender=_explodes)
        with pytest.raises(PostRefused, match="already gone out"):
            _post(history=history)

    def test_a_send_returning_no_id_still_holds_the_day(self, posting_permitted):
        """We cannot tell whether it landed, so the day stays spent."""
        from virtual_context.core.engagement import pending_claims

        history = InMemoryPostHistory()
        with pytest.raises(PostRefused, match="no message id"):
            _post(history=history, sender=lambda **kw: "")
        assert already_posted_today(history, now=NOW) is True
        assert len(pending_claims(history)) == 1

    def test_a_successful_send_confirms_the_claim(self, posting_permitted):
        """The claim resolves to `staged`, and stops being pending.

        `posted` is reserved for a message in a community channel, so a
        successful stage must not claim it — otherwise the ledger says a
        question reached members when it is sitting awaiting approval.
        """
        from virtual_context.core.engagement import pending_claims

        history = InMemoryPostHistory()
        _post(history=history)
        record = history.all()[0]
        assert record.status == "staged"
        assert record.staged_message_id == "9990001"
        assert record.discord_message_id == ""
        assert pending_claims(history) == []

    def test_the_claim_exists_before_the_sender_is_called(self, posting_permitted):
        """Proves the order directly rather than inferring it from outcomes."""
        history = InMemoryPostHistory()
        seen = {}

        def _observing(**kw):
            seen["claimed_at_send_time"] = already_posted_today(
                history, now=NOW,
            )
            seen["status"] = history.all()[0].status
            return "9990001"

        _post(history=history, sender=_observing)
        assert seen["claimed_at_send_time"] is True
        assert seen["status"] == "pending"

    def test_a_pending_claim_is_never_auto_retried(self, posting_permitted):
        """Retrying an unconfirmed send is how a duplicate happens."""
        from virtual_context.core.engagement import pending_claims

        history = InMemoryPostHistory()
        with pytest.raises(PostRefused):
            _post(history=history, sender=lambda **kw: "")
        assert len(pending_claims(history)) == 1
        # Any later attempt that day refuses rather than resending.
        with pytest.raises(PostRefused, match="already gone out"):
            _post(history=history)
        assert len(pending_claims(history)) == 1


class TestAFailedConfirmationStillHoldsTheDay:
    """The message is out and the confirming write failed.

    This is the case the ordering exists for. The send succeeded, so the
    question is public; only the record of its id is missing. The day must
    stay claimed, because the alternative — treating a lost confirmation as
    "nothing happened" — posts the same question twice.
    """

    class _FailsOnUpdate(InMemoryPostHistory):
        def update(self, index, **changes):
            raise OSError("connection reset while confirming")

    def test_the_day_stays_claimed_when_the_confirmation_fails(self, posting_permitted):
        from virtual_context.core.engagement import pending_claims

        history = self._FailsOnUpdate()
        sent = {"n": 0}

        def _sender(**kw):
            sent["n"] += 1
            return "9990001"

        with pytest.raises(OSError):
            _post(history=history, sender=_sender)

        assert sent["n"] == 1, "the message did go out"
        assert already_posted_today(history, now=NOW) is True
        assert len(pending_claims(history)) == 1

    def test_it_never_re_sends_after_a_failed_confirmation(self, posting_permitted):
        history = self._FailsOnUpdate()
        sent = {"n": 0}

        def _sender(**kw):
            sent["n"] += 1
            return "9990001"

        with pytest.raises(OSError):
            _post(history=history, sender=_sender)
        with pytest.raises(PostRefused, match="already gone out"):
            _post(history=history, sender=_sender)

        assert sent["n"] == 1, "re-sent a question that had already gone out"

    def test_the_unconfirmed_claim_has_no_message_id(self, posting_permitted):
        """So an operator can tell it apart from a completed post."""
        history = self._FailsOnUpdate()
        with pytest.raises(OSError):
            _post(history=history, sender=lambda **kw: "9990001")
        record = history.all()[0]
        assert record.status == "pending"
        assert record.discord_message_id == ""


class TestTheFingerprintKeysOnTheQuestion:
    """Presentation must not decide whether a question counts as a repeat.

    Fingerprinting the delivery body inverts the rule in both directions.
    Measured against the shipped threshold: the same question bare vs wrapped
    scored 30, so a real repeat was missed; two different questions about one
    original scored 5, so a good question was rejected. Both while naming
    question_recently_asked in the ladder.
    """

    QUESTION = "Did you end up starting the SS-31?"
    OTHER = "What made you pick the morning dose over the evening one?"
    # Realistic length matters: a short quote does not dominate the token
    # set and the false-repeat direction does not reproduce. With this one
    # the defect scores 29 and 5 against a threshold of 12 — the same shape
    # cloud measured in production (30 and 5). A shorter fixture passes the
    # negative control while proving only half the property.
    ORIGINAL = (
        "Adding ss31 (5mg) for 4 weeks. Adding in MotsC after SS31. Labs "
        "pending, should have them back next week sometime. Been running the "
        "enclo at 25mg MWF alongside, sleep has been rough since I moved the "
        "modafinil earlier in the day, and the KPV 500mcg in the morning "
        "seems to be helping the gut stuff more than I expected honestly."
    )

    def _wrap(self, question):
        return (
            f"> {self.ORIGINAL}\n"
            f"— Rob in #p3ptides, 9 days ago\n\n"
            f"<@1338726888809697364> {question}"
        )

    def _recorded(self, posting_permitted, **over):
        history = InMemoryPostHistory()
        _post(history=history, **over)
        return history.all()[0]

    def test_wrapping_does_not_change_the_fingerprint(self, posting_permitted):
        bare = self._recorded(posting_permitted, question=self.QUESTION)
        wrapped = self._recorded(
            posting_permitted, question=self.QUESTION,
            delivery_body=self._wrap(self.QUESTION),
        )
        assert wrapped.topic_fingerprint == bare.topic_fingerprint

    def test_two_questions_about_one_original_stay_distinct(
        self, posting_permitted,
    ):
        """The quoted original must not dominate the token set."""
        from virtual_context.core.engagement import fingerprint_distance
        from virtual_context.core.engagement.history import SIMILARITY_DISTANCE

        one = self._recorded(
            posting_permitted, question=self.QUESTION,
            delivery_body=self._wrap(self.QUESTION),
        )
        two = self._recorded(
            posting_permitted, question=self.OTHER,
            delivery_body=self._wrap(self.OTHER),
        )
        distance = fingerprint_distance(
            one.topic_fingerprint, two.topic_fingerprint,
        )
        assert distance > SIMILARITY_DISTANCE, (
            f"distinct questions collapsed to distance {distance}"
        )

    def test_the_ledger_records_the_question_not_the_delivery(
        self, posting_permitted,
    ):
        """question_text is for near-duplicate review; the wrapper is noise."""
        row = self._recorded(
            posting_permitted, question=self.QUESTION,
            delivery_body=self._wrap(self.QUESTION),
        )
        assert row.question_text == self.QUESTION
        assert "— Rob in #p3ptides" not in row.question_text

    def test_the_delivery_body_is_what_actually_gets_sent(
        self, posting_permitted,
    ):
        seen = {}

        def _sender(**kw):
            seen.update(kw)
            return "9990001"

        wrapped = self._wrap(self.QUESTION)
        _post(question=self.QUESTION, delivery_body=wrapped, sender=_sender)
        assert seen["content"] == wrapped

    def test_without_a_delivery_body_the_question_is_sent(
        self, posting_permitted,
    ):
        seen = {}

        def _sender(**kw):
            seen.update(kw)
            return "9990001"

        _post(question=self.QUESTION, sender=_sender)
        assert seen["content"] == self.QUESTION


class TestStagingProducesTheStateTheApprovalPathConsumes:
    """The producer for `staged`, which did not exist.

    The columns, the statuses and both consumers shipped before anything
    wrote the state they consume: claim_for_publish and decline both match
    `WHERE status = 'staged'`, and nothing ever set it. The approval loop
    could never fire, and a staged question was recorded as posted.
    """

    def test_a_staged_row_is_what_the_approval_path_looks_for(
        self, posting_permitted,
    ):
        history = InMemoryPostHistory()
        _post(history=history)
        staged = [r for r in history.all() if r.status == "staged"]
        assert len(staged) == 1, "the poller's query would find nothing"

    def test_the_claim_can_actually_be_won_on_a_real_staged_row(
        self, posting_permitted,
    ):
        """claim_for_publish could never return True before this."""
        history = InMemoryPostHistory()
        _post(history=history)
        row = history.all()[0]
        assert history.claim_for_publish(row.id) is True

    def test_a_real_staged_row_can_be_declined(self, posting_permitted):
        history = InMemoryPostHistory()
        _post(history=history)
        row = history.all()[0]
        assert history.decline(row.id) is True
        assert already_posted_today(history, now=NOW) is False

    def test_posted_still_means_a_published_message(self, posting_permitted):
        """The publish path sets it; staging must not."""
        history = InMemoryPostHistory()
        _post(history=history)
        row = history.all()[0]
        history.claim_for_publish(row.id)
        history.update(row.id, status="posted",
                       discord_message_id="1533900000000000000")
        after = history.all()[0]
        assert after.status == "posted"
        assert after.discord_message_id == "1533900000000000000"
        assert after.staged_message_id == "9990001", "the stage id was lost"

    def test_the_two_ids_are_never_the_same_field(self, posting_permitted):
        """An operator must be able to tell what reached a community channel."""
        history = InMemoryPostHistory()
        _post(history=history)
        row = history.all()[0]
        assert row.staged_message_id and not row.discord_message_id
