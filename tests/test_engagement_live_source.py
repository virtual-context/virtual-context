"""Re-fetching a message before quoting it, and blocking when it changed.

The attested record proves what a message was at ingest, not what it is
now. These tests pin the four projection-independent signals and, crucially,
that a block reaches the rendered artifact rather than stopping at an object
nobody reads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.core.engagement import (
    BLOCKING_REASONS,
    LIVE_AUTHOR_MISMATCH,
    LIVE_DELETED,
    LIVE_EDITED,
    LIVE_RATE_LIMITED,
    Candidate,
    DryRunReport,
    Rejection,
    verify_source_live,
)

NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
CHAN = "1530567788949798963"
MSG = "1532400954878595094"
ACTOR = "actor:discord:1327457861143494767"


def _cand():
    return Candidate(
        canonical_turn_id="ct-1", source_message_id=MSG, actor_id=ACTOR,
        channel_id=CHAN, text="Adding ss31 (5mg) for 4 weeks.",
        sent_at=NOW - timedelta(days=4), sender="BigTex",
        question_type="timed",
    )


def _fetch(status=200, **over):
    payload = {
        "id": MSG, "channel_id": CHAN,
        "author": {"id": "1327457861143494767"},
        "edited_timestamp": None, "content": "Adding ss31 (5mg) for 4 weeks",
    }
    payload.update(over)
    return lambda **kw: (status, payload if status == 200 else None)


class TestTheFourSignals:
    def test_an_unchanged_message_verifies(self):
        assert verify_source_live(_cand(), fetcher=_fetch()).ok is True

    def test_a_deleted_message_blocks(self):
        v = verify_source_live(_cand(), fetcher=_fetch(404))
        assert v.ok is False and v.reason == LIVE_DELETED

    def test_an_edited_message_blocks(self):
        v = verify_source_live(
            _cand(), fetcher=_fetch(edited_timestamp="2026-08-01T10:00:00Z"),
        )
        assert v.ok is False and v.reason == LIVE_EDITED

    def test_a_different_author_blocks(self):
        """The P0 shape: the row's actor is not who actually wrote it."""
        v = verify_source_live(
            _cand(), fetcher=_fetch(author={"id": "999999999999999999"}),
        )
        assert v.ok is False and v.reason == LIVE_AUTHOR_MISMATCH
        assert "999999999999999999" in v.detail

    def test_a_different_channel_blocks(self):
        v = verify_source_live(_cand(), fetcher=_fetch(channel_id="123"))
        assert v.ok is False

    def test_rate_limiting_blocks_rather_than_proceeding(self):
        v = verify_source_live(_cand(), fetcher=_fetch(429))
        assert v.ok is False and v.reason == LIVE_RATE_LIMITED

    def test_an_unreachable_source_blocks(self):
        def _boom(**kw):
            raise OSError("network down")

        assert verify_source_live(_cand(), fetcher=_boom).ok is False


class TestContentIsDeliberatelyNotCompared:
    def test_a_changed_body_alone_does_not_block(self):
        """Content is not compared; the edited marker is the signal.

        A body check would mismatch on ~99% of rows because the stored hash
        covers a speaker-prefixed body. This test exists so that adding one
        as an 'improvement' fails here.
        """
        v = verify_source_live(
            _cand(), fetcher=_fetch(content="completely different text"),
        )
        assert v.ok is True

    def test_the_module_says_why_content_is_not_compared(self):
        import inspect

        from virtual_context.core.engagement import live_source

        source = inspect.getsource(live_source)
        assert "CONTENT IS DELIBERATELY NOT COMPARED" in source
        assert "19 of 1645" in source


class TestMembershipIsNotAuthorship:
    def test_leaving_the_guild_is_not_a_blocking_reason(self):
        assert not any("guild" in r or "member" in r for r in BLOCKING_REASONS)


class TestABlockReachesTheRenderedArtifact:
    """The claim that matters is the artifact, not the object."""

    def test_a_blocked_verification_is_a_counted_row(self):
        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=CHAN,
            outcome_kind="skip", skip_stage="live_source",
            rejections=[Rejection("ct-1", "live_source", LIVE_DELETED,
                                  f"message {MSG}")],
        )
        rendered = report.render()
        assert f"[live_source] {LIVE_DELETED}: 1" in rendered
        assert MSG in rendered

    def test_the_skip_names_the_live_source_stage(self):
        report = DryRunReport(
            generated_at=NOW, conversation_id="c", channel_id=CHAN,
            outcome_kind="skip", skip_stage="live_source",
            skip_reason="source_message_deleted",
        )
        rendered = report.render()
        assert "stage  : live_source" in rendered


class TestSelectionWalksTheRanking:
    def test_the_first_verifying_candidate_is_chosen(self):
        from virtual_context.core.engagement import select_live_verified

        deleted = _cand()
        good = Candidate(
            canonical_turn_id="ct-2", source_message_id="222", actor_id=ACTOR,
            channel_id=CHAN, text="t", sent_at=NOW, sender="BigTex",
        )

        def fetcher(*, channel_id, message_id):
            if message_id == MSG:
                return 404, None
            return 200, {"channel_id": CHAN, "author": {"id": "1327457861143494767"},
                         "edited_timestamp": None}

        chosen, rejections = select_live_verified(
            [deleted, good], fetcher=fetcher,
        )
        assert chosen.canonical_turn_id == "ct-2"
        assert [r.reason for r in rejections] == [LIVE_DELETED]

    def test_a_rate_limit_stops_the_walk_rather_than_hammering(self):
        from virtual_context.core.engagement import select_live_verified

        calls = {"n": 0}

        def fetcher(**kw):
            calls["n"] += 1
            return 429, None

        chosen, rejections = select_live_verified(
            [_cand(), _cand(), _cand()], fetcher=fetcher,
        )
        assert chosen is None
        assert calls["n"] == 1
        assert rejections[0].reason == LIVE_RATE_LIMITED

    def test_the_walk_is_bounded(self):
        from virtual_context.core.engagement import select_live_verified

        calls = {"n": 0}

        def fetcher(**kw):
            calls["n"] += 1
            return 404, None

        select_live_verified([_cand()] * 10, fetcher=fetcher, max_attempts=3)
        assert calls["n"] == 3

    def test_every_block_is_a_named_rejection_at_the_live_source_stage(self):
        from virtual_context.core.engagement import select_live_verified

        _, rejections = select_live_verified(
            [_cand()], fetcher=lambda **kw: (404, None),
        )
        assert rejections[0].stage == "live_source"
        assert rejections[0].reason in BLOCKING_REASONS
