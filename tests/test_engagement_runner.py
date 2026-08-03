"""One run of the daily job. Dry run is the default; posting is asked for.

No test here sends anything real: the sender is always a mock and a test
asserts it is never called on a dry run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import virtual_context.core.engagement.poster as poster_module
from virtual_context.core.engagement import (
    Draft, FidelityVerdict, InMemoryPostHistory, MessageSourceRecord,
    rehearsal_allowlist, run_once,
)
from virtual_context.types import QuoteResult, SourceProvenance

EASTERN = ZoneInfo("America/New_York")
NOW = datetime(2026, 8, 3, 14, 0, tzinfo=EASTERN)
CONV = "sk:agent:vast:discord:guild:1524917037191925871"
P3 = "1524917968440524990"
VASTTEST = "1524946242499514418"
MSG = "1532400954878595094"
ACTOR = "actor:discord:1327457861143494767"


def _result():
    return QuoteResult(
        text="BigTex: Adding ss31 (5mg) for 4 weeks.", tag="",
        segment_ref="ct-1", source_scope="turn", matched_side="user",
        provenance=SourceProvenance(
            conversation_id=CONV, canonical_turn_id="ct-1",
            source_role="requester", actor_id=ACTOR,
            audience_conversation_id=CONV, audience_attribution_version=1,
            origin_channel_id=P3, source_message_id=MSG,
        ),
    )


def _sources():
    return {"ct-1": MessageSourceRecord(
        canonical_turn_id="ct-1", message_id=MSG, channel_id=P3,
        guild_id="1524917037191925871",
        author_id="1327457861143494767", source_actor_id=ACTOR,
    )}


def _fetcher(**kw):
    return 200, {"channel_id": P3, "author": {"id": "1327457861143494767"},
                 "edited_timestamp": None}


def _qualifier(verified, *, now):
    import dataclasses
    return [dataclasses.replace(c, question_type="timed") for c in verified], []


def _drafter(candidate):
    return Draft("Did you end up starting the SS-31?", ""), FidelityVerdict(True)


def _run(**over):
    kw = dict(
        results=[_result()], sources=_sources(), senders={"ct-1": "BigTex"},
        allowlist=rehearsal_allowlist(), history=InMemoryPostHistory(),
        now=NOW, conversation_id=CONV, qualifier=_qualifier,
        drafter=_drafter, source_fetcher=_fetcher,
    )
    kw.update(over)
    return run_once(**kw)


@pytest.fixture
def posting_permitted(monkeypatch):
    """Permission is shipped config; the send path costs an explicit patch."""
    monkeypatch.setattr(poster_module, "POSTING_ENABLED", True)


class TestDryRunIsTheDefault:
    def test_a_run_sends_nothing_by_default(self):
        calls = {"n": 0}

        def _sender(**kw):
            calls["n"] += 1
            return "x"

        result = _run(message_sender=_sender)
        assert calls["n"] == 0
        assert result.posted_message_id == ""
        assert result.refused == "dry_run"

    def test_a_dry_run_still_produces_the_artifact(self):
        rendered = _run().report.render()
        assert "Did you end up starting the SS-31?" in rendered
        assert "SOURCE RE-FETCHED LIVE" in rendered

    def test_asking_to_post_in_the_shipped_build_refuses(self):
        """Two separate switches: intent, and permission."""
        result = _run(post=True, message_sender=lambda **kw: "1")
        assert result.posted_message_id == ""
        assert "disabled in this build" in result.refused


class TestPostingWhenAskedAndEnabled:
    def test_it_posts_to_the_rehearsal_channel_only(self, posting_permitted):
        seen = {}

        def _sender(*, channel_id, content):
            seen.update(channel_id=channel_id, content=content)
            return "9990001"

        result = _run(post=True, message_sender=_sender)
        assert result.posted_message_id == "9990001"
        assert seen["channel_id"] == VASTTEST
        assert seen["content"] == "Did you end up starting the SS-31?"

    def test_the_post_is_recorded_and_blocks_a_second_that_day(self, posting_permitted):
        history = InMemoryPostHistory()
        first = _run(post=True, history=history,
                     message_sender=lambda **kw: "1")
        assert first.posted_message_id == "1"
        second = _run(post=True, history=history,
                      message_sender=lambda **kw: "2")
        assert second.posted_message_id == ""
        assert "already gone out" in second.refused


class TestNoVerifiedCandidate:
    def test_a_deleted_source_stops_the_run_and_is_counted(self):
        result = _run(source_fetcher=lambda **kw: (404, None))
        assert result.refused == "no_verified_candidate"
        rendered = result.report.render()
        assert "[live_source] source_message_deleted: 1" in rendered
        assert "CANNOT detect a message edited or deleted" in rendered

    def test_nothing_is_sent_when_no_candidate_verifies(self):
        calls = {"n": 0}

        def _sender(**kw):
            calls["n"] += 1
            return "x"

        _run(post=True, source_fetcher=lambda **kw: (404, None),
             message_sender=_sender)
        assert calls["n"] == 0


class TestTheRunnerReachesNothingItself:
    def test_it_imports_no_http_client(self):
        import ast
        import inspect

        from virtual_context.core.engagement import runner

        tree = ast.parse(inspect.getsource(runner))
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
        for forbidden in ("httpx", "requests", "urllib", "socket"):
            assert forbidden not in imported, forbidden


class TestEveryQuestionTypeRenders:
    """Listing kinds meant a new one silently rendered nothing."""

    def test_a_timed_followup_renders_its_question(self):
        assert "PROPOSED QUESTION" in _run().report.render()

    def test_a_personal_continuation_renders_its_question(self):
        import dataclasses

        def _personal(verified, *, now):
            return [dataclasses.replace(
                c, question_type="personal",
                hook_kind="dose_or_compound_change",
            ) for c in verified], []

        rendered = _run(qualifier=_personal).report.render()
        assert "PROPOSED QUESTION" in rendered
        assert "question type: personal" in rendered
        assert "hook: dose_or_compound_change" in rendered

    def test_a_skip_renders_no_question(self):
        rendered = _run(source_fetcher=lambda **kw: (404, None)).report.render()
        assert "PROPOSED QUESTION" not in rendered
        assert "SKIPPED" in rendered


class TestAFailedDraftCostsACandidateNotTheDay:
    """One flaky model call must not decide there was nothing worth asking.

    Cloud measured both halves of this against production: one run lost the
    day to a detector that returned no hook on its second call, another
    composed from a hook that never passed qualification. The hook fix closed
    the second; this closes the first.
    """

    def _three(self, n=3):
        """n distinct candidates, each with its own turn and message id."""
        import dataclasses

        from virtual_context.core.discord_snowflake import (
            datetime_to_snowflake_floor,
        )

        out = []
        for i in range(n):
            sent = NOW - timedelta(days=3 + i)
            base = _result()
            out.append(dataclasses.replace(
                base, provenance=dataclasses.replace(
                    base.provenance, canonical_turn_id=f"ct-{i}",
                    source_message_id=str(
                        datetime_to_snowflake_floor(sent) + 7 + i,
                    ),
                ),
            ))
        return out

    def _sources_for(self, results):
        """An attestation per candidate, matching what the rows claim.

        Without this every candidate is correctly rejected as
        no_attested_source and the drafter is never reached — which is the
        verifier working, not the fall-through failing.
        """
        return {
            r.provenance.canonical_turn_id: MessageSourceRecord(
                canonical_turn_id=r.provenance.canonical_turn_id,
                message_id=r.provenance.source_message_id,
                channel_id=P3, guild_id="1524917037191925871",
                author_id="1327457861143494767", source_actor_id=ACTOR,
            )
            for r in results
        }

    def _run_three(self, n=3, **over):
        rows = self._three(n)
        kw = dict(results=rows, sources=self._sources_for(rows),
                  senders={r.provenance.canonical_turn_id: "BigTex"
                           for r in rows})
        kw.update(over)
        return _run(**kw)

    def test_it_advances_past_a_rejected_draft(self):
        seen: list = []

        def _drafter(candidate):
            seen.append(candidate.canonical_turn_id)
            if len(seen) == 1:
                return Draft("", "evidence_not_in_quote"), FidelityVerdict(False)
            return Draft("a good question?", ""), FidelityVerdict(True)

        result = self._run_three(drafter=_drafter)
        assert len(seen) == 2, "the run stopped at the first failure"
        assert result.report.question == "a good question?"

    def test_a_failed_attempt_is_a_counted_row_not_silence(self):
        """'Tried 3, all rejected' must not read as 'nothing qualified'."""
        def _drafter(candidate):
            return Draft("", "evidence_not_in_quote"), FidelityVerdict(False)

        result = self._run_three(drafter=_drafter)
        draft_rejections = [r for r in result.rejections if r.stage == "draft"]
        assert len(draft_rejections) >= 1
        assert all(r.reason for r in draft_rejections), "unnamed rejection"
        assert result.refused == "every_draft_rejected"

    def test_exhausting_attempts_is_distinguishable_from_none_qualifying(self):
        def _drafter(candidate):
            return Draft("", "empty_draft"), FidelityVerdict(False)

        tried = self._run_three(drafter=_drafter)
        nothing = _run(results=[], drafter=_drafter)
        assert tried.refused == "every_draft_rejected"
        assert nothing.refused == "no_verified_candidate"

    def test_the_walk_is_bounded(self):
        from virtual_context.core.engagement.runner import DRAFT_ATTEMPT_CAP

        calls: list = []

        def _drafter(candidate):
            calls.append(candidate.canonical_turn_id)
            return Draft("", "empty_draft"), FidelityVerdict(False)

        self._run_three(n=12, drafter=_drafter)
        assert len(calls) <= DRAFT_ATTEMPT_CAP, "the walk was unbounded"

    def test_live_verification_runs_for_every_attempt(self):
        """A pass belongs to one message in one run and is not transferable.

        Falling through must re-verify, not carry the previous candidate's
        verdict forward — otherwise the guarantee erodes exactly where it is
        load-bearing, on the candidate that actually gets posted.
        """
        fetched: list = []
        drafted: list = []

        def _fetcher(*, channel_id, message_id):
            fetched.append(message_id)
            return 200, {"author": {"id": "1327457861143494767"},
                         "channel_id": channel_id, "edited_timestamp": None}

        def _drafter(candidate):
            # Record what had been verified at the moment this draft began.
            drafted.append((candidate.source_message_id, list(fetched)))
            if len(drafted) == 1:
                return Draft("", "empty_draft"), FidelityVerdict(False)
            return Draft("q?", ""), FidelityVerdict(True)

        self._run_three(drafter=_drafter, source_fetcher=_fetcher)
        assert len(drafted) == 2, "the run did not fall through"
        for message_id, verified_before in drafted:
            # Each drafted candidate had its OWN source fetched before the
            # draft. A verdict is not transferable between candidates, so
            # reusing the first pass for the second would show up here as a
            # draft whose message was never fetched.
            assert message_id in verified_before, (
                f"drafted {message_id} without verifying it in this run"
            )
        assert len(set(fetched)) == len(fetched), "a message was re-fetched"

    def test_the_day_is_claimed_once_by_the_candidate_that_posts(
        self, posting_permitted,
    ):
        history = InMemoryPostHistory()

        def _drafter(candidate):
            if candidate.canonical_turn_id == "ct-0":
                return Draft("", "empty_draft"), FidelityVerdict(False)
            return Draft("a good question?", ""), FidelityVerdict(True)

        result = self._run_three(drafter=_drafter, post=True,
                                 history=history,
                                 message_sender=lambda **kw: "9001")
        assert result.posted_message_id == "9001"
        assert len(history.all()) == 1, "a rejected attempt claimed the day"
        assert history.all()[0].status == "posted"
