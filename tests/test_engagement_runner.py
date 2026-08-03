"""One run of the daily job. Dry run is the default; posting is asked for.

No test here sends anything real: the sender is always a mock and a test
asserts it is never called on a dry run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

    def test_asking_to_post_without_enabling_refuses(self):
        """Two separate switches: intent, and permission."""
        result = _run(post=True, message_sender=lambda **kw: "1")
        assert result.posted_message_id == ""
        assert "not enabled" in result.refused


class TestPostingWhenAskedAndEnabled:
    def test_it_posts_to_the_rehearsal_channel_only(self):
        seen = {}

        def _sender(*, channel_id, content):
            seen.update(channel_id=channel_id, content=content)
            return "9990001"

        result = _run(post=True, enabled=True, message_sender=_sender)
        assert result.posted_message_id == "9990001"
        assert seen["channel_id"] == VASTTEST
        assert seen["content"] == "Did you end up starting the SS-31?"

    def test_the_post_is_recorded_and_blocks_a_second_that_day(self):
        history = InMemoryPostHistory()
        first = _run(post=True, enabled=True, history=history,
                     message_sender=lambda **kw: "1")
        assert first.posted_message_id == "1"
        second = _run(post=True, enabled=True, history=history,
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

        _run(post=True, enabled=True, source_fetcher=lambda **kw: (404, None),
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
