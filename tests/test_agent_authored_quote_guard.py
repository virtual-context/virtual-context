"""A reply quoting the agent must not be filed as a person's disclosure.

``_build_actor_roster`` resolves a quote-reply against the transport messages
it already holds. When the quoted message is not among them it creates a
subject lane, which says "this text was written by the person named here".

The agent's own outbound message ids were recorded nowhere, so a reply quoting
the agent never resolved and its own output was filed as a named human's
disclosure. That is how the agent came to hold, as evidence about a person,
words the person never said.

The correction is one-directional on purpose. A recorded identity is positive
evidence and suppresses the lane. Absence is unknown and must keep today's
behaviour, because the recorded set is always partial: a reply split across
several platform messages reports at most one of their ids.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

import pytest

from virtual_context.core.compaction_pipeline import CompactionPipeline
from virtual_context.types import (
    AUTHOR_ROLE_SUBJECT,
    CanonicalTurnRow,
    Message,
    SOURCE_CANONICAL_TURN_IDS_KEY,
    TaggedSegment,
)

CONV = "c"
CHAN = "chan-1"
NAMESPACE = ("scope-1", "discord", "acct-1")


class _Ledger:
    """Records what the guard asked, and answers only for exact identities."""

    def __init__(self, known: set[tuple] | None = None, namespace=NAMESPACE):
        self.known = known or set()
        self.namespace = namespace
        self.asked: list[tuple] = []

    def get_all_canonical_turns(self, _cid):
        return []

    def resolve_channel_namespace(self, *, conversation_id, channel_id):
        return self.namespace

    def is_bot_authored_message(
        self, *, tenant_id, agent_scope_id, conversation_id,
        platform, account_id, channel_id, message_id,
    ):
        key = (agent_scope_id, platform, account_id, channel_id, message_id)
        self.asked.append(key)
        return key in self.known


def _pipeline(store) -> CompactionPipeline:
    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(conversation_id=CONV, tenant_id="t")
    return pipeline


def _reply_quoting(message_id: str):
    """A reply quoting a message this conversation does NOT hold a row for."""
    reply = CanonicalTurnRow(
        conversation_id=CONV, canonical_turn_id="reply",
        user_content="what do you think?",
        sender_actor_id="actor:discord:member",
        reply_target_message_id=message_id,
        reply_subject_actor_id="actor:discord:member",
        reply_target_body="I take 500mg of berberine daily",
        reply_attribution_version=1,
        audience_conversation_id="guild", origin_channel_id=CHAN,
    )
    segment = TaggedSegment(messages=[
        Message(
            role="user", content="what do you think?",
            metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["reply"]},
        ),
    ])
    return segment, {"reply": reply}


def _subject_lanes(roster):
    return [lane for lane in roster.lanes if lane.role == AUTHOR_ROLE_SUBJECT]


def _key(message_id: str) -> tuple:
    scope, platform, account = NAMESPACE
    return (scope, platform, account, CHAN, message_id)


def test_a_quote_of_a_recorded_agent_message_creates_no_subject_lane():
    """A1. The defect being closed."""
    store = _Ledger(known={_key("bot-msg-1")})
    segment, rows = _reply_quoting("bot-msg-1")

    roster = _pipeline(store)._build_actor_roster(segment, rows)

    assert _subject_lanes(roster) == [], (
        "the agent's own quoted output was filed as a person's disclosure"
    )


def test_an_unrecorded_quote_still_creates_a_subject_lane():
    """A2 and A5 together, and the row most likely to be got wrong.

    Nothing recorded is not evidence that the agent did not write it, but it is
    equally not a licence to drop the lane. Degrading to today's behaviour is
    the required fallback; degrading to suppression would delete a real
    person's disclosure and log a success.
    """
    store = _Ledger(known=set())
    segment, rows = _reply_quoting("unreported-msg")

    roster = _pipeline(store)._build_actor_roster(segment, rows)

    assert len(_subject_lanes(roster)) == 1, (
        "an unrecorded quote was suppressed; absence was treated as evidence"
    )
    assert _subject_lanes(roster)[0].text == "I take 500mg of berberine daily"


def test_a_non_member_of_a_partial_set_still_creates_a_subject_lane():
    """A5 proper. The set holds one id; a DIFFERENT id must not be suppressed
    merely because the set exists and does not contain it."""
    store = _Ledger(known={_key("bot-msg-1")})
    segment, rows = _reply_quoting("bot-msg-2")

    roster = _pipeline(store)._build_actor_roster(segment, rows)

    assert len(_subject_lanes(roster)) == 1, (
        "non-membership in a partial set was read as known-not-ours"
    )


def test_the_guard_asks_with_the_full_namespaced_identity():
    """A8. The question must carry every component, or the answer is about a
    different message."""
    store = _Ledger(known=set())
    segment, rows = _reply_quoting("bot-msg-9")

    _pipeline(store)._build_actor_roster(segment, rows)

    assert store.asked == [_key("bot-msg-9")]


def test_an_unresolvable_channel_namespace_does_not_suppress():
    """Unknown namespace is unknown, not a licence to pick one."""
    store = _Ledger(known={_key("bot-msg-1")}, namespace=None)
    segment, rows = _reply_quoting("bot-msg-1")

    roster = _pipeline(store)._build_actor_roster(segment, rows)

    assert len(_subject_lanes(roster)) == 1
    assert store.asked == [], "the guard asked with a namespace it had to guess"


def test_a_store_without_the_ledger_keeps_todays_behaviour():
    """A6-adjacent. An enhancement that is absent must change nothing."""
    store = SimpleNamespace(get_all_canonical_turns=lambda _cid: [])
    segment, rows = _reply_quoting("bot-msg-1")

    roster = _pipeline(store)._build_actor_roster(segment, rows)

    assert len(_subject_lanes(roster)) == 1


def test_a_raising_ledger_keeps_todays_behaviour():
    """A failing enhancement must never change how a turn is filed."""
    class _Broken(_Ledger):
        def resolve_channel_namespace(self, **_kw):
            raise RuntimeError("ledger unavailable")

    segment, rows = _reply_quoting("bot-msg-1")

    roster = _pipeline(_Broken())._build_actor_roster(segment, rows)

    assert len(_subject_lanes(roster)) == 1


def test_a_resolved_in_conversation_target_is_still_suppressed_without_the_ledger():
    """The pre-existing suppression path must be untouched: a quote whose
    target IS held as a transport row still creates no subject lane, and the
    ledger is not consulted for it."""
    store = _Ledger(known=set())
    target = CanonicalTurnRow(
        conversation_id=CONV, canonical_turn_id="target", user_content="claim",
        source_message_id="m1", sender_actor_id="actor:discord:member",
        audience_conversation_id="guild", origin_channel_id=CHAN,
    )
    segment, rows = _reply_quoting("m1")
    rows["target"] = target

    roster = _pipeline(store)._build_actor_roster(segment, rows)

    assert _subject_lanes(roster) == []
    assert store.asked == []


@pytest.mark.regression("BUG-054")
def test_a_suppression_says_so(caplog):
    """The guard must announce when it fires.

    Its only other outward sign is the absence of a mis-filed row, and absence
    of rows looks exactly like absence of traffic. Without this line a feature
    sitting inert and a feature working perfectly produce identical evidence.
    """
    import logging

    store = _Ledger(known={_key("bot-msg-1")})
    segment, rows = _reply_quoting("bot-msg-1")

    with caplog.at_level(logging.INFO, logger="virtual_context.core.compaction_pipeline"):
        roster = _pipeline(store)._build_actor_roster(segment, rows)

    assert _subject_lanes(roster) == []
    assert any("AGENT_QUOTE_SUPPRESSED" in r.message for r in caplog.records), (
        "the guard suppressed a lane without leaving any trace that it did"
    )
    line = next(r.message for r in caplog.records if "AGENT_QUOTE_SUPPRESSED" in r.message)
    for field in ("conv=", "channel=", "target_message_id=", "canonical_turn_id="):
        assert field in line


@pytest.mark.regression("BUG-054")
def test_no_suppression_means_no_suppression_line(caplog):
    """The line must mean what it says, or a reader counting it overcounts."""
    import logging

    store = _Ledger(known=set())
    segment, rows = _reply_quoting("unreported-msg")

    with caplog.at_level(logging.INFO, logger="virtual_context.core.compaction_pipeline"):
        _pipeline(store)._build_actor_roster(segment, rows)

    assert not any("AGENT_QUOTE_SUPPRESSED" in r.message for r in caplog.records)
