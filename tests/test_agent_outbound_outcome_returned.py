"""The store's outcome must reach the sender, not only the engine's log.

A sender whose delivery succeeds while every identity is refused otherwise
reports success. That happened: a producer-side counter read four carried while
the receiver logged three offered, zero accepted, three declined — and the
discrepancy was only found by recovering container logs the sender cannot read.

So the counts are returned, not just logged. What they must NOT carry is any
notion of a set that was not seen: no total, no denominator, nothing from which
a reader could conclude that the identities it knows about are all of them.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import MagicMock

from virtual_context.types import AGENT_OUTBOUND_IDS_KEY


def _ident(**over):
    base = {"platform": "discord", "account_id": "vast", "channel_id": "c1",
            "message_id": "m1", "observed_at": "2026-08-20T16:00:00Z"}
    base.update(over)
    return base


def _engine(outcome):
    from virtual_context.engine import VirtualContextEngine
    eng = object.__new__(VirtualContextEngine)
    eng.config = SimpleNamespace(conversation_id="conv-1", tenant_id="t")
    eng._store = MagicMock()
    eng._store.record_bot_outbound_messages.return_value = outcome
    return eng


def test_the_outcome_is_returned_not_only_logged():
    eng = _engine({"accepted": 0, "duplicate": 0, "fence_rejection": 3})

    result = eng._record_agent_outbound_ids_from_metadata(
        {AGENT_OUTBOUND_IDS_KEY: [_ident(), _ident(message_id="m2")]}
    )

    assert result == {"accepted": 0, "duplicate": 0, "fence_rejection": 3}, (
        "the refusal is visible only in a log the sender cannot read"
    )


def test_the_reason_strings_are_passed_through_verbatim():
    """The party that owns the semantic is the only one that should express it.
    A rename on the store side must reach the sender unmapped, so their
    unknown-is-retried default can absorb it."""
    eng = _engine({"accepted": 0, "duplicate": 0, "some_future_reason": 2})

    result = eng._record_agent_outbound_ids_from_metadata(
        {AGENT_OUTBOUND_IDS_KEY: [_ident()]}
    )

    assert "some_future_reason" in result
    assert result["some_future_reason"] == 2


def test_nothing_carried_returns_none_rather_than_an_empty_report():
    """Absent must stay distinguishable from zero. A turn that carried no
    identities is not a turn whose identities were all refused."""
    eng = _engine({"accepted": 1, "duplicate": 0})

    assert eng._record_agent_outbound_ids_from_metadata({}) is None
    assert eng._record_agent_outbound_ids_from_metadata(None) is None
    assert eng._record_agent_outbound_ids_from_metadata(
        {AGENT_OUTBOUND_IDS_KEY: []}
    ) is None


def test_a_raising_store_returns_none_and_never_propagates():
    """The turn is the product. A failure reporting on identities must not
    become a failure of the turn."""
    eng = _engine(None)
    eng._store.record_bot_outbound_messages.side_effect = RuntimeError("down")

    assert eng._record_agent_outbound_ids_from_metadata(
        {AGENT_OUTBOUND_IDS_KEY: [_ident()]}
    ) is None


def test_the_outcome_carries_no_total_and_no_denominator():
    """The constraint that matters. A field implying completeness would let a
    reader conclude that the identities it saw are all that exist, which is the
    hazard the whole contract exists to prevent: non-membership must stay
    unknown, and a count of "how many there were" would make it inferable.
    """
    eng = _engine({"accepted": 1, "duplicate": 2, "fence_rejection": 1})

    result = eng._record_agent_outbound_ids_from_metadata(
        {AGENT_OUTBOUND_IDS_KEY: [_ident(), _ident(message_id="m2")]}
    )

    forbidden = {"total", "offered", "count", "all", "all_accepted",
                 "complete", "final", "expected", "denominator"}
    assert not (forbidden & set(result)), (
        f"the outcome implies a set size: {forbidden & set(result)}"
    )
