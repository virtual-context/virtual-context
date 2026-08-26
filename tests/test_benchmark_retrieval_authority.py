"""The benchmark harness constructs real retrieval authority (BUG-059).

Model-facing quote search fails closed without request-owned authority:
the tool runtime coerces an absent ``speaker_context`` to the ineligible
sentinel and ``find_quote`` refuses with "retrieval authority is
unproved". The LongMemEval harness called ``query_with_tools`` with no
context at all, so every ``vc_find_quote`` in a benchmark run refused
and the reader was told no search was performed.

The harness owns the single-conversation store it built, so it now
constructs the owner-routed DM shape: audience is the conversation
itself, empty channel, exact-match channel scope. These tests pin that
the constructed shape passes the quote-search request gate while the
ineligible sentinel keeps failing it.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

import pytest

from virtual_context.core.quote_search import _speaker_request_scope_is_valid
from virtual_context.types import SpeakerRetrievalContext


def _engine_double(conversation_id: str = "bench-q1", tenant_id: str = ""):
    return SimpleNamespace(
        config=SimpleNamespace(
            conversation_id=conversation_id, tenant_id=tenant_id,
        ),
    )


@pytest.mark.regression("BUG-059")
def test_benchmark_context_passes_quote_search_request_gate():
    from benchmarks.longmemeval.vc_runner import benchmark_speaker_context

    engine = _engine_double("bench-6d550036")
    ctx = benchmark_speaker_context(engine, "What hotel did we pick?")

    assert ctx.eligible
    assert ctx.owner_conversation_id == "bench-6d550036"
    assert ctx.audience_conversation_id == "bench-6d550036"
    assert ctx.audience_channel_scope == "channel"
    assert ctx.audience_channel_id == ""
    assert _speaker_request_scope_is_valid(ctx, "bench-6d550036")


@pytest.mark.regression("BUG-059")
def test_benchmark_context_is_scoped_to_its_own_conversation():
    from benchmarks.longmemeval.vc_runner import benchmark_speaker_context

    ctx = benchmark_speaker_context(_engine_double("bench-a"), "q")
    assert not _speaker_request_scope_is_valid(ctx, "bench-b")


@pytest.mark.regression("BUG-059")
def test_ineligible_sentinel_still_fails_the_gate():
    ctx = SpeakerRetrievalContext.ineligible()
    assert not _speaker_request_scope_is_valid(ctx, "bench-a")
