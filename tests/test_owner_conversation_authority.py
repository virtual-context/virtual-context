"""The owner-conversation authority constructor (the DM shape).

A single-participant conversation addressed as itself has a proved
route with no channel dimension. ``for_owner_conversation`` builds that
authority in one sanctioned place: audience equals the owner, empty
channel, exact-match channel scope. A blank owner yields the explicit
ineligible sentinel — never a partial context. Callers must have proved
the route (resolver owner routing) before constructing it; the
constructor itself stays pure.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

from virtual_context.core.quote_search import (
    _candidate_is_in_speaker_request_scope,
    _speaker_request_scope_is_valid,
)
from virtual_context.types import SpeakerRetrievalContext


def test_owner_conversation_context_shape():
    ctx = SpeakerRetrievalContext.for_owner_conversation(
        "tenant-1", "sk:agent:coach:web:direct:abc",
        requester_actor_id="actor:web:9",
        original_active_user_text="what was my program",
    )
    assert ctx.eligible
    assert ctx.owner_conversation_id == "sk:agent:coach:web:direct:abc"
    assert ctx.audience_conversation_id == "sk:agent:coach:web:direct:abc"
    assert ctx.audience_channel_id == ""
    assert ctx.audience_channel_scope == "channel"
    assert ctx.request_origin_channel_id == ""
    assert ctx.requester_actor_id == "actor:web:9"
    assert ctx.original_active_user_text == "what was my program"


def test_blank_owner_yields_the_ineligible_sentinel():
    for blank in ("", "   ", None):
        ctx = SpeakerRetrievalContext.for_owner_conversation(
            "tenant-1", blank or "",
        )
        assert not ctx.eligible
        assert ctx.owner_conversation_id == ""
        assert ctx.audience_conversation_id == ""


def test_owner_conversation_context_passes_the_request_gate():
    conv = "sk:agent:coach:web:direct:abc"
    ctx = SpeakerRetrievalContext.for_owner_conversation("t", conv)
    assert _speaker_request_scope_is_valid(ctx, conv)
    # And only for its own conversation.
    assert not _speaker_request_scope_is_valid(ctx, "some-other-conv")


def _turn_candidate(conv: str, *, audience: str, version: int, channel: str):
    return SimpleNamespace(
        source_scope="turn",
        provenance=SimpleNamespace(
            conversation_id=conv,
            audience_conversation_id=audience,
            audience_attribution_version=version,
            origin_channel_id=channel,
        ),
    )


def test_candidate_admission_is_empty_channel_exact():
    """Scope 'channel' with an empty request channel admits only rows with
    no channel provenance: guild and channel rows stay invisible."""
    conv = "sk:agent:coach:web:direct:abc"
    ctx = SpeakerRetrievalContext.for_owner_conversation("t", conv)

    no_channel = _turn_candidate(conv, audience=conv, version=1, channel="")
    assert _candidate_is_in_speaker_request_scope(no_channel, ctx)

    guild_row = _turn_candidate(conv, audience=conv, version=1, channel="15249")
    assert not _candidate_is_in_speaker_request_scope(guild_row, ctx)

    unstamped = _turn_candidate(conv, audience="", version=0, channel="")
    assert not _candidate_is_in_speaker_request_scope(unstamped, ctx)
