"""SpeakerRetrievalContext: request-owned authority semantics."""

import dataclasses

import pytest

from virtual_context.types import RequestRoles, SpeakerRetrievalContext


def _roles(**kw):
    base = dict(
        requester_actor_id="actor:discord:111",
        owner_conversation_id="owner-1",
        audience_conversation_id="aud-1",
        audience_channel_id="chan-1",
    )
    base.update(kw)
    return RequestRoles(**base)


def test_from_roles_carries_trusted_fields():
    ctx = SpeakerRetrievalContext.from_roles("t1", _roles(), "what did I say?")
    assert ctx.tenant_id == "t1"
    assert ctx.owner_conversation_id == "owner-1"
    assert ctx.audience_conversation_id == "aud-1"
    assert ctx.audience_channel_id == "chan-1"
    assert ctx.requester_actor_id == "actor:discord:111"
    assert ctx.original_active_user_text == "what did I say?"
    assert ctx.eligible


def test_unproved_audience_is_ineligible_and_never_owner():
    ctx = SpeakerRetrievalContext.from_roles(
        "t1", _roles(audience_conversation_id=""), "hi",
    )
    assert not ctx.eligible
    assert ctx.audience_conversation_id == ""
    assert ctx.owner_conversation_id == "owner-1"


def test_ineligible_constructor_is_empty_and_ineligible():
    ctx = SpeakerRetrievalContext.ineligible()
    assert not ctx.eligible
    assert ctx.tenant_id == ""


def test_context_is_immutable():
    ctx = SpeakerRetrievalContext.ineligible()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.tenant_id = "x"


def test_repr_never_leaks_actor_id_or_user_text():
    ctx = SpeakerRetrievalContext.from_roles(
        "t1", _roles(), "private question about dosage",
    )
    rendered = repr(ctx)
    assert "actor:discord:111" not in rendered
    assert "dosage" not in rendered
