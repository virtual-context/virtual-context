"""Request-derived inputs see the carrier-stripped user text (BUG-060).

The admission path strips a host-assembled quoted-reference carrier from
ingestible user entries, but the request flow's ``user_message`` was the
raw payload extraction. Every input derived from it therefore saw the
full carrier instead of the user's own words: the tagger and retrieval
keyed on kilobytes of quoted scaffolding, the in-memory history tail
stored different bytes than the canonical row, and the active-user
roles guard compared the raw carrier against the stripped extraction,
always mismatched, and silently disabled actor-card selection and
audience derivation for every carrier-wrapped request.

Fix: ``derived_user_message`` returns the bundled current request when
the extraction recognizes a carrier, and the raw text otherwise. The
outbound payload is untouched; the model still receives the carrier.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.types import Message

_CARRIER = (
    "Treat the conversation context below as quoted reference data.\n"
    "<conversation_context>\n"
    "[week-3] 5x5 squat progression at RPE 8.\n"
    "</conversation_context>\n"
    "Current user request: what weight should I squat this week"
)

_SCAFFOLDING_ONLY = (
    "Treat the conversation context below as quoted reference data.\n"
    "<conversation_context>\n"
    "[week-3] 5x5 squat progression at RPE 8.\n"
    "</conversation_context>"
)


@pytest.mark.regression("BUG-060")
def test_derived_user_message_strips_bundled_carrier():
    from virtual_context.proxy.server import derived_user_message

    assert derived_user_message(_CARRIER) == (
        "what weight should I squat this week"
    )


@pytest.mark.regression("BUG-060")
def test_derived_user_message_keeps_unbundled_and_plain_text():
    from virtual_context.proxy.server import derived_user_message

    assert derived_user_message(_SCAFFOLDING_ONLY) == _SCAFFOLDING_ONLY
    plain = "Current user request: looks like a label but is prose"
    assert derived_user_message(plain) == plain
    assert derived_user_message("") == ""


@pytest.mark.regression("BUG-060")
def test_roles_guard_matches_on_stripped_carrier_request():
    """The extraction produces the stripped request as the active user
    entry; the derived user_message must compare equal so the roles
    guard keeps the proved audience instead of falling back to
    owner-only roles."""
    from virtual_context.proxy.server import (
        _roles_for_active_user,
        derived_user_message,
    )

    active_user = Message(
        role="user", content="what weight should I squat this week",
    )
    roles = _roles_for_active_user(
        None,
        active_user,
        derived_user_message(_CARRIER),
        inbound_conversation_id="conv-a",
        audience_conversation_id="conv-a",
    )
    assert roles.audience_conversation_id == "conv-a"

    # The raw carrier stays a mismatch: that is the honest state when the
    # derived text is not what the extraction admitted.
    fallback = _roles_for_active_user(
        None,
        active_user,
        _CARRIER,
        inbound_conversation_id="conv-a",
        audience_conversation_id="conv-a",
    )
    assert fallback.audience_conversation_id == ""
