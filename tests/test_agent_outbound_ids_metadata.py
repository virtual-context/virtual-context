"""Agent-authored message identities ride BESIDE the attested claim.

The attested source claim is a fixed set of fields describing one inbound
message, and a sender that folds a time-varying set into it changes the shape
it fingerprints. On the delivery side that costs a real user's turn. So the
identities travel under their own metadata key, read by their own accessor,
unable to touch attestation validation or admission.

Everything here is one-directional: an identity that is present may later
suppress a re-extraction of the agent's own words, and an identity that is
absent, malformed or dropped changes nothing at all.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.types import (
    AGENT_OUTBOUND_IDS_KEY,
    SOURCE_ATTESTATION_KEY,
    get_agent_outbound_ids,
    get_source_attestation,
)


def _ident(**over):
    base = {
        "platform": "discord", "account_id": "acct-1",
        "channel_id": "111", "message_id": "222",
        "observed_at": "2026-08-20T10:00:00Z",
    }
    base.update(over)
    return base


def _attestation():
    return {
        "version": 1, "agent_scope_id": "scope", "platform": "discord",
        "account_id": "acct-1", "message_id": "inbound-1", "channel_id": "111",
        "guild_id": "g", "author_id": "member-1",
        "transport_body_sha256": "a" * 64, "canonical_body_sha256": "b" * 64,
        "projection_version": "1", "reply_target_message_id": "",
    }


def test_the_key_is_a_sibling_of_the_attestation_not_part_of_it():
    """The whole point. Adding or changing identities must leave the attested
    claim byte-identical, or a sender cannot carry them without altering what
    it fingerprints."""
    without = {SOURCE_ATTESTATION_KEY: _attestation()}
    with_ids = dict(without, **{AGENT_OUTBOUND_IDS_KEY: [_ident()]})
    more_ids = dict(without, **{AGENT_OUTBOUND_IDS_KEY: [_ident(), _ident(message_id="333")]})

    assert get_source_attestation(with_ids) == get_source_attestation(without)
    assert get_source_attestation(more_ids) == get_source_attestation(without)
    assert get_source_attestation(without) != {}, "fixture does not attest"


def test_a_malformed_identity_set_cannot_damage_the_attestation():
    """Fail open on the metadata, and never let it reach the attested claim."""
    for junk in ("not-a-list", 42, None, [1, 2, 3], [{"platform": ""}]):
        metadata = {SOURCE_ATTESTATION_KEY: _attestation(), AGENT_OUTBOUND_IDS_KEY: junk}
        assert get_source_attestation(metadata) == get_source_attestation(
            {SOURCE_ATTESTATION_KEY: _attestation()}
        )


def test_entries_are_dropped_individually_rather_than_rejecting_the_set():
    """One bad entry must not discard the good ones beside it."""
    good = _ident(message_id="keep")
    metadata = {AGENT_OUTBOUND_IDS_KEY: [
        good, "junk", {"platform": "discord"}, _ident(message_id=""),
        _ident(channel_id=" 111 "), _ident(message_id="x" * 300),
    ]}

    parsed = get_agent_outbound_ids(metadata)

    assert [entry["message_id"] for entry in parsed] == ["keep"]


def test_platform_is_lowercased_and_non_canonical_values_are_dropped():
    """The reader builds its key from what the inbound path stored, so a value
    a trim would change must never become a key."""
    assert get_agent_outbound_ids(
        {AGENT_OUTBOUND_IDS_KEY: [_ident(platform="Discord")]}
    )[0]["platform"] == "discord"
    for bad in (" 111 ", "111 ", " 111"):
        assert get_agent_outbound_ids(
            {AGENT_OUTBOUND_IDS_KEY: [_ident(channel_id=bad)]}
        ) == []


def test_absence_yields_an_empty_set_and_never_an_error():
    for metadata in (None, {}, {"other": 1}, {AGENT_OUTBOUND_IDS_KEY: []}):
        assert get_agent_outbound_ids(metadata) == []


def test_the_set_is_bounded_so_one_turn_cannot_carry_unbounded_work():
    parsed = get_agent_outbound_ids(
        {AGENT_OUTBOUND_IDS_KEY: [_ident(message_id=str(i)) for i in range(1000)]}
    )
    assert len(parsed) <= 256
