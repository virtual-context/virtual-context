"""Channel allowlist, candidate collection, and two-source verification.

The pipeline's integrity rests on one rule: a candidate is one member's own
words, proved twice, or it is not a candidate. Attribution is verified by
cross-checking the canonical row against the independently written source
record minted by the trusted adapter at ingest. Any disagreement rejects the
candidate — nothing is ever repaired into agreement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.core.discord_snowflake import datetime_to_snowflake_floor
from virtual_context.core.engagement import (
    ChannelAllowlist,
    MessageSourceRecord,
    collect_candidates,
    load_channel_allowlist,
    verify_candidates,
)
from virtual_context.types import QuoteResult, SourceProvenance

GUILD = "sk:agent:vast:discord:guild:1524917037191925871"
P3PTIDES = "1524917968440524990"
GENERAL = "1524917037787250834"
VASTTEST = "1524946242499514418"
THOTS = "1524926343047811142"

BIGTEX = "actor:discord:1338726888809697364"
ROO = "actor:discord:1485681229608259666"

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)


def _id_at(moment: datetime) -> str:
    return str(datetime_to_snowflake_floor(moment) + 7)


ALLOWLIST_CONFIG = {
    "source_channel_ids": [P3PTIDES, GENERAL],
    "post_channel_ids": [P3PTIDES, GENERAL, VASTTEST],
    "labels": {P3PTIDES: "#p3ptides", GENERAL: "#general", VASTTEST: "#vasttest"},
}


def _quote(
    *,
    turn_id: str,
    message_id: str,
    actor: str = BIGTEX,
    channel: str = P3PTIDES,
    role: str = "requester",
    text: str = "Adding ss31 (5mg) for 4 weeks.",
) -> QuoteResult:
    return QuoteResult(
        text=text, tag="peptides", segment_ref=turn_id,
        source_scope="turn", matched_side="user",
        provenance=SourceProvenance(
            conversation_id=GUILD,
            canonical_turn_id=turn_id,
            source_role=role,
            actor_id=actor,
            audience_conversation_id=GUILD,
            audience_attribution_version=1,
            origin_channel_id=channel,
            source_message_id=message_id,
        ),
    )


def _source(
    *,
    turn_id: str,
    message_id: str,
    author: str = "1338726888809697364",
    channel: str = P3PTIDES,
    actor: str = BIGTEX,
) -> MessageSourceRecord:
    return MessageSourceRecord(
        canonical_turn_id=turn_id,
        message_id=message_id,
        channel_id=channel,
        guild_id="1524917037191925871",
        author_id=author,
        source_actor_id=actor,
    )


# ---------------------------------------------------------------- Task 5


class TestChannelAllowlist:
    def test_source_and_post_lists_are_separate(self):
        allow = load_channel_allowlist(ALLOWLIST_CONFIG)
        assert allow.may_source(P3PTIDES)
        assert allow.may_post(P3PTIDES)
        # The private test channel is a valid target and never a source.
        assert allow.may_post(VASTTEST)
        assert not allow.may_source(VASTTEST)

    def test_an_excluded_channel_can_never_be_selected(self):
        allow = load_channel_allowlist(ALLOWLIST_CONFIG)
        assert not allow.may_source(THOTS)
        assert not allow.may_post(THOTS)

    def test_a_renamed_channel_still_resolves(self):
        """Labels drift; ids do not. One id has carried two labels already."""
        allow = load_channel_allowlist(ALLOWLIST_CONFIG)
        assert allow.may_source(P3PTIDES)
        # The label is display-only and cannot be used for membership.
        assert allow.label_for(P3PTIDES) == "#p3ptides"
        assert allow.label_for("9999") == ""
        assert not allow.may_source("#p3ptides")

    def test_a_post_only_channel_is_not_promoted_by_appearing_in_labels(self):
        allow = load_channel_allowlist(ALLOWLIST_CONFIG)
        assert VASTTEST in allow.post_channel_ids
        assert VASTTEST not in allow.source_channel_ids

    def test_empty_config_allows_nothing(self):
        allow = load_channel_allowlist({})
        assert not allow.may_source(P3PTIDES)
        assert not allow.may_post(P3PTIDES)

    def test_allowlist_is_immutable(self):
        allow = load_channel_allowlist(ALLOWLIST_CONFIG)
        assert isinstance(allow, ChannelAllowlist)
        with pytest.raises((AttributeError, TypeError)):
            allow.source_channel_ids = frozenset()


# ---------------------------------------------------------------- Task 6


class TestCandidateCollection:
    def test_a_subject_lane_row_never_becomes_a_candidate(self):
        """Content authored by someone else where the member was replied to."""
        rows = [
            _quote(turn_id="ct-1", message_id=_id_at(NOW), role="subject"),
        ]
        kept, rejected = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        assert kept == []
        assert [r.reason for r in rejected] == ["not_authored_by_actor"]

    def test_requester_rows_are_collected(self):
        rows = [_quote(turn_id="ct-1", message_id=_id_at(NOW))]
        kept, rejected = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        assert [c.canonical_turn_id for c in kept] == ["ct-1"]
        assert rejected == []

    def test_rows_outside_the_source_allowlist_are_rejected(self):
        rows = [
            _quote(turn_id="ct-1", message_id=_id_at(NOW), channel=VASTTEST),
            _quote(turn_id="ct-2", message_id=_id_at(NOW), channel=THOTS),
        ]
        kept, rejected = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        assert kept == []
        assert {r.reason for r in rejected} == {"channel_not_sourceable"}

    def test_duplicates_are_deduped_by_source_message_id(self):
        mid = _id_at(NOW)
        rows = [
            _quote(turn_id="ct-1", message_id=mid),
            _quote(turn_id="ct-2", message_id=mid),
        ]
        kept, rejected = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        assert len(kept) == 1
        assert [r.reason for r in rejected] == ["duplicate_source_message_id"]

    def test_a_row_without_a_message_id_is_rejected(self):
        rows = [_quote(turn_id="ct-1", message_id="")]
        kept, rejected = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        assert kept == []
        assert [r.reason for r in rejected] == ["no_source_message_id"]

    def test_send_time_comes_from_the_message_id(self):
        sent = NOW - timedelta(days=3)
        rows = [_quote(turn_id="ct-1", message_id=_id_at(sent))]
        kept, _ = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        assert kept[0].sent_at == sent

    def test_every_rejection_names_its_candidate_and_stage(self):
        rows = [_quote(turn_id="ct-1", message_id=_id_at(NOW), role="subject")]
        _, rejected = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        assert rejected[0].canonical_turn_id == "ct-1"
        assert rejected[0].stage == "collect"


# ---------------------------------------------------------------- Task 7


class TestTwoSourceVerification:
    def _candidates(self, **kw):
        rows = [_quote(**kw)]
        kept, _ = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
        )
        return kept

    def test_matching_records_verify(self):
        mid = _id_at(NOW)
        kept = self._candidates(turn_id="ct-1", message_id=mid)
        verified, rejected = verify_candidates(
            kept, {"ct-1": _source(turn_id="ct-1", message_id=mid)},
        )
        assert [c.canonical_turn_id for c in verified] == ["ct-1"]
        assert rejected == []

    def test_a_candidate_with_no_source_record_is_rejected(self):
        kept = self._candidates(turn_id="ct-1", message_id=_id_at(NOW))
        verified, rejected = verify_candidates(kept, {})
        assert verified == []
        assert rejected[0].reason == "no_attested_source"

    def test_disagreeing_author_is_rejected_not_repaired(self):
        """The exact conflation this system exists to prevent."""
        mid = _id_at(NOW)
        kept = self._candidates(turn_id="ct-1", message_id=mid, actor=BIGTEX)
        verified, rejected = verify_candidates(
            kept,
            {"ct-1": _source(
                turn_id="ct-1", message_id=mid,
                author="1485681229608259666", actor=ROO,
            )},
        )
        assert verified == []
        assert rejected[0].reason == "author_mismatch"
        # And the candidate is not rewritten to the source's actor.
        assert kept[0].actor_id == BIGTEX

    def test_disagreeing_channel_is_rejected(self):
        mid = _id_at(NOW)
        kept = self._candidates(turn_id="ct-1", message_id=mid)
        verified, rejected = verify_candidates(
            kept, {"ct-1": _source(
                turn_id="ct-1", message_id=mid, channel=GENERAL,
            )},
        )
        assert verified == []
        assert rejected[0].reason == "channel_mismatch"

    def test_disagreeing_message_id_is_rejected(self):
        kept = self._candidates(turn_id="ct-1", message_id=_id_at(NOW))
        verified, rejected = verify_candidates(
            kept, {"ct-1": _source(
                turn_id="ct-1", message_id=_id_at(NOW - timedelta(days=5)),
            )},
        )
        assert verified == []
        assert rejected[0].reason == "message_id_mismatch"

    def test_two_records_assigning_different_speakers_reject_both(self):
        """The spec's named contradiction case."""
        mid = _id_at(NOW)
        rows = [
            _quote(turn_id="ct-1", message_id=mid, actor=BIGTEX),
            _quote(turn_id="ct-2", message_id=mid, actor=ROO),
        ]
        kept, _ = collect_candidates(
            rows, allowlist=load_channel_allowlist(ALLOWLIST_CONFIG),
            dedupe=False,
        )
        verified, rejected = verify_candidates(
            kept,
            {
                "ct-1": _source(turn_id="ct-1", message_id=mid, actor=BIGTEX),
                "ct-2": _source(
                    turn_id="ct-2", message_id=mid,
                    author="1485681229608259666", actor=ROO,
                ),
            },
        )
        assert verified == []
        assert {r.reason for r in rejected} == {"contradictory_speaker"}

    def test_verification_never_mutates_a_candidate(self):
        mid = _id_at(NOW)
        kept = self._candidates(turn_id="ct-1", message_id=mid)
        before = (kept[0].actor_id, kept[0].channel_id, kept[0].text)
        verify_candidates(
            kept, {"ct-1": _source(
                turn_id="ct-1", message_id=mid, channel=GENERAL,
            )},
        )
        assert (kept[0].actor_id, kept[0].channel_id, kept[0].text) == before

    def test_every_rejection_names_the_verify_stage(self):
        kept = self._candidates(turn_id="ct-1", message_id=_id_at(NOW))
        _, rejected = verify_candidates(kept, {})
        assert rejected[0].stage == "verify"
        assert rejected[0].canonical_turn_id == "ct-1"
