"""Audience-scoped speaker label resolution and source-class projection.

Labels come only from the most recent audience-admissible physical
canonical USER row per actor with a deterministic physical tiebreak —
never from tenant-global actor profiles, never from another audience's
rows, and never widened past a durable channel. The projectors are
allowlists: internal actor ids can never appear in any projected field.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dataclasses

import pytest

from virtual_context.core.speaker_labels import (
    ASSISTANT_SPEAKER_LABEL,
    annotate_aggregate_entry,
    annotation_speaker_context,
    collect_fact_author_actor_ids,
    collect_quote_actor_ids,
    fact_attribution_basis,
    project_fact_speaker_fields,
    project_quote_speaker_fields,
    resolve_speaker_labels,
    strip_to_structural_speaker_fields,
)
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    CanonicalTurnRow,
    Fact,
    SearchConfig,
    SourceProvenance,
    SpeakerRetrievalContext,
)

GUILD = "aud-guild"
DM = "aud-dm"
OWNER = "owner-conv"
ACTOR = "actor:discord:111"
OTHER_ACTOR = "actor:discord:222"


def _ctx(audience: str = GUILD, channel: str = "", owner: str = OWNER):
    return SpeakerRetrievalContext(
        tenant_id="t",
        owner_conversation_id=owner,
        audience_conversation_id=audience,
        audience_channel_id=channel,
    )


def _row(
    *,
    actor: str = ACTOR,
    sender: str = "Sania",
    audience: str = GUILD,
    version: int = AUDIENCE_ATTRIBUTION_VERSION,
    sort_key: float = 1000.0,
    ctid: str = "ct-1",
    user: str = "hello there",
    channel: str = "",
    conv: str = OWNER,
) -> CanonicalTurnRow:
    return CanonicalTurnRow(
        conversation_id=conv,
        canonical_turn_id=ctid,
        sort_key=sort_key,
        user_content=user,
        sender=sender,
        sender_actor_id=actor,
        audience_conversation_id=audience,
        audience_attribution_version=version,
        origin_channel_id=channel,
    )


class _RowStore:
    """Store double exposing ONLY the recent-rows read the resolver uses.

    Deliberately has no ``get_actor_profile`` / ``actor_profiles`` surface,
    so resolution succeeding against it proves labels never consult the
    tenant-global profile name.
    """

    def __init__(self, rows):
        self.rows = list(rows)
        self.calls: list[tuple[str, int]] = []

    def get_recent_canonical_turns(self, conversation_id, *, limit):
        self.calls.append((conversation_id, limit))
        return [r for r in self.rows if r.conversation_id == conversation_id]


class TestResolveSpeakerLabels:
    def test_most_recent_admissible_row_wins(self):
        store = _RowStore([
            _row(sender="Old Name", sort_key=1000.0, ctid="ct-a"),
            _row(sender="New Name", sort_key=2000.0, ctid="ct-b"),
        ])
        labels = resolve_speaker_labels(store, {ACTOR}, speaker_context=_ctx())
        assert labels == {ACTOR: "New Name"}

    def test_equal_sort_key_breaks_on_physical_row_id(self):
        rows = [
            _row(sender="Alpha", sort_key=1000.0, ctid="ct-a"),
            _row(sender="Beta", sort_key=1000.0, ctid="ct-b"),
        ]
        forward = resolve_speaker_labels(
            _RowStore(rows), {ACTOR}, speaker_context=_ctx(),
        )
        reverse = resolve_speaker_labels(
            _RowStore(list(reversed(rows))), {ACTOR}, speaker_context=_ctx(),
        )
        # Deterministic regardless of the store's return order.
        assert forward == reverse == {ACTOR: "Beta"}

    def test_dm_label_never_surfaces_in_guild(self):
        store = _RowStore([
            _row(sender="Sania", audience=GUILD, sort_key=1000.0, ctid="ct-g"),
            # The DM rename is NEWER but belongs to the other audience.
            _row(sender="SnookieBear", audience=DM, sort_key=9000.0, ctid="ct-d"),
        ])
        guild = resolve_speaker_labels(store, {ACTOR}, speaker_context=_ctx(GUILD))
        dm = resolve_speaker_labels(store, {ACTOR}, speaker_context=_ctx(DM))
        assert guild == {ACTOR: "Sania"}
        assert dm == {ACTOR: "SnookieBear"}

    def test_dm_only_actor_has_no_label_in_guild(self):
        store = _RowStore([
            _row(actor=OTHER_ACTOR, sender="SnookieBear", audience=DM),
        ])
        labels = resolve_speaker_labels(
            store, {OTHER_ACTOR}, speaker_context=_ctx(GUILD),
        )
        assert labels == {}

    def test_stale_attribution_version_rows_are_inadmissible(self):
        store = _RowStore([
            _row(sender="Current", sort_key=1000.0, ctid="ct-a"),
            _row(sender="Stale", sort_key=5000.0, ctid="ct-b", version=0),
        ])
        labels = resolve_speaker_labels(store, {ACTOR}, speaker_context=_ctx())
        assert labels == {ACTOR: "Current"}

    def test_durable_channel_requires_exact_match_and_empty_fails_closed(self):
        store = _RowStore([
            _row(sender="Wrong", sort_key=3000.0, ctid="ct-a", channel="chan-2"),
            _row(sender="Blank", sort_key=2500.0, ctid="ct-b", channel=""),
            _row(sender="Right", sort_key=1000.0, ctid="ct-c", channel="chan-1"),
        ])
        labels = resolve_speaker_labels(
            store, {ACTOR}, speaker_context=_ctx(channel="chan-1"),
        )
        assert labels == {ACTOR: "Right"}

    def test_assistant_lane_rows_never_supply_labels(self):
        store = _RowStore([
            _row(sender="Ghost", user="", sort_key=9000.0, ctid="ct-a"),
        ])
        assert resolve_speaker_labels(store, {ACTOR}, speaker_context=_ctx()) == {}

    def test_most_recent_admissible_row_with_empty_sender_stays_empty(self):
        # No fallback to an older row's label: the most recent admissible
        # row decides, and its empty label stays empty.
        store = _RowStore([
            _row(sender="Older", sort_key=1000.0, ctid="ct-a"),
            _row(sender="", sort_key=2000.0, ctid="ct-b"),
        ])
        assert resolve_speaker_labels(store, {ACTOR}, speaker_context=_ctx()) == {}

    def test_ineligible_or_missing_context_reads_nothing(self):
        store = _RowStore([_row()])
        assert resolve_speaker_labels(store, {ACTOR}, speaker_context=None) == {}
        assert resolve_speaker_labels(
            store, {ACTOR},
            speaker_context=SpeakerRetrievalContext(
                tenant_id="t", owner_conversation_id=OWNER,
            ),
        ) == {}
        assert store.calls == []

    def test_empty_actor_set_reads_nothing(self):
        store = _RowStore([_row()])
        assert resolve_speaker_labels(store, set(), speaker_context=_ctx()) == {}
        assert resolve_speaker_labels(store, {""}, speaker_context=_ctx()) == {}
        assert store.calls == []

    def test_store_failure_fails_open_to_empty_labels(self):
        class _Broken:
            def get_recent_canonical_turns(self, conversation_id, *, limit):
                raise RuntimeError("db down")

        assert resolve_speaker_labels(
            _Broken(), {ACTOR}, speaker_context=_ctx(),
        ) == {}

    def test_scan_is_bounded_and_scoped_to_the_owner(self):
        store = _RowStore([_row()])
        resolve_speaker_labels(
            store, {ACTOR}, speaker_context=_ctx(), scan_limit=7,
        )
        assert store.calls == [(OWNER, 7)]

    def test_actor_profiles_are_never_consulted(self):
        # _RowStore has no profile surface at all; resolution still works.
        store = _RowStore([_row(sender="Scoped Name")])
        labels = resolve_speaker_labels(store, {ACTOR}, speaker_context=_ctx())
        assert labels == {ACTOR: "Scoped Name"}


class TestAnnotationGateRouter:
    def _config(self, enabled: bool):
        return type("Cfg", (), {
            "search": SearchConfig(speaker_annotations_enabled=enabled),
        })()

    def test_gate_off_disables_annotation(self):
        assert annotation_speaker_context(self._config(False), _ctx()) is None

    def test_gate_on_requires_a_proved_audience(self):
        ineligible = SpeakerRetrievalContext(
            tenant_id="t", owner_conversation_id=OWNER,
        )
        assert annotation_speaker_context(self._config(True), ineligible) is None
        assert annotation_speaker_context(self._config(True), None) is None

    def test_gate_on_forwards_the_exact_eligible_context(self):
        context = _ctx()
        assert annotation_speaker_context(self._config(True), context) is context


def _prov(role: str, actor: str = "", claimed: str = "") -> SourceProvenance:
    return SourceProvenance(
        conversation_id=OWNER,
        canonical_turn_id="ct-1",
        source_role=role,
        actor_id=actor,
        audience_conversation_id=GUILD,
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        claimed_subject_label=claimed,
    )


class TestQuoteProjection:
    def test_requester_lane_with_scoped_label(self):
        fields = project_quote_speaker_fields(
            _prov("requester", ACTOR), {ACTOR: "Sania"},
        )
        assert fields == {
            "source_role": "requester",
            "speaker_label": "Sania",
            "speaker_handle": "",
            "speaker_actor_known": True,
            "speaker_verified": True,
        }

    def test_known_actor_without_scoped_label_stays_empty(self):
        fields = project_quote_speaker_fields(_prov("requester", ACTOR), {})
        assert fields["speaker_label"] == ""
        assert fields["speaker_actor_known"] is True
        assert fields["speaker_handle"] == ""

    def test_requester_lane_without_actor_is_unverified(self):
        fields = project_quote_speaker_fields(_prov("requester"), {})
        assert fields == {
            "source_role": "requester",
            "speaker_label": "",
            "speaker_handle": "",
            "speaker_actor_known": False,
            "speaker_verified": False,
        }

    def test_subject_lane_uses_only_the_subject_actor(self):
        fields = project_quote_speaker_fields(
            _prov("subject", OTHER_ACTOR), {OTHER_ACTOR: "Bast"},
        )
        assert fields["speaker_label"] == "Bast"
        assert fields["source_role"] == "subject"
        assert fields["speaker_verified"] is True

    def test_unresolved_reply_surfaces_only_an_unverified_claim(self):
        fields = project_quote_speaker_fields(
            _prov("subject", claimed="someone typed this"), {},
        )
        assert fields["speaker_label"] == ""
        assert fields["speaker_handle"] == ""
        assert fields["speaker_verified"] is False
        assert fields["claimed_speaker_label"] == "someone typed this"

    def test_assistant_lane_uses_the_reserved_identity(self):
        fields = project_quote_speaker_fields(_prov("assistant"), {})
        assert fields == {
            "source_role": "assistant",
            "speaker_label": ASSISTANT_SPEAKER_LABEL,
            "speaker_handle": "",
            "speaker_actor_known": False,
            "speaker_verified": True,
        }

    def test_mixed_and_unattributed_expose_scope_not_a_speaker(self):
        mixed = project_quote_speaker_fields(_prov("mixed"), {})
        assert mixed == {"source_role": "mixed", "speaker_scope": "mixed"}
        unattributed = project_quote_speaker_fields(_prov("unattributed"), {})
        assert unattributed == {
            "source_role": "unattributed",
            "speaker_scope": "unknown",
        }

    def test_missing_provenance_projects_nothing(self):
        assert project_quote_speaker_fields(None, {}) == {}

    def test_actor_id_never_appears_in_projected_values(self):
        for role in ("requester", "subject", "assistant", "mixed"):
            fields = project_quote_speaker_fields(_prov(role, ACTOR), {})
            assert ACTOR not in repr(fields)

    def test_collect_quote_actor_ids_is_role_local(self):
        wanted = collect_quote_actor_ids([
            _prov("requester", ACTOR),
            _prov("subject", OTHER_ACTOR),
            _prov("assistant", "actor:should:never-count"),
            _prov("mixed", "actor:should:never-count"),
            None,
        ])
        assert wanted == {ACTOR, OTHER_ACTOR}


def _fact(version: int, role: str = "", actor: str = "") -> Fact:
    return Fact(
        subject="user",
        verb="visited",
        object="boston",
        what="user visited boston",
        author_actor_id=actor,
        author_attribution_version=version,
        author_source_role=role,
    )


class TestFactProjection:
    def test_version_zero_is_unattributed(self):
        fact = _fact(0)
        assert fact_attribution_basis(fact) == "unattributed"
        fields = project_fact_speaker_fields(fact, {})
        assert fields == {
            "author_attribution_version": 0,
            "attribution_basis": "unattributed",
        }

    def test_version_one_is_visibly_model_assisted_without_speaker_fields(self):
        fact = _fact(1, role="requester", actor=ACTOR)
        fields = project_fact_speaker_fields(fact, {ACTOR: "Sania"})
        assert fields == {
            "author_attribution_version": 1,
            "attribution_basis": "model_assisted",
            "source_role": "requester",
        }
        assert "speaker_label" not in fields
        assert "speaker_verified" not in fields

    def test_version_two_complete_lane_is_role_local_and_verified(self):
        fact = _fact(2, role="requester", actor=ACTOR)
        fields = project_fact_speaker_fields(fact, {ACTOR: "Sania"})
        assert fields == {
            "author_attribution_version": 2,
            "attribution_basis": "role_local",
            "source_role": "requester",
            "speaker_label": "Sania",
            "speaker_handle": "",
            "speaker_actor_known": True,
            "speaker_verified": True,
        }

    def test_version_two_empty_actor_stays_unattributed(self):
        fact = _fact(2, role="subject", actor="")
        fields = project_fact_speaker_fields(fact, {})
        assert fields["attribution_basis"] == "unattributed"
        assert "speaker_label" not in fields

    def test_version_two_assistant_lane_never_becomes_a_human_speaker(self):
        fact = _fact(2, role="assistant", actor=ACTOR)
        fields = project_fact_speaker_fields(fact, {ACTOR: "Sania"})
        assert fields["attribution_basis"] == "unattributed"
        assert "speaker_label" not in fields

    def test_collect_fact_author_actor_ids_takes_only_role_local(self):
        wanted = collect_fact_author_actor_ids([
            _fact(2, role="requester", actor=ACTOR),
            _fact(2, role="subject", actor=OTHER_ACTOR),
            _fact(1, role="requester", actor="actor:no:model-assisted"),
            _fact(2, role="assistant", actor="actor:no:assistant"),
            _fact(0),
        ])
        assert wanted == {ACTOR, OTHER_ACTOR}

    def test_actor_id_never_appears_in_projected_values(self):
        fields = project_fact_speaker_fields(
            _fact(2, role="requester", actor=ACTOR), {},
        )
        assert ACTOR not in repr(fields)


class TestAggregateAndStatelessHelpers:
    def test_aggregate_entry_gets_unknown_scope_once(self):
        entry: dict = {"excerpt": "x"}
        annotate_aggregate_entry(entry)
        assert entry["speaker_scope"] == "unknown"
        annotate_aggregate_entry(entry, scope="mixed")
        assert entry["speaker_scope"] == "unknown"

    def test_aggregate_marker_never_overrides_singular_projection(self):
        entry = {"excerpt": "x", "speaker_label": "Sania"}
        annotate_aggregate_entry(entry)
        assert "speaker_scope" not in entry

    def test_non_dict_entries_pass_through(self):
        assert annotate_aggregate_entry("plain") == "plain"

    def test_stateless_strip_keeps_only_structural_fields(self):
        payload = {
            "results": [{
                "excerpt": "x",
                "source_role": "requester",
                "speaker_label": "Sania",
                "speaker_handle": "sania",
                "speaker_actor_known": True,
                "speaker_verified": True,
                "claimed_speaker_label": "claim",
                "speakers": ["Sania", "Bast"],
                "attribution_basis": "role_local",
                "author_attribution_version": 2,
            }],
            "speaker_scope": "mixed",
        }
        strip_to_structural_speaker_fields(payload)
        entry = payload["results"][0]
        assert entry["speaker_label"] == ""
        assert entry["speaker_handle"] == ""
        assert "speaker_actor_known" not in entry
        assert "speaker_verified" not in entry
        assert "claimed_speaker_label" not in entry
        assert "speakers" not in entry
        assert entry["source_role"] == "requester"
        assert entry["attribution_basis"] == "role_local"
        assert entry["author_attribution_version"] == 2
        assert payload["speaker_scope"] == "mixed"


class TestSQLiteBackedLabelResolution:
    """One end-to-end pass over the real SQLite recent-rows read."""

    def test_labels_resolve_from_persisted_rows_per_audience(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=str(tmp_path / "labels.db"))
        common = dict(
            conversation_id=OWNER,
            assistant_content="ok",
            audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        )
        store.save_canonical_turn(
            turn_number=0,
            user_content="guild hello",
            sender="Sania",
            sender_actor_id=ACTOR,
            audience_conversation_id=GUILD,
            **common,
        )
        store.save_canonical_turn(
            turn_number=1,
            user_content="dm hello",
            sender="SnookieBear",
            sender_actor_id=ACTOR,
            audience_conversation_id=DM,
            **common,
        )
        store.save_canonical_turn(
            turn_number=2,
            user_content="dm only",
            sender="Hidden Person",
            sender_actor_id=OTHER_ACTOR,
            audience_conversation_id=DM,
            **common,
        )

        guild_labels = resolve_speaker_labels(
            store, {ACTOR, OTHER_ACTOR}, speaker_context=_ctx(GUILD),
        )
        assert guild_labels == {ACTOR: "Sania"}

        dm_labels = resolve_speaker_labels(
            store, {ACTOR, OTHER_ACTOR}, speaker_context=_ctx(DM),
        )
        assert dm_labels == {ACTOR: "SnookieBear", OTHER_ACTOR: "Hidden Person"}
