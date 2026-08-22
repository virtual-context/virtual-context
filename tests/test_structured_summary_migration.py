"""Focused safety tests for the structured segment migration command."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from virtual_context.cli.structured_summary_migration_cmd import (
    _CAS_UPDATE_SQL,
    _Inventory,
    _ResumeCursor,
    _Scope,
    _cache_action,
    _canonical_ids,
    _cas_tag_persist,
    _deterministic_tag_envelope,
    _deterministic_tag_selection,
    _envelope_candidate_reason,
    _expected_source_digest,
    _has_bounded_claim_source,
    _json_string_list,
    _build_migration_actor_roster,
    _reconstruct_segment,
    _row_tags,
    _selection_sql,
    _source_validation_reasons,
    _segment_migration_block_reason,
    _structured_source_block_reason,
    _tag_claim_selection_digest,
    _tag_inventory,
    _tag_source_coordinates,
    _update_reason_counts,
    _validate_tag_result,
    _validate_envelope,
    configure_parser,
)
from virtual_context.core.structured_summary import (
    structured_source_digest,
    structured_source_provenance_digest,
)
from virtual_context.core.compactor import build_deterministic_structured_summary
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    STRUCTURED_SUMMARY_MAX_CLAIMS,
    STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS,
    STRUCTURED_SUMMARY_SCHEMA_VERSION,
    SegmentMetadata,
    StoredSummary,
    StructuredSummary,
    SummaryClaim,
    SummarySource,
    TagSummary,
    structured_summary_to_dict,
)


def _source(**updates):
    row = {
        "canonical_turn_id": "ct-1",
        "conversation_id": "owner-1",
        "turn_group_number": 7,
        "user_content": "BigTex stopped tesamorelin after the appointment.",
        "assistant_content": "Understood; the medication is no longer active.",
        "sender": "BigTex",
        "sender_actor_id": "actor:discord:123",
        "session_date": "2026-08-18",
        "audience_conversation_id": "guild-source",
        "origin_channel_id": "channel-9",
        "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
        "first_seen_at": "2026-08-18T12:00:00+00:00",
    }
    row.update(updates)
    return row


def _segment_row(**updates):
    row = {
        "ref": "seg-1",
        "primary_tag": "health",
        "tags": ["health", "medication"],
        "full_tokens": 42,
        "start_timestamp": "2026-08-18T12:00:00+00:00",
        "end_timestamp": "2026-08-18T12:01:00+00:00",
        "metadata_json": {
            "canonical_turn_ids": ["ct-1"],
            "source_mapping_complete": True,
            "turn_count": 1,
            "session_date": "2026-08-18",
        },
        # Poisoned legacy projections must never influence reconstruction.
        "summary": "BigTex currently takes tesamorelin.",
        "full_text": "POISONED STORED FULL TEXT",
        "messages_json": '[{"role":"user","content":"POISON"}]',
    }
    row.update(updates)
    return row


def _provenance_digest(
    source: dict,
    *,
    content: str | None = None,
    canonical_turn_id: str | None = None,
) -> str:
    return structured_source_provenance_digest({
        "canonical_turn_id": canonical_turn_id or source["canonical_turn_id"],
        "source_role": "requester",
        "actor_id": source["sender_actor_id"],
        "speaker_label": source["sender"],
        "content": (content if content is not None else source["user_content"]).strip(),
        "session_date": source["session_date"],
        "audience_conversation_id": source["audience_conversation_id"],
        "origin_channel_id": source["origin_channel_id"],
        "audience_attribution_version": source["audience_attribution_version"],
    })


def _valid_envelope(digest: str, source: dict | None = None) -> StructuredSummary:
    source = source or _source(assistant_content="")
    return StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="test-model",
        claims=(SummaryClaim(
            text="BigTex stopped tesamorelin after the appointment.",
            claim_type="conversation",
            temporal_status="",
            modality="asserted",
            event_time="",
            sources=(SummarySource(
                canonical_turn_id="ct-1",
                source_role="requester",
                speaker_label="BigTex",
                evidence_excerpt=(
                    "BigTex stopped tesamorelin after the appointment."
                ),
                session_date="2026-08-18",
                source_provenance_digest=_provenance_digest(source),
            ),),
        ),),
    )


def test_selection_never_loads_legacy_model_input():
    sql = _selection_sql("seg-0")
    assert "full_text" not in sql
    assert "messages_json" not in sql
    # The old synopsis is available only as a database-side checksum, never
    # as model input or a returned prose column.
    assert "md5(s.summary) AS old_retrieval_synopsis_md5" in sql
    assert "s.summary," not in sql
    assert "xmin::text" in sql
    assert "s.ref > %(after_ref)s" in sql


def test_segment_cas_updates_only_structured_metadata():
    assert "SET metadata_json = jsonb_set" in _CAS_UPDATE_SQL
    assert "'{structured_summary}'" in _CAS_UPDATE_SQL
    assert "SET summary =" not in _CAS_UPDATE_SQL
    assert "summary_tokens" not in _CAS_UPDATE_SQL
    assert "compression_ratio" not in _CAS_UPDATE_SQL
    assert "compaction_model" not in _CAS_UPDATE_SQL
    assert "xmin::text = %s" in _CAS_UPDATE_SQL
    assert "c.lifecycle_epoch = %s" in _CAS_UPDATE_SQL
    assert "cl.generation = %s" in _CAS_UPDATE_SQL
    assert "pending_raw_payload_entries = 0" in _CAS_UPDATE_SQL
    assert "compaction_operation" in _CAS_UPDATE_SQL
    assert "full_text" not in _CAS_UPDATE_SQL
    assert "messages_json" not in _CAS_UPDATE_SQL
    assert "canonical_turn_ids" not in _CAS_UPDATE_SQL


def test_reconstruction_uses_only_exact_canonical_lanes():
    source = _source()
    segment = _reconstruct_segment(_segment_row(), {"ct-1": source})
    assert [message.content for message in segment.messages] == [
        source["user_content"], source["assistant_content"],
    ]
    assert all("POISON" not in message.content for message in segment.messages)
    assert segment.messages[0].source_actor_id == "actor:discord:123"
    assert segment.messages[0].source_audience_conversation_id == "guild-source"
    assert segment.messages[0].metadata["_vc_source_canonical_turn_ids"] == ["ct-1"]


def test_migration_builds_deterministic_claims_without_provider_objects():
    source = _source(assistant_content="")
    segment = _reconstruct_segment(_segment_row(), {"ct-1": source})
    roster = _build_migration_actor_roster(["ct-1"], {"ct-1": source})

    envelope = build_deterministic_structured_summary(
        roster=roster,
        segment=segment,
        generation_model="deterministic-extractive-migration-v1",
    )

    assert [claim.text for claim in envelope.claims] == [source["user_content"]]
    assert envelope.claims[0].sources[0].canonical_turn_id == "ct-1"
    assert envelope.generation_model == "deterministic-extractive-migration-v1"


def test_source_shape_blocks_unrepresentable_segments_without_truncation():
    overlong_ordinary = "ordinary " + (
        "detail " * STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS
    )
    overlong_critical = "I stopped tesamorelin. " + (
        "detail " * STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS
    )

    assert _structured_source_block_reason([
        {"content": overlong_ordinary},
    ]) is None
    assert _structured_source_block_reason([
        {"content": overlong_critical},
    ]) == "critical_source_over_excerpt_limit"
    assert _structured_source_block_reason([
        {"content": f"bounded lane {index}"}
        for index in range(STRUCTURED_SUMMARY_MAX_CLAIMS + 1)
    ]) == "bounded_claim_overflow"
    assert _segment_migration_block_reason(
        ({"content": "bounded requester lane"},),
        retrieval_synopsis_chars=0,
    ) == "retrieval_synopsis_missing"
    assert _segment_migration_block_reason(
        (
            {"content": "I saw actor-b in the record.", "actor_id": "actor-a"},
            {"content": "ordinary lane", "actor_id": "actor-b"},
        ),
        retrieval_synopsis_chars=10,
    ) == "internal_identity_in_source"


def test_digest_matches_public_algorithm_and_binds_scope():
    source = _source()
    digest = _expected_source_digest(
        ["ct-1"], {"ct-1": source}, session_date="2026-08-18",
    )
    expected = structured_source_digest([{
        "canonical_turn_id": "ct-1",
        "source_role": "requester",
        "actor_id": "actor:discord:123",
        "speaker_label": "BigTex",
        "content": source["user_content"],
        "session_date": "2026-08-18",
        "audience_conversation_id": "guild-source",
        "origin_channel_id": "channel-9",
        "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
    }])
    assert digest == expected
    # Prior assistant output is not a trusted v1 source and cannot make an
    # otherwise-current human evidence envelope stale.
    assistant_changed = _source(assistant_content="A totally different reply.")
    assert _expected_source_digest(
        ["ct-1"], {"ct-1": assistant_changed}, session_date="2026-08-18",
    ) == digest
    changed = _source(origin_channel_id="channel-other")
    assert _expected_source_digest(
        ["ct-1"], {"ct-1": changed}, session_date="2026-08-18",
    ) != digest


def test_digest_normalizes_only_outer_requester_lane_whitespace():
    source = _source(user_content="  BigTex stopped tesamorelin.  ")
    digest = _expected_source_digest(
        ["ct-1"], {"ct-1": source}, session_date="2026-08-18",
    )
    expected = structured_source_digest([{
        "canonical_turn_id": "ct-1",
        "source_role": "requester",
        "actor_id": "actor:discord:123",
        "speaker_label": "BigTex",
        "content": "BigTex stopped tesamorelin.",
        "session_date": "2026-08-18",
        "audience_conversation_id": "guild-source",
        "origin_channel_id": "channel-9",
        "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
    }])
    assert digest == expected


def test_source_alias_must_resolve_to_exact_owner():
    rows = {"ct-1": _source()}
    assert _source_validation_reasons(
        ["ct-1"], rows,
        owner_conversation_id="owner-1",
        alias_graph={"guild-source": "owner-1"},
    ) == ()
    assert "canonical_audience_not_owned" in _source_validation_reasons(
        ["ct-1"], rows,
        owner_conversation_id="owner-1",
        alias_graph={"guild-source": "different-owner"},
    )


def test_legacy_assistant_only_row_does_not_gate_requester_claim_audience():
    requester = _source(canonical_turn_id="ct-user")
    assistant_only = _source(
        canonical_turn_id="ct-assistant",
        user_content="",
        assistant_content="Prior model output only.",
        audience_conversation_id="",
        audience_attribution_version=0,
    )

    assert _source_validation_reasons(
        ["ct-user", "ct-assistant"],
        {"ct-user": requester, "ct-assistant": assistant_only},
        owner_conversation_id="owner-1",
        alias_graph={"guild-source": "owner-1"},
    ) == ()


def test_zero_claim_v1_envelope_is_still_a_candidate():
    digest = "a" * 64
    envelope = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="model",
        claims=(),
    )
    assert _envelope_candidate_reason(
        structured_summary_to_dict(envelope), envelope, digest,
    ) == "zero_claim_envelope"


def test_only_over_limit_requester_lanes_are_not_segment_candidates():
    assert _has_bounded_claim_source([{"content": "short exact lane"}]) is True
    assert _has_bounded_claim_source([{"content": "x" * 801}]) is False


def test_tag_migration_blocks_when_any_source_segment_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd."
        "_load_existing_tag_rows",
        lambda _conn, _conversation_id, _tags: {},
    )
    old = {"ref": "seg-old", "tags": ["health"], "primary_tag": "health"}
    overlong_stop = {
        "ref": "seg-stop", "tags": ["health"], "primary_tag": "health",
        "_skip_reasons": ("no_bounded_structured_claim_source",),
    }
    inventory = _Inventory(
        scanned=2,
        candidates=[],
        current_rows=[old],
        skipped_rows=[overlong_stop],
        current=1,
        skipped_reason_counts=Counter({
            "no_bounded_structured_claim_source": 1,
        }),
        candidate_reason_counts=Counter(),
        affected_tags={"health"},
    )

    tags = _tag_inventory(
        object(),
        SimpleNamespace(conversation_id="owner", after_tag=None),
        inventory,
    )

    assert tags.candidates == []
    assert tags.current == 0
    assert tags.source_segment_count == 2
    assert tags.blocked == [{
        "tag": "health",
        "reason": "source_segments_ineligible",
        "ineligible_source_count": 1,
    }]
    assert tags.reason_counts == {"source_segments_ineligible": 1}


def test_skipped_inventory_reasons_increment_without_counter_type_error():
    counts = Counter()
    _update_reason_counts(
        counts,
        ["source_mapping_incomplete", "metadata_invalid", "metadata_invalid"],
    )

    assert counts == {
        "source_mapping_incomplete": 1,
        "metadata_invalid": 1,
    }


def test_current_nonempty_envelope_is_idempotently_skipped():
    source = _source(assistant_content="")
    digest = _expected_source_digest(
        ["ct-1"], {"ct-1": source}, session_date="2026-08-18",
    )
    envelope = _valid_envelope(digest, source)
    assert _envelope_candidate_reason(
        structured_summary_to_dict(envelope), envelope, digest,
    ) is None


def test_envelope_accepts_bigtex_cessation_from_requester_evidence():
    source = _source(assistant_content="")
    rows = {"ct-1": source}
    digest = _expected_source_digest(
        ["ct-1"], rows, session_date="2026-08-18",
    )
    assert _validate_envelope(
        _valid_envelope(digest, source),
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-1"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) is None


@pytest.mark.parametrize(
    "critical_lane",
    [
        "I stopped tesamorelin yesterday.",
        "I never took tesamorelin.",
        "I was prescribed tesamorelin but never took it.",
        "I wasn’t taking tesamorelin.",
        "I was not taking tesamorelin.",
        "I haven't taken tesamorelin.",
        "I have not taken tesamorelin.",
        "I haven't used tesamorelin.",
        "I haven't been taking tesamorelin.",
        "I was prescribed tesamorelin but haven't taken it.",
        "I've not taken tesamorelin.",
        "I've not used tesamorelin.",
        "I've not been taking tesamorelin.",
        "I've never been on tesamorelin.",
        "I don't actually take tesamorelin.",
        "I was prescribed tesamorelin but I've not taken it.",
        "I'm definitely not taking tesamorelin.",
        "I am absolutely not taking tesamorelin.",
        "I definitely don't take tesamorelin.",
        "I've certainly not used tesamorelin.",
    ],
)
def test_envelope_requires_every_safety_critical_requester_lane(
    critical_lane: str,
) -> None:
    old = _source(
        canonical_turn_id="ct-old",
        user_content="I am currently taking tesamorelin.",
        session_date="2026-08-17",
    )
    new = _source(
        canonical_turn_id="ct-new",
        user_content=critical_lane,
        session_date="2026-08-18",
    )
    rows = {"ct-old": old, "ct-new": new}
    digest = _expected_source_digest(
        ["ct-old", "ct-new"], rows, session_date="2026-08-18",
    )
    active_only = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="test-model",
        claims=(SummaryClaim(
            text=old["user_content"],
            claim_type="conversation",
            temporal_status="",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id="ct-old",
                source_role="requester",
                speaker_label="BigTex",
                evidence_excerpt=old["user_content"],
                session_date=old["session_date"],
                source_provenance_digest=_provenance_digest(old),
            ),),
        ),),
    )

    assert _validate_envelope(
        active_only,
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-old", "ct-new"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "bounded_source_claim_missing"


def test_envelope_requires_every_bounded_ordinary_requester_lane() -> None:
    first = _source(
        canonical_turn_id="ct-first",
        user_content="I prefer concise technical explanations.",
    )
    omitted = _source(
        canonical_turn_id="ct-omitted",
        user_content="I keep project notes in Markdown.",
    )
    rows = {"ct-first": first, "ct-omitted": omitted}
    digest = _expected_source_digest(
        ["ct-first", "ct-omitted"], rows, session_date="2026-08-18",
    )
    first_only = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="test-model",
        claims=(SummaryClaim(
            text=first["user_content"],
            claim_type="conversation",
            temporal_status="",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id="ct-first",
                source_role="requester",
                speaker_label="BigTex",
                evidence_excerpt=first["user_content"],
                session_date=first["session_date"],
                source_provenance_digest=_provenance_digest(first),
            ),),
        ),),
    )

    assert _validate_envelope(
        first_only,
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-first", "ct-omitted"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "bounded_source_claim_missing"


def test_envelope_rejects_another_admitted_actor_id_in_exact_evidence() -> None:
    first = _source(
        canonical_turn_id="ct-a",
        user_content="I saw actor-b in the internal record.",
        sender="Alice",
        sender_actor_id="actor-a",
    )
    second = _source(
        canonical_turn_id="ct-b",
        user_content="I updated the project notes.",
        sender="BigTex",
        sender_actor_id="actor-b",
    )
    rows = {"ct-a": first, "ct-b": second}
    digest = _expected_source_digest(
        ["ct-a", "ct-b"], rows, session_date="2026-08-18",
    )

    def claim_for(source: dict) -> SummaryClaim:
        return SummaryClaim(
            text=source["user_content"],
            claim_type="conversation",
            temporal_status="",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id=source["canonical_turn_id"],
                source_role="requester",
                speaker_label=source["sender"],
                evidence_excerpt=source["user_content"],
                session_date=source["session_date"],
                source_provenance_digest=_provenance_digest(source),
            ),),
        )

    envelope = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="test-model",
        claims=(claim_for(first), claim_for(second)),
    )

    assert _validate_envelope(
        envelope,
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-a", "ct-b"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "internal_identity_in_claim"


def test_envelope_rejects_provider_order_for_complete_ordinary_claims() -> None:
    first = _source(
        canonical_turn_id="ct-first",
        user_content="I enjoy tea.",
    )
    second = _source(
        canonical_turn_id="ct-second",
        user_content="My cup is blue.",
    )
    rows = {"ct-first": first, "ct-second": second}
    digest = _expected_source_digest(
        ["ct-first", "ct-second"], rows, session_date="2026-08-18",
    )

    def claim_for(source: dict) -> SummaryClaim:
        return SummaryClaim(
            text=source["user_content"],
            claim_type="conversation",
            temporal_status="",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id=source["canonical_turn_id"],
                source_role="requester",
                speaker_label=source["sender"],
                evidence_excerpt=source["user_content"],
                session_date=source["session_date"],
                source_provenance_digest=_provenance_digest(source),
            ),),
        )

    envelope = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="old-provider-model",
        claims=(claim_for(second), claim_for(first)),
    )

    assert _validate_envelope(
        envelope,
        expected_digest=digest,
        expected_model="",
        canonical_ids=["ct-first", "ct-second"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "claim_order_mismatch"


def test_envelope_rejects_unevidenced_state_and_session_event_time():
    source = _source(assistant_content="")
    rows = {"ct-1": source}
    digest = _expected_source_digest(
        ["ct-1"], rows, session_date="2026-08-18",
    )
    valid = _valid_envelope(digest, source)
    forged_status = StructuredSummary(
        schema_version=valid.schema_version,
        claims=(replace(valid.claims[0], temporal_status="active"),),
        source_digest=valid.source_digest,
        generation_model=valid.generation_model,
    )
    assert _validate_envelope(
        forged_status,
        expected_digest=digest,
        expected_model="",
        canonical_ids=["ct-1"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "temporal_status_not_evidenced"

    forged_time = StructuredSummary(
        schema_version=valid.schema_version,
        claims=(replace(valid.claims[0], event_time="2026-08-18"),),
        source_digest=valid.source_digest,
        generation_model=valid.generation_model,
    )
    assert _validate_envelope(
        forged_time,
        expected_digest=digest,
        expected_model="",
        canonical_ids=["ct-1"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "event_time_not_evidenced"


def test_envelope_rejects_actor_mutation_even_when_text_and_label_match():
    source = _source(assistant_content="")
    digest = _expected_source_digest(
        ["ct-1"], {"ct-1": source}, session_date="2026-08-18",
    )
    envelope = _valid_envelope(digest, source)
    changed = _source(
        assistant_content="",
        sender_actor_id="actor:discord:different-person",
    )
    assert _validate_envelope(
        envelope,
        # Isolate the per-claim proof: the segment digest independently binds
        # this mutation at the segment currentness gate.
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-1"],
        rows_by_id={"ct-1": changed},
        session_date="2026-08-18",
    ) == "source_provenance_digest_mismatch"


def test_envelope_rejects_a_requester_lane_with_a_colliding_human_label():
    first = _source(assistant_content="")
    second = _source(
        canonical_turn_id="ct-2",
        sender_actor_id="actor:discord:other",
        assistant_content="",
    )
    envelope = _valid_envelope("a" * 64, first)
    assert _validate_envelope(
        envelope,
        expected_digest="a" * 64,
        expected_model="test-model",
        canonical_ids=["ct-1", "ct-2"],
        rows_by_id={"ct-1": first, "ct-2": second},
        session_date="2026-08-18",
    ) == "requester_source_not_admissible"


def test_envelope_rejects_substring_instead_of_complete_physical_lane():
    source = _source(
        user_content="I am no longer currently taking tesamorelin.",
        assistant_content="",
    )
    rows = {"ct-1": source}
    digest = _expected_source_digest(
        ["ct-1"], rows, session_date="2026-08-18",
    )
    valid = _valid_envelope(digest, source)
    bad = StructuredSummary(
        schema_version=valid.schema_version,
        source_digest=valid.source_digest,
        generation_model=valid.generation_model,
        claims=(SummaryClaim(
            text="currently taking tesamorelin",
            claim_type="conversation",
            temporal_status="active",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id="ct-1",
                source_role="requester",
                speaker_label="BigTex",
                evidence_excerpt="currently taking tesamorelin",
                session_date="2026-08-18",
                source_provenance_digest=_provenance_digest(
                    source, content="currently taking tesamorelin",
                ),
            ),),
        ),),
    )
    assert _validate_envelope(
        bad,
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-1"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "evidence_excerpt_mismatch"


def test_assistant_cannot_supply_personal_state():
    source = _source(user_content="")
    rows = {"ct-1": source}
    digest = _expected_source_digest(
        ["ct-1"], rows, session_date="2026-08-18",
    )
    envelope = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="test-model",
        claims=(SummaryClaim(
            text="The medication is no longer active.",
            claim_type="personal",
            temporal_status="",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id="ct-1",
                source_role="assistant",
                speaker_label="Assistant",
                evidence_excerpt="the medication is no longer active",
                session_date="2026-08-18",
                source_provenance_digest="f" * 64,
            ),),
        ),),
    )
    # The strict schema itself rejects this before the migration's redundant
    # role check can run.
    assert _validate_envelope(
        envelope,
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-1"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "schema_round_trip_failed"


def test_segment_envelope_rejects_multiple_sources_per_claim():
    source = _source(assistant_content="")
    rows = {"ct-1": source}
    digest = _expected_source_digest(
        ["ct-1"], rows, session_date="2026-08-18",
    )
    evidence = source["user_content"]
    envelope = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        source_digest=digest,
        generation_model="test-model",
        claims=(SummaryClaim(
            text=evidence,
            claim_type="personal",
            temporal_status="ceased",
            modality="asserted",
            sources=(
                SummarySource(
                    canonical_turn_id="ct-1",
                    source_role="requester",
                    speaker_label="BigTex",
                    evidence_excerpt=evidence,
                    session_date="2026-08-18",
                    source_provenance_digest=_provenance_digest(source),
                ),
                SummarySource(
                    canonical_turn_id="ct-1",
                    source_role="requester",
                    speaker_label="BigTex",
                    evidence_excerpt=evidence,
                    session_date="2026-08-18",
                    source_provenance_digest=_provenance_digest(source),
                ),
            ),
        ),),
    )
    assert _validate_envelope(
        envelope,
        expected_digest=digest,
        expected_model="test-model",
        canonical_ids=["ct-1"],
        rows_by_id=rows,
        session_date="2026-08-18",
    ) == "schema_round_trip_failed"


def _tag_claim(
    text: str,
    canonical_turn_id: str,
    *,
    temporal_status: str = "",
) -> SummaryClaim:
    physical = _source(
        canonical_turn_id=canonical_turn_id,
        user_content=text,
        assistant_content="",
    )
    return SummaryClaim(
        text=text,
        claim_type="personal",
        temporal_status=temporal_status,
        modality="asserted",
        sources=(SummarySource(
            canonical_turn_id=canonical_turn_id,
            source_role="requester",
            speaker_label="BigTex",
            evidence_excerpt=text,
            session_date="2026-08-18",
            source_provenance_digest=_provenance_digest(physical),
        ),),
    )


def _tag_source_row(ref: str, created_at: str, claim: SummaryClaim) -> dict:
    structured = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        claims=(claim,),
        source_digest=(ref[0] * 64),
        generation_model="segment-model",
    )
    return {
        "ref": ref,
        "primary_tag": "health",
        "tags": ["health"],
        "created_at": created_at,
        "start_timestamp": created_at,
        "metadata_json": {
            "canonical_turn_ids": [claim.sources[0].canonical_turn_id],
            "source_mapping_complete": True,
            "session_date": claim.sources[0].session_date,
            "structured_summary": structured_summary_to_dict(structured),
        },
        "summary": "Internal retrieval synopsis.",
        "summary_tokens": 4,
        "full_tokens": 20,
        "row_version": "1",
    }


def _current_segment_inventory(source_row: dict) -> _Inventory:
    return _Inventory(
        scanned=1,
        candidates=[],
        current_rows=[source_row],
        skipped_rows=[],
        current=1,
        skipped_reason_counts=Counter(),
        candidate_reason_counts=Counter(),
        affected_tags={"health"},
    )


def test_tag_inventory_blocks_a_missing_tag_summary_instead_of_inserting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _tag_claim("BigTex stopped tesamorelin.", "ct-1")
    source_row = _tag_source_row(
        "a", "2026-08-18T12:00:00+00:00", claim,
    )
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd."
        "_load_existing_tag_rows",
        lambda _conn, _conversation_id, _tags: {},
    )
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd._load_source_rows",
        lambda _conn, _conversation_id, _ids, lock: [_source(
            canonical_turn_id="ct-1",
            user_content=claim.text,
            assistant_content="",
            turn_group_number=7,
        )],
    )

    inventory = _tag_inventory(
        object(),
        SimpleNamespace(conversation_id="owner-1", after_tag=None),
        _current_segment_inventory(source_row),
    )

    assert inventory.candidates == []
    assert inventory.current == 0
    assert inventory.blocked == [{
        "tag": "health",
        "reason": "tag_summary_missing",
    }]
    assert inventory.reason_counts == {"tag_summary_missing": 1}


@pytest.mark.parametrize("embedding_json", [None, "not-json"])
def test_tag_inventory_treats_embedding_health_as_nonstructural(
    monkeypatch: pytest.MonkeyPatch,
    embedding_json: object,
) -> None:
    claim = _tag_claim("BigTex stopped tesamorelin.", "ct-1")
    source_row = _tag_source_row(
        "a", "2026-08-18T12:00:00+00:00", claim,
    )
    expected = _deterministic_tag_envelope(
        [source_row], generation_model="inventory",
    )
    structured = _deterministic_tag_selection(expected, ["ct-1"])
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd."
        "_load_existing_tag_rows",
        lambda _conn, _conversation_id, _tags: {
            "health": {
                "row_version": "11",
                "source_segment_refs": '["a"]',
                "source_turn_numbers": "[7]",
                "source_canonical_turn_ids": '["ct-1"]',
                "structured_summary_json": structured_summary_to_dict(structured),
                "covers_through_turn": 7,
                "covers_through_canonical_turn_id": "ct-1",
                "embedding_json": embedding_json,
            },
        },
    )
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd._load_source_rows",
        lambda _conn, _conversation_id, _ids, lock: [_source(
            canonical_turn_id="ct-1",
            user_content=claim.text,
            assistant_content="",
            turn_group_number=7,
        )],
    )

    inventory = _tag_inventory(
        object(),
        SimpleNamespace(conversation_id="owner-1", after_tag=None),
        _current_segment_inventory(source_row),
    )

    assert inventory.current == 1
    assert inventory.candidates == []
    assert inventory.blocked == []


def test_migration_rejects_duplicate_or_normalized_segment_source_ids():
    assert _canonical_ids({"canonical_turn_ids": ["ct-1", "ct-2"]}) == [
        "ct-1", "ct-2",
    ]
    assert _canonical_ids({"canonical_turn_ids": ["ct-1", "ct-1"]}) is None
    assert _canonical_ids({"canonical_turn_ids": [" ct-1"]}) is None


def test_tag_rollup_copies_newest_claims_first_and_dedupes_exactly():
    old_claim = _tag_claim(
        "BigTex was taking tesamorelin.", "ct-old", temporal_status="active",
    )
    new_claim = _tag_claim(
        "BigTex stopped tesamorelin.", "ct-new", temporal_status="ceased",
    )
    older = _tag_source_row(
        "a-old", "2026-08-17T12:00:00+00:00", old_claim,
    )
    newer = _tag_source_row(
        "b-new", "2026-08-18T12:00:00+00:00", new_claim,
    )
    duplicate = _tag_source_row(
        "c-duplicate", "2026-08-16T12:00:00+00:00", new_claim,
    )

    envelope = _deterministic_tag_envelope(
        [older, duplicate, newer], generation_model="tag-model",
    )

    assert envelope.claims == (new_claim, old_claim)
    assert (
        envelope.claims[0].sources[0].source_provenance_digest
        == new_claim.sources[0].source_provenance_digest
    )
    assert envelope.generation_model == "tag-model"
    assert len(envelope.source_digest) == 64


def test_tag_pool_preserves_authenticated_segment_claim_order():
    active = _tag_claim(
        "BigTex is currently taking tesamorelin.",
        "ct-old",
        temporal_status="",
    )
    ceased = _tag_claim(
        "BigTex stopped taking tesamorelin.",
        "ct-new",
        temporal_status="",
    )
    row = _tag_source_row(
        "a", "2026-08-18T12:00:00+00:00", active,
    )
    row["metadata_json"]["canonical_turn_ids"] = ["ct-old", "ct-new"]
    row["metadata_json"]["structured_summary"] = structured_summary_to_dict(
        StructuredSummary(
            schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
            claims=(ceased, active),
            source_digest="a" * 64,
            generation_model="segment-model",
        ),
    )

    envelope = _deterministic_tag_envelope(
        [row], generation_model="tag-model",
    )

    assert envelope.claims == (ceased, active)


def test_tag_rollup_refuses_a_multi_source_segment_claim():
    evidence = "BigTex stopped tesamorelin."
    source = SummarySource(
        canonical_turn_id="ct-1",
        source_role="requester",
        speaker_label="BigTex",
        evidence_excerpt=evidence,
        session_date="2026-08-18",
        source_provenance_digest="e" * 64,
    )
    multi = SummaryClaim(
        text=evidence,
        claim_type="personal",
        temporal_status="ceased",
        modality="asserted",
        sources=(source, source),
    )
    # The strict segment codec fails the malformed source envelope empty, so
    # the tag layer cannot copy either source into a trusted rollup.
    envelope = _deterministic_tag_envelope(
        [_tag_source_row("a", "2026-08-18T12:00:00+00:00", multi)],
        generation_model="tag-model",
    )
    assert envelope.claims == ()


def test_general_primary_tag_is_never_dropped():
    assert _row_tags({"primary_tag": "_general", "tags": []}) == {"_general"}
    assert _row_tags({
        "primary_tag": "health", "tags": ["_general", "health"],
    }) == {"_general", "health"}


@pytest.mark.parametrize(
    "raw",
    [
        '["ct-1",""]',
        '["ct-1","ct-1"]',
        '["ct-1"," ct-2"]',
        ["ct-1", "ct-2 "],
    ],
)
def test_tag_coordinate_lists_reject_empty_duplicate_or_normalized_ids(raw):
    assert _json_string_list(raw) is None


def test_tag_source_coordinates_use_physical_turn_group_number():
    summary = StoredSummary(
        ref="seg-1",
        metadata=SegmentMetadata(canonical_turn_ids=["ct-1", "ct-2"]),
    )
    turns, canonical_ids, max_turn = _tag_source_coordinates(
        [summary],
        {
            "ct-1": {"turn_group_number": 0},
            "ct-2": {"turn_group_number": 7, "turn_number": 999},
        },
    )
    assert turns == [0, 7]
    assert canonical_ids == ["ct-1", "ct-2"]
    assert max_turn == 7


def test_tag_result_must_preserve_deterministic_claims_and_source_order():
    claim = _tag_claim(
        "BigTex stopped tesamorelin.", "ct-1", temporal_status="ceased",
    )
    expected = _deterministic_tag_envelope(
        [_tag_source_row("a", "2026-08-18T12:00:00+00:00", claim)],
        generation_model="tag-model",
    )
    result = TagSummary(
        tag="health",
        summary="Retrieval synopsis.",
        source_segment_refs=["a"],
        source_turn_numbers=[7],
        source_canonical_turn_ids=["ct-1"],
        structured_summary=expected,
    )
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["a"],
        turn_numbers=[7],
        canonical_ids=["ct-1"],
    ) is None
    result.source_canonical_turn_ids = ["ct-extra", "ct-1"]
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["a"],
        turn_numbers=[7],
        canonical_ids=["ct-1"],
    ) == "tag_source_digest_mismatch"
    result.source_canonical_turn_ids = ["ct-1"]
    result.source_segment_refs = ["different"]
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["a"],
        turn_numbers=[7],
        canonical_ids=["ct-1"],
    ) == "tag_source_refs_mismatch"


def test_tag_result_accepts_only_exact_ordered_subset_with_selected_digest():
    ceased = _tag_claim(
        "I stopped taking tesamorelin.", "ct-new", temporal_status="",
    )
    active = _tag_claim(
        "I was taking tesamorelin.", "ct-old", temporal_status="",
    )
    expected = _deterministic_tag_envelope(
        [
            _tag_source_row("a-old", "2026-08-17T12:00:00+00:00", active),
            _tag_source_row("b-new", "2026-08-18T12:00:00+00:00", ceased),
        ],
        generation_model="tag-model",
    )
    selected = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        claims=(ceased,),
        source_digest=_tag_claim_selection_digest(
            (ceased,), ("ct-new", "ct-old"),
        ),
        generation_model="tag-model",
    )
    result = TagSummary(
        tag="health",
        summary="Retrieval synopsis.",
        source_segment_refs=["b-new", "a-old"],
        source_turn_numbers=[7],
        source_canonical_turn_ids=["ct-new", "ct-old"],
        structured_summary=selected,
    )

    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["b-new", "a-old"],
        turn_numbers=[7],
        canonical_ids=["ct-new", "ct-old"],
    ) is None

    result.source_canonical_turn_ids = ["ct-old", "ct-new"]
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["b-new", "a-old"],
        turn_numbers=[7],
        canonical_ids=["ct-new", "ct-old"],
    ) == "tag_source_digest_mismatch"
    result.source_canonical_turn_ids = ["ct-new"]
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["b-new", "a-old"],
        turn_numbers=[7],
        canonical_ids=["ct-new", "ct-old"],
    ) == "tag_source_digest_mismatch"
    result.source_canonical_turn_ids = ["ct-new", "ct-old"]

    result.structured_summary = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        claims=(active,),
        source_digest=_tag_claim_selection_digest(
            (active,), ("ct-new", "ct-old"),
        ),
        generation_model="tag-model",
    )
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["b-new", "a-old"],
        turn_numbers=[7],
        canonical_ids=["ct-new", "ct-old"],
    ) == "tag_claim_safety_floor_mismatch"

    rewritten = SummaryClaim(
        text=ceased.text,
        claim_type=ceased.claim_type,
        temporal_status="active",
        modality=ceased.modality,
        event_time=ceased.event_time,
        sources=ceased.sources,
    )
    result.structured_summary = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        claims=(rewritten,),
        source_digest=_tag_claim_selection_digest(
            (rewritten,), ("ct-new", "ct-old"),
        ),
        generation_model="tag-model",
    )
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["b-new", "a-old"],
        turn_numbers=[7],
        canonical_ids=["ct-new", "ct-old"],
    ) == "tag_claim_copy_mismatch"

    result.structured_summary = StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        claims=(ceased, ceased),
        source_digest=_tag_claim_selection_digest(
            (ceased, ceased), ("ct-new", "ct-old"),
        ),
        generation_model="tag-model",
    )
    assert _validate_tag_result(
        result,
        tag="health",
        expected=expected,
        expected_model="tag-model",
        source_refs=["b-new", "a-old"],
        turn_numbers=[7],
        canonical_ids=["ct-new", "ct-old"],
    ) == "tag_claim_duplicate"


@pytest.mark.parametrize(
    ("embedding_row", "embedding_version", "embedding_md5"),
    [
        (None, None, ""),
        ({"row_version": "12", "embedding_md5": "opaque"}, "12", "opaque"),
    ],
)
def test_tag_metadata_cas_preserves_every_retrieval_artifact(
    monkeypatch,
    embedding_row,
    embedding_version,
    embedding_md5,
):
    claim = _tag_claim(
        "BigTex stopped tesamorelin.", "ct-1", temporal_status="ceased",
    )
    source_row = _tag_source_row(
        "a", "2026-08-18T12:00:00+00:00", claim,
    )
    pool = _deterministic_tag_envelope(
        [source_row], generation_model="inventory",
    )
    expected = _deterministic_tag_selection(pool, ["ct-1"])
    result = TagSummary(
        tag="health",
        # These deliberately differ from storage. Metadata-only persistence
        # must never place them in SQL parameters.
        summary="MUST NOT BE WRITTEN",
        description="MUST NOT BE WRITTEN",
        summary_tokens=999,
        code_refs=[{"path": "MUST NOT BE WRITTEN"}],
        generated_by_turn_id="MUST NOT BE WRITTEN",
        source_segment_refs=["a"],
        source_turn_numbers=[7],
        source_canonical_turn_ids=["ct-1"],
        covers_through_turn=7,
        covers_through_canonical_turn_id="ct-1",
        structured_summary=expected,
    )
    scope = _Scope("tenant-1", "owner-1", 3, 4, "active", 0)
    physical = _source(
        canonical_turn_id="ct-1",
        user_content=claim.text,
        assistant_content="",
        turn_group_number=7,
    )

    class Result:
        def __init__(self, row=None, rowcount=0):
            self._row = row
            self.rowcount = rowcount

        def fetchone(self):
            return self._row

    class Transaction:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            assert self.conn.active is False
            self.conn.active = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self.conn.active = False
            self.conn.rolled_back = exc_type is not None
            return False

    class Connection:
        def __init__(self):
            self.active = False
            self.rolled_back = False
            self.writes = []

        def transaction(self):
            return Transaction(self)

        def execute(self, sql, params=()):
            if "FROM tag_summaries" in sql and "FOR UPDATE" in sql:
                return Result({
                    "row_version": "11",
                    "summary_md5": "summary",
                    "description_md5": "description",
                    "retrieval_artifacts_md5": "retrieval",
                })
            if "FROM tag_summary_embeddings" in sql and "FOR UPDATE" in sql:
                return Result(embedding_row)
            if sql.startswith("UPDATE tag_summaries"):
                assert self.active
                set_clause = sql.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
                assert "summary =" not in set_clause
                assert "description =" not in set_clause
                assert "summary_tokens =" not in set_clause
                assert "code_refs =" not in set_clause
                assert "generated_by_turn_id =" not in set_clause
                assert "created_at =" not in set_clause
                assert "updated_at =" not in set_clause
                self.writes.append("tag")
                return Result({
                    "summary_md5": "summary",
                    "description_md5": "description",
                    "retrieval_artifacts_md5": "retrieval",
                }, rowcount=1)
            raise AssertionError(sql)

    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd._preflight",
        lambda conn, args, lock: scope,
    )
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd."
        "_locked_current_tag_sources",
        lambda conn, args, tag, expected_rows: (None, [source_row]),
    )
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd._load_source_rows",
        lambda conn, conversation_id, canonical_ids, lock: [physical],
    )
    journal_events = []
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd._journal_append",
        lambda handle, payload: journal_events.append(payload),
    )
    conn = Connection()
    args = argparse.Namespace(
        tenant_id="tenant-1", conversation_id="owner-1",
    )
    outcome = _cas_tag_persist(
        conn,
        args,
        initial_scope=scope,
        candidate={
            "tag": "health",
            "_source_rows": [source_row],
            "_existing": {
                "row_version": "11",
                "embedding_row_version": embedding_version,
                "old_summary_md5": "summary",
                "old_description_md5": "description",
                "old_retrieval_artifacts_md5": "retrieval",
                "old_embedding_md5": embedding_md5,
            },
        },
        result=result,
        journal=object(),
        run_id="run-1",
    )

    assert outcome == "accepted"
    assert conn.writes == ["tag"]
    assert conn.rolled_back is False
    assert journal_events[0]["event"] == "tag_prepared"
    assert journal_events[0]["migration_mode"] == "deterministic_metadata_only_v1"
    assert journal_events[0]["provider_calls"] == 0
    assert journal_events[0]["old_retrieval_artifacts_md5"] == "retrieval"
    assert journal_events[0]["new_retrieval_artifacts_md5"] == "retrieval"
    assert journal_events[0]["old_embedding_md5"] == embedding_md5
    assert journal_events[0]["new_embedding_md5"] == embedding_md5


def test_tag_metadata_cas_rejects_embedding_checksum_race(monkeypatch):
    claim = _tag_claim("BigTex stopped tesamorelin.", "ct-1")
    source_row = _tag_source_row(
        "a", "2026-08-18T12:00:00+00:00", claim,
    )
    expected = _deterministic_tag_selection(
        _deterministic_tag_envelope([source_row], generation_model="inventory"),
        ["ct-1"],
    )
    result = TagSummary(
        tag="health",
        source_segment_refs=["a"],
        source_turn_numbers=[7],
        source_canonical_turn_ids=["ct-1"],
        covers_through_turn=7,
        covers_through_canonical_turn_id="ct-1",
        structured_summary=expected,
    )
    scope = _Scope("tenant-1", "owner-1", 3, 4, "active", 0)

    class Result:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class Connection:
        def __init__(self):
            self.writes = []

        def transaction(self):
            return Transaction()

        def execute(self, sql, params=()):
            if "FROM tag_summaries" in sql and "FOR UPDATE" in sql:
                return Result({
                    "row_version": "11",
                    "summary_md5": "summary",
                    "description_md5": "description",
                    "retrieval_artifacts_md5": "retrieval",
                })
            if "FROM tag_summary_embeddings" in sql and "FOR UPDATE" in sql:
                return Result({
                    "row_version": "12",
                    "embedding_md5": "changed",
                })
            if sql.startswith("UPDATE"):
                self.writes.append(sql)
            raise AssertionError(sql)

    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd._preflight",
        lambda conn, args, lock: scope,
    )
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd."
        "_locked_current_tag_sources",
        lambda conn, args, tag, expected_rows: (None, [source_row]),
    )
    monkeypatch.setattr(
        "virtual_context.cli.structured_summary_migration_cmd._load_source_rows",
        lambda conn, conversation_id, canonical_ids, lock: [_source(
            canonical_turn_id="ct-1",
            user_content=claim.text,
            assistant_content="",
            turn_group_number=7,
        )],
    )
    conn = Connection()
    outcome = _cas_tag_persist(
        conn,
        argparse.Namespace(tenant_id="tenant-1", conversation_id="owner-1"),
        initial_scope=scope,
        candidate={
            "tag": "health",
            "_source_rows": [source_row],
            "_existing": {
                "row_version": "11",
                "embedding_row_version": "12",
                "old_summary_md5": "summary",
                "old_description_md5": "description",
                "old_retrieval_artifacts_md5": "retrieval",
                "old_embedding_md5": "original",
            },
        },
        result=result,
        journal=object(),
        run_id="run-1",
    )

    assert outcome == "tag_or_embedding_changed"
    assert conn.writes == []


def test_cache_action_escapes_conversation_id_in_redis_glob():
    action = _cache_action("owner[*]?")
    assert action["redis_delete_glob"] == r"vc:context_hint:owner\[\*\]\?:*"


def test_resume_cursor_freezes_before_undecided_row():
    cursor = _ResumeCursor("seg-0")
    cursor.on_decided("seg-1")
    cursor.freeze()
    cursor.on_decided("seg-2")
    assert cursor.ref == "seg-1"
    assert cursor.frozen is True


def test_parser_defaults_to_dry_run_and_positive_limits():
    parser = argparse.ArgumentParser()
    admin = parser.add_subparsers(dest="admin_command")

    def positive(value):
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("positive")
        return parsed

    configure_parser(admin, positive)
    args = parser.parse_args([
        "migrate-structured-summaries", "owner-1", "--tenant-id", "tenant-1",
    ])
    assert args.apply is False
    assert args.phase == "all"
    with pytest.raises(SystemExit):
        parser.parse_args([
            "migrate-structured-summaries", "owner-1",
            "--tenant-id", "tenant-1", "--limit", "0",
        ])
