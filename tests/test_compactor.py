"""Tests for DomainCompactor (tag-based)."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import MockLLMProvider
from virtual_context.core.compactor import (
    DomainCompactor,
    SegmentSummaryGenerationError,
    TagSummaryGenerationError,
    TAG_SUMMARY_ROLLUP_PROMPT,
    _format_tag_rollup_source,
)
from virtual_context.core.structured_summary import (
    infer_modality,
    is_safety_critical_personal_evidence,
    structured_source_digest,
    structured_source_provenance_digest,
    structured_tag_claim_digest,
    validate_tag_rollup_inputs,
)
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    AUTHOR_ROLE_ASSISTANT,
    AUTHOR_ROLE_REQUESTER,
    AUTHOR_ROLE_SUBJECT,
    SOURCE_CANONICAL_TURN_IDS_KEY,
    ActorRoster,
    CanonicalTurnRow,
    CompactorConfig,
    FactLane,
    Message,
    SegmentMetadata,
    StoredSummary,
    StructuredSummary,
    SummaryClaim,
    SummarySource,
    TaggedSegment,
    TagPromptRule,
    TagSummary,
)


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def compactor(mock_llm):
    return DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )


def _named_roster(*labels: str, complete: bool = True) -> ActorRoster:
    actors = {f"actor:discord:{index}" for index, _ in enumerate(labels, 1)}
    lanes = [
        FactLane(
            role=AUTHOR_ROLE_REQUESTER,
            text=f"source words from {label}",
            actor_id=f"actor:discord:{index}",
            speaker_label=label,
        )
        for index, label in enumerate(labels, 1)
    ]
    return ActorRoster(
        actor_ids=actors,
        labels={
            label.casefold(): {f"actor:discord:{index}"}
            for index, label in enumerate(labels, 1)
        },
        complete=complete,
        lanes=lanes,
    )


class _SequenceProvider:
    def __init__(self, *responses: object):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, max_tokens: int):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        response = self.responses[index]
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, str)
        return response, {}


def _stored_summary(
    text: str,
    *,
    ref: str = "seg-1",
    label: str = "",
    identity: str = "",
    scope: str = "scope-proof",
) -> StoredSummary:
    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    metadata = SegmentMetadata(
        source_speaker_labels=[label] if label else [],
        source_speaker_identity_count=1 if label else 0,
        source_speaker_identity_fingerprint=(identity or label) if label else "",
        source_audience_fingerprint=scope if label else "",
    )
    return StoredSummary(
        ref=ref,
        primary_tag="medical",
        tags=["medical"],
        summary=text,
        summary_tokens=max(1, len(text) // 4),
        metadata=metadata,
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    )


def _structured_stored_summary() -> StoredSummary:
    source, _claim = _stored_claim(
        "I stopped tesamorelin after edema.",
        canonical_id="ct-1",
        label="BigTex",
        status="",
        ref="seg-1",
    )
    source.summary = "Internal synopsis: BigTex stopped tesamorelin."
    return source


def _stored_claim(
    evidence: str,
    *,
    canonical_id: str,
    label: str,
    status: str,
    ref: str,
) -> tuple[StoredSummary, SummaryClaim]:
    source = _stored_summary(
        f"Internal retrieval synopsis for {label}.", ref=ref, label=label,
    )
    actor = f"actor:test:{label.casefold()}"
    record = {
        "canonical_turn_id": canonical_id,
        "source_role": "requester",
        "actor_id": actor,
        "speaker_label": label,
        "content": evidence,
        "session_date": "2026-08-18",
        "audience_conversation_id": "conv-test",
        "origin_channel_id": "channel:test",
        "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
    }
    claim = SummaryClaim(
        text=evidence,
        claim_type="conversation",
        temporal_status="",
        modality=infer_modality(evidence),
        sources=(SummarySource(
            canonical_turn_id=canonical_id,
            source_role="requester",
            speaker_label=label,
            evidence_excerpt=evidence,
            session_date="2026-08-18",
            source_provenance_digest=structured_source_provenance_digest(record),
        ),),
    )
    source.metadata.structured_summary = StructuredSummary(
        schema_version=1,
        claims=(claim,),
        source_digest=structured_source_digest((record,), namespace="segment"),
        generation_model="segment-model",
    )
    source.metadata.canonical_turn_ids = [canonical_id]
    source.metadata.source_mapping_complete = True
    source.metadata.session_date = "2026-08-18"
    assert status == ""
    return source, claim


def _tag_rollup_proof(*summaries: StoredSummary):
    rows_by_id: dict[str, CanonicalTurnRow] = {}
    for summary in summaries:
        claims = summary.metadata.structured_summary.claims
        sources_by_id = {
            claim.sources[0].canonical_turn_id: claim.sources[0]
            for claim in claims
        }
        records = []
        for canonical_id in summary.metadata.canonical_turn_ids:
            source = sources_by_id[canonical_id]
            actor = f"actor:test:{source.speaker_label.casefold()}"
            row = CanonicalTurnRow(
                conversation_id="conv-test",
                canonical_turn_id=canonical_id,
                user_content=source.evidence_excerpt,
                sender=source.speaker_label,
                sender_actor_id=actor,
                session_date=source.session_date,
                audience_conversation_id="conv-test",
                audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
                origin_channel_id="channel:test",
            )
            rows_by_id[canonical_id] = row
            records.append({
                "canonical_turn_id": canonical_id,
                "source_role": "requester",
                "actor_id": actor,
                "speaker_label": source.speaker_label,
                "content": source.evidence_excerpt,
                "session_date": source.session_date,
                "audience_conversation_id": "conv-test",
                "origin_channel_id": "channel:test",
                "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
            })
        current = summary.metadata.structured_summary
        summary.metadata.structured_summary = StructuredSummary(
            schema_version=current.schema_version,
            claims=current.claims,
            source_digest=structured_source_digest(
                records, namespace="segment",
            ),
            generation_model=current.generation_model,
        )
        summary.metadata.source_mapping_complete = True
    return validate_tag_rollup_inputs(
        summaries, rows_by_id, conversation_id="conv-test",
    )


def _physical_row_for_claim(claim: SummaryClaim) -> CanonicalTurnRow:
    source = claim.sources[0]
    return CanonicalTurnRow(
        conversation_id="conv-test",
        canonical_turn_id=source.canonical_turn_id,
        user_content=source.evidence_excerpt,
        sender=source.speaker_label,
        sender_actor_id=f"actor:test:{source.speaker_label.casefold()}",
        session_date=source.session_date,
        audience_conversation_id="conv-test",
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        origin_channel_id="channel:test",
    )


def _structured_record_for_row(row: CanonicalTurnRow) -> dict[str, object]:
    return {
        "canonical_turn_id": row.canonical_turn_id,
        "source_role": "requester",
        "actor_id": row.sender_actor_id,
        "speaker_label": row.sender,
        "content": row.user_content,
        "session_date": row.session_date,
        "audience_conversation_id": row.audience_conversation_id,
        "origin_channel_id": row.origin_channel_id,
        "audience_attribution_version": row.audience_attribution_version,
    }


def _scoped_user_message(
    content: str,
    *,
    canonical_id: str,
    actor: str,
    label: str,
    audience: str = "audience:guild:1",
    channel: str = "channel:medical",
) -> Message:
    return Message(
        role="user",
        content=content,
        metadata={
            "sender": {"name": label},
            SOURCE_CANONICAL_TURN_IDS_KEY: [canonical_id],
        },
        source_actor_id=actor,
        source_audience_conversation_id=audience,
        source_origin_channel_id=channel,
        source_audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
    )


def _structured_segment_and_roster() -> tuple[TaggedSegment, ActorRoster]:
    message = _scoped_user_message(
        "I stopped tesamorelin after edema and have not restarted it.",
        canonical_id="ct-bigtex-1",
        actor="actor:discord:bigtex",
        label="BigTex",
    )
    segment = TaggedSegment(
        id="seg-bigtex",
        primary_tag="tesamorelin",
        tags=["tesamorelin"],
        messages=[message],
        token_count=32,
        turn_count=1,
        session_date="2026-08-18",
    )
    roster = ActorRoster(
        actor_ids={"actor:discord:bigtex"},
        labels={"bigtex": {"actor:discord:bigtex"}},
        complete=True,
        lanes=[FactLane(
            role=AUTHOR_ROLE_REQUESTER,
            text=message.content,
            actor_id="actor:discord:bigtex",
            canonical_turn_id="ct-bigtex-1",
            speaker_label="BigTex",
        )],
    )
    return segment, roster


@pytest.fixture
def legal_segment(ts):
    return TaggedSegment(
        primary_tag="legal",
        tags=["legal", "court"],
        messages=[
            Message(
                role="user",
                content="What's the court filing deadline?",
                timestamp=ts,
                metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"]},
                source_actor_id="actor:discord:1",
                source_audience_conversation_id="audience:guild:1",
                source_origin_channel_id="channel:legal",
                source_audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
            ),
            Message(role="assistant", content="The filing is due January 30.", timestamp=ts + timedelta(seconds=30)),
        ],
        token_count=50,
        start_timestamp=ts,
        end_timestamp=ts + timedelta(seconds=30),
        turn_count=1,
    )


def test_compact_single(compactor, legal_segment, mock_llm):
    results = compactor.compact([legal_segment])
    assert len(results) == 1
    assert results[0].primary_tag == "legal"
    assert results[0].summary == "Test summary"
    assert len(mock_llm.calls) == 1


def test_compact_preserves_metadata(compactor, legal_segment):
    results = compactor.compact([legal_segment])
    assert results[0].metadata.entities == ["entity1"]
    assert results[0].metadata.key_decisions == ["decision1"]


def test_compact_preserves_message_provenance_metadata(compactor, legal_segment):
    legal_segment.messages[0].metadata = {
        "sender": {"name": "BigTex"},
        "_vc_source_canonical_turn_ids": ["ct-user"],
    }

    result = compactor.compact([legal_segment])[0]

    assert result.messages[0]["metadata"] == legal_segment.messages[0].metadata


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
def test_named_roster_identity_contract_is_post_template_and_never_exposes_actor_ids(
    legal_segment,
    code_mode,
    custom_prompt,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster("BigTex", "Optics"),
    )

    assert "SUMMARY SPEAKER IDENTITY CONTRACT" in request.prompt
    assert '["BigTex", "Optics"]' in request.prompt
    assert "actor:discord:" not in request.prompt
    if custom_prompt:
        assert request.prompt.index("Custom legal summary.") \
            < request.prompt.index("SUMMARY SPEAKER IDENTITY CONTRACT")


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
@pytest.mark.parametrize(
    "unsafe_label",
    [
        "actor:discord:1",
        "User.",
        "(You)",
        "U\u200bser",
        "actor\u2060:discord:1",
    ],
    ids=[
        "actor-id",
        "decorated-generic",
        "decorated-pronoun",
        "zw-generic",
        "zw-actor-id",
    ],
)
def test_unsafe_roster_label_never_enters_any_segment_prompt(
    legal_segment,
    code_mode,
    custom_prompt,
    unsafe_label,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )
    legal_segment.messages[0].metadata = {
        "sender": {"name": unsafe_label},
        SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"],
    }

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster(unsafe_label),
    )

    assert "SUMMARY SPEAKER IDENTITY CONTRACT" in request.prompt
    assert "are: []." in request.prompt
    assert unsafe_label not in request.prompt
    assert "Source (" in request.prompt
    assert "What's the court filing deadline?" in request.prompt


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
def test_compatibility_equivalent_distinct_actor_labels_collide_in_every_segment_prompt(
    legal_segment,
    code_mode,
    custom_prompt,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )
    legal_segment.messages = [
        Message(
            role="user",
            content="first source text",
            metadata={"sender": {"name": "Alex"}},
            source_actor_id="actor:discord:alex-one",
        ),
        Message(
            role="user",
            content="second source text",
            metadata={"sender": {"name": "Ａｌｅｘ"}},
            source_actor_id="actor:discord:alex-two",
        ),
    ]

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster("Alex", "Ａｌｅｘ"),
    )

    contract = request.prompt.split("SUMMARY SPEAKER IDENTITY CONTRACT", 1)[1]
    assert "[]" in contract
    assert "Alex" not in contract
    assert "Ａｌｅｘ" not in contract
    transcript = request.prompt.split("SUMMARY SPEAKER IDENTITY CONTRACT", 1)[0]
    assert "Alex:" not in transcript
    assert "Ａｌｅｘ:" not in transcript
    assert "Source: first source text" in transcript
    assert "Source: second source text" in transcript


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
def test_valid_unicode_roster_labels_enter_every_segment_prompt(
    legal_segment,
    code_mode,
    custom_prompt,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster("BigTex", "Renée", "李雷"),
    )

    assert '["BigTex", "Renée", "李雷"]' in request.prompt


def test_structured_claim_prompt_uses_ephemeral_refs_and_no_durable_ids():
    segment, roster = _structured_segment_and_roster()
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=False),
    )

    request = compactor.build_segment_summary_request(segment, roster=roster)

    assert "SOURCE-BACKED SUMMARY CLAIMS CONTRACT" in request.prompt
    assert '"source_ref":"src_1"' in request.prompt
    assert '"speaker":"BigTex"' in request.prompt
    assert "I stopped tesamorelin" in request.prompt
    assert "ct-bigtex-1" not in request.prompt
    assert "actor:discord:bigtex" not in request.prompt


def test_structured_claim_rejects_active_status_against_stopped_evidence():
    segment, roster = _structured_segment_and_roster()
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": "BigTex still uses tesamorelin.",
            "claim_type": "personal",
            "temporal_status": "active",
            "modality": "asserted",
            "event_time": "",
            "sources": [{
                "source_ref": "src_1",
                "evidence_excerpt": "I stopped tesamorelin after edema",
            }],
        }],
    }, roster=roster, segment=segment)

    # The unsafe model classification is rejected, but the provider cannot
    # hide the exact correction: the deterministic safety floor retains it.
    assert len(structured.claims) == 1
    assert structured.claims[0].text == (
        "I stopped tesamorelin after edema and have not restarted it."
    )
    assert structured.claims[0].claim_type == "conversation"
    assert structured.claims[0].temporal_status == ""


def test_structured_claim_never_promotes_unstated_model_temporal_status():
    segment, roster = _structured_segment_and_roster()
    segment.messages[0].content = "I take tesamorelin."
    roster.lanes[0].text = "I take tesamorelin."
    compactor = DomainCompactor(MockLLMProvider(), CompactorConfig(code_mode=False))

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": "I take tesamorelin.",
            "claim_type": "personal",
            "temporal_status": "active",
            "modality": "asserted",
            "event_time": "",
            "sources": [{
                "source_ref": "src_1",
                "evidence_excerpt": "I take tesamorelin.",
            }],
        }],
    }, roster=roster, segment=segment)

    assert structured.claims[0].temporal_status == ""


def test_structured_claim_persists_exact_evidence_not_model_paraphrase():
    segment, roster = _structured_segment_and_roster()
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            # Even a semantically false gloss cannot become model evidence.
            "text": "BigTex still uses tesamorelin.",
            "claim_type": "personal",
            "temporal_status": "ceased",
            "modality": "asserted",
            "event_time": "",
            "sources": [{
                "source_ref": "src_1",
                "evidence_excerpt": "I stopped tesamorelin after edema",
            }],
        }],
    }, roster=roster, segment=segment)

    assert len(structured.claims) == 1
    claim = structured.claims[0]
    assert claim.text == (
        "I stopped tesamorelin after edema and have not restarted it."
    )
    assert claim.temporal_status == ""
    assert claim.modality == "asserted"
    assert claim.claim_type == "conversation"
    assert claim.sources[0].speaker_label == "BigTex"
    assert structured.source_digest
    assert structured.generation_model


def test_structured_claim_cannot_select_a_positive_substring_out_of_negation():
    segment, roster = _structured_segment_and_roster()
    source = "I am no longer currently taking tesamorelin."
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": "currently taking tesamorelin",
            "claim_type": "personal",
            "temporal_status": "active",
            "modality": "asserted",
            "event_time": "",
            "sources": [{
                "source_ref": "src_1",
                "evidence_excerpt": "currently taking tesamorelin",
            }],
        }],
    }, roster=roster, segment=segment)

    assert len(structured.claims) == 1
    assert structured.claims[0].text == source
    assert structured.claims[0].temporal_status == ""


@pytest.mark.parametrize(
    "source",
    [
        "I did not stop tesamorelin.",
        "I never discontinued tesamorelin.",
        "I haven't completed the tesamorelin course.",
        "I do not plan to start tesamorelin.",
    ],
)
def test_structured_claim_never_adds_positive_status_under_negation(source):
    segment, roster = _structured_segment_and_roster()
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": source,
            "claim_type": "personal",
            "temporal_status": "",
            "modality": "asserted",
            "event_time": "",
            "sources": [{
                "source_ref": "src_1",
                "evidence_excerpt": source,
            }],
        }],
    }, roster=roster, segment=segment)

    assert len(structured.claims) == 1
    assert structured.claims[0].temporal_status == ""


def test_structured_claim_leaves_status_blank_for_mixed_state_lane():
    segment, roster = _structured_segment_and_roster()
    source = "I stopped alcohol but still currently take tesamorelin."
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": source,
            "claim_type": "personal",
            "temporal_status": "",
            "modality": "asserted",
            "event_time": "",
            "sources": [{"source_ref": "src_1", "evidence_excerpt": source}],
        }],
    }, roster=roster, segment=segment)

    assert structured.claims[0].temporal_status == ""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("I'm off tesamorelin currently.", ""),
        ("I am currently off tesamorelin.", ""),
        ("I paused tesamorelin and currently take magnesium.", ""),
        ("I halted tesamorelin; currently recovering.", ""),
        ("Should I stop tesamorelin?", ""),
        ("I might discontinue tesamorelin.", ""),
    ],
)
def test_structured_claim_temporal_status_is_clause_conservative(source, expected):
    segment, roster = _structured_segment_and_roster()
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": source,
            "claim_type": "personal",
            "temporal_status": expected,
            "modality": "asserted",
            "event_time": "",
            "sources": [{"source_ref": "src_1", "evidence_excerpt": source}],
        }],
    }, roster=roster, segment=segment)

    # Modality is also application-derived; a question/hypothetical remains a
    # valid exact claim but can never be annotated as an executed state.
    assert structured.claims[0].temporal_status == expected


@pytest.mark.parametrize(
    "source",
    [
        "I stopped tesamorelin last month but restarted yesterday.",
        "I was off tesamorelin, then resumed it.",
        "I stopped tesamorelin temporarily and am back on it now.",
        "I almost stopped tesamorelin.",
        "I considered stopping tesamorelin.",
        "I was thinking about stopping tesamorelin.",
        "I wish I had stopped tesamorelin.",
        "I stopped by the clinic to ask about tesamorelin.",
        "I stopped the car while carrying tesamorelin.",
        "I remember stopping tesamorelin, but I may be mistaken.",
        "Vast falsely said I currently take tesamorelin.",
        "The assistant claimed I currently take tesamorelin, which is false.",
        "Vast said I stopped tesamorelin, but he was mistaken.",
        "I deny that I stopped tesamorelin.",
        "It is false that I stopped tesamorelin.",
    ],
)
def test_structured_claim_never_promotes_ambiguous_or_refuted_status(source):
    segment, roster = _structured_segment_and_roster()
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": source,
            "claim_type": "personal",
            "temporal_status": "ceased",
            "modality": "asserted",
            "event_time": "",
            "sources": [{"source_ref": "src_1", "evidence_excerpt": source}],
        }],
    }, roster=roster, segment=segment)

    assert structured.claims[0].text == source
    assert structured.claims[0].claim_type == "conversation"
    assert structured.claims[0].temporal_status == ""


def test_structured_claim_safety_floor_orders_newer_correction_before_old_state():
    segment, roster = _structured_segment_and_roster()
    old = "I am currently taking tesamorelin."
    new = "I stopped tesamorelin yesterday and have not restarted it."
    segment.messages[0].content = old
    roster.lanes[0].text = old
    roster.lanes[0].session_date = "2026-08-17"
    roster.lanes.append(FactLane(
        role=AUTHOR_ROLE_REQUESTER,
        text=new,
        canonical_turn_id="ct-bigtex-2",
        actor_id="actor:discord:bigtex",
        speaker_label="BigTex",
        session_date="2026-08-18",
        audience_conversation_id="guild-1",
        origin_channel_id="health",
        audience_attribution_version=1,
    ))
    segment.messages.append(Message(
        role="user",
        content=new,
        source_audience_conversation_id="guild-1",
        source_origin_channel_id="health",
        source_audience_attribution_version=1,
        metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-bigtex-2"]},
    ))
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": old,
            "claim_type": "personal",
            "temporal_status": "active",
            "modality": "asserted",
            "event_time": "",
            "sources": [{"source_ref": "src_1", "evidence_excerpt": old}],
        }],
    }, roster=roster, segment=segment)

    assert [claim.text for claim in structured.claims] == [new, old]
    assert [claim.temporal_status for claim in structured.claims] == ["", ""]


@pytest.mark.parametrize(
    "nonuse",
    [
        "I never took tesamorelin.",
        "I was prescribed tesamorelin but never took it.",
        "I wasn’t taking tesamorelin.",
        "I was not taking tesamorelin.",
        "I've never taken tesamorelin.",
        "I had never used tesamorelin.",
        "I was never on tesamorelin.",
        "I got a prescription for tesamorelin but never filled it.",
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
def test_segment_writer_prefixes_newer_exact_nonuse_when_provider_selects_old_state(
    nonuse: str,
) -> None:
    segment, roster = _structured_segment_and_roster()
    active = "I am currently taking tesamorelin."
    segment.messages[0].content = active
    roster.lanes[0].text = active
    roster.lanes[0].session_date = "2026-08-17"
    roster.lanes.append(FactLane(
        role=AUTHOR_ROLE_REQUESTER,
        text=nonuse,
        canonical_turn_id="ct-bigtex-2",
        actor_id="actor:discord:bigtex",
        speaker_label="BigTex",
        session_date="2026-08-18",
        audience_conversation_id="audience:guild:1",
        origin_channel_id="channel:medical",
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
    ))
    segment.messages.append(_scoped_user_message(
        nonuse,
        canonical_id="ct-bigtex-2",
        actor="actor:discord:bigtex",
        label="BigTex",
    ))
    segment.turn_count = 2
    provider = _SequenceProvider(json.dumps({
        "summary": "Internal retrieval synopsis.",
        # Simulate the original failure mode: the model preserves only the
        # older active lane and omits the newer direct non-use correction.
        "summary_claims": [{
            "text": active,
            "claim_type": "personal",
            "temporal_status": "active",
            "modality": "asserted",
            "event_time": "",
            "sources": [{
                "source_ref": "src_1",
                "evidence_excerpt": active,
            }],
        }],
    }))
    compactor = DomainCompactor(
        provider, CompactorConfig(code_mode=False), model_name="test-model",
    )

    result = compactor.summarize_segment(segment, roster=roster)

    assert len(provider.calls) == 1
    assert [
        claim.text for claim in result.metadata.structured_summary.claims
    ] == [nonuse, active]
    assert all(
        claim.claim_type == "conversation"
        and claim.temporal_status == ""
        for claim in result.metadata.structured_summary.claims
    )


@pytest.mark.parametrize(
    "incidental",
    [
        "I thought he never took tesamorelin.",
        "He was prescribed tesamorelin but never took it.",
        "I was prescribed tesamorelin, but my brother never took it.",
    ],
)
def test_nonuse_floor_does_not_promote_another_persons_nonuse(
    incidental: str,
) -> None:
    assert not is_safety_critical_personal_evidence(incidental)


def test_structured_claim_does_not_promote_session_date_to_event_time():
    segment, roster = _structured_segment_and_roster()
    source = "I stopped tesamorelin last month."
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": source,
            "claim_type": "personal",
            "temporal_status": "ceased",
            "modality": "asserted",
            "event_time": segment.session_date,
            "sources": [{"source_ref": "src_1", "evidence_excerpt": source}],
        }],
    }, roster=roster, segment=segment)

    assert structured.claims[0].event_time == ""


def test_structured_claim_does_not_bind_unrelated_literal_event_time():
    segment, roster = _structured_segment_and_roster()
    source = (
        "I stopped tesamorelin last month. "
        "My next appointment is August 22."
    )
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": source,
            "claim_type": "personal",
            "temporal_status": "ceased",
            "modality": "asserted",
            "event_time": "August 22",
            "sources": [{"source_ref": "src_1", "evidence_excerpt": source}],
        }],
    }, roster=roster, segment=segment)

    assert structured.claims[0].event_time == ""


def test_structured_claim_rejects_multiple_sources_even_for_one_speaker():
    segment, roster = _structured_segment_and_roster()
    first = "I enjoy tea."
    second = "I prefer coffee."
    segment.messages[0].content = first
    roster.lanes[0].text = first
    roster.lanes.append(FactLane(
        role=AUTHOR_ROLE_REQUESTER,
        text=second,
        canonical_turn_id="ct-bigtex-2",
        actor_id="actor:discord:bigtex",
        speaker_label="BigTex",
    ))
    segment.messages.append(Message(
        role="user",
        content=second,
        source_audience_conversation_id="guild-1",
        source_origin_channel_id="health",
        source_audience_attribution_version=1,
        metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-bigtex-2"]},
    ))
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [{
            "text": second,
            "claim_type": "personal",
            "temporal_status": "",
            "modality": "asserted",
            "event_time": "yesterday",
            "sources": [
                {
                    "source_ref": "src_1",
                    "evidence_excerpt": first,
                },
                {
                    "source_ref": "src_2",
                    "evidence_excerpt": second,
                },
            ],
        }],
    }, roster=roster, segment=segment)

    assert structured.claims == ()


def test_assistant_personal_state_is_dropped_without_losing_valid_human_claim():
    segment, roster = _structured_segment_and_roster()
    segment.messages.append(Message(
        role="assistant",
        content="You already run tesamorelin.",
    ))
    roster.lanes.append(FactLane(
        role=AUTHOR_ROLE_ASSISTANT,
        text="You already run tesamorelin.",
        canonical_turn_id="ct-bigtex-1",
        speaker_label="Assistant",
    ))
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary({
        "summary_claims": [
            {
                "text": "I stopped tesamorelin after edema",
                "claim_type": "personal",
                "temporal_status": "ceased",
                "modality": "asserted",
                "event_time": "",
                "sources": [{
                    "source_ref": "src_1",
                    "evidence_excerpt": "I stopped tesamorelin after edema",
                }],
            },
            {
                "text": "You already run tesamorelin.",
                "claim_type": "world",
                "temporal_status": "",
                "modality": "asserted",
                "event_time": "",
                "sources": [{
                    "source_ref": "src_2",
                    "evidence_excerpt": "You already run tesamorelin.",
                }],
            },
        ],
    }, roster=roster, segment=segment)

    assert [claim.text for claim in structured.claims] == [
        "I stopped tesamorelin after edema and have not restarted it.",
    ]


def test_summarize_segment_strict_retries_malformed_then_succeeds():
    segment, roster = _structured_segment_and_roster()
    provider = _SequenceProvider(
        "not json",
        json.dumps({
            "summary": "BigTex stopped tesamorelin after edema.",
            "summary_claims": [{
                "text": "I stopped tesamorelin after edema",
                "claim_type": "personal",
                "temporal_status": "ceased",
                "modality": "asserted",
                "event_time": "",
                "sources": [{
                    "source_ref": "src_1",
                    "evidence_excerpt": "I stopped tesamorelin after edema",
                }],
            }],
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor.summarize_segment(segment, roster=roster)

    assert len(provider.calls) == 2
    assert result.metadata.structured_summary.claims[0].temporal_status == ""


def test_summarize_segment_cannot_omit_a_safety_critical_source_lane():
    segment, roster = _structured_segment_and_roster()
    provider = _SequenceProvider(
        json.dumps({
            "summary": "BigTex stopped tesamorelin after edema.",
            "summary_claims": [],
        }),
        json.dumps({
            "summary": "BigTex stopped tesamorelin after edema.",
            "summary_claims": [{
                "text": "I stopped tesamorelin after edema",
                "claim_type": "personal",
                "temporal_status": "ceased",
                "modality": "asserted",
                "event_time": "",
                "sources": [{
                    "source_ref": "src_1",
                    "evidence_excerpt": "I stopped tesamorelin after edema",
                }],
            }],
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor.summarize_segment(segment, roster=roster)

    assert len(provider.calls) == 1
    assert len(result.metadata.structured_summary.claims) == 1
    assert result.metadata.structured_summary.claims[0].text == (
        "I stopped tesamorelin after edema and have not restarted it."
    )


@pytest.mark.parametrize(
    "source",
    [
        "I don’t take Tesa where did that come from?",
        "Glad to be off tesa.",
        "I am not taking tesamorelin.",
        "I'm not using tesamorelin.",
        "I do not currently take tesamorelin.",
        "I don't currently take tesamorelin.",
        "I have never taken tesamorelin.",
        "I never started tesamorelin.",
        "I didn't start tesamorelin.",
        "I haven't started tesamorelin.",
        "I did not stop tesamorelin.",
        "I never discontinued tesamorelin.",
        "I haven't completed the tesamorelin course.",
    ],
)
def test_empty_model_selection_cannot_omit_direct_correction_variants(source):
    segment, roster = _structured_segment_and_roster()
    segment.messages[0].content = source
    roster.lanes[0].text = source
    compactor = DomainCompactor(
        MockLLMProvider(), CompactorConfig(code_mode=False), model_name="test-model",
    )

    structured = compactor.build_structured_summary(
        {"summary_claims": []}, roster=roster, segment=segment,
    )

    assert [claim.text for claim in structured.claims] == [source]
    assert structured.claims[0].claim_type == "conversation"
    assert structured.claims[0].temporal_status == ""


def test_summarize_segment_strict_rejects_persistently_missing_claims():
    segment, roster = _structured_segment_and_roster()
    segment.messages[0].content = "I enjoy tea."
    roster.lanes[0].text = "I enjoy tea."
    empty = json.dumps({
        "summary": "Enjoys tea.",
        "summary_claims": [],
    })
    provider = _SequenceProvider(empty, empty)
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    with pytest.raises(SegmentSummaryGenerationError, match="no admissible"):
        compactor.summarize_segment(segment, roster=roster)

    assert len(provider.calls) == 2


def test_summarize_segment_strict_never_turns_provider_failure_into_v1_row():
    segment, roster = _structured_segment_and_roster()
    provider = _SequenceProvider(RuntimeError("provider unavailable"))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    with pytest.raises(RuntimeError, match="provider unavailable"):
        compactor.summarize_segment(segment, roster=roster)


def test_summarize_segment_strict_does_not_run_reply_lane_fact_calls():
    segment, roster = _structured_segment_and_roster()
    roster.reply_bearing = True
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex stopped tesamorelin after edema.",
        "summary_claims": [{
            "text": "I stopped tesamorelin after edema",
            "claim_type": "personal",
            "temporal_status": "ceased",
            "modality": "asserted",
            "event_time": "",
            "sources": [{
                "source_ref": "src_1",
                "evidence_excerpt": "I stopped tesamorelin after edema",
            }],
        }],
        "facts": [{
            "subject": "BigTex", "verb": "uses", "object": "tesamorelin",
        }],
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor.summarize_segment(segment, roster=roster)

    assert len(provider.calls) == 1
    assert result.facts == []


def test_summarize_segment_strict_rejects_persistently_malformed_json():
    segment, roster = _structured_segment_and_roster()
    provider = _SequenceProvider("not json", "still not json")
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    with pytest.raises(SegmentSummaryGenerationError, match="malformed"):
        compactor.summarize_segment(segment, roster=roster)
    assert len(provider.calls) == 2


def test_reply_subject_is_counted_and_free_synopsis_remains_retrieval_only(
    legal_segment,
):
    roster = ActorRoster(
        # ActorRoster.actor_ids intentionally mirrors the production requester
        # set; compaction proof must still count the subject lane below.
        actor_ids={"actor:discord:1"},
        labels={"bigtex": {"actor:discord:1"}},
        complete=True,
        reply_bearing=True,
        lanes=[
            FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text="What do you think?",
                actor_id="actor:discord:1",
                speaker_label="BigTex",
            ),
            FactLane(
                role=AUTHOR_ROLE_SUBJECT,
                text="I stopped the medication.",
                actor_id="actor:discord:2",
                speaker_label="Optics",
            ),
        ],
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex asked about Optics stopping the medication.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=roster,
    )
    result = compactor._compact_one(legal_segment, roster=roster)

    assert '["BigTex", "Optics"]' in request.prompt
    assert "actor:discord:" not in request.prompt
    assert result.metadata.source_speaker_labels == ["BigTex", "Optics"]
    assert result.metadata.source_speaker_identity_count == 2
    assert result.summary == "BigTex asked about Optics stopping the medication."
    assert result.metadata.structured_summary.claims == ()


def test_incomplete_or_partially_named_roster_gets_fail_closed_empty_contract(
    compactor,
    legal_segment,
):
    incomplete = _named_roster("BigTex", complete=False)
    partially_named = _named_roster("BigTex", "")

    for roster in (incomplete, partially_named, None):
        prompt = compactor.build_segment_summary_request(
            legal_segment, roster=roster,
        ).prompt
        assert "SUMMARY SPEAKER IDENTITY CONTRACT" in prompt
        assert "are: []." in prompt
        assert "actor:discord:" not in prompt


def test_colliding_display_label_does_not_authorize_named_attribution(
    compactor,
    legal_segment,
):
    roster = ActorRoster(
        actor_ids={"actor:discord:1", "actor:discord:2"},
        labels={"alex": {"actor:discord:1", "actor:discord:2"}},
        complete=True,
        lanes=[
            FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text="first disclosure",
                actor_id="actor:discord:1",
                speaker_label="Alex",
            ),
            FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text="second disclosure",
                actor_id="actor:discord:2",
                speaker_label="Alex",
            ),
        ],
    )

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=roster,
    )

    contract = request.prompt.split("SUMMARY SPEAKER IDENTITY CONTRACT", 1)[1]
    assert "[]" in contract
    assert "actor:discord:" not in request.prompt


def test_named_roster_ambiguous_summary_retries_then_accepts_exact_label(
    legal_segment,
):
    provider = _SequenceProvider(
        json.dumps({"summary": "The user stopped tesamorelin."}),
        json.dumps({"summary": "BigTex stopped tesamorelin."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 2
    assert result.summary == "BigTex stopped tesamorelin."
    assert result.metadata.source_speaker_labels == ["BigTex"]
    assert result.metadata.source_speaker_identity_count == 1
    assert len(result.metadata.source_speaker_identity_fingerprint) == 64
    assert "actor:discord:" not in result.metadata.source_speaker_identity_fingerprint
    assert len(result.metadata.source_audience_fingerprint) == 64
    assert "audience:guild:1" not in result.metadata.source_audience_fingerprint
    assert "ambiguous generic human referent" in provider.calls[1]["system"]


def test_ambiguous_summary_without_roster_still_retries_fail_closed(
    legal_segment,
):
    provider = _SequenceProvider(
        json.dumps({"summary": "A member stopped tesamorelin."}),
        json.dumps({"summary": "Tesamorelin cessation was discussed."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(legal_segment, roster=None)

    assert len(provider.calls) == 2
    assert result.summary == "Tesamorelin cessation was discussed."
    assert "are: []." in provider.calls[0]["user"]


def test_named_roster_second_ambiguous_summary_is_retained_only_as_index_text(
    legal_segment,
):
    legal_segment.messages[0].metadata = {
        "sender": {"name": "BigTex"},
        SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"],
    }
    provider = _SequenceProvider(
        json.dumps({"summary": "The user's tesamorelin use continued."}),
        json.dumps({"summary": "This person still uses tesamorelin."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 2
    assert result.summary == "This person still uses tesamorelin."
    assert result.metadata.structured_summary.claims == ()


def test_segment_identity_retry_failure_uses_source_fallback(legal_segment):
    legal_segment.messages[0].metadata = {
        "sender": {"name": "BigTex"},
        SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"],
    }
    provider = _SequenceProvider(
        json.dumps({"summary": "The user stopped tesamorelin."}),
        RuntimeError("retry provider failure"),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 2
    assert "BigTex" in result.summary
    assert "court filing deadline" in result.summary
    assert "The user stopped tesamorelin" not in result.summary


def test_named_roster_does_not_reject_technical_user_compound(legal_segment):
    provider = _SequenceProvider(
        json.dumps({"summary": "The user interface was updated."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=True))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 1
    assert result.summary == "The user interface was updated."


@pytest.mark.regression("BUG-004")
def test_compact_refined_tags(compactor, legal_segment):
    results = compactor.compact([legal_segment])
    assert "test-tag" in results[0].tags  # From mock LLM response


def test_compact_multiple(compactor):
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    segments = [
        TaggedSegment(
            primary_tag="legal",
            tags=["legal"],
            messages=[Message(role="user", content="Court case update")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
        TaggedSegment(
            primary_tag="medical",
            tags=["medical"],
            messages=[Message(role="user", content="Blood test results")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
    ]
    results = compactor.compact(segments)
    assert len(results) == 2


def test_preceding_context_never_crosses_proved_actor_boundary():
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    first = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "BigTex stopped tesamorelin.",
            canonical_id="ct-bigtex",
            actor="actor:discord:1",
            label="BigTex",
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )
    second = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "Kuw9239 discussed recovery.",
            canonical_id="ct-kuw",
            actor="actor:discord:2",
            label="Kuw9239",
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )

    def roster(actor: str, label: str) -> ActorRoster:
        return ActorRoster(
            actor_ids={actor},
            labels={label.casefold(): {actor}},
            complete=True,
            lanes=[FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text=f"source words from {label}",
                actor_id=actor,
                speaker_label=label,
            )],
        )

    provider = _SequenceProvider(
        json.dumps({"summary": "Named source statement."}),
    )
    compactor = DomainCompactor(
        provider,
        CompactorConfig(code_mode=False, max_concurrent_summaries=1),
    )
    compactor.compact(
        [first, second],
        actor_rosters_by_segment={
            first.id: roster("actor:discord:1", "BigTex"),
            second.id: roster("actor:discord:2", "Kuw9239"),
        },
    )

    second_prompt = next(
        call["user"] for call in provider.calls
        if "Kuw9239 discussed recovery." in call["user"]
    )
    assert "BigTex stopped tesamorelin." not in second_prompt
    assert "context_for_pronoun_resolution_only" not in second_prompt


@pytest.mark.parametrize(
    ("first_channel", "second_audience", "second_channel", "expect_prior"),
    [
        ("channel:medical", "audience:guild:1", "channel:medical", True),
        ("channel:medical", "audience:guild:2", "channel:medical", False),
        ("channel:medical", "audience:guild:1", "channel:other", False),
        ("", "audience:guild:1", "", True),
    ],
    ids=["same-scope", "cross-audience", "cross-channel", "same-dm-audience"],
)
def test_preceding_context_requires_same_actor_and_exact_audience_channel(
    first_channel,
    second_audience,
    second_channel,
    expect_prior,
):
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    actor = "actor:discord:1"
    first = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "BigTex stopped tesamorelin.",
            canonical_id="ct-1",
            actor=actor,
            label="BigTex",
            channel=first_channel,
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )
    second = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "BigTex discussed recovery.",
            canonical_id="ct-2",
            actor=actor,
            label="BigTex",
            audience=second_audience,
            channel=second_channel,
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )
    roster = _named_roster("BigTex")
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex discussed recovery.",
    }))
    compactor = DomainCompactor(
        provider,
        CompactorConfig(code_mode=False, max_concurrent_summaries=1),
    )

    results = compactor.compact(
        [first, second],
        actor_rosters_by_segment={first.id: roster, second.id: roster},
    )

    second_prompt = next(
        call["user"] for call in provider.calls
        if "BigTex discussed recovery." in call["user"]
    )
    assert ("BigTex stopped tesamorelin." in second_prompt) is expect_prior
    assert (
        "context_for_pronoun_resolution_only" in second_prompt
    ) is expect_prior
    assert results[1].summary == "BigTex discussed recovery."
    assert len(results[1].metadata.source_audience_fingerprint) == 64


def test_format_conversation(compactor):
    ts = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
    messages = [
        Message(role="user", content="Hello", timestamp=ts),
        Message(role="assistant", content="Hi there", timestamp=ts),
    ]
    text = compactor._format_conversation(messages)
    assert "Source (10:30): Hello" in text
    assert "Assistant (10:30): Hi there" in text


@pytest.mark.parametrize(
    ("first_label", "second_label"),
    [
        ("Alex", "Ａｌｅｘ"),
        ("Alex", "A\u200blex"),
    ],
    ids=["fullwidth", "default-ignorable"],
)
def test_format_conversation_neutralizes_normalized_label_collisions(
    compactor,
    first_label,
    second_label,
):
    messages = [
        Message(
            role="user",
            content="first source text",
            metadata={"sender": {"name": first_label}},
            source_actor_id="actor:discord:alex-one",
        ),
        Message(
            role="user",
            content="second source text",
            metadata={"sender": {"name": second_label}},
            source_actor_id="actor:discord:alex-two",
        ),
    ]

    text = compactor._format_conversation(messages)

    assert text == "Source: first source text\n\nSource: second source text"
    assert first_label not in text
    assert second_label not in text
    assert "actor:discord:" not in text


def test_format_conversation_preserves_distinct_safe_unicode_labels(compactor):
    labels = ["BigTex", "Renée", "李雷"]
    messages = [
        Message(
            role="user",
            content=f"source text {index}",
            metadata={"sender": {"name": label}},
            source_actor_id=f"actor:discord:{index}",
        )
        for index, label in enumerate(labels, 1)
    ]

    text = compactor._format_conversation(messages)

    assert text == (
        "BigTex: source text 1\n\n"
        "Renée: source text 2\n\n"
        "李雷: source text 3"
    )
    assert "actor:discord:" not in text


def test_format_conversation_neutralizes_generic_and_actor_id_labels(compactor):
    messages = [
        Message(
            role="user",
            content="generic source text",
            metadata={"sender": {"name": "User."}},
            source_actor_id="actor:discord:generic",
        ),
        Message(
            role="user",
            content="actor id source text",
            metadata={"sender": {"name": "Kuw9239"}},
            source_actor_id="Kuw9239",
        ),
    ]

    text = compactor._format_conversation(messages)

    assert text == "Source: generic source text\n\nSource: actor id source text"
    assert "User." not in text
    assert "Kuw9239" not in text


def test_format_conversation_missing_actor_poison_is_label_wide(compactor):
    messages = [
        Message(
            role="user",
            content="proved source text",
            metadata={"sender": {"name": "BigTex"}},
            source_actor_id="actor:discord:bigtex",
        ),
        Message(
            role="user",
            content="unproved source text",
            metadata={"sender": {"name": "Big\u200bTex"}},
        ),
    ]

    text = compactor._format_conversation(messages)

    assert text == "Source: proved source text\n\nSource: unproved source text"
    assert "BigTex" not in text
    assert "Big\u200bTex" not in text
    assert "actor:discord:" not in text


def test_parse_response_valid(compactor):
    result = compactor._parse_response('{"summary": "test", "entities": ["a"]}')
    assert result["summary"] == "test"


def test_parse_response_with_fences(compactor):
    result = compactor._parse_response('```json\n{"summary": "test"}\n```')
    assert result["summary"] == "test"


def test_parse_response_with_thinking(compactor):
    result = compactor._parse_response('<think>analyzing...</think>{"summary": "test"}')
    assert result["summary"] == "test"


def test_parse_response_fallback(compactor):
    result = compactor._parse_response("Just plain text summary")
    assert result["summary"] == "Just plain text summary"


def test_compact_retries_incomplete_json_summary(legal_segment):
    class RetryProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            if self.calls == 1:
                return "```json\n{", {}
            return '{"summary":"Recovered summary","refined_tags":[]}', {}

    provider = RetryProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor.compact([legal_segment])[0]

    assert provider.calls == 2
    assert result.summary == "Recovered summary"


def test_compact_uses_source_fallback_after_two_degenerate_summaries(legal_segment):
    class BrokenProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            return "```json\n{", {}

    provider = BrokenProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor.compact([legal_segment])[0]

    assert provider.calls == 2
    assert result.summary == result.full_text
    assert result.metadata.structured_summary.claims == ()
    assert not result.summary.startswith("```")


def test_compact_retries_summary_longer_than_long_source(legal_segment):
    class ContextPollutingProvider:
        def __init__(self):
            self.calls = []

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls.append((system, user))
            if len(self.calls) == 1:
                imported = (
                    "The user accepted an installation recommendation from the previous "
                    "conversation and the assistant completed a detailed deployment. " * 10
                )
                return json.dumps({"summary": imported, "refined_tags": []}), {}
            return '{"summary":"Discussion covered a court deadline.","refined_tags":[]}', {}

    import json

    provider = ContextPollutingProvider()
    legal_segment.messages[0].content += " " + ("filing detail " * 40)
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor._compact_one(
        legal_segment,
        prev_context="A long unrelated discussion about software installation.",
    )

    assert len(provider.calls) == 2
    assert result.summary == "Discussion covered a court deadline."
    retry_system = provider.calls[1][0]
    assert "Do not import prior context" in retry_system
    assert "invert negation or intent" in retry_system


def test_compact_short_source_immediately_falls_back_for_oversized_summary(
    legal_segment,
):
    class PollutingProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            return json.dumps({"summary": "unrelated history " * 100}), {}

    import json

    provider = PollutingProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor.compact([legal_segment])[0]

    assert provider.calls == 1
    assert result.summary == result.full_text
    assert result.metadata.structured_summary.claims == ()
    assert result.full_text == compactor._format_conversation(legal_segment.messages)


def test_default_prompt_requires_preserving_negation_and_intent():
    from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT

    assert "Preserve polarity, negation, intent" in DEFAULT_SUMMARY_PROMPT
    assert '"wants to remain infertile"' in DEFAULT_SUMMARY_PROMPT


def test_validator_no_longer_rejects_on_lexical_negation_overlap(ts):
    """The lexical negation-inversion heuristic is deliberately gone.

    It once caught a real inversion (this fixture's polarity flip is that
    incident), but in production it rejected 36-63% of faithful summaries
    of large or repetitive segments, because any source sentence with a
    negative marker sharing two stemmed terms with the summary counted as
    an inversion. Polarity protection now lives at the prompt layer, as it
    did for the months before the heuristic existed. This test pins the
    REMOVAL: the first response is accepted, no retry fires, and the trade
    is documented where the old test used to assert the opposite.
    """
    class InvertingProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            return (
                '{"summary":"Considered HCG.","refined_tags":[]}', {}
            )

    segment = TaggedSegment(
        primary_tag="hgh", tags=["hgh"],
        messages=[Message(
            role="user",
            content=(
                "I want to remain infertile and considered HCG for a stronger climax."
            ),
            timestamp=ts,
            metadata={"sender": {"name": "Reshi"}},
        )],
        token_count=20, start_timestamp=ts, end_timestamp=ts, turn_count=1,
    )
    provider = InvertingProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
    )
    result = compactor.compact([segment])[0]
    # A concise summary that omits the negated clause but shares stemmed
    # terms with the negative sentence is exactly what the old heuristic
    # rejected. It must now be accepted on the first call.
    assert provider.calls == 1, "no retry may fire on lexical overlap alone"
    assert result.summary == "Considered HCG."


def test_unusable_reason_names_degenerate_and_overshoot(ts):
    """Rejections carry their criterion, so the next validator defect is
    diagnosable from one log line instead of a week of archaeology."""
    assert DomainCompactor._unusable_reason("", "source text here") == "degenerate"
    assert DomainCompactor._unusable_reason(None, "source text here") == "degenerate"
    long_summary = "x" * 500
    assert DomainCompactor._unusable_reason(long_summary, "short") == "overshoot"
    assert DomainCompactor._unusable_reason("fine summary", "a much longer source text than the summary") is None

def test_custom_prompt_from_tag_rules():
    """Custom summary prompt should be used when tag matches a rule."""
    rules = [
        TagPromptRule(match="legal*", summary_prompt="Summarize legal matters carefully."),
    ]
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(),
        tag_rules=rules,
    )
    prompt = compactor._get_prompt_for_tags(["legal-case", "court"])
    assert prompt == "Summarize legal matters carefully."


def test_no_custom_prompt_for_unmatched_tags():
    rules = [
        TagPromptRule(match="legal*", summary_prompt="Summarize legal matters."),
    ]
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(),
        tag_rules=rules,
    )
    prompt = compactor._get_prompt_for_tags(["medical", "health"])
    assert prompt is None


def test_compact_tag_summaries_builds(mock_llm):
    """compact_tag_summaries builds summaries for tags with segments."""
    compactor = DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(),
    )

    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    summaries = [
        StoredSummary(
            ref="seg-1", primary_tag="legal", tags=["legal"],
            summary="Case discussion", summary_tokens=20,
            metadata=SegmentMetadata(
                source_speaker_labels=["BigTex"],
                source_speaker_identity_count=1,
                source_speaker_identity_fingerprint="speaker-proof",
                source_audience_fingerprint="scope-proof",
            ),
            created_at=now, start_timestamp=now, end_timestamp=now,
        ),
    ]

    result = compactor.compact_tag_summaries(
        cover_tags=["legal"],
        tag_to_summaries={"legal": summaries},
        tag_to_turns={"legal": [0, 1]},
        existing_tag_summaries={},
        max_turn=5,
    )
    assert len(result) == 1
    assert result[0].tag == "legal"
    assert result[0].covers_through_turn == 5
    assert len(mock_llm.calls) == 1


@pytest.mark.regression("BUG-003")
def test_compact_tag_summaries_skips_fresh(mock_llm):
    """compact_tag_summaries skips tags where existing summary is fresh."""
    compactor = DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(),
    )

    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    summaries = [
        StoredSummary(
            ref="seg-1", primary_tag="legal", tags=["legal"],
            summary="Case discussion", summary_tokens=20,
            created_at=now, start_timestamp=now, end_timestamp=now,
        ),
    ]
    existing = TagSummary(
        tag="legal", summary="Already fresh", summary_tokens=20,
        covers_through_turn=10,  # >= max_turn
        created_at=now, updated_at=now,
    )

    result = compactor.compact_tag_summaries(
        cover_tags=["legal"],
        tag_to_summaries={"legal": summaries},
        tag_to_turns={"legal": [0, 1]},
        existing_tag_summaries={"legal": existing},
        max_turn=5,  # existing covers_through_turn (10) >= max_turn (5)
    )
    assert len(result) == 0  # Nothing to build
    assert len(mock_llm.calls) == 0  # No LLM calls


def test_compact_tag_summaries_rebuilds_new_source_at_same_watermark():
    """Per-tag source additions cannot hide behind a global watermark."""
    provider = _SequenceProvider(json.dumps({
        "summary": "Current medication history.",
        "description": "Medication history.",
        "selected_claim_refs": ["claim_1", "claim_2"],
    }))
    compactor = DomainCompactor(
        provider, CompactorConfig(code_mode=False), model_name="tag-model",
    )
    newer, newer_stop = _stored_claim(
        "I stopped taking tesamorelin.",
        canonical_id="ct-new",
        label="BigTex",
        status="",
        ref="seg-new",
    )
    _older, older_active = _stored_claim(
        "I was still taking tesamorelin then.",
        canonical_id="ct-old",
        label="BigTex",
        status="",
        ref="seg-old",
    )
    existing = TagSummary(
        tag="medical",
        summary="Prior tag synopsis.",
        description="Prior description.",
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(older_active,),
            source_digest=structured_tag_claim_digest(
                (older_active,), ("ct-old",),
            ),
            generation_model="old-model",
        ),
        source_segment_refs=["seg-old"],
        source_turn_numbers=[4],
        source_canonical_turn_ids=["ct-old"],
        covers_through_turn=9,
        covers_through_canonical_turn_id="ct-watermark",
    )

    [result] = compactor.compact_tag_summaries(
        cover_tags=["medical"],
        tag_to_summaries={"medical": [newer]},
        tag_to_turns={"medical": [7]},
        tag_to_canonical_turn_ids={"medical": ["ct-new"]},
        existing_tag_summaries={"medical": existing},
        max_turn=9,
        validated_tag_rollup_inputs=_tag_rollup_proof(newer),
    )

    assert len(provider.calls) == 1
    assert result.covers_through_turn == 9
    assert result.source_segment_refs == ["seg-new", "seg-old"]
    assert result.source_canonical_turn_ids == ["ct-new", "ct-old"]
    assert result.structured_summary.claims == (newer_stop, older_active)


def test_incremental_tag_rollup_keeps_prior_claims_and_prior_source_coverage():
    """A bounded latest-segment window must not erase layer-two history."""
    provider = _SequenceProvider(json.dumps({
        "summary": "Latest cessation plus prior medication history.",
        "description": "Medication history across sessions.",
        "selected_claim_refs": ["claim_1", "claim_2"],
    }))
    compactor = DomainCompactor(
        provider, CompactorConfig(code_mode=False), model_name="tag-model",
    )

    newest = _structured_stored_summary()
    newest.ref = "seg-new"
    _old_summary, old_claim = _stored_claim(
        "I was still taking tesamorelin then.",
        canonical_id="ct-old",
        label="BigTex",
        status="",
        ref="seg-old",
    )
    existing = TagSummary(
        tag="medical",
        summary="Prior tag synopsis.",
        description="Prior description.",
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(old_claim,),
            source_digest=structured_tag_claim_digest(
                (old_claim,), ("ct-old",),
            ),
            generation_model="old-model",
        ),
        source_segment_refs=["seg-old"],
        source_turn_numbers=[1],
        source_canonical_turn_ids=["ct-old"],
        covers_through_turn=1,
        covers_through_canonical_turn_id="ct-old",
    )

    [result] = compactor.compact_tag_summaries(
        cover_tags=["medical"],
        tag_to_summaries={"medical": [newest]},
        tag_to_turns={"medical": [9]},
        tag_to_canonical_turn_ids={"medical": ["ct-1"]},
        existing_tag_summaries={"medical": existing},
        max_turn=9,
        validated_tag_rollup_inputs=_tag_rollup_proof(newest),
    )

    assert result.structured_summary.claims[0].text.startswith("I stopped")
    assert result.structured_summary.claims[1].text.startswith("I was still")
    assert result.source_segment_refs == ["seg-new", "seg-old"]
    assert result.source_turn_numbers == [1, 9]
    assert result.source_canonical_turn_ids == ["ct-1", "ct-old"]
    assert "Prior tag synopsis." in provider.calls[0]["user"]


# ---------------------------------------------------------------------------
# TagSummary.description extraction
# ---------------------------------------------------------------------------


class TestTagSummaryDescription:
    """Tests for TagSummary.description field populated from rollup LLM response."""

    def test_description_extracted_from_rollup_response(self):
        """Mock LLM returns JSON with description — verify TagSummary.description is set."""
        mock_llm = MockLLMProvider(
            response=(
                '{"summary": "Cycle tracking discussion", '
                '"description": "Sania\'s cycle tracking via Mira", '
                '"entities": ["Sania", "Mira"], '
                '"key_decisions": ["use Mira device"], '
                '"action_items": []}'
            )
        )
        compactor = DomainCompactor(
            llm_provider=mock_llm,
            config=CompactorConfig(),
        )

        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        summaries = [
            StoredSummary(
                ref="seg-1", primary_tag="cycle-tracking",
                tags=["cycle-tracking"],
                summary="Discussed Mira device for cycle tracking",
                summary_tokens=30,
                metadata=SegmentMetadata(
                    source_speaker_labels=["Sania"],
                    source_speaker_identity_count=1,
                    source_speaker_identity_fingerprint="sania-proof",
                    source_audience_fingerprint="scope-proof",
                ),
                created_at=now, start_timestamp=now, end_timestamp=now,
            ),
        ]

        result = compactor.compact_tag_summaries(
            cover_tags=["cycle-tracking"],
            tag_to_summaries={"cycle-tracking": summaries},
            tag_to_turns={"cycle-tracking": [0, 1, 2]},
            existing_tag_summaries={},
            max_turn=5,
        )
        assert len(result) == 1
        assert result[0].description == "Sania's cycle tracking via Mira"

    def test_description_fallback_when_omitted(self):
        """Mock LLM returns JSON without description key — verify description == ''."""
        mock_llm = MockLLMProvider(
            response=(
                '{"summary": "Legal case discussion", '
                '"entities": ["Judge Smith"], '
                '"key_decisions": ["file motion"], '
                '"action_items": []}'
            )
        )
        compactor = DomainCompactor(
            llm_provider=mock_llm,
            config=CompactorConfig(),
        )

        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        summaries = [
            StoredSummary(
                ref="seg-1", primary_tag="legal",
                tags=["legal"],
                summary="Case discussion",
                summary_tokens=20,
                metadata=SegmentMetadata(
                    source_speaker_labels=["BigTex"],
                    source_speaker_identity_count=1,
                    source_speaker_identity_fingerprint="speaker-proof",
                    source_audience_fingerprint="scope-proof",
                ),
                created_at=now, start_timestamp=now, end_timestamp=now,
            ),
        ]

        result = compactor.compact_tag_summaries(
            cover_tags=["legal"],
            tag_to_summaries={"legal": summaries},
            tag_to_turns={"legal": [0, 1]},
            existing_tag_summaries={},
            max_turn=5,
        )
        assert len(result) == 1
        assert result[0].description == ""


def test_tag_rollup_removes_generic_user_priming_and_prefixes_source_labels():
    provider = _SequenceProvider(
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "BigTex discussed stopping tesamorelin.",
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert "the user" not in TAG_SUMMARY_ROLLUP_PROMPT.lower()
    assert 'source display labels: ["BigTex"]' in provider.calls[0]["user"]
    assert "SPEAKER IDENTITY CONTRACT" in provider.calls[0]["user"]
    assert "actor:" not in provider.calls[0]["user"]
    assert result.summary == "BigTex stopped tesamorelin."


def test_summarize_tag_strict_copies_segment_claims_without_reauthoring():
    provider = _SequenceProvider(json.dumps({
        "summary": "Tesamorelin cessation after edema.",
        "description": "A source-backed medication history.",
        "selected_claim_refs": ["claim_1"],
    }))
    compactor = DomainCompactor(
        provider, CompactorConfig(code_mode=False), model_name="tag-model",
    )

    summary = _structured_stored_summary()
    result = compactor.summarize_tag(
        "medical", [summary], [7], ["ct-1"], 7,
        validated_tag_rollup_inputs=_tag_rollup_proof(summary),
    )

    assert result.summary == "Tesamorelin cessation after edema."
    assert len(result.structured_summary.claims) == 1
    claim = result.structured_summary.claims[0]
    assert claim.text == "I stopped tesamorelin after edema."
    assert claim.temporal_status == ""
    assert claim.sources[0].speaker_label == "BigTex"
    assert result.structured_summary.generation_model == "tag-model"


def test_tag_claim_selection_prompt_uses_only_ephemeral_refs():
    summary, _claim = _stored_claim(
        "I stopped tesamorelin after edema.",
        canonical_id="ct-private-bigtex",
        label="BigTex",
        status="",
        ref="seg-private",
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "Medication history.",
        "description": "Source-backed medication history.",
        "selected_claim_refs": ["claim_1"],
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    compactor._build_one_tag_summary(
        "medical", [summary], [7], ["ct-private-bigtex"], 7,
        validated_tag_rollup_inputs=_tag_rollup_proof(summary),
    )

    prompt = provider.calls[0]["user"]
    assert "SOURCE-BOUND LAYER-TWO CLAIM SELECTION" in prompt
    assert '"claim_ref":"claim_1"' in prompt
    assert '"speaker":"BigTex"' in prompt
    assert "ct-private-bigtex" not in prompt
    assert "c" * 64 not in prompt


def test_tag_selection_cannot_rewrite_bigtex_ceased_claim_as_active():
    newest, ceased = _stored_claim(
        "I stopped tesamorelin after edema.",
        canonical_id="ct-new",
        label="BigTex",
        status="",
        ref="seg-new",
    )
    older, _active = _stored_claim(
        "I was still taking tesamorelin then.",
        canonical_id="ct-old",
        label="BigTex",
        status="",
        ref="seg-old",
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex currently takes tesamorelin.",
        "description": "Generated retrieval prose can be wrong.",
        # Even a syntactically valid attempt to select only the older active
        # state cannot suppress the newer cessation anchor.
        "selected_claim_refs": ["claim_2"],
        "summary_claims": [{
            "text": "BigTex currently takes tesamorelin.",
            "temporal_status": "active",
        }],
    }))
    compactor = DomainCompactor(
        provider, CompactorConfig(code_mode=False), model_name="tag-model",
    )

    result = compactor.summarize_tag(
        "medical", [newest, older], [7, 8], ["ct-new", "ct-old"], 8,
        validated_tag_rollup_inputs=_tag_rollup_proof(newest, older),
    )

    assert result.summary == "BigTex currently takes tesamorelin."
    assert result.structured_summary.claims == (ceased, _active)
    assert result.structured_summary.claims[0] is ceased
    assert result.structured_summary.claims[0].temporal_status == ""
    assert result.structured_summary.claims[0].sources[0].canonical_turn_id == "ct-new"


def test_tag_pool_prioritizes_critical_lane_over_model_selected_context():
    old_summary, old_active = _stored_claim(
        "I am currently taking tesamorelin.",
        canonical_id="ct-old",
        label="BigTex",
        status="",
        ref="seg-both",
    )
    _new_summary, new_ceased = _stored_claim(
        "I stopped taking tesamorelin.",
        canonical_id="ct-new",
        label="BigTex",
        status="",
        ref="seg-new",
    )
    # Segment model order is actively misleading: it selected the old active
    # lane first and the deterministic segment floor appended the newer stop.
    old_summary.metadata.canonical_turn_ids = ["ct-old", "ct-new"]
    old_summary.metadata.structured_summary = StructuredSummary(
        schema_version=1,
        claims=(new_ceased, old_active),
        source_digest="e" * 64,
        generation_model="segment-model",
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "Medication history.",
        "description": "Medication history.",
        "selected_claim_refs": ["claim_2"],
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor.summarize_tag(
        "medical", [old_summary], [7], ["ct-old", "ct-new"], 7,
        validated_tag_rollup_inputs=_tag_rollup_proof(old_summary),
    )

    # The exact correction is mandatory and always precedes the ordinary
    # model-selected historical context; the latter may remain as context.
    assert result.structured_summary.claims == (new_ceased, old_active)
    prompt = provider.calls[0]["user"]
    assert "I stopped taking tesamorelin." in prompt
    assert "I am currently taking tesamorelin." in prompt


def test_tag_safety_floor_preserves_authenticated_same_day_and_unrelated_claims():
    restarted_summary, _restarted = _stored_claim(
        "I have restarted tesamorelin.",
        canonical_id="ct-restart",
        label="BigTex",
        status="",
        ref="seg-restart",
    )
    stopped_summary, _stopped = _stored_claim(
        "I stopped tesamorelin this morning.",
        canonical_id="ct-stop",
        label="BigTex",
        status="",
        ref="seg-stop",
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "Medication history.",
        "description": "Medication history.",
        # Both exact transitions are mandatory; segment/source order, not a
        # day-only timestamp, proves which lane is newer.
        "selected_claim_refs": ["claim_2"],
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor.summarize_tag(
        "medical",
        [restarted_summary, stopped_summary],
        [7, 8],
        ["ct-restart", "ct-stop"],
        8,
        validated_tag_rollup_inputs=_tag_rollup_proof(
            restarted_summary, stopped_summary,
        ),
    )
    # Authenticated segment order preserves the newer restart ahead of the
    # older stop even when both physical rows share one session date.
    assert result.structured_summary.claims == (_restarted, _stopped)

    alcohol_summary, alcohol_stop = _stored_claim(
        "I stopped using alcohol.",
        canonical_id="ct-alcohol",
        label="BigTex",
        status="",
        ref="seg-alcohol",
    )
    medication_summary, medication_active = _stored_claim(
        "I am currently taking tesamorelin.",
        canonical_id="ct-medication",
        label="BigTex",
        status="",
        ref="seg-medication",
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "Health history.",
        "description": "Health history.",
        "selected_claim_refs": ["claim_2"],
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor.summarize_tag(
        "health",
        [alcohol_summary, medication_summary],
        [7, 8],
        ["ct-alcohol", "ct-medication"],
        8,
        validated_tag_rollup_inputs=_tag_rollup_proof(
            alcohol_summary, medication_summary,
        ),
    )
    assert result.structured_summary.claims == (
        alcohol_stop, medication_active,
    )


def test_tag_selection_dedupes_refs_and_preserves_multi_human_order():
    bigtex_summary, bigtex_claim = _stored_claim(
        "I stopped tesamorelin.",
        canonical_id="ct-bigtex",
        label="BigTex",
        status="",
        ref="seg-bigtex",
    )
    sania_summary, sania_claim = _stored_claim(
        "I scheduled the follow-up.",
        canonical_id="ct-sania",
        label="Sania",
        status="",
        ref="seg-sania",
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "Two-person care history.",
        "description": "BigTex and Sania care history.",
        "selected_claim_refs": ["claim_1", "claim_1", "claim_2"],
    }))
    compactor = DomainCompactor(
        provider, CompactorConfig(code_mode=False), model_name="tag-model",
    )

    result = compactor.summarize_tag(
        "medical",
        [sania_summary, bigtex_summary],
        [7, 8],
        ["ct-bigtex", "ct-sania"],
        8,
        validated_tag_rollup_inputs=_tag_rollup_proof(
            sania_summary, bigtex_summary,
        ),
    )

    # The exact cessation correction is safety-critical, so it precedes the
    # model-selected claim while both speakers remain independently bound.
    assert result.structured_summary.claims == (bigtex_claim, sania_claim)
    assert [
        claim.sources[0].speaker_label
        for claim in result.structured_summary.claims
    ] == ["BigTex", "Sania"]


def test_tag_claim_digest_authenticates_order_and_cardinality():
    first_summary, first = _stored_claim(
        "I stopped tesamorelin.",
        canonical_id="ct-new",
        label="BigTex",
        status="",
        ref="seg-new",
    )
    _ = first_summary
    second_summary, second = _stored_claim(
        "I scheduled the follow-up.",
        canonical_id="ct-plan",
        label="Sania",
        status="",
        ref="seg-plan",
    )
    _ = second_summary

    source_ids = ("ct-new", "ct-plan")
    assert structured_tag_claim_digest((first, second), source_ids) != (
        structured_tag_claim_digest((second, first), source_ids)
    )
    assert structured_tag_claim_digest((first,), source_ids) != (
        structured_tag_claim_digest((first, first), source_ids)
    )
    assert structured_tag_claim_digest((first,), source_ids) != (
        structured_tag_claim_digest((first,), tuple(reversed(source_ids)))
    )
    assert structured_tag_claim_digest((first,), source_ids) != (
        structured_tag_claim_digest((first,), source_ids[:1])
    )


def test_tag_rollup_proof_rejects_parent_omitting_critical_source_claim():
    parent, old_active = _stored_claim(
        "I am currently taking tesamorelin.",
        canonical_id="ct-old",
        label="BigTex",
        status="",
        ref="seg-parent",
    )
    _stop_summary, newer_stop = _stored_claim(
        "I stopped taking tesamorelin.",
        canonical_id="ct-stop",
        label="BigTex",
        status="",
        ref="seg-stop",
    )
    rows = [_physical_row_for_claim(old_active), _physical_row_for_claim(newer_stop)]
    parent.metadata.canonical_turn_ids = ["ct-old", "ct-stop"]
    parent.metadata.structured_summary = StructuredSummary(
        schema_version=1,
        # The aggregate digest and source-id coverage are valid, but the
        # selected claims maliciously omit the newer stop lane.
        claims=(old_active,),
        source_digest=structured_source_digest(
            tuple(_structured_record_for_row(row) for row in rows),
            namespace="segment",
        ),
        generation_model="segment-model",
    )
    proof = validate_tag_rollup_inputs(
        [parent],
        {row.canonical_turn_id: row for row in rows},
        conversation_id="conv-test",
    )

    assert not proof.admits(parent)
    provider = _SequenceProvider(json.dumps({
        "summary": "Retrieval-only medication synopsis.",
        "description": "Medication history.",
        "selected_claim_refs": ["claim_1"],
    }))
    result = DomainCompactor(
        provider, CompactorConfig(code_mode=False),
    )._build_one_tag_summary(
        "medical",
        [parent],
        [7],
        ["ct-old", "ct-stop"],
        7,
        validated_tag_rollup_inputs=proof,
    )
    assert result.structured_summary.claims == ()


def test_tag_rollup_proof_cannot_be_reused_after_presented_source_mutation():
    summary, claim = _stored_claim(
        "I stopped taking tesamorelin.",
        canonical_id="ct-stop",
        label="BigTex",
        status="",
        ref="seg-stop",
    )
    proof = _tag_rollup_proof(summary)
    assert proof.admits(summary)

    forged_source = replace(
        claim.sources[0],
        speaker_label="actor:discord:secret",
        session_date="ct-secret",
    )
    summary.metadata.structured_summary = replace(
        summary.metadata.structured_summary,
        claims=(replace(claim, sources=(forged_source,)),),
    )
    assert not proof.admits(summary)

    provider = _SequenceProvider(json.dumps({
        "summary": "Retrieval-only medication synopsis.",
        "description": "Medication history.",
        "selected_claim_refs": ["claim_1"],
    }))
    result = DomainCompactor(
        provider, CompactorConfig(code_mode=False),
    )._build_one_tag_summary(
        "medical",
        [summary],
        [7],
        ["ct-stop"],
        7,
        validated_tag_rollup_inputs=proof,
    )
    assert result.structured_summary.claims == ()
    assert "actor:discord:secret" not in provider.calls[0]["user"]
    assert "ct-secret" not in provider.calls[0]["user"]


def test_invalid_prior_tag_digest_cannot_carry_old_claim_or_source_ids():
    fresh, stopped = _stored_claim(
        "I stopped taking tesamorelin.",
        canonical_id="ct-new",
        label="BigTex",
        status="",
        ref="seg-new",
    )
    _old_summary, old_active = _stored_claim(
        "I am currently taking tesamorelin.",
        canonical_id="ct-old",
        label="BigTex",
        status="",
        ref="seg-old",
    )
    prior_ids = ["ct-old", "ct-prior-extra"]
    prior = TagSummary(
        tag="medical",
        summary="Prior retrieval synopsis.",
        source_segment_refs=["seg-old"],
        source_turn_numbers=[1],
        source_canonical_turn_ids=prior_ids,
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(old_active,),
            source_digest=structured_tag_claim_digest(
                (old_active,), prior_ids,
            ),
            generation_model="old-tag-model",
        ),
        covers_through_turn=1,
    )
    # Deleting a source id without regenerating the ordered digest invalidates
    # the entire prior structured envelope.
    prior.source_canonical_turn_ids = ["ct-old"]
    provider = _SequenceProvider(json.dumps({
        "summary": "Current medication history.",
        "description": "Medication history.",
        "selected_claim_refs": ["claim_1"],
    }))
    result = DomainCompactor(
        provider, CompactorConfig(code_mode=False),
    )._build_one_tag_summary(
        "medical",
        [fresh],
        [9],
        ["ct-new"],
        9,
        existing_tag_summary=prior,
        validated_tag_rollup_inputs=_tag_rollup_proof(fresh),
    )

    assert result.structured_summary.claims == (stopped,)
    assert result.source_canonical_turn_ids == ["ct-new"]
    assert result.source_segment_refs == ["seg-new"]


def test_invalid_tag_selection_falls_back_online_but_strict_fails_closed():
    summary, claim = _stored_claim(
        "I stopped tesamorelin.",
        canonical_id="ct-bigtex",
        label="BigTex",
        status="",
        ref="seg-bigtex",
    )
    response = json.dumps({
        "summary": "Medication history.",
        "description": "Medication history.",
        "selected_claim_refs": ["claim_999"],
    })
    online = DomainCompactor(
        _SequenceProvider(response), CompactorConfig(code_mode=False),
    )
    strict = DomainCompactor(
        _SequenceProvider(response), CompactorConfig(code_mode=False),
    )

    result = online._build_one_tag_summary(
        "medical", [summary], [7], ["ct-bigtex"], 7,
        validated_tag_rollup_inputs=_tag_rollup_proof(summary),
    )
    assert result.structured_summary.claims == (claim,)
    with pytest.raises(TagSummaryGenerationError):
        strict.summarize_tag(
            "medical", [summary], [7], ["ct-bigtex"], 7,
            validated_tag_rollup_inputs=_tag_rollup_proof(summary),
        )


@pytest.mark.parametrize(
    "response",
    [
        RuntimeError("provider down"),
        "not json",
        json.dumps({"summary": "Valid synopsis, missing selection."}),
    ],
    ids=["provider", "malformed", "missing-selection"],
)
def test_summarize_tag_strict_never_persists_provider_or_parse_fallback(response):
    provider = _SequenceProvider(response)
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    expected = RuntimeError if isinstance(response, RuntimeError) else TagSummaryGenerationError
    summary = _structured_stored_summary()
    with pytest.raises(expected):
        compactor.summarize_tag(
            "medical", [summary], [7], ["ct-1"], 7,
            validated_tag_rollup_inputs=_tag_rollup_proof(summary),
        )


@pytest.mark.parametrize(
    "unsafe_label",
    [
        "actor:discord:1",
        "User.",
        "(You)",
        "U\u200bser",
        "actor\u2060:discord:1",
    ],
    ids=[
        "actor-id",
        "decorated-generic",
        "decorated-pronoun",
        "zw-generic",
        "zw-actor-id",
    ],
)
def test_unsafe_label_never_enters_tag_rollup_source_block(unsafe_label):
    source = _stored_summary(
        "Tesamorelin cessation was discussed.",
        label=unsafe_label,
    )

    rendered = _format_tag_rollup_source(source)

    assert "source display labels" not in rendered
    assert unsafe_label not in rendered


def test_compatibility_equivalent_tag_source_labels_fail_closed():
    source = _stored_summary(
        "Tesamorelin cessation was discussed.",
        label="Alex",
    )
    source.metadata.source_speaker_labels = ["Alex", "Ａｌｅｘ"]
    source.metadata.source_speaker_identity_count = 2

    rendered = _format_tag_rollup_source(source)

    assert "source display labels" not in rendered
    assert "Ａｌｅｘ" not in rendered


@pytest.mark.parametrize("label", ["BigTex", "Renée", "李雷"])
def test_valid_unicode_label_enters_tag_rollup_source_block(label):
    rendered = _format_tag_rollup_source(
        _stored_summary("A source-grounded topic was discussed.", label=label),
    )

    expected = json.dumps([label], ensure_ascii=False)
    assert f"source display labels: {expected}" in rendered


@pytest.mark.parametrize("unsafe_field", ["summary", "description"])
def test_tag_rollup_retries_when_either_prose_field_has_ambiguous_referent(
    unsafe_field,
):
    first = {
        "summary": "BigTex stopped tesamorelin.",
        "description": "BigTex discussed stopping tesamorelin.",
    }
    first[unsafe_field] = "The user stopped tesamorelin."
    provider = _SequenceProvider(
        json.dumps(first),
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "BigTex discussed stopping tesamorelin.",
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert unsafe_field in provider.calls[1]["system"]
    assert result.summary == "BigTex stopped tesamorelin."
    assert result.description == "BigTex discussed stopping tesamorelin."


def test_tag_rollup_second_ambiguous_result_uses_safe_source_concatenation():
    provider = _SequenceProvider(
        json.dumps({
            "summary": "The member stopped tesamorelin.",
            "description": "Medical discussion.",
        }),
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "This person's medical discussion.",
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert 'source display labels: ["BigTex"]' in result.summary
    assert "BigTex stopped tesamorelin." in result.summary
    assert "This person" not in result.summary
    assert result.description == ""


def test_tag_rollup_retry_failure_uses_safe_source_fallback():
    provider = _SequenceProvider(
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "The user's medical discussion.",
        }),
        RuntimeError("retry provider failure"),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert 'source display labels: ["BigTex"]' in result.summary
    assert "BigTex stopped tesamorelin." in result.summary
    assert result.description == ""


def test_tag_rollup_quarantines_when_all_source_prose_is_ambiguous():
    provider = _SequenceProvider(
        json.dumps({"summary": "The user stopped tesamorelin."}),
        json.dumps({"summary": "A member stopped tesamorelin."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary(
            "The user's tesamorelin experience ended.", label="BigTex",
        )],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert result.summary == (
        "[tag summary withheld: source prose lacks explicit speaker attribution]"
    )
    assert result.description == ""


def test_multi_speaker_tag_rollup_prose_is_retrieval_only():
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex takes the protocol Kuw9239 described.",
        "description": "A synthesized personal claim.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [
            _stored_summary(
                "BigTex stopped tesamorelin.", ref="seg-bigtex", label="BigTex",
            ),
            _stored_summary(
                "Kuw9239 tolerates tesamorelin.", ref="seg-kuw", label="Kuw9239",
            ),
        ],
        [1, 2],
        ["ct-1", "ct-2"],
        2,
    )

    assert len(provider.calls) == 1
    assert result.summary == "BigTex takes the protocol Kuw9239 described."
    assert result.description == "A synthesized personal claim."
    assert result.structured_summary.claims == ()


def test_same_actor_cross_audience_tag_rollup_prose_is_retrieval_only():
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex combined private and guild disclosures.",
        "description": "Cross-audience synthesis.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [
            _stored_summary(
                "BigTex discussed a private concern.",
                ref="seg-private",
                label="BigTex",
                identity="same-actor-proof",
                scope="private-audience-proof",
            ),
            _stored_summary(
                "BigTex discussed a guild concern.",
                ref="seg-guild",
                label="BigTex",
                identity="same-actor-proof",
                scope="guild-audience-proof",
            ),
        ],
        [1, 2],
        ["ct-private", "ct-guild"],
        2,
    )

    assert len(provider.calls) == 1
    assert result.summary == "BigTex combined private and guild disclosures."
    assert result.description == "Cross-audience synthesis."
    assert result.structured_summary.claims == ()


def test_same_display_label_tag_rollup_prose_is_retrieval_only():
    provider = _SequenceProvider(json.dumps({
        "summary": "Alex stopped tesamorelin.",
        "description": "A synthesized claim.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [
            _stored_summary(
                "Alex discussed tesamorelin.",
                ref="seg-alex-1",
                label="Alex",
                identity="actor-proof-1",
            ),
            _stored_summary(
                "Alex reported edema.",
                ref="seg-alex-2",
                label="Alex",
                identity="actor-proof-2",
            ),
        ],
        [1, 2],
        ["ct-1", "ct-2"],
        2,
    )

    assert len(provider.calls) == 1
    assert result.summary == "Alex stopped tesamorelin."
    assert result.description == "A synthesized claim."
    assert result.structured_summary.claims == ()


def test_faithful_summary_of_repetitive_machine_content_is_accepted():
    """The production failure shape, as a fixture.

    Large segments of near-duplicate machine records (home-automation
    event JSON) almost always contain negative markers ("no person
    detected"), and any faithful summary shares stemmed topic terms with
    those sentences while containing no negation token of its own. The
    removed heuristic rejected every such summary; measured in production
    at 36-63% of new segments, concentrated on the largest conversations.
    This fixture is shaped like the confirmed production specimen (real
    conversation content stays out of a public repository); the live
    specimen itself is re-verified operationally after deploy.
    """
    records = []
    for i in range(40):
        records.append(
            '{"before": {"id": "1785207%03d.77-x", "camera": "cam1", '
            '"label": "person", "stationary": false, "score": 0.58%02d}}' % (i, i)
        )
        if i % 7 == 3:
            records.append(
                "System: no person detected on camera cam1; event not retained, "
                "snapshot won't be stored."
            )
    source = "\n".join(records)
    summary = (
        "Processed a batch of camera person-detection events from cam1, "
        "with several events concluding as stationary detections and "
        "snapshots stored for the retained events."
    )
    assert len(source) > 4000, "fixture must be in the large-segment regime"
    assert DomainCompactor._unusable_reason(summary, source) is None, (
        "a faithful summary of repetitive machine content must be accepted"
    )


def test_five_faithful_rewordings_are_all_accepted():
    """Acceptance must not depend on wording luck.

    Under the removed heuristic, three of these five faithful rewordings
    were rejected and two passed, purely on whether the phrasing happened
    to include a negation token. All five must be accepted.
    """
    source = (
        "User: I don't want the pipeline analytics change yet.\n"
        "Assistant: Updated the staging pipeline configuration."
    )
    rewordings = [
        "Updated the staging pipeline configuration; the analytics change is on hold.",
        "Deferred the analytics change; staging updated.",
        "Held the analytics change; updated staging.",
        "Decided not to apply the analytics change yet; staging updated.",
        "The analytics change won't be applied yet; staging updated.",
    ]
    for summary in rewordings:
        assert DomainCompactor._unusable_reason(summary, source) is None, summary


def test_summary_request_conversation_text_override(ts):
    """The override must replace the formatted text everywhere it flows.

    A caller repairing a stored segment builds the request from the stored
    full_text; the prompt, the length-derived token target, and the
    request's own conversation_text must all come from the override, and
    the formatted message text must not appear anywhere.
    """
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )
    segment = TaggedSegment(
        primary_tag="legal",
        tags=["legal", "court"],
        messages=[
            Message(role="user", content="What's the court filing deadline?", timestamp=ts),
            Message(role="assistant", content="The filing is due January 30.", timestamp=ts + timedelta(seconds=30)),
        ],
        token_count=50,
        start_timestamp=ts,
        end_timestamp=ts + timedelta(seconds=30),
        turn_count=1,
    )
    formatted = compactor._format_conversation(segment.messages)

    # Passing the formatted text explicitly is identical to not passing it.
    assert compactor.build_segment_summary_request(
        segment, conversation_text=formatted,
    ) == compactor.build_segment_summary_request(segment)

    stored = "Stored transcript bytes " * 200  # long enough to move target_tokens
    req = compactor.build_segment_summary_request(
        segment, conversation_text=stored,
    )
    assert req.conversation_text == stored
    assert stored[:200] in req.prompt
    assert "court filing deadline" not in req.prompt
    assert req.original_tokens == compactor.token_counter(stored)
    expected_target = max(50, min(500, int(req.original_tokens * 0.15)))
    assert req.target_tokens == expected_target
    assert str(expected_target) in req.prompt
