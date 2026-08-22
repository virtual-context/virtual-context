"""Tests for ContextAssembler (tag-based)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from virtual_context.core.assembler import ContextAssembler, format_tag_section
from virtual_context.core.summary_identity import SUMMARY_ATTRIBUTION_QUARANTINE
from virtual_context.core.structured_summary import (
    structured_source_digest,
    structured_source_provenance_digest,
    structured_tag_claim_digest,
)
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    AssemblerConfig,
    DepthLevel,
    Message,
    RetrievalResult,
    SegmentMetadata,
    SpeakerRetrievalContext,
    StoredSegment,
    StoredSummary,
    StructuredSummary,
    SummaryClaim,
    SummarySource,
    TagSummary,
    TagPromptRule,
    WorkingSetEntry,
)


@pytest.fixture
def assembler():
    return ContextAssembler(
        config=AssemblerConfig(
            core_context_max_tokens=1000,
            tag_context_max_tokens=2000,
        )
    )


@pytest.fixture
def retrieval_result():
    now = datetime.now(timezone.utc)
    return RetrievalResult(
        tags_matched=["legal"],
        summaries=[
            StoredSummary(
                ref="ref-1",
                primary_tag="legal",
                tags=["legal", "court"],
                summary="Case 24-cv-1234 discussed. Filing due Jan 30.",
                summary_tokens=20,
                full_tokens=100,
                metadata=SegmentMetadata(entities=["Case 24-cv-1234"]),
                created_at=now,
                start_timestamp=now,
                end_timestamp=now,
            ),
        ],
        total_tokens=20,
    )


def test_assemble_basic(assembler, retrieval_result):
    history = [
        Message(role="user", content="What about the case?"),
        Message(role="assistant", content="Let me check."),
    ]
    result = assembler.assemble(
        core_context="# IDENTITY\nYou are a helpful assistant.",
        retrieval_result=retrieval_result,
        conversation_history=history,
        token_budget=10000,
    )
    assert result.total_tokens > 0
    assert "legal" in result.tag_sections
    assert len(result.conversation_history) == 2


def test_assemble_xml_tags(assembler, retrieval_result):
    result = assembler.assemble(
        core_context="",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
    )
    section = result.tag_sections.get("legal", "")
    assert '<virtual-context tags="court, legal"' in section
    assert "last_updated=" not in section
    assert "</virtual-context>" in section


def test_format_tag_section_omits_unproved_code_refs():
    now = datetime.now(timezone.utc)
    section = format_tag_section(
        "backend",
        [
            StoredSummary(
                ref="ref-1",
                primary_tag="backend",
                tags=["backend"],
                summary="Request cache boundary moved ahead of the mutable reminder.",
                summary_tokens=18,
                full_tokens=50,
                metadata=SegmentMetadata(
                    code_refs=[
                        {"file": "virtual_context/proxy/formats.py", "line": 1312, "symbol": "inject_context"},
                        {"file": "virtual_context/core/provider_adapters.py", "symbol": "AnthropicAdapter"},
                    ],
                ),
                created_at=now,
                start_timestamp=now,
                end_timestamp=now,
            ),
        ],
    )

    assert "[refs:" not in section
    assert "virtual_context/proxy/formats.py" not in section
    assert "virtual_context/core/provider_adapters.py" not in section


def test_trim_conversation(assembler):
    messages = [
        Message(role="user", content="x" * 400),
        Message(role="assistant", content="y" * 400),
        Message(role="user", content="z" * 400),
    ]
    trimmed = assembler._trim_conversation(messages, budget=250)
    assert len(trimmed) < len(messages)


def test_assemble_empty_retrieval(assembler):
    result = assembler.assemble(
        core_context="core",
        retrieval_result=RetrievalResult(),
        conversation_history=[Message(role="user", content="hello")],
        token_budget=10000,
    )
    assert result.tag_sections == {}
    assert len(result.conversation_history) == 1


def test_prepend_text(assembler, retrieval_result):
    result = assembler.assemble(
        core_context="# Core\nIdentity file",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
    )
    assert "Core" in result.prepend_text
    assert "virtual-context" in result.prepend_text


def test_tag_priority_from_rules():
    """Tags with higher priority rules should appear first."""
    rules = [
        TagPromptRule(match="architecture*", priority=10),
        TagPromptRule(match="debug*", priority=7),
        TagPromptRule(match="*", priority=5),
    ]
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=5000),
        tag_rules=rules,
    )
    assert assembler._tag_priority("architecture-decisions") == 10
    assert assembler._tag_priority("debugging") == 7
    assert assembler._tag_priority("random-tag") == 5


def test_budget_breakdown(assembler, retrieval_result):
    result = assembler.assemble(
        core_context="core context here",
        retrieval_result=retrieval_result,
        conversation_history=[Message(role="user", content="hello")],
        token_budget=10000,
    )
    assert "core" in result.budget_breakdown
    assert "tags" in result.budget_breakdown
    assert "conversation" in result.budget_breakdown


def test_context_hint_injected(assembler, retrieval_result):
    """Context hint appears between core context and tag sections."""
    hint = "<context-topics>\n- recipes (5 turns): recipe app...\n</context-topics>"
    result = assembler.assemble(
        core_context="# Core\nIdentity",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
        context_hint=hint,
    )
    assert "context-topics" in result.prepend_text
    # Hint appears after core, before tag sections
    core_pos = result.prepend_text.index("Core")
    hint_pos = result.prepend_text.index("context-topics")
    tag_pos = result.prepend_text.index("virtual-context")
    assert core_pos < hint_pos < tag_pos


def test_context_hint_empty(assembler, retrieval_result):
    """No hint block when context_hint is empty."""
    result = assembler.assemble(
        core_context="core",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
        context_hint="",
    )
    assert "context-topics" not in result.prepend_text


def test_context_hint_in_budget(assembler, retrieval_result):
    """Hint tokens counted in budget breakdown."""
    hint = "<context-topics>\nSome topics here\n</context-topics>"
    result = assembler.assemble(
        core_context="",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
        context_hint=hint,
    )
    assert result.budget_breakdown["context_hint"] > 0


def test_budget_cap_truncates_least_relevant_tags():
    """When assembled prepend_text exceeds token_budget, drop least-relevant tags."""
    now = datetime.now(timezone.utc)
    assembler = ContextAssembler(
        config=AssemblerConfig(
            core_context_max_tokens=1000,
            tag_context_max_tokens=5000,
        )
    )
    summaries = []
    for i, tag in enumerate(["high-priority", "medium-priority", "low-priority"]):
        summaries.append(StoredSummary(
            ref=f"ref-{i}",
            primary_tag=tag,
            tags=[tag],
            summary="x" * 400,  # ~100 tokens each
            summary_tokens=100,
            full_tokens=400,
            metadata=SegmentMetadata(),
            created_at=now,
            start_timestamp=now,
            end_timestamp=now,
        ))
    rr = RetrievalResult(
        tags_matched=["high-priority", "medium-priority", "low-priority"],
        summaries=summaries,
        total_tokens=300,
    )
    result = assembler.assemble(
        core_context="core",
        retrieval_result=rr,
        conversation_history=[],
        token_budget=250,
    )
    prepend_tokens = assembler.token_counter(result.prepend_text)
    assert prepend_tokens <= 250


def test_budget_cap_logs_error(caplog):
    """Exceeding token_budget logs an ERROR with config guidance."""
    import logging
    now = datetime.now(timezone.utc)
    assembler = ContextAssembler(
        config=AssemblerConfig(
            core_context_max_tokens=1000,
            tag_context_max_tokens=5000,
        )
    )
    summaries = [StoredSummary(
        ref="ref-0",
        primary_tag="big-tag",
        tags=["big-tag"],
        summary="x" * 400,
        summary_tokens=100,
        full_tokens=400,
        metadata=SegmentMetadata(),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    )]
    rr = RetrievalResult(
        tags_matched=["big-tag"],
        summaries=summaries,
        total_tokens=100,
    )
    with caplog.at_level(logging.ERROR):
        assembler.assemble(
            core_context="core",
            retrieval_result=rr,
            conversation_history=[],
            token_budget=5,
        )
    assert any("exceeds token_budget" in r.message for r in caplog.records)
    assert any("tag_context_max_tokens" in r.message for r in caplog.records)


def test_full_depth_does_not_bypass_quarantined_summary(assembler):
    """A rejected source scope cannot reappear through the raw full text."""
    segment = StoredSegment(
        ref="private-ref",
        primary_tag="health",
        tags=["health"],
        summary="BigTex discussed a private treatment.",
        full_text="BigTex: private treatment details",
    )

    section = assembler._format_full_section(
        "health",
        [segment],
        rendered_summary_by_object={id(segment): SUMMARY_ATTRIBUTION_QUARANTINE},
    )

    assert SUMMARY_ATTRIBUTION_QUARANTINE in section
    assert "private treatment details" not in section


def test_full_depth_never_trusts_stored_full_text_after_scope_admission(assembler):
    """Source-id proof does not prove a legacy full_text blob's bytes."""
    segment = StoredSegment(
        ref="admitted-ref",
        primary_tag="health",
        tags=["health"],
        summary="BigTex discussed treatment timing.",
        full_text="BigTex: treatment timing was discussed",
    )

    section = assembler._format_full_section(
        "health",
        [segment],
        rendered_summary_by_object={id(segment): "proved historical speaker: BigTex"},
    )

    assert "BigTex: treatment timing was discussed" not in section
    assert "proved historical speaker: BigTex" not in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE in section


class _LayerStore:
    def __init__(self, rows, tag_summary=None, segments=()):
        self.rows = rows
        self.tag_summary = tag_summary
        self.segments = {
            segment.ref: segment for segment in segments
        }

    def get_canonical_turn_rows_by_id(self, keys, *, speaker_context):
        wanted = set(keys)
        return {
            (row.conversation_id, row.canonical_turn_id): row
            for row in self.rows
            if (row.conversation_id, row.canonical_turn_id) in wanted
        }

    def get_tag_summary(self, tag, conversation_id=""):
        if (
            self.tag_summary is not None
            and self.tag_summary.tag == tag
            and conversation_id == "owner"
        ):
            return self.tag_summary
        return None

    def get_segment(self, ref, *, conversation_id=None):
        if conversation_id != "owner":
            return None
        return self.segments.get(ref)


def _layer_row(
    canonical_id: str,
    user_content: str,
    *,
    assistant_content: str = "",
):
    return SimpleNamespace(
        conversation_id="owner",
        canonical_turn_id=canonical_id,
        user_content=user_content,
        assistant_content=assistant_content,
        reply_target_body="",
        sender_actor_id="actor-a",
        sender="BigTex",
        session_date="2026-08-18",
        audience_conversation_id="guild",
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        origin_channel_id="channel",
        sort_key=1.0,
    )


def _layer_context():
    return SpeakerRetrievalContext(
        tenant_id="tenant",
        owner_conversation_id="owner",
        audience_conversation_id="guild",
        audience_channel_id="channel",
        requester_actor_id="actor-a",
    )


def _segment_structured(row, claim):
    records = [{
        "canonical_turn_id": row.canonical_turn_id,
        "source_role": "requester",
        "actor_id": row.sender_actor_id,
        "speaker_label": row.sender,
        "content": row.user_content.strip(),
        "session_date": "2026-08-18",
        "audience_conversation_id": row.audience_conversation_id,
        "origin_channel_id": row.origin_channel_id,
        "audience_attribution_version": row.audience_attribution_version,
    }]
    return StructuredSummary(
        schema_version=1,
        claims=(claim,),
        source_digest=structured_source_digest(records),
        generation_model="test-model",
    )


def _layer_claim(canonical_id: str, evidence: str) -> SummaryClaim:
    return SummaryClaim(
        text="generated prose is not rendered",
        claim_type="personal",
        temporal_status="ceased",
        modality="asserted",
        event_time="",
        sources=(SummarySource(
            canonical_turn_id=canonical_id,
            source_role="requester",
            speaker_label="BigTex",
            evidence_excerpt=evidence,
            session_date="2026-08-18",
            source_provenance_digest=structured_source_provenance_digest({
                "canonical_turn_id": canonical_id,
                "source_role": "requester",
                "actor_id": "actor-a",
                "speaker_label": "BigTex",
                "content": evidence,
                "session_date": "2026-08-18",
                "audience_conversation_id": "guild",
                "origin_channel_id": "channel",
                "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
            }),
        ),),
    )


def test_summary_segments_and_full_are_distinct_model_presentations():
    evidence = "I stopped tesamorelin because of edema."
    assistant = "We can discuss alternatives."
    row = _layer_row("ct-1", evidence, assistant_content=assistant)
    claim = _layer_claim("ct-1", evidence)
    metadata = SegmentMetadata(
        canonical_turn_ids=["ct-1"],
        source_mapping_complete=True,
        session_date="2026-08-18",
        structured_summary=_segment_structured(row, claim),
    )
    segment = StoredSegment(
        ref="seg-private-handle",
        primary_tag="health",
        tags=["health"],
        summary="unsafe free-form summary",
        full_text="unsafe stored full text",
        metadata=metadata,
    )
    store = _LayerStore([row])
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=store,
        conversation_id="owner",
    )

    summary = format_tag_section(
        "health",
        [segment],
        store=store,
        conversation_id="owner",
        speaker_context=_layer_context(),
        depth="summary",
    )
    segments = assembler._format_segments_section(
        "health", [segment], speaker_context=_layer_context(),
    )
    full = assembler._format_full_section(
        "health", [segment], speaker_context=_layer_context(),
    )

    assert '"depth":"summary"' in summary
    assert '"event_time"' not in summary
    assert '"depth":"segments"' in segments
    assert '"event_time":""' in segments
    assert '"sources":[' in segments
    assert "<canonical-source-transcript>" in full
    assert '"role":"historical_human"' in full
    assert '"role":"historical_assistant"' in full
    assert evidence in full
    assert assistant in full
    assert "unsafe stored full text" not in full
    assert "structured_summary_v1" not in full
    assert "actor-a" not in summary + segments + full
    assert "ct-1" not in summary + segments + full


def test_compressed_segment_ignores_unproved_assistant_only_legacy_row():
    requester_evidence = "I stopped tesamorelin because of edema."
    assistant_text = "You are currently taking tesamorelin."
    requester_row = _layer_row("ct-requester", requester_evidence)
    assistant_row = _layer_row(
        "ct-assistant", "", assistant_content=assistant_text,
    )
    assistant_row.sender_actor_id = ""
    assistant_row.sender = "Assistant"
    assistant_row.audience_conversation_id = ""
    assistant_row.origin_channel_id = ""
    assistant_row.audience_attribution_version = 0
    claim = _layer_claim("ct-requester", requester_evidence)
    segment = StoredSegment(
        ref="seg-legacy-assistant",
        primary_tag="health",
        tags=["health"],
        summary="unsafe free-form summary",
        full_text="unsafe stored full text",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-requester", "ct-assistant"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(requester_row, claim),
        ),
    )
    store = _LayerStore([requester_row, assistant_row])
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=store,
        conversation_id="owner",
    )

    summary = format_tag_section(
        "health",
        [segment],
        store=store,
        conversation_id="owner",
        speaker_context=_layer_context(),
        depth="summary",
    )
    segments = assembler._format_segments_section(
        "health", [segment], speaker_context=_layer_context(),
    )
    full = assembler._format_full_section(
        "health", [segment], speaker_context=_layer_context(),
    )

    for compressed in (summary, segments):
        assert '"source":"structured_summary_v1"' in compressed
        assert requester_evidence in compressed
        assert assistant_text not in compressed
        assert SUMMARY_ATTRIBUTION_QUARANTINE not in compressed
    assert SUMMARY_ATTRIBUTION_QUARANTINE in full
    assert assistant_text not in full
    assert requester_evidence not in full


def test_assemble_prefers_real_structured_tag_summary_at_summary_depth():
    evidence = "I stopped tesamorelin because of edema."
    segment_row = _layer_row("ct-1", evidence)
    segment_claim = _layer_claim("ct-1", evidence)
    segment_summary = StoredSummary(
        ref="seg-1",
        primary_tag="health",
        tags=["health"],
        summary="legacy segment synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-1"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(segment_row, segment_claim),
        ),
        start_timestamp=datetime.now(timezone.utc),
    )
    tag_summary = TagSummary(
        tag="health",
        summary="legacy tag synopsis",
        source_segment_refs=["seg-1"],
        source_canonical_turn_ids=["ct-1"],
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(segment_claim,),
            source_digest=structured_tag_claim_digest(
                (segment_claim,), ("ct-1",),
            ),
            generation_model="test-model",
        ),
    )
    store = _LayerStore([segment_row], tag_summary=tag_summary)
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=store,
        conversation_id="owner",
    )

    result = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[segment_summary],
        ),
        conversation_history=[],
        token_budget=10_000,
        working_set={
            "health": WorkingSetEntry(
                tag="health", depth=DepthLevel.SUMMARY,
            ),
        },
        speaker_context=_layer_context(),
    )

    section = result.tag_sections["health"]
    assert evidence in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE not in section
    assert "legacy tag synopsis" not in section
    assert "legacy segment synopsis" not in section

    tag_summary.structured_summary = StructuredSummary(
        schema_version=1,
        claims=(segment_claim,),
        source_digest="f" * 64,
        generation_model="test-model",
    )
    fallback_result = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[segment_summary],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    )
    fallback_section = fallback_result.tag_sections["health"]
    assert SUMMARY_ATTRIBUTION_QUARANTINE in fallback_section
    assert evidence not in fallback_section


def test_valid_stale_tag_renders_complete_union_with_newer_stop_segment():
    older_active = "I am currently taking tesamorelin."
    newer_stop = "I stopped tesamorelin yesterday."
    older_time = datetime(2026, 8, 17, tzinfo=timezone.utc)
    stop_time = datetime(2026, 8, 18, tzinfo=timezone.utc)
    older_row = _layer_row("ct-old", older_active)
    stop_row = _layer_row("ct-stop", newer_stop)
    older_claim = _layer_claim("ct-old", older_active)
    stop_claim = _layer_claim("ct-stop", newer_stop)
    older_segment = StoredSummary(
        ref="seg-old",
        primary_tag="health",
        tags=["health"],
        summary="unsafe older segment synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-old"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(older_row, older_claim),
        ),
        start_timestamp=older_time,
    )
    stop_segment = StoredSummary(
        ref="seg-stop",
        primary_tag="health",
        tags=["health"],
        summary="unsafe stop segment synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-stop"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(stop_row, stop_claim),
        ),
        start_timestamp=stop_time,
    )
    tag_summary = TagSummary(
        tag="health",
        summary="unsafe active-only tag synopsis",
        source_segment_refs=["seg-old"],
        source_canonical_turn_ids=["ct-old"],
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(older_claim,),
            source_digest=structured_tag_claim_digest(
                (older_claim,), ("ct-old",),
            ),
            generation_model="test-model",
        ),
    )
    store = _LayerStore(
        [older_row, stop_row],
        tag_summary=tag_summary,
        segments=[older_segment, stop_segment],
    )
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=store,
        conversation_id="owner",
    )

    result = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[stop_segment],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    )

    section = result.tag_sections["health"]
    assert 'segments="2"' in section
    assert older_active in section
    assert newer_stop in section
    assert section.index(newer_stop) < section.index(older_active)
    assert "unsafe active-only tag synopsis" not in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE not in section

    stop_segment.metadata.source_mapping_complete = False
    unproved_correction = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[stop_segment],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    ).tag_sections["health"]

    assert SUMMARY_ATTRIBUTION_QUARANTINE in unproved_correction
    assert older_active not in unproved_correction
    assert newer_stop not in unproved_correction

    stop_segment.metadata.source_mapping_complete = True
    stop_segment.metadata.structured_summary = StructuredSummary(
        schema_version=1,
        claims=(stop_claim,),
        source_digest="f" * 64,
        generation_model="test-model",
    )
    quarantined = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[stop_segment],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    ).tag_sections["health"]

    assert SUMMARY_ATTRIBUTION_QUARANTINE in quarantined
    assert older_active not in quarantined
    assert newer_stop not in quarantined


def test_tag_claim_is_rejected_after_same_label_actor_mutation():
    evidence = "I stopped tesamorelin because of edema."
    claim = _layer_claim("ct-1", evidence)
    tag_summary = TagSummary(
        tag="health",
        summary="unsafe tag synopsis",
        source_canonical_turn_ids=["ct-1"],
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(claim,),
            source_digest=structured_tag_claim_digest((claim,), ("ct-1",)),
            generation_model="test-model",
        ),
    )
    mutated_row = _layer_row("ct-1", evidence)
    mutated_row.sender_actor_id = "actor-b"

    section = format_tag_section(
        "health",
        [tag_summary],
        store=_LayerStore([mutated_row]),
        conversation_id="owner",
        speaker_context=_layer_context(),
    )

    assert '"source":"structured_summary_v1"' not in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE in section
    assert evidence not in section
    assert "unsafe tag synopsis" not in section


def test_tag_claim_validation_is_atomic_before_segment_fallback():
    newer_stop = "I stopped tesamorelin yesterday."
    older_active = "I am currently taking tesamorelin."
    fallback_stop = "I do not take tesamorelin."
    newer_row = _layer_row("ct-new", newer_stop)
    older_row = _layer_row("ct-old", older_active)
    fallback_row = _layer_row("ct-fallback", fallback_stop)
    newer_claim = _layer_claim("ct-new", newer_stop)
    older_claim = _layer_claim("ct-old", older_active)
    fallback_claim = _layer_claim("ct-fallback", fallback_stop)
    tag_summary = TagSummary(
        tag="health",
        summary="unsafe tag synopsis",
        source_canonical_turn_ids=["ct-new", "ct-old"],
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(newer_claim, older_claim),
            source_digest=structured_tag_claim_digest(
                (newer_claim, older_claim), ("ct-new", "ct-old"),
            ),
            generation_model="test-model",
        ),
    )
    segment_summary = StoredSummary(
        ref="seg-fallback",
        primary_tag="health",
        tags=["health"],
        summary="unsafe segment synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-fallback"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(
                fallback_row, fallback_claim,
            ),
        ),
        start_timestamp=datetime.now(timezone.utc),
    )

    # Only the newer correction changes after the tag envelope was written.
    # The older active claim still validates in isolation, but must never be
    # salvaged as the complete layer-two presentation.
    newer_row.user_content = "I asked about tesamorelin dosage."
    store = _LayerStore(
        [newer_row, older_row, fallback_row], tag_summary=tag_summary,
    )
    tag_only = format_tag_section(
        "health",
        [tag_summary],
        store=store,
        conversation_id="owner",
        speaker_context=_layer_context(),
    )

    assert '"source":"structured_summary_v1"' not in tag_only
    assert SUMMARY_ATTRIBUTION_QUARANTINE in tag_only
    assert older_active not in tag_only

    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=store,
        conversation_id="owner",
    )
    result = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[segment_summary],
        ),
        conversation_history=[],
        token_budget=10_000,
        working_set={
            "health": WorkingSetEntry(
                tag="health", depth=DepthLevel.SUMMARY,
            ),
        },
        speaker_context=_layer_context(),
    )

    section = result.tag_sections["health"]
    assert SUMMARY_ATTRIBUTION_QUARANTINE in section
    assert fallback_stop not in section
    assert older_active not in section
    assert newer_stop not in section
    assert "unsafe tag synopsis" not in section


def test_invalid_tag_uses_complete_declared_fallback_not_partial_retrieval():
    older_active = "I am currently taking tesamorelin."
    newer_stop = "I stopped tesamorelin yesterday."
    older_row = _layer_row("ct-old", older_active)
    stop_row = _layer_row("ct-stop", newer_stop)
    older_claim = _layer_claim("ct-old", older_active)
    stop_claim = _layer_claim("ct-stop", newer_stop)
    older_segment = StoredSummary(
        ref="seg-old",
        primary_tag="health",
        tags=["health"],
        summary="unsafe older synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-old"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(older_row, older_claim),
        ),
        start_timestamp=datetime.now(timezone.utc),
    )
    stop_segment = StoredSummary(
        ref="seg-stop",
        primary_tag="health",
        tags=["health"],
        summary="unsafe stop synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-stop"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(stop_row, stop_claim),
        ),
        start_timestamp=datetime.now(timezone.utc),
    )
    source_ids = ("ct-old", "ct-stop")
    tag_summary = TagSummary(
        tag="health",
        summary="unsafe active-only tag synopsis",
        source_segment_refs=["seg-old", "seg-stop"],
        source_canonical_turn_ids=list(source_ids),
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(older_claim,),
            source_digest=structured_tag_claim_digest(
                (older_claim,), source_ids,
            ),
            generation_model="test-model",
        ),
    )
    store = _LayerStore(
        [older_row, stop_row],
        tag_summary=tag_summary,
        segments=[older_segment, stop_segment],
    )
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=store,
        conversation_id="owner",
    )

    result = assembler.assemble(
        core_context="",
        # Retrieval found only the stale older segment. The invalid tag's
        # complete digest-bound refs must replace this partial subset.
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[older_segment],
        ),
        conversation_history=[],
        token_budget=10_000,
        working_set={
            "health": WorkingSetEntry(
                tag="health", depth=DepthLevel.SUMMARY,
            ),
        },
        speaker_context=_layer_context(),
    )

    section = result.tag_sections["health"]
    assert 'segments="2"' in section
    assert older_active in section
    assert newer_stop in section
    assert "unsafe active-only tag synopsis" not in section

    stop_segment.metadata.structured_summary = StructuredSummary(
        schema_version=1,
        claims=(stop_claim,),
        source_digest="f" * 64,
        generation_model="test-model",
    )
    invalid_fallback = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[older_segment],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    ).tag_sections["health"]

    assert SUMMARY_ATTRIBUTION_QUARANTINE in invalid_fallback
    assert older_active not in invalid_fallback
    assert newer_stop not in invalid_fallback


def test_empty_v1_tag_uses_complete_declared_segment_fallback():
    older_active = "I am currently taking tesamorelin."
    newer_stop = "I stopped tesamorelin yesterday."
    older_row = _layer_row("ct-old", older_active)
    stop_row = _layer_row("ct-stop", newer_stop)
    older_claim = _layer_claim("ct-old", older_active)
    stop_claim = _layer_claim("ct-stop", newer_stop)
    older_segment = StoredSummary(
        ref="seg-old",
        primary_tag="health",
        tags=["health"],
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-old"],
            source_mapping_complete=True,
            structured_summary=_segment_structured(older_row, older_claim),
        ),
        start_timestamp=datetime.now(timezone.utc),
    )
    stop_segment = StoredSummary(
        ref="seg-stop",
        primary_tag="health",
        tags=["health"],
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-stop"],
            source_mapping_complete=True,
            structured_summary=_segment_structured(stop_row, stop_claim),
        ),
        start_timestamp=datetime.now(timezone.utc),
    )
    source_ids = ("ct-old", "ct-stop")
    empty_tag = TagSummary(
        tag="health",
        source_segment_refs=["seg-old", "seg-stop"],
        source_canonical_turn_ids=list(source_ids),
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(),
            source_digest=structured_tag_claim_digest((), source_ids),
            generation_model="test-model",
        ),
    )
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=_LayerStore(
            [older_row, stop_row],
            tag_summary=empty_tag,
            segments=[older_segment, stop_segment],
        ),
        conversation_id="owner",
    )

    section = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[older_segment],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    ).tag_sections["health"]

    assert 'segments="2"' in section
    assert older_active in section
    assert newer_stop in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE not in section

    # Empty selections still authenticate the complete parent coordinates.
    # Deleting the newer stop from only the persisted refs/IDs must invalidate
    # the original digest instead of degrading to an old-active-only fallback.
    empty_tag.source_segment_refs = ["seg-old"]
    empty_tag.source_canonical_turn_ids = ["ct-old"]
    mutated_section = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[older_segment],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    ).tag_sections["health"]

    assert SUMMARY_ATTRIBUTION_QUARANTINE in mutated_section
    assert older_active not in mutated_section
    assert newer_stop not in mutated_section


def test_active_only_tag_is_rejected_when_full_sources_include_newer_stop():
    older_active = "I am currently taking tesamorelin."
    newer_stop = "I stopped tesamorelin yesterday."
    older_row = _layer_row("ct-old", older_active)
    newer_row = _layer_row("ct-new", newer_stop)
    older_claim = _layer_claim("ct-old", older_active)
    tag_summary = TagSummary(
        tag="health",
        summary="unsafe active-only tag synopsis",
        source_canonical_turn_ids=["ct-old", "ct-new"],
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(older_claim,),
            source_digest=structured_tag_claim_digest(
                (older_claim,), ("ct-old", "ct-new"),
            ),
            generation_model="test-model",
        ),
    )

    section = format_tag_section(
        "health",
        [tag_summary],
        store=_LayerStore([older_row, newer_row]),
        conversation_id="owner",
        speaker_context=_layer_context(),
    )

    assert '"source":"structured_summary_v1"' not in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE in section
    assert older_active not in section
    assert "unsafe active-only tag synopsis" not in section


def test_tag_rejects_distinct_claims_that_duplicate_one_canonical_source():
    evidence = "I discussed tesamorelin dosing."
    row = _layer_row("ct-1", evidence)
    first = _layer_claim("ct-1", evidence)
    duplicate_source = SummaryClaim(
        text="a different generated classification",
        claim_type=first.claim_type,
        temporal_status=first.temporal_status,
        modality=first.modality,
        event_time=first.event_time,
        sources=first.sources,
    )
    claims = (first, duplicate_source)
    tag_summary = TagSummary(
        tag="health",
        summary="unsafe duplicate tag synopsis",
        source_canonical_turn_ids=["ct-1"],
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=claims,
            source_digest=structured_tag_claim_digest(claims, ("ct-1",)),
            generation_model="test-model",
        ),
    )

    section = format_tag_section(
        "health",
        [tag_summary],
        store=_LayerStore([row]),
        conversation_id="owner",
        speaker_context=_layer_context(),
    )

    assert '"source":"structured_summary_v1"' not in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE in section
    assert evidence not in section


@pytest.mark.parametrize("mutated_ids", [
    ["ct-1", "ct-1"],
    [" ct-1"],
    ["ct-1 "],
])
def test_tag_and_segment_reject_malformed_persisted_canonical_ids(
    mutated_ids,
):
    evidence = "I discussed tesamorelin dosing."
    row = _layer_row("ct-1", evidence)
    claim = _layer_claim("ct-1", evidence)
    tag_summary = TagSummary(
        tag="health",
        summary="unsafe duplicate tag synopsis",
        source_canonical_turn_ids=mutated_ids,
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=(claim,),
            source_digest=structured_tag_claim_digest((claim,), ("ct-1",)),
            generation_model="test-model",
        ),
    )
    segment = StoredSegment(
        ref="seg-duplicate",
        primary_tag="health",
        tags=["health"],
        summary="unsafe duplicate segment synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=mutated_ids,
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(row, claim),
        ),
    )
    store = _LayerStore([row])

    for item in (tag_summary, segment):
        section = format_tag_section(
            "health",
            [item],
            store=store,
            conversation_id="owner",
            speaker_context=_layer_context(),
        )
        assert '"source":"structured_summary_v1"' not in section
        assert SUMMARY_ATTRIBUTION_QUARANTINE in section
        assert evidence not in section


def test_assemble_keeps_segment_v1_when_tag_summary_is_legacy():
    segment_evidence = "I stopped tesamorelin because of edema."
    row = _layer_row("ct-1", segment_evidence)
    claim = _layer_claim("ct-1", segment_evidence)
    segment_summary = StoredSummary(
        ref="seg-1",
        primary_tag="health",
        tags=["health"],
        summary="legacy segment synopsis",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-1"],
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=_segment_structured(row, claim),
        ),
        start_timestamp=datetime.now(timezone.utc),
    )
    legacy_tag_summary = TagSummary(
        tag="health",
        summary="BigTex currently uses tesamorelin.",
        source_canonical_turn_ids=["ct-1"],
    )
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=_LayerStore([row], tag_summary=legacy_tag_summary),
        conversation_id="owner",
    )

    result = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            tags_matched=["health"], summaries=[segment_summary],
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    )

    section = result.tag_sections["health"]
    assert '"source":"structured_summary_v1"' in section
    assert segment_evidence in section
    assert "currently uses tesamorelin" not in section


def test_summary_floor_v0_hydrates_tag_source_segments_for_canonical_fallback():
    evidence = "I stopped tesamorelin because of edema."
    row = _layer_row("ct-1", evidence)
    source_segment = StoredSegment(
        ref="seg-1",
        primary_tag="health",
        tags=["health"],
        summary="unsafe source segment synopsis",
        full_text="unsafe stored full text",
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-1"],
            source_mapping_complete=True,
            session_date="2026-08-18",
        ),
    )
    legacy_tag_summary = TagSummary(
        tag="health",
        summary="BigTex currently uses tesamorelin.",
        source_segment_refs=["seg-1"],
        source_canonical_turn_ids=["ct-1"],
    )
    synthetic_floor_summary = StoredSummary(
        ref="tag-summary-health",
        primary_tag="health",
        tags=["health"],
        summary="synthetic summary-floor prose",
        metadata=SegmentMetadata(),
        start_timestamp=datetime.now(timezone.utc),
    )
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=10_000),
        store=_LayerStore(
            [row],
            tag_summary=legacy_tag_summary,
            segments=[source_segment],
        ),
        conversation_id="owner",
    )

    result = assembler.assemble(
        core_context="",
        retrieval_result=RetrievalResult(
            summaries=[synthetic_floor_summary],
            retrieval_metadata={"summary_floor": True},
        ),
        conversation_history=[],
        token_budget=10_000,
        speaker_context=_layer_context(),
    )

    section = result.tag_sections["health"]
    assert "<historical-source-transcript>" in section
    assert evidence in section
    assert SUMMARY_ATTRIBUTION_QUARANTINE not in section
    assert "synthetic summary-floor prose" not in section
    assert "currently uses tesamorelin" not in section
