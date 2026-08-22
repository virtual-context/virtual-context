"""Focused model-boundary tests for structured summary retrieval surfaces."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from virtual_context.core.quote_search import search_summaries
from virtual_context.core.structured_summary import (
    structured_source_digest,
    structured_source_provenance_digest,
    structured_tag_claim_digest,
)
from virtual_context.core.summary_identity import (
    is_proved_summary_rendering,
    sanitize_summary_payload_for_model,
)
from virtual_context.core.temporal_resolver import TemporalResolver
from virtual_context.core.tool_loop import (
    _render_recall_all_payload,
    execute_vc_tool,
)
from virtual_context.proxy.formats import detect_format
from virtual_context.proxy.message_filter import fill_pass
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    AssembledContext,
    QuoteResult,
    RetrievalResult,
    SearchConfig,
    SegmentMetadata,
    SpeakerRetrievalContext,
    StoredSegment,
    StructuredSummary,
    SummaryClaim,
    SummarySource,
    TagSummary,
    VirtualContextConfig,
)


def _context() -> SpeakerRetrievalContext:
    return SpeakerRetrievalContext(
        tenant_id="tenant",
        owner_conversation_id="conversation",
        audience_conversation_id="guild",
        audience_channel_id="health",
    )


def _row(
    canonical_id: str,
    actor_id: str,
    label: str,
    content: str,
    *,
    audience: str = "guild",
    channel: str = "health",
    session_date: str = "2026-08-18",
) -> SimpleNamespace:
    return SimpleNamespace(
        conversation_id="conversation",
        canonical_turn_id=canonical_id,
        user_content=content,
        assistant_content="",
        sender_actor_id=actor_id,
        sender=label,
        session_date=session_date,
        audience_conversation_id=audience,
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
        origin_channel_id=channel,
        reply_target_body="",
        reply_subject_actor_id="",
        reply_subject_label="",
        sort_key=1.0,
    )


def _claim(
    *,
    canonical_id: str,
    label: str,
    evidence: str,
    text: str,
    temporal_status: str,
    claim_type: str = "personal",
    actor_id: str = "",
    audience: str = "guild",
    channel: str = "health",
    session_date: str = "2026-08-18",
) -> SummaryClaim:
    actor = actor_id or {
        "BigTex": "actor-bigtex",
        "Kuw9239": "actor-kuw",
        "PrivateName": "actor-private",
    }[label]
    return SummaryClaim(
        text=text,
        claim_type=claim_type,
        temporal_status=temporal_status,
        modality="asserted",
        sources=(SummarySource(
            canonical_turn_id=canonical_id,
            source_role="requester",
            speaker_label=label,
            evidence_excerpt=evidence,
            session_date=session_date,
            source_provenance_digest=structured_source_provenance_digest({
                "canonical_turn_id": canonical_id,
                "source_role": "requester",
                "actor_id": actor,
                "speaker_label": label,
                "content": evidence,
                "session_date": session_date,
                "audience_conversation_id": audience,
                "origin_channel_id": channel,
                "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
            }),
        ),),
    )


def _segment(
    ref: str,
    canonical_ids: list[str],
    claims: tuple[SummaryClaim, ...],
    *,
    free_summary: str,
) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        conversation_id="conversation",
        primary_tag="health",
        tags=["health"],
        # This prose is deliberately unsafe/incorrect. It may rank the row but
        # must never cross the model boundary.
        summary=free_summary,
        metadata=SegmentMetadata(
            canonical_turn_ids=canonical_ids,
            source_mapping_complete=True,
            session_date="2026-08-18",
            structured_summary=StructuredSummary(
                schema_version=1,
                claims=claims,
                source_digest="a" * 64,
                generation_model="test",
            ),
        ),
    )


def _stamp_segment_digest(
    segment: StoredSegment,
    rows: list[SimpleNamespace],
) -> StoredSegment:
    records = []
    for row in rows:
        if row.user_content:
            records.append({
                "canonical_turn_id": row.canonical_turn_id,
                "source_role": "requester",
                "actor_id": row.sender_actor_id,
                "speaker_label": row.sender,
                "content": row.user_content,
                "session_date": segment.metadata.session_date,
                "audience_conversation_id": row.audience_conversation_id,
                "origin_channel_id": row.origin_channel_id,
                "audience_attribution_version": row.audience_attribution_version,
            })
    current = segment.metadata.structured_summary
    segment.metadata.structured_summary = StructuredSummary(
        schema_version=current.schema_version,
        claims=current.claims,
        source_digest=structured_source_digest(records, namespace="segment"),
        generation_model=current.generation_model,
    )
    return segment


def _tag_source_digest(claims: tuple[SummaryClaim, ...]) -> str:
    source_ids = tuple(
        source.canonical_turn_id
        for claim in claims
        for source in claim.sources
    )
    return structured_tag_claim_digest(claims, source_ids)


def _store_for(
    segments: list[StoredSegment],
    rows: list[SimpleNamespace],
) -> MagicMock:
    store = MagicMock()
    by_ref = {segment.ref: segment for segment in segments}
    by_key = {
        (row.conversation_id, row.canonical_turn_id): row for row in rows
    }
    store.get_segment.side_effect = (
        lambda ref, conversation_id=None: by_ref.get(ref)
    )
    store.get_canonical_turn_rows_by_id.side_effect = (
        lambda keys, *, speaker_context: {
            key: by_key[key] for key in keys if key in by_key
        }
    )
    store.get_all_tag_summaries.return_value = []
    store.search_tool_outputs.return_value = []
    return store


def _structured_payload(rendered: str) -> dict:
    prefix = "<structured-summary>\n"
    suffix = "\n</structured-summary>"
    assert rendered.startswith(prefix)
    assert rendered.endswith(suffix)
    return json.loads(rendered[len(prefix):-len(suffix)])


def test_search_summaries_bigtex_false_active_claim_cannot_cross_boundary() -> None:
    evidence = "I stopped tesamorelin because it caused edema."
    rows = [_row("ct-bigtex", "actor-bigtex", "BigTex", evidence)]
    segment = _stamp_segment_digest(_segment(
        "seg-bigtex",
        ["ct-bigtex"],
        (
            _claim(
                canonical_id="ct-bigtex",
                label="BigTex",
                evidence=evidence,
                text="BigTex is actively taking tesamorelin.",
                temporal_status="active",
            ),
            _claim(
                canonical_id="ct-bigtex",
                label="BigTex",
                evidence=evidence,
                text="BigTex discontinued tesamorelin due to edema.",
                temporal_status="ceased",
            ),
        ),
        free_summary="The user currently takes tesamorelin.",
    ), rows)
    store = _store_for([
        segment,
    ], rows)
    store.search_full_text.return_value = [QuoteResult(
        text=segment.summary,
        tag="health",
        segment_ref=segment.ref,
        tags=["health"],
        match_type="fts",
        session_date="2026-08-18",
    )]
    semantic = MagicMock()
    semantic.semantic_search.return_value = []

    context = _context()
    result = search_summaries(
        store,
        semantic,
        "tesamorelin",
        conversation_id="conversation",
        speaker_context=context,
    )

    assert result["found"] is True
    rendered = result["results"][0]["excerpt"]
    assert is_proved_summary_rendering(rendered)
    payload = _structured_payload(rendered)
    assert len(payload["claims"]) == 1
    assert payload["claims"][0]["text"] == evidence
    assert payload["claims"][0]["temporal_status"] == ""
    serialized = json.dumps(result)
    assert "actively taking" not in serialized
    assert "currently takes" not in serialized
    assert "actor-bigtex" not in serialized
    assert "ct-bigtex" not in serialized
    assert sanitize_summary_payload_for_model(
        result,
        allow_proved_renderings=True,
        speaker_context=context,
    ) == result


def test_remember_when_engine_preserves_same_request_structured_rendering() -> None:
    from virtual_context.engine import VirtualContextEngine

    evidence = "I stopped tesamorelin because it caused edema."
    rows = [_row("ct-bigtex", "actor-bigtex", "BigTex", evidence)]
    segment = _stamp_segment_digest(_segment(
        "seg-bigtex",
        ["ct-bigtex"],
        (_claim(
            canonical_id="ct-bigtex",
            label="BigTex",
            evidence=evidence,
            text="BigTex stopped tesamorelin.",
            temporal_status="ceased",
        ),),
        free_summary="The user currently takes tesamorelin.",
    ), rows)
    store = _store_for([segment], rows)
    context = _context()
    resolver = TemporalResolver(
        store,
        MagicMock(),
        VirtualContextConfig(conversation_id="conversation"),
    )
    scoped, _actors = resolver._scope_segment_results_for_request(
        [{
            "excerpt": segment.summary,
            "topic": "health",
            "segment_ref": segment.ref,
            "session": "2026-08-18",
            "session_date_normalized": "2026-08-18",
            "match_type": "summary",
        }],
        speaker_context=context,
    )
    temporal = MagicMock()
    temporal.remember_when.return_value = {
        "found": True,
        "results": scoped,
        "facts_in_window": [],
    }

    result = VirtualContextEngine.remember_when(
        SimpleNamespace(_temporal=temporal),
        "tesamorelin",
        {"last_n_days": 30},
        speaker_context=context,
    )

    rendered = result["results"][0]["excerpt"]
    assert is_proved_summary_rendering(rendered)
    assert _structured_payload(rendered)["claims"][0]["text"] == evidence


def test_search_summaries_preserves_multi_speaker_claims_without_reassignment() -> None:
    bigtex_evidence = "I stopped the tesamorelin protocol."
    kuw_evidence = "I plan to start the CJC-1295 protocol next month."
    rows = [
        _row("ct-bigtex", "actor-bigtex", "BigTex", bigtex_evidence),
        _row("ct-kuw", "actor-kuw", "Kuw9239", kuw_evidence),
    ]
    segment = _stamp_segment_digest(_segment(
        "seg-multi",
        ["ct-bigtex", "ct-kuw"],
        (
            _claim(
                canonical_id="ct-bigtex",
                label="BigTex",
                evidence=bigtex_evidence,
                text="BigTex stopped the protocol.",
                temporal_status="ceased",
            ),
            _claim(
                canonical_id="ct-kuw",
                label="Kuw9239",
                evidence=kuw_evidence,
                text="Kuw9239 plans the other protocol.",
                temporal_status="planned",
            ),
        ),
        free_summary="The user changed protocols while another user planned one.",
    ), rows)
    store = _store_for([segment], rows)
    store.search_full_text.return_value = [QuoteResult(
        text=segment.summary,
        tag="health",
        segment_ref=segment.ref,
        tags=["health"],
        match_type="fts",
        session_date="2026-08-18",
    )]
    semantic = MagicMock()
    semantic.semantic_search.return_value = []

    result = search_summaries(
        store,
        semantic,
        "protocol",
        conversation_id="conversation",
        speaker_context=_context(),
    )

    claims = _structured_payload(result["results"][0]["excerpt"])["claims"]
    assert [claim["sources"][0]["display_name"] for claim in claims] == [
        "BigTex", "Kuw9239",
    ]
    assert [claim["temporal_status"] for claim in claims] == ["", ""]
    assert claims[0]["sources"][0]["evidence_excerpt"] == bigtex_evidence
    assert claims[1]["sources"][0]["evidence_excerpt"] == kuw_evidence


def test_temporal_surface_withholds_segment_with_any_cross_channel_source() -> None:
    public_evidence = "I stopped the tesamorelin protocol."
    private_evidence = "I am currently using a private peptide protocol."
    rows = [
        _row("ct-public", "actor-bigtex", "BigTex", public_evidence),
        _row(
            "ct-private", "actor-private", "PrivateName", private_evidence,
            audience="guild", channel="private-channel",
        ),
    ]
    segment = _stamp_segment_digest(_segment(
        "seg-scope",
        ["ct-public", "ct-private"],
        (
            _claim(
                canonical_id="ct-public",
                label="BigTex",
                evidence=public_evidence,
                text="BigTex stopped tesamorelin.",
                temporal_status="ceased",
            ),
            _claim(
                canonical_id="ct-private",
                label="PrivateName",
                evidence=private_evidence,
                text="PrivateName uses another protocol.",
                temporal_status="active",
                channel="private-channel",
            ),
        ),
        free_summary="The user currently uses both protocols.",
    ), rows)
    store = _store_for([segment], rows)
    config = VirtualContextConfig(conversation_id="conversation")
    resolver = TemporalResolver(store, MagicMock(), config)

    scoped, _actors = resolver._scope_segment_results_for_request(
        [{
            "excerpt": segment.summary,
            "topic": "health",
            "segment_ref": segment.ref,
            "session": "2026-08-18",
            "session_date_normalized": "2026-08-18",
            "match_type": "summary",
        }],
        speaker_context=_context(),
    )

    # A segment digest binds the complete source snapshot. A sibling lane
    # outside the request channel invalidates the whole segment rather than
    # silently presenting a partial chronology.
    assert scoped == []


def test_recall_all_attaches_validated_tag_summary_not_free_rollup_prose() -> None:
    evidence = "I stopped tesamorelin because it caused edema."
    claims = (_claim(
        canonical_id="ct-bigtex",
        label="BigTex",
        evidence=evidence,
        text="BigTex stopped tesamorelin.",
        temporal_status="ceased",
    ),)
    structured = StructuredSummary(
        schema_version=1,
        claims=claims,
        source_digest=_tag_source_digest(claims),
        generation_model="test",
    )
    tag_summary = TagSummary(
        tag="health",
        summary="The user currently takes tesamorelin.",
        description="Current peptide use",
        summary_tokens=20,
        source_segment_refs=["seg-bigtex"],
        source_turn_numbers=[7],
        source_canonical_turn_ids=["ct-bigtex"],
        structured_summary=structured,
    )
    store = _store_for([], [
        _row("ct-bigtex", "actor-bigtex", "BigTex", evidence),
    ])
    store.get_all_tag_summaries.return_value = [tag_summary]
    engine = SimpleNamespace(
        _store=store,
        config=SimpleNamespace(conversation_id="conversation"),
    )
    raw = {
        "found": True,
        "summaries": [{
            "tag": "health",
            "tokens": 20,
            "source_segment_refs": ["seg-bigtex"],
            "source_turn_numbers": [7],
        }],
    }

    rendered = _render_recall_all_payload(
        engine, raw, speaker_context=_context(),
    )

    entry = rendered["summaries"][0]
    assert is_proved_summary_rendering(entry["summary"])
    assert _structured_payload(entry["summary"])["claims"][0][
        "temporal_status"
    ] == ""
    serialized = json.dumps(rendered)
    assert "currently takes" not in serialized
    assert "Current peptide use" not in serialized
    assert "actor-bigtex" not in serialized
    assert "ct-bigtex" not in serialized

    # The real tool boundary must preserve the proved typed envelope through
    # its final recursive sanitizer, not collapse it back to quarantine.
    tool_engine = SimpleNamespace(
        _store=store,
        config=SimpleNamespace(
            conversation_id="conversation",
            search=SearchConfig(),
        ),
        recall_all=lambda: copy.deepcopy(raw),
    )
    tool_result = json.loads(execute_vc_tool(
        tool_engine,
        "vc_recall_all",
        {},
        speaker_context=_context(),
    ))
    assert is_proved_summary_rendering(
        tool_result["summaries"][0]["summary"],
    )
    assert "currently takes" not in json.dumps(tool_result)


def test_fill_pass_uses_validated_tag_summary_as_breadth_layer() -> None:
    evidence = "I stopped tesamorelin because it caused edema."
    claims = (_claim(
        canonical_id="ct-bigtex",
        label="BigTex",
        evidence=evidence,
        text="BigTex stopped tesamorelin.",
        temporal_status="ceased",
    ),)
    tag_summary = TagSummary(
        tag="health",
        summary="The user currently takes tesamorelin.",
        description="Current peptide use",
        summary_tokens=20,
        source_segment_refs=["seg-bigtex"],
        source_turn_numbers=[7],
        source_canonical_turn_ids=["ct-bigtex"],
        structured_summary=StructuredSummary(
            schema_version=1,
            claims=claims,
            source_digest=_tag_source_digest(claims),
            generation_model="test",
        ),
    )
    store = _store_for([], [
        _row("ct-bigtex", "actor-bigtex", "BigTex", evidence),
    ])
    store.get_all_tag_summaries.return_value = [tag_summary]
    body = {
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "current question"}],
        "model": "claude-opus-4-6",
    }
    assembled = AssembledContext(
        speaker_context=_context(),
        presented_segment_refs=set(),
        presented_tags=set(),
        tag_sections={},
        retrieval_result=RetrievalResult(),
    )

    result, summaries_added, _turns_added = fill_pass(
        body=body,
        fmt=detect_format(body),
        outbound_tokens=1_000,
        target_tokens=10_000,
        assembled=assembled,
        pre_filter_body=copy.deepcopy(body),
        store=store,
        conversation_id="conversation",
        summary_ratio=1.0,
    )

    assert summaries_added == 1
    serialized = json.dumps(result)
    assert "<structured-summary>" in serialized
    assert "temporal_status" in serialized
    assert "ceased" not in serialized
    assert "currently takes" not in serialized
    assert "Current peptide use" not in serialized
    assert "actor-bigtex" not in serialized
    assert "ct-bigtex" not in serialized
