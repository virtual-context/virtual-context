"""Canonical evidence fixtures for paging/assembly budget contracts."""

import uuid

from virtual_context.core.structured_summary import structured_source_digest, structured_source_provenance_digest
from virtual_context.types import (
    RetrievalResult, SegmentMetadata, SpeakerRetrievalContext, StoredSegment,
    StructuredSummary, SummaryClaim, SummarySource,
)


def seed_paging_sources(engine, tag, n=1, tokens_per=100, *, context=None):
    owner = engine.config.conversation_id
    context = context or getattr(engine, "_test_speaker_context", None) or SpeakerRetrievalContext(
        tenant_id="paging-test", owner_conversation_id=owner, audience_conversation_id=owner,
        audience_channel_id="paging-channel", request_origin_channel_id="paging-channel",
        requester_actor_id="actor:discord:paging",
    )
    actor = context.requester_actor_id or "actor:discord:paging"
    result = []
    for index in range(n):
        canonical_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{owner}/{tag}/{index}"))
        evidence = f"I discussed {tag} evidence {index}."
        order = getattr(engine, "_test_source_order", 0)
        engine._test_source_order = order + 1
        engine._store.save_canonical_turn(
            owner, order, evidence, "expanded assistant detail " * tokens_per,
            canonical_turn_id=canonical_id, turn_group_number=order, sort_key=float(order),
            sender="Alice", sender_actor_id=actor,
            audience_conversation_id=context.audience_conversation_id, audience_attribution_version=1,
            origin_channel_id=context.audience_channel_id, primary_tag=tag, tags=[tag],
        )
        record = dict(canonical_turn_id=canonical_id, source_role="requester", actor_id=actor,
                      speaker_label="Alice", content=evidence, session_date="",
                      audience_conversation_id=context.audience_conversation_id,
                      origin_channel_id=context.audience_channel_id, audience_attribution_version=1)
        claim = SummaryClaim(text=evidence, claim_type="conversation", modality="asserted", sources=(SummarySource(
            canonical_turn_id=canonical_id, source_role="requester", speaker_label="Alice",
            evidence_excerpt=evidence, source_provenance_digest=structured_source_provenance_digest(record),
        ),))
        segment = StoredSegment(
            ref=f"{tag}-seg-{index}", conversation_id=owner, primary_tag=tag, tags=[tag],
            summary="Unproved stored synopsis must not be rendered", summary_tokens=1,
            full_text="Unproved stored full text must not be rendered", full_tokens=1,
            metadata=SegmentMetadata(canonical_turn_ids=[canonical_id], source_mapping_complete=True,
                                     structured_summary=StructuredSummary(schema_version=1, claims=(claim,),
                                                                          source_digest=structured_source_digest([record]))),
        )
        engine._store.store_segment(segment)
        result.append(segment)
    engine._test_speaker_context = context
    # Bind a real source proof to one request snapshot. These tests exercise
    # paging directly rather than invoking an unrelated query/tagging turn.
    engine._retrieval._last_reassembly_snapshot = (RetrievalResult(), [], "", None, context, None)
    return result, context
