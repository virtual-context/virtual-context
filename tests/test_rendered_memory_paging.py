"""Paging uses the same immutable, proof-validated rendering as assembly."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from virtual_context.core.assembler import ContextAssembler
from virtual_context.core.paging_manager import PagingManager
from virtual_context.token_counter import create_token_counter
from virtual_context.types import (
    AssemblerConfig, DepthLevel, RetrievalResult, SegmentMetadata,
    SpeakerRetrievalContext, StoredSegment, WorkingSetEntry,
)


class _Sources:
    def __init__(self):
        self.row = SimpleNamespace(
            conversation_id="owner", canonical_turn_id="ct1", user_content="I planned a trip. 你 🧭",
            assistant_content="I can help with that.", sender="Alice", sender_actor_id="actor:a",
            audience_conversation_id="guild", audience_attribution_version=1,
            origin_channel_id="channel", session_date="2026-09-05", sort_key=1.0,
        )
        self.segment = StoredSegment(
            ref="segment", conversation_id="owner", primary_tag="travel", tags=["travel"],
            full_text="forged stored text", full_tokens=1, summary_tokens=1,
            metadata=SegmentMetadata(canonical_turn_ids=["ct1"], source_mapping_complete=True),
        )

    def get_segments_by_tags(self, **kwargs):
        return [self.segment] if "travel" in kwargs["tags"] else []

    def get_tag_summary(self, tag, **kwargs):
        return None

    def get_canonical_turn_rows_by_id(self, keys, *, speaker_context):
        return {("owner", "ct1"): self.row}

    def get_recent_canonical_turns(self, *args, **kwargs):
        return [self.row]


@pytest.fixture
def setup():
    counter = create_token_counter("tiktoken")
    store = _Sources()
    context = SpeakerRetrievalContext(
        tenant_id="tenant", owner_conversation_id="owner", audience_conversation_id="guild",
        audience_channel_id="channel", requester_actor_id="actor:a",
    )
    assembler = ContextAssembler(AssemblerConfig(), counter, store=store, conversation_id="owner")
    paging = PagingManager(store, counter, tag_context_max_tokens=10000, conversation_id="owner")
    paging.set_memory_renderer(assembler.render_topic_memory)
    return store, context, assembler, paging, counter


@pytest.mark.parametrize("depth", [DepthLevel.SUMMARY, DepthLevel.SEGMENTS, DepthLevel.FULL])
def test_pages_charge_actual_admitted_rendering_and_keep_proof_metadata(setup, depth):
    store, context, assembler, paging, counter = setup
    assembled = assembler.assemble(
        "", RetrievalResult(summaries=[store.segment]), [], 10000,
        working_set={"travel": WorkingSetEntry(tag="travel", depth=depth)},
        full_segments={"travel": [store.segment]}, speaker_context=context,
    )
    memory, = assembled.rendered_memories
    assert memory.text == assembled.tag_sections["travel"]
    assert memory.measured_cost == counter(memory.text) > 1
    assert memory.presented_source_ids == ("ct1",)
    assert memory.sources[0].canonical_turn_id == "ct1"
    assert len(memory.sources[0].version) == 64
    assert memory.scope is context
    assert "forged stored text" not in memory.text
    assert "ct1" not in memory.text
    with pytest.raises(FrozenInstanceError):
        memory.measured_cost = 1
    result = paging.expand_topic("travel", depth.value, speaker_context=context)
    assert "error" not in result
    page = paging.rendered_memories["travel"]
    assert paging.working_set["travel"].tokens == counter(page.text) == page.measured_cost
    if depth == DepthLevel.FULL:
        assert page == memory


def test_page_rejects_wrapper_overflow_and_does_not_charge_stored_estimate(setup):
    _, context, assembler, paging, _ = setup
    memory = assembler.render_topic_memory("travel", DepthLevel.FULL, speaker_context=context)
    paging._tag_context_max_tokens = memory.measured_cost - 1
    result = paging.expand_topic("travel", "full", speaker_context=context)
    assert result["error"] == "insufficient budget"
    assert result["needed"] == memory.measured_cost
    assert paging.working_set == {} and paging.rendered_memories == {}


def test_source_correction_revalidates_proof_and_changes_version(setup):
    store, context, _, paging, _ = setup
    paging.expand_topic("travel", "full", speaker_context=context)
    old = paging.rendered_memories["travel"]
    store.row.user_content = "I cancelled the trip."
    paging.expand_topic("travel", "full", speaker_context=context)
    new = paging.rendered_memories["travel"]
    assert old.sources != new.sources
    assert "cancelled" in new.text and "planned" not in new.text
    store.row.sender_actor_id = ""
    assert paging.calculate_depth_tokens("travel", DepthLevel.FULL, speaker_context=context) == 0
    result = paging.expand_topic("travel", "full", speaker_context=context)
    assert "error" in result
    assert paging.rendered_memories["travel"] is new  # failed transaction is unchanged


def test_missing_or_wrong_scope_cannot_admit_stored_prose(setup):
    _, _, _, paging, _ = setup
    assert "error" in paging.expand_topic("travel", "full")
    wrong = SpeakerRetrievalContext(
        tenant_id="tenant", owner_conversation_id="owner", audience_conversation_id="private",
        audience_channel_id="other",
    )
    assert paging.calculate_depth_tokens("travel", DepthLevel.FULL, speaker_context=wrong) == 0
