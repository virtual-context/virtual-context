"""Tests for the fill pass pipeline stage."""

import copy
import json
from unittest.mock import MagicMock
from datetime import datetime, timezone

from virtual_context.proxy.formats import detect_format
from virtual_context.types import (
    RetrievalResult, StoredSummary, AssembledContext, TagSummary, SegmentMetadata,
)


def test_retrieval_result_has_overflow_field():
    rr = RetrievalResult()
    assert hasattr(rr, "overflow_summaries")
    assert rr.overflow_summaries == []


def test_retrieval_result_overflow_accepts_summaries():
    s = StoredSummary(ref="seg_001", primary_tag="cooking", summary="test", summary_tokens=50)
    rr = RetrievalResult(overflow_summaries=[s])
    assert len(rr.overflow_summaries) == 1
    assert rr.overflow_summaries[0].ref == "seg_001"


def test_assembled_context_has_retrieval_result():
    ac = AssembledContext()
    assert ac.retrieval_result is None


def test_assembled_context_has_presented_tags():
    ac = AssembledContext()
    assert isinstance(ac.presented_tags, set)
    assert len(ac.presented_tags) == 0


def test_assembled_context_presented_tags_populated():
    ac = AssembledContext(presented_tags={"cooking", "baking", "recipes"})
    assert "cooking" in ac.presented_tags
    assert len(ac.presented_tags) == 3


def test_format_tag_section_standalone():
    """Standalone format_tag_section produces the same XML format as the assembler."""
    from virtual_context.core.assembler import format_tag_section
    from virtual_context.types import StoredSummary, SegmentMetadata
    from datetime import datetime, timezone

    s1 = StoredSummary(
        ref="s1", primary_tag="cooking", summary="Italian cooking techniques",
        summary_tokens=100, tags=["cooking", "italian"],
        metadata=SegmentMetadata(),
        start_timestamp=datetime.now(timezone.utc),
    )

    result = format_tag_section("cooking", [s1])
    assert '<virtual-context tags="cooking, italian"' in result
    assert "[1/1]" in result
    assert "Italian cooking techniques" in result
    assert "</virtual-context>" in result


def _make_summary(ref: str, tag: str, tokens: int) -> StoredSummary:
    return StoredSummary(
        ref=ref, primary_tag=tag, tags=[tag],
        summary=f"Summary for {tag}", summary_tokens=tokens,
        metadata=SegmentMetadata(),
        start_timestamp=datetime.now(timezone.utc),
    )


def _make_anthropic_body(user_texts: list[str]) -> dict:
    messages = []
    for text in user_texts:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"Reply to: {text}"}]})
    messages.append({"role": "user", "content": "current question"})
    return {"system": "You are helpful.", "messages": messages, "model": "claude-opus-4-6"}


def test_fill_pass_no_op_when_at_target():
    from virtual_context.proxy.message_filter import fill_pass
    body = _make_anthropic_body(["hello", "world"])
    fmt = detect_format(body)
    result, summaries, turns = fill_pass(
        body=body, fmt=fmt, outbound_tokens=90000, target_tokens=90000,
        assembled=None, pre_filter_body=body,
        store=MagicMock(), conversation_id="test",
    )
    assert summaries == 0
    assert turns == 0


def test_fill_pass_no_op_when_over_target():
    from virtual_context.proxy.message_filter import fill_pass
    body = _make_anthropic_body(["hello"])
    fmt = detect_format(body)
    result, summaries, turns = fill_pass(
        body=body, fmt=fmt, outbound_tokens=100000, target_tokens=90000,
        assembled=None, pre_filter_body=body,
        store=MagicMock(), conversation_id="test",
    )
    assert summaries == 0
    assert turns == 0


def test_fill_pass_adds_breadth_summaries():
    from virtual_context.proxy.message_filter import fill_pass

    body = _make_anthropic_body(["hello"])
    fmt = detect_format(body)

    ts1 = TagSummary(tag="cooking", summary="Italian cooking", summary_tokens=50,
                     source_segment_refs=["seg_a"], updated_at=datetime.now(timezone.utc))
    ts2 = TagSummary(tag="baking", summary="Bread baking", summary_tokens=50,
                     source_segment_refs=["seg_b"], updated_at=datetime.now(timezone.utc))
    mock_store = MagicMock()
    mock_store.get_all_tag_summaries.return_value = [ts1, ts2]

    assembled = AssembledContext(
        presented_segment_refs=set(),
        presented_tags=set(),
        tag_sections={},
        retrieval_result=RetrievalResult(),
    )

    result, summaries_added, turns_added = fill_pass(
        body=body, fmt=fmt, outbound_tokens=70000, target_tokens=90000,
        assembled=assembled, pre_filter_body=copy.deepcopy(body),
        store=mock_store, conversation_id="test",
        summary_ratio=1.0,
    )
    assert summaries_added >= 1
    mock_store.get_all_tag_summaries.assert_called_once_with(conversation_id="test")
