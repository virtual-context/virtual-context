"""Tests for the fill pass pipeline stage."""

import copy
import json
import re
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


def test_presented_tags_from_segments_and_full_sections():
    """presented_tags must include secondary tags from SEGMENTS/FULL sections."""
    tag_sections = {
        "cooking": (
            '<virtual-context tags="cooking, italian" segments="2">\n'
            "[1/2]\nPasta techniques\n\n---\n\n[2/2]\nRisotto methods\n"
            "</virtual-context>"
        ),
        "baking": (
            '<virtual-context tags="baking, bread, sourdough" segments="1">\n'
            "[1/1]\nSourdough starter maintenance\n"
            "</virtual-context>"
        ),
    }

    _vc_tags_re = re.compile(r'<virtual-context\s+tags="([^"]*)"')
    presented_tags: set[str] = set()
    for section_text in tag_sections.values():
        for m in _vc_tags_re.finditer(section_text):
            for t in m.group(1).split(", "):
                t = t.strip()
                if t:
                    presented_tags.add(t)
    # Also include section keys
    presented_tags.update(tag_sections.keys())

    assert "cooking" in presented_tags
    assert "baking" in presented_tags
    assert "italian" in presented_tags
    assert "bread" in presented_tags
    assert "sourdough" in presented_tags
    assert "grilling" not in presented_tags


def test_presented_tags_fallback_to_section_key():
    """When tags attr is empty, section key should still be covered."""
    tag_sections = {
        "misc": '<virtual-context tags="" segments="1">\n[1/1]\nRandom\n</virtual-context>',
    }

    _vc_tags_re = re.compile(r'<virtual-context\s+tags="([^"]*)"')
    presented_tags: set[str] = set()
    for section_text in tag_sections.values():
        for m in _vc_tags_re.finditer(section_text):
            for t in m.group(1).split(", "):
                t = t.strip()
                if t:
                    presented_tags.add(t)
    presented_tags.update(tag_sections.keys())

    # "misc" comes from keys(), not from regex (empty attr)
    assert "misc" in presented_tags


def test_fill_pass_accounting_summary_and_turns():
    """After fill pass, counts must match actual content added."""
    from virtual_context.proxy.message_filter import fill_pass

    body = _make_anthropic_body(["hello", "world"])
    pre_filter = copy.deepcopy(body)
    extra_msgs = [
        {"role": "user", "content": "older message 1"},
        {"role": "assistant", "content": [{"type": "text", "text": "older reply 1"}]},
        {"role": "user", "content": "older message 2"},
        {"role": "assistant", "content": [{"type": "text", "text": "older reply 2"}]},
    ]
    pre_filter["messages"] = extra_msgs + pre_filter["messages"]

    fmt = detect_format(body)

    ts1 = TagSummary(tag="history", summary="Historical events discussed",
                     summary_tokens=30, source_segment_refs=["seg_h1"],
                     updated_at=datetime.now(timezone.utc))
    mock_store = MagicMock()
    mock_store.get_all_tag_summaries.return_value = [ts1]

    assembled = AssembledContext(
        presented_segment_refs=set(),
        presented_tags=set(),
        tag_sections={},
        retrieval_result=RetrievalResult(),
    )

    result_body, summaries_added, turns_added = fill_pass(
        body=body, fmt=fmt,
        outbound_tokens=70000, target_tokens=90000,
        assembled=assembled, pre_filter_body=pre_filter,
        store=mock_store, conversation_id="test-conv",
        summary_ratio=0.5,
    )

    assert summaries_added >= 1
    result_json = json.dumps(result_body)
    assert "Historical events discussed" in result_json


def test_fill_pass_sanitizes_restored_turns():
    """Restored turns must have thinking blocks stripped and media replaced."""
    from virtual_context.proxy.message_filter import _sanitize_restored_turn

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "show me the image"},
            {"type": "image", "source": {"type": "base64", "data": "abc123"}},
        ]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "let me think..."},
            {"type": "text", "text": "here is my answer"},
        ]},
    ]

    sanitized = _sanitize_restored_turn(messages)

    user_content = sanitized[0]["content"]
    assert any(b.get("text") == "[image removed from restored turn]" for b in user_content)
    assert not any(b.get("type") == "image" for b in user_content)

    asst_content = sanitized[1]["content"]
    assert not any(b.get("type") == "thinking" for b in asst_content)
    assert any(b.get("text") == "here is my answer" for b in asst_content)


def test_fill_pass_no_restore_tool_references():
    """Media placeholders must NOT reference vc_restore_tool."""
    from virtual_context.proxy.message_filter import _sanitize_restored_turn

    messages = [
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
        ]},
    ]

    sanitized = _sanitize_restored_turn(messages)
    result_json = json.dumps(sanitized)
    assert "vc_restore_tool" not in result_json
    assert "image removed from restored turn" in result_json
