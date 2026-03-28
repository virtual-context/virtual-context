"""Tests for the fill pass pipeline stage."""

from virtual_context.types import RetrievalResult, StoredSummary, AssembledContext


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
