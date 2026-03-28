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
