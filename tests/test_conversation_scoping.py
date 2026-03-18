"""Tests for conversation-scoped retrieval across all store methods."""

import inspect

import pytest

from virtual_context.core.store import ContextStore


class TestABCSignatures:
    """Every retrieval method on ContextStore must accept conversation_id."""

    SCOPED_METHODS = [
        "get_summaries_by_tags",
        "get_segments_by_tags",
        "search",
        "search_full_text",
        "get_segment",
        "get_summary",
        "query_facts",
        "search_facts",
        "search_tool_outputs",
        "get_fact_count_by_tags",
        "get_unique_fact_verbs",
        "get_all_tags",
    ]

    @pytest.mark.parametrize("method_name", SCOPED_METHODS)
    def test_method_accepts_conversation_id(self, method_name):
        method = getattr(ContextStore, method_name)
        sig = inspect.signature(method)
        assert "conversation_id" in sig.parameters, (
            f"{method_name} missing conversation_id parameter"
        )
        param = sig.parameters["conversation_id"]
        assert param.default is None, (
            f"{method_name} conversation_id should default to None"
        )


from datetime import datetime, timezone

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import StoredSegment, SegmentMetadata


def _make_two_conversation_store(tmp_path):
    """Create a SQLite store with segments in two different conversations."""
    store = SQLiteStore(db_path=str(tmp_path / "test.db"))
    now = datetime.now(timezone.utc)
    base = dict(
        summary_tokens=10, full_tokens=50, full_text="full text here",
        metadata=SegmentMetadata(), created_at=now,
        start_timestamp=now, end_timestamp=now,
    )
    store.store_segment(StoredSegment(
        ref="conv-a-1", conversation_id="conv-a",
        primary_tag="cooking", tags=["cooking", "recipes"],
        summary="How to make pasta.", **base,
    ))
    store.store_segment(StoredSegment(
        ref="conv-a-2", conversation_id="conv-a",
        primary_tag="travel", tags=["travel", "europe"],
        summary="Trip to Italy planned.", **base,
    ))
    store.store_segment(StoredSegment(
        ref="conv-b-1", conversation_id="conv-b",
        primary_tag="cooking", tags=["cooking", "baking"],
        summary="Sourdough bread recipe.", **base,
    ))
    store.store_segment(StoredSegment(
        ref="conv-b-2", conversation_id="conv-b",
        primary_tag="fitness", tags=["fitness", "running"],
        summary="Marathon training plan.", **base,
    ))
    return store


class TestSQLiteSegmentScoping:
    def test_get_summaries_by_tags_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.get_summaries_by_tags(
            tags=["cooking"], conversation_id="conv-a",
        )
        refs = [r.ref for r in results]
        assert "conv-a-1" in refs
        assert "conv-b-1" not in refs

    def test_get_summaries_by_tags_unscoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.get_summaries_by_tags(tags=["cooking"])
        refs = [r.ref for r in results]
        assert "conv-a-1" in refs
        assert "conv-b-1" in refs

    def test_get_segments_by_tags_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.get_segments_by_tags(
            tags=["cooking"], conversation_id="conv-a",
        )
        refs = [r.ref for r in results]
        assert "conv-a-1" in refs
        assert "conv-b-1" not in refs

    def test_search_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.search(query="pasta sourdough", conversation_id="conv-a")
        refs = [r.ref for r in results]
        for ref in refs:
            assert ref.startswith("conv-a")

    def test_search_full_text_scoped(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        results = store.search_full_text(query="full text", conversation_id="conv-a")
        refs = [r.segment_ref for r in results]
        for ref in refs:
            assert ref.startswith("conv-a")

    def test_get_segment_scoped_returns_own(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        seg = store.get_segment("conv-a-1", conversation_id="conv-a")
        assert seg is not None
        assert seg.ref == "conv-a-1"

    def test_get_segment_scoped_rejects_other(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        seg = store.get_segment("conv-b-1", conversation_id="conv-a")
        assert seg is None

    def test_get_summary_scoped_rejects_other(self, tmp_path):
        store = _make_two_conversation_store(tmp_path)
        summary = store.get_summary("conv-b-1", conversation_id="conv-a")
        assert summary is None
