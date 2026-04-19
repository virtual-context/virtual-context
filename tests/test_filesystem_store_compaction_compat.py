"""Regression: FilesystemStore must accept compaction-guard kwargs
(operation_id, owner_worker_id, lifecycle_epoch) and ignore them.

Filesystem backend has no compaction_operation table, so guards are a
no-op. Previously the pipeline would TypeError on CLI/TUI/headless
compaction when passing these kwargs to:
  - store_segment
  - save_tag_summary
  - replace_facts_for_segment
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
from pathlib import Path


def _make_store(tmp_path: Path):
    from virtual_context.storage.filesystem import FilesystemStore
    return FilesystemStore(tmp_path / "store")


def _make_segment():
    from virtual_context.types import StoredSegment
    return StoredSegment(
        ref="seg-test-01",
        conversation_id="conv-test",
        primary_tag="general",
        tags=["general"],
        summary="test summary",
        summary_tokens=3,
        full_text="test full text",
        full_tokens=3,
    )


def _make_tag_summary():
    from virtual_context.types import TagSummary
    return TagSummary(
        tag="general",
        summary="tag summary",
        description="test tag",
        summary_tokens=2,
    )


# ---------------------------------------------------------------------------
# store_segment
# ---------------------------------------------------------------------------

def test_store_segment_accepts_guard_kwargs(tmp_path):
    """store_segment must not raise TypeError when called with guard kwargs."""
    store = _make_store(tmp_path)
    seg = _make_segment()

    # Must not raise
    ref = store.store_segment(
        seg,
        operation_id="op-abc",
        owner_worker_id="worker-1",
        lifecycle_epoch=3,
    )
    assert ref == seg.ref


def test_store_segment_writes_unconditionally(tmp_path):
    """Filesystem store ignores guard kwargs — write always succeeds."""
    store = _make_store(tmp_path)
    seg = _make_segment()

    # Write with guards set
    store.store_segment(
        seg,
        operation_id="op-xyz",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )

    # Segment must be readable back
    result = store.get_segment(seg.ref)
    assert result is not None
    assert result.ref == seg.ref
    assert result.summary == seg.summary


def test_store_segment_works_without_guard_kwargs(tmp_path):
    """store_segment must still work when called without guard kwargs (regression guard)."""
    store = _make_store(tmp_path)
    seg = _make_segment()
    seg.ref = "seg-no-guard"

    ref = store.store_segment(seg)
    assert ref == seg.ref
    assert store.get_segment(seg.ref) is not None


# ---------------------------------------------------------------------------
# save_tag_summary
# ---------------------------------------------------------------------------

def test_save_tag_summary_accepts_guard_kwargs(tmp_path):
    """save_tag_summary must not raise TypeError when called with guard kwargs."""
    store = _make_store(tmp_path)
    ts = _make_tag_summary()

    # Must not raise
    store.save_tag_summary(
        ts,
        conversation_id="conv-test",
        operation_id="op-abc",
        owner_worker_id="worker-1",
        lifecycle_epoch=3,
    )


def test_save_tag_summary_writes_unconditionally(tmp_path):
    """Filesystem store ignores guard kwargs — tag summary write always succeeds."""
    store = _make_store(tmp_path)
    ts = _make_tag_summary()

    store.save_tag_summary(
        ts,
        conversation_id="conv-test",
        operation_id="op-xyz",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )

    result = store.get_tag_summary("general")
    assert result is not None
    assert result.tag == "general"
    assert result.summary == ts.summary


def test_save_tag_summary_works_without_guard_kwargs(tmp_path):
    """save_tag_summary must still work when called without guard kwargs."""
    store = _make_store(tmp_path)
    ts = _make_tag_summary()

    store.save_tag_summary(ts, conversation_id="conv-test")
    assert store.get_tag_summary("general") is not None


# ---------------------------------------------------------------------------
# replace_facts_for_segment
# ---------------------------------------------------------------------------

def test_replace_facts_for_segment_accepts_guard_kwargs(tmp_path):
    """replace_facts_for_segment must not raise TypeError with guard kwargs."""
    store = _make_store(tmp_path)

    # Must not raise; returns (deleted, inserted) = (0, 0) for filesystem
    deleted, inserted = store.replace_facts_for_segment(
        "conv-test",
        "seg-test-01",
        [],
        operation_id="op-abc",
        owner_worker_id="worker-1",
        lifecycle_epoch=3,
    )
    assert deleted == 0
    assert inserted == 0


def test_replace_facts_for_segment_works_without_guard_kwargs(tmp_path):
    """replace_facts_for_segment must still work when called without guard kwargs."""
    store = _make_store(tmp_path)
    deleted, inserted = store.replace_facts_for_segment("conv-test", "seg-1", [])
    assert deleted == 0
    assert inserted == 0
