"""Tests for FilesystemStore."""

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.types import SegmentMetadata, StoredSegment


def make_segment(domain: str = "legal", ref: str = "test-ref-1") -> StoredSegment:
    now = datetime.now(timezone.utc)
    return StoredSegment(
        ref=ref,
        session_id="test-session",
        domain=domain,
        secondary_domains=[],
        summary="This is a test summary about legal matters.",
        summary_tokens=20,
        full_text="User: What about the case?\nAssistant: The case is pending.",
        full_tokens=100,
        messages=[
            {"role": "user", "content": "What about the case?"},
            {"role": "assistant", "content": "The case is pending."},
        ],
        metadata=SegmentMetadata(
            entities=["Judge Smith", "Case 24-cv-1234"],
            key_decisions=["File by January 30"],
            action_items=["Review brief"],
            turn_count=1,
        ),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
        compaction_model="test-model",
        compression_ratio=0.2,
    )


@pytest.mark.asyncio
async def test_store_and_retrieve(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    seg = make_segment()
    ref = await store.store_segment(seg)
    assert ref == "test-ref-1"

    loaded = await store.get_segment("test-ref-1")
    assert loaded is not None
    assert loaded.domain == "legal"
    assert loaded.summary == "This is a test summary about legal matters."
    assert loaded.metadata.entities == ["Judge Smith", "Case 24-cv-1234"]


@pytest.mark.asyncio
async def test_get_summary(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    await store.store_segment(make_segment())
    summary = await store.get_summary("test-ref-1")
    assert summary is not None
    assert summary.domain == "legal"
    assert summary.summary_tokens == 20


@pytest.mark.asyncio
async def test_get_summaries_by_domain(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    await store.store_segment(make_segment("legal", "ref-1"))
    await store.store_segment(make_segment("medical", "ref-2"))
    await store.store_segment(make_segment("legal", "ref-3"))

    legal = await store.get_summaries(domain="legal")
    assert len(legal) == 2

    medical = await store.get_summaries(domain="medical")
    assert len(medical) == 1


@pytest.mark.asyncio
async def test_list_domains(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    await store.store_segment(make_segment("legal", "ref-1"))
    await store.store_segment(make_segment("legal", "ref-2"))
    await store.store_segment(make_segment("medical", "ref-3"))

    domains = await store.list_domains()
    assert len(domains) == 2
    legal = next(d for d in domains if d.domain == "legal")
    assert legal.segment_count == 2


@pytest.mark.asyncio
async def test_delete_segment(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    await store.store_segment(make_segment())
    assert await store.delete_segment("test-ref-1")
    assert await store.get_segment("test-ref-1") is None


@pytest.mark.asyncio
async def test_search(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    await store.store_segment(make_segment())
    results = await store.search("Judge Smith")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_upsert(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    seg = make_segment()
    await store.store_segment(seg)
    seg.summary = "Updated summary"
    await store.store_segment(seg)
    loaded = await store.get_segment("test-ref-1")
    assert loaded.summary == "Updated summary"


@pytest.mark.asyncio
async def test_not_found(tmp_store_dir):
    store = FilesystemStore(tmp_store_dir)
    assert await store.get_segment("nonexistent") is None
    assert await store.get_summary("nonexistent") is None
