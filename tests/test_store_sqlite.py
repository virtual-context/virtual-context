"""Tests for SQLite storage backend."""

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import SegmentMetadata, SessionStats, StoredSegment, StoredSummary, TagSummary


@pytest.fixture
def store(tmp_sqlite_db):
    s = SQLiteStore(db_path=tmp_sqlite_db)
    yield s
    s.close()


def _make_segment(
    ref: str = "test-ref-1",
    primary_tag: str = "legal",
    tags: list[str] | None = None,
    summary: str = "Test summary about legal matters",
    created_at: datetime | None = None,
) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        session_id="session-1",
        primary_tag=primary_tag,
        tags=tags or [primary_tag],
        summary=summary,
        summary_tokens=50,
        full_text="Full conversation text here",
        full_tokens=200,
        messages=[{"role": "user", "content": "test"}],
        metadata=SegmentMetadata(
            entities=["entity1"],
            key_decisions=["decision1"],
            turn_count=2,
        ),
        created_at=created_at or datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        start_timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        end_timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
        compaction_model="qwen3:4b-instruct-2507-fp16",
        compression_ratio=0.25,
    )


class TestSQLiteStore:
    def test_store_and_retrieve_segment(self, store):
        seg = _make_segment()
        ref = store.store_segment(seg)
        assert ref == "test-ref-1"

        retrieved = store.get_segment("test-ref-1")
        assert retrieved is not None
        assert retrieved.primary_tag == "legal"
        assert retrieved.summary == "Test summary about legal matters"
        assert retrieved.tags == ["legal"]

    def test_get_segment_not_found(self, store):
        result = store.get_segment("nonexistent")
        assert result is None

    def test_get_summary(self, store):
        store.store_segment(_make_segment())
        summary = store.get_summary("test-ref-1")
        assert summary is not None
        assert summary.summary == "Test summary about legal matters"
        assert summary.primary_tag == "legal"

    def test_upsert(self, store):
        seg1 = _make_segment(summary="First version")
        store.store_segment(seg1)

        seg2 = _make_segment(summary="Updated version")
        store.store_segment(seg2)

        retrieved = store.get_segment("test-ref-1")
        assert retrieved.summary == "Updated version"

    def test_get_summaries_by_tags_single_tag(self, store):
        store.store_segment(_make_segment(ref="r1", tags=["legal", "court"]))
        store.store_segment(_make_segment(ref="r2", tags=["medical"]))
        store.store_segment(_make_segment(ref="r3", tags=["legal"]))

        results = store.get_summaries_by_tags(tags=["legal"])
        assert len(results) == 2
        refs = {r.ref for r in results}
        assert "r1" in refs
        assert "r3" in refs

    def test_get_summaries_by_tags_overlap(self, store):
        store.store_segment(_make_segment(ref="r1", tags=["legal", "court", "case"]))
        store.store_segment(_make_segment(ref="r2", tags=["legal"]))

        # min_overlap=2: only r1 matches both "legal" and "court"
        results = store.get_summaries_by_tags(
            tags=["legal", "court"], min_overlap=2
        )
        assert len(results) == 1
        assert results[0].ref == "r1"

    def test_get_summaries_by_tags_empty(self, store):
        results = store.get_summaries_by_tags(tags=[])
        assert results == []

    def test_get_summaries_by_tags_with_time_filter(self, store):
        early = datetime(2026, 1, 10, tzinfo=timezone.utc)
        late = datetime(2026, 1, 20, tzinfo=timezone.utc)

        store.store_segment(_make_segment(ref="r1", tags=["legal"], created_at=early))
        store.store_segment(_make_segment(ref="r2", tags=["legal"], created_at=late))

        cutoff = datetime(2026, 1, 15, tzinfo=timezone.utc)
        results = store.get_summaries_by_tags(tags=["legal"], before=cutoff)
        assert len(results) == 1
        assert results[0].ref == "r1"

    def test_get_all_tags(self, store):
        store.store_segment(_make_segment(ref="r1", tags=["legal", "court"]))
        store.store_segment(_make_segment(ref="r2", tags=["legal", "medical"]))

        tag_stats = store.get_all_tags()
        tag_names = {t.tag for t in tag_stats}
        assert "legal" in tag_names
        assert "court" in tag_names
        assert "medical" in tag_names

        legal_stat = next(t for t in tag_stats if t.tag == "legal")
        assert legal_stat.usage_count == 2

    def test_search(self, store):
        store.store_segment(_make_segment(ref="r1", summary="Database migration failed"))
        store.store_segment(_make_segment(ref="r2", summary="Fixed the auth bug"))

        results = store.search("migration")
        assert len(results) >= 1
        assert any("migration" in r.summary.lower() for r in results)

    def test_search_with_tag_filter(self, store):
        store.store_segment(
            _make_segment(ref="r1", tags=["database"], summary="Database migration")
        )
        store.store_segment(
            _make_segment(ref="r2", tags=["auth"], summary="Auth migration")
        )

        results = store.search("migration", tags=["database"])
        refs = {r.ref for r in results}
        assert "r1" in refs

    def test_delete_segment(self, store):
        store.store_segment(_make_segment())
        deleted = store.delete_segment("test-ref-1")
        assert deleted is True

        result = store.get_segment("test-ref-1")
        assert result is None

    def test_delete_nonexistent(self, store):
        deleted = store.delete_segment("nonexistent")
        assert deleted is False

    def test_cleanup_by_age(self, store):
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        store.store_segment(_make_segment(ref="old", created_at=old))
        store.store_segment(_make_segment(ref="new"))

        deleted = store.cleanup(max_age=timedelta(days=365))
        assert deleted == 1

        assert store.get_segment("old") is None
        assert store.get_segment("new") is not None

    def test_tag_aliases(self, store):
        store.set_tag_alias("db", "database")
        aliases = store.get_tag_aliases()
        assert aliases["db"] == "database"

    def test_metadata_preserved(self, store):
        seg = _make_segment()
        seg.metadata = SegmentMetadata(
            entities=["Judge Smith", "Case 123"],
            key_decisions=["Approved motion"],
            action_items=["File by Jan 30"],
            date_references=["January 30"],
            turn_count=3,
        )
        store.store_segment(seg)

        retrieved = store.get_segment("test-ref-1")
        assert retrieved.metadata.entities == ["Judge Smith", "Case 123"]
        assert retrieved.metadata.key_decisions == ["Approved motion"]
        assert retrieved.metadata.action_items == ["File by Jan 30"]
        assert retrieved.metadata.turn_count == 3

    def test_tag_summary_crud(self, store):
        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        ts = TagSummary(
            tag="legal",
            summary="Legal discussion summary",
            summary_tokens=30,
            source_segment_refs=["seg-1", "seg-2"],
            source_turn_numbers=[0, 1, 2],
            covers_through_turn=5,
            created_at=now,
            updated_at=now,
        )
        store.save_tag_summary(ts)

        retrieved = store.get_tag_summary("legal")
        assert retrieved is not None
        assert retrieved.tag == "legal"
        assert retrieved.summary == "Legal discussion summary"
        assert retrieved.summary_tokens == 30
        assert retrieved.source_segment_refs == ["seg-1", "seg-2"]
        assert retrieved.source_turn_numbers == [0, 1, 2]
        assert retrieved.covers_through_turn == 5

    def test_tag_summary_not_found(self, store):
        assert store.get_tag_summary("nonexistent") is None

    def test_tag_summary_upsert(self, store):
        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        ts1 = TagSummary(tag="legal", summary="First", summary_tokens=10, created_at=now, updated_at=now)
        store.save_tag_summary(ts1)

        ts2 = TagSummary(tag="legal", summary="Updated", summary_tokens=15, covers_through_turn=10, created_at=now, updated_at=now)
        store.save_tag_summary(ts2)

        retrieved = store.get_tag_summary("legal")
        assert retrieved.summary == "Updated"
        assert retrieved.covers_through_turn == 10

    def test_get_all_tag_summaries(self, store):
        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        store.save_tag_summary(TagSummary(tag="legal", summary="Legal stuff", summary_tokens=10, created_at=now, updated_at=now))
        store.save_tag_summary(TagSummary(tag="medical", summary="Medical stuff", summary_tokens=15, created_at=now, updated_at=now))

        all_ts = store.get_all_tag_summaries()
        assert len(all_ts) == 2
        tags = {ts.tag for ts in all_ts}
        assert tags == {"legal", "medical"}


class TestGetSessionStats:
    def test_empty_store(self, store):
        result = store.get_session_stats()
        assert result == []

    def test_single_session(self, store):
        store.store_segment(_make_segment(ref="r1", tags=["legal", "court"]))
        store.store_segment(_make_segment(ref="r2", tags=["medical"]))

        stats = store.get_session_stats()
        assert len(stats) == 1
        assert stats[0].session_id == "session-1"
        assert stats[0].segment_count == 2

    def test_two_sessions_grouped(self, store):
        seg1 = _make_segment(ref="r1", tags=["legal"])
        store.store_segment(seg1)

        seg2 = _make_segment(ref="r2", tags=["medical"])
        seg2.session_id = "session-2"
        store.store_segment(seg2)

        seg3 = _make_segment(ref="r3", tags=["court"])
        seg3.session_id = "session-2"
        store.store_segment(seg3)

        stats = store.get_session_stats()
        assert len(stats) == 2

        by_id = {s.session_id: s for s in stats}
        assert by_id["session-1"].segment_count == 1
        assert by_id["session-2"].segment_count == 2

    def test_token_aggregation(self, store):
        seg1 = _make_segment(ref="r1", tags=["legal"])
        seg1.full_tokens = 1000
        seg1.summary_tokens = 250
        store.store_segment(seg1)

        seg2 = _make_segment(ref="r2", tags=["medical"])
        seg2.full_tokens = 3000
        seg2.summary_tokens = 750
        store.store_segment(seg2)

        stats = store.get_session_stats()
        assert len(stats) == 1
        assert stats[0].total_full_tokens == 4000
        assert stats[0].total_summary_tokens == 1000
        assert stats[0].compression_ratio == 0.25

    def test_distinct_tags(self, store):
        store.store_segment(_make_segment(ref="r1", tags=["legal", "court"]))
        store.store_segment(_make_segment(ref="r2", tags=["legal", "medical"]))

        stats = store.get_session_stats()
        assert len(stats) == 1
        tags = stats[0].distinct_tags
        assert sorted(tags) == ["court", "legal", "medical"]

    def test_sorted_newest_first(self, store):
        early = datetime(2026, 1, 10, tzinfo=timezone.utc)
        late = datetime(2026, 1, 20, tzinfo=timezone.utc)

        seg1 = _make_segment(ref="r1", tags=["legal"], created_at=early)
        seg1.session_id = "old-session"
        store.store_segment(seg1)

        seg2 = _make_segment(ref="r2", tags=["legal"], created_at=late)
        seg2.session_id = "new-session"
        store.store_segment(seg2)

        stats = store.get_session_stats()
        assert len(stats) == 2
        assert stats[0].session_id == "new-session"
        assert stats[1].session_id == "old-session"

    def test_empty_session_id_excluded(self, store):
        seg = _make_segment(ref="r1", tags=["legal"])
        seg.session_id = ""
        store.store_segment(seg)

        stats = store.get_session_stats()
        assert len(stats) == 0

    def test_compaction_model_preserved(self, store):
        store.store_segment(_make_segment(ref="r1", tags=["legal"]))

        stats = store.get_session_stats()
        assert stats[0].compaction_model == "qwen3:4b-instruct-2507-fp16"

    def test_time_range(self, store):
        early = datetime(2026, 1, 10, tzinfo=timezone.utc)
        late = datetime(2026, 1, 20, tzinfo=timezone.utc)

        store.store_segment(_make_segment(ref="r1", tags=["a"], created_at=early))
        store.store_segment(_make_segment(ref="r2", tags=["b"], created_at=late))

        stats = store.get_session_stats()
        assert stats[0].oldest_segment == early
        assert stats[0].newest_segment == late
