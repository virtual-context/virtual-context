"""Tests for session date propagation through the pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from virtual_context.core.segmenter import TopicSegmenter, _parse_session_date
from virtual_context.types import (
    Message,
    QuoteResult,
    SegmentMetadata,
    SegmenterConfig,
    StoredSegment,
    StoredSummary,
    TaggedSegment,
    TagResult,
    TurnPair,
    TurnTagEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubTagGenerator:
    """Tag generator that returns a fixed primary tag."""

    def __init__(self, primary: str = "topic-a"):
        self.primary = primary

    def generate_tags(self, text: str, *args, **kwargs) -> TagResult:
        return TagResult(tags=[self.primary], primary=self.primary, source="stub")


class MultiTagGenerator:
    """Tag generator that returns different tags based on call count."""

    def __init__(self, tags: list[str]):
        self._tags = tags
        self._idx = 0

    def generate_tags(self, text: str, *args, **kwargs) -> TagResult:
        tag = self._tags[self._idx % len(self._tags)]
        self._idx += 1
        return TagResult(tags=[tag], primary=tag, source="stub")


def _make_pair(user_text: str, asst_text: str = "ok") -> list[Message]:
    """Make a user+assistant message pair."""
    return [
        Message(role="user", content=user_text),
        Message(role="assistant", content=asst_text),
    ]


# ---------------------------------------------------------------------------
# Tests: session date parsing
# ---------------------------------------------------------------------------

class TestParseSessionDate:

    def test_parse_session_date_from_text(self):
        """[Session from 2023/05/25] -> '2023/05/25'"""
        pair = TurnPair(messages=[
            Message(role="user", content="[Session from 2023/05/25 (Thu) 10:04]\nHello"),
            Message(role="assistant", content="Hi there"),
        ])
        assert _parse_session_date(pair) == "2023/05/25 (Thu) 10:04"

    def test_parse_session_date_missing(self):
        """No header -> empty string."""
        pair = TurnPair(messages=[
            Message(role="user", content="Just a regular message"),
            Message(role="assistant", content="Sure"),
        ])
        assert _parse_session_date(pair) == ""

    def test_parse_session_date_simple(self):
        """[Session from 2023/05/29] -> '2023/05/29'"""
        pair = TurnPair(messages=[
            Message(role="user", content="[Session from 2023/05/29]\nWhat's up"),
            Message(role="assistant", content="Not much"),
        ])
        assert _parse_session_date(pair) == "2023/05/29"


# ---------------------------------------------------------------------------
# Tests: segmenter session date handling
# ---------------------------------------------------------------------------

class TestSegmenterSessionDate:

    def test_segmenter_splits_on_session_change(self):
        """Session boundary forces a segment split even with same primary tag."""
        messages = [
            Message(role="user", content="[Session from 2023/05/25]\nFirst topic"),
            Message(role="assistant", content="Response 1"),
            Message(role="user", content="Still session 1"),
            Message(role="assistant", content="Response 2"),
            Message(role="user", content="[Session from 2023/05/29]\nNew session"),
            Message(role="assistant", content="Response 3"),
        ]

        segmenter = TopicSegmenter(
            tag_generator=StubTagGenerator("topic-a"),
            config=SegmenterConfig(),
        )
        segments = segmenter.segment(messages)

        assert len(segments) == 2
        assert segments[0].session_date == "2023/05/25"
        assert segments[1].session_date == "2023/05/29"

    def test_segmenter_same_tag_different_sessions(self):
        """Same primary_tag + different sessions -> 2 segments."""
        messages = [
            Message(role="user", content="[Session from 2023/05/25]\nHello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="[Session from 2023/05/29]\nHello again"),
            Message(role="assistant", content="Hi again"),
        ]

        segmenter = TopicSegmenter(
            tag_generator=StubTagGenerator("greetings"),
            config=SegmenterConfig(),
        )
        segments = segmenter.segment(messages)

        assert len(segments) == 2
        assert segments[0].primary_tag == "greetings"
        assert segments[1].primary_tag == "greetings"
        assert segments[0].session_date == "2023/05/25"
        assert segments[1].session_date == "2023/05/29"

    def test_segmenter_mid_session_tag_change_inherits_date(self):
        """Tag change mid-session: new segment inherits running session date."""
        messages = [
            Message(role="user", content="[Session from 2023/05/25]\nTalk about shoes"),
            Message(role="assistant", content="Sure, shoes are great"),
            Message(role="user", content="Now let's discuss cooking"),
            Message(role="assistant", content="Cooking is fun"),
        ]

        segmenter = TopicSegmenter(
            tag_generator=MultiTagGenerator(["shoes", "cooking"]),
            config=SegmenterConfig(),
        )
        segments = segmenter.segment(messages)

        assert len(segments) == 2
        assert segments[0].primary_tag == "shoes"
        assert segments[0].session_date == "2023/05/25"
        # Key: second segment has no [Session from] header but inherits it
        assert segments[1].primary_tag == "cooking"
        assert segments[1].session_date == "2023/05/25"

    def test_segmenter_same_session_same_tag(self):
        """Same session + same tag -> 1 segment."""
        messages = [
            Message(role="user", content="[Session from 2023/05/25]\nFirst"),
            Message(role="assistant", content="Response 1"),
            Message(role="user", content="Second in same session"),
            Message(role="assistant", content="Response 2"),
        ]

        segmenter = TopicSegmenter(
            tag_generator=StubTagGenerator("topic-a"),
            config=SegmenterConfig(),
        )
        segments = segmenter.segment(messages)

        assert len(segments) == 1
        assert segments[0].session_date == "2023/05/25"
        assert segments[0].turn_count == 2


# ---------------------------------------------------------------------------
# Tests: storage round-trip
# ---------------------------------------------------------------------------

class TestSessionDateStorage:

    def test_session_date_stored_in_sqlite_metadata(self, tmp_path):
        """Round-trip: store segment -> retrieve -> metadata.session_date."""
        from virtual_context.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=tmp_path / "test.db")
        seg = StoredSegment(
            ref="test-ref-1",
            session_id="sess-1",
            primary_tag="topic-a",
            tags=["topic-a"],
            summary="test summary",
            full_text="test full text",
            metadata=SegmentMetadata(
                session_date="2023/05/25 (Thu) 10:04",
                turn_count=1,
            ),
        )
        store.store_segment(seg)

        retrieved = store.get_segment("test-ref-1")
        assert retrieved is not None
        assert retrieved.metadata.session_date == "2023/05/25 (Thu) 10:04"

        # Also check summary view
        summary = store.get_summary("test-ref-1")
        assert summary is not None
        assert summary.metadata.session_date == "2023/05/25 (Thu) 10:04"

        store.close()

    def test_session_date_stored_in_filesystem(self, tmp_path):
        """Round-trip: filesystem store -> metadata.session_date."""
        from virtual_context.storage.filesystem import FilesystemStore

        store = FilesystemStore(root=tmp_path / "store")
        seg = StoredSegment(
            ref="test-ref-2",
            session_id="sess-1",
            primary_tag="topic-b",
            tags=["topic-b"],
            summary="fs test summary",
            full_text="fs test full text",
            metadata=SegmentMetadata(
                session_date="2023/05/29",
                turn_count=1,
            ),
        )
        store.store_segment(seg)

        retrieved = store.get_segment("test-ref-2")
        assert retrieved is not None
        assert retrieved.metadata.session_date == "2023/05/29"


# ---------------------------------------------------------------------------
# Tests: find_quote includes session_date
# ---------------------------------------------------------------------------

class TestFindQuoteSessionDate:

    def test_find_quote_includes_session_date_sqlite(self, tmp_path):
        """QuoteResult has session_date field from search_full_text."""
        from virtual_context.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=tmp_path / "test.db")
        seg = StoredSegment(
            ref="quote-ref-1",
            session_id="sess-1",
            primary_tag="shoes",
            tags=["shoes"],
            summary="shoes summary",
            full_text="I put my shoes under the bed",
            metadata=SegmentMetadata(
                session_date="2023/05/25 (Thu) 10:04",
                turn_count=1,
            ),
        )
        store.store_segment(seg)

        results = store.search_full_text("shoes")
        assert len(results) >= 1
        assert results[0].session_date == "2023/05/25 (Thu) 10:04"

        store.close()

    def test_find_quote_includes_session_date_filesystem(self, tmp_path):
        """QuoteResult has session_date field from filesystem search."""
        from virtual_context.storage.filesystem import FilesystemStore

        store = FilesystemStore(root=tmp_path / "store")
        seg = StoredSegment(
            ref="quote-ref-2",
            session_id="sess-1",
            primary_tag="shoes",
            tags=["shoes"],
            summary="shoes summary",
            full_text="I moved my shoes to the shoe rack",
            metadata=SegmentMetadata(
                session_date="2023/05/29",
                turn_count=1,
            ),
        )
        store.store_segment(seg)

        results = store.search_full_text("shoes")
        assert len(results) >= 1
        assert results[0].session_date == "2023/05/29"


# ---------------------------------------------------------------------------
# Tests: assembler session attribute
# ---------------------------------------------------------------------------

class TestAssemblerSessionDate:

    def _make_assembler(self):
        from virtual_context.core.assembler import ContextAssembler
        from virtual_context.types import AssemblerConfig
        return ContextAssembler(config=AssemblerConfig())

    def test_expanded_segment_shows_session(self):
        """_format_segments_section includes session= attribute."""
        assembler = self._make_assembler()
        seg = StoredSegment(
            ref="seg-1",
            primary_tag="shoes",
            tags=["shoes"],
            summary="shoes summary",
            metadata=SegmentMetadata(session_date="2023/05/25"),
        )
        result = assembler._format_segments_section("shoes", [seg])
        assert 'session="2023/05/25"' in result

    def test_expanded_full_shows_session(self):
        """_format_full_section includes session= attribute."""
        assembler = self._make_assembler()
        seg = StoredSegment(
            ref="seg-2",
            primary_tag="shoes",
            tags=["shoes"],
            summary="shoes summary",
            full_text="full text here",
            metadata=SegmentMetadata(session_date="2023/05/29"),
        )
        result = assembler._format_full_section("shoes", [seg])
        assert 'session="2023/05/29"' in result

    def test_no_session_attr_when_empty(self):
        """No session= attribute when session_date is empty."""
        assembler = self._make_assembler()
        seg = StoredSegment(
            ref="seg-3",
            primary_tag="shoes",
            tags=["shoes"],
            summary="shoes summary",
            metadata=SegmentMetadata(),
        )
        result = assembler._format_segments_section("shoes", [seg])
        assert "session=" not in result

    def test_summary_section_shows_session_dates(self):
        """_format_tag_section prefixes each summary with its session date."""
        assembler = self._make_assembler()
        summaries = [
            StoredSummary(
                ref="s1",
                primary_tag="shoes",
                tags=["shoes"],
                summary="User keeps sneakers under bed",
                metadata=SegmentMetadata(session_date="2023/05/25 (Thu) 10:04"),
                start_timestamp=datetime(2023, 5, 25, tzinfo=timezone.utc),
                end_timestamp=datetime(2023, 5, 25, tzinfo=timezone.utc),
            ),
            StoredSummary(
                ref="s2",
                primary_tag="shoes",
                tags=["shoes"],
                summary="User moved sneakers to shoe rack in closet",
                metadata=SegmentMetadata(session_date="2023/05/29 (Mon) 15:01"),
                start_timestamp=datetime(2023, 5, 29, tzinfo=timezone.utc),
                end_timestamp=datetime(2023, 5, 29, tzinfo=timezone.utc),
            ),
        ]
        result = assembler._format_tag_section("shoes", summaries)
        # Both session dates present as prefixes
        assert "[2023/05/25 (Thu) 10:04]" in result
        assert "[2023/05/29 (Mon) 15:01]" in result
        # Chronological order: May 25 before May 29
        idx_25 = result.index("[2023/05/25")
        idx_29 = result.index("[2023/05/29")
        assert idx_25 < idx_29

    def test_summary_section_no_prefix_when_no_session(self):
        """_format_tag_section omits prefix when session_date is empty."""
        assembler = self._make_assembler()
        summaries = [
            StoredSummary(
                ref="s3",
                primary_tag="shoes",
                tags=["shoes"],
                summary="Some shoe info",
                metadata=SegmentMetadata(),
                start_timestamp=datetime(2023, 5, 25, tzinfo=timezone.utc),
                end_timestamp=datetime(2023, 5, 25, tzinfo=timezone.utc),
            ),
        ]
        result = assembler._format_tag_section("shoes", summaries)
        assert "Some shoe info" in result
        assert "[" not in result.split("Some shoe info")[0].split("\n")[-1]

    def test_summary_section_sorted_chronologically(self):
        """_format_tag_section sorts summaries old→new by start_timestamp."""
        assembler = self._make_assembler()
        # Provide in reverse order — should still appear chronologically
        summaries = [
            StoredSummary(
                ref="s2",
                primary_tag="shoes",
                tags=["shoes"],
                summary="Later event",
                metadata=SegmentMetadata(session_date="2023/05/29"),
                start_timestamp=datetime(2023, 5, 29, tzinfo=timezone.utc),
                end_timestamp=datetime(2023, 5, 29, tzinfo=timezone.utc),
            ),
            StoredSummary(
                ref="s1",
                primary_tag="shoes",
                tags=["shoes"],
                summary="Earlier event",
                metadata=SegmentMetadata(session_date="2023/05/25"),
                start_timestamp=datetime(2023, 5, 25, tzinfo=timezone.utc),
                end_timestamp=datetime(2023, 5, 25, tzinfo=timezone.utc),
            ),
        ]
        result = assembler._format_tag_section("shoes", summaries)
        assert result.index("Earlier event") < result.index("Later event")


# ---------------------------------------------------------------------------
# Tests: ingest_history session date tracking
# ---------------------------------------------------------------------------

class TestIngestHistorySessionDate:

    def test_ingest_history_tracks_session_date(self, tmp_path):
        """TurnTagEntry.session_date populated from [Session from ...] headers."""
        from virtual_context.types import VirtualContextConfig, TagGeneratorConfig, StorageConfig
        from virtual_context.engine import VirtualContextEngine

        config = VirtualContextConfig(
            storage_root=str(tmp_path / "vc"),
            storage=StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
                root=str(tmp_path / "store"),
            ),
            tag_generator=TagGeneratorConfig(type="keyword"),
        )

        engine = VirtualContextEngine(config=config)
        history = [
            Message(role="user", content="[Session from 2023/05/25 (Thu) 10:04]\nHello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="What's new?"),
            Message(role="assistant", content="Nothing much"),
            Message(role="user", content="[Session from 2023/05/29 (Mon) 14:30]\nBack again"),
            Message(role="assistant", content="Welcome back"),
        ]

        engine.ingest_history(history)

        entries = list(engine._turn_tag_index.entries)
        assert len(entries) == 3
        assert entries[0].session_date == "2023/05/25 (Thu) 10:04"
        assert entries[1].session_date == "2023/05/25 (Thu) 10:04"  # inherited
        assert entries[2].session_date == "2023/05/29 (Mon) 14:30"

    def test_timestamp_fallback_for_proxy(self, tmp_path):
        """No text header + Message.timestamp -> session_date from timestamp."""
        from virtual_context.types import VirtualContextConfig, TagGeneratorConfig, StorageConfig
        from virtual_context.engine import VirtualContextEngine

        config = VirtualContextConfig(
            storage_root=str(tmp_path / "vc"),
            storage=StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "test.db"),
                root=str(tmp_path / "store"),
            ),
            tag_generator=TagGeneratorConfig(type="keyword"),
        )

        engine = VirtualContextEngine(config=config)
        ts = datetime(2023, 5, 25, 10, 4, 0, tzinfo=timezone.utc)
        history = [
            Message(role="user", content="Hello", timestamp=ts),
            Message(role="assistant", content="Hi"),
        ]

        engine.ingest_history(history)

        entries = list(engine._turn_tag_index.entries)
        assert len(entries) == 1
        assert entries[0].session_date == "2023-05-25T10:04:00"
