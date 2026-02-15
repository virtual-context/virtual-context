"""Integration tests for DomainCompactor with real Ollama."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import Message, TaggedSegment


def _make_segment(
    messages: list[Message],
    primary_tag: str = "legal",
    tags: list[str] | None = None,
) -> TaggedSegment:
    """Helper to build a TaggedSegment from messages."""
    ts = messages[0].timestamp or datetime.now(timezone.utc)
    return TaggedSegment(
        primary_tag=primary_tag,
        tags=tags or [primary_tag],
        messages=messages,
        token_count=sum(len(m.content) // 4 for m in messages),
        start_timestamp=ts,
        end_timestamp=ts + timedelta(minutes=5),
        turn_count=len(messages) // 2,
    )


@pytest.fixture
def ts() -> datetime:
    return datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def legal_segment(ts) -> TaggedSegment:
    messages = [
        Message(role="user", content="What's the deadline for the court filing in case 24-cv-1234?", timestamp=ts),
        Message(role="assistant", content="The filing deadline for case 24-cv-1234 is January 30th. The motion must be submitted to the court by 5pm.", timestamp=ts + timedelta(seconds=30)),
        Message(role="user", content="Has the attorney reviewed the settlement offer of $50,000?", timestamp=ts + timedelta(minutes=2)),
        Message(role="assistant", content="Yes, the attorney reviewed the settlement offer and recommends we counter at $75,000 based on precedent.", timestamp=ts + timedelta(minutes=2, seconds=30)),
    ]
    return _make_segment(messages, primary_tag="legal", tags=["legal", "court-filing"])


@pytest.fixture
def medical_segment(ts) -> TaggedSegment:
    base = ts + timedelta(minutes=10)
    messages = [
        Message(role="user", content="My blood glucose was 180 this morning. Should I adjust my insulin?", timestamp=base),
        Message(role="assistant", content="A reading of 180 is above target. Consider adjusting your insulin dosage. Check with your doctor about increasing by 1 unit.", timestamp=base + timedelta(seconds=30)),
        Message(role="user", content="The lab results from last week showed HbA1c of 7.2%.", timestamp=base + timedelta(minutes=2)),
        Message(role="assistant", content="An HbA1c of 7.2% indicates room for improvement. Schedule an appointment with your endocrinologist to discuss adjustments.", timestamp=base + timedelta(minutes=2, seconds=30)),
    ]
    return _make_segment(messages, primary_tag="medical", tags=["medical", "glucose"])


@pytest.mark.timeout(600)
class TestCompactorStructure:
    """Assert compaction result structure, not exact summary content."""

    def test_compact_legal_segment(
        self, ollama_compactor: DomainCompactor, legal_segment: TaggedSegment
    ):
        results = ollama_compactor.compact([legal_segment])
        assert len(results) == 1
        r = results[0]
        assert r.summary, "Expected non-empty summary"
        assert r.summary_tokens > 0
        assert r.original_tokens > 0

    def test_compact_medical_segment(
        self, ollama_compactor: DomainCompactor, medical_segment: TaggedSegment
    ):
        results = ollama_compactor.compact([medical_segment])
        assert len(results) == 1
        r = results[0]
        assert r.summary, "Expected non-empty summary"
        assert r.summary_tokens > 0

    def test_summary_preserves_some_details(
        self, ollama_compactor: DomainCompactor, legal_segment: TaggedSegment
    ):
        """Summary should mention at least one specific detail from the conversation."""
        results = ollama_compactor.compact([legal_segment])
        summary = results[0].summary.lower()
        details = ["24-cv-1234", "january 30", "50,000", "75,000", "attorney", "settlement"]
        found = [d for d in details if d in summary]
        assert found, f"Expected at least one detail in summary, found none. Summary: {summary}"

    def test_metadata_extracted(
        self, ollama_compactor: DomainCompactor, legal_segment: TaggedSegment
    ):
        results = ollama_compactor.compact([legal_segment])
        meta = results[0].metadata
        assert isinstance(meta.entities, list)
        assert isinstance(meta.key_decisions, list)

    def test_refined_tags_are_list(
        self, ollama_compactor: DomainCompactor, legal_segment: TaggedSegment
    ):
        results = ollama_compactor.compact([legal_segment])
        assert isinstance(results[0].tags, list)
        assert len(results[0].tags) > 0

    def test_compression_ratio_reasonable(
        self, ollama_compactor: DomainCompactor, legal_segment: TaggedSegment
    ):
        results = ollama_compactor.compact([legal_segment])
        ratio = results[0].compression_ratio
        assert 0.0 < ratio < 1.0, f"Expected compression ratio in (0, 1), got {ratio}"

    def test_compact_multiple_segments(
        self,
        ollama_compactor: DomainCompactor,
        legal_segment: TaggedSegment,
        medical_segment: TaggedSegment,
        ts: datetime,
    ):
        """Multiple segments should each get their own summary."""
        # Make a third segment
        code_messages = [
            Message(role="user", content="How do I add a REST endpoint in FastAPI?", timestamp=ts + timedelta(hours=1)),
            Message(role="assistant", content="Create a router with @router.get, define Pydantic models, and include the router in main.py.", timestamp=ts + timedelta(hours=1, seconds=30)),
        ]
        code_segment = _make_segment(code_messages, primary_tag="code", tags=["code", "fastapi"])

        results = ollama_compactor.compact([legal_segment, medical_segment, code_segment])
        assert len(results) == 3
        for r in results:
            assert r.summary, f"Expected non-empty summary for segment {r.segment_id}"

    def test_truncated_output_fallback(
        self, ollama_provider, legal_segment: TaggedSegment
    ):
        """Very small max_tokens should not cause an exception -- the compactor should handle it."""
        from virtual_context.types import CompactorConfig

        tiny_compactor = DomainCompactor(
            llm_provider=ollama_provider,
            config=CompactorConfig(
                summary_ratio=0.15,
                min_summary_tokens=10,
                max_summary_tokens=30,
            ),
            model_name="qwen3:4b-instruct-2507-fp16",
        )
        results = tiny_compactor.compact([legal_segment])
        assert len(results) == 1
        # Should have a summary even if truncated -- compactor has fallback logic
        assert results[0].summary is not None
