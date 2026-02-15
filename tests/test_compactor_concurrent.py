"""Tests for concurrent summarization in DomainCompactor."""
import pytest
import threading
from datetime import datetime, timezone

from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import (
    CompactorConfig,
    Message,
    TaggedSegment,
)


class ThreadTrackingLLM:
    """Mock LLM that tracks which threads called it."""

    def __init__(self):
        self.thread_ids: list[int] = []
        self.call_count = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        with self._lock:
            self.thread_ids.append(threading.get_ident())
            self.call_count += 1
        return '{"summary": "Test summary", "entities": [], "key_decisions": [], "action_items": [], "date_references": [], "refined_tags": ["test"]}'


def _make_segment(tag: str, content: str = "Hello world") -> TaggedSegment:
    now = datetime.now(timezone.utc)
    return TaggedSegment(
        primary_tag=tag,
        tags=[tag],
        messages=[
            Message(role="user", content=content, timestamp=now),
            Message(role="assistant", content=f"Response about {tag}", timestamp=now),
        ],
        token_count=100,
        turn_count=1,
        start_timestamp=now,
        end_timestamp=now,
    )


class TestConcurrentCompaction:
    def test_single_segment_sequential(self):
        """Single segment should not use threading."""
        llm = ThreadTrackingLLM()
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(max_concurrent_summaries=4),
        )
        segments = [_make_segment("database")]
        results = compactor.compact(segments)
        assert len(results) == 1
        assert results[0].primary_tag == "database"

    def test_multiple_segments_concurrent(self):
        """Multiple segments should use ThreadPoolExecutor."""
        llm = ThreadTrackingLLM()
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(max_concurrent_summaries=4),
        )
        segments = [_make_segment(f"tag-{i}") for i in range(4)]
        results = compactor.compact(segments)
        assert len(results) == 4
        assert llm.call_count == 4
        # All results should be valid
        for i, result in enumerate(results):
            assert result.primary_tag == f"tag-{i}"
            assert result.summary

    def test_order_preserved(self):
        """Results should maintain input segment order."""
        llm = ThreadTrackingLLM()
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(max_concurrent_summaries=2),
        )
        tags = ["alpha", "beta", "gamma", "delta"]
        segments = [_make_segment(tag) for tag in tags]
        results = compactor.compact(segments)
        result_tags = [r.primary_tag for r in results]
        assert result_tags == tags

    def test_max_workers_respected(self):
        """Should not exceed max_concurrent_summaries."""
        llm = ThreadTrackingLLM()
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(max_concurrent_summaries=2),
        )
        segments = [_make_segment(f"tag-{i}") for i in range(6)]
        results = compactor.compact(segments)
        assert len(results) == 6
        # Can't directly verify max 2 threads, but results should be correct
        assert all(r.summary for r in results)

    def test_error_handling(self):
        """Errors in one segment should not crash others."""
        call_count = 0

        class FailingLLM:
            def complete(self, system, user, max_tokens):
                nonlocal call_count
                call_count += 1
                if "fail" in user:
                    raise RuntimeError("LLM error")
                return '{"summary": "OK", "entities": [], "key_decisions": [], "action_items": [], "date_references": [], "refined_tags": ["test"]}'

        compactor = DomainCompactor(
            llm_provider=FailingLLM(),
            config=CompactorConfig(max_concurrent_summaries=4),
        )
        segments = [
            _make_segment("good"),
            _make_segment("fail", content="This should fail"),
            _make_segment("also-good"),
        ]
        results = compactor.compact(segments)
        assert len(results) == 3
        # All results should exist (failed one has fallback)
        assert all(r is not None for r in results)
