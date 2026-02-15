"""Integration tests for the full segment->compact->store->retrieve pipeline with real Ollama."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from virtual_context.config import load_config
from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.segmenter import TopicSegmenter
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.engine import VirtualContextEngine
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CompactorConfig,
    Message,
    SegmenterConfig,
    StoredSegment,
    TagGeneratorConfig,
)


def _build_conversation(ts: datetime) -> list[Message]:
    """Build a 20-message multi-topic conversation."""
    messages = []
    t = ts

    # Legal block (6 messages)
    exchanges = [
        ("What's the filing deadline for case 24-cv-1234?",
         "The deadline is January 30th. The motion must be submitted by 5pm."),
        ("Has the attorney reviewed the settlement offer?",
         "Yes, the attorney recommends countering at $75,000 based on precedent from Smith v. Jones."),
        ("What about the judge's ruling on the pretrial motion?",
         "Judge Williams denied the motion to dismiss. We proceed to discovery."),
    ]
    for q, a in exchanges:
        messages.append(Message(role="user", content=q, timestamp=t))
        t += timedelta(seconds=30)
        messages.append(Message(role="assistant", content=a, timestamp=t))
        t += timedelta(minutes=2)

    # Medical block (6 messages)
    exchanges = [
        ("My blood glucose was 180 this morning. Should I adjust insulin?",
         "A reading of 180 is above target. Consider increasing by 1 unit."),
        ("Lab results showed HbA1c of 7.2%. Is that concerning?",
         "An HbA1c of 7.2% suggests room for improvement. Schedule an endocrinologist visit."),
        ("Should I change my medication from metformin?",
         "Don't change medications without consulting your doctor. Metformin is a good baseline."),
    ]
    for q, a in exchanges:
        messages.append(Message(role="user", content=q, timestamp=t))
        t += timedelta(seconds=30)
        messages.append(Message(role="assistant", content=a, timestamp=t))
        t += timedelta(minutes=2)

    # Code/tech block (4 messages)
    exchanges = [
        ("How do I add a REST endpoint to our FastAPI app?",
         "Create a router in app/routers/, define path operations and Pydantic models."),
        ("What about adding authentication middleware?",
         "Use FastAPI's Depends with a JWT bearer scheme. Add it to your router dependencies."),
    ]
    for q, a in exchanges:
        messages.append(Message(role="user", content=q, timestamp=t))
        t += timedelta(seconds=30)
        messages.append(Message(role="assistant", content=a, timestamp=t))
        t += timedelta(minutes=2)

    # General block (4 messages)
    exchanges = [
        ("What's the weather forecast for this weekend?",
         "Expect partly cloudy skies with temperatures around 55F on Saturday."),
        ("Any restaurant recommendations downtown?",
         "Try the new Italian place on Main Street, great pasta and reasonable prices."),
    ]
    for q, a in exchanges:
        messages.append(Message(role="user", content=q, timestamp=t))
        t += timedelta(seconds=30)
        messages.append(Message(role="assistant", content=a, timestamp=t))
        t += timedelta(minutes=2)

    return messages


@pytest.mark.timeout(600)
class TestFullPipeline:
    """End-to-end pipeline tests with real Ollama."""

    def test_segment_compact_store_retrieve(self, ollama_provider):
        """Full pipeline: segment -> compact -> store in SQLite -> retrieve by tag."""
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        messages = _build_conversation(ts)

        tag_config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1, max_tokens=8192)
        tag_gen = LLMTagGenerator(llm_provider=ollama_provider, config=tag_config)

        # Segment
        segmenter = TopicSegmenter(
            tag_generator=tag_gen,
            config=SegmenterConfig(),
        )
        segments = segmenter.segment(messages)
        assert len(segments) >= 2, f"Expected at least 2 segments, got {len(segments)}"

        # Compact
        compactor = DomainCompactor(
            llm_provider=ollama_provider,
            config=CompactorConfig(
                summary_ratio=0.15,
                min_summary_tokens=50,
                max_summary_tokens=500,
                llm_token_overhead=8000,
            ),
            model_name="qwen3:4b-instruct-2507-fp16",
        )
        results = compactor.compact(segments)
        assert len(results) == len(segments)

        # Store
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            store = SQLiteStore(db_path=db_path)

            for result in results:
                stored = StoredSegment(
                    ref=result.segment_id,
                    session_id="test-session",
                    primary_tag=result.primary_tag,
                    tags=result.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model="qwen3:4b-instruct-2507-fp16",
                    compression_ratio=result.compression_ratio,
                )
                store.store_segment(stored)

            # Verify distinct tags
            all_tags_stats = store.get_all_tags()
            all_tags = [s.tag for s in all_tags_stats]
            assert len(all_tags) >= 2, f"Expected at least 2 distinct tags, got {all_tags}"

            # Retrieve by tags
            summaries = store.get_summaries_by_tags(tags=all_tags[:3], min_overlap=1)
            assert len(summaries) > 0, "Expected at least one retrieved summary"

            store.close()

    def test_engine_on_turn_complete(self, ollama_provider):
        """Full engine with Ollama: on_turn_complete should produce a CompactionReport."""
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        messages = _build_conversation(ts)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "engine_test.db"

            config = load_config(config_dict={
                "context_window": 2000,  # Very small to trigger compaction
                "storage_root": tmp,
                "token_counter": "estimate",
                "tag_generator": {
                    "type": "llm",
                    "provider": "ollama",
                    "model": "qwen3:4b-instruct-2507-fp16",
                    "max_tokens": 8192,
                },
                "compaction": {
                    "soft_threshold": 0.10,
                    "hard_threshold": 0.20,
                    "protected_recent_turns": 2,
                    "summary_ratio": 0.15,
                    "min_summary_tokens": 50,
                    "max_summary_tokens": 500,
                    "llm_token_overhead": 8000,
                },
                "summarization": {
                    "provider": "ollama",
                    "model": "qwen3:4b-instruct-2507-fp16",
                    "temperature": 0.3,
                },
                "storage": {
                    "backend": "sqlite",
                    "sqlite": {"path": str(db_path)},
                },
                "providers": {
                    "ollama": {
                        "type": "generic_openai",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen3:4b-instruct-2507-fp16",
                    },
                },
            })

            engine = VirtualContextEngine(config=config)
            report = engine.on_turn_complete(messages)

            assert report is not None, "Expected a CompactionReport (thresholds should trigger)"
            assert report.segments_compacted > 0, "Expected at least one segment compacted"
            assert report.tokens_freed > 0

    def test_engine_on_message_inbound(self, ollama_provider):
        """After storing segments, an inbound query should retrieve relevant context."""
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        messages = _build_conversation(ts)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "inbound_test.db"

            config = load_config(config_dict={
                "context_window": 2000,
                "storage_root": tmp,
                "token_counter": "estimate",
                "tag_generator": {
                    "type": "llm",
                    "provider": "ollama",
                    "model": "qwen3:4b-instruct-2507-fp16",
                    "max_tokens": 8192,
                },
                "compaction": {
                    "soft_threshold": 0.10,
                    "hard_threshold": 0.20,
                    "protected_recent_turns": 2,
                    "summary_ratio": 0.15,
                    "min_summary_tokens": 50,
                    "max_summary_tokens": 500,
                    "llm_token_overhead": 8000,
                },
                "summarization": {
                    "provider": "ollama",
                    "model": "qwen3:4b-instruct-2507-fp16",
                    "temperature": 0.3,
                },
                "storage": {
                    "backend": "sqlite",
                    "sqlite": {"path": str(db_path)},
                },
                "providers": {
                    "ollama": {
                        "type": "generic_openai",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen3:4b-instruct-2507-fp16",
                    },
                },
                "retrieval": {
                    "skip_active_tags": False,
                },
            })

            engine = VirtualContextEngine(config=config)

            # First compact to populate the store
            report = engine.on_turn_complete(messages)
            assert report is not None and report.segments_compacted > 0

            # Now query -- use a legal-themed message to trigger retrieval
            assembled = engine.on_message_inbound(
                message="What was the court filing deadline?",
                conversation_history=messages[-4:],  # only recent messages
            )

            assert assembled is not None
            # The assembled context should have some content
            assert assembled.total_tokens > 0

    def test_round_trip_consistency(self, ollama_provider):
        """Compact -> store -> retrieve -> stored summary text matches."""
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        segment_messages = [
            Message(role="user", content="What is the filing deadline for case 24-cv-1234?", timestamp=ts),
            Message(role="assistant", content="The deadline is January 30th at 5pm.", timestamp=ts + timedelta(seconds=30)),
        ]

        tag_config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1, max_tokens=8192)
        tag_gen = LLMTagGenerator(llm_provider=ollama_provider, config=tag_config)

        segmenter = TopicSegmenter(
            tag_generator=tag_gen,
            config=SegmenterConfig(),
        )
        segments = segmenter.segment(segment_messages)
        assert len(segments) >= 1

        compactor = DomainCompactor(
            llm_provider=ollama_provider,
            config=CompactorConfig(
                summary_ratio=0.15,
                min_summary_tokens=50,
                max_summary_tokens=500,
                llm_token_overhead=8000,
            ),
            model_name="qwen3:4b-instruct-2507-fp16",
        )
        results = compactor.compact(segments)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "roundtrip.db"
            store = SQLiteStore(db_path=db_path)

            for result in results:
                stored = StoredSegment(
                    ref=result.segment_id,
                    session_id="test-session",
                    primary_tag=result.primary_tag,
                    tags=result.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model="qwen3:4b-instruct-2507-fp16",
                    compression_ratio=result.compression_ratio,
                )
                store.store_segment(stored)

            # Retrieve and verify round-trip
            for result in results:
                retrieved = store.get_segment(result.segment_id)
                assert retrieved is not None, f"Segment {result.segment_id} not found in store"
                assert retrieved.summary == result.summary, (
                    f"Summary mismatch:\n  stored: {result.summary!r}\n  retrieved: {retrieved.summary!r}"
                )
                assert retrieved.primary_tag == result.primary_tag

            store.close()
