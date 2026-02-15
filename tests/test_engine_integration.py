"""Integration test: multi-tag conversation through full pipeline."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.conftest import MockLLMProvider, MockTagGenerator
from virtual_context.config import load_config
from virtual_context.core.assembler import ContextAssembler
from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.monitor import ContextMonitor
from virtual_context.core.retriever import ContextRetriever
from virtual_context.core.segmenter import TopicSegmenter
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CompactorConfig,
    Message,
    RetrieverConfig,
    SegmenterConfig,
    StoredSegment,
    StrategyConfig,
    TagResult,
)


def make_conversation(n_turns: int = 50) -> list[Message]:
    """Generate a multi-tag conversation.

    Pattern: 10 legal, 10 medical, 10 legal, 10 general, 10 medical
    """
    messages = []
    base = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    turn = 0

    legal_pairs = [
        ("What's the status of case 24-cv-1234?", "The court hearing is scheduled for February 5th."),
        ("Did the judge approve the discovery motion?", "Yes, Judge Martinez approved it."),
        ("What's the settlement offer?", "Opposing attorney proposed $45,000."),
        ("When is the next court appearance?", "February 12th at 9am."),
        ("Has the legal brief been reviewed?", "Two citations need updating per NJSA."),
    ]

    medical_pairs = [
        ("My blood glucose was 195 this morning.", "That's above target. Adjust insulin."),
        ("The doctor ordered new lab work.", "Lab results should be ready Friday."),
        ("Should I take medication with food?", "Yes, 15 minutes before meals."),
        ("My blood pressure was 145/92.", "That's elevated, may need adjustment."),
        ("When is my next appointment?", "February 8th at 2pm with Dr. Chen."),
    ]

    general_pairs = [
        ("What's the weather today?", "I can help you find a forecast."),
        ("Recommend a restaurant?", "What cuisine do you prefer?"),
        ("What time is it in Tokyo?", "14 hours ahead of Eastern."),
        ("Tell me a joke.", "Why did the scarecrow win? Outstanding in his field."),
        ("Good book to read?", "Thinking, Fast and Slow by Kahneman."),
    ]

    for pairs in [legal_pairs, medical_pairs, legal_pairs, general_pairs, medical_pairs]:
        for q, a in pairs:
            messages.append(Message(role="user", content=q, timestamp=base + timedelta(minutes=turn)))
            turn += 1
            messages.append(Message(role="assistant", content=a, timestamp=base + timedelta(minutes=turn)))
            turn += 1

    return messages


def test_full_pipeline():
    """Integration test: segment, compact, store, retrieve across multiple tags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "store.db"

        # Setup tag generator with deterministic behavior
        tag_gen = MockTagGenerator(default_tag="_general")
        tag_gen.set_override("court", TagResult(tags=["legal", "court"], primary="legal", source="mock"))
        tag_gen.set_override("judge", TagResult(tags=["legal", "court"], primary="legal", source="mock"))
        tag_gen.set_override("attorney", TagResult(tags=["legal"], primary="legal", source="mock"))
        tag_gen.set_override("settlement", TagResult(tags=["legal"], primary="legal", source="mock"))
        tag_gen.set_override("brief", TagResult(tags=["legal"], primary="legal", source="mock"))
        tag_gen.set_override("glucose", TagResult(tags=["medical", "health"], primary="medical", source="mock"))
        tag_gen.set_override("insulin", TagResult(tags=["medical", "health"], primary="medical", source="mock"))
        tag_gen.set_override("doctor", TagResult(tags=["medical"], primary="medical", source="mock"))
        tag_gen.set_override("blood", TagResult(tags=["medical", "health"], primary="medical", source="mock"))
        tag_gen.set_override("medication", TagResult(tags=["medical"], primary="medical", source="mock"))
        tag_gen.set_override("appointment", TagResult(tags=["medical"], primary="medical", source="mock"))

        store = SQLiteStore(db_path=db_path)
        segmenter = TopicSegmenter(
            tag_generator=tag_gen,
            config=SegmenterConfig(),
        )

        # Use a mock that echoes back refined_tags based on content
        class EchoTagsLLM:
            def complete(self, system: str, user: str, max_tokens: int) -> str:
                import json
                # Determine tags from conversation content
                tags = []
                if any(w in user.lower() for w in ["court", "judge", "attorney", "filing"]):
                    tags.append("legal")
                if any(w in user.lower() for w in ["glucose", "insulin", "doctor", "blood"]):
                    tags.append("medical")
                if not tags:
                    tags = ["_general"]
                return json.dumps({
                    "summary": "Test summary",
                    "entities": ["entity1"],
                    "key_decisions": ["decision1"],
                    "action_items": [],
                    "date_references": [],
                    "refined_tags": tags,
                })

        compactor = DomainCompactor(
            llm_provider=EchoTagsLLM(),
            config=CompactorConfig(
                summary_ratio=0.15,
                min_summary_tokens=50,
            ),
            model_name="test-model",
        )

        monitor = ContextMonitor(config=load_config(config_dict={
            "context_window": 5000,
            "compaction": {"soft_threshold": 0.5, "hard_threshold": 0.7},
        }).monitor)

        retriever = ContextRetriever(
            tag_generator=tag_gen,
            store=store,
            config=RetrieverConfig(
                tag_context_max_tokens=2000,
                strategy_configs={"default": StrategyConfig(min_overlap=1, max_results=10)},
            ),
        )

        # Generate conversation
        messages = make_conversation(50)
        assert len(messages) == 50

        # 1. Segment the conversation
        segments = segmenter.segment(messages)
        assert len(segments) >= 3, f"Expected at least 3 segments, got {len(segments)}"

        # Verify tag separation
        tag_set = {s.primary_tag for s in segments}
        assert "legal" in tag_set, f"Expected legal tag, got {tag_set}"
        assert "medical" in tag_set, f"Expected medical tag, got {tag_set}"

        # 2. Compact each segment
        results = compactor.compact(segments)
        assert len(results) == len(segments)

        # 3. Store results
        for result in results:
            stored = StoredSegment(
                ref=result.segment_id,
                session_id="test",
                primary_tag=result.primary_tag,
                tags=result.tags,
                summary=result.summary,
                summary_tokens=result.summary_tokens,
                full_text=result.full_text,
                full_tokens=result.original_tokens,
                messages=result.messages,
                metadata=result.metadata,
                compaction_model="test-model",
                compression_ratio=result.compression_ratio,
                start_timestamp=result.timestamp,
                end_timestamp=result.timestamp,
            )
            store.store_segment(stored)

        # 4. Verify storage
        tag_stats = store.get_all_tags()
        assert len(tag_stats) >= 2

        legal_summaries = store.get_summaries_by_tags(tags=["legal"])
        assert len(legal_summaries) >= 1

        medical_summaries = store.get_summaries_by_tags(tags=["medical"])
        assert len(medical_summaries) >= 1

        # 5. Retrieve for legal query
        legal_result = retriever.retrieve("What's the court filing deadline?")
        assert "legal" in legal_result.tags_matched
        assert len(legal_result.summaries) > 0

        # 6. Retrieve for medical query
        medical_result = retriever.retrieve("How is my insulin dosage?")
        assert "medical" in medical_result.tags_matched

        # 7. Monitor check
        snapshot = monitor.build_snapshot(messages)
        assert snapshot.total_tokens > 0

        store.close()
