"""Integration test: 50-turn multi-domain conversation."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.conftest import MockLLMProvider
from virtual_context.classifiers.base import ClassifierPipeline
from virtual_context.classifiers.keyword import KeywordClassifier
from virtual_context.config import load_config
from virtual_context.core.assembler import ContextAssembler
from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.monitor import ContextMonitor
from virtual_context.core.retriever import ContextRetriever
from virtual_context.core.segmenter import TopicSegmenter
from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.types import (
    CompactorConfig,
    Message,
    RetrieverConfig,
    SegmenterConfig,
    StoredSegment,
)


def make_conversation(n_turns: int = 50) -> list[Message]:
    """Generate a multi-domain conversation.

    Pattern: 10 legal, 10 medical, 10 legal, 10 general, 10 medical
    """
    messages = []
    base = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    turn = 0

    legal_pairs = [
        ("What's the status of case 24-cv-1234?", "The court hearing is scheduled for February 5th. The attorney needs to file the motion by January 30."),
        ("Did the judge approve the discovery motion?", "Yes, Judge Martinez approved the discovery motion. Depositions can proceed."),
        ("What's the settlement offer from opposing counsel?", "Opposing attorney proposed $45,000. I recommend countering at $65,000."),
        ("When is the next court appearance?", "The next court appearance is February 12th at 9am, Courtroom 3B."),
        ("Has the legal brief been reviewed?", "The attorney completed the brief review. Two citations need updating per NJSA standards."),
    ]

    medical_pairs = [
        ("My blood glucose was 195 this morning.", "That's above target. Have you adjusted your insulin? Consider increasing your evening dose by 1 unit."),
        ("The doctor ordered new lab work.", "Lab results should be ready by Friday. We'll need to check your A1C and liver function."),
        ("Should I take my medication with food?", "Yes, your insulin should be taken 15 minutes before meals for optimal glucose control."),
        ("My blood pressure was 145/92 today.", "That reading is elevated. The doctor may want to adjust your medication at your next appointment."),
        ("When is my next medical appointment?", "Your appointment with Dr. Chen is February 8th at 2pm. Bring your glucose log."),
    ]

    general_pairs = [
        ("What's the weather like today?", "I don't have real-time weather data, but I can help you find a forecast."),
        ("Can you recommend a good restaurant?", "What cuisine are you in the mood for? I can suggest options."),
        ("What time is it in Tokyo?", "Tokyo is typically 14 hours ahead of Eastern time."),
        ("Tell me a joke.", "Why did the scarecrow win an award? He was outstanding in his field."),
        ("What's a good book to read?", "I'd recommend 'Thinking, Fast and Slow' by Daniel Kahneman."),
    ]

    # 10 legal turns (5 pairs)
    for q, a in legal_pairs:
        messages.append(Message(role="user", content=q, timestamp=base + timedelta(minutes=turn)))
        turn += 1
        messages.append(Message(role="assistant", content=a, timestamp=base + timedelta(minutes=turn)))
        turn += 1

    # 10 medical turns
    for q, a in medical_pairs:
        messages.append(Message(role="user", content=q, timestamp=base + timedelta(minutes=turn)))
        turn += 1
        messages.append(Message(role="assistant", content=a, timestamp=base + timedelta(minutes=turn)))
        turn += 1

    # 10 legal turns (repeat with slight variation)
    for q, a in legal_pairs:
        messages.append(Message(role="user", content=f"Follow-up: {q}", timestamp=base + timedelta(minutes=turn)))
        turn += 1
        messages.append(Message(role="assistant", content=f"Update: {a}", timestamp=base + timedelta(minutes=turn)))
        turn += 1

    # 10 general turns
    for q, a in general_pairs:
        messages.append(Message(role="user", content=q, timestamp=base + timedelta(minutes=turn)))
        turn += 1
        messages.append(Message(role="assistant", content=a, timestamp=base + timedelta(minutes=turn)))
        turn += 1

    # 10 medical turns (repeat)
    for q, a in medical_pairs:
        messages.append(Message(role="user", content=f"Update: {q}", timestamp=base + timedelta(minutes=turn)))
        turn += 1
        messages.append(Message(role="assistant", content=f"Follow-up: {a}", timestamp=base + timedelta(minutes=turn)))
        turn += 1

    return messages


@pytest.mark.asyncio
async def test_full_pipeline():
    """Integration test: segment, compact, store, retrieve across multiple domains."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"

        # Setup
        domains_config = {
            "legal": {
                "description": "Legal matters",
                "keywords": ["court", "filing", "motion", "attorney", "legal", "case", "judge", "settlement", "deposition", "brief"],
                "patterns": [r"\b\d{2}-cv-\d+"],
                "priority": 9,
                "retrieval_limit": 5,
            },
            "medical": {
                "description": "Medical matters",
                "keywords": ["insulin", "medication", "doctor", "lab", "blood", "glucose", "appointment", "A1C"],
                "priority": 8,
                "retrieval_limit": 3,
            },
        }

        config = load_config(config_dict={
            "context_window": 5000,  # small for testing
            "domains": domains_config,
            "storage": {"filesystem": {"root": str(store_path)}},
            "compaction": {
                "soft_threshold": 0.5,
                "hard_threshold": 0.7,
                "protected_recent_turns": 2,
                "summary_ratio": 0.15,
                "min_summary_tokens": 50,
            },
        })

        # Initialize components
        classifier = ClassifierPipeline([KeywordClassifier()], min_confidence=0.3)
        await classifier.initialize(list(config.domains.values()))

        store = FilesystemStore(store_path)
        segmenter = TopicSegmenter(
            classifier_pipeline=classifier,
            config=config.segmenter,
            domains=list(config.domains.values()),
        )

        mock_llm = MockLLMProvider()
        compactor = DomainCompactor(
            llm_provider=mock_llm,
            config=config.compactor,
            model_name="test-model",
        )

        monitor = ContextMonitor(config=config.monitor)

        retriever_config = RetrieverConfig(
            domains=list(config.domains.values()),
            domain_context_max_tokens=2000,
        )
        retriever = ContextRetriever(
            classifier_pipeline=classifier,
            store=store,
            config=retriever_config,
        )

        # Generate conversation
        messages = make_conversation(50)
        assert len(messages) == 50  # 25 pairs = 50 messages

        # 1. Segment the conversation
        segments = await segmenter.segment(messages)
        assert len(segments) >= 3, f"Expected at least 3 segments, got {len(segments)}"

        # Verify domain separation
        domain_set = {s.domain for s in segments}
        assert "legal" in domain_set, f"Expected legal domain, got {domain_set}"
        assert "medical" in domain_set, f"Expected medical domain, got {domain_set}"

        # 2. Compact each segment
        results = await compactor.compact(segments)
        assert len(results) == len(segments)

        # 3. Store results
        for result in results:
            stored = StoredSegment(
                ref=result.segment_id,
                session_id="test",
                domain=result.domain,
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
            await store.store_segment(stored)

        # 4. Verify storage
        domain_stats = await store.list_domains()
        assert len(domain_stats) >= 2

        legal_summaries = await store.get_summaries(domain="legal")
        assert len(legal_summaries) >= 1

        medical_summaries = await store.get_summaries(domain="medical")
        assert len(medical_summaries) >= 1

        # 5. Retrieve for legal query
        legal_result = await retriever.retrieve("What's the court filing deadline?")
        assert "legal" in legal_result.domains_matched
        assert len(legal_result.summaries) > 0
        assert legal_result.summaries[0].domain == "legal"

        # 6. Retrieve for medical query - should NOT return legal
        medical_result = await retriever.retrieve("How is my insulin dosage?")
        assert "medical" in medical_result.domains_matched
        # Medical summaries should be returned, not legal
        for s in medical_result.summaries:
            assert s.domain == "medical"

        # 7. Monitor check
        snapshot = monitor.build_snapshot(messages)
        assert snapshot.total_tokens > 0
