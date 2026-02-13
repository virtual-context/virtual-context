"""Tests for ContextAssembler."""

from datetime import datetime, timezone

import pytest

from virtual_context.core.assembler import ContextAssembler
from virtual_context.types import (
    AssemblerConfig,
    Message,
    RetrievalResult,
    SegmentMetadata,
    StoredSummary,
)


@pytest.fixture
def assembler():
    return ContextAssembler(
        config=AssemblerConfig(
            core_context_max_tokens=1000,
            domain_context_max_tokens=2000,
        )
    )


@pytest.fixture
def retrieval_result():
    now = datetime.now(timezone.utc)
    return RetrievalResult(
        domains_matched=["legal"],
        summaries=[
            StoredSummary(
                ref="ref-1",
                domain="legal",
                summary="Case 24-cv-1234 discussed. Filing due Jan 30.",
                summary_tokens=20,
                full_tokens=100,
                metadata=SegmentMetadata(entities=["Case 24-cv-1234"]),
                created_at=now,
                start_timestamp=now,
                end_timestamp=now,
            ),
        ],
        total_tokens=20,
    )


def test_assemble_basic(assembler, retrieval_result):
    history = [
        Message(role="user", content="What about the case?"),
        Message(role="assistant", content="Let me check."),
    ]
    result = assembler.assemble(
        core_context="# IDENTITY\nYou are a helpful assistant.",
        retrieval_result=retrieval_result,
        conversation_history=history,
        token_budget=10000,
    )
    assert result.total_tokens > 0
    assert "legal" in result.domain_sections
    assert len(result.conversation_history) == 2


def test_assemble_xml_tags(assembler, retrieval_result):
    result = assembler.assemble(
        core_context="",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
    )
    section = result.domain_sections.get("legal", "")
    assert '<virtual-context domain="legal"' in section
    assert "</virtual-context>" in section


def test_trim_conversation(assembler):
    # Create messages that exceed budget
    messages = [
        Message(role="user", content="x" * 400),
        Message(role="assistant", content="y" * 400),
        Message(role="user", content="z" * 400),
    ]
    trimmed = assembler._trim_conversation(messages, budget=250)
    # Should keep most recent that fits
    assert len(trimmed) < len(messages)


def test_assemble_empty_retrieval(assembler):
    result = assembler.assemble(
        core_context="core",
        retrieval_result=RetrievalResult(),
        conversation_history=[Message(role="user", content="hello")],
        token_budget=10000,
    )
    assert result.domain_sections == {}
    assert len(result.conversation_history) == 1


def test_prepend_text(assembler, retrieval_result):
    result = assembler.assemble(
        core_context="# Core\nIdentity file",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
    )
    assert "Core" in result.prepend_text
    assert "virtual-context" in result.prepend_text
