"""Tests for ContextAssembler (tag-based)."""

from datetime import datetime, timezone

import pytest

from virtual_context.core.assembler import ContextAssembler, format_tag_section
from virtual_context.types import (
    AssemblerConfig,
    Message,
    RetrievalResult,
    SegmentMetadata,
    StoredSummary,
    TagPromptRule,
)


@pytest.fixture
def assembler():
    return ContextAssembler(
        config=AssemblerConfig(
            core_context_max_tokens=1000,
            tag_context_max_tokens=2000,
        )
    )


@pytest.fixture
def retrieval_result():
    now = datetime.now(timezone.utc)
    return RetrievalResult(
        tags_matched=["legal"],
        summaries=[
            StoredSummary(
                ref="ref-1",
                primary_tag="legal",
                tags=["legal", "court"],
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
    assert "legal" in result.tag_sections
    assert len(result.conversation_history) == 2


def test_assemble_xml_tags(assembler, retrieval_result):
    result = assembler.assemble(
        core_context="",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
    )
    section = result.tag_sections.get("legal", "")
    assert '<virtual-context tags="court, legal"' in section
    assert "last_updated=" not in section
    assert "</virtual-context>" in section


def test_format_tag_section_renders_code_refs():
    now = datetime.now(timezone.utc)
    section = format_tag_section(
        "backend",
        [
            StoredSummary(
                ref="ref-1",
                primary_tag="backend",
                tags=["backend"],
                summary="Request cache boundary moved ahead of the mutable reminder.",
                summary_tokens=18,
                full_tokens=50,
                metadata=SegmentMetadata(
                    code_refs=[
                        {"file": "virtual_context/proxy/formats.py", "line": 1312, "symbol": "inject_context"},
                        {"file": "virtual_context/core/provider_adapters.py", "symbol": "AnthropicAdapter"},
                    ],
                ),
                created_at=now,
                start_timestamp=now,
                end_timestamp=now,
            ),
        ],
    )

    assert "[refs:" in section
    assert "virtual_context/proxy/formats.py:1312 inject_context" in section
    assert "virtual_context/core/provider_adapters.py AnthropicAdapter" in section


def test_trim_conversation(assembler):
    messages = [
        Message(role="user", content="x" * 400),
        Message(role="assistant", content="y" * 400),
        Message(role="user", content="z" * 400),
    ]
    trimmed = assembler._trim_conversation(messages, budget=250)
    assert len(trimmed) < len(messages)


def test_assemble_empty_retrieval(assembler):
    result = assembler.assemble(
        core_context="core",
        retrieval_result=RetrievalResult(),
        conversation_history=[Message(role="user", content="hello")],
        token_budget=10000,
    )
    assert result.tag_sections == {}
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


def test_tag_priority_from_rules():
    """Tags with higher priority rules should appear first."""
    rules = [
        TagPromptRule(match="architecture*", priority=10),
        TagPromptRule(match="debug*", priority=7),
        TagPromptRule(match="*", priority=5),
    ]
    assembler = ContextAssembler(
        config=AssemblerConfig(tag_context_max_tokens=5000),
        tag_rules=rules,
    )
    assert assembler._tag_priority("architecture-decisions") == 10
    assert assembler._tag_priority("debugging") == 7
    assert assembler._tag_priority("random-tag") == 5


def test_budget_breakdown(assembler, retrieval_result):
    result = assembler.assemble(
        core_context="core context here",
        retrieval_result=retrieval_result,
        conversation_history=[Message(role="user", content="hello")],
        token_budget=10000,
    )
    assert "core" in result.budget_breakdown
    assert "tags" in result.budget_breakdown
    assert "conversation" in result.budget_breakdown


def test_context_hint_injected(assembler, retrieval_result):
    """Context hint appears between core context and tag sections."""
    hint = "<context-topics>\n- recipes (5 turns): recipe app...\n</context-topics>"
    result = assembler.assemble(
        core_context="# Core\nIdentity",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
        context_hint=hint,
    )
    assert "context-topics" in result.prepend_text
    # Hint appears after core, before tag sections
    core_pos = result.prepend_text.index("Core")
    hint_pos = result.prepend_text.index("context-topics")
    tag_pos = result.prepend_text.index("virtual-context")
    assert core_pos < hint_pos < tag_pos


def test_context_hint_empty(assembler, retrieval_result):
    """No hint block when context_hint is empty."""
    result = assembler.assemble(
        core_context="core",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
        context_hint="",
    )
    assert "context-topics" not in result.prepend_text


def test_context_hint_in_budget(assembler, retrieval_result):
    """Hint tokens counted in budget breakdown."""
    hint = "<context-topics>\nSome topics here\n</context-topics>"
    result = assembler.assemble(
        core_context="",
        retrieval_result=retrieval_result,
        conversation_history=[],
        token_budget=10000,
        context_hint=hint,
    )
    assert result.budget_breakdown["context_hint"] > 0


def test_budget_cap_truncates_least_relevant_tags():
    """When assembled prepend_text exceeds token_budget, drop least-relevant tags."""
    now = datetime.now(timezone.utc)
    assembler = ContextAssembler(
        config=AssemblerConfig(
            core_context_max_tokens=1000,
            tag_context_max_tokens=5000,
        )
    )
    summaries = []
    for i, tag in enumerate(["high-priority", "medium-priority", "low-priority"]):
        summaries.append(StoredSummary(
            ref=f"ref-{i}",
            primary_tag=tag,
            tags=[tag],
            summary="x" * 400,  # ~100 tokens each
            summary_tokens=100,
            full_tokens=400,
            metadata=SegmentMetadata(),
            created_at=now,
            start_timestamp=now,
            end_timestamp=now,
        ))
    rr = RetrievalResult(
        tags_matched=["high-priority", "medium-priority", "low-priority"],
        summaries=summaries,
        total_tokens=300,
    )
    result = assembler.assemble(
        core_context="core",
        retrieval_result=rr,
        conversation_history=[],
        token_budget=250,
    )
    prepend_tokens = assembler.token_counter(result.prepend_text)
    assert prepend_tokens <= 250


def test_budget_cap_logs_error(caplog):
    """Exceeding token_budget logs an ERROR with config guidance."""
    import logging
    now = datetime.now(timezone.utc)
    assembler = ContextAssembler(
        config=AssemblerConfig(
            core_context_max_tokens=1000,
            tag_context_max_tokens=5000,
        )
    )
    summaries = [StoredSummary(
        ref="ref-0",
        primary_tag="big-tag",
        tags=["big-tag"],
        summary="x" * 400,
        summary_tokens=100,
        full_tokens=400,
        metadata=SegmentMetadata(),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    )]
    rr = RetrievalResult(
        tags_matched=["big-tag"],
        summaries=summaries,
        total_tokens=100,
    )
    with caplog.at_level(logging.ERROR):
        assembler.assemble(
            core_context="core",
            retrieval_result=rr,
            conversation_history=[],
            token_budget=5,
        )
    assert any("exceeds token_budget" in r.message for r in caplog.records)
    assert any("tag_context_max_tokens" in r.message for r in caplog.records)
