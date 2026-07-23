"""The all-topics coverage line must obey the configured hard budget."""

from __future__ import annotations

from virtual_context.core.hint_builder import (
    build_autonomous_hint,
    build_default_hint,
    build_supervised_hint,
)
from virtual_context.types import TagSummary


def _counter(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _summaries(count: int = 1800) -> list[TagSummary]:
    return [
        TagSummary(
            tag=f"very-long-production-topic-name-{index:04d}",
            summary="A detailed production conversation topic.",
            description="A long description that should be dropped before coverage names.",
            summary_tokens=10,
            source_turn_numbers=[index],
        )
        for index in range(count)
    ]


def test_supervised_large_all_topics_line_is_strictly_bounded() -> None:
    hint = build_supervised_hint(
        _summaries(), {}, 8000, _counter,
    )
    assert _counter(hint) <= 8000
    assert "topic names omitted by the hint budget" in hint


def test_autonomous_large_all_topics_line_is_strictly_bounded() -> None:
    hint = build_autonomous_hint(
        _summaries(), {}, 50_000, 8000, _counter,
    )
    assert _counter(hint) <= 8000
    assert "topic names omitted by the hint budget" in hint


def test_default_large_all_topics_line_is_strictly_bounded() -> None:
    hint = build_default_hint(_summaries(), 8000, _counter)
    assert _counter(hint) <= 8000
    assert "topic names omitted by the hint budget" in hint


def test_even_tiny_budget_never_returns_an_oversized_hint() -> None:
    for builder in (
        lambda: build_supervised_hint(_summaries(2), {}, 1, _counter),
        lambda: build_autonomous_hint(_summaries(2), {}, 100, 1, _counter),
        lambda: build_default_hint(_summaries(2), 1, _counter),
    ):
        hint = builder()
        assert _counter(hint) <= 1
