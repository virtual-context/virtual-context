"""ContextMonitor: two-tier threshold checking for compaction triggers."""

from __future__ import annotations

from typing import Callable

from ..types import CompactionSignal, ContextSnapshot, MonitorConfig


class ContextMonitor:
    """Monitor context token usage against soft/hard thresholds.

    - Soft threshold (default 70%): proactive compaction recommended
    - Hard threshold (default 85%): mandatory blocking compaction
    """

    def __init__(
        self,
        config: MonitorConfig,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self._last_snapshot: ContextSnapshot | None = None

    def check(self, snapshot: ContextSnapshot) -> CompactionSignal | None:
        """Check snapshot against thresholds. Returns signal if compaction needed."""
        self._last_snapshot = snapshot

        budget = snapshot.budget_tokens
        if budget <= 0:
            return None

        usage_ratio = snapshot.total_tokens / budget

        if usage_ratio >= self.config.hard_threshold:
            overflow = snapshot.total_tokens - int(budget * self.config.soft_threshold)
            return CompactionSignal(
                priority="hard",
                current_tokens=snapshot.total_tokens,
                budget_tokens=budget,
                overflow_tokens=overflow,
            )

        if usage_ratio >= self.config.soft_threshold:
            overflow = snapshot.total_tokens - int(budget * self.config.soft_threshold)
            return CompactionSignal(
                priority="soft",
                current_tokens=snapshot.total_tokens,
                budget_tokens=budget,
                overflow_tokens=overflow,
            )

        return None

    def force_compact(self) -> CompactionSignal:
        """Force immediate compaction (e.g., for context_length_exceeded error recovery)."""
        snapshot = self._last_snapshot
        if snapshot:
            overflow = snapshot.total_tokens - int(
                snapshot.budget_tokens * self.config.soft_threshold
            )
        else:
            overflow = 10_000

        return CompactionSignal(
            priority="hard",
            current_tokens=snapshot.total_tokens if snapshot else 0,
            budget_tokens=snapshot.budget_tokens if snapshot else 0,
            overflow_tokens=max(overflow, 1000),
        )

    def build_snapshot(
        self,
        conversation_history: list,
        core_tokens: int = 0,
        domain_tokens: int = 0,
        system_tokens: int = 0,
    ) -> ContextSnapshot:
        """Build a ContextSnapshot from conversation history."""
        conv_text = " ".join(
            m.content if hasattr(m, "content") else str(m)
            for m in conversation_history
        )
        conv_tokens = self.token_counter(conv_text)

        total = system_tokens + core_tokens + domain_tokens + conv_tokens

        return ContextSnapshot(
            system_tokens=system_tokens,
            core_context_tokens=core_tokens,
            retrieved_domain_tokens=domain_tokens,
            conversation_tokens=conv_tokens,
            total_tokens=total,
            budget_tokens=self.config.context_window,
            turn_count=len(conversation_history) // 2,
        )
