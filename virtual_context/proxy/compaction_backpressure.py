"""Decide whether a request should wait on a pending hard compaction.

A hard compaction signal makes the next request block until the compaction
lands, so the payload it sends upstream is the reduced one. Compaction only
removes the turns before the protected window, so when the protected window
alone already reaches the hard threshold the wait cannot change what goes
upstream and only adds the compaction's duration to the request.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactionWaitDecision:
    """What the last tag pass concluded about waiting for its compaction."""

    priority: str = ""  # "", "soft" or "hard"
    floor_tokens: int = 0  # tokens compaction cannot remove (protected window)
    hard_tokens: int = 0  # budget × hard threshold

    def should_wait(self) -> bool:
        return self.priority == "hard" and self.floor_tokens < self.hard_tokens


def hard_tokens_for(budget_tokens: int, hard_threshold: float) -> int:
    try:
        return int(int(budget_tokens) * float(hard_threshold))
    except (TypeError, ValueError):
        return 0


def decide_compaction_wait(signal, *, floor_tokens: int, hard_threshold: float) -> CompactionWaitDecision:
    """Decision for *signal* (a ``CompactionSignal`` or ``None``).

    *floor_tokens* is what the payload keeps whatever compaction does. A
    request waits only for a hard compaction that can bring the payload
    under the hard threshold, i.e. when that floor is below it.
    """
    if signal is None:
        return CompactionWaitDecision()
    priority = str(getattr(signal, "priority", "") or "")
    hard = hard_tokens_for(getattr(signal, "budget_tokens", 0), hard_threshold)
    return CompactionWaitDecision(priority=priority, floor_tokens=max(0, int(floor_tokens)), hard_tokens=hard)
