"""Budget tracking and cost estimation for LongMemEval benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field


# Pricing per million tokens (USD)
PRICING = {
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

# Aliases for convenience
PRICING["sonnet"] = PRICING["claude-sonnet-4-5-20250929"]
PRICING["haiku"] = PRICING["claude-haiku-4-5-20251001"]


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts."""
    key = model
    for alias, prices in PRICING.items():
        if alias in model.lower():
            key = alias
            break
    prices = PRICING.get(key, PRICING["sonnet"])
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


@dataclass
class CostEntry:
    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class BudgetTracker:
    """Tracks spending and enforces a budget cap."""

    budget: float = 5.0
    entries: list[CostEntry] = field(default_factory=list)

    @property
    def spent(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent)

    def can_afford(self, estimated_cost: float) -> bool:
        return self.spent + estimated_cost <= self.budget

    def record(self, label: str, model: str, input_tokens: int, output_tokens: int) -> CostEntry:
        cost = _cost(model, input_tokens, output_tokens)
        entry = CostEntry(
            label=label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.entries.append(entry)
        return entry

    def estimate_question_cost(self, haystack_tokens: int) -> float:
        """Estimate total cost for one question (baseline + VC + judge)."""
        # Baseline: full haystack to Sonnet
        baseline_cost = _cost("sonnet", haystack_tokens + 500, 512)
        # VC tagging: ~230 pairs × ~300 input tokens each to Haiku
        n_pairs = haystack_tokens // 500  # rough estimate of pair count
        tagging_cost = _cost("haiku", n_pairs * 600, n_pairs * 100)
        # VC compaction: ~4 events × ~2000 tokens each to Haiku
        compaction_cost = _cost("haiku", 4 * 3000, 4 * 500)
        # VC query: ~32K to Sonnet
        vc_query_cost = _cost("sonnet", 32_000 + 500, 512)
        # Judge: 2 calls to Haiku (~1000 tokens each)
        judge_cost = _cost("haiku", 2 * 1000, 2 * 100)
        return baseline_cost + tagging_cost + compaction_cost + vc_query_cost + judge_cost

    def summary(self) -> dict:
        return {
            "budget": self.budget,
            "spent": round(self.spent, 4),
            "remaining": round(self.remaining, 4),
            "entries": [
                {
                    "label": e.label,
                    "model": e.model,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cost_usd": round(e.cost_usd, 6),
                }
                for e in self.entries
            ],
        }
