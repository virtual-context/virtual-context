"""CostTracker: accumulate token usage and cost estimates per session."""

from __future__ import annotations

from ..types import CostTrackingConfig, SessionCostSummary


class CostTracker:
    """Track token usage and estimated costs for a session."""

    def __init__(self, config: CostTrackingConfig) -> None:
        self.config = config
        self._summary = SessionCostSummary()

    def log_retrieval(self, input_tokens: int = 0, output_tokens: int = 0, provider: str = "") -> None:
        """Log a retrieval event."""
        self._summary.total_retrievals += 1
        self._summary.total_input_tokens += input_tokens
        self._summary.total_output_tokens += output_tokens
        self._update_cost(input_tokens, output_tokens, provider)

    def log_compaction(self, input_tokens: int = 0, output_tokens: int = 0, provider: str = "") -> None:
        """Log a compaction event."""
        self._summary.total_compactions += 1
        self._summary.total_input_tokens += input_tokens
        self._summary.total_output_tokens += output_tokens
        self._update_cost(input_tokens, output_tokens, provider)

    def log_tag_generation(self, input_tokens: int = 0, output_tokens: int = 0, provider: str = "") -> None:
        """Log a tag generation event."""
        self._summary.total_tag_generations += 1
        self._summary.total_input_tokens += input_tokens
        self._summary.total_output_tokens += output_tokens
        self._update_cost(input_tokens, output_tokens, provider)

    def _update_cost(self, input_tokens: int, output_tokens: int, provider: str) -> None:
        """Update estimated cost based on pricing config."""
        # Exact match first, then substring match (e.g. "haiku" in "claude-haiku-4-5-20251001")
        pricing = self.config.pricing.get(provider, {})
        if not pricing:
            provider_lower = provider.lower()
            for key, val in self.config.pricing.items():
                if key.lower() in provider_lower:
                    pricing = val
                    break
        input_rate = pricing.get("input_per_1k", 0.0)
        output_rate = pricing.get("output_per_1k", 0.0)
        self._summary.estimated_cost_usd += (
            (input_tokens / 1000) * input_rate
            + (output_tokens / 1000) * output_rate
        )

    def get_summary(self) -> SessionCostSummary:
        """Return current session cost summary."""
        return SessionCostSummary(
            total_retrievals=self._summary.total_retrievals,
            total_compactions=self._summary.total_compactions,
            total_tag_generations=self._summary.total_tag_generations,
            total_input_tokens=self._summary.total_input_tokens,
            total_output_tokens=self._summary.total_output_tokens,
            estimated_cost_usd=self._summary.estimated_cost_usd,
        )
