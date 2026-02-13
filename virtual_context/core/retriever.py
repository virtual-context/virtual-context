"""ContextRetriever: classify inbound message and fetch relevant domain summaries."""

from __future__ import annotations

import logging
import time
from collections import Counter

from ..classifiers.base import ClassifierPipeline
from ..core.store import ContextStore
from ..types import (
    ClassificationResult,
    DomainDef,
    Message,
    RetrievalResult,
    RetrieverConfig,
    StoredSummary,
)

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Retrieve relevant domain summaries for an inbound message.

    When the classifier can't match the inbound message to a domain (e.g. "What
    do you think about that?"), falls back to domain velocity: the more the recent
    conversation has concentrated on a domain, the more likely the next unkeyworded
    message is about it too.
    """

    def __init__(
        self,
        classifier_pipeline: ClassifierPipeline,
        store: ContextStore,
        config: RetrieverConfig,
    ) -> None:
        self.classifier = classifier_pipeline
        self.store = store
        self.config = config
        self._domain_defs = {d.name: d for d in config.domains}

    async def retrieve(
        self,
        message: str,
        current_domains_in_context: list[str] | None = None,
        conversation_history: list[Message] | None = None,
    ) -> RetrievalResult:
        """Classify inbound message, fetch relevant summaries from store.

        If the classifier returns no confident domain match and velocity_fallback
        is enabled, computes domain velocity from recent conversation and retrieves
        summaries for high-velocity domains.
        """
        start_time = time.monotonic()
        active_domains = set(current_domains_in_context or [])

        # Classify the inbound message
        domains_list = list(self._domain_defs.values())
        results = await self.classifier.classify(message, domains_list)

        # Check if we got a real match or just _general fallback
        real_matches = [r for r in results if r.domain != "_general"]
        used_velocity = False
        velocity_scores: dict[str, float] = {}

        if not real_matches and self.config.velocity_fallback and conversation_history:
            # No keyword match — compute domain velocity from recent conversation
            velocity_scores = await self._compute_velocity(conversation_history, domains_list)
            # Convert high-velocity domains into synthetic classification results
            for domain, velocity in sorted(velocity_scores.items(), key=lambda x: x[1], reverse=True):
                if velocity >= self.config.velocity_threshold:
                    results.append(ClassificationResult(
                        domain=domain,
                        confidence=velocity,  # velocity as confidence proxy
                        source="velocity",
                    ))
                    used_velocity = True

        matched_domains: list[str] = []
        all_summaries: list[StoredSummary] = []
        total_tokens = 0
        global_budget = self.config.domain_context_max_tokens

        for result in results:
            domain = result.domain

            # Skip active domains already in recent conversation
            if self.config.skip_active_domains and domain in active_domains:
                logger.debug(f"Skipping active domain: {domain}")
                continue

            if domain == "_general":
                continue

            matched_domains.append(domain)

            # Get per-domain config
            domain_def = self._domain_defs.get(domain)
            limit = domain_def.retrieval_limit if domain_def else 3
            max_tokens = domain_def.retrieval_max_tokens if domain_def else 5000

            # Fetch summaries from store
            summaries = await self.store.get_summaries(domain=domain, limit=limit)

            # Apply per-domain token budget
            domain_tokens = 0
            for summary in summaries:
                if domain_tokens + summary.summary_tokens > max_tokens:
                    break
                if total_tokens + summary.summary_tokens > global_budget:
                    break
                all_summaries.append(summary)
                domain_tokens += summary.summary_tokens
                total_tokens += summary.summary_tokens

        elapsed = time.monotonic() - start_time

        return RetrievalResult(
            domains_matched=matched_domains,
            summaries=all_summaries,
            full_detail=[],  # Deep retrieval deferred to v0.2
            total_tokens=total_tokens,
            retrieval_metadata={
                "elapsed_ms": round(elapsed * 1000, 1),
                "domains_checked": len(results),
                "domains_matched": len(matched_domains),
                "domains_skipped_active": len(active_domains & {r.domain for r in results}),
                "summaries_returned": len(all_summaries),
                "used_velocity_fallback": used_velocity,
                "velocity_scores": velocity_scores,
            },
        )

    async def _compute_velocity(
        self,
        conversation_history: list[Message],
        domains: list[DomainDef],
    ) -> dict[str, float]:
        """Compute domain velocity from recent conversation.

        Looks at the last N turn pairs, classifies each, and returns
        the concentration ratio per domain.

        Returns:
            {domain_name: velocity} where velocity = domain_turns / total_turns
        """
        lookback = self.config.velocity_lookback * 2  # messages, not turn pairs
        recent = conversation_history[-lookback:] if len(conversation_history) > lookback else conversation_history

        if not recent:
            return {}

        # Classify each turn pair in the lookback window
        domain_counts: Counter[str] = Counter()
        total_pairs = 0

        # Walk through messages pairing user+assistant
        i = 0
        while i < len(recent):
            # Collect one turn pair worth of text
            pair_text_parts: list[str] = []

            if recent[i].role == "user":
                pair_text_parts.append(recent[i].content)
                # Look for the assistant response
                if i + 1 < len(recent) and recent[i + 1].role == "assistant":
                    pair_text_parts.append(recent[i + 1].content)
                    i += 2
                else:
                    i += 1
            else:
                pair_text_parts.append(recent[i].content)
                i += 1

            pair_text = " ".join(pair_text_parts)
            pair_results = await self.classifier.classify(pair_text, domains)

            # Count the primary domain (first result)
            if pair_results and pair_results[0].domain != "_general":
                domain_counts[pair_results[0].domain] += 1

            total_pairs += 1

        if total_pairs == 0:
            return {}

        return {domain: count / total_pairs for domain, count in domain_counts.items()}
