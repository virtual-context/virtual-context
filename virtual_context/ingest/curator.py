"""Fact curator: LLM-based relevance filter for the facts block."""

from __future__ import annotations

import logging
import re

from ..types import CurationConfig, Fact, LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a memory relevance assistant. "
    "Given a user question and a list of personal facts, identify which facts "
    "could possibly be relevant to answering the question. "
    "Respond only with the fact numbers, comma-separated (e.g. '0, 3, 7'). "
    "No explanation. If none are relevant, respond with an empty string."
)


class FactCurator:
    """Filter a facts list down to query-relevant facts using an LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        config: CurationConfig,
    ) -> None:
        self.llm = llm_provider
        self.model = model
        self.config = config

    def curate(self, facts: list[Fact], question: str) -> list[Fact]:
        """Return the subset of facts relevant to the question.

        Falls back to the original list on any error.
        """
        if not facts:
            return facts

        facts_text = self._format_facts(facts)
        user_prompt = (
            f'User question: "{question}"\n\n'
            f"Which of these facts could be relevant to answering the question? "
            f"Output only the fact numbers, comma-separated.\n\n"
            f"Facts:\n{facts_text}"
        )

        try:
            response = self.llm.complete(
                system=_SYSTEM,
                user=user_prompt,
                max_tokens=self.config.max_response_tokens,
            )
        except Exception as e:
            logger.warning("Fact curation LLM call failed: %s — returning all facts", e)
            return facts

        selected = self._parse_response(response, len(facts))
        if not selected:
            logger.debug("Fact curation returned no indices — returning all facts")
            return facts

        logger.info("Fact curation: %d → %d facts", len(facts), len(selected))
        return [facts[i] for i in selected]

    def _format_facts(self, facts: list[Fact]) -> str:
        lines = []
        for i, f in enumerate(facts):
            line = f"[{i}] {f.verb} | {f.object}"
            if f.when_date:
                line += f" [when: {f.when_date}]"
            elif f.session_date:
                line += f" [session: {f.session_date}]"
            lines.append(line)
        return "\n".join(lines)

    def _parse_response(self, response: str, total: int) -> list[int]:
        """Extract valid fact indices from LLM response."""
        # Strip thinking tags (e.g. Qwen3)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        nums = re.findall(r"\d+", response)
        indices = [int(n) for n in nums if 0 <= int(n) < total]
        return list(dict.fromkeys(indices))  # deduplicate preserving order
