"""Fact supersession checker: detect and mark contradicted facts."""

from __future__ import annotations

import json
import logging
import re

from ..core.store import ContextStore
from ..types import Fact, LLMProvider, SupersessionConfig

logger = logging.getLogger(__name__)


class FactSupersessionChecker:
    """Check new facts against existing facts and mark superseded ones."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        store: ContextStore,
        config: SupersessionConfig,
    ):
        self.llm = llm_provider
        self.model = model
        self.store = store
        self.config = config

    def check_and_supersede(self, new_facts: list[Fact]) -> int:
        """For each new fact, find candidates by subject, ask LLM, mark superseded.

        Returns count of superseded facts.
        """
        if not self.config.enabled or not new_facts:
            return 0

        superseded_count = 0
        for fact in new_facts:
            if not fact.subject:
                continue
            # Query existing non-superseded facts with same subject
            candidates = self.store.query_facts(
                subject=fact.subject, limit=self.config.batch_size,
            )
            candidates = [c for c in candidates if c.id != fact.id]
            if not candidates:
                continue

            superseded_ids = self._check_batch(fact, candidates)
            for old_id in superseded_ids:
                self.store.set_fact_superseded(old_id, fact.id)
                superseded_count += 1

        return superseded_count

    def _check_batch(self, new_fact: Fact, candidates: list[Fact]) -> list[str]:
        """Ask LLM which candidates are superseded by new_fact."""
        prompt = self._build_prompt(new_fact, candidates)
        try:
            response = self.llm.complete(
                system="You are a fact comparison assistant. Respond only with a JSON array.",
                user=prompt,
                max_tokens=200,
            )
        except Exception as e:
            logger.warning("Supersession LLM call failed: %s", e)
            return []
        return self._parse_response(response, candidates)

    def _build_prompt(self, new_fact: Fact, candidates: list[Fact]) -> str:
        """Build prompt asking which candidates are superseded or duplicated."""
        lines = [
            "A new fact has been extracted from a conversation:",
            f"  Subject: {new_fact.subject}",
            f"  Verb: {new_fact.verb}",
            f"  Object: {new_fact.object}",
            f"  Status: {new_fact.status}",
        ]
        if new_fact.what:
            lines.append(f"  What: {new_fact.what}")
        lines.append("")
        lines.append("Existing facts with the same subject:")
        for i, c in enumerate(candidates):
            line = f"  [{i}] {c.verb} {c.object} (status: {c.status})"
            if c.what:
                line += f" — {c.what}"
            lines.append(line)
        lines.append("")
        lines.append(
            "Which existing facts (by index) are CONTRADICTED, SUPERSEDED, or "
            "DUPLICATED by the new fact? A fact is duplicated if it describes the "
            "same underlying event/state with different wording. When duplicates "
            "are found, mark the LESS detailed version for removal. "
            "Reply with a JSON array of indices, e.g. [0, 2]. "
            "Reply [] if none are superseded or duplicated."
        )
        return "\n".join(lines)

    def _parse_response(self, response: str, candidates: list[Fact]) -> list[str]:
        """Extract superseded fact IDs from LLM response."""
        match = re.search(r'\[[\d,\s]*\]', response)
        if not match:
            return []
        try:
            indices = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return []
        return [candidates[i].id for i in indices if 0 <= i < len(candidates)]
