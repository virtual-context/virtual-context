"""Fact supersession checker: detect and mark contradicted facts."""

from __future__ import annotations

import json
import logging
import re

from ..core.store import ContextStore
from ..types import Fact, LLMProvider, SupersessionConfig

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "from", "with", "that", "about", "into", "over", "have", "been", "will",
    "this", "their", "there", "where", "which", "would", "could", "should",
    "after", "before", "during", "while", "other", "another", "these", "those",
    "trip", "solo", "recent", "just", "back", "today", "recently", "returned",
    "camping", "hiking", "visited", "started", "began",
})


def _extract_object_keyword(object_str: str) -> str | None:
    """Extract the most distinctive word from a fact's object string.

    Used to find cross-session duplicate facts via object_contains lookup.
    Prefers proper nouns (initial capital); falls back to longest word >= 5 chars.
    Returns None if no distinctive word is found.
    """
    words = re.findall(r"[A-Za-z']+", object_str)
    candidates = [w for w in words if len(w) >= 5 and w.lower() not in _STOPWORDS]
    if not candidates:
        return None
    proper = [w for w in candidates if w[0].isupper()]
    pool = proper if proper else candidates
    return max(pool, key=len)


_MERGE_SYSTEM = (
    "You are a memory consolidation assistant. Respond only with JSON."
)

_MERGE_PROMPT = """\
A fact has been superseded by a newer fact. Produce a single merged fact \
that captures the current state and what changed.

State what IS true now as durable knowledge. Do not echo the original \
conversational phrasing — restate the facts in neutral, declarative language.

SUPERSEDED (old) fact:
  Subject: {old_subject}
  Verb: {old_verb}
  Object: {old_object}
  Status: {old_status}
  What: {old_what}

CURRENT (new) fact:
  Subject: {new_subject}
  Verb: {new_verb}
  Object: {new_object}
  Status: {new_status}
  What: {new_what}

Produce a merged fact with updated fields:
- "verb": a declarative verb describing the current state (e.g. "has", "holds", "improved")
- "object": the current value with specifics preserved
- "status": the appropriate temporal status
- "what": one or two sentences of durable knowledge — current state and what changed

Reply with JSON: {{"verb": "...", "object": "...", "status": "...", "what": "..."}}\
"""


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
            # Query existing non-superseded facts with same subject.
            # When tags are available, filter by them to avoid sending
            # unrelated facts to the LLM (reduces false supersessions).
            candidates = self.store.query_facts(
                subject=fact.subject,
                tags=fact.tags if fact.tags else None,
                limit=self.config.batch_size,
            )
            candidates = [c for c in candidates if c.id != fact.id]
            if not candidates:
                continue

            superseded_ids = self._check_batch(fact, candidates)
            for old_id in superseded_ids:
                self.store.set_fact_superseded(old_id, fact.id)
                superseded_count += 1
                # Merge the old fact's knowledge into the surviving fact
                old_fact = next((c for c in candidates if c.id == old_id), None)
                if old_fact:
                    self._merge_facts(fact, old_fact)

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
            "A fact is SUPERSEDED when it describes an earlier value of the same "
            "attribute (e.g. a previous record, an old address, a former preference). "
            "Look at the underlying attribute being described, not just the verb phrasing. "
            "Reply with a JSON array of indices, e.g. [0, 2]. "
            "Reply [] if none are superseded or duplicated."
        )
        return "\n".join(lines)

    def _merge_facts(self, winning_fact: Fact, old_fact: Fact) -> None:
        """Merge old fact's knowledge into the winning fact via LLM.

        Updates verb, object, status, and what on the winning fact to capture
        the temporal transition as durable knowledge.
        """
        prompt = _MERGE_PROMPT.format(
            old_subject=old_fact.subject,
            old_verb=old_fact.verb,
            old_object=old_fact.object,
            old_status=old_fact.status,
            old_what=old_fact.what or f"{old_fact.subject} {old_fact.verb} {old_fact.object}",
            new_subject=winning_fact.subject,
            new_verb=winning_fact.verb,
            new_object=winning_fact.object,
            new_status=winning_fact.status,
            new_what=winning_fact.what or f"{winning_fact.subject} {winning_fact.verb} {winning_fact.object}",
        )
        try:
            response = self.llm.complete(
                system=_MERGE_SYSTEM,
                user=prompt,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning("Supersession merge LLM call failed: %s", e)
            return

        merged = self._parse_merge_response(response)
        if not merged:
            return

        verb = merged.get("verb", winning_fact.verb)
        obj = merged.get("object", winning_fact.object)
        status = merged.get("status", winning_fact.status)
        what = merged.get("what", winning_fact.what)

        self.store.update_fact_fields(winning_fact.id, verb, obj, status, what)
        # Update the in-memory object so subsequent merges see current state
        winning_fact.verb = verb
        winning_fact.object = obj
        winning_fact.status = status
        winning_fact.what = what
        logger.info(
            "Merged superseded fact into %s: verb=%r, object=%r",
            winning_fact.id[:8], verb, obj,
        )

    def _parse_merge_response(self, response: str) -> dict | None:
        """Parse the merge LLM response into a dict with verb/object/status/what."""
        text = response.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "verb" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            # Try extracting JSON object from response
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    if isinstance(data, dict) and "verb" in data:
                        return data
                except (json.JSONDecodeError, ValueError):
                    pass
        logger.warning("Failed to parse merge response: %s", text[:200])
        return None

    def _parse_response(self, response: str, candidates: list[Fact]) -> list[str]:
        """Extract superseded fact IDs from LLM response."""
        text = response.strip()
        # Strip thinking tags from models like Qwen3
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Try to parse as JSON object first (Qwen3 returns {"updated": [0]})
        try:
            data = json.loads(text)
            if isinstance(data, list):
                indices = data
            elif isinstance(data, dict):
                # Accept various key names models may use
                for key in ("updated", "superseded", "indices",
                            "contradicted_or_updated", "result"):
                    if key in data and isinstance(data[key], list):
                        indices = data[key]
                        break
                else:
                    indices = []
            else:
                indices = []
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract bare array from response text
            match = re.search(r'\[[\d,\s]*\]', text)
            if not match:
                return []
            try:
                indices = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                return []

        return [candidates[i].id for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
