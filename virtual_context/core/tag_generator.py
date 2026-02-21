"""Tag Generator: LLM-based and keyword-based semantic tagging."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from ..patterns import DEFAULT_TEMPORAL_PATTERNS
from ..types import (
    KeywordTagConfig,
    LLMProvider,
    TagGeneratorConfig,
    TagResult,
)
from .cost_tracker import CostTracker
from .tag_canonicalizer import TagCanonicalizer

logger = logging.getLogger(__name__)

TAG_GENERATOR_PROMPT_DETAILED = """\
You are a semantic tagger for conversation segments. Given a piece of conversation,
generate {min_tags}-{max_tags} short, lowercase tags that capture the key topics.

Rules:
- Prefer specific tags over generic ones. A good tag should narrow down WHICH
  conversation this came from, not just what broad category it falls into.
  Single-word tags are fine when already specific ("database", "teeth", "fitness").
  But when a single word is too broad and could match many unrelated conversations,
  qualify it with a hyphenated compound: "reservation-timing" not "timing",
  "cycle-tracking" not "tracking", "transit-schedule" not "schedule".
  Ask yourself: "Would this tag match conversations about completely different topics?"
  If yes, make it more specific.
- When the text genuinely discusses a topic already covered by an existing tag, reuse
  that tag instead of inventing a synonym (e.g. use "teeth" not "dental"
  if "teeth" already exists and the text is about teeth).
- When the text introduces a genuinely NEW topic not covered by any existing tag,
  create a new tag. Do NOT force-fit unrelated text into existing tags.
- Only add the tag "rule" when the user gives the assistant an explicit standing
  instruction about response style — phrases like "always ...", "never ...",
  "from now on ...", "don't sugarcoat", "be honest with me".
  Do NOT tag as "rule": personal opinions ("I hate running"), feelings ("I'm
  overwhelmed"), desires ("I want to learn X"), questions, or topic switches.
- Set "broad" to true ONLY when the query asks for synthesis, summary, or
  patterns across MULTIPLE previously-discussed topics. The key signal is
  that answering requires context from several different conversations, not
  just finding one specific past discussion.

  broad: true examples (needs many topics):
  - "Looking back at everything we've discussed, what would you change?"
  - "What patterns do you see across all of this?"
  - "Summarize what we've covered"
  - "How does everything fit into a 6-month roadmap?"
  - "What's the most important thing from each thread?"

  broad: false examples (specific, even if referencing the past):
  - "What did you say about X earlier?" (looking for ONE topic)
  - "Remind me what we decided about the schema" (specific topic)
  - "How do I implement pagination?" (specific question)
  - "Can you summarize the auth approach?" (one topic, not everything)
  - "What was the knee issue again?" (specific back-reference)
  - "How does feature X work for screen readers?" (specific cross-reference)
  - "Would that approach work at scale?" (follow-up to current topic)
- Set "temporal" to true when the query references a specific time position in
  the conversation — "the first thing we discussed", "at the beginning",
  "early on", "going way back", "when we first started". False for general
  queries, even if they reference the past ("remind me about X" is NOT temporal
  — it's looking for a topic, not a time position).

  temporal: true examples (references conversation position):
  - "Going back to the very first thing we discussed"
  - "What did we decide at the beginning?"
  - "Early on we talked about X — has that changed?"
  - "When we first started, what was the architecture?"

  temporal: false examples (topic recall, not time-positioned):
  - "Remind me what we said about auth" (looking for a topic)
  - "What was the middleware pattern?" (specific topic)
  - "Summarize everything" (broad, not temporal)
- Generate 2-5 related_tags: alternate words someone might use when referring back
  to these same concepts later (e.g. if tagging a discussion about "materialized views",
  related_tags might include "caching", "precomputed", "feed-optimization").
  These help future recall when the user uses different vocabulary.
- Messages may contain channel metadata (e.g. "[Telegram NAME ...]", "[WhatsApp ...]",
  "[Discord ...]", "[message_id: NNN]", timestamps, sender info). Ignore all metadata
  formatting — tag only the actual conversational content within the message.
- Do NOT generate tags about the communication medium, channel, group, or server
  (e.g. "messaging", "threading", "chat", "telegram-group",
  "discord-server", "slack-channel", "texting", "communication", "conversation").
  These describe WHERE or HOW the conversation happens, not WHAT it is about. Tag only the
  substantive topics being discussed.
- For very short or trivial messages (greetings, reactions, single-word responses,
  emoji), return only the tags that genuinely apply — it is acceptable to return
  fewer than {min_tags} tags when the content does not warrant more.
- Tag the concrete subject being discussed, not the conversational framing.
  "What do you think of trees?" → tag "trees" or "nature", NOT "introspection"
  or "cognition". The question format ("what do you think", "how do you feel",
  "tell me about") is framing — the subject is what matters for retrieval.
  Even if the assistant's response is philosophical or reflective, always include
  at least one tag for the concrete noun or topic the user asked about.
- Return JSON only: {{"tags": ["tag1", "tag2"], "primary": "tag1", "broad": false, "temporal": false, "related_tags": ["alt1", "alt2"]}}
- The "primary" tag is the single most relevant tag
- No markdown fences, no extra text
"""

TAG_GENERATOR_PROMPT_COMPACT = """\
You are a semantic tagger. Generate {min_tags}-{max_tags} short, lowercase tags for the key topics.

Rules:
- Prefer single-word tags ("database", "fitness"). Hyphenate only when ambiguous ("machine-learning").
- Reuse existing tags when the topic matches. Create new tags only for genuinely new topics.
- Set "broad" to true for vague/broad/retrospective/recall queries, false otherwise.
- Set "temporal" to true when the query references a time position ("the first thing", "at the beginning", "early on").
- Ignore channel metadata in messages (e.g. "[Telegram ...]", "[message_id: NNN]"). Tag only actual content.
- Do NOT generate tags about the communication medium itself (e.g. "messaging", "threading", "chat"). Tag substantive topics only.
- For very short/trivial messages, return fewer than {min_tags} tags if the content does not warrant more.
- Tag the concrete subject, not conversational framing. "What do you think of trees?" → "trees", NOT "introspection".
- Generate 2-5 related_tags: alternate words for future recall.
- Return JSON only: {{"tags": ["tag1", "tag2"], "primary": "tag1", "broad": false, "temporal": false, "related_tags": ["alt1", "alt2"]}}
- No markdown fences, no extra text
"""

TAG_GENERATOR_PROMPTS = {
    "detailed": TAG_GENERATOR_PROMPT_DETAILED,
    "compact": TAG_GENERATOR_PROMPT_COMPACT,
}


@runtime_checkable
class TagGenerator(Protocol):
    def generate_tags(
        self, text: str, existing_tags: list[str] | None = None,
        context_turns: list[str] | None = None,
    ) -> TagResult: ...


def _compile_broad_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Compile broad-detection regex patterns, skipping invalid ones."""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            logger.warning(f"Invalid broad_pattern regex, skipping: {pattern}")
    return compiled


def detect_broad_heuristic(text: str, patterns: list[re.Pattern]) -> bool:
    """Deterministic broad-query detection via regex patterns."""
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False


def detect_temporal_heuristic(text: str, patterns: list[re.Pattern]) -> bool:
    """Deterministic temporal-query detection via regex patterns."""
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False


class LLMTagGenerator:
    """Generate semantic tags using a local LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: TagGeneratorConfig,
        canonicalizer: TagCanonicalizer | None = None,
        cost_tracker: CostTracker | None = None,
        embed_fn_factory: "Callable[[], Callable[[list[str]], list[list[float]]] | None] | None" = None,
    ) -> None:
        self.llm = llm_provider
        self.config = config
        self._tag_vocabulary: dict[str, int] = {}
        self._canonicalizer = canonicalizer
        self._cost_tracker = cost_tracker
        self._embed_fn_factory = embed_fn_factory
        self._broad_patterns = _compile_broad_patterns(config.broad_patterns)
        self._temporal_patterns = _compile_broad_patterns(config.temporal_patterns)

    def generate_tags(
        self, text: str, existing_tags: list[str] | None = None,
        context_turns: list[str] | None = None,
    ) -> TagResult:
        """Generate semantic tags for the given text."""
        prompt = self._build_prompt(text, existing_tags, context_turns=context_turns)

        prompt_template = TAG_GENERATOR_PROMPTS.get(
            self.config.prompt_mode, TAG_GENERATOR_PROMPT_DETAILED
        )
        system = prompt_template.format(
            min_tags=self.config.min_tags,
            max_tags=self.config.max_tags,
        )

        # Disable thinking mode for models that support it (e.g. qwen3)
        if self.config.disable_thinking:
            prompt = "/no_think\n" + prompt

        try:
            response = self.llm.complete(
                system=system,
                user=prompt,
                max_tokens=self.config.max_tokens,
            )
            self._log_usage()
            result = self._parse_response(response)
        except Exception as e:
            logger.warning(f"LLM tag generation failed: {e}")
            result = TagResult(
                tags=["_general"],
                primary="_general",
                source="fallback",
            )

        # Deterministic override: catch broad queries the LLM missed
        if self.config.broad_heuristic_enabled and not result.broad and detect_broad_heuristic(text, self._broad_patterns):
            logger.debug("Broad heuristic override: LLM missed broad, heuristic caught it")
            result.broad = True

        # Deterministic override: catch temporal queries the LLM missed
        if self.config.temporal_heuristic_enabled and not result.temporal and detect_temporal_heuristic(text, self._temporal_patterns):
            logger.debug("Temporal heuristic override: LLM missed temporal, heuristic caught it")
            result.temporal = True

        return result

    def _select_relevant_store_tags(
        self, text: str, store_tags: list[str], limit: int = 30,
        similarity_threshold: float = 0.25,
    ) -> list[str]:
        """Select store tags most relevant to *text* using embedding similarity,
        then fill remaining slots with high-usage tags.

        Strategy: embed the text and all store tags, take those above
        *similarity_threshold* (up to *limit*). If fewer than *limit* tags
        qualify, pad with the highest-usage tags (which come first in
        *store_tags* since the caller orders by usage_count DESC).

        Falls back entirely to usage-count ordering when no embedding
        function is available.
        """
        if not store_tags:
            return []

        embed_fn: Callable | None = None
        if self._embed_fn_factory is not None:
            try:
                embed_fn = self._embed_fn_factory()
            except Exception:
                embed_fn = None

        if embed_fn is None:
            return store_tags[:limit]

        try:
            import numpy as np

            # Truncate text for embedding (first 500 chars is plenty for topic signal)
            snippet = text[:500]
            all_texts = [snippet] + store_tags
            vectors = embed_fn(all_texts)
            text_vec = np.array(vectors[0])
            tag_vecs = np.array(vectors[1:])

            # Cosine similarity
            norms = np.linalg.norm(tag_vecs, axis=1)
            norms[norms == 0] = 1.0
            text_norm = np.linalg.norm(text_vec)
            if text_norm == 0:
                return store_tags[:limit]
            similarities = tag_vecs @ text_vec / (norms * text_norm)

            # Take tags above threshold, ranked by similarity
            above = [(float(similarities[i]), i) for i in range(len(store_tags))
                     if similarities[i] >= similarity_threshold]
            above.sort(reverse=True)
            selected_indices = [i for _, i in above[:limit]]
            selected_set = set(selected_indices)

            # Fill remaining slots with highest-usage tags (preserve caller order)
            for i in range(len(store_tags)):
                if len(selected_indices) >= limit:
                    break
                if i not in selected_set:
                    selected_indices.append(i)
                    selected_set.add(i)

            return [store_tags[i] for i in selected_indices]
        except Exception:
            logger.debug("Embed-based tag selection failed, falling back to usage-count", exc_info=True)
            return store_tags[:limit]

    def _build_prompt(self, text: str, existing_tags: list[str] | None = None, context_turns: list[str] | None = None) -> str:
        """Build the tagging prompt with vocabulary hint.

        Splits tags into recent session tags (highest reuse priority) and
        store tags (secondary, selected by embedding relevance to the text).
        This encourages the LLM to converge on the same tags within a session
        rather than inventing synonyms.
        """
        parts = []

        # Recent session tags (from vocabulary tracker) — highest reuse priority
        session_tags = sorted(
            self._tag_vocabulary.keys(),
            key=lambda t: self._tag_vocabulary.get(t, 0),
            reverse=True,
        )

        # Store/external tags (passed in by caller)
        store_tags = existing_tags or []

        # Select store tags relevant to this text (embed-based when available)
        session_set = set(session_tags)
        candidates = [t for t in store_tags if t not in session_set]
        extra_store = self._select_relevant_store_tags(text, candidates, limit=30)

        if session_tags:
            parts.append(
                f"Existing tags (reuse when the topic genuinely matches, "
                f"but create new tags for new topics): {', '.join(session_tags)}"
            )
        if extra_store:
            parts.append(f"Other known tags: {', '.join(extra_store)}")

        # Inject recent conversation context when provided
        if context_turns:
            parts.append("")
            parts.append("Recent conversation context:")
            for j in range(0, len(context_turns) - 1, 2):
                parts.append(f"User: {context_turns[j]}")
                if j + 1 < len(context_turns):
                    parts.append(f"Assistant: {context_turns[j + 1]}")
            parts.append("")

        parts.append(f"Tag this conversation (return {self.config.min_tags}-{self.config.max_tags} tags):")
        parts.append("")

        # Truncate text to avoid overwhelming the model
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated]"
        parts.append(text)

        return "\n".join(parts)

    def _parse_response(self, response: str) -> TagResult:
        """Parse LLM JSON response into TagResult."""
        text = response.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Strip thinking tags (qwen3 often wraps in <think>...</think>)
        if "<think>" in text:
            # Remove everything between <think> and </think>
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        tags = data.get("tags", [])
        primary = data.get("primary", "")
        broad = bool(data.get("broad", False))
        temporal = bool(data.get("temporal", False))

        if not tags:
            return TagResult(
                tags=["_general"],
                primary="_general",
                source="fallback",
            )

        tags = self._normalize_tags(tags)
        if primary:
            primary = self._normalize_tag(primary)
            if primary not in tags:
                tags.insert(0, primary)
        else:
            primary = tags[0]

        # Enforce min/max
        tags = tags[:self.config.max_tags]

        # Parse and normalize related_tags
        related_tags_raw = data.get("related_tags", [])
        if isinstance(related_tags_raw, list):
            related_tags = self._normalize_tags(related_tags_raw)
            # Deduplicate: remove any that overlap with primary tags
            tag_set = set(tags)
            related_tags = [t for t in related_tags if t not in tag_set]
        else:
            related_tags = []

        # Update vocabulary
        for tag in tags:
            self._tag_vocabulary[tag] = self._tag_vocabulary.get(tag, 0) + 1

        return TagResult(
            tags=tags,
            primary=primary,
            source="llm",
            broad=broad,
            temporal=temporal,
            related_tags=related_tags,
        )

    def _normalize_tag(self, tag: str) -> str:
        """Normalize a single tag: lowercase, hyphenate, resolve aliases."""
        tag = tag.lower().strip()
        tag = re.sub(r"[^a-z0-9-]", "-", tag)
        tag = re.sub(r"-+", "-", tag).strip("-")
        if self._canonicalizer:
            tag = self._canonicalizer.canonicalize(tag)
        return tag

    def _normalize_tags(self, tags: list) -> list[str]:
        """Normalize and deduplicate a list of tags."""
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            normalized = self._normalize_tag(tag)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def load_vocabulary(self, tag_counts: dict[str, int]) -> None:
        """Bootstrap vocabulary from existing stored tag counts."""
        self._tag_vocabulary.update(tag_counts)

    def _log_usage(self) -> None:
        """Log LLM token usage from the provider's last_usage to the cost tracker."""
        if not self._cost_tracker:
            return
        usage = getattr(self.llm, "last_usage", {})
        if not usage:
            return
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        if input_tokens or output_tokens:
            self._cost_tracker.log_tag_generation(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider=getattr(self.llm, "model", ""),
            )


class KeywordTagGenerator:
    """Deterministic tag generation using keywords and regex patterns."""

    def __init__(self, config: KeywordTagConfig) -> None:
        self.config = config
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Precompile regex patterns."""
        for tag, patterns in self.config.tag_patterns.items():
            self._compiled_patterns[tag] = []
            for pattern in patterns:
                try:
                    self._compiled_patterns[tag].append(
                        re.compile(pattern, re.IGNORECASE)
                    )
                except re.error:
                    logger.warning(f"Invalid regex pattern for tag '{tag}': {pattern}")

    def generate_tags(
        self, text: str, existing_tags: list[str] | None = None,
        context_turns: list[str] | None = None,
    ) -> TagResult:
        """Generate tags from keyword/pattern matching."""
        text_lower = text.lower()
        matched_tags: set[str] = set()

        # Keyword matching
        for tag, keywords in self.config.tag_keywords.items():
            if any(kw.lower() in text_lower for kw in keywords):
                matched_tags.add(tag)

        # Pattern matching
        for tag, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    matched_tags.add(tag)
                    break

        if not matched_tags:
            return TagResult(
                tags=["_general"],
                primary="_general",
                source="fallback",
            )

        tags = sorted(matched_tags)
        primary = tags[0]

        return TagResult(
            tags=tags,
            primary=primary,
            source="keyword",
        )


def build_tag_generator(
    config: TagGeneratorConfig,
    llm_provider: LLMProvider | None = None,
    canonicalizer: TagCanonicalizer | None = None,
    cost_tracker: CostTracker | None = None,
    embed_fn_factory: "Callable[[], Callable[[list[str]], list[list[float]]] | None] | None" = None,
) -> TagGenerator:
    """Build a tag generator from config. Falls back to keyword if no LLM available."""
    if config.type == "llm" and llm_provider is not None:
        return LLMTagGenerator(
            llm_provider=llm_provider, config=config,
            canonicalizer=canonicalizer, cost_tracker=cost_tracker,
            embed_fn_factory=embed_fn_factory,
        )

    if config.type == "embedding":
        from .embedding_tag_generator import EmbeddingTagGenerator
        return EmbeddingTagGenerator(config=config)

    if config.keyword_fallback:
        return KeywordTagGenerator(config=config.keyword_fallback)

    # Default: empty keyword config
    return KeywordTagGenerator(config=KeywordTagConfig())
