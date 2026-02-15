"""DomainCompactor: summarizes TaggedSegments using a configurable LLM."""

from __future__ import annotations

import fnmatch
import json
import logging
from typing import Callable

from ..types import (
    CompactionResult,
    CompactorConfig,
    LLMProvider,
    Message,
    SegmentMetadata,
    StoredSummary,
    TaggedSegment,
    TagPromptRule,
    TagSummary,
)

logger = logging.getLogger(__name__)

TAG_SUMMARY_ROLLUP_PROMPT = """\
You are summarizing all stored context about the tag "{tag}".
Below are {count} segment summaries that each cover a portion of conversation
where "{tag}" was discussed. Roll them up into a single coherent summary that
preserves all key decisions, action items, entities, names, dates, and numbers.
Keep the chronological progression. The summary should be comprehensive enough
that someone could resume the conversation from it.

Target length: {target_tokens} tokens or fewer.

Segment summaries:
{segment_summaries}

Respond with JSON:
{{
  "summary": "...",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."]
}}"""


DEFAULT_SUMMARY_PROMPT = """\
Summarize the following conversation segment (tags: {tags}).
Preserve: key decisions, action items, entities mentioned, specific data points (numbers, dates, names),
and specific feature/concept names exactly as discussed (e.g. "cook mode", "dark theme", "rate limiter" —
do NOT generalize these into broader categories like "UI features" or "infrastructure").
Be concise but retain enough detail that the conversation could be resumed from this summary.
The summary should be {target_tokens} tokens or fewer.

Conversation:
{conversation_text}

Respond with JSON:
{{
  "summary": "...",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."],
  "date_references": ["..."],
  "refined_tags": ["tag1", "tag2"],
  "related_tags": ["alternate-term1", "alternate-term2"]
}}

For "related_tags", generate 3-8 alternate terms someone might use to refer to these
concepts later (e.g. if discussing "materialized views", related_tags might include
"caching", "precomputed", "feed-optimization", "query-cache")."""


class DomainCompactor:
    """Summarize each TaggedSegment independently using an LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: CompactorConfig,
        token_counter: Callable[[str], int] | None = None,
        model_name: str = "",
        tag_rules: list[TagPromptRule] | None = None,
    ) -> None:
        self.llm = llm_provider
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self.model_name = model_name
        self.tag_rules = tag_rules or []

    def compact(self, segments: list[TaggedSegment]) -> list[CompactionResult]:
        """Summarize each segment independently.

        Uses ThreadPoolExecutor for concurrent summarization when there are
        multiple segments. Falls back to sequential for single segments.
        """
        if len(segments) <= 1:
            # Sequential for single segment — no threading overhead
            return [self._compact_one(s) for s in segments]

        max_workers = min(self.config.max_concurrent_summaries, len(segments))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[CompactionResult] = [None] * len(segments)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._compact_one, segment): i
                for i, segment in enumerate(segments)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Concurrent compaction failed for segment {idx}: {e}")
                    # Create a fallback result
                    segment = segments[idx]
                    conversation_text = self._format_conversation(segment.messages)
                    results[idx] = CompactionResult(
                        segment_id=segment.id,
                        primary_tag=segment.primary_tag,
                        tags=segment.tags,
                        summary=conversation_text[:2000],
                        summary_tokens=self.token_counter(conversation_text[:2000]),
                        original_tokens=self.token_counter(conversation_text),
                        compression_ratio=0.0,
                        full_text=conversation_text,
                    )

        return results

    def _compact_one(self, segment: TaggedSegment) -> CompactionResult:
        """Summarize a single segment."""
        conversation_text = self._format_conversation(segment.messages)
        original_tokens = self.token_counter(conversation_text)

        target_tokens = max(
            self.config.min_summary_tokens,
            min(
                self.config.max_summary_tokens,
                int(original_tokens * self.config.summary_ratio),
            ),
        )

        # Check for custom prompt from tag rules
        custom_prompt = self._get_prompt_for_tags(segment.tags)

        if custom_prompt:
            prompt = (
                f"{custom_prompt}\n\n"
                f"Target length: {target_tokens} tokens or fewer.\n\n"
                f"Conversation:\n{conversation_text}\n\n"
                'Respond with JSON: {{"summary": "...", "entities": ["..."], '
                '"key_decisions": ["..."], "action_items": ["..."], '
                '"date_references": ["..."], "refined_tags": ["tag1", "tag2"]}}'
            )
        else:
            tags_str = ", ".join(segment.tags) if segment.tags else segment.primary_tag
            prompt = DEFAULT_SUMMARY_PROMPT.format(
                tags=tags_str,
                target_tokens=target_tokens,
                conversation_text=conversation_text,
            )

        system = (
            "You are a conversation summarizer. Output valid JSON only. "
            "No markdown fences, no extra text."
        )

        try:
            response_text = self.llm.complete(
                system=system,
                user=prompt,
                max_tokens=self.config.max_summary_tokens + self.config.llm_token_overhead,
            )
            parsed = self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"LLM summarization failed for segment {segment.id}: {e}")
            parsed = {
                "summary": conversation_text[:target_tokens * 4],
                "entities": [],
                "key_decisions": [],
                "action_items": [],
                "date_references": [],
                "refined_tags": segment.tags,
            }

        summary = parsed.get("summary", "")
        summary_tokens = self.token_counter(summary)

        # Always preserve original segment tags (source of truth from TurnTagIndex).
        # LLM refined_tags and related_tags can ADD new tags but never remove originals.
        refined_tags = parsed.get("refined_tags", [])
        related_tags = parsed.get("related_tags", [])
        all_tags = set(segment.tags)
        if refined_tags:
            all_tags.update(self._normalize_tag_list(refined_tags))
        if related_tags:
            all_tags.update(self._normalize_tag_list(related_tags))
        refined_tags = sorted(all_tags)

        metadata = SegmentMetadata(
            entities=parsed.get("entities", []),
            key_decisions=parsed.get("key_decisions", []),
            action_items=parsed.get("action_items", []),
            date_references=parsed.get("date_references", []),
            turn_count=segment.turn_count,
            time_span=(segment.start_timestamp, segment.end_timestamp),
        )

        messages_dicts = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in segment.messages
        ]

        return CompactionResult(
            segment_id=segment.id,
            primary_tag=segment.primary_tag,
            tags=refined_tags,
            summary=summary,
            summary_tokens=summary_tokens,
            original_tokens=original_tokens,
            compression_ratio=summary_tokens / original_tokens if original_tokens > 0 else 0.0,
            metadata=metadata,
            full_text=conversation_text,
            messages=messages_dicts,
        )

    @staticmethod
    def _normalize_tag_list(tags: list) -> list[str]:
        """Normalize a list of raw tag strings: lowercase, hyphenate, strip special chars."""
        import re
        result = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            t = tag.lower().strip()
            t = re.sub(r"[^a-z0-9-]", "-", t)
            t = re.sub(r"-+", "-", t).strip("-")
            if t:
                result.append(t)
        return result

    def _get_prompt_for_tags(self, tags: list[str]) -> str | None:
        """Find a custom summary prompt matching any of the tags."""
        for rule in self.tag_rules:
            for tag in tags:
                if fnmatch.fnmatch(tag, rule.match) and rule.summary_prompt:
                    return rule.summary_prompt
        return None

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format messages as 'Role (HH:MM): content' blocks."""
        lines: list[str] = []
        for m in messages:
            ts = ""
            if m.timestamp:
                ts = f" ({m.timestamp.strftime('%H:%M')})"
            lines.append(f"{m.role.capitalize()}{ts}: {m.content}")
        return "\n\n".join(lines)

    def _parse_response(self, response: str) -> dict:
        """Parse LLM JSON response with fallback for malformed output."""
        import re
        text = response.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Strip thinking tags
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return {
            "summary": response.strip(),
            "entities": [],
            "key_decisions": [],
            "action_items": [],
            "date_references": [],
            "refined_tags": [],
        }

    def compact_tag_summaries(
        self,
        cover_tags: list[str],
        tag_to_summaries: dict[str, list[StoredSummary]],
        tag_to_turns: dict[str, list[int]],
        existing_tag_summaries: dict[str, TagSummary],
        max_turn: int,
    ) -> list[TagSummary]:
        """Build or update tag summaries for cover tags.

        Only rebuilds tag summaries that are stale (``covers_through_turn`` <
        ``max_turn``) or missing.  Uses ThreadPoolExecutor for concurrency.
        """
        tags_to_build: list[str] = []
        for tag in cover_tags:
            summaries = tag_to_summaries.get(tag, [])
            if not summaries:
                continue  # no segment summaries yet
            existing = existing_tag_summaries.get(tag)
            if existing is None or existing.covers_through_turn < max_turn:
                tags_to_build.append(tag)

        if not tags_to_build:
            return []

        if len(tags_to_build) <= 1:
            return [
                self._build_one_tag_summary(
                    tag,
                    tag_to_summaries.get(tag, []),
                    tag_to_turns.get(tag, []),
                    max_turn,
                )
                for tag in tags_to_build
            ]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(self.config.max_concurrent_summaries, len(tags_to_build))
        results: list[TagSummary | None] = [None] * len(tags_to_build)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._build_one_tag_summary,
                    tag,
                    tag_to_summaries.get(tag, []),
                    tag_to_turns.get(tag, []),
                    max_turn,
                ): i
                for i, tag in enumerate(tags_to_build)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    tag = tags_to_build[idx]
                    logger.error(f"Tag summary build failed for '{tag}': {e}")
                    summaries = tag_to_summaries.get(tag, [])
                    fallback_text = "\n\n".join(s.summary for s in summaries)[:4000]
                    results[idx] = TagSummary(
                        tag=tag,
                        summary=fallback_text,
                        summary_tokens=self.token_counter(fallback_text),
                        source_segment_refs=[s.ref for s in summaries],
                        source_turn_numbers=sorted(set(tag_to_turns.get(tag, []))),
                        covers_through_turn=max_turn,
                    )

        return [r for r in results if r is not None]

    def _build_one_tag_summary(
        self,
        tag: str,
        summaries: list[StoredSummary],
        turn_numbers: list[int],
        max_turn: int,
    ) -> TagSummary:
        """Build a single tag summary by rolling up segment summaries via LLM."""
        combined = "\n\n---\n\n".join(
            f"[Segment {s.ref}, tags: {', '.join(s.tags)}]\n{s.summary}"
            for s in summaries
        )
        combined_tokens = self.token_counter(combined)
        target_tokens = max(
            self.config.min_summary_tokens,
            min(self.config.max_summary_tokens, int(combined_tokens * self.config.summary_ratio)),
        )

        prompt = TAG_SUMMARY_ROLLUP_PROMPT.format(
            tag=tag,
            count=len(summaries),
            target_tokens=target_tokens,
            segment_summaries=combined,
        )

        system = (
            "You are a conversation summarizer. Output valid JSON only. "
            "No markdown fences, no extra text."
        )

        try:
            response_text = self.llm.complete(
                system=system,
                user=prompt,
                max_tokens=self.config.max_summary_tokens + self.config.llm_token_overhead,
            )
            parsed = self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"LLM tag summary rollup failed for '{tag}': {e}")
            parsed = {"summary": combined[:4000]}

        summary_text = parsed.get("summary", "")
        return TagSummary(
            tag=tag,
            summary=summary_text,
            summary_tokens=self.token_counter(summary_text),
            source_segment_refs=[s.ref for s in summaries],
            source_turn_numbers=sorted(set(turn_numbers)),
            covers_through_turn=max_turn,
        )
