"""DomainCompactor: summarizes TaggedSegments using a configurable LLM."""

from __future__ import annotations

import fnmatch
import json
import logging
from typing import Callable

from ..types import (
    CompactionResult,
    CompactorConfig,
    Fact,
    FactSignal,
    LLMProvider,
    Message,
    SegmentMetadata,
    StoredSummary,
    TaggedSegment,
    TagPromptRule,
    TagSummary,
    TemporalStatus,
)
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)

TAG_SUMMARY_ROLLUP_PROMPT = """\
You are summarizing all stored context about the tag "{tag}".
Below are {count} segment summaries that each cover a portion of conversation
where "{tag}" was discussed. Roll them up into a single coherent summary that
preserves all key decisions, action items, entities, and names.
Keep the chronological progression. The summary should be comprehensive enough
that someone could resume the conversation from it.

CRITICAL — Any text involving numbers is mandatory and absolutely essential to the summary, always include them exactly as in the source.
Dates, prices, any number is important and should not be modified.
Never round, approximate, or paraphrase a number.

Also provide a "description": a concise paragraph (max 80 words) capturing who is
involved, what is being discussed, key facts, and distinctive details. This will
be shown as a topic label in a compact topic list — it is the reader's primary
way to decide which topics are relevant, so include enough detail to be useful.
Any text involving numbers is mandatory and absolutely essential to the description.
Dates, prices, any number is important and should not be modified.
Never round, approximate, or paraphrase a number (e.g. "2 hours" must stay "2 hours", not
"about an hour"; "$45" must stay "$45", not "around $50").
Always preserve the user's role and relationship to the topic — phrases like
"I led", "my project", "solo project", "I built", "I'm responsible for" are
critical personal context. Never paraphrase these into passive voice or drop them
in favor of technical details.

Target length: {target_tokens} tokens or fewer.

Segment summaries:
{segment_summaries}

Respond with JSON:
{{
  "summary": "...",
  "description": "concise topic paragraph, max 80 words",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."]
}}"""


DEFAULT_SUMMARY_PROMPT = """\
Summarize the following conversation segment (tags: {tags}).
Preserve: key decisions, action items, entities mentioned, specific data points,
and specific feature/concept names exactly as discussed (e.g. "cook mode", "dark theme", "rate limiter" —
do NOT generalize these into broader categories like "UI features" or "infrastructure").

CRITICAL — Any text involving numbers is mandatory and absolutely essential to the summary, always include them exactly as in the conversation.
Dates, prices, any number is important and should not be modified.
Never round, approximate, or paraphrase a number (e.g. "2 hours" must stay "2 hours", not
"about an hour"; "$45" must stay "$45", not "around $50").

When the user states what they are doing, have done, or where they keep/store something,
preserve that as a direct assertion, not as a plan or intention.
For example, "I'm storing my sneakers in a shoe rack" should be summarized as
"User stores/is storing sneakers in shoe rack", NOT "User plans to store sneakers in shoe rack."

Always preserve the user's role and relationship to the topic — phrases like
"I led", "my project", "solo project", "I built", "I'm responsible for" are
critical personal context. Never paraphrase these into passive voice or drop them
in favor of technical details.

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
"caching", "precomputed", "feed-optimization", "query-cache").

Also extract facts from the conversation. For each fact:
- "subject": who (usually "user"; proper names for others)
- "verb": the exact action verb (e.g. "led", "built", "prefers", "lives in", "ordered")
- "object": what (specific noun phrase)
- "status": one of: active, completed, planned, abandoned, recurring
- "what": one-sentence summary of the fact
- "who": people involved (omit if n/a)
- "when": date if mentioned (ISO format, omit if n/a)
- "where": location (omit if n/a)
- "why": context/significance (omit if n/a)
Include "facts" in the JSON response.
Only extract facts with genuine substance. Skip greetings and filler."""


class DomainCompactor:
    """Summarize each TaggedSegment independently using an LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: CompactorConfig,
        token_counter: Callable[[str], int] | None = None,
        model_name: str = "",
        tag_rules: list[TagPromptRule] | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.llm = llm_provider
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self.model_name = model_name
        self.tag_rules = tag_rules or []
        self._cost_tracker = cost_tracker

    def compact(
        self,
        segments: list[TaggedSegment],
        fact_signals_by_segment: dict[str, list[FactSignal]] | None = None,
    ) -> list[CompactionResult]:
        """Summarize each segment independently.

        Uses ThreadPoolExecutor for concurrent summarization when there are
        multiple segments. Falls back to sequential for single segments.

        *fact_signals_by_segment* maps segment.id → list of FactSignal
        collected from per-turn tagging.  Passed as hints to the compactor
        prompt for verification and consolidation.
        """
        signals = fact_signals_by_segment or {}
        logger.info(
            "Compacting %d segments (%d workers)...",
            len(segments), min(self.config.max_concurrent_summaries, len(segments)),
        )
        if len(segments) <= 1:
            # Sequential for single segment — no threading overhead
            return [self._compact_one(s, signals.get(s.id, [])) for s in segments]

        max_workers = min(self.config.max_concurrent_summaries, len(segments))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[CompactionResult] = [None] * len(segments)  # type: ignore[list-item]
        done_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._compact_one, segment, signals.get(segment.id, [])): i
                for i, segment in enumerate(segments)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                done_count += 1
                try:
                    results[idx] = future.result()
                    logger.info(
                        "  Segment %d/%d done (%s, %dt → %dt)",
                        done_count, len(segments),
                        results[idx].primary_tag,
                        results[idx].original_tokens,
                        results[idx].summary_tokens,
                    )
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

    def _compact_one(
        self, segment: TaggedSegment, fact_signals: list[FactSignal] | None = None,
    ) -> CompactionResult:
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

        # D1: build signal hints from per-turn tagger fact signals
        signals_text = ""
        if fact_signals:
            hint_lines = []
            for s in fact_signals:
                if s.subject and s.object:
                    hint_lines.append(f"- {s.subject} {s.verb} {s.object} ({s.status})")
            if hint_lines:
                signals_text = (
                    "\n\nPer-turn fact signals (verify and consolidate with full context):\n"
                    + "\n".join(hint_lines)
                )

        if custom_prompt:
            prompt = (
                f"{custom_prompt}\n\n"
                f"Target length: {target_tokens} tokens or fewer.\n\n"
                f"Conversation:\n{conversation_text}"
                f"{signals_text}\n\n"
                'Respond with JSON: {{"summary": "...", "entities": ["..."], '
                '"key_decisions": ["..."], "action_items": ["..."], '
                '"date_references": ["..."], "refined_tags": ["tag1", "tag2"], '
                '"facts": [...]}}'
            )
        else:
            tags_str = ", ".join(segment.tags) if segment.tags else segment.primary_tag
            prompt = DEFAULT_SUMMARY_PROMPT.format(
                tags=tags_str,
                target_tokens=target_tokens,
                conversation_text=conversation_text,
            )
            if signals_text:
                prompt += signals_text

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
            self._log_usage("compaction")
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
            session_date=segment.session_date,
        )

        messages_dicts = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in segment.messages
        ]

        # D1: Parse extracted facts
        valid_statuses = {e.value for e in TemporalStatus}
        raw_facts = parsed.get("facts", [])
        facts: list[Fact] = []
        if isinstance(raw_facts, list):
            for f in raw_facts:
                if not isinstance(f, dict) or not f.get("subject") or not f.get("object"):
                    continue
                status = f.get("status", "active")
                if status not in valid_statuses:
                    status = "active"
                def _str(val) -> str:
                    if isinstance(val, list):
                        return ", ".join(str(v) for v in val)
                    return str(val) if val else ""

                facts.append(Fact(
                    subject=_str(f.get("subject", "")),
                    verb=_str(f.get("verb", f.get("role", ""))),
                    object=_str(f.get("object", "")),
                    status=status,
                    what=_str(f.get("what", "")),
                    who=_str(f.get("who", "")),
                    when_date=_str(f.get("when", "")),
                    where=_str(f.get("where", "")),
                    why=_str(f.get("why", "")),
                    tags=refined_tags,
                ))

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
            facts=facts,
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

    def _log_usage(self, event_type: str) -> None:
        """Log LLM token usage from the provider's last_usage to the cost tracker."""
        if not self._cost_tracker:
            return
        usage = getattr(self.llm, "last_usage", {})
        if not usage:
            return
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        if input_tokens or output_tokens:
            self._cost_tracker.log_compaction(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider=self.model_name,
            )

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

        logger.info("Building tag summaries for %d tags: %s", len(tags_to_build), tags_to_build[:10])

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
            done_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                done_count += 1
                try:
                    results[idx] = future.result()
                    logger.info(
                        "  Tag summary %d/%d done (%s, %dt)",
                        done_count, len(tags_to_build),
                        tags_to_build[idx], results[idx].summary_tokens,
                    )
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
            self._log_usage("compaction")
            parsed = self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"LLM tag summary rollup failed for '{tag}': {e}")
            parsed = {"summary": combined[:4000]}

        summary_text = parsed.get("summary", "")
        description = parsed.get("description", "")
        return TagSummary(
            tag=tag,
            summary=summary_text,
            description=description,
            summary_tokens=self.token_counter(summary_text),
            source_segment_refs=[s.ref for s in summaries],
            source_turn_numbers=sorted(set(turn_numbers)),
            covers_through_turn=max_turn,
        )
