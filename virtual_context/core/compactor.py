"""DomainCompactor: summarizes DomainSegments using a configurable LLM."""

from __future__ import annotations

import json
import logging
from typing import Callable

from ..types import (
    CompactionResult,
    CompactorConfig,
    DomainSegment,
    LLMProvider,
    Message,
    SegmentMetadata,
)

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_PROMPT = """\
Summarize the following {domain} conversation segment.
Preserve: key decisions, action items, entities mentioned, specific data points (numbers, dates, names).
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
  "date_references": ["..."]
}}"""


class DomainCompactor:
    """Summarize each DomainSegment independently using an LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: CompactorConfig,
        token_counter: Callable[[str], int] | None = None,
        model_name: str = "",
    ) -> None:
        self.llm = llm_provider
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self.model_name = model_name

    async def compact(self, segments: list[DomainSegment]) -> list[CompactionResult]:
        """Summarize each segment independently. MVP: sequential."""
        results: list[CompactionResult] = []
        for segment in segments:
            result = await self._compact_one(segment)
            results.append(result)
        return results

    async def _compact_one(self, segment: DomainSegment) -> CompactionResult:
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

        prompt = DEFAULT_SUMMARY_PROMPT.format(
            domain=segment.domain,
            target_tokens=target_tokens,
            conversation_text=conversation_text,
        )

        system = (
            "You are a conversation summarizer. Output valid JSON only. "
            "No markdown fences, no extra text."
        )

        try:
            response_text = await self.llm.complete(
                system=system,
                user=prompt,
                max_tokens=self.config.max_summary_tokens + 200,
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
            }

        summary = parsed.get("summary", "")
        summary_tokens = self.token_counter(summary)

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
            domain=segment.domain,
            summary=summary,
            summary_tokens=summary_tokens,
            original_tokens=original_tokens,
            compression_ratio=summary_tokens / original_tokens if original_tokens > 0 else 0.0,
            metadata=metadata,
            full_text=conversation_text,
            messages=messages_dicts,
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
        text = response.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

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
        }
