"""TagSplitter: LLM-based splitting of overly-broad tags into specific sub-tags."""

from __future__ import annotations

import json
import logging
import re

from ..types import LLMProvider, SplitResult, TagSplittingConfig

logger = logging.getLogger(__name__)

TAG_SPLIT_SYSTEM_PROMPT = """\
You are refining a tag vocabulary for a conversation memory system.
Output valid JSON only. No markdown fences."""

TAG_SPLIT_USER_PROMPT = """\
The tag "{tag}" appears on {count} out of {total} conversation turns.
Analyze whether these turns cover multiple distinct sub-topics or one
uniform topic.

Turns tagged with "{tag}":
{turn_list}

Step 1 — Decide: Do these turns cover 2 or more distinct sub-topics?

If NO (all turns are about the same specific topic):
  Respond: {{"splittable": false, "reason": "..."}}

If YES (turns span distinct sub-topics):
  Group turns by sub-topic. For each group, create a NEW compound tag.
  - Format: "{{domain}}-{{qualifier}}" (e.g., "resy-troubleshooting", "browser-debugging")
  - NEVER reuse an existing tag name — always create a new compound tag
  - Each turn belongs to exactly one group
  - Minimum 2 groups
  Respond: {{"splittable": true, "groups": {{"new-tag": [turn_numbers_as_integers], ...}}}}
  IMPORTANT: Turn numbers must be plain integers (e.g., 9, 13, 20), NOT strings like "T9"."""


class TagSplitter:
    """Analyze overly-broad tags and split them into specific sub-tags via LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        config: TagSplittingConfig,
    ) -> None:
        self.llm = llm
        self.config = config

    def split(
        self,
        tag: str,
        turn_contents: list[tuple[int, str]],
        existing_tags: set[str],
        total_turns: int,
    ) -> SplitResult:
        """Attempt to split a broad tag into specific sub-tags.

        Args:
            tag: The broad tag to split.
            turn_contents: [(turn_number, truncated_user_text), ...].
            existing_tags: Tags already in TurnTagIndex (for collision detection).
            total_turns: Total number of turns in the index.

        Returns:
            SplitResult with groups if splittable, reason if not.
        """
        count = len(turn_contents)

        # Build turn list for prompt
        turn_lines = []
        for turn_num, text in turn_contents:
            truncated = text[:200].replace("\n", " ")
            turn_lines.append(f"[T{turn_num}] {truncated}")
        turn_list = "\n".join(turn_lines)

        prompt = TAG_SPLIT_USER_PROMPT.format(
            tag=tag,
            count=count,
            total=total_turns,
            turn_list=turn_list,
        )

        try:
            response = self.llm.complete(
                system=TAG_SPLIT_SYSTEM_PROMPT,
                user=prompt,
                max_tokens=2048,
            )
            return self._parse_response(tag, response, turn_contents, existing_tags)
        except Exception as e:
            logger.warning("Tag split LLM call failed for '%s': %s", tag, e)
            return SplitResult(tag=tag, splittable=False, reason=f"LLM error: {e}")

    def _parse_response(
        self,
        tag: str,
        response: str,
        turn_contents: list[tuple[int, str]],
        existing_tags: set[str],
    ) -> SplitResult:
        """Parse LLM response into a validated SplitResult."""
        text = response.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Strip thinking tags
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return SplitResult(tag=tag, splittable=False, reason="JSON parse error")
            else:
                return SplitResult(tag=tag, splittable=False, reason="No JSON found")

        splittable = data.get("splittable", False)
        if not splittable:
            return SplitResult(
                tag=tag,
                splittable=False,
                reason=data.get("reason", "LLM determined unsplittable"),
            )

        groups_raw = data.get("groups", {})
        if not isinstance(groups_raw, dict) or len(groups_raw) < 2:
            return SplitResult(
                tag=tag,
                splittable=False,
                reason="Fewer than 2 groups returned",
            )

        # Validate and normalize
        valid_turn_nums = {tn for tn, _ in turn_contents}
        groups: dict[str, list[int]] = {}

        for new_tag, turn_nums in groups_raw.items():
            # Normalize tag name
            normalized = self._normalize_tag(new_tag)
            if not normalized:
                continue

            # Collision detection: append suffix if tag already exists
            final_tag = normalized
            if final_tag in existing_tags or final_tag == tag:
                final_tag = f"{normalized}-split"
                if final_tag in existing_tags:
                    final_tag = f"{normalized}-{tag}"

            # Validate turn numbers (handle int, float, "9", "T9" formats)
            if not isinstance(turn_nums, list):
                continue
            valid_nums = []
            for n in turn_nums:
                try:
                    if isinstance(n, (int, float)):
                        num = int(n)
                    elif isinstance(n, str):
                        num = int(n.lstrip("Tt"))
                    else:
                        continue
                    if num in valid_turn_nums:
                        valid_nums.append(num)
                except (ValueError, TypeError):
                    continue
            if valid_nums:
                groups[final_tag] = valid_nums

        if len(groups) < 2:
            return SplitResult(
                tag=tag,
                splittable=False,
                reason="Fewer than 2 valid groups after validation",
            )

        return SplitResult(tag=tag, splittable=True, groups=groups)

    @staticmethod
    def _normalize_tag(tag: str) -> str:
        """Normalize a tag name: lowercase, hyphenated, no special chars."""
        tag = tag.lower().strip()
        tag = re.sub(r"[^a-z0-9-]", "-", tag)
        tag = re.sub(r"-+", "-", tag).strip("-")
        return tag
