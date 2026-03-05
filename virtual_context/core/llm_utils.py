"""Shared LLM response parsing and tag normalization utilities.

Centralizes logic that was previously duplicated across compactor,
tag_generator, tag_consolidator, and tag_splitter.
"""

from __future__ import annotations

import json
import re


def parse_llm_json(response: str) -> dict:
    """Parse an LLM response expected to contain JSON.

    Handles common LLM output quirks:
    - Strips markdown code fences (```json ... ```)
    - Removes <think>...</think> tags (e.g. from Qwen3)
    - Extracts the first JSON object from surrounding text

    Returns the parsed dict, or an empty dict on failure.
    """
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

    # Try to extract JSON object from surrounding text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {}


def normalize_tag(tag: str) -> str:
    """Normalize a tag: lowercase, replace non-alphanumeric with hyphens, collapse runs.

    Examples:
        "My Tag"   -> "my-tag"
        "foo__bar" -> "foo-bar"
        " --hi-- " -> "hi"
    """
    tag = tag.lower().strip()
    tag = re.sub(r"[^a-z0-9-]", "-", tag)
    tag = re.sub(r"-+", "-", tag).strip("-")
    return tag
