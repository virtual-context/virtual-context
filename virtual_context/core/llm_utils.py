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


def normalize_code_refs(
    raw_refs: object,
    *,
    max_refs: int = 12,
    max_file_len: int = 256,
    max_symbol_len: int = 128,
) -> list[dict]:
    """Normalize LLM-emitted code references into a stable, compact shape.

    Accepted input items are dict-like objects with a required ``file`` or
    ``path`` key and optional ``line`` and ``symbol``/``name``/``function``/
    ``class`` fields. Invalid items are dropped, refs are deduplicated, and the
    final list is capped to keep prompt bloat under control.
    """
    if not isinstance(raw_refs, list):
        return []

    normalized: list[dict] = []
    seen: set[tuple[str, int | None, str]] = set()

    for item in raw_refs:
        if not isinstance(item, dict):
            continue

        file_ref = item.get("file") or item.get("path")
        if not isinstance(file_ref, str):
            continue
        file_ref = file_ref.strip()
        if not file_ref:
            continue
        file_ref = file_ref[:max_file_len]

        line_value = item.get("line")
        line: int | None = None
        if isinstance(line_value, int):
            line = line_value if line_value > 0 else None
        elif isinstance(line_value, str):
            stripped = line_value.strip()
            if stripped.isdigit():
                parsed = int(stripped)
                line = parsed if parsed > 0 else None

        symbol_value = (
            item.get("symbol")
            or item.get("name")
            or item.get("function")
            or item.get("class")
        )
        symbol = symbol_value.strip()[:max_symbol_len] if isinstance(symbol_value, str) else ""

        dedupe_key = (file_ref, line, symbol)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        entry = {"file": file_ref}
        if line is not None:
            entry["line"] = line
        if symbol:
            entry["symbol"] = symbol
        normalized.append(entry)
        if len(normalized) >= max_refs:
            break

    return normalized


def format_code_ref(ref: dict) -> str:
    """Render a normalized code-ref dict as ``file[:line] [symbol]``."""
    file_ref = ref.get("file", "")
    line = ref.get("line")
    symbol = ref.get("symbol", "")
    label = file_ref
    if isinstance(line, int) and line > 0:
        label = f"{label}:{line}"
    if symbol:
        label = f"{label} {symbol}".strip()
    return label.strip()
