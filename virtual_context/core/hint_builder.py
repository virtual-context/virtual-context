"""Context hint builder: renders topic lists for post-compaction prompts.

Pure functions — no engine state mutation. Extracted from engine.py.
"""

from __future__ import annotations

from typing import Callable

from ..types import DepthLevel, TagSummary, WorkingSetEntry


def build_autonomous_hint(
    tag_summaries: list[TagSummary],
    working_set: dict[str, WorkingSetEntry],
    budget: int,
    max_hint_tokens: int,
    token_counter: Callable[[str], int],
    calculate_depth_tokens: Callable[[str, DepthLevel], int],
) -> str:
    """Build compact autonomous paging hint with two-tier layout.

    Expanded tags (in working set) listed first with full metadata.
    Available tags (depth:none) listed compactly below.
    Truncation drops available tags first, preserving expanded tags.
    """
    used = sum(ws.tokens for ws in working_set.values())

    # Partition into expanded (in working set) vs available (depth:none)
    expanded_lines: list[str] = []
    available_entries: list[str] = []
    for ts in tag_summaries:
        ws = working_set.get(ts.tag)
        if ws and ws.depth != DepthLevel.NONE:
            full_t = calculate_depth_tokens(ts.tag, DepthLevel.FULL)
            desc_part = f" — {ts.description}" if ts.description else ""
            expanded_lines.append(
                f"  {ts.tag}: {ws.depth.value} {ws.tokens}t"
                f" \u2192 {full_t}t full{desc_part}"
            )
        else:
            full_t = calculate_depth_tokens(ts.tag, DepthLevel.FULL)
            entry = ts.tag
            if full_t > 0:
                entry += f"({full_t}t)"
            if ts.description:
                entry += f" — {ts.description}"
            available_entries.append(entry)

    _RULES = (
        "RULE: These are compressed topic summaries, not the full conversation.\n"
        "- For specific facts (names, numbers, dosages, decisions): "
        "use vc_find_quote — it searches raw text across all topics.\n"
        "- For broad questions (summarize, put together, plan, walk me through, itinerary): "
        "use vc_expand_topic to load full conversation detail before answering.\n"
        "- For deeper understanding of a topic: "
        "use vc_expand_topic to load the full conversation text.\n"
        "- To free budget after expanding: use vc_collapse_topic.\n"
        "- Never claim you don't remember without searching first.\n"
        "- Never give a vague answer when you could expand a topic for specifics."
    )

    _COMPACT_RULES = (
        "RULE: Compressed summaries. Use vc_find_quote for facts, "
        "vc_expand_topic for detail, vc_collapse_topic to free budget."
    )

    def _assemble(exp_lines: list[str], avail: list[str], *, compact: bool = False) -> str:
        parts: list[str] = []
        if exp_lines:
            parts.append("[in context \u2014 expand for full detail]")
            parts.extend(exp_lines)
        if avail:
            if exp_lines:
                parts.append("")
            parts.append(
                "[available] " + ", ".join(avail)
            )
        body = "\n".join(parts)
        rules = _COMPACT_RULES if compact else _RULES
        return (
            f'<context-topics budget="{budget}" used="{used}"'
            f' available="{budget - used}">\n'
            f"{rules}\n\n"
            f"{body}\n\n"
            f"Tools: find_quote(query) | expand_topic(tag, depth?) | collapse_topic(tag, depth?)\n"
            f"</context-topics>"
        )

    hint = _assemble(expanded_lines, available_entries)

    # Truncate: drop available entries first, then expanded lines
    if token_counter(hint) > max_hint_tokens:
        while available_entries and token_counter(hint) > max_hint_tokens:
            available_entries.pop()
            hint = _assemble(expanded_lines, available_entries)
        # If expanded lines still don't fit, switch to compact boilerplate
        # before dropping any expanded lines — tags are more valuable than rules.
        if expanded_lines and token_counter(hint) > max_hint_tokens:
            hint = _assemble(expanded_lines, available_entries, compact=True)
        while expanded_lines and token_counter(hint) > max_hint_tokens:
            expanded_lines.pop()
            hint = _assemble(expanded_lines, available_entries, compact=True)

    return hint


def build_supervised_hint(
    tag_summaries: list[TagSummary],
    working_set: dict[str, WorkingSetEntry],
    max_hint_tokens: int,
    token_counter: Callable[[str], int],
) -> str:
    """Build compact supervised paging hint.

    Expanded tags first with depth info, available tags as compact list.
    """
    expanded_lines: list[str] = []
    available_entries: list[str] = []
    for ts in tag_summaries:
        ws = working_set.get(ts.tag)
        if ws and ws.depth != DepthLevel.NONE:
            desc = ts.description or ts.summary[:60].rstrip()
            if not ts.description and len(ts.summary) > 60:
                desc += "..."
            expanded_lines.append(
                f"  {ts.tag} ({ws.depth.value}, {ws.tokens}t): {desc}"
            )
        else:
            entry = ts.tag
            if ts.description:
                entry += f" — {ts.description}"
            available_entries.append(entry)

    def _assemble(exp_lines: list[str], avail: list[str]) -> str:
        parts: list[str] = []
        if exp_lines:
            parts.append("[in context]")
            parts.extend(exp_lines)
        if avail:
            if exp_lines:
                parts.append("")
            parts.append("[available] " + ", ".join(avail))
        body = "\n".join(parts)
        return (
            "<context-topics>\n"
            "RULE: These are compressed topic summaries, not the full conversation.\n"
            "- For specific facts (names, numbers, dosages, decisions): "
            "use vc_find_quote — it searches raw text across all topics.\n"
            "- For broad questions (summarize, put together, plan, walk me through, itinerary): "
            "use vc_expand_topic to load full conversation detail before answering.\n"
            "- For deeper understanding of a topic: "
            "use vc_expand_topic to load the full conversation text.\n"
            "- To free budget after expanding: use vc_collapse_topic.\n"
            "- Never claim you don't remember without searching first.\n"
            "- Never give a vague answer when you could expand a topic for specifics.\n\n"
            f"{body}\n"
            "</context-topics>"
        )

    hint = _assemble(expanded_lines, available_entries)

    if token_counter(hint) > max_hint_tokens:
        while available_entries and token_counter(hint) > max_hint_tokens:
            available_entries.pop()
            hint = _assemble(expanded_lines, available_entries)
        while expanded_lines and token_counter(hint) > max_hint_tokens:
            expanded_lines.pop()
            hint = _assemble(expanded_lines, available_entries)

    return hint


def build_default_hint(
    tag_summaries: list[TagSummary],
    max_hint_tokens: int,
    token_counter: Callable[[str], int],
) -> str:
    """Build simple topic list (no paging)."""
    lines: list[str] = []
    for ts in tag_summaries:
        turn_count = len(ts.source_turn_numbers)
        desc = ts.description or ts.summary[:60].rstrip()
        if not ts.description and len(ts.summary) > 60:
            desc += "..."
        lines.append(f"- {ts.tag} ({turn_count} turns): {desc}")

    body = "\n".join(lines)
    hint = (
        "<context-topics>\n"
        "Prior conversation topics available for recall:\n"
        f"{body}\n"
        "</context-topics>"
    )

    if token_counter(hint) > max_hint_tokens:
        while lines and token_counter(hint) > max_hint_tokens:
            lines.pop()
            body = "\n".join(lines)
            hint = (
                "<context-topics>\n"
                "Prior conversation topics available for recall:\n"
                f"{body}\n"
                "</context-topics>"
            )

    return hint
