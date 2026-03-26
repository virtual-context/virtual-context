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
    fact_counts: dict[str, int] | None = None,
    max_tool_rounds: int = 10,
) -> str:
    """Build compact autonomous paging hint with two-tier layout.

    Expanded tags (in working set) listed first with full metadata.
    Available tags (depth:none) listed compactly below.
    Truncation drops available tags first, preserving expanded tags.
    """
    used = sum(ws.tokens for ws in working_set.values())
    fc = fact_counts or {}

    # Partition into expanded (in working set) vs available (depth:none)
    expanded_lines: list[str] = []
    available_entries: list[str] = []
    for ts in tag_summaries:
        n_facts = fc.get(ts.tag, 0)
        facts_label = f", {n_facts} facts" if n_facts > 0 else ""
        ws = working_set.get(ts.tag)
        if ws and ws.depth != DepthLevel.NONE:
            full_t = calculate_depth_tokens(ts.tag, DepthLevel.FULL)
            desc_part = f" — {ts.description}" if ts.description else ""
            expanded_lines.append(
                f"  {ts.tag}: {ws.depth.value} {ws.tokens}t"
                f" \u2192 {full_t}t full{facts_label}{desc_part}"
            )
        else:
            full_t = calculate_depth_tokens(ts.tag, DepthLevel.FULL)
            entry = ts.tag
            if full_t > 0:
                entry += f"({full_t}t{facts_label})"
            elif facts_label:
                entry += f"({facts_label.lstrip(', ')})"
            if ts.description:
                entry += f" — {ts.description}"
            available_entries.append(entry)

    _RULES = (
        "RULE: These are compressed summaries — Summaries DO omit details.\n"
        "To find detailed information you have the following tools:\n"
        "- vc_restore_tool(ref): ALL compacted tool turns can be restored to "
        "full fidelity — thinking, tool calls, and raw output. Use the ref "
        "from the compacted stub. This is the fastest way to get exact data.\n"
        "  MANDATORY: If the user asks for exact values (file paths, line counts, "
        "code, error messages, search results, command output) and the relevant "
        "turn is compacted, you MUST restore before answering. Summaries never "
        "contain these details — do not guess or hallucinate them.\n"
        "- vc_find_quote(query): search raw text across ALL topics.\n"
        "- vc_query_facts(subject?, verb?, status?, object_contains?): "
        "structured fact lookup.\n"
        "- vc_expand_topic(tag, collapse_tags?): load original text for a "
        "topic. Pass collapse_tags to free budget from other topics in the "
        "same call.\n"
        "- vc_remember_when(query, time_range): time-scoped recall.\n"
        "- vc_recall_all(): load every summary at once.\n"
        f"You have a maximum of {max_tool_rounds} tool rounds. "
        "Plan your strategy upfront: use diverse queries, not repetitions.\n"
        "For counting/listing questions: scan [all topics] for every topic "
        "that could relate — items are often spread across unrelated topics.\n"
        "If a search already returned the answer, stop and respond.\n"
        "Never answer without searching first.\n"
        "FACT vs SUMMARY: The <facts> block contains structured events with "
        "statuses (completed, planned, active). Summaries describe topics "
        "DISCUSSED — they include plans, itineraries, and ideas that may "
        "never have been executed. For questions about what the user DID or "
        "experienced, trust facts with status=completed. Do not treat "
        "detailed planning discussions (itineraries, schedules, bookings) "
        "as evidence that a trip or event actually happened.\n"
        "FACT FORMAT: Each fact is rendered as: "
        "subject | verb | object — description [when: YYYY-MM-DD]\n"
        "  • [when: ...] is the ACTUAL CALENDAR DATE of the event.\n"
        "  • [session: ...] is the date of the conversation session "
        "where the fact was discussed (fallback when exact event date is unknown).\n"
        "  • Words like 'yesterday', 'last week' in descriptions and summaries "
        "are RELATIVE TO THE CONVERSATION DATE — the questioner was not part of "
        "the conversation and cannot interpret them. Resolve relative time "
        "references to actual dates using [when:] or [session:] metadata."
    )

    _COMPACT_RULES = (
        "RULE: Summaries DO omit details. "
        "Tools: find_quote, query_facts, expand_topic(collapse_tags?), "
        "remember_when, recall_all. "
        f"Max {max_tool_rounds} tool rounds — be strategic. "
        "Scan [all topics]. Never answer without searching first."
    )

    # Compact name-only list of ALL tags — never truncated.
    all_tag_names = [ts.tag for ts in tag_summaries]
    all_topics_line = (
        f"[all {len(all_tag_names)} topics] "
        + ", ".join(all_tag_names)
        + "\nScan before answering — relevant context may be under "
        "an unexpected topic name."
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
        parts.append("")
        parts.append(all_topics_line)
        body = "\n".join(parts)
        rules = _COMPACT_RULES if compact else _RULES
        return (
            f'<context-topics budget="{budget}" used="{used}"'
            f' available="{budget - used}">\n'
            f"{rules}\n\n"
            f"{body}\n\n"
            f"Tools: find_quote(query) | query_facts(subject?, verb?, status?, object_contains?) | "
            f"recall_all() | remember_when(query, time_range) | "
            f"expand_topic(tag, depth?, collapse_tags?)\n"
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
    max_tool_rounds: int = 10,
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

    # Compact name-only list of ALL tags — never truncated.
    all_tag_names = [ts.tag for ts in tag_summaries]
    all_topics_line = (
        f"[all {len(all_tag_names)} topics] "
        + ", ".join(all_tag_names)
        + "\nScan before answering — relevant context may be under "
        "an unexpected topic name."
    )

    def _assemble(exp_lines: list[str], avail: list[str]) -> str:
        parts: list[str] = []
        if exp_lines:
            parts.append("[in context]")
            parts.extend(exp_lines)
        if avail:
            if exp_lines:
                parts.append("")
            parts.append("[available] " + ", ".join(avail))
        parts.append("")
        parts.append(all_topics_line)
        body = "\n".join(parts)
        return (
            "<context-topics>\n"
            "RULE: These are compressed summaries — Summaries DO omit details.\n"
            "To find detailed information you have the following tools:\n"
            "- vc_find_quote(query): search raw text across ALL topics.\n"
            "- vc_query_facts(subject?, verb?, status?, object_contains?): "
            "structured fact lookup.\n"
            "- vc_expand_topic(tag, collapse_tags?): load original text for a "
            "topic. Pass collapse_tags to free budget from other topics in the "
            "same call.\n"
            "- vc_remember_when(query, time_range): time-scoped recall.\n"
            "- vc_recall_all(): load every summary at once.\n"
            f"You have a maximum of {max_tool_rounds} tool rounds. "
            "Plan your strategy upfront: use diverse queries, not repetitions. "
            "If a search already returned the answer, stop and respond.\n"
            "For counting/listing questions: scan [all topics] for every topic "
            "that could relate — items are often spread across unrelated topics.\n"
            "Never answer without searching first.\n"
            "FACT vs SUMMARY: The <facts> block contains structured events with "
            "statuses (completed, planned, active). Summaries describe topics "
            "DISCUSSED — they include plans, itineraries, and ideas that may "
            "never have been executed. For questions about what the user DID or "
            "experienced, trust facts with status=completed. Do not treat "
            "detailed planning discussions (itineraries, schedules, bookings) "
            "as evidence that a trip or event actually happened.\n"
            "FACT FORMAT: Each fact is rendered as: "
            "subject | verb | object — description [when: YYYY-MM-DD]\n"
            "  • [when: ...] is the ACTUAL CALENDAR DATE of the event.\n"
            "  • [session: ...] is the date of the conversation session "
            "where the fact was discussed (fallback when exact event date is unknown).\n"
            "  • Words like 'yesterday', 'last week' in descriptions and summaries "
            "are RELATIVE TO THE CONVERSATION DATE — the questioner was not part of "
            "the conversation and cannot interpret them. Resolve relative time "
            "references to actual dates using [when:] or [session:] metadata.\n\n"
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
    # Compact name-only list of ALL tags — never truncated.
    all_tag_names = [ts.tag for ts in tag_summaries]
    all_topics_line = (
        f"[all {len(all_tag_names)} topics] "
        + ", ".join(all_tag_names)
        + "\nScan before answering — relevant context may be under "
        "an unexpected topic name."
    )

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
        f"{body}\n\n"
        f"{all_topics_line}\n"
        "</context-topics>"
    )

    if token_counter(hint) > max_hint_tokens:
        while lines and token_counter(hint) > max_hint_tokens:
            lines.pop()
            body = "\n".join(lines)
            hint = (
                "<context-topics>\n"
                "Prior conversation topics available for recall:\n"
                f"{body}\n\n"
                f"{all_topics_line}\n"
                "</context-topics>"
            )

    return hint
