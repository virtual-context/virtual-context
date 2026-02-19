"""ContextAssembler: build final context from core files + tag summaries + conversation."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Callable

from ..types import (
    AssembledContext,
    AssemblerConfig,
    DepthLevel,
    Message,
    RetrievalResult,
    StoredSegment,
    StoredSummary,
    TagPromptRule,
    WorkingSetEntry,
)


class ContextAssembler:
    """Assemble enriched context within token budget.

    Assembly order (top to bottom in final prompt):
    1. [CORE CONTEXT] - always-on files (SOUL.md, USER.md, etc.)
    2. [TAG CONTEXT] - retrieved summaries in <virtual-context> tags
    3. [CONVERSATION HISTORY] - recent turns, most recent at bottom
    """

    def __init__(
        self,
        config: AssemblerConfig,
        token_counter: Callable[[str], int] | None = None,
        tag_rules: list[TagPromptRule] | None = None,
    ) -> None:
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self.tag_rules = tag_rules or []

    def assemble(
        self,
        core_context: str,
        retrieval_result: RetrievalResult,
        conversation_history: list[Message],
        token_budget: int,
        context_hint: str = "",
        working_set: dict[str, WorkingSetEntry] | None = None,
        full_segments: dict[str, list[StoredSegment]] | None = None,
        max_context_tokens: int | None = None,
    ) -> AssembledContext:
        """Build final context within token budget.

        When working_set is provided, tags are served at their working set depth:
        - NONE: skip (hint only)
        - SUMMARY: tag summary (current default)
        - SEGMENTS: individual segment summaries
        - FULL: StoredSegment.full_text
        When working_set is None, all tags served as SUMMARY (backward compat).

        max_context_tokens: If set, caps the total VC context (core + hint + tags)
        to fit within available headroom. Used by proxy to prevent exceeding
        the upstream model's context limit.
        """
        core_budget = self.config.core_context_max_tokens
        tag_budget = self.config.tag_context_max_tokens

        # Truncate core context to budget
        core = self._truncate_core(core_context, core_budget)
        core_tokens = self.token_counter(core)

        # Context hint tokens
        hint_tokens = self.token_counter(context_hint) if context_hint else 0

        # Headroom cap: if max_context_tokens is set, reduce tag budget so
        # total VC context (core + hint + tags) fits within available headroom.
        if max_context_tokens is not None:
            available_for_tags = max_context_tokens - core_tokens - hint_tokens
            tag_budget = max(0, min(tag_budget, available_for_tags))

        # Build tag sections
        tag_sections: dict[str, str] = {}
        tag_tokens = 0

        # Group summaries by primary_tag
        summaries_by_tag: dict[str, list[StoredSummary]] = {}
        for s in retrieval_result.summaries:
            summaries_by_tag.setdefault(s.primary_tag, []).append(s)

        # Also collect tags from full_segments that might not be in summaries
        if full_segments:
            for tag in full_segments:
                if tag not in summaries_by_tag:
                    summaries_by_tag[tag] = []

        # Sort tags by priority (higher priority first)
        sorted_tags = sorted(
            summaries_by_tag.keys(),
            key=lambda t: self._tag_priority(t),
            reverse=True,
        )

        for tag in sorted_tags:
            # Determine depth for this tag
            depth = DepthLevel.SUMMARY
            if working_set and tag in working_set:
                depth = working_set[tag].depth

            if depth == DepthLevel.NONE:
                continue

            if depth == DepthLevel.FULL and full_segments and tag in full_segments:
                section = self._format_full_section(tag, full_segments[tag])
            elif depth == DepthLevel.SEGMENTS and full_segments and tag in full_segments:
                section = self._format_segments_section(tag, full_segments[tag])
            else:
                # SUMMARY depth or fallback
                summaries = summaries_by_tag.get(tag, [])
                if not summaries:
                    continue
                section = self._format_tag_section(tag, summaries)

            section_tokens = self.token_counter(section)

            if tag_tokens + section_tokens > tag_budget:
                break

            tag_sections[tag] = section
            tag_tokens += section_tokens

        # Conversation budget = remaining tokens
        conversation_budget = token_budget - core_tokens - tag_tokens - hint_tokens

        # Trim conversation to budget
        trimmed = self._trim_conversation(conversation_history, conversation_budget)
        conv_tokens = sum(self.token_counter(m.content) for m in trimmed)

        # Build prepend text (core + context hint + tag sections)
        prepend_parts: list[str] = []
        if core:
            prepend_parts.append(core)
        if context_hint:
            prepend_parts.append(context_hint)
        for tag in sorted_tags:
            if tag in tag_sections:
                prepend_parts.append(tag_sections[tag])

        prepend_text = "\n\n".join(prepend_parts)

        total_tokens = core_tokens + tag_tokens + conv_tokens

        return AssembledContext(
            core_context=core,
            tag_sections=tag_sections,
            conversation_history=trimmed,
            total_tokens=total_tokens,
            budget_breakdown={
                "core": core_tokens,
                "context_hint": hint_tokens,
                "tags": tag_tokens,
                "conversation": conv_tokens,
            },
            prepend_text=prepend_text,
        )

    def _format_tag_section(self, tag: str, summaries: list[StoredSummary]) -> str:
        """Format summaries for a tag as XML-tagged section (SUMMARY depth)."""
        if not summaries:
            return ""

        last_updated = max(s.end_timestamp for s in summaries)
        all_tags = sorted({t for s in summaries for t in s.tags})
        tags_attr = ", ".join(all_tags) if all_tags else tag
        summary_texts = [s.summary for s in summaries]
        body = "\n\n---\n\n".join(summary_texts)

        return (
            f'<virtual-context tags="{tags_attr}" segments="{len(summaries)}" '
            f'last_updated="{last_updated.isoformat()}">\n'
            f"{body}\n"
            f"</virtual-context>"
        )

    def _format_segments_section(self, tag: str, segments: list[StoredSegment]) -> str:
        """Format individual segment summaries (SEGMENTS depth)."""
        if not segments:
            return ""

        parts: list[str] = []
        for seg in segments:
            all_tags = sorted(seg.tags) if seg.tags else [tag]
            tags_attr = ", ".join(all_tags)
            parts.append(
                f'<virtual-context tags="{tags_attr}" depth="segments" '
                f'ref="{seg.ref}" created="{seg.created_at.isoformat()}">\n'
                f"{seg.summary}\n"
                f"</virtual-context>"
            )
        return "\n\n".join(parts)

    def _format_full_section(self, tag: str, segments: list[StoredSegment]) -> str:
        """Format full original text (FULL depth)."""
        if not segments:
            return ""

        parts: list[str] = []
        for seg in segments:
            all_tags = sorted(seg.tags) if seg.tags else [tag]
            tags_attr = ", ".join(all_tags)
            text = seg.full_text if seg.full_text else seg.summary
            parts.append(
                f'<virtual-context tags="{tags_attr}" depth="full" '
                f'ref="{seg.ref}" created="{seg.created_at.isoformat()}">\n'
                f"{text}\n"
                f"</virtual-context>"
            )
        return "\n\n".join(parts)

    def _trim_conversation(self, history: list[Message], budget: int) -> list[Message]:
        """Keep most recent messages that fit within budget."""
        if budget <= 0:
            return []

        result: list[Message] = []
        tokens_used = 0

        # Work backwards from most recent
        for msg in reversed(history):
            msg_tokens = self.token_counter(msg.content)
            if tokens_used + msg_tokens > budget:
                break
            result.append(msg)
            tokens_used += msg_tokens

        result.reverse()
        return result

    def _truncate_core(self, core: str, max_tokens: int) -> str:
        """Truncate core context, keeping the beginning."""
        if self.token_counter(core) <= max_tokens:
            return core

        # Rough char estimate: 4 chars per token
        max_chars = max_tokens * 4
        return core[:max_chars]

    def _tag_priority(self, tag: str) -> int:
        """Get priority for a tag from tag rules."""
        for rule in self.tag_rules:
            if fnmatch.fnmatch(tag, rule.match):
                return rule.priority
        return 5

    def load_core_context(self, base_path: Path | None = None) -> str:
        """Load and concatenate core files defined in config."""
        if not self.config.core_files:
            return ""

        parts: list[str] = []
        # Sort by priority descending
        sorted_files = sorted(
            self.config.core_files,
            key=lambda f: f.get("priority", 5),
            reverse=True,
        )

        for file_conf in sorted_files:
            file_path = Path(file_conf["path"])
            if base_path:
                file_path = base_path / file_path
            if file_path.is_file():
                content = file_path.read_text()
                parts.append(f"# {file_path.name}\n\n{content}")

        return "\n\n---\n\n".join(parts)
