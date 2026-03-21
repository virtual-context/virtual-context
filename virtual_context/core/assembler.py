"""ContextAssembler: build final context from core files + tag summaries + conversation."""

from __future__ import annotations

import fnmatch
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

from ..types import (
    AssembledContext,
    AssemblerConfig,
    DepthLevel,
    Fact,
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
        When working_set is None, all tags served as SUMMARY.

        max_context_tokens: If set, caps the total VC context (core + hint + tags)
        to fit within available headroom. Used by proxy to prevent exceeding
        the upstream model's context limit.
        """
        core_budget = self.config.core_context_max_tokens

        # Truncate core context to budget
        core = self._truncate_core(core_context, core_budget)
        core_tokens = self.token_counter(core)

        # Context hint tokens
        hint_tokens = self.token_counter(context_hint) if context_hint else 0

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

        # --- Unified pool allocation ---
        pool = self.config.context_injection_max_tokens
        # Headroom cap (proxy mode)
        if max_context_tokens is not None:
            available = max(0, max_context_tokens - core_tokens - hint_tokens)
            pool = min(pool, available)
        tag_cap = self.config.tag_context_max_tokens
        facts_cap = self.config.facts_max_tokens

        # Build all tag section candidates
        _built_sections: dict[str, str] = {}
        _section_tokens: dict[str, int] = {}
        for tag in sorted_tags:
            depth = DepthLevel.SUMMARY
            if working_set and tag in working_set:
                depth = working_set[tag].depth
            if depth == DepthLevel.NONE:
                logger.info("Tag '%s' SKIP (depth=NONE, hint-only)", tag)
                continue
            if depth == DepthLevel.FULL and full_segments and tag in full_segments:
                section = self._format_full_section(tag, full_segments[tag])
            elif depth == DepthLevel.SEGMENTS and full_segments and tag in full_segments:
                section = self._format_segments_section(tag, full_segments[tag])
            else:
                sums = summaries_by_tag.get(tag, [])
                if not sums:
                    logger.info("Tag '%s' SKIP (no summaries available)", tag)
                    continue
                section = self._format_tag_section(tag, sums)
            _built_sections[tag] = section
            _section_tokens[tag] = self.token_counter(section)

        # Score all candidates
        scored_items: list[tuple[float, str, str, int]] = []  # (score, kind, key, tokens)

        for tag in _built_sections:
            score = retrieval_result.retrieval_scores.get(tag, float(self._tag_priority(tag)))
            scored_items.append((score, "tag", tag, _section_tokens[tag]))

        # Score facts
        expanded_tags = set(
            retrieval_result.retrieval_metadata.get("tags_queried", [])
            + retrieval_result.retrieval_metadata.get("related_tags_used", [])
        )
        _fact_lines: dict[int, str] = {}
        for i, fact in enumerate(retrieval_result.facts):
            line = fact.format_for_prompt()
            line_tokens = self.token_counter(line)
            tag_overlap = len(set(fact.tags) & expanded_tags) if expanded_tags else 0
            try:
                age_days = (datetime.now(timezone.utc) - fact.mentioned_at).days
            except (TypeError, AttributeError):
                age_days = 365
            recency = 0.1 * max(0.0, 1.0 - age_days / 365)
            fact_score = tag_overlap + recency
            scored_items.append((fact_score, "fact", str(i), line_tokens))
            _fact_lines[i] = line

        # Sort by score descending
        scored_items.sort(key=lambda x: x[0], reverse=True)

        # Greedy fill with soft caps
        tag_tokens = 0
        facts_tokens = 0
        pool_used = 0
        tag_sections: dict[str, str] = {}
        selected_fact_indices: list[int] = []

        for score, kind, key, tokens in scored_items:
            if kind == "tag":
                if tag_tokens + tokens > tag_cap:
                    logger.info("Tag '%s' SKIP (tag cap: %d+%d > %d)", key, tag_tokens, tokens, tag_cap)
                    continue
                if pool_used + tokens > pool:
                    logger.info("Tag '%s' SKIP (pool: need %dt, have %dt remaining of %dt)",
                                key, tokens, pool - pool_used, pool)
                    continue
                tag_sections[key] = _built_sections[key]
                tag_tokens += tokens
                pool_used += tokens
                logger.info("Pool: '%s' INCLUDE (tag, score=%.2f, %dt, pool %d/%dt)",
                            key, score, tokens, pool_used, pool)
            else:  # fact
                if facts_tokens + tokens > facts_cap:
                    continue
                if pool_used + tokens > pool:
                    continue
                selected_fact_indices.append(int(key))
                facts_tokens += tokens
                pool_used += tokens

        logger.info("Pool allocation: tags=%dt (%d sections), facts=%dt (%d facts), total=%d/%dt",
                    tag_tokens, len(tag_sections), facts_tokens, len(selected_fact_indices),
                    pool_used, pool)

        # Format selected facts (budget already enforced by pool allocation)
        selected_facts = [retrieval_result.facts[i] for i in sorted(selected_fact_indices)]
        facts_text = self._format_facts(selected_facts, pool) if selected_facts else ""
        facts_tokens_actual = self.token_counter(facts_text) if facts_text else 0

        # Track presented segment refs
        presented_refs: set[str] = set()
        for s in retrieval_result.summaries:
            if s.primary_tag in tag_sections and s.ref:
                presented_refs.add(s.ref)
        if full_segments:
            for tag, segs in full_segments.items():
                if tag in tag_sections:
                    for seg in segs:
                        if seg.ref:
                            presented_refs.add(seg.ref)

        # Conversation budget = remaining tokens
        conversation_budget = (
            token_budget - core_tokens - tag_tokens - hint_tokens - facts_tokens_actual
        )

        # Trim conversation to budget
        trimmed = self._trim_conversation(conversation_history, conversation_budget)
        conv_tokens = sum(self.token_counter(m.content) for m in trimmed)

        # Build prepend text (core + context hint + tag sections + facts)
        prepend_parts: list[str] = []
        if core:
            prepend_parts.append(core)
        if context_hint:
            prepend_parts.append(context_hint)
        for tag in sorted_tags:
            if tag in tag_sections:
                prepend_parts.append(tag_sections[tag])
        if facts_text:
            prepend_parts.append(facts_text)

        prepend_text = "\n\n".join(prepend_parts)

        # Hard budget cap: if prepend_text exceeds token_budget,
        # drop least-relevant tag sections until it fits.
        prepend_tokens = self.token_counter(prepend_text)
        if prepend_tokens > token_budget:
            logger.error(
                "Assembled context (%d tokens) exceeds token_budget (%d). "
                "Consider increasing context_window or reducing assembly config "
                "values (tag_context_max_tokens=%d, facts_max_tokens=%d). "
                "Truncating least-relevant tag sections to fit.",
                prepend_tokens, token_budget,
                self.config.tag_context_max_tokens,
                self.config.facts_max_tokens,
            )
            # Drop tags from end (least relevant — sorted_tags is priority-ordered)
            for drop_tag in reversed(sorted_tags):
                if drop_tag not in tag_sections:
                    continue
                dropped_tokens = self.token_counter(tag_sections[drop_tag])
                logger.info("Tag '%s' DROP (hard cap: %dt over budget %dt, freeing %dt)",
                            drop_tag, prepend_tokens, token_budget, dropped_tokens)
                del tag_sections[drop_tag]
                tag_tokens -= dropped_tokens
                # Rebuild prepend_text
                prepend_parts = []
                if core:
                    prepend_parts.append(core)
                if context_hint:
                    prepend_parts.append(context_hint)
                for tag in sorted_tags:
                    if tag in tag_sections:
                        prepend_parts.append(tag_sections[tag])
                if facts_text:
                    prepend_parts.append(facts_text)
                prepend_text = "\n\n".join(prepend_parts)
                prepend_tokens = self.token_counter(prepend_text)
                if prepend_tokens <= token_budget:
                    break

        total_tokens = core_tokens + tag_tokens + facts_tokens_actual + conv_tokens

        return AssembledContext(
            core_context=core,
            tag_sections=tag_sections,
            facts_text=facts_text,
            conversation_history=trimmed,
            total_tokens=total_tokens,
            budget_breakdown={
                "core": core_tokens,
                "context_hint": hint_tokens,
                "tags": tag_tokens,
                "facts": facts_tokens_actual,
                "conversation": conv_tokens,
            },
            prepend_text=prepend_text,
            presented_segment_refs=presented_refs,
        )

    def _format_facts(self, facts: list[Fact], max_tokens: int) -> str:
        if not facts:
            return ""
        lines: list[str] = []
        tokens_used = 0
        # Reserve tokens for the XML wrapper
        wrapper_overhead = self.token_counter("<facts>\n</facts>")
        tokens_used += wrapper_overhead
        for fact in facts:
            line = fact.format_for_prompt()
            line_tokens = self.token_counter(line)
            if tokens_used + line_tokens > max_tokens:
                break
            lines.append(line)
            tokens_used += line_tokens
        if not lines:
            return ""
        return "<facts>\n" + "\n".join(lines) + "\n</facts>"

    def _format_tag_section(self, tag: str, summaries: list[StoredSummary]) -> str:
        if not summaries:
            return ""

        # Sort chronologically so reader sees old → new progression
        summaries = sorted(summaries, key=lambda s: s.start_timestamp)

        all_tags = sorted({t for s in summaries for t in s.tags})
        tags_attr = ", ".join(all_tags) if all_tags else tag

        # Prefix each summary with sequence number and optional session date
        total = len(summaries)
        summary_texts: list[str] = []
        for idx, s in enumerate(summaries, 1):
            prefix = f"[{idx}/{total}]"
            session = s.metadata.session_date
            if session:
                prefix += f" [{session}]"
            text = f"{prefix}\n{s.summary}"
            # Expansion hint for tool segments
            tool_tags = [t for t in s.tags if t.startswith("tool_")]
            if tool_tags:
                text += f'\n[tool output truncated — vc_expand_topic("{tool_tags[0]}") for full result]'
            summary_texts.append(text)

        body = "\n\n---\n\n".join(summary_texts)

        return (
            f'<virtual-context tags="{tags_attr}" segments="{len(summaries)}">\n'
            f"{body}\n"
            f"</virtual-context>"
        )

    def _format_segments_section(self, tag: str, segments: list[StoredSegment]) -> str:
        if not segments:
            return ""

        # Sort chronologically so reader sees old → new progression
        segments = sorted(segments, key=lambda s: s.created_at)

        parts: list[str] = []
        for seg in segments:
            all_tags = sorted(seg.tags) if seg.tags else [tag]
            tags_attr = ", ".join(all_tags)
            session_attr = f' session="{seg.metadata.session_date}"' if seg.metadata.session_date else ""
            parts.append(
                f'<virtual-context tags="{tags_attr}" depth="segments" '
                f'ref="{seg.ref}"{session_attr} created="{seg.created_at.isoformat()}">\n'
                f"{seg.summary}\n"
                f"</virtual-context>"
            )
        return "\n\n".join(parts)

    def _format_full_section(self, tag: str, segments: list[StoredSegment]) -> str:
        if not segments:
            return ""

        # Sort chronologically so reader sees old → new progression
        segments = sorted(segments, key=lambda s: s.created_at)

        parts: list[str] = []
        for seg in segments:
            all_tags = sorted(seg.tags) if seg.tags else [tag]
            tags_attr = ", ".join(all_tags)
            text = seg.full_text if seg.full_text else seg.summary
            session_attr = f' session="{seg.metadata.session_date}"' if seg.metadata.session_date else ""
            parts.append(
                f'<virtual-context tags="{tags_attr}" depth="full" '
                f'ref="{seg.ref}"{session_attr} created="{seg.created_at.isoformat()}">\n'
                f"{text}\n"
                f"</virtual-context>"
            )
        return "\n\n".join(parts)

    def _trim_conversation(self, history: list[Message], budget: int) -> list[Message]:
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
        if self.token_counter(core) <= max_tokens:
            return core

        # Rough char estimate: 4 chars per token
        max_chars = max_tokens * 4
        return core[:max_chars]

    def _tag_priority(self, tag: str) -> int:
        for rule in self.tag_rules:
            if fnmatch.fnmatch(tag, rule.match):
                return rule.priority
        return 5

    def load_core_context(self, base_path: Path | None = None) -> str:
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
