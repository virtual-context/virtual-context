"""ContextAssembler: build final context from core files + tag summaries + conversation."""

from __future__ import annotations

import fnmatch
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_ASSEMBLE_BREAKDOWN_LOG_THRESHOLD_MS = 200.0
_ASSEMBLE_BREAKDOWN_MAX_STAGES = 8

from .llm_utils import format_code_ref

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


def format_tag_section(
    tag: str,
    summaries: list["StoredSummary"],
    store: "ContextStore | None" = None,
    conversation_id: str = "",
) -> str:
    """Render a tag section in the canonical <virtual-context> format.

    Shared by the assembler and the fill pass. Tool-hint enrichment is
    optional (gracefully degrades if store is None).
    """
    if not summaries:
        return ""

    summaries = sorted(summaries, key=lambda s: s.start_timestamp)

    all_tags = sorted({t for s in summaries for t in s.tags})
    tags_attr = ", ".join(all_tags) if all_tags else tag

    total = len(summaries)
    summary_texts: list[str] = []
    for idx, s in enumerate(summaries, 1):
        prefix = f"[{idx}/{total}]"
        session = s.metadata.session_date
        if session:
            prefix += f" [{session}]"
        text = f"{prefix}\n{s.summary}"
        code_refs = getattr(s.metadata, "code_refs", None) or []
        if code_refs:
            refs = [format_code_ref(ref) for ref in code_refs if ref.get("file")]
            if refs:
                text += f"\n[refs: {', '.join(refs)}]"
        tool_tags = [t for t in s.tags if t.startswith("tool_")]
        if tool_tags:
            text += f'\n[tool output truncated — vc_expand_topic("{tool_tags[0]}") for full result]'
        # Optional tool hint enrichment
        if store and conversation_id and s.ref:
            get_refs = getattr(store, "get_tool_outputs_for_segment", None)
            get_names = getattr(store, "get_tool_names_for_segment", None)
            if callable(get_refs) and callable(get_names):
                try:
                    refs = get_refs(conversation_id, s.ref)
                    if refs:
                        names = get_names(conversation_id, s.ref)
                        names_str = ", ".join(names) if names else "tools"
                        text += f"\n[Tools: {names_str} -- {len(refs)} outputs restorable via vc_restore_tool]"
                except Exception:
                    pass
        summary_texts.append(text)

    body = "\n\n---\n\n".join(summary_texts)

    return (
        f'<virtual-context tags="{tags_attr}" segments="{len(summaries)}">\n'
        f"{body}\n"
        f"</virtual-context>"
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
        store: object | None = None,
        conversation_id: str = "",
    ) -> None:
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self.tag_rules = tag_rules or []
        self._store = store
        self._conversation_id = conversation_id

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
        _started = time.monotonic()
        _breakdown: dict[str, float] = {}

        def _note(stage: str, started_at: float) -> None:
            _breakdown[stage] = round((time.monotonic() - started_at) * 1000, 1)

        core_budget = self.config.core_context_max_tokens

        # Truncate core context to budget
        _stage = time.monotonic()
        core = self._truncate_core(core_context, core_budget)
        _note("truncate_core", _stage)
        _stage = time.monotonic()
        core_tokens = self.token_counter(core)
        _note("count_core_tokens", _stage)

        # Context hint tokens
        _stage = time.monotonic()
        hint_tokens = self.token_counter(context_hint) if context_hint else 0
        _note("count_hint_tokens", _stage)

        # Group summaries by primary_tag
        _stage = time.monotonic()
        summaries_by_tag: dict[str, list[StoredSummary]] = {}
        for s in retrieval_result.summaries:
            summaries_by_tag.setdefault(s.primary_tag, []).append(s)

        # Also collect tags from full_segments that might not be in summaries
        if full_segments:
            for tag in full_segments:
                if tag not in summaries_by_tag:
                    summaries_by_tag[tag] = []
        _note("prepare_summary_groups", _stage)

        # Sort tags by priority (higher priority first)
        _stage = time.monotonic()
        sorted_tags = sorted(
            summaries_by_tag.keys(),
            key=lambda t: self._tag_priority(t),
            reverse=True,
        )
        _note("sort_tags", _stage)

        # --- Unified pool allocation ---
        pool = self.config.context_injection_max_tokens
        # Headroom cap (proxy mode)
        if max_context_tokens is not None:
            available = max(0, max_context_tokens - core_tokens - hint_tokens)
            pool = min(pool, available)
        tag_cap = self.config.tag_context_max_tokens
        facts_cap = self.config.facts_max_tokens

        # Build all tag section candidates
        _stage = time.monotonic()
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
        _note("build_tag_sections", _stage)

        # Score all candidates
        _stage = time.monotonic()
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
        _note("score_candidates", _stage)

        # Sort by score descending
        _stage = time.monotonic()
        scored_items.sort(key=lambda x: x[0], reverse=True)
        _note("sort_candidates", _stage)

        # Greedy fill with soft caps
        _stage = time.monotonic()
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
                    logger.debug("Fact #%s SKIP (facts cap: %d+%d > %d)", key, facts_tokens, tokens, facts_cap)
                    continue
                if pool_used + tokens > pool:
                    logger.debug("Fact #%s SKIP (pool: need %dt, have %dt remaining)", key, tokens, pool - pool_used)
                    continue
                selected_fact_indices.append(int(key))
                facts_tokens += tokens
                pool_used += tokens
        _note("pool_fill", _stage)

        logger.info("Pool allocation: tags=%dt (%d sections), facts=%dt (%d facts), total=%d/%dt",
                    tag_tokens, len(tag_sections), facts_tokens, len(selected_fact_indices),
                    pool_used, pool)

        # Format selected facts (budget already enforced by pool allocation)
        _stage = time.monotonic()
        selected_facts = [retrieval_result.facts[i] for i in sorted(selected_fact_indices)]
        facts_text = self._format_facts(selected_facts, facts_tokens + 100) if selected_facts else ""
        facts_tokens_actual = self.token_counter(facts_text) if facts_text else 0
        _note("format_facts", _stage)

        # Track presented segment refs
        _stage = time.monotonic()
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
        _note("track_presented_refs", _stage)

        # Conversation budget = remaining tokens
        conversation_budget = (
            token_budget - core_tokens - tag_tokens - hint_tokens - facts_tokens_actual
        )

        # Trim conversation to budget
        _stage = time.monotonic()
        trimmed = self._trim_conversation(conversation_history, conversation_budget)
        _note("trim_conversation", _stage)
        _stage = time.monotonic()
        conv_tokens = sum(self.token_counter(m.content) for m in trimmed)
        _note("count_conversation_tokens", _stage)

        # Build prepend text (core + context hint + tag sections + facts)
        _stage = time.monotonic()
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
        _note("build_prepend", _stage)

        # Hard budget cap: if prepend_text exceeds token_budget,
        # drop least-relevant tag sections until it fits.
        _stage = time.monotonic()
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
        _note("hard_cap_trim", _stage)

        total_tokens = core_tokens + tag_tokens + facts_tokens_actual + conv_tokens

        # Compute presented_tags from rendered <virtual-context tags="..."> headers
        _stage = time.monotonic()
        _vc_tags_re = re.compile(r'<virtual-context\s+tags="([^"]*)"')
        _presented_tags: set[str] = set()
        for _section_text in tag_sections.values():
            for _m in _vc_tags_re.finditer(_section_text):
                for _t in _m.group(1).split(", "):
                    _t = _t.strip()
                    if _t:
                        _presented_tags.add(_t)
        # Also include section keys as baseline for edge cases
        _presented_tags.update(tag_sections.keys())
        _note("extract_presented_tags", _stage)

        total_ms = round((time.monotonic() - _started) * 1000, 1)
        if total_ms >= _ASSEMBLE_BREAKDOWN_LOG_THRESHOLD_MS:
            stages = sorted(
                ((stage, ms) for stage, ms in _breakdown.items() if ms > 0),
                key=lambda item: item[1],
                reverse=True,
            )[:_ASSEMBLE_BREAKDOWN_MAX_STAGES]
            stage_bits = [f"{stage}={ms:.1f}ms" for stage, ms in stages]
            logger.info(
                "ASSEMBLE_BREAKDOWN tags=%d facts=%d history=%d total=%sms %s",
                len(tag_sections),
                len(selected_facts),
                len(trimmed),
                total_ms,
                " ".join(stage_bits) if stage_bits else "no-stages",
            )

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
            selected_facts=selected_facts,
            retrieval_result=retrieval_result,
            presented_tags=_presented_tags,
            assembly_breakdown=_breakdown,
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
        return format_tag_section(
            tag,
            summaries,
            store=getattr(self, "_store", None),
            conversation_id=getattr(self, "_conversation_id", ""),
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
            summary_text = self._maybe_append_tool_hint(seg.summary, seg.ref)
            parts.append(
                f'<virtual-context tags="{tags_attr}" depth="segments" '
                f'ref="{seg.ref}"{session_attr} created="{seg.created_at.isoformat()}">\n'
                f"{summary_text}\n"
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

    def _maybe_append_tool_hint(self, text: str, segment_ref: str) -> str:
        """Append a tool hint to summary text if the segment has linked tool outputs.

        Queries the store at assembly time to discover tool names linked via
        segment_tool_outputs, then appends a hint like:
        [Tools: Bash, Read, Glob -- 48 outputs restorable via vc_restore_tool]
        """
        if not self._store or not self._conversation_id or not segment_ref:
            return text
        store = self._store
        get_refs = getattr(store, "get_tool_outputs_for_segment", None)
        get_names = getattr(store, "get_tool_names_for_segment", None)
        if not callable(get_refs) or not callable(get_names):
            return text
        try:
            refs = get_refs(self._conversation_id, segment_ref)
            if not refs:
                return text
            names = get_names(self._conversation_id, segment_ref)
            names_str = ", ".join(names) if names else "tools"
            text += f"\n[Tools: {names_str} \u2014 {len(refs)} outputs restorable via vc_restore_tool]"
        except Exception:
            logger.debug("Failed to enrich segment %s with tool hint", segment_ref, exc_info=True)
        return text

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
