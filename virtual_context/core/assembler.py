"""ContextAssembler: build final context from core files + domain summaries + conversation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..types import (
    AssembledContext,
    AssemblerConfig,
    Message,
    RetrievalResult,
    StoredSummary,
)


class ContextAssembler:
    """Assemble enriched context within token budget.

    Assembly order (top to bottom in final prompt):
    1. [CORE CONTEXT] - always-on files (SOUL.md, USER.md, etc.)
    2. [DOMAIN CONTEXT] - retrieved summaries in <virtual-context> tags
    3. [CONVERSATION HISTORY] - recent turns, most recent at bottom
    """

    def __init__(
        self,
        config: AssemblerConfig,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)

    def assemble(
        self,
        core_context: str,
        retrieval_result: RetrievalResult,
        conversation_history: list[Message],
        token_budget: int,
    ) -> AssembledContext:
        """Build final context within token budget."""
        core_budget = self.config.core_context_max_tokens
        domain_budget = self.config.domain_context_max_tokens

        # Truncate core context to budget
        core = self._truncate_core(core_context, core_budget)
        core_tokens = self.token_counter(core)

        # Build domain sections
        domain_sections: dict[str, str] = {}
        domain_tokens = 0

        # Group summaries by domain
        summaries_by_domain: dict[str, list[StoredSummary]] = {}
        for s in retrieval_result.summaries:
            summaries_by_domain.setdefault(s.domain, []).append(s)

        # Sort domains by priority (higher priority first)
        sorted_domains = sorted(
            summaries_by_domain.keys(),
            key=lambda d: self._domain_priority(d),
            reverse=True,
        )

        for domain in sorted_domains:
            summaries = summaries_by_domain[domain]
            section = self._format_domain_section(domain, summaries)
            section_tokens = self.token_counter(section)

            if domain_tokens + section_tokens > domain_budget:
                break

            domain_sections[domain] = section
            domain_tokens += section_tokens

        # Conversation budget = remaining tokens
        conversation_budget = token_budget - core_tokens - domain_tokens

        # Trim conversation to budget
        trimmed = self._trim_conversation(conversation_history, conversation_budget)
        conv_tokens = sum(self.token_counter(m.content) for m in trimmed)

        # Build prepend text (core + domain sections)
        prepend_parts: list[str] = []
        if core:
            prepend_parts.append(core)
        for domain in sorted_domains:
            if domain in domain_sections:
                prepend_parts.append(domain_sections[domain])

        prepend_text = "\n\n".join(prepend_parts)

        total_tokens = core_tokens + domain_tokens + conv_tokens

        return AssembledContext(
            core_context=core,
            domain_sections=domain_sections,
            conversation_history=trimmed,
            total_tokens=total_tokens,
            budget_breakdown={
                "core": core_tokens,
                "domain": domain_tokens,
                "conversation": conv_tokens,
            },
            prepend_text=prepend_text,
        )

    def _format_domain_section(self, domain: str, summaries: list[StoredSummary]) -> str:
        """Format summaries for a domain as XML-tagged section."""
        if not summaries:
            return ""

        last_updated = max(s.end_timestamp for s in summaries)
        summary_texts = [s.summary for s in summaries]
        body = "\n\n---\n\n".join(summary_texts)

        return (
            f'<virtual-context domain="{domain}" segments="{len(summaries)}" '
            f'last_updated="{last_updated.isoformat()}">\n'
            f"{body}\n"
            f"</virtual-context>"
        )

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

    def _domain_priority(self, domain: str) -> int:
        """Get priority for a domain from config."""
        for cf in self.config.core_files:
            if cf.get("domain") == domain:
                return cf.get("priority", 5)
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
