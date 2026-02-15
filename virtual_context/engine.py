"""VirtualContextEngine: main orchestrator wiring all components together."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from .config import load_config
from .core.assembler import ContextAssembler
from .core.compactor import DomainCompactor
from .core.cost_tracker import CostTracker
from .core.monitor import ContextMonitor
from .core.retriever import ContextRetriever
from .core.segmenter import TopicSegmenter
from .core.tag_canonicalizer import TagCanonicalizer
from .core.tag_generator import build_tag_generator, TagGenerator
from .core.turn_tag_index import TurnTagIndex
from .storage.filesystem import FilesystemStore
from .storage.sqlite import SQLiteStore
from .token_counter import create_token_counter
from .types import (
    AssembledContext,
    CompactionReport,
    CompactionResult,
    Message,
    RetrievalResult,
    SessionCostSummary,
    StoredSegment,
    TurnTagEntry,
    VirtualContextConfig,
)

logger = logging.getLogger(__name__)


class VirtualContextEngine:
    """Main orchestrator: two entry points for inbound messages and turn completion.

    Usage:
        engine = VirtualContextEngine(config_path="./virtual-context.yaml")

        # Before sending to LLM
        assembled = engine.on_message_inbound(message, history)

        # After LLM responds
        report = engine.on_turn_complete(history)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: VirtualContextConfig | None = None,
    ) -> None:
        self.config = config or load_config(config_path)
        self._token_counter = create_token_counter(self.config.token_counter)

        # Initialize components
        self._turn_tag_index = TurnTagIndex()
        self._init_store()
        self._init_canonicalizer()
        self._init_tag_generator()
        self._init_monitor()
        self._init_segmenter()
        self._init_assembler()
        self._init_retriever()
        self._init_compactor()
        self._init_cost_tracker()
        self._compacted_through = 0  # message index watermark: messages before this already compacted

    def _init_canonicalizer(self) -> None:
        """Initialize the tag canonicalizer with store aliases."""
        self._canonicalizer = TagCanonicalizer(store=self._store)
        self._canonicalizer.load()

    def _init_tag_generator(self) -> None:
        """Build the tag generator from config."""
        llm_provider = None

        # Try to build an LLM provider for tagging
        if self.config.tag_generator.type == "llm":
            provider_name = self.config.tag_generator.provider
            provider_config = self.config.providers.get(provider_name, {})
            llm_provider = self._build_provider(provider_name, provider_config)

        self._tag_generator: TagGenerator = build_tag_generator(
            self.config.tag_generator, llm_provider, canonicalizer=self._canonicalizer
        )

    def _init_store(self) -> None:
        """Initialize the storage backend."""
        if self.config.storage.backend == "sqlite":
            self._store = SQLiteStore(db_path=self.config.storage.sqlite_path)
        else:
            self._store = FilesystemStore(root=self.config.storage.root)

    def _init_monitor(self) -> None:
        self._monitor = ContextMonitor(
            config=self.config.monitor,
            token_counter=self._token_counter,
        )

    def _init_segmenter(self) -> None:
        self._segmenter = TopicSegmenter(
            tag_generator=self._tag_generator,
            config=self.config.segmenter,
            token_counter=self._token_counter,
            turn_tag_index=self._turn_tag_index,
        )

    def _init_assembler(self) -> None:
        self._assembler = ContextAssembler(
            config=self.config.assembler,
            token_counter=self._token_counter,
            tag_rules=self.config.tag_rules,
        )

    def _init_retriever(self) -> None:
        self._retriever = ContextRetriever(
            tag_generator=self._tag_generator,
            store=self._store,
            config=self.config.retriever,
            turn_tag_index=self._turn_tag_index,
        )

    def _init_compactor(self) -> None:
        """Initialize the compactor with an LLM provider."""
        self._llm_provider = None
        self._compactor = None

        provider_name = self.config.summarization.provider
        provider_config = self.config.providers.get(provider_name, {})
        self._llm_provider = self._build_provider(provider_name, provider_config)

        if self._llm_provider:
            self._compactor = DomainCompactor(
                llm_provider=self._llm_provider,
                config=self.config.compactor,
                token_counter=self._token_counter,
                model_name=self.config.summarization.model,
                tag_rules=self.config.tag_rules,
            )

    def _init_cost_tracker(self) -> None:
        self._cost_tracker = CostTracker(config=self.config.cost_tracking)

    def _build_provider(self, provider_name: str, provider_config: dict):
        """Build an LLM provider from config."""
        ptype = provider_config.get("type", provider_name)

        if ptype == "generic_openai":
            from .providers.generic_openai import GenericOpenAIProvider
            return GenericOpenAIProvider(
                base_url=provider_config.get("base_url", "http://127.0.0.1:11434/v1"),
                model=provider_config.get("model", self.config.summarization.model),
                temperature=self.config.summarization.temperature,
                api_key=provider_config.get("api_key", "not-needed"),
            )

        if ptype == "anthropic":
            api_key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
            api_key = provider_config.get("api_key") or os.environ.get(api_key_env, "")
            if api_key:
                from .providers.anthropic import AnthropicProvider
                return AnthropicProvider(
                    api_key=api_key,
                    model=provider_config.get("model", self.config.summarization.model),
                    temperature=self.config.summarization.temperature,
                )

        return None

    def on_message_inbound(
        self,
        message: str,
        conversation_history: list[Message],
    ) -> AssembledContext:
        """Before sending to LLM: tag, retrieve, assemble enriched context."""
        # Determine active tags from recent tag results
        # Post-compaction: don't suppress retrieval — stored summaries are needed
        # since raw turns have been compacted away
        if self._compacted_through > 0:
            active_tags = []
        else:
            active_tags = self._get_active_tags(conversation_history)

        # Compute current utilization (only count un-compacted history)
        snapshot = self._monitor.build_snapshot(
            conversation_history[self._compacted_through:]
        )
        utilization = snapshot.total_tokens / snapshot.budget_tokens if snapshot.budget_tokens > 0 else 0.0

        # Retrieve relevant tag summaries
        retrieval_result = self._retriever.retrieve(
            message=message,
            current_active_tags=active_tags,
            current_utilization=utilization,
        )

        # Build context awareness hint (post-compaction only)
        context_hint = self._build_context_hint()

        # Load core context
        core_context = self._assembler.load_core_context()

        # Assemble enriched context
        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=retrieval_result,
            conversation_history=conversation_history,
            token_budget=self.config.context_window,
            context_hint=context_hint,
        )

        # Expose the message's own tags for downstream use (e.g. history filtering).
        # Use tags_from_message (what the tag generator produced for this message)
        # rather than tags_matched (which only includes tags found in the store).
        message_tags = retrieval_result.retrieval_metadata.get(
            "tags_from_message", retrieval_result.tags_matched
        )
        assembled.matched_tags = message_tags
        assembled.context_hint = context_hint
        assembled.broad = retrieval_result.broad

        return assembled

    def on_turn_complete(
        self,
        conversation_history: list[Message],
    ) -> CompactionReport | None:
        """After LLM responds: check thresholds, compact if needed."""
        # Tag the latest round trip
        latest_pair = self._get_latest_turn_pair(conversation_history)
        if latest_pair:
            combined_text = " ".join(m.content for m in latest_pair)
            store_tags = [ts.tag for ts in self._store.get_all_tags()]
            tag_result = self._tag_generator.generate_tags(combined_text, store_tags)
            self._turn_tag_index.append(TurnTagEntry(
                turn_number=len(self._turn_tag_index.entries),
                message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                tags=tag_result.tags,
                primary_tag=tag_result.primary,
            ))

        # Build snapshot (only count un-compacted messages)
        snapshot = self._monitor.build_snapshot(
            conversation_history[self._compacted_through:]
        )

        # Check thresholds
        signal = self._monitor.check(snapshot)

        if signal is None:
            return None

        if self._compactor is None:
            logger.warning(
                "Compaction triggered but no LLM provider configured. "
                "Configure a provider in the providers section."
            )
            return None

        logger.info(
            f"Compaction triggered ({signal.priority}): "
            f"{signal.current_tokens}/{signal.budget_tokens} tokens, "
            f"overflow={signal.overflow_tokens}"
        )

        # Select messages to compact (not in protected zone)
        protected_turns = self.config.monitor.protected_recent_turns
        protected_count = protected_turns * 2  # user + assistant per turn

        if len(conversation_history) <= protected_count:
            logger.info("Not enough messages outside protected zone to compact")
            return None

        # Messages to compact: everything between watermark and protected zone.
        # Compact all available messages (not just the minimum) so compaction
        # fires infrequently — one big batch instead of many small ones.
        compact_messages = conversation_history[self._compacted_through:-protected_count]

        if not compact_messages:
            return None

        # Segment and compact (turn_offset maps pair indices to global turn numbers)
        turn_offset = self._compacted_through // 2
        segments = self._segmenter.segment(compact_messages, turn_offset=turn_offset)
        results = self._compactor.compact(segments)

        # Advance watermark past compacted messages
        self._compacted_through += len(compact_messages)

        # Store compacted segments and track tags
        for result in results:
            stored = StoredSegment(
                ref=result.segment_id,
                session_id=self.config.session_id,
                primary_tag=result.primary_tag,
                tags=result.tags,
                summary=result.summary,
                summary_tokens=result.summary_tokens,
                full_text=result.full_text,
                full_tokens=result.original_tokens,
                messages=result.messages,
                metadata=result.metadata,
                compaction_model=self._compactor.model_name,
                compression_ratio=result.compression_ratio,
                start_timestamp=result.timestamp,
                end_timestamp=result.timestamp,
            )
            self._store.store_segment(stored)

        tokens_freed = sum(r.original_tokens - r.summary_tokens for r in results)
        tags = list({tag for r in results for tag in r.tags})

        # Build/update tag summaries — only for tags in newly compacted segments
        tag_summaries_built = 0
        cover_tags: list[str] = []
        if results and self._compactor:
            # Only rebuild tag summaries for tags that were just compacted
            compacted_tags = {tag for r in results for tag in r.tags}
            cover_tags = [
                t for t in self._turn_tag_index.compute_cover_set()
                if t in compacted_tags
            ]
            if cover_tags:
                # Gather segment summaries per cover tag
                tag_to_summaries: dict[str, list] = {}
                for tag in cover_tags:
                    summaries = self._store.get_summaries_by_tags(
                        tags=[tag], min_overlap=1, limit=50
                    )
                    if summaries:
                        tag_to_summaries[tag] = summaries

                # Gather turn numbers per tag from index
                tag_to_turns: dict[str, list[int]] = {}
                for entry in self._turn_tag_index.entries:
                    for tag in entry.tags:
                        if tag in cover_tags:
                            tag_to_turns.setdefault(tag, []).append(entry.turn_number)

                # Load existing tag summaries for staleness check
                existing_tag_summaries = {}
                for tag in cover_tags:
                    ts = self._store.get_tag_summary(tag)
                    if ts:
                        existing_tag_summaries[tag] = ts

                max_turn = max(e.turn_number for e in self._turn_tag_index.entries)

                new_tag_summaries = self._compactor.compact_tag_summaries(
                    cover_tags=cover_tags,
                    tag_to_summaries=tag_to_summaries,
                    tag_to_turns=tag_to_turns,
                    existing_tag_summaries=existing_tag_summaries,
                    max_turn=max_turn,
                )

                for ts in new_tag_summaries:
                    self._store.save_tag_summary(ts)
                tag_summaries_built = len(new_tag_summaries)

        report = CompactionReport(
            segments_compacted=len(results),
            tokens_freed=tokens_freed,
            tags=tags,
            results=results,
            tag_summaries_built=tag_summaries_built,
            cover_tags=cover_tags,
        )

        # Enforce TTL from tag rules
        if self.config.tag_rules:
            min_ttl = min(
                (r.ttl_days for r in self.config.tag_rules if r.ttl_days is not None),
                default=None,
            )
            if min_ttl is not None:
                from datetime import timedelta
                self._store.cleanup(max_age=timedelta(days=min_ttl))

        return report

    def filter_history(
        self,
        conversation_history: list[Message],
        current_tags: list[str],
        recent_turns: int | None = None,
        broad: bool = False,
    ) -> list[Message]:
        """Filter conversation history by tag relevance.

        When ``broad`` is True, all remaining turns are included without
        tag-based filtering.  Pre-compaction the full history fits within the
        context window; post-compaction old turns are already gone, so "all
        remaining" is bounded.

        Always includes the last ``recent_turns`` turn pairs.  For older turns,
        includes only those whose TurnTagIndex tags overlap with *current_tags*.
        If a turn has no index entry (e.g. first turns before tagging kicks in),
        it is included conservatively.

        Returns a new list — the original is not mutated.
        """
        if recent_turns is None:
            recent_turns = self.config.assembler.recent_turns_always_included

        total = len(conversation_history)
        # Each turn = 2 messages (user + assistant), except possibly the
        # very last entry which may be an unpaired user message.
        protected_count = recent_turns * 2

        if total <= protected_count:
            return list(conversation_history)

        # Broad query — include everything, but skip compacted messages
        # (tag summaries from retriever replace them)
        if broad:
            watermark = getattr(self, "_compacted_through", 0)
            if watermark > 0:
                return list(conversation_history[watermark:])
            return list(conversation_history)

        # Skip compacted messages — their content is in stored summaries
        watermark = getattr(self, "_compacted_through", 0)
        older = conversation_history[watermark:-protected_count]
        recent = conversation_history[-protected_count:]

        current_tag_set = set(current_tags)
        filtered: list[Message] = []

        # Walk older messages in pairs (user, assistant)
        i = 0
        while i < len(older):
            # Determine pair boundaries
            if i + 1 < len(older):
                pair = [older[i], older[i + 1]]
                step = 2
            else:
                # Trailing unpaired message — keep it
                filtered.append(older[i])
                break

            turn_idx = (watermark + i) // 2
            entry = self._turn_tag_index.get_tags_for_turn(turn_idx)

            if entry is None:
                # No tag data — conservatively include
                filtered.extend(pair)
            elif "rule" in entry.tags or set(entry.tags) & current_tag_set:
                # Rule turns always included; tag overlap — include
                filtered.extend(pair)
            # else: no overlap — drop this turn

            i += step

        filtered.extend(recent)
        return filtered

    def _get_active_tags(self, history: list[Message]) -> list[str]:
        """Get tags from recent turns via live index."""
        lookback = self.config.retriever.active_tag_lookback
        return list(self._turn_tag_index.get_active_tags(lookback=lookback))

    def _build_context_hint(self) -> str:
        """Build a lightweight topic list for post-compaction prompts.

        Returns an XML block listing stored topics with turn counts so the LLM
        knows what prior context is available.  Returns empty string if
        compaction hasn't occurred or the feature is disabled.
        """
        if not self.config.assembler.context_hint_enabled:
            return ""
        if getattr(self, "_compacted_through", 0) == 0:
            return ""

        tag_summaries = self._store.get_all_tag_summaries()
        if not tag_summaries:
            return ""

        lines: list[str] = []
        for ts in tag_summaries:
            turn_count = len(ts.source_turn_numbers)
            # First ~60 chars of summary as description
            desc = ts.summary[:60].rstrip()
            if len(ts.summary) > 60:
                desc += "..."
            lines.append(f"- {ts.tag} ({turn_count} turns): {desc}")

        body = "\n".join(lines)
        hint = (
            "<context-topics>\n"
            "Prior conversation topics available for recall:\n"
            f"{body}\n"
            "</context-topics>"
        )

        # Truncate to budget
        max_tokens = self.config.assembler.context_hint_max_tokens
        if self._token_counter(hint) > max_tokens:
            # Trim lines from the end until within budget
            while lines and self._token_counter(hint) > max_tokens:
                lines.pop()
                body = "\n".join(lines)
                hint = (
                    "<context-topics>\n"
                    "Prior conversation topics available for recall:\n"
                    f"{body}\n"
                    "</context-topics>"
                )

        return hint

    def _get_latest_turn_pair(self, history: list[Message]) -> list[Message] | None:
        """Extract the most recent user+assistant pair."""
        if len(history) < 2:
            return None
        for i in range(len(history) - 1, 0, -1):
            if history[i].role == "assistant" and history[i-1].role == "user":
                return [history[i-1], history[i]]
        return None

    def compact_manual(
        self,
        conversation_history: list[Message],
    ) -> CompactionReport | None:
        """Trigger manual compaction regardless of thresholds."""
        if self._compactor is None:
            logger.warning("No LLM provider configured for compaction")
            return None

        if not conversation_history:
            return None

        segments = self._segmenter.segment(conversation_history)
        results = self._compactor.compact(segments)

        for result in results:
            stored = StoredSegment(
                ref=result.segment_id,
                session_id=self.config.session_id,
                primary_tag=result.primary_tag,
                tags=result.tags,
                summary=result.summary,
                summary_tokens=result.summary_tokens,
                full_text=result.full_text,
                full_tokens=result.original_tokens,
                messages=result.messages,
                metadata=result.metadata,
                compaction_model=self._compactor.model_name,
                compression_ratio=result.compression_ratio,
                start_timestamp=result.timestamp,
                end_timestamp=result.timestamp,
            )
            self._store.store_segment(stored)

        tokens_freed = sum(r.original_tokens - r.summary_tokens for r in results)
        tags = list({tag for r in results for tag in r.tags})

        return CompactionReport(
            segments_compacted=len(results),
            tokens_freed=tokens_freed,
            tags=tags,
            results=results,
        )

    def retrieve(self, message: str, active_tags: list[str] | None = None) -> RetrievalResult:
        """Retrieve relevant context for a message without assembling."""
        return self._retriever.retrieve(message, current_active_tags=active_tags or [])

    def transform(self, message: str, active_tags: list[str] | None = None, budget: int | None = None) -> str:
        """Retrieve + assemble context block for a message. Returns prepend text."""
        result = self.retrieve(message, active_tags)
        if not result.summaries:
            return ""
        core_context = self._assembler.load_core_context()
        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=result,
            conversation_history=[],
            token_budget=budget or self.config.context_window,
        )
        return assembled.prepend_text

    def get_cost_report(self) -> SessionCostSummary:
        """Return current session cost summary."""
        return self._cost_tracker.get_summary()

    def cleanup(self, max_age_days: int | None = None, max_total_tokens: int | None = None) -> int:
        """Run cleanup on the store. Returns count of segments deleted."""
        from datetime import timedelta
        max_age = timedelta(days=max_age_days) if max_age_days else None
        return self._store.cleanup(max_age=max_age, max_total_tokens=max_total_tokens)
