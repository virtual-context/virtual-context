"""VirtualContextEngine: main orchestrator wiring all components together."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .classifiers import ClassifierPipeline, KeywordClassifier
from .config import load_config
from .core.assembler import ContextAssembler
from .core.compactor import DomainCompactor
from .core.monitor import ContextMonitor
from .core.retriever import ContextRetriever
from .core.segmenter import TopicSegmenter
from .storage.filesystem import FilesystemStore
from .token_counter import create_token_counter
from .types import (
    AssembledContext,
    CompactionReport,
    CompactionResult,
    Message,
    StoredSegment,
    VirtualContextConfig,
)

logger = logging.getLogger(__name__)


class VirtualContextEngine:
    """Main orchestrator: two entry points for inbound messages and turn completion.

    Usage:
        engine = VirtualContextEngine(config_path="./virtual-context.yaml")

        # Before sending to LLM
        assembled = await engine.on_message_inbound(message, history)

        # After LLM responds
        report = await engine.on_turn_complete(history)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: VirtualContextConfig | None = None,
    ) -> None:
        self.config = config or load_config(config_path)
        self._token_counter = create_token_counter(self.config.token_counter)

        # Initialize components
        self._init_classifier()
        self._init_store()
        self._init_monitor()
        self._init_segmenter()
        self._init_assembler()
        self._init_retriever()
        self._init_compactor()
        self._initialized = False

    def _init_classifier(self) -> None:
        """Build the classifier pipeline from config."""
        classifiers = []
        for entry in self.config.classifier_pipeline:
            ctype = entry.get("type", "keyword")
            if ctype == "keyword":
                classifiers.append(KeywordClassifier())
            # embedding and llm classifiers deferred to v0.2/v0.3
        if not classifiers:
            classifiers.append(KeywordClassifier())
        self._classifier = ClassifierPipeline(
            classifiers=classifiers,
            min_confidence=self.config.segmenter.min_confidence,
        )

    def _init_store(self) -> None:
        """Initialize the storage backend."""
        self._store = FilesystemStore(root=self.config.storage.root)

    def _init_monitor(self) -> None:
        self._monitor = ContextMonitor(
            config=self.config.monitor,
            token_counter=self._token_counter,
        )

    def _init_segmenter(self) -> None:
        self._segmenter = TopicSegmenter(
            classifier_pipeline=self._classifier,
            config=self.config.segmenter,
            domains=list(self.config.domains.values()),
            token_counter=self._token_counter,
        )

    def _init_assembler(self) -> None:
        self._assembler = ContextAssembler(
            config=self.config.assembler,
            token_counter=self._token_counter,
        )

    def _init_retriever(self) -> None:
        self._retriever = ContextRetriever(
            classifier_pipeline=self._classifier,
            store=self._store,
            config=self.config.retriever,
        )

    def _init_compactor(self) -> None:
        """Initialize the compactor with an LLM provider."""
        self._llm_provider = None
        self._compactor = None

        provider_name = self.config.summarization.provider
        provider_config = self.config.providers.get(provider_name, {})

        if provider_config.get("type") == "anthropic" or provider_name == "anthropic":
            api_key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
            api_key = provider_config.get("api_key") or os.environ.get(api_key_env, "")

            if api_key:
                from .providers.anthropic import AnthropicProvider

                self._llm_provider = AnthropicProvider(
                    api_key=api_key,
                    model=self.config.summarization.model,
                    temperature=self.config.summarization.temperature,
                )

        if self._llm_provider:
            self._compactor = DomainCompactor(
                llm_provider=self._llm_provider,
                config=self.config.compactor,
                token_counter=self._token_counter,
                model_name=self.config.summarization.model,
            )

    async def initialize(self) -> None:
        """Initialize async components (classifier pipeline)."""
        if not self._initialized:
            await self._classifier.initialize(list(self.config.domains.values()))
            self._initialized = True

    async def on_message_inbound(
        self,
        message: str,
        conversation_history: list[Message],
    ) -> AssembledContext:
        """Before sending to LLM: classify, retrieve, assemble enriched context."""
        await self.initialize()

        # Determine active domains from recent conversation
        active_domains = self._get_active_domains(conversation_history)

        # Retrieve relevant domain summaries
        retrieval_result = await self._retriever.retrieve(
            message=message,
            current_domains_in_context=active_domains,
            conversation_history=conversation_history,
        )

        # Load core context
        core_context = self._assembler.load_core_context()

        # Assemble enriched context
        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=retrieval_result,
            conversation_history=conversation_history,
            token_budget=self.config.context_window,
        )

        return assembled

    async def on_turn_complete(
        self,
        conversation_history: list[Message],
    ) -> CompactionReport | None:
        """After LLM responds: check thresholds, compact if needed."""
        await self.initialize()

        # Build snapshot
        snapshot = self._monitor.build_snapshot(conversation_history)

        # Check thresholds
        signal = self._monitor.check(snapshot)

        if signal is None:
            return None

        if self._compactor is None:
            logger.warning(
                "Compaction triggered but no LLM provider configured. "
                "Set ANTHROPIC_API_KEY or configure a provider."
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

        # Messages to compact: everything before protected zone
        to_compact = conversation_history[:-protected_count]

        # Accumulate messages until we have enough tokens to free
        target_tokens = int(signal.overflow_tokens * self.config.compactor.overflow_buffer)
        compact_messages: list[Message] = []
        compact_tokens = 0

        for msg in to_compact:
            compact_messages.append(msg)
            compact_tokens += self._token_counter(msg.content)
            if compact_tokens >= target_tokens:
                break

        if not compact_messages:
            return None

        # Segment and compact
        segments = await self._segmenter.segment(compact_messages)
        results = await self._compactor.compact(segments)

        # Store compacted segments
        for result in results:
            stored = StoredSegment(
                ref=result.segment_id,
                session_id=self.config.session_id,
                domain=result.domain,
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
            await self._store.store_segment(stored)

        tokens_freed = sum(r.original_tokens - r.summary_tokens for r in results)
        domains = list({r.domain for r in results})

        return CompactionReport(
            segments_compacted=len(results),
            tokens_freed=tokens_freed,
            domains=domains,
            results=results,
        )

    def _get_active_domains(self, history: list[Message]) -> list[str]:
        """Get domains present in recent conversation turns."""
        lookback = self.config.retriever.active_domain_lookback * 2
        recent = history[-lookback:] if len(history) > lookback else history

        # Simple heuristic: check keywords in recent messages
        active: set[str] = set()
        recent_text = " ".join(m.content for m in recent).lower()

        for domain_def in self.config.domains.values():
            if domain_def.name == "_general":
                continue
            if domain_def.keywords:
                matches = sum(1 for kw in domain_def.keywords if kw.lower() in recent_text)
                if matches >= 2:
                    active.add(domain_def.name)

        return list(active)

    async def compact_manual(
        self,
        conversation_history: list[Message],
    ) -> CompactionReport | None:
        """Trigger manual compaction regardless of thresholds."""
        await self.initialize()

        if self._compactor is None:
            logger.warning("No LLM provider configured for compaction")
            return None

        if not conversation_history:
            return None

        segments = await self._segmenter.segment(conversation_history)
        results = await self._compactor.compact(segments)

        for result in results:
            stored = StoredSegment(
                ref=result.segment_id,
                session_id=self.config.session_id,
                domain=result.domain,
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
            await self._store.store_segment(stored)

        tokens_freed = sum(r.original_tokens - r.summary_tokens for r in results)
        domains = list({r.domain for r in results})

        return CompactionReport(
            segments_compacted=len(results),
            tokens_freed=tokens_freed,
            domains=domains,
            results=results,
        )
