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
    EngineStateSnapshot,
    Message,
    RetrievalResult,
    SessionCostSummary,
    SplitResult,
    StoredSegment,
    StoredSummary,
    TagResult,
    TagSummary,
    TurnTagEntry,
    VirtualContextConfig,
)

logger = logging.getLogger(__name__)

_EMBED_NOT_LOADED = object()  # sentinel for lazy embed function loading


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
        self._init_tag_splitter()
        self._init_cost_tracker()
        self._compacted_through = 0  # message index watermark: messages before this already compacted
        self._embed_fn = _EMBED_NOT_LOADED  # lazy-loaded for context bleed gate
        self._split_processed_tags: set[str] = set()
        self._last_split_result: SplitResult | None = None

        # Restore persisted state if available
        self._load_persisted_state()

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
        inbound_tagger = None
        if self.config.retriever.inbound_tagger_type == "embedding":
            inbound_tagger = self._build_inbound_embedding_tagger()

        self._retriever = ContextRetriever(
            tag_generator=self._tag_generator,
            store=self._store,
            config=self.config.retriever,
            turn_tag_index=self._turn_tag_index,
            inbound_tagger=inbound_tagger,
        )

    def _build_inbound_embedding_tagger(self) -> TagGenerator:
        """Build an EmbeddingTagGenerator for inbound vocabulary matching."""
        from .core.embedding_tag_generator import EmbeddingTagGenerator

        logger.info(
            "Using embedding-based inbound matching (model=%s, threshold=%.2f)",
            self.config.retriever.embedding_model,
            self.config.retriever.embedding_threshold,
        )
        return EmbeddingTagGenerator(
            config=self.config.tag_generator,
            model_name=self.config.retriever.embedding_model,
            similarity_threshold=self.config.retriever.embedding_threshold,
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

    def _init_tag_splitter(self) -> None:
        """Initialize tag splitter if enabled in config."""
        self._tag_splitter = None
        cfg = self.config.tag_generator.tag_splitting
        if cfg.enabled and self._llm_provider:
            from .core.tag_splitter import TagSplitter
            self._tag_splitter = TagSplitter(
                llm=self._llm_provider,
                config=cfg,
            )

    def _init_cost_tracker(self) -> None:
        self._cost_tracker = CostTracker(config=self.config.cost_tracking)

    def _load_persisted_state(self) -> None:
        """Restore TurnTagIndex and compaction watermark from store if available."""
        try:
            saved = self._store.load_engine_state(self.config.session_id)
        except Exception:
            return
        if not saved:
            return
        self.config.session_id = saved.session_id
        self._compacted_through = saved.compacted_through
        self._turn_tag_index = TurnTagIndex()
        for entry in saved.turn_tag_entries:
            self._turn_tag_index.append(entry)
        self._split_processed_tags = set(saved.split_processed_tags)
        logger.info(
            "Restored engine state: session=%s, compacted_through=%d, turns=%d, split_processed=%d",
            saved.session_id[:12], saved.compacted_through,
            len(saved.turn_tag_entries), len(saved.split_processed_tags),
        )

    def _save_state(self, conversation_history: list[Message]) -> None:
        """Persist current engine state to store."""
        try:
            self._store.save_engine_state(EngineStateSnapshot(
                session_id=self.config.session_id,
                compacted_through=self._compacted_through,
                turn_tag_entries=list(self._turn_tag_index.entries),
                turn_count=len(conversation_history) // 2,
                split_processed_tags=sorted(self._split_processed_tags),
            ))
        except Exception as e:
            logger.error("Failed to save engine state: %s", e)

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

        # Build context for inbound tagger
        n_context = self.config.tag_generator.context_lookback_pairs
        # For inbound, the current message is not yet in history — no need to exclude
        context = self._get_recent_context(
            conversation_history, n_context, exclude_last=0,
        )

        # Retrieve relevant tag summaries
        retrieval_result = self._retriever.retrieve(
            message=message,
            current_active_tags=active_tags,
            current_utilization=utilization,
            post_compaction=(self._compacted_through > 0),
            context_turns=context,
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

        # Retry with expanded context if only _general was produced
        if message_tags == ["_general"]:
            expanded = self._get_recent_context(
                conversation_history, n_context * 2, exclude_last=0,
            )
            if expanded:
                retry_result = self._retriever.retrieve(
                    message=message,
                    current_active_tags=active_tags,
                    current_utilization=utilization,
                    post_compaction=(self._compacted_through > 0),
                    context_turns=expanded,
                )
                retry_tags = retry_result.retrieval_metadata.get(
                    "tags_from_message", retry_result.tags_matched
                )
                if retry_tags != ["_general"]:
                    message_tags = retry_tags
                    retrieval_result = retry_result

        # Final fallback: inherit from most recent meaningful turn in the index
        if message_tags == ["_general"]:
            prev = self._turn_tag_index.latest_meaningful_tags()
            if prev:
                message_tags = list(prev.tags)

        assembled.matched_tags = message_tags
        assembled.context_hint = context_hint
        assembled.broad = retrieval_result.broad
        assembled.temporal = retrieval_result.temporal

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

            # BUG-013: Skip empty turns (tool_use/tool_result with no text)
            if not combined_text.strip():
                latest_pair = None

        if latest_pair:
            store_tags = [ts.tag for ts in self._store.get_all_tags()]
            n_context = self.config.tag_generator.context_lookback_pairs
            context = self._get_recent_context(
                conversation_history, n_context, current_text=combined_text,
            )
            tag_result = self._tag_generator.generate_tags(
                combined_text, store_tags, context_turns=context,
            )

            # Retry with expanded context if only _general was produced
            if tag_result.tags == ["_general"]:
                expanded = self._get_recent_context(
                    conversation_history, n_context * 2,
                    current_text=combined_text,
                )
                if expanded:
                    tag_result = self._tag_generator.generate_tags(
                        combined_text, store_tags, context_turns=expanded,
                    )

            # Final fallback: inherit from most recent meaningful turn
            if tag_result.tags == ["_general"]:
                prev = self._turn_tag_index.latest_meaningful_tags()
                if prev:
                    tag_result = TagResult(
                        tags=list(prev.tags),
                        primary=prev.primary_tag,
                        source="inherited",
                    )

            self._turn_tag_index.append(TurnTagEntry(
                turn_number=len(self._turn_tag_index.entries),
                message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                tags=tag_result.tags,
                primary_tag=tag_result.primary,
            ))

        # Check for overly-broad tags needing splitting
        if self._tag_splitter:
            self._check_and_split_broad_tags(conversation_history)

        # Build snapshot (only count un-compacted messages)
        snapshot = self._monitor.build_snapshot(
            conversation_history[self._compacted_through:]
        )

        # Check thresholds
        signal = self._monitor.check(snapshot)

        if signal is None:
            self._save_state(conversation_history)
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

        self._save_state(conversation_history)
        return report

    def _check_and_split_broad_tags(
        self, conversation_history: list[Message],
    ) -> SplitResult | None:
        """Check for overly-broad tags and split or summarize them."""
        if not self._tag_splitter:
            return None

        cfg = self.config.tag_generator.tag_splitting
        tag_counts = self._turn_tag_index.get_tag_counts()
        total_turns = len(self._turn_tag_index.entries)

        if total_turns == 0:
            return None

        # Find candidates: above both thresholds, not already processed
        candidates = [
            (tag, count) for tag, count in tag_counts.items()
            if tag != "_general"
            and tag not in self._split_processed_tags
            and count >= cfg.frequency_threshold
            and count / total_turns >= cfg.frequency_pct_threshold
        ]

        if not candidates:
            return None

        # Pick highest-frequency first
        candidates.sort(key=lambda x: -x[1])
        tag, count = candidates[0]

        # Collect turn content
        turn_contents = self._collect_turn_text(tag, conversation_history)
        if not turn_contents:
            self._split_processed_tags.add(tag)
            return None

        existing_tags = {t for e in self._turn_tag_index.entries for t in e.tags}
        result = self._tag_splitter.split(tag, turn_contents, existing_tags, total_turns)

        if result.splittable:
            # Apply split to TurnTagIndex
            turn_to_new: dict[int, list[str]] = {}
            for new_tag, turn_numbers in result.groups.items():
                for tn in turn_numbers:
                    turn_to_new.setdefault(tn, []).append(new_tag)
            self._turn_tag_index.replace_tag(tag, turn_to_new)

            # Register alias so old tag queries still resolve
            if self._canonicalizer:
                first_new = next(iter(result.groups))
                self._canonicalizer.register_alias(tag, first_new)

            # Update tagger vocabulary
            if hasattr(self._tag_generator, '_tag_vocabulary'):
                self._tag_generator._tag_vocabulary.pop(tag, None)
                for new_tag, turns in result.groups.items():
                    self._tag_generator._tag_vocabulary[new_tag] = len(turns)

            logger.info(
                "Split '%s' (%d turns) → %s",
                tag, count, list(result.groups.keys()),
            )
        else:
            # Fallback: build tag summary from raw turn text
            self._build_broad_tag_summary(tag, conversation_history)
            logger.info(
                "Tag '%s' unsplittable (%s), built summary", tag, result.reason,
            )

        self._split_processed_tags.add(tag)
        self._last_split_result = result
        return result

    @staticmethod
    def _extract_turn_pairs(history: list[Message]) -> list[tuple[str, str]]:
        """Extract user→assistant turn pairs from history, handling non-alternating messages.

        Returns list of (user_text, assistant_text) tuples. Skips preamble-only
        user messages (e.g., MemOS '# Role' injections) and handles consecutive
        user messages by using the last user message before each assistant response.
        """
        pairs: list[tuple[str, str]] = []
        last_user_text = ""
        for msg in history:
            if msg.role == "user":
                last_user_text = msg.content
            elif msg.role == "assistant" and last_user_text:
                pairs.append((last_user_text, msg.content))
                last_user_text = ""
        return pairs

    def _collect_turn_text(
        self, tag: str, history: list[Message],
    ) -> list[tuple[int, str]]:
        """Collect truncated user text for turns tagged with the given tag."""
        pairs = self._extract_turn_pairs(history)
        result = []
        for entry in self._turn_tag_index.entries:
            if tag in entry.tags:
                if entry.turn_number < len(pairs):
                    text = pairs[entry.turn_number][0][:200]
                    result.append((entry.turn_number, text))
        return result

    def _build_broad_tag_summary(
        self, tag: str, history: list[Message],
    ) -> None:
        """Build a tag summary directly from raw turn text for unsplittable broad tags."""
        if not self._compactor:
            return

        pairs = self._extract_turn_pairs(history)
        texts = []
        turn_numbers = []
        for entry in self._turn_tag_index.entries:
            if tag in entry.tags:
                if entry.turn_number < len(pairs):
                    user_text, assistant_text = pairs[entry.turn_number]
                    texts.append(
                        f"User: {user_text[:300]}\n"
                        f"Assistant: {assistant_text[:300]}"
                    )
                    turn_numbers.append(entry.turn_number)

        if not texts:
            return

        combined = "\n\n---\n\n".join(texts)
        max_turn = max(turn_numbers) if turn_numbers else 0

        synthetic = [StoredSummary(
            ref=f"broad-{tag}",
            tags=[tag],
            summary=combined[:4000],
            summary_tokens=len(combined[:4000]) // 4,
        )]
        summaries = self._compactor.compact_tag_summaries(
            cover_tags=[tag],
            tag_to_summaries={tag: synthetic},
            tag_to_turns={tag: turn_numbers},
            existing_tag_summaries={},
            max_turn=max_turn,
        )
        for ts in summaries:
            self._store.save_tag_summary(ts)

    def filter_history(
        self,
        conversation_history: list[Message],
        current_tags: list[str],
        recent_turns: int | None = None,
        broad: bool = False,
        temporal: bool = False,
    ) -> list[Message]:
        """Filter conversation history by tag relevance.

        When ``broad`` or ``temporal`` is True, all remaining turns are included
        without tag-based filtering.  Pre-compaction the full history fits within
        the context window; post-compaction old turns are already gone, so "all
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

        # Broad or temporal query — include everything, but skip compacted messages
        # (summaries from retriever replace them)
        if broad or temporal:
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

    def _get_embed_fn(self):
        """Lazy-load the embedding function for the context bleed gate.

        Returns a callable that takes a list of strings and returns a list of
        float vectors, or ``None`` if sentence-transformers is not installed.
        """
        if self._embed_fn is _EMBED_NOT_LOADED:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.config.retriever.embedding_model
                model = SentenceTransformer(model_name)

                def embed(texts: list[str]) -> list[list[float]]:
                    return model.encode(texts, convert_to_numpy=True).tolist()

                self._embed_fn = embed
            except ImportError:
                logger.debug(
                    "sentence-transformers not installed, context bleed gate disabled"
                )
                self._embed_fn = None
        return self._embed_fn

    def _context_is_relevant(
        self, current_text: str, context_pairs: list[str],
    ) -> bool:
        """Check if current turn is semantically similar to the most recent context pair.

        Compares the current turn's combined text against the last user+assistant
        pair in the collected context using embedding cosine similarity.
        Returns ``True`` (pass context) when similarity >= threshold, or when
        embeddings are unavailable (graceful degradation).
        """
        embed_fn = self._get_embed_fn()
        if embed_fn is None:
            return True

        # Compare against the most recent pair in context
        if len(context_pairs) >= 2:
            recent = context_pairs[-2] + " " + context_pairs[-1]
        else:
            recent = " ".join(context_pairs)

        embeddings = embed_fn([current_text[:2000], recent[:2000]])
        sim = _cosine_sim(embeddings[0], embeddings[1])
        threshold = self.config.tag_generator.context_bleed_threshold

        logger.debug("Context bleed gate: sim=%.3f threshold=%.3f", sim, threshold)
        return sim >= threshold

    def _get_recent_context(
        self, history: list[Message], n_pairs: int, exclude_last: int = 2,
        current_text: str | None = None,
    ) -> list[str] | None:
        """Collect up to *n_pairs* recent user+assistant text strings.

        Walks backward from the end of *history* (skipping the last
        *exclude_last* messages which are the current turn) and returns
        alternating user/assistant content strings.

        When *current_text* is provided and ``context_bleed_threshold > 0``,
        an embedding similarity gate checks whether the current turn is
        semantically related to the most recent context pair.  If the
        similarity is below the threshold (topic shift), context is skipped
        to prevent stale tags from bleeding across topics (BUG-010).

        Returns ``None`` when no context is available or when the gate blocks.
        """
        # Messages available for context (before the current turn)
        if exclude_last > 0 and len(history) > exclude_last:
            avail = history[:-exclude_last]
        elif exclude_last == 0:
            avail = list(history)
        else:
            avail = []
        if not avail:
            return None

        pairs: list[str] = []
        # Walk backward collecting user+assistant pairs
        i = len(avail) - 1
        collected = 0
        while i >= 1 and collected < n_pairs:
            if avail[i].role == "assistant" and avail[i - 1].role == "user":
                # Prepend so order is chronological
                pairs.insert(0, avail[i].content)
                pairs.insert(0, avail[i - 1].content)
                collected += 1
                i -= 2
            else:
                i -= 1

        if not pairs:
            return None

        # Context bleed gate (BUG-010): skip context on topic shift
        if (
            current_text
            and self.config.tag_generator.context_bleed_threshold > 0
            and not self._context_is_relevant(current_text, pairs)
        ):
            logger.debug("Context bleed gate: topic shift detected, skipping context")
            return None

        return pairs

    def compact_manual(
        self,
        conversation_history: list[Message],
    ) -> CompactionReport | None:
        """Trigger manual compaction regardless of thresholds.

        Uses the same pipeline as on_turn_complete: respects the compaction
        watermark, protected recent turns, advances the watermark, stores
        segments, and rebuilds tag summaries for affected tags.
        """
        if self._compactor is None:
            logger.warning("No LLM provider configured for compaction")
            return None

        if not conversation_history:
            return None

        # Select messages to compact (same logic as on_turn_complete)
        protected_turns = self.config.monitor.protected_recent_turns
        protected_count = protected_turns * 2

        if len(conversation_history) <= protected_count:
            logger.info("Not enough messages outside protected zone to compact")
            return None

        compact_messages = conversation_history[self._compacted_through:-protected_count]

        if not compact_messages:
            return None

        # Segment and compact (turn_offset maps pair indices to global turn numbers)
        turn_offset = self._compacted_through // 2
        segments = self._segmenter.segment(compact_messages, turn_offset=turn_offset)
        results = self._compactor.compact(segments)

        # Advance watermark past compacted messages
        self._compacted_through += len(compact_messages)

        # Store compacted segments
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
        if results:
            compacted_tags = {tag for r in results for tag in r.tags}
            cover_tags = [
                t for t in self._turn_tag_index.compute_cover_set()
                if t in compacted_tags
            ]
            if cover_tags:
                tag_to_summaries: dict[str, list] = {}
                for tag in cover_tags:
                    summaries = self._store.get_summaries_by_tags(
                        tags=[tag], min_overlap=1, limit=50
                    )
                    if summaries:
                        tag_to_summaries[tag] = summaries

                tag_to_turns: dict[str, list[int]] = {}
                for entry in self._turn_tag_index.entries:
                    for tag in entry.tags:
                        if tag in cover_tags:
                            tag_to_turns.setdefault(tag, []).append(entry.turn_number)

                existing_tag_summaries = {}
                for tag in cover_tags:
                    ts = self._store.get_tag_summary(tag)
                    if ts:
                        existing_tag_summaries[tag] = ts

                if self._turn_tag_index.entries:
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

        self._save_state(conversation_history)
        return report

    def ingest_history(
        self,
        history_pairs: list[Message],
        progress_callback: callable | None = None,
    ) -> int:
        """Bootstrap TurnTagIndex from pre-existing conversation history.

        Tags each user+assistant pair and appends entries to the live index.
        Does NOT trigger compaction — the next on_turn_complete() handles that.

        Args:
            history_pairs: Flat list [user_0, asst_0, user_1, asst_1, ...].
            progress_callback: Optional ``(done, total, entry)`` called after
                each turn is ingested.  Used by the proxy for live progress.

        Returns:
            Number of turns ingested.
        """
        store_tags = [ts.tag for ts in self._store.get_all_tags()]
        ingested = 0
        n_context = self.config.tag_generator.context_lookback_pairs

        for i in range(0, len(history_pairs) - 1, 2):
            user_msg = history_pairs[i]
            asst_msg = history_pairs[i + 1]

            # BUG-013: Skip empty turns (tool_use/tool_result with no text)
            if not user_msg.content.strip() and not asst_msg.content.strip():
                logger.debug("Skipping empty turn at pair index %d", i // 2)
                continue

            combined_text = f"{user_msg.content} {asst_msg.content}"

            # Build context from preceding pairs in the flat history
            context: list[str] | None = None
            if i >= 2:
                ctx_pairs: list[str] = []
                start = max(0, i - n_context * 2)
                for j in range(start, i, 2):
                    if j + 1 < len(history_pairs):
                        ctx_pairs.append(history_pairs[j].content)
                        ctx_pairs.append(history_pairs[j + 1].content)
                context = ctx_pairs if ctx_pairs else None

            # Context bleed gate (BUG-010): skip stale context on topic shift
            if (
                context
                and self.config.tag_generator.context_bleed_threshold > 0
                and not self._context_is_relevant(combined_text, context)
            ):
                context = None

            tag_result = self._tag_generator.generate_tags(
                combined_text, store_tags, context_turns=context,
            )

            # Retry with expanded context on _general
            if tag_result.tags == ["_general"] and i >= 2:
                expanded_start = max(0, i - n_context * 4)
                expanded_ctx: list[str] = []
                for j in range(expanded_start, i, 2):
                    if j + 1 < len(history_pairs):
                        expanded_ctx.append(history_pairs[j].content)
                        expanded_ctx.append(history_pairs[j + 1].content)
                # Gate expanded context too
                if (
                    expanded_ctx
                    and self.config.tag_generator.context_bleed_threshold > 0
                    and not self._context_is_relevant(combined_text, expanded_ctx)
                ):
                    expanded_ctx = []
                if expanded_ctx:
                    tag_result = self._tag_generator.generate_tags(
                        combined_text, store_tags, context_turns=expanded_ctx,
                    )

            # Final fallback: inherit from most recent meaningful turn
            if tag_result.tags == ["_general"]:
                prev = self._turn_tag_index.latest_meaningful_tags()
                if prev:
                    tag_result = TagResult(
                        tags=list(prev.tags),
                        primary=prev.primary_tag,
                        source="inherited",
                    )

            entry = TurnTagEntry(
                turn_number=i // 2,
                message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                tags=tag_result.tags,
                primary_tag=tag_result.primary,
            )
            self._turn_tag_index.append(entry)
            ingested += 1

            if progress_callback:
                total = len(history_pairs) // 2
                progress_callback(ingested, total, entry)

            # Refresh store tags every 10 turns so new tags influence later tagging
            if ingested % 10 == 0:
                store_tags = [ts.tag for ts in self._store.get_all_tags()]

        logger.info("Ingested %d historical turns into TurnTagIndex", ingested)
        return ingested

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
