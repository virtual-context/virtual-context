"""VirtualContextEngine: main orchestrator wiring all components together."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import load_config
from .core.assembler import ContextAssembler
from .core.hint_builder import build_autonomous_hint, build_supervised_hint, build_default_hint
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
    ChunkEmbedding,
    CompactionReport,
    CompactionResult,
    CompactionSignal,
    DepthLevel,
    EngineStateSnapshot,
    Message,
    QuoteResult,
    RetrievalResult,
    SegmentMetadata,
    SessionCostSummary,
    SplitResult,
    StoredSegment,
    StoredSummary,
    TagResult,
    TagSummary,
    ToolLoopResult,
    TurnTagEntry,
    VirtualContextConfig,
    WorkingSetEntry,
)

logger = logging.getLogger(__name__)

_SESSION_HEADER_RE = re.compile(r'\[Session from ([^\]]+)\]')
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


from .core.math_utils import cosine_similarity as _cosine_sim
from .core.paging_manager import PagingManager
from .core.quote_search import find_quote as _find_quote, supplement_from_descriptions as _supplement_from_descriptions
from .core.semantic_search import SemanticSearchManager, chunk_segment_text as _chunk_segment_text


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
        self._init_cost_tracker()
        self._init_canonicalizer()
        self._init_tag_generator()
        self._init_monitor()
        self._init_segmenter()
        self._init_assembler()
        self._init_retriever()
        self._init_compactor()
        self._init_tag_splitter()
        self._compacted_through = 0  # message index watermark: messages before this already compacted
        self._last_tag_ms: float = 0.0
        self._last_compact_ms: float = 0.0
        self._semantic = SemanticSearchManager(store=self._store, config=self.config)
        self._paging = PagingManager(
            store=self._store,
            token_counter=self._token_counter,
            tag_context_max_tokens=self.config.assembler.tag_context_max_tokens,
            auto_evict=self.config.paging.auto_evict,
            paging_enabled=self.config.paging.enabled,
        )
        self._split_processed_tags: set[str] = set()
        self._last_split_result: SplitResult | None = None
        self._trailing_fingerprint: str = ""  # set by proxy for session matching on restart

        # Restore persisted state if available
        self._load_persisted_state()
        # Re-sync segmenter's index reference — _load_persisted_state replaces
        # self._turn_tag_index with a new object, but the segmenter was initialized
        # with the old (empty) one.
        self._segmenter._turn_tag_index = self._turn_tag_index
        self._bootstrap_vocabulary()

    def close(self) -> None:
        """Release backend resources held by the engine."""
        store = getattr(self, "_store", None)
        if store is not None and hasattr(store, "close"):
            try:
                store.close()
            except Exception:
                logger.debug("Engine store close failed", exc_info=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def _embed_fn(self):
        """Proxy to SemanticSearchManager's embed function (for test compat)."""
        return self._semantic._embed_fn

    @_embed_fn.setter
    def _embed_fn(self, value):
        self._semantic._embed_fn = value

    @property
    def _working_set(self) -> dict[str, WorkingSetEntry]:
        """Proxy to PagingManager's working set (for backward compat)."""
        return self._paging.working_set

    @_working_set.setter
    def _working_set(self, value: dict[str, WorkingSetEntry]):
        self._paging.working_set = value

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
            self.config.tag_generator, llm_provider,
            canonicalizer=self._canonicalizer, cost_tracker=self._cost_tracker,
            embed_fn_factory=self._get_embed_fn,
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
                cost_tracker=self._cost_tracker,
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

    _COMPACT_BATCH_SIZE = 20  # segments per compaction batch → DB after each batch

    def _compact_and_store(
        self, segments: list, compact_messages_len: int,
    ) -> list[CompactionResult]:
        """Compact segments in batches of ``_COMPACT_BATCH_SIZE`` and store each
        batch immediately so results are visible in the DB incrementally."""
        from .types import FactSignal

        all_results: list[CompactionResult] = []
        batch_size = self._COMPACT_BATCH_SIZE

        # D1: Gather fact signals from TurnTagIndex for compacted turns.
        # Build a flat list per segment.  Since we don't have a direct
        # segment→turn mapping, collect all signals and let the compactor
        # verify against the segment's conversation text.
        turn_offset = self._compacted_through // 2
        compact_turns = compact_messages_len // 2
        all_signals: list[FactSignal] = []
        for t in range(turn_offset, turn_offset + compact_turns):
            entry = self._turn_tag_index.get_tags_for_turn(t)
            if entry and entry.fact_signals:
                all_signals.extend(entry.fact_signals)

        for start in range(0, len(segments), batch_size):
            batch = segments[start:start + batch_size]
            batch_num = start // batch_size + 1
            total_batches = (len(segments) + batch_size - 1) // batch_size
            logger.info(
                "Compacting batch %d/%d (%d segments)...",
                batch_num, total_batches, len(batch),
            )
            # D1: Pass all signals to the compactor; each segment's prompt
            # will include them as hints for verification.
            fact_signals_by_segment = {seg.id: all_signals for seg in batch} if all_signals else None
            results = self._compactor.compact(batch, fact_signals_by_segment=fact_signals_by_segment)
            # Store each result to DB right away
            for i, result in enumerate(results):
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
                self._embed_and_store_chunks(stored)
                # D1: Store extracted facts with provenance
                if result.facts:
                    for fact in result.facts:
                        fact.segment_ref = stored.ref
                        fact.session_id = self.config.session_id
                    self._store.store_facts(result.facts)
                    logger.info(
                        "  Stored %d facts for segment %s",
                        len(result.facts), result.primary_tag,
                    )
                session_date = getattr(result.metadata, 'session_date', '') if result.metadata else ''
                logger.info(
                    "  Stored segment %d/%d: %s (session_date=%s, %dt→%dt)",
                    start + i + 1, len(segments), result.primary_tag,
                    session_date or 'none',
                    result.original_tokens, result.summary_tokens,
                )
            all_results.extend(results)

        return all_results

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
        # Restore paging working set (backward-compatible: old snapshots have empty list)
        self._working_set = {ws.tag: ws for ws in (saved.working_set or [])}
        self._trailing_fingerprint = saved.trailing_fingerprint
        logger.info(
            "Restored engine state: session=%s, compacted_through=%d, turns=%d, split_processed=%d, working_set=%d",
            saved.session_id[:12], saved.compacted_through,
            len(saved.turn_tag_entries), len(saved.split_processed_tags),
            len(self._working_set),
        )

    def _bootstrap_vocabulary(self) -> None:
        """Load historical tag frequencies into the tagger's vocabulary.

        Called once at init after ``_load_persisted_state()``.  Populates the
        LLM tagger's vocabulary from two sources:

        1. **Store tags** — cross-session tag statistics (``get_all_tags()``).
        2. **TurnTagIndex** — restored session entries (higher priority).

        Without this, a freshly-started engine invents novel tags instead of
        reusing the established vocabulary (e.g. "ai-memory" instead of
        "skincare" for skincare-related content).
        """
        if not hasattr(self._tag_generator, "load_vocabulary"):
            return  # KeywordTagGenerator doesn't have this

        tag_counts: dict[str, int] = {}

        # Store tags (cross-session)
        for ts in self._store.get_all_tags():
            tag_counts[ts.tag] = ts.usage_count

        # TurnTagIndex (restored session, higher priority)
        for tag, count in self._turn_tag_index.get_tag_counts().items():
            tag_counts[tag] = max(tag_counts.get(tag, 0), count)

        if tag_counts:
            self._tag_generator.load_vocabulary(tag_counts)
            logger.info(
                "Bootstrapped tagger vocabulary: %d tags from store + index",
                len(tag_counts),
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
                working_set=list(self._working_set.values()),
                trailing_fingerprint=self._trailing_fingerprint,
            ))
        except Exception as e:
            logger.error("Failed to save engine state: %s", e)

    def _build_provider(self, provider_name: str, provider_config: dict):
        """Build an LLM provider from config."""
        ptype = provider_config.get("type", provider_name)

        if ptype == "generic_openai":
            from .providers.generic_openai import GenericOpenAIProvider
            api_key_env = provider_config.get("api_key_env", "")
            api_key = provider_config.get("api_key") or (
                os.environ.get(api_key_env, "") if api_key_env else "not-needed"
            )
            return GenericOpenAIProvider(
                base_url=provider_config.get("base_url", "http://127.0.0.1:11434/v1"),
                model=provider_config.get("model", self.config.summarization.model),
                temperature=self.config.summarization.temperature,
                api_key=api_key,
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
        model_name: str = "",
        max_context_tokens: int | None = None,
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

        # Build context for inbound tagger.
        # Include recent turns even post-compaction so query-time tagging can
        # use immediate conversational cues.
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
        _paging_mode = self._resolve_paging_mode(model_name) if self.config.paging.enabled else None
        context_hint = self._build_context_hint(paging_mode=_paging_mode)

        # Load core context
        core_context = self._assembler.load_core_context()

        # Paging: load content at working set depth levels
        # Working-set paging applies uniformly; time-scoped retrieval now runs
        # through vc_remember_when instead of a temporal retrieval branch.
        ws_param = None
        full_segments_param = None
        if self.config.paging.enabled and self._working_set:
            ws_param = self._working_set
            # Load full segments for tags at SEGMENTS or FULL depth
            full_segments_param = {}
            for tag, entry in self._working_set.items():
                if entry.depth in (DepthLevel.SEGMENTS, DepthLevel.FULL):
                    segs = self._store.get_segments_by_tags(tags=[tag], min_overlap=1, limit=50)
                    if segs:
                        full_segments_param[tag] = segs
                # Update last_accessed_turn for tags matched by current query
                query_tags = retrieval_result.retrieval_metadata.get("tags_from_message", [])
                if tag in query_tags:
                    entry.last_accessed_turn = len(self._turn_tag_index.entries)

        # Assemble enriched context
        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=retrieval_result,
            conversation_history=conversation_history,
            token_budget=self.config.context_window,
            context_hint=context_hint,
            working_set=ws_param,
            full_segments=full_segments_param,
            max_context_tokens=max_context_tokens,
        )

        # Expose the message's own tags for downstream use (e.g. history filtering).
        # Use tags_from_message (what the tag generator produced for this message)
        # rather than tags_matched (which only includes tags found in the store).
        message_tags = retrieval_result.retrieval_metadata.get(
            "tags_from_message", retrieval_result.tags_matched
        )

        # Retry with expanded context if only _general was produced.
        # Applies both pre- and post-compaction.
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

        # Cache for reassemble_context() — used after paging tool execution
        self._last_retrieval_result = retrieval_result
        self._last_conversation_history = conversation_history
        self._last_model_name = model_name

        return assembled

    def tag_turn(
        self,
        conversation_history: list[Message],
        payload_tokens: int | None = None,
    ) -> CompactionSignal | None:
        """Phase 1 of turn processing: tag the latest turn and check thresholds.

        Fast (~2-3s with LLM tagger). Must complete before the next inbound
        request so the turn-tag index is up-to-date for retrieval.

        Returns a CompactionSignal if compaction is needed, None otherwise.

        *payload_tokens* (proxy mode): actual client payload token count.
        Overrides the stripped conversation_history token count in the
        compaction monitor so thresholds trigger at the right level.
        """
        # Tag the latest round trip
        _t_tag = time.monotonic()
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
                fact_signals=tag_result.fact_signals,
            ))

        self._last_tag_ms = round((time.monotonic() - _t_tag) * 1000, 1)

        # Check for overly-broad tags needing splitting
        if self._tag_splitter:
            self._check_and_split_broad_tags(conversation_history)

        # Build snapshot (only count un-compacted messages)
        snapshot = self._monitor.build_snapshot(
            conversation_history[self._compacted_through:],
            payload_tokens=payload_tokens,
        )

        # Check thresholds
        signal = self._monitor.check(snapshot)

        if signal is None:
            self._last_compact_ms = 0.0
            self._save_state(conversation_history)

        return signal

    def compact_if_needed(
        self,
        conversation_history: list[Message],
        signal: CompactionSignal,
    ) -> CompactionReport | None:
        """Phase 2 of turn processing: run compaction.

        Slow (~10s with LLM summarizer). Can run in background after
        tag_turn() completes — the next request only needs the tag index.

        *signal*: the CompactionSignal returned by tag_turn().
        """
        _t_compact = time.monotonic()

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

        # Segment and compact in batches (results stored to DB incrementally)
        turn_offset = self._compacted_through // 2
        segments = self._segmenter.segment(compact_messages, turn_offset=turn_offset)
        logger.info(
            "Segmented %d messages into %d segments (watermark=%d)",
            len(compact_messages), len(segments), self._compacted_through,
        )
        results = self._compact_and_store(segments, len(compact_messages))

        # Advance watermark past compacted messages
        self._compacted_through += len(compact_messages)

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

        self._last_compact_ms = round((time.monotonic() - _t_compact) * 1000, 1)
        self._save_state(conversation_history)
        return report

    def on_turn_complete(
        self,
        conversation_history: list[Message],
        payload_tokens: int | None = None,
    ) -> CompactionReport | None:
        """Full tag+compact cycle (blocking).

        Convenience wrapper that calls tag_turn() then compact_if_needed().
        Used by CLI, tests, and non-proxy callers that don't need async
        compaction.
        """
        signal = self.tag_turn(conversation_history, payload_tokens)
        if signal is None:
            return None
        return self.compact_if_needed(conversation_history, signal)

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
    ) -> list[Message]:
        """Filter conversation history by tag relevance.

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

    def _build_context_hint(self, paging_mode: str | None = None) -> str:
        """Build a topic list for post-compaction prompts.

        *paging_mode* overrides the resolved mode (``"autonomous"`` or
        ``"supervised"``).  The proxy passes this from the per-request model
        check; headless/MCP callers omit it and the method resolves from config.

        When paging is enabled, builds a richer hint with depth info and budget.
        Mode determines detail level:
        - supervised: topic list with depth, "call expand_topic for detail"
        - autonomous: full budget dashboard with token costs

        Returns empty string if compaction hasn't occurred or the feature is disabled.
        """
        if not self.config.assembler.context_hint_enabled:
            return ""
        if getattr(self, "_compacted_through", 0) == 0:
            return ""

        tag_summaries = self._store.get_all_tag_summaries()
        if not tag_summaries:
            return ""

        # Determine paging mode
        paging_enabled = self.config.paging.enabled
        if paging_mode is None and paging_enabled:
            paging_mode = self._resolve_paging_mode()

        if paging_enabled and paging_mode == "autonomous":
            hint = self._build_autonomous_hint(tag_summaries)
        elif paging_enabled and paging_mode == "supervised":
            hint = self._build_supervised_hint(tag_summaries)
        else:
            hint = self._build_default_hint(tag_summaries)

        return hint

    def _build_autonomous_hint(self, tag_summaries: list) -> str:
        return build_autonomous_hint(
            tag_summaries=tag_summaries,
            working_set=self._working_set,
            budget=self.config.assembler.tag_context_max_tokens,
            max_hint_tokens=self.config.assembler.context_hint_max_tokens,
            token_counter=self._token_counter,
            calculate_depth_tokens=self._calculate_depth_tokens,
            fact_counts=self._store.get_fact_count_by_tags(),
            max_tool_rounds=self.config.paging.max_tool_loops,
        )

    def _build_supervised_hint(self, tag_summaries: list) -> str:
        return build_supervised_hint(
            tag_summaries=tag_summaries,
            working_set=self._working_set,
            max_hint_tokens=self.config.assembler.context_hint_max_tokens,
            token_counter=self._token_counter,
            max_tool_rounds=self.config.paging.max_tool_loops,
        )

    def _build_default_hint(self, tag_summaries: list) -> str:
        return build_default_hint(
            tag_summaries=tag_summaries,
            max_hint_tokens=self.config.assembler.context_hint_max_tokens,
            token_counter=self._token_counter,
        )

    def _resolve_paging_mode(self, model_name: str = "") -> str:
        """Check if *model_name* matches any ``autonomous_models`` entry.

        Returns ``"autonomous"`` if the model is trusted to page itself
        (tools + budget dashboard injected), ``"supervised"`` otherwise
        (VC manages paging silently via auto_promote / auto_evict).
        """
        model = model_name.lower()
        for pattern in self.config.paging.autonomous_models:
            if pattern.lower() in model:
                return "autonomous"
        return "supervised"

    def _get_latest_turn_pair(self, history: list[Message]) -> list[Message] | None:
        """Extract the most recent user+assistant pair."""
        if len(history) < 2:
            return None
        for i in range(len(history) - 1, 0, -1):
            if history[i].role == "assistant" and history[i-1].role == "user":
                return [history[i-1], history[i]]
        return None

    def _get_embed_fn(self):
        """Lazy-load the embedding function for the context bleed gate."""
        return self._semantic.get_embed_fn()

    def _embed_and_store_chunks(self, stored: "StoredSegment") -> None:
        """Chunk a segment's full_text, embed, and store vectors."""
        self._semantic.embed_and_store_chunks(stored)

    def _semantic_search(self, query: str, max_results: int = 5) -> list[QuoteResult]:
        """Embedding-based semantic search over stored chunk vectors."""
        return self._semantic.semantic_search(query, max_results)

    def _backfill_chunk_embeddings(self) -> list[ChunkEmbedding]:
        """One-time backfill: embed all existing segments' full_text."""
        return self._semantic.backfill_chunk_embeddings()

    def _context_is_relevant(
        self, current_text: str, context_pairs: list[str],
    ) -> bool:
        """Check if current turn is semantically similar to the most recent context pair."""
        return self._semantic.context_is_relevant(current_text, context_pairs)

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

        # Segment and compact in batches (results stored to DB incrementally)
        turn_offset = self._compacted_through // 2
        logger.info(
            "compact_manual: about to segment %d messages (watermark=%d, turn_offset=%d)",
            len(compact_messages), self._compacted_through, turn_offset,
        )
        segments = self._segmenter.segment(compact_messages, turn_offset=turn_offset)
        logger.info(
            "compact_manual: segmented into %d segments",
            len(segments),
        )
        results = self._compact_and_store(segments, len(compact_messages))

        # Advance watermark past compacted messages
        self._compacted_through += len(compact_messages)

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
        running_session_date = ""

        for i in range(0, len(history_pairs) - 1, 2):
            user_msg = history_pairs[i]
            asst_msg = history_pairs[i + 1]

            # BUG-013: Skip empty turns (tool_use/tool_result with no text)
            if not user_msg.content.strip() and not asst_msg.content.strip():
                logger.debug("Skipping empty turn at pair index %d", i // 2)
                continue

            # Track running session date from [Session from ...] headers
            m = _SESSION_HEADER_RE.search(user_msg.content)
            if m:
                running_session_date = m.group(1)
            elif not running_session_date and user_msg.timestamp:
                running_session_date = user_msg.timestamp.strftime("%Y-%m-%dT%H:%M:%S")

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
                session_date=running_session_date,
            )
            self._turn_tag_index.append(entry)
            ingested += 1

            if progress_callback:
                total = len(history_pairs) // 2
                progress_callback(ingested, total, entry)

            # Refresh store tags every 10 turns so new tags influence later tagging
            if ingested % 10 == 0:
                store_tags = [ts.tag for ts in self._store.get_all_tags()]

            # Periodic state save so session_date + tags are queryable during ingestion
            if ingested % 20 == 0:
                self._save_state(history_pairs)

        # Final save after all turns ingested
        self._save_state(history_pairs)
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

    def reassemble_context(self) -> str:
        """Re-assemble context with the current working set.

        Call after ``expand_topic()`` / ``collapse_topic()`` to get an
        updated ``prepend_text`` that reflects the new depth levels.
        Reuses the retrieval result from the most recent
        ``on_message_inbound()`` call — no re-tagging or re-retrieval.

        Returns the updated prepend_text, or "" if no prior inbound call.
        """
        rr = getattr(self, "_last_retrieval_result", None)
        history = getattr(self, "_last_conversation_history", None)
        if rr is None:
            return ""

        model_name = getattr(self, "_last_model_name", "")
        _pm = self._resolve_paging_mode(model_name) if self.config.paging.enabled else None
        context_hint = self._build_context_hint(paging_mode=_pm)
        core_context = self._assembler.load_core_context()

        ws_param = None
        full_segments_param = None
        if self.config.paging.enabled and self._working_set:
            ws_param = self._working_set
            full_segments_param = {}
            for tag, entry in self._working_set.items():
                if entry.depth in (DepthLevel.SEGMENTS, DepthLevel.FULL):
                    segs = self._store.get_segments_by_tags(
                        tags=[tag], min_overlap=1, limit=50,
                    )
                    if segs:
                        full_segments_param[tag] = segs

        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=rr,
            conversation_history=history or [],
            token_budget=self.config.context_window,
            context_hint=context_hint,
            working_set=ws_param,
            full_segments=full_segments_param,
        )
        return assembled.prepend_text

    # ------------------------------------------------------------------
    # Paging API: expand / collapse / working set
    # ------------------------------------------------------------------

    def expand_topic(self, tag: str, depth: str = "full") -> dict:
        """Expand a topic to deeper detail in the working set."""
        return self._paging.expand_topic(tag, depth)

    def collapse_topic(self, tag: str, depth: str = "summary") -> dict:
        """Collapse a topic to shallower detail. Returns freed tokens."""
        return self._paging.collapse_topic(tag, depth)

    def recall_all(self) -> dict:
        """Load all tag summaries. Used by vc_recall_all tool."""
        tag_summaries = self._store.get_all_tag_summaries()
        if not tag_summaries:
            return {"found": False, "message": "No stored summaries yet."}
        budget = self.config.assembler.tag_context_max_tokens
        selected = []
        total_tokens = 0
        for ts in tag_summaries:
            if total_tokens + ts.summary_tokens > budget:
                break
            selected.append({
                "tag": ts.tag,
                "summary": ts.summary,
                "tokens": ts.summary_tokens,
                "description": ts.description or "",
            })
            total_tokens += ts.summary_tokens
        return {
            "found": True,
            "topics_loaded": len(selected),
            "total_tokens": total_tokens,
            "summaries": selected,
        }

    def get_working_set_summary(self) -> dict:
        """Return current working set with budget info."""
        return self._paging.get_working_set_summary()

    def _calculate_depth_tokens(self, tag: str, depth: DepthLevel) -> int:
        """Calculate token cost for a tag at a given depth level."""
        return self._paging.calculate_depth_tokens(tag, depth)

    def _auto_evict(self, needed: int, exclude_tag: str = "") -> tuple[list[str], int]:
        """Auto-evict coldest topics to free `needed` tokens."""
        return self._paging._auto_evict(needed, exclude_tag)

    # ------------------------------------------------------------------
    # Description-aware search supplement for find_quote
    # ------------------------------------------------------------------

    def _supplement_from_descriptions(
        self,
        query: str,
        results: list[QuoteResult],
        max_results: int,
    ) -> list[QuoteResult]:
        """Add results from tags whose descriptions match query terms."""
        return _supplement_from_descriptions(self._store, query, results, max_results)

    def find_quote(
        self,
        query: str,
        max_results: int = 5,
        intent_context: str = "",
        session_filter: str = "",
    ) -> dict:
        """Search stored conversation text for a specific phrase or keyword."""
        return _find_quote(
            self._store,
            self._semantic,
            query,
            max_results,
            intent_context=intent_context,
            session_filter=session_filter,
        )

    def _parse_session_date(self, raw: str) -> date | None:
        """Best-effort parse for session date strings from stored metadata."""
        s = (raw or "").strip()
        if not s:
            return None
        # Fast-path ISO date/month prefixes.
        m = re.search(r"\d{4}-\d{2}-\d{2}", s)
        if m:
            try:
                return date.fromisoformat(m.group(0))
            except ValueError:
                pass
        m = re.search(r"\d{4}/\d{2}/\d{2}", s)
        if m:
            try:
                y, mo, d = [int(x) for x in m.group(0).split("/")]
                return date(y, mo, d)
            except ValueError:
                pass
        return None

    def _resolve_remember_when_range(self, time_range: dict) -> tuple[date, date, str]:
        """Resolve tool time_range input to absolute [start_date, end_date]."""
        if not isinstance(time_range, dict):
            raise ValueError("time_range must be an object")

        kind = str(time_range.get("kind", "")).strip().lower()
        today = datetime.now(timezone.utc).date()

        if kind == "relative":
            preset = str(time_range.get("preset", "")).strip().lower()
            if preset == "last_24_hours":
                return today - timedelta(days=1), today, preset
            if preset == "last_7_days":
                return today - timedelta(days=6), today, preset
            if preset == "last_30_days":
                return today - timedelta(days=29), today, preset
            if preset == "this_week":
                start = today - timedelta(days=today.weekday())
                return start, start + timedelta(days=6), preset
            if preset == "last_week":
                this_week_start = today - timedelta(days=today.weekday())
                start = this_week_start - timedelta(days=7)
                return start, start + timedelta(days=6), preset
            if preset == "this_month":
                start = date(today.year, today.month, 1)
                end = date(today.year, today.month, monthrange(today.year, today.month)[1])
                return start, end, preset
            if preset == "last_month":
                year = today.year
                month = today.month - 1
                if month == 0:
                    month = 12
                    year -= 1
                start = date(year, month, 1)
                end = date(year, month, monthrange(year, month)[1])
                return start, end, preset
            if preset == "this_year":
                return date(today.year, 1, 1), date(today.year, 12, 31), preset
            if preset == "last_year":
                y = today.year - 1
                return date(y, 1, 1), date(y, 12, 31), preset
            raise ValueError(f"unsupported relative preset: {preset}")

        if kind == "between_dates":
            start_raw = str(time_range.get("start", "")).strip()
            end_raw = str(time_range.get("end", "")).strip()
            if not start_raw or not end_raw:
                raise ValueError("between_dates requires start and end")

            def parse_boundary(raw: str, is_end: bool) -> date:
                if _ISO_DATE_RE.match(raw):
                    return date.fromisoformat(raw)
                if _ISO_MONTH_RE.match(raw):
                    y, mo = [int(x) for x in raw.split("-")]
                    if is_end:
                        return date(y, mo, monthrange(y, mo)[1])
                    return date(y, mo, 1)
                raise ValueError(f"invalid date format: {raw}")

            start = parse_boundary(start_raw, is_end=False)
            end = parse_boundary(end_raw, is_end=True)
            if end < start:
                raise ValueError("time_range end must be >= start")
            return start, end, "between_dates"

        raise ValueError("time_range.kind must be 'relative' or 'between_dates'")

    def query_facts(self, **kwargs) -> list | dict:
        """Query structured facts by filters. Expands verb semantically, then delegates to store.

        Returns list[Fact] normally.  When called with ``_return_meta=True``
        (used by the tool loop), returns a dict with ``facts``, ``expanded_verbs``,
        ``object_relaxed``, and ``semantic_note`` keys so the caller can
        annotate the response.
        """
        return_meta = kwargs.pop("_return_meta", False)
        intent_context = kwargs.pop("_intent_context", "") or ""
        expanded_verbs: list[str] | None = None
        semantic_note: str | None = None

        # Save original params before verb expansion mutates kwargs
        orig_verb = kwargs.get("verb")
        orig_subject = kwargs.get("subject")
        orig_object = kwargs.get("object_contains")
        status_filter = kwargs.get("status")

        verb = kwargs.get("verb")
        if verb and not kwargs.get("verbs"):
            expanded = self._expand_verb(verb)
            if expanded and len(expanded) > 1:
                expanded_verbs = expanded
                kwargs["verbs"] = expanded
                kwargs.pop("verb", None)

        results = self._store.query_facts(**kwargs)

        # Semantic search: find additional facts via embedding on 'what'
        # field.  Runs before auto-relax so it can provide precise results
        # and prevent the noisy fallback that drops object_contains.
        sem_all = self._semantic_fact_search(
            existing=results,
            subject=orig_subject,
            verb=orig_verb,
            object_contains=orig_object,
            intent_context=intent_context,
        )
        if sem_all:
            sem_filtered = sem_all
            if status_filter:
                sem_filtered = [f for f in sem_filtered if f.status == status_filter]
            # Respect the reader's explicit object_contains filter —
            # semantic search ignores structured constraints when building
            # candidates, so post-filter here to avoid returning facts
            # that contradict the reader's request (BUG-032).
            if orig_object:
                obj_lower = orig_object.lower()
                sem_filtered = [
                    f for f in sem_filtered
                    if obj_lower in (f.object or "").lower()
                    or obj_lower in (f.what or "").lower()
                ]
            if sem_filtered:
                results = results + sem_filtered
                semantic_note = f"semantic search added {len(sem_filtered)} fact(s)"

        # Auto-relax removed (BUG-032): dropping object_contains returns facts
        # that contradict the reader's explicit constraint, causing over-counting.
        # If 0 results, the reader should broaden its own query intentionally.

        if return_meta:
            meta: dict = {
                "facts": results,
                "expanded_verbs": expanded_verbs,
                "object_relaxed": False,  # auto-relax removed (BUG-032)
                "semantic_note": semantic_note,
            }
            # When status was filtered, also query without status so the
            # caller can show total_all_statuses — prevents the reader from
            # splitting into per-status calls and never seeing the grand total.
            if status_filter:
                # Strip both status and object_contains — we want the
                # grand total for this verb+subject across all statuses.
                # object_contains uses strict LIKE at the store layer and
                # would return 0 when the auto-relax only happened above.
                unfiltered = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("status", "object_contains")
                }
                all_facts = self._store.query_facts(**unfiltered)
                # Merge semantic matches into the unfiltered total
                if sem_all:
                    all_ids = {f.id for f in all_facts}
                    for f in sem_all:
                        if f.id not in all_ids:
                            all_facts.append(f)
                            all_ids.add(f.id)
                status_counts: dict[str, int] = {}
                for f in all_facts:
                    s = f.status or "unknown"
                    status_counts[s] = status_counts.get(s, 0) + 1
                meta["all_statuses"] = status_counts
                meta["total_all_statuses"] = len(all_facts)
            return meta
        return results

    def _expand_verb(self, verb: str) -> list[str] | None:
        """Find semantically similar verbs in the facts DB via embedding similarity.

        Returns list of matching verbs (including original) if expansions found,
        None if embeddings unavailable or no expansions.
        """
        embed_fn = self._get_embed_fn()
        if embed_fn is None:
            return None
        all_verbs = self._store.get_unique_fact_verbs()
        if not all_verbs:
            return None
        from .core.math_utils import cosine_similarity

        texts = [verb] + all_verbs
        vectors = embed_fn(texts)
        query_vec = vectors[0]
        threshold = 0.53
        matches = [verb]
        for i, v in enumerate(all_verbs):
            if v != verb:
                sim = cosine_similarity(query_vec, vectors[i + 1])
                if sim >= threshold:
                    matches.append(v)
        return matches if len(matches) > 1 else None

    def _semantic_fact_search(
        self,
        existing: list,
        subject: str | None = None,
        verb: str | None = None,
        object_contains: str | None = None,
        intent_context: str = "",
    ) -> list:
        """Find additional facts by embedding similarity on the ``what`` field.

        Builds a natural-language query from the provided structured params
        (and optionally the user's question via *intent_context*), retrieves
        all facts for the given subject, and returns those whose ``what``
        field is semantically close to the query but that were NOT already
        in *existing*.  Returns an empty list when embeddings are unavailable
        or no new matches are found.
        """
        # Need at least a verb or object to form a meaningful query
        if not verb and not object_contains:
            return []
        embed_fn = self._get_embed_fn()
        if embed_fn is None:
            return []

        # Build query string from whatever params were provided.
        # When intent_context (the user's question) is available, use it
        # as the primary query — it carries richer semantic signal than the
        # short structured params (e.g. "How many health-related devices
        # do I use?" vs just "user use").
        if intent_context.strip():
            query_str = intent_context.strip()
            # The intent_context may be the full enriched prompt (context
            # summaries + question).  Strip the <virtual-context> block so
            # only the trailing question/instruction remains.
            if "</virtual-context>" in query_str:
                query_str = query_str.split("</virtual-context>")[-1].strip()
            # Strip leading preamble that the benchmark wraps around the
            # context block — only keep meaningful trailing text.
            import re
            m = re.search(r"(?:^|\n)\s*Question:\s*(.+)", query_str, re.IGNORECASE)
            if m:
                query_str = m.group(1).strip()
                # Remove trailing "Answer:" marker if present
                query_str = re.sub(r"\s*Answer:\s*$", "", query_str).strip()
        else:
            parts: list[str] = []
            if subject:
                parts.append(subject)
            if verb:
                parts.append(verb)
            if object_contains:
                parts.append(object_contains)
            query_str = " ".join(parts)

        # Broad candidate set: all facts for this subject (no verb/object filter)
        cand_kwargs: dict = {"limit": 200}
        if subject:
            cand_kwargs["subject"] = subject
        candidates = self._store.query_facts(**cand_kwargs)

        # Exclude already-found facts and facts without a 'what' description
        existing_ids = {f.id for f in existing}
        candidates = [f for f in candidates if f.id not in existing_ids and f.what]
        if not candidates:
            return []

        from .core.math_utils import cosine_similarity

        texts = [query_str] + [f.what for f in candidates]
        vectors = embed_fn(texts)
        query_vec = vectors[0]

        threshold = 0.35
        matches = []
        for i, fact in enumerate(candidates):
            sim = cosine_similarity(query_vec, vectors[i + 1])
            if sim >= threshold:
                matches.append(fact)

        return matches

    def remember_when(
        self,
        query: str,
        time_range: dict,
        max_results: int = 5,
    ) -> dict:
        """Find memory snippets for *query* constrained to a resolved date window."""
        if not query.strip():
            return {"error": "empty query"}

        try:
            start, end, resolved_kind = self._resolve_remember_when_range(time_range)
        except ValueError as exc:
            return {"error": str(exc)}

        # Overfetch, then filter by session_date bounds.
        raw = self.find_quote(query=query, max_results=max(max_results * 4, 20))
        if not raw.get("found"):
            return {
                "query": query,
                "found": False,
                "range": {
                    "kind": resolved_kind,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                "results": [],
                "message": raw.get("message", "No matches found."),
            }

        filtered: list[dict] = []
        for item in raw.get("results", []):
            session = str(item.get("session", "")).strip()
            parsed = self._parse_session_date(session)
            if parsed is None:
                # No parseable session date -> exclude from time-filtered recall.
                continue
            if start <= parsed <= end:
                filtered.append(item)
            if len(filtered) >= max_results:
                break

        return {
            "query": query,
            "found": bool(filtered),
            "range": {
                "kind": resolved_kind,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "results": filtered,
            "message": (
                f"No matches for '{query}' in the requested time window."
                if not filtered else ""
            ),
        }

    # ------------------------------------------------------------------
    # query_with_tools: sync tool loop for non-proxy callers
    # ------------------------------------------------------------------

    def query_with_tools(
        self,
        messages: list[dict],
        *,
        model: str = "claude-sonnet-4-5-20250929",
        system: str = "",
        max_tokens: int = 4096,
        api_key: str = "",
        api_url: str = "",
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        force_tools: bool = False,
        require_tools: bool | None = None,
        max_loops: int | None = None,
        provider: str = "anthropic",
    ) -> "ToolLoopResult":
        """Send a query to an LLM with VC tool support.

        Builds a provider-specific request, optionally injects VC paging
        tools, sends a non-streaming POST, and runs a synchronous tool
        loop if the model invokes any VC tools.

        Supports Anthropic, OpenAI, OpenAI Codex, and Gemini providers via the adapter
        pattern.

        Parameters
        ----------
        messages : list[dict]
            Messages in ``[{"role": "user", "content": "..."}]`` format.
        model : str
            Model ID (e.g. ``"claude-sonnet-4-5-20250929"``, ``"gpt-4o"``).
        system : str
            System prompt.
        max_tokens : int
            Maximum tokens for the response.
        api_key : str
            API key for the provider.
        api_url : str
            Override for the API endpoint URL (default per provider).
        temperature : float
            Sampling temperature.
        tools : list[dict] | None
            Additional (non-VC) tool definitions to include (Anthropic format).
        force_tools : bool
            If True, inject VC tools even when the normal gate (paging
            enabled + compaction occurred) is not met.
        require_tools : bool | None
            If set, overrides provider tool policy: ``True`` requires at
            least one tool call, ``False`` leaves tool use optional.
        max_loops : int
            Maximum continuation rounds for the tool loop.
        provider : str
            LLM provider: ``"anthropic"``, ``"openai"``,
            ``"openai-codex"``, or ``"gemini"``.

        Returns
        -------
        ToolLoopResult
            Final text, tool call records, and usage metrics.
        """
        import httpx

        from .core.tool_loop import (
            _parse_provider_http_response,
            get_adapter,
            run_tool_loop,
            vc_tool_definitions,
        )

        adapter = get_adapter(provider, api_key, api_url)

        # Decide whether to inject VC tools
        inject_vc = force_tools or (
            self.config.paging.enabled and self._compacted_through > 0
        )
        all_tools: list[dict] = []
        if inject_vc:
            all_tools.extend(vc_tool_definitions())
        if tools:
            all_tools.extend(tools)

        # Convert tool definitions to provider format
        converted_tools = adapter.convert_tool_defs(all_tools) if all_tools else None

        # Build provider-specific request body
        body = adapter.build_request_body(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=converted_tools,
        )

        # Optional tool-policy override for thresholded behavior.
        if converted_tools and require_tools is not None:
            if provider == "anthropic":
                if require_tools:
                    body["tool_choice"] = {"type": "any"}
                else:
                    body.pop("tool_choice", None)
            elif provider in {"openai", "openai-codex", "openai_codex"}:
                if require_tools:
                    body["tool_choice"] = "required"
                else:
                    body.pop("tool_choice", None)

        url = adapter.get_url(model)
        headers = adapter.get_headers()

        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, headers=headers, json=body)

        if resp.status_code >= 300:
            raise RuntimeError(
                f"{provider} API error {resp.status_code}: {resp.text[:500]}"
            )

        data = _parse_provider_http_response(resp)

        # Check for VC tool calls
        tool_calls = adapter.extract_tool_calls(data)
        has_vc_tools = any(
            tc["name"].startswith("vc_") for tc in tool_calls
        )

        if has_vc_tools:
            effective_loops = (
                max_loops
                if max_loops is not None
                else self.config.paging.max_tool_loops
            )
            loop_result = run_tool_loop(
                self, data, body, adapter,
                url=url, max_loops=effective_loops,
            )
            # Prepend the initial request to raw_requests
            loop_result.raw_requests.insert(0, body)
            return loop_result

        # No tool calls — return text directly
        result = ToolLoopResult()
        result.raw_requests.append(body)
        result.raw_responses.append(data)
        input_toks, output_toks = adapter.extract_usage(data)
        result.input_tokens = input_toks
        result.output_tokens = output_toks
        result.stop_reason = adapter.get_stop_reason(data)
        result.text = adapter.extract_text(data)
        return result

    def get_cost_report(self) -> SessionCostSummary:
        """Return current session cost summary."""
        return self._cost_tracker.get_summary()

    def cleanup(self, max_age_days: int | None = None, max_total_tokens: int | None = None) -> int:
        """Run cleanup on the store. Returns count of segments deleted."""
        from datetime import timedelta
        max_age = timedelta(days=max_age_days) if max_age_days else None
        return self._store.cleanup(max_age=max_age, max_total_tokens=max_total_tokens)
