"""VirtualContextEngine: main orchestrator wiring all components together."""

from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from collections.abc import Callable
from pathlib import Path

from .config import load_config
from .core.assembler import ContextAssembler
from .core.compactor import DomainCompactor
from .core.model_catalog import ModelCatalog
from .core.telemetry import TelemetryLedger
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
    CompactionSignal,
    DEFAULT_CHAT_MODEL,
    EngineState,
    EngineStateSnapshot,
    Message,
    RetrievalResult,
    ToolLoopResult,
    TurnTagEntry,
    VirtualContextConfig,
)

logger = logging.getLogger(__name__)

_SESSION_HEADER_RE = re.compile(r'\[Session from ([^\]]+)\]')


# Patterns for stub content detection (media attachments, image placeholders, etc.)
_STUB_PATTERNS = [
    re.compile(r"\[image data removed[^\]]*\]"),
    re.compile(r"\[media attached:[^\]]*\]"),
    re.compile(r"\[file attached[^\]]*\]"),
    re.compile(r"\[attachment:[^\]]*\]"),
    re.compile(r"To send an image back to the user.*", re.IGNORECASE),
]


def _is_stub_content(text: str) -> bool:
    """Check if text is a stub (attachments, image placeholders, boilerplate) with no real user content.

    Returns True only when the text contains at least one stub pattern AND
    after stripping all stub patterns, fewer than 3 words of actual content remain.
    Plain short messages ("im good", "ok") are NOT stubs — they're real conversation.
    """
    if not text or not text.strip():
        return True
    # Must contain at least one stub pattern to be considered a stub
    has_stub = any(pattern.search(text) for pattern in _STUB_PATTERNS)
    if not has_stub:
        return False
    residual = text
    for pattern in _STUB_PATTERNS:
        residual = pattern.sub("", residual)
    residual = residual.strip()
    return len(residual.split()) < 3


from .core.compaction_pipeline import CompactionPipeline
from .core.paging_manager import PagingManager
from .core.retrieval_assembler import RetrievalAssembler
from .core.semantic_search import SemanticSearchManager
from .core.tagging_pipeline import TaggingPipeline


class VirtualContextEngine:
    """Main orchestrator: two entry points for inbound messages and turn completion.

    Usage:
        engine = VirtualContextEngine(config_path="./virtual-context.yaml")

        # Before sending to LLM
        assembled = engine.on_message_inbound(message, history)

        # After LLM responds
        report = engine.on_turn_complete(history)

    Threading contract:
        - ``tag_turn`` and ``on_message_inbound`` must NOT run concurrently
          with each other.  They mutate shared state (TurnTagIndex, watermark)
          without internal locking.
        - ``compact_if_needed`` MAY run concurrently with inbound methods.
          Stale watermark reads are benign by design: worst case the assembler
          includes slightly extra context, never less.
        - Direct usage of the engine without the proxy's sequencing guarantees
          is NOT thread-safe.  Callers are responsible for serializing calls.
        - The proxy layer (``ProxyState``) enforces safe sequencing via
          ``wait_for_tag()`` -- each inbound request waits for the previous
          tag_turn to complete before proceeding.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: VirtualContextConfig | None = None,
        session_cache=None,  # RedisSessionCache or None
        embedding_provider=None,  # EmbeddingProvider or None
    ) -> None:
        self._config_path = str(config_path) if config_path else None
        self.config = config or load_config(config_path)
        self._token_counter = create_token_counter(self.config.token_counter)

        # Shared embedding provider — single model load across all components
        if embedding_provider is not None:
            self._embedding_provider = embedding_provider
        else:
            from .core.embedding_provider import EmbeddingProvider
            self._embedding_provider = EmbeddingProvider(
                model_name=self.config.retriever.embedding_model,
            )

        # Initialize components
        self._turn_tag_index = TurnTagIndex()
        self._init_store()
        self._init_telemetry()
        self._init_canonicalizer()
        self._init_tag_generator()
        self._init_monitor()
        self._init_segmenter()
        self._init_assembler()
        self._init_retriever()
        self._init_compactor()
        self._init_tag_splitter()
        self._engine_state = EngineState()  # mutable shared state for delegates
        self._session_cache = session_cache
        self._reference_date: date | None = None  # override "today" for remember_when relative presets
        self._request_captures_provider: Callable[[], list[dict]] | None = None  # set by ProxyState
        self._restored_request_captures: list[dict] = []  # loaded from persisted state, consumed by ProxyState
        self._restored_conversation_history: list = []  # (turn, user, asst) from store or dicts from Redis

        # Restore persisted state BEFORE creating delegates so they get the
        # final turn_tag_index / engine_state — no re-sync needed.
        # Try Redis cache first (fast) — fall back to store if miss
        _redis_loaded = False
        if self._session_cache and self._session_cache.is_available():
            try:
                cached = self._session_cache.load_snapshot(self.config.conversation_id)
                if cached and cached.get("conversation_id") == self.config.conversation_id:
                    self._apply_cached_state(cached)
                    _redis_loaded = True
                    logger.info(
                        "Session cache hit: %d messages, %d turns from Redis (version=%s)",
                        len(cached.get("history", [])),
                        len(cached.get("turn_tag_entries", [])),
                        cached.get("version", "?"),
                    )
            except Exception as e:
                logger.warning("Redis cache load failed: %s — falling back to store", e)

        # Cross-check compacted_through against the store — the Redis snapshot
        # may be stale if compaction advanced the watermark after the last
        # Redis write (e.g. post-ingestion compaction wrote to store but the
        # container restarted before the next Redis save).
        if _redis_loaded:
            try:
                db_state = self._store.load_engine_state(self.config.conversation_id)
                if (
                    db_state
                    and db_state.compacted_through > self._engine_state.compacted_through
                ):
                    logger.info(
                        "Watermark reconciliation: Redis=%d < store=%d — using store value",
                        self._engine_state.compacted_through,
                        db_state.compacted_through,
                    )
                    self._engine_state.compacted_through = db_state.compacted_through
            except Exception:
                pass  # store may not be initialized yet; non-critical

        if not _redis_loaded:
            self._load_persisted_state()

        # Create delegates with the (possibly restored) turn_tag_index
        self._semantic = SemanticSearchManager(
            store=self._store, config=self.config,
            embedding_provider=self._embedding_provider,
        )
        from .core.fact_query import FactQueryEngine
        self._facts = FactQueryEngine(store=self._store, semantic=self._semantic, config=self.config)
        from .core.search_engine import SearchEngine
        self._search = SearchEngine(
            store=self._store, semantic=self._semantic,
            turn_tag_index=self._turn_tag_index, config=self.config,
        )
        from .core.temporal_resolver import TemporalResolver
        self._temporal = TemporalResolver(
            store=self._store, search_engine=self._search, config=self.config,
        )
        from .core.tool_query import ToolQueryRunner
        self._tool_query = ToolQueryRunner(engine=self, config=self.config)
        self._paging = PagingManager(
            store=self._store,
            token_counter=self._token_counter,
            tag_context_max_tokens=self.config.assembler.tag_context_max_tokens,
            auto_evict=self.config.paging.auto_evict,
            paging_enabled=self.config.paging.enabled,
            conversation_id=self.config.conversation_id,
        )
        self._supersession_checker = None
        self._init_supersession_checker()
        self._fact_curator = None
        self._init_fact_curator()
        self._tagging = TaggingPipeline(
            tag_generator=self._tag_generator,
            turn_tag_index=self._turn_tag_index,
            store=self._store,
            semantic=self._semantic,
            engine_state=self._engine_state,
            config=self.config,
            tag_splitter=self._tag_splitter,
            canonicalizer=self._canonicalizer,
            telemetry=self._telemetry,
            monitor=self._monitor,
            compactor=self._compactor,
            save_state_callback=self._save_state,
        )
        self._compaction = CompactionPipeline(
            compactor=self._compactor,
            segmenter=self._segmenter,
            store=self._store,
            turn_tag_index=self._turn_tag_index,
            engine_state=self._engine_state,
            config=self.config,
            supersession_checker=self._supersession_checker,
            fact_curator=self._fact_curator,
            semantic=self._semantic,
            telemetry=self._telemetry,
            save_state_callback=self._save_state,
        )
        self._retrieval = RetrievalAssembler(
            retriever=self._retriever,
            assembler=self._assembler,
            monitor=self._monitor,
            paging=self._paging,
            store=self._store,
            turn_tag_index=self._turn_tag_index,
            engine_state=self._engine_state,
            fact_curator=self._fact_curator,
            config=self.config,
            token_counter=self._token_counter,
        )
        self._retrieval._set_semantic(self._semantic)
        self._apply_persisted_state_to_delegates()
        self._bootstrap_vocabulary()

    def close(self) -> None:
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
    def reference_date(self) -> date | None:
        return self._reference_date

    @reference_date.setter
    def reference_date(self, value: date | None) -> None:
        self._reference_date = value
        self._temporal.reference_date = value

    def _init_canonicalizer(self) -> None:
        self._canonicalizer = TagCanonicalizer(store=self._store)
        self._canonicalizer.load()

    def _init_tag_generator(self) -> None:
        llm_provider = None

        # Try to build an LLM provider for tagging
        if self.config.tag_generator.type == "llm":
            provider_name = self.config.tag_generator.provider
            provider_config = self.config.providers.get(provider_name, {})
            llm_provider = self._build_provider(provider_name, provider_config)
            if llm_provider:
                llm_provider._llm_log_path = getattr(self.config.proxy, "llm_calls_log", "") or ""

        self._tag_generator: TagGenerator = build_tag_generator(
            self.config.tag_generator, llm_provider,
            canonicalizer=self._canonicalizer, telemetry_ledger=self._telemetry,
            embed_fn_factory=lambda: self._embedding_provider.get_embed_fn() if self._embedding_provider else None,
        )

    def _init_store(self) -> None:
        from .core.composite_store import CompositeStore
        from .storage.noop_fact_link_store import NoopFactLinkStore

        if self.config.storage.backend == "sqlite":
            sqlite = SQLiteStore(db_path=self.config.storage.sqlite_path)
            fact_links = sqlite if self.config.facts.graph_links else NoopFactLinkStore()
            self._store = CompositeStore(
                segments=sqlite, facts=sqlite, fact_links=fact_links,
                state=sqlite, search=sqlite,
            )
        elif self.config.storage.backend == "postgres":
            from .storage.postgres import PostgresStore
            pg = PostgresStore(dsn=self.config.storage.postgres_dsn)
            fact_links = pg if self.config.facts.graph_links else NoopFactLinkStore()
            self._store = CompositeStore(
                segments=pg, facts=pg, fact_links=fact_links,
                state=pg, search=pg,
            )
        elif self.config.storage.backend == "neo4j":
            from .storage.neo4j import Neo4jFactStore
            neo = Neo4jFactStore(
                uri=self.config.storage.neo4j_uri,
                auth=(self.config.storage.neo4j_user, self.config.storage.neo4j_password),
            )
            # Fallback: use Postgres if configured, otherwise SQLite
            if self.config.storage.postgres_dsn:
                from .storage.postgres import PostgresStore
                fallback = PostgresStore(dsn=self.config.storage.postgres_dsn)
            else:
                fallback = SQLiteStore(db_path=self.config.storage.sqlite_path)
            self._store = CompositeStore(
                segments=fallback, facts=neo, fact_links=neo,
                state=fallback, search=fallback,
            )
        elif self.config.storage.backend == "falkordb":
            from .storage.falkordb import FalkorDBFactStore
            fdb = FalkorDBFactStore(
                host=self.config.storage.falkordb_host,
                port=self.config.storage.falkordb_port,
                password=self.config.storage.falkordb_password,
            )
            if self.config.storage.postgres_dsn:
                from .storage.postgres import PostgresStore
                fallback = PostgresStore(dsn=self.config.storage.postgres_dsn)
            else:
                fallback = SQLiteStore(db_path=self.config.storage.sqlite_path)
            self._store = CompositeStore(
                segments=fallback, facts=fdb, fact_links=fdb,
                state=fallback, search=fallback,
            )
        elif self.config.storage.backend == "filesystem":
            fs = FilesystemStore(root=self.config.storage.root)
            self._store = CompositeStore(
                segments=fs, facts=fs, fact_links=NoopFactLinkStore(),
                state=fs, search=fs,
            )
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage.backend}")

        # Propagate search config to the store for excerpt/snippet lengths
        self._store.search_config = self.config.search

    def _init_monitor(self) -> None:
        self._monitor = ContextMonitor(
            config=self.config.monitor,
            token_counter=self._token_counter,
        )

    def _init_segmenter(self) -> None:
        embed_fn = self._embedding_provider.get_embed_fn() if self._embedding_provider else None
        self._segmenter = TopicSegmenter(
            tag_generator=self._tag_generator,
            config=self.config.segmenter,
            token_counter=self._token_counter,
            turn_tag_index=self._turn_tag_index,
            embed_fn=embed_fn,
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
            conversation_id=self.config.conversation_id,
        )

    def _build_inbound_embedding_tagger(self) -> TagGenerator:
        """Build an EmbeddingTagGenerator for inbound vocabulary matching."""
        from .core.embedding_tag_generator import EmbeddingTagGenerator

        logger.info(
            "Using embedding-based inbound matching (model=%s, threshold=%.2f)",
            self.config.retriever.embedding_model,
            self.config.retriever.embedding_threshold,
        )
        embed_fn = self._embedding_provider.get_embed_fn() if self._embedding_provider else None
        return EmbeddingTagGenerator(
            config=self.config.tag_generator,
            model_name=self.config.retriever.embedding_model,
            similarity_threshold=self.config.retriever.embedding_threshold,
            embed_fn=embed_fn,
        )

    def _init_compactor(self) -> None:
        self._llm_provider = None
        self._compactor = None

        provider_name = self.config.summarization.provider
        provider_config = self.config.providers.get(provider_name, {})
        self._llm_provider = self._build_provider(provider_name, provider_config)
        if self._llm_provider:
            self._llm_provider._llm_log_path = getattr(self.config.proxy, "llm_calls_log", "") or ""

        if self._llm_provider:
            self._compactor = DomainCompactor(
                llm_provider=self._llm_provider,
                config=self.config.compactor,
                token_counter=self._token_counter,
                model_name=self.config.summarization.model,
                tag_rules=self.config.tag_rules,
                telemetry_ledger=self._telemetry,
            )

    def _init_tag_splitter(self) -> None:
        self._tag_splitter = None
        cfg = self.config.tag_generator.tag_splitting
        if cfg.enabled and self._llm_provider:
            from .core.tag_splitter import TagSplitter
            self._tag_splitter = TagSplitter(
                llm=self._llm_provider,
                config=cfg,
            )

    def _init_telemetry(self) -> None:
        models_file = self.config.telemetry.models_file
        if not os.path.isabs(models_file):
            if self._config_path:
                config_dir = os.path.dirname(os.path.abspath(self._config_path))
            else:
                config_dir = os.getcwd()
            models_path = os.path.join(config_dir, models_file)
        else:
            models_path = models_file
        # Fall back to bundled default if resolved path doesn't exist
        if not os.path.exists(models_path):
            self._model_catalog = ModelCatalog.default()
        else:
            self._model_catalog = ModelCatalog(models_path)
        self._telemetry = TelemetryLedger(self._model_catalog)

    def _load_persisted_state(self) -> None:
        """Restore TurnTagIndex and compaction watermark from store if available.

        Only sets state on ``self`` (turn_tag_index, engine_state, config).
        Delegate-specific wiring (``_search``, ``_paging``) is deferred to
        ``_apply_persisted_state_to_delegates`` which runs after delegate
        creation.
        """
        try:
            saved = self._store.load_engine_state(self.config.conversation_id)
        except Exception:
            logger.warning("Failed to load persisted state, starting fresh", exc_info=True)
            return
        if not saved:
            return
        self.config.conversation_id = saved.conversation_id
        # Reset compacted_through to 0 — the persisted value was an index into the
        # previous session's conversation_history which no longer exists. The proxy
        # advances this after ingestion completes to cover re-ingested messages.
        self._engine_state.compacted_through = 0
        # Populate in-place so all existing references (retriever, etc.) see the restored entries
        for entry in saved.turn_tag_entries:
            self._turn_tag_index.append(entry)
        self._engine_state.split_processed_tags = set(saved.split_processed_tags)
        self._engine_state.trailing_fingerprint = saved.trailing_fingerprint
        # Restore telemetry counters from persisted rollup
        if saved.telemetry_rollup:
            self._telemetry.restore_from_rollup(saved.telemetry_rollup)
        # Stash request captures for ProxyState to pick up after init
        self._restored_request_captures = saved.request_captures or []
        self._engine_state.provider = saved.provider or ""
        self._engine_state.tool_tag_counter = saved.tool_tag_counter or 0
        # Stash working set entries for _apply_persisted_state_to_delegates
        self._restored_working_set = saved.working_set or []
        # Load conversation history from turn messages for post-restart rebuild
        try:
            self._restored_conversation_history = self._store.load_recent_turn_messages(
                saved.conversation_id, limit=200,
            )
        except Exception:
            self._restored_conversation_history = []
        logger.info(
            "Restored engine state: conversation=%s, compacted_through=%d, turns=%d, "
            "split_processed=%d, working_set=%d, history_messages=%d",
            saved.conversation_id[:12], saved.compacted_through,
            len(saved.turn_tag_entries), len(saved.split_processed_tags),
            len(self._restored_working_set), len(self._restored_conversation_history),
        )

        # Validate watermark against actual stored segments.
        # If the store has no segments for this conversation (e.g., user deleted
        # the conversation or segments were purged), reset the watermark so
        # ingested history can be compacted fresh.
        if self._engine_state.compacted_through > 0:
            try:
                segs = self._store.get_segments_by_tags(
                    tags=["_general"], min_overlap=0, limit=1,
                    conversation_id=self.config.conversation_id,
                )
                # Also check if ANY tag has segments
                if not segs:
                    tags = self._store.get_all_tags(conversation_id=self.config.conversation_id)
                    if not tags:
                        logger.warning(
                            "Watermark=%d but store has no segments for conversation %s — resetting to 0",
                            self._engine_state.compacted_through, self.config.conversation_id[:12],
                        )
                        self._engine_state.compacted_through = 0
            except Exception:
                pass  # Don't crash on validation failure

    def _apply_cached_state(self, cached: dict) -> None:
        """Restore engine state from a Redis snapshot."""
        from .types import TurnTagEntry, FactSignal, WorkingSetEntry, DepthLevel
        from datetime import datetime, timezone

        self.config.conversation_id = cached["conversation_id"]

        # Engine state — compacted_through is 0 in snapshot (history is uncompacted suffix)
        es = cached.get("engine_state", {})
        self._engine_state.compacted_through = es.get("compacted_through", 0)
        self._engine_state.split_processed_tags = set(es.get("split_processed_tags", []))
        self._engine_state.trailing_fingerprint = es.get("trailing_fingerprint", "")
        self._engine_state.provider = es.get("provider", "")
        self._engine_state.tool_tag_counter = es.get("tool_tag_counter", 0)

        # TurnTagIndex — populate in-place
        for entry_dict in cached.get("turn_tag_entries", []):
            fs = []
            for sig in entry_dict.get("fact_signals", []) or []:
                fs.append(FactSignal(
                    subject=sig.get("subject", ""),
                    verb=sig.get("verb", ""),
                    object=sig.get("object", ""),
                    status=sig.get("status", ""),
                    fact_type=sig.get("fact_type", "personal"),
                    what=sig.get("what", ""),
                ))
            ts = entry_dict.get("timestamp")
            if isinstance(ts, str) and ts:
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)
            self._turn_tag_index.append(TurnTagEntry(
                turn_number=entry_dict["turn_number"],
                tags=entry_dict["tags"],
                primary_tag=entry_dict.get("primary_tag", ""),
                message_hash=entry_dict.get("message_hash", ""),
                sender=entry_dict.get("sender", ""),
                timestamp=ts,
                fact_signals=fs or None,
            ))

        # Working set — stash for _apply_persisted_state_to_delegates
        self._restored_working_set = []
        for ws in es.get("working_set", []):
            try:
                self._restored_working_set.append(WorkingSetEntry(
                    tag=ws["tag"],
                    depth=DepthLevel(ws["depth"]) if isinstance(ws.get("depth"), str) else DepthLevel.SUMMARY,
                    tokens=ws.get("tokens", 0),
                    last_accessed_turn=ws.get("last_accessed_turn", 0),
                ))
            except (KeyError, ValueError):
                pass

        # Telemetry
        if es.get("telemetry_rollup"):
            self._telemetry.restore_from_rollup(es["telemetry_rollup"])

        # Request captures
        self._restored_request_captures = es.get("request_captures", [])

        # History — stash as list of dicts for ProxyState to pick up
        # (ProxyState handles the dict->Message conversion)
        self._restored_conversation_history = cached.get("history", [])

    def _apply_persisted_state_to_delegates(self) -> None:
        """Wire restored state into delegates created after _load_persisted_state."""
        # Restore paging working set (old snapshots may have empty list)
        ws_entries = getattr(self, "_restored_working_set", [])
        self._paging.working_set = {ws.tag: ws for ws in ws_entries}

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
        for ts in self._store.get_all_tags(conversation_id=self.config.conversation_id):
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
        try:
            # Persist telemetry rollup (totals + by_model) without raw events
            telemetry_dict = self._telemetry.to_dict()
            telemetry_dict.pop("events", None)  # too large for state blob
            # Pull request captures from proxy metrics if available
            captures = []
            if self._request_captures_provider:
                try:
                    captures = self._request_captures_provider()
                except Exception:
                    pass
            self._store.save_engine_state(EngineStateSnapshot(
                conversation_id=self.config.conversation_id,
                compacted_through=self._engine_state.compacted_through,
                turn_tag_entries=list(self._turn_tag_index.entries),
                turn_count=len(conversation_history) // 2,
                split_processed_tags=sorted(self._engine_state.split_processed_tags),
                working_set=list(self._paging.working_set.values()),
                trailing_fingerprint=self._engine_state.trailing_fingerprint,
                telemetry_rollup=telemetry_dict,
                request_captures=captures,
                provider=self._engine_state.provider,
                tool_tag_counter=self._engine_state.tool_tag_counter,
            ))
            # Write-through to Redis cache
            if self._session_cache:
                try:
                    cache_snapshot = self._build_cache_snapshot(conversation_history)
                    self._session_cache.save_snapshot(
                        self.config.conversation_id, cache_snapshot,
                    )
                except Exception as _redis_err:
                    logger.debug("Redis snapshot write failed: %s", _redis_err)
        except Exception as e:
            logger.error("Failed to save engine state: %s", e)

    _cache_version = 0

    def _build_cache_snapshot(self, conversation_history: list) -> dict:
        """Build atomic snapshot dict for Redis cache."""
        self._cache_version += 1
        ct = self._engine_state.compacted_through

        # History: uncompacted suffix only
        suffix = conversation_history[ct:] if ct < len(conversation_history) else conversation_history
        cap = getattr(self.config, 'proxy', None)
        cap = getattr(cap, 'redis_history_cap', 600) if cap else 600
        if len(suffix) > cap:
            suffix = suffix[-cap:]

        history = []
        for msg in suffix:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "metadata": msg.metadata,
                "raw_content": msg.raw_content,
            })

        entries = []
        for e in self._turn_tag_index.entries:
            entries.append({
                "turn_number": e.turn_number,
                "tags": e.tags,
                "primary_tag": e.primary_tag,
                "message_hash": e.message_hash,
                "sender": e.sender,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "fact_signals": [
                    {"subject": fs.subject, "verb": fs.verb, "object": fs.object,
                     "status": fs.status, "fact_type": fs.fact_type, "what": fs.what}
                    for fs in (e.fact_signals or [])
                ] if e.fact_signals else [],
            })

        telemetry_dict = self._telemetry.to_dict()
        telemetry_dict.pop("events", None)
        captures = []
        if self._request_captures_provider:
            try:
                captures = self._request_captures_provider()
            except Exception:
                pass

        return {
            "version": self._cache_version,
            "conversation_id": self.config.conversation_id,
            "history": history,
            "turn_tag_entries": entries,
            "engine_state": {
                "compacted_through": 0,  # history IS the uncompacted suffix
                "split_processed_tags": sorted(self._engine_state.split_processed_tags),
                "working_set": [
                    {"tag": ws.tag, "depth": ws.depth.value if hasattr(ws.depth, 'value') else ws.depth,
                     "tokens": ws.tokens, "last_accessed_turn": ws.last_accessed_turn}
                    for ws in self._paging.working_set.values()
                ],
                "trailing_fingerprint": self._engine_state.trailing_fingerprint,
                "telemetry_rollup": telemetry_dict,
                "request_captures": captures,
                "provider": self._engine_state.provider,
                "tool_tag_counter": self._engine_state.tool_tag_counter,
            },
        }

    def _init_supersession_checker(self) -> None:
        sc = self.config.supersession
        if not sc.enabled:
            return
        provider_name = sc.provider or self.config.summarization.provider
        model = sc.model or self.config.summarization.model
        provider_config = self.config.providers.get(provider_name, {})
        llm = self._build_provider(provider_name, provider_config)
        if not llm:
            logger.warning("Supersession enabled but provider '%s' could not be built", provider_name)
            return
        if self.config.facts.graph_links:
            from .ingest.supersession import FactLinkChecker
            self._supersession_checker = FactLinkChecker(
                llm_provider=llm,
                model=model,
                store=self._store,
                config=sc,
                graph_links=True,
                telemetry_ledger=self._telemetry,
                embed_fn=self._semantic.get_embed_fn(),
            )
            logger.info("Fact link checker initialized (provider=%s, model=%s, graph_links=True)", provider_name, model)
        else:
            from .ingest.supersession import FactSupersessionChecker
            self._supersession_checker = FactSupersessionChecker(
                llm_provider=llm,
                model=model,
                store=self._store,
                config=sc,
                telemetry_ledger=self._telemetry,
                embed_fn=self._semantic.get_embed_fn(),
            )
            logger.info("Supersession checker initialized (provider=%s, model=%s)", provider_name, model)

    def _init_fact_curator(self) -> None:
        cc = self.config.curation
        if not cc.enabled:
            return
        provider_name = cc.provider or self.config.summarization.provider
        model = cc.model or self.config.summarization.model
        provider_config = self.config.providers.get(provider_name, {})
        llm = self._build_provider(provider_name, provider_config)
        if not llm:
            logger.warning("Curation enabled but provider '%s' could not be built", provider_name)
            return
        from .ingest.curator import FactCurator
        self._fact_curator = FactCurator(
            llm_provider=llm,
            model=model,
            config=cc,
            telemetry_ledger=self._telemetry,
        )
        logger.info("Fact curator initialized (provider=%s, model=%s)", provider_name, model)

    def _build_provider(self, provider_name: str, provider_config: dict):
        ptype = provider_config.get("type", provider_name)

        if ptype in ("generic_openai", "openrouter"):
            from .providers.generic_openai import GenericOpenAIProvider
            if ptype == "openrouter":
                default_url = "https://openrouter.ai/api/v1"
                default_key_env = "OPENROUTER_API_KEY"
            else:
                default_url = "http://127.0.0.1:11434/v1"
                default_key_env = ""
            api_key_env = provider_config.get("api_key_env", default_key_env)
            api_key = provider_config.get("api_key") or (
                os.environ.get(api_key_env, "") if api_key_env else "not-needed"
            )
            return GenericOpenAIProvider(
                base_url=provider_config.get("base_url", default_url),
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
            logger.warning(
                "Anthropic provider '%s' skipped: no API key (checked env var '%s')",
                provider_name, api_key_env,
            )

        if ptype == "ollama_native":
            from .providers.ollama_native import OllamaNativeProvider
            return OllamaNativeProvider(
                base_url=provider_config.get("base_url", "http://127.0.0.1:11434"),
                model=provider_config.get("model", self.config.summarization.model),
                temperature=self.config.summarization.temperature,
                num_predict=provider_config.get("num_predict", 500),
                force_json=provider_config.get("force_json", True),
            )

        logger.warning("Unknown provider type '%s' for provider '%s'", ptype, provider_name)
        return None

    def on_message_inbound(
        self,
        message: str,
        conversation_history: list[Message],
        model_name: str = "",
        max_context_tokens: int | None = None,
    ) -> AssembledContext:
        return self._retrieval.on_message_inbound(message, conversation_history, model_name, max_context_tokens)

    def tag_turn(
        self,
        conversation_history: list[Message],
        payload_tokens: int | None = None,
    ) -> CompactionSignal | None:
        return self._tagging.tag_turn(conversation_history, payload_tokens)

    def compact_if_needed(
        self,
        conversation_history: list[Message],
        signal: CompactionSignal,
        progress_callback: Callable[..., None] | None = None,
    ) -> CompactionReport | None:
        return self._compaction.compact_if_needed(conversation_history, signal, progress_callback)

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

    def filter_history(
        self,
        conversation_history: list[Message],
        current_tags: list[str],
        recent_turns: int | None = None,
    ) -> list[Message]:
        retrieval = getattr(self, "_retrieval", None)
        if retrieval is None:
            # Lightweight fallback for tests that bypass __init__ via __new__
            retrieval = RetrievalAssembler.__new__(RetrievalAssembler)
            retrieval.config = self.config
            retrieval._engine_state = self._engine_state
            retrieval._turn_tag_index = self._turn_tag_index
        return retrieval.filter_history(conversation_history, current_tags, recent_turns)

    def compact_manual(
        self,
        conversation_history: list[Message],
    ) -> CompactionReport | None:
        return self._compaction.compact_manual(conversation_history)

    def ingest_history(
        self,
        history_pairs: list[Message],
        progress_callback: Callable[..., None] | None = None,
        turn_offset: int = 0,
    ) -> int:
        return self._tagging.ingest_history(history_pairs, progress_callback, turn_offset)

    def retrieve(self, message: str, active_tags: list[str] | None = None) -> RetrievalResult:
        return self._retrieval.retrieve(message, active_tags)

    def transform(self, message: str, active_tags: list[str] | None = None, budget: int | None = None) -> str:
        return self._retrieval.transform(message, active_tags, budget)

    def reassemble_context(self) -> str:
        return self._retrieval.reassemble_context()

    # ------------------------------------------------------------------
    # Paging API: expand / collapse / working set
    # ------------------------------------------------------------------

    def expand_topic(self, tag: str, depth: str = "full") -> dict:
        return self._paging.expand_topic(tag, depth)

    def collapse_topic(self, tag: str, depth: str = "summary") -> dict:
        return self._paging.collapse_topic(tag, depth)

    def recall_all(self) -> dict:
        return self._retrieval.recall_all()

    def get_working_set_summary(self) -> dict:
        return self._paging.get_working_set_summary()

    def find_quote(
        self,
        query: str,
        max_results: int | None = None,
        intent_context: str = "",
        session_filter: str = "",
    ) -> dict:
        return self._search.find_quote(query, max_results, intent_context=intent_context, session_filter=session_filter)

    def get_turns_by_tag(
        self,
        tag: str,
        conversation_history: list[Message] | None = None,
    ) -> dict:
        return self._search.get_turns_by_tag(tag, conversation_history)

    def _parse_session_date(self, raw: str) -> date | None:
        return self._temporal._parse_session_date(raw)

    def _resolve_remember_when_range(self, time_range: dict) -> tuple[date, date, str]:
        return self._temporal._resolve_remember_when_range(time_range)

    def query_facts(self, **kwargs) -> list | dict:
        return self._facts.query(**kwargs)

    def remember_when(
        self,
        query: str,
        time_range: dict,
        max_results: int | None = None,
    ) -> dict:
        return self._temporal.remember_when(query, time_range, max_results)

    # ------------------------------------------------------------------
    # query_with_tools: sync tool loop for non-proxy callers
    # ------------------------------------------------------------------

    def query_with_tools(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_CHAT_MODEL,
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
        extended_thinking: bool = False,
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
        return self._tool_query.query_with_tools(
            messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            api_key=api_key,
            api_url=api_url,
            temperature=temperature,
            tools=tools,
            force_tools=force_tools,
            require_tools=require_tools,
            max_loops=max_loops,
            provider=provider,
            extended_thinking=extended_thinking,
        )

    def get_telemetry(self) -> TelemetryLedger:
        return self._telemetry

    def cleanup(self, max_age_days: int | None = None, max_total_tokens: int | None = None) -> int:
        max_age = timedelta(days=max_age_days) if max_age_days else None
        return self._store.cleanup(max_age=max_age, max_total_tokens=max_total_tokens)

    def get_latest_turn_tags(self) -> tuple[list[str], str]:
        """Return (tags, primary_tag) from the latest TurnTagIndex entry.

        Returns ``([], "_general")`` if the index is empty.
        """
        entries = self._turn_tag_index.entries
        if entries:
            latest = entries[-1]
            return list(latest.tags), latest.primary_tag
        return [], "_general"
