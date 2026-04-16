"""VirtualContextEngine: main orchestrator wiring all components together."""

from __future__ import annotations

import logging
import json
import hashlib
import os
import re
from datetime import date, datetime, timedelta, timezone
from collections.abc import Callable
from pathlib import Path

from .config import load_config
from .core.assembler import ContextAssembler
from .core.compactor import DomainCompactor
from .core.ingest_reconciler import IngestReconciler
from .core.model_catalog import ModelCatalog
from .core.telemetry import TelemetryLedger
from .core.monitor import ContextMonitor
from .core.retriever import ContextRetriever
from .core.segmenter import TopicSegmenter, pair_messages_into_turns
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
    SplitResult,
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


def _restored_flushed_prefix_messages(
    compacted_prefix_messages: int,
    flushed_prefix_messages: int | None,
    *,
    present: bool,
) -> int:
    """Restore the flush watermark while preserving a valid persisted zero."""
    if not present:
        return compacted_prefix_messages
    return int(flushed_prefix_messages or 0)


from .core.compaction_pipeline import CompactionPipeline
from .core.conversation_store import ConversationStoreView, StaleConversationWriteError
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
        session_state_provider=None,  # SessionStateProvider or None
    ) -> None:
        self._config_path = str(config_path) if config_path else None
        self.config = config or load_config(config_path)
        self._token_counter = create_token_counter(self.config.token_counter)
        self._session_cache = session_cache
        self._session_state_provider = session_state_provider

        # Shared embedding provider — single model load across all components
        if embedding_provider is not None:
            self._embedding_provider = embedding_provider
        else:
            from .core.embedding_provider import EmbeddingProvider
            self._embedding_provider = EmbeddingProvider(
                model_name=self.config.retriever.embedding_model,
            )

        self._conversation_generation = 0
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
        self._engine_state.conversation_generation = self._conversation_generation
        self._session_state_version: int = 0  # Tracks loaded Redis version for optimistic save
        self._reference_date: date | None = None  # override "today" for remember_when relative presets
        self._request_captures_provider: Callable[[], list[dict]] | None = None  # set by ProxyState
        self._restored_request_captures: list[dict] = []  # loaded from persisted state, consumed by ProxyState
        self._restored_conversation_history: list = []  # (turn, user, asst) from store or dicts from Redis
        self._restored_pending_turns: list[tuple[int, str, str, str | None, str | None]] = []
        self._restored_from_checkpoint = False
        self._restored_checkpoint_source = ""

        if self._session_state_provider is None:
            # Self-managed restore (local proxy mode)
            # Restore persisted state BEFORE creating delegates so they get the
            # final turn_tag_index / engine_state — no re-sync needed.
            # The durable store is authoritative; Redis may only supply a richer
            # history snapshot when it matches the committed checkpoint.
            _store_loaded = False
            _cached = None
            try:
                _saved = self._store.load_engine_state(self.config.conversation_id)
            except Exception:
                logger.warning("Failed to load persisted state, starting fresh", exc_info=True)
                _saved = None
            if _saved:
                self._load_persisted_state(saved=_saved)
                _store_loaded = True

            if self._session_cache and self._session_cache.is_available():
                try:
                    _cached = self._session_cache.load_snapshot(self.config.conversation_id)
                except Exception as e:
                    logger.warning("Redis cache load failed: %s — continuing without cache", e)
                    _cached = None

            if _cached and _cached.get("conversation_id") == self.config.conversation_id:
                cached_generation = int(_cached.get("conversation_generation", -1) or -1)
                cache_generation_matches = (
                    cached_generation == self._conversation_generation
                    if cached_generation >= 0
                    else self._conversation_generation == 0
                )
                if not cache_generation_matches:
                    logger.info(
                        "Ignoring Redis snapshot for conversation %s because its generation does not match the active lifecycle",
                        self.config.conversation_id[:12],
                    )
                elif not _store_loaded:
                    self._apply_cached_state(_cached)
                    logger.info(
                        "Session cache restore: %d messages, %d turns from Redis (version=%s)",
                        len(_cached.get("history", [])),
                        len(_cached.get("turn_tag_entries", [])),
                        _cached.get("version", "?"),
                    )
                elif self._cache_checkpoint_matches_store(_saved, _cached):
                    self._restored_conversation_history = _cached.get("history", [])
                    if self._restored_conversation_history:
                        self._restored_checkpoint_source = "store+redis"
                        logger.info(
                            "Session cache history accepted for committed checkpoint %s (messages=%d, version=%s)",
                            self.config.conversation_id[:12],
                            len(self._restored_conversation_history),
                            _cached.get("version", "?"),
                        )
                else:
                    logger.info(
                        "Ignoring Redis snapshot for conversation %s because it does not match the committed store checkpoint",
                        self.config.conversation_id[:12],
                    )
        else:
            # Provider mode: state will be injected by caller before each request.
            # Skip store/Redis restore entirely.
            self._session_cache = None
            self._restored_from_checkpoint = False
            self._restored_checkpoint_source = ""

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
            store=self._store,
            search_engine=self._search,
            config=self.config,
            semantic=self._semantic,
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
        _tool_tag_cb = None
        if self._session_state_provider:
            _conv_id = self.config.conversation_id
            _provider = self._session_state_provider
            _tool_tag_cb = lambda: _provider.next_tool_tag(_conv_id)
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
            next_tool_tag_callback=_tool_tag_cb,
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
            session_state_provider=self._session_state_provider,
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
            session_state_provider=self._session_state_provider,
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
        self._canonicalizer = TagCanonicalizer(
            store=self._store,
            conversation_id=self.config.conversation_id,
        )
        self._canonicalizer.load()

    def _init_tag_generator(self) -> None:
        llm_provider = None
        shared_embedding_loader = None
        shared_embedding_saver = None
        if self._session_state_provider is not None:
            shared_embedding_loader = self._session_state_provider.load_tag_embeddings
            shared_embedding_saver = self._session_state_provider.save_tag_embeddings

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
            embedding_model_name=self.config.retriever.embedding_model,
            load_cached_embeddings=shared_embedding_loader,
            save_cached_embeddings=shared_embedding_saver,
            code_mode=self.config.compactor.code_mode,
        )

    def _init_store(self) -> None:
        from .core.composite_store import CompositeStore
        from .storage.noop_fact_link_store import NoopFactLinkStore

        if self.config.storage.backend == "sqlite":
            sqlite = SQLiteStore(db_path=self.config.storage.sqlite_path)
            fact_links = sqlite if self.config.facts.graph_links else NoopFactLinkStore()
            store = CompositeStore(
                segments=sqlite, facts=sqlite, fact_links=fact_links,
                state=sqlite, search=sqlite,
            )
        elif self.config.storage.backend == "postgres":
            from .storage.postgres import PostgresStore
            pg = PostgresStore(dsn=self.config.storage.postgres_dsn)
            fact_links = pg if self.config.facts.graph_links else NoopFactLinkStore()
            store = CompositeStore(
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
            store = CompositeStore(
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
            store = CompositeStore(
                segments=fallback, facts=fdb, fact_links=fdb,
                state=fallback, search=fallback,
            )
        elif self.config.storage.backend == "filesystem":
            fs = FilesystemStore(root=self.config.storage.root)
            store = CompositeStore(
                segments=fs, facts=fs, fact_links=NoopFactLinkStore(),
                state=fs, search=fs,
            )
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage.backend}")

        self._conversation_generation = int(
            getattr(store, "activate_conversation", lambda _cid: 0)(self.config.conversation_id) or 0
        )
        self._store = ConversationStoreView(
            store,
            self.config.conversation_id,
            self._conversation_generation,
        )
        # Propagate search config to the store for excerpt/snippet lengths
        self._store.search_config = self.config.search

    def _init_monitor(self) -> None:
        self._monitor = ContextMonitor(
            config=self.config.monitor,
            token_counter=self._token_counter,
            conversation_id=self.config.conversation_id,
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
            store=self._store,
            conversation_id=self.config.conversation_id,
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
            session_state_provider=self._session_state_provider,
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
            load_cached_embeddings=(
                self._session_state_provider.load_tag_embeddings
                if self._session_state_provider is not None else None
            ),
            save_cached_embeddings=(
                self._session_state_provider.save_tag_embeddings
                if self._session_state_provider is not None else None
            ),
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

    @staticmethod
    def _parse_turn_timestamp(*values: str | None) -> datetime:
        for value in values:
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
        return datetime.now(timezone.utc)

    @staticmethod
    def _group_canonical_rows_into_pairs(rows: list[object]) -> list[tuple[int, list[object]]]:
        explicit_groups = [
            int(getattr(row, "turn_group_number"))
            if getattr(row, "turn_group_number", None) is not None
            else -1
            for row in rows
        ]
        if rows and all(group >= 0 for group in explicit_groups):
            grouped_by_turn: dict[int, list[object]] = {}
            for row, turn_group_number in zip(rows, explicit_groups, strict=False):
                grouped_by_turn.setdefault(turn_group_number, []).append(row)
            return sorted(grouped_by_turn.items(), key=lambda item: item[0])

        grouped: list[tuple[int, list[object]]] = []
        pending: list[object] = []

        def _flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            grouped.append((len(grouped), list(pending)))
            pending = []

        for row in rows:
            has_user = bool(getattr(row, "user_content", ""))
            has_assistant = bool(getattr(row, "assistant_content", ""))
            if has_user and has_assistant:
                _flush_pending()
                grouped.append((len(grouped), [row]))
                continue
            if has_user:
                _flush_pending()
                pending = [row]
                continue
            if has_assistant:
                if pending:
                    pending.append(row)
                    _flush_pending()
                else:
                    grouped.append((len(grouped), [row]))
                continue
            _flush_pending()

        _flush_pending()
        return grouped

    @staticmethod
    def _canonical_prefix_watermark(paired_rows: list[tuple[int, list[object]]]) -> tuple[int, int]:
        last_prefix_turn = -1
        for turn_number, rows in paired_rows:
            if rows and all(getattr(row, "compacted_at", None) for row in rows):
                last_prefix_turn = turn_number
                continue
            break
        if last_prefix_turn < 0:
            return 0, -1
        return ((last_prefix_turn + 1) * 2), last_prefix_turn

    @staticmethod
    def _pair_payload_from_rows(rows: list[object]) -> tuple[str, str, str | None, str | None]:
        user_content = ""
        assistant_content = ""
        user_raw_content = None
        assistant_raw_content = None
        for row in rows:
            if not user_content and getattr(row, "user_content", ""):
                user_content = str(getattr(row, "user_content", "") or "")
                user_raw_content = getattr(row, "user_raw_content", None)
            if not assistant_content and getattr(row, "assistant_content", ""):
                assistant_content = str(getattr(row, "assistant_content", "") or "")
                assistant_raw_content = getattr(row, "assistant_raw_content", None)
        return user_content, assistant_content, user_raw_content, assistant_raw_content

    def _restore_from_canonical_rows(self, conversation_id: str) -> bool:
        try:
            rows = list(self._store.get_all_canonical_turns(conversation_id))
        except Exception:
            logger.warning(
                "Failed to load canonical turns for conversation %s",
                conversation_id[:12],
                exc_info=True,
            )
            return False
        if not rows:
            return False

        paired_rows = self._group_canonical_rows_into_pairs(rows)
        tagged_pairs = [
            (turn_number, pair_rows)
            for turn_number, pair_rows in paired_rows
            if pair_rows and all(getattr(row, "tagged_at", None) for row in pair_rows)
        ]
        self._turn_tag_index = TurnTagIndex()
        for turn_number, pair_rows in tagged_pairs:
            user_content, assistant_content, *_ = self._pair_payload_from_rows(pair_rows)
            primary_tag = "_general"
            sender = ""
            session_date = ""
            canonical_turn_id = ""
            tags: list[str] = []
            fact_signals = []
            code_refs = []
            timestamp_values: list[str | None] = []
            for row in pair_rows:
                if not canonical_turn_id:
                    canonical_turn_id = getattr(row, "canonical_turn_id", "") or ""
                if getattr(row, "primary_tag", "") not in ("", "_general"):
                    primary_tag = row.primary_tag
                elif primary_tag == "_general":
                    primary_tag = getattr(row, "primary_tag", "_general") or "_general"
                if not sender:
                    sender = getattr(row, "sender", "") or ""
                if not session_date:
                    session_date = getattr(row, "session_date", "") or ""
                tags.extend(list(getattr(row, "tags", []) or []))
                fact_signals.extend(list(getattr(row, "fact_signals", []) or []))
                code_refs.extend(list(getattr(row, "code_refs", []) or []))
                timestamp_values.extend(
                    [
                        getattr(row, "last_seen_at", None),
                        getattr(row, "first_seen_at", None),
                        getattr(row, "updated_at", None),
                        getattr(row, "created_at", None),
                    ]
                )
            self._turn_tag_index.append(TurnTagEntry(
                turn_number=turn_number,
                canonical_turn_id=canonical_turn_id,
                message_hash=hashlib.sha256(f"{user_content} {assistant_content}".encode()).hexdigest()[:16],
                tags=list(dict.fromkeys(tags)) or [primary_tag],
                primary_tag=primary_tag,
                timestamp=self._parse_turn_timestamp(*timestamp_values),
                session_date=session_date,
                fact_signals=fact_signals,
                sender=sender,
                code_refs=code_refs,
            ))

        compacted_pairs = [
            (turn_number, pair_rows)
            for turn_number, pair_rows in paired_rows
            if pair_rows and all(getattr(row, "compacted_at", None) for row in pair_rows)
        ]
        live_pairs = [
            (turn_number, pair_rows)
            for turn_number, pair_rows in tagged_pairs
            if not all(getattr(row, "compacted_at", None) for row in pair_rows)
        ]
        pending_pairs = [
            (turn_number, pair_rows)
            for turn_number, pair_rows in paired_rows
            if not pair_rows or not all(getattr(row, "tagged_at", None) for row in pair_rows)
        ]
        compacted_prefix_messages, last_compacted_turn = self._canonical_prefix_watermark(paired_rows)
        self._engine_state.compacted_prefix_messages = compacted_prefix_messages
        self._engine_state.flushed_prefix_messages = self._engine_state.compacted_prefix_messages
        self._engine_state.last_compacted_turn = last_compacted_turn
        last_completed_turn = paired_rows[-1][0] if paired_rows else -1
        last_indexed_turn = tagged_pairs[-1][0] if tagged_pairs else -1
        self._engine_state.last_completed_turn = max(self._engine_state.last_completed_turn, last_completed_turn)
        self._engine_state.last_indexed_turn = max(self._engine_state.last_indexed_turn, last_indexed_turn)
        self._engine_state.checkpoint_version = max(self._engine_state.checkpoint_version, 4)
        self._restored_conversation_history = [
            (turn_number, *self._pair_payload_from_rows(pair_rows)[:2])
            for turn_number, pair_rows in live_pairs
        ]
        self._restored_pending_turns = [
            (
                turn_number,
                *self._pair_payload_from_rows(pair_rows),
            )
            for turn_number, pair_rows in pending_pairs
        ]
        logger.info(
            "Canonical restore loaded %d turns (%d indexed, %d compacted, %d live, %d pending) for conversation %s",
            len(paired_rows),
            len(tagged_pairs),
            len(compacted_pairs),
            len(live_pairs),
            len(pending_pairs),
            conversation_id[:12],
        )
        return True

    def _load_persisted_state(
        self,
        saved: EngineStateSnapshot | None = None,
    ) -> None:
        """Restore TurnTagIndex and compaction watermark from store if available.

        Only sets state on ``self`` (turn_tag_index, engine_state, config).
        Delegate-specific wiring (``_search``, ``_paging``) is deferred to
        ``_apply_persisted_state_to_delegates`` which runs after delegate
        creation.
        """
        if saved is None:
            try:
                saved = self._store.load_engine_state(self.config.conversation_id)
            except Exception:
                logger.warning("Failed to load persisted state, starting fresh", exc_info=True)
                return
        if not saved:
            return
        self.config.conversation_id = saved.conversation_id
        self._engine_state.compacted_prefix_messages = saved.compacted_prefix_messages
        self._engine_state.flushed_prefix_messages = _restored_flushed_prefix_messages(
            saved.compacted_prefix_messages,
            getattr(saved, 'flushed_prefix_messages', 0),
            present=getattr(saved, 'flushed_prefix_messages_present', True),
        )
        self._engine_state.last_request_time = getattr(saved, 'last_request_time', 0.0)
        self._engine_state.last_compacted_turn = saved.last_compacted_turn
        self._engine_state.last_completed_turn = saved.last_completed_turn
        self._engine_state.last_indexed_turn = saved.last_indexed_turn
        self._engine_state.checkpoint_version = saved.checkpoint_version
        self._engine_state.conversation_generation = max(
            saved.conversation_generation,
            self._conversation_generation,
        )
        # Populate in-place so all existing references (retriever, etc.) see the restored entries
        for entry in saved.turn_tag_entries:
            self._turn_tag_index.append(entry)
        self._update_checkpoint_markers()
        self._engine_state.split_processed_tags = set(saved.split_processed_tags)
        self._engine_state.trailing_fingerprint = saved.trailing_fingerprint
        # Restore telemetry counters from persisted rollup
        if saved.telemetry_rollup:
            self._telemetry.restore_from_rollup(saved.telemetry_rollup)
        # Stash request captures for ProxyState to pick up after init
        self._restored_request_captures = saved.request_captures or []
        self._engine_state.provider = saved.provider or ""
        self._engine_state.tool_tag_counter = saved.tool_tag_counter or 0
        self._restored_from_checkpoint = True
        self._restored_checkpoint_source = "store"
        # Stash working set entries for _apply_persisted_state_to_delegates
        self._restored_working_set = saved.working_set or []
        if not self._restore_from_canonical_rows(saved.conversation_id):
            self._restored_conversation_history = []
            self._restored_pending_turns = []
        logger.info(
            "Restored engine state: conversation=%s, compacted_prefix_messages=%d, indexed=%d, completed=%d, "
            "turns=%d, split_processed=%d, working_set=%d, history_messages=%d pending_turns=%d",
            saved.conversation_id[:12], saved.compacted_prefix_messages,
            self._engine_state.last_indexed_turn,
            self._engine_state.last_completed_turn,
            len(saved.turn_tag_entries), len(saved.split_processed_tags),
            len(self._restored_working_set), len(self._restored_conversation_history),
            len(self._restored_pending_turns),
        )

        # Validate watermark against actual stored segments.
        # If the store has no segments for this conversation (e.g., user deleted
        # the conversation or segments were purged), reset the watermark so
        # ingested history can be compacted fresh.
        if self._engine_state.compacted_prefix_messages > 0:
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
                            self._engine_state.compacted_prefix_messages, self.config.conversation_id[:12],
                        )
                        self._engine_state.compacted_prefix_messages = 0
                        self._engine_state.flushed_prefix_messages = 0
                        self._engine_state.last_request_time = 0.0
            except Exception:
                pass  # Don't crash on validation failure

    def _apply_cached_state(self, cached: dict) -> None:
        """Restore engine state from a Redis snapshot."""
        from .types import TurnTagEntry, FactSignal, WorkingSetEntry, DepthLevel
        from datetime import datetime, timezone

        self.config.conversation_id = cached["conversation_id"]

        # Engine state — compacted_prefix_messages is 0 in snapshot (history is uncompacted suffix)
        es = cached.get("engine_state", {})
        self._engine_state.compacted_prefix_messages = es.get("compacted_prefix_messages", 0)
        self._engine_state.flushed_prefix_messages = _restored_flushed_prefix_messages(
            es.get("compacted_prefix_messages", 0),
            es.get("flushed_prefix_messages", 0),
            present=("flushed_prefix_messages" in es),
        )
        self._engine_state.last_request_time = es.get("last_request_time", 0.0)
        self._engine_state.last_compacted_turn = es.get(
            "last_compacted_turn",
            (self._engine_state.compacted_prefix_messages // 2) - 1 if self._engine_state.compacted_prefix_messages > 0 else -1,
        )
        self._engine_state.last_completed_turn = es.get(
            "last_completed_turn",
            len(cached.get("turn_tag_entries", []) or []) - 1,
        )
        self._engine_state.last_indexed_turn = es.get(
            "last_indexed_turn",
            len(cached.get("turn_tag_entries", []) or []) - 1,
        )
        self._engine_state.checkpoint_version = es.get("checkpoint_version", cached.get("version", 0) or 0)
        self._engine_state.conversation_generation = es.get(
            "conversation_generation",
            cached.get("conversation_generation", self._conversation_generation),
        )
        self._engine_state.split_processed_tags = set(es.get("split_processed_tags", []))
        self._engine_state.trailing_fingerprint = es.get("trailing_fingerprint", "")
        self._engine_state.provider = es.get("provider", "")
        self._engine_state.tool_tag_counter = es.get("tool_tag_counter", 0)
        self._restored_from_checkpoint = True
        self._restored_checkpoint_source = "redis"

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
                canonical_turn_id=entry_dict.get("canonical_turn_id", "") or "",
                tags=entry_dict["tags"],
                primary_tag=entry_dict.get("primary_tag", ""),
                message_hash=entry_dict.get("message_hash", ""),
                sender=entry_dict.get("sender", ""),
                timestamp=ts,
                fact_signals=fs or None,
                code_refs=entry_dict.get("code_refs", []) or [],
            ))
        self._update_checkpoint_markers()

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

    # ---- SessionStateProvider integration ----

    def hydrate_from_session_state(self, state) -> None:
        """Inject checkpoint state from SessionStateProvider before processing.

        IMPORTANT: After replacing _turn_tag_index and _engine_state, this method
        must rebind delegate references so tagging, compaction, retrieval, and
        search operate on the new objects. Follows the same pattern as
        ProxyState._rebind_engine_references().
        """
        from .proxy.session_state import SessionState  # noqa: F811

        # Carry the Redis version so extract_session_state can pass it back
        self._session_state_version = state.version

        # Engine state markers (including tool_tag_counter for fallback continuity)
        self._engine_state.tool_tag_counter = state.tool_tag_counter
        self._engine_state.compacted_prefix_messages = state.compacted_prefix_messages
        self._engine_state.flushed_prefix_messages = _restored_flushed_prefix_messages(
            state.compacted_prefix_messages,
            getattr(state, 'flushed_prefix_messages', 0),
            present=getattr(state, 'flushed_prefix_messages_present', True),
        )
        self._engine_state.last_request_time = getattr(state, 'last_request_time', 0.0)
        self._engine_state.last_compacted_turn = state.last_compacted_turn
        self._engine_state.last_completed_turn = state.last_completed_turn
        self._engine_state.last_indexed_turn = state.last_indexed_turn
        self._engine_state.checkpoint_version = state.checkpoint_version
        self._engine_state.conversation_generation = state.conversation_generation
        self._engine_state.split_processed_tags = set(state.split_processed_tags)
        self._engine_state.trailing_fingerprint = state.trailing_fingerprint
        self._engine_state.provider = state.provider

        # Turn tag index — replace and rebuild
        self._turn_tag_index = TurnTagIndex()
        for entry_dict in state.turn_tag_entries:
            # Parse timestamp from ISO string, default to now if missing
            ts_raw = entry_dict.get("timestamp")
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            # Restore fact_signals if present
            fs_raw = entry_dict.get("fact_signals", [])
            fs = None
            if fs_raw:
                from .types import FactSignal
                fs = [
                    FactSignal(
                        subject=f.get("subject", ""),
                        verb=f.get("verb", ""),
                        object=f.get("object", ""),
                        status=f.get("status", ""),
                        fact_type=f.get("fact_type", ""),
                        what=f.get("what", ""),
                    )
                    for f in fs_raw if isinstance(f, dict)
                ]

            self._turn_tag_index.append(TurnTagEntry(
                turn_number=entry_dict.get("turn_number", 0),
                canonical_turn_id=entry_dict.get("canonical_turn_id", "") or "",
                tags=entry_dict.get("tags", []),
                primary_tag=entry_dict.get("primary_tag", "_general"),
                message_hash=entry_dict.get("message_hash", ""),
                sender=entry_dict.get("sender", ""),
                timestamp=ts,
                session_date=entry_dict.get("session_date", ""),
                fact_signals=fs,
                code_refs=entry_dict.get("code_refs", []) or [],
            ))

        # Working set — uses DepthLevel enum, NOT PagingDepth
        from .types import WorkingSetEntry, DepthLevel
        self._paging.working_set = {}
        for ws in state.working_set:
            tag = ws.get("tag", "")
            self._paging.working_set[tag] = WorkingSetEntry(
                tag=tag,
                depth=DepthLevel(ws.get("depth", "summary")),
                tokens=ws.get("tokens", 0),
                last_accessed_turn=ws.get("last_accessed_turn", 0),
            )

        # Restore telemetry rollup. On a REUSED engine, the ledger already
        # has state from the previous request. Reset it first to avoid
        # double-counting, then restore the persisted rollup.
        if hasattr(self, '_telemetry'):
            reset = getattr(self._telemetry, 'reset', None)
            if callable(reset):
                reset()
            if state.telemetry_rollup:
                restore = getattr(self._telemetry, 'restore_from_rollup', None)
                if callable(restore):
                    restore(state.telemetry_rollup)

        # Request captures: _restored_request_captures is consumed only during
        # ProxyState.__init__, which has ALREADY run by the time hydrate is called.
        # So we don't set that field — it would never be read.
        #
        # Instead, request captures are NOT restored per-request. They accumulate
        # naturally through metrics.capture_request() during normal processing.
        # The persisted request_captures in SessionState serve only as a Postgres
        # backup for the dashboard's /dashboard/requests endpoint, which reads
        # from the metrics DB (SQLite), not from in-memory state.
        #
        # On a full cold start (no metrics DB), request captures are lost.
        # This matches current behavior — the metrics DB is local per-worker.

        # Rebind delegates — critical! Without this, tagging/compaction/retrieval
        # still point at the pre-hydration _turn_tag_index and _engine_state.
        new_tti = self._turn_tag_index
        new_es = self._engine_state
        for attr in ("_tagging", "_compaction", "_retrieval", "_search"):
            delegate = getattr(self, attr, None)
            if delegate is None:
                continue
            if hasattr(delegate, "_turn_tag_index"):
                delegate._turn_tag_index = new_tti
            if hasattr(delegate, "_engine_state"):
                delegate._engine_state = new_es
        if hasattr(self, "_retriever"):
            self._retriever._turn_tag_index = new_tti
        retrieval = getattr(self, "_retrieval", None)
        if retrieval and hasattr(retrieval, "_retriever"):
            retrieval._retriever._turn_tag_index = new_tti
        # Segmenter also reads _turn_tag_index during compaction
        if hasattr(self, "_segmenter") and hasattr(self._segmenter, "_turn_tag_index"):
            self._segmenter._turn_tag_index = new_tti

    def extract_session_state(self):
        """Extract current checkpoint state for SessionStateProvider save.

        Preserves the loaded Redis version so provider.save() can do
        optimistic version checking. Also preserves timestamp and fact_signals
        on TurnTagEntry — these are used by tag velocity and compaction.
        """
        from .proxy.session_state import SessionState

        return SessionState(
            compacted_prefix_messages=self._engine_state.compacted_prefix_messages,
            flushed_prefix_messages=self._engine_state.flushed_prefix_messages,
            last_request_time=self._engine_state.last_request_time,
            last_compacted_turn=self._engine_state.last_compacted_turn,
            last_completed_turn=self._engine_state.last_completed_turn,
            last_indexed_turn=self._engine_state.last_indexed_turn,
            checkpoint_version=self._engine_state.checkpoint_version,
            conversation_generation=self._engine_state.conversation_generation,
            split_processed_tags=set(self._engine_state.split_processed_tags),
            trailing_fingerprint=self._engine_state.trailing_fingerprint,
            provider=self._engine_state.provider,
            version=self._session_state_version,  # Carry loaded version for CAS
            tool_tag_counter=self._engine_state.tool_tag_counter,
            # Use rollup only — strip raw events to keep Redis blobs small.
            telemetry_rollup={
                k: v for k, v in (
                    self._telemetry.to_dict() if hasattr(self._telemetry, 'to_dict') else {}
                ).items()
                if k != "events"
            },
            request_captures=(
                self._request_captures_provider()
                if callable(self._request_captures_provider) else []
            ),
            turn_tag_entries=[
                {
                    "turn_number": e.turn_number,
                    "canonical_turn_id": getattr(e, "canonical_turn_id", "") or "",
                    "tags": e.tags,
                    "primary_tag": e.primary_tag,
                    "message_hash": e.message_hash,
                    "sender": e.sender,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                    "session_date": getattr(e, "session_date", ""),
                    "fact_signals": [
                        {"subject": fs.subject, "verb": fs.verb,
                         "object": fs.object, "status": fs.status,
                         "fact_type": getattr(fs, "fact_type", ""),
                         "what": getattr(fs, "what", "")}
                        for fs in (e.fact_signals or [])
                    ] if e.fact_signals else [],
                    "code_refs": list(getattr(e, "code_refs", []) or []),
                }
                for e in self._turn_tag_index.entries
            ],
            working_set=[
                {
                    "tag": ws.tag,
                    "depth": ws.depth.value if hasattr(ws.depth, 'value') else ws.depth,
                    "tokens": ws.tokens,
                    "last_accessed_turn": ws.last_accessed_turn,
                }
                for ws in self._paging.working_set.values()
            ],
        )

    def _save_state(
        self,
        conversation_history: list[Message] | None,
        *,
        last_completed_turn: int | None = None,
        last_indexed_turn: int | None = None,
    ) -> bool:
        if self._session_state_provider is not None:
            # Provider mode: caller manages saves. Don't write to store/Redis directly.
            # Still update checkpoint markers for in-memory consistency.
            self._update_checkpoint_markers(
                conversation_history,
                last_completed_turn=last_completed_turn,
                last_indexed_turn=last_indexed_turn,
            )
            self._engine_state.checkpoint_version += 1
            return True
        try:
            self._update_checkpoint_markers(
                conversation_history,
                last_completed_turn=last_completed_turn,
                last_indexed_turn=last_indexed_turn,
            )
            self._engine_state.checkpoint_version += 1
            saved_at = datetime.now(timezone.utc)
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
            snapshot = EngineStateSnapshot(
                conversation_id=self.config.conversation_id,
                compacted_prefix_messages=self._engine_state.compacted_prefix_messages,
                flushed_prefix_messages=self._engine_state.flushed_prefix_messages,
                last_request_time=self._engine_state.last_request_time,
                last_compacted_turn=self._engine_state.last_compacted_turn,
                last_completed_turn=self._engine_state.last_completed_turn,
                last_indexed_turn=self._engine_state.last_indexed_turn,
                checkpoint_version=self._engine_state.checkpoint_version,
                conversation_generation=self._engine_state.conversation_generation,
                turn_tag_entries=list(self._turn_tag_index.entries),
                turn_count=max(
                    len(pair_messages_into_turns(list(conversation_history))) if conversation_history else 0,
                    self._engine_state.last_completed_turn + 1,
                    self._engine_state.last_indexed_turn + 1,
                ),
                saved_at=saved_at,
                split_processed_tags=sorted(self._engine_state.split_processed_tags),
                working_set=list(self._paging.working_set.values()),
                trailing_fingerprint=self._engine_state.trailing_fingerprint,
                telemetry_rollup=telemetry_dict,
                request_captures=captures,
                provider=self._engine_state.provider,
                tool_tag_counter=self._engine_state.tool_tag_counter,
            )
            self._store.save_engine_state(snapshot)
            # Write-through to Redis cache
            if self._session_cache:
                try:
                    history_turns = len(pair_messages_into_turns(list(conversation_history))) if conversation_history else 0
                    if conversation_history and history_turns >= self._engine_state.last_completed_turn + 1:
                        cache_snapshot = self._build_cache_snapshot(
                            conversation_history,
                            saved_at=snapshot.saved_at,
                        )
                        self._session_cache.save_snapshot(
                            self.config.conversation_id, cache_snapshot,
                        )
                except Exception as _redis_err:
                    logger.debug("Redis snapshot write failed: %s", _redis_err)
            return True
        except StaleConversationWriteError as e:
            logger.info(
                "Suppressed stale state save for conversation %s: %s",
                self.config.conversation_id[:12],
                e,
            )
            return False
        except Exception as e:
            logger.error("Failed to save engine state: %s", e)
            return False

    def _build_cache_snapshot(
        self,
        conversation_history: list,
        *,
        saved_at: datetime | None = None,
    ) -> dict:
        """Build atomic snapshot dict for Redis cache."""
        ct = self._engine_state.compacted_prefix_messages
        saved_at = saved_at or datetime.now(timezone.utc)

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
                "canonical_turn_id": getattr(e, "canonical_turn_id", "") or "",
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
                "code_refs": list(getattr(e, "code_refs", []) or []),
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
            "version": self._engine_state.checkpoint_version,
            "saved_at": saved_at.isoformat(),
            "conversation_id": self.config.conversation_id,
            "conversation_generation": self._engine_state.conversation_generation,
            "history": history,
            "turn_tag_entries": entries,
            "engine_state": {
                "compacted_prefix_messages": self._engine_state.compacted_prefix_messages,
                "flushed_prefix_messages": self._engine_state.flushed_prefix_messages,
                "last_request_time": self._engine_state.last_request_time,
                "last_compacted_turn": self._engine_state.last_compacted_turn,
                "last_completed_turn": self._engine_state.last_completed_turn,
                "last_indexed_turn": self._engine_state.last_indexed_turn,
                "checkpoint_version": self._engine_state.checkpoint_version,
                "conversation_generation": self._engine_state.conversation_generation,
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

    def persist_completed_turn(self, conversation_history: list[Message]) -> None:
        """Durably record the latest completed user/assistant pair before indexing catches up."""
        grouped = pair_messages_into_turns(list(conversation_history))
        if not grouped:
            return
        latest_turn = grouped[-1]
        assistant_messages = [msg for msg in latest_turn.messages if msg.role == "assistant"]
        if not assistant_messages:
            return
        user_messages = [msg for msg in latest_turn.messages if msg.role == "user"]
        turn_number = len(grouped) - 1
        user_msg = Message(
            role="user",
            content="\n".join(msg.content for msg in user_messages),
            timestamp=user_messages[-1].timestamp if user_messages else None,
            metadata=user_messages[-1].metadata if user_messages else None,
            raw_content=user_messages[-1].raw_content if user_messages else None,
        )
        assistant_msg = Message(
            role="assistant",
            content="\n".join(msg.content for msg in assistant_messages),
            timestamp=assistant_messages[-1].timestamp if assistant_messages else None,
            metadata=assistant_messages[-1].metadata if assistant_messages else None,
            raw_content=assistant_messages[-1].raw_content if assistant_messages else None,
        )
        entry = next(
            (
                candidate
                for candidate in reversed(self._turn_tag_index.entries)
                if candidate.turn_number == turn_number
            ),
            None,
        )
        try:
            result = IngestReconciler(
                self._store,
                self._semantic,
            ).ingest_single(
                conversation_id=self.config.conversation_id,
                user_content=user_msg.content,
                assistant_content=assistant_msg.content,
                user_raw_content=json.dumps(user_msg.raw_content) if user_msg.raw_content else None,
                assistant_raw_content=json.dumps(assistant_msg.raw_content) if assistant_msg.raw_content else None,
                primary_tag=entry.primary_tag if entry else "_general",
                tags=list(entry.tags) if entry else [],
                session_date=entry.session_date if entry else "",
                sender=entry.sender if entry else "",
                fact_signals=list(entry.fact_signals) if entry else [],
                code_refs=list(entry.code_refs) if entry else [],
            )
            if entry is not None and result.rows:
                entry.canonical_turn_id = result.rows[0].canonical_turn_id or entry.canonical_turn_id
        except StaleConversationWriteError as exc:
            logger.info(
                "Suppressed stale completed-turn persist for conversation %s: %s",
                self.config.conversation_id[:12],
                exc,
            )
            return
        except Exception:
            logger.warning(
                "Failed to persist completed turn %d for conversation %s",
                turn_number,
                self.config.conversation_id[:12],
                exc_info=True,
            )
            return
        self._save_state(
            conversation_history,
            last_completed_turn=turn_number,
        )

    def _reset_restored_state(self) -> None:
        self._turn_tag_index = TurnTagIndex()
        self._engine_state = EngineState()
        self._restored_working_set = []
        self._restored_request_captures = []
        self._restored_conversation_history = []
        self._restored_pending_turns = []
        self._restored_from_checkpoint = False
        self._restored_checkpoint_source = ""

    def _highest_indexed_turn(self) -> int:
        try:
            return max((entry.turn_number for entry in self._turn_tag_index.entries), default=-1)
        except Exception:
            return -1

    def _update_checkpoint_markers(
        self,
        conversation_history: list[Message] | None = None,
        *,
        last_completed_turn: int | None = None,
        last_indexed_turn: int | None = None,
    ) -> None:
        indexed = self._highest_indexed_turn()
        if last_indexed_turn is not None:
            indexed = max(indexed, last_indexed_turn)
        self._engine_state.last_indexed_turn = max(self._engine_state.last_indexed_turn, indexed)

        completed = -1
        grouped = pair_messages_into_turns(list(conversation_history)) if conversation_history else []
        if grouped:
            latest_turn = grouped[-1]
            if any(msg.role == "assistant" for msg in latest_turn.messages):
                completed = len(grouped) - 1
        if last_completed_turn is not None:
            completed = max(completed, last_completed_turn)
        completed = max(completed, self._engine_state.last_indexed_turn)
        self._engine_state.last_completed_turn = max(self._engine_state.last_completed_turn, completed)

        compacted_turn = (self._engine_state.compacted_prefix_messages // 2) - 1
        if self._engine_state.compacted_prefix_messages <= 0:
            compacted_turn = -1
        self._engine_state.last_compacted_turn = max(self._engine_state.last_compacted_turn, compacted_turn)

    def _cache_checkpoint_matches_store(
        self,
        db_state: EngineStateSnapshot | None,
        cached: dict,
    ) -> bool:
        if db_state is None:
            return False
        db_generation = int(
            getattr(db_state, "conversation_generation", self._conversation_generation) or 0
        )
        cached_generation = int(
            cached.get("conversation_generation", -1) if isinstance(cached, dict) else -1
        )
        if cached_generation >= 0:
            if cached_generation != db_generation:
                return False
        elif db_generation > 0:
            return False
        cached_state = cached.get("engine_state", {}) if isinstance(cached, dict) else {}
        cached_version = cached_state.get("checkpoint_version", cached.get("version", 0) if isinstance(cached, dict) else 0)
        if db_state.checkpoint_version > 0 and cached_version > 0:
            return int(cached_version) == int(db_state.checkpoint_version)
        cached_saved_at = self._parse_checkpoint_saved_at(cached.get("saved_at") if isinstance(cached, dict) else None)
        if cached_saved_at is not None and db_state.saved_at != cached_saved_at:
            return False
        cached_turns = len(cached.get("turn_tag_entries", []) or []) if isinstance(cached, dict) else 0
        cached_ct = cached_state.get("compacted_prefix_messages", 0) if isinstance(cached_state, dict) else 0
        return (
            cached_turns == len(db_state.turn_tag_entries)
            and cached_ct == db_state.compacted_prefix_messages
        )

    @staticmethod
    def _parse_checkpoint_saved_at(raw: object) -> datetime | None:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            return None
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    def _store_checkpoint_is_newer(
        self,
        db_state: EngineStateSnapshot,
        cached: dict,
    ) -> bool:
        cached_saved_at = self._parse_checkpoint_saved_at(cached.get("saved_at"))
        if cached_saved_at and db_state.saved_at > cached_saved_at:
            return True
        if cached_saved_at and db_state.saved_at < cached_saved_at:
            return False
        cached_turns = len(cached.get("turn_tag_entries", []) or [])
        cached_ct = (
            cached.get("engine_state", {}).get("compacted_prefix_messages", 0)
            if isinstance(cached.get("engine_state"), dict) else 0
        )
        return (
            db_state.turn_count > cached_turns
            or db_state.compacted_prefix_messages > cached_ct
        )

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

        # Backwards compat: bare "ollama" or "local" without explicit type → generic_openai
        if ptype in ("ollama", "local"):
            ptype = "generic_openai"

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
        *,
        run_broad_split: bool = True,
        turn_number: int | None = None,
    ) -> CompactionSignal | None:
        return self._tagging.tag_turn(
            conversation_history,
            payload_tokens,
            run_broad_split=run_broad_split,
            turn_number=turn_number,
        )

    def process_broad_tag_split(
        self,
        conversation_history: list[Message],
        *,
        mode: str = "deferred",
    ) -> SplitResult | None:
        return self._tagging.process_broad_tag_split(
            conversation_history,
            mode=mode,
        )

    def compact_if_needed(
        self,
        conversation_history: list[Message],
        signal: CompactionSignal,
        progress_callback: Callable[..., None] | None = None,
        turn_id: str = "",
    ) -> CompactionReport | None:
        return self._compaction.compact_if_needed(
            conversation_history,
            signal,
            progress_callback,
            turn_id=turn_id,
        )

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
        turn_id: str = "",
    ) -> CompactionReport | None:
        return self._compaction.compact_manual(conversation_history, turn_id=turn_id)

    def ingest_history(
        self,
        history_messages: list[Message],
        progress_callback: Callable[..., None] | None = None,
        turn_offset: int = 0,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
    ) -> int:
        return self._tagging.ingest_history(
            history_messages,
            progress_callback,
            turn_offset,
            tool_output_refs_by_turn=tool_output_refs_by_turn,
        )

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

    def sync_turns_from_payload(
        self,
        body: dict,
        fmt: "PayloadFormat",
        conversation_id: str | None = None,
    ) -> int:
        """Persist normalized ingestible chat entries from a client payload.

        Uses the shared normalized-entry extractor across Anthropic, OpenAI
        Chat, OpenAI Responses, and Gemini. Explicit scaffolding-only entries
        are skipped, and each remaining chat entry is reconciled into the
        canonical transcript ledger.

        Returns the number of genuinely new canonical entries persisted
        (0 if all already existed).
        """
        conv_id = conversation_id or self.config.conversation_id
        store = self._store
        _inner = getattr(store, '_store', None)
        if _inner is not None:
            store = _inner
        result = IngestReconciler(store, self._semantic).ingest_batch(
            conv_id,
            body=body,
            fmt=fmt,
        )
        logger.info(
            "SYNC_CANONICAL_ENTRIES: conv=%s mode=%s written=%d matched=%d appended=%d prepended=%d inserted=%d",
            conv_id[:12],
            result.merge_mode,
            result.turns_written,
            result.turns_matched,
            result.turns_appended,
            result.turns_prepended,
            result.turns_inserted,
        )
        return result.turns_written

    def find_quote(
        self,
        query: str,
        max_results: int | None = None,
        intent_context: str = "",
        session_filter: str = "",
        mode: str = "lookup",
    ) -> dict:
        return self._search.find_quote(
            query,
            max_results,
            intent_context=intent_context,
            session_filter=session_filter,
            mode=mode,
        )

    def get_turns_by_tag(
        self,
        tag: str,
        conversation_history: list[Message] | None = None,
    ) -> dict:
        return self._search.get_turns_by_tag(tag, conversation_history)

    def search_summaries(
        self,
        query: str,
        max_results: int | None = None,
        intent_context: str = "",
        session_filter: str = "",
        mode: str = "lookup",
    ) -> dict:
        return self._search.search_summaries(
            query,
            max_results,
            intent_context=intent_context,
            session_filter=session_filter,
            mode=mode,
        )

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
        mode: str = "auto",
        intent_context: str = "",
    ) -> dict:
        return self._temporal.remember_when(
            query,
            time_range,
            max_results,
            mode=mode,
            intent_context=intent_context,
        )

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
        tool_runtime=None,
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
            tool_runtime=tool_runtime,
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
