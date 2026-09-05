"""Semantic search: embedding-based chunk search and context relevance.

Extracted from engine.py. Owns lazy model loading, chunk embedding,
and cosine-similarity search.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

from ..types import (
    ChunkEmbedding,
    FactSignal,
    CanonicalTurnRow,
    CanonicalTurnChunkEmbedding,
    QuoteResult,
    SourceProvenance,
    SpeakerRetrievalContext,
    StoredSegment,
    VirtualContextConfig,
    channel_excerpt_prefix,
    channel_matches,
)
from .math_utils import cosine_similarity
from .store import ContextStore

logger = logging.getLogger(__name__)

_EMBED_NOT_LOADED = object()  # sentinel for lazy embed function loading
_NATIVE_VECTOR_PAGE_SIZE = 200


@dataclass(frozen=True)
class _ScoredTurnChunk:
    """Native-search metadata consumed by the existing source renderer.

    No embedding field: vectors stay inside the database on this path.
    """

    conversation_id: str
    canonical_turn_id: str
    turn_number: int
    side: str
    text: str

    @classmethod
    def from_row(cls, row: dict, conversation_id: str | None) -> _ScoredTurnChunk:
        return cls(
            conversation_id=row.get("conversation_id") or conversation_id or "",
            canonical_turn_id=row.get("canonical_turn_id") or "",
            turn_number=int(row.get("turn_number", -1)),
            side=row.get("side") or "",
            text=row.get("text") or "",
        )


def chunk_segment_text(full_text: str, max_words: int = 250, min_words: int = 20) -> list[str]:
    """Split segment full_text into overlapping chunks for embedding.

    Splits on double-newline (message boundaries), merges tiny chunks,
    and applies sliding window with overlap for oversized chunks.
    """
    if not full_text or not full_text.strip():
        return []

    # Split on message boundaries
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # Merge tiny paragraphs
    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            candidate = buffer + "\n\n" + para
        else:
            candidate = para
        if len(candidate.split()) <= max_words:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)

    # Split oversized chunks with sliding window
    chunks: list[str] = []
    overlap_words = 30
    for chunk in merged:
        words = chunk.split()
        if len(words) <= max_words:
            chunks.append(chunk)
        else:
            start = 0
            while start < len(words):
                end = min(start + max_words, len(words))
                chunks.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start += max_words - overlap_words

    # Filter fragments that are too small
    return [c for c in chunks if len(c.split()) >= min_words]


def chunk_turn_text(text: str, max_words: int = 180, min_words: int = 3) -> list[str]:
    """Split turn text into smaller embedding chunks.

    Turn text is usually shorter than segment text, so the minimum word
    threshold is lower and we preserve a single short chunk when needed.
    """
    chunks = chunk_segment_text(text, max_words=max_words, min_words=min_words)
    if chunks:
        return chunks
    stripped = (text or "").strip()
    return [stripped] if stripped else []


_HOST_SCAFFOLD_MARKERS = (
    "Conversation info (untrusted metadata):",
    "OpenClaw assembled context for this turn:",
    "Conversation context (untrusted, chronological, selected for current message):",
)


def _user_embedding_text(user_text: str, user_raw_content: str | None) -> str:
    """Choose the user lane without re-admitting host wrapper scaffolding.

    Canonical text is the admitted representation and therefore wins whenever
    it is non-empty.  Raw content remains the fallback for attachment-only
    messages, except when it contains a known host wrapper: an empty canonical
    value in that case means there is no admitted user text to index.
    """
    canonical = user_text or ""
    if canonical.strip():
        return canonical
    raw = user_raw_content or ""
    if any(marker in raw for marker in _HOST_SCAFFOLD_MARKERS):
        return canonical
    return raw


class SemanticSearchManager:
    """Manages embedding model loading, chunk storage, and semantic search."""

    def __init__(
        self,
        store: ContextStore,
        config: VirtualContextConfig,
        embedding_provider=None,
    ) -> None:
        self._store = store
        self._config = config
        self._embedding_provider = embedding_provider
        self._embed_fn = _EMBED_NOT_LOADED

    def get_embed_fn(self) -> Callable[[list[str]], list[list[float]]] | None:
        """Lazy-load the embedding function.

        Returns a callable that takes a list of strings and returns a list of
        float vectors, or ``None`` if sentence-transformers is not installed.
        """
        if self._embed_fn is _EMBED_NOT_LOADED:
            if self._embedding_provider is not None:
                fn = self._embedding_provider.get_embed_fn()
                if fn is None:
                    # Do not cache a provider's None: for a disabled provider
                    # re-consulting is free, and for a provider that has not
                    # produced its callable yet, caching None here would make
                    # a transient condition permanent.
                    return None
                self._embed_fn = fn
            else:
                # Original lazy-load path for backward compat, served from
                # the process-wide model cache.
                from .embedding_provider import get_shared_embed_fn

                fn = get_shared_embed_fn(
                    self._config.retriever.embedding_model,
                )
                if fn is None:
                    logger.debug(
                        "embedding model unavailable, semantic search disabled"
                    )
                self._embed_fn = fn
        return self._embed_fn

    def _native_vector_enabled(self) -> bool:
        return bool(getattr(self._config.retriever, "vector_search_enabled", False))

    def _native_embedding_pages(
        self,
        query: str,
        method_name: str,
        *,
        conversation_id: str | None,
        speaker_context: SpeakerRetrievalContext | None = None,
    ) -> Iterator[list[dict]]:
        """Stream exact DB-ranked pages without an embedding materialization fallback.

        Enabling native search is an operational contract. A missing migration
        or query failure must remain visible instead of silently restoring the
        unbounded Python scan the operator explicitly opted out of.
        """
        ready = getattr(self._store, "vector_search_ready", None)
        search = getattr(self._store, method_name, None)
        migration_hint = (
            "Native vector search is unavailable. Run `virtual-context admin migrate-semantic-vectors` "
            "for this PostgreSQL store and verify its configured embedding model, "
            "or explicitly disable retrieval.vector_search_enabled."
        )
        try:
            supported = callable(ready) and ready(self._config.retriever.embedding_model) is True
        except Exception as exc:
            logger.error("VECTOR_SEARCH_CAPABILITY_FAILED method=%s", method_name, exc_info=True)
            raise RuntimeError(migration_hint) from exc
        if not supported or not callable(search):
            logger.error("VECTOR_SEARCH_UNAVAILABLE method=%s", method_name)
            raise RuntimeError(migration_hint)

        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return
        try:
            query_vec = embed_fn([query])[0]
        except Exception:
            logger.debug("Failed to embed query for native semantic search", exc_info=True)
            return
        if not query_vec or not any(value != 0.0 for value in query_vec):
            return
        if not all(math.isfinite(value) for value in query_vec):
            raise ValueError("Native vector search requires a finite query embedding")

        after = None
        while True:
            kwargs = {
                "conversation_id": conversation_id,
                "limit": _NATIVE_VECTOR_PAGE_SIZE,
                "after": after,
                "min_similarity": 0.25,
            }
            if speaker_context is not None:
                kwargs["speaker_context"] = speaker_context
            try:
                page = search(query_vec, **kwargs)
            except Exception as exc:
                logger.error("VECTOR_SEARCH_QUERY_FAILED method=%s", method_name, exc_info=True)
                raise RuntimeError(migration_hint) from exc
            if not page:
                return
            next_cursor = page[-1].get("cursor")
            if next_cursor is None or (after is not None and next_cursor <= after):
                raise RuntimeError("Native vector search returned a non-advancing page cursor")
            yield page
            after = next_cursor

    def _native_segment_search(
        self, query: str, max_results: int, conversation_id: str | None,
    ) -> list[QuoteResult]:
        results: list[QuoteResult] = []
        seen_refs: set[str] = set()
        for page in self._native_embedding_pages(
            query, "search_segment_chunks_by_embedding", conversation_id=conversation_id,
        ):
            for candidate in page:
                ref = candidate["segment_ref"]
                similarity = float(candidate["similarity"])
                if ref in seen_refs or not similarity >= 0.25:
                    continue
                segment = self._store.get_segment(ref, conversation_id=conversation_id)
                if segment is None:
                    continue
                seen_refs.add(ref)
                results.append(QuoteResult(
                    text=candidate["text"], tag=segment.primary_tag,
                    segment_ref=ref, tags=segment.tags, match_type="semantic",
                    similarity=round(similarity, 3), session_date=segment.metadata.session_date,
                ))
                if len(results) >= max_results:
                    return results
        return results

    def _native_speaker_turn_search(
        self, query: str, *, max_results: int, conversation_id: str | None,
        channel: str, speaker_context: SpeakerRetrievalContext,
    ) -> list[QuoteResult]:
        results: list[QuoteResult] = []
        seen: set[tuple[str, str, str]] = set()
        wanted_channel = (channel or "").strip()
        skipped_no_row = 0
        for page in self._native_embedding_pages(
            query, "search_speaker_turn_chunks_by_embedding",
            conversation_id=conversation_id, speaker_context=speaker_context,
        ):
            chunks = [
                (float(candidate["similarity"]), _ScoredTurnChunk.from_row(candidate, conversation_id))
                for candidate in page
                if float(candidate["similarity"]) >= 0.25
            ]
            keys = list(dict.fromkeys(
                (chunk.conversation_id, chunk.canonical_turn_id)
                for _similarity, chunk in chunks
                if chunk.conversation_id and chunk.canonical_turn_id
                and (chunk.conversation_id, chunk.canonical_turn_id, chunk.side) not in seen
            ))
            physical = self._store.get_canonical_turn_rows_by_id(
                keys, speaker_context=speaker_context,
            ) if keys else {}
            for similarity, chunk in chunks:
                identity = (chunk.conversation_id, chunk.canonical_turn_id, chunk.side)
                if identity in seen:
                    continue
                row = physical.get((chunk.conversation_id, chunk.canonical_turn_id))
                if row is None:
                    skipped_no_row += 1
                    continue
                if wanted_channel and not channel_matches(
                    wanted_channel, row.origin_channel_id, row.origin_channel_label,
                ):
                    continue
                seen.add(identity)
                results.append(self._format_physical_semantic_result(
                    similarity, chunk, row, channel=wanted_channel,
                ))
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        if skipped_no_row:
            logger.warning(
                "SEMANTIC_CHUNK_NO_PHYSICAL_ROW conv=%s skipped=%d",
                (conversation_id or "")[:12], skipped_no_row,
            )
        return results

    def _native_canonical_turn_search(
        self, query: str, *, max_results: int, conversation_id: str | None,
        channel: str,
    ) -> list[QuoteResult]:
        results: list[QuoteResult] = []
        seen: set[tuple] = set()
        wanted_channel = (channel or "").strip()
        for page in self._native_embedding_pages(
            query, "search_canonical_turn_chunks_by_embedding", conversation_id=conversation_id,
        ):
            candidates = [
                (candidate, _ScoredTurnChunk.from_row(candidate, conversation_id))
                for candidate in page
                if candidate.get("side") != "subject"
                and float(candidate["similarity"]) >= 0.25
            ]
            for candidate, chunk in candidates:
                identity = (chunk.conversation_id, chunk.canonical_turn_id, chunk.side)
                # Ordinal turn numbers are not logical group IDs. Both
                # unscoped and channel-local legacy results use the exact
                # physical source projected by the bounded storage page.
                row = candidate.get("physical_row")
                if row is None or (
                    row.conversation_id != chunk.conversation_id
                    or row.canonical_turn_id != chunk.canonical_turn_id
                    or (wanted_channel and not channel_matches(
                        wanted_channel, row.origin_channel_id, row.origin_channel_label,
                    ))
                ):
                    continue
                if identity in seen:
                    continue
                seen.add(identity)
                results.append(self._format_legacy_semantic_result(
                    float(candidate["similarity"]), chunk, row, channel=wanted_channel,
                ))
                if len(results) >= max_results:
                    return results
        return results

    @staticmethod
    def _format_legacy_semantic_result(
        similarity: float,
        chunk: CanonicalTurnChunkEmbedding | _ScoredTurnChunk,
        row: CanonicalTurnRow,
        *,
        channel: str = "",
    ) -> QuoteResult:
        """Keep legacy turn labels while rendering the exact physical source."""
        if chunk.side == "user":
            content = row.user_content
            excerpt, matched_side = f"User: {content or ''}".strip(), "user"
        elif chunk.side == "assistant":
            content = row.assistant_content
            excerpt, matched_side = f"Assistant: {content or ''}".strip(), "assistant"
        else:
            excerpt = f"User: {row.user_content or ''}\n\nAssistant: {row.assistant_content or ''}".strip()
            matched_side = "unknown"
        turn_number = chunk.turn_number
        if channel:
            excerpt = channel_excerpt_prefix(
                row.origin_channel_id, row.origin_channel_label,
            ) + excerpt
            if row.turn_number >= 0:
                turn_number = row.turn_number
        return QuoteResult(
            text=excerpt,
            tag=row.primary_tag,
            segment_ref=(f"canonical_turn_{chunk.canonical_turn_id}"
                         if chunk.canonical_turn_id else f"turn_{turn_number}"),
            tags=list(row.tags or []),
            match_type="full_text_semantic", similarity=round(similarity, 3),
            session_date=row.session_date,
            source_scope="turn", turn_number=turn_number, matched_side=matched_side,
        )

    def embed_and_store_chunks(
        self,
        stored: StoredSegment,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
        conversation_id: str | None = None,
        disable_replacement_passes: bool = False,
    ) -> None:
        """Compute and store chunk embeddings for a segment.

        When called from a compaction phase, the caller forwards the
        guard kwargs so ``store_chunk_embeddings`` writes through the
        active operation-id fence (fencing plan §5.6 caller-side
        propagation). Legacy non-compaction callers (the lazy backfill
        path at line ~409 below) omit the kwargs and continue through
        the documented all-None branch.

        When ``disable_replacement_passes`` is True (backlog-sweeper
        dispatch), the caller suppresses the DELETE-then-INSERT
        semantics by skipping the write entirely when the segment_ref
        already has chunks. The new-segment path is a pure insert and
        proceeds normally. Per fencing plan §7.2 #4.
        """
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return
        if disable_replacement_passes:
            # Single-row probe on segment_chunks(segment_ref) replaces
            # the previous O(N) ``get_all_chunk_embeddings`` scan that
            # filtered by ref in Python. Backends override
            # ``has_chunks_for_segment`` with a ``LIMIT 1`` SELECT; the
            # default falls back to the scan so non-backend stores
            # stay functional. Per codex P5 follow-up.
            #
            # Log shape preserves the pre-cleanup ``(%d pre-existing
            # chunks)`` field so downstream log parsers / dashboards
            # do not regress. The probe itself is boolean so we cannot
            # report the actual count without a second query; ``>=1``
            # is the closest faithful value at no additional cost.
            if self._store.has_chunks_for_segment(stored.ref):
                logger.info(
                    "C2R gate: skipping chunk embedding write for segment %s "
                    "(%s pre-existing chunks)",
                    stored.ref, ">=1",
                )
                return
        chunks = chunk_segment_text(stored.full_text)
        if not chunks:
            return
        try:
            vectors = embed_fn(chunks)
        except Exception:
            logger.debug("Failed to embed chunks for %s", stored.ref)
            return
        chunk_embeddings = [
            ChunkEmbedding(
                segment_ref=stored.ref,
                chunk_index=i,
                text=text,
                embedding=vec,
            )
            for i, (text, vec) in enumerate(zip(chunks, vectors))
        ]
        self._store.store_chunk_embeddings(
            stored.ref, chunk_embeddings,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
            conversation_id=conversation_id,
            embedding_model=self._config.retriever.embedding_model,
        )
        logger.debug("Stored %d chunk embeddings for segment %s", len(chunk_embeddings), stored.ref)

    def embed_and_store_turn(
        self,
        conversation_id: str,
        turn_number: int,
        canonical_turn_id: str | None = None,
        *,
        user_text: str = "",
        assistant_text: str = "",
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
        reply_target_body: str = "",
    ) -> bool:
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return True

        self._store.delete_canonical_turn_chunk_embeddings(
            conversation_id,
            turn_number=turn_number,
            canonical_turn_id=canonical_turn_id,
        )

        # ``subject`` indexes the reply-target lane under the same physical
        # row. It is a separate side, never concatenated into requester text,
        # and the shipped search branch explicitly ignores it.
        sides = [
            ("user", _user_embedding_text(user_text, user_raw_content)),
            ("assistant", (assistant_raw_content or assistant_text or "")),
            ("subject", (reply_target_body or "")),
        ]
        complete = True
        for side, text in sides:
            chunks = chunk_turn_text(text)
            if not chunks:
                continue
            try:
                vectors = embed_fn(chunks)
            except Exception:
                complete = False
                logger.warning(
                    "CANONICAL_TURN_EMBED_FAILED side=%s conv=%s turn=%d",
                    side,
                    conversation_id,
                    turn_number,
                    exc_info=True,
                )
                continue
            chunk_embeddings = [
                CanonicalTurnChunkEmbedding(
                    conversation_id=conversation_id,
                    canonical_turn_id=canonical_turn_id or "",
                    turn_number=turn_number,
                    side=side,
                    chunk_index=i,
                    text=chunk_text,
                    embedding=vec,
                )
                for i, (chunk_text, vec) in enumerate(zip(chunks, vectors))
            ]
            try:
                self._store.store_canonical_turn_chunk_embeddings(
                    conversation_id,
                    turn_number,
                    side,
                    chunk_embeddings,
                    canonical_turn_id=canonical_turn_id,
                    embedding_model=self._config.retriever.embedding_model,
                )
            except Exception:
                complete = False
                logger.warning(
                    "CANONICAL_TURN_EMBED_STORE_FAILED side=%s conv=%s turn=%d",
                    side,
                    conversation_id,
                    turn_number,
                    exc_info=True,
                )
        return complete

    def semantic_canonical_turn_search(
        self,
        query: str,
        *,
        max_results: int = 5,
        conversation_id: str | None = None,
        channel: str = "",
        speaker_context: SpeakerRetrievalContext | None = None,
    ) -> list[QuoteResult]:
        """Run semantic retrieval over canonical turn chunks.

        ``speaker_context`` is the branch selector. ``None`` runs the shipped
        legacy branch: turn-label presentation from exact physical source
        hydration, and no ``subject``-side consumption. A non-None
        context selects the physical role-local branch, which threads the
        same immutable context through candidate enumeration and one batched
        physical-row hydration.

        A non-empty ``channel`` filters scored chunks post-score but
        PRE-acceptance-limit: scanning continues down the ranking until
        ``max_results`` in-channel results are accepted, so a global top hit
        outside the channel cannot starve a lower-ranked in-channel one.
        """
        if max_results <= 0:
            return []
        if speaker_context is not None:
            return self._speaker_semantic_turn_search(
                query,
                max_results=max_results,
                conversation_id=conversation_id,
                channel=channel,
                speaker_context=speaker_context,
            )
        if self._native_vector_enabled():
            return self._native_canonical_turn_search(
                query, max_results=max_results, conversation_id=conversation_id,
                channel=channel,
            )

        return self._stream_canonical_turn_search(
            query, max_results=max_results, conversation_id=conversation_id,
            channel=channel, speaker_context=None,
        )

    def _speaker_semantic_turn_search(
        self,
        query: str,
        *,
        max_results: int,
        conversation_id: str | None,
        channel: str,
        speaker_context: SpeakerRetrievalContext,
    ) -> list[QuoteResult]:
        """Physical, role-local semantic retrieval.

        Candidate enumeration and hydration both receive the same immutable
        request context; the store proves scope before anything is scored or
        limited here. Hydration is ONE batched physical lookup by
        ``(conversation_id, canonical_turn_id)`` on both the scoped and
        unscoped paths — never the logical seam, which merges sibling rows
        and can transfer provenance across them. A chunk whose physical row
        is missing or inadmissible proves nothing: it is skipped and
        reported, and the admin reindex owns the repair.
        """
        if max_results <= 0:
            return []
        if self._native_vector_enabled():
            return self._native_speaker_turn_search(
                query, max_results=max_results, conversation_id=conversation_id,
                channel=channel, speaker_context=speaker_context,
            )
        return self._stream_canonical_turn_search(
            query, max_results=max_results, conversation_id=conversation_id,
            channel=channel, speaker_context=speaker_context,
        )

    def _embedding_pages(
        self, method_name: str, *, conversation_id: str | None,
        speaker_context: SpeakerRetrievalContext | None = None,
    ) -> Iterator[list[dict]]:
        """Hydrate one keyset page, never an archive-sized embedding list."""
        getter = getattr(self._store, method_name)
        after = None
        while True:
            kwargs = dict(conversation_id=conversation_id, limit=200, after=after)
            if method_name == "get_canonical_turn_chunk_embedding_page":
                kwargs["speaker_context"] = speaker_context
            page = getter(**kwargs)
            if not page:
                return
            cursor = page[-1].get("cursor")
            if cursor is None or (after is not None and tuple(cursor) <= after):
                raise RuntimeError("Embedding page cursor did not advance")
            if len(page) > 200:
                raise RuntimeError("Embedding store exceeded its page limit")
            after = tuple(cursor)
            yield page

    def _query_vector(self, query: str) -> list[float] | None:
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return None
        try:
            vector = embed_fn([query])[0]
        except Exception:
            logger.debug("Failed to embed semantic query")
            return None
        if not vector or not any(vector) or not all(math.isfinite(x) for x in vector):
            return None
        return vector

    @staticmethod
    def _keep_best_result(top: dict, identity: tuple, score: float, sequence: int,
                          result: QuoteResult, limit: int) -> None:
        """Bound result state by requested distinct hits, retaining stable ties."""
        rank = (score, -sequence)
        existing = top.get(identity)
        if existing is not None and existing[0] >= rank:
            return
        if existing is None and len(top) >= limit:
            weakest = min(top, key=lambda key: top[key][0])
            if top[weakest][0] >= rank:
                return
            del top[weakest]
        top[identity] = (rank, result)

    def _stream_canonical_turn_search(
        self, query: str, *, max_results: int, conversation_id: str | None,
        channel: str, speaker_context: SpeakerRetrievalContext | None,
    ) -> list[QuoteResult]:
        vector = self._query_vector(query)
        if vector is None:
            return []
        top: dict = {}
        sequence = 0
        missing = 0
        wanted_channel = (channel or "").strip()
        for page in self._embedding_pages(
            "get_canonical_turn_chunk_embedding_page",
            conversation_id=conversation_id, speaker_context=speaker_context,
        ):
            candidates = []
            for candidate in page:
                sequence += 1
                if speaker_context is None and candidate.get("side") == "subject":
                    continue
                score = cosine_similarity(vector, candidate["embedding"])
                if math.isfinite(score) and score >= 0.25:
                    candidates.append((sequence, score, candidate,
                                       _ScoredTurnChunk.from_row(candidate, conversation_id)))
            if speaker_context is not None:
                keys = list(dict.fromkeys(
                    (chunk.conversation_id, chunk.canonical_turn_id)
                    for _, _, _, chunk in candidates
                ))
                physical = self._store.get_canonical_turn_rows_by_id(
                    keys, speaker_context=speaker_context,
                ) if keys else {}
            for order, score, candidate, chunk in candidates:
                physical_key = (chunk.conversation_id, chunk.canonical_turn_id)
                if speaker_context is not None:
                    row = physical.get(physical_key)
                else:
                    row = candidate.get("physical_row")
                    if row is not None and (row.conversation_id, row.canonical_turn_id) != physical_key:
                        row = None
                if row is None:
                    missing += 1
                    continue
                if wanted_channel and not channel_matches(
                    wanted_channel, row.origin_channel_id, row.origin_channel_label,
                ):
                    continue
                identity = (*physical_key, chunk.side)
                formatter = (self._format_physical_semantic_result
                             if speaker_context is not None else self._format_legacy_semantic_result)
                result = formatter(score, chunk, row, channel=wanted_channel)
                self._keep_best_result(top, identity, score, order, result, max_results)
        if missing:
            logger.warning("SEMANTIC_CHUNK_NO_PHYSICAL_ROW conv=%s skipped=%d",
                           (conversation_id or "")[:12], missing)
        return [value[1] for value in sorted(top.values(), key=lambda item: item[0], reverse=True)]

    def _format_physical_semantic_result(
        self,
        sim: float,
        chunk: CanonicalTurnChunkEmbedding | _ScoredTurnChunk,
        row: CanonicalTurnRow,
        *,
        channel: str,
    ) -> QuoteResult:
        """Format one candidate from its exact physical row.

        Attribution is role-local: the requester lane carries only the row's
        ``sender_actor_id``, the subject lane only ``reply_subject_actor_id``,
        and the assistant lane never a human actor. A subject excerpt is the
        copied reply text alone, with no ``User:`` label that would misassign
        the quote, and the raw stored reply label rides along only as an
        unverified claim. An unrecognized side is honestly unattributed
        rather than guessed.
        """
        user_text = row.user_content or ""
        assistant_text = row.assistant_content or ""
        claimed_subject_label = ""
        if chunk.side == "user":
            excerpt = f"User: {user_text}".strip()
            matched_side = "user"
            source_role = "requester"
            actor_id = row.sender_actor_id or ""
        elif chunk.side == "assistant":
            excerpt = f"Assistant: {assistant_text}".strip()
            matched_side = "assistant"
            source_role = "assistant"
            actor_id = ""
        elif chunk.side == "subject":
            excerpt = (row.reply_target_body or "").strip()
            matched_side = ""
            source_role = "subject"
            actor_id = row.reply_subject_actor_id or ""
            claimed_subject_label = row.reply_subject_label or ""
        else:
            excerpt = (
                f"User: {user_text}\n\nAssistant: {assistant_text}"
            ).strip()
            matched_side = "unknown"
            source_role = "unattributed"
            actor_id = ""
        if channel:
            excerpt = channel_excerpt_prefix(
                row.origin_channel_id, row.origin_channel_label,
            ) + excerpt
        turn_number = (
            row.turn_number if row.turn_number >= 0 else chunk.turn_number
        )
        canonical_turn_id = chunk.canonical_turn_id or row.canonical_turn_id or ""
        return QuoteResult(
            text=excerpt,
            tag=row.primary_tag,
            segment_ref=f"canonical_turn_{canonical_turn_id or turn_number}",
            tags=list(row.tags or []),
            match_type="full_text_semantic",
            similarity=round(sim, 3),
            session_date=row.session_date,
            source_scope="turn",
            turn_number=turn_number,
            matched_side=matched_side,
            provenance=SourceProvenance(
                conversation_id=row.conversation_id or chunk.conversation_id or "",
                canonical_turn_id=canonical_turn_id,
                source_role=source_role,
                actor_id=actor_id,
                audience_conversation_id=row.audience_conversation_id or "",
                audience_attribution_version=int(
                    row.audience_attribution_version or 0
                ),
                origin_channel_id=row.origin_channel_id or "",
                claimed_subject_label=claimed_subject_label,
                source_message_id=getattr(row, "source_message_id", "") or "",
            ),
        )

    def semantic_search(
        self, query: str, max_results: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        if max_results <= 0:
            return []
        if self._native_vector_enabled():
            return self._native_segment_search(query, max_results, conversation_id)
        vector = self._query_vector(query)
        if vector is None:
            return []
        top: dict = {}
        sequence = 0
        found_chunks = False
        for attempt in range(2):
            for page in self._embedding_pages(
                "get_segment_chunk_embedding_page", conversation_id=conversation_id,
            ):
                found_chunks = True
                for candidate in page:
                    sequence += 1
                    score = cosine_similarity(vector, candidate["embedding"])
                    if not math.isfinite(score) or score < 0.25:
                        continue
                    ref = candidate["segment_ref"]
                    if all(key in candidate for key in ("conversation_id", "primary_tag", "tags", "session_date")):
                        # Built-in page stores join/check the live scoped
                        # source before returning these small metadata fields.
                        if conversation_id is not None and candidate["conversation_id"] != conversation_id:
                            continue
                        tag, tags, session_date = candidate["primary_tag"], candidate["tags"], candidate["session_date"]
                    else:
                        # Compatibility for third-party stores implementing
                        # the original page contract without display metadata.
                        segment = self._store.get_segment(ref, conversation_id=conversation_id)
                        if segment is None:
                            continue
                        tag, tags, session_date = segment.primary_tag, segment.tags, segment.metadata.session_date
                    result = QuoteResult(
                        text=candidate["text"], tag=tag,
                        segment_ref=ref, tags=tags, match_type="semantic",
                        similarity=round(score, 3), session_date=session_date,
                    )
                    self._keep_best_result(top, (ref,), score, sequence, result, max_results)
            if found_chunks or attempt:
                break
            # Lazy repair writes bounded segment batches; it does not return
            # an archive-sized collection to the search caller.
            self.backfill_chunk_embeddings(conversation_id=conversation_id)
        return [value[1] for value in sorted(top.values(), key=lambda item: item[0], reverse=True)]

    def backfill_chunk_embeddings(
        self, conversation_id: str | None = None,
    ) -> int:
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return 0

        all_tags = self._store.get_all_tags(conversation_id=conversation_id)
        if not all_tags:
            return 0

        logger.info("Backfilling chunk embeddings for semantic search...")
        chunk_count = 0
        for tag_stat in all_tags:
            segments = self._store.get_segments_by_tags(
                [tag_stat.tag], limit=100, conversation_id=conversation_id,
            )
            for seg in segments:
                chunks = chunk_segment_text(seg.full_text)
                if not chunks:
                    continue
                try:
                    vectors = embed_fn(chunks)
                except Exception:
                    continue
                chunk_embeddings = [
                    ChunkEmbedding(
                        segment_ref=seg.ref,
                        chunk_index=i,
                        text=text,
                        embedding=vec,
                    )
                    for i, (text, vec) in enumerate(zip(chunks, vectors))
                ]
                self._store.store_chunk_embeddings(
                    seg.ref, chunk_embeddings,
                    embedding_model=self._config.retriever.embedding_model,
                )
                chunk_count += len(chunk_embeddings)

        logger.info("Backfilled %d chunk embeddings", chunk_count)
        return chunk_count

    def context_is_relevant(
        self, current_text: str, context_pairs: list[str],
    ) -> bool | tuple[bool, float]:
        """Check if current turn is semantically similar to the most recent context pair.

        Compares the current turn's combined text against the last user+assistant
        pair in the collected context using embedding cosine similarity.
        Returns ``True`` (pass context) when similarity >= threshold, or when
        embeddings are unavailable (graceful degradation).
        """
        return self.context_is_relevant_with_score(current_text, context_pairs)[0]

    def context_is_relevant_with_score(
        self, current_text: str, context_pairs: list[str],
    ) -> tuple[bool, float]:
        """Like context_is_relevant but also returns the cosine similarity score.

        Returns (is_relevant, similarity). When embeddings are unavailable,
        returns (True, -1.0) to indicate graceful pass-through.
        """
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return True, -1.0

        # Compare against the most recent pair in context
        if len(context_pairs) >= 2:
            recent = context_pairs[-2] + " " + context_pairs[-1]
        else:
            recent = " ".join(context_pairs)

        try:
            embeddings = embed_fn([current_text[:2000], recent[:2000]])
        except Exception:
            # A failed embed call gets the same graceful pass-through as an
            # unavailable one: the gate may not turn an embedding outage into
            # dropped context.
            logger.debug("Context bleed gate embed failed; passing through")
            return True, -1.0
        sim = cosine_similarity(embeddings[0], embeddings[1])
        threshold = self._config.tag_generator.context_bleed_threshold

        logger.debug("Context bleed gate: sim=%.3f threshold=%.3f", sim, threshold)
        return sim >= threshold, sim


def persist_turn_with_embeddings(
    store: ContextStore,
    semantic: SemanticSearchManager,
    *,
    conversation_id: str,
    turn_number: int,
    canonical_turn_id: str | None = None,
    sort_key: float | None = None,
    user_content: str,
    assistant_content: str,
    user_raw_content: str | None = None,
    assistant_raw_content: str | None = None,
    primary_tag: str = "_general",
    tags: list[str] | None = None,
    session_date: str = "",
    sender: str = "",
    fact_signals: list[FactSignal] | None = None,
    code_refs: list[dict] | None = None,
) -> None:
    """Persist a turn pair into the canonical turn ledger and embeddings store."""
    from .ingest_reconciler import IngestReconciler

    IngestReconciler(store, semantic).ingest_single(
        conversation_id=conversation_id,
        user_content=user_content,
        assistant_content=assistant_content,
        user_raw_content=user_raw_content,
        assistant_raw_content=assistant_raw_content,
        primary_tag=primary_tag,
        tags=tags,
        session_date=session_date,
        sender=sender,
        fact_signals=fact_signals,
        code_refs=code_refs,
    )
