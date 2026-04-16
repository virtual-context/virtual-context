"""Semantic search: embedding-based chunk search and context relevance.

Extracted from engine.py. Owns lazy model loading, chunk embedding,
and cosine-similarity search.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..types import ChunkEmbedding, FactSignal, CanonicalTurnChunkEmbedding, QuoteResult, StoredSegment, VirtualContextConfig
from .math_utils import cosine_similarity
from .store import ContextStore

logger = logging.getLogger(__name__)

_EMBED_NOT_LOADED = object()  # sentinel for lazy embed function loading


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
                self._embed_fn = self._embedding_provider.get_embed_fn()
            else:
                # Original lazy-load path for backward compat
                try:
                    import os
                    import sys

                    from sentence_transformers import SentenceTransformer

                    model_name = self._config.retriever.embedding_model

                    # Suppress progress bar output during model loading.
                    old_stderr = sys.stderr
                    try:
                        sys.stderr = open(os.devnull, "w")
                        model = SentenceTransformer(model_name)
                    finally:
                        try:
                            sys.stderr.close()
                        except Exception:
                            pass
                        sys.stderr = old_stderr

                    def embed(texts: list[str]) -> list[list[float]]:
                        return model.encode(
                            texts, convert_to_numpy=True, show_progress_bar=False,
                        ).tolist()

                    self._embed_fn = embed
                except ImportError:
                    logger.debug(
                        "sentence-transformers not installed, context bleed gate disabled"
                    )
                    self._embed_fn = None
                except Exception:
                    logger.debug(
                        "Failed to load embedding model, semantic search disabled",
                        exc_info=True,
                    )
                    self._embed_fn = None
        return self._embed_fn

    def embed_and_store_chunks(self, stored: StoredSegment) -> None:
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
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
        self._store.store_chunk_embeddings(stored.ref, chunk_embeddings)
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
    ) -> None:
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return

        self._store.delete_canonical_turn_chunk_embeddings(
            conversation_id,
            turn_number=turn_number,
            canonical_turn_id=canonical_turn_id,
        )

        sides = [
            ("user", (user_raw_content or user_text or "")),
            ("assistant", (assistant_raw_content or assistant_text or "")),
        ]
        for side, text in sides:
            chunks = chunk_turn_text(text)
            if not chunks:
                continue
            try:
                vectors = embed_fn(chunks)
            except Exception:
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
            self._store.store_canonical_turn_chunk_embeddings(
                conversation_id,
                turn_number,
                side,
                chunk_embeddings,
                canonical_turn_id=canonical_turn_id,
            )

    def semantic_canonical_turn_search(
        self,
        query: str,
        *,
        max_results: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        """Run semantic retrieval over canonical turn chunks."""
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return []

        all_chunks = self._store.get_all_canonical_turn_chunk_embeddings(
            conversation_id=conversation_id,
        )
        if not all_chunks:
            return []

        try:
            query_vec = embed_fn([query])[0]
        except Exception:
            logger.debug("Failed to embed query for semantic turn search")
            return []

        scored: list[tuple[float, CanonicalTurnChunkEmbedding]] = []
        for chunk in all_chunks:
            sim = cosine_similarity(query_vec, chunk.embedding)
            if sim >= 0.25:
                scored.append((sim, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)

        grouped: list[QuoteResult] = []
        seen_turn_sides: set[tuple[int, str]] = set()
        for sim, chunk in scored:
            identity = (chunk.turn_number, chunk.side)
            if identity in seen_turn_sides:
                continue
            seen_turn_sides.add(identity)
            row = self._store.get_canonical_turn_rows(
                conversation_id or "",
                [chunk.turn_number],
            ).get(chunk.turn_number)
            if row is None:
                if chunk.side == "user":
                    excerpt = f"User: {chunk.text or ''}".strip()
                    matched_side = "user"
                elif chunk.side == "assistant":
                    excerpt = f"Assistant: {chunk.text or ''}".strip()
                    matched_side = "assistant"
                else:
                    excerpt = (chunk.text or "").strip()
                    matched_side = "unknown"
            else:
                user_text = row.user_content or ""
                assistant_text = row.assistant_content or ""
                if chunk.side == "user":
                    excerpt = f"User: {user_text or ''}".strip()
                    matched_side = "user"
                elif chunk.side == "assistant":
                    excerpt = f"Assistant: {assistant_text or ''}".strip()
                    matched_side = "assistant"
                else:
                    excerpt = (
                        f"User: {user_text or ''}\n\n"
                        f"Assistant: {assistant_text or ''}"
                    ).strip()
                    matched_side = "unknown"
            grouped.append(
                QuoteResult(
                    text=excerpt,
                    tag=(row.primary_tag if row is not None else "_general"),
                    segment_ref=f"turn_{chunk.turn_number}",
                    tags=list(row.tags if row is not None else []),
                    match_type="full_text_semantic",
                    similarity=round(sim, 3),
                    session_date=(row.session_date if row is not None else ""),
                    source_scope="turn",
                    turn_number=chunk.turn_number,
                    matched_side=matched_side,
                )
            )
            if len(grouped) >= max_results:
                break
        return grouped

    def semantic_search(
        self, query: str, max_results: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return []

        all_chunks = self._store.get_all_chunk_embeddings()
        if not all_chunks:
            # Lazy backfill: embed all existing segments if chunks table is empty
            all_chunks = self.backfill_chunk_embeddings(conversation_id=conversation_id)
            if not all_chunks:
                return []

        try:
            query_vec = embed_fn([query])[0]
        except Exception:
            logger.debug("Failed to embed query for semantic search")
            return []

        # Score all chunks
        scored: list[tuple[float, ChunkEmbedding]] = []
        for chunk in all_chunks:
            sim = cosine_similarity(query_vec, chunk.embedding)
            if sim >= 0.25:
                scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by segment_ref (best chunk per segment)
        seen_refs: set[str] = set()
        results: list[QuoteResult] = []
        for sim, chunk in scored:
            if chunk.segment_ref in seen_refs:
                continue
            seen_refs.add(chunk.segment_ref)
            # Look up segment tags and metadata; conversation_id filter
            # naturally excludes chunks from other conversations (get_segment
            # returns None for non-matching conversation).
            seg = self._store.get_segment(chunk.segment_ref, conversation_id=conversation_id)
            if seg is None:
                continue
            results.append(QuoteResult(
                text=chunk.text,
                tag=seg.primary_tag,
                segment_ref=chunk.segment_ref,
                tags=seg.tags,
                match_type="semantic",
                similarity=round(sim, 3),
                session_date=seg.metadata.session_date if seg else "",
            ))
            if len(results) >= max_results:
                break

        return results

    def backfill_chunk_embeddings(
        self, conversation_id: str | None = None,
    ) -> list[ChunkEmbedding]:
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return []

        all_tags = self._store.get_all_tags(conversation_id=conversation_id)
        if not all_tags:
            return []

        logger.info("Backfilling chunk embeddings for semantic search...")
        all_chunks: list[ChunkEmbedding] = []
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
                self._store.store_chunk_embeddings(seg.ref, chunk_embeddings)
                all_chunks.extend(chunk_embeddings)

        logger.info("Backfilled %d chunk embeddings", len(all_chunks))
        return all_chunks

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

        embeddings = embed_fn([current_text[:2000], recent[:2000]])
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
