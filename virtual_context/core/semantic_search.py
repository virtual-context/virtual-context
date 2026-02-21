"""Semantic search: embedding-based chunk search and context relevance.

Extracted from engine.py. Owns lazy model loading, chunk embedding,
and cosine-similarity search.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..types import ChunkEmbedding, QuoteResult, StoredSegment, VirtualContextConfig
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


class SemanticSearchManager:
    """Manages embedding model loading, chunk storage, and semantic search."""

    def __init__(
        self,
        store: ContextStore,
        config: VirtualContextConfig,
    ) -> None:
        self._store = store
        self._config = config
        self._embed_fn = _EMBED_NOT_LOADED

    def get_embed_fn(self) -> Callable[[list[str]], list[list[float]]] | None:
        """Lazy-load the embedding function.

        Returns a callable that takes a list of strings and returns a list of
        float vectors, or ``None`` if sentence-transformers is not installed.
        """
        if self._embed_fn is _EMBED_NOT_LOADED:
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
        """Chunk a segment's full_text, embed, and store vectors."""
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

    def semantic_search(self, query: str, max_results: int = 5) -> list[QuoteResult]:
        """Embedding-based semantic search over stored chunk vectors."""
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return []

        all_chunks = self._store.get_all_chunk_embeddings()
        if not all_chunks:
            # Lazy backfill: embed all existing segments if chunks table is empty
            all_chunks = self.backfill_chunk_embeddings()
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
            # Look up segment tags and metadata
            seg = self._store.get_segment(chunk.segment_ref)
            results.append(QuoteResult(
                text=chunk.text,
                tag=seg.primary_tag if seg else "",
                segment_ref=chunk.segment_ref,
                tags=seg.tags if seg else [],
                match_type="semantic",
                similarity=round(sim, 3),
                session_date=seg.metadata.session_date if seg else "",
            ))
            if len(results) >= max_results:
                break

        return results

    def backfill_chunk_embeddings(self) -> list[ChunkEmbedding]:
        """One-time backfill: embed all existing segments' full_text."""
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return []

        all_tags = self._store.get_all_tags()
        if not all_tags:
            return []

        logger.info("Backfilling chunk embeddings for semantic search...")
        all_chunks: list[ChunkEmbedding] = []
        for tag_stat in all_tags:
            segments = self._store.get_segments_by_tags([tag_stat.tag], limit=100)
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
    ) -> bool:
        """Check if current turn is semantically similar to the most recent context pair.

        Compares the current turn's combined text against the last user+assistant
        pair in the collected context using embedding cosine similarity.
        Returns ``True`` (pass context) when similarity >= threshold, or when
        embeddings are unavailable (graceful degradation).
        """
        embed_fn = self.get_embed_fn()
        if embed_fn is None:
            return True

        # Compare against the most recent pair in context
        if len(context_pairs) >= 2:
            recent = context_pairs[-2] + " " + context_pairs[-1]
        else:
            recent = " ".join(context_pairs)

        embeddings = embed_fn([current_text[:2000], recent[:2000]])
        sim = cosine_similarity(embeddings[0], embeddings[1])
        threshold = self._config.tag_generator.context_bleed_threshold

        logger.debug("Context bleed gate: sim=%.3f threshold=%.3f", sim, threshold)
        return sim >= threshold
