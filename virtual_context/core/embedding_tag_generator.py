"""EmbeddingTagGenerator: semantic tag generation using sentence-transformers."""

from __future__ import annotations

import logging
import math
import time
from typing import Callable

from ..types import TagGeneratorConfig, TagResult
from .math_utils import cosine_similarity as _cosine_similarity

logger = logging.getLogger(__name__)

_EMBED_TAG_BREAKDOWN_LOG_THRESHOLD_MS = 100.0
_EMBED_TAG_BREAKDOWN_MAX_STAGES = 6

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional in test/dev envs
    np = None


class EmbeddingTagGenerator:
    """Generate semantic tags by comparing text embeddings against tag embeddings.

    Uses sentence-transformers for embedding. Falls back gracefully if not installed.
    Implements the TagGenerator protocol.
    """

    def __init__(
        self,
        config: TagGeneratorConfig,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.3,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        load_cached_embeddings: Callable[[str, list[str]], dict[str, list[float]]] | None = None,
        save_cached_embeddings: Callable[[str, dict[str, list[float]]], None] | None = None,
    ) -> None:
        self.config = config
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._tag_embeddings: dict[str, list[float]] = {}
        self._tag_vocabulary: dict[str, int] = {}
        self._load_cached_embeddings = load_cached_embeddings
        self._save_cached_embeddings = save_cached_embeddings

        if embed_fn:
            self._embed = embed_fn
        else:
            self._embed = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> Callable[[list[str]], list[list[float]]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install virtual-context[embeddings]"
            )
        model = SentenceTransformer(model_name)

        def embed(texts: list[str]) -> list[list[float]]:
            return model.encode(texts, convert_to_numpy=True).tolist()

        return embed

    @staticmethod
    def _dedupe_tags(tags: list[str] | None) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for tag in tags or []:
            if not tag or tag == "_general" or tag in seen:
                continue
            seen.add(tag)
            unique.append(tag)
        return unique

    @staticmethod
    def _note_breakdown(breakdown: dict[str, float], stage: str, started_at: float) -> None:
        elapsed = round((time.monotonic() - started_at) * 1000, 1)
        breakdown[stage] = round(breakdown.get(stage, 0.0) + elapsed, 1)

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        if not embedding:
            return []
        if np is not None:
            arr = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm == 0.0:
                return arr.tolist()
            return (arr / norm).tolist()
        norm = math.sqrt(sum(value * value for value in embedding))
        if norm == 0.0:
            return list(embedding)
        return [value / norm for value in embedding]

    def _ensure_tag_embeddings(
        self,
        tags: list[str] | None,
        *,
        breakdown: dict[str, float] | None = None,
    ) -> tuple[int, int, int]:
        unique_tags = self._dedupe_tags(tags)
        if not unique_tags:
            return 0, 0, 0

        missing = [tag for tag in unique_tags if tag not in self._tag_embeddings]
        local_hits = len(unique_tags) - len(missing)
        shared_hits = 0

        if missing and self._load_cached_embeddings is not None:
            _load_stage = time.monotonic()
            cached = self._load_cached_embeddings(self.model_name, missing) or {}
            if breakdown is not None:
                self._note_breakdown(breakdown, "shared_cache_load", _load_stage)
            for tag, embedding in cached.items():
                if embedding is not None:
                    self._tag_embeddings[tag] = self._normalize_embedding(list(embedding))
            shared_hits = sum(1 for tag in missing if tag in self._tag_embeddings)
            missing = [tag for tag in missing if tag not in self._tag_embeddings]

        embedded_missing = 0
        if missing:
            _embed_stage = time.monotonic()
            embeddings = self._embed(missing)
            if breakdown is not None:
                self._note_breakdown(breakdown, "embed_missing_tags", _embed_stage)
            saved: dict[str, list[float]] = {}
            for tag, embedding in zip(missing, embeddings):
                normalized = self._normalize_embedding(list(embedding))
                self._tag_embeddings[tag] = normalized
                saved[tag] = normalized
            embedded_missing = len(saved)
            if saved and self._save_cached_embeddings is not None:
                _save_stage = time.monotonic()
                self._save_cached_embeddings(self.model_name, saved)
                if breakdown is not None:
                    self._note_breakdown(breakdown, "shared_cache_save", _save_stage)

        return local_hits, shared_hits, embedded_missing

    def generate_tags(
        self, text: str, existing_tags: list[str] | None = None,
        context_turns: list[str] | None = None,
    ) -> TagResult:
        del context_turns
        _started = time.monotonic()
        _breakdown: dict[str, float] = {}
        unique_tags = self._dedupe_tags(existing_tags)
        local_hits, shared_hits, embedded_missing = self._ensure_tag_embeddings(
            unique_tags,
            breakdown=_breakdown,
        )

        if not self._tag_embeddings:
            return TagResult(
                tags=["_general"],
                primary="_general",
                source="fallback",
            )

        # Embed the query text
        _query_stage = time.monotonic()
        text_embedding = self._embed([text[:2000]])[0]  # truncate to avoid OOM
        self._note_breakdown(_breakdown, "embed_query", _query_stage)

        # Score only the vocabulary in scope for this request. The generator's
        # cache can outlive a single request, so scoring the whole cache would
        # waste time on stale tags and could bias matching.
        _similarity_stage = time.monotonic()
        scores: list[tuple[str, float]] = []
        candidate_tags = [tag for tag in unique_tags if tag in self._tag_embeddings]
        if not candidate_tags:
            candidate_tags = list(self._tag_embeddings.keys())
        if np is not None and candidate_tags:
            query_vec = np.asarray(text_embedding, dtype=np.float32)
            query_norm = float(np.linalg.norm(query_vec))
            if query_norm > 0.0:
                query_vec = query_vec / query_norm
                embedding_matrix = np.asarray(
                    [self._tag_embeddings[tag] for tag in candidate_tags],
                    dtype=np.float32,
                )
                similarities = embedding_matrix @ query_vec
                matching = np.flatnonzero(similarities >= self.similarity_threshold)
                if matching.size:
                    ordered = matching[np.argsort(similarities[matching])[::-1]]
                    scores = [
                        (candidate_tags[int(idx)], float(similarities[int(idx)]))
                        for idx in ordered
                    ]
        else:
            normalized_query = self._normalize_embedding(list(text_embedding))
            for tag in candidate_tags:
                tag_emb = self._tag_embeddings[tag]
                sim = _cosine_similarity(normalized_query, tag_emb)
                if sim >= self.similarity_threshold:
                    scores.append((tag, sim))
        self._note_breakdown(_breakdown, "similarity", _similarity_stage)

        if not scores:
            return TagResult(
                tags=["_general"],
                primary="_general",
                source="fallback",
            )

        # Sort by similarity, take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:self.config.max_tags]

        tags = [t for t, _ in top]
        primary = tags[0]

        # Update vocabulary
        for tag in tags:
            self._tag_vocabulary[tag] = self._tag_vocabulary.get(tag, 0) + 1

        total_ms = round((time.monotonic() - _started) * 1000, 1)
        if total_ms >= _EMBED_TAG_BREAKDOWN_LOG_THRESHOLD_MS or embedded_missing > 0:
            stage_bits = [
                f"{stage}={ms:.1f}ms"
                for stage, ms in sorted(_breakdown.items(), key=lambda item: item[1], reverse=True)
                [:_EMBED_TAG_BREAKDOWN_MAX_STAGES]
                if ms > 0
            ]
            logger.info(
                "EMBED_TAG_BREAKDOWN model=%s total=%sms vocab=%d scored=%d local_hits=%d shared_hits=%d embedded_missing=%d matched=%d %s",
                self.model_name,
                total_ms,
                len(unique_tags),
                len(candidate_tags),
                local_hits,
                shared_hits,
                embedded_missing,
                len(scores),
                " ".join(stage_bits) if stage_bits else "no-stages",
            )

        return TagResult(
            tags=tags,
            primary=primary,
            source="embedding",
            query_embedding=text_embedding,
        )

    def load_vocabulary(self, tag_counts: dict[str, int]) -> None:
        self._tag_vocabulary.update(tag_counts)
        self._ensure_tag_embeddings(list(tag_counts))
