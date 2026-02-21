"""EmbeddingTagGenerator: semantic tag generation using sentence-transformers."""

from __future__ import annotations

import logging
from typing import Callable

from ..types import TagGeneratorConfig, TagResult
from .math_utils import cosine_similarity as _cosine_similarity

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.config = config
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._tag_embeddings: dict[str, list[float]] = {}
        self._tag_vocabulary: dict[str, int] = {}

        if embed_fn:
            self._embed = embed_fn
        else:
            self._embed = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> Callable[[list[str]], list[list[float]]]:
        """Load sentence-transformers model. Raises ImportError if not installed."""
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

    def generate_tags(
        self, text: str, existing_tags: list[str] | None = None,
        context_turns: list[str] | None = None,
    ) -> TagResult:
        """Generate tags by embedding text and comparing against tag embeddings."""
        # Ensure we have tag embeddings for existing tags
        tags_to_embed = []
        if existing_tags:
            for tag in existing_tags:
                if tag not in self._tag_embeddings and tag != "_general":
                    tags_to_embed.append(tag)

        if tags_to_embed:
            embeddings = self._embed(tags_to_embed)
            for tag, emb in zip(tags_to_embed, embeddings):
                self._tag_embeddings[tag] = emb

        if not self._tag_embeddings:
            return TagResult(
                tags=["_general"],
                primary="_general",
                source="fallback",
            )

        # Embed the query text
        text_embedding = self._embed([text[:2000]])[0]  # truncate to avoid OOM

        # Compute cosine similarity against all tag embeddings
        scores: list[tuple[str, float]] = []
        for tag, tag_emb in self._tag_embeddings.items():
            sim = _cosine_similarity(text_embedding, tag_emb)
            if sim >= self.similarity_threshold:
                scores.append((tag, sim))

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

        return TagResult(
            tags=tags,
            primary=primary,
            source="embedding",
        )

    def load_vocabulary(self, tag_counts: dict[str, int]) -> None:
        """Bootstrap vocabulary from existing stored tag counts."""
        self._tag_vocabulary.update(tag_counts)
        # Pre-embed all known tags
        tags_to_embed = [t for t in tag_counts if t not in self._tag_embeddings and t != "_general"]
        if tags_to_embed:
            embeddings = self._embed(tags_to_embed)
            for tag, emb in zip(tags_to_embed, embeddings):
                self._tag_embeddings[tag] = emb


