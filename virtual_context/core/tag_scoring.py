"""Shared scoring — tag overlap, embedding similarity, pairwise relatedness.

Used by retriever (RRF), segmenter (grouping), and compaction (merge decisions).
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def compute_tag_overlap_score(
    query_tags: set[str],
    candidate_tags: set[str],
    idf_weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Score overlap between two tag sets.

    Returns (raw_score, overlap_ratio).
    - raw_score: sum of IDF weights for overlapping tags (or count if no weights)
    - overlap_ratio: Jaccard-like ratio (0.0-1.0) = |intersection| / |union|
    """
    if not query_tags or not candidate_tags:
        return 0.0, 0.0

    intersection = query_tags & candidate_tags
    union = query_tags | candidate_tags

    overlap_ratio = len(intersection) / len(union) if union else 0.0

    if idf_weights:
        raw_score = sum(idf_weights.get(t, 1.0) for t in intersection)
    else:
        raw_score = float(len(intersection))

    return raw_score, overlap_ratio


# Default signal weights for pairwise relatedness
TAG_WEIGHT = 0.5
EMBEDDING_WEIGHT = 0.3
BM25_WEIGHT = 0.2


def compute_relatedness(
    tags_a: set[str],
    tags_b: set[str],
    text_a: str = "",
    text_b: str = "",
    embedding_a: list[float] | None = None,
    embedding_b: list[float] | None = None,
    idf_weights: dict[str, float] | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    tag_weight: float = TAG_WEIGHT,
    embedding_weight: float = EMBEDDING_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> float:
    """Compute pairwise relatedness between two items using available signals.

    Uses MAX-of-signals: each signal can only LIFT the score, never drag it down.
    Tag overlap is the primary signal. Embedding and keyword are rescue signals —
    they help when tags miss, but can't reduce a strong tag match.

    Returns a score in [0.0, 1.0] where higher = more related.
    """
    from .math_utils import cosine_similarity

    # Signal 1: Tag overlap (overlap coefficient: shared / min)
    intersection = tags_a & tags_b
    min_size = min(len(tags_a), len(tags_b))
    tag_score = len(intersection) / min_size if min_size > 0 else 0.0

    # Signal 2: Embedding similarity (slight discount — rescue signal, not primary)
    embed_score = 0.0
    if embedding_a is not None and embedding_b is not None:
        embed_score = max(0.0, cosine_similarity(embedding_a, embedding_b)) * 0.9
    elif embed_fn is not None and text_a and text_b:
        try:
            embeddings = embed_fn([text_a[:2000], text_b[:2000]])
            embed_score = max(0.0, cosine_similarity(embeddings[0], embeddings[1])) * 0.9
        except Exception:
            pass

    # Signal 3: Keyword overlap (more discount — weakest rescue signal)
    keyword_score = 0.0
    if text_a and text_b:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if words_a and words_b:
            keyword_score = len(words_a & words_b) / len(words_a | words_b) * 0.8

    # MAX-of-signals: each signal can only lift, never dilute
    score = max(tag_score, embed_score, keyword_score)
    return score
