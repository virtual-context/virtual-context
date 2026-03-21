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

    Used by segmenter (grouping consecutive turns) and compaction (merge decisions).
    Signals that can't be computed (missing data) are skipped and weights renormalized.

    Returns a score in [0.0, 1.0] where higher = more related.
    """
    from .math_utils import cosine_similarity

    signals: list[tuple[str, float, float]] = []  # (name, score, weight)

    # Signal 1: Tag overlap (overlap coefficient: shared / min, more lenient than Jaccard)
    # This matches the historical segmenter behavior and keeps threshold calibration intact.
    intersection = tags_a & tags_b
    min_size = min(len(tags_a), len(tags_b))
    overlap = len(intersection) / min_size if min_size > 0 else 0.0
    signals.append(("tag", overlap, tag_weight))

    # Signal 2: Embedding similarity
    if embedding_a is not None and embedding_b is not None:
        sim = cosine_similarity(embedding_a, embedding_b)
        signals.append(("embedding", max(0.0, sim), embedding_weight))
    elif embed_fn is not None and text_a and text_b:
        try:
            embeddings = embed_fn([text_a[:2000], text_b[:2000]])
            sim = cosine_similarity(embeddings[0], embeddings[1])
            signals.append(("embedding", max(0.0, sim), embedding_weight))
        except Exception:
            pass

    # Signal 3: BM25 proxy — keyword overlap between texts
    # Only include when overlap is meaningful (>0.1) — near-zero keyword overlap
    # is noise that dilutes the tag signal on short/ambiguous messages.
    if text_a and text_b:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if words_a and words_b:
            word_overlap = len(words_a & words_b) / len(words_a | words_b)
            if word_overlap > 0.1:
                signals.append(("keyword", word_overlap, bm25_weight))

    if not signals:
        return 0.0

    # Renormalize weights for available signals
    total_weight = sum(w for _, _, w in signals)
    if total_weight == 0:
        return 0.0

    score = sum(s * (w / total_weight) for _, s, w in signals)
    return score
