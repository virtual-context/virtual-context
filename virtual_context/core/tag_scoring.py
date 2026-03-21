"""Shared tag overlap scoring — used by retriever and compaction merge-back."""

from __future__ import annotations


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
