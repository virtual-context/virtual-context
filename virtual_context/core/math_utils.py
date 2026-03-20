"""Shared math utilities."""

from __future__ import annotations

from collections.abc import Callable

try:
    import numpy as np

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

except ImportError:

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def rank_by_embedding(
    query: str,
    candidates: list,
    texts: list[str],
    embed_fn: Callable,
    threshold: float = 0.5,
) -> list[tuple[float, object]]:
    """Embed *query* alongside *texts* and return candidates above *threshold*.

    *candidates* and *texts* must be parallel lists.  Returns
    ``[(similarity, candidate), ...]`` sorted by descending similarity.
    """
    all_texts = [query] + texts
    vectors = embed_fn(all_texts)
    query_vec = vectors[0]
    scored = []
    for i, item in enumerate(candidates):
        sim = cosine_similarity(query_vec, vectors[i + 1])
        if sim >= threshold:
            scored.append((sim, item))
    scored.sort(key=lambda x: -x[0])
    return scored
