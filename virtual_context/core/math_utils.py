"""Shared math utilities."""

from __future__ import annotations

try:
    import numpy as np

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors (numpy-accelerated)."""
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

except ImportError:

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors (pure Python fallback)."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
