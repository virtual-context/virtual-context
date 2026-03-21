"""3-signal RRF fusion for retrieval scoring."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .tag_scoring import compute_tag_overlap_score
from .math_utils import cosine_similarity

if TYPE_CHECKING:
    from .store import ContextStore
    from ..types import ScoringConfig

logger = logging.getLogger(__name__)


def rrf_fuse(
    rankings: dict[str, dict[str, int]],
    weights: dict[str, float],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple signal rankings.

    rankings = {signal_name: {tag: rank_position (0-based)}}
    Candidates missing from a signal get penalty rank = k * 2.
    """
    all_tags: set[str] = set()
    for ranking in rankings.values():
        all_tags.update(ranking.keys())

    penalty_rank = k * 2
    scores: dict[str, float] = {}
    for tag in all_tags:
        score = 0.0
        for signal, ranking in rankings.items():
            rank = ranking.get(tag, penalty_rank)
            weight = weights.get(signal, 0.0)
            score += weight * (1.0 / (k + rank + 1))
        scores[tag] = score
    return scores


def compute_idf_candidates(
    query_tags: list[str],
    related_tags: list[str],
    store: ContextStore,
    idf_weights: dict[str, float],
    conversation_id: str | None,
    overfetch_limit: int = 30,
) -> dict[str, float]:
    """Signal 1: IDF tag overlap. Returns {primary_tag: idf_score}."""
    expanded = list(set(query_tags) | set(related_tags))
    if not expanded:
        return {}

    summaries = store.get_summaries_by_tags(
        tags=expanded, min_overlap=1, limit=overfetch_limit,
        conversation_id=conversation_id,
    )

    query_set = set(query_tags)
    related_set = set(related_tags)
    tag_scores: dict[str, float] = {}
    for s in summaries:
        s_tags = set(s.tags)
        primary, _ = compute_tag_overlap_score(query_set, s_tags, idf_weights)
        related, _ = compute_tag_overlap_score(related_set, s_tags, idf_weights)
        score = primary + 0.5 * related
        if s.primary_tag not in tag_scores or score > tag_scores[s.primary_tag]:
            tag_scores[s.primary_tag] = score

    logger.info("Retriever: IDF path: %d candidates for tags=%s", len(tag_scores), expanded)
    return tag_scores


def compute_bm25_candidates(
    query_text: str,
    store: ContextStore,
    conversation_id: str | None,
    limit: int = 20,
) -> dict[str, float]:
    """Signal 2: BM25 on tag summaries + segment summaries. Returns {primary_tag: bm25_score}."""
    tag_scores: dict[str, float] = {}

    # BM25 on tag summaries (direct tag mapping)
    try:
        fts_results = store.search_tag_summaries_fts(
            query_text, limit=limit, conversation_id=conversation_id,
        )
        for tag, score in fts_results:
            if tag not in tag_scores or score > tag_scores[tag]:
                tag_scores[tag] = score
    except Exception:
        pass

    # Extend with segment summary FTS if under limit
    if len(tag_scores) < limit:
        try:
            seg_results = store.search_full_text(
                query_text, limit=limit - len(tag_scores),
                conversation_id=conversation_id,
            )
            for qr in seg_results:
                tag = qr.tag  # primary_tag from QuoteResult
                if tag not in tag_scores:
                    tag_scores[tag] = 1.0  # FTS match = baseline score
        except Exception:
            pass

    logger.info("Retriever: BM25 path: %d candidates from FTS (query='%s')",
                len(tag_scores), query_text[:60])
    return tag_scores


def compute_embedding_candidates(
    query_embedding: list[float] | None,
    store: ContextStore,
    conversation_id: str | None,
    limit: int = 20,
    min_threshold: float = 0.25,
) -> dict[str, float]:
    """Signal 3: Embedding cosine similarity. Returns {primary_tag: cosine_score}."""
    if query_embedding is None:
        return {}

    stored = store.load_tag_summary_embeddings(conversation_id=conversation_id)
    if not stored:
        return {}

    scored: list[tuple[str, float]] = []
    for tag, emb in stored.items():
        sim = cosine_similarity(query_embedding, emb)
        if sim >= min_threshold:
            scored.append((tag, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    result = {tag: sim for tag, sim in scored[:limit]}

    logger.info("Retriever: Embedding path: %d candidates above threshold=%.2f",
                len(result), min_threshold)
    return result


def score_candidates(
    query_tags: list[str],
    related_tags: list[str],
    query_text: str,
    query_embedding: list[float] | None,
    store: ContextStore,
    idf_weights: dict[str, float],
    conversation_id: str | None,
    config: ScoringConfig,
) -> tuple[dict[str, float], dict[str, dict]]:
    """3-signal RRF fusion. Returns ({tag: fused_score}, {tag: signal_breakdown})."""

    # Compute 3 independent candidate sets
    idf_scores = compute_idf_candidates(
        query_tags, related_tags, store, idf_weights, conversation_id,
    )
    bm25_scores = compute_bm25_candidates(
        query_text, store, conversation_id, limit=config.bm25_limit,
    )
    embed_scores = compute_embedding_candidates(
        query_embedding, store, conversation_id,
        limit=config.embedding_limit,
        min_threshold=config.embedding_min_threshold,
    )

    # Union all candidates
    all_tags = set(idf_scores) | set(bm25_scores) | set(embed_scores)
    logger.info(
        "Retriever: Union: %d unique candidates from 3 signals (idf=%d, bm25=%d, embed=%d)",
        len(all_tags), len(idf_scores), len(bm25_scores), len(embed_scores),
    )

    if not all_tags:
        return {}, {}

    # Build per-signal rankings (0-based rank by score descending)
    def _to_ranking(scores: dict[str, float]) -> dict[str, int]:
        sorted_tags = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        return {tag: rank for rank, tag in enumerate(sorted_tags)}

    rankings = {
        "idf": _to_ranking(idf_scores),
        "bm25": _to_ranking(bm25_scores),
        "embedding": _to_ranking(embed_scores),
    }
    weights = {
        "idf": config.idf_weight,
        "bm25": config.bm25_weight,
        "embedding": config.embedding_weight,
    }

    # RRF fusion
    fused = rrf_fuse(rankings, weights, k=config.rrf_k)

    # Build breakdowns for logging
    breakdowns: dict[str, dict] = {}
    penalty = config.rrf_k * 2
    for tag in all_tags:
        bd = {
            "idf_rank": rankings["idf"].get(tag, penalty),
            "bm25_rank": rankings["bm25"].get(tag, penalty),
            "embed_rank": rankings["embedding"].get(tag, penalty),
            "idf_score": idf_scores.get(tag, 0.0),
            "bm25_score": bm25_scores.get(tag, 0.0),
            "embed_score": embed_scores.get(tag, 0.0),
            "fused": fused[tag],
        }
        breakdowns[tag] = bd
        logger.info(
            "Retriever: RRF '%s': idf_rank=%d bm25_rank=%d embed_rank=%d -> fused=%.4f",
            tag, bd["idf_rank"], bd["bm25_rank"], bd["embed_rank"], bd["fused"],
        )

    # Log top results
    top = sorted(fused.keys(), key=lambda t: fused[t], reverse=True)[:10]
    logger.info("Retriever: Top %d by RRF: %s", len(top),
                ", ".join(f"{t}={fused[t]:.4f}" for t in top))

    return fused, breakdowns
