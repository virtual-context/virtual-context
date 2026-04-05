"""3-signal RRF fusion for retrieval scoring."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .tag_scoring import compute_tag_overlap_score
from .math_utils import cosine_similarity

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional in test/dev envs
    np = None

if TYPE_CHECKING:
    from .store import ContextStore
    from ..types import ScoringConfig

logger = logging.getLogger(__name__)

_RETRIEVAL_SCORE_BREAKDOWN_LOG_THRESHOLD_MS = 300.0


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
    stored_embeddings: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    """Signal 3: Embedding cosine similarity. Returns {primary_tag: cosine_score}."""
    if query_embedding is None:
        return {}

    stored = stored_embeddings
    if stored is None:
        stored = store.load_tag_summary_embeddings(conversation_id=conversation_id)
    if not stored:
        return {}

    scored: list[tuple[str, float]] = []
    if np is not None:
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm > 0.0:
            query_vec = query_vec / query_norm
            candidate_tags: list[str] = []
            embedding_rows: list[list[float]] = []
            for tag, emb in stored.items():
                if not emb:
                    continue
                candidate_tags.append(tag)
                embedding_rows.append(emb)
            if candidate_tags:
                embedding_matrix = np.asarray(embedding_rows, dtype=np.float32)
                row_norms = np.linalg.norm(embedding_matrix, axis=1)
                valid = row_norms > 0.0
                if np.any(valid):
                    embedding_matrix = embedding_matrix[valid] / row_norms[valid][:, None]
                    filtered_tags = [
                        candidate_tags[idx] for idx, ok in enumerate(valid.tolist()) if ok
                    ]
                    similarities = embedding_matrix @ query_vec
                    matching = np.flatnonzero(similarities >= min_threshold)
                    if matching.size:
                        ordered = matching[np.argsort(similarities[matching])[::-1]]
                        scored = [
                            (filtered_tags[int(idx)], float(similarities[int(idx)]))
                            for idx in ordered
                        ]
    else:
        for tag, emb in stored.items():
            sim = cosine_similarity(query_embedding, emb)
            if sim >= min_threshold:
                scored.append((tag, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    result = {tag: sim for tag, sim in scored[:limit]}

    logger.info("Retriever: Embedding path: %d candidates above threshold=%.2f",
                len(result), min_threshold)
    return result


def apply_gravity_dampening(
    embed_scores: dict[str, float],
    bm25_scores: dict[str, float],
    threshold: float = 0.5,
    factor: float = 0.5,
) -> None:
    """Pre-RRF: halve embedding scores that have zero BM25 support (in-place)."""
    for tag in list(embed_scores.keys()):
        if bm25_scores.get(tag, 0) == 0 and embed_scores[tag] > threshold:
            old = embed_scores[tag]
            embed_scores[tag] = old * factor
            logger.info("Retriever: GRAVITY '%s': embedding %.3f->%.3f (bm25=0, no keyword support)",
                        tag, old, embed_scores[tag])


def apply_hub_dampening(
    fused_scores: dict[str, float],
    tag_stats: dict[str, int],
    query_tags: set[str],
    penalty_strength: float = 0.6,
    min_score_fraction: float = 0.2,
) -> None:
    """Post-RRF: penalize high-segment-count tags (in-place)."""
    if not tag_stats:
        return
    counts = sorted(tag_stats.values())
    if len(counts) < 2:
        return
    p90_idx = int(len(counts) * 0.9)
    p90 = counts[p90_idx]
    max_count = counts[-1]
    if p90 >= max_count:
        return

    above_p90 = sum(1 for t in fused_scores if tag_stats.get(t, 0) > p90)
    if above_p90:
        logger.info("Retriever: HUB dampening stats: p90=%d, max=%d, %d tags above p90",
                    p90, max_count, above_p90)

    for tag in list(fused_scores.keys()):
        seg_count = tag_stats.get(tag, 0)
        if seg_count <= p90:
            continue
        if tag in query_tags:
            logger.info("Retriever: HUB exempt '%s': segments=%d above p90 but in query tags", tag, seg_count)
            continue
        old = fused_scores[tag]
        penalty = 1.0 - penalty_strength * (seg_count - p90) / (max_count - p90)
        fused_scores[tag] = old * max(min_score_fraction, penalty)
        logger.info("Retriever: HUB '%s': score %.4f->%.4f (segments=%d, p90=%d, penalty=%.2f)",
                    tag, old, fused_scores[tag], seg_count, p90, penalty)


def apply_resolution_boost(
    fused_scores: dict[str, float],
    actionable_tags: set[str],
    boost: float = 1.15,
) -> None:
    """Post-RRF: boost fact-bearing tags (in-place)."""
    boosted = 0
    for tag in list(fused_scores.keys()):
        if tag in actionable_tags:
            old = fused_scores[tag]
            fused_scores[tag] = old * boost
            logger.info("Retriever: RESOLUTION boost '%s': score %.4f->%.4f (has actionable facts)",
                        tag, old, fused_scores[tag])
            boosted += 1
    if boosted:
        logger.info("Retriever: RESOLUTION: %d/%d candidates boosted", boosted, len(fused_scores))


def score_candidates(
    query_tags: list[str],
    related_tags: list[str],
    query_text: str,
    query_embedding: list[float] | None,
    store: "ContextStore",
    idf_weights: dict[str, float],
    conversation_id: str | None,
    config: "ScoringConfig",
    tag_stats: dict[str, int] | None = None,
    stored_embeddings: dict[str, list[float]] | None = None,
) -> tuple[dict[str, float], dict[str, dict]]:
    """3-signal RRF fusion. Returns ({tag: fused_score}, {tag: signal_breakdown})."""
    _started = time.monotonic()
    _breakdown: dict[str, float] = {}

    def _note(stage: str, started_at: float) -> None:
        _breakdown[stage] = round((time.monotonic() - started_at) * 1000, 1)

    # Compute 3 independent candidate sets
    _idf_stage = time.monotonic()
    idf_scores = compute_idf_candidates(
        query_tags, related_tags, store, idf_weights, conversation_id,
    )
    _note("idf_candidates", _idf_stage)
    _bm25_stage = time.monotonic()
    bm25_scores = compute_bm25_candidates(
        query_text, store, conversation_id, limit=config.bm25_limit,
    )
    _note("bm25_candidates", _bm25_stage)
    _embed_stage = time.monotonic()
    embed_scores = compute_embedding_candidates(
        query_embedding, store, conversation_id,
        limit=config.embedding_limit,
        min_threshold=config.embedding_min_threshold,
        stored_embeddings=stored_embeddings,
    )
    _note("embedding_candidates", _embed_stage)

    # --- Phase 2: Dampening (pre-RRF) ---
    dampening = config.dampening

    # Gravity (pre-RRF): halve embedding scores with zero BM25 support
    if dampening.gravity_enabled and embed_scores:
        _gravity_stage = time.monotonic()
        apply_gravity_dampening(
            embed_scores, bm25_scores,
            threshold=dampening.gravity_threshold,
            factor=dampening.gravity_factor,
        )
        _note("gravity_dampening", _gravity_stage)

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
    _rrf_stage = time.monotonic()
    fused = rrf_fuse(rankings, weights, k=config.rrf_k)
    _note("rrf_fuse", _rrf_stage)

    # --- Phase 2: Dampening (post-RRF) ---

    # Hub dampening (post-RRF)
    if dampening.hub_enabled and tag_stats:
        _hub_stage = time.monotonic()
        apply_hub_dampening(
            fused, tag_stats, set(query_tags),
            penalty_strength=dampening.hub_penalty_strength,
            min_score_fraction=dampening.hub_min_score,
        )
        _note("hub_dampening", _hub_stage)

    # Resolution boost (post-RRF)
    if dampening.resolution_enabled:
        _resolution_stage = time.monotonic()
        try:
            actionable = store.get_actionable_fact_tags(
                list(all_tags), conversation_id=conversation_id,
            )
            if actionable:
                apply_resolution_boost(fused, actionable, boost=dampening.resolution_boost)
        except Exception:
            pass
        _note("resolution_boost", _resolution_stage)

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

    total_ms = round((time.monotonic() - _started) * 1000, 1)
    if total_ms >= _RETRIEVAL_SCORE_BREAKDOWN_LOG_THRESHOLD_MS:
        stages = " ".join(
            f"{stage}={ms:.1f}ms"
            for stage, ms in sorted(_breakdown.items(), key=lambda item: item[1], reverse=True)
            if ms > 0
        )
        logger.info(
            "RETRIEVAL_SCORE_BREAKDOWN total=%sms candidates=%d idf=%d bm25=%d embed=%d %s",
            total_ms,
            len(all_tags),
            len(idf_scores),
            len(bm25_scores),
            len(embed_scores),
            stages or "no-stages",
        )

    return fused, breakdowns
