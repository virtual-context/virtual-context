"""Context-augmented embedding retrieval signal (guard) — BUG-042.

Covers the ``retrieval.scoring.embedding_context_turns`` /
``embedding_context_guard`` axes: the per-tag MAX guard math, the N=0
byte-identical legacy path, the crater-repair regression, config plumbing,
and the retriever's concat-embedding construction.

The guard math mirrors ``scripts/embedding_ladder.py::guard_score`` — the
reference oracle here replicates that function's logic (unit-normalize both
query vectors, per-tag elementwise max over unit-normalized doc rows, rank by
descending similarity) so the production path is checked against it directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_context.config import load_config
from virtual_context.core.retrieval_scoring import compute_embedding_candidates
from virtual_context.core.retriever import ContextRetriever
from virtual_context.types import RetrieverConfig, ScoringConfig

pytestmark = pytest.mark.regression("BUG-042")


# --------------------------------------------------------------------------
# Reference oracle — replicated from scripts/embedding_ladder.py:guard_score
# --------------------------------------------------------------------------
def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def _norm_matrix(vectors):
    m = np.asarray(vectors, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def guard_order(doc_vecs, tags, bare_vec, ctx_vec):
    """Per-tag max(sim(bare), sim(concat)) ranking — the guard_score() key."""
    doc_matrix = _norm_matrix([doc_vecs[t] for t in tags])
    b, c = _unit(bare_vec), _unit(ctx_vec)
    sims = np.maximum(doc_matrix @ b, doc_matrix @ c)
    order = np.argsort(-sims)
    return [tags[i] for i in order]


def _rank_of(result_dict, gold_tags):
    """1-indexed best rank of any gold tag in a candidate dict, or None."""
    gold = set(gold_tags)
    for pos, tag in enumerate(result_dict.keys()):
        if tag in gold:
            return pos + 1
    return None


# --------------------------------------------------------------------------
# 1. Guard math parity vs the ladder reference oracle
# --------------------------------------------------------------------------
class TestGuardMathParity:
    def test_guard_ranking_matches_ladder_oracle(self):
        rng = np.random.RandomState(0)
        dim = 16
        tags = [f"tag-{i}" for i in range(12)]
        doc_vecs = {t: rng.randn(dim).astype(np.float32).tolist() for t in tags}
        bare = rng.randn(dim).astype(np.float32).tolist()
        ctx = rng.randn(dim).astype(np.float32).tolist()

        oracle = guard_order(doc_vecs, tags, bare, ctx)

        result = compute_embedding_candidates(
            query_embedding=bare,
            store=None,
            conversation_id=None,
            limit=len(tags),
            min_threshold=-2.0,  # keep all candidates so the full order is comparable
            stored_embeddings=doc_vecs,
            query_embedding_context=ctx,
            guard=True,
        )
        assert list(result.keys()) == oracle

    def test_guard_never_scores_below_bare(self):
        """Per-tag guard similarity is >= the bare similarity for every tag."""
        rng = np.random.RandomState(7)
        dim = 12
        tags = [f"t{i}" for i in range(8)]
        doc_vecs = {t: rng.randn(dim).astype(np.float32).tolist() for t in tags}
        bare = rng.randn(dim).astype(np.float32).tolist()
        ctx = rng.randn(dim).astype(np.float32).tolist()

        bare_only = compute_embedding_candidates(
            bare, None, None, limit=len(tags), min_threshold=-2.0,
            stored_embeddings=doc_vecs,
        )
        guarded = compute_embedding_candidates(
            bare, None, None, limit=len(tags), min_threshold=-2.0,
            stored_embeddings=doc_vecs, query_embedding_context=ctx, guard=True,
        )
        for tag in tags:
            assert guarded[tag] >= bare_only[tag] - 1e-6


# --------------------------------------------------------------------------
# 2. N=0 / no-context path is byte-identical to legacy
# --------------------------------------------------------------------------
class TestLegacyByteIdentical:
    def _fixture(self):
        rng = np.random.RandomState(3)
        dim = 10
        tags = [f"tag{i}" for i in range(9)]
        doc_vecs = {t: rng.randn(dim).astype(np.float32).tolist() for t in tags}
        bare = rng.randn(dim).astype(np.float32).tolist()
        return doc_vecs, bare

    def test_no_context_kwargs_matches_legacy(self):
        doc_vecs, bare = self._fixture()
        legacy = compute_embedding_candidates(
            bare, None, None, limit=20, min_threshold=0.0,
            stored_embeddings=doc_vecs,
        )
        # New signature, context vector absent — must be identical.
        new = compute_embedding_candidates(
            bare, None, None, limit=20, min_threshold=0.0,
            stored_embeddings=doc_vecs, query_embedding_context=None, guard=True,
        )
        assert list(new.items()) == list(legacy.items())

    def test_guard_flag_irrelevant_without_context(self):
        doc_vecs, bare = self._fixture()
        g_true = compute_embedding_candidates(
            bare, None, None, limit=20, min_threshold=0.0,
            stored_embeddings=doc_vecs, query_embedding_context=None, guard=True,
        )
        g_false = compute_embedding_candidates(
            bare, None, None, limit=20, min_threshold=0.0,
            stored_embeddings=doc_vecs, query_embedding_context=None, guard=False,
        )
        assert list(g_true.items()) == list(g_false.items())

    def test_scoring_defaults_are_legacy(self):
        sc = ScoringConfig()
        assert sc.embedding_context_turns == 0
        assert sc.embedding_context_guard is True


# --------------------------------------------------------------------------
# 3. Crater regression — irrelevant context cannot demote a tag (perfume-gift-2)
# --------------------------------------------------------------------------
class TestCraterRegression:
    """Reproduce the perfume-gift-2 shape: bare rank 1 -> plain-concat craters
    the gold tag behind a cluster of irrelevant-context noise tags -> the guard
    recovers it near its bare rank. Built with two orthonormal query directions
    (u = bare, v = context) so each tag's (bare_sim, context_sim) is dialed
    exactly."""

    def _build(self):
        n_low, n_high = 14, 6
        n_noise = n_low + n_high
        dim = 2 + 1 + n_noise  # e0=bare dir, e1=ctx dir, +unique residual per tag
        u = np.zeros(dim, dtype=np.float32); u[0] = 1.0
        v = np.zeros(dim, dtype=np.float32); v[1] = 1.0

        def make(sb, sc, residual_idx):
            w = np.zeros(dim, dtype=np.float32)
            w[2 + residual_idx] = 1.0
            r = float(np.sqrt(max(0.0, 1.0 - sb * sb - sc * sc)))
            vec = sb * u + sc * v + r * w
            return vec.tolist()

        doc_vecs = {}
        # Gold: strong bare alignment, weak context alignment.
        doc_vecs["gold"] = make(0.98, 0.10, 0)
        # Low noise: context sim between the gold's context sim and the gold's
        # bare sim — beats gold under plain concat, loses to it under guard.
        lows = np.linspace(0.11, 0.90, n_low)
        for i, sc in enumerate(lows):
            doc_vecs[f"noise-low-{i}"] = make(0.0, float(sc), 1 + i)
        # High noise: context sim above the gold's bare sim — the few tags that
        # legitimately outrank gold even under the guard.
        highs = np.linspace(0.985, 0.999, n_high)
        for i, sc in enumerate(highs):
            doc_vecs[f"noise-high-{i}"] = make(0.0, float(sc), 1 + n_low + i)

        bare = u.tolist()   # bare query
        ctx = v.tolist()    # irrelevant-context-augmented query
        return doc_vecs, bare, ctx

    def test_guard_repairs_irrelevant_context_crater(self):
        doc_vecs, bare, ctx = self._build()
        kw = dict(store=None, conversation_id=None, limit=100, min_threshold=-2.0,
                  stored_embeddings=doc_vecs)

        bare_res = compute_embedding_candidates(bare, **kw)
        concat_res = compute_embedding_candidates(
            bare, query_embedding_context=ctx, guard=False, **kw)
        guard_res = compute_embedding_candidates(
            bare, query_embedding_context=ctx, guard=True, **kw)

        bare_rank = _rank_of(bare_res, ["gold"])
        concat_rank = _rank_of(concat_res, ["gold"])
        guard_rank = _rank_of(guard_res, ["gold"])

        assert bare_rank == 1
        # Plain concat with irrelevant context craters the gold tag.
        assert concat_rank is not None and concat_rank >= 15
        # The guard recovers it near the bare rank and never worse than concat.
        assert guard_rank is not None
        assert guard_rank < concat_rank
        assert guard_rank <= bare_rank + 6


# --------------------------------------------------------------------------
# 4. Config plumbing (dataclass + YAML parse)
# --------------------------------------------------------------------------
class TestConfigPlumbing:
    def test_yaml_parses_context_axes(self):
        cfg = load_config(config_dict={
            "retrieval": {
                "scoring": {
                    "embedding_context_turns": 2,
                    "embedding_context_guard": False,
                }
            }
        })
        assert cfg.retriever.scoring.embedding_context_turns == 2
        assert cfg.retriever.scoring.embedding_context_guard is False

    def test_yaml_defaults_when_absent(self):
        cfg = load_config(config_dict={"retrieval": {"scoring": {}}})
        assert cfg.retriever.scoring.embedding_context_turns == 0
        assert cfg.retriever.scoring.embedding_context_guard is True


# --------------------------------------------------------------------------
# 5. Retriever concat-embedding construction
# --------------------------------------------------------------------------
class _StubTagger:
    def __init__(self):
        self.calls = []

    def embed_text(self, text):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


def _retriever(scoring, inbound_tagger):
    return ContextRetriever(
        tag_generator=None,
        store=None,
        config=RetrieverConfig(scoring=scoring),
        inbound_tagger=inbound_tagger,
    )


class TestRetrieverContextEmbedding:
    CONTEXT = ["u1", "a1", "u2", "a2"]

    def test_builds_concat_embedding_in_guard_mode(self):
        stub = _StubTagger()
        r = _retriever(ScoringConfig(embedding_context_turns=2), stub)
        emb, mode, ms = r._build_context_query_embedding("what should I get her", self.CONTEXT)
        assert emb == [0.1, 0.2, 0.3]
        assert mode == "guard"
        assert ms is not None and ms >= 0.0
        # Concat uses the last N=2 context turns + the current message.
        assert len(stub.calls) == 1
        assert "u2" in stub.calls[0] and "a2" in stub.calls[0]
        assert "what should I get her" in stub.calls[0]
        assert "u1" not in stub.calls[0]

    def test_disabled_when_turns_zero(self):
        stub = _StubTagger()
        r = _retriever(ScoringConfig(embedding_context_turns=0), stub)
        emb, mode, ms = r._build_context_query_embedding("msg", self.CONTEXT)
        assert emb is None and mode == "bare" and ms is None
        assert stub.calls == []

    def test_concat_mode_when_guard_disabled(self):
        stub = _StubTagger()
        r = _retriever(
            ScoringConfig(embedding_context_turns=1, embedding_context_guard=False), stub)
        emb, mode, _ = r._build_context_query_embedding("msg", self.CONTEXT)
        assert emb == [0.1, 0.2, 0.3]
        assert mode == "concat"

    def test_no_context_turns_falls_back_to_bare(self):
        stub = _StubTagger()
        r = _retriever(ScoringConfig(embedding_context_turns=2), stub)
        emb, mode, ms = r._build_context_query_embedding("msg", None)
        assert emb is None and mode == "bare" and ms is None
        assert stub.calls == []

    def test_missing_embed_text_falls_back_to_bare(self):
        r = _retriever(ScoringConfig(embedding_context_turns=2), object())
        emb, mode, ms = r._build_context_query_embedding("msg", self.CONTEXT)
        assert emb is None and mode == "bare" and ms is None
