"""Embedding reserved seats in RRF fusion — BUG-043.

Covers the ``retrieval.scoring.embedding_reserved_seats`` axis: after fusion,
dampening, and boosting, the top-N embedding-signal candidates outside the
fused top-K are forced into it by displacing the lowest-ranked entries.

The oracle is the rule itself (RRF's exp_results.json replica was unavailable
at implementation time): ``reference_seated_topk`` below reproduces "take the
top-N embedding candidates not already in the top-K and insert them by
displacing the lowest-ranked entries of the top-K, K = max_results", and the
production path is checked against it — on a synthetic fixture and on a real
prod fused/embedding dump (``tests/fixtures/reserved_seats_diag_ctx.json``, the
context-augmented "Sania's birthday" query).
"""

from __future__ import annotations

import json
import os

import pytest

from virtual_context.config import load_config
from virtual_context.core.retrieval_scoring import apply_embedding_reserved_seats
from virtual_context.types import ScoringConfig

pytestmark = pytest.mark.regression("BUG-043")

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "reserved_seats_diag_ctx.json")


# --------------------------------------------------------------------------
# Reference oracle — the seating rule stated in the brief
# --------------------------------------------------------------------------
def reference_seated_topk(fused, embedding_ranked, n, k):
    """Final top-K tag list after seating, per the rule."""
    ranked = sorted(fused.keys(), key=lambda t: fused[t], reverse=True)
    if len(ranked) <= k:
        return ranked[:k]
    top = ranked[:k]
    top_set = set(top)
    reserved = [t for t in embedding_ranked if t in fused and t not in top_set][:n]
    m = min(len(reserved), k)
    reserved = reserved[:m]
    surviving = top[: k - m]
    return surviving + reserved


def production_topk(fused, embedding_ranked, n, k):
    """Run the production seating on a copy, return the resulting top-K order."""
    scores = dict(fused)
    apply_embedding_reserved_seats(scores, embedding_ranked, n, k)
    return sorted(scores.keys(), key=lambda t: scores[t], reverse=True)[:k]


# --------------------------------------------------------------------------
# 1. Seating math parity (synthetic + real prod dump)
# --------------------------------------------------------------------------
class TestReservedSeatMath:
    def _synthetic(self):
        # 15 tags; embedding surfaces e0,e1,e2 that sit low in the fused order.
        fused = {f"f{i}": 1.0 - i * 0.01 for i in range(12)}  # f0..f11 descending
        fused.update({"e0": 0.30, "e1": 0.29, "e2": 0.28})    # buried embedding-only
        embedding_ranked = ["e0", "e1", "e2", "f0", "f1"]     # embedding-signal order
        return fused, embedding_ranked

    def test_matches_reference_synthetic(self):
        fused, emb = self._synthetic()
        for n in (0, 1, 2, 3):
            k = 10
            assert production_topk(fused, emb, n, k) == reference_seated_topk(fused, emb, n, k), n

    def test_seated_displace_lowest_of_topk(self):
        fused, emb = self._synthetic()
        top = production_topk(fused, emb, 3, 10)
        # e0,e1,e2 seated; they displaced the lowest 3 of the original top-10
        # (f7,f8,f9). f0..f6 survive.
        assert set(["e0", "e1", "e2"]).issubset(set(top))
        assert top[:7] == [f"f{i}" for i in range(7)]
        assert top[7:] == ["e0", "e1", "e2"]
        assert "f7" not in top and "f8" not in top and "f9" not in top

    def test_matches_reference_real_prod_dump(self):
        # Anonymized real prod fused/embedding dump (tag names opaqued, scores
        # preserved verbatim) for a context-augmented analog query where the top
        # embedding candidates all sit outside the fused top-10.
        data = json.load(open(_FIXTURE))
        es, fused = data["embedding_scores"], dict(data["fused_scores"])
        emb = sorted(es.keys(), key=lambda t: es[t], reverse=True)

        orig_top10 = sorted(fused.keys(), key=lambda t: fused[t], reverse=True)[:10]
        to_seat = [t for t in emb if t not in set(orig_top10)][:3]

        prod = production_topk(fused, emb, 3, 10)
        ref = reference_seated_topk(fused, emb, 3, 10)
        assert prod == ref
        # The three buried embedding candidates are seated; the three
        # lowest-fused originals of the top-10 are displaced.
        for tag in to_seat:
            assert tag in prod
        for tag in orig_top10[7:]:
            assert tag not in prod
        assert prod[:7] == orig_top10[:7]  # top-7 survivors keep their order


# --------------------------------------------------------------------------
# 2. N=0 byte-identical
# --------------------------------------------------------------------------
class TestLegacyByteIdentical:
    def test_zero_seats_no_mutation(self):
        fused = {"a": 0.9, "b": 0.5, "c": 0.4, "d": 0.1}
        before = dict(fused)
        seated = apply_embedding_reserved_seats(fused, ["c", "d"], 0, 10)
        assert seated == []
        assert fused == before

    def test_everything_already_fits(self):
        # <= top_k candidates: nothing to displace.
        fused = {"a": 0.9, "b": 0.5, "c": 0.4}
        before = dict(fused)
        seated = apply_embedding_reserved_seats(fused, ["c"], 3, 10)
        assert seated == []
        assert fused == before

    def test_embedding_candidates_already_in_topk(self):
        fused = {f"t{i}": 1.0 - i * 0.01 for i in range(15)}
        before = dict(fused)
        # t0,t1 are already top-of-fused; no seating needed.
        seated = apply_embedding_reserved_seats(fused, ["t0", "t1"], 2, 10)
        assert seated == []
        assert fused == before

    def test_defaults_are_legacy(self):
        assert ScoringConfig().embedding_reserved_seats == 0


# --------------------------------------------------------------------------
# 3. No-gold-displacement regression (flip-case shape)
# --------------------------------------------------------------------------
class TestNoGoldDisplacement:
    def test_gold_seated_and_existing_gold_survives(self):
        # gold-in-topk sits at rank 1 (from another signal); gold-embedding-only
        # is buried at the bottom of the fused order. Seating must flip the
        # buried gold into the top-K WITHOUT evicting the gold already seated.
        fused = {"gold_top": 5.0}
        fused.update({f"n{i}": 1.0 - i * 0.01 for i in range(11)})  # noise n0..n10
        fused.update({"gold_emb": 0.20})                            # buried gold
        emb = ["gold_emb", "n0"]  # embedding signal surfaces the buried gold
        k = 10

        top = production_topk(fused, emb, 1, k)
        assert "gold_emb" in top          # buried gold rescued
        assert "gold_top" == top[0]       # pre-existing gold never displaced
        assert len(top) == k
        # Exactly one displacement: the lowest-ranked non-seated top-K entry.
        assert "n9" not in top            # n9 was the lowest of the original top-10

    def test_seated_scores_land_between_survivor_and_displaced(self):
        fused = {f"f{i}": 1.0 - i * 0.05 for i in range(12)}
        fused["e0"] = 0.10
        scores = dict(fused)
        apply_embedding_reserved_seats(scores, ["e0"], 1, 10)
        # e0 seated strictly below the last survivor (f8) and above the first
        # displaced original (f9), so it lands exactly at rank 10.
        assert fused["f8"] > scores["e0"] > fused["f9"]
        order = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        assert order[9] == "e0"


# --------------------------------------------------------------------------
# 4. Config plumbing
# --------------------------------------------------------------------------
class TestConfigPlumbing:
    def test_yaml_parses_reserved_seats(self):
        cfg = load_config(config_dict={
            "retrieval": {"scoring": {"embedding_reserved_seats": 3}}
        })
        assert cfg.retriever.scoring.embedding_reserved_seats == 3

    def test_yaml_default_absent(self):
        cfg = load_config(config_dict={"retrieval": {"scoring": {}}})
        assert cfg.retriever.scoring.embedding_reserved_seats == 0
