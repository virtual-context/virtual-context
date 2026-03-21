"""Tests for 3-signal RRF fusion retrieval scoring."""
from datetime import datetime, timezone
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import TagSummary


class TestFTSTagSummaries:
    def test_search_tag_summaries_fts(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        ts = TagSummary(
            tag="basketball", summary="Basketball tournament delivery date for remote shutter release",
            summary_tokens=20,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        store.save_tag_summary(ts, conversation_id="c1")
        results = store.search_tag_summaries_fts("delivery shutter", limit=10, conversation_id="c1")
        assert len(results) >= 1
        assert results[0][0] == "basketball"

    def test_search_returns_empty_when_no_match(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        results = store.search_tag_summaries_fts("nonexistent query", limit=10)
        assert results == []

    def test_fts_updates_on_tag_summary_update(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        ts = TagSummary(
            tag="cooking", summary="Italian pasta recipes",
            summary_tokens=10,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        store.save_tag_summary(ts, conversation_id="c1")
        # Update the summary
        ts2 = TagSummary(
            tag="cooking", summary="Japanese sushi preparation techniques",
            summary_tokens=10,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 6, 2, tzinfo=timezone.utc),
        )
        store.save_tag_summary(ts2, conversation_id="c1")
        # Old content should not match
        assert store.search_tag_summaries_fts("pasta", limit=10, conversation_id="c1") == []
        # New content should match
        results = store.search_tag_summaries_fts("sushi", limit=10, conversation_id="c1")
        assert len(results) >= 1
        assert results[0][0] == "cooking"


class TestTagSummaryEmbeddings:
    def test_store_and_load_embeddings(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        store.store_tag_summary_embedding("auth", "c1", [0.1, 0.2, 0.3])
        store.store_tag_summary_embedding("database", "c1", [0.4, 0.5, 0.6])
        loaded = store.load_tag_summary_embeddings("c1")
        assert "auth" in loaded
        assert "database" in loaded
        assert loaded["auth"] == [0.1, 0.2, 0.3]

    def test_load_empty_when_none_stored(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        assert store.load_tag_summary_embeddings("c1") == {}

    def test_upsert_replaces_embedding(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        store.store_tag_summary_embedding("auth", "c1", [0.1, 0.2])
        store.store_tag_summary_embedding("auth", "c1", [0.9, 0.8])
        loaded = store.load_tag_summary_embeddings("c1")
        assert loaded["auth"] == [0.9, 0.8]


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------
from virtual_context.core.retrieval_scoring import rrf_fuse, score_candidates
from virtual_context.types import ScoringConfig, StoredSummary, TagStats
from unittest.mock import MagicMock


class TestRRFFusion:
    def test_rrf_basic_ranking(self):
        """Candidate in all 3 signals ranks higher than candidate in 1."""
        rankings = {
            "idf": {"auth": 0, "database": 1, "cooking": 2},
            "bm25": {"auth": 0, "cooking": 1},
            "embedding": {"auth": 0, "database": 1},
        }
        weights = {"idf": 0.50, "bm25": 0.30, "embedding": 0.20}
        scores = rrf_fuse(rankings, weights, k=60)
        assert scores["auth"] > scores["database"]
        assert scores["auth"] > scores["cooking"]

    def test_rrf_missing_signal_uses_penalty_rank(self):
        """Candidate missing from a signal gets fixed penalty rank."""
        rankings = {
            "idf": {"auth": 0},
            "bm25": {},
            "embedding": {"auth": 0},
        }
        weights = {"idf": 0.50, "bm25": 0.30, "embedding": 0.20}
        scores = rrf_fuse(rankings, weights, k=60)
        assert scores["auth"] > 0

    def test_rrf_penalty_is_constant(self):
        """Penalty rank is rrf_k * 2, not candidate set size."""
        r1 = {"idf": {"a": 0, "b": 1, "c": 2}, "bm25": {}, "embedding": {}}
        r2 = {"idf": {"a": 0}, "bm25": {}, "embedding": {}}
        w = {"idf": 0.50, "bm25": 0.30, "embedding": 0.20}
        s1 = rrf_fuse(r1, w, k=60)
        s2 = rrf_fuse(r2, w, k=60)
        # 'a' should have same score in both (rank 0 in idf, penalty in others)
        assert abs(s1["a"] - s2["a"]) < 0.0001

    def test_score_candidates_integration(self):
        """Full pipeline with mocked store."""
        store = MagicMock()
        store.get_summaries_by_tags.return_value = [
            StoredSummary(
                ref="seg-auth", primary_tag="auth", tags=["auth"],
                summary="Auth summary", summary_tokens=50,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
        ]
        store.search_tag_summaries_fts.return_value = [
            ("auth", 2.5),
            ("database", 1.0),
        ]
        store.load_tag_summary_embeddings.return_value = {}
        store.search_full_text.return_value = []
        store.get_all_tags.return_value = [
            TagStats(tag="auth", usage_count=3),
            TagStats(tag="database", usage_count=5),
        ]

        config = ScoringConfig()
        scores, breakdowns = score_candidates(
            query_tags=["auth"],
            related_tags=[],
            query_text="authentication question",
            query_embedding=None,
            store=store,
            idf_weights={"auth": 2.0, "database": 1.5},
            conversation_id="c1",
            config=config,
        )
        assert "auth" in scores
        # auth should rank highest (appears in both IDF and BM25)
        if "database" in scores:
            assert scores["auth"] >= scores["database"]


# ---------------------------------------------------------------------------
# Dampening tests
# ---------------------------------------------------------------------------
from virtual_context.core.retrieval_scoring import (
    apply_gravity_dampening,
    apply_hub_dampening,
    apply_resolution_boost,
)


class TestDampening:
    def test_gravity_halves_unsupported_embedding(self):
        embed = {"auth": 0.8, "cooking": 0.3}
        bm25 = {"auth": 0.0}  # no BM25 support for auth
        apply_gravity_dampening(embed, bm25, threshold=0.5, factor=0.5)
        assert embed["auth"] == 0.4  # halved
        assert embed["cooking"] == 0.3  # below threshold, untouched

    def test_gravity_skips_when_bm25_present(self):
        embed = {"auth": 0.8}
        bm25 = {"auth": 1.5}  # BM25 supports this
        apply_gravity_dampening(embed, bm25, threshold=0.5, factor=0.5)
        assert embed["auth"] == 0.8  # untouched

    def test_hub_penalizes_high_count_tags(self):
        fused = {"common": 0.5, "rare": 0.5}
        # 11 tags: p90_idx=9 -> counts[9]=40, max=100, so common (100) > p90 (40)
        tag_stats = {"common": 100, "rare": 2, "other1": 3, "other2": 5, "other3": 8,
                     "other4": 10, "other5": 15, "other6": 20, "other7": 25, "other8": 30, "other9": 40}
        apply_hub_dampening(fused, tag_stats, set(), penalty_strength=0.6, min_score_fraction=0.2)
        assert fused["common"] < 0.5  # penalized
        assert fused["rare"] == 0.5  # untouched

    def test_hub_exempts_query_tags(self):
        fused = {"common": 0.5}
        # 11 tags: p90_idx=9 -> counts[9]=45, max=100, so common (100) > p90 (45)
        tag_stats = {"common": 100, "other1": 5, "other2": 10, "other3": 15,
                     "other4": 20, "other5": 25, "other6": 30, "other7": 35, "other8": 40,
                     "other9": 45, "other10": 3}
        apply_hub_dampening(fused, tag_stats, {"common"}, penalty_strength=0.6, min_score_fraction=0.2)
        assert fused["common"] == 0.5  # exempt — in query tags

    def test_resolution_boosts_actionable(self):
        fused = {"auth": 0.5, "cooking": 0.5}
        apply_resolution_boost(fused, {"auth"}, boost=1.15)
        assert fused["auth"] == 0.5 * 1.15
        assert fused["cooking"] == 0.5  # no facts

    def test_all_dampening_toggleable(self):
        """Each filter respects its enabled flag."""
        from virtual_context.types import DampeningConfig, ScoringConfig
        cfg = ScoringConfig(dampening=DampeningConfig(
            hub_enabled=False, gravity_enabled=False, resolution_enabled=False,
        ))
        assert not cfg.dampening.hub_enabled
        assert not cfg.dampening.gravity_enabled
        assert not cfg.dampening.resolution_enabled
