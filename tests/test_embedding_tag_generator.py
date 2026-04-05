"""Tests for EmbeddingTagGenerator."""
import pytest
from virtual_context.core.embedding_tag_generator import EmbeddingTagGenerator, _cosine_similarity
from virtual_context.types import TagGeneratorConfig


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Simple char-frequency embedding for testing. Returns 26-dim vector."""
    result = []
    for text in texts:
        vec = [0.0] * 26
        text_lower = text.lower()
        for c in text_lower:
            if 'a' <= c <= 'z':
                vec[ord(c) - ord('a')] += 1
        # Normalize
        total = sum(v * v for v in vec) ** 0.5
        if total > 0:
            vec = [v / total for v in vec]
        result.append(vec)
    return result


class TestCosineSimilarity:
    def test_identical(self):
        a = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestEmbeddingTagGenerator:
    def _make_generator(self, **kwargs):
        config = TagGeneratorConfig(type="embedding", max_tags=3, min_tags=1)
        return EmbeddingTagGenerator(
            config=config,
            embed_fn=_mock_embed,
            similarity_threshold=kwargs.get("threshold", 0.3),
            load_cached_embeddings=kwargs.get("load_cached_embeddings"),
            save_cached_embeddings=kwargs.get("save_cached_embeddings"),
        )

    def test_generate_with_existing_tags(self):
        gen = self._make_generator()
        result = gen.generate_tags("database schema migration", existing_tags=["database", "api", "frontend"])
        assert result.source == "embedding"
        assert len(result.tags) > 0
        assert len(result.tags) > 0

    def test_no_existing_tags_returns_general(self):
        gen = self._make_generator()
        result = gen.generate_tags("hello world")
        assert result.tags == ["_general"]
        assert result.source == "fallback"

    def test_vocabulary_tracking(self):
        gen = self._make_generator()
        gen.generate_tags("database query", existing_tags=["database"])
        assert gen._tag_vocabulary.get("database", 0) > 0

    def test_max_tags_limit(self):
        gen = self._make_generator(threshold=0.01)  # very low threshold
        result = gen.generate_tags("database api frontend auth testing", existing_tags=["database", "api", "frontend", "auth", "testing"])
        assert len(result.tags) <= 3

    def test_load_vocabulary(self):
        gen = self._make_generator()
        gen.load_vocabulary({"database": 5, "api": 3})
        assert "database" in gen._tag_embeddings
        assert "api" in gen._tag_embeddings

    def test_high_threshold_returns_general(self):
        gen = self._make_generator(threshold=0.99)
        result = gen.generate_tags("something random", existing_tags=["database", "api"])
        assert result.tags == ["_general"]

    def test_shared_cache_reuses_remote_embeddings(self):
        saved: dict[str, dict[str, list[float]]] = {}
        embed_calls: list[list[str]] = []

        def cached_loader(model_name: str, tags: list[str]) -> dict[str, list[float]]:
            assert model_name == "all-MiniLM-L6-v2"
            return {"database": _mock_embed(["database"])[0]}

        def cached_saver(model_name: str, embeddings: dict[str, list[float]]) -> None:
            saved[model_name] = dict(embeddings)

        def recording_embed(texts: list[str]) -> list[list[float]]:
            embed_calls.append(list(texts))
            return _mock_embed(texts)

        config = TagGeneratorConfig(type="embedding", max_tags=3, min_tags=1)
        gen = EmbeddingTagGenerator(
            config=config,
            embed_fn=recording_embed,
            load_cached_embeddings=cached_loader,
            save_cached_embeddings=cached_saver,
        )

        result = gen.generate_tags("database schema migration", existing_tags=["database", "api"])

        assert result.source == "embedding"
        assert any(call == ["api"] for call in embed_calls)
        assert all(call != ["database", "api"] for call in embed_calls)
        assert "api" in saved["all-MiniLM-L6-v2"]

    def test_generate_only_scores_current_request_vocabulary(self):
        vectors = {
            "frontend": [1.0, 0.0],
            "database": [0.0, 1.0],
            "frontend question": [1.0, 0.0],
        }

        def embed(texts: list[str]) -> list[list[float]]:
            return [vectors[t.lower()] for t in texts]

        config = TagGeneratorConfig(type="embedding", max_tags=3, min_tags=1)
        gen = EmbeddingTagGenerator(
            config=config,
            embed_fn=embed,
            similarity_threshold=0.5,
        )

        first = gen.generate_tags("frontend question", existing_tags=["frontend"])
        assert first.tags == ["frontend"]

        second = gen.generate_tags("frontend question", existing_tags=["database"])
        assert second.tags == ["_general"]
        assert second.source == "fallback"
