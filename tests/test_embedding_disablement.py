"""Disabled embedding providers and runtime embed-failure degradation."""
import pytest

from virtual_context.core.embedding_provider import EmbeddingProvider
from virtual_context.core.embedding_tag_generator import EmbeddingTagGenerator
from virtual_context.core.tag_generator import KeywordTagGenerator, build_tag_generator
from virtual_context.types import TagGeneratorConfig


def _mock_embed(texts):
    result = []
    for text in texts:
        vec = [0.0] * 26
        for c in text.lower():
            if "a" <= c <= "z":
                vec[ord(c) - ord("a")] += 1
        total = sum(v * v for v in vec) ** 0.5
        result.append([v / total for v in vec] if total else vec)
    return result


class TestDisabledProvider:
    def test_disabled_returns_none_without_loading(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def forbid_st(name, *args, **kwargs):
            if name.startswith("sentence_transformers"):
                raise AssertionError("disabled provider attempted a local model load")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", forbid_st)
        provider = EmbeddingProvider(disabled=True)
        assert provider.disabled is True
        assert provider.get_embed_fn() is None
        assert provider.get_embed_fn() is None

    def test_disabled_rejects_embed_fn(self):
        with pytest.raises(ValueError):
            EmbeddingProvider(embed_fn=_mock_embed, disabled=True)

    def test_injected_fn_unaffected_by_default_flag(self):
        provider = EmbeddingProvider(embed_fn=_mock_embed)
        assert provider.disabled is False
        assert provider.get_embed_fn() is _mock_embed


class TestFactoryKeywordFallback:
    def test_factory_yielding_none_degrades_to_general_not_keyword(self):
        """An unavailable factory must not silently substitute a keyword
        tagger; it builds a degraded embedding generator whose calls return
        the established _general fallback, which health monitoring alerts
        on."""
        config = TagGeneratorConfig(type="embedding", max_tags=3, min_tags=1)
        gen = build_tag_generator(config, embed_fn_factory=lambda: None)
        assert isinstance(gen, EmbeddingTagGenerator)
        result = gen.generate_tags("some text", ["real-tag"])
        assert result.primary == "_general"
        assert result.source == "fallback"

    def test_factory_yielding_none_ignores_configured_keyword_fallback(self):
        from virtual_context.types import KeywordTagConfig
        config = TagGeneratorConfig(
            type="embedding", max_tags=3, min_tags=1,
            keyword_fallback=KeywordTagConfig(),
        )
        gen = build_tag_generator(config, embed_fn_factory=lambda: None)
        assert isinstance(gen, EmbeddingTagGenerator)
        result = gen.generate_tags("anything")
        assert result.primary == "_general"
        assert result.source == "fallback"

    def test_factory_yielding_fn_selects_embedding(self):
        config = TagGeneratorConfig(type="embedding", max_tags=3, min_tags=1)
        gen = build_tag_generator(config, embed_fn_factory=lambda: _mock_embed)
        assert isinstance(gen, EmbeddingTagGenerator)


class TestRuntimeDegradation:
    def _generator(self, embed_fn):
        config = TagGeneratorConfig(type="embedding", max_tags=3, min_tags=1)
        return EmbeddingTagGenerator(
            config=config, embed_fn=embed_fn, similarity_threshold=0.1,
        )

    def test_embed_failure_degrades_to_keyword_and_recovers(self):
        calls = {"n": 0}

        def flaky(texts):
            calls["n"] += 1
            if calls["n"] <= 1:
                raise ConnectionError("embed backend down")
            return _mock_embed(texts)

        gen = self._generator(flaky)
        degraded = gen.generate_tags(
            "database connection pooling strategies",
            existing_tags=["databases"],
        )
        assert degraded.tags, "degraded call must still produce tags"
        assert not gen._tag_embeddings, "no embeddings written by failed call"

        recovered = gen.generate_tags(
            "database connection pooling strategies",
            existing_tags=["databases"],
        )
        assert recovered.source != "error"
        assert gen._degraded_logged is False

    def test_degraded_call_never_raises(self):
        def always_down(texts):
            raise TimeoutError("embed backend timeout")

        gen = self._generator(always_down)
        for _ in range(3):
            result = gen.generate_tags("alpine gardening drainage")
            assert result.tags


class TestSemanticNoneNotCached:
    def test_provider_none_is_reconsulted(self):
        from virtual_context.core.semantic_search import SemanticSearchManager

        class FlakyProvider:
            def __init__(self):
                self.calls = 0

            def get_embed_fn(self):
                self.calls += 1
                return None if self.calls == 1 else _mock_embed

        provider = FlakyProvider()
        manager = SemanticSearchManager.__new__(SemanticSearchManager)
        manager._embedding_provider = provider
        from virtual_context.core.semantic_search import _EMBED_NOT_LOADED
        manager._embed_fn = _EMBED_NOT_LOADED
        assert manager.get_embed_fn() is None
        assert manager.get_embed_fn() is _mock_embed
        assert provider.calls == 2


class TestRelevanceGateFailOpen:
    def test_embed_failure_passes_through(self):
        from virtual_context.core.semantic_search import SemanticSearchManager

        def always_down(texts):
            raise ConnectionError("down")

        manager = SemanticSearchManager.__new__(SemanticSearchManager)
        manager._embedding_provider = EmbeddingProvider(embed_fn=always_down)
        from virtual_context.core.semantic_search import _EMBED_NOT_LOADED
        manager._embed_fn = _EMBED_NOT_LOADED

        class _Cfg:
            class tag_generator:
                context_bleed_threshold = 0.1

        manager._config = _Cfg()
        ok, score = manager.context_is_relevant_with_score("current turn", ["a", "b"])
        assert ok is True
        assert score == -1.0


def test_degraded_generator_never_substitutes_keyword_matching():
    """The degraded result is an alertable _general, not a silent keyword
    substitute — even when a keyword config would have matched the text."""
    from virtual_context.types import KeywordTagConfig, TagGeneratorConfig
    from virtual_context.core.tag_generator import build_tag_generator

    config = TagGeneratorConfig(
        type="embedding",
        keyword_fallback=KeywordTagConfig(
            tag_keywords={"databases": ["postgres"]},
        ),
    )
    gen = build_tag_generator(config, embed_fn_factory=lambda: None)
    result = gen.generate_tags("postgres connection pooling question")
    assert result.primary == "_general"
    assert result.tags == ["_general"]
    assert result.source == "fallback"
