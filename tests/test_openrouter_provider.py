"""Tests for first-class OpenRouter provider support."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestBuildProviderOpenRouter:
    """engine._build_provider() handles 'openrouter' type."""

    def _make_engine(self):
        from virtual_context.config import load_config
        from virtual_context.engine import VirtualContextEngine

        cfg = load_config(config_dict={
            "version": "0.2",
            "storage_root": "",
            "providers": {},
            "storage": {"backend": "sqlite", "sqlite": {"path": ":memory:"}},
        })
        return VirtualContextEngine(config=cfg)

    def test_openrouter_returns_generic_openai_provider(self):
        from virtual_context.providers.generic_openai import GenericOpenAIProvider

        engine = self._make_engine()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-123"}):
            provider = engine._build_provider("openrouter", {"type": "openrouter", "model": "qwen/qwen3-30b-a3b"})

        assert isinstance(provider, GenericOpenAIProvider)
        assert provider.base_url == "https://openrouter.ai/api/v1"
        assert provider.model == "qwen/qwen3-30b-a3b"
        assert provider.api_key == "test-key-123"
        engine.close()

    def test_openrouter_defaults_api_key_env(self):
        engine = self._make_engine()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "from-env"}, clear=False):
            provider = engine._build_provider("openrouter", {"type": "openrouter"})
        assert provider.api_key == "from-env"
        engine.close()

    def test_openrouter_explicit_api_key_overrides_env(self):
        engine = self._make_engine()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "from-env"}, clear=False):
            provider = engine._build_provider("openrouter", {
                "type": "openrouter", "api_key": "explicit-key",
            })
        assert provider.api_key == "explicit-key"
        engine.close()

    def test_openrouter_custom_base_url(self):
        engine = self._make_engine()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            provider = engine._build_provider("openrouter", {
                "type": "openrouter", "base_url": "https://custom.openrouter.ai/v1",
            })
        assert provider.base_url == "https://custom.openrouter.ai/v1"
        engine.close()

    def test_generic_openai_still_works(self):
        """Ensure existing generic_openai type is not broken."""
        from virtual_context.providers.generic_openai import GenericOpenAIProvider

        engine = self._make_engine()
        provider = engine._build_provider("ollama", {
            "type": "generic_openai",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3:4b",
        })
        assert isinstance(provider, GenericOpenAIProvider)
        assert provider.base_url == "http://127.0.0.1:11434/v1"
        engine.close()

    def test_reasoning_effort_is_opt_in_and_null_content_is_empty(self):
        from virtual_context.providers.generic_openai import GenericOpenAIProvider

        ordinary = GenericOpenAIProvider(model="ordinary")
        assert "reasoning" not in ordinary._build_payload("s", "u", 100)

        admission = GenericOpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-fable-5",
            reasoning_effort="low",
        )
        payload = admission._build_payload("s", "u", 100)
        assert payload["reasoning"] == {"effort": "low"}
        assert admission._extract_text({
            "choices": [{"message": {"content": None}}],
        }) == ""

        openai = GenericOpenAIProvider(
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
            reasoning_effort="low",
        )
        assert (
            openai._build_payload("s", "u", 100)["reasoning_effort"]
            == "low"
        )

        strict_compatible = GenericOpenAIProvider(
            base_url="https://strict-compatible.example/v1",
            model="some-model",
            reasoning_effort="low",
        )
        strict_payload = strict_compatible._build_payload("s", "u", 100)
        assert "reasoning" not in strict_payload
        assert "reasoning_effort" not in strict_payload


class TestBuildVcConfigOpenRouter:
    """_build_vc_config() in benchmark harness supports 'openrouter'."""

    def test_openrouter_tagger_creates_provider_entry(self):
        import sys
        sys.path.insert(0, ".")
        from benchmarks.longmemeval.vc_runner import _build_vc_config

        cfg = _build_vc_config(
            tagger_provider="openrouter",
            tagger_model="qwen/qwen3-30b-a3b",
        )
        assert "openrouter" in cfg["providers"]
        entry = cfg["providers"]["openrouter"]
        assert entry["type"] == "openrouter"
        assert entry["base_url"] == "https://openrouter.ai/api/v1"
        assert cfg["tag_generator"]["provider"] == "openrouter"
        assert cfg["tag_generator"]["model"] == "qwen/qwen3-30b-a3b"
        assert cfg["summarization"]["provider"] == "openrouter"

    def test_openrouter_summarizer_only(self):
        import sys
        sys.path.insert(0, ".")
        from benchmarks.longmemeval.vc_runner import _build_vc_config

        cfg = _build_vc_config(
            tagger_provider="anthropic",
            summarizer_provider="openrouter",
            summarizer_model="qwen/qwen3-30b-a3b",
        )
        assert "openrouter" in cfg["providers"]
        assert "anthropic" in cfg["providers"]
