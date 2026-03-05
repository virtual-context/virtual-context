"""Tests for ModelCatalog: pricing from YAML."""

from __future__ import annotations

import os

import pytest
import yaml

from virtual_context.core.model_catalog import ModelCatalog, ModelInfo


@pytest.fixture
def sample_yaml(tmp_path):
    """Create a temporary models.yaml for testing."""
    data = {
        "models": {
            "claude-haiku-4-5-20251001": {
                "provider": "anthropic",
                "input_per_mtok": 1.00,
                "output_per_mtok": 5.00,
                "context_window": 200000,
                "aliases": ["haiku", "claude-haiku", "claude-haiku-4-5"],
            },
            "gpt-4.1-nano": {
                "provider": "openai",
                "input_per_mtok": 0.10,
                "output_per_mtok": 0.40,
                "context_window": 128000,
                "aliases": ["gpt4-nano", "gpt-4.1-nano"],
            },
            "qwen3:4b-instruct-2507-fp16": {
                "provider": "ollama",
                "input_per_mtok": 0.00,
                "output_per_mtok": 0.00,
                "context_window": 32768,
                "aliases": ["qwen3-4b"],
            },
        }
    }
    path = tmp_path / "models.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


@pytest.fixture
def catalog(sample_yaml):
    return ModelCatalog(sample_yaml)


class TestLoadFromYAML:
    def test_loads_all_models(self, catalog):
        """Catalog loads all models from the YAML file."""
        assert catalog.get_pricing("claude-haiku-4-5-20251001") == (1.00, 5.00)
        assert catalog.get_pricing("gpt-4.1-nano") == (0.10, 0.40)
        assert catalog.get_pricing("qwen3:4b-instruct-2507-fp16") == (0.00, 0.00)

    def test_exact_match(self, catalog):
        """Exact canonical name resolves correctly."""
        inp, out = catalog.get_pricing("claude-haiku-4-5-20251001")
        assert inp == 1.00
        assert out == 5.00


class TestAliasResolution:
    def test_short_alias(self, catalog):
        """Short alias resolves to the canonical model."""
        assert catalog.get_pricing("haiku") == (1.00, 5.00)

    def test_medium_alias(self, catalog):
        """Medium alias resolves correctly."""
        assert catalog.get_pricing("claude-haiku") == (1.00, 5.00)

    def test_versioned_alias(self, catalog):
        """Versioned alias resolves correctly."""
        assert catalog.get_pricing("claude-haiku-4-5") == (1.00, 5.00)

    def test_alias_case_insensitive(self, catalog):
        """Alias matching is case-insensitive."""
        assert catalog.get_pricing("Haiku") == (1.00, 5.00)
        assert catalog.get_pricing("HAIKU") == (1.00, 5.00)

    def test_alias_for_nano(self, catalog):
        """gpt4-nano alias resolves to gpt-4.1-nano."""
        assert catalog.get_pricing("gpt4-nano") == (0.10, 0.40)

    def test_alias_for_ollama(self, catalog):
        """qwen3-4b alias resolves to local model."""
        assert catalog.get_pricing("qwen3-4b") == (0.00, 0.00)


class TestSubstringFallback:
    def test_substring_of_canonical(self, catalog):
        """Substring of canonical name resolves via fallback."""
        # "haiku" is also an alias, so test with a substring that isn't an alias
        inp, out = catalog.get_pricing("claude-haiku-4-5-20251001-extra")
        # canonical "claude-haiku-4-5-20251001" is a substring of input
        assert inp == 1.00
        assert out == 5.00

    def test_canonical_substring_of_input(self, catalog):
        """When canonical name is a substring of the queried name."""
        # "gpt-4.1-nano" is contained within "my-gpt-4.1-nano-deployment"
        inp, out = catalog.get_pricing("my-gpt-4.1-nano-deployment")
        assert inp == 0.10
        assert out == 0.40


class TestUnknownModel:
    def test_unknown_returns_zero_pricing(self, catalog):
        """Unknown model returns zero pricing."""
        assert catalog.get_pricing("totally-unknown-model") == (0.0, 0.0)

    def test_unknown_returns_zero_context_window(self, catalog):
        """Unknown model returns zero context window."""
        assert catalog.get_context_window("totally-unknown-model") == 0


class TestCalculateCost:
    def test_basic_calculation(self, catalog):
        """calculate_cost returns correct USD value."""
        # 1M input tokens at $1.00/Mtok + 500K output tokens at $5.00/Mtok
        cost = catalog.calculate_cost("haiku", input_tokens=1_000_000, output_tokens=500_000)
        expected = (1_000_000 * 1.00 + 500_000 * 5.00) / 1_000_000
        assert cost == pytest.approx(expected)
        assert cost == pytest.approx(3.50)

    def test_zero_tokens(self, catalog):
        """Zero tokens produces zero cost."""
        cost = catalog.calculate_cost("haiku", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_unknown_model_zero_cost(self, catalog):
        """Unknown model produces zero cost regardless of token count."""
        cost = catalog.calculate_cost("unknown", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 0.0

    def test_free_model_zero_cost(self, catalog):
        """Local/free model produces zero cost."""
        cost = catalog.calculate_cost("qwen3-4b", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 0.0

    def test_small_token_count(self, catalog):
        """Small token counts produce fractional costs."""
        # 1000 input tokens at $0.10/Mtok = $0.0001
        cost = catalog.calculate_cost("gpt4-nano", input_tokens=1000, output_tokens=0)
        assert cost == pytest.approx(0.0001)


class TestGetContextWindow:
    def test_anthropic_context_window(self, catalog):
        """Anthropic model returns 200K context window."""
        assert catalog.get_context_window("haiku") == 200000

    def test_openai_context_window(self, catalog):
        """OpenAI model returns 128K context window."""
        assert catalog.get_context_window("gpt4-nano") == 128000

    def test_ollama_context_window(self, catalog):
        """Ollama model returns 32K context window."""
        assert catalog.get_context_window("qwen3-4b") == 32768


class TestMissingFile:
    def test_missing_file_creates_empty_catalog(self):
        """Missing YAML file creates an empty catalog (no crash)."""
        catalog = ModelCatalog("/nonexistent/path/models.yaml")
        assert catalog.get_pricing("haiku") == (0.0, 0.0)
        assert catalog.get_context_window("haiku") == 0
        assert catalog.calculate_cost("haiku", 1_000_000, 1_000_000) == 0.0


class TestDefaultCatalog:
    def test_default_loads_bundled_yaml(self):
        """ModelCatalog.default() loads the bundled models.yaml from project root."""
        catalog = ModelCatalog.default()
        # Should have haiku pricing from the real models.yaml
        inp, out = catalog.get_pricing("claude-haiku-4-5-20251001")
        assert inp == 1.00
        assert out == 5.00

    def test_default_has_all_models(self):
        """Default catalog contains all expected models."""
        catalog = ModelCatalog.default()
        # Spot-check a few canonical names
        assert catalog.get_context_window("claude-haiku-4-5-20251001") == 200000
        assert catalog.get_context_window("gpt-5-mini") == 128000
        assert catalog.get_context_window("qwen3:4b-instruct-2507-fp16") == 32768

    def test_default_opus_pricing(self):
        """Default catalog has correct Opus pricing."""
        catalog = ModelCatalog.default()
        assert catalog.get_pricing("opus") == (15.00, 75.00)


class TestModelInfo:
    def test_frozen_dataclass(self):
        """ModelInfo is immutable (frozen=True)."""
        info = ModelInfo(
            canonical_name="test",
            provider="test",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
            context_window=100,
        )
        with pytest.raises(AttributeError):
            info.input_per_mtok = 999  # type: ignore[misc]
