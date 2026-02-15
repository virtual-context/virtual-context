"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from virtual_context.config import load_config, validate_config


class TestLoadConfig:
    def test_load_defaults(self):
        config = load_config(config_dict={})
        assert config.version == "0.2"
        assert config.context_window == 120_000
        assert config.tag_generator.type == "keyword"
        assert config.storage.backend == "sqlite"

    def test_load_from_dict(self):
        config = load_config(config_dict={
            "context_window": 50_000,
            "tag_generator": {
                "type": "llm",
                "provider": "ollama",
                "model": "qwen3:4b-instruct-2507-fp16",
            },
        })
        assert config.context_window == 50_000
        assert config.tag_generator.type == "llm"
        assert config.tag_generator.provider == "ollama"

    def test_load_from_yaml_file(self):
        raw = {
            "version": "0.2",
            "context_window": 80_000,
            "tag_generator": {"type": "keyword"},
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(raw, f)
            f.flush()
            config = load_config(config_path=f.name)
        assert config.context_window == 80_000

    def test_load_tag_rules(self):
        config = load_config(config_dict={
            "tag_rules": [
                {"match": "architecture*", "priority": 10, "ttl_days": None},
                {"match": "debug*", "priority": 7, "ttl_days": 7},
            ],
        })
        assert len(config.tag_rules) == 2
        assert config.tag_rules[0].match == "architecture*"
        assert config.tag_rules[0].priority == 10
        assert config.tag_rules[1].ttl_days == 7

    def test_load_strategy_config(self):
        config = load_config(config_dict={
            "retrieval": {
                "strategy_config": {
                    "default": {
                        "min_overlap": 2,
                        "max_results": 5,
                        "max_budget_fraction": 0.3,
                    },
                },
            },
        })
        sc = config.retriever.strategy_configs["default"]
        assert sc.min_overlap == 2
        assert sc.max_results == 5
        assert sc.max_budget_fraction == 0.3

    def test_load_keyword_fallback(self):
        config = load_config(config_dict={
            "tag_generator": {
                "type": "keyword",
                "keyword_fallback": {
                    "tag_keywords": {"legal": ["court", "filing"]},
                    "tag_patterns": {"legal": [r"\bcase\b"]},
                },
            },
        })
        fb = config.tag_generator.keyword_fallback
        assert fb is not None
        assert "legal" in fb.tag_keywords
        assert "legal" in fb.tag_patterns

    def test_load_cost_tracking(self):
        config = load_config(config_dict={
            "cost_tracking": {
                "enabled": True,
                "pricing": {
                    "ollama": {"input_per_1k": 0.0, "output_per_1k": 0.0},
                },
            },
        })
        assert config.cost_tracking.enabled is True
        assert "ollama" in config.cost_tracking.pricing

    def test_default_strategy_added(self):
        config = load_config(config_dict={})
        assert "default" in config.retriever.strategy_configs

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/path.yaml")


class TestValidateConfig:
    def test_valid_default_config(self):
        config = load_config(config_dict={})
        errors = validate_config(config)
        assert errors == []

    def test_invalid_tag_generator_type(self):
        config = load_config(config_dict={})
        config.tag_generator.type = "invalid"
        errors = validate_config(config)
        assert any("tag_generator.type" in e for e in errors)

    def test_llm_requires_provider(self):
        config = load_config(config_dict={
            "tag_generator": {"type": "llm"},
        })
        errors = validate_config(config)
        assert any("provider" in e for e in errors)

    def test_max_tags_less_than_min(self):
        config = load_config(config_dict={})
        config.tag_generator.max_tags = 1
        config.tag_generator.min_tags = 5
        errors = validate_config(config)
        assert any("max_tags" in e for e in errors)

    def test_soft_gte_hard(self):
        config = load_config(config_dict={
            "compaction": {"soft_threshold": 0.9, "hard_threshold": 0.85},
        })
        errors = validate_config(config)
        assert any("soft_threshold" in e for e in errors)

    def test_protected_recent_turns_zero(self):
        config = load_config(config_dict={
            "compaction": {"protected_recent_turns": 0},
        })
        errors = validate_config(config)
        assert any("protected_recent_turns" in e for e in errors)

    def test_invalid_strategy_budget_fraction(self):
        config = load_config(config_dict={})
        config.retriever.strategy_configs["default"].max_budget_fraction = 1.5
        errors = validate_config(config)
        assert any("max_budget_fraction" in e for e in errors)

    def test_invalid_storage_backend(self):
        config = load_config(config_dict={})
        config.storage.backend = "redis"
        errors = validate_config(config)
        assert any("storage.backend" in e for e in errors)
