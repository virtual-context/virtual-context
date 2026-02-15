"""Tests for presets: registration, tag config, template validity."""

from __future__ import annotations

import re

import yaml

from virtual_context.config import load_config, validate_config
from virtual_context.presets import get_preset, list_presets
from virtual_context.presets.coding import (
    CODING_CONFIG,
    CODING_TAG_KEYWORDS,
    CODING_TAG_PATTERNS,
    CODING_TAG_RULES,
    CODING_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_coding_preset_registered():
    preset = get_preset("coding")
    assert preset is not None
    assert preset.name == "coding"


def test_list_presets_includes_coding():
    names = [p.name for p in list_presets()]
    assert "coding" in names


def test_unknown_preset_returns_none():
    assert get_preset("nonexistent") is None


# ---------------------------------------------------------------------------
# Tag keywords taxonomy
# ---------------------------------------------------------------------------

EXPECTED_KEYWORD_TAGS = [
    "architecture", "database", "auth", "backend", "frontend",
    "debugging", "infrastructure", "testing",
]


def test_all_keyword_tags_present():
    assert sorted(CODING_TAG_KEYWORDS.keys()) == sorted(EXPECTED_KEYWORD_TAGS)


def test_keywords_non_empty():
    for tag, keywords in CODING_TAG_KEYWORDS.items():
        assert len(keywords) > 0, f"Tag '{tag}' has empty keywords"


def test_patterns_have_matching_keywords():
    """Every tag with patterns should also have keywords."""
    for tag in CODING_TAG_PATTERNS:
        assert tag in CODING_TAG_KEYWORDS, f"Tag '{tag}' has patterns but no keywords"


# ---------------------------------------------------------------------------
# Tag rules
# ---------------------------------------------------------------------------

def test_tag_rules_non_empty():
    assert len(CODING_TAG_RULES) > 0


def test_architecture_highest_priority():
    arch_rule = next(r for r in CODING_TAG_RULES if r["match"] == "architecture*")
    assert arch_rule["priority"] == 10


def test_debugging_short_ttl():
    debug_rule = next(r for r in CODING_TAG_RULES if r["match"] == "debugging*")
    assert debug_rule["ttl_days"] == 7


def test_catch_all_rule_exists():
    catch_all = next(r for r in CODING_TAG_RULES if r["match"] == "*")
    assert catch_all["priority"] == 5


# ---------------------------------------------------------------------------
# Config dict validity
# ---------------------------------------------------------------------------

def test_coding_config_loads():
    config = load_config(config_dict=CODING_CONFIG)
    assert config.tag_generator.type == "llm"
    assert config.tag_generator.provider == "ollama"


def test_coding_config_validates():
    config = load_config(config_dict=CODING_CONFIG)
    errors = validate_config(config)
    assert errors == [], f"Validation errors: {errors}"


def test_coding_tuned_thresholds():
    config = load_config(config_dict=CODING_CONFIG)
    assert config.monitor.soft_threshold == 0.60
    assert config.monitor.hard_threshold == 0.80
    assert config.monitor.protected_recent_turns == 8


def test_coding_storage_sqlite():
    config = load_config(config_dict=CODING_CONFIG)
    assert config.storage.backend == "sqlite"


def test_coding_cost_tracking():
    config = load_config(config_dict=CODING_CONFIG)
    assert config.cost_tracking.enabled is True


# ---------------------------------------------------------------------------
# Template round-trip
# ---------------------------------------------------------------------------

def test_template_parses_as_yaml():
    parsed = yaml.safe_load(CODING_TEMPLATE)
    assert isinstance(parsed, dict)
    assert "tag_generator" in parsed
    assert "tag_rules" in parsed


def test_template_roundtrip_loads_and_validates():
    parsed = yaml.safe_load(CODING_TEMPLATE)
    config = load_config(config_dict=parsed)
    errors = validate_config(config)
    assert errors == [], f"Template round-trip validation errors: {errors}"


def test_template_roundtrip_thresholds_match():
    parsed = yaml.safe_load(CODING_TEMPLATE)
    config = load_config(config_dict=parsed)
    assert config.monitor.soft_threshold == 0.60
    assert config.monitor.hard_threshold == 0.80
    assert config.monitor.protected_recent_turns == 8


def test_template_has_tag_generator():
    parsed = yaml.safe_load(CODING_TEMPLATE)
    assert parsed["tag_generator"]["type"] == "llm"
    assert parsed["tag_generator"]["provider"] == "ollama"


def test_template_regex_patterns_compile():
    """All regex patterns in the YAML template should compile after parsing."""
    parsed = yaml.safe_load(CODING_TEMPLATE)
    tag_gen = parsed.get("tag_generator", {})
    fallback = tag_gen.get("keyword_fallback", {})
    for tag, patterns in fallback.get("tag_patterns", {}).items():
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise AssertionError(
                    f"Tag '{tag}' has invalid regex: {pattern!r} — {exc}"
                )
