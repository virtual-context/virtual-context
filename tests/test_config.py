"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from virtual_context.config import load_config, validate_config


def test_load_defaults():
    config = load_config(config_dict={})
    assert config.context_window == 120_000
    assert "_general" in config.domains
    assert config.monitor.soft_threshold == 0.70
    assert config.monitor.hard_threshold == 0.85


def test_load_with_domains():
    config = load_config(config_dict={
        "domains": {
            "legal": {
                "description": "Legal matters",
                "keywords": ["court", "attorney"],
                "priority": 9,
            },
        },
    })
    assert "legal" in config.domains
    assert config.domains["legal"].priority == 9
    assert "_general" in config.domains


def test_load_from_yaml_file():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump({
            "context_window": 50000,
            "domains": {
                "code": {
                    "description": "Programming",
                    "keywords": ["function", "bug"],
                },
            },
        }, f)
        f.flush()
        config = load_config(config_path=f.name)

    assert config.context_window == 50000
    assert "code" in config.domains


def test_validate_valid_config():
    config = load_config(config_dict={
        "domains": {"legal": {"keywords": ["court"]}},
        "classifier": {"pipeline": [{"type": "keyword"}]},
    })
    errors = validate_config(config)
    assert errors == []


def test_validate_threshold_order():
    config = load_config(config_dict={
        "compaction": {
            "soft_threshold": 0.9,
            "hard_threshold": 0.7,
        },
    })
    errors = validate_config(config)
    assert any("soft_threshold" in e for e in errors)


def test_validate_protected_turns():
    config = load_config(config_dict={
        "compaction": {"protected_recent_turns": 0},
    })
    errors = validate_config(config)
    assert any("protected_recent_turns" in e for e in errors)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(config_path="/nonexistent/path.yaml")
