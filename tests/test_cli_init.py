"""Tests for the `virtual-context init` CLI command."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.fixture()
def tmp_cwd(tmp_path, monkeypatch):
    """Run test in a clean temporary directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "virtual_context.cli.main", *args],
        capture_output=True,
        text=True,
    )


def test_init_coding_creates_config(tmp_cwd):
    result = _run_cli("init", "coding")
    assert result.returncode == 0
    config_path = tmp_cwd / "virtual-context.yaml"
    assert config_path.exists()
    content = config_path.read_text()
    assert "tag_generator:" in content
    assert "tag_rules:" in content
    assert "architecture" in content


def test_init_refuses_overwrite(tmp_cwd):
    (tmp_cwd / "virtual-context.yaml").write_text("existing content")
    result = _run_cli("init", "coding")
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert (tmp_cwd / "virtual-context.yaml").read_text() == "existing content"


def test_init_force_overwrites(tmp_cwd):
    (tmp_cwd / "virtual-context.yaml").write_text("existing content")
    result = _run_cli("init", "coding", "--force")
    assert result.returncode == 0
    content = (tmp_cwd / "virtual-context.yaml").read_text()
    assert "tag_generator:" in content
    assert content != "existing content"


def test_init_unknown_preset(tmp_cwd):
    result = _run_cli("init", "nonexistent")
    assert result.returncode != 0
    assert "Unknown preset" in result.stderr
