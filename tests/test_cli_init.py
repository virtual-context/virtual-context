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


def test_onboard_creates_and_validates_config(tmp_cwd):
    result = _run_cli("onboard")
    assert result.returncode == 0
    config_path = tmp_cwd / "virtual-context.yaml"
    assert config_path.exists()
    assert "Config is valid." in result.stdout
    assert "Onboarding complete." in result.stdout


def test_onboard_uses_existing_config(tmp_cwd):
    _run_cli("init", "coding")
    result = _run_cli("onboard")
    assert result.returncode == 0
    assert "Using existing config" in result.stdout


def test_onboard_unknown_preset(tmp_cwd):
    result = _run_cli("onboard", "--preset", "nonexistent")
    assert result.returncode != 0
    assert "Unknown preset" in result.stderr


def test_daemon_help(tmp_cwd):
    result = _run_cli("daemon", "--help")
    assert result.returncode == 0
    assert "status" in result.stdout
    assert "uninstall" in result.stdout


def test_daemon_requires_action(tmp_cwd):
    result = _run_cli("daemon")
    assert result.returncode != 0


def test_daemon_restart_in_help(tmp_cwd):
    result = _run_cli("daemon", "--help")
    assert result.returncode == 0
    assert "restart" in result.stdout


def test_daemon_not_installed_guard(tmp_cwd):
    """daemon commands should fail with a helpful message when no daemon is installed."""
    env = {**__import__("os").environ, "HOME": str(tmp_cwd)}
    for action in ("status", "start", "stop", "restart"):
        result = subprocess.run(
            [sys.executable, "-m", "virtual_context.cli.main", "daemon", action],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0, f"daemon {action} should fail when not installed"
        assert "not installed" in result.stdout.lower(), (
            f"daemon {action} should mention 'not installed'"
        )
        assert "daemon install" in result.stdout, (
            f"daemon {action} should suggest 'daemon install'"
        )


def test_daemon_install_creates_config(tmp_cwd):
    """daemon install should auto-create config from agentic preset if none exists."""
    config_path = tmp_cwd / "virtual-context.yaml"
    assert not config_path.exists()
    # daemon install will fail on launchctl/systemctl, but should create config first
    result = _run_cli("daemon", "install")
    assert config_path.exists(), "daemon install should create config"
    assert "Created config" in result.stdout
    content = config_path.read_text()
    assert "tag_generator" in content


def test_daemon_install_preserves_existing_config(tmp_cwd):
    """daemon install should not overwrite an existing config."""
    config_path = tmp_cwd / "virtual-context.yaml"
    config_path.write_text("custom: true\n")
    result = _run_cli("daemon", "install")
    assert config_path.read_text() == "custom: true\n"
    assert "Created config" not in result.stdout


def test_presets_list(tmp_cwd):
    result = _run_cli("presets", "list")
    assert result.returncode == 0
    assert "coding" in result.stdout
    assert "agentic" in result.stdout


def test_presets_show_coding(tmp_cwd):
    result = _run_cli("presets", "show", "coding")
    assert result.returncode == 0
    assert "tag_generator" in result.stdout


def test_presets_show_agentic(tmp_cwd):
    result = _run_cli("presets", "show", "agentic")
    assert result.returncode == 0
    assert "tag_generator" in result.stdout


def test_presets_show_nonexistent(tmp_cwd):
    result = _run_cli("presets", "show", "nonexistent")
    assert result.returncode != 0
    assert "Unknown preset" in result.stderr
