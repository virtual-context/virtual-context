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


def test_daemon_install_creates_home_config(tmp_cwd):
    """daemon install should auto-create ~/.virtualcontext/config.yaml with absolute paths."""
    fake_home = tmp_cwd / "fakehome"
    fake_home.mkdir()
    env = {**__import__("os").environ, "HOME": str(fake_home)}
    result = subprocess.run(
        [sys.executable, "-m", "virtual_context.cli.main", "daemon", "install"],
        capture_output=True,
        text=True,
        env=env,
    )
    config_path = fake_home / ".virtualcontext" / "config.yaml"
    assert config_path.exists(), "daemon install should create ~/.virtualcontext/config.yaml"
    assert "Created config" in result.stdout
    content = config_path.read_text()
    assert "tag_generator" in content
    # Storage paths should be absolute, not relative
    assert str(fake_home / ".virtualcontext") in content
    assert str(fake_home / ".virtualcontext" / "store.db") in content


def test_daemon_install_preserves_existing_home_config(tmp_cwd):
    """daemon install should not overwrite an existing home config."""
    fake_home = tmp_cwd / "fakehome"
    vc_dir = fake_home / ".virtualcontext"
    vc_dir.mkdir(parents=True)
    config_path = vc_dir / "config.yaml"
    config_path.write_text("custom: true\n")
    env = {**__import__("os").environ, "HOME": str(fake_home)}
    result = subprocess.run(
        [sys.executable, "-m", "virtual_context.cli.main", "daemon", "install"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert config_path.read_text() == "custom: true\n"
    assert "Created config" not in result.stdout


def test_daemon_install_uses_explicit_config(tmp_cwd):
    """daemon install -c ./my.yaml should use that path, not ~/.virtualcontext/."""
    config_path = tmp_cwd / "my.yaml"
    config_path.write_text("custom: explicit\n")
    result = _run_cli("daemon", "install", "-c", str(config_path))
    assert "Created config" not in result.stdout


def test_discover_config_falls_back_to_home(tmp_cwd, monkeypatch):
    """_discover_config() should find ~/.virtualcontext/config.yaml as last resort."""
    from virtual_context.config import _discover_config

    fake_home = tmp_cwd / "fakehome"
    vc_dir = fake_home / ".virtualcontext"
    vc_dir.mkdir(parents=True)
    config_path = vc_dir / "config.yaml"
    config_path.write_text("version: '0.2'\n")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))
    # CWD is tmp_cwd (no config files there)
    result = _discover_config()
    assert result == config_path


def test_discover_config_prefers_local_over_home(tmp_cwd, monkeypatch):
    """A local config in CWD should take priority over ~/.virtualcontext/config.yaml."""
    from virtual_context.config import _discover_config

    fake_home = tmp_cwd / "fakehome"
    vc_dir = fake_home / ".virtualcontext"
    vc_dir.mkdir(parents=True)
    (vc_dir / "config.yaml").write_text("home config\n")
    local_config = tmp_cwd / "virtual-context.yaml"
    local_config.write_text("local config\n")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))
    result = _discover_config()
    assert result == local_config


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
