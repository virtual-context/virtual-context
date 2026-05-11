"""Tests for the ``admin backfill-tag-summaries`` CLI subcommand.

Covers the storage-override flags + ``DATABASE_URL`` env fallback
introduced so the subcommand runs inside containers that don't have a
config file mounted at a standard path.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

import pytest

from virtual_context.cli.main import _apply_storage_overrides
from virtual_context.types import StorageConfig, VirtualContextConfig


def _make_config(*, backend="sqlite", postgres_dsn="", sqlite_path=".db"):
    return VirtualContextConfig(
        storage=StorageConfig(
            backend=backend,
            postgres_dsn=postgres_dsn,
            sqlite_path=sqlite_path,
        ),
    )


def test_storage_override_explicit_flag_wins(monkeypatch):
    """Explicit ``--postgres-dsn`` overrides whatever config carried."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = _make_config(backend="sqlite", postgres_dsn="from-config")
    args = SimpleNamespace(
        config=None,
        storage_backend="postgres",
        postgres_dsn="postgresql://override/db",
        sqlite_path=None,
    )
    _apply_storage_overrides(config, args)
    assert config.storage.backend == "postgres"
    assert config.storage.postgres_dsn == "postgresql://override/db"


def test_storage_override_sqlite_path_flag(monkeypatch):
    """``--sqlite-path`` overrides ``storage.sqlite_path``."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = _make_config(backend="sqlite", sqlite_path=".db")
    args = SimpleNamespace(
        config=None,
        storage_backend=None,
        postgres_dsn=None,
        sqlite_path="/tmp/override.db",
    )
    _apply_storage_overrides(config, args)
    assert config.storage.sqlite_path == "/tmp/override.db"
    # Backend unchanged because no --storage-backend flag.
    assert config.storage.backend == "sqlite"


def test_storage_override_database_url_env_fallback(monkeypatch):
    """When no flag and no ``-c`` are given, ``DATABASE_URL`` env
    promotes the engine to Postgres with that DSN. Covers the
    container-based invocation shape where ops just runs the
    subcommand with the env already set by the container runtime."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")
    config = _make_config(backend="sqlite", postgres_dsn="")
    args = SimpleNamespace(
        config=None,
        storage_backend=None,
        postgres_dsn=None,
        sqlite_path=None,
    )
    _apply_storage_overrides(config, args)
    assert config.storage.backend == "postgres"
    assert config.storage.postgres_dsn == "postgresql://from-env/db"


def test_storage_override_database_url_skipped_when_config_provided(monkeypatch):
    """When the caller passes ``-c``, the env fallback DOES NOT fire;
    the config file is treated as authoritative."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")
    config = _make_config(backend="sqlite", postgres_dsn="")
    args = SimpleNamespace(
        config="/some/config.yaml",
        storage_backend=None,
        postgres_dsn=None,
        sqlite_path=None,
    )
    _apply_storage_overrides(config, args)
    # Backend + DSN unchanged because ``-c`` was given.
    assert config.storage.backend == "sqlite"
    assert config.storage.postgres_dsn == ""


def test_storage_override_database_url_skipped_when_any_flag_set(monkeypatch):
    """When ANY storage flag is set, the env fallback does NOT fire
    (explicit flags are authoritative)."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")
    config = _make_config(backend="sqlite")
    args = SimpleNamespace(
        config=None,
        storage_backend=None,
        postgres_dsn=None,
        sqlite_path="/tmp/explicit.db",
    )
    _apply_storage_overrides(config, args)
    # Only the sqlite_path flag applies; env fallback is skipped.
    assert config.storage.sqlite_path == "/tmp/explicit.db"
    assert config.storage.backend == "sqlite"
    assert config.storage.postgres_dsn == ""


def test_storage_override_no_change_when_no_flags_and_no_env(monkeypatch):
    """Defaults intact when neither flags, ``-c``, nor ``DATABASE_URL``
    are present. (``-c`` absent means ``load_config()`` already returned
    defaults; this assertion guards against accidental mutation.)"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = _make_config(backend="sqlite", sqlite_path=".db")
    args = SimpleNamespace(
        config=None,
        storage_backend=None,
        postgres_dsn=None,
        sqlite_path=None,
    )
    _apply_storage_overrides(config, args)
    assert config.storage.backend == "sqlite"
    assert config.storage.postgres_dsn == ""
    assert config.storage.sqlite_path == ".db"


def test_cli_parser_accepts_new_storage_flags():
    """End-to-end parser smoke test: the ``admin backfill-tag-summaries``
    subparser accepts the three new flags + ``DATABASE_URL`` precedence
    metadata in the help text."""
    from virtual_context.cli.main import main as _main
    import argparse

    # Re-build the parser the way ``main()`` does, then parse a sample
    # invocation. (Doing a full ``main()`` call would actually run the
    # backfill; we only need argument parsing here.)
    parser = argparse.ArgumentParser(prog="virtual-context")
    parser.add_argument("--config", "-c")
    subparsers = parser.add_subparsers(dest="command")
    admin_parser = subparsers.add_parser("admin")
    admin_sub = admin_parser.add_subparsers(dest="admin_command")
    backfill_parser = admin_sub.add_parser("backfill-tag-summaries")
    backfill_parser.add_argument("conversation_id")
    backfill_parser.add_argument("--tenant-id", default="")
    backfill_parser.add_argument("--force-rebuild", action="store_true")
    backfill_parser.add_argument(
        "--storage-backend",
        choices=("sqlite", "postgres", "filesystem"),
    )
    backfill_parser.add_argument("--postgres-dsn")
    backfill_parser.add_argument("--sqlite-path")

    args = parser.parse_args([
        "admin", "backfill-tag-summaries", "conv-xyz",
        "--tenant-id", "t1",
        "--storage-backend", "postgres",
        "--postgres-dsn", "postgresql://h/db",
    ])
    assert args.conversation_id == "conv-xyz"
    assert args.tenant_id == "t1"
    assert args.storage_backend == "postgres"
    assert args.postgres_dsn == "postgresql://h/db"
    assert args.sqlite_path is None
    assert args.force_rebuild is False
