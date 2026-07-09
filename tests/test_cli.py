"""CLI surface tests for the ``admin`` subcommands.

These assert the argument *shape* of ``admin backfill-fact-embeddings``
(positional + flags) matches its sibling ``admin
backfill-tag-summaries`` per the dense-fact-retrieval plan, without
constructing a real engine.
"""

from __future__ import annotations

import sys

import virtual_context.cli.main as cli_main


def _parse_admin(monkeypatch, argv, handler_name):
    """Drive ``cli_main.main()`` with the given argv, capturing the
    parsed ``args`` namespace handed to *handler_name* instead of
    running the real command handler."""
    captured = {}

    def _capture(args):
        captured["args"] = args

    monkeypatch.setattr(cli_main, handler_name, _capture)
    monkeypatch.setattr(sys, "argv", ["virtual-context", *argv])
    cli_main.main()
    return captured["args"]


def test_admin_backfill_fact_embeddings_cli_shape_matches_backfill_tag_summaries(
    monkeypatch,
):
    fact_args = _parse_admin(
        monkeypatch,
        [
            "admin", "backfill-fact-embeddings", "conv-xyz",
            "--since", "2026-01-01",
            "--until", "2026-02-01",
            "--force-rebuild",
            "--tenant-id", "tenant-7",
            "--storage-backend", "sqlite",
            "--postgres-dsn", "postgresql://x/y",
            "--sqlite-path", "/tmp/x.db",
        ],
        "cmd_admin_backfill_fact_embeddings",
    )

    # Positional conversation_id (NOT --conversation-id) and the
    # locked flag names (--force-rebuild, NOT --force).
    assert fact_args.conversation_id == "conv-xyz"
    assert fact_args.since == "2026-01-01"
    assert fact_args.until == "2026-02-01"
    assert fact_args.force_rebuild is True
    assert fact_args.tenant_id == "tenant-7"
    assert fact_args.storage_backend == "sqlite"
    assert fact_args.postgres_dsn == "postgresql://x/y"
    assert fact_args.sqlite_path == "/tmp/x.db"

    # Storage-override + operational flags are identical to the sibling.
    ts_args = _parse_admin(
        monkeypatch,
        [
            "admin", "backfill-tag-summaries", "conv-xyz",
            "--force-rebuild",
            "--tenant-id", "tenant-7",
            "--storage-backend", "sqlite",
            "--postgres-dsn", "postgresql://x/y",
            "--sqlite-path", "/tmp/x.db",
        ],
        "cmd_admin_backfill_tag_summaries",
    )
    shared = ("conversation_id", "tenant_id", "force_rebuild",
              "storage_backend", "postgres_dsn", "sqlite_path")
    for name in shared:
        assert getattr(fact_args, name) == getattr(ts_args, name)


def test_admin_backfill_fact_embeddings_defaults(monkeypatch):
    args = _parse_admin(
        monkeypatch,
        ["admin", "backfill-fact-embeddings", "conv-only"],
        "cmd_admin_backfill_fact_embeddings",
    )
    assert args.conversation_id == "conv-only"
    assert args.since is None
    assert args.until is None
    assert args.force_rebuild is False
    assert args.storage_backend is None
    assert args.postgres_dsn is None
    assert args.sqlite_path is None
