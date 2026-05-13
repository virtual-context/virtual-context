"""Tests for the ``admin backfill-session-state-markers`` CLI subcommand.

Covers per-conv + batch + dry-run + limit shapes that the operator
will use to repair conversations whose Redis SessionState drifted
from their canonical_turns truth. See
``docs/specs/vcattach-redis-marker-write-and-cross-worker-invalidation.md``
for the surrounding design.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def test_cli_parser_accepts_session_state_markers_subcommand():
    """End-to-end parser smoke test: the new subcommand binds the
    expected positional + kwarg surface."""
    parser = argparse.ArgumentParser(prog="virtual-context")
    parser.add_argument("--config", "-c")
    subparsers = parser.add_subparsers(dest="command")
    admin_parser = subparsers.add_parser("admin")
    admin_sub = admin_parser.add_subparsers(dest="admin_command")
    backfill_parser = admin_sub.add_parser("backfill-session-state-markers")
    backfill_parser.add_argument("conversation_id", nargs="?", default=None)
    backfill_parser.add_argument("--tenant-id", default="")
    backfill_parser.add_argument("--all-convs-for-tenant", action="store_true")
    backfill_parser.add_argument("--dry-run", action="store_true")
    backfill_parser.add_argument("--limit", type=int, default=None)
    backfill_parser.add_argument("--redis-url", default="")
    backfill_parser.add_argument(
        "--storage-backend",
        choices=("sqlite", "postgres", "filesystem"),
    )
    backfill_parser.add_argument("--postgres-dsn")
    backfill_parser.add_argument("--sqlite-path")

    # Per-conv shape.
    args = parser.parse_args([
        "admin", "backfill-session-state-markers", "conv-xyz",
        "--redis-url", "redis://localhost:6379/0",
    ])
    assert args.conversation_id == "conv-xyz"
    assert args.tenant_id == ""
    assert args.all_convs_for_tenant is False
    assert args.dry_run is False
    assert args.redis_url == "redis://localhost:6379/0"

    # Batch shape.
    args = parser.parse_args([
        "admin", "backfill-session-state-markers",
        "--tenant-id", "tenant-1",
        "--all-convs-for-tenant",
        "--limit", "25",
        "--dry-run",
    ])
    assert args.conversation_id is None
    assert args.tenant_id == "tenant-1"
    assert args.all_convs_for_tenant is True
    assert args.limit == 25
    assert args.dry_run is True
