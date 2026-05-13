"""Tests for the ``admin backfill-session-state-markers`` CLI subcommand.

Covers per-conv + batch + dry-run + limit shapes that the operator
will use to repair conversations whose Redis SessionState drifted
from their canonical_turns truth. See
``docs/specs/vcattach-redis-marker-write-and-cross-worker-invalidation.md``
for the surrounding design.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace


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


def test_cmd_admin_backfill_session_state_markers_dry_run_skips_provider(
    monkeypatch,
    capsys,
):
    """Function-level dry-run: derive + print markers without building
    a SessionStateProvider, even if --redis-url is present."""
    import virtual_context.cli.main as cli_main
    import virtual_context.core.state_recovery as state_recovery
    import virtual_context.engine as engine_module
    import virtual_context.proxy.session_state as session_state

    raw_store = object()
    closed: list[bool] = []
    derive_calls: list[tuple[object, str, object]] = []

    class StubEngine:
        def __init__(self, config):
            self._store = SimpleNamespace(_store=raw_store)

        def close(self):
            closed.append(True)

    class ExplodingProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run must not construct provider")

    def fake_derive(store, conversation_id, *, existing_state=None):
        derive_calls.append((store, conversation_id, existing_state))
        return SimpleNamespace(
            compacted_prefix_messages=4,
            flushed_prefix_messages=4,
            last_completed_turn=3,
            last_indexed_turn=3,
            turn_tag_entries=[{"turn_number": 0}],
        )

    monkeypatch.setattr(cli_main, "load_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(cli_main, "_apply_storage_overrides", lambda config, args: None)
    monkeypatch.setattr(engine_module, "VirtualContextEngine", StubEngine)
    monkeypatch.setattr(session_state, "SessionStateProvider", ExplodingProvider)
    monkeypatch.setattr(state_recovery, "derive_session_state_markers", fake_derive)

    args = SimpleNamespace(
        config=None,
        conversation_id="conv-dry",
        tenant_id="",
        all_convs_for_tenant=False,
        dry_run=True,
        limit=None,
        redis_url="redis://unused",
    )

    cli_main.cmd_admin_backfill_session_state_markers(args)

    lines = [
        json.loads(line)
        for line in capsys.readouterr().out.strip().splitlines()
    ]
    assert lines[0]["status"] == "dry_run"
    assert lines[0]["conversation_id"] == "conv-dry"
    assert lines[0]["saved"] is False
    assert lines[1]["status"] == "complete"
    assert lines[1]["processed"] == 1
    assert lines[1]["saved"] == 0
    assert derive_calls == [(raw_store, "conv-dry", None)]
    assert closed == [True]


def test_cmd_admin_backfill_session_state_markers_batch_filters_tenant(
    monkeypatch,
    capsys,
):
    """The --all-convs-for-tenant path enumerates storage stats, filters
    through the tenant-aware liveness predicate, then applies --limit."""
    import virtual_context.cli.main as cli_main
    import virtual_context.core.state_recovery as state_recovery
    import virtual_context.engine as engine_module
    import virtual_context.proxy.session_state as session_state

    class RawStore:
        def get_conversation_stats(self):
            return [
                SimpleNamespace(conversation_id="conv-a"),
                SimpleNamespace(conversation_id="conv-b"),
                SimpleNamespace(conversation_id="conv-c"),
            ]

        def is_attachable_target(self, *, conversation_id, tenant_id=None):
            return tenant_id == "tenant-1" and conversation_id in {"conv-a", "conv-c"}

    raw_store = RawStore()
    constructed_configs: list[SimpleNamespace] = []

    class StubEngine:
        def __init__(self, config):
            constructed_configs.append(config)
            self._store = SimpleNamespace(_store=raw_store)

        def close(self):
            pass

    provider_instances: list[object] = []

    class StubProvider:
        def __init__(self, redis_url):
            self.redis_url = redis_url
            self.loaded: list[str] = []
            self.saved: list[tuple[str, object]] = []
            provider_instances.append(self)

        def load(self, conversation_id):
            self.loaded.append(conversation_id)
            return SimpleNamespace(version=5, checkpoint_version=9)

        def save(self, conversation_id, state):
            self.saved.append((conversation_id, state))
            return 6

    derived_for: list[tuple[str, object]] = []

    def fake_derive(store, conversation_id, *, existing_state=None):
        assert store is raw_store
        derived_for.append((conversation_id, existing_state))
        return SimpleNamespace(
            compacted_prefix_messages=2,
            flushed_prefix_messages=2,
            last_completed_turn=1,
            last_indexed_turn=1,
            turn_tag_entries=[{"turn_number": 0}],
        )

    monkeypatch.setattr(cli_main, "load_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(cli_main, "_apply_storage_overrides", lambda config, args: None)
    monkeypatch.setattr(engine_module, "VirtualContextEngine", StubEngine)
    monkeypatch.setattr(session_state, "SessionStateProvider", StubProvider)
    monkeypatch.setattr(state_recovery, "derive_session_state_markers", fake_derive)

    args = SimpleNamespace(
        config=None,
        conversation_id=None,
        tenant_id="tenant-1",
        all_convs_for_tenant=True,
        dry_run=False,
        limit=2,
        redis_url="redis://localhost:6379/0",
    )

    cli_main.cmd_admin_backfill_session_state_markers(args)

    provider = provider_instances[0]
    assert provider.redis_url == "redis://localhost:6379/0"
    assert provider.loaded == ["conv-a", "conv-c"]
    assert [cid for cid, _ in provider.saved] == ["conv-a", "conv-c"]
    assert [cid for cid, _ in derived_for] == ["conv-a", "conv-c"]
    assert constructed_configs[0].tenant_id == "tenant-1"

    lines = [
        json.loads(line)
        for line in capsys.readouterr().out.strip().splitlines()
    ]
    assert [line["status"] for line in lines[:2]] == ["ok", "ok"]
    assert lines[2]["status"] == "complete"
    assert lines[2]["processed"] == 2
    assert lines[2]["saved"] == 2
