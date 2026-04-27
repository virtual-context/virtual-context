"""Tenant_id propagation tests for the conversations table (v1.16-1 prod fold).

Per codex iter-5 prod blocker (cloud E2E surfaced: every existing user conv
plus every freshly-created conv had ``conversations.tenant_id = ''``,
which broke VCMERGE Layer C tenant validation with HTTP 403):

1. ``VirtualContextConfig.tenant_id`` field carries the request-scope
   tenant context to every engine-internal ``upsert_conversation`` call.
2. The engine's ``_load_lifecycle_epoch_into_engine_state`` and
   ``sync_turns_from_payload`` paths now pass ``self.config.tenant_id``
   instead of the empty-string placeholder.
3. PostgreSQL ``_ensure_schema`` runs a one-time backfill UPDATE from
   ``cloud_conversations.tenant_id`` on bootstrap (DATABASE_URL gated;
   skipped silently when ``cloud_conversations`` is absent).

Tests pin the structural invariants behaviorally where possible and via
inspection where the cloud/PG cross-table join can't be exercised on
SQLite alone.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") so the
entire backstory is searchable.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import VirtualContextConfig


pytestmark = pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26")


# ---------------------------------------------------------------------------
# Config field: VirtualContextConfig.tenant_id is a string; defaults to ''.
# ---------------------------------------------------------------------------

def test_config_tenant_id_field_defaults_to_empty_string():
    cfg = VirtualContextConfig()
    assert hasattr(cfg, "tenant_id")
    assert cfg.tenant_id == ""


def test_config_tenant_id_field_round_trips_via_yaml(tmp_path):
    """Config builder accepts ``tenant_id`` in raw yaml dict."""
    from virtual_context.config import _build_config
    cfg = _build_config({"tenant_id": "tenant-xyz"})
    assert cfg.tenant_id == "tenant-xyz"


def test_config_tenant_id_optional_in_yaml(tmp_path):
    """No ``tenant_id`` key in yaml means empty default (single-user dev)."""
    from virtual_context.config import _build_config
    cfg = _build_config({})
    assert cfg.tenant_id == ""


# ---------------------------------------------------------------------------
# Engine populates conversations.tenant_id from config when creating rows.
# ---------------------------------------------------------------------------

def test_upsert_conversation_writes_tenant_id_from_caller(tmp_path):
    """``Store.upsert_conversation`` writes the caller-supplied tenant_id
    into the conversations row (the storage primitive has always supported
    this; the bug was at engine call sites that passed '')."""
    store = SQLiteStore(tmp_path / "store.db")
    conv_id = "test-conv-" + uuid.uuid4().hex[:8]
    store.upsert_conversation(tenant_id="tenant-A", conversation_id=conv_id)
    conn = store._get_conn()
    row = conn.execute(
        "SELECT tenant_id FROM conversations WHERE conversation_id = ?",
        (conv_id,),
    ).fetchone()
    assert row["tenant_id"] == "tenant-A"


def test_engine_lifecycle_load_uses_config_tenant_id(tmp_path):
    """Structural pin (v1.16-1): ``Engine._load_lifecycle_epoch_into_engine_state``
    no longer hard-codes ``tenant_id=""`` in its upsert_conversation call;
    the call sources tenant_id from ``self.config.tenant_id``.
    """
    import inspect
    from virtual_context.engine import VirtualContextEngine
    src = inspect.getsource(
        VirtualContextEngine._load_lifecycle_epoch_into_engine_state
    )
    # The literal hard-coded empty placeholder is gone.
    assert 'tenant_id=""' not in src, (
        "v1.16-1: Engine._load_lifecycle_epoch_into_engine_state still "
        "hard-codes tenant_id=\"\"; tenant scoping regression"
    )
    # The config-derived form is present.
    assert "self.config.tenant_id" in src or "tenant_id=tenant_id" in src, (
        "v1.16-1: Engine._load_lifecycle_epoch_into_engine_state should "
        "source tenant_id from self.config.tenant_id"
    )


def test_engine_sync_turns_uses_config_tenant_id(tmp_path):
    """Structural pin (v1.16-1): ``Engine.sync_turns_from_payload`` no
    longer hard-codes ``tenant_id=""`` in its upsert_conversation call.
    """
    import inspect
    from virtual_context.engine import VirtualContextEngine
    src = inspect.getsource(VirtualContextEngine.sync_turns_from_payload)
    assert 'tenant_id=""' not in src, (
        "v1.16-1: Engine.sync_turns_from_payload still hard-codes "
        "tenant_id=\"\"; tenant scoping regression"
    )
    assert "self.config.tenant_id" in src, (
        "v1.16-1: Engine.sync_turns_from_payload should source tenant_id "
        "from self.config.tenant_id"
    )


def test_engine_creates_conversations_row_with_config_tenant_id(tmp_path):
    """End-to-end: instantiate engine with ``cfg.tenant_id`` set; the
    bootstrap-created conversations row carries that tenant_id (was '' in
    pre-fold prod, breaking VCMERGE tenant scoping)."""
    from virtual_context.engine import VirtualContextEngine
    cfg = VirtualContextConfig(
        tenant_id="tenant-prod-fix",
        conversation_id="conv-test-" + uuid.uuid4().hex[:8],
    )
    cfg.storage.backend = "sqlite"
    cfg.storage.sqlite_path = str(tmp_path / "store.db")
    engine = VirtualContextEngine(config=cfg)
    try:
        store = engine._store
        # Drill into composite/inner stores to reach the SQLite path.
        inner = getattr(store, "_segments", None) or getattr(store, "_store", None) or store
        conn = inner._get_conn()
        row = conn.execute(
            "SELECT tenant_id FROM conversations WHERE conversation_id = ?",
            (cfg.conversation_id,),
        ).fetchone()
        assert row is not None, "engine bootstrap did not create conversations row"
        assert row["tenant_id"] == "tenant-prod-fix", (
            f"conversations.tenant_id should be 'tenant-prod-fix' but got "
            f"{row['tenant_id']!r}; pre-fold regression"
        )
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# v1.16-1 backfill behavior: source tenant_id from cloud_conversations.
# Structural-only on SQLite (no cloud_conversations table); behavioral
# in tests/test_conversations_tenant_id_backfill_postgres.py against PG.
# ---------------------------------------------------------------------------

def test_pg_backfill_sql_present_in_ensure_schema():
    """v1.16-1 structural pin: ``PostgresStore._ensure_schema`` must contain
    the cloud_conversations backfill UPDATE. A future regression that drops
    the migration would fail this test before reaching staging."""
    import inspect
    from virtual_context.storage import postgres as postgres_mod
    src = inspect.getsource(postgres_mod.PostgresStore._ensure_schema)
    # The backfill SQL fingerprint: "FROM cloud_conversations" + tenant_id update target.
    assert "FROM cloud_conversations" in src, (
        "v1.16-1: PostgresStore._ensure_schema missing cloud_conversations "
        "backfill UPDATE; existing prod convs would stay with empty tenant_id"
    )
    assert "UPDATE conversations" in src, (
        "v1.16-1: PostgresStore._ensure_schema backfill missing UPDATE "
        "conversations target"
    )
    # Idempotency safeguard: WHERE filters on empty target.
    assert (
        "conversations.tenant_id = ''" in src
        or 'conversations.tenant_id IS NULL' in src
    ), (
        "v1.16-1: backfill UPDATE missing the empty-source idempotency "
        "filter; would re-write tenant_id on every schema bootstrap"
    )
    # Best-effort gate: UndefinedTable swallowed.
    assert "UndefinedTable" in src, (
        "v1.16-1: backfill should swallow only psycopg.errors.UndefinedTable "
        "(missing cloud_conversations on engine-only deploys); other "
        "exceptions must propagate"
    )
