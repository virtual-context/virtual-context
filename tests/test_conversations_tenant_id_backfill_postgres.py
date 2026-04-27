"""PG smoke for v1.16-1 conversations.tenant_id backfill from cloud_conversations.

Skipped when DATABASE_URL is absent (the standard project pattern at
tests/test_vcmerge_schema_postgres.py:28). Verifies the one-time
backfill UPDATE that ``PostgresStore._ensure_schema`` runs on bootstrap
correctly populates ``conversations.tenant_id`` from
``cloud_conversations.tenant_id`` for rows that existed before the
insertion-path fix landed.

Per codex iter-5 prod blocker: 14 existing user convs in prod tenant
``43bd6d7f0f8d6798`` had ``conversations.tenant_id = ''`` (engine
predecessor passed empty placeholder); cloud_conversations was correctly
populated by TenantMiddleware. The migration needs to be idempotent so
re-running schema bootstrap is safe.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26").
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="DATABASE_URL not set: Postgres smoke skipped",
    ),
    pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26"),
]


@pytest.fixture
def pg_store():
    """Build a PostgresStore against DATABASE_URL. Cleanup teardown drops
    test rows under the test tenant prefix."""
    from virtual_context.storage.postgres import PostgresStore
    dsn = os.environ["DATABASE_URL"]
    store = PostgresStore(dsn)
    yield store
    try:
        conn = store._get_conn()
        conn.execute("DELETE FROM conversations WHERE tenant_id LIKE 'pgbf-%' "
                     "OR conversation_id LIKE 'pgbf-%'")
        try:
            conn.execute("DELETE FROM cloud_conversations WHERE tenant_id LIKE 'pgbf-%' "
                         "OR conversation_id LIKE 'pgbf-%'")
        except Exception:
            pass
    except Exception:
        pass


def _now():
    return datetime.now(timezone.utc)


def _ensure_cloud_conversations_table(conn):
    """If the test environment's PG doesn't have cloud_conversations
    (pure-engine deploys), create a minimal version sufficient for the
    backfill smoke. The real cloud schema is richer; this stub just has
    the columns the backfill UPDATE references.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cloud_conversations (
            conversation_id TEXT NOT NULL,
            tenant_id       TEXT NOT NULL DEFAULT '',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def test_pg_backfill_populates_empty_tenant_id_from_cloud_conversations(pg_store):
    """v1.16-1: simulate the prod state (conversations row with empty
    tenant_id; matching cloud_conversations row with the right tenant_id).
    Run schema bootstrap; assert backfill populated the empty cell.
    """
    conn = pg_store._get_conn()
    _ensure_cloud_conversations_table(conn)

    tid = f"pgbf-{uuid.uuid4().hex[:8]}"
    cid = f"pgbf-conv-{uuid.uuid4().hex[:8]}"
    # Pre-fold prod state: empty tenant_id on conversations.
    conn.execute(
        "INSERT INTO conversations (conversation_id, tenant_id, phase, "
        "created_at, updated_at) VALUES (%s, '', 'active', %s, %s)",
        (cid, _now(), _now()),
    )
    # Cloud's wrapper has the right tenant_id.
    conn.execute(
        "INSERT INTO cloud_conversations (conversation_id, tenant_id) "
        "VALUES (%s, %s)",
        (cid, tid),
    )
    # Re-trigger schema bootstrap (which runs the backfill UPDATE).
    pg_store._ensure_schema()
    # Backfill should have populated the empty cell from cloud_conversations.
    row = conn.execute(
        "SELECT tenant_id FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    assert row["tenant_id"] == tid, (
        f"Backfill should have set conversations.tenant_id to {tid!r}, "
        f"got {row['tenant_id']!r}"
    )


def test_pg_backfill_is_idempotent(pg_store):
    """v1.16-1: second bootstrap run is a no-op (the WHERE filter on
    empty target makes the migration idempotent)."""
    conn = pg_store._get_conn()
    _ensure_cloud_conversations_table(conn)

    tid = f"pgbf-{uuid.uuid4().hex[:8]}"
    cid = f"pgbf-conv-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO conversations (conversation_id, tenant_id, phase, "
        "created_at, updated_at) VALUES (%s, '', 'active', %s, %s)",
        (cid, _now(), _now()),
    )
    conn.execute(
        "INSERT INTO cloud_conversations (conversation_id, tenant_id) "
        "VALUES (%s, %s)",
        (cid, tid),
    )
    pg_store._ensure_schema()  # First backfill
    # Manually mutate cloud_conversations to a different tenant; bootstrap
    # again. The WHERE clause filters on empty target so the row's
    # already-populated tenant_id should NOT be overwritten.
    conn.execute(
        "UPDATE cloud_conversations SET tenant_id = %s WHERE conversation_id = %s",
        (f"{tid}-mutated", cid),
    )
    pg_store._ensure_schema()  # Second run; should be no-op for this row
    final = conn.execute(
        "SELECT tenant_id FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()["tenant_id"]
    assert final == tid, (
        "Backfill should be idempotent; populated row's tenant_id must "
        "NOT be overwritten on subsequent bootstraps"
    )


def test_pg_backfill_skips_when_cloud_conversations_absent(pg_store, tmp_path):
    """v1.16-1: when ``cloud_conversations`` is absent (engine-only deploy),
    the backfill is silently skipped via the UndefinedTable narrow-catch.
    Bootstrap completes without raising.

    Validates the SAVEPOINT-via-nested-transaction pattern around the
    backfill UPDATE: the missing-table raise rolls back ONLY the savepoint
    and leaves the outer schema bootstrap alive.
    """
    conn = pg_store._get_conn()
    # Drop cloud_conversations if it exists (simulating engine-only deploy).
    try:
        conn.execute("DROP TABLE IF EXISTS cloud_conversations")
    except Exception:
        pass
    # Schema bootstrap must NOT raise.
    pg_store._ensure_schema()
    # And bootstrap should still create (or leave intact) other engine
    # tables, evidenced by being able to insert a conversations row.
    cid = f"pgbf-skipped-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO conversations (conversation_id, tenant_id, phase, "
        "created_at, updated_at) VALUES (%s, '', 'active', %s, %s)",
        (cid, _now(), _now()),
    )
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    assert row is not None
