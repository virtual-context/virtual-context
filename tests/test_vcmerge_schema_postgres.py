"""PG-parametrized smoke for VCMERGE Phase 0 schema ( fold).

Skipped when DATABASE_URL is absent (the standard project pattern at
tests/test_cleanup_abandoned_compaction_postgres.py:24). Verifies that
 + + + + + + migrations work against
a real Postgres backend, not just SQLite.

 + cloud's review:
SQLite-only test coverage missed PG-specific concerns (FOR UPDATE,
SAVEPOINT semantics, ADD COLUMN IF NOT EXISTS gating, trigger DDL
syntax). This file is the PG mirror of test_vcmerge_schema.py's
smoke set.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
.
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
    """Build a PostgresStore against DATABASE_URL. Returns the store;
    cleanup via teardown drops merge_audit + merge_post_commit_pending.
    """
    from virtual_context.storage.postgres import PostgresStore
    dsn = os.environ["DATABASE_URL"]
    store = PostgresStore(dsn)
    yield store
    # Teardown: clear test rows from the merge tables. Don't drop the
    # tables themselves; that would race with parallel test workers.
    try:
        conn = store._get_conn()
        conn.execute("DELETE FROM merge_post_commit_pending WHERE tenant_id LIKE 'pgtest-%'")
        conn.execute("DELETE FROM merge_audit WHERE tenant_id LIKE 'pgtest-%'")
        conn.execute("DELETE FROM conversations WHERE tenant_id LIKE 'pgtest-%'")
    except Exception:
        pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# : phase CHECK admits 'merged' on PG
# ---------------------------------------------------------------------------

def test_pg_conversations_phase_check_admits_merged(pg_store):
    conn = pg_store._get_conn()
    tid = f"pgtest-{uuid.uuid4()}"
    cid = f"conv-{uuid.uuid4()}"
    now = _now()
    conn.execute(
        "INSERT INTO conversations (conversation_id, tenant_id, phase, "
        "created_at, updated_at) VALUES (%s, %s, 'merged', %s, %s)",
        (cid, tid, now, now),
    )
    row = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    assert row["phase"] == "merged"


# ---------------------------------------------------------------------------
# + : merge_audit + unique partial index on PG
# ---------------------------------------------------------------------------

def test_pg_merge_audit_unique_partial_index_rejects_duplicate_in_progress(pg_store):
    conn = pg_store._get_conn()
    tid = f"pgtest-{uuid.uuid4()}"
    src = f"src-{uuid.uuid4()}"
    tgt = f"tgt-{uuid.uuid4()}"
    conn.execute(
        "INSERT INTO merge_audit (merge_id, tenant_id, source_conversation_id, "
        "target_conversation_id, status, started_at) VALUES (%s, %s, %s, %s, "
        "'in_progress', %s)",
        (str(uuid.uuid4()), tid, src, tgt, _now()),
    )
    import psycopg
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO merge_audit (merge_id, tenant_id, "
            "source_conversation_id, target_conversation_id, status, "
            "started_at) VALUES (%s, %s, %s, %s, 'in_progress', %s)",
            (str(uuid.uuid4()), tid, src, tgt, _now()),
        )


# ---------------------------------------------------------------------------
# : tenant-consistency triggers on PG
# ---------------------------------------------------------------------------

def test_pg_merge_post_commit_pending_insert_trigger_rejects_tenant_mismatch(pg_store):
    """ INSERT trigger: tenant_id must match parent merge_audit row."""
    conn = pg_store._get_conn()
    tid = f"pgtest-{uuid.uuid4()}"
    merge_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO merge_audit (merge_id, tenant_id, source_conversation_id, "
        "target_conversation_id, status, started_at) "
        "VALUES (%s, %s, %s, %s, 'in_progress', %s)",
        (merge_id, tid, f"src-{uuid.uuid4()}", f"tgt-{uuid.uuid4()}", _now()),
    )
    # Try to INSERT pending with mismatched tenant_id; must fail.
    import psycopg
    with pytest.raises(Exception) as exc_info:
        conn.execute(
            "INSERT INTO merge_post_commit_pending (pending_id, merge_id, "
            "tenant_id, kind, payload_json, status, created_at) "
            "VALUES (%s, %s, %s, 'sse_event', '{}', 'pending', %s)",
            (str(uuid.uuid4()), merge_id, "pgtest-WRONG-TENANT", _now()),
        )
    # Should be the trigger-raised error, not just any error.
    assert "tenant_id" in str(exc_info.value).lower() or "check_violation" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# : conversation_aliases.epoch column exists on PG
# ---------------------------------------------------------------------------

def test_pg_conversation_aliases_has_epoch_column(pg_store):
    conn = pg_store._get_conn()
    cols = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'conversation_aliases'",
    ).fetchall()
    names = {r["column_name"] for r in cols}
    assert "epoch" in names, f"conversation_aliases.epoch missing; cols: {names}"


# ---------------------------------------------------------------------------
# : origin_conversation_id columns on per-conv tables
# ---------------------------------------------------------------------------

def test_pg_per_conv_tables_have_origin_conversation_id(pg_store):
    conn = pg_store._get_conn()
    for tbl in ("segments", "canonical_turns", "facts", "tool_outputs",
                "tag_summaries"):
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s",
            (tbl,),
        ).fetchall()
        names = {r["column_name"] for r in cols}
        assert "origin_conversation_id" in names, (
            f"{tbl}.origin_conversation_id missing; cols: {names}"
        )
