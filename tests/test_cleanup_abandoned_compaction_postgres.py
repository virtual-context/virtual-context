"""Postgres mirror of test_cleanup_abandoned_compaction.py.

Same 5 tests, adapted for PostgresStore:
- PostgresStore(dsn=...) instead of SQLiteStore(tmp_path / "...")
- %s placeholders instead of ?
- conn.transaction() context manager for atomic blocks
- Skipped when DATABASE_URL is absent

Between tests each fixture tears down all rows created in
compaction_operation / segments / facts / tag_summaries /
tag_summary_embeddings / canonical_turns / conversations for the
test-specific conversation_id so tests do not leak state.
"""
from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — Postgres tests skipped",
)

from virtual_context.storage.postgres import PostgresStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_ids():
    """Return a unique (conv_id, dead_op_id, live_op_id) triple."""
    return (
        f"pg-conv-{uuid.uuid4().hex[:8]}",
        f"pg-dead-{uuid.uuid4().hex[:8]}",
        f"pg-live-{uuid.uuid4().hex[:8]}",
    )


def _seed(store: PostgresStore, conv: str, dead_op: str, live_op: str) -> None:
    """Postgres mirror of the SQLite _seed helper.

    Seeds:
    - live_op: status='completed' (prior finished compaction)
    - dead_op: status='running'  (the one that will be abandoned)
    - one row each in segments/facts/tag_summaries/tag_summary_embeddings
      for both operations
    - one canonical_turn marked by dead_op, one by live_op
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    earlier_str = "2026-01-01T00:00:00+00:00"
    earlier = datetime.fromisoformat(earlier_str)

    conn = store._get_conn()
    with conn.transaction():
        # live_op: completed prior compaction — control group
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at,
                completed_at)
               VALUES (%s, %s, 1, 6, 7, 'tag_summaries', 'completed',
                       %s, %s, %s, %s, %s)""",
            (live_op, conv, earlier, earlier, "prior-worker", earlier, earlier),
        )
        # dead_op: running — will be abandoned by cleanup
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (%s, %s, 1, 0, 7, 'starting', 'running',
                       %s, %s, %s, %s)""",
            (dead_op, conv, now, now, "dead-worker", now),
        )
        for op_id in (dead_op, live_op):
            conn.execute(
                """INSERT INTO segments (ref, conversation_id, summary,
                   full_text, primary_tag, compaction_model, created_at,
                   start_timestamp, end_timestamp, operation_id)
                   VALUES (%s, %s, 's', 'f', 't', 'passthrough', %s, %s, %s, %s)""",
                (f"seg-{op_id[:6]}", conv, now, now, now, op_id),
            )
            conn.execute(
                """INSERT INTO facts (id, subject, verb, object, status, what,
                   conversation_id, mentioned_at, session_date, operation_id)
                   VALUES (%s, 'S', 'V', 'O', 'active', 'what', %s, %s, %s, %s)""",
                (f"fact-{op_id[:5]}", conv, now, now, op_id),
            )
            conn.execute(
                """INSERT INTO tag_summaries (tag, conversation_id, summary,
                   created_at, updated_at, operation_id)
                   VALUES (%s, %s, 's', %s, %s, %s)""",
                (f"tag-{op_id[:5]}", conv, now, now, op_id),
            )
            conn.execute(
                """INSERT INTO tag_summary_embeddings
                   (tag, conversation_id, embedding_json, operation_id)
                   VALUES (%s, %s, '[]', %s)""",
                (f"tag-{op_id[:5]}", conv, op_id),
            )
        # canonical_turns: one marked by dead_op, one by live_op
        for op_id, canonical_id, sort_key in (
            (dead_op, f"ct-dead-{conv}", 1000.0),
            (live_op, f"ct-live-{conv}", 2000.0),
        ):
            conn.execute(
                """INSERT INTO canonical_turns
                   (canonical_turn_id, conversation_id, sort_key, turn_hash,
                    hash_version, user_content, assistant_content, tagged_at,
                    compacted_at, compaction_operation_id,
                    first_seen_at, last_seen_at, created_at, updated_at,
                    covered_ingestible_entries, turn_group_number)
                   VALUES (%s, %s, %s, %s, 1, 'u', 'a', %s, %s, %s,
                           %s, %s, %s, %s, 1, 0)""",
                (
                    canonical_id, conv, sort_key,
                    f"h-{canonical_id[:10]}",
                    now, now, op_id, now, now, now, now,
                ),
            )


def _counts(store: PostgresStore, conv: str, op_id: str) -> dict[str, int]:
    out: dict[str, int] = {}
    conn = store._get_conn()
    for table in (
        "segments", "facts", "tag_summaries", "tag_summary_embeddings",
    ):
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM {table} "
            f"WHERE conversation_id = %s AND operation_id = %s",
            (conv, op_id),
        ).fetchone()
        out[table] = int(row["cnt"])
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM compaction_operation "
        "WHERE conversation_id = %s AND operation_id = %s",
        (conv, op_id),
    ).fetchone()
    out["compaction_operation"] = int(row["cnt"])
    return out


def _teardown(store: PostgresStore, conv: str) -> None:
    """Delete all rows seeded for conv so tests don't bleed state."""
    conn = store._get_conn()
    with conn.transaction():
        for table in (
            "tag_summary_embeddings", "tag_summaries", "facts", "segments",
            "canonical_turns", "compaction_operation",
        ):
            conn.execute(
                f"DELETE FROM {table} WHERE conversation_id = %s", (conv,)
            )
        conn.execute(
            "DELETE FROM conversations WHERE conversation_id = %s", (conv,)
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    dsn = os.environ["DATABASE_URL"]
    return PostgresStore(dsn=dsn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pg_cleanup_scopes_by_operation_id(store: PostgresStore):
    conv, dead_op, live_op = _fresh_ids()
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    try:
        _seed(store, conv, dead_op=dead_op, live_op=live_op)

        store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead_op,
            new_operation_id=f"new-op-{uuid.uuid4().hex[:8]}",
            lifecycle_epoch=1,
            worker_id="new-worker",
            phase_count=7,
        )

        dead = _counts(store, conv, dead_op)
        live = _counts(store, conv, live_op)
        assert dead["segments"] == 0
        assert dead["facts"] == 0
        assert dead["tag_summaries"] == 0
        assert dead["tag_summary_embeddings"] == 0
        assert live["segments"] == 1
        assert live["facts"] == 1
        assert live["tag_summaries"] == 1
        assert live["tag_summary_embeddings"] == 1
    finally:
        _teardown(store, conv)


def test_pg_cleanup_unsets_canonical_compacted_at(store: PostgresStore):
    conv, dead_op, live_op = _fresh_ids()
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    try:
        _seed(store, conv, dead_op=dead_op, live_op=live_op)

        store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead_op,
            new_operation_id=f"new-op-{uuid.uuid4().hex[:8]}",
            lifecycle_epoch=1,
            worker_id="new-worker",
            phase_count=7,
        )

        conn = store._get_conn()
        dead_ct = conn.execute(
            "SELECT compacted_at, compaction_operation_id FROM canonical_turns "
            "WHERE canonical_turn_id = %s",
            (f"ct-dead-{conv}",),
        ).fetchone()
        live_ct = conn.execute(
            "SELECT compacted_at, compaction_operation_id FROM canonical_turns "
            "WHERE canonical_turn_id = %s",
            (f"ct-live-{conv}",),
        ).fetchone()
        assert dead_ct["compacted_at"] is None
        assert dead_ct["compaction_operation_id"] is None
        assert live_ct["compacted_at"] is not None, "live-op's ct must be untouched"
        assert str(live_ct["compaction_operation_id"]) == live_op
    finally:
        _teardown(store, conv)


def test_pg_cleanup_inserts_new_running_row(store: PostgresStore):
    conv, dead_op, live_op = _fresh_ids()
    new_op = f"new-op-{uuid.uuid4().hex[:8]}"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    try:
        _seed(store, conv, dead_op=dead_op, live_op=live_op)

        store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead_op,
            new_operation_id=new_op,
            lifecycle_epoch=1,
            worker_id="new-worker",
            phase_count=7,
        )

        conn = store._get_conn()
        new_row = conn.execute(
            "SELECT status, owner_worker_id, phase_count FROM compaction_operation "
            "WHERE operation_id = %s",
            (new_op,),
        ).fetchone()
        dead_row = conn.execute(
            "SELECT status FROM compaction_operation WHERE operation_id = %s",
            (dead_op,),
        ).fetchone()
        assert new_row["status"] == "running"
        assert new_row["owner_worker_id"] == "new-worker"
        assert new_row["phase_count"] == 7
        assert dead_row["status"] == "abandoned"
    finally:
        _teardown(store, conv)


def test_pg_cleanup_is_idempotent_and_preserves_one_active_invariant(store: PostgresStore):
    """Calling cleanup twice for the same dead_op must produce exactly one
    running row (new-op-1). The second call sees zero rows matched on the
    UPDATE, skips the INSERT, and does NOT create a second running row.
    """
    conv, dead_op, live_op = _fresh_ids()
    new_op_1 = f"new-op-1-{uuid.uuid4().hex[:6]}"
    new_op_2 = f"new-op-2-{uuid.uuid4().hex[:6]}"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    try:
        _seed(store, conv, dead_op=dead_op, live_op=live_op)

        ret1 = store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead_op,
            new_operation_id=new_op_1,
            lifecycle_epoch=1,
            worker_id="new-worker",
            phase_count=7,
        )
        ret2 = store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead_op,
            new_operation_id=new_op_2,
            lifecycle_epoch=1,
            worker_id="new-worker",
            phase_count=7,
        )

        assert ret1 is True, "first call must be a fresh takeover"
        assert ret2 is False, "second call must be idempotent (False)"

        conn = store._get_conn()
        rows = conn.execute(
            "SELECT operation_id, status FROM compaction_operation "
            "WHERE conversation_id = %s AND lifecycle_epoch = 1 "
            "ORDER BY operation_id",
            (conv,),
        ).fetchall()
        running = [r for r in rows if r["status"] == "running"]
        abandoned = [r for r in rows if r["status"] == "abandoned"]

        assert len(running) == 1, (
            f"One-active invariant violated: expected exactly 1 running "
            f"row at ({conv}, epoch=1); got {[dict(r) for r in running]}"
        )
        assert str(running[0]["operation_id"]) == new_op_1
        assert {str(r["operation_id"]) for r in abandoned} == {dead_op}
        assert all(str(r["operation_id"]) != new_op_2 for r in rows), (
            "new-op-2 was inserted on a redundant cleanup call — invariant violated"
        )
    finally:
        _teardown(store, conv)


def test_pg_cleanup_respects_one_active_invariant_on_concurrent_peer(store: PostgresStore):
    """The unique partial index must prevent two running rows from coexisting
    at the same (conv, epoch). Cleanup runs normally after the rejected insert.
    """
    import psycopg

    conv, dead_op, live_op = _fresh_ids()
    new_op = f"new-op-{uuid.uuid4().hex[:8]}"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    try:
        _seed(store, conv, dead_op=dead_op, live_op=live_op)

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        peer_op = f"peer-op-{uuid.uuid4().hex[:8]}"

        # Attempting to seed a second 'running' row at the same epoch must fail
        conn = store._get_conn()
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.transaction():
                conn.execute(
                    """INSERT INTO compaction_operation
                       (operation_id, conversation_id, lifecycle_epoch,
                        phase_index, phase_count, phase_name, status,
                        started_at, heartbeat_ts, owner_worker_id, created_at)
                       VALUES (%s, %s, 1, 0, 7, 'starting', 'running',
                               %s, %s, %s, %s)""",
                    (peer_op, conv, now, now, "peer", now),
                )

        # The index held — cleanup now runs normally
        store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead_op,
            new_operation_id=new_op,
            lifecycle_epoch=1,
            worker_id="new-worker",
            phase_count=7,
        )
        running = conn.execute(
            "SELECT operation_id FROM compaction_operation "
            "WHERE conversation_id = %s AND lifecycle_epoch = 1 "
            "  AND status = 'running'",
            (conv,),
        ).fetchall()
        assert len(running) == 1
        assert str(running[0]["operation_id"]) == new_op
    finally:
        _teardown(store, conv)
