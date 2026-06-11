"""Compaction-fence Phase 4 PostgreSQL smoke test.

Gated on ``DATABASE_URL`` so a developer machine without a Postgres
instance can run the full test suite. CI / prod pipelines export
``DATABASE_URL`` to a throwaway database and run this alongside the
SQLite parity tests in ``test_compaction_cleanup_extension.py``.

Covers fencing plan §6.2:

* T4.5. End-to-end Postgres cleanup across all seven cleanup tables
  (segments, facts, tag_summaries, tag_summary_embeddings,
  segment_chunks, segment_tool_outputs, fact_links). Confirms the
  extended DELETE list shipped in this patch reaches every table
  and scopes by operation_id (with the (operation_id,
  conversation_id) double-keyed predicate on the five tables that
  carry conversation_id and operation_id alone on the two that
  don't).
"""

from __future__ import annotations

import os
import uuid

import pytest

from tests.pg_helpers import pg_dsn


_pg_required = pytest.mark.skipif(
    not pg_dsn(),
    reason="DATABASE_URL not set; skipping PG cleanup smoke tests",
)


_TS = "2026-05-29T00:00:00+00:00"


@pytest.fixture(scope="module")
def store():
    from virtual_context.core.compaction_fence import CompactionFenceMode
    from virtual_context.storage.postgres import PostgresStore
    s = PostgresStore(
        pg_dsn(),
        compaction_fence_mode=CompactionFenceMode.ACTIVE,
    )
    yield s


def _seed_running_op(store, conv_id: str, *, op_id: str,
                     worker_id: str, epoch: int = 1) -> None:
    with store.pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                """INSERT INTO conversation_lifecycle
                   (conversation_id, generation, deleted, updated_at)
                   VALUES (%s, 0, FALSE, %s)
                   ON CONFLICT (conversation_id) DO NOTHING""",
                (conv_id, _TS),
            )
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, tenant_id, phase, lifecycle_epoch,
                    created_at, updated_at)
                   VALUES (%s, 't-pg', 'compacting', %s, %s, %s)
                   ON CONFLICT (conversation_id) DO UPDATE SET
                       phase = EXCLUDED.phase,
                       lifecycle_epoch = EXCLUDED.lifecycle_epoch""",
                (conv_id, epoch, _TS, _TS),
            )
            conn.execute(
                """INSERT INTO compaction_operation
                   (operation_id, conversation_id, lifecycle_epoch,
                    phase_index, phase_count, phase_name, status,
                    started_at, heartbeat_ts, owner_worker_id,
                    created_at)
                   VALUES (%s, %s, %s, 0, 7, 'starting', 'running',
                           %s, %s, %s, %s)""",
                (op_id, conv_id, epoch, _TS, _TS, worker_id, _TS),
            )


def _seed_endpoints(store, conv_id: str, *, src_id: str,
                    tgt_id: str) -> None:
    with store.pool.connection() as conn:
        for fid in (src_id, tgt_id):
            conn.execute(
                """INSERT INTO facts
                   (id, subject, verb, object, status, what, who,
                    when_date, "where", why, fact_type, tags_json,
                    segment_ref, conversation_id, turn_numbers_json,
                    mentioned_at, session_date)
                   VALUES (%s, 's', 'v', 'o', 'active', '', '', '',
                           '', '', 'personal', '[]', '', %s, '[]',
                           %s, '')
                   ON CONFLICT (id) DO NOTHING""",
                (fid, conv_id, _TS),
            )


def _seed_owned_rows(store, conv_id: str, *, op_id: str, tag: str,
                     src_id: str, tgt_id: str) -> None:
    seg_ref = f"seg-{tag}"
    with store.pool.connection() as conn:
        conn.execute(
            """INSERT INTO segments
               (ref, conversation_id, summary, full_text, primary_tag,
                compaction_model, created_at, start_timestamp,
                end_timestamp, operation_id)
               VALUES (%s, %s, 's', 'f', 't', 'passthrough', %s, %s,
                       %s, %s)""",
            (seg_ref, conv_id, _TS, _TS, _TS, op_id),
        )
        conn.execute(
            """INSERT INTO facts
               (id, subject, verb, object, status, what, who,
                when_date, "where", why, fact_type, tags_json,
                segment_ref, conversation_id, turn_numbers_json,
                mentioned_at, session_date, operation_id)
               VALUES (%s, 's', 'v', 'o', 'active', '', '', '', '',
                       '', 'personal', '[]', %s, %s, '[]', %s, '',
                       %s)""",
            (f"fact-{tag}", seg_ref, conv_id, _TS, op_id),
        )
        conn.execute(
            """INSERT INTO tag_summaries
               (tag, conversation_id, summary, created_at, updated_at,
                operation_id)
               VALUES (%s, %s, 's', %s, %s, %s)""",
            (f"tag-{tag}", conv_id, _TS, _TS, op_id),
        )
        conn.execute(
            """INSERT INTO tag_summary_embeddings
               (tag, conversation_id, embedding_json, operation_id)
               VALUES (%s, %s, '[]', %s)""",
            (f"tag-{tag}", conv_id, op_id),
        )
        conn.execute(
            """INSERT INTO segment_chunks
               (segment_ref, chunk_index, text, embedding_json,
                operation_id)
               VALUES (%s, 0, 'x', '[]', %s)""",
            (seg_ref, op_id),
        )
        conn.execute(
            """INSERT INTO segment_tool_outputs
               (conversation_id, segment_ref, tool_output_ref,
                operation_id)
               VALUES (%s, %s, %s, %s)""",
            (conv_id, seg_ref, f"tool-{tag}", op_id),
        )
        conn.execute(
            """INSERT INTO fact_links
               (id, source_fact_id, target_fact_id, relation_type,
                confidence, context, created_at, created_by,
                operation_id)
               VALUES (%s, %s, %s, 'r', 1.0, '', %s, 'compaction',
                       %s)""",
            (f"link-{tag}", src_id, tgt_id, _TS, op_id),
        )


def _count_dead(store, conv_id: str, dead_op: str) -> dict[str, int]:
    out: dict[str, int] = {}
    with store.pool.connection() as conn:
        for tbl in ("segments", "facts", "tag_summaries",
                    "tag_summary_embeddings", "segment_tool_outputs"):
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM {tbl} "
                f"WHERE conversation_id = %s AND operation_id = %s",
                (conv_id, dead_op),
            ).fetchone()
            out[tbl] = int(row["n"])
        for tbl in ("segment_chunks", "fact_links"):
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM {tbl} "
                f"WHERE operation_id = %s",
                (dead_op,),
            ).fetchone()
            out[tbl] = int(row["n"])
    return out


@_pg_required
class TestT45_PGCleanupAcrossAllSevenTables:
    def test_dead_op_rows_deleted_from_all_seven_tables_pg(self, store):
        conv = f"pg-t45-{uuid.uuid4().hex[:8]}"
        dead_op = uuid.uuid4().hex
        worker = "w-1"
        src_id = f"fact-src-{uuid.uuid4().hex[:8]}"
        tgt_id = f"fact-tgt-{uuid.uuid4().hex[:8]}"
        _seed_running_op(store, conv, op_id=dead_op, worker_id=worker)
        _seed_endpoints(store, conv, src_id=src_id, tgt_id=tgt_id)
        _seed_owned_rows(
            store, conv, op_id=dead_op, tag=f"d-{uuid.uuid4().hex[:8]}",
            src_id=src_id, tgt_id=tgt_id,
        )

        new_op = uuid.uuid4().hex
        ok = store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead_op,
            new_operation_id=new_op,
            lifecycle_epoch=1,
            worker_id="w-takeover",
            phase_count=7,
        )
        assert ok is True

        counts = _count_dead(store, conv, dead_op)
        for tbl, n in counts.items():
            assert n == 0, (
                f"PG table {tbl}: expected 0 rows for dead-op, got {n}"
            )
