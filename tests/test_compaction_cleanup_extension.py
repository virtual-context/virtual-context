"""Compaction-fence Phase 4 tests for the extended
``cleanup_abandoned_compaction`` DELETE list.

Per fencing plan §6.1 + §6.2:

* T4.1. Seed an op_X with operation-owned rows across all seven
  cleanup tables (segments, facts, tag_summaries,
  tag_summary_embeddings, segment_chunks, segment_tool_outputs,
  fact_links). Mark op_X 'running'. Run cleanup. Assert every table
  is empty for op_X and no pre-existing row with NULL or other
  operation_id was touched.
* T4.2. Cleanup is idempotent: running it twice produces no
  additional deletes and no error.
* T4.3. Cleanup does NOT touch rows with ``operation_id IS NULL``
  (pre-migration legacy rows are protected).
* T4.4. Cleanup does NOT touch rows with ``operation_id != dead_op``
  (other operations' rows are protected).

T4.5 PG smoke lives in ``test_compaction_cleanup_extension_postgres.py``.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.core.compaction_fence import CompactionFenceMode
from virtual_context.storage.sqlite import SQLiteStore


_EARLIER = "2026-01-01T00:00:00+00:00"


def _make_active_store(path) -> SQLiteStore:
    """Cleanup-extension tests pin the store to ACTIVE mode so the
    new-table DELETEs execute. The default OFF mode skips those
    DELETEs per the P7 tier-rollout discipline (fencing plan §9.1).
    """
    return SQLiteStore(path, compaction_fence_mode=CompactionFenceMode.ACTIVE)


def _seed_op_row(
    store: SQLiteStore, *, conv: str, op_id: str, status: str = "running",
    worker_id: str = "w", started_at: str | None = None,
) -> None:
    """Seed a compaction_operation row + a conversation_lifecycle row
    (idempotent). The partial unique index allows multiple non-active
    rows to coexist; the test only seeds at most one 'running' per
    (conv, epoch).
    """
    now = utcnow_iso()
    conn = store._get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv, now),
    )
    started = started_at or now
    if status == "running":
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (op_id, conv, started, started, worker_id, started),
        )
    else:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at,
                completed_at)
               VALUES (?, ?, 1, 6, 7, 'tag_summaries', ?,
                       ?, ?, ?, ?, ?)""",
            (op_id, conv, status, started, started, worker_id,
             started, started),
        )
    conn.commit()


def _seed_fact_link_endpoints(store: SQLiteStore, *, conv: str) -> None:
    """Seed two persistent endpoint facts (operation_id=NULL) the
    test's fact_links rows reference. NULL operation_id keeps them
    immune from cleanup, so the fact_links DELETE we assert on is the
    P4 operation_id-scoped DELETE -- not a cascade from a deleted
    endpoint fact.
    """
    now = utcnow_iso()
    conn = store._get_conn()
    for fid in ("fact-shared-src", "fact-shared-tgt"):
        conn.execute(
            """INSERT OR IGNORE INTO facts
               (id, subject, verb, object, status, what,
                conversation_id, mentioned_at, session_date,
                operation_id)
               VALUES (?, 'shared', 'V', 'O', 'active', 'w', ?, ?, ?,
                       NULL)""",
            (fid, conv, now, now),
        )
    conn.commit()


def _seed_owned_rows(
    store: SQLiteStore, *, conv: str, op_id: str | None, tag: str,
) -> None:
    """Seed one row per cleanup table stamped with ``op_id`` for
    conversation ``conv``. When ``op_id`` is None, simulates a
    pre-migration legacy row.
    """
    now = utcnow_iso()
    conn = store._get_conn()
    # segments: stamped with operation_id when supplied.
    conn.execute(
        """INSERT INTO segments
           (ref, conversation_id, summary, full_text, primary_tag,
            compaction_model, created_at, start_timestamp,
            end_timestamp, operation_id)
           VALUES (?, ?, 's', 'f', 't', 'passthrough', ?, ?, ?, ?)""",
        (f"seg-{tag}", conv, now, now, now, op_id),
    )
    # facts: P3-stamped.
    conn.execute(
        """INSERT INTO facts
           (id, subject, verb, object, status, what,
            conversation_id, mentioned_at, session_date, operation_id)
           VALUES (?, 'S', 'V', 'O', 'active', 'w', ?, ?, ?, ?)""",
        (f"fact-{tag}", conv, now, now, op_id),
    )
    # tag_summaries.
    conn.execute(
        """INSERT INTO tag_summaries
           (tag, conversation_id, summary, created_at, updated_at,
            operation_id)
           VALUES (?, ?, 's', ?, ?, ?)""",
        (f"tag-{tag}", conv, now, now, op_id),
    )
    # tag_summary_embeddings.
    conn.execute(
        """INSERT INTO tag_summary_embeddings
           (tag, conversation_id, embedding_json, operation_id)
           VALUES (?, ?, '[]', ?)""",
        (f"tag-{tag}", conv, op_id),
    )
    # segment_chunks: no conversation_id column.
    conn.execute(
        """INSERT INTO segment_chunks
           (segment_ref, chunk_index, text, embedding_json,
            operation_id)
           VALUES (?, 0, 'x', '[]', ?)""",
        (f"seg-{tag}", op_id),
    )
    # segment_tool_outputs.
    conn.execute(
        """INSERT INTO segment_tool_outputs
           (conversation_id, segment_ref, tool_output_ref,
            operation_id)
           VALUES (?, ?, ?, ?)""",
        (conv, f"seg-{tag}", f"tool-{tag}", op_id),
    )
    # fact_links: no conversation_id column. The endpoints are the
    # SHARED facts seeded once per test via
    # ``_seed_fact_link_endpoints`` so that cleanup's FK CASCADE on
    # facts DELETE does NOT collaterally remove the link. The P4
    # cleanup must reach this row via the operation_id-scoped DELETE
    # on fact_links itself.
    conn.execute(
        """INSERT INTO fact_links
           (id, source_fact_id, target_fact_id, relation_type,
            confidence, context, created_at, created_by, operation_id)
           VALUES (?, 'fact-shared-src', 'fact-shared-tgt', 'r',
                   1.0, '', ?, 'compaction', ?)""",
        (f"link-{tag}", now, op_id),
    )
    conn.commit()


def _counts(
    store: SQLiteStore, *, conv: str, op_id_predicate: str,
) -> dict[str, int]:
    """Return per-table row counts matching the SQL predicate on
    ``operation_id`` (e.g. ``= 'dead-op'`` or ``IS NULL``)."""
    out: dict[str, int] = {}
    conn = store._get_conn()
    for table in (
        "segments", "facts", "tag_summaries", "tag_summary_embeddings",
        "segment_tool_outputs",
    ):
        r = conn.execute(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE conversation_id = ? AND operation_id {op_id_predicate}",
            (conv,),
        ).fetchone()
        out[table] = int(r[0])
    for table in ("segment_chunks", "fact_links"):
        r = conn.execute(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE operation_id {op_id_predicate}",
        ).fetchone()
        out[table] = int(r[0])
    return out


# ---------------------------------------------------------------------------
# T4.1: scoped DELETE across all seven cleanup tables.
# ---------------------------------------------------------------------------


class TestT41_ScopedDeleteAcrossAllSevenTables:
    def test_dead_op_rows_deleted_from_all_seven_tables(self, tmp_path: Path):
        store = _make_active_store(tmp_path / "t41.db")
        store.upsert_conversation(tenant_id="t", conversation_id="conv-1")
        _seed_op_row(store, conv="conv-1", op_id="dead-op", status="running",
                     worker_id="dead-worker")
        _seed_op_row(store, conv="conv-1", op_id="live-op",
                     status="completed", worker_id="live-worker",
                     started_at=_EARLIER)
        _seed_fact_link_endpoints(store, conv="conv-1")
        _seed_owned_rows(store, conv="conv-1", op_id="dead-op", tag="d")
        _seed_owned_rows(store, conv="conv-1", op_id="live-op", tag="l")

        store.cleanup_abandoned_compaction(
            conversation_id="conv-1",
            dead_operation_id="dead-op",
            new_operation_id="new-op",
            lifecycle_epoch=1,
            worker_id="new-worker",
            phase_count=7,
        )

        # dead-op rows: gone from every cleanup table.
        dead = _counts(store, conv="conv-1", op_id_predicate="= 'dead-op'")
        for tbl in (
            "segments", "facts", "tag_summaries", "tag_summary_embeddings",
            "segment_chunks", "segment_tool_outputs", "fact_links",
        ):
            assert dead[tbl] == 0, (
                f"table {tbl}: expected 0 rows for dead-op, got {dead[tbl]}"
            )

        # live-op rows: untouched in every cleanup table.
        live = _counts(store, conv="conv-1", op_id_predicate="= 'live-op'")
        for tbl in (
            "segments", "facts", "tag_summaries", "tag_summary_embeddings",
            "segment_chunks", "segment_tool_outputs", "fact_links",
        ):
            assert live[tbl] == 1, (
                f"table {tbl}: expected 1 row for live-op, got {live[tbl]}"
            )


# ---------------------------------------------------------------------------
# T4.2: cleanup is idempotent.
# ---------------------------------------------------------------------------


class TestT42_CleanupIsIdempotent:
    def test_second_run_no_error_no_extra_deletes(self, tmp_path: Path):
        store = _make_active_store(tmp_path / "t42.db")
        store.upsert_conversation(tenant_id="t", conversation_id="conv-2")
        _seed_op_row(store, conv="conv-2", op_id="dead-op", status="running")
        _seed_fact_link_endpoints(store, conv="conv-2")
        _seed_owned_rows(store, conv="conv-2", op_id="dead-op", tag="d")

        ok_first = store.cleanup_abandoned_compaction(
            conversation_id="conv-2",
            dead_operation_id="dead-op",
            new_operation_id="new-op-1",
            lifecycle_epoch=1, worker_id="w-1", phase_count=7,
        )
        # First call performs the takeover transition.
        assert ok_first is True
        dead_after_first = _counts(
            store, conv="conv-2", op_id_predicate="= 'dead-op'",
        )
        for tbl, count in dead_after_first.items():
            assert count == 0, (
                f"first cleanup left {count} dead-op rows in {tbl}"
            )

        # Second call: idempotent. The dead-op row is now status='abandoned'
        # so the UPDATE matches zero rows -> fresh_takeover=False, no new
        # row inserted, but DELETEs still run as no-ops without error.
        ok_second = store.cleanup_abandoned_compaction(
            conversation_id="conv-2",
            dead_operation_id="dead-op",
            new_operation_id="new-op-2",
            lifecycle_epoch=1, worker_id="w-2", phase_count=7,
        )
        assert ok_second is False
        dead_after_second = _counts(
            store, conv="conv-2", op_id_predicate="= 'dead-op'",
        )
        for tbl, count in dead_after_second.items():
            assert count == 0, (
                f"second cleanup left {count} dead-op rows in {tbl}"
            )


# ---------------------------------------------------------------------------
# T4.3: rows with operation_id IS NULL (legacy) are protected.
# ---------------------------------------------------------------------------


class TestT43_LegacyNullOpIdRowsProtected:
    def test_null_operation_id_rows_not_touched(self, tmp_path: Path):
        store = _make_active_store(tmp_path / "t43.db")
        store.upsert_conversation(tenant_id="t", conversation_id="conv-3")
        _seed_op_row(store, conv="conv-3", op_id="dead-op", status="running")
        _seed_fact_link_endpoints(store, conv="conv-3")
        _seed_owned_rows(store, conv="conv-3", op_id="dead-op", tag="d")
        # Legacy rows with operation_id = NULL.
        _seed_owned_rows(store, conv="conv-3", op_id=None, tag="legacy")

        store.cleanup_abandoned_compaction(
            conversation_id="conv-3",
            dead_operation_id="dead-op",
            new_operation_id="new-op",
            lifecycle_epoch=1, worker_id="w", phase_count=7,
        )

        # Verify each legacy row (by its tag-suffixed id) survives the
        # cleanup. Querying by id avoids conflating the legacy NULL-op
        # row with the shared endpoint facts (also NULL-op) seeded by
        # ``_seed_fact_link_endpoints``.
        conn = store._get_conn()
        legacy_checks = (
            ("segments", "SELECT 1 FROM segments WHERE ref = 'seg-legacy'"),
            ("facts", "SELECT 1 FROM facts WHERE id = 'fact-legacy'"),
            ("tag_summaries",
             "SELECT 1 FROM tag_summaries WHERE tag = 'tag-legacy'"),
            ("tag_summary_embeddings",
             "SELECT 1 FROM tag_summary_embeddings WHERE tag = 'tag-legacy'"),
            ("segment_chunks",
             "SELECT 1 FROM segment_chunks WHERE segment_ref = 'seg-legacy'"),
            ("segment_tool_outputs",
             "SELECT 1 FROM segment_tool_outputs "
             "WHERE segment_ref = 'seg-legacy'"),
            ("fact_links",
             "SELECT 1 FROM fact_links WHERE id = 'link-legacy'"),
        )
        for tbl, sql in legacy_checks:
            row = conn.execute(sql).fetchone()
            assert row is not None, (
                f"table {tbl}: legacy NULL-op_id row was DELETEd; "
                f"cleanup must scope by op_id"
            )


# ---------------------------------------------------------------------------
# T4.4: rows for OTHER operations (concurrent / queued / completed) are
# protected.
# ---------------------------------------------------------------------------


class TestT44_OtherOperationRowsProtected:
    def test_other_op_rows_not_touched(self, tmp_path: Path):
        store = _make_active_store(tmp_path / "t44.db")
        store.upsert_conversation(tenant_id="t", conversation_id="conv-4")
        _seed_op_row(store, conv="conv-4", op_id="dead-op", status="running")
        _seed_op_row(store, conv="conv-4", op_id="other-op",
                     status="completed", worker_id="other-w",
                     started_at=_EARLIER)
        _seed_fact_link_endpoints(store, conv="conv-4")
        _seed_owned_rows(store, conv="conv-4", op_id="dead-op", tag="d")
        _seed_owned_rows(store, conv="conv-4", op_id="other-op", tag="o")

        store.cleanup_abandoned_compaction(
            conversation_id="conv-4",
            dead_operation_id="dead-op",
            new_operation_id="new-op",
            lifecycle_epoch=1, worker_id="w", phase_count=7,
        )

        other = _counts(
            store, conv="conv-4", op_id_predicate="= 'other-op'",
        )
        for tbl in (
            "segments", "facts", "tag_summaries", "tag_summary_embeddings",
            "segment_chunks", "segment_tool_outputs", "fact_links",
        ):
            assert other[tbl] == 1, (
                f"table {tbl}: other-op row was collaterally DELETEd "
                f"(count={other[tbl]}); cleanup must scope by op_id"
            )
