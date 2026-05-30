"""Compaction-fence M0 schema migration tests.

Covers fencing plan §2.5 Phase 0 (M0.1 + M0.2 + M0.3):

* M0.1 operation_id columns on segment_chunks, segment_tool_outputs,
  fact_links exist after schema bootstrap.
* M0.2 supporting indexes on operation_id exist (segment_chunks,
  segment_tool_outputs, fact_links, facts).
* M0.3 fence kwargs on the five widened method signatures accept
  None for backward compatibility AND accept fully-supplied guard
  triples.
* Bootstrap re-run is idempotent.

PostgreSQL smoke tests live in
``tests/test_compaction_fence_schema_postgres.py`` and are skipped
unless ``DATABASE_URL`` is set, per the existing project convention.
"""

from __future__ import annotations

import tempfile
import sqlite3
import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import ChunkEmbedding, FactLink


def _make_store():
    """Build a fresh SQLiteStore on a temp file. Pins ACTIVE mode so
    the M0 fenced-call smoke tests see the per-write fence's raise
    behavior (default OFF would silently absorb the rejection per the
    P7 rollout discipline).
    """
    from virtual_context.core.compaction_fence import CompactionFenceMode
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return SQLiteStore(
        handle.name,
        compaction_fence_mode=CompactionFenceMode.ACTIVE,
    )


def _columns(store: SQLiteStore, table: str) -> set[str]:
    conn = store._get_conn()
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # Row factory may be tuple or sqlite3.Row; the column name is at
    # index 1 in either case.
    return {r[1] for r in rows}


def _indexes(store: SQLiteStore, table: str) -> set[str]:
    conn = store._get_conn()
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    return {r[1] for r in rows}


class TestM0Schema:
    """T0.1, T0.3: column + index presence after bootstrap."""

    def test_segment_chunks_has_operation_id_column(self):
        store = _make_store()
        assert "operation_id" in _columns(store, "segment_chunks")

    def test_segment_tool_outputs_has_operation_id_column(self):
        store = _make_store()
        assert "operation_id" in _columns(store, "segment_tool_outputs")

    def test_fact_links_has_operation_id_column(self):
        store = _make_store()
        assert "operation_id" in _columns(store, "fact_links")

    def test_segment_chunks_operation_id_index_exists(self):
        store = _make_store()
        assert "idx_segment_chunks_operation_id" in _indexes(
            store, "segment_chunks",
        )

    def test_segment_tool_outputs_operation_id_index_exists(self):
        store = _make_store()
        assert "idx_segment_tool_outputs_operation_id" in _indexes(
            store, "segment_tool_outputs",
        )

    def test_fact_links_operation_id_index_exists(self):
        store = _make_store()
        assert "idx_fact_links_operation_id" in _indexes(
            store, "fact_links",
        )

    def test_facts_operation_id_index_exists(self):
        # `facts.operation_id` is added by the existing
        # _ensure_compaction_scoping_columns helper. M0.2 contributes the
        # supporting index. The fencing plan requires the index for cleanup
        # DELETE efficiency on the per-write fence surface.
        store = _make_store()
        assert "idx_facts_operation_id" in _indexes(store, "facts")


class TestM0Idempotency:
    """T0.2: bootstrap re-run does not raise."""

    def test_ensure_schema_idempotent(self):
        store = _make_store()
        # Second bootstrap on the same store should be a no-op. Any
        # exception (duplicate column, missing column, etc.) would fail
        # this assertion.
        store._ensure_schema()
        assert "operation_id" in _columns(store, "segment_chunks")
        assert "operation_id" in _columns(store, "fact_links")
        assert "operation_id" in _columns(store, "segment_tool_outputs")

    def test_existing_db_missing_new_table_columns_migrates(self):
        store = _make_store()
        conn = store._get_conn()
        conn.executescript("""
            DROP TRIGGER IF EXISTS segments_chunks_ad;
            DROP INDEX IF EXISTS idx_segment_chunks_operation_id;
            ALTER TABLE segment_chunks RENAME TO segment_chunks_old;
            CREATE TABLE segment_chunks (
                segment_ref TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                PRIMARY KEY (segment_ref, chunk_index)
            );
            INSERT INTO segment_chunks
                (segment_ref, chunk_index, text, embedding_json)
            SELECT segment_ref, chunk_index, text, embedding_json
              FROM segment_chunks_old;
            DROP TABLE segment_chunks_old;

            DROP INDEX IF EXISTS idx_fact_links_operation_id;
            ALTER TABLE fact_links RENAME TO fact_links_old;
            CREATE TABLE fact_links (
                id TEXT PRIMARY KEY,
                source_fact_id TEXT NOT NULL,
                target_fact_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                context TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                created_by TEXT NOT NULL DEFAULT 'compaction',
                FOREIGN KEY (source_fact_id) REFERENCES facts(id) ON DELETE CASCADE,
                FOREIGN KEY (target_fact_id) REFERENCES facts(id) ON DELETE CASCADE
            );
            INSERT INTO fact_links
                (id, source_fact_id, target_fact_id, relation_type,
                 confidence, context, created_at, created_by)
            SELECT id, source_fact_id, target_fact_id, relation_type,
                   confidence, context, created_at, created_by
              FROM fact_links_old;
            DROP TABLE fact_links_old;

            DROP INDEX IF EXISTS idx_segment_tool_outputs_operation_id;
            ALTER TABLE segment_tool_outputs RENAME TO segment_tool_outputs_old;
            CREATE TABLE segment_tool_outputs (
                conversation_id TEXT NOT NULL,
                segment_ref TEXT NOT NULL,
                tool_output_ref TEXT NOT NULL,
                PRIMARY KEY (conversation_id, segment_ref, tool_output_ref)
            );
            INSERT INTO segment_tool_outputs
                (conversation_id, segment_ref, tool_output_ref)
            SELECT conversation_id, segment_ref, tool_output_ref
              FROM segment_tool_outputs_old;
            DROP TABLE segment_tool_outputs_old;
        """)
        conn.commit()

        assert "operation_id" in _columns(store, "segments")
        assert "operation_id" not in _columns(store, "segment_chunks")
        assert "operation_id" not in _columns(store, "fact_links")
        assert "operation_id" not in _columns(store, "segment_tool_outputs")

        store._ensure_schema()

        assert "operation_id" in _columns(store, "segment_chunks")
        assert "operation_id" in _columns(store, "fact_links")
        assert "operation_id" in _columns(store, "segment_tool_outputs")
        assert "idx_segment_chunks_operation_id" in _indexes(
            store, "segment_chunks",
        )
        assert "idx_fact_links_operation_id" in _indexes(store, "fact_links")
        assert "idx_segment_tool_outputs_operation_id" in _indexes(
            store, "segment_tool_outputs",
        )


class TestM0SignatureBackwardCompat:
    """T0.4: M0.3 method signatures accept None for backward compatibility.

    Legacy callers (CLI, tests, benchmarks, ingest/supersession,
    lazy semantic backfill) omit the fence kwargs entirely. The
    methods must remain callable without raising. The plan body §2.3
    contract: when all guard kwargs are None, the method writes
    unguarded (legacy behavior).
    """

    def test_set_fact_superseded_legacy_call(self):
        # No fence kwargs: legacy path. Should not raise on signature
        # alone. The actual write fails because the fact_ids do not
        # exist; that is expected for this signature-only test.
        store = _make_store()
        # The method swallows missing-row UPDATEs without raising on
        # SQLite, so this signature smoke test simply confirms the
        # call site accepts the legacy positional arguments.
        store.set_fact_superseded("a", "b")

    def test_set_fact_superseded_fenced_call(self):
        # P3 contract: when guard kwargs are supplied and no matching
        # running compaction_operation row exists, the method rejects
        # the write by raising CompactionLeaseLost. The M0 signature
        # contract is preserved (kwargs are accepted) while P3 takes
        # over the behavior.
        from virtual_context.types import CompactionLeaseLost
        store = _make_store()
        with pytest.raises(CompactionLeaseLost):
            store.set_fact_superseded(
                "a", "b",
                operation_id="op-1", owner_worker_id="w-1", lifecycle_epoch=2,
            )

    def test_update_fact_fields_legacy_call(self):
        store = _make_store()
        store.update_fact_fields("a", "v", "o", "active", "what")

    def test_update_fact_fields_fenced_call(self):
        from virtual_context.types import CompactionLeaseLost
        store = _make_store()
        with pytest.raises(CompactionLeaseLost):
            store.update_fact_fields(
                "a", "v", "o", "active", "what",
                operation_id="op-1", owner_worker_id="w-1", lifecycle_epoch=2,
            )

    def test_store_fact_links_legacy_call(self):
        store = _make_store()
        # store_fact_links accepts an empty list trivially and returns 0;
        # this confirms the signature is backward-compatible.
        assert store.store_fact_links([]) == 0

    def test_store_fact_links_fenced_call(self):
        # P3: empty links list short-circuits before the guard, so an
        # empty-list call with fenced kwargs still trivially returns 0.
        store = _make_store()
        assert (
            store.store_fact_links(
                [],
                operation_id="op-1", owner_worker_id="w-1",
                lifecycle_epoch=2, conversation_id="conv-1",
            )
            == 0
        )

    def test_store_chunk_embeddings_legacy_call(self):
        store = _make_store()
        # Empty chunks list trivially succeeds.
        store.store_chunk_embeddings("seg-1", [])

    def test_store_chunk_embeddings_fenced_call(self):
        # P3 contract: the segment-ownership probe fires before the
        # DELETE/INSERT loop, so an unseeded store rejects the write
        # via CompactionLeaseLost.
        from virtual_context.types import CompactionLeaseLost
        store = _make_store()
        with pytest.raises(CompactionLeaseLost):
            store.store_chunk_embeddings(
                "seg-1", [],
                operation_id="op-1", owner_worker_id="w-1",
                lifecycle_epoch=2, conversation_id="conv-1",
            )

    def test_link_segment_tool_output_legacy_call(self):
        store = _make_store()
        # Legacy callers pass only the three positional args.
        store.link_segment_tool_output("conv-1", "seg-1", "tool-1")

    def test_link_segment_tool_output_fenced_call(self):
        # P3 contract: guard-gated INSERT rejects when no matching op.
        from virtual_context.types import CompactionLeaseLost
        store = _make_store()
        with pytest.raises(CompactionLeaseLost):
            store.link_segment_tool_output(
                "conv-1", "seg-1", "tool-1",
                operation_id="op-1", owner_worker_id="w-1", lifecycle_epoch=2,
            )
