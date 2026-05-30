"""Compaction-fence M0 schema migration smoke tests against PostgreSQL.

Gated on ``DATABASE_URL`` so a developer machine without a Postgres
instance can run the full test suite. CI / prod pipelines export
``DATABASE_URL`` to a throwaway database and run these alongside the
SQLite parity tests in ``test_compaction_fence_schema.py``.

Covers fencing plan §2.5 T0.5 + T0.6:

* M0.1 + M0.2 result in the operation_id columns and supporting
  indexes existing in the real Postgres catalog after bootstrap.
"""

from __future__ import annotations

import importlib.util
import os
import pytest


_pg_required = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping PG catalog smoke tests",
)
_psycopg_required = pytest.mark.skipif(
    importlib.util.find_spec("psycopg") is None,
    reason="psycopg not installed; skipping PG handler safety tests",
)


@pytest.fixture(scope="module")
def store():
    from virtual_context.storage.postgres import PostgresStore

    s = PostgresStore(os.environ["DATABASE_URL"])
    yield s
    # No teardown: schema additions are idempotent ALTERs and the test
    # database is reused across runs.


def _column_exists(store: PostgresStore, table: str, column: str) -> bool:
    with store.pool.connection() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM information_schema.columns
             WHERE table_name = %s AND column_name = %s
            """,
            (table, column),
        ).fetchone()
        return row is not None


def _index_exists(store: PostgresStore, index_name: str) -> bool:
    with store.pool.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s",
            (index_name,),
        ).fetchone()
        return row is not None


@_pg_required
class TestM0PostgresSchema:
    """T0.5: column existence via information_schema."""

    def test_segment_chunks_operation_id_column(self, store):
        assert _column_exists(store, "segment_chunks", "operation_id")

    def test_segment_tool_outputs_operation_id_column(self, store):
        assert _column_exists(store, "segment_tool_outputs", "operation_id")

    def test_fact_links_operation_id_column(self, store):
        assert _column_exists(store, "fact_links", "operation_id")

    def test_facts_operation_id_column(self, store):
        assert _column_exists(store, "facts", "operation_id")


@_pg_required
class TestM0PostgresIndexes:
    """T0.6: index existence via pg_indexes."""

    def test_segment_chunks_operation_id_index(self, store):
        assert _index_exists(store, "idx_segment_chunks_operation_id")

    def test_segment_tool_outputs_operation_id_index(self, store):
        assert _index_exists(store, "idx_segment_tool_outputs_operation_id")

    def test_fact_links_operation_id_index(self, store):
        assert _index_exists(store, "idx_fact_links_operation_id")

    def test_facts_operation_id_index(self, store):
        assert _index_exists(store, "idx_facts_operation_id")


@_psycopg_required
class TestM0HandlerSafety:
    """T0.7: production-handler safety contracts.

    The fencing plan §2.5 T0.7 requires that the dedicated production
    migration handler:

    - Runs ``SET LOCAL lock_timeout`` inside an explicit transaction so
      it actually governs the ALTERs (SET LOCAL is transaction-scoped).
    - Retries on ``LockNotAvailable`` and raises after a bounded retry
      exhausts (so a busy production table fails startup loudly rather
      than skipping silently).
    - Runs ``CREATE INDEX CONCURRENTLY`` on a connection with
      ``autocommit = True`` so the statement is not rejected by the
      implicit transaction.

    These tests mock the connection pool so they exercise the handler
    logic without requiring a live Postgres instance. They are NOT
    gated on ``DATABASE_URL`` because the contract being tested is
    pure-Python control flow over the psycopg pool. They still require
    the optional ``psycopg`` dependency for PostgresStore import and
    exception classes.
    """

    def _make_mock_store(self):
        from virtual_context.storage.postgres import PostgresStore
        from unittest.mock import MagicMock
        store = PostgresStore.__new__(PostgresStore)
        store.pool = MagicMock()
        return store

    def test_alter_runs_inside_explicit_transaction_with_lock_timeout(self):
        """Verify SET LOCAL lock_timeout precedes the ALTERs inside an
        explicit transaction context. Without the transaction wrapper,
        SET LOCAL has no effect and ALTERs are unbounded.
        """
        from virtual_context.storage.postgres import PostgresStore
        from unittest.mock import MagicMock
        store = self._make_mock_store()

        events: list[str] = []
        conn = MagicMock()

        def _execute(sql, *args, **kwargs):
            sql_str = sql if isinstance(sql, str) else str(sql)
            if "SET LOCAL lock_timeout" in sql_str:
                events.append("set_local")
            elif "ALTER TABLE" in sql_str:
                events.append("alter")
            result = MagicMock()
            result.fetchone.return_value = None
            result.fetchall.return_value = []
            return result

        conn.execute = _execute

        class _Txn:
            def __enter__(self):
                events.append("txn_enter")

            def __exit__(self, exc_type, exc, tb):
                events.append("txn_exit")
                return False

        conn.transaction.return_value = _Txn()

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=None)
        store.pool.connection.return_value = cm

        PostgresStore._ensure_compaction_fence_schema(store)

        assert "set_local" in events, (
            "lock_timeout must be set inside the transaction; otherwise "
            "the ALTERs are unbounded against a busy writer"
        )
        assert "alter" in events, "test must exercise at least one ALTER"
        txn_enter = events.index("txn_enter")
        set_local = events.index("set_local")
        first_alter = events.index("alter")
        txn_exit = events.index("txn_exit")
        assert txn_enter < set_local < first_alter < txn_exit, (
            "ALTERs must run inside an explicit conn.transaction() so "
            "SET LOCAL actually governs them"
        )

    def test_concurrent_index_runs_on_autocommit_connection(self):
        """Verify CREATE INDEX CONCURRENTLY runs on autocommit=True so
        psycopg does not reject the statement inside an implicit
        transaction. The handler must flip autocommit on the dedicated
        connection used for the index pass.
        """
        from virtual_context.storage.postgres import PostgresStore
        from unittest.mock import MagicMock
        store = self._make_mock_store()

        create_index_autocommit: list[bool] = []

        class _AutocommitConn:
            def __init__(self):
                self._autocommit = False

            @property
            def autocommit(self):
                return self._autocommit

            @autocommit.setter
            def autocommit(self, value):
                self._autocommit = bool(value)

            def execute(self, sql, *args, **kwargs):
                sql_str = sql if isinstance(sql, str) else str(sql)
                if "CREATE INDEX CONCURRENTLY" in sql_str:
                    create_index_autocommit.append(self.autocommit)
                result = MagicMock()
                result.fetchone.return_value = None
                result.fetchall.return_value = []
                return result

            def transaction(self):
                return MagicMock(
                    __enter__=MagicMock(return_value=None),
                    __exit__=MagicMock(return_value=None),
                )

        alter_conn = _AutocommitConn()
        index_conn = _AutocommitConn()
        alter_cm = MagicMock()
        alter_cm.__enter__ = MagicMock(return_value=alter_conn)
        alter_cm.__exit__ = MagicMock(return_value=None)
        index_cm = MagicMock()
        index_cm.__enter__ = MagicMock(return_value=index_conn)
        index_cm.__exit__ = MagicMock(return_value=None)
        store.pool.connection.side_effect = [alter_cm, index_cm]

        PostgresStore._ensure_compaction_fence_schema(store)

        assert create_index_autocommit, (
            "test must exercise CREATE INDEX CONCURRENTLY statements"
        )
        assert all(create_index_autocommit), (
            "CREATE INDEX CONCURRENTLY must run on autocommit=True; the "
            "handler must flip autocommit on the dedicated index pass "
            "connection"
        )

    def test_handler_raises_on_persistent_lock_timeout(self):
        """Verify the handler raises RuntimeError after the bounded
        retry exhausts on a persistent LockNotAvailable. A silent skip
        would let startup proceed without the required fence columns
        in place.
        """
        import psycopg
        from virtual_context.storage.postgres import PostgresStore
        from unittest.mock import MagicMock
        store = self._make_mock_store()

        conn = MagicMock()

        alter_attempts = 0

        def _execute(sql, *args, **kwargs):
            nonlocal alter_attempts
            sql_str = sql if isinstance(sql, str) else str(sql)
            if "ALTER TABLE" in sql_str:
                alter_attempts += 1
                raise psycopg.errors.LockNotAvailable("synthetic")
            result = MagicMock()
            result.fetchone.return_value = None
            result.fetchall.return_value = []
            return result

        conn.execute = _execute
        conn.transaction.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=None)
        store.pool.connection.return_value = cm

        with pytest.raises(RuntimeError, match="acquire lock"):
            PostgresStore._ensure_compaction_fence_schema(store)
        assert alter_attempts == 3
        assert store.pool.connection.call_count == 3
