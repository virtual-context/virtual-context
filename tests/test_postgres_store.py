"""Focused unit tests for PostgresStore connection management."""

from __future__ import annotations

class _FakeConn:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, sql: str, params=None):
        self.executed.append((sql, params))
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    class _Txn:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            return False

    def transaction(self):
        return self._Txn()

    def close(self) -> None:
        self.closed = True


class _FakeRowsResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _ConnCheckout:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn

    def __enter__(self) -> _FakeConn:
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakePool:
    instances: list["_FakePool"] = []

    def __init__(self, conninfo: str, **kwargs) -> None:
        self.conninfo = conninfo
        self.kwargs = kwargs
        self.conn = _FakeConn(f"conn-{len(self.instances)}")
        self.checkouts = 0
        self.closed = False
        self.instances.append(self)

    def connection(self) -> _ConnCheckout:
        self.checkouts += 1
        return _ConnCheckout(self.conn)

    def close(self) -> None:
        self.closed = True
        self.conn.close()


def test_postgres_store_uses_bounded_connection_pool(monkeypatch):
    from virtual_context.storage import postgres as pg

    _FakePool.instances.clear()
    monkeypatch.setattr(pg, "ConnectionPool", _FakePool)

    store = pg.PostgresStore("postgresql://example")
    pool = _FakePool.instances[0]

    assert pool.conninfo == "postgresql://example"
    assert pool.kwargs == {
        "min_size": 1,
        "max_size": 8,
        "timeout": 30.0,
        "max_idle": 300.0,
        "kwargs": {"row_factory": pg.dict_row, "autocommit": True},
    }
    assert pool.checkouts > 0

    store.close()

    assert pool.closed
    assert pool.conn.closed


def test_postgres_store_get_all_segments_uses_batch_tag_lookup(monkeypatch):
    from virtual_context.storage import postgres as pg

    class _RowsConn(_FakeConn):
        def execute(self, sql: str, params=None):
            self.executed.append((sql, params))
            if "FROM segments" in sql:
                return _FakeRowsResult([
                    {
                        "ref": "seg-1",
                        "conversation_id": "conv-1",
                        "primary_tag": "tag-a",
                        "summary": "summary",
                        "summary_tokens": 3,
                        "full_text": "full text",
                        "full_tokens": 6,
                        "messages_json": "[]",
                        "metadata_json": "{\"turn_count\": 2}",
                        "created_at": "2026-04-14T00:00:00+00:00",
                        "start_timestamp": "2026-04-14T00:00:00+00:00",
                        "end_timestamp": "2026-04-14T00:01:00+00:00",
                        "compaction_model": "test-model",
                        "compression_ratio": 0.5,
                    }
                ])
            return _FakeRowsResult([])

    conn = _RowsConn("conn-0")

    class _RowsPool(_FakePool):
        def __init__(self, conninfo: str, **kwargs) -> None:
            super().__init__(conninfo, **kwargs)
            self.conn = conn

    monkeypatch.setattr(pg, "ConnectionPool", _RowsPool)

    store = pg.PostgresStore("postgresql://example")
    monkeypatch.setattr(store, "_batch_get_tags", lambda refs: {"seg-1": ["tag-a", "tag-b"]})

    segments = store.get_all_segments(conversation_id="conv-1")

    assert len(segments) == 1
    assert segments[0].ref == "seg-1"
    assert segments[0].tags == ["tag-a", "tag-b"]


def test_normalize_request_turn_sequences_works_without_executemany(monkeypatch):
    from virtual_context.storage import postgres as pg

    class _NormalizeConn(_FakeConn):
        def execute(self, sql: str, params=None):
            self.executed.append((sql, params))
            if "SELECT id, conversation_id, request_turn, timestamp FROM request_context" in sql:
                return _FakeRowsResult([
                    {
                        "id": 10,
                        "conversation_id": "conv-1",
                        "request_turn": 494,
                        "timestamp": "2026-04-14T22:11:59.177648+00:00",
                    },
                    {
                        "id": 11,
                        "conversation_id": "conv-1",
                        "request_turn": 37,
                        "timestamp": "2026-04-15T04:42:18.486891+00:00",
                    },
                ])
            if "SELECT id, conversation_id, request_turn, timestamp FROM tool_calls" in sql:
                return _FakeRowsResult([
                    {
                        "id": 20,
                        "conversation_id": "conv-1",
                        "request_turn": 999,
                        "timestamp": "2026-04-15T04:50:00+00:00",
                    }
                ])
            return _FakeRowsResult([])

    conn = _NormalizeConn("conn-0")

    class _NormalizePool(_FakePool):
        def __init__(self, conninfo: str, **kwargs) -> None:
            super().__init__(conninfo, **kwargs)
            self.conn = conn

    monkeypatch.setattr(pg, "ConnectionPool", _NormalizePool)

    store = pg.PostgresStore("postgresql://example")
    conn.executed.clear()

    store._normalize_request_turn_sequences()

    assert (
        "UPDATE request_context SET request_turn = %s WHERE id = %s",
        (1, 10),
    ) in conn.executed
    assert (
        "UPDATE request_context SET request_turn = %s WHERE id = %s",
        (2, 11),
    ) in conn.executed
    assert (
        "UPDATE tool_calls SET request_turn = %s WHERE id = %s",
        (2, 20),
    ) in conn.executed
    assert any(
        sql.startswith("INSERT INTO request_turn_counters")
        and params == ("conv-1", 2)
        for sql, params in conn.executed
    )


# ---------------------------------------------------------------------------
# fact_embeddings schema bootstrap (real Postgres; DSN-gated, run -n0)
# ---------------------------------------------------------------------------

import threading as _threading

import pytest as _pytest

from tests.pg_helpers import pg_dsn as _pg_dsn, pg_test_conn as _pg_test_conn

_PG_URL = _pg_dsn()


@_pytest.mark.skipif(
    not _PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)
def test_fact_embeddings_schema_bootstrap_is_idempotent_under_parallel_startup():
    """Concurrent bootstrap converges: fact_embeddings + FK + index exist,
    and no worker raises on the required post-DDL catalog assertion."""
    from virtual_context.storage.postgres import PostgresStore

    errors: list[BaseException] = []

    def _boot():
        try:
            store = PostgresStore(_PG_URL)
            store.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [_threading.Thread(target=_boot) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors

    conn = _pg_test_conn()
    assert conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
        ("fact_embeddings",),
    ).fetchone() is not None
    assert conn.execute(
        "SELECT 1 FROM pg_indexes WHERE indexname = %s",
        ("idx_fact_embeddings_conv_model",),
    ).fetchone() is not None
    assert conn.execute(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE table_name = %s AND constraint_type = %s",
        ("fact_embeddings", "FOREIGN KEY"),
    ).fetchone() is not None
