"""Focused unit tests for PostgresStore connection management."""

from __future__ import annotations

import threading


class _FakeConn:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, sql: str, params=None):
        self.executed.append((sql, params))
        return self

    def close(self) -> None:
        self.closed = True


def test_postgres_store_uses_thread_local_connections(monkeypatch):
    from virtual_context.storage import postgres as pg

    created: list[_FakeConn] = []

    def _fake_connect(*args, **kwargs):
        conn = _FakeConn(f"conn-{len(created)}")
        created.append(conn)
        return conn

    monkeypatch.setattr(pg.psycopg, "connect", _fake_connect)

    store = pg.PostgresStore("postgresql://example")
    main_conn = store._get_conn()

    from_worker: list[_FakeConn] = []

    def _worker() -> None:
        from_worker.append(store._get_conn())

    worker = threading.Thread(target=_worker)
    worker.start()
    worker.join()

    assert store._get_conn() is main_conn
    assert from_worker == [created[1]]
    assert from_worker[0] is not main_conn
    assert len(created) == 2

    store.close()

    assert all(conn.closed for conn in created)
