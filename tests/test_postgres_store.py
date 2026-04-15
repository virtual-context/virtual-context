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

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    def executemany(self, sql: str, params_seq):
        for params in params_seq:
            self.executed.append((sql, params))
        return self

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

    monkeypatch.setattr(pg.psycopg, "connect", lambda *args, **kwargs: conn)

    store = pg.PostgresStore("postgresql://example")
    monkeypatch.setattr(store, "_batch_get_tags", lambda refs: {"seg-1": ["tag-a", "tag-b"]})

    segments = store.get_all_segments(conversation_id="conv-1")

    assert len(segments) == 1
    assert segments[0].ref == "seg-1"
    assert segments[0].tags == ["tag-a", "tag-b"]
