"""DSN-gated transaction and lock checks for the remote PostgreSQL fleet."""

from contextlib import contextmanager
import threading
import uuid

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn
from virtual_context.types import Fact


pytestmark = pytest.mark.skipif(not pg_dsn(), reason="PostgreSQL fleet DSN not configured")


@pytest.fixture
def fact_store():
    from virtual_context.storage.postgres import PostgresStore

    store = PostgresStore(pg_dsn())
    conversation = f"mutation-{uuid.uuid4().hex}"
    fact = Fact(id=conversation, conversation_id=conversation, subject="user", verb="likes", object="old")
    store.store_facts([fact])
    store.store_fact_embeddings(fact.id, conversation, "model", [1.0, 0.0])
    try:
        yield store, fact
    finally:
        store.delete_conversation(conversation)
        store.close()


def test_fact_rewrite_rolls_back_when_vector_invalidation_fails(fact_store):
    import psycopg
    from psycopg import sql

    store, fact = fact_store
    conn = pg_test_conn()
    name = sql.Identifier(f"reject_vector_delete_{uuid.uuid4().hex}")
    # Only this test's unique fact is affected while the trigger is installed.
    body = (
        "BEGIN IF OLD.fact_id = " + sql.Literal(fact.id).as_string(conn)
        + " THEN RAISE EXCEPTION 'injected vector delete failure'; END IF; RETURN OLD; END;"
    )
    conn.execute(sql.SQL("CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS {}").format(name, sql.Literal(body)))
    try:
        conn.execute(sql.SQL("CREATE TRIGGER {} BEFORE DELETE ON fact_embeddings FOR EACH ROW EXECUTE FUNCTION {}()").format(name, name))
        with pytest.raises(psycopg.errors.RaiseException, match="injected vector delete failure"):
            store.update_fact_fields(fact.id, "likes", "new", "active", "")
        assert store.query_facts(conversation_id=fact.conversation_id)[0].object == "old"
        assert store.load_fact_embeddings(fact.conversation_id, "model")[fact.id][1] == [1.0, 0.0]
    finally:
        conn.execute(sql.SQL("DROP TRIGGER IF EXISTS {} ON fact_embeddings").format(name))
        conn.execute(sql.SQL("DROP FUNCTION IF EXISTS {}()").format(name))


def test_fact_pre_read_locks_row_until_rewrite_and_invalidation_finish(fact_store, monkeypatch):
    import psycopg

    store, fact = fact_store
    real_pool = store.pool
    read_finished = threading.Event()
    release = threading.Event()
    errors = []

    class PausedConnection:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def execute(self, statement, params=None):
            result = self.conn.execute(statement, params)
            normalized = " ".join(str(statement).split())
            if "FROM facts WHERE id=" in normalized and normalized.endswith("FOR UPDATE"):
                read_finished.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release locked fact")
            return result

    class PausedPool:
        @contextmanager
        def connection(self):
            with real_pool.connection() as conn:
                yield PausedConnection(conn)

    def write():
        try:
            store.update_fact_fields(fact.id, "likes", "new", "active", "")
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(store, "pool", PausedPool())
    thread = threading.Thread(target=write)
    thread.start()
    try:
        assert read_finished.wait(5)
        # NOWAIT proves actual exclusion instead of inferring it from elapsed
        # time or merely asserting that the emitted SQL contains FOR UPDATE.
        with pytest.raises(psycopg.errors.LockNotAvailable):
            pg_test_conn().execute(
                "SELECT id FROM facts WHERE id=%s FOR UPDATE NOWAIT", (fact.id,),
            )
    finally:
        release.set()
        thread.join(timeout=5)
        monkeypatch.setattr(store, "pool", real_pool)
    assert not thread.is_alive()
    assert not errors
    assert store.query_facts(conversation_id=fact.conversation_id)[0].object == "new"
    assert store.load_fact_embeddings(fact.conversation_id, "model") == {}
