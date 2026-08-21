"""Relevance ordering of the tag-gated fact floor.

The invariants that matter here are all about what does NOT happen: other
callers must keep date ordering, a half-backfilled column must never be read,
and the pre-backfill window must not pay for an embedding nobody can use.
"""
from __future__ import annotations

import tempfile, os
import pytest

from virtual_context.config import load_config
from virtual_context.core.retriever import ContextRetriever
from virtual_context.storage.postgres import PostgresStore


class _Store:
    def __init__(self, ready=True):
        self._ready = ready
        self.calls: list[dict] = []

    def vector_ordering_ready(self):
        return self._ready

    def query_facts(self, **kwargs):
        self.calls.append(kwargs)
        return []


class _Cfg:
    fact_relevance_ordering = True
    prefetch_facts = True
    fact_dense_retrieval = False


def _retriever(store, cfg=None, embed=None):
    r = ContextRetriever.__new__(ContextRetriever)
    r.store = store
    r.config = cfg or _Cfg()
    r._conversation_id = "conv-1"
    r._query_embed_fn = embed
    return r


# --- the floor is opt-in -------------------------------------------------

def test_floor_without_embedding_never_mentions_the_parameter():
    """Date ordering must be the shape of the CALL, not just its default."""
    s = _Store()
    _retriever(s)._fetch_facts_by_tags(["t"])
    assert "order_by_embedding" not in s.calls[0]


def test_floor_with_embedding_passes_it():
    s = _Store()
    _retriever(s)._fetch_facts_by_tags(["t"], query_embedding=[0.1, 0.2])
    assert s.calls[0]["order_by_embedding"] == [0.1, 0.2]


def test_store_rejecting_the_parameter_falls_back_to_date_order():
    class Old(_Store):
        def query_facts(self, **kwargs):
            if "order_by_embedding" in kwargs:
                raise TypeError("query_facts() got an unexpected keyword "
                                "argument 'order_by_embedding'")
            return super().query_facts(**kwargs)
    s = Old()
    out = _retriever(s)._fetch_facts_by_tags(["t"], query_embedding=[0.1])
    assert out == [] and len(s.calls) == 1
    assert "order_by_embedding" not in s.calls[0]


def test_unrelated_typeerror_is_not_retried():
    """A genuine bug inside the query must not be masked as a fallback."""
    calls = []
    class Broken(_Store):
        def query_facts(self, **kwargs):
            calls.append(kwargs)
            raise TypeError("unsupported operand type(s)")
    assert _retriever(Broken())._fetch_facts_by_tags(["t"], query_embedding=[0.1]) == []
    assert len(calls) == 1


# --- the pre-backfill window costs nothing -------------------------------

def test_unready_store_does_not_embed():
    """The whole deploy-to-backfill window must not pay for a refused vector."""
    embedded = []
    r = _retriever(_Store(ready=False), embed=lambda xs: embedded.append(xs) or [[0.1]])
    assert r._relevance_vector("hello") is None
    assert embedded == [], "embedded a query the store cannot use"


def test_ready_store_embeds_once():
    embedded = []
    r = _retriever(_Store(ready=True), embed=lambda xs: embedded.append(xs) or [[0.1, 0.2]])
    assert r._relevance_vector("hello") == [0.1, 0.2]
    assert embedded == [["hello"]]


def test_kill_switch_off_skips_store_and_embedder():
    class Off(_Cfg):
        fact_relevance_ordering = False
    s = _Store(ready=True)
    embedded = []
    r = _retriever(s, cfg=Off(), embed=lambda xs: embedded.append(xs) or [[0.1]])
    assert r._relevance_vector("hello") is None
    assert embedded == []


def test_embedder_failure_is_not_fatal():
    def boom(_):
        raise RuntimeError("model gone")
    assert _retriever(_Store(), embed=boom)._relevance_vector("hi") is None


def test_no_embedder_configured():
    assert _retriever(_Store(), embed=None)._relevance_vector("hi") is None


def test_store_without_the_gate_declines():
    class Bare:
        def query_facts(self, **kw): return []
    r = _retriever(Bare(), embed=lambda xs: [[0.1]])
    assert r._relevance_vector("hi") is None


# --- the completeness gate is a predicate, not a flag --------------------

class _Conn:
    def __init__(self, row, exc=None):
        self._row, self._exc = row, exc
        self.sql: list[str] = []
    def execute(self, sql, *a):
        if self._exc:
            raise self._exc
        self.sql.append(sql)
        return self
    def fetchone(self):
        return self._row
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Pool:
    def __init__(self, conn): self._c = conn; self.opened = 0
    def connection(self):
        self.opened += 1
        return self._c


def _store_with(row=None, exc=None):
    s = PostgresStore.__new__(PostgresStore)
    s.pool = _Pool(_Conn(row, exc))
    return s


def test_gate_true_only_when_no_null_remains():
    assert PostgresStore.vector_ordering_ready(_store_with(row=None)) is True


def test_gate_false_when_any_null_remains():
    """A single unconverted row must veto the whole read path."""
    assert PostgresStore.vector_ordering_ready(_store_with(row=(1,))) is False


def test_gate_false_when_column_or_extension_missing():
    s = _store_with(exc=Exception('column "embedding" does not exist'))
    assert PostgresStore.vector_ordering_ready(s) is False


def test_gate_is_cached():
    s = _store_with(row=None)
    assert PostgresStore.vector_ordering_ready(s) is True
    assert PostgresStore.vector_ordering_ready(s) is True
    assert s.pool.opened == 1


def test_gate_caches_false_briefly():
    s = _store_with(row=(1,))
    PostgresStore.vector_ordering_ready(s)
    PostgresStore.vector_ordering_ready(s)
    assert s.pool.opened == 1


def test_false_is_rechecked_so_a_backfill_needs_no_restart():
    """A process that started before the backfill must pick the column up.

    Caching False permanently would leave every long-lived process on date
    ordering until it was restarted, which reads as 'the feature did not turn
    on' rather than as a stale cache.
    """
    import virtual_context.storage.postgres as pg
    conn = _Conn((1,))                      # a NULL still remains
    s = PostgresStore.__new__(PostgresStore)
    s.pool = _Pool(conn)
    clock = [1000.0]
    real = pg.time.monotonic
    pg.time.monotonic = lambda: clock[0]
    try:
        assert PostgresStore.vector_ordering_ready(s) is False
        clock[0] += 5                        # inside the window: no new query
        assert PostgresStore.vector_ordering_ready(s) is False
        assert s.pool.opened == 1
        conn._row = None                     # backfill completes
        clock[0] += 120                      # window expires
        assert PostgresStore.vector_ordering_ready(s) is True
        assert s.pool.opened == 2
        clock[0] += 10_000                   # True is never re-checked
        assert PostgresStore.vector_ordering_ready(s) is True
        assert s.pool.opened == 2
    finally:
        pg.time.monotonic = real


# --- the parameter cannot be reached positionally ------------------------

def test_parameter_is_keyword_only():
    """Positional reachability would silently reorder some other caller."""
    import inspect
    p = inspect.signature(PostgresStore.query_facts).parameters["order_by_embedding"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_kill_switch_is_wired_to_yaml():
    """A field with no parser is inert; this pins the parser, not the field."""
    def load(body):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(body); n = f.name
        try: return load_config(n)
        finally: os.unlink(n)
    assert load("retrieval: {}\n").retriever.fact_relevance_ordering is True
    assert load("retrieval:\n  fact_relevance_ordering: false\n"
                ).retriever.fact_relevance_ordering is False
