"""Targeting get_all_segments by ref.

The invariant that carries the safety here is the empty set. ``None`` means
no filter and an empty collection means nothing -- a caller that computes a
target set, gets zero back, and then runs unbounded over every segment is the
failure this filter exists to make impossible.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.storage.postgres import PostgresStore
from virtual_context.types import SegmentMetadata, StoredSegment


def _seg(ref: str, conv: str = "c") -> StoredSegment:
    return StoredSegment(
        ref=ref, conversation_id=conv, primary_tag="t", tags=["t"],
        summary=f"s-{ref}", messages=[], metadata=SegmentMetadata(),
        created_at="2026-08-21T00:00:00+00:00",
        start_timestamp="2026-08-21T00:00:00+00:00",
        end_timestamp="2026-08-21T00:00:00+00:00",
    )


def _sqlite(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    for r in ("a", "b", "d"):
        s.store_segment(_seg(r))
    return s


def _refs(segs):
    return sorted(x.ref for x in segs)


# --- sqlite: a real backend -----------------------------------------------

def test_sqlite_none_means_no_filter(tmp_path):
    s = _sqlite(tmp_path)
    assert _refs(s.get_all_segments(conversation_id="c")) == ["a", "b", "d"]
    assert _refs(s.get_all_segments(conversation_id="c", segment_refs=None)) == ["a", "b", "d"]


def test_sqlite_empty_set_returns_nothing_not_everything(tmp_path):
    """THE safety property. An empty target must not become a full run."""
    s = _sqlite(tmp_path)
    assert s.get_all_segments(conversation_id="c", segment_refs=set()) == []
    assert s.get_all_segments(conversation_id="c", segment_refs=[]) == []


def test_sqlite_selects_exactly_the_named_refs(tmp_path):
    s = _sqlite(tmp_path)
    assert _refs(s.get_all_segments(conversation_id="c", segment_refs={"a", "d"})) == ["a", "d"]


def test_sqlite_unknown_ref_is_simply_absent(tmp_path):
    s = _sqlite(tmp_path)
    assert _refs(s.get_all_segments(conversation_id="c", segment_refs={"a", "nope"})) == ["a"]


def test_sqlite_ref_filter_composes_with_conversation(tmp_path):
    s = _sqlite(tmp_path)
    s.upsert_conversation(tenant_id="t", conversation_id="other")
    s.store_segment(_seg("z", conv="other"))
    assert s.get_all_segments(conversation_id="c", segment_refs={"z"}) == []
    assert _refs(s.get_all_segments(conversation_id="other", segment_refs={"z"})) == ["z"]


# --- filesystem: the other real backend -----------------------------------

def test_filesystem_empty_set_returns_nothing(tmp_path):
    s = FilesystemStore(tmp_path / "fs")
    for r in ("a", "b"):
        s.store_segment(_seg(r))
    assert _refs(s.get_all_segments(conversation_id="c")) == ["a", "b"]
    assert s.get_all_segments(conversation_id="c", segment_refs=set()) == []
    assert _refs(s.get_all_segments(conversation_id="c", segment_refs={"b"})) == ["b"]


# --- postgres: the SQL it builds ------------------------------------------

class _Conn:
    def __init__(self): self.sql = None; self.params = None
    def execute(self, sql, params=None):
        self.sql, self.params = sql, params; return self
    def fetchall(self): return []
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Pool:
    def __init__(self, c): self._c = c; self.opened = 0
    def connection(self):
        self.opened += 1; return self._c


def _pg():
    s = PostgresStore.__new__(PostgresStore)
    c = _Conn(); s.pool = _Pool(c)
    return s, c


def test_postgres_empty_set_never_reaches_the_database(tmp_path):
    """Not merely an empty result -- no query at all."""
    s, c = _pg()
    assert s.get_all_segments(conversation_id="c", segment_refs=set()) == []
    assert s.pool.opened == 0, "opened a connection for an empty target set"
    assert c.sql is None


def test_postgres_none_builds_no_ref_predicate():
    s, c = _pg()
    s.get_all_segments(conversation_id="c")
    assert "ref = ANY" not in c.sql
    assert c.params == ["c"]


def test_postgres_refs_build_an_any_predicate():
    s, c = _pg()
    s.get_all_segments(conversation_id="c", segment_refs=["r1", "r2"])
    assert "ref = ANY(%s)" in c.sql
    assert c.params == ["c", ["r1", "r2"]]


def test_postgres_limit_still_applies_after_refs():
    s, c = _pg()
    s.get_all_segments(conversation_id="c", segment_refs=["r1"], limit=5)
    assert c.sql.rstrip().endswith("LIMIT %s")
    assert c.params == ["c", ["r1"], 5]


def test_postgres_refs_without_conversation():
    s, c = _pg()
    s.get_all_segments(segment_refs=["r1"])
    assert "conversation_id" not in c.sql
    assert c.params == [["r1"]]


# --- the interface is uniform --------------------------------------------

@pytest.mark.parametrize("cls", [PostgresStore, SQLiteStore, FilesystemStore])
def test_every_backend_accepts_the_parameter_keyword_only(cls):
    import inspect
    p = inspect.signature(cls.get_all_segments).parameters["segment_refs"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_composite_forwards_the_parameter():
    from virtual_context.core.composite_store import CompositeStore
    seen = {}

    class Seg:
        def get_all_segments(self, **kw):
            seen.update(kw); return []
    c = CompositeStore.__new__(CompositeStore)
    c._segments = Seg()
    c.get_all_segments(conversation_id="c", segment_refs={"r"})
    assert seen["segment_refs"] == {"r"}, "composite dropped the filter"
