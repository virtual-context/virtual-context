"""Real SQLite checks for row identity, FTS parity, and targeted repair."""

import ast
import sqlite3
from pathlib import Path

import pytest

from virtual_context.core.compaction_fence import CompactionFenceMode
from virtual_context.core.composite_store import CompositeStore
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, FactLink, StoredSegment, TagSummary


@pytest.fixture
def store(tmp_path):
    result = SQLiteStore(tmp_path / "upserts.db", compaction_fence_mode=CompactionFenceMode.ACTIVE)
    yield result
    result.close()


def guard(store, enabled):
    if not enabled:
        return {}
    store.upsert_conversation(tenant_id="tenant", conversation_id="c")
    operation = store.start_compaction_operation(
        conversation_id="c", lifecycle_epoch=1, worker_id="worker",
        phase_count=1, phase_name="persist",
    )
    store._get_conn().execute(
        "UPDATE compaction_operation SET status='running' WHERE operation_id=?", (operation,),
    )
    return dict(operation_id=operation, owner_worker_id="worker", lifecycle_epoch=1)


@pytest.mark.parametrize("guarded", [False, True])
def test_segment_upsert_preserves_identity_and_search(store, guarded):
    kwargs = guard(store, guarded)
    segment = StoredSegment(ref="s", conversation_id="c", summary="aardvark", full_text="aardvark")
    store.store_segment(segment, **kwargs)
    store.store_facts([Fact(id="f", conversation_id="c", segment_ref="s")])
    conn = store._get_conn()
    original = conn.execute("SELECT rowid FROM segments WHERE ref='s'").fetchone()[0]
    segment.summary = segment.full_text = "zebra"
    store.update_segment(segment, **kwargs)
    assert conn.execute("SELECT rowid FROM segments WHERE ref='s'").fetchone()[0] == original
    assert store.get_facts_by_segment("s")[0].id == "f"
    assert store.search("aardvark") == []
    assert [item.ref for item in store.search("zebra")] == ["s"]
    assert store.repair_fts_indexes(["segments_fts", "segments_fts_full"]) == {
        "segments_fts": "ok", "segments_fts_full": "ok",
    }


@pytest.mark.parametrize("guarded", [False, True])
def test_fact_upsert_preserves_links_and_invalidates_only_changed_vectors(store, guarded):
    kwargs = guard(store, guarded)
    fact = Fact(id="f", conversation_id="c", subject="user", verb="likes", object="aardvark")
    store.store_facts([fact, Fact(id="g", conversation_id="c")], **kwargs)
    store.store_fact_links([FactLink(id="l", source_fact_id="f", target_fact_id="g", relation_type="related_to")])
    store.store_fact_embeddings("f", "c", "model", [1.0, 0.0])
    conn = store._get_conn()
    original = conn.execute("SELECT rowid FROM facts WHERE id='f'").fetchone()[0]
    store.store_facts([fact], **kwargs)
    assert conn.execute("SELECT rowid FROM facts WHERE id='f'").fetchone()[0] == original
    assert [link.id for link in store.get_fact_links("f")] == ["l"]
    assert "f" in store.load_fact_embeddings("c", "model")
    fact.object = "zebra"
    store.store_facts([fact], **kwargs)
    assert store.load_fact_embeddings("c", "model") == {}
    assert [link.id for link in store.get_fact_links("f")] == ["l"]
    assert store.search_facts("aardvark") == []
    assert [item.id for item in store.search_facts("zebra")] == ["f"]


def test_tag_and_tool_upserts_keep_fts_consistent(store):
    for text in ["aardvark", "zebra"]:
        store.save_tag_summary(TagSummary(tag="topic", summary=text), conversation_id="c")
        store.store_tool_output("tool", "c", "read", "", 1, text, len(text))
    assert store.search_tag_summaries_fts("aardvark", conversation_id="c") == []
    assert store.search_tag_summaries_fts("zebra", conversation_id="c")
    assert store.search_tool_outputs("aardvark", conversation_id="c") == []
    assert store.search_tool_outputs("zebra", conversation_id="c")
    assert set(store.repair_fts_indexes().values()) == {"ok"}


@pytest.mark.parametrize("table,index", [
    ("segments", "segments_fts"), ("facts", "facts_fts"),
    ("tag_summaries", "tag_summaries_fts"), ("tool_outputs", "tool_outputs_fts"),
])
def test_repair_is_targeted_dry_by_default_and_preserves_source_rows(store, table, index):
    store.store_segment(StoredSegment(ref="s", conversation_id="c", summary="aardvark"))
    store.store_facts([Fact(id="f", conversation_id="c", object="aardvark")])
    store.save_tag_summary(TagSummary(tag="topic", summary="aardvark"), conversation_id="c")
    store.store_tool_output("tool", "c", "read", "", 1, "aardvark", 8)
    conn = store._get_conn()
    # Reproduce the old write semantics, leaving postings for a removed rowid.
    conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM {table}")
    before = [tuple(row) for row in conn.execute(f"SELECT rowid, * FROM {table}")]
    assert store.repair_fts_indexes([index]) == {index: "needs_rebuild"}
    assert store.repair_fts_indexes([index]) == {index: "needs_rebuild"}
    assert store.repair_fts_indexes([index], dry_run=False) == {index: "rebuilt"}
    assert store.repair_fts_indexes([index]) == {index: "ok"}
    assert [tuple(row) for row in conn.execute(f"SELECT rowid, * FROM {table}")] == before


def test_repair_rejects_unknown_indexes_and_does_not_hide_missing_tables(store):
    with pytest.raises(ValueError, match="Unsupported FTS indexes"):
        store.repair_fts_indexes(["facts"])
    store._get_conn().execute("DROP TABLE facts_fts")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        store.repair_fts_indexes(["facts_fts"], dry_run=False)


@pytest.mark.parametrize("filename,class_name", [
    ("neo4j.py", "Neo4jFactStore"), ("falkordb.py", "FalkorDBFactStore"),
])
def test_graph_utilities_are_refused_as_engine_delegates(filename, class_name):
    # Load the actual capability declaration without optional network drivers.
    tree = ast.parse((Path(__file__).parents[1] / "virtual_context/storage" / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    declaration = next(node for node in cls.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "supports_engine_lifecycle" for target in node.targets))
    capability = ast.literal_eval(declaration.value)
    delegate = type(class_name, (), {"supports_engine_lifecycle": capability})()
    with pytest.raises(ValueError, match="durable lifecycle fencing"):
        CompositeStore(segments=object(), facts=delegate, fact_links=delegate,
                       state=object(), search=object())
