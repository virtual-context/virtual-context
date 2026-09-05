"""All fact discovery paths preserve the caller's structured constraints."""

from types import SimpleNamespace

import pytest

from virtual_context.core.fact_query import FactQueryEngine
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, FactLink


@pytest.fixture
def query_engine(tmp_path):
    store = SQLiteStore(tmp_path / "facts.db")
    engine = FactQueryEngine(
        store=store, semantic=SimpleNamespace(get_embed_fn=lambda: None),
        config=SimpleNamespace(conversation_id="audit", facts=SimpleNamespace(graph_links=True)),
    )
    return engine, store


def _fact(identifier, **kwargs):
    fields = dict(
        subject="Alice", verb="takes", object="aspirin", what="Alice takes aspirin",
        status="active", fact_type="personal", conversation_id="audit", tags=["health"],
    )
    fields.update(kwargs)
    return Fact(id=identifier, **fields)


def test_tag_siblings_preserve_object_status_and_type(query_engine):
    engine, store = query_engine
    store.store_facts([
        _fact("requested"),
        _fact("other-object", object="metformin"),
        _fact("other-status", status="completed"),
        _fact("other-type", fact_type="world"),
    ])
    result = engine.query(
        subject="Alice", object_contains="aspirin", status="active", fact_type="personal",
    )
    assert [fact.id for fact in result] == ["requested"]


def test_semantic_candidates_and_counts_share_all_nonstatus_filters(query_engine, monkeypatch):
    engine, store = query_engine
    selected = _fact("selected")
    completed = _fact("completed", status="completed")
    invalid = [
        _fact("other-object", object="metformin", what="Previously tried aspirin"),
        _fact("other-subject", subject="Bob"),
        _fact("other-type", fact_type="world"),
        _fact("other-owner", conversation_id="other"),
        _fact("other-tag", tags=["unrelated"]),
        _fact("other-verb", verb="refuses"),
        _fact("superseded", superseded_by="selected"),
    ]
    store.store_facts([selected, completed, *invalid])
    monkeypatch.setattr(engine, "_semantic_fact_search", lambda **kwargs: [completed, *invalid])
    result = engine.query(
        subject="Alice", verb="takes", object_contains="aspirin", status="active",
        fact_type="personal", tags=["health"], _return_meta=True,
    )
    assert [fact.id for fact in result["facts"]] == ["selected"]
    assert result["all_statuses"] == {"completed": 1, "active": 1}
    assert result["total_all_statuses"] == 2


def test_graph_neighbors_obey_query_constraints(query_engine):
    engine, store = query_engine
    selected = _fact("selected")
    unrelated = _fact("unrelated", object="metformin", status="completed")
    store.store_facts([selected, unrelated])
    store.store_fact_links([
        FactLink(source_fact_id=selected.id, target_fact_id=unrelated.id,
                 relation_type="related_to", confidence=0.9),
    ])
    result = engine.query(subject="Alice", object_contains="aspirin", _return_meta=True)
    assert [fact.id for fact in result["facts"]] == ["selected"]
    assert not result.get("linked_facts")


def test_candidate_union_preserves_explicit_limit(query_engine, monkeypatch):
    engine, store = query_engine
    candidates = [_fact(str(index)) for index in range(4)]
    store.store_facts(candidates)
    monkeypatch.setattr(engine, "_semantic_fact_search", lambda **kwargs: candidates)
    result = engine.query(subject="Alice", object_contains="aspirin", limit=1)
    assert len(result) == 1
