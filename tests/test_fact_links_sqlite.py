"""Tests for fact link storage in SQLite."""

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, FactLink, LinkedFact


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    return SQLiteStore(db_path=str(db))


@pytest.fixture
def two_facts(store):
    """Store two facts and return them."""
    f1 = Fact(id="fact-1", subject="user", verb="led", object="Project Alpha", tags=["work"])
    f2 = Fact(id="fact-2", subject="user", verb="uses", object="Python", tags=["tech"])
    store.store_facts([f1, f2])
    return f1, f2


class TestStoreFactLinks:
    def test_store_and_retrieve(self, store, two_facts):
        f1, f2 = two_facts
        link = FactLink(
            source_fact_id=f1.id,
            target_fact_id=f2.id,
            relation_type="part_of",
            confidence=0.9,
            context="Project Alpha uses Python",
        )
        count = store.store_fact_links([link])
        assert count == 1

        links = store.get_fact_links(f1.id, direction="outgoing")
        assert len(links) == 1
        assert links[0].target_fact_id == f2.id
        assert links[0].relation_type == "part_of"

    def test_store_empty_list(self, store):
        assert store.store_fact_links([]) == 0

    def test_get_links_both_directions(self, store, two_facts):
        f1, f2 = two_facts
        link = FactLink(source_fact_id=f1.id, target_fact_id=f2.id, relation_type="caused_by")
        store.store_fact_links([link])

        out = store.get_fact_links(f1.id, direction="outgoing")
        assert len(out) == 1

        inc = store.get_fact_links(f2.id, direction="incoming")
        assert len(inc) == 1

        both = store.get_fact_links(f1.id, direction="both")
        assert len(both) == 1

    def test_delete_fact_links(self, store, two_facts):
        f1, f2 = two_facts
        link = FactLink(source_fact_id=f1.id, target_fact_id=f2.id, relation_type="related_to")
        store.store_fact_links([link])
        deleted = store.delete_fact_links(f1.id)
        assert deleted == 1
        assert store.get_fact_links(f1.id) == []


class TestGetLinkedFacts:
    def test_one_hop(self, store):
        f1 = Fact(id="f1", subject="user", verb="led", object="Alpha")
        f2 = Fact(id="f2", subject="Alpha", verb="uses", object="Python")
        f3 = Fact(id="f3", subject="Python", verb="has", object="FastAPI")
        store.store_facts([f1, f2, f3])

        store.store_fact_links([
            FactLink(source_fact_id="f1", target_fact_id="f2", relation_type="part_of"),
            FactLink(source_fact_id="f2", target_fact_id="f3", relation_type="part_of"),
        ])

        linked = store.get_linked_facts(["f1"], depth=1)
        assert len(linked) == 1
        assert linked[0].fact.id == "f2"
        assert linked[0].relation_type == "part_of"

    def test_two_hops(self, store):
        f1 = Fact(id="f1", subject="user", verb="led", object="Alpha")
        f2 = Fact(id="f2", subject="Alpha", verb="uses", object="Python")
        f3 = Fact(id="f3", subject="Python", verb="has", object="FastAPI")
        store.store_facts([f1, f2, f3])

        store.store_fact_links([
            FactLink(source_fact_id="f1", target_fact_id="f2", relation_type="part_of"),
            FactLink(source_fact_id="f2", target_fact_id="f3", relation_type="part_of"),
        ])

        linked = store.get_linked_facts(["f1"], depth=2)
        linked_ids = {lf.fact.id for lf in linked}
        assert "f2" in linked_ids
        assert "f3" in linked_ids

    def test_excludes_superseded_facts(self, store):
        f1 = Fact(id="f1", subject="user", verb="runs", object="5k")
        f2 = Fact(id="f2", subject="user", verb="ran", object="5k in 25:50", superseded_by="f1")
        store.store_facts([f1, f2])
        store.store_fact_links([
            FactLink(source_fact_id="f1", target_fact_id="f2", relation_type="supersedes"),
        ])

        linked = store.get_linked_facts(["f1"], depth=1)
        assert len(linked) == 0

    def test_empty_fact_ids(self, store):
        assert store.get_linked_facts([], depth=1) == []

    def test_no_links_returns_empty(self, store):
        f1 = Fact(id="f1", subject="user", verb="likes", object="cats")
        store.store_facts([f1])
        assert store.get_linked_facts(["f1"], depth=1) == []
