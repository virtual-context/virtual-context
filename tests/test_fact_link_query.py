"""Tests for auto-follow-links in vc_query_facts."""

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, FactLink


class TestAutoFollowLinks:
    def test_linked_facts_returned_from_store(self, tmp_path):
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        f1 = Fact(id="f1", subject="user", verb="led", object="Alpha", tags=["work"])
        f2 = Fact(id="f2", subject="Alpha", verb="uses", object="Python", tags=["tech"])
        f3 = Fact(id="f3", subject="user", verb="likes", object="cats", tags=["pets"])
        store.store_facts([f1, f2, f3])
        store.store_fact_links([
            FactLink(source_fact_id="f1", target_fact_id="f2", relation_type="part_of", confidence=0.9),
        ])

        # Primary query finds f1
        primary = store.query_facts(subject="user", verb="led")
        assert len(primary) == 1
        assert primary[0].id == "f1"

        # Link following finds f2
        linked = store.get_linked_facts(["f1"], depth=1)
        assert len(linked) == 1
        assert linked[0].fact.id == "f2"

    def test_no_linked_facts_when_no_links(self, tmp_path):
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        f1 = Fact(id="f1", subject="user", verb="likes", object="cats")
        store.store_facts([f1])
        linked = store.get_linked_facts(["f1"], depth=1)
        assert len(linked) == 0
