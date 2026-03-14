"""Integration test: full pipeline with graph_links enabled."""

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.storage.noop_fact_link_store import NoopFactLinkStore
from virtual_context.core.composite_store import CompositeStore
from virtual_context.types import Fact, FactLink, LinkedFact


class TestFullPipeline:
    def test_sqlite_composite_with_links(self, tmp_path):
        """SQLite as all-in-one backend with graph_links enabled."""
        sqlite = SQLiteStore(db_path=str(tmp_path / "test.db"))
        comp = CompositeStore(
            segments=sqlite, facts=sqlite, fact_links=sqlite,
            state=sqlite, search=sqlite,
        )

        f1 = Fact(id="f1", subject="user", verb="led", object="Alpha")
        f2 = Fact(id="f2", subject="Alpha", verb="uses", object="Python")
        comp.store_facts([f1, f2])

        link = FactLink(source_fact_id="f1", target_fact_id="f2", relation_type="part_of")
        comp.store_fact_links([link])

        results = comp.query_facts(subject="user", verb="led")
        assert len(results) == 1

        linked = comp.get_linked_facts(["f1"], depth=1)
        assert len(linked) == 1
        assert linked[0].fact.id == "f2"

    def test_sqlite_composite_without_links(self, tmp_path):
        """SQLite with NoopFactLinkStore (graph_links disabled)."""
        sqlite = SQLiteStore(db_path=str(tmp_path / "test.db"))
        comp = CompositeStore(
            segments=sqlite, facts=sqlite, fact_links=NoopFactLinkStore(),
            state=sqlite, search=sqlite,
        )

        f1 = Fact(id="f1", subject="user", verb="led", object="Alpha")
        comp.store_facts([f1])

        link = FactLink(source_fact_id="f1", target_fact_id="f2", relation_type="part_of")
        assert comp.store_fact_links([link]) == 0
        assert comp.get_linked_facts(["f1"]) == []

    def test_migration_then_query(self, tmp_path):
        """Migrate superseded_by data, then query with link following."""
        sqlite = SQLiteStore(db_path=str(tmp_path / "test.db"))

        old = Fact(id="old", subject="user", verb="lives-in", object="NYC")
        new = Fact(id="new", subject="user", verb="lives-in", object="Chicago")
        sqlite.store_facts([old, new])
        sqlite.set_fact_superseded("old", "new")

        sqlite.migrate_supersession_to_links()

        links = sqlite.get_fact_links("old")
        assert len(links) == 1
        assert links[0].relation_type == "supersedes"

    def test_composite_close(self, tmp_path):
        """CompositeStore.close() delegates without error."""
        sqlite = SQLiteStore(db_path=str(tmp_path / "test.db"))
        comp = CompositeStore(
            segments=sqlite, facts=sqlite, fact_links=sqlite,
            state=sqlite, search=sqlite,
        )
        comp.close()  # should not raise

    def test_composite_close_with_noop(self, tmp_path):
        """close() works with NoopFactLinkStore (no close method)."""
        sqlite = SQLiteStore(db_path=str(tmp_path / "test.db"))
        comp = CompositeStore(
            segments=sqlite, facts=sqlite, fact_links=NoopFactLinkStore(),
            state=sqlite, search=sqlite,
        )
        comp.close()  # should not raise
