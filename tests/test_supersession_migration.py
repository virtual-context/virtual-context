"""Tests for migrating superseded_by column to SUPERSEDES fact links."""

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact


class TestSupersessionMigration:
    def test_migration_creates_links_from_superseded_by(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path=db_path)
        old_fact = Fact(id="old-1", subject="user", verb="lives-in", object="NYC")
        new_fact = Fact(id="new-1", subject="user", verb="lives-in", object="Chicago")
        store.store_facts([old_fact, new_fact])
        store.set_fact_superseded("old-1", "new-1")

        store.migrate_supersession_to_links()

        links = store.get_fact_links("old-1", direction="incoming")
        assert len(links) == 1
        assert links[0].source_fact_id == "new-1"
        assert links[0].target_fact_id == "old-1"
        assert links[0].relation_type == "supersedes"
        assert links[0].created_by == "migration"

    def test_migration_is_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path=db_path)
        old = Fact(id="old", subject="user", verb="has", object="dog")
        new = Fact(id="new", subject="user", verb="has", object="cat")
        store.store_facts([old, new])
        store.set_fact_superseded("old", "new")

        store.migrate_supersession_to_links()
        store.migrate_supersession_to_links()

        links = store.get_fact_links("old")
        assert len(links) == 1

    def test_migration_no_superseded_facts(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path=db_path)
        f = Fact(id="f1", subject="user", verb="likes", object="cats")
        store.store_facts([f])

        store.migrate_supersession_to_links()
        assert store.get_fact_links("f1") == []
