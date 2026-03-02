"""Tests for enriched fact extraction fields (fact_type, what)."""

from virtual_context.types import FactSignal, Fact
from virtual_context.storage.sqlite import SQLiteStore


class TestFactSignalEnrichment:
    def test_fact_signal_has_fact_type_default(self):
        fs = FactSignal(subject="user", verb="runs", object="5K")
        assert fs.fact_type == "personal"

    def test_fact_signal_accepts_fact_type(self):
        fs = FactSignal(subject="user", verb="runs", object="5K", fact_type="experience")
        assert fs.fact_type == "experience"

    def test_fact_signal_has_what_default(self):
        fs = FactSignal(subject="user", verb="runs", object="5K")
        assert fs.what == ""

    def test_fact_signal_accepts_what(self):
        fs = FactSignal(subject="user", verb="runs", object="5K",
                        what="User runs a 5K charity race every spring.")
        assert fs.what == "User runs a 5K charity race every spring."


class TestFactEnrichment:
    def test_fact_has_fact_type_default(self):
        f = Fact(subject="user", verb="runs", object="5K")
        assert f.fact_type == "personal"

    def test_fact_accepts_fact_type(self):
        f = Fact(subject="user", verb="runs", object="5K", fact_type="world")
        assert f.fact_type == "world"

    def test_fact_type_values(self):
        for ft in ("personal", "experience", "world"):
            f = Fact(fact_type=ft)
            assert f.fact_type == ft


class TestSQLiteFactType:
    def test_store_and_query_fact_type(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        f = Fact(
            subject="user", verb="runs", object="5K",
            fact_type="experience", what="User runs 5K races.",
        )
        store.store_facts([f])
        results = store.query_facts(subject="user")
        assert len(results) == 1
        assert results[0].fact_type == "experience"

    def test_fact_type_defaults_to_personal_in_db(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        f = Fact(subject="user", verb="cooks", object="pasta")
        store.store_facts([f])
        results = store.query_facts(subject="user")
        assert results[0].fact_type == "personal"

    def test_query_facts_with_fact_type_filter(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.store_facts([
            Fact(subject="user", verb="runs", object="5K", fact_type="personal"),
            Fact(subject="user", verb="learned", object="interval training", fact_type="experience"),
            Fact(subject="Emily", verb="lives in", object="Portland", fact_type="world"),
        ])
        personal = store.query_facts(subject="user", fact_type="personal")
        assert len(personal) == 1
        assert personal[0].verb == "runs"
