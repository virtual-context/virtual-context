"""Tests for enriched fact extraction fields (fact_type, what)."""

from virtual_context.types import FactSignal, Fact


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
