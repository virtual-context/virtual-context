"""Tests for FactLink and LinkedFact dataclasses."""

from virtual_context.types import Fact, FactLink, LinkedFact, RelationType


class TestRelationType:
    def test_all_types_exist(self):
        assert RelationType.SUPERSEDES == "supersedes"
        assert RelationType.CAUSED_BY == "caused_by"
        assert RelationType.PART_OF == "part_of"
        assert RelationType.CONTRADICTS == "contradicts"
        assert RelationType.SAME_AS == "same_as"
        assert RelationType.RELATED_TO == "related_to"

    def test_is_string_enum(self):
        assert isinstance(RelationType.SUPERSEDES, str)
        assert RelationType.SUPERSEDES == "supersedes"


class TestFactLink:
    def test_defaults(self):
        link = FactLink(source_fact_id="a", target_fact_id="b", relation_type="supersedes")
        assert link.id  # auto-generated UUID
        assert link.confidence == 1.0
        assert link.context == ""
        assert link.created_by == "compaction"
        assert link.created_at is not None

    def test_all_fields(self):
        link = FactLink(
            source_fact_id="src",
            target_fact_id="tgt",
            relation_type="caused_by",
            confidence=0.85,
            context="A caused B because of deadline pressure",
            created_by="migration",
        )
        assert link.source_fact_id == "src"
        assert link.target_fact_id == "tgt"
        assert link.relation_type == "caused_by"
        assert link.confidence == 0.85


class TestLinkedFact:
    def test_wraps_fact_with_link_metadata(self):
        fact = Fact(subject="user", verb="led", object="Project Alpha")
        linked = LinkedFact(
            fact=fact,
            linked_from_fact_id="src-fact-id",
            relation_type="part_of",
            confidence=0.9,
            link_context="Project Alpha is part of user's portfolio",
        )
        assert linked.fact.subject == "user"
        assert linked.relation_type == "part_of"
        assert linked.confidence == 0.9
