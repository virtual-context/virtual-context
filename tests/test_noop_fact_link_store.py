"""Tests for NoopFactLinkStore — used when graph_links is disabled."""

from virtual_context.storage.noop_fact_link_store import NoopFactLinkStore
from virtual_context.types import FactLink


class TestNoopFactLinkStore:
    def test_store_returns_zero(self):
        store = NoopFactLinkStore()
        assert store.store_fact_links([FactLink(source_fact_id="a", target_fact_id="b", relation_type="x")]) == 0

    def test_get_fact_links_returns_empty(self):
        store = NoopFactLinkStore()
        assert store.get_fact_links("any-id") == []

    def test_get_linked_facts_returns_empty(self):
        store = NoopFactLinkStore()
        assert store.get_linked_facts(["f1", "f2"], depth=2) == []

    def test_delete_returns_zero(self):
        store = NoopFactLinkStore()
        assert store.delete_fact_links("any-id") == 0
