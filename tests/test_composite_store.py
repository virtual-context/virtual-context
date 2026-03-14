"""Tests for CompositeStore delegation."""

from unittest.mock import MagicMock

from virtual_context.core.composite_store import CompositeStore
from virtual_context.types import Fact, FactLink


class TestCompositeStoreDelegation:
    def _make_composite(self):
        segments = MagicMock()
        facts = MagicMock()
        fact_links = MagicMock()
        state = MagicMock()
        search = MagicMock()
        composite = CompositeStore(
            segments=segments, facts=facts, fact_links=fact_links,
            state=state, search=search,
        )
        return composite, segments, facts, fact_links, state, search

    def test_store_segment_delegates(self):
        comp, segments, *_ = self._make_composite()
        comp.store_segment("seg")
        segments.store_segment.assert_called_once_with("seg")

    def test_store_facts_delegates(self):
        comp, _, facts, *_ = self._make_composite()
        fact_list = [Fact(subject="user")]
        comp.store_facts(fact_list)
        facts.store_facts.assert_called_once_with(fact_list)

    def test_store_fact_links_delegates(self):
        comp, _, _, fact_links, *_ = self._make_composite()
        links = [FactLink(source_fact_id="a", target_fact_id="b", relation_type="related_to")]
        comp.store_fact_links(links)
        fact_links.store_fact_links.assert_called_once_with(links)

    def test_save_engine_state_delegates(self):
        comp, _, _, _, state, _ = self._make_composite()
        comp.save_engine_state("state")
        state.save_engine_state.assert_called_once_with("state")

    def test_search_full_text_delegates(self):
        comp, _, _, _, _, search = self._make_composite()
        comp.search_full_text("query")
        search.search_full_text.assert_called_once_with("query")

    def test_query_facts_delegates(self):
        comp, _, facts, *_ = self._make_composite()
        facts.query_facts.return_value = []
        comp.query_facts(subject="user")
        facts.query_facts.assert_called_once_with(subject="user")

    def test_get_linked_facts_delegates(self):
        comp, _, _, fact_links, *_ = self._make_composite()
        fact_links.get_linked_facts.return_value = []
        comp.get_linked_facts(["f1"], depth=1)
        fact_links.get_linked_facts.assert_called_once_with(["f1"], depth=1)
