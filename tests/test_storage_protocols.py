"""Tests for storage protocol definitions."""

from virtual_context.core.protocols import (
    FactLinkStore,
    FactStore,
    SearchStore,
    SegmentStore,
    StateStore,
)


class TestProtocolsExist:
    def test_segment_store_is_protocol(self):
        assert hasattr(SegmentStore, 'store_segment')
        assert hasattr(SegmentStore, 'get_segment')
        assert hasattr(SegmentStore, 'get_summary')
        assert hasattr(SegmentStore, 'get_summaries_by_tags')
        assert hasattr(SegmentStore, 'search')
        assert hasattr(SegmentStore, 'get_all_tags')
        assert hasattr(SegmentStore, 'get_all_segments')
        assert hasattr(SegmentStore, 'save_tag_summary')
        assert hasattr(SegmentStore, 'get_tag_summary')
        assert hasattr(SegmentStore, 'get_all_tag_summaries')
        assert hasattr(SegmentStore, 'delete_segment')
        assert hasattr(SegmentStore, 'cleanup')

    def test_fact_store_is_protocol(self):
        assert hasattr(FactStore, 'store_facts')
        assert hasattr(FactStore, 'query_facts')
        assert hasattr(FactStore, 'get_unique_fact_verbs')
        assert hasattr(FactStore, 'search_facts')
        assert hasattr(FactStore, 'set_fact_superseded')
        assert hasattr(FactStore, 'update_fact_fields')

    def test_fact_link_store_is_protocol(self):
        assert hasattr(FactLinkStore, 'store_fact_links')
        assert hasattr(FactLinkStore, 'get_fact_links')
        assert hasattr(FactLinkStore, 'get_linked_facts')
        assert hasattr(FactLinkStore, 'delete_fact_links')

    def test_state_store_is_protocol(self):
        assert hasattr(StateStore, 'save_engine_state')
        assert hasattr(StateStore, 'load_engine_state')
        assert hasattr(StateStore, 'load_latest_engine_state')

    def test_search_store_is_protocol(self):
        assert hasattr(SearchStore, 'search_full_text')
        assert hasattr(SearchStore, 'store_chunk_embeddings')
        assert hasattr(SearchStore, 'get_all_chunk_embeddings')
