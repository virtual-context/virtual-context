"""Tests for store-backed pipeline recovery."""

from virtual_context.types import MonitorConfig


def test_monitor_config_has_store_recovery_threshold():
    mc = MonitorConfig()
    assert hasattr(mc, "store_recovery_threshold")
    assert mc.store_recovery_threshold == 0.70


def _make_dummy_store():
    """Create a minimal concrete ContextStore for testing default method implementations."""
    from virtual_context.core.store import ContextStore

    class DummyStore(ContextStore):
        def store_segment(self, segment): return ""
        def get_segment(self, ref, *, conversation_id=None): return None
        def get_summary(self, ref, *, conversation_id=None): return None
        def get_summaries_by_tags(self, tags, min_overlap=1, limit=10, before=None, after=None, conversation_id=None): return []
        def search(self, query, tags=None, limit=5, conversation_id=None): return []
        def get_all_tags(self, conversation_id=None): return []
        def get_conversation_stats(self): return []
        def get_tag_aliases(self, conversation_id=None): return {}
        def set_tag_alias(self, alias, canonical, conversation_id=""): pass
        def delete_segment(self, ref): return False
        def cleanup(self, max_age=None, max_total_tokens=None): return 0
        def save_tag_summary(self, tag_summary, conversation_id=""): pass
        def get_tag_summary(self, tag, conversation_id=""): return None
        def get_all_tag_summaries(self, *, conversation_id=None): return []
        def search_full_text(self, query, limit=5, conversation_id=None): return []
        def get_segments_by_tags(self, tags, min_overlap=1, limit=20, conversation_id=None): return []

    return DummyStore()


def test_get_chain_snapshots_for_conversation_abstract():
    s = _make_dummy_store()
    assert s.get_chain_snapshots_for_conversation("conv1") == []
    assert s.get_chain_snapshots_for_conversation("conv1", min_turn=100) == []


def test_get_tool_names_for_refs_abstract():
    s = _make_dummy_store()
    assert s.get_tool_names_for_refs(["ref1", "ref2"]) == []
