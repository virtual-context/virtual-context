"""An expansion and its eviction plan succeed or fail together."""

import pytest
from types import SimpleNamespace

from virtual_context.core.paging_manager import PagingManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import DepthLevel, StoredSegment, TagSummary, WorkingSetEntry


@pytest.mark.parametrize("new_tokens, expected_success", [(200, False), (80, True)])
def test_page_admission_commits_eviction_only_when_expansion_fits(
    tmp_path, new_tokens, expected_success,
):
    store = SQLiteStore(tmp_path / "paging.db")
    for tag, tokens in [("old", 80), ("new", new_tokens)]:
        store.store_segment(StoredSegment(
            ref=tag, conversation_id="audit", primary_tag=tag, tags=[tag],
            full_tokens=tokens, full_text=f"{tag} content",
        ))
    store.save_tag_summary(TagSummary(tag="old", summary_tokens=10), conversation_id="audit")
    manager = PagingManager(
        store, len, tag_context_max_tokens=100, conversation_id="audit",
    )
    manager.set_memory_renderer(lambda tag, depth, **kwargs: SimpleNamespace(
        measured_cost=(10 if tag == "old" and depth == DepthLevel.SUMMARY
                       else 80 if tag == "old" else new_tokens),
    ))
    old_entry = WorkingSetEntry(tag="old", depth=DepthLevel.FULL, tokens=80)
    manager.working_set["old"] = old_entry
    before = manager.get_working_set_summary()
    result = manager.expand_topic("new", "full")
    if expected_success:
        assert result["evicted_tags"] == ["old"]
        assert result["tokens_evicted"] == 70
        assert manager.working_set["old"].depth == DepthLevel.SUMMARY
        assert manager.working_set["new"].depth == DepthLevel.FULL
        assert manager.get_working_set_summary()["used"] == 90
    else:
        assert result["error"] == "insufficient budget"
        assert result["available"] == before["available"]
        assert manager.get_working_set_summary() == before
        assert manager.working_set["old"] is old_entry
        assert old_entry.depth == DepthLevel.FULL
