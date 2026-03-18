"""Tests for conversation-scoped retrieval across all store methods."""

import inspect

import pytest

from virtual_context.core.store import ContextStore


class TestABCSignatures:
    """Every retrieval method on ContextStore must accept conversation_id."""

    SCOPED_METHODS = [
        "get_summaries_by_tags",
        "get_segments_by_tags",
        "search",
        "search_full_text",
        "get_segment",
        "get_summary",
        "query_facts",
        "search_facts",
        "search_tool_outputs",
        "get_fact_count_by_tags",
        "get_unique_fact_verbs",
        "get_all_tags",
    ]

    @pytest.mark.parametrize("method_name", SCOPED_METHODS)
    def test_method_accepts_conversation_id(self, method_name):
        method = getattr(ContextStore, method_name)
        sig = inspect.signature(method)
        assert "conversation_id" in sig.parameters, (
            f"{method_name} missing conversation_id parameter"
        )
        param = sig.parameters["conversation_id"]
        assert param.default is None, (
            f"{method_name} conversation_id should default to None"
        )
