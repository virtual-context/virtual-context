"""Tests for vc_recall_all tool and engine.recall_all()."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from virtual_context.core.tool_loop import (
    VC_TOOL_NAMES,
    execute_vc_tool,
    is_vc_tool,
    vc_tool_definitions,
)
from virtual_context.types import TagSummary


class TestVcRecallAllToolDefinition:
    """Verify vc_recall_all appears in tool catalogue."""

    def test_in_vc_tool_names(self):
        assert "vc_recall_all" in VC_TOOL_NAMES

    def test_is_vc_tool(self):
        assert is_vc_tool("vc_recall_all") is True

    def test_in_tool_definitions(self):
        defs = vc_tool_definitions()
        names = [d["name"] for d in defs]
        assert "vc_recall_all" in names

    def test_tool_definition_has_description(self):
        defs = vc_tool_definitions()
        recall_def = next(d for d in defs if d["name"] == "vc_recall_all")
        assert "broad overview" in recall_def["description"].lower()


class TestEngineRecallAll:
    """Test engine.recall_all() method."""

    def _make_engine(self, tmp_path):
        from virtual_context.config import load_config
        from virtual_context.engine import VirtualContextEngine

        config = load_config(config_dict={
            "context_window": 50000,
            "storage_root": str(tmp_path),
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "store.db")}},
            "tag_generator": {"type": "keyword"},
            "assembly": {"tag_context_max_tokens": 500},
        })
        return VirtualContextEngine(config=config)

    def test_recall_all_empty_store(self, tmp_path):
        """Returns found=False when no tag summaries exist."""
        engine = self._make_engine(tmp_path)
        result = engine.recall_all()
        assert result["found"] is False
        assert "message" in result

    def test_recall_all_returns_summaries(self, tmp_path):
        """Returns all tag summaries when they fit in budget."""
        engine = self._make_engine(tmp_path)
        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

        engine._store.save_tag_summary(TagSummary(
            tag="legal", summary="Legal overview.", summary_tokens=20,
            source_segment_refs=["seg-1"], covers_through_turn=5,
            created_at=now, updated_at=now,
        ))
        engine._store.save_tag_summary(TagSummary(
            tag="medical", summary="Medical overview.", summary_tokens=20,
            source_segment_refs=["seg-2"], covers_through_turn=5,
            created_at=now, updated_at=now,
        ))

        result = engine.recall_all()
        assert result["found"] is True
        assert result["topics_loaded"] == 2
        assert result["total_tokens"] == 40
        tags = [s["tag"] for s in result["summaries"]]
        assert "legal" in tags
        assert "medical" in tags

    def test_recall_all_respects_token_budget(self, tmp_path):
        """Stops adding summaries when budget is exceeded."""
        engine = self._make_engine(tmp_path)
        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

        # Budget is 500 tokens. Add summaries that exceed it.
        for i in range(20):
            engine._store.save_tag_summary(TagSummary(
                tag=f"topic-{i}", summary=f"Topic {i} summary.",
                summary_tokens=50,  # 20 * 50 = 1000 > 500 budget
                source_segment_refs=[f"seg-{i}"], covers_through_turn=5,
                created_at=now, updated_at=now,
            ))

        result = engine.recall_all()
        assert result["found"] is True
        assert result["topics_loaded"] == 10  # 10 * 50 = 500
        assert result["total_tokens"] == 500


class TestVcRecallAllExecution:
    """Test execute_vc_tool dispatches to engine.recall_all()."""

    def test_execute_calls_recall_all(self, tmp_path):
        import json
        from unittest.mock import MagicMock

        engine = MagicMock()
        engine.recall_all.return_value = {"found": True, "topics_loaded": 3}

        result_str = execute_vc_tool(engine, "vc_recall_all", {})
        engine.recall_all.assert_called_once()
        result = json.loads(result_str)
        assert result["found"] is True
