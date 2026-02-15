"""Tests for MCP server tools and resources."""
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from virtual_context.types import (
    CompactionReport,
    RetrievalResult,
    StoredSummary,
    TagStats,
)


class TestMCPServerTools:
    """Test MCP tool functions directly by mocking the engine."""

    def _mock_engine(self):
        engine = MagicMock()
        engine.transform.return_value = "<virtual-context>test summary</virtual-context>"
        engine.compact_manual.return_value = CompactionReport(
            segments_compacted=2,
            tokens_freed=500,
            tags=["database", "api"],
        )
        engine._store.get_all_tags.return_value = [
            TagStats(tag="database", usage_count=5, total_full_tokens=1000, total_summary_tokens=200,
                     oldest_segment=datetime(2024, 1, 1, tzinfo=timezone.utc),
                     newest_segment=datetime(2024, 6, 1, tzinfo=timezone.utc)),
            TagStats(tag="api", usage_count=3, total_full_tokens=800, total_summary_tokens=150),
        ]
        engine._store.get_summaries_by_tags.return_value = [
            StoredSummary(ref="seg-1", primary_tag="database", tags=["database"],
                         summary="Test summary", summary_tokens=50,
                         created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        ]
        return engine

    @patch("virtual_context.mcp.server._get_engine")
    def test_recall_context(self, mock_get_engine):
        mock_get_engine.return_value = self._mock_engine()

        from virtual_context.mcp.server import recall_context
        result = recall_context("What about the database?")
        assert "virtual-context" in result

    @patch("virtual_context.mcp.server._get_engine")
    def test_recall_context_with_active_tags(self, mock_get_engine):
        engine = self._mock_engine()
        mock_get_engine.return_value = engine

        from virtual_context.mcp.server import recall_context
        recall_context("database query", active_tags=["database"])
        engine.transform.assert_called_with("database query", active_tags=["database"])

    @patch("virtual_context.mcp.server._get_engine")
    def test_compact_context(self, mock_get_engine):
        mock_get_engine.return_value = self._mock_engine()

        from virtual_context.mcp.server import compact_context
        result = compact_context([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ])
        data = json.loads(result)
        assert data["status"] == "compacted"
        assert data["segments_compacted"] == 2
        assert data["tokens_freed"] == 500

    @patch("virtual_context.mcp.server._get_engine")
    def test_compact_context_no_result(self, mock_get_engine):
        engine = self._mock_engine()
        engine.compact_manual.return_value = None
        mock_get_engine.return_value = engine

        from virtual_context.mcp.server import compact_context
        result = compact_context([{"role": "user", "content": "Hello"}])
        data = json.loads(result)
        assert data["status"] == "no_compaction"

    @patch("virtual_context.mcp.server._get_engine")
    def test_domain_status(self, mock_get_engine):
        mock_get_engine.return_value = self._mock_engine()

        from virtual_context.mcp.server import domain_status
        result = domain_status()
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["tag"] == "database"
        assert data[0]["usage_count"] == 5

    @patch("virtual_context.mcp.server._get_engine")
    def test_list_domains_resource(self, mock_get_engine):
        mock_get_engine.return_value = self._mock_engine()

        from virtual_context.mcp.server import list_domains
        result = list_domains()
        assert "database" in result
        assert "api" in result

    @patch("virtual_context.mcp.server._get_engine")
    def test_get_domain_summaries_resource(self, mock_get_engine):
        mock_get_engine.return_value = self._mock_engine()

        from virtual_context.mcp.server import get_domain_summaries
        result = get_domain_summaries("database")
        assert "Test summary" in result

    @patch("virtual_context.mcp.server._get_engine")
    def test_get_domain_summaries_empty(self, mock_get_engine):
        engine = self._mock_engine()
        engine._store.get_summaries_by_tags.return_value = []
        mock_get_engine.return_value = engine

        from virtual_context.mcp.server import get_domain_summaries
        result = get_domain_summaries("nonexistent")
        assert "No summaries found" in result


class TestMCPPrompts:
    def test_recall_prompt(self):
        from virtual_context.mcp.server import recall
        result = recall("database schema")
        assert "database schema" in result

    def test_summarize_session_prompt(self):
        from virtual_context.mcp.server import summarize_session
        result = summarize_session()
        assert "compact" in result.lower()
