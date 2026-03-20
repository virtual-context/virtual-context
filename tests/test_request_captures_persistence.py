"""Tests for request capture persistence through ContextStore."""
import time
from virtual_context.storage.sqlite import SQLiteStore


class TestRequestCapturesPersistence:
    def test_load_returns_empty_by_default(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        assert store.load_request_captures() == []
