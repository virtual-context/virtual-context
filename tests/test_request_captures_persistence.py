"""Tests for request capture persistence through ContextStore."""
import time
from virtual_context.storage.sqlite import SQLiteStore


class TestRequestCapturesPersistence:
    def test_load_returns_empty_by_default(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        assert store.load_request_captures() == []

    def test_save_and_load_round_trip(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        capture = {
            "turn": 1,
            "ts": "2026-03-20T10:00:00+00:00",
            "api_format": "anthropic",
            "model": "claude-sonnet-4-6",
            "stream": True,
            "message_count": 5,
            "conversation_id": "conv-1",
            "inbound_tags": ["python", "testing"],
            "response_tags": ["code-review"],
            "passthrough": False,
            "inbound_tokens": 1000,
            "outbound_tokens": 500,
            "inbound_bytes": 4096,
            "outbound_bytes": 2048,
            "context_tokens": 800,
            "overhead_ms": 42.5,
            "turns_dropped": 0,
            "turns_stubbed": 1,
            "message_preview": "Write a test for...",
            "upstream_input_tokens": 1200,
            "upstream_output_tokens": 600,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 300,
        }
        store.save_request_capture(capture)
        loaded = store.load_request_captures()
        assert len(loaded) == 1
        assert loaded[0]["turn"] == 1
        assert loaded[0]["model"] == "claude-sonnet-4-6"
        assert loaded[0]["inbound_tags"] == ["python", "testing"]
        assert loaded[0]["overhead_ms"] == 42.5
        assert loaded[0]["message_preview"] == "Write a test for..."

    def test_prune_keeps_newest_50(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        for i in range(60):
            store.save_request_capture({
                "turn": i,
                "ts": f"2026-03-20T10:{i:02d}:00+00:00",
                "api_format": "anthropic",
                "model": "test",
                "stream": False,
                "message_count": 1,
                "conversation_id": "",
                "inbound_tags": [],
                "response_tags": [],
                "passthrough": False,
                "inbound_tokens": 0,
                "outbound_tokens": 0,
                "inbound_bytes": 0,
                "outbound_bytes": 0,
                "context_tokens": 0,
                "overhead_ms": 0,
                "turns_dropped": 0,
                "turns_stubbed": 0,
                "message_preview": "",
                "upstream_input_tokens": 0,
                "upstream_output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            })
        loaded = store.load_request_captures(limit=50)
        assert len(loaded) == 50
        # Oldest (turn 0-9) should be pruned
        turns = [c["turn"] for c in loaded]
        assert min(turns) == 10
        assert max(turns) == 59

    def test_upsert_updates_existing_turn(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        cap = {
            "turn": 5, "ts": "2026-03-20T10:00:00+00:00",
            "api_format": "anthropic", "model": "test", "stream": False,
            "message_count": 1, "conversation_id": "",
            "inbound_tags": [], "response_tags": [],
            "passthrough": False,
            "inbound_tokens": 100, "outbound_tokens": 0,
            "inbound_bytes": 0, "outbound_bytes": 0,
            "context_tokens": 0, "overhead_ms": 0,
            "turns_dropped": 0, "turns_stubbed": 0,
            "message_preview": "",
            "upstream_input_tokens": 0, "upstream_output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
        store.save_request_capture(cap)
        # Update with response tokens
        cap["upstream_output_tokens"] = 500
        cap["response_tags"] = ["updated"]
        store.save_request_capture(cap)
        loaded = store.load_request_captures()
        assert len(loaded) == 1
        assert loaded[0]["upstream_output_tokens"] == 500
        assert loaded[0]["response_tags"] == ["updated"]

    def test_load_survives_new_store_instance(self, tmp_path):
        db = tmp_path / "test.db"
        store1 = SQLiteStore(db)
        store1.save_request_capture({
            "turn": 1, "ts": "2026-03-20T10:00:00+00:00",
            "api_format": "openai", "model": "gpt-4", "stream": False,
            "message_count": 3, "conversation_id": "c1",
            "inbound_tags": ["x"], "response_tags": [],
            "passthrough": False,
            "inbound_tokens": 0, "outbound_tokens": 0,
            "inbound_bytes": 0, "outbound_bytes": 0,
            "context_tokens": 0, "overhead_ms": 0,
            "turns_dropped": 0, "turns_stubbed": 0,
            "message_preview": "hello",
            "upstream_input_tokens": 0, "upstream_output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        })
        store1.close()
        # New instance = simulates restart
        store2 = SQLiteStore(db)
        loaded = store2.load_request_captures()
        assert len(loaded) == 1
        assert loaded[0]["conversation_id"] == "c1"
        store2.close()


class TestFilesystemRequestCaptures:
    def test_save_and_load(self, tmp_path):
        from virtual_context.storage.filesystem import FilesystemStore
        store = FilesystemStore(tmp_path / "fs_store")
        cap = {
            "turn": 1, "ts": "2026-03-20T10:00:00+00:00",
            "api_format": "anthropic", "model": "test", "stream": False,
            "message_count": 1, "conversation_id": "",
            "inbound_tags": [], "response_tags": [],
            "passthrough": False,
            "inbound_tokens": 0, "outbound_tokens": 0,
            "inbound_bytes": 0, "outbound_bytes": 0,
            "context_tokens": 0, "overhead_ms": 0,
            "turns_dropped": 0, "turns_stubbed": 0,
            "message_preview": "test",
            "upstream_input_tokens": 0, "upstream_output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
        store.save_request_capture(cap)
        loaded = store.load_request_captures()
        assert len(loaded) == 1
        assert loaded[0]["turn"] == 1

    def test_prune_to_50(self, tmp_path):
        from virtual_context.storage.filesystem import FilesystemStore
        store = FilesystemStore(tmp_path / "fs_store")
        for i in range(55):
            store.save_request_capture({
                "turn": i, "ts": f"2026-03-20T10:{i:02d}:00+00:00",
                "api_format": "anthropic", "model": "test", "stream": False,
                "message_count": 1, "conversation_id": "",
                "inbound_tags": [], "response_tags": [],
                "passthrough": False,
                "inbound_tokens": 0, "outbound_tokens": 0,
                "inbound_bytes": 0, "outbound_bytes": 0,
                "context_tokens": 0, "overhead_ms": 0,
                "turns_dropped": 0, "turns_stubbed": 0,
                "message_preview": "",
                "upstream_input_tokens": 0, "upstream_output_tokens": 0,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            })
        loaded = store.load_request_captures()
        assert len(loaded) == 50
        assert loaded[0]["turn"] == 5  # oldest 5 pruned


class TestProxyMetricsStoreIntegration:
    def _make_store(self, tmp_path):
        return SQLiteStore(tmp_path / "test.db")

    def test_capture_persists_to_store(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics
        store = self._make_store(tmp_path)
        m = ProxyMetrics(store=store)
        m.capture_request(
            turn=1, body={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            api_format="anthropic", inbound_tags=["python"],
            conversation_id="c1", message_preview="hi",
        )
        loaded = store.load_request_captures()
        assert len(loaded) == 1
        assert loaded[0]["turn"] == 1
        assert loaded[0]["inbound_tags"] == ["python"]

    def test_capture_enriched_persists_to_store(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics
        from unittest.mock import patch
        store = self._make_store(tmp_path)
        m = ProxyMetrics(store=store)
        m.capture_request(
            turn=1, body={"model": "test", "messages": []},
            api_format="anthropic",
        )
        # Verify _persist_capture is called by capture_enriched
        with patch.object(m, '_persist_capture', wraps=m._persist_capture) as mock_persist:
            m.capture_enriched(turn=1, body={"model": "test", "messages": [], "system": "ctx"})
            mock_persist.assert_called_once_with(1)
        # The store should still have the capture (enriched body excluded from summary)
        loaded = store.load_request_captures()
        assert len(loaded) == 1
        assert loaded[0]["turn"] == 1
        assert "enriched" not in loaded[0]

    def test_capture_response_updates_store(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics
        store = self._make_store(tmp_path)
        m = ProxyMetrics(store=store)
        m.capture_request(
            turn=1, body={"model": "test", "messages": []},
            api_format="anthropic",
        )
        m.capture_response(
            turn=1, body={"content": "hello"},
            upstream_input_tokens=100, upstream_output_tokens=50,
        )
        loaded = store.load_request_captures()
        assert loaded[0]["upstream_input_tokens"] == 100
        assert loaded[0]["upstream_output_tokens"] == 50

    def test_update_tags_persists(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics
        store = self._make_store(tmp_path)
        m = ProxyMetrics(store=store)
        m.capture_request(
            turn=1, body={"model": "test", "messages": []},
            api_format="anthropic",
        )
        m.update_request_tags(turn=1, response_tags=["code-review"])
        loaded = store.load_request_captures()
        assert loaded[0]["response_tags"] == ["code-review"]

    def test_restore_from_store_on_init(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics
        store = self._make_store(tmp_path)
        m1 = ProxyMetrics(store=store)
        m1.capture_request(
            turn=1, body={"model": "test", "messages": [{"role": "user", "content": "x"}]},
            api_format="anthropic", conversation_id="c1",
        )
        # Simulate restart: new ProxyMetrics, same store
        m2 = ProxyMetrics(store=store)
        summary = m2.get_captured_requests_summary()
        assert len(summary) == 1
        assert summary[0]["turn"] == 1
        assert summary[0]["conversation_id"] == "c1"

    def test_no_store_still_works(self, tmp_path):
        """Existing behavior: no store param = in-memory only."""
        from virtual_context.proxy.metrics import ProxyMetrics
        m = ProxyMetrics()
        m.capture_request(
            turn=1, body={"model": "test", "messages": []},
            api_format="anthropic",
        )
        assert len(m.get_captured_requests_summary()) == 1

    def test_restore_deduplicates_by_turn(self, tmp_path):
        """Dual restore paths (store + EngineState) must not create duplicates."""
        from virtual_context.proxy.metrics import ProxyMetrics
        store = self._make_store(tmp_path)
        m1 = ProxyMetrics(store=store)
        m1.capture_request(
            turn=1, body={"model": "test", "messages": []},
            api_format="anthropic", conversation_id="c1",
        )
        # Simulate restart: store loads in __init__
        m2 = ProxyMetrics(store=store)
        assert len(m2.get_captured_requests_summary()) == 1
        # Then EngineState restore path also fires (via ProxyState)
        m2.restore_request_captures([{"turn": 1, "ts": "2026-03-20T10:00:00+00:00",
            "api_format": "anthropic", "model": "test", "stream": False,
            "message_count": 0, "conversation_id": "c1",
            "inbound_tags": [], "response_tags": [], "passthrough": False,
            "inbound_tokens": 0, "outbound_tokens": 0,
            "inbound_bytes": 0, "outbound_bytes": 0,
            "context_tokens": 0, "overhead_ms": 0,
            "turns_dropped": 0, "turns_stubbed": 0,
            "message_preview": "",
            "upstream_input_tokens": 0, "upstream_output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }])
        # Should still be 1, not 2
        assert len(m2.get_captured_requests_summary()) == 1
