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
