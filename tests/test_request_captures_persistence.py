"""Tests for request capture persistence through ContextStore."""
from datetime import datetime, timezone

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
            "prepare_total_ms": 84.2,
            "prepare_breakdown": {
                "filter_body_messages": 51.0,
                "collapse_turn_chains": 19.7,
            },
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
        assert loaded[0]["prepare_total_ms"] == 84.2
        assert loaded[0]["prepare_breakdown"] == {
            "filter_body_messages": 51.0,
            "collapse_turn_chains": 19.7,
        }
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

    def test_same_turns_are_isolated_by_conversation(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        for conversation_id, model in (("c1", "model-a"), ("c2", "model-b")):
            store.save_request_capture({
                "turn": 7,
                "ts": "2026-03-20T10:00:00+00:00",
                "api_format": "anthropic",
                "model": model,
                "stream": False,
                "message_count": 1,
                "conversation_id": conversation_id,
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

        all_caps = store.load_request_captures(limit=10)
        assert len(all_caps) == 2
        assert [c["model"] for c in store.load_request_captures(conversation_id="c1")] == ["model-a"]
        assert [c["model"] for c in store.load_request_captures(conversation_id="c2")] == ["model-b"]

    def test_same_turns_are_isolated_by_turn_id(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        for turn_id, model in (("req-a", "model-a"), ("req-b", "model-b")):
            store.save_request_capture({
                "turn": 7,
                "turn_id": turn_id,
                "ts": "2026-03-20T10:00:00+00:00",
                "api_format": "anthropic",
                "model": model,
                "stream": False,
                "message_count": 1,
                "conversation_id": "c1",
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

        loaded = store.load_request_captures(conversation_id="c1")
        assert len(loaded) == 2
        assert sorted(c["turn_id"] for c in loaded) == ["req-a", "req-b"]

    def test_prune_turn_messages_keeps_suffix(self, tmp_path):
        store = SQLiteStore(tmp_path / "test.db")
        for turn in range(6):
            store.save_turn_message("conv-1", turn, f"user-{turn}", f"assistant-{turn}")

        removed = store.prune_turn_messages("conv-1", keep_from_turn=4)

        assert removed == 4
        assert store.load_recent_turn_messages("conv-1", limit=10) == [
            (4, "user-4", "assistant-4"),
            (5, "user-5", "assistant-5"),
        ]


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

    def test_capture_response_uses_turn_id_when_turn_collides(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics

        store = self._make_store(tmp_path)
        m = ProxyMetrics(store=store)
        m.capture_request(
            turn=1,
            body={"model": "test-a", "messages": []},
            api_format="anthropic",
            conversation_id="c1",
            turn_id="req-a",
        )
        m.capture_request(
            turn=1,
            body={"model": "test-b", "messages": []},
            api_format="anthropic",
            conversation_id="c1",
            turn_id="req-b",
        )
        m.capture_response(
            turn=1,
            body={"content": "hello-a"},
            upstream_input_tokens=100,
            conversation_id="c1",
            turn_id="req-a",
        )
        m.capture_response(
            turn=1,
            body={"content": "hello-b"},
            upstream_input_tokens=200,
            conversation_id="c1",
            turn_id="req-b",
        )

        loaded = sorted(
            store.load_request_captures(conversation_id="c1"),
            key=lambda capture: capture["turn_id"],
        )
        assert [capture["upstream_input_tokens"] for capture in loaded] == [100, 200]

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

    def test_same_turns_do_not_collide_across_conversations(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics

        store = self._make_store(tmp_path)
        m = ProxyMetrics(store=store)
        m.capture_request(
            turn=3,
            body={"model": "a", "messages": []},
            api_format="anthropic",
            conversation_id="conv-a",
        )
        m.capture_request(
            turn=3,
            body={"model": "b", "messages": []},
            api_format="anthropic",
            conversation_id="conv-b",
        )
        m.capture_response(
            turn=3,
            body={"content": "resp-b"},
            conversation_id="conv-b",
            upstream_output_tokens=9,
        )

        cap_a = m.get_captured_request(3, conversation_id="conv-a")
        cap_b = m.get_captured_request(3, conversation_id="conv-b")

        assert cap_a is not None
        assert cap_b is not None
        assert cap_a["model"] == "a"
        assert cap_b["model"] == "b"
        assert cap_b["upstream_output_tokens"] == 9

    def test_delete_conversation_artifacts_purges_scoped_metrics(self, tmp_path):
        from virtual_context.proxy.metrics import ProxyMetrics

        db_path = tmp_path / "metrics.db"
        m = ProxyMetrics(db_path=str(db_path))
        m.capture_request(
            turn=1,
            body={"model": "a", "messages": []},
            api_format="anthropic",
            conversation_id="conv-a",
        )
        m.capture_request(
            turn=1,
            body={"model": "b", "messages": []},
            api_format="anthropic",
            conversation_id="conv-b",
        )
        m.record({"type": "request", "conversation_id": "conv-a"})
        m.record({"type": "request", "conversation_id": "conv-b"})

        removed = m.delete_conversation_artifacts("conv-a")

        assert removed["captures_removed"] == 1
        assert removed["events_removed"] == 1
        assert m.get_captured_requests_summary(conversation_id="conv-a") == []
        assert len(m.get_captured_requests_summary(conversation_id="conv-b")) == 1
        assert all(
            (event.get("conversation_id", "") or "") != "conv-a"
            for event in m.events_since(-1)
        )

        restored = ProxyMetrics(db_path=str(db_path))
        assert all(
            (event.get("conversation_id", "") or "") != "conv-a"
            for event in restored.events_since(-1)
        )
        assert any(
            (event.get("conversation_id", "") or "") == "conv-b"
            for event in restored.events_since(-1)
        )


class TestDeleteConversationCleanup:
    def test_sqlite_delete_conversation_removes_scoped_artifacts(self, tmp_path):
        from virtual_context.types import SegmentMetadata, StoredSegment

        store = SQLiteStore(tmp_path / "test.db")
        now = datetime.now(timezone.utc)
        base_segment = dict(
            summary_tokens=5,
            full_tokens=10,
            full_text="full text",
            metadata=SegmentMetadata(),
            created_at=now,
            start_timestamp=now,
            end_timestamp=now,
        )

        for conversation_id, tag_embedding, tool_term in (
            ("conv-a", [0.1, 0.2], "artifact-alpha"),
            ("conv-b", [0.3, 0.4], "artifact-beta"),
        ):
            store.store_segment(StoredSegment(
                ref=f"{conversation_id}-seg",
                conversation_id=conversation_id,
                primary_tag="topic",
                tags=["topic"],
                summary=f"{conversation_id} summary",
                **base_segment,
            ))
            store.save_turn_message(conversation_id, 0, "user", "assistant")
            store.save_request_capture({
                "turn": 1,
                "ts": "2026-03-20T10:00:00+00:00",
                "api_format": "anthropic",
                "model": conversation_id,
                "stream": False,
                "message_count": 1,
                "conversation_id": conversation_id,
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
            store.store_tool_output(
                ref=f"{conversation_id}-tool",
                conversation_id=conversation_id,
                tool_name="search",
                command="lookup",
                turn=1,
                content=tool_term,
                original_bytes=10,
            )
            store.save_tool_call({
                "conversation_id": conversation_id,
                "request_turn": 1,
                "round": 1,
                "group_id": f"group-{conversation_id}",
                "tool_name": "search",
                "tool_input": {"query": tool_term},
                "tool_result": tool_term,
                "result_length": len(tool_term),
                "duration_ms": 1.0,
                "found": True,
                "timestamp": "2026-03-20T10:00:00+00:00",
            })
            store.save_request_context({
                "conversation_id": conversation_id,
                "request_turn": 1,
                "timestamp": "2026-03-20T10:00:00+00:00",
                "user_message": "hello",
                "inbound_tags": ["topic"],
                "retrieval_method": "default",
                "candidates_found": 1,
                "candidates_selected": 1,
                "segments_injected": [f"{conversation_id}-seg"],
                "facts_injected": [],
                "facts_count": 0,
                "facts_tags": [],
                "pool_used": 1,
                "pool_budget": 10,
                "total_context_tokens": 5,
                "non_virtualizable_floor": 0,
                "tool_call_count": 1,
            })
            store.store_tag_summary_embedding("topic", conversation_id, tag_embedding)

        deleted = store.delete_conversation("conv-a")

        assert deleted == 1
        assert store.get_segment("conv-a-seg", conversation_id="conv-a") is None
        assert store.get_segment("conv-b-seg", conversation_id="conv-b") is not None
        assert store.load_recent_turn_messages("conv-a", limit=5) == []
        assert store.load_recent_turn_messages("conv-b", limit=5) == [
            (0, "user", "assistant"),
        ]
        assert store.load_request_captures(conversation_id="conv-a") == []
        assert len(store.load_request_captures(conversation_id="conv-b")) == 1
        assert store.search_tool_outputs("artifact-alpha", conversation_id="conv-a") == []
        assert len(store.search_tool_outputs("artifact-beta", conversation_id="conv-b")) == 1
        assert store.load_tool_calls("conv-a") == []
        assert len(store.load_tool_calls("conv-b")) == 1
        assert store.load_request_contexts("conv-a") == []
        assert len(store.load_request_contexts("conv-b")) == 1
        assert store.load_tag_summary_embeddings("conv-a") == {}
        assert store.load_tag_summary_embeddings("conv-b") == {"topic": [0.3, 0.4]}
