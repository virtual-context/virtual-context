# tests/test_vcattach.py
"""Tests for VCATTACH command."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sqlite_store(tmp_path):
    from virtual_context.storage.sqlite import SQLiteStore
    store = SQLiteStore(str(tmp_path / "test.db"))
    return store


def test_save_and_resolve_alias(sqlite_store):
    sqlite_store.save_conversation_alias("old-conv-123", "target-conv-456")
    resolved = sqlite_store.resolve_conversation_alias("old-conv-123")
    assert resolved == "target-conv-456"


def test_resolve_unknown_alias(sqlite_store):
    resolved = sqlite_store.resolve_conversation_alias("nonexistent")
    assert resolved is None


def test_alias_overwrite(sqlite_store):
    sqlite_store.save_conversation_alias("old-123", "target-a")
    sqlite_store.save_conversation_alias("old-123", "target-b")
    assert sqlite_store.resolve_conversation_alias("old-123") == "target-b"


def test_alias_chain_not_followed(sqlite_store):
    """Aliases are single-hop — no chain following."""
    sqlite_store.save_conversation_alias("a", "b")
    sqlite_store.save_conversation_alias("b", "c")
    assert sqlite_store.resolve_conversation_alias("a") == "b"


# --- VCATTACH regex tests ---

import re

_VCATTACH_RE = re.compile(r"^VCATTACH\s+(.+)$", re.IGNORECASE)


def test_vcattach_regex_label():
    m = _VCATTACH_RE.match("VCATTACH website")
    assert m and m.group(1) == "website"


def test_vcattach_regex_uuid():
    m = _VCATTACH_RE.match("VCATTACH d4f83259-4ffc-fa3f-5914-a266d0a4577c")
    assert m and m.group(1) == "d4f83259-4ffc-fa3f-5914-a266d0a4577c"


def test_vcattach_regex_prefix():
    m = _VCATTACH_RE.match("VCATTACH d4f83259")
    assert m and m.group(1) == "d4f83259"


def test_vcattach_regex_case_insensitive():
    m = _VCATTACH_RE.match("vcattach Website")
    assert m and m.group(1) == "Website"


def test_vcattach_regex_no_target():
    m = _VCATTACH_RE.match("VCATTACH")
    assert m is None


def test_vcattach_regex_not_triggered_by_history():
    """Only the last user message should trigger, not history."""
    m = _VCATTACH_RE.match("I said VCATTACH website earlier")
    assert m is None


def test_emit_fake_response_sse_anthropic():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    events = fmt.emit_fake_response_sse("Test response", "conv-123")
    assert isinstance(events, bytes)
    text = events.decode()
    assert "Test response" in text
    assert "vc:conversation=conv-123" in text
    assert "message_start" in text
    assert "message_stop" in text


def test_emit_fake_response_sse_openai():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    events = fmt.emit_fake_response_sse("Test response", "conv-123")
    text = events.decode()
    assert "Test response" in text
    assert "vc:conversation=conv-123" in text
    assert "[DONE]" in text


def test_build_fake_response_anthropic():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    resp = fmt.build_fake_response("Test response", "conv-123")
    assert isinstance(resp, dict)
    assert resp["role"] == "assistant"
    assert "Test response" in resp["content"][0]["text"]
    assert "vc:conversation=conv-123" in resp["content"][0]["text"]


def test_build_fake_response_openai():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    resp = fmt.build_fake_response("Test response", "conv-123")
    assert isinstance(resp, dict)
    assert "choices" in resp
    assert "Test response" in resp["choices"][0]["message"]["content"]
    assert "vc:conversation=conv-123" in resp["choices"][0]["message"]["content"]


def test_vcstatus_uses_effective_engine_conversation_id():
    from virtual_context.proxy.handlers import _handle_vcstatus

    class _TurnTagIndex:
        entries = [1, 2, 3]

        def get_active_tags(self, lookback=6):
            return {"alpha", "beta"}

    class _Store:
        def get_conversation_stats(self):
            return [
                SimpleNamespace(conversation_id="alias-conv", segment_count=1),
                SimpleNamespace(conversation_id="target-conv", segment_count=7),
            ]

    state = SimpleNamespace(
        engine=SimpleNamespace(
            config=SimpleNamespace(conversation_id="target-conv"),
            _engine_state=SimpleNamespace(compacted_prefix_messages=4, conversation_generation=9),
            _turn_tag_index=_TurnTagIndex(),
            _store=_Store(),
            _paging=SimpleNamespace(working_set={}),
        )
    )

    text = _handle_vcstatus("alias-conv", state, tenant_registry=None, tenant_id=None)

    assert "Conversation: target-conv" in text
    assert "Stored: 7 segments, 0 tag summaries" in text


def test_vcstatus_without_state_reports_shell_status():
    from virtual_context.proxy.handlers import _handle_vcstatus

    text = _handle_vcstatus(
        "56315149-9812-9cf5-21a7-ade5a2279ad8",
        None,
        tenant_registry=None,
        tenant_id=None,
    )

    assert "Conversation: 56315149-9812-9cf5-21a7-ade5a2279ad8" in text
    assert "Status: ready" in text
    assert "Ingestion: 0 / 0 (0.0%)" in text
    assert "Turn state: 0 turns, 0 live history messages" in text
    assert "Stored: 0 segments, 0 tag summaries" in text


def test_vcstatus_surfaces_ingestion_payload_and_cache_metrics():
    from virtual_context.proxy.handlers import _handle_vcstatus

    class _TurnTagIndex:
        entries = [1, 2, 3]

        def get_active_tags(self, lookback=6):
            return {"alpha", "beta"}

    class _Store:
        def get_conversation_stats(self):
            return [
                SimpleNamespace(conversation_id="target-conv", segment_count=7),
            ]

        def get_all_segments(self, *, conversation_id=None, limit=None):
            return []

        def get_all_tag_summaries(self, *, conversation_id=None):
            return [SimpleNamespace(created_at=datetime.now(timezone.utc), covers_through_turn=12)]

        def load_request_captures(self, limit=50, conversation_id=None):
            return [
                {
                    "turn": 12,
                    "turn_id": "t12",
                    "ts": "2026-04-14T21:15:16.642392+00:00",
                    "client_payload_message_count": 989,
                    "client_payload_user_prompt_count": 300,
                    "client_payload_timestamped_message_count": 290,
                    "client_payload_earliest_timestamp": "2026-03-15T16:10:00+00:00",
                    "client_payload_latest_timestamp": "2026-04-14T21:15:00+00:00",
                    "raw_payload_entry_count": 4632,
                    "ingestible_entry_count": 989,
                    "upstream_input_tokens": 100000,
                    "cache_read_input_tokens": 82000,
                }
            ]

        def read_progress_snapshot(self, conversation_id):
            return SimpleNamespace(
                conversation_id=conversation_id,
                phase="ingesting",
                done_ingestible=989,
                total_ingestible=989,
                active_episode=None,
                active_compaction=None,
            )

    class _Metrics:
        def get_captured_requests_summary(self, conversation_id=None):
            return [
                {
                    "raw_payload_entry_count": 4632,
                    "ingestible_entry_count": 989,
                    "upstream_input_tokens": 100000,
                    "cache_read_input_tokens": 82000,
                }
            ]

    state = SimpleNamespace(
        session_state=SimpleNamespace(value="ingesting"),
        _ingestion_progress=(99, 494),
        _ingestible_entry_count=989,
        _raw_payload_entry_count=4632,
        _last_payload_kb=22550.3,
        _last_payload_tokens=12331759,
        conversation_history=[SimpleNamespace(content="x")] * 989,
        metrics=_Metrics(),
        compaction_snapshot=lambda: {},
        engine=SimpleNamespace(
            config=SimpleNamespace(
                conversation_id="target-conv",
                monitor=SimpleNamespace(soft_threshold=80000, hard_threshold=120000),
            ),
            _engine_state=SimpleNamespace(
                compacted_prefix_messages=986,
                flushed_prefix_messages=983,
                conversation_generation=15,
            ),
            _turn_tag_index=_TurnTagIndex(),
            _store=_Store(),
            _paging=SimpleNamespace(working_set={}),
        ),
    )

    text = _handle_vcstatus("target-conv", state, tenant_registry=None, tenant_id=None)

    assert "Status: ingesting" in text
    assert "Ingestion: 989 / 989 (100.0%)" in text
    assert "Turn state: 989 canonical chat entries, 4632 raw payload entries" in text
    assert "Stored: 7 segments, 1 tag summaries" in text
    assert "Thresholds: soft 80,000 / hard 120,000" in text
    assert "Last payload: 22.022 MB, 989 ingestible entries, 12,331,759 tokens" in text
    assert "Last raw payload: 4,632 entries" in text
    assert "Cache hit (last 5): 82% (avg 82.0%)" in text


def test_vcattach_preserves_target_engine_state():
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    class _InnerStore:
        def __init__(self):
            self.aliases = []
            self.load_called = False
            self.save_called = False
            self.deleted = []

        def save_conversation_alias(self, alias_id, target_id):
            self.aliases.append((alias_id, target_id))

        def load_engine_state(self, conversation_id):
            self.load_called = True
            return {"conversation_id": conversation_id}

        def save_engine_state(self, snapshot):
            self.save_called = True

        def delete_conversation(self, conversation_id):
            self.deleted.append(conversation_id)

    inner = _InnerStore()
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))
    registry = SimpleNamespace(remove_conversation=lambda cid: None)
    result = SimpleNamespace(
        vcattach_label="Cloud Claude",
        conversation_id="old-shell-conv",
        is_streaming=False,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH Cloud Claude"}],
    })

    response = asyncio.run(
        _handle_vcattach(
            result,
            fmt,
            state,
            registry,
            labels={"target-conv": "Cloud Claude"},
            conv_ids=["old-shell-conv", "target-conv"],
        )
    )

    assert response.status_code == 200
    assert inner.aliases == [("old-shell-conv", "target-conv")]
    assert inner.load_called is False
    assert inner.save_called is False


def test_vcattach_does_not_delete_old_conversation_in_cloud_mode():
    """VCATTACH is a durable redirect — the old conversation is preserved."""
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    inner = MagicMock()
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))
    registry = MagicMock()
    tenant_registry = MagicMock()
    result = SimpleNamespace(
        vcattach_label="d4f83259-4ffc-fa3f-5914-a266d0a4577c",
        conversation_id="56315149-9812-9cf5-21a7-ade5a2279ad8",
        is_streaming=False,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH d4f83259-4ffc-fa3f-5914-a266d0a4577c"}],
    })

    response = asyncio.run(
        _handle_vcattach(
            result,
            fmt,
            state,
            registry,
            conv_ids=["d4f83259-4ffc-fa3f-5914-a266d0a4577c"],
            tenant_registry=tenant_registry,
            tenant_id="tenant-123",
        )
    )

    assert response.status_code == 200
    # Old conversation must NOT be deleted via tenant registry
    tenant_registry.delete_conversation.assert_not_called()
    # Alias must be written old -> target
    inner.save_conversation_alias.assert_called_once_with(
        "56315149-9812-9cf5-21a7-ade5a2279ad8",
        "d4f83259-4ffc-fa3f-5914-a266d0a4577c",
    )
    # Any dangling alias FROM target must be cleared (self-attach unlock)
    inner.delete_conversation_alias.assert_called_once_with(
        "d4f83259-4ffc-fa3f-5914-a266d0a4577c",
    )


def test_execute_attach_clears_reverse_alias():
    """If alias A -> B exists and user VCATTACHes to A, the A -> B alias is cleared."""
    from virtual_context.proxy.vcattach import execute_attach

    store = MagicMock()
    execute_attach(
        old_id="current-shell-id",
        target_id="conversation-A",
        store=store,
    )

    store.delete_conversation_alias.assert_called_once_with("conversation-A")
    store.save_conversation_alias.assert_called_once_with(
        "current-shell-id",
        "conversation-A",
    )


def test_execute_attach_no_longer_accepts_delete_conversation():
    """Old delete_conversation parameter has been removed."""
    import inspect
    from virtual_context.proxy.vcattach import execute_attach

    sig = inspect.signature(execute_attach)
    assert "delete_conversation" not in sig.parameters


def test_delete_conversation_alias_sqlite_roundtrip(sqlite_store):
    """delete_conversation_alias removes the row so resolve returns None."""
    sqlite_store.save_conversation_alias("a", "b")
    assert sqlite_store.resolve_conversation_alias("a") == "b"
    sqlite_store.delete_conversation_alias("a")
    assert sqlite_store.resolve_conversation_alias("a") is None


def test_delete_conversation_alias_missing_is_noop(sqlite_store):
    """Deleting an alias that doesn't exist must not error."""
    sqlite_store.delete_conversation_alias("never-existed")
    assert sqlite_store.resolve_conversation_alias("never-existed") is None
