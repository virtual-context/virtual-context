# tests/test_session_state.py
"""Tests for Redis-backed SessionStateProvider."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import pytest
from unittest.mock import MagicMock, patch
from virtual_context.proxy.session_state import SessionState, SessionStateProvider
from virtual_context.proxy.formats import PayloadTokenCache
from virtual_context.types import TagStats


@pytest.fixture
def mock_redis():
    """Mock Redis client with pipeline support for WATCH/MULTI saves."""
    r = MagicMock()
    r.get.return_value = None
    r.ping.return_value = True

    # Storage for pipeline writes
    _store = {}

    # Pipeline mock — supports WATCH/MULTI/EXEC pattern
    pipe = MagicMock()
    pipe.get.side_effect = lambda k: _store.get(k)
    pipe.set.side_effect = lambda k, v, **kw: _store.update({k: v})
    pipe.execute.return_value = [True]

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=pipe)
    ctx.__exit__ = MagicMock(return_value=False)
    r.pipeline.return_value = ctx

    # Direct get for load() path
    r.get.side_effect = lambda k: _store.get(k)
    r.mget.side_effect = lambda keys: [_store.get(k) for k in keys]

    # Direct set for delete() tombstone path
    r.set.side_effect = lambda k, v, **kw: _store.update({k: v})
    r.delete.side_effect = lambda k: _store.pop(k, None)

    r._test_store = _store  # expose for assertions
    return r


@pytest.fixture
def mock_store():
    """Mock Postgres store for backup/fallback."""
    s = MagicMock()
    s.delete_conversation.return_value = 0
    s.load_engine_state.return_value = None
    s.save_engine_state.return_value = None
    return s


@pytest.fixture
def provider(mock_redis, mock_store):
    return SessionStateProvider(redis_client=mock_redis, store=mock_store)


def test_load_returns_none_for_missing_key(provider, mock_redis):
    result = provider.load("conv-123")
    assert result is None


def test_save_and_load_roundtrip(provider, mock_redis):
    state = SessionState()
    state.compacted_prefix_messages = 42
    state.last_indexed_turn = 10
    state.session_state = "ingesting"
    state.live_turn_count = 500
    state.history_message_count = 1000
    state.ingestion_done = 20
    state.ingestion_total = 499
    state.last_payload_kb = 23062.4
    state.last_payload_tokens = 12541785

    provider.save("conv-123", state)

    loaded = provider.load("conv-123")
    assert loaded is not None
    assert loaded.compacted_prefix_messages == 42
    assert loaded.last_indexed_turn == 10
    assert loaded.session_state == "ingesting"
    assert loaded.live_turn_count == 500
    assert loaded.history_message_count == 1000
    assert loaded.ingestion_done == 20
    assert loaded.ingestion_total == 499
    assert loaded.last_payload_kb == 23062.4
    assert loaded.last_payload_tokens == 12541785
    assert loaded.version == 1  # incremented on save


def test_save_increments_version(provider, mock_redis):
    state = SessionState()
    state.version = 5

    saved_version = provider.save("conv-123", state)

    raw = mock_redis._test_store.get("vc:session:conv-123")
    blob = json.loads(raw)
    assert saved_version == 6
    assert blob["version"] == 6


def test_save_rejects_newer_redis_version_without_mutating_local_version(provider, mock_redis):
    current = SessionState(version=2)
    assert provider.save("conv-123", current) == 3

    stale = SessionState(version=2)
    assert provider.save("conv-123", stale) is None
    assert stale.version == 2

    raw = mock_redis._test_store.get("vc:session:conv-123")
    blob = json.loads(raw)
    assert blob["version"] == 3


def test_delete_sets_tombstone(provider, mock_redis):
    provider.delete("conv-123")

    raw = mock_redis._test_store.get("vc:session:conv-123")
    blob = json.loads(raw)
    assert blob["deleted"] is True
    # Verify TTL was set (24h = 86400s)
    mock_redis.set.assert_called_once()
    call_kwargs = mock_redis.set.call_args
    assert call_kwargs[1].get("ex") == 86400


def test_load_returns_tombstoned_state(provider, mock_redis):
    mock_redis._test_store["vc:session:conv-123"] = json.dumps(
        {"deleted": True, "version": 999999}).encode()
    result = provider.load("conv-123")
    assert result is not None
    assert result.deleted is True


def test_exists_false_for_missing(provider):
    assert provider.exists("conv-123") is False


def test_exists_false_for_tombstoned(provider, mock_redis):
    mock_redis._test_store["vc:session:conv-123"] = json.dumps(
        {"deleted": True}).encode()
    assert provider.exists("conv-123") is False


def test_exists_true_for_live(provider, mock_redis):
    mock_redis._test_store["vc:session:conv-123"] = json.dumps(
        {"deleted": False, "version": 1}).encode()
    assert provider.exists("conv-123") is True


def test_save_rejected_after_tombstone(provider, mock_redis):
    """A save after delete should be rejected — tombstone wins."""
    provider.delete("conv-123")
    state = SessionState(version=1)
    provider.save("conv-123", state)
    # Tombstone should still be there
    raw = mock_redis._test_store.get("vc:session:conv-123")
    blob = json.loads(raw)
    assert blob["deleted"] is True


def test_undelete_allows_reuse_of_same_conversation_id(provider, mock_redis):
    provider.delete("conv-123")

    provider.undelete("conv-123")

    state = SessionState()
    state.last_indexed_turn = 7
    provider.save("conv-123", state)

    loaded = provider.load("conv-123")
    assert loaded is not None
    assert loaded.deleted is False
    assert loaded.last_indexed_turn == 7


def test_turn_tag_entries_roundtrip(provider, mock_redis):
    state = SessionState()
    state.turn_tag_entries = [
        {"turn_number": 0, "tags": ["auth", "debug"], "primary_tag": "auth",
         "message_hash": "abc123", "sender": "user1"},
        {"turn_number": 1, "tags": ["database"], "primary_tag": "database",
         "message_hash": "def456", "sender": ""},
    ]

    provider.save("conv-123", state)

    loaded = provider.load("conv-123")
    assert len(loaded.turn_tag_entries) == 2
    assert loaded.turn_tag_entries[0]["tags"] == ["auth", "debug"]


def test_payload_token_cache_roundtrip(provider, mock_redis):
    cache = PayloadTokenCache(
        format_name="anthropic",
        message_key="messages",
        shell_fingerprint="shell-123",
        shell_tokens=42,
        message_fingerprints=["m1", "m2"],
        message_tokens=[10, 12],
        separator_tokens=1,
        total_tokens=65,
    )

    provider.save_payload_token_cache("conv-123", cache)

    loaded = provider.load_payload_token_cache("conv-123")
    assert loaded == cache


def test_payload_token_cache_roundtrip_outbound_scope(provider, mock_redis):
    cache = PayloadTokenCache(
        format_name="anthropic",
        message_key="messages",
        shell_fingerprint="shell-out",
        shell_tokens=55,
        message_fingerprints=["m1"],
        message_tokens=[18],
        separator_tokens=0,
        total_tokens=73,
    )

    provider.save_payload_token_cache("conv-123", cache, scope="outbound")

    loaded = provider.load_payload_token_cache("conv-123", scope="outbound")
    assert loaded == cache


def test_delete_clears_payload_token_cache(provider, mock_redis):
    provider.save_payload_token_cache(
        "conv-123",
        PayloadTokenCache(
            format_name="anthropic",
            message_key="messages",
            shell_fingerprint="shell-123",
            shell_tokens=42,
            message_fingerprints=["m1"],
            message_tokens=[10],
            separator_tokens=0,
            total_tokens=52,
        ),
    )

    provider.delete("conv-123")

    assert mock_redis._test_store.get("vc:payload_tokens:inbound:conv-123") is None
    assert mock_redis._test_store.get("vc:payload_tokens:outbound:conv-123") is None


def test_tag_embedding_cache_roundtrip(provider):
    embeddings = {
        "database": [0.1, 0.2, 0.3],
        "api": [0.4, 0.5, 0.6],
    }

    provider.save_tag_embeddings("model-x", embeddings)

    loaded = provider.load_tag_embeddings("model-x", ["database", "api", "missing"])
    assert loaded == embeddings


def test_tag_embedding_runtime_cache_avoids_repeat_redis_loads(provider, mock_redis):
    embeddings = {
        "database": [0.1, 0.2, 0.3],
        "api": [0.4, 0.5, 0.6],
    }
    provider.save_tag_embeddings("model-x", embeddings)
    provider._tag_embedding_runtime_cache.clear()

    first = provider.load_tag_embeddings("model-x", ["database", "api"])
    assert first == embeddings
    assert mock_redis.mget.call_count == 1

    mock_redis.mget.reset_mock()
    second = provider.load_tag_embeddings("model-x", ["database", "api"])
    assert second == embeddings
    mock_redis.mget.assert_not_called()


def test_context_hint_cache_roundtrip(provider):
    provider.save_context_hint_cache("conv-123", "fingerprint-1", "<context-topics>cached</context-topics>")

    loaded = provider.load_context_hint_cache("conv-123", "fingerprint-1")
    assert loaded == "<context-topics>cached</context-topics>"


def test_tag_stats_snapshot_roundtrip(provider):
    stats = [
        TagStats(tag="api", usage_count=3, total_full_tokens=300, total_summary_tokens=75),
        TagStats(tag="auth", usage_count=1, total_full_tokens=120, total_summary_tokens=30),
    ]

    provider.save_tag_stats_snapshot("conv-123", stats)
    loaded = provider.load_tag_stats_snapshot("conv-123")

    assert loaded == stats


def test_tag_summary_embedding_snapshot_roundtrip(provider):
    embeddings = {
        "api": [3.0, 4.0],
        "auth": [0.0, 2.0],
    }

    provider.save_tag_summary_embedding_snapshot("conv-123", embeddings)
    loaded = provider.load_tag_summary_embedding_snapshot("conv-123")

    assert loaded is not None
    assert pytest.approx(loaded["api"][0], rel=1e-4) == 0.6
    assert pytest.approx(loaded["api"][1], rel=1e-4) == 0.8
    assert pytest.approx(loaded["auth"][0], rel=1e-4) == 0.0
    assert pytest.approx(loaded["auth"][1], rel=1e-4) == 1.0
