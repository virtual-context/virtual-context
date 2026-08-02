# tests/test_session_state.py
"""Tests for Redis-backed SessionStateProvider."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import fakeredis
import pytest
import redis
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
    s.activate_conversation.return_value = 0
    return s


@pytest.fixture
def provider(mock_redis, mock_store):
    return SessionStateProvider(redis_client=mock_redis, store=mock_store)


def test_load_returns_none_for_missing_key(provider, mock_redis):
    result = provider.load("conv-123")
    assert result is None


def test_successful_redis_read_clears_prior_degraded_state(provider, mock_redis):
    # First load: primary GET fails, fallback's freshness marker GET succeeds.
    # Second load: primary GET succeeds with a clean miss.
    mock_redis.get.side_effect = [RuntimeError("redis unavailable"), None, None]

    assert provider.load("conv-recovery") is None
    assert provider.is_degraded is True

    assert provider.load("conv-recovery") is None
    assert provider.is_degraded is False


def test_authoritative_load_never_uses_durable_fallback(provider, mock_redis, mock_store):
    mock_redis.get.side_effect = RuntimeError("selective redis failure")

    with pytest.raises(RuntimeError, match="selective redis failure"):
        provider.load_authoritative("conv-authority")

    mock_store.load_engine_state.assert_not_called()
    assert provider.is_degraded is True


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


def test_save_rejects_checkpoint_from_previous_conversation_generation(
    provider, mock_redis,
):
    mock_redis._test_store["vc:session:conv-123"] = SessionState(
        conversation_generation=2,
        version=0,
    ).to_json()

    stale = SessionState(
        conversation_generation=0,
        version=7,
        last_completed_turn=99,
    )
    assert provider.save("conv-123", stale) is None

    loaded = provider.load("conv-123")
    assert loaded is not None
    assert loaded.conversation_generation == 2
    assert loaded.last_completed_turn == -1


class _LifecycleStore:
    def __init__(self, *, generation=1, deleted=False):
        self.generation = generation
        self.deleted = deleted
        self.saved = []

    def get_conversation_generation(self, _conversation_id):
        return self.generation

    def is_conversation_deleted(self, _conversation_id):
        return self.deleted

    def save_engine_state(self, snapshot):
        self.saved.append(snapshot)


def _repair_provider(*, generation=1, deleted=False):
    redis_client = fakeredis.FakeRedis(decode_responses=False)
    store = _LifecycleStore(generation=generation, deleted=deleted)
    return SessionStateProvider(redis_client=redis_client, store=store), redis_client, store


def test_marker_repair_exact_cas_advances_generation_and_preserves_payload():
    repair_provider, redis_client, store = _repair_provider(generation=1)
    key = "vc:session:conv-repair"
    original = json.loads(SessionState(
        conversation_generation=0,
        checkpoint_version=3,
        version=7,
        provider="keep-provider",
        working_set=[{"tag": "keep-me"}],
    ).to_json())
    original["future_unknown_field"] = {"nested": [1, 2, 3]}
    original_raw = json.dumps(original, separators=(",", ":")).encode()
    redis_client.set(key, original_raw)

    expected_raw, existing = repair_provider.load_authoritative_snapshot(
        "conv-repair"
    )
    markers = SessionState(
        compacted_prefix_messages=12,
        flushed_prefix_messages=10,
        last_compacted_turn=8,
        last_completed_turn=11,
        last_indexed_turn=11,
        turn_tag_entries=[{"turn_number": 11, "tags": ["topic"]}],
        conversation_generation=1,
    )
    saved_version = repair_provider.repair_session_state_markers(
        "conv-repair",
        expected_raw=expected_raw,
        markers=markers,
        durable_generation=1,
        allow_generation_promotion=True,
    )

    committed = json.loads(redis_client.get(key))
    assert existing is not None
    assert saved_version == 8
    assert committed["version"] == 8
    assert committed["checkpoint_version"] == 4
    assert committed["conversation_generation"] == 1
    assert committed["compacted_prefix_messages"] == 12
    assert committed["flushed_prefix_messages"] == 10
    assert committed["last_completed_turn"] == 11
    assert committed["turn_tag_entries"] == markers.turn_tag_entries
    assert committed["provider"] == "keep-provider"
    assert committed["working_set"] == [{"tag": "keep-me"}]
    assert committed["future_unknown_field"] == {"nested": [1, 2, 3]}
    assert len(store.saved) == 1
    assert store.saved[0].conversation_generation == 1


def test_marker_repair_creates_first_checkpoint_from_missing_key():
    repair_provider, _redis_client, _store = _repair_provider(generation=1)

    saved_version = repair_provider.repair_session_state_markers(
        "conv-missing",
        expected_raw=None,
        markers=SessionState(
            conversation_generation=1,
            checkpoint_version=1,
            last_completed_turn=2,
        ),
        durable_generation=1,
    )

    committed = repair_provider.load_authoritative("conv-missing")
    assert committed is not None
    assert saved_version == 1
    assert committed.version == 1
    assert committed.checkpoint_version == 1
    assert committed.conversation_generation == 1
    assert committed.last_completed_turn == 2


def test_marker_repair_requires_explicit_admin_generation_promotion():
    repair_provider, redis_client, _store = _repair_provider(generation=1)
    key = "vc:session:conv-promotion-policy"
    raw = SessionState(conversation_generation=0, version=2).to_json()
    redis_client.set(key, raw)

    with pytest.raises(RuntimeError, match="non-administrative"):
        repair_provider.repair_session_state_markers(
            "conv-promotion-policy",
            expected_raw=raw,
            markers=SessionState(conversation_generation=1),
            durable_generation=1,
        )

    assert redis_client.get(key) == raw


@pytest.mark.parametrize(
    ("redis_state", "store_deleted", "error"),
    [
        (SessionState(deleted=True, version=2**53), False, "tombstone"),
        (SessionState(conversation_generation=2), False, "downgrade"),
        (SessionState(conversation_generation=-1), False, "invalid"),
        (SessionState(conversation_generation=0), True, "deleted conversation"),
    ],
)
def test_marker_repair_rejects_unsafe_lifecycle_state(
    redis_state,
    store_deleted,
    error,
):
    repair_provider, redis_client, _store = _repair_provider(
        generation=1,
        deleted=store_deleted,
    )
    key = "vc:session:conv-unsafe"
    raw = redis_state.to_json()
    redis_client.set(key, raw)
    markers = SessionState(conversation_generation=1)

    with pytest.raises(RuntimeError, match=error):
        repair_provider.repair_session_state_markers(
            "conv-unsafe",
            expected_raw=raw,
            markers=markers,
            durable_generation=1,
        )

    assert redis_client.get(key) == raw


def test_marker_repair_rejects_changed_exact_preimage_and_busy_lease():
    repair_provider, redis_client, _store = _repair_provider(generation=1)
    key = "vc:session:conv-race"
    expected = SessionState(conversation_generation=0, version=1).to_json()
    changed = SessionState(conversation_generation=0, version=2).to_json()
    redis_client.set(key, changed)
    markers = SessionState(conversation_generation=1)

    with pytest.raises(RuntimeError, match="changed before marker repair"):
        repair_provider.repair_session_state_markers(
            "conv-race",
            expected_raw=expected,
            markers=markers,
            durable_generation=1,
        )
    assert redis_client.get(key) == changed

    redis_client.set("vc:lifecycle_lease:conv-race", "other-owner")
    with pytest.raises(RuntimeError, match="lifecycle lease is busy"):
        repair_provider.repair_session_state_markers(
            "conv-race",
            expected_raw=changed,
            markers=markers,
            durable_generation=1,
        )
    assert redis_client.get(key) == changed


def test_marker_repair_retries_benign_watched_lease_renewal():
    backing = fakeredis.FakeRedis(decode_responses=False)

    class OneWatchRaceRedis:
        inject_race = True

        def pipeline(self, *args, **kwargs):
            pipe = backing.pipeline(*args, **kwargs)
            if self.inject_race:
                self.inject_race = False

                def _raise_watch_error(*_args, **_kwargs):
                    raise redis.WatchError("simulated lease renewal")

                pipe.execute = _raise_watch_error
            return pipe

        def __getattr__(self, name):
            return getattr(backing, name)

    store = _LifecycleStore(generation=1, deleted=False)
    repair_provider = SessionStateProvider(
        redis_client=OneWatchRaceRedis(),
        store=store,
    )
    conversation_id = "conv-renewal-race"
    key = f"vc:session:{conversation_id}"
    raw = SessionState(conversation_generation=1, version=4).to_json()
    backing.set(key, raw)

    saved_version = repair_provider.repair_session_state_markers(
        conversation_id,
        expected_raw=raw,
        markers=SessionState(
            conversation_generation=1,
            last_completed_turn=9,
        ),
        durable_generation=1,
    )

    committed = repair_provider.load_authoritative(conversation_id)
    assert saved_version == 5
    assert committed is not None
    assert committed.last_completed_turn == 9

@pytest.mark.parametrize(
    ("payload_update", "error"),
    [
        ({"conversation_generation": "0"}, "generation is invalid"),
        ({"version": "1"}, "version is invalid"),
        ({"checkpoint_version": True}, "checkpoint version is invalid"),
        ({"deleted": "false"}, "deletion marker is invalid"),
    ],
)
def test_marker_repair_rejects_malformed_security_fields(
    payload_update,
    error,
):
    repair_provider, redis_client, _store = _repair_provider(generation=1)
    key = "vc:session:conv-malformed"
    payload = json.loads(SessionState(conversation_generation=1).to_json())
    payload.update(payload_update)
    raw = json.dumps(payload).encode()
    redis_client.set(key, raw)

    with pytest.raises(RuntimeError, match=error):
        repair_provider.repair_session_state_markers(
            "conv-malformed",
            expected_raw=raw,
            markers=SessionState(conversation_generation=1),
            durable_generation=1,
        )

    assert redis_client.get(key) == raw


def test_sqlite_redis_generation_drift_repair_then_normal_save(tmp_path):
    """Reproduce DB=1/Redis=0 with real lifecycle and checkpoint storage."""
    from virtual_context.storage.sqlite import SQLiteStore

    conversation_id = "sqlite-generation-drift"
    store = SQLiteStore(db_path=str(tmp_path / "generation-drift.db"))
    try:
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO conversation_lifecycle
               (conversation_id, generation, deleted, updated_at)
               VALUES (?, 1, 0, datetime('now'))""",
            (conversation_id,),
        )
        conn.commit()
        redis_client = fakeredis.FakeRedis(decode_responses=False)
        provider = SessionStateProvider(
            redis_client=redis_client,
            store=store,
        )
        redis_client.set(
            f"vc:session:{conversation_id}",
            SessionState(
                conversation_generation=0,
                checkpoint_version=2,
                version=4,
                provider="preserved",
            ).to_json(),
        )
        expected_raw, _existing = provider.load_authoritative_snapshot(
            conversation_id
        )

        repaired_version = provider.repair_session_state_markers(
            conversation_id,
            expected_raw=expected_raw,
            markers=SessionState(
                conversation_generation=1,
                last_completed_turn=6,
                last_indexed_turn=6,
            ),
            durable_generation=1,
            allow_generation_promotion=True,
        )
        repaired = provider.load_authoritative(conversation_id)
        assert repaired is not None
        assert repaired_version == 5
        assert repaired.conversation_generation == 1
        assert repaired.provider == "preserved"

        repaired.last_completed_turn = 7
        assert provider.save(conversation_id, repaired) == 6
        committed = provider.load_authoritative(conversation_id)
        assert committed is not None
        assert committed.conversation_generation == 1
        assert committed.version == 6
        assert committed.last_completed_turn == 7
        durable = store.load_engine_state(conversation_id)
        assert durable is not None
        assert durable.conversation_generation == 1
        assert durable.last_completed_turn == 7
    finally:
        store.close()

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
