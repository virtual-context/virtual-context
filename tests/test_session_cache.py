"""Tests for RedisSessionCache."""
import json
import pytest
from unittest.mock import MagicMock
from virtual_context.proxy.session_cache import RedisSessionCache


class TestRedisSessionCacheNoRedis:
    def test_empty_url_disables(self):
        cache = RedisSessionCache("")
        assert not cache.is_available()
        assert cache.load_snapshot("conv-1") is None

    def test_save_when_unavailable_is_noop(self):
        cache = RedisSessionCache("")
        cache.save_snapshot("conv-1", {"version": 1})  # should not raise

    def test_delete_when_unavailable_is_noop(self):
        cache = RedisSessionCache("")
        cache.delete_conversation("conv-1")  # should not raise

    def test_degraded_save_is_noop(self):
        cache = RedisSessionCache("redis://localhost:6379")
        cache._degraded = True
        cache.save_snapshot("conv-1", {"version": 1})  # no-op


class TestRedisSessionCacheMocked:
    def _make_cache(self):
        cache = RedisSessionCache("redis://localhost:6379")
        cache._redis = MagicMock()
        cache._redis_available = True
        cache._degraded = False
        return cache

    def test_save_sets_correct_key(self):
        cache = self._make_cache()
        cache.save_snapshot("conv-1", {"version": 1, "history": []})
        cache._redis.set.assert_called_once()
        key = cache._redis.set.call_args[0][0]
        assert key == "vc:conv-1:snapshot"

    def test_load_returns_parsed_json(self):
        cache = self._make_cache()
        snapshot = {"version": 1, "history": [{"role": "user"}]}
        cache._redis.get.return_value = json.dumps(snapshot).encode()
        result = cache.load_snapshot("conv-1")
        assert result == snapshot

    def test_load_returns_none_on_miss(self):
        cache = self._make_cache()
        cache._redis.get.return_value = None
        assert cache.load_snapshot("conv-1") is None

    def test_delete_calls_redis_delete(self):
        cache = self._make_cache()
        cache.delete_conversation("conv-1")
        cache._redis.delete.assert_called_once_with("vc:conv-1:snapshot")

    def test_retry_succeeds_after_failures(self):
        cache = self._make_cache()
        cache._redis.set.side_effect = [ConnectionError, ConnectionError, None]
        cache.save_snapshot("conv-1", {"version": 1})
        assert cache._redis.set.call_count == 3
        assert not cache._degraded

    def test_degraded_after_all_retries_fail(self):
        cache = self._make_cache()
        cache._redis.set.side_effect = ConnectionError("down")
        cache.save_snapshot("conv-1", {"version": 1})
        assert cache._degraded is True

    def test_is_available_reflects_state(self):
        cache = self._make_cache()
        assert cache.is_available()
        cache._degraded = True
        assert not cache.is_available()

    def test_history_cap_property(self):
        cache = RedisSessionCache("", history_cap=800)
        assert cache.history_cap == 800
