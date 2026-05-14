"""Unit tests for ``SessionStateProvider.get_marker`` accessor.

Tier 2 of the cross-channel-mirror gate needs a cheap single-marker
Redis read instead of the full ``load`` deserialization. Tests cover
the happy path plus the four ``None`` paths (missing key, malformed
JSON, Redis raises, missing marker name).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from virtual_context.proxy.session_state import SessionStateProvider


def _provider_with_redis(redis_mock) -> SessionStateProvider:
    return SessionStateProvider(redis_client=redis_mock)


def test_get_marker_happy_path_int() -> None:
    redis = MagicMock()
    redis.get.return_value = json.dumps({"last_completed_turn": 7}).encode("utf-8")
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-1", "last_completed_turn") == 7
    redis.get.assert_called_once_with("vc:session:conv-1")


def test_get_marker_returns_none_when_key_missing() -> None:
    redis = MagicMock()
    redis.get.return_value = None
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-2", "last_completed_turn") is None


def test_get_marker_returns_none_when_marker_name_absent() -> None:
    redis = MagicMock()
    redis.get.return_value = json.dumps({"other_field": 5}).encode("utf-8")
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-3", "last_completed_turn") is None


def test_get_marker_returns_none_on_malformed_json() -> None:
    redis = MagicMock()
    redis.get.return_value = b"{not valid json"
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-4", "last_completed_turn") is None


def test_get_marker_returns_none_when_redis_raises() -> None:
    redis = MagicMock()
    redis.get.side_effect = RuntimeError("redis down")
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-5", "last_completed_turn") is None


def test_get_marker_empty_conv_id_returns_none() -> None:
    redis = MagicMock()
    provider = _provider_with_redis(redis)
    assert provider.get_marker("", "last_completed_turn") is None
    redis.get.assert_not_called()


def test_get_marker_empty_marker_name_returns_none() -> None:
    redis = MagicMock()
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-x", "") is None
    redis.get.assert_not_called()


def test_get_marker_string_blob_decoded() -> None:
    """Some Redis clients return str (decode_responses=True). Helper
    must tolerate both bytes and str without exploding."""
    redis = MagicMock()
    redis.get.return_value = json.dumps({"last_completed_turn": 12})
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-y", "last_completed_turn") == 12


def test_get_marker_returns_raw_typed_value() -> None:
    """The returned value is whatever JSON deserialization produced;
    callers are responsible for int-coercion if they need it."""
    redis = MagicMock()
    redis.get.return_value = json.dumps({"some_field": "string-value"}).encode("utf-8")
    provider = _provider_with_redis(redis)
    assert provider.get_marker("conv-z", "some_field") == "string-value"
