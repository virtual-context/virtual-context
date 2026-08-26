"""The per-model tag-embedding runtime cache bound is configurable.

The bound was hardcoded at 5000 while real conversations carry larger
tag vocabularies, so the layer-two runtime cache evicted live entries on
every prepare and re-materialized them from the shared cache — a
self-inflicted per-request tax that grows with vocabulary. The bound is
now a constructor argument with an environment override
(``VC_TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL``), parsed once at
construction; an invalid value fails loudly rather than silently
falling back.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from unittest.mock import MagicMock

import pytest

from virtual_context.proxy.session_state import SessionStateProvider


def _provider(**kwargs) -> SessionStateProvider:
    return SessionStateProvider(redis_client=MagicMock(), **kwargs)


def _fill(provider: SessionStateProvider, count: int) -> None:
    for i in range(count):
        provider._remember_runtime_tag_embedding("m", f"tag-{i}", [0.5])


def test_default_bound_is_unchanged():
    provider = _provider()
    assert provider._tag_embedding_runtime_max_per_model == 5000


def test_constructor_bound_governs_eviction():
    provider = _provider(tag_embedding_runtime_max_per_model=3)
    _fill(provider, 5)
    cache = provider._runtime_tag_cache("m")
    assert len(cache) == 3
    assert list(cache) == ["tag-2", "tag-3", "tag-4"]


def test_env_override_applies_when_no_argument(monkeypatch):
    monkeypatch.setenv("VC_TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL", "7")
    provider = _provider()
    assert provider._tag_embedding_runtime_max_per_model == 7
    _fill(provider, 9)
    assert len(provider._runtime_tag_cache("m")) == 7


def test_argument_wins_over_environment(monkeypatch):
    monkeypatch.setenv("VC_TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL", "7")
    provider = _provider(tag_embedding_runtime_max_per_model=11)
    assert provider._tag_embedding_runtime_max_per_model == 11


@pytest.mark.parametrize("raw", ["", "  "])
def test_blank_environment_keeps_the_default(monkeypatch, raw):
    monkeypatch.setenv("VC_TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL", raw)
    provider = _provider()
    assert provider._tag_embedding_runtime_max_per_model == 5000


@pytest.mark.parametrize("raw", ["banana", "0", "-5"])
def test_invalid_values_fail_loudly(monkeypatch, raw):
    monkeypatch.setenv("VC_TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL", raw)
    with pytest.raises(ValueError):
        _provider()


def test_invalid_argument_fails_loudly():
    with pytest.raises(ValueError):
        _provider(tag_embedding_runtime_max_per_model=0)
