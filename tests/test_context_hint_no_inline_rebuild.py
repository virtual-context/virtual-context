"""A request never rebuilds the context hint just because compaction moved on.

The hint a request uses depends only on what compaction produced. The
post-compaction prewarm writes it under the key the next request computes,
and when that key is not there yet the request proceeds with the hint built
for the earlier compaction instead of rendering one inline.
"""
from __future__ import annotations

import fakeredis
import pytest
from unittest.mock import MagicMock

from tests.test_context_hint_prewarm import _FakeSessionProvider, _compacted_engine
from virtual_context.proxy.session_state import SessionStateProvider


class _LatestAwareProvider(_FakeSessionProvider):
    def __init__(self):
        super().__init__()
        self.latest: dict[str, dict] = {}

    def save_latest_context_hint(self, conversation_id, generation, paging_mode, hint):
        self.latest[conversation_id] = {"generation": generation, "mode": paging_mode, "hint": hint}

    def load_latest_context_hint(self, conversation_id, generation, paging_mode):
        data = self.latest.get(conversation_id)
        if not data or data["generation"] != generation or data["mode"] != paging_mode:
            return None
        return data["hint"]


def _mode(engine):
    return engine._retrieval._resolve_paging_mode("") if engine.config.paging.enabled else None


@pytest.mark.regression("PROXY-030")
def test_the_flushed_watermark_does_not_change_the_hint_key(tmp_path):
    engine, _provider, _report = _compacted_engine(tmp_path)
    try:
        before = engine._retrieval._build_context_hint_cache_key(_mode(engine))
        engine._engine_state.flushed_prefix_messages += 7  # a different, lagging session watermark
        assert engine._retrieval._build_context_hint_cache_key(_mode(engine)) == before
        engine._engine_state.compacted_prefix_messages += 2
        assert engine._retrieval._build_context_hint_cache_key(_mode(engine)) != before
    finally:
        engine.close()


@pytest.mark.regression("PROXY-030")
def test_a_request_after_the_prewarm_hits_the_prewarmed_hint(tmp_path):
    engine, provider, _report = _compacted_engine(tmp_path, provider=_LatestAwareProvider())
    try:
        prewarmed = engine._retrieval.prewarm_context_hint_cache()
        engine._engine_state.flushed_prefix_messages += 7  # the request's session watermark differs
        engine._retrieval._context_hint_cache_key = ""    # another worker: nothing in process
        engine._retrieval._store = MagicMock(wraps=engine._retrieval._store)
        assert engine._retrieval._build_context_hint(_mode(engine)) == prewarmed
        assert not engine._retrieval._store.get_tag_summary_topics.called
    finally:
        engine.close()


@pytest.mark.regression("PROXY-030")
def test_a_request_before_the_prewarm_uses_the_earlier_hint_instead_of_rebuilding(tmp_path):
    engine, provider, _report = _compacted_engine(tmp_path, provider=_LatestAwareProvider())
    try:
        earlier = engine._retrieval.prewarm_context_hint_cache()
        engine._engine_state.compacted_prefix_messages += 2  # compaction moved; no prewarm yet
        engine._engine_state.last_compacted_turn += 1
        engine._retrieval._context_hint_cache_key = ""
        engine._retrieval._store = MagicMock(wraps=engine._retrieval._store)
        info: dict = {}
        assert engine._retrieval._build_context_hint(_mode(engine), instrumentation=info) == earlier
        assert info.get("cache_layer") == "latest_fallback"
        assert not engine._retrieval._store.get_tag_summary_topics.called
    finally:
        engine.close()


@pytest.mark.regression("PROXY-030")
def test_an_earlier_generation_is_never_served(tmp_path):
    engine, provider, _report = _compacted_engine(tmp_path, provider=_LatestAwareProvider())
    try:
        engine._retrieval.prewarm_context_hint_cache()
        engine._engine_state.conversation_generation += 1  # deleted and recreated
        engine._retrieval._context_hint_cache_key = ""
        engine._retrieval._store = MagicMock(wraps=engine._retrieval._store)
        info: dict = {}
        engine._retrieval._build_context_hint(_mode(engine), instrumentation=info)
        assert info.get("cache_layer") == "both_miss"
        assert engine._retrieval._store.get_tag_summary_topics.called
    finally:
        engine.close()


@pytest.mark.regression("PROXY-030")
def test_redis_latest_slot_round_trip_and_generation_guard():
    p = SessionStateProvider(fakeredis.FakeRedis(decode_responses=False))
    assert p.load_latest_context_hint("conv", 1, "autonomous") is None
    p.save_latest_context_hint("conv", 1, "autonomous", "<context-topics>a</context-topics>")
    assert p.load_latest_context_hint("conv", 1, "autonomous") == "<context-topics>a</context-topics>"
    assert p.load_latest_context_hint("conv", 2, "autonomous") is None
    assert p.load_latest_context_hint("conv", 1, "supervised") is None


@pytest.mark.regression("BUG-121")
def test_redis_latest_slot_ignores_hints_rendered_by_another_hint_builder(monkeypatch):
    from virtual_context.core import hint_builder

    p = SessionStateProvider(fakeredis.FakeRedis(decode_responses=False))
    p.save_latest_context_hint("conv", 1, "supervised", "<context-topics>old</context-topics>")
    monkeypatch.setattr(hint_builder, "TEMPLATE_FINGERPRINT", "changed-template")
    assert p.load_latest_context_hint("conv", 1, "supervised") is None
