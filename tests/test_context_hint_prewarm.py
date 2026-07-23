"""Context-hint cache pre-warm at compaction commit.

Compaction changes the engine-state fields baked into the context-hint
cache key (`compacted_prefix_messages`, `last_compacted_turn`, ...), so
the first request after every compaction missed BOTH cache layers and
rebuilt the hint from every tag summary inside the request hot path —
tens of seconds on large conversations.

The compaction pipeline now pre-warms the hint at commit time: it
rebuilds the hint under the NEW cache key and saves it through the same
in-process + cross-worker layers `_build_context_hint` already uses, so
the first post-compaction request on any worker gets a cache hit.

Guarantees pinned here:
* commit warms both layers under the exact key the next request computes
* same-worker ordering: the next request is an in-process hit
* cross-worker ordering: a different engine (fresh in-process cache,
  shared cross-worker layer) gets a cross_worker_hit without touching
  tag summaries
* pre-warm failure never fails the compaction commit
* fenced path: pre-warm is skipped when the compaction lease is lost
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeSessionProvider:
    """In-memory stand-in for the cross-worker (Redis) cache layer."""

    def __init__(self) -> None:
        self.hints: dict[tuple[str, str], str] = {}
        self.save_calls: list[tuple[str, str]] = []
        self.load_calls: list[tuple[str, str]] = []

    def load_context_hint_cache(self, conversation_id: str, cache_key: str):
        self.load_calls.append((conversation_id, cache_key))
        return self.hints.get((conversation_id, cache_key))

    def save_context_hint_cache(
        self, conversation_id: str, cache_key: str, hint: str,
    ) -> None:
        self.save_calls.append((conversation_id, cache_key))
        self.hints[(conversation_id, cache_key)] = hint

    def refresh_tag_stats_snapshot(self, conversation_id: str) -> None:
        pass

    def refresh_tag_summary_embedding_snapshot(self, conversation_id: str) -> None:
        pass

    def load_tag_stats_snapshot(self, conversation_id: str):
        return None

    def save_tag_stats_snapshot(self, *args, **kwargs) -> None:
        pass


def _make_engine(tmp_path, conversation_id: str = "c"):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    cfg = load_config(config_dict={
        "context_window": 10000,
        "conversation_id": conversation_id,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / f"{conversation_id}.db")},
        },
        "tag_generator": {"type": "keyword"},
        "compaction": {"protected_recent_turns": 1},
    })
    return VirtualContextEngine(config=cfg)


def _stub_compactor():
    from virtual_context.types import CompactionResult, SegmentMetadata, TagSummary
    now = datetime.now(timezone.utc)
    compactor = MagicMock()

    def _compact(segments, **_kwargs):
        return [
            CompactionResult(
                segment_id=getattr(seg, "id", f"seg-{i}"),
                primary_tag=getattr(seg, "primary_tag", "topic"),
                tags=list(getattr(seg, "tags", ["topic"])),
                summary=f"summary {i}",
                summary_tokens=4,
                full_text="full text",
                original_tokens=20,
                messages=[{"role": "user", "content": "x"}],
                metadata=SegmentMetadata(turn_count=1, session_date=""),
                compression_ratio=0.5,
                timestamp=now,
                facts=[],
            )
            for i, seg in enumerate(segments)
        ]

    compactor.compact.side_effect = _compact
    compactor.compact_tag_summaries.return_value = [
        TagSummary(tag="topic", summary="tag summary text"),
    ]
    compactor.model_name = "test-model"
    return compactor


def _ingest_pairs(engine, n_pairs: int) -> list:
    """Persist canonical rows + index entries the way the proxy flow does."""
    from virtual_context.proxy.formats import detect_format, extract_ingestible_messages
    body = {"messages": []}
    for i in range(n_pairs):
        body["messages"] += [
            {"role": "user", "content": f"tell me about topic number {i} in detail"},
            {"role": "assistant", "content": f"here is a long reply about topic {i}"},
        ]
    fmt = detect_format(body)
    engine._ingest_reconciler.ingest_batch(
        engine.config.conversation_id, body=body, fmt=fmt,
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )
    messages, _ = extract_ingestible_messages(body, fmt)
    engine.ingest_history(
        messages,
        require_existing_canonical=True,
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )
    return messages


def _compacted_engine(tmp_path, provider=None, conversation_id: str = "c"):
    """Full engine that has run one compaction through compact_manual."""
    engine = _make_engine(tmp_path, conversation_id)
    provider = provider if provider is not None else _FakeSessionProvider()
    engine._retrieval._session_state_provider = provider
    engine._compaction._session_state_provider = provider
    compactor = _stub_compactor()
    engine._compaction._compactor = compactor
    engine._tagging._compactor = compactor
    history = _ingest_pairs(engine, 6)
    report = engine._compaction.compact_manual(history)
    return engine, provider, report


# ---------------------------------------------------------------------------
# Commit-time warm
# ---------------------------------------------------------------------------


class TestWarmAtCommit:
    def test_prewarm_overwrites_stale_same_key_in_both_layers(self, tmp_path):
        """A derived rebuild may change summaries without changing the key."""
        engine, provider, _report = _compacted_engine(tmp_path)
        try:
            mode = (
                engine._retrieval._resolve_paging_mode("")
                if engine.config.paging.enabled
                else None
            )
            key = engine._retrieval._build_context_hint_cache_key(mode)
            conv = engine.config.conversation_id
            stale = "<context-topics>stale derived content</context-topics>"
            engine._retrieval._context_hint_cache_key = key
            engine._retrieval._context_hint_cache_value = stale
            provider.hints[(conv, key)] = stale
            provider.load_calls.clear()

            engine._retrieval._store = MagicMock(
                wraps=engine._retrieval._store,
            )
            hint = engine._retrieval.prewarm_context_hint_cache()

            assert hint and hint != stale
            assert provider.hints[(conv, key)] == hint
            assert not provider.load_calls, "forced prewarm must bypass shared reads"
            assert engine._retrieval._store.get_all_tag_summaries.called
        finally:
            engine.close()

    def test_commit_saves_hint_under_next_request_key(self, tmp_path):
        engine, provider, report = _compacted_engine(tmp_path)
        try:
            assert report is not None and report.segments_compacted > 0
            assert engine._engine_state.compacted_prefix_messages > 0

            # The key the NEXT request will compute (post-commit state).
            paging_enabled = engine.config.paging.enabled
            mode = (
                engine._retrieval._resolve_paging_mode("")
                if paging_enabled else None
            )
            next_key = engine._retrieval._build_context_hint_cache_key(mode)

            conv = engine.config.conversation_id
            assert (conv, next_key) in provider.hints, (
                f"commit did not warm the cross-worker layer under the "
                f"next-request key; saved keys: {provider.save_calls}"
            )
            assert provider.hints[(conv, next_key)], "warmed hint is empty"
        finally:
            engine.close()

    def test_same_worker_next_request_is_in_process_hit(self, tmp_path):
        engine, provider, _report = _compacted_engine(tmp_path)
        try:
            fetch_calls = []
            original = engine._retrieval._store.get_all_tag_summaries

            def _counting(*args, **kwargs):
                fetch_calls.append(args)
                return original(*args, **kwargs)

            engine._retrieval._store = MagicMock(wraps=engine._retrieval._store)
            engine._retrieval._store.get_all_tag_summaries.side_effect = _counting

            hint = engine._retrieval._build_context_hint()
            assert hint, "post-compaction hint must be non-empty"
            assert not fetch_calls, (
                "same-worker post-commit request rebuilt the hint instead "
                "of hitting the pre-warmed in-process cache"
            )
        finally:
            engine.close()

    def test_cross_worker_fresh_engine_hits_shared_layer(self, tmp_path):
        engine, provider, _report = _compacted_engine(tmp_path)
        try:
            # Second engine: same store + shared cross-worker layer, cold
            # in-process cache — simulates the prepare landing on another
            # worker after the compacting worker committed.
            engine2 = _make_engine(tmp_path, "c")
            try:
                engine2._retrieval._session_state_provider = provider
                # Mirror the committed engine-state fields the key hashes.
                for field in (
                    "compacted_prefix_messages",
                    "flushed_prefix_messages",
                    "last_compacted_turn",
                    "conversation_generation",
                ):
                    setattr(
                        engine2._engine_state, field,
                        getattr(engine._engine_state, field),
                    )
                engine2._retrieval._store = MagicMock(
                    wraps=engine2._retrieval._store
                )
                instrumentation: dict[str, object] = {"force": True}
                instrumentation.clear()
                hint = engine2._retrieval._build_context_hint(
                    instrumentation=instrumentation,
                )
                assert hint == engine._retrieval._context_hint_cache_value
                assert not engine2._retrieval._store.get_all_tag_summaries.called, (
                    "fresh worker rebuilt the hint instead of hitting the "
                    "cross-worker layer warmed at commit"
                )
            finally:
                engine2.close()
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    def test_prewarm_failure_does_not_fail_commit(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            provider = _FakeSessionProvider()
            engine._retrieval._session_state_provider = provider
            engine._compaction._session_state_provider = provider
            compactor = _stub_compactor()
            engine._compaction._compactor = compactor
            engine._tagging._compactor = compactor
            history = _ingest_pairs(engine, 6)

            def _boom():
                raise RuntimeError("prewarm exploded")

            engine._retrieval.prewarm_context_hint_cache = _boom
            engine._compaction._prewarm_context_hint_callback = _boom

            report = engine._compaction.compact_manual(history)
            assert report is not None and report.segments_compacted > 0
        finally:
            engine.close()

    def test_cache_save_failure_does_not_fail_commit(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            provider = _FakeSessionProvider()

            def _raise(*args, **kwargs):
                raise ConnectionError("redis down")

            provider.save_context_hint_cache = _raise
            engine._retrieval._session_state_provider = provider
            engine._compaction._session_state_provider = provider
            compactor = _stub_compactor()
            engine._compaction._compactor = compactor
            engine._tagging._compactor = compactor
            history = _ingest_pairs(engine, 6)
            report = engine._compaction.compact_manual(history)
            assert report is not None and report.segments_compacted > 0
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# Fence semantics
# ---------------------------------------------------------------------------


class _ClaimResult:
    def __init__(self, claimed: bool) -> None:
        self.claimed = claimed


class TestFenceSemantics:
    def _pipeline(self, tmp_path, *, claimed: bool):
        engine = _make_engine(tmp_path)
        pipeline = engine._compaction
        pipeline._worker_id = "worker-1"
        pipeline._store = MagicMock(wraps=pipeline._store)
        pipeline._store.claim_compaction_lease.side_effect = None
        pipeline._store.claim_compaction_lease.return_value = _ClaimResult(claimed)
        calls = []
        pipeline._prewarm_context_hint_callback = lambda: calls.append(True) or ""
        return engine, pipeline, calls

    def test_fenced_prewarm_runs_when_lease_held(self, tmp_path):
        engine, pipeline, calls = self._pipeline(tmp_path, claimed=True)
        try:
            pipeline._prewarm_context_hint("op-1")
            assert calls, "pre-warm must run when the lease is still held"
            assert pipeline._store.claim_compaction_lease.called
        finally:
            engine.close()

    def test_fenced_prewarm_skipped_when_lease_lost(self, tmp_path):
        engine, pipeline, calls = self._pipeline(tmp_path, claimed=False)
        try:
            pipeline._prewarm_context_hint("op-1")
            assert not calls, (
                "pre-warm must NOT write a hint after the compaction lease "
                "was lost"
            )
        finally:
            engine.close()

    def test_unfenced_prewarm_skips_ownership_probe(self, tmp_path):
        engine, pipeline, calls = self._pipeline(tmp_path, claimed=False)
        try:
            pipeline._prewarm_context_hint(None)
            assert calls, "unfenced (manual/headless) pre-warm must run"
            assert not pipeline._store.claim_compaction_lease.called
        finally:
            engine.close()

    def test_ownership_probe_failure_falls_back_to_skip(self, tmp_path):
        engine, pipeline, calls = self._pipeline(tmp_path, claimed=True)
        try:
            pipeline._store.claim_compaction_lease.side_effect = RuntimeError("db down")
            pipeline._prewarm_context_hint("op-1")
            assert not calls, (
                "when ownership cannot be verified the pre-warm must not run"
            )
        finally:
            engine.close()
