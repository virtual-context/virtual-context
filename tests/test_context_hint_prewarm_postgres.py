"""Postgres twin for the context-hint pre-warm at compaction commit.

Skipped unless a Postgres DSN is configured. Exercises the
warm-at-commit path against the Postgres store: the tag-summary reads
feeding the hint build and the canonical/compaction writes all go
through the PG backend, and a fresh engine (cold in-process cache,
shared cross-worker layer) must get a cache hit for the key its first
post-compaction request computes.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from tests.pg_helpers import pg_dsn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


class _FakeSessionProvider:
    def __init__(self) -> None:
        self.hints: dict[tuple[str, str], str] = {}

    def load_context_hint_cache(self, conversation_id: str, cache_key: str):
        return self.hints.get((conversation_id, cache_key))

    def save_context_hint_cache(
        self, conversation_id: str, cache_key: str, hint: str,
    ) -> None:
        self.hints[(conversation_id, cache_key)] = hint

    def refresh_tag_stats_snapshot(self, conversation_id: str) -> None:
        pass

    def refresh_tag_summary_embedding_snapshot(self, conversation_id: str) -> None:
        pass

    def load_tag_stats_snapshot(self, conversation_id: str):
        return None

    def save_tag_stats_snapshot(self, *args, **kwargs) -> None:
        pass


def _make_engine(conversation_id: str):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    cfg = load_config(config_dict={
        "context_window": 10000,
        "conversation_id": conversation_id,
        "storage": {"backend": "postgres", "postgres": {"dsn": PG_URL}},
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


class TestFenceProbePostgres:
    """The pre-warm ownership probe against the real lease machinery."""

    def _seed_operation(self, conv: str, *, owner: str, heartbeat_age_s: float):
        from datetime import datetime, timedelta, timezone
        from tests.pg_helpers import pg_test_conn
        conn = pg_test_conn()
        op_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        conn.execute(
            """
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id
            ) VALUES (%s, %s, 1, 0, 7, 'segment_tagging', 'running',
                      %s, %s, %s)
            """,
            (op_id, conv, now, now - timedelta(seconds=heartbeat_age_s), owner),
        )
        return op_id

    def _pipeline(self, conv: str, worker_id: str):
        """Bind the real pre-warm method onto a stub — no engine build.

        Only the pieces `_prewarm_context_hint` touches are supplied:
        the REAL Postgres store (the probe under test), config identity,
        engine-state epoch, worker id, and a spy callback.
        """
        from types import SimpleNamespace
        from virtual_context.core.compaction_pipeline import CompactionPipeline
        from virtual_context.storage.postgres import PostgresStore

        store = PostgresStore(PG_URL)
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        calls: list[bool] = []
        stub = SimpleNamespace(
            _store=store,
            _config=SimpleNamespace(conversation_id=conv),
            _engine_state=SimpleNamespace(lifecycle_epoch=1),
            _worker_id=worker_id,
            _prewarm_context_hint_callback=lambda: calls.append(True) or "",
            _PREWARM_OWNERSHIP_PROBE_TTL_S=(
                CompactionPipeline._PREWARM_OWNERSHIP_PROBE_TTL_S
            ),
        )
        prewarm = CompactionPipeline._prewarm_context_hint.__get__(stub)
        return store, prewarm, calls

    def test_owner_probe_allows_warm(self):
        conv = f"fence-{uuid.uuid4().hex[:12]}"
        store, prewarm, calls = self._pipeline(conv, "worker-me")
        try:
            op_id = self._seed_operation(conv, owner="worker-me", heartbeat_age_s=0)
            prewarm(op_id)
            assert calls, "owner must be allowed to warm"
        finally:
            store.close()

    def test_other_owner_fresh_heartbeat_skips_warm(self):
        conv = f"fence-{uuid.uuid4().hex[:12]}"
        store, prewarm, calls = self._pipeline(conv, "worker-me")
        try:
            op_id = self._seed_operation(conv, owner="worker-other", heartbeat_age_s=0)
            prewarm(op_id)
            assert not calls, "lost lease (fresh other owner) must skip the warm"
        finally:
            store.close()

    def test_probe_never_steals_stale_other_owner_lease(self):
        """The huge probe TTL disables the stale-heartbeat takeover branch:
        even a long-dead other owner is NOT stolen from — the warm is
        skipped and the row's owner is untouched."""
        from tests.pg_helpers import pg_test_conn
        conv = f"fence-{uuid.uuid4().hex[:12]}"
        store, prewarm, calls = self._pipeline(conv, "worker-me")
        try:
            op_id = self._seed_operation(
                conv, owner="worker-dead", heartbeat_age_s=86400,
            )
            prewarm(op_id)
            assert not calls, "probe must not steal a stale lease to warm"
            row = pg_test_conn().execute(
                "SELECT owner_worker_id FROM compaction_operation "
                "WHERE operation_id = %s",
                (op_id,),
            ).fetchone()
            assert row["owner_worker_id"] == "worker-dead", (
                "ownership probe mutated the operation row's owner"
            )
        finally:
            store.close()


def test_commit_prewarm_serves_fresh_worker_on_postgres():
    from virtual_context.proxy.formats import detect_format, extract_ingestible_messages

    conv = f"prewarm-{uuid.uuid4().hex[:12]}"
    provider = _FakeSessionProvider()
    engine = _make_engine(conv)
    try:
        engine._retrieval._session_state_provider = provider
        engine._compaction._session_state_provider = provider
        compactor = _stub_compactor()
        engine._compaction._compactor = compactor
        engine._tagging._compactor = compactor

        body = {"messages": []}
        for i in range(6):
            body["messages"] += [
                {"role": "user", "content": f"tell me about topic number {i} in detail"},
                {"role": "assistant", "content": f"here is a long reply about topic {i}"},
            ]
        fmt = detect_format(body)
        engine._ingest_reconciler.ingest_batch(
            conv, body=body, fmt=fmt,
            expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
        )
        messages, _ = extract_ingestible_messages(body, fmt)
        engine.ingest_history(
            messages,
            require_existing_canonical=True,
            expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
        )

        report = engine._compaction.compact_manual(messages)
        assert report is not None and report.segments_compacted > 0
        assert provider.hints, "commit must warm the cross-worker layer"

        # Fresh engine = the prepare landing on another worker.
        engine2 = _make_engine(conv)
        try:
            engine2._retrieval._session_state_provider = provider
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
            engine2._retrieval._store = MagicMock(wraps=engine2._retrieval._store)
            hint = engine2._retrieval._build_context_hint()
            assert hint, "fresh worker must see the warmed hint"
            assert not engine2._retrieval._store.get_all_tag_summaries.called, (
                "fresh worker rebuilt the hint instead of hitting the "
                "cross-worker layer warmed at commit"
            )
        finally:
            engine2.close()
    finally:
        engine.close()
