"""Compaction-backlog sweeper Phase 0 PostgreSQL smoke test.

Gated on ``DATABASE_URL`` so a developer machine without a Postgres
instance can run the full test suite. CI / prod pipelines export
``DATABASE_URL`` to a throwaway database and run these alongside the
SQLite parity tests in ``test_backlog_sweeper_detection.py``.

Covers compaction-backlog sweeper plan v1.3 §2.5 T0.14: end-to-end
against real Postgres with the perfume-conv fixture shape. T0.15
(``EXPLAIN ANALYZE`` confirming index usage on
``idx_canonical_turns_compaction_queue``) is deferred outside this
patch -- it requires a populated catalog and is a CI-stage assertion
rather than a per-run smoke.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest


_pg_required = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping PG backlog detection smoke",
)


@pytest.fixture(scope="module")
def store():
    from virtual_context.storage.postgres import PostgresStore
    s = PostgresStore(os.environ["DATABASE_URL"])
    yield s


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_conv(
    store, *, conv_id: str, tenant_id: str = "t-bl-pg",
    phase: str = "active", lifecycle_epoch: int = 1,
) -> None:
    with store.pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                """INSERT INTO conversation_lifecycle
                   (conversation_id, generation, deleted, updated_at)
                   VALUES (%s, 0, FALSE, %s)
                   ON CONFLICT (conversation_id) DO NOTHING""",
                (conv_id, _now()),
            )
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, tenant_id, phase, lifecycle_epoch,
                    created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (conversation_id) DO UPDATE SET
                       tenant_id = EXCLUDED.tenant_id,
                       phase = EXCLUDED.phase,
                       lifecycle_epoch = EXCLUDED.lifecycle_epoch""",
                (conv_id, tenant_id, phase, lifecycle_epoch,
                 _now(), _now()),
            )


def _seed_turns(store, *, conv_id: str, count: int) -> None:
    """Insert ``count`` canonical_turns rows for ``conv_id``. The
    Postgres schema declares ``canonical_turn_id`` as ``UUID`` so we
    generate ``uuid.uuid4()`` per row rather than a synthetic
    ``ct-{conv_id}-{i}`` string (which would error during INSERT
    parsing on the gated PG smoke run).
    """
    with store.pool.connection() as conn:
        for i in range(count):
            conn.execute(
                """INSERT INTO canonical_turns
                   (canonical_turn_id, conversation_id, sort_key,
                    turn_hash, hash_version, tagged_at, compacted_at,
                    created_at, updated_at)
                   VALUES (%s, %s, %s, %s, 1, %s, NULL, %s, %s)""",
                (uuid.uuid4(), conv_id, 1000.0 + i,
                 f"h-{conv_id}-{i}", _now(), _now(), _now()),
            )


def _seed_terminal_op(
    store, *, op_id: str, conv_id: str, lifecycle_epoch: int = 1,
    completed_seconds_ago: float = 0,
) -> None:
    completed_at = _now() - timedelta(seconds=completed_seconds_ago)
    with store.pool.connection() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at,
                completed_at)
               VALUES (%s, %s, %s, 0, 7, 'starting', 'completed',
                       %s, %s, %s, %s, %s)""",
            (op_id, conv_id, lifecycle_epoch,
             completed_at, completed_at, "w-pg",
             completed_at, completed_at),
        )


@_pg_required
class TestT014_PGEndToEndPerfumeShape:
    """T0.14: end-to-end PG detection against a fixture mimicking the
    spec's ``perfume-conv`` shape -- a conversation with hundreds of
    tagged-uncompacted canonical_turns and a terminal compaction
    operation that completed well outside the grace window.
    """

    def test_detection_surfaces_perfume_shape(self, store):
        conv = f"pg-bl-{uuid.uuid4().hex[:8]}"
        tenant = f"t-{uuid.uuid4().hex[:8]}"
        _seed_conv(store, conv_id=conv, tenant_id=tenant)
        # 50 tagged-uncompacted rows -- well above the spec's
        # tentative default of 20 -- so a high-min-backlog detection
        # still surfaces it.
        _seed_turns(store, conv_id=conv, count=50)
        # Terminal compaction completed 10 minutes ago; grace_s=300
        # so it's well outside.
        _seed_terminal_op(
            store, op_id=str(uuid.uuid4()),
            conv_id=conv, completed_seconds_ago=600,
        )

        candidates = store.find_compaction_backlog_conversations(
            min_backlog_turns=20, grace_s=300.0, limit=100,
        )
        # The fixture conv must be present in the result set.
        match = [c for c in candidates if c.conversation_id == conv]
        assert match, (
            f"perfume-shape conversation {conv} not surfaced. "
            f"Got {len(candidates)} candidates."
        )
        c = match[0]
        assert c.tenant_id == tenant
        assert c.backlog_turns == 50
        assert c.lifecycle_epoch == 1
        assert c.last_terminal_compaction_at is not None

    def test_detection_respects_grace_window_pg(self, store):
        conv = f"pg-bl-grace-{uuid.uuid4().hex[:8]}"
        _seed_conv(store, conv_id=conv)
        _seed_turns(store, conv_id=conv, count=30)
        # Terminal completed 2 minutes ago; grace_s=300 -> still inside
        # the window so the conv must NOT surface.
        _seed_terminal_op(
            store, op_id=str(uuid.uuid4()),
            conv_id=conv, completed_seconds_ago=120,
        )
        candidates = store.find_compaction_backlog_conversations(
            min_backlog_turns=20, grace_s=300.0, limit=100,
        )
        assert not [c for c in candidates if c.conversation_id == conv]
