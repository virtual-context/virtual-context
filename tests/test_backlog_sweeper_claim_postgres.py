"""Compaction-backlog sweeper Phase 1 PostgreSQL smoke test.

Gated on ``DATABASE_URL`` per the existing convention. Covers spec
plan §3.2 T1.12 end-to-end claim flow against real Postgres. The
T1.13-T1.15 race/recovery scenarios require multi-connection
concurrency and are deferred outside this patch.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from tests.pg_helpers import pg_dsn


_pg_required = pytest.mark.skipif(
    not pg_dsn(),
    reason="DATABASE_URL not set; skipping PG backlog claim smoke",
)


@pytest.fixture(scope="module")
def store():
    from virtual_context.storage.postgres import PostgresStore
    s = PostgresStore(pg_dsn())
    yield s


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_conv(store, conv_id: str, *, tenant_id: str = "t-bl-claim-pg",
               lifecycle_epoch: int = 1) -> None:
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
                   VALUES (%s, %s, 'active', %s, %s, %s)
                   ON CONFLICT (conversation_id) DO UPDATE SET
                       tenant_id = EXCLUDED.tenant_id,
                       phase = EXCLUDED.phase,
                       lifecycle_epoch = EXCLUDED.lifecycle_epoch""",
                (conv_id, tenant_id, lifecycle_epoch, _now(), _now()),
            )


def _seed_turns(store, conv_id: str, count: int) -> None:
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


@_pg_required
class TestT112_PGClaimEndToEnd:
    def test_claim_happy_path_pg(self, store):
        from virtual_context.types import BacklogCandidate
        conv = f"pg-claim-{uuid.uuid4().hex[:8]}"
        tenant = f"t-{uuid.uuid4().hex[:8]}"
        _seed_conv(store, conv, tenant_id=tenant)
        _seed_turns(store, conv, count=30)

        candidate = BacklogCandidate(
            conversation_id=conv, tenant_id=tenant,
            lifecycle_epoch=1, backlog_turns=30,
            last_terminal_compaction_at=None,
        )
        op_id = str(uuid.uuid4())
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=op_id,
            owner_worker_id="w-pg-sweeper",
            phase_count=7,
            min_backlog_turns=20,
            grace_s=300.0,
        )
        assert ok is True
        # Phase flipped + running op exists.
        with store.pool.connection() as conn:
            row = conn.execute(
                "SELECT phase FROM conversations "
                "WHERE conversation_id = %s",
                (conv,),
            ).fetchone()
            assert row["phase"] == "compacting"
            row = conn.execute(
                """SELECT COUNT(*) AS n FROM compaction_operation
                    WHERE conversation_id = %s
                      AND status = 'running'""",
                (conv,),
            ).fetchone()
            assert int(row["n"]) == 1

    def test_epoch_mismatch_pg(self, store):
        from virtual_context.types import BacklogCandidate
        conv = f"pg-claim-epoch-{uuid.uuid4().hex[:8]}"
        _seed_conv(store, conv, lifecycle_epoch=3)
        _seed_turns(store, conv, count=30)

        candidate = BacklogCandidate(
            conversation_id=conv, tenant_id="t-bl-claim-pg",
            lifecycle_epoch=1, backlog_turns=30,
            last_terminal_compaction_at=None,
        )
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=str(uuid.uuid4()),
            owner_worker_id="w-pg-sweeper",
            phase_count=7,
            min_backlog_turns=20,
            grace_s=300.0,
        )
        assert ok is False
