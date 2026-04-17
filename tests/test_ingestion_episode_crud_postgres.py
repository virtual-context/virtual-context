"""Tests for the Postgres ingestion_episode CRUD API.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors
the SQLite tests in ``test_ingestion_episode_crud.py`` so both backends stay
in lockstep on the ownership-free widening upsert + epoch-scoped
lease/heartbeat/complete semantics. The correlated subquery in
``refresh_ingestion_heartbeat`` / ``complete_ingestion_episode`` pins the
epoch filter to the authoritative ``conversations.lifecycle_epoch`` so a
stale thread whose conversation was resurrected to a newer epoch is
rejected at SQL level.

Note: ``conversations.conversation_id`` is ``UUID PRIMARY KEY`` in Postgres
(see postgres.py:828) and ``ingestion_episode.episode_id`` /
``compaction_operation.operation_id`` are also UUID, so test IDs use
``uuid.uuid4()`` rather than free-form strings. UUIDs are generated
per-test to keep the suite idempotent across reruns against a shared test
database. psycopg3 adapts ``datetime.now(timezone.utc)`` directly to
``TIMESTAMPTZ`` so no ISO-string conversion is needed when seeding rows.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def _store():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    return PostgresStore(PG_URL)


def _fresh(conv_id: str | None = None):
    """Create a fresh conversation with a unique UUID and return (store, conv_id)."""
    conv_id = conv_id or str(uuid.uuid4())
    s = _store()
    s.upsert_conversation(tenant_id="t", conversation_id=conv_id)
    return s, conv_id


def _seed_canonical(s, *, conv_id: str, count: int, tagged: int) -> None:
    """Insert ``count`` canonical rows; first ``tagged`` of them have tagged_at set."""
    now = datetime.now(timezone.utc)
    conn = s._get_conn()
    for i in range(count):
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES (%s, %s, %s, 1, 'u','a','u_raw','a_raw', %s, %s, %s, %s, 1, %s, %s, %s)
            """,
            (
                uuid.uuid4(),
                conv_id,
                f"h{i}_{conv_id[:8]}",
                float((i + 1) * 1000),
                uuid.uuid4(),
                now,
                now,
                now if i < tagged else None,
                now,
                now,
            ),
        )


# ----------------------------------------------------------------------
# upsert_ingestion_episode
# ----------------------------------------------------------------------

def test_upsert_creates_row_with_initial_worker_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode is not None
    assert snap.active_episode.owner_worker_id == "w1"
    assert snap.active_episode.raw_payload_entries == 100


def test_upsert_on_conflict_widens_raw_via_greatest_and_preserves_owner_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        raw_payload_entries=50,
    )  # smaller, different worker
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode.raw_payload_entries == 100  # GREATEST, not 50
    assert snap.active_episode.owner_worker_id == "w1"  # unchanged


def test_upsert_widens_to_larger_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        raw_payload_entries=500,
    )
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode.raw_payload_entries == 500  # widened
    assert snap.active_episode.owner_worker_id == "w1"  # still w1


# ----------------------------------------------------------------------
# claim_ingestion_lease
# ----------------------------------------------------------------------

def test_claim_succeeds_when_caller_already_owns_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    assert s.claim_ingestion_lease(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        lease_ttl_s=30.0,
    ) is True


def test_claim_fails_when_other_worker_holds_fresh_lease_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    assert s.claim_ingestion_lease(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        lease_ttl_s=30.0,
    ) is False


def test_claim_succeeds_when_other_worker_lease_is_stale_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    # Force stale heartbeat.
    stale = datetime(2000, 1, 1, tzinfo=timezone.utc)
    conn = s._get_conn()
    conn.execute(
        "UPDATE ingestion_episode SET heartbeat_ts = %s WHERE conversation_id = %s",
        (stale, cid),
    )
    assert s.claim_ingestion_lease(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
        lease_ttl_s=30.0,
    ) is True
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode.owner_worker_id == "w2"


def test_claim_fails_on_different_epoch_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    # Attempt to claim at epoch=2 — should fail (no running row at epoch 2).
    assert s.claim_ingestion_lease(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        lease_ttl_s=30.0,
    ) is False


# ----------------------------------------------------------------------
# refresh_ingestion_heartbeat
# ----------------------------------------------------------------------

def test_refresh_heartbeat_succeeds_for_owner_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    assert s.refresh_ingestion_heartbeat(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is True


def test_refresh_heartbeat_fails_for_non_owner_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    assert s.refresh_ingestion_heartbeat(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
    ) is False


def test_refresh_heartbeat_fails_on_stale_epoch_pg():
    """SQL-level epoch guard: stale caller cannot refresh new lifecycle."""
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    # Simulate a resurrect → new episode at epoch 2 owned by same worker.
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        raw_payload_entries=50,
    )
    # Stale thread thinks epoch=1, tries to refresh. Must fail.
    assert s.refresh_ingestion_heartbeat(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is False


# ----------------------------------------------------------------------
# complete_ingestion_episode
# ----------------------------------------------------------------------

def test_complete_fails_when_untagged_rows_remain_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    _seed_canonical(s, conv_id=cid, count=3, tagged=1)  # 2 untagged
    assert s.complete_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is False
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode is not None  # still running


def test_complete_succeeds_when_all_rows_tagged_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    _seed_canonical(s, conv_id=cid, count=3, tagged=3)  # all tagged
    assert s.complete_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is True
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode is None  # completed, no longer active


def test_complete_succeeds_when_no_canonical_rows_pg():
    """Empty conversation — NOT EXISTS (no untagged rows) is vacuously true."""
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=0,
    )
    assert s.complete_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is True


def test_complete_fails_for_non_owner_pg():
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    assert s.complete_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w2",
    ) is False


def test_complete_fails_on_stale_epoch_pg():
    """Stale caller carrying epoch=1 cannot complete an epoch=2 episode."""
    s, cid = _fresh()
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
        raw_payload_entries=100,
    )
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)
    s.upsert_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=2, worker_id="w1",
        raw_payload_entries=50,
    )
    assert s.complete_ingestion_episode(
        conversation_id=cid, lifecycle_epoch=1, worker_id="w1",
    ) is False
