"""Tests for the Postgres ``read_progress_snapshot`` API.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set in the environment. Mirrors
the SQLite tests in ``test_progress_snapshot.py`` so both backends stay in
lockstep on the DB-derived progress snapshot semantics: the conversations
row header, the ``SUM(covered_ingestible_entries)`` numerator/denominator
pair over ``canonical_turns``, and the point lookups into
``ingestion_episode`` / ``compaction_operation``.

Note: ``conversations.conversation_id`` is ``UUID PRIMARY KEY`` in Postgres
(see postgres.py:823), and ``ingestion_episode.episode_id`` /
``compaction_operation.operation_id`` are also UUID, so test IDs use
``uuid.uuid4()`` rather than free-form strings. UUIDs are generated
per-test to keep the suite idempotent across reruns against a shared test
database.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

PG_URL = os.environ.get("VC_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="VC_TEST_POSTGRES_URL not set")


def _store():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    return PostgresStore(PG_URL)


def _cid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_read_progress_snapshot_raises_keyerror_for_unknown_pg():
    s = _store()
    with pytest.raises(KeyError):
        s.read_progress_snapshot(_cid())


def test_read_progress_snapshot_empty_conversation_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    snap = s.read_progress_snapshot(cid)
    assert snap.conversation_id == cid
    assert snap.lifecycle_epoch == 1
    assert snap.phase == "init"
    assert snap.total_ingestible == 0
    assert snap.done_ingestible == 0
    assert snap.last_raw_payload_entries == 0
    assert snap.last_ingestible_payload_entries == 0
    assert snap.active_episode is None
    assert snap.active_compaction is None


def test_read_progress_snapshot_derives_total_and_done_from_canonical_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    # Insert 3 canonical rows: 2 tagged, 1 untagged. All have covered_ingestible_entries=1.
    now = _now()
    conn = s._get_conn()
    for i, tagged in enumerate([True, True, False]):
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
                str(uuid.uuid4()),
                cid,
                f"h{i}",
                float((i + 1) * 1000),
                str(uuid.uuid4()),
                now,
                now,
                now if tagged else None,
                now,
                now,
            ),
        )
    snap = s.read_progress_snapshot(cid)
    assert snap.total_ingestible == 3
    assert snap.done_ingestible == 2


def test_read_progress_snapshot_includes_active_episode_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    now = _now()
    ep_id = str(uuid.uuid4())
    conn = s._get_conn()
    conn.execute(
        """
        INSERT INTO ingestion_episode (
            episode_id, conversation_id, lifecycle_epoch,
            raw_payload_entries, started_at, status, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 500, %s, 'running', 'workerA', %s)
        """,
        (ep_id, cid, now, now),
    )
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode is not None
    assert snap.active_episode.episode_id == ep_id
    assert snap.active_episode.raw_payload_entries == 500
    assert snap.active_episode.owner_worker_id == "workerA"


def test_read_progress_snapshot_skips_completed_episode_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    now = _now()
    conn = s._get_conn()
    conn.execute(
        """
        INSERT INTO ingestion_episode (
            episode_id, conversation_id, lifecycle_epoch,
            raw_payload_entries, started_at, completed_at,
            status, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 500, %s, %s, 'completed', 'w', %s)
        """,
        (str(uuid.uuid4()), cid, now, now, now),
    )
    snap = s.read_progress_snapshot(cid)
    assert snap.active_episode is None


def test_read_progress_snapshot_includes_active_compaction_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    now = _now()
    op_id = str(uuid.uuid4())
    conn = s._get_conn()
    conn.execute(
        """
        INSERT INTO compaction_operation (
            operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 2, 5, 'summarizing', 'running', %s, 'w', %s)
        """,
        (op_id, cid, now, now),
    )
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None
    assert snap.active_compaction.operation_id == op_id
    assert snap.active_compaction.phase_name == "summarizing"
    assert snap.active_compaction.phase_index == 2
    assert snap.active_compaction.phase_count == 5
    assert snap.active_compaction.status == "running"


def test_read_progress_snapshot_queued_compaction_is_also_active_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    now = _now()
    conn = s._get_conn()
    conn.execute(
        """
        INSERT INTO compaction_operation (
            operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, owner_worker_id, heartbeat_ts
        ) VALUES (%s, %s, 1, 0, 5, 'init', 'queued', %s, 'w', %s)
        """,
        (str(uuid.uuid4()), cid, now, now),
    )
    snap = s.read_progress_snapshot(cid)
    assert snap.active_compaction is not None
    assert snap.active_compaction.status == "queued"


def test_read_progress_snapshot_reads_request_metadata_pg():
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    conn = s._get_conn()
    conn.execute(
        """
        UPDATE conversations
           SET last_raw_payload_entries = 1000,
               last_ingestible_payload_entries = 400
         WHERE conversation_id = %s
        """,
        (cid,),
    )
    snap = s.read_progress_snapshot(cid)
    assert snap.last_raw_payload_entries == 1000
    assert snap.last_ingestible_payload_entries == 400


def test_read_progress_snapshot_returns_frozen_dataclass_pg():
    from dataclasses import FrozenInstanceError
    s = _store()
    cid = _cid()
    s.upsert_conversation(tenant_id="t", conversation_id=cid)
    snap = s.read_progress_snapshot(cid)
    with pytest.raises(FrozenInstanceError):
        snap.phase = "ingesting"
