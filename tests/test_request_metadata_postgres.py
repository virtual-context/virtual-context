"""Tests for epoch-guarded request-metadata + phase helpers on the Postgres backend.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set. Mirrors
``test_request_metadata.py`` (SQLite) 1:1 so both backends stay in lockstep on
the SQL-level epoch fence. Postgres uses scalar ``GREATEST()`` where SQLite
uses ``MAX()`` and wraps the atomic drain in ``conn.transaction()`` instead of
``BEGIN IMMEDIATE``.

``conversations.conversation_id`` is ``UUID PRIMARY KEY`` in Postgres (see
postgres.py:828) so test IDs use ``uuid.uuid4()`` per-test to keep the suite
idempotent across reruns against a shared test database. psycopg3 adapts
``datetime.now(timezone.utc)`` directly to ``TIMESTAMPTZ``.
"""

from __future__ import annotations

import os
import uuid

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


# ----------------------------------------------------------------------
# update_request_metadata
# ----------------------------------------------------------------------

def test_update_request_metadata_overwrites_last_fields_pg():
    s, cid = _fresh()
    ok = s.update_request_metadata(
        conversation_id=cid, lifecycle_epoch=1,
        last_raw_payload_entries=500,
        last_ingestible_payload_entries=200,
    )
    assert ok is True
    snap = s.read_progress_snapshot(cid)
    assert snap.last_raw_payload_entries == 500
    assert snap.last_ingestible_payload_entries == 200


def test_update_request_metadata_rejects_stale_epoch_pg():
    s, cid = _fresh()
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)  # epoch=2
    ok = s.update_request_metadata(
        conversation_id=cid, lifecycle_epoch=1,  # stale
        last_raw_payload_entries=999,
        last_ingestible_payload_entries=999,
    )
    assert ok is False
    snap = s.read_progress_snapshot(cid)
    assert snap.last_raw_payload_entries == 0
    assert snap.last_ingestible_payload_entries == 0


def test_update_request_metadata_returns_false_for_unknown_conversation_pg():
    s = _store()
    ok = s.update_request_metadata(
        conversation_id=str(uuid.uuid4()), lifecycle_epoch=1,
        last_raw_payload_entries=1, last_ingestible_payload_entries=1,
    )
    assert ok is False


# ----------------------------------------------------------------------
# widen_pending_raw_payload_entries
# ----------------------------------------------------------------------

def test_widen_pending_raw_is_monotonic_pg():
    s, cid = _fresh()
    s.widen_pending_raw_payload_entries(conversation_id=cid, lifecycle_epoch=1, value=100)
    s.widen_pending_raw_payload_entries(conversation_id=cid, lifecycle_epoch=1, value=50)
    conn = s._get_conn()
    row = conn.execute(
        "SELECT pending_raw_payload_entries FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    val = row["pending_raw_payload_entries"] if isinstance(row, dict) else row[0]
    assert val == 100  # GREATEST preserved


def test_widen_pending_raw_rejects_stale_epoch_pg():
    s, cid = _fresh()
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)
    ok = s.widen_pending_raw_payload_entries(
        conversation_id=cid, lifecycle_epoch=1, value=500,
    )
    assert ok is False
    conn = s._get_conn()
    row = conn.execute(
        "SELECT pending_raw_payload_entries FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    val = row["pending_raw_payload_entries"] if isinstance(row, dict) else row[0]
    assert val == 0


def test_widen_pending_raw_returns_false_for_unknown_conversation_pg():
    s = _store()
    ok = s.widen_pending_raw_payload_entries(
        conversation_id=str(uuid.uuid4()), lifecycle_epoch=1, value=100,
    )
    assert ok is False


# ----------------------------------------------------------------------
# set_phase
# ----------------------------------------------------------------------

def test_set_phase_changes_phase_pg():
    s, cid = _fresh()
    ok = s.set_phase(conversation_id=cid, lifecycle_epoch=1, phase="ingesting")
    assert ok is True
    snap = s.read_progress_snapshot(cid)
    assert snap.phase == "ingesting"


def test_set_phase_rejected_on_epoch_mismatch_pg():
    s, cid = _fresh()
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)  # epoch=2
    ok = s.set_phase(conversation_id=cid, lifecycle_epoch=1, phase="active")
    assert ok is False
    snap = s.read_progress_snapshot(cid)
    assert snap.phase == "init"  # new lifecycle untouched


def test_set_phase_returns_false_for_unknown_conversation_pg():
    s = _store()
    ok = s.set_phase(
        conversation_id=str(uuid.uuid4()), lifecycle_epoch=1, phase="active",
    )
    assert ok is False


# ----------------------------------------------------------------------
# set_phase_and_drain_pending_raw
# ----------------------------------------------------------------------

def test_set_phase_and_drain_pending_raw_success_pg():
    s, cid = _fresh()
    ok = s.set_phase(conversation_id=cid, lifecycle_epoch=1, phase="compacting")
    assert ok is True
    s.widen_pending_raw_payload_entries(conversation_id=cid, lifecycle_epoch=1, value=42)
    drained = s.set_phase_and_drain_pending_raw(
        conversation_id=cid, lifecycle_epoch=1, new_phase="active",
    )
    assert drained == 42
    snap = s.read_progress_snapshot(cid)
    assert snap.phase == "active"
    conn = s._get_conn()
    row = conn.execute(
        "SELECT pending_raw_payload_entries FROM conversations WHERE conversation_id = %s",
        (cid,),
    ).fetchone()
    val = row["pending_raw_payload_entries"] if isinstance(row, dict) else row[0]
    assert val == 0


def test_set_phase_and_drain_returns_none_on_epoch_mismatch_pg():
    s, cid = _fresh()
    s.widen_pending_raw_payload_entries(conversation_id=cid, lifecycle_epoch=1, value=99)
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)  # epoch=2
    result = s.set_phase_and_drain_pending_raw(
        conversation_id=cid, lifecycle_epoch=1, new_phase="active",
    )
    assert result is None
    # Phase must not have changed on the new lifecycle row.
    snap = s.read_progress_snapshot(cid)
    assert snap.phase == "init"


def test_set_phase_and_drain_with_zero_pending_pg():
    s, cid = _fresh()
    drained = s.set_phase_and_drain_pending_raw(
        conversation_id=cid, lifecycle_epoch=1, new_phase="active",
    )
    assert drained == 0
