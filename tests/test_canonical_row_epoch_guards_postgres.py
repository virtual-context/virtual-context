"""Tests for epoch-guarded ``iter_untagged_canonical_rows`` +
``mark_canonical_row_tagged`` on the Postgres backend.

Skipped unless ``VC_TEST_POSTGRES_URL`` is set. Mirrors
``test_canonical_row_epoch_guards.py`` (SQLite) 1:1 so both backends stay in
lockstep on the SQL-level epoch fence. ``conversations.conversation_id`` is
``TEXT PRIMARY KEY`` (aligned with ``canonical_turns.conversation_id``), and
``canonical_turns.canonical_turn_id`` is ``UUID``, so test conversation IDs
use ``str(uuid.uuid4())`` and canonical IDs use ``uuid.uuid4()``.
``source_batch_id`` is also UUID, so the seed helper generates a fresh UUID
per row. psycopg3 adapts ``datetime`` directly so we pass
``datetime.now(timezone.utc)`` rather than ISO strings for the
``TIMESTAMPTZ`` lifecycle columns on ``canonical_turns`` — except that
``tagged_at`` / ``first_seen_at`` / ``last_seen_at`` / ``created_at`` /
``updated_at`` on ``canonical_turns`` are ``TEXT`` in the Postgres schema
(not TIMESTAMPTZ), so ISO strings are the right shape for those.
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


def _fresh(conv_id: str | None = None):
    """Create a fresh conversation with a unique UUID and return (store, conv_id)."""
    conv_id = conv_id or str(uuid.uuid4())
    s = _store()
    s.upsert_conversation(tenant_id="t", conversation_id=conv_id)
    return s, conv_id


def _seed_canonical_row(
    s,
    *,
    conv_id: str,
    canonical_id: str,
    sort_key: float,
    tagged: bool = False,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = s._get_conn()
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
            canonical_id,
            conv_id,
            f"h_{canonical_id}",
            sort_key,
            uuid.uuid4(),  # source_batch_id (UUID)
            now_iso,
            now_iso,
            now_iso if tagged else None,
            now_iso,
            now_iso,
        ),
    )


def test_iter_untagged_returns_untagged_in_sort_key_order_pg():
    s, cid = _fresh()
    ids = [str(uuid.uuid4()) for _ in range(3)]
    _seed_canonical_row(s, conv_id=cid, canonical_id=ids[0], sort_key=3000.0)
    _seed_canonical_row(s, conv_id=cid, canonical_id=ids[1], sort_key=1000.0)
    _seed_canonical_row(s, conv_id=cid, canonical_id=ids[2], sort_key=2000.0)
    rows = s.iter_untagged_canonical_rows(
        conversation_id=cid, expected_lifecycle_epoch=1, batch_size=10,
    )
    ids_in_order = [r.canonical_turn_id for r in rows]
    assert ids_in_order == [ids[1], ids[2], ids[0]]  # sorted by sort_key


def test_iter_untagged_respects_batch_size_pg():
    s, cid = _fresh()
    for i in range(5):
        _seed_canonical_row(
            s, conv_id=cid, canonical_id=str(uuid.uuid4()),
            sort_key=float(i * 1000),
        )
    rows = s.iter_untagged_canonical_rows(
        conversation_id=cid, expected_lifecycle_epoch=1, batch_size=2,
    )
    assert len(rows) == 2


def test_iter_untagged_excludes_tagged_rows_pg():
    s, cid = _fresh()
    t_tagged = str(uuid.uuid4())
    t_pending = str(uuid.uuid4())
    _seed_canonical_row(s, conv_id=cid, canonical_id=t_tagged, sort_key=1000.0, tagged=True)
    _seed_canonical_row(s, conv_id=cid, canonical_id=t_pending, sort_key=2000.0, tagged=False)
    rows = s.iter_untagged_canonical_rows(
        conversation_id=cid, expected_lifecycle_epoch=1, batch_size=10,
    )
    assert [r.canonical_turn_id for r in rows] == [t_pending]


def test_iter_untagged_returns_empty_on_epoch_mismatch_pg():
    """SQL-level epoch guard. Stale-epoch caller sees no rows even if rows exist."""
    s, cid = _fresh()
    old_id = str(uuid.uuid4())
    _seed_canonical_row(s, conv_id=cid, canonical_id=old_id, sort_key=1000.0)
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)  # epoch=2
    # Seed a new-lifecycle row.
    new_id = str(uuid.uuid4())
    _seed_canonical_row(s, conv_id=cid, canonical_id=new_id, sort_key=2000.0)
    # Stale caller: epoch=1.
    rows = s.iter_untagged_canonical_rows(
        conversation_id=cid, expected_lifecycle_epoch=1, batch_size=10,
    )
    assert rows == []  # sees nothing
    # Current caller: epoch=2 — sees both (old from old lifecycle not purged, plus new).
    rows = s.iter_untagged_canonical_rows(
        conversation_id=cid, expected_lifecycle_epoch=2, batch_size=10,
    )
    assert len(rows) == 2


def test_mark_canonical_row_tagged_sets_timestamp_pg():
    s, cid = _fresh()
    t_pending = str(uuid.uuid4())
    _seed_canonical_row(s, conv_id=cid, canonical_id=t_pending, sort_key=1000.0)
    ok = s.mark_canonical_row_tagged(
        canonical_turn_id=t_pending, conversation_id=cid,
        expected_lifecycle_epoch=1,
    )
    assert ok is True
    conn = s._get_conn()
    row = conn.execute(
        "SELECT tagged_at FROM canonical_turns WHERE canonical_turn_id = %s",
        (t_pending,),
    ).fetchone()
    assert row["tagged_at"] is not None


def test_mark_canonical_row_tagged_rejects_on_epoch_mismatch_pg():
    """Proves SQL-level epoch guard on the mark boundary."""
    s, cid = _fresh()
    t0 = str(uuid.uuid4())
    _seed_canonical_row(s, conv_id=cid, canonical_id=t0, sort_key=1000.0)
    s.mark_conversation_deleted(cid)
    s.increment_lifecycle_epoch_on_resurrect(cid)  # epoch=2
    # Stale caller tries to mark a row that still exists.
    ok = s.mark_canonical_row_tagged(
        canonical_turn_id=t0, conversation_id=cid,
        expected_lifecycle_epoch=1,
    )
    assert ok is False
    conn = s._get_conn()
    row = conn.execute(
        "SELECT tagged_at FROM canonical_turns WHERE canonical_turn_id = %s",
        (t0,),
    ).fetchone()
    assert row["tagged_at"] is None  # untouched


def test_mark_canonical_row_tagged_is_idempotent_for_already_tagged_pg():
    """Calling mark_canonical_row_tagged a second time on an already-tagged row returns False (no-op)."""
    s, cid = _fresh()
    t0 = str(uuid.uuid4())
    _seed_canonical_row(s, conv_id=cid, canonical_id=t0, sort_key=1000.0, tagged=True)
    ok = s.mark_canonical_row_tagged(
        canonical_turn_id=t0, conversation_id=cid,
        expected_lifecycle_epoch=1,
    )
    assert ok is False  # already tagged, WHERE tagged_at IS NULL excludes it


def test_mark_canonical_row_tagged_wrong_conversation_id_returns_false_pg():
    """Belt-and-suspenders: mismatched conversation_id doesn't accidentally mark another conversation's row."""
    s1, c1 = _fresh()
    _, c2 = _fresh()
    t0 = str(uuid.uuid4())
    _seed_canonical_row(s1, conv_id=c1, canonical_id=t0, sort_key=1000.0)
    ok = s1.mark_canonical_row_tagged(
        canonical_turn_id=t0, conversation_id=c2,  # wrong conv
        expected_lifecycle_epoch=1,
    )
    assert ok is False
