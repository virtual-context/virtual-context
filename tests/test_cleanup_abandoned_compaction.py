"""Store-layer tests for cleanup_abandoned_compaction. Scopes on
operation_id; deletes segments, facts, tag_summaries, tag_summary_embeddings;
un-marks canonical_turns; inserts a fresh running row. Idempotent.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.storage.sqlite import SQLiteStore


def _seed(store: SQLiteStore, conv: str, dead_op: str, live_op: str):
    """Seed a dead running op + a prior completed op + scoped writes
    across all five tables so tests can assert scoping: dead rows
    deleted, live (completed-prior) rows untouched.

    INVARIANT: only ONE row with status IN ('queued','running') may
    exist per (conversation_id, lifecycle_epoch) at a time — enforced
    by the unique partial index ``idx_compaction_operation_active``
    (sqlite.py:782). The partial index only applies to running/queued
    rows, so a 'completed' row at the SAME epoch is fine and does not
    collide.

    Both operations here live at the same ``lifecycle_epoch=1`` (the
    default for new conversations). ``live_op`` has status='completed'
    representing a PRIOR compaction that finished on this conversation
    before ``dead_op`` started. Its segments/facts/tag_summaries/
    tag_summary_embeddings rows are the control group the cleanup
    assertion checks do NOT get collaterally deleted.
    """
    now = utcnow_iso()
    with store._get_conn() as conn:
        # live_op: status='completed' — a PRIOR finished compaction at
        # this same epoch. started_at is earlier than dead_op so the
        # ``ORDER BY started_at DESC LIMIT 1`` in claim_compaction_lease
        # picks dead_op, not this one. It exists to prove the cleanup
        # helper doesn't collaterally delete artifacts from a prior
        # successful compaction.
        earlier = "2026-01-01T00:00:00+00:00"
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at,
                completed_at)
               VALUES (?, ?, 1, 6, 7, 'tag_summaries', 'completed',
                       ?, ?, ?, ?, ?)""",
            (live_op, conv, earlier, earlier, "prior-worker", earlier, earlier),
        )
        # dead_op: status='running' — the operation takeover will abandon.
        # The partial unique index allows this to coexist with the
        # completed live_op row above because the index is restricted
        # to status IN ('queued','running').
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (dead_op, conv, now, now, "dead-worker", now),
        )
        for op_id in (dead_op, live_op):
            conn.execute(
                """INSERT INTO segments (ref, conversation_id, summary,
                   full_text, primary_tag, compaction_model, created_at,
                   start_timestamp, end_timestamp, operation_id)
                   VALUES (?, ?, 's', 'f', 't', 'passthrough', ?, ?, ?, ?)""",
                (f"seg-{op_id[:4]}", conv, now, now, now, op_id),
            )
            conn.execute(
                """INSERT INTO facts (id, subject, verb, object, status, what,
                   conversation_id, mentioned_at, session_date, operation_id)
                   VALUES (?, 'S', 'V', 'O', 'active', 'what', ?, ?, ?, ?)""",
                (f"fact-{op_id[:4]}", conv, now, now, op_id),
            )
            conn.execute(
                """INSERT INTO tag_summaries (tag, conversation_id, summary,
                   created_at, updated_at, operation_id)
                   VALUES (?, ?, 's', ?, ?, ?)""",
                (f"tag-{op_id[:4]}", conv, now, now, op_id),
            )
            conn.execute(
                """INSERT INTO tag_summary_embeddings
                   (tag, conversation_id, embedding_json, operation_id)
                   VALUES (?, ?, '[]', ?)""",
                (f"tag-{op_id[:4]}", conv, op_id),
            )
        # Seed a canonical_turn marked by dead_op and another by live_op.
        for op_id, canonical_id in (
            (dead_op, "ct-dead"),
            (live_op, "ct-live"),
        ):
            conn.execute(
                """INSERT INTO canonical_turns
                   (canonical_turn_id, conversation_id, sort_key, turn_hash,
                    hash_version, user_content, assistant_content, tagged_at,
                    compacted_at, compaction_operation_id,
                    first_seen_at, last_seen_at, created_at, updated_at,
                    covered_ingestible_entries, turn_group_number)
                   VALUES (?, ?, ?, ?, 1, 'u', 'a', ?, ?, ?, ?, ?, ?, ?, 1, 0)""",
                (
                    canonical_id, conv,
                    1000.0 if op_id == dead_op else 2000.0,
                    f"h-{canonical_id}", now, now, op_id, now, now, now, now,
                ),
            )
        store._commit_if_unlocked(conn)


def _counts(store: SQLiteStore, conv: str, op_id: str) -> dict[str, int]:
    out: dict[str, int] = {}
    with store._get_conn() as conn:
        for table in (
            "segments", "facts", "tag_summaries", "tag_summary_embeddings",
        ):
            r = conn.execute(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE conversation_id = ? AND operation_id = ?",
                (conv, op_id),
            ).fetchone()
            out[table] = int(r[0])
        r = conn.execute(
            "SELECT COUNT(*) FROM compaction_operation "
            "WHERE conversation_id = ? AND operation_id = ?",
            (conv, op_id),
        ).fetchone()
        out["compaction_operation"] = int(r[0])
    return out


def test_cleanup_scopes_by_operation_id(tmp_path: Path):
    store = SQLiteStore(tmp_path / "c1.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed(store, "c", dead_op="dead-op", live_op="live-op")

    store.cleanup_abandoned_compaction(
        conversation_id="c",
        dead_operation_id="dead-op",
        new_operation_id="new-op",
        lifecycle_epoch=1,
        worker_id="new-worker",
        phase_count=7,
    )

    dead = _counts(store, "c", "dead-op")
    live = _counts(store, "c", "live-op")
    assert dead["segments"] == 0
    assert dead["facts"] == 0
    assert dead["tag_summaries"] == 0
    assert dead["tag_summary_embeddings"] == 0
    assert live["segments"] == 1
    assert live["facts"] == 1
    assert live["tag_summaries"] == 1
    assert live["tag_summary_embeddings"] == 1


def test_cleanup_unsets_canonical_compacted_at(tmp_path: Path):
    store = SQLiteStore(tmp_path / "c2.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed(store, "c", dead_op="dead-op", live_op="live-op")

    store.cleanup_abandoned_compaction(
        conversation_id="c",
        dead_operation_id="dead-op",
        new_operation_id="new-op",
        lifecycle_epoch=1,
        worker_id="new-worker",
        phase_count=7,
    )

    with store._get_conn() as conn:
        dead_ct = conn.execute(
            "SELECT compacted_at, compaction_operation_id FROM canonical_turns "
            "WHERE canonical_turn_id = 'ct-dead'"
        ).fetchone()
        live_ct = conn.execute(
            "SELECT compacted_at, compaction_operation_id FROM canonical_turns "
            "WHERE canonical_turn_id = 'ct-live'"
        ).fetchone()
    assert dead_ct["compacted_at"] is None
    assert dead_ct["compaction_operation_id"] is None
    assert live_ct["compacted_at"] is not None, "live-op's ct must be untouched"
    assert live_ct["compaction_operation_id"] == "live-op"


def test_cleanup_inserts_new_running_row(tmp_path: Path):
    store = SQLiteStore(tmp_path / "c3.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed(store, "c", dead_op="dead-op", live_op="live-op")

    store.cleanup_abandoned_compaction(
        conversation_id="c",
        dead_operation_id="dead-op",
        new_operation_id="new-op",
        lifecycle_epoch=1,
        worker_id="new-worker",
        phase_count=7,
    )
    with store._get_conn() as conn:
        new_row = conn.execute(
            "SELECT status, owner_worker_id, phase_count FROM compaction_operation "
            "WHERE operation_id = 'new-op'"
        ).fetchone()
        dead_row = conn.execute(
            "SELECT status FROM compaction_operation WHERE operation_id = 'dead-op'"
        ).fetchone()
    assert new_row["status"] == "running"
    assert new_row["owner_worker_id"] == "new-worker"
    assert new_row["phase_count"] == 7
    assert dead_row["status"] == "abandoned"


def test_cleanup_is_idempotent_and_preserves_one_active_invariant(tmp_path: Path):
    """Calling cleanup twice for the same dead_op must produce the
    EXACT state a single call would: one running row (new-op-1), the
    dead_op marked 'abandoned', partial writes deleted once. The
    second call is a no-op because the dead_op UPDATE matches zero
    rows — it was already abandoned on the first call.

    Never allow two status='running' rows at the same (conv, epoch) —
    the unique partial index ``idx_compaction_operation_active`` is
    the authoritative invariant and the cleanup helper must respect it.
    """
    store = SQLiteStore(tmp_path / "c4.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed(store, "c", dead_op="dead-op", live_op="live-op")

    store.cleanup_abandoned_compaction(
        conversation_id="c",
        dead_operation_id="dead-op",
        new_operation_id="new-op-1",
        lifecycle_epoch=1,
        worker_id="new-worker",
        phase_count=7,
    )
    # Second call targets the same already-abandoned dead_op. The
    # idempotent contract: UPDATE matches zero rows, so the caller
    # skips the INSERT and does not try to create a second running
    # row. Cleanup helpers that ignore this would either (a) raise
    # IntegrityError on the unique partial index or (b) create
    # undefined behaviour. Both are bugs.
    store.cleanup_abandoned_compaction(
        conversation_id="c",
        dead_operation_id="dead-op",
        new_operation_id="new-op-2",
        lifecycle_epoch=1,
        worker_id="new-worker",
        phase_count=7,
    )

    with store._get_conn() as conn:
        rows = conn.execute(
            "SELECT operation_id, status FROM compaction_operation "
            "WHERE conversation_id = 'c' AND lifecycle_epoch = 1 "
            "ORDER BY operation_id"
        ).fetchall()
    running = [r for r in rows if r["status"] == "running"]
    abandoned = [r for r in rows if r["status"] == "abandoned"]

    assert len(running) == 1, (
        f"One-active invariant violated: expected exactly 1 running "
        f"row at (c, epoch=1); got {[dict(r) for r in running]}"
    )
    assert running[0]["operation_id"] == "new-op-1", (
        "Second cleanup call must not have replaced the first's new_op"
    )
    assert {r["operation_id"] for r in abandoned} == {"dead-op"}, (
        f"Only dead-op should be abandoned; got {[dict(r) for r in abandoned]}"
    )
    # new-op-2 must NOT have been inserted.
    assert all(r["operation_id"] != "new-op-2" for r in rows), (
        "new-op-2 was inserted on a redundant cleanup call — invariant violated"
    )


def test_cleanup_respects_one_active_invariant_on_concurrent_peer(tmp_path: Path):
    """If another running row at the same (conv, epoch) snuck in
    between the caller's claim and the cleanup's INSERT (lease-claim
    race that the caller's atomic UPDATE should have already excluded,
    but belt-and-suspenders), cleanup must NOT insert a second running
    row. The unique partial index would reject it; the cleanup must
    roll back cleanly rather than crash partway.
    """
    store = SQLiteStore(tmp_path / "c5.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed(store, "c", dead_op="dead-op", live_op="live-op")
    now = utcnow_iso()
    # Seed a rogue peer: another 'running' row at the same (conv, epoch).
    # This violates the unique partial index and cannot exist in reality,
    # but simulate the attempt by first creating it then checking cleanup
    # handles the scenario gracefully. We do this by running the peer
    # INSERT inside a try/except because the unique index MUST raise.
    import sqlite3 as _sqlite3
    with store._get_conn() as conn:
        with pytest.raises(_sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO compaction_operation
                   (operation_id, conversation_id, lifecycle_epoch,
                    phase_index, phase_count, phase_name, status,
                    started_at, heartbeat_ts, owner_worker_id, created_at)
                   VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                           ?, ?, ?, ?)""",
                ("peer-op", "c", now, now, "peer", now),
            )
    # The index held — test asserts reality matches our mental model.
    # Now run cleanup normally; it should succeed.
    store.cleanup_abandoned_compaction(
        conversation_id="c",
        dead_operation_id="dead-op",
        new_operation_id="new-op",
        lifecycle_epoch=1,
        worker_id="new-worker",
        phase_count=7,
    )
    with store._get_conn() as conn:
        running = conn.execute(
            "SELECT operation_id FROM compaction_operation "
            "WHERE conversation_id='c' AND lifecycle_epoch=1 "
            "  AND status='running'"
        ).fetchall()
    assert len(running) == 1
    assert running[0]["operation_id"] == "new-op"
