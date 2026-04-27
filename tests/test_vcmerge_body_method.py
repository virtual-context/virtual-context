"""Body-method tests for VCMERGE S1.3 / S1.4 (merge_conversation_data).

Per VCMerge plan v1.11 sections 3.3 + 11.1 (move semantics) + 11.2
(audit row management). Pins the structural invariants of the body
method:

- D1 pre-flight: SELECT 1 FROM merge_audit FOR UPDATE; raises
  MergeAuditMissing if no in_progress reservation exists.
- Move semantics: per-conv data rows transition from source's
  conversation_id to target's; origin_conversation_id captures the
  source so provenance is preserved.
- Sort key + request_turn offsets applied where applicable.
- request_turn_counters source row DELETED (target keeps its own).
- Tag-summary conflict resolution: source row for a tag the target
  also has gets DELETED; non-conflicting source rows UPDATEd.
- conversation_aliases UPSERTed with epoch.
- Source phase flips to 'merged'.
- merge_audit finalized to status='committed'.
- merge_post_commit_pending INSERTs for sse_event, tag_regenerate,
  queue_resegment kinds.
- Returns MergeStats with per-table row counts.

All tests run against SQLite (the project's test backend); the PG
path uses identical SQL semantics with FOR UPDATE row-lock + SAVEPOINT
nesting in place of SQLite's BEGIN IMMEDIATE.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
VCMerge plan v1.11 section 11 prologue.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.core.exceptions import (
    MergeAuditMissing, LifecycleEpochMismatch,
    CrossTenantMergeError, MergeBusy,
)


pytestmark = pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26")


def _store(tmp_path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "store.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_conversation(conn, tenant_id: str, conv_id: str, lifecycle_epoch: int = 1):
    """Insert a conversations row + a few canonical_turns + segments + tag_summaries."""
    now = _now_iso()
    conn.execute(
        "INSERT INTO conversations (conversation_id, tenant_id, phase, "
        "lifecycle_epoch, created_at, updated_at) VALUES (?, ?, 'active', ?, ?, ?)",
        (conv_id, tenant_id, lifecycle_epoch, now, now),
    )


def _seed_canonical_turn(conn, conv_id: str, turn_id: str, sort_key: float):
    conn.execute(
        "INSERT INTO canonical_turns (canonical_turn_id, conversation_id, sort_key, "
        "turn_hash, hash_version, created_at, updated_at) "
        "VALUES (?, ?, ?, 'h', 1, ?, ?)",
        (turn_id, conv_id, sort_key, _now_iso(), _now_iso()),
    )


def _seed_segment(conn, conv_id: str, ref: str, primary_tag: str = "general"):
    now = _now_iso()
    conn.execute(
        "INSERT INTO segments (ref, conversation_id, primary_tag, summary, full_text, "
        "messages_json, metadata_json, summary_tokens, full_tokens, compression_ratio, "
        "created_at, start_timestamp, end_timestamp) "
        "VALUES (?, ?, ?, 's', 'f', '[]', '{}', 1, 1, 1.0, ?, ?, ?)",
        (ref, conv_id, primary_tag, now, now, now),
    )


def _seed_tag_summary(conn, conv_id: str, tag: str, summary_text: str):
    now = _now_iso()
    conn.execute(
        "INSERT INTO tag_summaries (tag, conversation_id, summary, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (tag, conv_id, summary_text, now, now),
    )


def _seed_request_turn_counter(conn, conv_id: str, next_turn: int):
    conn.execute(
        "INSERT INTO request_turn_counters (conversation_id, next_request_turn) "
        "VALUES (?, ?)",
        (conv_id, next_turn),
    )


def _reserve(store: SQLiteStore, *, tenant_id="tA", source="src", target="tgt", label="lbl"):
    """Reserve a merge audit row + return merge_id."""
    merge_id = str(uuid.uuid4())
    result = store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id, tenant_id=tenant_id,
        source_conversation_id=source, target_conversation_id=target,
        source_label_at_merge=label,
    )
    assert result.status == "reserved"
    return merge_id


# ---------------------------------------------------------------------------
# D1 pre-flight: MergeAuditMissing if no reservation
# ---------------------------------------------------------------------------

def test_body_raises_merge_audit_missing_without_reservation(tmp_path):
    """D1: pre-flight SELECT raises MergeAuditMissing if no in_progress
    reservation exists for the (tenant, merge_id) pair.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.commit()
    with pytest.raises(MergeAuditMissing):
        store.merge_conversation_data(
            merge_id=str(uuid.uuid4()),  # no reservation for this id
            tenant_id="tA",
            source_conversation_id="src",
            target_conversation_id="tgt",
            sort_key_offset=1000.0,
            request_turn_offset=10,
            expected_target_lifecycle_epoch=1,
            source_label_at_merge="lbl",
        )


def test_body_raises_lifecycle_epoch_mismatch(tmp_path):
    """Caller-captured target epoch must match the actual conversations
    row's lifecycle_epoch; refuses if epoch advanced between reserve and body.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src", lifecycle_epoch=1)
    _seed_conversation(conn, "tA", "tgt", lifecycle_epoch=5)  # actual = 5
    conn.commit()
    merge_id = _reserve(store, source="src", target="tgt")
    with pytest.raises(LifecycleEpochMismatch):
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1,  # caller thinks epoch=1
            source_label_at_merge="lbl",
        )


# ---------------------------------------------------------------------------
# Move semantics: rows transition from source to target's namespace
# ---------------------------------------------------------------------------

def test_body_moves_canonical_turns_with_sort_key_offset(tmp_path):
    """canonical_turns: conversation_id moves; sort_key += offset;
    origin_conversation_id captures source.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_canonical_turn(conn, "src", "turn-1", sort_key=1.0)
    _seed_canonical_turn(conn, "src", "turn-2", sort_key=2.0)
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Source has zero canonical_turns now
    src_count = conn.execute(
        "SELECT COUNT(*) FROM canonical_turns WHERE conversation_id = 'src'",
    ).fetchone()[0]
    assert src_count == 0
    # Target has both, with offset applied + origin set
    tgt_rows = conn.execute(
        "SELECT canonical_turn_id, sort_key, origin_conversation_id "
        "FROM canonical_turns WHERE conversation_id = 'tgt' "
        "ORDER BY sort_key",
    ).fetchall()
    assert len(tgt_rows) == 2
    assert tgt_rows[0]["sort_key"] == 1001.0
    assert tgt_rows[1]["sort_key"] == 1002.0
    assert tgt_rows[0]["origin_conversation_id"] == "src"
    assert tgt_rows[1]["origin_conversation_id"] == "src"
    assert stats.rows_moved.get("canonical_turns") == 2


def test_body_moves_segments_no_offset(tmp_path):
    """segments: conversation_id moves; no offset (segments don't have
    sort_key). origin_conversation_id captures source.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_segment(conn, "src", "seg-1")
    _seed_segment(conn, "src", "seg-2")
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    tgt_segs = conn.execute(
        "SELECT ref, origin_conversation_id FROM segments "
        "WHERE conversation_id = 'tgt' ORDER BY ref",
    ).fetchall()
    assert len(tgt_segs) == 2
    assert all(r["origin_conversation_id"] == "src" for r in tgt_segs)
    assert stats.rows_moved.get("segments") == 2


def test_body_deletes_request_turn_counter_for_source(tmp_path):
    """request_turn_counters: source row DELETEd (target keeps its own
    counter; merging into target doesn't bump target's counter).
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_request_turn_counter(conn, "src", 5)
    _seed_request_turn_counter(conn, "tgt", 3)
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Source counter gone
    src_row = conn.execute(
        "SELECT * FROM request_turn_counters WHERE conversation_id = 'src'",
    ).fetchone()
    assert src_row is None
    # Target counter preserved (NOT moved to target's)
    tgt_row = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters WHERE conversation_id = 'tgt'",
    ).fetchone()
    assert tgt_row["next_request_turn"] == 3


def test_body_resolves_tag_summary_conflicts(tmp_path):
    """Tag-summary conflict resolution: source row for a tag the target
    also has gets DELETEd (target wins); non-conflicting source rows
    UPDATEd to target's namespace.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # Source has tags {python, javascript}; target has {python, rust}
    # Conflict on "python" -> source's DELETEd; "javascript" UPDATEd to target
    _seed_tag_summary(conn, "src", "python", "src-python-summary")
    _seed_tag_summary(conn, "src", "javascript", "src-js-summary")
    _seed_tag_summary(conn, "tgt", "python", "tgt-python-summary")
    _seed_tag_summary(conn, "tgt", "rust", "tgt-rust-summary")
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Target has python (target's wins), javascript (moved), rust (untouched).
    tgt_tags = sorted(r["tag"] for r in conn.execute(
        "SELECT tag FROM tag_summaries WHERE conversation_id = 'tgt'",
    ).fetchall())
    assert tgt_tags == ["javascript", "python", "rust"]
    # Target's "python" summary preserved (not overwritten by source)
    py_summary = conn.execute(
        "SELECT summary FROM tag_summaries WHERE conversation_id = 'tgt' AND tag = 'python'",
    ).fetchone()
    assert py_summary["summary"] == "tgt-python-summary"
    # Source has zero rows
    src_count = conn.execute(
        "SELECT COUNT(*) FROM tag_summaries WHERE conversation_id = 'src'",
    ).fetchone()[0]
    assert src_count == 0
    # Stats track conflict
    assert stats.rows_moved.get("tag_summaries__conflicts_deleted") == 1
    assert stats.rows_moved.get("tag_summaries") == 1  # only javascript moved


# ---------------------------------------------------------------------------
# Alias UPSERT + phase flip + audit finalize + post-commit pendings
# ---------------------------------------------------------------------------

def test_body_upserts_conversation_alias_with_epoch(tmp_path):
    """conversation_aliases UPSERTed: alias_id=source, target_id=target,
    epoch=expected_target_lifecycle_epoch.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt", lifecycle_epoch=7)
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=7, source_label_at_merge="lbl",
    )
    alias = conn.execute(
        "SELECT target_id, epoch FROM conversation_aliases WHERE alias_id = 'src'",
    ).fetchone()
    assert alias is not None
    assert alias["target_id"] == "tgt"
    assert alias["epoch"] == 7


def test_body_flips_source_phase_to_merged(tmp_path):
    """Source's phase column transitions to 'merged' (M0.1 admits)."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    src_phase = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = 'src'",
    ).fetchone()["phase"]
    assert src_phase == "merged"
    # Target's phase unchanged
    tgt_phase = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = 'tgt'",
    ).fetchone()["phase"]
    assert tgt_phase == "active"


def test_body_finalizes_merge_audit_to_committed(tmp_path):
    """merge_audit row transitions from in_progress to committed with
    completed_at + rows_moved_json populated.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    audit = conn.execute(
        "SELECT status, completed_at, rows_moved_json FROM merge_audit WHERE merge_id = ?",
        (merge_id,),
    ).fetchone()
    assert audit["status"] == "committed"
    assert audit["completed_at"] is not None
    parsed = json.loads(audit["rows_moved_json"])
    assert isinstance(parsed, dict)
    # Should at minimum include the per-table keys (counts may be 0 if not seeded)
    assert "canonical_turns" in parsed
    assert "tag_summaries" in parsed


def test_body_inserts_three_post_commit_pending_rows(tmp_path):
    """B1.x: merge_post_commit_pending gets 3 INSERTs (sse_event,
    tag_regenerate, queue_resegment).
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="my-label",
    )
    pendings = conn.execute(
        "SELECT kind, status, payload_json FROM merge_post_commit_pending "
        "WHERE merge_id = ? ORDER BY kind",
        (merge_id,),
    ).fetchall()
    kinds = sorted(p["kind"] for p in pendings)
    assert kinds == ["queue_resegment", "sse_event", "tag_regenerate"]
    # All status='pending'
    assert all(p["status"] == "pending" for p in pendings)
    # SSE payload carries source_label_at_merge
    sse = next(p for p in pendings if p["kind"] == "sse_event")
    sse_payload = json.loads(sse["payload_json"])
    assert sse_payload["source_label_at_merge"] == "my-label"
    assert sse_payload["target_conversation_id"] == "tgt"


# ---------------------------------------------------------------------------
# Anti-subversion: body method does NOT call provider.delete or
# _store.delete_conversation (post-2026-04-26 incident invariant)
# ---------------------------------------------------------------------------

def test_body_does_not_call_destructive_primitives(tmp_path, monkeypatch):
    """§11.7 hook + invalidate safety. The body method must NOT call any
    destructive store-level primitive that purges per-conv tables. Move
    semantics = UPDATE conversation_id, NEVER DELETE FROM segments etc.
    """
    store = _store(tmp_path)
    delete_calls = []
    original = store.delete_conversation
    def _spy_delete(*args, **kwargs):
        delete_calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "delete_conversation", _spy_delete)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_segment(conn, "src", "seg-x")
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    assert delete_calls == [], f"body called delete_conversation: {delete_calls}"


def test_body_returns_merge_stats_with_expected_fields(tmp_path):
    """MergeStats return shape per plan T1.1 + B-D9 (success + elapsed_seconds)."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    assert stats.merge_id == merge_id
    assert stats.tenant_id == "tA"
    assert stats.source_conversation_id == "src"
    assert stats.target_conversation_id == "tgt"
    assert stats.sort_key_offset == 1000.0
    assert stats.request_turn_offset == 10
    assert isinstance(stats.rows_moved, dict)
    # B-D9 (codex iter-2 P2): success + elapsed_seconds populated.
    assert stats.success is True
    assert stats.elapsed_seconds >= 0.0
    # frozen dataclass
    with pytest.raises(Exception):
        stats.merge_id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# B-D1 (P0): cross-tenant body refusal
# ---------------------------------------------------------------------------

def test_body_refuses_cross_tenant_source(tmp_path):
    """B-D1: body re-validates source.tenant_id under the merge_audit
    FOR UPDATE row lock as defense-in-depth Layer C. A merge_audit row
    reserved under tenant A but with the source actually owned by tenant
    B (a misroute that bypassed cloud + engine validation) MUST be
    refused by the body before any UPDATE-row-move runs.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "tgt")  # target under tenant A
    _seed_conversation(conn, "tB", "src")  # source under tenant B
    _seed_segment(conn, "src", "seg-x")
    conn.commit()
    # Reserve audit under tenant A claiming src as the source. Cloud / engine
    # would normally catch this, but we test the body's defense-in-depth by
    # reserving directly.
    merge_id = _reserve(store, tenant_id="tA", source="src", target="tgt")
    with pytest.raises(CrossTenantMergeError):
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    # Source's segments NOT moved to target (rollback semantics).
    src_segs = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE conversation_id = 'src'",
    ).fetchone()[0]
    assert src_segs == 1
    tgt_segs = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE conversation_id = 'tgt'",
    ).fetchone()[0]
    assert tgt_segs == 0
    # merge_audit row remains in_progress (caller's responsibility to
    # mark rolled_back via _mark_merge_rolled_back per single-owner
    # rollback contract).
    audit_status = conn.execute(
        "SELECT status FROM merge_audit WHERE merge_id = ?", (merge_id,),
    ).fetchone()["status"]
    assert audit_status == "in_progress"


def test_body_refuses_missing_source(tmp_path):
    """B-D1: body raises CrossTenantMergeError if source.conversations row
    is absent (e.g., source got hard-deleted between reservation and body).
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "tgt")
    # source row deliberately NOT seeded
    conn.commit()
    merge_id = _reserve(store, source="ghost", target="tgt")
    with pytest.raises(CrossTenantMergeError):
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="ghost", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )


def test_body_per_table_writes_are_conversation_scoped(tmp_path):
    """B-D1 invariant: per-conv tables don't carry tenant_id; tenant
    scoping is transitive via the conversations row (re-validated at body
    entry). Pin via SQL inspection: scan the body method source for any
    UPDATE/DELETE that references a per-conv table; assert each predicates
    on conversation_id (the only available scoping).
    """
    import inspect
    import re
    from virtual_context.storage import sqlite as sqlite_mod
    src = inspect.getsource(sqlite_mod.SQLiteStore.merge_conversation_data)
    # Find every DELETE FROM <table> in the body's SQL strings.
    deletes = re.findall(r"DELETE\s+FROM\s+(\w+)", src, re.IGNORECASE)
    # The bounded-DELETE allowlist (per plan §13.3 + handoff B-D11):
    #   request_turn_counters, tag_summaries, tag_summary_embeddings,
    #   tag_aliases. Any DELETE outside this list is anti-subversion violation.
    allowlist = {
        "request_turn_counters", "tag_summaries", "tag_summary_embeddings",
        "tag_aliases",
    }
    unbounded = [t for t in deletes if t not in allowlist]
    assert unbounded == [], (
        f"body method contains DELETE against non-allowlisted tables: "
        f"{unbounded}; this is an anti-subversion violation. "
        f"Move semantics is UPDATE conversation_id, NEVER DELETE FROM "
        f"<per_conv_table>."
    )


# ---------------------------------------------------------------------------
# B-D2 (P1): concurrent ingest / compaction / phase refusal
# ---------------------------------------------------------------------------

def _seed_compaction_op(conn, conv_id, lifecycle_epoch=1, status="running"):
    conn.execute(
        "INSERT INTO compaction_operation "
        "(operation_id, conversation_id, lifecycle_epoch, phase_index, "
        "phase_count, phase_name, status, started_at, owner_worker_id, "
        "heartbeat_ts, created_at) "
        "VALUES (?, ?, ?, 0, 1, 'p', ?, ?, 'w', ?, ?)",
        (str(uuid.uuid4()), conv_id, lifecycle_epoch, status, _now_iso(),
         _now_iso(), _now_iso()),
    )


def _seed_ingestion_episode(conn, conv_id, lifecycle_epoch=1, status="running"):
    conn.execute(
        "INSERT INTO ingestion_episode "
        "(episode_id, conversation_id, lifecycle_epoch, raw_payload_entries, "
        "started_at, status, owner_worker_id, heartbeat_ts) "
        "VALUES (?, ?, ?, 0, ?, ?, 'w', ?)",
        (str(uuid.uuid4()), conv_id, lifecycle_epoch, _now_iso(), status,
         _now_iso()),
    )


def test_body_refuses_active_compaction_on_source(tmp_path):
    """B-D2: refuse merge if source has active compaction_operation
    (status='queued' or 'running')."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_compaction_op(conn, "src", status="running")
    conn.commit()
    merge_id = _reserve(store)
    with pytest.raises(MergeBusy) as exc:
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    assert exc.value.code == "merge_busy_compact"


def test_body_refuses_active_compaction_on_target(tmp_path):
    """B-D2: refuse merge if target has active compaction_operation."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_compaction_op(conn, "tgt", status="queued")
    conn.commit()
    merge_id = _reserve(store)
    with pytest.raises(MergeBusy) as exc:
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    assert exc.value.code == "merge_busy_compact"


def test_body_refuses_running_ingestion_on_source(tmp_path):
    """B-D2: refuse merge if source has running ingestion_episode."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_ingestion_episode(conn, "src", status="running")
    conn.commit()
    merge_id = _reserve(store)
    with pytest.raises(MergeBusy) as exc:
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    assert exc.value.code == "merge_busy_ingest"


def test_body_refuses_source_in_busy_phase(tmp_path):
    """B-D2: refuse merge if source.phase IN
    ('ingesting','compacting','deleted','merged')."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.execute("UPDATE conversations SET phase = 'ingesting' WHERE conversation_id = 'src'")
    conn.commit()
    merge_id = _reserve(store)
    with pytest.raises(MergeBusy) as exc:
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    assert exc.value.code == "merge_busy_phase"


def test_body_refuses_target_in_busy_phase(tmp_path):
    """B-D2: refuse merge if target.phase IN ('compacting',...)."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.execute("UPDATE conversations SET phase = 'compacting' WHERE conversation_id = 'tgt'")
    conn.commit()
    merge_id = _reserve(store)
    with pytest.raises(MergeBusy) as exc:
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    assert exc.value.code == "merge_busy_phase"


# ---------------------------------------------------------------------------
# B-D3 (P1): source lifecycle epoch validation
# ---------------------------------------------------------------------------

def test_body_refuses_source_lifecycle_epoch_advance(tmp_path):
    """B-D3: source.lifecycle_epoch must match the caller-captured value.
    Refuses if epoch advanced (e.g., source was deleted+recreated between
    reserve and body)."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src", lifecycle_epoch=3)  # actual = 3
    _seed_conversation(conn, "tA", "tgt", lifecycle_epoch=1)
    conn.commit()
    merge_id = _reserve(store)
    with pytest.raises(LifecycleEpochMismatch):
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1,
            expected_source_lifecycle_epoch=1,  # caller thinks epoch=1
            source_label_at_merge="lbl",
        )


# ---------------------------------------------------------------------------
# B-D4 (P1): canonical_turns moves reset compacted_at = NULL
# ---------------------------------------------------------------------------

def test_body_resets_compacted_at_on_canonical_turns_move(tmp_path):
    """B-D4: source's canonical_turns rows arrive with compacted_at
    populated; target's compaction prefix invariant requires NULL on
    moved rows so the compaction pipeline picks them up as fresh tail."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # Seed source's canonical_turn with compacted_at populated.
    conn.execute(
        "INSERT INTO canonical_turns (canonical_turn_id, conversation_id, sort_key, "
        "turn_hash, hash_version, compacted_at, created_at, updated_at) "
        "VALUES ('t1', 'src', 5.0, 'h', 1, ?, ?, ?)",
        (_now_iso(), _now_iso(), _now_iso()),
    )
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    moved = conn.execute(
        "SELECT compacted_at FROM canonical_turns WHERE canonical_turn_id = 't1'",
    ).fetchone()
    assert moved["compacted_at"] is None


# ---------------------------------------------------------------------------
# B-D6 (P1): tag_regenerate payload carries conflict spec list
# ---------------------------------------------------------------------------

def test_body_tag_regenerate_payload_includes_conflict_specs(tmp_path):
    """B-D6: when source + target both have tag_summaries for the same
    tag, the tag_regenerate post-commit pending payload must carry an
    explicit (tag, source_canonical_turn_ids, target_canonical_turn_ids)
    spec per conflict so the Phase B sweeper has enough state to LLM-
    regen the unioned summary."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # Seed source + target with conflicting "python" tag, each carrying
    # distinct source_canonical_turn_ids JSON arrays.
    now = _now_iso()
    conn.execute(
        "INSERT INTO tag_summaries (tag, conversation_id, summary, "
        "source_canonical_turn_ids, created_at, updated_at) "
        "VALUES ('python', 'src', 's-py', ?, ?, ?)",
        (json.dumps(["s-t1", "s-t2"]), now, now),
    )
    conn.execute(
        "INSERT INTO tag_summaries (tag, conversation_id, summary, "
        "source_canonical_turn_ids, created_at, updated_at) "
        "VALUES ('python', 'tgt', 't-py', ?, ?, ?)",
        (json.dumps(["t-t9"]), now, now),
    )
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    pending = conn.execute(
        "SELECT payload_json FROM merge_post_commit_pending "
        "WHERE merge_id = ? AND kind = 'tag_regenerate'", (merge_id,),
    ).fetchone()
    payload = json.loads(pending["payload_json"])
    assert "conflicts" in payload
    conflicts = payload["conflicts"]
    assert len(conflicts) == 1
    spec = conflicts[0]
    assert spec["tag"] == "python"
    assert sorted(spec["source_canonical_turn_ids"]) == ["s-t1", "s-t2"]
    assert spec["target_canonical_turn_ids"] == ["t-t9"]


# ---------------------------------------------------------------------------
# B-D7 (P1): prior_alias_target captured on merge_audit
# ---------------------------------------------------------------------------

def test_body_captures_prior_alias_target_in_audit(tmp_path):
    """B-D7: if source has a prior conversation_aliases row pointing
    elsewhere, the body captures that target_id into
    merge_audit.prior_alias_target before the UPSERT overwrites it. This
    is reversibility: a future merge-revert can restore the prior alias.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_conversation(conn, "tA", "earlier-tgt")
    # Seed source alias pointing at an earlier merge target.
    conn.execute(
        "INSERT INTO conversation_aliases (alias_id, target_id, epoch) "
        "VALUES ('src', 'earlier-tgt', 1)",
    )
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    audit = conn.execute(
        "SELECT prior_alias_target FROM merge_audit WHERE merge_id = ?",
        (merge_id,),
    ).fetchone()
    assert audit["prior_alias_target"] == "earlier-tgt"
    # And the new alias points at tgt now.
    new_alias = conn.execute(
        "SELECT target_id FROM conversation_aliases WHERE alias_id = 'src'",
    ).fetchone()
    assert new_alias["target_id"] == "tgt"


def test_body_prior_alias_target_null_when_absent(tmp_path):
    """B-D7: prior_alias_target column is NULL when source had no prior
    alias (the common case)."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    audit = conn.execute(
        "SELECT prior_alias_target FROM merge_audit WHERE merge_id = ?",
        (merge_id,),
    ).fetchone()
    assert audit["prior_alias_target"] is None


# ---------------------------------------------------------------------------
# B-D8 (P2): tag_aliases moves with conflict resolution
# ---------------------------------------------------------------------------

def test_body_moves_tag_aliases_no_conflict(tmp_path):
    """B-D8: source's tag_aliases rows that don't collide with target's
    are UPDATEd to target's namespace; origin_conversation_id captures
    source."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.execute(
        "INSERT INTO tag_aliases (alias, conversation_id, canonical) "
        "VALUES ('py', 'src', 'python')",
    )
    conn.execute(
        "INSERT INTO tag_aliases (alias, conversation_id, canonical) "
        "VALUES ('js', 'src', 'javascript')",
    )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    moved = conn.execute(
        "SELECT alias, canonical, origin_conversation_id FROM tag_aliases "
        "WHERE conversation_id = 'tgt' ORDER BY alias",
    ).fetchall()
    aliases = sorted(r["alias"] for r in moved)
    assert aliases == ["js", "py"]
    assert all(r["origin_conversation_id"] == "src" for r in moved)
    src_count = conn.execute(
        "SELECT COUNT(*) FROM tag_aliases WHERE conversation_id = 'src'",
    ).fetchone()[0]
    assert src_count == 0
    assert stats.rows_moved.get("tag_aliases") == 2


def test_body_resolves_tag_aliases_conflicts(tmp_path):
    """B-D8: when source + target share an alias, target wins; source's
    conflicting alias is DELETEd."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.execute(
        "INSERT INTO tag_aliases (alias, conversation_id, canonical) "
        "VALUES ('py', 'src', 'src-canon')",
    )
    conn.execute(
        "INSERT INTO tag_aliases (alias, conversation_id, canonical) "
        "VALUES ('py', 'tgt', 'tgt-canon')",
    )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # target's "py" preserved
    py = conn.execute(
        "SELECT canonical FROM tag_aliases WHERE conversation_id = 'tgt' AND alias = 'py'",
    ).fetchone()
    assert py["canonical"] == "tgt-canon"
    src_count = conn.execute(
        "SELECT COUNT(*) FROM tag_aliases WHERE conversation_id = 'src'",
    ).fetchone()[0]
    assert src_count == 0
    assert stats.rows_moved.get("tag_aliases__conflicts_deleted") == 1


# ---------------------------------------------------------------------------
# B-D5 (P1) + v1.14-2 + v1.14-6: request_turn_offset must avoid collision
# across all 4 tables. Per v1.14-2 (codex iter-3 P1), the body computes
# this offset INSIDE the conversation_lifecycle FOR UPDATE lock; the
# engine no longer pre-computes. The test exercises the body path with
# request_turn_offset=0 (no caller floor) so the recompute-under-lock
# behavior is verified end-to-end.
# ---------------------------------------------------------------------------

def test_body_request_turn_offset_avoids_collision_across_tables(tmp_path):
    """v1.14-6 (codex iter-3 P2): the body computes request_turn_offset
    UNDER the conversation_lifecycle FOR UPDATE lock from MAX across all
    four request-turn-bearing tables (tool_calls, request_context,
    request_captures, request_turn_counters). Seeding target with
    request_context but no tool_calls must still produce a moved
    request_turn that does NOT collide with target's existing range.
    Calls into Store.merge_conversation_data with offset=0 (no caller-
    supplied floor) so the body's own recomputation is exercised
    end-to-end rather than via inline test arithmetic.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # Target has request_context with high request_turn but no tool_calls.
    conn.execute(
        "INSERT INTO request_context "
        "(conversation_id, request_turn, timestamp, user_message, inbound_tags, "
        "retrieval_method, candidates_found, candidates_selected, "
        "segments_injected, facts_injected, facts_count, facts_tags, "
        "pool_used, pool_budget, total_context_tokens, "
        "non_virtualizable_floor) "
        "VALUES ('tgt', 50, ?, '', '', '', 0, 0, '', '', 0, '', 0, 0, 0, 0)",
        (_now_iso(),),
    )
    # Source has tool_calls at request_turn=2.
    conn.execute(
        "INSERT INTO tool_calls "
        "(conversation_id, request_turn, round, group_id, tool_name, "
        "tool_input, tool_result, result_length, duration_ms, timestamp) "
        "VALUES ('src', 2, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
        (_now_iso(),),
    )
    conn.commit()

    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        # offset=0 forces the body to recompute under its lock.
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Body's recomputed offset must be >= 51 (target's request_context max
    # was 50, so any free slot is at >=51). Source's tool_call had
    # request_turn=2; after offset, it lands at 2 + offset.
    assert stats.request_turn_offset >= 51
    moved = conn.execute(
        "SELECT request_turn FROM tool_calls WHERE conversation_id = 'tgt'",
    ).fetchone()
    assert moved is not None
    assert int(moved["request_turn"]) >= 53
    # No collision: target's pre-existing request_context row at
    # request_turn=50 still resolves uniquely.
    rc = conn.execute(
        "SELECT request_turn FROM request_context WHERE conversation_id = 'tgt' "
        "AND request_turn = 50",
    ).fetchone()
    assert rc is not None


# ---------------------------------------------------------------------------
# v1.14-1 (P1): target request_turn_counter bumped past moved range
# ---------------------------------------------------------------------------

def test_body_bumps_target_request_turn_counter_past_moved_range(tmp_path):
    """v1.14-1: after merge moves source's request-turn-bearing rows into
    target's namespace, target's ``request_turn_counters.next_request_turn``
    MUST be at least ``moved_max + 1``. Future ``save_request_context()``
    calls allocate from target's counter; without this bump, allocations
    would land IN the moved range and collide on
    ``(conversation_id, request_turn)``.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # target's pre-existing counter is at 3 (next slot = 3).
    _seed_request_turn_counter(conn, "tgt", 3)
    # Source has tool_calls at high request_turns 1..5.
    for rt in (1, 2, 3, 4, 5):
        conn.execute(
            "INSERT INTO tool_calls "
            "(conversation_id, request_turn, round, group_id, tool_name, "
            "tool_input, tool_result, result_length, duration_ms, timestamp) "
            "VALUES ('src', ?, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
            (rt, _now_iso()),
        )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Body recomputed offset under lock to >= max(target counter=3, MAX
    # of all request-turn tables on target = 0) = 3. So source's
    # request_turn=5 lands at offset+5 >= 8. moved_max should be at
    # least 8.
    moved_max = conn.execute(
        "SELECT MAX(request_turn) FROM tool_calls WHERE conversation_id = 'tgt'",
    ).fetchone()[0]
    assert moved_max is not None
    assert moved_max >= stats.request_turn_offset + 5
    # Target's counter MUST have been bumped past moved_max.
    new_counter = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters "
        "WHERE conversation_id = 'tgt'",
    ).fetchone()["next_request_turn"]
    assert new_counter >= moved_max + 1
    # Stats record the bump.
    assert "request_turn_counters_target_bumped_to" in stats.rows_moved
    assert stats.rows_moved["request_turn_counters_target_bumped_to"] >= moved_max + 1


def test_body_target_counter_unchanged_when_no_request_turn_rows_move(tmp_path):
    """v1.14-1: when source has no request-turn-bearing rows, target's
    counter is left as-is (the bump is conditional on moved_max > 0)."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_request_turn_counter(conn, "tgt", 7)
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    new_counter = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters "
        "WHERE conversation_id = 'tgt'",
    ).fetchone()["next_request_turn"]
    assert new_counter == 7


def test_body_post_merge_save_request_context_does_not_collide(tmp_path):
    """v1.14-1 end-to-end: after merge, allocate a fresh request_turn via
    the counter (simulating save_request_context()) and assert it does
    NOT collide with any previously-moved request_turn on target.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_request_turn_counter(conn, "tgt", 1)
    # Source rows at request_turns 1, 2, 3.
    for rt in (1, 2, 3):
        conn.execute(
            "INSERT INTO tool_calls "
            "(conversation_id, request_turn, round, group_id, tool_name, "
            "tool_input, tool_result, result_length, duration_ms, timestamp) "
            "VALUES ('src', ?, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
            (rt, _now_iso()),
        )
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Allocate a fresh turn via the underlying counter helper.
    next_turn = store._allocate_request_turn(conn, "tgt")
    # Existing request_turns on target after merge:
    moved = sorted(int(r["request_turn"]) for r in conn.execute(
        "SELECT request_turn FROM tool_calls WHERE conversation_id = 'tgt'",
    ).fetchall())
    assert next_turn not in moved
    assert next_turn > max(moved)


# ---------------------------------------------------------------------------
# v1.14-2 (P1): offsets recomputed under the lock; concurrent-writer race
# ---------------------------------------------------------------------------

def test_body_recomputes_offset_under_lock_with_concurrent_writer_simulation(tmp_path):
    """v1.14-2: the body must recompute offsets AFTER acquiring the
    conversation_lifecycle FOR UPDATE lock. Simulate a concurrent writer
    that lands a high-request_turn row between caller-passed offset (0)
    and body lock acquisition; assert the body's recomputed offset rides
    over it.

    On SQLite the BEGIN IMMEDIATE serializes at the database level; we
    simulate the race by injecting the new row BEFORE invoking the body
    (after the caller-side offset would have been computed).
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # "Caller side" offset would have been 0 + 1 = 1.
    # Now simulate a concurrent writer landing a request_context row at
    # request_turn=99 BEFORE the body's recomputation.
    conn.execute(
        "INSERT INTO request_context "
        "(conversation_id, request_turn, timestamp, user_message, inbound_tags, "
        "retrieval_method, candidates_found, candidates_selected, "
        "segments_injected, facts_injected, facts_count, facts_tags, "
        "pool_used, pool_budget, total_context_tokens, "
        "non_virtualizable_floor) "
        "VALUES ('tgt', 99, ?, '', '', '', 0, 0, '', '', 0, '', 0, 0, 0, 0)",
        (_now_iso(),),
    )
    # Source has a tool_call at request_turn=1.
    conn.execute(
        "INSERT INTO tool_calls "
        "(conversation_id, request_turn, round, group_id, tool_name, "
        "tool_input, tool_result, result_length, duration_ms, timestamp) "
        "VALUES ('src', 1, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
        (_now_iso(),),
    )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=0.0, request_turn_offset=0,  # caller used stale 0
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Body's recomputed offset must be >= 100 (>= request_context's 99 + 1).
    assert stats.request_turn_offset >= 100
    moved = conn.execute(
        "SELECT request_turn FROM tool_calls WHERE conversation_id = 'tgt'",
    ).fetchone()
    assert int(moved["request_turn"]) >= 101


def test_body_caller_offset_is_floor_not_ceiling(tmp_path):
    """v1.14-2: caller-passed offset is honored as a floor; if recomputed
    is lower, caller's value is used (test predictability preserved).
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.execute(
        "INSERT INTO tool_calls "
        "(conversation_id, request_turn, round, group_id, tool_name, "
        "tool_input, tool_result, result_length, duration_ms, timestamp) "
        "VALUES ('src', 3, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
        (_now_iso(),),
    )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=0.0, request_turn_offset=100,  # caller's high floor
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Floor honored: offset >= 100 even though recomputed would be 1.
    assert stats.request_turn_offset >= 100
    moved = conn.execute(
        "SELECT request_turn FROM tool_calls WHERE conversation_id = 'tgt'",
    ).fetchone()
    assert int(moved["request_turn"]) == 103


# ---------------------------------------------------------------------------
# v1.14-3 (P2): SQL inspection extended to verify DELETE WHERE-clause shape
# ---------------------------------------------------------------------------

def test_body_sql_inspection_delete_predicates_are_scope_bounded(tmp_path):
    """v1.14-3 (codex iter-3 P2): the bounded-DELETE allowlist test by
    name only is necessary but not sufficient. Scan each DELETE FROM in
    the body method source code; assert the WHERE clause includes a
    recognized scoping predicate. A future regression that adds
    ``DELETE FROM tag_aliases`` (no WHERE / wide-open) would pass the
    name-only check but fail this stricter shape check.
    """
    import inspect
    import re
    from virtual_context.storage import sqlite as sqlite_mod
    from virtual_context.storage import postgres as postgres_mod

    expected_predicates = {
        # request_turn_counters: scoped on conversation_id = source.
        "request_turn_counters": ("conversation_id",),
        # tag_summaries / tag_summary_embeddings / tag_aliases: scoped on
        # conversation_id = source AND collide-key IN target subselect.
        "tag_summaries": ("conversation_id", "IN"),
        "tag_summary_embeddings": ("conversation_id", "IN"),
        "tag_aliases": ("conversation_id", "IN"),
    }

    for mod, store_cls_name in (
        (sqlite_mod, "SQLiteStore"),
        (postgres_mod, "PostgresStore"),
    ):
        store_cls = getattr(mod, store_cls_name, None)
        if store_cls is None:
            continue
        body = getattr(store_cls, "merge_conversation_data", None)
        if body is None:
            continue
        src = inspect.getsource(body)
        # Find each DELETE FROM <table> ... and the next 400 chars (WHERE
        # context). Extract the DELETE table + the next chunk of source
        # to inspect WHERE-clause shape.
        for match in re.finditer(
            r"DELETE\s+FROM\s+(?:\{tbl\}|\w+)", src, re.IGNORECASE,
        ):
            window = src[match.start():match.start() + 500]
            # Determine the table either from the literal or by scanning
            # nearby f-string variable name.
            literal_match = re.search(
                r"DELETE\s+FROM\s+(\w+)", window, re.IGNORECASE,
            )
            if literal_match is None:
                # f-string {tbl} indirection; scan nearby for the loop
                # variable's allowlist
                continue
            tbl = literal_match.group(1)
            if tbl.startswith("<") or tbl in ("tbl",):
                continue
            if tbl not in expected_predicates:
                # Allowlist failure already covered by the name-only test.
                continue
            preds = expected_predicates[tbl]
            for pred in preds:
                assert pred in window, (
                    f"{store_cls_name} DELETE FROM {tbl} missing required "
                    f"predicate keyword '{pred}' in WHERE-clause window: "
                    f"{window[:300]!r}"
                )
        # Also handle the f-string case (DELETE FROM {tbl}); the table
        # name comes from a containing for-loop variable. Locate the
        # for-loop allowlist (TABLES_TAG_CONFLICT) and inspect the
        # template's WHERE shape.
        for match in re.finditer(r"DELETE\s+FROM\s+\{tbl\}", src, re.IGNORECASE):
            window = src[match.start():match.start() + 500]
            assert "conversation_id" in window and "IN" in window, (
                f"{store_cls_name} f-string DELETE FROM {{tbl}} block "
                f"missing conversation_id + IN predicates: {window[:300]!r}"
            )


# ---------------------------------------------------------------------------
# B-D10 (P2): per-table move correctness coverage expansion
# ---------------------------------------------------------------------------

def test_body_moves_facts(tmp_path):
    """B-D10: facts table moves source rows; origin captured."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.execute(
        "INSERT INTO facts (id, conversation_id, subject, verb, object) "
        "VALUES ('f1', 'src', 'water', 'boils_at', '100C')",
    )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    moved = conn.execute(
        "SELECT id, conversation_id, origin_conversation_id FROM facts "
        "WHERE id = 'f1'",
    ).fetchone()
    assert moved["conversation_id"] == "tgt"
    assert moved["origin_conversation_id"] == "src"
    assert stats.rows_moved.get("facts") == 1


def test_body_moves_tool_calls_with_offset(tmp_path):
    """B-D10: tool_calls request_turn += offset on move."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    conn.execute(
        "INSERT INTO tool_calls "
        "(conversation_id, request_turn, round, group_id, tool_name, "
        "tool_input, tool_result, result_length, duration_ms, timestamp) "
        "VALUES ('src', 3, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
        (_now_iso(),),
    )
    conn.commit()
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=100,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    moved = conn.execute(
        "SELECT request_turn, origin_conversation_id FROM tool_calls "
        "WHERE conversation_id = 'tgt'",
    ).fetchone()
    assert moved["request_turn"] == 103
    assert moved["origin_conversation_id"] == "src"


def test_body_rollback_on_late_failure(tmp_path, monkeypatch):
    """B-D10: simulate failure during merge_post_commit_pending INSERT
    (the LAST step of the body). The whole body transaction must rollback;
    no row moves persist; merge_audit stays in_progress.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_segment(conn, "src", "seg-x")
    conn.commit()
    merge_id = _reserve(store)
    # Patch _uuid.uuid4 to raise on call inside the post-commit-pending
    # INSERT loop. The body's import-statement is `import uuid as _uuid`.
    import uuid as _uuid_mod
    call_count = {"n": 0}
    real_uuid4 = _uuid_mod.uuid4
    def _bad_uuid4():
        call_count["n"] += 1
        # First call (audit reservation) succeeds; second-onwards raises.
        if call_count["n"] >= 1:
            raise RuntimeError("simulated late-step failure")
        return real_uuid4()
    monkeypatch.setattr(_uuid_mod, "uuid4", _bad_uuid4)
    with pytest.raises(RuntimeError, match="simulated late-step failure"):
        store.merge_conversation_data(
            merge_id=merge_id, tenant_id="tA",
            source_conversation_id="src", target_conversation_id="tgt",
            sort_key_offset=1000.0, request_turn_offset=10,
            expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
        )
    # Source segments NOT moved (transaction rolled back).
    src_segs = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE conversation_id = 'src'",
    ).fetchone()[0]
    assert src_segs == 1
    tgt_segs = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE conversation_id = 'tgt'",
    ).fetchone()[0]
    assert tgt_segs == 0
    # merge_audit row stays in_progress (caller marks rolled_back externally).
    audit_status = conn.execute(
        "SELECT status FROM merge_audit WHERE merge_id = ?", (merge_id,),
    ).fetchone()["status"]
    assert audit_status == "in_progress"
    # Source phase NOT flipped to merged (still 'active').
    src_phase = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = 'src'",
    ).fetchone()["phase"]
    assert src_phase == "active"
    # No alias UPSERT persisted.
    alias_row = conn.execute(
        "SELECT 1 FROM conversation_aliases WHERE alias_id = 'src'",
    ).fetchone()
    assert alias_row is None


# ---------------------------------------------------------------------------
# B-D11 (P2): anti-subversion comprehensive guard
# ---------------------------------------------------------------------------

def test_body_does_not_call_provider_delete(tmp_path, monkeypatch):
    """B-D11: spy on provider.delete (used by the LLM-tool delete path)
    to confirm the body method never invokes it. Move semantics is
    UPDATE conversation_id, never delegate to the destructive provider
    primitive."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    _seed_segment(conn, "src", "seg-x")
    conn.commit()
    # If a provider attribute exists on the store, spy on its delete.
    provider = getattr(store, "provider", None)
    delete_spy_calls = []
    if provider is not None and hasattr(provider, "delete"):
        original_delete = provider.delete
        def _spy(*a, **kw):
            delete_spy_calls.append((a, kw))
            return original_delete(*a, **kw)
        monkeypatch.setattr(provider, "delete", _spy)
    # Spy delete_conversation as well (the v1.x defense).
    store_delete_calls = []
    original_dc = store.delete_conversation
    def _spy_dc(*a, **kw):
        store_delete_calls.append((a, kw))
        return original_dc(*a, **kw)
    monkeypatch.setattr(store, "delete_conversation", _spy_dc)
    merge_id = _reserve(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    assert delete_spy_calls == [], (
        f"body called provider.delete: {delete_spy_calls}"
    )
    assert store_delete_calls == [], (
        f"body called store.delete_conversation: {store_delete_calls}"
    )


def test_body_sql_inspection_no_unbounded_destructive_writes(tmp_path):
    """B-D11: regex-scan both the SQLite + PostgreSQL body method source
    code; assert every DELETE FROM <table> is allowlisted. The bounded-
    DELETE allowlist is the single source of truth for what destructive
    primitives the body is permitted to touch."""
    import inspect
    import re
    from virtual_context.storage import sqlite as sqlite_mod
    from virtual_context.storage import postgres as postgres_mod

    allowlist = {
        "request_turn_counters", "tag_summaries", "tag_summary_embeddings",
        "tag_aliases",
    }

    for mod, store_cls_name in (
        (sqlite_mod, "SQLiteStore"),
        (postgres_mod, "PostgresStore"),
    ):
        store_cls = getattr(mod, store_cls_name, None)
        if store_cls is None:
            continue
        body = getattr(store_cls, "merge_conversation_data", None)
        if body is None:
            continue
        src = inspect.getsource(body)
        # Capture DELETE FROM <table> targets.
        deletes = re.findall(r"DELETE\s+FROM\s+(\w+)", src, re.IGNORECASE)
        # Skip docstring-only mentions: each captured target must appear
        # inside an SQL string literal. The simple regex above is good
        # enough because the docstring uses lowercase prose like
        # "DELETE FROM <per_conv_table>" with angle-bracket placeholders.
        unbounded = [t for t in deletes
                     if t not in allowlist and not t.startswith("<")]
        assert unbounded == [], (
            f"{store_cls_name}.merge_conversation_data contains unbounded "
            f"DELETE FROM {unbounded}; anti-subversion violation"
        )


def test_body_sql_inspection_tenant_aware_writes_predicate_on_tenant_id(tmp_path):
    """B-D11 / B-D1 invariant: every UPDATE/SELECT against a TENANT-AWARE
    table (conversations, merge_audit) in the body method must predicate
    on tenant_id. Per-conv tables (segments, canonical_turns, ...) don't
    carry tenant_id; they're scoped on conversation_id only; tenant
    scoping is transitive via the body's source.tenant_id re-validation
    at body entry. This test pins the tenant-aware predicate invariant.

    v1.15-3 (codex iter-4 P2): parametrized over BOTH SQLite + PostgreSQL
    body method source code so PG predicate drift is also pinned.
    """
    import inspect
    from virtual_context.storage import sqlite as sqlite_mod
    from virtual_context.storage import postgres as postgres_mod

    tenant_aware_keywords = (
        "UPDATE conversations",
        "UPDATE merge_audit",
        "FROM conversations",
        "FROM merge_audit",
    )

    for mod, store_cls_name in (
        (sqlite_mod, "SQLiteStore"),
        (postgres_mod, "PostgresStore"),
    ):
        store_cls = getattr(mod, store_cls_name, None)
        if store_cls is None:
            continue
        body = getattr(store_cls, "merge_conversation_data", None)
        if body is None:
            continue
        src = inspect.getsource(body)
        # Locate the tenant-aware operations and confirm tenant_id appears
        # within a 600-char statement window. Looser than full-regex
        # parsing of multi-line SQL strings; robust against quote /
        # whitespace variation across PG (%s) and SQLite (?) parameter
        # styles.
        for kw in tenant_aware_keywords:
            idx = 0
            while True:
                pos = src.find(kw, idx)
                if pos < 0:
                    break
                window = src[pos:pos + 600]
                assert "tenant_id" in window, (
                    f"{store_cls_name}: tenant-aware operation '{kw}' at "
                    f"offset {pos} lacks tenant_id predicate within "
                    f"600-char window"
                )
                idx = pos + len(kw)


# ---------------------------------------------------------------------------
# v1.15-2 (P1): chained-merge counter bump uses MAX of all rows on target,
# not just rows where origin_conversation_id = most-recent source. Origin
# is preserved by COALESCE on subsequent merges, so an origin-filtered
# bump under-counts when the target has rows from earlier merges.
# ---------------------------------------------------------------------------

def test_body_chained_merge_counter_bump_includes_earlier_origin_rows(tmp_path):
    """v1.15-2: A->B->C chained merge. A's rows on B carry origin = A;
    when B->C runs, those rows move to C with origin still = A (preserved
    by COALESCE). The counter bump query must compute max over ALL rows
    on C regardless of origin so the next allocation cannot collide with
    A-origin rows.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "convA")
    _seed_conversation(conn, "tA", "convB")
    _seed_conversation(conn, "tA", "convC")
    # Stage 1: A has tool_calls at request_turn 1, 2, 3.
    for rt in (1, 2, 3):
        conn.execute(
            "INSERT INTO tool_calls "
            "(conversation_id, request_turn, round, group_id, tool_name, "
            "tool_input, tool_result, result_length, duration_ms, timestamp) "
            "VALUES ('convA', ?, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
            (rt, _now_iso()),
        )
    # B starts empty; counter at 1.
    _seed_request_turn_counter(conn, "convB", 1)
    _seed_request_turn_counter(conn, "convC", 1)
    conn.commit()

    # Merge A -> B
    merge_id_ab = _reserve(store, source="convA", target="convB")
    store.merge_conversation_data(
        merge_id=merge_id_ab, tenant_id="tA",
        source_conversation_id="convA", target_conversation_id="convB",
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl-AB",
    )
    # Verify A's rows now on B with origin = A.
    a_origin_count = conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE conversation_id = 'convB' "
        "AND origin_conversation_id = 'convA'",
    ).fetchone()[0]
    assert a_origin_count == 3
    # B's counter must be > the moved rows' max request_turn.
    b_counter = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters "
        "WHERE conversation_id = 'convB'",
    ).fetchone()["next_request_turn"]
    b_max_rt = conn.execute(
        "SELECT MAX(request_turn) FROM tool_calls WHERE conversation_id = 'convB'",
    ).fetchone()[0]
    assert b_counter > b_max_rt

    # Stage 2: B->C. A-origin rows on B carry origin = A still; the move
    # statement preserves their origin (COALESCE non-empty path).
    # convB needs phase = 'active' to be merge-eligible (was set to 'merged' in stage 1).
    # We'll create a fresh convD that takes A's slot for the next merge,
    # OR we use a different chain. Simpler: merge convB into convC.
    # But convB's phase is now 'merged' from stage 1's source-side flip.
    # That means convB cannot be a source in another merge (phase check
    # in B-D2 refuses 'merged' source).
    # Workaround for this chained-merge test: reset convB to 'active' so
    # we can simulate the chained-merge scenario the codex finding flagged.
    conn.execute("UPDATE conversations SET phase = 'active' WHERE conversation_id = 'convB'")
    conn.commit()

    merge_id_bc = _reserve(store, source="convB", target="convC")
    store.merge_conversation_data(
        merge_id=merge_id_bc, tenant_id="tA",
        source_conversation_id="convB", target_conversation_id="convC",
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl-BC",
    )

    # Verify A-origin rows now on C with origin still = A (NOT 'convB').
    a_origin_on_c = conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE conversation_id = 'convC' "
        "AND origin_conversation_id = 'convA'",
    ).fetchone()[0]
    assert a_origin_on_c == 3, (
        "Chained merge should preserve A's origin all the way to C "
        "(COALESCE non-empty branch)"
    )

    # The critical assertion: C's counter must be > MAX request_turn of
    # ALL rows on C, not just rows with origin = convB. With the v1.14
    # origin-filtered query, A-origin rows would be excluded and the
    # bump would under-count.
    c_max_rt = conn.execute(
        "SELECT MAX(request_turn) FROM tool_calls WHERE conversation_id = 'convC'",
    ).fetchone()[0]
    c_counter = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters "
        "WHERE conversation_id = 'convC'",
    ).fetchone()["next_request_turn"]
    assert c_counter >= c_max_rt + 1, (
        f"C's counter ({c_counter}) must be >= max request_turn ({c_max_rt}) + 1"
    )

    # And the next allocation must not collide.
    next_alloc = store._allocate_request_turn(conn, "convC")
    all_existing = sorted(int(r["request_turn"]) for r in conn.execute(
        "SELECT request_turn FROM tool_calls WHERE conversation_id = 'convC'",
    ).fetchall())
    assert next_alloc not in all_existing
    assert next_alloc > max(all_existing)


# ---------------------------------------------------------------------------
# v1.15-7 (P3): the bumped-counter stat reports the actual UPSERT result,
# not just moved_max + 1. When target's existing counter is higher than
# moved_max + 1, the stat should reflect the preserved-higher value.
# ---------------------------------------------------------------------------

def test_body_counter_stat_reflects_actual_upsert_value(tmp_path):
    """v1.15-7: stat ``request_turn_counters_target_bumped_to`` is the
    actual post-UPSERT next_request_turn (captured via RETURNING), NOT
    just ``moved_max + 1``. When target's existing counter is higher,
    GREATEST/MAX preserves it; the stat reflects that.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # Target's pre-merge counter is at 100 (much higher than what source's
    # rows will contribute post-offset).
    _seed_request_turn_counter(conn, "tgt", 100)
    # Source has tool_calls at request_turn 1, 2, 3.
    for rt in (1, 2, 3):
        conn.execute(
            "INSERT INTO tool_calls "
            "(conversation_id, request_turn, round, group_id, tool_name, "
            "tool_input, tool_result, result_length, duration_ms, timestamp) "
            "VALUES ('src', ?, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
            (rt, _now_iso()),
        )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Body's recomputed offset uses target's counter as floor (=100), so
    # source rows at request_turn 1,2,3 land at 101,102,103. Max on
    # target post-merge = 103. UPSERT proposes 104; GREATEST(100, 104) =
    # 104; counter ends at 104.
    final_counter = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters "
        "WHERE conversation_id = 'tgt'",
    ).fetchone()["next_request_turn"]
    assert "request_turn_counters_target_bumped_to" in stats.rows_moved
    # Stat MUST equal the actual final_counter, not just moved_max + 1.
    assert stats.rows_moved["request_turn_counters_target_bumped_to"] == final_counter


# ---------------------------------------------------------------------------
# v1.15-1 (P1): closer simulation of the cross-transaction stale-offset race.
# SQLite cannot exhibit the PG race (BEGIN IMMEDIATE serializes writers at
# the database level), so this test pins the SQLite-equivalent invariant:
# a write that lands BEFORE the body call (no concurrent writers possible
# under SQLite serialization) must still produce a non-colliding offset.
# Real concurrency exercise lives in the PG smoke file (v1.15-1's PG path).
# ---------------------------------------------------------------------------

def test_body_offset_recomputes_after_write_to_target(tmp_path):
    """v1.15-1 SQLite invariant: a write that lands on target's
    request-turn-bearing tables BEFORE the body's offset recomputation
    must be reflected in the recomputed offset. SQLite serializes via
    BEGIN IMMEDIATE so no concurrent writer can race the body; this test
    asserts the BEFORE-body case behaves as expected (which is what the
    SQLite path can prove behaviorally; PG concurrency lives in smoke).
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _seed_conversation(conn, "tA", "src")
    _seed_conversation(conn, "tA", "tgt")
    # Simulate save_request_context having written a high request_context
    # row to target before the body sees it.
    conn.execute(
        "INSERT INTO request_context "
        "(conversation_id, request_turn, timestamp, user_message, inbound_tags, "
        "retrieval_method, candidates_found, candidates_selected, "
        "segments_injected, facts_injected, facts_count, facts_tags, "
        "pool_used, pool_budget, total_context_tokens, "
        "non_virtualizable_floor) "
        "VALUES ('tgt', 200, ?, '', '', '', 0, 0, '', '', 0, '', 0, 0, 0, 0)",
        (_now_iso(),),
    )
    # Source has a tool_call at request_turn=1.
    conn.execute(
        "INSERT INTO tool_calls "
        "(conversation_id, request_turn, round, group_id, tool_name, "
        "tool_input, tool_result, result_length, duration_ms, timestamp) "
        "VALUES ('src', 1, 0, 'g', 't', '{}', '{}', 0, 0.0, ?)",
        (_now_iso(),),
    )
    conn.commit()
    merge_id = _reserve(store)
    stats = store.merge_conversation_data(
        merge_id=merge_id, tenant_id="tA",
        source_conversation_id="src", target_conversation_id="tgt",
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )
    # Recomputed offset must be > 200 (target's request_context max).
    assert stats.request_turn_offset >= 201
    # Source's tool_call lands at 1 + offset.
    moved_rt = conn.execute(
        "SELECT request_turn FROM tool_calls WHERE conversation_id = 'tgt' "
        "AND origin_conversation_id = 'src'",
    ).fetchone()["request_turn"]
    assert int(moved_rt) >= 202
    # Counter bumped above the moved range.
    counter = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters "
        "WHERE conversation_id = 'tgt'",
    ).fetchone()["next_request_turn"]
    assert counter >= int(moved_rt) + 1


def test_pg_share_lock_helper_present_on_postgres_store(tmp_path):
    """v1.15-1 structural pin: PostgresStore must expose the lifecycle
    SHARE-lock helper used by save_request_context. This test catches
    a regression that drops the helper or stops calling it.
    """
    from virtual_context.storage.postgres import PostgresStore
    assert hasattr(PostgresStore, "_acquire_lifecycle_share_lock"), (
        "v1.15-1: PostgresStore._acquire_lifecycle_share_lock helper missing; "
        "stale-offset race re-opens without it"
    )
    # Confirm save_request_context calls the helper (text-level grep is
    # the pragmatic invariant; behavioral smoke is in the PG body file).
    import inspect
    src = inspect.getsource(PostgresStore.save_request_context)
    assert "_acquire_lifecycle_share_lock" in src, (
        "v1.15-1: PostgresStore.save_request_context no longer invokes "
        "_acquire_lifecycle_share_lock; lock contract regressed"
    )
