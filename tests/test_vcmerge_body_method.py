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
from virtual_context.core.exceptions import MergeAuditMissing, LifecycleEpochMismatch


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
    """MergeStats return shape per plan T1.1."""
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
    # frozen dataclass
    with pytest.raises(Exception):
        stats.merge_id = "modified"  # type: ignore[misc]
