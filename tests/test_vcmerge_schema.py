"""Schema-smoke tests for VCMERGE Phase 0 migrations (, , ).

 Pins:
- conversations.phase CHECK admits the 'merged' value
- merge_audit table exists with expected columns + tenant_id
- merge_post_commit_pending table exists with tenant_id + the two
  tenant-consistency triggers
- idx_merge_audit_active_source rejects duplicate (tenant, source) at
  status IN ('in_progress', 'committed')
- _ensure_schema() is idempotent re-runnable

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
: every test in the anti-
subversion bundle carries the marker because the bundle's existence
traces to the VCATTACH data-loss anchor incident at
commit 502db28. See tests/REGRESSION_MAP.md.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore


pytestmark = pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(tmp_path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "store.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _insert_conversation(conn, tenant_id: str, conv_id: str, phase: str = "active") -> None:
    now = _now_iso()
    conn.execute(
        "INSERT INTO conversations "
        "(conversation_id, tenant_id, phase, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conv_id, tenant_id, phase, now, now),
    )


def _insert_merge_audit(
    conn,
    *,
    merge_id: str,
    tenant_id: str,
    source: str,
    target: str,
    status: str = "in_progress",
    label: str = "",
) -> None:
    conn.execute(
        "INSERT INTO merge_audit "
        "(merge_id, tenant_id, source_conversation_id, target_conversation_id, "
        "source_label_at_merge, status, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (merge_id, tenant_id, source, target, label, status, _now_iso()),
    )


# ---------------------------------------------------------------------------
# : conversations.phase CHECK admits 'merged'
# ---------------------------------------------------------------------------

def test_conversations_phase_check_admits_merged(tmp_path):
    """: the phase CHECK must accept the new 'merged' value."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _insert_conversation(conn, "tenant-A", "conv-1", phase="merged")
    row = conn.execute(
        "SELECT phase FROM conversations WHERE conversation_id = ?",
        ("conv-1",),
    ).fetchone()
    assert row is not None
    assert row["phase"] == "merged"


def test_conversations_phase_check_rejects_invalid_value(tmp_path):
    """: the CHECK still rejects unknown phase values."""
    store = _store(tmp_path)
    conn = store._get_conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_conversation(conn, "tenant-A", "conv-bad", phase="banana")


def test_conversations_phase_check_admits_existing_values(tmp_path):
    """: every prior phase value (init/active/ingesting/compacting/deleted)
    still accepted post-relaxation. Regression for accidental phase removal.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    for phase in ("init", "ingesting", "compacting", "active", "deleted"):
        _insert_conversation(conn, "tenant-A", f"conv-{phase}", phase=phase)


# ---------------------------------------------------------------------------
# : merge_audit table
# ---------------------------------------------------------------------------

def test_merge_audit_table_exists_with_expected_columns(tmp_path):
    """: table exists with all columns including tenant_id."""
    store = _store(tmp_path)
    conn = store._get_conn()
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(merge_audit)").fetchall()
    }
    expected = {
        "merge_id",
        "tenant_id",
        "source_conversation_id",
        "target_conversation_id",
        "source_label_at_merge",
        "status",
        "started_at",
        "completed_at",
        "rows_moved_json",
        "error_message",
    }
    missing = expected - cols
    assert not missing, f"merge_audit missing columns: {missing}"


def test_merge_audit_status_check_admits_three_states(tmp_path):
    """: status CHECK admits in_progress, committed, rolled_back."""
    store = _store(tmp_path)
    conn = store._get_conn()
    for status in ("in_progress", "committed", "rolled_back"):
        _insert_merge_audit(
            conn,
            merge_id=str(uuid.uuid4()),
            tenant_id="tenant-A",
            source=f"src-{status}",
            target=f"tgt-{status}",
            status=status,
        )


def test_merge_audit_status_check_rejects_invalid(tmp_path):
    """: status CHECK rejects unknown values."""
    store = _store(tmp_path)
    conn = store._get_conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_merge_audit(
            conn,
            merge_id=str(uuid.uuid4()),
            tenant_id="t1",
            source="s1",
            target="t1c",
            status="banana",
        )


# ---------------------------------------------------------------------------
# : idx_merge_audit_active_source unique partial index
# ---------------------------------------------------------------------------

def test_unique_partial_index_rejects_duplicate_in_progress(tmp_path):
    """: second in_progress row for same (tenant, source) violates."""
    store = _store(tmp_path)
    conn = store._get_conn()
    _insert_merge_audit(
        conn, merge_id=str(uuid.uuid4()), tenant_id="t1",
        source="src1", target="tgt1", status="in_progress",
    )
    with pytest.raises(sqlite3.IntegrityError):
        _insert_merge_audit(
            conn, merge_id=str(uuid.uuid4()), tenant_id="t1",
            source="src1", target="tgt1", status="in_progress",
        )


def test_unique_partial_index_rejects_committed_after_in_progress(tmp_path):
    """ + D4: committed rows MUST stay in the unique-index predicate so
    re-merge attempts collide and resolve via the 5-state idempotency
    discriminator.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _insert_merge_audit(
        conn, merge_id=str(uuid.uuid4()), tenant_id="t1",
        source="src1", target="tgt1", status="committed",
    )
    with pytest.raises(sqlite3.IntegrityError):
        _insert_merge_audit(
            conn, merge_id=str(uuid.uuid4()), tenant_id="t1",
            source="src1", target="tgt2", status="in_progress",
        )


def test_unique_partial_index_admits_rolled_back(tmp_path):
    """ + D4: rolled_back rows are OUT of the partial index, so a fresh
    in_progress row for the same (tenant, source) succeeds (retry path).
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _insert_merge_audit(
        conn, merge_id=str(uuid.uuid4()), tenant_id="t1",
        source="src1", target="tgt1", status="rolled_back",
    )
    # Should NOT raise: rolled_back is excluded from the unique predicate.
    _insert_merge_audit(
        conn, merge_id=str(uuid.uuid4()), tenant_id="t1",
        source="src1", target="tgt1", status="in_progress",
    )


def test_unique_partial_index_scoped_per_tenant(tmp_path):
    """ + tenant isolation (per ): two tenants
    with the same source_id can both reserve in_progress.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    _insert_merge_audit(
        conn, merge_id=str(uuid.uuid4()), tenant_id="tenant-A",
        source="shared-src", target="tgt-A", status="in_progress",
    )
    _insert_merge_audit(
        conn, merge_id=str(uuid.uuid4()), tenant_id="tenant-B",
        source="shared-src", target="tgt-B", status="in_progress",
    )


# ---------------------------------------------------------------------------
# : merge_post_commit_pending + tenant-consistency triggers
# ---------------------------------------------------------------------------

def test_merge_post_commit_pending_table_exists_with_tenant_id(tmp_path):
    """ + D2: table exists with tenant_id column for tenant-isolated
    consumer support.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    cols = {
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(merge_post_commit_pending)"
        ).fetchall()
    }
    expected = {
        "pending_id", "merge_id", "tenant_id", "kind", "payload_json",
        "status", "attempts", "created_at", "last_attempt_at",
        "completed_at", "error_message",
    }
    missing = expected - cols
    assert not missing, f"merge_post_commit_pending missing columns: {missing}"


def test_merge_post_commit_pending_insert_trigger_rejects_tenant_mismatch(tmp_path):
    """ trigger (insert variant): INSERT with tenant_id != audit's
    tenant_id raises a constraint error.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    merge_id = str(uuid.uuid4())
    _insert_merge_audit(
        conn, merge_id=merge_id, tenant_id="tenant-A",
        source="src", target="tgt", status="in_progress",
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO merge_post_commit_pending "
            "(pending_id, merge_id, tenant_id, kind, payload_json, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()), merge_id, "tenant-WRONG",
                "sse_event", "{}", "pending", _now_iso(),
            ),
        )


def test_merge_post_commit_pending_insert_trigger_accepts_tenant_match(tmp_path):
    """ trigger: INSERT with matching tenant_id succeeds."""
    store = _store(tmp_path)
    conn = store._get_conn()
    merge_id = str(uuid.uuid4())
    _insert_merge_audit(
        conn, merge_id=merge_id, tenant_id="tenant-A",
        source="src", target="tgt", status="in_progress",
    )
    conn.execute(
        "INSERT INTO merge_post_commit_pending "
        "(pending_id, merge_id, tenant_id, kind, payload_json, status, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()), merge_id, "tenant-A",
            "sse_event", "{}", "pending", _now_iso(),
        ),
    )


def test_merge_post_commit_pending_update_trigger_rejects_tenant_mutation(tmp_path):
    """ trigger (update variant): UPDATE that mutates tenant_id to a
    mismatching value raises.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    merge_id = str(uuid.uuid4())
    pending_id = str(uuid.uuid4())
    _insert_merge_audit(
        conn, merge_id=merge_id, tenant_id="tenant-A",
        source="src", target="tgt", status="in_progress",
    )
    conn.execute(
        "INSERT INTO merge_post_commit_pending "
        "(pending_id, merge_id, tenant_id, kind, payload_json, status, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            pending_id, merge_id, "tenant-A",
            "sse_event", "{}", "pending", _now_iso(),
        ),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE merge_post_commit_pending SET tenant_id = ? WHERE pending_id = ?",
            ("tenant-WRONG", pending_id),
        )


def test_merge_post_commit_pending_update_trigger_admits_status_only_update(tmp_path):
    """ trigger (update variant): status-only UPDATE does NOT fire the
    consistency check (the BEFORE UPDATE OF tenant_id event filter
    short-circuits). Validates that the trigger does not regress to
    firing on every UPDATE.
    """
    store = _store(tmp_path)
    conn = store._get_conn()
    merge_id = str(uuid.uuid4())
    pending_id = str(uuid.uuid4())
    _insert_merge_audit(
        conn, merge_id=merge_id, tenant_id="tenant-A",
        source="src", target="tgt", status="in_progress",
    )
    conn.execute(
        "INSERT INTO merge_post_commit_pending "
        "(pending_id, merge_id, tenant_id, kind, payload_json, status, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            pending_id, merge_id, "tenant-A",
            "sse_event", "{}", "pending", _now_iso(),
        ),
    )
    # status-only UPDATE: should succeed without firing the trigger.
    conn.execute(
        "UPDATE merge_post_commit_pending SET status = ?, attempts = attempts + 1 "
        "WHERE pending_id = ?",
        ("done", pending_id),
    )
    row = conn.execute(
        "SELECT status, attempts FROM merge_post_commit_pending WHERE pending_id = ?",
        (pending_id,),
    ).fetchone()
    assert row["status"] == "done"
    assert row["attempts"] == 1


# ---------------------------------------------------------------------------
# Idempotency: _ensure_schema() re-run is no-op
# ---------------------------------------------------------------------------

def test_ensure_schema_is_idempotent(tmp_path):
    """Re-running _ensure_schema() against an already-bootstrapped DB does
    not error and does not break existing schema. The IF NOT EXISTS guards
    + DROP TRIGGER IF EXISTS pattern make the bootstrap re-runnable.
    """
    store = _store(tmp_path)
    # Insert a few rows to make sure re-bootstrap doesn't drop them.
    conn = store._get_conn()
    _insert_conversation(conn, "tenant-A", "conv-keep", phase="active")
    _insert_merge_audit(
        conn, merge_id=str(uuid.uuid4()), tenant_id="tenant-A",
        source="src-keep", target="tgt-keep", status="committed",
    )
    # Re-run.
    store._ensure_schema()
    # Rows still present.
    assert conn.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = ?",
        ("conv-keep",),
    ).fetchone()
    assert conn.execute(
        "SELECT 1 FROM merge_audit WHERE source_conversation_id = ?",
        ("src-keep",),
    ).fetchone()
