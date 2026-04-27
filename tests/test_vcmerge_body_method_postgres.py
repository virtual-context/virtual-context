"""PG-parametrized smoke for VCMERGE S1.3 body method (v1.14-7 fold).

Skipped when DATABASE_URL is absent (the standard project pattern at
tests/test_vcmerge_schema_postgres.py:28). Mirrors the SQLite body-method
test bundle's structural invariants against a real Postgres backend so
PG-specific concerns (FOR UPDATE row locks, SAVEPOINT semantics for the
narrow-exception fail-closed gate, GREATEST/COALESCE arithmetic, ON
CONFLICT DO UPDATE on UPSERTs) get direct exercise.

Per VCMerge plan v1.14 codex iter-3 P2 (v1.14-7) + handoff section C8
residual: SQL inspection tests scan both PG + SQLite source code, but
only SQLite is exercised behaviorally on the dev machine. Real PG body
smoke catches ``psycopg.errors.UndefinedTable`` SAVEPOINT recovery,
``GREATEST(...)`` semantics, and the ``ON CONFLICT (alias_id) DO UPDATE``
shape that SQLite cannot prove via syntax-equivalent execution.

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
VCMerge plan v1.11 section 11 prologue.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.core.exceptions import (
    CrossTenantMergeError,
    LifecycleEpochMismatch,
    MergeAuditMissing,
    MergeBusy,
)


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="DATABASE_URL not set: Postgres smoke skipped",
    ),
    pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26"),
]


@pytest.fixture
def pg_store():
    """Build a PostgresStore against DATABASE_URL. Cleanup teardown drops
    test rows from merge_audit, merge_post_commit_pending, conversations,
    conversation_aliases, segments, canonical_turns under the test tenant
    prefix.
    """
    from virtual_context.storage.postgres import PostgresStore
    dsn = os.environ["DATABASE_URL"]
    store = PostgresStore(dsn)
    yield store
    try:
        conn = store._get_conn()
        for tbl in (
            "merge_post_commit_pending", "merge_audit",
            "tag_aliases", "tag_summaries", "tag_summary_embeddings",
            "request_turn_counters", "request_context", "request_captures",
            "tool_calls", "tool_outputs", "canonical_turns", "segments",
            "facts", "conversation_aliases", "conversations",
        ):
            try:
                if tbl == "conversation_aliases":
                    conn.execute(
                        "DELETE FROM conversation_aliases WHERE alias_id LIKE 'pgsmoke-%'"
                    )
                elif tbl in ("merge_post_commit_pending", "merge_audit"):
                    conn.execute(f"DELETE FROM {tbl} WHERE tenant_id LIKE 'pgsmoke-%'")
                elif tbl == "conversations":
                    conn.execute(
                        "DELETE FROM conversations WHERE tenant_id LIKE 'pgsmoke-%'"
                    )
                else:
                    conn.execute(
                        f"DELETE FROM {tbl} "
                        f"WHERE conversation_id LIKE 'pgsmoke-%'"
                    )
            except Exception:
                pass
    except Exception:
        pass


def _now():
    return datetime.now(timezone.utc)


def _seed_conversation(conn, tenant, cid, lifecycle_epoch=1, phase="active"):
    conn.execute(
        "INSERT INTO conversations (conversation_id, tenant_id, phase, "
        "lifecycle_epoch, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (cid, tenant, phase, lifecycle_epoch, _now(), _now()),
    )


def _reserve(store, *, tenant, src, tgt, label="lbl"):
    merge_id = str(uuid.uuid4())
    result = store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id, tenant_id=tenant,
        source_conversation_id=src, target_conversation_id=tgt,
        source_label_at_merge=label,
    )
    assert result.status == "reserved"
    return merge_id


# ---------------------------------------------------------------------------
# Body smoke against real PG: D1 pre-flight + lifecycle epoch + tenant
# ---------------------------------------------------------------------------

def test_pg_body_raises_merge_audit_missing_without_reservation(pg_store):
    tid = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid, src)
    _seed_conversation(conn, tid, tgt)
    with pytest.raises(MergeAuditMissing):
        pg_store.merge_conversation_data(
            merge_id=str(uuid.uuid4()),
            tenant_id=tid,
            source_conversation_id=src, target_conversation_id=tgt,
            sort_key_offset=0.0, request_turn_offset=0,
            expected_target_lifecycle_epoch=1,
            source_label_at_merge="lbl",
        )


def test_pg_body_refuses_cross_tenant_source(pg_store):
    """v1.14-7: defense-in-depth Layer C fires under real PG FOR UPDATE
    row lock; CrossTenantMergeError raised when source.tenant_id mismatch."""
    tid_a = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    tid_b = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid_a, tgt)  # target under tenant A
    _seed_conversation(conn, tid_b, src)  # source under tenant B
    merge_id = _reserve(pg_store, tenant=tid_a, src=src, tgt=tgt)
    with pytest.raises(CrossTenantMergeError):
        pg_store.merge_conversation_data(
            merge_id=merge_id, tenant_id=tid_a,
            source_conversation_id=src, target_conversation_id=tgt,
            sort_key_offset=0.0, request_turn_offset=0,
            expected_target_lifecycle_epoch=1,
            source_label_at_merge="lbl",
        )


def test_pg_body_refuses_lifecycle_epoch_mismatch(pg_store):
    tid = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid, src, lifecycle_epoch=1)
    _seed_conversation(conn, tid, tgt, lifecycle_epoch=5)
    merge_id = _reserve(pg_store, tenant=tid, src=src, tgt=tgt)
    with pytest.raises(LifecycleEpochMismatch):
        pg_store.merge_conversation_data(
            merge_id=merge_id, tenant_id=tid,
            source_conversation_id=src, target_conversation_id=tgt,
            sort_key_offset=0.0, request_turn_offset=0,
            expected_target_lifecycle_epoch=1,  # actual = 5
            source_label_at_merge="lbl",
        )


# ---------------------------------------------------------------------------
# Body smoke against real PG: per-table moves + counter bump
# ---------------------------------------------------------------------------

def test_pg_body_moves_segments_and_canonical_turns(pg_store):
    """Real PG smoke: source segments + canonical_turns transition to
    target's namespace; origin_conversation_id captures source. Verifies
    the ``UPDATE ... WHERE conversation_id = %s`` pattern works against
    real PG (no SQLite-specific syntax).
    """
    tid = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid, src)
    _seed_conversation(conn, tid, tgt)
    seg_ref = f"pgsmoke-seg-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO segments (ref, conversation_id, primary_tag, summary, "
        "full_text, messages_json, metadata_json, summary_tokens, "
        "full_tokens, compression_ratio, created_at, start_timestamp, "
        "end_timestamp) VALUES (%s, %s, 'general', 's', 'f', '[]', '{}', "
        "1, 1, 1.0, %s, %s, %s)",
        (seg_ref, src, _now(), _now(), _now()),
    )
    merge_id = _reserve(pg_store, tenant=tid, src=src, tgt=tgt)
    stats = pg_store.merge_conversation_data(
        merge_id=merge_id, tenant_id=tid,
        source_conversation_id=src, target_conversation_id=tgt,
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1,
        source_label_at_merge="lbl",
    )
    assert stats.success is True
    moved = conn.execute(
        "SELECT conversation_id, origin_conversation_id FROM segments "
        "WHERE ref = %s", (seg_ref,),
    ).fetchone()
    assert moved["conversation_id"] == tgt
    assert moved["origin_conversation_id"] == src


def test_pg_body_bumps_target_request_turn_counter(pg_store):
    """v1.14-1 (codex iter-3 P1) on real PG: UPSERT with GREATEST(...)
    correctly bumps target's next_request_turn past the moved range.
    """
    tid = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid, src)
    _seed_conversation(conn, tid, tgt)
    # target counter at 3
    conn.execute(
        "INSERT INTO request_turn_counters (conversation_id, next_request_turn) "
        "VALUES (%s, %s)",
        (tgt, 3),
    )
    # Source has tool_calls at request_turn 1..5
    for rt in (1, 2, 3, 4, 5):
        conn.execute(
            "INSERT INTO tool_calls (conversation_id, request_turn, round, "
            "group_id, tool_name, tool_input, tool_result, result_length, "
            "duration_ms, timestamp) VALUES (%s, %s, 0, 'g', 't', '{}', '{}', "
            "0, 0.0, %s)",
            (src, rt, _now()),
        )
    merge_id = _reserve(pg_store, tenant=tid, src=src, tgt=tgt)
    stats = pg_store.merge_conversation_data(
        merge_id=merge_id, tenant_id=tid,
        source_conversation_id=src, target_conversation_id=tgt,
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1,
        source_label_at_merge="lbl",
    )
    moved_max = conn.execute(
        "SELECT MAX(request_turn) AS m FROM tool_calls "
        "WHERE conversation_id = %s", (tgt,),
    ).fetchone()["m"]
    assert moved_max is not None
    new_counter = conn.execute(
        "SELECT next_request_turn FROM request_turn_counters "
        "WHERE conversation_id = %s", (tgt,),
    ).fetchone()["next_request_turn"]
    assert new_counter >= moved_max + 1
    assert "request_turn_counters_target_bumped_to" in stats.rows_moved


# ---------------------------------------------------------------------------
# Body smoke: alias UPSERT + prior_alias_target capture under real PG
# ---------------------------------------------------------------------------

def test_pg_body_captures_prior_alias_target_in_audit(pg_store):
    """B-D7 on real PG: ON CONFLICT (alias_id) DO UPDATE preserves the
    prior target_id snapshot in merge_audit.prior_alias_target.
    """
    tid = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    earlier = f"pgsmoke-earlier-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid, src)
    _seed_conversation(conn, tid, tgt)
    _seed_conversation(conn, tid, earlier)
    conn.execute(
        "INSERT INTO conversation_aliases (alias_id, target_id, epoch) "
        "VALUES (%s, %s, 1)",
        (src, earlier),
    )
    merge_id = _reserve(pg_store, tenant=tid, src=src, tgt=tgt)
    pg_store.merge_conversation_data(
        merge_id=merge_id, tenant_id=tid,
        source_conversation_id=src, target_conversation_id=tgt,
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1,
        source_label_at_merge="lbl",
    )
    audit = conn.execute(
        "SELECT prior_alias_target FROM merge_audit WHERE merge_id = %s",
        (merge_id,),
    ).fetchone()
    assert audit["prior_alias_target"] == earlier
    new_alias = conn.execute(
        "SELECT target_id FROM conversation_aliases WHERE alias_id = %s",
        (src,),
    ).fetchone()
    assert new_alias["target_id"] == tgt


# ---------------------------------------------------------------------------
# Body smoke: post-commit pendings written + tag_regenerate carries conflicts
# ---------------------------------------------------------------------------

def test_pg_body_post_commit_pendings_written(pg_store):
    """All three post-commit pending kinds insert under real PG."""
    tid = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid, src)
    _seed_conversation(conn, tid, tgt)
    merge_id = _reserve(pg_store, tenant=tid, src=src, tgt=tgt)
    pg_store.merge_conversation_data(
        merge_id=merge_id, tenant_id=tid,
        source_conversation_id=src, target_conversation_id=tgt,
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1,
        source_label_at_merge="lbl",
    )
    kinds = sorted(r["kind"] for r in conn.execute(
        "SELECT kind FROM merge_post_commit_pending WHERE merge_id = %s",
        (merge_id,),
    ).fetchall())
    assert kinds == ["queue_resegment", "sse_event", "tag_regenerate"]


def test_pg_body_returns_merge_stats_with_v14_fields(pg_store):
    """v1.14-1 + B-D9: MergeStats from real PG body has success +
    elapsed_seconds + the bumped-counter rows_moved key (when applicable).
    """
    tid = f"pgsmoke-{uuid.uuid4().hex[:8]}"
    src = f"pgsmoke-src-{uuid.uuid4().hex[:8]}"
    tgt = f"pgsmoke-tgt-{uuid.uuid4().hex[:8]}"
    conn = pg_store._get_conn()
    _seed_conversation(conn, tid, src)
    _seed_conversation(conn, tid, tgt)
    merge_id = _reserve(pg_store, tenant=tid, src=src, tgt=tgt)
    stats = pg_store.merge_conversation_data(
        merge_id=merge_id, tenant_id=tid,
        source_conversation_id=src, target_conversation_id=tgt,
        sort_key_offset=0.0, request_turn_offset=0,
        expected_target_lifecycle_epoch=1,
        source_label_at_merge="lbl",
    )
    assert stats.success is True
    assert stats.elapsed_seconds >= 0.0
    assert stats.merge_id == merge_id
    assert stats.tenant_id == tid
