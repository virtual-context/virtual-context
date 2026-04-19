"""Regression: once cleanup marks the operation 'abandoned', subsequent
write attempts by the stale worker must write zero rows (silently
rejected by the DB-layer guard).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timezone
from pathlib import Path

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, SegmentMetadata, StoredSegment, TagSummary

_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _seed_running(store, conv, op_id, worker="w"):
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (op_id, conv, now, now, worker, now),
        )


def _segment(conv, ref):
    return StoredSegment(
        ref=ref, conversation_id=conv, primary_tag="t", tags=["t"],
        summary="s", summary_tokens=10, full_text="f", full_tokens=20,
        messages=[], metadata=SegmentMetadata(), compaction_model="passthrough",
        compression_ratio=0.5, start_timestamp=_TS, end_timestamp=_TS,
    )


def test_store_segment_write_succeeds_while_operation_running(tmp_path: Path):
    store = SQLiteStore(tmp_path / "ok.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", worker="w")

    store.store_segment(
        _segment("c", "ref-1"),
        operation_id="op-1",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-1'"
        ).fetchone()[0]
    assert n == 1


def test_store_segment_raises_lease_lost_when_operation_abandoned(tmp_path: Path):
    """Rejection is NOT silent. The guard raises CompactionLeaseLost
    so the compactor pipeline catches it specifically, logs
    COMPACTION_WRITE_REJECTED, and exits cleanly instead of walking
    every remaining phase unaware.
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "abandon.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", worker="w")

    # Simulate takeover: cleanup flips dead op to 'abandoned'.
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-1'",
        )

    # Stale worker still trying to write — must raise.
    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.store_segment(
            _segment("c", "ref-ghost"),
            operation_id="op-1",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.operation_id == "op-1"
    assert exc_info.value.write_site == "store_segment"

    # And zero rows persisted.
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-ghost'"
        ).fetchone()[0]
    assert n == 0


def test_store_segment_raises_lease_lost_when_wrong_owner(tmp_path: Path):
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "wrongowner.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-1", worker="w-real")

    with pytest.raises(CompactionLeaseLost):
        store.store_segment(
            _segment("c", "ref-wrong"),
            operation_id="op-1",
            owner_worker_id="w-impostor",
            lifecycle_epoch=1,
        )
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-wrong'"
        ).fetchone()[0]
    assert n == 0


def test_store_segment_legacy_path_no_guard_kwargs_still_inserts(tmp_path: Path):
    """Callers that don't pass guard kwargs (pre-change tests,
    non-compaction write sites) keep the unconditional INSERT path.
    Otherwise we'd break all existing test harnesses that construct
    rows directly.
    """
    store = SQLiteStore(tmp_path / "legacy.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    ref = store.store_segment(_segment("c", "ref-legacy"))
    assert ref == "ref-legacy"
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE ref='ref-legacy'"
        ).fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# Task 11: per-write ownership guard on store_facts
# ---------------------------------------------------------------------------

def _fact(fact_id: str, conv: str = "c") -> Fact:
    return Fact(
        id=fact_id,
        subject="user",
        verb="likes",
        object="coffee",
        status="active",
        conversation_id=conv,
        tags=["preferences"],
    )


def test_store_facts_write_succeeds_while_operation_running(tmp_path: Path):
    """Batch insert with guard kwargs succeeds when op is still running."""
    store = SQLiteStore(tmp_path / "facts_ok.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-facts-1", worker="w")

    inserted = store.store_facts(
        [_fact("f-1"), _fact("f-2")],
        operation_id="op-facts-1",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )
    assert inserted == 2

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE conversation_id='c'"
        ).fetchone()[0]
    assert n == 2


def test_store_facts_raises_lease_lost_when_operation_abandoned(tmp_path: Path):
    """Guard raises CompactionLeaseLost with write_site=='store_facts'
    when the operation has been marked abandoned before the batch insert.

    Note: this test simulates the at-rest correctness case (operation
    abandoned before the call). The guard is checked per-row via
    INSERT-SELECT, so a concurrent takeover mid-batch may let the
    in-flight transaction complete (documented trade-off: the up-front
    per-row DB check fires on each INSERT but the same transaction
    holds the write lock for the whole batch).
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "facts_abandon.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-facts-2", worker="w")

    # Simulate takeover: mark op as abandoned.
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-facts-2'",
        )

    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.store_facts(
            [_fact("f-ghost-1"), _fact("f-ghost-2")],
            operation_id="op-facts-2",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.operation_id == "op-facts-2"
    assert exc_info.value.write_site == "store_facts"

    # Zero rows persisted.
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE conversation_id='c'"
        ).fetchone()[0]
    assert n == 0


def test_store_facts_legacy_path_no_guard_kwargs_still_inserts(tmp_path: Path):
    """Callers that omit guard kwargs use the unconditional INSERT path.
    Existing test harnesses and non-compaction call sites must not break.
    """
    store = SQLiteStore(tmp_path / "facts_legacy.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    inserted = store.store_facts([_fact("f-legacy-1"), _fact("f-legacy-2")])
    assert inserted == 2

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE conversation_id='c'"
        ).fetchone()[0]
    assert n == 2


# ---------------------------------------------------------------------------
# Task 12: per-write ownership guard on save_tag_summary
# ---------------------------------------------------------------------------

def _tag_summary(tag: str) -> TagSummary:
    return TagSummary(
        tag=tag,
        summary="Summary of " + tag,
        summary_tokens=10,
        source_segment_refs=["seg-1"],
        source_turn_numbers=[1],
        covers_through_turn=1,
    )


def test_save_tag_summary_write_succeeds_while_operation_running(tmp_path: Path):
    """UPSERT with guard kwargs succeeds when the operation is still running."""
    store = SQLiteStore(tmp_path / "tag_summary_ok.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-ts-1", worker="w")

    store.save_tag_summary(
        _tag_summary("python"),
        conversation_id="c",
        operation_id="op-ts-1",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM tag_summaries WHERE tag='python' AND conversation_id='c'"
        ).fetchone()[0]
    assert n == 1


def test_save_tag_summary_raises_lease_lost_when_operation_abandoned(tmp_path: Path):
    """Guard raises CompactionLeaseLost with write_site=='save_tag_summary'
    when the operation has been marked abandoned before the write.
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "tag_summary_abandon.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-ts-2", worker="w")

    # Simulate takeover: mark op as abandoned.
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-ts-2'",
        )

    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.save_tag_summary(
            _tag_summary("python-ghost"),
            conversation_id="c",
            operation_id="op-ts-2",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.operation_id == "op-ts-2"
    assert exc_info.value.write_site == "save_tag_summary"

    # Zero rows persisted.
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM tag_summaries WHERE tag='python-ghost'"
        ).fetchone()[0]
    assert n == 0


def test_save_tag_summary_legacy_path_no_guard_kwargs_still_inserts(tmp_path: Path):
    """Callers that omit guard kwargs use the unconditional UPSERT path.
    Existing test harnesses and non-compaction call sites must not break.
    """
    store = SQLiteStore(tmp_path / "tag_summary_legacy.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    store.save_tag_summary(_tag_summary("legacy-tag"), conversation_id="c")

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM tag_summaries WHERE tag='legacy-tag' AND conversation_id='c'"
        ).fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# Task 13: per-write ownership guard on store_tag_summary_embedding
# ---------------------------------------------------------------------------


def test_store_tag_summary_embedding_write_succeeds_while_running(tmp_path: Path):
    """Embedding UPSERT with guard kwargs succeeds when the operation is still running."""
    store = SQLiteStore(tmp_path / "emb_ok.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-emb-1", worker="w")

    store.store_tag_summary_embedding(
        "python", "c", [0.1, 0.2, 0.3],
        operation_id="op-emb-1",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM tag_summary_embeddings "
            "WHERE tag='python' AND conversation_id='c'"
        ).fetchone()[0]
    assert n == 1


def test_store_tag_summary_embedding_raises_lease_lost_when_operation_abandoned(tmp_path: Path):
    """Guard raises CompactionLeaseLost(write_site='store_tag_summary_embedding')
    when the operation has been marked abandoned before the write.
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "emb_abandon.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-emb-2", worker="w")

    # Simulate takeover: mark op as abandoned.
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-emb-2'",
        )

    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.store_tag_summary_embedding(
            "python-ghost", "c", [0.9, 0.8, 0.7],
            operation_id="op-emb-2",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.operation_id == "op-emb-2"
    assert exc_info.value.write_site == "store_tag_summary_embedding"

    # Zero rows persisted.
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM tag_summary_embeddings "
            "WHERE tag='python-ghost'"
        ).fetchone()[0]
    assert n == 0


def test_store_tag_summary_embedding_legacy_path_no_guard_kwargs_still_inserts(tmp_path: Path):
    """Callers that omit guard kwargs use the unconditional UPSERT path.
    Existing test harnesses and non-compaction call sites must not break.
    """
    store = SQLiteStore(tmp_path / "emb_legacy.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    store.store_tag_summary_embedding("legacy-tag", "c", [1.0, 2.0, 3.0])

    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM tag_summary_embeddings "
            "WHERE tag='legacy-tag' AND conversation_id='c'"
        ).fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# Task 14: per-write ownership guard on mark_canonical_turns_compacted
# ---------------------------------------------------------------------------

def _seed_split_turn_group_for_guard(
    store: SQLiteStore,
    conv: str,
    turn_group_number: int,
    *,
    u_id: str,
    a_id: str,
) -> None:
    """Insert user-half and assistant-half rows for a single turn_group."""
    from virtual_context.core.canonical_turns import utcnow_iso
    now = utcnow_iso()
    store.save_canonical_turn(
        conv, -1,
        f"user tg{turn_group_number}", "",
        turn_group_number=turn_group_number,
        canonical_turn_id=u_id,
        sort_key=float((turn_group_number * 2 + 1) * 1000),
        turn_hash=f"h-{u_id}",
        hash_version=1,
        tagged_at=now,
        created_at=now, updated_at=now,
        first_seen_at=now, last_seen_at=now,
    )
    store.save_canonical_turn(
        conv, -1,
        "", f"assistant tg{turn_group_number}",
        turn_group_number=turn_group_number,
        canonical_turn_id=a_id,
        sort_key=float((turn_group_number * 2 + 2) * 1000),
        turn_hash=f"h-{a_id}",
        hash_version=1,
        tagged_at=now,
        created_at=now, updated_at=now,
        first_seen_at=now, last_seen_at=now,
    )


def test_mark_canonical_turns_compacted_guarded_path_preserves_turn_group_merge_expansion(
    tmp_path: Path,
) -> None:
    """Guarded path marks BOTH halves of each turn_group AND stamps
    compaction_operation_id on each row, preserving the merge-expansion
    behaviour from 6e2d5bd.
    """
    store = SQLiteStore(tmp_path / "mark_guard_ok.db")
    conv = "conv-mark-guard"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    _seed_running(store, conv, "op-mark-1", worker="w")

    _seed_split_turn_group_for_guard(store, conv, 0, u_id="ct-U0", a_id="ct-A0")

    # Pass the user-half id only — the turn_group expansion must also hit ct-A0.
    marked = store.mark_canonical_turns_compacted(
        conv,
        ["ct-U0"],
        operation_id="op-mark-1",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )
    assert marked == 2, f"expected both halves marked; got {marked}"

    with store._get_conn() as conn:
        rows = conn.execute(
            "SELECT canonical_turn_id, compacted_at, compaction_operation_id "
            "FROM canonical_turns WHERE conversation_id=? ORDER BY sort_key",
            (conv,),
        ).fetchall()

    assert len(rows) == 2
    for row in rows:
        assert row[1] is not None, f"{row[0]}: compacted_at should be set"
        assert row[2] == "op-mark-1", (
            f"{row[0]}: compaction_operation_id should be 'op-mark-1', got {row[2]}"
        )


def test_mark_canonical_turns_compacted_raises_lease_lost_when_operation_abandoned(
    tmp_path: Path,
) -> None:
    """Guard raises CompactionLeaseLost(write_site='mark_canonical_turns_compacted')
    when the operation has been marked abandoned before the UPDATE.
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "mark_guard_abandon.db")
    conv = "conv-mark-abandon"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)
    _seed_running(store, conv, "op-mark-2", worker="w")

    _seed_split_turn_group_for_guard(store, conv, 0, u_id="ct-U0g", a_id="ct-A0g")

    # Simulate takeover: mark op as abandoned.
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-mark-2'",
        )

    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.mark_canonical_turns_compacted(
            conv,
            ["ct-U0g"],
            operation_id="op-mark-2",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.operation_id == "op-mark-2"
    assert exc_info.value.write_site == "mark_canonical_turns_compacted"

    # Zero rows should have been marked.
    with store._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM canonical_turns "
            "WHERE conversation_id=? AND compacted_at IS NOT NULL",
            (conv,),
        ).fetchone()[0]
    assert n == 0, f"expected 0 rows marked after lease lost; got {n}"


def test_mark_canonical_turns_compacted_legacy_path_no_guard_kwargs_still_marks(
    tmp_path: Path,
) -> None:
    """Regression: calling without guard kwargs uses the legacy path,
    which still expands through turn_group_number (behaviour from 6e2d5bd).
    Existing call sites and test harnesses must not break.
    """
    store = SQLiteStore(tmp_path / "mark_legacy.db")
    conv = "conv-mark-legacy"
    store.upsert_conversation(tenant_id="t", conversation_id=conv)

    _seed_split_turn_group_for_guard(store, conv, 0, u_id="ct-UL0", a_id="ct-AL0")

    # No guard kwargs — legacy path.
    marked = store.mark_canonical_turns_compacted(conv, ["ct-UL0"])
    assert marked == 2, f"legacy path should still mark both halves; got {marked}"

    with store._get_conn() as conn:
        rows = conn.execute(
            "SELECT canonical_turn_id, compacted_at FROM canonical_turns "
            "WHERE conversation_id=? ORDER BY sort_key",
            (conv,),
        ).fetchall()
    for row in rows:
        assert row[1] is not None, f"{row[0]}: legacy path should set compacted_at"


# ---------------------------------------------------------------------------
# P1 #1 regression: start_compaction_operation uses caller-supplied id
# ---------------------------------------------------------------------------

def test_start_compaction_operation_uses_caller_supplied_operation_id(tmp_path: Path):
    """The DB row PK must equal the id supplied by the caller.

    Before the fix, start_compaction_operation always generated its own UUID,
    causing the per-write guard kwargs threaded by _run_compact to match zero
    rows and raising CompactionLeaseLost on every normal compaction.
    """
    store = SQLiteStore(tmp_path / "caller_id.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    caller_id = "my-fixed-op-id-123"
    returned_id = store.start_compaction_operation(
        conversation_id="c",
        lifecycle_epoch=1,
        worker_id="w",
        phase_count=7,
        phase_name="init",
        operation_id=caller_id,
    )

    # The return value must equal the caller's id.
    assert returned_id == caller_id, (
        f"expected returned_id={caller_id!r}, got {returned_id!r}"
    )

    # The DB row must be keyed by the caller's id.
    with store._get_conn() as conn:
        row = conn.execute(
            "SELECT operation_id FROM compaction_operation WHERE operation_id = ?",
            (caller_id,),
        ).fetchone()
    assert row is not None, "DB row not found for caller-supplied operation_id"
    assert row[0] == caller_id


def test_start_compaction_operation_auto_generates_id_when_none_supplied(tmp_path: Path):
    """Legacy path: when operation_id is not supplied, a UUID is auto-generated."""
    store = SQLiteStore(tmp_path / "auto_id.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    returned_id = store.start_compaction_operation(
        conversation_id="c",
        lifecycle_epoch=1,
        worker_id="w",
        phase_count=7,
        phase_name="init",
    )

    assert returned_id is not None
    assert len(returned_id) > 0
    with store._get_conn() as conn:
        row = conn.execute(
            "SELECT operation_id FROM compaction_operation WHERE operation_id = ?",
            (returned_id,),
        ).fetchone()
    assert row is not None, "DB row not found for auto-generated operation_id"


# ---------------------------------------------------------------------------
# P1 #2 regression: replace_facts_for_segment atomicity
# ---------------------------------------------------------------------------

def test_replace_facts_for_segment_does_not_lose_preexisting_facts_on_lease_lost(
    tmp_path: Path,
) -> None:
    """P1 regression: DELETE of pre-existing facts must roll back when
    the guarded INSERT raises CompactionLeaseLost. Otherwise a stale
    worker wipes facts it had no authority to modify.
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "replace_race.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")

    # Seed pre-existing facts attributed to segment_ref=seg-1 via the
    # legacy (unguarded) path so they exist regardless of operation state.
    pre_fact = _fact("pre-f1", conv="c")
    pre_fact.segment_ref = "seg-1"
    store.store_facts([pre_fact])

    # Verify pre-existing facts are present.
    with store._get_conn() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE conversation_id='c' AND segment_ref='seg-1'"
        ).fetchone()[0]
    assert before == 1, f"expected 1 pre-existing fact; got {before}"

    # Seed a running compaction_operation and immediately abandon it to
    # simulate a stale worker holding the id after a takeover.
    _seed_running(store, "c", "op-race-1", worker="w")
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-race-1'",
        )

    new_fact = _fact("new-f1", conv="c")
    new_fact.segment_ref = "seg-1"

    # Call must raise CompactionLeaseLost.
    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.replace_facts_for_segment(
            "c", "seg-1", [new_fact],
            operation_id="op-race-1",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.write_site == "replace_facts_for_segment"

    # Pre-existing facts must still be present (DELETE was rolled back).
    with store._get_conn() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE conversation_id='c' AND segment_ref='seg-1'"
        ).fetchone()[0]
    assert after == 1, (
        f"Pre-existing facts were deleted despite CompactionLeaseLost — "
        f"expected 1 row, got {after}"
    )


# ---------------------------------------------------------------------------
# update_segment ownership guard
# ---------------------------------------------------------------------------

def test_update_segment_succeeds_while_operation_running(tmp_path: Path) -> None:
    """update_segment with guard kwargs succeeds when the operation is running."""
    store = SQLiteStore(tmp_path / "update_seg_ok.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-us-1", worker="w")

    # First store the segment so it exists.
    store.store_segment(_segment("c", "seg-upd-1"))

    # Now update it with guard kwargs — must succeed.
    updated = _segment("c", "seg-upd-1")
    updated.summary = "updated summary"
    store.update_segment(
        updated,
        operation_id="op-us-1",
        owner_worker_id="w",
        lifecycle_epoch=1,
    )

    with store._get_conn() as conn:
        row = conn.execute(
            "SELECT summary FROM segments WHERE ref='seg-upd-1'"
        ).fetchone()
    assert row is not None
    assert row[0] == "updated summary"


def test_update_segment_raises_lease_lost_when_operation_abandoned(tmp_path: Path) -> None:
    """update_segment with guard kwargs raises CompactionLeaseLost when the
    operation has been abandoned — write_site must be 'update_segment'.
    """
    import pytest
    from virtual_context.types import CompactionLeaseLost

    store = SQLiteStore(tmp_path / "update_seg_abandon.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_running(store, "c", "op-us-2", worker="w")

    # Store the segment first (legacy path, no guard).
    store.store_segment(_segment("c", "seg-upd-2"))

    # Abandon the operation.
    with store._get_conn() as conn:
        conn.execute(
            "UPDATE compaction_operation SET status='abandoned' "
            "WHERE operation_id='op-us-2'",
        )

    with pytest.raises(CompactionLeaseLost) as exc_info:
        store.update_segment(
            _segment("c", "seg-upd-2"),
            operation_id="op-us-2",
            owner_worker_id="w",
            lifecycle_epoch=1,
        )
    assert exc_info.value.operation_id == "op-us-2"
    assert exc_info.value.write_site == "update_segment"
