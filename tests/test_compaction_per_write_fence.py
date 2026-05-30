"""Compaction-fence Phase 3 tests for per-write fence on the five
methods plus facts.operation_id stamping on store_facts and
replace_facts_for_segment.

Covers fencing plan §5.7 Phase 3 SQLite scenarios:

* T3.1-T3.5: each fenced method writes successfully when guard kwargs
  match a running compaction_operation row.
* T3.6-T3.10: each fenced method raises CompactionLeaseLost when the
  guard does not match (mismatched operation_id).
* T3.11: set_fact_superseded rejects when old and new facts belong to
  different conversations (both-endpoint validation).
* T3.12: store_fact_links rejects when endpoint facts belong to
  different conversations.
* T3.13: set_fact_superseded rejects when the supplied conversation_id
  does not match the active op's conversation (probed indirectly via
  the fact's conversation_id).
* T3.14: legacy callers (all guard kwargs None) succeed without the
  guard for backward compatibility.
* T3.15: store_chunk_embeddings rejects a segment_ref from another
  conversation even when the caller supplies a valid active operation
  id for its own conversation (segments JOIN, not trusting caller).
* T3.16: store_fact_links inserts all required schema columns
  (id, endpoints, relation metadata, operation_id) and an ON CONFLICT
  on a pre-existing id is a no-op that does not stamp the current
  operation id.
* T3.19: mixed partial guard kwargs are rejected with ValueError
  (programming error, not a fence rejection).
* T3.20: guarded store_facts stamps facts.operation_id on insert and
  conflict replace; legacy all-None calls do not stamp a new operation id.
* T3.21: guarded replace_facts_for_segment stamps facts.operation_id
  on inserted replacement facts; a guard mismatch rolls back the
  DELETE rather than leaving the segment factless.

State-side caller-wiring tests (T3.17-T3.18) and PG smoke tests
(T3.22-T3.24) are deferred outside this patch.
"""

from __future__ import annotations

import tempfile
import uuid

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    ChunkEmbedding,
    CompactionLeaseLost,
    Fact,
    FactLink,
)


def _make_store() -> SQLiteStore:
    """Per-write fence tests pin ACTIVE mode so the helper raises
    CompactionLeaseLost on guard mismatch. The default OFF mode would
    silently absorb the rejection per the P7 rollout discipline."""
    from virtual_context.core.compaction_fence import CompactionFenceMode
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return SQLiteStore(
        handle.name,
        compaction_fence_mode=CompactionFenceMode.ACTIVE,
    )


_TS = "2026-05-29T00:00:00+00:00"


def _seed_running_op(
    store: SQLiteStore,
    conv_id: str,
    *,
    op_id: str,
    worker_id: str,
    epoch: int = 1,
) -> None:
    """Seed a conversation + lifecycle row + a running compaction_operation.

    Mirrors the post-begin_compaction state immediately before the
    compactor enters the per-write phase.
    """
    conn = store._get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv_id, _TS),
    )
    conn.execute(
        "INSERT OR IGNORE INTO conversations "
        "(conversation_id, tenant_id, phase, lifecycle_epoch, "
        " created_at, updated_at) "
        "VALUES (?, 't1', 'compacting', ?, ?, ?)",
        (conv_id, epoch, _TS, _TS),
    )
    conn.execute(
        """INSERT INTO compaction_operation
           (operation_id, conversation_id, lifecycle_epoch,
            phase_index, phase_count, phase_name, status,
            started_at, heartbeat_ts, owner_worker_id, created_at)
           VALUES (?, ?, ?, 0, 7, 'starting', 'running',
                   ?, ?, ?, ?)""",
        (op_id, conv_id, epoch, _TS, _TS, worker_id, _TS),
    )
    conn.commit()


def _seed_segment(store: SQLiteStore, ref: str, conv_id: str) -> None:
    conn = store._get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO segments
           (ref, conversation_id, primary_tag, created_at,
            start_timestamp, end_timestamp)
           VALUES (?, ?, '_general', ?, ?, ?)""",
        (ref, conv_id, _TS, _TS, _TS),
    )
    conn.commit()


def _seed_fact(
    store: SQLiteStore,
    fact_id: str,
    conv_id: str,
    *,
    subject: str = "alice",
    verb: str = "likes",
    obj: str = "tea",
) -> None:
    conn = store._get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO facts
           (id, subject, verb, object, status, what, who, when_date,
            "where", why, fact_type, tags_json, segment_ref,
            conversation_id, turn_numbers_json, mentioned_at,
            session_date)
           VALUES (?, ?, ?, ?, 'active', '', '', '', '', '', 'personal',
                   '[]', '', ?, '[]', ?, '')""",
        (fact_id, subject, verb, obj, conv_id, _TS),
    )
    conn.commit()


def _fact_operation_id(store: SQLiteStore, fact_id: str) -> str | None:
    conn = store._get_conn()
    row = conn.execute(
        "SELECT operation_id FROM facts WHERE id = ?",
        (fact_id,),
    ).fetchone()
    return row[0] if row else None


def _link_operation_id(store: SQLiteStore, link_id: str) -> str | None:
    conn = store._get_conn()
    row = conn.execute(
        "SELECT operation_id FROM fact_links WHERE id = ?",
        (link_id,),
    ).fetchone()
    return row[0] if row else None


def _chunk_operation_id(
    store: SQLiteStore, segment_ref: str, chunk_index: int,
) -> str | None:
    conn = store._get_conn()
    row = conn.execute(
        """SELECT operation_id FROM segment_chunks
            WHERE segment_ref = ? AND chunk_index = ?""",
        (segment_ref, chunk_index),
    ).fetchone()
    return row[0] if row else None


def _tool_link_operation_id(
    store: SQLiteStore, conv_id: str, seg_ref: str, tool_ref: str,
) -> str | None:
    conn = store._get_conn()
    row = conn.execute(
        """SELECT operation_id FROM segment_tool_outputs
            WHERE conversation_id = ?
              AND segment_ref = ?
              AND tool_output_ref = ?""",
        (conv_id, seg_ref, tool_ref),
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# T3.1-T3.5: happy path per fenced method
# ---------------------------------------------------------------------------


class TestT31_StoreChunkEmbeddingsHappy:
    def test_matching_guard_inserts_and_stamps_operation_id(self):
        store = _make_store()
        conv = "conv-t3-1"
        op = uuid.uuid4().hex
        worker = "w-1"
        seg_ref = "seg-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        _seed_segment(store, seg_ref, conv)

        store.store_chunk_embeddings(
            seg_ref,
            [ChunkEmbedding(segment_ref=seg_ref, chunk_index=0,
                            text="hello", embedding=[0.1, 0.2])],
            operation_id=op, owner_worker_id=worker,
            lifecycle_epoch=1, conversation_id=conv,
        )
        assert _chunk_operation_id(store, seg_ref, 0) == op


class TestT32_LinkSegmentToolOutputHappy:
    def test_matching_guard_inserts_and_stamps_operation_id(self):
        store = _make_store()
        conv = "conv-t3-2"
        op = uuid.uuid4().hex
        worker = "w-1"
        seg_ref = "seg-2"
        tool_ref = "tool-2"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)

        store.link_segment_tool_output(
            conv, seg_ref, tool_ref,
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )
        assert _tool_link_operation_id(store, conv, seg_ref, tool_ref) == op


class TestT33_StoreFactLinksHappy:
    def test_matching_guard_inserts_and_stamps_operation_id(self):
        store = _make_store()
        conv = "conv-t3-3"
        op = uuid.uuid4().hex
        worker = "w-1"
        src_fact = "fact-src"
        tgt_fact = "fact-tgt"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        _seed_fact(store, src_fact, conv)
        _seed_fact(store, tgt_fact, conv)

        link = FactLink(
            source_fact_id=src_fact, target_fact_id=tgt_fact,
            relation_type="supersedes", confidence=1.0, context="t",
            created_by="compaction",
        )
        count = store.store_fact_links(
            [link],
            operation_id=op, owner_worker_id=worker,
            lifecycle_epoch=1, conversation_id=conv,
        )
        assert count == 1
        assert _link_operation_id(store, link.id) == op


class TestT34_SetFactSupersededHappy:
    def test_matching_guard_updates_superseded_by(self):
        store = _make_store()
        conv = "conv-t3-4"
        op = uuid.uuid4().hex
        worker = "w-1"
        old_id = "fact-old"
        new_id = "fact-new"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        _seed_fact(store, old_id, conv)
        _seed_fact(store, new_id, conv)

        store.set_fact_superseded(
            old_id, new_id,
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )
        conn = store._get_conn()
        row = conn.execute(
            "SELECT superseded_by FROM facts WHERE id = ?", (old_id,),
        ).fetchone()
        assert row[0] == new_id


class TestT35_UpdateFactFieldsHappy:
    def test_matching_guard_updates_fields(self):
        store = _make_store()
        conv = "conv-t3-5"
        op = uuid.uuid4().hex
        worker = "w-1"
        fact_id = "fact-upd"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        _seed_fact(store, fact_id, conv)

        store.update_fact_fields(
            fact_id, "knows", "python", "completed", "learned syntax",
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )
        conn = store._get_conn()
        row = conn.execute(
            "SELECT verb, object, status, what FROM facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        assert tuple(row) == ("knows", "python", "completed", "learned syntax")


# ---------------------------------------------------------------------------
# T3.6-T3.10: mismatched op_id raises CompactionLeaseLost
# ---------------------------------------------------------------------------


class TestT36_StoreChunkEmbeddingsMismatch:
    def test_mismatched_op_raises_lease_lost(self):
        store = _make_store()
        conv = "conv-t3-6"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        seg_ref = "seg-6"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        _seed_segment(store, seg_ref, conv)

        with pytest.raises(CompactionLeaseLost) as exc:
            store.store_chunk_embeddings(
                seg_ref,
                [ChunkEmbedding(segment_ref=seg_ref, chunk_index=0,
                                text="x", embedding=[0.0])],
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1, conversation_id=conv,
            )
        assert exc.value.write_site == "store_chunk_embeddings"
        # No chunks should have been inserted.
        conn = store._get_conn()
        n = conn.execute(
            "SELECT COUNT(*) FROM segment_chunks WHERE segment_ref = ?",
            (seg_ref,),
        ).fetchone()[0]
        assert n == 0


class TestT37_LinkSegmentToolOutputMismatch:
    def test_mismatched_op_raises_lease_lost(self):
        store = _make_store()
        conv = "conv-t3-7"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)

        with pytest.raises(CompactionLeaseLost) as exc:
            store.link_segment_tool_output(
                conv, "seg-7", "tool-7",
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1,
            )
        assert exc.value.write_site == "link_segment_tool_output"


class TestT38_StoreFactLinksMismatch:
    def test_mismatched_op_raises_lease_lost(self):
        store = _make_store()
        conv = "conv-t3-8"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        _seed_fact(store, "fa", conv)
        _seed_fact(store, "fb", conv)
        link = FactLink(
            source_fact_id="fa", target_fact_id="fb",
            relation_type="r", confidence=1.0, context="c",
            created_by="compaction",
        )
        with pytest.raises(CompactionLeaseLost) as exc:
            store.store_fact_links(
                [link],
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1, conversation_id=conv,
            )
        assert exc.value.write_site == "store_fact_links"


class TestT39_SetFactSupersededMismatch:
    def test_mismatched_op_raises_lease_lost(self):
        store = _make_store()
        conv = "conv-t3-9"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        _seed_fact(store, "f-old", conv)
        _seed_fact(store, "f-new", conv)
        with pytest.raises(CompactionLeaseLost) as exc:
            store.set_fact_superseded(
                "f-old", "f-new",
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1,
            )
        assert exc.value.write_site == "set_fact_superseded"


class TestT310_UpdateFactFieldsMismatch:
    def test_mismatched_op_raises_lease_lost(self):
        store = _make_store()
        conv = "conv-t3-10"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        _seed_fact(store, "f-x", conv)
        with pytest.raises(CompactionLeaseLost) as exc:
            store.update_fact_fields(
                "f-x", "v", "o", "active", "what",
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1,
            )
        assert exc.value.write_site == "update_fact_fields"


# ---------------------------------------------------------------------------
# T3.11: set_fact_superseded rejects when old/new facts belong to
# different conversations.
# ---------------------------------------------------------------------------


class TestT311_SetFactSupersededCrossConvReject:
    def test_endpoints_in_different_convs_raise_lease_lost(self):
        store = _make_store()
        conv_a = "conv-t3-11a"
        conv_b = "conv-t3-11b"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv_a, op_id=op, worker_id=worker)
        _seed_running_op(
            store, conv_b, op_id=uuid.uuid4().hex, worker_id="w-other",
        )
        _seed_fact(store, "fa", conv_a)
        _seed_fact(store, "fb", conv_b)
        with pytest.raises(CompactionLeaseLost):
            store.set_fact_superseded(
                "fa", "fb",
                operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
            )


# ---------------------------------------------------------------------------
# T3.12: store_fact_links rejects when endpoint facts belong to
# different conversations.
# ---------------------------------------------------------------------------


class TestT312_StoreFactLinksCrossConvReject:
    def test_endpoints_in_different_convs_raise_lease_lost(self):
        store = _make_store()
        conv_a = "conv-t3-12a"
        conv_b = "conv-t3-12b"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv_a, op_id=op, worker_id=worker)
        _seed_fact(store, "fa", conv_a)
        _seed_fact(store, "fb", conv_b)
        link = FactLink(
            source_fact_id="fa", target_fact_id="fb",
            relation_type="r", confidence=1.0, context="c",
            created_by="compaction",
        )
        with pytest.raises(CompactionLeaseLost):
            store.store_fact_links(
                [link],
                operation_id=op, owner_worker_id=worker,
                lifecycle_epoch=1, conversation_id=conv_a,
            )


# ---------------------------------------------------------------------------
# T3.13: set_fact_superseded rejects when both facts share a conv
# different from the active op's conversation.
# ---------------------------------------------------------------------------


class TestT313_SetFactSupersededWrongConvReject:
    def test_facts_in_other_conv_than_active_op_reject(self):
        store = _make_store()
        active_conv = "conv-t3-13-active"
        other_conv = "conv-t3-13-other"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, active_conv, op_id=op, worker_id=worker)
        # Both facts live in a different conversation. The active op
        # is on active_conv; the EXISTS clause requires
        # co.conversation_id = f_old.conversation_id, so the guard
        # rejects.
        _seed_fact(store, "fo1", other_conv)
        _seed_fact(store, "fo2", other_conv)
        with pytest.raises(CompactionLeaseLost):
            store.set_fact_superseded(
                "fo1", "fo2",
                operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
            )


# ---------------------------------------------------------------------------
# T3.14: legacy all guard kwargs None path still works for all fenced methods.
# ---------------------------------------------------------------------------


class TestT314_LegacyNoneKwargsBackwardCompat:
    def test_legacy_store_chunk_embeddings_writes_unguarded(self):
        store = _make_store()
        seg_ref = "seg-leg-1"
        store.store_chunk_embeddings(
            seg_ref,
            [ChunkEmbedding(segment_ref=seg_ref, chunk_index=0,
                            text="x", embedding=[1.0])],
        )
        conn = store._get_conn()
        op_id = conn.execute(
            "SELECT operation_id FROM segment_chunks WHERE segment_ref = ?",
            (seg_ref,),
        ).fetchone()[0]
        assert op_id is None

    def test_legacy_store_chunk_embeddings_all_guard_kwargs_none_allows_conversation_id(self):
        store = _make_store()
        seg_ref = "seg-leg-conv"
        store.store_chunk_embeddings(
            seg_ref,
            [ChunkEmbedding(segment_ref=seg_ref, chunk_index=0,
                            text="x", embedding=[1.0])],
            conversation_id="conv-leg",
        )
        assert _chunk_operation_id(store, seg_ref, 0) is None

    def test_legacy_link_segment_tool_output_writes_unguarded(self):
        store = _make_store()
        store.link_segment_tool_output("conv-leg", "seg-leg", "tool-leg")
        assert _tool_link_operation_id(
            store, "conv-leg", "seg-leg", "tool-leg",
        ) is None

    def test_legacy_store_fact_links_writes_unguarded(self):
        store = _make_store()
        _seed_fact(store, "fl-a", "any-conv")
        _seed_fact(store, "fl-b", "any-conv")
        link = FactLink(
            source_fact_id="fl-a", target_fact_id="fl-b",
            relation_type="r", confidence=1.0, context="",
            created_by="migration",
        )
        n = store.store_fact_links([link])
        assert n == 1
        assert _link_operation_id(store, link.id) is None

    def test_legacy_store_fact_links_all_guard_kwargs_none_allows_conversation_id(self):
        store = _make_store()
        _seed_fact(store, "fl-conv-a", "any-conv")
        _seed_fact(store, "fl-conv-b", "any-conv")
        link = FactLink(
            source_fact_id="fl-conv-a", target_fact_id="fl-conv-b",
            relation_type="r", confidence=1.0, context="",
            created_by="migration",
        )
        n = store.store_fact_links([link], conversation_id="any-conv")
        assert n == 1
        assert _link_operation_id(store, link.id) is None

    def test_legacy_set_fact_superseded_writes_unguarded(self):
        store = _make_store()
        _seed_fact(store, "fa", "any-conv")
        _seed_fact(store, "fb", "any-conv")
        store.set_fact_superseded("fa", "fb")
        conn = store._get_conn()
        row = conn.execute(
            "SELECT superseded_by FROM facts WHERE id = ?", ("fa",),
        ).fetchone()
        assert row[0] == "fb"

    def test_legacy_update_fact_fields_writes_unguarded(self):
        store = _make_store()
        _seed_fact(store, "fa", "any-conv")
        store.update_fact_fields("fa", "vv", "oo", "active", "ww")
        conn = store._get_conn()
        row = conn.execute(
            "SELECT verb, object FROM facts WHERE id = ?", ("fa",),
        ).fetchone()
        assert tuple(row) == ("vv", "oo")


# ---------------------------------------------------------------------------
# T3.15: store_chunk_embeddings rejects a segment_ref from another
# conversation even when the caller supplies a valid active operation
# for its own conversation.
# ---------------------------------------------------------------------------


class TestT315_StoreChunkEmbeddingsCrossConvSegmentReject:
    def test_segment_belongs_to_other_conv_rejects(self):
        store = _make_store()
        own_conv = "conv-t3-15-own"
        other_conv = "conv-t3-15-other"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, own_conv, op_id=op, worker_id=worker)
        # Segment lives in other_conv. The segments JOIN in the probe
        # blocks the cross-conv write even though the caller-supplied
        # conversation_id matches the active op.
        _seed_segment(store, "seg-other", other_conv)
        with pytest.raises(CompactionLeaseLost):
            store.store_chunk_embeddings(
                "seg-other",
                [ChunkEmbedding(segment_ref="seg-other", chunk_index=0,
                                text="x", embedding=[1.0])],
                operation_id=op, owner_worker_id=worker,
                lifecycle_epoch=1, conversation_id=own_conv,
            )


# ---------------------------------------------------------------------------
# T3.16: store_fact_links inserts all required columns and a conflict
# on a pre-existing id is a no-op that does not overwrite the prior
# operation_id stamp.
# ---------------------------------------------------------------------------


class TestT316_StoreFactLinksSchemaAndConflict:
    def test_insert_carries_all_columns(self):
        store = _make_store()
        conv = "conv-t3-16"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        _seed_fact(store, "fa", conv)
        _seed_fact(store, "fb", conv)
        link = FactLink(
            source_fact_id="fa", target_fact_id="fb",
            relation_type="supersedes", confidence=0.85,
            context="alice updated tea -> coffee",
            created_by="compaction",
        )
        store.store_fact_links(
            [link],
            operation_id=op, owner_worker_id=worker,
            lifecycle_epoch=1, conversation_id=conv,
        )
        conn = store._get_conn()
        row = conn.execute(
            """SELECT id, source_fact_id, target_fact_id, relation_type,
                      confidence, context, created_by, operation_id
                 FROM fact_links WHERE id = ?""",
            (link.id,),
        ).fetchone()
        assert row is not None
        assert row[0] == link.id
        assert row[1] == "fa"
        assert row[2] == "fb"
        assert row[3] == "supersedes"
        assert abs(row[4] - 0.85) < 1e-9
        assert row[5] == "alice updated tea -> coffee"
        assert row[6] == "compaction"
        assert row[7] == op

    def test_conflict_on_id_does_not_overwrite_prior_op_stamp(self):
        store = _make_store()
        conv = "conv-t3-16b"
        op_first = uuid.uuid4().hex
        op_second = uuid.uuid4().hex
        worker = "w-1"
        _seed_fact(store, "fa", conv)
        _seed_fact(store, "fb", conv)

        # First insert under op_first.
        _seed_running_op(store, conv, op_id=op_first, worker_id=worker)
        link = FactLink(
            source_fact_id="fa", target_fact_id="fb",
            relation_type="r", confidence=1.0, context="c",
            created_by="compaction",
        )
        store.store_fact_links(
            [link],
            operation_id=op_first, owner_worker_id=worker,
            lifecycle_epoch=1, conversation_id=conv,
        )
        assert _link_operation_id(store, link.id) == op_first

        # Mark op_first completed so the partial unique index allows
        # op_second to be inserted as the new running op for (conv,
        # epoch). This mirrors the post-complete_compaction_operation
        # state during a takeover.
        conn = store._get_conn()
        conn.execute(
            "UPDATE compaction_operation SET status = 'completed', "
            "completed_at = ? WHERE operation_id = ?",
            (_TS, op_first),
        )
        conn.commit()
        _seed_running_op(store, conv, op_id=op_second, worker_id=worker)
        # Re-insert with the same link.id. ON CONFLICT (id) DO NOTHING
        # leaves the row untouched; the existing operation_id stamp
        # stays op_first, NOT overwritten to op_second.
        store.store_fact_links(
            [link],
            operation_id=op_second, owner_worker_id=worker,
            lifecycle_epoch=1, conversation_id=conv,
        )
        assert _link_operation_id(store, link.id) == op_first


# ---------------------------------------------------------------------------
# T3.19: mixed partial guard kwargs are rejected as ValueError.
# ---------------------------------------------------------------------------


class TestT319_MixedPartialKwargsRejected:
    def test_store_chunk_embeddings_partial_kwargs_raises_value_error(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.store_chunk_embeddings(
                "seg", [],
                operation_id="op-1",
                # owner_worker_id, lifecycle_epoch, conversation_id missing
            )

    def test_link_segment_tool_output_partial_kwargs_raises_value_error(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.link_segment_tool_output(
                "conv", "seg", "tool",
                operation_id="op-1", owner_worker_id="w-1",
                # lifecycle_epoch missing
            )

    def test_store_fact_links_partial_kwargs_raises_value_error(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.store_fact_links(
                [],
                operation_id="op-1", owner_worker_id="w-1",
                lifecycle_epoch=1,
                # conversation_id missing
            )

    def test_set_fact_superseded_partial_kwargs_raises_value_error(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.set_fact_superseded(
                "a", "b",
                operation_id="op-1",
                # owner_worker_id, lifecycle_epoch missing
            )

    def test_update_fact_fields_partial_kwargs_raises_value_error(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.update_fact_fields(
                "f", "v", "o", "active", "w",
                operation_id="op-1", owner_worker_id="w-1",
                # lifecycle_epoch missing
            )

    def test_store_facts_partial_kwargs_raises_value_error(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.store_facts(
                [],
                operation_id="op-1", lifecycle_epoch=1,
                # owner_worker_id missing
            )

    def test_replace_facts_for_segment_partial_kwargs_raises_value_error(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.replace_facts_for_segment(
                "conv", "seg", [],
                operation_id="op-1",
                # owner_worker_id, lifecycle_epoch missing
            )


# ---------------------------------------------------------------------------
# T3.20: guarded store_facts stamps facts.operation_id on insert and
# conflict replace; legacy all-None calls do not stamp.
# ---------------------------------------------------------------------------


class TestT320_StoreFactsStampsOperationId:
    def test_guarded_insert_stamps_operation_id(self):
        store = _make_store()
        conv = "conv-t3-20"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        fact = Fact(id="f-stamped", subject="s", verb="v", object="o",
                    conversation_id=conv)
        n = store.store_facts(
            [fact],
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )
        assert n == 1
        assert _fact_operation_id(store, "f-stamped") == op

    def test_guarded_conflict_replace_stamps_latest_operation_id(self):
        store = _make_store()
        conv = "conv-t3-20-conflict"
        op_first = uuid.uuid4().hex
        op_second = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_first, worker_id=worker)
        fact = Fact(id="f-conflict", subject="s", verb="v1",
                    object="o", conversation_id=conv)
        assert store.store_facts(
            [fact],
            operation_id=op_first, owner_worker_id=worker, lifecycle_epoch=1,
        ) == 1
        assert _fact_operation_id(store, "f-conflict") == op_first

        conn = store._get_conn()
        conn.execute(
            "UPDATE compaction_operation SET status = 'completed', "
            "completed_at = ? WHERE operation_id = ?",
            (_TS, op_first),
        )
        conn.commit()
        _seed_running_op(store, conv, op_id=op_second, worker_id=worker)

        replacement = Fact(id="f-conflict", subject="s", verb="v2",
                           object="o", conversation_id=conv)
        assert store.store_facts(
            [replacement],
            operation_id=op_second, owner_worker_id=worker,
            lifecycle_epoch=1,
        ) == 1
        row = conn.execute(
            "SELECT verb, operation_id FROM facts WHERE id = ?",
            ("f-conflict",),
        ).fetchone()
        assert tuple(row) == ("v2", op_second)

    def test_legacy_insert_does_not_stamp(self):
        store = _make_store()
        fact = Fact(id="f-legacy", subject="s", verb="v", object="o",
                    conversation_id="any-conv")
        n = store.store_facts([fact])
        assert n == 1
        assert _fact_operation_id(store, "f-legacy") is None


# ---------------------------------------------------------------------------
# T3.21: guarded replace_facts_for_segment stamps facts.operation_id
# on inserted replacement facts; a guard mismatch rolls back the DELETE
# rather than leaving the segment factless.
# ---------------------------------------------------------------------------


class TestT321_ReplaceFactsForSegmentStampsAndRollsBack:
    def test_guarded_replace_stamps_operation_id(self):
        store = _make_store()
        conv = "conv-t3-21"
        op = uuid.uuid4().hex
        worker = "w-1"
        seg_ref = "seg-replace"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        replacement = Fact(id="f-r1", subject="s", verb="v", object="o",
                           segment_ref=seg_ref, conversation_id=conv)
        store.replace_facts_for_segment(
            conv, seg_ref, [replacement],
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )
        assert _fact_operation_id(store, "f-r1") == op

    def test_guard_mismatch_rolls_back_delete(self):
        store = _make_store()
        conv = "conv-t3-21b"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        seg_ref = "seg-rb"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        # Seed a pre-existing fact for the segment. The guard mismatch
        # must NOT delete this row even though the DELETE runs before
        # the INSERT-SELECT raises CompactionLeaseLost.
        _seed_fact(store, "f-existing", conv)
        conn = store._get_conn()
        conn.execute(
            "UPDATE facts SET segment_ref = ? WHERE id = ?",
            (seg_ref, "f-existing"),
        )
        conn.commit()

        replacement = Fact(
            id="f-new", subject="s", verb="v", object="o",
            segment_ref=seg_ref, conversation_id=conv,
        )
        with pytest.raises(CompactionLeaseLost):
            store.replace_facts_for_segment(
                conv, seg_ref, [replacement],
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1,
            )
        # Pre-existing fact should still be there: the BEGIN IMMEDIATE
        # transaction rolled back when CompactionLeaseLost fired.
        row = conn.execute(
            "SELECT id FROM facts WHERE id = ?", ("f-existing",),
        ).fetchone()
        assert row is not None
