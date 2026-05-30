"""Compaction-fence Phase 7 tests for the CompactionFenceMode
parser and the store-construction holder discipline.

Per fencing plan §9.0 + §9.6:

* T7.4. Invalid ``VC_COMPACTION_FENCE_MODE`` fails at store
  construction with a clear ``ValueError``, never silently
  downgrading to a weaker mode.
* T7.5. ``CompactionFenceMode.from_env()`` parses the documented
  values (default + accepted + whitespace/case normalization) under
  env isolation so no mode state leaks across cases.
* T7.6. The mode resolved at store construction is pinned for the
  lifetime of the store. Flipping the env var after construction
  does NOT affect existing stores; only fresh construction observes
  the new value. Cleanup and write paths read the same holder.

The cleanup tier gates (T7.1 / T7.2 / T7.3), the
``_enforce_or_observe_mismatch`` helper, and the representative
per-write mode matrix live in this file.

Tests use ``monkeypatch`` to scope every ``VC_COMPACTION_FENCE_MODE``
manipulation to a single test case, restoring the prior value at
teardown.
"""

from __future__ import annotations

import pytest

from virtual_context.core.compaction_fence import CompactionFenceMode
from virtual_context.storage.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# T7.5: parser tests for CompactionFenceMode.from_env.
# ---------------------------------------------------------------------------


class TestT75_FromEnvParser:
    """T7.5: env-isolated parser tests covering default, accepted
    values, normalization, and unset behavior.
    """

    def test_default_is_off_when_var_unset(self, monkeypatch):
        monkeypatch.delenv("VC_COMPACTION_FENCE_MODE", raising=False)
        assert CompactionFenceMode.from_env() is CompactionFenceMode.OFF

    @pytest.mark.parametrize("raw,expected", [
        ("off", CompactionFenceMode.OFF),
        ("observe", CompactionFenceMode.OBSERVE),
        ("active", CompactionFenceMode.ACTIVE),
    ])
    def test_accepted_values(self, monkeypatch, raw, expected):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        assert CompactionFenceMode.from_env() is expected

    @pytest.mark.parametrize("raw", ["OFF", "OBSERVE", "ACTIVE", "Off",
                                     "Observe", "Active"])
    def test_case_insensitive(self, monkeypatch, raw):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        assert CompactionFenceMode.from_env().value == raw.lower()

    @pytest.mark.parametrize("raw,expected", [
        ("  off  ", CompactionFenceMode.OFF),
        ("\tobserve\n", CompactionFenceMode.OBSERVE),
        (" ACTIVE ", CompactionFenceMode.ACTIVE),
    ])
    def test_whitespace_stripped(self, monkeypatch, raw, expected):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        assert CompactionFenceMode.from_env() is expected

    def test_explicit_environ_overrides_os_environ(self, monkeypatch):
        # Set os.environ to one value and pass a different mapping; the
        # passed mapping must win so callers can scope a from_env() call
        # to a captured environment snapshot.
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "off")
        assert (
            CompactionFenceMode.from_env(
                environ={"VC_COMPACTION_FENCE_MODE": "active"},
            )
            is CompactionFenceMode.ACTIVE
        )


# ---------------------------------------------------------------------------
# T7.4: invalid env values fail loud at construction time.
# ---------------------------------------------------------------------------


class TestT74_InvalidValueFailsLoud:
    @pytest.mark.parametrize("raw", [
        "enforce", "ENFORCED", "1", "true", "off,observe",
        " on ", "act ive",
    ])
    def test_invalid_value_raises_value_error(self, monkeypatch, raw):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        with pytest.raises(ValueError, match="VC_COMPACTION_FENCE_MODE"):
            CompactionFenceMode.from_env()

    def test_invalid_env_blocks_sqlite_store_construction(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "ENFORCED")
        with pytest.raises(ValueError, match="VC_COMPACTION_FENCE_MODE"):
            SQLiteStore(tmp_path / "bad.db")

    def test_explicit_kwarg_bypasses_invalid_env(self, monkeypatch, tmp_path):
        """When the caller passes an explicit ``CompactionFenceMode``,
        the env var is not consulted -- a bad env var must not break
        construction in that case.
        """
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "garbage")
        store = SQLiteStore(
            tmp_path / "ok.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        assert store._compaction_fence_mode is CompactionFenceMode.OBSERVE


# ---------------------------------------------------------------------------
# T7.6: store-lifetime invariants.
# ---------------------------------------------------------------------------


class TestT76_StoreLifetimePin:
    """T7.6: mode resolved at construction is pinned for the
    store's lifetime. Flipping the env afterward must not affect
    the existing store; a new store sees the new value.
    """

    def test_existing_store_unchanged_when_env_flips(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "active")
        store = SQLiteStore(tmp_path / "pin.db")
        assert store._compaction_fence_mode is CompactionFenceMode.ACTIVE

        # Flip the env mid-lifetime. The store keeps its pinned holder.
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "off")
        assert store._compaction_fence_mode is CompactionFenceMode.ACTIVE

    def test_new_store_observes_new_env_after_flip(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "active")
        store_a = SQLiteStore(tmp_path / "a.db")
        assert store_a._compaction_fence_mode is CompactionFenceMode.ACTIVE

        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "observe")
        store_b = SQLiteStore(tmp_path / "b.db")
        # store_a still ACTIVE; store_b OBSERVE.
        assert store_a._compaction_fence_mode is CompactionFenceMode.ACTIVE
        assert store_b._compaction_fence_mode is CompactionFenceMode.OBSERVE


# ---------------------------------------------------------------------------
# Holder property surface: enforces / stamps_operation_id flags map
# the tier semantics consistently for cleanup and per-write paths.
# ---------------------------------------------------------------------------


class TestModePropertySurface:
    def test_off_holder_flags(self):
        m = CompactionFenceMode.OFF
        assert m.is_off and not m.is_observe and not m.is_active
        assert not m.enforces
        assert not m.stamps_operation_id

    def test_observe_holder_flags(self):
        m = CompactionFenceMode.OBSERVE
        assert m.is_observe and not m.is_off and not m.is_active
        assert not m.enforces
        assert m.stamps_operation_id

    def test_active_holder_flags(self):
        m = CompactionFenceMode.ACTIVE
        assert m.is_active and not m.is_off and not m.is_observe
        assert m.enforces
        assert m.stamps_operation_id


# ---------------------------------------------------------------------------
# CompositeStore mode-mismatch guard: when a delegate exposes a
# different mode holder than the composite's, construction fails so
# cleanup and write paths cannot run on divergent sources.
# ---------------------------------------------------------------------------


class _SeedHelper:
    """Compact seeding for the cleanup-mode tests below. Mirrors the
    minimal subset of test_compaction_cleanup_extension's helpers so
    these tests don't share fixtures across modules.
    """

    @staticmethod
    def seed(store: SQLiteStore, *, conv: str, dead_op: str) -> None:
        from virtual_context.core.canonical_turns import utcnow_iso
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO conversation_lifecycle "
            "(conversation_id, generation, deleted, updated_at) "
            "VALUES (?, 0, 0, ?)",
            (conv, now),
        )
        # compaction_operation has an FK to conversations(conversation_id)
        # in the SQLite schema -- seed the parent row before the op.
        conn.execute(
            "INSERT OR IGNORE INTO conversations "
            "(conversation_id, tenant_id, phase, lifecycle_epoch, "
            " created_at, updated_at) "
            "VALUES (?, 't', 'active', 1, ?, ?)",
            (conv, now, now),
        )
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, 'w', ?)""",
            (dead_op, conv, now, now, now),
        )
        # Seed one row per new-table that the tier gate touches:
        # segment_chunks, segment_tool_outputs, fact_links.
        # segment_chunks rows are removed when the parent segment is
        # deleted. To isolate the tier-gate effect on segment_chunks,
        # we seed the parent segment with operation_id=NULL so the
        # (always-on) segments cleanup does
        # NOT cascade-delete the child chunk rows. The dead op still
        # owns the segment_chunks row via its own operation_id stamp
        # so the tier-gate logic decides whether to DELETE it.
        seg_ref = f"seg-{dead_op[:6]}"
        conn.execute(
            """INSERT INTO segments
               (ref, conversation_id, summary, full_text, primary_tag,
                compaction_model, created_at, start_timestamp,
                end_timestamp, operation_id)
               VALUES (?, ?, 's', 'f', 't', 'passthrough', ?, ?, ?,
                       NULL)""",
            (seg_ref, conv, now, now, now),
        )
        conn.execute(
            """INSERT INTO segment_chunks
               (segment_ref, chunk_index, text, embedding_json,
                operation_id)
               VALUES (?, 0, 'x', '[]', ?)""",
            (seg_ref, dead_op),
        )
        conn.execute(
            """INSERT INTO segment_tool_outputs
               (conversation_id, segment_ref, tool_output_ref,
                operation_id)
               VALUES (?, ?, ?, ?)""",
            (conv, seg_ref, f"tool-{dead_op[:6]}", dead_op),
        )
        # fact_links needs endpoint facts to satisfy the FK.
        for fid in ("src", "tgt"):
            conn.execute(
                """INSERT OR IGNORE INTO facts
                   (id, subject, verb, object, status, what,
                    conversation_id, mentioned_at, session_date,
                    operation_id)
                   VALUES (?, 's', 'v', 'o', 'active', '', ?, ?, ?,
                           NULL)""",
                (fid, conv, now, now),
            )
        conn.execute(
            """INSERT INTO fact_links
               (id, source_fact_id, target_fact_id, relation_type,
                confidence, context, created_at, created_by,
                operation_id)
               VALUES (?, 'src', 'tgt', 'r', 1.0, '', ?, 'compaction',
                       ?)""",
            (f"link-{dead_op[:6]}", now, dead_op),
        )
        conn.commit()

    @staticmethod
    def count(store: SQLiteStore, *, table: str, dead_op: str) -> int:
        conn = store._get_conn()
        if table == "segment_tool_outputs":
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE operation_id = ?",
                (dead_op,),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE operation_id = ?",
                (dead_op,),
            ).fetchone()
        return int(row[0])


class TestT71_OffModeSkipsNewTableCleanup:
    """T7.1: at OFF tier, ``cleanup_abandoned_compaction`` leaves the
    three new tables (segment_chunks, segment_tool_outputs,
    fact_links) untouched. The pre-P4 cleanup tables (segments,
    facts, tag_summaries, tag_summary_embeddings) still execute --
    they were active from M0 and the rollout discipline applies only
    to the new surface.
    """

    def test_new_tables_not_deleted_at_off(self, tmp_path, caplog):
        store = SQLiteStore(
            tmp_path / "off.db",
            compaction_fence_mode=CompactionFenceMode.OFF,
        )
        conv, dead = "conv-off", "op-off"
        _SeedHelper.seed(store, conv=conv, dead_op=dead)
        with caplog.at_level("WARNING"):
            store.cleanup_abandoned_compaction(
                conversation_id=conv,
                dead_operation_id=dead,
                new_operation_id="op-new-off",
                lifecycle_epoch=1, worker_id="w", phase_count=7,
            )
        # New-table rows survive at OFF.
        for table in ("segment_chunks", "segment_tool_outputs", "fact_links"):
            assert _SeedHelper.count(store, table=table, dead_op=dead) == 1, (
                f"OFF mode: expected {table} dead-op row to survive"
            )
        assert not [
            r for r in caplog.records
            if "COMPACTION_FENCE_CLEANUP_OBSERVED" in r.message
        ], "OFF mode must skip new-table cleanup silently"


class TestT72_ObserveModeLogsButDoesNotDelete:
    """T7.2: at OBSERVE tier, ``cleanup_abandoned_compaction`` logs
    a ``COMPACTION_FENCE_CLEANUP_OBSERVED`` line per new-table
    candidate without executing the DELETE.
    """

    def test_observe_logs_would_delete_and_skips(self, tmp_path, caplog):
        store = SQLiteStore(
            tmp_path / "obs.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        conv, dead = "conv-obs", "op-obs"
        _SeedHelper.seed(store, conv=conv, dead_op=dead)
        with caplog.at_level("WARNING"):
            store.cleanup_abandoned_compaction(
                conversation_id=conv,
                dead_operation_id=dead,
                new_operation_id="op-new-obs",
                lifecycle_epoch=1, worker_id="w", phase_count=7,
            )
        # Rows still there.
        for table in ("segment_chunks", "segment_tool_outputs", "fact_links"):
            assert _SeedHelper.count(store, table=table, dead_op=dead) == 1, (
                f"OBSERVE mode: expected {table} dead-op row to survive"
            )
        # Log lines fired with the right shape.
        observed = [
            r for r in caplog.records
            if "COMPACTION_FENCE_CLEANUP_OBSERVED" in r.message
        ]
        assert len(observed) == 3, (
            "OBSERVE mode must log exactly one would-delete line per new table"
        )
        seen: set[str] = set()
        for record in observed:
            assert record.levelname == "WARNING"
            for table in (
                "segment_chunks", "segment_tool_outputs", "fact_links",
            ):
                if f"table={table}" in record.message:
                    assert table not in seen, (
                        f"OBSERVE log duplicated table={table}"
                    )
                    seen.add(table)
                    assert f"operation_id={dead}" in record.message
                    assert "would_delete=1" in record.message
                    break
            else:
                pytest.fail(
                    f"OBSERVE log had unexpected table: {record.message}"
                )
        assert seen == {
            "segment_chunks", "segment_tool_outputs", "fact_links",
        }


class TestT73_ActiveModeDeletes:
    """T7.3: at ACTIVE tier, ``cleanup_abandoned_compaction``
    executes the new-table DELETEs (the current P4 behavior).
    """

    def test_active_deletes_new_table_rows(self, tmp_path):
        store = SQLiteStore(
            tmp_path / "act.db",
            compaction_fence_mode=CompactionFenceMode.ACTIVE,
        )
        conv, dead = "conv-act", "op-act"
        _SeedHelper.seed(store, conv=conv, dead_op=dead)
        store.cleanup_abandoned_compaction(
            conversation_id=conv,
            dead_operation_id=dead,
            new_operation_id="op-new-act",
            lifecycle_epoch=1, worker_id="w", phase_count=7,
        )
        for table in ("segment_chunks", "segment_tool_outputs", "fact_links"):
            assert _SeedHelper.count(store, table=table, dead_op=dead) == 0, (
                f"ACTIVE mode: expected {table} dead-op row to be deleted"
            )


# ---------------------------------------------------------------------------
# Mode-aware fence rejection helper: per-write raises are gated on
# ``_enforce_or_observe_mismatch`` per fencing plan §9.1-9.3. The
# tests below cover the helper contract and verify that representative
# storage calls route mismatches through it.
# ---------------------------------------------------------------------------


class TestEnforceOrObserveHelper:
    def test_active_raises_compaction_lease_lost(self, tmp_path):
        from virtual_context.types import CompactionLeaseLost
        store = SQLiteStore(
            tmp_path / "h-act.db",
            compaction_fence_mode=CompactionFenceMode.ACTIVE,
        )
        with pytest.raises(CompactionLeaseLost) as exc:
            store._enforce_or_observe_mismatch(
                operation_id="op-1", write_site="store_facts",
            )
        assert exc.value.write_site == "store_facts"
        assert exc.value.operation_id == "op-1"

    def test_observe_logs_without_raising(self, tmp_path, caplog):
        store = SQLiteStore(
            tmp_path / "h-obs.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        with caplog.at_level("WARNING"):
            store._enforce_or_observe_mismatch(
                operation_id="op-2", write_site="set_fact_superseded",
            )
        msgs = [
            r.message for r in caplog.records
            if "COMPACTION_FENCE_OBSERVED_MISMATCH" in r.message
        ]
        assert msgs, "OBSERVE mode must log the mismatch"
        assert "operation_id=op-2" in msgs[0]
        assert "write_site=set_fact_superseded" in msgs[0]
        assert "mode=observe" in msgs[0]

    def test_off_silent_no_raise_no_log(self, tmp_path, caplog):
        store = SQLiteStore(
            tmp_path / "h-off.db",
            compaction_fence_mode=CompactionFenceMode.OFF,
        )
        with caplog.at_level("WARNING"):
            store._enforce_or_observe_mismatch(
                operation_id="op-3", write_site="update_fact_fields",
            )
        msgs = [
            r.message for r in caplog.records
            if "COMPACTION_FENCE" in r.message
        ]
        assert not msgs, "OFF mode must be silent"


class TestPerWriteFenceModeMatrix:
    """Per-write fence raise sites are now mode-aware via the helper.
    Verify the same guarded write at all three tiers:

    * ACTIVE: rejects via ``CompactionLeaseLost`` (existing behavior).
    * OBSERVE: the method takes the legacy unguarded path via the
      OFF=legacy-SQL downgrade; the write LANDS and no mismatch log
      fires because the guard SQL never runs. Documented spec
      deviation: OBSERVE no longer stamps ``operation_id`` for the
      six fenced methods. Per fencing plan §9.1 + the lead's
      directive on OFF acting as a kill switch.
    * OFF: same legacy unguarded path as OBSERVE; the write lands
      silently. Per spec §9.1 kill-switch semantics.

    ``replace_facts_for_segment`` was the first method to use this
    pattern (deferred from fencing P7-behavioral pt.2 because of
    the pre-INSERT DELETE data-loss window). The other six
    methods now share the same downgrade for consistency.
    """

    @pytest.fixture
    def conv(self):
        return "conv-pwfm"

    def _seed_running_op(
        self, store: SQLiteStore, conv: str, op: str, worker: str,
    ) -> None:
        from virtual_context.core.canonical_turns import utcnow_iso
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO conversation_lifecycle "
            "(conversation_id, generation, deleted, updated_at) "
            "VALUES (?, 0, 0, ?)",
            (conv, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO conversations "
            "(conversation_id, tenant_id, phase, lifecycle_epoch, "
            " created_at, updated_at) "
            "VALUES (?, 't', 'compacting', 1, ?, ?)",
            (conv, now, now),
        )
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (op, conv, now, now, worker, now),
        )
        conn.execute(
            """INSERT INTO facts
               (id, subject, verb, object, status, what,
                conversation_id, mentioned_at, session_date)
               VALUES ('f-existing', 's', 'v', 'o', 'active', 'w',
                       ?, ?, ?)""",
            (conv, now, now),
        )
        conn.commit()

    def _fact_fields(self, store: SQLiteStore) -> tuple[str, str, str]:
        row = store._get_conn().execute(
            "SELECT verb, object, what FROM facts WHERE id = 'f-existing'",
        ).fetchone()
        assert row is not None
        return row["verb"], row["object"], row["what"]

    def test_active_raises_on_mismatch(self, tmp_path, conv):
        from virtual_context.types import CompactionLeaseLost
        store = SQLiteStore(
            tmp_path / "pwm-act.db",
            compaction_fence_mode=CompactionFenceMode.ACTIVE,
        )
        self._seed_running_op(store, conv, "op-real", "w-1")
        with pytest.raises(CompactionLeaseLost):
            store.update_fact_fields(
                "f-existing", "v2", "o2", "active", "w2",
                operation_id="op-mismatch",
                owner_worker_id="w-1", lifecycle_epoch=1,
            )
        assert self._fact_fields(store) == ("v", "o", "w")

    def test_observe_writes_via_legacy_path_no_mismatch_log(
        self, tmp_path, conv, caplog,
    ):
        """At OBSERVE the OFF=legacy-SQL downgrade applies: the
        method takes the legacy unguarded UPDATE path. The write
        lands and no mismatch is logged because the guard SQL
        never runs. Documented spec deviation: OBSERVE no longer
        stamps ``operation_id`` for the six fenced methods that
        share the ``replace_facts_for_segment``-style downgrade.
        Per fencing plan §9.1 + the lead's directive on OFF acting
        as a kill switch rather than a soft-drop of writes.
        """
        store = SQLiteStore(
            tmp_path / "pwm-obs.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        self._seed_running_op(store, conv, "op-real", "w-1")
        with caplog.at_level("WARNING"):
            store.update_fact_fields(
                "f-existing", "v2", "o2", "active", "w2",
                operation_id="op-mismatch",
                owner_worker_id="w-1", lifecycle_epoch=1,
            )
        # Legacy path ran: write landed with the new field values.
        assert self._fact_fields(store) == ("v2", "o2", "w2")
        # No mismatch log because the guard SQL was bypassed by the
        # OFF=legacy-SQL downgrade.
        msgs = [
            r.message for r in caplog.records
            if "COMPACTION_FENCE_OBSERVED_MISMATCH" in r.message
            and "update_fact_fields" in r.message
        ]
        assert not msgs

    def test_observe_legacy_path_does_not_leak_open_transaction(
        self, tmp_path, conv,
    ):
        """At OBSERVE the OFF=legacy-SQL downgrade bypasses the guard
        probe for per-write methods that open ``BEGIN IMMEDIATE``. The
        legacy write path must still commit/close the transaction so
        the next per-write call can start its own transaction.
        """
        from virtual_context.types import ChunkEmbedding
        store = SQLiteStore(
            tmp_path / "leak-obs.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        self._seed_running_op(store, conv, "op-real", "w-1")
        # Seed a segment outside the supplied conversation. OBSERVE
        # bypasses the guard and takes the legacy path, which ignores
        # the caller-supplied conversation_id for this write.
        from virtual_context.core.canonical_turns import utcnow_iso
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO segments
               (ref, conversation_id, summary, full_text, primary_tag,
                compaction_model, created_at, start_timestamp,
                end_timestamp)
               VALUES ('seg-real', 'conv-other', 's', 'f', 't',
                       'pass', ?, ?, ?)""",
            (now, now, now),
        )
        conn.commit()

        chunk = ChunkEmbedding(
            segment_ref="seg-real", chunk_index=0, text="x",
            embedding=[0.0],
        )
        # First call: OBSERVE takes the legacy path. The connection
        # MUST NOT be left in an open transaction.
        store.store_chunk_embeddings(
            "seg-real", [chunk],
            operation_id="op-real", owner_worker_id="w-1",
            lifecycle_epoch=1, conversation_id=conv,
        )
        assert not conn.in_transaction, (
            "OBSERVE-mode legacy write left an open transaction; "
            "the next BEGIN IMMEDIATE will fail"
        )
        # Second call confirms no transaction-in-transaction error.
        store.store_chunk_embeddings(
            "seg-real", [chunk],
            operation_id="op-real", owner_worker_id="w-1",
            lifecycle_epoch=1, conversation_id=conv,
        )

    def test_replace_facts_no_data_loss_at_observe(self, tmp_path, conv):
        """Regression for the second Codex P1: at OBSERVE/OFF, a
        per-fact INSERT mismatch inside ``replace_facts_for_segment``
        used to commit the pre-loop DELETE without a matching INSERT
        and leave the segment factless. The fix downgrades the
        method to the legacy unguarded path at OBSERVE/OFF so the
        DELETE+INSERT batch runs atomically without per-fact guard
        rejections. At ACTIVE the original guarded behavior holds.
        """
        from virtual_context.types import Fact
        from virtual_context.core.canonical_turns import utcnow_iso
        store = SQLiteStore(
            tmp_path / "rfs-obs.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        self._seed_running_op(store, conv, "op-real", "w-1")
        # Seed a pre-existing fact at the segment.
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO facts
               (id, subject, verb, object, status, what,
                conversation_id, segment_ref, mentioned_at,
                session_date)
               VALUES ('f-pre', 's', 'v', 'o', 'active', 'w',
                       ?, 'seg-rf', ?, ?)""",
            (conv, now, now),
        )
        conn.commit()

        # Use a MISMATCHED operation_id (no running op_row matches)
        # so the probe at the top of replace_facts_for_segment would
        # fail under the pre-fix code path at OBSERVE. The data-loss
        # window opened when the DELETE committed before the per-fact
        # INSERT mismatch; the helper continued the loop while the
        # DELETE stayed, leaving the segment factless. With the fix
        # in place, the method downgrades to the legacy unguarded
        # path at OBSERVE/OFF so the DELETE+INSERT batch runs
        # atomically without consulting the guard.
        new_fact = Fact(
            id="f-new", subject="s", verb="v", object="o",
            conversation_id=conv, segment_ref="seg-rf",
        )
        deleted, inserted = store.replace_facts_for_segment(
            conv, "seg-rf", [new_fact],
            operation_id="op-mismatched-no-such-op",
            owner_worker_id="w-1",
            lifecycle_epoch=1,
        )
        # The legacy path replaced the fact atomically.
        assert deleted == 1
        assert inserted == 1
        # Segment now has exactly the new fact, no data loss.
        row = conn.execute(
            "SELECT id FROM facts WHERE segment_ref = 'seg-rf'",
        ).fetchall()
        ids = {r[0] for r in row}
        assert ids == {"f-new"}, (
            f"OBSERVE-mode replace_facts_for_segment left the segment "
            f"with fact ids={ids}; expected just the new fact"
        )

    def test_off_writes_via_legacy_path_silently(
        self, tmp_path, conv, caplog,
    ):
        """At OFF the kill-switch downgrade applies: the method
        takes the legacy unguarded UPDATE path with no
        ``operation_id`` stamp. The write lands and no log
        fires because the guard SQL never runs. Per fencing
        plan §9.1: OFF is a kill switch, not a soft-drop. A
        mismatched ``operation_id`` is irrelevant when the
        guard is bypassed.
        """
        store = SQLiteStore(
            tmp_path / "pwm-off.db",
            compaction_fence_mode=CompactionFenceMode.OFF,
        )
        self._seed_running_op(store, conv, "op-real", "w-1")
        with caplog.at_level("WARNING"):
            store.update_fact_fields(
                "f-existing", "v2", "o2", "active", "w2",
                operation_id="op-mismatch",
                owner_worker_id="w-1", lifecycle_epoch=1,
            )
        # Legacy path ran: write landed with the new field values.
        assert self._fact_fields(store) == ("v2", "o2", "w2")
        # OFF stays silent: no mismatch helper invocation, no log.
        msgs = [
            r.message for r in caplog.records
            if "COMPACTION_FENCE_OBSERVED_MISMATCH" in r.message
        ]
        assert not msgs


class TestOffLegacySqlGate:
    """At OFF the six fenced methods take the legacy unguarded SQL
    path with no ``operation_id`` stamp. The mismatched
    ``operation_id`` kwarg is irrelevant because the guard SQL is
    bypassed entirely. Per fencing plan §9.1: OFF is a kill
    switch, mirroring ``replace_facts_for_segment``'s carve-out.

    Each test calls the method with guard kwargs that would NOT
    match any running ``compaction_operation`` row, then verifies
    the write landed and the persisted ``operation_id`` column is
    ``NULL`` on tables that carry it.
    """

    @pytest.fixture
    def conv(self):
        return "conv-off-legacy"

    def _store(self, tmp_path, name):
        return SQLiteStore(
            tmp_path / f"off-legacy-{name}.db",
            compaction_fence_mode=CompactionFenceMode.OFF,
        )

    def _seed_conv(self, store: SQLiteStore, conv: str) -> None:
        from virtual_context.core.canonical_turns import utcnow_iso
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO conversation_lifecycle "
            "(conversation_id, generation, deleted, updated_at) "
            "VALUES (?, 0, 0, ?)",
            (conv, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO conversations "
            "(conversation_id, tenant_id, phase, lifecycle_epoch, "
            " created_at, updated_at) "
            "VALUES (?, 't', 'active', 1, ?, ?)",
            (conv, now, now),
        )
        conn.commit()

    def _seed_segment(
        self, store: SQLiteStore, conv: str, seg_ref: str,
    ) -> None:
        from virtual_context.core.canonical_turns import utcnow_iso
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO segments
               (ref, conversation_id, summary, full_text, primary_tag,
                compaction_model, created_at, start_timestamp,
                end_timestamp)
               VALUES (?, ?, 's', 'f', 't', 'pass', ?, ?, ?)""",
            (seg_ref, conv, now, now, now),
        )
        conn.commit()

    def _seed_fact(
        self, store: SQLiteStore, conv: str, fact_id: str,
    ) -> None:
        from virtual_context.core.canonical_turns import utcnow_iso
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO facts
               (id, subject, verb, object, status, what,
                conversation_id, mentioned_at, session_date)
               VALUES (?, 's', 'v', 'o', 'active', 'w',
                       ?, ?, ?)""",
            (fact_id, conv, now, now),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # store_facts
    # ------------------------------------------------------------------
    def test_store_facts_legacy_path_lands_write_without_operation_id(
        self, tmp_path, conv,
    ):
        from virtual_context.types import Fact
        store = self._store(tmp_path, "store-facts")
        self._seed_conv(store, conv)
        # No running compaction_operation row exists; mismatched op
        # would fail the guarded INSERT-SELECT. OFF downgrades guard
        # so the legacy INSERT OR REPLACE path runs unconditionally.
        fact = Fact(
            id="f-off", subject="s", verb="v", object="o",
            conversation_id=conv,
        )
        count = store.store_facts(
            [fact],
            operation_id="op-bogus",
            owner_worker_id="w-bogus", lifecycle_epoch=1,
        )
        assert count == 1
        row = store._get_conn().execute(
            "SELECT id, operation_id FROM facts WHERE id = 'f-off'",
        ).fetchone()
        assert row is not None
        assert row["operation_id"] is None, (
            "OFF mode must use legacy INSERT with no operation_id stamp"
        )

    # ------------------------------------------------------------------
    # set_fact_superseded
    # ------------------------------------------------------------------
    def test_set_fact_superseded_legacy_path_lands_write(
        self, tmp_path, conv,
    ):
        store = self._store(tmp_path, "set-superseded")
        self._seed_conv(store, conv)
        self._seed_fact(store, conv, "f-old")
        self._seed_fact(store, conv, "f-new")
        # No running op matches; ACTIVE would reject; OFF downgrades.
        store.set_fact_superseded(
            "f-old", "f-new",
            operation_id="op-bogus",
            owner_worker_id="w-bogus", lifecycle_epoch=1,
        )
        row = store._get_conn().execute(
            """SELECT superseded_by, operation_id
                 FROM facts WHERE id = 'f-old'""",
        ).fetchone()
        assert row is not None
        assert row["superseded_by"] == "f-new"
        assert row["operation_id"] is None, (
            "OFF mode must use legacy SQL with no operation_id stamp"
        )

    # ------------------------------------------------------------------
    # update_fact_fields
    # ------------------------------------------------------------------
    def test_update_fact_fields_legacy_path_lands_write(
        self, tmp_path, conv,
    ):
        store = self._store(tmp_path, "update-fields")
        self._seed_conv(store, conv)
        self._seed_fact(store, conv, "f-edit")
        store.update_fact_fields(
            "f-edit", "v2", "o2", "active", "w2",
            operation_id="op-bogus",
            owner_worker_id="w-bogus", lifecycle_epoch=1,
        )
        row = store._get_conn().execute(
            """SELECT verb, object, what, operation_id
                 FROM facts WHERE id = 'f-edit'""",
        ).fetchone()
        assert row is not None
        assert (row["verb"], row["object"], row["what"]) == (
            "v2", "o2", "w2",
        )
        assert row["operation_id"] is None, (
            "OFF mode must use legacy SQL with no operation_id stamp"
        )

    # ------------------------------------------------------------------
    # store_chunk_embeddings
    # ------------------------------------------------------------------
    def test_store_chunk_embeddings_legacy_path_lands_write(
        self, tmp_path, conv,
    ):
        from virtual_context.types import ChunkEmbedding
        store = self._store(tmp_path, "chunk-emb")
        self._seed_conv(store, conv)
        self._seed_segment(store, conv, "seg-off")
        chunk = ChunkEmbedding(
            segment_ref="seg-off", chunk_index=0, text="x",
            embedding=[0.0],
        )
        store.store_chunk_embeddings(
            "seg-off", [chunk],
            operation_id="op-bogus", owner_worker_id="w-bogus",
            lifecycle_epoch=1, conversation_id=conv,
        )
        row = store._get_conn().execute(
            """SELECT segment_ref, chunk_index, operation_id
                 FROM segment_chunks WHERE segment_ref = 'seg-off'""",
        ).fetchone()
        assert row is not None
        assert row["chunk_index"] == 0
        assert row["operation_id"] is None, (
            "OFF mode must use legacy INSERT with no operation_id stamp"
        )

    # ------------------------------------------------------------------
    # store_fact_links
    # ------------------------------------------------------------------
    def test_store_fact_links_legacy_path_lands_write(
        self, tmp_path, conv,
    ):
        from virtual_context.types import FactLink
        store = self._store(tmp_path, "fact-links")
        self._seed_conv(store, conv)
        self._seed_fact(store, conv, "f-src")
        self._seed_fact(store, conv, "f-tgt")
        link = FactLink(
            id="fl-off",
            source_fact_id="f-src",
            target_fact_id="f-tgt",
            relation_type="related_to",
            confidence=1.0,
            context="ctx",
            created_by="u",
        )
        count = store.store_fact_links(
            [link],
            operation_id="op-bogus",
            owner_worker_id="w-bogus", lifecycle_epoch=1,
            conversation_id=conv,
        )
        assert count == 1
        row = store._get_conn().execute(
            "SELECT id, operation_id FROM fact_links WHERE id = 'fl-off'",
        ).fetchone()
        assert row is not None
        assert row["operation_id"] is None, (
            "OFF mode must use legacy INSERT with no operation_id stamp"
        )

    # ------------------------------------------------------------------
    # link_segment_tool_output
    # ------------------------------------------------------------------
    def test_link_segment_tool_output_legacy_path_lands_write(
        self, tmp_path, conv,
    ):
        store = self._store(tmp_path, "link-stout")
        self._seed_conv(store, conv)
        self._seed_segment(store, conv, "seg-link")
        # Seed a tool_outputs row so the link references a real ref;
        # the link table itself does not enforce FK to tool_outputs
        # but seeding keeps the fixture realistic.
        from virtual_context.core.canonical_turns import utcnow_iso
        now = utcnow_iso()
        conn = store._get_conn()
        conn.execute(
            """INSERT OR IGNORE INTO tool_outputs
               (ref, conversation_id, turn, content, created_at)
               VALUES ('to-off', ?, 0, '{}', ?)""",
            (conv, now),
        )
        conn.commit()
        store.link_segment_tool_output(
            conv, "seg-link", "to-off",
            operation_id="op-bogus",
            owner_worker_id="w-bogus", lifecycle_epoch=1,
        )
        row = conn.execute(
            """SELECT conversation_id, segment_ref, tool_output_ref,
                       operation_id
                 FROM segment_tool_outputs
                WHERE segment_ref = 'seg-link'""",
        ).fetchone()
        assert row is not None
        assert (row["conversation_id"], row["segment_ref"],
                row["tool_output_ref"]) == (conv, "seg-link", "to-off")
        assert row["operation_id"] is None, (
            "OFF mode must use legacy INSERT with no operation_id stamp"
        )


class TestCompositeStoreModeMismatch:
    def test_matching_modes_pass(self, tmp_path):
        from virtual_context.core.composite_store import CompositeStore
        store = SQLiteStore(
            tmp_path / "comp-match.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        composite = CompositeStore(
            segments=store, facts=store, fact_links=store,
            state=store, search=store,
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        assert composite._compaction_fence_mode is CompactionFenceMode.OBSERVE

    def test_mismatched_modes_raise(self, tmp_path):
        from virtual_context.core.composite_store import CompositeStore
        active_store = SQLiteStore(
            tmp_path / "comp-mix.db",
            compaction_fence_mode=CompactionFenceMode.ACTIVE,
        )
        with pytest.raises(ValueError, match="compaction_fence_mode mismatch"):
            CompositeStore(
                segments=active_store, facts=active_store,
                fact_links=active_store, state=active_store,
                search=active_store,
                compaction_fence_mode=CompactionFenceMode.OBSERVE,
            )
