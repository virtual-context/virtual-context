"""Lifecycle contract: when the takeover path pre-inserts a
compaction_operation row, _run_compact must skip enter_compaction() and
use the pre-inserted row's operation_id throughout. Normal-start path
is unchanged (still calls enter_compaction).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from unittest.mock import patch

from virtual_context.core.canonical_turns import utcnow_iso


def _pre_insert_row(store, conv, op, worker="w"):
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, ?, ?)""",
            (op, conv, now, now, worker, now),
        )


def test_takeover_skips_enter_compaction_when_preexisting_operation_id_passed(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    _pre_insert_row(inner, conv, "pre-op-uuid", worker=state._worker_id)

    with patch.object(state, "enter_compaction") as mock_enter:
        state._run_compact_wrapper(
            history=[], signal=None, turn=0, target_end=0,
            turn_id="",
            preexisting_operation_id="pre-op-uuid",
        )
    assert mock_enter.call_count == 0, (
        "enter_compaction must be skipped when preexisting_operation_id is set"
    )

    # Exactly one compaction_operation row — no duplicate.
    with inner._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM compaction_operation WHERE conversation_id=?",
            (conv,),
        ).fetchone()[0]
    assert n == 1, f"expected 1 compaction_operation row, got {n}"


def test_normal_post_ingestion_compaction_still_calls_enter_compaction(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)
    with patch.object(state, "enter_compaction") as mock_enter:
        state._run_compact_wrapper(
            history=[], signal=None, turn=0, target_end=0, turn_id="",
            preexisting_operation_id=None,
        )
    assert mock_enter.called, (
        "enter_compaction must still be called on the normal post-ingest path"
    )
