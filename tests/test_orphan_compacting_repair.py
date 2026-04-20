"""Regression: phase='compacting' with no active compaction_operation
row must self-repair on the next handle_prepare_payload.

Observed in production after deploying a core build with a UUID schema
mismatch: state.py:2078 generated operation_id via uuid.uuid4().hex[:12]
but compaction_operation.operation_id is Postgres UUID type.
enter_compaction caught the INSERT failure and let compaction run on
the legacy in-memory path — but because no DB row was written, phase
was flipped to 'compacting' while no compaction was tracked. The
takeover predicate then saw prev_operation_id=None and treated it as
"live owner on another worker", never repairing the phase.

Fix: detect this state at the top of handle_prepare_payload and flip
phase back to 'active' so normal flow can resume.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path


def test_orphan_compacting_phase_self_repairs(tmp_path: Path):
    """phase='compacting' + no compaction_operation row → handle_prepare_payload
    repairs to 'active' and returns.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    # Seed the invalid state: phase='compacting' but no row in compaction_operation.
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    # Sanity: row really is absent.
    with inner._get_conn() as c:
        count = c.execute(
            "SELECT COUNT(*) FROM compaction_operation WHERE conversation_id=?",
            (conv,),
        ).fetchone()[0]
    assert count == 0, (
        f"precondition broken: compaction_operation has {count} rows for conv"
    )

    decision = state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0,
                            "ingestible_entry_count": 0},
    )

    assert decision.phase != "compacting", (
        f"orphan 'compacting' phase must be repaired; got phase={decision.phase!r}"
    )
    # And the DB reflects the repair.
    assert inner.get_conversation_phase(conv) != "compacting"


def test_legitimate_compacting_phase_not_repaired(tmp_path: Path):
    """Guard: when a real compaction_operation row exists, phase must
    stay 'compacting' and the takeover logic runs as before.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    from virtual_context.core.canonical_turns import utcnow_iso

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")
    now = utcnow_iso()
    with inner._get_conn() as c:
        c.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running', ?, ?, ?, ?)""",
            ("live-op-xyz", conv, now, now, state._worker_id, now),
        )
    # Also need at least the same worker registered as the active op so
    # takeover predicate doesn't trigger cleanup on our live op.
    state._active_compaction_op = "live-op-xyz"

    decision = state.handle_prepare_payload(
        body={"messages": [{"role": "user", "content": "hi"}]},
        payload_accounting={"raw_payload_entry_count": 1,
                            "ingestible_entry_count": 1},
    )

    # Legitimate compaction stays compacting.
    assert decision.phase == "compacting", (
        f"legitimate 'compacting' must not be repaired away; got {decision.phase!r}"
    )
    # Row stays 'running'.
    with inner._get_conn() as c:
        status = c.execute(
            "SELECT status FROM compaction_operation WHERE operation_id=?",
            ("live-op-xyz",),
        ).fetchone()[0]
    assert status == "running"
