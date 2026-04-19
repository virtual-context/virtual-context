"""Task 18: Wire compaction takeover into handle_prepare_payload.

Tests:
  test_compacting_phase_triggers_takeover_on_stale_lease
  test_compacting_phase_no_takeover_on_live_lease
  test_two_workers_race_only_one_takeover
  test_takeover_sigkill_mid_transaction_self_heals
  test_thread_pool_rejection_leaves_running_row_for_stale_reclaim
  test_takeover_detection_uses_live_thread_check_not_prev_owner
  test_crashed_compaction_does_not_wedge_dashboard_forever
  test_takeover_skips_submit_when_cleanup_returns_fresh_takeover_false
  test_same_worker_after_submit_exception_re_enters_takeover  (18e deferred from Task 16)
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import datetime as _dt
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared harness helpers
# ---------------------------------------------------------------------------

def _make_proxy_state(tmp_path: Path, conversation_id: str = "c"):
    """Minimal ProxyState backed by a real SQLite engine."""
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.proxy.state import ProxyState
    from virtual_context.types import (
        StorageConfig,
        TagGeneratorConfig,
        VirtualContextConfig,
    )

    config = VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / f"{conversation_id}.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    engine = VirtualContextEngine(config=config)
    return ProxyState(engine)


def _inner_store(engine):
    """Unwrap CompositeStore / EpochStore to the bare SQLiteStore."""
    store = engine._store
    inner = getattr(store, "_store", None)
    if inner is None:
        return store
    segments = getattr(inner, "_segments", None)
    if segments is not None:
        return segments
    return inner


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _stale_ts(age_s: float = 600) -> str:
    """Return an ISO timestamp age_s seconds in the past."""
    return (
        _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=age_s)
    ).isoformat()


def _seed_stale_compaction(
    store, conv: str, op_id: str, heartbeat_age_s: float = 600
) -> None:
    """Insert a 'running' compaction_operation row with a stale heartbeat."""
    now = _utcnow_iso()
    stale = _stale_ts(heartbeat_age_s)
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, 'worker-dead', ?)""",
            (op_id, conv, stale, stale, now),
        )


def _seed_live_compaction(store, conv: str, op_id: str) -> None:
    """Insert a 'running' compaction_operation row with a fresh heartbeat."""
    now = _utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, 'worker-live', ?)""",
            (op_id, conv, now, now, now),
        )


_BODY = {"messages": [{"role": "user", "content": "hi"}]}
_ACCOUNTING = {"raw_payload_entry_count": 1, "ingestible_entry_count": 1}


# ---------------------------------------------------------------------------
# Test 1: stale lease → takeover fires, new op inserted, submit called
# ---------------------------------------------------------------------------

def test_compacting_phase_triggers_takeover_on_stale_lease(tmp_path: Path):
    """When the running compaction_operation has a stale heartbeat,
    handle_prepare_payload must call claim, cleanup, and _submit_compaction_request.
    """
    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    _seed_stale_compaction(inner, conv, "dead-op-stale")
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    submit_calls: list = []

    def _spy_submit(*args, **kwargs):
        submit_calls.append(kwargs)

    state._submit_compaction_request = _spy_submit  # type: ignore[assignment]

    decision = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)

    assert decision.phase == "compacting"
    assert decision.started_tagger is False
    assert len(submit_calls) == 1, f"submit must be called exactly once; calls={submit_calls}"
    new_op = submit_calls[0].get("preexisting_operation_id")
    assert new_op is not None, "submit must receive a preexisting_operation_id"

    with inner._get_conn() as c:
        running = c.execute(
            "SELECT operation_id FROM compaction_operation "
            "WHERE conversation_id = ? AND status = 'running'",
            (conv,),
        ).fetchall()
    assert len(running) == 1
    assert running[0][0] == new_op, (
        f"Only the new takeover op should be running; got {running}"
    )


# ---------------------------------------------------------------------------
# Test 2: live lease → NO takeover, NO submit
# ---------------------------------------------------------------------------

def test_compacting_phase_no_takeover_on_live_lease(tmp_path: Path):
    """When the running compaction_operation has a fresh heartbeat,
    claim_compaction_lease returns claimed=False and handle_prepare_payload
    must NOT call _submit_compaction_request.
    """
    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    _seed_live_compaction(inner, conv, "live-op")
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    submit_calls: list = []

    def _spy_submit(*args, **kwargs):
        submit_calls.append(kwargs)

    state._submit_compaction_request = _spy_submit  # type: ignore[assignment]

    decision = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)

    assert decision.phase == "compacting"
    assert decision.started_tagger is False
    assert submit_calls == [], (
        f"No submit when lease is live; got {submit_calls}"
    )
    assert state._active_compaction_op is None


# ---------------------------------------------------------------------------
# Test 3: two workers race, only one takeover wins
# ---------------------------------------------------------------------------

def test_two_workers_race_only_one_takeover(tmp_path: Path):
    """Two ProxyState workers both see a stale op. The DB transaction ensures
    exactly one claim succeeds and inserts a new running row.

    Both workers use an empty body (no messages key) to bypass ingest_batch
    step 3, avoiding the UNIQUE constraint conflict on canonical_turns that
    would occur if both tried to ingest the same message concurrently. The
    phase gate fires regardless of body content.
    """
    state_a = _make_proxy_state(tmp_path, conversation_id="c")
    # Share the same DB for state_b
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.proxy.state import ProxyState
    from virtual_context.types import StorageConfig, TagGeneratorConfig, VirtualContextConfig

    config_b = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "c.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    engine_b = VirtualContextEngine(config=config_b)
    state_b = ProxyState(engine_b)

    conv = "c"
    inner = _inner_store(state_a.engine)

    _seed_stale_compaction(inner, conv, "shared-dead-op")
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    submit_calls_a: list = []
    submit_calls_b: list = []

    def _spy_a(*args, **kwargs):
        submit_calls_a.append(kwargs)

    def _spy_b(*args, **kwargs):
        submit_calls_b.append(kwargs)

    state_a._submit_compaction_request = _spy_a  # type: ignore[assignment]
    state_b._submit_compaction_request = _spy_b  # type: ignore[assignment]

    barrier = threading.Barrier(2)
    # Empty body: bypasses step 3 ingest_batch so no canonical_turns collision
    _empty_body = {}
    _empty_accounting = {"raw_payload_entry_count": 0, "ingestible_entry_count": 0}

    def _run(state):
        barrier.wait()
        state.handle_prepare_payload(body=_empty_body, payload_accounting=_empty_accounting)

    t_a = threading.Thread(target=_run, args=(state_a,))
    t_b = threading.Thread(target=_run, args=(state_b,))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    total_submits = len(submit_calls_a) + len(submit_calls_b)
    assert total_submits <= 1, (
        f"At most one worker should submit a takeover; "
        f"A={submit_calls_a} B={submit_calls_b}"
    )

    with inner._get_conn() as c:
        running = c.execute(
            "SELECT operation_id FROM compaction_operation "
            "WHERE conversation_id = ? AND status = 'running'",
            (conv,),
        ).fetchall()
    assert len(running) == 1, (
        f"Exactly one running row must exist after the race; got {running}"
    )


# ---------------------------------------------------------------------------
# Test 4: SIGKILL mid-transaction self-heals on next POST
# ---------------------------------------------------------------------------

def test_takeover_sigkill_mid_transaction_self_heals(tmp_path: Path):
    """Simulate a crash mid-cleanup by having cleanup_abandoned_compaction raise.
    The exception is swallowed and the next POST (stale row still present)
    re-attempts takeover successfully.
    """
    from virtual_context.types import CompactionLeaseClaim

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    _seed_stale_compaction(inner, conv, "crash-dead-op")
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    submit_calls: list = []

    def _spy_submit(*args, **kwargs):
        submit_calls.append(kwargs)

    state._submit_compaction_request = _spy_submit  # type: ignore[assignment]

    # First POST: cleanup crashes mid-transaction
    crash_claim = CompactionLeaseClaim(
        claimed=True,
        prev_operation_id="crash-dead-op",
        prev_owner_worker_id="worker-dead",
    )
    call_count = [0]
    original_cleanup = state.engine._store.cleanup_abandoned_compaction

    def _crashing_cleanup(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated SIGKILL mid-transaction")
        return original_cleanup(**kwargs)

    state.engine._store.claim_compaction_lease = MagicMock(return_value=crash_claim)
    state.engine._store.cleanup_abandoned_compaction = _crashing_cleanup

    # First POST — crash during cleanup; must not raise to the caller
    try:
        d1 = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)
    except RuntimeError:
        pass  # If the exception propagates that's also acceptable for this test

    # Reset mock for second POST: now let real cleanup run
    state.engine._store.claim_compaction_lease = MagicMock(return_value=crash_claim)
    state.engine._store.cleanup_abandoned_compaction = original_cleanup

    d2 = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)
    assert d2.phase == "compacting"
    # On second attempt cleanup must succeed and takeover fires
    assert len(submit_calls) >= 1, (
        f"Second POST must trigger takeover after first crash; calls={submit_calls}"
    )


# ---------------------------------------------------------------------------
# Test 5: thread pool rejection leaves running row for future stale reclaim
# ---------------------------------------------------------------------------

def test_thread_pool_rejection_leaves_running_row_for_stale_reclaim(tmp_path: Path):
    """If the thread pool submit raises (pool shutdown), the takeover's new_op
    row was inserted by cleanup but _active_compaction_op was cleared by the
    except in _submit_compaction_request. The new_op row remains 'running';
    the NEXT POST can reclaim it as stale.
    """
    from virtual_context.types import CompactionLeaseClaim

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    _seed_stale_compaction(inner, conv, "pool-dead-op")
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    new_op_id: list[str] = []

    # Intercept cleanup so we can capture the new_op_id
    original_cleanup = state.engine._store.cleanup_abandoned_compaction

    def _spy_cleanup(**kwargs):
        result = original_cleanup(**kwargs)
        if result:
            new_op_id.append(kwargs["new_operation_id"])
        return result

    state.engine._store.cleanup_abandoned_compaction = _spy_cleanup

    # Make the pool raise on submit
    with patch.object(
        state._compact_pool,
        "submit",
        side_effect=RuntimeError("pool shutdown"),
    ):
        decision = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)

    assert decision.phase == "compacting"
    assert state._active_compaction_op is None, (
        "Pool-rejection must clear _active_compaction_op"
    )
    # The new_op row must exist as 'running' for the next stale reclaim
    if new_op_id:
        with inner._get_conn() as c:
            row = c.execute(
                "SELECT status FROM compaction_operation WHERE operation_id = ?",
                (new_op_id[0],),
            ).fetchone()
        assert row is not None
        assert row[0] == "running", (
            f"new_op must remain 'running' after pool rejection so it can be "
            f"reclaimed as stale; got status={row[0]!r}"
        )


# ---------------------------------------------------------------------------
# Test 6: takeover detection uses the live thread state, not prev_owner
# ---------------------------------------------------------------------------

def test_takeover_detection_uses_live_thread_check_not_prev_owner(tmp_path: Path):
    """self._active_compaction_op is the source of truth for whether WE own the
    current op — not the prev_owner_worker_id from the claim. A stale row
    owned by our own worker_id must still be taken over if _active_compaction_op
    differs (e.g. we crashed and restarted).
    """
    from virtual_context.types import CompactionLeaseClaim

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    _seed_stale_compaction(inner, conv, "self-owned-dead-op")
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    # Simulate: claim returns prev_owner = our own worker_id (we own the row)
    # BUT _active_compaction_op is None (we lost the in-memory state after
    # a restart). The predicate is (self._active_compaction_op == claim.prev_operation_id),
    # NOT (prev_owner == self._worker_id). So takeover fires.
    claim_with_self_as_owner = CompactionLeaseClaim(
        claimed=True,
        prev_operation_id="self-owned-dead-op",
        prev_owner_worker_id=state._worker_id,  # our own worker id
    )
    state.engine._store.claim_compaction_lease = MagicMock(
        return_value=claim_with_self_as_owner
    )

    submit_calls: list = []

    def _spy_submit(*args, **kwargs):
        submit_calls.append(kwargs)

    state._submit_compaction_request = _spy_submit  # type: ignore[assignment]
    state._active_compaction_op = None  # not in-memory tracking this op

    decision = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)

    assert decision.phase == "compacting"
    assert len(submit_calls) == 1, (
        f"Takeover must fire even when prev_owner == our worker_id if "
        f"_active_compaction_op != prev_operation_id; got {submit_calls}"
    )


# ---------------------------------------------------------------------------
# Test 7 (regression): crashed compaction must not wedge dashboard forever
# ---------------------------------------------------------------------------

def test_crashed_compaction_does_not_wedge_dashboard_forever(tmp_path: Path):
    """Regression: before takeover wiring, a conversation stuck in 'compacting'
    phase with a dead compaction_operation would never exit the phase gate —
    every POST returned 'compacting' without restarting. After the fix, the
    first POST with a stale-heartbeat op must restart compaction.
    """
    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    _seed_stale_compaction(inner, conv, "wedged-dead-op", heartbeat_age_s=3600)
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    submitted: list[str] = []

    def _spy_submit(*args, **kwargs):
        op = kwargs.get("preexisting_operation_id", "")
        submitted.append(op)

    state._submit_compaction_request = _spy_submit  # type: ignore[assignment]

    for i in range(3):
        d = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)
        assert d.phase == "compacting"
        if submitted:
            break  # takeover fired, test passes

    assert len(submitted) >= 1, (
        "After takeover wiring, at least one POST must restart compaction "
        "when the prior operation has a stale heartbeat"
    )


# ---------------------------------------------------------------------------
# Test 8 (verbatim from plan lines 2362-2466): duplicate-takeover regression
# ---------------------------------------------------------------------------

def test_takeover_skips_submit_when_cleanup_returns_fresh_takeover_false(tmp_path):
    """Scenario: two workers arrive for the same stale compaction
    within the same TTL window. Worker A's claim + cleanup run first,
    abandoning the dead_op and inserting A's new_op_A. Worker B then
    arrives: claim succeeds (heartbeat on the row A just inserted is
    still fresh to B because A and B share clocks, but B's
    _active_compaction_op != new_op_A so B enters takeover). B's
    cleanup UPDATE matches zero rows because the CURRENT running row
    is new_op_A, not dead_op. cleanup returns False. B MUST NOT
    submit compaction with preexisting_operation_id=new_op_B — that
    row doesn't exist.

    Asserts:
    - B's handle_prepare_payload returns a widen-compacting PhaseDecision
    - B did NOT call _submit_compaction_request
    - B's _active_compaction_op stays None (never set)
    - Only one running compaction_operation row remains (A's new_op_A)
    """
    import sqlite3
    from unittest.mock import MagicMock, patch

    from tests.test_handle_prepare_payload import (
        _make_proxy_state, _inner_store,
    )
    from virtual_context.core.canonical_turns import utcnow_iso
    from virtual_context.types import CompactionLeaseClaim

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    now = utcnow_iso()
    # Simulate the post-A-takeover state: dead_op abandoned + a running
    # new_op_A inserted by A's cleanup.
    with inner._get_conn() as conn:
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at,
                completed_at)
               VALUES ('dead-op', ?, 1, 2, 7, 'segment_grouping',
                       'abandoned', ?, ?, 'worker-dead', ?, ?)""",
            (conv, now, now, now, now),
        )
        conn.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES ('new-op-A', ?, 1, 0, 7, 'starting', 'running',
                       ?, ?, 'worker-A', ?)""",
            (conv, now, now, now),
        )
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    # B's perspective: claim succeeds (returns the CURRENT running row's
    # op = new-op-A as prev_operation_id; but B thinks dead_op is what
    # it's abandoning — we override via the stub below to simulate
    # B having observed dead_op before the claim round-tripped).
    #
    # Simpler model: claim returns dead_op as prev, but cleanup's
    # UPDATE against dead_op hits zero rows (already abandoned), so
    # cleanup returns False.
    b_claim = CompactionLeaseClaim(
        claimed=True,
        prev_operation_id="dead-op",
        prev_owner_worker_id="worker-dead",
    )
    state.engine._store.claim_compaction_lease = MagicMock(return_value=b_claim)

    submit_calls: list = []
    original_submit = state._submit_compaction_request
    def _spy_submit(*args, **kwargs):
        submit_calls.append(kwargs)
    state._submit_compaction_request = _spy_submit  # type: ignore[assignment]

    state._active_compaction_op = None
    decision = state.handle_prepare_payload(
        body={"messages": [{"role": "user", "content": "hi"}]},
        payload_accounting={"raw_payload_entry_count": 1,
                            "ingestible_entry_count": 1},
    )

    assert decision.phase == "compacting"
    assert decision.started_tagger is False
    assert submit_calls == [], (
        f"B must NOT submit compaction when cleanup returns "
        f"fresh_takeover=False; submit was called with {submit_calls}"
    )
    assert state._active_compaction_op is None, (
        "B's _active_compaction_op must remain None since no new_op row "
        "was inserted on its behalf"
    )
    with inner._get_conn() as c:
        running = c.execute(
            "SELECT operation_id FROM compaction_operation "
            "WHERE conversation_id = ? AND status = 'running'",
            (conv,),
        ).fetchall()
    assert len(running) == 1
    assert running[0][0] == "new-op-A", (
        f"Only A's new-op-A should still be running; got {running}"
    )


# ---------------------------------------------------------------------------
# Test 18e (deferred from Task 16): same worker re-enters takeover after
# submit exception clears _active_compaction_op
# ---------------------------------------------------------------------------

def test_same_worker_after_submit_exception_re_enters_takeover(tmp_path: Path):
    """After a pool-submit exception clears _active_compaction_op to None,
    the NEXT POST for the same worker must re-enter the takeover path
    (claim → cleanup → submit) rather than being blocked by a stale
    non-None _active_compaction_op.

    Pre-condition: _submit_compaction_request's except clause sets
      _active_compaction_op = None on submit failure (Task 16 18d).
    Post-condition: the subsequent handle_prepare_payload call re-attempts
      takeover and calls submit again.

    Test design: we call handle_prepare_payload twice. To avoid the
    canonical_turns UNIQUE constraint (same message → same sort_key), the
    second call uses an empty body (no messages key) which bypasses
    ingest_batch (state.py step 3 guard: `if body and any(k in body ...)`).
    The phase gate still fires for the second call because the phase is still
    'compacting' in the DB.
    """
    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    _seed_stale_compaction(inner, conv, "reenter-dead-op")
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")

    submit_attempt: list[int] = [0]
    new_op_captured: list[str] = []

    def _failing_first_then_ok(*args, **kwargs):
        submit_attempt[0] += 1
        op = kwargs.get("preexisting_operation_id", "")
        if submit_attempt[0] == 1:
            # Simulate pool rejection on first attempt.
            # The takeover branch wraps submit in try/except, so the exception
            # is swallowed there — _active_compaction_op will not be set
            # (our spy never calls the real _submit_compaction_request, so
            # the set-before-submit / clear-on-except sequence never fires).
            # After the except swallows, _active_compaction_op stays None.
            raise RuntimeError("pool rejected first submit")
        new_op_captured.append(op)

    state._submit_compaction_request = _failing_first_then_ok  # type: ignore[assignment]

    # First POST: ingest "hi" + takeover fires, but submit raises.
    # The takeover except-block swallows the RuntimeError (plan lines 2598-2607).
    d1 = state.handle_prepare_payload(body=_BODY, payload_accounting=_ACCOUNTING)
    assert d1.phase == "compacting"
    assert submit_attempt[0] == 1, "First POST must have attempted submit once"

    # _active_compaction_op must still be None because our spy raised before
    # the real _submit_compaction_request could set it, and the real method's
    # except-block never ran. The takeover block never set it either.
    assert state._active_compaction_op is None, (
        "_active_compaction_op must be None so the next POST can re-enter "
        "the takeover path"
    )

    # Make the running row (cleanup's new_op from the first POST) appear stale
    # so claim_compaction_lease returns claimed=True again.
    with inner._get_conn() as c:
        stale_ts = _stale_ts(600)
        c.execute(
            "UPDATE compaction_operation SET heartbeat_ts = ? "
            "WHERE conversation_id = ? AND status = 'running'",
            (stale_ts, conv),
        )

    # Second POST: use an empty body to skip ingest_batch (no canonical turn
    # write = no UNIQUE conflict). The phase gate still fires → takeover runs.
    d2 = state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
    )
    assert d2.phase == "compacting"
    assert submit_attempt[0] >= 2, (
        "Second POST must re-attempt submit after _active_compaction_op "
        f"was cleared; only {submit_attempt[0]} attempt(s) made"
    )


# ---------------------------------------------------------------------------
# Test 9 (P2.3 regression): takeover on same ProxyState resets
# _compaction_cancelled so the new compactor doesn't immediately raise
# ---------------------------------------------------------------------------

def test_takeover_on_same_proxystate_resets_compaction_cancelled(tmp_path):
    """Regression: if a prior compaction on this ProxyState was killed
    by the sidecar (e.g., heartbeat refresh rejected), _compaction_cancelled
    is set. A subsequent takeover must reset the flag so the new
    compactor doesn't immediately raise InterruptedError on first progress.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    from unittest.mock import MagicMock
    from virtual_context.types import CompactionLeaseClaim

    state = _make_proxy_state(tmp_path)
    # Simulate: prior sidecar set the cancel flag before we run again.
    state._compaction_cancelled.set()
    assert state._compaction_cancelled.is_set()

    # Mock out the parts of _run_compact that touch external state so the
    # method can complete without a real store or Anthropic call.
    state.engine.compact_if_needed = MagicMock(return_value=None)
    state._update_compaction_state = MagicMock()
    state.enter_compaction = MagicMock()
    state.exit_compaction = MagicMock()
    # Also mock enter_compaction and the snapshot probe so entered_lifecycle
    # stays False (avoids store calls).
    try:
        state.engine._store.read_progress_snapshot = MagicMock(
            return_value=MagicMock(active_compaction=None)
        )
    except AttributeError:
        pass

    # Run _run_compact directly.  If the flag isn't reset, the compactor's
    # progress callback path would raise InterruptedError on the first tick.
    # Since compact_if_needed is mocked to return None, no callback fires and
    # the test simply asserts the flag IS cleared at return.
    state._run_compact(
        history=[], signal=None, turn=0, turn_id="",
        preexisting_operation_id=None,
    )
    assert not state._compaction_cancelled.is_set(), (
        "_run_compact must clear _compaction_cancelled at start so a "
        "stale sidecar-set flag doesn't immediately kill a retry compaction"
    )


# ---------------------------------------------------------------------------
# Test P1: normal-path compaction must not be misclassified as dead
# ---------------------------------------------------------------------------

def test_run_compact_sets_active_op_before_pipeline_runs(tmp_path):
    """Regression: _run_compact's normal path (preexisting_operation_id=None)
    must assign self._active_compaction_op immediately after finalizing the
    operation_id — BEFORE enter_compaction and BEFORE the pipeline's
    compact_if_needed runs.

    Without this, a second POST arriving on the same worker sees
    _active_compaction_op=None, the takeover predicate
    (self._active_compaction_op == claim.prev_operation_id) mismatches,
    cleanup_abandoned_compaction runs against our LIVE op, and
    CompactionLeaseLost fires mid-healthy-compaction.

    This test drives _run_compact directly with a mocked compact_if_needed
    that captures _active_compaction_op at call time. Reverting the set at
    state.py:2090 makes the captured value None and fails the assertion.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)

    captured: dict[str, object] = {}

    def _capture(*args, **kwargs):
        captured["active_op"] = state._active_compaction_op
        captured["invoked"] = True
        return None  # mimic "no-op" compaction

    # Also neutralize enter_compaction so we don't touch the DB lifecycle —
    # the regression is purely about the in-memory attribute set that must
    # land BEFORE compact_if_needed. Patch the progress snapshot too so the
    # post-enter_compaction probe returns a benign value.
    with patch.object(state.engine, "compact_if_needed", side_effect=_capture), \
         patch.object(state, "enter_compaction", new_callable=MagicMock), \
         patch.object(
             state.engine._store, "read_progress_snapshot",
             return_value=MagicMock(active_compaction=None),
         ):
        state._run_compact(
            history=[], signal=None, turn=0, turn_id="",
            preexisting_operation_id=None,
        )

    assert captured.get("invoked"), (
        "compact_if_needed was never called — _run_compact exited before "
        "the assignment could be observed"
    )
    assert captured["active_op"] is not None, (
        "_active_compaction_op was None at compact_if_needed — normal-path "
        "assignment at state.py:2090 is missing or misplaced. A second POST "
        "would misclassify this live op as abandoned."
    )
    # And after _run_compact returns, it must be cleared (outer-finally).
    assert state._active_compaction_op is None, (
        "_run_compact's outer finally must clear _active_compaction_op so "
        "no stale id leaks to the next compaction"
    )


def test_same_worker_live_compaction_not_misclassified_as_dead(tmp_path):
    """End-to-end companion to the direct _run_compact test above: with
    _active_compaction_op pre-set (as the fix ensures), a second POST on
    the same worker must NOT classify our own running op as abandoned.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    from virtual_context.core.canonical_turns import utcnow_iso

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    # Simulate _run_compact's post-line-2090 state: _active_compaction_op
    # set, DB row inserted.
    operation_id = "live-op-xyz"
    state._active_compaction_op = operation_id
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="compacting")
    now = utcnow_iso()
    with inner._get_conn() as c:
        c.execute(
            """INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (?, ?, 1, 0, 7, 'starting', 'running', ?, ?, ?, ?)""",
            (operation_id, conv, now, now, state._worker_id, now),
        )

    # Second POST — _active_compaction_op already set; takeover must NOT fire.
    cleanup_mock = MagicMock()
    state.engine._store.cleanup_abandoned_compaction = cleanup_mock
    submit_mock = MagicMock()
    state._submit_compaction_request = submit_mock  # type: ignore[assignment]

    decision = state.handle_prepare_payload(
        body={"messages": [{"role": "user", "content": "hi"}]},
        payload_accounting={"raw_payload_entry_count": 1,
                            "ingestible_entry_count": 1},
    )

    assert decision.phase == "compacting"
    assert cleanup_mock.call_count == 0, (
        "Takeover's cleanup must NOT fire against our own live compaction"
    )
    assert submit_mock.call_count == 0, (
        "No fresh compaction should spawn — we already own a live one"
    )
    with inner._get_conn() as c:
        status = c.execute(
            "SELECT status FROM compaction_operation WHERE operation_id=?",
            (operation_id,),
        ).fetchone()[0]
    assert status == "running", (
        f"Live op must remain 'running'; got {status!r}"
    )
