"""Task 16: ProxyState._active_compaction_op four-site lifecycle invariant.

Tests:
  18a: test_takeover_sets_active_compaction_op_before_submit
  18b: test_wrapper_finally_clears_active_compaction_op_on_normal_exit
  18c: test_wrapper_finally_clears_active_compaction_op_on_exception
  18d: test_submit_exception_clears_active_compaction_op  (regression guard)

Test 18e (same-worker re-enter after submit exception) is deferred to
tests/test_compacting_phase_takeover.py (Task 18) because it requires the
handle_prepare_payload compacting-branch takeover wiring that Task 18 adds.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 18a: _submit_compaction_request sets _active_compaction_op BEFORE submitting
# the future when preexisting_operation_id is provided.
# ---------------------------------------------------------------------------

def test_takeover_sets_active_compaction_op_before_submit(tmp_path: Path):
    """_active_compaction_op must be set to preexisting_operation_id before
    the thread-pool submit call so the takeover predicate is already live if
    a concurrent worker checks between set and submit.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)

    captured_op_at_submit: list[str | None] = []

    original_submit = state._compact_pool.submit

    def capturing_submit(fn, *args, **kwargs):
        # Capture the value of _active_compaction_op at the moment submit fires
        captured_op_at_submit.append(state._active_compaction_op)
        # Return a done future so the wrapper doesn't block
        f: Future[None] = Future()
        f.set_result(None)
        return f

    with patch.object(state._compact_pool, "submit", side_effect=capturing_submit), \
         patch.object(state, "_run_compact_wrapper"):
        state._submit_compaction_request(
            history=[],
            signal=None,
            turn=0,
            target_end=5,
            turn_id="",
            preexisting_operation_id="op-abc-123",
        )

    assert len(captured_op_at_submit) == 1, "submit must be called exactly once"
    assert captured_op_at_submit[0] == "op-abc-123", (
        f"_active_compaction_op must equal preexisting_operation_id at submit time, "
        f"got {captured_op_at_submit[0]!r}"
    )


# ---------------------------------------------------------------------------
# 18b: _run_compact_wrapper finally clears _active_compaction_op on normal exit
# ---------------------------------------------------------------------------

def test_wrapper_finally_clears_active_compaction_op_on_normal_exit(tmp_path: Path):
    """After _run_compact_wrapper returns normally, _active_compaction_op must
    be None regardless of what it was set to before the call.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)
    state._active_compaction_op = "op-should-be-cleared"

    with patch.object(state, "_run_compact", return_value=None):
        state._run_compact_wrapper(
            history=[],
            signal=None,
            turn=0,
            target_end=5,
            turn_id="",
            preexisting_operation_id="op-should-be-cleared",
        )

    assert state._active_compaction_op is None, (
        f"_active_compaction_op must be None after wrapper normal exit, "
        f"got {state._active_compaction_op!r}"
    )


# ---------------------------------------------------------------------------
# 18c: _run_compact_wrapper finally clears _active_compaction_op on exception
# ---------------------------------------------------------------------------

def test_wrapper_finally_clears_active_compaction_op_on_exception(tmp_path: Path):
    """Even when _run_compact raises, _run_compact_wrapper's finally block must
    clear _active_compaction_op so future takeover checks are not blocked.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)
    state._active_compaction_op = "op-raise-case"

    with patch.object(state, "_run_compact", side_effect=RuntimeError("boom")):
        # The wrapper must NOT re-raise (it already swallows exceptions via
        # the existing try/finally structure — the finally still runs).
        try:
            state._run_compact_wrapper(
                history=[],
                signal=None,
                turn=0,
                target_end=5,
                turn_id="",
                preexisting_operation_id="op-raise-case",
            )
        except Exception:
            # Whether the wrapper propagates or swallows, _active_compaction_op
            # must be None after the call returns/raises.
            pass

    assert state._active_compaction_op is None, (
        f"_active_compaction_op must be None after wrapper exception exit, "
        f"got {state._active_compaction_op!r}"
    )


# ---------------------------------------------------------------------------
# 18d: submit exception clears _active_compaction_op (CRITICAL regression guard)
# ---------------------------------------------------------------------------

def test_submit_exception_clears_active_compaction_op(tmp_path: Path):
    """If the thread-pool submit() call itself raises (e.g. pool shutdown),
    _submit_compaction_request must clear _active_compaction_op back to None
    so subsequent requests are not permanently blocked by a stale op-id.

    This is the regression guard: without the except-clause cleanup, a pool
    shutdown exception would leave _active_compaction_op set forever and every
    future takeover check would see a non-None value for an operation that
    never ran.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)

    assert state._active_compaction_op is None, "pre-condition: starts as None"

    with patch.object(
        state._compact_pool,
        "submit",
        side_effect=RuntimeError("pool is shut down"),
    ):
        with pytest.raises(RuntimeError, match="pool is shut down"):
            state._submit_compaction_request(
                history=[],
                signal=None,
                turn=0,
                target_end=5,
                turn_id="",
                preexisting_operation_id="op-submit-fail",
            )

    assert state._active_compaction_op is None, (
        f"_active_compaction_op must be reset to None after submit() raises, "
        f"got {state._active_compaction_op!r}"
    )
