"""Task 15a: _run_compact aborts cleanly on CompactionLeaseLost.

When compact_if_needed raises CompactionLeaseLost, _run_compact must:
1. Call exit_compaction(success=False) exactly once.
2. Emit a COMPACTION_WRITE_REJECTED log line.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.types import CompactionLeaseLost


def test_compactor_aborts_on_lease_lost(tmp_path: Path, caplog):
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)

    exc = CompactionLeaseLost("op-stolen", write_site="store_segment")

    with patch.object(state.engine, "compact_if_needed", side_effect=exc), \
         patch.object(state, "exit_compaction", new_callable=MagicMock) as mock_exit:

        with caplog.at_level(logging.INFO, logger="virtual_context.proxy.state"):
            state._run_compact(
                history=[],
                signal=None,
                turn=0,
                turn_id="",
                preexisting_operation_id="op-stolen",
            )

    # 1. exit_compaction called exactly once with success=False
    mock_exit.assert_called_once()
    call_kwargs = mock_exit.call_args
    assert call_kwargs.kwargs.get("success") is False or (
        len(call_kwargs.args) >= 1 and call_kwargs.args[0] is False
    ), f"exit_compaction not called with success=False: {call_kwargs}"

    # 2. COMPACTION_WRITE_REJECTED log line emitted
    assert any(
        "COMPACTION_WRITE_REJECTED" in record.message
        for record in caplog.records
    ), (
        f"Expected COMPACTION_WRITE_REJECTED in logs; got: "
        f"{[r.message for r in caplog.records]}"
    )
