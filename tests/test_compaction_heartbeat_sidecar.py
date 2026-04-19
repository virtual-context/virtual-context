"""Task 17: Compaction heartbeat sidecar.

Tests:
  test_heartbeat_sidecar_refreshes_while_compactor_runs  — sidecar calls
      refresh_compaction_heartbeat repeatedly while the stop_event is unset.
  test_heartbeat_sidecar_signals_cancel_on_failed_refresh  — sidecar sets
      _compaction_cancelled when refresh_compaction_heartbeat returns False.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_proxy_state(tmp_path: Path, conversation_id: str = "c"):
    """Minimal ProxyState backed by a real SQLite engine."""
    from virtual_context.engine import VirtualContextEngine
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
    from virtual_context.proxy.state import ProxyState
    return ProxyState(engine)


# ---------------------------------------------------------------------------
# Test 1: sidecar refreshes heartbeat while stop_event is unset
# ---------------------------------------------------------------------------

def test_heartbeat_sidecar_refreshes_while_compactor_runs(tmp_path: Path):
    """_run_compaction_heartbeat_sidecar calls refresh_compaction_heartbeat
    repeatedly as long as the stop_event is not set and the refresh succeeds.
    After at least 3 calls, we set the stop_event and verify
    _compaction_cancelled was NOT set.
    """
    state = _make_proxy_state(tmp_path)

    call_count = 0
    call_event = threading.Event()  # signals when we've seen enough calls

    def fake_refresh(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            call_event.set()
        return True  # always succeeds

    state.engine._store.refresh_compaction_heartbeat = fake_refresh

    stop_event = threading.Event()
    conversation_id = "test-conv-aabbcc"
    lifecycle_epoch = 1
    operation_id = "op12345678"

    # Use a very short interval (0.05 s) so the test is fast
    from virtual_context.proxy import state as state_module
    original_ttl = state_module.INGESTION_LEASE_TTL_S
    state_module.INGESTION_LEASE_TTL_S = 0.1  # interval = 0.05 s

    try:
        t = threading.Thread(
            target=state._run_compaction_heartbeat_sidecar,
            args=(conversation_id, lifecycle_epoch, operation_id, stop_event),
            daemon=True,
        )
        t.start()

        # Wait up to 5 s for 3 calls
        triggered = call_event.wait(timeout=5.0)
        stop_event.set()
        t.join(timeout=2.0)
    finally:
        state_module.INGESTION_LEASE_TTL_S = original_ttl

    assert triggered, f"Expected at least 3 refresh calls within timeout; got {call_count}"
    assert call_count >= 3, f"refresh called {call_count} times, expected >= 3"
    assert not state._compaction_cancelled.is_set(), (
        "_compaction_cancelled must NOT be set when all refreshes succeed"
    )


# ---------------------------------------------------------------------------
# Test 2: sidecar signals cancel when refresh returns False
# ---------------------------------------------------------------------------

def test_heartbeat_sidecar_signals_cancel_on_failed_refresh(tmp_path: Path):
    """When refresh_compaction_heartbeat returns False, the sidecar must set
    _compaction_cancelled so the progress callback can abort the compactor.
    """
    state = _make_proxy_state(tmp_path)

    def fake_refresh_fail(**kwargs):
        return False  # lease lost / epoch mismatch

    state.engine._store.refresh_compaction_heartbeat = fake_refresh_fail

    stop_event = threading.Event()
    conversation_id = "test-conv-ddeeff"
    lifecycle_epoch = 2
    operation_id = "op99887766"

    from virtual_context.proxy import state as state_module
    original_ttl = state_module.INGESTION_LEASE_TTL_S
    state_module.INGESTION_LEASE_TTL_S = 0.1  # interval = 0.05 s

    try:
        t = threading.Thread(
            target=state._run_compaction_heartbeat_sidecar,
            args=(conversation_id, lifecycle_epoch, operation_id, stop_event),
            daemon=True,
        )
        t.start()

        # Give the sidecar enough time to tick and signal cancel
        t.join(timeout=3.0)
    finally:
        state_module.INGESTION_LEASE_TTL_S = original_ttl

    assert state._compaction_cancelled.is_set(), (
        "_compaction_cancelled must be set when refresh_compaction_heartbeat returns False"
    )
