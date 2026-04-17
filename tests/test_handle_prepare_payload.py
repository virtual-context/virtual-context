"""Task A23: ProxyState.handle_prepare_payload — steps 1-3 + step 5 metadata.

These tests exercise the minimal shell of the new ingestion flow:

* Entry epoch verify (``verify_epoch``).
* Canonical row persistence via ``engine._ingest_reconciler.ingest_batch``.
* Defense-in-depth epoch verify.
* Per-request metadata update via ``store.update_request_metadata``.
* Return ``PhaseDecision(phase='init', started_tagger=False)``.

Tasks A24-A29 layer on top of this; A30 removes the legacy helpers.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch
from virtual_context.proxy.state import PhaseDecision, ProxyState


def _make_proxy_state(tmp_path: Path, conversation_id: str = "c") -> ProxyState:
    """Construct a minimal ProxyState backed by a real SQLite engine.

    Mirrors ``tests/test_engine_lifecycle_epoch.py::_make_test_engine`` and
    wraps the resulting engine in a bare ``ProxyState`` (no metrics, no
    upstream URL) so unit assertions touch the real store/epoch paths.
    """
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
    return ProxyState(engine)


def _inner_store(engine):
    """Return the concrete SQLite store underneath the wrappers.

    Mirrors the helper in ``tests/test_engine_lifecycle_epoch.py``.
    """
    store = engine._store
    inner = getattr(store, "_store", None)
    if inner is None:
        return store
    segments = getattr(inner, "_segments", None)
    if segments is not None:
        return segments
    return inner


def test_handle_prepare_payload_updates_request_metadata(tmp_path):
    """Step 5 — ``last_raw_payload_entries`` / ``last_ingestible_payload_entries``
    are overwritten on the conversations row."""
    state = _make_proxy_state(tmp_path)
    try:
        state.handle_prepare_payload(
            body={"messages": [{"role": "user", "content": "hi"}]},
            payload_accounting={
                "raw_payload_entry_count": 500,
                "ingestible_entry_count": 200,
            },
        )
        snap = state.engine._store.read_progress_snapshot(
            state.engine.config.conversation_id,
        )
        assert snap.last_raw_payload_entries == 500
        assert snap.last_ingestible_payload_entries == 200
    finally:
        state.engine.close()


def test_handle_prepare_payload_verifies_epoch_at_entry(tmp_path):
    """Step 2 — entry verify fires BEFORE any DB write.

    External resurrect bumps the DB epoch. The engine still holds the
    stale epoch in memory; the entry ``verify_epoch`` must reject it.
    """
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        inner.mark_conversation_deleted(conv_id)
        inner.increment_lifecycle_epoch_on_resurrect(conv_id)
        with pytest.raises(LifecycleEpochMismatch):
            state.handle_prepare_payload(
                body={},
                payload_accounting={
                    "raw_payload_entry_count": 0,
                    "ingestible_entry_count": 0,
                },
            )
    finally:
        state.engine.close()


def test_handle_prepare_payload_persists_canonical_rows(tmp_path):
    """Step 3 — IngestReconciler writes canonical rows for a fresh payload."""
    state = _make_proxy_state(tmp_path)
    try:
        state.handle_prepare_payload(
            body={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
            payload_accounting={
                "raw_payload_entry_count": 2,
                "ingestible_entry_count": 2,
            },
        )
        conv_id = state.engine.config.conversation_id
        rows = state.engine._store.get_all_canonical_turns(conv_id)
        assert len(rows) >= 1, "at least one canonical row was persisted"
    finally:
        state.engine.close()


def test_handle_prepare_payload_returns_phase_decision(tmp_path):
    """The shell returns a ``PhaseDecision`` with the initial fields."""
    state = _make_proxy_state(tmp_path)
    try:
        decision = state.handle_prepare_payload(
            body={},
            payload_accounting={
                "raw_payload_entry_count": 0,
                "ingestible_entry_count": 0,
            },
        )
        assert isinstance(decision, PhaseDecision)
        assert decision.phase == "init"
        assert decision.started_tagger is False
    finally:
        state.engine.close()


# ---------------------------------------------------------------------------
# Task A24 — phase gate (step 4)
# ---------------------------------------------------------------------------


def test_phase_gate_deleted_resurrects(tmp_path):
    """``phase == 'deleted'`` resurrects the conversation — lifecycle_epoch
    bumps and the engine's in-memory epoch is updated to match."""
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        inner.mark_conversation_deleted(conv_id)
        # Engine's cached epoch is still 1; DB has phase='deleted', epoch=1.
        state.handle_prepare_payload(
            body={},
            payload_accounting={
                "raw_payload_entry_count": 0,
                "ingestible_entry_count": 0,
            },
        )
        # Resurrect bumped to epoch=2; engine cache updated.
        assert state.engine._engine_state.lifecycle_epoch == 2
        assert inner.get_lifecycle_epoch(conv_id) == 2
        snap = inner.read_progress_snapshot(conv_id)
        assert snap.phase == "init"  # resurrect resets phase to init
    finally:
        state.engine.close()


def test_phase_gate_compacting_widens_pending_and_returns(tmp_path):
    """``phase == 'compacting'`` widens pending_raw (epoch-scoped) and
    returns a ``PhaseDecision(phase='compacting', started_tagger=False)``
    without triggering episode creation."""
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        # Force phase='compacting'.
        inner.set_phase(
            conversation_id=conv_id, lifecycle_epoch=1, phase="compacting",
        )
        decision = state.handle_prepare_payload(
            body={"messages": [{"role": "user", "content": "hi"}]},
            payload_accounting={
                "raw_payload_entry_count": 1000,
                "ingestible_entry_count": 500,
            },
        )
        assert decision.phase == "compacting"
        assert decision.started_tagger is False
        # pending_raw should reflect the new_raw widening.
        with inner._get_conn() as conn:
            pending = conn.execute(
                "SELECT pending_raw_payload_entries FROM conversations "
                "WHERE conversation_id = ?",
                (conv_id,),
            ).fetchone()[0]
        assert pending == 1000
        # Per-request metadata still updated (step 5 runs before step 4).
        snap = inner.read_progress_snapshot(conv_id)
        assert snap.last_raw_payload_entries == 1000
        assert snap.last_ingestible_payload_entries == 500
    finally:
        state.engine.close()


def test_phase_gate_ingesting_falls_through(tmp_path):
    """``phase == 'ingesting'`` does not widen pending or resurrect; the
    gate returns a stub ``PhaseDecision`` for now (A25-A29 will extend)."""
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        inner.set_phase(
            conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting",
        )
        decision = state.handle_prepare_payload(
            body={"messages": [{"role": "user", "content": "hi"}]},
            payload_accounting={
                "raw_payload_entry_count": 1,
                "ingestible_entry_count": 1,
            },
        )
        # Stub for now — subsequent tasks refine.
        assert decision.phase == "ingesting"
        # pending should NOT have been bumped.
        with inner._get_conn() as conn:
            pending = conn.execute(
                "SELECT pending_raw_payload_entries FROM conversations "
                "WHERE conversation_id = ?",
                (conv_id,),
            ).fetchone()[0]
        assert pending == 0
    finally:
        state.engine.close()
