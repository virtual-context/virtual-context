"""Task A29: compaction lifecycle methods on ProxyState + store helper.

Covers:

* ``ProxyState.enter_compaction`` — phase 'active' → 'compacting' + insert a
  ``compaction_operation`` row at the caller's epoch.
* ``ProxyState.advance_compaction_phase`` — update phase_index/phase_name on
  the active operation.
* ``ProxyState.exit_compaction`` — finalize the operation, then drain pending
  and transition phase atomically via ``drain_compaction_exit``.
* ``drain_compaction_exit`` decides 'ingesting' vs 'active' via a direct
  ``EXISTS`` on ``canonical_turns.tagged_at IS NULL`` inside a single
  transaction — NOT by calling ``read_progress_snapshot``.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import utcnow_iso


def _seed_canonical_row(inner, conv_id, canonical_id, sort_key, tagged=False):
    """Insert a minimal canonical_turns row for exit-decision coverage.

    Only the columns required by ``drain_compaction_exit``'s EXISTS guard
    (``conversation_id`` + ``tagged_at``) and the table's NOT NULL defaults
    are set; everything else rides on column defaults.
    """
    now = utcnow_iso()
    with inner._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, ?, ?, ?)
            """,
            (
                canonical_id, conv_id, f"h_{canonical_id}", sort_key,
                now, now, now if tagged else None, now, now,
            ),
        )


def test_enter_compaction_transitions_phase_and_creates_operation(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")

    state.enter_compaction(phase_count=3, initial_phase_name="init")

    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "compacting"
    assert snap.active_compaction is not None
    assert snap.active_compaction.phase_name == "init"
    assert snap.active_compaction.phase_count == 3


def test_advance_compaction_phase_updates_index(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")
    state.enter_compaction(phase_count=3)

    state.advance_compaction_phase(phase_index=1, phase_name="summarizing")

    snap = inner.read_progress_snapshot(conv_id)
    assert snap.active_compaction is not None
    assert snap.active_compaction.phase_index == 1
    assert snap.active_compaction.phase_name == "summarizing"


def test_exit_compaction_drains_pending_and_creates_episode_if_work(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Seed: one untagged canonical row + pending_raw=1000 + phase='compacting'.
    _seed_canonical_row(inner, conv_id, "t0", 1000.0)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="compacting")
    inner.widen_pending_raw_payload_entries(
        conversation_id=conv_id, lifecycle_epoch=1, value=1000,
    )
    inner.start_compaction_operation(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, phase_count=3, phase_name="init",
    )

    state.exit_compaction(success=True)

    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "ingesting"
    assert snap.active_episode is not None
    assert snap.active_episode.raw_payload_entries == 1000
    # Pending drained.
    with inner._get_conn() as conn:
        pending = conn.execute(
            "SELECT pending_raw_payload_entries FROM conversations"
            " WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()[0]
    assert pending == 0


def test_exit_compaction_goes_to_active_when_no_untagged(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Seed: canonical row is tagged → EXISTS returns false → phase 'active'.
    _seed_canonical_row(inner, conv_id, "t0", 1000.0, tagged=True)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="compacting")
    inner.start_compaction_operation(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, phase_count=3, phase_name="init",
    )

    state.exit_compaction(success=True)

    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "active"
    assert snap.active_episode is None


def test_drain_compaction_exit_uses_direct_exists_not_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the exit decision goes through the direct EXISTS inside the
    transaction — NOT through ``read_progress_snapshot``.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="compacting")
    _seed_canonical_row(inner, conv_id, "t0", 1000.0)  # untagged

    orig = inner.read_progress_snapshot
    calls = {"n": 0}

    def spy(conv):
        calls["n"] += 1
        return orig(conv)

    monkeypatch.setattr(inner, "read_progress_snapshot", spy)

    new_phase = inner.drain_compaction_exit(
        conversation_id=conv_id, lifecycle_epoch=1, worker_id=state._worker_id,
    )
    assert new_phase == "ingesting"
    # drain_compaction_exit MUST NOT invoke read_progress_snapshot.
    assert calls["n"] == 0
