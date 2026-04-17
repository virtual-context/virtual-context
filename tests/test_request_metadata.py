"""Tests for epoch-guarded request-metadata + phase helpers on the SQLite backend.

The four methods under test write to the ``conversations`` table and are all
epoch-guarded at the SQL layer:

* ``update_request_metadata`` — overwrites ``last_raw_payload_entries`` and
  ``last_ingestible_payload_entries`` (the snapshot counters surfaced by the
  progress tracker) and returns a bool so callers can detect stale-epoch
  rejections.
* ``widen_pending_raw_payload_entries`` — monotonic widener using ``MAX()``
  so concurrent writers coalesce to the largest seen value without ever
  going backwards.
* ``set_phase`` — epoch-guarded phase write used by the progress state
  machine.
* ``set_phase_and_drain_pending_raw`` — atomic transactional method that
  both transitions the phase AND returns the drained ``pending_raw``
  counter. Returns ``None`` on epoch mismatch so callers can distinguish
  "no-op because stale" from "no-op because zero pending".

A stale caller whose in-memory epoch no longer matches the authoritative
row is rejected at SQL level and never stomps a new lifecycle's row.
"""

from __future__ import annotations

from pathlib import Path

from virtual_context.storage.sqlite import SQLiteStore


def _fresh(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    return s


# ----------------------------------------------------------------------
# update_request_metadata
# ----------------------------------------------------------------------

def test_update_request_metadata_overwrites_last_fields(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    ok = s.update_request_metadata(
        conversation_id="c", lifecycle_epoch=1,
        last_raw_payload_entries=500,
        last_ingestible_payload_entries=200,
    )
    assert ok is True
    snap = s.read_progress_snapshot("c")
    assert snap.last_raw_payload_entries == 500
    assert snap.last_ingestible_payload_entries == 200


def test_update_request_metadata_rejects_stale_epoch(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")  # epoch=2
    ok = s.update_request_metadata(
        conversation_id="c", lifecycle_epoch=1,  # stale
        last_raw_payload_entries=999,
        last_ingestible_payload_entries=999,
    )
    assert ok is False
    snap = s.read_progress_snapshot("c")
    assert snap.last_raw_payload_entries == 0
    assert snap.last_ingestible_payload_entries == 0


def test_update_request_metadata_returns_false_for_unknown_conversation(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "vc.db")
    ok = s.update_request_metadata(
        conversation_id="unknown", lifecycle_epoch=1,
        last_raw_payload_entries=1, last_ingestible_payload_entries=1,
    )
    assert ok is False


# ----------------------------------------------------------------------
# widen_pending_raw_payload_entries
# ----------------------------------------------------------------------

def test_widen_pending_raw_is_monotonic(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    s.widen_pending_raw_payload_entries(conversation_id="c", lifecycle_epoch=1, value=100)
    s.widen_pending_raw_payload_entries(conversation_id="c", lifecycle_epoch=1, value=50)
    with s._get_conn() as conn:
        val = conn.execute(
            "SELECT pending_raw_payload_entries FROM conversations WHERE conversation_id='c'"
        ).fetchone()[0]
    assert val == 100  # MAX preserved


def test_widen_pending_raw_rejects_stale_epoch(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    ok = s.widen_pending_raw_payload_entries(
        conversation_id="c", lifecycle_epoch=1, value=500,
    )
    assert ok is False
    with s._get_conn() as conn:
        val = conn.execute(
            "SELECT pending_raw_payload_entries FROM conversations WHERE conversation_id='c'"
        ).fetchone()[0]
    assert val == 0


def test_widen_pending_raw_returns_false_for_unknown_conversation(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "vc.db")
    ok = s.widen_pending_raw_payload_entries(
        conversation_id="unknown", lifecycle_epoch=1, value=100,
    )
    assert ok is False


# ----------------------------------------------------------------------
# set_phase
# ----------------------------------------------------------------------

def test_set_phase_changes_phase(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    ok = s.set_phase(conversation_id="c", lifecycle_epoch=1, phase="ingesting")
    assert ok is True
    snap = s.read_progress_snapshot("c")
    assert snap.phase == "ingesting"


def test_set_phase_rejected_on_epoch_mismatch(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")  # epoch=2
    ok = s.set_phase(conversation_id="c", lifecycle_epoch=1, phase="active")
    assert ok is False
    snap = s.read_progress_snapshot("c")
    assert snap.phase == "init"  # new lifecycle untouched


def test_set_phase_returns_false_for_unknown_conversation(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "vc.db")
    ok = s.set_phase(conversation_id="unknown", lifecycle_epoch=1, phase="active")
    assert ok is False


# ----------------------------------------------------------------------
# set_phase_and_drain_pending_raw
# ----------------------------------------------------------------------

def test_set_phase_and_drain_pending_raw_success(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    ok = s.set_phase(conversation_id="c", lifecycle_epoch=1, phase="compacting")
    assert ok is True
    s.widen_pending_raw_payload_entries(conversation_id="c", lifecycle_epoch=1, value=42)
    drained = s.set_phase_and_drain_pending_raw(
        conversation_id="c", lifecycle_epoch=1, new_phase="active",
    )
    assert drained == 42
    snap = s.read_progress_snapshot("c")
    assert snap.phase == "active"
    with s._get_conn() as conn:
        val = conn.execute(
            "SELECT pending_raw_payload_entries FROM conversations WHERE conversation_id='c'"
        ).fetchone()[0]
    assert val == 0


def test_set_phase_and_drain_returns_none_on_epoch_mismatch(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    s.widen_pending_raw_payload_entries(conversation_id="c", lifecycle_epoch=1, value=99)
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")  # epoch=2
    result = s.set_phase_and_drain_pending_raw(
        conversation_id="c", lifecycle_epoch=1, new_phase="active",
    )
    assert result is None
    # Phase and pending untouched on the new lifecycle row (pending is 0 in the
    # new epoch because the column is owned by the row not the epoch, but the
    # write must not have fired — phase must still be 'init').
    snap = s.read_progress_snapshot("c")
    assert snap.phase == "init"


def test_set_phase_and_drain_with_zero_pending(tmp_path: Path) -> None:
    s = _fresh(tmp_path)
    drained = s.set_phase_and_drain_pending_raw(
        conversation_id="c", lifecycle_epoch=1, new_phase="active",
    )
    assert drained == 0
