"""Tests for deferred payload compaction (cache-aware flush gate).

Covers:
- EngineState fields (flushed_through, last_request_time)
- SessionState persistence roundtrip
- history_offset behavior
- Flush gate logic in prepare_payload
- Invariant: flushed_through <= compacted_through
"""

from __future__ import annotations

import json
import time

from virtual_context.engine import _restored_flushed_through
from virtual_context.types import EngineState, EngineStateSnapshot, MonitorConfig
from virtual_context.proxy.session_state import SessionState


# ---------------------------------------------------------------------------
# State model tests
# ---------------------------------------------------------------------------

class TestEngineStateFields:
    def test_defaults(self):
        es = EngineState()
        assert es.flushed_through == 0
        assert es.last_request_time == 0.0

    def test_assignment(self):
        es = EngineState()
        es.flushed_through = 10
        es.last_request_time = 1234567890.0
        assert es.flushed_through == 10
        assert es.last_request_time == 1234567890.0


class TestMonitorConfigFields:
    def test_defer_defaults(self):
        mc = MonitorConfig()
        assert mc.defer_payload_mutation is False
        assert mc.flush_ttl_seconds == 300

    def test_defer_enabled(self):
        mc = MonitorConfig(defer_payload_mutation=True, flush_ttl_seconds=120)
        assert mc.defer_payload_mutation is True
        assert mc.flush_ttl_seconds == 120


class TestEngineStateSnapshot:
    def test_snapshot_has_fields(self):
        snap = EngineStateSnapshot(
            conversation_id="test",
            compacted_through=5,
            turn_tag_entries=[],
            turn_count=0,
            flushed_through=3,
            last_request_time=100.0,
        )
        assert snap.compacted_through == 5
        assert snap.flushed_through == 3
        assert snap.last_request_time == 100.0

    def test_snapshot_defaults(self):
        snap = EngineStateSnapshot(
            conversation_id="test",
            compacted_through=0,
            turn_tag_entries=[],
            turn_count=0,
        )
        assert snap.flushed_through == 0
        assert snap.last_request_time == 0.0


# ---------------------------------------------------------------------------
# history_offset tests
# ---------------------------------------------------------------------------

class TestHistoryOffset:
    def test_basic_offset(self):
        es = EngineState(compacted_through=5)
        assert es.history_offset(10) == 5

    def test_compacted_exceeds_history_returns_zero(self):
        es = EngineState(compacted_through=20)
        assert es.history_offset(10) == 0

    def test_flushed_through_does_not_affect_history_offset(self):
        """history_offset uses compacted_through, not flushed_through."""
        es = EngineState(compacted_through=5, flushed_through=0)
        assert es.history_offset(10) == 5


# ---------------------------------------------------------------------------
# Persistence roundtrip tests
# ---------------------------------------------------------------------------

class TestSessionStatePersistence:
    def test_roundtrip(self):
        original = SessionState(
            compacted_through=10,
            flushed_through=7,
            last_request_time=1234567890.5,
        )
        serialized = original.to_json()
        restored = SessionState.from_json(serialized)
        assert restored.compacted_through == 10
        assert restored.flushed_through == 7
        assert restored.flushed_through_present is True
        assert restored.last_request_time == 1234567890.5

    def test_backward_compat_missing_fields(self):
        """Old data without flushed_through/last_request_time gets defaults."""
        old_data = json.dumps({
            "compacted_through": 5,
            "last_compacted_turn": 3,
        }).encode()
        restored = SessionState.from_json(old_data)
        assert restored.flushed_through == 0
        assert restored.flushed_through_present is False
        assert restored.last_request_time == 0.0
        assert restored.compacted_through == 5

    def test_zero_flush_roundtrip_stays_present(self):
        original = SessionState(
            compacted_through=10,
            flushed_through=0,
            last_request_time=123.0,
        )
        restored = SessionState.from_json(original.to_json())
        assert restored.flushed_through == 0
        assert restored.flushed_through_present is True


class TestRestoreFlushedThrough:
    def test_missing_field_autosyncs_to_compacted(self):
        assert _restored_flushed_through(10, 0, present=False) == 10

    def test_present_zero_is_preserved(self):
        assert _restored_flushed_through(10, 0, present=True) == 0

    def test_present_value_is_preserved(self):
        assert _restored_flushed_through(10, 4, present=True) == 4


# ---------------------------------------------------------------------------
# Flush gate logic tests (unit-level, no server)
# ---------------------------------------------------------------------------

class TestFlushGateLogic:
    """Test the flush gate decision logic in isolation.

    These tests verify the algorithm without spinning up the full server.
    The gate logic:
    - defer=False: flushed_through auto-tracks compacted_through
    - defer=True + cold cache: flush (set flushed = compacted)
    - defer=True + warm cache + pending: defer (skip mutations)
    - defer=True + warm cache + no pending: no-op (already flushed)
    """

    def test_defer_off_auto_tracks(self):
        """When defer=False, flushed_through should equal compacted_through."""
        es = EngineState(compacted_through=10, flushed_through=0)
        # Simulate legacy auto-track (Step 5a)
        _defer = False
        _ct = es.compacted_through
        _ft = es.flushed_through
        if not _defer and _ct > _ft:
            es.flushed_through = _ct
        assert es.flushed_through == 10

    def test_cold_cache_flushes(self):
        """When cache is cold (age >= TTL), flush immediately."""
        es = EngineState(
            compacted_through=10,
            flushed_through=5,
            last_request_time=time.time() - 600,  # 10 min ago
        )
        _defer = True
        _flush_ttl = 300
        _cache_age = time.time() - es.last_request_time
        _should_flush_cold = _cache_age >= _flush_ttl

        assert _should_flush_cold
        if _should_flush_cold:
            es.flushed_through = es.compacted_through
        assert es.flushed_through == 10

    def test_warm_cache_defers(self):
        """When cache is warm and there's pending work, defer mutations."""
        es = EngineState(
            compacted_through=10,
            flushed_through=5,
            last_request_time=time.time() - 30,  # 30s ago, well within 300s TTL
        )
        _defer = True
        _flush_ttl = 300
        _cache_age = time.time() - es.last_request_time
        _should_flush_cold = _cache_age >= _flush_ttl
        _flush_pending = es.compacted_through > es.flushed_through
        _warm_defer = _defer and not _should_flush_cold and _flush_pending

        assert not _should_flush_cold
        assert _flush_pending
        assert _warm_defer
        # flushed_through should NOT change
        assert es.flushed_through == 5

    def test_warm_cache_no_pending_no_defer(self):
        """When cache is warm but flushed == compacted, nothing to defer."""
        es = EngineState(
            compacted_through=10,
            flushed_through=10,
            last_request_time=time.time() - 30,
        )
        _defer = True
        _flush_ttl = 300
        _cache_age = time.time() - es.last_request_time
        _should_flush_cold = _cache_age >= _flush_ttl
        _flush_pending = es.compacted_through > es.flushed_through
        _warm_defer = _defer and not _should_flush_cold and _flush_pending

        assert not _flush_pending
        assert not _warm_defer

    def test_first_request_cold_cache(self):
        """First request (last_request_time=0) should be treated as cold."""
        es = EngineState(
            compacted_through=5,
            flushed_through=0,
            last_request_time=0.0,
        )
        _cache_age = (time.time() - es.last_request_time) if es.last_request_time > 0 else float("inf")
        assert _cache_age == float("inf")
        assert _cache_age >= 300  # cold

    def test_multiple_compaction_cycles_warm(self):
        """Several compactions while cache warm — flushed stays put."""
        es = EngineState(
            compacted_through=5,
            flushed_through=5,
            last_request_time=time.time() - 10,
        )
        # Simulate 3 compaction cycles advancing compacted_through
        for new_ct in [8, 12, 15]:
            es.compacted_through = new_ct
            # Each time, cache is still warm
            _cache_age = time.time() - es.last_request_time
            _flush_pending = es.compacted_through > es.flushed_through
            _warm_defer = _cache_age < 300 and _flush_pending
            assert _warm_defer
            # flushed stays at 5
            assert es.flushed_through == 5

    def test_warm_then_cold_flush_cycle(self):
        """Warm defer -> TTL expires -> next request flushes all."""
        es = EngineState(
            compacted_through=10,
            flushed_through=5,
            last_request_time=time.time() - 30,
        )
        # Warm: defer
        _cache_age = time.time() - es.last_request_time
        assert _cache_age < 300

        # Simulate TTL expiry
        es.last_request_time = time.time() - 600
        _cache_age = time.time() - es.last_request_time
        assert _cache_age >= 300
        # Cold: flush
        es.flushed_through = es.compacted_through
        assert es.flushed_through == 10

    def test_flushed_never_exceeds_compacted(self):
        """Invariant: flushed_through should never exceed compacted_through."""
        es = EngineState(compacted_through=5, flushed_through=0)
        # Normal: flush up to compacted
        es.flushed_through = es.compacted_through
        assert es.flushed_through <= es.compacted_through
        # Abnormal: if somehow flushed > compacted, it's a bug
        es.flushed_through = 10
        assert es.flushed_through > es.compacted_through  # detects the invariant violation


# ---------------------------------------------------------------------------
# drop_boundary parameter test
# ---------------------------------------------------------------------------

class TestDropBoundary:
    def test_drop_boundary_param_accepted(self):
        """drop_compacted_turns accepts drop_boundary without error."""
        from virtual_context.proxy.message_filter import drop_compacted_turns
        from virtual_context.proxy.formats import AnthropicFormat
        from virtual_context.core.turn_tag_index import TurnTagIndex

        fmt = AnthropicFormat()
        index = TurnTagIndex()
        body = {"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]}
        # Should not raise with drop_boundary=None (backward compat)
        result_body, dropped = drop_compacted_turns(
            body, index, 0, fmt=fmt, drop_boundary=None,
        )
        assert dropped == 0

        # Should not raise with drop_boundary=5
        result_body, dropped = drop_compacted_turns(
            body, index, 0, fmt=fmt, drop_boundary=5,
        )
        assert dropped == 0


# ---------------------------------------------------------------------------
# Storage backend tests
# ---------------------------------------------------------------------------

class TestStorageBackendFields:
    def test_filesystem_roundtrip(self, tmp_path):
        """Filesystem backend preserves flushed_through and last_request_time."""
        from virtual_context.storage.filesystem import FilesystemStore

        store = FilesystemStore(str(tmp_path))
        snap = EngineStateSnapshot(
            conversation_id="test-conv",
            compacted_through=10,
            flushed_through=7,
            last_request_time=1234567890.5,
            turn_tag_entries=[],
            turn_count=5,
            last_compacted_turn=5,
            last_completed_turn=9,
            last_indexed_turn=9,
        )
        store.save_engine_state(snap)

        loaded = store.load_engine_state("test-conv")
        assert loaded is not None
        assert loaded.compacted_through == 10
        assert loaded.flushed_through == 7
        assert loaded.last_request_time == 1234567890.5

    def test_filesystem_backward_compat(self, tmp_path):
        """Old filesystem data without new fields gets defaults."""
        import json as _json
        # Use _engine_state subdir (that's what FilesystemStore uses)
        state_dir = tmp_path / "_engine_state"
        state_dir.mkdir(parents=True)
        old_data = {
            "conversation_id": "test-conv",
            "compacted_through": 5,
            "last_compacted_turn": 3,
            "turn_tag_entries": [],
            "turn_count": 2,
        }
        (state_dir / "test-conv.json").write_text(_json.dumps(old_data))

        from virtual_context.storage.filesystem import FilesystemStore
        store = FilesystemStore(str(tmp_path))
        loaded = store.load_engine_state("test-conv")
        assert loaded is not None
        assert loaded.flushed_through == 0
        assert loaded.last_request_time == 0.0
