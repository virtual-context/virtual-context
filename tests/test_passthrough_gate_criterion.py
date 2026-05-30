"""Routing gate is content-based, not watermark-based.

``resolve_prepare_state`` no longer treats ``has_pending_indexing()``
as a passthrough trigger. The new gate routes a prepare request to
ACTIVE whenever the engine has any retrievable content
(``compacted_prefix_messages > 0`` OR any ``turn_tag_index`` entries)
and to PASSTHROUGH on a cold-start conversation (neither). Bootstrap
and legacy-tagger-spawn paths continue to consult
``has_pending_indexing()``; the routing path does not.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from virtual_context.proxy.state import ProxyState, SessionState
from virtual_context.types import Message, TurnTagEntry


def _make_proxy_state(tmp_path: Path, conv: str = "c") -> ProxyState:
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.types import (
        RetrieverConfig,
        StorageConfig,
        TagGeneratorConfig,
        VirtualContextConfig,
    )
    config = VirtualContextConfig(
        conversation_id=conv,
        storage=StorageConfig(
            backend="sqlite", sqlite_path=str(tmp_path / f"{conv}.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
        retriever=RetrieverConfig(inbound_tagger_type="keyword"),
    )
    engine = VirtualContextEngine(config=config)
    return ProxyState(engine)


def _msg(role: str, content: str = "x") -> Message:
    return Message(
        role=role, content=content,
        timestamp=datetime.now(timezone.utc),
    )


def _completed_pair() -> list[Message]:
    return [_msg("user", "hello"), _msg("assistant", "hi")]


def _append_entry(state: ProxyState, turn_number: int) -> None:
    state.engine._turn_tag_index.append(TurnTagEntry(
        turn_number=turn_number,
        message_hash=f"h{turn_number}",
        tags=["_general"],
        primary_tag="_general",
    ))


# ---------------------------------------------------------------------------
# _has_retrievable_content
# ---------------------------------------------------------------------------


class TestHasRetrievableContent:
    @pytest.mark.parametrize("compacted,entries,expected", [
        (0, 0, False),
        (0, 1, True),
        (1, 0, True),
        (100, 5, True),
        (-1, 0, False),  # defensive on negative
    ])
    def test_parametrized(self, tmp_path, compacted, entries, expected):
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = compacted
        for i in range(entries):
            _append_entry(state, i)
        assert state._has_retrievable_content() is expected


# ---------------------------------------------------------------------------
# resolve_prepare_state
# ---------------------------------------------------------------------------


class TestResolvePrepareState:
    def test_compacted_content_routes_active(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 100
        _append_entry(state, 0)
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.ACTIVE
        assert reason is None

    def test_tag_index_only_routes_active(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 0
        for i in range(5):
            _append_entry(state, i)
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.ACTIVE
        assert reason is None

    def test_cold_start_routes_passthrough_initial_ingest(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 0
        # turn_tag_index empty
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.PASSTHROUGH
        # `initial_ingest` reason is preserved verbatim for downstream
        # log/dashboard compatibility.
        assert reason == "initial_ingest"

    def test_pending_indexing_does_not_trigger_passthrough(self, tmp_path):
        """Load-bearing test: the watermark inequality is no longer a
        routing predicate. Even with last_completed_turn > last_indexed_turn,
        a content-bearing conv routes ACTIVE.
        """
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 20
        # Drive has_pending_indexing() True via the watermark integers:
        # last_completed_turn=10 > last_indexed_turn=9 -> pending=True.
        state.engine._engine_state.last_completed_turn = 10
        state.engine._engine_state.last_indexed_turn = 9
        assert state.has_pending_indexing() is True
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.ACTIVE
        assert reason is None
        # Critical regression assertion: the legacy reason string is
        # never emitted from resolve_prepare_state under the new gate.
        assert reason != "pending_indexing"

    def test_manual_passthrough_still_wins(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 100
        state.set_manual_passthrough(True)
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.PASSTHROUGH
        assert reason == "manual_override"

    def test_empty_history_routes_active(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        result, reason = state.resolve_prepare_state([])
        assert result == SessionState.ACTIVE
        assert reason is None

    def test_bot_symptom_reproduction(self, tmp_path):
        """Conv 77f110fc-style state: compacted_prefix_messages=5658,
        last_completed_turn=2816, last_indexed_turn=2815. The legacy
        gate produced PASSTHROUGH reason=pending_indexing. The new
        gate routes ACTIVE.
        """
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 5658
        state.engine._engine_state.last_completed_turn = 2816
        state.engine._engine_state.last_indexed_turn = 2815
        # Seed enough turn_tag_index entries to mirror the real conv's
        # in-memory state (not strictly required for the assertion;
        # compacted_prefix alone is sufficient).
        for i in range(50):
            _append_entry(state, i)
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.ACTIVE
        assert reason is None


# ---------------------------------------------------------------------------
# _can_activate_from_persisted_state simplification
# ---------------------------------------------------------------------------


class TestCanActivateFromPersistedState:
    def test_returns_true_under_pending_indexing(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 10
        state.engine._engine_state.last_completed_turn = 5
        state.engine._engine_state.last_indexed_turn = 4
        assert state.has_pending_indexing() is True
        assert state._can_activate_from_persisted_state() is True

    def test_returns_true_when_client_posts_more_than_indexed(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        state.engine._engine_state.compacted_prefix_messages = 20
        for i in range(5):
            _append_entry(state, i)
        # Build a 10-turn history (20 messages).
        history = []
        for _ in range(10):
            history.extend(_completed_pair())
        assert state._can_activate_from_persisted_state(history) is True

    def test_returns_false_on_cold_start(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        assert state._can_activate_from_persisted_state() is False


# ---------------------------------------------------------------------------
# has_pending_indexing is preserved for bootstrap / spawn callers
# ---------------------------------------------------------------------------


class TestHasPendingIndexingPreserved:
    def test_callable_and_returns_correct_value(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        # No pending work initially.
        assert state.has_pending_indexing() is False
        # Create a watermark gap.
        state.engine._engine_state.last_completed_turn = 3
        state.engine._engine_state.last_indexed_turn = 2
        assert state.has_pending_indexing() is True


# ---------------------------------------------------------------------------
# Threading: bounded snapshot of turn_tag_index.entries length
# ---------------------------------------------------------------------------


class TestThreadingSnapshot:
    def test_concurrent_append_does_not_raise(self, tmp_path):
        """Stress test: a background thread appends to turn_tag_index
        while the main thread reads len() repeatedly. Asserts no
        RuntimeError and reads always return non-negative ints
        (GIL-atomic semantics).
        """
        state = _make_proxy_state(tmp_path)
        _append_entry(state, 0)

        stop = threading.Event()

        def appender():
            n = 1
            while not stop.is_set():
                _append_entry(state, n)
                n += 1

        t = threading.Thread(target=appender, daemon=True)
        t.start()
        try:
            for _ in range(2000):
                length = len(state.engine._turn_tag_index.entries)
                assert length >= 1
        finally:
            stop.set()
            t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Reason-string regression: `pending_indexing` never emitted from
# resolve_prepare_state
# ---------------------------------------------------------------------------


class TestPendingIndexingReasonNotEmitted:
    def test_under_pending_state_returns_other_reason(self, tmp_path):
        state = _make_proxy_state(tmp_path)
        # Force pending watermark gap via the integers
        # ``has_pending_indexing`` reads. Under the legacy gate this
        # would have produced reason ``pending_indexing``; under the
        # new gate the reason MUST be something else.
        state.engine._engine_state.last_completed_turn = 5
        state.engine._engine_state.last_indexed_turn = 4
        assert state.has_pending_indexing() is True
        # Case A: no compacted content yet -> passthrough but with a
        # non-``pending_indexing`` reason (either ``initial_ingest``
        # for fully-empty engines, or ``restore_not_ready`` when the
        # watermark integers suggest a partially-loaded restore).
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.PASSTHROUGH
        assert reason in {"initial_ingest", "restore_not_ready"}
        assert reason != "pending_indexing"
        # Case B: content present -> ACTIVE regardless of watermark gap.
        state.engine._engine_state.compacted_prefix_messages = 10
        result, reason = state.resolve_prepare_state(_completed_pair())
        assert result == SessionState.ACTIVE
        assert reason is None
        assert reason != "pending_indexing"
