"""Tests for the VCATTACH SessionState marker write + cross-worker ordering.

Covers the changes from ``119ac91`` + follow-up commits on
``fix/vcattach-redis-marker-write``: ``virtual_context/core/state_recovery.py``
+ ``virtual_context/proxy/vcattach.py:execute_attach`` Step A (marker
write) + the new strict T2-before-T1 cross-worker invalidation
ordering.

See ``docs/specs/vcattach-redis-marker-write-and-cross-worker-invalidation.md``
for the surrounding design.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.core.state_recovery import derive_session_state_markers
from virtual_context.proxy.session_state import SessionState
from virtual_context.proxy.vcattach import (
    _alias_created_event_for,
    _alias_deleted_event_for,
    execute_attach,
)


def _row(
    turn_number: int,
    *,
    tagged: bool = False,
    compacted: bool = False,
    primary_tag: str = "_general",
) -> SimpleNamespace:
    """Build a canonical_turn-row-shaped namespace for derivation tests."""
    return SimpleNamespace(
        turn_group_number=turn_number,
        canonical_turn_id=f"ct-{turn_number}",
        primary_tag=primary_tag,
        tags=[primary_tag] if primary_tag != "_general" else [],
        fact_signals=[],
        code_refs=[],
        sender="",
        session_date="",
        user_content=f"user-{turn_number}",
        assistant_content=f"assistant-{turn_number}",
        last_seen_at="2026-05-13T12:00:00+00:00",
        first_seen_at="2026-05-13T12:00:00+00:00",
        updated_at="2026-05-13T12:00:00+00:00",
        created_at="2026-05-13T12:00:00+00:00",
        tagged_at="2026-05-13T12:00:00+00:00" if tagged else None,
        compacted_at="2026-05-13T12:00:00+00:00" if compacted else None,
    )


def _store_with_rows(rows: list) -> SimpleNamespace:
    """Build a minimal store stub exposing the rows-iter API derivation uses."""
    return SimpleNamespace(
        get_all_canonical_turns=lambda conv_id: list(rows),
    )


# ---------------------------------------------------------------------------
# derive_session_state_markers — happy paths + edge cases
# ---------------------------------------------------------------------------


def test_derive_session_state_markers_from_canonical_rows():
    """Six tagged-and-compacted pairs (turns 0-2) plus three tagged-only
    pairs (turns 3-5). Expected:
    - compacted_prefix_messages = (last_compacted_turn + 1) * 2 = 6
    - last_compacted_turn = 2
    - last_completed_turn = 5 (any paired row advances)
    - last_indexed_turn = 5 (every pair is tagged)
    - turn_tag_entries: 6 entries (only tagged pairs included)
    - flushed_prefix_messages: 6 (mirrors compacted)
    - working_set: []
    """
    rows = [
        _row(0, tagged=True, compacted=True, primary_tag="topic-a"),
        _row(1, tagged=True, compacted=True, primary_tag="topic-b"),
        _row(2, tagged=True, compacted=True, primary_tag="topic-c"),
        _row(3, tagged=True, compacted=False, primary_tag="topic-d"),
        _row(4, tagged=True, compacted=False, primary_tag="topic-e"),
        _row(5, tagged=True, compacted=False, primary_tag="topic-f"),
    ]
    store = _store_with_rows(rows)

    state = derive_session_state_markers(store, "conv-1")

    assert state is not None
    assert state.compacted_prefix_messages == 6
    assert state.last_compacted_turn == 2
    assert state.last_completed_turn == 5
    assert state.last_indexed_turn == 5
    assert state.flushed_prefix_messages == 6
    assert len(state.turn_tag_entries) == 6
    assert state.working_set == []
    # Sentinel: derivation always bumps checkpoint_version to make the
    # marker write observable to version-check consumers.
    assert state.checkpoint_version >= 1


def test_derive_session_state_markers_empty_conv_returns_none():
    """Fresh conv with no canonical_turns → derivation returns None and
    caller must leave Redis untouched."""
    store = _store_with_rows([])

    state = derive_session_state_markers(store, "fresh-conv")

    assert state is None


def test_derive_session_state_markers_partial_pair_still_advances_last_completed():
    """Codex finding 6: pin the semantic that ``last_completed_turn``
    advances on any paired canonical row (matches
    ``_restore_from_canonical_rows`` semantics). A pair with only the
    user row populated still counts as completed; only ``tagged_at``
    presence gates ``last_indexed_turn``."""
    rows = [
        _row(0, tagged=True),
        _row(1, tagged=True),
        # turn 2: untagged — counts toward last_completed_turn but NOT
        # last_indexed_turn.
        _row(2, tagged=False),
    ]
    store = _store_with_rows(rows)

    state = derive_session_state_markers(store, "conv-partial")

    assert state is not None
    assert state.last_completed_turn == 2
    assert state.last_indexed_turn == 1
    # Only the two tagged pairs contribute turn_tag_entries.
    assert len(state.turn_tag_entries) == 2


def test_derive_session_state_markers_carries_existing_non_derivable_fields():
    """Codex finding 5: the carry-forward set covers session-scoped
    fields. checkpoint_version bumps on every successful repair so
    version-check consumers see a state change."""
    rows = [_row(0, tagged=True, compacted=True)]
    store = _store_with_rows(rows)
    existing = SessionState(
        compacted_prefix_messages=0,
        flushed_prefix_messages=0,
        last_completed_turn=-1,
        last_indexed_turn=-1,
        session_state="active",
        live_turn_count=500,
        history_message_count=1000,
        ingestion_done=20,
        ingestion_total=25,
        last_payload_kb=23062.4,
        last_payload_tokens=123456,
        raw_payload_entry_count=4632,
        ingestible_entry_count=989,
        skipped_payload_entry_count=17,
        checkpoint_version=7,
        conversation_generation=3,
        tool_tag_counter=42,
        split_processed_tags={"old-tag"},
        trailing_fingerprint="abc",
        provider="anthropic",
        telemetry_rollup={"requests": 5},
        request_captures=[{"turn": 1, "model": "claude"}],
        version=99,
    )

    state = derive_session_state_markers(store, "conv-existing", existing_state=existing)

    assert state is not None
    assert state.checkpoint_version == 8  # bumped from 7
    assert state.conversation_generation == 3
    assert state.tool_tag_counter == 42
    assert state.split_processed_tags == {"old-tag"}
    assert state.trailing_fingerprint == "abc"
    assert state.provider == "anthropic"
    assert state.session_state == "active"
    assert state.live_turn_count == 500
    assert state.history_message_count == 1000
    assert state.ingestion_done == 20
    assert state.ingestion_total == 25
    assert state.last_payload_kb == 23062.4
    assert state.last_payload_tokens == 123456
    assert state.raw_payload_entry_count == 4632
    assert state.ingestible_entry_count == 989
    assert state.skipped_payload_entry_count == 17
    assert state.telemetry_rollup == {"requests": 5}
    assert state.request_captures == [{"turn": 1, "model": "claude"}]
    assert state.version == 99


# ---------------------------------------------------------------------------
# execute_attach — marker write + T2-before-T1 ordering
# ---------------------------------------------------------------------------


def _execute_attach_test_store():
    """Build a minimal store that satisfies execute_attach's API surface."""
    aliases: dict[str, str] = {}

    def _save_alias(source, target):
        aliases[source] = target

    def _delete_alias(target):
        aliases.pop(target, None)

    def _get_all_canonical_turns(conv_id):
        return [
            _row(0, tagged=True, compacted=True),
            _row(1, tagged=True, compacted=False),
        ]

    store = SimpleNamespace(
        save_conversation_alias=_save_alias,
        delete_conversation_alias=_delete_alias,
        get_all_canonical_turns=_get_all_canonical_turns,
    )
    store._aliases_view = aliases
    return store


def test_execute_attach_writes_session_state_when_provider_supplied():
    """When ``session_state_provider`` is supplied, the marker write
    fires after alias commit and before cross-worker publish. Provider
    receives the derived SessionState."""
    store = _execute_attach_test_store()
    provider = MagicMock()
    provider.load.return_value = None  # no existing state

    execute_attach(
        old_id="src",
        target_id="tgt",
        store=store,
        session_state_provider=provider,
    )

    assert provider.save.call_count == 1
    saved_state = provider.save.call_args.args[1]
    assert isinstance(saved_state, SessionState)
    assert saved_state.compacted_prefix_messages == 2
    assert saved_state.last_completed_turn == 1
    assert saved_state.last_indexed_turn == 1


def test_execute_attach_skips_session_state_write_when_provider_none():
    """No provider supplied → no derivation, no save call. Alias write
    still fires (verified via the in-memory aliases dict)."""
    store = _execute_attach_test_store()

    execute_attach(
        old_id="src",
        target_id="tgt",
        store=store,
        session_state_provider=None,
    )

    assert store._aliases_view == {"src": "tgt"}


def test_execute_attach_swallows_provider_save_exception():
    """Provider.save raising must NOT propagate. Alias is committed;
    sibling workers will apply hydrate-time defensive recovery."""
    store = _execute_attach_test_store()
    provider = MagicMock()
    provider.load.return_value = None
    provider.save.side_effect = RuntimeError("redis transport blip")

    # Should not raise.
    execute_attach(
        old_id="src",
        target_id="tgt",
        store=store,
        session_state_provider=provider,
    )

    # Alias still landed.
    assert store._aliases_view == {"src": "tgt"}
    # Provider was attempted.
    assert provider.save.call_count == 1


def test_execute_attach_marker_write_commits_before_cross_worker_publish():
    """Codex finding 3: strict T2-before-T1 ordering pinned by call-order
    inspection. Records the invocation index of provider.save versus
    each cross_worker_invalidate call; asserts the save happens FIRST."""
    store = _execute_attach_test_store()
    provider = MagicMock()
    provider.load.return_value = None
    call_order: list[str] = []
    provider.save.side_effect = lambda conv, state: call_order.append("provider.save")

    def _xworker(event):
        call_order.append(f"cross_worker.{event.get('type', '?')}")

    execute_attach(
        old_id="src",
        target_id="tgt",
        store=store,
        cross_worker_invalidate=_xworker,
        session_state_provider=provider,
    )

    # provider.save must precede every cross_worker.* call.
    save_idx = call_order.index("provider.save")
    xworker_indices = [
        i for i, e in enumerate(call_order) if e.startswith("cross_worker.")
    ]
    assert xworker_indices, "cross_worker_invalidate should have fired at least once"
    assert all(i > save_idx for i in xworker_indices), (
        f"cross_worker_invalidate must fire AFTER provider.save; "
        f"call_order = {call_order}"
    )


def test_execute_attach_cross_worker_fires_when_marker_write_fails():
    """Codex finding 3 follow-on: even when provider.save raises, the
    cross-worker invalidation MUST still fire — the alias is committed
    and sibling workers must learn about it. They'll apply hydrate-time
    defensive recovery to patch missing markers from canonical_turns."""
    store = _execute_attach_test_store()
    provider = MagicMock()
    provider.load.return_value = None
    provider.save.side_effect = RuntimeError("redis down")
    received_events: list[dict] = []

    execute_attach(
        old_id="src",
        target_id="tgt",
        store=store,
        cross_worker_invalidate=lambda e: received_events.append(dict(e)),
        session_state_provider=provider,
    )

    types = [e["type"] for e in received_events]
    assert "alias_deleted" in types
    assert "alias_created" in types


def test_execute_attach_local_registry_invalidate_fires_for_both_ids():
    """Both old_id and target_id must be evicted from the local
    registry. Order: registry_invalidate runs AFTER provider.save AND
    after cross_worker_invalidate."""
    store = _execute_attach_test_store()
    provider = MagicMock()
    provider.load.return_value = None
    call_order: list[str] = []
    provider.save.side_effect = lambda *_: call_order.append("provider.save")

    def _registry(cid):
        call_order.append(f"registry.{cid}")

    execute_attach(
        old_id="src",
        target_id="tgt",
        store=store,
        registry_invalidate=_registry,
        cross_worker_invalidate=lambda e: call_order.append(f"xworker.{e.get('type', '?')}"),
        session_state_provider=provider,
    )

    assert "registry.src" in call_order
    assert "registry.tgt" in call_order
    save_idx = call_order.index("provider.save")
    last_registry_idx = max(
        i for i, e in enumerate(call_order) if e.startswith("registry.")
    )
    assert last_registry_idx > save_idx


def test_execute_attach_self_target_still_writes_markers():
    """Per spec Q4: marker write fires unconditionally on every VCATTACH
    success, including the self-VCATTACH path (resolver maps source to
    same id). Cheap idempotent repair."""
    store = _execute_attach_test_store()
    provider = MagicMock()
    provider.load.return_value = None

    execute_attach(
        old_id="X",
        target_id="X",
        store=store,
        session_state_provider=provider,
    )

    assert provider.save.call_count == 1
    saved_state = provider.save.call_args.args[1]
    assert saved_state.compacted_prefix_messages == 2


def test_alias_event_helpers_reach_composite_segments_builders():
    """CompositeStore keeps the alias event builders on its _segments
    backend; execute_attach should use that richer event shape."""
    inner = SimpleNamespace(
        _build_alias_deleted_event=lambda alias_id: {
            "type": "alias_deleted",
            "alias_id": alias_id,
            "reverse_dependents": ["child"],
        },
        _build_alias_created_event=lambda alias_id, target_id: {
            "type": "alias_created",
            "source": alias_id,
            "target": target_id,
            "reverse_dependents": ["child"],
        },
    )
    store = SimpleNamespace(_segments=inner)

    deleted = _alias_deleted_event_for(store, "target")
    created = _alias_created_event_for(store, "source", "target")

    assert deleted["reverse_dependents"] == ["child"]
    assert created["reverse_dependents"] == ["child"]


def test_alias_event_helpers_fallback_without_store_builders():
    """Fallback path remains stable for custom stores that expose alias
    persistence but no private event-builder helpers."""
    store = SimpleNamespace()

    assert _alias_deleted_event_for(store, "target") == {
        "type": "alias_deleted",
        "alias_id": "target",
    }
    assert _alias_created_event_for(store, "source", "target") == {
        "type": "alias_created",
        "source": "source",
        "target": "target",
    }
