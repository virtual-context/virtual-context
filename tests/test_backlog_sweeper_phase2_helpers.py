"""Compaction-backlog sweeper Phase 2 engine-side helper tests.

Per compaction-backlog sweeper spec v1.4 §4 + §5 + plan §4.5. The
sweeper tick LOOP lives cloud-side and is not tested in the engine
repo. This file owns the four engine-side surfaces cloud invokes:

* ``CompactionSignal.priority`` widened to
  ``Literal["soft", "hard", "takeover", "backlog"]`` so cloud's
  backlog dispatcher can construct a signal that
  ``_run_compact_wrapper`` recognizes as a backlog source label.
* ``virtual_context.proxy.state.compaction_phase_count()`` returns
  the canonical phase plan length so cloud passes the same
  ``phase_count`` to ``claim_compaction_backlog`` as the proxy LLM
  compaction path uses.
* ``ContextStore.get_compaction_fence_mode()`` accessor: cloud's
  tick body reads the runtime fence holder via the store rather
  than ``os.environ`` so the active-tier precondition stays
  fail-closed for stores constructed in a weaker tier even if the
  env var is later flipped.
* ``ContextStore.reconstruct_history_for_conv(conversation_id) ->
  list[Message]`` rebuilds the full canonical history from
  ``get_all_canonical_turns``. The history includes previously
  compacted rows AND current uncompacted tagged rows so the
  dispatched recovery compaction does not silently truncate engine
  state to only the backlog-window rows; incomplete trailing turn
  groups without an assistant message are excluded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from virtual_context.core.compaction_fence import CompactionFenceMode
from virtual_context.proxy.state import compaction_phase_count
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CompactionSignal, Message


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# CompactionSignal.priority literal widening.
# ---------------------------------------------------------------------------


class TestCompactionSignalPriorityLiteral:
    @pytest.mark.parametrize(
        "priority", ["soft", "hard", "takeover", "backlog"],
    )
    def test_accepted_priorities(self, priority):
        sig = CompactionSignal(
            priority=priority,
            current_tokens=0, budget_tokens=0, overflow_tokens=0,
        )
        assert sig.priority == priority

    def test_backlog_priority_routes_through_c2r_gate_resolver(self):
        """The backlog dispatcher relies on
        ``ProxyState._resolve_c2r_gate`` to derive
        ``disable_replacement_passes=True`` from
        ``signal.priority == 'backlog'`` when the explicit kwarg is
        omitted. The signal carries the source label; the resolver
        is verified by test_compaction_c2r_gate. Here we confirm the
        type contract: a backlog signal constructs cleanly and the
        resolver agrees.
        """
        from virtual_context.proxy.state import ProxyState
        sig = CompactionSignal(
            priority="backlog",
            current_tokens=0, budget_tokens=0, overflow_tokens=0,
        )
        assert ProxyState._resolve_c2r_gate(sig, None) is True


# ---------------------------------------------------------------------------
# compaction_phase_count().
# ---------------------------------------------------------------------------


class TestCompactionPhaseCount:
    def test_returns_canonical_phase_plan_length(self):
        # The canonical plan is 7 phases per the
        # ``_COMPACT_PHASE_PLAN`` tuple. Cloud uses this to populate
        # ``compaction_operation.phase_count`` consistently with the
        # proxy LLM compaction path.
        assert compaction_phase_count() == 7

    def test_helper_is_a_callable_module_level_function(self):
        """Cloud imports ``compaction_phase_count`` directly from
        ``virtual_context.proxy.state``; the spec pins it as a
        public module-level callable (not a class attribute) so a
        ``from ... import compaction_phase_count`` works.
        """
        import virtual_context.proxy.state as proxy_state
        assert callable(proxy_state.compaction_phase_count)
        assert getattr(
            proxy_state.compaction_phase_count, "__module__",
        ) == "virtual_context.proxy.state"


# ---------------------------------------------------------------------------
# get_compaction_fence_mode() accessor on stores.
# ---------------------------------------------------------------------------


class TestGetCompactionFenceMode:
    @pytest.mark.parametrize("mode", [
        CompactionFenceMode.OFF,
        CompactionFenceMode.OBSERVE,
        CompactionFenceMode.ACTIVE,
    ])
    def test_accessor_returns_pinned_holder(
        self, tmp_path: Path, mode: CompactionFenceMode,
    ):
        store = SQLiteStore(
            tmp_path / f"fm-{mode.value}.db",
            compaction_fence_mode=mode,
        )
        assert store.get_compaction_fence_mode() is mode

    def test_composite_forwarder_returns_own_holder(self, tmp_path: Path):
        from virtual_context.core.composite_store import CompositeStore
        seg = SQLiteStore(
            tmp_path / "cs.db",
            compaction_fence_mode=CompactionFenceMode.ACTIVE,
        )
        composite = CompositeStore(
            segments=seg, facts=seg, fact_links=seg,
            state=seg, search=seg,
            compaction_fence_mode=CompactionFenceMode.ACTIVE,
        )
        assert composite.get_compaction_fence_mode() is CompactionFenceMode.ACTIVE


# ---------------------------------------------------------------------------
# reconstruct_history_for_conv().
# ---------------------------------------------------------------------------


def _seed_canonical_row(
    store: SQLiteStore,
    *,
    conv_id: str,
    canonical_turn_id: str,
    sort_key: float,
    user_content: str,
    assistant_content: str,
    compacted: bool = False,
) -> None:
    now = _iso(_now())
    conn = store._get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv_id, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO conversations "
        "(conversation_id, tenant_id, phase, lifecycle_epoch, "
        " created_at, updated_at) "
        "VALUES (?, 't-rh', 'active', 1, ?, ?)",
        (conv_id, now, now),
    )
    conn.execute(
        """INSERT INTO canonical_turns
           (canonical_turn_id, conversation_id, sort_key, turn_hash,
            hash_version, user_content, assistant_content,
            tagged_at, compacted_at, created_at, updated_at)
           VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
        (canonical_turn_id, conv_id, sort_key,
         f"h-{canonical_turn_id}", user_content, assistant_content,
         now, now if compacted else None, now, now),
    )
    conn.commit()


class TestReconstructHistoryForConv:
    def test_includes_compacted_prefix_and_uncompacted_tail(
        self, tmp_path: Path,
    ):
        """Per spec §5.2: the reconstructed history MUST include
        previously compacted rows so the dispatched recovery
        compaction sees the conversation's actual prior state, not
        just the backlog window.
        """
        store = SQLiteStore(tmp_path / "rh.db")
        conv = "conv-rh"
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-1", sort_key=1.0,
            user_content="u1", assistant_content="a1", compacted=True,
        )
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-2", sort_key=2.0,
            user_content="u2", assistant_content="a2", compacted=True,
        )
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-3", sort_key=3.0,
            user_content="u3", assistant_content="a3", compacted=False,
        )
        history = store.reconstruct_history_for_conv(conv)
        # 3 turns * (user + assistant) = 6 messages.
        assert len(history) == 6
        # Canonical order preserved.
        assert [m.role for m in history] == [
            "user", "assistant", "user", "assistant", "user", "assistant",
        ]
        assert [m.content for m in history] == [
            "u1", "a1", "u2", "a2", "u3", "a3",
        ]

    def test_skips_trailing_turn_without_assistant(self, tmp_path: Path):
        """Per spec §5.2: incomplete trailing turn groups (user
        message but no assistant response) MUST be excluded so the
        dispatched compaction does not see a half-formed turn at
        the end of history.
        """
        store = SQLiteStore(tmp_path / "rh-trail.db")
        conv = "conv-rh-trail"
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-1", sort_key=1.0,
            user_content="u1", assistant_content="a1",
        )
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-2", sort_key=2.0,
            user_content="u2-trailing", assistant_content="",
        )
        history = store.reconstruct_history_for_conv(conv)
        assert len(history) == 2
        assert history[0] == Message(role="user", content="u1")
        assert history[1] == Message(role="assistant", content="a1")

    def test_preserves_canonical_message_content(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "rh-content.db")
        conv = "conv-rh-content"
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-1", sort_key=1.0,
            user_content="  u1\n", assistant_content="\n  a1  ",
        )
        history = store.reconstruct_history_for_conv(conv)
        assert [m.content for m in history] == ["  u1\n", "\n  a1  "]

    def test_empty_conversation_returns_empty_history(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "rh-empty.db")
        history = store.reconstruct_history_for_conv("conv-none")
        assert history == []


class TestCompositeReconstructHistoryForConv:
    def test_prefers_delegate_native_reconstruction(self):
        from virtual_context.core.composite_store import CompositeStore

        class NativeSegments:
            def reconstruct_history_for_conv(self, conversation_id: str):
                assert conversation_id == "conv-native"
                return (Message(role="assistant", content="native"),)

        seg = NativeSegments()
        composite = CompositeStore(
            segments=seg, facts=seg, fact_links=seg, state=seg, search=seg,
        )
        assert composite.reconstruct_history_for_conv("conv-native") == [
            Message(role="assistant", content="native"),
        ]

    def test_falls_back_to_base_shape_when_delegate_lacks_native_method(self):
        from virtual_context.core.composite_store import CompositeStore

        class LegacySegments:
            def get_all_canonical_turns(self, conversation_id: str):
                assert conversation_id == "conv-legacy"
                return [
                    SimpleNamespace(
                        user_content=" u1 ",
                        assistant_content=" a1 ",
                    ),
                    SimpleNamespace(
                        user_content="u2",
                        assistant_content="",
                    ),
                ]

        seg = LegacySegments()
        composite = CompositeStore(
            segments=seg, facts=seg, fact_links=seg, state=seg, search=seg,
        )
        assert composite.reconstruct_history_for_conv("conv-legacy") == [
            Message(role="user", content=" u1 "),
            Message(role="assistant", content=" a1 "),
        ]
