"""link_predecessor: idempotent predecessor-conversation linking.

Exercises the 4-branch decision tree:
- same id / same terminal -> noop
- predecessor or stable bound to a different terminal -> conflict (never repoint)
- stable empty + predecessor has data -> alias stable -> predecessor
- predecessor empty -> alias predecessor -> stable
- both have data -> needs_merge (no alias written)
"""

import pytest

from virtual_context.proxy.vcattach import link_predecessor


@pytest.fixture
def store(tmp_path):
    from virtual_context.storage.sqlite import SQLiteStore
    s = SQLiteStore(str(tmp_path / "test.db"))
    yield s
    s.close()


def _seed_turns(store, conversation_id: str, n: int = 2) -> None:
    for i in range(n):
        store.save_canonical_turn(
            conversation_id,
            i,
            f"user message {i}",
            f"assistant reply {i}",
        )


PRED = "11111111-1111-1111-1111-111111111111"
STABLE = "sk:agent:bastkid-dedicated:telegram:group:-5156869263"
OTHER = "22222222-2222-2222-2222-222222222222"


class TestNoop:
    def test_same_id_noop(self, store):
        res = link_predecessor(PRED, PRED, store)
        assert res.outcome == "noop"
        assert res.reason == "same_id"
        assert store.resolve_conversation_alias(PRED) is None

    def test_blank_predecessor_noop(self, store):
        res = link_predecessor("", STABLE, store)
        assert res.outcome == "noop"
        assert res.reason == "invalid_input"

    def test_blank_stable_noop(self, store):
        res = link_predecessor(PRED, "  ", store)
        assert res.outcome == "noop"
        assert res.reason == "invalid_input"

    def test_predecessor_already_aliased_to_stable_noop(self, store):
        store.save_conversation_alias(PRED, STABLE)
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "noop"
        assert res.reason == "already_linked"
        # Alias unchanged.
        assert store.resolve_conversation_alias(PRED) == STABLE

    def test_replay_after_linked_is_noop(self, store):
        _seed_turns(store, PRED)
        first = link_predecessor(PRED, STABLE, store)
        assert first.outcome == "linked"
        second = link_predecessor(PRED, STABLE, store)
        assert second.outcome == "noop"
        assert second.reason == "already_linked"


class TestConflict:
    def test_predecessor_bound_elsewhere_never_repoints(self, store):
        store.save_conversation_alias(PRED, OTHER)
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "conflict"
        assert res.reason == "predecessor_bound_elsewhere"
        # Never repoint: existing alias intact, no new alias.
        assert store.resolve_conversation_alias(PRED) == OTHER
        assert store.resolve_conversation_alias(STABLE) is None

    def test_stable_bound_elsewhere_never_repoints(self, store):
        store.save_conversation_alias(STABLE, OTHER)
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "conflict"
        assert res.reason == "stable_bound_elsewhere"
        assert store.resolve_conversation_alias(STABLE) == OTHER
        assert store.resolve_conversation_alias(PRED) is None

    def test_alias_cycle_reports_conflict(self, store):
        # Manually corrupt: A -> B -> A. Walking must not loop forever
        # and must not write anything.
        store.save_conversation_alias(PRED, OTHER)
        store.save_conversation_alias(OTHER, PRED)
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "conflict"
        assert res.reason == "alias_chain_error"
        assert store.resolve_conversation_alias(STABLE) is None


class TestLinked:
    def test_stable_empty_predecessor_has_data_aliases_stable_to_predecessor(self, store):
        _seed_turns(store, PRED)
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "linked"
        assert res.alias_source == STABLE
        assert res.alias_target == PRED
        assert store.resolve_conversation_alias(STABLE) == PRED
        assert store.resolve_conversation_alias(PRED) is None

    def test_predecessor_empty_stable_has_data_aliases_predecessor_to_stable(self, store):
        _seed_turns(store, STABLE)
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "linked"
        assert res.alias_source == PRED
        assert res.alias_target == STABLE
        assert store.resolve_conversation_alias(PRED) == STABLE
        assert store.resolve_conversation_alias(STABLE) is None

    def test_both_empty_aliases_predecessor_to_stable(self, store):
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "linked"
        assert res.alias_source == PRED
        assert res.alias_target == STABLE

    def test_linked_fires_invalidation_callbacks(self, store):
        _seed_turns(store, PRED)
        invalidated: list[str] = []
        events: list[dict] = []
        res = link_predecessor(
            PRED,
            STABLE,
            store,
            registry_invalidate=invalidated.append,
            cross_worker_invalidate=events.append,
        )
        assert res.outcome == "linked"
        # execute_attach invalidates both ids and publishes alias events.
        assert set(invalidated) == {PRED, STABLE}
        assert any(e.get("type") == "alias_created" for e in events)


class TestNeedsMerge:
    def test_both_have_data_needs_merge_writes_nothing(self, store):
        _seed_turns(store, PRED)
        _seed_turns(store, STABLE)
        res = link_predecessor(PRED, STABLE, store)
        assert res.outcome == "needs_merge"
        assert store.resolve_conversation_alias(PRED) is None
        assert store.resolve_conversation_alias(STABLE) is None
