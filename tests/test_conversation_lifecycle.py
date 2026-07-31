from __future__ import annotations

import pytest

from virtual_context.core.conversation_store import (
    ConversationStoreView,
    StaleConversationWriteError,
)
from virtual_context.core.exceptions import ConversationLifecycleConflict
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import EngineStateSnapshot


def test_conversation_store_view_blocks_stale_writes_after_delete(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    conversation_id = "conv-delete"

    generation0 = store.activate_conversation(conversation_id)
    view0 = ConversationStoreView(store, conversation_id, generation0)
    view0.save_canonical_turn(conversation_id, 0, "u0", "a0")
    assert store.get_canonical_turn_rows(conversation_id, [0])[0].user_content == "u0"
    assert store.get_canonical_turn_rows(conversation_id, [0])[0].assistant_content == "a0"

    deleted_generation = store.begin_conversation_deletion(conversation_id)
    with pytest.raises(StaleConversationWriteError):
        view0.save_canonical_turn(conversation_id, 1, "stale-u", "stale-a")

    store.delete_conversation(
        conversation_id,
        expected_generation=deleted_generation,
    )
    generation1 = store.activate_conversation(
        conversation_id,
        recreate_deleted=True,
    )
    assert generation1 == deleted_generation + 1

    view1 = ConversationStoreView(store, conversation_id, generation1)
    view1.save_canonical_turn(conversation_id, 0, "fresh-u", "fresh-a")
    assert store.get_canonical_turn_rows(conversation_id, [0])[0].user_content == "fresh-u"
    assert store.get_canonical_turn_rows(conversation_id, [0])[0].assistant_content == "fresh-a"


def test_conversation_store_view_blocks_chain_and_tool_link_writes_after_delete(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    conversation_id = "conv-delete"

    generation0 = store.activate_conversation(conversation_id)
    view0 = ConversationStoreView(store, conversation_id, generation0)
    store.begin_conversation_deletion(conversation_id)

    with pytest.raises(StaleConversationWriteError):
        view0.store_chain_snapshot("chain-1", conversation_id, 0, "{}", 0)

    with pytest.raises(StaleConversationWriteError):
        view0.link_turn_tool_output(conversation_id, 0, "tool-turn-1")

    with pytest.raises(StaleConversationWriteError):
        view0.link_segment_tool_output(conversation_id, "seg-1", "tool-seg-1")

    assert store.get_chain_snapshot(conversation_id, "chain-1") is None
    assert store.get_tool_outputs_for_turn(conversation_id, 0) == []
    assert store.get_tool_outputs_for_segment(conversation_id, "seg-1") == []


def test_stale_delete_cannot_destroy_recreated_generation(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    conversation_id = "conv-delete-race"

    store.activate_conversation(conversation_id)
    store.save_canonical_turn(conversation_id, 0, "old-u", "old-a")
    deleting_generation = store.begin_conversation_deletion(conversation_id)

    successor_generation = store.activate_conversation(
        conversation_id,
        recreate_deleted=True,
    )
    assert successor_generation == deleting_generation + 1
    store.save_canonical_turn(conversation_id, 1, "new-u", "new-a")

    with pytest.raises(ConversationLifecycleConflict):
        store.delete_conversation(
            conversation_id,
            expected_generation=deleting_generation,
        )

    rows = store.get_canonical_turn_rows(conversation_id, [0, 1])
    assert sorted(rows) == [0, 1]


def test_stale_engine_checkpoint_cannot_replace_recreated_generation(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    conversation_id = "conv-checkpoint-race"

    generation0 = store.activate_conversation(conversation_id)
    stale = EngineStateSnapshot(
        conversation_id=conversation_id,
        compacted_prefix_messages=0,
        turn_tag_entries=[],
        turn_count=0,
        last_completed_turn=7,
        conversation_generation=generation0,
    )
    store.save_engine_state(stale)

    deleting_generation = store.begin_conversation_deletion(conversation_id)
    store.delete_conversation(
        conversation_id,
        expected_generation=deleting_generation,
    )
    successor_generation = store.activate_conversation(
        conversation_id,
        recreate_deleted=True,
    )
    successor = EngineStateSnapshot(
        conversation_id=conversation_id,
        compacted_prefix_messages=0,
        turn_tag_entries=[],
        turn_count=0,
        last_completed_turn=1,
        conversation_generation=successor_generation,
    )
    store.save_engine_state(successor)

    with pytest.raises(ConversationLifecycleConflict):
        store.save_engine_state(stale)

    loaded = store.load_engine_state(conversation_id)
    assert loaded is not None
    assert loaded.conversation_generation == successor_generation
    assert loaded.last_completed_turn == 1
