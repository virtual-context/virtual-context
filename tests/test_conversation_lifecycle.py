from __future__ import annotations

import pytest

from virtual_context.core.conversation_store import (
    ConversationStoreView,
    StaleConversationWriteError,
)
from virtual_context.storage.sqlite import SQLiteStore


def test_conversation_store_view_blocks_stale_writes_after_delete(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    conversation_id = "conv-delete"

    generation0 = store.activate_conversation(conversation_id)
    view0 = ConversationStoreView(store, conversation_id, generation0)
    view0.save_turn_message(conversation_id, 0, "u0", "a0")
    assert store.get_turn_messages(conversation_id, [0])[0][:2] == ("u0", "a0")

    deleted_generation = store.begin_conversation_deletion(conversation_id)
    with pytest.raises(StaleConversationWriteError):
        view0.save_turn_message(conversation_id, 1, "stale-u", "stale-a")

    store.delete_conversation(conversation_id)
    generation1 = store.activate_conversation(conversation_id)
    assert generation1 == deleted_generation

    view1 = ConversationStoreView(store, conversation_id, generation1)
    view1.save_turn_message(conversation_id, 0, "fresh-u", "fresh-a")
    assert store.get_turn_messages(conversation_id, [0])[0][:2] == ("fresh-u", "fresh-a")
