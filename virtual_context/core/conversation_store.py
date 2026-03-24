"""Conversation-scoped store wrapper with durable lifecycle fencing."""

from __future__ import annotations

from typing import Any


class StaleConversationWriteError(RuntimeError):
    """Raised when a deleted/stale conversation generation attempts a write."""


class ConversationStoreView:
    """Wrap a store so stale workers cannot write after delete/recreate."""

    _GUARDED_METHODS = {
        "delete_segment",
        "save_engine_state",
        "save_request_capture",
        "save_tag_summary",
        "save_tool_call",
        "save_turn_message",
        "save_request_context",
        "set_fact_superseded",
        "set_tag_alias",
        "store_chunk_embeddings",
        "store_fact_links",
        "store_facts",
        "store_segment",
        "store_tag_summary_embedding",
        "store_tool_output",
        "update_fact_fields",
        "update_segment",
        "prune_turn_messages",
    }

    def __init__(self, store: Any, conversation_id: str, generation: int) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_conversation_id", conversation_id)
        object.__setattr__(self, "_generation", int(generation or 0))

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def generation(self) -> int:
        return self._generation

    def refresh_generation(self) -> int:
        activate = getattr(self._store, "activate_conversation", None)
        if callable(activate):
            self._generation = int(activate(self._conversation_id) or 0)
        return self._generation

    def begin_conversation_deletion(self, conversation_id: str) -> int:
        begin = getattr(self._store, "begin_conversation_deletion", None)
        if not callable(begin):
            return self._generation
        return int(begin(conversation_id) or 0)

    def get_conversation_generation(self, conversation_id: str) -> int:
        get_generation = getattr(self._store, "get_conversation_generation", None)
        if not callable(get_generation):
            return self._generation
        return int(get_generation(conversation_id) or 0)

    def is_conversation_generation_current(
        self,
        conversation_id: str,
        generation: int,
    ) -> bool:
        check = getattr(self._store, "is_conversation_generation_current", None)
        if not callable(check):
            return True
        return bool(check(conversation_id, generation))

    def _guard_write(self, method_name: str) -> None:
        if self.is_conversation_generation_current(
            self._conversation_id,
            self._generation,
        ):
            return
        raise StaleConversationWriteError(
            f"Suppressed stale write via {method_name} for "
            f"{self._conversation_id[:12]} generation={self._generation}"
        )

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._store, name)
        if callable(attr) and name in self._GUARDED_METHODS:
            def _guarded(*args, **kwargs):
                self._guard_write(name)
                return attr(*args, **kwargs)

            return _guarded
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_store", "_conversation_id", "_generation"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._store, name, value)
