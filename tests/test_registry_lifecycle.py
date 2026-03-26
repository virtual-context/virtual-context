from __future__ import annotations

from types import SimpleNamespace

from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.registry import SessionRegistry


def _dummy_state(conversation_id: str):
    return SimpleNamespace(
        engine=SimpleNamespace(
            config=SimpleNamespace(conversation_id=conversation_id),
        ),
        shutdown=lambda *args, **kwargs: None,
    )


def test_remove_conversation_clears_all_routing_maps():
    registry = SessionRegistry(
        config_path=None,
        upstream="",
        metrics=ProxyMetrics(),
    )
    keep_state = _dummy_state("conv-keep")
    delete_state = _dummy_state("conv-delete")

    registry._conversations["conv-keep"] = keep_state
    registry._conversations["conv-delete"] = delete_state
    registry._sys_hashes = {"sys-keep": "conv-keep", "sys-delete": "conv-delete"}
    registry._chat_ids = {"chat-keep": "conv-keep", "chat-delete": "conv-delete"}
    registry._last_msg_hashes = {"msg-keep": "conv-keep", "msg-delete": "conv-delete"}

    removed = registry.remove_conversation("conv-delete")

    assert removed is delete_state
    assert registry.get_state("conv-delete") is None
    assert registry.get_state("conv-keep") is keep_state
    assert "conv-delete" not in registry._conversations
    assert "conv-delete" not in registry._sys_hashes.values()
    assert "conv-delete" not in registry._chat_ids.values()
    assert "conv-delete" not in registry._last_msg_hashes.values()
