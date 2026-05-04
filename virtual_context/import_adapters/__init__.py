"""Conversation export adapters for importing historical chat data."""

from virtual_context.import_adapters.base import ExportAdapter
from virtual_context.import_adapters.chatgpt import ChatGPTAdapter
from virtual_context.import_adapters.claude import ClaudeAdapter
from virtual_context.import_adapters.grok import GrokAdapter

__all__ = ["ExportAdapter", "get_adapter", "ADAPTERS", "ChatGPTAdapter", "ClaudeAdapter", "GrokAdapter"]

ADAPTERS: dict[str, type[ExportAdapter]] = {
    "chatgpt": ChatGPTAdapter,
    "claude": ClaudeAdapter,
    "grok": GrokAdapter,
}


def get_adapter(provider: str) -> ExportAdapter:
    """Get adapter instance for the specified provider.

    Args:
        provider: Provider name (chatgpt, claude, grok).

    Returns:
        Instantiated adapter for the provider.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider not in ADAPTERS:
        supported = ", ".join(sorted(ADAPTERS.keys()))
        raise ValueError(f"Unknown provider '{provider}'. Supported: {supported}")
    return ADAPTERS[provider]()
