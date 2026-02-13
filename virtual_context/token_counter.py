"""Token counting utilities."""

from __future__ import annotations

from typing import Callable


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def create_token_counter(mode: str = "estimate") -> Callable[[str], int]:
    """Factory for token counters.

    Modes:
        "estimate" - len(text) // 4 (zero deps)
        "tiktoken" - requires tiktoken package (v0.2)
        "callable:module.path:func" - custom callable (v0.2)
    """
    if mode == "estimate":
        return estimate_tokens

    if mode == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            return lambda text: len(enc.encode(text))
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Install with: pip install virtual-context[tiktoken]"
            )

    if mode.startswith("callable:"):
        # Format: callable:module.path:func_name
        parts = mode[len("callable:"):].rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid callable spec: {mode}. Expected callable:module:func")
        module_path, func_name = parts
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)

    raise ValueError(f"Unknown token counter mode: {mode}")
