"""Token counting utilities."""

from __future__ import annotations

from typing import Callable


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def create_token_counter(mode: str = "estimate") -> Callable[[str], int]:
    """Factory for token counters.

    Modes:
        "estimate" - len(text) // 4 (zero deps)
        "tiktoken" - requires tiktoken package (v0.2)
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
        raise ValueError(
            f"The 'callable:' token counter mode has been removed for security reasons. "
            f"Use 'estimate' or 'tiktoken' instead."
        )

    raise ValueError(f"Unknown token counter mode: {mode}")
