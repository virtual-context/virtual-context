"""Token counting utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Bundled Anthropic tokenizer path (relative to this file)
_ANTHROPIC_TOKENIZER_PATH = Path(__file__).parent / "data" / "anthropic-tokenizer" / "tokenizer.json"


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def create_token_counter(mode: str = "estimate") -> Callable[[str], int]:
    """Factory for token counters.

    Modes:
        "estimate"  - len(text) // 4 (zero deps)
        "tiktoken"  - OpenAI tokenizer (requires tiktoken package)
        "anthropic" - bundled Claude tokenizer (~5.9% error vs API ground truth)
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

    if mode == "anthropic":
        try:
            from tokenizers import Tokenizer
        except ImportError:
            logger.warning("tokenizers package not installed, falling back to estimate mode")
            return estimate_tokens
        if not _ANTHROPIC_TOKENIZER_PATH.exists():
            logger.warning("Anthropic tokenizer file not found at %s, falling back to estimate mode",
                           _ANTHROPIC_TOKENIZER_PATH)
            return estimate_tokens
        tok = Tokenizer.from_file(str(_ANTHROPIC_TOKENIZER_PATH))
        logger.info("Loaded Anthropic tokenizer from %s", _ANTHROPIC_TOKENIZER_PATH)
        return lambda text: len(tok.encode(text).ids) if text else 0

    if mode.startswith("callable:"):
        raise ValueError(
            f"The 'callable:' token counter mode has been removed for security reasons. "
            f"Use 'estimate', 'tiktoken', or 'anthropic' instead."
        )

    raise ValueError(f"Unknown token counter mode: {mode}")


# Cache for format-specific counters (created lazily)
_format_counters: dict[str, Callable[[str], int]] = {}


def get_counter_for_format(api_format: str) -> Callable[[str], int]:
    """Return the best token counter for the given API format.

    - "anthropic" → anthropic tokenizer (falls back to tiktoken, then estimate)
    - everything else → tiktoken (falls back to estimate)
    """
    if api_format in _format_counters:
        return _format_counters[api_format]

    if api_format == "anthropic":
        order = ["anthropic", "tiktoken", "estimate"]
    else:
        order = ["tiktoken", "estimate"]

    for mode in order:
        try:
            counter = create_token_counter(mode)
            _format_counters[api_format] = counter
            logger.info("Token counter for %s: %s", api_format, mode)
            return counter
        except (ImportError, ValueError):
            continue

    # Should never reach here since estimate always works
    counter = estimate_tokens
    _format_counters[api_format] = counter
    return counter
