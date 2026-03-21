# virtual_context/model_limits.py
"""Model-aware upstream context window limits."""

from __future__ import annotations

# Ordered list of (prefix, context_window). First match wins.
# Strip OpenRouter provider prefix (e.g., "anthropic/") before matching.
MODEL_CONTEXT_LIMITS: list[tuple[str, int]] = [
    # Anthropic
    ("claude-opus-4", 1_000_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3.5-sonnet", 200_000),
    ("claude-3-opus", 200_000),
    ("claude-", 200_000),
    # OpenAI
    ("gpt-5", 1_000_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4-mini", 200_000),
    # Google
    ("gemini-2.5", 1_000_000),
    ("gemini-2.0", 1_000_000),
    ("gemini-", 1_000_000),
    # Open source
    ("deepseek", 128_000),
    ("llama-4", 128_000),
]

DEFAULT_UPSTREAM_LIMIT = 200_000


def resolve_upstream_limit(
    model_name: str,
    instance_limit: int = 0,
    global_limit: int = 0,
) -> int:
    """Resolve the upstream context window for a model.

    Priority: instance config > global config > model-name auto-detect > fallback.
    """
    if instance_limit > 0:
        return instance_limit
    if global_limit > 0:
        return global_limit

    # Strip OpenRouter provider prefix (e.g., "anthropic/claude-opus-4-6" → "claude-opus-4-6")
    name = model_name
    if "/" in name:
        name = name.split("/", 1)[1]

    for prefix, limit in MODEL_CONTEXT_LIMITS:
        if name.startswith(prefix):
            return limit

    return DEFAULT_UPSTREAM_LIMIT
