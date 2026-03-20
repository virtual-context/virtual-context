"""Shared utility functions used by multiple engine delegates."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.semantic_search import SemanticSearchManager
    from ..types import Message

logger = logging.getLogger(__name__)


def extract_turn_pairs(history: list[Message]) -> list[tuple[str, str]]:
    """Extract user->assistant turn pairs from history, handling non-alternating messages.

    Returns list of (user_text, assistant_text) tuples. Skips preamble-only
    user messages (e.g., MemOS '# Role' injections) and handles consecutive
    user messages by using the last user message before each assistant response.
    """
    pairs: list[tuple[str, str]] = []
    last_user_text = ""
    for msg in history:
        if msg.role == "user":
            last_user_text = msg.content
        elif msg.role == "assistant" and last_user_text:
            pairs.append((last_user_text, msg.content))
            last_user_text = ""
    return pairs


def get_recent_context(
    history: list[Message],
    n_pairs: int,
    semantic: SemanticSearchManager,
    bleed_threshold: float,
    exclude_last: int = 2,
    current_text: str | None = None,
) -> list[str] | None:
    """Collect up to *n_pairs* recent user+assistant text strings.

    Walks backward from the end of *history* (skipping the last
    *exclude_last* messages which are the current turn) and returns
    alternating user/assistant content strings.

    When *current_text* is provided and *bleed_threshold* > 0,
    an embedding similarity gate checks whether the current turn is
    semantically related to the most recent context pair.  If the
    similarity is below the threshold (topic shift), context is skipped
    to prevent stale tags from bleeding across topics (BUG-010).

    Returns ``None`` when no context is available or when the gate blocks.
    """
    # Messages available for context (before the current turn)
    if exclude_last > 0 and len(history) > exclude_last:
        avail = history[:-exclude_last]
    elif exclude_last == 0:
        avail = list(history)
    else:
        avail = []
    if not avail:
        return None

    pairs: list[str] = []
    # Walk backward collecting user+assistant pairs
    i = len(avail) - 1
    collected = 0
    while i >= 1 and collected < n_pairs:
        if avail[i].role == "assistant" and avail[i - 1].role == "user":
            # Prepend so order is chronological
            pairs.insert(0, avail[i].content)
            pairs.insert(0, avail[i - 1].content)
            collected += 1
            i -= 2
        else:
            i -= 1

    if not pairs:
        return None

    # Context bleed gate (BUG-010): skip context on topic shift
    if (
        current_text
        and bleed_threshold > 0
        and not semantic.context_is_relevant(current_text, pairs)
    ):
        logger.debug("Context bleed gate: topic shift detected, skipping context")
        return None

    return pairs
