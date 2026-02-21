"""Paging manager: expand/collapse topics in the working set.

Owns the working set dict and all depth-level calculations.
Extracted from engine.py. No engine-level state mutation.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..types import DepthLevel, WorkingSetEntry
from .store import ContextStore

logger = logging.getLogger(__name__)


class PagingManager:
    """Manages the working set: expand, collapse, evict, depth tokens."""

    def __init__(
        self,
        store: ContextStore,
        token_counter: Callable[[str], int],
        *,
        tag_context_max_tokens: int,
        auto_evict: bool = True,
        paging_enabled: bool = True,
    ) -> None:
        self._store = store
        self._token_counter = token_counter
        self._tag_context_max_tokens = tag_context_max_tokens
        self._auto_evict_enabled = auto_evict
        self._paging_enabled = paging_enabled
        self.working_set: dict[str, WorkingSetEntry] = {}

    def expand_topic(self, tag: str, depth: str = "full") -> dict:
        """Expand a topic to deeper detail in the working set.

        Returns dict with tag, depth, tokens_added, tokens_evicted, evicted_tags.
        """
        if not self._paging_enabled:
            return {"error": "paging not enabled"}

        try:
            target_depth = DepthLevel(depth)
        except ValueError:
            return {"error": f"invalid depth: {depth}"}

        if target_depth == DepthLevel.NONE:
            return self.collapse_topic(tag, "none")

        # Calculate token cost at target depth
        tokens_at_depth = self.calculate_depth_tokens(tag, target_depth)
        if tokens_at_depth == 0:
            return {"error": f"no stored content for tag: {tag}"}

        # Current working set total
        current_total = sum(ws.tokens for ws in self.working_set.values())
        current_tag_tokens = self.working_set[tag].tokens if tag in self.working_set else 0
        delta = tokens_at_depth - current_tag_tokens
        budget = self._tag_context_max_tokens

        # Auto-evict if over budget
        evicted_tags: list[str] = []
        tokens_evicted = 0
        if self._auto_evict_enabled and current_total + delta > budget:
            evicted_tags, tokens_evicted = self._auto_evict(
                needed=current_total + delta - budget,
                exclude_tag=tag,
            )

        # Check if expansion fits after eviction
        new_total = current_total + delta - tokens_evicted
        if new_total > budget:
            return {
                "error": "insufficient budget",
                "tag": tag,
                "needed": tokens_at_depth,
                "available": budget - (current_total - current_tag_tokens - tokens_evicted),
            }

        # Update working set
        turn = max((ws.last_accessed_turn for ws in self.working_set.values()), default=0)
        self.working_set[tag] = WorkingSetEntry(
            tag=tag,
            depth=target_depth,
            tokens=tokens_at_depth,
            last_accessed_turn=turn + 1,
        )

        return {
            "tag": tag,
            "depth": target_depth.value,
            "tokens_added": delta,
            "tokens_evicted": tokens_evicted,
            "evicted_tags": evicted_tags,
        }

    def collapse_topic(self, tag: str, depth: str = "summary") -> dict:
        """Collapse a topic to shallower detail. Returns freed tokens."""
        if not self._paging_enabled:
            return {"error": "paging not enabled"}

        try:
            target_depth = DepthLevel(depth)
        except ValueError:
            return {"error": f"invalid depth: {depth}"}

        if tag not in self.working_set:
            return {"tag": tag, "depth": target_depth.value, "tokens_freed": 0}

        old_tokens = self.working_set[tag].tokens

        if target_depth == DepthLevel.NONE:
            del self.working_set[tag]
            return {"tag": tag, "depth": "none", "tokens_freed": old_tokens}

        new_tokens = self.calculate_depth_tokens(tag, target_depth)
        self.working_set[tag].depth = target_depth
        self.working_set[tag].tokens = new_tokens

        return {
            "tag": tag,
            "depth": target_depth.value,
            "tokens_freed": max(0, old_tokens - new_tokens),
        }

    def get_working_set_summary(self) -> dict:
        """Return current working set with budget info."""
        budget = self._tag_context_max_tokens
        used = sum(ws.tokens for ws in self.working_set.values())
        entries = [
            {
                "tag": ws.tag,
                "depth": ws.depth.value,
                "tokens": ws.tokens,
                "last_accessed_turn": ws.last_accessed_turn,
            }
            for ws in sorted(self.working_set.values(), key=lambda w: w.last_accessed_turn, reverse=True)
        ]
        return {
            "budget": budget,
            "used": used,
            "available": budget - used,
            "entries": entries,
        }

    def calculate_depth_tokens(self, tag: str, depth: DepthLevel) -> int:
        """Calculate token cost for a tag at a given depth level."""
        if depth == DepthLevel.NONE:
            return 0

        if depth == DepthLevel.SUMMARY:
            ts = self._store.get_tag_summary(tag)
            return ts.summary_tokens if ts else 0

        # SEGMENTS or FULL: need stored segments
        segments = self._store.get_segments_by_tags(tags=[tag], min_overlap=1, limit=50)
        if not segments:
            return 0

        if depth == DepthLevel.SEGMENTS:
            return sum(s.summary_tokens for s in segments)
        else:  # FULL
            return sum(s.full_tokens or self._token_counter(s.full_text) for s in segments)

    def _auto_evict(self, needed: int, exclude_tag: str = "") -> tuple[list[str], int]:
        """Auto-evict coldest topics to free `needed` tokens.

        Returns (evicted_tag_names, total_tokens_freed).
        """
        # Sort by last_accessed_turn ascending (coldest first)
        candidates = sorted(
            ((tag, ws) for tag, ws in self.working_set.items() if tag != exclude_tag),
            key=lambda x: x[1].last_accessed_turn,
        )

        evicted: list[str] = []
        freed = 0
        for tag, ws in candidates:
            if freed >= needed:
                break
            # Collapse to SUMMARY (not NONE) to keep minimum context
            summary_tokens = self.calculate_depth_tokens(tag, DepthLevel.SUMMARY)
            delta = ws.tokens - summary_tokens
            if delta <= 0:
                # Already at summary or less, remove entirely
                freed += ws.tokens
                del self.working_set[tag]
            else:
                freed += delta
                self.working_set[tag].depth = DepthLevel.SUMMARY
                self.working_set[tag].tokens = summary_tokens
            evicted.append(tag)

        return evicted, freed
