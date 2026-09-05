"""Paging manager: expand/collapse topics in the working set.

Owns the working set dict and all depth-level calculations.
Extracted from engine.py. No engine-level state mutation.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable

from ..types import DepthLevel, WorkingSetEntry
from .store import ContextStore
from .rendered_memory import RenderedMemory

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
        conversation_id: str = "",
    ) -> None:
        self._store = store
        self._token_counter = token_counter
        self._tag_context_max_tokens = tag_context_max_tokens
        self._auto_evict_enabled = auto_evict
        self._paging_enabled = paging_enabled
        self._conversation_id = conversation_id
        self.working_set: dict[str, WorkingSetEntry] = {}
        self.rendered_memories: dict[str, RenderedMemory] = {}
        self._memory_renderer = None

    def set_memory_renderer(self, renderer: Callable) -> None:
        self._memory_renderer = renderer

    def render_memory(self, tag: str, depth: DepthLevel, *, speaker_context=None) -> RenderedMemory | None:
        if depth == DepthLevel.NONE or self._memory_renderer is None:
            return None
        return self._memory_renderer(tag, depth, speaker_context=speaker_context)

    def expand_topic(self, tag: str, depth: str = "full", *, speaker_context=None) -> dict:
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
            return self.collapse_topic(tag, "none", speaker_context=speaker_context)

        # Calculate token cost at target depth
        memory = self.render_memory(tag, target_depth, speaker_context=speaker_context)
        tokens_at_depth = memory.measured_cost if memory is not None else 0
        if tokens_at_depth == 0:
            return {"error": f"no stored content for tag: {tag}"}

        # Revalidate existing pages too: persisted token estimates and stale
        # source proofs cannot determine today's admission decision.
        planned: dict[str, WorkingSetEntry] = {}
        planned_memories: dict[str, RenderedMemory] = {}
        for name, entry in self.working_set.items():
            current_memory = self.render_memory(name, entry.depth, speaker_context=speaker_context)
            if current_memory is not None:
                planned_memories[name] = current_memory
                planned[name] = replace(entry, tokens=current_memory.measured_cost)
        current_total = sum(entry.tokens for entry in planned.values())
        current_tag_tokens = planned[tag].tokens if tag in planned else 0
        delta = tokens_at_depth - current_tag_tokens
        budget = self._tag_context_max_tokens
        # Auto-evict if over budget
        evicted_tags: list[str] = []
        tokens_evicted = 0
        if self._auto_evict_enabled and current_total + delta > budget:
            evicted_tags, tokens_evicted = self._auto_evict(
                needed=current_total + delta - budget,
                exclude_tag=tag,
                working_set=planned,
                rendered_memories=planned_memories,
                speaker_context=speaker_context,
            )

        # Check if expansion fits after eviction
        new_total = current_total + delta - tokens_evicted
        if new_total > budget:
            return {
                "error": "insufficient budget",
                "tag": tag,
                "needed": tokens_at_depth,
                "available": max(0, budget - (current_total - current_tag_tokens)),
            }

        # Update working set
        turn = max((ws.last_accessed_turn for ws in planned.values()), default=0)
        planned[tag] = WorkingSetEntry(
            tag=tag,
            depth=target_depth,
            tokens=tokens_at_depth,
            last_accessed_turn=turn + 1,
        )
        self.working_set = planned
        planned_memories[tag] = memory
        self.rendered_memories = planned_memories

        return {
            "tag": tag,
            "depth": target_depth.value,
            "tokens_added": delta,
            "tokens_evicted": tokens_evicted,
            "evicted_tags": evicted_tags,
        }

    def collapse_topic(self, tag: str, depth: str = "summary", *, speaker_context=None) -> dict:
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
            self.rendered_memories.pop(tag, None)
            return {"tag": tag, "depth": "none", "tokens_freed": old_tokens}

        memory = self.render_memory(tag, target_depth, speaker_context=speaker_context)
        new_tokens = memory.measured_cost if memory is not None else 0
        if memory is None:
            return {"error": f"no admitted content for tag: {tag}"}
        if new_tokens > old_tokens:
            return self.expand_topic(tag, depth, speaker_context=speaker_context)
        self.working_set[tag] = replace(self.working_set[tag], depth=target_depth, tokens=new_tokens)
        self.rendered_memories[tag] = memory

        return {
            "tag": tag,
            "depth": target_depth.value,
            "tokens_freed": max(0, old_tokens - new_tokens),
        }

    def get_working_set_summary(self) -> dict:
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

    def calculate_depth_tokens(self, tag: str, depth: DepthLevel, *, speaker_context=None) -> int:
        memory = self.render_memory(tag, depth, speaker_context=speaker_context)
        return memory.measured_cost if memory is not None else 0

    def _auto_evict(
        self, needed: int, exclude_tag: str = "", *,
        working_set: dict[str, WorkingSetEntry] | None = None,
        rendered_memories: dict[str, RenderedMemory] | None = None,
        speaker_context=None,
    ) -> tuple[list[str], int]:
        """Auto-evict coldest topics to free `needed` tokens.

        Returns (evicted_tag_names, total_tokens_freed).
        """
        target = self.working_set if working_set is None else working_set
        memories = self.rendered_memories if rendered_memories is None else rendered_memories
        # Sort by last_accessed_turn ascending (coldest first)
        candidates = sorted(
            ((tag, ws) for tag, ws in target.items() if tag != exclude_tag),
            key=lambda x: x[1].last_accessed_turn,
        )

        evicted: list[str] = []
        freed = 0
        for tag, ws in candidates:
            if freed >= needed:
                break
            # Collapse to SUMMARY (not NONE) to keep minimum context
            memory = self.render_memory(tag, DepthLevel.SUMMARY, speaker_context=speaker_context)
            summary_tokens = memory.measured_cost if memory is not None else 0
            delta = ws.tokens - summary_tokens
            if delta <= 0 or memory is None:
                # Already at summary or less, remove entirely
                freed += ws.tokens
                del target[tag]
                memories.pop(tag, None)
            else:
                freed += delta
                target[tag].depth = DepthLevel.SUMMARY
                target[tag].tokens = summary_tokens
                memories[tag] = memory
            evicted.append(tag)

        return evicted, freed
