"""SearchEngine: quote search and turn-by-tag retrieval.

Delegates find_quote (via core.quote_search) and get_turns_by_tag
(store segments + TurnTagIndex live turns).  Extracted from engine.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .quote_search import find_quote as _find_quote
from .quote_search import search_summaries as _search_summaries
from .store import ContextStore
from .turn_tag_index import TurnTagIndex

if TYPE_CHECKING:
    from .semantic_search import SemanticSearchManager
    from ..types import Message, SegmentMetadata, VirtualContextConfig

logger = logging.getLogger(__name__)


class SearchEngine:
    """Quote search and turn-by-tag retrieval.

    Constructor takes:
        store:           a ContextStore instance
        semantic:        a SemanticSearchManager instance
        turn_tag_index:  a TurnTagIndex instance (shared mutable reference)
        config:          a VirtualContextConfig instance
    """

    def __init__(
        self,
        store: ContextStore,
        semantic: SemanticSearchManager,
        turn_tag_index: TurnTagIndex,
        config: VirtualContextConfig,
    ) -> None:
        self._store = store
        self._semantic = semantic
        self._turn_tag_index = turn_tag_index
        self._config = config

    def find_quote(
        self,
        query: str,
        max_results: int | None = None,
        intent_context: str = "",
        session_filter: str = "",
        mode: str = "lookup",
    ) -> dict:
        if max_results is None:
            max_results = self._config.search.find_quote_default_results
        return _find_quote(
            self._store,
            self._semantic,
            query,
            max_results,
            intent_context=intent_context,
            session_filter=session_filter,
            mode=mode,
            conversation_id=self._config.conversation_id,
        )

    def get_turns_by_tag(
        self,
        tag: str,
        conversation_history: list[Message] | None = None,
    ) -> dict:
        """Return all raw turns associated with a tag, from both stored segments and live history.

        Stored turns come from compacted segments in the store.
        Live turns come from the TurnTagIndex matched against conversation_history.
        """
        from ..types import SegmentMetadata

        result: dict = {
            "tag": tag,
            "stored_turns": [],
            "live_turns": [],
            "total_turns": 0,
        }

        # Stored turns: segments matching this tag
        segments = self._store.get_segments_by_tags(
            tags=[tag], min_overlap=1, limit=500,
            conversation_id=self._config.conversation_id,
        )
        seen_refs: set[str] = set()
        for seg in segments:
            if seg.ref in seen_refs:
                continue
            seen_refs.add(seg.ref)
            meta = seg.metadata or SegmentMetadata(turn_count=0)
            result["stored_turns"].append({
                "segment_ref": seg.ref,
                "messages": seg.messages,
                "full_text": seg.full_text,
                "summary": seg.summary,
                "turn_count": meta.turn_count,
                "created_at": str(seg.created_at),
            })

        # Live turns: TurnTagIndex entries where tag appears,
        # paired with conversation_history messages or persisted turn_messages
        history = conversation_history or []
        missing_turn_numbers: list[int] = []
        live_entries: list[tuple] = []  # (entry, msg_idx)
        for entry in self._turn_tag_index.entries:
            if tag not in entry.tags:
                continue
            msg_idx = entry.turn_number * 2
            if msg_idx < len(history):
                messages = [{"role": "user", "content": history[msg_idx].content}]
                if msg_idx + 1 < len(history):
                    messages.append({"role": "assistant", "content": history[msg_idx + 1].content})
                result["live_turns"].append({
                    "turn_number": entry.turn_number,
                    "tags": entry.tags,
                    "primary_tag": entry.primary_tag,
                    "messages": messages,
                })
            else:
                missing_turn_numbers.append(entry.turn_number)
                live_entries.append(entry)

        # Fall back to canonical full_text for restored turns
        if missing_turn_numbers:
            persisted = self._store.get_full_text_rows(
                self._config.conversation_id, missing_turn_numbers,
            )
            for entry in live_entries:
                pair = persisted.get(entry.turn_number)
                messages = []
                if pair:
                    if pair.user_content:
                        messages.append({"role": "user", "content": pair.user_content})
                    if pair.assistant_content:
                        messages.append({"role": "assistant", "content": pair.assistant_content})
                result["live_turns"].append({
                    "turn_number": entry.turn_number,
                    "tags": entry.tags,
                    "primary_tag": entry.primary_tag,
                    "messages": messages,
                })

        result["total_turns"] = len(result["stored_turns"]) + len(result["live_turns"])
        return result

    def search_summaries(
        self,
        query: str,
        max_results: int | None = None,
        intent_context: str = "",
        session_filter: str = "",
        mode: str = "lookup",
    ) -> dict:
        if max_results is None:
            max_results = self._config.search.find_quote_default_results
        return _search_summaries(
            self._store,
            self._semantic,
            query,
            max_results,
            intent_context=intent_context,
            session_filter=session_filter,
            mode=mode,
            conversation_id=self._config.conversation_id,
        )
