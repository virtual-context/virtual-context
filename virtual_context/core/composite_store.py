"""CompositeStore -- delegates each ContextStore method to the appropriate protocol."""

from __future__ import annotations

from datetime import timedelta

from .protocols import FactLinkStore, FactStore, SearchStore, SegmentStore, StateStore
from ..types import (
    ChunkEmbedding,
    ConversationStats,
    EngineStateSnapshot,
    Fact,
    FactLink,
    LinkedFact,
    QuoteResult,
    StoredSegment,
    StoredSummary,
    TagStats,
    TagSummary,
)


class CompositeStore:
    """Wires five protocol implementations into a single ContextStore-compatible surface.

    Each method is forwarded to whichever sub-store owns that concern:

    * ``SegmentStore``  -- segments, summaries, tags, aliases, cleanup
    * ``FactStore``     -- facts CRUD, querying, supersession
    * ``FactLinkStore`` -- fact link CRUD and traversal
    * ``StateStore``    -- engine state persistence
    * ``SearchStore``   -- full-text search, embeddings, tool outputs
    """

    def __init__(
        self,
        *,
        segments: SegmentStore,
        facts: FactStore,
        fact_links: FactLinkStore,
        state: StateStore,
        search: SearchStore,
    ) -> None:
        self._segments = segments
        self._facts = facts
        self._fact_links = fact_links
        self._state = state
        self._search = search

    # ------------------------------------------------------------------
    # SegmentStore
    # ------------------------------------------------------------------

    def store_segment(self, segment: StoredSegment) -> str:
        return self._segments.store_segment(segment)

    def get_segment(self, ref: str) -> StoredSegment | None:
        return self._segments.get_segment(ref)

    def get_summary(self, ref: str) -> StoredSummary | None:
        return self._segments.get_summary(ref)

    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before=None,
        after=None,
    ) -> list[StoredSummary]:
        return self._segments.get_summaries_by_tags(
            tags, min_overlap=min_overlap, limit=limit, before=before, after=after,
        )

    def search(
        self, query: str, tags: list[str] | None = None, limit: int = 5,
    ) -> list[StoredSummary]:
        return self._segments.search(query, tags=tags, limit=limit)

    def get_all_tags(self) -> list[TagStats]:
        return self._segments.get_all_tags()

    def get_conversation_stats(self) -> list[ConversationStats]:
        return self._segments.get_conversation_stats()

    def get_tag_aliases(self) -> dict[str, str]:
        return self._segments.get_tag_aliases()

    def set_tag_alias(self, alias: str, canonical: str) -> None:
        return self._segments.set_tag_alias(alias, canonical)

    def delete_segment(self, ref: str) -> bool:
        return self._segments.delete_segment(ref)

    def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int:
        return self._segments.cleanup(max_age=max_age, max_total_tokens=max_total_tokens)

    def save_tag_summary(self, tag_summary: TagSummary) -> None:
        return self._segments.save_tag_summary(tag_summary)

    def get_tag_summary(self, tag: str) -> TagSummary | None:
        return self._segments.get_tag_summary(tag)

    def get_all_tag_summaries(self) -> list[TagSummary]:
        return self._segments.get_all_tag_summaries()

    def get_segments_by_tags(
        self, tags: list[str], min_overlap: int = 1, limit: int = 20,
    ) -> list[StoredSegment]:
        return self._segments.get_segments_by_tags(tags, min_overlap=min_overlap, limit=limit)

    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]:
        return self._segments.get_orphan_tag_snippets(limit=limit)

    # ------------------------------------------------------------------
    # FactStore
    # ------------------------------------------------------------------

    def store_facts(self, facts: list[Fact]) -> int:
        return self._facts.store_facts(facts)

    def query_facts(self, **kwargs) -> list[Fact]:
        return self._facts.query_facts(**kwargs)

    def get_unique_fact_verbs(self) -> list[str]:
        return self._facts.get_unique_fact_verbs()

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        return self._facts.get_facts_by_segment(segment_ref)

    def search_facts(self, query: str, limit: int = 10) -> list[Fact]:
        return self._facts.search_facts(query, limit=limit)

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        return self._facts.set_fact_superseded(old_fact_id, new_fact_id)

    def update_fact_fields(
        self, fact_id: str, verb: str, object: str, status: str, what: str,
    ) -> None:
        return self._facts.update_fact_fields(fact_id, verb, object, status, what)

    def get_fact_count_by_tags(self) -> dict[str, int]:
        return self._facts.get_fact_count_by_tags()

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        return self._facts.get_superseded_facts(fact_ids)

    # ------------------------------------------------------------------
    # FactLinkStore
    # ------------------------------------------------------------------

    def store_fact_links(self, links: list[FactLink]) -> int:
        return self._fact_links.store_fact_links(links)

    def get_fact_links(self, fact_id: str, direction: str = "both") -> list[FactLink]:
        return self._fact_links.get_fact_links(fact_id, direction=direction)

    def get_linked_facts(self, fact_ids: list[str], depth: int = 1) -> list[LinkedFact]:
        return self._fact_links.get_linked_facts(fact_ids, depth=depth)

    def delete_fact_links(self, fact_id: str) -> int:
        return self._fact_links.delete_fact_links(fact_id)

    # ------------------------------------------------------------------
    # StateStore
    # ------------------------------------------------------------------

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        return self._state.save_engine_state(state)

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        return self._state.load_engine_state(conversation_id)

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        return self._state.load_latest_engine_state()

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        return self._state.list_engine_state_fingerprints()

    # ------------------------------------------------------------------
    # SearchStore
    # ------------------------------------------------------------------

    def search_full_text(self, *args, **kwargs) -> list[QuoteResult]:
        return self._search.search_full_text(*args, **kwargs)

    def store_chunk_embeddings(
        self, segment_ref: str, chunks: list[ChunkEmbedding],
    ) -> None:
        return self._search.store_chunk_embeddings(segment_ref, chunks)

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        return self._search.get_all_chunk_embeddings()

    def store_tool_output(
        self,
        ref: str,
        conversation_id: str,
        tool_name: str,
        command: str,
        turn: int,
        content: str,
        original_bytes: int,
    ) -> None:
        return self._search.store_tool_output(
            ref, conversation_id, tool_name, command, turn, content, original_bytes,
        )

    def search_tool_outputs(self, query: str, limit: int = 5) -> list:
        return self._search.search_tool_outputs(query, limit=limit)
