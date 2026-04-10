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

    def update_segment(self, segment: StoredSegment) -> None:
        self._segments.update_segment(segment)

    def get_segment(self, ref: str, *, conversation_id: str | None = None) -> StoredSegment | None:
        return self._segments.get_segment(ref, conversation_id=conversation_id)

    def get_summary(self, ref: str, *, conversation_id: str | None = None) -> StoredSummary | None:
        return self._segments.get_summary(ref, conversation_id=conversation_id)

    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before=None,
        after=None,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        return self._segments.get_summaries_by_tags(
            tags, min_overlap=min_overlap, limit=limit, before=before, after=after,
            conversation_id=conversation_id,
        )

    def search(
        self, query: str, tags: list[str] | None = None, limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        return self._segments.search(query, tags=tags, limit=limit, conversation_id=conversation_id)

    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]:
        return self._segments.get_all_tags(conversation_id=conversation_id)

    def get_conversation_stats(self) -> list[ConversationStats]:
        return self._segments.get_conversation_stats()

    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[StoredSegment]:
        return self._segments.get_all_segments(
            conversation_id=conversation_id,
            limit=limit,
        )

    def get_tag_aliases(self, conversation_id: str | None = None) -> dict[str, str]:
        return self._segments.get_tag_aliases(conversation_id=conversation_id)

    def set_tag_alias(
        self,
        alias: str,
        canonical: str,
        conversation_id: str = "",
    ) -> None:
        return self._segments.set_tag_alias(
            alias,
            canonical,
            conversation_id=conversation_id,
        )

    def delete_segment(self, ref: str) -> bool:
        return self._segments.delete_segment(ref)

    def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int:
        return self._segments.cleanup(max_age=max_age, max_total_tokens=max_total_tokens)

    def save_tag_summary(self, tag_summary: TagSummary, conversation_id: str = "") -> None:
        return self._segments.save_tag_summary(tag_summary, conversation_id=conversation_id)

    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None:
        return self._segments.get_tag_summary(tag, conversation_id=conversation_id)

    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]:
        return self._segments.get_all_tag_summaries(conversation_id=conversation_id)

    def search_tag_summaries_fts(self, query: str, limit: int = 20, conversation_id: str | None = None) -> list[tuple[str, float]]:
        return self._segments.search_tag_summaries_fts(query, limit=limit, conversation_id=conversation_id)

    def store_tag_summary_embedding(self, tag: str, conversation_id: str, embedding: list[float]) -> None:
        return self._segments.store_tag_summary_embedding(tag, conversation_id, embedding)

    def load_tag_summary_embeddings(self, conversation_id: str | None = None) -> dict[str, list[float]]:
        return self._segments.load_tag_summary_embeddings(conversation_id=conversation_id)

    def get_segments_by_tags(
        self, tags: list[str], min_overlap: int = 1, limit: int = 20,
        conversation_id: str | None = None,
    ) -> list[StoredSegment]:
        return self._segments.get_segments_by_tags(tags, min_overlap=min_overlap, limit=limit, conversation_id=conversation_id)

    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]:
        return self._segments.get_orphan_tag_snippets(limit=limit)

    def delete_conversation(self, conversation_id: str) -> int:
        deleted = 0
        seen: set[int] = set()
        for store in (
            self._segments,
            self._facts,
            self._fact_links,
            self._state,
            self._search,
        ):
            marker = id(store)
            if marker in seen or not hasattr(store, "delete_conversation"):
                continue
            seen.add(marker)
            deleted = max(
                deleted,
                int(store.delete_conversation(conversation_id) or 0),
            )
        return deleted

    def delete_tag_aliases_for_conversation(self, conversation_id: str) -> int:
        delete_aliases = getattr(self._segments, "delete_tag_aliases_for_conversation", None)
        if callable(delete_aliases):
            return int(delete_aliases(conversation_id) or 0)
        return 0

    def save_turn_message(
        self, conversation_id: str, turn_number: int,
        user_content: str, assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
    ) -> None:
        return self._segments.save_turn_message(
            conversation_id, turn_number, user_content, assistant_content,
            user_raw_content=user_raw_content,
            assistant_raw_content=assistant_raw_content,
        )

    def get_turn_messages(
        self, conversation_id: str, turn_numbers: list[int],
    ) -> dict[int, tuple[str, str, str | None, str | None]]:
        return self._segments.get_turn_messages(conversation_id, turn_numbers)

    def load_recent_turn_messages(
        self, conversation_id: str, limit: int = 100,
    ) -> list[tuple[int, str, str]]:
        return self._segments.load_recent_turn_messages(conversation_id, limit=limit)

    def prune_turn_messages(self, conversation_id: str, keep_from_turn: int) -> int:
        return self._segments.prune_turn_messages(conversation_id, keep_from_turn)

    def search_turn_messages(
        self, query: str, limit: int = 5, conversation_id: str | None = None,
    ) -> list:
        _search = getattr(self._segments, "search_turn_messages", None)
        if callable(_search):
            return _search(query, limit=limit, conversation_id=conversation_id)
        return []

    def save_conversation_alias(self, alias_id: str, target_id: str) -> None:
        self._segments.save_conversation_alias(alias_id, target_id)

    def resolve_conversation_alias(self, alias_id: str) -> str | None:
        return self._segments.resolve_conversation_alias(alias_id)

    # ------------------------------------------------------------------
    # FactStore
    # ------------------------------------------------------------------

    def store_facts(self, facts: list[Fact]) -> int:
        return self._facts.store_facts(facts)

    def query_facts(self, **kwargs) -> list[Fact]:
        return self._facts.query_facts(**kwargs)

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        return self._facts.get_unique_fact_verbs(conversation_id=conversation_id)

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        return self._facts.get_facts_by_segment(segment_ref)

    def replace_facts_for_segment(self, conversation_id: str, segment_ref: str, facts: list) -> tuple[int, int]:
        return self._facts.replace_facts_for_segment(conversation_id, segment_ref, facts)

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list[Fact]:
        return self._facts.search_facts(query, limit=limit, conversation_id=conversation_id)

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        return self._facts.set_fact_superseded(old_fact_id, new_fact_id)

    def update_fact_fields(
        self, fact_id: str, verb: str, object: str, status: str, what: str,
    ) -> None:
        return self._facts.update_fact_fields(fact_id, verb, object, status, what)

    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]:
        return self._facts.get_fact_count_by_tags(conversation_id=conversation_id)

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        return self._facts.get_superseded_facts(fact_ids)

    def get_actionable_fact_tags(self, tags: list[str], conversation_id: str | None = None) -> set[str]:
        return self._facts.get_actionable_fact_tags(tags, conversation_id=conversation_id)

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

    def save_request_capture(self, capture: dict) -> None:
        return self._state.save_request_capture(capture)

    def load_request_captures(
        self,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[dict]:
        return self._state.load_request_captures(
            limit=limit,
            conversation_id=conversation_id,
        )

    def activate_conversation(self, conversation_id: str) -> int:
        lifecycle = getattr(self._state, "activate_conversation", None)
        if callable(lifecycle):
            return int(lifecycle(conversation_id) or 0)
        return 0

    def begin_conversation_deletion(self, conversation_id: str) -> int:
        lifecycle = getattr(self._state, "begin_conversation_deletion", None)
        if callable(lifecycle):
            return int(lifecycle(conversation_id) or 0)
        return 0

    def get_conversation_generation(self, conversation_id: str) -> int:
        lifecycle = getattr(self._state, "get_conversation_generation", None)
        if callable(lifecycle):
            return int(lifecycle(conversation_id) or 0)
        return 0

    def is_conversation_generation_current(
        self,
        conversation_id: str,
        generation: int,
    ) -> bool:
        lifecycle = getattr(self._state, "is_conversation_generation_current", None)
        if callable(lifecycle):
            return bool(lifecycle(conversation_id, generation))
        return True

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

    def search_tool_outputs(self, query: str, limit: int = 5, conversation_id: str | None = None) -> list:
        return self._search.search_tool_outputs(query, limit=limit, conversation_id=conversation_id)

    def save_tool_call(self, call: dict) -> None:
        return self._search.save_tool_call(call)

    def load_tool_calls(self, conversation_id: str, limit: int = 50) -> list[dict]:
        return self._search.load_tool_calls(conversation_id, limit=limit)

    def load_tool_call(self, call_id: int) -> dict | None:
        return self._search.load_tool_call(call_id)

    def link_turn_tool_output(self, conversation_id: str, turn_number: int, tool_output_ref: str) -> None:
        return self._search.link_turn_tool_output(conversation_id, turn_number, tool_output_ref)

    def get_tool_outputs_for_turn(self, conversation_id: str, turn_number: int) -> list[str]:
        return self._search.get_tool_outputs_for_turn(conversation_id, turn_number)

    def link_segment_tool_output(self, conversation_id: str, segment_ref: str, tool_output_ref: str) -> None:
        return self._search.link_segment_tool_output(conversation_id, segment_ref, tool_output_ref)

    def get_tool_outputs_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        return self._search.get_tool_outputs_for_segment(conversation_id, segment_ref)

    def get_tool_output_refs_for_turn(self, conversation_id: str, turn: int) -> list[str]:
        return self._search.get_tool_output_refs_for_turn(conversation_id, turn)

    def get_tool_output_by_ref(self, conversation_id: str, ref: str) -> str | None:
        return self._search.get_tool_output_by_ref(conversation_id, ref)

    def store_media_output(self, ref: str, conversation_id: str, media_type: str, width: int, height: int, original_bytes: int, compressed_bytes: int, file_path: str) -> None:
        return self._search.store_media_output(ref, conversation_id, media_type, width, height, original_bytes, compressed_bytes, file_path)

    def get_media_output(self, conversation_id: str, ref: str) -> dict | None:
        return self._search.get_media_output(conversation_id, ref)

    def store_chain_snapshot(
        self,
        ref: str,
        conversation_id: str,
        turn_number: int,
        chain_json: str,
        message_count: int,
        tool_output_refs: str = "",
    ) -> None:
        return self._search.store_chain_snapshot(
            ref, conversation_id, turn_number, chain_json, message_count, tool_output_refs,
        )

    def get_chain_snapshot(self, conversation_id: str, ref: str) -> dict | None:
        return self._search.get_chain_snapshot(conversation_id, ref)

    def get_chain_snapshots_for_conversation(self, conversation_id: str, min_turn: int = 0) -> list[dict]:
        return self._search.get_chain_snapshots_for_conversation(conversation_id, min_turn=min_turn)

    def get_chain_recovery_manifest(self, conversation_id: str, min_turn: int = 0) -> list[dict]:
        return self._search.get_chain_recovery_manifest(conversation_id, min_turn=min_turn)

    def get_tool_names_for_refs(self, refs: list[str]) -> list[str]:
        return self._search.get_tool_names_for_refs(refs)

    def get_tool_names_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        return self._search.get_tool_names_for_segment(conversation_id, segment_ref)

    def save_request_context(self, context: dict) -> None:
        return self._search.save_request_context(context)

    def load_request_contexts(self, conversation_id: str, limit: int = 50) -> list[dict]:
        return self._search.load_request_contexts(conversation_id, limit=limit)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        closed: set[int] = set()
        for sub in (self._segments, self._facts, self._fact_links, self._state, self._search):
            sid = id(sub)
            if sid not in closed and hasattr(sub, "close"):
                sub.close()
                closed.add(sid)
