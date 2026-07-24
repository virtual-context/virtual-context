"""Storage protocols -- focused interfaces for the pluggable storage layer."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from ..types import (
    ChunkEmbedding,
    ConversationStats,
    EngineStateSnapshot,
    Fact,
    FactLink,
    FactSignal,
    CanonicalTurnChunkEmbedding,
    CanonicalTurnRow,
    LinkedFact,
    QuoteResult,
    SpeakerHandleAssignment,
    SpeakerHandleCandidate,
    SpeakerRetrievalContext,
    StoredSegment,
    StoredSummary,
    TagStats,
    TagSummary,
)


@runtime_checkable
class SegmentStore(Protocol):
    """Segments, summaries, tag summaries, tags, and aliases."""

    def store_segment(
        self,
        segment: StoredSegment,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> str: ...
    def get_segment(self, ref: str, *, conversation_id: str | None = None) -> StoredSegment | None: ...
    def get_summary(self, ref: str, *, conversation_id: str | None = None) -> StoredSummary | None: ...
    def get_summaries_by_tags(
        self, tags: list[str], min_overlap: int = 1, limit: int = 10,
        before: datetime | None = None, after: datetime | None = None,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]: ...
    def search(self, query: str, tags: list[str] | None = None, limit: int = 5, conversation_id: str | None = None) -> list[StoredSummary]: ...
    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]: ...
    def get_conversation_stats(self) -> list[ConversationStats]: ...
    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[StoredSegment]: ...
    def get_tag_aliases(self, conversation_id: str | None = None) -> dict[str, str]: ...
    def set_tag_alias(
        self,
        alias: str,
        canonical: str,
        conversation_id: str = "",
    ) -> None: ...
    def delete_segment(self, ref: str) -> bool: ...
    def cleanup(self, max_age: timedelta | None = None, max_total_tokens: int | None = None) -> int: ...
    def save_tag_summary(
        self,
        tag_summary: TagSummary,
        conversation_id: str = "",
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None: ...
    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None: ...
    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]: ...
    def get_segments_by_tags(self, tags: list[str], min_overlap: int = 1, limit: int = 20, conversation_id: str | None = None) -> list[StoredSegment]: ...
    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]: ...
    def update_segment(
        self,
        segment: StoredSegment,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None: ...
    def delete_conversation(self, conversation_id: str) -> int: ...
    def save_canonical_turn(
        self,
        conversation_id: str,
        turn_number: int,
        user_content: str,
        assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
        primary_tag: str = "_general",
        tags: list[str] | None = None,
        session_date: str = "",
        sender: str = "",
        fact_signals: list[FactSignal] | None = None,
        code_refs: list[dict] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        canonical_turn_id: str | None = None,
        sort_key: float | None = None,
        turn_hash: str = "",
        hash_version: int = 0,
        normalized_user_text: str = "",
        normalized_assistant_text: str = "",
        tagged_at: str | None = None,
        compacted_at: str | None = None,
        first_seen_at: str | None = None,
        last_seen_at: str | None = None,
        source_batch_id: str | None = None,
        turn_group_number: int = -1,
        origin_channel_id: str = "",
        origin_channel_label: str = "",
        sender_actor_id: str = "",
        source_message_id: str = "",
        reply_target_message_id: str = "",
        reply_subject_actor_id: str = "",
        reply_subject_label: str = "",
        reply_target_body: str = "",
        reply_attribution_version: int = 0,
        audience_conversation_id: str = "",
        audience_attribution_version: int = 0,
    ) -> None: ...
    def search_canonical_turns_by_actor(
        self,
        actor_id: str,
        limit: int,
        conversation_id: str | None,
        *,
        speaker_context,
    ) -> list: ...
    def update_canonical_turn_senders_if_empty(
        self,
        conversation_id: str,
        updates: dict[str, str],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int: ...
    def update_canonical_turn_senders_if_matches(
        self,
        conversation_id: str,
        updates: dict[str, tuple[str, str]],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int: ...
    def update_canonical_turn_channels_if_empty(
        self,
        conversation_id: str,
        updates: dict[str, tuple[str, str]],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int: ...
    def update_canonical_turn_actors_if_empty(
        self,
        conversation_id: str,
        updates: dict[str, str],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int: ...
    def update_canonical_turn_reply_roles_if_empty(
        self,
        conversation_id: str,
        updates: dict[str, dict],
        *,
        expected_lifecycle_epoch: int | None = None,
    ) -> int: ...
    def find_canonical_turn_by_source_message_id(
        self,
        conversation_id: str,
        source_message_id: str,
        *,
        audience_conversation_id: str = "",
        origin_channel_id: str = "",
    ) -> "CanonicalTurnRow | None": ...
    def find_actor_ids_by_display_label(
        self,
        conversation_id: str,
        label: str,
        *,
        audience_conversation_id: str = "",
        origin_channel_id: str = "",
    ) -> list[str]: ...
    def list_canonical_conversation_ids(
        self,
        *,
        tenant_id: str | None = None,
        limit: int | None = None,
    ) -> list[str]: ...
    def recompute_canonical_turn_groups(
        self,
        conversation_id: str,
    ) -> int: ...
    def get_canonical_turn_rows(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, CanonicalTurnRow]: ...
    def get_canonical_turn_rows_by_id(
        self,
        keys: list[tuple[str, str]],
        *,
        speaker_context: SpeakerRetrievalContext,
    ) -> dict[tuple[str, str], CanonicalTurnRow]: ...
    def get_all_canonical_turns(
        self,
        conversation_id: str,
    ) -> list[CanonicalTurnRow]: ...
    def get_uncompacted_canonical_turns(
        self,
        conversation_id: str,
        *,
        protected_recent_turns: int = 0,
    ) -> list[CanonicalTurnRow]: ...
    def mark_canonical_turns_tagged(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        tagged_at: str | None = None,
    ) -> int: ...
    def mark_canonical_turns_compacted(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        compacted_at: str | None = None,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int: ...
    def delete_canonical_turns(
        self,
        conversation_id: str,
        turn_number: int | None = None,
    ) -> int: ...
    def replace_canonical_turn_anchors(
        self,
        conversation_id: str,
        anchors: list[tuple[int, str, str]],
    ) -> int: ...
    def get_canonical_turn_anchor_positions(
        self,
        conversation_id: str,
        window_size: int,
    ) -> dict[str, list[int]]: ...
    def search_tag_summaries_fts(self, query: str, limit: int = 20, conversation_id: str | None = None) -> list[tuple[str, float]]: ...
    def store_tag_summary_embedding(self, tag: str, conversation_id: str, embedding: list[float], *, operation_id: str | None = None, owner_worker_id: str | None = None, lifecycle_epoch: int | None = None) -> None: ...
    def load_tag_summary_embeddings(self, conversation_id: str | None = None) -> dict[str, list[float]]: ...
    def delete_tag_aliases_for_conversation(self, conversation_id: str) -> int: ...
    def save_conversation_alias(
        self,
        alias_id: str,
        target_id: str,
        *,
        epoch: int | None = None,
        on_committed=None,
    ) -> None: ...
    def resolve_conversation_alias(self, alias_id: str) -> str | None: ...
    def delete_conversation_alias(
        self,
        alias_id: str,
        *,
        on_committed=None,
    ) -> None: ...
    def list_conversation_aliases_by_target(self, target_id: str) -> list[str]: ...
    def resolve_request_audience(
        self,
        tenant_id: str,
        audience_conversation_id: str,
        owner_conversation_id: str,
    ) -> str: ...
    def has_any_alias(self, conversation_id: str) -> bool: ...
    def get_recent_canonical_turns(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list[CanonicalTurnRow]: ...


@runtime_checkable
class SpeakerHandleStore(Protocol):
    """Durable, audience-scoped speaker-handle assignments.

    Assignments are keyed ``(tenant_id, audience_conversation_id, actor_id)``
    and must be co-located with the canonical conversation lifecycle state so
    allocation, deletion, and merge share one transactional lock domain. Every
    method is tenant- and audience-scoped; an owner-only or actor-only surface
    is not contract-compatible. A backend that cannot provide the fenced
    transaction must not claim support: reads may degrade to empty, but
    allocation fails rather than minting unstable process-local handles.
    """

    def supports_speaker_handles(self) -> bool: ...
    def get_speaker_handles(
        self,
        tenant_id: str,
        audience_conversation_id: str,
        actor_ids: list[str],
    ) -> list[SpeakerHandleAssignment]: ...
    def allocate_speaker_handles(
        self,
        tenant_id: str,
        audience_conversation_id: str,
        candidates: list[SpeakerHandleCandidate],
        *,
        owner_conversation_id: str,
        expected_lifecycle_epoch: int,
    ) -> list[SpeakerHandleAssignment]: ...
    def delete_speaker_handles_for_audience(
        self,
        tenant_id: str,
        audience_conversation_id: str,
    ) -> int: ...


@runtime_checkable
class FactStore(Protocol):
    """Fact CRUD, querying, and supersession."""

    def store_facts(
        self,
        facts: list[Fact],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int: ...
    def query_facts(
        self, *, subject: str | None = None, verb: str | None = None,
        verbs: list[str] | None = None, object_contains: str | None = None,
        status: str | None = None, fact_type: str | None = None,
        tags: list[str] | None = None, limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]: ...
    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]: ...
    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]: ...
    def store_fact_embeddings(
        self,
        fact_id: str,
        conversation_id: str,
        model: str,
        embedding: list[float],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None: ...
    def load_fact_embeddings(
        self,
        conversation_id: str,
        model: str,
        *,
        expected_dim: int | None = None,
    ) -> dict[str, tuple[Fact, list[float]]]: ...
    def iter_facts_for_embedding_backfill(
        self,
        conversation_id: str,
        *,
        since: str | None = None,
        until: str | None = None,
        batch_size: int = 1000,
    ): ...
    def get_fact_embedding_index(
        self,
        conversation_id: str,
    ) -> dict[str, tuple[str, str]]: ...
    def replace_facts_for_segment(
        self,
        conversation_id: str,
        segment_ref: str,
        facts: list,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
        expected_lifecycle_epoch: int | None = None,
    ) -> tuple[int, int]: ...
    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list[Fact]: ...
    def set_fact_superseded(
        self,
        old_fact_id: str,
        new_fact_id: str,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None: ...
    def update_fact_fields(
        self,
        fact_id: str,
        verb: str,
        object: str,
        status: str,
        what: str,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> bool: ...
    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]: ...
    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]: ...
    def get_actionable_fact_tags(self, tags: list[str], conversation_id: str | None = None) -> set[str]: ...
    # Person cards live on the FactStore side, colocated with the facts they
    # are derived from. Enabling the card gate on a delegate that cannot back
    # these must fail startup rather than silently serve an unscoped card.
    def upsert_actor_profile_from_turn(
        self,
        conversation_id: str,
        actor_id: str,
        display_name: str = "",
        *,
        seen_at: str,
        expected_lifecycle_epoch: int | None = None,
    ) -> bool: ...
    def list_actor_facts(
        self, tenant_id: str, actor_id: str, *, limit: int = 60,
    ) -> list: ...
    def list_actor_turn_sources(
        self, tenant_id: str, actor_id: str, *, limit: int = 500,
    ) -> list: ...
    def list_actor_card_carryovers(
        self, tenant_id: str, actor_id: str,
    ) -> list: ...
    def get_actor_profile(self, tenant_id: str, actor_id: str): ...
    def mark_actor_card_dirty(
        self, tenant_id: str, actor_id: str, *, build_input_hash: str = "",
    ) -> bool: ...
    def replace_actor_card(
        self,
        tenant_id: str,
        actor_id: str,
        entries_with_sources: list,
        *,
        input_hash: str = "",
        expected_source_epochs: dict[str, int] | None = None,
        expected_build_marker: str | None = None,
    ) -> int: ...
    def record_actor_card_rebuild_status(
        self,
        tenant_id: str,
        actor_id: str,
        *,
        attempted_at: str,
        input_hash: str,
        source_count: int,
        raw_entry_count: int,
        accepted_entry_count: int,
        rejected_counts: dict[str, int],
        outcome: str,
        response_hash: str,
        written_count: int,
    ) -> None: ...
    def get_actor_card_rebuild_status(
        self, tenant_id: str, actor_id: str,
    ) -> dict | None: ...
    def list_due_actor_card_rebuilds(
        self, tenant_id: str, *, due_at: str, limit: int = 25,
    ) -> list[str]: ...
    def get_actor_card(
        self,
        tenant_id: str,
        actor_id: str,
        *,
        owner_conversation_id: str,
        audience_conversation_id: str,
        audience_channel_id: str = "",
    ): ...
    def invalidate_actor_cards_for_conversation(
        self, conversation_id: str, *, reason: str = "",
    ) -> int: ...


@runtime_checkable
class FactLinkStore(Protocol):
    """Fact link CRUD and traversal."""

    def store_fact_links(
        self,
        links: list[FactLink],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
        conversation_id: str | None = None,
    ) -> int: ...
    def get_fact_links(self, fact_id: str, direction: str = "both") -> list[FactLink]: ...
    def get_linked_facts(self, fact_ids: list[str], depth: int = 1) -> list[LinkedFact]: ...
    def delete_fact_links(self, fact_id: str) -> int: ...


@runtime_checkable
class StateStore(Protocol):
    """Engine state persistence."""

    def save_engine_state(self, state: EngineStateSnapshot) -> None: ...
    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None: ...
    def load_latest_engine_state(self) -> EngineStateSnapshot | None: ...
    def list_engine_state_fingerprints(self) -> dict[str, str]: ...
    def activate_conversation(self, conversation_id: str) -> int: ...
    def begin_conversation_deletion(self, conversation_id: str) -> int: ...
    def get_conversation_generation(self, conversation_id: str) -> int: ...
    def is_conversation_generation_current(self, conversation_id: str, generation: int) -> bool: ...
    def save_request_capture(self, capture: dict) -> None: ...
    def load_request_captures(
        self,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[dict]: ...


@runtime_checkable
class SearchStore(Protocol):
    """Full-text search and embedding storage."""

    def search_full_text(self, query: str, limit: int = 5, conversation_id: str | None = None) -> list[QuoteResult]: ...
    def search_canonical_turn_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
        channel: str = "",
        *,
        speaker_context: SpeakerRetrievalContext | None = None,
    ) -> list[QuoteResult]: ...
    def store_chunk_embeddings(
        self,
        segment_ref: str,
        chunks: list[ChunkEmbedding],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
        conversation_id: str | None = None,
    ) -> None: ...
    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]: ...
    def has_chunks_for_segment(self, segment_ref: str) -> bool: ...
    def store_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int,
        side: str,
        chunks: list[CanonicalTurnChunkEmbedding],
        canonical_turn_id: str | None = None,
    ) -> None: ...
    def get_all_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str | None = None,
        *,
        speaker_context: SpeakerRetrievalContext | None = None,
    ) -> list[CanonicalTurnChunkEmbedding]: ...
    def get_orphan_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str | None = None,
    ) -> list[CanonicalTurnChunkEmbedding]: ...
    def delete_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int | None = None,
        canonical_turn_id: str | None = None,
    ) -> int: ...
    def store_tool_output(self, ref: str, conversation_id: str, tool_name: str, command: str, turn: int, content: str, original_bytes: int) -> None: ...
    def search_tool_outputs(self, query: str, limit: int = 5, conversation_id: str | None = None) -> list: ...
    def link_turn_tool_output(self, conversation_id: str, turn_number: int, tool_output_ref: str) -> None: ...
    def get_tool_outputs_for_turn(self, conversation_id: str, turn_number: int) -> list[str]: ...
    def link_segment_tool_output(
        self,
        conversation_id: str,
        segment_ref: str,
        tool_output_ref: str,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None: ...
    def get_tool_outputs_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]: ...
    def get_tool_output_refs_for_turn(self, conversation_id: str, turn: int) -> list[str]: ...
    def get_tool_output_by_ref(self, conversation_id: str, ref: str) -> str | None: ...
    def store_chain_snapshot(self, ref: str, conversation_id: str, turn_number: int, chain_json: str, message_count: int, tool_output_refs: str = "") -> None: ...
    def get_chain_snapshot(self, conversation_id: str, ref: str) -> dict | None: ...
    def get_chain_snapshots_for_conversation(self, conversation_id: str, min_turn: int = 0) -> list[dict]: ...
    def get_chain_recovery_manifest(self, conversation_id: str, min_turn: int = 0) -> list[dict]: ...
    def get_tool_names_for_refs(self, refs: list[str]) -> list[str]: ...
    def get_tool_names_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]: ...
    def store_media_output(self, ref: str, conversation_id: str, media_type: str, width: int, height: int, original_bytes: int, compressed_bytes: int, file_path: str) -> None: ...
    def get_media_output(self, conversation_id: str, ref: str) -> dict | None: ...
