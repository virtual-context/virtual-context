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
    FactSignal,
    CanonicalTurnChunkEmbedding,
    CanonicalTurnRow,
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

    def store_segment(
        self,
        segment: StoredSegment,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> str:
        return self._segments.store_segment(
            segment,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        )

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

    def save_tag_summary(
        self,
        tag_summary: TagSummary,
        conversation_id: str = "",
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None:
        return self._segments.save_tag_summary(
            tag_summary,
            conversation_id=conversation_id,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        )

    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None:
        return self._segments.get_tag_summary(tag, conversation_id=conversation_id)

    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]:
        return self._segments.get_all_tag_summaries(conversation_id=conversation_id)

    def search_tag_summaries_fts(self, query: str, limit: int = 20, conversation_id: str | None = None) -> list[tuple[str, float]]:
        return self._segments.search_tag_summaries_fts(query, limit=limit, conversation_id=conversation_id)

    def store_tag_summary_embedding(self, tag: str, conversation_id: str, embedding: list[float], *, operation_id: str | None = None, owner_worker_id: str | None = None, lifecycle_epoch: int | None = None) -> None:
        return self._segments.store_tag_summary_embedding(tag, conversation_id, embedding, operation_id=operation_id, owner_worker_id=owner_worker_id, lifecycle_epoch=lifecycle_epoch)

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
    ) -> None:
        return self._segments.save_canonical_turn(
            conversation_id,
            turn_number,
            user_content,
            assistant_content,
            user_raw_content=user_raw_content,
            assistant_raw_content=assistant_raw_content,
            primary_tag=primary_tag,
            tags=tags,
            session_date=session_date,
            sender=sender,
            fact_signals=fact_signals,
            code_refs=code_refs,
            created_at=created_at,
            updated_at=updated_at,
            canonical_turn_id=canonical_turn_id,
            sort_key=sort_key,
            turn_hash=turn_hash,
            hash_version=hash_version,
            normalized_user_text=normalized_user_text,
            normalized_assistant_text=normalized_assistant_text,
            tagged_at=tagged_at,
            compacted_at=compacted_at,
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
            source_batch_id=source_batch_id,
            turn_group_number=turn_group_number,
        )

    def recompute_canonical_turn_groups(
        self,
        conversation_id: str,
    ) -> int:
        recompute = getattr(self._segments, "recompute_canonical_turn_groups", None)
        if callable(recompute):
            return int(recompute(conversation_id) or 0)
        return 0

    def get_canonical_turn_rows(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, CanonicalTurnRow]:
        return self._segments.get_canonical_turn_rows(conversation_id, turn_numbers)

    def get_all_canonical_turns(
        self,
        conversation_id: str,
    ) -> list[CanonicalTurnRow]:
        return self._segments.get_all_canonical_turns(conversation_id)

    def get_uncompacted_canonical_turns(
        self,
        conversation_id: str,
        *,
        protected_recent_turns: int = 0,
    ) -> list[CanonicalTurnRow]:
        return self._segments.get_uncompacted_canonical_turns(
            conversation_id,
            protected_recent_turns=protected_recent_turns,
        )

    def mark_canonical_turns_tagged(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        tagged_at: str | None = None,
    ) -> int:
        return self._segments.mark_canonical_turns_tagged(
            conversation_id,
            canonical_turn_ids,
            tagged_at=tagged_at,
        )

    def mark_canonical_turns_compacted(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        compacted_at: str | None = None,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int:
        return self._segments.mark_canonical_turns_compacted(
            conversation_id,
            canonical_turn_ids,
            compacted_at=compacted_at,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        )

    def delete_canonical_turns(
        self,
        conversation_id: str,
        turn_number: int | None = None,
    ) -> int:
        return int(
            self._segments.delete_canonical_turns(
                conversation_id,
                turn_number=turn_number,
            ) or 0
        )

    def delete_canonical_turns_by_batch_id(
        self,
        *,
        conversation_id: str,
        batch_id: str,
    ) -> int:
        deleter = getattr(self._segments, "delete_canonical_turns_by_batch_id", None)
        if not callable(deleter):
            return 0
        return int(
            deleter(
                conversation_id=conversation_id,
                batch_id=batch_id,
            ) or 0
        )

    def save_conversation_alias(self, alias_id: str, target_id: str) -> None:
        self._segments.save_conversation_alias(alias_id, target_id)

    def resolve_conversation_alias(self, alias_id: str) -> str | None:
        return self._segments.resolve_conversation_alias(alias_id)

    def delete_conversation_alias(self, alias_id: str) -> None:
        self._segments.delete_conversation_alias(alias_id)

    # ------------------------------------------------------------------
    # FactStore
    # ------------------------------------------------------------------

    def store_facts(
        self,
        facts: list[Fact],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int:
        return self._facts.store_facts(
            facts,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        )

    def query_facts(self, **kwargs) -> list[Fact]:
        return self._facts.query_facts(**kwargs)

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        return self._facts.get_unique_fact_verbs(conversation_id=conversation_id)

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        return self._facts.get_facts_by_segment(segment_ref)

    def replace_facts_for_segment(
        self,
        conversation_id: str,
        segment_ref: str,
        facts: list,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> tuple[int, int]:
        return self._facts.replace_facts_for_segment(
            conversation_id,
            segment_ref,
            facts,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        )

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

    def query_experience_facts_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        return self._facts.query_experience_facts_by_date(
            start_date=start_date, end_date=end_date, limit=limit, conversation_id=conversation_id,
        )

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

    def search_canonical_turn_text(self, *args, **kwargs) -> list[QuoteResult]:
        return self._search.search_canonical_turn_text(*args, **kwargs)

    def store_chunk_embeddings(
        self, segment_ref: str, chunks: list[ChunkEmbedding],
    ) -> None:
        return self._search.store_chunk_embeddings(segment_ref, chunks)

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        return self._search.get_all_chunk_embeddings()

    def store_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int,
        side: str,
        chunks: list[CanonicalTurnChunkEmbedding],
        canonical_turn_id: str | None = None,
    ) -> None:
        return self._search.store_canonical_turn_chunk_embeddings(
            conversation_id,
            turn_number,
            side,
            chunks,
            canonical_turn_id=canonical_turn_id,
        )

    def get_all_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str | None = None,
    ) -> list[CanonicalTurnChunkEmbedding]:
        return self._search.get_all_canonical_turn_chunk_embeddings(
            conversation_id=conversation_id,
        )

    def delete_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int | None = None,
        canonical_turn_id: str | None = None,
    ) -> int:
        return int(
            self._search.delete_canonical_turn_chunk_embeddings(
                conversation_id,
                turn_number=turn_number,
                canonical_turn_id=canonical_turn_id,
            ) or 0
        )

    def replace_canonical_turn_anchors(
        self,
        conversation_id: str,
        anchors: list[tuple[int, str, str]],
    ) -> int:
        replacer = getattr(self._segments, "replace_canonical_turn_anchors", None)
        if not callable(replacer):
            return 0
        return int(replacer(conversation_id, anchors) or 0)

    def get_canonical_turn_anchor_positions(
        self,
        conversation_id: str,
        window_size: int,
    ) -> dict[str, list[int]]:
        loader = getattr(self._segments, "get_canonical_turn_anchor_positions", None)
        if not callable(loader):
            return {}
        return loader(conversation_id, window_size)

    def save_ingest_batch(self, batch: dict) -> str:
        saver = getattr(self._segments, "save_ingest_batch", None)
        if callable(saver):
            return str(saver(batch) or "")
        return str(batch.get("batch_id", "") or "")

    def conversation_reconcile(self, conversation_id: str):
        locker = getattr(self._segments, "conversation_reconcile", None)
        if callable(locker):
            return locker(conversation_id)
        from contextlib import nullcontext
        return nullcontext()

    # ------------------------------------------------------------------
    # Conversation row lifecycle (progress-bar redesign)
    # ------------------------------------------------------------------
    # Delegate lifecycle-epoch methods to the segments store. These are used
    # by IngestReconciler's epoch-safe ingest_prepared_turns to fence against
    # delete+resurrect races.
    def upsert_conversation(self, *, tenant_id: str, conversation_id: str) -> None:
        fn = getattr(self._segments, "upsert_conversation", None)
        if callable(fn):
            fn(tenant_id=tenant_id, conversation_id=conversation_id)

    def get_lifecycle_epoch(self, conversation_id: str) -> int:
        fn = getattr(self._segments, "get_lifecycle_epoch", None)
        if callable(fn):
            return int(fn(conversation_id))
        raise KeyError(conversation_id)

    def get_conversation_phase(self, conversation_id: str) -> str:
        fn = getattr(self._segments, "get_conversation_phase", None)
        if callable(fn):
            return str(fn(conversation_id))
        raise KeyError(conversation_id)

    def mark_conversation_deleted(self, conversation_id: str) -> None:
        fn = getattr(self._segments, "mark_conversation_deleted", None)
        if callable(fn):
            fn(conversation_id)

    def increment_lifecycle_epoch_on_resurrect(self, conversation_id: str) -> int:
        fn = getattr(self._segments, "increment_lifecycle_epoch_on_resurrect", None)
        if callable(fn):
            return int(fn(conversation_id))
        raise KeyError(conversation_id)

    # ------------------------------------------------------------------
    # Progress snapshot + request metadata (progress-bar redesign)
    # ------------------------------------------------------------------
    # Delegate the read-side snapshot and the four epoch-guarded write
    # helpers introduced in A20 to the segments store. These are invoked
    # directly from the engine-owned ``handle_prepare_payload`` flow so
    # ``engine._store.<method>(...)`` must resolve without traversing the
    # nested composite/inner-segment chain.
    def read_progress_snapshot(self, conversation_id: str):
        fn = getattr(self._segments, "read_progress_snapshot", None)
        if callable(fn):
            return fn(conversation_id)
        raise NotImplementedError

    def update_request_metadata(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        last_raw_payload_entries: int,
        last_ingestible_payload_entries: int,
    ) -> bool:
        fn = getattr(self._segments, "update_request_metadata", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                last_raw_payload_entries=last_raw_payload_entries,
                last_ingestible_payload_entries=last_ingestible_payload_entries,
            ))
        raise NotImplementedError

    def widen_pending_raw_payload_entries(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        value: int,
    ) -> bool:
        fn = getattr(self._segments, "widen_pending_raw_payload_entries", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                value=value,
            ))
        raise NotImplementedError

    def set_phase(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        phase: str,
    ) -> bool:
        fn = getattr(self._segments, "set_phase", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                phase=phase,
            ))
        raise NotImplementedError

    def set_phase_and_drain_pending_raw(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        new_phase: str,
    ) -> int | None:
        fn = getattr(self._segments, "set_phase_and_drain_pending_raw", None)
        if callable(fn):
            return fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                new_phase=new_phase,
            )
        raise NotImplementedError

    def drain_compaction_exit(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> str | None:
        fn = getattr(self._segments, "drain_compaction_exit", None)
        if callable(fn):
            return fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
            )
        raise NotImplementedError

    def start_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_count: int,
        phase_name: str,
    ) -> str:
        fn = getattr(self._segments, "start_compaction_operation", None)
        if callable(fn):
            return fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
                phase_count=phase_count,
                phase_name=phase_name,
            )
        raise NotImplementedError

    def advance_compaction_phase(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_index: int,
        phase_name: str,
    ) -> bool:
        fn = getattr(self._segments, "advance_compaction_phase", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
                phase_index=phase_index,
                phase_name=phase_name,
            ))
        raise NotImplementedError

    def complete_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        fn = getattr(self._segments, "complete_compaction_operation", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
            ))
        raise NotImplementedError

    def fail_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        error_message: str,
    ) -> bool:
        fn = getattr(self._segments, "fail_compaction_operation", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
                error_message=error_message,
            ))
        raise NotImplementedError

    def upsert_ingestion_episode(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        raw_payload_entries: int,
    ) -> None:
        fn = getattr(self._segments, "upsert_ingestion_episode", None)
        if callable(fn):
            return fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
                raw_payload_entries=raw_payload_entries,
            )
        raise NotImplementedError

    def claim_ingestion_lease(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        lease_ttl_s: float,
    ) -> bool:
        fn = getattr(self._segments, "claim_ingestion_lease", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
                lease_ttl_s=lease_ttl_s,
            ))
        raise NotImplementedError

    def refresh_ingestion_heartbeat(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        fn = getattr(self._segments, "refresh_ingestion_heartbeat", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
            ))
        raise NotImplementedError

    def complete_ingestion_episode(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        fn = getattr(self._segments, "complete_ingestion_episode", None)
        if callable(fn):
            return bool(fn(
                conversation_id=conversation_id,
                lifecycle_epoch=lifecycle_epoch,
                worker_id=worker_id,
            ))
        raise NotImplementedError

    def iter_untagged_canonical_rows(
        self,
        *,
        conversation_id: str,
        expected_lifecycle_epoch: int,
        batch_size: int = 32,
    ) -> list[CanonicalTurnRow]:
        fn = getattr(self._segments, "iter_untagged_canonical_rows", None)
        if callable(fn):
            return list(fn(
                conversation_id=conversation_id,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
                batch_size=batch_size,
            ) or [])
        raise NotImplementedError

    def mark_canonical_row_tagged(
        self,
        *,
        canonical_turn_id: str,
        conversation_id: str,
        expected_lifecycle_epoch: int,
    ) -> bool:
        fn = getattr(self._segments, "mark_canonical_row_tagged", None)
        if callable(fn):
            return bool(fn(
                canonical_turn_id=canonical_turn_id,
                conversation_id=conversation_id,
                expected_lifecycle_epoch=expected_lifecycle_epoch,
            ))
        raise NotImplementedError

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

    def save_request_context(self, context: dict) -> int:
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
