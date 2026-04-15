"""ContextStore abstract base class — tag-based storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from ..types import ChunkEmbedding, ConversationStats, DepthLevel, EngineStateSnapshot, Fact, FactSignal, FullTextChunkEmbedding, FullTextRow, QuoteResult, StoredSegment, StoredSummary, TagStats, TagSummary, WorkingSetEntry


class ContextStore(ABC):
    """Pluggable storage backend for compacted conversation segments."""

    @abstractmethod
    def store_segment(self, segment: StoredSegment) -> str:
        """Upsert by ref. Returns ref."""

    def update_segment(self, segment: StoredSegment) -> None:
        """Update an existing segment in-place (same ref). Falls back to store_segment (upsert)."""
        self.store_segment(segment)

    @abstractmethod
    def get_segment(self, ref: str, *, conversation_id: str | None = None) -> StoredSegment | None: ...

    @abstractmethod
    def get_summary(self, ref: str, *, conversation_id: str | None = None) -> StoredSummary | None: ...

    @abstractmethod
    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        """Retrieve summaries matching tags by overlap count, newest first."""

    @abstractmethod
    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]: ...

    @abstractmethod
    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]:
        """List all tags with statistics.

        If *conversation_id* is given, only return tags from segments
        belonging to that conversation.
        """

    @abstractmethod
    def get_conversation_stats(self) -> list[ConversationStats]:
        """Return aggregate statistics grouped by conversation_id, newest first."""

    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[StoredSegment]:
        """Return full stored segments, newest first when supported."""
        return []

    @abstractmethod
    def get_tag_aliases(self, conversation_id: str | None = None) -> dict[str, str]: ...

    @abstractmethod
    def set_tag_alias(
        self,
        alias: str,
        canonical: str,
        conversation_id: str = "",
    ) -> None: ...

    @abstractmethod
    def delete_segment(self, ref: str) -> bool: ...

    @abstractmethod
    def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int: ...

    @abstractmethod
    def save_tag_summary(self, tag_summary: TagSummary, conversation_id: str = "") -> None:
        """Upsert on (tag, conversation_id)."""

    @abstractmethod
    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None: ...

    @abstractmethod
    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]:
        """Retrieve all tag summaries, ordered by tag name.

        If *conversation_id* is given, only return tag summaries whose
        source segments belong to that conversation.
        """

    @abstractmethod
    def search_full_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        """Search full_text across all segments.

        Returns QuoteResult objects with excerpts (~200 chars context around match).
        """

    def search_canonical_full_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        return []

    @abstractmethod
    def get_segments_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 20,
        conversation_id: str | None = None,
    ) -> list[StoredSegment]: ...

    def store_chunk_embeddings(self, segment_ref: str, chunks: list[ChunkEmbedding]) -> None:
        """Idempotent: replaces any existing chunks for this segment."""

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        return []

    def store_full_text_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int,
        side: str,
        chunks: list[FullTextChunkEmbedding],
    ) -> None:
        """Idempotent: replaces any existing full_text chunks for this turn side."""

    def get_all_full_text_chunk_embeddings(
        self,
        conversation_id: str | None = None,
    ) -> list[FullTextChunkEmbedding]:
        return []

    def delete_full_text_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int | None = None,
    ) -> int:
        return 0

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        """Upsert by conversation_id."""

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        return None

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        return None

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        """Return {trailing_fingerprint: conversation_id} for all persisted conversations.

        Used by SessionRegistry on restart to match inbound requests to
        existing conversations when conversation markers are unavailable.
        """

    # ------------------------------------------------------------------
    # Compaction dedup: turn numbers already covered by stored segments
    # ------------------------------------------------------------------

    def get_compacted_turn_numbers(self, conversation_id: str) -> set[int]:
        """Return the set of turn numbers already covered by stored tag summaries.

        Used by the compaction pipeline to skip segments whose turns have
        already been compacted, preventing redundant LLM calls when the
        compaction watermark drifts ahead of the in-memory history window.
        """
        tag_summaries = self.get_all_tag_summaries(conversation_id=conversation_id)
        covered: set[int] = set()
        for ts in tag_summaries:
            covered.update(ts.source_turn_numbers)
        return covered

    # ------------------------------------------------------------------
    # Turn messages (lightweight per-turn text for post-restart recall)
    # ------------------------------------------------------------------

    def save_turn_message(
        self,
        conversation_id: str,
        turn_number: int,
        user_content: str,
        assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
    ) -> None:
        """Upsert by (conversation_id, turn_number)."""

    def get_turn_messages(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, tuple[str, str, str | None, str | None]]:
        """Retrieve message text for specific turns.

        Returns {turn_number: (user_content, assistant_content, user_raw_content, assistant_raw_content)}.
        Missing turns are omitted from the result.
        """
        return {}

    def load_recent_turn_messages(
        self,
        conversation_id: str,
        limit: int = 100,
    ) -> list[tuple[int, str, str]]:
        """Load the most recent turn messages, ordered by turn_number ascending.

        Returns list of (turn_number, user_content, assistant_content).
        Used to rebuild conversation_history after restart.
        """
        return []

    def prune_turn_messages(self, conversation_id: str, keep_from_turn: int) -> int:
        """Delete persisted turn messages older than ``keep_from_turn``."""
        return 0

    # ------------------------------------------------------------------
    # Canonical archived full_text (permanent quote-search source of truth)
    # ------------------------------------------------------------------

    def save_full_text(
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
    ) -> None:
        """Upsert canonical archived text by (conversation_id, turn_number)."""

    def get_full_text_rows(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, FullTextRow]:
        return {}

    def get_all_full_text_rows(
        self,
        conversation_id: str,
    ) -> list[FullTextRow]:
        return []

    def delete_full_text_rows(
        self,
        conversation_id: str,
        turn_number: int | None = None,
    ) -> int:
        return 0

    # ------------------------------------------------------------------
    # Conversation lifecycle fencing
    # ------------------------------------------------------------------

    def activate_conversation(self, conversation_id: str) -> int:
        """Mark a conversation as live and return its current generation."""
        return 0

    def begin_conversation_deletion(self, conversation_id: str) -> int:
        """Fence future writes for a conversation by advancing its generation."""
        return self.activate_conversation(conversation_id)

    def get_conversation_generation(self, conversation_id: str) -> int:
        """Return the current durable generation for a conversation."""
        return 0

    def is_conversation_generation_current(
        self,
        conversation_id: str,
        generation: int,
    ) -> bool:
        """Whether ``generation`` is the live write generation for ``conversation_id``."""
        return generation == self.get_conversation_generation(conversation_id)

    # ------------------------------------------------------------------
    # Tag summary search (used by RRF retrieval scoring)
    # ------------------------------------------------------------------

    def search_tag_summaries_fts(
        self, query: str, limit: int = 20, conversation_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """FTS search on tag summary text. Returns [(tag, bm25_score)]."""
        return []

    def store_tag_summary_embedding(
        self, tag: str, conversation_id: str, embedding: list[float],
    ) -> None:
        """Store embedding vector for a tag summary."""
        pass

    def load_tag_summary_embeddings(
        self, conversation_id: str | None = None,
    ) -> dict[str, list[float]]:
        """Load all tag summary embeddings. Returns {tag: embedding_vector}."""
        return {}

    # ------------------------------------------------------------------
    # Cross-cutting queries (used by consolidator, tool loop, etc.)
    # ------------------------------------------------------------------

    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]:
        """Return snippet descriptions for tags that have no tag_summary entry.

        Each dict has keys: ``tag`` (str), ``snippet`` (str -- first ~100
        chars of the segment summary for one segment carrying that tag).

        Used by the tag consolidator to provide descriptions for orphan
        tags so the LLM can make informed consolidation decisions.

        Backends that do not support this may return an empty list.
        """
        return []

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        """Return facts that were superseded *by* the given fact IDs.

        Each dict has keys: ``superseded_by`` (str — the ID of the newer
        fact), ``subject``, ``verb``, ``object`` (all str).

        This is the reverse lookup: given a set of current (non-superseded)
        fact IDs, find the older facts they replaced.

        Backends that do not support facts may return an empty list.
        """
        return []

    # ------------------------------------------------------------------
    # D1: Fact Extraction
    # ------------------------------------------------------------------

    def store_facts(self, facts: list[Fact]) -> int:
        return 0

    def query_facts(
        self,
        *,
        subject: str | None = None,
        verb: str | None = None,
        verbs: list[str] | None = None,
        object_contains: str | None = None,
        status: str | None = None,
        fact_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        return []

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        return []

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        return []

    def replace_facts_for_segment(self, conversation_id: str, segment_ref: str, facts: list) -> tuple[int, int]:
        """Atomically replace all facts for a segment. Returns (deleted, inserted)."""
        return 0, self.store_facts(facts)

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list[Fact]:
        """FTS search across fact fields. Returns non-superseded facts."""
        return []

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        pass

    def update_fact_fields(
        self, fact_id: str, verb: str, object: str, status: str, what: str
    ) -> None:
        pass

    def get_actionable_fact_tags(
        self, tags: list[str], conversation_id: str | None = None,
    ) -> set[str]:
        """Return subset of tags that have non-superseded active/completed/personal facts."""
        return set()

    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]:
        return {}

    def query_experience_facts_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        """Return facts within a when_date range.

        *start_date* and *end_date* are ``YYYY-MM-DD`` ISO strings compared
        lexicographically against the ``when_date`` column.  Returns
        facts ordered by when_date ASC.
        """
        return []

    # ------------------------------------------------------------------
    # Turn / Segment ↔ Tool Output linkage (join tables)
    # ------------------------------------------------------------------

    def link_turn_tool_output(self, conversation_id: str, turn_number: int, tool_output_ref: str) -> None:
        """Link a tool output ref to a specific turn."""
        pass  # default no-op for backwards compatibility

    def get_tool_outputs_for_turn(self, conversation_id: str, turn_number: int) -> list[str]:
        """Return tool_output refs linked to a turn."""
        return []

    def link_segment_tool_output(self, conversation_id: str, segment_ref: str, tool_output_ref: str) -> None:
        """Link a tool output ref to a segment."""
        pass

    def get_tool_outputs_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        """Return tool_output refs linked to a segment."""
        return []

    def get_tool_output_refs_for_turn(self, conversation_id: str, turn: int) -> list[str]:
        """Return refs from the tool_outputs table for a given conversation + turn.

        Unlike ``get_tool_outputs_for_turn`` (which reads the join table),
        this queries the ``tool_outputs`` table directly by its ``turn`` column.
        Used during ingestion to discover intercepted tool outputs that should
        be linked to a canonical turn.
        """
        return []

    # ------------------------------------------------------------------
    # Media Output Storage
    # ------------------------------------------------------------------

    def store_media_output(
        self,
        ref: str,
        conversation_id: str,
        media_type: str,
        width: int,
        height: int,
        original_bytes: int,
        compressed_bytes: int,
        file_path: str,
    ) -> None:
        """Store metadata for a compressed media output. Default no-op."""
        pass

    def get_media_output(self, conversation_id: str, ref: str) -> dict | None:
        """Look up media output metadata by conversation_id and ref.

        Returns dict with keys {ref, conversation_id, media_type, width, height,
        original_bytes, compressed_bytes, file_path} or None if not found.
        Default returns None.
        """
        return None

    # ------------------------------------------------------------------
    # Tool Output Storage
    # ------------------------------------------------------------------

    def delete_conversation(self, conversation_id: str) -> int:
        return 0

    def delete_tag_aliases_for_conversation(self, conversation_id: str) -> int:
        """Delete aliases owned by ``conversation_id``.

        Backends may keep legacy/global aliases under the empty conversation id.
        """
        return 0

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
        pass

    def search_tool_outputs(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list:
        return []

    def get_tool_output_by_ref(self, conversation_id: str, ref: str) -> str | None:
        """Look up a stored tool output by conversation_id and ref.

        Returns the full stored content string, or None if not found.
        """
        return None

    # ------------------------------------------------------------------
    # Chain Snapshots (turn chain collapse)
    # ------------------------------------------------------------------

    def store_chain_snapshot(
        self,
        ref: str,
        conversation_id: str,
        turn_number: int,
        chain_json: str,
        message_count: int,
        tool_output_refs: str = "",
    ) -> None:
        """Upsert a chain snapshot by ref."""
        pass

    def get_chain_snapshot(self, conversation_id: str, ref: str) -> dict | None:
        """Retrieve a chain snapshot by conversation_id and ref.

        Returns {ref, conversation_id, turn_number, chain_json,
        message_count, tool_output_refs} or None.
        """
        return None

    def get_chain_snapshots_for_conversation(
        self,
        conversation_id: str,
        min_turn: int = 0,
    ) -> list[dict]:
        """Return metadata for chain snapshots where turn_number >= min_turn.
        Returns list of {ref, turn_number, tool_output_refs, message_count}. No chain_json."""
        return []

    def get_chain_recovery_manifest(
        self,
        conversation_id: str,
        min_turn: int = 0,
    ) -> list[dict]:
        """Return recovery metadata for collapsed chain stubs.

        Default implementation falls back to snapshot metadata plus one tool-name
        lookup per snapshot. Storage backends can override this with a single
        optimized query.
        """
        manifest: list[dict] = []
        for snap in self.get_chain_snapshots_for_conversation(
            conversation_id,
            min_turn=min_turn,
        ):
            raw_refs = [
                ref.strip()
                for ref in str(snap.get("tool_output_refs", "")).split(",")
                if ref.strip()
            ]
            tool_names = self.get_tool_names_for_refs(raw_refs) if raw_refs else []
            manifest.append({
                "ref": snap.get("ref", ""),
                "turn_number": snap.get("turn_number", -1),
                "tool_output_refs": snap.get("tool_output_refs", ""),
                "message_count": snap.get("message_count", 0),
                "tool_names": ", ".join(tool_names) if tool_names else "",
            })
        return manifest

    def get_tool_names_for_refs(self, refs: list[str]) -> list[str]:
        return []

    def get_tool_names_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        """Return distinct tool names linked to a segment via segment_tool_outputs.

        Performs a JOIN between segment_tool_outputs and tool_outputs to get
        distinct tool_name values.
        """
        return []

    def save_tool_call(self, call: dict) -> None:
        """Persist a tool call record."""
        pass

    def load_tool_calls(self, conversation_id: str, limit: int = 50) -> list[dict]:
        """Load recent tool call records for a conversation."""
        return []

    def load_tool_call(self, call_id: int) -> dict | None:
        """Load a single tool call by ID."""
        return None

    def save_request_context(self, context: dict) -> int:
        """Persist retrieval/assembly context for a request.

        Returns the durable per-conversation request sequence assigned to this
        context. Implementations may honor an explicit ``request_turn`` value
        when provided for migrations/backfills, but live request handling
        should rely on the returned value rather than supplying its own.
        """
        return int(context.get("request_turn", 0) or 0)

    def load_request_contexts(self, conversation_id: str, limit: int = 50) -> list[dict]:
        """Load recent request contexts for a conversation."""
        return []

    def save_request_capture(self, capture: dict) -> None:
        pass

    def load_request_captures(
        self,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[dict]:
        return []
