"""ContextStore abstract base class — tag-based storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from ..types import ChunkEmbedding, ConversationStats, DepthLevel, EngineStateSnapshot, Fact, QuoteResult, StoredSegment, StoredSummary, TagStats, TagSummary, WorkingSetEntry


class ContextStore(ABC):
    """Pluggable storage backend for compacted conversation segments."""

    @abstractmethod
    def store_segment(self, segment: StoredSegment) -> str:
        """Store a segment. Idempotent on ref (upsert). Returns ref."""

    @abstractmethod
    def get_segment(self, ref: str, *, conversation_id: str | None = None) -> StoredSegment | None:
        """Retrieve full segment by ref. None if not found."""

    @abstractmethod
    def get_summary(self, ref: str, *, conversation_id: str | None = None) -> StoredSummary | None:
        """Retrieve lightweight summary by ref. None if not found."""

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
    ) -> list[StoredSummary]:
        """Search summaries by keyword. Ordered by relevance."""

    @abstractmethod
    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]:
        """List all tags with statistics.

        If *conversation_id* is given, only return tags from segments
        belonging to that conversation.
        """

    @abstractmethod
    def get_conversation_stats(self) -> list[ConversationStats]:
        """Return aggregate statistics grouped by conversation_id, newest first."""

    @abstractmethod
    def get_tag_aliases(self) -> dict[str, str]:
        """Get all tag alias mappings."""

    @abstractmethod
    def set_tag_alias(self, alias: str, canonical: str) -> None:
        """Register a tag alias mapping."""

    @abstractmethod
    def delete_segment(self, ref: str) -> bool:
        """Delete a segment by ref. Returns True if deleted."""

    @abstractmethod
    def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int:
        """Remove old/excess segments. Returns count deleted."""

    @abstractmethod
    def save_tag_summary(self, tag_summary: TagSummary, conversation_id: str = "") -> None:
        """Store or update a tag summary. Upsert on (tag, conversation_id)."""

    @abstractmethod
    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None:
        """Retrieve a tag summary by tag name and conversation. None if not found."""

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

    @abstractmethod
    def get_segments_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 20,
        conversation_id: str | None = None,
    ) -> list[StoredSegment]:
        """Retrieve full segments (including full_text) matching tags by overlap."""

    def store_chunk_embeddings(self, segment_ref: str, chunks: list[ChunkEmbedding]) -> None:
        """Store embedding vectors for text chunks of a segment. Idempotent (replaces)."""

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        """Retrieve all stored chunk embeddings. Returns empty list if none."""
        return []

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        """Persist engine state (TurnTagIndex + watermark). Upsert by conversation_id."""

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        """Load persisted engine state for a conversation. None if not found."""

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        """Load the most recently saved engine state (any conversation). None if empty."""

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        """Return {trailing_fingerprint: conversation_id} for all persisted conversations.

        Used by SessionRegistry on restart to match inbound requests to
        existing conversations when conversation markers are unavailable.
        """

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
        """Persist turn message text. Upsert by (conversation_id, turn_number)."""

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
        """Store extracted facts. Returns count stored."""
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
        """Query facts by structured filters."""
        return []

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        """Return all distinct non-empty verbs from non-superseded facts."""
        return []

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        """Get all facts extracted from a given segment."""
        return []

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list[Fact]:
        """FTS search across fact fields. Returns non-superseded facts."""
        return []

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        """Mark old_fact_id as superseded by new_fact_id."""
        pass

    def update_fact_fields(
        self, fact_id: str, verb: str, object: str, status: str, what: str
    ) -> None:
        """Update structured fields on a fact (used after supersession merge)."""
        pass

    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]:
        """Return {tag: fact_count} for context hint annotations."""
        return {}

    def query_experience_facts_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        """Return completed experience facts within a session_date range.

        *start_date* and *end_date* are ``YYYY/MM/DD`` strings compared
        lexicographically against the ``session_date`` column.  Returns
        facts ordered by session_date ASC.
        """
        return []

    # ------------------------------------------------------------------
    # Tool Output Storage
    # ------------------------------------------------------------------

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete all segments, facts, engine state, turn messages, and tag summaries for a conversation. Returns segment count deleted."""
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
        """Store full tool output for FTS5 search via find_quote."""
        pass

    def search_tool_outputs(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list:
        """Search indexed tool outputs by FTS. Returns list of QuoteResult."""
        return []

    def save_request_capture(self, capture: dict) -> None:
        """Persist a single request capture summary. Prunes to *limit* newest."""
        pass

    def load_request_captures(self, limit: int = 50) -> list[dict]:
        """Load persisted request capture summaries, newest last."""
        return []
