"""ContextStore abstract base class — tag-based storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from ..types import ChunkEmbedding, DepthLevel, EngineStateSnapshot, Fact, QuoteResult, SessionStats, StoredSegment, StoredSummary, TagStats, TagSummary, WorkingSetEntry


class ContextStore(ABC):
    """Pluggable storage backend for compacted conversation segments."""

    @abstractmethod
    def store_segment(self, segment: StoredSegment) -> str:
        """Store a segment. Idempotent on ref (upsert). Returns ref."""

    @abstractmethod
    def get_segment(self, ref: str) -> StoredSegment | None:
        """Retrieve full segment by ref. None if not found."""

    @abstractmethod
    def get_summary(self, ref: str) -> StoredSummary | None:
        """Retrieve lightweight summary by ref. None if not found."""

    @abstractmethod
    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> list[StoredSummary]:
        """Retrieve summaries matching tags by overlap count, newest first."""

    @abstractmethod
    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[StoredSummary]:
        """Search summaries by keyword. Ordered by relevance."""

    @abstractmethod
    def get_all_tags(self) -> list[TagStats]:
        """List all tags with statistics."""

    @abstractmethod
    def get_session_stats(self) -> list[SessionStats]:
        """Return aggregate statistics grouped by session_id, newest first."""

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
    def save_tag_summary(self, tag_summary: TagSummary) -> None:
        """Store or update a tag summary. Upsert on tag name."""

    @abstractmethod
    def get_tag_summary(self, tag: str) -> TagSummary | None:
        """Retrieve a tag summary by tag name. None if not found."""

    @abstractmethod
    def get_all_tag_summaries(self) -> list[TagSummary]:
        """Retrieve all tag summaries, ordered by tag name."""

    @abstractmethod
    def search_full_text(
        self,
        query: str,
        limit: int = 5,
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
    ) -> list[StoredSegment]:
        """Retrieve full segments (including full_text) matching tags by overlap."""

    def store_chunk_embeddings(self, segment_ref: str, chunks: list[ChunkEmbedding]) -> None:
        """Store embedding vectors for text chunks of a segment. Idempotent (replaces)."""

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        """Retrieve all stored chunk embeddings. Returns empty list if none."""
        return []

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        """Persist engine state (TurnTagIndex + watermark). Upsert by session_id."""

    def load_engine_state(self, session_id: str) -> EngineStateSnapshot | None:
        """Load persisted engine state for a session. None if not found."""

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        """Load the most recently saved engine state (any session). None if empty."""

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        """Return {trailing_fingerprint: session_id} for all persisted sessions.

        Used by SessionRegistry on restart to match inbound requests to
        existing sessions when session markers are unavailable.
        """

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
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        """Query facts by structured filters."""
        return []

    def get_unique_fact_verbs(self) -> list[str]:
        """Return all distinct non-empty verbs from non-superseded facts."""
        return []

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        """Get all facts extracted from a given segment."""
        return []

    def get_fact_count_by_tags(self) -> dict[str, int]:
        """Return {tag: fact_count} for context hint annotations."""
        return {}
