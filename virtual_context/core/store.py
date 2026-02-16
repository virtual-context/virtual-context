"""ContextStore abstract base class — tag-based storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from ..types import SessionStats, StoredSegment, StoredSummary, TagStats, TagSummary


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
