"""ContextStore abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from ..types import DomainStats, StoredSegment, StoredSummary


class ContextStore(ABC):
    """Pluggable storage backend for compacted conversation segments."""

    @abstractmethod
    async def store_segment(self, segment: StoredSegment) -> str:
        """Store a segment. Idempotent on ref (upsert). Returns ref."""

    @abstractmethod
    async def get_segment(self, ref: str) -> StoredSegment | None:
        """Retrieve full segment by ref. None if not found."""

    @abstractmethod
    async def get_summary(self, ref: str) -> StoredSummary | None:
        """Retrieve lightweight summary by ref. None if not found."""

    @abstractmethod
    async def get_summaries(
        self,
        domain: str | None = None,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> list[StoredSummary]:
        """Retrieve summaries, ordered by created_at DESC (newest first)."""

    @abstractmethod
    async def search(
        self,
        query: str,
        domains: list[str] | None = None,
        limit: int = 5,
    ) -> list[StoredSummary]:
        """Search summaries by keyword. Ordered by relevance."""

    @abstractmethod
    async def list_domains(self) -> list[DomainStats]:
        """List all domains with statistics."""

    @abstractmethod
    async def delete_segment(self, ref: str) -> bool:
        """Delete a segment by ref. Returns True if deleted."""

    @abstractmethod
    async def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int:
        """Remove old/excess segments. Returns count deleted."""
