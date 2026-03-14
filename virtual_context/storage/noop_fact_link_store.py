"""NoopFactLinkStore — silent no-op when graph_links is disabled."""

from __future__ import annotations

from ..types import FactLink, LinkedFact


class NoopFactLinkStore:
    """All methods return empty results. Used when graph_links config is false."""

    def store_fact_links(self, links: list[FactLink]) -> int:
        return 0

    def get_fact_links(self, fact_id: str, direction: str = "both") -> list[FactLink]:
        return []

    def get_linked_facts(self, fact_ids: list[str], depth: int = 1) -> list[LinkedFact]:
        return []

    def delete_fact_links(self, fact_id: str) -> int:
        return 0
