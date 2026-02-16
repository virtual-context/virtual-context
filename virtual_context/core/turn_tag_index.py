"""TurnTagIndex: live index of per-turn tag metadata."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..types import TurnTagEntry


class TurnTagIndex:
    """Live index maintained as conversation progresses.

    Updated every round trip by the engine. Read by segmenter and retriever.
    Purely in-memory — not persisted. Rebuilt each session.
    """

    def __init__(self) -> None:
        self.entries: list[TurnTagEntry] = []

    def append(self, entry: TurnTagEntry) -> None:
        self.entries.append(entry)

    def get_active_tags(self, lookback: int = 4) -> set[str]:
        """Tags present in the last N turns."""
        recent = self.entries[-lookback:] if len(self.entries) >= lookback else self.entries
        tags: set[str] = set()
        for entry in recent:
            tags.update(entry.tags)
        return tags

    def get_tags_for_turn(self, turn_number: int) -> TurnTagEntry | None:
        """Look up pre-computed tags for a specific turn."""
        for entry in self.entries:
            if entry.turn_number == turn_number:
                return entry
        return None

    def get_tag_velocity(self, tag: str, window_hours: float = 72) -> float:
        """Compute velocity: entries per hour for a tag over a time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        recent = [e for e in self.entries if e.timestamp >= cutoff and tag in e.tags]
        if not recent:
            return 0.0
        time_span = (datetime.now(timezone.utc) - recent[0].timestamp).total_seconds() / 3600
        return len(recent) / max(time_span, 1.0)

    def latest_meaningful_tags(self) -> TurnTagEntry | None:
        """Return the most recent entry with real tags (not ``_general`` only).

        Walks backwards through entries to find the last turn whose tags
        contain at least one non-``_general`` tag.  Used to propagate topic
        continuity to ultra-short messages during history ingestion.
        """
        for entry in reversed(self.entries):
            if any(t != "_general" for t in entry.tags):
                return entry
        return None

    def compute_cover_set(self, exclude_tags: set[str] | None = None) -> list[str]:
        """Greedy set cover: find minimum tags to touch every indexed turn.

        Returns cover tags ordered by coverage (most-covering first).
        Excludes ``_general`` by default since it carries no semantic value.
        """
        if not self.entries:
            return []

        exclude = exclude_tags if exclude_tags is not None else {"_general"}

        # Build tag -> turn numbers mapping
        tag_to_turns: dict[str, set[int]] = {}
        all_turns: set[int] = set()
        for entry in self.entries:
            all_turns.add(entry.turn_number)
            for tag in entry.tags:
                if tag not in exclude:
                    tag_to_turns.setdefault(tag, set()).add(entry.turn_number)

        if not tag_to_turns:
            return []

        uncovered = set(all_turns)
        cover: list[str] = []

        while uncovered:
            best_tag = max(
                tag_to_turns,
                key=lambda t: len(tag_to_turns[t] & uncovered),
            )
            covered_by_best = tag_to_turns[best_tag] & uncovered
            if not covered_by_best:
                break  # remaining turns only have excluded tags
            uncovered -= covered_by_best
            cover.append(best_tag)

        return cover
