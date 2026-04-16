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
        self._by_logical_turn: dict[int, TurnTagEntry] = {}
        self._by_canonical_turn: dict[str, TurnTagEntry] = {}
        self._by_hash: dict[str, TurnTagEntry] = {}
        self._all_tags: set[str] = set()

    def append(self, entry: TurnTagEntry) -> None:
        if entry.turn_number in self._by_logical_turn:
            import logging
            logging.getLogger(__name__).warning(
                "OVERWRITE_BLOCKED turn=%d existing_tags=%s new_tags=%s — keeping original",
                entry.turn_number,
                self._by_logical_turn[entry.turn_number].tags,
                entry.tags,
            )
            return  # silently reject duplicate turn_number
        if entry.canonical_turn_id and entry.canonical_turn_id in self._by_canonical_turn:
            import logging
            logging.getLogger(__name__).warning(
                "OVERWRITE_BLOCKED canonical_turn_id=%s existing_tags=%s new_tags=%s — keeping original",
                entry.canonical_turn_id,
                self._by_canonical_turn[entry.canonical_turn_id].tags,
                entry.tags,
            )
            return
        self.entries.append(entry)
        self._by_logical_turn[entry.turn_number] = entry
        if entry.canonical_turn_id:
            self._by_canonical_turn[entry.canonical_turn_id] = entry
        if entry.message_hash:
            self._by_hash[entry.message_hash] = entry
        self._all_tags.update(entry.tags)

    def get_active_tags(self, lookback: int = 4) -> set[str]:
        recent = self.entries[-lookback:] if len(self.entries) >= lookback else self.entries
        tags: set[str] = set()
        for entry in recent:
            tags.update(entry.tags)
        tags -= self._NON_INHERITABLE_TAGS  # exclude _general, _stub from retrieval queries
        return tags

    def get_tags_for_logical_turn(self, turn_number: int) -> TurnTagEntry | None:
        return self._by_logical_turn.get(turn_number)

    def get_tags_for_canonical_turn(self, canonical_turn_id: str) -> TurnTagEntry | None:
        return self._by_canonical_turn.get(canonical_turn_id)

    def bind_canonical_turn_id(
        self,
        turn_number: int,
        canonical_turn_id: str,
    ) -> TurnTagEntry | None:
        if not canonical_turn_id:
            return None
        existing = self._by_canonical_turn.get(canonical_turn_id)
        if existing is not None:
            return existing
        entry = self._by_logical_turn.get(turn_number)
        if entry is None:
            return None
        entry.canonical_turn_id = canonical_turn_id
        self._by_canonical_turn[canonical_turn_id] = entry
        return entry

    def get_entry_by_hash(self, message_hash: str) -> TurnTagEntry | None:
        return self._by_hash.get(message_hash)

    def all_tags(self) -> set[str]:
        """Return every tag currently present in the index."""
        return set(self._all_tags)

    def get_tag_velocity(self, tag: str, window_hours: float = 72) -> float:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        recent = [e for e in self.entries if e.timestamp >= cutoff and tag in e.tags]
        if not recent:
            return 0.0
        time_span = (datetime.now(timezone.utc) - recent[0].timestamp).total_seconds() / 3600
        return len(recent) / max(time_span, 1.0)

    _NON_INHERITABLE_TAGS = {"_general", "_stub"}

    def latest_meaningful_tags(self) -> TurnTagEntry | None:
        """Return the most recent entry with real tags (not ``_general``/``_stub`` only).

        Walks backwards through entries to find the last turn whose tags
        contain at least one substantive tag.  Used to propagate topic
        continuity to ultra-short messages during history ingestion.
        """
        for entry in reversed(self.entries):
            if any(t not in self._NON_INHERITABLE_TAGS for t in entry.tags):
                return entry
        return None

    def replace_tag(self, old_tag: str, turn_to_new_tags: dict[int, list[str]]) -> int:
        """Replace old_tag with new sub-tags in matching entries.

        Args:
            old_tag: Tag to remove from entries.
            turn_to_new_tags: {turn_number: [replacement_tags]}.

        Returns:
            Number of entries modified.
        """
        modified = 0
        for entry in self.entries:
            if old_tag in entry.tags:
                new_tags = turn_to_new_tags.get(entry.turn_number)
                if new_tags:
                    entry.tags = [t for t in entry.tags if t != old_tag] + new_tags
                    if entry.primary_tag == old_tag:
                        entry.primary_tag = new_tags[0]
                    modified += 1
        if modified:
            self._all_tags = {tag for entry in self.entries for tag in entry.tags}
        return modified

    def get_tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            for tag in entry.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

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
