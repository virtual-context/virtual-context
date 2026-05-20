"""TurnTagIndex: live index of per-turn tag metadata."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..types import TurnTagEntry


class TurnTagIndex:
    """Live index maintained as conversation progresses.

    Updated every round trip by the engine. Read by segmenter and retriever.
    Purely in-memory — not persisted. Rebuilt each session.

    Bounded at :pyattr:`MAX_ENTRIES` to cap per-instance memory.  When
    the bound is reached, the oldest entry is evicted from
    :pyattr:`entries` and from the three reference dicts
    (:pyattr:`_by_logical_turn`, :pyattr:`_by_canonical_turn`,
    :pyattr:`_by_hash`); :pyattr:`_all_tags` is recomputed from the
    remaining entries so tags whose sole carrier was the evicted entry
    drop out.

    Aggregation methods (:py:meth:`compute_cover_set`,
    :py:meth:`get_tag_counts`, :py:meth:`get_tag_velocity`,
    :py:meth:`replace_tag`) now operate on the bounded window rather
    than the full conversation history.  For typical conversations
    under :pyattr:`MAX_ENTRIES` turns this is identical to the
    pre-bound behavior; for longer conversations the aggregations
    approximate "recent activity" rather than "full history."  The
    durable source of truth in the ``canonical_turns`` SQL table is
    unchanged; callers that need full-history aggregation can route
    through :py:meth:`VirtualContextEngine._restore_from_canonical_rows`
    to rebuild a fresh full-history index for that one operation.

    Eviction uses identity-guarded ``is``-comparison before removing
    each dict entry because the duplicate-detection guards in
    :py:meth:`append` reject incoming duplicates and KEEP the original
    entry; without the identity check, evicting an older slot that
    happened to share a key with a later-rejected duplicate would
    silently drop the kept entry from the dicts.
    """

    MAX_ENTRIES = 5_000

    def __init__(self) -> None:
        self.entries: list[TurnTagEntry] = []
        self._by_logical_turn: dict[int, TurnTagEntry] = {}
        self._by_canonical_turn: dict[str, TurnTagEntry] = {}
        self._by_hash: dict[str, TurnTagEntry] = {}
        self._all_tags: set[str] = set()
        self._evicted_count: int = 0

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

        # LRU eviction BEFORE the new append so the post-condition is
        # always ``len(self.entries) <= MAX_ENTRIES`` and one eviction
        # at most per append keeps the amortized cost bounded.
        if len(self.entries) >= self.MAX_ENTRIES:
            self._evict_oldest()

        self.entries.append(entry)
        self._by_logical_turn[entry.turn_number] = entry
        if entry.canonical_turn_id:
            self._by_canonical_turn[entry.canonical_turn_id] = entry
        if entry.message_hash:
            self._by_hash[entry.message_hash] = entry
        self._all_tags.update(entry.tags)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry and drop it from all reference dicts.

        Identity-guarded ``is``-comparison protects against the case
        where ``append``'s duplicate-detection rejected a later entry
        with the same ``turn_number`` / ``canonical_turn_id`` /
        ``message_hash`` as the oldest entry; without the guard, the
        dict slot pointing at the kept entry would be silently
        deleted along with the eviction.
        """
        if not self.entries:
            return
        oldest = self.entries.pop(0)
        if self._by_logical_turn.get(oldest.turn_number) is oldest:
            del self._by_logical_turn[oldest.turn_number]
        if oldest.canonical_turn_id and self._by_canonical_turn.get(oldest.canonical_turn_id) is oldest:
            del self._by_canonical_turn[oldest.canonical_turn_id]
        if oldest.message_hash and self._by_hash.get(oldest.message_hash) is oldest:
            del self._by_hash[oldest.message_hash]
        self._evicted_count += 1
        # Rebuild ``_all_tags`` from the remaining entries so tags
        # whose only carrier was ``oldest`` drop out of the set.
        # Per design doc §2.4 benchmark: ~0.43 ms at MAX_ENTRIES=5_000
        # with average ~3 tags per entry; acceptable on the append
        # hot path.
        self._all_tags = {tag for entry in self.entries for tag in entry.tags}

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
