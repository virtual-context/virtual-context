"""TagCanonicalizer: normalize tag variants to canonical forms."""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


class TagCanonicalizer:
    """Maps tag variants to canonical forms.

    Loaded from store at startup. Updated when aliases are registered.
    Normalization (lowercase, hyphenate) is always applied even without aliases.
    """

    def __init__(self, store=None) -> None:
        self._alias_cache: dict[str, str] = {}
        self._store = store
        self._known_tags: set[str] = set()

    def load(self) -> None:
        """Load aliases from store at startup."""
        if self._store:
            self._alias_cache = self._store.get_tag_aliases()
            # Seed known tags from canonical values
            self._known_tags.update(self._alias_cache.values())

    def canonicalize(self, tag: str) -> str:
        """Resolve a tag to its canonical form.

        Checks explicit aliases first, then auto-folds simple plurals
        (e.g. "filters" → "filter") when the singular is already known.
        """
        tag = tag.lower().strip().replace(" ", "-").replace("_", "-")
        tag = re.sub(r"-+", "-", tag).strip("-")

        # Explicit alias
        if tag in self._alias_cache:
            return self._alias_cache[tag]

        # Auto-fold plurals in either direction
        if tag.endswith("s") and len(tag) > 2:
            singular = tag[:-1]
            if singular in self._known_tags:
                return singular
        else:
            plural = tag + "s"
            if plural in self._known_tags:
                return plural

        self._known_tags.add(tag)
        return tag

    def canonicalize_list(self, tags: list[str]) -> list[str]:
        """Canonicalize and deduplicate a tag list, preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags:
            canonical = self.canonicalize(tag)
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register a new alias mapping."""
        normalized = alias.lower().strip().replace(" ", "-").replace("_", "-")
        self._alias_cache[normalized] = canonical
        if self._store:
            self._store.set_tag_alias(normalized, canonical)

    def get_aliases(self) -> dict[str, str]:
        """Return all registered aliases."""
        return dict(self._alias_cache)

    def auto_detect_aliases(self, threshold: float = 0.85) -> list[tuple[str, str]]:
        """Detect potential aliases using edit distance.
        Returns (alias_candidate, canonical_candidate) pairs.
        Canonical = the tag with higher usage count."""
        if not self._store:
            return []
        all_tags = self._store.get_all_tags()
        tag_names = [t.tag for t in all_tags]
        suggestions: list[tuple[str, str]] = []

        for i, t1 in enumerate(tag_names):
            for t2 in tag_names[i+1:]:
                max_len = max(len(t1), len(t2))
                if max_len == 0:
                    continue
                similarity = 1 - (_edit_distance(t1, t2) / max_len)
                if similarity >= threshold and t1 != t2:
                    c1 = next(t for t in all_tags if t.tag == t1)
                    c2 = next(t for t in all_tags if t.tag == t2)
                    if c1.usage_count >= c2.usage_count:
                        suggestions.append((t2, t1))
                    else:
                        suggestions.append((t1, t2))

        return suggestions


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Inline to avoid external dependency."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,       # deletion
                curr[j] + 1,            # insertion
                prev[j] + (ca != cb),   # substitution
            ))
        prev = curr
    return prev[-1]
