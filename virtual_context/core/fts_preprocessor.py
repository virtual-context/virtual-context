"""FTS query preprocessor: detects patterns that benefit from phrase quoting
or post-retrieval re-extraction.

Patterns handled:
- Sequential identifiers: "Step 7", "Turn 15", "Session 3"
- Tool/action names: underscore-joined identifiers like "mark_step", "file_read"
- Named entities: multi-word capitalized phrases like "Fan Showdown", "Red Rocks"
- Season/episode refs: "S4E6", "Season 4", "Episode 10"
- Date patterns: "May 2023", "March 10"
- User-provided quoted phrases: preserved as-is
"""

from __future__ import annotations

import re

# Sequential identifier patterns: "Step 7", "Turn 15", etc.
_SEQ_ID_RE = re.compile(
    r'\b(Step|Turn|Session|Episode|Round|Phase|Stage|Iteration)\s+(\d+)\b',
    re.IGNORECASE,
)

# Tool/function names: underscore-joined identifiers
_TOOL_NAME_RE = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b')

# Season/episode references: "S4E6", "Season 4 Episode 6"
_SEASON_EP_RE = re.compile(r'\bS(\d+)E(\d+)\b', re.IGNORECASE)
_SEASON_RE = re.compile(r'\b(Season|Series)\s+(\d+)\b', re.IGNORECASE)

# Date patterns: "May 2023", "March 10", "January 15, 2024"
_MONTHS = (
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
)
_MONTH_PATTERN = '|'.join(_MONTHS)
_DATE_RE = re.compile(
    rf'\b({_MONTH_PATTERN})\s+(\d{{1,4}}(?:,?\s+\d{{4}})?)\b',
    re.IGNORECASE,
)

# Multi-word capitalized phrases (2-4 words): "Fan Showdown", "Red Rocks"
_PROPER_NOUN_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b')


def extract_phrase_candidates(query: str) -> list[str]:
    """Extract substrings from the query that should be treated as exact phrases.

    Returns a list of phrase strings found in the query.
    These can be used for:
    - FTS5 phrase quoting
    - Post-retrieval substring matching within returned segments
    """
    phrases = []

    # 1. User-provided quoted phrases (already in quotes)
    for match in re.finditer(r'"([^"]+)"', query):
        phrases.append(match.group(1))

    # 2. Sequential identifiers: "Step 7", "Turn 15", etc.
    for match in _SEQ_ID_RE.finditer(query):
        phrases.append(match.group(0))

    # 3. Tool/function names
    for match in _TOOL_NAME_RE.finditer(query):
        phrases.append(match.group(0))

    # 4. Season/episode references
    for match in _SEASON_EP_RE.finditer(query):
        phrases.append(match.group(0))
    for match in _SEASON_RE.finditer(query):
        phrases.append(match.group(0))

    # 5. Date patterns
    for match in _DATE_RE.finditer(query):
        phrases.append(match.group(0))

    # 6. Multi-word proper nouns (only if not already captured)
    existing = set(phrases)
    for match in _PROPER_NOUN_RE.finditer(query):
        candidate = match.group(0)
        if candidate not in existing:
            phrases.append(candidate)

    return phrases


def re_extract_around_phrases(
    full_text: str,
    phrases: list[str],
    context_chars: int = 300,
) -> str | None:
    """Find the first occurrence of any phrase in full_text and return
    a context window around it.

    Used after FTS retrieval to re-extract the relevant excerpt when
    FTS snippet() returned a misleading region.

    Returns None if no phrase is found in the text.
    """
    for phrase in phrases:
        idx = full_text.find(phrase)
        if idx == -1:
            # Try case-insensitive
            idx = full_text.lower().find(phrase.lower())
        if idx >= 0:
            start = max(0, idx - context_chars)
            end = min(len(full_text), idx + len(phrase) + context_chars)
            # Extend to line boundaries
            while start > 0 and full_text[start] != '\n':
                start -= 1
            while end < len(full_text) and full_text[end] != '\n':
                end += 1
            return full_text[start:end].strip()
    return None
