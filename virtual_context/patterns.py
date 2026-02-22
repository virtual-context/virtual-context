"""Regex patterns for temporal-query detection.

Kept in a standalone module to avoid circular imports between types.py
and core/tag_generator.py.
"""

DEFAULT_TEMPORAL_PATTERNS: list[str] = [
    r"\b(?:the )?(?:very )?first (?:thing|topic|discussion|time|question)\b",
    r"\bat the (?:very )?(?:beginning|start)\b",
    r"\bearly (?:on|in our)\b",
    r"\b(?:initially|originally) (?:we|you|i)\b",
    r"\bgoing (?:all the )?way back\b",
    r"\bwhen we (?:first|started)\b",
    r"\bthe (?:very )?first thing we (?:discussed|talked about|covered)\b",
    r"\bback to the (?:very )?(?:beginning|start|first)\b",
]
