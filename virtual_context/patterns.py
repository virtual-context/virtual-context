"""Regex patterns for broad-query and temporal-query detection.

Kept in a standalone module to avoid circular imports between types.py
and core/tag_generator.py.
"""

DEFAULT_BROAD_PATTERNS: list[str] = [
    r"\bwhat did (?:you|we) (?:say|mention|discuss|talk about|decide)\b",
    r"\bremind me (?:what|about|of)\b",
    r"\blooking back at (?:everything|what|our)\b",
    r"\b(?:summarize|recap) (?:what|everything|our|the)\b",
    r"\bcan you (?:summarize|recap|review)\b",
    r"\bwhat (?:have )?we (?:covered|discussed|talked about|decided)\b",
    r"\b(?:you|we) (?:mentioned|discussed|said|talked about) (?:earlier|before|previously)\b",
    r"\bgo (?:back over|back to|over) (?:what|everything)\b",
    r"\beach of (?:these|the|our) (?:threads|topics|discussions|conversations|areas|subjects)\b",
    r"\bacross (?:everything|all|the things) we(?:'ve)? (?:discussed|covered|talked about)\b",
    r"\bfrom (?:everything|all) we(?:'ve)? (?:discussed|covered|talked about)\b",
    # BUG-007: catch "summary of everything" and "everything we've talked about"
    r"\b(?:high-level )?summary of everything\b",
    r"\beverything we(?:'ve)? (?:talked|discussed|covered|worked on)\b",
    r"\b(?:complete|full) (?:checklist|list|summary) of (?:all|everything)\b",
    r"\bgiven everything we(?:'ve)?\b",
]

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
