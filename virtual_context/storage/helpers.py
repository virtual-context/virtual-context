"""Shared helpers for storage backends."""

from __future__ import annotations

from datetime import datetime, timezone


def dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def extract_excerpt(text: str, query: str, context_chars: int = 200) -> str:
    """Extract text around the first occurrence of query."""
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:context_chars * 2]
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(query) + context_chars)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    return excerpt
