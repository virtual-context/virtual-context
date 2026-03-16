"""Resolve relative date expressions in fact text against a session date.

Used as a post-processing fallback when the LLM fails to resolve
'yesterday', 'last Saturday', etc. to actual calendar dates.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Patterns for VC session date formats:
#   "1:56 pm on 8 May, 2023"
#   "10:31 am on 13 October, 2023"
#   "2023-05-08T13:56:00"
#   "2023/05/08 (Mon) 13:56"
_VC_SESSION_RE = re.compile(
    r"(\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?"
    r"(\d{1,2})\s+(\w+),?\s+(\d{4})",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Vague terms that should NOT be resolved
_VAGUE_TERMS = re.compile(
    r"\b(recently|a while ago|some time ago|long ago|ages ago|in the past)\b",
    re.IGNORECASE,
)


def parse_session_date(s: str) -> datetime | None:
    """Parse VC session date string into datetime.

    Handles: "1:56 pm on 8 May, 2023", "8 May, 2023", ISO formats.
    """
    if not s:
        return None

    # Try ISO first
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip()[:len(fmt) + 2], fmt)
        except (ValueError, IndexError):
            continue

    # Try VC format: "1:56 pm on 8 May, 2023" or "8 May, 2023"
    m = _VC_SESSION_RE.search(s)
    if m:
        day = int(m.group(2))
        month_str = m.group(3).lower()
        year = int(m.group(4))
        month = _MONTH_MAP.get(month_str)
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                pass

    return None


def resolve_relative_date(text: str, session_date_str: str) -> str | None:
    """Resolve relative date expressions in text against a session date.

    Returns ISO date string (YYYY-MM-DD) or None if no resolvable pattern found.
    Uses dateparser for complex/multilingual expressions, regex fallback for simple ones.
    """
    if not text or not session_date_str:
        return None

    ref = parse_session_date(session_date_str)
    if ref is None:
        return None

    # Skip vague terms
    if _VAGUE_TERMS.search(text):
        return None

    # Try dateparser first — handles multilingual, day-of-week math, etc.
    try:
        resolved = _try_dateparser(text, ref)
        if resolved:
            return resolved
    except Exception:
        pass

    # Regex fallback for simple patterns
    return _try_regex_fallback(text, ref)


def _try_dateparser(text: str, ref: datetime) -> str | None:
    import dateparser

    # Extract candidate phrases from text
    candidates = _extract_temporal_phrases(text)
    if not candidates:
        return None

    settings = {
        "RELATIVE_BASE": ref,
        "PREFER_DATES_FROM": "past",
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    # False positives: common words dateparser might parse as dates
    false_positives = {
        "may", "march", "will", "can", "sat", "sun",
        "mon", "tue", "wed", "thu", "fri",
    }

    for phrase in candidates:
        if phrase.lower() in false_positives:
            continue
        # Normalize "last Saturday" → "Saturday" (dateparser handles bare day names)
        # Also normalize "last weekend" → "Saturday"
        normalized = re.sub(
            r"^last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",
            r"\1", phrase, flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^last\s+weekend$", "Saturday", normalized, flags=re.IGNORECASE,
        )
        result = dateparser.parse(normalized, settings=settings)
        if result and result != ref:
            return result.strftime("%Y-%m-%d")

    return None


def _extract_temporal_phrases(text: str) -> list[str]:
    patterns = [
        r"\b(yesterday)\b",
        r"\b(today)\b",
        r"\b(tomorrow)\b",
        r"\b(last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        r"\b(next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        r"\b(last\s+(?:week|month|year|night))\b",
        r"\b(last\s+weekend)\b",
        r"\b(next\s+(?:week|month|year))\b",
        r"\b(this\s+(?:week|month|year|morning|afternoon|evening))\b",
        r"\b(\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
        r"\b(a\s+(?:week|month|year)\s+ago)\b",
        r"\b(two\s+(?:days?|weeks?|months?)\s+ago)\b",
        r"\b(three\s+(?:days?|weeks?|months?)\s+ago)\b",
    ]
    found = []
    text_lower = text.lower()
    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            found.append(m.group(1))
    return found


_SIMPLE_OFFSETS = {
    r"\byesterday\b": -1,
    r"\btomorrow\b": 1,
    r"\blast night\b": -1,
    r"\btoday\b": 0,
    r"\bthis morning\b": 0,
    r"\bthis afternoon\b": 0,
    r"\bthis evening\b": 0,
    r"\btonight\b": 0,
    r"\blast week\b": -7,
    r"\bnext week\b": 7,
    r"\blast month\b": -30,
    r"\bnext month\b": 30,
    r"\blast year\b": -365,
    r"\bnext year\b": 365,
}


def normalize_fact_text(text: str, session_date_str: str) -> str:
    """Replace relative date expressions in fact text with resolved calendar dates.

    E.g. "ran a race last Saturday" with session May 25, 2023
      → "ran a race on Saturday, May 20, 2023"
    Returns original text unchanged if no resolution possible.
    """
    if not text or not session_date_str:
        return text

    ref = parse_session_date(session_date_str)
    if ref is None:
        return text

    if _VAGUE_TERMS.search(text):
        return text

    # Build replacement map: pattern → (regex, resolved date string)
    replacements: list[tuple[re.Pattern, str]] = []

    # Day-of-week patterns: "last Saturday" → "on Saturday, May 20, 2023"
    for day_name in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        pat = re.compile(rf"\blast\s+{day_name}\b", re.IGNORECASE)
        if pat.search(text):
            try:
                import dateparser
                resolved = dateparser.parse(day_name, settings={
                    "RELATIVE_BASE": ref,
                    "PREFER_DATES_FROM": "past",
                    "RETURN_AS_TIMEZONE_AWARE": False,
                })
                if resolved:
                    date_str = resolved.strftime(f"on %A, %B %-d, %Y")
                    replacements.append((pat, date_str))
            except Exception:
                pass

    # Simple offset patterns
    _TEXT_REPLACEMENTS = [
        (r"\byesterday\b", -1),
        (r"\blast night\b", -1),
        (r"\btomorrow\b", 1),
        (r"\blast week\b", -7),
        (r"\bnext week\b", 7),
        (r"\blast month\b", -30),
        (r"\bnext month\b", 30),
        (r"\blast year\b", -365),
        (r"\bnext year\b", 365),
        (r"\blast weekend\b", -2),  # approximate to Saturday
    ]
    for pattern_str, offset in _TEXT_REPLACEMENTS:
        pat = re.compile(pattern_str, re.IGNORECASE)
        if pat.search(text) and not any(p.pattern == pat.pattern for p, _ in replacements):
            target = ref + timedelta(days=offset)
            if abs(offset) <= 7:
                date_str = f"on {target.strftime('%B %-d, %Y')}"
            elif abs(offset) <= 31:
                date_str = f"in {target.strftime('%B %Y')}"
            else:
                date_str = f"in {target.strftime('%Y')}"
            replacements.append((pat, date_str))

    result = text
    for pat, replacement in replacements:
        result = pat.sub(replacement, result, count=1)

    return result


def _try_regex_fallback(text: str, ref: datetime) -> str | None:
    text_lower = text.lower()
    for pattern, offset_days in _SIMPLE_OFFSETS.items():
        if re.search(pattern, text_lower):
            target = ref + timedelta(days=offset_days)
            return target.strftime("%Y-%m-%d")
    return None
