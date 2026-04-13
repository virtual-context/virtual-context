"""Quote search: full-text + semantic + description-aware search.

Orchestrates FTS, semantic embedding search, and description scanning
to find specific phrases or keywords across all stored conversation text.
Extracted from engine.py.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from datetime import date
from itertools import product

from ..types import QuoteResult, SegmentMetadata
from .semantic_search import SemanticSearchManager
from .store import ContextStore

logger = logging.getLogger(__name__)

_FIND_QUOTE_MODES = {"lookup", "exact_value"}
_SEARCH_SUMMARY_MODES = {"lookup", "coverage", "aggregate_total"}
_MULTI_EVIDENCE_SUMMARY_MODES = {"coverage", "aggregate_total"}
_OVERFETCH_MULTIPLIER = 3
_OVERFETCH_MIN_RESULTS = 20
_COVERAGE_EXTRA_QUERY_LIMIT = 2
_AGGREGATE_SUMMARY_SCAN_LIMIT = 1500
_AGGREGATE_COMPONENT_CANDIDATES_LIMIT = 12
_AGGREGATE_COMPONENT_DISTANCE_CHARS = 120
_AGGREGATE_SUMMARY_PER_COMPONENT_LIMIT = 8
_AGGREGATE_SUMMARY_TARGET_ADDITIONS = 24
_AGGREGATE_AMBIGUITY_RELATIVE_DELTA = 0.05
_AGGREGATE_AMBIGUITY_SOURCE_GAP = 0.75
_AGGREGATE_AMBIGUITY_TERM_HITS_GAP = 2.0
_AGGREGATE_AMBIGUITY_SPREAD_DAYS_GAP = 21
_AGGREGATE_AMBIGUITY_ANCHOR_GAP = 1.5
_QUERY_STOPWORDS = {
    "about", "across", "after", "again", "around", "been", "between", "bring",
    "called", "catch", "did", "does", "doing", "during", "each", "exact",
    "from", "have", "into", "just", "mention", "mentioned", "need", "only",
    "over", "past", "same", "search", "setting", "specific", "that", "their",
    "them", "then", "these", "this", "those", "through", "under", "using",
    "what", "when", "where", "which", "while", "with", "would",
}
_PERCENT_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%")
_VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_QUERY_RATE_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s+queries\s*(?:/|per)\s*(hour|sec|second)\b", re.IGNORECASE)
_QUANTITY_WITH_UNIT_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\s+"
    r"(?:(?:[a-z][a-z0-9-]*\s+){0,2})?"
    r"(documents?|records?|vectors?|queries?|items?|tasks?|users?)\b",
    re.IGNORECASE,
)
_COVERAGE_LIST_START_RE = re.compile(
    r"\b(?:across|between|among|including|covering|combining|combined)\b(.+?)(?:[?.!]|$)",
    re.IGNORECASE,
)
_COVERAGE_SPLIT_RE = re.compile(r"\s*(?:,|/|\band\b|\bor\b|\bplus\b)\s*", re.IGNORECASE)
_COVERAGE_LEADING_FILLER_RE = re.compile(
    r"^(?:my|our|the|these|those|their|all|both)\b\s*",
    re.IGNORECASE,
)
_COVERAGE_TRAILING_FILLER_RE = re.compile(
    r"\b(?:efforts?|systems?|components?|sessions?|workstreams?|initiatives?|"
    r"projects?|plans?|pipelines?|modules?|features?|combined|altogether|"
    r"overall|total|totals)\b",
    re.IGNORECASE,
)
_COVERAGE_SKIP_COMPONENTS = {
    "combined",
    "overall",
    "queries per second",
    "total",
}
_QUANTITY_SCALE_MAP = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}
_QUANTITY_UNIT_MAP = {
    "document": "documents",
    "documents": "documents",
    "record": "records",
    "records": "records",
    "vector": "vectors",
    "vectors": "vectors",
    "query": "queries",
    "queries": "queries",
    "item": "items",
    "items": "items",
    "task": "tasks",
    "tasks": "tasks",
    "user": "users",
    "users": "users",
}


# Deterministic query intent detection for state-updated questions.
_CURRENT_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcurrently\b", re.IGNORECASE),
    re.compile(r"\bnow\b", re.IGNORECASE),
    re.compile(r"\blatest\b", re.IGNORECASE),
    re.compile(r"\bat the moment\b", re.IGNORECASE),
    re.compile(r"\bthese days\b", re.IGNORECASE),
)
_YES_NO_SELF_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:have|did|do|am|was|were)\s+i\b", re.IGNORECASE),
    re.compile(r"\b(?:have|did|do|am|was|were)\s+i\b", re.IGNORECASE),
)
_AFFIRMING_USER_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\buser:\s*i(?:'m| am)\s+setting up\b", re.IGNORECASE),
    re.compile(r"\buser:\s*i(?:'ve| have)\s+set up\b", re.IGNORECASE),
    re.compile(r"\buser:\s*i(?:'ve| have)\s+configured\b", re.IGNORECASE),
    re.compile(r"\buser:\s*i(?:'ve| have)\s+implemented\b", re.IGNORECASE),
)
_NEGATING_USER_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\buser:\s*i(?:'ve| have)?\s*never\s+\w+", re.IGNORECASE),
    re.compile(r"\buser:\s*i\s+haven't\s+\w+", re.IGNORECASE),
    re.compile(r"\buser:\s*i\s+have\s+not\s+\w+", re.IGNORECASE),
    re.compile(r"\buser:\s*i\s+did(?:n't| not)\s+\w+", re.IGNORECASE),
    re.compile(r"\buser:\s*i(?:'m| am)\s+concerned about the lack of\b", re.IGNORECASE),
)
_PREFERENCE_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\buser\s+(?:specified|requested|preferred|wanted|required)\b", re.IGNORECASE),
    re.compile(r"\b(?:requirement|preference)\b", re.IGNORECASE),
)
_RATE_PER_HOUR_RE = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*/\s*hour\b", re.IGNORECASE)
_INSTANCE_COUNT_RE = re.compile(
    r"\b(\d[\d,]*)\s+(?:[A-Za-z0-9_.-]+\s+){0,3}?instances?\b",
    re.IGNORECASE,
)


def _detect_query_intent(query: str) -> str:
    for pattern in _CURRENT_STATE_PATTERNS:
        if pattern.search(query):
            # "or" in the query signals a disjunction (e.g. "led or am
            # currently leading") — the question spans past and present,
            # so treating it as pure current-state would suppress the
            # historical part.
            if re.search(r"\bor\b", query, re.IGNORECASE):
                return "default"
            return "current_state"
    return "default"


def _is_yes_no_self_state_question(query: str, intent_context: str) -> bool:
    haystacks = [query, intent_context]
    return any(
        haystack.strip() and any(pattern.search(haystack) for pattern in _YES_NO_SELF_STATE_PATTERNS)
        for haystack in haystacks
    )


def _maybe_build_lookup_reader_hint(
    *,
    query: str,
    intent_context: str,
    formatted: list[dict[str, object]],
) -> str | None:
    if not _is_yes_no_self_state_question(query, intent_context):
        return None

    has_affirming = False
    has_negating = False
    for row in formatted:
        excerpt = str(row.get("excerpt", "")).strip()
        if not excerpt:
            continue
        if any(pattern.search(excerpt) for pattern in _AFFIRMING_USER_STATE_PATTERNS):
            has_affirming = True
        if any(pattern.search(excerpt) for pattern in _NEGATING_USER_STATE_PATTERNS):
            has_negating = True
        if has_affirming and has_negating:
            return (
                "CONTRADICTION CHECK: The returned user quotes may conflict. "
                "Before answering, compare them for mutually incompatible user "
                "statements. If one quote says the setup exists and another says "
                "it was never set up or is still missing, answer that the history "
                "contains contradictory information, cite both sides briefly, and "
                "ask which is correct."
            )
    return None


def _default_lookup_reader_hint() -> str:
    return (
        "LOOKUP MODE: Compare the returned quotes before answering. If the "
        "evidence conflicts, say the history contains contradictory information, "
        "cite both sides briefly, and ask which is correct. If the evidence does "
        "not conflict, answer directly from the best-supported quotes."
    )


def _split_summary_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]


def _infer_anchor_provider(sentence: str) -> str:
    lowered = sentence.lower()
    if "aws" in lowered and "ec2" in lowered:
        return "AWS EC2"
    if "ec2" in lowered:
        return "AWS EC2"
    if "aws" in lowered:
        return "AWS"
    if "gcp" in lowered or "google cloud" in lowered:
        return "GCP"
    if "azure" in lowered:
        return "Azure"
    return ""


def _format_hourly_total(rate: float, count: int) -> str:
    total = rate * count
    if abs(total - round(total)) < 1e-9:
        return f"${int(round(total)):,}/hour"
    return f"${total:,.2f}/hour"


def _extract_summary_preference_anchor(
    formatted: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]] | None:
    for idx, row in enumerate(formatted[:5], start=1):
        excerpt = str(row.get("excerpt", "")).strip()
        if not excerpt:
            continue
        for sentence in _split_summary_sentences(excerpt):
            rate_match = _RATE_PER_HOUR_RE.search(sentence)
            count_match = _INSTANCE_COUNT_RE.search(sentence)
            if not rate_match or not count_match:
                continue
            if not any(pattern.search(sentence) for pattern in _PREFERENCE_CUE_PATTERNS):
                continue
            rate_text = f"${rate_match.group(1)}/hour"
            try:
                rate_value = float(rate_match.group(1))
                count_value = int(count_match.group(1).replace(",", ""))
            except ValueError:
                continue
            anchor = {
                "type": "user_requirement",
                "provider": _infer_anchor_provider(sentence),
                "hourly_rate": rate_text,
                "instance_count": f"{count_value:,}",
                "evidence": sentence,
                "source_result_index": idx,
            }
            calc = {
                "hourly_compute_total": _format_hourly_total(rate_value, count_value),
                "formula": f"{rate_text} * {count_value:,} instances = {_format_hourly_total(rate_value, count_value)}",
            }
            return anchor, calc
    return None


_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTH_MAP, key=len, reverse=True))

# Month-DD-YYYY: July-01-2024, Aug 5, 2024, September 15 2024
_MDY_RE = re.compile(
    r"\b(" + _MONTH_PATTERN + r")[-.\s]+(\d{1,2})[-,.\s]+(\d{4})\b",
    re.IGNORECASE,
)
# DD-Month-YYYY: 01-July-2024, 5 Aug 2024, 15.Sep.2024
_DMY_RE = re.compile(
    r"\b(\d{1,2})[-.\s]+(" + _MONTH_PATTERN + r")[-,.\s]+(\d{4})\b",
    re.IGNORECASE,
)
# Month-YYYY (no day): July-2024, Aug 2024
_MY_RE = re.compile(
    r"\b(" + _MONTH_PATTERN + r")[-.\s]+(\d{4})\b",
    re.IGNORECASE,
)


def _parse_session_date(raw: str) -> date | None:
    """Best-effort parse for session date strings into ``date`` values."""
    s = (raw or "").strip()
    if not s:
        return None

    # ISO numeric: 2024-07-01 or 2024/07/01
    m = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Month-Day-Year: July-01-2024, Aug 5, 2024, September 15 2024
    m = _MDY_RE.search(s)
    if m:
        mo = _MONTH_MAP.get(m.group(1).lower())
        if mo:
            try:
                return date(int(m.group(3)), mo, int(m.group(2)))
            except ValueError:
                pass

    # Day-Month-Year: 01-July-2024, 5 Aug 2024, 15.Sep.2024
    m = _DMY_RE.search(s)
    if m:
        mo = _MONTH_MAP.get(m.group(2).lower())
        if mo:
            try:
                return date(int(m.group(3)), mo, int(m.group(1)))
            except ValueError:
                pass

    # Month-Year (no day): July-2024, Aug 2024
    m = _MY_RE.search(s)
    if m:
        mo = _MONTH_MAP.get(m.group(1).lower())
        if mo:
            try:
                return date(int(m.group(2)), mo, 1)
            except ValueError:
                pass

    # Year-Month numeric (no day): 2024-07
    m = re.search(r"\b(\d{4})[-/](\d{2})\b", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass

    return None


def _normalize_session_date(raw: str) -> str:
    parsed = _parse_session_date(raw)
    return parsed.isoformat() if parsed else ""


def _normalize_find_quote_mode(mode: str) -> str:
    normalized = (mode or "lookup").strip().lower()
    return normalized if normalized in _FIND_QUOTE_MODES else "lookup"


def _normalize_search_summary_mode(mode: str) -> str:
    normalized = (mode or "lookup").strip().lower()
    return normalized if normalized in _SEARCH_SUMMARY_MODES else "lookup"


def _candidate_limit(max_results: int, mode: str) -> int:
    if mode == "lookup":
        return max_results
    return max(max_results * _OVERFETCH_MULTIPLIER, _OVERFETCH_MIN_RESULTS)


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-zA-Z0-9_.%-]+", query.lower()):
        if len(raw) < 2 or raw in _QUERY_STOPWORDS:
            continue
        if raw not in seen:
            seen.add(raw)
            terms.append(raw)
    return terms


def _query_bigrams(terms: list[str]) -> list[str]:
    return [f"{terms[i]} {terms[i + 1]}" for i in range(len(terms) - 1)]


def _ordered_query_tokens(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9_.%-]+", query or "") if len(token) >= 2]


def _query_trigrams(terms: list[str]) -> list[str]:
    return [f"{terms[i]} {terms[i + 1]} {terms[i + 2]}" for i in range(len(terms) - 2)]


def _matched_query_terms(text: str, terms: list[str]) -> list[str]:
    lowered = (text or "").lower()
    if not lowered or not terms:
        return []
    matched: list[str] = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            matched.append(term)
    return matched


def _matched_query_phrases(text: str, phrases: list[str]) -> list[str]:
    lowered = (text or "").lower()
    if not lowered or not phrases:
        return []
    return [phrase for phrase in phrases if phrase in lowered]


def _matched_term_span(text: str, matched_terms: list[str]) -> int:
    lowered = (text or "").lower()
    if len(matched_terms) < 2 or not lowered:
        return 1_000_000
    starts: list[int] = []
    for term in matched_terms:
        match = re.search(rf"\b{re.escape(term)}\b", lowered)
        if match:
            starts.append(match.start())
    if len(starts) < 2:
        return 1_000_000
    return max(starts) - min(starts)


def _normalize_coverage_component(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return ""
    text = _COVERAGE_LEADING_FILLER_RE.sub("", text)
    text = _COVERAGE_TRAILING_FILLER_RE.sub(" ", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    if not text or text in _COVERAGE_SKIP_COMPONENTS:
        return ""
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return ""
    if all(token in _QUERY_STOPWORDS for token in tokens):
        return ""
    return text


def _extract_coverage_components(*texts: str) -> list[str]:
    components: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _COVERAGE_LIST_START_RE.finditer(text):
            for part in _COVERAGE_SPLIT_RE.split(match.group(1)):
                component = _normalize_coverage_component(part)
                if not component or component in seen:
                    continue
                seen.add(component)
                components.append(component)
            if len(components) >= 2:
                return components
    return components


def _coverage_anchor_query(query: str, components: list[str]) -> str:
    working = (query or "").lower()
    for component in components:
        for token in re.findall(r"[a-z0-9]+", component):
            working = re.sub(rf"\b{re.escape(token)}\b", " ", working)
    terms = [
        token
        for token in re.findall(r"[a-z0-9_.%-]+", working)
        if token not in _QUERY_STOPWORDS
    ]
    return " ".join(terms[:4]).strip()


def _requested_quantity_units(*texts: str) -> list[str]:
    units: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in re.findall(r"[a-z]+", (text or "").lower()):
            normalized = _QUANTITY_UNIT_MAP.get(token)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            units.append(normalized)
    return units


def _build_coverage_queries(query: str, intent_context: str) -> tuple[list[str], list[str]]:
    components = _extract_coverage_components(intent_context, query)
    if not components:
        return [], []

    anchor = _coverage_anchor_query(intent_context or query, components)
    if not anchor:
        anchor = _coverage_anchor_query(query, components)
    queries: list[str] = []
    seen: set[str] = set()
    max_queries = max(len(components) * _COVERAGE_EXTRA_QUERY_LIMIT, 1)
    for component in components:
        for candidate in (
            f"{anchor} {component}".strip() if anchor else "",
            component,
        ):
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            queries.append(candidate)
            if len(queries) >= max_queries:
                return components, queries
    return components, queries


def _match_coverage_components(
    text: str,
    components: list[str],
    *,
    tag: str = "",
) -> list[str]:
    haystack = f"{tag} {text}".lower()
    matched: list[str] = []
    for component in components:
        tokens = re.findall(r"[a-z0-9]+", component)
        if tokens and all(re.search(rf"\b{re.escape(token)}\b", haystack) for token in tokens):
            matched.append(component)
    return matched


def _extract_value_tokens(text: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    seen_spans: list[tuple[int, int]] = []
    values: list[str] = []
    for pattern in (_ISO_DATE_RE, _VERSION_RE, _PERCENT_RE, _NUMBER_RE):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < prev_end and end > prev_start for prev_start, prev_end in seen_spans):
                continue
            value = match.group(0).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            seen_spans.append((start, end))
            values.append(value)
            if len(values) >= limit:
                return values
    return values


def _extract_percentages(text: str, *, limit: int = 4) -> list[str]:
    percentages: list[str] = []
    seen: set[str] = set()
    for match in _PERCENT_RE.finditer(text or ""):
        value = match.group(0).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        percentages.append(value)
        if len(percentages) >= limit:
            break
    return percentages


def _extract_versions(text: str, *, limit: int = 4) -> list[str]:
    versions: list[str] = []
    seen: set[str] = set()
    for match in _VERSION_RE.finditer(text or ""):
        value = match.group(0).strip()
        if not value:
            continue
        normalized = value.lstrip("vV")
        if normalized in seen:
            continue
        seen.add(normalized)
        versions.append(normalized)
        if len(versions) >= limit:
            break
    return versions


def _wants_percentage_value(*texts: str) -> bool:
    for text in texts:
        lowered = (text or "").lower()
        if "%" in lowered:
            return True
        if re.search(r"\b(percent|percentage|rate|uptime)\b", lowered):
            return True
    return False


def _wants_version_value(*texts: str) -> bool:
    for text in texts:
        lowered = (text or "").lower()
        if re.search(r"\bversion\b", lowered):
            return True
    return False


def _question_is_first_person(*texts: str) -> bool:
    for text in texts:
        lowered = (text or "").lower()
        if re.search(r"\b(am i|i am|i'm|i’ve|i've|my|me)\b", lowered):
            return True
    return False


def _excerpt_starts_with_user_statement(text: str) -> bool:
    stripped = (text or "").lstrip()
    return stripped.startswith("User:")


def _parse_scaled_quantity(amount_raw: str, scale_raw: str) -> float | None:
    try:
        amount = float(amount_raw.replace(",", ""))
    except ValueError:
        return None
    scale = _QUANTITY_SCALE_MAP.get((scale_raw or "").lower(), 1.0)
    return amount * scale


def _format_aggregate_value(value: float, unit: str) -> str:
    if value >= 1_000_000:
        millions = value / 1_000_000
        if abs(millions - round(millions)) < 1e-9:
            amount = f"{int(round(millions))} million"
        else:
            amount = f"{millions:.1f}".rstrip("0").rstrip(".") + " million"
        return f"{amount} {unit}"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,} {unit}"
    return f"{value:.1f}".rstrip("0").rstrip(".") + f" {unit}"


def _extract_quantity_mentions(text: str, *, limit: int = 8) -> list[dict[str, object]]:
    mentions: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for match in _QUANTITY_WITH_UNIT_RE.finditer(text or ""):
        amount_raw = match.group(1)
        scale_raw = match.group(2) or ""
        unit_raw = match.group(3).lower()
        normalized_unit = _QUANTITY_UNIT_MAP.get(unit_raw)
        if not normalized_unit:
            continue
        normalized_value = _parse_scaled_quantity(amount_raw, scale_raw)
        if normalized_value is None:
            continue
        key = (normalized_unit, match.group(0).lower(), match.start())
        if key in seen:
            continue
        seen.add(key)
        mentions.append({
            "value": match.group(0),
            "unit": normalized_unit,
            "normalized_value": normalized_value,
            "start": match.start(),
            "end": match.end(),
        })
        if len(mentions) >= limit:
            break
    return mentions


def _aggregate_quantity_profile(
    quantities: list[dict[str, object]],
    requested_units: set[str],
) -> tuple[int, int, int, int]:
    if requested_units:
        requested_quantities = [
            quantity for quantity in quantities if quantity.get("unit") in requested_units
        ]
    else:
        requested_quantities = list(quantities)
    distinct_requested_values = {
        (str(quantity.get("unit", "")), float(quantity.get("normalized_value")))
        for quantity in requested_quantities
        if isinstance(quantity.get("normalized_value"), (int, float))
    }
    other_unit_count = max(0, len(quantities) - len(requested_quantities))
    return (
        len(requested_quantities),
        len(distinct_requested_values),
        other_unit_count,
        len(quantities),
    )


def _source_weight(match_type: str, mode: str) -> int:
    lookup_weights = {
        "fts": 5,
        "like": 5,
        "turn_search": 4,
        "turn_semantic": 4,
        "tool_output": 4,
        "description": 3,
        "semantic": 2,
    }
    exact_value_weights = {
        "fts": 6,
        "like": 6,
        "turn_search": 5,
        "turn_semantic": 5,
        "tool_output": 5,
        "description": 3,
        "semantic": 1,
    }
    coverage_weights = {
        "fts": 5,
        "like": 5,
        "turn_search": 4,
        "turn_semantic": 3,
        "tool_output": 4,
        "description": 3,
        "semantic": 3,
    }
    aggregate_total_weights = {
        "fts": 6,
        "like": 6,
        "turn_search": 5,
        "turn_semantic": 3,
        "tool_output": 5,
        "summary_scan": 5,
        "description": 4,
        "semantic": 2,
    }
    table = (
        exact_value_weights if mode == "exact_value"
        else aggregate_total_weights if mode == "aggregate_total"
        else coverage_weights if mode == "coverage"
        else lookup_weights
    )
    return table.get(match_type or "", 1)


def _base_quote_score(qr: QuoteResult, query: str, mode: str) -> tuple[float, ...]:
    text_lower = (qr.text or "").lower()
    query_lower = query.lower().strip()
    terms = _query_terms(query)
    bigrams = _query_bigrams(terms)
    term_hits = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", text_lower))
    bigram_hits = sum(1 for phrase in bigrams if phrase in text_lower)
    exact_phrase = 1 if query_lower and query_lower in text_lower else 0
    value_tokens = _extract_value_tokens(qr.text or "")
    value_count = len(value_tokens)
    quantity_mentions = _extract_quantity_mentions(qr.text or "", limit=4)
    quantity_count = len(quantity_mentions)
    semantic_score = qr.similarity if qr.match_type == "semantic" else 0.0
    source_weight = _source_weight(qr.match_type, mode)

    if mode == "exact_value":
        return (
            1.0 if value_count > 0 else 0.0,
            float(value_count),
            float(exact_phrase),
            float((bigram_hits * 2) + term_hits),
            float(source_weight),
            float(semantic_score),
        )
    if mode == "aggregate_total":
        return (
            1.0 if quantity_count > 0 else 0.0,
            float(quantity_count),
            float(exact_phrase),
            float((bigram_hits * 2) + term_hits),
            float(source_weight),
            float(value_count),
            float(semantic_score),
        )
    return (
        float(exact_phrase),
        float((bigram_hits * 2) + term_hits),
        float(source_weight),
        float(value_count),
        float(semantic_score),
    )


def _rerank_quote_results(
    results: list[QuoteResult],
    query: str,
    *,
    max_results: int,
    mode: str,
    coverage_components: list[str] | None = None,
) -> list[QuoteResult]:
    if mode == "lookup":
        ordered_tokens = _ordered_query_tokens(query)
        filtered_terms = _query_terms(query)
        bigrams = _query_bigrams(ordered_tokens)
        trigrams = _query_trigrams(ordered_tokens)
        if len(results) <= 1 or (not filtered_terms and not bigrams and not trigrams):
            return results[:max_results]

        profiles: list[dict[str, object]] = []
        term_doc_counts: OrderedDict[str, int] = OrderedDict()
        bigram_doc_counts: OrderedDict[str, int] = OrderedDict()
        trigram_doc_counts: OrderedDict[str, int] = OrderedDict()

        for index, qr in enumerate(results):
            matched_terms = _matched_query_terms(qr.text or "", filtered_terms)
            matched_bigrams = _matched_query_phrases(qr.text or "", bigrams)
            matched_trigrams = _matched_query_phrases(qr.text or "", trigrams)
            for token in matched_terms:
                term_doc_counts[token] = term_doc_counts.get(token, 0) + 1
            for phrase in matched_bigrams:
                bigram_doc_counts[phrase] = bigram_doc_counts.get(phrase, 0) + 1
            for phrase in matched_trigrams:
                trigram_doc_counts[phrase] = trigram_doc_counts.get(phrase, 0) + 1
            profiles.append({
                "index": index,
                "result": qr,
                "matched_terms": matched_terms,
                "matched_bigrams": matched_bigrams,
                "matched_trigrams": matched_trigrams,
                "term_span": _matched_term_span(qr.text or "", matched_terms),
                "starts_with_user": 1 if _excerpt_starts_with_user_statement(qr.text or "") else 0,
            })

        if not term_doc_counts and not bigram_doc_counts and not trigram_doc_counts:
            return results[:max_results]

        ranked = sorted(
            profiles,
            key=lambda item: (
                sum(1.0 / trigram_doc_counts[phrase] for phrase in item["matched_trigrams"]),
                len(item["matched_trigrams"]),
                sum(1.0 / bigram_doc_counts[phrase] for phrase in item["matched_bigrams"]),
                len(item["matched_bigrams"]),
                sum(1.0 / term_doc_counts[token] for token in item["matched_terms"]),
                len(item["matched_terms"]),
                item["starts_with_user"],
                -int(item["term_span"]),
                *_base_quote_score(item["result"], query, mode),
                -int(item["index"]),
            ),
            reverse=True,
        )

        selected: list[QuoteResult] = []
        covered_terms: set[str] = set()
        covered_bigrams: set[str] = set()
        covered_trigrams: set[str] = set()
        remaining = ranked[:]

        while remaining and len(selected) < max_results:
            best_idx = 0
            best_key: tuple[float, ...] | None = None
            for idx, item in enumerate(remaining):
                new_trigrams = [phrase for phrase in item["matched_trigrams"] if phrase not in covered_trigrams]
                new_bigrams = [phrase for phrase in item["matched_bigrams"] if phrase not in covered_bigrams]
                new_terms = [token for token in item["matched_terms"] if token not in covered_terms]
                key = (
                    sum(1.0 / trigram_doc_counts[phrase] for phrase in new_trigrams),
                    len(new_trigrams),
                    sum(1.0 / bigram_doc_counts[phrase] for phrase in new_bigrams),
                    len(new_bigrams),
                    sum(1.0 / term_doc_counts[token] for token in new_terms),
                    len(new_terms),
                    sum(1.0 / trigram_doc_counts[phrase] for phrase in item["matched_trigrams"]),
                    len(item["matched_trigrams"]),
                    sum(1.0 / bigram_doc_counts[phrase] for phrase in item["matched_bigrams"]),
                    len(item["matched_bigrams"]),
                    sum(1.0 / term_doc_counts[token] for token in item["matched_terms"]),
                    len(item["matched_terms"]),
                    item["starts_with_user"],
                    -int(item["term_span"]),
                    *_base_quote_score(item["result"], query, mode),
                    -int(item["index"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            chosen = remaining.pop(best_idx)
            selected.append(chosen["result"])
            covered_terms.update(chosen["matched_terms"])
            covered_bigrams.update(chosen["matched_bigrams"])
            covered_trigrams.update(chosen["matched_trigrams"])

        return selected
    target_results = max_results
    if mode == "aggregate_total":
        target_results = max(max_results * 4, 20)

    ranked = sorted(
        results,
        key=lambda qr: _base_quote_score(qr, query, mode),
        reverse=True,
    )

    if mode == "exact_value":
        # Keep the full overfetched pool for value resolution. Exact-value
        # selection may need a semantically matched quote that would
        # otherwise be cut off by strong but misleading lexical hits.
        return ranked

    selected: list[QuoteResult] = []
    remaining = ranked[:]
    seen_segments: set[str] = set()
    seen_sessions: set[str] = set()
    seen_tags: set[str] = set()
    seen_components: set[str] = set()
    coverage_components = coverage_components or []
    requested_units = (
        set(_requested_quantity_units(query, query))
        if mode == "aggregate_total"
        else set()
    )

    if mode == "aggregate_total" and coverage_components:
        # Seed aggregate-total retrieval with the strongest component-specific
        # quantity-bearing result for each requested component. Otherwise
        # broad comparison rows can crowd out the per-component values the
        # reader needs to compute the total.
        for component in coverage_components:
            best_idx: int | None = None
            best_score: tuple[float, ...] | None = None
            for idx, qr in enumerate(remaining):
                matched_components = _match_coverage_components(
                    qr.text or "",
                    coverage_components,
                    tag=qr.tag or "",
                )
                if component not in matched_components:
                    continue
                quantities = _extract_quantity_mentions(qr.text or "", limit=6)
                if requested_units:
                    quantities = [
                        quantity
                        for quantity in quantities
                        if quantity.get("unit") in requested_units
                    ]
                if not quantities:
                    continue
                component_specific = len(matched_components) == 1 and matched_components[0] == component
                requested_quantity_count, distinct_requested_count, _other_unit_count, total_quantity_count = (
                    _aggregate_quantity_profile(quantities, requested_units)
                )
                score = (
                    1.0 if component_specific else 0.0,
                    1.0 if requested_quantity_count > 0 and distinct_requested_count == 1 else 0.0,
                    -float(max(0, distinct_requested_count - 1)),
                    -float(max(0, total_quantity_count - requested_quantity_count)),
                    -float(total_quantity_count),
                    float(_source_weight(qr.match_type, mode)),
                    *_base_quote_score(qr, query, mode),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                continue
            qr = remaining.pop(best_idx)
            selected.append(qr)
            if qr.segment_ref:
                seen_segments.add(qr.segment_ref)
            if qr.session_date:
                seen_sessions.add(qr.session_date)
            if qr.tag:
                seen_tags.add(qr.tag)
            seen_components.update(
                _match_coverage_components(qr.text or "", coverage_components, tag=qr.tag or "")
            )

    while remaining and len(selected) < target_results:
        best_idx = 0
        best_score: tuple[float, ...] | None = None
        for idx, qr in enumerate(remaining):
            base = _base_quote_score(qr, query, mode)
            matched_components = _match_coverage_components(
                qr.text or "",
                coverage_components,
                tag=qr.tag or "",
            )
            quantity_mentions = _extract_quantity_mentions(qr.text or "", limit=4)
            if requested_units:
                quantity_mentions = [
                    quantity
                    for quantity in quantity_mentions
                    if quantity.get("unit") in requested_units
                ]
            component_specific = (
                mode == "aggregate_total"
                and len(matched_components) == 1
                and bool(quantity_mentions)
            )
            novelty = (
                float(sum(1 for component in matched_components if component not in seen_components)),
                2.0 if component_specific else 0.0,
                1.5 if mode == "aggregate_total" and quantity_mentions else 0.0,
                2.0 if qr.segment_ref not in seen_segments else 0.0,
                1.5 if (qr.session_date or "") not in seen_sessions else 0.0,
                1.0 if qr.tag not in seen_tags else 0.0,
            )
            score = base + novelty
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        qr = remaining.pop(best_idx)
        selected.append(qr)
        seen_segments.add(qr.segment_ref)
        if qr.session_date:
            seen_sessions.add(qr.session_date)
        if qr.tag:
            seen_tags.add(qr.tag)
        seen_components.update(
            _match_coverage_components(qr.text or "", coverage_components, tag=qr.tag or "")
        )

    return selected


def _search_summary_candidates(
    store: ContextStore,
    semantic: SemanticSearchManager,
    query: str,
    *,
    limit: int,
    conversation_id: str | None,
    include_semantic: bool = True,
    include_descriptions: bool = True,
    include_tool_outputs: bool = True,
) -> list[QuoteResult]:
    results = store.search_full_text(query, limit=limit, conversation_id=conversation_id)

    if include_semantic:
        remaining = limit - len(results)
        if remaining > 0:
            semantic_results = semantic.semantic_search(
                query, max_results=remaining, conversation_id=conversation_id,
            )
            results.extend(semantic_results)

    if include_descriptions:
        results = supplement_from_descriptions(
            store, query, results, limit, conversation_id=conversation_id,
        )

    if include_tool_outputs:
        tool_remaining = limit - len(results)
        if tool_remaining > 0:
            tool_results = store.search_tool_outputs(
                query, limit=tool_remaining, conversation_id=conversation_id,
            )
            results.extend(tool_results)

    return results


def _hydrate_lookup_summary_results(
    store: ContextStore,
    results: list[QuoteResult],
    *,
    conversation_id: str | None,
) -> list[QuoteResult]:
    """Replace segment-hit excerpts with intact segment summaries.

    Summary lookup should return the stored summary text for the selected
    segment, not the best-matching full-text chunk from that segment.
    """
    hydrated: list[QuoteResult] = []
    for qr in results:
        if qr.source_scope == "turn" or not qr.segment_ref:
            hydrated.append(qr)
            continue
        seg = store.get_segment(qr.segment_ref, conversation_id=conversation_id)
        summary = getattr(seg, "summary", None)
        if not isinstance(summary, str) or not summary.strip():
            hydrated.append(qr)
            continue
        primary_tag = getattr(seg, "primary_tag", None)
        if not isinstance(primary_tag, str) or not primary_tag.strip():
            primary_tag = qr.tag
        seg_tags = getattr(seg, "tags", None)
        if not isinstance(seg_tags, list) or not all(isinstance(tag, str) for tag in seg_tags):
            seg_tags = list(qr.tags)
        metadata = getattr(seg, "metadata", None)
        session_date = getattr(metadata, "session_date", "") if metadata is not None else ""
        if not isinstance(session_date, str) or not session_date.strip():
            session_date = qr.session_date
        hydrated.append(
            QuoteResult(
                text=summary,
                tag=primary_tag,
                segment_ref=qr.segment_ref,
                tags=list(seg_tags or qr.tags),
                score=qr.score,
                match_type=qr.match_type,
                similarity=qr.similarity,
                session_date=session_date,
                created_at=qr.created_at,
                source_scope="segment",
            )
        )
    return hydrated


def _search_find_quote_candidates(
    store: ContextStore,
    semantic: SemanticSearchManager,
    query: str,
    *,
    limit: int,
    mode: str,
    conversation_id: str | None,
    include_tool_outputs: bool = True,
) -> list[QuoteResult]:
    results: list[QuoteResult] = []
    lexical_limit = limit
    semantic_limit = 0

    if mode == "exact_value" and limit > 1:
        semantic_limit = max(1, min(limit // 3, 8))
        lexical_limit = max(1, limit - semantic_limit)

    search_full_text = getattr(store, "search_canonical_full_text", None)
    if callable(search_full_text):
        results.extend(
            search_full_text(
                query,
                limit=lexical_limit,
                conversation_id=conversation_id,
            )
        )

    if mode != "exact_value":
        semantic_limit = max(0, limit - len(results))

    if semantic_limit > 0:
        results.extend(
            semantic.semantic_full_text_search(
                query,
                max_results=semantic_limit,
                conversation_id=conversation_id,
            )
        )

    deduped: list[QuoteResult] = []
    seen: set[tuple[object, ...]] = set()
    for qr in results:
        if qr.turn_number is not None:
            key = ("turn", qr.turn_number)
        elif qr.segment_ref:
            key = ("segment", qr.segment_ref)
        else:
            key = ("text", qr.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qr)

    return deduped


def _build_coverage_summary(results: list[dict], components: list[str]) -> dict[str, object]:
    covered_components: list[str] = []
    seen: set[str] = set()
    for row in results:
        for component in row.get("matched_components", []) or []:
            if component in components and component not in seen:
                seen.add(component)
                covered_components.append(component)
    missing_components = [component for component in components if component not in seen]
    return {
        "distinct_sessions": len({row.get("session", "") for row in results if row.get("session")}),
        "distinct_topics": len({row.get("topic", "") for row in results if row.get("topic")}),
        "distinct_results": len(results),
        "requested_components": components,
        "covered_components": covered_components,
        "missing_components": missing_components,
    }


def _build_coverage_value_candidates(results: list[dict], query: str) -> list[dict]:
    wants_per_second = "per second" in query.lower() or "queries per second" in query.lower()
    candidates: dict[str, dict[str, object]] = {}
    for row in results:
        excerpt = row.get("excerpt", "")
        matched_components = row.get("matched_components", []) or []
        for match in _QUERY_RATE_RE.finditer(excerpt):
            amount = match.group(1)
            unit = match.group(2).lower()
            normalized_unit = "second" if unit in {"sec", "second"} else "hour"
            key = f"{amount} queries/{normalized_unit}"
            entry = candidates.setdefault(
                key,
                {
                    "value": f"{amount} queries/{normalized_unit}",
                    "unit": normalized_unit,
                    "occurrences": 0,
                    "matched_components": [],
                },
            )
            entry["occurrences"] = int(entry["occurrences"]) + 1
            existing = set(entry["matched_components"])
            for component in matched_components:
                if component not in existing:
                    entry["matched_components"].append(component)
                    existing.add(component)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            1 if wants_per_second and item["unit"] == "second" else 0,
            len(item["matched_components"]),
            int(item["occurrences"]),
        ),
        reverse=True,
    )
    return ranked[:5]


def _build_shared_value_candidates(
    results: list[dict],
    exact_candidates: list[dict[str, object]],
    *,
    query: str,
    intent_context: str,
    chosen_candidate: dict[str, object] | None,
) -> list[dict[str, object]]:
    components = _extract_coverage_components(intent_context, query)
    seen_overall_components: set[str] = set()
    for row in results:
        excerpt = str(row.get("excerpt", ""))
        topic = str(row.get("topic", ""))
        for component in _match_coverage_components(excerpt, components, tag=topic):
            seen_overall_components.add(component)
    overall_covered_components = [
        component for component in components if component in seen_overall_components
    ]

    wants_per_second = "per second" in query.lower() or "queries per second" in query.lower()
    wants_per_hour = "per hour" in query.lower() or "queries per hour" in query.lower()
    chosen_excerpt = str(chosen_candidate.get("excerpt", "")).strip() if chosen_candidate else ""
    aggregated: dict[str, dict[str, object]] = {}
    for candidate in exact_candidates:
        excerpt = str(candidate.get("excerpt", "")).strip()
        if not excerpt:
            continue
        topic = str(candidate.get("topic", ""))
        term_hits = int(candidate.get("term_hits", 0) or 0)
        direct_components = _match_coverage_components(excerpt, components, tag=topic)
        for match in _QUERY_RATE_RE.finditer(excerpt):
            amount = match.group(1)
            unit = match.group(2).lower()
            normalized_unit = "second" if unit in {"sec", "second"} else "hour"
            key = f"{amount} queries/{normalized_unit}"
            entry = aggregated.setdefault(
                key,
                {
                    "value": key,
                    "unit": normalized_unit,
                    "occurrences": 0,
                    "_supporting_excerpts": set(),
                    "_direct_components": [],
                    "_direct_component_seen": set(),
                    "_term_hits": 0,
                    "_chosen_support": False,
                },
            )
            entry["occurrences"] = int(entry["occurrences"]) + 1
            entry["_supporting_excerpts"].add(excerpt)
            entry["_term_hits"] = max(int(entry["_term_hits"]), term_hits)
            if chosen_excerpt and excerpt == chosen_excerpt:
                entry["_chosen_support"] = True
            seen_direct = entry["_direct_component_seen"]
            for component in direct_components:
                if component in seen_direct:
                    continue
                seen_direct.add(component)
                entry["_direct_components"].append(component)

    if not aggregated:
        return []

    def _unit_preference(unit: str) -> int:
        if wants_per_second:
            return 1 if unit == "second" else 0
        if wants_per_hour:
            return 1 if unit == "hour" else 0
        return 0

    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            _unit_preference(str(item.get("unit", ""))),
            1 if item.get("_chosen_support") else 0,
            len(item.get("_direct_components", [])),
            len(item.get("_supporting_excerpts", [])),
            int(item.get("occurrences", 0)),
            int(item.get("_term_hits", 0)),
        ),
        reverse=True,
    )

    public_candidates: list[dict[str, object]] = []
    for item in ranked:
        if int(item.get("occurrences", 0)) < 2:
            continue
        matched_components = list(item.get("_direct_components", []))
        # When the top repeated rate is backed by the chosen quote and the
        # broader retrieved quote set covers the full component framing,
        # surface that broader framing to keep the reader grounded in the
        # multi-part question without switching tools.
        if item.get("_chosen_support") and overall_covered_components:
            matched_components = list(overall_covered_components)
        public_candidates.append(
            {
                "value": item["value"],
                "unit": item["unit"],
                "occurrences": item["occurrences"],
                "matched_components": matched_components,
                "reason": (
                    "Repeated throughput target across quote evidence for the named components."
                    if matched_components
                    else "Repeated throughput target across quote evidence."
                ),
            }
        )
        if len(public_candidates) >= 3:
            break
    return public_candidates


def _component_spans(text: str, component: str) -> list[tuple[int, int]]:
    tokens = re.findall(r"[a-z0-9]+", component.lower())
    if not tokens:
        return []
    pattern = r"\b" + r"\W+".join(re.escape(token) for token in tokens) + r"\b"
    return [match.span() for match in re.finditer(pattern, (text or "").lower())]


def _aggregate_component_anchor_metadata(
    text: str,
    component: str,
    quantity: dict[str, object],
) -> dict[str, float]:
    spans = _component_spans(text, component)
    component_position = spans[0][0] if spans else 99999
    quantity_position = int(quantity.get("start", 99999))
    proximity = abs(quantity_position - component_position)
    component_lead_bonus = (
        2.0 if component_position <= 40 else 1.0 if component_position <= 80 else 0.0
    )
    quantity_lead_bonus = (
        1.0 if quantity_position <= 120 else 0.5 if quantity_position <= 220 else 0.0
    )
    proximity_bonus = max(0.0, 1.5 - (float(proximity) / 80.0))
    return {
        "component_lead_position": float(component_position),
        "quantity_position": float(quantity_position),
        "component_anchor_strength": (
            component_lead_bonus + quantity_lead_bonus + proximity_bonus
        ),
    }


def _assign_quantities_to_components(
    text: str,
    matched_components: list[str],
    quantities: list[dict[str, object]],
) -> list[tuple[str, dict[str, object]]]:
    if not matched_components or not quantities:
        return []

    spans_by_component = {
        component: _component_spans(text, component) for component in matched_components
    }
    assignments: list[tuple[str, dict[str, object]]] = []
    for quantity in quantities:
        start = quantity.get("start")
        end = quantity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        best_component = ""
        best_distance: int | None = None
        best_before_component = ""
        best_before_distance: int | None = None
        best_after_component = ""
        best_after_distance: int | None = None
        for component, spans in spans_by_component.items():
            for comp_start, comp_end in spans:
                if comp_end <= start:
                    distance = start - comp_end
                    if best_before_distance is None or distance < best_before_distance:
                        best_before_distance = distance
                        best_before_component = component
                elif comp_start >= end:
                    distance = comp_start - end
                    if best_after_distance is None or distance < best_after_distance:
                        best_after_distance = distance
                        best_after_component = component
        if best_before_component:
            best_component = best_before_component
            best_distance = best_before_distance
        elif best_after_component:
            best_component = best_after_component
            best_distance = best_after_distance
        if best_component and best_distance is not None and best_distance <= _AGGREGATE_COMPONENT_DISTANCE_CHARS:
            assignments.append((best_component, quantity))
    return assignments


def _aggregate_candidate_metrics(
    combo: list[dict[str, object]],
    components: list[str],
    unit: str,
    requested_units: set[str],
) -> dict[str, object]:
    parsed_dates: list[date] = []
    for item in combo:
        raw = item.get("session_date_normalized", "")
        if isinstance(raw, str):
            parsed = _parse_session_date(raw)
            if parsed is not None:
                parsed_dates.append(parsed)
    if len(parsed_dates) >= 2:
        spread_days = (max(parsed_dates) - min(parsed_dates)).days
    else:
        spread_days = 9999
    component_specific_count = float(sum(1 for item in combo if item.get("component_specific")))
    anchor_strength_total = float(sum(
        float(item.get("component_anchor_strength", 0.0))
        for item in combo
    ))
    source_weight_total = float(sum(
        _source_weight(str(item.get("match_type", "")), "aggregate_total")
        for item in combo
    ))
    term_hits_total = float(sum(float(item.get("term_hits", 0.0)) for item in combo))
    requested_unit_match = 1.0 if unit in requested_units else 0.0
    return {
        "spread_days": spread_days,
        "component_specific_count": component_specific_count,
        "anchor_strength_total": anchor_strength_total,
        "source_weight_total": source_weight_total,
        "term_hits_total": term_hits_total,
        "requested_unit_match": requested_unit_match,
        "full_coverage": float(len(combo) == len(components)),
        "score": (
            float(len(combo) == len(components)),
            requested_unit_match,
            component_specific_count,
            anchor_strength_total,
            source_weight_total,
            term_hits_total,
            float(-spread_days),
        ),
    }


def _public_aggregate_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in candidate.items()
        if not key.startswith("_")
    }


def _aggregate_totals_materially_different(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    left_total = float(left.get("_normalized_total_value", 0.0))
    right_total = float(right.get("_normalized_total_value", 0.0))
    scale = max(abs(left_total), abs(right_total), 1.0)
    return abs(left_total - right_total) / scale > _AGGREGATE_AMBIGUITY_RELATIVE_DELTA


def _aggregate_candidates_have_similar_support(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    left_metrics = left.get("_metrics")
    right_metrics = right.get("_metrics")
    if not isinstance(left_metrics, dict) or not isinstance(right_metrics, dict):
        return False
    if left.get("unit") != right.get("unit"):
        return False
    if set(left.get("covered_components", [])) != set(right.get("covered_components", [])):
        return False
    if left_metrics.get("component_specific_count") != right_metrics.get("component_specific_count"):
        return False
    if abs(
        float(left_metrics.get("anchor_strength_total", 0.0))
        - float(right_metrics.get("anchor_strength_total", 0.0))
    ) > _AGGREGATE_AMBIGUITY_ANCHOR_GAP:
        return False
    if abs(
        float(left_metrics.get("source_weight_total", 0.0))
        - float(right_metrics.get("source_weight_total", 0.0))
    ) > _AGGREGATE_AMBIGUITY_SOURCE_GAP:
        return False
    if abs(
        float(left_metrics.get("term_hits_total", 0.0))
        - float(right_metrics.get("term_hits_total", 0.0))
    ) > _AGGREGATE_AMBIGUITY_TERM_HITS_GAP:
        return False
    if abs(
        int(left_metrics.get("spread_days", 9999))
        - int(right_metrics.get("spread_days", 9999))
    ) > _AGGREGATE_AMBIGUITY_SPREAD_DAYS_GAP:
        return False
    return True


def _aggregate_component_sort_key(item: dict[str, object]) -> tuple[float, ...]:
    return (
        1.0 if item.get("component_specific") else 0.0,
        1.0 if item.get("requested_quantity_count", 0) and item.get("distinct_requested_value_count", 0) == 1 else 0.0,
        float(item.get("component_anchor_strength", 0.0)),
        float(_source_weight(str(item.get("match_type", "")), "aggregate_total")),
        float(item.get("term_hits", 0.0)),
        -float(item.get("other_unit_count", 0.0)),
        -float(max(0, int(item.get("distinct_requested_value_count", 0)) - 1)),
        -float(item.get("total_quantity_count", 0.0)),
        -float(item.get("component_lead_position", 99999.0)),
        -float(item.get("quantity_position", 99999.0)),
    )


def _aggregate_component_identity(item: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(item.get("component", "")),
        str(item.get("segment_ref", "")),
        str(item.get("value", "")),
    )


def _aggregate_candidate_identity(candidate: dict[str, object]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        _aggregate_component_identity(item)
        for item in candidate.get("component_values", [])
    )


def _build_aggregate_competing_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not candidates:
        return []

    full_coverage = [candidate for candidate in candidates if not candidate.get("missing_components")]
    pool = full_coverage or candidates
    components = list(pool[0].get("covered_components", []))
    if len(components) < 2:
        return pool[:3]

    component_buckets: dict[str, list[dict[str, object]]] = {component: [] for component in components}
    for candidate in pool:
        for item in candidate.get("component_values", []):
            component = str(item.get("component", ""))
            if component in component_buckets:
                component_buckets[component].append(item)

    top_component_values: dict[str, list[dict[str, object]]] = {}
    for component, items in component_buckets.items():
        ranked = sorted(items, key=_aggregate_component_sort_key, reverse=True)
        unique: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in ranked:
            identity = _aggregate_component_identity(item)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
            if len(unique) >= 2:
                break
        if unique:
            top_component_values[component] = unique

    if len(top_component_values) != len(components):
        return pool[:3]

    candidate_by_identity = {
        _aggregate_candidate_identity(candidate): candidate
        for candidate in pool
    }
    selected: list[dict[str, object]] = []
    seen_candidates: set[tuple[tuple[str, str, str], ...]] = set()
    per_component_lists = [top_component_values[component] for component in components]
    for combo in product(*per_component_lists):
        identity = tuple(_aggregate_component_identity(item) for item in combo)
        candidate = candidate_by_identity.get(identity)
        if candidate is None or identity in seen_candidates:
            continue
        seen_candidates.add(identity)
        selected.append(candidate)

    if len(selected) < 2:
        return pool[:3]

    selected.sort(
        key=lambda item: (
            item.get("_score", (0.0,)),
            len(item.get("covered_components", [])),
            -len(item.get("missing_components", [])),
        ),
        reverse=True,
    )
    return selected[:4]


def _resolve_aggregate_total_candidates(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    if not candidates:
        return {}

    full_coverage = [c for c in candidates if not c.get("missing_components")]
    pool = full_coverage or candidates
    top = pool[0]
    competing = [top]
    for candidate in pool[1:]:
        if not _aggregate_totals_materially_different(top, candidate):
            continue
        if _aggregate_candidates_have_similar_support(top, candidate):
            competing.append(candidate)
    if len(competing) > 1:
        packaged_candidates = _build_aggregate_competing_candidates(pool)
        return {
            "ambiguity_detected": True,
            "ambiguity_reason": (
                "multiple full-coverage totals remain plausible with similar support"
            ),
            "competing_aggregate_totals": [
                _public_aggregate_candidate(candidate)
                for candidate in (packaged_candidates or competing[:3])
            ],
        }
    return {
        "chosen_aggregate_total": _public_aggregate_candidate(top),
    }


def _build_aggregate_total_candidates(
    results: list[dict],
    components: list[str],
    *,
    query: str = "",
    intent_context: str = "",
) -> list[dict]:
    requested_units = set(_requested_quantity_units(query, intent_context))
    signal_terms = _query_terms(f"{intent_context} {query}")
    component_buckets: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in results:
        matched_components = row.get("matched_components", []) or []
        quantities = row.get("quantities", []) or _extract_quantity_mentions(row.get("excerpt", ""))
        if not matched_components or not quantities:
            continue
        requested_quantity_count, distinct_requested_count, other_unit_count, total_quantity_count = (
            _aggregate_quantity_profile(quantities, requested_units)
        )
        assignments = _assign_quantities_to_components(
            row.get("excerpt", ""),
            matched_components,
            quantities,
        )
        if not assignments:
            continue
        row_text = (row.get("excerpt", "") or "").lower()
        term_hits = sum(
            1
            for term in signal_terms
            if re.search(rf"\b{re.escape(term)}\b", row_text)
        )
        for component, quantity in assignments:
            unit = quantity.get("unit")
            normalized_value = quantity.get("normalized_value")
            if not isinstance(unit, str) or not unit or not isinstance(normalized_value, (int, float)):
                continue
            if requested_units and unit not in requested_units:
                continue
            bucket = component_buckets.setdefault(unit, {})
            anchor_metadata = _aggregate_component_anchor_metadata(
                row.get("excerpt", ""),
                component,
                quantity,
            )
            bucket.setdefault(component, []).append({
                "component": component,
                "value": quantity.get("value", ""),
                "normalized_value": float(normalized_value),
                "session": row.get("session", ""),
                "session_date_normalized": row.get("session_date_normalized", ""),
                "topic": row.get("topic", ""),
                "match_type": row.get("match_type", ""),
                "segment_ref": row.get("segment_ref", ""),
                "component_specific": len(matched_components) == 1,
                "term_hits": term_hits,
                "requested_quantity_count": requested_quantity_count,
                "distinct_requested_value_count": distinct_requested_count,
                "other_unit_count": other_unit_count,
                "total_quantity_count": total_quantity_count,
                **anchor_metadata,
            })

    candidates: list[dict[str, object]] = []
    for unit, component_map in component_buckets.items():
        covered_components = [component for component in components if component_map.get(component)]
        if len(covered_components) < 2:
            continue
        per_component_lists: list[list[dict[str, object]]] = []
        for component in covered_components:
            ranked_entries = sorted(
                component_map[component],
                key=lambda item: (
                    1.0 if item.get("component_specific") else 0.0,
                    1.0 if item.get("requested_quantity_count", 0) and item.get("distinct_requested_value_count", 0) == 1 else 0.0,
                    float(item.get("component_anchor_strength", 0.0)),
                    float(_source_weight(str(item.get("match_type", "")), "aggregate_total")),
                    float(item.get("term_hits", 0.0)),
                    -float(item.get("other_unit_count", 0.0)),
                    -float(max(0, int(item.get("distinct_requested_value_count", 0)) - 1)),
                    -float(item.get("total_quantity_count", 0.0)),
                    -float(item.get("component_lead_position", 99999.0)),
                    -float(item.get("quantity_position", 99999.0)),
                ),
                reverse=True,
            )
            per_component_lists.append(ranked_entries[:_AGGREGATE_COMPONENT_CANDIDATES_LIMIT])
        for combo in product(*per_component_lists):
            combo_list = list(combo)
            total_value = sum(float(item["normalized_value"]) for item in combo_list)
            metrics = _aggregate_candidate_metrics(
                combo_list,
                components,
                unit,
                requested_units,
            )
            candidates.append({
                "value": _format_aggregate_value(total_value, unit),
                "unit": unit,
                "covered_components": covered_components,
                "missing_components": [component for component in components if component not in covered_components],
                "component_values": combo_list,
                "_normalized_total_value": total_value,
                "_metrics": metrics,
                "_score": metrics["score"],
            })

    candidates.sort(
        key=lambda item: (
            item.get("_score", (0.0,)),
            len(item["covered_components"]),
            -len(item["missing_components"]),
        ),
        reverse=True,
    )
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for candidate in candidates:
        key = (
            candidate["value"],
            tuple(
                (item["component"], item["value"])
                for item in candidate["component_values"]
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= 5:
            break
    return deduped


def _apply_aggregate_total_metadata(
    response: dict[str, object],
    formatted: list[dict[str, object]],
    coverage_components: list[str],
    *,
    query: str,
    intent_context: str,
) -> None:
    response["coverage_summary"] = _build_coverage_summary(formatted, coverage_components)
    raw_candidates = _build_aggregate_total_candidates(
        formatted,
        coverage_components,
        query=query,
        intent_context=intent_context,
    )
    response["aggregate_total_candidates"] = [
        _public_aggregate_candidate(candidate)
        for candidate in raw_candidates
    ]
    resolution = _resolve_aggregate_total_candidates(raw_candidates)
    response.update(resolution)
    missing_components = response["coverage_summary"].get("missing_components", [])
    if resolution.get("ambiguity_detected") is True:
        response["reader_hint"] = (
            "AGGREGATE-TOTAL MODE: Multiple coherent totals remain with similar "
            "support. Do not choose one confidently. Use competing_aggregate_totals "
            "to explain the ambiguity, and treat each total as a supported "
            "possibility derived from component-specific evidence rather than a "
            "confirmed answer. Include all returned totals when summarizing the "
            "ambiguity, ask a follow-up that identifies the intended "
            "project/date/version, or say the memory is ambiguous. "
            + (
                f"Coverage is still missing for: {', '.join(missing_components)}. "
                if isinstance(missing_components, list) and missing_components
                else ""
            )
        )
        return
    response["reader_hint"] = (
        "AGGREGATE-TOTAL MODE: Use component-specific values to compute a total "
        "across the named components. Prefer aggregate_total_candidates that "
        "cover all requested components. Do not require one excerpt to state the "
        "final total verbatim. Only sum values when they are component-specific "
        "and use the same unit. "
        + (
            f"Coverage is still missing for: {', '.join(missing_components)}. "
            if isinstance(missing_components, list) and missing_components
            else ""
        )
        + "If a component is missing, search again with a narrower follow-up query."
    )


def _build_value_candidates(results: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in results:
        excerpt = item.get("excerpt", "")
        values = _extract_value_tokens(excerpt)
        if not values:
            continue
        key = (item.get("session", ""), tuple(values))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "values": values,
            "topic": item.get("topic", ""),
            "session": item.get("session", ""),
            "session_date_normalized": item.get("session_date_normalized", ""),
            "excerpt": excerpt,
        })
        if len(candidates) >= 5:
            break
    return candidates


def _build_exact_value_candidates(
    results: list[dict],
    *,
    query: str,
    intent_context: str,
) -> list[dict[str, object]]:
    requested_units = set(_requested_quantity_units(query, intent_context))
    wants_percentage = _wants_percentage_value(query, intent_context)
    wants_version = _wants_version_value(query, intent_context)
    first_person_question = _question_is_first_person(query, intent_context)
    signal_terms = _query_terms(f"{intent_context} {query}")
    candidates: list[dict[str, object]] = []
    for row in results:
        excerpt = row.get("excerpt", "") or ""
        versions = _extract_versions(excerpt)
        distinct_version_count = len(set(versions))
        user_statement = _excerpt_starts_with_user_statement(excerpt)
        percentages = _extract_percentages(excerpt)
        quantities = _extract_quantity_mentions(excerpt, limit=6)
        if requested_units:
            requested_quantities = [
                quantity for quantity in quantities
                if quantity.get("unit") in requested_units
            ]
        else:
            requested_quantities = list(quantities)
        requested_quantity_count, distinct_requested_count, other_unit_count, total_quantity_count = (
            _aggregate_quantity_profile(quantities, requested_units)
        )
        row_text = excerpt.lower()
        term_hits = sum(
            1
            for term in signal_terms
            if re.search(rf"\b{re.escape(term)}\b", row_text)
        )
        values: list[str] = []
        if wants_version:
            values.extend(versions[:2])
        values.extend(percentages[:2])
        for quantity in requested_quantities[:3]:
            value = str(quantity.get("value", "")).strip()
            if value and value not in values:
                values.append(value)
        if wants_version:
            score = (
                1.0 if bool(versions) else 0.0,
                1.0 if (first_person_question and user_statement) else 0.0,
                1.0 if distinct_version_count == 1 and bool(versions) else 0.0,
                -float(max(0, distinct_version_count - 1)),
                float(term_hits),
                float(_source_weight(str(row.get("match_type", "")), "exact_value")),
                1.0 if (not requested_units or requested_quantity_count > 0) else 0.0,
                -float(other_unit_count),
                -float(max(0, total_quantity_count - requested_quantity_count)),
            )
        else:
            score = (
                1.0 if (not wants_percentage or bool(percentages)) else 0.0,
                float(requested_quantity_count),
                1.0 if ((not wants_percentage or bool(percentages)) and (not requested_units or requested_quantity_count > 0)) else 0.0,
                float(term_hits),
                float(_source_weight(str(row.get("match_type", "")), "exact_value")),
                -float(other_unit_count),
                -float(max(0, total_quantity_count - requested_quantity_count)),
            )
        candidates.append({
            "values": values or _extract_value_tokens(excerpt)[:4],
            "version_values": versions[:3],
            "percentages": percentages[:3],
            "quantity_values": requested_quantities[:3],
            "topic": row.get("topic", ""),
            "session": row.get("session", ""),
            "session_date_normalized": row.get("session_date_normalized", ""),
            "turn_number": row.get("turn_number"),
            "created_at": row.get("created_at", ""),
            "excerpt": excerpt,
            "distinct_version_count": distinct_version_count,
            "user_statement": user_statement,
            "requested_quantity_count": requested_quantity_count,
            "distinct_requested_value_count": distinct_requested_count,
            "term_hits": term_hits,
            "_score": score,
        })
    candidates.sort(key=lambda item: item["_score"], reverse=True)
    deduped: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, ...], str, str]] = set()
    for candidate in candidates:
        key = (
            tuple(candidate.get("values", [])),
            str(candidate.get("topic", "")),
            str(candidate.get("session_date_normalized", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= 5:
            break
    return deduped


def _public_exact_value_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in candidate.items()
        if not key.startswith("_")
    }


def _resolve_exact_value_candidates(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    if not candidates:
        return {}
    top = candidates[0]
    if len(candidates) == 1:
        return {"chosen_exact_value_candidate": _public_exact_value_candidate(top)}

    second = candidates[1]
    top_score = tuple(top.get("_score", ()))
    second_score = tuple(second.get("_score", ()))
    if top_score[:4] > second_score[:4]:
        return {"chosen_exact_value_candidate": _public_exact_value_candidate(top)}

    latest_version_candidate = _latest_user_version_candidate(candidates)
    if latest_version_candidate is not None:
        return {
            "chosen_exact_value_candidate": _public_exact_value_candidate(
                latest_version_candidate
            )
        }

    top_values = tuple(str(v) for v in top.get("values", []))
    second_values = tuple(str(v) for v in second.get("values", []))
    if top_values != second_values:
        return {
            "conflicting_exact_value_candidates": [
                _public_exact_value_candidate(top),
                _public_exact_value_candidate(second),
            ],
        }
    return {"chosen_exact_value_candidate": _public_exact_value_candidate(top)}


def _latest_user_version_candidate(
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    version_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("version_values")
        and candidate.get("user_statement")
        and candidate.get("session_date_normalized")
    ]
    if len(version_candidates) < 2:
        return None
    return max(
        version_candidates,
        key=lambda candidate: (
            str(candidate.get("session_date_normalized", "")),
            str(candidate.get("created_at", "")),
            tuple(candidate.get("_score", ())),
        ),
    )


def _filter_exact_value_results(
    formatted: list[dict[str, object]],
    chosen_candidate: dict[str, object] | None,
    conflicting_candidates: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    excerpts: list[str] = []
    if chosen_candidate:
        excerpt = str(chosen_candidate.get("excerpt", "")).strip()
        if excerpt:
            excerpts.append(excerpt)
    elif conflicting_candidates:
        for candidate in conflicting_candidates:
            excerpt = str(candidate.get("excerpt", "")).strip()
            if excerpt and excerpt not in excerpts:
                excerpts.append(excerpt)

    if not excerpts:
        return formatted

    prioritized: list[dict[str, object]] = []
    remainder = list(formatted)
    for excerpt in excerpts:
        for idx, row in enumerate(remainder):
            if str(row.get("excerpt", "")).strip() == excerpt:
                prioritized.append(remainder.pop(idx))
                break
    return prioritized + remainder


def _apply_exact_value_metadata(
    response: dict[str, object],
    formatted: list[dict[str, object]],
    *,
    query: str,
    intent_context: str,
) -> None:
    response["value_candidates"] = _build_value_candidates(formatted)
    exact_candidates = _build_exact_value_candidates(
        formatted,
        query=query,
        intent_context=intent_context,
    )
    if exact_candidates:
        response["exact_value_candidates"] = [
            _public_exact_value_candidate(candidate)
            for candidate in exact_candidates
        ]
    resolution = _resolve_exact_value_candidates(exact_candidates)
    response.update(resolution)
    shared_value_candidates = _build_shared_value_candidates(
        formatted,
        exact_candidates,
        query=query,
        intent_context=intent_context,
        chosen_candidate=(
            response.get("chosen_exact_value_candidate")
            if isinstance(response.get("chosen_exact_value_candidate"), dict)
            else None
        ),
    )
    if shared_value_candidates:
        response["shared_value_candidates"] = shared_value_candidates
    if response.get("chosen_exact_value_candidate"):
        chosen_candidate = response.get("chosen_exact_value_candidate")
        response["results"] = _filter_exact_value_results(
            formatted,
            chosen_candidate if isinstance(chosen_candidate, dict) else None,
            None,
        )[:1]
        response["value_candidates"] = [
            _public_exact_value_candidate(chosen_candidate)
        ] if isinstance(chosen_candidate, dict) else response["value_candidates"]
        response.pop("exact_value_candidates", None)
        response["answer_ready"] = True
        if shared_value_candidates:
            response["reader_hint"] = (
                "EXACT-VALUE MODE: chosen_exact_value_candidate is useful "
                "supporting evidence. If shared_value_candidates is present, "
                "prefer the top shared candidate when it matches the requested "
                "unit and component framing. Use raw excerpts as support; do "
                "not convert unrelated hourly values into a new combined answer "
                "unless the excerpts explicitly instruct summation."
            )
        else:
            response["reader_hint"] = (
                "EXACT-VALUE MODE: chosen_exact_value_candidate is the "
                "authoritative answer candidate. Answer using only its explicit "
                "values. Treat the remaining raw excerpt as supporting evidence "
                "only, and ignore any competing values unless they exactly restate "
                "the same answer."
            )
        return
    if response.get("conflicting_exact_value_candidates"):
        conflicting = response.get("conflicting_exact_value_candidates")
        response["results"] = _filter_exact_value_results(
            formatted,
            None,
            conflicting if isinstance(conflicting, list) else None,
        )[:2]
        response["value_candidates"] = (
            conflicting if isinstance(conflicting, list) else response["value_candidates"]
        )
        response["answer_ready"] = False
        response["reader_hint"] = (
            "EXACT-VALUE MODE: Multiple explicit value candidates conflict. "
            "Do not answer from this call. Call vc_find_quote again with a "
            "narrower query before answering, and rely only on "
            "conflicting_exact_value_candidates when describing the conflict."
        )
        return
    response["answer_ready"] = False
    response["reader_hint"] = (
        "EXACT-VALUE MODE: Prefer explicit values shown in the excerpts. "
        "If multiple value candidates conflict, call vc_find_quote again "
        "with a narrower query before answering."
    )


# ---------------------------------------------------------------------------
# Span-union helpers — merge overlapping excerpts from the same segment
# ---------------------------------------------------------------------------


def _union_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _locate_excerpt(full_text: str, excerpt: str) -> tuple[int, int] | None:
    """Find an excerpt's (start, end) char position in full_text.

    Strips ellipsis bookends and FTS5 highlight markers (>>> / <<<)
    before searching.  Returns None if the cleaned text can't be found.
    """
    clean = excerpt
    # Strip FTS5 highlight markers
    clean = clean.replace(">>>", "").replace("<<<", "")
    # Strip leading/trailing ellipsis added by extract_excerpt
    if clean.startswith("..."):
        clean = clean[3:]
    if clean.endswith("..."):
        clean = clean[:-3]
    clean = clean.strip()
    if not clean:
        return None

    idx = full_text.find(clean)
    if idx >= 0:
        return (idx, idx + len(clean))

    # Case-insensitive fallback
    idx = full_text.lower().find(clean.lower())
    if idx >= 0:
        return (idx, idx + len(clean))

    return None


def _merge_segment_excerpts(
    store: ContextStore,
    results: list[QuoteResult],
    conversation_id: str | None = None,
) -> list[QuoteResult]:
    """Merge overlapping excerpts from the same segment_ref via span-union.

    Groups results by segment_ref.  For segments with multiple hits,
    locates each excerpt in the segment's full_text, computes the
    interval union, and re-extracts the merged (non-overlapping) text.
    Single-hit segments pass through unchanged.
    """
    # Group by segment_ref, preserving insertion order
    by_ref: OrderedDict[str, list[QuoteResult]] = OrderedDict()
    for qr in results:
        by_ref.setdefault(qr.segment_ref, []).append(qr)

    merged: list[QuoteResult] = []
    for ref, group in by_ref.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Multiple excerpts from the same segment — try span-union
        seg = store.get_segment(ref, conversation_id=conversation_id)
        if not seg or not seg.full_text:
            # Can't locate spans — keep first result only
            merged.append(group[0])
            continue

        full_text = seg.full_text
        spans: list[tuple[int, int]] = []
        unlocatable: list[QuoteResult] = []

        for qr in group:
            span = _locate_excerpt(full_text, qr.text)
            if span is not None:
                spans.append(span)
            else:
                unlocatable.append(qr)

        if not spans:
            # None located — keep first result
            merged.append(group[0])
            continue

        # Compute span union
        united = _union_spans(spans)

        # Extract text for each merged span
        parts: list[str] = []
        for start, end in united:
            part = full_text[start:end]
            if start > 0:
                part = "..." + part
            if end < len(full_text):
                part = part + "..."
            parts.append(part)

        # Combine metadata from all results in this group
        all_tags: list[str] = []
        for qr in group:
            for t in (qr.tags or []):
                if t not in all_tags:
                    all_tags.append(t)

        best = group[0]
        best_similarity = max(qr.similarity for qr in group)

        merged.append(QuoteResult(
            text="\n---\n".join(parts),
            tag=best.tag,
            segment_ref=ref,
            tags=all_tags,
            match_type=best.match_type,
            similarity=best_similarity,
            session_date=best.session_date,
            created_at=best.created_at,
        ))

        # Append any excerpts we couldn't locate (don't lose data)
        merged.extend(unlocatable)

    return merged


def supplement_from_descriptions(
    store: ContextStore,
    query: str,
    results: list[QuoteResult],
    max_results: int,
    conversation_id: str | None = None,
) -> list[QuoteResult]:
    """Add results from tags whose descriptions match query terms.

    Scans all tag descriptions for query words.  For tags that match
    but aren't already represented in *results*, fetches their
    segments and searches full_text for the best excerpt.  This
    bridges the vocabulary gap: the compacted description may use
    words that don't appear in FTS or embedding results.
    """
    from ..storage.helpers import extract_excerpt

    # Tags already covered by existing results
    covered_tags: set[str] = set()
    for qr in results:
        covered_tags.add(qr.tag)

    # Tokenize query into meaningful words (3+ chars)
    query_words = [
        w.lower()
        for w in re.findall(r"[a-zA-Z']+", query)
        if len(w) >= 3
    ]
    if not query_words:
        return results

    # Precompile word-boundary patterns to avoid substring false positives
    # (e.g. "old" matching "gold").
    word_patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in query_words]

    # Score each tag description by how many query words it contains
    all_summaries = store.get_all_tag_summaries(conversation_id=conversation_id)
    candidates: list[tuple[str, int, str]] = []  # (tag, score, description)
    for ts in all_summaries:
        if ts.tag in covered_tags:
            continue
        desc_lower = ts.description.lower() if ts.description else ""
        if not desc_lower:
            continue
        score = sum(1 for p in word_patterns if p.search(desc_lower))
        if score > 0:
            candidates.append((ts.tag, score, ts.description))

    # Sort by score descending, take top candidates
    candidates.sort(key=lambda x: -x[1])
    slots = max_results - len(results)
    if slots <= 0:
        return results

    for tag, _score, _desc in candidates[:slots]:
        # Fetch segments for this tag and search their full_text
        segments = store.get_segments_by_tags([tag], limit=5, conversation_id=conversation_id)
        best: QuoteResult | None = None
        best_score = 0
        for seg in segments:
            if not seg.full_text:
                continue
            text_lower = seg.full_text.lower()
            seg_score = sum(1 for p in word_patterns if p.search(text_lower))
            if seg_score > best_score:
                best_score = seg_score
                _sc = getattr(store, 'search_config', None)
                _ctx = _sc.excerpt_context_chars if _sc else 200
                excerpt = extract_excerpt(
                    seg.full_text, query, context_chars=_ctx
                )
                meta = seg.metadata or SegmentMetadata(turn_count=0)
                best = QuoteResult(
                    text=excerpt,
                    tag=seg.primary_tag,
                    segment_ref=seg.ref,
                    tags=seg.tags,
                    match_type="description",
                    session_date=meta.session_date,
                )
        if best is not None:
            results.append(best)
            covered_tags.add(tag)

    return results


def _supplement_aggregate_total_from_summaries(
    store: ContextStore,
    query: str,
    intent_context: str,
    results: list[QuoteResult],
    components: list[str],
    *,
    conversation_id: str | None = None,
) -> list[QuoteResult]:
    if not components:
        return results

    requested_units = set(_requested_quantity_units(query, intent_context))
    signal_terms = _query_terms(f"{intent_context} {query}")
    existing_quantity_refs: set[str] = set()
    for qr in results:
        if not qr.segment_ref:
            continue
        quantities = _extract_quantity_mentions(qr.text, limit=4)
        if requested_units:
            quantities = [q for q in quantities if q.get("unit") in requested_units]
        if quantities:
            existing_quantity_refs.add(qr.segment_ref)
    candidates: list[tuple[tuple[float, ...], QuoteResult]] = []

    try:
        # Aggregate totals often need older component snapshots, so scan the
        # full conversation's summaries instead of only the most recent slice.
        segments = store.get_all_segments(conversation_id=conversation_id)
    except Exception:
        return results

    for seg in segments:
        if seg.ref in existing_quantity_refs or not seg.summary:
            continue
        matched_components = _match_coverage_components(
            seg.summary,
            components,
            tag=seg.primary_tag,
        )
        if not matched_components:
            continue
        all_quantities = _extract_quantity_mentions(seg.summary, limit=6)
        if not all_quantities:
            continue
        requested_quantity_count, distinct_requested_count, other_unit_count, total_quantity_count = (
            _aggregate_quantity_profile(all_quantities, requested_units)
        )
        if requested_units and requested_quantity_count == 0:
            continue
        quantities = all_quantities
        if requested_units and not any(
            quantity.get("unit") in requested_units for quantity in quantities
        ):
            continue
        requested_quantities = [
            quantity for quantity in quantities
            if not requested_units or quantity.get("unit") in requested_units
        ]
        anchor_strength = 0.0
        component_lead_position = 99999.0
        quantity_position = 99999.0
        if len(matched_components) == 1 and requested_quantities:
            anchor_metadata = _aggregate_component_anchor_metadata(
                seg.summary,
                matched_components[0],
                requested_quantities[0],
            )
            anchor_strength = float(anchor_metadata["component_anchor_strength"])
            component_lead_position = float(anchor_metadata["component_lead_position"])
            quantity_position = float(anchor_metadata["quantity_position"])

        summary_lower = seg.summary.lower()
        term_hits = sum(
            1
            for term in signal_terms
            if re.search(rf"\b{re.escape(term)}\b", summary_lower)
        )
        requested_unit_hits = sum(
            1 for quantity in quantities if quantity.get("unit") in requested_units
        )
        score = (
            1.0 if len(matched_components) == 1 else 0.0,
            1.0 if requested_quantity_count > 0 and distinct_requested_count == 1 else 0.0,
            anchor_strength,
            float(term_hits),
            -float(other_unit_count),
            -float(max(0, distinct_requested_count - 1)),
            -float(total_quantity_count),
            float(requested_unit_hits),
            -component_lead_position,
            -quantity_position,
        )
        candidates.append((
            score,
            QuoteResult(
                text=seg.summary,
                tag=seg.primary_tag,
                segment_ref=seg.ref,
                tags=seg.tags,
                match_type="summary_scan",
                session_date=seg.metadata.session_date if seg.metadata else "",
                created_at=str(seg.created_at) if seg.created_at else "",
            ),
        ))

    if not candidates:
        return results

    candidates.sort(key=lambda item: item[0], reverse=True)
    augmented = list(results)
    target_additions = max(_AGGREGATE_SUMMARY_TARGET_ADDITIONS, len(components) * 10)
    added_refs: set[str] = set()
    for component in components:
        component_added = 0
        for _score, candidate in candidates:
            matched = _match_coverage_components(
                candidate.text,
                components,
                tag=candidate.tag,
            )
            if len(matched) != 1 or matched[0] != component:
                continue
            if candidate.segment_ref in added_refs:
                continue
            augmented.append(candidate)
            added_refs.add(candidate.segment_ref)
            component_added += 1
            if (
                component_added >= _AGGREGATE_SUMMARY_PER_COMPONENT_LIMIT
                or len(added_refs) >= target_additions
            ):
                break
        if len(added_refs) >= target_additions:
            break
    if len(added_refs) < target_additions:
        for _score, candidate in candidates:
            if candidate.segment_ref in added_refs:
                continue
            augmented.append(candidate)
            added_refs.add(candidate.segment_ref)
            if len(added_refs) >= target_additions:
                break
    return augmented


def search_summaries(
    store: ContextStore,
    semantic: SemanticSearchManager,
    query: str,
    max_results: int = 5,
    intent_context: str = "",
    session_filter: str = "",
    mode: str = "lookup",
    conversation_id: str | None = None,
) -> dict:
    """Search compacted summaries, segment text, and related stored context.

    Summary search is segment-first and supports multi-evidence retrieval
    modes like aggregate totals and coverage.
    """
    if not query.strip():
        return {"error": "empty query"}

    mode = _normalize_search_summary_mode(mode)
    coverage_components: list[str] = []

    query_intent = _detect_query_intent(query)
    if query_intent == "default" and intent_context.strip():
        # Fall back to the original user question/intent context when
        # the model issues a narrow tool query (e.g. "shoe rack").
        query_intent = _detect_query_intent(intent_context)

    limit = _candidate_limit(max_results, mode)
    results = _search_summary_candidates(
        store,
        semantic,
        query,
        limit=limit,
        conversation_id=conversation_id,
    )
    if mode == "lookup":
        results = _hydrate_lookup_summary_results(
            store,
            results,
            conversation_id=conversation_id,
        )
    if mode in _MULTI_EVIDENCE_SUMMARY_MODES:
        coverage_components, coverage_queries = _build_coverage_queries(query, intent_context)
        facet_limit = max(max_results, 6)
        for facet_query in coverage_queries:
            results.extend(
                _search_summary_candidates(
                    store,
                    semantic,
                    facet_query,
                    limit=facet_limit,
                    conversation_id=conversation_id,
                    include_semantic=False,
                    include_descriptions=False,
                    include_tool_outputs=False,
                )
            )

    # ---- Span-union: merge overlapping excerpts from the same segment ----
    results = _merge_segment_excerpts(store, results, conversation_id=conversation_id)
    if mode == "aggregate_total" and coverage_components:
        results = _supplement_aggregate_total_from_summaries(
            store,
            query,
            intent_context,
            results,
            coverage_components,
            conversation_id=conversation_id,
        )
    results = _rerank_quote_results(
        results,
        query,
        max_results=max_results,
        mode=mode,
        coverage_components=coverage_components,
    )

    if not results:
        return {
            "query": query,
            "mode": mode,
            "found": False,
            "results": [],
            "message": f"No matches for '{query}' in stored conversation history.",
        }

    # ---- Merge snippets from the same session ----
    session_groups: OrderedDict[str, list[QuoteResult]] = OrderedDict()
    no_session: list[QuoteResult] = []
    for qr in results:
        key = qr.session_date.strip() if qr.session_date else ""
        if key:
            session_groups.setdefault(key, []).append(qr)
        else:
            no_session.append(qr)

    # Parse + normalize session dates for every session-group hit.
    session_dates_parsed: dict[str, date | None] = {}
    session_dates_normalized: dict[str, str] = {}
    for session_date in session_groups:
        parsed = _parse_session_date(session_date)
        session_dates_parsed[session_date] = parsed
        session_dates_normalized[session_date] = parsed.isoformat() if parsed else ""

    # When session_filter is provided, restrict to matching sessions and
    # skip current-state suppression (the model is explicitly drilling in).
    if session_filter.strip():
        _sf = session_filter.strip()
        filtered = OrderedDict()
        for k, v in session_groups.items():
            norm = session_dates_normalized.get(k, "")
            if _sf in k or _sf in norm:
                filtered[k] = v
        if not filtered:
            return {
                "query": query,
                "mode": mode,
                "found": False,
                "results": [],
                "message": f"No matches for '{query}' in session '{_sf}'.",
            }
        session_groups = filtered
        # Bypass current-state multi-session logic when drilling into a
        # specific session — return full excerpts, no suppression.
        session_items = list(session_groups.items())
        formatted = []
        for session_date, group in session_items:
            if mode == "lookup" or mode in _MULTI_EVIDENCE_SUMMARY_MODES or len(group) == 1:
                coverage_group = group if mode == "lookup" or mode in _MULTI_EVIDENCE_SUMMARY_MODES else [group[0]]
                for qr in coverage_group:
                    entry: dict = {
                        "excerpt": qr.text,
                        "topic": qr.tag,
                        "session": session_date,
                        "session_date_normalized": session_dates_normalized.get(session_date, ""),
                        "match_type": qr.match_type,
                    }
                    if mode in _MULTI_EVIDENCE_SUMMARY_MODES and coverage_components:
                        matched_components = _match_coverage_components(
                            qr.text,
                            coverage_components,
                            tag=qr.tag,
                        )
                        if matched_components:
                            entry["matched_components"] = matched_components
                    if mode == "aggregate_total":
                        quantities = _extract_quantity_mentions(qr.text)
                        if quantities:
                            entry["quantities"] = quantities
                    if qr.created_at:
                        entry["created_at"] = qr.created_at
                    if qr.match_type == "semantic":
                        entry["similarity"] = qr.similarity
                    formatted.append(entry)
            else:
                seen_texts: set[str] = set()
                merged_parts: list[str] = []
                topics: list[str] = []
                group_created = [qr.created_at for qr in group if qr.created_at]
                earliest_created = min(group_created) if group_created else ""
                for qr in group:
                    norm = qr.text.strip()
                    if norm not in seen_texts:
                        seen_texts.add(norm)
                        merged_parts.append(qr.text.strip())
                    if qr.tag not in topics:
                        topics.append(qr.tag)
                entry = {
                    "excerpt": "\n---\n".join(merged_parts),
                    "topic": ", ".join(topics),
                    "session": session_date,
                    "session_date_normalized": session_dates_normalized.get(session_date, ""),
                    "merged_count": len(merged_parts),
                    "match_type": group[0].match_type,
                }
                if mode in _MULTI_EVIDENCE_SUMMARY_MODES and coverage_components:
                    matched_components = _match_coverage_components(
                        entry["excerpt"],
                        coverage_components,
                        tag=entry["topic"],
                    )
                    if matched_components:
                        entry["matched_components"] = matched_components
                if mode == "aggregate_total":
                    quantities = _extract_quantity_mentions(entry["excerpt"])
                    if quantities:
                        entry["quantities"] = quantities
                if earliest_created:
                    entry["created_at"] = earliest_created
                formatted.append(entry)
        for qr in no_session:
            entry = {
                "excerpt": qr.text,
                "topic": qr.tag,
                "match_type": qr.match_type,
            }
            if mode in _MULTI_EVIDENCE_SUMMARY_MODES and coverage_components:
                matched_components = _match_coverage_components(
                    qr.text,
                    coverage_components,
                    tag=qr.tag,
                )
                if matched_components:
                    entry["matched_components"] = matched_components
            if mode == "aggregate_total":
                quantities = _extract_quantity_mentions(qr.text)
                if quantities:
                    entry["quantities"] = quantities
            formatted.append(entry)
        response = {
            "query": query,
            "mode": mode,
            "query_intent": query_intent,
            "session_filter": _sf,
            "found": True,
            "results": formatted,
        }
        if mode == "exact_value":
            _apply_exact_value_metadata(
                response,
                formatted,
                query=query,
                intent_context=intent_context,
            )
        elif mode == "coverage":
            response["coverage_summary"] = _build_coverage_summary(formatted, coverage_components)
            response["coverage_value_candidates"] = _build_coverage_value_candidates(formatted, query)
            missing_components = response["coverage_summary"].get("missing_components", [])
            response["reader_hint"] = (
                "COVERAGE MODE: Use the returned excerpts to cover multiple sides "
                "of the question. "
                "Do not add separate numeric values unless an excerpt explicitly says they should be summed. "
                "Prefer repeated or shared values that recur across multiple matched components. "
                + (
                    f"Coverage is still missing for: {', '.join(missing_components)}. "
                    if isinstance(missing_components, list) and missing_components
                    else ""
                )
                + "If one side is still missing, search again with "
                "a narrower follow-up query."
            )
        elif mode == "aggregate_total":
            _apply_aggregate_total_metadata(
                response,
                formatted,
                coverage_components,
                query=query,
                intent_context=intent_context,
            )
        return response

    session_items = list(session_groups.items())
    if query_intent == "current_state":
        # For "currently/now/latest" queries, prefer most recent sessions first.
        session_items.sort(
            key=lambda kv: (
                session_dates_parsed.get(kv[0]) is not None,
                session_dates_parsed.get(kv[0]) or date.min,
            ),
            reverse=True,
        )

    current_state_multi_session = False
    if query_intent == "current_state" and len(session_items) > 1:
        # Only suppress older sessions when the newest session has a
        # topically relevant match (FTS/like hit, or semantic >= 0.4).
        # Without this gate, an unrelated session that happened to be
        # more recent can suppress the sessions that actually answer
        # the question (e.g. a gaming-keyboard session suppressing
        # sneaker-storage sessions).
        _SEMANTIC_GATE = 0.4
        newest_group = session_items[0][1]  # already sorted by recency
        for qr in newest_group:
            if qr.match_type in ("fts", "like", "description"):
                current_state_multi_session = True
                break
            if qr.match_type == "semantic" and qr.similarity >= _SEMANTIC_GATE:
                current_state_multi_session = True
                break
    formatted = []

    for session_rank, (session_date, group) in enumerate(session_items, start=1):
        if mode == "lookup" or mode in _MULTI_EVIDENCE_SUMMARY_MODES or len(group) == 1:
            coverage_group = group if mode == "lookup" or mode in _MULTI_EVIDENCE_SUMMARY_MODES else [group[0]]
            for qr in coverage_group:
                entry: dict = {
                    "excerpt": qr.text,
                    "topic": qr.tag,
                    "segment_ref": qr.segment_ref,
                    "session": session_date,
                    "session_date_normalized": session_dates_normalized.get(session_date, ""),
                    "match_type": qr.match_type,
                }
                if mode in _MULTI_EVIDENCE_SUMMARY_MODES and coverage_components:
                    matched_components = _match_coverage_components(
                        qr.text,
                        coverage_components,
                        tag=qr.tag,
                    )
                    if matched_components:
                        entry["matched_components"] = matched_components
                if mode == "aggregate_total":
                    quantities = _extract_quantity_mentions(qr.text)
                    if quantities:
                        entry["quantities"] = quantities
                if qr.created_at:
                    entry["created_at"] = qr.created_at
                if current_state_multi_session:
                    entry["session_recency_rank"] = session_rank
                    if session_rank == 1:
                        entry["priority"] = "HIGHEST_PRIORITY_CURRENT_STATE"
                if qr.match_type == "semantic":
                    entry["similarity"] = qr.similarity
                formatted.append(entry)
        else:
            # Multiple hits from the same session — merge excerpts
            seen_texts: set[str] = set()
            merged_parts: list[str] = []
            all_refs: list[str] = []
            topics: list[str] = []
            for qr in group:
                norm = qr.text.strip()
                if norm not in seen_texts:
                    seen_texts.add(norm)
                    merged_parts.append(qr.text.strip())
                if qr.tag not in topics:
                    topics.append(qr.tag)
                if qr.segment_ref not in all_refs:
                    all_refs.append(qr.segment_ref)

            # Use earliest created_at from the group for chronological ordering
            group_created = [qr.created_at for qr in group if qr.created_at]
            earliest_created = min(group_created) if group_created else ""
            entry = {
                "excerpt": "\n---\n".join(merged_parts),
                "topic": ", ".join(topics),
                "segment_refs": all_refs,
                "session": session_date,
                "session_date_normalized": session_dates_normalized.get(session_date, ""),
                "merged_count": len(merged_parts),
                "match_type": group[0].match_type,
            }
            if mode in _MULTI_EVIDENCE_SUMMARY_MODES and coverage_components:
                matched_components = _match_coverage_components(
                    entry["excerpt"],
                    coverage_components,
                    tag=entry["topic"],
                )
                if matched_components:
                    entry["matched_components"] = matched_components
            if mode == "aggregate_total":
                quantities = _extract_quantity_mentions(entry["excerpt"])
                if quantities:
                    entry["quantities"] = quantities
            if earliest_created:
                entry["created_at"] = earliest_created
            if current_state_multi_session:
                entry["session_recency_rank"] = session_rank
                if session_rank == 1:
                    entry["priority"] = "HIGHEST_PRIORITY_CURRENT_STATE"
            formatted.append(entry)

    # Sort no-session results chronologically when created_at is available
    no_session.sort(key=lambda qr: qr.created_at or "")

    # Append results with no session date individually
    for qr in no_session:
        entry = {
            "excerpt": qr.text,
            "topic": qr.tag,
            "segment_ref": qr.segment_ref,
            "match_type": qr.match_type,
        }
        if mode in _MULTI_EVIDENCE_SUMMARY_MODES and coverage_components:
            matched_components = _match_coverage_components(
                qr.text,
                coverage_components,
                tag=qr.tag,
            )
            if matched_components:
                entry["matched_components"] = matched_components
        if mode == "aggregate_total":
            quantities = _extract_quantity_mentions(qr.text)
            if quantities:
                entry["quantities"] = quantities
        if qr.created_at:
            entry["created_at"] = qr.created_at
        if qr.match_type == "semantic":
            entry["similarity"] = qr.similarity
        formatted.append(entry)

    response = {
        "query": query,
        "mode": mode,
        "query_intent": query_intent,
        "found": True,
        "results": formatted,
    }
    if current_state_multi_session:
        response["current_state_multi_session"] = True
        response["priority_label"] = "HIGHEST_PRIORITY_CURRENT_STATE"
        # Build a deterministic resolution from the newest session's date.
        newest_date = ""
        if formatted:
            newest_date = formatted[0].get("session_date_normalized", "") or formatted[0].get("session", "")
        response["reader_hint"] = (
            "CURRENT-STATE RESOLUTION RULE (mandatory): "
            "The most recent session"
            + (f" ({newest_date})" if newest_date else "")
            + " is the single authoritative source for the user's current state. "
            "Older sessions are superseded history — do NOT use them to override the newest session. "
            "Intent language ('going to', 'planning to', 'looking forward to') in the newest session "
            "counts as the current answer because the question date is after that session."
        )
    elif mode == "exact_value":
        _apply_exact_value_metadata(
            response,
            formatted,
            query=query,
            intent_context=intent_context,
        )
    elif mode == "coverage":
        response["coverage_summary"] = _build_coverage_summary(formatted, coverage_components)
        response["coverage_value_candidates"] = _build_coverage_value_candidates(formatted, query)
        missing_components = response["coverage_summary"].get("missing_components", [])
        response["reader_hint"] = (
            "COVERAGE MODE: Use the returned excerpts to cover multiple sides "
            "of the question. "
            "Do not add separate numeric values unless an excerpt explicitly says they should be summed. "
            "Prefer repeated or shared values that recur across multiple matched components. "
            + (
                f"Coverage is still missing for: {', '.join(missing_components)}. "
                if isinstance(missing_components, list) and missing_components
                else ""
            )
            + "If one side is still missing, search again with "
            "a narrower follow-up query."
        )
    elif mode == "aggregate_total":
        _apply_aggregate_total_metadata(
            response,
            formatted,
            coverage_components,
            query=query,
            intent_context=intent_context,
        )
    elif mode == "lookup":
        preference_anchor = _extract_summary_preference_anchor(formatted)
        if preference_anchor is not None:
            anchor, calc = preference_anchor
            response["chosen_preference_anchor"] = anchor
            response["anchor_example_calculation"] = calc
            provider = str(anchor.get("provider", "")).strip() or "the anchored provider"
            hourly_rate = str(anchor.get("hourly_rate", "")).strip()
            instance_count = str(anchor.get("instance_count", "")).strip()
            response["reader_hint"] = (
                "PREFERENCE-FOLLOWING RULE (mandatory): A concrete user-specified "
                "requirement was found in the summaries. Carry the chosen_preference_anchor "
                "forward into the answer as the user's example. "
                f"Explicitly mention {provider} at {hourly_rate} for {instance_count} instances, "
                "and you may use anchor_example_calculation to illustrate the structure. "
                "Do not substitute alternate illustrative rates or counts from supporting summaries."
            )
    return response


def find_quote(
    store: ContextStore,
    semantic: SemanticSearchManager,
    query: str,
    max_results: int = 5,
    intent_context: str = "",
    session_filter: str = "",
    mode: str = "lookup",
    conversation_id: str | None = None,
) -> dict:
    """Search canonical archived full_text only."""
    if not query.strip():
        return {"error": "empty query"}

    mode = _normalize_find_quote_mode(mode)
    query_intent = _detect_query_intent(query)
    if query_intent == "default" and intent_context.strip():
        query_intent = _detect_query_intent(intent_context)

    if session_filter.strip():
        # Session drill-down is summary/segment oriented; keep quote search
        # turn-first and let callers use search_summaries when they need
        # explicit session scoping.
        return {
            "error": "session_filter is not supported for turn-first find_quote",
            "mode": mode,
        }

    limit = _candidate_limit(max_results, mode)
    results = _search_find_quote_candidates(
        store,
        semantic,
        query,
        limit=limit,
        mode=mode,
        conversation_id=conversation_id,
    )
    results = _rerank_quote_results(
        results,
        query,
        max_results=max_results,
        mode=mode,
        coverage_components=[],
    )

    if not results:
        return {
            "query": query,
            "mode": mode,
            "found": False,
            "results": [],
            "message": f"No matches for '{query}' in stored conversation history.",
        }

    formatted: list[dict[str, object]] = []
    for qr in results:
        entry: dict[str, object] = {
            "excerpt": qr.text,
            "topic": qr.tag,
            "source_scope": qr.source_scope or "segment",
        }
        if qr.turn_number is not None:
            entry["turn_number"] = qr.turn_number
        if qr.matched_side:
            entry["matched_side"] = qr.matched_side
        if qr.session_date:
            entry["session"] = qr.session_date
            normalized = _normalize_session_date(qr.session_date)
            if normalized:
                entry["session_date_normalized"] = normalized
        if qr.created_at:
            entry["created_at"] = qr.created_at
        if qr.match_type in {"semantic", "turn_semantic", "full_text_semantic"} and qr.similarity:
            entry["match_type"] = qr.match_type
            entry["similarity"] = qr.similarity
        entry["segment_ref"] = qr.segment_ref
        formatted.append(entry)

    response = {
        "query": query,
        "mode": mode,
        "query_intent": query_intent,
        "found": True,
        "results": formatted,
    }
    if mode == "exact_value":
        _apply_exact_value_metadata(
            response,
            formatted,
            query=query,
            intent_context=intent_context,
        )
    elif mode == "lookup":
        contradiction_hint = _maybe_build_lookup_reader_hint(
            query=query,
            intent_context=intent_context,
            formatted=formatted,
        )
        if contradiction_hint:
            response["reader_hint"] = contradiction_hint
        else:
            response["reader_hint"] = _default_lookup_reader_hint()
    return response
