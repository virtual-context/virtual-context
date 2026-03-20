"""TemporalResolver: time-bounded recall over stored conversation data.

Handles ``remember_when`` queries: resolves relative/absolute time ranges,
filters quote search results by session date, and queries experience facts
within the date window.  Extracted from engine.py.
"""

from __future__ import annotations

import logging
import re
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from .quote_search import _parse_session_date as _parse_session_date_str

if TYPE_CHECKING:
    from .search_engine import SearchEngine
    from .store import ContextStore
    from ..types import VirtualContextConfig

logger = logging.getLogger(__name__)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


class TemporalResolver:
    """Time-bounded recall: resolves date windows, filters quotes, queries facts.

    Constructor takes:
        store:          a ContextStore instance
        search_engine:  a SearchEngine instance (for ``find_quote``)
        config:         a VirtualContextConfig instance
    """

    def __init__(
        self,
        store: ContextStore,
        search_engine: SearchEngine,
        config: VirtualContextConfig,
    ) -> None:
        self._store = store
        self._search = search_engine
        self._config = config
        self.reference_date: date | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember_when(
        self,
        query: str,
        time_range: dict,
        max_results: int | None = None,
    ) -> dict:
        """Find memory snippets for *query* constrained to a resolved date window."""
        if max_results is None:
            max_results = self._config.search.remember_when_max_results
        if not query.strip():
            return {"error": "empty query"}

        try:
            start, end, resolved_kind = self._resolve_remember_when_range(time_range)
        except ValueError as exc:
            return {"error": str(exc)}

        # Overfetch, then filter by session_date bounds.
        raw = self._search.find_quote(query=query, max_results=max(max_results * 4, 20))
        if not raw.get("found"):
            return {
                "query": query,
                "found": False,
                "range": {
                    "kind": resolved_kind,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                "results": [],
                "message": raw.get("message", "No matches found."),
            }

        filtered: list[dict] = []
        for item in raw.get("results", []):
            session = str(item.get("session", "")).strip()
            parsed = self._parse_session_date(session)
            if parsed is None:
                # No parseable session date -> exclude from time-filtered recall.
                continue
            if start <= parsed <= end:
                filtered.append(item)
            if len(filtered) >= max_results:
                break

        # Also query structured facts within the date window.
        # Return completed experience facts — these are the most relevant
        # for "what did I do" temporal questions.
        fact_results: list[dict] = []
        try:
            start_str = start.strftime("%Y/%m/%d")
            end_str = end.strftime("%Y/%m/%d")
            facts = self._store.query_experience_facts_by_date(
                start_date=start_str,
                end_date=end_str,
                limit=max_results * 5,
                conversation_id=self._config.conversation_id or None,
            )
            for f in facts:
                fact_results.append({
                    "type": "fact",
                    "what": f.what or f"{f.subject} {f.verb} {f.object}",
                    "when": f.when_date or f.session_date or "",
                    "where": f.where or "",
                    "status": f.status or "",
                })
        except Exception as exc:
            logger.warning("remember_when fact query failed: %s", exc)

        return {
            "query": query,
            "found": bool(filtered) or bool(fact_results),
            "range": {
                "kind": resolved_kind,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "results": filtered,
            "facts_in_window": fact_results,
            "message": (
                f"No matches for '{query}' in the requested time window."
                if not filtered and not fact_results else ""
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_session_date(self, raw: str) -> date | None:
        """Best-effort parse for session date strings from stored metadata."""
        return _parse_session_date_str(raw)

    def _resolve_remember_when_range(self, time_range: dict) -> tuple[date, date, str]:
        """Resolve tool time_range input to absolute [start_date, end_date]."""
        if not isinstance(time_range, dict):
            raise ValueError("time_range must be an object")

        kind = str(time_range.get("kind", "")).strip().lower()
        today = self.reference_date or datetime.now(timezone.utc).date()

        if kind == "relative":
            preset = str(time_range.get("preset", "")).strip().lower()
            if preset == "last_24_hours":
                return today - timedelta(days=2), today, preset
            if preset == "last_7_days":
                return today - timedelta(days=7), today, preset
            if preset == "last_30_days":
                return today - timedelta(days=30), today, preset
            if preset == "last_90_days":
                return today - timedelta(days=90), today, preset
            if preset == "this_week":
                start = today - timedelta(days=today.weekday())
                return start, start + timedelta(days=6), preset
            if preset == "last_week":
                this_week_start = today - timedelta(days=today.weekday())
                start = this_week_start - timedelta(days=7)
                return start, start + timedelta(days=6), preset
            if preset == "this_month":
                start = date(today.year, today.month, 1)
                end = date(today.year, today.month, monthrange(today.year, today.month)[1])
                return start, end, preset
            if preset == "last_month":
                year = today.year
                month = today.month - 1
                if month == 0:
                    month = 12
                    year -= 1
                start = date(year, month, 1)
                end = date(year, month, monthrange(year, month)[1])
                return start, end, preset
            if preset == "this_year":
                return date(today.year, 1, 1), date(today.year, 12, 31), preset
            if preset == "last_year":
                y = today.year - 1
                return date(y, 1, 1), date(y, 12, 31), preset
            raise ValueError(f"unsupported relative preset: {preset}")

        if kind == "between_dates":
            start_raw = str(time_range.get("start", "")).strip()
            end_raw = str(time_range.get("end", "")).strip()
            if not start_raw or not end_raw:
                raise ValueError("between_dates requires start and end")

            def parse_boundary(raw: str, is_end: bool) -> date:
                if _ISO_DATE_RE.match(raw):
                    return date.fromisoformat(raw)
                if _ISO_MONTH_RE.match(raw):
                    y, mo = [int(x) for x in raw.split("-")]
                    if is_end:
                        return date(y, mo, monthrange(y, mo)[1])
                    return date(y, mo, 1)
                raise ValueError(f"invalid date format: {raw}")

            start = parse_boundary(start_raw, is_end=False)
            end = parse_boundary(end_raw, is_end=True)
            if end < start:
                raise ValueError("time_range end must be >= start")
            return start, end, "between_dates"

        raise ValueError("time_range.kind must be 'relative' or 'between_dates'")
