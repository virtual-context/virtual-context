"""In-process repetition guard for query-driven VC search tools.

Tracks recent search-tool executions per ``(conversation_id, tool_name)``
in a sliding time window.  Each execution is recorded with a normalized
fingerprint of its search-intent arguments (lowercased, whitespace
collapsed; volatile arguments such as modes, limits, channels, and time
ranges are ignored).  Once the window already holds ``threshold``
executions for one conversation/tool pair, further calls are answered
with a synthetic tool result that tells the model how many similar
searches already ran, lists the distinct queries tried, and directs it
to answer from the evidence already gathered.

Guard state is process-local and never persisted: each worker guards
only its own traffic.  Non-search tools are never guarded, and the
first execution recorded for a window can never trip the guard.  Any
internal guard failure fails open — the tool executes normally.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Callable

logger = logging.getLogger(__name__)

# Query-driven search tools eligible for guarding.  Structural or
# navigation tools (expand/collapse, recall_all, restore) are never
# guarded.
GUARDED_TOOL_NAMES: frozenset[str] = frozenset({
    "vc_find_quote",
    "vc_search_summaries",
    "vc_find_session",
    "vc_query_facts",
    "vc_remember_when",
})

# Argument keys that carry search intent and thus participate in the
# fingerprint.  Everything else (mode, channel, max_results, depth,
# time_range, status, fact_type, ...) is volatile and excluded.
_FINGERPRINT_KEYS: tuple[str, ...] = (
    "query",
    "subject",
    "verb",
    "object_contains",
    "session",
)

_WS_RE = re.compile(r"\s+")

# Bounds on the synthetic result so a runaway loop cannot inflate it.
_MAX_LISTED_QUERIES = 20
_MAX_QUERY_CHARS = 120

# Fallbacks used when a config object carries non-numeric knob values.
_DEFAULT_WINDOW_SECONDS = 120.0
_DEFAULT_THRESHOLD = 10


def normalize_query_fingerprint(tool_input: dict | None) -> str:
    """Return the normalized search-intent fingerprint for a tool call.

    Lowercases and whitespace-collapses each intent-bearing string
    argument; ignores volatile arguments entirely.
    """
    parts: list[str] = []
    for key in _FINGERPRINT_KEYS:
        value = (tool_input or {}).get(key)
        if isinstance(value, str):
            collapsed = _WS_RE.sub(" ", value).strip().lower()
            if collapsed:
                parts.append(collapsed)
    return " | ".join(parts)


class ToolRepetitionGuard:
    """Sliding-window repetition guard keyed by (conversation_id, tool_name)."""

    def __init__(self, *, time_fn: Callable[[], float] = time.monotonic) -> None:
        self._time_fn = time_fn
        self._lock = threading.Lock()
        self._windows: dict[tuple[str, str], deque[tuple[float, str]]] = {}

    def reset(self) -> None:
        """Drop all recorded state (testing/maintenance hook)."""
        with self._lock:
            self._windows.clear()

    def check_and_record(
        self,
        conversation_id: str,
        tool_name: str,
        tool_input: dict | None,
        *,
        window_seconds: float,
        threshold: int,
    ) -> dict | None:
        """Record one attempted execution, or report a guard trip.

        Returns ``None`` when the call should execute normally (the
        execution is recorded in the window).  Returns a payload dict
        ``{"count": int, "distinct_queries": list[str]}`` when the
        window already holds ``threshold`` executions; tripped calls
        are not recorded, so an idle window drains and resets.
        """
        if tool_name not in GUARDED_TOOL_NAMES:
            return None
        now = self._time_fn()
        horizon = now - max(float(window_seconds), 0.0)
        key = (conversation_id, tool_name)
        fingerprint = normalize_query_fingerprint(tool_input)
        with self._lock:
            window = self._windows.get(key)
            if window is None:
                window = deque()
                self._windows[key] = window
            while window and window[0][0] <= horizon:
                window.popleft()
            count = len(window)
            # A window's first execution can never trip, regardless of
            # configuration: only previously recorded executions count.
            if count and count >= max(int(threshold), 1):
                distinct: list[str] = []
                for _, seen in window:
                    if seen and seen not in distinct:
                        distinct.append(seen)
                return {"count": count, "distinct_queries": distinct}
            window.append((now, fingerprint))
            self._sweep_locked(horizon)
            return None

    def _sweep_locked(self, horizon: float) -> None:
        """Drop dead windows so long-lived workers do not accumulate
        one deque per historical conversation.  Caller holds the lock."""
        if len(self._windows) <= 512:
            return
        dead = [
            key
            for key, window in self._windows.items()
            if not window or window[-1][0] <= horizon
        ]
        for key in dead:
            del self._windows[key]


_default_guard = ToolRepetitionGuard()


def reset_default_guard() -> None:
    """Clear the process-wide guard state (testing hook)."""
    _default_guard.reset()


def _synthetic_result(
    tool_name: str,
    count: int,
    distinct_queries: list[str],
    window_seconds: float,
) -> dict:
    listed = [q[:_MAX_QUERY_CHARS] for q in distinct_queries[:_MAX_LISTED_QUERIES]]
    remainder = len(distinct_queries) - len(listed)
    tried = "; ".join(f'"{q}"' for q in listed) if listed else "(no query text)"
    if remainder > 0:
        tried += f"; and {remainder} more"
    message = (
        f"[REPETITION GUARD] {count} similar {tool_name} searches already "
        f"ran for this conversation in the last {int(window_seconds)} "
        f"seconds, so this call was not executed. Distinct queries already "
        f"tried: {tried}. Stop searching now and answer the question from "
        f"the results already gathered above. If the evidence is not there, "
        f"state what is missing instead of repeating searches."
    )
    return {
        "repetition_guard": True,
        "searches_already_run": count,
        "queries_tried": listed,
        "message": message,
    }


def _coerce_number(value: object, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def guard_tool_execution(
    conversation_id: str,
    tool_name: str,
    tool_input: dict | None,
    search_config: object,
) -> str | None:
    """Check the process-wide repetition guard for one tool execution.

    Returns ``None`` when the tool should execute normally, or a JSON
    tool-result string that replaces execution when the guard trips.
    The guard fails open: disabled config, missing conversation
    identity, malformed knobs, or internal errors all execute the tool.
    """
    try:
        if tool_name not in GUARDED_TOOL_NAMES:
            return None
        if getattr(search_config, "tool_guard_enabled", False) is not True:
            return None
        if not conversation_id or not isinstance(conversation_id, str):
            return None
        window_seconds = _coerce_number(
            getattr(search_config, "tool_guard_window_seconds", None),
            _DEFAULT_WINDOW_SECONDS,
        )
        threshold = int(_coerce_number(
            getattr(search_config, "tool_guard_threshold", None),
            _DEFAULT_THRESHOLD,
        ))
        trip = _default_guard.check_and_record(
            conversation_id,
            tool_name,
            tool_input,
            window_seconds=window_seconds,
            threshold=threshold,
        )
        if trip is None:
            return None
        logger.warning(
            "TOOL_GUARD_TRIPPED conv=%s tool=%s count=%d",
            conversation_id, tool_name, trip["count"],
        )
        return json.dumps(_synthetic_result(
            tool_name, trip["count"], trip["distinct_queries"], window_seconds,
        ))
    except Exception:
        logger.debug("tool guard failed open", exc_info=True)
        return None
