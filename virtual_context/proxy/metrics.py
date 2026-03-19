"""Thread-safe event collector for the proxy dashboard."""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ProxyMetrics:
    """Collects structured events from the proxy pipeline.

    Thread-safe: ``record()`` can be called from the background
    ThreadPoolExecutor that runs ``on_turn_complete``.

    Events are kept in a 24-hour rolling buffer capped at
    :pyattr:`MAX_EVENTS`.  Older entries are evicted lazily --
    every 100 ``record()`` calls and at the start of
    ``events_since()`` / ``snapshot()``.
    """

    BASELINE_RATIO = 0.30  # typical summarization compression (3.3x)
    MAX_EVENTS = 50_000
    MAX_AGE_S = 86_400  # 24 hours

    def __init__(self, context_window: int = 120_000, telemetry_ledger=None, db_path: str | None = None) -> None:
        self.start_time: float = time.time()
        self.context_window: int = context_window
        self._events: list[dict] = []
        self._buckets: dict[str, int] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._request_bodies: deque[dict] = deque(maxlen=50)
        self._telemetry_ledger = telemetry_ledger
        self._db: sqlite3.Connection | None = None

        if db_path:
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS metrics_events ("
                "  seq INTEGER PRIMARY KEY,"
                "  ts TEXT NOT NULL,"
                "  type TEXT NOT NULL,"
                "  conversation_id TEXT,"
                "  recorded_at REAL NOT NULL,"
                "  data_json TEXT NOT NULL"
                ")"
            )
            self._db.commit()
            # Restore seq from existing data
            row = self._db.execute("SELECT MAX(seq) FROM metrics_events").fetchone()
            if row and row[0] is not None:
                self._seq = row[0] + 1
            # Load existing events into memory for snapshot/events_since
            self._load_from_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """Load events from SQLite into memory. Must be called during __init__."""
        if not self._db:
            return
        cutoff = time.time() - self.MAX_AGE_S
        rows = self._db.execute(
            "SELECT data_json FROM metrics_events WHERE recorded_at >= ? ORDER BY seq",
            (cutoff,),
        ).fetchall()
        for (data_json,) in rows:
            try:
                self._events.append(json.loads(data_json))
            except (json.JSONDecodeError, TypeError):
                pass

    def _evict_old(self) -> None:
        """Remove events older than *MAX_AGE_S*.  Must hold ``_lock``."""
        cutoff = time.time() - self.MAX_AGE_S
        # Events are appended in chronological order -- scan from the
        # front and find the first index that is still fresh.
        idx = 0
        while idx < len(self._events):
            if self._events[idx].get("_recorded_at", 0) >= cutoff:
                break
            idx += 1
        if idx:
            del self._events[:idx]
        # Also evict from SQLite
        if self._db:
            self._db.execute("DELETE FROM metrics_events WHERE recorded_at < ?", (cutoff,))
            self._db.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event: dict) -> None:
        """Append an event (thread-safe). Adds ``_seq``, ``ts``, and ``_recorded_at``."""
        with self._lock:
            event = dict(event)  # shallow copy to avoid caller mutation
            event["_seq"] = self._seq
            if "_recorded_at" not in event:
                event["_recorded_at"] = time.time()
            if "ts" not in event:
                event["ts"] = datetime.now(timezone.utc).isoformat()
            self._seq += 1
            self._events.append(event)
            # Persist to SQLite
            if self._db:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO metrics_events (seq, ts, type, conversation_id, recorded_at, data_json) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            event["_seq"],
                            event.get("ts", ""),
                            event.get("type", ""),
                            event.get("conversation_id", ""),
                            event["_recorded_at"],
                            json.dumps(event),
                        ),
                    )
                    self._db.commit()
                except Exception:
                    pass  # never block event recording for DB errors
            # Debug: log every event for diagnostics
            etype = event.get("type", "?")
            conv = event.get("conversation_id", "")[:12]
            extras = ""
            if etype == "request":
                extras = f" turn={event.get('turn')} passthrough={event.get('passthrough')} in={event.get('input_tokens',0)} ctx={event.get('context_tokens',0)}"
            elif etype == "response":
                extras = f" turn={event.get('turn')} ms={event.get('upstream_ms',0)}"
            elif etype == "turn_complete":
                extras = f" turn={event.get('turn')} tags={event.get('tags',[])} primary={event.get('primary_tag','')}"
            elif etype == "compaction":
                extras = f" segments={event.get('segments')} freed={event.get('tokens_freed')} summaries={event.get('tag_summaries_built')}"
            elif etype == "compaction_progress":
                extras = f" phase={event.get('phase')} done={event.get('done')}/{event.get('total')} tag={event.get('primary_tag','')}"
            elif etype == "ingested_turn":
                extras = f" turn={event.get('turn')} done={event.get('done')}/{event.get('total')} tags={event.get('tags',[])} preview=\"{event.get('message_preview','')[:40]}\""
            elif etype == "history_ingestion":
                extras = f" turns={event.get('turns_ingested')} ms={event.get('elapsed_ms',0):.0f}"
            elif etype == "conversation_deleted":
                extras = f" conv={event.get('conversation_id','')[:12]} segments={event.get('segments_removed')}"
            elif etype == "budget_exceeded":
                extras = f" total={event.get('total')} budget={event.get('budget')}"
            logger.info("EVENT seq=%d type=%s conv=%s%s", self._seq - 1, etype, conv, extras)
            event_type = event.get("type", "")
            if event_type:
                self._buckets[event_type] = self._buckets.get(event_type, 0) + 1
            # Periodic eviction + hard cap
            if self._seq % 100 == 0:
                self._evict_old()
            if len(self._events) > self.MAX_EVENTS:
                del self._events[:len(self._events) - self.MAX_EVENTS]

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            self._evict_old()
            return [e for e in self._events if e["_seq"] > seq]

    def snapshot(self) -> dict:
        """Aggregate stats for the initial SSE load."""
        with self._lock:
            self._evict_old()
            requests = [e for e in self._events if e.get("type") == "request"]
            compactions = [e for e in self._events if e.get("type") == "compaction"]
            turn_completes = [e for e in self._events if e.get("type") == "turn_complete"]
            ingestions = [e for e in self._events if e.get("type") == "history_ingestion"]
            responses = [e for e in self._events if e.get("type") == "response"]
            ingested_turns = [e for e in self._events if e.get("type") == "ingested_turn"]
            tool_intercepts = [e for e in self._events if e.get("type") == "tool_intercept"]
            compaction_progress = [e for e in self._events if e.get("type") == "compaction_progress"]

            wait_values = [r["wait_ms"] for r in requests if "wait_ms" in r]
            inbound_values = [r["inbound_ms"] for r in requests if "inbound_ms" in r]
            context_values = [r["context_tokens"] for r in requests if "context_tokens" in r]

            total_original = sum(c.get("original_tokens", 0) for c in compactions)
            total_summary = sum(c.get("summary_tokens", 0) for c in compactions)
            total_freed = sum(c.get("tokens_freed", 0) for c in compactions)
            compression_ratio = (
                round(total_summary / total_original, 3)
                if total_original > 0 else 0
            )
            total_context_injected = sum(
                r.get("context_tokens", 0) for r in requests
            )

            # Session efficiency: actual vs baseline input tokens
            total_actual_input = sum(
                r.get("input_tokens", 0) for r in requests
            )

            # Raw baseline: sum of raw_input_tokens from request events.
            # This is the true cost a naive system would incur by forwarding
            # the full unfiltered payload each turn.
            total_raw_baseline = sum(
                r.get("raw_input_tokens", 0) for r in requests
            )

            # Fallback: accumulated turn-pair baseline (for headless mode
            # where raw_input_tokens is not available on request events)
            system_tokens_per_turn = 0
            if requests:
                system_tokens_per_turn = requests[-1].get("system_tokens", 0)
            baseline_history_tokens = 0
            for ing in ingestions:
                baseline_history_tokens += ing.get("baseline_history_tokens", 0)
            cumulative_baseline = 0
            for tc in turn_completes:
                tpt = tc.get("turn_pair_tokens", 0)
                baseline_history_tokens += tpt
                if baseline_history_tokens > self.context_window:
                    protected = tpt * 4
                    compactable = max(0, baseline_history_tokens - protected)
                    baseline_history_tokens = (
                        round(compactable * self.BASELINE_RATIO) + protected
                    )
                cumulative_baseline += system_tokens_per_turn + baseline_history_tokens

            # Prefer raw baseline (accurate for proxy mode) over accumulated
            total_baseline_input = (
                total_raw_baseline if total_raw_baseline > 0
                else cumulative_baseline
            )

            # Telemetry
            telemetry = {}
            if self._telemetry_ledger:
                try:
                    telem_dict = self._telemetry_ledger.to_dict()
                    if isinstance(telem_dict, dict):
                        # Remove raw events for snapshot (too large for SSE)
                        telem_dict.pop("events", None)
                        telemetry = telem_dict
                except Exception:
                    pass

            return {
                "type": "snapshot",
                "uptime_s": round(time.time() - self.start_time, 1),
                "total_requests": len(requests),
                "total_compactions": len(compactions),
                "total_tokens_freed": total_freed,
                "total_original_tokens": total_original,
                "total_summary_tokens": total_summary,
                "compression_ratio": compression_ratio,
                "total_context_injected": total_context_injected,
                "total_actual_input": total_actual_input,
                "total_baseline_input": total_baseline_input,
                "baseline_ratio": self.BASELINE_RATIO,
                "avg_wait_ms": round(statistics.mean(wait_values), 1) if wait_values else 0,
                "avg_inbound_ms": round(statistics.mean(inbound_values), 1) if inbound_values else 0,
                "avg_context_tokens": round(statistics.mean(context_values), 1) if context_values else 0,
                "recent_requests": list(requests[-50:]),
                "compactions": list(compactions),
                "turn_completes": list(turn_completes[-50:]),
                "responses": list(responses[-50:]),
                "history_ingestions": list(ingestions),
                "ingested_turns": list(ingested_turns),
                "total_turns_ingested": sum(
                    e.get("turns_ingested", 0) for e in ingestions
                ),
                "tool_intercepts": list(tool_intercepts),
                "total_tool_intercepts": len(tool_intercepts),
                "compaction_progress": list(compaction_progress),
                "telemetry": telemetry,
                "budget_promoted": next(
                    (e for e in reversed(self._events)
                     if e.get("type") == "budget_auto_promoted"), None,
                ),
                "budget_exceeded": next(
                    (e for e in reversed(self._events)
                     if e.get("type") == "budget_exceeded"), None,
                ),
            }

    def capture_request(
        self,
        turn: int,
        body: dict,
        api_format: str,
        *,
        inbound_tags: list[str] | None = None,
        conversation_id: str = "",
        passthrough: bool = False,
        inbound_tokens: int = 0,
        outbound_tokens: int = 0,
        inbound_bytes: int = 0,
        outbound_bytes: int = 0,
        context_tokens: int = 0,
        overhead_ms: float = 0,
        turns_dropped: int = 0,
        turns_stubbed: int = 0,
        message_preview: str = "",
    ) -> None:
        """Capture raw request body for inspection (thread-safe, ring buffer)."""
        with self._lock:
            self._request_bodies.append({
                "turn": turn,
                "ts": datetime.now(timezone.utc).isoformat(),
                "api_format": api_format,
                "model": body.get("model"),
                "stream": body.get("stream", False),
                "system": body.get("system"),
                "messages": body.get("messages", []),
                "message_count": len(body.get("messages", [])),
                "inbound_tags": inbound_tags or [],
                "response_tags": [],
                "conversation_id": conversation_id,
                "passthrough": passthrough,
                "inbound_tokens": inbound_tokens,
                "outbound_tokens": outbound_tokens,
                "inbound_bytes": inbound_bytes,
                "outbound_bytes": outbound_bytes,
                "context_tokens": context_tokens,
                "overhead_ms": overhead_ms,
                "turns_dropped": turns_dropped,
                "turns_stubbed": turns_stubbed,
                "message_preview": message_preview,
            })

    def capture_enriched(self, turn: int, body: dict) -> None:
        """Capture the enriched request body sent to the LLM."""
        with self._lock:
            for req in self._request_bodies:
                if req["turn"] == turn:
                    req["enriched"] = body
                    return

    def capture_response(
        self, turn: int, body: dict, *,
        upstream_input_tokens: int = 0,
        upstream_output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        """Capture the LLM response body and upstream token usage."""
        with self._lock:
            for req in self._request_bodies:
                if req["turn"] == turn:
                    req["response"] = body
                    if upstream_input_tokens:
                        req["upstream_input_tokens"] = upstream_input_tokens
                    if upstream_output_tokens:
                        req["upstream_output_tokens"] = upstream_output_tokens
                    if cache_creation_input_tokens:
                        req["cache_creation_input_tokens"] = cache_creation_input_tokens
                    if cache_read_input_tokens:
                        req["cache_read_input_tokens"] = cache_read_input_tokens
                    return

    def update_request_tags(
        self, turn: int, *, response_tags: list[str] | None = None,
    ) -> None:
        with self._lock:
            for req in self._request_bodies:
                if req["turn"] == turn:
                    if response_tags is not None:
                        req["response_tags"] = response_tags
                    return

    def get_captured_request(self, turn: int) -> dict | None:
        with self._lock:
            for req in self._request_bodies:
                if req["turn"] == turn:
                    return dict(req)
            return None

    def get_captured_requests_summary(self) -> list[dict]:
        """Return summaries (without full messages/system) for the list view."""
        with self._lock:
            return [{
                "turn": r["turn"],
                "ts": r["ts"],
                "api_format": r["api_format"],
                "model": r["model"],
                "stream": r.get("stream", False),
                "message_count": r["message_count"],
                "conversation_id": r.get("conversation_id", ""),
                "inbound_tags": r.get("inbound_tags", []),
                "response_tags": r.get("response_tags", []),
                "passthrough": r.get("passthrough", False),
                "inbound_tokens": r.get("inbound_tokens", 0),
                "outbound_tokens": r.get("outbound_tokens", 0),
                "inbound_bytes": r.get("inbound_bytes", 0),
                "outbound_bytes": r.get("outbound_bytes", 0),
                "context_tokens": r.get("context_tokens", 0),
                "overhead_ms": r.get("overhead_ms", 0),
                "turns_dropped": r.get("turns_dropped", 0),
                "turns_stubbed": r.get("turns_stubbed", 0),
                "message_preview": r.get("message_preview", ""),
                "upstream_input_tokens": r.get("upstream_input_tokens", 0),
                "upstream_output_tokens": r.get("upstream_output_tokens", 0),
                "cache_creation_input_tokens": r.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": r.get("cache_read_input_tokens", 0),
            } for r in self._request_bodies]

    def restore_request_captures(self, captures: list[dict]) -> None:
        """Restore persisted request captures into the ring buffer."""
        with self._lock:
            for cap in captures:
                self._request_bodies.append(cap)
