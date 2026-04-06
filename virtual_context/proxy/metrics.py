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

# Summary fields persisted for request captures and returned by get_captured_requests_summary.
# Keys with no default are read directly (KeyError if missing); keys with defaults use .get().
_SUMMARY_FIELDS_DEFAULTS: dict[str, object] = {
    "turn": ...,
    "ts": ...,
    "api_format": ...,
    "model": ...,
    "stream": False,
    "message_count": ...,
    "conversation_id": "",
    "inbound_tags": [],
    "response_tags": [],
    "passthrough": False,
    "inbound_tokens": 0,
    "outbound_tokens": 0,
    "inbound_bytes": 0,
    "outbound_bytes": 0,
    "context_tokens": 0,
    "overhead_ms": 0,
    "prepare_total_ms": 0,
    "prepare_breakdown": {},
    "turns_dropped": 0,
    "turns_stubbed": 0,
    "message_preview": "",
    "upstream_input_tokens": 0,
    "upstream_output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "non_virtualizable_floor": 0,
    "upstream_context_limit": 0,
    "passthrough_trim_limit": 0,
    "system_tokens": 0,
    "protected_turn_tokens": 0,
    "protected_turn_count": 0,
}


def _extract_summary(req: dict) -> dict:
    out: dict = {}
    for key, default in _SUMMARY_FIELDS_DEFAULTS.items():
        if default is ...:
            out[key] = req[key]
        else:
            out[key] = req.get(key, default)
    return out


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

    def __init__(self, context_window: int = 120_000, telemetry_ledger=None,
                 db_path: str | None = None, store=None,
                 restore_captures: bool = True,
                 on_event=None) -> None:
        self.start_time: float = time.time()
        self.context_window: int = context_window
        self._events: list[dict] = []
        self._buckets: dict[str, int] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._request_bodies: deque[dict] = deque(maxlen=50)
        self._telemetry_ledger = telemetry_ledger
        self._on_event = on_event  # Optional callback for cross-worker broadcasting
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

        self._store = store
        if self._store and restore_captures:
            try:
                persisted = self._store.load_request_captures(limit=50)
                if persisted:
                    self.restore_request_captures(persisted)
            except Exception:
                logger.debug("Failed to restore request captures from store", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
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

    @staticmethod
    def _capture_conversation_id(value: str | None) -> str:
        return value or ""

    def _capture_matches(
        self,
        req: dict,
        turn: int,
        conversation_id: str | None = None,
    ) -> bool:
        if req.get("turn") != turn:
            return False
        if conversation_id is None:
            return True
        return (req.get("conversation_id", "") or "") == conversation_id

    def _find_capture(
        self,
        turn: int,
        conversation_id: str | None = None,
    ) -> dict | None:
        for req in reversed(self._request_bodies):
            if self._capture_matches(req, turn, conversation_id):
                return req
        return None

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
                    logger.debug("metrics DB write failed", exc_info=True)  # never block event recording for DB errors
            # Debug: log every event for diagnostics
            etype = event.get("type", "?")
            conv = event.get("conversation_id", "")[:12]
            extras = ""
            if etype == "request":
                extras = f" turn={event.get('turn')} passthrough={event.get('passthrough')} in={event.get('input_tokens',0)} ctx={event.get('context_tokens',0)}"
                _prep_total = event.get("prepare_total_ms", 0)
                if _prep_total:
                    extras += f" prep={_prep_total}"
            elif etype == "response":
                _cr = event.get('cache_read_input_tokens', 0)
                _cc = event.get('cache_creation_input_tokens', 0)
                _cache_str = f" cache_read={_cr} cache_create={_cc}" if _cr or _cc else ""
                extras = f" turn={event.get('turn')} ms={event.get('upstream_ms',0)}{_cache_str}"
            elif etype == "turn_complete":
                extras = f" turn={event.get('turn')} tags={event.get('tags',[])} primary={event.get('primary_tag','')}"
            elif etype == "compaction":
                extras = f" segments={event.get('segments')} freed={event.get('tokens_freed')} summaries={event.get('tag_summaries_built')}"
            elif etype == "compaction_progress":
                extras = f" phase={event.get('phase')} done={event.get('done')}/{event.get('total')} tag={event.get('primary_tag','')}"
            elif etype == "compaction_error":
                extras = f" error={event.get('error','')[:120]}"
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

            # Broadcast to cross-worker listeners (e.g., Redis pub/sub)
            if self._on_event:
                try:
                    self._on_event(event)
                except Exception:
                    pass

    _STALE_EVENT_TYPES = frozenset((
        "ingested_turn", "history_ingestion",
        "compaction_progress", "compaction",
    ))

    def clear_ingestion_events(self, conversation_id: str) -> int:
        """Remove stale ingestion and compaction events for a conversation.

        Called when a new ingestion starts so old progress data from a
        previous (possibly interrupted) run doesn't pollute the dashboard.
        Returns the number of events removed.
        """
        with self._lock:
            before = len(self._events)
            self._events = [
                e for e in self._events
                if not (
                    e.get("conversation_id") == conversation_id
                    and e.get("type") in self._STALE_EVENT_TYPES
                )
            ]
            removed = before - len(self._events)
            # Also clear from SQLite
            if self._db and removed:
                try:
                    self._db.execute(
                        "DELETE FROM metrics_events WHERE conversation_id = ? "
                        "AND type IN ('ingested_turn', 'history_ingestion', 'compaction_progress', 'compaction')",
                        (conversation_id,),
                    )
                    self._db.commit()
                except Exception:
                    pass
            if removed:
                logger.info(
                    "Cleared %d stale ingestion/compaction events for conv=%s",
                    removed, conversation_id[:12],
                )
            return removed

    def delete_conversation_artifacts(self, conversation_id: str) -> dict[str, int]:
        """Purge all in-memory and persisted metrics for a conversation."""
        conv_id = self._capture_conversation_id(conversation_id)
        if not conv_id:
            return {"captures_removed": 0, "events_removed": 0}

        with self._lock:
            captures = [
                req for req in self._request_bodies
                if (req.get("conversation_id", "") or "") != conv_id
            ]
            captures_removed = len(self._request_bodies) - len(captures)
            if captures_removed:
                self._request_bodies = deque(
                    captures,
                    maxlen=self._request_bodies.maxlen,
                )

            events = [
                event for event in self._events
                if (event.get("conversation_id", "") or "") != conv_id
            ]
            events_removed = len(self._events) - len(events)
            if events_removed:
                self._events = events

            if self._db:
                try:
                    self._db.execute(
                        "DELETE FROM metrics_events WHERE conversation_id = ?",
                        (conv_id,),
                    )
                    self._db.commit()
                except Exception:
                    logger.debug(
                        "Failed to delete metrics artifacts for conv=%s",
                        conv_id[:12],
                        exc_info=True,
                    )

            if captures_removed or events_removed:
                logger.info(
                    "Deleted metrics artifacts for conv=%s captures=%d events=%d",
                    conv_id[:12],
                    captures_removed,
                    events_removed,
                )

            return {
                "captures_removed": captures_removed,
                "events_removed": events_removed,
            }

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            self._evict_old()
            return [e for e in self._events if e["_seq"] > seq]

    def _snapshot_locked(self) -> dict:
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

        # Cache metrics: last response only (not cumulative)
        last_resp = responses[-1] if responses else {}
        last_cache_read = last_resp.get("cache_read_input_tokens", 0)
        last_cache_creation = last_resp.get("cache_creation_input_tokens", 0)
        last_upstream_input = last_resp.get("upstream_input_tokens", 0)
        _cache_denom = last_upstream_input if last_upstream_input > 0 else 1
        last_cache_hit_ratio = round(last_cache_read / _cache_denom, 3) if last_upstream_input > 0 else 0.0

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
                logger.debug("telemetry snapshot failed", exc_info=True)

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
            "last_wait_ms": round(wait_values[-1], 1) if wait_values else 0,
            "last_inbound_ms": round(inbound_values[-1], 1) if inbound_values else 0,
            "last_context_tokens": round(context_values[-1], 1) if context_values else 0,
            "last_vc_overhead_ms": round(requests[-1].get("prepare_total_ms", 0), 1) if requests else 0,
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
            "total_tool_calls": len(tool_intercepts),
            "tool_calls_by_name": {
                name: sum(1 for tc in tool_intercepts if tc.get("tool_name", "unknown") == name)
                for name in {tc.get("tool_name", "unknown") for tc in tool_intercepts}
            },
            "recent_tool_calls": [
                {
                    "tool_name": tc.get("tool_name"),
                    "tool_input": tc.get("tool_input"),
                    "result": (tc.get("tool_result", "") or "")[:200],
                    "duration_ms": tc.get("duration_ms", 0),
                    "turn": tc.get("turn"),
                    "conversation_id": tc.get("conversation_id", ""),
                    "group_id": tc.get("group_id", ""),
                    "timestamp": tc.get("ts", ""),
                }
                for tc in tool_intercepts[-20:]
            ],
            "compaction_progress": list(compaction_progress),
            "cache": {
                "cache_read_tokens": last_cache_read,
                "cache_creation_tokens": last_cache_creation,
                "upstream_input_tokens": last_upstream_input,
                "cache_hit_ratio": last_cache_hit_ratio,
            },
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

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def snapshot_with_cursor(self) -> tuple[dict, int]:
        with self._lock:
            snap = self._snapshot_locked()
            return snap, self._seq - 1

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
        prepare_total_ms: float = 0,
        prepare_breakdown: dict | None = None,
        turns_dropped: int = 0,
        turns_stubbed: int = 0,
        message_preview: str = "",
        non_virtualizable_floor: int = 0,
        upstream_context_limit: int = 0,
        passthrough_trim_limit: int = 0,
        system_tokens: int = 0,
        protected_turn_tokens: int = 0,
        protected_turn_count: int = 0,
    ) -> None:
        """Capture raw request body for inspection (thread-safe, ring buffer)."""
        with self._lock:
            conv_id = self._capture_conversation_id(conversation_id)
            payload = {
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
                "prepare_total_ms": prepare_total_ms,
                "prepare_breakdown": dict(prepare_breakdown or {}),
                "turns_dropped": turns_dropped,
                "turns_stubbed": turns_stubbed,
                "message_preview": message_preview,
                "non_virtualizable_floor": non_virtualizable_floor,
                "upstream_context_limit": upstream_context_limit,
                "passthrough_trim_limit": passthrough_trim_limit,
                "system_tokens": system_tokens,
                "protected_turn_tokens": protected_turn_tokens,
                "protected_turn_count": protected_turn_count,
            }
            existing = self._find_capture(turn, conv_id)
            if existing is not None:
                existing.clear()
                existing.update(payload)
            else:
                self._request_bodies.append(payload)
            if conv_id:
                self._persist_capture(turn, conversation_id=conv_id)
            else:
                self._persist_capture(turn)

    def capture_enriched(
        self,
        turn: int,
        body: dict,
        *,
        conversation_id: str | None = None,
    ) -> None:
        with self._lock:
            conv_id = self._capture_conversation_id(conversation_id) if conversation_id is not None else None
            req = self._find_capture(turn, conv_id)
            if req is not None:
                req["enriched"] = body
                if conv_id:
                    self._persist_capture(turn, conversation_id=conv_id)
                else:
                    self._persist_capture(turn)

    def capture_response(
        self, turn: int, body: dict, *,
        upstream_input_tokens: int = 0,
        upstream_output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        conversation_id: str | None = None,
    ) -> None:
        with self._lock:
            conv_id = self._capture_conversation_id(conversation_id) if conversation_id is not None else None
            req = self._find_capture(turn, conv_id)
            if req is not None:
                req["response"] = body
                if upstream_input_tokens:
                    req["upstream_input_tokens"] = upstream_input_tokens
                if upstream_output_tokens:
                    req["upstream_output_tokens"] = upstream_output_tokens
                if cache_creation_input_tokens:
                    req["cache_creation_input_tokens"] = cache_creation_input_tokens
                if cache_read_input_tokens:
                    req["cache_read_input_tokens"] = cache_read_input_tokens
                if conv_id:
                    self._persist_capture(turn, conversation_id=conv_id)
                else:
                    self._persist_capture(turn)

    def update_request_tags(
        self,
        turn: int,
        *,
        response_tags: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> None:
        with self._lock:
            conv_id = self._capture_conversation_id(conversation_id) if conversation_id is not None else None
            req = self._find_capture(turn, conv_id)
            if req is not None:
                if response_tags is not None:
                    req["response_tags"] = response_tags
                if conv_id:
                    self._persist_capture(turn, conversation_id=conv_id)
                else:
                    self._persist_capture(turn)

    def get_captured_request(
        self,
        turn: int,
        conversation_id: str | None = None,
    ) -> dict | None:
        with self._lock:
            conv_id = self._capture_conversation_id(conversation_id) if conversation_id is not None else None
            req = self._find_capture(turn, conv_id)
            return dict(req) if req is not None else None

    def get_captured_requests_summary(
        self,
        conversation_id: str | None = None,
    ) -> list[dict]:
        with self._lock:
            conv_id = self._capture_conversation_id(conversation_id) if conversation_id is not None else None
            return [
                _extract_summary(r)
                for r in self._request_bodies
                if conv_id is None
                or (r.get("conversation_id", "") or "") == conv_id
            ]

    def restore_request_captures(self, captures: list[dict]) -> None:
        with self._lock:
            existing_keys = {
                ((r.get("conversation_id", "") or ""), r["turn"])
                for r in self._request_bodies
            }
            for cap in captures:
                key = ((cap.get("conversation_id", "") or ""), cap.get("turn"))
                if key not in existing_keys:
                    self._request_bodies.append(cap)
                    existing_keys.add(key)

    def _persist_capture(
        self,
        turn: int,
        *,
        conversation_id: str | None = None,
    ) -> None:
        if not self._store:
            logger.warning("CAPTURE_PERSIST store=None turn=%s", turn)
            return
        conv_id = self._capture_conversation_id(conversation_id) if conversation_id is not None else None
        req = self._find_capture(turn, conv_id)
        if req is None:
            return
        try:
            summary = _extract_summary(req)
            self._store.save_request_capture(summary)
            logger.info(
                "CAPTURE_PERSIST turn=%s conv=%s ok",
                turn,
                summary.get("conversation_id", "")[:12],
            )
        except Exception:
            logger.warning("CAPTURE_PERSIST turn=%s FAILED", turn, exc_info=True)
