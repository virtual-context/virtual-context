"""Thread-safe event collector for the proxy dashboard."""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone


class ProxyMetrics:
    """Collects structured events from the proxy pipeline.

    Thread-safe: ``record()`` can be called from the background
    ThreadPoolExecutor that runs ``on_turn_complete``.
    """

    BASELINE_RATIO = 0.30  # typical summarization compression (3.3x)

    def __init__(self, context_window: int = 120_000) -> None:
        self.start_time: float = time.time()
        self.context_window: int = context_window
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._request_bodies: deque[dict] = deque(maxlen=50)

    def record(self, event: dict) -> None:
        """Append an event (thread-safe). Adds ``_seq`` and ``ts``."""
        with self._lock:
            event = dict(event)  # shallow copy to avoid caller mutation
            event["_seq"] = self._seq
            if "ts" not in event:
                event["ts"] = datetime.now(timezone.utc).isoformat()
            self._seq += 1
            self._events.append(event)

    def events_since(self, seq: int) -> list[dict]:
        """Return events with ``_seq`` > *seq*."""
        with self._lock:
            return [e for e in self._events if e["_seq"] > seq]

    def snapshot(self) -> dict:
        """Aggregate stats for the initial SSE load."""
        with self._lock:
            requests = [e for e in self._events if e.get("type") == "request"]
            compactions = [e for e in self._events if e.get("type") == "compaction"]
            turn_completes = [e for e in self._events if e.get("type") == "turn_complete"]
            ingestions = [e for e in self._events if e.get("type") == "history_ingestion"]
            responses = [e for e in self._events if e.get("type") == "response"]
            ingested_turns = [e for e in self._events if e.get("type") == "ingested_turn"]

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

            # Baseline simulation: naive system with compaction at window limit
            # System prompt is sent every turn by any system — use latest estimate
            system_tokens_per_turn = 0
            if requests:
                system_tokens_per_turn = requests[-1].get("system_tokens", 0)

            # Bootstrap baseline from ingested history
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
            total_baseline_input = cumulative_baseline

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
            }

    def capture_request(
        self,
        turn: int,
        body: dict,
        api_format: str,
        *,
        inbound_tags: list[str] | None = None,
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
            })

    def update_request_tags(
        self, turn: int, *, response_tags: list[str] | None = None,
    ) -> None:
        """Update tags on a previously captured request (thread-safe)."""
        with self._lock:
            for req in self._request_bodies:
                if req["turn"] == turn:
                    if response_tags is not None:
                        req["response_tags"] = response_tags
                    return

    def get_captured_request(self, turn: int) -> dict | None:
        """Return the full captured request for a specific turn."""
        with self._lock:
            for req in self._request_bodies:
                if req["turn"] == turn:
                    return dict(req)
            return None

    def get_captured_requests_summary(self) -> list[dict]:
        """Return summaries (without full messages) for the list view."""
        with self._lock:
            return [{
                "turn": r["turn"],
                "ts": r["ts"],
                "api_format": r["api_format"],
                "model": r["model"],
                "message_count": r["message_count"],
            } for r in self._request_bodies]
