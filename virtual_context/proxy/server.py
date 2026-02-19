"""HTTP proxy server for virtual-context LLM enrichment.

Sits between any LLM client and an upstream provider (OpenAI or Anthropic),
transparently injecting <virtual-context> blocks into requests and capturing
assistant responses for on_turn_complete.

Usage:
    virtual-context -c config.yaml proxy --upstream https://api.anthropic.com
"""

from __future__ import annotations

import asyncio
import copy
import enum
import logging
import re
import sys
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..engine import VirtualContextEngine
from ..core.turn_tag_index import TurnTagIndex
from ..types import Message, SplitResult

from .dashboard import register_dashboard_routes
from .formats import (
    PayloadFormat,
    detect_format,
    get_format,
    _strip_openclaw_envelope as _strip_openclaw_envelope_fmt,
    _last_text_block as _last_text_block_fmt,
)
from .metrics import ProxyMetrics

logger = logging.getLogger(__name__)

_VC_PROMPT_MARKER = "[vc:prompt]\n"
# MemOS preamble: starts with "# Role", ends with this delimiter line (zero-width spaces)
_MEMOS_QUERY_DELIM = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b"

# Session marker: injected into assistant responses, extracted from inbound history
_VC_SESSION_RE = re.compile(r"<!-- vc:session=([a-f0-9-]+) -->")

# OpenClaw envelope patterns — consistent across all channels
_VC_USER_RE = re.compile(r"^\[vc:user\](.*?)\[/vc:user\]", re.DOTALL)
_SYSTEM_EVENT_RE = re.compile(r"^(?:System:\s*\[[^\]]*\][^\n]*\n+)+")
_CHANNEL_HEADER_RE = re.compile(r"^\[[A-Z][a-zA-Z]*\s[^\]]*\bid:-?\d+\b[^\]]*\]\s*")
_MESSAGE_ID_RE = re.compile(r"\n?\[message_id:\s*\d+\]\s*$")

_HOP_BY_HOP = frozenset({
    "host", "connection", "transfer-encoding", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "upgrade", "content-length",
})


# ---------------------------------------------------------------------------
# SessionState — state machine for non-blocking ingestion
# ---------------------------------------------------------------------------

class SessionState(enum.Enum):
    PASSTHROUGH = "passthrough"  # forwarding without enrichment (ingestion pending/running)
    INGESTING = "ingesting"     # background thread tagging historical turns
    ACTIVE = "active"           # normal enrichment mode


class _IngestionCancelled(Exception):
    """Raised inside progress callback to abort a running ingestion."""
    def __init__(self, done: int, total: int) -> None:
        self.done = done
        self.total = total
        super().__init__(f"Cancelled at {done}/{total}")


# ---------------------------------------------------------------------------
# ProxyState — mirrors HeadlessRunner threading pattern
# ---------------------------------------------------------------------------

class ProxyState:
    """Shared mutable state for the proxy lifetime."""

    def __init__(
        self,
        engine: VirtualContextEngine,
        metrics: ProxyMetrics | None = None,
        upstream: str = "",
    ) -> None:
        self.engine = engine
        self.conversation_history: list[Message] = []
        self.metrics = metrics
        self.upstream = upstream
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending_complete: Future | None = None
        self._ingested_sessions: set[str] = set()
        self._ingestion_lock = threading.Lock()
        self._compaction_lock = threading.Lock()
        # State machine for non-blocking ingestion
        self._state = SessionState.ACTIVE
        self._latest_body: dict | None = None
        self._ingestion_progress: tuple[int, int] = (0, 0)
        self._manual_passthrough = False
        self._ingestion_thread: threading.Thread | None = None
        self._ingestion_cancel = threading.Event()
        # Initial snapshot: captured at first ingestion start for growth tracking
        self._initial_turns: int | None = None
        self._initial_tag_count: int | None = None
        # Payload size tracking (KB + tokens)
        self._initial_payload_kb: float | None = None
        self._last_payload_kb: float = 0.0
        self._last_enriched_payload_kb: float = 0.0
        self._initial_payload_tokens: int | None = None
        self._last_payload_tokens: int = 0
        self._last_enriched_payload_tokens: int = 0

    @property
    def turn_offset(self) -> int:
        """Starting turn number from persisted engine state.

        When the proxy restarts, conversation_history is empty but the
        TurnTagIndex may have been restored from the store.  Use the
        highest indexed turn + 1 as the offset so turn numbering
        continues from the previous session.
        """
        try:
            entries = self.engine._turn_tag_index.entries
            if entries and len(entries) > 0:
                return max(e.turn_number for e in entries) + 1
        except (TypeError, AttributeError):
            pass
        return 0

    @property
    def session_state(self) -> SessionState:
        """Current session state, accounting for manual passthrough override."""
        if self._manual_passthrough:
            return SessionState.PASSTHROUGH
        return self._state

    def set_manual_passthrough(self, enabled: bool) -> None:
        """Toggle manual passthrough mode from the dashboard."""
        self._manual_passthrough = enabled

    def _transition_to(self, new_state: SessionState) -> None:
        """Update internal state and emit a metric event."""
        old = self._state
        self._state = new_state
        if self.metrics and old != new_state:
            self.metrics.record({
                "type": "session_state_change",
                "from": old.value,
                "to": new_state.value,
                "session_id": self.engine.config.session_id,
            })
        logger.info(
            "Session %s: %s → %s",
            self.engine.config.session_id[:12], old.value, new_state.value,
        )

    def live_snapshot(self) -> dict:
        """Build a snapshot dict of this session's live state for the dashboard."""
        engine = self.engine
        idx = engine._turn_tag_index

        # KB stats: tag summaries
        tag_summary_count = 0
        tag_summary_tokens = 0
        try:
            summaries = engine._store.get_all_tag_summaries()
            tag_summary_count = len(summaries)
            tag_summary_tokens = sum(ts.summary_tokens for ts in summaries)
        except Exception:
            pass

        # Estimate history size in tokens (chars / 4)
        history_tokens = 0
        for m in self.conversation_history:
            history_tokens += len(m.content) // 4

        context_window = engine.config.monitor.context_window
        utilization_pct = round(history_tokens / context_window * 100, 1) if context_window > 0 else 0

        # Distinct tag count from TurnTagIndex
        all_tags: set[str] = set()
        for entry in idx.entries:
            all_tags.update(entry.tags)
        all_tags.discard("_general")

        snap = {
            "session_id": engine.config.session_id,
            "turn_count": len(self.conversation_history) // 2,
            "compacted_through": getattr(engine, "_compacted_through", 0),
            "tag_count": len(idx.entries),
            "distinct_tags": len(all_tags),
            "active_tags": list(idx.get_active_tags(lookback=6)),
            "session_state": self.session_state.value,
            "ingestion_progress": list(self._ingestion_progress),
            "manual_passthrough": self._manual_passthrough,
            "context_window": context_window,
            "history_tokens": history_tokens,
            "utilization_pct": utilization_pct,
            "tag_summary_count": tag_summary_count,
            "tag_summary_tokens": tag_summary_tokens,
            "initial_turns": self._initial_turns,
            "initial_tag_count": self._initial_tag_count,
            "initial_payload_kb": self._initial_payload_kb,
            "last_payload_kb": self._last_payload_kb,
            "last_enriched_payload_kb": self._last_enriched_payload_kb,
            "initial_payload_tokens": self._initial_payload_tokens,
            "last_payload_tokens": self._last_payload_tokens,
            "last_enriched_payload_tokens": self._last_enriched_payload_tokens,
        }
        return snap

    def wait_for_complete(self) -> None:
        """Block until the pending on_turn_complete finishes."""
        if self._pending_complete is not None:
            self._pending_complete.result()
            self._pending_complete = None

    def fire_turn_complete(
        self,
        history_snapshot: list[Message],
        payload_tokens: int | None = None,
    ) -> None:
        """Submit on_turn_complete to background thread."""
        self._pending_complete = self._pool.submit(
            self._run_turn_complete, history_snapshot, payload_tokens,
        )

    def _run_turn_complete(
        self,
        history: list[Message],
        payload_tokens: int | None = None,
    ) -> None:
        t0 = time.monotonic()
        turn = len(history) // 2 - 1
        session_id = self.engine.config.session_id
        try:
            report = self.engine.on_turn_complete(
                history, payload_tokens=payload_tokens,
            )

            complete_ms = round((time.monotonic() - t0) * 1000, 1)
            entry = self.engine._turn_tag_index.get_tags_for_turn(turn)
            _tags = entry.tags if entry else []
            _primary = entry.primary_tag if entry else ""
            print(
                f"[T{turn}] COMPLETE {int(complete_ms)}ms "
                f"tags=[{', '.join(_tags)}] primary={_primary}"
                + (f" COMPACTION freed={report.tokens_freed}t" if report else "")
            )
            logger.info(
                "T%d complete (%dms) session=%s compacted_through=%d history=%d%s",
                turn, int(complete_ms), session_id[:12],
                getattr(self.engine, "_compacted_through", 0),
                len(history),
                " COMPACTION" if report else "",
            )

            if report is not None:
                logger.info(
                    "  compaction: %d segments, freed %d tokens, tags=%s, "
                    "summaries_built=%d",
                    report.segments_compacted,
                    report.tokens_freed,
                    report.tags,
                    report.tag_summaries_built,
                )

            # Emit turn_complete event
            if self.metrics:
                entry = self.engine._turn_tag_index.get_tags_for_turn(turn)
                active_tags = list(
                    self.engine._turn_tag_index.get_active_tags(lookback=6)
                )
                turn_pair_tokens = (
                    sum(len(m.content) for m in history[-2:]) // 4
                    if len(history) >= 2 else 0
                )
                # Write response tags to captured request
                response_tags = entry.tags if entry else []
                self.metrics.update_request_tags(
                    turn, response_tags=response_tags,
                )
                self.metrics.record({
                    "type": "turn_complete",
                    "turn": turn,
                    "tags": entry.tags if entry else [],
                    "primary_tag": entry.primary_tag if entry else "",
                    "complete_ms": complete_ms,
                    "active_tags": active_tags,
                    "store_tag_count": len(self.engine._store.get_all_tags()),
                    "turn_pair_tokens": turn_pair_tokens,
                    "session_id": session_id,
                })

                # Emit tag split event if splitting occurred
                split_result = getattr(self.engine, '_last_split_result', None)
                if isinstance(split_result, SplitResult):
                    if split_result.splittable:
                        new_tags = list(split_result.groups.keys())
                        print(
                            f"[T{turn}] SPLIT \"{split_result.tag}\" → "
                            f"{new_tags} ({sum(len(v) for v in split_result.groups.values())} turns)"
                        )
                    else:
                        print(
                            f"[T{turn}] SUMMARIZED \"{split_result.tag}\" "
                            f"(unsplittable: {split_result.reason})"
                        )
                    self.metrics.record({
                        "type": "tag_split",
                        "turn": turn,
                        "tag": split_result.tag,
                        "splittable": split_result.splittable,
                        "new_tags": list(split_result.groups.keys()) if split_result.splittable else [],
                        "session_id": session_id,
                    })
                    self.engine._last_split_result = None  # consume

                # Emit compaction event if compaction occurred
                if report is not None:
                    original_tokens = sum(
                        r.original_tokens for r in report.results
                    )
                    summary_tokens = sum(
                        r.summary_tokens for r in report.results
                    )
                    self.metrics.record({
                        "type": "compaction",
                        "turn": turn,
                        "segments": report.segments_compacted,
                        "tokens_freed": report.tokens_freed,
                        "original_tokens": original_tokens,
                        "summary_tokens": summary_tokens,
                        "tags": report.tags,
                        "tag_summaries_built": report.tag_summaries_built,
                        "compacted_through": getattr(
                            self.engine, "_compacted_through", 0
                        ),
                        "session_id": session_id,
                    })
        except Exception as e:
            logger.error("on_turn_complete error: %s", e, exc_info=True)

    def _history_ingested(self) -> bool:
        """Whether the current session's history has been ingested."""
        return self.engine.config.session_id in self._ingested_sessions

    def ingest_if_needed(self, history_pairs: list[Message]) -> None:
        """Bootstrap TurnTagIndex from pre-existing history (once per session).

        Double-checked locking: fast path skips the lock entirely.
        """
        session_id = self.engine.config.session_id
        if session_id in self._ingested_sessions:
            return
        with self._ingestion_lock:
            if session_id in self._ingested_sessions:
                return
            t0 = time.monotonic()
            turns = self.engine.ingest_history(history_pairs)
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            self._ingested_sessions.add(session_id)

            print(
                f"[INGEST] {turns} turns in {int(elapsed_ms)}ms "
                f"(session={session_id[:12]})"
            )
            logger.info(
                "History ingestion: %d turns in %dms (session=%s)",
                turns, int(elapsed_ms), session_id[:12],
            )

            if self.metrics:
                # Emit per-turn events so the dashboard grid shows history
                baseline_history_tokens = 0
                for i in range(0, len(history_pairs) - 1, 2):
                    turn_num = i // 2
                    entry = self.engine._turn_tag_index.get_tags_for_turn(
                        turn_num,
                    )
                    raw_content = history_pairs[i].content
                    preview = _strip_openclaw_envelope(raw_content)[:60]
                    # Estimate turn pair tokens for baseline calculation
                    pair_chars = len(history_pairs[i].content) + len(history_pairs[i + 1].content)
                    tpt = pair_chars // 4
                    baseline_history_tokens += tpt
                    self.metrics.record({
                        "type": "ingested_turn",
                        "turn": turn_num,
                        "tags": entry.tags if entry else [],
                        "primary_tag": entry.primary_tag if entry else "",
                        "message_preview": preview,
                        "turn_pair_tokens": tpt,
                        "session_id": session_id,
                    })
                self.metrics.record({
                    "type": "history_ingestion",
                    "turns_ingested": turns,
                    "pairs_received": len(history_pairs) // 2,
                    "elapsed_ms": elapsed_ms,
                    "session_id": session_id,
                    "baseline_history_tokens": baseline_history_tokens,
                })

    # ------------------------------------------------------------------
    # Non-blocking ingestion (background thread)
    # ------------------------------------------------------------------

    def start_ingestion_if_needed(self, history_pairs: list[Message]) -> None:
        """Start non-blocking history ingestion in a background thread.

        Returns immediately — the session stays in INGESTING while the
        background thread tags historical turns.  If called while ingestion
        is already running, cancels the old thread and resumes from the
        last tagged turn (PROXY-013).
        """
        session_id = self.engine.config.session_id
        if session_id in self._ingested_sessions:
            return
        with self._ingestion_lock:
            if session_id in self._ingested_sessions:
                return

            if not history_pairs:
                self._ingested_sessions.add(session_id)
                return

            # Skip if persisted TurnTagIndex already covers history
            existing_turns = len(self.engine._turn_tag_index.entries)
            needed_turns = len(history_pairs) // 2
            if existing_turns >= needed_turns:
                self._ingested_sessions.add(session_id)
                logger.info(
                    "Skipping ingestion: persisted index (%d) covers history (%d)",
                    existing_turns, needed_turns,
                )
                return

            # ---- PROXY-013: cancel-and-resume if already running ----
            if (
                self._ingestion_thread is not None
                and self._ingestion_thread.is_alive()
            ):
                done, total = self._ingestion_progress
                logger.info(
                    "Cancelling running ingestion at turn %d/%d "
                    "(new request has %d pairs)",
                    done, total, needed_turns,
                )
                self._ingestion_cancel.set()
                self._ingestion_thread.join(timeout=5.0)
                if self._ingestion_thread.is_alive():
                    logger.warning("Old ingestion thread did not exit in 5s")
                # Reset cancel event for the new thread
                self._ingestion_cancel.clear()

                # Re-read existing_turns AFTER old thread stopped —
                # the thread may have appended one more entry between
                # the last callback and the cancel taking effect.
                existing_turns = len(self.engine._turn_tag_index.entries)

                print(
                    f"[INGEST] Cancel at T{done}/{total}, "
                    f"resuming from T{existing_turns} "
                    f"(session={session_id[:12]})"
                )

                # Verify hash at handoff point
                self._verify_handoff_hash(history_pairs, existing_turns)

                # Slice to remaining pairs only
                history_pairs = list(history_pairs[existing_turns * 2:])
                if not history_pairs:
                    self._ingested_sessions.add(session_id)
                    self._transition_to(SessionState.ACTIVE)
                    return
                needed_turns = len(history_pairs) // 2 + existing_turns

            total = needed_turns
            self._ingestion_progress = (existing_turns, total)

            # Capture initial snapshot once (first ingestion start only)
            if self._initial_turns is None:
                self._initial_turns = existing_turns
                self._initial_tag_count = len(self.engine._turn_tag_index.entries)

            self._transition_to(SessionState.INGESTING)

            # Separate daemon thread (not the _pool) so on_turn_complete
            # can use the pool once we transition to ACTIVE.
            self._ingestion_thread = threading.Thread(
                target=self._run_ingestion_with_catchup,
                args=(list(history_pairs),),
                daemon=True,
                name="vc-ingest",
            )
            self._ingestion_thread.start()

    def _verify_handoff_hash(
        self, new_pairs: list[Message], handoff_turn: int,
    ) -> None:
        """Verify the last tagged turn matches the same content in new history.

        Logs a warning if the hash doesn't match — indicates potential data
        loss or history divergence between requests.
        """
        import hashlib as _hl
        if handoff_turn <= 0:
            return
        prev_turn = handoff_turn - 1
        entry = self.engine._turn_tag_index.get_tags_for_turn(prev_turn)
        if entry is None:
            return
        pair_idx = prev_turn * 2
        if pair_idx + 1 >= len(new_pairs):
            logger.warning(
                "Handoff verification: turn %d not in new history "
                "(new history has %d pairs) — potential data loss",
                prev_turn, len(new_pairs) // 2,
            )
            return
        combined = f"{new_pairs[pair_idx].content} {new_pairs[pair_idx + 1].content}"
        new_hash = _hl.sha256(combined.encode()).hexdigest()[:16]
        if new_hash != entry.message_hash:
            logger.warning(
                "Handoff hash MISMATCH at turn %d: "
                "indexed=%s new=%s — history may have diverged",
                prev_turn, entry.message_hash, new_hash,
            )
            print(
                f"[INGEST] WARNING: hash mismatch at T{prev_turn} "
                f"(indexed={entry.message_hash} vs new={new_hash})"
            )
        else:
            logger.info(
                "Handoff hash verified at turn %d: %s",
                prev_turn, new_hash,
            )

    def _run_ingestion_with_catchup(self, initial_pairs: list[Message]) -> None:
        """Background thread: ingest initial pairs, then catch up any gap."""
        session_id = self.engine.config.session_id
        cancelled = False
        try:
            # Phase 1: tag all initial history
            self._ingest_pairs_with_progress(initial_pairs)

            # Phase 2: catch-up loop — tag any turns that arrived during ingestion
            for _ in range(10):  # bounded to avoid infinite loops
                if self._ingestion_cancel.is_set():
                    break
                latest = self._latest_body
                if latest is None:
                    break
                latest_pairs = _extract_history_pairs(latest)
                needed = len(latest_pairs) // 2
                have = len(self.engine._turn_tag_index.entries)
                if needed <= have:
                    break
                # Tag the gap
                gap_pairs = latest_pairs[have * 2:]
                if not gap_pairs:
                    break
                logger.info(
                    "Ingestion catch-up: %d gap turns (have=%d, need=%d)",
                    len(gap_pairs) // 2, have, needed,
                )
                self._ingest_pairs_with_progress(gap_pairs)

        except _IngestionCancelled as e:
            # New request is taking over — exit cleanly without
            # transitioning to ACTIVE or marking as ingested.
            # NOTE: Python's finally block runs even after return,
            # so we use a flag to skip the finalization actions.
            cancelled = True
            logger.info("Ingestion cancelled at %d/%d", e.done, e.total)
        except Exception as e:
            logger.error("Ingestion error: %s", e, exc_info=True)
        finally:
            if not cancelled:
                self._ingested_sessions.add(session_id)
                self._transition_to(SessionState.ACTIVE)

    def _ingest_pairs_with_progress(self, pairs: list[Message]) -> None:
        """Call engine.ingest_history with a progress callback that emits events.

        Raises ``_IngestionCancelled`` if ``_ingestion_cancel`` is set.
        """
        session_id = self.engine.config.session_id
        t0 = time.monotonic()
        baseline_history_tokens = 0

        def on_progress(done: int, total: int, entry) -> None:
            nonlocal baseline_history_tokens
            # Check cancellation before updating progress
            if self._ingestion_cancel.is_set():
                raise _IngestionCancelled(done, total)
            self._ingestion_progress = (done, total)
            if self.metrics:
                # Find the pair for this turn to compute preview + tokens
                turn_num = entry.turn_number
                pair_idx = turn_num * 2
                preview = ""
                tpt = 0
                if pair_idx < len(pairs):
                    raw_content = pairs[pair_idx].content
                    preview = _strip_openclaw_envelope(raw_content)[:60]
                    if pair_idx + 1 < len(pairs):
                        pair_chars = len(pairs[pair_idx].content) + len(pairs[pair_idx + 1].content)
                        tpt = pair_chars // 4
                        baseline_history_tokens += tpt
                self.metrics.record({
                    "type": "ingested_turn",
                    "turn": turn_num,
                    "tags": entry.tags if entry else [],
                    "primary_tag": entry.primary_tag if entry else "",
                    "message_preview": preview,
                    "turn_pair_tokens": tpt,
                    "session_id": session_id,
                    "done": done,
                    "total": total,
                })

        turns = self.engine.ingest_history(pairs, progress_callback=on_progress)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        print(
            f"[INGEST] {turns} turns in {int(elapsed_ms)}ms "
            f"(session={session_id[:12]})"
        )
        logger.info(
            "History ingestion: %d turns in %dms (session=%s)",
            turns, int(elapsed_ms), session_id[:12],
        )

        if self.metrics:
            self.metrics.record({
                "type": "history_ingestion",
                "turns_ingested": turns,
                "pairs_received": len(pairs) // 2,
                "elapsed_ms": elapsed_ms,
                "session_id": session_id,
                "baseline_history_tokens": baseline_history_tokens,
            })

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# SessionRegistry — multi-session routing
# ---------------------------------------------------------------------------

class SessionRegistry:
    """Manages multiple concurrent ProxyState instances, one per session.

    Routing priority:
    1. Session marker (``<!-- vc:session=UUID -->``) in assistant messages
    2. Trailing fingerprint — hash of last N user messages (before current
       turn) in the request body, matched against in-memory or persisted
       fingerprints from previous sessions
    3. Fallback — claim unclaimed session or create new

    Trailing fingerprints survive client-side compaction (which rewrites
    early messages) because they sample from the tail of the history.

    Future: ``X-VC-Session`` request header overrides all (requires client changes).
    """

    _FINGERPRINT_SAMPLE_SIZE = 1  # last N user messages to hash

    def __init__(
        self,
        config_path: str | None,
        upstream: str,
        metrics: ProxyMetrics,
        *,
        store: "Store | None" = None,
    ) -> None:
        self._config_path = config_path
        self._upstream = upstream
        self._metrics = metrics
        self._sessions: dict[str, ProxyState] = {}
        self._fingerprints: dict[str, str] = {}  # fingerprint → session_id
        self._lock = threading.Lock()
        self._store = store  # for loading persisted fingerprints on restart

    @staticmethod
    def _msg_text(msg: dict) -> str:
        """Extract plain text from a message content (str or content blocks)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return ""

    @staticmethod
    def _compute_fingerprint(body: dict, offset: int = 0) -> str:
        """Trailing conversation fingerprint from the last N user messages.

        Hashes ``_FINGERPRINT_SAMPLE_SIZE`` (S) user messages sampled from
        the tail of the history (excluding the current turn).

        The ``offset`` parameter shifts the sampling window back from the
        tail.  Two conventions:

        - **offset=0** (store): hash the last S history messages.
          Used in ``catch_all`` to persist the fingerprint after each turn.
        - **offset=1** (match): hash S messages ending one position before
          the tail.  Used in ``get_or_create`` and restart matching.

        Why offset=1 matches offset=0 from the previous request:

        Request N  has history [u0 … u49].  Store fp = hash(u49).
        Request N+1 has history [u0 … u49, u50].  Match fp = hash(u49).
        The Nth message from the tail of request N+1 is the (N-1)th from
        request N — the tail is stable, only the tip grows.
        """
        messages = body.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]

        # Exclude the current turn (last user message)
        if len(user_msgs) < 2:
            return ""
        history_user = user_msgs[:-1]  # all except current turn

        n = SessionRegistry._FINGERPRINT_SAMPLE_SIZE
        # Apply offset: shift window back by `offset` positions
        end = len(history_user) - offset
        start = end - n
        if start < 0 or end <= 0:
            return ""
        sample = history_user[start:end]
        if not sample:
            return ""

        texts = [SessionRegistry._msg_text(m) for m in sample]
        combined = "\n".join(texts)
        if not combined.strip():
            return ""

        import hashlib
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _match_persisted_fingerprint(self, body: dict) -> str | None:
        """Match inbound request against persisted session fingerprints.

        Compares the request's tail-1 fingerprint (offset=1) against stored
        tail fingerprints (offset=0).  The one-turn shift between the last
        save and the next inbound request is exactly bridged by offset=1.

        Returns the matched session_id, or None.
        """
        if not self._store:
            return None
        try:
            persisted = self._store.list_engine_state_fingerprints()
            if not isinstance(persisted, dict) or not persisted:
                return None
        except Exception:
            return None

        fp = self._compute_fingerprint(body, offset=1)
        if fp and fp in persisted:
            matched = persisted[fp]
            if isinstance(matched, str) and matched:
                logger.info(
                    "Persisted fingerprint match: fp=%s → session=%s",
                    fp[:8], matched[:12],
                )
                return matched
        return None

    def get_or_create(
        self,
        session_id: str | None,
        *,
        body: dict | None = None,
    ) -> tuple[ProxyState, bool]:
        """Look up or create a ProxyState for the given session ID.

        Returns (state, is_new).

        Routing priority: marker (session_id) > in-memory fingerprint >
        persisted fingerprint > claim unclaimed session > create new session.
        """
        # Fast path: session marker found and session already in memory
        if session_id and session_id in self._sessions:
            return self._sessions[session_id], False

        # Compute tail-1 fingerprint for matching (offset=1).
        # The incoming request's tail-1 matches the previous request's tail
        # because the conversation grew by exactly one turn.  catch_all
        # stores each request's tail (offset=0) in _fingerprints, so the
        # match here uses offset=1 to align with the stored value.
        fp_match = ""  # offset=1 — for matching against stored tail
        fp_store = ""  # offset=0 — saved in _fingerprints after session created
        if session_id is None and body is not None:
            fp_match = self._compute_fingerprint(body, offset=1)
            fp_store = self._compute_fingerprint(body)
            # Fast path: in-memory fingerprint match
            if fp_match and fp_match in self._fingerprints:
                matched_sid = self._fingerprints[fp_match]
                if matched_sid in self._sessions:
                    return self._sessions[matched_sid], False

        with self._lock:
            # Double-check after acquiring lock
            if session_id and session_id in self._sessions:
                return self._sessions[session_id], False

            if fp_match and fp_match in self._fingerprints:
                matched_sid = self._fingerprints[fp_match]
                if matched_sid in self._sessions:
                    return self._sessions[matched_sid], False

            # Check persisted fingerprints (survives proxy restart +
            # client-side compaction that destroys session markers)
            if session_id is None and body is not None:
                persisted_sid = self._match_persisted_fingerprint(body)
                if persisted_sid:
                    session_id = persisted_sid

            # No marker, no fingerprint match — claim an unclaimed session
            # if one exists.  This handles the startup case: create_app makes
            # a default session before any request arrives.  The first
            # conversation claims it; a second distinct conversation creates
            # a new session.
            if session_id is None:
                claimed_sids = set(self._fingerprints.values())
                for sid, st in self._sessions.items():
                    if sid not in claimed_sids:
                        if fp_store:
                            self._fingerprints[fp_store] = sid
                        logger.info(
                            "Session claimed: %s (fp=%s, total=%d)",
                            sid[:12],
                            fp_store[:8] if fp_store else "none",
                            len(self._sessions),
                        )
                        return st, False

            # Create a new engine instance
            engine = VirtualContextEngine(config_path=self._config_path)

            if session_id:
                # Override the auto-generated session_id so load_engine_state
                # can find the persisted state for this session.
                engine.config.session_id = session_id
                # Trigger state reload (engine.__init__ already called
                # _load_persisted_state but with the wrong session_id).
                engine._load_persisted_state()
                engine._bootstrap_vocabulary()

            actual_id = engine.config.session_id
            state = ProxyState(
                engine, metrics=self._metrics, upstream=self._upstream,
            )
            self._sessions[actual_id] = state

            # Record fingerprint → session mapping (offset=0 = tail)
            if fp_store:
                self._fingerprints[fp_store] = actual_id

            logger.info(
                "Session %s: %s (fp=%s, total=%d)",
                "resumed" if session_id else "created",
                actual_id[:12],
                fp_store[:8] if fp_store else "none",
                len(self._sessions),
            )
            return state, True

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def shutdown_all(self) -> None:
        """Shut down all session states."""
        for state in self._sessions.values():
            state.shutdown()
        self._sessions.clear()


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no side effects)
# ---------------------------------------------------------------------------

def _detect_api_format(body: dict) -> str:
    """Detect whether this is an Anthropic, OpenAI, or Gemini request.

    Delegates to ``detect_format()`` and returns the format name string
    for backward compatibility.
    """
    return detect_format(body).name


def _last_text_block(content: list) -> str:
    """Return the text of the last ``type: "text"`` block in *content*.

    LLM clients often place extended-thinking or system-level content in
    earlier text blocks while the actual conversational content occupies the
    final text block.  Extracting only the last block filters out that noise
    without relying on client-specific markers or heuristics.
    """
    for block in reversed(content):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def _strip_vc_prompt(text: str) -> str:
    """Strip the ``[vc:prompt]`` marker injected by the OpenClaw plugin.

    Returns the text with the marker removed.  If no marker is present,
    returns the original text unchanged.
    """
    if text.startswith(_VC_PROMPT_MARKER):
        return text[len(_VC_PROMPT_MARKER):]
    return text


def _strip_openclaw_envelope(text: str) -> str:
    """Strip OpenClaw channel metadata from a message.

    Handles (in order):

    1. ``[vc:prompt]`` marker from the virtual-context-tagger plugin
    2. ``[vc:user]...[/vc:user]`` backward-compatible wrapper (extracts
       inner content and returns immediately — inner content is already clean)
    3. ``System: [TIMESTAMP] event`` lines prepended by OpenClaw
    4. ``[ChannelName ... id:NNN ...] `` header (Telegram, WhatsApp, etc.)
    5. ``[message_id: NNN]`` footer

    Returns the actual conversational content with all metadata removed.
    """
    if not text:
        return text

    # 1. Strip [vc:prompt] marker and any trailing whitespace
    if text.startswith(_VC_PROMPT_MARKER):
        text = text[len(_VC_PROMPT_MARKER):].lstrip()

    # 1b. Strip MemOS preamble: "# Role ... user原始query：" → keep only content after delimiter
    if text.startswith("# Role"):
        idx = text.find(_MEMOS_QUERY_DELIM)
        if idx != -1:
            text = text[idx + len(_MEMOS_QUERY_DELIM):].lstrip()

    # 2. Handle [vc:user]...[/vc:user] — inner content is already clean
    m = _VC_USER_RE.match(text)
    if m:
        return m.group(1).strip()

    # 3. Strip System: [...] event lines
    text = _SYSTEM_EVENT_RE.sub("", text)

    # 4. Strip channel header  [ChannelName ... id:NNN ...]
    text = _CHANNEL_HEADER_RE.sub("", text)

    # 5. Strip [message_id: NNN] footer
    text = _MESSAGE_ID_RE.sub("", text)

    return text.strip()


def _extract_session_id(body: dict) -> str | None:
    """Scan assistant messages for vc:session marker. Returns UUID or None.

    Delegates to the detected ``PayloadFormat``.
    """
    fmt = detect_format(body)
    return fmt.extract_session_id(body)


def _strip_session_markers(body: dict) -> dict:
    """Strip vc:session markers from all assistant messages in the request body.

    Delegates to the detected ``PayloadFormat``.
    """
    fmt = detect_format(body)
    return fmt.strip_session_markers(body)


def _extract_user_message(body: dict) -> str:
    """Extract the last user message text from a request body.

    Delegates to the detected ``PayloadFormat``.
    """
    fmt = detect_format(body)
    return fmt.extract_user_message(body)


def _extract_message_text(msg: dict) -> str:
    """Extract text from a single message dict (string or content blocks).

    Strips OpenClaw envelope metadata, then uses last-text-block for arrays.
    Uses AnthropicFormat as default (works for both Anthropic and OpenAI
    since they share the same content structure).
    """
    # Both Anthropic and OpenAI share the same message structure
    return get_format("anthropic").extract_message_text(msg)


def _extract_history_pairs(body: dict) -> list[Message]:
    """Extract complete user+assistant pairs from request history.

    Delegates to the detected ``PayloadFormat``.
    """
    fmt = detect_format(body)
    return fmt.extract_history_pairs(body)


def _inject_context(body: dict, prepend_text: str, api_format: str) -> dict:
    """Inject <virtual-context> block into a deep-copied request body.

    Delegates to the appropriate ``PayloadFormat`` based on *api_format*.
    """
    return get_format(api_format).inject_context(body, prepend_text)


def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter out hop-by-hop headers for forwarding."""
    return {
        k: v for k, v in headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


def _filter_body_messages(
    body: dict,
    turn_tag_index: TurnTagIndex,
    matched_tags: list[str],
    *,
    recent_turns: int = 3,
    broad: bool = False,
    temporal: bool = False,
    compacted_turn: int = 0,
    fmt: PayloadFormat | None = None,
) -> tuple[dict, int]:
    """Filter request body messages to remove irrelevant history turns.

    Operates on the raw API body, preserving original message format
    (content blocks, metadata, etc.).  Uses the TurnTagIndex to decide
    which user+assistant pairs to keep based on tag overlap.

    When *compacted_turn* > 0 (paging active), pairs whose turn index is
    below the watermark are unconditionally dropped — their content lives
    in VC summaries and is retrievable via ``vc_expand_topic``.

    Returns (filtered_body, turns_dropped).
    """
    if fmt is None:
        fmt = detect_format(body)

    # Determine message key and assistant role based on format
    if fmt.name == "gemini":
        _msg_key = "contents"
        _asst_role = "model"
    else:
        _msg_key = "messages"
        _asst_role = "assistant"

    messages = body.get(_msg_key, [])
    if not messages:
        return body, 0

    # Separate system messages (OpenAI format) and chat messages
    prefix: list[dict] = []
    chat_msgs: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and not chat_msgs:
            prefix.append(msg)
        else:
            chat_msgs.append(msg)

    if not chat_msgs:
        return body, 0

    # Split trailing user message (current turn) from history pairs
    current_user = None
    if chat_msgs and chat_msgs[-1].get("role") == "user":
        current_user = chat_msgs[-1]
        chat_msgs = chat_msgs[:-1]

    # Group into user+assistant pairs, tracking which message indices are paired.
    # Unpaired messages (tool_results between consecutive users, batched messages,
    # etc.) are always kept — they're structural and may be required by the API.
    pairs: list[tuple[int, int]] = []  # (msg_idx_user, msg_idx_assistant)
    paired_indices: set[int] = set()
    i = 0
    while i + 1 < len(chat_msgs):
        if (chat_msgs[i].get("role") == "user"
                and chat_msgs[i + 1].get("role") == _asst_role):
            pairs.append((i, i + 1))
            paired_indices.add(i)
            paired_indices.add(i + 1)
            i += 2
        else:
            i += 1

    total_pairs = len(pairs)
    protected = min(recent_turns, total_pairs)

    if total_pairs <= protected or not turn_tag_index.entries:
        return body, 0

    # Broad/temporal: keep everything
    if broad or temporal:
        return body, 0

    tag_set = set(matched_tags)

    # First pass: mark each pair as keep/drop based on tags
    keep_pair = [False] * total_pairs
    for pair_idx, (u_idx, a_idx) in enumerate(pairs):
        if pair_idx >= total_pairs - protected:
            keep_pair[pair_idx] = True
        elif compacted_turn > 0 and pair_idx < compacted_turn:
            # PROXY-023: paging active — compacted turns are dropped
            # unconditionally. Their content is in VC summaries and
            # retrievable via vc_expand_topic.
            keep_pair[pair_idx] = False
        else:
            entry = turn_tag_index.get_tags_for_turn(pair_idx)
            if entry is None:
                keep_pair[pair_idx] = True
            elif "rule" in entry.tags or set(entry.tags) & tag_set:
                keep_pair[pair_idx] = True

    # Second pass: fix tool_use/tool_result dependencies.
    # If assistant has tool_use, the next pair (with tool_result) must also be kept.
    # If user has tool_result, the previous pair (with tool_use) must also be kept.
    # Iterate until stable (handles multi-step tool chains).
    changed = True
    while changed:
        changed = False
        for pair_idx in range(total_pairs):
            if not keep_pair[pair_idx]:
                continue
            u_idx, a_idx = pairs[pair_idx]
            if _has_tool_use(chat_msgs[a_idx]) and pair_idx + 1 < total_pairs and not keep_pair[pair_idx + 1]:
                keep_pair[pair_idx + 1] = True
                changed = True
            if _has_tool_result(chat_msgs[u_idx]) and pair_idx > 0 and not keep_pair[pair_idx - 1]:
                keep_pair[pair_idx - 1] = True
                changed = True

    # Build per-message keep set: unpaired messages always kept, pairs based on filter
    keep_msg: set[int] = set()
    for msg_idx in range(len(chat_msgs)):
        if msg_idx not in paired_indices:
            keep_msg.add(msg_idx)  # always keep unpaired messages
    for pair_idx, (u_idx, a_idx) in enumerate(pairs):
        if keep_pair[pair_idx]:
            keep_msg.add(u_idx)
            keep_msg.add(a_idx)

    # Final tool chain safety: any kept assistant with tool_use must have its
    # tool_result in the immediately following message(s) also kept, and vice versa.
    changed = True
    while changed:
        changed = False
        for msg_idx in range(len(chat_msgs)):
            if msg_idx not in keep_msg:
                continue
            msg = chat_msgs[msg_idx]
            if msg.get("role") == _asst_role and _has_tool_use(msg):
                # Keep all following messages until we find the tool_result
                for j in range(msg_idx + 1, len(chat_msgs)):
                    if j not in keep_msg:
                        keep_msg.add(j)
                        changed = True
                    if _has_tool_result(chat_msgs[j]):
                        break
            if _has_tool_result(msg):
                # Keep all preceding messages back to the tool_use
                for j in range(msg_idx - 1, -1, -1):
                    if j not in keep_msg:
                        keep_msg.add(j)
                        changed = True
                    if chat_msgs[j].get("role") == _asst_role and _has_tool_use(chat_msgs[j]):
                        break

    # Build filtered message list preserving original order
    kept: list[dict] = list(prefix)
    for msg_idx in range(len(chat_msgs)):
        if msg_idx in keep_msg:
            kept.append(chat_msgs[msg_idx])

    if current_user:
        kept.append(current_user)

    # PROXY-022: Enforce strict role alternation.
    # OpenClaw can send consecutive same-role messages (batched Telegram messages,
    # tool_result followed by new user text without intervening assistant). When
    # we drop pairs around these "unpaired" messages the result can have
    # consecutive same-role entries, which the Anthropic API rejects.
    # Fix: walk the kept list and drop any message that repeats the previous role.
    alternating: list[dict] = []
    for msg in kept:
        if alternating and msg.get("role") == alternating[-1].get("role"):
            continue  # skip — would create consecutive same-role
        alternating.append(msg)
    kept = alternating

    # Compute drops from final kept list (not incremental tracking, which
    # undercounts when alternation enforcement removes additional messages).
    kept_user_count = sum(1 for m in kept if m.get("role") == "user")
    if current_user:
        kept_user_count -= 1  # exclude current turn from pair count
    dropped = max(0, total_pairs - kept_user_count)
    if dropped == 0:
        return body, 0

    body = dict(body)
    body[_msg_key] = kept
    return body, dropped


def _has_tool_use(msg: dict) -> bool:
    """Check if an assistant message contains tool_use blocks."""
    content = msg.get("content", [])
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
    return False


def _has_tool_result(msg: dict) -> bool:
    """Check if a user message contains tool_result blocks."""
    content = msg.get("content", [])
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _inject_session_marker(response_body: dict, marker: str, api_format: str) -> dict:
    """Append session marker text to the last text content block in a non-streaming response.

    Delegates to the appropriate ``PayloadFormat``.
    """
    return get_format(api_format).inject_session_marker(response_body, marker)


def _extract_delta_text(data: dict, api_format: str) -> str:
    """Extract text delta from a streaming SSE event payload.

    Delegates to the appropriate ``PayloadFormat``.
    """
    return get_format(api_format).extract_delta_text(data)


def _extract_assistant_text(response_body: dict, api_format: str) -> str:
    """Extract assistant text from a non-streaming response.

    Delegates to the appropriate ``PayloadFormat``.
    """
    return get_format(api_format).extract_assistant_text(response_body)


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------

def create_app(
    upstream: str,
    config_path: str | None = None,
    *,
    shared_engine: VirtualContextEngine | None = None,
    shared_metrics: ProxyMetrics | None = None,
    instance_label: str = "",
) -> FastAPI:
    """Create the FastAPI proxy application.

    Args:
        upstream: Upstream provider base URL (e.g. https://api.anthropic.com).
        config_path: Path to virtual-context config file.
        shared_engine: Reuse an existing engine (multi-instance mode).
        shared_metrics: Reuse an existing metrics collector (multi-instance mode).
        instance_label: Human-readable label for this instance (e.g. "anthropic").
    """
    upstream = upstream.rstrip("/")

    # Initialize engine + session registry
    registry: SessionRegistry | None = None
    default_state: ProxyState | None = None
    try:
        if shared_engine is not None:
            engine = shared_engine
        else:
            engine = VirtualContextEngine(config_path=config_path)

            # Lossless restart: if engine has no persisted state for its
            # auto-generated session_id, try loading the most recent session.
            # This avoids re-ingestion on proxy restart.
            if (
                hasattr(engine, '_store')
                and hasattr(engine._store, 'load_latest_engine_state')
                and not engine._turn_tag_index.entries
            ):
                try:
                    latest = engine._store.load_latest_engine_state()
                    if (
                        latest
                        and isinstance(getattr(latest, 'turn_tag_entries', None), list)
                        and latest.turn_tag_entries
                    ):
                        engine.config.session_id = latest.session_id
                        engine._load_persisted_state()
                        print(
                            f"Lossless restart: restored session {latest.session_id[:12]} "
                            f"({len(latest.turn_tag_entries)} turns, compacted={latest.compacted_through})",
                            flush=True,
                        )
                except Exception as _e:
                    print(f"Lossless restart failed: {_e}", flush=True)

        if shared_metrics is not None:
            metrics = shared_metrics
        else:
            metrics = ProxyMetrics(
                context_window=engine.config.monitor.context_window,
            )
        # Create the default session (used by dashboard and first requests)
        default_state = ProxyState(engine, metrics=metrics, upstream=upstream)

        # If lossless restart recovered state, mark session as already ingested
        # so the proxy doesn't re-ingest history on first request.
        if engine._turn_tag_index.entries:
            default_state._ingested_sessions.add(engine.config.session_id)

        # Build registry and pre-register the default session
        registry = SessionRegistry(
            config_path=config_path,
            upstream=upstream,
            metrics=metrics,
            store=engine._store,
        )
        registry._sessions[engine.config.session_id] = default_state

        logger.info(
            "Engine ready — session_id=%s, window=%d, storage=%s%s",
            engine.config.session_id,
            engine.config.monitor.context_window,
            engine.config.storage.backend,
            f", label={instance_label}" if instance_label else "",
        )
    except Exception as e:
        print(f"Engine init failed: {e}", file=sys.stderr)
        metrics = shared_metrics or ProxyMetrics()

    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    shutdown_event = asyncio.Event()

    # --------------- Raw request log setup ---------------
    import os as _os
    from pathlib import Path as _Path

    _request_log_dir: _Path | None = None
    _request_log_max: int = 50

    if default_state:
        try:
            from ..types import ProxyConfig as _ProxyConfig
            proxy_cfg = default_state.engine.config.proxy
            if isinstance(proxy_cfg, _ProxyConfig):
                _request_log_dir = _Path(proxy_cfg.request_log_dir)
                _request_log_max = proxy_cfg.request_log_max_files

                _request_log_dir.mkdir(parents=True, exist_ok=True)

                # Prune old files on startup — keep only the newest N sets
                # (each request produces up to 6 files: 1-inbound, 2-to-llm,
                # 3-from-llm, 4-to-client, session, plus continuation files)
                existing = sorted(
                    list(_request_log_dir.glob("*.json"))
                    + list(_request_log_dir.glob("*.txt")),
                    key=lambda p: p.stat().st_mtime,
                )
                keep_files = _request_log_max * 6
                if len(existing) > keep_files:
                    for stale in existing[: len(existing) - keep_files]:
                        stale.unlink(missing_ok=True)
                    pruned = len(existing) - keep_files
                    print(f"Request log: pruned {pruned} old files, kept {keep_files} in {_request_log_dir}", flush=True)
                else:
                    print(f"Request log: {len(existing)} existing files in {_request_log_dir}", flush=True)
        except Exception:
            pass  # engine may be a mock in tests

    _log_seq = 0  # monotonic counter for filenames

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        yield
        shutdown_event.set()
        await client.aclose()
        if registry:
            registry.shutdown_all()

    _app_title = "virtual-context proxy"
    if instance_label:
        _app_title += f" [{instance_label}]"
    app = FastAPI(title=_app_title, lifespan=lifespan)
    app.state.instance_label = instance_label

    # Register dashboard routes BEFORE the catch-all so /dashboard is not swallowed
    # Dashboard uses the default state for settings and config access
    register_dashboard_routes(
        app, metrics, default_state, shutdown_event,
        registry=registry, instance_label=instance_label,
    )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def catch_all(request: Request, path: str):
        url = f"{upstream}/{path}"
        raw_headers = dict(request.headers)
        fwd_headers = _forward_headers(raw_headers)

        # Non-POST or no body → passthrough
        if request.method != "POST":
            return await _passthrough(client, request, url, fwd_headers)

        body_bytes = await request.body()
        if not body_bytes:
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        # --- Raw request log: dump entire payload before any processing ---
        nonlocal _log_seq
        _response_log_path: _Path | None = None
        _session_log_path: _Path | None = None
        _log_prefix = ""
        if _request_log_dir and body_bytes:
            _log_seq += 1
            import datetime as _dt_log
            ts = _dt_log.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            _log_prefix = f"{_log_seq:06d}_{ts}_{path.replace('/', '_')}"
            # 1-inbound: raw request from client
            req_log = _request_log_dir / f"{_log_prefix}.1-inbound.json"
            _response_log_path = _request_log_dir / f"{_log_prefix}.3-from-llm.json"
            _session_log_path = _request_log_dir / f"{_log_prefix}.session.json"
            try:
                req_log.write_bytes(body_bytes)
            except Exception:
                pass  # never let logging break the request

        import json as _json
        try:
            body = _json.loads(body_bytes)
        except _json.JSONDecodeError:
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        # Only intercept if it has a messages/contents array (chat completion)
        fmt = detect_format(body)
        if not fmt.has_messages(body):
            return await _passthrough_bytes(client, request.method, url, fwd_headers, body_bytes)

        # Extract session ID from markers before stripping them
        inbound_session_id = _extract_session_id(body)

        # Strip session markers from assistant messages before any processing.
        # The LLM should never see stale markers.
        body = _strip_session_markers(body)

        # Route to the correct session
        state: ProxyState | None = None
        if registry:
            state, is_new = registry.get_or_create(
                inbound_session_id, body=body,
            )
            # Update trailing fingerprint on the engine so it gets persisted
            # with the next _save_state() call (inside on_turn_complete).
            if state:
                fp = SessionRegistry._compute_fingerprint(body)
                if fp:
                    state.engine._trailing_fingerprint = fp
                    registry._fingerprints[fp] = state.engine.config.session_id

        api_format = fmt.name
        user_message = fmt.extract_user_message(body)
        is_streaming = body.get("stream", False)

        # Track payload size for dashboard (KB + rough token estimate)
        _payload_kb = round(len(body_bytes) / 1024, 1)
        _payload_tok = fmt.estimate_payload_tokens(body)
        if state:
            state._last_payload_kb = _payload_kb
            state._last_payload_tokens = _payload_tok
            if state._initial_payload_kb is None:
                state._initial_payload_kb = _payload_kb
                state._initial_payload_tokens = _payload_tok

        import datetime as _dt
        _now = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _msg_count = len(body.get("messages", []))
        _sid = state.engine.config.session_id[:12] if state else "none"
        print(f"[{_now}] POST /{path} msgs={_msg_count} stream={is_streaming} session={_sid} payload={_payload_kb}KB")

        if not user_message:
            # Tool-result or non-text turn — skip VC enrichment but
            # preserve streaming so the client SDK doesn't break.
            _skip_sid = state.engine.config.session_id if state else ""
            if is_streaming:
                return await _handle_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=(state.turn_offset + len(state.conversation_history) // 2) if state else 0,
                    session_id=_skip_sid, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                )
            else:
                return await _handle_non_streaming(
                    client, url, fwd_headers, body, api_format, state,
                    metrics=metrics, turn=(state.turn_offset + len(state.conversation_history) // 2) if state else 0,
                    session_id=_skip_sid, response_log_path=_response_log_path,
                    session_log_path=_session_log_path,
                    request_log_dir=_request_log_dir, log_prefix=_log_prefix,
                )

        # ---------------------------------------------------------------
        # State-aware dispatch: PASSTHROUGH/INGESTING vs ACTIVE
        # ---------------------------------------------------------------
        if state:
            current_state = state.session_state

            # Fresh session starts ACTIVE but may need ingestion — check and
            # redirect to passthrough path if there's history to ingest.
            if (
                current_state == SessionState.ACTIVE
                and state.engine.config.session_id not in state._ingested_sessions
            ):
                history_pairs = _extract_history_pairs(body)
                needed = len(history_pairs) // 2
                existing = len(state.engine._turn_tag_index.entries)
                if needed > 0 and existing < needed:
                    current_state = SessionState.PASSTHROUGH

            if current_state in (SessionState.PASSTHROUGH, SessionState.INGESTING):
                # Store latest body for catch-up loop
                state._latest_body = body

                # On first request: kick off non-blocking ingestion
                if not state._history_ingested():
                    history_pairs = _extract_history_pairs(body)
                    if history_pairs:
                        state.conversation_history = list(history_pairs)
                    await asyncio.to_thread(
                        state.start_ingestion_if_needed, history_pairs,
                    )

                state.conversation_history.append(
                    Message(role="user", content=user_message)
                )

                _session_id = state.engine.config.session_id
                turn = state.turn_offset + len(state.conversation_history) // 2

                # Record passthrough request event
                metrics.record({
                    "type": "request",
                    "turn": turn,
                    "message_preview": user_message[:60],
                    "api_format": api_format,
                    "streaming": is_streaming,
                    "tags": [],
                    "broad": False,
                    "temporal": False,
                    "context_tokens": 0,
                    "budget": {},
                    "history_len": len(state.conversation_history),
                    "compacted_through": 0,
                    "wait_ms": 0,
                    "inbound_ms": 0,
                    "overhead_ms": 0,
                    "total_turns": turn,
                    "filtered_turns": turn,
                    "input_tokens": 0,
                    "raw_input_tokens": 0,
                    "system_tokens": 0,
                    "turns_dropped": 0,
                    "session_id": _session_id,
                    "passthrough": True,
                })

                metrics.capture_request(
                    turn, body, api_format,
                    session_id=_session_id,
                    passthrough=True,
                )

                print(
                    f"[T{turn}] PASSTHROUGH {api_format} "
                    f"stream={is_streaming} state={current_state.value} "
                    f"| {user_message[:60]}"
                )

                if is_streaming:
                    return await _handle_streaming(
                        client, url, fwd_headers, body, api_format, state,
                        metrics=metrics, turn=turn,
                        session_id=_session_id,
                        passthrough=True, response_log_path=_response_log_path,
                        session_log_path=_session_log_path,
                    )
                else:
                    return await _handle_non_streaming(
                        client, url, fwd_headers, body, api_format, state,
                        metrics=metrics, turn=turn,
                        session_id=_session_id,
                        passthrough=True, response_log_path=_response_log_path,
                        session_log_path=_session_log_path,
                        request_log_dir=_request_log_dir, log_prefix=_log_prefix,
                    )

        # ---------------------------------------------------------------
        # ACTIVE path: full enrichment
        # ---------------------------------------------------------------
        prepend_text = ""
        assembled = None
        wait_ms = 0.0
        inbound_ms = 0.0
        if state:
            try:
                t0 = time.monotonic()
                await asyncio.to_thread(state.wait_for_complete)
                wait_ms = round((time.monotonic() - t0) * 1000, 1)

                state.conversation_history.append(
                    Message(role="user", content=user_message)
                )

                # Compute available headroom for VC context injection
                _available_for_vc: int | None = None
                try:
                    _upstream_limit = int(state.engine.config.proxy.upstream_context_limit)
                    _output_budget = body.get("max_tokens", 4096)
                    _overhead = 5000  # tools, XML wrappers, safety margin
                    _available_for_vc = max(0, _upstream_limit - _payload_tok - _output_budget - _overhead)
                except (TypeError, ValueError, AttributeError):
                    pass  # headroom unknown — assembler uses default budget

                t1 = time.monotonic()
                assembled = await asyncio.to_thread(
                    state.engine.on_message_inbound,
                    user_message,
                    state.conversation_history,
                    body.get("model", ""),
                    max_context_tokens=_available_for_vc,
                )
                inbound_ms = round((time.monotonic() - t1) * 1000, 1)

                prepend_text = assembled.prepend_text
            except Exception as e:
                logger.error("Engine error (forwarding unmodified): %s", e)

        # Filter irrelevant history turns from the request body
        _pre_filter_body = body  # preserve for request capture
        turns_dropped = 0
        _real_tags = [t for t in (assembled.matched_tags if assembled else []) if t != "_general"]
        if _real_tags and state:
            recent = state.engine.config.assembler.recent_turns_always_included
            # PROXY-023: when paging is active, drop compacted turns so the
            # LLM relies on VC summaries + vc_expand_topic for old content.
            _ct = 0
            if (
                state.engine.config.paging.enabled
                and state.engine._compacted_through > 0
            ):
                _ct = state.engine._compacted_through // 2
            body, turns_dropped = _filter_body_messages(
                body,
                state.engine._turn_tag_index,
                _real_tags,
                recent_turns=recent,
                broad=assembled.broad,
                temporal=assembled.temporal,
                compacted_turn=_ct,
                fmt=fmt,
            )

        enriched_body = _inject_context(body, prepend_text, api_format)

        # Inject VC paging tools for autonomous mode (formats that support it)
        paging_enabled = False
        if (
            state
            and fmt.supports_tool_interception
            and state.engine.config.paging.enabled
        ):
            _paging_mode = state.engine._resolve_paging_mode(
                enriched_body.get("model", ""),
            )
            if _paging_mode == "autonomous":
                enriched_body = _inject_vc_tools(enriched_body, state.engine)
                paging_enabled = True
                _vc_names = [t["name"] for t in enriched_body.get("tools", []) if t.get("name", "").startswith("vc_")]
                print(f"[PAGING] Tools injected: {_vc_names} (total tools: {len(enriched_body.get('tools', []))})")
            else:
                print(f"[PAGING] Mode={_paging_mode} for model={enriched_body.get('model', '?')} — tools NOT injected")

        # Track enriched payload size
        import json as _json2
        if state:
            state._last_enriched_payload_kb = round(len(_json2.dumps(enriched_body)) / 1024, 1)

        # 2-to-llm: enriched body sent to the LLM (after filtering + context + tools)
        if _request_log_dir and _log_prefix:
            try:
                _to_llm_log = _request_log_dir / f"{_log_prefix}.2-to-llm.json"
                _to_llm_log.write_text(_json2.dumps(enriched_body, default=str))
            except Exception:
                pass

        is_streaming = body.get("stream", False)

        # Estimate system prompt tokens from original body (before VC enrichment)
        system_tokens = fmt._estimate_system_tokens(body)

        # Estimate input tokens from enriched body
        input_tokens = fmt.estimate_payload_tokens(enriched_body)

        # Estimate raw (pre-filter) input tokens — the baseline cost of sending
        # the full unfiltered payload.  Uses _pre_filter_body so it reflects
        # what a naive system without VC would actually send upstream.
        raw_input_tokens = fmt.estimate_payload_tokens(_pre_filter_body)

        # Track enriched payload tokens for dashboard
        # (raw payload tokens already set earlier alongside KB tracking)
        if state:
            state._last_enriched_payload_tokens = input_tokens

        # Record request event
        turn = (state.turn_offset + len(state.conversation_history) // 2) if state else 0
        context_tokens = len(prepend_text) // 4 if prepend_text else 0
        total_turns = (state.turn_offset + len(state.conversation_history) // 2) if state else 0
        overhead_ms = round(wait_ms + inbound_ms, 1)
        _session_id = state.engine.config.session_id if state else ""
        metrics.record({
            "type": "request",
            "turn": turn,
            "model": body.get("model", ""),
            "message_preview": user_message[:60],
            "api_format": api_format,
            "streaming": is_streaming,
            "tags": assembled.matched_tags if assembled else [],
            "broad": assembled.broad if assembled else False,
            "temporal": assembled.temporal if assembled else False,
            "context_tokens": context_tokens,
            "budget": assembled.budget_breakdown if assembled else {},
            "history_len": len(state.conversation_history) if state else 0,
            "compacted_through": getattr(
                state.engine, "_compacted_through", 0
            ) if state else 0,
            "wait_ms": wait_ms,
            "inbound_ms": inbound_ms,
            "overhead_ms": overhead_ms,
            "total_turns": total_turns,
            "filtered_turns": total_turns - turns_dropped,
            "input_tokens": input_tokens,
            "raw_input_tokens": raw_input_tokens,
            "system_tokens": system_tokens,
            "turns_dropped": turns_dropped,
            "session_id": _session_id,
        })

        # Log request to terminal for debugging
        _tags_str = ", ".join(assembled.matched_tags) if assembled else "none"
        _flags = []
        if assembled and assembled.broad:
            _flags.append("BROAD")
        if assembled and assembled.temporal:
            _flags.append("TEMPORAL")
        _flag_str = f" [{' '.join(_flags)}]" if _flags else ""
        print(
            f"[T{turn}] POST {api_format} stream={is_streaming} "
            f"tags=[{_tags_str}]{_flag_str} "
            f"msgs={len(body.get('messages', []))} "
            f"dropped={turns_dropped} "
            f"ctx={context_tokens}t input={input_tokens}t "
            f"vc={overhead_ms}ms | {user_message[:60]}"
        )

        # Capture pre-filter request body for dashboard inspection
        metrics.capture_request(
            turn, _pre_filter_body, api_format,
            inbound_tags=assembled.matched_tags if assembled else [],
            session_id=_session_id,
        )

        if is_streaming:
            return await _handle_streaming(
                client, url, fwd_headers, enriched_body, api_format, state,
                metrics=metrics, turn=turn, overhead_ms=overhead_ms,
                session_id=_session_id, response_log_path=_response_log_path,
                session_log_path=_session_log_path,
                paging_enabled=paging_enabled,
                request_log_dir=_request_log_dir,
                log_prefix=_log_prefix if _request_log_dir else "",
            )
        else:
            return await _handle_non_streaming(
                client, url, fwd_headers, enriched_body, api_format, state,
                metrics=metrics, turn=turn, overhead_ms=overhead_ms,
                session_id=_session_id, response_log_path=_response_log_path,
                session_log_path=_session_log_path,
                request_log_dir=_request_log_dir, log_prefix=_log_prefix,
            )

    return app


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

async def _passthrough(
    client: httpx.AsyncClient,
    request: Request,
    url: str,
    headers: dict[str, str],
) -> StreamingResponse:
    """Transparent forwarding for non-chat requests."""
    body = await request.body()
    return await _passthrough_bytes(client, request.method, url, headers, body)


async def _passthrough_bytes(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> StreamingResponse:
    """Forward raw bytes to upstream and stream back."""
    resp = await client.request(method, url, headers=headers, content=body)
    return JSONResponse(
        content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


def _dump_session_state(
    state: "ProxyState",
    session_log_path: object,
) -> None:
    """Write full proxy memory dump to disk alongside request/response logs.

    Includes TurnTagIndex, tag summaries, aliases, session metadata,
    and compaction state — everything needed to reconstruct the proxy's
    understanding of the conversation at this point in time.
    """
    import json as _json_dump
    try:
        engine = state.engine
        idx = engine._turn_tag_index

        # TurnTagIndex entries
        entries = []
        for e in idx.entries:
            entries.append({
                "turn": e.turn_number,
                "tags": e.tags,
                "primary_tag": e.primary_tag,
                "message_hash": e.message_hash,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            })

        # Tag counts (how many turns per tag)
        tag_counts: dict[str, int] = {}
        for e in idx.entries:
            for t in e.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        # Tag summaries from store
        tag_summaries = []
        try:
            for ts in engine._store.get_all_tag_summaries():
                tag_summaries.append({
                    "tag": ts.tag,
                    "summary": ts.summary,
                    "summary_tokens": ts.summary_tokens,
                    "source_turn_numbers": ts.source_turn_numbers,
                    "covers_through_turn": ts.covers_through_turn,
                })
        except Exception:
            pass

        # Tag aliases
        aliases: dict[str, str] = {}
        try:
            aliases = engine._store.get_tag_aliases()
        except Exception:
            pass

        # Tag stats from store
        tag_stats = []
        try:
            for st in engine._store.get_all_tags():
                tag_stats.append({
                    "tag": st.tag,
                    "usage_count": st.usage_count,
                    "total_full_tokens": st.total_full_tokens,
                    "total_summary_tokens": st.total_summary_tokens,
                })
        except Exception:
            pass

        # Split processed tags
        split_tags = list(getattr(engine, "_split_processed_tags", set()))

        # Working set (paging state)
        working_set_dump: list[dict] = []
        ws = getattr(engine, "_working_set", None)
        if ws:
            for tag, entry in ws.items():
                working_set_dump.append({
                    "tag": tag,
                    "depth": entry.depth.value if hasattr(entry.depth, "value") else str(entry.depth),
                    "tokens": entry.tokens,
                    "last_accessed_turn": entry.last_accessed_turn,
                })

        dump = {
            "session_id": engine.config.session_id,
            "session_state": state._state.value if hasattr(state._state, "value") else str(state._state),
            "turn_count": len(state.conversation_history) // 2,
            "compacted_through": getattr(engine, "_compacted_through", 0),
            "turn_tag_index": entries,
            "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
            "tag_summaries": tag_summaries,
            "tag_aliases": aliases,
            "tag_stats": tag_stats,
            "split_processed_tags": split_tags,
            "working_set": working_set_dump,
            "conversation_history": [
                {"role": m.role, "content": m.content[:500]}
                for m in state.conversation_history
            ],
        }

        session_log_path.write_text(
            _json_dump.dumps(dump, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass  # never let session dump break the request


# ---------------------------------------------------------------------------
# Paging tool interception helpers (Phase 6)
# ---------------------------------------------------------------------------

_VC_TOOL_NAMES = frozenset({"vc_expand_topic", "vc_collapse_topic", "vc_find_quote"})


def _vc_tool_definitions() -> list[dict]:
    """Return Anthropic tool definitions for VC context tools."""
    return [
        {
            "name": "vc_expand_topic",
            "description": (
                "Zoom into a topic you can already see in the context-topics list. "
                "Use when a topic summary mentions the area you need but lacks "
                "detail — e.g. it says 'discussed supplements' but you need the "
                "specific dosage. Requires knowing which tag to expand. "
                "For specific facts when you don't know which topic holds them, "
                "use vc_find_quote instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Topic tag from the context-topics list to expand.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["segments", "full"],
                        "description": (
                            "Target depth: 'segments' for individual summaries, "
                            "'full' for original conversation text."
                        ),
                    },
                },
                "required": ["tag"],
            },
        },
        {
            "name": "vc_collapse_topic",
            "description": (
                "Collapse an expanded topic back to its summary to free context "
                "budget. Use after you've retrieved what you need from an expanded "
                "topic, or to make room before expanding a different one."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Topic tag to collapse.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["summary", "none"],
                        "description": (
                            "Target depth: 'summary' for brief overview, "
                            "'none' to remove from context entirely."
                        ),
                    },
                },
                "required": ["tag"],
            },
        },
        {
            "name": "vc_find_quote",
            "description": (
                "Search the full original conversation text for a specific word, "
                "phrase, or detail. Use this as your first tool when the user asks "
                "about a specific fact — a name, number, dosage, recommendation, "
                "date, or decision — especially when no topic summary mentions it "
                "or you don't know which topic it falls under. This bypasses tags "
                "entirely and searches raw text, so it finds content even when "
                "it's filed under an unexpected topic."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The word or phrase to search for. Use the most specific and "
                            "distinctive terms — e.g. 'magnesium glycinate' rather than "
                            "'supplement', or 'reservation 7pm' rather than 'dinner'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    ]


def _is_vc_tool(name: str) -> bool:
    """Return True if *name* is a known VC paging tool."""
    return name in _VC_TOOL_NAMES


def _execute_vc_tool(
    engine: "VirtualContextEngine", name: str, tool_input: dict,
) -> str:
    """Execute a VC paging tool and return a JSON result string."""
    import json as _json
    try:
        if name == "vc_expand_topic":
            result = engine.expand_topic(
                tag=tool_input.get("tag", ""),
                depth=tool_input.get("depth", "full"),
            )
        elif name == "vc_collapse_topic":
            result = engine.collapse_topic(
                tag=tool_input.get("tag", ""),
                depth=tool_input.get("depth", "summary"),
            )
        elif name == "vc_find_quote":
            result = engine.find_quote(
                query=tool_input.get("query", ""),
                max_results=tool_input.get("max_results", 5),
            )
        else:
            result = {"error": f"unknown VC tool: {name}"}
        return _json.dumps(result)
    except Exception as e:
        return _json.dumps({"is_error": True, "content": str(e)})


def _inject_vc_tools(body: dict, engine: "VirtualContextEngine") -> dict:
    """Append VC paging tool definitions to the request body's tools array.

    Delegates to the detected ``PayloadFormat`` for format-specific injection.
    """
    fmt = detect_format(body)
    return fmt.inject_tools(body, _vc_tool_definitions())


def _parse_sse_events(
    buf: bytes,
) -> tuple[list[tuple[str, str, bytes]], bytes]:
    """Split a byte buffer into complete SSE events.

    Returns ``(events, remainder)`` where each event is
    ``(event_type, data_str, raw_bytes)``.  Handles both ``\\r\\n\\r\\n``
    and ``\\n\\n`` event boundaries.
    """
    events: list[tuple[str, str, bytes]] = []
    while True:
        idx_rn = buf.find(b"\r\n\r\n")
        idx_n = buf.find(b"\n\n")
        if idx_rn == -1 and idx_n == -1:
            break
        # Use whichever boundary comes first
        if idx_rn != -1 and (idx_n == -1 or idx_rn <= idx_n):
            end = idx_rn + 4
        else:
            end = idx_n + 2

        raw_event = buf[:end]
        buf = buf[end:]

        decoded = raw_event.decode("utf-8", errors="replace")
        event_type = ""
        data_str = ""
        for line in decoded.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:]

        events.append((event_type, data_str, raw_event))

    return events, buf


def _build_continuation_request(
    original_body: dict,
    assistant_content: list[dict],
    tool_results: list[dict],
) -> dict:
    """Build a non-streaming continuation request after VC tool execution.

    Delegates to the detected PayloadFormat for format-specific message structure.
    """
    fmt = detect_format(original_body)
    return fmt.build_continuation_request(original_body, assistant_content, tool_results)


def _emit_text_as_sse(text: str, block_index: int) -> list[bytes]:
    """Convert *text* into Anthropic SSE events at *block_index*."""
    import json as _json
    events: list[bytes] = []
    start = _json.dumps({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {"type": "text", "text": ""},
    })
    events.append(f"event: content_block_start\ndata: {start}\n\n".encode())

    delta = _json.dumps({
        "type": "content_block_delta",
        "index": block_index,
        "delta": {"type": "text_delta", "text": text},
    })
    events.append(f"event: content_block_delta\ndata: {delta}\n\n".encode())

    stop = _json.dumps({
        "type": "content_block_stop",
        "index": block_index,
    })
    events.append(f"event: content_block_stop\ndata: {stop}\n\n".encode())
    return events


def _emit_tool_use_as_sse(
    tool: dict, block_index: int,
) -> list[bytes]:
    """Convert a tool_use content block into Anthropic SSE events."""
    import json as _json
    events: list[bytes] = []
    start = _json.dumps({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {
            "type": "tool_use",
            "id": tool.get("id", ""),
            "name": tool.get("name", ""),
            "input": {},
        },
    })
    events.append(f"event: content_block_start\ndata: {start}\n\n".encode())

    input_str = _json.dumps(tool.get("input", {}))
    delta = _json.dumps({
        "type": "content_block_delta",
        "index": block_index,
        "delta": {"type": "input_json_delta", "partial_json": input_str},
    })
    events.append(f"event: content_block_delta\ndata: {delta}\n\n".encode())

    stop = _json.dumps({
        "type": "content_block_stop",
        "index": block_index,
    })
    events.append(f"event: content_block_stop\ndata: {stop}\n\n".encode())
    return events


def _emit_message_end_sse(
    stop_reason: str = "end_turn",
    usage: dict | None = None,
) -> list[bytes]:
    """Emit ``message_delta`` and ``message_stop`` SSE events."""
    import json as _json
    events: list[bytes] = []
    md_payload: dict = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
    }
    if usage:
        md_payload["usage"] = usage
    md = _json.dumps(md_payload)
    events.append(f"event: message_delta\ndata: {md}\n\n".encode())

    ms = _json.dumps({"type": "message_stop"})
    events.append(f"event: message_stop\ndata: {ms}\n\n".encode())
    return events


async def _handle_streaming(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict,
    api_format: str,
    state: ProxyState | None,
    *,
    metrics: ProxyMetrics | None = None,
    turn: int = 0,
    overhead_ms: float = 0.0,
    session_id: str = "",
    passthrough: bool = False,
    response_log_path: object | None = None,
    session_log_path: object | None = None,
    paging_enabled: bool = False,
    request_log_dir: object | None = None,
    log_prefix: str = "",
) -> StreamingResponse | JSONResponse:
    """Forward SSE stream, accumulating assistant text for on_turn_complete.

    Forwards raw bytes from the upstream to preserve exact SSE framing.
    The Node.js Anthropic SDK is strict about SSE formatting — decoding
    and re-encoding via ``aiter_lines()`` can break its parser.

    When *paging_enabled* is True, uses event-level forwarding: parses SSE
    events individually, forwards text events immediately, suppresses VC
    tool_use events, executes them locally, sends non-streaming continuations,
    and emits the continuation text as SSE back to the client.

    Non-2xx upstream responses (rate limits, overloads) are returned as
    JSON errors instead of broken SSE streams.
    """
    import json as _json

    _MAX_CONTINUATION_LOOPS = 5

    headers = dict(headers)
    headers.pop("accept-encoding", None)

    # Open upstream connection — resolves after response headers arrive,
    # body streams lazily via aiter_bytes().
    t_upstream = time.monotonic()
    req = client.build_request("POST", url, headers=headers, json=body)
    upstream = await client.send(req, stream=True)

    # Non-2xx: drain body and return as JSON error (not broken SSE)
    if upstream.status_code >= 300:
        error_bytes = await upstream.aread()
        await upstream.aclose()
        upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)
        if metrics:
            metrics.record({
                "type": "response",
                "turn": turn,
                "upstream_ms": upstream_ms,
                "total_ms": round(overhead_ms + upstream_ms, 1),
                "streaming": True,
                "error": True,
                "session_id": session_id,
            })
        print(
            f"[T{turn}] ERROR {upstream.status_code} "
            f"llm={int(upstream_ms)}ms | {error_bytes[:200].decode('utf-8', errors='replace')}"
        )
        try:
            error_body = _json.loads(error_bytes)
        except (ValueError, _json.JSONDecodeError):
            error_body = {"error": error_bytes.decode("utf-8", errors="replace")}
        return JSONResponse(
            content=error_body,
            status_code=upstream.status_code,
            headers=_forward_headers(dict(upstream.headers)),
        )

    # Forward upstream response headers + SSE-critical headers
    resp_headers = _forward_headers(dict(upstream.headers))
    resp_headers.setdefault("cache-control", "no-cache")
    resp_headers.setdefault("x-accel-buffering", "no")

    # ----- shared post-stream processing -----
    def _post_stream(text_chunks, raw_events):
        """Return (assistant_text, upstream_ms) and handle side-effects."""
        nonlocal t_upstream
        upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)
        if metrics:
            metrics.record({
                "type": "response",
                "turn": turn,
                "upstream_ms": upstream_ms,
                "total_ms": round(overhead_ms + upstream_ms, 1),
                "streaming": True,
                "session_id": session_id,
            })
        assistant_text = "".join(text_chunks)
        print(
            f"[T{turn}] RESPONSE stream={True} "
            f"llm={int(upstream_ms)}ms "
            f"total={int(round(overhead_ms + upstream_ms))}ms "
            f"chars={len(assistant_text)}"
        )
        if response_log_path:
            try:
                response_log_path.write_text(
                    _json.dumps({
                        "streaming": True,
                        "assistant_text": assistant_text,
                        "upstream_ms": upstream_ms,
                        "raw_events": "".join(raw_events),
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return assistant_text, upstream_ms

    async def _inner_stream():
        text_chunks: list[str] = []
        raw_events: list[str] = []

        # ---------------------------------------------------------------
        # Paging path: event-level forwarding with VC tool interception
        # ---------------------------------------------------------------
        if paging_enabled:
            buf = b""
            vc_tools: list[dict] = []          # [{id, name, input}]
            non_vc_tools: list[dict] = []      # [{id, name}]
            all_content_blocks: list[dict] = []  # for continuation
            forwarded_block_count = 0
            suppressing = False
            current_vc_tool: dict | None = None
            current_text_parts: list[str] = []
            suppressed_raw: list[bytes] = []
            need_continuation = False

            try:
                async for raw_chunk in upstream.aiter_bytes():
                    raw_events.append(
                        raw_chunk.decode("utf-8", errors="replace"),
                    )
                    buf += raw_chunk
                    events, buf = _parse_sse_events(buf)

                    for _evt_type, data_str, raw_bytes in events:
                        if not data_str or data_str.strip() == "[DONE]":
                            yield raw_bytes
                            continue

                        try:
                            data = _json.loads(data_str)
                        except _json.JSONDecodeError:
                            yield raw_bytes
                            continue

                        dtype = data.get("type", "")

                        # -- content_block_start --
                        if dtype == "content_block_start":
                            block = data.get("content_block", {})
                            btype = block.get("type", "")
                            if (
                                btype == "tool_use"
                                and _is_vc_tool(block.get("name", ""))
                            ):
                                suppressing = True
                                current_vc_tool = {
                                    "id": block["id"],
                                    "name": block["name"],
                                    "input_parts": [],
                                }
                                suppressed_raw.append(raw_bytes)
                                continue
                            elif btype == "tool_use":
                                non_vc_tools.append({
                                    "id": block["id"],
                                    "name": block["name"],
                                })
                            elif btype == "text":
                                current_text_parts = []

                        # -- content_block_delta --
                        elif dtype == "content_block_delta":
                            if suppressing:
                                delta = data.get("delta", {})
                                if delta.get("type") == "input_json_delta":
                                    current_vc_tool["input_parts"].append(
                                        delta.get("partial_json", ""),
                                    )
                                suppressed_raw.append(raw_bytes)
                                continue
                            else:
                                dt = _extract_delta_text(data, "anthropic")
                                if dt:
                                    text_chunks.append(dt)
                                    current_text_parts.append(dt)

                        # -- content_block_stop --
                        elif dtype == "content_block_stop":
                            if suppressing:
                                if current_vc_tool:
                                    input_str = "".join(
                                        current_vc_tool["input_parts"],
                                    )
                                    try:
                                        parsed_input = _json.loads(input_str)
                                    except _json.JSONDecodeError:
                                        parsed_input = {}
                                    vc_tools.append({
                                        "id": current_vc_tool["id"],
                                        "name": current_vc_tool["name"],
                                        "input": parsed_input,
                                    })
                                    all_content_blocks.append({
                                        "type": "tool_use",
                                        "id": current_vc_tool["id"],
                                        "name": current_vc_tool["name"],
                                        "input": parsed_input,
                                    })
                                    current_vc_tool = None
                                suppressing = False
                                suppressed_raw.append(raw_bytes)
                                continue
                            else:
                                # Finalize text block
                                if current_text_parts:
                                    all_content_blocks.append({
                                        "type": "text",
                                        "text": "".join(current_text_parts),
                                    })
                                    current_text_parts = []
                                    forwarded_block_count += 1

                        # -- message_delta --
                        elif dtype == "message_delta":
                            sr = data.get("delta", {}).get("stop_reason")
                            if sr == "tool_use" and vc_tools:
                                if non_vc_tools:
                                    # BAIL: mixed VC + non-VC tools
                                    logger.warning(
                                        "Mixed VC + non-VC tools in "
                                        "response — passing all through",
                                    )
                                    for s in suppressed_raw:
                                        yield s
                                    suppressed_raw.clear()
                                    vc_tools.clear()
                                    need_continuation = False
                                else:
                                    # All VC — suppress, handle after
                                    # message_stop
                                    need_continuation = True
                                    suppressed_raw.append(raw_bytes)
                                    continue

                        # -- message_stop --
                        elif dtype == "message_stop":
                            if need_continuation:
                                suppressed_raw.append(raw_bytes)
                                continue

                        # Default: forward event to client
                        yield raw_bytes
            finally:
                await upstream.aclose()

            # --- Continuation phase ---
            if need_continuation and vc_tools and state:
                cont_body: dict | None = None
                cont_data: dict | None = None
                loop_content_blocks: list[dict] = []

                for loop_i in range(_MAX_CONTINUATION_LOOPS):
                    # Execute VC tools
                    tool_results: list[dict] = []
                    for tool in vc_tools:
                        t_tool = time.monotonic()
                        result_str = _execute_vc_tool(
                            state.engine,
                            tool["name"],
                            tool["input"],
                        )
                        tool_ms = round(
                            (time.monotonic() - t_tool) * 1000, 1,
                        )
                        if metrics:
                            metrics.record({
                                "type": "tool_intercept",
                                "turn": turn,
                                "tool_name": tool["name"],
                                "tool_input": tool["input"],
                                "result": result_str[:200],
                                "duration_ms": tool_ms,
                                "continuation_count": loop_i + 1,
                                "session_id": session_id,
                            })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": result_str,
                        })

                    # Re-assemble context with updated working set
                    # so the LLM sees the expanded content in this
                    # turn (not deferred to next turn).
                    new_prepend = state.engine.reassemble_context()
                    reassembled_body = _inject_context(
                        body, new_prepend, api_format,
                    ) if new_prepend else body

                    # Build or extend continuation request
                    if cont_body is None:
                        cont_body = _build_continuation_request(
                            reassembled_body,
                            all_content_blocks,
                            tool_results,
                        )
                    else:
                        # Update system prompt with re-assembled context
                        if new_prepend and "system" in reassembled_body:
                            cont_body["system"] = reassembled_body["system"]
                        cont_body["messages"].append({
                            "role": "assistant",
                            "content": loop_content_blocks,
                        })
                        cont_body["messages"].append({
                            "role": "user",
                            "content": tool_results,
                        })

                    # Send non-streaming continuation
                    cont_resp = await client.post(
                        url, headers=headers, json=cont_body,
                    )

                    # Log continuation request/response to disk
                    if request_log_dir and log_prefix:
                        _cn = loop_i + 1
                        try:
                            from pathlib import Path as _CPath
                            _cdir = _CPath(request_log_dir)
                            _cdir.joinpath(
                                f"{log_prefix}.continuation-{_cn}.request.json"
                            ).write_text(
                                _json.dumps(cont_body, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            _cdir.joinpath(
                                f"{log_prefix}.continuation-{_cn}.response.json"
                            ).write_text(
                                cont_resp.text, encoding="utf-8",
                            )
                        except Exception:
                            pass  # never let logging break the request

                    if cont_resp.status_code >= 300:
                        logger.error(
                            "Continuation failed: %d",
                            cont_resp.status_code,
                        )
                        break

                    cont_data = cont_resp.json()
                    stop_reason = cont_data.get("stop_reason", "end_turn")
                    content = cont_data.get("content", [])
                    loop_content_blocks = content

                    # Emit text blocks as SSE
                    text_blocks = [
                        b for b in content if b.get("type") == "text"
                    ]
                    tool_blocks = [
                        b for b in content if b.get("type") == "tool_use"
                    ]
                    vc_next = [
                        b for b in tool_blocks
                        if _is_vc_tool(b.get("name", ""))
                    ]

                    for tb in text_blocks:
                        t = tb.get("text", "")
                        if t:
                            text_chunks.append(t)
                            for sse_evt in _emit_text_as_sse(
                                t, forwarded_block_count,
                            ):
                                yield sse_evt
                            forwarded_block_count += 1

                    # More VC-only tool calls → loop
                    if (
                        stop_reason == "tool_use"
                        and vc_next
                        and all(
                            _is_vc_tool(b.get("name", ""))
                            for b in tool_blocks
                        )
                    ):
                        vc_tools = [
                            {
                                "id": b["id"],
                                "name": b["name"],
                                "input": b.get("input", {}),
                            }
                            for b in vc_next
                        ]
                        continue

                    # Done — forward any non-VC tools to client
                    non_vc_in_cont = [
                        b for b in tool_blocks
                        if not _is_vc_tool(b.get("name", ""))
                    ]
                    if non_vc_in_cont:
                        for nvc in non_vc_in_cont:
                            for sse_evt in _emit_tool_use_as_sse(
                                nvc, forwarded_block_count,
                            ):
                                yield sse_evt
                            forwarded_block_count += 1
                    break

                # Emit message end events
                cont_usage = cont_data.get("usage") if cont_data else None
                cont_stop = "end_turn"
                if cont_data:
                    # If the LLM stopped for tool_use and we forwarded
                    # non-VC tools, tell the client so it handles them.
                    raw_stop = cont_data.get("stop_reason", "end_turn")
                    non_vc_forwarded = any(
                        b.get("type") == "tool_use"
                        and not _is_vc_tool(b.get("name", ""))
                        for b in cont_data.get("content", [])
                    )
                    if raw_stop == "tool_use" and non_vc_forwarded:
                        cont_stop = "tool_use"
                for sse_evt in _emit_message_end_sse(cont_stop, usage=cont_usage):
                    yield sse_evt

            # Post-stream processing
            assistant_text, _ = _post_stream(text_chunks, raw_events)
            if state and assistant_text:
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text),
                )
                if not passthrough:
                    state.fire_turn_complete(
                        list(state.conversation_history),
                        payload_tokens=state._last_payload_tokens or None,
                    )
                _marker_sid = state.engine.config.session_id
                _fmt = get_format(api_format)
                yield _fmt.emit_session_marker_sse(_marker_sid)

            if session_log_path and state:
                _dump_session_state(state, session_log_path)

            return  # exit — don't fall through to raw-byte path

        # ---------------------------------------------------------------
        # Non-paging path: raw-byte forwarding (unchanged)
        # ---------------------------------------------------------------
        line_buf = ""
        try:
            async for raw_chunk in upstream.aiter_bytes():
                yield raw_chunk  # forward raw bytes unchanged

                # Side-channel: parse for text accumulation + log capture
                decoded = raw_chunk.decode("utf-8", errors="replace")
                raw_events.append(decoded)
                line_buf += decoded
                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        try:
                            data = _json.loads(data_str)
                            delta = _extract_delta_text(data, api_format)
                            if delta:
                                text_chunks.append(delta)
                        except _json.JSONDecodeError:
                            pass
        finally:
            await upstream.aclose()
            assistant_text, _ = _post_stream(text_chunks, raw_events)
            if state and assistant_text:
                state.conversation_history.append(
                    Message(role="assistant", content=assistant_text)
                )
                if not passthrough:
                    state.fire_turn_complete(
                        list(state.conversation_history),
                        payload_tokens=state._last_payload_tokens or None,
                    )

                # Inject session marker as a final SSE delta so the client SDK
                # accumulates it into the stored assistant message.
                _marker_sid = state.engine.config.session_id
                _fmt = get_format(api_format)
                yield _fmt.emit_session_marker_sse(_marker_sid)

            # Session state dump (after response + history update)
            if session_log_path and state:
                _dump_session_state(state, session_log_path)

    async def stream_generator():
        """Wrapper that captures all bytes sent to the client."""
        client_chunks: list[bytes] = []
        async for chunk in _inner_stream():
            client_chunks.append(chunk)
            yield chunk
        # 4-to-client: log everything sent back to the client
        if request_log_dir and log_prefix:
            try:
                from pathlib import Path as _P4
                _P4(request_log_dir).joinpath(
                    f"{log_prefix}.4-to-client.txt",
                ).write_bytes(b"".join(client_chunks))
            except Exception:
                pass

    return StreamingResponse(
        stream_generator(),
        status_code=upstream.status_code,
        headers=resp_headers,
    )


async def _handle_non_streaming(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict,
    api_format: str,
    state: ProxyState | None,
    *,
    metrics: ProxyMetrics | None = None,
    turn: int = 0,
    overhead_ms: float = 0.0,
    session_id: str = "",
    passthrough: bool = False,
    response_log_path: object | None = None,
    session_log_path: object | None = None,
    request_log_dir: object | None = None,
    log_prefix: str = "",
) -> JSONResponse:
    """Forward JSON response, parse assistant text, fire on_turn_complete."""
    import json as _json_ns
    t_upstream = time.monotonic()
    resp = await client.request("POST", url, headers=headers, json=body)
    upstream_ms = round((time.monotonic() - t_upstream) * 1000, 1)

    try:
        response_body = resp.json()
    except Exception:
        return JSONResponse(content=resp.text, status_code=resp.status_code)

    # 3-from-llm: raw upstream response before any modification
    if request_log_dir and log_prefix:
        try:
            from pathlib import Path as _P3
            _P3(request_log_dir).joinpath(
                f"{log_prefix}.3-from-llm.json",
            ).write_text(
                _json_ns.dumps(response_body, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # Extract and record assistant text
    assistant_text = _extract_assistant_text(response_body, api_format)
    if state and assistant_text:
        state.conversation_history.append(
            Message(role="assistant", content=assistant_text)
        )
        if not passthrough:
            state.fire_turn_complete(
                list(state.conversation_history),
                payload_tokens=state._last_payload_tokens or None,
            )

        # Inject session marker into the response body so the client stores it
        session_id = state.engine.config.session_id
        marker = f"\n<!-- vc:session={session_id} -->"
        response_body = _inject_session_marker(response_body, marker, api_format)

    if metrics:
        metrics.record({
            "type": "response",
            "turn": turn,
            "upstream_ms": upstream_ms,
            "total_ms": round(overhead_ms + upstream_ms, 1),
            "streaming": False,
            "session_id": session_id,
        })

    print(
        f"[T{turn}] RESPONSE stream=False "
        f"llm={int(upstream_ms)}ms "
        f"total={int(round(overhead_ms + upstream_ms))}ms "
        f"chars={len(assistant_text or '')}"
    )

    # Session state dump
    if session_log_path and state:
        _dump_session_state(state, session_log_path)

    # Forward response headers (filter hop-by-hop)
    resp_headers = _forward_headers(dict(resp.headers))

    # 4-to-client: response body after session marker injection
    if request_log_dir and log_prefix:
        try:
            from pathlib import Path as _P4c
            _P4c(request_log_dir).joinpath(
                f"{log_prefix}.4-to-client.json",
            ).write_text(
                _json_ns.dumps(response_body, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    return JSONResponse(
        content=response_body,
        status_code=resp.status_code,
        headers=resp_headers,
    )
