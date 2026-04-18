"""TaggingPipeline: turn tagging and history ingestion.

Extracted from engine.py — handles Phase 1 of turn processing (tag_turn),
bulk historical ingestion (ingest_history), and tag splitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from .conversation_store import StaleConversationWriteError
from .engine_utils import extract_turn_pairs, get_recent_context
from .canonical_turns import HASH_VERSION, compute_turn_hash_from_raw, utcnow_iso
from .ingest_reconciler import IngestReconciler
from .segmenter import pair_messages_into_turns
from .store import ContextStore
from .turn_tag_index import TurnTagIndex
from ..types import Message

if TYPE_CHECKING:
    from .compactor import DomainCompactor
    from .monitor import ContextMonitor
    from .semantic_search import SemanticSearchManager
    from .tag_canonicalizer import TagCanonicalizer
    from .tag_generator import TagGenerator
    from .tag_splitter import TagSplitter
    from .telemetry import TelemetryLedger
    from ..types import (
        CanonicalTurnRow,
        CompactionSignal,
        EngineState,
        Message,
        SplitResult,
        StoredSummary,
        TagResult,
        TurnTagEntry,
        VirtualContextConfig,
    )

logger = logging.getLogger(__name__)

_TAG_BREAKDOWN_LOG_THRESHOLD_MS = 250.0
_TAG_BREAKDOWN_MAX_STAGES = 8

# Imported lazily at module level so tests that patch engine._is_stub_content
# still work.  The actual definitions live in engine.py.
_SESSION_HEADER_RE: re.Pattern | None = None
_is_stub_content_fn: Callable[[str], bool] | None = None

# Extraction patterns beyond the canonical "[Session from ...]" header. Order
# matters — earlier patterns win. Each pattern's group(1) is the raw date
# string we adopt verbatim (no parsing/reformatting).
_SESSION_DATE_EXTRACT_PATTERNS: tuple[re.Pattern, ...] = (
    # Proxy system-event envelope: "System: [2026-02-15T22:00:00Z] Model switched"
    # or "System (untrusted): [2026-02-15 22:00:00 UTC] ...".
    re.compile(r"System(?:\s*\([^)]*\))?:\s*\[(\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\s+[A-Z]{2,4})?[^\]]*)\]"),
    # Bare "[YYYY-MM-DD HH:MM:SS UTC]" at the start of a line — seen in some
    # ingest payloads that prepend a timestamp without the Session/System prefix.
    re.compile(r"(?m)^\[(\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\s+[A-Z]{2,4})?)\]"),
)


def _extract_session_date_from_content(content: str) -> str | None:
    # Preferentially use the canonical header (set by the segmenter-aware
    # bulk ingest paths), then fall back to the broader patterns above.
    if _SESSION_HEADER_RE is not None:
        m = _SESSION_HEADER_RE.search(content)
        if m:
            return m.group(1).strip()
    for pat in _SESSION_DATE_EXTRACT_PATTERNS:
        m = pat.search(content)
        if m:
            return m.group(1).strip()
    return None


def _bump_session_date_one_second(date_str: str) -> str | None:
    # Adds +1 second to an ISO 8601 date string so consecutive inherited turns
    # don't collapse onto an identical timestamp when the downstream consumer
    # relies on session_date for ordering. Returns None when the string is
    # empty or not ISO-parseable, letting the caller leave running_session_date
    # unchanged rather than corrupting a non-ISO format (e.g. "2023/05/25 (Thu)").
    if not date_str:
        return None
    normalized = date_str.replace("Z", "+00:00") if date_str.endswith("Z") else date_str
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    bumped = dt + timedelta(seconds=1)
    if dt.tzinfo is None:
        return bumped.strftime("%Y-%m-%dT%H:%M:%S")
    if date_str.endswith("Z"):
        return bumped.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    return bumped.isoformat()


def _ensure_engine_imports() -> None:
    """Lazy-import module-level symbols from engine to avoid circular imports."""
    global _SESSION_HEADER_RE, _is_stub_content_fn
    if _SESSION_HEADER_RE is None:
        from ..engine import _SESSION_HEADER_RE as _hdr, _is_stub_content as _stub
        _SESSION_HEADER_RE = _hdr
        _is_stub_content_fn = _stub


class TaggingPipeline:
    """Turn tagging, tag splitting, and history ingestion.

    Constructor takes:
        tag_generator:       TagGenerator instance
        turn_tag_index:      TurnTagIndex (shared mutable reference)
        store:               ContextStore instance
        semantic:            SemanticSearchManager instance
        engine_state:        EngineState (shared mutable dataclass)
        config:              VirtualContextConfig instance
        tag_splitter:        TagSplitter | None
        canonicalizer:       TagCanonicalizer | None
        telemetry:           TelemetryLedger instance
        monitor:             ContextMonitor instance
        compactor:           DomainCompactor | None
        save_state_callback: Callable[[list[Message]], None]
    """

    def __init__(
        self,
        tag_generator: TagGenerator,
        turn_tag_index: TurnTagIndex,
        store: ContextStore,
        semantic: SemanticSearchManager,
        engine_state: EngineState,
        config: VirtualContextConfig,
        tag_splitter: TagSplitter | None,
        canonicalizer: TagCanonicalizer | None,
        telemetry: TelemetryLedger,
        monitor: ContextMonitor,
        compactor: DomainCompactor | None,
        save_state_callback: Callable,
        next_tool_tag_callback: Callable[[], int] | None = None,
    ) -> None:
        self._tag_generator = tag_generator
        self._turn_tag_index = turn_tag_index
        self._store = store
        self._semantic = semantic
        self._engine_state = engine_state
        self.config = config
        self._tag_splitter = tag_splitter
        self._canonicalizer = canonicalizer
        self._telemetry = telemetry
        self._monitor = monitor
        self._compactor = compactor
        self._save_state_callback = save_state_callback
        self._next_tool_tag = next_tool_tag_callback

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tool_turn(messages: list[Message]) -> bool:
        """Check if a turn is tool-only: has tool blocks in raw_content and empty text content."""
        combined_text = " ".join(m.content for m in messages)
        if combined_text.strip():
            return False  # has real text content — use LLM tagger
        has_tool_block = False
        for m in messages:
            if not m.raw_content:
                continue
            for block in m.raw_content:
                if block.get("type") in ("tool_use", "tool_result"):
                    has_tool_block = True
                    break
            if has_tool_block:
                break
        return has_tool_block

    @staticmethod
    def _merge_role_messages(messages: list[Message], role: str) -> Message:
        selected = [msg for msg in messages if msg.role == role]
        if not selected:
            return Message(role=role, content="")
        raw_blocks: list[dict] = []
        for msg in selected:
            if msg.raw_content:
                raw_blocks.extend(msg.raw_content)
        return Message(
            role=role,
            content="\n".join(msg.content for msg in selected),
            timestamp=selected[-1].timestamp,
            metadata=selected[-1].metadata,
            raw_content=raw_blocks or selected[-1].raw_content,
        )

    @classmethod
    def _split_pair_messages(cls, messages: list[Message]) -> tuple[Message, Message]:
        return (
            cls._merge_role_messages(messages, "user"),
            cls._merge_role_messages(messages, "assistant"),
        )

    @classmethod
    def _flatten_context_pairs(cls, pairs: list) -> list[str] | None:
        flattened: list[str] = []
        for pair in pairs:
            user_msg, asst_msg = cls._split_pair_messages(pair.messages)
            flattened.extend([user_msg.content, asst_msg.content])
        return flattened if flattened else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _link_turn_tool_outputs(
        self,
        turn_number: int,
        refs: list[str] | None = None,
    ) -> None:
        """Link intercepted tool outputs to a canonical turn.

        ``refs`` is the preferred path: ingestion computes canonical turn
        ownership from parsed payload history and passes the matching tool
        output refs directly. When ``refs`` is omitted, fall back to the legacy
        request-time ``tool_outputs.turn`` lookup for compatibility with older
        single-turn flows.
        """
        try:
            if refs is None:
                refs = self._store.get_tool_output_refs_for_turn(
                    self.config.conversation_id, turn_number,
                )
            for ref in refs:
                self._store.link_turn_tool_output(
                    self.config.conversation_id, turn_number, ref,
                )
        except Exception:
            pass  # non-critical

    def _get_latest_turn_pair(self, history: list[Message]) -> list[Message] | None:
        """Extract the most recent completed user/assistant turn."""
        pairs = pair_messages_into_turns(list(history))
        if not pairs:
            return None
        latest = pairs[-1].messages
        if any(msg.role == "assistant" for msg in latest):
            return latest
        return None

    def _persist_canonical_turn(
        self,
        entry: "TurnTagEntry",
        user_msg: "Message",
        asst_msg: "Message",
        *,
        pair_messages: list["Message"] | None = None,
        existing_rows: list["CanonicalTurnRow"] | None = None,
        append_missing: bool = True,
    ) -> int:
        """Persist/enrich a single logical-turn pair.

        Returns the number of rows consumed from the head of ``existing_rows``
        (0 when ``existing_rows`` is None or was not used — e.g. the fallback
        hash-search / append path). Strict callers use the returned count to
        advance their shared cursor across ``existing_rows``.
        """
        if existing_rows is not None:
            consumed = self._persist_existing_canonical_rows(
                entry,
                pair_messages or [],
                existing_rows,
            )
            if consumed > 0:
                return consumed
            if not append_missing:
                raise RuntimeError(
                    "strict canonical tagging could not map payload messages to "
                    f"existing rows for logical turn {entry.turn_number}"
                )

        user_hash, user_norm, _ = compute_turn_hash_from_raw(
            user_msg.content,
            "",
            version=HASH_VERSION,
        )
        assistant_hash, _, assistant_norm = compute_turn_hash_from_raw(
            "",
            asst_msg.content,
            version=HASH_VERSION,
        )
        rows = list(self._store.get_all_canonical_turns(self.config.conversation_id))
        matched_pair: tuple[object, object] | None = None
        search_order: list[int] = []
        approx_idx = max(0, entry.turn_number * 2)
        for idx in range(max(0, approx_idx - 4), min(len(rows) - 1, approx_idx + 4) + 1):
            search_order.append(idx)
        for idx in range(0, len(rows) - 1):
            if idx not in search_order:
                search_order.append(idx)
        for idx in search_order:
            first = rows[idx]
            second = rows[idx + 1]
            if first.turn_hash == user_hash and second.turn_hash == assistant_hash:
                matched_pair = (first, second)
                break
        if matched_pair is not None:
            user_row, assistant_row = matched_pair
            tagged_at = utcnow_iso()
            self._store.save_canonical_turn(
                self.config.conversation_id,
                entry.turn_number,
                user_msg.content,
                "",
                user_raw_content=json.dumps(user_msg.raw_content) if user_msg.raw_content else None,
                assistant_raw_content=None,
                primary_tag=entry.primary_tag,
                tags=list(entry.tags or []),
                session_date=entry.session_date,
                sender=entry.sender,
                fact_signals=list(entry.fact_signals or []),
                code_refs=list(entry.code_refs or []),
                canonical_turn_id=user_row.canonical_turn_id,
                sort_key=user_row.sort_key,
                turn_hash=user_hash,
                hash_version=HASH_VERSION,
                normalized_user_text=user_norm,
                normalized_assistant_text="",
                tagged_at=tagged_at,
                compacted_at=user_row.compacted_at,
                first_seen_at=user_row.first_seen_at,
                last_seen_at=user_row.last_seen_at or tagged_at,
                source_batch_id=user_row.source_batch_id,
                created_at=user_row.created_at,
                updated_at=tagged_at,
                turn_group_number=entry.turn_number,
            )
            self._store.save_canonical_turn(
                self.config.conversation_id,
                entry.turn_number + 1,
                "",
                asst_msg.content,
                user_raw_content=None,
                assistant_raw_content=json.dumps(asst_msg.raw_content) if asst_msg.raw_content else None,
                primary_tag=entry.primary_tag,
                tags=list(entry.tags or []),
                session_date=entry.session_date,
                sender=entry.sender,
                fact_signals=list(entry.fact_signals or []),
                code_refs=list(entry.code_refs or []),
                canonical_turn_id=assistant_row.canonical_turn_id,
                sort_key=assistant_row.sort_key,
                turn_hash=assistant_hash,
                hash_version=HASH_VERSION,
                normalized_user_text="",
                normalized_assistant_text=assistant_norm,
                tagged_at=tagged_at,
                compacted_at=assistant_row.compacted_at,
                first_seen_at=assistant_row.first_seen_at,
                last_seen_at=assistant_row.last_seen_at or tagged_at,
                source_batch_id=assistant_row.source_batch_id,
                created_at=assistant_row.created_at,
                updated_at=tagged_at,
                turn_group_number=entry.turn_number,
            )
            entry.canonical_turn_id = user_row.canonical_turn_id or entry.canonical_turn_id
            # Fallback hash-search path: 0 rows consumed from
            # ``existing_rows`` (we bypassed it).
            return 0
        result = IngestReconciler(self._store, self._semantic).ingest_single(
            conversation_id=self.config.conversation_id,
            user_content=user_msg.content,
            assistant_content=asst_msg.content,
            user_raw_content=json.dumps(user_msg.raw_content) if user_msg.raw_content else None,
            assistant_raw_content=json.dumps(asst_msg.raw_content) if asst_msg.raw_content else None,
            primary_tag=entry.primary_tag,
            tags=list(entry.tags or []),
            session_date=entry.session_date,
            sender=entry.sender,
            fact_signals=list(entry.fact_signals or []),
            code_refs=list(entry.code_refs or []),
        )
        if result.rows:
            entry.canonical_turn_id = result.rows[0].canonical_turn_id or entry.canonical_turn_id
        # Append-missing path did not consume any ``existing_rows`` entry.
        return 0

    def _persist_existing_canonical_rows(
        self,
        entry: "TurnTagEntry",
        pair_messages: list["Message"],
        existing_rows: list["CanonicalTurnRow"],
    ) -> int:
        """Update pre-persisted canonical rows in-place for payload ingest.

        ``handle_prepare_payload`` persists canonical rows before the legacy
        pair-based tagger runs. During that follow-up tagging pass we must only
        enrich and mark those existing rows; appending fresh canonical rows here
        would inflate the DB-derived denominator mid-run.

        Returns the number of rows consumed from ``existing_rows`` (0 means
        "could not map" — callers in strict mode surface this as an error).
        The caller is responsible for advancing its cursor by the returned
        count.

        Matching is role-shape aware AND resilient to a prefix of untagged
        rows that has been left mid-pair by a crashed tagger on another
        worker. When a strict role-shape match at the head of ``existing_rows``
        fails, we skip over any leading "orphan halves" (rows whose role
        shape does not match the expected first payload message) before
        attempting the pair match again.
        """

        source_messages = [msg for msg in pair_messages if msg.role in {"user", "assistant"}]
        if not source_messages or not existing_rows:
            return 0
        sorted_rows = sorted(existing_rows, key=lambda row: (row.sort_key, row.canonical_turn_id))

        # Older conversations can still contain a single legacy canonical row
        # that stores the whole logical user/assistant turn. Strict tagging
        # should enrich that row in place instead of treating it as unmappable.
        if len(sorted_rows) == 1 and len(source_messages) >= 2:
            row = sorted_rows[0]
            user_msg, asst_msg = self._split_pair_messages(pair_messages)
            turn_hash, normalized_user_text, normalized_assistant_text = compute_turn_hash_from_raw(
                user_msg.content,
                asst_msg.content,
                version=row.hash_version or HASH_VERSION,
            )
            tagged_at = utcnow_iso()
            self._store.save_canonical_turn(
                self.config.conversation_id,
                entry.turn_number,
                user_msg.content,
                asst_msg.content,
                user_raw_content=json.dumps(user_msg.raw_content) if user_msg.raw_content else None,
                assistant_raw_content=json.dumps(asst_msg.raw_content) if asst_msg.raw_content else None,
                primary_tag=entry.primary_tag,
                tags=list(entry.tags or []),
                session_date=entry.session_date,
                sender=entry.sender,
                fact_signals=list(entry.fact_signals or []),
                code_refs=list(entry.code_refs or []),
                canonical_turn_id=row.canonical_turn_id,
                sort_key=row.sort_key,
                turn_hash=turn_hash,
                hash_version=row.hash_version or HASH_VERSION,
                normalized_user_text=normalized_user_text,
                normalized_assistant_text=normalized_assistant_text,
                tagged_at=tagged_at,
                compacted_at=row.compacted_at,
                first_seen_at=row.first_seen_at,
                last_seen_at=row.last_seen_at or tagged_at,
                source_batch_id=row.source_batch_id,
                created_at=row.created_at,
                updated_at=tagged_at,
                turn_group_number=row.turn_group_number,
            )
            entry.canonical_turn_id = row.canonical_turn_id or entry.canonical_turn_id
            return 1

        # Locate a role-shape-compatible contiguous window of
        # ``len(source_messages)`` rows inside ``sorted_rows``. This tolerates
        # a prefix of "orphan" rows left behind by a crashed tagger on another
        # worker (e.g. the user half of a prior pair was tagged but the
        # assistant half was not — ``iter_untagged_canonical_rows`` returns
        # the remaining assistant half as the first row, which would otherwise
        # misalign a naive slice-by-position cursor).
        needed = len(source_messages)
        if len(sorted_rows) < needed:
            return 0
        match_offset = -1
        for start in range(0, len(sorted_rows) - needed + 1):
            window = sorted_rows[start : start + needed]
            if all(
                self._row_shape_matches_role(row, message.role)
                for row, message in zip(window, source_messages, strict=True)
            ):
                match_offset = start
                break
        if match_offset < 0:
            return 0

        window = sorted_rows[match_offset : match_offset + needed]
        tagged_at = utcnow_iso()
        for row, message in zip(window, source_messages, strict=True):
            role = message.role
            user_content = message.content if role == "user" else ""
            assistant_content = message.content if role == "assistant" else ""

            turn_hash, normalized_user_text, normalized_assistant_text = compute_turn_hash_from_raw(
                user_content,
                assistant_content,
                version=row.hash_version or HASH_VERSION,
            )
            if role == "user":
                user_raw_content = json.dumps(message.raw_content) if message.raw_content else None
                assistant_raw_content = None
            else:
                user_raw_content = None
                assistant_raw_content = json.dumps(message.raw_content) if message.raw_content else None
            self._store.save_canonical_turn(
                self.config.conversation_id,
                entry.turn_number,
                user_content,
                assistant_content,
                user_raw_content=user_raw_content,
                assistant_raw_content=assistant_raw_content,
                primary_tag=entry.primary_tag,
                tags=list(entry.tags or []),
                session_date=entry.session_date,
                sender=entry.sender,
                fact_signals=list(entry.fact_signals or []),
                code_refs=list(entry.code_refs or []),
                canonical_turn_id=row.canonical_turn_id,
                sort_key=row.sort_key,
                turn_hash=turn_hash,
                hash_version=row.hash_version or HASH_VERSION,
                normalized_user_text=normalized_user_text,
                normalized_assistant_text=normalized_assistant_text,
                tagged_at=tagged_at,
                compacted_at=row.compacted_at,
                first_seen_at=row.first_seen_at,
                last_seen_at=row.last_seen_at or tagged_at,
                source_batch_id=row.source_batch_id,
                created_at=row.created_at,
                updated_at=tagged_at,
                turn_group_number=row.turn_group_number,
            )
        entry.canonical_turn_id = window[0].canonical_turn_id or entry.canonical_turn_id
        # Report rows consumed from the head of ``existing_rows``: the offset
        # we skipped over PLUS the pair we matched. The caller advances its
        # strict cursor by this count so subsequent pairs continue to pick
        # up where this one left off.
        return match_offset + needed

    @staticmethod
    def _row_shape_matches_role(row: "CanonicalTurnRow", role: str) -> bool:
        """True when ``row``'s user/assistant shape is compatible with ``role``.

        A row is "compatible" with a given role when:
        - It has the matching half populated AND the other half empty, OR
        - Both halves are empty (freshly reconciled row with neither half
          persisted yet — accepted so the tagger can fill it in), OR
        - Both halves are populated (a legacy combined row — accepted and
          the caller will overwrite the role's half in place).
        """
        row_has_user = bool((row.user_content or "").strip())
        row_has_assistant = bool((row.assistant_content or "").strip())
        if role == "user":
            return not (row_has_assistant and not row_has_user)
        if role == "assistant":
            return not (row_has_user and not row_has_assistant)
        return False

    def _get_recent_context(
        self, history: list[Message], n_pairs: int, exclude_last: int = 2,
        current_text: str | None = None,
    ) -> list[str] | None:
        """Collect up to *n_pairs* recent user+assistant text strings."""
        return get_recent_context(
            history,
            n_pairs,
            semantic=self._semantic,
            bleed_threshold=self.config.tag_generator.context_bleed_threshold,
            exclude_last=exclude_last,
            current_text=current_text,
        )

    def _collect_turn_text(
        self, tag: str, history: list[Message],
    ) -> list[tuple[int, str]]:
        """Collect truncated user text for turns tagged with the given tag."""
        pairs = extract_turn_pairs(history)
        result = []
        for entry in self._turn_tag_index.entries:
            if tag in entry.tags:
                if entry.turn_number < len(pairs):
                    text = pairs[entry.turn_number][0][:200]
                    result.append((entry.turn_number, text))
        return result

    def _build_broad_tag_summary(
        self, tag: str, history: list[Message],
    ) -> None:
        """Build a tag summary directly from raw turn text for unsplittable broad tags."""
        if not self._compactor:
            return

        from ..types import StoredSummary

        pairs = extract_turn_pairs(history)
        texts = []
        turn_numbers = []
        canonical_turn_ids: list[str] = []
        for entry in self._turn_tag_index.entries:
            if tag in entry.tags:
                if entry.turn_number < len(pairs):
                    user_text, assistant_text = pairs[entry.turn_number]
                    texts.append(
                        f"User: {user_text[:300]}\n"
                        f"Assistant: {assistant_text[:300]}"
                    )
                    turn_numbers.append(entry.turn_number)
                    if entry.canonical_turn_id:
                        canonical_turn_ids.append(entry.canonical_turn_id)

        if not texts:
            return

        combined = "\n\n---\n\n".join(texts)
        max_turn = max(turn_numbers) if turn_numbers else 0

        synthetic = [StoredSummary(
            ref=f"broad-{tag}",
            tags=[tag],
            summary=combined[:4000],
            summary_tokens=len(combined[:4000]) // 4,
        )]
        summaries = self._compactor.compact_tag_summaries(
            cover_tags=[tag],
            tag_to_summaries={tag: synthetic},
            tag_to_turns={tag: turn_numbers},
            tag_to_canonical_turn_ids={tag: canonical_turn_ids},
            existing_tag_summaries={},
            max_turn=max_turn,
        )
        for ts in summaries:
            self._store.save_tag_summary(ts, conversation_id=self.config.conversation_id)

    @staticmethod
    def _record_timing(
        breakdown: dict[str, float],
        stage: str,
        started_at: float,
    ) -> float:
        elapsed = round((time.monotonic() - started_at) * 1000, 1)
        breakdown[stage] = round(breakdown.get(stage, 0.0) + elapsed, 1)
        return elapsed

    def _log_breakdown(
        self,
        label: str,
        *,
        turn_number: int,
        total_ms: float,
        breakdown: dict[str, float],
        extras: list[str],
    ) -> None:
        if total_ms < _TAG_BREAKDOWN_LOG_THRESHOLD_MS:
            return
        stages = sorted(
            ((stage, ms) for stage, ms in breakdown.items() if ms > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:_TAG_BREAKDOWN_MAX_STAGES]
        stage_bits = [f"{stage}={ms:.1f}ms" for stage, ms in stages]
        parts = [*extras, *stage_bits]
        logger.info(
            "%s conv=%s turn=%d total=%.1fms %s",
            label,
            self.config.conversation_id[:12],
            turn_number,
            total_ms,
            " ".join(parts) if parts else "no-stages",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_turn(
        self,
        conversation_history: list[Message],
        payload_tokens: int | None = None,
        *,
        run_broad_split: bool = True,
        turn_number: int | None = None,
    ) -> CompactionSignal | None:
        """Phase 1 of turn processing: tag the latest turn and check thresholds.

        Fast (~2-3s with LLM tagger). Must complete before the next inbound
        request so the turn-tag index is up-to-date for retrieval.

        Returns a CompactionSignal if compaction is needed, None otherwise.

        *payload_tokens* (proxy mode): actual client payload token count.
        Overrides the stripped conversation_history token count in the
        compaction monitor so thresholds trigger at the right level.
        """
        from ..types import TagResult, TurnTagEntry, get_sender_name

        turn_number = (
            int(turn_number)
            if turn_number is not None
            else len(self._turn_tag_index.entries)
        )
        tag_started = time.monotonic()
        breakdown: dict[str, float] = {}

        # Tag the latest round trip
        latest_pair = self._get_latest_turn_pair(conversation_history)
        sender = get_sender_name(latest_pair[0].metadata) if latest_pair else ""
        if latest_pair:
            combined_text = " ".join(m.content for m in latest_pair)

            # Tool-only turns: skip LLM tagger, assign sequential tool_N tag
            if self._is_tool_turn(latest_pair):
                if self._next_tool_tag is not None:
                    tag_num = self._next_tool_tag()
                else:
                    self._engine_state.tool_tag_counter += 1
                    tag_num = self._engine_state.tool_tag_counter
                tag_name = f"tool_{tag_num}"
                self._turn_tag_index.append(TurnTagEntry(
                    turn_number=turn_number,
                    message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                    tags=[tag_name],
                    primary_tag=tag_name,
                    sender=sender or "",
                    session_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                ))
                latest_pair = None  # skip normal tagger flow below

            # BUG-013: Skip empty turns with no tool blocks
            elif not combined_text.strip():
                latest_pair = None

        if latest_pair:
            t_stage = time.monotonic()
            store_tags = [
                ts.tag for ts in self._store.get_all_tags(
                    conversation_id=self.config.conversation_id,
                )
            ]
            self._record_timing(breakdown, "load_store_tags", t_stage)
            n_context = self.config.tag_generator.context_lookback_pairs
            t_stage = time.monotonic()
            context = self._get_recent_context(
                conversation_history, n_context, current_text=combined_text,
            )
            self._record_timing(breakdown, "build_context", t_stage)
            t_stage = time.monotonic()
            tag_result = self._tag_generator.generate_tags(
                combined_text, store_tags, context_turns=context,
            )
            self._record_timing(breakdown, "generate_tags", t_stage)

            # Retry with expanded context if only _general was produced
            if tag_result.tags == ["_general"]:
                t_stage = time.monotonic()
                expanded = self._get_recent_context(
                    conversation_history, n_context * 2,
                    current_text=combined_text,
                )
                if expanded:
                    tag_result = self._tag_generator.generate_tags(
                        combined_text, store_tags, context_turns=expanded,
                    )
                self._record_timing(breakdown, "retry_general", t_stage)

            # Final fallback: inherit from most recent meaningful turn
            if tag_result.tags == ["_general"]:
                t_stage = time.monotonic()
                prev = self._turn_tag_index.latest_meaningful_tags()
                if prev:
                    tag_result = TagResult(
                        tags=list(prev.tags),
                        primary=prev.primary_tag,
                        source="inherited",
                    )
                self._record_timing(breakdown, "inherit_fallback", t_stage)

            t_stage = time.monotonic()
            self._turn_tag_index.append(TurnTagEntry(
                turn_number=turn_number,
                message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                tags=tag_result.tags,
                primary_tag=tag_result.primary,
                fact_signals=tag_result.fact_signals,
                code_refs=tag_result.code_refs,
                sender=sender or "",
                session_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            ))
            self._record_timing(breakdown, "append_turn_index", t_stage)

        # Build snapshot (only count un-compacted messages)
        t_stage = time.monotonic()
        _total_turns = len(self._turn_tag_index.entries) if self._turn_tag_index else None
        _offset = self._engine_state.history_offset(
            len(conversation_history), total_turns_indexed=_total_turns,
        )
        snapshot = self._monitor.build_snapshot(
            conversation_history[_offset:],
            payload_tokens=payload_tokens,
        )
        self._record_timing(breakdown, "build_snapshot", t_stage)

        # Check thresholds
        t_stage = time.monotonic()
        signal = self._monitor.check(snapshot)
        self._record_timing(breakdown, "monitor_check", t_stage)

        if signal is None:
            self._engine_state.last_compact_ms = 0.0
            # Persist turn message text for post-restart recall
            if latest_pair:
                turn_num = turn_number
                entry = self._turn_tag_index.get_tags_for_logical_turn(turn_num)
                t_stage = time.monotonic()
                # Persistence failures in the live-tagging path must not stop
                # progressive tagging (the next turn will re-attempt), but
                # they MUST be visible. Never silently ``pass`` — log with
                # full context so coherence bugs don't hide behind an
                # empty except clause.
                try:
                    if entry is not None:
                        self._persist_canonical_turn(entry, latest_pair[0], latest_pair[1])
                except StaleConversationWriteError as exc:
                    logger.info(
                        "TAGGER turn=%d canonical-persist deferred (stale): %s",
                        turn_num, exc,
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    logger.error(
                        "TAGGER turn=%d canonical-persist failed (structural): %s",
                        turn_num, exc,
                        exc_info=True,
                    )
                self._link_turn_tool_outputs(turn_num)
                self._record_timing(breakdown, "persist_turn_message", t_stage)
            t_stage = time.monotonic()
            self._save_state_callback(
                conversation_history,
                last_indexed_turn=turn_number,
            )
            self._record_timing(breakdown, "save_state", t_stage)

        if self._tag_splitter and run_broad_split:
            t_stage = time.monotonic()
            self.process_broad_tag_split(conversation_history, mode="inline")
            self._record_timing(breakdown, "broad_split", t_stage)

        total_ms = round((time.monotonic() - tag_started) * 1000, 1)
        self._engine_state.last_tag_ms = total_ms
        self._log_breakdown(
            "TAG_BREAKDOWN",
            turn_number=turn_number,
            total_ms=total_ms,
            breakdown=breakdown,
            extras=[
                f"history={len(conversation_history)}",
                f"payload={payload_tokens if payload_tokens is not None else 'na'}t",
                f"split_mode={'inline' if self._tag_splitter and run_broad_split else 'deferred' if self._tag_splitter else 'disabled'}",
                f"signal={signal.priority if signal is not None else 'none'}",
            ],
        )

        return signal

    def _check_and_split_broad_tags(
        self, conversation_history: list[Message],
    ) -> SplitResult | None:
        return self.process_broad_tag_split(conversation_history, mode="direct")

    def process_broad_tag_split(
        self,
        conversation_history: list[Message],
        *,
        mode: str = "deferred",
    ) -> SplitResult | None:
        """Check for overly-broad tags and split or summarize them."""
        if not self._tag_splitter:
            return None

        split_started = time.monotonic()
        breakdown: dict[str, float] = {}
        cfg = self.config.tag_generator.tag_splitting
        t_stage = time.monotonic()
        tag_counts = self._turn_tag_index.get_tag_counts()
        total_turns = len(self._turn_tag_index.entries)
        self._record_timing(breakdown, "find_candidates", t_stage)

        if total_turns == 0:
            return None

        # Find candidates: above both thresholds, not already processed
        candidates = [
            (tag, count) for tag, count in tag_counts.items()
            if tag != "_general"
            and tag not in self._engine_state.split_processed_tags
            and count >= cfg.frequency_threshold
            and count / total_turns >= cfg.frequency_pct_threshold
        ]

        if not candidates:
            return None

        # Pick highest-frequency first
        candidates.sort(key=lambda x: -x[1])
        tag, count = candidates[0]

        # Collect turn content
        t_stage = time.monotonic()
        turn_contents = self._collect_turn_text(tag, conversation_history)
        self._record_timing(breakdown, "collect_turn_text", t_stage)
        if not turn_contents:
            self._engine_state.split_processed_tags.add(tag)
            t_stage = time.monotonic()
            self._save_state_callback(
                conversation_history,
                last_indexed_turn=len(self._turn_tag_index.entries) - 1,
            )
            self._record_timing(breakdown, "save_state", t_stage)
            return None

        t_stage = time.monotonic()
        existing_tags = {t for e in self._turn_tag_index.entries for t in e.tags}
        self._record_timing(breakdown, "collect_existing_tags", t_stage)
        t_stage = time.monotonic()
        result = self._tag_splitter.split(tag, turn_contents, existing_tags, total_turns)
        self._record_timing(breakdown, "split_llm", t_stage)

        if result.splittable:
            # Apply split to TurnTagIndex
            t_stage = time.monotonic()
            turn_to_new: dict[int, list[str]] = {}
            for new_tag, turn_numbers in result.groups.items():
                for tn in turn_numbers:
                    turn_to_new.setdefault(tn, []).append(new_tag)
            self._turn_tag_index.replace_tag(tag, turn_to_new)

            # Register alias so old tag queries still resolve
            if self._canonicalizer:
                first_new = next(iter(result.groups))
                self._canonicalizer.register_alias(tag, first_new)

            # Update tagger vocabulary
            if hasattr(self._tag_generator, '_tag_vocabulary'):
                self._tag_generator._tag_vocabulary.pop(tag, None)
                for new_tag, turns in result.groups.items():
                    self._tag_generator._tag_vocabulary[new_tag] = len(turns)
            self._record_timing(breakdown, "apply_split", t_stage)

            logger.info(
                "Split '%s' (%d turns) → %s",
                tag, count, list(result.groups.keys()),
            )
        else:
            # Fallback: build tag summary from raw turn text
            t_stage = time.monotonic()
            self._build_broad_tag_summary(tag, conversation_history)
            self._record_timing(breakdown, "build_summary", t_stage)
            logger.info(
                "Tag '%s' unsplittable (%s), built summary", tag, result.reason,
            )

        self._engine_state.split_processed_tags.add(tag)
        self._engine_state.last_split_result = result
        t_stage = time.monotonic()
        self._save_state_callback(
            conversation_history,
            last_indexed_turn=len(self._turn_tag_index.entries) - 1,
        )
        self._record_timing(breakdown, "save_state", t_stage)
        total_ms = round((time.monotonic() - split_started) * 1000, 1)
        self._log_breakdown(
            "TAG_SPLIT_BREAKDOWN",
            turn_number=len(self._turn_tag_index.entries) - 1,
            total_ms=total_ms,
            breakdown=breakdown,
            extras=[
                f"mode={mode}",
                f"tag={tag}",
                f"candidate_turns={count}",
                f"total_turns={total_turns}",
                f"result={'split' if result.splittable else 'summary'}",
            ],
        )
        return result

    def tag_canonical_row(self, row: "CanonicalTurnRow") -> None:
        """Tag a single canonical row in-place (background backfill entry).

        Unlike ``tag_turn``, this is designed for the background tagger loop
        which processes REST-ingested rows after ``IngestReconciler`` has
        persisted them without tags. It runs the tag generator on the row's
        content, assigns a primary tag + tag list, and re-saves the row via
        ``save_canonical_turn`` preserving all other fields. It does NOT:

        * build or consult ``conversation_history`` (one-row scope only),
        * fire compaction signals (``_monitor.check`` is intentionally
          skipped — those signals belong to the live ``tag_turn`` flow),
        * mutate ``TurnTagIndex`` (the index is owned by proxy-flow callers
          via ``ingest_history`` and ``tag_turn``),
        * stamp ``tagged_at`` (the outer ``_tagger_run`` loop flips that
          atomically via ``mark_canonical_row_tagged`` on the same row).

        Row text selection: canonical rows carry one side only
        (user_content OR assistant_content) per the split-ingest layout in
        ``IngestReconciler._prepare_message_row``. We concatenate both sides
        so the tag generator sees whichever is populated.
        """
        _ensure_engine_imports()

        from ..types import TagResult

        user_text = (row.user_content or "").strip()
        asst_text = (row.assistant_content or "").strip()
        combined_text = f"{user_text} {asst_text}".strip()

        # Fast-path: empty or stub content → assign `_stub` and persist.
        if not combined_text or (
            _is_stub_content_fn is not None and _is_stub_content_fn(combined_text)
        ):
            tag_result = TagResult(tags=["_stub"], primary="_stub", source="stub")
        else:
            store_tags = [
                ts.tag for ts in self._store.get_all_tags(
                    conversation_id=self.config.conversation_id,
                )
            ]
            try:
                tag_result = self._tag_generator.generate_tags(
                    combined_text, store_tags, context_turns=None,
                )
            except Exception:
                logger.exception(
                    "tag_canonical_row: generator failed conv=%s turn_id=%s",
                    self.config.conversation_id[:12],
                    row.canonical_turn_id[:12] if row.canonical_turn_id else "?",
                )
                raise

        # Preserve session_date if already set on the row; else try to extract
        # from content so a background-tagged row carries the same date hint
        # the proxy flow produces.
        session_date = (row.session_date or "").strip()
        if not session_date:
            extracted = _extract_session_date_from_content(row.user_content or "")
            if extracted:
                session_date = extracted

        # Persist the enriched fields. We DO NOT set ``tagged_at`` here —
        # the outer ``_tagger_run`` loop owns the epoch-guarded flip via
        # ``mark_canonical_row_tagged``. We re-supply every existing field so
        # the UPSERT does not clobber pre-existing metadata.
        self._store.save_canonical_turn(
            self.config.conversation_id,
            row.turn_number,
            row.user_content,
            row.assistant_content,
            user_raw_content=row.user_raw_content,
            assistant_raw_content=row.assistant_raw_content,
            primary_tag=tag_result.primary,
            tags=list(tag_result.tags),
            session_date=session_date,
            sender=row.sender,
            fact_signals=list(tag_result.fact_signals) or list(row.fact_signals),
            code_refs=list(tag_result.code_refs) or list(row.code_refs),
            canonical_turn_id=row.canonical_turn_id,
            sort_key=row.sort_key,
            turn_hash=row.turn_hash,
            hash_version=row.hash_version,
            normalized_user_text=row.normalized_user_text,
            normalized_assistant_text=row.normalized_assistant_text,
            tagged_at=row.tagged_at,  # intentionally None — outer loop flips it
            compacted_at=row.compacted_at,
            first_seen_at=row.first_seen_at,
            last_seen_at=row.last_seen_at,
            source_batch_id=row.source_batch_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            turn_group_number=row.turn_group_number,
        )

    def ingest_history(
        self,
        history_messages: list[Message],
        progress_callback: Callable[..., None] | None = None,
        turn_offset: int = 0,
        tool_output_refs_by_turn: dict[int, list[str]] | None = None,
        require_existing_canonical: bool = False,
        expected_lifecycle_epoch: int | None = None,
    ) -> int:
        """Bootstrap TurnTagIndex from pre-existing conversation history.

        Groups the message stream into local turns and appends tagged entries
        to the live index.
        Does NOT trigger compaction — the next on_turn_complete() handles that.

        Args:
            history_messages: Ingestible message stream for the conversation.
            progress_callback: Optional ``(done, total, entry)`` called after
                each turn is ingested.  Used by the proxy for live progress.
            turn_offset: Global turn number of the first turn. Used by catch-up
                ingestion to prevent TurnTagIndex overwrites when multiple
                batches are ingested sequentially.
            tool_output_refs_by_turn: Mapping of batch-local turn index to
                intercepted tool-output refs discovered from the raw payload.

        Returns:
            Number of turns ingested.
        """
        import sys as _sys
        import time as _time

        from ..types import TagResult, TurnTagEntry, get_sender_name

        _ensure_engine_imports()

        _tag_start = _time.time()

        store_tags = [ts.tag for ts in self._store.get_all_tags(conversation_id=self.config.conversation_id)]
        ingested = 0
        history_turns = pair_messages_into_turns(list(history_messages))
        _total_turns = len(history_turns)
        n_context = self.config.tag_generator.context_lookback_pairs
        strict_rows: list["CanonicalTurnRow"] = []
        strict_row_cursor = 0
        if require_existing_canonical:
            expected_entries = sum(
                1 for msg in history_messages if msg.role in {"user", "assistant"}
            )
            if expected_lifecycle_epoch is not None:
                strict_rows = list(
                    self._store.iter_untagged_canonical_rows(
                        conversation_id=self.config.conversation_id,
                        expected_lifecycle_epoch=expected_lifecycle_epoch,
                        batch_size=max(expected_entries, 32),
                    )
                )
            else:
                strict_rows = [
                    row
                    for row in self._store.get_all_canonical_turns(self.config.conversation_id)
                    if not row.tagged_at
                ]
            strict_covered_entries = sum(
                max(1, int(getattr(row, "covered_ingestible_entries", 1) or 1))
                for row in strict_rows
            )
            if strict_covered_entries < expected_entries:
                raise RuntimeError(
                    "strict canonical tagging expected at least "
                    f"{expected_entries} covered ingestible entries, found "
                    f"{strict_covered_entries} across {len(strict_rows)} rows"
                )
        # Seed from DB when resuming bulk ingest mid-conversation: the caller
        # passes turn_offset > 0 when there are already-persisted turns that
        # preceded this batch. Without seeding, the first few turns of the
        # new batch fall into the no-session bucket even when the prior batch
        # carried a session_date the next turn would otherwise inherit.
        running_session_date = ""
        if turn_offset > 0:
            try:
                prior_rows = self._store.get_all_canonical_turns(self.config.conversation_id)
            except Exception:
                prior_rows = []
            for row in reversed(prior_rows):
                if row.turn_number < turn_offset and (row.session_date or "").strip():
                    running_session_date = row.session_date.strip()
                    break

        # Strict-mode cursor windowing: previously we sliced ``strict_rows``
        # into a fixed ``pair_row_count`` window per payload pair and advanced
        # the cursor by that fixed count. That works when the untagged row
        # list is perfectly pair-aligned — which breaks when a prior tagger
        # on another worker tagged one half of a pair and then crashed
        # before tagging the other half. ``iter_untagged_canonical_rows``
        # returns the orphan half as the first row, misaligning the cursor
        # with the payload's pair structure for every subsequent pair.
        #
        # Fix: pass a larger window (``pair_row_count + STRICT_WINDOW_SLACK``)
        # so ``_persist_existing_canonical_rows`` can skip orphan halves at
        # the head of the window before matching, and then advance the cursor
        # by the number of rows the persister actually consumed (return
        # value from ``_persist_canonical_turn``).
        STRICT_WINDOW_SLACK = 4

        for batch_turn, pair in enumerate(history_turns):
            user_msg, asst_msg = self._split_pair_messages(pair.messages)
            strict_pair_rows: list["CanonicalTurnRow"] | None = None
            pair_row_count = 0
            if require_existing_canonical:
                pair_row_count = sum(
                    1 for msg in pair.messages if msg.role in {"user", "assistant"}
                )
                strict_pair_rows = strict_rows[
                    strict_row_cursor: strict_row_cursor
                    + pair_row_count
                    + STRICT_WINDOW_SLACK
                ]
            turn_tool_refs = None
            if tool_output_refs_by_turn is not None:
                turn_tool_refs = tool_output_refs_by_turn.get(batch_turn, [])

            sender = ""
            for pair_msg in pair.messages:
                sender = get_sender_name(pair_msg.metadata) if pair_msg.metadata else ""
                if sender:
                    break

            # Tool-only turns: skip LLM tagger, assign sequential tool_N tag
            if self._is_tool_turn(pair.messages):
                if self._next_tool_tag is not None:
                    tag_num = self._next_tool_tag()
                else:
                    self._engine_state.tool_tag_counter += 1
                    tag_num = self._engine_state.tool_tag_counter
                tag_name = f"tool_{tag_num}"
                combined_text = " ".join(msg.content for msg in pair.messages)
                entry = TurnTagEntry(
                    turn_number=turn_offset + batch_turn,
                    message_hash=hashlib.sha256(
                        combined_text.encode()
                    ).hexdigest()[:16],
                    tags=[tag_name],
                    primary_tag=tag_name,
                    sender=sender or "",
                    session_date=running_session_date,
                )
                self._turn_tag_index.append(entry)
                try:
                    consumed = self._persist_canonical_turn(
                        entry,
                        user_msg,
                        asst_msg,
                        pair_messages=pair.messages,
                        existing_rows=strict_pair_rows,
                        append_missing=not require_existing_canonical,
                    )
                except RuntimeError:
                    # Strict-mode mapping failure — propagate for
                    # ``require_existing_canonical`` callers so the
                    # background ingestion thread can back off and let the
                    # lease transition to another worker. Never swallow.
                    if require_existing_canonical:
                        raise
                    consumed = 0
                if require_existing_canonical and strict_pair_rows is not None:
                    strict_row_cursor += consumed if consumed else pair_row_count
                self._link_turn_tool_outputs(entry.turn_number, turn_tool_refs)
                ingested += 1
                continue

            # BUG-013: Skip empty turns with no tool blocks
            if not user_msg.content.strip() and not asst_msg.content.strip():
                logger.debug("Skipping empty turn at turn index %d", batch_turn)
                continue

            # Track running session date BEFORE stub/tagger — stubs need
            # timestamps too. Priority: (1) explicit header/envelope in content,
            # (2) message timestamp from payload metadata (live proxy wall-clock
            # or REST-payload conversation time), (3) inherit from the prior
            # turn. When inheriting, bump by +1s on ISO-parseable dates so
            # consecutive turns don't collapse onto the same timestamp and
            # lose their ordering signal. We never synthesize a "now" value —
            # if nothing is known and running_session_date is empty, it stays
            # empty, and the turn is written with an empty session_date
            # rather than a misleading ingestion-time placeholder.
            extracted = _extract_session_date_from_content(user_msg.content)
            if extracted:
                running_session_date = extracted
            elif user_msg.timestamp:
                running_session_date = user_msg.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            elif running_session_date:
                bumped = _bump_session_date_one_second(running_session_date)
                if bumped:
                    running_session_date = bumped

            # Stub turns (media attachments, image placeholders, etc.):
            # skip tagger, assign _stub tag, preserve raw text for passthrough.
            combined_for_stub = f"{user_msg.content} {asst_msg.content}"
            if _is_stub_content_fn(combined_for_stub):
                entry = TurnTagEntry(
                    turn_number=turn_offset + batch_turn,
                    message_hash=hashlib.sha256(combined_for_stub.encode()).hexdigest()[:16],
                    tags=["_stub"],
                    primary_tag="_stub",
                    sender=sender or "",
                    session_date=running_session_date,
                )
                self._turn_tag_index.append(entry)
                try:
                    consumed = self._persist_canonical_turn(
                        entry,
                        user_msg,
                        asst_msg,
                        pair_messages=pair.messages,
                        existing_rows=strict_pair_rows,
                        append_missing=not require_existing_canonical,
                    )
                except RuntimeError:
                    if require_existing_canonical:
                        raise
                    consumed = 0
                if require_existing_canonical and strict_pair_rows is not None:
                    strict_row_cursor += consumed if consumed else pair_row_count
                self._link_turn_tool_outputs(entry.turn_number, turn_tool_refs)
                ingested += 1
                logger.info(
                    "TAGGER turn=%d STUB content_len=%d preview=\"%s\"",
                    turn_offset + batch_turn, len(combined_for_stub),
                    combined_for_stub[:60].replace("\n", " "),
                )
                if progress_callback:
                    progress_callback(ingested, _total_turns, entry)
                continue

            combined_text = f"{user_msg.content} {asst_msg.content}"
            _turn_num = turn_offset + batch_turn  # global turn number
            _content_preview = combined_text[:60].replace("\n", " ")

            # Flag short content that may be dominated by context
            if len(combined_text) < 80:
                logger.info(
                    "TAGGER turn=%d SHORT_CONTENT len=%d \"%s\"",
                    _turn_num, len(combined_text), _content_preview,
                )

            # Build context from preceding pairs in the flat history
            context: list[str] | None = None
            if batch_turn > 0:
                start = max(0, batch_turn - n_context)
                context = self._flatten_context_pairs(history_turns[start:batch_turn])

            # Context bleed gate (BUG-010): skip stale context on topic shift
            _bleed_gate = "no_context"
            _bleed_sim = -1.0
            if (
                context
                and self.config.tag_generator.context_bleed_threshold > 0
            ):
                _relevant, _bleed_sim = self._semantic.context_is_relevant_with_score(combined_text, context)
                if not _relevant:
                    _bleed_gate = f"BLOCKED (similarity={_bleed_sim:.2f} threshold={self.config.tag_generator.context_bleed_threshold})"
                    context = None
                else:
                    _bleed_gate = f"passed (similarity={_bleed_sim:.2f})"
            elif context:
                _bleed_gate = "disabled"

            _ctx_preview = context[-2][:60].replace("\n", " ") if context and len(context) >= 2 else ""
            logger.info(
                "TAGGER turn=%d content_len=%d content_preview=\"%s\" "
                "context_pairs=%d context_preview=\"%s\" bleed_gate=%s",
                _turn_num, len(combined_text), _content_preview,
                len(context) // 2 if context else 0, _ctx_preview, _bleed_gate,
            )

            tag_result = self._tag_generator.generate_tags(
                combined_text, store_tags, context_turns=context,
            )

            logger.info(
                "TAGGER turn=%d result primary=%s tags=%s source=%s",
                _turn_num, tag_result.primary, sorted(tag_result.tags), tag_result.source,
            )

            # Retry with expanded context on _general
            if tag_result.tags == ["_general"] and batch_turn > 0:
                expanded_start = max(0, batch_turn - (n_context * 2))
                expanded_ctx = self._flatten_context_pairs(history_turns[expanded_start:batch_turn]) or []
                # Gate expanded context too
                _expanded_gate = "no_context"
                if (
                    expanded_ctx
                    and self.config.tag_generator.context_bleed_threshold > 0
                ):
                    _rel, _sim = self._semantic.context_is_relevant_with_score(combined_text, expanded_ctx)
                    if not _rel:
                        _expanded_gate = f"BLOCKED (similarity={_sim:.2f})"
                        expanded_ctx = []
                    else:
                        _expanded_gate = f"passed (similarity={_sim:.2f})"
                if expanded_ctx:
                    logger.info(
                        "TAGGER turn=%d retry=expanded_context expanded_pairs=%d bleed_gate=%s",
                        _turn_num, len(expanded_ctx) // 2, _expanded_gate,
                    )
                    tag_result = self._tag_generator.generate_tags(
                        combined_text, store_tags, context_turns=expanded_ctx,
                    )
                    logger.info(
                        "TAGGER turn=%d retry_result primary=%s tags=%s source=%s",
                        _turn_num, tag_result.primary, sorted(tag_result.tags), tag_result.source,
                    )

            # Final fallback: inherit from most recent meaningful turn
            if tag_result.tags == ["_general"]:
                prev = self._turn_tag_index.latest_meaningful_tags()
                if prev:
                    tag_result = TagResult(
                        tags=list(prev.tags),
                        primary=prev.primary_tag,
                        source="inherited",
                    )
                    logger.info(
                        "TAGGER turn=%d fallback=inherited from_turn=%d tags=%s",
                        _turn_num, prev.turn_number, sorted(prev.tags),
                    )

            entry = TurnTagEntry(
                turn_number=turn_offset + batch_turn,
                message_hash=hashlib.sha256(combined_text.encode()).hexdigest()[:16],
                tags=list(tag_result.tags or []),
                primary_tag=tag_result.primary,
                fact_signals=list(tag_result.fact_signals or []),
                code_refs=list(tag_result.code_refs or []),
                sender=sender or "",
                session_date=running_session_date,
            )
            self._turn_tag_index.append(entry)
            try:
                consumed = self._persist_canonical_turn(
                    entry,
                    user_msg,
                    asst_msg,
                    pair_messages=pair.messages,
                    existing_rows=strict_pair_rows,
                    append_missing=not require_existing_canonical,
                )
            except RuntimeError:
                if require_existing_canonical:
                    raise
                consumed = 0
            if require_existing_canonical and strict_pair_rows is not None:
                strict_row_cursor += consumed if consumed else pair_row_count
            self._link_turn_tool_outputs(entry.turn_number, turn_tool_refs)
            ingested += 1

            # Stderr progress for visibility
            if ingested % 5 == 0 or ingested == _total_turns:
                _elapsed = _time.time() - _tag_start
                _rate = ingested / _elapsed if _elapsed > 0 else 0
                _eta = int((_total_turns - ingested) / _rate) if _rate > 0 else 0
                _sys.stderr.write(
                    f"\r  TAGGING: {ingested}/{_total_turns} turns | "
                    f"{_rate:.1f} turn/s | ETA {_eta}s   "
                )
                _sys.stderr.flush()

            if progress_callback:
                progress_callback(ingested, _total_turns, entry)

            # Refresh store tags every 10 turns so new tags influence later tagging
            if ingested % 10 == 0:
                store_tags = [ts.tag for ts in self._store.get_all_tags(conversation_id=self.config.conversation_id)]

            # Periodic state save so session_date + tags are queryable during ingestion
            if ingested % 20 == 0:
                checkpoint_turn = turn_offset + ingested - 1
                self._save_state_callback(
                    history_messages,
                    last_completed_turn=checkpoint_turn,
                    last_indexed_turn=checkpoint_turn,
                )

        # Final save after all turns ingested
        final_turn = turn_offset + ingested - 1
        self._save_state_callback(
            history_messages,
            last_completed_turn=final_turn,
            last_indexed_turn=final_turn,
        )
        _sys.stderr.write("\n")
        _sys.stderr.flush()
        logger.info("Ingested %d historical turns into TurnTagIndex", ingested)
        return ingested
