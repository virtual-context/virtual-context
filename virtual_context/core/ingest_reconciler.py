"""Canonical turn ingest reconciler for proxy and REST paths."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ..core.canonical_turns import (
    CanonicalIngestResult,
    HASH_VERSION,
    build_anchor_index,
    compute_anchor_hash,
    compute_turn_hash_from_raw,
    default_sort_key,
    generate_canonical_turn_id,
    utcnow_iso,
)
from ..core.semantic_search import SemanticSearchManager
from ..core.store import ContextStore
from ..types import FactSignal, CanonicalTurnRow, IngestBatchRecord, TurnTagEntry


@dataclass
class _Alignment:
    existing_start: int
    incoming_start: int
    overlap_len: int
    window_size: int
    merge_mode: str


logger = logging.getLogger(__name__)


class IngestReconciler:
    """Merges inbound turns into canonical turn storage."""

    def __init__(self, store: ContextStore, semantic: SemanticSearchManager) -> None:
        self._store = store
        self._semantic = semantic

    def ingest_single(
        self,
        conversation_id: str,
        *,
        user_content: str,
        assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
        primary_tag: str = "_general",
        tags: list[str] | None = None,
        session_date: str = "",
        sender: str = "",
        fact_signals: list[FactSignal] | None = None,
        code_refs: list[dict] | None = None,
    ) -> CanonicalIngestResult:
        with self._conversation_merge_lock(conversation_id):
            existing = self._store.get_all_canonical_turns(conversation_id)
            prepared = [
                self._prepare_message_row(
                    conversation_id,
                    role="user",
                    content=user_content,
                    raw_content=user_raw_content,
                    primary_tag=primary_tag,
                    tags=tags,
                    session_date=session_date,
                    sender=sender,
                    fact_signals=fact_signals,
                    code_refs=code_refs,
                ),
                self._prepare_message_row(
                    conversation_id,
                    role="assistant",
                    content=assistant_content,
                    raw_content=assistant_raw_content,
                    primary_tag=primary_tag,
                    tags=tags,
                    session_date=session_date,
                    sender=sender,
                    fact_signals=fact_signals,
                    code_refs=code_refs,
                ),
            ]
            if len(existing) >= len(prepared):
                recent_window = existing[-min(5, len(existing)):]
                recent = recent_window[-len(prepared):]
                if (
                    [row.turn_hash for row in recent] == [row.turn_hash for row in prepared]
                    and all(self._seen_recently(row.last_seen_at or "") for row in recent)
                ):
                    for idx, row in enumerate(prepared):
                        existing_row = recent[idx]
                        row.canonical_turn_id = existing_row.canonical_turn_id
                        row.sort_key = existing_row.sort_key
                        row.first_seen_at = existing_row.first_seen_at or row.first_seen_at
                        row.last_seen_at = utcnow_iso()
                        self._preserve_existing_enrichment(row, existing_row)
                        self._write_turn(
                            row,
                            turn_number=self._ordinal_for_row(existing, existing_row.canonical_turn_id),
                            first_seen_at=row.first_seen_at,
                            last_seen_at=row.last_seen_at,
                        )
                    self._refresh_persisted_anchors(conversation_id)
                    return CanonicalIngestResult(
                        merge_mode="exact_resend",
                        turns_written=0,
                        turns_matched=len(prepared),
                        turns_appended=0,
                        turns_prepended=0,
                        turns_inserted=0,
                        rows=recent,
                    )
            return self._ingest_prepared_turns_locked(
                conversation_id,
                prepared_turns=prepared,
                raw_turn_count=len(prepared),
                existing=existing,
            )

    def ingest_batch(
        self,
        conversation_id: str,
        *,
        body: dict,
        fmt: Any,
    ) -> CanonicalIngestResult:
        from ..proxy.formats import extract_ingestible_messages

        entries, _stats = extract_ingestible_messages(body, fmt)
        prepared = [
            self._prepare_message_row(
                conversation_id,
                role=message.role,
                content=message.content,
                raw_content=message.raw_content,
            )
            for message in entries
        ]
        return self.ingest_prepared_turns(
            conversation_id,
            prepared_turns=prepared,
            raw_turn_count=len(prepared),
        )

    def ingest_prepared_turns(
        self,
        conversation_id: str,
        *,
        prepared_turns: list[CanonicalTurnRow],
        raw_turn_count: int,
    ) -> CanonicalIngestResult:
        with self._conversation_merge_lock(conversation_id):
            existing = self._store.get_all_canonical_turns(conversation_id)
            return self._ingest_prepared_turns_locked(
                conversation_id,
                prepared_turns=prepared_turns,
                raw_turn_count=raw_turn_count,
                existing=existing,
            )

    def _ingest_prepared_turns_locked(
        self,
        conversation_id: str,
        *,
        prepared_turns: list[CanonicalTurnRow],
        raw_turn_count: int,
        existing: list[CanonicalTurnRow],
    ) -> CanonicalIngestResult:
        if not prepared_turns:
            logger.info(
                "INGEST_EMPTY_PAYLOAD: conv=%s raw_turn_count=%d",
                conversation_id[:12],
                raw_turn_count,
            )
            batch = self._save_batch(
                conversation_id,
                raw_turn_count=0,
                merge_mode="empty_payload",
                first_turn_hash="",
                last_turn_hash="",
                turns_matched=0,
                turns_appended=0,
                turns_prepended=0,
                turns_inserted=0,
            )
            return CanonicalIngestResult("empty_payload", 0, 0, 0, 0, 0, batch=batch, rows=[])

        alignment = self._find_alignment(conversation_id, existing, prepared_turns)
        merge_mode = alignment.merge_mode if alignment else "no_overlap_append"
        if alignment is None and existing and prepared_turns:
            logger.warning(
                "CANONICAL_TURN_NO_ALIGNMENT: conv=%s existing=%d incoming=%d -> no_overlap_append",
                conversation_id[:12],
                len(existing),
                len(prepared_turns),
            )
        turns_written = 0
        turns_matched = 0
        turns_appended = 0
        turns_prepended = 0
        turns_inserted = 0
        batch_id = generate_canonical_turn_id()
        now = utcnow_iso()
        rows_touched: list[CanonicalTurnRow] = []

        if not existing:
            merge_mode = "no_overlap_append"
            for idx, row in enumerate(prepared_turns):
                row.canonical_turn_id = generate_canonical_turn_id()
                row.sort_key = float((idx + 1) * 1000.0)
                row.source_batch_id = batch_id
                row.last_seen_at = now
                self._write_turn(row, turn_number=idx)
                rows_touched.append(row)
                turns_written += 1
                turns_appended += 1
        elif alignment is None:
            start_key = default_sort_key(existing)
            for idx, row in enumerate(prepared_turns):
                row.canonical_turn_id = generate_canonical_turn_id()
                row.sort_key = start_key + (1000.0 * idx)
                row.source_batch_id = batch_id
                row.last_seen_at = now
                self._write_turn(row, turn_number=len(existing) + idx)
                rows_touched.append(row)
                turns_written += 1
                turns_appended += 1
        else:
            overlap_existing = existing[alignment.existing_start:alignment.existing_start + alignment.overlap_len]
            overlap_incoming = prepared_turns[alignment.incoming_start:alignment.incoming_start + alignment.overlap_len]
            for offset, row in enumerate(overlap_incoming):
                existing_row = overlap_existing[offset]
                row.canonical_turn_id = existing_row.canonical_turn_id
                row.sort_key = existing_row.sort_key
                row.source_batch_id = batch_id
                row.first_seen_at = existing_row.first_seen_at or row.first_seen_at
                row.last_seen_at = now
                self._preserve_existing_enrichment(row, existing_row)
                self._write_turn(row, turn_number=alignment.existing_start + offset)
                rows_touched.append(row)
                turns_matched += 1

            prefix = prepared_turns[:alignment.incoming_start]
            if prefix:
                left_key = existing[alignment.existing_start - 1].sort_key if alignment.existing_start > 0 else None
                right_key = existing[alignment.existing_start].sort_key
                for row, key in zip(prefix, self._allocate_sort_keys(left_key, right_key, len(prefix))):
                    row.canonical_turn_id = generate_canonical_turn_id()
                    row.sort_key = key
                    row.source_batch_id = batch_id
                    row.last_seen_at = now
                    self._write_turn(row, turn_number=-1)
                    rows_touched.append(row)
                    turns_written += 1
                    if merge_mode == "prefix_widening":
                        turns_prepended += 1
                    else:
                        turns_inserted += 1

            suffix = prepared_turns[alignment.incoming_start + alignment.overlap_len:]
            if suffix:
                left_idx = alignment.existing_start + alignment.overlap_len - 1
                left_key = existing[left_idx].sort_key if left_idx >= 0 else None
                next_existing_idx = alignment.existing_start + alignment.overlap_len
                right_key = existing[next_existing_idx].sort_key if next_existing_idx < len(existing) else None
                for row, key in zip(suffix, self._allocate_sort_keys(left_key, right_key, len(suffix))):
                    row.canonical_turn_id = generate_canonical_turn_id()
                    row.sort_key = key
                    row.source_batch_id = batch_id
                    row.last_seen_at = now
                    self._write_turn(row, turn_number=-1)
                    rows_touched.append(row)
                    turns_written += 1
                    if merge_mode == "tail_append":
                        turns_appended += 1
                    else:
                        turns_inserted += 1

        batch = self._save_batch(
            conversation_id,
            raw_turn_count=raw_turn_count,
            merge_mode=merge_mode,
            first_turn_hash=prepared_turns[0].turn_hash,
            last_turn_hash=prepared_turns[-1].turn_hash,
            turns_matched=turns_matched,
            turns_appended=turns_appended,
            turns_prepended=turns_prepended,
            turns_inserted=turns_inserted,
            batch_id=batch_id,
        )
        recompute_groups = getattr(self._store, "recompute_canonical_turn_groups", None)
        if callable(recompute_groups):
            recompute_groups(conversation_id)
        self._refresh_persisted_anchors(conversation_id)
        return CanonicalIngestResult(
            merge_mode=merge_mode,
            turns_written=turns_written,
            turns_matched=turns_matched,
            turns_appended=turns_appended,
            turns_prepended=turns_prepended,
            turns_inserted=turns_inserted,
            batch=batch,
            rows=rows_touched,
        )

    @staticmethod
    def _preserve_existing_enrichment(
        row: CanonicalTurnRow, existing_row: CanonicalTurnRow,
    ) -> None:
        # Re-ingest of an already-stored turn constructs a new prepared row
        # from the incoming payload and overwrites the stored row. When the
        # incoming payload lacks metadata that the stored row already has —
        # for example a resync batch that doesn't carry the session_date the
        # original ingestion extracted from a "[Session from ...]" header —
        # the naive overwrite clobbers good data with empty. Merge alignment
        # is content-hash only (see _find_alignment), so preserving the
        # existing enrichment here does not affect row identity or ordering.
        if not (row.session_date or "").strip() and (existing_row.session_date or "").strip():
            row.session_date = existing_row.session_date

    def _prepare_message_row(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        raw_content: str | list[dict] | None = None,
        primary_tag: str = "_general",
        tags: list[str] | None = None,
        session_date: str = "",
        sender: str = "",
        fact_signals: list[FactSignal] | None = None,
        code_refs: list[dict] | None = None,
        entry: TurnTagEntry | None = None,
    ) -> CanonicalTurnRow:
        if role == "assistant":
            user_content = ""
            assistant_content = content
        else:
            user_content = content
            assistant_content = ""
        turn_hash, norm_user, norm_asst = compute_turn_hash_from_raw(user_content, assistant_content, version=HASH_VERSION)
        if isinstance(raw_content, list):
            import json

            raw_payload = json.dumps(raw_content)
        else:
            raw_payload = raw_content
        now = utcnow_iso()
        if entry is not None:
            primary_tag = entry.primary_tag or primary_tag
            tags = list(entry.tags or [])
            session_date = entry.session_date or session_date
            sender = entry.sender or sender
            fact_signals = list(entry.fact_signals or [])
            code_refs = list(entry.code_refs or [])
        tagged_at = now if (
            entry is not None
            or bool(tags)
            or bool(fact_signals)
            or bool(code_refs)
            or bool(session_date)
            or bool(sender)
            or (primary_tag not in ("", "_general"))
        ) else None
        return CanonicalTurnRow(
            conversation_id=conversation_id,
            turn_number=-1,
            turn_group_number=-1,
            sort_key=0.0,
            turn_hash=turn_hash,
            hash_version=HASH_VERSION,
            normalized_user_text=norm_user,
            normalized_assistant_text=norm_asst,
            user_content=user_content,
            assistant_content=assistant_content,
            user_raw_content=raw_payload if role == "user" else None,
            assistant_raw_content=raw_payload if role == "assistant" else None,
            primary_tag=primary_tag or "_general",
            tags=list(tags or []),
            session_date=session_date or "",
            sender=sender or "",
            fact_signals=list(fact_signals or []),
            code_refs=list(code_refs or []),
            tagged_at=tagged_at,
            first_seen_at=now,
            last_seen_at=now,
            created_at=now,
            updated_at=now,
        )

    def _find_alignment(
        self,
        conversation_id: str,
        existing: list[CanonicalTurnRow],
        incoming: list[CanonicalTurnRow],
    ) -> _Alignment | None:
        if not existing or not incoming:
            return None
        existing_hashes = [row.turn_hash for row in existing]
        incoming_hashes = [row.turn_hash for row in incoming]

        if existing_hashes == incoming_hashes:
            return _Alignment(0, 0, len(existing), 0, "exact_resend")

        if len(existing_hashes) <= len(incoming_hashes) and incoming_hashes[:len(existing_hashes)] == existing_hashes:
            return _Alignment(0, 0, len(existing_hashes), 0, "tail_append")

        if len(existing_hashes) <= len(incoming_hashes) and incoming_hashes[-len(existing_hashes):] == existing_hashes:
            return _Alignment(0, len(incoming_hashes) - len(existing_hashes), len(existing_hashes), 0, "prefix_widening")

        best: _Alignment | None = None
        for window_size in (5, 4, 3):
            existing_index = self._load_existing_anchor_index(
                conversation_id,
                existing,
                window_size,
            )
            if not existing_index:
                continue
            incoming_index = build_anchor_index(incoming, window_size)
            for digest, incoming_positions in incoming_index.items():
                existing_positions = existing_index.get(digest, [])
                if not existing_positions:
                    continue
                for incoming_start in incoming_positions:
                    for existing_start in existing_positions:
                        left = 0
                        while (
                            incoming_start - left - 1 >= 0
                            and existing_start - left - 1 >= 0
                            and incoming_hashes[incoming_start - left - 1] == existing_hashes[existing_start - left - 1]
                        ):
                            left += 1
                        right = window_size
                        while (
                            incoming_start + right < len(incoming_hashes)
                            and existing_start + right < len(existing_hashes)
                            and incoming_hashes[incoming_start + right] == existing_hashes[existing_start + right]
                        ):
                            right += 1
                        overlap_len = left + right
                        normalized_incoming_start = incoming_start - left
                        normalized_existing_start = existing_start - left
                        mode = "interior_overlap"
                        if normalized_incoming_start == 0 and normalized_existing_start == 0:
                            mode = "tail_append" if len(incoming_hashes) > overlap_len else "exact_resend"
                        elif normalized_existing_start == 0 and normalized_incoming_start > 0 and normalized_incoming_start + overlap_len == len(incoming_hashes):
                            mode = "prefix_widening"
                        candidate = _Alignment(
                            existing_start=normalized_existing_start,
                            incoming_start=normalized_incoming_start,
                            overlap_len=overlap_len,
                            window_size=window_size,
                            merge_mode=mode,
                        )
                        if best is None or candidate.overlap_len > best.overlap_len or (
                            candidate.overlap_len == best.overlap_len and candidate.window_size > best.window_size
                        ):
                            best = candidate
            if best is not None:
                return best
        return None

    def _write_turn(
        self,
        row: CanonicalTurnRow,
        *,
        turn_number: int,
        first_seen_at: str | None = None,
        last_seen_at: str | None = None,
    ) -> None:
        self._store.save_canonical_turn(
            row.conversation_id,
            turn_number,
            row.user_content,
            row.assistant_content,
            user_raw_content=row.user_raw_content,
            assistant_raw_content=row.assistant_raw_content,
            primary_tag=row.primary_tag,
            tags=list(row.tags or []),
            session_date=row.session_date,
            sender=row.sender,
            fact_signals=list(row.fact_signals or []),
            code_refs=list(row.code_refs or []),
            created_at=row.created_at or first_seen_at or utcnow_iso(),
            updated_at=utcnow_iso(),
            canonical_turn_id=row.canonical_turn_id or None,
            sort_key=row.sort_key,
            turn_hash=row.turn_hash,
            hash_version=row.hash_version or HASH_VERSION,
            normalized_user_text=row.normalized_user_text,
            normalized_assistant_text=row.normalized_assistant_text,
            tagged_at=row.tagged_at,
            compacted_at=row.compacted_at,
            first_seen_at=first_seen_at or row.first_seen_at,
            last_seen_at=last_seen_at or row.last_seen_at,
            source_batch_id=row.source_batch_id or None,
            turn_group_number=row.turn_group_number,
        )
        resolved_turn_number = turn_number
        if turn_number < 0 and row.canonical_turn_id:
            lookup = getattr(self._store, "_lookup_ordinal_for_canonical_turn_id", None)
            if callable(lookup):
                resolved_turn_number = int(lookup(row.conversation_id, row.canonical_turn_id))
        if resolved_turn_number >= 0:
            try:
                self._semantic.embed_and_store_turn(
                    row.conversation_id,
                    resolved_turn_number,
                    canonical_turn_id=row.canonical_turn_id or None,
                    user_text=row.user_content,
                    assistant_text=row.assistant_content,
                    user_raw_content=row.user_raw_content,
                    assistant_raw_content=row.assistant_raw_content,
                )
            except Exception:
                logger.warning(
                    "CANONICAL_TURN_EMBED_FAILED side=both conv=%s turn=%d",
                    row.conversation_id[:12],
                    resolved_turn_number,
                    exc_info=True,
                )

    def _conversation_merge_lock(self, conversation_id: str):
        locker = getattr(self._store, "conversation_reconcile", None)
        if callable(locker):
            return locker(conversation_id)
        return nullcontext()

    def _save_batch(
        self,
        conversation_id: str,
        *,
        raw_turn_count: int,
        merge_mode: str,
        first_turn_hash: str,
        last_turn_hash: str,
        turns_matched: int,
        turns_appended: int,
        turns_prepended: int,
        turns_inserted: int,
        batch_id: str | None = None,
    ) -> IngestBatchRecord:
        batch_payload = {
            "batch_id": batch_id or generate_canonical_turn_id(),
            "conversation_id": conversation_id,
            "received_at": utcnow_iso(),
            "raw_turn_count": raw_turn_count,
            "merge_mode": merge_mode,
            "turns_matched": turns_matched,
            "turns_appended": turns_appended,
            "turns_prepended": turns_prepended,
            "turns_inserted": turns_inserted,
            "first_turn_hash": first_turn_hash,
            "last_turn_hash": last_turn_hash,
        }
        batch_id = self._store.save_ingest_batch(batch_payload)
        return IngestBatchRecord(
            batch_id=batch_id,
            conversation_id=conversation_id,
            received_at=batch_payload["received_at"],
            raw_turn_count=raw_turn_count,
            merge_mode=merge_mode,
            turns_matched=turns_matched,
            turns_appended=turns_appended,
            turns_prepended=turns_prepended,
            turns_inserted=turns_inserted,
            first_turn_hash=first_turn_hash,
            last_turn_hash=last_turn_hash,
        )

    def _allocate_sort_keys(
        self,
        left_key: float | None,
        right_key: float | None,
        count: int,
    ) -> list[float]:
        if count <= 0:
            return []
        if left_key is None and right_key is None:
            return [float((idx + 1) * 1000.0) for idx in range(count)]
        if left_key is None:
            start = float(right_key - (1000.0 * count))
            return [start + (1000.0 * idx) for idx in range(count)]
        if right_key is None:
            return [float(left_key + (1000.0 * (idx + 1))) for idx in range(count)]
        step = (right_key - left_key) / float(count + 1)
        if step <= 0.001:
            step = 0.001
        return [float(left_key + (step * (idx + 1))) for idx in range(count)]

    def _seen_recently(self, last_seen_at: str) -> bool:
        try:
            seen_at = datetime.fromisoformat(str(last_seen_at).replace("Z", "+00:00"))
        except Exception:
            if last_seen_at:
                logger.warning(
                    "CANONICAL_TURN_DEDUP_TIMESTAMP_INVALID: value=%r",
                    last_seen_at,
                )
            return False
        return datetime.now(timezone.utc) - seen_at <= timedelta(minutes=10)

    def _ordinal_for_row(self, rows: list[CanonicalTurnRow], canonical_turn_id: str) -> int:
        for idx, row in enumerate(rows):
            if row.canonical_turn_id == canonical_turn_id:
                return idx
        return -1

    def _load_existing_anchor_index(
        self,
        conversation_id: str,
        existing: list[CanonicalTurnRow],
        window_size: int,
    ) -> dict[str, list[int]]:
        loader = getattr(self._store, "get_canonical_turn_anchor_positions", None)
        if callable(loader):
            anchors = loader(conversation_id, window_size)
            if anchors:
                return anchors
        return build_anchor_index(existing, window_size)

    def _refresh_persisted_anchors(self, conversation_id: str) -> None:
        saver = getattr(self._store, "replace_canonical_turn_anchors", None)
        if not callable(saver):
            return
        rows = self._store.get_all_canonical_turns(conversation_id)
        anchors: list[tuple[int, str, str]] = []
        for window_size in (3, 4, 5):
            if len(rows) < window_size:
                continue
            for start in range(0, len(rows) - window_size + 1):
                start_turn_id = rows[start].canonical_turn_id
                if not start_turn_id:
                    continue
                anchors.append(
                    (
                        window_size,
                        compute_anchor_hash(rows, start, window_size),
                        start_turn_id,
                    )
                )
        saver(conversation_id, anchors)
