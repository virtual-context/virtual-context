"""Maintenance helpers for canonical full_text rows and full_text_chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import load_config
from ..types import ChunkEmbedding, FullTextChunkEmbedding, Message, StoredSegment, VirtualContextConfig
from .segmenter import pair_messages_into_turns, split_session_boundary_messages
from .semantic_search import SemanticSearchManager

_SESSION_HEADER_RE = re.compile(r"\[Session from ([^\]]+)\]")


def _extract_explicit_session_date(*contents: str | None) -> str:
    """Return the first explicit [Session from ...] header found in text."""
    for content in contents:
        if not isinstance(content, str) or not content:
            continue
        match = _SESSION_HEADER_RE.search(content)
        if match:
            return match.group(1).strip()
    return ""


def _segment_sort_key(segment: StoredSegment) -> tuple:
    return (
        segment.created_at,
        segment.ref,
    )


def _explicit_segment_range(segment: StoredSegment) -> tuple[int, int] | None:
    meta = getattr(segment, "metadata", None)
    start_raw = getattr(meta, "start_turn_number", -1)
    end_raw = getattr(meta, "end_turn_number", -1)
    start = -1 if start_raw is None else int(start_raw)
    end = -1 if end_raw is None else int(end_raw)
    if start < 0 or end < start:
        return None
    return (start, end)


def _segment_turn_count(segment: StoredSegment) -> int:
    meta = getattr(segment, "metadata", None)
    raw = getattr(meta, "turn_count", 0)
    try:
        return max(0, int(raw or 0))
    except Exception:
        return 0


def resolve_segment_turn_ranges(
    segments: list[StoredSegment],
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Resolve turn ranges using explicit metadata or chronological turn_count fallback."""
    resolved: dict[str, tuple[int, int]] = {}
    inferred_segments: list[str] = []
    cursor = 0

    for segment in sorted(segments, key=_segment_sort_key):
        explicit = _explicit_segment_range(segment)
        if explicit is not None:
            start, end = explicit
            resolved[segment.ref] = (start, end)
            cursor = max(cursor, end + 1)
            continue

        turn_count = _segment_turn_count(segment)
        if turn_count <= 0:
            continue

        start = cursor
        end = start + turn_count - 1
        resolved[segment.ref] = (start, end)
        inferred_segments.append(segment.ref)
        cursor = end + 1

    return resolved, inferred_segments


def build_authoritative_turn_segment_map(
    segments: list[StoredSegment],
) -> dict[int, StoredSegment]:
    """Choose the newest valid segment covering each turn."""
    resolved_ranges, _ = resolve_segment_turn_ranges(segments)
    selected: dict[int, StoredSegment] = {}
    for segment in sorted(segments, key=_segment_sort_key, reverse=True):
        resolved = resolved_ranges.get(segment.ref)
        if resolved is None:
            continue
        start, end = resolved
        for turn_number in range(start, end + 1):
            selected.setdefault(turn_number, segment)
    return selected


def _group_chunks_by_segment(
    chunks: list[ChunkEmbedding],
) -> dict[str, list[ChunkEmbedding]]:
    grouped: dict[str, list[ChunkEmbedding]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.segment_ref, []).append(chunk)
    for segment_ref in grouped:
        grouped[segment_ref].sort(key=lambda item: item.chunk_index)
    return grouped


def _message_from_dict(raw: dict[str, Any]) -> Message:
    content = raw.get("content", "")
    if isinstance(content, list):
        text_parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = "\n".join(part for part in text_parts if part).strip() or json.dumps(content)
    elif content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)
    raw_content = raw.get("raw_content")
    if raw_content is not None and not isinstance(raw_content, list):
        raw_content = None
    return Message(
        role=str(raw.get("role", "")),
        content=content,
        raw_content=raw_content,
    )


def _pair_segment_messages(segment: StoredSegment) -> list[tuple[str, str]]:
    messages = [
        _message_from_dict(raw)
        for raw in (segment.messages or [])
        if isinstance(raw, dict)
    ]
    messages = split_session_boundary_messages(messages)
    pairs = pair_messages_into_turns(messages)
    reconstructed: list[tuple[str, str]] = []
    for pair in pairs:
        user_parts = [
            msg.content.strip()
            for msg in pair.messages
            if msg.role == "user" and (msg.content or "").strip()
        ]
        assistant_parts = [
            msg.content.strip()
            for msg in pair.messages
            if msg.role == "assistant" and (msg.content or "").strip()
        ]
        reconstructed.append(
            (
                "\n\n".join(user_parts).strip(),
                "\n\n".join(assistant_parts).strip(),
            )
        )
    return reconstructed


def _load_turn_tag_entries(store: Any, conversation_id: str) -> dict[int, Any]:
    load_engine_state = getattr(store, "load_engine_state", None)
    if not callable(load_engine_state):
        return {}
    try:
        snapshot = load_engine_state(conversation_id)
    except Exception:
        return {}
    if snapshot is None:
        return {}
    entries: dict[int, Any] = {}
    for entry in getattr(snapshot, "turn_tag_entries", []) or []:
        turn_number = getattr(entry, "turn_number", None)
        if isinstance(turn_number, int):
            entries[turn_number] = entry
    return entries


def backfill_full_text_rows(store: Any, conversation_id: str) -> dict[str, Any]:
    """Reconstruct canonical full_text rows from segment messages_json plus per-turn tag state."""
    segments = list(store.get_all_segments(conversation_id=conversation_id))
    turn_entries = _load_turn_tag_entries(store, conversation_id)
    resolved_ranges, inferred_segments = resolve_segment_turn_ranges(segments)
    deleted = int(store.delete_full_text_rows(conversation_id) or 0)
    turns_written = 0
    overlapping_turns_skipped = 0
    malformed_segments: list[str] = []
    invalid_range_segments: list[str] = []
    selected_turns: set[int] = set()

    for segment in sorted(segments, key=_segment_sort_key, reverse=True):
        resolved = resolved_ranges.get(segment.ref)
        if resolved is None:
            invalid_range_segments.append(segment.ref)
            continue
        start, end = resolved

        pairs = _pair_segment_messages(segment)
        expected_pairs = end - start + 1
        if len(pairs) != expected_pairs:
            malformed_segments.append(segment.ref)
            continue

        created_at = getattr(segment, "created_at", None)
        created_raw = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or "")
        for offset, (user_content, assistant_content) in enumerate(pairs):
            turn_number = start + offset
            if turn_number in selected_turns:
                overlapping_turns_skipped += 1
                continue
            store.save_full_text(
                conversation_id,
                turn_number,
                user_content,
                assistant_content,
                primary_tag=getattr(turn_entries.get(turn_number), "primary_tag", "_general"),
                tags=list(getattr(turn_entries.get(turn_number), "tags", []) or []),
                session_date=getattr(turn_entries.get(turn_number), "session_date", ""),
                sender=getattr(turn_entries.get(turn_number), "sender", ""),
                fact_signals=list(getattr(turn_entries.get(turn_number), "fact_signals", []) or []),
                code_refs=list(getattr(turn_entries.get(turn_number), "code_refs", []) or []),
                created_at=created_raw,
                updated_at=created_raw,
            )
            selected_turns.add(turn_number)
            turns_written += 1

    return {
        "deleted_rows": deleted,
        "segments_considered": len(segments),
        "turns_written": turns_written,
        "unique_turns_written": len(selected_turns),
        "overlapping_turns_skipped": overlapping_turns_skipped,
        "inferred_range_segments": inferred_segments,
        "invalid_range_segments": invalid_range_segments,
        "malformed_segments": malformed_segments,
    }


def repair_full_text_session_dates(
    store: Any,
    conversation_id: str,
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Repair canonical full_text.session_date from explicit session headers.

    This is a cheap local reconciliation pass for rows whose stored
    `session_date` drifted away from the authoritative inline header already
    present in the archived turn text.
    """
    rows = list(store.get_all_full_text_rows(conversation_id))
    repaired: list[dict[str, Any]] = []
    already_correct = 0
    missing_header = 0

    for row in rows:
        explicit_session_date = _extract_explicit_session_date(
            row.user_content,
            row.assistant_content,
        )
        if not explicit_session_date:
            missing_header += 1
            continue
        if explicit_session_date == (row.session_date or "").strip():
            already_correct += 1
            continue

        if apply:
            store.save_full_text(
                conversation_id=row.conversation_id,
                turn_number=row.turn_number,
                user_content=row.user_content,
                assistant_content=row.assistant_content,
                user_raw_content=row.user_raw_content,
                assistant_raw_content=row.assistant_raw_content,
                primary_tag=row.primary_tag,
                tags=list(row.tags),
                session_date=explicit_session_date,
                sender=row.sender,
                fact_signals=list(row.fact_signals),
                code_refs=list(row.code_refs),
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        repaired.append(
            {
                "turn_number": row.turn_number,
                "from": row.session_date,
                "to": explicit_session_date,
            }
        )

    return {
        "rows_scanned": len(rows),
        "rows_repaired": len(repaired),
        "rows_already_correct": already_correct,
        "rows_without_explicit_header": missing_header,
        "applied": apply,
        "repairs": repaired,
    }


def _load_engine_snapshot(store: Any, conversation_id: str) -> Any | None:
    load_engine_state = getattr(store, "load_engine_state", None)
    if not callable(load_engine_state):
        return None
    try:
        return load_engine_state(conversation_id)
    except Exception:
        return None


def _iter_current_turn_pairs(
    store: Any,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Return current archived turn pairs across full_text and protected tail."""
    rows = list(store.get_all_full_text_rows(conversation_id))
    current: list[dict[str, Any]] = []
    seen_turns: set[int] = set()
    for row in rows:
        current.append(
            {
                "turn_number": row.turn_number,
                "user_content": row.user_content,
                "assistant_content": row.assistant_content,
                "user_raw_content": row.user_raw_content,
                "assistant_raw_content": row.assistant_raw_content,
                "source_scope": "full_text",
            }
        )
        seen_turns.add(row.turn_number)

    snapshot = _load_engine_snapshot(store, conversation_id)
    turn_count = int(getattr(snapshot, "turn_count", len(rows)) or len(rows))
    tail_turns = [turn_number for turn_number in range(turn_count) if turn_number not in seen_turns]
    if tail_turns:
        turn_map = store.get_turn_messages(conversation_id, tail_turns)
        for turn_number in sorted(turn_map):
            user_content, assistant_content, user_raw_content, assistant_raw_content = turn_map[turn_number]
            current.append(
                {
                    "turn_number": turn_number,
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                    "user_raw_content": user_raw_content,
                    "assistant_raw_content": assistant_raw_content,
                    "source_scope": "turn_messages",
                }
            )

    current.sort(key=lambda item: int(item["turn_number"]))
    return current


def build_authoritative_session_dates_from_source_pairs(
    store: Any,
    conversation_id: str,
    *,
    source_turn_pairs: list[tuple[str, str]],
    source_session_dates: list[str],
) -> dict[str, Any]:
    """Map current archived turns to authoritative session dates by exact text identity.

    The current archive can drift away from source turn numbering. Instead of
    assuming row N corresponds to source turn N, this function matches the
    exact `(user_content, assistant_content)` pair against the source archive
    and uses the matched source row's session date.
    """
    if len(source_turn_pairs) != len(source_session_dates):
        raise ValueError("source_turn_pairs and source_session_dates must have equal length")

    pair_to_indexes: dict[tuple[str, str], list[int]] = {}
    for idx, pair in enumerate(source_turn_pairs):
        pair_to_indexes.setdefault(pair, []).append(idx)

    turn_session_dates: dict[int, str] = {}
    turn_source_indexes: dict[int, int | None] = {}
    exact_unique_matches = 0
    same_date_multi_matches = 0
    ambiguous_matches: list[dict[str, Any]] = []
    unmatched_turns: list[dict[str, Any]] = []

    for row in _iter_current_turn_pairs(store, conversation_id):
        pair = (row["user_content"], row["assistant_content"])
        matches = pair_to_indexes.get(pair, [])
        if not matches:
            unmatched_turns.append(
                {
                    "turn_number": row["turn_number"],
                    "source_scope": row["source_scope"],
                }
            )
            continue

        matched_dates = sorted({source_session_dates[idx] for idx in matches})
        if len(matches) == 1:
            idx = matches[0]
            turn_session_dates[row["turn_number"]] = source_session_dates[idx]
            turn_source_indexes[row["turn_number"]] = idx
            exact_unique_matches += 1
            continue

        if len(matched_dates) == 1:
            turn_session_dates[row["turn_number"]] = matched_dates[0]
            turn_source_indexes[row["turn_number"]] = None
            same_date_multi_matches += 1
            continue

        ambiguous_matches.append(
            {
                "turn_number": row["turn_number"],
                "source_scope": row["source_scope"],
                "source_indexes": matches,
                "source_session_dates": matched_dates,
            }
        )

    return {
        "conversation_id": conversation_id,
        "turns_scanned": len(_iter_current_turn_pairs(store, conversation_id)),
        "turn_session_dates": turn_session_dates,
        "turn_source_indexes": turn_source_indexes,
        "exact_unique_matches": exact_unique_matches,
        "same_date_multi_matches": same_date_multi_matches,
        "ambiguous_match_count": len(ambiguous_matches),
        "unmatched_count": len(unmatched_turns),
        "ambiguous_matches": ambiguous_matches,
        "unmatched_turns": unmatched_turns,
    }


def repair_session_date_lineage(
    store: Any,
    conversation_id: str,
    *,
    turn_session_dates: dict[int, str],
    apply: bool = True,
) -> dict[str, Any]:
    """Propagate authoritative session dates through all local date-bearing layers."""
    engine_repairs: list[dict[str, Any]] = []
    full_text_repairs: list[dict[str, Any]] = []
    segment_repairs: list[dict[str, Any]] = []
    fact_repairs: list[dict[str, Any]] = []

    snapshot = _load_engine_snapshot(store, conversation_id)
    if snapshot is not None:
        snapshot_changed = False
        for entry in getattr(snapshot, "turn_tag_entries", []) or []:
            expected = turn_session_dates.get(getattr(entry, "turn_number", -1))
            if not expected or expected == getattr(entry, "session_date", ""):
                continue
            engine_repairs.append(
                {
                    "turn_number": entry.turn_number,
                    "from": getattr(entry, "session_date", ""),
                    "to": expected,
                }
            )
            entry.session_date = expected
            snapshot_changed = True
        if apply and snapshot_changed:
            store.save_engine_state(snapshot)

    full_rows = list(store.get_all_full_text_rows(conversation_id))
    for row in full_rows:
        expected = turn_session_dates.get(row.turn_number)
        if not expected or expected == (row.session_date or ""):
            continue
        full_text_repairs.append(
            {
                "turn_number": row.turn_number,
                "from": row.session_date,
                "to": expected,
            }
        )
        if apply:
            store.save_full_text(
                conversation_id=row.conversation_id,
                turn_number=row.turn_number,
                user_content=row.user_content,
                assistant_content=row.assistant_content,
                user_raw_content=row.user_raw_content,
                assistant_raw_content=row.assistant_raw_content,
                primary_tag=row.primary_tag,
                tags=list(row.tags),
                session_date=expected,
                sender=row.sender,
                fact_signals=list(row.fact_signals),
                code_refs=list(row.code_refs),
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    segments = list(store.get_all_segments(conversation_id=conversation_id))
    segment_ranges, _ = resolve_segment_turn_ranges(segments)
    segment_expected_dates: dict[str, str] = {}
    for segment in segments:
        resolved = segment_ranges.get(segment.ref)
        if resolved is None:
            continue
        start_turn, end_turn = resolved
        expected = ""
        for turn_number in range(start_turn, end_turn + 1):
            expected = turn_session_dates.get(turn_number, "")
            if expected:
                break
        if not expected:
            continue
        segment_expected_dates[segment.ref] = expected
        actual = getattr(segment.metadata, "session_date", "") if getattr(segment, "metadata", None) else ""
        if actual == expected:
            continue
        segment_repairs.append(
            {
                "segment_ref": segment.ref,
                "primary_tag": segment.primary_tag,
                "from": actual,
                "to": expected,
                "turn_range": [start_turn, end_turn],
            }
        )
        if apply:
            segment.metadata.session_date = expected
            update_segment = getattr(store, "update_segment", None)
            if callable(update_segment):
                update_segment(segment)
            else:
                store.store_segment(segment)

    for segment in segments:
        expected = segment_expected_dates.get(segment.ref)
        if not expected:
            continue
        facts = list(store.get_facts_by_segment(segment.ref))
        changed = 0
        for fact in facts:
            if getattr(fact, "session_date", "") == expected:
                continue
            fact.session_date = expected
            changed += 1
        if changed:
            fact_repairs.append(
                {
                    "segment_ref": segment.ref,
                    "fact_count": changed,
                    "to": expected,
                }
            )
            if apply:
                store.replace_facts_for_segment(conversation_id, segment.ref, facts)

    return {
        "conversation_id": conversation_id,
        "applied": apply,
        "engine_state_repairs": engine_repairs,
        "engine_state_repair_count": len(engine_repairs),
        "full_text_repairs": full_text_repairs,
        "full_text_repair_count": len(full_text_repairs),
        "segment_repairs": segment_repairs,
        "segment_repair_count": len(segment_repairs),
        "fact_repairs": fact_repairs,
        "fact_repair_count": sum(item["fact_count"] for item in fact_repairs),
    }


def bootstrap_full_text_chunks(store: Any, conversation_id: str) -> dict[str, int]:
    """Populate full_text_chunks by copying existing segment chunk rows."""
    segments = list(store.get_all_segments(conversation_id=conversation_id))
    authoritative = build_authoritative_turn_segment_map(segments)
    segment_refs = {segment.ref for segment in authoritative.values()}
    chunk_map = _group_chunks_by_segment(
        [
            chunk
            for chunk in store.get_all_chunk_embeddings()
            if chunk.segment_ref in segment_refs
        ]
    )

    store.delete_full_text_chunk_embeddings(conversation_id)

    turns_written = 0
    rows_written = 0
    missing_chunk_turns = 0
    used_segment_refs: set[str] = set()
    for turn_number, segment in sorted(authoritative.items()):
        segment_chunks = chunk_map.get(segment.ref, [])
        if not segment_chunks:
            missing_chunk_turns += 1
            continue
        store.store_full_text_chunk_embeddings(
            conversation_id,
            turn_number,
            "combined",
            [
                FullTextChunkEmbedding(
                    conversation_id=conversation_id,
                    turn_number=turn_number,
                    side="combined",
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding=chunk.embedding,
                )
                for chunk in segment_chunks
            ],
        )
        turns_written += 1
        rows_written += len(segment_chunks)
        used_segment_refs.add(segment.ref)

    return {
        "segments_considered": len(segments),
        "turns_selected": len(authoritative),
        "turns_written": turns_written,
        "rows_written": rows_written,
        "segments_used": len(used_segment_refs),
        "turns_missing_segment_chunks": missing_chunk_turns,
    }


def rebuild_full_text_chunks(
    store: Any,
    semantic: SemanticSearchManager,
    conversation_id: str,
) -> dict[str, int]:
    """Overwrite a conversation's full_text_chunks with true per-side rows."""
    rows = list(store.get_all_full_text_rows(conversation_id))
    deleted = int(store.delete_full_text_chunk_embeddings(conversation_id) or 0)
    turns_embedded = 0
    for row in rows:
        semantic.embed_and_store_turn(
            conversation_id,
            row.turn_number,
            user_text=row.user_content,
            assistant_text=row.assistant_content,
            user_raw_content=row.user_raw_content,
            assistant_raw_content=row.assistant_raw_content,
        )
        turns_embedded += 1
    rows_written = len(store.get_all_full_text_chunk_embeddings(conversation_id=conversation_id))
    return {
        "deleted_rows": deleted,
        "turns_expected": len(rows),
        "turns_embedded": turns_embedded,
        "rows_written": rows_written,
    }


def open_store_for_config(config: VirtualContextConfig) -> Any:
    """Open the same segment/search store stack the engine uses."""
    from .composite_store import CompositeStore
    from ..storage.noop_fact_link_store import NoopFactLinkStore

    if config.storage.backend == "sqlite":
        from ..storage.sqlite import SQLiteStore

        sqlite = SQLiteStore(db_path=config.storage.sqlite_path)
        fact_links = sqlite if config.facts.graph_links else NoopFactLinkStore()
        return CompositeStore(
            segments=sqlite,
            facts=sqlite,
            fact_links=fact_links,
            state=sqlite,
            search=sqlite,
        )

    if config.storage.backend == "postgres":
        from ..storage.postgres import PostgresStore

        pg = PostgresStore(dsn=config.storage.postgres_dsn)
        fact_links = pg if config.facts.graph_links else NoopFactLinkStore()
        return CompositeStore(
            segments=pg,
            facts=pg,
            fact_links=fact_links,
            state=pg,
            search=pg,
        )

    raise ValueError(f"Unsupported backend for full_text backfill: {config.storage.backend}")


def load_store_and_semantic(config_path: str | Path) -> tuple[Any, SemanticSearchManager, VirtualContextConfig]:
    config = load_config(str(config_path))
    store = open_store_for_config(config)
    semantic = SemanticSearchManager(store, config)
    return store, semantic, config
