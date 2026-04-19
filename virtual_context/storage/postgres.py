"""PostgresStore: storage backend using psycopg (PostgreSQL)."""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row

from ..core.store import ContextStore
from ..core.canonical_turns import (
    HASH_VERSION,
    compute_turn_hash_from_raw,
    generate_canonical_turn_id,
    utcnow_iso,
)
from ..core.progress_snapshot import (
    ActiveCompactionSnapshot,
    ActiveEpisodeSnapshot,
    ProgressSnapshot,
)
from ..types import (
    ChunkEmbedding,
    ConversationStats,
    DepthLevel,
    EngineStateSnapshot,
    Fact,
    FactLink,
    FactSignal,
    CanonicalTurnChunkEmbedding,
    CanonicalTurnRow,
    LinkedFact,
    QuoteResult,
    SegmentMetadata,
    StoredSegment,
    StoredSummary,
    TagStats,
    TagSummary,
    TemporalStatus,
    TurnTagEntry,
    WorkingSetEntry,
)
from .helpers import dt_to_str as _dt_to_str, str_to_dt as _str_to_dt, extract_excerpt as _extract_excerpt

logger = logging.getLogger(__name__)


def _parse_sequence_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS segments (
    ref TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL DEFAULT '',
    primary_tag TEXT NOT NULL DEFAULT '_general',
    summary TEXT NOT NULL DEFAULT '',
    full_text TEXT NOT NULL DEFAULT '',
    messages_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    summary_tokens INTEGER NOT NULL DEFAULT 0,
    full_tokens INTEGER NOT NULL DEFAULT 0,
    compression_ratio REAL NOT NULL DEFAULT 0.0,
    compaction_model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    start_timestamp TEXT NOT NULL,
    end_timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_tags (
    segment_ref TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (segment_ref, tag),
    FOREIGN KEY (segment_ref) REFERENCES segments(ref) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tag_aliases (
    alias TEXT NOT NULL,
    conversation_id TEXT NOT NULL DEFAULT '',
    canonical TEXT NOT NULL,
    PRIMARY KEY (alias, conversation_id)
);

CREATE TABLE IF NOT EXISTS tag_summaries (
    tag TEXT NOT NULL,
    conversation_id TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    code_refs TEXT NOT NULL DEFAULT '[]',
    summary_tokens INTEGER NOT NULL DEFAULT 0,
    source_segment_refs TEXT NOT NULL DEFAULT '[]',
    source_turn_numbers TEXT NOT NULL DEFAULT '[]',
    source_canonical_turn_ids TEXT NOT NULL DEFAULT '[]',
    covers_through_turn INTEGER NOT NULL DEFAULT -1,
    covers_through_canonical_turn_id TEXT NOT NULL DEFAULT '',
    generated_by_turn_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tag, conversation_id)
);

CREATE TABLE IF NOT EXISTS engine_state (
    conversation_id TEXT PRIMARY KEY,
    compacted_prefix_messages INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    turn_tag_entries TEXT NOT NULL,
    saved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_lifecycle (
    conversation_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL DEFAULT 0,
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_turns (
    canonical_turn_id UUID PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_group_number INTEGER NOT NULL DEFAULT -1,
    sort_key DOUBLE PRECISION NOT NULL,
    turn_hash TEXT NOT NULL,
    hash_version SMALLINT NOT NULL DEFAULT 1,
    normalized_user_text TEXT NOT NULL DEFAULT '',
    normalized_assistant_text TEXT NOT NULL DEFAULT '',
    user_content TEXT NOT NULL DEFAULT '',
    assistant_content TEXT NOT NULL DEFAULT '',
    user_raw_content TEXT,
    assistant_raw_content TEXT,
    primary_tag TEXT NOT NULL DEFAULT '_general',
    tags_json TEXT NOT NULL DEFAULT '[]',
    session_date TEXT NOT NULL DEFAULT '',
    sender TEXT NOT NULL DEFAULT '',
    fact_signals_json TEXT NOT NULL DEFAULT '[]',
    code_refs_json TEXT NOT NULL DEFAULT '[]',
    tagged_at TEXT,
    compacted_at TEXT,
    first_seen_at TEXT,
    last_seen_at TEXT,
    source_batch_id UUID,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    UNIQUE (conversation_id, sort_key)
);

ALTER TABLE canonical_turns
ADD COLUMN IF NOT EXISTS turn_group_number INTEGER NOT NULL DEFAULT -1;

CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_order
ON canonical_turns (conversation_id, sort_key);

CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_hash
ON canonical_turns (conversation_id, turn_hash);

CREATE INDEX IF NOT EXISTS idx_canonical_turns_compaction_queue
ON canonical_turns (conversation_id, sort_key)
WHERE compacted_at IS NULL;

CREATE TABLE IF NOT EXISTS canonical_turn_anchors (
    conversation_id TEXT NOT NULL,
    anchor_hash TEXT NOT NULL,
    start_turn_id UUID NOT NULL,
    window_size INTEGER NOT NULL DEFAULT 3
);

CREATE INDEX IF NOT EXISTS idx_canonical_turn_anchors_lookup
ON canonical_turn_anchors (conversation_id, window_size, anchor_hash);

CREATE TABLE IF NOT EXISTS ingest_batches (
    batch_id UUID PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    received_at TEXT NOT NULL DEFAULT '',
    raw_turn_count INTEGER NOT NULL DEFAULT 0,
    merge_mode TEXT NOT NULL DEFAULT '',
    turns_matched INTEGER NOT NULL DEFAULT 0,
    turns_appended INTEGER NOT NULL DEFAULT 0,
    turns_prepended INTEGER NOT NULL DEFAULT 0,
    turns_inserted INTEGER NOT NULL DEFAULT 0,
    first_turn_hash TEXT NOT NULL DEFAULT '',
    last_turn_hash TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS conversation_aliases (
    alias_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_chunks (
    segment_ref TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (segment_ref, chunk_index)
);

CREATE TABLE IF NOT EXISTS canonical_turn_chunks (
    conversation_id TEXT NOT NULL,
    canonical_turn_id UUID NOT NULL,
    side TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, canonical_turn_id, side, chunk_index)
);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL DEFAULT '',
    verb TEXT NOT NULL DEFAULT '',
    object TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    what TEXT NOT NULL DEFAULT '',
    who TEXT NOT NULL DEFAULT '',
    when_date TEXT NOT NULL DEFAULT '',
    "where" TEXT NOT NULL DEFAULT '',
    why TEXT NOT NULL DEFAULT '',
    fact_type TEXT NOT NULL DEFAULT 'personal',
    tags_json TEXT NOT NULL DEFAULT '[]',
    segment_ref TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    turn_numbers_json TEXT NOT NULL DEFAULT '[]',
    mentioned_at TEXT NOT NULL DEFAULT '',
    session_date TEXT NOT NULL DEFAULT '',
    superseded_by TEXT
);

CREATE TABLE IF NOT EXISTS fact_tags (
    fact_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (fact_id, tag),
    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fact_links (
    id TEXT PRIMARY KEY,
    source_fact_id TEXT NOT NULL,
    target_fact_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    context TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT 'compaction',
    FOREIGN KEY (source_fact_id) REFERENCES facts(id) ON DELETE CASCADE,
    FOREIGN KEY (target_fact_id) REFERENCES facts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_outputs (
    ref TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    command TEXT NOT NULL DEFAULT '',
    turn INTEGER NOT NULL,
    content TEXT NOT NULL,
    original_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS request_captures (
    conversation_id TEXT NOT NULL DEFAULT '',
    turn INTEGER NOT NULL,
    turn_id TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL,
    recorded_at DOUBLE PRECISION NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn, turn_id)
);

CREATE TABLE IF NOT EXISTS request_turn_counters (
    conversation_id TEXT PRIMARY KEY,
    next_request_turn INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    request_turn INTEGER NOT NULL,
    round INTEGER NOT NULL,
    group_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_input TEXT NOT NULL,
    tool_result TEXT NOT NULL,
    result_length INTEGER NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    found BOOLEAN,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conv ON tool_calls(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_group ON tool_calls(group_id);

CREATE TABLE IF NOT EXISTS request_context (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    request_turn INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    user_message TEXT NOT NULL,
    inbound_tags TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    candidates_found INTEGER NOT NULL,
    candidates_selected INTEGER NOT NULL,
    segments_injected TEXT NOT NULL,
    facts_injected TEXT NOT NULL,
    facts_count INTEGER NOT NULL,
    facts_tags TEXT NOT NULL,
    pool_used INTEGER NOT NULL,
    pool_budget INTEGER NOT NULL,
    total_context_tokens INTEGER NOT NULL,
    non_virtualizable_floor INTEGER NOT NULL,
    tool_call_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_request_context_conv ON request_context(conversation_id);

CREATE TABLE IF NOT EXISTS tag_summary_embeddings (
    tag TEXT NOT NULL,
    conversation_id TEXT NOT NULL DEFAULT '',
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (tag, conversation_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_segments_primary_tag ON segments(primary_tag);
CREATE INDEX IF NOT EXISTS idx_segments_created_at ON segments(created_at);
CREATE INDEX IF NOT EXISTS idx_segments_conversation_id ON segments(conversation_id);
CREATE INDEX IF NOT EXISTS idx_segment_tags_tag ON segment_tags(tag);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_facts_verb ON facts(verb);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_subject_verb ON facts(subject, verb);
CREATE INDEX IF NOT EXISTS idx_facts_segment_ref ON facts(segment_ref);
CREATE INDEX IF NOT EXISTS idx_facts_conversation_id ON facts(conversation_id);
CREATE INDEX IF NOT EXISTS idx_facts_fact_type ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_fact_tags_tag ON fact_tags(tag);
CREATE INDEX IF NOT EXISTS idx_fact_links_source ON fact_links(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_links_target ON fact_links(target_fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_links_type ON fact_links(relation_type);

CREATE TABLE IF NOT EXISTS turn_tool_outputs (
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    tool_output_ref TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn_number, tool_output_ref)
);

CREATE TABLE IF NOT EXISTS segment_tool_outputs (
    conversation_id TEXT NOT NULL,
    segment_ref TEXT NOT NULL,
    tool_output_ref TEXT NOT NULL,
    PRIMARY KEY (conversation_id, segment_ref, tool_output_ref)
);

CREATE TABLE IF NOT EXISTS chain_snapshots (
    ref TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    chain_json TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    tool_output_refs TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS media_outputs (
    ref TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    media_type TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    original_bytes INTEGER NOT NULL,
    compressed_bytes INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (conversation_id, ref)
);
"""

# Postgres FTS: tsvector columns + GIN indexes
FTS_SQL = """\
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='segments' AND column_name='summary_tsv') THEN
        ALTER TABLE segments ADD COLUMN summary_tsv tsvector;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='segments' AND column_name='full_text_tsv') THEN
        ALTER TABLE segments ADD COLUMN full_text_tsv tsvector;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='facts' AND column_name='facts_tsv') THEN
        ALTER TABLE facts ADD COLUMN facts_tsv tsvector;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='tool_outputs' AND column_name='content_tsv') THEN
        ALTER TABLE tool_outputs ADD COLUMN content_tsv tsvector;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_segments_summary_tsv ON segments USING gin(summary_tsv);
CREATE INDEX IF NOT EXISTS idx_segments_full_text_tsv ON segments USING gin(full_text_tsv);
CREATE INDEX IF NOT EXISTS idx_facts_tsv ON facts USING gin(facts_tsv);
CREATE INDEX IF NOT EXISTS idx_tool_outputs_tsv ON tool_outputs USING gin(content_tsv);

-- Triggers to keep tsvector columns in sync
CREATE OR REPLACE FUNCTION segments_summary_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.summary_tsv := to_tsvector('english', COALESCE(NEW.summary, ''));
    RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION segments_full_text_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.full_text_tsv := to_tsvector('english', COALESCE(NEW.full_text, ''));
    RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION facts_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.facts_tsv := to_tsvector('english',
        COALESCE(NEW.subject, '') || ' ' || COALESCE(NEW.verb, '') || ' ' ||
        COALESCE(NEW.object, '') || ' ' || COALESCE(NEW.what, ''));
    RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION tool_outputs_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_segments_summary_tsv ON segments;
CREATE TRIGGER trg_segments_summary_tsv BEFORE INSERT OR UPDATE ON segments
    FOR EACH ROW EXECUTE FUNCTION segments_summary_tsv_trigger();

DROP TRIGGER IF EXISTS trg_segments_full_text_tsv ON segments;
CREATE TRIGGER trg_segments_full_text_tsv BEFORE INSERT OR UPDATE ON segments
    FOR EACH ROW EXECUTE FUNCTION segments_full_text_tsv_trigger();

DROP TRIGGER IF EXISTS trg_facts_tsv ON facts;
CREATE TRIGGER trg_facts_tsv BEFORE INSERT OR UPDATE ON facts
    FOR EACH ROW EXECUTE FUNCTION facts_tsv_trigger();

DROP TRIGGER IF EXISTS trg_tool_outputs_tsv ON tool_outputs;
CREATE TRIGGER trg_tool_outputs_tsv BEFORE INSERT OR UPDATE ON tool_outputs
    FOR EACH ROW EXECUTE FUNCTION tool_outputs_tsv_trigger();

"""


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _turn_query_terms(query: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[a-zA-Z0-9_.%-]+", query or "") if len(term) >= 2]


def _text_term_hits(text: str, terms: list[str]) -> int:
    lowered = (text or "").lower()
    if not lowered or not terms:
        return 0
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))


def _matched_turn_side(query: str, user_text: str, assistant_text: str) -> str:
    query_lower = (query or "").strip().lower()
    user_lower = (user_text or "").lower()
    assistant_lower = (assistant_text or "").lower()
    user_hits = 0
    assistant_hits = 0
    if query_lower:
        if query_lower in user_lower:
            user_hits += 2
        if query_lower in assistant_lower:
            assistant_hits += 2
    terms = _turn_query_terms(query)
    user_hits += _text_term_hits(user_text, terms)
    assistant_hits += _text_term_hits(assistant_text, terms)
    if user_hits and assistant_hits:
        return "both"
    if user_hits:
        return "user"
    if assistant_hits:
        return "assistant"
    return "unknown"


def _build_turn_excerpt(
    query: str,
    user_text: str,
    assistant_text: str,
    matched_side: str,
    *,
    context_chars: int = 200,
) -> str:
    if matched_side == "user":
        return f"User: {_extract_excerpt(user_text or '', query, context_chars=context_chars)}"
    if matched_side == "assistant":
        return f"Assistant: {_extract_excerpt(assistant_text or '', query, context_chars=context_chars)}"
    if matched_side == "both":
        return (
            f"User: {_extract_excerpt(user_text or '', query, context_chars=context_chars)}\n\n"
            f"Assistant: {_extract_excerpt(assistant_text or '', query, context_chars=context_chars)}"
        )
    combined = f"User: {user_text or ''}\n\nAssistant: {assistant_text or ''}".strip()
    return _extract_excerpt(combined, query, context_chars=context_chars)


def _row_to_segment(row: dict, tags: list[str]) -> StoredSegment:
    metadata_raw = json.loads(row["metadata_json"])
    return StoredSegment(
        ref=row["ref"],
        conversation_id=row["conversation_id"],
        primary_tag=row["primary_tag"],
        tags=tags,
        summary=row["summary"],
        summary_tokens=row["summary_tokens"],
        full_text=row["full_text"],
        full_tokens=row["full_tokens"],
        messages=json.loads(row["messages_json"]),
        metadata=SegmentMetadata(
            entities=metadata_raw.get("entities", []),
            key_decisions=metadata_raw.get("key_decisions", []),
            action_items=metadata_raw.get("action_items", []),
            date_references=metadata_raw.get("date_references", []),
            code_refs=metadata_raw.get("code_refs", []),
            turn_count=metadata_raw.get("turn_count", 0),
            canonical_turn_ids=metadata_raw.get("canonical_turn_ids", []),
            start_turn_number=metadata_raw.get("start_turn_number", -1),
            end_turn_number=metadata_raw.get("end_turn_number", -1),
            generated_by_turn_id=metadata_raw.get("generated_by_turn_id", ""),
            session_date=metadata_raw.get("session_date", ""),
        ),
        created_at=_str_to_dt(row["created_at"]),
        start_timestamp=_str_to_dt(row["start_timestamp"]),
        end_timestamp=_str_to_dt(row["end_timestamp"]),
        compaction_model=row["compaction_model"],
        compression_ratio=row["compression_ratio"],
    )


def _row_to_summary(row: dict, tags: list[str]) -> StoredSummary:
    metadata_raw = json.loads(row["metadata_json"])
    return StoredSummary(
        ref=row["ref"],
        primary_tag=row["primary_tag"],
        tags=tags,
        summary=row["summary"],
        summary_tokens=row["summary_tokens"],
        full_tokens=row["full_tokens"],
        metadata=SegmentMetadata(
            entities=metadata_raw.get("entities", []),
            key_decisions=metadata_raw.get("key_decisions", []),
            action_items=metadata_raw.get("action_items", []),
            date_references=metadata_raw.get("date_references", []),
            code_refs=metadata_raw.get("code_refs", []),
            turn_count=metadata_raw.get("turn_count", 0),
            canonical_turn_ids=metadata_raw.get("canonical_turn_ids", []),
            start_turn_number=metadata_raw.get("start_turn_number", -1),
            end_turn_number=metadata_raw.get("end_turn_number", -1),
            generated_by_turn_id=metadata_raw.get("generated_by_turn_id", ""),
            session_date=metadata_raw.get("session_date", ""),
        ),
        created_at=_str_to_dt(row["created_at"]),
        start_timestamp=_str_to_dt(row["start_timestamp"]),
        end_timestamp=_str_to_dt(row["end_timestamp"]),
    )


def _row_to_canonical_turn(row: dict) -> CanonicalTurnRow:
    try:
        tags = json.loads(row.get("tags_json", "[]") or "[]")
    except Exception:
        tags = []
    try:
        code_refs = json.loads(row.get("code_refs_json", "[]") or "[]")
    except Exception:
        code_refs = []
    fact_signals: list[FactSignal] = []
    try:
        for item in json.loads(row.get("fact_signals_json", "[]") or "[]"):
            if isinstance(item, dict):
                fact_signals.append(
                    FactSignal(
                        subject=item.get("subject", ""),
                        verb=item.get("verb", ""),
                        object=item.get("object", ""),
                        status=item.get("status", ""),
                        fact_type=item.get("fact_type", ""),
                        what=item.get("what", ""),
                    )
                )
    except Exception:
        fact_signals = []
    return CanonicalTurnRow(
        conversation_id=row["conversation_id"],
        canonical_turn_id=str(row.get("canonical_turn_id", "") or ""),
        turn_number=int(row["turn_number"]) if row.get("turn_number") is not None else -1,
        turn_group_number=int(row["turn_group_number"]) if row.get("turn_group_number") is not None else -1,
        sort_key=float(row.get("sort_key", 0.0) or 0.0),
        turn_hash=row.get("turn_hash", "") or "",
        hash_version=int(row.get("hash_version", 0) or 0),
        normalized_user_text=row.get("normalized_user_text", "") or "",
        normalized_assistant_text=row.get("normalized_assistant_text", "") or "",
        user_content=row["user_content"] or "",
        assistant_content=row["assistant_content"] or "",
        user_raw_content=row["user_raw_content"],
        assistant_raw_content=row["assistant_raw_content"],
        primary_tag=row.get("primary_tag", "_general") or "_general",
        tags=list(tags or []),
        session_date=row.get("session_date", "") or "",
        sender=row.get("sender", "") or "",
        fact_signals=fact_signals,
        code_refs=list(code_refs or []),
        tagged_at=row.get("tagged_at") or None,
        compacted_at=row.get("compacted_at") or None,
        first_seen_at=row.get("first_seen_at") or None,
        last_seen_at=row.get("last_seen_at") or None,
        source_batch_id=str(row.get("source_batch_id", "") or "") or None,
        covered_ingestible_entries=int(row.get("covered_ingestible_entries", 1) or 1),
        created_at=row.get("created_at", "") or "",
        updated_at=row.get("updated_at", "") or "",
    )


def _merge_canonical_turn_rows(rows: list[CanonicalTurnRow]) -> dict[int, CanonicalTurnRow]:
    if not rows:
        return {}

    grouped: list[tuple[int, list[CanonicalTurnRow]]] = []
    explicit_groups = [
        int(getattr(row, "turn_group_number")) if getattr(row, "turn_group_number", None) is not None else -1
        for row in rows
    ]
    if all(group >= 0 for group in explicit_groups):
        grouped_by_turn: dict[int, list[CanonicalTurnRow]] = {}
        for row, turn_group_number in zip(rows, explicit_groups, strict=False):
            grouped_by_turn.setdefault(turn_group_number, []).append(row)
        grouped = sorted(grouped_by_turn.items(), key=lambda item: item[0])
    else:
        pending: list[CanonicalTurnRow] = []

        def _flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            grouped.append((len(grouped), list(pending)))
            pending = []

        for row in rows:
            has_user = bool(row.user_content)
            has_assistant = bool(row.assistant_content)
            if has_user and has_assistant:
                _flush_pending()
                grouped.append((len(grouped), [row]))
                continue
            if has_user:
                _flush_pending()
                pending = [row]
                continue
            if has_assistant:
                if pending:
                    pending.append(row)
                    _flush_pending()
                else:
                    grouped.append((len(grouped), [row]))
                continue
            _flush_pending()

        _flush_pending()

    merged: dict[int, CanonicalTurnRow] = {}
    for turn_number, group_rows in grouped:
        if not group_rows:
            continue
        first = group_rows[0]
        merged_row = CanonicalTurnRow(
            conversation_id=first.conversation_id,
            canonical_turn_id=first.canonical_turn_id,
            turn_number=turn_number,
            turn_group_number=turn_number,
            sort_key=min(row.sort_key for row in group_rows),
            turn_hash=first.turn_hash,
            hash_version=first.hash_version,
            normalized_user_text="",
            normalized_assistant_text="",
            user_content="",
            assistant_content="",
            user_raw_content=None,
            assistant_raw_content=None,
            primary_tag=first.primary_tag or "_general",
            tags=[],
            session_date="",
            sender="",
            fact_signals=[],
            code_refs=[],
            tagged_at=None,
            compacted_at=None,
            first_seen_at=None,
            last_seen_at=None,
            source_batch_id=first.source_batch_id,
            created_at=first.created_at,
            updated_at=first.updated_at,
        )
        tagged_values: list[str] = []
        compacted_values: list[str] = []
        for row in group_rows:
            if row.user_content and not merged_row.user_content:
                merged_row.user_content = row.user_content
                merged_row.user_raw_content = row.user_raw_content
                merged_row.normalized_user_text = row.normalized_user_text
                if row.canonical_turn_id:
                    merged_row.canonical_turn_id = row.canonical_turn_id
            if row.assistant_content and not merged_row.assistant_content:
                merged_row.assistant_content = row.assistant_content
                merged_row.assistant_raw_content = row.assistant_raw_content
                merged_row.normalized_assistant_text = row.normalized_assistant_text
                if not merged_row.canonical_turn_id and row.canonical_turn_id:
                    merged_row.canonical_turn_id = row.canonical_turn_id
            if row.primary_tag and (merged_row.primary_tag == "_general" or not merged_row.primary_tag):
                merged_row.primary_tag = row.primary_tag
            if not merged_row.session_date and row.session_date:
                merged_row.session_date = row.session_date
            if not merged_row.sender and row.sender:
                merged_row.sender = row.sender
            if row.source_batch_id and not merged_row.source_batch_id:
                merged_row.source_batch_id = row.source_batch_id
            if row.created_at and (not merged_row.created_at or row.created_at < merged_row.created_at):
                merged_row.created_at = row.created_at
            if row.updated_at and row.updated_at > merged_row.updated_at:
                merged_row.updated_at = row.updated_at
            if row.first_seen_at and (
                not merged_row.first_seen_at or row.first_seen_at < merged_row.first_seen_at
            ):
                merged_row.first_seen_at = row.first_seen_at
            if row.last_seen_at and (
                not merged_row.last_seen_at or row.last_seen_at > merged_row.last_seen_at
            ):
                merged_row.last_seen_at = row.last_seen_at
            if row.tagged_at:
                tagged_values.append(row.tagged_at)
            if row.compacted_at:
                compacted_values.append(row.compacted_at)
            for tag in row.tags:
                if tag not in merged_row.tags:
                    merged_row.tags.append(tag)
            for signal in row.fact_signals:
                if signal not in merged_row.fact_signals:
                    merged_row.fact_signals.append(signal)
            for code_ref in row.code_refs:
                if code_ref not in merged_row.code_refs:
                    merged_row.code_refs.append(code_ref)
        if len(tagged_values) == len(group_rows):
            merged_row.tagged_at = max(tagged_values)
        if len(compacted_values) == len(group_rows):
            merged_row.compacted_at = max(compacted_values)
        merged[turn_number] = merged_row
    return merged


class PostgresStore(ContextStore):
    """PostgreSQL storage backend with tsvector FTS and full protocol support."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn_local = threading.local()
        self._conn_lock = threading.Lock()
        self._connections: dict[int, psycopg.Connection] = {}
        self.search_config = None  # set by engine after construction
        self._ensure_schema()

    def _get_conn(self) -> psycopg.Connection:
        conn = getattr(self._conn_local, "conn", None)
        if conn is not None and not conn.closed:
            return conn

        thread_id = threading.get_ident()
        with self._conn_lock:
            conn = self._connections.get(thread_id)
            if conn is None or conn.closed:
                conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
                self._connections[thread_id] = conn
            self._conn_local.conn = conn
            return conn

    @contextmanager
    def conversation_reconcile(self, conversation_id: str):
        conn = self._get_conn()
        now = _dt_to_str(datetime.now(timezone.utc))
        with conn.transaction():
            conn.execute(
                """INSERT INTO conversation_lifecycle (conversation_id, generation, deleted, updated_at)
                VALUES (%s, 0, FALSE, %s)
                ON CONFLICT (conversation_id) DO UPDATE SET updated_at = EXCLUDED.updated_at""",
                (conversation_id, now),
            )
            conn.execute(
                "SELECT conversation_id FROM conversation_lifecycle WHERE conversation_id = %s FOR UPDATE",
                (conversation_id,),
            ).fetchone()
            yield

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        # Split SCHEMA_SQL by statements and execute individually
        for stmt in SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # Table/index already exists
        # Lifecycle/phase-tracked conversations table. Mirrors the SQLite
        # schema (see sqlite.py) — carries lifecycle_epoch (for
        # delete+resurrect invariants), a phase state machine
        # (init/ingesting/compacting/active/deleted), and per-request
        # metadata counters consumed by the progress tracker. The partial
        # index on (tenant_id, phase) excludes deleted rows so tenant-scoped
        # phase queries never scan tombstones.
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id                TEXT PRIMARY KEY,
                    tenant_id                      VARCHAR NOT NULL,
                    lifecycle_epoch                INT NOT NULL DEFAULT 1,
                    phase                          VARCHAR NOT NULL DEFAULT 'init'
                                                   CHECK (phase IN ('init','ingesting','compacting','active','deleted')),
                    pending_raw_payload_entries    INT NOT NULL DEFAULT 0,
                    last_raw_payload_entries       INT NOT NULL DEFAULT 0,
                    last_ingestible_payload_entries INT NOT NULL DEFAULT 0,
                    created_at                     TIMESTAMPTZ NOT NULL,
                    updated_at                     TIMESTAMPTZ NOT NULL,
                    deleted_at                     TIMESTAMPTZ NULL,
                    UNIQUE (tenant_id, conversation_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_tenant_phase
                    ON conversations(tenant_id, phase)
                     WHERE phase <> 'deleted'
            """)
        except Exception:
            logger.warning("conversations table bootstrap failed", exc_info=True)
        # Progress-tracking columns for the DB-derived progress model.
        # Mirrors the SQLite schema (see sqlite.py):
        #   covered_ingestible_entries — how many ingestible payload entries
        #     this canonical row represents (set at insert time). The progress
        #     denominator is SUM(covered_ingestible_entries).
        #   tagged_at — timestamp set when the tagger enriches the row. The
        #     progress numerator is
        #     SUM(covered_ingestible_entries WHERE tagged_at IS NOT NULL).
        # The two partial indexes below make each SUM path an index-only scan.
        # Postgres supports ADD COLUMN IF NOT EXISTS natively, so the ALTERs
        # are idempotent. Note: ``tagged_at`` already exists on the base
        # canonical_turns schema (as TEXT) — the ADD COLUMN IF NOT EXISTS is
        # defensive parity with SQLite. The partial indexes use sort_key
        # (not turn_number) because turn_number is view-derived on both
        # backends; sort_key is the physical ordering column.
        try:
            conn.execute("""
                ALTER TABLE canonical_turns
                    ADD COLUMN IF NOT EXISTS covered_ingestible_entries INT NOT NULL DEFAULT 1
            """)
            conn.execute("""
                ALTER TABLE canonical_turns
                    ADD COLUMN IF NOT EXISTS tagged_at TIMESTAMPTZ NULL
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_untagged
                    ON canonical_turns (conversation_id, sort_key)
                    WHERE tagged_at IS NULL
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_tagged
                    ON canonical_turns (conversation_id, tagged_at)
                    WHERE tagged_at IS NOT NULL
            """)
        except Exception:
            logger.warning("canonical_turns progress columns bootstrap failed", exc_info=True)
        # Ownership + lifecycle record for an ingestion episode. Mirrors the
        # SQLite schema (see sqlite.py). Progress counters (done/total) are
        # DERIVED from canonical_turns SUMs at read time — this row only
        # tracks ownership (`owner_worker_id`, `heartbeat_ts`), the largest
        # raw payload observed during the episode (`raw_payload_entries`),
        # and the status transitions. The partial unique index below
        # enforces at-most-one running episode per
        # (conversation, lifecycle_epoch) at the DB layer without requiring
        # a distributed lock — a concurrent worker attempting to INSERT a
        # second 'running' row collides at INSERT time.
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_episode (
                    episode_id            UUID PRIMARY KEY,
                    conversation_id       TEXT NOT NULL,
                    lifecycle_epoch       INT NOT NULL,
                    raw_payload_entries   INT NOT NULL DEFAULT 0,
                    started_at            TIMESTAMPTZ NOT NULL,
                    completed_at          TIMESTAMPTZ NULL,
                    status                VARCHAR NOT NULL
                                          CHECK (status IN ('running','completed','cancelled','abandoned')),
                    owner_worker_id       VARCHAR NOT NULL,
                    heartbeat_ts          TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_episode_active
                    ON ingestion_episode(conversation_id, lifecycle_epoch)
                    WHERE status = 'running'
            """)
        except Exception:
            logger.warning("ingestion_episode table bootstrap failed", exc_info=True)
        # Ownership + lifecycle record for a compaction operation. Mirrors
        # the SQLite schema (see sqlite.py). Sibling to ingestion_episode
        # but tracks the multi-phase compaction pipeline
        # (phase_index/phase_count/phase_name) rather than raw payload
        # counts. `status` carries a `queued` state in addition to the
        # episode lifecycle so workers can enqueue work ahead of execution,
        # and the partial unique index treats both `queued` and `running`
        # as active — only one pending-or-in-flight compaction is allowed
        # per (conversation, lifecycle_epoch) at the DB layer.
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compaction_operation (
                    operation_id      UUID PRIMARY KEY,
                    conversation_id   TEXT NOT NULL,
                    lifecycle_epoch   INT NOT NULL,
                    phase_index       INT NOT NULL DEFAULT 0,
                    phase_count       INT NOT NULL,
                    phase_name        VARCHAR NOT NULL,
                    status            VARCHAR NOT NULL
                                      CHECK (status IN ('queued','running','completed','cancelled','failed')),
                    started_at        TIMESTAMPTZ NOT NULL,
                    completed_at      TIMESTAMPTZ NULL,
                    owner_worker_id   VARCHAR NOT NULL,
                    heartbeat_ts      TIMESTAMPTZ NOT NULL,
                    error_message     TEXT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_compaction_operation_active
                    ON compaction_operation(conversation_id, lifecycle_epoch)
                    WHERE status IN ('queued','running')
            """)
        except Exception:
            logger.warning("compaction_operation table bootstrap failed", exc_info=True)
        # FTS setup
        try:
            conn.execute(FTS_SQL)
        except Exception as e:
            logger.warning("FTS setup issue (non-fatal): %s", e)
        # Backfill tsvector columns for existing rows
        try:
            conn.execute("UPDATE segments SET summary_tsv = to_tsvector('english', COALESCE(summary, '')) WHERE summary_tsv IS NULL")
            conn.execute("UPDATE segments SET full_text_tsv = to_tsvector('english', COALESCE(full_text, '')) WHERE full_text_tsv IS NULL")
            conn.execute("UPDATE facts SET facts_tsv = to_tsvector('english', COALESCE(subject,'') || ' ' || COALESCE(verb,'') || ' ' || COALESCE(object,'') || ' ' || COALESCE(what,'')) WHERE facts_tsv IS NULL")
        except Exception:
            pass
        try:
            conn.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_name='tag_summaries' AND column_name='code_refs') THEN
                        ALTER TABLE tag_summaries ADD COLUMN code_refs TEXT NOT NULL DEFAULT '[]';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_name='tag_summaries' AND column_name='generated_by_turn_id') THEN
                        ALTER TABLE tag_summaries ADD COLUMN generated_by_turn_id TEXT NOT NULL DEFAULT '';
                    END IF;
                END $$;
            """)
        except Exception:
            pass
        try:
            conn.execute("""
                DO $$ DECLARE
                    pk_cols text[];
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'request_captures'
                    ) THEN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='request_captures' AND column_name='conversation_id'
                        ) THEN
                            ALTER TABLE request_captures
                                ADD COLUMN conversation_id TEXT NOT NULL DEFAULT '';
                            UPDATE request_captures
                            SET conversation_id = COALESCE(data_json::jsonb->>'conversation_id', '');
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='request_captures' AND column_name='turn_id'
                        ) THEN
                            ALTER TABLE request_captures
                                ADD COLUMN turn_id TEXT NOT NULL DEFAULT '';
                            UPDATE request_captures
                            SET turn_id = COALESCE(data_json::jsonb->>'turn_id', '');
                        END IF;

                        SELECT array_agg(kcu.column_name ORDER BY kcu.ordinal_position)
                        INTO pk_cols
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                          ON tc.constraint_name = kcu.constraint_name
                         AND tc.table_schema = kcu.table_schema
                         AND tc.table_name = kcu.table_name
                        WHERE tc.table_name = 'request_captures'
                          AND tc.constraint_type = 'PRIMARY KEY';

                        IF pk_cols IS NULL OR pk_cols <> ARRAY['conversation_id', 'turn', 'turn_id'] THEN
                            IF EXISTS (
                                SELECT 1 FROM information_schema.table_constraints
                                WHERE table_name='request_captures'
                                  AND constraint_type='PRIMARY KEY'
                            ) THEN
                                ALTER TABLE request_captures DROP CONSTRAINT request_captures_pkey;
                            END IF;
                            ALTER TABLE request_captures
                                ADD PRIMARY KEY (conversation_id, turn, turn_id);
                        END IF;
                    END IF;
                END $$;
            """)
        except Exception:
            pass
        try:
            conn.execute("""
                DO $$ DECLARE
                    pk_cols text[];
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'tag_aliases'
                    ) THEN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='tag_aliases' AND column_name='conversation_id'
                        ) THEN
                            ALTER TABLE tag_aliases
                                ADD COLUMN conversation_id TEXT NOT NULL DEFAULT '';
                        END IF;

                        SELECT array_agg(kcu.column_name ORDER BY kcu.ordinal_position)
                        INTO pk_cols
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                          ON tc.constraint_name = kcu.constraint_name
                         AND tc.table_schema = kcu.table_schema
                         AND tc.table_name = kcu.table_name
                        WHERE tc.table_name = 'tag_aliases'
                          AND tc.constraint_type = 'PRIMARY KEY';

                        IF pk_cols IS NULL OR pk_cols <> ARRAY['alias', 'conversation_id'] THEN
                            IF EXISTS (
                                SELECT 1 FROM information_schema.table_constraints
                                WHERE table_name='tag_aliases'
                                  AND constraint_type='PRIMARY KEY'
                            ) THEN
                                ALTER TABLE tag_aliases DROP CONSTRAINT tag_aliases_pkey;
                            END IF;
                            ALTER TABLE tag_aliases
                                ADD PRIMARY KEY (alias, conversation_id);
                        END IF;
                    END IF;
                END $$;
            """)
        except Exception:
            pass
        try:
            self._normalize_request_turn_sequences()
        except Exception:
            logger.warning("request turn normalization failed", exc_info=True)
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_request_context_conv_turn_unique "
                "ON request_context(conversation_id, request_turn)"
            )
        except Exception:
            logger.warning("request_context unique index setup failed", exc_info=True)
        try:
            self._ensure_canonical_turn_schema()
            self._ensure_tag_summary_schema()
            self._ensure_canonical_turn_views()
        except Exception:
            logger.warning("canonical turn bootstrap failed", exc_info=True)
        try:
            self._ensure_compaction_scoping_columns()
        except Exception:
            logger.warning("compaction scoping columns bootstrap failed", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_canonical_turn_views(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """CREATE OR REPLACE VIEW canonical_turns_ordinal AS
               SELECT
                   ct.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY ct.conversation_id
                       ORDER BY ct.sort_key, ct.first_seen_at, ct.canonical_turn_id
                   ) - 1 AS turn_number
               FROM canonical_turns ct"""
        )

    def _ensure_canonical_turn_schema(self) -> None:
        conn = self._get_conn()
        for column in ("tagged_at", "compacted_at", "first_seen_at", "last_seen_at"):
            try:
                conn.execute(
                    f"ALTER TABLE canonical_turns ALTER COLUMN {column} DROP NOT NULL"
                )
            except Exception:
                pass
            try:
                conn.execute(
                    f"ALTER TABLE canonical_turns ALTER COLUMN {column} DROP DEFAULT"
                )
            except Exception:
                pass
            try:
                conn.execute(
                    f"UPDATE canonical_turns SET {column} = NULL WHERE {column} = ''"
                )
            except Exception:
                pass

    def _ensure_tag_summary_schema(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "ALTER TABLE tag_summaries ADD COLUMN source_canonical_turn_ids TEXT NOT NULL DEFAULT '[]'"
            )
        except Exception:
            pass
        try:
            conn.execute(
                "ALTER TABLE tag_summaries ADD COLUMN covers_through_canonical_turn_id TEXT NOT NULL DEFAULT ''"
            )
        except Exception:
            pass

    def _ensure_compaction_scoping_columns(self) -> None:
        """Add operation_id / compaction_operation_id columns used by the
        compaction-resume-parity takeover path. Postgres ``ADD COLUMN IF NOT
        EXISTS`` is idempotent. Existing rows backfill to the zero-UUID
        sentinel per the approved spec (line 61-63, rollout line 397-401).

        Also adds:
        - compaction_operation.created_at  (TIMESTAMPTZ NULL) — mirrors the
          SQLite column added in Task 7's migration.
        - 'abandoned' to the compaction_operation.status CHECK constraint —
          mirrors the SQLite constraint widening done in Task 7.
        """
        zero_uuid = "00000000-0000-0000-0000-000000000000"
        conn = self._get_conn()
        for table, column in (
            ("segments", "operation_id"),
            ("facts", "operation_id"),
            ("tag_summaries", "operation_id"),
            ("tag_summary_embeddings", "operation_id"),
            ("canonical_turns", "compaction_operation_id"),
        ):
            try:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} UUID"
                )
            except Exception:
                pass
            try:
                # Idempotent backfill. Matches zero rows on second run.
                conn.execute(
                    f"UPDATE {table} SET {column} = %s WHERE {column} IS NULL",
                    (zero_uuid,),
                )
            except Exception:
                pass

        # Add created_at to compaction_operation if not already present.
        # The CREATE TABLE above (Task 4) did not include this column; Task 7
        # added it to SQLite and we mirror it here. NULL is acceptable for
        # pre-existing rows.
        try:
            conn.execute(
                "ALTER TABLE compaction_operation "
                "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NULL"
            )
        except Exception:
            pass

        # Widen the status CHECK constraint to include 'abandoned'.
        # Postgres CHECK constraints cannot be modified in-place; we must
        # drop the existing one and add a replacement. The constraint name
        # was auto-generated by Postgres as compaction_operation_status_check.
        # Both the DROP and ADD are wrapped in individual try/except so the
        # method remains idempotent (second run: DROP fails on missing
        # constraint, ADD fails on duplicate — both silent).
        try:
            conn.execute(
                "ALTER TABLE compaction_operation "
                "DROP CONSTRAINT IF EXISTS compaction_operation_status_check"
            )
        except Exception:
            pass
        try:
            conn.execute(
                "ALTER TABLE compaction_operation "
                "ADD CONSTRAINT compaction_operation_status_check "
                "CHECK (status IN ('queued','running','completed','cancelled','failed','abandoned'))"
            )
        except Exception:
            pass

    def _get_tags_for_ref(self, ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag FROM segment_tags WHERE segment_ref = %s ORDER BY tag", (ref,)
        ).fetchall()
        return [r["tag"] for r in rows]

    def _lookup_canonical_turn_id_for_ordinal(self, conversation_id: str, turn_number: int) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT canonical_turn_id
               FROM canonical_turns_ordinal
               WHERE conversation_id = %s AND turn_number = %s""",
            (conversation_id, turn_number),
        ).fetchone()
        return str(row["canonical_turn_id"]) if row else None

    def _lookup_ordinal_for_canonical_turn_id(self, conversation_id: str, canonical_turn_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT turn_number
               FROM canonical_turns_ordinal
               WHERE conversation_id = %s AND canonical_turn_id = %s""",
            (conversation_id, canonical_turn_id),
        ).fetchone()
        return int(row["turn_number"]) if row else -1

    def _load_canonical_turn_rows_raw(self, conversation_id: str) -> list[CanonicalTurnRow]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT canonical_turn_id, conversation_id, turn_number, turn_group_number, sort_key, turn_hash, hash_version,
                      normalized_user_text, normalized_assistant_text, user_content, assistant_content,
                      user_raw_content, assistant_raw_content, primary_tag, tags_json, session_date,
                      sender, fact_signals_json, code_refs_json, tagged_at, compacted_at, first_seen_at,
                      last_seen_at, source_batch_id, created_at, updated_at
               FROM canonical_turns_ordinal
               WHERE conversation_id = %s
               ORDER BY sort_key, canonical_turn_id""",
            (conversation_id,),
        ).fetchall()
        return [_row_to_canonical_turn(row) for row in rows]

    def _load_canonical_turn_rows(self, conversation_id: str) -> list[CanonicalTurnRow]:
        # Lazy backfill: conversations ingested before turn_group_number was
        # introduced sit at -1 on every row and rely on the content-heuristic
        # fallback in _merge_canonical_turn_rows forever. Detect the all-legacy
        # case at read time and trigger a one-shot recompute so subsequent
        # reads use explicit groups (faster, deterministic under edits).
        rows = self._load_canonical_turn_rows_raw(conversation_id)
        if rows and all(r.turn_group_number < 0 for r in rows):
            try:
                self.recompute_canonical_turn_groups(conversation_id)
                rows = self._load_canonical_turn_rows_raw(conversation_id)
            except Exception:
                logger.warning(
                    "Lazy turn_group_number backfill failed for %s; falling back to content heuristics",
                    conversation_id[:12],
                    exc_info=True,
                )
        return rows

    def _batch_get_tags(self, refs: list[str]) -> dict[str, list[str]]:
        if not refs:
            return {}
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT segment_ref, tag FROM segment_tags WHERE segment_ref = ANY(%s) ORDER BY segment_ref, tag",
            (refs,),
        ).fetchall()
        result: dict[str, list[str]] = {ref: [] for ref in refs}
        for row in rows:
            result[row["segment_ref"]].append(row["tag"])
        return result

    def _row_to_fact(self, row: dict) -> Fact:
        return Fact.from_dict(row, dt_parser=_str_to_dt)

    def _row_to_fact_link(self, row: dict) -> FactLink:
        return FactLink(
            id=row["id"],
            source_fact_id=row["source_fact_id"],
            target_fact_id=row["target_fact_id"],
            relation_type=row["relation_type"],
            confidence=row["confidence"],
            context=row["context"],
            created_at=_str_to_dt(row["created_at"]) if row.get("created_at") else datetime.now(timezone.utc),
            created_by=row["created_by"],
        )

    # ------------------------------------------------------------------
    # SegmentStore
    # ------------------------------------------------------------------

    def store_segment(
        self,
        segment: StoredSegment,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> str:
        from ..types import CompactionLeaseLost
        conn = self._get_conn()
        primary_tag = segment.primary_tag
        summary_text = segment.summary
        full_text = segment.full_text
        metadata_dict = {
            "entities": segment.metadata.entities,
            "key_decisions": segment.metadata.key_decisions,
            "action_items": segment.metadata.action_items,
            "date_references": segment.metadata.date_references,
            "code_refs": getattr(segment.metadata, "code_refs", []),
            "turn_count": segment.metadata.turn_count,
            "canonical_turn_ids": getattr(segment.metadata, "canonical_turn_ids", []),
            "start_turn_number": getattr(segment.metadata, "start_turn_number", -1),
            "end_turn_number": getattr(segment.metadata, "end_turn_number", -1),
            "generated_by_turn_id": getattr(segment.metadata, "generated_by_turn_id", ""),
        }
        if segment.metadata.session_date:
            metadata_dict["session_date"] = segment.metadata.session_date

        guard_all = (
            operation_id is not None
            and owner_worker_id is not None
            and lifecycle_epoch is not None
        )

        with conn.transaction():
            if guard_all:
                # INSERT-SELECT form: writes zero rows if the compaction_operation
                # row no longer matches (status != 'running', owner mismatch, etc).
                cur = conn.execute(
                    """INSERT INTO segments
                    (ref, conversation_id, primary_tag, summary, full_text, messages_json,
                     metadata_json, summary_tokens, full_tokens, compression_ratio,
                     compaction_model, created_at, start_timestamp, end_timestamp,
                     operation_id)
                    SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                      FROM compaction_operation
                     WHERE operation_id = %s
                       AND conversation_id = %s
                       AND status = 'running'
                       AND owner_worker_id = %s
                       AND lifecycle_epoch = %s
                    ON CONFLICT (ref) DO UPDATE SET
                        conversation_id=EXCLUDED.conversation_id,
                        primary_tag=EXCLUDED.primary_tag,
                        summary=EXCLUDED.summary,
                        full_text=EXCLUDED.full_text,
                        messages_json=EXCLUDED.messages_json,
                        metadata_json=EXCLUDED.metadata_json,
                        summary_tokens=EXCLUDED.summary_tokens,
                        full_tokens=EXCLUDED.full_tokens,
                        compression_ratio=EXCLUDED.compression_ratio,
                        compaction_model=EXCLUDED.compaction_model,
                        created_at=EXCLUDED.created_at,
                        start_timestamp=EXCLUDED.start_timestamp,
                        end_timestamp=EXCLUDED.end_timestamp,
                        operation_id=EXCLUDED.operation_id""",
                    (
                        segment.ref, segment.conversation_id, primary_tag, summary_text,
                        full_text, json.dumps(segment.messages, default=str),
                        json.dumps(metadata_dict), segment.summary_tokens, segment.full_tokens,
                        segment.compression_ratio, segment.compaction_model,
                        _dt_to_str(segment.created_at), _dt_to_str(segment.start_timestamp),
                        _dt_to_str(segment.end_timestamp),
                        operation_id,
                        # WHERE clause params:
                        operation_id, segment.conversation_id,
                        owner_worker_id, lifecycle_epoch,
                    ),
                )
                if (cur.rowcount or 0) == 0:
                    raise CompactionLeaseLost(
                        operation_id=operation_id,
                        write_site="store_segment",
                    )
            else:
                # Legacy unconditional path — existing callers and test harnesses.
                conn.execute(
                    """INSERT INTO segments
                    (ref, conversation_id, primary_tag, summary, full_text, messages_json,
                     metadata_json, summary_tokens, full_tokens, compression_ratio,
                     compaction_model, created_at, start_timestamp, end_timestamp)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (ref) DO UPDATE SET
                        conversation_id=EXCLUDED.conversation_id, primary_tag=EXCLUDED.primary_tag,
                        summary=EXCLUDED.summary, full_text=EXCLUDED.full_text,
                        messages_json=EXCLUDED.messages_json, metadata_json=EXCLUDED.metadata_json,
                        summary_tokens=EXCLUDED.summary_tokens, full_tokens=EXCLUDED.full_tokens,
                        compression_ratio=EXCLUDED.compression_ratio, compaction_model=EXCLUDED.compaction_model,
                        created_at=EXCLUDED.created_at, start_timestamp=EXCLUDED.start_timestamp,
                        end_timestamp=EXCLUDED.end_timestamp""",
                    (segment.ref, segment.conversation_id, primary_tag, summary_text,
                     full_text, json.dumps(segment.messages, default=str),
                     json.dumps(metadata_dict), segment.summary_tokens, segment.full_tokens,
                     segment.compression_ratio, segment.compaction_model,
                     _dt_to_str(segment.created_at), _dt_to_str(segment.start_timestamp),
                     _dt_to_str(segment.end_timestamp)),
                )
            conn.execute("DELETE FROM segment_tags WHERE segment_ref = %s", (segment.ref,))
            for tag in segment.tags:
                conn.execute(
                    "INSERT INTO segment_tags (segment_ref, tag) VALUES (%s, %s)",
                    (segment.ref, tag),
                )
        return segment.ref

    def get_segment(self, ref: str, conversation_id: str | None = None) -> StoredSegment | None:
        conn = self._get_conn()
        if conversation_id is not None:
            row = conn.execute(
                "SELECT * FROM segments WHERE ref = %s AND conversation_id = %s",
                (ref, conversation_id),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM segments WHERE ref = %s", (ref,)).fetchone()
        if not row:
            return None
        return _row_to_segment(row, self._get_tags_for_ref(ref))

    def get_summary(self, ref: str, conversation_id: str | None = None) -> StoredSummary | None:
        conn = self._get_conn()
        if conversation_id is not None:
            row = conn.execute(
                "SELECT * FROM segments WHERE ref = %s AND conversation_id = %s",
                (ref, conversation_id),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM segments WHERE ref = %s", (ref,)).fetchone()
        if not row:
            return None
        return _row_to_summary(row, self._get_tags_for_ref(ref))

    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[StoredSegment]:
        conn = self._get_conn()
        if conversation_id is not None and limit is not None and limit > 0:
            rows = conn.execute(
                "SELECT * FROM segments WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s",
                (conversation_id, limit),
            ).fetchall()
        elif conversation_id is not None:
            rows = conn.execute(
                "SELECT * FROM segments WHERE conversation_id = %s ORDER BY created_at DESC",
                (conversation_id,),
            ).fetchall()
        elif limit is not None and limit > 0:
            rows = conn.execute(
                "SELECT * FROM segments ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM segments ORDER BY created_at DESC").fetchall()
        if not rows:
            return []
        refs = [row["ref"] for row in rows]
        tags_map = self._batch_get_tags(refs)
        return [_row_to_segment(row, tags_map.get(row["ref"], [])) for row in rows]

    def get_summaries_by_tags(
        self, tags: list[str], min_overlap: int = 1, limit: int = 10,
        before: datetime | None = None, after: datetime | None = None,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        if not tags:
            return []
        conn = self._get_conn()
        query = """
            SELECT s.*, COUNT(st.tag) as overlap_count
            FROM segments s
            JOIN segment_tags st ON s.ref = st.segment_ref
            WHERE st.tag = ANY(%s)
        """
        params: list = [tags]
        if conversation_id is not None:
            query += " AND s.conversation_id = %s"
            params.append(conversation_id)
        if before:
            query += " AND s.created_at < %s"
            params.append(_dt_to_str(before))
        if after:
            query += " AND s.created_at > %s"
            params.append(_dt_to_str(after))
        query += """
            GROUP BY s.ref
            HAVING COUNT(st.tag) >= %s
            ORDER BY COUNT(st.tag) DESC, s.created_at DESC
            LIMIT %s
        """
        params.extend([min_overlap, limit])
        rows = conn.execute(query, params).fetchall()
        refs = [row["ref"] for row in rows]
        tags_map = self._batch_get_tags(refs)
        return [_row_to_summary(row, tags_map[row["ref"]]) for row in rows]

    def search(self, query: str, tags: list[str] | None = None, limit: int = 5, conversation_id: str | None = None) -> list[StoredSummary]:
        conn = self._get_conn()
        tsquery = " & ".join(query.split())
        if tags:
            sql = """SELECT DISTINCT s.* FROM segments s
                JOIN segment_tags st ON s.ref = st.segment_ref
                WHERE s.summary_tsv @@ to_tsquery('english', %s)
                AND st.tag = ANY(%s)"""
            params: list = [tsquery, tags]
            if conversation_id is not None:
                sql += " AND s.conversation_id = %s"
                params.append(conversation_id)
            sql += " ORDER BY s.created_at DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        else:
            sql = """SELECT * FROM segments
                WHERE summary_tsv @@ to_tsquery('english', %s)"""
            params = [tsquery]
            if conversation_id is not None:
                sql += " AND conversation_id = %s"
                params.append(conversation_id)
            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        refs = [row["ref"] for row in rows]
        tags_map = self._batch_get_tags(refs)
        return [_row_to_summary(row, tags_map[row["ref"]]) for row in rows]

    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute("""
                SELECT st.tag,
                       COUNT(DISTINCT st.segment_ref) as usage_count,
                       COALESCE(SUM(s.full_tokens), 0) as total_full,
                       COALESCE(SUM(s.summary_tokens), 0) as total_summary,
                       MIN(s.created_at) as oldest,
                       MAX(s.created_at) as newest
                FROM segment_tags st
                JOIN segments s ON s.ref = st.segment_ref
                WHERE s.conversation_id = %s
                GROUP BY st.tag
                ORDER BY usage_count DESC
            """, (conversation_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT st.tag,
                       COUNT(DISTINCT st.segment_ref) as usage_count,
                       COALESCE(SUM(s.full_tokens), 0) as total_full,
                       COALESCE(SUM(s.summary_tokens), 0) as total_summary,
                       MIN(s.created_at) as oldest,
                       MAX(s.created_at) as newest
                FROM segment_tags st
                JOIN segments s ON s.ref = st.segment_ref
                GROUP BY st.tag
                ORDER BY usage_count DESC
            """).fetchall()
        return [
            TagStats(
                tag=row["tag"],
                usage_count=row["usage_count"],
                total_full_tokens=row["total_full"],
                total_summary_tokens=row["total_summary"],
                oldest_segment=_str_to_dt(row["oldest"]),
                newest_segment=_str_to_dt(row["newest"]),
            )
            for row in rows
        ]

    def get_conversation_stats(self) -> list[ConversationStats]:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT conversation_id, COUNT(*) as seg_count,
                   SUM(full_tokens) as total_full, SUM(summary_tokens) as total_summary,
                   MIN(created_at) as oldest, MAX(created_at) as newest,
                   compaction_model
            FROM segments WHERE conversation_id != ''
            GROUP BY conversation_id, compaction_model
            ORDER BY MAX(created_at) DESC
        """).fetchall()
        # Batch-fetch distinct tags per conversation (avoids N+1)
        conv_ids = [row["conversation_id"] for row in rows]
        tags_by_conv: dict[str, list[str]] = {cid: [] for cid in conv_ids}
        if conv_ids:
            tag_rows = conn.execute(
                """SELECT s.conversation_id, st.tag
                FROM segment_tags st
                JOIN segments s ON s.ref = st.segment_ref
                WHERE s.conversation_id = ANY(%s)
                GROUP BY s.conversation_id, st.tag
                ORDER BY st.tag""",
                (conv_ids,),
            ).fetchall()
            for tr in tag_rows:
                tags_by_conv[tr["conversation_id"]].append(tr["tag"])

        results = []
        for row in rows:
            results.append(ConversationStats(
                conversation_id=row["conversation_id"],
                segment_count=row["seg_count"],
                total_full_tokens=row["total_full"],
                total_summary_tokens=row["total_summary"],
                oldest_segment=_str_to_dt(row["oldest"]),
                newest_segment=_str_to_dt(row["newest"]),
                distinct_tags=tags_by_conv.get(row["conversation_id"], []),
                compaction_model=row["compaction_model"],
            ))
        return results

    def get_tag_aliases(self, conversation_id: str | None = None) -> dict[str, str]:
        conn = self._get_conn()
        params: list[object] = []
        query = "SELECT alias, canonical, conversation_id FROM tag_aliases"
        if conversation_id:
            query += " WHERE conversation_id IN ('', %s)"
            params.append(conversation_id)
        else:
            query += " WHERE conversation_id = ''"
        query += " ORDER BY CASE WHEN conversation_id = '' THEN 0 ELSE 1 END, alias"
        rows = conn.execute(query, params).fetchall()
        aliases: dict[str, str] = {}
        for row in rows:
            aliases[row["alias"]] = row["canonical"]
        return aliases

    def set_tag_alias(
        self,
        alias: str,
        canonical: str,
        conversation_id: str = "",
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tag_aliases (alias, conversation_id, canonical)
            VALUES (%s, %s, %s)
            ON CONFLICT (alias, conversation_id)
            DO UPDATE SET canonical = EXCLUDED.canonical""",
            (alias, conversation_id or "", canonical),
        )

    def delete_tag_aliases_for_conversation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM tag_aliases WHERE conversation_id = %s",
            (conversation_id,),
        )
        return int(cur.rowcount or 0)

    def delete_segment(self, ref: str) -> bool:
        conn = self._get_conn()
        with conn.transaction():
            conn.execute("DELETE FROM segment_tags WHERE segment_ref = %s", (ref,))
            conn.execute("DELETE FROM segment_chunks WHERE segment_ref = %s", (ref,))
            conn.execute("DELETE FROM facts WHERE segment_ref = %s", (ref,))
            cur = conn.execute("DELETE FROM segments WHERE ref = %s", (ref,))
        return cur.rowcount > 0

    def cleanup(self, max_age: timedelta | None = None, max_total_tokens: int | None = None) -> int:
        if not max_age:
            return 0
        conn = self._get_conn()
        cutoff = _dt_to_str(datetime.now(timezone.utc) - max_age)
        cur = conn.execute("DELETE FROM segments WHERE created_at < %s", (cutoff,))
        return cur.rowcount

    def _table_exists(self, conn, table: str) -> bool:
        row = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (table,),
        ).fetchone()
        if not row:
            return False
        if isinstance(row, dict):
            return bool(next(iter(row.values())))
        return bool(row[0])

    def _delete_conversation_rows(self, conn, table: str, conversation_id: str) -> int:
        if not self._table_exists(conn, table):
            return 0
        cur = conn.execute(
            f"DELETE FROM {table} WHERE conversation_id = %s",
            (conversation_id,),
        )
        return int(cur.rowcount or 0)

    def activate_conversation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            """INSERT INTO conversation_lifecycle (conversation_id, generation, deleted, updated_at)
            VALUES (%s, 0, FALSE, %s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                deleted = FALSE,
                updated_at = EXCLUDED.updated_at
            RETURNING generation""",
            (conversation_id, _dt_to_str(datetime.now(timezone.utc))),
        ).fetchone()
        return int(row["generation"] if isinstance(row, dict) else row[0])

    def begin_conversation_deletion(self, conversation_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            """INSERT INTO conversation_lifecycle (conversation_id, generation, deleted, updated_at)
            VALUES (%s, 1, TRUE, %s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                generation = conversation_lifecycle.generation + 1,
                deleted = TRUE,
                updated_at = EXCLUDED.updated_at
            RETURNING generation""",
            (conversation_id, _dt_to_str(datetime.now(timezone.utc))),
        ).fetchone()
        return int(row["generation"] if isinstance(row, dict) else row[0])

    def get_conversation_generation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT generation FROM conversation_lifecycle WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if not row:
            return 0
        return int(row["generation"] if isinstance(row, dict) else row[0])

    def is_conversation_generation_current(
        self,
        conversation_id: str,
        generation: int,
    ) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT generation, deleted FROM conversation_lifecycle WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if not row:
            return int(generation or 0) == 0
        current = int(row["generation"] if isinstance(row, dict) else row[0])
        deleted = bool(row["deleted"] if isinstance(row, dict) else row[1])
        return current == int(generation or 0) and not deleted

    # ------------------------------------------------------------------
    # Conversation row lifecycle (progress-bar redesign `conversations` table)
    # ------------------------------------------------------------------
    # These methods mirror the SQLiteStore implementations (see sqlite.py) on
    # the newer `conversations` table — the one that carries
    # ``lifecycle_epoch`` and ``phase`` for the progress tracker and
    # delete+resurrect invariants. They are separate from the legacy
    # ``conversation_lifecycle`` table (activate/begin deletion) just above.

    def upsert_conversation(self, *, tenant_id: str, conversation_id: str) -> None:
        """Create the conversations row if missing; otherwise just refresh updated_at.

        Epoch starts at 1 on new rows; never bumped by this method.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id, tenant_id, created_at, updated_at
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at
            """,
            (conversation_id, tenant_id, now, now),
        )

    def get_lifecycle_epoch(self, conversation_id: str) -> int:
        """Return the current lifecycle_epoch. Raises KeyError if no row exists."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT lifecycle_epoch FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return int(row["lifecycle_epoch"] if isinstance(row, dict) else row[0])

    def get_conversation_phase(self, conversation_id: str) -> str:
        """Return the current phase for the conversation.

        Returns one of ``"init" | "ingesting" | "compacting" | "active" |
        "deleted"``. Raises ``KeyError`` if no row exists.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT phase FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return str(row["phase"] if isinstance(row, dict) else row[0])

    def mark_conversation_deleted(self, conversation_id: str) -> None:
        """Admin-flow delete: sets phase='deleted' and stamps deleted_at.

        Called only by the delete endpoint — caller is authoritative; no
        epoch check needed. Raises KeyError if no row exists so callers get
        symmetric signaling with ``increment_lifecycle_epoch_on_resurrect``.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE conversations
               SET phase = 'deleted',
                   deleted_at = %s,
                   updated_at = %s
             WHERE conversation_id = %s
            """,
            (now, now, conversation_id),
        )
        if cur.rowcount == 0:
            raise KeyError(conversation_id)
        conn.execute(
            """
            UPDATE ingestion_episode
               SET status = 'abandoned',
                   completed_at = COALESCE(completed_at, %s)
             WHERE conversation_id = %s
               AND status = 'running'
            """,
            (now, conversation_id),
        )
        conn.execute(
            """
            UPDATE compaction_operation
               SET status = 'cancelled',
                   completed_at = COALESCE(completed_at, %s)
             WHERE conversation_id = %s
               AND status IN ('queued', 'running')
            """,
            (now, conversation_id),
        )

    def increment_lifecycle_epoch_on_resurrect(self, conversation_id: str) -> int:
        """Bump lifecycle_epoch ONLY when phase == 'deleted'.

        Uses a single-statement ``UPDATE ... WHERE phase='deleted' RETURNING``
        which is atomic under Postgres's autocommit — a concurrent caller
        cannot double-bump because the guard predicate only matches once. If
        the UPDATE returns no rows (phase already 'init'/'active'/etc.), we
        fall back to reading the current epoch. Raises KeyError if no row
        exists.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        row = conn.execute(
            """
            UPDATE conversations
               SET lifecycle_epoch = lifecycle_epoch + 1,
                   phase = 'init',
                   deleted_at = NULL,
                   updated_at = %s
             WHERE conversation_id = %s
               AND phase = 'deleted'
            RETURNING lifecycle_epoch
            """,
            (now, conversation_id),
        ).fetchone()
        if row is not None:
            new_epoch = int(row["lifecycle_epoch"] if isinstance(row, dict) else row[0])
            conn.execute(
                """
                UPDATE ingestion_episode
                   SET status = 'abandoned',
                       completed_at = COALESCE(completed_at, %s)
                 WHERE conversation_id = %s
                   AND lifecycle_epoch < %s
                   AND status = 'running'
                """,
                (now, conversation_id, new_epoch),
            )
            conn.execute(
                """
                UPDATE compaction_operation
                   SET status = 'cancelled',
                       completed_at = COALESCE(completed_at, %s)
                 WHERE conversation_id = %s
                   AND lifecycle_epoch < %s
                   AND status IN ('queued', 'running')
                """,
                (now, conversation_id, new_epoch),
            )
            return new_epoch
        # Not deleted (already init/active or unknown) — read current epoch.
        row = conn.execute(
            "SELECT lifecycle_epoch FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return int(row["lifecycle_epoch"] if isinstance(row, dict) else row[0])

    def read_progress_snapshot(self, conversation_id: str) -> ProgressSnapshot:
        """Derive the current progress state for a conversation.

        ``total_ingestible`` and ``done_ingestible`` are computed at read
        time via ``SUM(covered_ingestible_entries)`` over ``canonical_turns``
        (filtered by ``tagged_at IS NOT NULL`` for the numerator) — never
        stored counters that could drift from canonical truth. Point
        lookups in ``ingestion_episode`` / ``compaction_operation`` are
        scoped to the conversation's current ``lifecycle_epoch`` so stale
        rows from earlier resurrected lifecycles cannot shadow the live one.

        Raises ``KeyError`` if the conversation row does not exist.
        """
        conn = self._get_conn()
        with conn.transaction():
            row = conn.execute(
                """
                SELECT lifecycle_epoch, phase,
                       last_raw_payload_entries, last_ingestible_payload_entries
                  FROM conversations
                 WHERE conversation_id = %s
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            epoch = row["lifecycle_epoch"]
            phase = row["phase"]
            last_raw = row["last_raw_payload_entries"]
            last_ing = row["last_ingestible_payload_entries"]

            totals = conn.execute(
                """
                SELECT COALESCE(SUM(covered_ingestible_entries), 0) AS total_ing,
                       COALESCE(SUM(CASE WHEN tagged_at IS NOT NULL
                                         THEN covered_ingestible_entries ELSE 0 END), 0) AS done_ing
                  FROM canonical_turns
                 WHERE conversation_id = %s
                """,
                (conversation_id,),
            ).fetchone()
            total_ing = totals["total_ing"]
            done_ing = totals["done_ing"]

            ep_row = conn.execute(
                """
                SELECT episode_id, raw_payload_entries, owner_worker_id, heartbeat_ts
                  FROM ingestion_episode
                 WHERE conversation_id = %s
                   AND lifecycle_epoch = %s
                   AND status = 'running'
                 ORDER BY started_at DESC, episode_id DESC
                 LIMIT 1
                """,
                (conversation_id, epoch),
            ).fetchone()
            active_episode = (
                ActiveEpisodeSnapshot(
                    episode_id=str(ep_row["episode_id"]),
                    raw_payload_entries=int(ep_row["raw_payload_entries"]),
                    owner_worker_id=str(ep_row["owner_worker_id"]),
                    heartbeat_ts=str(ep_row["heartbeat_ts"]),
                )
                if ep_row is not None
                else None
            )

            cop_row = conn.execute(
                """
                SELECT operation_id, phase_name, phase_index, phase_count, status
                  FROM compaction_operation
                 WHERE conversation_id = %s
                   AND lifecycle_epoch = %s
                   AND status IN ('queued','running')
                 ORDER BY
                     CASE status WHEN 'running' THEN 0 ELSE 1 END,
                     started_at DESC,
                     operation_id DESC
                 LIMIT 1
                """,
                (conversation_id, epoch),
            ).fetchone()
            active_compaction = (
                ActiveCompactionSnapshot(
                    operation_id=str(cop_row["operation_id"]),
                    phase_name=str(cop_row["phase_name"]),
                    phase_index=int(cop_row["phase_index"]),
                    phase_count=int(cop_row["phase_count"]),
                    status=str(cop_row["status"]),
                )
                if cop_row is not None
                else None
            )

        return ProgressSnapshot(
            conversation_id=conversation_id,
            lifecycle_epoch=int(epoch),
            phase=str(phase),
            total_ingestible=int(total_ing),
            done_ingestible=int(done_ing),
            last_raw_payload_entries=int(last_raw),
            last_ingestible_payload_entries=int(last_ing),
            active_episode=active_episode,
            active_compaction=active_compaction,
        )

    # ------------------------------------------------------------------
    # Ingestion episode CRUD (epoch-guarded)
    # ------------------------------------------------------------------
    # Ownership-free widening upsert + epoch-scoped lease/heartbeat/complete
    # operations on the `ingestion_episode` table created in `_ensure_schema`.
    # All four methods filter on `lifecycle_epoch` in SQL so a stale thread
    # carrying an old epoch cannot mutate the current lifecycle's row. The
    # partial unique index `(conversation_id, lifecycle_epoch) WHERE status
    # = 'running'` supplies the conflict target for the upsert.
    #
    # Note: Postgres uses ``GREATEST()`` (scalar) where SQLite uses
    # ``MAX()`` — in PG, ``MAX()`` is an aggregate and cannot appear in the
    # SET clause of an ``ON CONFLICT DO UPDATE``. ``refresh_ingestion_heartbeat``
    # and ``complete_ingestion_episode`` double-scope the epoch filter with a
    # correlated subquery against ``conversations.lifecycle_epoch`` (the
    # authoritative source) so a stale thread whose conversation was
    # resurrected to a newer epoch is rejected at SQL level even if its
    # supplied ``lifecycle_epoch`` still matches an old ``ingestion_episode``
    # row.

    def upsert_ingestion_episode(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        raw_payload_entries: int,
    ) -> None:
        """Ownership-free upsert. On INSERT, creates a running episode with
        the given worker as initial owner. On CONFLICT (another running row
        exists), ONLY widens ``raw_payload_entries`` via ``GREATEST`` — does
        NOT change ownership or other fields. Idempotent.

        Uses the partial unique index
        ``(conversation_id, lifecycle_epoch) WHERE status = 'running'`` as
        the conflict target.
        """
        import uuid
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO ingestion_episode (
                episode_id, conversation_id, lifecycle_epoch,
                raw_payload_entries, started_at, status,
                owner_worker_id, heartbeat_ts
            ) VALUES (%s, %s, %s, %s, %s, 'running', %s, %s)
            ON CONFLICT (conversation_id, lifecycle_epoch) WHERE status = 'running'
            DO UPDATE SET
                raw_payload_entries =
                    GREATEST(ingestion_episode.raw_payload_entries, EXCLUDED.raw_payload_entries)
            """,
            (
                uuid.uuid4(), conversation_id, lifecycle_epoch,
                raw_payload_entries, now, worker_id, now,
            ),
        )

    def claim_ingestion_lease(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        lease_ttl_s: float,
    ) -> bool:
        """Claim the ingestion lease for this (conversation, lifecycle_epoch).

        Returns True iff the caller now owns the lease. Rules:
          - If caller already owns it: refresh heartbeat, return True.
          - If current heartbeat is stale (older than ``lease_ttl_s``):
            take over, return True.
          - Otherwise: another worker owns a fresh lease; return False.
        Epoch-scoped: filters on ``lifecycle_epoch`` so a stale epoch
        cannot claim a lease on the current lifecycle.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=lease_ttl_s)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE ingestion_episode
               SET owner_worker_id = %s, heartbeat_ts = %s
             WHERE conversation_id = %s AND status = 'running'
               AND lifecycle_epoch = %s
               AND (owner_worker_id = %s OR heartbeat_ts < %s)
            """,
            (
                worker_id, now, conversation_id, lifecycle_epoch,
                worker_id, cutoff,
            ),
        )
        return cur.rowcount == 1

    def refresh_ingestion_heartbeat(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        """Epoch-scoped heartbeat refresh. A stale thread carrying an old
        epoch cannot refresh a new lifecycle's heartbeat. Returns True iff
        a matching (running, caller-owned, epoch-matched) row was updated.

        The ``lifecycle_epoch`` filter is doubly-scoped: it must match both
        the episode row AND the authoritative ``conversations`` row — a
        stale thread that thinks it is still on epoch N but whose
        conversation was resurrected to epoch N+1 is rejected at SQL level.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE ingestion_episode SET heartbeat_ts = %s
             WHERE conversation_id = %s AND status = 'running'
               AND lifecycle_epoch = %s
               AND lifecycle_epoch = (
                   SELECT lifecycle_epoch FROM conversations
                    WHERE conversation_id = %s
               )
               AND owner_worker_id = %s
            """,
            (now, conversation_id, lifecycle_epoch, conversation_id, worker_id),
        )
        return cur.rowcount == 1

    def complete_ingestion_episode(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        """Race-guarded completion. Epoch-scoped. Returns True iff:
          - A running episode exists at the caller's ``lifecycle_epoch``.
          - The caller is the current owner.
          - The caller's epoch equals the conversation's current
            ``lifecycle_epoch`` (stale-epoch guard — see
            ``refresh_ingestion_heartbeat`` for the same mechanism).
          - No untagged canonical rows remain (NOT EXISTS guard).
        Returns False if any condition fails.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE ingestion_episode
               SET status = 'completed', completed_at = %s
             WHERE conversation_id = %s AND status = 'running'
               AND lifecycle_epoch = %s
               AND lifecycle_epoch = (
                   SELECT lifecycle_epoch FROM conversations
                    WHERE conversation_id = %s
               )
               AND owner_worker_id = %s
               AND NOT EXISTS (
                   SELECT 1 FROM canonical_turns
                    WHERE conversation_id = %s AND tagged_at IS NULL
               )
            """,
            (
                now, conversation_id, lifecycle_epoch, conversation_id,
                worker_id, conversation_id,
            ),
        )
        return cur.rowcount == 1

    # ------------------------------------------------------------------
    # Compaction operation CRUD (epoch-guarded)
    # ------------------------------------------------------------------
    # Multi-phase compaction pipeline tracked on the ``compaction_operation``
    # table. ``start_compaction_operation`` issues a new row in ``'queued'``
    # status and relies on the partial unique index
    # ``(conversation_id, lifecycle_epoch) WHERE status IN
    # ('queued','running')`` to keep at most one active operation per
    # (conversation, epoch) — a second start on the same active window
    # raises ``psycopg.errors.UniqueViolation`` and the caller is
    # expected to retry/wait. ``claim_compaction_lease`` matches the
    # ingestion-lease pattern (owner-or-stale heartbeat), and the three
    # terminal/phase operations double-scope the epoch filter with a
    # correlated subquery against ``conversations.lifecycle_epoch`` so a
    # stale thread whose conversation was resurrected to a newer epoch
    # is rejected at SQL level.

    def start_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_count: int,
        phase_name: str,
    ) -> str:
        """Insert a fresh ``compaction_operation`` row in ``'queued'``
        status. Returns the new ``operation_id`` (UUID string).

        Raises ``psycopg.errors.UniqueViolation`` (via the partial
        unique index on status IN ('queued','running')) if another
        active operation already exists for this (conversation, epoch).
        The caller is expected to retry or wait.
        """
        import uuid
        op_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO compaction_operation (
                operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, owner_worker_id, heartbeat_ts
            ) VALUES (%s, %s, %s, 0, %s, %s, 'queued', %s, %s, %s)
            """,
            (
                op_id, conversation_id, lifecycle_epoch,
                phase_count, phase_name, now, worker_id, now,
            ),
        )
        return str(op_id)

    def claim_compaction_lease(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        lease_ttl_s: float,
    ) -> "CompactionLeaseClaim":
        """Postgres flavor — see SQLiteStore docstring for full semantics.

        Reads (operation_id, owner_worker_id) in a SELECT … FOR UPDATE before
        applying the conditional UPDATE so the takeover path receives
        prev_operation_id atomically without a second round-trip.

        Returns a CompactionLeaseClaim with:
          - claimed=True  iff caller already owns the row OR heartbeat is stale.
          - prev_operation_id / prev_owner_worker_id from the pre-update row
            (None when no active row existed at the given lifecycle_epoch).
        """
        from ..types import CompactionLeaseClaim

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=lease_ttl_s)
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.transaction():
            pre = conn.execute(
                """
                SELECT operation_id, owner_worker_id
                  FROM compaction_operation
                 WHERE conversation_id = %s AND status IN ('queued','running')
                   AND lifecycle_epoch = %s
                 ORDER BY started_at DESC
                 LIMIT 1
                 FOR UPDATE
                """,
                (conversation_id, lifecycle_epoch),
            ).fetchone()
            prev_op = str(pre["operation_id"]) if pre else None
            prev_owner = str(pre["owner_worker_id"]) if pre else None

            cur = conn.execute(
                """
                UPDATE compaction_operation
                   SET owner_worker_id = %s, heartbeat_ts = %s
                 WHERE conversation_id = %s AND status IN ('queued','running')
                   AND lifecycle_epoch = %s
                   AND (owner_worker_id = %s OR heartbeat_ts < %s)
                """,
                (
                    worker_id, now, conversation_id, lifecycle_epoch,
                    worker_id, cutoff,
                ),
            )
            claimed = (cur.rowcount or 0) > 0
        return CompactionLeaseClaim(
            claimed=claimed,
            prev_operation_id=prev_op,
            prev_owner_worker_id=prev_owner,
        )

    def cleanup_abandoned_compaction(
        self,
        *,
        conversation_id: str,
        dead_operation_id: str,
        new_operation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_count: int,
    ) -> bool:
        """Postgres flavor of the atomic takeover-cleanup transaction.

        Returns True iff this call performed the transition (dead_op was
        'running' and we abandoned it + inserted new_op). Returns False
        when the dead_op was already abandoned/completed — idempotent
        re-run; the new-row INSERT is skipped to preserve the
        one-active invariant enforced by ``idx_compaction_operation_active``.

        Ordering inside the single ``conn.transaction()`` block:

        1. UPDATE dead_op → 'abandoned'. rowcount decides fresh vs idempotent.
        2. DELETE scoped partial writes from segments / facts /
           tag_summaries / tag_summary_embeddings (idempotent no-ops on
           already-absent rows).
        3. UPDATE canonical_turns: NULL out compacted_at /
           compaction_operation_id where compaction_operation_id = dead_op
           (also idempotent).
        4. ONLY IF step 1 matched a row: INSERT a fresh 'running' row for
           new_operation_id. Skipping on the idempotent path keeps at most
           one status='running' row per (conversation_id, lifecycle_epoch).
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.transaction():
            cur = conn.execute(
                """UPDATE compaction_operation
                      SET status = 'abandoned', completed_at = %s
                    WHERE operation_id = %s
                      AND conversation_id = %s
                      AND lifecycle_epoch = %s
                      AND status = 'running'""",
                (now, dead_operation_id, conversation_id, lifecycle_epoch),
            )
            fresh_takeover = (cur.rowcount or 0) > 0
            for table in (
                "segments", "facts", "tag_summaries", "tag_summary_embeddings",
            ):
                conn.execute(
                    f"DELETE FROM {table} "
                    f"WHERE operation_id = %s AND conversation_id = %s",
                    (dead_operation_id, conversation_id),
                )
            conn.execute(
                """UPDATE canonical_turns
                      SET compacted_at = NULL,
                          compaction_operation_id = NULL,
                          updated_at = %s
                    WHERE conversation_id = %s
                      AND compaction_operation_id = %s""",
                (_dt_to_str(now), conversation_id, dead_operation_id),
            )
            if fresh_takeover:
                conn.execute(
                    """INSERT INTO compaction_operation
                       (operation_id, conversation_id, lifecycle_epoch,
                        phase_index, phase_count, phase_name, status,
                        started_at, heartbeat_ts, owner_worker_id, created_at)
                       VALUES (%s, %s, %s, 0, %s, 'starting', 'running',
                               %s, %s, %s, %s)""",
                    (
                        new_operation_id, conversation_id, lifecycle_epoch,
                        phase_count, now, now, worker_id, now,
                    ),
                )
        return fresh_takeover

    def refresh_compaction_heartbeat(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        operation_id: str,
    ) -> bool:
        from datetime import datetime, timezone
        conn = self._get_conn()
        cur = conn.execute(
            """UPDATE compaction_operation
                  SET heartbeat_ts = %s
                WHERE operation_id = %s
                  AND conversation_id = %s
                  AND lifecycle_epoch = %s
                  AND owner_worker_id = %s
                  AND status = 'running'""",
            (
                datetime.now(timezone.utc),
                operation_id, conversation_id, lifecycle_epoch, worker_id,
            ),
        )
        return (cur.rowcount or 0) > 0

    def advance_compaction_phase(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_index: int,
        phase_name: str,
    ) -> bool:
        """Epoch-scoped phase advance. Also transitions status from
        ``'queued'`` to ``'running'`` (by unconditionally setting status
        to ``'running'``; the WHERE clause restricts the row to an
        already-active operation). Returns True iff the owner match
        and epoch match (both local + correlated subquery against
        ``conversations.lifecycle_epoch``) both hold.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE compaction_operation
               SET phase_index = %s, phase_name = %s, heartbeat_ts = %s,
                   status = 'running'
             WHERE conversation_id = %s AND status IN ('queued','running')
               AND lifecycle_epoch = %s
               AND lifecycle_epoch = (
                   SELECT lifecycle_epoch FROM conversations
                    WHERE conversation_id = %s
               )
               AND owner_worker_id = %s
            """,
            (
                phase_index, phase_name, now,
                conversation_id, lifecycle_epoch, conversation_id, worker_id,
            ),
        )
        return cur.rowcount == 1

    def complete_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        """Epoch-scoped completion. Returns True iff an active (queued
        or running) compaction exists, the caller owns it, and the
        caller's epoch equals the authoritative
        ``conversations.lifecycle_epoch`` (correlated subquery guard).
        Transitions status to ``'completed'`` and stamps
        ``completed_at``.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE compaction_operation
               SET status = 'completed', completed_at = %s
             WHERE conversation_id = %s AND status IN ('queued','running')
               AND lifecycle_epoch = %s
               AND lifecycle_epoch = (
                   SELECT lifecycle_epoch FROM conversations
                    WHERE conversation_id = %s
               )
               AND owner_worker_id = %s
            """,
            (
                now, conversation_id, lifecycle_epoch,
                conversation_id, worker_id,
            ),
        )
        return cur.rowcount == 1

    def fail_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        error_message: str,
    ) -> bool:
        """Epoch-scoped failure. Records ``error_message`` and stamps
        ``completed_at`` alongside the terminal ``'failed'`` status.
        Returns True iff the same ownership + epoch guards as
        ``complete_compaction_operation`` pass.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE compaction_operation
               SET status = 'failed', completed_at = %s,
                   error_message = %s
             WHERE conversation_id = %s AND status IN ('queued','running')
               AND lifecycle_epoch = %s
               AND lifecycle_epoch = (
                   SELECT lifecycle_epoch FROM conversations
                    WHERE conversation_id = %s
               )
               AND owner_worker_id = %s
            """,
            (
                now, error_message, conversation_id, lifecycle_epoch,
                conversation_id, worker_id,
            ),
        )
        return cur.rowcount == 1

    # ------------------------------------------------------------------
    # Request metadata + phase helpers (epoch-guarded)
    # ------------------------------------------------------------------
    # Postgres mirror of the SQLite helpers. All four methods filter on
    # ``lifecycle_epoch`` in the WHERE clause so a stale caller whose
    # in-memory epoch no longer matches the authoritative ``conversations``
    # row sees a ``False``/``None`` return and never stomps a new
    # lifecycle's counters/phase. Postgres uses scalar ``GREATEST()`` for
    # the monotonic widener (SQLite uses ``MAX()``).
    # ``set_phase_and_drain_pending_raw`` uses ``conn.transaction()``
    # (matches the A14 pattern) so the read-then-UPDATE is atomic under
    # autocommit; a concurrent resurrect cannot slip between the epoch
    # check and the drain.

    def update_request_metadata(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        last_raw_payload_entries: int,
        last_ingestible_payload_entries: int,
    ) -> bool:
        """Overwrite the per-request snapshot counters
        (``last_raw_payload_entries`` + ``last_ingestible_payload_entries``)
        on the conversations row. Epoch-guarded: returns ``True`` iff the
        UPDATE matched a row at the caller's ``lifecycle_epoch``.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE conversations
               SET last_raw_payload_entries = %s,
                   last_ingestible_payload_entries = %s,
                   updated_at = %s
             WHERE conversation_id = %s
               AND lifecycle_epoch = %s
            """,
            (
                last_raw_payload_entries,
                last_ingestible_payload_entries,
                now,
                conversation_id,
                lifecycle_epoch,
            ),
        )
        return cur.rowcount == 1

    def widen_pending_raw_payload_entries(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        value: int,
    ) -> bool:
        """Monotonic widener for ``pending_raw_payload_entries``. Uses
        Postgres's scalar ``GREATEST()`` so the column can only move
        forwards. Epoch-guarded: returns ``True`` iff the UPDATE matched
        a row at the caller's ``lifecycle_epoch``.
        """
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE conversations
               SET pending_raw_payload_entries =
                   GREATEST(pending_raw_payload_entries, %s)
             WHERE conversation_id = %s
               AND lifecycle_epoch = %s
            """,
            (value, conversation_id, lifecycle_epoch),
        )
        return cur.rowcount == 1

    def set_phase(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        phase: str,
    ) -> bool:
        """Epoch-guarded phase write. Returns ``True`` iff the UPDATE
        matched a row at the caller's epoch — a stale thread cannot
        stomp a new lifecycle's phase.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE conversations
               SET phase = %s, updated_at = %s
             WHERE conversation_id = %s
               AND lifecycle_epoch = %s
            """,
            (phase, now, conversation_id, lifecycle_epoch),
        )
        return cur.rowcount == 1

    def set_phase_and_drain_pending_raw(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        new_phase: str,
    ) -> int | None:
        """Atomically transition phase and return the drained
        ``pending_raw_payload_entries``. Wraps the read-then-UPDATE in
        ``conn.transaction()`` so a concurrent resurrect cannot slip
        between the epoch check and the drain. Returns the drained
        integer on success, or ``None`` when the caller's
        ``lifecycle_epoch`` does not match the authoritative
        conversations row.
        """
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.transaction():
            row = conn.execute(
                """
                SELECT pending_raw_payload_entries, lifecycle_epoch
                  FROM conversations WHERE conversation_id = %s
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            if isinstance(row, dict):
                current_epoch = int(row["lifecycle_epoch"])
                drained = int(row["pending_raw_payload_entries"])
            else:
                current_epoch = int(row[1])
                drained = int(row[0])
            if current_epoch != lifecycle_epoch:
                return None
            conn.execute(
                """
                UPDATE conversations
                   SET phase = %s,
                       pending_raw_payload_entries = 0,
                       updated_at = %s
                 WHERE conversation_id = %s
                   AND lifecycle_epoch = %s
                """,
                (new_phase, now, conversation_id, lifecycle_epoch),
            )
        return drained

    def drain_compaction_exit(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> str | None:
        """Atomic compaction-exit decision + pending drain.

        Postgres mirror of the SQLite helper. Wraps the epoch check, the
        ``EXISTS`` probe against ``canonical_turns``, the phase UPDATE, and
        (on untagged-exists) a fresh ``ingestion_episode`` INSERT in a
        single ``conn.transaction()`` so a concurrent tagger cannot flip
        the answer between read and write.

        Returns ``'ingesting'`` (work remains — episode row inserted) or
        ``'active'`` (all canonical rows tagged) on success, or ``None``
        when the caller's ``lifecycle_epoch`` does not match the
        authoritative conversations row. Epoch-guarded.
        """
        import uuid

        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.transaction():
            row = conn.execute(
                """
                SELECT pending_raw_payload_entries, lifecycle_epoch,
                       EXISTS (
                         SELECT 1 FROM canonical_turns
                          WHERE conversation_id = %s AND tagged_at IS NULL
                       )
                  FROM conversations
                 WHERE conversation_id = %s
                """,
                (conversation_id, conversation_id),
            ).fetchone()
            if row is None:
                return None
            if isinstance(row, dict):
                current_epoch = int(row["lifecycle_epoch"])
                pending_raw = int(row["pending_raw_payload_entries"])
                # When psycopg row factory returns dict rows, the
                # unnamed EXISTS expression is keyed by its auto-generated
                # alias "exists".
                has_untagged = bool(row.get("exists"))
            else:
                pending_raw = int(row[0])
                current_epoch = int(row[1])
                has_untagged = bool(row[2])
            if current_epoch != lifecycle_epoch:
                return None
            if has_untagged:
                conn.execute(
                    """
                    UPDATE conversations
                       SET phase = 'ingesting',
                           pending_raw_payload_entries = 0,
                           updated_at = %s
                     WHERE conversation_id = %s
                       AND lifecycle_epoch = %s
                    """,
                    (now, conversation_id, lifecycle_epoch),
                )
                conn.execute(
                    """
                    INSERT INTO ingestion_episode (
                        episode_id, conversation_id, lifecycle_epoch,
                        raw_payload_entries, started_at, status,
                        owner_worker_id, heartbeat_ts
                    ) VALUES (%s, %s, %s, %s, %s, 'running', %s, %s)
                    """,
                    (
                        uuid.uuid4(), conversation_id, lifecycle_epoch,
                        pending_raw, now, worker_id, now,
                    ),
                )
                return "ingesting"
            conn.execute(
                """
                UPDATE conversations
                   SET phase = 'active',
                       pending_raw_payload_entries = 0,
                       updated_at = %s
                 WHERE conversation_id = %s
                   AND lifecycle_epoch = %s
                """,
                (now, conversation_id, lifecycle_epoch),
            )
            return "active"

    def delete_conversation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        with conn.transaction():
            deleted = self._delete_conversation_rows(conn, "segments", conversation_id)
            for table in (
                "engine_state",
                "facts",
                "canonical_turns",
                "canonical_turn_chunks",
                "canonical_turn_anchors",
                "ingest_batches",
                "tag_summaries",
                "tag_aliases",
                "request_captures",
                "tool_outputs",
                "tool_calls",
                "request_context",
                "request_turn_counters",
                "tag_summary_embeddings",
                "turn_tool_outputs",
                "segment_tool_outputs",
                "chain_snapshots",
                "media_outputs",
                "ingestion_episode",
                "compaction_operation",
                "conversations",
            ):
                self._delete_conversation_rows(conn, table, conversation_id)

        # Disk cleanup: remove media files for this conversation
        import os
        import shutil
        _data_dir = os.environ.get("VC_DATA_DIR", "/data/tenants")
        media_dir = os.path.join(_data_dir, "media", conversation_id) if _data_dir else ""
        if media_dir and os.path.isdir(media_dir):
            shutil.rmtree(media_dir, ignore_errors=True)

        return deleted

    def save_tag_summary(self, tag_summary: TagSummary, conversation_id: str = "") -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tag_summaries
            (tag, conversation_id, summary, description, code_refs, summary_tokens, source_segment_refs, source_turn_numbers,
             source_canonical_turn_ids, covers_through_turn, covers_through_canonical_turn_id, generated_by_turn_id, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (tag, conversation_id) DO UPDATE SET
                summary=EXCLUDED.summary, description=EXCLUDED.description, code_refs=EXCLUDED.code_refs,
                summary_tokens=EXCLUDED.summary_tokens,
                source_segment_refs=EXCLUDED.source_segment_refs,
                source_turn_numbers=EXCLUDED.source_turn_numbers,
                source_canonical_turn_ids=EXCLUDED.source_canonical_turn_ids,
                covers_through_turn=EXCLUDED.covers_through_turn,
                covers_through_canonical_turn_id=EXCLUDED.covers_through_canonical_turn_id,
                generated_by_turn_id=EXCLUDED.generated_by_turn_id,
                updated_at=EXCLUDED.updated_at""",
            (tag_summary.tag, conversation_id, tag_summary.summary, getattr(tag_summary, "description", ""),
             json.dumps(getattr(tag_summary, "code_refs", []) or []),
             tag_summary.summary_tokens, json.dumps(tag_summary.source_segment_refs),
             json.dumps(tag_summary.source_turn_numbers),
             json.dumps(getattr(tag_summary, "source_canonical_turn_ids", []) or []),
             tag_summary.covers_through_turn,
             getattr(tag_summary, "covers_through_canonical_turn_id", "") or "",
             getattr(tag_summary, "generated_by_turn_id", "") or "",
             _dt_to_str(tag_summary.created_at), _dt_to_str(tag_summary.updated_at)),
        )

    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tag_summaries WHERE tag = %s AND conversation_id = %s", (tag, conversation_id)).fetchone()
        if not row:
            return None
        return TagSummary(
            tag=row["tag"], summary=row["summary"],
            description=row.get("description", ""),
            code_refs=json.loads(row.get("code_refs", "[]") or "[]"),
            summary_tokens=row["summary_tokens"],
            source_segment_refs=json.loads(row["source_segment_refs"]),
            source_turn_numbers=json.loads(row["source_turn_numbers"]),
            source_canonical_turn_ids=json.loads(row.get("source_canonical_turn_ids", "[]") or "[]"),
            covers_through_turn=row["covers_through_turn"],
            covers_through_canonical_turn_id=row.get("covers_through_canonical_turn_id", "") or "",
            generated_by_turn_id=row.get("generated_by_turn_id", ""),
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )

    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT * FROM tag_summaries WHERE conversation_id = %s ORDER BY updated_at DESC",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tag_summaries ORDER BY updated_at DESC").fetchall()
        return [
            TagSummary(
                tag=row["tag"], summary=row["summary"],
                description=row.get("description", ""),
                code_refs=json.loads(row.get("code_refs", "[]") or "[]"),
                summary_tokens=row["summary_tokens"],
                source_segment_refs=json.loads(row["source_segment_refs"]),
                source_turn_numbers=json.loads(row["source_turn_numbers"]),
                source_canonical_turn_ids=json.loads(row.get("source_canonical_turn_ids", "[]") or "[]"),
                covers_through_turn=row["covers_through_turn"],
                covers_through_canonical_turn_id=row.get("covers_through_canonical_turn_id", "") or "",
                generated_by_turn_id=row.get("generated_by_turn_id", ""),
                created_at=_str_to_dt(row["created_at"]),
                updated_at=_str_to_dt(row["updated_at"]),
            )
            for row in rows
        ]

    def get_segments_by_tags(self, tags: list[str], min_overlap: int = 1, limit: int = 20, conversation_id: str | None = None) -> list[StoredSegment]:
        if not tags:
            return []
        conn = self._get_conn()
        sql = """SELECT s.*, COUNT(st.tag) as overlap_count
            FROM segments s JOIN segment_tags st ON s.ref = st.segment_ref
            WHERE st.tag = ANY(%s)"""
        params: list = [tags]
        if conversation_id is not None:
            sql += " AND s.conversation_id = %s"
            params.append(conversation_id)
        sql += """ GROUP BY s.ref HAVING COUNT(st.tag) >= %s
            ORDER BY COUNT(st.tag) DESC, s.created_at DESC LIMIT %s"""
        params.extend([min_overlap, limit])
        rows = conn.execute(sql, params).fetchall()
        refs = [row["ref"] for row in rows]
        tags_map = self._batch_get_tags(refs)
        return [_row_to_segment(row, tags_map[row["ref"]]) for row in rows]

    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT st.tag, substring(s.summary from 1 for 100) as snippet
            FROM segment_tags st JOIN segments s ON s.ref = st.segment_ref
            WHERE st.tag NOT IN (SELECT tag FROM tag_summaries)
            GROUP BY st.tag, substring(s.summary from 1 for 100) LIMIT %s""",
            (limit,),
        ).fetchall()
        return [{"tag": row["tag"], "snippet": row["snippet"]} for row in rows]

    # ------------------------------------------------------------------
    # SearchStore
    # ------------------------------------------------------------------

    def search_full_text(self, query: str, limit: int = 5, conversation_id: str | None = None) -> list[QuoteResult]:
        conn = self._get_conn()
        _sc = self.search_config
        _pg_words = _sc.postgres_max_words if _sc else 100
        tsquery = " & ".join(query.split())
        try:
            sql = """SELECT s.ref, s.primary_tag, s.metadata_json,
                    ts_headline('english', s.full_text, to_tsquery('english', %s),
                                'StartSel=>>>, StopSel=<<<, MaxFragments=1, MaxWords=""" + str(_pg_words) + """') as excerpt,
                    s.created_at
                FROM segments s
                WHERE s.full_text_tsv @@ to_tsquery('english', %s)"""
            params: list = [tsquery, tsquery]
            if conversation_id is not None:
                sql += " AND s.conversation_id = %s"
                params.append(conversation_id)
            sql += """
                ORDER BY ts_rank(s.full_text_tsv, to_tsquery('english', %s)) DESC
                LIMIT %s"""
            params.extend([tsquery, limit])
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # Fallback to ILIKE
            _excerpt_chars = _sc.excerpt_context_chars if _sc else 200
            sql = "SELECT ref, primary_tag, full_text, metadata_json, created_at FROM segments WHERE full_text ILIKE %s"
            params = [f"%{query}%"]
            if conversation_id is not None:
                sql += " AND conversation_id = %s"
                params.append(conversation_id)
            sql += " LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                meta = json.loads(row["metadata_json"])
                results.append(QuoteResult(
                    text=_extract_excerpt(row["full_text"], query, context_chars=_excerpt_chars),
                    tag=row["primary_tag"], segment_ref=row["ref"],
                    session_date=meta.get("session_date", ""),
                    match_type="text_search",
                ))
            return results

        results = []
        for row in rows:
            meta = json.loads(row["metadata_json"])
            results.append(QuoteResult(
                text=row["excerpt"], tag=row["primary_tag"],
                segment_ref=row["ref"],
                session_date=meta.get("session_date", ""),
                match_type="fts",
            ))
        return results

    def search_canonical_turn_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        conn = self._get_conn()
        pattern = f"%{query}%"
        sql = """SELECT canonical_turn_id, turn_number, user_content, assistant_content, created_at,
                        primary_tag, tags_json, session_date
                 FROM canonical_turns_ordinal
                 WHERE (user_content ILIKE %s OR assistant_content ILIKE %s)"""
        params: list[object] = [pattern, pattern]
        if conversation_id is not None:
            sql += " AND conversation_id = %s"
            params.append(conversation_id)
        sql += " ORDER BY sort_key DESC LIMIT %s"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

        results = []
        _sc = getattr(self, "search_config", None)
        _ctx = _sc.excerpt_context_chars if _sc else 200
        for row in rows:
            turn = row["turn_number"]
            u = row["user_content"] or ""
            a = row["assistant_content"] or ""
            primary_tag = row.get("primary_tag", "_general") or "_general"
            try:
                tags = json.loads(row.get("tags_json", "[]") or "[]")
            except Exception:
                tags = []
            session_date = row.get("session_date", "") or ""
            matched_side = _matched_turn_side(query, u, a)
            excerpt = _build_turn_excerpt(
                query,
                u,
                a,
                matched_side,
                context_chars=_ctx,
            )
            results.append(QuoteResult(
                text=excerpt,
                tag=primary_tag,
                segment_ref=f"canonical_turn_{row.get('canonical_turn_id', '') or turn}",
                tags=list(tags or []),
                match_type="full_text_search",
                session_date=session_date,
                source_scope="turn",
                turn_number=turn,
                matched_side=matched_side,
            ))
        return results

    def store_chunk_embeddings(self, segment_ref: str, chunks: list[ChunkEmbedding]) -> None:
        conn = self._get_conn()
        with conn.transaction():
            conn.execute("DELETE FROM segment_chunks WHERE segment_ref = %s", (segment_ref,))
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO segment_chunks (segment_ref, chunk_index, text, embedding_json) VALUES (%s,%s,%s,%s)",
                    (segment_ref, chunk.chunk_index, chunk.text, json.dumps(chunk.embedding)),
                )

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT segment_ref, chunk_index, text, embedding_json FROM segment_chunks ORDER BY segment_ref, chunk_index"
        ).fetchall()
        return [
            ChunkEmbedding(
                segment_ref=row["segment_ref"], chunk_index=row["chunk_index"],
                text=row["text"], embedding=json.loads(row["embedding_json"]),
            )
            for row in rows
        ]

    def store_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int,
        side: str,
        chunks: list[CanonicalTurnChunkEmbedding],
        canonical_turn_id: str | None = None,
    ) -> None:
        conn = self._get_conn()
        canonical_turn_id = canonical_turn_id or self._lookup_canonical_turn_id_for_ordinal(conversation_id, turn_number)
        if canonical_turn_id is None:
            return
        with conn.transaction():
            conn.execute(
                "DELETE FROM canonical_turn_chunks WHERE conversation_id = %s AND canonical_turn_id = %s AND side = %s",
                (conversation_id, canonical_turn_id, side),
            )
            for chunk in chunks:
                conn.execute(
                    """INSERT INTO canonical_turn_chunks
                    (conversation_id, canonical_turn_id, side, chunk_index, text, embedding_json)
                    VALUES (%s,%s,%s,%s,%s,%s)""",
                    (
                        chunk.conversation_id,
                        canonical_turn_id,
                        chunk.side,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(chunk.embedding),
                    ),
                )

    def get_all_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str | None = None,
    ) -> list[CanonicalTurnChunkEmbedding]:
        conn = self._get_conn()
        if conversation_id is None:
            rows = conn.execute(
                """SELECT ctc.conversation_id, ctc.canonical_turn_id, cto.turn_number, ctc.side, ctc.chunk_index, ctc.text, ctc.embedding_json
                FROM canonical_turn_chunks ctc
                JOIN canonical_turns_ordinal cto
                  ON cto.conversation_id = ctc.conversation_id
                 AND cto.canonical_turn_id = ctc.canonical_turn_id
                ORDER BY ctc.conversation_id, cto.turn_number, ctc.side, ctc.chunk_index"""
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT ctc.conversation_id, ctc.canonical_turn_id, cto.turn_number, ctc.side, ctc.chunk_index, ctc.text, ctc.embedding_json
                FROM canonical_turn_chunks ctc
                JOIN canonical_turns_ordinal cto
                  ON cto.conversation_id = ctc.conversation_id
                 AND cto.canonical_turn_id = ctc.canonical_turn_id
                WHERE ctc.conversation_id = %s
                ORDER BY cto.turn_number, ctc.side, ctc.chunk_index""",
                (conversation_id,),
            ).fetchall()
        return [
            CanonicalTurnChunkEmbedding(
                conversation_id=row["conversation_id"],
                canonical_turn_id=str(row.get("canonical_turn_id", "") or ""),
                turn_number=row["turn_number"],
                side=row["side"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                embedding=json.loads(row["embedding_json"]),
            )
            for row in rows
        ]

    def delete_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int | None = None,
        canonical_turn_id: str | None = None,
    ) -> int:
        conn = self._get_conn()
        if canonical_turn_id is None and turn_number is not None:
            canonical_turn_id = self._lookup_canonical_turn_id_for_ordinal(conversation_id, turn_number)
        if turn_number is None and canonical_turn_id is None:
            cur = conn.execute(
                "DELETE FROM canonical_turn_chunks WHERE conversation_id = %s",
                (conversation_id,),
            )
        else:
            if canonical_turn_id is None:
                return 0
            cur = conn.execute(
                "DELETE FROM canonical_turn_chunks WHERE conversation_id = %s AND canonical_turn_id = %s",
                (conversation_id, canonical_turn_id),
            )
        return int(cur.rowcount or 0)

    def store_tool_output(self, ref: str, conversation_id: str, tool_name: str, command: str, turn: int, content: str, original_bytes: int) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tool_outputs (ref, conversation_id, tool_name, command, turn, content, original_bytes, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ref) DO UPDATE SET content=EXCLUDED.content, original_bytes=EXCLUDED.original_bytes""",
            (ref, conversation_id, tool_name, command, turn, content, original_bytes, _dt_to_str(datetime.now(timezone.utc))),
        )

    def search_tool_outputs(self, query: str, limit: int = 5, conversation_id: str | None = None) -> list:
        conn = self._get_conn()
        tsquery = " & ".join(query.split())
        try:
            sql = """SELECT t.ref, t.tool_name,
                    ts_headline('english', t.content, to_tsquery('english', %s),
                                'StartSel=>>>, StopSel=<<<, MaxFragments=1, MaxWords=20') as excerpt
                FROM tool_outputs t
                WHERE t.content_tsv @@ to_tsquery('english', %s)"""
            params: list = [tsquery, tsquery]
            if conversation_id is not None:
                sql += " AND t.conversation_id = %s"
                params.append(conversation_id)
            sql += """
                ORDER BY ts_rank(t.content_tsv, to_tsquery('english', %s)) DESC
                LIMIT %s"""
            params.extend([tsquery, limit])
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []
        return [
            QuoteResult(
                text=row["excerpt"],
                tag=row["tool_name"],
                segment_ref=row["ref"],
                session_date="",
                match_type="tool_output",
                source_scope="tool_output",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Turn / Segment ↔ Tool Output linkage
    # ------------------------------------------------------------------

    def link_turn_tool_output(self, conversation_id: str, turn_number: int, tool_output_ref: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO turn_tool_outputs (conversation_id, turn_number, tool_output_ref)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING""",
            (conversation_id, turn_number, tool_output_ref),
        )

    def get_tool_outputs_for_turn(self, conversation_id: str, turn_number: int) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tool_output_ref FROM turn_tool_outputs WHERE conversation_id = %s AND turn_number = %s",
            (conversation_id, turn_number),
        ).fetchall()
        return [row["tool_output_ref"] for row in rows]

    def link_segment_tool_output(self, conversation_id: str, segment_ref: str, tool_output_ref: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO segment_tool_outputs (conversation_id, segment_ref, tool_output_ref)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING""",
            (conversation_id, segment_ref, tool_output_ref),
        )

    def get_tool_outputs_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tool_output_ref FROM segment_tool_outputs WHERE conversation_id = %s AND segment_ref = %s",
            (conversation_id, segment_ref),
        ).fetchall()
        return [row["tool_output_ref"] for row in rows]

    def get_tool_output_refs_for_turn(self, conversation_id: str, turn: int) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT ref FROM tool_outputs WHERE conversation_id = %s AND turn = %s",
            (conversation_id, turn),
        ).fetchall()
        return [row["ref"] for row in rows]

    def get_tool_output_by_ref(self, conversation_id: str, ref: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT content FROM tool_outputs WHERE conversation_id = %s AND ref = %s",
            (conversation_id, ref),
        ).fetchone()
        return row["content"] if row else None

    # ------------------------------------------------------------------
    # Media Output Storage
    # ------------------------------------------------------------------

    def store_media_output(
        self,
        ref: str,
        conversation_id: str,
        media_type: str,
        width: int,
        height: int,
        original_bytes: int,
        compressed_bytes: int,
        file_path: str,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO media_outputs (ref, conversation_id, media_type, width, height, original_bytes, compressed_bytes, file_path, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (conversation_id, ref) DO UPDATE SET
                media_type=EXCLUDED.media_type, width=EXCLUDED.width, height=EXCLUDED.height,
                original_bytes=EXCLUDED.original_bytes, compressed_bytes=EXCLUDED.compressed_bytes,
                file_path=EXCLUDED.file_path""",
            (ref, conversation_id, media_type, width, height, original_bytes, compressed_bytes, file_path, _dt_to_str(datetime.now(timezone.utc))),
        )

    def get_media_output(self, conversation_id: str, ref: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT ref, conversation_id, media_type, width, height, original_bytes, compressed_bytes, file_path FROM media_outputs WHERE conversation_id = %s AND ref = %s",
            (conversation_id, ref),
        ).fetchone()
        if not row:
            return None
        return {
            "ref": row["ref"],
            "conversation_id": row["conversation_id"],
            "media_type": row["media_type"],
            "width": row["width"],
            "height": row["height"],
            "original_bytes": row["original_bytes"],
            "compressed_bytes": row["compressed_bytes"],
            "file_path": row["file_path"],
        }

    # ------------------------------------------------------------------
    # Chain Snapshots (turn chain collapse)
    # ------------------------------------------------------------------

    def store_chain_snapshot(
        self,
        ref: str,
        conversation_id: str,
        turn_number: int,
        chain_json: str,
        message_count: int,
        tool_output_refs: str = "",
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO chain_snapshots
            (ref, conversation_id, turn_number, chain_json, message_count, tool_output_refs, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ref) DO UPDATE SET
                chain_json=EXCLUDED.chain_json, message_count=EXCLUDED.message_count,
                tool_output_refs=EXCLUDED.tool_output_refs""",
            (ref, conversation_id, turn_number, chain_json, message_count, tool_output_refs,
             _dt_to_str(datetime.now(timezone.utc))),
        )

    def get_chain_snapshot(self, conversation_id: str, ref: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT ref, conversation_id, turn_number, chain_json, message_count, tool_output_refs
            FROM chain_snapshots WHERE conversation_id = %s AND ref = %s""",
            (conversation_id, ref),
        ).fetchone()
        if not row:
            return None
        return {
            "ref": row["ref"],
            "conversation_id": row["conversation_id"],
            "turn_number": row["turn_number"],
            "chain_json": row["chain_json"],
            "message_count": row["message_count"],
            "tool_output_refs": row["tool_output_refs"],
        }

    def get_chain_snapshots_for_conversation(self, conversation_id: str, min_turn: int = 0) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ref, turn_number, tool_output_refs, message_count
            FROM chain_snapshots WHERE conversation_id = %s AND turn_number >= %s
            ORDER BY turn_number""",
            (conversation_id, min_turn),
        ).fetchall()
        return [{"ref": r["ref"], "turn_number": r["turn_number"],
                 "tool_output_refs": r["tool_output_refs"], "message_count": r["message_count"]} for r in rows]

    def get_chain_recovery_manifest(self, conversation_id: str, min_turn: int = 0) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT
                s.ref,
                s.turn_number,
                s.tool_output_refs,
                s.message_count,
                COALESCE(
                    STRING_AGG(DISTINCT t.tool_name, ', ' ORDER BY t.tool_name)
                        FILTER (WHERE t.tool_name != ''),
                    ''
                ) AS tool_names
            FROM chain_snapshots s
            LEFT JOIN tool_outputs t
                ON t.ref = ANY(string_to_array(NULLIF(s.tool_output_refs, ''), ','))
            WHERE s.conversation_id = %s
              AND s.turn_number >= %s
            GROUP BY s.ref, s.turn_number, s.tool_output_refs, s.message_count
            ORDER BY s.turn_number
            """,
            (conversation_id, min_turn),
        ).fetchall()
        return [
            {
                "ref": row["ref"],
                "turn_number": row["turn_number"],
                "tool_output_refs": row["tool_output_refs"],
                "message_count": row["message_count"],
                "tool_names": row["tool_names"] or "",
            }
            for row in rows
        ]

    def get_tool_names_for_refs(self, refs: list[str]) -> list[str]:
        if not refs:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT tool_name FROM tool_outputs WHERE ref = ANY(%s) AND tool_name != ''",
            (refs,),
        ).fetchall()
        return [row["tool_name"] for row in rows]

    def get_tool_names_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT DISTINCT t.tool_name
            FROM segment_tool_outputs s
            JOIN tool_outputs t ON t.ref = s.tool_output_ref AND t.conversation_id = s.conversation_id
            WHERE s.conversation_id = %s AND s.segment_ref = %s
            ORDER BY t.tool_name""",
            (conversation_id, segment_ref),
        ).fetchall()
        return [row["tool_name"] for row in rows]

    def save_request_capture(self, capture: dict) -> None:
        conn = self._get_conn()
        import time as _time
        conversation_id = capture.get("conversation_id", "") or ""
        turn_id = capture.get("turn_id", "") or ""
        conn.execute(
            """INSERT INTO request_captures
            (conversation_id, turn, turn_id, ts, recorded_at, data_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (conversation_id, turn, turn_id) DO UPDATE SET
                ts=EXCLUDED.ts,
                recorded_at=EXCLUDED.recorded_at,
                data_json=EXCLUDED.data_json""",
            (
                conversation_id,
                capture["turn"],
                turn_id,
                capture.get("ts", ""),
                _time.time(),
                json.dumps(capture),
            ),
        )
        conn.execute(
            """DELETE FROM request_captures
            WHERE conversation_id = %s
              AND (conversation_id, turn, turn_id) NOT IN (
                SELECT conversation_id, turn, turn_id
                FROM request_captures
                WHERE conversation_id = %s
                ORDER BY recorded_at DESC LIMIT 50
            )""",
            (conversation_id, conversation_id),
        )

    def load_request_captures(
        self,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[dict]:
        conn = self._get_conn()
        try:
            if conversation_id is None:
                rows = conn.execute(
                    "SELECT data_json FROM request_captures ORDER BY recorded_at ASC LIMIT %s",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT data_json FROM request_captures
                    WHERE conversation_id = %s
                    ORDER BY recorded_at ASC LIMIT %s""",
                    (conversation_id, limit),
                ).fetchall()
        except Exception:
            return []
        result = []
        for row in rows:
            try:
                result.append(json.loads(row["data_json"]))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        return result

    # ------------------------------------------------------------------
    # StateStore
    # ------------------------------------------------------------------

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        conn = self._get_conn()
        entries_data = {
            "entries": [
                {
                    "turn_number": e.turn_number,
                    "canonical_turn_id": getattr(e, "canonical_turn_id", "") or "",
                    "tags": e.tags,
                    "primary_tag": e.primary_tag,
                    "message_hash": e.message_hash,
                    "sender": e.sender,
                    "fact_signals": [
                        {"subject": fs.subject, "verb": fs.verb, "object": fs.object,
                         "status": fs.status, "fact_type": fs.fact_type, "what": fs.what}
                        for fs in (e.fact_signals or [])
                    ] if e.fact_signals else [],
                    "code_refs": list(getattr(e, "code_refs", []) or []),
                }
                for e in state.turn_tag_entries
            ],
            "split_processed_tags": list(state.split_processed_tags) if state.split_processed_tags else [],
            "working_set": [
                {
                    "tag": ws.tag,
                    "depth": ws.depth.value if hasattr(ws.depth, 'value') else ws.depth,
                    "tokens": ws.tokens,
                    "last_accessed_turn": ws.last_accessed_turn,
                }
                for ws in (state.working_set or [])
            ],
            "trailing_fingerprint": state.trailing_fingerprint or "",
            "request_captures": state.request_captures,
            "provider": state.provider,
            "flushed_prefix_messages": state.flushed_prefix_messages,
            "last_request_time": state.last_request_time,
            "tool_tag_counter": state.tool_tag_counter,
            "last_compacted_turn": state.last_compacted_turn,
            "last_completed_turn": state.last_completed_turn,
            "last_indexed_turn": state.last_indexed_turn,
            "checkpoint_version": state.checkpoint_version,
        }
        conn.execute(
            """INSERT INTO engine_state (conversation_id, compacted_prefix_messages, turn_count, turn_tag_entries, saved_at, flushed_prefix_messages, last_request_time)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                compacted_prefix_messages=EXCLUDED.compacted_prefix_messages, turn_count=EXCLUDED.turn_count,
                turn_tag_entries=EXCLUDED.turn_tag_entries, saved_at=EXCLUDED.saved_at,
                flushed_prefix_messages=EXCLUDED.flushed_prefix_messages, last_request_time=EXCLUDED.last_request_time""",
            (state.conversation_id, state.compacted_prefix_messages, state.turn_count,
             json.dumps(entries_data), _dt_to_str(state.saved_at),
             state.flushed_prefix_messages, state.last_request_time),
        )

    def _parse_engine_state_row(self, row: dict) -> EngineStateSnapshot:
        raw = json.loads(row["turn_tag_entries"])
        if isinstance(raw, list):
            entries_list = raw
            split_tags: set[str] = set()
            working_set: dict = {}
            fingerprint = ""
            request_captures = []
            provider = ""
            flushed_prefix_messages = row.get("flushed_prefix_messages") or 0
            flushed_prefix_messages_present = bool(row.get("flushed_prefix_messages"))
            last_request_time = row.get("last_request_time") or 0.0
            tool_tag_counter = 0
            last_compacted_turn = (row["compacted_prefix_messages"] // 2) - 1 if row["compacted_prefix_messages"] > 0 else -1
            last_completed_turn = max(row["turn_count"] - 1, len(entries_list) - 1)
            last_indexed_turn = len(entries_list) - 1
            checkpoint_version = 0
        elif isinstance(raw, dict):
            entries_list = raw.get("entries", raw.get("turn_tag_entries", []))
            split_tags = set(raw.get("split_processed_tags", []))
            ws_raw = raw.get("working_set", [])
            if isinstance(ws_raw, dict):
                # Legacy format: convert dict to list
                working_set = [
                    WorkingSetEntry(
                        tag=tag,
                        depth=DepthLevel(ws["depth"]),
                        tokens=ws.get("tokens", 0),
                        last_accessed_turn=ws.get("last_accessed_turn", 0),
                    )
                    for tag, ws in ws_raw.items()
                ]
            else:
                working_set = [
                    WorkingSetEntry(
                        tag=ws["tag"],
                        depth=DepthLevel(ws["depth"]),
                        tokens=ws.get("tokens", 0),
                        last_accessed_turn=ws.get("last_accessed_turn", 0),
                    )
                    for ws in ws_raw
                ]
            fingerprint = raw.get("trailing_fingerprint", "")
            request_captures = raw.get("request_captures", [])
            provider = raw.get("provider", "")
            flushed_prefix_messages = raw.get("flushed_prefix_messages", 0)
            flushed_prefix_messages_present = "flushed_prefix_messages" in raw
            last_request_time = raw.get("last_request_time", 0.0)
            tool_tag_counter = raw.get("tool_tag_counter", 0)
            last_compacted_turn = raw.get(
                "last_compacted_turn",
                (row["compacted_prefix_messages"] // 2) - 1 if row["compacted_prefix_messages"] > 0 else -1,
            )
            last_completed_turn = raw.get(
                "last_completed_turn",
                max(row["turn_count"] - 1, len(entries_list) - 1),
            )
            last_indexed_turn = raw.get("last_indexed_turn", len(entries_list) - 1)
            checkpoint_version = raw.get("checkpoint_version", 0)
        else:
            entries_list = []
            split_tags = set()
            working_set = {}
            fingerprint = ""
            request_captures = []
            provider = ""
            flushed_prefix_messages = row.get("flushed_prefix_messages") or 0
            flushed_prefix_messages_present = bool(row.get("flushed_prefix_messages"))
            last_request_time = row.get("last_request_time") or 0.0
            tool_tag_counter = 0
            last_compacted_turn = (row["compacted_prefix_messages"] // 2) - 1 if row["compacted_prefix_messages"] > 0 else -1
            last_completed_turn = row["turn_count"] - 1
            last_indexed_turn = -1
            checkpoint_version = 0

        entries = []
        for e in entries_list:
            signals = []
            for fs in e.get("fact_signals") or []:
                signals.append(FactSignal(
                    subject=fs.get("subject", ""), verb=fs.get("verb", ""),
                    object=fs.get("object", ""), status=fs.get("status", ""),
                    fact_type=fs.get("fact_type", "personal"), what=fs.get("what", ""),
                ))
            # ``TurnTagEntry`` declares ``fact_signals`` / ``tags`` / ``code_refs``
            # with ``default_factory=list``. Preserve that contract — ``None``
            # would make ``list(entry.fact_signals)`` raise ``TypeError`` in
            # ``persist_completed_turn``. (Previous code stored ``None`` when
            # the list was empty, which crashed ingestion on resume.)
            tags_list = list(e.get("tags") or [])
            entries.append(TurnTagEntry(
                turn_number=e["turn_number"], tags=tags_list,
                canonical_turn_id=e.get("canonical_turn_id", "") or "",
                primary_tag=e.get("primary_tag") or (tags_list[0] if tags_list else "_general"),
                message_hash=e.get("message_hash", ""),
                fact_signals=signals,
                code_refs=list(e.get("code_refs") or []),
                sender=e.get("sender", ""),
            ))

        return EngineStateSnapshot(
            conversation_id=row["conversation_id"],
            compacted_prefix_messages=row["compacted_prefix_messages"],
            flushed_prefix_messages=flushed_prefix_messages,
            flushed_prefix_messages_present=flushed_prefix_messages_present,
            last_request_time=last_request_time,
            turn_count=row["turn_count"],
            turn_tag_entries=entries,
            last_compacted_turn=last_compacted_turn,
            last_completed_turn=last_completed_turn,
            last_indexed_turn=last_indexed_turn,
            checkpoint_version=checkpoint_version,
            conversation_generation=self.get_conversation_generation(row["conversation_id"]),
            split_processed_tags=split_tags,
            working_set=working_set,
            trailing_fingerprint=fingerprint,
            request_captures=request_captures,
            provider=provider,
            tool_tag_counter=tool_tag_counter,
        )

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM engine_state WHERE conversation_id = %s", (conversation_id,)).fetchone()
        if not row:
            return None
        return self._parse_engine_state_row(row)

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM engine_state ORDER BY compacted_prefix_messages DESC, saved_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        return self._parse_engine_state_row(row)

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT conversation_id, turn_tag_entries FROM engine_state").fetchall()
        result: dict[str, str] = {}
        for row in rows:
            raw = json.loads(row["turn_tag_entries"])
            fp = raw.get("trailing_fingerprint", "") if isinstance(raw, dict) else ""
            if fp:
                result[fp] = row["conversation_id"]
        return result

    # ------------------------------------------------------------------
    def save_conversation_alias(self, alias_id: str, target_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO conversation_aliases (alias_id, target_id)
            VALUES (%s, %s)
            ON CONFLICT (alias_id) DO UPDATE SET target_id = EXCLUDED.target_id""",
            (alias_id, target_id),
        )

    def resolve_conversation_alias(self, alias_id: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT target_id FROM conversation_aliases WHERE alias_id = %s",
            (alias_id,),
        ).fetchone()
        return row["target_id"] if row else None

    def delete_conversation_alias(self, alias_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM conversation_aliases WHERE alias_id = %s",
            (alias_id,),
        )

    def save_canonical_turn(
        self,
        conversation_id: str,
        turn_number: int,
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
        created_at: str | None = None,
        updated_at: str | None = None,
        canonical_turn_id: str | None = None,
        sort_key: float | None = None,
        turn_hash: str = "",
        hash_version: int = 0,
        normalized_user_text: str = "",
        normalized_assistant_text: str = "",
        tagged_at: str | None = None,
        compacted_at: str | None = None,
        first_seen_at: str | None = None,
        last_seen_at: str | None = None,
        source_batch_id: str | None = None,
        turn_group_number: int = -1,
    ) -> None:
        now = _dt_to_str(datetime.now(timezone.utc))
        created = created_at or now
        updated = updated_at or now
        first_seen = first_seen_at or created
        last_seen = last_seen_at or updated
        if not turn_hash:
            turn_hash, normalized_user_text, normalized_assistant_text = compute_turn_hash_from_raw(
                user_content,
                assistant_content,
                version=hash_version or HASH_VERSION,
            )
        if not hash_version:
            hash_version = HASH_VERSION
        fact_signal_payload = [
            {
                "subject": fs.subject,
                "verb": fs.verb,
                "object": fs.object,
                "status": fs.status,
                "fact_type": getattr(fs, "fact_type", ""),
                "what": getattr(fs, "what", ""),
            }
            for fs in (fact_signals or [])
        ]
        conn = self._get_conn()
        if canonical_turn_id is None and turn_number >= 0:
            slot_row = conn.execute(
                "SELECT canonical_turn_id FROM canonical_turns WHERE conversation_id = %s AND sort_key = %s",
                (conversation_id, float((turn_number + 1) * 1000.0)),
            ).fetchone()
            canonical_turn_id = str(slot_row["canonical_turn_id"]) if slot_row else None
        existing_sort_key = None
        if canonical_turn_id:
            existing = conn.execute(
                "SELECT sort_key FROM canonical_turns WHERE conversation_id = %s AND canonical_turn_id = %s",
                (conversation_id, canonical_turn_id),
            ).fetchone()
            if existing:
                existing_sort_key = float(existing["sort_key"])
        if canonical_turn_id is None:
            canonical_turn_id = generate_canonical_turn_id()
        if sort_key is None:
            if existing_sort_key is not None:
                sort_key = existing_sort_key
            elif turn_number >= 0:
                sort_key = float((turn_number + 1) * 1000.0)
            else:
                tail = conn.execute(
                    "SELECT COALESCE(MAX(sort_key), 0) AS max_sort_key FROM canonical_turns WHERE conversation_id = %s",
                    (conversation_id,),
                ).fetchone()
                sort_key = float((tail["max_sort_key"] or 0.0) + 1000.0)
        conn.execute(
            """INSERT INTO canonical_turns
            (canonical_turn_id, conversation_id, turn_group_number, sort_key, turn_hash, hash_version,
             normalized_user_text, normalized_assistant_text, user_content, assistant_content,
             user_raw_content, assistant_raw_content, primary_tag, tags_json, session_date, sender,
             fact_signals_json, code_refs_json, tagged_at, compacted_at, first_seen_at, last_seen_at,
             source_batch_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (canonical_turn_id) DO UPDATE SET
                turn_group_number=EXCLUDED.turn_group_number,
                sort_key=EXCLUDED.sort_key,
                turn_hash=EXCLUDED.turn_hash,
                hash_version=EXCLUDED.hash_version,
                normalized_user_text=EXCLUDED.normalized_user_text,
                normalized_assistant_text=EXCLUDED.normalized_assistant_text,
                user_content=EXCLUDED.user_content,
                assistant_content=EXCLUDED.assistant_content,
                user_raw_content=EXCLUDED.user_raw_content,
                assistant_raw_content=EXCLUDED.assistant_raw_content,
                primary_tag=EXCLUDED.primary_tag,
                tags_json=EXCLUDED.tags_json,
                session_date=EXCLUDED.session_date,
                sender=EXCLUDED.sender,
                fact_signals_json=EXCLUDED.fact_signals_json,
                code_refs_json=EXCLUDED.code_refs_json,
                tagged_at=EXCLUDED.tagged_at,
                compacted_at=EXCLUDED.compacted_at,
                last_seen_at=EXCLUDED.last_seen_at,
                source_batch_id=EXCLUDED.source_batch_id,
                updated_at=EXCLUDED.updated_at""",
            (
                canonical_turn_id,
                conversation_id,
                int(turn_group_number),
                sort_key,
                turn_hash,
                hash_version,
                normalized_user_text,
                normalized_assistant_text,
                user_content,
                assistant_content,
                user_raw_content,
                assistant_raw_content,
                primary_tag or "_general",
                json.dumps(list(tags or [])),
                session_date or "",
                sender or "",
                json.dumps(fact_signal_payload),
                json.dumps(list(code_refs or [])),
                tagged_at,
                compacted_at,
                first_seen,
                last_seen,
                source_batch_id,
                created,
                updated,
            ),
        )

    def recompute_canonical_turn_groups(
        self,
        conversation_id: str,
    ) -> int:
        # Use the raw loader to avoid recursion: the non-raw loader triggers
        # this method on legacy (all -1) conversations.
        rows = self._load_canonical_turn_rows_raw(conversation_id)
        if not rows:
            return 0

        assignments: list[tuple[int, str]] = []
        current_group = -1
        pending_user_group = -1
        for row in rows:
            has_user = bool(row.user_content)
            has_assistant = bool(row.assistant_content)
            if has_user and has_assistant:
                current_group += 1
                pending_user_group = -1
                assignments.append((current_group, row.canonical_turn_id))
                continue
            if has_user:
                current_group += 1
                pending_user_group = current_group
                assignments.append((current_group, row.canonical_turn_id))
                continue
            if has_assistant:
                if pending_user_group >= 0:
                    assignments.append((pending_user_group, row.canonical_turn_id))
                    pending_user_group = -1
                else:
                    current_group += 1
                    assignments.append((current_group, row.canonical_turn_id))
                continue
            current_group += 1
            pending_user_group = -1
            assignments.append((current_group, row.canonical_turn_id))

        conn = self._get_conn()
        changed = 0
        for turn_group_number, canonical_turn_id in assignments:
            cursor = conn.execute(
                """UPDATE canonical_turns
                   SET turn_group_number = %s
                   WHERE conversation_id = %s
                     AND canonical_turn_id = %s
                     AND turn_group_number <> %s""",
                (turn_group_number, conversation_id, canonical_turn_id, turn_group_number),
            )
            changed += int(cursor.rowcount or 0)
        return changed

    def get_canonical_turn_rows(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, CanonicalTurnRow]:
        if not turn_numbers:
            return {}
        merged_rows = _merge_canonical_turn_rows(self._load_canonical_turn_rows(conversation_id))
        return {
            turn_number: merged_rows[turn_number]
            for turn_number in turn_numbers
            if turn_number in merged_rows
        }

    def get_all_canonical_turns(
        self,
        conversation_id: str,
    ) -> list[CanonicalTurnRow]:
        return self._load_canonical_turn_rows(conversation_id)

    def get_uncompacted_canonical_turns(
        self,
        conversation_id: str,
        *,
        protected_recent_turns: int = 0,
    ) -> list[CanonicalTurnRow]:
        merged_rows = list(_merge_canonical_turn_rows(self._load_canonical_turn_rows(conversation_id)).values())
        uncompacted = [row for row in merged_rows if not row.compacted_at]
        if protected_recent_turns > 0 and len(uncompacted) > protected_recent_turns:
            uncompacted = uncompacted[:-protected_recent_turns]
        elif protected_recent_turns > 0:
            uncompacted = []
        return uncompacted

    def mark_canonical_turns_tagged(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        tagged_at: str | None = None,
    ) -> int:
        if not canonical_turn_ids:
            return 0
        conn = self._get_conn()
        timestamp = tagged_at or _dt_to_str(datetime.now(timezone.utc))
        rows = conn.execute(
            """UPDATE canonical_turns
               SET tagged_at = %s, updated_at = %s
               WHERE conversation_id = %s AND canonical_turn_id = ANY(%s)""",
            (timestamp, timestamp, conversation_id, canonical_turn_ids),
        )
        return int(rows.rowcount or 0)

    def iter_untagged_canonical_rows(
        self,
        *,
        conversation_id: str,
        expected_lifecycle_epoch: int,
        batch_size: int = 32,
    ) -> list[CanonicalTurnRow]:
        """Epoch-guarded. Stale-epoch caller receives an empty list.

        Query runs against the ``canonical_turns`` base table — NOT the
        ``canonical_turns_ordinal`` view — so the partial index
        ``idx_canonical_turns_conv_untagged ON canonical_turns
        (conversation_id, sort_key) WHERE tagged_at IS NULL`` can drive the
        index scan. Routing through the view forces a full scan plus sort,
        which is unacceptable on the tagger hot path. ``turn_number`` is a
        view-only computed column; omitting it here is tolerated by the
        shared ``_row_to_canonical_turn`` parser (defaults to ``-1``). We
        JOIN against ``conversations.lifecycle_epoch`` so a stale-epoch
        caller matches zero rows at SQL level rather than raising.
        ``conversations.conversation_id`` and ``canonical_turns.conversation_id``
        are both TEXT, so the equality join uses the column directly and
        lets the partial index on ``canonical_turns`` drive the scan.
        Ordering uses ``sort_key`` — the same column the partial index is
        sorted on.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT ct.canonical_turn_id, ct.conversation_id, ct.turn_group_number,
                   ct.sort_key, ct.turn_hash, ct.hash_version,
                   ct.normalized_user_text, ct.normalized_assistant_text,
                   ct.user_content, ct.assistant_content,
                   ct.user_raw_content, ct.assistant_raw_content,
                   ct.primary_tag, ct.tags_json, ct.session_date, ct.sender,
                   ct.fact_signals_json, ct.code_refs_json,
                   ct.covered_ingestible_entries,
                   ct.tagged_at, ct.compacted_at,
                   ct.first_seen_at, ct.last_seen_at,
                   ct.source_batch_id, ct.created_at, ct.updated_at
              FROM canonical_turns AS ct
              JOIN conversations AS c
                ON c.conversation_id = ct.conversation_id
             WHERE ct.conversation_id = %s
               AND ct.tagged_at IS NULL
               AND c.lifecycle_epoch = %s
             ORDER BY ct.sort_key ASC
             LIMIT %s
            """,
            (conversation_id, expected_lifecycle_epoch, batch_size),
        ).fetchall()
        return [_row_to_canonical_turn(row) for row in rows]

    def mark_canonical_row_tagged(
        self,
        *,
        canonical_turn_id: str,
        conversation_id: str,
        expected_lifecycle_epoch: int,
    ) -> bool:
        """Epoch-guarded flip of a single row's ``tagged_at``. Returns
        ``True`` iff exactly one row was updated.

        The ``EXISTS`` subclause pins the write to
        ``conversations.lifecycle_epoch == expected_lifecycle_epoch`` so a
        stale caller (whose conversation has been resurrected to a newer
        epoch) matches nothing and quietly gets ``False``. The
        ``tagged_at IS NULL`` predicate also makes the call idempotent — a
        retry on an already-tagged row is ``False`` (no-op).
        """
        now = _dt_to_str(datetime.now(timezone.utc))
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE canonical_turns
               SET tagged_at = %s, updated_at = %s
             WHERE canonical_turn_id = %s
               AND conversation_id = %s
               AND tagged_at IS NULL
               AND EXISTS (
                   SELECT 1 FROM conversations c
                    WHERE c.conversation_id = %s
                      AND c.lifecycle_epoch = %s
               )
            """,
            (now, now, canonical_turn_id, conversation_id,
             conversation_id, expected_lifecycle_epoch),
        )
        return int(cur.rowcount or 0) == 1

    def mark_canonical_turns_compacted(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        compacted_at: str | None = None,
    ) -> int:
        """Mark compacted_at on all physical rows whose turn_group_number
        matches any of the provided canonical_turn_ids.

        The compaction pipeline operates on MERGED rows (one per logical
        turn_group) produced by ``_merge_canonical_turn_rows``. Each merged
        row carries a single ``canonical_turn_id`` even though the
        underlying turn_group typically has two physical rows (user half +
        assistant half, each its own canonical_turn_id). Marking only the
        merged row's id would leave the sibling half NULL, re-surface it
        on the next ``get_uncompacted_canonical_turns`` call, and cause
        the next compaction to re-process the orphan half. Observed in
        prod on conv 77f110fc-0c00 (2026-04-19): first compaction marked
        1107 user halves, left 1082 assistant halves NULL, next compaction
        treated those 1082 orphans as "new uncompacted content" and ran
        the full LLM pipeline on them — doubled the compute cost and
        stuck compacted_prefix_messages at 2 forever.

        Fix: expand the WHERE clause to also match any row in the same
        turn_group_number as one of the input ids. turn_group_number < 0
        (legacy rows without explicit grouping) falls back to id-only
        match so we don't accidentally mark unrelated legacy rows.
        """
        if not canonical_turn_ids:
            return 0
        conn = self._get_conn()
        timestamp = compacted_at or _dt_to_str(datetime.now(timezone.utc))
        rows = conn.execute(
            """UPDATE canonical_turns
               SET compacted_at = %s, updated_at = %s
               WHERE conversation_id = %s
                 AND (
                     canonical_turn_id = ANY(%s)
                     OR turn_group_number IN (
                         SELECT DISTINCT turn_group_number
                           FROM canonical_turns
                          WHERE conversation_id = %s
                            AND canonical_turn_id = ANY(%s)
                            AND turn_group_number >= 0
                     )
                 )""",
            (
                timestamp, timestamp, conversation_id,
                canonical_turn_ids, conversation_id, canonical_turn_ids,
            ),
        )
        return int(rows.rowcount or 0)

    def delete_canonical_turns(
        self,
        conversation_id: str,
        turn_number: int | None = None,
    ) -> int:
        conn = self._get_conn()
        if turn_number is None:
            cur = conn.execute(
                "DELETE FROM canonical_turns WHERE conversation_id = %s",
                (conversation_id,),
            )
        else:
            canonical_turn_id = self._lookup_canonical_turn_id_for_ordinal(conversation_id, turn_number)
            if canonical_turn_id is None:
                return 0
            cur = conn.execute(
                "DELETE FROM canonical_turns WHERE conversation_id = %s AND canonical_turn_id = %s",
                (conversation_id, canonical_turn_id),
            )
        return int(cur.rowcount or 0)

    def delete_canonical_turns_by_batch_id(
        self,
        *,
        conversation_id: str,
        batch_id: str,
    ) -> int:
        """Delete rows from ``canonical_turns`` matching both ``conversation_id``
        AND ``source_batch_id``. Used by ``IngestReconciler`` for commit-time
        rollback on epoch race. Returns rows deleted."""
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM canonical_turns WHERE conversation_id = %s AND source_batch_id = %s",
            (conversation_id, batch_id),
        )
        return int(cur.rowcount or 0)

    def replace_canonical_turn_anchors(
        self,
        conversation_id: str,
        anchors: list[tuple[int, str, str]],
    ) -> int:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM canonical_turn_anchors WHERE conversation_id = %s",
            (conversation_id,),
        )
        rows = [
            (conversation_id, anchor_hash, start_turn_id, int(window_size))
            for window_size, anchor_hash, start_turn_id in anchors
            if anchor_hash and start_turn_id
        ]
        if not rows:
            return 0
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO canonical_turn_anchors
                   (conversation_id, anchor_hash, start_turn_id, window_size)
                   VALUES (%s, %s, %s, %s)""",
                rows,
            )
        return len(rows)

    def get_canonical_turn_anchor_positions(
        self,
        conversation_id: str,
        window_size: int,
    ) -> dict[str, list[int]]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT cta.anchor_hash, cto.turn_number
               FROM canonical_turn_anchors cta
               JOIN canonical_turns_ordinal cto
                 ON cto.conversation_id = cta.conversation_id
                AND cto.canonical_turn_id = cta.start_turn_id
               WHERE cta.conversation_id = %s
                 AND cta.window_size = %s
               ORDER BY cto.turn_number""",
            (conversation_id, int(window_size)),
        ).fetchall()
        anchors: dict[str, list[int]] = {}
        for row in rows:
            digest = str(row["anchor_hash"] or "")
            if not digest:
                continue
            anchors.setdefault(digest, []).append(int(row["turn_number"]))
        return anchors

    def search_tag_summaries_fts(
        self, query: str, limit: int = 20, conversation_id: str | None = None,
    ) -> list[tuple[str, float]]:
        conn = self._get_conn()
        tsquery = " & ".join(query.split()[:10])
        try:
            sql = """SELECT ts.tag,
                    ts_rank(to_tsvector('english', ts.summary), to_tsquery('english', %s)) as score
                FROM tag_summaries ts
                WHERE to_tsvector('english', ts.summary) @@ to_tsquery('english', %s)"""
            params: list = [tsquery, tsquery]
            if conversation_id is not None:
                sql += " AND ts.conversation_id = %s"
                params.append(conversation_id)
            sql += " ORDER BY score DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [(row["tag"], float(row["score"])) for row in rows]
        except Exception:
            return []

    def store_tag_summary_embedding(
        self, tag: str, conversation_id: str, embedding: list[float],
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tag_summary_embeddings (tag, conversation_id, embedding_json)
            VALUES (%s, %s, %s)
            ON CONFLICT (tag, conversation_id) DO UPDATE SET embedding_json = EXCLUDED.embedding_json""",
            (tag, conversation_id, json.dumps(embedding)),
        )

    def load_tag_summary_embeddings(
        self, conversation_id: str | None = None,
    ) -> dict[str, list[float]]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT tag, embedding_json FROM tag_summary_embeddings WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT tag, embedding_json FROM tag_summary_embeddings").fetchall()
        result = {}
        for row in rows:
            try:
                result[row["tag"]] = json.loads(row["embedding_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    def get_actionable_fact_tags(
        self, tags: list[str], conversation_id: str | None = None,
    ) -> set[str]:
        if not tags:
            return set()
        conn = self._get_conn()
        placeholders = ",".join("%s" for _ in tags)
        params: list = list(tags)
        conv_clause = ""
        if conversation_id is not None:
            conv_clause = " AND f.conversation_id = %s"
            params.append(conversation_id)
        try:
            rows = conn.execute(
                f"""SELECT DISTINCT ft.tag FROM fact_tags ft
                JOIN facts f ON f.id = ft.fact_id
                WHERE ft.tag IN ({placeholders})
                AND f.superseded_by IS NULL
                AND (f.status IN ('active', 'completed') OR f.fact_type = 'personal'){conv_clause}""",
                params,
            ).fetchall()
            return {row["tag"] for row in rows}
        except Exception:
            return set()

    # ------------------------------------------------------------------
    # FactStore
    # ------------------------------------------------------------------

    def store_facts(self, facts: list[Fact]) -> int:
        if not facts:
            return 0
        conn = self._get_conn()
        count = 0
        with conn.transaction():
            for fact in facts:
                conn.execute(
                    """INSERT INTO facts
                    (id, subject, verb, object, status, what, who, when_date, "where", why,
                     fact_type, tags_json, segment_ref, conversation_id, turn_numbers_json,
                     mentioned_at, session_date, superseded_by)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        subject=EXCLUDED.subject, verb=EXCLUDED.verb, object=EXCLUDED.object,
                        status=EXCLUDED.status, what=EXCLUDED.what, who=EXCLUDED.who,
                        when_date=EXCLUDED.when_date, "where"=EXCLUDED."where", why=EXCLUDED.why,
                        fact_type=EXCLUDED.fact_type, tags_json=EXCLUDED.tags_json,
                        segment_ref=EXCLUDED.segment_ref, conversation_id=EXCLUDED.conversation_id,
                        turn_numbers_json=EXCLUDED.turn_numbers_json, mentioned_at=EXCLUDED.mentioned_at,
                        session_date=EXCLUDED.session_date, superseded_by=EXCLUDED.superseded_by""",
                    (fact.id, fact.subject, fact.verb, fact.object, fact.status,
                     fact.what, fact.who, fact.when_date, fact.where, fact.why,
                     fact.fact_type, json.dumps(fact.tags), fact.segment_ref,
                     fact.conversation_id, json.dumps(fact.turn_numbers),
                     _dt_to_str(fact.mentioned_at), fact.session_date, fact.superseded_by),
                )
                conn.execute("DELETE FROM fact_tags WHERE fact_id = %s", (fact.id,))
                for tag in fact.tags:
                    conn.execute("INSERT INTO fact_tags (fact_id, tag) VALUES (%s, %s)", (fact.id, tag))
                count += 1
        return count

    def query_facts(
        self, *, subject: str | None = None, verb: str | None = None,
        verbs: list[str] | None = None, object_contains: str | None = None,
        status: str | None = None, fact_type: str | None = None,
        tags: list[str] | None = None, limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        conn = self._get_conn()
        conditions: list[str] = []
        params: list = []

        if conversation_id is not None:
            conditions.append("f.conversation_id = %s")
            params.append(conversation_id)
        if subject:
            conditions.append("f.subject = %s")
            params.append(subject)
        if verbs is not None:
            like_clauses = [f"f.verb ILIKE %s" for _ in verbs]
            conditions.append("(" + " OR ".join(like_clauses) + ")")
            params.extend(f"%{_escape_like(v)}%" for v in verbs)
        elif verb is not None:
            conditions.append("f.verb ILIKE %s")
            params.append(f"%{_escape_like(verb)}%")
        if object_contains:
            conditions.append("f.object ILIKE %s")
            params.append(f"%{_escape_like(object_contains)}%")
        if status:
            conditions.append("f.status = %s")
            params.append(status)
        if fact_type:
            conditions.append("f.fact_type = %s")
            params.append(fact_type)
        conditions.append("f.superseded_by IS NULL")

        if tags:
            where = " AND ".join(conditions) if conditions else "TRUE"
            sql = f"""
                SELECT DISTINCT f.* FROM facts f
                JOIN fact_tags ft ON f.id = ft.fact_id
                WHERE ft.tag = ANY(%s) AND {where}
                ORDER BY f.mentioned_at DESC LIMIT %s
            """
            params_list = [tags] + params + [limit]
            rows = conn.execute(sql, params_list).fetchall()
        else:
            where = " AND ".join(conditions) if conditions else "TRUE"
            sql = f"SELECT * FROM facts f WHERE {where} ORDER BY f.mentioned_at DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_fact(row) for row in rows]

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT DISTINCT verb FROM facts WHERE verb != '' AND superseded_by IS NULL AND conversation_id = %s",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT DISTINCT verb FROM facts WHERE verb != '' AND superseded_by IS NULL").fetchall()
        return [row["verb"] for row in rows]

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM facts WHERE segment_ref = %s ORDER BY mentioned_at", (segment_ref,)).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def replace_facts_for_segment(self, conversation_id: str, segment_ref: str, facts: list) -> tuple[int, int]:
        conn = self._get_conn()
        with conn.transaction():
            result = conn.execute(
                "DELETE FROM facts WHERE conversation_id = %s AND segment_ref = %s",
                (conversation_id, segment_ref),
            )
            deleted = result.rowcount
            inserted = self.store_facts(facts) if facts else 0
        return deleted, inserted

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list[Fact]:
        conn = self._get_conn()
        tsquery = " & ".join(query.split())
        try:
            sql = """SELECT f.* FROM facts f
                WHERE f.facts_tsv @@ to_tsquery('english', %s) AND f.superseded_by IS NULL"""
            params: list = [tsquery]
            if conversation_id is not None:
                sql += " AND f.conversation_id = %s"
                params.append(conversation_id)
            sql += " ORDER BY ts_rank(f.facts_tsv, to_tsquery('english', %s)) DESC LIMIT %s"
            params.extend([tsquery, limit])
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            like = f"%{query}%"
            sql = """SELECT * FROM facts f
                WHERE (f.subject ILIKE %s OR f.verb ILIKE %s OR f.object ILIKE %s OR f.what ILIKE %s)
                AND f.superseded_by IS NULL"""
            params = [like, like, like, like]
            if conversation_id is not None:
                sql += " AND f.conversation_id = %s"
                params.append(conversation_id)
            sql += " ORDER BY f.mentioned_at DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        conn = self._get_conn()
        conn.execute("UPDATE facts SET superseded_by = %s WHERE id = %s", (new_fact_id, old_fact_id))

    def update_fact_fields(self, fact_id: str, verb: str, object: str, status: str, what: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE facts SET verb = %s, object = %s, status = %s, what = %s WHERE id = %s",
            (verb, object, status, what, fact_id),
        )

    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT ft.tag, COUNT(*) as cnt FROM fact_tags ft JOIN facts f ON f.id = ft.fact_id WHERE f.conversation_id = %s GROUP BY ft.tag",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT tag, COUNT(*) as cnt FROM fact_tags GROUP BY tag").fetchall()
        return {row["tag"]: row["cnt"] for row in rows}

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        if not fact_ids:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT superseded_by, subject, verb, object FROM facts WHERE superseded_by = ANY(%s)",
            (fact_ids,),
        ).fetchall()
        return [dict(row) for row in rows]

    def query_experience_facts_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        conn = self._get_conn()
        sql = """SELECT * FROM facts
                 WHERE when_date >= %s AND when_date <= %s"""
        params: list = [start_date, end_date + "~"]
        if conversation_id:
            sql += " AND conversation_id = %s"
            params.append(conversation_id)
        sql += " ORDER BY when_date ASC LIMIT %s"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    # ------------------------------------------------------------------
    # FactLinkStore
    # ------------------------------------------------------------------

    def store_fact_links(self, links: list[FactLink]) -> int:
        if not links:
            return 0
        conn = self._get_conn()
        count = 0
        with conn.transaction():
            for link in links:
                conn.execute(
                    """INSERT INTO fact_links (id, source_fact_id, target_fact_id, relation_type,
                       confidence, context, created_at, created_by)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        source_fact_id=EXCLUDED.source_fact_id, target_fact_id=EXCLUDED.target_fact_id,
                        relation_type=EXCLUDED.relation_type, confidence=EXCLUDED.confidence,
                        context=EXCLUDED.context""",
                    (link.id, link.source_fact_id, link.target_fact_id, link.relation_type,
                     link.confidence, link.context, _dt_to_str(link.created_at), link.created_by),
                )
                count += 1
        return count

    def get_fact_links(self, fact_id: str, direction: str = "both") -> list[FactLink]:
        conn = self._get_conn()
        if direction == "outgoing":
            rows = conn.execute("SELECT * FROM fact_links WHERE source_fact_id = %s", (fact_id,)).fetchall()
        elif direction == "incoming":
            rows = conn.execute("SELECT * FROM fact_links WHERE target_fact_id = %s", (fact_id,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fact_links WHERE source_fact_id = %s OR target_fact_id = %s",
                (fact_id, fact_id),
            ).fetchall()
        return [self._row_to_fact_link(row) for row in rows]

    def get_linked_facts(self, fact_ids: list[str], depth: int = 1) -> list[LinkedFact]:
        if not fact_ids:
            return []
        conn = self._get_conn()
        visited = set(fact_ids)
        result: list[LinkedFact] = []
        current_layer = set(fact_ids)

        for _hop in range(depth):
            if not current_layer:
                break
            layer_list = list(current_layer)
            visited_list = list(visited)
            rows = conn.execute(
                """SELECT fl.source_fact_id, fl.target_fact_id, fl.relation_type, fl.confidence, fl.context,
                    f.id AS fact_id, f.subject, f.verb, f.object, f.status, f.what, f.who,
                    f.when_date, f."where", f.why, f.fact_type, f.tags_json, f.segment_ref,
                    f.conversation_id, f.turn_numbers_json, f.mentioned_at, f.session_date, f.superseded_by
                FROM fact_links fl
                JOIN facts f ON (
                    (fl.source_fact_id = ANY(%s) AND f.id = fl.target_fact_id)
                    OR (fl.target_fact_id = ANY(%s) AND f.id = fl.source_fact_id)
                )
                WHERE f.superseded_by IS NULL
                AND f.id != ALL(%s)""",
                (layer_list, layer_list, visited_list),
            ).fetchall()

            next_layer: set[str] = set()
            for row in rows:
                fact = Fact(
                    id=row["fact_id"], subject=row["subject"], verb=row["verb"],
                    object=row["object"], status=row["status"], what=row["what"],
                    who=row["who"], when_date=row["when_date"], where=row["where"],
                    why=row["why"], fact_type=row.get("fact_type", "personal"),
                    tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
                    segment_ref=row["segment_ref"], conversation_id=row["conversation_id"],
                    turn_numbers=json.loads(row["turn_numbers_json"]) if row["turn_numbers_json"] else [],
                    mentioned_at=_str_to_dt(row["mentioned_at"]) if row["mentioned_at"] else datetime.now(timezone.utc),
                    session_date=row.get("session_date", ""), superseded_by=row["superseded_by"],
                )
                src = row["source_fact_id"]
                tgt = row["target_fact_id"]
                linked_from = src if src in current_layer else tgt
                result.append(LinkedFact(
                    fact=fact, linked_from_fact_id=linked_from,
                    relation_type=row["relation_type"], confidence=row["confidence"],
                    link_context=row["context"],
                ))
                visited.add(fact.id)
                next_layer.add(fact.id)
            current_layer = next_layer

        return result

    def delete_fact_links(self, fact_id: str) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM fact_links WHERE source_fact_id = %s OR target_fact_id = %s",
            (fact_id, fact_id),
        )
        return cur.rowcount

    def migrate_supersession_to_links(self) -> int:
        conn = self._get_conn()
        rows = conn.execute("SELECT id, superseded_by FROM facts WHERE superseded_by IS NOT NULL").fetchall()
        if not rows:
            return 0
        count = 0
        for row in rows:
            old_id, new_id = row["id"], row["superseded_by"]
            existing = conn.execute(
                "SELECT 1 FROM fact_links WHERE source_fact_id = %s AND target_fact_id = %s AND relation_type = 'supersedes'",
                (new_id, old_id),
            ).fetchone()
            if existing:
                continue
            link = FactLink(source_fact_id=new_id, target_fact_id=old_id, relation_type="supersedes",
                            confidence=1.0, context="Migrated from superseded_by column", created_by="migration")
            conn.execute(
                """INSERT INTO fact_links (id, source_fact_id, target_fact_id, relation_type,
                   confidence, context, created_at, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (link.id, link.source_fact_id, link.target_fact_id, link.relation_type,
                 link.confidence, link.context, _dt_to_str(link.created_at), link.created_by),
            )
            count += 1
        return count

    def _allocate_request_turn(self, conn: psycopg.Connection, conversation_id: str) -> int:
        row = conn.execute(
            """INSERT INTO request_turn_counters (conversation_id, next_request_turn)
               VALUES (%s, 1)
               ON CONFLICT (conversation_id)
               DO UPDATE SET next_request_turn = request_turn_counters.next_request_turn + 1
               RETURNING next_request_turn""",
            (conversation_id,),
        ).fetchone()
        return int((row or {}).get("next_request_turn", 0) or 0)

    def _bump_request_turn_counter(
        self,
        conn: psycopg.Connection,
        conversation_id: str,
        request_turn: int,
    ) -> None:
        conn.execute(
            """INSERT INTO request_turn_counters (conversation_id, next_request_turn)
               VALUES (%s, %s)
               ON CONFLICT (conversation_id)
               DO UPDATE
               SET next_request_turn = GREATEST(
                   request_turn_counters.next_request_turn,
                   EXCLUDED.next_request_turn
               )""",
            (conversation_id, int(request_turn)),
        )

    def _normalize_request_turn_sequences(self) -> None:
        conn = self._get_conn()
        with conn.transaction():
            context_rows = conn.execute(
                "SELECT id, conversation_id, request_turn, timestamp FROM request_context "
                "ORDER BY conversation_id, id"
            ).fetchall()
            if not context_rows:
                return

            grouped_contexts: dict[str, list[dict]] = {}
            context_updates: list[tuple[int, int]] = []
            for row in context_rows:
                conversation_id = row["conversation_id"]
                seq = len(grouped_contexts.setdefault(conversation_id, [])) + 1
                timestamp = _parse_sequence_timestamp(row.get("timestamp"))
                grouped_contexts[conversation_id].append({
                    "id": int(row["id"]),
                    "request_turn": seq,
                    "timestamp": timestamp,
                })
                if int(row.get("request_turn", 0) or 0) != seq:
                    context_updates.append((seq, int(row["id"])))

            if context_updates:
                for params in context_updates:
                    conn.execute(
                        "UPDATE request_context SET request_turn = %s WHERE id = %s",
                        params,
                    )

            tool_rows = conn.execute(
                "SELECT id, conversation_id, request_turn, timestamp FROM tool_calls "
                "ORDER BY conversation_id, id"
            ).fetchall()
            tool_updates: list[tuple[int, int]] = []
            for row in tool_rows:
                conversation_id = row["conversation_id"]
                contexts = grouped_contexts.get(conversation_id)
                if not contexts:
                    continue
                tool_ts = _parse_sequence_timestamp(row.get("timestamp"))
                assigned_turn = contexts[0]["request_turn"]
                if tool_ts is not None:
                    for ctx in contexts:
                        ctx_ts = ctx["timestamp"]
                        if ctx_ts is None or ctx_ts <= tool_ts:
                            assigned_turn = ctx["request_turn"]
                        else:
                            break
                else:
                    assigned_turn = contexts[-1]["request_turn"]
                if int(row.get("request_turn", 0) or 0) != assigned_turn:
                    tool_updates.append((assigned_turn, int(row["id"])))

            if tool_updates:
                for params in tool_updates:
                    conn.execute(
                        "UPDATE tool_calls SET request_turn = %s WHERE id = %s",
                        params,
                    )

            counter_rows = [
                (conversation_id, contexts[-1]["request_turn"])
                for conversation_id, contexts in grouped_contexts.items()
                if contexts
            ]
            if counter_rows:
                for params in counter_rows:
                    conn.execute(
                        """INSERT INTO request_turn_counters (conversation_id, next_request_turn)
                           VALUES (%s, %s)
                           ON CONFLICT (conversation_id)
                           DO UPDATE
                           SET next_request_turn = GREATEST(
                               request_turn_counters.next_request_turn,
                               EXCLUDED.next_request_turn
                           )""",
                        params,
                    )

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def save_tool_call(self, call: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tool_calls
            (conversation_id, request_turn, round, group_id, tool_name,
             tool_input, tool_result, result_length, duration_ms, found, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                call.get("conversation_id", ""),
                call.get("request_turn", 0),
                call.get("round", 1),
                call.get("group_id", ""),
                call.get("tool_name", ""),
                json.dumps(call.get("tool_input", {})),
                call.get("tool_result", ""),
                call.get("result_length", 0),
                call.get("duration_ms", 0),
                call.get("found"),
                call.get("timestamp", ""),
            ),
        )
        conv_id = call.get("conversation_id", "")
        conn.execute(
            """DELETE FROM tool_calls WHERE id NOT IN (
                SELECT id FROM tool_calls WHERE conversation_id = %s
                ORDER BY id DESC LIMIT 50
            ) AND conversation_id = %s""",
            (conv_id, conv_id),
        )

    def load_tool_calls(self, conversation_id: str, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tool_calls WHERE conversation_id = %s ORDER BY id DESC LIMIT %s",
            (conversation_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def load_tool_call(self, call_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tool_calls WHERE id = %s", (call_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Request context persistence (dashboard recall page)
    # ------------------------------------------------------------------

    def save_request_context(self, context: dict) -> int:
        conn = self._get_conn()
        conv_id = context.get("conversation_id", "")
        explicit_turn = int(context.get("request_turn", 0) or 0)
        with conn.transaction():
            request_turn = explicit_turn or self._allocate_request_turn(conn, conv_id)
            if explicit_turn:
                self._bump_request_turn_counter(conn, conv_id, request_turn)
            conn.execute(
                """INSERT INTO request_context
                (conversation_id, request_turn, timestamp, user_message, inbound_tags,
                 retrieval_method, candidates_found, candidates_selected,
                 segments_injected, facts_injected, facts_count, facts_tags,
                 pool_used, pool_budget, total_context_tokens,
                 non_virtualizable_floor, tool_call_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    conv_id,
                    request_turn,
                    context.get("timestamp", ""),
                    context.get("user_message", ""),
                    json.dumps(context.get("inbound_tags", [])),
                    context.get("retrieval_method", ""),
                    context.get("candidates_found", 0),
                    context.get("candidates_selected", 0),
                    json.dumps(context.get("segments_injected", [])),
                    json.dumps(context.get("facts_injected", [])),
                    context.get("facts_count", 0),
                    json.dumps(context.get("facts_tags", [])),
                    context.get("pool_used", 0),
                    context.get("pool_budget", 0),
                    context.get("total_context_tokens", 0),
                    context.get("non_virtualizable_floor", 0),
                    context.get("tool_call_count", 0),
                ),
            )
            conn.execute(
                """DELETE FROM request_context WHERE id NOT IN (
                    SELECT id FROM request_context WHERE conversation_id = %s
                    ORDER BY id DESC LIMIT 50
                ) AND conversation_id = %s""",
                (conv_id, conv_id),
            )
        return request_turn

    def load_request_contexts(self, conversation_id: str, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM (
                SELECT
                    rc.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY rc.conversation_id
                        ORDER BY rc.id
                    ) AS sequence_number
                FROM request_context rc
                WHERE rc.conversation_id = %s
            ) ranked
            ORDER BY id DESC
            LIMIT %s""",
            (conversation_id, limit),
        ).fetchall()
        result = []
        for row in reversed(rows):
            d = dict(row)
            for json_field in ("inbound_tags", "segments_injected", "facts_injected", "facts_tags"):
                try:
                    d[json_field] = json.loads(d.get(json_field, "[]"))
                except (json.JSONDecodeError, TypeError):
                    d[json_field] = []
            result.append(d)
        return result

    def save_ingest_batch(self, batch: dict) -> str:
        conn = self._get_conn()
        batch_id = str(batch.get("batch_id", "") or generate_canonical_turn_id())
        conn.execute(
            """INSERT INTO ingest_batches
            (batch_id, conversation_id, received_at, raw_turn_count, merge_mode,
             turns_matched, turns_appended, turns_prepended, turns_inserted,
             first_turn_hash, last_turn_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (batch_id) DO UPDATE SET
                merge_mode = EXCLUDED.merge_mode,
                turns_matched = EXCLUDED.turns_matched,
                turns_appended = EXCLUDED.turns_appended,
                turns_prepended = EXCLUDED.turns_prepended,
                turns_inserted = EXCLUDED.turns_inserted,
                first_turn_hash = EXCLUDED.first_turn_hash,
                last_turn_hash = EXCLUDED.last_turn_hash""",
            (
                batch_id,
                batch.get("conversation_id", ""),
                batch.get("received_at", "") or utcnow_iso(),
                int(batch.get("raw_turn_count", 0) or 0),
                batch.get("merge_mode", "") or "",
                int(batch.get("turns_matched", 0) or 0),
                int(batch.get("turns_appended", 0) or 0),
                int(batch.get("turns_prepended", 0) or 0),
                int(batch.get("turns_inserted", 0) or 0),
                batch.get("first_turn_hash", "") or "",
                batch.get("last_turn_hash", "") or "",
            ),
        )
        return batch_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._conn_lock:
            conns = list(self._connections.values())
            self._connections.clear()
        for conn in conns:
            if not conn.closed:
                conn.close()
        self._conn_local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
