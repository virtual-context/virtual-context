"""SQLiteStore: primary storage backend using stdlib sqlite3."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_sequence_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None

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
from ..core.store import ContextStore
from ..types import ChunkEmbedding, ConversationStats, DepthLevel, EngineStateSnapshot, Fact, FactLink, FactSignal, CanonicalTurnChunkEmbedding, CanonicalTurnRow, LinkedFact, QuoteResult, SegmentMetadata, StoredSegment, StoredSummary, TagStats, TagSummary, TemporalStatus, TurnTagEntry, WorkingSetEntry
from .helpers import dt_to_str as _dt_to_str, str_to_dt as _str_to_dt, extract_excerpt as _extract_excerpt

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

CREATE TABLE IF NOT EXISTS cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
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
    covers_through_turn INTEGER NOT NULL DEFAULT -1,
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
    deleted INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_chunks (
    segment_ref TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (segment_ref, chunk_index)
);

CREATE TABLE IF NOT EXISTS canonical_turns (
    canonical_turn_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    sort_key REAL NOT NULL,
    turn_hash TEXT NOT NULL,
    hash_version INTEGER NOT NULL DEFAULT 1,
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
    source_batch_id TEXT,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    UNIQUE (conversation_id, sort_key)
);

CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_order
ON canonical_turns(conversation_id, sort_key);

CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_hash
ON canonical_turns(conversation_id, turn_hash);

CREATE INDEX IF NOT EXISTS idx_canonical_turns_compaction_queue
ON canonical_turns(conversation_id, compacted_at, sort_key);

CREATE TABLE IF NOT EXISTS canonical_turn_anchors (
    conversation_id TEXT NOT NULL,
    anchor_hash TEXT NOT NULL,
    start_turn_id TEXT NOT NULL,
    window_size INTEGER NOT NULL DEFAULT 3
);

CREATE INDEX IF NOT EXISTS idx_canonical_turn_anchors_lookup
ON canonical_turn_anchors(conversation_id, window_size, anchor_hash);

CREATE TABLE IF NOT EXISTS ingest_batches (
    batch_id TEXT PRIMARY KEY,
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

CREATE TABLE IF NOT EXISTS canonical_turn_chunks (
    conversation_id TEXT NOT NULL,
    canonical_turn_id TEXT NOT NULL,
    side TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, canonical_turn_id, side, chunk_index)
);

CREATE TABLE IF NOT EXISTS conversation_aliases (
    alias_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_primary_tag ON segments(primary_tag);
CREATE INDEX IF NOT EXISTS idx_segments_created_at ON segments(created_at);
CREATE INDEX IF NOT EXISTS idx_segments_conversation_id ON segments(conversation_id);
CREATE INDEX IF NOT EXISTS idx_segment_tags_tag ON segment_tags(tag);
"""

FTS_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    ref UNINDEXED,
    summary,
    content='segments',
    content_rowid='rowid'
);
"""

FTS_TRIGGER_SQL = """\
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, ref, summary) VALUES (new.rowid, new.ref, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, ref, summary) VALUES('delete', old.rowid, old.ref, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, ref, summary) VALUES('delete', old.rowid, old.ref, old.summary);
    INSERT INTO segments_fts(rowid, ref, summary) VALUES (new.rowid, new.ref, new.summary);
END;
"""

FTS_FULLTEXT_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts_full USING fts5(
    ref UNINDEXED,
    full_text,
    content='segments',
    content_rowid='rowid'
);
"""

FTS_FULLTEXT_TRIGGER_SQL = """\
CREATE TRIGGER IF NOT EXISTS segments_ft_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts_full(rowid, ref, full_text) VALUES (new.rowid, new.ref, new.full_text);
END;
CREATE TRIGGER IF NOT EXISTS segments_ft_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts_full(segments_fts_full, rowid, ref, full_text) VALUES('delete', old.rowid, old.ref, old.full_text);
END;
CREATE TRIGGER IF NOT EXISTS segments_ft_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts_full(segments_fts_full, rowid, ref, full_text) VALUES('delete', old.rowid, old.ref, old.full_text);
    INSERT INTO segments_fts_full(rowid, ref, full_text) VALUES (new.rowid, new.ref, new.full_text);
END;
"""

FACTS_FTS_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    id UNINDEXED,
    subject,
    verb,
    object,
    what,
    content='facts',
    content_rowid='rowid'
);
"""

FACTS_FTS_TRIGGER_SQL = """\
CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, id, subject, verb, object, what)
    VALUES (new.rowid, new.id, new.subject, new.verb, new.object, new.what);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, id, subject, verb, object, what)
    VALUES('delete', old.rowid, old.id, old.subject, old.verb, old.object, old.what);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, id, subject, verb, object, what)
    VALUES('delete', old.rowid, old.id, old.subject, old.verb, old.object, old.what);
    INSERT INTO facts_fts(rowid, id, subject, verb, object, what)
    VALUES (new.rowid, new.id, new.subject, new.verb, new.object, new.what);
END;
"""


def _escape_like(text: str) -> str:
    """Escape ``%`` and ``_`` wildcards for use in LIKE clauses with ``ESCAPE '\\'``."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sanitize_fts_query(query: str) -> str:
    """Quote user input so FTS5 treats it as a phrase, not operator syntax.

    FTS5 supports operators like AND, OR, NOT, NEAR, *, column filters (col:val),
    etc.  Wrapping the query in double quotes forces phrase matching and prevents
    users from injecting FTS5 syntax.
    """
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


def _sanitize_fts_query_terms(query: str) -> str:
    """Sanitize user input as individual OR-joined terms for BM25 scoring.

    Unlike ``_sanitize_fts_query`` (phrase match), this splits the query
    into individual words and joins them with OR so that BM25 scoring
    rewards documents matching *any* of the query terms.

    Each term is individually quoted to prevent FTS5 operator injection.
    Returns an empty string if no valid terms are found.
    """
    terms = []
    for word in query.split():
        cleaned = word.strip().replace('"', '""')
        if cleaned:
            terms.append(f'"{cleaned}"')
    return " OR ".join(terms)


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


def _row_to_segment(row: sqlite3.Row, tags: list[str]) -> StoredSegment:
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


def _row_to_summary(row: sqlite3.Row, tags: list[str]) -> StoredSummary:
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


def _row_to_canonical_turn(row: sqlite3.Row) -> CanonicalTurnRow:
    tags_raw = row["tags_json"] if "tags_json" in row.keys() else "[]"
    fact_signals_raw = row["fact_signals_json"] if "fact_signals_json" in row.keys() else "[]"
    code_refs_raw = row["code_refs_json"] if "code_refs_json" in row.keys() else "[]"
    try:
        tags = json.loads(tags_raw or "[]")
    except Exception:
        tags = []
    try:
        code_refs = json.loads(code_refs_raw or "[]")
    except Exception:
        code_refs = []
    fact_signals: list[FactSignal] = []
    try:
        for item in json.loads(fact_signals_raw or "[]"):
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
    # turn_number is a VIEW-only column computed by canonical_turns_ordinal via
    # ROW_NUMBER(). Base-table queries (e.g. iter_untagged_canonical_rows, which
    # hits the partial index idx_canonical_turns_conv_untagged and therefore
    # cannot join through the view) won't supply it — default to -1.
    try:
        turn_number = row["turn_number"]
    except (KeyError, IndexError):
        turn_number = -1
    return CanonicalTurnRow(
        conversation_id=row["conversation_id"],
        canonical_turn_id=str(row["canonical_turn_id"]) if "canonical_turn_id" in row.keys() and row["canonical_turn_id"] else "",
        turn_number=turn_number,
        turn_group_number=int(row["turn_group_number"]) if "turn_group_number" in row.keys() and row["turn_group_number"] is not None else -1,
        sort_key=float(row["sort_key"]) if "sort_key" in row.keys() and row["sort_key"] is not None else 0.0,
        turn_hash=(row["turn_hash"] if "turn_hash" in row.keys() else "") or "",
        hash_version=int(row["hash_version"]) if "hash_version" in row.keys() and row["hash_version"] is not None else 0,
        normalized_user_text=(row["normalized_user_text"] if "normalized_user_text" in row.keys() else "") or "",
        normalized_assistant_text=(row["normalized_assistant_text"] if "normalized_assistant_text" in row.keys() else "") or "",
        user_content=row["user_content"] or "",
        assistant_content=row["assistant_content"] or "",
        user_raw_content=row["user_raw_content"],
        assistant_raw_content=row["assistant_raw_content"],
        primary_tag=(row["primary_tag"] if "primary_tag" in row.keys() else "_general") or "_general",
        tags=list(tags or []),
        session_date=(row["session_date"] if "session_date" in row.keys() else "") or "",
        sender=(row["sender"] if "sender" in row.keys() else "") or "",
        fact_signals=fact_signals,
        code_refs=list(code_refs or []),
        tagged_at=((row["tagged_at"] if "tagged_at" in row.keys() else None) or None),
        compacted_at=((row["compacted_at"] if "compacted_at" in row.keys() else None) or None),
        first_seen_at=((row["first_seen_at"] if "first_seen_at" in row.keys() else None) or None),
        last_seen_at=((row["last_seen_at"] if "last_seen_at" in row.keys() else None) or None),
        source_batch_id=((row["source_batch_id"] if "source_batch_id" in row.keys() else None) or None),
        covered_ingestible_entries=int(row["covered_ingestible_entries"])
        if "covered_ingestible_entries" in row.keys() and row["covered_ingestible_entries"] is not None
        else 1,
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
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


class SQLiteStore(ContextStore):
    """SQLite-based storage with tag-overlap queries and FTS5 search."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.search_config = None  # set by engine after construction
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            # 30s busy timeout: tool_output_interceptor writes on the request
            # path while compaction holds a write lock in a background thread.
            # 5s was too short — caused sqlite3.OperationalError: database is
            # locked, crashing the proxy with 500s (A/B test 2026-03-01).
            conn = sqlite3.connect(
                str(self.db_path), timeout=30, isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _reconcile_lock_depth(self) -> int:
        return int(getattr(self._local, "reconcile_lock_depth", 0) or 0)

    def _reconcile_lock_active(self) -> bool:
        return self._reconcile_lock_depth() > 0

    def _commit_if_unlocked(self, conn: sqlite3.Connection) -> None:
        if not self._reconcile_lock_active():
            conn.commit()

    @contextmanager
    def conversation_reconcile(self, conversation_id: str):
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        self._local.reconcile_lock_depth = self._reconcile_lock_depth() + 1
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._local.reconcile_lock_depth = max(self._reconcile_lock_depth() - 1, 0)

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        # Migration: rename session_id → conversation_id (idempotent).
        # Must run BEFORE SCHEMA_SQL because SCHEMA_SQL creates indexes
        # on conversation_id that would fail on pre-migration tables.
        for table in ("segments", "engine_state", "facts", "tool_outputs"):
            try:
                conn.execute(f"ALTER TABLE {table} RENAME COLUMN session_id TO conversation_id")
            except sqlite3.OperationalError:
                pass  # Column already renamed or table doesn't exist
        conn.executescript(SCHEMA_SQL)
        # Lifecycle/phase-tracked conversations table. New in the progress-bar
        # redesign — carries lifecycle_epoch (for delete+resurrect invariants),
        # a phase state machine (init/ingesting/compacting/active/deleted),
        # and per-request metadata counters consumed by the progress tracker.
        # Future tables (ingestion_episode, compaction_operation) will
        # reference this one.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id                TEXT PRIMARY KEY,
                tenant_id                      TEXT NOT NULL,
                lifecycle_epoch                INTEGER NOT NULL DEFAULT 1,
                phase                          TEXT NOT NULL DEFAULT 'init'
                                               CHECK (phase IN ('init','ingesting','compacting','active','deleted')),
                pending_raw_payload_entries    INTEGER NOT NULL DEFAULT 0,
                last_raw_payload_entries       INTEGER NOT NULL DEFAULT 0,
                last_ingestible_payload_entries INTEGER NOT NULL DEFAULT 0,
                created_at                     TEXT NOT NULL,
                updated_at                     TEXT NOT NULL,
                deleted_at                     TEXT NULL,
                UNIQUE (tenant_id, conversation_id)
            )
        """)
        # Drop-and-recreate keeps the partial-index clause in lockstep with
        # Postgres (see postgres.py). SQLite 3.8+ supports partial indexes;
        # the `WHERE phase <> 'deleted'` predicate excludes tombstones so
        # tenant-scoped phase queries never scan them. The drop is
        # idempotent on fresh DBs (no-op) and replaces any pre-partial index
        # on upgraded DBs.
        conn.execute("DROP INDEX IF EXISTS idx_conversations_tenant_phase")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_tenant_phase
                ON conversations(tenant_id, phase)
                 WHERE phase <> 'deleted'
        """)
        # Ownership + lifecycle record for an ingestion episode. Progress
        # counters (done/total) are DERIVED from canonical_turns SUMs at read
        # time — this row only tracks ownership (`owner_worker_id`,
        # `heartbeat_ts`), the largest raw payload observed during the episode
        # (`raw_payload_entries`), and the status transitions. The partial
        # unique index below enforces at-most-one running episode per
        # (conversation, lifecycle_epoch) at the DB layer without requiring a
        # distributed lock — a concurrent worker attempting to INSERT a second
        # 'running' row collides at INSERT time.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_episode (
                episode_id            TEXT PRIMARY KEY,
                conversation_id       TEXT NOT NULL,
                lifecycle_epoch       INTEGER NOT NULL,
                raw_payload_entries   INTEGER NOT NULL DEFAULT 0,
                started_at            TEXT NOT NULL,
                completed_at          TEXT NULL,
                status                TEXT NOT NULL
                                      CHECK (status IN ('running','completed','cancelled','abandoned')),
                owner_worker_id       TEXT NOT NULL,
                heartbeat_ts          TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_episode_active
                ON ingestion_episode(conversation_id, lifecycle_epoch)
                WHERE status = 'running'
        """)
        # Ownership + lifecycle record for a compaction operation. Sibling to
        # ingestion_episode but tracks the multi-phase compaction pipeline
        # (phase_index/phase_count/phase_name) rather than raw payload
        # counts. `status` carries a `queued` state in addition to the
        # episode lifecycle so workers can enqueue work ahead of execution,
        # and the partial unique index treats both `queued` and `running`
        # as active — only one pending-or-in-flight compaction is allowed
        # per (conversation, lifecycle_epoch) at the DB layer.
        # ``'abandoned'`` is added to the CHECK for the takeover cleanup path.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compaction_operation (
                operation_id      TEXT PRIMARY KEY,
                conversation_id   TEXT NOT NULL,
                lifecycle_epoch   INTEGER NOT NULL,
                phase_index       INTEGER NOT NULL DEFAULT 0,
                phase_count       INTEGER NOT NULL,
                phase_name        TEXT NOT NULL,
                status            TEXT NOT NULL
                                  CHECK (status IN ('queued','running','completed','cancelled','failed','abandoned')),
                started_at        TEXT NOT NULL,
                completed_at      TEXT NULL,
                owner_worker_id   TEXT NOT NULL,
                heartbeat_ts      TEXT NOT NULL,
                created_at        TEXT NOT NULL DEFAULT '',
                error_message     TEXT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_compaction_operation_active
                ON compaction_operation(conversation_id, lifecycle_epoch)
                WHERE status IN ('queued','running')
        """)
        # Migration: compaction_operation — add 'abandoned' to status CHECK
        # and add created_at column. SQLite cannot ALTER a CHECK constraint
        # in place, so we detect old schema (missing created_at column) and
        # recreate the table via the standard rename dance. This is idempotent:
        # if created_at already exists (new schema), the probe SELECT succeeds
        # and the migration is skipped.
        try:
            conn.execute("SELECT created_at FROM compaction_operation LIMIT 0")
        except sqlite3.OperationalError:
            # Old schema: recreate with expanded CHECK + created_at column.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS compaction_operation_new (
                    operation_id      TEXT PRIMARY KEY,
                    conversation_id   TEXT NOT NULL,
                    lifecycle_epoch   INTEGER NOT NULL,
                    phase_index       INTEGER NOT NULL DEFAULT 0,
                    phase_count       INTEGER NOT NULL,
                    phase_name        TEXT NOT NULL,
                    status            TEXT NOT NULL
                                      CHECK (status IN ('queued','running','completed','cancelled','failed','abandoned')),
                    started_at        TEXT NOT NULL,
                    completed_at      TEXT NULL,
                    owner_worker_id   TEXT NOT NULL,
                    heartbeat_ts      TEXT NOT NULL,
                    created_at        TEXT NOT NULL DEFAULT '',
                    error_message     TEXT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );
                INSERT INTO compaction_operation_new
                    (operation_id, conversation_id, lifecycle_epoch,
                     phase_index, phase_count, phase_name, status,
                     started_at, completed_at, owner_worker_id, heartbeat_ts,
                     created_at, error_message)
                SELECT operation_id, conversation_id, lifecycle_epoch,
                       phase_index, phase_count, phase_name, status,
                       started_at, completed_at, owner_worker_id, heartbeat_ts,
                       '', error_message
                FROM compaction_operation;
                DROP TABLE compaction_operation;
                ALTER TABLE compaction_operation_new RENAME TO compaction_operation;
                CREATE UNIQUE INDEX IF NOT EXISTS idx_compaction_operation_active
                    ON compaction_operation(conversation_id, lifecycle_epoch)
                    WHERE status IN ('queued','running');
            """)
        try:
            conn.executescript(FTS_SQL)
            conn.executescript(FTS_TRIGGER_SQL)
        except sqlite3.OperationalError:
            pass  # FTS5 not available, search will fall back
        # FTS full-text index (searches full_text column, separate from summary FTS)
        try:
            conn.executescript(FTS_FULLTEXT_SQL)
            conn.executescript(FTS_FULLTEXT_TRIGGER_SQL)
            # Backfill: populate from existing segments not yet indexed
            count = conn.execute("SELECT COUNT(*) FROM segments_fts_full").fetchone()[0]
            if count == 0:
                conn.execute("""
                    INSERT INTO segments_fts_full(rowid, ref, full_text)
                    SELECT rowid, ref, full_text FROM segments WHERE full_text != ''
                """)
        except sqlite3.OperationalError:
            pass
        # Cascade delete chunk embeddings when parent segment is deleted
        try:
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS segments_chunks_ad AFTER DELETE ON segments BEGIN
                    DELETE FROM segment_chunks WHERE segment_ref = old.ref;
                END;
            """)
        except sqlite3.OperationalError:
            pass
        # Migrations: add columns that didn't exist in earlier schema versions
        try:
            conn.execute("ALTER TABLE tag_summaries ADD COLUMN description TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute("ALTER TABLE tag_summaries ADD COLUMN code_refs TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute("ALTER TABLE tag_summaries ADD COLUMN generated_by_turn_id TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cur = conn.execute("PRAGMA table_info(tag_aliases)")
            cols = cur.fetchall()
            col_names = [row[1] for row in cols]
            pk_names = [row[1] for row in sorted(cols, key=lambda row: row[5]) if row[5] > 0]
            if col_names and (
                "conversation_id" not in col_names
                or pk_names != ["alias", "conversation_id"]
            ):
                legacy_rows = conn.execute(
                    "SELECT alias, canonical FROM tag_aliases"
                ).fetchall()
                migrated_rows = [
                    (row["alias"], "", row["canonical"])
                    for row in legacy_rows
                ]
                conn.executescript("""
                    DROP TABLE IF EXISTS tag_aliases_new;
                    CREATE TABLE tag_aliases_new (
                        alias TEXT NOT NULL,
                        conversation_id TEXT NOT NULL DEFAULT '',
                        canonical TEXT NOT NULL,
                        PRIMARY KEY (alias, conversation_id)
                    );
                    DROP TABLE tag_aliases;
                    ALTER TABLE tag_aliases_new RENAME TO tag_aliases;
                """)
                if migrated_rows:
                    conn.executemany(
                        """INSERT OR REPLACE INTO tag_aliases
                        (alias, conversation_id, canonical)
                        VALUES (?, ?, ?)""",
                        migrated_rows,
                    )
        except sqlite3.OperationalError:
            pass
        try:
            cur = conn.execute("PRAGMA table_info(request_captures)")
            cols = cur.fetchall()
            col_names = [row[1] for row in cols]
            pk_names = [row[1] for row in sorted(cols, key=lambda row: row[5]) if row[5] > 0]
            if col_names and (
                "conversation_id" not in col_names
                or "turn_id" not in col_names
                or pk_names != ["conversation_id", "turn", "turn_id"]
            ):
                select_cols = ["turn", "ts", "recorded_at", "data_json"]
                if "turn_id" in col_names:
                    select_cols.append("turn_id")
                legacy_rows = conn.execute(
                    f"SELECT {', '.join(select_cols)} FROM request_captures"
                ).fetchall()
                migrated_rows: list[tuple[str, int, str, str, float, str]] = []
                for row in legacy_rows:
                    data_json = row["data_json"]
                    conversation_id = ""
                    turn = row["turn"]
                    turn_id = row["turn_id"] if "turn_id" in row.keys() else ""
                    try:
                        payload = json.loads(data_json)
                        conversation_id = payload.get("conversation_id", "") or ""
                        turn = int(payload.get("turn", turn))
                        turn_id = payload.get("turn_id", turn_id) or ""
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    migrated_rows.append((
                        conversation_id,
                        turn,
                        turn_id,
                        row["ts"],
                        row["recorded_at"],
                        data_json,
                    ))
                conn.executescript("""
                    DROP TABLE IF EXISTS request_captures_new;
                    CREATE TABLE request_captures_new (
                        conversation_id TEXT NOT NULL DEFAULT '',
                        turn INTEGER NOT NULL,
                        turn_id TEXT NOT NULL DEFAULT '',
                        ts TEXT NOT NULL,
                        recorded_at REAL NOT NULL,
                        data_json TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, turn, turn_id)
                    );
                    DROP TABLE request_captures;
                    ALTER TABLE request_captures_new RENAME TO request_captures;
                """)
                if migrated_rows:
                    conn.executemany(
                        """INSERT OR REPLACE INTO request_captures
                        (conversation_id, turn, turn_id, ts, recorded_at, data_json)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        migrated_rows,
                    )
        except sqlite3.OperationalError:
            pass
        # D1: migrate role→verb — drop old schema if it has the 'role' column
        try:
            cur = conn.execute("PRAGMA table_info(facts)")
            cols = [row[1] for row in cur.fetchall()]
            if cols and "role" in cols and "verb" not in cols:
                conn.execute("DROP TABLE IF EXISTS fact_tags")
                conn.execute("DROP TABLE IF EXISTS facts")
        except sqlite3.OperationalError:
            pass
        # D1: facts + fact_tags tables
        conn.executescript("""
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
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_verb ON facts(verb);
            CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
            CREATE INDEX IF NOT EXISTS idx_facts_subject_verb ON facts(subject, verb);
            CREATE INDEX IF NOT EXISTS idx_facts_segment_ref ON facts(segment_ref);
            CREATE INDEX IF NOT EXISTS idx_facts_conversation_id ON facts(conversation_id);

            CREATE TABLE IF NOT EXISTS fact_tags (
                fact_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (fact_id, tag),
                FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_fact_tags_tag ON fact_tags(tag);

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
            CREATE INDEX IF NOT EXISTS idx_fact_links_source ON fact_links(source_fact_id);
            CREATE INDEX IF NOT EXISTS idx_fact_links_target ON fact_links(target_fact_id);
            CREATE INDEX IF NOT EXISTS idx_fact_links_type ON fact_links(relation_type);
        """)
        # D1: migrate — add fact_type column to existing facts tables
        try:
            conn.execute("SELECT fact_type FROM facts LIMIT 1")
        except Exception:
            conn.execute("ALTER TABLE facts ADD COLUMN fact_type TEXT NOT NULL DEFAULT 'personal'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_fact_type ON facts(fact_type)")
        try:
            conn.execute("SELECT session_date FROM facts LIMIT 1")
        except Exception:
            conn.execute("ALTER TABLE facts ADD COLUMN session_date TEXT NOT NULL DEFAULT ''")
        # Cascade delete facts when parent segment is deleted
        try:
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS segments_facts_ad AFTER DELETE ON segments BEGIN
                    DELETE FROM facts WHERE segment_ref = old.ref;
                END;
            """)
        except sqlite3.OperationalError:
            pass
        # FTS index on facts (searches subject, verb, object, what)
        try:
            # Check if FTS table already exists before creating — rebuild is
            # only needed on first creation when facts table already has rows.
            fts_existed = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts_fts'"
            ).fetchone() is not None
            conn.executescript(FACTS_FTS_SQL)
            conn.executescript(FACTS_FTS_TRIGGER_SQL)
            if not fts_existed:
                fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
                if fact_count > 0:
                    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        except sqlite3.OperationalError:
            pass
        # Tool output storage for intercepted large tool_result blocks
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tool_outputs (
                ref TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                command TEXT NOT NULL DEFAULT '',
                turn INTEGER NOT NULL,
                content TEXT NOT NULL,
                original_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS tool_outputs_fts
                USING fts5(content, content=tool_outputs, content_rowid=rowid);
        """)
        try:
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS tool_outputs_ai AFTER INSERT ON tool_outputs BEGIN
                    INSERT INTO tool_outputs_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS tool_outputs_ad AFTER DELETE ON tool_outputs BEGIN
                    INSERT INTO tool_outputs_fts(tool_outputs_fts, rowid, content)
                        VALUES('delete', old.rowid, old.content);
                END;
            """)
        except sqlite3.OperationalError:
            pass
        # Migration: add conversation_id to tag_summaries (compound PK).
        # Old schema had tag as sole PK — two conversations with the same tag
        # would overwrite each other's summaries.
        try:
            conn.execute("SELECT conversation_id FROM tag_summaries LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist — migrate
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tag_summaries_new (
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
                INSERT OR IGNORE INTO tag_summaries_new
                    (tag, conversation_id, summary, description, code_refs, summary_tokens,
                     source_segment_refs, source_turn_numbers, source_canonical_turn_ids,
                     covers_through_turn, covers_through_canonical_turn_id, generated_by_turn_id,
                     created_at, updated_at)
                SELECT tag, '', summary, description, '[]', summary_tokens,
                       source_segment_refs, source_turn_numbers, '[]', covers_through_turn, '', '',
                       created_at, updated_at
                FROM tag_summaries;
                DROP TABLE tag_summaries;
                ALTER TABLE tag_summaries_new RENAME TO tag_summaries;
            """)
        try:
            conn.execute("SELECT source_canonical_turn_ids FROM tag_summaries LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE tag_summaries ADD COLUMN source_canonical_turn_ids TEXT NOT NULL DEFAULT '[]'"
            )
        try:
            conn.execute("SELECT covers_through_canonical_turn_id FROM tag_summaries LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE tag_summaries ADD COLUMN covers_through_canonical_turn_id TEXT NOT NULL DEFAULT ''"
            )
        # Request capture persistence for proxy dashboard
        conn.executescript("""
CREATE TABLE IF NOT EXISTS request_captures (
    conversation_id TEXT NOT NULL DEFAULT '',
    turn INTEGER NOT NULL,
    turn_id TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn, turn_id)
);
        """)
        # Tag summary FTS for BM25 retrieval scoring
        try:
            conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS tag_summaries_fts
                    USING fts5(summary, content=tag_summaries, content_rowid=rowid);
            """)
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS tag_summaries_fts_ai AFTER INSERT ON tag_summaries BEGIN
                    INSERT INTO tag_summaries_fts(rowid, summary) VALUES (new.rowid, new.summary);
                END;
                CREATE TRIGGER IF NOT EXISTS tag_summaries_fts_ad AFTER DELETE ON tag_summaries BEGIN
                    INSERT INTO tag_summaries_fts(tag_summaries_fts, rowid, summary)
                        VALUES('delete', old.rowid, old.summary);
                END;
                CREATE TRIGGER IF NOT EXISTS tag_summaries_fts_au AFTER UPDATE ON tag_summaries BEGIN
                    INSERT INTO tag_summaries_fts(tag_summaries_fts, rowid, summary)
                        VALUES('delete', old.rowid, old.summary);
                    INSERT INTO tag_summaries_fts(rowid, summary) VALUES (new.rowid, new.summary);
                END;
            """)
        except sqlite3.OperationalError:
            pass
        # Backfill tag_summaries_fts if empty but tag_summaries has data
        try:
            fts_count = conn.execute("SELECT count(*) FROM tag_summaries_fts").fetchone()[0]
            if fts_count == 0:
                ts_count = conn.execute("SELECT count(*) FROM tag_summaries").fetchone()[0]
                if ts_count > 0:
                    conn.execute("INSERT INTO tag_summaries_fts(tag_summaries_fts) VALUES('rebuild')")
                    conn.commit()
        except Exception as _fts_err:
            logger.warning("tag_summaries_fts rebuild failed: %s", _fts_err)
        # Tag summary embeddings for retrieval scoring
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tag_summary_embeddings (
                tag TEXT NOT NULL,
                conversation_id TEXT NOT NULL DEFAULT '',
                embedding_json TEXT NOT NULL,
                PRIMARY KEY (tag, conversation_id)
            );
        """)
        # Tool call persistence for dashboard
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                request_turn INTEGER NOT NULL,
                round INTEGER NOT NULL,
                group_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_input TEXT NOT NULL,
                tool_result TEXT NOT NULL,
                result_length INTEGER NOT NULL,
                duration_ms REAL NOT NULL,
                found BOOLEAN,
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_calls_conv ON tool_calls(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_group ON tool_calls(group_id);
        """)
        # Request context persistence for dashboard recall page
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS request_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        """)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS request_turn_counters (
                conversation_id TEXT PRIMARY KEY,
                next_request_turn INTEGER NOT NULL
            );
        """)
        # Turn / Segment ↔ Tool Output linkage (join tables)
        conn.executescript("""
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
        """)
        # Chain snapshots for turn chain collapse
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chain_snapshots (
                ref TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                chain_json TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                tool_output_refs TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        # Media output metadata for compressed images
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS media_outputs (
                ref TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                media_type TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                original_bytes INTEGER NOT NULL,
                compressed_bytes INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (conversation_id, ref)
            );
        """)
        try:
            self._normalize_request_turn_sequences(conn)
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
            self._ensure_canonical_turn_schema(conn)
            self._ensure_compaction_scoping_columns(conn)
            self._ensure_canonical_turn_views(conn)
        except Exception:
            logger.warning("canonical turn bootstrap failed", exc_info=True)
        conn.commit()
        self._repair_fts_if_needed(conn)

    def _ensure_canonical_turn_views(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP VIEW IF EXISTS canonical_turns_ordinal")
        conn.execute(
            """CREATE VIEW canonical_turns_ordinal AS
               SELECT
                   ct.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY ct.conversation_id
                       ORDER BY ct.sort_key, ct.first_seen_at, ct.canonical_turn_id
                   ) - 1 AS turn_number
               FROM canonical_turns ct"""
        )

    @staticmethod
    def _add_column_if_missing(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """Add a column with race-safety.

        Multiple worker processes opening the same SQLite file during
        startup all execute ``_ensure_canonical_turn_schema`` against the
        same DB. The naive ``PRAGMA table_info`` + ``ALTER TABLE ADD
        COLUMN`` pattern has a classic TOCTOU window: two workers both
        see the column missing, both try to add it, the second one
        raises ``sqlite3.OperationalError: duplicate column name``.
        The migration is idempotent at the DB level (the column is
        either there or not) but the noisy traceback at every startup
        masks real errors.

        Swallow the duplicate-column case narrowly (by sqlite3 error
        message) — any other ``OperationalError`` propagates so real
        bugs stay visible.
        """
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

    def _ensure_canonical_turn_schema(self, conn: sqlite3.Connection) -> None:
        pragma_rows = conn.execute("PRAGMA table_info(canonical_turns)").fetchall()
        by_name = {row["name"]: row for row in pragma_rows}
        if "turn_group_number" not in by_name:
            self._add_column_if_missing(
                conn,
                "canonical_turns",
                "turn_group_number",
                "INTEGER NOT NULL DEFAULT -1",
            )
            pragma_rows = conn.execute("PRAGMA table_info(canonical_turns)").fetchall()
            by_name = {row["name"]: row for row in pragma_rows}
        lifecycle_columns = ("tagged_at", "compacted_at", "first_seen_at", "last_seen_at")
        needs_rebuild = any(
            name in by_name and int(by_name[name]["notnull"] or 0) == 1
            for name in lifecycle_columns
        )
        if needs_rebuild:
            conn.executescript(
                """
                BEGIN IMMEDIATE;
                DROP TABLE IF EXISTS canonical_turns_new;
                CREATE TABLE canonical_turns_new (
                    canonical_turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    turn_group_number INTEGER NOT NULL DEFAULT -1,
                    sort_key REAL NOT NULL,
                    turn_hash TEXT NOT NULL,
                    hash_version INTEGER NOT NULL DEFAULT 1,
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
                    source_batch_id TEXT,
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    UNIQUE (conversation_id, sort_key)
                );
                INSERT INTO canonical_turns_new
                SELECT
                    canonical_turn_id,
                    conversation_id,
                    COALESCE(turn_group_number, -1),
                    sort_key,
                    turn_hash,
                    hash_version,
                    normalized_user_text,
                    normalized_assistant_text,
                    user_content,
                    assistant_content,
                    user_raw_content,
                    assistant_raw_content,
                    primary_tag,
                    tags_json,
                    session_date,
                    sender,
                    fact_signals_json,
                    code_refs_json,
                    NULLIF(tagged_at, ''),
                    NULLIF(compacted_at, ''),
                    NULLIF(first_seen_at, ''),
                    NULLIF(last_seen_at, ''),
                    source_batch_id,
                    created_at,
                    updated_at
                FROM canonical_turns;
                DROP TABLE canonical_turns;
                ALTER TABLE canonical_turns_new RENAME TO canonical_turns;
                CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_order
                    ON canonical_turns (conversation_id, sort_key);
                CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_hash
                    ON canonical_turns (conversation_id, turn_hash);
                COMMIT;
                """
            )
            pragma_rows = conn.execute("PRAGMA table_info(canonical_turns)").fetchall()
            by_name = {row["name"]: row for row in pragma_rows}
        else:
            for column in lifecycle_columns:
                try:
                    conn.execute(f"UPDATE canonical_turns SET {column} = NULL WHERE {column} = ''")
                except sqlite3.OperationalError:
                    pass
        # Progress-tracking columns for the DB-derived progress model:
        #   covered_ingestible_entries — how many ingestible payload entries
        #     this canonical row represents (set at insert time). The progress
        #     denominator is SUM(covered_ingestible_entries).
        #   tagged_at — ISO timestamp set when the tagger enriches the row.
        #     The progress numerator is
        #     SUM(covered_ingestible_entries WHERE tagged_at IS NOT NULL).
        # The two partial indexes below make each SUM path an index-only scan.
        if "covered_ingestible_entries" not in by_name:
            self._add_column_if_missing(
                conn,
                "canonical_turns",
                "covered_ingestible_entries",
                "INTEGER NOT NULL DEFAULT 1",
            )
        if "tagged_at" not in by_name:
            self._add_column_if_missing(
                conn,
                "canonical_turns",
                "tagged_at",
                "TEXT NULL",
            )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_untagged
                   ON canonical_turns (conversation_id, sort_key)
                   WHERE tagged_at IS NULL"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_canonical_turns_conv_tagged
                   ON canonical_turns (conversation_id, tagged_at)
                   WHERE tagged_at IS NOT NULL"""
        )

    def _ensure_compaction_scoping_columns(self, conn: sqlite3.Connection) -> None:
        """Add operation_id / compaction_operation_id columns used by the
        compaction-resume-parity takeover path. Idempotent and race-safe
        via ``_add_column_if_missing``. Existing rows backfill to the
        zero-UUID sentinel ``00000000-0000-0000-0000-000000000000`` per
        the approved spec (line 61-63 + rollout line 397-401). Cleanup
        predicates scope on ``operation_id = :target`` so zero-UUID rows
        are invisible to cleanup — they never match a real compaction's
        UUID.
        """
        zero_uuid = "00000000-0000-0000-0000-000000000000"
        for table, column, definition in (
            ("segments", "operation_id", "TEXT"),
            ("facts", "operation_id", "TEXT"),
            ("tag_summaries", "operation_id", "TEXT"),
            ("tag_summary_embeddings", "operation_id", "TEXT"),
            ("canonical_turns", "compaction_operation_id", "TEXT"),
        ):
            self._add_column_if_missing(conn, table, column, definition)
            # Backfill pre-migration rows to the zero-UUID sentinel.
            # Idempotent: UPDATE ... WHERE <col> IS NULL matches zero rows
            # on second run.
            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE {column} IS NULL",
                (zero_uuid,),
            )

    def _lookup_canonical_turn_id_for_ordinal(self, conversation_id: str, turn_number: int) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT canonical_turn_id
               FROM canonical_turns_ordinal
               WHERE conversation_id = ? AND turn_number = ?""",
            (conversation_id, turn_number),
        ).fetchone()
        return str(row["canonical_turn_id"]) if row else None

    def _lookup_ordinal_for_canonical_turn_id(self, conversation_id: str, canonical_turn_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT turn_number
               FROM canonical_turns_ordinal
               WHERE conversation_id = ? AND canonical_turn_id = ?""",
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
               WHERE conversation_id = ?
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

    def _repair_fts_if_needed(self, conn: sqlite3.Connection) -> None:
        """Check FTS indexes and rebuild only if corrupted.

        FTS5 indexes can become inconsistent after a hard kill (SIGKILL,
        power loss) because the FTS internal structures aren't covered by
        SQLite's main integrity check.  A quick ``integrity-check`` on each
        FTS table detects this; a full ``rebuild`` fixes it.
        """
        for fts_table in ("segments_fts", "segments_fts_full"):
            try:
                conn.execute(
                    f"INSERT INTO {fts_table}({fts_table}) VALUES('integrity-check')"
                )
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc):
                    logger.debug("FTS integrity-check skipped for %s: %s", fts_table, exc)
                    continue
                logger.warning("FTS index %s corrupted — rebuilding", fts_table)
                try:
                    conn.execute(
                        f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')"
                    )
                    conn.commit()
                    logger.info("FTS index %s rebuilt successfully", fts_table)
                except sqlite3.DatabaseError as exc2:
                    logger.error("FTS rebuild failed for %s: %s", fts_table, exc2)
            except sqlite3.DatabaseError:
                logger.warning("FTS index %s corrupted — rebuilding", fts_table)
                try:
                    conn.execute(
                        f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')"
                    )
                    conn.commit()
                    logger.info("FTS index %s rebuilt successfully", fts_table)
                except sqlite3.DatabaseError as exc:
                    logger.error("FTS rebuild failed for %s: %s", fts_table, exc)

    def _get_tags_for_ref(self, ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag FROM segment_tags WHERE segment_ref = ? ORDER BY tag",
            (ref,),
        ).fetchall()
        return [r["tag"] for r in rows]

    def _batch_get_tags(self, refs: list[str]) -> dict[str, list[str]]:
        """Fetch tags for multiple segment refs in a single query.

        Returns a dict mapping each ref to its sorted tag list.
        Refs with no tags will map to an empty list.
        """
        if not refs:
            return {}
        conn = self._get_conn()
        placeholders = ",".join("?" * len(refs))
        rows = conn.execute(
            f"SELECT segment_ref, tag FROM segment_tags WHERE segment_ref IN ({placeholders}) ORDER BY segment_ref, tag",
            refs,
        ).fetchall()
        result: dict[str, list[str]] = {ref: [] for ref in refs}
        for row in rows:
            result[row["segment_ref"]].append(row["tag"])
        return result

    def _allocate_request_turn(self, conn: sqlite3.Connection, conversation_id: str) -> int:
        row = conn.execute(
            """INSERT INTO request_turn_counters (conversation_id, next_request_turn)
               VALUES (?, 1)
               ON CONFLICT(conversation_id)
               DO UPDATE SET next_request_turn = request_turn_counters.next_request_turn + 1
               RETURNING next_request_turn""",
            (conversation_id,),
        ).fetchone()
        return int((row[0] if row else 0) or 0)

    def _bump_request_turn_counter(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        request_turn: int,
    ) -> None:
        conn.execute(
            """INSERT INTO request_turn_counters (conversation_id, next_request_turn)
               VALUES (?, ?)
               ON CONFLICT(conversation_id)
               DO UPDATE SET next_request_turn = MAX(request_turn_counters.next_request_turn, excluded.next_request_turn)""",
            (conversation_id, int(request_turn)),
        )

    def _normalize_request_turn_sequences(self, conn: sqlite3.Connection) -> None:
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
            grouped_contexts[conversation_id].append({
                "id": int(row["id"]),
                "request_turn": seq,
                "timestamp": _parse_sequence_timestamp(row["timestamp"]),
            })
            if int(row["request_turn"] or 0) != seq:
                context_updates.append((seq, int(row["id"])))

        if context_updates:
            conn.executemany(
                "UPDATE request_context SET request_turn = ? WHERE id = ?",
                context_updates,
            )

        tool_rows = conn.execute(
            "SELECT id, conversation_id, request_turn, timestamp FROM tool_calls "
            "ORDER BY conversation_id, id"
        ).fetchall()
        tool_updates: list[tuple[int, int]] = []
        for row in tool_rows:
            contexts = grouped_contexts.get(row["conversation_id"])
            if not contexts:
                continue
            tool_ts = _parse_sequence_timestamp(row["timestamp"])
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
            if int(row["request_turn"] or 0) != assigned_turn:
                tool_updates.append((assigned_turn, int(row["id"])))

        if tool_updates:
            conn.executemany(
                "UPDATE tool_calls SET request_turn = ? WHERE id = ?",
                tool_updates,
            )

        counter_rows = [
            (conversation_id, contexts[-1]["request_turn"])
            for conversation_id, contexts in grouped_contexts.items()
            if contexts
        ]
        if counter_rows:
            conn.executemany(
                """INSERT INTO request_turn_counters (conversation_id, next_request_turn)
                   VALUES (?, ?)
                   ON CONFLICT(conversation_id)
                   DO UPDATE SET next_request_turn = MAX(request_turn_counters.next_request_turn, excluded.next_request_turn)""",
                counter_rows,
            )

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
        metadata_json = json.dumps(metadata_dict)

        guard_all = (
            operation_id is not None
            and owner_worker_id is not None
            and lifecycle_epoch is not None
        )

        conn.execute("BEGIN IMMEDIATE")
        try:
            if guard_all:
                # INSERT-SELECT form: writes zero rows if the compaction_operation
                # row no longer matches (status != 'running', owner mismatch, etc).
                cur = conn.execute(
                    """INSERT OR REPLACE INTO segments
                    (ref, conversation_id, primary_tag, summary, full_text, messages_json,
                     metadata_json, summary_tokens, full_tokens, compression_ratio,
                     compaction_model, created_at, start_timestamp, end_timestamp,
                     operation_id)
                    SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                      FROM compaction_operation
                     WHERE operation_id = ?
                       AND conversation_id = ?
                       AND status = 'running'
                       AND owner_worker_id = ?
                       AND lifecycle_epoch = ?""",
                    (
                        segment.ref,
                        segment.conversation_id,
                        primary_tag,
                        summary_text,
                        full_text,
                        json.dumps(segment.messages, default=str),
                        metadata_json,
                        segment.summary_tokens,
                        segment.full_tokens,
                        segment.compression_ratio,
                        segment.compaction_model,
                        _dt_to_str(segment.created_at),
                        _dt_to_str(segment.start_timestamp),
                        _dt_to_str(segment.end_timestamp),
                        operation_id,
                        # WHERE clause params:
                        operation_id,
                        segment.conversation_id,
                        owner_worker_id,
                        lifecycle_epoch,
                    ),
                )
                if (cur.rowcount or 0) == 0:
                    conn.execute("ROLLBACK")
                    raise CompactionLeaseLost(
                        operation_id=operation_id,
                        write_site="store_segment",
                    )
            else:
                # Legacy unconditional path — existing callers and test harnesses.
                conn.execute(
                    """INSERT OR REPLACE INTO segments
                    (ref, conversation_id, primary_tag, summary, full_text, messages_json,
                     metadata_json, summary_tokens, full_tokens, compression_ratio,
                     compaction_model, created_at, start_timestamp, end_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        segment.ref,
                        segment.conversation_id,
                        primary_tag,
                        summary_text,
                        full_text,
                        json.dumps(segment.messages, default=str),
                        metadata_json,
                        segment.summary_tokens,
                        segment.full_tokens,
                        segment.compression_ratio,
                        segment.compaction_model,
                        _dt_to_str(segment.created_at),
                        _dt_to_str(segment.start_timestamp),
                        _dt_to_str(segment.end_timestamp),
                    ),
                )

            # Update tags (same for both paths)
            conn.execute("DELETE FROM segment_tags WHERE segment_ref = ?", (segment.ref,))
            for tag in segment.tags:
                conn.execute(
                    "INSERT INTO segment_tags (segment_ref, tag) VALUES (?, ?)",
                    (segment.ref, tag),
                )

            conn.execute("COMMIT")
        except CompactionLeaseLost:
            # Already rolled back above; re-raise without swallowing.
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return segment.ref

    def get_segment(self, ref: str, conversation_id: str | None = None) -> StoredSegment | None:
        conn = self._get_conn()
        if conversation_id is not None:
            row = conn.execute(
                "SELECT * FROM segments WHERE ref = ? AND conversation_id = ?",
                (ref, conversation_id),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM segments WHERE ref = ?", (ref,)).fetchone()
        if not row:
            return None
        tags = self._get_tags_for_ref(ref)
        return _row_to_segment(row, tags)

    def get_summary(self, ref: str, conversation_id: str | None = None) -> StoredSummary | None:
        conn = self._get_conn()
        if conversation_id is not None:
            row = conn.execute(
                "SELECT * FROM segments WHERE ref = ? AND conversation_id = ?",
                (ref, conversation_id),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM segments WHERE ref = ?", (ref,)).fetchone()
        if not row:
            return None
        tags = self._get_tags_for_ref(ref)
        return _row_to_summary(row, tags)

    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        if not tags:
            return []

        conn = self._get_conn()
        placeholders = ",".join("?" * len(tags))

        query = f"""
            SELECT s.*, COUNT(st.tag) as overlap_count
            FROM segments s
            JOIN segment_tags st ON s.ref = st.segment_ref
            WHERE st.tag IN ({placeholders})
        """
        params: list = list(tags)

        if conversation_id is not None:
            query += " AND s.conversation_id = ?"
            params.append(conversation_id)

        if before:
            query += " AND s.created_at < ?"
            params.append(_dt_to_str(before))
        if after:
            query += " AND s.created_at > ?"
            params.append(_dt_to_str(after))

        query += f"""
            GROUP BY s.ref
            HAVING overlap_count >= ?
            ORDER BY overlap_count DESC, s.created_at DESC
            LIMIT ?
        """
        params.extend([min_overlap, limit])

        rows = conn.execute(query, params).fetchall()

        refs = [row["ref"] for row in rows]
        tags_map = self._batch_get_tags(refs)
        results = []
        for row in rows:
            results.append(_row_to_summary(row, tags_map[row["ref"]]))
        return results

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        conn = self._get_conn()

        # Build optional conversation_id filter fragment
        _conv_filter = ""
        _conv_params: list = []
        if conversation_id is not None:
            _conv_filter = " AND s.conversation_id = ?"
            _conv_params = [conversation_id]

        # Try FTS5 first
        try:
            if tags:
                placeholders = ",".join("?" * len(tags))
                rows = conn.execute(
                    f"""SELECT s.* FROM segments_fts fts
                    JOIN segments s ON s.ref = fts.ref
                    JOIN segment_tags st ON s.ref = st.segment_ref
                    WHERE segments_fts MATCH ?
                    AND st.tag IN ({placeholders})
                    {_conv_filter}
                    GROUP BY s.ref
                    ORDER BY rank
                    LIMIT ?""",
                    [_sanitize_fts_query(query), *tags, *_conv_params, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""SELECT s.* FROM segments_fts fts
                    JOIN segments s ON s.ref = fts.ref
                    WHERE segments_fts MATCH ?
                    {_conv_filter}
                    ORDER BY rank
                    LIMIT ?""",
                    [_sanitize_fts_query(query), *_conv_params, limit],
                ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 not available, fall back to LIKE
            # Escape LIKE wildcards in user input to prevent pattern injection
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like_query = f"%{escaped}%"
            if tags:
                placeholders = ",".join("?" * len(tags))
                rows = conn.execute(
                    f"""SELECT DISTINCT s.* FROM segments s
                    JOIN segment_tags st ON s.ref = st.segment_ref
                    WHERE s.summary LIKE ? ESCAPE '\\'
                    AND st.tag IN ({placeholders})
                    {_conv_filter}
                    LIMIT ?""",
                    [like_query, *tags, *_conv_params, limit],
                ).fetchall()
            else:
                # No JOIN — use conversation_id directly on segments table
                _conv_filter_simple = ""
                if conversation_id is not None:
                    _conv_filter_simple = " AND conversation_id = ?"
                rows = conn.execute(
                    f"SELECT * FROM segments WHERE summary LIKE ? ESCAPE '\\'{_conv_filter_simple} LIMIT ?",
                    [like_query, *_conv_params, limit],
                ).fetchall()

        refs = [row["ref"] for row in rows]
        tags_map = self._batch_get_tags(refs)
        results = []
        for row in rows:
            results.append(_row_to_summary(row, tags_map[row["ref"]]))
        return results

    def search_full_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        conn = self._get_conn()
        results: list[QuoteResult] = []

        # Build optional conversation_id filter fragment
        _conv_filter = ""
        _conv_params: list = []
        if conversation_id is not None:
            _conv_filter = " AND s.conversation_id = ?"
            _conv_params = [conversation_id]

        # Try FTS5 first (with snippet extraction)
        _sc = self.search_config
        _fts_chars = _sc.fts_snippet_chars if _sc else 500
        _excerpt_chars = _sc.excerpt_context_chars if _sc else 200
        try:
            rows = conn.execute(
                f"""SELECT fts.ref, s.primary_tag, s.metadata_json,
                          snippet(segments_fts_full, 1, '>>>', '<<<', '...', {_fts_chars}),
                          s.created_at
                   FROM segments_fts_full fts
                   JOIN segments s ON s.ref = fts.ref
                   WHERE segments_fts_full MATCH ?
                   {_conv_filter}
                   ORDER BY rank
                   LIMIT ?""",
                [_sanitize_fts_query(query), *_conv_params, limit],
            ).fetchall()
            fts_refs = [row[0] for row in rows]
            fts_tags_map = self._batch_get_tags(fts_refs)
            for row in rows:
                meta = json.loads(row[2]) if row[2] else {}
                results.append(QuoteResult(
                    text=row[3],
                    tag=row[1],
                    segment_ref=row[0],
                    tags=fts_tags_map[row[0]],
                    match_type="fts",
                    session_date=meta.get("session_date", ""),
                ))
            return results
        except sqlite3.OperationalError:
            pass

        # Fallback: LIKE search on full_text with manual excerpt
        # Escape LIKE wildcards in user input to prevent pattern injection
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_query = f"%{escaped}%"
        _conv_filter_like = ""
        _conv_params_like: list = []
        if conversation_id is not None:
            _conv_filter_like = " AND conversation_id = ?"
            _conv_params_like = [conversation_id]
        rows = conn.execute(
            f"""SELECT ref, primary_tag, full_text, metadata_json, created_at FROM segments
               WHERE full_text LIKE ? ESCAPE '\\'
               {_conv_filter_like}
               LIMIT ?""",
            [like_query, *_conv_params_like, limit],
        ).fetchall()
        like_refs = [row[0] for row in rows]
        like_tags_map = self._batch_get_tags(like_refs)
        for row in rows:
            excerpt = _extract_excerpt(row[2], query, context_chars=_excerpt_chars)
            meta = json.loads(row[3]) if row[3] else {}
            results.append(QuoteResult(
                text=excerpt,
                tag=row[1],
                segment_ref=row[0],
                tags=like_tags_map[row[0]],
                match_type="like",
                session_date=meta.get("session_date", ""),
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
                 WHERE (user_content LIKE ? OR assistant_content LIKE ?)"""
        params: list[object] = [pattern, pattern]
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY turn_number DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

        results = []
        _sc = getattr(self, "search_config", None)
        _ctx = _sc.excerpt_context_chars if _sc else 200
        for row in rows:
            turn = row["turn_number"] if isinstance(row, sqlite3.Row) else row[0]
            u = (row["user_content"] if isinstance(row, sqlite3.Row) else row[1]) or ""
            a = (row["assistant_content"] if isinstance(row, sqlite3.Row) else row[2]) or ""
            primary_tag = (row["primary_tag"] if isinstance(row, sqlite3.Row) else row[4]) or "_general"
            tags_json = (row["tags_json"] if isinstance(row, sqlite3.Row) else row[5]) or "[]"
            session_date = (row["session_date"] if isinstance(row, sqlite3.Row) else row[6]) or ""
            try:
                tags = json.loads(tags_json or "[]")
            except Exception:
                tags = []
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
                segment_ref=f"canonical_turn_{row['canonical_turn_id'] or turn}",
                tags=list(tags or []),
                match_type="full_text_search",
                session_date=session_date,
                source_scope="turn",
                turn_number=turn,
                matched_side=matched_side,
            ))
        return results

    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute("""
                SELECT st.tag,
                       COUNT(DISTINCT st.segment_ref) as usage_count,
                       COALESCE(SUM(s.full_tokens), 0) as total_full_tokens,
                       COALESCE(SUM(s.summary_tokens), 0) as total_summary_tokens,
                       MIN(s.created_at) as oldest,
                       MAX(s.created_at) as newest
                FROM segment_tags st
                JOIN segments s ON s.ref = st.segment_ref
                WHERE s.conversation_id = ?
                GROUP BY st.tag
                ORDER BY usage_count DESC
            """, (conversation_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT st.tag,
                       COUNT(DISTINCT st.segment_ref) as usage_count,
                       COALESCE(SUM(s.full_tokens), 0) as total_full_tokens,
                       COALESCE(SUM(s.summary_tokens), 0) as total_summary_tokens,
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
                total_full_tokens=row["total_full_tokens"],
                total_summary_tokens=row["total_summary_tokens"],
                oldest_segment=_str_to_dt(row["oldest"]) if row["oldest"] else None,
                newest_segment=_str_to_dt(row["newest"]) if row["newest"] else None,
            )
            for row in rows
        ]

    def get_conversation_stats(self) -> list[ConversationStats]:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT s.conversation_id,
                   COUNT(*) as segment_count,
                   COALESCE(SUM(s.full_tokens), 0) as total_full_tokens,
                   COALESCE(SUM(s.summary_tokens), 0) as total_summary_tokens,
                   MIN(s.created_at) as oldest,
                   MAX(s.created_at) as newest,
                   s.compaction_model
            FROM segments s
            WHERE s.conversation_id != ''
            GROUP BY s.conversation_id
            ORDER BY newest DESC
        """).fetchall()

        # Fetch provider from engine_state blobs (one query for all)
        provider_map: dict[str, str] = {}
        try:
            es_rows = conn.execute(
                "SELECT conversation_id, turn_tag_entries FROM engine_state"
            ).fetchall()
            for es_row in es_rows:
                try:
                    blob = json.loads(es_row["turn_tag_entries"])
                    if isinstance(blob, dict) and blob.get("provider"):
                        provider_map[es_row["conversation_id"]] = blob["provider"]
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass

        # Batch-fetch distinct tags per conversation (avoids N+1)
        conv_ids = [row["conversation_id"] for row in rows]
        tags_by_conv: dict[str, list[str]] = {cid: [] for cid in conv_ids}
        if conv_ids:
            placeholders = ",".join("?" for _ in conv_ids)
            tag_rows = conn.execute(f"""
                SELECT s.conversation_id, st.tag
                FROM segment_tags st
                JOIN segments s ON s.ref = st.segment_ref
                WHERE s.conversation_id IN ({placeholders})
                GROUP BY s.conversation_id, st.tag
                ORDER BY st.tag
            """, conv_ids).fetchall()
            for tr in tag_rows:
                tags_by_conv[tr["conversation_id"]].append(tr["tag"])

        results = []
        for row in rows:
            total_full = row["total_full_tokens"]
            total_summary = row["total_summary_tokens"]
            ratio = round(total_summary / total_full, 3) if total_full > 0 else 0.0

            results.append(ConversationStats(
                conversation_id=row["conversation_id"],
                segment_count=row["segment_count"],
                total_full_tokens=total_full,
                total_summary_tokens=total_summary,
                compression_ratio=ratio,
                distinct_tags=tags_by_conv.get(row["conversation_id"], []),
                oldest_segment=_str_to_dt(row["oldest"]) if row["oldest"] else None,
                newest_segment=_str_to_dt(row["newest"]) if row["newest"] else None,
                compaction_model=row["compaction_model"] or "",
                provider=provider_map.get(row["conversation_id"], ""),
            ))

        return results

    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[StoredSegment]:
        conn = self._get_conn()
        query = "SELECT * FROM segments"
        params: list[object] = []
        if conversation_id is not None:
            query += " WHERE conversation_id = ?"
            params.append(conversation_id)
        query += " ORDER BY created_at DESC"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        if not rows:
            return []
        refs = [row["ref"] for row in rows]
        placeholders = ",".join("?" for _ in refs)
        tags_rows = conn.execute(
            f"SELECT segment_ref, tag FROM segment_tags WHERE segment_ref IN ({placeholders})",
            refs,
        ).fetchall()
        tags_map: dict[str, list[str]] = {ref: [] for ref in refs}
        for row in tags_rows:
            tags_map.setdefault(row["segment_ref"], []).append(row["tag"])
        return [_row_to_segment(row, tags_map.get(row["ref"], [])) for row in rows]

    def get_tag_aliases(self, conversation_id: str | None = None) -> dict[str, str]:
        conn = self._get_conn()
        params: list[str] = []
        query = "SELECT alias, canonical, conversation_id FROM tag_aliases"
        if conversation_id:
            query += " WHERE conversation_id IN ('', ?)"
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
            """INSERT OR REPLACE INTO tag_aliases
            (alias, conversation_id, canonical) VALUES (?, ?, ?)""",
            (alias, conversation_id or "", canonical),
        )
        conn.commit()

    def delete_tag_aliases_for_conversation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM tag_aliases WHERE conversation_id = ?",
            (conversation_id,),
        )
        conn.commit()
        return int(cursor.rowcount or 0)

    def delete_segment(self, ref: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM segments WHERE ref = ?", (ref,))
        conn.commit()
        return cursor.rowcount > 0

    def _table_exists(self, conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    def _delete_conversation_rows(
        self,
        conn: sqlite3.Connection,
        table: str,
        conversation_id: str,
    ) -> int:
        if not self._table_exists(conn, table):
            return 0
        cursor = conn.execute(
            f"DELETE FROM {table} WHERE conversation_id = ?",
            (conversation_id,),
        )
        return int(cursor.rowcount or 0)

    def activate_conversation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT generation FROM conversation_lifecycle WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        now = _dt_to_str(datetime.now(timezone.utc))
        if row is None:
            conn.execute(
                """INSERT INTO conversation_lifecycle
                (conversation_id, generation, deleted, updated_at)
                VALUES (?, 0, 0, ?)""",
                (conversation_id, now),
            )
            conn.commit()
            return 0
        generation = int(row[0])
        conn.execute(
            """UPDATE conversation_lifecycle
            SET deleted = 0, updated_at = ?
            WHERE conversation_id = ?""",
            (now, conversation_id),
        )
        conn.commit()
        return generation

    def begin_conversation_deletion(self, conversation_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT generation FROM conversation_lifecycle WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        generation = (int(row[0]) + 1) if row is not None else 1
        conn.execute(
            """INSERT INTO conversation_lifecycle
            (conversation_id, generation, deleted, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                generation = excluded.generation,
                deleted = 1,
                updated_at = excluded.updated_at""",
            (conversation_id, generation, _dt_to_str(datetime.now(timezone.utc))),
        )
        conn.commit()
        return generation

    def get_conversation_generation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT generation FROM conversation_lifecycle WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def is_conversation_generation_current(
        self,
        conversation_id: str,
        generation: int,
    ) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT generation, deleted FROM conversation_lifecycle WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return int(generation or 0) == 0
        return int(row[0]) == int(generation or 0) and not bool(row[1])

    # ------------------------------------------------------------------
    # Conversation row lifecycle (progress-bar redesign `conversations` table)
    # ------------------------------------------------------------------
    # These methods operate on the `conversations` table created in
    # `_ensure_schema` (not the legacy `conversation_lifecycle` table used
    # by activate_conversation/begin_conversation_deletion above). The new
    # table carries `lifecycle_epoch` + `phase` for the progress tracker
    # and delete+resurrect invariants.

    def upsert_conversation(self, *, tenant_id: str, conversation_id: str) -> None:
        """Create the conversations row if missing; otherwise just refresh updated_at.

        Epoch starts at 1 on new rows; never bumped by this method.
        """
        now = utcnow_iso()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, tenant_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    updated_at = excluded.updated_at
                """,
                (conversation_id, tenant_id, now, now),
            )

    def get_lifecycle_epoch(self, conversation_id: str) -> int:
        """Return the current lifecycle_epoch. Raises KeyError if no row exists."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT lifecycle_epoch FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return int(row[0])

    def get_conversation_phase(self, conversation_id: str) -> str:
        """Return the current phase for the conversation.

        Returns one of ``"init" | "ingesting" | "compacting" | "active" |
        "deleted"``. Raises ``KeyError`` if no row exists.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT phase FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return str(row[0])

    def mark_conversation_deleted(self, conversation_id: str) -> None:
        """Admin-flow delete: sets phase='deleted' and stamps deleted_at.

        Called only by the delete endpoint — caller is authoritative; no
        epoch check needed. Raises KeyError if no row exists so callers
        get symmetric signaling with ``increment_lifecycle_epoch_on_resurrect``.
        """
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE conversations
                   SET phase = 'deleted',
                       deleted_at = ?,
                       updated_at = ?
                 WHERE conversation_id = ?
                """,
                (now, now, conversation_id),
            )
            if cur.rowcount == 0:
                raise KeyError(conversation_id)
            conn.execute(
                """
                UPDATE ingestion_episode
                   SET status = 'abandoned',
                       completed_at = COALESCE(completed_at, ?)
                 WHERE conversation_id = ?
                   AND status = 'running'
                """,
                (now, conversation_id),
            )
            conn.execute(
                """
                UPDATE compaction_operation
                   SET status = 'cancelled',
                       completed_at = COALESCE(completed_at, ?)
                 WHERE conversation_id = ?
                   AND status IN ('queued', 'running')
                """,
                (now, conversation_id),
            )

    def increment_lifecycle_epoch_on_resurrect(self, conversation_id: str) -> int:
        """Bump lifecycle_epoch ONLY when phase == 'deleted'.

        Concurrent resurrect calls cannot double-bump: the second caller
        finds the guard fails and returns the current (already-bumped)
        epoch. Raises KeyError if no row exists.
        """
        now = utcnow_iso()
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                UPDATE conversations
                   SET lifecycle_epoch = lifecycle_epoch + 1,
                       phase = 'init',
                       deleted_at = NULL,
                       updated_at = ?
                 WHERE conversation_id = ?
                   AND phase = 'deleted'
                """,
                (now, conversation_id),
            )
            row = conn.execute(
                "SELECT lifecycle_epoch FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is not None:
                new_epoch = int(row[0])
                conn.execute(
                    """
                    UPDATE ingestion_episode
                       SET status = 'abandoned',
                           completed_at = COALESCE(completed_at, ?)
                     WHERE conversation_id = ?
                       AND lifecycle_epoch < ?
                       AND status = 'running'
                    """,
                    (now, conversation_id, new_epoch),
                )
                conn.execute(
                    """
                    UPDATE compaction_operation
                       SET status = 'cancelled',
                           completed_at = COALESCE(completed_at, ?)
                     WHERE conversation_id = ?
                       AND lifecycle_epoch < ?
                       AND status IN ('queued', 'running')
                    """,
                    (now, conversation_id, new_epoch),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        if row is None:
            raise KeyError(conversation_id)
        return int(row[0])

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
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT lifecycle_epoch, phase,
                       last_raw_payload_entries, last_ingestible_payload_entries
                  FROM conversations
                 WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            epoch, phase, last_raw, last_ing = row[0], row[1], row[2], row[3]

            totals = conn.execute(
                """
                SELECT COALESCE(SUM(covered_ingestible_entries), 0),
                       COALESCE(SUM(CASE WHEN tagged_at IS NOT NULL
                                         THEN covered_ingestible_entries ELSE 0 END), 0)
                  FROM canonical_turns
                 WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            total_ing, done_ing = totals[0], totals[1]

            ep_row = conn.execute(
                """
                SELECT episode_id, raw_payload_entries, owner_worker_id, heartbeat_ts
                  FROM ingestion_episode
                 WHERE conversation_id = ?
                   AND lifecycle_epoch = ?
                   AND status = 'running'
                 ORDER BY started_at DESC, episode_id DESC
                 LIMIT 1
                """,
                (conversation_id, epoch),
            ).fetchone()
            active_episode = (
                ActiveEpisodeSnapshot(
                    episode_id=str(ep_row[0]),
                    raw_payload_entries=int(ep_row[1]),
                    owner_worker_id=str(ep_row[2]),
                    heartbeat_ts=str(ep_row[3]),
                )
                if ep_row is not None
                else None
            )

            cop_row = conn.execute(
                """
                SELECT operation_id, phase_name, phase_index, phase_count, status
                  FROM compaction_operation
                 WHERE conversation_id = ?
                   AND lifecycle_epoch = ?
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
                    operation_id=str(cop_row[0]),
                    phase_name=str(cop_row[1]),
                    phase_index=int(cop_row[2]),
                    phase_count=int(cop_row[3]),
                    status=str(cop_row[4]),
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
        exists), ONLY widens ``raw_payload_entries`` via MAX — does NOT
        change ownership or other fields. Idempotent.

        Uses SQLite's partial-index conflict target (requires SQLite 3.38+).
        """
        import uuid
        now = utcnow_iso()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_episode (
                    episode_id, conversation_id, lifecycle_epoch,
                    raw_payload_entries, started_at, status,
                    owner_worker_id, heartbeat_ts
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                ON CONFLICT (conversation_id, lifecycle_epoch) WHERE status = 'running'
                DO UPDATE SET
                    raw_payload_entries =
                        MAX(ingestion_episode.raw_payload_entries, excluded.raw_payload_entries)
                """,
                (
                    str(uuid.uuid4()), conversation_id, lifecycle_epoch,
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
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=lease_ttl_s)
        ).isoformat()
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE ingestion_episode
                   SET owner_worker_id = ?, heartbeat_ts = ?
                 WHERE conversation_id = ? AND status = 'running'
                   AND lifecycle_epoch = ?
                   AND (owner_worker_id = ? OR heartbeat_ts < ?)
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
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE ingestion_episode SET heartbeat_ts = ?
                 WHERE conversation_id = ? AND status = 'running'
                   AND lifecycle_epoch = ?
                   AND owner_worker_id = ?
                   AND lifecycle_epoch = (
                       SELECT lifecycle_epoch FROM conversations
                        WHERE conversation_id = ?
                   )
                """,
                (now, conversation_id, lifecycle_epoch, worker_id, conversation_id),
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
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE ingestion_episode
                   SET status = 'completed', completed_at = ?
                 WHERE conversation_id = ? AND status = 'running'
                   AND lifecycle_epoch = ?
                   AND owner_worker_id = ?
                   AND lifecycle_epoch = (
                       SELECT lifecycle_epoch FROM conversations
                        WHERE conversation_id = ?
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM canonical_turns
                        WHERE conversation_id = ? AND tagged_at IS NULL
                   )
                """,
                (
                    now, conversation_id, lifecycle_epoch, worker_id,
                    conversation_id, conversation_id,
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
    # raises ``sqlite3.IntegrityError`` and the caller is expected to
    # retry/wait. ``claim_compaction_lease`` matches the ingestion-lease
    # pattern (owner-or-stale heartbeat), and the three terminal/phase
    # operations (``advance_compaction_phase``,
    # ``complete_compaction_operation``, ``fail_compaction_operation``)
    # double-scope the epoch filter with a correlated subquery against
    # ``conversations.lifecycle_epoch`` so a stale thread whose
    # conversation was resurrected to a newer epoch is rejected at SQL
    # level.

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

        Raises ``sqlite3.IntegrityError`` (via the partial unique index
        on status IN ('queued','running')) if another active operation
        already exists for this (conversation, epoch). The caller is
        expected to retry or wait.
        """
        import uuid
        op_id = str(uuid.uuid4())
        now = utcnow_iso()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO compaction_operation (
                    operation_id, conversation_id, lifecycle_epoch,
                    phase_index, phase_count, phase_name, status,
                    started_at, owner_worker_id, heartbeat_ts
                ) VALUES (?, ?, ?, 0, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    op_id, conversation_id, lifecycle_epoch,
                    phase_count, phase_name, now, worker_id, now,
                ),
            )
        return op_id

    def claim_compaction_lease(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        lease_ttl_s: float,
    ) -> "CompactionLeaseClaim":
        """Claim the compaction lease for this (conversation, epoch).

        Returns ``CompactionLeaseClaim`` carrying both the claim decision
        AND the previous row's identity, so the takeover path can scope
        cleanup on ``prev_operation_id`` without a second round-trip.

        Atomic flow inside a single transaction:

        1. Read the current `(operation_id, owner_worker_id, heartbeat_ts)`
           of the unique queued/running row at this epoch.
        2. If the caller already owns it OR heartbeat is older than TTL,
           UPDATE to set owner=caller, heartbeat=NOW().
        3. Return claim reflecting pre-update ``prev_operation_id`` /
           ``prev_owner_worker_id`` and whether the UPDATE hit a row.
        """
        from ..types import CompactionLeaseClaim
        import datetime as _dt

        cutoff = (
            _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=lease_ttl_s)
        ).isoformat()
        now = utcnow_iso()
        with self._get_conn() as conn:
            pre = conn.execute(
                """
                SELECT operation_id, owner_worker_id, heartbeat_ts
                  FROM compaction_operation
                 WHERE conversation_id = ? AND status IN ('queued','running')
                   AND lifecycle_epoch = ?
                 ORDER BY started_at DESC
                 LIMIT 1
                """,
                (conversation_id, lifecycle_epoch),
            ).fetchone()
            prev_op = pre["operation_id"] if pre else None
            prev_owner = pre["owner_worker_id"] if pre else None

            cur = conn.execute(
                """
                UPDATE compaction_operation
                   SET owner_worker_id = ?, heartbeat_ts = ?
                 WHERE conversation_id = ? AND status IN ('queued','running')
                   AND lifecycle_epoch = ?
                   AND (owner_worker_id = ? OR heartbeat_ts < ?)
                """,
                (
                    worker_id, now, conversation_id, lifecycle_epoch,
                    worker_id, cutoff,
                ),
            )
            claimed = (cur.rowcount or 0) > 0
            self._commit_if_unlocked(conn)

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
        """Atomic takeover-cleanup transaction. Returns True iff this call
        performed the transition (dead_op was 'running' and we abandoned
        it + inserted new_op). Returns False when the dead_op was already
        abandoned/completed — the caller (or a prior duplicate call) had
        already handled the takeover; skip the new-row INSERT so the
        one-active invariant holds.

        Ordering inside the single transaction:

        1. UPDATE dead_op to 'abandoned'. Use rowcount to decide whether
           this is a fresh takeover (> 0) or an idempotent re-run (== 0).
        2. DELETE scoped partial writes from segments/facts/tag_summaries/
           tag_summary_embeddings. (Idempotent: no-ops on already-absent
           rows. Safe to run even on the idempotent re-run path because
           there's nothing left to delete.)
        3. UPDATE canonical_turns to NULL compacted_at / compaction_operation_id
           where compaction_operation_id = dead_op. (Also idempotent.)
        4. ONLY IF the dead_op UPDATE in step 1 matched a row, INSERT a
           fresh running row for new_operation_id. Skipping this INSERT
           on the idempotent path preserves the unique-active-row
           invariant (``idx_compaction_operation_active``) and prevents
           two status='running' rows from coexisting at the same
           (conversation_id, lifecycle_epoch).
        """
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """UPDATE compaction_operation
                      SET status = 'abandoned', completed_at = ?
                    WHERE operation_id = ?
                      AND conversation_id = ?
                      AND lifecycle_epoch = ?
                      AND status = 'running'""",
                (now, dead_operation_id, conversation_id, lifecycle_epoch),
            )
            fresh_takeover = (cur.rowcount or 0) > 0
            for table in (
                "segments", "facts", "tag_summaries", "tag_summary_embeddings",
            ):
                conn.execute(
                    f"DELETE FROM {table} "
                    f"WHERE operation_id = ? AND conversation_id = ?",
                    (dead_operation_id, conversation_id),
                )
            conn.execute(
                """UPDATE canonical_turns
                      SET compacted_at = NULL,
                          compaction_operation_id = NULL,
                          updated_at = ?
                    WHERE conversation_id = ?
                      AND compaction_operation_id = ?""",
                (now, conversation_id, dead_operation_id),
            )
            if fresh_takeover:
                conn.execute(
                    """INSERT INTO compaction_operation
                       (operation_id, conversation_id, lifecycle_epoch,
                        phase_index, phase_count, phase_name, status,
                        started_at, heartbeat_ts, owner_worker_id, created_at)
                       VALUES (?, ?, ?, 0, ?, 'starting', 'running',
                               ?, ?, ?, ?)""",
                    (
                        new_operation_id, conversation_id, lifecycle_epoch,
                        phase_count, now, now, worker_id, now,
                    ),
                )
            self._commit_if_unlocked(conn)
        return fresh_takeover

    def refresh_compaction_heartbeat(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        operation_id: str,
    ) -> bool:
        """Refresh compaction_operation.heartbeat_ts atomically, scoped on
        (operation_id, lifecycle_epoch, owner_worker_id). Returns True iff
        the update hit a row. Sidecar callers interpret False as "our
        claim was stolen" and signal the compactor to abort.
        """
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """UPDATE compaction_operation
                      SET heartbeat_ts = ?
                    WHERE operation_id = ?
                      AND conversation_id = ?
                      AND lifecycle_epoch = ?
                      AND owner_worker_id = ?
                      AND status = 'running'""",
                (now, operation_id, conversation_id, lifecycle_epoch, worker_id),
            )
            self._commit_if_unlocked(conn)
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
        already-active operation). Returns True iff a matching row was
        updated: owner match + epoch match against both the local row
        and the authoritative ``conversations.lifecycle_epoch`` via
        correlated subquery.
        """
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE compaction_operation
                   SET phase_index = ?, phase_name = ?, heartbeat_ts = ?,
                       status = 'running'
                 WHERE conversation_id = ? AND status IN ('queued','running')
                   AND lifecycle_epoch = ?
                   AND lifecycle_epoch = (
                       SELECT lifecycle_epoch FROM conversations
                        WHERE conversation_id = ?
                   )
                   AND owner_worker_id = ?
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
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE compaction_operation
                   SET status = 'completed', completed_at = ?
                 WHERE conversation_id = ? AND status IN ('queued','running')
                   AND lifecycle_epoch = ?
                   AND lifecycle_epoch = (
                       SELECT lifecycle_epoch FROM conversations
                        WHERE conversation_id = ?
                   )
                   AND owner_worker_id = ?
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
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE compaction_operation
                   SET status = 'failed', completed_at = ?,
                       error_message = ?
                 WHERE conversation_id = ? AND status IN ('queued','running')
                   AND lifecycle_epoch = ?
                   AND lifecycle_epoch = (
                       SELECT lifecycle_epoch FROM conversations
                        WHERE conversation_id = ?
                   )
                   AND owner_worker_id = ?
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
    # All four methods filter on ``lifecycle_epoch`` in the WHERE clause
    # so a stale caller whose in-memory epoch no longer matches the
    # authoritative ``conversations`` row sees a ``False``/``None`` return
    # and never stomps a new lifecycle's counters/phase. SQLite uses
    # scalar ``MAX()`` for the monotonic widener (Postgres uses
    # ``GREATEST()``). ``set_phase_and_drain_pending_raw`` wraps its
    # read-then-UPDATE in ``BEGIN IMMEDIATE`` so a concurrent resurrect
    # cannot slip between the epoch check and the drain.

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
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE conversations
                   SET last_raw_payload_entries = ?,
                       last_ingestible_payload_entries = ?,
                       updated_at = ?
                 WHERE conversation_id = ?
                   AND lifecycle_epoch = ?
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
        SQLite's scalar ``MAX()`` (NOT the aggregate) so the column can
        only move forwards. Epoch-guarded: returns ``True`` iff the
        UPDATE matched a row at the caller's ``lifecycle_epoch``.
        """
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE conversations
                   SET pending_raw_payload_entries =
                       MAX(pending_raw_payload_entries, ?)
                 WHERE conversation_id = ?
                   AND lifecycle_epoch = ?
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
        now = utcnow_iso()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE conversations
                   SET phase = ?, updated_at = ?
                 WHERE conversation_id = ?
                   AND lifecycle_epoch = ?
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
        ``BEGIN IMMEDIATE`` so a concurrent resurrect cannot slip between
        the epoch check and the drain. Returns the drained integer on
        success, or ``None`` when the caller's ``lifecycle_epoch`` does
        not match the authoritative conversations row.
        """
        now = utcnow_iso()
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                """
                SELECT pending_raw_payload_entries, lifecycle_epoch
                  FROM conversations WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None or int(row[1]) != lifecycle_epoch:
                conn.rollback()
                return None
            drained = int(row[0])
            conn.execute(
                """
                UPDATE conversations
                   SET phase = ?,
                       pending_raw_payload_entries = 0,
                       updated_at = ?
                 WHERE conversation_id = ?
                   AND lifecycle_epoch = ?
                """,
                (new_phase, now, conversation_id, lifecycle_epoch),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return drained

    def drain_compaction_exit(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> str | None:
        """Atomic compaction-exit decision + pending drain.

        Single ``BEGIN IMMEDIATE`` transaction containing: epoch check,
        ``EXISTS (SELECT 1 FROM canonical_turns WHERE tagged_at IS NULL)``,
        phase UPDATE with pending drain, and (on untagged-exists) a fresh
        ``ingestion_episode`` INSERT seeded with the drained ``pending_raw``
        value. Serialising the read and write in one transaction closes
        the race where a concurrent tagger marks the last row tagged
        between a separate snapshot read and the phase UPDATE.

        Returns ``'ingesting'`` (work remains — episode row inserted) or
        ``'active'`` (all canonical rows tagged) on success, or ``None``
        when the caller's ``lifecycle_epoch`` does not match the
        authoritative conversations row. Epoch-guarded.
        """
        import uuid

        now = utcnow_iso()
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                """
                SELECT pending_raw_payload_entries, lifecycle_epoch,
                       EXISTS (
                         SELECT 1 FROM canonical_turns
                          WHERE conversation_id = ? AND tagged_at IS NULL
                       )
                  FROM conversations
                 WHERE conversation_id = ?
                """,
                (conversation_id, conversation_id),
            ).fetchone()
            if row is None or int(row[1]) != lifecycle_epoch:
                conn.rollback()
                return None
            pending_raw = int(row[0])
            has_untagged = bool(row[2])
            if has_untagged:
                conn.execute(
                    """
                    UPDATE conversations
                       SET phase = 'ingesting',
                           pending_raw_payload_entries = 0,
                           updated_at = ?
                     WHERE conversation_id = ?
                       AND lifecycle_epoch = ?
                    """,
                    (now, conversation_id, lifecycle_epoch),
                )
                conn.execute(
                    """
                    INSERT INTO ingestion_episode (
                        episode_id, conversation_id, lifecycle_epoch,
                        raw_payload_entries, started_at, status,
                        owner_worker_id, heartbeat_ts
                    ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                    """,
                    (
                        str(uuid.uuid4()), conversation_id, lifecycle_epoch,
                        pending_raw, now, worker_id, now,
                    ),
                )
                new_phase = "ingesting"
            else:
                conn.execute(
                    """
                    UPDATE conversations
                       SET phase = 'active',
                           pending_raw_payload_entries = 0,
                           updated_at = ?
                     WHERE conversation_id = ?
                       AND lifecycle_epoch = ?
                    """,
                    (now, conversation_id, lifecycle_epoch),
                )
                new_phase = "active"
            conn.commit()
            return new_phase
        except Exception:
            conn.rollback()
            raise

    def delete_conversation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        deleted = self._delete_conversation_rows(conn, "segments", conversation_id)
        # Also clear persisted state and diagnostics so restarts do not
        # resurrect a partially deleted conversation.
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
        conn.commit()

        # Disk cleanup: remove media files for this conversation
        import os
        import shutil
        _data_dir = str(self.db_path.parent) if hasattr(self, "db_path") else ""
        media_dir = os.path.join(_data_dir, "media", conversation_id) if _data_dir else ""
        if media_dir and os.path.isdir(media_dir):
            shutil.rmtree(media_dir, ignore_errors=True)

        return deleted

    def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int:
        conn = self._get_conn()
        deleted = 0

        if max_age:
            cutoff = datetime.now(timezone.utc) - max_age
            cursor = conn.execute(
                "DELETE FROM segments WHERE created_at < ?",
                (_dt_to_str(cutoff),),
            )
            deleted += cursor.rowcount

        conn.commit()
        return deleted

    def save_tag_summary(
        self,
        tag_summary: TagSummary,
        conversation_id: str = "",
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None:
        from ..types import CompactionLeaseLost

        guard_all = (
            operation_id is not None
            and owner_worker_id is not None
            and lifecycle_epoch is not None
        )

        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if guard_all:
                # INSERT-SELECT form: writes zero rows if the compaction_operation
                # row no longer matches (status != 'running', owner mismatch, etc).
                # The ON CONFLICT DO UPDATE clause only fires when the SELECT produces
                # a row candidate — i.e., when the guard passes.
                cur = conn.execute(
                    """INSERT INTO tag_summaries
                    (tag, conversation_id, summary, description, code_refs, summary_tokens,
                     source_segment_refs, source_turn_numbers, source_canonical_turn_ids,
                     covers_through_turn, covers_through_canonical_turn_id, generated_by_turn_id,
                     created_at, updated_at, operation_id)
                    SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                      FROM compaction_operation
                     WHERE operation_id = ?
                       AND conversation_id = ?
                       AND status = 'running'
                       AND owner_worker_id = ?
                       AND lifecycle_epoch = ?
                    ON CONFLICT (tag, conversation_id) DO UPDATE SET
                        summary = excluded.summary,
                        description = excluded.description,
                        code_refs = excluded.code_refs,
                        summary_tokens = excluded.summary_tokens,
                        source_segment_refs = excluded.source_segment_refs,
                        source_turn_numbers = excluded.source_turn_numbers,
                        source_canonical_turn_ids = excluded.source_canonical_turn_ids,
                        covers_through_turn = excluded.covers_through_turn,
                        covers_through_canonical_turn_id = excluded.covers_through_canonical_turn_id,
                        generated_by_turn_id = excluded.generated_by_turn_id,
                        updated_at = excluded.updated_at,
                        operation_id = excluded.operation_id""",
                    (
                        tag_summary.tag,
                        conversation_id,
                        tag_summary.summary,
                        tag_summary.description,
                        json.dumps(getattr(tag_summary, "code_refs", []) or []),
                        tag_summary.summary_tokens,
                        json.dumps(tag_summary.source_segment_refs),
                        json.dumps(tag_summary.source_turn_numbers),
                        json.dumps(getattr(tag_summary, "source_canonical_turn_ids", []) or []),
                        tag_summary.covers_through_turn,
                        getattr(tag_summary, "covers_through_canonical_turn_id", "") or "",
                        getattr(tag_summary, "generated_by_turn_id", "") or "",
                        _dt_to_str(tag_summary.created_at),
                        _dt_to_str(tag_summary.updated_at),
                        operation_id,
                        # WHERE clause params:
                        operation_id,
                        conversation_id,
                        owner_worker_id,
                        lifecycle_epoch,
                    ),
                )
                if (cur.rowcount or 0) == 0:
                    conn.execute("ROLLBACK")
                    raise CompactionLeaseLost(
                        operation_id=operation_id,
                        write_site="save_tag_summary",
                    )
            else:
                # Legacy unconditional path — existing callers and test harnesses.
                conn.execute(
                    """INSERT OR REPLACE INTO tag_summaries
                    (tag, conversation_id, summary, description, code_refs, summary_tokens,
                     source_segment_refs, source_turn_numbers, source_canonical_turn_ids,
                     covers_through_turn, covers_through_canonical_turn_id, generated_by_turn_id,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        tag_summary.tag,
                        conversation_id,
                        tag_summary.summary,
                        tag_summary.description,
                        json.dumps(getattr(tag_summary, "code_refs", []) or []),
                        tag_summary.summary_tokens,
                        json.dumps(tag_summary.source_segment_refs),
                        json.dumps(tag_summary.source_turn_numbers),
                        json.dumps(getattr(tag_summary, "source_canonical_turn_ids", []) or []),
                        tag_summary.covers_through_turn,
                        getattr(tag_summary, "covers_through_canonical_turn_id", "") or "",
                        getattr(tag_summary, "generated_by_turn_id", "") or "",
                        _dt_to_str(tag_summary.created_at),
                        _dt_to_str(tag_summary.updated_at),
                    ),
                )
            conn.execute("COMMIT")
        except CompactionLeaseLost:
            # Already rolled back above; re-raise without swallowing.
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tag_summaries WHERE tag = ? AND conversation_id = ?", (tag, conversation_id)
        ).fetchone()
        if not row:
            return None
        # description column may not exist in pre-v0.2 rows
        desc = ""
        try:
            desc = row["description"]
        except (IndexError, KeyError):
            pass
        refs = []
        try:
            refs = json.loads(row["code_refs"] or "[]")
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            pass
        return TagSummary(
            tag=row["tag"],
            summary=row["summary"],
            description=desc,
            code_refs=refs,
            summary_tokens=row["summary_tokens"],
            source_segment_refs=json.loads(row["source_segment_refs"]),
            source_turn_numbers=json.loads(row["source_turn_numbers"]),
            source_canonical_turn_ids=json.loads(row["source_canonical_turn_ids"] or "[]")
            if "source_canonical_turn_ids" in row.keys()
            else [],
            covers_through_turn=row["covers_through_turn"],
            covers_through_canonical_turn_id=row["covers_through_canonical_turn_id"]
            if "covers_through_canonical_turn_id" in row.keys()
            else "",
            generated_by_turn_id=row["generated_by_turn_id"] if "generated_by_turn_id" in row.keys() else "",
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )

    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT * FROM tag_summaries WHERE conversation_id = ? ORDER BY updated_at DESC",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tag_summaries ORDER BY updated_at DESC"
            ).fetchall()
        results: list[TagSummary] = []
        for row in rows:
            desc = ""
            try:
                desc = row["description"]
            except (IndexError, KeyError):
                pass
            refs = []
            try:
                refs = json.loads(row["code_refs"] or "[]")
            except (IndexError, KeyError, TypeError, json.JSONDecodeError):
                pass
            results.append(TagSummary(
                tag=row["tag"],
                summary=row["summary"],
                description=desc,
                code_refs=refs,
                summary_tokens=row["summary_tokens"],
                source_segment_refs=json.loads(row["source_segment_refs"]),
                source_turn_numbers=json.loads(row["source_turn_numbers"]),
                source_canonical_turn_ids=json.loads(row["source_canonical_turn_ids"] or "[]")
                if "source_canonical_turn_ids" in row.keys()
                else [],
                covers_through_turn=row["covers_through_turn"],
                covers_through_canonical_turn_id=row["covers_through_canonical_turn_id"]
                if "covers_through_canonical_turn_id" in row.keys()
                else "",
                generated_by_turn_id=row["generated_by_turn_id"] if "generated_by_turn_id" in row.keys() else "",
                created_at=_str_to_dt(row["created_at"]),
                updated_at=_str_to_dt(row["updated_at"]),
            ))
        return results

    def get_segments_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 20,
        conversation_id: str | None = None,
    ) -> list[StoredSegment]:
        if not tags:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(tags))
        query = f"""
            SELECT s.*, COUNT(st.tag) as overlap_count
            FROM segments s
            JOIN segment_tags st ON s.ref = st.segment_ref
            WHERE st.tag IN ({placeholders})
        """
        params: list = list(tags)

        if conversation_id is not None:
            query += " AND s.conversation_id = ?"
            params.append(conversation_id)

        query += """
            GROUP BY s.ref
            HAVING overlap_count >= ?
            ORDER BY overlap_count DESC, s.created_at DESC
            LIMIT ?
        """
        params.extend([min_overlap, limit])
        rows = conn.execute(query, params).fetchall()
        refs = [row["ref"] for row in rows]
        tags_map = self._batch_get_tags(refs)
        results = []
        for row in rows:
            results.append(_row_to_segment(row, tags_map[row["ref"]]))
        return results

    def store_chunk_embeddings(self, segment_ref: str, chunks: list[ChunkEmbedding]) -> None:
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM segment_chunks WHERE segment_ref = ?", (segment_ref,))
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO segment_chunks (segment_ref, chunk_index, text, embedding_json) VALUES (?, ?, ?, ?)",
                    (chunk.segment_ref, chunk.chunk_index, chunk.text, json.dumps(chunk.embedding)),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT segment_ref, chunk_index, text, embedding_json FROM segment_chunks ORDER BY segment_ref, chunk_index"
        ).fetchall()
        return [
            ChunkEmbedding(
                segment_ref=row[0],
                chunk_index=row[1],
                text=row[2],
                embedding=json.loads(row[3]),
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
        own_txn = not self._reconcile_lock_active()
        if own_txn:
            conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "DELETE FROM canonical_turn_chunks WHERE conversation_id = ? AND canonical_turn_id = ? AND side = ?",
                (conversation_id, canonical_turn_id, side),
            )
            for chunk in chunks:
                conn.execute(
                    """INSERT INTO canonical_turn_chunks
                    (conversation_id, canonical_turn_id, side, chunk_index, text, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.conversation_id,
                        canonical_turn_id,
                        chunk.side,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(chunk.embedding),
                    ),
                )
            if own_txn:
                conn.commit()
        except Exception:
            if own_txn:
                conn.rollback()
            raise

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
                WHERE ctc.conversation_id = ?
                ORDER BY cto.turn_number, ctc.side, ctc.chunk_index""",
                (conversation_id,),
            ).fetchall()
        return [
            CanonicalTurnChunkEmbedding(
                conversation_id=row["conversation_id"],
                canonical_turn_id=(row["canonical_turn_id"] or ""),
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
                "DELETE FROM canonical_turn_chunks WHERE conversation_id = ?",
                (conversation_id,),
            )
        else:
            if canonical_turn_id is None:
                return 0
            cur = conn.execute(
                "DELETE FROM canonical_turn_chunks WHERE conversation_id = ? AND canonical_turn_id = ?",
                (conversation_id, canonical_turn_id),
            )
        self._commit_if_unlocked(conn)
        return int(cur.rowcount or 0)

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        conn = self._get_conn()
        entries_json = json.dumps([
            {
                "turn_number": e.turn_number,
                "canonical_turn_id": getattr(e, "canonical_turn_id", "") or "",
                "message_hash": e.message_hash,
                "tags": e.tags,
                "primary_tag": e.primary_tag,
                "timestamp": _dt_to_str(e.timestamp),
                "session_date": e.session_date,
                "fact_signals": [
                    {"subject": fs.subject, "verb": fs.verb,
                     "object": fs.object, "status": fs.status}
                    for fs in e.fact_signals
                ] if e.fact_signals else [],
                "code_refs": list(getattr(e, "code_refs", []) or []),
                "sender": e.sender,
            }
            for e in state.turn_tag_entries
        ])
        # Include split_processed_tags, working_set, and trailing_fingerprint
        # in the entries JSON blob (avoids schema migrations)
        state_blob = json.dumps({
            "turn_tag_entries": json.loads(entries_json),
            "split_processed_tags": state.split_processed_tags,
            "working_set": [
                {
                    "tag": ws.tag,
                    "depth": ws.depth.value if hasattr(ws.depth, 'value') else ws.depth,
                    "tokens": ws.tokens,
                    "last_accessed_turn": ws.last_accessed_turn,
                }
                for ws in state.working_set
            ],
            "trailing_fingerprint": state.trailing_fingerprint,
            "telemetry_rollup": state.telemetry_rollup,
            "request_captures": state.request_captures,
            "provider": state.provider,
            "flushed_prefix_messages": state.flushed_prefix_messages,
            "last_request_time": state.last_request_time,
            "tool_tag_counter": state.tool_tag_counter,
            "last_compacted_turn": state.last_compacted_turn,
            "last_completed_turn": state.last_completed_turn,
            "last_indexed_turn": state.last_indexed_turn,
            "checkpoint_version": state.checkpoint_version,
        })
        conn.execute(
            """INSERT OR REPLACE INTO engine_state
            (conversation_id, compacted_prefix_messages, turn_count, turn_tag_entries, saved_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                state.conversation_id,
                state.compacted_prefix_messages,
                state.turn_count,
                state_blob,
                _dt_to_str(state.saved_at),
            ),
        )
        conn.commit()

    def _parse_engine_state_row(self, row) -> EngineStateSnapshot:
        raw = json.loads(row["turn_tag_entries"])
        # Support both old format (list of entries) and new format (dict with split_processed_tags)
        if isinstance(raw, dict):
            entries_raw = raw.get("turn_tag_entries", [])
            split_processed_tags = raw.get("split_processed_tags", [])
            working_set_raw = raw.get("working_set", [])
            trailing_fingerprint = raw.get("trailing_fingerprint", "")
            telemetry_rollup = raw.get("telemetry_rollup", {})
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
                max(row["turn_count"] - 1, len(entries_raw) - 1),
            )
            last_indexed_turn = raw.get(
                "last_indexed_turn",
                len(entries_raw) - 1,
            )
            checkpoint_version = raw.get("checkpoint_version", 0)
        else:
            entries_raw = raw
            split_processed_tags = []
            working_set_raw = []
            trailing_fingerprint = ""
            telemetry_rollup = {}
            request_captures = []
            provider = ""
            flushed_prefix_messages = 0
            flushed_prefix_messages_present = False
            last_request_time = 0.0
            tool_tag_counter = 0
            last_compacted_turn = (row["compacted_prefix_messages"] // 2) - 1 if row["compacted_prefix_messages"] > 0 else -1
            last_completed_turn = max(row["turn_count"] - 1, len(entries_raw) - 1)
            last_indexed_turn = len(entries_raw) - 1
            checkpoint_version = 0
        entries = [
            TurnTagEntry(
                turn_number=e["turn_number"],
                canonical_turn_id=e.get("canonical_turn_id", "") or "",
                message_hash=e["message_hash"],
                tags=e["tags"],
                primary_tag=e["primary_tag"],
                timestamp=_str_to_dt(e["timestamp"]),
                session_date=e.get("session_date", ""),
                fact_signals=[
                    FactSignal(
                        subject=fs.get("subject", ""),
                        verb=fs.get("verb", fs.get("role", "")),
                        object=fs.get("object", ""),
                        status=fs.get("status", ""),
                    )
                    for fs in e.get("fact_signals", [])
                ],
                code_refs=e.get("code_refs", []) or [],
                sender=e.get("sender", ""),
            )
            for e in entries_raw
        ]
        working_set = [
            WorkingSetEntry(
                tag=ws["tag"],
                depth=DepthLevel(ws["depth"]),
                tokens=ws.get("tokens", 0),
                last_accessed_turn=ws.get("last_accessed_turn", 0),
            )
            for ws in working_set_raw
        ]
        return EngineStateSnapshot(
            conversation_id=row["conversation_id"],
            compacted_prefix_messages=row["compacted_prefix_messages"],
            flushed_prefix_messages=flushed_prefix_messages,
            flushed_prefix_messages_present=flushed_prefix_messages_present,
            last_request_time=last_request_time,
            turn_tag_entries=entries,
            turn_count=row["turn_count"],
            last_compacted_turn=last_compacted_turn,
            last_completed_turn=last_completed_turn,
            last_indexed_turn=last_indexed_turn,
            checkpoint_version=checkpoint_version,
            conversation_generation=self.get_conversation_generation(row["conversation_id"]),
            saved_at=_str_to_dt(row["saved_at"]),
            split_processed_tags=split_processed_tags,
            working_set=working_set,
            trailing_fingerprint=trailing_fingerprint,
            telemetry_rollup=telemetry_rollup,
            request_captures=request_captures,
            provider=provider,
            tool_tag_counter=tool_tag_counter,
        )

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM engine_state WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        if not row:
            return None
        return self._parse_engine_state_row(row)

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM engine_state ORDER BY compacted_prefix_messages DESC, saved_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return self._parse_engine_state_row(row)

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        """Return {trailing_fingerprint: conversation_id} for all persisted conversations."""
        conn = self._get_conn()
        result: dict[str, str] = {}
        for row in conn.execute("SELECT conversation_id, turn_tag_entries FROM engine_state").fetchall():
            try:
                raw = json.loads(row["turn_tag_entries"])
                fp = raw.get("trailing_fingerprint", "") if isinstance(raw, dict) else ""
                if fp:
                    result[fp] = row["conversation_id"]
            except (json.JSONDecodeError, TypeError):
                continue
        return result

    def save_conversation_alias(self, alias_id: str, target_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO conversation_aliases (alias_id, target_id) VALUES (?, ?)",
            (alias_id, target_id),
        )
        conn.commit()

    def resolve_conversation_alias(self, alias_id: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT target_id FROM conversation_aliases WHERE alias_id = ?",
            (alias_id,),
        ).fetchone()
        return row[0] if row else None

    def delete_conversation_alias(self, alias_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM conversation_aliases WHERE alias_id = ?",
            (alias_id,),
        )
        conn.commit()

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
                "SELECT canonical_turn_id FROM canonical_turns WHERE conversation_id = ? AND sort_key = ?",
                (conversation_id, float((turn_number + 1) * 1000.0)),
            ).fetchone()
            canonical_turn_id = str(slot_row["canonical_turn_id"]) if slot_row else None
        existing_sort_key = None
        if canonical_turn_id:
            row = conn.execute(
                "SELECT sort_key FROM canonical_turns WHERE conversation_id = ? AND canonical_turn_id = ?",
                (conversation_id, canonical_turn_id),
            ).fetchone()
            if row:
                existing_sort_key = float(row["sort_key"])
        if canonical_turn_id is None:
            canonical_turn_id = generate_canonical_turn_id()
        if sort_key is None:
            if existing_sort_key is not None:
                sort_key = existing_sort_key
            elif turn_number >= 0:
                sort_key = float((turn_number + 1) * 1000.0)
            else:
                tail = conn.execute(
                    "SELECT COALESCE(MAX(sort_key), 0) AS max_sort_key FROM canonical_turns WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                sort_key = float((tail["max_sort_key"] if tail else 0.0) or 0.0) + 1000.0
        conn.execute(
            """INSERT INTO canonical_turns
            (canonical_turn_id, conversation_id, turn_group_number, sort_key, turn_hash, hash_version,
             normalized_user_text, normalized_assistant_text, user_content, assistant_content,
             user_raw_content, assistant_raw_content, primary_tag, tags_json, session_date, sender,
             fact_signals_json, code_refs_json, tagged_at, compacted_at, first_seen_at, last_seen_at,
             source_batch_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_turn_id) DO UPDATE SET
                turn_group_number=excluded.turn_group_number,
                sort_key=excluded.sort_key,
                turn_hash=excluded.turn_hash,
                hash_version=excluded.hash_version,
                normalized_user_text=excluded.normalized_user_text,
                normalized_assistant_text=excluded.normalized_assistant_text,
                user_content=excluded.user_content,
                assistant_content=excluded.assistant_content,
                user_raw_content=excluded.user_raw_content,
                assistant_raw_content=excluded.assistant_raw_content,
                primary_tag=excluded.primary_tag,
                tags_json=excluded.tags_json,
                session_date=excluded.session_date,
                sender=excluded.sender,
                fact_signals_json=excluded.fact_signals_json,
                code_refs_json=excluded.code_refs_json,
                tagged_at=excluded.tagged_at,
                compacted_at=excluded.compacted_at,
                last_seen_at=excluded.last_seen_at,
                source_batch_id=excluded.source_batch_id,
                updated_at=excluded.updated_at""",
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
                   SET turn_group_number = ?
                   WHERE conversation_id = ?
                     AND canonical_turn_id = ?
                     AND turn_group_number <> ?""",
                (turn_group_number, conversation_id, canonical_turn_id, turn_group_number),
            )
            changed += int(cursor.rowcount or 0)
        conn.commit()
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
        placeholders = ",".join("?" for _ in canonical_turn_ids)
        cur = conn.execute(
            f"""UPDATE canonical_turns
                SET tagged_at = ?, updated_at = ?
                WHERE conversation_id = ? AND canonical_turn_id IN ({placeholders})""",
            [timestamp, timestamp, conversation_id, *canonical_turn_ids],
        )
        self._commit_if_unlocked(conn)
        return int(cur.rowcount or 0)

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
        walk (EXPLAIN QUERY PLAN: ``SEARCH ct USING INDEX
        idx_canonical_turns_conv_untagged``). Routing through the view forces
        SQLite into a full scan of ``canonical_turns`` plus a temp B-tree
        sort, which is unacceptable on the tagger hot path.
        ``turn_number`` is a view-only computed column; omitting it here is
        tolerated by ``_row_to_canonical_turn`` (defaults to ``-1``). We JOIN
        against ``conversations.lifecycle_epoch`` so a caller carrying a
        stale epoch simply matches zero rows at SQL level rather than
        raising. Ordering uses ``sort_key`` — the same column the partial
        index is sorted on.
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
             WHERE ct.conversation_id = ?
               AND ct.tagged_at IS NULL
               AND c.lifecycle_epoch = ?
             ORDER BY ct.sort_key ASC
             LIMIT ?
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
        epoch) matches nothing and quietly gets ``False`` — the caller is
        expected to interpret that as "my lifecycle is over, exit". The
        ``tagged_at IS NULL`` predicate keeps the call idempotent: a retry
        on an already-tagged row is also ``False`` (no-op).
        """
        now = utcnow_iso()
        conn = self._get_conn()
        cur = conn.execute(
            """
            UPDATE canonical_turns
               SET tagged_at = ?, updated_at = ?
             WHERE canonical_turn_id = ?
               AND conversation_id = ?
               AND tagged_at IS NULL
               AND EXISTS (
                   SELECT 1 FROM conversations c
                    WHERE c.conversation_id = ?
                      AND c.lifecycle_epoch = ?
               )
            """,
            (now, now, canonical_turn_id, conversation_id,
             conversation_id, expected_lifecycle_epoch),
        )
        self._commit_if_unlocked(conn)
        return int(cur.rowcount or 0) == 1

    def mark_canonical_turns_compacted(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        compacted_at: str | None = None,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
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

        Ownership guard: when operation_id/owner_worker_id/lifecycle_epoch
        are all provided, appends an EXISTS sub-select that verifies the
        compaction_operation row is still 'running' and owned by this
        worker. If rowcount == 0, raises CompactionLeaseLost. The
        turn_group merge-expansion is preserved on BOTH paths.
        Also sets compaction_operation_id on the guarded path so cleanup
        can find these rows by operation.
        """
        if not canonical_turn_ids:
            return 0
        from ..types import CompactionLeaseLost

        guard_all = (
            operation_id is not None
            and owner_worker_id is not None
            and lifecycle_epoch is not None
        )

        conn = self._get_conn()
        timestamp = compacted_at or _dt_to_str(datetime.now(timezone.utc))
        placeholders = ",".join("?" for _ in canonical_turn_ids)

        if guard_all:
            cur = conn.execute(
                f"""UPDATE canonical_turns
                    SET compacted_at = ?, updated_at = ?,
                        compaction_operation_id = ?
                    WHERE conversation_id = ?
                      AND (
                          canonical_turn_id IN ({placeholders})
                          OR turn_group_number IN (
                              SELECT DISTINCT turn_group_number
                                FROM canonical_turns
                               WHERE conversation_id = ?
                                 AND canonical_turn_id IN ({placeholders})
                                 AND turn_group_number >= 0
                          )
                      )
                      AND EXISTS (
                          SELECT 1 FROM compaction_operation
                           WHERE operation_id = ?
                             AND conversation_id = ?
                             AND status = 'running'
                             AND owner_worker_id = ?
                             AND lifecycle_epoch = ?
                      )""",
                [
                    timestamp, timestamp, operation_id, conversation_id,
                    *canonical_turn_ids, conversation_id, *canonical_turn_ids,
                    operation_id, conversation_id, owner_worker_id, lifecycle_epoch,
                ],
            )
            self._commit_if_unlocked(conn)
            if (cur.rowcount or 0) == 0:
                raise CompactionLeaseLost(
                    operation_id=operation_id,
                    write_site="mark_canonical_turns_compacted",
                )
            return int(cur.rowcount)
        else:
            cur = conn.execute(
                f"""UPDATE canonical_turns
                    SET compacted_at = ?, updated_at = ?
                    WHERE conversation_id = ?
                      AND (
                          canonical_turn_id IN ({placeholders})
                          OR turn_group_number IN (
                              SELECT DISTINCT turn_group_number
                                FROM canonical_turns
                               WHERE conversation_id = ?
                                 AND canonical_turn_id IN ({placeholders})
                                 AND turn_group_number >= 0
                          )
                      )""",
                [
                    timestamp, timestamp, conversation_id,
                    *canonical_turn_ids, conversation_id, *canonical_turn_ids,
                ],
            )
            self._commit_if_unlocked(conn)
            return int(cur.rowcount or 0)

    def delete_canonical_turns(
        self,
        conversation_id: str,
        turn_number: int | None = None,
    ) -> int:
        conn = self._get_conn()
        if turn_number is None:
            cur = conn.execute(
                "DELETE FROM canonical_turns WHERE conversation_id = ?",
                (conversation_id,),
            )
        else:
            canonical_turn_id = self._lookup_canonical_turn_id_for_ordinal(conversation_id, turn_number)
            if canonical_turn_id is None:
                return 0
            cur = conn.execute(
                "DELETE FROM canonical_turns WHERE conversation_id = ? AND canonical_turn_id = ?",
                (conversation_id, canonical_turn_id),
            )
        conn.commit()
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
            "DELETE FROM canonical_turns WHERE conversation_id = ? AND source_batch_id = ?",
            (conversation_id, batch_id),
        )
        self._commit_if_unlocked(conn)
        return int(cur.rowcount or 0)

    def replace_canonical_turn_anchors(
        self,
        conversation_id: str,
        anchors: list[tuple[int, str, str]],
    ) -> int:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM canonical_turn_anchors WHERE conversation_id = ?",
            (conversation_id,),
        )
        rows = [
            (conversation_id, anchor_hash, start_turn_id, int(window_size))
            for window_size, anchor_hash, start_turn_id in anchors
            if anchor_hash and start_turn_id
        ]
        if not rows:
            self._commit_if_unlocked(conn)
            return 0
        conn.executemany(
            """INSERT INTO canonical_turn_anchors
               (conversation_id, anchor_hash, start_turn_id, window_size)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        self._commit_if_unlocked(conn)
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
               WHERE cta.conversation_id = ?
                 AND cta.window_size = ?
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

    def save_ingest_batch(self, batch: dict) -> str:
        conn = self._get_conn()
        batch_id = str(batch.get("batch_id", "") or generate_canonical_turn_id())
        conn.execute(
            """INSERT OR REPLACE INTO ingest_batches
            (batch_id, conversation_id, received_at, raw_turn_count, merge_mode, turns_matched,
             turns_appended, turns_prepended, turns_inserted, first_turn_hash, last_turn_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                batch.get("conversation_id", "") or "",
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
        self._commit_if_unlocked(conn)
        return batch_id

    # ------------------------------------------------------------------
    # D1: Fact Extraction
    # ------------------------------------------------------------------

    def store_facts(
        self,
        facts: list[Fact],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int:
        if not facts:
            return 0
        from ..types import CompactionLeaseLost

        guard_all = (
            operation_id is not None
            and owner_worker_id is not None
            and lifecycle_epoch is not None
        )

        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            count = 0
            for fact in facts:
                if guard_all:
                    # INSERT-SELECT form: writes zero rows if the
                    # compaction_operation row no longer matches
                    # (status != 'running', owner mismatch, epoch mismatch).
                    # The same transaction holds the write lock for the entire
                    # batch, so a concurrent takeover cannot interleave between
                    # rows. However, a takeover that completes *before* the
                    # first INSERT fires (at-rest stale) is caught here.
                    cur = conn.execute(
                        """INSERT OR REPLACE INTO facts
                        (id, subject, verb, object, status, what, who, when_date,
                         "where", why, fact_type, tags_json, segment_ref, conversation_id,
                         turn_numbers_json, mentioned_at, session_date, superseded_by)
                        SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                          FROM compaction_operation
                         WHERE operation_id = ?
                           AND conversation_id = ?
                           AND status = 'running'
                           AND owner_worker_id = ?
                           AND lifecycle_epoch = ?""",
                        (
                            fact.id,
                            fact.subject,
                            fact.verb,
                            fact.object,
                            fact.status,
                            fact.what,
                            fact.who,
                            fact.when_date,
                            fact.where,
                            fact.why,
                            fact.fact_type,
                            json.dumps(fact.tags),
                            fact.segment_ref,
                            fact.conversation_id,
                            json.dumps(fact.turn_numbers),
                            _dt_to_str(fact.mentioned_at),
                            fact.session_date or "",
                            fact.superseded_by,
                            # WHERE clause params:
                            operation_id,
                            fact.conversation_id,
                            owner_worker_id,
                            lifecycle_epoch,
                        ),
                    )
                    if (cur.rowcount or 0) == 0:
                        conn.execute("ROLLBACK")
                        raise CompactionLeaseLost(
                            operation_id=operation_id,
                            write_site="store_facts",
                        )
                else:
                    # Legacy unconditional path — existing callers and
                    # non-compaction write sites.
                    conn.execute(
                        """INSERT OR REPLACE INTO facts
                        (id, subject, verb, object, status, what, who, when_date,
                         "where", why, fact_type, tags_json, segment_ref, conversation_id,
                         turn_numbers_json, mentioned_at, session_date, superseded_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            fact.id,
                            fact.subject,
                            fact.verb,
                            fact.object,
                            fact.status,
                            fact.what,
                            fact.who,
                            fact.when_date,
                            fact.where,
                            fact.why,
                            fact.fact_type,
                            json.dumps(fact.tags),
                            fact.segment_ref,
                            fact.conversation_id,
                            json.dumps(fact.turn_numbers),
                            _dt_to_str(fact.mentioned_at),
                            fact.session_date or "",
                            fact.superseded_by,
                        ),
                    )
                # Update fact_tags junction (same for both paths)
                conn.execute("DELETE FROM fact_tags WHERE fact_id = ?", (fact.id,))
                for tag in fact.tags:
                    conn.execute(
                        "INSERT INTO fact_tags (fact_id, tag) VALUES (?, ?)",
                        (fact.id, tag),
                    )
                count += 1
            conn.execute("COMMIT")
            return count
        except CompactionLeaseLost:
            # Already rolled back above; re-raise without swallowing.
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact.from_dict(dict(row), dt_parser=_str_to_dt)

    def _row_to_fact_with_session_date(self, row: sqlite3.Row) -> Fact:
        fact = self._row_to_fact(row)
        seg_meta = row["_seg_meta"] if "_seg_meta" in row.keys() else None
        if seg_meta:
            try:
                meta = json.loads(seg_meta)
                fact.session_date = meta.get("session_date", "")
            except (json.JSONDecodeError, TypeError):
                pass
        return fact

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT DISTINCT f.verb FROM facts f"
                " JOIN segments s ON f.segment_ref = s.ref"
                " WHERE s.conversation_id = ? AND f.verb != '' AND f.superseded_by IS NULL",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT verb FROM facts WHERE verb != '' AND superseded_by IS NULL"
            ).fetchall()
        return [r[0] for r in rows]

    def query_facts(
        self,
        *,
        subject: str | None = None,
        verb: str | None = None,
        verbs: list[str] | None = None,
        object_contains: str | None = None,
        status: str | None = None,
        fact_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[object] = []

        if conversation_id is not None:
            conditions.append("f.conversation_id = ?")
            params.append(conversation_id)
        if subject is not None:
            conditions.append("f.subject = ?")
            params.append(subject)
        if verbs is not None:
            # Disjunctive LIKE match across multiple expanded verbs
            like_clauses = ["f.verb LIKE ? ESCAPE '\\'" for _ in verbs]
            conditions.append("(" + " OR ".join(like_clauses) + ")")
            params.extend(f"%{_escape_like(v)}%" for v in verbs)
        elif verb is not None:
            conditions.append("f.verb LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(verb)}%")
        if object_contains is not None:
            conditions.append("f.object LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(object_contains)}%")
        if status is not None:
            conditions.append("f.status = ?")
            params.append(status)
        if fact_type is not None:
            conditions.append("f.fact_type = ?")
            params.append(fact_type)
        # Only return non-superseded facts by default
        conditions.append("f.superseded_by IS NULL")

        if tags:
            # JOIN fact_tags and require overlap
            placeholders = ",".join("?" for _ in tags)
            sql = f"""
                SELECT DISTINCT f.* FROM facts f
                JOIN fact_tags ft ON f.id = ft.fact_id
                WHERE ft.tag IN ({placeholders})
            """
            params_list = list(tags) + params
            if conditions:
                sql += " AND " + " AND ".join(conditions)
            sql += f" ORDER BY f.mentioned_at DESC LIMIT ?"
            params_list.append(limit)
            rows = conn.execute(sql, params_list).fetchall()
        else:
            where = " AND ".join(conditions) if conditions else "1=1"
            sql = f"SELECT * FROM facts f WHERE {where} ORDER BY f.mentioned_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM facts WHERE segment_ref = ? ORDER BY mentioned_at",
            (segment_ref,),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def replace_facts_for_segment(
        self,
        conversation_id: str,
        segment_ref: str,
        facts: list,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> tuple[int, int]:
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Delete fact_tags first (before facts are removed)
            conn.execute(
                "DELETE FROM fact_tags WHERE fact_id IN "
                "(SELECT id FROM facts WHERE conversation_id = ? AND segment_ref = ?)",
                (conversation_id, segment_ref),
            )
            cursor = conn.execute(
                "DELETE FROM facts WHERE conversation_id = ? AND segment_ref = ?",
                (conversation_id, segment_ref),
            )
            deleted = cursor.rowcount
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Delegate to store_facts (which handles the ownership guard and
        # transaction internally).
        inserted = self.store_facts(
            facts,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        ) if facts else 0
        return deleted, inserted

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list[Fact]:
        """FTS search across fact subject, verb, object, what fields.

        Only returns non-superseded facts. Falls back to LIKE if FTS
        is unavailable. Populates session_date from the parent segment.
        """
        conn = self._get_conn()
        conv_clause = ""
        conv_params: list[object] = []
        if conversation_id is not None:
            conv_clause = " AND f.conversation_id = ?"
            conv_params = [conversation_id]
        try:
            rows = conn.execute("""
                SELECT f.*, s.metadata_json AS _seg_meta FROM facts f
                JOIN facts_fts fts ON f.id = fts.id
                LEFT JOIN segments s ON f.segment_ref = s.ref
                WHERE facts_fts MATCH ?
                AND f.superseded_by IS NULL""" + conv_clause + """
                ORDER BY rank
                LIMIT ?
            """, [_sanitize_fts_query(query)] + conv_params + [limit]).fetchall()
            return [self._row_to_fact_with_session_date(row) for row in rows]
        except sqlite3.OperationalError:
            # FTS not available — fall back to LIKE across key fields
            terms = [t.strip() for t in query.split() if len(t.strip()) > 2]
            if not terms:
                return []
            conditions = []
            params: list[object] = []
            for term in terms[:5]:
                escaped = _escape_like(term)
                conditions.append(
                    "(f.subject LIKE ? ESCAPE '\\' OR f.verb LIKE ? ESCAPE '\\'"
                    " OR f.object LIKE ? ESCAPE '\\' OR f.what LIKE ? ESCAPE '\\')"
                )
                params.extend([f"%{escaped}%"] * 4)
            where = " OR ".join(conditions)
            sql = f"""
                SELECT f.*, s.metadata_json AS _seg_meta FROM facts f
                LEFT JOIN segments s ON f.segment_ref = s.ref
                WHERE ({where})
                AND f.superseded_by IS NULL""" + conv_clause + """
                ORDER BY f.mentioned_at DESC LIMIT ?
            """
            params.extend(conv_params)
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_fact_with_session_date(row) for row in rows]

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE facts SET superseded_by = ? WHERE id = ?",
            (new_fact_id, old_fact_id),
        )
        conn.commit()

    def update_fact_fields(
        self, fact_id: str, verb: str, object: str, status: str, what: str
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE facts SET verb = ?, object = ?, status = ?, what = ? WHERE id = ?",
            (verb, object, status, what, fact_id),
        )
        # FTS5 sync handled by AFTER UPDATE trigger (facts_fts_au)
        conn.commit()

    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT ft.tag, COUNT(*) as cnt FROM fact_tags ft"
                " JOIN facts f ON f.id = ft.fact_id"
                " WHERE f.conversation_id = ?"
                " GROUP BY ft.tag",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT tag, COUNT(*) as cnt FROM fact_tags GROUP BY tag"
            ).fetchall()
        return {row["tag"]: row["cnt"] for row in rows}

    def query_experience_facts_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        conn = self._get_conn()
        sql = """SELECT * FROM facts
                 WHERE when_date >= ? AND when_date <= ?"""
        params: list = [start_date, end_date + "~"]
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY when_date ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    # ------------------------------------------------------------------
    # Fact links
    # ------------------------------------------------------------------

    def store_fact_links(self, links: list[FactLink]) -> int:
        if not links:
            return 0
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            count = 0
            for link in links:
                conn.execute(
                    """INSERT OR REPLACE INTO fact_links
                    (id, source_fact_id, target_fact_id, relation_type,
                     confidence, context, created_at, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        link.id,
                        link.source_fact_id,
                        link.target_fact_id,
                        link.relation_type,
                        link.confidence,
                        link.context,
                        _dt_to_str(link.created_at),
                        link.created_by,
                    ),
                )
                count += 1
            conn.execute("COMMIT")
            return count
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def get_fact_links(self, fact_id: str, direction: str = "both") -> list[FactLink]:
        """Get fact links for a given fact ID.

        Args:
            fact_id: The fact ID to query links for.
            direction: "outgoing" (source), "incoming" (target), or "both".
        """
        conn = self._get_conn()
        if direction == "outgoing":
            rows = conn.execute(
                "SELECT * FROM fact_links WHERE source_fact_id = ?", (fact_id,)
            ).fetchall()
        elif direction == "incoming":
            rows = conn.execute(
                "SELECT * FROM fact_links WHERE target_fact_id = ?", (fact_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fact_links WHERE source_fact_id = ? OR target_fact_id = ?",
                (fact_id, fact_id),
            ).fetchall()
        return [self._row_to_fact_link(row) for row in rows]

    def get_linked_facts(self, fact_ids: list[str], depth: int = 1) -> list[LinkedFact]:
        """BFS traversal through fact_links, returning linked Fact objects.

        Excludes superseded facts (superseded_by IS NOT NULL).
        """
        if not fact_ids:
            return []
        conn = self._get_conn()
        visited: set[str] = set(fact_ids)
        current_layer: set[str] = set(fact_ids)
        results: list[LinkedFact] = []

        for _hop in range(depth):
            if not current_layer:
                break
            placeholders = ",".join("?" * len(current_layer))
            visited_placeholders = ",".join("?" * len(visited))
            rows = conn.execute(
                f"""SELECT fl.source_fact_id, fl.target_fact_id, fl.relation_type,
                        fl.confidence, fl.context,
                        f.id AS fact_id, f.subject, f.verb, f.object, f.status, f.what, f.who,
                        f.when_date, f."where", f.why, f.fact_type, f.tags_json, f.segment_ref,
                        f.conversation_id, f.turn_numbers_json, f.mentioned_at, f.session_date,
                        f.superseded_by
                    FROM fact_links fl
                    JOIN facts f ON (
                        (fl.source_fact_id IN ({placeholders}) AND f.id = fl.target_fact_id)
                        OR (fl.target_fact_id IN ({placeholders}) AND f.id = fl.source_fact_id)
                    )
                    WHERE f.superseded_by IS NULL
                    AND f.id NOT IN ({visited_placeholders})""",
                list(current_layer) + list(current_layer) + list(visited),
            ).fetchall()

            next_layer: set[str] = set()
            for row in rows:
                fid = row["fact_id"]
                if fid in visited:
                    continue
                visited.add(fid)
                next_layer.add(fid)
                # Determine which seed fact this is linked from
                src = row["source_fact_id"]
                tgt = row["target_fact_id"]
                linked_from = src if src in current_layer else tgt
                fact = Fact(
                    id=fid,
                    subject=row["subject"],
                    verb=row["verb"],
                    object=row["object"],
                    status=row["status"],
                    what=row["what"],
                    who=row["who"],
                    when_date=row["when_date"],
                    where=row["where"],
                    why=row["why"],
                    fact_type=row["fact_type"] if row["fact_type"] else "personal",
                    tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
                    segment_ref=row["segment_ref"],
                    conversation_id=row["conversation_id"],
                    turn_numbers=json.loads(row["turn_numbers_json"]) if row["turn_numbers_json"] else [],
                    mentioned_at=_str_to_dt(row["mentioned_at"]) if row["mentioned_at"] else datetime.now(timezone.utc),
                    session_date=row["session_date"] if row["session_date"] else "",
                    superseded_by=row["superseded_by"],
                )
                results.append(LinkedFact(
                    fact=fact,
                    linked_from_fact_id=linked_from,
                    relation_type=row["relation_type"],
                    confidence=row["confidence"],
                    link_context=row["context"],
                ))
            current_layer = next_layer

        return results

    def delete_fact_links(self, fact_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM fact_links WHERE source_fact_id = ? OR target_fact_id = ?",
            (fact_id, fact_id),
        )
        conn.commit()
        return cursor.rowcount

    def migrate_supersession_to_links(self) -> int:
        """Migrate superseded_by column data to SUPERSEDES fact links. Idempotent."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, superseded_by FROM facts WHERE superseded_by IS NOT NULL"
        ).fetchall()
        if not rows:
            return 0
        count = 0
        for row in rows:
            old_id = row["id"]
            new_id = row["superseded_by"]
            existing = conn.execute(
                "SELECT 1 FROM fact_links WHERE source_fact_id = ? AND target_fact_id = ? AND relation_type = 'supersedes'",
                (new_id, old_id),
            ).fetchone()
            if existing:
                continue
            link = FactLink(
                source_fact_id=new_id,
                target_fact_id=old_id,
                relation_type="supersedes",
                confidence=1.0,
                context="Migrated from superseded_by column",
                created_by="migration",
            )
            conn.execute(
                """INSERT INTO fact_links (id, source_fact_id, target_fact_id, relation_type,
                   confidence, context, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (link.id, link.source_fact_id, link.target_fact_id, link.relation_type,
                 link.confidence, link.context, _dt_to_str(link.created_at), link.created_by),
            )
            count += 1
        conn.commit()
        return count

    def _row_to_fact_link(self, row: sqlite3.Row) -> FactLink:
        return FactLink(
            id=row["id"],
            source_fact_id=row["source_fact_id"],
            target_fact_id=row["target_fact_id"],
            relation_type=row["relation_type"],
            confidence=row["confidence"],
            context=row["context"],
            created_at=_str_to_dt(row["created_at"]) if row["created_at"] else datetime.now(timezone.utc),
            created_by=row["created_by"],
        )

    # ------------------------------------------------------------------
    # Cross-cutting queries
    # ------------------------------------------------------------------

    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]:
        """Return snippet info for orphan tags (no tag_summary entry).

        For each tag NOT already in tag_summaries, fetches the first
        segment's summary text (truncated to 100 chars) as a description
        hint for the consolidator.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT st.tag, substr(s.summary, 1, 100) as snippet
                FROM segment_tags st
                JOIN segments s ON s.ref = st.segment_ref
                WHERE st.tag NOT IN (SELECT tag FROM tag_summaries)
                GROUP BY st.tag
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [{"tag": row["tag"], "snippet": row["snippet"]} for row in rows]
        except Exception:
            return []

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        if not fact_ids:
            return []
        conn = self._get_conn()
        try:
            placeholders = ",".join("?" * len(fact_ids))
            rows = conn.execute(
                f"SELECT superseded_by, subject, verb, object FROM facts "
                f"WHERE superseded_by IN ({placeholders})",
                fact_ids,
            ).fetchall()
            return [
                {
                    "superseded_by": row["superseded_by"],
                    "subject": row["subject"],
                    "verb": row["verb"],
                    "object": row["object"],
                }
                for row in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Tool Output Storage
    # ------------------------------------------------------------------

    def store_tool_output(
        self,
        ref: str,
        conversation_id: str,
        tool_name: str,
        command: str,
        turn: int,
        content: str,
        original_bytes: int,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tool_outputs
            (ref, conversation_id, tool_name, command, turn, content, original_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ref, conversation_id, tool_name, command, turn, content, original_bytes),
        )
        conn.commit()

    def search_tool_outputs(self, query: str, limit: int = 5, conversation_id: str | None = None) -> list:
        from ..types import QuoteResult

        conn = self._get_conn()
        _sc = self.search_config
        _tool_chars = _sc.tool_output_snippet_chars if _sc else 100
        conv_clause = ""
        conv_params: list[object] = []
        if conversation_id is not None:
            conv_clause = " AND t.conversation_id = ?"
            conv_params = [conversation_id]
        try:
            rows = conn.execute(
                """SELECT t.ref, t.tool_name,
                          snippet(tool_outputs_fts, 0, '>>>', '<<<', '...', """ + str(_tool_chars) + """) as snippet
                   FROM tool_outputs_fts fts
                   JOIN tool_outputs t ON t.rowid = fts.rowid
                   WHERE tool_outputs_fts MATCH ?""" + conv_clause + """
                   ORDER BY rank
                   LIMIT ?""",
                [_sanitize_fts_query(query)] + conv_params + [limit],
            ).fetchall()
        except Exception:
            return []
        return [
            QuoteResult(
                text=row["snippet"],
                tag=row["tool_name"],
                segment_ref=row["ref"],
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
            """INSERT OR IGNORE INTO turn_tool_outputs (conversation_id, turn_number, tool_output_ref)
            VALUES (?, ?, ?)""",
            (conversation_id, turn_number, tool_output_ref),
        )
        conn.commit()

    def get_tool_outputs_for_turn(self, conversation_id: str, turn_number: int) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tool_output_ref FROM turn_tool_outputs WHERE conversation_id = ? AND turn_number = ?",
            (conversation_id, turn_number),
        ).fetchall()
        return [row["tool_output_ref"] for row in rows]

    def link_segment_tool_output(self, conversation_id: str, segment_ref: str, tool_output_ref: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR IGNORE INTO segment_tool_outputs (conversation_id, segment_ref, tool_output_ref)
            VALUES (?, ?, ?)""",
            (conversation_id, segment_ref, tool_output_ref),
        )
        conn.commit()

    def get_tool_outputs_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tool_output_ref FROM segment_tool_outputs WHERE conversation_id = ? AND segment_ref = ?",
            (conversation_id, segment_ref),
        ).fetchall()
        return [row["tool_output_ref"] for row in rows]

    def get_tool_output_refs_for_turn(self, conversation_id: str, turn: int) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT ref FROM tool_outputs WHERE conversation_id = ? AND turn = ?",
            (conversation_id, turn),
        ).fetchall()
        return [row["ref"] for row in rows]

    def get_tool_output_by_ref(self, conversation_id: str, ref: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT content FROM tool_outputs WHERE conversation_id = ? AND ref = ?",
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
            """INSERT OR REPLACE INTO media_outputs
            (ref, conversation_id, media_type, width, height, original_bytes, compressed_bytes, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ref, conversation_id, media_type, width, height, original_bytes, compressed_bytes, file_path),
        )
        conn.commit()

    def get_media_output(self, conversation_id: str, ref: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT ref, conversation_id, media_type, width, height, original_bytes, compressed_bytes, file_path FROM media_outputs WHERE conversation_id = ? AND ref = ?",
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
            """INSERT OR REPLACE INTO chain_snapshots
            (ref, conversation_id, turn_number, chain_json, message_count, tool_output_refs, created_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (ref, conversation_id, turn_number, chain_json, message_count, tool_output_refs),
        )
        conn.commit()

    def get_chain_snapshot(self, conversation_id: str, ref: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT ref, conversation_id, turn_number, chain_json, message_count, tool_output_refs
            FROM chain_snapshots WHERE conversation_id = ? AND ref = ?""",
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
            FROM chain_snapshots WHERE conversation_id = ? AND turn_number >= ?
            ORDER BY turn_number""",
            (conversation_id, min_turn),
        ).fetchall()
        return [{"ref": row[0] if isinstance(row, tuple) else row["ref"],
                 "turn_number": row[1] if isinstance(row, tuple) else row["turn_number"],
                 "tool_output_refs": row[2] if isinstance(row, tuple) else row["tool_output_refs"],
                 "message_count": row[3] if isinstance(row, tuple) else row["message_count"]} for row in rows]

    def get_tool_names_for_refs(self, refs: list[str]) -> list[str]:
        if not refs:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(refs))
        rows = conn.execute(
            f"SELECT DISTINCT tool_name FROM tool_outputs WHERE ref IN ({placeholders}) AND tool_name != ''",
            refs,
        ).fetchall()
        return [row[0] if isinstance(row, tuple) else row["tool_name"] for row in rows]

    def get_tool_names_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT DISTINCT t.tool_name
            FROM segment_tool_outputs s
            JOIN tool_outputs t ON t.ref = s.tool_output_ref AND t.conversation_id = s.conversation_id
            WHERE s.conversation_id = ? AND s.segment_ref = ?
            ORDER BY t.tool_name""",
            (conversation_id, segment_ref),
        ).fetchall()
        return [row["tool_name"] for row in rows]

    # ------------------------------------------------------------------
    # Request capture persistence
    # ------------------------------------------------------------------

    def save_request_capture(self, capture: dict) -> None:
        conn = self._get_conn()
        import time as _time
        conversation_id = capture.get("conversation_id", "") or ""
        turn_id = capture.get("turn_id", "") or ""
        conn.execute(
            """INSERT OR REPLACE INTO request_captures
            (conversation_id, turn, turn_id, ts, recorded_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
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
            WHERE conversation_id = ?
              AND (conversation_id, turn, turn_id) NOT IN (
                SELECT conversation_id, turn, turn_id
                FROM request_captures
                WHERE conversation_id = ?
                ORDER BY recorded_at DESC LIMIT 50
            )""",
            (conversation_id, conversation_id),
        )
        conn.commit()

    def load_request_captures(
        self,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[dict]:
        conn = self._get_conn()
        try:
            if conversation_id is None:
                rows = conn.execute(
                    "SELECT data_json FROM request_captures ORDER BY recorded_at ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT data_json FROM request_captures
                    WHERE conversation_id = ?
                    ORDER BY recorded_at ASC LIMIT ?""",
                    (conversation_id, limit),
                ).fetchall()
        except Exception:
            return []
        result = []
        for (data_json,) in rows:
            try:
                result.append(json.loads(data_json))
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    # ------------------------------------------------------------------
    # Tag summary search (RRF retrieval scoring)
    # ------------------------------------------------------------------

    def search_tag_summaries_fts(
        self, query: str, limit: int = 20, conversation_id: str | None = None,
    ) -> list[tuple[str, float]]:
        conn = self._get_conn()
        try:
            sanitized = _sanitize_fts_query_terms(query)
            if not sanitized:
                return []
            conv_clause = ""
            params: list = [sanitized]
            if conversation_id is not None:
                conv_clause = " AND ts.conversation_id = ?"
                params.append(conversation_id)
            rows = conn.execute(
                f"""SELECT ts.tag, bm25(tag_summaries_fts) as score
                FROM tag_summaries_fts fts
                JOIN tag_summaries ts ON ts.rowid = fts.rowid
                WHERE tag_summaries_fts MATCH ?{conv_clause}
                ORDER BY score
                LIMIT ?""",
                params + [limit],
            ).fetchall()
            return [(row["tag"], -row["score"]) for row in rows]
        except Exception:
            return []

    def store_tag_summary_embedding(
        self,
        tag: str,
        conversation_id: str,
        embedding: list[float],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None:
        from ..types import CompactionLeaseLost

        guard_all = (
            operation_id is not None
            and owner_worker_id is not None
            and lifecycle_epoch is not None
        )

        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if guard_all:
                # INSERT-SELECT form: writes zero rows if the compaction_operation
                # row no longer matches (status != 'running', owner mismatch, etc).
                # The ON CONFLICT DO UPDATE clause only fires when the SELECT produces
                # a row candidate — i.e., when the guard passes.
                cur = conn.execute(
                    """INSERT INTO tag_summary_embeddings
                    (tag, conversation_id, embedding_json, operation_id)
                    SELECT ?, ?, ?, ?
                      FROM compaction_operation
                     WHERE operation_id = ?
                       AND conversation_id = ?
                       AND status = 'running'
                       AND owner_worker_id = ?
                       AND lifecycle_epoch = ?
                    ON CONFLICT (tag, conversation_id) DO UPDATE SET
                        embedding_json = excluded.embedding_json,
                        operation_id = excluded.operation_id""",
                    (
                        tag,
                        conversation_id,
                        json.dumps(embedding),
                        operation_id,
                        # WHERE clause params:
                        operation_id,
                        conversation_id,
                        owner_worker_id,
                        lifecycle_epoch,
                    ),
                )
                if (cur.rowcount or 0) == 0:
                    conn.execute("ROLLBACK")
                    raise CompactionLeaseLost(
                        operation_id=operation_id,
                        write_site="store_tag_summary_embedding",
                    )
            else:
                # Legacy unconditional path — existing callers and test harnesses.
                conn.execute(
                    """INSERT OR REPLACE INTO tag_summary_embeddings (tag, conversation_id, embedding_json)
                    VALUES (?, ?, ?)""",
                    (tag, conversation_id, json.dumps(embedding)),
                )
            conn.execute("COMMIT")
        except CompactionLeaseLost:
            # Already rolled back above; re-raise without swallowing.
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def load_tag_summary_embeddings(
        self, conversation_id: str | None = None,
    ) -> dict[str, list[float]]:
        conn = self._get_conn()
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT tag, embedding_json FROM tag_summary_embeddings WHERE conversation_id = ?",
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
        placeholders = ",".join("?" for _ in tags)
        params: list = list(tags)
        conv_clause = ""
        if conversation_id is not None:
            conv_clause = " AND f.conversation_id = ?"
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
    # Tool call persistence (dashboard)
    # ------------------------------------------------------------------

    def save_tool_call(self, call: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tool_calls
            (conversation_id, request_turn, round, group_id, tool_name,
             tool_input, tool_result, result_length, duration_ms, found, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        # Ring buffer: keep last 50 per conversation
        conv_id = call.get("conversation_id", "")
        conn.execute(
            """DELETE FROM tool_calls WHERE id NOT IN (
                SELECT id FROM tool_calls WHERE conversation_id = ?
                ORDER BY id DESC LIMIT 50
            ) AND conversation_id = ?""",
            (conv_id, conv_id),
        )
        conn.commit()

    def load_tool_calls(self, conversation_id: str, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tool_calls WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def load_tool_call(self, call_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tool_calls WHERE id = ?", (call_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Request context persistence (dashboard recall page)
    # ------------------------------------------------------------------

    def save_request_context(self, context: dict) -> int:
        conn = self._get_conn()
        conv_id = context.get("conversation_id", "")
        explicit_turn = int(context.get("request_turn", 0) or 0)
        conn.execute("BEGIN IMMEDIATE")
        try:
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    SELECT id FROM request_context WHERE conversation_id = ?
                    ORDER BY id DESC LIMIT 50
                ) AND conversation_id = ?""",
                (conv_id, conv_id),
            )
            conn.commit()
            return request_turn
        except Exception:
            conn.rollback()
            raise

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
                WHERE rc.conversation_id = ?
            ) ranked
            ORDER BY id DESC
            LIMIT ?""",
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

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
