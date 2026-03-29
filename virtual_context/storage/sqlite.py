"""SQLiteStore: primary storage backend using stdlib sqlite3."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from ..core.store import ContextStore
from ..types import ChunkEmbedding, ConversationStats, DepthLevel, EngineStateSnapshot, Fact, FactLink, FactSignal, LinkedFact, QuoteResult, SegmentMetadata, StoredSegment, StoredSummary, TagStats, TagSummary, TemporalStatus, TurnTagEntry, WorkingSetEntry
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
    summary_tokens INTEGER NOT NULL DEFAULT 0,
    source_segment_refs TEXT NOT NULL DEFAULT '[]',
    source_turn_numbers TEXT NOT NULL DEFAULT '[]',
    covers_through_turn INTEGER NOT NULL DEFAULT -1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tag, conversation_id)
);

CREATE TABLE IF NOT EXISTS engine_state (
    conversation_id TEXT PRIMARY KEY,
    compacted_through INTEGER NOT NULL,
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

CREATE TABLE IF NOT EXISTS turn_messages (
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    user_content TEXT NOT NULL DEFAULT '',
    assistant_content TEXT NOT NULL DEFAULT '',
    user_raw_content TEXT,
    assistant_raw_content TEXT,
    PRIMARY KEY (conversation_id, turn_number)
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
            turn_count=metadata_raw.get("turn_count", 0),
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
            turn_count=metadata_raw.get("turn_count", 0),
            session_date=metadata_raw.get("session_date", ""),
        ),
        created_at=_str_to_dt(row["created_at"]),
        start_timestamp=_str_to_dt(row["start_timestamp"]),
        end_timestamp=_str_to_dt(row["end_timestamp"]),
    )


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
            conn.execute("ALTER TABLE turn_messages ADD COLUMN user_raw_content TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE turn_messages ADD COLUMN assistant_raw_content TEXT")
        except Exception:
            pass
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
                or pk_names != ["conversation_id", "turn"]
            ):
                legacy_rows = conn.execute(
                    "SELECT turn, ts, recorded_at, data_json FROM request_captures"
                ).fetchall()
                migrated_rows: list[tuple[str, int, str, float, str]] = []
                for row in legacy_rows:
                    data_json = row["data_json"]
                    conversation_id = ""
                    turn = row["turn"]
                    try:
                        payload = json.loads(data_json)
                        conversation_id = payload.get("conversation_id", "") or ""
                        turn = int(payload.get("turn", turn))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    migrated_rows.append((
                        conversation_id,
                        turn,
                        row["ts"],
                        row["recorded_at"],
                        data_json,
                    ))
                conn.executescript("""
                    DROP TABLE IF EXISTS request_captures_new;
                    CREATE TABLE request_captures_new (
                        conversation_id TEXT NOT NULL DEFAULT '',
                        turn INTEGER NOT NULL,
                        ts TEXT NOT NULL,
                        recorded_at REAL NOT NULL,
                        data_json TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, turn)
                    );
                    DROP TABLE request_captures;
                    ALTER TABLE request_captures_new RENAME TO request_captures;
                """)
                if migrated_rows:
                    conn.executemany(
                        """INSERT OR REPLACE INTO request_captures
                        (conversation_id, turn, ts, recorded_at, data_json)
                        VALUES (?, ?, ?, ?, ?)""",
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
                    summary_tokens INTEGER NOT NULL DEFAULT 0,
                    source_segment_refs TEXT NOT NULL DEFAULT '[]',
                    source_turn_numbers TEXT NOT NULL DEFAULT '[]',
                    covers_through_turn INTEGER NOT NULL DEFAULT -1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tag, conversation_id)
                );
                INSERT OR IGNORE INTO tag_summaries_new
                    (tag, conversation_id, summary, description, summary_tokens,
                     source_segment_refs, source_turn_numbers, covers_through_turn,
                     created_at, updated_at)
                SELECT tag, '', summary, description, summary_tokens,
                       source_segment_refs, source_turn_numbers, covers_through_turn,
                       created_at, updated_at
                FROM tag_summaries;
                DROP TABLE tag_summaries;
                ALTER TABLE tag_summaries_new RENAME TO tag_summaries;
            """)
        # Request capture persistence for proxy dashboard
        conn.executescript("""
CREATE TABLE IF NOT EXISTS request_captures (
    conversation_id TEXT NOT NULL DEFAULT '',
    turn INTEGER NOT NULL,
    ts TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn)
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
        conn.commit()
        self._repair_fts_if_needed(conn)

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

    def store_segment(self, segment: StoredSegment) -> str:
        conn = self._get_conn()
        primary_tag = segment.primary_tag
        summary_text = segment.summary
        full_text = segment.full_text
        metadata_dict = {
            "entities": segment.metadata.entities,
            "key_decisions": segment.metadata.key_decisions,
            "action_items": segment.metadata.action_items,
            "date_references": segment.metadata.date_references,
            "turn_count": segment.metadata.turn_count,
        }
        if segment.metadata.session_date:
            metadata_dict["session_date"] = segment.metadata.session_date
        metadata_json = json.dumps(metadata_dict)

        conn.execute("BEGIN IMMEDIATE")
        try:
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

            # Update tags
            conn.execute("DELETE FROM segment_tags WHERE segment_ref = ?", (segment.ref,))
            for tag in segment.tags:
                conn.execute(
                    "INSERT INTO segment_tags (segment_ref, tag) VALUES (?, ?)",
                    (segment.ref, tag),
                )

            conn.execute("COMMIT")
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
                    created_at=row[4] or "",
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
                created_at=row[4] or "",
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

    def delete_conversation(self, conversation_id: str) -> int:
        conn = self._get_conn()
        deleted = self._delete_conversation_rows(conn, "segments", conversation_id)
        # Also clear persisted state and diagnostics so restarts do not
        # resurrect a partially deleted conversation.
        for table in (
            "engine_state",
            "facts",
            "turn_messages",
            "tag_summaries",
            "tag_aliases",
            "request_captures",
            "tool_outputs",
            "tool_calls",
            "request_context",
            "tag_summary_embeddings",
            "turn_tool_outputs",
            "segment_tool_outputs",
            "chain_snapshots",
            "media_outputs",
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

    def save_tag_summary(self, tag_summary: TagSummary, conversation_id: str = "") -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tag_summaries
            (tag, conversation_id, summary, description, summary_tokens, source_segment_refs,
             source_turn_numbers, covers_through_turn, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tag_summary.tag,
                conversation_id,
                tag_summary.summary,
                tag_summary.description,
                tag_summary.summary_tokens,
                json.dumps(tag_summary.source_segment_refs),
                json.dumps(tag_summary.source_turn_numbers),
                tag_summary.covers_through_turn,
                _dt_to_str(tag_summary.created_at),
                _dt_to_str(tag_summary.updated_at),
            ),
        )
        conn.commit()

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
        return TagSummary(
            tag=row["tag"],
            summary=row["summary"],
            description=desc,
            summary_tokens=row["summary_tokens"],
            source_segment_refs=json.loads(row["source_segment_refs"]),
            source_turn_numbers=json.loads(row["source_turn_numbers"]),
            covers_through_turn=row["covers_through_turn"],
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
            results.append(TagSummary(
                tag=row["tag"],
                summary=row["summary"],
                description=desc,
                summary_tokens=row["summary_tokens"],
                source_segment_refs=json.loads(row["source_segment_refs"]),
                source_turn_numbers=json.loads(row["source_turn_numbers"]),
                covers_through_turn=row["covers_through_turn"],
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

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        conn = self._get_conn()
        entries_json = json.dumps([
            {
                "turn_number": e.turn_number,
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
            "tool_tag_counter": state.tool_tag_counter,
            "last_compacted_turn": state.last_compacted_turn,
            "last_completed_turn": state.last_completed_turn,
            "last_indexed_turn": state.last_indexed_turn,
            "checkpoint_version": state.checkpoint_version,
        })
        conn.execute(
            """INSERT OR REPLACE INTO engine_state
            (conversation_id, compacted_through, turn_count, turn_tag_entries, saved_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                state.conversation_id,
                state.compacted_through,
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
            tool_tag_counter = raw.get("tool_tag_counter", 0)
            last_compacted_turn = raw.get(
                "last_compacted_turn",
                (row["compacted_through"] // 2) - 1 if row["compacted_through"] > 0 else -1,
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
            tool_tag_counter = 0
            last_compacted_turn = (row["compacted_through"] // 2) - 1 if row["compacted_through"] > 0 else -1
            last_completed_turn = max(row["turn_count"] - 1, len(entries_raw) - 1)
            last_indexed_turn = len(entries_raw) - 1
            checkpoint_version = 0
        entries = [
            TurnTagEntry(
                turn_number=e["turn_number"],
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
            compacted_through=row["compacted_through"],
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
            "SELECT * FROM engine_state ORDER BY compacted_through DESC, saved_at DESC LIMIT 1"
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

    # ------------------------------------------------------------------
    # Turn messages
    # ------------------------------------------------------------------

    def save_turn_message(
        self,
        conversation_id: str,
        turn_number: int,
        user_content: str,
        assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO turn_messages
            (conversation_id, turn_number, user_content, assistant_content,
             user_raw_content, assistant_raw_content)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (conversation_id, turn_number, user_content, assistant_content,
             user_raw_content, assistant_raw_content),
        )
        conn.commit()

    def get_turn_messages(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, tuple[str, str, str | None, str | None]]:
        if not turn_numbers:
            return {}
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in turn_numbers)
        rows = conn.execute(
            f"""SELECT turn_number, user_content, assistant_content,
                       user_raw_content, assistant_raw_content
            FROM turn_messages
            WHERE conversation_id = ? AND turn_number IN ({placeholders})""",
            [conversation_id] + turn_numbers,
        ).fetchall()
        return {
            row["turn_number"]: (
                row["user_content"],
                row["assistant_content"],
                row["user_raw_content"],
                row["assistant_raw_content"],
            )
            for row in rows
        }

    def load_recent_turn_messages(
        self,
        conversation_id: str,
        limit: int = 100,
    ) -> list[tuple[int, str, str]]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT turn_number, user_content, assistant_content
            FROM turn_messages
            WHERE conversation_id = ?
            ORDER BY turn_number DESC
            LIMIT ?""",
            (conversation_id, limit),
        ).fetchall()
        # Return in ascending order (oldest first)
        return [(r["turn_number"], r["user_content"], r["assistant_content"]) for r in reversed(rows)]

    def prune_turn_messages(self, conversation_id: str, keep_from_turn: int) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """DELETE FROM turn_messages
            WHERE conversation_id = ? AND turn_number < ?""",
            (conversation_id, keep_from_turn),
        )
        conn.commit()
        return int(cur.rowcount or 0)

    # ------------------------------------------------------------------
    # D1: Fact Extraction
    # ------------------------------------------------------------------

    def store_facts(self, facts: list[Fact]) -> int:
        if not facts:
            return 0
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            count = 0
            for fact in facts:
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
                # Update fact_tags junction
                conn.execute("DELETE FROM fact_tags WHERE fact_id = ?", (fact.id,))
                for tag in fact.tags:
                    conn.execute(
                        "INSERT INTO fact_tags (fact_id, tag) VALUES (?, ?)",
                        (fact.id, tag),
                    )
                count += 1
            conn.execute("COMMIT")
            return count
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

    def replace_facts_for_segment(self, conversation_id: str, segment_ref: str, facts: list) -> tuple[int, int]:
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
            # Insert new facts using same logic as store_facts
            inserted = 0
            for fact in facts:
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
                # Update fact_tags junction
                conn.execute("DELETE FROM fact_tags WHERE fact_id = ?", (fact.id,))
                for tag in fact.tags:
                    conn.execute(
                        "INSERT INTO fact_tags (fact_id, tag) VALUES (?, ?)",
                        (fact.id, tag),
                    )
                inserted += 1
            conn.execute("COMMIT")
            return deleted, inserted
        except Exception:
            conn.execute("ROLLBACK")
            raise

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
                 WHERE fact_type = 'experience'
                   AND status = 'completed'
                   AND session_date >= ? AND session_date <= ?"""
        params: list = [start_date, end_date + "~"]
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY session_date ASC LIMIT ?"
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
        conn.execute(
            """INSERT OR REPLACE INTO request_captures
            (conversation_id, turn, ts, recorded_at, data_json)
            VALUES (?, ?, ?, ?, ?)""",
            (
                conversation_id,
                capture["turn"],
                capture.get("ts", ""),
                _time.time(),
                json.dumps(capture),
            ),
        )
        conn.execute(
            """DELETE FROM request_captures
            WHERE conversation_id = ?
              AND (conversation_id, turn) NOT IN (
                SELECT conversation_id, turn
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
        self, tag: str, conversation_id: str, embedding: list[float],
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tag_summary_embeddings (tag, conversation_id, embedding_json)
            VALUES (?, ?, ?)""",
            (tag, conversation_id, json.dumps(embedding)),
        )
        conn.commit()

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

    def save_request_context(self, context: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO request_context
            (conversation_id, request_turn, timestamp, user_message, inbound_tags,
             retrieval_method, candidates_found, candidates_selected,
             segments_injected, facts_injected, facts_count, facts_tags,
             pool_used, pool_budget, total_context_tokens,
             non_virtualizable_floor, tool_call_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.get("conversation_id", ""),
                context.get("request_turn", 0),
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
        conv_id = context.get("conversation_id", "")
        conn.execute(
            """DELETE FROM request_context WHERE id NOT IN (
                SELECT id FROM request_context WHERE conversation_id = ?
                ORDER BY id DESC LIMIT 50
            ) AND conversation_id = ?""",
            (conv_id, conv_id),
        )
        conn.commit()

    def load_request_contexts(self, conversation_id: str, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM request_context WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
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
