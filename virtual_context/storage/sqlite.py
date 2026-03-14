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
    alias TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
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
    tag TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    summary_tokens INTEGER NOT NULL DEFAULT 0,
    source_segment_refs TEXT NOT NULL DEFAULT '[]',
    source_turn_numbers TEXT NOT NULL DEFAULT '[]',
    covers_through_turn INTEGER NOT NULL DEFAULT -1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS engine_state (
    conversation_id TEXT PRIMARY KEY,
    compacted_through INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    turn_tag_entries TEXT NOT NULL,
    saved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_chunks (
    segment_ref TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (segment_ref, chunk_index)
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
            conn.executescript(FACTS_FTS_SQL)
            conn.executescript(FACTS_FTS_TRIGGER_SQL)
            # Content-sync FTS tables need 'rebuild' to populate from
            # existing rows (direct INSERT doesn't work for content= tables).
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
        primary_tag = (
            segment.primary_tag
            if isinstance(segment.primary_tag, str)
            else json.dumps(segment.primary_tag, ensure_ascii=True, default=str)
        )
        summary_text = (
            segment.summary
            if isinstance(segment.summary, str)
            else json.dumps(segment.summary, ensure_ascii=True, default=str)
        )
        full_text = (
            segment.full_text
            if isinstance(segment.full_text, str)
            else json.dumps(segment.full_text, ensure_ascii=True, default=str)
        )
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
                normalized_tag = (
                    tag if isinstance(tag, str) else json.dumps(tag, ensure_ascii=True, default=str)
                )
                conn.execute(
                    "INSERT INTO segment_tags (segment_ref, tag) VALUES (?, ?)",
                    (segment.ref, normalized_tag),
                )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return segment.ref

    def get_segment(self, ref: str) -> StoredSegment | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM segments WHERE ref = ?", (ref,)).fetchone()
        if not row:
            return None
        tags = self._get_tags_for_ref(ref)
        return _row_to_segment(row, tags)

    def get_summary(self, ref: str) -> StoredSummary | None:
        conn = self._get_conn()
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
    ) -> list[StoredSummary]:
        conn = self._get_conn()

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
                    GROUP BY s.ref
                    ORDER BY rank
                    LIMIT ?""",
                    [_sanitize_fts_query(query), *tags, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT s.* FROM segments_fts fts
                    JOIN segments s ON s.ref = fts.ref
                    WHERE segments_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?""",
                    [_sanitize_fts_query(query), limit],
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
                    LIMIT ?""",
                    [like_query, *tags, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM segments WHERE summary LIKE ? ESCAPE '\\' LIMIT ?",
                    [like_query, limit],
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
    ) -> list[QuoteResult]:
        conn = self._get_conn()
        results: list[QuoteResult] = []

        # Try FTS5 first (with snippet extraction)
        try:
            rows = conn.execute(
                """SELECT fts.ref, s.primary_tag, s.metadata_json,
                          snippet(segments_fts_full, 1, '>>>', '<<<', '...', 500),
                          s.created_at
                   FROM segments_fts_full fts
                   JOIN segments s ON s.ref = fts.ref
                   WHERE segments_fts_full MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                [_sanitize_fts_query(query), limit],
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
        rows = conn.execute(
            """SELECT ref, primary_tag, full_text, metadata_json, created_at FROM segments
               WHERE full_text LIKE ? ESCAPE '\\'
               LIMIT ?""",
            [like_query, limit],
        ).fetchall()
        like_refs = [row[0] for row in rows]
        like_tags_map = self._batch_get_tags(like_refs)
        for row in rows:
            excerpt = _extract_excerpt(row[2], query, context_chars=200)
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

    def get_all_tags(self) -> list[TagStats]:
        conn = self._get_conn()
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

        results = []
        for row in rows:
            total_full = row["total_full_tokens"]
            total_summary = row["total_summary_tokens"]
            ratio = round(total_summary / total_full, 3) if total_full > 0 else 0.0

            tag_rows = conn.execute("""
                SELECT DISTINCT st.tag
                FROM segment_tags st
                JOIN segments s ON s.ref = st.segment_ref
                WHERE s.conversation_id = ?
                ORDER BY st.tag
            """, (row["conversation_id"],)).fetchall()

            results.append(ConversationStats(
                conversation_id=row["conversation_id"],
                segment_count=row["segment_count"],
                total_full_tokens=total_full,
                total_summary_tokens=total_summary,
                compression_ratio=ratio,
                distinct_tags=[r["tag"] for r in tag_rows],
                oldest_segment=_str_to_dt(row["oldest"]) if row["oldest"] else None,
                newest_segment=_str_to_dt(row["newest"]) if row["newest"] else None,
                compaction_model=row["compaction_model"] or "",
            ))

        return results

    def get_tag_aliases(self) -> dict[str, str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT alias, canonical FROM tag_aliases").fetchall()
        return {row["alias"]: row["canonical"] for row in rows}

    def set_tag_alias(self, alias: str, canonical: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO tag_aliases (alias, canonical) VALUES (?, ?)",
            (alias, canonical),
        )
        conn.commit()

    def delete_segment(self, ref: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM segments WHERE ref = ?", (ref,))
        conn.commit()
        return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete all segments for a given conversation_id. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM segments WHERE conversation_id = ?", (conversation_id,),
        )
        conn.commit()
        return cursor.rowcount

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

    def save_tag_summary(self, tag_summary: TagSummary) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tag_summaries
            (tag, summary, description, summary_tokens, source_segment_refs,
             source_turn_numbers, covers_through_turn, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tag_summary.tag,
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

    def get_tag_summary(self, tag: str) -> TagSummary | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tag_summaries WHERE tag = ?", (tag,)
        ).fetchone()
        if not row:
            return None
        # Backward compat: description column may not exist in old rows
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

    def get_all_tag_summaries(self) -> list[TagSummary]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tag_summaries ORDER BY tag"
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
            GROUP BY s.ref
            HAVING overlap_count >= ?
            ORDER BY overlap_count DESC, s.created_at DESC
            LIMIT ?
        """
        params: list = list(tags) + [min_overlap, limit]
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
        """Parse a SQLite row into an EngineStateSnapshot."""
        raw = json.loads(row["turn_tag_entries"])
        # Support both old format (list of entries) and new format (dict with split_processed_tags)
        if isinstance(raw, dict):
            entries_raw = raw.get("turn_tag_entries", [])
            split_processed_tags = raw.get("split_processed_tags", [])
            working_set_raw = raw.get("working_set", [])
            trailing_fingerprint = raw.get("trailing_fingerprint", "")
        else:
            entries_raw = raw
            split_processed_tags = []
            working_set_raw = []
            trailing_fingerprint = ""
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
            saved_at=_str_to_dt(row["saved_at"]),
            split_processed_tags=split_processed_tags,
            working_set=working_set,
            trailing_fingerprint=trailing_fingerprint,
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
        return Fact(
            id=row["id"],
            subject=row["subject"],
            verb=row["verb"],
            object=row["object"],
            status=row["status"],
            what=row["what"],
            who=row["who"],
            when_date=row["when_date"],
            where=row["where"],
            why=row["why"],
            fact_type=row["fact_type"] if "fact_type" in row.keys() else "personal",
            tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
            segment_ref=row["segment_ref"],
            conversation_id=row["conversation_id"],
            turn_numbers=json.loads(row["turn_numbers_json"]) if row["turn_numbers_json"] else [],
            mentioned_at=_str_to_dt(row["mentioned_at"]) if row["mentioned_at"] else datetime.now(timezone.utc),
            session_date=row["session_date"] if "session_date" in row.keys() else "",
            superseded_by=row["superseded_by"],
        )

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

    def get_unique_fact_verbs(self) -> list[str]:
        conn = self._get_conn()
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
    ) -> list[Fact]:
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[object] = []

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

    def search_facts(self, query: str, limit: int = 10) -> list[Fact]:
        """FTS search across fact subject, verb, object, what fields.

        Only returns non-superseded facts. Falls back to LIKE if FTS
        is unavailable. Populates session_date from the parent segment.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT f.*, s.metadata_json AS _seg_meta FROM facts f
                JOIN facts_fts fts ON f.id = fts.id
                LEFT JOIN segments s ON f.segment_ref = s.ref
                WHERE facts_fts MATCH ?
                AND f.superseded_by IS NULL
                ORDER BY rank
                LIMIT ?
            """, (_sanitize_fts_query(query), limit)).fetchall()
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
                AND f.superseded_by IS NULL
                ORDER BY f.mentioned_at DESC LIMIT ?
            """
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

    def get_fact_count_by_tags(self) -> dict[str, int]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag, COUNT(*) as cnt FROM fact_tags GROUP BY tag"
        ).fetchall()
        return {row["tag"]: row["cnt"] for row in rows}

    # ------------------------------------------------------------------
    # Fact links
    # ------------------------------------------------------------------

    def store_fact_links(self, links: list[FactLink]) -> int:
        """Store fact links, returning the number stored."""
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
        """Delete all links where fact_id is source or target. Returns count deleted."""
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
        """Convert a sqlite3.Row to a FactLink dataclass."""
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

    def search_tool_outputs(self, query: str, limit: int = 5) -> list:
        from ..types import QuoteResult

        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT t.ref, t.tool_name,
                          snippet(tool_outputs_fts, 0, '>>>', '<<<', '...', 100) as snippet
                   FROM tool_outputs_fts fts
                   JOIN tool_outputs t ON t.rowid = fts.rowid
                   WHERE tool_outputs_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                [_sanitize_fts_query(query), limit],
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
