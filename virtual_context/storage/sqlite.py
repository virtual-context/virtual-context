"""SQLiteStore: primary storage backend using stdlib sqlite3."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..core.store import ContextStore
from ..types import DepthLevel, EngineStateSnapshot, QuoteResult, SegmentMetadata, SessionStats, StoredSegment, StoredSummary, TagStats, TagSummary, TurnTagEntry, WorkingSetEntry

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS segments (
    ref TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
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
    session_id TEXT PRIMARY KEY,
    compacted_through INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    turn_tag_entries TEXT NOT NULL,
    saved_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_primary_tag ON segments(primary_tag);
CREATE INDEX IF NOT EXISTS idx_segments_created_at ON segments(created_at);
CREATE INDEX IF NOT EXISTS idx_segments_session_id ON segments(session_id);
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


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_segment(row: sqlite3.Row, tags: list[str]) -> StoredSegment:
    metadata_raw = json.loads(row["metadata_json"])
    return StoredSegment(
        ref=row["ref"],
        session_id=row["session_id"],
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
        ),
        created_at=_str_to_dt(row["created_at"]),
        start_timestamp=_str_to_dt(row["start_timestamp"]),
        end_timestamp=_str_to_dt(row["end_timestamp"]),
    )


def _extract_excerpt(text: str, query: str, context_chars: int = 200) -> str:
    """Extract text around the first occurrence of query."""
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:context_chars * 2]
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(query) + context_chars)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    return excerpt


class SQLiteStore(ContextStore):
    """SQLite-based storage with tag-overlap queries and FTS5 search."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
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
        # Migrations: add columns that didn't exist in earlier schema versions
        try:
            conn.execute("ALTER TABLE tag_summaries ADD COLUMN description TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()

    def _get_tags_for_ref(self, ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag FROM segment_tags WHERE segment_ref = ? ORDER BY tag",
            (ref,),
        ).fetchall()
        return [r["tag"] for r in rows]

    def store_segment(self, segment: StoredSegment) -> str:
        conn = self._get_conn()
        metadata_json = json.dumps({
            "entities": segment.metadata.entities,
            "key_decisions": segment.metadata.key_decisions,
            "action_items": segment.metadata.action_items,
            "date_references": segment.metadata.date_references,
            "turn_count": segment.metadata.turn_count,
        })

        conn.execute(
            """INSERT OR REPLACE INTO segments
            (ref, session_id, primary_tag, summary, full_text, messages_json,
             metadata_json, summary_tokens, full_tokens, compression_ratio,
             compaction_model, created_at, start_timestamp, end_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                segment.ref,
                segment.session_id,
                segment.primary_tag,
                segment.summary,
                segment.full_text,
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

        conn.commit()
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

        results = []
        for row in rows:
            seg_tags = self._get_tags_for_ref(row["ref"])
            results.append(_row_to_summary(row, seg_tags))
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
                    [query, *tags, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT s.* FROM segments_fts fts
                    JOIN segments s ON s.ref = fts.ref
                    WHERE segments_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?""",
                    [query, limit],
                ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 not available, fall back to LIKE
            like_query = f"%{query}%"
            if tags:
                placeholders = ",".join("?" * len(tags))
                rows = conn.execute(
                    f"""SELECT DISTINCT s.* FROM segments s
                    JOIN segment_tags st ON s.ref = st.segment_ref
                    WHERE s.summary LIKE ?
                    AND st.tag IN ({placeholders})
                    LIMIT ?""",
                    [like_query, *tags, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM segments WHERE summary LIKE ? LIMIT ?",
                    [like_query, limit],
                ).fetchall()

        results = []
        for row in rows:
            seg_tags = self._get_tags_for_ref(row["ref"])
            results.append(_row_to_summary(row, seg_tags))
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
                """SELECT fts.ref, s.primary_tag,
                          snippet(segments_fts_full, 1, '>>>', '<<<', '...', 40)
                   FROM segments_fts_full fts
                   JOIN segments s ON s.ref = fts.ref
                   WHERE segments_fts_full MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                [query, limit],
            ).fetchall()
            for row in rows:
                results.append(QuoteResult(
                    text=row[2],
                    tag=row[1],
                    segment_ref=row[0],
                    tags=self._get_tags_for_ref(row[0]),
                ))
            return results
        except sqlite3.OperationalError:
            pass

        # Fallback: LIKE search on full_text with manual excerpt
        like_query = f"%{query}%"
        rows = conn.execute(
            """SELECT ref, primary_tag, full_text FROM segments
               WHERE full_text LIKE ?
               LIMIT ?""",
            [like_query, limit],
        ).fetchall()
        for row in rows:
            excerpt = _extract_excerpt(row[2], query, context_chars=200)
            results.append(QuoteResult(
                text=excerpt,
                tag=row[1],
                segment_ref=row[0],
                tags=self._get_tags_for_ref(row[0]),
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

    def get_session_stats(self) -> list[SessionStats]:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT s.session_id,
                   COUNT(*) as segment_count,
                   COALESCE(SUM(s.full_tokens), 0) as total_full_tokens,
                   COALESCE(SUM(s.summary_tokens), 0) as total_summary_tokens,
                   MIN(s.created_at) as oldest,
                   MAX(s.created_at) as newest,
                   s.compaction_model
            FROM segments s
            WHERE s.session_id != ''
            GROUP BY s.session_id
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
                WHERE s.session_id = ?
                ORDER BY st.tag
            """, (row["session_id"],)).fetchall()

            results.append(SessionStats(
                session_id=row["session_id"],
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

    def delete_session(self, session_id: str) -> int:
        """Delete all segments for a given session_id. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM segments WHERE session_id = ?", (session_id,),
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
        results = []
        for row in rows:
            seg_tags = self._get_tags_for_ref(row["ref"])
            results.append(_row_to_segment(row, seg_tags))
        return results

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        conn = self._get_conn()
        entries_json = json.dumps([
            {
                "turn_number": e.turn_number,
                "message_hash": e.message_hash,
                "tags": e.tags,
                "primary_tag": e.primary_tag,
                "timestamp": _dt_to_str(e.timestamp),
            }
            for e in state.turn_tag_entries
        ])
        # Include split_processed_tags and working_set in the entries JSON blob
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
        })
        conn.execute(
            """INSERT OR REPLACE INTO engine_state
            (session_id, compacted_through, turn_count, turn_tag_entries, saved_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                state.session_id,
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
        else:
            entries_raw = raw
            split_processed_tags = []
            working_set_raw = []
        entries = [
            TurnTagEntry(
                turn_number=e["turn_number"],
                message_hash=e["message_hash"],
                tags=e["tags"],
                primary_tag=e["primary_tag"],
                timestamp=_str_to_dt(e["timestamp"]),
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
            session_id=row["session_id"],
            compacted_through=row["compacted_through"],
            turn_tag_entries=entries,
            turn_count=row["turn_count"],
            saved_at=_str_to_dt(row["saved_at"]),
            split_processed_tags=split_processed_tags,
            working_set=working_set,
        )

    def load_engine_state(self, session_id: str) -> EngineStateSnapshot | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM engine_state WHERE session_id = ?", (session_id,)
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

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
