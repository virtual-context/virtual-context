"""PostgresStore: storage backend using psycopg (PostgreSQL)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row

from ..core.store import ContextStore
from ..types import (
    ChunkEmbedding,
    ConversationStats,
    DepthLevel,
    EngineStateSnapshot,
    Fact,
    FactLink,
    FactSignal,
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

CREATE TABLE IF NOT EXISTS tag_summaries (
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

CREATE TABLE IF NOT EXISTS engine_state (
    conversation_id TEXT PRIMARY KEY,
    compacted_through INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    turn_tag_entries TEXT NOT NULL,
    saved_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS segment_chunks (
    segment_ref TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (segment_ref, chunk_index)
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
            turn_count=metadata_raw.get("turn_count", 0),
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
            turn_count=metadata_raw.get("turn_count", 0),
            session_date=metadata_raw.get("session_date", ""),
        ),
        created_at=_str_to_dt(row["created_at"]),
        start_timestamp=_str_to_dt(row["start_timestamp"]),
        end_timestamp=_str_to_dt(row["end_timestamp"]),
    )


class PostgresStore(ContextStore):
    """PostgreSQL storage backend with tsvector FTS and full protocol support."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None
        self.search_config = None  # set by engine after construction
        self._ensure_schema()

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

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
        # Migration: add raw_content columns to turn_messages
        try:
            conn.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_name='turn_messages' AND column_name='user_raw_content') THEN
                        ALTER TABLE turn_messages ADD COLUMN user_raw_content TEXT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_name='turn_messages' AND column_name='assistant_raw_content') THEN
                        ALTER TABLE turn_messages ADD COLUMN assistant_raw_content TEXT;
                    END IF;
                END $$;
            """)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tags_for_ref(self, ref: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag FROM segment_tags WHERE segment_ref = %s ORDER BY tag", (ref,)
        ).fetchall()
        return [r["tag"] for r in rows]

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

        with conn.transaction():
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

    def get_tag_aliases(self) -> dict[str, str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT alias, canonical FROM tag_aliases").fetchall()
        return {row["alias"]: row["canonical"] for row in rows}

    def set_tag_alias(self, alias: str, canonical: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO tag_aliases (alias, canonical) VALUES (%s, %s) ON CONFLICT (alias) DO UPDATE SET canonical = EXCLUDED.canonical",
            (alias, canonical),
        )

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

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete all segments, facts, engine state, turn messages, and tag summaries for a conversation. Returns segment count deleted."""
        conn = self._get_conn()
        with conn.transaction():
            cur = conn.execute(
                "DELETE FROM segments WHERE conversation_id = %s", (conversation_id,),
            )
            deleted = cur.rowcount
            conn.execute(
                "DELETE FROM engine_state WHERE conversation_id = %s", (conversation_id,),
            )
            conn.execute(
                "DELETE FROM facts WHERE conversation_id = %s", (conversation_id,),
            )
            conn.execute(
                "DELETE FROM turn_messages WHERE conversation_id = %s", (conversation_id,),
            )
            conn.execute(
                "DELETE FROM tag_summaries WHERE conversation_id = %s", (conversation_id,),
            )
        return deleted

    def save_tag_summary(self, tag_summary: TagSummary, conversation_id: str = "") -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tag_summaries
            (tag, conversation_id, summary, description, summary_tokens, source_segment_refs, source_turn_numbers,
             covers_through_turn, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (tag, conversation_id) DO UPDATE SET
                summary=EXCLUDED.summary, description=EXCLUDED.description,
                summary_tokens=EXCLUDED.summary_tokens,
                source_segment_refs=EXCLUDED.source_segment_refs,
                source_turn_numbers=EXCLUDED.source_turn_numbers,
                covers_through_turn=EXCLUDED.covers_through_turn,
                updated_at=EXCLUDED.updated_at""",
            (tag_summary.tag, conversation_id, tag_summary.summary, getattr(tag_summary, "description", ""),
             tag_summary.summary_tokens, json.dumps(tag_summary.source_segment_refs),
             json.dumps(tag_summary.source_turn_numbers), tag_summary.covers_through_turn,
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
                """SELECT ts.* FROM tag_summaries ts
                   WHERE NOT EXISTS (
                       SELECT 1 FROM segments s,
                              jsonb_array_elements_text(ts.source_segment_refs::jsonb) je
                       WHERE s.ref = je AND s.conversation_id != %s
                   )
                   OR EXISTS (
                       SELECT 1 FROM segments s,
                              jsonb_array_elements_text(ts.source_segment_refs::jsonb) je
                       WHERE s.ref = je AND s.conversation_id = %s
                   )
                   ORDER BY ts.tag""",
                (conversation_id, conversation_id),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tag_summaries ORDER BY tag").fetchall()
        return [
            TagSummary(
                tag=row["tag"], summary=row["summary"],
                description=row.get("description", ""),
                summary_tokens=row["summary_tokens"],
                source_segment_refs=json.loads(row["source_segment_refs"]),
                source_turn_numbers=json.loads(row["source_turn_numbers"]),
                covers_through_turn=row["covers_through_turn"],
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
            QuoteResult(text=row["excerpt"], tag=row["tool_name"], segment_ref=row["ref"], session_date="", match_type="tool_output")
            for row in rows
        ]

    # ------------------------------------------------------------------
    # StateStore
    # ------------------------------------------------------------------

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        conn = self._get_conn()
        entries_data = {
            "entries": [
                {
                    "turn_number": e.turn_number,
                    "tags": e.tags,
                    "primary_tag": e.primary_tag,
                    "sender": e.sender,
                    "fact_signals": [
                        {"subject": fs.subject, "verb": fs.verb, "object": fs.object,
                         "status": fs.status, "fact_type": fs.fact_type, "what": fs.what}
                        for fs in (e.fact_signals or [])
                    ] if e.fact_signals else [],
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
        }
        conn.execute(
            """INSERT INTO engine_state (conversation_id, compacted_through, turn_count, turn_tag_entries, saved_at)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                compacted_through=EXCLUDED.compacted_through, turn_count=EXCLUDED.turn_count,
                turn_tag_entries=EXCLUDED.turn_tag_entries, saved_at=EXCLUDED.saved_at""",
            (state.conversation_id, state.compacted_through, state.turn_count,
             json.dumps(entries_data), _dt_to_str(datetime.now(timezone.utc))),
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
        else:
            entries_list = []
            split_tags = set()
            working_set = {}
            fingerprint = ""
            request_captures = []
            provider = ""

        entries = []
        for e in entries_list:
            signals = []
            for fs in e.get("fact_signals", []):
                signals.append(FactSignal(
                    subject=fs.get("subject", ""), verb=fs.get("verb", ""),
                    object=fs.get("object", ""), status=fs.get("status", ""),
                    fact_type=fs.get("fact_type", "personal"), what=fs.get("what", ""),
                ))
            entries.append(TurnTagEntry(
                turn_number=e["turn_number"], tags=e["tags"],
                primary_tag=e.get("primary_tag", e["tags"][0] if e["tags"] else "_general"),
                fact_signals=signals if signals else None,
                sender=e.get("sender", ""),
            ))

        return EngineStateSnapshot(
            conversation_id=row["conversation_id"],
            compacted_through=row["compacted_through"],
            turn_count=row["turn_count"],
            turn_tag_entries=entries,
            split_processed_tags=split_tags,
            working_set=working_set,
            trailing_fingerprint=fingerprint,
            request_captures=request_captures,
            provider=provider,
        )

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM engine_state WHERE conversation_id = %s", (conversation_id,)).fetchone()
        if not row:
            return None
        return self._parse_engine_state_row(row)

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM engine_state ORDER BY compacted_through DESC, saved_at DESC LIMIT 1").fetchone()
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
            """INSERT INTO turn_messages
            (conversation_id, turn_number, user_content, assistant_content,
             user_raw_content, assistant_raw_content)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (conversation_id, turn_number) DO UPDATE SET
                user_content=EXCLUDED.user_content,
                assistant_content=EXCLUDED.assistant_content,
                user_raw_content=EXCLUDED.user_raw_content,
                assistant_raw_content=EXCLUDED.assistant_raw_content""",
            (conversation_id, turn_number, user_content, assistant_content,
             user_raw_content, assistant_raw_content),
        )

    def get_turn_messages(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, tuple[str, str, str | None, str | None]]:
        if not turn_numbers:
            return {}
        conn = self._get_conn()
        placeholders = ",".join("%s" for _ in turn_numbers)
        rows = conn.execute(
            f"""SELECT turn_number, user_content, assistant_content,
                       user_raw_content, assistant_raw_content
            FROM turn_messages
            WHERE conversation_id = %s AND turn_number IN ({placeholders})""",
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
                 WHERE fact_type = 'experience'
                   AND status = 'completed'
                   AND session_date >= %s AND session_date <= %s"""
        params: list = [start_date, end_date + "~"]
        if conversation_id:
            sql += " AND conversation_id = %s"
            params.append(conversation_id)
        sql += " ORDER BY session_date ASC LIMIT %s"
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
