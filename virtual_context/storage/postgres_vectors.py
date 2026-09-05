"""Opt-in exact pgvector ranking and explicit derived-cache migration.

The JSON embedding remains authoritative. Database triggers keep the optional
cache synchronized even when an older writer does not know about its columns;
such a writer has no model attestation and makes activation fail closed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from ..types import SpeakerRetrievalContext

VECTOR_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
_TABLE_KEYS = {
    "segment_chunks": ("segment_ref", "chunk_index"),
    "canonical_turn_chunks": ("conversation_id", "canonical_turn_id", "side", "chunk_index"),
}
_MIGRATION_LOCK = 0x7663566563746F72


def _residue_sql(prefix: str = "") -> str:
    col = f"{prefix}." if prefix else ""
    return (
        f"({col}embedding_model IS DISTINCT FROM '{VECTOR_MODEL}'"
        f" OR {col}embedding_zero IS NULL"
        f" OR {col}embedding_source_hash IS DISTINCT FROM md5({col}embedding_json)"
        f" OR ({col}embedding IS NULL AND NOT {col}embedding_zero)"
        f" OR ({col}embedding IS NOT NULL AND {col}embedding_zero))"
    )


def _query_vector(values: Sequence[float]) -> str | None:
    if len(values) != VECTOR_DIM:
        raise ValueError(f"Native semantic search requires {VECTOR_DIM} dimensions")
    vector = [float(value) for value in values]
    if any(isinstance(value, bool) for value in values) or not all(map(math.isfinite, vector)):
        raise ValueError("Native semantic search requires finite numeric vectors")
    if not any(vector):
        return None  # cosine_similarity returns zero, below every supported threshold
    return json.dumps(vector, separators=(",", ":"))


_SYNC_FUNCTION_SQL = f"""
CREATE OR REPLACE FUNCTION public.vc_sync_semantic_vector_v1()
RETURNS trigger LANGUAGE plpgsql AS $function$
DECLARE
    source jsonb;
    component jsonb;
    value double precision;
    nonzero boolean := FALSE;
BEGIN
    NEW.embedding := NULL;
    NEW.embedding_zero := FALSE;
    NEW.embedding_source_hash := md5(NEW.embedding_json);
    NEW.embedding_model := COALESCE(
        current_setting('virtual_context.embedding_model', TRUE), ''
    );
    IF NEW.embedding_model <> '{VECTOR_MODEL}' THEN
        RETURN NEW;
    END IF;
    BEGIN
        source := NEW.embedding_json::jsonb;
        IF jsonb_typeof(source) <> 'array' OR jsonb_array_length(source) <> {VECTOR_DIM} THEN
            RETURN NEW;
        END IF;
        FOR component IN SELECT jsonb_array_elements(source) LOOP
            IF jsonb_typeof(component) <> 'number' THEN
                RETURN NEW;
            END IF;
            value := component::text::double precision;
            IF value IN ('Infinity'::double precision, '-Infinity'::double precision, 'NaN'::double precision) THEN
                RETURN NEW;
            END IF;
            nonzero := nonzero OR value <> 0;
        END LOOP;
        IF NOT nonzero THEN
            NEW.embedding_zero := TRUE;
        ELSE
            NEW.embedding := source::text::public.vector({VECTOR_DIM});
        END IF;
    EXCEPTION WHEN data_exception THEN
        -- Malformed, wrong-dimensional or non-finite legacy JSON remains
        -- canonical, but can never masquerade as a usable vector cache.
        NEW.embedding := NULL;
        NEW.embedding_zero := FALSE;
    END;
    RETURN NEW;
END;
$function$
"""


class PostgresVectorSearchMixin:
    """Optional cache operations; ordinary store bootstrap performs no DDL."""

    def _set_semantic_vector_write_model(self, conn, embedding_model: str) -> None:
        # SET LOCAL is harmless before migration and cannot leak across pooled
        # connections or failed transactions. Unknown/unsupported writers make
        # the cache ineligible rather than guessing an embedding model.
        conn.execute(
            "SELECT set_config('virtual_context.embedding_model', %s, TRUE)",
            (embedding_model or "",),
        )

    @staticmethod
    def _vector_schema_installed(conn) -> bool:
        extension = conn.execute(
            "SELECT 1 FROM pg_extension e JOIN pg_namespace n ON n.oid=e.extnamespace "
            "WHERE e.extname='vector' AND n.nspname='public'"
        ).fetchone()
        if not extension:
            return False
        rows = conn.execute(
            """SELECT c.relname AS table_name, a.attname AS column_name,
                      format_type(a.atttypid, a.atttypmod) AS type_name
                 FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
                 JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname='public'
                  AND c.relname IN ('segment_chunks','canonical_turn_chunks')
                  AND a.attname IN ('embedding','embedding_zero','embedding_source_hash','embedding_model')
                  AND a.attnum > 0 AND NOT a.attisdropped"""
        ).fetchall()
        actual = {(row["table_name"], row["column_name"]): row["type_name"] for row in rows}
        expected = {"embedding": "vector(384)", "embedding_zero": "boolean",
                    "embedding_source_hash": "text", "embedding_model": "text"}
        if any(actual.get((table, column)) not in (kind, f"public.{kind}")
               for table in _TABLE_KEYS for column, kind in expected.items()):
            return False
        triggers = conn.execute(
            """SELECT c.relname AS table_name FROM pg_trigger t
                 JOIN pg_class c ON c.oid=t.tgrelid
                 JOIN pg_namespace n ON n.oid=c.relnamespace
                 JOIN pg_proc p ON p.oid=t.tgfoid
                WHERE n.nspname='public' AND NOT t.tgisinternal
                  AND t.tgenabled IN ('O','A')
                  AND t.tgname='vc_sync_semantic_vector_v1'
                  AND p.proname='vc_sync_semantic_vector_v1'
                  AND c.relname IN ('segment_chunks','canonical_turn_chunks')"""
        ).fetchall()
        return {row["table_name"] for row in triggers} == set(_TABLE_KEYS)

    def _vector_ready_on_connection(self, conn, model: str) -> bool:
        if model != VECTOR_MODEL or not self._vector_schema_installed(conn):
            return False
        for table in _TABLE_KEYS:
            if conn.execute(
                f"SELECT 1 FROM public.{table} WHERE {_residue_sql()} LIMIT 1"
            ).fetchone():
                return False
        return True

    def vector_search_ready(self, model: str) -> bool:
        if model != VECTOR_MODEL:
            return False
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return self._vector_ready_on_connection(conn, model)

    @staticmethod
    def _semantic_vector_residue_report(conn, table):
        rows = conn.execute(
            f"SELECT embedding_model AS model, count(*) AS count FROM public.{table} "
            f"WHERE {_residue_sql()} GROUP BY embedding_model ORDER BY embedding_model NULLS FIRST"
        ).fetchall()
        return {"residue": sum(int(row["count"]) for row in rows),
                "residue_by_model": [{"model": row["model"], "rows": int(row["count"])} for row in rows]}

    def migrate_semantic_vectors(
        self, *, dry_run: bool = True, batch_size: int = 1000,
        model: str = VECTOR_MODEL,
    ) -> dict:
        """Explicit additive migration and bounded backfill, never bootstrap.

        Applying attests that existing chunk JSON was produced with ``model``.
        Legacy chunk tables have no model history, so this must be confirmed by
        the operator. Re-running repairs residue without touching valid rows.
        """
        if model != VECTOR_MODEL:
            raise ValueError(f"Semantic vector migration supports only {VECTOR_MODEL}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 10000:
            raise ValueError("batch_size must be between 1 and 10000")
        with self.pool.connection() as conn:
            available = conn.execute(
                "SELECT default_version, installed_version FROM pg_available_extensions WHERE name='vector'"
            ).fetchone()
            installed = self._vector_schema_installed(conn)
            report = {"dry_run": dry_run, "model": model, "available": bool(available),
                      "schema_installed": installed, "tables": {}, "ready": False}
            for table in _TABLE_KEYS:
                row = conn.execute(f"SELECT count(*) AS count FROM public.{table}").fetchone()
                report["tables"][table] = {"rows": int(row["count"]), "backfilled": 0}
                if installed:
                    report["tables"][table].update(self._semantic_vector_residue_report(conn, table))
                else:
                    # No attestation can be inferred from JSON dimensionality.
                    report["tables"][table].update(residue=int(row["count"]), residue_by_model=None)
            if dry_run:
                report["ready"] = self._vector_ready_on_connection(conn, model)
                return report
            if not available:
                raise RuntimeError("pgvector is unavailable; install the PostgreSQL vector extension package first")
            if not conn.autocommit or conn.info.transaction_status != 0:
                raise RuntimeError("Semantic vector migration requires an idle autocommit connection")
            lock = conn.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (_MIGRATION_LOCK,)).fetchone()
            if not lock["acquired"]:
                raise RuntimeError("Another semantic vector migration is running")
            try:
                with conn.transaction():
                    conn.execute("SET LOCAL lock_timeout = '2s'")
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
                    for table in _TABLE_KEYS:
                        conn.execute(
                            f"""ALTER TABLE public.{table}
                                ADD COLUMN IF NOT EXISTS embedding public.vector(384),
                                ADD COLUMN IF NOT EXISTS embedding_zero boolean NOT NULL DEFAULT FALSE,
                                ADD COLUMN IF NOT EXISTS embedding_source_hash text,
                                ADD COLUMN IF NOT EXISTS embedding_model text NOT NULL DEFAULT ''"""
                        )
                    conn.execute(_SYNC_FUNCTION_SQL)
                    for table in _TABLE_KEYS:
                        conn.execute(f"DROP TRIGGER IF EXISTS vc_sync_semantic_vector_v1 ON public.{table}")
                        conn.execute(
                            f"""CREATE TRIGGER vc_sync_semantic_vector_v1
                                BEFORE INSERT OR UPDATE OF embedding_json, embedding, embedding_zero,
                                    embedding_source_hash, embedding_model ON public.{table}
                                FOR EACH ROW EXECUTE FUNCTION public.vc_sync_semantic_vector_v1()"""
                        )
                if not self._vector_schema_installed(conn):
                    raise RuntimeError("Semantic vector schema has incompatible columns or extension schema")
                for table, keys in _TABLE_KEYS.items():
                    after = None
                    key_sql = ", ".join(keys)
                    while True:
                        params = []
                        clause = ""
                        if after is not None:
                            clause = f" AND ({key_sql}) > ({', '.join(['%s'] * len(keys))})"
                            params.extend(after)
                        params.append(batch_size)
                        with conn.transaction():
                            conn.execute("SET LOCAL lock_timeout = '2s'")
                            conn.execute("SELECT set_config('virtual_context.embedding_model', %s, TRUE)", (model,))
                            rows = conn.execute(
                                f"""WITH batch AS MATERIALIZED (
                                    SELECT {key_sql} FROM public.{table}
                                     WHERE {_residue_sql()}
                                       AND (embedding_model IS NULL OR embedding_model IN ('', '{VECTOR_MODEL}')){clause}
                                     ORDER BY {key_sql} LIMIT %s FOR UPDATE
                                ), updated AS (UPDATE public.{table} target
                                     SET embedding_json = target.embedding_json
                                    FROM batch b
                                   WHERE {' AND '.join(f'target.{key}=b.{key}' for key in keys)}
                                RETURNING {', '.join('target.' + key for key in keys)})
                                SELECT {key_sql} FROM updated ORDER BY {key_sql}""",
                                params,
                            ).fetchall()
                        if not rows:
                            break
                        report["tables"][table]["backfilled"] += len(rows)
                        # Preserve the database's text collation in keyset
                        # continuation, including non-ASCII conversation ids.
                        after = tuple(rows[-1][key] for key in keys)
                    index = f"idx_{table}_vector_residue"
                    validity = conn.execute(
                        "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid=to_regclass(%s)",
                        (f"public.{index}",),
                    ).fetchone()
                    if validity and not validity["indisvalid"]:
                        conn.execute(f"DROP INDEX CONCURRENTLY public.{index}")
                    conn.execute(
                        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index} ON public.{table} ((1)) WHERE {_residue_sql()}"
                    )
                    report["tables"][table].update(self._semantic_vector_residue_report(conn, table))
                report["schema_installed"] = True
                report["ready"] = self._vector_ready_on_connection(conn, model)
                return report
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK,))

    def _semantic_vector_page(
        self, query_embedding, *, projection: str, source: str, predicates: str,
        payload_projection: str, payload_join: str,
        params: list, key_names: tuple[str, ...], limit: int,
        after: tuple | None, min_similarity: float,
    ) -> list[dict]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("Vector page limit must be between 1 and 1000")
        if not math.isfinite(min_similarity) or not 0.25 <= min_similarity <= 1:
            raise ValueError("Native semantic similarity threshold must be between 0.25 and 1")
        vector = _query_vector(query_embedding)
        if vector is None:
            return []
        cursor_names = ("distance", *key_names)
        where = "distance <= %s"
        page_params = [vector, *params, 1.0 - min_similarity]
        if after is not None:
            if len(after) != len(cursor_names):
                raise ValueError("Invalid semantic vector continuation cursor")
            where += f" AND ({', '.join(cursor_names)}) > ({', '.join(['%s'] * len(after))})"
            page_params.extend(after)
        page_params.append(limit)
        # Materialization separates the complete exact distance calculation
        # from ORDER BY, so an optional ANN index cannot alter recall.
        query = f"""WITH scored AS MATERIALIZED (
            SELECT {projection}, chunk.embedding OPERATOR(public.<=>) %s::public.vector AS distance
              FROM {source}
             WHERE chunk.embedding IS NOT NULL AND NOT chunk.embedding_zero {predicates}
        ), selected AS MATERIALIZED (
            SELECT *, 1.0 - distance AS similarity FROM scored
             WHERE {where} ORDER BY {', '.join(cursor_names)} LIMIT %s
        ) SELECT selected.*, {payload_projection}
            FROM selected {payload_join}
           ORDER BY {', '.join('selected.' + name for name in cursor_names)}"""
        with self.pool.connection() as conn, conn.transaction():
            # Readiness and candidate selection share one snapshot; a stale
            # concurrent writer cannot slip unranked chunks past the gate.
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            if not self._vector_ready_on_connection(conn, VECTOR_MODEL):
                raise RuntimeError("Native vector search requires a complete semantic vector migration; run admin migrate-semantic-vectors")
            rows = conn.execute(query, page_params).fetchall()
        results = []
        for raw in rows:
            row = dict(raw)
            if "canonical_turn_id" in row:
                row["canonical_turn_id"] = str(row["canonical_turn_id"])
            row["cursor"] = tuple(row[name] for name in cursor_names)
            results.append(row)
        return results

    def search_segment_chunks_by_embedding(
        self, query_embedding, *, conversation_id: str | None = None,
        limit: int = 200, after: tuple | None = None, min_similarity: float = 0.25,
    ) -> list[dict]:
        return self._semantic_vector_page(
            query_embedding, projection="chunk.segment_ref, chunk.chunk_index",
            source="public.segment_chunks chunk JOIN public.segments s ON s.ref=chunk.segment_ref",
            payload_projection="payload.text",
            payload_join="JOIN public.segment_chunks payload ON payload.segment_ref=selected.segment_ref AND payload.chunk_index=selected.chunk_index",
            predicates=" AND s.conversation_id=%s" if conversation_id is not None else "",
            params=[conversation_id] if conversation_id is not None else [],
            key_names=("segment_ref", "chunk_index"), limit=limit, after=after,
            min_similarity=min_similarity,
        )

    def search_canonical_turn_chunks_by_embedding(
        self, query_embedding, *, conversation_id: str | None = None,
        limit: int = 200, after: tuple | None = None, min_similarity: float = 0.25,
    ) -> list[dict]:
        predicates = " AND chunk.side <> 'subject'"
        params = []
        if conversation_id is not None:
            predicates += " AND chunk.conversation_id=%s"
            params.append(conversation_id)
        rows = self._semantic_vector_page(
            query_embedding,
            projection="chunk.conversation_id, chunk.canonical_turn_id, cto.turn_number, cto.sort_key, chunk.side, chunk.chunk_index",
            source="public.canonical_turn_chunks chunk JOIN public.canonical_turns_ordinal cto ON cto.conversation_id=chunk.conversation_id AND cto.canonical_turn_id=chunk.canonical_turn_id",
            payload_projection="payload.text, row_to_json(physical) AS physical_row",
            payload_join="""JOIN public.canonical_turn_chunks payload
                ON payload.conversation_id=selected.conversation_id
               AND payload.canonical_turn_id=selected.canonical_turn_id
               AND payload.side=selected.side AND payload.chunk_index=selected.chunk_index
                JOIN public.canonical_turns physical
                  ON physical.conversation_id=selected.conversation_id
                 AND physical.canonical_turn_id=selected.canonical_turn_id""",
            predicates=predicates, params=params,
            key_names=("conversation_id", "sort_key", "side", "chunk_index", "canonical_turn_id"),
            limit=limit, after=after, min_similarity=min_similarity,
        )
        from .postgres import _row_to_canonical_turn

        for row in rows:
            physical = dict(row["physical_row"])
            physical["turn_number"] = row["turn_number"]
            row["physical_row"] = _row_to_canonical_turn(physical)
        return rows

    def search_speaker_turn_chunks_by_embedding(
        self, query_embedding, *, speaker_context: SpeakerRetrievalContext,
        conversation_id: str | None = None, limit: int = 200,
        after: tuple | None = None, min_similarity: float = 0.25,
    ) -> list[dict]:
        # Import only at call time: the concrete store inherits this mixin.
        from .postgres import _speaker_canonical_scope_sql

        scope = _speaker_canonical_scope_sql(speaker_context, conversation_id, prefix="ct")
        if scope is None:
            return []
        predicates, params = scope
        return self._semantic_vector_page(
            query_embedding,
            projection="chunk.conversation_id, chunk.canonical_turn_id, -1 AS turn_number, ct.sort_key, chunk.side, chunk.chunk_index",
            source="public.canonical_turn_chunks chunk JOIN public.canonical_turns ct ON ct.conversation_id=chunk.conversation_id AND ct.canonical_turn_id=chunk.canonical_turn_id",
            payload_projection="payload.text",
            payload_join="""JOIN public.canonical_turn_chunks payload
                ON payload.conversation_id=selected.conversation_id
               AND payload.canonical_turn_id=selected.canonical_turn_id
               AND payload.side=selected.side AND payload.chunk_index=selected.chunk_index""",
            predicates=predicates, params=list(params),
            key_names=("conversation_id", "sort_key", "side", "chunk_index", "canonical_turn_id"),
            limit=limit, after=after, min_similarity=min_similarity,
        )
