"""Shared relational read boundaries and durable request-state transactions.

The concrete backends retain their SQL-specific provenance predicates and row
decoders. This module owns the common pagination and atomic exchange contracts.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import time
import uuid

from ..core.store_capabilities import RELATIONAL_CAPABILITIES
from .fact_mutations import FactMutationMixin


class RelationalStoreMixin(FactMutationMixin):
    capabilities = RELATIONAL_CAPABILITIES
    _relational_dialect = "sqlite"

    @property
    def _placeholder(self):
        return "%s" if self._relational_dialect == "postgres" else "?"

    @contextmanager
    def _relational_connection(self, *, write=False, scope=""):
        if self._relational_dialect == "postgres":
            with self.pool.connection() as conn, conn.transaction():
                if write:
                    key = int.from_bytes(hashlib.sha256(scope.encode()).digest()[:8], "big", signed=True)
                    conn.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
                else:
                    conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                yield conn
        else:
            conn = self._get_conn()
            # Reuse a caller-owned read snapshot; never commit its transaction.
            owned = not conn.in_transaction
            if write and not owned:
                raise RuntimeError("Request-state mutation requires its own transaction")
            if owned:
                conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield conn
                if owned:
                    conn.commit()
            except BaseException:
                if owned:
                    conn.rollback()
                raise

    def _ensure_request_state_schema(self):
        with self._relational_connection(write=True, scope="vc-request-state-schema") as conn:
            self._ensure_fact_decision_schema(conn)
            # PostgreSQL builds this potentially large index concurrently via
            # the explicit read-index migration, never under bootstrap locks.
            if self._relational_dialect != "postgres":
                conn.execute("""CREATE INDEX IF NOT EXISTS idx_canonical_turn_group_read
                    ON canonical_turns (conversation_id,turn_group_number,sort_key)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS pending_tool_exchanges (
                conversation_id TEXT NOT NULL, exchange_id TEXT NOT NULL,
                payload_json TEXT NOT NULL, payload_bytes INTEGER NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL, created_at DOUBLE PRECISION NOT NULL,
                claim_id TEXT, lease_until DOUBLE PRECISION,
                owner_version TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (conversation_id, exchange_id))""")
            if self._relational_dialect == "postgres":
                conn.execute("ALTER TABLE pending_tool_exchanges ADD COLUMN IF NOT EXISTS owner_version TEXT NOT NULL DEFAULT ''")
            elif not any(row["name"] == "owner_version" for row in conn.execute("PRAGMA table_info(pending_tool_exchanges)")):
                conn.execute("ALTER TABLE pending_tool_exchanges ADD COLUMN owner_version TEXT NOT NULL DEFAULT ''")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_pending_exchange_expiry
                ON pending_tool_exchanges (expires_at)""")

    def _canonical_decoder(self):
        if self._relational_dialect == "postgres":
            from .postgres import _row_to_canonical_turn
        else:
            from .sqlite import _row_to_canonical_turn
        return _row_to_canonical_turn

    def _exchange_owner_version(self, conn, conversation_id, *, create=False):
        """Fence hidden state against deletion, recreation and owner merges."""
        p = self._placeholder
        if create:
            # Only a new checkpoint needs a rowless local lifecycle. Lookup,
            # renewal and completion of unknown IDs must not create owners.
            # Materialization fences a concurrent delete against this INSERT.
            conn.execute(f"""INSERT INTO conversation_lifecycle
                (conversation_id,generation,deleted,updated_at) VALUES ({p},0,{p},{p})
                ON CONFLICT (conversation_id) DO NOTHING""",
                (conversation_id,False,datetime.now(timezone.utc).isoformat()))
        locking = " FOR SHARE" if self._relational_dialect == "postgres" else ""
        lifecycle = conn.execute(f"SELECT generation,deleted FROM conversation_lifecycle WHERE conversation_id={p}{locking}",(conversation_id,)).fetchone()
        owner = conn.execute(f"SELECT lifecycle_epoch,phase,deleted_at FROM conversations WHERE conversation_id={p}{locking}",(conversation_id,)).fetchone()
        if lifecycle is None or lifecycle["deleted"] or (owner and (owner["phase"] in ("deleted","merged") or owner["deleted_at"] is not None)):
            return None
        return json.dumps([lifecycle["generation"],owner["lifecycle_epoch"] if owner else None])

    @staticmethod
    def _page_limit(limit):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Page limit must be between 1 and 1000")

    def get_segment_chunk_embedding_page(self, *, conversation_id=None, limit=200, after=None):
        self._page_limit(limit)
        p = self._placeholder
        predicates, params = [], []
        if conversation_id is not None:
            predicates.append(f"s.conversation_id={p}")
            params.append(conversation_id)
        if after is not None:
            if len(after) != 2:
                raise ValueError("Invalid segment embedding cursor")
            predicates.append(f"(chunk.segment_ref,chunk.chunk_index)>({p},{p})")
            params.extend(after)
            # PostgreSQL can retain a generic merge/nested-loop plan that
            # repeatedly walks the entire source prefix as the cursor moves.
            # State the implied lower bound on both sides of the join so
            # continuation starts at the cursor under either plan kind.
            predicates.append(f"s.ref>={p}")
            params.append(after[0])
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        session_date = ("s.metadata_json::jsonb->>'session_date'"
                        if self._relational_dialect == "postgres"
                        else "json_extract(s.metadata_json,'$.session_date')")
        query = f"""SELECT chunk.segment_ref,chunk.chunk_index,chunk.text,chunk.embedding_json,
            s.conversation_id,s.primary_tag,{session_date} AS session_date
            FROM segment_chunks chunk JOIN segments s ON s.ref=chunk.segment_ref
            {where} ORDER BY chunk.segment_ref,chunk.chunk_index LIMIT {p}"""
        with self._relational_connection() as conn:
            rows = conn.execute(query, [*params, limit]).fetchall()
            # Hydrate display metadata only for this page. Full segment bodies
            # are neither needed for ranking nor safe to fetch per candidate.
            refs = list(dict.fromkeys(row["segment_ref"] for row in rows))
            tags = {ref: [] for ref in refs}
            for offset in range(0, len(refs), 400):
                batch = refs[offset:offset + 400]
                for tag in conn.execute(
                    f"SELECT segment_ref,tag FROM segment_tags WHERE segment_ref IN ({','.join([p] * len(batch))}) ORDER BY tag",
                    batch,
                ).fetchall():
                    tags[tag["segment_ref"]].append(tag["tag"])
        result = []
        for raw in rows:
            row = dict(raw)
            row["tags"] = tags[row["segment_ref"]]
            row["session_date"] = row["session_date"] or ""
            row["embedding"] = json.loads(row.pop("embedding_json"))
            row["cursor"] = (row["segment_ref"], row["chunk_index"])
            result.append(row)
        return result

    def get_canonical_turn_chunk_embedding_page(
        self, *, conversation_id=None, speaker_context=None, limit=200, after=None,
    ):
        self._page_limit(limit)
        p = self._placeholder
        if speaker_context is None:
            source = "canonical_turns_ordinal"
            # Physical order is stable when an earlier source is inserted;
            # the derived ordinal is presentation metadata, never a cursor.
            order_value, ordinal = "ct.sort_key", "ct.turn_number"
            predicates = f" AND chunk.conversation_id={p}" if conversation_id is not None else ""
            params = [conversation_id] if conversation_id is not None else []
        else:
            if self._relational_dialect == "postgres":
                from .postgres import _speaker_canonical_scope_sql
            else:
                from .sqlite import _speaker_canonical_scope_sql
            scope = _speaker_canonical_scope_sql(speaker_context, conversation_id, prefix="ct")
            if scope is None:
                return []
            predicates, params = scope
            source, order_value, ordinal = "canonical_turns", "ct.sort_key", "-1"
        keys = ("chunk.conversation_id", order_value, "chunk.side", "chunk.chunk_index", "chunk.canonical_turn_id")
        if after is not None:
            if len(after) != len(keys):
                raise ValueError("Invalid canonical embedding cursor")
            predicates += f" AND ({','.join(keys)})>({','.join([p]*len(keys))})"
            params = [*params, *after]
        query = f"""SELECT chunk.conversation_id,chunk.canonical_turn_id,
            {ordinal} AS turn_number,{order_value} AS order_value,
            chunk.side,chunk.chunk_index,chunk.text,chunk.embedding_json
            FROM canonical_turn_chunks chunk JOIN {source} ct
              ON ct.conversation_id=chunk.conversation_id
             AND ct.canonical_turn_id=chunk.canonical_turn_id
            WHERE 1=1 {predicates} ORDER BY {','.join(keys)} LIMIT {p}"""
        with self._relational_connection() as conn:
            rows = [dict(row) for row in conn.execute(query, [*params, limit]).fetchall()]
            physical = {}
            if rows and speaker_context is None:
                identities = list(dict.fromkeys((row["conversation_id"], str(row["canonical_turn_id"])) for row in rows))
                # Two binds per physical row; keep SQLite's oldest supported
                # variable limit safe even when callers request a large page.
                for offset in range(0, len(identities), 400):
                    batch = identities[offset:offset+400]
                    pairs = " OR ".join(f"(conversation_id={p} AND canonical_turn_id={p})" for _ in batch)
                    raw_rows = conn.execute("SELECT * FROM canonical_turns WHERE " + pairs,
                                            [value for pair in batch for value in pair]).fetchall()
                    physical.update({(row["conversation_id"], str(row["canonical_turn_id"])): dict(row) for row in raw_rows})
            decoder = self._canonical_decoder()
            for row in rows:
                row["canonical_turn_id"] = str(row["canonical_turn_id"])
                row["embedding"] = json.loads(row.pop("embedding_json"))
                row["cursor"] = (row["conversation_id"], row.pop("order_value"), row["side"], row["chunk_index"], row["canonical_turn_id"])
                raw = physical.get((row["conversation_id"], row["canonical_turn_id"]))
                if raw is not None:
                    row["physical_row"] = decoder({**raw, "turn_number": row["turn_number"]})
        return rows

    def get_canonical_turn_rows_by_group(self, conversation_id, turn_group_numbers, *, internal_validation=False):
        if internal_validation is not True:
            raise PermissionError("Physical group reads require internal validation authority")
        groups = list(dict.fromkeys(turn_group_numbers))
        if any(type(group) is not int or group < 0 for group in groups):
            raise ValueError("Physical group ids must be nonnegative integers")
        if len(groups) > 500:
            raise ValueError("At most 500 source groups may be hydrated at once")
        if not groups:
            return []
        p = self._placeholder
        with self._relational_connection() as conn:
            rows = conn.execute(f"""SELECT * FROM canonical_turns
                WHERE conversation_id={p} AND turn_group_number IN ({','.join([p]*len(groups))})
                ORDER BY sort_key,canonical_turn_id""", [conversation_id, *groups]).fetchall()
        decoder = self._canonical_decoder()
        return [decoder(row) for row in rows]

    def get_canonical_turn_rows_by_source_message_ids(self, conversation_id, source_message_ids, *, internal_validation=False):
        if internal_validation is not True:
            raise PermissionError("Physical source reads require internal validation authority")
        ids = list(dict.fromkeys(source_message_ids))
        if len(ids) > 500 or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("At most 500 nonempty source message ids are allowed")
        if not ids:
            return []
        p = self._placeholder
        with self._relational_connection() as conn:
            rows = conn.execute(f"""SELECT * FROM canonical_turns
                WHERE conversation_id={p} AND source_message_id IN ({','.join([p]*len(ids))})
                ORDER BY sort_key,canonical_turn_id""", [conversation_id, *ids]).fetchall()
        decoder = self._canonical_decoder()
        return [decoder(row) for row in rows]

    def get_compaction_watermark(self, conversation_id):
        """Return the exact compacted prefix using only streamed scalar rows.

        A single snapshot includes the explicit/legacy grouping decision and
        the prefix. Python whitespace semantics are supplied to both SQL
        dialects rather than approximated with SQL's space-only TRIM.
        """
        p = self._placeholder
        from ..core.canonical_turns import STRIP_WHITESPACE
        whitespace = STRIP_WHITESPACE
        trim = "btrim" if self._relational_dialect == "postgres" else "trim"
        user = f"CASE WHEN {trim}(COALESCE(user_content,''),{p})<>'' THEN 1 ELSE 0 END"
        assistant = f"CASE WHEN {trim}(COALESCE(assistant_content,''),{p})<>'' THEN 1 ELSE 0 END"
        with self._relational_connection() as conn:
            legacy = conn.execute(f"SELECT 1 FROM canonical_turns WHERE conversation_id={p} AND (turn_group_number IS NULL OR turn_group_number<0) LIMIT 1", (conversation_id,)).fetchone() is not None
            if legacy:
                query = f"""SELECT CASE WHEN COALESCE(user_content,'')<>'' THEN 1 ELSE 0 END AS has_user,
                    CASE WHEN COALESCE(assistant_content,'')<>'' THEN 1 ELSE 0 END AS has_assistant,
                    {user} AS users,{assistant} AS assistants,
                    CASE WHEN compacted_at IS NULL OR compacted_at='' THEN 1 ELSE 0 END AS incomplete
                    FROM canonical_turns WHERE conversation_id={p} ORDER BY sort_key,canonical_turn_id"""
            else:
                query = f"""SELECT turn_group_number AS group_id,SUM({user}) AS users,
                    SUM({assistant}) AS assistants,
                    SUM(CASE WHEN compacted_at IS NULL OR compacted_at='' THEN 1 ELSE 0 END) AS incomplete
                    FROM canonical_turns WHERE conversation_id={p}
                    GROUP BY turn_group_number ORDER BY turn_group_number"""
            # psycopg's ordinary cursor buffers a complete result even when
            # fetchmany is used; a named cursor keeps the wire window bounded.
            cursor = conn.cursor(name="vc_watermark_"+uuid.uuid4().hex) if self._relational_dialect == "postgres" else conn.cursor()
            try:
                cursor.execute(query, (whitespace, whitespace, conversation_id))

                def rows():
                    while batch := cursor.fetchmany(200):
                        yield from batch

                def groups():
                    if not legacy:
                        for row in rows():
                            yield row["group_id"], row["users"], row["assistants"], row["incomplete"]
                        return
                    pending = None
                    index = 0
                    for row in rows():
                        value = (row["users"],row["assistants"],row["incomplete"])
                        if row["has_user"] or not row["has_assistant"]:
                            if pending is not None:
                                yield index, *pending
                                index += 1
                                pending = None
                        if row["has_user"] and not row["has_assistant"]:
                            pending = value
                        elif row["has_assistant"]:
                            if pending is not None:
                                value = tuple(a+b for a,b in zip(pending,value))
                                pending = None
                            yield index, *value
                            index += 1
                    if pending is not None:
                        yield index, *pending

                count, last = 0, -1
                for group, users, assistants, incomplete in groups():
                    if users != 1 or assistants != 1 or incomplete:
                        break
                    count, last = count+2, int(group)
                return count, last
            finally:
                cursor.close()

    @staticmethod
    def _exchange_identity(conversation_id, exchange_id):
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("Pending exchange requires a conversation")
        if not isinstance(exchange_id, str) or not exchange_id or len(exchange_id) > 200:
            raise ValueError("Invalid pending exchange id")

    def put_pending_exchange(self, conversation_id, exchange_id, payload_json, *, expires_at, max_entries=4, max_bytes=2097152):
        self._exchange_identity(conversation_id, exchange_id)
        now = time.time()
        if not math.isfinite(expires_at) or not now < expires_at <= now + 86400:
            raise ValueError("Pending exchange expiry must be within the next day")
        if type(max_entries) is not int or not 1 <= max_entries <= 16 or type(max_bytes) is not int or not 1 <= max_bytes <= 8388608:
            raise ValueError("Invalid pending exchange capacity")
        size = len(payload_json.encode("utf-8"))
        if size > max_bytes:
            return False
        json.loads(payload_json)
        p = self._placeholder
        with self._relational_connection(write=True, scope="exchange:"+conversation_id) as conn:
            owner_version = self._exchange_owner_version(conn,conversation_id,create=True)
            if owner_version is None:
                return False
            # Retain expired entries until a subsequent insert, allowing an
            # id-less protocol to identify a stale exchange before cleanup.
            conn.execute(f"DELETE FROM pending_tool_exchanges WHERE conversation_id={p} AND (expires_at<={p} OR owner_version<>{p})", (conversation_id, now, owner_version))
            old = conn.execute(f"SELECT payload_json,expires_at FROM pending_tool_exchanges WHERE conversation_id={p} AND exchange_id={p}", (conversation_id, exchange_id)).fetchone()
            if old is not None:
                return old["payload_json"] == payload_json
            capacity = conn.execute(f"SELECT count(*) AS count,COALESCE(sum(payload_bytes),0) AS bytes FROM pending_tool_exchanges WHERE conversation_id={p}", (conversation_id,)).fetchone()
            if capacity["count"] >= max_entries or capacity["bytes"] + size > max_bytes:
                return False
            conn.execute(f"INSERT INTO pending_tool_exchanges (conversation_id,exchange_id,payload_json,payload_bytes,expires_at,created_at,owner_version) VALUES ({','.join([p]*7)})", (conversation_id, exchange_id, payload_json, size, expires_at, now,owner_version))
        return True

    def list_pending_exchanges(self, conversation_id, *, now):
        self._exchange_identity(conversation_id, "list")
        with self._relational_connection() as conn:
            return [row["exchange_id"] for row in conn.execute(f"SELECT exchange_id FROM pending_tool_exchanges WHERE conversation_id={self._placeholder} ORDER BY created_at,exchange_id LIMIT 16", (conversation_id,)).fetchall()]

    def get_pending_exchange(self, conversation_id, exchange_id, *, now):
        """Inspect a live checkpoint without borrowing its execution lease."""
        self._exchange_identity(conversation_id, exchange_id)
        if not math.isfinite(now):
            raise ValueError("Invalid pending exchange time")
        p = self._placeholder
        with self._relational_connection() as conn:
            lifecycle = conn.execute(f"SELECT generation,deleted FROM conversation_lifecycle WHERE conversation_id={p}",(conversation_id,)).fetchone()
            owner = conn.execute(f"SELECT lifecycle_epoch,phase,deleted_at FROM conversations WHERE conversation_id={p}",(conversation_id,)).fetchone()
            if lifecycle is None or lifecycle["deleted"] or (owner and (owner["phase"] in ("deleted","merged") or owner["deleted_at"] is not None)):
                return None
            version = json.dumps([lifecycle["generation"],owner["lifecycle_epoch"] if owner else None])
            row = conn.execute(f"""SELECT payload_json FROM pending_tool_exchanges
                WHERE conversation_id={p} AND exchange_id={p} AND expires_at>{p} AND owner_version={p}""",
                (conversation_id,exchange_id,now,version)).fetchone()
            return row["payload_json"] if row else None

    def claim_pending_exchange(self, conversation_id, exchange_id, claim_id, *, now, lease_seconds=120):
        self._exchange_identity(conversation_id, exchange_id)
        if not claim_id or not math.isfinite(now) or not 0 < lease_seconds <= 600:
            raise ValueError("Invalid pending exchange lease")
        p = self._placeholder
        with self._relational_connection(write=True, scope="exchange:"+conversation_id) as conn:
            owner_version = self._exchange_owner_version(conn,conversation_id)
            row = conn.execute(f"SELECT payload_json,expires_at,claim_id,lease_until,owner_version FROM pending_tool_exchanges WHERE conversation_id={p} AND exchange_id={p}", (conversation_id, exchange_id)).fetchone()
            if owner_version is None or row is None or row["expires_at"] <= now or row["owner_version"] != owner_version:
                return None
            if row["claim_id"] is not None and (row["lease_until"] or 0) > now:
                return None
            conn.execute(f"UPDATE pending_tool_exchanges SET claim_id={p},lease_until={p} WHERE conversation_id={p} AND exchange_id={p}", (claim_id, now + lease_seconds, conversation_id, exchange_id))
            return row["payload_json"]

    def renew_pending_exchange(self, conversation_id, exchange_id, claim_id, *, now, lease_seconds=120):
        self._exchange_identity(conversation_id, exchange_id)
        if not claim_id or not math.isfinite(now) or not 0 < lease_seconds <= 600:
            raise ValueError("Invalid pending exchange lease")
        p = self._placeholder
        with self._relational_connection(write=True, scope="exchange:"+conversation_id) as conn:
            owner_version = self._exchange_owner_version(conn,conversation_id)
            if owner_version is None:
                return False
            cursor = conn.execute(f"UPDATE pending_tool_exchanges SET lease_until={p} WHERE conversation_id={p} AND exchange_id={p} AND claim_id={p} AND lease_until>{p} AND expires_at>{p} AND owner_version={p}", (now+lease_seconds, conversation_id, exchange_id, claim_id, now, now,owner_version))
            return cursor.rowcount == 1

    def finish_pending_exchange(self, conversation_id, exchange_id, claim_id, *, consume):
        self._exchange_identity(conversation_id, exchange_id)
        if not claim_id or type(consume) is not bool:
            raise ValueError("Invalid pending exchange completion")
        p = self._placeholder
        action = "DELETE FROM pending_tool_exchanges" if consume else "UPDATE pending_tool_exchanges SET claim_id=NULL,lease_until=NULL"
        with self._relational_connection(write=True, scope="exchange:"+conversation_id) as conn:
            params = [conversation_id, exchange_id, claim_id]
            validity = ""
            if consume:
                now = time.time()
                owner_version = self._exchange_owner_version(conn,conversation_id)
                if owner_version is None:
                    return False
                validity = f" AND lease_until>{p} AND expires_at>{p} AND owner_version={p}"
                params.extend((now, now,owner_version))
            cursor = conn.execute(f"{action} WHERE conversation_id={p} AND exchange_id={p} AND claim_id={p}{validity}", params)
            return cursor.rowcount == 1
