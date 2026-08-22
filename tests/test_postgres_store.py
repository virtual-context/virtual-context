"""Focused unit tests for PostgresStore connection management."""

from __future__ import annotations

# Mirrored from the assertions in ``_assert_canonical_message_source_schema``.
# Stated once here so a double that drifts from the relation the store demands
# fails loudly on the shape rather than silently on a None row.
CANONICAL_MESSAGE_SOURCE_COLUMNS = (
    "tenant_id", "agent_scope_id", "platform", "account_id",
    "message_id", "canonical_turn_id",
    "assistant_canonical_turn_id", "assistant_turn_hash",
    "turn_group_number", "pair_version",
    "audience_conversation_id", "channel_id", "guild_id",
    "author_id", "source_actor_id", "transport_body_sha256",
    "canonical_body_sha256", "projection_version",
    "canonical_turn_hash", "reply_target_message_id",
)
CANONICAL_MESSAGE_SOURCE_PK = (
    "PRIMARY KEY (tenant_id, agent_scope_id, platform, account_id, message_id)"
)
CANONICAL_MESSAGE_SOURCE_FK = (
    "FOREIGN KEY (assistant_canonical_turn_id) REFERENCES "
    "canonical_turns(canonical_turn_id) ON DELETE CASCADE"
)
CANONICAL_MESSAGE_SOURCE_ASSISTANT_INDEX = (
    "CREATE UNIQUE INDEX idx_canonical_message_sources_assistant "
    "ON public.canonical_message_sources USING btree "
    "(assistant_canonical_turn_id) "
    "WHERE (assistant_canonical_turn_id IS NOT NULL)"
)
TAG_SUMMARY_REQUIRED_COLUMNS = (
    "source_canonical_turn_ids",
    "covers_through_canonical_turn_id",
    "structured_summary_json",
)


def _fact_embeddings_catalog_result(sql: str):
    """Truthy catalog rows for the constructor's required-DDL assertions.

    ``PostgresStore.__init__`` asserts ``fact_embeddings`` + its index + FK
    and the full ``speaker_handles`` relation (table, columns, unique keys)
    exist after schema bootstrap, so the fake DB doubles must model a present
    catalog for those probes or every store construction raises. Returns a
    truthy result for those queries, else ``None``.
    """
    from virtual_context.storage.postgres import (
        SPEAKER_HANDLE_COLUMNS,
        SPEAKER_HANDLE_UNIQUE_KEYS,
    )

    if "to_regclass('public.speaker_handles')" in sql:
        return _FakeRowsResult([{"reg": "speaker_handles"}])
    if "information_schema.columns" in sql and "speaker_handles" in sql:
        return _FakeRowsResult(
            [{"column_name": c} for c in SPEAKER_HANDLE_COLUMNS]
        )
    if "indisunique" in sql and "speaker_handles" in sql:
        return _FakeRowsResult(
            [{"cols": list(key)} for key in SPEAKER_HANDLE_UNIQUE_KEYS]
        )
    if (
        "fact_embeddings" in sql
        or "idx_fact_embeddings_conv_model" in sql
    ):
        return _FakeRowsResult([{"present": 1}])

    if "information_schema.columns" in sql and "tag_summaries" in sql:
        return _FakeRowsResult(
            [{"column_name": column} for column in TAG_SUMMARY_REQUIRED_COLUMNS]
        )

    if "trg_guard_attested_canonical_turn_update" in sql:
        return _FakeRowsResult([{"present": 1}])

    # ``canonical_message_sources``. The bootstrap counts unpaired legacy rows
    # and then asserts the relation's shape, so a double whose every query
    # answers "no rows" makes the count subscript None and the assertion
    # condemn a database it never looked at. Model a freshly installed,
    # correctly shaped, empty table.
    if "canonical_message_sources" in sql:
        if "count(*)" in sql.lower():
            return _FakeRowsResult([{"n": 0}])
        if "information_schema.columns" in sql:
            return _FakeRowsResult(
                [{"column_name": c} for c in CANONICAL_MESSAGE_SOURCE_COLUMNS]
            )
        if "information_schema.tables" in sql:
            return _FakeRowsResult([{"present": 1}])
        if "contype = 'p'" in sql:
            return _FakeRowsResult([{"definition": CANONICAL_MESSAGE_SOURCE_PK}])
        if "canonical_message_sources_assistant_fk" in sql:
            return _FakeRowsResult([{"definition": CANONICAL_MESSAGE_SOURCE_FK}])
        if "canonical_message_sources_pair_shape" in sql:
            return _FakeRowsResult([{"present": 1}])
        if "idx_canonical_message_sources_assistant" in sql:
            return _FakeRowsResult(
                [{"indexdef": CANONICAL_MESSAGE_SOURCE_ASSISTANT_INDEX}]
            )
        if "pg_trigger" in sql:
            return _FakeRowsResult([{"present": 1}])
    return None


class _FakeConn:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, sql: str, params=None):
        self.executed.append((sql, params))
        catalog = _fact_embeddings_catalog_result(sql)
        if catalog is not None:
            return catalog
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    class _Txn:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            return False

    def transaction(self):
        return self._Txn()

    def close(self) -> None:
        self.closed = True


class _FakeRowsResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _ConnCheckout:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn

    def __enter__(self) -> _FakeConn:
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakePool:
    instances: list["_FakePool"] = []

    def __init__(self, conninfo: str, **kwargs) -> None:
        self.conninfo = conninfo
        self.kwargs = kwargs
        self.conn = _FakeConn(f"conn-{len(self.instances)}")
        self.checkouts = 0
        self.closed = False
        self.instances.append(self)

    def connection(self) -> _ConnCheckout:
        self.checkouts += 1
        return _ConnCheckout(self.conn)

    def close(self) -> None:
        self.closed = True
        self.conn.close()


def test_postgres_store_uses_bounded_connection_pool(monkeypatch):
    from virtual_context.storage import postgres as pg

    _FakePool.instances.clear()
    monkeypatch.setattr(pg, "ConnectionPool", _FakePool)

    store = pg.PostgresStore("postgresql://example")
    pool = _FakePool.instances[0]

    assert pool.conninfo == "postgresql://example"
    assert pool.kwargs == {
        "min_size": 1,
        "max_size": 8,
        "timeout": 30.0,
        "max_idle": 300.0,
        "kwargs": {"row_factory": pg.dict_row, "autocommit": True},
    }
    assert pool.checkouts > 0

    store.close()

    assert pool.closed
    assert pool.conn.closed


def test_postgres_store_refuses_missing_structured_tag_summary_column(monkeypatch):
    import pytest

    from virtual_context.storage import postgres as pg

    class _MissingTagColumnConn(_FakeConn):
        def execute(self, sql: str, params=None):
            if "information_schema.columns" in sql and "tag_summaries" in sql:
                self.executed.append((sql, params))
                return _FakeRowsResult([
                    {"column_name": "source_canonical_turn_ids"},
                    {"column_name": "covers_through_canonical_turn_id"},
                ])
            return super().execute(sql, params)

    class _MissingTagColumnPool(_FakePool):
        def __init__(self, conninfo: str, **kwargs) -> None:
            super().__init__(conninfo, **kwargs)
            self.conn = _MissingTagColumnConn("missing-tag-column")

    monkeypatch.setattr(pg, "ConnectionPool", _MissingTagColumnPool)

    with pytest.raises(
        RuntimeError,
        match="tag_summaries schema incomplete: structured_summary_json",
    ):
        pg.PostgresStore("postgresql://example")


class _TagSummarySchemaConn(_FakeConn):
    """Mutable catalog double for the focused tag-summary bootstrap tests."""

    def __init__(
        self,
        present: set[str],
        *,
        fail_column: str = "",
    ) -> None:
        super().__init__("tag-summary-schema")
        self.present = set(present)
        self.fail_column = fail_column

    def execute(self, sql: str, params=None):
        self.executed.append((sql, params))
        if "information_schema.columns" in sql and "tag_summaries" in sql:
            return _FakeRowsResult([
                {"column_name": column}
                for column in sorted(self.present)
            ])
        prefix = "ALTER TABLE tag_summaries ADD COLUMN "
        if sql.startswith(prefix):
            column = sql[len(prefix):].split(None, 1)[0]
            if column == self.fail_column:
                raise RuntimeError(f"DDL refused for {column}")
            self.present.add(column)
        return self


class _TagSummarySchemaPool:
    def __init__(self, conn: _TagSummarySchemaConn) -> None:
        self.conn = conn

    def connection(self) -> _ConnCheckout:
        return _ConnCheckout(self.conn)


def _tag_summary_schema_store(conn: _TagSummarySchemaConn):
    from virtual_context.storage import postgres as pg

    store = object.__new__(pg.PostgresStore)
    store.pool = _TagSummarySchemaPool(conn)
    return store


def test_tag_summary_schema_present_issues_no_alter():
    conn = _TagSummarySchemaConn(set(TAG_SUMMARY_REQUIRED_COLUMNS))
    store = _tag_summary_schema_store(conn)

    store._ensure_tag_summary_schema()
    store._assert_tag_summary_schema()

    statements = [sql for sql, _params in conn.executed]
    assert not any(
        sql.startswith("ALTER TABLE tag_summaries") for sql in statements
    )
    assert "SET LOCAL lock_timeout = '2s'" not in statements


def test_tag_summary_schema_missing_column_uses_bounded_add():
    missing = "structured_summary_json"
    conn = _TagSummarySchemaConn(
        set(TAG_SUMMARY_REQUIRED_COLUMNS) - {missing},
    )
    store = _tag_summary_schema_store(conn)

    store._ensure_tag_summary_schema()
    store._assert_tag_summary_schema()

    statements = [sql for sql, _params in conn.executed]
    lock_index = statements.index("SET LOCAL lock_timeout = '2s'")
    alters = [
        (index, sql)
        for index, sql in enumerate(statements)
        if sql.startswith("ALTER TABLE tag_summaries")
    ]
    assert alters == [(
        lock_index + 1,
        "ALTER TABLE tag_summaries ADD COLUMN structured_summary_json "
        "TEXT NOT NULL DEFAULT '{\"schema_version\":0,\"claims\":[],"
        "\"source_digest\":\"\",\"generation_model\":\"\"}'",
    )]


def test_tag_summary_schema_add_failure_defers_to_required_assertion():
    import pytest

    missing = "structured_summary_json"
    conn = _TagSummarySchemaConn(
        set(TAG_SUMMARY_REQUIRED_COLUMNS) - {missing},
        fail_column=missing,
    )
    store = _tag_summary_schema_store(conn)

    # The best-effort bootstrap catches DDL failure solely so the required
    # manifest assertion below remains the startup error.
    store._ensure_tag_summary_schema()

    with pytest.raises(
        RuntimeError,
        match="tag_summaries schema incomplete: structured_summary_json",
    ):
        store._assert_tag_summary_schema()

    statements = [sql for sql, _params in conn.executed]
    assert "SET LOCAL lock_timeout = '2s'" in statements
    assert any(
        sql.startswith(
            "ALTER TABLE tag_summaries ADD COLUMN structured_summary_json",
        )
        for sql in statements
    )


def test_postgres_store_get_all_segments_uses_batch_tag_lookup(monkeypatch):
    from virtual_context.storage import postgres as pg

    class _RowsConn(_FakeConn):
        def execute(self, sql: str, params=None):
            self.executed.append((sql, params))
            catalog = _fact_embeddings_catalog_result(sql)
            if catalog is not None:
                return catalog
            if "FROM segments" in sql:
                return _FakeRowsResult([
                    {
                        "ref": "seg-1",
                        "conversation_id": "conv-1",
                        "primary_tag": "tag-a",
                        "summary": "summary",
                        "summary_tokens": 3,
                        "full_text": "full text",
                        "full_tokens": 6,
                        "messages_json": "[]",
                        "metadata_json": "{\"turn_count\": 2}",
                        "created_at": "2026-04-14T00:00:00+00:00",
                        "start_timestamp": "2026-04-14T00:00:00+00:00",
                        "end_timestamp": "2026-04-14T00:01:00+00:00",
                        "compaction_model": "test-model",
                        "compression_ratio": 0.5,
                    }
                ])
            return _FakeRowsResult([])

    conn = _RowsConn("conn-0")

    class _RowsPool(_FakePool):
        def __init__(self, conninfo: str, **kwargs) -> None:
            super().__init__(conninfo, **kwargs)
            self.conn = conn

    monkeypatch.setattr(pg, "ConnectionPool", _RowsPool)

    store = pg.PostgresStore("postgresql://example")
    monkeypatch.setattr(store, "_batch_get_tags", lambda refs: {"seg-1": ["tag-a", "tag-b"]})

    segments = store.get_all_segments(conversation_id="conv-1")

    assert len(segments) == 1
    assert segments[0].ref == "seg-1"
    assert segments[0].tags == ["tag-a", "tag-b"]


def test_normalize_request_turn_sequences_works_without_executemany(monkeypatch):
    from virtual_context.storage import postgres as pg

    class _NormalizeConn(_FakeConn):
        def execute(self, sql: str, params=None):
            self.executed.append((sql, params))
            catalog = _fact_embeddings_catalog_result(sql)
            if catalog is not None:
                return catalog
            if "SELECT id, conversation_id, request_turn, timestamp FROM request_context" in sql:
                return _FakeRowsResult([
                    {
                        "id": 10,
                        "conversation_id": "conv-1",
                        "request_turn": 494,
                        "timestamp": "2026-04-14T22:11:59.177648+00:00",
                    },
                    {
                        "id": 11,
                        "conversation_id": "conv-1",
                        "request_turn": 37,
                        "timestamp": "2026-04-15T04:42:18.486891+00:00",
                    },
                ])
            if "SELECT id, conversation_id, request_turn, timestamp FROM tool_calls" in sql:
                return _FakeRowsResult([
                    {
                        "id": 20,
                        "conversation_id": "conv-1",
                        "request_turn": 999,
                        "timestamp": "2026-04-15T04:50:00+00:00",
                    }
                ])
            return _FakeRowsResult([])

    conn = _NormalizeConn("conn-0")

    class _NormalizePool(_FakePool):
        def __init__(self, conninfo: str, **kwargs) -> None:
            super().__init__(conninfo, **kwargs)
            self.conn = conn

    monkeypatch.setattr(pg, "ConnectionPool", _NormalizePool)

    store = pg.PostgresStore("postgresql://example")
    conn.executed.clear()

    store._normalize_request_turn_sequences()

    assert (
        "UPDATE request_context SET request_turn = %s WHERE id = %s",
        (1, 10),
    ) in conn.executed
    assert (
        "UPDATE request_context SET request_turn = %s WHERE id = %s",
        (2, 11),
    ) in conn.executed
    assert (
        "UPDATE tool_calls SET request_turn = %s WHERE id = %s",
        (2, 20),
    ) in conn.executed
    assert any(
        sql.startswith("INSERT INTO request_turn_counters")
        and params == ("conv-1", 2)
        for sql, params in conn.executed
    )


def test_segment_compaction_provenance_serializes_and_hydrates(monkeypatch):
    import json
    from datetime import datetime, timezone

    from virtual_context.storage import postgres as pg
    from virtual_context.types import (
        SegmentMetadata,
        StoredSegment,
        StructuredSummary,
        SummaryClaim,
        SummarySource,
    )

    _FakePool.instances.clear()
    monkeypatch.setattr(pg, "ConnectionPool", _FakePool)
    store = pg.PostgresStore("postgresql://example")
    conn = _FakePool.instances[0].conn
    conn.executed.clear()
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    structured = StructuredSummary(
        schema_version=1,
        claims=(
            SummaryClaim(
                text="I stopped tesamorelin after side effects.",
                claim_type="personal",
                temporal_status="ceased",
                modality="asserted",
                sources=(SummarySource(
                    canonical_turn_id="ct-1",
                    source_role="requester",
                    speaker_label="BigTex",
                    evidence_excerpt="I stopped tesamorelin after side effects.",
                    session_date="2026-08-18",
                    source_provenance_digest="b" * 64,
                ),),
            ),
        ),
        source_digest="a" * 64,
        generation_model="test-summary-model",
    )
    segment = StoredSegment(
        ref="seg-provenance",
        conversation_id="conv",
        primary_tag="medical",
        tags=["medical"],
        summary="Exact source summary.",
        summary_tokens=4,
        full_text="Exact source text.",
        full_tokens=4,
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-1"],
            source_mapping_complete=True,
            source_speaker_labels=["BigTex"],
            source_speaker_identity_count=1,
            source_speaker_identity_fingerprint="speaker-proof",
            source_audience_fingerprint="audience-proof",
            structured_summary=structured,
        ),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    )

    store.store_segment(segment)

    insert_params = next(
        params for sql, params in conn.executed
        if "INSERT INTO segments" in sql
    )
    stored_metadata = json.loads(insert_params[6])
    assert stored_metadata["canonical_turn_ids"] == ["ct-1"]
    assert stored_metadata["source_mapping_complete"] is True
    assert stored_metadata["source_speaker_labels"] == ["BigTex"]
    assert stored_metadata["source_speaker_identity_count"] == 1
    assert stored_metadata["source_speaker_identity_fingerprint"] == "speaker-proof"
    assert stored_metadata["source_audience_fingerprint"] == "audience-proof"
    assert pg.strict_structured_summary(
        stored_metadata["structured_summary"]
    ) == structured

    row = {
        "ref": segment.ref,
        "conversation_id": segment.conversation_id,
        "primary_tag": segment.primary_tag,
        "summary": segment.summary,
        "summary_tokens": segment.summary_tokens,
        "full_text": segment.full_text,
        "full_tokens": segment.full_tokens,
        "messages_json": "[]",
        "metadata_json": json.dumps(stored_metadata),
        "created_at": now.isoformat(),
        "start_timestamp": now.isoformat(),
        "end_timestamp": now.isoformat(),
        "compaction_model": "test",
        "compression_ratio": 1.0,
    }
    hydrated = pg._row_to_segment(row, ["medical"])
    lightweight = pg._row_to_summary(row, ["medical"])
    for value in (hydrated, lightweight):
        assert value.metadata.canonical_turn_ids == ["ct-1"]
        assert value.metadata.source_mapping_complete is True
        assert value.metadata.source_speaker_labels == ["BigTex"]
        assert value.metadata.source_speaker_identity_count == 1
        assert value.metadata.source_speaker_identity_fingerprint == "speaker-proof"
        assert value.metadata.source_audience_fingerprint == "audience-proof"
        assert value.metadata.structured_summary == structured


def test_tag_summary_structured_payload_serializes_and_hydrates(monkeypatch):
    from datetime import datetime, timezone

    from virtual_context.storage import postgres as pg
    from virtual_context.types import (
        StructuredSummary,
        SummaryClaim,
        SummarySource,
        TagSummary,
    )

    structured = StructuredSummary(
        schema_version=1,
        claims=(SummaryClaim(
            text="I stopped tesamorelin after side effects.",
            claim_type="personal",
            temporal_status="ceased",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id="ct-1",
                source_role="requester",
                speaker_label="BigTex",
                evidence_excerpt="I stopped tesamorelin after side effects.",
                session_date="2026-08-18",
                source_provenance_digest="b" * 64,
            ),),
        ),),
        source_digest="a" * 64,
        generation_model="test-summary-model",
    )
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    tag_summary = TagSummary(
        tag="medical",
        summary="Tesamorelin history.",
        source_segment_refs=["seg-1"],
        source_turn_numbers=[1],
        source_canonical_turn_ids=["ct-1"],
        covers_through_turn=1,
        covers_through_canonical_turn_id="ct-1",
        structured_summary=structured,
        created_at=now,
        updated_at=now,
    )

    _FakePool.instances.clear()
    monkeypatch.setattr(pg, "ConnectionPool", _FakePool)
    store = pg.PostgresStore("postgresql://example")
    conn = _FakePool.instances[0].conn
    conn.executed.clear()

    store.save_tag_summary(tag_summary, "conv")
    insert_sql, insert_params = next(
        (sql, params) for sql, params in conn.executed
        if "INSERT INTO tag_summaries" in sql
    )
    assert "structured_summary_json" in insert_sql
    encoded = insert_params[9]
    assert pg._structured_summary_from_json(encoded) == structured

    row = {
        "tag": "medical",
        "summary": "Tesamorelin history.",
        "description": "",
        "code_refs": "[]",
        "summary_tokens": 4,
        "source_segment_refs": '["seg-1"]',
        "source_turn_numbers": "[1]",
        "source_canonical_turn_ids": '["ct-1"]',
        "structured_summary_json": encoded,
        "covers_through_turn": 1,
        "covers_through_canonical_turn_id": "ct-1",
        "generated_by_turn_id": "",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    original_execute = conn.execute

    def execute(sql: str, params=None):
        if "SELECT * FROM tag_summaries" in sql:
            return _FakeRowsResult([row])
        return original_execute(sql, params)

    monkeypatch.setattr(conn, "execute", execute)
    loaded = store.get_tag_summary("medical", "conv")
    assert loaded is not None
    assert loaded.structured_summary == structured
    assert loaded.source_canonical_turn_ids == ["ct-1"]
    assert loaded.covers_through_canonical_turn_id == "ct-1"


# ---------------------------------------------------------------------------
# fact_embeddings schema bootstrap (real Postgres; DSN-gated, run -n0)
# ---------------------------------------------------------------------------

import threading as _threading

import pytest as _pytest

from tests.pg_helpers import pg_dsn as _pg_dsn, pg_test_conn as _pg_test_conn

_PG_URL = _pg_dsn()


@_pytest.mark.skipif(
    not _PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)
def test_fact_embeddings_schema_bootstrap_is_idempotent_under_parallel_startup():
    """Concurrent bootstrap converges: fact_embeddings + FK + index exist,
    and no worker raises on the required post-DDL catalog assertion."""
    from virtual_context.storage.postgres import PostgresStore

    errors: list[BaseException] = []

    def _boot():
        try:
            store = PostgresStore(_PG_URL)
            store.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [_threading.Thread(target=_boot) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors

    conn = _pg_test_conn()
    assert conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
        ("fact_embeddings",),
    ).fetchone() is not None
    assert conn.execute(
        "SELECT 1 FROM pg_indexes WHERE indexname = %s",
        ("idx_fact_embeddings_conv_model",),
    ).fetchone() is not None
    assert conn.execute(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE table_name = %s AND constraint_type = %s",
        ("fact_embeddings", "FOREIGN KEY"),
    ).fetchone() is not None
