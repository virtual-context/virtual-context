"""Bounded native-query contracts, without a local PostgreSQL server."""

from contextlib import contextmanager

import pytest

from virtual_context.storage.postgres_vectors import PostgresVectorSearchMixin, VECTOR_MODEL
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import ChunkEmbedding, SpeakerRetrievalContext, StoredSegment


VECTOR = [1.0] + [0.0] * 383


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class Connection:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.statements = []
        self.rolled_back = False

    @contextmanager
    def transaction(self):
        try:
            yield
        except Exception:
            self.rolled_back = True
            raise

    def execute(self, statement, params=()):
        self.statements.append((statement, params))
        if statement.startswith("WITH scored"):
            return Rows(self.rows)
        if "count(*) AS count" in statement:
            return Rows([{"count": 2}])
        return Rows([])


class Pool:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connection(self):
        yield self.conn


class Store(PostgresVectorSearchMixin):
    def __init__(self, rows=(), ready=True):
        self.conn = Connection(rows)
        self.pool = Pool(self.conn)
        self.ready = ready

    def _vector_ready_on_connection(self, conn, model):
        return self.ready and model == VECTOR_MODEL

    @staticmethod
    def _vector_schema_installed(conn):
        return False


def query_statement(store):
    return next((statement, params) for statement, params in store.conn.statements
                if statement.startswith("WITH scored"))


def test_segment_page_is_exact_scoped_bounded_and_continuable():
    store = Store([{"segment_ref": "s", "chunk_index": 3, "text": "text", "distance": 0.1, "similarity": 0.9}])
    rows = store.search_segment_chunks_by_embedding(VECTOR, conversation_id="owner", limit=7)
    assert rows[0]["cursor"] == (0.1, "s", 3)
    statement, params = query_statement(store)
    assert "AS MATERIALIZED" in statement and "OPERATOR(public.<=>)" in statement
    assert "s.conversation_id=%s" in statement and params[-1] == 7
    assert "embedding_json" not in statement
    assert "LIMIT %s" in statement and "OFFSET" not in statement
    store.conn.statements.clear()
    store.search_segment_chunks_by_embedding(VECTOR, conversation_id="owner", after=rows[0]["cursor"])
    statement, params = query_statement(store)
    assert "(distance, segment_ref, chunk_index) > (%s, %s, %s)" in statement
    assert params[-4:-1] == [0.1, "s", 3]


def test_legacy_page_excludes_subject_and_projects_exact_physical_row():
    store = Store()
    store.search_canonical_turn_chunks_by_embedding(VECTOR, conversation_id="owner")
    statement, params = query_statement(store)
    assert "chunk.side <> 'subject'" in statement
    assert "row_to_json(physical) AS physical_row" in statement
    scoring, hydration = statement.split(") SELECT selected.*", 1)
    assert "row_to_json" not in scoring and "chunk.text" not in scoring
    assert "LIMIT %s" in scoring and "payload.text" in hydration
    assert "cto.canonical_turn_id=chunk.canonical_turn_id" in statement
    assert "embedding_json" not in statement
    assert "owner" in params


def test_speaker_page_preserves_audience_and_channel_scope():
    store = Store()
    context = SpeakerRetrievalContext(
        tenant_id="t", owner_conversation_id="owner", audience_conversation_id="audience",
        audience_channel_id="channel", request_origin_channel_id="channel",
    )
    store.search_speaker_turn_chunks_by_embedding(VECTOR, conversation_id="owner", speaker_context=context)
    statement, params = query_statement(store)
    assert "JOIN public.canonical_turns ct" in statement
    assert "ct.audience_conversation_id = %s" in statement
    assert "ct.audience_attribution_version = %s" in statement
    assert "COALESCE(ct.origin_channel_id, '') = %s" in statement
    assert "chunk.side <> 'subject'" not in statement
    assert "audience" in params and "channel" in params
    assert "embedding_json" not in statement
    store.conn.statements.clear()
    assert store.search_speaker_turn_chunks_by_embedding(VECTOR, conversation_id="foreign", speaker_context=context) == []
    assert store.conn.statements == []


def test_missing_readiness_is_visible_and_rolls_back_without_legacy_scan():
    store = Store(ready=False)
    with pytest.raises(RuntimeError, match="migration"):
        store.search_segment_chunks_by_embedding(VECTOR, conversation_id="owner")
    assert store.conn.rolled_back
    assert not any(statement.startswith("WITH scored") for statement, _ in store.conn.statements)


def test_model_and_input_gates():
    store = Store()
    assert not store.vector_search_ready("different-384-model")
    assert store.conn.statements == []
    assert store.search_segment_chunks_by_embedding([0.0] * 384) == []
    with pytest.raises(ValueError, match="dimensions"):
        store.search_segment_chunks_by_embedding([1.0])
    with pytest.raises(ValueError, match="finite"):
        store.search_segment_chunks_by_embedding([float("nan")] + VECTOR[1:])


def test_model_attestation_is_explicit_and_transaction_local():
    store = Store()
    store.semantic_embedding_model = "unrelated mutable attribute"
    store._set_semantic_vector_write_model(store.conn, VECTOR_MODEL)
    store._set_semantic_vector_write_model(store.conn, "different-model")
    assert [params[0] for _, params in store.conn.statements] == [VECTOR_MODEL, "different-model"]
    assert all("TRUE" in statement for statement, _ in store.conn.statements)


def test_migration_dry_run_does_not_create_extension_or_modify_rows():
    store = Store(ready=False)
    report = store.migrate_semantic_vectors()
    assert report["dry_run"] and not report["available"] and not report["ready"]
    assert report["tables"]["segment_chunks"]["rows"] == 2
    assert all(statement.lstrip().startswith("SELECT") for statement, _ in store.conn.statements)
    with pytest.raises(RuntimeError, match="unavailable"):
        store.migrate_semantic_vectors(dry_run=False)


def test_sqlite_legacy_loader_scopes_before_parsing_embeddings(tmp_path):
    store = SQLiteStore(tmp_path / "scope.db")
    try:
        for owner in ("mine", "foreign"):
            store.store_segment(StoredSegment(ref=owner, conversation_id=owner))
            store.store_chunk_embeddings(owner, [ChunkEmbedding(segment_ref=owner, chunk_index=0, text=owner, embedding=[1.0])])
        store._get_conn().execute("UPDATE segment_chunks SET embedding_json='not-json' WHERE segment_ref='foreign'")
        assert [chunk.segment_ref for chunk in store.get_all_chunk_embeddings(conversation_id="mine")] == ["mine"]
    finally:
        store.close()
