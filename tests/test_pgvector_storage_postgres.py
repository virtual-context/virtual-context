"""Remote-fleet SQL parity gates; never start a local PostgreSQL server.

These cases require a fleet role that may create an isolated test database.
They are intentionally skipped without that capability or a pgvector package.
"""

import math
import os
import uuid

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn
from virtual_context.storage.postgres_vectors import VECTOR_MODEL
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION, CanonicalTurnChunkEmbedding, ChunkEmbedding,
    SpeakerRetrievalContext, StoredSegment,
)


_REQUIRE_FLEET = os.environ.get("VC_REQUIRE_PGVECTOR_TESTS") == "1"
pytestmark = pytest.mark.skipif(not pg_dsn() and not _REQUIRE_FLEET, reason="PostgreSQL fleet DSN not configured")


def vector(x=1.0):
    return [x, math.sqrt(1.0 - x * x)] + [0.0] * 382


@pytest.fixture
def store():
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    from virtual_context.storage.postgres import PostgresStore

    if not pg_dsn():
        pytest.fail("Required pgvector fleet run has no configured DSN")
    admin = pg_test_conn()
    if not admin.execute("SELECT 1 FROM pg_available_extensions WHERE name='vector'").fetchone():
        if _REQUIRE_FLEET:
            pytest.fail("Required pgvector fleet run has no installed extension package")
        pytest.skip("pgvector package not installed on fleet server")
    name = f"vc_vector_test_{uuid.uuid4().hex}"
    try:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0 ENCODING 'UTF8'").format(sql.Identifier(name)))
    except psycopg.errors.InsufficientPrivilege:
        if _REQUIRE_FLEET:
            pytest.fail("Required pgvector fleet role cannot create isolated test databases")
        pytest.skip("fleet role cannot create isolated vector-test database")
    result = None
    try:
        params = conninfo_to_dict(pg_dsn())
        params["dbname"] = name
        result = PostgresStore(make_conninfo(**params))
        report = result.migrate_semantic_vectors(dry_run=False, batch_size=2)
        assert report["ready"]
        yield result
    finally:
        if result:
            result.close()
        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def pages(fn, query, **kwargs):
    output = []
    cursor = None
    for _ in range(20):
        page = fn(query, limit=1, after=cursor, **kwargs)
        if not page:
            return output
        output.extend(page)
        cursor = page[-1]["cursor"]
    raise AssertionError("keyset pagination did not exhaust")


def test_exact_segment_distance_keysets_scope_and_zero_vectors(store):
    from virtual_context.core.math_utils import cosine_similarity

    for owner, ref, values in [
        ("mine", "a", vector(0.9)), ("mine", "b", vector(0.9)),
        ("mine", "c", vector(0.5)), ("mine", "zero", [0.0] * 384),
        ("foreign", "foreign", vector()),
    ]:
        store.store_segment(StoredSegment(ref=ref, conversation_id=owner))
        store.store_chunk_embeddings(ref, [ChunkEmbedding(segment_ref=ref, chunk_index=0, text=ref, embedding=values)], embedding_model=VECTOR_MODEL)
    assert store.vector_search_ready(VECTOR_MODEL)
    expected = sorted(
        [(cosine_similarity(vector(), chunk.embedding), chunk.segment_ref) for chunk in store.get_all_chunk_embeddings(conversation_id="mine")
         if cosine_similarity(vector(), chunk.embedding) >= 0.25],
        key=lambda value: (-value[0], value[1]),
    )
    result = pages(store.search_segment_chunks_by_embedding, vector(), conversation_id="mine")
    assert [row["segment_ref"] for row in result] == [ref for _, ref in expected]
    assert [row["similarity"] for row in result] == pytest.approx([score for score, _ in expected], abs=1e-5)
    assert all("embedding_json" not in row and "embedding" not in row for row in result)


def test_canonical_and_speaker_pages_preserve_physical_scope(store):
    ids = {}
    for ordinal, (name, audience, channel) in enumerate([
        ("good", "guild", "channel"), ("private", "dm", ""),
        ("sibling", "guild", "other-channel"),
    ]):
        key = str(uuid.uuid4())
        ids[name] = key
        store.save_canonical_turn(
            "owner", ordinal, f"User source {name}", f"Assistant source {name}",
            canonical_turn_id=key, sort_key=float(ordinal), audience_conversation_id=audience,
            audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION, origin_channel_id=channel,
        )
        for side in ("user", "subject"):
            chunk = CanonicalTurnChunkEmbedding(conversation_id="owner", canonical_turn_id=key,
                                               turn_number=ordinal, side=side, chunk_index=0,
                                               text=f"{name}-{side}", embedding=vector())
            store.store_canonical_turn_chunk_embeddings("owner", ordinal, side, [chunk], canonical_turn_id=key, embedding_model=VECTOR_MODEL)
    # Orphan chunks cannot enter either physical join even with the top score.
    orphan = str(uuid.uuid4())
    store.store_canonical_turn_chunk_embeddings("owner", 99, "user", [
        CanonicalTurnChunkEmbedding(conversation_id="owner", canonical_turn_id=orphan,
                                   turn_number=99, side="user", chunk_index=0, text="orphan", embedding=vector()),
    ], canonical_turn_id=orphan, embedding_model=VECTOR_MODEL)
    legacy = pages(store.search_canonical_turn_chunks_by_embedding, vector(), conversation_id="owner")
    assert len(legacy) == 3 and {row["side"] for row in legacy} == {"user"}
    assert all(row["physical_row"].canonical_turn_id == row["canonical_turn_id"] for row in legacy)
    context = SpeakerRetrievalContext(tenant_id="tenant", owner_conversation_id="owner",
                                      audience_conversation_id="guild", audience_channel_id="channel",
                                      request_origin_channel_id="channel")
    selected = pages(store.search_speaker_turn_chunks_by_embedding, vector(), conversation_id="owner", speaker_context=context)
    assert {row["canonical_turn_id"] for row in selected} == {ids["good"]}
    assert {row["side"] for row in selected} == {"user", "subject"}
    assert store.search_speaker_turn_chunks_by_embedding(vector(), conversation_id="foreign", speaker_context=context) == []


def test_dual_write_legacy_residue_and_incompatible_model_gate(store):
    store.store_segment(StoredSegment(ref="s", conversation_id="mine"))
    chunk = ChunkEmbedding(segment_ref="s", chunk_index=0, text="text", embedding=vector())
    store.store_chunk_embeddings("s", [chunk], embedding_model=VECTOR_MODEL)
    assert store.vector_search_ready(VECTOR_MODEL)
    # A legacy writer has no model attestation and must invalidate the cache.
    with store.pool.connection() as conn:
        conn.execute("UPDATE segment_chunks SET embedding_json=embedding_json WHERE segment_ref='s'")
    assert not store.vector_search_ready(VECTOR_MODEL)
    assert store.migrate_semantic_vectors(dry_run=True)["tables"]["segment_chunks"]["residue"] == 1
    assert store.migrate_semantic_vectors(dry_run=False, batch_size=1)["ready"]
    store.store_chunk_embeddings("s", [chunk], embedding_model="another-384-dimensional-model")
    assert not store.vector_search_ready(VECTOR_MODEL)
    # Backfill may attest unknown legacy rows, but cannot relabel explicitly
    # incompatible model evidence as MiniLM just to satisfy activation.
    assert not store.migrate_semantic_vectors(dry_run=False)["ready"]
    with pytest.raises(RuntimeError, match="migrate-semantic-vectors"):
        store.search_segment_chunks_by_embedding(vector(), conversation_id="mine")


def test_invalid_legacy_json_dimensions_nonfinite_and_zero_migration_residue(store):
    import json

    store.store_segment(StoredSegment(ref="invalid", conversation_id="mine"))
    invalid = ["not-json", "{}", "[1.0, 0.0]", json.dumps([float("nan")] * 384),
               json.dumps([float("inf")] * 384), json.dumps(["1"] * 384)]
    with store.pool.connection() as conn:
        for index, encoded in enumerate([*invalid, json.dumps([0.0] * 384)]):
            conn.execute(
                "INSERT INTO segment_chunks(segment_ref,chunk_index,text,embedding_json) VALUES (%s,%s,%s,%s)",
                ("invalid", index, f"invalid-{index}", encoded),
            )
    report = store.migrate_semantic_vectors(dry_run=False, batch_size=2)
    assert not report["ready"]
    assert report["tables"]["segment_chunks"]["residue"] == len(invalid)
    with store.pool.connection() as conn:
        rows = conn.execute(
            "SELECT chunk_index, embedding IS NULL AS missing, embedding_zero, embedding_json FROM segment_chunks ORDER BY chunk_index"
        ).fetchall()
    assert [row["embedding_json"] for row in rows[:-1]] == invalid
    assert all(row["missing"] and not row["embedding_zero"] for row in rows[:-1])
    assert rows[-1]["missing"] and rows[-1]["embedding_zero"]


def test_dry_run_reports_residue_per_attested_model_without_changing_rows(store):
    for name, model in [('unknown', ''), ('foreign-model', 'other-model'), ('supported', VECTOR_MODEL)]:
        store.store_segment(StoredSegment(ref=name, conversation_id='owner'))
        store.store_chunk_embeddings(name, [ChunkEmbedding(segment_ref=name, chunk_index=0, text=name, embedding=vector())], embedding_model=model)
    with store.pool.connection() as conn:
        before = conn.execute('SELECT segment_ref,embedding_json,embedding_model,embedding_source_hash FROM segment_chunks ORDER BY segment_ref').fetchall()
    report = store.migrate_semantic_vectors(dry_run=True)
    assert not report['ready']
    assert report['tables']['segment_chunks']['residue_by_model'] == [
        {'model': '', 'rows': 1}, {'model': 'other-model', 'rows': 1},
    ]
    assert report['tables']['segment_chunks']['residue'] == 2
    assert report['tables']['canonical_turn_chunks']['residue_by_model'] == []
    with store.pool.connection() as conn:
        assert conn.execute('SELECT segment_ref,embedding_json,embedding_model,embedding_source_hash FROM segment_chunks ORDER BY segment_ref').fetchall() == before
