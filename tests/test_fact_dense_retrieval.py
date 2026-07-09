"""Dense fact retrieval — Phase 1 (schema + stores) tests.

Covers the ``fact_embeddings`` table, the model-versioned
store/load/backfill surfaces, live-fact eligibility, malformed/wrong-dim
skipping, the empty-conversation-id write guard, and the FK cascade, on
both the SQLite and Postgres backends. Postgres cases are gated on a
configured DSN and run serially (``-n0``).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact

from tests.pg_helpers import pg_dsn, pg_test_conn

MODEL_A = "all-MiniLM-L6-v2"
MODEL_B = "bge-small-en-v1.5"

PG_URL = pg_dsn()
pg_only = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


def _fact(fact_id: str, conv: str, *, subject="user", verb="likes",
          obj="tea", what="", superseded_by=None,
          mentioned_at: datetime | None = None) -> Fact:
    return Fact(
        id=fact_id,
        subject=subject,
        verb=verb,
        object=obj,
        what=what,
        conversation_id=conv,
        mentioned_at=mentioned_at or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        superseded_by=superseded_by,
    )


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "fdr.db"))


def test_sqlite_fact_embeddings_schema_fk_index_and_foreign_keys(sqlite_store):
    conn = sqlite_store._get_conn()
    # table exists
    tbl = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_embeddings'"
    ).fetchone()
    assert tbl is not None
    # index exists
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='idx_fact_embeddings_conv_model'"
    ).fetchone()
    assert idx is not None
    # FK to facts(id) ON DELETE CASCADE
    fks = conn.execute("PRAGMA foreign_key_list(fact_embeddings)").fetchall()
    assert any(fk["table"] == "facts" and fk["on_delete"].upper() == "CASCADE"
               for fk in fks)
    # foreign_keys pragma is ON
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    # FK cascade removes vectors when the parent fact is deleted.
    conv = "conv-fk"
    f = _fact("fk-1", conv)
    sqlite_store.store_facts([f])
    sqlite_store.store_fact_embeddings("fk-1", conv, MODEL_A, [0.1, 0.2, 0.3])
    assert sqlite_store.load_fact_embeddings(conv, MODEL_A)
    conn.execute("DELETE FROM facts WHERE id = ?", ("fk-1",))
    assert conn.execute(
        "SELECT COUNT(*) FROM fact_embeddings WHERE fact_id = ?", ("fk-1",)
    ).fetchone()[0] == 0


def test_sqlite_fact_embeddings_store_load_model_filter_live_eligibility_and_bad_rows(
    sqlite_store,
):
    conv = "conv-1"
    live = _fact("live", conv, what="drinks tea daily")
    dead = _fact("dead", conv, superseded_by="live")
    sqlite_store.store_facts([live, dead])

    sqlite_store.store_fact_embeddings("live", conv, MODEL_A, [0.1, 0.2, 0.3])
    sqlite_store.store_fact_embeddings("dead", conv, MODEL_A, [0.4, 0.5, 0.6])

    loaded = sqlite_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)
    # superseded fact excluded (live-fact eligibility)
    assert set(loaded) == {"live"}
    fact, vec = loaded["live"]
    assert isinstance(fact, Fact)
    assert fact.id == "live"
    assert vec == [0.1, 0.2, 0.3]

    # model filter: a different model is a clean cache miss
    assert sqlite_store.load_fact_embeddings(conv, MODEL_B) == {}

    # malformed JSON + wrong-dim rows are skipped, never scored
    conn = sqlite_store._get_conn()
    good2 = _fact("good2", conv)
    bad_json = _fact("badj", conv)
    bad_dim = _fact("badd", conv)
    sqlite_store.store_facts([good2, bad_json, bad_dim])
    sqlite_store.store_fact_embeddings("good2", conv, MODEL_A, [0.7, 0.8, 0.9])
    conn.execute(
        "INSERT OR REPLACE INTO fact_embeddings (fact_id, conversation_id, model, embedding_json) "
        "VALUES (?, ?, ?, ?)",
        ("badj", conv, MODEL_A, "not-json"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO fact_embeddings (fact_id, conversation_id, model, embedding_json) "
        "VALUES (?, ?, ?, ?)",
        ("badd", conv, MODEL_A, "[1.0, 2.0]"),
    )
    loaded2 = sqlite_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)
    assert set(loaded2) == {"live", "good2"}


def test_sqlite_fact_embedding_backfill_iterator_uses_text_timestamp_window(sqlite_store):
    conv = "conv-win"
    early = _fact("early", conv,
                  mentioned_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    mid = _fact("mid", conv,
                mentioned_at=datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc))
    late = _fact("late", conv,
                 mentioned_at=datetime(2026, 1, 3, 12, 0, 0, tzinfo=timezone.utc))
    superseded = _fact("sup", conv, superseded_by="mid",
                       mentioned_at=datetime(2026, 1, 2, 13, 0, 0, tzinfo=timezone.utc))
    sqlite_store.store_facts([late, early, mid, superseded])

    # ordered (mentioned_at, id); superseded excluded
    all_ids = [f.id for f in sqlite_store.iter_facts_for_embedding_backfill(conv)]
    assert all_ids == ["early", "mid", "late"]

    # half-open [since, until): T-separated and space-separated bounds agree
    t_sel = [f.id for f in sqlite_store.iter_facts_for_embedding_backfill(
        conv, since="2026-01-02T00:00:00", until="2026-01-03T00:00:00")]
    space_sel = [f.id for f in sqlite_store.iter_facts_for_embedding_backfill(
        conv, since="2026-01-02 00:00:00", until="2026-01-03 00:00:00")]
    assert t_sel == space_sel == ["mid"]

    # batch_size paginates deterministically
    batched = [f.id for f in sqlite_store.iter_facts_for_embedding_backfill(
        conv, batch_size=1)]
    assert batched == ["early", "mid", "late"]


def test_store_fact_embeddings_raises_on_empty_conversation_id(sqlite_store):
    f = _fact("f-empty", "conv-x")
    sqlite_store.store_facts([f])
    with pytest.raises(ValueError):
        sqlite_store.store_fact_embeddings("f-empty", "", MODEL_A, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        sqlite_store.store_fact_embeddings("", "conv-x", MODEL_A, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        sqlite_store.store_fact_embeddings("f-empty", "conv-x", "", [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

@pytest.fixture
def pg_store():
    from virtual_context.storage.postgres import PostgresStore
    store = PostgresStore(PG_URL)
    yield store
    store.close()


@pg_only
def test_postgres_fact_embeddings_schema_fk_index_and_advisory_bootstrap(pg_store):
    conn = pg_test_conn()
    tbl = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'fact_embeddings'"
    ).fetchone()
    assert tbl is not None
    idx = conn.execute(
        "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_fact_embeddings_conv_model'"
    ).fetchone()
    assert idx is not None
    fk = conn.execute(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE table_name = 'fact_embeddings' AND constraint_type = 'FOREIGN KEY'"
    ).fetchone()
    assert fk is not None

    # FK cascade fires natively on parent fact delete.
    conv = "pg-conv-fk"
    pg_store.delete_conversation(conv)
    pg_store.store_facts([_fact("pgfk-1", conv)])
    pg_store.store_fact_embeddings("pgfk-1", conv, MODEL_A, [0.1, 0.2, 0.3])
    assert pg_store.load_fact_embeddings(conv, MODEL_A)
    conn.execute("DELETE FROM facts WHERE id = %s", ("pgfk-1",))
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM fact_embeddings WHERE fact_id = %s", ("pgfk-1",)
    ).fetchone()["c"] == 0
    pg_store.delete_conversation(conv)


@pg_only
def test_postgres_fact_embeddings_store_load_model_filter_live_eligibility_and_bad_rows(
    pg_store,
):
    conv = "pg-conv-1"
    pg_store.delete_conversation(conv)
    live = _fact("pglive", conv, what="drinks tea daily")
    dead = _fact("pgdead", conv, superseded_by="pglive")
    pg_store.store_facts([live, dead])
    pg_store.store_fact_embeddings("pglive", conv, MODEL_A, [0.1, 0.2, 0.3])
    pg_store.store_fact_embeddings("pgdead", conv, MODEL_A, [0.4, 0.5, 0.6])

    loaded = pg_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)
    assert set(loaded) == {"pglive"}
    fact, vec = loaded["pglive"]
    assert isinstance(fact, Fact)
    assert vec == [0.1, 0.2, 0.3]

    assert pg_store.load_fact_embeddings(conv, MODEL_B) == {}

    conn = pg_test_conn()
    pg_store.store_facts([_fact("pggood2", conv), _fact("pgbadj", conv),
                          _fact("pgbadd", conv)])
    pg_store.store_fact_embeddings("pggood2", conv, MODEL_A, [0.7, 0.8, 0.9])
    conn.execute(
        "INSERT INTO fact_embeddings (fact_id, conversation_id, model, embedding_json) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (fact_id, conversation_id) DO UPDATE "
        "SET model = EXCLUDED.model, embedding_json = EXCLUDED.embedding_json",
        ("pgbadj", conv, MODEL_A, "not-json"),
    )
    conn.execute(
        "INSERT INTO fact_embeddings (fact_id, conversation_id, model, embedding_json) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (fact_id, conversation_id) DO UPDATE "
        "SET model = EXCLUDED.model, embedding_json = EXCLUDED.embedding_json",
        ("pgbadd", conv, MODEL_A, "[1.0, 2.0]"),
    )
    loaded2 = pg_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)
    assert set(loaded2) == {"pglive", "pggood2"}
    pg_store.delete_conversation(conv)


@pg_only
def test_postgres_store_fact_embeddings_raises_on_empty_conversation_id(pg_store):
    conv = "pg-conv-x"
    pg_store.delete_conversation(conv)
    pg_store.store_facts([_fact("pg-empty", conv)])
    with pytest.raises(ValueError):
        pg_store.store_fact_embeddings("pg-empty", "", MODEL_A, [0.1, 0.2, 0.3])
    pg_store.delete_conversation(conv)
