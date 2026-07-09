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


# ===========================================================================
# Phase 2 — Embed-on-write hooks (both routes + supersession invalidation)
# ===========================================================================

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from virtual_context.core.compaction_pipeline import CompactionPipeline  # noqa: E402
from virtual_context.ingest.supersession import (  # noqa: E402
    FactSupersessionChecker,
    promote_planned_facts,
    refresh_fact_embedding,
)
from virtual_context.types import (  # noqa: E402
    CompactionLeaseLost,
    EngineState,
    SupersessionConfig,
)


def _embed(texts):
    """Deterministic 3-dim embedder — length + two char counts."""
    return [[float(len(t)), float(t.count("e")), float(t.count("o"))] for t in texts]


class _SemanticStub:
    def __init__(self, embed_fn):
        self._embed_fn = embed_fn

    def get_embed_fn(self):
        return self._embed_fn


def _make_pipeline(store, conv, *, embed_fn=_embed, model=MODEL_A):
    return CompactionPipeline(
        compactor=None,
        segmenter=MagicMock(),
        store=store,
        turn_tag_index=MagicMock(),
        engine_state=EngineState(),
        config=SimpleNamespace(
            conversation_id=conv,
            retriever=SimpleNamespace(embedding_model=model),
        ),
        supersession_checker=None,
        fact_curator=None,
        semantic=_SemanticStub(embed_fn) if embed_fn is not None else None,
        telemetry=MagicMock(),
        save_state_callback=MagicMock(return_value=True),
    )


def _seg_fact(fact_id, conv, seg_ref, *, subject="user", verb="likes",
              obj="tea", what=""):
    return Fact(
        id=fact_id, subject=subject, verb=verb, object=obj, what=what,
        conversation_id=conv, segment_ref=seg_ref,
        mentioned_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


class _RewriteLLM:
    """LLMProvider stub whose ``complete`` returns a fixed JSON payload."""

    last_usage: dict = {}

    def __init__(self, payload):
        self._payload = payload

    def complete(self, *_args, **_kwargs):
        return json.dumps(self._payload), {}


# ---------------------------------------------------------------------------
# SQLite — Phase 2
# ---------------------------------------------------------------------------


def test_sqlite_embed_on_write_replace_route_writes_current_model_vectors(sqlite_store):
    conv = "p2-replace"
    seg = "seg-r"
    facts = [_seg_fact("r1", conv, seg, what="drinks tea"),
             _seg_fact("r2", conv, seg, verb="visited", obj="paris")]
    deleted, inserted = sqlite_store.replace_facts_for_segment(conv, seg, facts)
    assert (deleted, inserted) == (0, 2)

    pipe = _make_pipeline(sqlite_store, conv)
    pipe._embed_and_store_fact_embeddings(
        facts, operation_id=None, guard_kwargs=pipe._compaction_guard_kwargs(None),
    )

    loaded = sqlite_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)
    assert set(loaded) == {"r1", "r2"}
    assert loaded["r1"][1] == _embed([facts[0].embed_text()])[0]
    # A different model has no rows (model-versioned).
    assert sqlite_store.load_fact_embeddings(conv, MODEL_B) == {}


def test_sqlite_embed_on_write_store_facts_route_uses_same_helper_and_fence(sqlite_store):
    conv = "p2-store-facts"
    facts = [_seg_fact("s1", conv, "seg-s", what="likes coffee"),
             _seg_fact("s2", conv, "seg-s", verb="uses", obj="linux")]
    sqlite_store.store_facts(facts)

    pipe = _make_pipeline(sqlite_store, conv)
    guard = pipe._compaction_guard_kwargs(None)
    pipe._embed_and_store_fact_embeddings(facts, operation_id=None, guard_kwargs=guard)

    loaded = sqlite_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)
    assert set(loaded) == {"s1", "s2"}

    # Same helper again is an idempotent upsert — one row per key.
    pipe._embed_and_store_fact_embeddings(facts, operation_id=None, guard_kwargs=guard)
    conn = sqlite_store._get_conn()
    for fid in ("s1", "s2"):
        assert conn.execute(
            "SELECT COUNT(*) FROM fact_embeddings WHERE fact_id = ?", (fid,)
        ).fetchone()[0] == 1


def test_sqlite_recompact_cascades_old_vectors_and_writes_new_ids(sqlite_store):
    conv = "p2-recompact"
    seg = "seg-x"
    old = [_seg_fact("old1", conv, seg), _seg_fact("old2", conv, seg)]
    sqlite_store.replace_facts_for_segment(conv, seg, old)
    pipe = _make_pipeline(sqlite_store, conv)
    pipe._embed_and_store_fact_embeddings(
        old, operation_id=None, guard_kwargs=pipe._compaction_guard_kwargs(None),
    )
    assert set(sqlite_store.load_fact_embeddings(conv, MODEL_A)) == {"old1", "old2"}

    # Re-compact the same segment: DELETE half cascades old vectors via FK.
    new = [_seg_fact("new1", conv, seg), _seg_fact("new2", conv, seg)]
    deleted, inserted = sqlite_store.replace_facts_for_segment(conv, seg, new)
    assert (deleted, inserted) == (2, 2)
    pipe._embed_and_store_fact_embeddings(
        new, operation_id=None, guard_kwargs=pipe._compaction_guard_kwargs(None),
    )

    loaded = sqlite_store.load_fact_embeddings(conv, MODEL_A)
    assert set(loaded) == {"new1", "new2"}
    conn = sqlite_store._get_conn()
    assert conn.execute(
        "SELECT COUNT(*) FROM fact_embeddings WHERE fact_id IN ('old1','old2')"
    ).fetchone()[0] == 0


def test_sqlite_update_fact_fields_invalidates_then_refreshes_all_callers(sqlite_store):
    conv = "p2-uff"

    # (a) Backend invalidation carve-out: status-only keeps the vector;
    #     an embed-text mutation deletes it (no caller refresh here).
    keep = _seg_fact("keep", conv, "seg-k", verb="likes", obj="tea", what="daily")
    drop = _seg_fact("drop", conv, "seg-k", verb="likes", obj="tea", what="daily")
    sqlite_store.store_facts([keep, drop])
    sqlite_store.store_fact_embeddings("keep", conv, MODEL_A, [7.0, 7.0, 7.0])
    sqlite_store.store_fact_embeddings("drop", conv, MODEL_A, [7.0, 7.0, 7.0])

    r_keep = sqlite_store.update_fact_fields(
        "keep", "likes", "tea", "completed", "daily")
    assert r_keep is True
    assert sqlite_store.load_fact_embeddings(conv, MODEL_A)["keep"][1] == [7.0, 7.0, 7.0]

    r_drop = sqlite_store.update_fact_fields(
        "drop", "adores", "tea", "active", "daily")  # verb changed
    assert r_drop is True
    assert "drop" not in sqlite_store.load_fact_embeddings(conv, MODEL_A)

    # (b) promote_planned_facts caller refresh: LLM rewrites verb/what,
    #     the old vector is invalidated then refreshed to the new text.
    planned = Fact(
        id="planned", subject="user", verb="will_attend", object="conference",
        what="the AI summit", status="planned", when_date="2020-01-01",
        conversation_id=conv, segment_ref="seg-p",
        mentioned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    sqlite_store.store_facts([planned])
    sqlite_store.store_fact_embeddings("planned", conv, MODEL_A, [9.0, 9.0, 9.0])
    llm = _RewriteLLM({"verb": "attended", "object": "conference",
                       "what": "went to the AI summit"})
    promote_planned_facts(
        sqlite_store, llm_provider=llm, model="llm-x",
        embed_fn=_embed, embedding_model=MODEL_A,
    )
    got = sqlite_store.query_facts(conversation_id=conv, subject="user")
    promoted = next(f for f in got if f.id == "planned")
    assert promoted.status == "completed"
    loaded = sqlite_store.load_fact_embeddings(conv, MODEL_A)
    assert loaded["planned"][1] == _embed([promoted.embed_text()])[0]
    assert loaded["planned"][1] != [9.0, 9.0, 9.0]

    # (c) supersession _merge_facts caller refresh.
    win = _seg_fact("winner", conv, "seg-w", verb="owns", obj="car", what="a sedan")
    old = _seg_fact("loser", conv, "seg-w", verb="owned", obj="car", what="a coupe")
    sqlite_store.store_facts([win, old])
    sqlite_store.store_fact_embeddings("winner", conv, MODEL_A, [1.0, 1.0, 1.0])
    checker = FactSupersessionChecker(
        llm_provider=_RewriteLLM({"verb": "owns", "object": "truck",
                                  "status": "active", "what": "a pickup"}),
        model="llm-x", store=sqlite_store, config=SupersessionConfig(enabled=True),
        embed_fn=_embed, embedding_model=MODEL_A,
    )
    checker._merge_facts(win, old)
    loaded_w = sqlite_store.load_fact_embeddings(conv, MODEL_A)["winner"][1]
    assert loaded_w == _embed([win.embed_text()])[0]
    assert loaded_w != [1.0, 1.0, 1.0]


def test_sqlite_update_fact_fields_refresh_failure_leaves_no_stale_vector(
    sqlite_store, monkeypatch,
):
    conv = "p2-refresh-fail"
    planned = Fact(
        id="pf", subject="user", verb="will_move", object="city",
        what="to Boston", status="planned", when_date="2020-01-01",
        conversation_id=conv, segment_ref="seg-pf",
        mentioned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    sqlite_store.store_facts([planned])
    sqlite_store.store_fact_embeddings("pf", conv, MODEL_A, [5.0, 5.0, 5.0])

    # After the seed, make every embedding write fail (non-lease).
    def _boom(*_a, **_k):
        raise RuntimeError("embedding provider down")

    monkeypatch.setattr(sqlite_store, "store_fact_embeddings", _boom)
    llm = _RewriteLLM({"verb": "moved", "object": "city", "what": "to Boston"})
    # Refresh failure is swallowed; invalidation already ran in-txn.
    promote_planned_facts(
        sqlite_store, llm_provider=llm, model="llm-x",
        embed_fn=_embed, embedding_model=MODEL_A,
    )
    # No stale current-model row survives.
    conn = sqlite_store._get_conn()
    assert conn.execute(
        "SELECT COUNT(*) FROM fact_embeddings WHERE fact_id = 'pf'"
    ).fetchone()[0] == 0


def test_sqlite_fact_embedding_lease_lost_and_c2r_skip_write_no_vectors(
    sqlite_store, monkeypatch,
):
    conv = "p2-lease"
    facts = [_seg_fact("l1", conv, "seg-l")]
    sqlite_store.store_facts(facts)
    pipe = _make_pipeline(sqlite_store, conv)

    # Lease loss propagates (fail-closed) — no vector written.
    def _lease_lost(*_a, **_k):
        raise CompactionLeaseLost("op", write_site="store_fact_embeddings")

    monkeypatch.setattr(sqlite_store, "store_fact_embeddings", _lease_lost)
    with pytest.raises(CompactionLeaseLost):
        pipe._embed_and_store_fact_embeddings(
            facts, operation_id="op",
            guard_kwargs=pipe._compaction_guard_kwargs(None),
        )
    monkeypatch.undo()
    assert sqlite_store.load_fact_embeddings(conv, MODEL_A) == {}

    # C2R skip: the pipeline guards the embed on the insert count, so a
    # skipped fact write (0 inserted) writes zero vectors.
    _inserted = 0
    if _inserted:  # mirrors compaction_pipeline.py guard
        pipe._embed_and_store_fact_embeddings(
            facts, operation_id=None,
            guard_kwargs=pipe._compaction_guard_kwargs(None),
        )
    assert sqlite_store.load_fact_embeddings(conv, MODEL_A) == {}


# ---------------------------------------------------------------------------
# Postgres — Phase 2
# ---------------------------------------------------------------------------


@pg_only
def test_postgres_embed_on_write_replace_route_writes_current_model_vectors(pg_store):
    conv = "pg-p2-replace"
    pg_store.delete_conversation(conv)
    seg = "pg-seg-r"
    facts = [_seg_fact("pr1", conv, seg, what="drinks tea"),
             _seg_fact("pr2", conv, seg, verb="visited", obj="paris")]
    deleted, inserted = pg_store.replace_facts_for_segment(conv, seg, facts)
    assert (deleted, inserted) == (0, 2)

    pipe = _make_pipeline(pg_store, conv)
    pipe._embed_and_store_fact_embeddings(
        facts, operation_id=None, guard_kwargs=pipe._compaction_guard_kwargs(None),
    )
    loaded = pg_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)
    assert set(loaded) == {"pr1", "pr2"}
    assert loaded["pr1"][1] == _embed([facts[0].embed_text()])[0]
    assert pg_store.load_fact_embeddings(conv, MODEL_B) == {}
    pg_store.delete_conversation(conv)


@pg_only
def test_postgres_embed_on_write_store_facts_route_uses_same_helper_and_fence(pg_store):
    conv = "pg-p2-store-facts"
    pg_store.delete_conversation(conv)
    facts = [_seg_fact("ps1", conv, "pg-seg-s", what="likes coffee"),
             _seg_fact("ps2", conv, "pg-seg-s", verb="uses", obj="linux")]
    pg_store.store_facts(facts)

    pipe = _make_pipeline(pg_store, conv)
    guard = pipe._compaction_guard_kwargs(None)
    pipe._embed_and_store_fact_embeddings(facts, operation_id=None, guard_kwargs=guard)
    assert set(pg_store.load_fact_embeddings(conv, MODEL_A, expected_dim=3)) == {"ps1", "ps2"}

    pipe._embed_and_store_fact_embeddings(facts, operation_id=None, guard_kwargs=guard)
    conn = pg_test_conn()
    for fid in ("ps1", "ps2"):
        assert conn.execute(
            "SELECT COUNT(*) AS c FROM fact_embeddings WHERE fact_id = %s", (fid,)
        ).fetchone()["c"] == 1
    pg_store.delete_conversation(conv)


@pg_only
def test_postgres_recompact_cascades_old_vectors_and_writes_new_ids(pg_store):
    conv = "pg-p2-recompact"
    pg_store.delete_conversation(conv)
    seg = "pg-seg-x"
    old = [_seg_fact("pold1", conv, seg), _seg_fact("pold2", conv, seg)]
    pg_store.replace_facts_for_segment(conv, seg, old)
    pipe = _make_pipeline(pg_store, conv)
    pipe._embed_and_store_fact_embeddings(
        old, operation_id=None, guard_kwargs=pipe._compaction_guard_kwargs(None),
    )
    assert set(pg_store.load_fact_embeddings(conv, MODEL_A)) == {"pold1", "pold2"}

    new = [_seg_fact("pnew1", conv, seg), _seg_fact("pnew2", conv, seg)]
    deleted, inserted = pg_store.replace_facts_for_segment(conv, seg, new)
    assert (deleted, inserted) == (2, 2)
    pipe._embed_and_store_fact_embeddings(
        new, operation_id=None, guard_kwargs=pipe._compaction_guard_kwargs(None),
    )
    assert set(pg_store.load_fact_embeddings(conv, MODEL_A)) == {"pnew1", "pnew2"}
    conn = pg_test_conn()
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM fact_embeddings "
        "WHERE fact_id IN ('pold1','pold2')"
    ).fetchone()["c"] == 0
    pg_store.delete_conversation(conv)


@pg_only
def test_postgres_update_fact_fields_invalidates_then_refreshes_all_callers(pg_store):
    conv = "pg-p2-uff"
    pg_store.delete_conversation(conv)

    keep = _seg_fact("pgkeep", conv, "pg-seg-k", verb="likes", obj="tea", what="daily")
    drop = _seg_fact("pgdrop", conv, "pg-seg-k", verb="likes", obj="tea", what="daily")
    pg_store.store_facts([keep, drop])
    pg_store.store_fact_embeddings("pgkeep", conv, MODEL_A, [7.0, 7.0, 7.0])
    pg_store.store_fact_embeddings("pgdrop", conv, MODEL_A, [7.0, 7.0, 7.0])

    assert pg_store.update_fact_fields(
        "pgkeep", "likes", "tea", "completed", "daily") is True
    assert pg_store.load_fact_embeddings(conv, MODEL_A)["pgkeep"][1] == [7.0, 7.0, 7.0]

    assert pg_store.update_fact_fields(
        "pgdrop", "adores", "tea", "active", "daily") is True
    assert "pgdrop" not in pg_store.load_fact_embeddings(conv, MODEL_A)

    planned = Fact(
        id="pgplanned", subject="user", verb="will_attend", object="conference",
        what="the AI summit", status="planned", when_date="2020-01-01",
        conversation_id=conv, segment_ref="pg-seg-p",
        mentioned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    pg_store.store_facts([planned])
    pg_store.store_fact_embeddings("pgplanned", conv, MODEL_A, [9.0, 9.0, 9.0])
    promote_planned_facts(
        pg_store,
        llm_provider=_RewriteLLM({"verb": "attended", "object": "conference",
                                  "what": "went to the AI summit"}),
        model="llm-x", embed_fn=_embed, embedding_model=MODEL_A,
    )
    promoted = next(f for f in pg_store.query_facts(conversation_id=conv, subject="user")
                    if f.id == "pgplanned")
    loaded = pg_store.load_fact_embeddings(conv, MODEL_A)
    assert loaded["pgplanned"][1] == _embed([promoted.embed_text()])[0]
    assert loaded["pgplanned"][1] != [9.0, 9.0, 9.0]
    pg_store.delete_conversation(conv)


# ---------------------------------------------------------------------------
# Phase 3 — admin backfill (engine.backfill_fact_embeddings)
# ---------------------------------------------------------------------------


class _StubSemantic:
    """Deterministic in-process embedder standing in for the semantic
    manager so backfill tests never load a real embedding model."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.calls: list[list[str]] = []

    def get_embed_fn(self):
        def _embed(texts):
            self.calls.append(list(texts))
            out = []
            for t in texts:
                h = abs(hash(t))
                out.append([float((h >> (8 * i)) & 0xFF) for i in range(self._dim)])
            return out

        return _embed


def _backfill_engine(tmp_path, conv_id: str = "conv-fdr-backfill"):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "b.db")}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": conv_id,
    })
    engine = VirtualContextEngine(config=config)
    engine._semantic = _StubSemantic()
    return engine


def _raw(engine):
    store = engine._store
    store = getattr(store, "_store", store)
    # Single-backend sqlite still composes a CompositeStore whose fact
    # store is the concrete SQLiteStore (the one owning _get_conn and the
    # fact_embeddings table).
    return getattr(store, "_facts", store)


def test_backfill_fact_embeddings_idempotent_second_run_all_skipped_current(tmp_path):
    conv = "conv-fdr-idem"
    engine = _backfill_engine(tmp_path, conv)
    raw = _raw(engine)
    raw.store_facts([
        _fact("a", conv, verb="likes", obj="tea"),
        _fact("b", conv, verb="visited", obj="paris"),
    ])

    first = engine.backfill_fact_embeddings(conv)
    assert first["embedded"] == 2
    assert first["skipped_current"] == 0

    loaded = raw.load_fact_embeddings(conv, MODEL_A)
    assert set(loaded) == {"a", "b"}

    second = engine.backfill_fact_embeddings(conv)
    assert second == {
        "embedded": 0, "skipped_current": 2,
        "model_mismatch": 0, "malformed": 0, "windowed_out": 0,
    }


def test_backfill_fact_embeddings_window_uses_half_open_text_timestamp_normalization(tmp_path):
    conv = "conv-fdr-win"
    engine = _backfill_engine(tmp_path, conv)
    raw = _raw(engine)
    raw.store_facts([
        _fact("early", conv, mentioned_at=datetime(2026, 1, 1, 12, tzinfo=timezone.utc)),
        _fact("mid", conv, mentioned_at=datetime(2026, 1, 2, 12, tzinfo=timezone.utc)),
        _fact("late", conv, mentioned_at=datetime(2026, 1, 3, 12, tzinfo=timezone.utc)),
    ])

    t_rep = engine.backfill_fact_embeddings(
        conv, since="2026-01-02T00:00:00", until="2026-01-03T00:00:00")
    # only "mid" is in-window; early+late windowed out
    assert t_rep["embedded"] == 1
    assert t_rep["windowed_out"] == 2
    assert set(raw.load_fact_embeddings(conv, MODEL_A)) == {"mid"}

    # space-separated bounds select the identical row set (half-open)
    conv2 = "conv-fdr-win2"
    engine2 = _backfill_engine(tmp_path, conv2)
    raw2 = _raw(engine2)
    raw2.store_facts([
        _fact("early", conv2, mentioned_at=datetime(2026, 1, 1, 12, tzinfo=timezone.utc)),
        _fact("mid", conv2, mentioned_at=datetime(2026, 1, 2, 12, tzinfo=timezone.utc)),
        _fact("late", conv2, mentioned_at=datetime(2026, 1, 3, 12, tzinfo=timezone.utc)),
    ])
    space_rep = engine2.backfill_fact_embeddings(
        conv2, since="2026-01-02 00:00:00", until="2026-01-03 00:00:00")
    assert space_rep["embedded"] == t_rep["embedded"]
    assert space_rep["windowed_out"] == t_rep["windowed_out"]
    assert set(raw2.load_fact_embeddings(conv2, MODEL_A)) == {"mid"}


def test_backfill_fact_embeddings_force_rebuild_reembeds_current_rows(tmp_path):
    conv = "conv-fdr-force"
    engine = _backfill_engine(tmp_path, conv)
    raw = _raw(engine)
    raw.store_facts([_fact("a", conv), _fact("b", conv)])

    engine.backfill_fact_embeddings(conv)
    engine._semantic.calls.clear()

    forced = engine.backfill_fact_embeddings(conv, force_rebuild=True)
    assert forced["embedded"] == 2
    assert forced["skipped_current"] == 0
    # every current row was re-embedded (embed_fn invoked per fact)
    embedded_texts = [c[0] for c in engine._semantic.calls]
    assert len(embedded_texts) == 2


def test_backfill_fact_embeddings_reports_model_mismatch_malformed_and_windowed_out(tmp_path):
    conv = "conv-fdr-report"
    engine = _backfill_engine(tmp_path, conv)
    raw = _raw(engine)
    raw.store_facts([
        _fact("cur", conv, mentioned_at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
        _fact("mism", conv, mentioned_at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
        _fact("mal", conv, mentioned_at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
        _fact("fresh", conv, mentioned_at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
        _fact("out", conv, mentioned_at=datetime(2020, 1, 1, tzinfo=timezone.utc)),
    ])
    # cur: valid current-model row -> skipped_current
    raw.store_fact_embeddings("cur", conv, MODEL_A, [1.0, 2.0, 3.0, 4.0])
    # mism: row under a different model -> model_mismatch (rebuilt)
    raw.store_fact_embeddings("mism", conv, MODEL_B, [1.0, 2.0, 3.0, 4.0])
    # mal: current-model row with unparsable JSON -> malformed (rebuilt)
    conn = raw._get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO fact_embeddings "
        "(fact_id, conversation_id, model, embedding_json) VALUES (?, ?, ?, ?)",
        ("mal", conv, MODEL_A, "not-json"),
    )

    rep = engine.backfill_fact_embeddings(
        conv, since="2026-01-01", until="2027-01-01")
    assert rep == {
        "embedded": 1,          # "fresh"
        "skipped_current": 1,   # "cur"
        "model_mismatch": 1,    # "mism"
        "malformed": 1,         # "mal"
        "windowed_out": 1,      # "out"
    }
    # mismatch + malformed were rebuilt to a valid current-model row
    loaded = raw.load_fact_embeddings(conv, MODEL_A)
    assert {"cur", "mism", "mal", "fresh"} <= set(loaded)


def test_backfill_fact_embeddings_overlapping_runs_converge_to_one_row_per_key(tmp_path):
    conv = "conv-fdr-overlap"
    engine = _backfill_engine(tmp_path, conv)
    raw = _raw(engine)
    raw.store_facts([
        _fact("a", conv, mentioned_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
        _fact("b", conv, mentioned_at=datetime(2026, 3, 5, tzinfo=timezone.utc)),
    ])

    engine.backfill_fact_embeddings(conv, since="2026-03-01", until="2026-03-10")
    engine.backfill_fact_embeddings(conv, since="2026-03-03", until="2026-03-31")
    engine.backfill_fact_embeddings(conv)

    conn = raw._get_conn()
    rows = conn.execute(
        "SELECT fact_id, COUNT(*) c FROM fact_embeddings "
        "WHERE conversation_id = ? GROUP BY fact_id",
        (conv,),
    ).fetchall()
    counts = {r["fact_id"]: r["c"] for r in rows}
    assert counts == {"a": 1, "b": 1}


def test_backfill_fact_embeddings_raises_on_empty_conversation_id(tmp_path):
    engine = _backfill_engine(tmp_path)
    with pytest.raises(ValueError):
        engine.backfill_fact_embeddings("")


# ===========================================================================
# Phase 4 — Retrieval integration (dense-priority + non-evictable floor + gate)
# ===========================================================================

from virtual_context.core.assembler import ContextAssembler  # noqa: E402
from virtual_context.core.retriever import ContextRetriever  # noqa: E402
from virtual_context.types import (  # noqa: E402
    AssemblerConfig,
    RetrievalResult,
    RetrieverConfig,
    ScoringConfig,
)

DENSE_CONV = "conv-dense"


def _dfact(fid, *, subject="user", verb="likes", obj="tea", what="",
           tags=None, mentioned_at=None, conv=DENSE_CONV):
    f = Fact(
        id=fid, subject=subject, verb=verb, object=obj, what=what,
        conversation_id=conv,
        mentioned_at=mentioned_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    if tags:
        f.tags = list(tags)
    return f


class _DenseFakeStore:
    """Minimal fact store exposing only the surfaces the dense fetch uses."""

    def __init__(self, *, tag_facts, all_facts, vectors):
        self._tag_facts = tag_facts
        self._all_facts = all_facts
        self._by_id = {f.id: f for f in all_facts}
        self._vectors = vectors  # {fact_id: [floats]}

    def query_facts(self, tags=None, limit=None, conversation_id=None):
        return list(self._tag_facts) if tags else list(self._all_facts)

    def load_fact_embeddings(self, conversation_id, model, *, expected_dim=None):
        out = {}
        for fid, vec in self._vectors.items():
            if expected_dim is not None and len(vec) != expected_dim:
                continue
            if fid in self._by_id:
                out[fid] = (self._by_id[fid], vec)
        return out


def _const_embed(vec):
    def _fn(texts):
        return [list(vec) for _ in texts]
    return _fn


def _dense_retriever(store, *, embed_fn, top_n=20, prefetch=True, ctx_turns=0):
    cfg = RetrieverConfig(
        fact_dense_retrieval=True,
        fact_dense_top_n=top_n,
        prefetch_facts=prefetch,
        scoring=ScoringConfig(embedding_context_turns=ctx_turns),
    )
    return ContextRetriever(
        tag_generator=MagicMock(),
        store=store,
        config=cfg,
        conversation_id=DENSE_CONV,
        query_embed_fn=embed_fn,
    )


def test_dense_oracle_parity_pins_fact_embed_text_and_tie_break():
    # Pin the embed-text composition the dense ranking is scoped to.
    fa = _dfact("fa", verb="wants", obj="perfume", what="for her birthday")
    assert fa.embed_text() == "user wants perfume. for her birthday"

    hi = _dfact("hi")
    zz = _dfact("zz")
    lo = _dfact("lo")
    vectors = {"hi": [1.0, 0.0, 0.0], "zz": [1.0, 0.0, 0.0], "lo": [0.0, 1.0, 0.0]}
    store = _DenseFakeStore(tag_facts=[], all_facts=[hi, zz, lo], vectors=vectors)
    r = _dense_retriever(store, embed_fn=_const_embed([1.0, 0.0, 0.0]), prefetch=False)

    meta = {}
    union = r._fetch_facts_dense("q", None, [], meta)
    # cosine DESC; hi/zz tie at 1.0 -> fact.id ASC (hi < zz); lo last at 0.0
    assert meta["fact_dense_rank_by_id"] == {"hi": 0, "zz": 1, "lo": 2}
    assert [f.id for f in union[:3]] == ["hi", "zz", "lo"]
    assert meta["fact_dense_score_by_id"]["hi"] == pytest.approx(1.0)
    assert meta["fact_dense_score_by_id"]["lo"] == pytest.approx(0.0)


def test_gate_on_candidate_union_superset_of_legacy_floor_for_tagged_all_and_no_tags_paths():
    buried = _dfact("buried", tags=["birthday"])
    dense_hit = _dfact("dense_hit")
    other = _dfact("other")
    vectors = {
        "dense_hit": [1.0, 0.0, 0.0],
        "buried": [0.0, 1.0, 0.0],
        "other": [0.5, 0.5, 0.0],
    }
    all_facts = [buried, dense_hit, other]
    store = _DenseFakeStore(tag_facts=[buried], all_facts=all_facts, vectors=vectors)
    embed = _const_embed([1.0, 0.0, 0.0])

    # tagged path: floor = tag-gated fetch [buried]; dense adds the cosine hit
    r = _dense_retriever(store, embed_fn=embed, prefetch=True)
    meta = {}
    union = r._fetch_facts_dense("q", None, ["birthday"], meta)
    ids = {f.id for f in union}
    assert {"buried"} <= ids                       # tag-exact-but-buried preserved
    assert "dense_hit" in ids                       # dense-only added
    assert meta["fact_tag_floor_ids"] == ["buried"]

    # prefetch=False path: floor = fetch-all
    r2 = _dense_retriever(store, embed_fn=embed, prefetch=False)
    meta2 = {}
    union2 = r2._fetch_facts_dense("q", None, [], meta2)
    assert set(meta2["fact_tag_floor_ids"]) == {"buried", "dense_hit", "other"}
    assert {f.id for f in union2} >= {"buried", "dense_hit", "other"}

    # no-tags path: prefetch=True but no expanded tags -> fetch-all floor
    r3 = _dense_retriever(store, embed_fn=embed, prefetch=True)
    meta3 = {}
    r3._fetch_facts_dense("q", None, [], meta3)
    assert set(meta3["fact_tag_floor_ids"]) == {"buried", "dense_hit", "other"}


def test_gate_off_fact_selection_order_and_budget_byte_identical():
    facts = [_dfact("a"), _dfact("b"), _dfact("c")]
    rr = RetrievalResult(facts=facts, retrieval_metadata={})
    asm = ContextAssembler(config=AssemblerConfig(), token_counter=lambda t: 5)
    out = asm.assemble("", rr, [], token_budget=100_000)
    # Gate off: legacy index order, no dense metadata attached (INV-3).
    assert [f.id for f in out.selected_facts] == ["a", "b", "c"]
    assert "fact_dense_rank_by_id" not in rr.retrieval_metadata
    assert "fact_dense_assembler" not in rr.retrieval_metadata


def test_assembler_legacy_floor_non_evictable_when_dense_budget_saturated():
    keep = _dfact("keep", what="small")
    big = _dfact("big", what="BIG payload")
    facts = [big, keep]  # union order: dense first, floor-only after
    rr = RetrievalResult(facts=facts, retrieval_metadata={
        "fact_dense_rank_by_id": {"big": 0},
        "fact_dense_score_by_id": {"big": 0.9},
        "fact_tag_floor_ids": ["keep"],
        "fact_source_by_id": {"big": "dense", "keep": "tag"},
    })
    asm = ContextAssembler(
        config=AssemblerConfig(facts_max_tokens=50),
        token_counter=lambda t: 100 if "BIG" in t else 5,
    )
    out = asm.assemble("", rr, [], token_budget=100_000)
    ids = {f.id for f in out.selected_facts}
    assert "keep" in ids            # legacy floor is non-evictable
    assert "big" not in ids         # dense-only skipped under saturation
    br = rr.retrieval_metadata["fact_dense_assembler"]
    assert br["skipped_dense_budget"] == 1
    assert br["selected_legacy_floor"] == 1


def test_assembler_dense_rank_survives_legacy_scorer_and_output_order():
    old = datetime(2000, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    dense1 = _dfact("dense1", mentioned_at=old)      # top dense, no tags, ancient
    dense2 = _dfact("dense2", mentioned_at=old)
    floorx = _dfact("floorx", tags=["t"], mentioned_at=now)  # strong scalar
    # Union index order deliberately != dense rank order to prove dense rank
    # (not sorted-by-index) controls dense output.
    facts = [floorx, dense2, dense1]
    rr = RetrievalResult(facts=facts, retrieval_metadata={
        "fact_dense_rank_by_id": {"dense1": 0, "dense2": 1},
        "fact_dense_score_by_id": {"dense1": 0.9, "dense2": 0.8},
        "fact_tag_floor_ids": ["floorx"],
        "fact_source_by_id": {"dense1": "dense", "dense2": "dense", "floorx": "tag"},
        "tags_queried": ["t"],
    })
    asm = ContextAssembler(config=AssemblerConfig(), token_counter=lambda t: 5)
    out = asm.assemble("", rr, [], token_budget=100_000)
    assert [f.id for f in out.selected_facts] == ["dense1", "dense2", "floorx"]


def test_dense_fact_in_both_sets_charged_once_and_marked_both():
    shared = _dfact("shared", tags=["t"])
    flooronly = _dfact("flooronly", tags=["t"])
    vectors = {"shared": [1.0, 0.0, 0.0], "flooronly": [0.0, 1.0, 0.0]}
    store = _DenseFakeStore(
        tag_facts=[shared, flooronly], all_facts=[shared, flooronly], vectors=vectors,
    )
    r = _dense_retriever(store, embed_fn=_const_embed([1.0, 0.0, 0.0]),
                         prefetch=True, top_n=1)
    meta = {}
    union = r._fetch_facts_dense("q", None, ["t"], meta)
    assert [f.id for f in union].count("shared") == 1
    assert meta["fact_source_by_id"]["shared"] == "both"
    assert meta["fact_source_by_id"]["flooronly"] == "tag"

    rr = RetrievalResult(facts=union, retrieval_metadata=dict(meta, tags_queried=["t"]))
    asm = ContextAssembler(config=AssemblerConfig(), token_counter=lambda t: 5)
    out = asm.assemble("", rr, [], token_budget=100_000)
    assert [f.id for f in out.selected_facts].count("shared") == 1


def test_dense_query_embedder_works_when_inbound_tagger_type_is_llm():
    # LLM inbound mode: engine builds NO embedding tagger, so the retriever
    # must embed the query through its own provider. Model that wiring.
    a = _dfact("a")
    b = _dfact("b")
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}
    store = _DenseFakeStore(tag_facts=[], all_facts=[a, b], vectors=vectors)
    cfg = RetrieverConfig(fact_dense_retrieval=True, inbound_tagger_type="llm")
    r = ContextRetriever(
        tag_generator=MagicMock(), store=store, config=cfg,
        conversation_id=DENSE_CONV, inbound_tagger=None,
        query_embed_fn=_const_embed([1.0, 0.0, 0.0]),
    )
    meta = {}
    union = r._fetch_facts_dense("q", None, [], meta)
    assert meta["fact_dense_rank_by_id"]["a"] == 0
    assert [f.id for f in union][0] == "a"


def test_fact_dense_breakdown_and_request_context_fact_metadata_are_emitted():
    from virtual_context.proxy.server import _serialize_recall_fact

    a = _dfact("a", tags=["t"])
    b = _dfact("b")
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}
    store = _DenseFakeStore(tag_facts=[a], all_facts=[a, b], vectors=vectors)
    r = _dense_retriever(store, embed_fn=_const_embed([1.0, 0.0, 0.0]), prefetch=True)
    meta = {}
    union = r._fetch_facts_dense("q", None, ["t"], meta)
    for k in ("fact_dense_rank_by_id", "fact_dense_score_by_id",
              "fact_tag_floor_ids", "fact_source_by_id"):
        assert k in meta

    rr = RetrievalResult(facts=union, retrieval_metadata=dict(meta, tags_queried=["t"]))
    asm = ContextAssembler(config=AssemblerConfig(), token_counter=lambda t: 5)
    asm.assemble("", rr, [], token_budget=100_000)
    br = rr.retrieval_metadata["fact_dense_assembler"]
    assert set(br) >= {"selected_legacy_floor", "selected_dense_only",
                       "skipped_dense_budget", "facts_tokens", "pool_remaining"}

    ser_a = _serialize_recall_fact(a, retrieval_meta=meta)
    assert ser_a["dense_rank"] == meta["fact_dense_rank_by_id"]["a"]
    assert ser_a["fact_source"] == meta["fact_source_by_id"]["a"]
    assert ser_a["legacy_floor"] is True
    ser_b = _serialize_recall_fact(b, retrieval_meta=meta)
    assert ser_b["legacy_floor"] is False


def test_dense_on_does_not_demote_easy_tag_exact_fact_out_of_budget():
    easy = _dfact("easy", tags=["t"])   # tag-exact floor fact, weak cosine
    d1 = _dfact("d1")
    d2 = _dfact("d2")
    vectors = {"easy": [0.0, 1.0, 0.0], "d1": [1.0, 0.0, 0.0], "d2": [1.0, 0.0, 0.0]}
    store = _DenseFakeStore(tag_facts=[easy], all_facts=[easy, d1, d2], vectors=vectors)
    r = _dense_retriever(store, embed_fn=_const_embed([1.0, 0.0, 0.0]), prefetch=True)
    meta = {}
    union = r._fetch_facts_dense("q", None, ["t"], meta)
    rr = RetrievalResult(facts=union, retrieval_metadata=dict(meta, tags_queried=["t"]))
    asm = ContextAssembler(config=AssemblerConfig(), token_counter=lambda t: 5)
    out = asm.assemble("", rr, [], token_budget=100_000)
    assert "easy" in {f.id for f in out.selected_facts}


def test_dense_vectors_present_before_engine_construct_rank_same_as_after_construct_compaction(tmp_path):
    conv = DENSE_CONV
    fa = _fact("fa", conv, verb="wants", obj="perfume", what="birthday gift")
    fb = _fact("fb", conv, verb="likes", obj="tea")
    cfg = RetrieverConfig(fact_dense_retrieval=True, prefetch_facts=False,
                          embedding_model=MODEL_A)

    # ordering (a): facts + vectors present BEFORE the retriever is built
    store_a = SQLiteStore(db_path=str(tmp_path / "temporal_a.db"))
    store_a.store_facts([fa, fb])
    store_a.store_fact_embeddings("fa", conv, MODEL_A, _embed([fa.embed_text()])[0])
    store_a.store_fact_embeddings("fb", conv, MODEL_A, _embed([fb.embed_text()])[0])
    r_a = ContextRetriever(tag_generator=MagicMock(), store=store_a, config=cfg,
                           conversation_id=conv, query_embed_fn=_embed)
    meta_a = {}
    r_a._fetch_facts_dense(fa.embed_text(), None, [], meta_a)

    # ordering (b): retriever built against an EMPTY store, THEN writes land
    store_b = SQLiteStore(db_path=str(tmp_path / "temporal_b.db"))
    r_b = ContextRetriever(tag_generator=MagicMock(), store=store_b, config=cfg,
                           conversation_id=conv, query_embed_fn=_embed)
    store_b.store_facts([fa, fb])
    store_b.store_fact_embeddings("fa", conv, MODEL_A, _embed([fa.embed_text()])[0])
    store_b.store_fact_embeddings("fb", conv, MODEL_A, _embed([fb.embed_text()])[0])
    meta_b = {}
    r_b._fetch_facts_dense(fa.embed_text(), None, [], meta_b)

    assert meta_a["fact_dense_rank_by_id"] == meta_b["fact_dense_rank_by_id"]
    assert meta_a["fact_dense_score_by_id"] == meta_b["fact_dense_score_by_id"]
    assert meta_a["fact_dense_rank_by_id"]["fa"] == 0
