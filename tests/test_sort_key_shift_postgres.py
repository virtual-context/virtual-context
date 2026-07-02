"""Postgres twins for sort-key gap rebalance (BUG-036).

Skipped unless a Postgres DSN is configured. Mirrors
``test_ingest_sort_key_rebalance.py`` so both backends stay in lockstep
on ``shift_canonical_turn_sort_keys`` semantics.
"""
from __future__ import annotations

import uuid

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


def _store():
    from virtual_context.storage.postgres import PostgresStore  # deferred
    return PostgresStore(PG_URL)


def _fmt():
    from virtual_context.proxy.formats import detect_format
    return detect_format({"messages": []})


def _reconciler(store, conversation_id: str):
    from virtual_context.config import VirtualContextConfig
    from virtual_context.core.ingest_reconciler import IngestReconciler
    from virtual_context.core.semantic_search import SemanticSearchManager
    from virtual_context.types import StorageConfig, TagGeneratorConfig
    config = VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _pairs(*names) -> dict:
    msgs = []
    for n in names:
        msgs.append({"role": "user", "content": f"u{n}"})
        msgs.append({"role": "assistant", "content": f"a{n}"})
    return {"messages": msgs}


def _keys(conversation_id: str) -> list[tuple[str, float]]:
    conn = pg_test_conn()
    rows = conn.execute(
        "SELECT user_content || assistant_content AS content, sort_key "
        "FROM canonical_turns WHERE conversation_id = %s ORDER BY sort_key",
        (conversation_id,),
    ).fetchall()
    return [(row["content"], float(row["sort_key"])) for row in rows]


def _seed(store, conversation_id: str, *names):
    store.upsert_conversation(tenant_id="t", conversation_id=conversation_id)
    rec = _reconciler(store, conversation_id)
    rec.ingest_batch(
        conversation_id=conversation_id,
        body=_pairs(*names),
        fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )
    return rec


class TestShiftHelperPostgres:
    @pytest.mark.regression("BUG-036")
    def test_shifts_only_at_or_above_min(self):
        store = _store()
        conv = f"shift-{uuid.uuid4().hex[:12]}"
        _seed(store, conv, 0, 1, 2)
        moved = store.shift_canonical_turn_sort_keys(
            conv, min_sort_key=3000.0, delta=100000.0,
        )
        assert moved == 4
        keys = [key for _content, key in _keys(conv)]
        assert keys == [1000.0, 2000.0, 103000.0, 104000.0, 105000.0, 106000.0]

    def test_empty_range_returns_zero(self):
        store = _store()
        conv = f"shift-{uuid.uuid4().hex[:12]}"
        _seed(store, conv, 0)
        assert store.shift_canonical_turn_sort_keys(
            conv, min_sort_key=99999.0, delta=100000.0,
        ) == 0

    def test_insufficient_delta_rejected(self):
        store = _store()
        conv = f"shift-{uuid.uuid4().hex[:12]}"
        _seed(store, conv, 0, 1, 2)
        with pytest.raises(ValueError):
            store.shift_canonical_turn_sort_keys(
                conv, min_sort_key=1000.0, delta=2000.0,
            )


class TestMidInsertRebalancePostgres:
    @pytest.mark.regression("BUG-036")
    def test_mid_insert_into_exhausted_gap_succeeds(self):
        store = _store()
        conv = f"rebal-{uuid.uuid4().hex[:12]}"
        rec = _seed(store, conv, 0, 1, 2, 3)

        # Tighten the tail rows to 0.001 spacing after a2 — the state a
        # pre-fix clamped allocation left behind.
        conn = pg_test_conn()
        rows = conn.execute(
            "SELECT canonical_turn_id, sort_key FROM canonical_turns "
            "WHERE conversation_id = %s ORDER BY sort_key",
            (conv,),
        ).fetchall()
        base = float(rows[5]["sort_key"])
        for offset, row in enumerate(rows[6:], start=1):
            conn.execute(
                "UPDATE canonical_turns SET sort_key = %s "
                "WHERE canonical_turn_id = %s",
                (base + 0.001 * offset, row["canonical_turn_id"]),
            )

        result = rec.ingest_batch(
            conversation_id=conv, body=_pairs(0, 1, 2, "X"), fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
        assert result.turns_written == 2
        ordered = [content for content, _key in _keys(conv)]
        assert ordered == [
            "u0", "a0", "u1", "a1", "u2", "a2", "uX", "aX", "u3", "a3",
        ]
        keys = [key for _content, key in _keys(conv)]
        assert len(keys) == len(set(keys))
