"""The presence flag must answer identically on both backends.

The flag stands in for ``(value or "").strip()``. It was previously
written out per dialect, and the two dialects disagreed with each other
and with Python: both listed ASCII whitespace only, and the PostgreSQL
spelling additionally lost the vertical tab and gained the letter ``v``,
because escape strings define ``\\b \\f \\n \\r \\t`` and drop the
backslash on anything else.

Nothing caught that, because every test ran on SQLite. These tests run
against real PostgreSQL, and they are the reason the gap cannot reopen
silently: the equivalent SQLite assertions live in
``test_ingest_projected_rows.py`` and the two must agree.

Skipped unless a Postgres DSN is configured. The fleet runs serially.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid

import pytest

from virtual_context.core.canonical_turns import STRIP_WHITESPACE

from tests.pg_helpers import pg_dsn, pg_test_conn

pytestmark = pytest.mark.skipif(
    not pg_dsn(), reason="requires a Postgres DSN",
)


def _store():
    from virtual_context.storage.postgres import PostgresStore
    return PostgresStore(pg_dsn())


@pytest.fixture()
def conv(request):
    """A throwaway conversation id, cleaned up after the test."""
    conversation_id = f"presence-{uuid.uuid4()}"
    store = _store()
    store.upsert_conversation(tenant_id="t", conversation_id=conversation_id)
    yield conversation_id, store
    with pg_test_conn() as conn:
        conn.execute(
            "DELETE FROM canonical_turns WHERE conversation_id = %s",
            (conversation_id,),
        )
        conn.execute(
            "DELETE FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        )


@pytest.mark.regression("BUG-045")
def test_postgres_trims_exactly_what_python_strips(conv):
    """Every character Python strips must read as "no content" here.

    A disagreement is a false positive: a whitespace-only row reports as
    carrying user text, which lets a row that is not the user half of a
    turn take durable speaker attribution.
    """
    conversation_id, store = conv
    disagreements = []
    for index, ch in enumerate(sorted(STRIP_WHITESPACE)):
        store.save_canonical_turn(
            conversation_id, index, ch * 3, "assistant text",
            canonical_turn_id=str(uuid.uuid4()),
            sort_key=float((index + 1) * 1000.0),
            turn_hash=f"h{index}",
        )
    rows = store.get_canonical_turn_reconcile_rows(conversation_id)
    assert len(rows) == len(set(STRIP_WHITESPACE))
    for row in rows:
        if row.has_user_content:
            disagreements.append(row.turn_hash)
    assert not disagreements, (
        f"Postgres reports content for whitespace-only rows: {disagreements}"
    )


@pytest.mark.regression("BUG-045")
@pytest.mark.parametrize(
    "user_text",
    [
        "v",          # the letter the old escaped literal trimmed by mistake
        "vvv",
        "\x0b",       # the vertical tab the old escaped literal missed
        "\x0b\x0b",
        "real content",
        " padded ",
        "0",
        "",
    ],
)
def test_postgres_presence_flag_agrees_with_python_strip(conv, user_text):
    """The two spellings of the same question must give the same answer."""
    conversation_id, store = conv
    store.save_canonical_turn(
        conversation_id, 0, user_text, "assistant text",
        canonical_turn_id=str(uuid.uuid4()), sort_key=1000.0, turn_hash="h1",
    )
    row = store.get_canonical_turn_reconcile_rows(conversation_id)[0]
    assert row.has_user_content is bool(user_text.strip()), (
        f"Postgres disagrees with str.strip() for {user_text!r}"
    )


@pytest.mark.regression("BUG-045")
def test_postgres_and_sqlite_agree_on_every_stripped_character(tmp_path, conv):
    """Pin the two backends against each other, not just against Python.

    The original defect was cross-dialect: each backend was internally
    consistent and they disagreed with one another, which no single-
    backend test could see.
    """
    from virtual_context.storage.sqlite import SQLiteStore

    conversation_id, pg_store = conv
    lite = SQLiteStore(tmp_path / "s.db")
    lite.upsert_conversation(tenant_id="t", conversation_id=conversation_id)

    probes = sorted(STRIP_WHITESPACE) + ["v", "real", " x ", ""]
    for index, text in enumerate(probes):
        for store in (pg_store, lite):
            store.save_canonical_turn(
                conversation_id, index, text, "assistant text",
                canonical_turn_id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{index}")),
                sort_key=float((index + 1) * 1000.0),
                turn_hash=f"h{index}",
            )

    pg_flags = {
        r.turn_hash: r.has_user_content
        for r in pg_store.get_canonical_turn_reconcile_rows(conversation_id)
    }
    lite_flags = {
        r.turn_hash: r.has_user_content
        for r in lite.get_canonical_turn_reconcile_rows(conversation_id)
    }
    assert pg_flags == lite_flags
