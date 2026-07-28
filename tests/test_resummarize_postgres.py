"""Selection semantics for the repair command, against real Postgres.

Seeds one conversation with the four populations the predicate must
separate: a damaged row (strict prefix), an intentional passthrough stub
(summary == full_text), a short-source row, and a healthy row. The
strict-prefix clause and the stripped-length split cannot be tested on
SQLite (``xmin`` and ``btrim`` semantics are the production surface).

Skipped unless a Postgres DSN is configured. The fleet runs serially.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid

import pytest

from virtual_context.cli.resummarize_cmd import _selection_sql
from virtual_context.core.canonical_turns import STRIP_WHITESPACE

from tests.pg_helpers import pg_dsn, pg_test_conn

pytestmark = pytest.mark.skipif(
    not pg_dsn(), reason="requires a Postgres DSN",
)


def _seed_segment(conn, conversation_id, ref, summary, full_text):
    conn.execute(
        "INSERT INTO segments (ref, conversation_id, primary_tag, summary, "
        "full_text, summary_tokens, full_tokens, created_at, "
        "start_timestamp, end_timestamp) "
        "VALUES (%s, %s, 'legal', %s, %s, %s, %s, "
        "'2026-07-25T00:00:00+00:00', '2026-07-25T00:00:00+00:00', "
        "'2026-07-25T00:00:00+00:00')",
        (ref, conversation_id, summary, full_text,
         len(summary) // 4, len(full_text) // 4),
    )


@pytest.fixture()
def seeded(request):
    conversation_id = f"resum-{uuid.uuid4()}"
    long_text = "Filing detail: the deadline moved to March. " * 40
    with pg_test_conn() as conn:
        # try/finally around seeding AND the test body: the inserts are
        # autocommitted one by one, so a failure on a later seed must
        # still delete the earlier rows.
        try:
            rows = {
                "damaged": (long_text[:400], long_text),
                "stub": (long_text, long_text),
                "short": ("ok" * 10, ("ok" * 10) + "!" + STRIP_WHITESPACE),
                "healthy": ("A real summary of the filing story.", long_text),
            }
            refs = {}
            for name, (summary, full_text) in rows.items():
                ref = f"{conversation_id}-{name}"
                _seed_segment(conn, conversation_id, ref, summary, full_text)
                refs[name] = ref
            yield conversation_id, refs, conn
        finally:
            conn.execute(
                "DELETE FROM segments WHERE conversation_id = %s",
                (conversation_id,),
            )


def _select(conn, conversation_id, include_short=False, after_ref=None):
    sql = _selection_sql(include_short, None, None, after_ref)
    return conn.execute(sql, {
        "conversation_id": conversation_id,
        "strip_ws": STRIP_WHITESPACE,
        "since": None, "until": None, "after_ref": after_ref,
    }).fetchall()


def test_selection_takes_only_the_damaged_row(seeded):
    conversation_id, refs, conn = seeded
    selected = {r["ref"] for r in _select(conn, conversation_id)}
    assert selected == {refs["damaged"]}


def test_equality_stub_is_never_selected_even_with_include_short(seeded):
    conversation_id, refs, conn = seeded
    selected = {r["ref"] for r in _select(conn, conversation_id, include_short=True)}
    assert refs["stub"] not in selected


def test_include_short_opts_in_the_short_row(seeded):
    conversation_id, refs, conn = seeded
    gated = {r["ref"] for r in _select(conn, conversation_id)}
    opted = {r["ref"] for r in _select(conn, conversation_id, include_short=True)}
    assert refs["short"] not in gated
    assert refs["short"] in opted


def test_stripped_length_split_agrees_with_python_strip(seeded):
    """The short row's full_text is 21 content chars plus every character
    Python strips; the split must measure the stripped length."""
    conversation_id, refs, conn = seeded
    row = conn.execute(
        "SELECT length(btrim(full_text, %s)) AS stripped, length(full_text) AS raw "
        "FROM segments WHERE ref = %s",
        (STRIP_WHITESPACE, refs["short"]),
    ).fetchone()
    assert row["stripped"] == 21
    assert row["raw"] == 21 + len(STRIP_WHITESPACE)


def test_equality_overlap_probe_is_zero_with_strict_predicate(seeded):
    conversation_id, refs, conn = seeded
    sql = _selection_sql(True, None, None, None)
    params = {
        "conversation_id": conversation_id,
        "strip_ws": STRIP_WHITESPACE,
        "since": None, "until": None, "after_ref": None,
    }
    overlap = conn.execute(
        f"SELECT count(*) AS n FROM ({sql}) sel WHERE sel.summary = sel.full_text",
        params,
    ).fetchone()["n"]
    assert overlap == 0


def test_rows_carry_a_usable_row_version(seeded):
    conversation_id, refs, conn = seeded
    row = _select(conn, conversation_id)[0]
    assert row["row_version"].isdigit()
    # An unrelated-column write moves xmin, so a stale version no longer
    # matches: the CAS shape the apply path relies on.
    conn.execute(
        "UPDATE segments SET compaction_model = 'x' WHERE ref = %s",
        (row["ref"],),
    )
    fresh = _select(conn, conversation_id)[0]
    assert fresh["row_version"] != row["row_version"]
    stale_match = conn.execute(
        "SELECT count(*) AS n FROM segments "
        "WHERE ref = %s AND xmin::text = %s",
        (row["ref"], row["row_version"]),
    ).fetchone()["n"]
    assert stale_match == 0


def test_dry_run_connection_is_server_side_read_only():
    import psycopg

    from virtual_context.cli.resummarize_cmd import _connect

    with _connect(pg_dsn(), read_only=True) as conn:
        assert conn.execute("SELECT 1").fetchone()["?column?"] == 1
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute(
                "UPDATE segments SET compaction_model = compaction_model "
                "WHERE false",
            )
