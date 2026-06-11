"""Sentinel guards for the Postgres-gated test fleet.

The fleet's historical failure mode: every Postgres-backed test file is
env-gated, so a substrate break (store can't construct, helper can't
connect, fixtures error at setup) is indistinguishable from "Postgres
not configured" unless someone reads the error output. These guards
make the two states distinguishable:

* ``test_pg_substrate_preflight`` runs whenever a Postgres DSN is
  configured and fails loudly if the store or the shared test helper
  cannot reach the database — the first, fastest signal that the rest
  of the fleet's results are meaningful.
* ``test_pg_fleet_gates_are_uniform`` runs ALWAYS (no env gate) and
  fails if any Postgres test file gates on an unknown environment
  variable spelling — a file gated on a misspelled or novel variable
  never runs anywhere, silently.

Run the fleet serially (``-n0``): the files share one database and the
schema bootstrap DDL races under parallel workers.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

_PG_DSN = os.environ.get("DATABASE_URL") or os.environ.get("VC_TEST_POSTGRES_URL")

#: The only sanctioned gate variables for Postgres-backed test files.
_SANCTIONED_VARS = {"DATABASE_URL", "VC_TEST_POSTGRES_URL"}


@pytest.mark.skipif(not _PG_DSN, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set")
def test_pg_substrate_preflight():
    """Fail loudly if the Postgres substrate the fleet depends on is broken.

    Covers the three layers every fleet test needs: store construction
    (schema bootstrap), a pooled connection, and the shared direct test
    helper. If this test fails, every green/skip elsewhere in the fleet
    is unreliable.
    """
    from virtual_context.storage.postgres import PostgresStore
    from tests.pg_helpers import pg_test_conn

    store = PostgresStore(_PG_DSN)
    try:
        with store.pool.connection() as conn:
            row = conn.execute("SELECT 1 AS ok").fetchone()
            assert row["ok"] == 1
    finally:
        store.close()

    helper_row = pg_test_conn().execute("SELECT 1 AS ok").fetchone()
    assert helper_row["ok"] == 1


def test_pg_fleet_gates_are_uniform():
    """Every Postgres test file must gate through the shared DSN resolver.

    Two silent-skip shapes are banned:

    * gating on an unsanctioned env-variable spelling — the file never
      runs anywhere;
    * reading a sanctioned variable DIRECTLY via ``os.environ`` — the
      file honors only one of the two sanctioned spellings, so the
      fleet silently loses tests under the other (observed: 61 tests
      skipped under VC_TEST_POSTGRES_URL while green under
      DATABASE_URL).

    Files must import and use ``tests.pg_helpers.pg_dsn`` for both the
    skip gate and store construction.
    """
    tests_dir = pathlib.Path(__file__).parent
    env_read = re.compile(r"os\.environ(?:\.get)?[\(\[]\s*[\"']([A-Z0-9_]+)[\"']")
    offenders: dict[str, set[str]] = {}
    for path in sorted(tests_dir.glob("*postgres*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        vars_read = set(env_read.findall(text))
        unknown = vars_read - _SANCTIONED_VARS
        if unknown:
            offenders[path.name] = unknown
        # Direct reads of the sanctioned vars honor only one spelling;
        # require the shared resolver instead.
        direct_sanctioned = vars_read & _SANCTIONED_VARS
        if direct_sanctioned:
            offenders[path.name] = (
                offenders.get(path.name, set())
                | {f"direct read of {v} (use pg_dsn())" for v in direct_sanctioned}
            )
        # A Postgres-named test file with no gate at all would run
        # (and fail) everywhere — flag it unless it self-skips.
        uses_db = "pg_test_conn" in text or "pg_dsn" in text
        if not vars_read and "pg_dsn" not in text and "skipif" not in text and uses_db:
            offenders[path.name] = {"<no env gate>"}
    assert not offenders, (
        "Postgres test files must gate via tests.pg_helpers.pg_dsn(): "
        f"{offenders}"
    )
