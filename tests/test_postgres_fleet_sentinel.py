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


def test_pg_fleet_files_reference_only_symbols_that_exist():
    """ALWAYS-ON: a fleet file must not name a symbol the code lost.

    The fleet skips wholesale without a DSN, so a test in it can rot
    silently: a monkeypatch naming a renamed or deleted attribute raises
    only when the fleet actually runs, and on a DSN-less machine the file
    stays green-looking forever. That is the worst state for a safety
    test — skipped and broken — and it happened: a concurrency test
    patched a module constant that had been replaced, and the proof it
    carried stopped running without anyone seeing a failure.

    This lint runs regardless of DSN. It parses each fleet file, resolves
    every ``monkeypatch.setattr(target, "name", ...)`` and
    ``getattr(target, "name", ...)`` whose target is a module or class
    imported by that file, and asserts the named attribute exists.
    Targets it cannot resolve statically (locals, fixtures) are skipped:
    the point is the common dangerous pattern, not a type checker.
    """
    import ast
    import importlib
    from pathlib import Path

    fleet = sorted(Path(__file__).parent.glob("*postgres*.py"))
    assert fleet, "fleet glob found nothing; the lint is misconfigured"
    problems: list[str] = []

    for path in fleet:
        tree = ast.parse(path.read_text(encoding="utf-8"))

        # alias -> importable dotted path, module-level and function-level.
        targets: dict[str, object] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    try:
                        targets[a.asname or a.name.split(".")[0]] = (
                            importlib.import_module(a.name)
                        )
                    except ImportError:
                        pass
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    try:
                        mod = importlib.import_module(node.module)
                        obj = getattr(mod, a.name, None)
                        if obj is None:
                            obj = importlib.import_module(
                                f"{node.module}.{a.name}"
                            )
                        targets[a.asname or a.name] = obj
                    except ImportError:
                        pass

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            is_setattr = (
                isinstance(fn, ast.Attribute) and fn.attr == "setattr"
            ) or (isinstance(fn, ast.Name) and fn.id in ("setattr", "getattr", "delattr"))
            if not is_setattr or len(node.args) < 2:
                continue
            tgt, name_node = node.args[0], node.args[1]
            if not (
                isinstance(tgt, ast.Name)
                and isinstance(name_node, ast.Constant)
                and isinstance(name_node.value, str)
            ):
                continue
            # Deliberately-optional access is not rot. A three-argument
            # getattr supplies its own fallback, and a monkeypatch with
            # raising=False is announcing the attribute may be absent.
            if (
                isinstance(fn, ast.Name)
                and fn.id == "getattr"
                and len(node.args) >= 3
            ):
                continue
            if any(
                kw.arg == "raising"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            ):
                continue
            resolved = targets.get(tgt.id)
            if resolved is None:
                continue  # a local; not statically resolvable
            if not hasattr(resolved, name_node.value):
                problems.append(
                    f"{path.name}:{node.lineno}: patches "
                    f"{tgt.id}.{name_node.value!r}, which does not exist"
                )

    assert not problems, (
        "fleet files reference symbols the code no longer defines "
        "(these tests would die at runtime on the fleet host while "
        "looking skipped-green everywhere else):\n  " + "\n  ".join(problems)
    )
