"""The DSN resolver must let the test-scoped variable win a tie.

The fleet behind this resolver seeds and deletes rows. If a shell
happens to carry a ``DATABASE_URL`` pointing somewhere important while
the operator exports ``VC_TEST_POSTGRES_URL`` to sandbox the run, the
sandbox variable must be the one honored — a precedence inversion here
is silent and succeeds against the wrong database.
"""

from tests.pg_helpers import pg_dsn


def test_test_scoped_variable_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://other/db")
    monkeypatch.setenv("VC_TEST_POSTGRES_URL", "postgresql://scratch/test")
    assert pg_dsn() == "postgresql://scratch/test"


def test_database_url_still_enables_the_fleet_alone(monkeypatch):
    monkeypatch.delenv("VC_TEST_POSTGRES_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://other/db")
    assert pg_dsn() == "postgresql://other/db"


def test_unset_environment_disables_the_fleet(monkeypatch):
    monkeypatch.delenv("VC_TEST_POSTGRES_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert pg_dsn() is None


def test_fleet_sentinel_uses_the_shared_resolver():
    """The sentinel must resolve through pg_dsn(), not its own read.

    The sentinel polices every fleet file's gate; nothing polices the
    sentinel. A private environment read there once kept the old
    precedence, so the fleet and the sentinel's schema-bootstrapping
    preflight could target different databases with both variables set.
    """
    import pathlib

    src = (pathlib.Path(__file__).parent
           / "test_postgres_fleet_sentinel.py").read_text(encoding="utf-8")
    assert "_PG_DSN = pg_dsn()" in src
    module_level = src.split("def test_", 1)[0]
    assert 'environ.get("DATABASE_URL")' not in module_level
    assert 'environ.get("VC_TEST_POSTGRES_URL")' not in module_level
