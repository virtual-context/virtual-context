"""Shared Postgres test helper: direct seeding/assertion connection.

The DATABASE_URL-gated test fleet needs an imperative connection for
seeding rows and asserting on table state. The storage backend's pool
is not a substitute: checking test connections out of the store's pool
couples test reads to pool sizing, and ``store.close()`` in a teardown
would strand them.

``pg_test_conn()`` returns a process-cached direct psycopg connection
configured identically to the backend pool's connections (``dict_row``
+ ``autocommit``), so row-access and commit semantics in tests match
what the store's own code paths see.
"""

from __future__ import annotations

import os

_CONN = None


def pg_dsn() -> str | None:
    """The single sanctioned DSN resolver for the Postgres test fleet.

    Every fleet file must gate and connect through this function —
    never through a direct ``os.environ`` lookup — so that setting
    EITHER sanctioned variable enables the entire fleet. A file that
    reads the environment directly can silently honor only one
    spelling; the fleet sentinel's gate-uniformity lint enforces use
    of this resolver.
    """
    return os.environ.get("DATABASE_URL") or os.environ.get("VC_TEST_POSTGRES_URL")


def pg_test_conn():
    """Return the shared direct connection to ``DATABASE_URL``.

    Recreated transparently if a previous test closed it (e.g. by using
    it as a context manager, which closes on exit in psycopg 3).
    """
    global _CONN
    import psycopg
    from psycopg.rows import dict_row

    if _CONN is None or _CONN.closed:
        dsn = pg_dsn()
        assert dsn, "no Postgres DSN configured"
        _CONN = psycopg.connect(
            dsn,
            row_factory=dict_row,
            autocommit=True,
        )
    return _CONN
