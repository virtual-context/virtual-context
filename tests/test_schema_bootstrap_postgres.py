"""Schema-bootstrap serialization across concurrent workers (BUG-037).

Skipped unless a Postgres DSN is configured. Pre-fix, workers booting
together raced the bootstrap DDL: concurrent ``CREATE OR REPLACE
FUNCTION`` on the same function fails with "tuple concurrently updated"
and trigger ``DROP`` + ``CREATE`` pairs fail with DuplicateObject; the
losing worker logged a "bootstrap failed" warning and skipped the rest
of its guarded block. ``_ensure_schema`` now holds a session-scoped
advisory lock for the whole bootstrap.
"""
from __future__ import annotations

import logging
import threading

import pytest

from tests.pg_helpers import pg_dsn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


class TestSchemaBootstrapSerialization:
    @pytest.mark.regression("BUG-037")
    def test_concurrent_bootstrap_no_ddl_race_warnings(self, caplog):
        """Eight stores bootstrapping together must not lose the DDL race."""
        from virtual_context.storage.postgres import PostgresStore  # deferred

        errors: list[BaseException] = []

        def _boot():
            try:
                store = PostgresStore(PG_URL)
                store.close()
            except BaseException as exc:  # noqa: BLE001 — collect everything
                errors.append(exc)

        with caplog.at_level(
            logging.WARNING, logger="virtual_context.storage.postgres"
        ):
            threads = [threading.Thread(target=_boot) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)

        assert not errors, errors
        bad = [
            rec.message for rec in caplog.records
            if "bootstrap failed" in rec.message
            or "tuple concurrently updated" in str(rec.exc_text or "")
        ]
        assert not bad, bad
