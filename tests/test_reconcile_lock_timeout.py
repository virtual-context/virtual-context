from __future__ import annotations

from contextlib import contextmanager
import os
import signal
import threading

import psycopg
import pytest

from virtual_context.core.exceptions import ConversationReconcileBusy
from virtual_context.storage.postgres import PostgresStore


class _Result:
    def fetchone(self):
        # The pool is configured with dict_row; the reconcile body reads
        # named columns from the lifecycle row.
        return {"conversation_id": "conv", "generation": 0, "deleted": False}


class _Connection:
    def __init__(self):
        self.statements: list[str] = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, statement, _params=None):
        self.statements.append(str(statement))
        return _Result()


class _BusyConnection(_Connection):
    def execute(self, statement, _params=None):
        self.statements.append(str(statement))
        if "INSERT INTO conversation_lifecycle" in str(statement):
            raise psycopg.errors.LockNotAvailable("lock timeout")
        return _Result()


class _Pool:
    def __init__(self, connection):
        self._connection = connection

    @contextmanager
    def connection(self):
        yield self._connection


def test_reconcile_bounds_row_lock_before_entering_protected_body():
    connection = _Connection()
    store = PostgresStore.__new__(PostgresStore)
    store.pool = _Pool(connection)
    body_entered = False

    with store.conversation_reconcile("conv"):
        body_entered = True

    assert body_entered is True
    assert connection.statements[0] == "SET LOCAL lock_timeout = '5s'"
    assert "INSERT INTO conversation_lifecycle" in connection.statements[1]
    assert "FOR UPDATE" in connection.statements[2]


def test_reconcile_lock_timeout_is_explicitly_retryable_and_never_enters_body():
    connection = _BusyConnection()
    store = PostgresStore.__new__(PostgresStore)
    store.pool = _Pool(connection)
    body_entered = False

    with pytest.raises(ConversationReconcileBusy):
        with store.conversation_reconcile("conv"):
            body_entered = True

    assert body_entered is False


def test_reconcile_holder_watchdog_terminates_the_owning_worker(monkeypatch):
    connection = _Connection()
    store = PostgresStore.__new__(PostgresStore)
    store.pool = _Pool(connection)
    fired = threading.Event()
    calls: list[tuple[int, signal.Signals]] = []

    monkeypatch.setenv("VC_RECONCILE_HOLDER_TIMEOUT_SECONDS", "0.02")

    def _kill(pid, sig):
        calls.append((pid, sig))
        fired.set()

    monkeypatch.setattr("virtual_context.storage.postgres.os.kill", _kill)

    with store.conversation_reconcile("conv"):
        assert fired.wait(timeout=1.0)

    assert calls == [(os.getpid(), signal.SIGKILL)]
