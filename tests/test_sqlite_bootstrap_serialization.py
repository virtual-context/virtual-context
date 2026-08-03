"""The SQLite schema bootstrap must not race itself.

Connections booting together both find a trigger absent, both create it, and
the loser aborts the rest of its guarded block. Postgres solves this with a
cross-worker advisory lock; SQLite has none, so the equivalent is a lock file
plus an in-process mutex.

This closes the COLLISION defect only. It is deliberately not the
``conn.transaction()`` fix used on the Postgres trigger pairs, which closes a
different defect — a window where the table is writable while a guard trigger
is absent. The same probe that slipped 5014 unguarded writes against Postgres
slipped 0 against SQLite, whose database-level write lock stops a writer
interleaving with DDL at all. That is failure-to-reproduce rather than proof
of absence, but the asymmetry is real and the two need different fixes.
"""

from __future__ import annotations

import logging
import threading

import pytest

from virtual_context.storage.sqlite import SQLiteStore, _bootstrap_lock


class TestConcurrentBootstrap:
    def test_eight_concurrent_boots_produce_no_failure(self, tmp_path, caplog):
        """Before the guards and this lock: 41 failures in 40 such trials."""
        path = str(tmp_path / "race.db")
        SQLiteStore(path)  # first boot, schema exists

        errors: list = []

        def boot():
            try:
                SQLiteStore(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        with caplog.at_level(logging.WARNING):
            threads = [threading.Thread(target=boot) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert errors == []
        assert [r for r in caplog.records if "already exists" in r.getMessage()] == []

    def test_sequential_boots_are_still_clean(self, tmp_path):
        path = str(tmp_path / "seq.db")
        for _ in range(3):
            SQLiteStore(path)


class TestTheLockItself:
    def test_it_actually_excludes(self, tmp_path):
        """Two holders must not be inside at once."""
        path = str(tmp_path / "x.db")
        inside = []
        overlap = []
        barrier = threading.Barrier(2, timeout=5)

        def hold():
            barrier.wait()
            with _bootstrap_lock(path):
                if inside:
                    overlap.append(True)
                inside.append(1)
                threading.Event().wait(0.02)
                inside.pop()

        threads = [threading.Thread(target=hold) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert overlap == [], "two holders were inside the lock at once"

    def test_it_places_the_lock_file_beside_the_database(self, tmp_path):
        path = tmp_path / "y.db"
        with _bootstrap_lock(str(path)):
            pass
        assert (tmp_path / "y.db.bootstrap.lock").exists(), (
            "no lock file: cross-process exclusion is not in effect"
        )

    def test_it_is_released_when_the_body_raises(self, tmp_path):
        path = str(tmp_path / "z.db")
        with pytest.raises(RuntimeError):
            with _bootstrap_lock(path):
                raise RuntimeError("boom")
        # A leaked lock would deadlock here rather than fail.
        with _bootstrap_lock(path):
            pass

    def test_it_is_reentrant_across_sequential_holders(self, tmp_path):
        path = str(tmp_path / "w.db")
        for _ in range(3):
            with _bootstrap_lock(path):
                pass

    @pytest.mark.parametrize("target", [":memory:", ""])
    def test_a_memoryless_database_needs_no_lock_file(self, target):
        """Private to its connection: no second process to exclude."""
        with _bootstrap_lock(target):
            pass
