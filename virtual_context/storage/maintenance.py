"""Explicit SQLite maintenance without application startup or schema migration."""
from __future__ import annotations

from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile


FTS_INDEXES = (
    "segments_fts", "segments_fts_full", "facts_fts",
    "tag_summaries_fts", "tool_outputs_fts",
)


@contextmanager
def sqlite_maintenance_connection(
    db_path: str | Path, *, dry_run: bool = True,
) -> Iterator[sqlite3.Connection]:
    """Open an existing database, checking a private consistent copy by default.

    FTS5's integrity command uses INSERT even when it only checks postings.
    A read-only source connection and SQLite's backup API include committed WAL
    content without running those commands, schema migrations, or startup repair
    against the source. The private copy is removed on exit. Apply mode opens
    the existing file directly and cannot silently create a missing database.
    """
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"SQLite database does not exist: {path}")
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a boolean")
    if not dry_run:
        conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()
        return
    with tempfile.TemporaryDirectory(prefix="vc-sqlite-audit-") as directory:
        # TemporaryDirectory is mode 0700; no source text is exposed in logs.
        copy = sqlite3.connect(Path(directory) / "snapshot.db", isolation_level=None)
        try:
            source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)
            try:
                source.execute("PRAGMA query_only=ON")
                source.backup(copy, pages=256)
            finally:
                source.close()
            yield copy
        finally:
            copy.close()


def repair_fts_indexes(
    conn: sqlite3.Connection,
    index_names: Collection[str] | None = None,
    *,
    dry_run: bool = True,
) -> dict[str, str]:
    """Check content parity and optionally rebuild only named, known indexes.

    The caller supplies an already opened connection. Dry checks roll back their
    commands; use sqlite_maintenance_connection for a source file with no writes.
    Missing tables, lock errors and I/O failures propagate instead of prompting
    schema creation or being mislabeled as corrupt indexes.
    """
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a boolean")
    selected = list(dict.fromkeys(FTS_INDEXES if index_names is None else index_names))
    unknown = set(selected) - set(FTS_INDEXES)
    if unknown:
        raise ValueError(f"Unsupported FTS indexes: {', '.join(sorted(unknown))}")
    if conn.in_transaction:
        raise RuntimeError("FTS maintenance requires its own transaction")
    statuses: dict[str, str] = {}
    for name in selected:
        # Lock before reading so source rows cannot change before verification.
        conn.execute("BEGIN IMMEDIATE")
        try:
            try:
                conn.execute(f"INSERT INTO {name}({name}, rank) VALUES('integrity-check', 1)")
                statuses[name] = "ok"
            except sqlite3.DatabaseError as exc:
                code = getattr(exc, "sqlite_errorcode", 0) or 0
                if code & 0xFF != sqlite3.SQLITE_CORRUPT:
                    raise
                if dry_run:
                    statuses[name] = "needs_rebuild"
                else:
                    conn.execute(f"INSERT INTO {name}({name}) VALUES('rebuild')")
                    conn.execute(f"INSERT INTO {name}({name}, rank) VALUES('integrity-check', 1)")
                    statuses[name] = "rebuilt"
            conn.execute("ROLLBACK" if dry_run else "COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return statuses
