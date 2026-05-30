"""Shared helpers for the compaction-backlog sweeper backend
implementations.

This module is intentionally backend-agnostic: it lives in
``virtual_context.core`` so both ``virtual_context.storage.postgres``
and ``virtual_context.storage.sqlite`` can call it without one
backend importing the other's psycopg / sqlite3 dependency. The
helper signatures take a ``conn`` whose ``execute(...)`` method
accepts a single positional ``params`` tuple (the common shape both
backends already expose) plus a ``placeholder`` string the caller
passes to keep the SQL backend-agnostic.

Per compaction-backlog sweeper spec v1.4 §3.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _col(row, name: str, idx: int):
    """Read a column from a row that may be a Mapping (``dict_row``
    on Postgres) or a ``sqlite3.Row``. Both accept string keys; we
    fall through to integer indexing in case the row-factory wrapper
    is something more exotic.
    """
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        try:
            return row[idx]
        except (KeyError, IndexError, TypeError):
            return None


def verify_backlog_candidate_under_lock(
    *,
    conn,
    candidate,
    min_backlog_turns: int,
    grace_s: float,
    placeholder: str,
) -> bool:
    """Re-verify the spec §3.1 backlog predicates under the lifecycle
    lock held by ``begin_compaction_with_lock``. Both backends'
    ``claim_compaction_backlog`` adapters pass this as the
    ``pre_begin_check`` so the active-op INSERT routes through the
    fencing primitive's allowlisted site.

    ``placeholder`` is ``"%s"`` on Postgres or ``"?"`` on SQLite.
    The grace check pulls every terminal row and computes the most
    recent timestamp in Python rather than relying on backend-specific
    text-MAX or ``make_interval`` / ``datetime('now', ...)`` syntax --
    the same hazard codex flagged in Phase 0 on the SQLite detection
    query.

    Returns True iff every §3.2 predicate passes. False on any
    mismatch; the begin primitive then aborts cleanly without
    writing phase or inserting an op row.
    """
    p = placeholder

    # 1. Conversation liveness + tenant + epoch parity.
    row = conn.execute(
        f"""
        SELECT phase, deleted_at, tenant_id, lifecycle_epoch
          FROM conversations
         WHERE conversation_id = {p}
        """,
        (candidate.conversation_id,),
    ).fetchone()
    if row is None:
        return False
    phase = _col(row, "phase", 0)
    deleted_at = _col(row, "deleted_at", 1)
    tenant_id = _col(row, "tenant_id", 2)
    lifecycle_epoch = _col(row, "lifecycle_epoch", 3)
    if str(phase) != "active":
        return False
    if deleted_at is not None and deleted_at != "":
        return False
    if str(tenant_id or "") != candidate.tenant_id:
        return False
    if int(lifecycle_epoch) != candidate.lifecycle_epoch:
        return False

    # 2. Backlog above the configured threshold (NOT the stale
    #    ``candidate.backlog_turns`` snapshot).
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS n
          FROM canonical_turns
         WHERE conversation_id = {p}
           AND tagged_at IS NOT NULL
           AND compacted_at IS NULL
        """,
        (candidate.conversation_id,),
    ).fetchone()
    if row is None or int(_col(row, "n", 0) or 0) < int(min_backlog_turns):
        return False

    # 3. No untagged canonical row exists.
    row = conn.execute(
        f"""
        SELECT 1
          FROM canonical_turns
         WHERE conversation_id = {p}
           AND tagged_at IS NULL
         LIMIT 1
        """,
        (candidate.conversation_id,),
    ).fetchone()
    if row is not None:
        return False

    # 4. No queued/running op for the current epoch. The begin
    #    primitive's ON CONFLICT path also closes this gap; we
    #    re-check here so the adapter fails closed earlier on the
    #    expected race path.
    row = conn.execute(
        f"""
        SELECT 1
          FROM compaction_operation
         WHERE conversation_id = {p}
           AND lifecycle_epoch = {p}
           AND status IN ('queued', 'running')
         LIMIT 1
        """,
        (candidate.conversation_id, candidate.lifecycle_epoch),
    ).fetchone()
    if row is not None:
        return False

    # 5. Most-recent current-epoch terminal compaction is older than
    #    the grace window. Python-side max avoids backend-specific
    #    text-MAX semantics.
    rows = conn.execute(
        f"""
        SELECT completed_at, started_at
          FROM compaction_operation
         WHERE conversation_id = {p}
           AND lifecycle_epoch = {p}
           AND status IN ('completed', 'failed', 'abandoned', 'cancelled')
        """,
        (candidate.conversation_id, candidate.lifecycle_epoch),
    ).fetchall()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=grace_s)
    most_recent: datetime | None = None
    for r in rows:
        completed = _col(r, "completed_at", 0)
        started = _col(r, "started_at", 1)
        ts = completed or started
        if ts is None or ts == "":
            continue
        if isinstance(ts, datetime):
            ts_dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        else:
            try:
                ts_dt = datetime.fromisoformat(str(ts))
            except ValueError:
                continue
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        if most_recent is None or ts_dt > most_recent:
            most_recent = ts_dt
    if most_recent is not None and most_recent >= cutoff:
        return False

    return True


__all__ = ["verify_backlog_candidate_under_lock"]
