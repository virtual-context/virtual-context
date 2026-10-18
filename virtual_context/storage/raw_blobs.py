"""Blob storage for compacted raw bodies, shared by the SQL backends.

``canonical_raw_blobs`` holds each distinct inline image or file once, keyed
by the SHA-256 of its data; canonical rows reference it from their raw body
(see ``core.raw_bodies``). A blob is shared by every row that references it
and removed by :func:`delete_unreferenced_blobs` once no row does.

Attested canonical rows refuse any change to their raw bodies. Rewriting a
stored body into its compacted form is the one exception: the writer checks
that the new body is exactly the compaction of the stored one and records a
``canonical_raw_compaction_grants`` row for the turn, which the guard trigger
accepts and the writer deletes in the same transaction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..core.raw_bodies import BLOB_REF_PREFIX, blob_refs, compact_raw_body

# A blob referenced this recently is kept even when no committed row
# references it yet: a writer may be about to commit one.
GC_GRACE = timedelta(minutes=5)


def _placeholder(dialect: str) -> str:
    return "%s" if dialect == "postgres" else "?"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_raw_blob_schema(conn, dialect: str) -> None:
    canonical_id_type = "UUID" if dialect == "postgres" else "TEXT"
    conn.execute(
        """CREATE TABLE IF NOT EXISTS canonical_raw_blobs (
               blob_hash TEXT PRIMARY KEY,
               media_type TEXT NOT NULL DEFAULT '',
               data TEXT NOT NULL,
               byte_len INTEGER NOT NULL,
               created_at TEXT NOT NULL,
               last_referenced_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS canonical_raw_compaction_grants (
               canonical_turn_id {canonical_id_type} PRIMARY KEY
           )"""
    )


def raw_body_guard_sql(old: str = "OLD", new: str = "NEW", *, dialect: str) -> str:
    """Trigger predicate: the raw bodies changed outside a compaction grant."""
    distinct = "IS DISTINCT FROM" if dialect == "postgres" else "IS NOT"
    return (
        f"(({old}.user_raw_content {distinct} {new}.user_raw_content OR "
        f"{old}.assistant_raw_content {distinct} {new}.assistant_raw_content) AND NOT EXISTS ("
        f"SELECT 1 FROM canonical_raw_compaction_grants g "
        f"WHERE g.canonical_turn_id = {old}.canonical_turn_id))"
    )


def store_blobs(conn, dialect: str, blobs: dict[str, tuple[str, str]]) -> None:
    p = _placeholder(dialect)
    now = _now()
    for digest, (media_type, data) in blobs.items():
        conn.execute(
            f"""INSERT INTO canonical_raw_blobs
                   (blob_hash, media_type, data, byte_len, created_at, last_referenced_at)
                VALUES ({p}, {p}, {p}, {p}, {p}, {p})
                ON CONFLICT (blob_hash) DO UPDATE SET last_referenced_at = excluded.last_referenced_at""",
            (digest, media_type, data, len(data), now, now),
        )


def load_blobs(conn, dialect: str, hashes) -> dict[str, tuple[str, str]]:
    wanted = sorted(set(hashes))
    if not wanted:
        return {}
    if dialect == "postgres":
        rows = conn.execute(
            "SELECT blob_hash, media_type, data FROM canonical_raw_blobs WHERE blob_hash = ANY(%s)",
            (wanted,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT blob_hash, media_type, data FROM canonical_raw_blobs "
            f"WHERE blob_hash IN ({','.join('?' for _ in wanted)})",
            wanted,
        ).fetchall()
    return {row["blob_hash"]: (row["media_type"], row["data"]) for row in rows}


def prepare_raw_write(
    conn,
    dialect: str,
    canonical_turn_id: str,
    user_raw: str | None,
    assistant_raw: str | None,
) -> tuple[str | None, str | None, bool]:
    """Compact both raw bodies for a canonical write and store their blobs.

    Returns the bodies to write and whether a compaction grant was recorded
    for an existing row whose stored bodies are the uncompacted form of them;
    the caller deletes the grant with :func:`clear_grant` after its write.
    """
    user = compact_raw_body(user_raw)
    assistant = compact_raw_body(assistant_raw)
    store_blobs(conn, dialect, {**user.blobs, **assistant.blobs})
    granted = False
    if user.text != user_raw or assistant.text != assistant_raw:
        p = _placeholder(dialect)
        existing = conn.execute(
            f"""SELECT user_raw_content, assistant_raw_content FROM canonical_turns
                 WHERE canonical_turn_id = {p}""",
            (canonical_turn_id,),
        ).fetchone()
        if (
            existing is not None
            and (existing["user_raw_content"], existing["assistant_raw_content"]) != (user.text, assistant.text)
            and compact_raw_body(existing["user_raw_content"]).text == user.text
            and compact_raw_body(existing["assistant_raw_content"]).text == assistant.text
        ):
            grant(conn, dialect, canonical_turn_id)
            granted = True
    return user.text, assistant.text, granted


def grant(conn, dialect: str, canonical_turn_id: str) -> None:
    p = _placeholder(dialect)
    conn.execute(
        f"INSERT INTO canonical_raw_compaction_grants (canonical_turn_id) VALUES ({p}) "
        f"ON CONFLICT (canonical_turn_id) DO NOTHING",
        (canonical_turn_id,),
    )


def clear_grant(conn, dialect: str, canonical_turn_id: str) -> None:
    p = _placeholder(dialect)
    conn.execute(
        f"DELETE FROM canonical_raw_compaction_grants WHERE canonical_turn_id = {p}",
        (canonical_turn_id,),
    )


def delete_unreferenced_blobs(conn, dialect: str) -> int:
    """Delete blobs no canonical row references, outside the grace window."""
    pattern = f"%{BLOB_REF_PREFIX}%"
    p = _placeholder(dialect)
    referenced: set[str] = set()
    rows = conn.execute(
        f"""SELECT user_raw_content, assistant_raw_content FROM canonical_turns
             WHERE user_raw_content LIKE {p} OR assistant_raw_content LIKE {p}""",
        (pattern, pattern),
    ).fetchall()
    for row in rows:
        referenced |= blob_refs(row["user_raw_content"]) | blob_refs(row["assistant_raw_content"])
    cutoff = (datetime.now(timezone.utc) - GC_GRACE).isoformat()
    stale = [
        row["blob_hash"]
        for row in conn.execute(
            f"SELECT blob_hash FROM canonical_raw_blobs WHERE last_referenced_at < {p}",
            (cutoff,),
        ).fetchall()
        if row["blob_hash"] not in referenced
    ]
    deleted = 0
    for digest in stale:
        # Re-checking the timestamp in the DELETE keeps a blob a concurrent
        # writer has just referenced.
        cur = conn.execute(
            f"DELETE FROM canonical_raw_blobs WHERE blob_hash = {p} AND last_referenced_at < {p}",
            (digest, cutoff),
        )
        deleted += int(cur.rowcount or 0)
    return deleted


def compact_stored_raw_bodies(
    store,
    dialect: str,
    conversation_id: str,
    *,
    batch_size: int = 5,
    dry_run: bool = False,
) -> dict[str, int]:
    """Rewrite one conversation's stored raw bodies into their compacted form.

    Each row is read and its blobs stored before any transaction touches
    ``canonical_turns``; the row is then rewritten in a transaction of its
    own that applies only if its bodies are still the ones read. Lock time
    on the table stays at one small update per row, so schema setup and live
    writes never wait behind the transfer of large bodies. ``batch_size``
    bounds how many row ids are read per query.
    """
    p = _placeholder(dialect)
    counts = {"rows_scanned": 0, "rows_compacted": 0, "bytes_before": 0, "bytes_after": 0, "blobs": 0}
    with store._raw_blob_connection(transaction=False) as conn:
        ids = [
            str(row["canonical_turn_id"])
            for row in conn.execute(
                f"""SELECT canonical_turn_id FROM canonical_turns
                     WHERE conversation_id = {p}
                       AND (COALESCE(LENGTH(user_raw_content), 0) >= {p}
                            OR COALESCE(LENGTH(assistant_raw_content), 0) >= {p})
                     ORDER BY sort_key""",
                (conversation_id, 1024, 1024),
            ).fetchall()
        ]
    seen_blobs: set[str] = set()
    for start in range(0, len(ids), batch_size):
        for canonical_turn_id in ids[start:start + batch_size]:
            with store._raw_blob_connection(transaction=False) as conn:
                row = conn.execute(
                    f"""SELECT user_raw_content, assistant_raw_content FROM canonical_turns
                         WHERE canonical_turn_id = {p}""",
                    (canonical_turn_id,),
                ).fetchone()
            if row is None:
                continue
            counts["rows_scanned"] += 1
            old_user, old_assistant = row["user_raw_content"], row["assistant_raw_content"]
            del row
            user, assistant = compact_raw_body(old_user), compact_raw_body(old_assistant)
            counts["bytes_before"] += len(old_user or "") + len(old_assistant or "")
            counts["bytes_after"] += len(user.text or "") + len(assistant.text or "")
            if (user.text, assistant.text) == (old_user, old_assistant):
                continue
            counts["rows_compacted"] += 1
            blobs = {**user.blobs, **assistant.blobs}
            seen_blobs.update(blobs)
            if dry_run:
                continue
            with store._raw_blob_connection(transaction=False) as conn:
                store_blobs(conn, dialect, blobs)
                store._raw_blob_commit(conn)
            unchanged, unchanged_params = _unchanged_bodies_sql(dialect, old_user, old_assistant)
            with store._raw_blob_connection() as conn:
                grant(conn, dialect, canonical_turn_id)
                conn.execute(
                    f"""UPDATE canonical_turns SET user_raw_content = {p}, assistant_raw_content = {p}
                         WHERE canonical_turn_id = {p} AND {unchanged}""",
                    (user.text, assistant.text, canonical_turn_id, *unchanged_params),
                )
                clear_grant(conn, dialect, canonical_turn_id)
                store._raw_blob_commit(conn)
    counts["blobs"] = len(seen_blobs)
    return counts


def _unchanged_bodies_sql(dialect: str, user: str | None, assistant: str | None) -> tuple[str, tuple]:
    """Predicate: a row's raw bodies are still *user* and *assistant*."""
    import hashlib

    if dialect == "postgres":
        digest = lambda value: hashlib.md5((value or "").encode()).hexdigest()  # noqa: E731
        return (
            "md5(COALESCE(user_raw_content, '')) = %s AND md5(COALESCE(assistant_raw_content, '')) = %s",
            (digest(user), digest(assistant)),
        )
    return (
        "COALESCE(user_raw_content, '') = ? AND COALESCE(assistant_raw_content, '') = ?",
        (user or "", assistant or ""),
    )
