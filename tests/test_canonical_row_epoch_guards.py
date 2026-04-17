"""Tests for epoch-guarded ``iter_untagged_canonical_rows`` +
``mark_canonical_row_tagged`` on the SQLite backend.

Both methods scope their SQL to ``conversations.lifecycle_epoch`` so a stale
caller (whose in-memory epoch no longer matches the authoritative row) sees
zero rows on iter and a ``False`` return on mark. This keeps concurrent
delete/resurrect workflows from letting a background tagger touch rows that
belong to a new lifecycle.
"""

from __future__ import annotations

from pathlib import Path

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.core.canonical_turns import utcnow_iso


def _seed_canonical_row(
    s: SQLiteStore,
    *,
    conv_id: str,
    canonical_id: str,
    sort_key: float,
    tagged: bool = False,
) -> None:
    now = utcnow_iso()
    with s._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, ?, ?, ?)
            """,
            (
                canonical_id,
                conv_id,
                f"h_{canonical_id}",
                sort_key,
                now,
                now,
                now if tagged else None,
                now,
                now,
            ),
        )


def test_iter_untagged_returns_untagged_in_sort_key_order(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_canonical_row(s, conv_id="c", canonical_id="t0", sort_key=3000.0)
    _seed_canonical_row(s, conv_id="c", canonical_id="t1", sort_key=1000.0)
    _seed_canonical_row(s, conv_id="c", canonical_id="t2", sort_key=2000.0)
    rows = s.iter_untagged_canonical_rows(
        conversation_id="c", expected_lifecycle_epoch=1, batch_size=10,
    )
    ids_in_order = [r.canonical_turn_id for r in rows]
    assert ids_in_order == ["t1", "t2", "t0"]  # sorted by sort_key


def test_iter_untagged_respects_batch_size(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    for i in range(5):
        _seed_canonical_row(s, conv_id="c", canonical_id=f"t{i}", sort_key=float(i * 1000))
    rows = s.iter_untagged_canonical_rows(
        conversation_id="c", expected_lifecycle_epoch=1, batch_size=2,
    )
    assert len(rows) == 2


def test_iter_untagged_excludes_tagged_rows(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_canonical_row(s, conv_id="c", canonical_id="t_tagged", sort_key=1000.0, tagged=True)
    _seed_canonical_row(s, conv_id="c", canonical_id="t_pending", sort_key=2000.0, tagged=False)
    rows = s.iter_untagged_canonical_rows(
        conversation_id="c", expected_lifecycle_epoch=1, batch_size=10,
    )
    assert [r.canonical_turn_id for r in rows] == ["t_pending"]


def test_iter_untagged_returns_empty_on_epoch_mismatch(tmp_path: Path) -> None:
    """SQL-level epoch guard. Stale-epoch caller sees no rows even if rows exist."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_canonical_row(s, conv_id="c", canonical_id="t0", sort_key=1000.0)
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")  # epoch=2
    # Seed a new-lifecycle row.
    _seed_canonical_row(s, conv_id="c", canonical_id="t_new", sort_key=2000.0)
    # Stale caller: epoch=1.
    rows = s.iter_untagged_canonical_rows(
        conversation_id="c", expected_lifecycle_epoch=1, batch_size=10,
    )
    assert rows == []  # sees nothing
    # Current caller: epoch=2 — sees both (t0 from old lifecycle not purged, plus t_new).
    rows = s.iter_untagged_canonical_rows(
        conversation_id="c", expected_lifecycle_epoch=2, batch_size=10,
    )
    assert len(rows) == 2


def test_mark_canonical_row_tagged_sets_timestamp(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_canonical_row(s, conv_id="c", canonical_id="t_pending", sort_key=1000.0)
    ok = s.mark_canonical_row_tagged(
        canonical_turn_id="t_pending", conversation_id="c",
        expected_lifecycle_epoch=1,
    )
    assert ok is True
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT tagged_at FROM canonical_turns WHERE canonical_turn_id='t_pending'"
        ).fetchone()
    assert row[0] is not None


def test_mark_canonical_row_tagged_rejects_on_epoch_mismatch(tmp_path: Path) -> None:
    """Proves SQL-level epoch guard on the mark boundary."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_canonical_row(s, conv_id="c", canonical_id="t0", sort_key=1000.0)
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")  # epoch=2
    # Stale caller tries to mark a row that still exists.
    ok = s.mark_canonical_row_tagged(
        canonical_turn_id="t0", conversation_id="c",
        expected_lifecycle_epoch=1,
    )
    assert ok is False
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT tagged_at FROM canonical_turns WHERE canonical_turn_id='t0'"
        ).fetchone()
    assert row[0] is None  # untouched


def test_mark_canonical_row_tagged_is_idempotent_for_already_tagged(tmp_path: Path) -> None:
    """Calling mark_canonical_row_tagged a second time on an already-tagged row returns False (no-op)."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    _seed_canonical_row(s, conv_id="c", canonical_id="t0", sort_key=1000.0, tagged=True)
    ok = s.mark_canonical_row_tagged(
        canonical_turn_id="t0", conversation_id="c",
        expected_lifecycle_epoch=1,
    )
    assert ok is False  # already tagged, WHERE tagged_at IS NULL excludes it


def test_mark_canonical_row_tagged_wrong_conversation_id_returns_false(tmp_path: Path) -> None:
    """Belt-and-suspenders: mismatched conversation_id doesn't accidentally mark another conversation's row."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c1")
    s.upsert_conversation(tenant_id="t", conversation_id="c2")
    _seed_canonical_row(s, conv_id="c1", canonical_id="t0", sort_key=1000.0)
    ok = s.mark_canonical_row_tagged(
        canonical_turn_id="t0", conversation_id="c2",  # wrong conv
        expected_lifecycle_epoch=1,
    )
    assert ok is False
