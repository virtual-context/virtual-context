"""Pending compaction selection hydrates groups instead of the full archive."""

from dataclasses import asdict

import pytest

from virtual_context.storage.sqlite import SQLiteStore, _merge_canonical_turn_rows


@pytest.fixture
def store(tmp_path):
    store = SQLiteStore(tmp_path / "pending.db")
    yield store
    store.close()


def _pair(store, group, *, legacy=False):
    for index, (user, assistant) in enumerate([(f"user {group}", ""), ("", f"assistant {group}")]):
        store.save_canonical_turn(
            "owner", group * 2 + index, user, assistant,
            canonical_turn_id=f"g{group}-{index}", turn_group_number=group,
            sort_key=float(group * 2 + index + 1),
        )
    if legacy:
        store._get_conn().execute("UPDATE canonical_turns SET turn_group_number=-1 WHERE conversation_id='owner'")


def _old_result(store, protected):
    rows = _merge_canonical_turn_rows(store._load_canonical_turn_rows("owner"))
    result = [row for row in rows.values() if not row.compacted_at
              and row.user_content.strip() and row.assistant_content.strip()]
    return result[:-protected] if protected else result


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("protected", [0, 1, 3])
def test_pending_group_results_preserve_existing_merge_and_protected_tail(store, legacy, protected):
    for group in range(5):
        _pair(store, group)
    conn = store._get_conn()
    conn.execute("UPDATE canonical_turns SET compacted_at='done' WHERE turn_group_number IN (0,2)")
    conn.execute("UPDATE canonical_turns SET compacted_at='done' WHERE canonical_turn_id='g1-0'")
    store.save_canonical_turn("owner", 10, "orphan user", "", canonical_turn_id="orphan", turn_group_number=5, sort_key=11.0)
    if legacy:
        # Mixed legacy keeps the content-heuristic branch, including ordinals
        # contributed by fully compacted groups before a pending correction.
        conn.execute("UPDATE canonical_turns SET turn_group_number=-1 WHERE canonical_turn_id='g4-0'")
    expected = _old_result(store, protected)
    actual = store.get_uncompacted_canonical_turns("owner", protected_recent_turns=protected)
    assert [asdict(row) for row in actual] == [asdict(row) for row in expected]
    assert store.get_uncompacted_canonical_turns("owner", protected_recent_turns=protected, limit=1) == expected[:1]


def test_completed_archive_bodies_are_never_hydrated_and_split_sibling_is_retained(store, monkeypatch):
    for group in range(8):
        _pair(store, group)
    conn = store._get_conn()
    conn.execute("UPDATE canonical_turns SET compacted_at='done' WHERE turn_group_number<6")
    conn.execute("UPDATE canonical_turns SET compacted_at='done' WHERE canonical_turn_id='g6-0'")
    monkeypatch.setattr(store, "_load_canonical_turn_rows", lambda *_a: pytest.fail("archive body hydration"))
    decoded = []
    decoder = store._canonical_decoder()
    def observed(row):
        decoded.append(row["canonical_turn_id"])
        assert row["turn_group_number"] >= 6
        return decoder(row)
    monkeypatch.setattr(store, "_canonical_decoder", lambda: observed)
    result = store.get_uncompacted_canonical_turns("owner", protected_recent_turns=1)
    assert [row.turn_group_number for row in result] == [6]
    assert result[0].user_content == "user 6" and result[0].assistant_content == "assistant 6"
    assert set(decoded) == {"g6-0", "g6-1", "g7-0", "g7-1"}


def test_all_legacy_backfill_uses_scalar_rows_and_preserves_marking_siblings(store, monkeypatch):
    for group in range(3):
        _pair(store, group)
    store._get_conn().execute("UPDATE canonical_turns SET turn_group_number=-1")
    monkeypatch.setattr(store, "_load_canonical_turn_rows", lambda *_a: pytest.fail("archive hydration"))
    monkeypatch.setattr(store, "recompute_canonical_turn_groups", lambda *_a: pytest.fail("legacy projected body hydration"))
    rows = store.get_uncompacted_canonical_turns("owner", limit=1)
    assert rows[0].turn_group_number == 0
    assert store.mark_canonical_turns_compacted("owner", [rows[0].canonical_turn_id]) == 2
    assert [row.turn_group_number for row in store.get_uncompacted_canonical_turns("owner")] == [1, 2]


def test_unicode_blank_pairs_do_not_count_against_protected_recent_tail(store):
    for group in range(3):
        _pair(store, group)
    store._get_conn().execute("UPDATE canonical_turns SET user_content=? WHERE canonical_turn_id='g2-0'", ("\u00a0\u2003\t",))
    assert [row.turn_group_number for row in store.get_uncompacted_canonical_turns(
        "owner", protected_recent_turns=1,
    )] == [0]


@pytest.mark.parametrize("legacy", [False, True])
def test_logical_hydration_reads_only_requested_groups_with_original_ordinals(store, legacy, monkeypatch):
    for group in range(7):
        _pair(store, group)
    if legacy:
        store._get_conn().execute("UPDATE canonical_turns SET turn_group_number=-1 WHERE canonical_turn_id='g5-0'")
    expected = _merge_canonical_turn_rows(store._load_canonical_turn_rows("owner"))
    monkeypatch.setattr(store, "_load_canonical_turn_rows", lambda *_a: pytest.fail("logical archive hydration"))
    decoder = store._canonical_decoder()
    decoded = []
    def observed(row):
        decoded.append(row["canonical_turn_id"])
        return decoder(row)
    monkeypatch.setattr(store, "_canonical_decoder", lambda: observed)
    actual = store.get_canonical_turn_rows("owner", [2, 5])
    assert {number: asdict(row) for number, row in actual.items()} == {
        number: asdict(expected[number]) for number in (2, 5)
    }
    assert set(decoded) == {"g2-0", "g2-1", "g5-0", "g5-1"}
