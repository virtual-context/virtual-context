from pathlib import Path
from virtual_context.storage.sqlite import SQLiteStore


def test_canonical_turns_has_covered_and_tagged_columns(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    with store._get_conn() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(canonical_turns)")}
    assert "covered_ingestible_entries" in cols
    assert "tagged_at" in cols


def test_canonical_turns_untagged_index_exists(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    with store._get_conn() as conn:
        names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='canonical_turns'"
            )
        }
    assert any("untagged" in name for name in names), f"Got indexes: {names}"


def test_canonical_turns_tagged_index_exists(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    with store._get_conn() as conn:
        names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='canonical_turns'"
            )
        }
    assert any("tagged" in name and "untagged" not in name for name in names), f"Got indexes: {names}"
