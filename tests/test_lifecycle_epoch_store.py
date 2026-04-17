import pytest
from pathlib import Path
from virtual_context.storage.sqlite import SQLiteStore


def test_upsert_conversation_creates_row(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT tenant_id, lifecycle_epoch, phase FROM conversations WHERE conversation_id='c'"
        ).fetchone()
    assert row is not None
    assert row[0] == "t"
    assert row[1] == 1  # default epoch
    assert row[2] == "init"  # default phase


def test_upsert_conversation_is_idempotent(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.upsert_conversation(tenant_id="t", conversation_id="c")  # no error
    with s._get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE conversation_id='c'"
        ).fetchone()[0]
    assert count == 1


def test_get_lifecycle_epoch_returns_current(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    assert s.get_lifecycle_epoch("c") == 1


def test_get_lifecycle_epoch_raises_keyerror_for_unknown(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    with pytest.raises(KeyError):
        s.get_lifecycle_epoch("nonexistent")


def test_mark_conversation_deleted_sets_phase_and_deleted_at(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.mark_conversation_deleted("c")
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT phase, deleted_at FROM conversations WHERE conversation_id='c'"
        ).fetchone()
    assert row[0] == "deleted"
    assert row[1] is not None  # deleted_at is set


def test_increment_lifecycle_epoch_bumps_on_resurrect(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.mark_conversation_deleted("c")
    new_epoch = s.increment_lifecycle_epoch_on_resurrect("c")
    assert new_epoch == 2
    assert s.get_lifecycle_epoch("c") == 2
    with s._get_conn() as conn:
        row = conn.execute(
            "SELECT phase, deleted_at FROM conversations WHERE conversation_id='c'"
        ).fetchone()
    assert row[0] == "init"
    assert row[1] is None  # deleted_at cleared


def test_concurrent_resurrect_does_not_double_bump(tmp_path: Path):
    """Second resurrect on an already-init conversation is a no-op; returns
    the current epoch without incrementing again (TOCTOU-safe guard)."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    s.mark_conversation_deleted("c")
    e1 = s.increment_lifecycle_epoch_on_resurrect("c")  # -> 2
    e2 = s.increment_lifecycle_epoch_on_resurrect("c")  # second call: phase is now 'init', not 'deleted'
    assert e1 == 2
    assert e2 == 2  # NOT 3


def test_increment_on_never_deleted_conversation_is_noop(tmp_path: Path):
    """Calling resurrect on a conversation that was never deleted should not bump."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    e = s.increment_lifecycle_epoch_on_resurrect("c")
    assert e == 1  # unchanged


def test_increment_raises_keyerror_for_unknown(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    with pytest.raises(KeyError):
        s.increment_lifecycle_epoch_on_resurrect("nonexistent")
