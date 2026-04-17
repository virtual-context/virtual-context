"""Tests for the conversations table schema (lifecycle_epoch, phase, counters)."""

from pathlib import Path

from virtual_context.storage.sqlite import SQLiteStore


def test_conversations_table_has_phase_and_epoch_columns(tmp_path: Path):
    store = SQLiteStore(tmp_path / "vc.db")
    conn = store._get_conn()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
    expected = {
        "conversation_id", "tenant_id", "lifecycle_epoch", "phase",
        "pending_raw_payload_entries", "last_raw_payload_entries",
        "last_ingestible_payload_entries",
        "created_at", "updated_at", "deleted_at",
    }
    missing = expected - cols
    assert not missing, f"Missing columns: {missing}"
