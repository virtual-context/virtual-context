"""Reconciliation owns its SQLite transaction across all ingest helpers."""

import sqlite3
from types import SimpleNamespace

import pytest

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.proxy.formats import detect_format
from virtual_context.storage.sqlite import SQLiteStore


@pytest.fixture
def store(tmp_path):
    store = SQLiteStore(tmp_path / "reconcile.db")
    store.upsert_conversation(tenant_id="", conversation_id="owner")
    yield store
    store.close()


@pytest.mark.parametrize("helper", [
    "epoch", "groups", "sender", "channel", "actor", "reply", "sort_keys",
])
def test_ingest_helpers_preserve_writer_lock_and_outer_rollback(store, helper):
    store.save_canonical_turn(
        "owner", 0, "Source text", "", canonical_turn_id="physical-source",
    )
    conn = store._get_conn()
    before = tuple(conn.execute("SELECT * FROM canonical_turns").fetchone())
    actions = {
        "epoch": lambda: store.get_lifecycle_epoch("owner"),
        "groups": lambda: store.recompute_canonical_turn_groups("owner"),
        "sender": lambda: store.update_canonical_turn_senders_if_empty(
            "owner", {"physical-source": "Alice"}, expected_lifecycle_epoch=1,
        ),
        "channel": lambda: store.update_canonical_turn_channels_if_empty(
            "owner", {"physical-source": ("channel", "Room")},
            expected_lifecycle_epoch=1,
        ),
        "actor": lambda: store.update_canonical_turn_actors_if_empty(
            "owner", {"physical-source": "actor"}, expected_lifecycle_epoch=1,
        ),
        "reply": lambda: store.update_canonical_turn_reply_roles_if_empty(
            "owner", {"physical-source": {"source_message_id": "message"}},
            expected_lifecycle_epoch=1,
        ),
        "sort_keys": lambda: store.shift_canonical_turn_sort_keys(
            "owner", min_sort_key=1000, delta=2000,
        ),
    }
    competitor = sqlite3.connect(str(store.db_path), isolation_level=None, timeout=0)
    try:
        with pytest.raises(RuntimeError, match="abort after helper"):
            with store.conversation_reconcile("owner"):
                actions[helper]()
                # This is another actual SQLite connection, not a Python lock
                # double. A helper commit would let its writer proceed.
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    competitor.execute("BEGIN IMMEDIATE")
                assert conn.in_transaction
                raise RuntimeError("abort after helper")
        assert tuple(conn.execute("SELECT * FROM canonical_turns").fetchone()) == before
        assert not conn.in_transaction
    finally:
        competitor.rollback()
        competitor.close()


def test_failed_ingest_rolls_back_rows_groups_batch_and_anchors(store, monkeypatch):
    body = {"model": "gpt-4o", "messages": [
        {"role": "user", "content": "First source"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second source"},
        {"role": "assistant", "content": "Second response"},
    ]}
    semantic = SimpleNamespace(embed_and_store_turn=lambda *args, **kwargs: None)
    reconciler = IngestReconciler(store, semantic)

    def fail_after_writes(*args, **kwargs):
        assert store._get_conn().execute(
            "SELECT COUNT(*) FROM canonical_turns WHERE turn_group_number >= 0",
        ).fetchone()[0] == 4
        raise RuntimeError("abort after grouped ingest")

    monkeypatch.setattr(reconciler, "_log_ingest_breakdown", fail_after_writes)
    with pytest.raises(RuntimeError, match="abort after grouped ingest"):
        reconciler.ingest_batch(
            "owner", body=body, fmt=detect_format(body), expected_lifecycle_epoch=1,
        )
    for table in ("canonical_turns", "ingest_batches", "canonical_turn_anchors"):
        assert store._get_conn().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert not store._get_conn().in_transaction
    assert not store._reconcile_lock_active()
