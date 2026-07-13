"""The canonical-turn write path indexes the reply-target lane.

``_write_turn`` forwards the row's ``reply_target_body`` into
``embed_and_store_turn`` so a ``subject``-side chunk is written under the
same physical row on every new ingest. The lane is shadow data: the
shipped search branch explicitly ignores ``side="subject"`` chunks.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from unittest.mock import MagicMock

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CanonicalTurnRow


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    return store


def test_write_turn_forwards_the_reply_body_to_embedding(tmp_path: Path):
    store = _store(tmp_path)
    semantic = MagicMock()
    reconciler = IngestReconciler(store, semantic)
    row = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="ct-1",
        sort_key=1000.0,
        turn_hash="h-1",
        user_content="what about it?",
        reply_target_body="the boston trip was amazing",
    )
    reconciler._write_turn(row, turn_number=0)
    kwargs = semantic.embed_and_store_turn.call_args.kwargs
    assert kwargs["reply_target_body"] == "the boston trip was amazing"


def test_write_turn_defaults_an_absent_reply_body_to_empty(tmp_path: Path):
    store = _store(tmp_path)
    semantic = MagicMock()
    reconciler = IngestReconciler(store, semantic)
    row = CanonicalTurnRow(
        conversation_id="c",
        canonical_turn_id="ct-2",
        sort_key=2000.0,
        turn_hash="h-2",
        user_content="plain turn",
    )
    reconciler._write_turn(row, turn_number=0)
    kwargs = semantic.embed_and_store_turn.call_args.kwargs
    assert kwargs["reply_target_body"] == ""
