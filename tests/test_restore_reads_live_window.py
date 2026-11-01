"""Restoring an engine reads turn text only past the compaction watermark (BUG-143).

The turn index needs a hash of every turn's text, so the restore loaded the
full text of every canonical turn the conversation ever had. On a long
conversation that is tens of megabytes per worker per cold load, and the
allocator keeps most of it after the rows are dropped. The hashes now come
from the store; only the live and pending turns' text is read.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib

import pytest

from tests.test_tag_summary_materialization import _compacted
from virtual_context.engine import VirtualContextEngine
from virtual_context.storage.sqlite import SQLiteStore


def _expected_hashes(engine, conversation_id):
    rows = engine._store.get_all_canonical_turns(conversation_id)
    pairs = VirtualContextEngine._group_canonical_rows_into_pairs(rows)
    out = {}
    for turn_number, pair_rows in pairs:
        if not all(getattr(row, "tagged_at", None) for row in pair_rows):
            continue
        user, assistant, *_ = VirtualContextEngine._pair_payload_from_rows(pair_rows)
        out[turn_number] = hashlib.sha256(f"{user} {assistant}".encode()).hexdigest()[:16]
    return out


def _index_view(engine):
    return [
        (e.turn_number, e.canonical_turn_id, e.message_hash, e.tags, e.primary_tag,
         e.session_date, e.sender, e.timestamp, len(e.fact_signals), e.code_refs)
        for e in engine._turn_tag_index.entries
    ]


@pytest.mark.regression("BUG-143")
def test_restore_never_loads_compacted_turn_text(tmp_path, monkeypatch):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        store = engine._store
        compacted_ids = {
            row.canonical_turn_id
            for row in store.get_all_canonical_turns("c") if row.compacted_at
        }
        assert compacted_ids

        def no_whole_history(self, *args, **kwargs):
            raise AssertionError("restore loaded every canonical turn's text")

        monkeypatch.setattr(SQLiteStore, "get_all_canonical_turn_texts", no_whole_history)
        monkeypatch.setattr(SQLiteStore, "get_all_canonical_turns", no_whole_history)
        assert engine._restore_from_canonical_rows("c")
        assert engine._turn_tag_index.entries
    finally:
        engine.close()


@pytest.mark.regression("BUG-143")
def test_restore_from_projection_matches_full_rows(tmp_path, monkeypatch):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        expected = _expected_hashes(engine, "c")
        assert engine._restore_from_canonical_rows("c")
        got = {e.turn_number: e.message_hash for e in engine._turn_tag_index.entries}
        assert got == expected
        state = engine._engine_state
        watermark = (state.compacted_prefix_messages, state.last_compacted_turn,
                     state.last_completed_turn, state.last_indexed_turn)
        history = list(engine._restored_conversation_history)
        pending = list(engine._restored_pending_turns)
        index = _index_view(engine)

        # Same restore through the text loader the projection replaces.
        monkeypatch.setattr(SQLiteStore, "get_canonical_turn_index_rows", lambda self, cid: None)
        engine._engine_state.last_completed_turn = -1
        engine._engine_state.last_indexed_turn = -1
        assert engine._restore_from_canonical_rows("c")
        state = engine._engine_state
        assert (state.compacted_prefix_messages, state.last_compacted_turn,
                state.last_completed_turn, state.last_indexed_turn) == watermark
        assert engine._restored_conversation_history == history
        assert engine._restored_pending_turns == pending
        assert _index_view(engine) == index
    finally:
        engine.close()


@pytest.mark.regression("BUG-143")
def test_projection_hash_matches_python_for_non_ascii_and_split_halves(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "p.db"))
    try:
        store.activate_conversation("p")
        # group 0: one row with both halves; group 1: user and assistant on
        # separate rows, with an empty-text row before each.
        store.save_canonical_turn("p", -1, "naïve café — 東京 ✓", "réponse ü",
                                  canonical_turn_id="t0", sort_key=1000.0, turn_hash="h0",
                                  turn_group_number=0)
        store.save_canonical_turn("p", -1, "", "", canonical_turn_id="t1", sort_key=2000.0,
                                  turn_hash="h1", turn_group_number=1)
        store.save_canonical_turn("p", -1, "second user", "", canonical_turn_id="t2",
                                  sort_key=3000.0, turn_hash="h2", turn_group_number=1)
        store.save_canonical_turn("p", -1, "", "second answer", canonical_turn_id="t3",
                                  sort_key=4000.0, turn_hash="h3", turn_group_number=1)
        rows = store.get_canonical_turn_index_rows("p")
        assert rows is not None
        by_group = {row.turn_group_number: row.pair_hash for row in rows}
        h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]  # noqa: E731
        assert by_group == {0: h("naïve café — 東京 ✓ réponse ü"), 1: h("second user second answer")}
        assert [row.canonical_turn_id for row in rows] == ["t0", "t1", "t2", "t3"]
        assert [(row.has_user_content, row.has_assistant_content) for row in rows] == [
            (True, True), (False, False), (True, False), (False, True),
        ]
    finally:
        store.close()
