"""Per-request reads leave the raw request bodies in storage (BUG-130).

A raw body holds whatever the host sent with the turn, including base64
images and the host's own conversation context, and runs to megabytes per
row. The audience check, the protected window and the other recent-turn
readers use only text and audience fields, yet each loaded every selected
row's raw bodies on every request.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sqlite3
import types

import pytest

from tests.test_tag_summary_materialization import _compacted
from virtual_context.core.semantic_search import _user_embedding_text
from virtual_context.core.summary_identity import summaries_admitted_for_audience
from virtual_context.storage.sqlite import SQLiteStore

_IMAGE = "data:image/png;base64," + "iVBORw0KGgo" * 2000


def _give_rows_raw_bodies(tmp_path):
    conn = sqlite3.connect(tmp_path / "c.db")
    try:
        raw = json.dumps([
            {"type": "input_text", "text": "look at this"},
            {"type": "input_image", "image_url": _IMAGE},
        ])
        conn.execute(
            "UPDATE canonical_turns SET user_raw_content = ?, assistant_raw_content = ?",
            (raw, json.dumps("reply")),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.regression("BUG-130")
def test_recent_turns_carry_text_without_raw_bodies(tmp_path):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        _give_rows_raw_bodies(tmp_path)
        rows = engine._store.get_recent_canonical_turns("c", limit=3)
        assert rows
        assert all((row.user_content or row.assistant_content) for row in rows)
        assert all(row.user_raw_content is None and row.assistant_raw_content is None for row in rows)
    finally:
        engine.close()


@pytest.mark.regression("BUG-130")
def test_audience_check_reads_only_audience_fields(tmp_path, monkeypatch):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        store = engine._store
        full = store.get_all_canonical_turns("c")
        foreign, own = full[0], full[-1]
        conn = sqlite3.connect(tmp_path / "c.db")
        conn.execute(
            "UPDATE canonical_turns SET audience_conversation_id = 'other-guild', "
            "audience_attribution_version = 1 WHERE canonical_turn_id = ?",
            (foreign.canonical_turn_id,),
        )
        conn.commit()
        conn.close()

        def no_full_rows(self, *args, **kwargs):
            raise AssertionError("audience check loaded full canonical rows")

        monkeypatch.setattr(SQLiteStore, "get_canonical_turn_rows_by_id", no_full_rows)
        items = [
            types.SimpleNamespace(source_canonical_turn_ids=[foreign.canonical_turn_id]),
            types.SimpleNamespace(source_canonical_turn_ids=[own.canonical_turn_id]),
        ]
        admitted = summaries_admitted_for_audience(
            items, store=store, conversation_id="c", speaker_context=None,
        )
        assert admitted == [False, True]
    finally:
        engine.close()


@pytest.mark.regression("BUG-130")
def test_audiences_match_the_full_rows(tmp_path):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        store = engine._store
        full = store.get_all_canonical_turns("c")
        keys = [("c", row.canonical_turn_id) for row in full] + [("c", "missing")]
        audiences = store.get_canonical_turn_audiences(keys)
        assert audiences == {
            ("c", row.canonical_turn_id): (
                row.audience_conversation_id, row.audience_attribution_version,
            )
            for row in full
        }
    finally:
        engine.close()


@pytest.mark.regression("BUG-130")
def test_attachment_only_turn_embeds_text_parts_not_image_data():
    raw = json.dumps([
        {"type": "input_text", "text": "the red one"},
        {"type": "input_image", "image_url": _IMAGE},
        {"type": "image", "source": {"type": "base64", "data": "QUJD" * 500}},
    ])
    assert _user_embedding_text("", raw) == "the red one"
    attachment_only = _user_embedding_text("", json.dumps([{"type": "input_image", "image_url": _IMAGE}]))
    assert attachment_only == '[{"type":"input_image"}]'
    assert _user_embedding_text("", "plain text body") == "plain text body"


@pytest.mark.regression("BUG-130")
def test_restore_reads_raw_bodies_only_for_uncompacted_turns(tmp_path, monkeypatch):
    import inspect
    import virtual_context.storage.sqlite as sqlite_mod
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        _give_rows_raw_bodies(tmp_path)
        to_row = sqlite_mod._row_to_canonical_turn
        raw_readers: list[set[str]] = []

        def recording(row):
            parsed = to_row(row)
            if parsed.user_raw_content:
                raw_readers.append({frame.function for frame in inspect.stack()[1:]})
            return parsed

        monkeypatch.setattr(sqlite_mod, "_row_to_canonical_turn", recording)
        assert engine._restore_from_canonical_rows("c")
        assert raw_readers
        assert all("get_uncompacted_canonical_turns" in stack for stack in raw_readers)
        assert engine._turn_tag_index.entries
        assert all(turn[3] for turn in engine._restored_pending_turns)
    finally:
        engine.close()


@pytest.mark.regression("BUG-130")
def test_context_hint_reads_topic_names_not_summaries(tmp_path, monkeypatch):
    from virtual_context.core.hint_builder import build_default_hint
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        store = engine._store
        summaries = store.get_all_tag_summaries(conversation_id="c")
        assert summaries
        assert [(t.tag, t.turn_count) for t in store.get_tag_summary_topics("c")] == [
            (ts.tag, len(ts.source_turn_numbers)) for ts in summaries
        ]
        retrieval = engine._retrieval
        expected = build_default_hint(
            tag_summaries=summaries,
            max_hint_tokens=engine.config.assembler.context_hint_max_tokens,
            token_counter=retrieval._token_counter,
        )

        def no_full_summaries(self, *args, **kwargs):
            raise AssertionError("context hint loaded full tag summaries")

        monkeypatch.setattr(SQLiteStore, "get_all_tag_summaries", no_full_summaries)
        hint = retrieval._build_context_hint(paging_mode=None, force_rebuild=True)
        assert hint and hint == expected
    finally:
        engine.close()
