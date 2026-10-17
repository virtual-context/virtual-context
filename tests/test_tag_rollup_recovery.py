"""Tag summaries lost to an interrupted compaction are rebuilt by the next one (BUG-126).

A compaction stores its segments before it builds their tag summaries. When
the process dies between the two (a restart, an OOM kill), the new segments'
tags are left with no ``tag_summaries`` row, and nothing revisits them: the
next compaction only rolls up tags carried by the segments it compacts itself.
Every compaction therefore also rebuilds summaries for any of the
conversation's segment tags that have none.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3

import pytest

from tests.test_tag_summary_materialization import _compacted, _tag_sets


def _drop_summaries(tmp_path, tags):
    conn = sqlite3.connect(tmp_path / "c.db")
    try:
        conn.executemany(
            "DELETE FROM tag_summaries WHERE conversation_id = 'c' AND tag = ?",
            [(t,) for t in tags],
        )
        conn.commit()
    finally:
        conn.close()


def _compact_more(engine, texts):
    from virtual_context.proxy.formats import detect_format, extract_ingestible_messages
    body = {"messages": []}
    rows = engine._store.get_all_canonical_turns("c")
    for row in sorted(rows, key=lambda r: (r.sort_key, r.canonical_turn_id)):
        if (row.user_content or "").strip():
            body["messages"].append({"role": "user", "content": row.user_content})
        if (row.assistant_content or "").strip():
            body["messages"].append({"role": "assistant", "content": row.assistant_content})
    for i, text in enumerate(texts):
        body["messages"] += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"answer {i}: {text}"},
        ]
    fmt = detect_format(body)
    engine._ingest_reconciler.ingest_batch(
        "c", body=body, fmt=fmt,
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )
    messages, _ = extract_ingestible_messages(body, fmt)
    engine.ingest_history(
        messages,
        require_existing_canonical=True,
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )
    return engine._compaction.compact_manual(messages)


@pytest.mark.regression("BUG-126")
def test_store_lists_segment_tags_without_summaries(tmp_path):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        segment_tags, _ = _tag_sets(tmp_path)
        assert {"perfume", "gifts"} <= segment_tags
        assert engine._store.get_segment_tags_without_summaries("c") == []
        _drop_summaries(tmp_path, ["perfume", "gifts"])
        assert engine._store.get_segment_tags_without_summaries("c") == ["gifts", "perfume"]
        assert engine._store.get_segment_tags_without_summaries("other") == []
    finally:
        engine.close()


@pytest.mark.regression("BUG-126")
def test_next_compaction_rebuilds_summaries_an_interrupted_one_lost(tmp_path):
    engine, compactor, _report = _compacted(tmp_path)
    try:
        _drop_summaries(tmp_path, ["perfume", "gifts"])
        report = _compact_more(engine, [
            "more water for the saxifrage cushion",
            "saxifrage water check in winter",
        ])
        assert report is not None and report.segments_compacted > 0
        asked = set(compactor.compact_tag_summaries.call_args.kwargs["cover_tags"])
        assert {"perfume", "gifts"} <= asked, sorted(asked)
        segment_tags, summary_tags = _tag_sets(tmp_path)
        assert not segment_tags - summary_tags, sorted(segment_tags - summary_tags)
    finally:
        engine.close()


@pytest.mark.regression("BUG-126")
def test_recovered_tags_get_turn_anchors_after_a_restart(tmp_path):
    """After a restart the in-memory turn index is empty (a takeover
    compaction): recovered tags still get their turns from stored rows."""
    engine, compactor, _report = _compacted(tmp_path)
    try:
        _drop_summaries(tmp_path, ["perfume", "gifts"])
        engine._compaction._turn_tag_index.entries.clear()
        _compact_more(engine, ["saxifrage water in winter again"])
        kwargs = compactor.compact_tag_summaries.call_args.kwargs
        assert {"perfume", "gifts"} <= set(kwargs["cover_tags"])
        assert kwargs["tag_to_turns"].get("perfume"), kwargs["tag_to_turns"]
        assert kwargs["tag_to_canonical_turn_ids"].get("gifts")
    finally:
        engine.close()


@pytest.mark.regression("BUG-126")
def test_recovered_tag_turns_are_not_double_counted(tmp_path):
    engine, compactor, _report = _compacted(tmp_path)
    try:
        _drop_summaries(tmp_path, ["perfume"])
        _compact_more(engine, ["saxifrage water in winter again"])
        kwargs = compactor.compact_tag_summaries.call_args.kwargs
        turns = kwargs["tag_to_turns"]["perfume"]
        assert len(turns) == len(set(turns)), turns
    finally:
        engine.close()
