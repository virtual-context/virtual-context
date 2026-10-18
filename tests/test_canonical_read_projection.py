"""Hot paths read only the canonical-turn columns they use (BUG-127).

Canonical rows carry the raw request and response bodies, which for a long
conversation run to hundreds of megabytes. Persisting each tagged turn,
refreshing the compaction watermark, and rolling up recovered tags all loaded
every row with those bodies although none of them reads text, so each turn
moved the size of the whole conversation into the process.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import inspect
import types

import pytest

from tests.test_tag_rollup_recovery import _compact_more, _drop_summaries
from tests.test_tag_summary_materialization import _compacted
from virtual_context.proxy.state import ProxyState
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Message


def _record_full_reads(monkeypatch, method="get_all_canonical_turns"):
    """Names of every function on the stack of each full-row read."""
    callers: list[set[str]] = []
    original = getattr(SQLiteStore, method)

    def spy(self, *args, **kwargs):
        callers.append({frame.function for frame in inspect.stack()[1:]})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SQLiteStore, method, spy)
    return callers


def _read_from(callers, function):
    return any(function in stack for stack in callers)


@pytest.mark.regression("BUG-127")
def test_tagging_a_turn_does_not_load_canonical_bodies(tmp_path, monkeypatch):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        callers = _record_full_reads(monkeypatch)
        history = [Message(role="user", content="saxifrage water in winter"),
                   Message(role="assistant", content="water it less in winter")]
        engine.tag_turn(history, payload_tokens=10)
        assert not _read_from(callers, "_persist_canonical_turn")
        assert any("alpine-gardening" in row.tags for row in engine._store.get_all_canonical_turns("c"))
    finally:
        engine.close()


@pytest.mark.regression("BUG-127")
def test_watermark_refresh_reads_scalar_metadata(tmp_path, monkeypatch):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        expected = engine._store.get_compaction_watermark("c")[0]
        assert expected > 0, "the seed must compact a prefix"
        engine._engine_state.compacted_prefix_messages = 0
        callers = _record_full_reads(monkeypatch)
        ProxyState._advance_compaction_watermark(types.SimpleNamespace(engine=engine))
        assert engine._engine_state.compacted_prefix_messages == expected
        assert not callers
    finally:
        engine.close()


@pytest.mark.regression("BUG-127")
def test_recovered_tag_rollup_reads_tags_without_bodies(tmp_path, monkeypatch):
    engine, compactor, _report = _compacted(tmp_path)
    try:
        _drop_summaries(tmp_path, ["perfume", "gifts"])
        engine._compaction._turn_tag_index.entries.clear()
        callers = _record_full_reads(monkeypatch)
        _compact_more(engine, ["saxifrage water in winter again"])
        assert not _read_from(callers, "_build_tag_summaries")
        kwargs = compactor.compact_tag_summaries.call_args.kwargs
        assert {"perfume", "gifts"} <= set(kwargs["cover_tags"])
        assert kwargs["tag_to_turns"].get("perfume")
        assert kwargs["tag_to_canonical_turn_ids"].get("gifts")
    finally:
        engine.close()


@pytest.mark.regression("BUG-127")
def test_canonical_turn_tags_match_the_full_rows(tmp_path):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        full = [
            (row.canonical_turn_id, row.turn_number, row.primary_tag, list(row.tags))
            for row in engine._store.get_all_canonical_turns("c")
        ]
        assert full
        from virtual_context.core.store import canonical_turn_tags
        assert canonical_turn_tags(engine._store, "c") == full
    finally:
        engine.close()


# Restarts, takeovers and ingestion also read only the turns they act on
# (BUG-129): a multi-year conversation must not cost its whole history to
# resume ingestion, take over a compaction, or ingest a batch.

@pytest.mark.regression("BUG-129")
def test_group_range_and_markers_match_the_full_read(tmp_path):
    from virtual_context.core.store import ContextStore
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        store = engine._store
        full = store.get_all_canonical_turns("c")
        groups = sorted({int(row.turn_group_number) for row in full})
        assert len(groups) >= 4
        lo, hi = groups[1], groups[-2]
        ranged = store.get_canonical_turn_rows_in_groups("c", min_turn_group=lo, max_turn_group=hi)
        assert [r.canonical_turn_id for r in ranged] == [
            r.canonical_turn_id for r in full if lo <= int(r.turn_group_number) <= hi
        ]
        assert ranged[0].user_raw_content == next(
            r for r in full if r.canonical_turn_id == ranged[0].canonical_turn_id
        ).user_raw_content
        assert store.first_untagged_complete_group("c") == ContextStore.first_untagged_complete_group(store, "c")
        for turn in (0, groups[2], groups[-1] + 1):
            assert store.latest_session_date_before("c", turn) == ContextStore.latest_session_date_before(store, "c", turn)
    finally:
        engine.close()


@pytest.mark.regression("BUG-129")
def test_strict_history_ingest_reads_only_its_batch(tmp_path, monkeypatch):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        callers = _record_full_reads(monkeypatch)
        _compact_more(engine, ["saxifrage water in winter again"])
        assert not _read_from(callers, "ingest_history")
    finally:
        engine.close()


@pytest.mark.regression("BUG-129")
def test_takeover_history_holds_only_uncompacted_turns(tmp_path, monkeypatch):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        uncompacted = engine._store.get_uncompacted_canonical_turns("c")
        callers = _record_full_reads(monkeypatch)
        history, uncompacted_only = ProxyState._uncompacted_history(
            types.SimpleNamespace(engine=engine), "c",
        )
        assert uncompacted_only
        assert len(history) == 2 * len(uncompacted)
        assert not callers
    finally:
        engine.close()
