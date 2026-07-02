"""Strict payload tagging vs. concurrently-tagged canonical rows (BUG-038).

The strict-mode precondition in ``ingest_history`` used to require every
payload entry to map to an UNTAGGED canonical row. Rows legitimately
tagged between the prepare and the follow-up tagging pass — by another
worker's row sweep or by a prior pass over an overlapping payload —
broke the precondition: ``RuntimeError: strict canonical tagging
expected at least N covered ingestible entries, found M across K rows``.
The whole batch then fell through to the row-based DB sweep, losing
payload-context tagging for the rows that were still untagged and
firing alarm-grade log noise.

Fix: the precondition counts coverage over the conversation's row tail
INCLUDING tagged rows, and the pair walker hydrates TurnTagIndex
entries for fully-tagged pairs straight from the stored row tags —
consuming the strict cursor without invoking the tag generator.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timezone
from pathlib import Path

import pytest

from virtual_context.proxy.formats import detect_format, extract_ingestible_messages


def _make_engine(tmp_path: Path, conversation_id: str = "c"):
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.types import (
        StorageConfig,
        TagGeneratorConfig,
        VirtualContextConfig,
    )
    config = VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / f"{conversation_id}.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    return VirtualContextEngine(config=config)


def _inner_store(engine):
    store = engine._store
    inner = getattr(store, "_store", None)
    if inner is None:
        return store
    segments = getattr(inner, "_segments", None)
    if segments is not None:
        return segments
    return inner


def _pairs_body(n_pairs: int) -> dict:
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"user message {i} about topic alpha"})
        msgs.append({"role": "assistant", "content": f"assistant reply {i} about topic alpha"})
    return {"messages": msgs}


def _prepare_rows(engine, body: dict) -> None:
    """Persist canonical rows the way handle_prepare_payload does."""
    fmt = detect_format(body)
    engine._ingest_reconciler.ingest_batch(
        engine.config.conversation_id,
        body=body,
        fmt=fmt,
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )


def _tag_first_rows(engine, count: int, tags: str = '["stored-tag"]',
                    primary: str = "stored-tag") -> None:
    """Simulate another worker's sweep tagging the first ``count`` rows."""
    conv_id = engine.config.conversation_id
    inner = _inner_store(engine)
    conn = inner._get_conn()
    ids = [
        r["canonical_turn_id"]
        for r in conn.execute(
            "SELECT canonical_turn_id FROM canonical_turns "
            "WHERE conversation_id = ? ORDER BY sort_key",
            (conv_id,),
        ).fetchall()
    ]
    now = datetime.now(timezone.utc).isoformat()
    for ctid in ids[:count]:
        conn.execute(
            "UPDATE canonical_turns SET tagged_at = ?, tags_json = ?, "
            "primary_tag = ? WHERE canonical_turn_id = ?",
            (now, tags, primary, ctid),
        )
    conn.commit()


def _strict_ingest(engine, body: dict) -> int:
    messages, _ = extract_ingestible_messages(body, detect_format(body))
    return engine.ingest_history(
        messages,
        require_existing_canonical=True,
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )


@pytest.mark.regression("BUG-038")
def test_strict_ingest_survives_concurrently_tagged_rows(tmp_path):
    """The 2026-07-01 prod signature: 4-pair payload, 5 of 8 rows already
    tagged. Strict ingest must succeed, not raise the expected-entries
    RuntimeError."""
    engine = _make_engine(tmp_path)
    try:
        body = _pairs_body(4)
        _prepare_rows(engine, body)
        _tag_first_rows(engine, 5)
        ingested = _strict_ingest(engine, body)
        assert ingested == 4
        assert len(engine._turn_tag_index.entries) == 4
    finally:
        engine.close()


@pytest.mark.regression("BUG-038")
def test_hydrated_entries_carry_stored_tags_without_tagger_calls(tmp_path):
    """Fully-tagged pairs hydrate their index entries from stored row tags
    and never reach the tag generator."""
    engine = _make_engine(tmp_path)
    try:
        body = _pairs_body(3)
        _prepare_rows(engine, body)
        _tag_first_rows(engine, 4, tags='["alpha-topic"]', primary="alpha-topic")

        calls = []
        original = engine._tag_generator.generate_tags

        def _counting(*args, **kwargs):
            calls.append(args[0] if args else "")
            return original(*args, **kwargs)

        engine._tag_generator.generate_tags = _counting
        ingested = _strict_ingest(engine, body)
        assert ingested == 3

        entries = engine._turn_tag_index.entries
        # Pairs 0 and 1 (rows 0-3) were pre-tagged: hydrated.
        assert entries[0].primary_tag == "alpha-topic"
        assert entries[0].tags == ["alpha-topic"]
        assert entries[1].primary_tag == "alpha-topic"
        # The tagger only ever saw the untagged pair's content — hydrated
        # pairs never reach it. (The normal path may call the tagger more
        # than once per pair; the invariant is WHICH pairs reach it.)
        assert calls, "the untagged pair must go through the real tagger"
        for text in calls:
            assert "message 2" in text, f"hydrated pair leaked to tagger: {text[:60]}"
    finally:
        engine.close()


@pytest.mark.regression("BUG-038")
def test_untagged_tail_rows_still_get_tagged(tmp_path):
    """Rows that were NOT pre-tagged must end the strict pass tagged."""
    engine = _make_engine(tmp_path)
    try:
        body = _pairs_body(4)
        _prepare_rows(engine, body)
        _tag_first_rows(engine, 5)
        _strict_ingest(engine, body)
        conv_id = engine.config.conversation_id
        rows = engine._store.get_all_canonical_turns(conv_id)
        assert len(rows) == 8, "strict pass must not append new rows"
        untagged = [r for r in rows if not r.tagged_at]
        assert untagged == []
    finally:
        engine.close()


@pytest.mark.regression("BUG-038")
def test_fully_tagged_conversation_hydrates_with_zero_tagger_calls(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        body = _pairs_body(3)
        _prepare_rows(engine, body)
        _tag_first_rows(engine, 6, tags='["done-topic"]', primary="done-topic")

        calls = []
        original = engine._tag_generator.generate_tags

        def _counting(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        engine._tag_generator.generate_tags = _counting
        ingested = _strict_ingest(engine, body)
        assert ingested == 3
        assert calls == []
        assert [e.primary_tag for e in engine._turn_tag_index.entries] == [
            "done-topic", "done-topic", "done-topic",
        ]
    finally:
        engine.close()


@pytest.mark.regression("BUG-038")
def test_missing_rows_still_raise(tmp_path):
    """Genuinely absent rows (prepare never persisted them) must still be
    a strict-mode error — the precondition survives for real corruption."""
    engine = _make_engine(tmp_path)
    try:
        body = _pairs_body(2)
        # No _prepare_rows: nothing persisted.
        with pytest.raises(RuntimeError, match="strict canonical tagging"):
            _strict_ingest(engine, body)
    finally:
        engine.close()


@pytest.mark.regression("BUG-038")
def test_stale_epoch_still_raises(tmp_path):
    """A stale lifecycle epoch must keep failing the strict pass."""
    engine = _make_engine(tmp_path)
    try:
        body = _pairs_body(2)
        _prepare_rows(engine, body)
        conv_id = engine.config.conversation_id
        inner = _inner_store(engine)
        inner.mark_conversation_deleted(conv_id)
        inner.increment_lifecycle_epoch_on_resurrect(conv_id)
        messages, _ = extract_ingestible_messages(body, detect_format(body))
        with pytest.raises(RuntimeError, match="strict canonical tagging"):
            engine.ingest_history(
                messages,
                require_existing_canonical=True,
                expected_lifecycle_epoch=1,  # stale: resurrect bumped it
            )
    finally:
        engine.close()


@pytest.mark.regression("BUG-038")
def test_half_tagged_pair_falls_through_to_tagger(tmp_path):
    """A pair with one tagged and one untagged row is NOT hydrated — the
    normal tagger path re-tags both halves (idempotent)."""
    engine = _make_engine(tmp_path)
    try:
        body = _pairs_body(2)
        _prepare_rows(engine, body)
        _tag_first_rows(engine, 1, tags='["half-tag"]', primary="half-tag")
        ingested = _strict_ingest(engine, body)
        assert ingested == 2
        rows = engine._store.get_all_canonical_turns(engine.config.conversation_id)
        assert all(r.tagged_at for r in rows)
        # The first pair went through the real tagger (keyword), so its
        # entry does NOT carry the stored half-tag.
        assert engine._turn_tag_index.entries[0].primary_tag != "half-tag"
    finally:
        engine.close()
