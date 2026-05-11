"""Regression tests for engine commit-3 tag-summary fixes.

Two engine surfaces under test:

1. ``CompactionPipeline._build_tag_summaries`` (extracted from
   ``_run_compaction``): phase ``tag_summaries`` must build summaries
   even when the engine's in-memory ``_turn_tag_index.entries`` is
   empty (cold-start / takeover compactions). The previous gate
   short-circuited the block when entries were absent, even though
   ``cover_tags`` was correctly populated from the primary-tag
   guarantee. The fix derives ``tag_to_turns`` /
   ``tag_to_canonical_turn_ids`` / ``max_turn`` from ``compact_rows``
   as a fallback.

2. ``engine.backfill_tag_summaries()``: recovery primitive for
   conversations whose segments are durable but whose tag_summaries
   table is empty. Reads stored segments + canonical_turns, derives
   cover tags, runs the compactor's tag-summary builder, persists
   the rows. Idempotent by default; ``force_rebuild=True`` overwrites
   existing summaries.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timezone

import pytest

from virtual_context.types import (
    CanonicalTurnRow,
    CompactionResult,
    SegmentMetadata,
    StoredSegment,
    TagSummary,
    TurnTagEntry,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingCompactor:
    """Stub compactor that records ``compact_tag_summaries`` calls and
    returns a synthetic ``TagSummary`` per cover tag.

    Lets the tests assert what kwargs the pipeline / engine flowed
    through without burning an LLM provider.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.model_name = "stub"

    def compact_tag_summaries(
        self,
        *,
        cover_tags,
        tag_to_summaries,
        tag_to_turns,
        tag_to_canonical_turn_ids=None,
        existing_tag_summaries=None,
        max_turn,
        generated_by_turn_id="",
    ):
        self.calls.append({
            "cover_tags": list(cover_tags),
            "tag_to_summaries": {k: list(v) for k, v in tag_to_summaries.items()},
            "tag_to_turns": {k: list(v) for k, v in tag_to_turns.items()},
            "tag_to_canonical_turn_ids": {
                k: list(v) for k, v in (tag_to_canonical_turn_ids or {}).items()
            },
            "existing_tag_summaries": dict(existing_tag_summaries or {}),
            "max_turn": max_turn,
            "generated_by_turn_id": generated_by_turn_id,
        })
        return [
            TagSummary(
                tag=tag,
                summary=f"summary-for-{tag}",
                summary_tokens=10,
                covers_through_turn=max_turn,
                source_segment_refs=[
                    getattr(s, "ref", "") for s in tag_to_summaries.get(tag, [])
                ],
            )
            for tag in cover_tags
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_with_stub_compactor(tmp_path):
    """Build a fresh SQLite-backed engine and replace its compactor with
    a recording stub. Returns ``(engine, stub_compactor)``.
    """
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    db_path = str(tmp_path / "store.db")
    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": "conv-tag-backfill",
    })
    engine = VirtualContextEngine(config=config)
    stub = _RecordingCompactor()
    engine._compactor = stub
    engine._compaction._compactor = stub
    return engine, stub


def _underlying(engine):
    """Unwrap ConversationStoreView to reach the raw CompositeStore."""
    store = engine._store
    return getattr(store, "_store", store)


def _seed_segment(
    raw_store, conversation_id, *, ref, primary_tag, tags, summary_text,
):
    """Persist a single StoredSegment for the conversation so
    ``get_all_segments`` and ``get_summaries_by_tags`` return content.
    """
    now = datetime.now(timezone.utc)
    seg = StoredSegment(
        ref=ref,
        conversation_id=conversation_id,
        primary_tag=primary_tag,
        tags=list(tags),
        summary=summary_text,
        summary_tokens=len(summary_text) // 4,
        full_text=summary_text,
        messages=[],
        metadata=SegmentMetadata(turn_count=1),
        start_timestamp=now,
        end_timestamp=now,
        created_at=now,
    )
    raw_store.store_segment(seg)


def _seed_canonical_turn(
    raw_store, conversation_id, *, turn_number, canonical_turn_id,
    primary_tag, tags,
):
    """Persist a single canonical_turns row with the requested tags."""
    raw_store.save_canonical_turn(
        conversation_id=conversation_id,
        turn_number=turn_number,
        canonical_turn_id=canonical_turn_id,
        user_content=f"user-{turn_number}",
        assistant_content=f"asst-{turn_number}",
        primary_tag=primary_tag,
        tags=list(tags),
        session_date="2026-05-10",
        sender="user",
    )


def _make_result(*, segment_id, primary_tag, tags, summary):
    """CompactionResult shorthand for tests."""
    return CompactionResult(
        segment_id=segment_id,
        primary_tag=primary_tag,
        tags=list(tags),
        summary=summary,
        summary_tokens=len(summary) // 4,
        original_tokens=len(summary),
        full_text=summary,
    )


def _make_row(*, conversation_id, turn_number, canonical_turn_id, primary_tag, tags):
    """CanonicalTurnRow shorthand for tests."""
    return CanonicalTurnRow(
        conversation_id=conversation_id,
        canonical_turn_id=canonical_turn_id,
        turn_number=turn_number,
        primary_tag=primary_tag,
        tags=list(tags),
    )


# ---------------------------------------------------------------------------
# Part 1 — _build_tag_summaries with empty / populated _turn_tag_index
# ---------------------------------------------------------------------------


def test_compaction_tag_summaries_populated_when_in_memory_index_empty(
    engine_with_stub_compactor,
) -> None:
    """When the engine's ``_turn_tag_index`` is empty (cold-start /
    takeover scenario), ``_build_tag_summaries`` must still build
    summaries by deriving turn data from ``compact_rows``.

    Previously the block was gated on ``self._turn_tag_index.entries``,
    silently skipping the entire tag-summary pass even though
    ``cover_tags`` was correctly populated from each segment's
    primary_tag. The fix removes the gate and derives the needed
    turn/canonical_turn maps from the compact_rows passed in.
    """
    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    # Pre-seed segments so get_summaries_by_tags returns content.
    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha", "beta"], summary_text="content for alpha+beta",
    )
    _seed_segment(
        raw, conv_id, ref="seg-2", primary_tag="gamma",
        tags=["gamma"], summary_text="content for gamma",
    )

    # Sanity: in-memory index is empty.
    assert engine._turn_tag_index.entries == []

    results = [
        _make_result(segment_id="seg-1", primary_tag="alpha",
                     tags=["alpha", "beta"], summary="alpha+beta"),
        _make_result(segment_id="seg-2", primary_tag="gamma",
                     tags=["gamma"], summary="gamma"),
    ]
    compact_rows = [
        _make_row(conversation_id=conv_id, turn_number=0,
                  canonical_turn_id="ct-0", primary_tag="alpha",
                  tags=["alpha", "beta"]),
        _make_row(conversation_id=conv_id, turn_number=1,
                  canonical_turn_id="ct-1", primary_tag="gamma",
                  tags=["gamma"]),
    ]

    count, cover_tags = engine._compaction._build_tag_summaries(
        results=results,
        compact_rows=compact_rows,
        operation_id=None,
    )

    # The greedy cover-set (computed against the empty in-memory index)
    # returns []; the primary-tag guarantee then adds each result's
    # primary_tag. Two distinct primary_tags here (alpha + gamma) means
    # two cover tags. `beta` is a non-primary tag and is NOT eligible
    # for the cover-set when the in-memory index is empty (matches the
    # existing primary-tag-guarantee contract, just exercised on the
    # fallback path now). The fix's purpose is closing the silent-skip
    # gap, not expanding cover-tag eligibility.
    assert count == 2
    assert sorted(cover_tags) == ["alpha", "gamma"]

    # Stub recorded one call with the fallback-derived turn data.
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["tag_to_turns"]["alpha"] == [0]
    assert call["tag_to_turns"]["gamma"] == [1]
    assert call["tag_to_canonical_turn_ids"]["alpha"] == ["ct-0"]
    assert call["tag_to_canonical_turn_ids"]["gamma"] == ["ct-1"]
    assert call["max_turn"] == 1

    # Persisted via save_tag_summary.
    persisted_tags = {
        ts.tag
        for ts in engine._store.get_all_tag_summaries(conversation_id=conv_id)
    }
    assert persisted_tags == {"alpha", "gamma"}


def test_compaction_tag_summaries_unchanged_when_in_memory_index_populated(
    engine_with_stub_compactor,
) -> None:
    """Existing behavior preserved: when the in-memory index is
    populated, the tag-summary block uses it (not compact_rows) for
    ``tag_to_turns`` / ``tag_to_canonical_turn_ids`` / ``max_turn``.

    Regression guard against the fix accidentally re-deriving from
    compact_rows when the index is live.
    """
    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha"], summary_text="alpha content",
    )

    # Populate the in-memory index with an entry that has DIFFERENT
    # turn_number / canonical_turn_id from the compact_rows. The two
    # data sources disagree on purpose so the test discriminates the
    # path: if the index path is taken, the stub sees turn 42 / cid
    # "ct-from-index"; if the fallback fires, the stub sees turn 0 /
    # cid "ct-from-rows".
    engine._turn_tag_index.append(TurnTagEntry(
        turn_number=42,
        canonical_turn_id="ct-from-index",
        message_hash="h",
        primary_tag="alpha",
        tags=["alpha"],
    ))

    results = [
        _make_result(segment_id="seg-1", primary_tag="alpha",
                     tags=["alpha"], summary="alpha content"),
    ]
    compact_rows = [
        _make_row(conversation_id=conv_id, turn_number=0,
                  canonical_turn_id="ct-from-rows", primary_tag="alpha",
                  tags=["alpha"]),
    ]

    count, _ = engine._compaction._build_tag_summaries(
        results=results,
        compact_rows=compact_rows,
        operation_id=None,
    )

    assert count == 1
    assert len(stub.calls) == 1
    call = stub.calls[0]
    # Index path taken: turn_number from the index entry (42), not 0.
    assert call["tag_to_turns"]["alpha"] == [42]
    assert call["tag_to_canonical_turn_ids"]["alpha"] == ["ct-from-index"]
    assert call["max_turn"] == 42


# ---------------------------------------------------------------------------
# Part 2 — engine.backfill_tag_summaries
# ---------------------------------------------------------------------------


def test_backfill_tag_summaries_writes_summaries_for_stored_segments(
    engine_with_stub_compactor,
) -> None:
    """``engine.backfill_tag_summaries`` writes one tag_summary per
    cover tag derived from stored segments + canonical turns.

    Pre-seed segments + canonical turns with 0 tag_summaries (the
    production state of the 4 affected conversations). Call backfill.
    Assert summaries materialize for every distinct tag (excluding
    ``_general``), the right turn data was flowed to the compactor,
    and the returned count matches.
    """
    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha", "beta"], summary_text="content alpha+beta",
    )
    _seed_segment(
        raw, conv_id, ref="seg-2", primary_tag="gamma",
        tags=["gamma"], summary_text="content gamma",
    )
    _seed_canonical_turn(
        raw, conv_id, turn_number=0, canonical_turn_id="ct-0",
        primary_tag="alpha", tags=["alpha", "beta"],
    )
    _seed_canonical_turn(
        raw, conv_id, turn_number=1, canonical_turn_id="ct-1",
        primary_tag="gamma", tags=["gamma"],
    )

    # Sanity: store currently has no tag_summaries.
    assert engine._store.get_all_tag_summaries(conversation_id=conv_id) == []

    count = engine.backfill_tag_summaries()
    # Three distinct tags: alpha, beta, gamma. None was _general.
    assert count == 3
    persisted_tags = {
        ts.tag
        for ts in engine._store.get_all_tag_summaries(conversation_id=conv_id)
    }
    assert persisted_tags == {"alpha", "beta", "gamma"}

    # Verify the compactor saw the right turn data.
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert sorted(call["cover_tags"]) == ["alpha", "beta", "gamma"]
    assert call["max_turn"] == 1
    assert sorted(call["tag_to_turns"]["alpha"]) == [0]
    assert sorted(call["tag_to_turns"]["gamma"]) == [1]


def test_backfill_tag_summaries_idempotent_default(
    engine_with_stub_compactor,
) -> None:
    """Calling backfill twice in a row is idempotent by default. The
    second call returns 0 and does not invoke the compactor again."""
    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha"], summary_text="alpha content",
    )
    _seed_canonical_turn(
        raw, conv_id, turn_number=0, canonical_turn_id="ct-0",
        primary_tag="alpha", tags=["alpha"],
    )

    first = engine.backfill_tag_summaries()
    assert first == 1
    assert len(stub.calls) == 1

    second = engine.backfill_tag_summaries()
    assert second == 0
    # Compactor was NOT called again because the only cover tag
    # already had a summary row.
    assert len(stub.calls) == 1


def test_backfill_tag_summaries_force_rebuild_overwrites(
    engine_with_stub_compactor,
) -> None:
    """``force_rebuild=True`` rebuilds every cover tag's summary even
    when a row already exists. The compactor is invoked again and the
    persisted summary is the freshly-built one."""
    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha"], summary_text="alpha content",
    )
    _seed_canonical_turn(
        raw, conv_id, turn_number=0, canonical_turn_id="ct-0",
        primary_tag="alpha", tags=["alpha"],
    )

    first = engine.backfill_tag_summaries()
    assert first == 1
    assert len(stub.calls) == 1

    forced = engine.backfill_tag_summaries(force_rebuild=True)
    assert forced == 1
    # Compactor was called a second time.
    assert len(stub.calls) == 2
    # The force-rebuild call passed an EMPTY existing_tag_summaries
    # so the compactor's freshness gate does not suppress the rebuild.
    assert stub.calls[1]["existing_tag_summaries"] == {}


# ---------------------------------------------------------------------------
# Diagnostic logging in backfill_tag_summaries
# ---------------------------------------------------------------------------
#
# The "0-write but status=ok" outcome ops observed on small
# experimental conversations could mean: (a) all per-tag LLM calls
# failed silently, (b) freshness gate inside the compactor
# short-circuited, or (c) something else. Logging makes the path the
# invocation took visible so a re-run is diagnostic without engine
# changes.


def test_backfill_emits_start_and_done_lines_with_counts(
    engine_with_stub_compactor, caplog,
) -> None:
    """The backfill emits a ``starting ... segments=N force_rebuild=...``
    INFO line at entry and a ``done ... tags_attempted=N tags_persisted=N
    tags_missing=N`` INFO line at exit."""
    import logging as _logging

    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha"], summary_text="alpha content",
    )
    _seed_canonical_turn(
        raw, conv_id, turn_number=0, canonical_turn_id="ct-0",
        primary_tag="alpha", tags=["alpha"],
    )

    with caplog.at_level(_logging.INFO, logger="virtual_context.engine"):
        engine.backfill_tag_summaries()

    messages = [rec.getMessage() for rec in caplog.records]
    starting = [m for m in messages if "backfill_tag_summaries: starting" in m]
    done = [m for m in messages if "backfill_tag_summaries: done" in m]
    persisted = [m for m in messages if "backfill_tag_summaries: persisted tag=" in m]
    assert starting, f"missing 'starting' INFO line; got: {messages}"
    assert done, f"missing 'done' INFO line; got: {messages}"
    assert persisted, f"missing per-tag 'persisted' INFO line; got: {messages}"
    # The 'starting' line carries segments + force_rebuild.
    assert "segments=1" in starting[0]
    assert "force_rebuild=False" in starting[0]
    # The 'done' line carries the aggregate counts.
    assert "tags_attempted=1" in done[0]
    assert "tags_persisted=1" in done[0]
    assert "tags_missing=0" in done[0]


def test_backfill_warns_when_compactor_returns_fewer_than_requested(
    engine_with_stub_compactor, caplog,
) -> None:
    """When the compactor returns fewer summaries than the requested
    cover tags (per-tag LLM failures, internal freshness short-circuit,
    etc.), a WARNING line surfaces the missing-tag list. This is the
    discriminator for the 0-write / partial-write diagnostic case ops
    flagged on a small experimental conversation: a re-run with this
    logging tells ops which path the invocation took."""
    import logging as _logging

    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha", "beta"], summary_text="alpha+beta",
    )
    _seed_canonical_turn(
        raw, conv_id, turn_number=0, canonical_turn_id="ct-0",
        primary_tag="alpha", tags=["alpha", "beta"],
    )

    # Stub compactor to silently drop one tag — simulates the case where
    # per-tag LLM calls produced no result for some cover tags.
    def _selective(*, cover_tags, max_turn, **kwargs):
        return [
            TagSummary(
                tag="alpha",
                summary="summary-for-alpha",
                summary_tokens=10,
                covers_through_turn=max_turn,
            ),
            # 'beta' deliberately absent from the returned list.
        ]
    stub.compact_tag_summaries = _selective  # type: ignore[assignment]

    with caplog.at_level(_logging.WARNING, logger="virtual_context.engine"):
        engine.backfill_tag_summaries()

    warnings = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno >= _logging.WARNING
    ]
    matching = [
        m for m in warnings
        if "compactor returned no summary for" in m and "beta" in m
    ]
    assert matching, (
        "expected WARNING surfacing missing tags from compactor; "
        f"got warnings: {warnings}"
    )
    assert "1 of 2" in matching[0]  # 1 missing of 2 requested


def test_backfill_logs_idempotency_skip_count(
    engine_with_stub_compactor, caplog,
) -> None:
    """The idempotent re-run emits an INFO line documenting how many
    tags were skipped because they already had summaries. Lets ops see
    that the second run was indeed idempotent rather than a silent
    no-op."""
    import logging as _logging

    engine, stub = engine_with_stub_compactor
    conv_id = engine.config.conversation_id

    raw = _underlying(engine)
    _seed_segment(
        raw, conv_id, ref="seg-1", primary_tag="alpha",
        tags=["alpha"], summary_text="alpha content",
    )
    _seed_canonical_turn(
        raw, conv_id, turn_number=0, canonical_turn_id="ct-0",
        primary_tag="alpha", tags=["alpha"],
    )

    # First run writes the summary.
    engine.backfill_tag_summaries()

    # Second run should idempotently skip and log the skip count.
    caplog.clear()
    with caplog.at_level(_logging.INFO, logger="virtual_context.engine"):
        engine.backfill_tag_summaries()

    messages = [rec.getMessage() for rec in caplog.records]
    skip_lines = [
        m for m in messages
        if "all cover tags already have summaries" in m
    ]
    assert skip_lines, (
        "expected idempotency-skip INFO line on the second run; "
        f"got: {messages}"
    )


def test_backfill_warns_when_no_compactor_configured(tmp_path, caplog) -> None:
    """When the engine has no compactor (no LLM provider), the backfill
    emits a WARNING explaining the missing configuration rather than
    silently returning 0. Makes deploy-misconfiguration scenarios
    self-diagnostic."""
    import logging as _logging

    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    db_path = str(tmp_path / "store.db")
    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": "conv-no-compactor",
    })
    engine = VirtualContextEngine(config=config)
    # Default engine has no LLM provider configured → _compactor is None.
    assert engine._compactor is None

    with caplog.at_level(_logging.WARNING, logger="virtual_context.engine"):
        count = engine.backfill_tag_summaries()

    assert count == 0
    warnings = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno >= _logging.WARNING
    ]
    assert any("no compactor configured" in m for m in warnings), (
        f"expected 'no compactor configured' WARNING; got: {warnings}"
    )
