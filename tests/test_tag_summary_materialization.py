"""Tag-summary materialization contract at compaction (BUG-041).

Compaction used to materialize tag summaries only for the greedy
set-cover's tags (intersected with the just-compacted segments' tags)
plus each segment's primary tag. Every non-primary secondary tag
outside the greedy cover therefore landed in ``segment_tags`` with NO
``tag_summaries`` row — on every compaction, by construction. Those
tags were invisible to the context-hint topic list, absent from the
broad/recall-all summary floor, and missing from tag-summary-embedding
scoring; repair sweeps that materialized them from outside left them
permanently stale because later compactions still skipped them.

The read paths and the repair layer both assume the WIDE contract:
every non-``_general`` tag carried by stored segments has a summary.
The producer now honors it — compaction materializes (and, via the
existing staleness check, refreshes) a summary for every
non-``_general`` tag carried by the just-compacted segments.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from virtual_context.types import CompactionResult, SegmentMetadata, TagSummary


def _make_engine(tmp_path: Path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine
    cfg = load_config(config_dict={
        "context_window": 10000,
        "conversation_id": "c",
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "c.db")}},
        "tag_generator": {
            "type": "keyword",
            "keyword_fallback": {"tag_keywords": {
                "alpine-gardening": ["saxifrage"],
                "watering": ["water"],
                "winter-care": ["winter"],
                "perfume": ["perfume"],
                "gifts": ["gift"],
            }},
        },
        "compaction": {"protected_recent_turns": 1},
    })
    return VirtualContextEngine(config=cfg)


def _stub_compactor():
    now = datetime.now(timezone.utc)
    compactor = MagicMock()

    def _compact(segments, **_kwargs):
        return [
            CompactionResult(
                segment_id=getattr(seg, "id", f"seg-{i}"),
                primary_tag=seg.primary_tag,
                tags=list(seg.tags),
                summary=f"summary {i}",
                summary_tokens=4,
                full_text="x",
                original_tokens=20,
                messages=[{"role": "user", "content": "x"}],
                metadata=SegmentMetadata(turn_count=1, session_date=""),
                compression_ratio=0.5,
                timestamp=now,
                facts=[],
            )
            for i, seg in enumerate(segments)
        ]

    compactor.compact.side_effect = _compact
    compactor.compact_tag_summaries.side_effect = (
        lambda cover_tags, **_kwargs: [
            TagSummary(tag=tag, summary=f"tag summary for {tag}")
            for tag in cover_tags
        ]
    )
    compactor.model_name = "test-model"
    return compactor


def _compacted(tmp_path: Path):
    """Engine with one organic compaction over multi-tag turns whose
    secondary tags the greedy cover drops."""
    from virtual_context.proxy.formats import detect_format, extract_ingestible_messages
    engine = _make_engine(tmp_path)
    compactor = _stub_compactor()
    engine._compaction._compactor = compactor
    engine._tagging._compactor = compactor
    texts = [
        "saxifrage needs water in summer",
        "saxifrage cushion in winter needs cover",
        "perfume gift for the winter holidays",
        "water the saxifrage less in winter",
        "perfume and gift wrapping question",
        "saxifrage water winter recap",
    ]
    body = {"messages": []}
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
    report = engine._compaction.compact_manual(messages)
    return engine, compactor, report


def _tag_sets(tmp_path: Path) -> tuple[set, set]:
    conn = sqlite3.connect(tmp_path / "c.db")
    try:
        segment_tags = {
            row[0] for row in conn.execute("SELECT DISTINCT tag FROM segment_tags")
        }
        summary_tags = {
            row[0] for row in conn.execute(
                "SELECT tag FROM tag_summaries WHERE conversation_id = 'c'"
            )
        }
        segment_tags.discard("_general")
        segment_tags.discard("")
        return segment_tags, summary_tags
    finally:
        conn.close()


@pytest.mark.regression("BUG-041")
def test_every_segment_tag_gets_a_summary_at_compaction(tmp_path):
    engine, _compactor, report = _compacted(tmp_path)
    try:
        assert report is not None and report.segments_compacted > 0
        segment_tags, summary_tags = _tag_sets(tmp_path)
        assert segment_tags, "seed must produce multi-tag segments"
        gap = segment_tags - summary_tags
        assert not gap, (
            f"compaction left segment tags with no tag_summaries row: {sorted(gap)}"
        )
    finally:
        engine.close()


@pytest.mark.regression("BUG-041")
def test_general_never_gets_a_summary_from_secondary_widening(tmp_path):
    engine, _compactor, _report = _compacted(tmp_path)
    try:
        _segment_tags, summary_tags = _tag_sets(tmp_path)
        assert "_general" not in summary_tags
    finally:
        engine.close()


@pytest.mark.regression("BUG-041")
def test_report_cover_tags_reflect_materialized_set(tmp_path):
    engine, _compactor, report = _compacted(tmp_path)
    try:
        segment_tags, _summary_tags = _tag_sets(tmp_path)
        assert set(report.cover_tags) >= segment_tags, (
            f"report.cover_tags {sorted(report.cover_tags)} must cover the "
            f"segment tags {sorted(segment_tags)}"
        )
    finally:
        engine.close()


@pytest.mark.regression("BUG-041")
def test_secondary_tags_stay_eligible_for_refresh_on_later_compactions(tmp_path):
    """The staleness half of the defect: a secondary tag materialized once
    must be handed to the tag-summary builder again when a LATER
    compaction touches segments carrying it — under the old cover-set
    contract it was skipped forever and its summary went permanently
    stale."""
    from virtual_context.proxy.formats import detect_format, extract_ingestible_messages
    engine, compactor, _report = _compacted(tmp_path)
    try:
        first_pass_tags = set(compactor.compact_tag_summaries.call_args.kwargs["cover_tags"])
        assert "watering" in first_pass_tags

        # Second round of turns touching the same secondary tags.
        body = {"messages": []}
        for m in engine._store.get_all_canonical_turns("c"):
            pass  # rows already exist; extend the payload with new turns
        texts = [
            "more water questions about the saxifrage cushion",
            "winter watering cadence for the perfume workshop",
        ]
        prior = {"messages": []}
        rows = engine._store.get_all_canonical_turns("c")
        for row in sorted(rows, key=lambda r: (r.sort_key, r.canonical_turn_id)):
            if (row.user_content or "").strip():
                prior["messages"].append({"role": "user", "content": row.user_content})
            if (row.assistant_content or "").strip():
                prior["messages"].append({"role": "assistant", "content": row.assistant_content})
        body["messages"] = prior["messages"]
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
        report2 = engine._compaction.compact_manual(messages)
        assert report2 is not None and report2.segments_compacted > 0
        second_pass_tags = set(compactor.compact_tag_summaries.call_args.kwargs["cover_tags"])
        assert "watering" in second_pass_tags, (
            "a secondary tag on newly compacted segments must be handed to "
            "the tag-summary builder for the staleness check on every "
            f"compaction; second pass saw only {sorted(second_pass_tags)}"
        )
    finally:
        engine.close()
