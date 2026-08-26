"""Assistant rows carry the pair's proved audience (BUG-058).

Audience provenance rides the reply-edge struct, and both admission
surfaces gave assistant physical rows the empty edge: the reply edge is
speaker attribution and must stay user-only, but the audience is a
property of the request's PROVED route, and both halves of a turn were
produced inside the same proved channel. Unstamped assistant rows are
categorically invisible to every audience-scoped read (candidate
admission requires an exact audience and version match) even though
admission has no role rule, and no repair surface could stamp them:
the reply-roles backfill skips rows without user content.

Fix: an audience-only edge (audience fields set, every speaker field
empty) is stamped on assistant rows at both admission surfaces when the
pair's audience is proved, and ``backfill_assistant_audience`` repairs
existing rows from the sibling user row's proved audience.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from pathlib import Path

import pytest


def _make_engine(tmp_path: Path, conversation_id: str = "c"):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine
    cfg = load_config(config_dict={
        "context_window": 10000,
        "conversation_id": conversation_id,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / f"{conversation_id}.db")},
        },
        "tag_generator": {"type": "keyword"},
    })
    return VirtualContextEngine(config=cfg)


def _audience_rows(tmp_path: Path, conversation_id: str = "c"):
    """(role, audience_conversation_id, audience_attribution_version,
    sender, sender_actor_id) per row in sort order."""
    conn = sqlite3.connect(tmp_path / f"{conversation_id}.db")
    try:
        return [
            (
                "user" if (row[0] or "").strip() else "assistant",
                row[2] or "",
                int(row[3] or 0),
                row[4] or "",
                row[5] or "",
            )
            for row in conn.execute(
                "SELECT user_content, assistant_content, "
                "       audience_conversation_id, audience_attribution_version, "
                "       sender, sender_actor_id "
                "FROM canonical_turns WHERE conversation_id = ? "
                "ORDER BY sort_key",
                (conversation_id,),
            )
        ]
    finally:
        conn.close()


def _batch(engine, body: dict, audience: str = ""):
    from virtual_context.proxy.formats import detect_format
    return engine._ingest_reconciler.ingest_batch(
        engine.config.conversation_id,
        body=body,
        fmt=detect_format(body),
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
        source_audience_conversation_id=audience,
    )


_TURN_BODY = {
    "messages": [
        {"role": "user", "content": "how heavy should the squats be"},
        {"role": "assistant", "content": "185 for the working sets"},
    ],
}


@pytest.mark.regression("BUG-058")
def test_batch_ingest_stamps_assistant_rows_with_proved_audience(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _batch(engine, _TURN_BODY, audience="c")
        rows = _audience_rows(tmp_path)
        assert [(r[0], r[1], r[2]) for r in rows] == [
            ("user", "c", 1),
            ("assistant", "c", 1),
        ], rows
        # The audience stamp must never smear speaker attribution onto
        # the assistant half.
        assistant = [r for r in rows if r[0] == "assistant"][0]
        assert assistant[3] == "" and assistant[4] == "", rows
    finally:
        engine.close()


@pytest.mark.regression("BUG-058")
def test_batch_ingest_unproved_audience_leaves_assistant_unstamped(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _batch(engine, _TURN_BODY, audience="")
        rows = _audience_rows(tmp_path)
        assert all(r[1] == "" and r[2] == 0 for r in rows), rows
    finally:
        engine.close()


@pytest.mark.regression("BUG-058")
def test_ingest_single_stamps_assistant_from_user_edge(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        engine._ingest_reconciler.ingest_single(
            "c",
            user_content="log my deadlift",
            assistant_content="logged at 315",
            user_reply_edge={
                "audience_conversation_id": "c",
                "audience_attribution_version": 1,
            },
        )
        rows = _audience_rows(tmp_path)
        assert [(r[0], r[1], r[2]) for r in rows] == [
            ("user", "c", 1),
            ("assistant", "c", 1),
        ], rows
    finally:
        engine.close()


@pytest.mark.regression("BUG-058")
def test_ingest_single_unproved_leaves_assistant_unstamped(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        engine._ingest_reconciler.ingest_single(
            "c",
            user_content="plain question",
            assistant_content="plain answer",
        )
        rows = _audience_rows(tmp_path)
        assert all(r[1] == "" and r[2] == 0 for r in rows), rows
    finally:
        engine.close()


def _strip_assistant_stamps(tmp_path: Path, conversation_id: str = "c"):
    """Simulate rows persisted before assistant stamping existed."""
    conn = sqlite3.connect(tmp_path / f"{conversation_id}.db")
    try:
        conn.execute(
            "UPDATE canonical_turns "
            "SET audience_conversation_id = '', audience_attribution_version = 0 "
            "WHERE conversation_id = ? "
            "AND COALESCE(TRIM(user_content), '') = ''",
            (conversation_id,),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.regression("BUG-058")
def test_backfill_assistant_audience_dry_run_then_apply(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _batch(engine, _TURN_BODY, audience="c")
        _strip_assistant_stamps(tmp_path)
        assert [r for r in _audience_rows(tmp_path) if r[0] == "assistant"][0][2] == 0

        report = engine.backfill_assistant_audience("c", dry_run=True)
        assert report["dry_run"] is True
        assert report["audience_only"] == 1, report
        assert [r for r in _audience_rows(tmp_path) if r[0] == "assistant"][0][2] == 0

        report = engine.backfill_assistant_audience("c", dry_run=False)
        assert report["audience_only"] == 1, report
        assistant = [r for r in _audience_rows(tmp_path) if r[0] == "assistant"][0]
        assert (assistant[1], assistant[2]) == ("c", 1)

        # Idempotent: a second apply stages nothing.
        report = engine.backfill_assistant_audience("c", dry_run=False)
        assert report["audience_only"] == 0, report
    finally:
        engine.close()


@pytest.mark.regression("BUG-058")
def test_backfill_skips_assistant_with_unstamped_sibling(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _batch(engine, _TURN_BODY, audience="")
        report = engine.backfill_assistant_audience("c", dry_run=False)
        assert report["audience_only"] == 0, report
        assert report["skipped_no_sibling"] == 1, report
        rows = _audience_rows(tmp_path)
        assert all(r[1] == "" and r[2] == 0 for r in rows), rows
    finally:
        engine.close()
