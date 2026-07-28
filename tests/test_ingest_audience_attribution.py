"""Audience attribution on the completed-turn persist path.

A turn can reach canonical storage two ways: the payload reconcile during
prepare, or ``persist_completed_turn`` when only the completed pair is
available. The first threads the request's proved audience into the user
row's reply edge; the second historically did not, so any conversation
whose turns were first persisted on the completed-turn path accumulated
rows with actor and channel but no audience attribution — invisible to
every audience-scoped policy surface (speaker roster membership among
them).

``persist_completed_turn`` now accepts the caller's proved audience and
derives the same role-local reply edge the payload reconcile derives.
An empty audience keeps the exact legacy write: version 0, no audience,
no inherited owner.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from pathlib import Path

from virtual_context.types import Message


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
    conn = sqlite3.connect(tmp_path / f"{conversation_id}.db")
    try:
        return [
            {
                "user": row[0],
                "assistant": row[1],
                "audience": row[2],
                "version": row[3],
                "reply_target": row[4],
                "source_message_id": row[5],
            }
            for row in conn.execute(
                "SELECT user_content, assistant_content,"
                "       audience_conversation_id, audience_attribution_version,"
                "       reply_target_message_id, source_message_id"
                "  FROM canonical_turns WHERE conversation_id = ?"
                " ORDER BY sort_key",
                (conversation_id,),
            )
        ]
    finally:
        conn.close()


def _envelope_metadata(**extra) -> dict:
    """Metadata shaped as the envelope parser leaves it: the policy-grade
    ordered snapshot under the reserved key, plus the merged display block."""
    current = {
        "message_id": "m-100",
        "sender": {"id": "555000111", "name": "Roo"},
        **extra,
    }
    return {
        "_vc_current_conversation": current,
        "conversation info": dict(current),
    }


def test_completed_turn_persist_stamps_proved_audience(tmp_path):
    """The completed-pair path (no prior prepare) writes a stamped user row."""
    engine = _make_engine(tmp_path)
    try:
        history = [
            Message(
                role="user",
                content="what did roo say about saxifrage",
                metadata=_envelope_metadata(),
            ),
            Message(role="assistant", content="roo praised the saxifrage"),
        ]
        engine.persist_completed_turn(
            history, source_audience_conversation_id="c",
        )
        rows = _audience_rows(tmp_path)
        assert len(rows) == 2
        user_row = next(r for r in rows if r["user"])
        assistant_row = next(r for r in rows if r["assistant"])
        assert user_row["audience"] == "c"
        assert user_row["version"] == 1
        assert user_row["source_message_id"] == "m-100"
        # Role-local: the assistant row never receives an audience edge.
        assert assistant_row["audience"] == ""
        assert assistant_row["version"] == 0
    finally:
        engine.close()


def test_completed_turn_persist_returns_only_the_accepted_user_actor(tmp_path):
    from virtual_context.types import build_user_turn_metadata

    engine = _make_engine(tmp_path)
    actor_id = "actor:discord:387316537012518913"
    try:
        history = [
            Message(role="user", content="@Vast remember the live-test rule"),
            Message(role="assistant", content="Understood."),
        ]
        accepted = engine.persist_completed_turn(
            history,
            source_audience_conversation_id="c",
            user_turn_metadata=build_user_turn_metadata(
                sender_name="optics",
                sender_actor_id=actor_id,
                origin_channel_id="vasttest2",
                source_conversation_key=(
                    "sk:agent:vast:discord:guild:1524917037191925871"
                ),
            ),
        )
        assert accepted == actor_id
    finally:
        engine.close()


def test_completed_turn_persist_without_audience_is_unchanged(tmp_path):
    """Unproved audience keeps the legacy write: version 0, nothing inherited."""
    engine = _make_engine(tmp_path)
    try:
        history = [
            Message(
                role="user",
                content="unproved route question",
                metadata=_envelope_metadata(),
            ),
            Message(role="assistant", content="unproved route answer"),
        ]
        engine.persist_completed_turn(history)
        rows = _audience_rows(tmp_path)
        assert len(rows) == 2
        assert all(r["audience"] == "" for r in rows)
        assert all(r["version"] == 0 for r in rows)
    finally:
        engine.close()


def test_completed_turn_persist_threads_reply_lanes(tmp_path):
    """The reply edge rides the same derivation as the payload reconcile."""
    engine = _make_engine(tmp_path)
    try:
        history = [
            Message(
                role="user",
                content="replying to the earlier claim",
                metadata=_envelope_metadata(reply_to_id="m-042"),
            ),
            Message(role="assistant", content="noted the reply"),
        ]
        engine.persist_completed_turn(
            history, source_audience_conversation_id="c",
        )
        rows = _audience_rows(tmp_path)
        user_row = next(r for r in rows if r["user"])
        assert user_row["reply_target"] == "m-042"
        assert user_row["audience"] == "c"
    finally:
        engine.close()


def test_backfill_stamps_audience_on_rows_without_raw_content(tmp_path):
    """A row persisted without retained raw text can never recover a reply
    edge, but its audience is route-level: the recorded origin route (or the
    conversation itself) is proved through the resolver and stamped, so the
    row becomes visible to audience-scoped policy."""
    engine = _make_engine(tmp_path)
    try:
        history = [
            Message(role="user", content="legacy unstamped question"),
            Message(role="assistant", content="legacy unstamped answer"),
        ]
        # Legacy shape: persisted with no proved audience and no raw content.
        engine.persist_completed_turn(history)
        before = _audience_rows(tmp_path)
        assert all(r["version"] == 0 for r in before)

        upsert = getattr(engine._store, "upsert_conversation", None)
        if callable(upsert):
            upsert(tenant_id="", conversation_id="c")
        report = engine.backfill_reply_roles("c")
        assert report.get("audience_only", 0) >= 1, report

        rows = _audience_rows(tmp_path)
        user_row = next(r for r in rows if r["user"])
        assert user_row["audience"] == "c"
        assert user_row["version"] == 1
    finally:
        engine.close()


def test_prepare_stamped_row_survives_completed_turn_resend(tmp_path):
    """A user row the payload reconcile already stamped keeps its stamp when
    the completed pair arrives with no proved audience (the tail-anchor path
    mirrors the row instead of rewriting it)."""
    from virtual_context.proxy.formats import detect_format

    engine = _make_engine(tmp_path)
    try:
        body = {
            "messages": [
                {"role": "user", "content": "prepared saxifrage question"},
            ],
        }
        engine._ingest_reconciler.ingest_batch(
            engine.config.conversation_id,
            body=body,
            fmt=detect_format(body),
            expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
            source_conversation_key="c",
            source_audience_conversation_id="c",
        )
        history = [
            Message(role="user", content="prepared saxifrage question"),
            Message(role="assistant", content="prepared saxifrage answer"),
        ]
        engine.persist_completed_turn(history)
        rows = _audience_rows(tmp_path)
        assert len(rows) == 2
        user_row = next(r for r in rows if r["user"])
        assert user_row["audience"] == "c"
        assert user_row["version"] == 1
    finally:
        engine.close()
