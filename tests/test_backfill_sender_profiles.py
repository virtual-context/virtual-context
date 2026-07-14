"""Sender recovery from durable actor profiles.

Rows persisted without retained raw text cannot recover their sender label
from an envelope, but when the row carries an actor id the actor's profile
display name is durable evidence of who spoke. Separately, legacy
logical-turn attribution smeared one member's name onto another member's
row; such a row is detectable because its sender contradicts its own
actor's profile name while exactly matching a different participant's.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from pathlib import Path

from virtual_context.types import Message


CONV = "sk:agent:t:discord:guild:1"


def _make_engine(tmp_path: Path, conversation_id: str = CONV):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine
    cfg = load_config(config_dict={
        "context_window": 10000,
        "conversation_id": conversation_id,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / "t.db")},
        },
        "tag_generator": {"type": "keyword"},
    })
    return VirtualContextEngine(config=cfg)


def _senders(tmp_path: Path, conversation_id: str = CONV):
    conn = sqlite3.connect(tmp_path / "t.db")
    try:
        return {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT user_content, sender FROM canonical_turns"
                " WHERE conversation_id = ?"
                "   AND length(trim(coalesce(user_content,''))) > 0",
                (conversation_id,),
            )
        }
    finally:
        conn.close()


def _persist_pair(engine, text: str, actor: str, sender: str = ""):
    """One completed pair with an actor id and no raw content."""
    info = {
        "message_id": f"m-{abs(hash(text)) % 10_000}",
        "sender": {"id": actor.rsplit(":", 1)[-1], "name": sender},
    }
    metadata = {
        "conversation info": dict(info),
        "_vc_current_conversation": dict(info),
    }
    engine.persist_completed_turn([
        Message(role="user", content=text, metadata=metadata),
        Message(role="assistant", content=f"reply to {text}"),
    ])


def _seed_profile(engine, actor: str, name: str):
    engine._store.upsert_actor_profile_from_turn(
        CONV, actor, name, seen_at="2026-07-14T00:00:00+00:00",
    )


def test_profile_fills_empty_sender_without_raw(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _persist_pair(engine, "who ate the saxifrage", "actor:discord:111")
        _seed_profile(engine, "actor:discord:111", "Roo")
        report = engine.backfill_senders(CONV)
        assert report["profile_filled"] == 1, report
        senders = _senders(tmp_path)
        assert senders["who ate the saxifrage"] == "Roo"
    finally:
        engine.close()


def test_smeared_sender_corrected_to_own_profile_name(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        # BigTex's row was written carrying optics's name (legacy smear).
        _persist_pair(engine, "smeared saxifrage claim", "actor:discord:222")
        conn = sqlite3.connect(tmp_path / "t.db")
        conn.execute(
            "UPDATE canonical_turns SET sender = 'optics'"
            " WHERE length(trim(coalesce(user_content,''))) > 0",
        )
        conn.commit()
        conn.close()
        _seed_profile(engine, "actor:discord:222", "BigTex")
        _seed_profile(engine, "actor:discord:333", "optics")
        # The other participant must appear in the conversation for their
        # profile to be consulted.
        _persist_pair(engine, "optics own message", "actor:discord:333")

        report = engine.backfill_senders(CONV)
        assert report["smear_corrected"] == 1, report
        senders = _senders(tmp_path)
        assert senders["smeared saxifrage claim"] == "BigTex"
    finally:
        engine.close()


def test_old_display_name_is_never_rewritten(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        # The stored sender differs from the current profile name but matches
        # no other participant: a legitimate historical display name.
        _persist_pair(engine, "renamed user message", "actor:discord:444")
        conn = sqlite3.connect(tmp_path / "t.db")
        conn.execute(
            "UPDATE canonical_turns SET sender = 'OldNick'"
            " WHERE length(trim(coalesce(user_content,''))) > 0",
        )
        conn.commit()
        conn.close()
        _seed_profile(engine, "actor:discord:444", "NewNick")

        report = engine.backfill_senders(CONV)
        assert report["smear_corrected"] == 0, report
        senders = _senders(tmp_path)
        assert senders["renamed user message"] == "OldNick"
    finally:
        engine.close()
