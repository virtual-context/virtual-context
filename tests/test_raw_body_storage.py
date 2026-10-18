"""Raw bodies are stored compacted (BUG-131).

Hosts resend recent images and their own rendering of the conversation with
every request, and each stored turn kept a full copy: one long-running guild
conversation held 726 MB of raw bodies whose distinct images were 25 MB and
whose user text was under 2 MB. Every read of those rows then moved megabytes
per row into the process.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import json
import sqlite3
from datetime import timedelta

import pytest

from virtual_context.core.raw_bodies import (
    BLOB_REF_PREFIX,
    blob_refs,
    compact_raw_body,
    expand_raw_body,
)
from virtual_context.engine import VirtualContextEngine
from virtual_context.proxy._envelope import _extract_envelope_metadata
from virtual_context.storage import raw_blobs
from virtual_context.storage.sqlite import SQLiteStore


def _image(seed: str) -> str:
    return "data:image/png;base64," + base64.b64encode((seed * 4000).encode()).decode()


_CONVERSATION_INFO = (
    "Conversation info: ⟦openclaw:ctx⟧\n"
    "```json\n"
    '{"conversation_label": "Guild #general channel id:111", "chat_id": "channel:111"}\n'
    "```\n\n"
    "Sender (untrusted metadata): ⟦openclaw:ctx⟧\n"
    "```json\n"
    '{"label": "Kuw9239 (kuw9239)", "id": "42", "name": "Kuw9239"}\n'
    "```\n\n"
)
_REPLAYED = "".join(
    f"#15611879273320939{i:02d} 2026-10-18 01:{i % 60:02d}:27 UTC Someone (someone): message number {i} in the channel\n"
    for i in range(200)
)


def _host_prompt(words: str) -> str:
    return (
        _CONVERSATION_INFO
        + "Chat history since last reply: ⟦openclaw:ctx⟧\n"
        + _REPLAYED
        + "\nCurrent user request:\n"
        + words
    )


def _vast_like_body(words: str, images: list[str]) -> str:
    parts = [{"type": "input_text", "text": _host_prompt(words)}]
    parts += [{"type": "input_image", "image_url": image} for image in images]
    return json.dumps(parts)


@pytest.mark.regression("BUG-131")
def test_compaction_moves_images_out_and_keeps_what_normalization_reads():
    raw = _vast_like_body("is the red one better?", [_image("a"), _image("b"), _image("a")])
    compacted = compact_raw_body(raw)
    assert len(compacted.text) < len(raw) / 20
    assert len(compacted.blobs) == 2
    assert blob_refs(compacted.text) == set(compacted.blobs)
    for name in ("_effective_raw_text",):
        original = getattr(VirtualContextEngine, name)(raw)
        stored = getattr(VirtualContextEngine, name)(compacted.text)
        assert _extract_envelope_metadata(original) == _extract_envelope_metadata(stored)
    assert VirtualContextEngine._sender_from_raw_content(compacted.text) == "Kuw9239"
    assert (
        VirtualContextEngine._channel_from_raw_content(compacted.text)
        == VirtualContextEngine._channel_from_raw_content(raw)
    )
    assert compact_raw_body(compacted.text).text == compacted.text
    expanded = json.loads(expand_raw_body(compacted.text, lambda hashes: {
        h: compacted.blobs[h] for h in hashes
    }))
    assert [p["image_url"] for p in expanded[1:]] == [_image("a"), _image("b"), _image("a")]


@pytest.mark.regression("BUG-131")
def test_short_bodies_and_anthropic_images_are_handled():
    assert compact_raw_body('[{"type":"text","text":"hello"}]').text == '[{"type":"text","text":"hello"}]'
    assert compact_raw_body("plain words").text == "plain words"
    data = base64.b64encode(b"x" * 3000).decode()
    raw = json.dumps([{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}])
    compacted = compact_raw_body(raw)
    (digest, (media_type, stored)), = compacted.blobs.items()
    assert media_type == "image/jpeg" and stored == data
    assert json.loads(compacted.text)[0]["source"]["data"] == BLOB_REF_PREFIX + digest


def _store(tmp_path) -> SQLiteStore:
    return SQLiteStore(str(tmp_path / "s.db"))


_ATTESTED = {
    "sender_actor_id": "actor", "source_message_id": "m1",
    "audience_conversation_id": "c", "origin_channel_id": "ch",
}


def _save(store, conv, turn_id, raw, group, *, sort_key=None, **provenance):
    store.save_canonical_turn(
        conv, -1, f"user {turn_id}", "", user_raw_content=raw,
        canonical_turn_id=turn_id, sort_key=sort_key or 1000.0 * (group + 1),
        turn_hash=f"h-{turn_id}", turn_group_number=group, **provenance,
    )


def _stored_raw(store, turn_id):
    return store._get_conn().execute(
        "SELECT user_raw_content FROM canonical_turns WHERE canonical_turn_id = ?", (turn_id,),
    ).fetchone()[0]


@pytest.mark.regression("BUG-131")
def test_saved_rows_share_one_copy_of_each_image(tmp_path):
    store = _store(tmp_path)
    _save(store, "c", "t1", _vast_like_body("first", [_image("a"), _image("b")]), 0)
    _save(store, "c", "t2", _vast_like_body("second", [_image("a"), _image("b"), _image("c")]), 1)
    conn = store._get_conn()
    assert conn.execute("SELECT COUNT(*) FROM canonical_raw_blobs").fetchone()[0] == 3
    stored = _stored_raw(store, "t2")
    assert len(stored) < 8000
    expanded = json.loads(expand_raw_body(stored, store.get_raw_blobs))
    assert expanded[3]["image_url"] == _image("c")


def _attest(store, user_id, assistant_id):
    import hashlib
    body_sha = hashlib.sha256(f"user {user_id}".encode()).hexdigest()
    store.save_canonical_turn(
        "c", -1, "", "reply", canonical_turn_id=assistant_id, sort_key=5000.0,
        turn_hash=f"h-{assistant_id}", turn_group_number=0,
    )
    now = "2026-10-18T00:00:00+00:00"
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO canonical_message_sources (tenant_id, agent_scope_id, platform, account_id,
           message_id, canonical_turn_id, assistant_canonical_turn_id, assistant_turn_hash,
           pair_version, audience_conversation_id, channel_id, guild_id, author_id,
           source_actor_id, transport_body_sha256, canonical_body_sha256, projection_version,
           canonical_turn_hash, observed_at, created_at)
           VALUES ('t','a','discord','acc','m1',?,?,?,1,'c','ch','g','au','actor','x',?,'1',?,?,?)""",
        (user_id, assistant_id, f"h-{assistant_id}", body_sha, f"h-{user_id}", now, now),
    )
    conn.commit()


def _insert_uncompacted(store, turn_id, raw):
    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET user_raw_content = ? WHERE canonical_turn_id = ?", (raw, turn_id),
    )
    conn.commit()


@pytest.mark.regression("BUG-131")
def test_attested_row_accepts_only_its_own_compaction(tmp_path):
    store = _store(tmp_path)
    raw = _vast_like_body("attested words", [_image("a")])
    _save(store, "c", "t1", "x", 0, **_ATTESTED)
    _insert_uncompacted(store, "t1", raw)
    _attest(store, "t1", "t1a")

    # Re-saving the same body rewrites it into its compacted form.
    _save(store, "c", "t1", raw, 0, **_ATTESTED)
    assert _stored_raw(store, "t1") == compact_raw_body(raw).text
    assert store._get_conn().execute("SELECT COUNT(*) FROM canonical_raw_compaction_grants").fetchone()[0] == 0

    # Any other change to the attested body is still refused.
    with pytest.raises(sqlite3.IntegrityError, match="attested canonical source fields are immutable"):
        _save(store, "c", "t1", _vast_like_body("different words", [_image("a")]), 0, **_ATTESTED)


@pytest.mark.regression("BUG-131")
def test_backfill_compacts_existing_rows_in_place(tmp_path):
    store = _store(tmp_path)
    bodies = {f"t{i}": _vast_like_body(f"words {i}", [_image("a"), _image(str(i))]) for i in range(4)}
    for i, (turn_id, raw) in enumerate(bodies.items()):
        _save(store, "c", turn_id, "x", i)
        _insert_uncompacted(store, turn_id, raw)
    _attest(store, "t0", "t0a")

    dry = store.compact_stored_raw_bodies("c", batch_size=2, dry_run=True)
    assert dry["rows_compacted"] == 4 and _stored_raw(store, "t1") == bodies["t1"]

    counts = store.compact_stored_raw_bodies("c", batch_size=2)
    assert counts["rows_compacted"] == 4
    assert counts["bytes_after"] < counts["bytes_before"] / 10
    for turn_id, raw in bodies.items():
        assert _stored_raw(store, turn_id) == compact_raw_body(raw).text
        assert VirtualContextEngine._sender_from_raw_content(_stored_raw(store, turn_id)) == "Kuw9239"
    assert store._get_conn().execute("SELECT COUNT(*) FROM canonical_raw_blobs").fetchone()[0] == 5
    assert store.compact_stored_raw_bodies("c")["rows_compacted"] == 0


@pytest.mark.regression("BUG-131")
def test_deleting_a_conversation_removes_only_its_unshared_images(tmp_path, monkeypatch):
    monkeypatch.setattr(raw_blobs, "GC_GRACE", timedelta(0))
    store = _store(tmp_path)
    _save(store, "c", "t1", _vast_like_body("one", [_image("shared"), _image("mine")]), 0)
    _save(store, "d", "u1", _vast_like_body("two", [_image("shared")]), 0)
    store.delete_conversation("c")
    remaining = {
        row[0] for row in store._get_conn().execute("SELECT data FROM canonical_raw_blobs").fetchall()
    }
    assert remaining == {_image("shared")}


@pytest.mark.regression("BUG-131")
def test_admin_command_compacts_a_conversation(tmp_path, capsys):
    import types
    from virtual_context.cli.main import cmd_admin_compact_raw_bodies

    db = tmp_path / "s.db"
    store = SQLiteStore(str(db))
    raw = _vast_like_body("words", [_image("a")])
    _save(store, "c", "t1", "x", 0)
    _insert_uncompacted(store, "t1", raw)
    store.close()
    args = types.SimpleNamespace(
        config=None, conversation_id="c", tenant_id="", all_convs_for_tenant=False,
        dry_run=False, limit=None, batch_size=5, storage_backend="sqlite",
        postgres_dsn=None, sqlite_path=str(db),
    )
    cmd_admin_compact_raw_bodies(args)
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["status"] == "ok" and report["rows_compacted"] == 1
    assert _stored_raw(SQLiteStore(str(db)), "t1") == compact_raw_body(raw).text
