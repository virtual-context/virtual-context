"""Protected-window turns replayed from storage keep their images (BUG-132).

The protected window fills recent turns from storage when the payload lacks
them and rebuilt each as text, so an image a user sent in one of those turns
never reached the model.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import json

import pytest

from virtual_context.proxy.formats import (
    AnthropicFormat,
    GeminiFormat,
    OpenAIFormat,
    OpenAIResponsesFormat,
)
from virtual_context.proxy.replay_media import attach_replay_images
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Message

_A = base64.b64encode(b"a" * 3000).decode()
_B = base64.b64encode(b"b" * 3000).decode()


def _url(data: str) -> str:
    return f"data:image/png;base64,{data}"


def _db_user(turn_id: str, text: str) -> Message:
    return Message(role="user", content=text, metadata={
        "source": "db_recent", "canonical_turn_id": turn_id, "db_recent_group_key": turn_id,
    })


@pytest.mark.regression("BUG-132")
@pytest.mark.parametrize("fmt,find", [
    (AnthropicFormat(), lambda m: [b["source"]["data"] for b in m["content"] if b.get("type") == "image"]),
    (OpenAIFormat(), lambda m: [b["image_url"]["url"].split(",", 1)[1] for b in m["content"] if b.get("type") == "image_url"]),
    (OpenAIResponsesFormat(), lambda m: [b["image_url"].split(",", 1)[1] for b in m["content"] if b.get("type") == "input_image"]),
    (GeminiFormat(), lambda m: [p["inline_data"]["data"] for p in m["parts"] if "inline_data" in p]),
], ids=["anthropic", "openai", "responses", "gemini"])
def test_replayed_user_turn_serializes_its_images(fmt, find):
    message = fmt._serialize_replayed_message("user", "look", (("image/png", _A),))
    assert find(message) == [_A]
    assert fmt._serialize_replayed_message("assistant", "ok") == fmt._serialize_replayed_message("assistant", "ok", ())


@pytest.mark.regression("BUG-132")
def test_stored_images_are_attached_except_those_already_sent(tmp_path):
    store = SQLiteStore(str(tmp_path / "s.db"))
    raw = json.dumps([
        {"type": "input_text", "text": "which one?"},
        {"type": "input_image", "image_url": _url(_A)},
        {"type": "input_image", "image_url": _url(_B)},
    ])
    store.save_canonical_turn(
        "c", -1, "which one?", "", user_raw_content=raw, canonical_turn_id="t1",
        sort_key=1000.0, turn_hash="h1", turn_group_number=0,
    )
    assert "vc-blob:sha256:" in store._get_conn().execute(
        "SELECT user_raw_content FROM canonical_turns WHERE canonical_turn_id='t1'"
    ).fetchone()[0]
    body = {"input": [
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "now"},
            {"type": "input_image", "image_url": _url(_A)},
        ]},
    ]}
    replay = [_db_user("t1", "which one?"), Message(role="assistant", content="the blue one")]
    attached = attach_replay_images(replay, store, body, "c")
    assert attached[0].metadata["replay_images"] == [("image/png", _B)]
    assert "replay_images" not in replay[0].metadata
    assert attached[1] is replay[1]

    out, delivered = OpenAIResponsesFormat().inject_replayed_conversation_with_delivery(body, attached)
    assert len(delivered) == 2
    replayed_user = out["input"][0]
    assert [b["image_url"] for b in replayed_user["content"] if b["type"] == "input_image"] == [_url(_B)]


@pytest.mark.regression("BUG-132")
def test_lookup_failure_leaves_replay_text_only():
    class Broken:
        def get_canonical_turn_rows_by_id(self, *args, **kwargs):
            raise RuntimeError("down")

    replay = [_db_user("t1", "hi")]
    assert attach_replay_images(replay, Broken(), {"messages": []}, "c") is replay
