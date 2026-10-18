"""Images of protected-window turns replayed from storage.

A turn the protected window fills in from storage is rebuilt as text. Its
stored raw body still names the images the user sent with it (inline, or by
blob reference once compacted), so the replayed user message carries them
again, except images the outbound payload already holds.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
from dataclasses import replace

from ..core.raw_bodies import BLOB_REF_PREFIX, MIN_BLOB_CHARS

logger = logging.getLogger(__name__)

REPLAY_IMAGES_KEY = "replay_images"


def split_image(value: str, media_type: str = "") -> tuple[str, str] | None:
    """``(media_type, base64 data)`` of a data URL or bare base64 image value."""
    if value.startswith("data:"):
        header, _, data = value[5:].partition(",")
        kind, _, encoding = header.partition(";")
        if encoding != "base64" or not data:
            return None
        return kind or media_type, data
    try:
        base64.b64decode(value[:64] + "=" * (-len(value[:64]) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return media_type, value


def _image_key(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _inline_images(value, out: list[tuple[str, str]], *, media_type: str = "") -> None:
    """Collect inline images (``(media_type, data)``) found anywhere in *value*."""
    if isinstance(value, dict):
        kind = str(value.get("media_type") or value.get("mime_type") or value.get("mimeType") or media_type)
        for key, item in value.items():
            if isinstance(item, str) and len(item) >= MIN_BLOB_CHARS and (
                item.startswith("data:image/") or key == "data"
            ):
                image = split_image(item, kind)
                if image and (image[0] or "").startswith("image/"):
                    out.append(image)
            else:
                _inline_images(item, out, media_type=kind)
    elif isinstance(value, list):
        for item in value:
            _inline_images(item, out, media_type=media_type)


def _stored_images(raw: str | None, load_blobs) -> list[tuple[str, str]]:
    """Images named by a stored raw body, inline or by blob reference."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    refs: list[str] = []

    def collect_refs(value):
        if isinstance(value, str) and value.startswith(BLOB_REF_PREFIX):
            refs.append(value[len(BLOB_REF_PREFIX):])
        elif isinstance(value, dict):
            for item in value.values():
                collect_refs(item)
        elif isinstance(value, list):
            for item in value:
                collect_refs(item)

    collect_refs(parsed)
    images: list[tuple[str, str]] = []
    _inline_images(parsed, images)
    if refs:
        blobs = load_blobs(set(refs))
        for digest in refs:
            stored = blobs.get(digest)
            if stored is None:
                continue
            image = split_image(stored[1], stored[0])
            if image and (image[0] or "").startswith("image/"):
                images.append(image)
    return images


def attach_replay_images(replay_messages: list, store, body: dict, conversation_id: str) -> list:
    """Replay messages whose storage-filled user turns carry their images.

    Returns copies of the affected messages with ``metadata["replay_images"]``
    set; other messages are returned as they are. A lookup failure leaves the
    messages text-only.
    """
    targets = [
        index for index, message in enumerate(replay_messages)
        if getattr(message, "role", "") == "user"
        and isinstance(getattr(message, "metadata", None), dict)
        and message.metadata.get("source") == "db_recent"
        and message.metadata.get("canonical_turn_id")
    ]
    if not targets or store is None:
        return replay_messages
    try:
        rows = store.get_canonical_turn_rows_by_id(
            [(conversation_id, replay_messages[i].metadata["canonical_turn_id"]) for i in targets],
            internal_validation=True,
        )
        present: list[tuple[str, str]] = []
        _inline_images(body, present)
        seen = {_image_key(data) for _kind, data in present}
        out = list(replay_messages)
        for index in targets:
            message = replay_messages[index]
            row = rows.get((conversation_id, message.metadata["canonical_turn_id"]))
            images = []
            for media_type, data in _stored_images(getattr(row, "user_raw_content", None), store.get_raw_blobs):
                key = _image_key(data)
                if key in seen:
                    continue
                seen.add(key)
                images.append((media_type, data))
            if images:
                out[index] = replace(message, metadata={**message.metadata, REPLAY_IMAGES_KEY: images})
        return out
    except Exception:
        logger.warning("Replay images unavailable for conv=%s", conversation_id[:12], exc_info=True)
        return replay_messages
