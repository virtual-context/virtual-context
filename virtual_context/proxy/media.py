"""Media compression and stubbing for the proxy pipeline.

Handles image blocks across all provider formats (Anthropic, OpenAI, Gemini).
Compresses on first sight, stores on disk, stubs outside the protected window.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image

from .formats import PayloadFormat

if TYPE_CHECKING:
    from ..core.store import ContextStore

logger = logging.getLogger(__name__)

MAX_WIDTH = 1024
MAX_HEIGHT = 1024
JPEG_QUALITY = 75
MIN_COMPRESS_BYTES = 10000  # don't bother compressing tiny images


def compress_media_in_payload(
    body: dict,
    fmt: PayloadFormat,
    store: "ContextStore | None",
    conversation_id: str,
    media_dir: str,
) -> tuple[dict, int]:
    """Scan payload for media blocks, compress and replace inline.

    Handles Anthropic, OpenAI Chat, and Gemini image formats.
    Writes compressed files to disk, records metadata in store.

    Returns (modified_body, images_compressed).
    """
    messages = fmt.get_messages(body)
    if not messages:
        return body, 0

    count = 0
    conv_media_dir = os.path.join(media_dir, conversation_id)

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            # Gemini uses "parts"
            content = msg.get("parts", [])
            if not isinstance(content, list):
                continue

        for block in content:
            if not isinstance(block, dict):
                continue

            b64_data, media_type, setter = _extract_media_data(block, fmt.name)
            if b64_data is None or len(b64_data) < MIN_COMPRESS_BYTES:
                continue

            ref = f"media_{hashlib.sha256(b64_data.encode()).hexdigest()[:12]}"
            file_path = os.path.join(conv_media_dir, f"{ref}.jpg")

            # Check cache -- already compressed?
            if store is not None:
                cached = store.get_media_output(conversation_id, ref)
                if cached is not None and os.path.exists(cached["file_path"]):
                    # Use cached compressed version
                    with open(cached["file_path"], "rb") as f:
                        compressed_b64 = base64.b64encode(f.read()).decode("ascii")
                    setter(block, compressed_b64, "image/jpeg")
                    logger.info("MEDIA-CACHE: ref=%s conv=%s", ref, conversation_id[:12])
                    count += 1
                    continue

            # Compress
            try:
                compressed_b64, width, height, comp_bytes = _compress_image(b64_data)
            except Exception as e:
                logger.warning("MEDIA-COMPRESS: failed for ref=%s: %s", ref, e)
                continue

            # Write to disk
            os.makedirs(conv_media_dir, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(compressed_b64))

            # Store metadata
            if store is not None:
                try:
                    store.store_media_output(
                        ref=ref,
                        conversation_id=conversation_id,
                        media_type="image/jpeg",
                        width=width,
                        height=height,
                        original_bytes=len(b64_data),
                        compressed_bytes=comp_bytes,
                        file_path=file_path,
                    )
                except Exception:
                    logger.warning("MEDIA-COMPRESS: failed to store metadata ref=%s", ref, exc_info=True)

            # Replace inline
            setter(block, compressed_b64, "image/jpeg")
            logger.info(
                "MEDIA-COMPRESS: ref=%s %dx%d %d->%d bytes (%.0f%% reduction)",
                ref, width, height, len(b64_data), comp_bytes,
                (1 - comp_bytes / len(b64_data)) * 100,
            )
            count += 1

    return body, count


def _extract_media_data(block: dict, format_name: str):
    """Extract base64 data from a media block. Returns (b64_data, media_type, setter_fn) or (None, None, None).

    The setter_fn takes (block, new_b64, new_media_type) and updates the block in place.
    """
    # Anthropic: {"type": "image", "source": {"type": "base64", "data": "...", "media_type": "..."}}
    if block.get("type") == "image":
        source = block.get("source", {})
        if isinstance(source, dict) and source.get("type") == "base64":
            return source.get("data"), source.get("media_type", ""), _set_anthropic_media
        return None, None, None

    # OpenAI: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    if block.get("type") == "image_url":
        url = block.get("image_url", {}).get("url", "")
        if url.startswith("data:") and ";base64," in url:
            header, b64 = url.split(";base64,", 1)
            media_type = header.replace("data:", "")
            return b64, media_type, _set_openai_media
        return None, None, None

    # Gemini: {"inline_data": {"mime_type": "...", "data": "..."}}
    if "inline_data" in block:
        inline = block["inline_data"]
        if isinstance(inline, dict):
            return inline.get("data"), inline.get("mime_type", ""), _set_gemini_media
        return None, None, None

    return None, None, None


def _set_anthropic_media(block, b64, media_type):
    block["source"]["data"] = b64
    block["source"]["media_type"] = media_type


def _set_openai_media(block, b64, media_type):
    block["image_url"]["url"] = f"data:{media_type};base64,{b64}"


def _set_gemini_media(block, b64, media_type):
    block["inline_data"]["data"] = b64
    block["inline_data"]["mime_type"] = media_type


def _compress_image(b64_data: str) -> tuple[str, int, int, int]:
    """Compress a base64 image to JPEG. Returns (compressed_b64, width, height, compressed_bytes)."""
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(BytesIO(img_bytes))

    if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
        img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    compressed = base64.b64encode(buf.getvalue()).decode("ascii")
    return compressed, img.width, img.height, len(buf.getvalue())
