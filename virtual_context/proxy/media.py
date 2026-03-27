"""Media compression and stubbing for the proxy pipeline.

Handles image blocks across all provider formats (Anthropic, OpenAI, Gemini).
Compresses on first sight, stores on disk, stubs outside the protected window.
"""

from __future__ import annotations

import base64
import hashlib
import json
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


def stub_media_by_position(
    body: dict,
    fmt: PayloadFormat,
    protected_recent_turns: int,
    **kwargs,
) -> tuple[dict, int]:
    """Replace media blocks outside the protected window with text stubs.

    Supports conditional intrusion into the protected zone via kwargs:
    - protected_intrusion_threshold (float): ratio of protected zone to context budget
      that triggers intrusion. 0.0 disables (default).
    - context_budget (int): total context budget in tokens. Required for intrusion.

    Returns (modified_body, stubs_created).
    """
    messages = fmt.get_messages(body)
    if not messages:
        return body, 0

    # Hard-protect last 4 messages (last 2 turns)
    _hard_protected = max(0, len(messages) - 4)
    # Soft-protect last N*2 messages (protected window)
    _soft_protected = max(0, len(messages) - protected_recent_turns * 2)

    # Conditional intrusion: if protected zone exceeds a percentage of the
    # context budget, allow stubbing into the protected zone except for
    # the last 2 turns (last 4 messages).
    _intrusion_threshold = kwargs.get("protected_intrusion_threshold", 0.0)
    _context_budget = kwargs.get("context_budget", 0)
    _intrusion_start = len(messages)  # default: no intrusion

    if _intrusion_threshold > 0 and _context_budget > 0 and protected_recent_turns > 2:
        # Estimate protected zone size
        _prot_bytes = 0
        for mi in range(_soft_protected, len(messages)):
            _prot_bytes += len(json.dumps(messages[mi], default=str))
        _prot_tokens = _prot_bytes // 4
        _prot_ratio = _prot_tokens / _context_budget if _context_budget else 0

        if _prot_ratio > _intrusion_threshold:
            # Allow stubbing inside protected zone except last 2 turns
            _intrusion_start = _hard_protected
            logger.info(
                "MEDIA_INTRUSION: protected zone %dt is %.0f%% of budget %dt "
                "(threshold %.0f%%) -- stubbing turns 3+ in protected window",
                _prot_tokens, _prot_ratio * 100, _context_budget,
                _intrusion_threshold * 100,
            )

    count = 0
    for mi, msg in enumerate(messages):
        # Never stub last 2 turns
        if mi >= _hard_protected:
            continue
        # Respect protected window unless intrusion applies
        if mi >= _soft_protected and mi >= _intrusion_start:
            # Inside protected zone but intrusion not active for this message
            continue
        if mi >= _soft_protected and mi < _intrusion_start:
            # Intrusion active: allow stubbing
            pass
        elif mi < _soft_protected:
            # Outside protected window: always stub
            pass
        else:
            continue

        content = msg.get("content", "")
        if not isinstance(content, list):
            content = msg.get("parts", [])
            if not isinstance(content, list):
                continue

        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue

            b64_data, media_type, _ = _extract_media_data(block, fmt.name)
            if b64_data is None or len(b64_data) < MIN_COMPRESS_BYTES:
                continue

            ref = f"media_{hashlib.sha256(b64_data.encode()).hexdigest()[:12]}"

            # Get dimensions from the compressed image
            try:
                img_bytes = base64.b64decode(b64_data)
                img = Image.open(BytesIO(img_bytes))
                w, h = img.width, img.height
                orig_kb = len(b64_data) // 1024
            except Exception:
                w, h = 0, 0
                orig_kb = len(b64_data) // 1024

            stub_text = (
                f"[Image ({w}x{h} {media_type or 'jpeg'}, originally {orig_kb}KB) "
                f"compressed and stored by virtual context.\n"
                f'To restore and uncompact full image in place: '
                f'{{"type": "tool_use", "name": "vc_restore_tool", '
                f'"input": {{"ref": "{ref}"}}}}]'
            )

            # Format-aware stub block construction
            if fmt.name == "gemini":
                # Gemini uses {"text": "..."} not {"type": "text", "text": "..."}
                content[bi] = {"text": stub_text}
            else:
                content[bi] = {"type": "text", "text": stub_text}
            logger.info("MEDIA-STUB: ref=%s msg=%d", ref, mi)
            count += 1

    return body, count


def build_media_restore_result(
    b64_data: str,
    media_type: str,
    width: int,
    height: int,
) -> list[dict]:
    """Build the content blocks for a media restore tool result."""
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        },
        {
            "type": "text",
            "text": f"Restored image ({width}x{height} {media_type}).",
        },
    ]


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
