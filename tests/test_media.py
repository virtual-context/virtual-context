import base64
import hashlib
import os
import tempfile
from io import BytesIO

import pytest
from PIL import Image

from virtual_context.proxy.formats import detect_format
from virtual_context.proxy.media import compress_media_in_payload
from virtual_context.storage.sqlite import SQLiteStore


class TestMediaOutputsStorage:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SQLiteStore(os.path.join(self.tmpdir, "test.db"))

    def test_store_and_get_media_output(self):
        self.store.store_media_output(
            ref="media_abc123",
            conversation_id="conv1",
            media_type="image/jpeg",
            width=1024,
            height=768,
            original_bytes=500000,
            compressed_bytes=50000,
            file_path="/data/tenants/t1/media/conv1/media_abc123.jpg",
        )
        result = self.store.get_media_output("conv1", "media_abc123")
        assert result is not None
        assert result["ref"] == "media_abc123"
        assert result["media_type"] == "image/jpeg"
        assert result["width"] == 1024
        assert result["file_path"] == "/data/tenants/t1/media/conv1/media_abc123.jpg"

    def test_get_media_output_not_found(self):
        result = self.store.get_media_output("conv1", "media_nonexistent")
        assert result is None

    def test_delete_conversation_removes_media(self):
        self.store.store_media_output(
            ref="media_abc123",
            conversation_id="conv1",
            media_type="image/jpeg",
            width=1024, height=768,
            original_bytes=500000, compressed_bytes=50000,
            file_path="/tmp/test.jpg",
        )
        self.store.delete_conversation("conv1")
        result = self.store.get_media_output("conv1", "media_abc123")
        assert result is None

    def test_delete_conversation_removes_media_files_on_disk(self):
        """Verify delete_conversation also removes media files from disk."""
        conv_media_dir = os.path.join(self.tmpdir, "media", "conv1")
        os.makedirs(conv_media_dir, exist_ok=True)
        fake_file = os.path.join(conv_media_dir, "media_abc123.jpg")
        with open(fake_file, "wb") as f:
            f.write(b"fake image data")
        self.store.store_media_output(
            ref="media_abc123",
            conversation_id="conv1",
            media_type="image/jpeg",
            width=1024, height=768,
            original_bytes=500000, compressed_bytes=50000,
            file_path=fake_file,
        )
        self.store.delete_conversation("conv1")
        assert not os.path.isdir(conv_media_dir)

    def test_composite_pk_allows_same_ref_different_conversations(self):
        for conv_id in ("conv1", "conv2"):
            self.store.store_media_output(
                ref="media_abc123",
                conversation_id=conv_id,
                media_type="image/jpeg",
                width=1024, height=768,
                original_bytes=500000, compressed_bytes=50000,
                file_path=f"/tmp/{conv_id}.jpg",
            )
        r1 = self.store.get_media_output("conv1", "media_abc123")
        r2 = self.store.get_media_output("conv2", "media_abc123")
        assert r1["file_path"] != r2["file_path"]


def _make_png_b64(width=2000, height=2000, color="red"):
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestCompressMediaInPayload:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SQLiteStore(os.path.join(self.tmpdir, "test.db"))
        self.media_dir = os.path.join(self.tmpdir, "media")

    def _fmt(self):
        return detect_format({"model": "claude-sonnet-4-6", "messages": []})

    def test_compresses_anthropic_image(self):
        b64 = _make_png_b64()
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": "what is this?"},
                ]},
            ],
        }
        fmt = self._fmt()
        result, count = compress_media_in_payload(body, fmt, store=None, conversation_id="c1", media_dir=self.media_dir)
        assert count == 1
        source = result["messages"][0]["content"][0]["source"]
        assert source["media_type"] == "image/jpeg"
        assert len(source["data"]) < len(b64)

    def test_skips_small_images(self):
        img = Image.new("RGB", (50, 50), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                ]},
            ],
        }
        fmt = self._fmt()
        result, count = compress_media_in_payload(body, fmt, store=None, conversation_id="c1", media_dir=self.media_dir)
        assert count == 0

    def test_uses_cached_version(self):
        """Verify that the store cache path is exercised on second compress call."""
        b64 = _make_png_b64()
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                ]},
            ],
        }
        fmt = self._fmt()
        # First call compresses and stores metadata
        result1, c1 = compress_media_in_payload(
            body, fmt, store=self.store, conversation_id="c1", media_dir=self.media_dir,
        )
        assert c1 == 1
        # Verify file was written
        ref = f"media_{hashlib.sha256(b64.encode()).hexdigest()[:12]}"
        file_path = os.path.join(self.media_dir, "c1", f"{ref}.jpg")
        assert os.path.exists(file_path)
        # Verify metadata was stored
        cached = self.store.get_media_output("c1", ref)
        assert cached is not None
        assert cached["file_path"] == file_path
        # Second call should hit the cache path (store lookup returns existing record)
        body2 = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                ]},
            ],
        }
        result2, c2 = compress_media_in_payload(
            body2, fmt, store=self.store, conversation_id="c1", media_dir=self.media_dir,
        )
        assert c2 == 1
        # Both calls should produce identical output JPEG data
        out1 = result1["messages"][0]["content"][0]["source"]["data"]
        out2 = result2["messages"][0]["content"][0]["source"]["data"]
        assert out1 == out2

    def test_compresses_openai_image(self):
        b64 = _make_png_b64(2000, 2000)
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        }
        fmt = detect_format(body)
        result, count = compress_media_in_payload(body, fmt, store=None, conversation_id="c1", media_dir=self.media_dir)
        assert count == 1

    def test_compresses_gemini_image(self):
        """Verify Gemini inline_data format is compressed correctly."""
        b64 = _make_png_b64(2000, 2000)
        body = {
            "contents": [
                {"role": "user", "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                    {"text": "what is this?"},
                ]},
            ],
        }
        fmt = detect_format(body)
        result, count = compress_media_in_payload(body, fmt, store=None, conversation_id="c1", media_dir=self.media_dir)
        assert count == 1
        inline = result["contents"][0]["parts"][0]["inline_data"]
        assert inline["mime_type"] == "image/jpeg"
        assert len(inline["data"]) < len(b64)

    def test_skips_url_images(self):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
                ]},
            ],
        }
        fmt = self._fmt()
        result, count = compress_media_in_payload(body, fmt, store=None, conversation_id="c1", media_dir=self.media_dir)
        assert count == 0
