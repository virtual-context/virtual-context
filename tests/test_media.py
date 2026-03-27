import os
import tempfile
import pytest
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
