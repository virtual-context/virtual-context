"""Tests for stub turn detection and passthrough handling."""

import re

import pytest

from virtual_context.types import Message


# --- Test the stub detection function ---

class TestIsStubTurn:
    """Turns with only attachment markers, image stubs, or boilerplate
    instructions should be detected as stubs and skip the LLM tagger."""

    def test_image_data_removed_is_stub(self):
        from virtual_context.engine import _is_stub_content
        assert _is_stub_content(
            "[image data removed - already processed by model]"
        )

    def test_media_attached_plus_image_removed_is_stub(self):
        from virtual_context.engine import _is_stub_content
        assert _is_stub_content(
            "[media attached: /data/tenants/abc/file_143.jpg (image/jpeg)] "
            "[image data removed - already processed by model]"
        )

    def test_media_attached_only_is_stub(self):
        from virtual_context.engine import _is_stub_content
        assert _is_stub_content(
            "[media attached: /path/to/document.pdf (application/pdf)]"
        )

    def test_image_stub_with_send_instructions_is_stub(self):
        from virtual_context.engine import _is_stub_content
        assert _is_stub_content(
            "[media attached: photo.jpg (image/jpeg)] "
            "[image data removed - already processed by model] "
            "To send an image back to the user, use the send_image tool."
        )

    def test_real_content_is_not_stub(self):
        from virtual_context.engine import _is_stub_content
        assert not _is_stub_content(
            "What do you think of my blanket so far?"
        )

    def test_real_content_with_attachment_is_not_stub(self):
        from virtual_context.engine import _is_stub_content
        assert not _is_stub_content(
            "[media attached: photo.jpg (image/jpeg)] "
            "What do you think of my blanket so far?"
        )

    def test_short_real_content_after_stripping_is_not_stub(self):
        from virtual_context.engine import _is_stub_content
        # "im good" is real content even if short
        assert not _is_stub_content("im good")

    def test_empty_string_is_stub(self):
        from virtual_context.engine import _is_stub_content
        # Empty/whitespace is always a stub (no content at all)
        assert _is_stub_content("")
        assert _is_stub_content("   ")

    def test_plain_short_message_is_not_stub(self):
        from virtual_context.engine import _is_stub_content
        # Short messages without stub patterns are real content
        assert not _is_stub_content("ok")
        assert not _is_stub_content("yes please")

    def test_only_whitespace_and_brackets_is_stub(self):
        from virtual_context.engine import _is_stub_content
        assert _is_stub_content("[file attached]")


# --- Test that stub turns skip the tagger in ingest_history ---

class TestStubTurnIngestion:
    """Stub turns should get _stub tag without calling the LLM tagger."""

    def test_ingest_stub_turn_gets_stub_tag(self, tmp_path):
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config

        cfg = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "tag_generator": {"type": "keyword"},
        })
        engine = VirtualContextEngine(config=cfg)

        pairs = [
            Message(role="user", content="What do you think of this?"),
            Message(role="assistant", content="Looks great!"),
            Message(
                role="user",
                content="[media attached: /path/to/image.jpg (image/jpeg)] [image data removed - already processed by model]",
            ),
            Message(role="assistant", content=""),
            Message(role="user", content="I want to plan a trip to Tokyo next month for cherry blossoms"),
            Message(role="assistant", content="Great choice! The sakura season peaks in late March."),
        ]

        engine.ingest_history(pairs)

        entries = engine._turn_tag_index.entries
        assert len(entries) == 3  # 3 turns total

        # Turn 1 (index position 1) should be the stub
        stub_entry = entries[1]
        assert stub_entry.tags == ["_stub"]
        assert stub_entry.primary_tag == "_stub"

        # Turn 0 and 2 should have real tags (not _stub)
        assert entries[0].primary_tag != "_stub"
        assert entries[2].primary_tag != "_stub"

    def test_stub_turn_preserves_timestamp(self, tmp_path):
        from datetime import datetime, timezone
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config

        cfg = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "tag_generator": {"type": "keyword"},
        })
        engine = VirtualContextEngine(config=cfg)

        ts = datetime(2026, 3, 17, 0, 35, tzinfo=timezone.utc)
        pairs = [
            Message(
                role="user",
                content="[image data removed - already processed by model]",
                timestamp=ts,
            ),
            Message(role="assistant", content="", timestamp=ts),
        ]

        engine.ingest_history(pairs)

        entries = engine._turn_tag_index.entries
        assert len(entries) == 1
        assert entries[0].tags == ["_stub"]
        assert entries[0].session_date != ""  # timestamp should propagate
