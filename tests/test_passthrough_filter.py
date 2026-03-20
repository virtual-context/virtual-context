"""Tests for passthrough filter — only stub content should skip LLM summarization."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from virtual_context.engine import _is_stub_content


class TestPassthroughFilter:
    """Only stub content (media placeholders) should get passthrough.
    Real short messages must go through the LLM compactor."""

    def test_real_short_message_is_not_stub(self):
        assert not _is_stub_content("im good")
        assert not _is_stub_content("I see what you said")
        assert not _is_stub_content("What about me!")
        assert not _is_stub_content("Ok what else is going on?")
        assert not _is_stub_content("yes")
        assert not _is_stub_content("nah")

    def test_image_stub_is_stub(self):
        assert _is_stub_content("[image data removed - already processed by model]")
        assert _is_stub_content(
            "[media attached: /path/to/file.jpg (image/jpeg)] "
            "[image data removed - already processed by model]"
        )

    def test_empty_is_stub(self):
        assert _is_stub_content("")
        assert _is_stub_content("   ")

    def test_real_message_with_attachment_is_not_stub(self):
        assert not _is_stub_content(
            "[media attached: photo.jpg (image/jpeg)] "
            "What do you think of my blanket so far?"
        )

    def test_compactor_sends_short_real_messages_to_llm(self, tmp_path):
        """A 6-token segment with real content must NOT get passthrough."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config
        from virtual_context.types import Message, TaggedSegment

        now = datetime.now(timezone.utc)

        # Create a segment with real short content
        seg = TaggedSegment(
            id="seg-short",
            primary_tag="greeting",
            tags=["greeting"],
            messages=[
                Message(role="user", content="I see what you said"),
                Message(role="assistant", content="Makes sense!"),
            ],
            token_count=8,
            start_timestamp=now,
            end_timestamp=now,
            turn_count=1,
        )

        cfg = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "tag_generator": {"type": "keyword"},
        })
        engine = VirtualContextEngine(config=cfg)

        # Mock the compactor to track what gets sent to it
        from virtual_context.types import CompactionResult, SegmentMetadata
        mock_compactor = MagicMock()
        mock_compactor.compact.return_value = [CompactionResult(
            segment_id="seg-short",
            primary_tag="greeting",
            tags=["greeting"],
            summary="User acknowledged understanding",
            summary_tokens=4,
            full_text="I see what you said Makes sense!",
            original_tokens=8,
            messages=[{"role": "user", "content": "I see what you said"}],
            metadata=SegmentMetadata(turn_count=1, session_date=""),
            compression_ratio=0.5,
            timestamp=now,
            facts=[],
        )]
        mock_compactor.model_name = "test-model"
        engine._compaction._compactor = mock_compactor
        engine._tagging._compactor = mock_compactor

        results = engine._compaction._compact_and_store([seg], 2)

        # The compactor should have been called (NOT passthrough)
        assert mock_compactor.compact.called, (
            "Short real message should be sent to compactor, not passthrough"
        )

    def test_compactor_passthroughs_stub_content(self, tmp_path):
        """A segment with only stub content should get passthrough."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config
        from virtual_context.types import Message, TaggedSegment

        now = datetime.now(timezone.utc)

        seg = TaggedSegment(
            id="seg-stub",
            primary_tag="_stub",
            tags=["_stub"],
            messages=[
                Message(role="user", content="[image data removed - already processed by model]"),
                Message(role="assistant", content=""),
            ],
            token_count=10,
            start_timestamp=now,
            end_timestamp=now,
            turn_count=1,
        )

        cfg = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "tag_generator": {"type": "keyword"},
        })
        engine = VirtualContextEngine(config=cfg)

        mock_compactor = MagicMock()
        engine._compaction._compactor = mock_compactor
        engine._tagging._compactor = mock_compactor

        results = engine._compaction._compact_and_store([seg], 2)

        # Compactor should NOT have been called (passthrough)
        assert not mock_compactor.compact.called, (
            "Stub content should get passthrough, not sent to compactor"
        )
        # Result should exist with passthrough model
        assert len(results) == 1
