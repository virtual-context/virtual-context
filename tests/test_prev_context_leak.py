"""Tests for prev_context structural separation in compactor prompts."""

from unittest.mock import MagicMock
from datetime import datetime, timezone

from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import CompactorConfig, Message, TaggedSegment


class TestPrevContextXMLStructure:
    """prev_context must be wrapped in XML tags to prevent LLM from
    incorporating context content into the segment summary."""

    def test_prev_context_wrapped_in_xml_tags(self):
        """The prompt sent to the LLM must use <context_for_pronoun_resolution_only>
        and <segment_to_summarize> XML tags for structural separation."""
        now = datetime.now(timezone.utc)

        seg = TaggedSegment(
            id="seg-ramadan",
            primary_tag="ramadan",
            tags=["ramadan"],
            messages=[
                Message(role="user", content="Its too early to sleep and I'm waking up for sehri"),
                Message(role="assistant", content="Right, Ramadan. That sehri alarm will wreck your sleep score."),
            ],
            token_count=40,
            start_timestamp=now,
            end_timestamp=now,
            turn_count=1,
        )

        crochet_context = (
            "Sania (15:26): I'm at 2.5 skeins in. 30 inches. The diamond pattern is tricky.\n\n"
            "Assistant (15:26): Trust your gut. Grab one more skein."
        )

        mock_provider = MagicMock()
        mock_provider.complete.return_value = (
            '{"summary": "Discussing Ramadan sehri", "entities": [], '
            '"key_decisions": [], "action_items": [], "date_references": [], '
            '"refined_tags": ["ramadan"], "facts": []}',
            {"input_tokens": 100, "output_tokens": 50},
        )

        compactor = DomainCompactor(llm_provider=mock_provider, config=CompactorConfig())
        compactor._compact_one(seg, prev_context=crochet_context)

        # Extract the user prompt sent to the LLM
        call_args = mock_provider.complete.call_args
        user_prompt = call_args.kwargs.get("user", "")

        # Must contain XML tags
        assert "<context_for_pronoun_resolution_only>" in user_prompt, (
            "prev_context must be wrapped in <context_for_pronoun_resolution_only> XML tags"
        )
        assert "</context_for_pronoun_resolution_only>" in user_prompt
        assert "<segment_to_summarize>" in user_prompt, (
            "segment text must be wrapped in <segment_to_summarize> XML tags"
        )
        assert "</segment_to_summarize>" in user_prompt

        # The crochet content must be INSIDE the context tags, not loose
        ctx_start = user_prompt.index("<context_for_pronoun_resolution_only>")
        ctx_end = user_prompt.index("</context_for_pronoun_resolution_only>")
        seg_start = user_prompt.index("<segment_to_summarize>")
        seg_end = user_prompt.index("</segment_to_summarize>")

        ctx_block = user_prompt[ctx_start:ctx_end]
        seg_block = user_prompt[seg_start:seg_end]

        assert "skeins" in ctx_block, "crochet content should be inside context tags"
        assert "sehri" in seg_block, "segment content should be inside segment tags"
        assert "skeins" not in seg_block, "crochet content must NOT be inside segment tags"

    def test_system_message_reinforces_xml_boundary(self):
        """The system message must instruct the LLM to only summarize
        content inside <segment_to_summarize> tags."""
        now = datetime.now(timezone.utc)

        seg = TaggedSegment(
            id="seg-test",
            primary_tag="test",
            tags=["test"],
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
            token_count=10,
            start_timestamp=now,
            end_timestamp=now,
            turn_count=1,
        )

        mock_provider = MagicMock()
        mock_provider.complete.return_value = (
            '{"summary": "Greeting", "entities": [], "key_decisions": [], '
            '"action_items": [], "date_references": [], "refined_tags": []}',
            {"input_tokens": 50, "output_tokens": 20},
        )

        compactor = DomainCompactor(llm_provider=mock_provider, config=CompactorConfig())
        compactor._compact_one(seg, prev_context="some prior context")

        call_args = mock_provider.complete.call_args
        system_prompt = call_args.kwargs.get("system", "")

        assert "segment_to_summarize" in system_prompt, (
            "System message must reference <segment_to_summarize> tags"
        )
        assert "NEVER" in system_prompt or "never" in system_prompt, (
            "System message must instruct LLM to NEVER include context content"
        )

    def test_no_prev_context_no_xml_tags(self):
        """When there's no prev_context, XML tags should not appear."""
        now = datetime.now(timezone.utc)

        seg = TaggedSegment(
            id="seg-no-ctx",
            primary_tag="test",
            tags=["test"],
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            token_count=10,
            start_timestamp=now,
            end_timestamp=now,
            turn_count=1,
        )

        mock_provider = MagicMock()
        mock_provider.complete.return_value = (
            '{"summary": "Greeting", "entities": [], "key_decisions": [], '
            '"action_items": [], "date_references": [], "refined_tags": []}',
            {"input_tokens": 50, "output_tokens": 20},
        )

        compactor = DomainCompactor(llm_provider=mock_provider, config=CompactorConfig())
        compactor._compact_one(seg, prev_context="")

        call_args = mock_provider.complete.call_args
        user_prompt = call_args.kwargs.get("user", "")

        assert "<context_for_pronoun_resolution_only>" not in user_prompt, (
            "No XML context tags when prev_context is empty"
        )
