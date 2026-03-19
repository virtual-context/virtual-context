"""Tests for prev_context leak prevention in compactor."""

from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import CompactorConfig


class TestPrevContextNotInjectedForShortSegments:
    """prev_context should be suppressed for short segments (< 100 tokens)
    to prevent the LLM from incorporating unrelated context into the summary."""

    def test_short_segment_skips_prev_context(self):
        """A segment with < 100 tokens should NOT receive prev_context in its prompt."""
        from unittest.mock import MagicMock, patch
        from virtual_context.types import Message, TaggedSegment
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Segment B: short (< 100 tokens) about Ramadan
        seg_b = TaggedSegment(
            id="seg-b",
            primary_tag="ramadan",
            tags=["ramadan"],
            messages=[
                Message(role="user", content="Its too early to sleep and I'm waking up for sehri"),
                Message(role="assistant", content="What time is sehri tonight?"),
            ],
            token_count=25,
            start_timestamp=now,
            end_timestamp=now,
            turn_count=1,
        )

        # prev_context about crochet (from Segment A)
        crochet_context = (
            "User: I've been working on a filet crochet blanket. "
            "It's 27 inches square now with a diamond pattern in baby blue yarn. "
            "I taught myself left-handed crochet without any videos.\n\n"
            "Assistant: That's impressive! The diamond pattern in filet crochet "
            "requires careful counting."
        )

        # Mock the LLM provider to capture what prompt is sent
        mock_provider = MagicMock()
        mock_provider.complete.return_value = (
            '{"summary": "Discussing Ramadan sehri timing", "entities": [], '
            '"key_decisions": [], "action_items": [], "date_references": [], '
            '"refined_tags": ["ramadan"], "facts": []}',
            {"input_tokens": 100, "output_tokens": 50},
        )

        compactor = DomainCompactor(llm_provider=mock_provider, config=CompactorConfig())
        result = compactor._compact_one(seg_b, prev_context=crochet_context)

        # The LLM should have been called
        assert mock_provider.complete.called

        # Check the prompt sent to the LLM
        call_kwargs = mock_provider.complete.call_args
        prompt_sent = call_kwargs.kwargs.get("user", "") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else ""
        # For keyword calls, check both positional and keyword
        if not prompt_sent:
            # Try getting from positional args or kwargs
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "sehri" in arg:
                    prompt_sent = arg
                    break
            if not prompt_sent:
                prompt_sent = str(call_kwargs)

        # prev_context about crochet should NOT be in the prompt
        assert "filet crochet" not in prompt_sent, (
            f"prev_context leaked into prompt for short segment! "
            f"Prompt contains crochet content that should have been suppressed."
        )
        assert "diamond pattern" not in prompt_sent, (
            "prev_context crochet content leaked into short segment prompt"
        )

    def test_long_segment_keeps_prev_context(self):
        """A segment with >= 100 tokens should still receive prev_context."""
        from unittest.mock import MagicMock
        from virtual_context.types import Message, TaggedSegment
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Longer segment (> 100 tokens)
        long_content = " ".join(["discussing Ramadan traditions and fasting schedule"] * 10)
        seg = TaggedSegment(
            id="seg-long",
            primary_tag="ramadan",
            tags=["ramadan"],
            messages=[
                Message(role="user", content=long_content),
                Message(role="assistant", content="That's a wonderful tradition. " * 5),
            ],
            token_count=150,
            start_timestamp=now,
            end_timestamp=now,
            turn_count=1,
        )

        crochet_context = "User: Working on a crochet blanket"

        mock_provider = MagicMock()
        mock_provider.complete.return_value = (
            '{"summary": "Ramadan discussion", "entities": [], '
            '"key_decisions": [], "action_items": [], "date_references": [], '
            '"refined_tags": ["ramadan"], "facts": []}',
            {"input_tokens": 200, "output_tokens": 50},
        )

        compactor = DomainCompactor(llm_provider=mock_provider, config=CompactorConfig())
        result = compactor._compact_one(seg, prev_context=crochet_context)

        # The prompt SHOULD contain prev_context for long segments
        call_kwargs = mock_provider.complete.call_args
        prompt_text = str(call_kwargs)
        assert "crochet blanket" in prompt_text, (
            "prev_context should be included for segments >= 100 tokens"
        )
