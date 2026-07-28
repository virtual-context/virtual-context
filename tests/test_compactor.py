"""Tests for DomainCompactor (tag-based)."""

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import MockLLMProvider
from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import CompactorConfig, Message, TaggedSegment, TagPromptRule


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def compactor(mock_llm):
    return DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )


@pytest.fixture
def legal_segment(ts):
    return TaggedSegment(
        primary_tag="legal",
        tags=["legal", "court"],
        messages=[
            Message(role="user", content="What's the court filing deadline?", timestamp=ts),
            Message(role="assistant", content="The filing is due January 30.", timestamp=ts + timedelta(seconds=30)),
        ],
        token_count=50,
        start_timestamp=ts,
        end_timestamp=ts + timedelta(seconds=30),
        turn_count=1,
    )


def test_compact_single(compactor, legal_segment, mock_llm):
    results = compactor.compact([legal_segment])
    assert len(results) == 1
    assert results[0].primary_tag == "legal"
    assert results[0].summary == "Test summary"
    assert len(mock_llm.calls) == 1


def test_compact_preserves_metadata(compactor, legal_segment):
    results = compactor.compact([legal_segment])
    assert results[0].metadata.entities == ["entity1"]
    assert results[0].metadata.key_decisions == ["decision1"]


def test_compact_preserves_message_provenance_metadata(compactor, legal_segment):
    legal_segment.messages[0].metadata = {
        "sender": {"name": "BigTex"},
        "_vc_source_canonical_turn_ids": ["ct-user"],
    }

    result = compactor.compact([legal_segment])[0]

    assert result.messages[0]["metadata"] == legal_segment.messages[0].metadata


@pytest.mark.regression("BUG-004")
def test_compact_refined_tags(compactor, legal_segment):
    results = compactor.compact([legal_segment])
    assert "test-tag" in results[0].tags  # From mock LLM response


def test_compact_multiple(compactor):
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    segments = [
        TaggedSegment(
            primary_tag="legal",
            tags=["legal"],
            messages=[Message(role="user", content="Court case update")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
        TaggedSegment(
            primary_tag="medical",
            tags=["medical"],
            messages=[Message(role="user", content="Blood test results")],
            start_timestamp=ts,
            end_timestamp=ts,
        ),
    ]
    results = compactor.compact(segments)
    assert len(results) == 2


def test_format_conversation(compactor):
    ts = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
    messages = [
        Message(role="user", content="Hello", timestamp=ts),
        Message(role="assistant", content="Hi there", timestamp=ts),
    ]
    text = compactor._format_conversation(messages)
    assert "User (10:30): Hello" in text
    assert "Assistant (10:30): Hi there" in text


def test_parse_response_valid(compactor):
    result = compactor._parse_response('{"summary": "test", "entities": ["a"]}')
    assert result["summary"] == "test"


def test_parse_response_with_fences(compactor):
    result = compactor._parse_response('```json\n{"summary": "test"}\n```')
    assert result["summary"] == "test"


def test_parse_response_with_thinking(compactor):
    result = compactor._parse_response('<think>analyzing...</think>{"summary": "test"}')
    assert result["summary"] == "test"


def test_parse_response_fallback(compactor):
    result = compactor._parse_response("Just plain text summary")
    assert result["summary"] == "Just plain text summary"


def test_compact_retries_incomplete_json_summary(legal_segment):
    class RetryProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            if self.calls == 1:
                return "```json\n{", {}
            return '{"summary":"Recovered summary","refined_tags":[]}', {}

    provider = RetryProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor.compact([legal_segment])[0]

    assert provider.calls == 2
    assert result.summary == "Recovered summary"


def test_compact_uses_source_fallback_after_two_degenerate_summaries(legal_segment):
    class BrokenProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            return "```json\n{", {}

    provider = BrokenProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor.compact([legal_segment])[0]

    assert provider.calls == 2
    assert "User" in result.summary
    assert not result.summary.startswith("```")


def test_compact_retries_summary_longer_than_long_source(legal_segment):
    class ContextPollutingProvider:
        def __init__(self):
            self.calls = []

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls.append((system, user))
            if len(self.calls) == 1:
                imported = (
                    "The user accepted an installation recommendation from the previous "
                    "conversation and the assistant completed a detailed deployment. " * 10
                )
                return json.dumps({"summary": imported, "refined_tags": []}), {}
            return '{"summary":"They discussed a court deadline.","refined_tags":[]}', {}

    import json

    provider = ContextPollutingProvider()
    legal_segment.messages[0].content += " " + ("filing detail " * 40)
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor._compact_one(
        legal_segment,
        prev_context="A long unrelated discussion about software installation.",
    )

    assert len(provider.calls) == 2
    assert result.summary == "They discussed a court deadline."
    retry_system = provider.calls[1][0]
    assert "Do not import prior context" in retry_system
    assert "invert negation or intent" in retry_system


def test_compact_short_source_immediately_falls_back_for_oversized_summary(
    legal_segment,
):
    class PollutingProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            return json.dumps({"summary": "unrelated history " * 100}), {}

    import json

    provider = PollutingProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )

    result = compactor.compact([legal_segment])[0]

    assert provider.calls == 1
    assert result.summary in compactor._format_conversation(legal_segment.messages)
    assert len(result.summary) <= len(compactor._format_conversation(legal_segment.messages))


def test_default_prompt_requires_preserving_negation_and_intent():
    from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT

    assert "Preserve polarity, negation, intent" in DEFAULT_SUMMARY_PROMPT
    assert '"wants to remain infertile"' in DEFAULT_SUMMARY_PROMPT


def test_validator_no_longer_rejects_on_lexical_negation_overlap(ts):
    """The lexical negation-inversion heuristic is deliberately gone.

    It once caught a real inversion (this fixture's polarity flip is that
    incident), but in production it rejected 36-63% of faithful summaries
    of large or repetitive segments, because any source sentence with a
    negative marker sharing two stemmed terms with the summary counted as
    an inversion. Polarity protection now lives at the prompt layer, as it
    did for the months before the heuristic existed. This test pins the
    REMOVAL: the first response is accepted, no retry fires, and the trade
    is documented where the old test used to assert the opposite.
    """
    class InvertingProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            return (
                '{"summary":"Considered HCG.","refined_tags":[]}', {}
            )

    segment = TaggedSegment(
        primary_tag="hgh", tags=["hgh"],
        messages=[Message(
            role="user",
            content=(
                "I want to remain infertile and considered HCG for a stronger climax."
            ),
            timestamp=ts,
            metadata={"sender": {"name": "Reshi"}},
        )],
        token_count=20, start_timestamp=ts, end_timestamp=ts, turn_count=1,
    )
    provider = InvertingProvider()
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
    )
    result = compactor.compact([segment])[0]
    # A concise summary that omits the negated clause but shares stemmed
    # terms with the negative sentence is exactly what the old heuristic
    # rejected. It must now be accepted on the first call.
    assert provider.calls == 1, "no retry may fire on lexical overlap alone"
    assert result.summary == "Considered HCG."


def test_unusable_reason_names_degenerate_and_overshoot(ts):
    """Rejections carry their criterion, so the next validator defect is
    diagnosable from one log line instead of a week of archaeology."""
    assert DomainCompactor._unusable_reason("", "source text here") == "degenerate"
    assert DomainCompactor._unusable_reason(None, "source text here") == "degenerate"
    long_summary = "x" * 500
    assert DomainCompactor._unusable_reason(long_summary, "short") == "overshoot"
    assert DomainCompactor._unusable_reason("fine summary", "a much longer source text than the summary") is None

def test_custom_prompt_from_tag_rules():
    """Custom summary prompt should be used when tag matches a rule."""
    rules = [
        TagPromptRule(match="legal*", summary_prompt="Summarize legal matters carefully."),
    ]
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(),
        tag_rules=rules,
    )
    prompt = compactor._get_prompt_for_tags(["legal-case", "court"])
    assert prompt == "Summarize legal matters carefully."


def test_no_custom_prompt_for_unmatched_tags():
    rules = [
        TagPromptRule(match="legal*", summary_prompt="Summarize legal matters."),
    ]
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(),
        tag_rules=rules,
    )
    prompt = compactor._get_prompt_for_tags(["medical", "health"])
    assert prompt is None


def test_compact_tag_summaries_builds(mock_llm):
    """compact_tag_summaries builds summaries for tags with segments."""
    from virtual_context.types import StoredSummary, TagSummary

    compactor = DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(),
    )

    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    summaries = [
        StoredSummary(
            ref="seg-1", primary_tag="legal", tags=["legal"],
            summary="Case discussion", summary_tokens=20,
            created_at=now, start_timestamp=now, end_timestamp=now,
        ),
    ]

    result = compactor.compact_tag_summaries(
        cover_tags=["legal"],
        tag_to_summaries={"legal": summaries},
        tag_to_turns={"legal": [0, 1]},
        existing_tag_summaries={},
        max_turn=5,
    )
    assert len(result) == 1
    assert result[0].tag == "legal"
    assert result[0].covers_through_turn == 5
    assert len(mock_llm.calls) == 1


@pytest.mark.regression("BUG-003")
def test_compact_tag_summaries_skips_fresh(mock_llm):
    """compact_tag_summaries skips tags where existing summary is fresh."""
    from virtual_context.types import StoredSummary, TagSummary

    compactor = DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(),
    )

    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    summaries = [
        StoredSummary(
            ref="seg-1", primary_tag="legal", tags=["legal"],
            summary="Case discussion", summary_tokens=20,
            created_at=now, start_timestamp=now, end_timestamp=now,
        ),
    ]
    existing = TagSummary(
        tag="legal", summary="Already fresh", summary_tokens=20,
        covers_through_turn=10,  # >= max_turn
        created_at=now, updated_at=now,
    )

    result = compactor.compact_tag_summaries(
        cover_tags=["legal"],
        tag_to_summaries={"legal": summaries},
        tag_to_turns={"legal": [0, 1]},
        existing_tag_summaries={"legal": existing},
        max_turn=5,  # existing covers_through_turn (10) >= max_turn (5)
    )
    assert len(result) == 0  # Nothing to build
    assert len(mock_llm.calls) == 0  # No LLM calls


# ---------------------------------------------------------------------------
# TagSummary.description extraction
# ---------------------------------------------------------------------------


class TestTagSummaryDescription:
    """Tests for TagSummary.description field populated from rollup LLM response."""

    def test_description_extracted_from_rollup_response(self):
        """Mock LLM returns JSON with description — verify TagSummary.description is set."""
        from virtual_context.types import StoredSummary, TagSummary

        mock_llm = MockLLMProvider(
            response=(
                '{"summary": "Cycle tracking discussion", '
                '"description": "Sania\'s cycle tracking via Mira", '
                '"entities": ["Sania", "Mira"], '
                '"key_decisions": ["use Mira device"], '
                '"action_items": []}'
            )
        )
        compactor = DomainCompactor(
            llm_provider=mock_llm,
            config=CompactorConfig(),
        )

        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        summaries = [
            StoredSummary(
                ref="seg-1", primary_tag="cycle-tracking",
                tags=["cycle-tracking"],
                summary="Discussed Mira device for cycle tracking",
                summary_tokens=30,
                created_at=now, start_timestamp=now, end_timestamp=now,
            ),
        ]

        result = compactor.compact_tag_summaries(
            cover_tags=["cycle-tracking"],
            tag_to_summaries={"cycle-tracking": summaries},
            tag_to_turns={"cycle-tracking": [0, 1, 2]},
            existing_tag_summaries={},
            max_turn=5,
        )
        assert len(result) == 1
        assert result[0].description == "Sania's cycle tracking via Mira"

    def test_description_fallback_when_omitted(self):
        """Mock LLM returns JSON without description key — verify description == ''."""
        from virtual_context.types import StoredSummary, TagSummary

        mock_llm = MockLLMProvider(
            response=(
                '{"summary": "Legal case discussion", '
                '"entities": ["Judge Smith"], '
                '"key_decisions": ["file motion"], '
                '"action_items": []}'
            )
        )
        compactor = DomainCompactor(
            llm_provider=mock_llm,
            config=CompactorConfig(),
        )

        now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        summaries = [
            StoredSummary(
                ref="seg-1", primary_tag="legal",
                tags=["legal"],
                summary="Case discussion",
                summary_tokens=20,
                created_at=now, start_timestamp=now, end_timestamp=now,
            ),
        ]

        result = compactor.compact_tag_summaries(
            cover_tags=["legal"],
            tag_to_summaries={"legal": summaries},
            tag_to_turns={"legal": [0, 1]},
            existing_tag_summaries={},
            max_turn=5,
        )
        assert len(result) == 1
        assert result[0].description == ""


def test_faithful_summary_of_repetitive_machine_content_is_accepted():
    """The production failure shape, as a fixture.

    Large segments of near-duplicate machine records (home-automation
    event JSON) almost always contain negative markers ("no person
    detected"), and any faithful summary shares stemmed topic terms with
    those sentences while containing no negation token of its own. The
    removed heuristic rejected every such summary; measured in production
    at 36-63% of new segments, concentrated on the largest conversations.
    This fixture is shaped like the confirmed production specimen (real
    conversation content stays out of a public repository); the live
    specimen itself is re-verified operationally after deploy.
    """
    records = []
    for i in range(40):
        records.append(
            '{"before": {"id": "1785207%03d.77-x", "camera": "cam1", '
            '"label": "person", "stationary": false, "score": 0.58%02d}}' % (i, i)
        )
        if i % 7 == 3:
            records.append(
                "System: no person detected on camera cam1; event not retained, "
                "snapshot won't be stored."
            )
    source = "\n".join(records)
    summary = (
        "Processed a batch of camera person-detection events from cam1, "
        "with several events concluding as stationary detections and "
        "snapshots stored for the retained events."
    )
    assert len(source) > 4000, "fixture must be in the large-segment regime"
    assert DomainCompactor._unusable_reason(summary, source) is None, (
        "a faithful summary of repetitive machine content must be accepted"
    )


def test_five_faithful_rewordings_are_all_accepted():
    """Acceptance must not depend on wording luck.

    Under the removed heuristic, three of these five faithful rewordings
    were rejected and two passed, purely on whether the phrasing happened
    to include a negation token. All five must be accepted.
    """
    source = (
        "User: I don't want the pipeline analytics change yet.\n"
        "Assistant: Updated the staging pipeline configuration."
    )
    rewordings = [
        "Updated the staging pipeline configuration; the analytics change is on hold.",
        "Deferred the analytics change; staging updated.",
        "Held the analytics change; updated staging.",
        "Decided not to apply the analytics change yet; staging updated.",
        "The analytics change won't be applied yet; staging updated.",
    ]
    for summary in rewordings:
        assert DomainCompactor._unusable_reason(summary, source) is None, summary
