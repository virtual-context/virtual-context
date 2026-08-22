"""Tests for DomainCompactor (tag-based)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import MockLLMProvider
from virtual_context.core.compactor import (
    DomainCompactor,
    TAG_SUMMARY_ROLLUP_PROMPT,
    _format_tag_rollup_source,
)
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    AUTHOR_ROLE_REQUESTER,
    AUTHOR_ROLE_SUBJECT,
    SOURCE_CANONICAL_TURN_IDS_KEY,
    ActorRoster,
    CompactorConfig,
    FactLane,
    Message,
    SegmentMetadata,
    StoredSummary,
    TaggedSegment,
    TagPromptRule,
    TagSummary,
)


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


def _named_roster(*labels: str, complete: bool = True) -> ActorRoster:
    actors = {f"actor:discord:{index}" for index, _ in enumerate(labels, 1)}
    lanes = [
        FactLane(
            role=AUTHOR_ROLE_REQUESTER,
            text=f"source words from {label}",
            actor_id=f"actor:discord:{index}",
            speaker_label=label,
        )
        for index, label in enumerate(labels, 1)
    ]
    return ActorRoster(
        actor_ids=actors,
        labels={
            label.casefold(): {f"actor:discord:{index}"}
            for index, label in enumerate(labels, 1)
        },
        complete=complete,
        lanes=lanes,
    )


class _SequenceProvider:
    def __init__(self, *responses: object):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, max_tokens: int):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        response = self.responses[index]
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, str)
        return response, {}


def _stored_summary(
    text: str,
    *,
    ref: str = "seg-1",
    label: str = "",
    identity: str = "",
    scope: str = "scope-proof",
) -> StoredSummary:
    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    metadata = SegmentMetadata(
        source_speaker_labels=[label] if label else [],
        source_speaker_identity_count=1 if label else 0,
        source_speaker_identity_fingerprint=(identity or label) if label else "",
        source_audience_fingerprint=scope if label else "",
    )
    return StoredSummary(
        ref=ref,
        primary_tag="medical",
        tags=["medical"],
        summary=text,
        summary_tokens=max(1, len(text) // 4),
        metadata=metadata,
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    )


def _scoped_user_message(
    content: str,
    *,
    canonical_id: str,
    actor: str,
    label: str,
    audience: str = "audience:guild:1",
    channel: str = "channel:medical",
) -> Message:
    return Message(
        role="user",
        content=content,
        metadata={
            "sender": {"name": label},
            SOURCE_CANONICAL_TURN_IDS_KEY: [canonical_id],
        },
        source_actor_id=actor,
        source_audience_conversation_id=audience,
        source_origin_channel_id=channel,
        source_audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
    )


@pytest.fixture
def legal_segment(ts):
    return TaggedSegment(
        primary_tag="legal",
        tags=["legal", "court"],
        messages=[
            Message(
                role="user",
                content="What's the court filing deadline?",
                timestamp=ts,
                metadata={SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"]},
                source_actor_id="actor:discord:1",
                source_audience_conversation_id="audience:guild:1",
                source_origin_channel_id="channel:legal",
                source_audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
            ),
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


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
def test_named_roster_identity_contract_is_post_template_and_never_exposes_actor_ids(
    legal_segment,
    code_mode,
    custom_prompt,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster("BigTex", "Optics"),
    )

    assert "SUMMARY SPEAKER IDENTITY CONTRACT" in request.prompt
    assert '["BigTex", "Optics"]' in request.prompt
    assert "actor:discord:" not in request.prompt
    if custom_prompt:
        assert request.prompt.index("Custom legal summary.") \
            < request.prompt.index("SUMMARY SPEAKER IDENTITY CONTRACT")


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
@pytest.mark.parametrize(
    "unsafe_label",
    [
        "actor:discord:1",
        "User.",
        "(You)",
        "U\u200bser",
        "actor\u2060:discord:1",
    ],
    ids=[
        "actor-id",
        "decorated-generic",
        "decorated-pronoun",
        "zw-generic",
        "zw-actor-id",
    ],
)
def test_unsafe_roster_label_never_enters_any_segment_prompt(
    legal_segment,
    code_mode,
    custom_prompt,
    unsafe_label,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )
    legal_segment.messages[0].metadata = {
        "sender": {"name": unsafe_label},
        SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"],
    }

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster(unsafe_label),
    )

    assert "SUMMARY SPEAKER IDENTITY CONTRACT" in request.prompt
    assert "are: []." in request.prompt
    assert unsafe_label not in request.prompt
    assert "Source (" in request.prompt
    assert "What's the court filing deadline?" in request.prompt


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
def test_compatibility_equivalent_distinct_actor_labels_collide_in_every_segment_prompt(
    legal_segment,
    code_mode,
    custom_prompt,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )
    legal_segment.messages = [
        Message(
            role="user",
            content="first source text",
            metadata={"sender": {"name": "Alex"}},
            source_actor_id="actor:discord:alex-one",
        ),
        Message(
            role="user",
            content="second source text",
            metadata={"sender": {"name": "Ａｌｅｘ"}},
            source_actor_id="actor:discord:alex-two",
        ),
    ]

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster("Alex", "Ａｌｅｘ"),
    )

    contract = request.prompt.split("SUMMARY SPEAKER IDENTITY CONTRACT", 1)[1]
    assert "[]" in contract
    assert "Alex" not in contract
    assert "Ａｌｅｘ" not in contract
    transcript = request.prompt.split("SUMMARY SPEAKER IDENTITY CONTRACT", 1)[0]
    assert "Alex:" not in transcript
    assert "Ａｌｅｘ:" not in transcript
    assert "Source: first source text" in transcript
    assert "Source: second source text" in transcript


@pytest.mark.parametrize(
    ("code_mode", "custom_prompt"),
    [(False, False), (True, False), (False, True)],
    ids=["generic", "code", "custom"],
)
def test_valid_unicode_roster_labels_enter_every_segment_prompt(
    legal_segment,
    code_mode,
    custom_prompt,
):
    rules = (
        [TagPromptRule(match="legal*", summary_prompt="Custom legal summary.")]
        if custom_prompt else []
    )
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(code_mode=code_mode),
        tag_rules=rules,
    )

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=_named_roster("BigTex", "Renée", "李雷"),
    )

    assert '["BigTex", "Renée", "李雷"]' in request.prompt


def test_reply_subject_is_counted_and_both_exact_source_labels_are_retained(
    legal_segment,
):
    roster = ActorRoster(
        # ActorRoster.actor_ids intentionally mirrors the production requester
        # set; compaction proof must still count the subject lane below.
        actor_ids={"actor:discord:1"},
        labels={"bigtex": {"actor:discord:1"}},
        complete=True,
        reply_bearing=True,
        lanes=[
            FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text="What do you think?",
                actor_id="actor:discord:1",
                speaker_label="BigTex",
            ),
            FactLane(
                role=AUTHOR_ROLE_SUBJECT,
                text="I stopped the medication.",
                actor_id="actor:discord:2",
                speaker_label="Optics",
            ),
        ],
    )
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex asked about Optics stopping the medication.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=roster,
    )
    result = compactor._compact_one(legal_segment, roster=roster)

    assert '["BigTex", "Optics"]' in request.prompt
    assert "actor:discord:" not in request.prompt
    assert result.metadata.source_speaker_labels == ["BigTex", "Optics"]
    assert result.metadata.source_speaker_identity_count == 2
    assert result.summary == (
        "[summary withheld: speaker attribution is unresolved; retrieve exact "
        "source turns before making a person-specific claim]"
    )


def test_incomplete_or_partially_named_roster_gets_fail_closed_empty_contract(
    compactor,
    legal_segment,
):
    incomplete = _named_roster("BigTex", complete=False)
    partially_named = _named_roster("BigTex", "")

    for roster in (incomplete, partially_named, None):
        prompt = compactor.build_segment_summary_request(
            legal_segment, roster=roster,
        ).prompt
        assert "SUMMARY SPEAKER IDENTITY CONTRACT" in prompt
        assert "are: []." in prompt
        assert "actor:discord:" not in prompt


def test_colliding_display_label_does_not_authorize_named_attribution(
    compactor,
    legal_segment,
):
    roster = ActorRoster(
        actor_ids={"actor:discord:1", "actor:discord:2"},
        labels={"alex": {"actor:discord:1", "actor:discord:2"}},
        complete=True,
        lanes=[
            FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text="first disclosure",
                actor_id="actor:discord:1",
                speaker_label="Alex",
            ),
            FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text="second disclosure",
                actor_id="actor:discord:2",
                speaker_label="Alex",
            ),
        ],
    )

    request = compactor.build_segment_summary_request(
        legal_segment,
        roster=roster,
    )

    contract = request.prompt.split("SUMMARY SPEAKER IDENTITY CONTRACT", 1)[1]
    assert "[]" in contract
    assert "actor:discord:" not in request.prompt


def test_named_roster_ambiguous_summary_retries_then_accepts_exact_label(
    legal_segment,
):
    provider = _SequenceProvider(
        json.dumps({"summary": "The user stopped tesamorelin."}),
        json.dumps({"summary": "BigTex stopped tesamorelin."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 2
    assert result.summary == "BigTex stopped tesamorelin."
    assert result.metadata.source_speaker_labels == ["BigTex"]
    assert result.metadata.source_speaker_identity_count == 1
    assert len(result.metadata.source_speaker_identity_fingerprint) == 64
    assert "actor:discord:" not in result.metadata.source_speaker_identity_fingerprint
    assert len(result.metadata.source_audience_fingerprint) == 64
    assert "audience:guild:1" not in result.metadata.source_audience_fingerprint
    assert "ambiguous generic human referent" in provider.calls[1]["system"]


def test_ambiguous_summary_without_roster_still_retries_fail_closed(
    legal_segment,
):
    provider = _SequenceProvider(
        json.dumps({"summary": "A member stopped tesamorelin."}),
        json.dumps({"summary": "Tesamorelin cessation was discussed."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(legal_segment, roster=None)

    assert len(provider.calls) == 2
    assert result.summary == "Tesamorelin cessation was discussed."
    assert "are: []." in provider.calls[0]["user"]


def test_named_roster_second_ambiguous_summary_uses_named_source_fallback(
    legal_segment,
):
    legal_segment.messages[0].metadata = {
        "sender": {"name": "BigTex"},
        SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"],
    }
    provider = _SequenceProvider(
        json.dumps({"summary": "The user's tesamorelin use continued."}),
        json.dumps({"summary": "This person still uses tesamorelin."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 2
    assert "BigTex" in result.summary
    assert "court filing deadline" in result.summary
    assert "This person" not in result.summary


def test_segment_identity_retry_failure_uses_source_fallback(legal_segment):
    legal_segment.messages[0].metadata = {
        "sender": {"name": "BigTex"},
        SOURCE_CANONICAL_TURN_IDS_KEY: ["ct-legal"],
    }
    provider = _SequenceProvider(
        json.dumps({"summary": "The user stopped tesamorelin."}),
        RuntimeError("retry provider failure"),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 2
    assert "BigTex" in result.summary
    assert "court filing deadline" in result.summary
    assert "The user stopped tesamorelin" not in result.summary


def test_named_roster_does_not_reject_technical_user_compound(legal_segment):
    provider = _SequenceProvider(
        json.dumps({"summary": "The user interface was updated."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=True))

    result = compactor._compact_one(
        legal_segment,
        roster=_named_roster("BigTex"),
    )

    assert len(provider.calls) == 1
    assert result.summary == "The user interface was updated."


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


def test_preceding_context_never_crosses_proved_actor_boundary():
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    first = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "BigTex stopped tesamorelin.",
            canonical_id="ct-bigtex",
            actor="actor:discord:1",
            label="BigTex",
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )
    second = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "Kuw9239 discussed recovery.",
            canonical_id="ct-kuw",
            actor="actor:discord:2",
            label="Kuw9239",
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )

    def roster(actor: str, label: str) -> ActorRoster:
        return ActorRoster(
            actor_ids={actor},
            labels={label.casefold(): {actor}},
            complete=True,
            lanes=[FactLane(
                role=AUTHOR_ROLE_REQUESTER,
                text=f"source words from {label}",
                actor_id=actor,
                speaker_label=label,
            )],
        )

    provider = _SequenceProvider(
        json.dumps({"summary": "Named source statement."}),
    )
    compactor = DomainCompactor(
        provider,
        CompactorConfig(code_mode=False, max_concurrent_summaries=1),
    )
    compactor.compact(
        [first, second],
        actor_rosters_by_segment={
            first.id: roster("actor:discord:1", "BigTex"),
            second.id: roster("actor:discord:2", "Kuw9239"),
        },
    )

    second_prompt = next(
        call["user"] for call in provider.calls
        if "Kuw9239 discussed recovery." in call["user"]
    )
    assert "BigTex stopped tesamorelin." not in second_prompt
    assert "context_for_pronoun_resolution_only" not in second_prompt


@pytest.mark.parametrize(
    ("first_channel", "second_audience", "second_channel", "expect_prior"),
    [
        ("channel:medical", "audience:guild:1", "channel:medical", True),
        ("channel:medical", "audience:guild:2", "channel:medical", False),
        ("channel:medical", "audience:guild:1", "channel:other", False),
        ("", "audience:guild:1", "", True),
    ],
    ids=["same-scope", "cross-audience", "cross-channel", "same-dm-audience"],
)
def test_preceding_context_requires_same_actor_and_exact_audience_channel(
    first_channel,
    second_audience,
    second_channel,
    expect_prior,
):
    ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    actor = "actor:discord:1"
    first = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "BigTex stopped tesamorelin.",
            canonical_id="ct-1",
            actor=actor,
            label="BigTex",
            channel=first_channel,
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )
    second = TaggedSegment(
        primary_tag="medical",
        tags=["medical"],
        messages=[_scoped_user_message(
            "BigTex discussed recovery.",
            canonical_id="ct-2",
            actor=actor,
            label="BigTex",
            audience=second_audience,
            channel=second_channel,
        )],
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )
    roster = _named_roster("BigTex")
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex discussed recovery.",
    }))
    compactor = DomainCompactor(
        provider,
        CompactorConfig(code_mode=False, max_concurrent_summaries=1),
    )

    results = compactor.compact(
        [first, second],
        actor_rosters_by_segment={first.id: roster, second.id: roster},
    )

    second_prompt = next(
        call["user"] for call in provider.calls
        if "BigTex discussed recovery." in call["user"]
    )
    assert ("BigTex stopped tesamorelin." in second_prompt) is expect_prior
    assert (
        "context_for_pronoun_resolution_only" in second_prompt
    ) is expect_prior
    assert results[1].summary == "BigTex discussed recovery."
    assert len(results[1].metadata.source_audience_fingerprint) == 64


def test_format_conversation(compactor):
    ts = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
    messages = [
        Message(role="user", content="Hello", timestamp=ts),
        Message(role="assistant", content="Hi there", timestamp=ts),
    ]
    text = compactor._format_conversation(messages)
    assert "Source (10:30): Hello" in text
    assert "Assistant (10:30): Hi there" in text


@pytest.mark.parametrize(
    ("first_label", "second_label"),
    [
        ("Alex", "Ａｌｅｘ"),
        ("Alex", "A\u200blex"),
    ],
    ids=["fullwidth", "default-ignorable"],
)
def test_format_conversation_neutralizes_normalized_label_collisions(
    compactor,
    first_label,
    second_label,
):
    messages = [
        Message(
            role="user",
            content="first source text",
            metadata={"sender": {"name": first_label}},
            source_actor_id="actor:discord:alex-one",
        ),
        Message(
            role="user",
            content="second source text",
            metadata={"sender": {"name": second_label}},
            source_actor_id="actor:discord:alex-two",
        ),
    ]

    text = compactor._format_conversation(messages)

    assert text == "Source: first source text\n\nSource: second source text"
    assert first_label not in text
    assert second_label not in text
    assert "actor:discord:" not in text


def test_format_conversation_preserves_distinct_safe_unicode_labels(compactor):
    labels = ["BigTex", "Renée", "李雷"]
    messages = [
        Message(
            role="user",
            content=f"source text {index}",
            metadata={"sender": {"name": label}},
            source_actor_id=f"actor:discord:{index}",
        )
        for index, label in enumerate(labels, 1)
    ]

    text = compactor._format_conversation(messages)

    assert text == (
        "BigTex: source text 1\n\n"
        "Renée: source text 2\n\n"
        "李雷: source text 3"
    )
    assert "actor:discord:" not in text


def test_format_conversation_neutralizes_generic_and_actor_id_labels(compactor):
    messages = [
        Message(
            role="user",
            content="generic source text",
            metadata={"sender": {"name": "User."}},
            source_actor_id="actor:discord:generic",
        ),
        Message(
            role="user",
            content="actor id source text",
            metadata={"sender": {"name": "Kuw9239"}},
            source_actor_id="Kuw9239",
        ),
    ]

    text = compactor._format_conversation(messages)

    assert text == "Source: generic source text\n\nSource: actor id source text"
    assert "User." not in text
    assert "Kuw9239" not in text


def test_format_conversation_missing_actor_poison_is_label_wide(compactor):
    messages = [
        Message(
            role="user",
            content="proved source text",
            metadata={"sender": {"name": "BigTex"}},
            source_actor_id="actor:discord:bigtex",
        ),
        Message(
            role="user",
            content="unproved source text",
            metadata={"sender": {"name": "Big\u200bTex"}},
        ),
    ]

    text = compactor._format_conversation(messages)

    assert text == "Source: proved source text\n\nSource: unproved source text"
    assert "BigTex" not in text
    assert "Big\u200bTex" not in text
    assert "actor:discord:" not in text


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
    assert result.summary == (
        "[summary withheld: speaker attribution is unresolved; retrieve exact "
        "source turns before making a person-specific claim]"
    )
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
            return '{"summary":"Discussion covered a court deadline.","refined_tags":[]}', {}

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
    assert result.summary == "Discussion covered a court deadline."
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
    assert result.summary == (
        "[summary withheld: speaker attribution is unresolved; retrieve exact "
        "source turns before making a person-specific claim]"
    )
    assert result.full_text == compactor._format_conversation(legal_segment.messages)


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
    compactor = DomainCompactor(
        llm_provider=mock_llm,
        config=CompactorConfig(),
    )

    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    summaries = [
        StoredSummary(
            ref="seg-1", primary_tag="legal", tags=["legal"],
            summary="Case discussion", summary_tokens=20,
            metadata=SegmentMetadata(
                source_speaker_labels=["BigTex"],
                source_speaker_identity_count=1,
                source_speaker_identity_fingerprint="speaker-proof",
                source_audience_fingerprint="scope-proof",
            ),
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
                metadata=SegmentMetadata(
                    source_speaker_labels=["Sania"],
                    source_speaker_identity_count=1,
                    source_speaker_identity_fingerprint="sania-proof",
                    source_audience_fingerprint="scope-proof",
                ),
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
                metadata=SegmentMetadata(
                    source_speaker_labels=["BigTex"],
                    source_speaker_identity_count=1,
                    source_speaker_identity_fingerprint="speaker-proof",
                    source_audience_fingerprint="scope-proof",
                ),
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


def test_tag_rollup_removes_generic_user_priming_and_prefixes_source_labels():
    provider = _SequenceProvider(
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "BigTex discussed stopping tesamorelin.",
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert "the user" not in TAG_SUMMARY_ROLLUP_PROMPT.lower()
    assert 'source display labels: ["BigTex"]' in provider.calls[0]["user"]
    assert "SPEAKER IDENTITY CONTRACT" in provider.calls[0]["user"]
    assert "actor:" not in provider.calls[0]["user"]
    assert result.summary == "BigTex stopped tesamorelin."


@pytest.mark.parametrize(
    "unsafe_label",
    [
        "actor:discord:1",
        "User.",
        "(You)",
        "U\u200bser",
        "actor\u2060:discord:1",
    ],
    ids=[
        "actor-id",
        "decorated-generic",
        "decorated-pronoun",
        "zw-generic",
        "zw-actor-id",
    ],
)
def test_unsafe_label_never_enters_tag_rollup_source_block(unsafe_label):
    source = _stored_summary(
        "Tesamorelin cessation was discussed.",
        label=unsafe_label,
    )

    rendered = _format_tag_rollup_source(source)

    assert "source display labels" not in rendered
    assert unsafe_label not in rendered


def test_compatibility_equivalent_tag_source_labels_fail_closed():
    source = _stored_summary(
        "Tesamorelin cessation was discussed.",
        label="Alex",
    )
    source.metadata.source_speaker_labels = ["Alex", "Ａｌｅｘ"]
    source.metadata.source_speaker_identity_count = 2

    rendered = _format_tag_rollup_source(source)

    assert "source display labels" not in rendered
    assert "Ａｌｅｘ" not in rendered


@pytest.mark.parametrize("label", ["BigTex", "Renée", "李雷"])
def test_valid_unicode_label_enters_tag_rollup_source_block(label):
    rendered = _format_tag_rollup_source(
        _stored_summary("A source-grounded topic was discussed.", label=label),
    )

    expected = json.dumps([label], ensure_ascii=False)
    assert f"source display labels: {expected}" in rendered


@pytest.mark.parametrize("unsafe_field", ["summary", "description"])
def test_tag_rollup_retries_when_either_prose_field_has_ambiguous_referent(
    unsafe_field,
):
    first = {
        "summary": "BigTex stopped tesamorelin.",
        "description": "BigTex discussed stopping tesamorelin.",
    }
    first[unsafe_field] = "The user stopped tesamorelin."
    provider = _SequenceProvider(
        json.dumps(first),
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "BigTex discussed stopping tesamorelin.",
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert unsafe_field in provider.calls[1]["system"]
    assert result.summary == "BigTex stopped tesamorelin."
    assert result.description == "BigTex discussed stopping tesamorelin."


def test_tag_rollup_second_ambiguous_result_uses_safe_source_concatenation():
    provider = _SequenceProvider(
        json.dumps({
            "summary": "The member stopped tesamorelin.",
            "description": "Medical discussion.",
        }),
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "This person's medical discussion.",
        }),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert 'source display labels: ["BigTex"]' in result.summary
    assert "BigTex stopped tesamorelin." in result.summary
    assert "This person" not in result.summary
    assert result.description == ""


def test_tag_rollup_retry_failure_uses_safe_source_fallback():
    provider = _SequenceProvider(
        json.dumps({
            "summary": "BigTex stopped tesamorelin.",
            "description": "The user's medical discussion.",
        }),
        RuntimeError("retry provider failure"),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary("BigTex stopped tesamorelin.", label="BigTex")],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert 'source display labels: ["BigTex"]' in result.summary
    assert "BigTex stopped tesamorelin." in result.summary
    assert result.description == ""


def test_tag_rollup_quarantines_when_all_source_prose_is_ambiguous():
    provider = _SequenceProvider(
        json.dumps({"summary": "The user stopped tesamorelin."}),
        json.dumps({"summary": "A member stopped tesamorelin."}),
    )
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [_stored_summary(
            "The user's tesamorelin experience ended.", label="BigTex",
        )],
        [1],
        ["ct-1"],
        1,
    )

    assert len(provider.calls) == 2
    assert result.summary == (
        "[tag summary withheld: source prose lacks explicit speaker attribution]"
    )
    assert result.description == ""


def test_multi_speaker_tag_rollup_bypasses_generative_reassignment():
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex takes the protocol Kuw9239 described.",
        "description": "A synthesized personal claim.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [
            _stored_summary(
                "BigTex stopped tesamorelin.", ref="seg-bigtex", label="BigTex",
            ),
            _stored_summary(
                "Kuw9239 tolerates tesamorelin.", ref="seg-kuw", label="Kuw9239",
            ),
        ],
        [1, 2],
        ["ct-1", "ct-2"],
        2,
    )

    assert provider.calls == []
    assert "BigTex stopped tesamorelin." in result.summary
    assert "Kuw9239 tolerates tesamorelin." in result.summary
    assert result.description == ""


def test_same_actor_cross_audience_tag_rollup_bypasses_generation():
    provider = _SequenceProvider(json.dumps({
        "summary": "BigTex combined private and guild disclosures.",
        "description": "Cross-audience synthesis.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [
            _stored_summary(
                "BigTex discussed a private concern.",
                ref="seg-private",
                label="BigTex",
                identity="same-actor-proof",
                scope="private-audience-proof",
            ),
            _stored_summary(
                "BigTex discussed a guild concern.",
                ref="seg-guild",
                label="BigTex",
                identity="same-actor-proof",
                scope="guild-audience-proof",
            ),
        ],
        [1, 2],
        ["ct-private", "ct-guild"],
        2,
    )

    assert provider.calls == []
    assert "private concern" in result.summary
    assert "guild concern" in result.summary
    assert result.description == ""


def test_same_display_label_for_different_actors_bypasses_tag_synthesis():
    provider = _SequenceProvider(json.dumps({
        "summary": "Alex stopped tesamorelin.",
        "description": "A synthesized claim.",
    }))
    compactor = DomainCompactor(provider, CompactorConfig(code_mode=False))

    result = compactor._build_one_tag_summary(
        "medical",
        [
            _stored_summary(
                "Alex discussed tesamorelin.",
                ref="seg-alex-1",
                label="Alex",
                identity="actor-proof-1",
            ),
            _stored_summary(
                "Alex reported edema.",
                ref="seg-alex-2",
                label="Alex",
                identity="actor-proof-2",
            ),
        ],
        [1, 2],
        ["ct-1", "ct-2"],
        2,
    )

    assert provider.calls == []
    assert "Alex discussed tesamorelin." in result.summary
    assert "Alex reported edema." in result.summary


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


def test_summary_request_conversation_text_override(ts):
    """The override must replace the formatted text everywhere it flows.

    A caller repairing a stored segment builds the request from the stored
    full_text; the prompt, the length-derived token target, and the
    request's own conversation_text must all come from the override, and
    the formatted message text must not appear anywhere.
    """
    compactor = DomainCompactor(
        llm_provider=MockLLMProvider(),
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )
    segment = TaggedSegment(
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
    formatted = compactor._format_conversation(segment.messages)

    # Passing the formatted text explicitly is identical to not passing it.
    assert compactor.build_segment_summary_request(
        segment, conversation_text=formatted,
    ) == compactor.build_segment_summary_request(segment)

    stored = "Stored transcript bytes " * 200  # long enough to move target_tokens
    req = compactor.build_segment_summary_request(
        segment, conversation_text=stored,
    )
    assert req.conversation_text == stored
    assert stored[:200] in req.prompt
    assert "court filing deadline" not in req.prompt
    assert req.original_tokens == compactor.token_counter(stored)
    expected_target = max(50, min(500, int(req.original_tokens * 0.15)))
    assert req.target_tokens == expected_target
    assert str(expected_target) in req.prompt
