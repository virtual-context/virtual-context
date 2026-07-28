"""Byte-exact characterization of the source-text fallback sites.

When summarization fails, the compactor stores a bounded slice of the
segment's own source text as the summary. These tests pin each failure
site byte for byte: the exact slice bound, the call count, and the
relation between the stored summary and the segment's ``full_text``.
They exist so the summarize step can be restructured without any drift
in what a failure writes; a change to any fallback payload must fail
here before it ships.

The prefix relation is load-bearing: a fallback summary is a byte-exact
prefix of its own ``full_text``, which is how damaged rows are
identified after the fact. The short-source site is the deliberate
exception: when the source fits inside the slice bound, the stored
"summary" equals the full text, so a strict-prefix predicate correctly
does not match it.

Sites, by trigger:

- retry exhaust: summary unusable after one corrective retry
- provider exception: ``llm.complete`` raises
- concurrent driver: a segment's whole compaction future raises
- short source: unusable overshoot on a sub-256-char source, no retry
- parse failure: response JSON unparseable, raw text becomes summary
"""

import json
from datetime import timedelta

import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import CompactorConfig, Message, TaggedSegment

_CONFIG = dict(
    summary_ratio=0.15,
    min_summary_tokens=50,
    max_summary_tokens=500,
)


def _compactor(provider) -> DomainCompactor:
    return DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(**_CONFIG),
        model_name="test-model",
    )


def _segment(ts, user_text: str, assistant_text: str) -> TaggedSegment:
    return TaggedSegment(
        primary_tag="legal",
        tags=["legal", "court"],
        messages=[
            Message(role="user", content=user_text, timestamp=ts),
            Message(
                role="assistant",
                content=assistant_text,
                timestamp=ts + timedelta(seconds=30),
            ),
        ],
        token_count=50,
        start_timestamp=ts,
        end_timestamp=ts + timedelta(seconds=30),
        turn_count=1,
    )


def _long_segment(ts, min_chars: int) -> TaggedSegment:
    """A segment whose formatted text safely exceeds *min_chars*."""
    sentences = [
        f"Filing detail {i}: the deadline for exhibit {i} is day {i}."
        for i in range(min_chars // 40 + 4)
    ]
    return _segment(ts, " ".join(sentences), "Noted; the schedule stands.")


def _expected_slice(compactor: DomainCompactor, conversation_text: str) -> str:
    """The bounded source slice, spelled independently of the compactor."""
    original_tokens = compactor.token_counter(conversation_text)
    target_tokens = max(
        _CONFIG["min_summary_tokens"],
        min(
            _CONFIG["max_summary_tokens"],
            int(original_tokens * _CONFIG["summary_ratio"]),
        ),
    )
    return conversation_text[: target_tokens * 4]


class _DegenerateProvider:
    """Always returns an unusable (fence-fragment) response."""

    def __init__(self):
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int):
        self.calls += 1
        return "```json\n{", {}


class _RaisingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int):
        self.calls += 1
        raise RuntimeError("provider unavailable")


def test_retry_exhaust_fallback_is_byte_exact_source_slice(ts):
    provider = _DegenerateProvider()
    compactor = _compactor(provider)
    segment = _long_segment(ts, 3000)
    conversation_text = compactor._format_conversation(segment.messages)

    result = compactor.compact([segment])[0]

    assert provider.calls == 2
    assert result.summary == _expected_slice(compactor, conversation_text)
    assert result.full_text == conversation_text
    # Strict prefix: shorter than full_text and byte-identical up front.
    assert len(result.summary) < len(result.full_text)
    assert result.full_text[: len(result.summary)] == result.summary
    assert result.metadata.entities == []
    assert result.metadata.key_decisions == []
    assert result.metadata.action_items == []
    assert result.metadata.date_references == []
    assert set(segment.tags) <= set(result.tags)


def test_provider_exception_fallback_is_byte_exact_source_slice(ts):
    provider = _RaisingProvider()
    compactor = _compactor(provider)
    segment = _long_segment(ts, 3000)
    conversation_text = compactor._format_conversation(segment.messages)

    result = compactor.compact([segment])[0]

    assert provider.calls == 1  # the exception path does not retry
    assert result.summary == _expected_slice(compactor, conversation_text)
    assert result.full_text == conversation_text
    assert len(result.summary) < len(result.full_text)
    assert result.full_text[: len(result.summary)] == result.summary


def test_concurrent_driver_failure_falls_back_to_2000_char_slice(ts):
    provider = _DegenerateProvider()
    compactor = _compactor(provider)
    good = '{"summary": "A grounded filing-schedule summary.", "refined_tags": []}'
    compactor.llm.complete = lambda system, user, max_tokens: (good, {})

    segments = [
        _long_segment(ts, 5000),
        _long_segment(ts + timedelta(minutes=5), 5000),
    ]
    texts = [compactor._format_conversation(s.messages) for s in segments]
    assert all(len(t) > 2000 for t in texts)

    real_compact_one = compactor._compact_one

    def _explode_first(segment, *args, **kwargs):
        if segment.id == segments[0].id:
            raise ValueError("worker died")
        return real_compact_one(segment, *args, **kwargs)

    compactor._compact_one = _explode_first
    results = compactor.compact(segments)

    failed, survived = results[0], results[1]
    assert failed.summary == texts[0][:2000]
    assert failed.summary_tokens == compactor.token_counter(texts[0][:2000])
    assert failed.original_tokens == compactor.token_counter(texts[0])
    assert failed.compression_ratio == 0.0
    assert failed.full_text == texts[0]
    assert failed.timestamp == segments[0].start_timestamp
    assert failed.tags == segments[0].tags
    assert len(failed.summary) < len(failed.full_text)
    assert failed.full_text[: len(failed.summary)] == failed.summary
    # One segment failing must not damage its neighbor.
    assert survived.summary == "A grounded filing-schedule summary."


def test_short_source_fallback_is_equality_not_strict_prefix(ts):
    """A short source fits inside the slice bound, so summary == full_text.

    This is why a strict-prefix predicate deliberately does not match
    these rows: nothing was lost, there is nothing to recover.
    """

    class OvershootProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str, max_tokens: int):
            self.calls += 1
            return json.dumps({"summary": "unrelated history " * 100}), {}

    provider = OvershootProvider()
    compactor = _compactor(provider)
    segment = _segment(
        ts, "What's the court filing deadline?", "The filing is due January 30.",
    )
    conversation_text = compactor._format_conversation(segment.messages)
    assert len(conversation_text.strip()) < 256
    assert len(conversation_text) <= len(_expected_slice(compactor, conversation_text))

    result = compactor.compact([segment])[0]

    assert provider.calls == 1  # no retry for the short-source overshoot
    assert result.summary == conversation_text
    assert result.summary == result.full_text


def test_parse_failure_recovers_summary_field_from_malformed_json(ts):
    compactor = _compactor(_DegenerateProvider())
    malformed = '{"summary": "The hearing moved to March.", "entities": [unclosed'
    parsed = compactor._parse_response(malformed)
    assert parsed["summary"] == "The hearing moved to March."
    assert parsed["refined_tags"] == []
    assert parsed["code_refs"] == []


def test_parse_failure_raw_text_becomes_summary_with_pinned_shape(ts):
    compactor = _compactor(_DegenerateProvider())
    parsed = compactor._parse_response("An upstream error page, not JSON.")
    assert parsed == {
        "summary": "An upstream error page, not JSON.",
        "entities": [],
        "key_decisions": [],
        "action_items": [],
        "date_references": [],
        "refined_tags": [],
        "code_refs": [],
    }
