"""The typed summarize outcome must never substitute content of its own.

The compactor's summarize path converts failures into stored content (a
source-text fallback, or raw response text as the summary). The repair
path must instead surface every failure as a typed outcome. The decisive
case is the raw error page: under the compactor's semantics it becomes
the summary; here it must be Malformed, never Generated.
"""

import json

import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.segment_repair import (
    Generated,
    Malformed,
    ProviderFailure,
    summarize_segment_once,
)
from virtual_context.types import CompactorConfig, Message, TaggedSegment


class _ScriptedProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response, {"input_tokens": 100, "output_tokens": 20}


def _request(provider, ts):
    compactor = DomainCompactor(
        llm_provider=provider,
        config=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=500,
        ),
        model_name="test-model",
    )
    segment = TaggedSegment(
        primary_tag="legal",
        tags=["legal"],
        messages=[Message(role="user", content="deadline?", timestamp=ts)],
        token_count=10,
        start_timestamp=ts,
        end_timestamp=ts,
        turn_count=1,
    )
    stored_full_text = "User: What is the filing deadline?\n" * 40
    request = compactor.build_segment_summary_request(
        segment, conversation_text=stored_full_text,
    )
    return compactor, request


def test_valid_response_is_generated_with_usage(ts):
    provider = _ScriptedProvider(
        response=json.dumps({"summary": "They discussed a filing deadline."}),
    )
    compactor, request = _request(provider, ts)
    outcome = summarize_segment_once(compactor, request)
    assert outcome == Generated(
        summary="They discussed a filing deadline.",
        usage={"input_tokens": 100, "output_tokens": 20},
    )
    assert provider.calls == 1  # exactly one call, no retry of any kind


def test_provider_exception_is_provider_failure_not_content(ts):
    provider = _ScriptedProvider(error=RuntimeError("credits exhausted"))
    compactor, request = _request(provider, ts)
    outcome = summarize_segment_once(compactor, request)
    assert isinstance(outcome, ProviderFailure)
    assert "RuntimeError" in outcome.error
    assert "credits exhausted" in outcome.error


def test_raw_error_page_is_malformed_never_generated(ts):
    """The v-1 defect this type system exists to prevent.

    Under the compactor's semantics this text would BECOME the stored
    summary (non-degenerate, non-overshooting, non-prefix) and be
    permanently accepted. Here it must be Malformed with the raw text
    and paid usage preserved.
    """
    raw = "502 Bad Gateway: upstream connect error before headers"
    provider = _ScriptedProvider(response=raw)
    compactor, request = _request(provider, ts)
    outcome = summarize_segment_once(compactor, request)
    assert isinstance(outcome, Malformed)
    assert outcome.raw_text == raw
    assert outcome.usage == {"input_tokens": 100, "output_tokens": 20}


@pytest.mark.parametrize(
    "response",
    [
        json.dumps({"entities": ["no summary key"]}),
        json.dumps({"summary": None}),
        json.dumps({"summary": 42}),
        json.dumps({"summary": ["a", "list"]}),
        json.dumps({"summary": ""}),
        json.dumps({"summary": "   \n\t "}),
        "```json\n{\"summary\":",  # truncated fence fragment
        "{}",
        # Valid JSON whose top level is not an object: parse succeeds
        # and returns the bare value, which must classify as Malformed,
        # never crash the caller.
        "[]",
        '["a", "b"]',
        '"just a JSON string"',
        "42",
        "true",
        "null",
    ],
)
def test_unusable_summary_shapes_are_malformed(ts, response):
    provider = _ScriptedProvider(response=response)
    compactor, request = _request(provider, ts)
    outcome = summarize_segment_once(compactor, request)
    assert isinstance(outcome, Malformed)


def test_error_page_embedding_json_without_summary_is_malformed(ts):
    """parse_llm_json extracts embedded objects; absence of a summary
    field must still classify as Malformed."""
    provider = _ScriptedProvider(
        response='Upstream said: {"error": "rate limited", "code": 429}',
    )
    compactor, request = _request(provider, ts)
    outcome = summarize_segment_once(compactor, request)
    assert isinstance(outcome, Malformed)
