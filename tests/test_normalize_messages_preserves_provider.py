"""Regression tests for `normalize_messages` preserving `api`/`provider` on
assistant messages.

Context: downstream OpenAI-Codex provider plugins inspect `msg.provider`,
`msg.api`, and `msg.model` to decide whether to preserve thinking blocks and
their textSignatures vs converting thinking→text without signature. When
those fields were stripped, the downstream transform produced two adjacent
signatureless inner items per outer assistant message and synthesized the
same `msg_${msgIndex}` for both, triggering `Duplicate item found with id
msg_X` rejections from the upstream Codex Responses API.

This file pins the strip-tuple narrowing change at `formats.py:283-286` so
the original strip-of-everything regression cannot silently re-emerge.
"""
from __future__ import annotations

import copy

from virtual_context.proxy.formats import normalize_messages


# Convenience builder so individual tests stay legible.
def _asst(content, **extra):
    base = {"role": "assistant", "content": content}
    base.update(extra)
    return base


def test_normalize_preserves_provider_api_model_on_assistant():
    """Assistant msgs carrying `provider`/`api`/`model` retain those fields
    through normalize_messages. The legacy strip fields (`stopReason`) are
    still removed."""
    msgs = [
        _asst(
            [{"type": "text", "text": "ok"}],
            model="gpt-5.5",
            provider="openai-codex",
            api="openai-codex-responses",
            stopReason="end_turn",
        ),
    ]
    normalize_messages(msgs)
    assert msgs[0]["model"] == "gpt-5.5"
    assert msgs[0]["provider"] == "openai-codex"
    assert msgs[0]["api"] == "openai-codex-responses"
    assert "stopReason" not in msgs[0]


def test_normalize_still_strips_legacy_fields_on_assistant():
    """The six legacy strip fields remain stripped; only `api` and `provider`
    moved out of the strip tuple in this fix."""
    msgs = [
        _asst(
            [{"type": "text", "text": "ok"}],
            model="gpt-5.5",
            provider="openai-codex",
            api="openai-codex-responses",
            stopReason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 2},
            responseId="resp_abc",
            timestamp="2026-05-30T00:00:00Z",
            errorMessage="should_be_removed_but_not_treated_as_error_here",
            thinkingSignature="msg-level-sig-should-be-removed",
        ),
    ]
    # `errorMessage` + empty content would trigger removal at formats.py:230-234.
    # Use non-empty content above so we exercise only the strip-tuple branch.
    normalize_messages(msgs)
    assert len(msgs) == 1  # Not error-removed
    # Retained (the fix)
    assert msgs[0]["model"] == "gpt-5.5"
    assert msgs[0]["provider"] == "openai-codex"
    assert msgs[0]["api"] == "openai-codex-responses"
    # Still stripped
    for k in ("stopReason", "usage", "responseId",
              "timestamp", "errorMessage", "thinkingSignature"):
        assert k not in msgs[0], f"{k!r} should still be stripped"


def test_normalize_does_not_add_provider_api_to_user_or_tool_messages():
    """The strip tuple only runs inside the assistant branch. User/tool
    messages are not affected. No fields are added either."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_abc", "content": "result"},
        {"role": "system", "content": "you are a bot"},
    ]
    snapshot = copy.deepcopy(msgs)
    normalize_messages(msgs)
    # User: only `timestamp` would be popped if present (it isn't here);
    # otherwise unchanged.
    assert msgs[0] == snapshot[0]
    # Tool: unaffected by the strip-tuple branch.
    assert msgs[1] == snapshot[1]
    # System: unaffected.
    assert msgs[2] == snapshot[2]


def test_thinking_text_signature_shape_survives_normalize_round_trip():
    """The strip tuple only removes TOP-LEVEL keys on the message dict; inner
    content-block fields including textSignature on both thinking and text
    blocks are not touched. Combined with the fix, provider/api/model are
    also retained, preserving the exact set of fields openclaw's downstream
    isSameModel check needs."""
    msgs = [
        _asst(
            [
                {"type": "thinking", "thinking": "reasoning",
                 "textSignature": "sig_think"},
                {"type": "text", "text": "answer",
                 "textSignature": "sig_text"},
            ],
            model="gpt-5.5",
            provider="openai-codex",
            api="openai-codex-responses",
        ),
    ]
    normalize_messages(msgs)
    asst = msgs[0]
    # Top-level fields preserved
    assert asst["model"] == "gpt-5.5"
    assert asst["provider"] == "openai-codex"
    assert asst["api"] == "openai-codex-responses"
    # Content blocks untouched (including content-block-level textSignature)
    assert asst["content"][0]["type"] == "thinking"
    assert asst["content"][0]["textSignature"] == "sig_think"
    assert asst["content"][0]["thinking"] == "reasoning"
    assert asst["content"][1]["type"] == "text"
    assert asst["content"][1]["textSignature"] == "sig_text"
    assert asst["content"][1]["text"] == "answer"


def test_idempotent_on_already_normalized():
    """Calling normalize_messages twice produces the same result."""
    msgs = [
        _asst(
            [{"type": "text", "text": "ok"}],
            model="gpt-5.5",
            provider="openai-codex",
            api="openai-codex-responses",
            stopReason="end_turn",
        ),
        {"role": "user", "content": "hi"},
    ]
    normalize_messages(msgs)
    after_first = copy.deepcopy(msgs)
    normalize_messages(msgs)
    assert msgs == after_first


def test_preserves_provider_api_values_without_type_coercion():
    """The fix only narrows the strip tuple. It does NOT validate, coerce,
    synthesize, or backfill provider metadata. Whatever the caller supplied
    is preserved exactly — strings, None, or unexpected dict/list values
    all roundtrip identically."""
    msgs = [
        _asst([{"type": "text", "text": "ok"}],
              provider="openai-codex", api="openai-codex-responses"),
        _asst([{"type": "text", "text": "ok"}],
              provider=None, api=None),
        _asst([{"type": "text", "text": "ok"}],
              provider={"nested": "dict"}, api=["list", "value"]),
        _asst([{"type": "text", "text": "ok"}],
              provider=42, api=True),
    ]
    snapshot = copy.deepcopy(msgs)
    normalize_messages(msgs)
    for i, m in enumerate(msgs):
        assert m["provider"] == snapshot[i]["provider"], (
            f"msg {i} provider value should pass through unchanged"
        )
        assert m["api"] == snapshot[i]["api"], (
            f"msg {i} api value should pass through unchanged"
        )


def test_perfume_shape_through_normalize_preserves_collision_avoidance_fields():
    """Replay-shape regression: synthesize the exact failure-triggering shape
    from the perfume / 353e917e incidents and assert that the fields
    openclaw's downstream isSameModel check needs are preserved through
    normalization. With provider/api/model retained AND content-block
    textSignatures roundtripping unchanged, the downstream transform's
    isSameModel evaluates true, thinking→text conversion is skipped, no
    signatureless duplicate text blocks are synthesized, and the
    `Duplicate item found with id msg_X` collision is structurally
    avoided."""
    body = {"messages": [
        {"role": "user",
         "content": [{"type": "text", "text": "hi"}]},
        _asst(
            [
                {"type": "thinking", "thinking": "...",
                 "textSignature": "sig_think"},
                {"type": "text", "text": "answer",
                 "textSignature": "sig_abc"},
            ],
            model="gpt-5.5",
            provider="openai-codex",
            api="openai-codex-responses",
        ),
    ]}
    normalize_messages(body["messages"])
    asst = body["messages"][1]
    # The collision-avoidance fields openclaw's transport needs:
    assert asst["provider"] == "openai-codex"
    assert asst["api"] == "openai-codex-responses"
    assert asst["model"] == "gpt-5.5"
    # The original [thinking + textSignature, text + textSignature] shape
    # survives unchanged — both content blocks keep their textSignature.
    assert asst["content"][0]["type"] == "thinking"
    assert asst["content"][0]["textSignature"] == "sig_think"
    assert asst["content"][1]["type"] == "text"
    assert asst["content"][1]["textSignature"] == "sig_abc"
