"""Request-local delivery metadata for exact canonical conversation replay."""

from virtual_context.proxy.formats import AnthropicFormat, OpenAIResponsesFormat
from virtual_context.proxy.server import (
    _build_final_recent_conversation_native_metadata,
    _build_recent_conversation_native_metadata,
)
from virtual_context.types import Message


def test_native_delivery_metadata_pins_cross_language_hash_contract() -> None:
    metadata = _build_recent_conversation_native_metadata(
        [
            Message(role="user", content="Set prefix HarnessGuild17:"),
            Message(role="assistant", content="HarnessGuild17: Understood."),
        ]
    )

    assert metadata == {
        "message_count": 2,
        "message_hashes": [
            "7a43cc2e4423a55eb247dc8817c2ec7a071c8167fc1f995b3baa8f5d5c3dcde5",
            "eb8a0e6024f7c4a415be7794013a5b7d01e5fc23696f564103acc31ad3c12e0b",
        ],
    }


def test_native_delivery_metadata_is_empty_without_replay() -> None:
    assert _build_recent_conversation_native_metadata([]) == {}


def test_native_delivery_metadata_rejects_non_conversational_roles() -> None:
    assert _build_recent_conversation_native_metadata(
        [Message(role="system", content="not replay")]
    ) == {}


def test_native_delivery_metadata_rejects_odd_or_assistant_first_replay() -> None:
    assert _build_recent_conversation_native_metadata(
        [Message(role="user", content="unpaired")]
    ) == {}
    assert _build_recent_conversation_native_metadata(
        [
            Message(role="assistant", content="wrong first role"),
            Message(role="user", content="wrong second role"),
        ]
    ) == {}


def test_native_delivery_hash_contract_preserves_unicode_codepoints() -> None:
    metadata = _build_recent_conversation_native_metadata(
        [
            Message(role="user", content="Préférence: commence par 🧭:"),
            Message(role="assistant", content="🧭: Compris — Cafe\u0301."),
        ]
    )

    assert metadata["message_hashes"] == [
        "a783b72f8a7b34fe2543929a8a329192e57b7a99f9d6a42773437eece9cfdf7d",
        "2d159c07b3634e35f266fb1e6e80f4dbf8d1849996db45934c328f9c82baa48d",
    ]


def test_final_delivery_metadata_matches_actual_contiguous_suffix() -> None:
    replay = [
        Message(role="user", content="Begin with FinalBody61:."),
        Message(role="assistant", content="FinalBody61: Understood."),
    ]
    body = {
        "messages": [
            {"role": "user", "content": "Earlier local question."},
            {"role": "assistant", "content": "Earlier local answer."},
            {"role": "user", "content": replay[0].content},
            {"role": "assistant", "content": replay[1].content},
            {"role": "user", "content": "What is the chemical symbol for gold?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == _build_recent_conversation_native_metadata(replay)


def test_final_delivery_metadata_rejects_post_injection_mutation() -> None:
    replay = [
        Message(role="user", content="Begin with FinalBody61:."),
        Message(role="assistant", content="FinalBody61: Understood."),
    ]
    body = {
        "messages": [
            {"role": "user", "content": replay[0].content},
            {"role": "assistant", "content": "mutated after insertion"},
            {"role": "user", "content": "What is the chemical symbol for gold?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == {}


def test_final_delivery_metadata_supports_openai_responses_content_parts() -> None:
    replay = [
        Message(role="user", content="Begin with ResponsesLane38:."),
        Message(role="assistant", content="ResponsesLane38: Understood."),
    ]
    fmt = OpenAIResponsesFormat()
    enriched, delivered = fmt.inject_replayed_conversation_with_delivery(
        {
            "model": "gpt-5.6-sol",
            "input": "What is the chemical symbol for gold?",
        },
        replay,
    )

    assert delivered == replay
    assert _build_final_recent_conversation_native_metadata(
        delivered,
        enriched,
        fmt,
    ) == _build_recent_conversation_native_metadata(replay)
