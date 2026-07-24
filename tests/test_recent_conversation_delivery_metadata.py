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


def test_final_delivery_metadata_uses_complete_suffix_after_odd_old_group() -> None:
    replay = [
        Message(role="user", content="Older requester group lacks an answer."),
        Message(role="user", content="An older complete requester question."),
        Message(role="assistant", content="An older complete requester answer."),
        Message(role="user", content="Begin with CompactedLane73:."),
        Message(role="assistant", content="CompactedLane73: Understood."),
    ]
    expected = replay[-2:]
    body = {
        "messages": [
            {
                "role": "user",
                "content": f"{replay[0].content}\n{replay[1].content}",
            },
            {"role": "assistant", "content": replay[2].content},
            {"role": "user", "content": replay[3].content},
            {"role": "assistant", "content": replay[4].content},
            {"role": "user", "content": "What is the capital of Portugal?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == _build_recent_conversation_native_metadata(expected)


def test_final_delivery_metadata_does_not_skip_trailing_unpaired_group() -> None:
    replay = [
        Message(role="user", content="Begin with OlderLane11:."),
        Message(role="assistant", content="OlderLane11: Understood."),
        Message(role="user", content="Newer requester group lacks an answer."),
    ]
    body = {
        "messages": [
            {"role": "user", "content": replay[0].content},
            {"role": "assistant", "content": replay[1].content},
            {"role": "user", "content": replay[2].content},
            {"role": "user", "content": "What is the capital of Portugal?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == {}


def test_final_delivery_metadata_rejects_body_only_adjacency_gap() -> None:
    replay = [
        Message(role="user", content="Begin with GapLane19:."),
        Message(role="assistant", content="GapLane19: Understood."),
    ]
    body = {
        "messages": [
            {"role": "user", "content": replay[0].content},
            {"role": "assistant", "content": replay[1].content},
            {"role": "user", "content": "Body-only interloper."},
            {"role": "user", "content": "What is the capital of Portugal?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == {}


def test_final_delivery_metadata_caps_attestation_at_newest_200_messages() -> None:
    replay = [
        Message(
            role="user" if index % 2 == 0 else "assistant",
            content=f"Replay message {index}.",
        )
        for index in range(202)
    ]
    body = {
        "messages": [
            *[
                {"role": message.role, "content": message.content}
                for message in replay
            ],
            {"role": "user", "content": "What is the capital of Portugal?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == _build_recent_conversation_native_metadata(replay[-200:])


def test_final_delivery_metadata_rejects_non_user_active_slot() -> None:
    replay = [
        Message(role="user", content="Begin with PrefillLane41:."),
        Message(role="assistant", content="PrefillLane41: Understood."),
    ]
    body = {
        "messages": [
            {"role": "user", "content": replay[0].content},
            {"role": "assistant", "content": replay[1].content},
            {"role": "assistant", "content": "Prefilled active response."},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == {}


def test_final_delivery_metadata_uses_newer_pair_after_older_mutation() -> None:
    replay = [
        Message(role="user", content="Older complete question."),
        Message(role="assistant", content="Older complete answer."),
        Message(role="user", content="Begin with MutationLane23:."),
        Message(role="assistant", content="MutationLane23: Understood."),
    ]
    body = {
        "messages": [
            {"role": "user", "content": replay[0].content},
            {"role": "assistant", "content": "Mutated older answer."},
            {"role": "user", "content": replay[2].content},
            {"role": "assistant", "content": replay[3].content},
            {"role": "user", "content": "What is the capital of Portugal?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        replay,
        body,
        AnthropicFormat(),
    ) == _build_recent_conversation_native_metadata(replay[-2:])


def test_final_delivery_metadata_is_empty_without_inserted_replay() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "What is the capital of Portugal?"},
        ]
    }

    assert _build_final_recent_conversation_native_metadata(
        [],
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
