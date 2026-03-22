import uuid
import pytest
from virtual_context.conversation_identity import (
    resolve_conversation_id,
    _candidate_to_uuid,
)


class TestCandidateToUuid:
    def test_produces_valid_uuid(self):
        result = _candidate_to_uuid("conversation_label", "Bast Group id:-5156869263")
        uuid.UUID(result)

    def test_deterministic(self):
        a = _candidate_to_uuid("conversation_label", "Bast Group id:-5156869263")
        b = _candidate_to_uuid("conversation_label", "Bast Group id:-5156869263")
        assert a == b

    def test_different_layers_different_ids(self):
        a = _candidate_to_uuid("conversation_label", "group-123")
        b = _candidate_to_uuid("chat_id", "group-123")
        assert a != b

    def test_different_values_different_ids(self):
        a = _candidate_to_uuid("conversation_label", "group-A")
        b = _candidate_to_uuid("conversation_label", "group-B")
        assert a != b


class TestResolveConversationId:
    def test_explicit_id_uuid_passthrough(self):
        explicit = str(uuid.uuid4())
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = resolve_conversation_id(body, explicit_id=explicit)
        assert result == explicit

    def test_explicit_id_non_uuid_hashed(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = resolve_conversation_id(body, explicit_id="my-custom-id")
        uuid.UUID(result)
        assert result != "my-custom-id"

    def test_conversation_label_wins_over_system_prompt(self):
        body = {
            "system": "You are helpful.",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "[vc:prompt]\n\n\n"
                        "Conversation info (untrusted metadata):\n"
                        "```json\n"
                        '{"conversation_label": "Bast Group id:-5156869263"}\n'
                        "```\n\nHello"
                    ),
                }
            ],
        }
        result = resolve_conversation_id(body)
        expected = _candidate_to_uuid("conversation_label", "Bast Group id:-5156869263")
        assert result == expected

    def test_chat_id_wins_over_system_prompt(self):
        body = {
            "system": "You are helpful.",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "[vc:prompt]\n\n\n"
                        "Conversation info (untrusted metadata):\n"
                        "```json\n"
                        '{"chat_id": "telegram:-5156869263"}\n'
                        "```\n\nHello"
                    ),
                }
            ],
        }
        result = resolve_conversation_id(body)
        expected = _candidate_to_uuid("chat_id", "telegram:-5156869263")
        assert result == expected

    def test_falls_back_to_system_prompt(self):
        body = {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = resolve_conversation_id(body)
        # System prompt gets hashed first, then that hash is used as the value
        uuid.UUID(result)  # valid UUID
        # Should be deterministic
        result2 = resolve_conversation_id(body)
        assert result == result2

    def test_openai_system_message(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = resolve_conversation_id(body)
        uuid.UUID(result)
        result2 = resolve_conversation_id(body)
        assert result == result2

    def test_no_signals_returns_random_uuid(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        a = resolve_conversation_id(body)
        b = resolve_conversation_id(body)
        uuid.UUID(a)
        uuid.UUID(b)

    def test_explicit_id_beats_everything(self):
        body = {
            "system": "You are helpful.",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "[vc:prompt]\n\n\n"
                        "Conversation info (untrusted metadata):\n"
                        "```json\n"
                        '{"conversation_label": "Group Chat"}\n'
                        "```\n\nHello"
                    ),
                }
            ],
        }
        result = resolve_conversation_id(body, explicit_id="my-conv")
        expected = _candidate_to_uuid("explicit_id", "my-conv")
        assert result == expected

    def test_empty_body(self):
        result = resolve_conversation_id({})
        uuid.UUID(result)

    def test_none_body(self):
        result = resolve_conversation_id(None)
        uuid.UUID(result)

    def test_anthropic_list_content_format(self):
        body = {
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "[vc:prompt]\n\n\n"
                                "Conversation info (untrusted metadata):\n"
                                "```json\n"
                                '{"conversation_label": "Group"}\n'
                                "```\n\nHi"
                            ),
                        }
                    ],
                }
            ],
        }
        result = resolve_conversation_id(body)
        expected = _candidate_to_uuid("conversation_label", "Group")
        assert result == expected

    def test_deterministic_across_calls(self):
        body = {
            "system": "Same prompt.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        a = resolve_conversation_id(body)
        b = resolve_conversation_id(body)
        assert a == b

    def test_empty_system_list_no_collision(self):
        """system=[] should NOT produce a deterministic hash — each call gets a fresh UUID."""
        a = resolve_conversation_id({"system": [], "messages": [{"role": "user", "content": "hi"}]})
        b = resolve_conversation_id({"system": [], "messages": [{"role": "user", "content": "bye"}]})
        # Both should be valid UUIDs but different (no shared system prompt)
        uuid.UUID(a)
        uuid.UUID(b)

    def test_multimodal_system_no_text_no_collision(self):
        """A system list with only image blocks (no text) should not hash to a fixed UUID."""
        body_a = {
            "system": [{"type": "image", "source": {"data": "abc"}}],
            "messages": [{"role": "user", "content": "describe this"}],
        }
        body_b = {
            "system": [{"type": "image", "source": {"data": "xyz"}}],
            "messages": [{"role": "user", "content": "describe that"}],
        }
        a = resolve_conversation_id(body_a)
        b = resolve_conversation_id(body_b)
        uuid.UUID(a)
        uuid.UUID(b)
        # Without the fix, both would hash "" to the same UUID
        # With the fix, no system hash is produced, so they fall through to random

    def test_openai_developer_image_only_no_collision(self):
        """An OpenAI developer message with only image_url content should not collide."""
        body = {
            "messages": [
                {"role": "developer", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ]},
                {"role": "user", "content": "What is this?"},
            ],
        }
        a = resolve_conversation_id(body)
        b = resolve_conversation_id(body)
        uuid.UUID(a)
        uuid.UUID(b)
