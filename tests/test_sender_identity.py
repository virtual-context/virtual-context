"""Tests for sender identity attribution pipeline."""

from virtual_context.types import get_sender_name, Message
from virtual_context.proxy._envelope import _extract_envelope_metadata, _strip_envelope


class TestGetSenderName:
    def test_returns_name_from_sender(self):
        meta = {"sender": {"name": "Sania", "label": "Sania (123)"}}
        assert get_sender_name(meta) == "Sania"

    def test_falls_back_to_display_name(self):
        meta = {"sender": {"display_name": "Yur", "label": "Yur (456)"}}
        assert get_sender_name(meta) == "Yur"

    def test_falls_back_to_label(self):
        meta = {"sender": {"label": "Bast (789)"}}
        assert get_sender_name(meta) == "Bast (789)"

    def test_returns_none_for_no_metadata(self):
        assert get_sender_name(None) is None
        assert get_sender_name({}) is None

    def test_returns_none_for_no_sender(self):
        meta = {"conversation info": {"id": "123"}}
        assert get_sender_name(meta) is None

    def test_returns_none_for_non_dict_sender(self):
        meta = {"sender": "just a string"}
        assert get_sender_name(meta) is None

    def test_returns_none_for_empty_name(self):
        meta = {"sender": {"name": "", "label": ""}}
        assert get_sender_name(meta) is None


class TestExtractEnvelopeMetadata:
    def test_extracts_sender_block(self):
        text = (
            'Sender (untrusted metadata):\n'
            '```json\n'
            '{"label": "Sania (7281617716)", "name": "Sania"}\n'
            '```\n'
            'What about charlotte tilbury'
        )
        stripped, meta = _extract_envelope_metadata(text)
        assert stripped.strip() == "What about charlotte tilbury"
        assert meta["sender"]["name"] == "Sania"

    def test_extracts_multiple_blocks(self):
        text = (
            'Sender (untrusted metadata):\n'
            '```json\n'
            '{"name": "Sania"}\n'
            '```\n'
            'Conversation info (untrusted metadata):\n'
            '```json\n'
            '{"message_id": "12070", "sender_id": "7281617716"}\n'
            '```\n'
            'Hello world'
        )
        stripped, meta = _extract_envelope_metadata(text)
        assert stripped.strip() == "Hello world"
        assert "sender" in meta
        assert "conversation info" in meta

    def test_no_metadata_returns_empty_dict(self):
        text = "Just a regular message"
        stripped, meta = _extract_envelope_metadata(text)
        assert stripped == "Just a regular message"
        assert meta == {}

    def test_malformed_json_skipped(self):
        text = (
            'Sender (untrusted metadata):\n'
            '```json\n'
            '{not valid json}\n'
            '```\n'
            'Hello'
        )
        stripped, meta = _extract_envelope_metadata(text)
        assert "Hello" in stripped
        assert meta == {}

    def test_strip_envelope_backward_compat(self):
        text = (
            'Sender (untrusted metadata):\n'
            '```json\n'
            '{"name": "Sania"}\n'
            '```\n'
            'Hello world'
        )
        result = _strip_envelope(text)
        assert isinstance(result, str)
        assert "Hello world" in result
        assert "Sania" not in result

    def test_label_key_is_lowercase(self):
        text = (
            'Reply Context (untrusted metadata):\n'
            '```json\n'
            '{"sender_label": "Bast"}\n'
            '```\n'
            'Hey'
        )
        stripped, meta = _extract_envelope_metadata(text)
        assert "reply context" in meta
        assert meta["reply context"]["sender_label"] == "Bast"


class TestExtractMessageTextWithMeta:
    def test_anthropic_format_extracts_metadata(self):
        from virtual_context.proxy.formats import AnthropicFormat
        fmt = AnthropicFormat()
        msg = {
            "role": "user",
            "content": (
                'Sender (untrusted metadata):\n'
                '```json\n'
                '{"name": "Sania"}\n'
                '```\n'
                'What about charlotte tilbury'
            ),
        }
        text, meta = fmt.extract_message_text_with_meta(msg)
        assert "charlotte tilbury" in text
        assert meta.get("sender", {}).get("name") == "Sania"

    def test_no_metadata_returns_empty_dict(self):
        from virtual_context.proxy.formats import AnthropicFormat
        fmt = AnthropicFormat()
        msg = {"role": "user", "content": "plain message"}
        text, meta = fmt.extract_message_text_with_meta(msg)
        assert text == "plain message"
        assert meta == {}


class TestHistoryPairsMetadata:
    def test_anthropic_history_pairs_have_metadata(self):
        from virtual_context.proxy.formats import AnthropicFormat
        fmt = AnthropicFormat()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        'Sender (untrusted metadata):\n'
                        '```json\n'
                        '{"name": "Sania"}\n'
                        '```\n'
                        'Hello'
                    ),
                },
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        pairs = fmt.extract_history_pairs(body)
        assert len(pairs) == 2
        assert pairs[0].metadata is not None
        assert pairs[0].metadata.get("sender", {}).get("name") == "Sania"
        assert pairs[1].metadata is None

    def test_user_message_without_metadata_gets_none_or_empty(self):
        from virtual_context.proxy.formats import AnthropicFormat
        fmt = AnthropicFormat()
        body = {
            "messages": [
                {"role": "user", "content": "plain message"},
                {"role": "assistant", "content": "response"},
            ]
        }
        pairs = fmt.extract_history_pairs(body)
        assert pairs[0].metadata is None or pairs[0].metadata == {}


class TestTurnTagEntrySender:
    def test_turn_tag_entry_has_sender_field(self):
        from virtual_context.types import TurnTagEntry
        entry = TurnTagEntry(
            turn_number=0, message_hash="abc123",
            tags=["cooking"], primary_tag="cooking",
            sender="Sania",
        )
        assert entry.sender == "Sania"

    def test_sender_defaults_to_empty(self):
        from virtual_context.types import TurnTagEntry
        entry = TurnTagEntry(turn_number=0, message_hash="abc123")
        assert entry.sender == ""

    def test_sqlite_round_trip_preserves_sender(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import TurnTagEntry, EngineStateSnapshot
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        entry = TurnTagEntry(
            turn_number=0, message_hash="abc123",
            tags=["cooking"], primary_tag="cooking",
            sender="Sania",
        )
        state = EngineStateSnapshot(
            conversation_id="test-conv",
            compacted_through=0,
            turn_count=1,
            turn_tag_entries=[entry],
        )
        store.save_engine_state(state)
        loaded = store.load_engine_state("test-conv")
        assert loaded is not None
        assert loaded.turn_tag_entries[0].sender == "Sania"

    def test_sqlite_round_trip_missing_sender_defaults_empty(self, tmp_path):
        """Old state without sender field should deserialize with empty string."""
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import TurnTagEntry, EngineStateSnapshot
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        entry = TurnTagEntry(
            turn_number=0, message_hash="abc123",
            tags=["cooking"], primary_tag="cooking",
        )
        state = EngineStateSnapshot(
            conversation_id="test-conv",
            compacted_through=0,
            turn_count=1,
            turn_tag_entries=[entry],
        )
        store.save_engine_state(state)
        loaded = store.load_engine_state("test-conv")
        assert loaded is not None
        assert loaded.turn_tag_entries[0].sender == ""


class TestEndToEndSenderIdentity:
    def test_format_conversation_with_real_envelope(self):
        """Full pipeline: envelope with sender -> compactor sees real name."""
        from virtual_context.proxy._envelope import _extract_envelope_metadata
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, get_sender_name

        raw_text = (
            'Sender (untrusted metadata):\n'
            '```json\n'
            '{"label": "Sania (7281617716)", "name": "Sania"}\n'
            '```\n'
            'What about charlotte tilbury wonder skin'
        )
        stripped, meta = _extract_envelope_metadata(raw_text)
        msg = Message(role="user", content=stripped, metadata=meta)
        asst = Message(role="assistant", content="The wonder skin is great!")

        compactor = DomainCompactor(llm_provider=None, config=CompactorConfig())
        formatted = compactor._format_conversation([msg, asst])
        assert "Sania" in formatted
        assert "User:" not in formatted
        assert "charlotte tilbury" in formatted
        assert get_sender_name(meta) == "Sania"

    def test_no_metadata_falls_back_to_role(self):
        """Without metadata, compactor uses role as label."""
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig

        msg = Message(role="user", content="Hello")
        asst = Message(role="assistant", content="Hi")
        compactor = DomainCompactor(llm_provider=None, config=CompactorConfig())
        formatted = compactor._format_conversation([msg, asst])
        assert "User:" in formatted or "User (" in formatted
