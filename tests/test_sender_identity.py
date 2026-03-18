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
