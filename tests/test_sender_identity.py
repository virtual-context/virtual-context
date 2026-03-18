"""Tests for sender identity attribution pipeline."""

from virtual_context.types import get_sender_name, Message


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
