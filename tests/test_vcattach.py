# tests/test_vcattach.py
"""Tests for VCATTACH command."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest


@pytest.fixture
def sqlite_store(tmp_path):
    from virtual_context.storage.sqlite import SQLiteStore
    store = SQLiteStore(str(tmp_path / "test.db"))
    return store


def test_save_and_resolve_alias(sqlite_store):
    sqlite_store.save_conversation_alias("old-conv-123", "target-conv-456")
    resolved = sqlite_store.resolve_conversation_alias("old-conv-123")
    assert resolved == "target-conv-456"


def test_resolve_unknown_alias(sqlite_store):
    resolved = sqlite_store.resolve_conversation_alias("nonexistent")
    assert resolved is None


def test_alias_overwrite(sqlite_store):
    sqlite_store.save_conversation_alias("old-123", "target-a")
    sqlite_store.save_conversation_alias("old-123", "target-b")
    assert sqlite_store.resolve_conversation_alias("old-123") == "target-b"


def test_alias_chain_not_followed(sqlite_store):
    """Aliases are single-hop — no chain following."""
    sqlite_store.save_conversation_alias("a", "b")
    sqlite_store.save_conversation_alias("b", "c")
    assert sqlite_store.resolve_conversation_alias("a") == "b"


# --- VCATTACH regex tests ---

import re

_VCATTACH_RE = re.compile(r"^VCATTACH\s+(.+)$", re.IGNORECASE)


def test_vcattach_regex_label():
    m = _VCATTACH_RE.match("VCATTACH website")
    assert m and m.group(1) == "website"


def test_vcattach_regex_uuid():
    m = _VCATTACH_RE.match("VCATTACH d4f83259-4ffc-fa3f-5914-a266d0a4577c")
    assert m and m.group(1) == "d4f83259-4ffc-fa3f-5914-a266d0a4577c"


def test_vcattach_regex_prefix():
    m = _VCATTACH_RE.match("VCATTACH d4f83259")
    assert m and m.group(1) == "d4f83259"


def test_vcattach_regex_case_insensitive():
    m = _VCATTACH_RE.match("vcattach Website")
    assert m and m.group(1) == "Website"


def test_vcattach_regex_no_target():
    m = _VCATTACH_RE.match("VCATTACH")
    assert m is None


def test_vcattach_regex_not_triggered_by_history():
    """Only the last user message should trigger, not history."""
    m = _VCATTACH_RE.match("I said VCATTACH website earlier")
    assert m is None
