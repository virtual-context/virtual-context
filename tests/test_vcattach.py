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
