"""Tests for TagCanonicalizer."""
import pytest
from virtual_context.core.tag_canonicalizer import TagCanonicalizer, _edit_distance


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("database", "database") == 0

    def test_one_char_diff(self):
        assert _edit_distance("database", "databases") == 1

    def test_completely_different(self):
        assert _edit_distance("abc", "xyz") == 3

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0
        assert _edit_distance("abc", "") == 3
        assert _edit_distance("", "abc") == 3

    def test_db_database(self):
        dist = _edit_distance("db", "database")
        assert dist > 0


class TestTagCanonicalizer:
    def test_basic_normalization(self):
        c = TagCanonicalizer()
        assert c.canonicalize("Database") == "database"
        assert c.canonicalize("data_layer") == "data-layer"
        assert c.canonicalize("  api  ") == "api"

    def test_alias_resolution(self):
        c = TagCanonicalizer()
        c.register_alias("db", "database")
        assert c.canonicalize("db") == "database"
        assert c.canonicalize("DB") == "database"

    def test_canonicalize_list(self):
        c = TagCanonicalizer()
        c.register_alias("db", "database")
        result = c.canonicalize_list(["db", "database", "api", "API"])
        assert result == ["database", "api"]

    def test_canonicalize_list_preserves_order(self):
        c = TagCanonicalizer()
        result = c.canonicalize_list(["api", "database", "frontend"])
        assert result == ["api", "database", "frontend"]

    def test_get_aliases(self):
        c = TagCanonicalizer()
        c.register_alias("db", "database")
        c.register_alias("fe", "frontend")
        aliases = c.get_aliases()
        assert aliases == {"db": "database", "fe": "frontend"}

    def test_auto_detect_aliases_no_store(self):
        c = TagCanonicalizer()
        assert c.auto_detect_aliases() == []

    def test_load_from_store(self):
        class MockStore:
            def get_tag_aliases(self):
                return {"db": "database"}
        c = TagCanonicalizer(store=MockStore())
        c.load()
        assert c.canonicalize("db") == "database"

    def test_register_with_store(self):
        stored = {}
        class MockStore:
            def get_tag_aliases(self):
                return {}
            def set_tag_alias(self, alias, canonical):
                stored[alias] = canonical
        c = TagCanonicalizer(store=MockStore())
        c.register_alias("db", "database")
        assert stored["db"] == "database"

    def test_hyphen_normalization(self):
        c = TagCanonicalizer()
        assert c.canonicalize("data--layer") == "data-layer"
        assert c.canonicalize("-api-") == "api"

    def test_plural_folding(self):
        c = TagCanonicalizer()
        # First occurrence establishes the singular
        assert c.canonicalize("filter") == "filter"
        # Plural folds to the known singular
        assert c.canonicalize("filters") == "filter"

    def test_plural_folding_plural_first(self):
        c = TagCanonicalizer()
        # If plural is seen first, it becomes the canonical form
        assert c.canonicalize("recipes") == "recipes"
        # Singular folds to the known plural
        assert c.canonicalize("recipe") == "recipes"

    def test_plural_folding_short_words(self):
        c = TagCanonicalizer()
        # Two-letter words ending in 's' should not be folded
        assert c.canonicalize("os") == "os"
